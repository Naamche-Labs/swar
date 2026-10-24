"""Clone a studio voice and compare the clone with the real recording of the SAME sentence.

Rasa test sentences (never trained on) were recorded by the two studio voices. For each item we have
the real recording; we synthesize the sentence in that voice and measure how far the clone is from it:
  MCD (dB)     mel-cepstral distortion after DTW alignment: timbre, texture, noise (lower = closer)
  F0 corr      correlation of log-F0 along the aligned path: melody (higher = closer)
  F0 RMSE (st) pitch error along the path in semitones
  rate ratio   clone speech duration / real speech duration (1.0 = same pace)
  sim          ECAPA similarity clone vs real recording
  UTMOS        predicted naturalness (real recordings for reference)
Systems:
  real      the recording itself (UTMOS reference only)
  vocos     the real recording through F5's mel + Vocos and back: the best our pipeline could ever sound
  <run>     a synthesis run under /home/data/rasa_test/runs/<run> (tools/eval_rasa.py), or made here
Usage: python tools/match_test.py RUN [RUN ...]   (vocos ceiling computed once and cached)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = "/home/data/rasa_test"
OUT = f"{R}/match"


def mcep(w, sr=16000):
    import librosa
    return librosa.feature.mfcc(y=w, sr=sr, n_mfcc=25, n_fft=1024, hop_length=160)[1:]  # drop c0 (energy)


def logf0(w, sr=16000):
    import librosa
    f0, v, _ = librosa.pyin(w, fmin=60, fmax=500, sr=sr, frame_length=1024, hop_length=160)
    return np.where(v, np.log2(np.nan_to_num(f0, nan=1.0)), np.nan)


def compare(real_w, syn_w):
    import librosa
    a, b = mcep(real_w), mcep(syn_w)
    _, wp = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    wp = wp[::-1]
    diff = a[:, wp[:, 0]] - b[:, wp[:, 1]]
    mcd = float(np.mean((10 / np.log(10)) * np.sqrt(2 * np.sum(diff ** 2, axis=0))) / 10)  # MFCC scale ~10x mcep
    fa, fb = logf0(real_w), logf0(syn_w)
    n = min(len(fa), a.shape[1]), min(len(fb), b.shape[1])
    pa = np.array([fa[i] if i < n[0] else np.nan for i in wp[:, 0]])
    pb = np.array([fb[j] if j < n[1] else np.nan for j in wp[:, 1]])
    ok = ~np.isnan(pa) & ~np.isnan(pb)
    corr = float(np.corrcoef(pa[ok], pb[ok])[0, 1]) if ok.sum() > 20 else float("nan")
    rmse = float(np.sqrt(np.mean((12 * (pa[ok] - pb[ok])) ** 2))) if ok.sum() > 20 else float("nan")
    return mcd, corr, rmse


def speech_dur(w, sr=16000):
    import librosa
    iv = librosa.effects.split(w, top_db=35)
    return (iv[-1][1] - iv[0][0]) / sr if len(iv) else len(w) / sr


def vocos_copy(items):
    import librosa
    import soundfile as sf
    import torch
    from vocos import Vocos
    from f5_tts.model.modules import MelSpec
    d = f"{R}/runs/vocos_copy"
    os.makedirs(d, exist_ok=True)
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    mel = MelSpec(mel_spec_type="vocos").cuda()
    for it in items:
        p = f"{d}/{it['id']}.wav"
        if not os.path.exists(p):
            w, _ = librosa.load(it["real_wav"], sr=24000)
            with torch.no_grad():
                y = voc.decode(mel(torch.tensor(w).unsqueeze(0).cuda())).squeeze().cpu().numpy()
            sf.write(p, y, 24000)


def main(runs):
    import librosa
    import torch
    from mos import utmos
    from speechbrain.inference.speaker import EncoderClassifier
    items = json.load(open(f"{R}/items.json"))
    vocos_copy(items)
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"},
                                         savedir="/home/hf_cache/ecapa")

    def emb(w):
        with torch.no_grad():
            return spk.encode_batch(torch.tensor(w).unsqueeze(0).cuda()).squeeze()
    os.makedirs(OUT, exist_ok=True)
    for run in ["real", "vocos_copy"] + runs:
        rows = []
        for it in items:
            if it["style"] == "PROPER NOUN":  # 1-3 word names: too short for melody / rhythm
                continue
            real, _ = librosa.load(it["real_wav"], sr=16000)
            p = it["real_wav"] if run == "real" else f"{R}/runs/{run}/{it['id']}.wav"
            if not os.path.exists(p):
                continue
            syn, _ = librosa.load(p, sr=16000)
            mcd, corr, rmse = compare(real, syn) if run != "real" else (0.0, 1.0, 0.0)
            rows.append({"id": it["id"], "voice": it["voice"], "style": it["style"], "mcd": mcd, "f0_corr": corr,
                         "f0_rmse": rmse, "rate": speech_dur(syn) / max(0.3, speech_dur(real)),
                         "sim": float(torch.nn.functional.cosine_similarity(emb(syn), emb(real), dim=0)), "utmos": utmos(syn)})
        res = {"run": run, "n": len(rows), "rows": rows}
        for k in ("mcd", "f0_corr", "f0_rmse", "rate", "sim", "utmos"):
            res[k] = float(np.nanmean([r[k] for r in rows]))
        res["rate_abs_dev"] = float(np.mean([abs(np.log(r["rate"])) for r in rows]))
        for v in ("Female", "Male"):
            vr = [r for r in rows if r["voice"] == v]
            res[v] = {k: float(np.nanmean([r[k] for r in vr])) for k in ("mcd", "f0_corr", "sim", "utmos")}
        json.dump(res, open(f"{OUT}/{run}.json", "w"), indent=1)
        print(f"MATCH {run:22s} MCD {res['mcd']:.2f}  F0corr {res['f0_corr']:.3f}  F0rmse {res['f0_rmse']:.2f}st  "
              f"pace off {100 * (np.exp(res['rate_abs_dev']) - 1):.0f}%  sim {res['sim']:.3f}  UTMOS {res['utmos']:.2f}  "
              f"| F {res['Female']['mcd']:.2f}/{res['Female']['utmos']:.2f}  M {res['Male']['mcd']:.2f}/{res['Male']['utmos']:.2f}  (n={res['n']})",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
