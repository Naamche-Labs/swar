"""How exactly does the clone match the person? No per-user training anywhere.

Speakers: the 4 FLEURS Nepal speakers (tools/eval_fleurs.py) + optionally Aarjan.
Per FLEURS speaker, 5 of their recordings are joined into an "upload" (~50 s, like a
user's recording) and up to 8 other sentences are cloned and compared with the
speaker's real recording of that sentence.

Variants (all from the same general model):
  auto     prepare_voice: most typical piece as prompt + the upload's pace
  best3    auto, 3 seeds, keep the take closest (ECAPA) to the upload's mean voice
  long     two most typical pieces joined (~20 s) as the prompt
  knn      auto, then kNN-VC (bshall/knn-vc, MIT) with the whole upload as matching set
Metrics: Devanagari MMS CER, ECAPA similarity to the real recording, pitch level
difference (median F0, semitones), pitch movement ratio (F0 std clone / real), UTMOS.

Usage: python tools/voice_match.py CKPT OUT_DIR [VARIANTS]
"""
import json
import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from eval_heldout import DEV, IVF5_VOCAB, ed, norm  # noqa: E402

F = "/home/data/fleurs"


def f0_stats(path):
    import librosa
    import numpy as np
    w, _ = librosa.load(path, sr=16000)
    f0, voiced, _ = librosa.pyin(w, fmin=60, fmax=450, sr=16000, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    if len(f0) < 10:
        return None, None
    st = 12 * np.log2(f0 / 100.0)
    return float(np.median(st)), float(np.std(st))


def speakers():
    import soundfile as sf
    import numpy as np
    items = json.load(open(f"{F}/items.json"))
    by = {}
    for it in items:
        by.setdefault(it["speaker"], []).append(it)
    out = []
    for spk, its in sorted(by.items()):
        if len(its) < 8:
            continue
        up, tr = its[:5], its[5:13]
        path = f"{F}/upload_{spk}.wav"
        if not os.path.exists(path):
            parts = []
            for it in up:
                w, sr = sf.read(it["real_wav"], dtype="float32")
                parts += [w, np.zeros(int(0.6 * sr), dtype="float32")]
            sf.write(path, np.concatenate(parts), sr)
        out.append({"name": spk, "upload": path, "transcripts": [it["text"] for it in up],
                    "targets": [(it["id"], it["text"], it["real_wav"]) for it in tr]})
    return out


def main(ckpt, out_dir, variants):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from f5_tts.api import F5TTS
    from clone import patch_f5_chunker, prepare_voice, set_pace, speed_for
    from frontend import normalize
    from mos import utmos
    from rescore_deva import DevaMMS
    from speechbrain.inference.speaker import EncoderClassifier
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV},
                                         savedir="/home/hf_cache/ecapa")

    def emb(p):
        w, _ = librosa.load(p, sr=16000)
        with torch.no_grad():
            return torch.nn.functional.normalize(spk.encode_batch(torch.tensor(w).unsqueeze(0).to(DEV)).squeeze(), dim=-1)

    patch_f5_chunker()
    tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt, vocab_file=IVF5_VOCAB)
    tts.ema_model.float()
    knn = None
    if "knn" in variants:
        knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=DEV)
    rows = []
    for s in speakers():
        d = f"{out_dir}/{s['name']}"
        os.makedirs(d, exist_ok=True)
        v = prepare_voice(s["upload"], d + "/voice", transcripts=s["transcripts"])
        centroid = torch.nn.functional.normalize(torch.stack([emb(p["wav"]) for p in v["pieces"]]).mean(0), dim=-1)
        top2 = sorted(v["pieces"], key=lambda p: -p["typical"])[:2]
        lw = np.concatenate([librosa.load(p["wav"], sr=24000)[0] for p in top2])
        long_ref, long_text = f"{d}/voice/long_prompt.wav", " ".join(p["text"] for p in top2)
        sf.write(long_ref, lw, 24000)
        mset = knn.get_matching_set([p["wav"] for p in v["pieces"]]) if knn else None
        for tid, text, real in s["targets"]:
            g = normalize(text)

            def gen(ref, rt, seed, path):
                set_pace(normalize(rt), ref)
                tts.infer(ref_file=ref, ref_text=normalize(rt), gen_text=g, file_wave=path, remove_silence=True,
                          seed=seed, speed=speed_for(g, v["speed"]))
                return path
            outs = {}
            if {"auto", "best3", "knn"} & set(variants):
                outs["auto"] = gen(v["ref"], v["ref_text"], 0, f"{d}/{tid}_auto.wav")
            if "best3" in variants:
                takes = [outs["auto"]] + [gen(v["ref"], v["ref_text"], sd, f"{d}/{tid}_seed{sd}.wav") for sd in (1, 2)]
                outs["best3"] = max(takes, key=lambda p: float(emb(p) @ centroid))
            if "long" in variants:
                outs["long"] = gen(long_ref, long_text, 0, f"{d}/{tid}_long.wav")
            if knn:
                q = knn.get_features(outs["auto"])
                y = knn.match(q, mset, topk=4).cpu().numpy()
                sf.write(f"{d}/{tid}_knn.wav", y, 16000)
                outs["knn"] = f"{d}/{tid}_knn.wav"
            rows.append({"speaker": s["name"], "id": tid, "text": g, "real": real, "outs": outs})
    del tts
    torch.cuda.empty_cache()
    asr = DevaMMS()
    res = {}
    for r in rows:
        rf0 = f0_stats(r["real"])
        er = emb(r["real"])
        for k, p in r["outs"].items():
            if k not in variants:
                continue
            cf0 = f0_stats(p)
            w, _ = librosa.load(p, sr=16000)
            m = res.setdefault(k, {"cer": [], "sim": [], "pitch_st": [], "move": [], "mos": []})
            t = norm(r["text"])
            m["cer"].append(ed(t, norm(asr(p))) / max(1, len(t)))
            m["sim"].append(float(emb(p) @ er))
            if rf0[0] is not None and cf0[0] is not None:
                m["pitch_st"].append(abs(cf0[0] - rf0[0]))
                m["move"].append(cf0[1] / max(1e-3, rf0[1]))
            m["mos"].append(utmos(w))
    summary = {k: {"cer": 100 * np.mean(m["cer"]), "sim": np.mean(m["sim"]), "pitch_diff_semitones": np.mean(m["pitch_st"]),
                   "pitch_movement_ratio": np.mean(m["move"]), "utmos": np.mean(m["mos"]), "n": len(m["cer"])}
               for k, m in res.items()}
    for k, m in summary.items():
        print(f"VM {k:6s} CER {m['cer']:5.1f}  sim {m['sim']:.3f}  pitch diff {m['pitch_diff_semitones']:.2f} st  "
              f"movement {m['pitch_movement_ratio']:.2f}  UTMOS {m['utmos']:.2f}  (n={m['n']})", flush=True)
    json.dump({"summary": summary, "rows": rows}, open(f"{out_dir}/voice_match.json", "w"), ensure_ascii=False, indent=1,
              default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "auto,best3,long,knn").split(","))
