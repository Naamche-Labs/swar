"""Rasa test split: unseen sentences read by the two studio voices, with the human
recording as a true ceiling (NepTTS-Bench has no human audio).

200 items: 20 per (voice, style) for PROPER NOUN, WIKI, NEWS, BOOK, CONV, 1-15 s,
texts not in our training manifest. Prompt per voice = the calm bench prompt
(female_news / male_wiki). Scores: MMS CER (corpus + mean-file), ECAPA similarity
to the real recording, UTMOS, per style.

Usage:
    python tools/eval_rasa.py build            # box with the Rasa HF cache: writes /home/data/rasa_test
    python tools/eval_rasa.py run NAME CKPT    # synthesize + score
    python tools/eval_rasa.py real             # score the human recordings (ceiling)
    python tools/eval_rasa.py table
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_heldout import DEV, IVF5_VOCAB, ed, norm, text_converter  # noqa: E402

R = "/home/data/rasa_test"
STYLES = ["PROPER NOUN", "WIKI", "NEWS", "BOOK", "CONV"]
PROMPT = {"Female": "female_news", "Male": "male_wiki"}


def build(per=20):
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    sys.path.insert(0, "/home/nepali-tts/pipeline")
    sys.path.insert(0, "/home/nepali-tts")
    import prepare_data as P
    from frontend import normalize
    train = {json.loads(l)["text"] for l in open("/home/data/nepali_tts/manifest.jsonl", encoding="utf-8")}
    rows = []
    for f in sorted(glob.glob("/home/hf_cache/hub/datasets--ai4bharat--Rasa/snapshots/*/Nepali/test-*.parquet")):
        rows += pq.read_table(f).to_pylist()
    os.makedirs(f"{R}/real", exist_ok=True)
    items = []
    for g in ("Female", "Male"):
        for st in STYLES:
            c = [r for r in rows if r["gender"] == g and r["style"] == st and 1 <= float(r["duration"]) <= 15]
            c.sort(key=lambda r: hashlib.md5(r["filename"].encode()).hexdigest())
            n = 0
            for r in c:
                text = normalize(r["text"].strip())
                if text in train or n >= per:
                    continue
                arr, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=False)
                pc = P.preprocess_clip(arr, sr) if float(r["duration"]) >= 3 else None
                wav = pc[0] if pc else arr
                if not pc:  # short clips: P.preprocess_clip rejects < 3 s, so resample only
                    import librosa
                    wav = librosa.resample(arr, orig_sr=sr, target_sr=24000) if sr != 24000 else arr
                p = f"{R}/real/{r['filename']}.wav"
                sf.write(p, wav, 24000)
                items.append({"id": r["filename"], "voice": g, "style": st, "text": text, "real_wav": p})
                n += 1
    json.dump(items, open(f"{R}/items.json", "w"), ensure_ascii=False, indent=1)
    print(len(items), "items")


def score(name, wavs):
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from speechbrain.inference.speaker import EncoderClassifier
    from mos import utmos
    items = json.load(open(f"{R}/items.json"))
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                         ignore_mismatched_sizes=True).to(DEV).eval()
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV},
                                         savedir="/home/hf_cache/ecapa")

    def emb(w):
        with torch.no_grad():
            return spk.encode_batch(torch.tensor(w).unsqueeze(0).to(DEV)).squeeze()

    rows = []
    for it in items:
        p = wavs.get(it["id"])
        if not p or not os.path.exists(p):
            continue
        w, _ = librosa.load(p, sr=16000)
        inp = {k: v.to(DEV) for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            hyp = norm(proc.decode(asr(**inp).logits.argmax(-1)[0]))
        ref = norm(it["text"])
        real, _ = librosa.load(it["real_wav"], sr=16000)
        rows.append({"id": it["id"], "voice": it["voice"], "style": it["style"], "e": ed(ref, hyp), "n": len(ref),
                     "sim": float(torch.nn.functional.cosine_similarity(emb(w), emb(real), dim=0)),
                     "mos": utmos(w), "hyp": hyp, "ref": ref})

    def agg(rs):
        return {"n": len(rs), "cer": 100 * sum(r["e"] for r in rs) / max(1, sum(r["n"] for r in rs)),
                "mean_file_cer": 100 * sum(r["e"] / max(1, r["n"]) for r in rs) / max(1, len(rs)),
                "sim": sum(r["sim"] for r in rs) / max(1, len(rs)), "mos": sum(r["mos"] for r in rs) / max(1, len(rs))}
    res = {"name": name, **agg(rows), "by_style": {st: agg([r for r in rows if r["style"] == st]) for st in STYLES},
           "rows": rows}
    os.makedirs(f"{R}/runs", exist_ok=True)
    json.dump(res, open(f"{R}/runs/{name}.json", "w"), ensure_ascii=False, indent=1)
    show(res)
    return res


def show(r):
    st = "  ".join(f"{k[:4]} {v['mean_file_cer']:.1f}" for k, v in r["by_style"].items())
    print(f"{r['name']:28s} CER {r['cer']:5.2f}%  mean-file {r['mean_file_cer']:5.2f}%  sim {r['sim']:.3f}  "
          f"UTMOS {r['mos']:.2f} | {st}", flush=True)


def run(name, ckpt, vocab=IVF5_VOCAB):
    from f5_tts.api import F5TTS
    sys.path.insert(0, "/home/nepali-tts/pipeline")
    from clone import speed_for
    tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt, vocab_file=vocab)
    tts.ema_model.float()
    refs = json.load(open("/home/data/ref/bench/refs.json"))
    conv = text_converter()
    out = f"{R}/runs/{name}"
    os.makedirs(out, exist_ok=True)
    wavs = {}
    for it in json.load(open(f"{R}/items.json")):
        p = f"{out}/{it['id']}.wav"
        if not os.path.exists(p):
            ref = refs[PROMPT[it["voice"]]]
            gen = conv(it["text"])
            # preset pace calibration: refs.json "speed" = the speaker's median pace / this clip's pace
            tts.infer(ref_file=ref["audio"], ref_text=conv(ref["text"]), gen_text=gen, file_wave=p,
                      remove_silence=True, seed=0, speed=speed_for(gen, ref.get("speed", 1.0) if os.environ.get("PRESET_PACE") == "1" else 1.0),
                      nfe_step=int(os.environ.get("NFE", 32)))
        wavs[it["id"]] = p
    del tts
    return score(name, wavs)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "run":
        run(*sys.argv[2:4])
    elif cmd == "real":
        score("real", {it["id"]: it["real_wav"] for it in json.load(open(f"{R}/items.json"))})
    elif cmd == "table":
        for f in sorted(glob.glob(f"{R}/runs/*.json")):
            show(json.load(open(f)))
