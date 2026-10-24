"""Nepal-recorded speech for the accent stage (CC-BY-SA 4.0, user OK'd share-alike).

OpenSLR 54: 157k crowdsourced read utterances, 16 kHz phone/laptop mics, speakers in Nepal.
OpenSLR 43: 2,064 studio TTS utterances (Google, 48 kHz), several female speakers in Nepal.
Kept: 2-15 s after trimming, SNR >= 20 dB (SLR54) / any (SLR43), peak-normalized 24 kHz,
text normalized with our frontend, not an eval sentence. Writes manifest_slr.jsonl. CPU only.
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, "/home/nepali-tts")
sys.path.insert(0, "/home/nepali-tts/pipeline")
import prepare_data as P  # noqa: E402
from frontend import normalize  # noqa: E402

S = "/home/data/slr"
OUT = "/home/data/nepali_tts/wav/slr"
MIN_SNR = {"slr54": 20.0, "slr43": 0.0}


def job(a):
    src, path, text, spk = a
    import librosa
    import numpy as np
    import soundfile as sf
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if w.ndim > 1:
        w = w.mean(1)
    w = librosa.resample(w, orig_sr=sr, target_sr=P.TARGET_SR)
    w, _ = librosa.effects.trim(w, top_db=30)
    dur = len(w) / P.TARGET_SR
    if not 2.0 <= dur <= 15.0:
        return None
    snr = P.estimate_snr_db(w, P.TARGET_SR)
    if snr < MIN_SNR[src]:
        return None
    w = w * (10 ** (-1 / 20) / max(1e-6, float(np.abs(w).max())))
    out = f"{OUT}/{src}_{os.path.basename(path).rsplit('.', 1)[0]}.wav"
    sf.write(out, w, P.TARGET_SR)
    return {"audio": out, "text": normalize(text), "speaker": f"{src}:{spk}", "dur": round(dur, 3), "snr": round(snr, 1),
            "source": src}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    evals = set()
    for f in ["/home/data/eval/items.json", "/home/data/rasa_test/items.json"]:
        if os.path.exists(f):
            evals |= {it["text"] for it in json.load(open(f))}
    jobs = []
    for line in open(f"{S}/utt_spk_text.tsv", encoding="utf-8"):
        u, spk, text = line.rstrip("\n").split("\t")
        jobs.append(("slr54", f"{S}/asr_nepali/data/{u[:2]}/{u}.flac", text, spk))
    for line in open(f"{S}/ne_np_female/line_index.tsv", encoding="utf-8"):
        u, text = line.rstrip("\n").split("\t")
        jobs.append(("slr43", f"{S}/ne_np_female/wavs/{u}.wav", text, u.split("_")[1]))
    n = 0
    with Pool(26) as pool, open("/home/data/nepali_tts/manifest_slr.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=64)):
            if r and r["text"] not in evals:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            if i % 20000 == 0:
                print(i, "done,", n, "kept", flush=True)
    print("KEPT", n, "of", len(jobs))
