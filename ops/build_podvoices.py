"""Training set for the podcast voices (podcast A, podcast B and four interview shows, used with
permission, 2026-10-23): each chosen voice's pieces from ops/podcast_survey.py, kept when they are clean and
typical of the voice, transcribed by Kriti (ONNX, in-process) with its confidence, 10 per voice held out for the
eval. Replay so the model keeps zero-shot cloning and English: FLEURS ne_np (many speakers, CC-BY 4.0) and LibriTTS-R
dev.clean. Output: OUT/train.csv (audio_file|text, for prepare_csv_wavs.py) and OUT/heldout.json.
Usage (GPU box): KRITI_ONNX=/home/kriti_onnx python ops/build_podvoices.py OUT"""
import glob
import io
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402
from frontend import normalize  # noqa: E402

# voice: (survey dir, survey cluster names that are this person)
VOICES = {
    "host": ("/home/data/pod2/survey", ["v00", "v01", "v02", "v06", "v07", "v08", "v09", "v10", "v11"]),
    "guest1": ("/home/data/pod2/survey", ["v03"]),
    "guest2": ("/home/data/pod2/survey", ["v05"]),
    "podAf": ("/home/data/pod2/survey", ["v04"]),
    "thrA": ("/home/data/pod3/survey", ["v00"]),
    "thrB": ("/home/data/pod3/survey", ["v01"]),
    "p4a": ("/home/data/pod4/survey", ["v00"]),
    "p4b": ("/home/data/pod4/survey", ["v01"]),
    "p4c": ("/home/data/pod4/survey", ["v02"]),
    "p4d": ("/home/data/pod4/survey", ["v03"]),
}
CAP_MIN = 45          # most minutes of speech per voice (the host has 180)
# UTMOS is low on all casual podcast talk (podcast A clusters average 1.5-2.1; DeepFilterNet barely moves it), so the
# cleanliness cut is relative: drop each voice's worst 30%, never keep below 1.6
NAT_PCT, NAT_FLOOR, MIN_CONF, MIN_TYP = 30, 1.6, 0.80, 0.55
FLEURS_N, EN_N = 1500, 1500
VOCAB = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))

out = sys.argv[1]
os.makedirs(f"{out}/wav", exist_ok=True)
from kriti_onnx import KritiONNX  # noqa: E402
kr = KritiONNX(os.environ["KRITI_ONNX"])
rows, held, stats = [], {}, {}
for name, (d, clusters) in VOICES.items():
    sv = {v["name"]: v for v in json.load(open(f"{d}/survey.json"))["voices"]}
    P = json.load(open(f"{d}/pieces.json"))
    E = np.load(f"{d}/emb.npy")
    byw = {p["wav"].split("/")[-1]: i for i, p in enumerate(P)}
    labs = {P[byw[sv[c]["samples"][0]]]["voice"] for c in clusters}
    idx = [i for i, p in enumerate(P) if p["voice"] in labs and 3.5 <= p["speech"] <= 10]
    cut = max(NAT_FLOOR, float(np.percentile([P[i]["nat"] for i in idx], NAT_PCT)))
    idx = [i for i in idx if P[i]["nat"] >= cut]
    c = E[idx].mean(0)
    c /= np.linalg.norm(c)
    idx = [i for i in idx if float(E[i] @ c) >= MIN_TYP]
    res = kr.read_conf([P[i]["wav"] for i in idx])
    keep = []
    for i, (t, conf) in zip(idx, res):
        t = normalize(t, english=False)
        rate = len(t.replace(" ", "")) / max(1.0, P[i]["speech"])
        if conf >= MIN_CONF and len(t) >= 8 and 6 <= rate <= 22 and all(ch in VOCAB for ch in t):
            keep.append({"wav": P[i]["wav"], "text": t, "conf": conf, "nat": P[i]["nat"], "speech": P[i]["speech"]})
    keep.sort(key=lambda r: -r["conf"] - 0.1 * r["nat"])
    held[name] = keep[:10]
    train = keep[10:]
    random.Random(0).shuffle(train)
    tot, use = 0.0, []
    for r in train:
        if tot + r["speech"] > CAP_MIN * 60:
            break
        use.append(r)
        tot += r["speech"]
    rows += [f"{r['wav']}|{r['text']}" for r in use]
    stats[name] = {"candidates": len(idx), "kept": len(keep), "train": len(use), "train_min": round(tot / 60, 1)}
    print(name, stats[name], flush=True)

# replay: FLEURS ne_np (many Nepali speakers) and LibriTTS-R English
import pyarrow.parquet as pq  # noqa: E402
import librosa  # noqa: E402
fl = glob.glob("/home/hf_cache/hub/datasets--google--fleurs/snapshots/*/parquet-data/ne_np/train-*.parquet")[0]
t = pq.read_table(fl, columns=["id", "audio", "transcription", "num_samples"]).to_pylist()
random.Random(1).shuffle(t)
n = 0
for r in t:
    if n >= FLEURS_N or not 3 * 16000 <= r["num_samples"] <= 15 * 16000:
        continue
    txt = normalize(r["transcription"], english=False)
    if not all(ch in VOCAB for ch in txt):
        continue
    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
    p = f"{out}/wav/fleurs_{r['id']}_{n}.wav"
    sf.write(p, librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=24000), 24000)
    rows.append(f"{p}|{txt}")
    n += 1
stats["fleurs"] = n
en = sorted(glob.glob("/home/hf_cache/hub/datasets--mythicinfinity--libritts_r/snapshots/*/data/dev.clean/*.parquet"))
n = 0
for f in en:
    for r in pq.read_table(f, columns=["audio", "text_normalized", "id"]).to_pylist():
        if n >= EN_N:
            break
        txt = r["text_normalized"].strip()
        if not 10 <= len(txt) <= 200 or not all(ch in VOCAB for ch in txt):
            continue
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        if not 2 <= len(x) / sr <= 15:
            continue
        p = f"{out}/wav/en_{r['id']}.wav"
        sf.write(p, librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=24000), 24000)
        rows.append(f"{p}|{txt}")
        n += 1
stats["english"] = n
random.Random(2).shuffle(rows)
open(f"{out}/train.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(rows) + "\n")
json.dump(held, open(f"{out}/heldout.json", "w"), ensure_ascii=False, indent=1)
json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)
print("ROWS", len(rows), stats, "BUILD_DONE", flush=True)
