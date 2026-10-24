"""English stage set (2026-10-23). User: full English works best written in Nepali letters (our frontend's English
layer) but still sounds broken: "technology", "tomorrow", accent. The model has heard English only with Latin text,
so Nepali-letter English is read with Nepali sounds. Here real English speech (LibriTTS-R dev.clean) is paired with
the SAME Devanagari spelling the frontend writes at inference (frontend.normalize -> english_to_devanagari), so the
model learns to say that spelling the English way. Mixed with the podcast-voices set so the voices and Nepali stay.
Usage (GPU box): python ops/build_en_deva.py PODVOICES_DIR OUT [N]"""
import glob
import io
import os
import random
import sys

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path[:0] = ["pipeline", "."]
from clone import to_model_text  # noqa: E402

pod, out = sys.argv[1:3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
VOCAB = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))
os.makedirs(f"{out}/wav", exist_ok=True)
rows, n = [], 0
for f in sorted(glob.glob("/home/hf_cache/hub/datasets--mythicinfinity--libritts_r/snapshots/*/data/dev.clean/*.parquet")):
    for r in pq.read_table(f, columns=["audio", "text_normalized", "id"]).to_pylist():
        if n >= N:
            break
        txt = r["text_normalized"].strip()
        if not 10 <= len(txt) <= 200:
            continue
        try:
            dv = to_model_text(txt, "chars")  # exactly what the model will be given for English at inference
        except Exception as e:
            print("SKIP", txt[:40], repr(e), flush=True)
            continue
        if not dv or any("A" <= c <= "z" for c in dv) or not all(ch in VOCAB for ch in dv):
            continue
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        if not 2 <= len(x) / sr <= 15:
            continue
        p = f"{out}/wav/end_{r['id']}.wav"
        sf.write(p, librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=24000), 24000)
        rows.append(f"{p}|{dv}")
        n += 1
        if n <= 3:
            print("EXAMPLE", txt, "->", dv, flush=True)
base = open(f"{pod}/train.csv", encoding="utf-8").read().splitlines()[1:]
allrows = base + rows
random.Random(3).shuffle(allrows)
open(f"{out}/train.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(allrows) + "\n")
print("EN_DEVA", n, "BASE", len(base), "TOTAL", len(allrows), "BUILD_DONE", flush=True)
