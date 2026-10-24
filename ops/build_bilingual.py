"""Bilingual training set: real English speech so pure-English lines stop sounding like Nepali-letter English (user:
"tomorrow" came out "टुम्रो", "the English has to be very, very good"). English = LibriTTS-R train-clean-100 (CC-BY-4.0,
~54 h, 247 speakers, Latin text as written; IndicF5's vocab has all ASCII). Nepali = the Nepal-recorded part of our
set (Rasa studio + OpenSLR, ~110 h) so the accent stage is not undone. Writes /home/bi/bi.csv for prepare_csv_wavs.
Usage (box 2): python ops/build_bilingual.py"""
import glob
import io
import os

import pyarrow.parquet as pq
import soundfile as sf

V = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))
os.makedirs("/home/data/libritts_r/wav", exist_ok=True)
rows, en_sec = [], 0.0
for f in sorted(glob.glob("/home/data/libritts_r/data/train.clean.100/*.parquet")):
    for r in pq.read_table(f, columns=["audio", "text_normalized", "id"]).to_pylist():
        t = r["text_normalized"].replace("|", " ").replace('"', "").strip()
        if not t or not all(c in V for c in t):
            continue
        w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        if not 1.0 <= len(w) / sr <= 20.0:
            continue
        p = f"/home/data/libritts_r/wav/{r['id']}.wav"
        if not os.path.exists(p):
            sf.write(p, w, sr)
        rows.append(f"{p}|{t}")
        en_sec += len(w) / sr
n_en = len(rows)
for line in open("/home/data/nepali_tts/train_chars.csv", encoding="utf-8").read().splitlines()[1:]:
    if "/rasa/" in line or "/openslr" in line:
        rows.append(line)
os.makedirs("/home/bi", exist_ok=True)
open("/home/bi/bi.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(rows) + "\n")
print(f"BI_BUILT english {n_en} clips {en_sec / 3600:.1f} h, nepali {len(rows) - n_en} clips")
