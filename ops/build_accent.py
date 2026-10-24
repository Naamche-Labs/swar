"""Accent stage training set: Nepal-recorded speech (manifest_slr.jsonl, OpenSLR 54/43)
+ Rasa studio clips from the train manifest (keeps output quality). Drops rows with
characters outside the IndicF5 vocab and strips quotes (CSV quote bug, see v2)."""
import json

V = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))
rows, bad = [], 0
srcs = [("/home/data/nepali_tts/manifest_slr.jsonl", lambda x: True),
        ("/home/data/nepali_tts/manifest.train.jsonl", lambda x: x["speaker"].startswith("rasa"))]
for f, keep in srcs:
    for line in open(f, encoding="utf-8"):
        x = json.loads(line)
        if not keep(x):
            continue
        t = x["text"].replace('"', "").replace("|", " ").strip()
        if not t or any(c not in V for c in t):
            bad += 1
            continue
        rows.append(x["audio"] + "|" + t)
open("/home/accent/train.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(rows) + "\n")
print(len(rows), "rows,", bad, "dropped for chars outside vocab")
