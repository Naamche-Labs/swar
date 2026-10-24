"""Punctuation stage training set: IndicVoices-R with punctuation restored from audio
(tools/restore_punct.py: /home/punct/out_*.jsonl), Rasa as recorded (punctuated), OpenSLR 54/43 with a
sentence-final "।" added where missing (short read sentences). Same vocab filter / quote strip as v2."""
import glob
import json
import re

V = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))
rows, bad = [], 0


def add(audio, text):
    global bad
    t = text.replace('"', "").replace("|", " ").strip()
    if not t or any(c not in V for c in t):
        bad += 1
        return
    rows.append(audio + "|" + t)


for f in sorted(glob.glob("/home/punct/out_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        x = json.loads(line)
        add(x["audio"], x["text"])
for line in open("/home/data/nepali_tts/manifest.train.jsonl", encoding="utf-8"):
    x = json.loads(line)
    if x["speaker"].startswith("rasa"):
        add(x["audio"], x["text"])
for line in open("/home/data/nepali_tts/manifest_slr.jsonl", encoding="utf-8"):
    x = json.loads(line)
    t = x["text"].strip()
    add(x["audio"], t if re.search(r"[।?!]$", t) else t + "।")
open("/home/punct/train.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(rows) + "\n")
print(len(rows), "rows,", bad, "dropped")
