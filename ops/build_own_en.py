"""English in the voices' own accent (2026-10-23). User: English from the speaker's English prompt with Latin text is
worse ("can't understand"), the older Nepali-letter English was better; "maybe train English ... or a bit
Nepali-English combined". Here each podcast voice's own English pieces (Whisper language ID > 0.8, Whisper
transcript) are written the way the frontend writes English at inference (Nepali letters) and added to the
podcast-voices set, so the model learns how these Nepali speakers sound saying that spelling.
Usage (GPU box): python ops/build_own_en.py PODVOICES_DIR OUT"""
import json
import os
import random
import sys

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402
from clone import to_model_text  # noqa: E402

VOICES = {  # as ops/build_podvoices.py
    "host": ("/home/data/pod2/survey", ["v00", "v01", "v02", "v06", "v07", "v08", "v09", "v10", "v11"]),
    "guest1": ("/home/data/pod2/survey", ["v03"]), "guest2": ("/home/data/pod2/survey", ["v05"]),
    "podAf": ("/home/data/pod2/survey", ["v04"]), "thrA": ("/home/data/pod3/survey", ["v00"]),
    "thrB": ("/home/data/pod3/survey", ["v01"]), "p4a": ("/home/data/pod4/survey", ["v00"]),
    "p4b": ("/home/data/pod4/survey", ["v01"]), "p4c": ("/home/data/pod4/survey", ["v02"]),
    "p4d": ("/home/data/pod4/survey", ["v03"]),
}
PER_VOICE = 400  # candidates per voice (cleanest first)
VOCAB = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))
pod, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
held = {h["wav"] for hs in json.load(open(f"{pod}/heldout.json")).values() for h in hs}
rows, stats = [], {}
for v, (d, clusters) in VOICES.items():
    sv = {x["name"]: x for x in json.load(open(f"{d}/survey.json"))["voices"]}
    P = json.load(open(f"{d}/pieces.json"))
    byw = {p["wav"].split("/")[-1]: i for i, p in enumerate(P)}
    labs = {P[byw[sv[c]["samples"][0]]]["voice"] for c in clusters}
    cs = sorted([p for p in P if p["voice"] in labs and 3.5 <= p["speech"] <= 10 and p["wav"] not in held],
                key=lambda p: -p["nat"])[:PER_VOICE]
    n = 0
    for p in cs:
        p_en, text = A.whisper_lid_en(p["wav"])
        if p_en <= 0.8 or len(text.split()) < 4:
            continue
        try:
            dv = to_model_text(text, "chars")
        except Exception:
            continue
        if any("A" <= c <= "z" for c in dv) or not all(ch in VOCAB for ch in dv):
            continue
        rows.append(f"{p['wav']}|{dv}")
        n += 1
    stats[v] = n
    print("OWN_EN", v, "candidates", len(cs), "english", n, flush=True)
base = open(f"{pod}/train.csv", encoding="utf-8").read().splitlines()[1:]
rep = int(os.environ.get("REPEAT", "2"))  # the English rows are few: seen twice per epoch
allrows = base + rows * rep
random.Random(4).shuffle(allrows)
open(f"{out}/train.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(allrows) + "\n")
json.dump(stats, open(f"{out}/own_en_stats.json", "w"), indent=1)
print("OWN_EN_ROWS", len(rows), "BASE", len(base), "TOTAL", len(allrows), "BUILD_DONE", flush=True)
