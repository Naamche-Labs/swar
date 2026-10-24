"""English prompts in each podcast voice (2026-10-23). User: full English still bad ("everyone" said "eveeonee").
Nepali-letter English cannot carry English sounds (एभरीवन), and English text with a Nepali prompt is garbage on this
model; English text with the SAME speaker's English prompt is the path that worked before (Speaker S/Speaker B: WER 0 after
best-of-8, accent close to his real English). These speakers code-switch a lot, so their own English is in the
podcasts: Whisper language ID finds pieces that are English, its transcript is the prompt text.
Usage (GPU box): python ops/en_prompts.py OUT"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402

VOICES = {  # as ops/build_podvoices.py (survey dir, clusters that are this person)
    "host": ("/home/data/pod2/survey", ["v00", "v01", "v02", "v06", "v07", "v08", "v09", "v10", "v11"]),
    "guest1": ("/home/data/pod2/survey", ["v03"]), "guest2": ("/home/data/pod2/survey", ["v05"]),
    "thrA": ("/home/data/pod3/survey", ["v00"]), "p4a": ("/home/data/pod4/survey", ["v00"]),
    "p4b": ("/home/data/pod4/survey", ["v01"]),
}

NAME = {"maya": "thrA", "nisha": "p4a", "sabina": "p4b", "aarav": "host", "kiran": "guest1", "sujan": "guest2"}
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
found = {}
for name, v in NAME.items():
    d, clusters = VOICES[v]
    sv = {x["name"]: x for x in json.load(open(f"{d}/survey.json"))["voices"]}
    P = json.load(open(f"{d}/pieces.json"))
    byw = {p["wav"].split("/")[-1]: i for i, p in enumerate(P)}
    labs = {P[byw[sv[c]["samples"][0]]]["voice"] for c in clusters}
    cs = sorted([p for p in P if p["voice"] in labs and 4 <= p["speech"] <= 10], key=lambda p: -p["nat"])[:120]
    en = []
    for p in cs:
        p_en, text = A.whisper_lid_en(p["wav"])
        if p_en > 0.85 and len(text.split()) >= 6:
            en.append({**p, "p_en": p_en, "text": text})
        if len(en) >= 3:
            break
    found[name] = []
    for k, p in enumerate(en[:2]):
        x, sr = sf.read(p["wav"])
        sf.write(f"{out}/{name}{k}.wav", x, sr)
        open(f"{out}/{name}{k}.txt", "w").write(p["text"])
        found[name].append({"file": f"{name}{k}", "p_en": round(p["p_en"], 3), "nat": round(p["nat"], 2), "text": p["text"]})
    print("EN_PROMPTS", name, len(en), found[name], flush=True)
json.dump(found, open(f"{out}/en_prompts.json", "w"), ensure_ascii=False, indent=1)
print("EN_PROMPTS_DONE")
