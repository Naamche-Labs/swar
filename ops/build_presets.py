"""Preset voices for the cloner Space (space_clone/app.py): for each voice in data/voices.json, its chosen prompt plus
a few more clean, typical pieces of the same speaker go through the normal enrolment (autoclone.enroll), so a preset
is just a ready-made voice folder (say() needs more than one piece to compare a take against). Output:
OUT/<name>/ (voice.json + pieces) and OUT/voices.json, uploaded to MODEL_REPO/presets/ at deploy time.
Third-party audio (used with permission): goes only to the private model repo, never to git.
Usage (GPU box): KRITI_ONNX=/home/kriti_onnx python ops/build_presets.py OUT"""
import glob
import json
import os
import sys

import numpy as np

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402

SURVEY = {  # voice name -> (survey dir, clusters that are this person), as ops/build_podvoices.py
    "maya": ("/home/data/pod3/survey", ["v00"]), "nisha": ("/home/data/pod4/survey", ["v00"]),
    "sabina": ("/home/data/pod4/survey", ["v01"]),
    "aarav": ("/home/data/pod2/survey", ["v00", "v01", "v02", "v06", "v07", "v08", "v09", "v10", "v11"]),
    "bibek": ("/home/data/pod2/survey", ["v00", "v01", "v02", "v06", "v07", "v08", "v09", "v10", "v11"]),
    "kiran": ("/home/data/pod2/survey", ["v03"]), "sujan": ("/home/data/pod2/survey", ["v05"]),
}
EXTRA = 4  # extra pieces per voice besides the prompt

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
reg = json.load(open("data/voices.json"))["voices"]
meta = {}
for v in reg:
    name = v["name"]
    wavs = [v["prompt"]]
    if name in SURVEY:
        d, clusters = SURVEY[name]
        sv = {x["name"]: x for x in json.load(open(f"{d}/survey.json"))["voices"]}
        P = json.load(open(f"{d}/pieces.json"))
        E = np.load(f"{d}/emb.npy")
        byw = {p["wav"].split("/")[-1]: i for i, p in enumerate(P)}
        labs = {P[byw[sv[c]["samples"][0]]]["voice"] for c in clusters}
        idx = [i for i, p in enumerate(P) if p["voice"] in labs and 5 <= p["speech"] <= 10]
        c = E[idx].mean(0)
        c /= np.linalg.norm(c)
        idx.sort(key=lambda i: -(float(E[i] @ c) + 0.1 * P[i]["nat"]))
        wavs += [P[i]["wav"] for i in idx[:EXTRA]]
    else:  # asha: Rasa studio speaker, conversational clips
        wavs += sorted(glob.glob("/home/data/rasa_test/real/NEP_F_CONV_*.wav"))[:EXTRA * 2]
    A.enroll(f"{out}/{name}", wavs, one_speaker=True)
    meta[name] = {"gender": v["gender"], "style": v["style"]}
    print("PRESET", name, len(wavs), flush=True)
json.dump(meta, open(f"{out}/voices.json", "w"), indent=1)
print("PRESETS_DONE")
