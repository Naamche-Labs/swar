"""End-to-end with the bilingual checkpoint: add English detection to an enrolled voice, re-select prompts, and say
Nepali, English (from the speaker's own English) and mixed lines with the automatic take choice.
Usage (GPU box): F5_CKPT=/home/ema/nepali_bi_ema.pt python ops/bi_voice_demo.py VOICE_DIR OUT_DIR"""
import json
import os
import sys

import numpy as np

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402

vd, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
v = json.load(open(f"{vd}/voice.json"))
P = [p for p in v["pieces"] if os.path.exists(p["wav"])]
for p in P:
    p["emb"] = np.array(p["emb"])
    if p["main"] and 4.0 <= p["dur"] <= 11.0 and "p_en" not in p:
        p["p_en"], p["en_text"] = A.whisper_lid_en(p["wav"])
v = A.select(vd, P, v["uploads"])
print("english prompts:", [(os.path.basename(p["wav"]), p["text"][:50]) for p in v["en_prompts"]], flush=True)
T = {"np_story": "हिजो बेलुका हामी साथीहरू मिलेर खाना खान गयौँ, एकदमै रमाइलो भयो।",
     "np_today": "आजको खाना एकदमै मीठो थियो, आज बिहान म अलि ढिलो उठेँ।",
     "en_thanks": "Thank you so much for your support. See you tomorrow.",
     "en_show": "Hello everyone, welcome to the show. Today we are going to talk about technology in Nepal.",
     "en_story": "I was really nervous at first, but honestly, it turned out to be the best day of my life.",
     "mix_office": "आज office मा meeting धेरै लामो भयो, अब घर गएर rest गर्नुपर्छ।",
     "mix_order": "Customer service मा call गर्दा उनीहरूले भने, your order will be delivered by Friday.",
     "chat_mother": "साँच्चै भन्नुपर्दा, motherhood भनेको सजिलो कुरा होइन, तर यसले हामीलाई धेरै बलियो बनाउँछ।"}
res = {}
for k, t in T.items():
    b = A.say(vd, t, f"{out}/{k}.wav", takes=8)
    res[k] = {x: b[x] for x in ("sim", "cer", "nat", "heard", "prompt", "model")}
    os.system(f"ffmpeg -loglevel error -y -i {out}/{k}.wav -b:a 128k {out}/{k}.mp3")
json.dump({"texts": T, "res": res}, open(f"{out}/demo.json", "w"), ensure_ascii=False, indent=1)
print("BI_VOICE_DONE")
