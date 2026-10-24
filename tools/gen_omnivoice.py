"""Generate the comparison audio with OmniVoice (k2-fsa/OmniVoice, 0.6B, 600+ languages incl. npi),
in its own venv. Same inputs as ours: NepTTS-Bench sentences with the female_news prompt, and the
FLEURS cloning items (prompt clip + gold transcript). Scoring happens in our venv:
    python tools/rescore_deva.py bench omnivoice_femnews     (+ eval_bench-style UTMOS)
    python tools/eval_fleurs.py score-dir omnivoice /home/data/fleurs/runs/omnivoice
Usage: /home/venv_omni/bin/python tools/gen_omnivoice.py bench|fleurs"""
import json
import os
import sys

import soundfile as sf
import torch
from omnivoice import OmniVoice

m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
what = sys.argv[1]
if what == "bench":
    ref = json.load(open("/home/data/ref/bench/refs.json"))["female_news"]
    out = "/home/data/bench/runs/omnivoice_femnews"
    os.makedirs(out, exist_ok=True)
    for x in json.load(open("/home/data/bench/sentences.json")):
        if x["category"] == "natural_speech":
            continue
        p = f"{out}/{x['sent_id']}.wav"
        if not os.path.exists(p):
            a = m.generate(text=x["text_devanagari"], ref_audio=ref["audio"], ref_text=ref["text"], language="npi")
            sf.write(p, a[0], 24000)
elif what == "codeswitch":  # same prompt as ours; English left in Latin letters (OmniVoice reads both scripts)
    ref = json.load(open("/home/data/ref/bench/refs.json"))["female_news"]
    out = "/home/data/codeswitch/omnivoice_heldout"
    os.makedirs(out, exist_ok=True)
    for it in json.load(open("/home/nepali-tts/data/codeswitch_heldout.json")):
        p = f"{out}/{it['id']}.wav"
        if not os.path.exists(p):
            a = m.generate(text=it["text"], ref_audio=ref["audio"], ref_text=ref["text"], language="npi")
            sf.write(p, a[0], 24000)
else:
    out = "/home/data/fleurs/runs/omnivoice"
    os.makedirs(out, exist_ok=True)
    for it in json.load(open("/home/data/fleurs/items.json")):
        p = f"{out}/{it['id']}.wav"
        if not os.path.exists(p):
            a = m.generate(text=it["text"], ref_audio=it["ref_wav"], ref_text=it["ref_text"], language="npi")
            sf.write(p, a[0], 24000)
print("OMNI_DONE", what)
