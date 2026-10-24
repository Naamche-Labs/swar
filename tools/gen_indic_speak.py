"""NepTTS-Bench audio from Bodhan AI Indic-Speak (bodhan-ai/indic-speak: Llama-3.2-3B speech LM + SNAC codes
+ Vocos, 24 kHz; Indic Open Model License v1.0). Nepali is a "preview" language with two preset voices:
Srijana (female) and Sagar (male). The model has no voice cloning, so it cannot use our female_news prompt.
Uses the repo's own inference.py (needs transformers>=5 and snac). Runs in /home/venv_ispeak.
Sampling as the model card's defaults (temperature 0.6, top_p 0.9, top_k 50), seed 0 per sentence; if a
sentence yields no audio frame, retried with seeds 1 and 2.
Usage: /home/venv_ispeak/bin/python tools/gen_indic_speak.py [Srijana|Sagar]
  Srijana -> /home/data/bench/runs/indic_speak, Sagar -> /home/data/bench/runs/indic_speak_sagar"""
import json
import os
import sys
import time

from huggingface_hub import snapshot_download

spk = sys.argv[1] if len(sys.argv) > 1 else "Srijana"
repo = snapshot_download("bodhan-ai/indic-speak")
sys.path.insert(0, repo)
from inference import TTS  # noqa: E402  (the model repo's own inference code)

tts = TTS(repo)
out = "/home/data/bench/runs/" + ("indic_speak" if spk == "Srijana" else f"indic_speak_{spk.lower()}")
os.makedirs(out, exist_ok=True)
fails = []
t0 = time.time()
for x in json.load(open("/home/data/bench/sentences.json")):
    if x["category"] == "natural_speech":
        continue
    p = f"{out}/{x['sent_id']}.wav"
    if os.path.exists(p):
        continue
    for seed in (0, 1, 2):
        try:
            tts.save(p, tts(x["text_devanagari"], speaker=spk, seed=seed))
            break
        except ValueError as e:
            print("retry", x["sent_id"], seed, e, flush=True)
    else:
        fails.append(x["sent_id"])
    print(x["sent_id"], f"{time.time() - t0:.0f}s", flush=True)
print("FAILED", fails)
print("ISPEAK_DONE", spk)
