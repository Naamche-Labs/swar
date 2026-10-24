"""Score ops/english_test.py output: English intelligibility (Whisper-large-v3 English WER on the pure-English lines),
naturalness (UTMOS) and voice match (ECAPA to the prompt), per run and voice. Usage: python tools/english_score.py DIR"""
import json
import re
import sys
from collections import defaultdict

import librosa
import numpy as np
import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from eval_heldout import ed  # noqa: E402
from mos import utmos  # noqa: E402

d = sys.argv[1]
rows = json.load(open(f"{d}/rows.json"))
from transformers import pipeline  # noqa: E402
wh = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0, torch_dtype=torch.float16)
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"}, savedir="/home/hf_cache/ecapa")
PROMPT = {"podcast": "/home/podcast/pieces/pod_080.wav", "studio": "/home/data/ref/bench/female_news.wav"}


def emb(p):
    x = librosa.load(p, sr=16000)[0]
    with torch.no_grad():
        return torch.nn.functional.normalize(spk.encode_batch(torch.tensor(x).unsqueeze(0).cuda()).squeeze(), dim=-1)


words = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    p = f"{d}/{r['run']}_{r['voice']}_{r['k']}.wav"
    x = librosa.load(p, sr=16000)[0]
    key = (r["run"], r["voice"])
    agg[key]["utmos"].append(utmos(x))
    agg[key]["sim"].append(float(emb(p) @ emb(PROMPT[r["voice"]])))
    if r["k"].startswith("en_"):
        h = wh({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "english", "task": "transcribe"})["text"]
        ref = words(r["text"])
        agg[key]["wer"].append(ed(ref, words(h)) / len(ref))
        r["whisper"] = h
        print(f"{r['run']:8s} {r['voice']:7s} {r['k']:10s} WER {100 * agg[key]['wer'][-1]:4.0f} | {h.strip()[:70]}", flush=True)
for (run, voice), m in sorted(agg.items()):
    print(f"SCORE {run:8s} {voice:7s} English WER {100 * np.mean(m['wer']):5.1f}  UTMOS {np.mean(m['utmos']):.2f}  "
          f"voice match {np.mean(m['sim']):.3f}")
json.dump(rows, open(f"{d}/rows_scored.json", "w"), ensure_ascii=False, indent=1)
