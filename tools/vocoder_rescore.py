"""Re-score vocoder_eval.py outputs already on disk (UTMOS; timbre distance to the real recording for copy runs).
Usage (box 2): python tools/vocoder_rescore.py TAG [TAG ...]"""
import json
import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_test import compare  # noqa: E402
from mos import utmos  # noqa: E402

R = "/home/data/rasa_test"
real = {it["id"]: it["real_wav"] for it in json.load(open(f"{R}/items.json"))}
for tag in sys.argv[1:]:
    for kind in ("copy_bigvgan", "tts_bigvgan", "tts_vocos"):
        d = f"{R}/runs/{kind}_{tag}"
        U, T = [], []
        for f in sorted(os.listdir(d)):
            y = librosa.load(f"{d}/{f}", sr=16000)[0]
            U.append(utmos(y))
            if kind.startswith("copy"):
                T.append(compare(librosa.load(real[f[:-4]], sr=16000)[0], y)[0])
        print(f"VOC {tag} {kind:13s} UTMOS {np.mean(U):.3f}" + (f" timbre {np.mean(T):.2f}" if T else "") + f" (n={len(U)})", flush=True)
