"""How native does the English sound? CommonAccent ECAPA (speechbrain, 16 English accents) on the pure-English lines:
P(us)+P(england)+P(canada)+P(australia) = "native-like", P(indian) = South Asian accent; real LibriTTS-R speech as the
reference. Usage: python tools/accent_score.py DIR"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

d = sys.argv[1]
clf = EncoderClassifier.from_hparams("Jzuluaga/accent-id-commonaccent_ecapa", savedir="/home/hf_cache/accent",
                                     run_opts={"device": "cuda"})
labels = clf.hparams.label_encoder.decode_ndim(list(range(16)))
NATIVE = {"us", "england", "canada", "australia", "newzealand", "ireland", "scotland"}


def probs(p):
    with torch.no_grad():
        out = clf.classify_file(p)
    pr = torch.softmax(out[0][0], -1).cpu().numpy()  # out[0]: log-likelihoods
    return {labels[i]: float(pr[i]) for i in range(len(pr))}


agg = defaultdict(list)
for r in json.load(open(f"{d}/rows.json")):
    if r["k"].startswith("en_"):
        agg[(r["run"], r["voice"])].append(probs(f"{d}/{r['run']}_{r['voice']}_{r['k']}.wav"))
for f in sorted(glob.glob("/home/data/libritts_ref/*.wav"))[:12]:
    agg[("real_native", "librispeech")].append(probs(f))
for (run, voice), ps in sorted(agg.items()):
    nat = np.mean([sum(v for k, v in p.items() if k in NATIVE) for p in ps])
    ind = np.mean([p.get("indian", 0) for p in ps])
    top = max(labels, key=lambda l: np.mean([p.get(l, 0) for p in ps]))
    print(f"ACCENT {run:11s} {voice:11s} native-like {100 * nat:5.1f}%  indian {100 * ind:5.1f}%  top {top}")
