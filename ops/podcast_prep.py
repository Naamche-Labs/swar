"""Two-host Nepali podcast (used with permission) -> clean, labelled pieces for personality: conversational prompts
and a fine-tune set. Cut at pauses (3.5-10 s), split speakers by ECAPA clusters (the two clusters with the most
speech are the hosts; music, jingles and guests fall out), Kriti transcripts with RNNT/CTC agreement as confidence,
UTMOS for cleanliness, pace, and emotion2vec+ labels where funasr is installed.
Usage (box 2): python ops/podcast_prep.py AUDIO OUT_DIR"""
import json
import os
import sys
from collections import Counter

import librosa
import numpy as np

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402

audio, out = sys.argv[1:3]
os.makedirs(f"{out}/pieces", exist_ok=True)
P = A.cut_pieces(audio, f"{out}/pieces", "pod")
for p in P:
    p["emb"] = A.emb(p["wav"])
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
lab = fcluster(linkage(np.stack([p["emb"] for p in P]), method="average", metric="cosine"), t=1 - 0.62, criterion="distance")
tot = Counter()
for p, l in zip(P, lab):
    tot[l] += p["speech"]
hosts = [l for l, _ in tot.most_common(2)]
print("clusters by speech (s):", [(int(l), round(s)) for l, s in tot.most_common(6)], flush=True)
keep = [p for p, l in zip(P, lab) if l in hosts]
for p, l in zip(P, lab):
    p["host"] = "AB"[hosts.index(l)] if l in hosts else None
K = A.kriti().read([p["wav"] for p in keep])
try:
    from funasr import AutoModel
    emo = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)
except Exception as e:  # noqa: BLE001
    print("no emotion model:", e, flush=True)
    emo = None
for p in keep:
    h = K[p["wav"]]
    p["text"], p["confidence"] = h["rnnt"], 1 - min(1.0, A.pcer(h["rnnt"], h["ctc"]))
    p["utmos"] = A._nat(p["wav"])
    p["rate"] = A.letters(p["text"]) / max(0.5, p["speech"]) if p["text"] else 0.0
    if emo:
        r = emo.generate(p["wav"], granularity="utterance", extract_embedding=False)[0]
        sc = {lb.split("/")[-1]: s for lb, s in zip(r["labels"], r["scores"])}
        p["emotion"], p["emotion_p"] = max(sc.items(), key=lambda kv: kv[1])
json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items() if k != "emb"} for p in P],
          open(f"{out}/pieces.json", "w"), ensure_ascii=False, indent=1)
for h in "AB":
    ps = [p for p in keep if p["host"] == h]
    good = [p for p in ps if p["utmos"] >= 3.0 and p["confidence"] >= 0.8]
    print(f"HOST {h}: {len(ps)} pieces, {sum(p['speech'] for p in ps) / 60:.1f} min speech; clean+confident "
          f"{len(good)} ({sum(p['speech'] for p in good) / 60:.1f} min); UTMOS {np.mean([p['utmos'] for p in ps]):.2f}; "
          f"pace {np.median([p['rate'] for p in ps if p['rate']]):.1f} l/s; emotions {Counter(p.get('emotion') for p in ps).most_common(5)}",
          flush=True)
print("PODCAST_DONE")
