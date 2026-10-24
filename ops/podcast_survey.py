"""Who speaks in these podcasts, and which voices are clean enough to become a preset? (podcast A and two other shows,
used with permission, 2026-10-23.) Every file is cut at pauses (3.5-10 s), pieces are grouped by voice across all files
(ECAPA, average linkage at 0.62), and each voice gets its speech time, files, pitch (male / female), UTMOS and three
sample clips to listen to. No transcripts yet: Kriti runs later on the voice that is picked.
Usage (GPU box): python ops/podcast_survey.py OUT FILE..."""
import json
import os
import subprocess
import sys
from collections import defaultdict

import librosa
import numpy as np

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402
from mos import utmos  # noqa: E402


def f0_median(x):
    f = librosa.yin(x, fmin=60, fmax=400, sr=16000)
    v = librosa.feature.rms(y=x)[0][:len(f)] > 0.02
    return float(np.median(f[:len(v)][v])) if v.any() else 0.0


out, files = sys.argv[1], sys.argv[2:]
os.makedirs(f"{out}/pieces", exist_ok=True)
P = []
for fi, src in enumerate(files):
    wav = f"{out}/src{fi}.wav"
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "24000", wav], check=True)
    ps = A.cut_pieces(wav, f"{out}/pieces", f"f{fi}")
    for p in ps:
        x = librosa.load(p["wav"], sr=16000)[0]
        p.update(file=fi, emb=A.emb(x), f0=f0_median(x), nat=float(utmos(x)))
    P += ps
    print(f"file {fi} {os.path.basename(src)}: {len(ps)} pieces, {sum(p['speech'] for p in ps) / 60:.0f} min speech", flush=True)

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
lab = fcluster(linkage(np.stack([p["emb"] for p in P]), method="average", metric="cosine"), t=1 - 0.62, criterion="distance")
G = defaultdict(list)
for p, l in zip(P, lab):
    G[int(l)].append(p)
voices = sorted(G.values(), key=lambda g: -sum(p["speech"] for p in g))[:12]
os.makedirs(f"{out}/listen", exist_ok=True)
meta = []
for i, g in enumerate(voices):
    name = f"v{i:02d}"
    f0 = float(np.median([p["f0"] for p in g if p["f0"] > 0]))
    per_file = defaultdict(float)
    for p in g:
        per_file[p["file"]] += p["speech"] / 60
    c = np.mean([p["emb"] for p in g], 0)
    c /= np.linalg.norm(c)
    typ = sorted(g, key=lambda p: -(float(p["emb"] @ c) + 0.1 * p["nat"]))  # typical of the voice and clean
    for k, p in enumerate(typ[:3]):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p["wav"], "-b:a", "96k", f"{out}/listen/{name}_{k}.mp3"],
                       check=True)
    clean = [p for p in g if p["nat"] >= 3.0]
    meta.append({"name": name, "minutes": round(sum(p["speech"] for p in g) / 60, 1), "pieces": len(g),
                 "clean_minutes": round(sum(p["speech"] for p in clean) / 60, 1), "f0": round(f0),
                 "gender_guess": "male" if f0 < 165 else "female", "utmos": round(float(np.mean([p["nat"] for p in g])), 2),
                 "per_file_min": {str(k): round(v, 1) for k, v in sorted(per_file.items())},
                 "samples": [os.path.basename(p["wav"]) for p in typ[:20]]})
    print(meta[-1] | {"samples": len(meta[-1]["samples"])}, flush=True)
json.dump({"files": files, "voices": meta}, open(f"{out}/survey.json", "w"), ensure_ascii=False, indent=1)
json.dump([{k: p[k] for k in ("wav", "file", "speech", "f0", "nat")} | {"voice": int(l)} for p, l in zip(P, lab)],
          open(f"{out}/pieces.json", "w"))
np.save(f"{out}/emb.npy", np.stack([p["emb"] for p in P]))
print("SURVEY_DONE")
