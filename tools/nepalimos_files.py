"""Per-file NepaliMOS (NepTTS-Bench's Nepali naturalness predictor, public checkpoint) for one or more
directories of sent_XXX.wav/mp3. Runs in the neptts-eval venv (/home/venv_ne).
Usage: /home/venv_ne/bin/python tools/nepalimos_files.py OUT.json DIR [DIR ...]"""
import json
import sys
from pathlib import Path

from neptts_eval.nepalimos_eval import evaluate_nepalimos

out = {}
for d in sys.argv[2:]:
    files = {p.stem: p for p in sorted(Path(d).glob("sent_*.*")) if p.suffix in (".wav", ".mp3")}
    r = evaluate_nepalimos(files)
    out[d] = r["per_file"]
    print(d, round(r["avg_mos"], 3), r["n_scored"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
