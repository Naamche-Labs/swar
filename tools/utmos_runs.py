"""Mean UTMOS (tools/mos.py) over NepTTS-Bench run dirs, overall and per category; files found the same way
rescore_deva.py finds them (rescore_bench.audio_for). Merges into /home/data/deva/bench_utmos.json.
Usage: python tools/utmos_runs.py NAME [NAME ...]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import librosa  # noqa: E402

from eval_bench import sentences  # noqa: E402
from mos import utmos  # noqa: E402
from rescore_bench import audio_for  # noqa: E402

path = "/home/data/deva/bench_utmos.json"
res = json.load(open(path)) if os.path.exists(path) else {}
cat = {x["sent_id"]: x["category"] for x in sentences()}
for n in sys.argv[1:]:
    rows = [(cat[s], utmos(librosa.load(p, sr=16000)[0])) for s, p in sorted(audio_for(n).items())]
    m = lambda v: sum(v) / max(1, len(v))
    res[n] = {"n": len(rows), "utmos": m([v for _, v in rows]),
              "by_style": {c: m([v for k, v in rows if k == c]) for c in sorted({k for k, _ in rows})}}
    print(f"UTMOS {n:28s} {res[n]['utmos']:.3f}  n={res[n]['n']}", flush=True)
    json.dump(res, open(path, "w"), indent=1)
