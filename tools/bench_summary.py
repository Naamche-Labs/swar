"""Compact NepTTS-Bench summary (no per-sentence rows): standard CER (Devanagari MMS, mean per file),
pronunciation CER (tools/pron_cer.py) and mean UTMOS (tools/utmos_runs.py), overall and per category.
Usage: python tools/bench_summary.py OUT_JSON NAME [NAME ...]
Reads /home/data/deva/bench.json and /home/data/deva/bench_utmos.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pron_cer import pcer  # noqa: E402

deva = json.load(open("/home/data/deva/bench.json"))
mos = json.load(open("/home/data/deva/bench_utmos.json"))
out = {}
for n in sys.argv[2:]:
    r, m = deva[n], mos.get(n, {})
    cats = sorted(r["by_style"])
    out[n] = {"n": r["n"], "standard_cer": round(r["mean_file_cer"], 2), "corpus_cer": round(r["cer"], 2),
              "pron_cer": round(pcer(r["rows"]), 2), "utmos": round(m["utmos"], 3) if m else None,
              "by_category": {c: {"n": r["by_style"][c]["n"], "standard_cer": round(r["by_style"][c]["mean_file_cer"], 2),
                                  "pron_cer": round(pcer([x for x in r["rows"] if x["style"] == c]), 2),
                                  "utmos": round(m["by_style"][c], 3) if m else None} for c in cats}}
    print(f"{n:22s} std {out[n]['standard_cer']:6.2f}  pron {out[n]['pron_cer']:6.2f}  UTMOS {out[n]['utmos']}  n={out[n]['n']}")
    for c, v in out[n]["by_category"].items():
        print(f"    {c:28s} n={v['n']:3d}  std {v['standard_cer']:6.2f}  pron {v['pron_cer']:6.2f}  UTMOS {v['utmos']}")
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
