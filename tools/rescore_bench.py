"""Second-opinion CER on every NepTTS-Bench run with an independent Nepali ASR.

MMS alone is noisy on 1-3 word phrases (it falls back to Latin letters), and all our
models land at 10-12% mean-file CER. Re-scoring with a second ASR shows whether a
difference is real (both agree) or scorer noise.

Usage: python tools/rescore_bench.py [ASR_ID]   (default gagan3012/wav2vec2-xlsr-nepali)
Writes /home/data/bench/runs/second_opinion.json and prints a table.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_bench import BENCH, OUT, SYSTEMS, Reranker, sentences  # noqa: E402

ASR = sys.argv[1] if len(sys.argv) > 1 else "gagan3012/wav2vec2-xlsr-nepali"


def audio_for(name):
    """{sent_id: path} for one of our run dirs or a published system."""
    d = f"{OUT}/{name}" if os.path.isdir(f"{OUT}/{name}") else f"{BENCH}/audio/{name}"
    out = {}
    for x in sentences():
        hits = [p for p in glob.glob(f"{d}/{x['sent_id']}.*") if p.endswith((".wav", ".mp3"))]
        if hits:
            out[x["sent_id"]] = hits[0]
    return out


if __name__ == "__main__":
    rr = Reranker(ASR)
    text = {x["sent_id"]: x["text_devanagari"] for x in sentences()}
    path = f"{OUT}/second_opinion.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    names = SYSTEMS + ["gemini", "piper"] + sorted(
        os.path.basename(p) for p in glob.glob(f"{OUT}/*") if os.path.isdir(p))
    for n in dict.fromkeys(names):
        if n in res:
            continue
        wavs = audio_for(n)
        if not wavs:
            continue
        cers = [rr.cer(p, text[s]) for s, p in sorted(wavs.items())]
        mms = f"{OUT}/{n.replace('/', '_')}.json"
        res[n] = {"n": len(cers), "xlsr_mean_file_cer": 100 * sum(cers) / len(cers),
                  "mms_mean_file_cer": json.load(open(mms))["mean_file_cer"] if os.path.exists(mms) else None}
        json.dump(res, open(path, "w"), indent=1)
        r = res[n]
        print(f"{n:34s} n={r['n']:3d}  XLS-R {r['xlsr_mean_file_cer']:6.2f}%  MMS {r['mms_mean_file_cer'] or 0:6.2f}%", flush=True)
