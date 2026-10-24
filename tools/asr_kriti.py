"""Kriti Telephony (Naamche, 119M hybrid RNNT-CTC Conformer, tuned on spontaneous / code-switched Nepali) transcripts
for a list of wavs. Runs in /home/venv_nemo (AI4Bharat NeMo fork @8dce88c, pins in memory of the Kriti work).
Usage: NUMBA_CUDA_USE_NVIDIA_BINDING=1 python tools/asr_kriti.py IN.json OUT.json   (OUT: {wav: {"rnnt": .., "ctc": ..}})
       ... tools/asr_kriti.py --serve   (stays loaded: one JSON list of wavs per stdin line -> "RESULT {json}" line)"""
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

ckpt = os.environ.get("KRITI_NEMO") or hf_hub_download("Aarjan/kriti-telephony", "kriti_telephony.nemo")
m = EncDecHybridRNNTCTCBPEModel.restore_from(ckpt, strict=False, map_location="cuda" if torch.cuda.is_available() else "cpu")
m.eval()


def run(decoder, paths):
    m.cur_decoder = decoder
    with torch.no_grad():
        hy = m.transcribe(paths, batch_size=8, language_id="ne")
    hy = hy[0] if isinstance(hy, tuple) else hy
    return [h.text if hasattr(h, "text") else str(h) for h in hy]


def both(paths):
    rnnt, ctc = run("rnnt", paths), run("ctc", paths)
    return {p: {"rnnt": r.strip(), "ctc": c.strip()} for p, r, c in zip(paths, rnnt, ctc)}


if sys.argv[1] == "--serve":
    print("READY", flush=True)
    for line in sys.stdin:
        print("RESULT " + json.dumps(both(json.loads(line)), ensure_ascii=False), flush=True)
else:
    paths = json.load(open(sys.argv[1]))
    out = both(paths)
    for p in paths[:5]:
        print("KRITI", os.path.basename(p), "|", out[p]["rnnt"][:80], flush=True)
    json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    print("KRITI_DONE", len(out))
