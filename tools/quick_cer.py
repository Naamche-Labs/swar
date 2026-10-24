"""MMS-1b-all (npi) CER on dirs holding meta.json [{wav, text}].

Usage: python tools/quick_cer.py DIR [DIR ...]
"""
import json
import re
import sys
import unicodedata

import librosa
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                       ignore_mismatched_sizes=True).cuda().eval()


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[।॥.,?!;:\"'()\-–—]", " ", s)
    return " ".join(s.split())


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


for D in sys.argv[1:]:
    meta = json.load(open(f"{D}/meta.json"))
    E = N = 0
    for m in meta:
        w, _ = librosa.load(m["wav"], sr=16000)
        inp = {k: v.cuda() for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            ids = model(**inp).logits.argmax(-1)[0]
        hyp, ref = norm(proc.decode(ids)), norm(m["text"])
        e = ed(ref, hyp)
        E += e
        N += len(ref)
        m["hyp"], m["cer"] = hyp, e / max(1, len(ref))
        print(f"  {m['cer'] * 100:5.1f}%  REF {ref}\n         HYP {hyp}")
    json.dump(meta, open(f"{D}/meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"{D}: CER {100 * E / N:.2f}% over {len(meta)} clips")
