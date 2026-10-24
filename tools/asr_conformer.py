"""IndicConformer-600M (AI4Bharat, MIT) Nepali transcripts, CTC and RNNT decoding, for a list of wavs.
Runs in its own venv (/home/venv_conf: transformers 4.46 + onnxruntime-gpu; the model ships ONNX code).
Usage: python tools/asr_conformer.py IN.json OUT.json   (IN: ["a.wav", ...]; OUT: {wav: {"ctc": .., "rnnt": ..}})"""
import json
import sys

import librosa
import torch
from transformers import AutoModel

paths = json.load(open(sys.argv[1]))
m = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
out = {}
for p in paths:
    w, _ = librosa.load(p, sr=16000)
    x = torch.tensor(w).unsqueeze(0)
    out[p] = {"ctc": m(x, "ne", "ctc").strip(), "rnnt": m(x, "ne", "rnnt").strip()}
    print("CONF", p.rsplit("/", 1)[-1], "|", out[p]["rnnt"][:80], flush=True)
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("CONF_DONE", len(out))
