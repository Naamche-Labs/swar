"""Does the podcast-voices fine-tune beat zero-shot? Per voice, its 10 held-out pieces (ops/build_podvoices.py, never
trained on): the clone says the same words from the voice's prompt (/home/v4/podref), compared to the real piece
(ECAPA similarity), read back by Kriti (pronunciation CER against the piece's transcript) and UTMOS. Zero-shot on
unseen speakers too (FLEURS items, never trained on): similarity to the speaker's real clip and CER.
Usage (GPU box): KRITI_ONNX=/home/kriti_onnx python ops/eval_podvoices.py TAG CKPT"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402
from clone import patch_f5_chunker, set_pace, speed_for, to_model_text  # noqa: E402
from frontend import normalize  # noqa: E402

PROMPT = {"host": "host0", "guest1": "v03", "guest2": "v05", "podAf": "podAf", "thrA": "thrA", "thrB": "thrB",
          "p4a": "p4a", "p4b": "p4b", "p4c": "p4c", "p4d": "p4d"}
tag, ckpt = sys.argv[1:3]
out = f"/home/v4/evalpod/{tag}"
os.makedirs(out, exist_ok=True)
from f5_tts.api import F5TTS  # noqa: E402
patch_f5_chunker()
m = F5TTS(model="F5TTS_Base", ckpt_file=ckpt, vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
m.ema_model.float()
held = json.load(open("/home/data/podvoices/heldout.json"))


def gen(ref, rt, text, p):
    set_pace(rt, ref)
    gt = to_model_text(text, "chars")
    m.infer(ref_file=ref, ref_text=rt, gen_text=gt, file_wave=p, nfe_step=32, seed=0, cfg_strength=2.0,
            speed=speed_for(gt, 1.0), remove_silence=True)
    return p


res = {}
for v, pr in PROMPT.items():
    ref, rt = f"/home/v4/podref/{pr}.wav", normalize(open(f"/home/v4/podref/{pr}.txt").read().strip())
    items = [h for h in held[v] if normalize(h["text"]) != rt][:8]
    files = [gen(ref, rt, h["text"], f"{out}/{v}_{i}.wav") for i, h in enumerate(items)]
    heard = A.read_back(files)
    rows = [{"sim": float(A.emb(f) @ A.emb(h["wav"])), "cer": A.pcer(normalize(h["text"]), t), "nat": A._nat(f)}
            for f, h, t in zip(files, items, heard)]
    res[v] = {k: float(np.mean([r[k] for r in rows])) for k in ("sim", "cer", "nat")} | {"n": len(rows)}
    print(f"VOICE {tag:8s} {v:7s} sim {res[v]['sim']:.3f} cer {100 * res[v]['cer']:5.1f} utmos {res[v]['nat']:.2f}", flush=True)
fl = json.load(open("/home/data/fleurs/items.json"))[:24]
files = [gen(it["ref_wav"], normalize(it["ref_text"]), it["text"], f"{out}/fleurs_{i}.wav") for i, it in enumerate(fl)]
heard = A.read_back(files)
rows = [{"sim": float(A.emb(f) @ A.emb(it["real_wav"])), "cer": A.pcer(normalize(it["text"]), t), "nat": A._nat(f)}
        for f, it, t in zip(files, fl, heard)]
res["zero_shot_fleurs"] = {k: float(np.mean([r[k] for r in rows])) for k in ("sim", "cer", "nat")} | {"n": len(rows)}
z = res["zero_shot_fleurs"]
print(f"VOICE {tag:8s} unseen  sim {z['sim']:.3f} cer {100 * z['cer']:5.1f} utmos {z['nat']:.2f}", flush=True)
json.dump(res, open(f"{out}/eval.json", "w"), indent=1)
print("EVAL_DONE")
