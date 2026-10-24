"""Short-phrase training set (2026-10-24). The podcast voices (Sabina, Maya, Nisha) score the highest UTMOS but make
40-80% more errors than the studio voice on short phrases (NepTTS-Bench is 94% phrases under 30 letters), and no
prompt fixes it: the model has seen these voices only in 3.5-10 s casual pieces. Here every podcast training piece is
cut at word boundaries into 1-4 word clips (MMS forced alignment, cut in the middle of a pause of at least 60 ms), kept
only if Kriti reads the clip back as written (confidence >= 0.8, pronunciation CER <= 0.15). Output: OUT/train.csv =
the podvoices rows (so nothing is forgotten) + the short clips.
Usage (GPU box): KRITI_ONNX=/home/kriti_onnx python ops/build_short.py /home/data/podvoices OUT"""
import json
import os
import random
import re
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path[:0] = ["tools", "pipeline", "."]
import autoclone as A  # noqa: E402
from frontend import normalize  # noqa: E402
from kriti_onnx import KritiONNX  # noqa: E402
from rescore_deva import DevaMMS  # noqa: E402

src, out = sys.argv[1:3]
MIN_GAP, PER_PIECE = 0.06, int(os.environ.get("PER_PIECE", 3))
os.makedirs(f"{out}/wav", exist_ok=True)
mms = DevaMMS()
vocab = mms.proc.tokenizer.get_vocab()
pad = mms.proc.tokenizer.pad_token_id
rows = [l.rstrip("\n").split("|", 1) for l in open(f"{src}/train.csv", encoding="utf-8")][1:]  # header audio_file|text
pod = [(w, t) for w, t in rows if re.search(r"/pod[0-9]/", w)]
print(len(rows), "rows,", len(pod), "podcast pieces", flush=True)


def words_with_times(w16, text):
    """[(word, start_s, end_s)] by forced alignment; None if the text does not align."""
    inp = {k: v.to(mms.model.device) for k, v in mms.proc(w16, sampling_rate=16000, return_tensors="pt").items()}
    with torch.no_grad():
        lg = torch.log_softmax((mms.model(**inp).logits[0] + mms.mask).float(), -1).cpu()
    ws = [w for w in text.split() if any(c in vocab for c in w)]
    chars, owner = [], []
    for i, w in enumerate(ws):
        if i:
            chars.append("|"), owner.append(-1)
        for c in w:
            if c in vocab:
                chars.append(c), owner.append(i)
    try:
        al, sc = AF.forced_align(lg.unsqueeze(0), torch.tensor([[vocab[c] for c in chars]]), blank=pad)
    except Exception:
        return None
    spans = AF.merge_tokens(al[0], sc[0].exp())
    if len(spans) != len(chars):
        return None
    t = {}
    for s, o in zip(spans, owner):
        if o >= 0:
            a, b = t.get(o, (s.start, s.end))
            t[o] = (min(a, s.start), max(b, s.end))
    fr = 0.02  # MMS frame
    return [(w, t[i][0] * fr, t[i][1] * fr) for i, w in enumerate(ws) if i in t]


kr = KritiONNX(os.environ["KRITI_ONNX"])
cands, rng = [], random.Random(0)
for n, (wav, text) in enumerate(pod):
    y, sr = librosa.load(wav, sr=24000)
    ww = words_with_times(librosa.resample(y, orig_sr=24000, target_sr=16000), text)
    if not ww or len(ww) < 3:
        continue
    # boundary k sits between word k-1 and k; usable if the pause there is long enough (and at both ends)
    ok = [0] + [k for k in range(1, len(ww)) if ww[k][1] - ww[k - 1][2] >= MIN_GAP] + [len(ww)]
    spans = [(a, b) for i, a in enumerate(ok) for b in ok[i + 1:] if 1 <= b - a <= 4]
    rng.shuffle(spans)
    for a, b in spans[:PER_PIECE]:
        s = ww[a][1] - (0.03 if a == 0 else (ww[a][1] - ww[a - 1][2]) / 2)
        e = ww[b - 1][2] + (0.05 if b == len(ww) else (ww[b][1] - ww[b - 1][2]) / 2)
        clip = y[max(0, int(s * 24000)):int(e * 24000)]
        if len(clip) < 0.4 * 24000:
            continue
        p = f"{out}/wav/{n:05d}_{a}_{b}.wav"
        sf.write(p, clip, 24000)
        cands.append({"wav": p, "text": " ".join(w for w, _, _ in ww[a:b])})
    if n % 200 == 0:
        print(n, len(cands), flush=True)

keep = []
for i in range(0, len(cands), 64):
    batch = cands[i:i + 64]
    for c, (h, conf) in zip(batch, kr.read_conf([c["wav"] for c in batch])):
        err = A.pcer(normalize(c["text"]), normalize(h, english=False))
        if conf >= 0.8 and err <= 0.15:
            keep.append(c)
        else:
            os.remove(c["wav"])
dur = sum(sf.info(c["wav"]).duration for c in keep)
with open(f"{out}/train.csv", "w", encoding="utf-8") as f:
    f.write("audio_file|text\n")
    for w, t in rows:
        f.write(f"{w}|{t}\n")
    for c in keep:
        f.write(f"{c['wav']}|{c['text']}\n")
json.dump({"pieces": len(pod), "candidates": len(cands), "kept": len(keep), "short_min": round(dur / 60, 1)},
          open(f"{out}/stats.json", "w"), indent=1)
print("SHORT_DONE", len(cands), "candidates,", len(keep), "kept,", round(dur / 60, 1), "min", flush=True)
