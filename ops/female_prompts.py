"""Livelier preset prompts: the calm news prompt made every line sound like a newsreader (user: "add personality,
the way Nepali people actually speak"). Candidates: the female studio speaker's CONV and HAPPY clips, 6-10 s, clean,
lively pace. Each renders the 8 demo lines (2 seeds); the best prompt by naturalness + pronunciation is then rendered
best-of-4 through Vocos and through the fine-tuned BigVGAN. Usage (box 2): python ops/female_prompts.py OUT [SPEAKER]"""
import json
import os
import shutil
import sys

import numpy as np

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402
from clone import patch_f5_chunker, set_pace, speed_for, to_model_text  # noqa: E402
from f5_tts.api import F5TTS  # noqa: E402
from frontend import normalize  # noqa: E402

out = sys.argv[1]
spk = sys.argv[2] if len(sys.argv) > 2 else "Female"
CK = {"Female": "/home/ckpts/preset_female/model_last.pt", "Male": "/home/ckpts/preset_male/model_last.pt"}[spk]
os.makedirs(out, exist_ok=True)
T = json.load(open("/home/out/female_demo/demo.json"))["texts"]
cand = []
for line in open("/home/data/nepali_tts/manifest.train.jsonl", encoding="utf-8"):
    x = json.loads(line)
    if x["speaker"] == f"rasa:{spk}" and x.get("style") in ("CONV", "HAPPY") and 6 <= x["dur"] <= 10 and x["snr"] >= 35:
        x["rate"] = A.letters(x["text"]) / x["dur"]
        cand.append(x)
pick = []
for st in ("CONV", "HAPPY"):  # liveliest clean clips per style, not the extreme ones
    c = sorted([x for x in cand if x["style"] == st], key=lambda x: -x["rate"])
    pick += c[len(c) // 10: len(c) // 10 + 3]
patch_f5_chunker()
m = F5TTS(model="F5TTS_Base", ckpt_file=CK, vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
m.ema_model.float()
voc_default = m.vocoder
big = A.BigVGANDecoder("/home/models/vocoder/bigvgan_nepali_56k.pt")


def render(pr, k, t, seed, tag, vocoder):
    m.vocoder = vocoder
    gen, rt = to_model_text(t, "chars"), normalize(pr["text"])
    set_pace(rt, pr["audio"])
    p = f"{out}/{tag}_{k}_s{seed}.wav"
    m.infer(ref_file=pr["audio"], ref_text=rt, gen_text=gen, file_wave=p, nfe_step=32, seed=seed, cfg_strength=2.0,
            speed=speed_for(gen, 1.0), remove_silence=True)
    return p


score = []
for i, pr in enumerate(pick):
    files = [(k, t, render(pr, k, t, s, f"cand{i}", voc_default)) for k, t in T.items() for s in (0, 1)]
    heard = A.read_back([f for _, _, f in files])
    cer = np.mean([A.pcer(normalize(t), h) for (_, t, _), h in zip(files, heard)])
    nat = np.mean([A._nat(f) for _, _, f in files])
    import librosa
    pace = np.mean([A.letters(t) / A.speech_sec(librosa.load(f, sr=16000)[0], 16000) for _, t, f in files])
    score.append((nat - cer, i))
    print(f"PROMPT {i} {pr['style']:5s} {pr['dur']:.1f}s rate {pr['rate']:.1f} -> UTMOS {nat:.2f} CER {100 * cer:.1f} "
          f"pace {pace:.1f} | {pr['text'][:60]}", flush=True)
best = pick[max(score)[1]]
json.dump({k: best[k] for k in ("audio", "text", "style", "dur")}, open(f"{out}/best_prompt.json", "w"), ensure_ascii=False)
for vtag, voc in (("vocos", voc_default), ("bigvgan", big)):
    for k, t in T.items():
        cs = [render(best, k, t, s, f"final_{vtag}", voc) for s in range(4)]
        hs = A.read_back(cs)
        rows = [{"f": f, "cer": A.pcer(normalize(t), h), "nat": A._nat(f), "h": h} for f, h in zip(cs, hs)]
        fl = min(r["cer"] for r in rows)
        b = max([r for r in rows if r["cer"] <= fl + 0.03], key=lambda r: r["nat"])
        shutil.copy(b["f"], f"{out}/{vtag}_{k}.wav")
        print(f"FINAL {vtag:7s} {k:11s} UTMOS {b['nat']:.2f} CER {100 * b['cer']:.0f} | {b['h'][:60]}", flush=True)
print("PROMPTS_DONE")
