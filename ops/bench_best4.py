"""NepTTS-Bench with the full current pipeline (fixes -> Nepal pronunciation -> English layer, pace-calibrated
female_news preset, NFE 16), 4 seeds per sentence; per sentence keep the take an INDEPENDENT ASR (XLS-R Nepali,
gagan3012) transcribes best, so the pick is not tuned to the MMS scorer. Writes runs full_s0..3 and full_best4.
Usage (box): python ops/bench_best4.py CKPT"""
import json
import os
import shutil
import sys

sys.path[:0] = ["tools", "pipeline", "."]
from f5_tts.api import F5TTS  # noqa: E402
from clone import speed_for, to_model_text  # noqa: E402
from eval_bench import Reranker, sentences  # noqa: E402
from frontend import normalize  # noqa: E402

ck = sys.argv[1]
ref = json.load(open("/home/data/ref/bench/refs.json"))["female_news"]
tts = F5TTS(model="F5TTS_Base", ckpt_file=ck, vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
tts.ema_model.float()
R = "/home/data/bench/runs"
S = sentences()
for seed in range(4):
    d = f"{R}/full_s{seed}"
    os.makedirs(d, exist_ok=True)
    for x in S:
        p = f"{d}/{x['sent_id']}.wav"
        if not os.path.exists(p):
            g = to_model_text(x["text_devanagari"], "chars")
            tts.infer(ref_file=ref["audio"], ref_text=normalize(ref["text"]), gen_text=g, file_wave=p, remove_silence=True,
                      seed=seed, nfe_step=16, speed=speed_for(g, ref.get("speed", 1.0)))
del tts
rr = Reranker()
os.makedirs(f"{R}/full_best4", exist_ok=True)
for x in S:
    takes = [f"{R}/full_s{s}/{x['sent_id']}.wav" for s in range(4)]
    best = min(takes, key=lambda p: rr.cer(p, normalize(x["text_devanagari"])))
    shutil.copy(best, f"{R}/full_best4/{x['sent_id']}.wav")
print("BEST4_DONE")
