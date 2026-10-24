"""NepTTS-Bench for the final model (2026-10-23), same protocol as ops/bench_best4.py's full_s0 row (full frontend,
pace-calibrated female_news preset, NFE 16, seed 0, one take) so the paper's numbers come from the model that ships.
Usage (box): python ops/bench_final.py NAME CKPT   -> /home/data/bench/runs/NAME, then rescore_deva / pron_cer / utmos_runs"""
import json
import os
import sys

sys.path[:0] = ["tools", "pipeline", "."]
from f5_tts.api import F5TTS  # noqa: E402
from clone import patch_f5_chunker, speed_for, to_model_text  # noqa: E402
from eval_bench import sentences  # noqa: E402
from frontend import normalize  # noqa: E402

name, ck = sys.argv[1:3]
ref = json.load(open("/home/data/ref/bench/refs.json"))["female_news"]
patch_f5_chunker()
tts = F5TTS(model="F5TTS_Base", ckpt_file=ck, vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
tts.ema_model.float()
d = f"/home/data/bench/runs/{name}"
os.makedirs(d, exist_ok=True)
for x in sentences():
    p = f"{d}/{x['sent_id']}.wav"
    if not os.path.exists(p):
        g = to_model_text(x["text_devanagari"], "chars")
        tts.infer(ref_file=ref["audio"], ref_text=normalize(ref["text"]), gen_text=g, file_wave=p, remove_silence=True,
                  seed=0, nfe_step=16, speed=speed_for(g, ref.get("speed", 1.0)))
print("BENCH_FINAL_DONE", name, len(os.listdir(d)))
