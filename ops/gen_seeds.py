"""Extra seeds of a bench run (NFE 16, frontend on) for best-of-N reranking tests.
Usage: python ops/gen_seeds.py CKPT RUN_NAME SEED [SEED ...]"""
import os
import sys

sys.path[:0] = ["tools", "pipeline", "."]
from f5_tts.api import F5TTS  # noqa: E402
from eval_bench import bench_ref, sentences, speed_for  # noqa: E402
from frontend import normalize  # noqa: E402

ckpt, name = sys.argv[1], sys.argv[2]
tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt,
            vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
tts.ema_model.float()
ref = bench_ref()
for seed in map(int, sys.argv[3:]):
    out = f"/home/data/bench/runs/{name}_s{seed}"
    os.makedirs(out, exist_ok=True)
    for x in sentences():
        p = f"{out}/{x['sent_id']}.wav"
        if not os.path.exists(p):
            g = normalize(x["text_devanagari"])
            tts.infer(ref_file=ref["audio"], ref_text=ref["text"], gen_text=g, file_wave=p, remove_silence=True,
                      seed=seed, nfe_step=16, speed=speed_for(g, 1.0))
print("SEEDS_DONE")
