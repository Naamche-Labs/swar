#!/bin/bash
# bench_ckpt.sh NAME...: NepTTS-Bench with the chosen settings (calm male_wiki prompt, short-text slowdown)
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1 REF=male_wiki SHORT_CHARS=15 SHORT_SPEED=0.5
cd /home/nepali-tts
for N in "$@"; do
  M=chars; [[ $N == *respell* ]] && M=respell
  TEXT_MODE=$M /home/venv/bin/python tools/eval_bench.py run b_$N /home/ema/$N.pt 2>&1 | grep -E "CER|Error" | tail -2
done
