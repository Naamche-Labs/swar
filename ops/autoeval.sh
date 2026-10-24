#!/bin/bash
# autoeval.sh (PAT=prefix): score every new EMA checkpoint in /home/ema on the held-out set,
# then on NepTTS-Bench with the chosen settings (calm male_wiki prompt, short-text slowdown)
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home/nepali-tts
while true; do
  for f in /home/ema/${PAT:-nepali_}*.pt; do
    [ -e "$f" ] || continue
    N=$(basename $f .pt)
    M=chars; [[ $N == *respell* ]] && M=respell
    if [ ! -f /home/data/eval/$N/scores.json ]; then
      sleep 20  # let scp finish
      TEXT_MODE=$M /home/venv/bin/python tools/eval_heldout.py run $N $f 2>&1 | grep -E "CER|Error" | tail -2
    fi
    if [ ! -f /home/data/bench/runs/b_$N.json ]; then
      REF=male_wiki SHORT_CHARS=15 SHORT_SPEED=0.5 TEXT_MODE=$M /home/venv/bin/python tools/eval_bench.py run b_$N $f 2>&1 | grep -E "CER|Error" | tail -2
    fi
  done
  sleep 120
done
