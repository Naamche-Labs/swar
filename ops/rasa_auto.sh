#!/bin/bash
# eval box 2: Rasa test-split (human ceiling) on every new v3 EMA checkpoint
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home/nepali-tts
while true; do
  for f in /home/ema/nepali_v3*.pt /home/ema/nepali_polish_*.pt /home/ema/nepali_names_*.pt; do
    [ -e "$f" ] || continue
    N=$(basename $f .pt)
    [ -f /home/data/rasa_test/runs/$N.json ] && continue
    sleep 20
    /home/venv/bin/python tools/eval_rasa.py run $N $f 2>&1 | grep -E "CER|Error" | tail -2
  done
  sleep 120
done
