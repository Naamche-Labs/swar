#!/bin/bash
# respell A/B evals: held-out (vs pilot chars at same step) + bench
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1 TEXT_MODE=respell
cd /home/nepali-tts
for s in $STEPS; do
  /home/venv/bin/python tools/eval_heldout.py run respell_$s /home/ema/nepali_respell_$s.pt 2>&1 | grep -E "CER|Error" | tail -2
done
echo RQ_DONE
