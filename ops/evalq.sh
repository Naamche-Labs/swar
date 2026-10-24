#!/bin/bash
# eval-box queue: NepTTS-Bench head-to-head, then inference-knob sweep on the pilot 10k checkpoint
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home/nepali-tts
P=/home/venv/bin/python
C=/home/ema/chars_model_10000.pt
$P tools/eval_bench.py baselines
$P tools/eval_bench.py run chars10k $C
sw() { env "$@" $P tools/eval_heldout.py run "$NAME" $C 2>&1 | grep -E "CER|Error" | tail -2; }
NAME=chars10k_default sw NFE=32
NAME=sw_nfe64    sw NFE=64
NAME=sw_cfg1.5   sw CFG=1.5
NAME=sw_cfg2.5   sw CFG=2.5
NAME=sw_cfg3     sw CFG=3.0
NAME=sw_speed0.9 sw SPEED=0.9
NAME=sw_sway-0.5 sw SWAY=-0.5
$P tools/eval_bench.py table
echo EVALQ_DONE
