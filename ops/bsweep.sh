#!/bin/bash
# eval box 2: inference-knob sweep on NepTTS-Bench with the calm male_wiki prompt
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home/nepali-tts
C=/home/ema/chars_model_10000.pt
b() { env REF=male_wiki "$@" /home/venv/bin/python tools/eval_bench.py run "$NAME" $C 2>&1 | grep -E "CER|Error" | tail -2; }
NAME=c10k_mw_nfe64   b NFE=64
NAME=c10k_mw_cfg2.5  b CFG=2.5
NAME=c10k_mw_cfg1.5  b CFG=1.5
NAME=c10k_mw_speed0.9 b SPEED=0.9
NAME=c10k_mw_sway-0.5 b SWAY=-0.5
echo BSWEEP_DONE
