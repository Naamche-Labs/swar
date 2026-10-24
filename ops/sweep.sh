#!/bin/bash
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home/nepali-tts
C=/home/ckpts/nepali_chars/model_10000.pt
run() { env "$@" /home/venv/bin/python tools/eval_heldout.py run "$NAME" $C 2>&1 | grep -E "CER|Error|error" | tail -3; }
NAME=sw_nfe64      run NFE=64
NAME=sw_cfg1.5     run CFG=1.5
NAME=sw_cfg2.5     run CFG=2.5
NAME=sw_cfg3       run CFG=3.0
NAME=sw_speed0.9   run SPEED=0.9
NAME=sw_sway-0.5   run SWAY=-0.5
echo SWEEP_DONE
