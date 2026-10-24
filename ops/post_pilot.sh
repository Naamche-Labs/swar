#!/bin/bash
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home/nepali-tts
until grep -q TRAIN_EXIT /home/pilot.log; do sleep 60; done
python=/home/venv/bin/python
$python tools/eval_heldout.py run indicf5_base /home/models/IndicF5/pretrained_indicf5.safetensors
for c in $(ls /home/ckpts/nepali_chars/model_[0-9]*.pt | sort -V); do
  n=$(basename $c .pt)
  $python tools/eval_heldout.py run pilot_chars_$n $c
done
echo POST_DONE
