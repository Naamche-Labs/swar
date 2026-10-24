#!/bin/bash
# after the female preset: train the male one, then score both on the clone-vs-real test (pace-calibrated presets)
while ! grep -qa TRAIN_EXIT /home/preset_female.log; do sleep 60; done
bash /home/nepali-tts/ops/train_preset.sh Male > /home/preset_male.log 2>&1
source /home/venv/bin/activate; export HF_HOME=/home/hf_cache; cd /home/nepali-tts
for v in female male; do
  PRESET_PACE=1 NFE=16 python tools/eval_rasa.py run preset_$v /home/ckpts/preset_$v/model_last.pt
done
python tools/match_test.py accent8k_pace preset_female preset_male
echo PRESET_CHAIN_DONE
