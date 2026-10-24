#!/bin/bash
# after both v2 preset trainings: score them on the clone-vs-real Rasa test (pace-calibrated presets, NFE 16)
while ! grep -qa TRAIN_EXIT /home/preset2_male.log 2>/dev/null; do sleep 60; done
source /home/venv/bin/activate; export HF_HOME=/home/hf_cache; cd /home/nepali-tts
PRESET_PACE=1 NFE=16 python tools/eval_rasa.py run accent8k_pace /home/models/nepali_accent_8000.pt
for v in female male; do
  PRESET_PACE=1 NFE=16 python tools/eval_rasa.py run preset2_$v /home/ckpts/preset_$v/model_last.pt
done
python tools/match_test.py accent8k_pace preset2_female preset2_male
echo PRESET2_CHAIN_DONE
