#!/bin/bash
# v3 voices after the disk-full crash (2026-10-22): resume the female run, then the male, keeping one checkpoint and
# compacting each finished voice to an EMA file before the next starts; then score both against the real speakers.
set -o pipefail
ema() { /home/venv/bin/python -c "import torch; c=torch.load('$1', map_location='cpu', weights_only=False); torch.save({'ema_model_state_dict': c['ema_model_state_dict']}, '$2')"; }
for V in Female Male; do
  L=$(echo $V | tr A-Z a-z)
  PRETRAIN=/home/models/nepali_bi_ema.pt EXTRA_EN=1500 bash /home/nepali-tts/ops/train_preset2.sh $V > /home/preset3_$L.log 2>&1
  grep -q "TRAIN_EXIT 0" /home/preset3_$L.log || { echo "V3_FAILED $L"; exit 1; }
  ema /home/ckpts/preset_$L/model_last.pt /home/models/preset3_$L.pt && rm -rf /home/ckpts/preset_$L
  df -h /home | tail -1
done
source /home/venv/bin/activate; export HF_HOME=/home/hf_cache; cd /home/nepali-tts
for v in female male; do
  PRESET_PACE=1 NFE=16 python tools/eval_rasa.py run preset3_$v /home/models/preset3_$v.pt
done
python tools/match_test.py accent8k_pace preset2_female preset2_male preset3_female preset3_male
echo V3_DONE
