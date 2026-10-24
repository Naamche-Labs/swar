#!/bin/bash
# after the v3 voices (bilingual base + their studio speaker + 1,500 English clips): clone-vs-real test vs v2
while ! grep -q BI_CHAIN_DONE /home/bi_chain.log 2>/dev/null; do sleep 60; done
source /home/venv/bin/activate; export HF_HOME=/home/hf_cache; cd /home/nepali-tts
for v in female male; do
  PRESET_PACE=1 NFE=16 python tools/eval_rasa.py run preset3_$v /home/ckpts/preset_$v/model_last.pt
done
python tools/match_test.py accent8k_pace preset2_female preset2_male preset3_female preset3_male
for v in female male; do
  python3 -c "import torch; c=torch.load('/home/ckpts/preset_$v/model_last.pt', map_location='cpu', weights_only=False); torch.save({'ema_model_state_dict': c['ema_model_state_dict']}, '/home/models/preset3_$v.pt')"
done
echo PRESET3_EVAL_DONE
