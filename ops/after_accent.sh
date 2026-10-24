#!/bin/bash
# box 1, after the accent stage: score its checkpoints on every test, then mine hard words with the winner.
source /home/venv/bin/activate; export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1; cd /home/nepali-tts
while ! grep -qa TRAIN_EXIT /home/accent/train.log; do sleep 60; done
C=/home/ckpts/nepali_accent
for k in 4000 8000 last; do
  [ -f $C/model_$k.pt ] || continue
  python tools/eval_fleurs.py run accent_$k $C/model_$k.pt
  REF=female_news SHORT_CHARS=15 SHORT_SPEED=0.5 python tools/eval_bench.py run accent_$k $C/model_$k.pt
  python tools/rescore_deva.py bench accent_$k
  python tools/eval_rasa.py run accent_$k $C/model_$k.pt
  python tools/rescore_deva.py rasa accent_$k
done
python tools/eval_fleurs.py run polish4k_b1 /home/models/nepali_polish_4000.pt
echo AFTER_ACCENT_EVAL_DONE
