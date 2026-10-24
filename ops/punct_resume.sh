#!/bin/bash
# resume the punct stage after the disk-full crash (model_last.pt = update 2000 is intact), then score it
source /home/venv/bin/activate; export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
bash /home/punct/train_punct.sh >> /home/punct/train.log 2>&1
cd /home/nepali-tts
export REF=female_news SHORT_CHARS=15 SHORT_SPEED=0.5 NFE=16
for k in 4000 last; do
  C=/home/ckpts/nepali_punct/model_$k.pt
  [ -f $C ] || continue
  python tools/eval_bench.py run punct_$k $C && python tools/rescore_deva.py bench punct_$k
  python tools/question_rise.py v_accent8k_fe punct_$k
  python tools/eval_fleurs.py run punct_$k $C
  python tools/eval_codeswitch.py punct_$k $C heldout
done
echo PUNCT_CHAIN_DONE
