#!/bin/bash
# box 1: wait for punctuation restoration, build data, train the punct stage, then score it.
source /home/venv/bin/activate; export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
until [ $(grep -l "^DONE" /home/punct/log_* 2>/dev/null | wc -l) -eq 4 ]; do sleep 60; done
python /home/punct/build_punct.py
python /home/venv/lib/python3.10/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py /home/punct/train.csv /home/f5data/nepali_punct_custom --pretrain --workers 24
cp /home/models/IndicF5/checkpoints/vocab.txt /home/f5data/nepali_punct_custom/vocab.txt
bash /home/punct/train_punct.sh > /home/punct/train.log 2>&1
cd /home/nepali-tts
export REF=female_news SHORT_CHARS=15 SHORT_SPEED=0.5 NFE=16
for k in 3000 last; do
  C=/home/ckpts/nepali_punct/model_$k.pt
  python tools/eval_bench.py run punct_$k $C && python tools/rescore_deva.py bench punct_$k
  python tools/question_rise.py v_accent8k_fe punct_$k
  python tools/eval_fleurs.py run punct_$k $C
  python tools/eval_codeswitch.py punct_$k $C heldout
done
echo PUNCT_CHAIN_DONE
