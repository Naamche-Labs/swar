#!/bin/bash
# bilingual stage: accent_8000 + English (LibriTTS-R) + Nepal-recorded Nepali, then the dedicated voices again from it
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
source /home/venv/bin/activate
cd /home/nepali-tts && python ops/build_bilingual.py
python /home/venv/lib/python3.10/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py /home/bi/bi.csv /home/f5data/nepali_bi_custom --pretrain --workers 16
cp /home/models/IndicF5/checkpoints/vocab.txt /home/f5data/nepali_bi_custom/vocab.txt
cd /home
accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no /home/venv/lib/python3.10/site-packages/f5_tts/train/finetune_cli.py --exp_name F5TTS_Base --dataset_name nepali_bi \
  --finetune --pretrain /home/models/nepali_accent_8000.pt --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt \
  --learning_rate 2e-5 --batch_size_per_gpu 38400 --batch_size_type frame --max_samples 64 --num_warmup_updates 500 --epochs 6 \
  --save_per_updates 2000 --last_per_updates 500 --keep_last_n_checkpoints 2 --logger tensorboard
echo TRAIN_EXIT $?
