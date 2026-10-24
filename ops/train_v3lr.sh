#!/bin/bash
# v3lr = v3 data with lr 5e-5 (IndicF5 paper's lr); otherwise identical to train_v3.sh
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
DS=${1:-nepali_v2chars}
cd /home
/home/venv/bin/accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no   /home/venv/lib/python3.10/site-packages/f5_tts/train/finetune_cli.py --exp_name F5TTS_Base --dataset_name $DS   --finetune --pretrain /home/models/IndicF5/pretrained_indicf5.safetensors   --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt   --learning_rate 5e-5 --batch_size_per_gpu 38400 --batch_size_type frame --max_samples 64   --num_warmup_updates 2000 --epochs 10 --save_per_updates 5000 --last_per_updates 1000   --keep_last_n_checkpoints 10 --logger tensorboard
echo TRAIN_EXIT $?
