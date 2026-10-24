#!/bin/bash
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home
/home/venv/bin/accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no   /home/venv/lib/python3.10/site-packages/f5_tts/train/finetune_cli.py --exp_name F5TTS_Base --dataset_name nepali_respell   --finetune --pretrain /home/models/IndicF5/pretrained_indicf5.safetensors   --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt   --learning_rate 3e-5 --batch_size_per_gpu 38400 --batch_size_type frame --max_samples 64   --num_warmup_updates 2000 --epochs 18 --save_per_updates 2500 --last_per_updates 1000   --keep_last_n_checkpoints 6 --logger tensorboard
echo TRAIN_EXIT $?
