#!/bin/bash
# accent = polish_4000 continued at low lr on Nepal-recorded speech (OpenSLR 54+43, 70.7 h) + Rasa studio (43 h).
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home
/home/venv/bin/accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no   /home/venv/lib/python3.10/site-packages/f5_tts/train/finetune_cli.py --exp_name F5TTS_Base --dataset_name nepali_accent   --finetune --pretrain /home/models/polish_4000.safetensors   --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt   --learning_rate 1e-5 --batch_size_per_gpu 38400 --batch_size_type frame --max_samples 64   --num_warmup_updates 300 --epochs 5 --save_per_updates 1000 --last_per_updates 500   --keep_last_n_checkpoints 10 --logger tensorboard
echo TRAIN_EXIT $?
