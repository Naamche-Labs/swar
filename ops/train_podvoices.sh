#!/bin/bash
# Podcast voices stage (2026-10-23): nepali_bi_ema fine-tuned on the chosen podcast voices (ops/build_podvoices.py)
# plus FLEURS ne_np and LibriTTS-R replay, so one model carries the preset voices and keeps zero-shot cloning and
# English. Sized for one L4 (24 GB): 16k-frame batches. Usage: bash ops/train_podvoices.sh SET_DIR (LR, EPOCHS)
D=$1; DS=${DS:-podvoices}
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/venv/bin/activate
python /home/venv/lib/python3.10/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py $D/train.csv /home/f5data/${DS}_custom --pretrain --workers 16
cp /home/models/IndicF5/checkpoints/vocab.txt /home/f5data/${DS}_custom/vocab.txt
mkdir -p /home/data && ln -sfn /home/f5data /home/data/f5data 2>/dev/null
cd /home
/home/venv/bin/accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no /home/venv/lib/python3.10/site-packages/f5_tts/train/finetune_cli.py --exp_name F5TTS_Base --dataset_name $DS \
  --finetune --pretrain ${PRETRAIN:-/home/ema/nepali_bi_ema.pt} --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt \
  --learning_rate ${LR:-2e-5} --batch_size_per_gpu ${BATCH:-12000} --batch_size_type frame --max_samples 32 --grad_accumulation_steps ${ACCUM:-1} --num_warmup_updates 300 --epochs ${EPOCHS:-12} \
  --save_per_updates ${SAVE_EVERY:-1000} --last_per_updates 250 --keep_last_n_checkpoints 2 --logger tensorboard
echo TRAIN_EXIT $?
