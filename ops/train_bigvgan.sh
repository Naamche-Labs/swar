#!/bin/bash
# bigvgan stage: accent_8000 moved to BigVGAN-v2 mel targets (F5 supports both; the vocoder is
# nvidia/bigvgan_v2_24khz_100band_256x). Copy-synthesis of real clips: Vocos UTMOS 3.14, BigVGAN 3.80,
# original 3.88; no bridge from Vocos mel works (pinv 3.14, learned 2.82), so the model must
# predict the richer mel itself. Same data as accent (OpenSLR 54/43 + Rasa), lr 3e-5, warmup 500.
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
F=/home/venv/lib/python3.10/site-packages/f5_tts
sed 's/^mel_spec_type = "vocos"/mel_spec_type = "bigvgan"/' $F/train/finetune_cli.py > $F/train/finetune_cli_bigvgan.py
sed 's/mel_spec_type: vocos/mel_spec_type: bigvgan/; s/name: vocos/name: bigvgan/' $F/configs/F5TTS_Base.yaml > $F/configs/F5TTS_Base_bigvgan.yaml
[ -d /home/f5data/nepali_bigvgan_custom ] || cp -r /home/f5data/nepali_accent_custom /home/f5data/nepali_bigvgan_custom
python - <<'P'
import torch
from safetensors.torch import save_file
sd = torch.load("/home/ckpts/nepali_accent/model_8000.pt", map_location="cpu", weights_only=False)["ema_model_state_dict"]
save_file({k: v.contiguous() for k, v in sd.items()}, "/home/models/accent_8000.safetensors")
P
cd /home
/home/venv/bin/accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no $F/train/finetune_cli_bigvgan.py --exp_name F5TTS_Base --dataset_name nepali_bigvgan \
  --finetune --pretrain /home/models/accent_8000.safetensors --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt \
  --learning_rate 3e-5 --batch_size_per_gpu 30000 --batch_size_type frame --max_samples 64 --num_warmup_updates 500 --epochs 3 \
  --save_per_updates 1000 --last_per_updates 500 --keep_last_n_checkpoints 6 --logger tensorboard
echo TRAIN_EXIT $?
