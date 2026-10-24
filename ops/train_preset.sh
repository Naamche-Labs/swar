#!/bin/bash
# Preset voice stage: accent_8000 fine-tuned on ONE studio speaker (Rasa Female or Male, all styles),
# so the flagship presets match their real speaker; one-time cost per preset, users still clone zero-shot.
# Usage: bash ops/train_preset.sh Female|Male
V=$1; L=$(echo $V | tr A-Z a-z)
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
source /home/venv/bin/activate
mkdir -p /home/preset && python - <<P
import json, re
V = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))
rows = []
for line in open("/home/data/nepali_tts/manifest.train.jsonl", encoding="utf-8"):
    x = json.loads(line)
    if x["speaker"] == "rasa:$V":
        t = x["text"].replace('"', "").replace("|", " ").strip()
        if t and all(c in V for c in t):
            rows.append(x["audio"] + "|" + t)
open("/home/preset/$L.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(rows) + "\n")
print(len(rows), "rows")
P
python /home/venv/lib/python3.10/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py /home/preset/$L.csv /home/f5data/preset_${L}_custom --pretrain --workers 16
cp /home/models/IndicF5/checkpoints/vocab.txt /home/f5data/preset_${L}_custom/vocab.txt
cd /home
/home/venv/bin/accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no /home/venv/lib/python3.10/site-packages/f5_tts/train/finetune_cli.py --exp_name F5TTS_Base --dataset_name preset_$L \
  --finetune --pretrain /home/models/accent_8000.safetensors --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt \
  --learning_rate 1e-5 --batch_size_per_gpu 38400 --batch_size_type frame --max_samples 64 --num_warmup_updates 200 --epochs 12 \
  --save_per_updates 1000 --last_per_updates 500 --keep_last_n_checkpoints 3 --logger tensorboard
echo TRAIN_EXIT $?
