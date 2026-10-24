#!/bin/bash
# Preset voice stage: accent_8000 fine-tuned on ONE studio speaker (Rasa Female or Male, all styles),
# v2 (2026-10-22): the v1 pass (lr 1e-5, 12 epochs, ~2.6k updates) changed nothing; this is 3x the lr and 40 epochs.
# Usage: bash ops/train_preset2.sh Female|Male   (LR, EPOCHS, PRETRAIN override)
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
import os, random
n_en = int(os.environ.get("EXTRA_EN", "0"))  # keep English: a slice of the bilingual set's LibriTTS-R rows
if n_en and os.path.exists("/home/bi/bi.csv"):
    en = [l for l in open("/home/bi/bi.csv", encoding="utf-8").read().splitlines()[1:] if "/libritts_r/" in l]
    random.Random(0).shuffle(en)
    rows += en[:n_en]
open("/home/preset/$L.csv", "w", encoding="utf-8").write("audio_file|text\n" + "\n".join(rows) + "\n")
print(len(rows), "rows")
P
python /home/venv/lib/python3.10/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py /home/preset/$L.csv /home/f5data/preset_${L}_custom --pretrain --workers 16
cp /home/models/IndicF5/checkpoints/vocab.txt /home/f5data/preset_${L}_custom/vocab.txt
cd /home
/home/venv/bin/accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no /home/venv/lib/python3.10/site-packages/f5_tts/train/finetune_cli.py --exp_name F5TTS_Base --dataset_name preset_$L \
  --finetune --pretrain ${PRETRAIN:-/home/models/nepali_accent_8000.pt} --tokenizer custom --tokenizer_path /home/models/IndicF5/checkpoints/vocab.txt \
  --learning_rate ${LR:-3e-5} --batch_size_per_gpu 38400 --batch_size_type frame --max_samples 64 --num_warmup_updates 500 --epochs ${EPOCHS:-40} \
  --save_per_updates ${SAVE_EVERY:-4000} --last_per_updates 500 --keep_last_n_checkpoints ${KEEP:-1} --logger tensorboard
echo TRAIN_EXIT $?
