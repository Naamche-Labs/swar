#!/bin/bash
# waits for the GPU (preset v2 scoring + prompt search), then: bilingual stage -> dedicated voices v3 from it
while tmux has-session -t chain 2>/dev/null || tmux has-session -t fprompt 2>/dev/null; do sleep 60; done
bash /home/nepali-tts/ops/train_bilingual.sh > /home/bi_train.log 2>&1
grep -q "TRAIN_EXIT 0" /home/bi_train.log || { echo BI_FAILED; exit 1; }
python3 - <<'P'
import torch
c = torch.load("/home/ckpts/nepali_bi/model_last.pt", map_location="cpu", weights_only=False)
torch.save({"ema_model_state_dict": c["ema_model_state_dict"]}, "/home/models/nepali_bi_ema.pt")
P
# voices v3: each studio speaker + ~15% English so the voice keeps it
for V in Female Male; do
  L=$(echo $V | tr A-Z a-z)
  mv /home/ckpts/preset_$L /home/ckpts/preset2_$L 2>/dev/null
  PRETRAIN=/home/models/nepali_bi_ema.pt EXTRA_EN=1500 bash /home/nepali-tts/ops/train_preset2.sh $V > /home/preset3_$L.log 2>&1
done
echo BI_CHAIN_DONE
