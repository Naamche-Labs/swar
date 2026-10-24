#!/bin/bash
# shipper.sh RUN: every 5 min, send any new model_N.pt of ckpts/RUN to the eval box as EMA-only weights
R=$1
while true; do
  for f in /home/ckpts/$R/model_[0-9]*.pt; do
    [ -e "$f" ] || continue
    S=$(basename $f .pt | cut -d_ -f2)
    [ -f /home/ema/${R}_$S.pt ] || bash /home/ship_ema.sh $R $S
  done
  sleep 300
done
