#!/bin/bash
# ship_ema.sh RUN STEP EVALHOST: extract EMA weights of ckpts/RUN/model_STEP.pt and copy to the eval box
R=$1; S=$2; H=${3:-<gpu-box>}
/home/venv/bin/python -c "
import torch;c=torch.load('/home/ckpts/$R/model_$S.pt',map_location='cpu',weights_only=True)
torch.save({'ema_model_state_dict':c['ema_model_state_dict'],'update':c.get('update')},'/home/ema/${R}_$S.pt')"
scp -q -i /root/.ssh/id_box -o StrictHostKeyChecking=no /home/ema/${R}_$S.pt root@$H:/home/ema/ && echo "shipped ${R}_$S"
