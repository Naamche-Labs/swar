#!/bin/bash
# Build the cloner Space folder and push it: SPACE=user/name bash ops/deploy_space_clone.sh
# The model weights are not in the Space: they live in the private MODEL_REPO (see space_clone/README.md).
set -e
cd "$(dirname "$0")/.."
B=$(mktemp -d)
cp space_clone/{app.py,requirements.txt,README.md} "$B/"
mkdir -p "$B/pipeline" "$B/tools" "$B/data"
cp pipeline/{autoclone,clone,kriti_onnx}.py "$B/pipeline/"
cp tools/{mos,pron_cer,eval_heldout}.py "$B/tools/"
cp -r frontend "$B/"
cp data/{pron_fixes.json,loanwords.tsv,cmudict.dict.gz,cmudict.LICENSE,g2p_model.json.gz,ne_wordfreq.tsv.gz} "$B/data/"
cp -r data/lexicons "$B/data/"
find "$B" -name __pycache__ -prune -exec rm -rf {} +
python3 -c "
from huggingface_hub import HfApi; import sys
api = HfApi(); api.create_repo('$SPACE', repo_type='space', space_sdk='gradio', private=True, exist_ok=True)
api.upload_folder(repo_id='$SPACE', repo_type='space', folder_path='$B', commit_message='deploy')
print('deployed https://huggingface.co/spaces/$SPACE')"
