#!/bin/bash
# Put the cloner app next to the weights in one Hugging Face model repo, so a Colab notebook (or any GPU machine) needs
# only that repo: REPO/app/ = space_clone/app.py + pipeline/, frontend/, tools/, data/ (as ops/deploy_space_clone.sh),
# requirements_colab.txt (no torch pin: Colab's torch is used). Weights, vocab, Kriti ONNX and presets/ are uploaded
# separately (podvoices_ema.pt from a box; presets from ops/build_presets_synth.py, model-made audio only).
# Usage: REPO=Aarjan/swar bash ops/release_bundle.sh
set -e
cd "$(dirname "$0")/.."
B=$(mktemp -d)
cp space_clone/app.py "$B/"
mkdir -p "$B/pipeline" "$B/tools" "$B/data"
cp pipeline/{autoclone,clone,kriti_onnx}.py "$B/pipeline/"
cp tools/{mos,pron_cer,eval_heldout}.py "$B/tools/"
cp -r frontend "$B/"
cp data/{pron_fixes.json,loanwords.tsv,cmudict.dict.gz,cmudict.LICENSE,g2p_model.json.gz,ne_wordfreq.tsv.gz} "$B/data/"
cp -r data/lexicons "$B/data/"
# the versions the model was built and evaluated with (box venv, 2026-10-24); gradio comes with f5-tts
printf '%s\n' "f5-tts==1.1.22" "speechbrain==1.1.2" "numpy<2" onnxruntime-gpu librosa soundfile scipy transformers huggingface_hub \
  > "$B/requirements_colab.txt"
find "$B" -name __pycache__ -prune -exec rm -rf {} +
python3 -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='$REPO', folder_path='$B', path_in_repo='app', commit_message='cloner app bundle')
print('uploaded app/ to $REPO')"
