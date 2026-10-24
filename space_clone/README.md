---
title: Nepali Voice Cloning
emoji: 🗣️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---
Drop 20-30 s of one person talking, type Nepali, English or both, hear it in their voice. Runs on ZeroGPU (HF Pro).
Secrets: HF_TOKEN (read access to MODEL_REPO), MODEL_REPO (private model repo holding podvoices_ema.pt, vocab.txt,
kriti_onnx/, presets/ from ops/build_presets.py). Only clone a voice with the speaker's permission.
