"""Match-page clips with the fine-tuned BigVGAN: the real recording through it (bigvgan_copy) and the
accent_8000 clone decoded by it (clone_bigvgan), pace-calibrated presets, Nepal pronunciation on.
Usage (box 2, PYTHONPATH=/home/BigVGAN): python ops/gen_match_bigvgan.py G_CKPT PICK.json OUT_DIR"""
import json
import os
import sys

import torch

sys.path[:0] = ["tools", "pipeline", "."]
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
from f5_tts.api import F5TTS  # noqa: E402
from f5_tts.model.modules import MelSpec  # noqa: E402
from clone import speed_for, to_model_text  # noqa: E402
from frontend import normalize  # noqa: E402
from vocoder_eval import load_bigvgan  # noqa: E402

g = load_bigvgan(sys.argv[1])
pick, out = json.load(open(sys.argv[2])), sys.argv[3]
os.makedirs(out, exist_ok=True)
mel = MelSpec(mel_spec_type="vocos").cuda()
refs = json.load(open("/home/data/ref/bench/refs.json"))
preset = {"Female": refs["female_news"], "Male": refs["male_wiki"]}
tts = F5TTS(model="F5TTS_Base", ckpt_file="/home/models/nepali_accent_8000.pt", vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
tts.ema_model.float()
for it in pick:
    w, _ = librosa.load(it["real_wav"], sr=24000)
    with torch.no_grad():
        sf.write(f"{out}/bigvgan_copy_{it['id']}.wav", g(mel(torch.tensor(w).unsqueeze(0).cuda())).squeeze().cpu().numpy(), 24000)
    r = preset[it["voice"]]
    gen = to_model_text(it["text"], "chars")
    _, _, spect = tts.infer(ref_file=r["audio"], ref_text=normalize(r["text"]), gen_text=gen, nfe_step=16, seed=0,
                            speed=speed_for(gen, r.get("speed", 1.0)), remove_silence=False)
    with torch.no_grad():
        y = g(torch.tensor(spect).unsqueeze(0).float().cuda()).squeeze().cpu().numpy()
    sf.write(f"{out}/clone_bigvgan_{it['id']}.wav", y, 24000)
print("MATCH_BV_DONE")
