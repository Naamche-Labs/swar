"""Score a fine-tuned BigVGAN (trained on F5/Vocos mels, ops/patch_bigvgan.py) against Vocos.

copy:  real Rasa test recordings -> F5's Vocos mel -> vocoder (the pipeline's ceiling)
tts:   accent_8000 generates the mel for each Rasa test sentence (studio preset prompts, pace
       calibration) -> decoded by Vocos and by the new vocoder from the SAME mel
Scores: UTMOS, and for copy also MFCC-DTW timbre distance to the real recording.
Writes wavs under /home/data/rasa_test/runs/{copy_bigvgan,tts_vocos,tts_bigvgan}_<tag>.
Usage (box 2, PYTHONPATH=/home/BigVGAN): python tools/vocoder_eval.py G_CHECKPOINT TAG [N]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
R = "/home/data/rasa_test"


def load_bigvgan(ckpt):
    import bigvgan
    from env import AttrDict
    h = AttrDict(json.load(open("/home/voc/config.json")))
    g = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    g.load_state_dict(torch.load(ckpt, map_location="cpu")["generator"])
    g = g.cuda().eval()
    g.remove_weight_norm()
    return g


def main(ckpt, tag, n=60):
    import librosa
    import soundfile as sf
    from f5_tts.api import F5TTS
    from f5_tts.model.modules import MelSpec
    from vocos import Vocos
    from clone import speed_for
    from frontend import normalize
    from match_test import compare
    from mos import utmos
    g = load_bigvgan(ckpt)
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    mel = MelSpec(mel_spec_type="vocos").cuda()
    items = [x for x in json.load(open(f"{R}/items.json")) if x["style"] != "PROPER NOUN"][: int(n)]
    refs = json.load(open("/home/data/ref/bench/refs.json"))
    preset = {"Female": refs["female_news"], "Male": refs["male_wiki"]}
    tts = F5TTS(model="F5TTS_Base", ckpt_file="/home/models/nepali_accent_8000.pt",
                vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
    tts.ema_model.float()
    dirs = {k: f"{R}/runs/{k}_{tag}" for k in ("copy_bigvgan", "tts_vocos", "tts_bigvgan")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    S = {k: [] for k in ("real", "copy_vocos", "copy_bigvgan", "tts_vocos", "tts_bigvgan")}
    T = {"copy_vocos": [], "copy_bigvgan": []}
    for it in items:
        w, _ = librosa.load(it["real_wav"], sr=24000)
        x = torch.tensor(w).unsqueeze(0).cuda()
        with torch.no_grad():
            m = mel(x)
            yv = voc.decode(m).squeeze().cpu().numpy()
            yb = g(m).squeeze().cpu().numpy()
        sf.write(f"{dirs['copy_bigvgan']}/{it['id']}.wav", yb, 24000)
        r = preset[it["voice"]]
        gen = normalize(it["text"])
        _, _, spect = tts.infer(ref_file=r["audio"], ref_text=r["text"], gen_text=gen, nfe_step=16, seed=0,
                                speed=speed_for(gen, r.get("speed", 1.0)), remove_silence=False)
        sm = torch.tensor(spect).unsqueeze(0).float().cuda()
        with torch.no_grad():
            tv = voc.decode(sm).squeeze().cpu().numpy()
            tb = g(sm).squeeze().cpu().numpy()
        sf.write(f"{dirs['tts_vocos']}/{it['id']}.wav", tv, 24000)
        sf.write(f"{dirs['tts_bigvgan']}/{it['id']}.wav", tb, 24000)
        r16 = lambda a: librosa.resample(np.asarray(a, dtype="float32"), orig_sr=24000, target_sr=16000)
        for k, a in (("real", w), ("copy_vocos", yv), ("copy_bigvgan", yb), ("tts_vocos", tv), ("tts_bigvgan", tb)):
            S[k].append(utmos(r16(a)))
        T["copy_vocos"].append(compare(r16(w), r16(yv))[0])
        T["copy_bigvgan"].append(compare(r16(w), r16(yb))[0])
    for k, v in S.items():
        extra = f"  timbre distance {np.mean(T[k]):.2f}" if k in T else ""
        print(f"VOC {tag} {k:13s} UTMOS {np.mean(v):.3f}{extra}  (n={len(v)})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
