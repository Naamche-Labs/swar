"""Copy-synthesis only: real Rasa test recordings -> F5 (Vocos) mel -> Vocos vs fine-tuned BigVGAN checkpoints.
Usage (box 2, PYTHONPATH=/home/BigVGAN): python tools/vocoder_copy.py N G_CKPT [G_CKPT ...]"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import librosa
    import soundfile as sf
    from f5_tts.model.modules import MelSpec
    from vocos import Vocos
    from match_test import compare
    from mos import utmos
    from vocoder_eval import load_bigvgan
    n = int(sys.argv[1])
    items = [x for x in json.load(open("/home/data/rasa_test/items.json")) if x["style"] != "PROPER NOUN"][:n]
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    mel = MelSpec(mel_spec_type="vocos").cuda()
    gens = {os.path.basename(c): load_bigvgan(c) for c in sys.argv[2:]}
    S = {k: [] for k in ["real", "vocos", *gens]}
    T = {k: [] for k in ["vocos", *gens]}
    r16 = lambda a: librosa.resample(np.asarray(a, dtype="float32"), orig_sr=24000, target_sr=16000)
    for it in items:
        w, _ = librosa.load(it["real_wav"], sr=24000)
        with torch.no_grad():
            m = mel(torch.tensor(w).unsqueeze(0).cuda())
            outs = {"vocos": voc.decode(m).squeeze().cpu().numpy(), **{k: g(m).squeeze().cpu().numpy() for k, g in gens.items()}}
        S["real"].append(utmos(r16(w)))
        for k, y in outs.items():
            S[k].append(utmos(r16(y)))
            T[k].append(compare(r16(w), r16(y))[0])
            d = f"/home/data/rasa_test/runs/copy_{k}"
            os.makedirs(d, exist_ok=True)
            sf.write(f"{d}/{it['id']}.wav", y, 24000)
    for k in S:
        print(f"COPY {k:12s} UTMOS {np.mean(S[k]):.3f}" + (f"  timbre distance {np.mean(T[k]):.2f}" if k in T else ""), flush=True)
