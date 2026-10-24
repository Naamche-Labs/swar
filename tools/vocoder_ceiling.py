"""How much naturalness does the vocoder cost? Copy-synthesis on real human recordings:
original vs Vocos(mel(x)) (what F5 uses) vs BigVGAN-v2 24 kHz 100-band (its own mel, same
24 kHz / 100 bands / hop 256 as F5). If the vocoder alone loses a lot, a better vocoder is
a one-time gain for every voice. Scores: UTMOS, and NepaliMOS via neptts-eval if available.
Usage: python tools/vocoder_ceiling.py OUT_DIR [N]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(out, n=80):
    import librosa
    import soundfile as sf
    import torch
    from mos import utmos
    items = json.load(open("/home/data/rasa_test/items.json"))[: int(n)]
    os.makedirs(f"{out}/orig", exist_ok=True)
    os.makedirs(f"{out}/vocos", exist_ok=True)
    os.makedirs(f"{out}/bigvgan", exist_ok=True)
    from vocos import Vocos
    from f5_tts.model.modules import MelSpec
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    mel = MelSpec(mel_spec_type="vocos").cuda()
    bv = None
    try:
        import bigvgan
        bv = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False).cuda().eval()
        bv.remove_weight_norm()
        from meldataset import get_mel_spectrogram
    except Exception as e:
        print("bigvgan unavailable:", type(e).__name__, str(e)[:120])
    res = {"orig": [], "vocos": [], "bigvgan": []}
    for it in items:
        w, _ = librosa.load(it["real_wav"], sr=24000)
        x = torch.tensor(w).unsqueeze(0).cuda()
        sf.write(f"{out}/orig/{it['id']}.wav", w, 24000)
        with torch.no_grad():
            y = voc.decode(mel(x)).squeeze().cpu().numpy()
        sf.write(f"{out}/vocos/{it['id']}.wav", y, 24000)
        if bv is not None:
            with torch.no_grad():
                yb = bv(get_mel_spectrogram(x, bv.h).cuda()).squeeze().cpu().numpy()
            sf.write(f"{out}/bigvgan/{it['id']}.wav", yb, 24000)
        for k in res:
            p = f"{out}/{k}/{it['id']}.wav"
            if os.path.exists(p):
                res[k].append(utmos(librosa.load(p, sr=16000)[0]))
    for k, v in res.items():
        if v:
            print(f"VOC {k:8s} UTMOS {sum(v) / len(v):.3f} (n={len(v)})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
