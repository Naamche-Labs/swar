"""Vocos-mel (what F5 predicts) -> BigVGAN-v2 24 kHz mel, so F5 can use the better vocoder
with no retraining. Both are 24 kHz / 1024 FFT / hop 256 / 100 bands; they differ in the mel
filterbank (torchaudio HTK, no norm vs librosa Slaney) and log floor. Two training-free bridges:
  pinv:  magnitude spectrum = pinv(M_vocos) @ exp(mel), then BigVGAN's filterbank
  lstsq: 100x100 least-squares map between the two linear mels, fit on 60 real clips
Copy-synthesis UTMOS on 40 held-out real Rasa clips decides (tools/vocoder_ceiling.py has the
first numbers: original 3.88, Vocos 3.14, BigVGAN with its own mel 3.80).
Usage (box, PYTHONPATH=/home/BigVGAN): python tools/mel_bridge.py"""
import json

import librosa
import torch

from mos import utmos


def load_bigvgan():
    import bigvgan
    from env import AttrDict
    from huggingface_hub import hf_hub_download
    cfg = AttrDict(json.load(open(hf_hub_download("nvidia/bigvgan_v2_24khz_100band_256x", "config.json"))))
    bv = bigvgan.BigVGAN(cfg, use_cuda_kernel=False)
    sd = torch.load(hf_hub_download("nvidia/bigvgan_v2_24khz_100band_256x", "bigvgan_generator.pt"), map_location="cpu")
    bv.load_state_dict(sd["generator"])
    bv = bv.cuda().eval()
    bv.remove_weight_norm()
    return bv, cfg


def main():
    from librosa.filters import mel as lmel
    from meldataset import get_mel_spectrogram
    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    bv, cfg = load_bigvgan()
    Mv = voc.feature_extractor.mel_spec.mel_scale.fb.T.cuda()
    Mb = torch.tensor(lmel(sr=24000, n_fft=1024, n_mels=100, fmin=0, fmax=cfg.fmax), dtype=torch.float32).cuda()
    P = torch.linalg.pinv(Mv)

    def pinv(mv):
        return torch.log(torch.clamp(Mb @ torch.clamp(P @ torch.exp(mv[0]), min=0), min=1e-5)).unsqueeze(0)
    items = json.load(open("/home/data/rasa_test/items.json"))
    A, B = [], []
    for it in items[100:160]:
        w, _ = librosa.load(it["real_wav"], sr=24000)
        x = torch.tensor(w).unsqueeze(0).cuda()
        with torch.no_grad():
            a, b = torch.exp(voc.feature_extractor(x)[0]).T, torch.exp(get_mel_spectrogram(x, cfg).cuda()[0]).T
            n = min(len(a), len(b))  # vocos pads centre frames, BigVGAN does not
            A.append(a[:n])
            B.append(b[:n])
    W = torch.linalg.lstsq(torch.cat(A), torch.cat(B)).solution

    def lstsq(mv):
        return torch.log(torch.clamp(torch.exp(mv[0]).T @ W, min=1e-5)).T.unsqueeze(0)
    S = {"vocos": [], "bigvgan_own_mel": [], "bigvgan_pinv": [], "bigvgan_lstsq": []}
    for it in items[:40]:
        w, _ = librosa.load(it["real_wav"], sr=24000)
        x = torch.tensor(w).unsqueeze(0).cuda()
        with torch.no_grad():
            mv = voc.feature_extractor(x)
            outs = {"vocos": voc.decode(mv), "bigvgan_own_mel": bv(get_mel_spectrogram(x, cfg).cuda()),
                    "bigvgan_pinv": bv(pinv(mv)), "bigvgan_lstsq": bv(lstsq(mv))}
        for k, y in outs.items():
            S[k].append(utmos(librosa.resample(y.squeeze().cpu().numpy(), orig_sr=24000, target_sr=16000)))
    for k, v in S.items():
        print(f"R {k:16s} UTMOS {sum(v) / len(v):.3f} n={len(v)}", flush=True)
    torch.save({"W": W.cpu(), "P": P.cpu(), "Mb": Mb.cpu()}, "/home/models/mel_bridge.pt")


if __name__ == "__main__":
    main()
