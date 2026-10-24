# Make BigVGAN train on the mel our F5 model predicts (Vocos-style: torchaudio HTK filterbank,
# magnitude, centered frames, log clamp 1e-7) as GENERATOR INPUT; losses stay on BigVGAN's own mel.
p = "/home/BigVGAN/meldataset.py"
s = open(p).read()
old = """                mel = mel_spectrogram(
                    audio,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )  # [B(1), self.num_mels, self.segment_size // self.hop_size]
"""
new = """                mel = vocos_mel(audio)[:, :, : audio.shape[1] // self.hop_size]  # F5/Vocos mel, centered frames
"""
assert old in s
s = s.replace(old, new)
s = s.replace("class MelDataset(torch.utils.data.Dataset):", '''_VOCOS = {}


def vocos_mel(audio):
    """The mel F5-TTS predicts (f5_tts MelSpec mel_spec_type='vocos'): (1, T) -> (1, 100, T//256 + 1)."""
    import torchaudio
    if "m" not in _VOCOS:
        _VOCOS["m"] = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256,
                                                          n_mels=100, power=1, center=True, normalized=False, norm=None)
    return torch.log(torch.clamp(_VOCOS["m"](audio), min=1e-7))


class MelDataset(torch.utils.data.Dataset):''', 1)
open(p, "w").write(s)
print("patched")
