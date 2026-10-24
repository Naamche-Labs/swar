"""Clean a voice prompt before cloning. F5 copies the prompt's room and noise along with
the voice, so a noisy phone clip gives a noisy clone.

enhance(path, method) -> path of a cached enhanced 24 kHz wav next to /home/data/enh.
Methods: dfn (DeepFilterNet3, full band, denoise only); dfn6/12/18 cap the noise removed at 6/12/18 dB.
Usage: python tools/enhance.py METHOD WAV [WAV ...]
"""
import hashlib
import os
import sys

CACHE = "/home/data/enh"
_model = {}


def _dfn(path, out, atten=None):
    import librosa
    import soundfile as sf
    import types
    import torch
    # deepfilternet imports torchaudio.backend.common, which torchaudio 2.9+ removed
    if "torchaudio.backend.common" not in sys.modules:
        stub = types.ModuleType("torchaudio.backend.common")
        stub.AudioMetaData = object
        sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend.common"] = stub
    from df.enhance import enhance as df_enhance, init_df
    if "dfn" not in _model:
        _model["dfn"] = init_df()
    m, st, _ = _model["dfn"]
    w, _ = librosa.load(path, sr=st.sr())
    y = df_enhance(m, st, torch.tensor(w).unsqueeze(0), atten_lim_db=atten).squeeze(0).numpy()
    sf.write(out, librosa.resample(y, orig_sr=st.sr(), target_sr=24000), 24000)


METHODS = {"dfn": _dfn,  # dfnN: at most N dB of noise removed (a lighter clean that keeps more of the voice)
           "dfn6": lambda p, o: _dfn(p, o, 6), "dfn12": lambda p, o: _dfn(p, o, 12), "dfn18": lambda p, o: _dfn(p, o, 18)}


def enhance(path, method):
    if not method:
        return path
    os.makedirs(f"{CACHE}/{method}", exist_ok=True)
    out = f"{CACHE}/{method}/{hashlib.md5(path.encode()).hexdigest()[:12]}.wav"
    if not os.path.exists(out):
        METHODS[method](path, out)
    return out


if __name__ == "__main__":
    for p in sys.argv[2:]:
        print(enhance(p, sys.argv[1]))
