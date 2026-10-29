"""Automatic naturalness score (UTMOS22 strong, 1-5) so data or recipe changes that
hurt audio quality (e.g. 16 kHz training audio) show up next to CER and similarity.
Language-agnostic acoustic quality, not a substitute for Nepali listeners."""
import os

_model = None


def _auto_device():
    """cuda, else mps (Apple Silicon), else cpu; EVAL_DEVICE overrides."""
    if os.environ.get("EVAL_DEVICE"):
        return os.environ["EVAL_DEVICE"]
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


DEV = _auto_device()


def utmos(wav16k):
    """wav16k: 1-D float numpy array at 16 kHz -> predicted MOS."""
    global _model
    import torch
    if _model is None:
        _model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    with torch.no_grad():
        return float(_model(torch.tensor(wav16k).unsqueeze(0).to(DEV), 16000))
