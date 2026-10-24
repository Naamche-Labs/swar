"""Automatic naturalness score (UTMOS22 strong, 1-5) so data or recipe changes that
hurt audio quality (e.g. 16 kHz training audio) show up next to CER and similarity.
Language-agnostic acoustic quality, not a substitute for Nepali listeners."""
import os

_model = None
DEV = os.environ.get("EVAL_DEVICE", "cuda")


def utmos(wav16k):
    """wav16k: 1-D float numpy array at 16 kHz -> predicted MOS."""
    global _model
    import torch
    if _model is None:
        _model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    with torch.no_grad():
        return float(_model(torch.tensor(wav16k).unsqueeze(0).to(DEV), 16000))
