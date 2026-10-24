"""Which ASR should transcribe a user's prompt clip? CER on the 50 held-out prompt clips
(spontaneous IndicVoices-R speech) against their gold transcripts.
Usage: python tools/ref_asr_bench.py MODEL_ID [MODEL_ID ...]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_heldout import DEV, EVAL, ed, norm  # noqa: E402


def transcriber(mid):
    import torch
    from transformers import pipeline
    if "mms-1b" in mid:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        proc = AutoProcessor.from_pretrained(mid, target_lang="npi")
        m = Wav2Vec2ForCTC.from_pretrained(mid, target_lang="npi", ignore_mismatched_sizes=True).to(DEV).eval()

        def f(w):
            inp = {k: v.to(DEV) for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
            with torch.no_grad():
                return proc.decode(m(**inp).logits.argmax(-1)[0])
        return f
    if "indic-conformer" in mid:
        # ai4bharat/indic-conformer-600m-multilingual[:ctc|:rnnt], ONNX via its remote code
        from transformers import AutoModel
        mid, _, dec = mid.partition(":")
        m = AutoModel.from_pretrained(mid, trust_remote_code=True)
        return lambda w: m(torch.tensor(w).unsqueeze(0), "ne", dec or "rnnt")
    kw ={"generate_kwargs": {"language": "nepali", "task": "transcribe"}} if "whisper" in mid.lower() else {}
    p = pipeline("automatic-speech-recognition", model=mid, device=0, torch_dtype=torch.float16)
    return lambda w: p({"raw": w, "sampling_rate": 16000}, **kw)["text"]


if __name__ == "__main__":
    import librosa
    items = json.load(open(f"{EVAL}/items.json"))
    refs = {it["ref_wav"]: it["ref_text"] for it in items}
    for mid in sys.argv[1:]:
        try:
            f = transcriber(mid)
            E = N = 0
            for wav, text in refs.items():
                w, _ = librosa.load(wav, sr=16000)
                h, r = norm(f(w)), norm(text)
                E, N = E + ed(r, h), N + len(r)
            print(f"{mid:50s} prompt-transcript CER {100 * E / N:.2f}%  (n={len(refs)})", flush=True)
        except Exception as e:
            print(f"{mid:50s} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
