"""Evaluation harness for the Nepali zero-shot cloning TTS.

Axes (aligned with NepTTS-Bench findings):
  1. Intelligibility -> CER with MMS-1b-all (Nepali 'npi'), the open ASR that
     tracks human MOS best on NepTTS-Bench (rho 0.58), plus round-trip Kriti CER.
     Normalization mirrors neptts-eval (NFC, strip punctuation, digits->ASCII).
  2. NepTTS-Bench CLI report (SCOREQ, NepaliMOS, Whisper CER). NepaliMOS ranks
     unseen systems poorly and public weights are off-scale: report, don't decide.
  3. Cloning fidelity -> ECAPA speaker similarity at 16 kHz, each output vs its
     own prompt speaker (held-out speakers only). The bench has no cloning test.
Final "best" claims need a human CMOS vs Asmita / ElevenLabs v3 / Piper.

Usage (GPU box):
    python evaluate.py --wav_dir out/ --manifest eval.jsonl --system_name ours-v0
    eval.jsonl rows: {"audio": gen.wav, "text": "...", "ref_audio": prompt.wav}
"""
import argparse
import json
import os
import re
import subprocess
import unicodedata

_DEV = str.maketrans("०१२३४५६७८९", "0123456789")


def cer_normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_DEV).lower()
    s = re.sub(r"[^\w\sऀ-ॿ]", "", s)
    s = s.replace("।", "").replace("॥", "")
    return re.sub(r"\s+", " ", s).strip()


def load_16k(path):
    import torchaudio
    sig, sr = torchaudio.load(path)
    sig = sig.mean(0, keepdim=True)
    if sr != 16000:
        sig = torchaudio.functional.resample(sig, sr, 16000)
    return sig


def neptts_bench(wav_dir, system_name, out_json):
    """Official CLI. Files must be named sent_NNN.wav matching sentences.json IDs."""
    cmd = ["neptts-eval", "--wav-dir", wav_dir, "--system-name", system_name, "-o", out_json]
    try:
        subprocess.run(cmd, check=True)
        with open(out_json, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("neptts-eval failed:", e)
        return None


def mms_cer(rows):
    import jiwer
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                           ignore_mismatched_sizes=True).eval()
    hyps, refs = [], []
    for r in rows:
        wav = load_16k(r["audio"]).squeeze(0).numpy()
        inp = proc(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = model(**inp).logits.argmax(-1)[0]
        hyps.append(cer_normalize(proc.decode(ids)))
        refs.append(cer_normalize(r["text"]))
    return {"mms_cer": jiwer.cer(refs, hyps)}


def kriti_cer(rows):
    """Kriti ships as a .nemo with a custom danda head, so restore the file
    directly instead of ASRModel.from_pretrained."""
    import jiwer
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import HfApi, hf_hub_download
    repo = "harrrshall/kriti"
    nemo_file = next(f for f in HfApi().list_repo_files(repo) if f.endswith(".nemo"))
    asr = nemo_asr.models.ASRModel.restore_from(hf_hub_download(repo, nemo_file))
    out = asr.transcribe([r["audio"] for r in rows])
    out = out[0] if isinstance(out, tuple) else out
    hyps = [cer_normalize(h if isinstance(h, str) else h.text) for h in out]
    refs = [cer_normalize(r["text"]) for r in rows]
    return {"kriti_cer": jiwer.cer(refs, hyps), "kriti_wer": jiwer.wer(refs, hyps)}


def speaker_similarity(rows):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def emb(p):
        return enc.encode_batch(load_16k(p)).squeeze()

    sims = [torch.nn.functional.cosine_similarity(emb(r["ref_audio"]), emb(r["audio"]), dim=0).item()
            for r in rows if r.get("ref_audio")]
    return {"spk_sim_ecapa": sum(sims) / len(sims) if sims else None, "n_pairs": len(sims)}


def run(a):
    report = {"system": a.system_name}
    if a.bench_wav_dir:
        report["neptts_bench"] = neptts_bench(a.bench_wav_dir, a.system_name,
                                              os.path.join(a.bench_wav_dir, "neptts_report.json"))
    if a.manifest:
        with open(a.manifest, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for fn in (mms_cer, kriti_cer, speaker_similarity):
            try:
                report.update(fn(rows))
            except Exception as e:
                print(f"{fn.__name__} failed:", e)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="jsonl of generated clips with text + ref_audio")
    ap.add_argument("--bench_wav_dir", help="dir of sent_NNN.wav for neptts-eval")
    ap.add_argument("--system_name", default="naamche-tts")
    run(ap.parse_args())
