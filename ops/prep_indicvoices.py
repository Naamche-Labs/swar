"""IndicVoices Nepali (raw, CC-BY 4.0): 463 h, 1,076 speakers, 16 kHz.

Kept: clips 3-20 s, SNR >= 15 dB (prepare_data.preprocess_clip), no annotation tags
in the unsanitized transcript (any <noise>/<uhh>/... means audio the text does not
cover), not a held-out speaker, not an eval sentence, and not already in
IndicVoices-R (same speaker + same normalized text; IVR is the restored copy).
Text = verbatim (what was said). Each clip also gets `hf_ratio`, the share of
energy above 4 kHz, so narrowband (phone) audio can be filtered later.
Writes manifest_iv.jsonl. CPU only.
"""
import glob
import io
import json
import re
import sys

sys.path.insert(0, "/home/nepali-tts")
sys.path.insert(0, "/home/nepali-tts/pipeline")
import prepare_data as P  # noqa: E402
from frontend import normalize  # noqa: E402

D = "/home/data/nepali_tts"
CLIPS = f"{D}/wav/indicvoices"
SHARDS = sorted(glob.glob("/home/hf_cache/hub/datasets--ai4bharat--IndicVoices/snapshots/*/nepali/train-*.parquet"))
IVR = sorted(glob.glob("/home/hf_cache/hub/datasets--ai4bharat--indicvoices_r/snapshots/*/Nepali/*.parquet"))
HELD = {s.split(":", 1)[1] for s in json.load(open("/home/data/ref/holdout_speakers.json"))}
EVAL_TEXTS = {it["text"] for it in json.load(open("/home/data/eval/items.json"))}


def hf_ratio(wav, sr):
    import numpy as np
    spec = np.abs(np.fft.rfft(wav)) ** 2
    f = np.fft.rfftfreq(len(wav), 1 / sr)
    return float(spec[f > 4000].sum() / max(spec.sum(), 1e-12))


def process(row, idx, ivr_keys):
    import soundfile as sf
    rec = {"keep": False, "wav_path": "", "text": "", "speaker": "", "dur": 0.0, "snr": 0.0, "hf": 0.0, "scenario": ""}
    spk = row["speaker_id"]
    if spk in HELD or re.search(r"<[^>]+>", row["unsanitized_verbatim"] or "") or not row["verbatim"]:
        return rec
    if (spk, normalize(row["normalized"] or "")) in ivr_keys:
        return rec
    text = normalize(row["verbatim"].strip())
    if text in EVAL_TEXTS:
        return rec
    try:
        arr, sr = sf.read(io.BytesIO(row["audio_filepath"]["bytes"]), dtype="float32", always_2d=False)
    except Exception:
        return rec
    pc = P.preprocess_clip(arr.T if arr.ndim > 1 else arr, sr)
    if pc is None:
        return rec
    wav, dur, snr = pc
    path = f"{CLIPS}/iv_{idx:07d}.wav"
    sf.write(path, wav, P.TARGET_SR)
    rec.update(keep=True, wav_path=path, text=text, speaker=f"indicvoices_r:{spk}", dur=dur, snr=snr,
               hf=hf_ratio(arr if arr.ndim == 1 else arr.mean(1), sr), scenario=row["scenario"])
    return rec


if __name__ == "__main__":
    import os
    import pyarrow.parquet as pq
    from datasets import Audio, load_dataset
    os.makedirs(CLIPS, exist_ok=True)
    ivr_keys = set()
    for f in IVR:
        for r in pq.read_table(f, columns=["speaker_id", "normalized"]).to_pylist():
            ivr_keys.add((r["speaker_id"], normalize(r["normalized"] or "")))
    print("IVR keys", len(ivr_keys))
    ds = load_dataset("parquet", data_files=SHARDS, split="train", num_proc=20)
    ds = ds.cast_column("audio_filepath", Audio(decode=False))
    out = ds.map(process, with_indices=True, num_proc=20, remove_columns=ds.column_names,
                 fn_kwargs={"ivr_keys": ivr_keys})
    kept = out.filter(lambda k: k, input_columns="keep", num_proc=20)
    with open(f"{D}/manifest_iv.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({"audio": r["wav_path"], "text": r["text"], "speaker": r["speaker"],
                                "source": "indicvoices", "dur": r["dur"], "snr": r["snr"], "hf": r["hf"],
                                "scenario": r["scenario"]}, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{len(ds)}, {sum(kept['dur']) / 3600:.1f} h, {len(set(kept['speaker']))} speakers")
