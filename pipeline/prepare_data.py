"""Download + preprocess Nepali speech into a zero-shot TTS training manifest.

Runs on the GPU box (never locally). Resamples to 24 kHz mono, trims silence,
keeps 3-12 s clips (F5 sweet spot), drops noisy clips by an energy-based SNR
estimate, keeps real speaker IDs (cloning needs them), and writes both raw
text and our G2P phonemes so text-vs-phoneme input can be A/B tested.

CPU-only (no GPU needed). Runs one worker per core. Disk: ~85 GB HF cache for
IndicVoices-R + Rasa Nepali plus ~30 GB of 24 kHz wavs.

Usage:
    python prepare_data.py --out /data/nepali_tts --sources indicvoices_r rasa fleurs [--num_proc 32]
Outputs: manifest.jsonl, and metadata_chars.csv / metadata_phonemes.csv in the
`audio_file|text` format F5's prepare_csv_wavs.py expects (phonemes encoded one
symbol per phoneme via encode_for_f5, since F5 tokenizes per character).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend import encode_for_f5, g2p, normalize

# key -> (hf_id, {"data_dir" or "name": value}, grade)
SOURCES = {
    "indicvoices_r": ("ai4bharat/indicvoices_r", {"data_dir": "Nepali"}, "tts"),
    "rasa":          ("ai4bharat/Rasa",          {"data_dir": "Nepali"}, "tts"),
    "indicvoices":   ("ai4bharat/IndicVoices",   {"data_dir": "nepali"}, "asr"),
    "fleurs":        ("google/fleurs",           {"name": "ne_np"},     "clean"),
}

TARGET_SR = 24000
MIN_DUR, MAX_DUR = 3.0, 20.0  # Rasa clips run to ~15 s; F5 batches by frames
MIN_SNR_DB = 15.0
MAX_ASR_CER = 0.15  # IndicVoices-R ships a per-clip ASR CER: drops transcript/audio mismatches
SPEAKER_KEYS = ("speaker_id", "speaker", "spk_id", "client_id", "gender")  # Rasa: 2 voices, keyed by gender
TEXT_KEYS = ("text", "transcription", "verbatim", "normalized", "sentence")


def estimate_snr_db(wav, sr, frame_ms=25):
    """Crude SNR: loudest-decile frame energy vs quietest-decile (noise floor)."""
    import numpy as np
    hop = int(sr * frame_ms / 1000)
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    if len(frames) < 10:
        return 0.0
    e = np.sort((frames ** 2).mean(axis=1) + 1e-12)
    k = max(1, len(e) // 10)
    return float(10 * np.log10(e[-k:].mean() / e[:k].mean()))


def preprocess_clip(wav, sr):
    import librosa
    import numpy as np
    wav = np.asarray(wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    dur = len(wav) / TARGET_SR
    if not (MIN_DUR <= dur <= MAX_DUR):
        return None
    snr = estimate_snr_db(wav, TARGET_SR)
    if snr < MIN_SNR_DB:
        return None
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav * (10 ** (-1 / 20) / peak)
    return wav, dur, snr


def first(row, keys):
    for k in keys:
        if row.get(k):
            return str(row[k])
    return None


def _cer(row):
    v = row.get("cer")
    if v is None:
        return 0.0
    try:
        return float(str(v).replace("tensor(", "").rstrip(")"))
    except ValueError:
        return 0.0


def _process(row, idx, key, grade, clip_dir):
    """Per-clip worker (runs in parallel via datasets.map)."""
    import io
    import soundfile as sf
    # output column must not be named "audio": datasets would keep the Audio feature type
    rec = {"keep": False, "wav_path": "", "text": "", "phonemes": "", "f5_phonemes": "",
           "speaker": "", "dur": 0.0, "snr": 0.0, "style": ""}
    audio = row.get("audio") or row.get("speech")
    text = first(row, TEXT_KEYS)
    if not audio or not text or _cer(row) > MAX_ASR_CER:
        return rec
    try:
        if audio.get("bytes"):
            arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.T
        else:
            arr, sr = audio["array"], audio["sampling_rate"]
    except Exception:
        return rec
    pc = preprocess_clip(arr, sr)
    if pc is None:
        return rec
    wav, dur, snr = pc
    path = os.path.abspath(os.path.join(clip_dir, f"{key}_{idx:07d}.wav"))
    sf.write(path, wav, TARGET_SR)
    norm = normalize(text.strip())
    toks = g2p(norm)
    spk = first(row, SPEAKER_KEYS)
    rec.update(keep=True, wav_path=path, text=norm, phonemes=" ".join(toks),
               f5_phonemes=encode_for_f5(toks), speaker=f"{key}:{spk}" if spk else "",
               dur=dur, snr=snr, style=str(row.get("style") or ""))
    return rec


def run(out_dir, source_keys, num_proc, tag=""):
    from datasets import Audio, load_dataset
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for key in source_keys:
        hf_id, kw, grade = SOURCES[key]
        print(f"[{key}] {hf_id} {kw} (num_proc={num_proc})")
        # non-streaming: parquet shards download in parallel and are cached
        ds = load_dataset(hf_id, split="train", num_proc=num_proc, **kw)
        ds = ds.cast_column("audio", Audio(decode=False))  # decode bytes ourselves (no torchcodec)
        clip_dir = os.path.join(out_dir, "wav", key + tag)
        os.makedirs(clip_dir, exist_ok=True)
        out = ds.map(_process, with_indices=True, num_proc=num_proc,
                     remove_columns=ds.column_names,
                     fn_kwargs={"key": key, "grade": grade, "clip_dir": clip_dir})
        kept = [r for r in out.filter(lambda k: k, input_columns="keep", num_proc=num_proc)]
        for r in kept:
            manifest.append({
                "audio": r["wav_path"], "text": r["text"], "phonemes": r["phonemes"],
                "f5_phonemes": r["f5_phonemes"], "speaker": r["speaker"] or None,
                "source": key, "grade": grade, "dur": r["dur"], "snr": r["snr"], "style": r["style"],
            })
        print(f"[{key}] kept {len(kept)}/{len(ds)}")
    with open(os.path.join(out_dir, f"manifest{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for field, name in (("text", "metadata_chars.csv"), ("f5_phonemes", "metadata_phonemes.csv")):
        with open(os.path.join(out_dir, name.replace(".csv", f"{tag}.csv")), "w", encoding="utf-8") as f:
            f.write("audio_file|text\n")
            for r in manifest:
                f.write(f"{r['audio']}|{r[field]}\n")
    hrs = sum(r["dur"] for r in manifest) / 3600
    spks = len({r["speaker"] for r in manifest if r["speaker"]})
    print(f"\n{len(manifest)} clips, {hrs:.1f}h, {spks} speakers -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", default=["indicvoices_r", "rasa"])
    ap.add_argument("--num_proc", type=int, default=os.cpu_count())
    # e.g. --min_dur 1.0 --max_dur 3.0 --tag _short: add back the short clips (names, commands)
    ap.add_argument("--min_dur", type=float, default=MIN_DUR)
    ap.add_argument("--max_dur", type=float, default=MAX_DUR)
    ap.add_argument("--tag", default="", help="suffix for manifest/csv/wav dir names")
    a = ap.parse_args()
    MIN_DUR, MAX_DUR = a.min_dur, a.max_dur  # module globals, read by forked workers
    run(a.out, a.sources, a.num_proc, a.tag)
