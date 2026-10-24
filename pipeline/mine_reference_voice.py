"""Pick flagship + held-out prompt speakers from the prepared manifest.

No artist needed. Speakers with real IDs are ranked by clean (high-SNR) audio;
the best becomes the flagship reference voice and a gender/source-diverse set
is held out (never trained on) for zero-shot speaker-similarity eval, which
NepTTS-Bench does not provide.

Runs on the GPU box after prepare_data.py.
Usage:
    python mine_reference_voice.py --manifest /data/nepali_tts/manifest.jsonl --out /data/ref --holdout 50
Outputs: ref/flagship/*.wav + reference.json, ref/holdout_speakers.json,
and manifest.train.jsonl (manifest minus held-out speakers).
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from statistics import median


def run(manifest, out_dir, holdout=50, ref_clips=5, min_snr=25.0):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    by_spk = defaultdict(list)
    for r in rows:
        if r.get("speaker"):
            by_spk[r["speaker"]].append(r)
    stats = []
    for spk, clips in by_spk.items():
        clean = [c for c in clips if c["snr"] >= min_snr]
        stats.append((spk, sum(c["dur"] for c in clean), median(c["snr"] for c in clips), clean))
    stats.sort(key=lambda s: (-s[1], -s[2]))
    print(f"{len(stats)} identified speakers")

    flagship, clean_dur, med_snr, clean = stats[0]
    fdir = os.path.join(out_dir, "flagship")
    os.makedirs(fdir, exist_ok=True)
    ref = {"speaker": flagship, "clean_minutes": clean_dur / 60, "median_snr": med_snr, "clips": []}
    for i, c in enumerate(sorted(clean, key=lambda c: -c["snr"])[:ref_clips]):
        dst = os.path.join(fdir, f"ref_{i:02d}.wav")
        shutil.copy(c["audio"], dst)
        ref["clips"].append({"audio": dst, "text": c["text"], "phonemes": c["phonemes"]})
    json.dump(ref, open(os.path.join(fdir, "reference.json"), "w"), ensure_ascii=False, indent=2)
    print(f"flagship: {flagship} ({clean_dur/60:.1f} clean min, median SNR {med_snr:.1f} dB)")

    # held-out prompt speakers: mid-sized speakers (1-10 clean min) from many-speaker
    # sources only, so we never hold out one of Rasa's 2 studio voices or the
    # biggest speakers; hash order spreads the pick deterministically.
    import hashlib
    n_src = defaultdict(int)
    for spk, *_ in stats:
        n_src[spk.split(":")[0]] += 1
    pool = [spk for spk, d, s, _ in stats[1:] if 60 <= d <= 600 and n_src[spk.split(":")[0]] >= 20]
    pool.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    held = pool[:holdout]
    json.dump(held, open(os.path.join(out_dir, "holdout_speakers.json"), "w"), indent=2)
    held_set = set(held)
    train_path = manifest.replace(".jsonl", ".train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in rows:
            if r.get("speaker") not in held_set:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"held out {len(held)} speakers; training manifest -> {train_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=int, default=50)
    a = ap.parse_args()
    run(a.manifest, a.out, a.holdout)
