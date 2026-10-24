"""Add back Rasa clips of 1-3 s (names, short commands) from the local HF cache.
Rasa is gated and the box's HF token is wiped on pause, so read the cached parquet shards."""
import glob
import sys

import datasets

sys.path.insert(0, "/home/nepali-tts/pipeline")
import prepare_data as P

SHARDS = sorted(glob.glob("/home/hf_cache/hub/datasets--ai4bharat--Rasa/snapshots/*/Nepali/train-*.parquet"))
_load = datasets.load_dataset
datasets.load_dataset = lambda hf_id, split, num_proc, **kw: _load("parquet", data_files=SHARDS, split=split, num_proc=num_proc)
P.MIN_DUR, P.MAX_DUR = 1.0, 2.999
P.run("/home/data/nepali_tts", ["rasa"], 14, "_short")
