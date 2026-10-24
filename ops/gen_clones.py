"""Unseen-speaker cloning examples for the listening page: for 4 held-out speakers,
copy their prompt clip and real take, and clone the same sentence from the prompt alone.
Usage: python ops/gen_clones.py CKPT OUT_DIR"""
import json
import shutil
import sys

sys.path.insert(0, "/home/nepali-tts/pipeline")
sys.path.insert(0, "/home/nepali-tts")
import clone  # noqa: E402
from f5_tts.api import F5TTS  # noqa: E402

ckpt, out = sys.argv[1], sys.argv[2]
scores = {r["id"]: r for r in json.load(open("/home/data/eval/chars10k_default/scores.json"))["rows"]}
items = json.load(open("/home/data/eval/items.json"))
# varied speakers whose real take the ASR reads well (so the sentence is clear), one item each
seen, pick = set(), []
for it in sorted(items, key=lambda it: scores.get(it["id"], {}).get("cer", 1)):
    if it["speaker"] not in seen and 20 <= len(it["text"]) <= 110:
        seen.add(it["speaker"])
        pick.append(it)
    if len(pick) == 4:
        break
clone.patch_f5_chunker()
t = F5TTS(model="F5TTS_Base", ckpt_file=ckpt, vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
t.ema_model.float()
meta = []
for i, it in enumerate(pick):
    shutil.copy(it["ref_wav"], f"{out}/clone{i}_prompt.wav")
    shutil.copy(it["real_wav"], f"{out}/clone{i}_real.wav")
    clone.set_pace(it["ref_text"], it["ref_wav"])
    t.infer(ref_file=it["ref_wav"], ref_text=it["ref_text"], gen_text=it["text"], file_wave=f"{out}/clone{i}_ours.wav",
            remove_silence=True, seed=0, speed=clone.speed_for(it["text"]))
    meta.append({"i": i, "text": it["text"], "prompt_text": it["ref_text"]})
json.dump(meta, open(f"{out}/clones.json", "w"), ensure_ascii=False, indent=1)
print("done", len(meta))
