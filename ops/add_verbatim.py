"""Attach IndicVoices-R *verbatim* transcripts (what was actually said) to the manifest.
The pilot trained on the normalized `text` column, which differs from the audio in
~85% of clips (बेङ्गाल said, बङ्गाल written). Clip files are named by dataset index,
so index -> verbatim is exact; we check it against the stored text."""
import json
import sys

sys.path.insert(0, "/home/nepali-tts")
from datasets import load_dataset
from frontend import normalize

ds = load_dataset("ai4bharat/indicvoices_r", data_dir="Nepali", split="train", num_proc=16)
ds = ds.remove_columns([c for c in ds.column_names if c not in ("text", "verbatim")])
text, verb = ds["text"], ds["verbatim"]
out, checked, bad = [], 0, 0
for line in open("/home/data/nepali_tts/manifest.jsonl", encoding="utf-8"):
    m = json.loads(line)
    if m["source"] == "indicvoices_r":
        i = int(m["audio"].rsplit("_", 1)[1].split(".")[0])
        checked += 1
        bad += normalize(text[i].strip()) != m["text"]
        m["verbatim"] = normalize(verb[i].strip()) if verb[i] else m["text"]
    else:
        m["verbatim"] = m["text"]  # Rasa is read speech: script == speech
    out.append(m)
print(f"checked {checked}, index mismatches {bad}")
changed = sum(m["verbatim"] != m["text"] for m in out)
print(f"{changed}/{len(out)} transcripts change")
with open("/home/data/nepali_tts/manifest_v.jsonl", "w", encoding="utf-8") as f:
    for m in out:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
