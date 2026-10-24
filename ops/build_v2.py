"""v2/v3 training list (v3 = v2 + IndicVoices Read/Extempore, `build_v2.py v3`): verbatim IndicVoices-R transcripts + all train-speaker Rasa (incl. the
1,059 clips the pilot missed) + Rasa 1-3 s clips. Held-out speakers and eval sentences excluded.
Writes train_v2_chars.csv and train_v2_respell.csv (audio|text)."""
import json
import sys
from multiprocessing import Pool

sys.path.insert(0, "/home/nepali-tts")
from frontend import g2p
from frontend.g2p import respell

D = "/home/data/nepali_tts"
held = set(json.load(open("/home/data/ref/holdout_speakers.json")))
eval_texts = {it["text"] for it in json.load(open("/home/data/eval/items.json"))}


V = sys.argv[1] if len(sys.argv) > 1 else "v2"


def rows():
    srcs = [("manifest_v.jsonl", "verbatim"), ("manifest_short.jsonl", "text")]
    if V == "v3":
        srcs.append(("manifest_iv.jsonl", "text"))  # text is already verbatim there
    for name, field in srcs:
        for line in open(f"{D}/{name}", encoding="utf-8"):
            m = json.loads(line)
            if m.get("speaker") in held or m["text"] in eval_texts or m.get("scenario") == "Conversation":
                continue  # Conversation = telephone band (nothing above 4 kHz)
            yield m["audio"], m[field], m["dur"]


def conv(r):
    # F5's csv reader treats '"' as a quote: one stray quote swallows the following rows
    # into a single giant transcript. Quotes are not spoken, so drop them (and '|').
    t = " ".join(r[1].replace('"', " ").replace("'", " ").replace("|", " ").split())
    return r[0], t, respell(g2p(t))


if __name__ == "__main__":
    rs = list(rows())
    with Pool(28) as p:
        out = p.map(conv, rs, chunksize=500)
    for col, name in ((1, "chars"), (2, "respell")):
        with open(f"{D}/train_{V}_{name}.csv", "w", encoding="utf-8") as f:
            f.write("audio_file|text\n")
            for r in out:
                f.write(f"{r[0]}|{r[col]}\n")
    print(len(out), "clips", round(sum(r[2] for r in rs) / 3600, 1), "h")
