# Build train_respell.csv (audio|respelled text) from manifest.train.jsonl and check IndicF5 vocab coverage.
import json, sys
from multiprocessing import Pool
sys.path.insert(0, "/home/nepali-tts")
from frontend import g2p
from frontend.g2p import respell

def conv(line):
    m = json.loads(line)
    return m["audio"], respell(g2p(m["text"]))

if __name__ == "__main__":
    lines = open("/home/data/nepali_tts/manifest.train.jsonl", encoding="utf-8").readlines()
    with Pool(28) as p:
        rows = p.map(conv, lines, chunksize=500)
    vocab = set(l.rstrip("\n") for l in open("/home/models/IndicF5/checkpoints/vocab.txt", encoding="utf-8"))
    chars = {c for _, t in rows for c in t}
    print("rows", len(rows), "missing chars:", sorted(chars - vocab))
    with open("/home/data/nepali_tts/train_respell.csv", "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        for a, t in rows:
            f.write(f"{a}|{t}\n")
    for a, t in rows[:3]:
        print(t)
