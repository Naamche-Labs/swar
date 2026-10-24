"""Word n-gram LM for MMS prompt transcription (CTC + KenLM beam search).

The prompt transcript F5 gets matters: MMS greedy output (28% CER on spontaneous
prompts) costs +3 pp CER vs the gold transcript. Text comes from public Nepali corpora
only (Wikipedia + a news/web LM dataset), never from IndicVoices, so held-out prompt
transcripts cannot leak in.

Usage (box, needs kenlm's lmplz/build_binary on PATH):
    python tools/build_lm.py text      # /home/lm/ne.txt (normalized, one sentence per line)
    lmplz -o 5 --prune 0 0 1 < /home/lm/ne.txt > /home/lm/ne5.arpa && build_binary ...
    [DEVA=1] python tools/build_lm.py eval /home/lm/ne4.bin ["a,b;a,b"]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LM = "/home/lm"
KEEP = re.compile(r"[^ऀ-ॿ ]")


def clean(s):
    s = re.sub(r"[।॥.,?!;:\"'()\[\]\-–—]", " ", s)
    s = KEEP.sub(" ", s)
    s = " ".join(s.split())
    return s if len(s.split()) >= 3 else ""


def text():
    from datasets import load_dataset
    os.makedirs(LM, exist_ok=True)
    n = 0
    with open(f"{LM}/ne.txt", "w", encoding="utf-8") as f:
        srcs = [("wikimedia/wikipedia", "20231101.ne", "text"), ("Sakonii/nepalitext-language-model-dataset", None, "text")]
        for name, cfg, col in srcs:
            try:
                ds = load_dataset(name, cfg, split="train", streaming=True)
                k = 0
                for r in ds:
                    for line in re.split(r"[।\n?!]+", r[col]):
                        c = clean(line)
                        if c:
                            f.write(c + "\n")
                            k += 1
                    if k > 8_000_000:
                        break
                print(name, k, "sentences", flush=True)
                n += k
            except Exception as e:
                print(name, "FAILED", type(e).__name__, str(e)[:150], flush=True)
    print("total", n)


def decoder(lm_path, alpha=0.5, beta=1.0):
    from pyctcdecode import build_ctcdecoder
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    vocab = proc.tokenizer.get_vocab()
    labels = [None] * len(vocab)
    for tok, i in vocab.items():
        labels[i] = tok
    pad = proc.tokenizer.pad_token
    labels = ["" if t == pad else " " if t == "|" else (f"⁇{i}" if t.startswith("<") else t)
              for i, t in enumerate(labels)]
    return build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=alpha, beta=beta)


def evaluate(lm_path, grid):
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from eval_heldout import DEV, EVAL, ed, norm
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                         ignore_mismatched_sizes=True).to(DEV).eval()
    items = json.load(open(f"{EVAL}/items.json"))
    refs = {it["ref_wav"]: it["ref_text"] for it in items}
    logits = {}
    for wav in refs:
        w, _ = librosa.load(wav, sr=16000)
        inp = {k: v.to(DEV) for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            logits[wav] = torch.log_softmax(asr(**inp).logits[0].float(), -1).cpu().numpy()
    if os.environ.get("DEVA") == "1":  # Devanagari-only, as in rescore_deva.py
        from rescore_deva import DevaMMS
        mask = DevaMMS.latin_mask(proc).cpu().numpy()
        logits = {k: v + mask for k, v in logits.items()}
    E = N = 0
    for wav, t in refs.items():
        h = proc.decode(logits[wav].argmax(-1))
        E, N = E + ed(norm(t), norm(h)), N + len(norm(t))
    print(f"greedy CER {100 * E / N:.2f}%", flush=True)
    out = {}
    for a, b in grid:
        dec = decoder(lm_path, a, b)
        E = 0
        for wav, t in refs.items():
            E += ed(norm(t), norm(dec.decode(logits[wav], beam_width=64)))
        out[f"{a},{b}"] = 100 * E / N
        print(f"LM alpha {a} beta {b}: CER {100 * E / N:.2f}%", flush=True)
    json.dump(out, open(f"{LM}/eval.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "text":
        text()
    else:
        # optional grid "a,b;a,b;..."
        g = [tuple(map(float, x.split(","))) for x in sys.argv[3].split(";")] if len(sys.argv) > 3 else \
            [(a, b) for a in (0.3, 0.5, 0.8, 1.2) for b in (0.0, 1.0, 2.0)]
        evaluate(sys.argv[2], g)
