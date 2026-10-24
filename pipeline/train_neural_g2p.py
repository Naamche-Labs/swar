"""Neural char-level G2P (Transformer seq2seq) for unseen Nepali words.
Runs on the GPU box (needs torch); never imported by the frontend.

Same 80/10/10 split and scoring as pipeline/train_g2p.py, so its held-out
number is directly comparable to the current pipeline (90.61% / PER 1.52%).
With --export, and only if it beats the vote model on the held-out test,
it writes pronunciations for every out-of-lexicon word in a manifest's text
to data/lexicons/neural_oov.tsv, which frontend/g2p.py loads after the
Google lexicon. Inference in the TTS frontend therefore never needs torch.

Usage (GPU box):
    python pipeline/train_neural_g2p.py
    python pipeline/train_neural_g2p.py --export /data/nepali_tts/manifest.jsonl
"""
import argparse
import importlib
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
T = importlib.import_module("pipeline.train_g2p")
G = T.G
from pipeline.eval_g2p import edit_distance, norm_mine

import torch
import torch.nn as nn

PAD, BOS, EOS = 0, 1, 2
OUT_PATH = os.path.join(ROOT, "data", "lexicons", "neural_oov.tsv")


class Vocab:
    def __init__(self, seqs):
        self.itos = ["<pad>", "<s>", "</s>"] + sorted({t for s in seqs for t in s})
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, seq):
        return [BOS] + [self.stoi[t] for t in seq if t in self.stoi] + [EOS]


class G2PModel(nn.Module):
    def __init__(self, n_src, n_tgt, d=256, heads=4, layers=3, ff=1024, drop=0.1, max_len=64):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.tf = nn.Transformer(d, heads, layers, layers, ff, drop, batch_first=True)
        self.out = nn.Linear(d, n_tgt)
        self.d = d

    def embed(self, emb, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return emb(x) * math.sqrt(self.d) + self.pos(pos)

    def forward(self, src, tgt_in):
        mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=src.device)
        h = self.tf(self.embed(self.src_emb, src), self.embed(self.tgt_emb, tgt_in), tgt_mask=mask,
                    src_key_padding_mask=src.eq(PAD), tgt_key_padding_mask=tgt_in.eq(PAD),
                    memory_key_padding_mask=src.eq(PAD))
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src, max_len=48):
        mem = self.tf.encoder(self.embed(self.src_emb, src), src_key_padding_mask=src.eq(PAD))
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=src.device)
            h = self.tf.decoder(self.embed(self.tgt_emb, ys), mem, tgt_mask=mask,
                                memory_key_padding_mask=src.eq(PAD))
            nxt = self.out(h[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
            done |= nxt.eq(EOS)
            if done.all():
                break
        return ys[:, 1:]


def pad(batch):
    n = max(len(x) for x in batch)
    return torch.tensor([x + [PAD] * (n - len(x)) for x in batch], dtype=torch.long)


def decode(model, words, sv, tv, dev, bs=512):
    model.eval()
    res = []
    for i in range(0, len(words), bs):
        src = pad([sv.encode(list(w)) for w in words[i:i + bs]]).to(dev)
        for row in model.greedy(src).tolist():
            toks = []
            for t in row:
                if t in (EOS, PAD):
                    break
                toks.append(tv.itos[t])
            res.append(toks)
    return res


def evaluate(model, data, sv, tv, dev):
    hyps = decode(model, [w for w, _ in data], sv, tv, dev)
    ok = perr = ptot = 0
    for (w, ref), h in zip(data, hyps):
        h = norm_mine(h)
        ok += h == ref
        perr += edit_distance(ref, h)
        ptot += len(ref)
    return 100 * ok / len(data), 100 * perr / ptot


def main(a):
    torch.manual_seed(0)
    random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = T.load_pairs()
    sets = defaultdict(list)
    for w, ref in pairs:
        if w in G.LEXICON:
            sets[T.split_of(w)].append((w, ref, G.LEXICON[w]))
    sv = Vocab([list(w) for w, _, _ in sets["train"]])
    tv = Vocab([t for _, _, t in sets["train"]])
    train = [(sv.encode(list(w)), tv.encode(t)) for w, _, t in sets["train"]]
    model = G2PModel(len(sv.itos), len(tv.itos)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    dev_data = [(w, ref) for w, ref, _ in sets["dev"]]
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        random.shuffle(train)
        for i in range(0, len(train), a.bs):
            src = pad([s for s, _ in train[i:i + a.bs]]).to(dev)
            tgt = pad([t for _, t in train[i:i + a.bs]]).to(dev)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            acc, per = evaluate(model, dev_data, sv, tv, dev)
            print(f"epoch {ep + 1}: loss {loss.item():.3f}  dev acc {acc:.2f}%  PER {per:.2f}%")
            if acc > best:
                best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = [(w, ref) for w, ref, _ in sets["test"]]
    acc, per = evaluate(model, test, sv, tv, dev)
    vote = 90.61  # current pipeline, held-out test (pipeline/train_g2p.py)
    print(f"\nTEST (held out): neural {acc:.2f}% / PER {per:.2f}%   vs current pipeline {vote}% / PER 1.52%")

    if a.export:
        if acc <= vote:
            print("neural model does not beat the vote model on held-out test; not exporting")
            return
        words = set()
        with open(a.export, encoding="utf-8") as f:
            for line in f:
                for raw in json.loads(line)["text"].split():
                    w = raw.strip("।॥.?!,;:\"'()")
                    if w and not w.isascii() and w not in G.LEXICON:
                        words.add(w)
        words = sorted(words)
        prons = decode(model, words, sv, tv, dev)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for w, p in zip(words, prons):
                if p:
                    f.write(f"{w}\t{' '.join(p)}\n")
        print(f"wrote {len(words)} OOV pronunciations -> {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--export", help="manifest.jsonl whose OOV words get neural pronunciations")
    main(ap.parse_args())
