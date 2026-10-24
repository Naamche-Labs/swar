"""Learned bridge: Vocos log-mel (what F5 predicts) -> BigVGAN-v2 24 kHz log-mel.

Vocos copy-synthesis of real Rasa test clips scores UTMOS 3.14 vs 3.88 for the originals, while
BigVGAN-v2 (nvidia/bigvgan_v2_24khz_100band_256x, MIT) on its own mel keeps 3.80. A
training-free filterbank conversion (pinv) only reaches 3.14, so a small 1-D conv net
(~1.3M params) learns the mapping from real speech: train clips only (Rasa + IndicVoices-R
train manifest), L1 on log-mel. Trained once for everyone, no per-user cost.

Usage (box with the train manifest, PYTHONPATH=/home/BigVGAN):
    python tools/train_mel_bridge.py train OUT.pt [steps]
    python tools/train_mel_bridge.py eval OUT.pt        # copy-synthesis UTMOS on held-out Rasa test
"""
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Bridge(nn.Module):
    def __init__(self, ch=256, n=6):
        super().__init__()
        self.inp = nn.Conv1d(100, ch, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2 * (i % 3 + 1), dilation=i % 3 + 1),
                                                   nn.GELU(), nn.Conv1d(ch, ch, 1)) for i in range(n)])
        self.out = nn.Conv1d(ch, 100, 5, padding=2)

    def forward(self, x):  # x: (b, 100, t) vocos log-mel -> (b, 100, t) bigvgan log-mel
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return x + self.out(h)


def feats(paths, voc, cfg, max_sec=8):
    import librosa
    from meldataset import get_mel_spectrogram
    out = []
    for p in paths:
        w, _ = librosa.load(p, sr=24000)
        w = w[: 24000 * max_sec]
        x = torch.tensor(w).unsqueeze(0).cuda()
        with torch.no_grad():
            a, b = voc.feature_extractor(x)[0], get_mel_spectrogram(x, cfg).cuda()[0]
        n = min(a.shape[-1], b.shape[-1])
        out.append((a[:, :n].cpu(), b[:, :n].cpu()))
    return out


def train(out, steps=6000):
    from vocos import Vocos
    from mel_bridge import load_bigvgan
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    _, cfg = load_bigvgan()
    m = [json.loads(line) for line in open("/home/data/nepali_tts/manifest.train.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(m)
    rasa = [x["audio"] for x in m if x["speaker"].startswith("rasa")][:1500]
    ivr = [x["audio"] for x in m if not x["speaker"].startswith("rasa")][:1500]
    data = feats(rasa + ivr, voc, cfg)
    print("pairs", len(data), flush=True)
    net = Bridge().cuda()
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=int(steps))
    seg = 128
    for step in range(int(steps)):
        xs, ys = [], []
        for a, b in random.sample(data, 32):
            if a.shape[-1] <= seg:
                continue
            i = random.randrange(a.shape[-1] - seg)
            xs.append(a[:, i:i + seg])
            ys.append(b[:, i:i + seg])
        x, y = torch.stack(xs).cuda(), torch.stack(ys).cuda()
        loss = (net(x) - y).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} L1 {loss.item():.4f}", flush=True)
    torch.save(net.state_dict(), out)
    print("saved", out, sum(p.numel() for p in net.parameters()) / 1e6, "M params")


def evaluate(ckpt):
    import librosa
    from vocos import Vocos
    from mel_bridge import load_bigvgan
    from meldataset import get_mel_spectrogram
    from mos import utmos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
    bv, cfg = load_bigvgan()
    net = Bridge().cuda().eval()
    net.load_state_dict(torch.load(ckpt))
    S = {"original": [], "vocos": [], "bigvgan_own_mel": [], "bridge+bigvgan": []}
    for it in json.load(open("/home/data/rasa_test/items.json"))[:40]:
        w, _ = librosa.load(it["real_wav"], sr=24000)
        x = torch.tensor(w).unsqueeze(0).cuda()
        with torch.no_grad():
            mv = voc.feature_extractor(x)
            outs = {"original": x, "vocos": voc.decode(mv), "bigvgan_own_mel": bv(get_mel_spectrogram(x, cfg).cuda()),
                    "bridge+bigvgan": bv(net(mv))}
        for k, y in outs.items():
            S[k].append(utmos(librosa.resample(y.squeeze().cpu().numpy(), orig_sr=24000, target_sr=16000)))
    for k, v in S.items():
        print(f"BRIDGE {k:16s} UTMOS {sum(v) / len(v):.3f} n={len(v)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(*sys.argv[2:4])
    else:
        evaluate(sys.argv[2])
