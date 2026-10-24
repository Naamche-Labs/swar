"""Is voice cloning really working? Speaker-verification audit on the 50 held-out speakers.

For every held-out item we have: the prompt clip (ref), the speaker's real recording of
the target sentence (real), and our clone of that sentence (clone). With two independent
verifiers (ECAPA and WavLM-SV) we compare:
  real-vs-real   same person, two recordings            (ceiling)
  clone-vs-real  our clone vs the person's real take     (what we deliver)
  impostor       real takes of two different people      (floor)
and report how often each verifier, at the threshold where real pairs and impostors are
best separated (EER), would accept the clone as the same person.

Usage: python tools/clone_audit.py RUN_NAME [RUN_NAME ...]   (runs under /home/data/eval)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_heldout import DEV, EVAL  # noqa: E402


class Verifiers:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV},
                                                    savedir="/home/hf_cache/ecapa")
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEV).eval()
        self.cache = {}

    def emb(self, path):
        import librosa
        import torch
        if path not in self.cache:
            w, _ = librosa.load(path, sr=16000)
            with torch.no_grad():
                e1 = self.ecapa.encode_batch(torch.tensor(w).unsqueeze(0).to(DEV)).squeeze()
                inp = self.fe(w, sampling_rate=16000, return_tensors="pt").to(DEV)
                e2 = self.wavlm(**inp).embeddings.squeeze()
            self.cache[path] = (torch.nn.functional.normalize(e1, dim=-1), torch.nn.functional.normalize(e2, dim=-1))
        return self.cache[path]

    def sim(self, a, b):
        ea, eb = self.emb(a), self.emb(b)
        return [float((x * y).sum()) for x, y in zip(ea, eb)]


def eer_threshold(pos, neg):
    best = None
    for t in sorted(pos + neg):
        far = sum(n >= t for n in neg) / len(neg)
        frr = sum(p < t for p in pos) / len(pos)
        if best is None or abs(far - frr) < best[0]:
            best = (abs(far - frr), t, (far + frr) / 2)
    return best[1], best[2]


def main(runs):
    items = json.load(open(f"{EVAL}/items.json"))
    v = Verifiers()
    names = ["ECAPA", "WavLM-SV"]
    pos = [v.sim(it["real_wav"], it["ref_wav"]) for it in items]
    # impostor: pair each real take with the next item's speaker's real take (different speaker)
    neg = []
    for i, it in enumerate(items):
        for j in range(1, len(items)):
            o = items[(i + j) % len(items)]
            if o["speaker"] != it["speaker"]:
                neg.append(v.sim(it["real_wav"], o["real_wav"]))
                break
    out = {}
    for k, nm in enumerate(names):
        P, N = [p[k] for p in pos], [n[k] for n in neg]
        t, eer = eer_threshold(P, N)
        out[nm] = {"real_vs_real": sum(P) / len(P), "impostor": sum(N) / len(N), "eer_threshold": t, "eer": eer,
                   "real_accept": sum(p >= t for p in P) / len(P)}
        print(f"{nm:9s} real-vs-real {out[nm]['real_vs_real']:.3f}  impostor {out[nm]['impostor']:.3f}  "
              f"EER {100 * eer:.1f}% at {t:.3f}  (real pairs accepted {100 * out[nm]['real_accept']:.0f}%)")
    for run in runs:
        for k, nm in enumerate(names):
            t = out[nm]["eer_threshold"]
            C = [v.sim(f"{EVAL}/{run}/{it['id']}.wav", it["real_wav"])[k] for it in items
                 if os.path.exists(f"{EVAL}/{run}/{it['id']}.wav")]
            R = [v.sim(f"{EVAL}/{run}/{it['id']}.wav", it["ref_wav"])[k] for it in items
                 if os.path.exists(f"{EVAL}/{run}/{it['id']}.wav")]
            out[nm][run] = {"clone_vs_real": sum(C) / len(C), "clone_vs_prompt": sum(R) / len(R),
                            "accepted": sum(c >= t for c in C) / len(C)}
            r = out[nm][run]
            print(f"{nm:9s} {run:28s} clone-vs-real {r['clone_vs_real']:.3f}  clone-vs-prompt {r['clone_vs_prompt']:.3f}  "
                  f"accepted as same person {100 * r['accepted']:.0f}%", flush=True)
    json.dump(out, open(f"{EVAL}/clone_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
