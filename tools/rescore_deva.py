"""Re-score with MMS forced to write Devanagari only.

MMS-1b-all's Nepali head shares a vocabulary with Latin letters, and on names, English
words and 1-3 word phrases it switches script mid-word ("तानपुरा" -> "tan pूra"). Every
switch counts as errors even when the word was said right. Masking the Latin tokens out
of the CTC logits makes it spell everything in Devanagari. The mask is applied the same
way to every system and to the human recordings, so comparisons stay fair.

Usage:
    python tools/rescore_deva.py bench NAME [NAME ...]    # our run dirs or published systems
    python tools/rescore_deva.py rasa NAME [NAME ...]     # NAME=real for the human recordings
    python tools/rescore_deva.py heldout NAME [NAME ...]
Writes /home/data/deva/<suite>.json ({name: {cer, mean_file_cer, by_style?}}).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_heldout import DEV, EVAL, ed, norm  # noqa: E402

OUT = "/home/data/deva"


class DevaMMS:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                                    ignore_mismatched_sizes=True).to(DEV).eval()
        self.mask = self.latin_mask(self.proc)
        print("masked", int((self.mask != 0).sum()), "Latin tokens of", len(self.mask), flush=True)

    @staticmethod
    def latin_mask(proc):
        import torch
        vocab = proc.tokenizer.get_vocab()
        mask = torch.zeros(len(vocab), device=DEV)
        for tok, i in vocab.items():
            if re.search(r"[A-Za-z]", tok) and not tok.startswith("<"):
                mask[i] = float("-inf")
        return mask

    def __call__(self, path, deva=True):
        import librosa
        import torch
        w, _ = librosa.load(path, sr=16000)
        inp = {k: v.to(DEV) for k, v in self.proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            lg = self.model(**inp).logits[0]
        return self.proc.decode((lg + self.mask if deva else lg).argmax(-1))


def items_for(suite, name):
    """[(id, wav, text, style)] for one system."""
    if suite == "bench":
        from eval_bench import sentences
        from rescore_bench import audio_for
        a = audio_for(name)
        return [(x["sent_id"], a[x["sent_id"]], x["text_devanagari"], x["category"]) for x in sentences()
                if x["sent_id"] in a]
    if suite == "rasa":
        from eval_rasa import R
        its = json.load(open(f"{R}/items.json"))
        wav = (lambda it: it["real_wav"]) if name == "real" else (lambda it: f"{R}/runs/{name}/{it['id']}.wav")
        return [(it["id"], wav(it), it["text"], it["style"]) for it in its if os.path.exists(wav(it))]
    its = json.load(open(f"{EVAL}/items.json"))
    wav = (lambda it: it["real_wav"]) if name == "real" else (lambda it: f"{EVAL}/{name}/{it['id']}.wav")
    return [(it["id"], wav(it), it["text"], "all") for it in its if os.path.exists(wav(it))]


def agg(rows):
    return {"n": len(rows), "cer": 100 * sum(r["e"] for r in rows) / max(1, sum(r["n"] for r in rows)),
            "mean_file_cer": 100 * sum(r["e"] / max(1, r["n"]) for r in rows) / max(1, len(rows))}


if __name__ == "__main__":
    suite, names = sys.argv[1], sys.argv[2:]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{suite}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    asr = DevaMMS()
    for n in names:
        rows = []
        for i, wav, text, style in items_for(suite, n):
            ref = norm(text)
            hyp = norm(asr(wav))
            rows.append({"id": i, "style": style, "e": ed(ref, hyp), "n": len(ref), "hyp": hyp, "ref": ref})
        styles = sorted({r["style"] for r in rows})
        res[n] = {**agg(rows), "by_style": {s: agg([r for r in rows if r["style"] == s]) for s in styles}, "rows": rows}
        st = "  ".join(f"{s[:6]} {v['mean_file_cer']:.1f}" for s, v in res[n]["by_style"].items()) if suite == "rasa" else ""
        print(f"{suite} {n:28s} CER {res[n]['cer']:6.2f}  mean-file {res[n]['mean_file_cer']:6.2f}  {st}", flush=True)
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
