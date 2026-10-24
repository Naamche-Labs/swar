"""Put punctuation back into unpunctuated training transcripts, from the audio.

IndicVoices(-R) and OpenSLR transcripts (about 220k clips, most of our training data) carry no
punctuation at all, while ~30% of IndicVoices-R clips contain a question word: the model never saw
what "?", "," or "।" sound like outside Rasa. For each clip:
  1. CTC forced alignment of the transcript to the audio (MMS-1b-all, Nepali head, Devanagari-only
     vocabulary as in rescore_deva.py) gives each word's start and end;
  2. a pause >= 0.30 s between two words becomes ", " (>= 0.65 s after a sentence-final verb form
     becomes "। ");
  3. the end gets "?" when the transcript has a question word and the last 25% of voiced F0 is
     >= 1 semitone above the clip median, else "।".
Writes manifest_punct.jsonl (same rows, text replaced, plus "punct_src").
Usage (GPU box): python tools/restore_punct.py IN.jsonl [IN.jsonl ...] OUT.jsonl
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QWORDS = re.compile(r"(^| )(के|कहाँ|किन|कसरी|कति|कुन|कसले|कसको|कहिले|कस्तो|होला|हो र|छ र|छन् र)( |$)")
FINAL = re.compile(r"(छ|छन्|छु|छौँ|हो|हुन्|थियो|थिए|भयो|भए|गर्छ|गर्छन्|हुन्छ|हुन्छन्|पर्छ|सक्छ|गर्‍यो|गरे|गर्यो)$")


def f0_end_rise(w, sr):
    import librosa
    import numpy as np
    f0, v, _ = librosa.pyin(w, fmin=70, fmax=450, sr=sr, frame_length=1024)
    f0 = f0[v & ~np.isnan(f0)]
    if len(f0) < 20:
        return 0.0
    st = 12 * np.log2(f0 / 100)
    return float(np.median(st[-max(5, len(st) // 4):]) - np.median(st))


def main(inputs, out):
    import librosa
    import torch
    import torchaudio.functional as AF
    from rescore_deva import DevaMMS
    asr = DevaMMS()
    tok = asr.proc.tokenizer
    vocab = tok.get_vocab()
    blank = vocab[tok.pad_token]
    sep = vocab.get("|")
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["audio"] for line in open(out, encoding="utf-8")}
    fo = open(out, "a", encoding="utf-8")
    n = changed = q = 0
    for path in inputs:
        for line in open(path, encoding="utf-8"):
            x = json.loads(line)
            if x["audio"] in done:
                continue
            t = x["text"].strip()
            if x["speaker"].startswith("rasa") or re.search(r"[।?!,]", t):  # already punctuated
                x["punct_src"] = "orig"
                fo.write(json.dumps(x, ensure_ascii=False) + "\n")
                continue
            words = t.split()
            try:
                w, sr = librosa.load(x["audio"], sr=16000)
                ids = []
                for i, wd in enumerate(words):
                    ids += [vocab[c] for c in wd if c in vocab]
                    if i < len(words) - 1 and sep is not None:
                        ids.append(sep)
                inp = {k: v.to("cuda") for k, v in asr.proc(w, sampling_rate=16000, return_tensors="pt").items()}
                with torch.no_grad():
                    lp = torch.log_softmax((asr.model(**inp).logits[0] + asr.mask).float(), -1)
                if not ids or len(ids) > lp.shape[0]:
                    raise ValueError("cannot align")
                ali, _ = AF.forced_align(lp.unsqueeze(0), torch.tensor([ids], device=lp.device), blank=blank)
                ali = ali[0].tolist()
                # frames of each word: walk the alignment, word index advances at each separator token
                fr = 0.02 * (len(w) / 16000) / (0.02 * lp.shape[0])  # seconds per frame
                spans, cur, wi, prev = [], None, 0, blank
                for f, a in enumerate(ali):
                    if a == sep and prev != sep:
                        if cur:
                            spans.append(cur)
                        cur, wi = None, wi + 1
                    elif a not in (blank, sep):
                        cur = [cur[0] if cur else f, f]
                    prev = a
                if cur:
                    spans.append(cur)
                if len(spans) != len(words):
                    raise ValueError("span mismatch")
                outw = []
                for i, wd in enumerate(words):
                    outw.append(wd)
                    if i < len(words) - 1:
                        gap = (spans[i + 1][0] - spans[i][1]) * fr
                        if gap >= 0.65 and FINAL.search(wd):
                            outw[-1] += "।"
                        elif gap >= 0.30:
                            outw[-1] += ","
                rise = f0_end_rise(w, 16000) if QWORDS.search(t) else 0.0
                end = "?" if QWORDS.search(t) and rise >= 1.0 else "।"
                x["text"] = " ".join(outw) + end
                x["punct_src"] = "restored"
                changed += 1
                q += end == "?"
            except Exception:
                x["text"] = t + "।"
                x["punct_src"] = "end_only"
            fo.write(json.dumps(x, ensure_ascii=False) + "\n")
            n += 1
            if n % 2000 == 0:
                print(f"{n} done, {changed} restored, {q} questions", flush=True)
                fo.flush()
    print(f"DONE {n} clips, {changed} restored, {q} questions")


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
