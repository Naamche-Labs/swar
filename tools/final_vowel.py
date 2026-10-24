"""Are word-final vowels actually said? User: खाना comes out "khan", आज "aaj" (Nepali keeps them: khaa-naa, aa-ja).
Kriti's read-back cannot see this (its language model writes the word it expects), so this measures the audio:
  literal  Devanagari MMS greedy CTC (no language model): does the transcript keep the final vowel?
  final_ms forced alignment (MMS emissions, torchaudio forced_align) of the canonical spelling: time from the start of
           the word's last letter to the next sound; a clipped vowel makes it short
Variants: model-input respellings of the target word (the canonical word is always what is measured).
Usage: python tools/final_vowel.py OUT_DIR CKPT PROMPT_WAV "PROMPT_TEXT" [SPEED] [CASES_JSON]"""
import json
import os
import re
import sys

import numpy as np
import torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from rescore_deva import DevaMMS  # noqa: E402

# sentence with {} where the word goes, canonical word, model-input variants
CASES = [("हिजो बेलुका हामी साथीहरू मिलेर {} खान गयौँ।", "खाना", ["खाना", "खानाऽ", "खाना,", "खान्ना"]),
         ("मलाई {} पकाउन धेरै मन पर्छ।", "खाना", ["खाना", "खानाऽ", "खाना,"]),
         ("{} बिहान म अलि ढिलो उठेँ।", "आज", ["आज", "आजा", "आजऽ", "आज,"]),
         ("{} खाना एकदमै मीठो थियो।", "आजको", ["आजको", "आजाको", "आजऽको"]),
         ("म {} राति घर फर्किन्छु।", "आज", ["आज", "आजा", "आजऽ"])]


def main(out, ckpt, pwav, ptext, speed=1.0, cases=None):
    global CASES
    if cases:
        CASES = json.load(open(cases))
    from f5_tts.api import F5TTS
    from clone import patch_f5_chunker, set_pace
    from frontend import normalize
    import librosa
    import torchaudio.functional as AF
    os.makedirs(out, exist_ok=True)
    patch_f5_chunker()
    tts = F5TTS(model="F5TTS_Base", ckpt_file=ckpt, vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
    tts.ema_model.float()
    mms = DevaMMS()
    vocab = mms.proc.tokenizer.get_vocab()
    rt = normalize(ptext)
    rows = []
    for ci, (tpl, word, variants) in enumerate(CASES):
        canon = tpl.format(word)
        for vi, var in enumerate(variants):
            for seed in range(int(os.environ.get("SEEDS", "4"))):
                p = f"{out}/c{ci}_v{vi}_s{seed}.wav"
                set_pace(rt, pwav)
                tts.infer(ref_file=pwav, ref_text=rt, gen_text=normalize(tpl.format(var), english=False), file_wave=p, nfe_step=32,
                          seed=seed, cfg_strength=2.0, speed=float(speed), remove_silence=True)
                w = librosa.load(p, sr=16000)[0]
                inp = {k: v.to(mms.model.device) for k, v in mms.proc(w, sampling_rate=16000, return_tensors="pt").items()}
                with torch.no_grad():
                    lg = mms.model(**inp).logits[0] + mms.mask
                lit = mms.proc.decode(lg.argmax(-1))
                # forced alignment of the canonical sentence
                chars = [("|" if c == " " else c) for c in re.sub(r"[।,?!]", "", canon).strip()]
                ids = [vocab[c] for c in chars if c in vocab]
                keep = [c for c in chars if c in vocab]
                lp = torch.log_softmax(lg.float(), -1).unsqueeze(0).cpu()
                al, _ = AF.forced_align(lp, torch.tensor([ids]), blank=mms.proc.tokenizer.pad_token_id)
                al = al[0].tolist()
                starts, ti, prev = [], -1, None
                for f, a in enumerate(al):  # first frame of each target token
                    if a != 0 and a != mms.proc.tokenizer.pad_token_id and (a != prev):
                        ti += 1
                        starts.append(f)
                    prev = a if a != mms.proc.tokenizer.pad_token_id else None
                # the target word's last letter and the token after it
                txt = "".join(keep).split("|")
                wi = [i for i, t in enumerate(txt) if t == word][0]
                last = sum(len(t) + 1 for t in txt[:wi]) + len(word) - 1
                final_ms = (starts[last + 1] - starts[last]) * 20 if last + 1 < len(starts) else float("nan")
                heard_word = lit.split()[wi] if wi < len(lit.split()) else ""
                rows.append({"case": ci, "variant": var, "seed": seed, "literal": lit, "heard_word": heard_word,
                             "final_ms": final_ms, "file": p})
            vs = [r for r in rows if r["case"] == ci and r["variant"] == var]
            keep_ok = np.mean([r["heard_word"].startswith(word) and len(r["heard_word"]) >= len(word) for r in vs])
            print(f"CASE {ci} {word:5s} as {var:7s} final {np.nanmean([r['final_ms'] for r in vs]):5.0f} ms  literal keeps word "
                  f"{100 * keep_ok:3.0f}%  | {vs[0]['literal'][:50]}", flush=True)
    json.dump(rows, open(f"{out}/final_vowel.json", "w"), ensure_ascii=False, indent=1)
    print("FINAL_VOWEL_DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
