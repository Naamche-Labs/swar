"""Score our rule-based G2P against Google's Nepali lexicon (CC-BY 4.0, ~54k
words) so pronunciation quality is measured, not eyeballed.

Both sides are mapped to a shared phone set: no vowel length (Nepali has no
length contrast), diphthongs split into vowel sequences, nasal vowels as
vowel + n~. Reports word accuracy, phone error rate, and an error breakdown.

Usage:
    python pipeline/eval_g2p.py [--lexicon data/lexicons/google/lexicon.tsv] [--show 20]
"""
import argparse
import os
import sys
import unicodedata
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from frontend.g2p import LEXICON, _phonemize_word

GMAP = {"ax": ["a"], "tt": ["T"], "dd": ["D"], "tth": ["Th"], "ddh": ["Dh"], "nn": ["N"],
        "ts": ["c"], "tsh": ["ch"], "jh": ["j"], "jhh": ["jh"],
        "axj": ["a", "i"], "aaj": ["aa", "i"], "axw": ["a", "u"], "aaw": ["aa", "u"],
        "oj": ["o", "i"], "ew": ["e", "u"], "ow": ["o", "u"]}
_NASAL_BASES = {"aa", "ax", "i", "u", "e", "o", "aaw", "axw", "aaj", "axj", "ow", "oj"}
MMAP = {"ii": ["i"], "uu": ["u"], "ai": ["a", "i"], "au": ["a", "u"], "ri": ["r", "i"],
        "sh": ["s"], "f": ["ph"], "z": ["j"], "ny": ["n"]}


def norm_google(phones: str):
    out = []
    for t in phones.split():
        if t == ".":
            continue
        if t.endswith("n") and t[:-1] in _NASAL_BASES:
            out += GMAP.get(t[:-1], [t[:-1]]) + ["n~"]
        else:
            out += GMAP.get(t, [t])
    return out


def norm_mine(toks):
    out = []
    for t in toks:
        if t in ("|", ",", "."):
            continue
        out += MMAP.get(t, [t])
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def classify(ref, hyp):
    if hyp == ref + ["a"]:
        return "extra final schwa"
    if ref == hyp + ["a"]:
        return "missing final schwa"
    if len(hyp) == len(ref) + 1 and edit_distance(ref, hyp) == 1:
        return "extra medial schwa" if "a" in hyp else "insertion"
    if len(ref) == len(hyp) + 1 and edit_distance(ref, hyp) == 1:
        return "missing medial schwa" if "a" in ref else "deletion"
    if len(ref) == len(hyp):
        diffs = [(r, h) for r, h in zip(ref, hyp) if r != h]
        if len(diffs) == 1:
            return f"sub {diffs[0][1]}->{diffs[0][0]}"
    return "multiple"


def predict(word, unseen):
    """Rules only, or (unseen=True) the full pipeline with this word removed
    from the lexicon, simulating an out-of-vocabulary word honestly."""
    if not unseen:
        return _phonemize_word(word, use_lexicon=False)
    saved = LEXICON.pop(word, None)
    try:
        return _phonemize_word(word, use_lexicon=True)
    finally:
        if saved is not None:
            LEXICON[word] = saved


def run(lexicon, show, unseen=False):
    pairs = []
    for line in open(lexicon, encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        word, phones = line.rstrip("\n").split("\t")[:2]
        if word.isascii():
            continue
        pairs.append((unicodedata.normalize("NFC", word), norm_google(phones)))
    errs, phone_err, phone_tot, cats, examples = 0, 0, 0, Counter(), {}
    for word, ref in pairs:
        hyp = norm_mine(predict(word, unseen))
        d = edit_distance(ref, hyp)
        phone_err += d
        phone_tot += len(ref)
        if hyp != ref:
            errs += 1
            c = classify(ref, hyp)
            cats[c] += 1
            examples.setdefault(c, []).append((word, " ".join(ref), " ".join(hyp)))
    n = len(pairs)
    mode = "unseen-word pipeline (lexicon minus the word)" if unseen else "rules only"
    print(f"[{mode}] words: {n}   word accuracy: {100 * (1 - errs / n):.1f}%   phone error rate: {100 * phone_err / phone_tot:.2f}%")
    print("\nerror breakdown:")
    for c, k in cats.most_common(show):
        w, r, h = examples[c][0]
        print(f"  {k:6d}  {100 * k / n:5.1f}%  {c:28s} e.g. {w}: ref[{r}] ours[{h}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default=os.path.join(ROOT, "data", "lexicons", "google", "lexicon.tsv"))
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--unseen", action="store_true", help="score the full pipeline on words treated as unseen")
    a = ap.parse_args()
    run(a.lexicon, a.show, a.unseen)
