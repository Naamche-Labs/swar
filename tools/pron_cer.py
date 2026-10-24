"""Pronunciation CER: character error after folding spellings that Nepali speech does not distinguish, applied
identically to the reference and to every system's ASR transcript. Standard CER counts these as errors although
the sounds are the same: श/ष/स, long vs short i/u (ी/ि, ू/ु), chandrabindu vs anusvara, a nasal consonant + halanta
before a consonant vs anusvara (ङ्ग/ंग), व/ब, ण/न, ऋ/रि, zero-width joiners, word-final halanta, and spacing
(सकल जनसंख्या / सकलजन सङ्ख्या), and English words in the reference (written in Devanagari by every ASR; converted
with frontend.english for all systems alike). Use next to standard CER, never instead of it.
Usage: python tools/pron_cer.py BENCH_JSON [BENCH_JSON ...]   (rescore_deva.py output: rows with ref/hyp)"""
import json
import re
import sys
import unicodedata

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from eval_heldout import ed  # noqa: E402

FOLD = [("श", "स"), ("ष", "स"), ("ी", "ि"), ("ई", "इ"), ("ू", "ु"), ("ऊ", "उ"), ("ँ", "ं"), ("व", "ब"), ("ण", "न"),
        ("ृ", "्रि"), ("ऋ", "रि"), ("‍", ""), ("‌", ""), ("़", "")]


def fold(s):
    s = unicodedata.normalize("NFC", s)
    if re.search(r"[A-Za-z]", s):  # English words in a reference: every ASR writes them in Devanagari
        sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
        from frontend.english import english_to_devanagari
        s = english_to_devanagari(s)
    for a, b in FOLD:
        s = s.replace(a, b)
    s = re.sub(r"[ङञनम]्(?=[क-ह])", "ं", s)
    s = re.sub(r"्(?=\s|$)", "", s)
    return re.sub(r"\s+", "", s)


def pcer(rows):
    return 100 * sum(ed(fold(r["ref"]), fold(r["hyp"])) / max(1, len(fold(r["ref"]))) for r in rows) / max(1, len(rows))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        d = json.load(open(path))
        for name, r in d.items():
            std = r.get("mean_file_cer")
            print(f"PCER {name:28s} standard {std:6.2f}  pronunciation {pcer(r['rows']):6.2f}")
