"""Edge-case probe for the Nepali frontend. Prints normalize/g2p output for
tricky inputs and flags invariant violations:
  CRASH    exception
  DROP     a word with letters/digits produced no phonemes (silent loss)
  BADTOK   token outside the model vocabulary (would KeyError in F5 encoding)
  NONIDEM  normalize(normalize(x)) != normalize(x)
  PAIR     a minimal pair collapses to identical phonemes
Usage: python tests/edge_probe.py [--quiet]
"""
import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from frontend import encode_for_f5, g2p, normalize
from frontend.g2p import PHONEME_INVENTORY

CASES = {
    "numbers": ["०", "007", "२५", "1,00,000", "१२,३४,५६,७८९", "999999999999", "3.05", "०.५", ".5",
                "-5", "५-१०", "१२.५%", "२०वटा", "५ जना", "1st", "२रो", "३रो", "४थो", "१ला"],
    "dates_times": ["२०८१/०५/१२", "2024-05-12", "१२/०५/२०२४", "वि.सं. २०८१", "५:३०", "9:00 बजे", "१०:०५", "१/२"],
    "currency_units": ["रु २५०", "रु. ५००", "रू.१००", "Rs.50", "$100", "USD 20", "₹ ५", "५ कि.मि.", "१० kg", "३५°C"],
    "abbrev": ["डा. राम", "श्री हरि", "नं. ५", "प्रा. डा. केसी", "e.g.", "etc."],
    "english": ["AI", "NASA", "USA", "COVID-19", "5G", "iPhone", "YouTube", "Facebook मा post गर",
                "user's", "café", "www.google.com", "abc@gmail.com", "OK", "ok", "Hello!"],
    "punctuation": ["राम्रो छ|", "हो।।", "के?!", "हो...", "हो…", "\"राम\" भन्यो", "(नेपाल)", "आमा-बुबा",
                    "र/वा", "—", "हो ; तर", "  ", ""],
    "unicode": ["गर्‍यो", "तिर्नुपर्‍यो", "क्‍ष", "गाउँ", "गाउं", "तपाई", "तपाईं", "महिलाहरु", "महिलाहरू",
                "दुःख", "ॐ", "ळ", "ऋण", "कृष्ण", "श्रेष्ठ", "क्क्र", "ज़िरो", "ङ", "🙂 राम्रो", "नेपाल​"],
    "hindi_like": ["क़लम", "ख़बर", "ग़ज़ल"],
    "long": ["अन्तर्राष्ट्रियकरणका", "प्रधानमन्त्रीज्यूलाई"],
}
MINIMAL_PAIRS = [("कल", "खल"), ("तल", "थल"), ("पाल", "फाल"), ("पाल", "बाल"), ("बाल", "भाल"),
                 ("तल", "टल"), ("दल", "डल"), ("चपल", "चप्पल"), ("कप", "काँप"), ("बस", "बाँस"),
                 ("चाल", "छाल")]

VOCAB = set(PHONEME_INVENTORY) | {"|", ",", "."}


def has_content(w):
    return any(unicodedata.category(c)[0] in "LN" for c in w)


def probe(quiet=False):
    flags = []
    for cat, items in CASES.items():
        if not quiet:
            print(f"\n## {cat}")
        for x in items:
            try:
                n = normalize(x)
                t = g2p(n)
                enc = encode_for_f5(t)
            except Exception as e:
                flags.append(("CRASH", x, repr(e)))
                print(f"  CRASH {x!r}: {e!r}")
                continue
            bad = [tok for tok in t if tok not in VOCAB]
            if bad:
                flags.append(("BADTOK", x, bad))
            if normalize(n) != n:
                flags.append(("NONIDEM", x, (n, normalize(n))))
            for w in n.split():
                if has_content(w) and not g2p(w):
                    flags.append(("DROP", x, w))
            if not quiet:
                print(f"  {x!r:28s} -> {n!r:40s} {' '.join(t)}")
    for a, b in MINIMAL_PAIRS:
        if g2p(a) == g2p(b):
            flags.append(("PAIR", f"{a}/{b}", g2p(a)))
    print(f"\n{len(flags)} flags")
    for f in flags:
        print("  ", *f)
    return flags


if __name__ == "__main__":
    probe("--quiet" in sys.argv)
