"""GPU-free prep: (1) emit the phoneme vocabulary the trainer needs, and
(2) generate a G2P review sheet for a native speaker to sanity-check.

Usage:
    python build_vocab_and_review.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend import g2p, normalize
from frontend.g2p import PHONEME_INVENTORY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHONEMES = PHONEME_INVENTORY
SPECIAL = ["<pad>", "<unk>", "|", ",", "."]  # pad, unknown, word boundary, short/long pause

# Common Nepali words spanning tricky features: schwa retention, clusters,
# retroflex vs dental, affricates, nasalization, long vowels, loanwords.
REVIEW_WORDS = [
    "नमस्ते", "धन्यवाद", "नेपाल", "काठमाडौँ", "राम्रो", "मान्छे", "पानी", "खाना",
    "घर", "किताब", "विद्यालय", "अस्पताल", "सरकार", "प्रधानमन्त्री", "छोरा", "छोरी",
    "आमा", "बुबा", "दाइ", "बहिनी", "पहाड", "हिमाल", "गाउँ", "सहर",
    "बिहान", "साँझ", "आज", "भोलि", "हिजो", "मौसम", "घाम", "वर्षा",
    "एक", "दुई", "तीन", "चार", "पाँच", "दश", "सय", "हजार",
    "कम्प्युटर", "मोबाइल", "इन्टरनेट",  # English loanwords in Devanagari
    "ज्ञान", "क्षमता", "त्रिशूल",        # conjuncts ज्ञ क्ष त्र
    "अंक", "रंग", "सिंह",               # anusvara / nasal
]

REVIEW_SENTENCES = [
    "आज मौसम धेरै राम्रो छ।",
    "मेरो नाम राम हो।",
    "नेपाल एक सुन्दर देश हो।",
    "रू १,२५० तिर्नुपर्छ।",
    "९० प्रतिशत मानिस सहमत छन्।",
]


def main():
    # 1) vocab.txt
    vocab_path = os.path.join(ROOT, "configs", "phoneme_vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok in SPECIAL + PHONEMES:
            f.write(tok + "\n")
    print(f"wrote {len(SPECIAL) + len(PHONEMES)} tokens -> {vocab_path}")

    # 2) review sheet
    os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
    review_path = os.path.join(ROOT, "review", "g2p_review.tsv")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("word\tg2p_phonemes\tOK?(native check)\tcorrection\n")
        for w in REVIEW_WORDS:
            f.write(f"{w}\t{' '.join(g2p(w))}\t\t\n")
        f.write("\n# sentences (normalized + phonemized)\n")
        f.write("sentence\tnormalized\tg2p\tOK?\n")
        for s in REVIEW_SENTENCES:
            norm = normalize(s)
            f.write(f"{s}\t{norm}\t{' '.join(g2p(norm))}\t\n")
    print(f"wrote review sheet ({len(REVIEW_WORDS)} words, {len(REVIEW_SENTENCES)} sentences) -> {review_path}")
    print("\nnative-speaker task: mark OK? column, add corrections -> they become LEXICON entries in frontend/g2p.py")


if __name__ == "__main__":
    main()
