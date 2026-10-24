"""Devanagari -> Roman letters the way Nepali speakers write and say it in chat
("आज काठमाडौँमा मौसम धेरै राम्रो छ" -> "Aaja Kathmandu-ma mausam dherai ramro chha"),
not a letter-by-letter transliteration ("kāṭhmāḍau̐mā").

For showing people what to read or what a sentence sounds like, not a model input.
Rules: inherent "a" is silent at the end of a word after a single consonant (राम ->
ram), kept after a cluster (हुन्छ -> hunchha) and in one-letter words (छ -> chha, र ->
ra); -एर participles keep it (गरेर -> garera); व is "b" at the start of a word and after r;
क्ष -> chhy, ज्ञ -> gy; long and short i/u are the same; a word-final nasal on a vowel
is dropped (जान्छौँ -> janchhau); आ is "aa" at the start of a word, "a" inside it.
"""
import re

CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
        "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh",
        "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "w",
        "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "j", "ड़": "r",
        "ढ़": "rh", "फ़": "f"}
VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
       "ऋ": "ri", "ॠ": "ri"}
MAT = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
       "ॅ": "e", "ॉ": "o"}
HAL, NAS = "्", {"ं", "ँ"}
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# names people already know in Roman letters
NAMES = {"काठमाडौँ": "Kathmandu", "काठमाडौं": "Kathmandu", "नेपाल": "Nepal", "सगरमाथा": "Sagarmatha",
         "पोखरा": "Pokhara", "ललितपुर": "Lalitpur", "भक्तपुर": "Bhaktapur", "विराटनगर": "Biratnagar",
         "नाम्चे": "Namche", "अर्जन": "Arjan", "युनेस्को": "UNESCO"}
# common words whose spoken form the rules miss (final "a" kept, compounds)
WORDS = {"आज": "aaja", "दालभात": "dalbhat", "प्रधानमन्त्री": "pradhanmantri", "मनमोहक": "manmohak",
         "खेतबारी": "khetbari", "खानपान": "khanpan", "रंगीचंगी": "rangichangi", "बुबाआमा": "buba-aama",
         "अंकगणित": "ankaganit", "इन्टरनेट": "internet", "कम्प्युटर": "computer", "मोबाइल": "mobile",
         "डाक्टर": "doctor", "एप": "app", "फोन": "phone",
         "मनपर्ने": "manparne", "ल्याब्स": "labs", "ल्याब्सले": "labs-le"}
# case endings and plural: the stem is said as a word of its own (आकाशमा -> aakashma)
SUFFIXES = sorted(["मा", "को", "का", "की", "ले", "लाई", "बाट", "सँग", "सम्म", "हरू", "हरु", "भरि", "तिर", "देखि",
                   "माथि", "भित्र", "पछि", "अनुसार", "सहित", "मध्ये", "वटा"], key=len, reverse=True)


def _word(w, suffix_ok=True):
    w = w.replace("\u200d", "").replace("\u200c", "")
    if w in NAMES:
        return NAMES[w]
    if w in WORDS:
        return WORDS[w]
    for k, v in NAMES.items():  # name + case ending: काठमाडौँमा -> Kathmandu-ma
        if w.startswith(k) and len(w) > len(k) and w[len(k)] in CONS:
            return v + "-" + _word(w[len(k):])
    if suffix_ok:
        for suf in SUFFIXES:
            stem = w[:-len(suf)]
            if w.endswith(suf) and len(stem) >= 2 and stem[-1] != HAL and not (stem[-1] in MAT and len(stem) < 2):
                return _word(stem) + _word(suf, False)
    w = w.replace("क्ष", "\x01").replace("ज्ञ", "\x02")
    out, i, n = [], 0, len(w)
    while i < n:
        c = w[i]
        if c in ("\x01", "\x02") or c in CONS:
            if c == "\x01":
                r = "chhy"
            elif c == "\x02":
                r = "gy"
            elif c == "व" and (i == 0 or i + 1 == n or (out and out[-1].endswith(("r", "ra")))):  # विकास, प्रविधि, सम्भव
                r = "b"
            else:
                r = CONS[c]
            if r == "h" and out and out[-1].endswith("ng"):  # सिंह -> singh
                r = ""
                out[-1] += "h"
                if i + 1 == n:
                    i += 1
                    continue
            if i + 1 < n and w[i + 1] == "़":
                i += 1
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt == HAL:
                if r == "ng" and i + 2 < n and w[i + 2] in "कखगघ":  # जङ्गल -> jangal
                    r = "n"
                out.append(r)
                i += 2
                continue
            if nxt in MAT:
                out.append(r + MAT[nxt])
                i += 2
                continue
            # inherent a: silent at word end after a single consonant (not in 1-letter words, not after a cluster)
            end = i + 1 == n or (i + 2 == n and w[i + 1] in NAS)
            cluster = i >= 2 and w[i - 1] == HAL
            participle = i >= 1 and w[i - 1] == "े" and c == "र" and i + 1 == n
            silent = end and i > 0 and not cluster and not participle and c != "य"
            out.append(r + ("" if silent else "a"))
            i += 1
        elif c in VOW:
            v = VOW[c]
            out.append(v if i == 0 else ("a" if v == "aa" else v))
            i += 1
        elif c in NAS:
            # nasal: dropped at the end of the word, "m" before p/b/m, else "n"
            if i + 1 < n:
                nx = w[i + 1]
                out.append("m" if nx in "पफबभम" else "ng" if nx == "ह" else "n")
            i += 1
        elif c == "ः":
            i += 1
        else:
            out.append(c.translate(DIGITS))
            i += 1
    return "".join(out)


def romanize(text):
    text = text.replace("\u200d", "").replace("\u200c", "")
    parts = re.split(r"([ऀ-ॣ०-ॿ]+)", text.replace("।", ".").replace("॥", "."))
    s = "".join(_word(p) if re.match(r"[ऀ-ॿ]", p) else p for p in parts)
    s = re.sub(r"\s+([.,?!])", r"\1", s)
    return re.sub(r"(^|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s.strip())


if __name__ == "__main__":
    import sys
    for line in sys.argv[1:] or sys.stdin:
        print(romanize(line.strip()))
