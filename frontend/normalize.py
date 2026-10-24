"""Nepali text normalization for TTS.

Turns raw Nepali text into spoken-form Devanagari (plus any Latin words) for
the G2P. Handles digits in both scripts, cardinals (lakh/crore), decimals,
ordinals, dates (BS and AD), times, fractions, ranges, negatives, phone
numbers, currency (amount spoken first), units, abbreviations, URLs/emails,
and zero-width characters. This layer is where most Nepali TTS fails.
"""
import re
import unicodedata

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_DEV = "ऀ-ॿ"
_NO_LETTER_BEFORE = rf"(?<![{_DEV}A-Za-z])"
_NO_DEV_AFTER = rf"(?![{_DEV}])"

# 0-99 spelled out in Nepali (standard forms)
SMALL = [
    "शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ",
    "दश", "एघार", "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र", "सत्र", "अठार", "उन्नाइस",
    "बीस", "एक्काइस", "बाइस", "तेइस", "चौबिस", "पच्चिस", "छब्बिस", "सत्ताइस", "अठ्ठाइस", "उनन्तिस",
    "तीस", "एकतिस", "बत्तिस", "तेत्तिस", "चौँतिस", "पैँतिस", "छत्तिस", "सैँतिस", "अठतिस", "उनन्चालिस",
    "चालिस", "एकचालिस", "बयालिस", "त्रिचालिस", "चवालिस", "पैँतालिस", "छयालिस", "सतचालिस", "अठचालिस", "उनन्चास",
    "पचास", "एकाउन्न", "बाउन्न", "त्रिपन्न", "चवन्न", "पचपन्न", "छपन्न", "सन्ताउन्न", "अन्ठाउन्न", "उनान्साठी",
    "साठी", "एकसट्ठी", "बयसट्ठी", "त्रिसट्ठी", "चौंसट्ठी", "पैंसट्ठी", "छयसट्ठी", "सतसट्ठी", "अठसट्ठी", "उनन्सत्तरी",
    "सत्तरी", "एकहत्तर", "बहत्तर", "त्रिहत्तर", "चौहत्तर", "पचहत्तर", "छयहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "असी", "एकासी", "बयासी", "त्र्यासी", "चौरासी", "पचासी", "छयासी", "सतासी", "अठासी", "उनानब्बे",
    "नब्बे", "एकानब्बे", "बयानब्बे", "त्र्यानब्बे", "चौरानब्बे", "पन्चानब्बे", "छयानब्बे", "सन्तानब्बे", "अन्ठानब्बे", "उनान्सय",
]

_SCALES = [(10**11, "खरब"), (10**9, "अरब"), (10**7, "करोड"), (10**5, "लाख"), (10**3, "हजार")]


def number_to_words(n: int) -> str:
    """Integer -> Nepali words using the Indian (lakh/crore) system."""
    if n < 0:
        return "ऋणात्मक " + number_to_words(-n)
    if n < 100:
        return SMALL[n]
    if n < 1000:
        h, r = divmod(n, 100)
        return SMALL[h] + " सय" + (" " + number_to_words(r) if r else "")
    for value, name in _SCALES:
        if n >= value:
            q, r = divmod(n, value)
            return number_to_words(q) + " " + name + (" " + number_to_words(r) if r else "")
    return SMALL[0]


def _decimal_to_words(token: str) -> str:
    """'3.05' -> 'तीन दशमलव शून्य पाँच'; '.5' -> 'शून्य दशमलव पाँच'."""
    whole, _, frac = token.partition(".")
    words = number_to_words(int(whole)) if whole else SMALL[0]
    if frac:
        words += " दशमलव " + " ".join(SMALL[int(d)] for d in frac)
    return words


_ORD_IRREGULAR = {1: "पहिलो", 2: "दोस्रो", 3: "तेस्रो", 4: "चौथो", 5: "पाँचौँ",
                  6: "छैटौँ", 7: "सातौँ", 8: "आठौँ", 9: "नवौँ", 10: "दशौँ"}


def ordinal_to_words(n: int) -> str:
    if n in _ORD_IRREGULAR:
        return _ORD_IRREGULAR[n]
    w = number_to_words(n)
    return w + ("ौँ" if "क" <= w[-1] <= "ह" else "औँ")


_MONTHS_BS = [None, "बैशाख", "जेठ", "असार", "साउन", "भदौ", "असोज",
              "कात्तिक", "मंसिर", "पुष", "माघ", "फागुन", "चैत"]
_MONTHS_AD = [None, "जनवरी", "फेब्रुअरी", "मार्च", "अप्रिल", "मे", "जुन",
              "जुलाई", "अगस्ट", "सेप्टेम्बर", "अक्टोबर", "नोभेम्बर", "डिसेम्बर"]

_ABBREV = {
    "वि.सं.": "विक्रम संवत्", "ई.सं.": "ईस्वी संवत्", "ई.पू.": "ईसा पूर्व",
    "कि.मि.": "किलोमिटर", "कि.ग्रा.": "किलोग्राम", "मि.ली.": "मिलिलिटर", "से.मि.": "सेन्टिमिटर",
    "डा.": "डाक्टर", "नं.": "नम्बर", "प्रा.": "प्राध्यापक",
    "e.g.": "उदाहरणका लागि", "i.e.": "अर्थात्", "etc.": "आदि",
    "Dr.": "डाक्टर", "Mr.": "मिस्टर", "Mrs.": "मिसेस",
}
_ABBREV_RE = re.compile(_NO_LETTER_BEFORE + "("
                        + "|".join(re.escape(k) for k in sorted(_ABBREV, key=len, reverse=True)) + ")")

# (symbol pattern, spoken name). Amount is spoken before the currency name.
_CURRENCIES = [
    (r"NRs\.?|NPR|ने\.\s?रू\.?|रू\.?|रु\.?|Rs\.?|RS\.?", "रुपैयाँ"),
    (r"₹|INR|भा\.\s?रू\.?", "भारतीय रुपैयाँ"),
    (r"US\$|USD|\$", "डलर"),
    (r"€|EUR", "युरो"),
    (r"£|GBP", "पाउन्ड"),
]
_CURRENCY_AMOUNT = [(re.compile(_NO_LETTER_BEFORE + f"(?:{p})" + r"\s*(\d+(?:\.\d+)?)"), name)
                    for p, name in _CURRENCIES]
_CURRENCY_BARE = [(re.compile(rf"{_NO_LETTER_BEFORE}(?:रू|रु)\.?{_NO_DEV_AFTER}"), "रुपैयाँ"),
                  (re.compile(r"(?<![A-Za-z])Rs\.?(?![A-Za-z])"), "रुपैयाँ"),
                  (re.compile(r"₹"), "भारतीय रुपैयाँ"), (re.compile(r"\$"), "डलर"),
                  (re.compile(r"€"), "युरो"), (re.compile(r"£"), "पाउन्ड")]

_UNITS = {"km/h": "किलोमिटर प्रति घण्टा", "kmph": "किलोमिटर प्रति घण्टा", "km": "किलोमिटर",
          "kg": "केजी", "cm": "सेन्टिमिटर", "mm": "मिलिमिटर", "ml": "मिलिलिटर", "m": "मिटर",
          "g": "ग्राम", "l": "लिटर", "L": "लिटर", "°C": "डिग्री सेल्सियस", "°F": "डिग्री फरेनहाइट",
          "°": "डिग्री", "GB": "जीबी", "MB": "एमबी", "kW": "किलोवाट", "MW": "मेगावाट"}
_UNITS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
                       + r")(?![A-Za-z])")

_SYMBOLS = [(re.compile(r"%"), " प्रतिशत "), (re.compile(r"&"), " र "),
            (re.compile(r"\+"), " जोड "), (re.compile(r"="), " बराबर ")]

_FRACTIONS = {(1, 2): "आधा", (1, 4): "एक चौथाइ", (3, 4): "तीन चौथाइ"}
_URL_SPOKEN = {".": "डट", "@": "एट", "/": "स्ल्यास", "-": "ड्यास", "_": "अन्डरस्कोर", ":": ""}
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?:/[^\s]*)?")


def _digits_one_by_one(m):
    s = m.group(0)
    prefix = "प्लस " if s.startswith("+") else ""
    return " " + prefix + " ".join(SMALL[int(d)] for d in s if d.isdigit()) + " "


def _speak_url(m):
    s = re.sub(r"^https?://", "", m.group(0))
    out = []
    for part in re.split(r"([./@:_-])", s):
        if part in _URL_SPOKEN:
            out.append(_URL_SPOKEN[part])
        elif part.lower() == "www":
            out.append("डब्लु डब्लु डब्लु")
        elif part:
            out.append(part)
    return " " + " ".join(p for p in out if p) + " "


def _date(y, m, d, tail):
    y, m, d = int(y), int(m), int(d)
    if not (1 <= m <= 12 and 1 <= d <= 32):
        return None
    if y >= 2050:  # Bikram Sambat (current BS year is ~2083)
        return f" {y} साल {_MONTHS_BS[m]} {d} गते "
    return f" सन् {y} {_MONTHS_AD[m]} {d} "


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        return m.group(0)
    return f" {h} बजे " if mi == 0 else f" {h} बजेर {mi} मिनेट "


def _fraction(m):
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return m.group(0)
    return " " + _FRACTIONS.get((a, b), f"{a} बटा {b}") + " "


def _replace_numbers(text: str) -> str:
    def repl(m):
        tok = m.group(0)
        return " " + (_decimal_to_words(tok) if "." in tok else number_to_words(int(tok))) + " "
    return re.sub(r"\d+(?:\.\d+)?|(?<![\dA-Za-z])\.\d+", repl, text)


def normalize(text: str, english: bool = True, nepal: bool = False) -> str:
    """Normalize raw Nepali text into spoken form for the G2P. Idempotent."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH).translate(_DEV_DIGITS)
    text = _ABBREV_RE.sub(lambda m: f" {_ABBREV[m.group(1)]} ", text)
    text = re.sub(r"(?<![A-Za-z])No\.\s*(?=\d)", " नम्बर ", text)
    text = _EMAIL_RE.sub(_speak_url, text)
    text = _URL_RE.sub(_speak_url, text)
    # phone numbers / long IDs, digit by digit (Nepal mobiles are 10 digits)
    text = re.sub(r"\+\d{6,}|(?<!\d)\d{2,4}-\d{6,8}(?!\d)|\d{10,}", _digits_one_by_one, text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 1,25,000 -> 125000
    text = re.sub(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)(\s*गते)?",
                  lambda m: _date(m.group(1), m.group(2), m.group(3), m.group(4)) or m.group(0), text)
    text = re.sub(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?!\d)(\s*गते)?",
                  lambda m: _date(m.group(3), m.group(2), m.group(1), m.group(4)) or m.group(0), text)
    text = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)(?:\s*बजे)?", _time, text)
    text = re.sub(r"(?<![\d/])(\d+)/(\d+)(?![\d/])", _fraction, text)
    text = re.sub(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)(?![\d.])", r" \1 देखि \2 ", text)
    text = re.sub(r"(?:(?<=\s)|^)[-−](?=\d)", " माइनस ", text)
    text = re.sub(rf"(\d+)\s?(?:ला|लो|रा|रो|थो|औँ|औं|ौँ|ौं|st|nd|rd|th)(?![A-Za-z{_DEV}])",
                  lambda m: f" {ordinal_to_words(int(m.group(1)))} ", text)
    for pat, name in _CURRENCY_AMOUNT:
        text = pat.sub(lambda m, name=name: f" {m.group(1)} {name} ", text)
    for pat, name in _CURRENCY_BARE:
        text = pat.sub(f" {name} ", text)
    text = _UNITS_RE.sub(lambda m: f" {m.group(1)} {_UNITS[m.group(2)]} ", text)
    for pat, rep in _SYMBOLS:
        text = pat.sub(rep, text)
    text = _replace_numbers(text)
    # nepal=True: Nepal pronunciation respelling (frontend.pron.nepal_accent) of the Nepali words only,
    # before English words are converted, so "wifi" -> वाइफाइ keeps its व
    if nepal:
        from .pron import nepal_accent
        text = nepal_accent(text)
    # English words -> Devanagari as Nepali speakers say them (the TTS saw almost no Latin script)
    if english:
        from .english import english_to_devanagari
        text = english_to_devanagari(text)
    return re.sub(r"\s+", " ", text).strip()


# Latin-script runs are English code-switch.
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


def split_codeswitch(text: str):
    """Yield (segment, is_latin) tuples splitting Nepali vs embedded English."""
    idx = 0
    for m in _LATIN_RUN.finditer(text):
        if m.start() > idx:
            yield text[idx:m.start()], False
        yield m.group(0), True
        idx = m.end()
    if idx < len(text):
        yield text[idx:], False
