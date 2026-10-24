"""English (Latin-script) words inside Nepali text -> Devanagari, as Nepali speakers say them.

The TTS was trained on Devanagari only, so "मेरो laptop को battery" reached it as letters it
had barely seen and came out garbled. Order of lookup:
  1. LOAN: common loanwords in their usual Nepali spelling (battery -> ब्याट्री)
  2. ALL-CAPS 2-6 letters: spelled as letter names (ATM -> एटीएम, OTP -> ओटीपी)
  3. CMU Pronouncing Dictionary (data/cmudict.dict.gz, BSD licence) mapped to Nepali sounds,
     with the ambiguous sounds (unstressed vowels, "o" in shop, t/d, sh) expanded into every
     plausible Devanagari spelling; the one Nepali writers use most wins, counted over 8.4M
     sentences of Nepali Wikipedia + news (data/ne_wordfreq.tsv.gz): market -> मार्केट (1,067
     uses; मार्कट 0), idea -> आइडिया, shop -> सप. No match -> the default mapping
     (English t/d -> ट/ड, v -> भ, z -> ज, sh -> स, "a" in cat -> ्या: laptop -> ल्यापटप)
  4. otherwise: romanized Nepali rules (pathaunu -> पठाउनु)
"""
import gzip
import os
import re

LOAN = {
    "battery": "ब्याट्री", "laptop": "ल्यापटप", "charger": "चार्जर", "mobile": "मोबाइल", "phone": "फोन",
    "computer": "कम्प्युटर", "internet": "इन्टरनेट", "office": "अफिस", "meeting": "मिटिङ", "online": "अनलाइन",
    "email": "इमेल", "video": "भिडियो", "app": "एप", "apps": "एपहरू", "bus": "बस", "school": "स्कुल",
    "college": "कलेज", "hospital": "हस्पिटल", "doctor": "डाक्टर", "ok": "ओके", "okay": "ओके", "sorry": "सरी",
    "please": "प्लिज", "thank": "थ्याङ्क", "thanks": "थ्याङ्क्स", "you": "यु", "password": "पासवर्ड",
    "whatsapp": "ह्वाट्सएप", "facebook": "फेसबुक", "youtube": "युट्युब", "google": "गुगल", "tiktok": "टिकटक",
    "wifi": "वाइफाइ", "message": "म्यासेज", "call": "कल", "photo": "फोटो", "camera": "क्यामेरा",
    "car": "कार", "bike": "बाइक", "taxi": "ट्याक्सी", "ticket": "टिकट", "hotel": "होटल", "restaurant": "रेस्टुरेन्ट",
    "coffee": "कफी", "tea": "टी", "team": "टिम", "project": "प्रोजेक्ट", "company": "कम्पनी", "software": "सफ्टवेयर",
    "website": "वेबसाइट", "account": "एकाउन्ट", "bank": "बैंक", "card": "कार्ड", "cash": "क्यास", "money": "मनी",
    "cool": "कुल", "nice": "नाइस", "good": "गुड", "very": "भेरी", "actually": "एक्चुअली", "like": "लाइक",
    "but": "बट", "so": "सो", "yeah": "याह", "and": "एन्ड", "is": "इज", "the": "द", "a": "अ", "to": "टु",
    "machine": "मेसिन", "ai": "एआई", "data": "डाटा", "server": "सर्भर", "update": "अपडेट", "download": "डाउनलोड",
    "online": "अनलाइन", "offline": "अफलाइन", "booking": "बुकिङ", "delivery": "डेलिभरी", "order": "अर्डर",
    "potential": "पोटेन्सियल", "unesco": "युनेस्को", "xiaomi": "साओमी", "samsung": "साम्सङ", "iphone": "आइफोन",
    "nepal": "नेपाल", "nepali": "नेपाली", "kathmandu": "काठमाडौँ", "pokhara": "पोखरा", "lalitpur": "ललितपुर",
    "bhaktapur": "भक्तपुर", "biratnagar": "विराटनगर", "chitwan": "चितवन", "dharan": "धरान", "butwal": "बुटवल",
    "baneshwor": "बानेश्वर", "pulchowk": "पुल्चोक", "chowk": "चोक", "thamel": "ठमेल", "everest": "एभरेस्ट",
    "india": "इन्डिया", "bangalore": "बैङ्गलोर", "goa": "गोवा", "delhi": "दिल्ली",
    "com": "कम", "org": "अर्ग", "np": "एनपी", "www": "डब्लु डब्लु डब्लु",
    "shirt": "सर्ट", "tshirt": "टिसर्ट", "cash": "क्यास", "cashback": "क्यासब्याक", "document": "डकुमेन्ट",
    "airport": "एयरपोर्ट", "exam": "एक्जाम", "engineer": "इन्जिनियर", "assignment": "एसाइनमेन्ट",
    "forecast": "फोरकास्ट", "plan": "प्लान", "case": "केस", "ward": "वार्ड",
    "covid": "कोभिड", "nasa": "नासा", "unicef": "युनिसेफ", "nato": "नाटो", "saarc": "सार्क",
    "khalti": "खल्ती", "esewa": "इसेवा", "daraz": "दराज", "ncell": "एनसेल", "ntc": "एनटीसी", "worldlink": "वर्ल्डलिङ्क",
}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Google's Nepali lexicon (CC-BY 4.0, via Ampixa, MIT) has ~5.6k Latin-script entries (English loanwords and
# romanized Nepali) with their pronunciation as spoken in Nepal, syllables separated by "."
_G_CONS = {"p": "प", "ph": "फ", "b": "ब", "bh": "भ", "t": "त", "th": "थ", "d": "द", "dh": "ध", "tx": "ट", "txh": "ठ",
           "dx": "ड", "dxh": "ढ", "k": "क", "kh": "ख", "g": "ग", "gh": "घ", "ts": "च", "tsh": "छ", "dz": "ज",
           "dzh": "झ", "m": "म", "n": "न", "ng": "ङ", "r": "र", "s": "स", "sh": "श", "sx": "ष", "h": "ह", "l": "ल",
           "y": "य", "w": "व", "f": "फ", "z": "ज", "ny": "ञ", "nx": "ण"}
_G_VOW = {"ax": ("अ", ""), "aa": ("आ", "ा"), "i": ("इ", "ि"), "u": ("उ", "ु"), "e": ("ए", "े"), "o": ("ओ", "ो"),
          "axj": ("ऐ", "ै"), "axw": ("औ", "ौ"), "aaj": ("आइ", "ाइ"), "aaw": ("आउ", "ाउ"), "oj": ("ओइ", "ोइ"),
          "ew": ("एउ", "ेउ"), "ow": ("ओउ", "ोउ")}


def google_phones_to_devanagari(phones):
    """'m aa r . k e tx' -> मार्केट. Consonants with no vowel before the next consonant take a halanta; a
    word-final bare consonant does not (Nepali spelling drops it anyway)."""
    toks = phones.split()
    out, prev_cons, boundary, last = [], False, False, None
    for i, t in enumerate(toks):
        if t == ".":
            boundary = True
            continue
        nas = t.endswith("n") and t[:-1] in _G_VOW
        base = t[:-1] if nas else t
        if base in _G_VOW:
            ind, mat = _G_VOW[base]
            out.append((mat if prev_cons else ind) + ("ँ" if nas else ""))
            prev_cons = False
        elif t in _G_CONS:
            # across a syllable break only nasals, liquids and s join the next consonant (मार्केट, इन्टरनेट,
            # but फिडब्याक, ल्यापटप)
            if prev_cons and (not boundary or last in ("m", "n", "ng", "r", "l", "s")):
                out.append(HAL)
            out.append(_G_CONS[t])
            prev_cons = True
        boundary, last = False, t
    return "".join(out)


def _load_tables():
    loan, google = {}, {}
    p = os.path.join(_ROOT, "data", "loanwords.tsv")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if line.strip() and not line.startswith("#") and "\t" in line:
                en, dv = line.rstrip("\n").split("\t")[:2]
                loan[en.strip().lower()] = dv.strip()
    p = os.path.join(_ROOT, "data", "lexicons", "ampixa", "loanwords_latin.tsv")
    if os.path.exists(p):
        for i, line in enumerate(open(p, encoding="utf-8")):
            f = line.rstrip("\n").split("\t")
            if i and len(f) > 2 and f[2]:
                google.setdefault(f[0].lower(), google_phones_to_devanagari(f[2]))
    return loan, google


LETTERS = {"a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ", "g": "जी", "h": "एच", "i": "आई",
           "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन", "o": "ओ", "p": "पी", "q": "क्यू", "r": "आर",
           "s": "एस", "t": "टी", "u": "यू", "v": "भी", "w": "डब्लु", "x": "एक्स", "y": "वाई", "z": "जेड"}

CONS = {"B": "ब", "CH": "च", "D": "ड", "DH": "द", "F": "फ", "G": "ग", "HH": "ह", "JH": "ज", "K": "क", "L": "ल",
        "M": "म", "N": "न", "NG": "ङ", "P": "प", "R": "र", "S": "स", "SH": "स", "T": "ट", "TH": "थ", "V": "भ",
        "W": "व", "Y": "य", "Z": "ज", "ZH": "ज"}
# vowel: (independent form, matra after a consonant)
VOW = {"AA": ("आ", "ा"), "AE": ("ए", "्या"), "AH": ("अ", ""), "AO": ("अ", ""), "AW": ("आउ", "ाउ"),
       "AY": ("आइ", "ाइ"), "EH": ("ए", "े"), "ER": ("अर", "र"), "EY": ("ए", "े"), "IH": ("इ", "ि"),
       "IY": ("ई", "ी"), "OW": ("ओ", "ो"), "OY": ("ओइ", "ोइ"), "UH": ("उ", "ु"), "UW": ("ऊ", "ु")}
HAL = "्"

_CMU = None
_FREQ = None
# alternatives per ARPAbet sound, as (independent, matra) or consonant letters; first = default
ALT_V = {"AA": [("आ", "ा"), ("अ", "")], "AH": [("अ", ""), ("ए", "े"), ("आ", "ा"), ("इ", "ि"), ("उ", "ु")],
         "IH": [("इ", "ि"), ("ए", "े"), ("ई", "ी")],
         "IY": [("ई", "ी"), ("इ", "ि")], "UW": [("ऊ", "ु"), ("ऊ", "ू")], "AO": [("अ", ""), ("ओ", "ो"), ("आ", "ा")],
         "ER": [("अर", "र")], "AE": [("ए", "्या"), ("ए", "े"), ("ऐ", "ै"), ("आ", "ा")], "EY": [("ए", "े")], "EH": [("ए", "े")],
         "OW": [("ओ", "ो")], "UH": [("उ", "ु")], "AY": [("आइ", "ाइ")], "AW": [("आउ", "ाउ")], "OY": [("ओइ", "ोइ")]}
ALT_C = {"T": ["ट", "त"], "D": ["ड", "द"], "SH": ["स", "श"], "V": ["भ", "व"], "NG": ["ङ", "ङ्ग"], "Z": ["ज"],
         "Y": ["य", "\x03"]}  # \x03: consonant + ि + य, no halanta (genius -> जिनियस)


def _freq():
    global _FREQ
    if _FREQ is None:
        _FREQ = {}
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ne_wordfreq.tsv.gz")
        if os.path.exists(p):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    w, n = line.rstrip("\n").split("\t")
                    _FREQ[w] = int(n)
    return _FREQ


def candidates(phones, cap=2048):
    """All plausible Devanagari spellings of an ARPAbet sequence (Nepali conventions)."""
    import itertools
    slots = []
    stress = [re.sub(r"\D", "", p) for p in phones]
    phones = [re.sub(r"\d", "", p) for p in phones]
    for i, ph in enumerate(phones):
        if ph in ALT_V:
            alts = list(ALT_V[ph])
            if ph == "AH" and stress[i] == "1":  # stressed AH (cup, from, bus) is the inherent a
                alts = alts[:1]
            if i > 0 and phones[i - 1] in ("IY", "IH", "EY", "AY"):  # y-glide: idea -> आइडिया, radio -> रेडियो
                alts += [("य" + m, "य" + m) for _, m in ALT_V[ph]] + ([("या", "या")] if ph == "AH" else [])
            elif ph == "AH" and i > 0 and phones[i - 1] in ALT_V:
                alts += [("या", "या"), ("य", "य")]
            slots.append(("V", ph, alts))
        else:
            slots.append(("C", ph, ALT_C.get(ph, [CONS[ph]])))
    hal = joins(phones)
    sizes = 1
    for s in slots:
        sizes *= len(s[2])
    if sizes > cap:  # too ambiguous: only vowels vary
        slots = [(t, ph, a if t == "V" else a[:1]) for t, ph, a in slots]
    for combo in itertools.product(*[s[2] for s in slots]):
        out, prev_cons = [], False
        for idx, ((t, ph, _), choice) in enumerate(zip(slots, combo)):
            if t == "V":
                ind, mat = choice
                out.append(mat if prev_cons else ind)
                prev_cons = ph == "ER"
            elif choice == "\x03":
                out.append("िय" if prev_cons else "इय")
                prev_cons = True
            else:
                if prev_cons and (slots[idx - 1][0] == "V" or hal[idx - 1]):
                    out.append(HAL)
                out.append(choice)
                prev_cons = True
        yield "".join(out)


_NATIVE = None


def _native():
    """Devanagari words of Google's Nepali lexicon: the frequency picker must not turn an English word into a
    Nepali word or name (awesome -> असम 'Assam', shirt -> शर्त 'condition')."""
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = set()
        p = os.path.join(_ROOT, "data", "lexicons", "google", "lexicon.tsv")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                w = line.split("\t", 1)[0]
                if w and re.match(r"[\u0900-\u097F]", w):
                    _NATIVE.add(w)
    return _NATIVE


def best_spelling(phones):
    f, nat = _freq(), _native()
    default = arpabet_to_devanagari(phones)
    best, n = None, 4  # needs >= 5 uses to beat the default mapping
    for c in candidates(phones):
        k = f.get(c, 0)
        if k > n and (c not in nat or c == default):
            best, n = c, k
    return best or arpabet_to_devanagari(phones)


def _cmu():
    global _CMU
    if _CMU is None:
        _CMU = {}
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cmudict.dict.gz")
        if os.path.exists(p):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    w, *ph = line.split("#")[0].split()
                    if "(" not in w:  # first pronunciation only; stress digits kept (AH1 vs AH0)
                        _CMU[w] = ph
    return _CMU


# English syllable onsets (maximal onset principle): a medial consonant cluster splits so that the second
# syllable starts with the longest legal onset; no halanta joins the coda to the onset (feedback -> फिडब्याक,
# deadline -> डेडलाइन), halantas stay inside onsets and codas (station -> स्टेसन, next -> नेक्स्ट).
ONSETS = {("P", "L"), ("P", "R"), ("B", "L"), ("B", "R"), ("T", "R"), ("D", "R"), ("K", "L"), ("K", "R"), ("G", "L"),
          ("G", "R"), ("F", "L"), ("F", "R"), ("S", "L"), ("S", "M"), ("S", "N"), ("S", "P"), ("S", "T"), ("S", "K"),
          ("S", "W"), ("T", "W"), ("D", "W"), ("K", "W"), ("TH", "R"), ("SH", "R"), ("P", "Y"), ("B", "Y"), ("K", "Y"),
          ("M", "Y"), ("F", "Y"), ("V", "Y"), ("HH", "Y"), ("S", "T", "R"), ("S", "P", "R"), ("S", "P", "L"),
          ("S", "K", "R"), ("S", "K", "W")}


def joins(phones):
    """For each consonant, whether it is written with a halanta before the next consonant."""
    ph = [re.sub(r"\d", "", p) for p in phones]
    isv = [p in VOW for p in ph]
    hal = [False] * len(ph)
    i = 0
    while i < len(ph):
        if isv[i]:
            i += 1
            continue
        j = i
        while j < len(ph) and not isv[j]:
            j += 1
        run = ph[i:j]
        if i == 0 or j == len(ph):  # word-initial or word-final cluster: all joined
            for k in range(i, j - 1):
                hal[k] = True
        else:
            n_on = 1
            for m in (3, 2):
                if len(run) > m - 1 and tuple(run[-m:]) in ONSETS:
                    n_on = m
                    break
            cut = j - n_on  # first onset consonant
            for k in range(i, j - 1):
                # coda -> onset: nasals, liquids and s join (कम्प्लिट, अल्सो, मोस्टली), stops do not (फिडब्याक),
                # except k/p/f before s/sh/t/ch, which Nepal English writes joined: इक्स्पेक्ट, सेक्शन, एक्टिभ, सेफ्टी
                hal[k] = (k != cut - 1 or run[k - i] in ("M", "N", "NG", "L", "S", "R")
                          or (run[k - i] in ("K", "P", "F") and run[k - i + 1] in ("S", "SH", "T", "CH")))
        i = j
    return hal


def arpabet_to_devanagari(phones):
    hal = joins(phones)
    phones = [re.sub(r"\d", "", p) for p in phones]
    out, prev_cons, prev_i = [], False, -1
    for i, ph in enumerate(phones):
        if ph in VOW:
            # "o" before r is said ओ in Nepal English (more मोर, four फोर, before बिफोर, morning मोर्निङ); as अ it read
            # "mar" (मर, "die") and the frequency picker could not escape it
            ind, mat = ("ओ", "ो") if ph == "AO" and i + 1 < len(phones) and phones[i + 1] == "R" else VOW[ph]
            if prev_cons:
                out.append(mat)
            else:
                out.append(ind)
            prev_cons = ph == "ER"  # ER ends in र, which may take a halanta before a consonant
        else:
            if prev_cons and (prev_i < 0 or phones[prev_i] == "ER" or hal[prev_i]):
                out.append(HAL)
            out.append(CONS[ph])
            prev_cons = True
        prev_i = i
    return "".join(out)


# romanized Nepali (pathaunu, khana) -> Devanagari, longest match first
_ROM_C = [("chh", "छ"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"), ("dh", "ध"), ("ph", "फ"),
          ("bh", "भ"), ("sh", "श"), ("ng", "ङ"), ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"), ("t", "ट"),
          ("d", "ड"), ("n", "न"), ("p", "प"), ("f", "फ"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"),
          ("w", "व"), ("v", "भ"), ("s", "स"), ("h", "ह"), ("z", "ज"), ("x", "क्स"), ("q", "क")]
_ROM_V = [("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("ee", "ई", "ी"), ("oo", "ऊ", "ू"), ("a", "अ", ""),
          ("i", "इ", "ि"), ("u", "उ", "ु"), ("e", "ए", "े"), ("o", "ओ", "ो")]


def romanized_to_devanagari(w):
    w, out, i, prev_cons = w.lower(), [], 0, False
    while i < len(w):
        for r, d in _ROM_C:
            if w.startswith(r, i):
                if prev_cons:
                    out.append(HAL)
                out.append(d)
                i, prev_cons = i + len(r), True
                break
        else:
            for r, ind, mat in _ROM_V:
                if w.startswith(r, i):
                    out.append(mat if prev_cons else ind)
                    i, prev_cons = i + len(r), False
                    break
            else:
                i += 1
    return "".join(out)


_TABLES = None


def _keep_sh(out, phones):
    """English "sh" as श. Nepali writers (and our tables) spell it स (show -> सो, shopping -> सपिङ), and the model
    then says "so" (user, 2026-10-23: full English sounds broken). The S/SH phones and the स/श letters are matched in
    order; only when the counts agree, so a spelling that does not follow the phones is left alone."""
    sib = [p for p in phones if re.sub(r"\d", "", p) in ("S", "SH", "Z", "ZH")]
    if not any(re.sub(r"\d", "", p) in ("SH", "ZH") for p in sib):
        return out
    pos = [i for i, c in enumerate(out) if c in "सशष"]
    if len(pos) != len([p for p in sib if re.sub(r"\d", "", p) in ("S", "SH", "ZH")]):
        return out
    chars, k = list(out), 0
    for p in sib:
        p = re.sub(r"\d", "", p)
        if p == "Z":
            continue
        chars[pos[k]] = "श" if p in ("SH", "ZH") else "स"
        k += 1
    return "".join(chars)


def _plural_s(out, phones):
    """Final "z" after a consonant (years, things, others, needs) is said "s" in Nepal English: यिर्ज -> इयर्स style.
    After a vowel it stays ज (days डेज, news न्युज)."""
    p = [re.sub(r"\d", "", x) for x in phones]
    if len(p) >= 2 and p[-1] == "Z" and (p[-2] not in VOW or p[-2] == "ER") and out.endswith("ज"):
        return out[:-1] + "स"
    return out


def _endings(out, w, phones):
    """-ed after t/d and -es after a hissing sound keep their vowel the way Nepal English says it: needed नीडेड (was
    नीडड), decided डिसाइडेड, changes चेन्जेज (was चेन्जज), places प्लेसेज. The reduced vowel (AH0/IH0) has no
    letter of its own in the default mapping, so the ending came out as a bare cluster."""
    p = [re.sub(r"\d", "", x) for x in phones]
    low = w.lower()
    if low.endswith("ed") and len(p) >= 3 and p[-1] == "D" and p[-2] in ("AH", "IH") and p[-3] in ("T", "D"):
        for a, b in (("डिड", "डेड"), ("टिड", "टेड"), ("डड", "डेड"), ("टड", "टेड")):
            if out.endswith(a):
                return out[:-len(a)] + b
    if low.endswith("es") and len(p) >= 3 and p[-1] == "Z" and p[-2] in ("AH", "IH") and p[-3] in ("S", "Z", "CH", "JH", "SH", "ZH"):
        if out.endswith("ज") and not out.endswith("ेज") and not out.endswith("िज"):
            return out[:-1] + "ेज"
        if out.endswith("िज"):
            return out[:-2] + "ेज"
    return out


_JOIN = {("K", "S"): ("क", "स"), ("K", "SH"): ("क", "श"), ("K", "T"): ("क", "ट"), ("K", "CH"): ("क", "च"),
         ("P", "T"): ("प", "ट"), ("F", "T"): ("फ", "ट")}


def _join_clusters(out, phones):
    """Table spellings that split a cluster the phones say has no vowel (सेकशन, पिकचर्स, टेकसास) get the halanta:
    सेक्शन, पिक्चर्स, टेक्सास. Only where the two sounds are adjacent in CMUdict, so फोकस (fo-kus) is untouched."""
    p = [re.sub(r"\d", "", x) for x in phones]
    for a, b in zip(p, p[1:]):
        if (a, b) in _JOIN:
            x, y = _JOIN[(a, b)]
            out = re.sub(f"{x}{y}(?![ािीुूेैोौृ])", f"{x}्{y}", out, count=1) if f"{x}{y}" in out else out
    return out


def word_to_devanagari(w):
    r = _word_to_devanagari(w)
    ph = _cmu().get(w.lower())
    return _join_clusters(_endings(_plural_s(_keep_sh(r, ph), ph), w, ph), ph) if ph else r


def _word_to_devanagari(w):
    global _TABLES
    if _TABLES is None:
        _TABLES = _load_tables()
    curated, google = _TABLES
    low = w.lower()
    if low in curated:  # data/loanwords.tsv: everyday English as said in Nepal
        return curated[low]
    if low in LOAN:
        return LOAN[low]
    if low in google and not (w.isupper() and len(w) <= 6):
        return google[low]
    if w.isupper() and 2 <= len(w) <= 6:
        return "".join(LETTERS[c] for c in low if c in LETTERS)
    if "'" in low and _cmu().get(low):  # contractions are in CMUdict: it's, don't, I'd (not a possessive "it" + को)
        return best_spelling(_cmu()[low])
    base, poss = (low[:-2], True) if low.endswith("'s") else (low, False)
    ph = _cmu().get(base)
    if ph:
        return best_spelling(ph) + ("को" if poss else "")
    for i in range(3, len(base) - 2):  # compound of two dictionary words: cashback = cash + back
        a, b = _cmu().get(base[:i]), _cmu().get(base[i:])
        if a and b:
            return best_spelling(a) + best_spelling(b) + ("को" if poss else "")
    return romanized_to_devanagari(low)


def english_to_devanagari(text):
    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", lambda m: word_to_devanagari(m.group(0)), text)


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:]:
        print(english_to_devanagari(s))
