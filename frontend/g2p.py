"""Devanagari -> phoneme grapheme-to-phoneme (G2P) for Nepali.

Lexicon-first (Google Nepali lexicon, CC-BY 4.0: ~48.6k Devanagari + ~5.6k
Latin-script words), with rules for out-of-vocabulary words. Produces a list
of phoneme tokens (a compact ASCII scheme, not full IPA) suitable for feeding
a TTS frontend without relying on espeak-ng, which mis-maps Nepali affricates
and schwa. Nepali keeps medial schwa (medial deletion is written with a
halanta) but drops most word-final schwa; see _delete_final_schwa.
Sources: Khatiwada 2009 (JIPA), nepalilanguage.org, Wikipedia Nepali phonology.

Rule accuracy (lexicon bypassed) is measured by pipeline/eval_g2p.py.
"""
import os
import re
import unicodedata

from .normalize import normalize

# Consonant -> base phoneme (inherent schwa added separately)
CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "T", "ठ": "Th", "ड": "D", "ढ": "Dh", "ण": "N",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "w",
    # Nepali has one sibilant /s/ (Khatiwada 2009); श ष स all -> s
    "श": "s", "ष": "s", "स": "s", "ह": "h",
    "ळ": "l",  # rare (Marathi/Vedic); read as l
}

# NFC decomposes nukta letters (ज़ -> ज + ़), so they are matched as
# base consonant followed by NUKTA rather than as precomposed keys.
NUKTA = "़"
NUKTA_MAP = {"ड": "r", "ढ": "rh", "फ": "f", "ज": "z"}  # क़ ख़ ग़ fall back to base
# Conjunct rewrite: ज्ञ -> ग्य (gya). Other conjuncts go through the halanta
# cluster path.
_CONJUNCT_REWRITES = (("ज्ञ", "ग्य"),)  # क्ष is a learned decision (see _KSA_OPTIONS)

# Nepali has no vowel-length contrast (Khatiwada 2009), so ई/ई and उ/ऊ
# spellings map to the same phoneme, matching Google's Nepali lexicon.
IND_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऋ": "ri", "ॠ": "ri",
}

# Dependent vowel signs (matras) -> vowel phoneme
MATRAS = {
    "ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
}

SCHWA = "a"
HALANTA = "्"      # virama: suppresses inherent vowel
ANUSVARA = "ं"     # ं  nasalization
CHANDRABINDU = "ँ" # ँ  nasalization
VISARGA = "ः"      # ः  -> h
NASAL = "n~"            # phoneme tag for nasalization

# Google lexicon phones -> our tokens. ऐ/औ stay single tokens (ai/au) to match
# the rule path; vowel+i/u sequences (आइ) stay split.
_GOOGLE_MAP = {"ax": ["a"], "tt": ["T"], "dd": ["D"], "tth": ["Th"], "ddh": ["Dh"], "nn": ["N"],
               "ts": ["c"], "tsh": ["ch"], "jh": ["j"], "jhh": ["jh"],
               "axj": ["ai"], "axw": ["au"], "aaj": ["aa", "i"], "aaw": ["aa", "u"],
               "oj": ["o", "i"], "ew": ["e", "u"], "ow": ["o", "u"]}
_GOOGLE_NASAL_BASES = {"aa", "ax", "i", "u", "e", "o", "aaw", "axw", "aaj", "axj", "ow", "oj"}
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEXICON_PATH = os.path.join(_ROOT, "data", "lexicons", "google", "lexicon.tsv")
_MODEL_PATH = os.path.join(_ROOT, "data", "g2p_model.json.gz")


def google_to_tokens(phones: str):
    out = []
    for t in phones.split():
        if t == ".":
            continue
        if t.endswith("n") and t[:-1] in _GOOGLE_NASAL_BASES:
            out += _GOOGLE_MAP.get(t[:-1], [t[:-1]]) + [NASAL]
        else:
            out += _GOOGLE_MAP.get(t, [t])
    return out


def _load_lexicon(path=_LEXICON_PATH):
    lex = {}
    if not os.path.exists(path):
        return lex
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        word, phones = line.rstrip("\n").split("\t")[:2]
        tokens = google_to_tokens(phones)
        if not tokens:
            continue  # 113 entries have blank pronunciations; they would make words silent
        key = unicodedata.normalize("NFC", word)
        lex.setdefault(key.lower() if key.isascii() else key, tokens)
    return lex


_NEURAL_OOV_PATH = os.path.join(_ROOT, "data", "lexicons", "neural_oov.tsv")


def _load_token_tsv(path):
    """word<TAB>space-separated tokens (written by pipeline/train_neural_g2p.py
    only when the neural G2P beats the vote model on held-out words)."""
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        if "\t" in line:
            word, toks = line.rstrip("\n").split("\t", 1)
            if toks.split():
                out[unicodedata.normalize("NFC", word)] = toks.split()
    return out


# Priority: hand overrides > Google lexicon > neural OOV pronunciations.
MANUAL_OVERRIDES = {}
LEXICON = {**_load_token_tsv(_NEURAL_OOV_PATH), **_load_lexicon(), **MANUAL_OVERRIDES}


def _phonemize_word(word: str, use_lexicon: bool = True):
    norm_word = unicodedata.normalize("NFC", word)
    if use_lexicon and norm_word in LEXICON:
        return list(LEXICON[norm_word])
    split = _split_suffix(norm_word, LEXICON if use_lexicon else None)
    if split:
        if use_lexicon:
            # Inflected forms: the learned rules (trained on thousands of inflected
            # lexicon entries) beat stem-lookup + suffix on held-out words, 92.0% vs 86.4%.
            # (A stem-vs-rules selector here was tried: no gain on dev.)
            return _phonemize_word(norm_word, use_lexicon=False)
        return _phonemize_word(split[0], False) + _phonemize_word(split[1], False)
    if use_lexicon:
        parts = _compound_split(norm_word)
        if parts:
            return _choose_compound(parts, norm_word)
    items, chars = _rule_template(norm_word)
    return _realize(items, _decide(items, chars))


VOWELS = {"a", "aa", "i", "u", "e", "ai", "o", "au", "ri"}

# Postpositions, adverbs and some nouns keep the final schwa (बाट -> baaTa).
# Verb stems without a written halanta also keep it (गर -> gara) but need POS
# or a lexicon to tell apart from nouns, so they are left to LEXICON.
KEEP_FINAL_SCHWA = {"बाट", "सुख", "दुख", "तर", "अब", "तब", "जब"}

# Case markers / postpositions. The stem keeps its own final-schwa rule
# (नेपालको -> nepaal-ko, not nepaala-ko).
SUFFIXES = sorted(["हरूलाई", "हरूको", "हरूले", "हरूमा", "हरू", "हरु", "लाई", "बाट", "देखि",
                   "सम्म", "सँग", "सित", "भन्दा", "भित्र", "तिर", "माथि", "पछि",
                   "को", "का", "की", "मा", "ले", "नै"], key=len, reverse=True)


def _split_suffix(word, lexicon=None):
    """Split off a suffix when the stem ends in a bare consonant (its final
    schwa is at stake) or when the stem is a known lexicon word (so inflected
    forms like खानामा reuse the gold pronunciation of खाना)."""
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            stem = word[: -len(suf)]
            if stem[-1] in CONSONANTS or (lexicon is not None and stem in lexicon):
                return stem, suf
    return None


# Clusters the Google lexicon geminates in >=60% of words (measured by
# pipeline/eval_g2p.py analysis); elsewhere doubling is inconsistent.
_GEMINATE = {("त", "व"), ("म", "र"), ("द", "य"), ("श", "व"), ("त", "र"),
             ("त", "य"), ("श", "य"), ("न", "य")}


_SIGNS = set(MATRAS) | {HALANTA, ANUSVARA, CHANDRABINDU, VISARGA, NUKTA}


def _compound_split(word, min_len=2):
    """Split an out-of-lexicon word into two lexicon words at an akshara
    boundary (शोभबहादुर -> शोभ + बहादुर), preferring the most balanced split.
    On held-out words this lifts accuracy from 54% to 84% where it applies;
    compounds and names are the largest rule-error class."""
    best = None
    for k in range(min_len, len(word) - min_len + 1):
        if word[k] in _SIGNS or word[k - 1] == HALANTA:
            continue
        a, b = word[:k], word[k:]
        if a in LEXICON and b in LEXICON:
            score = min(len(a), len(b))
            if best is None or score > best[0]:
                best = (score, a, b)
    return best[1:] if best else None


def _base_phoneme(ch, chars, i, j, out):
    """Context-dependent consonant realizations, derived from Google's lexicon:
    व is /w/ as a conjunct's second member (स्वर) or before ा after a vowel or
    word-initially (वास्तव, सवारी); /b/ elsewhere (विद्यालय, धन्यवाद).
    The ड/ढ flap is allophonic and left to the acoustic model."""
    if ch == "व":
        if i > 0 and chars[i - 1] == HALANTA:
            return "w"
        nxt = chars[j] if j < len(chars) else ""
        if nxt == "ा" and not (i > 0 and chars[i - 1] in CONSONANTS):
            return "w"
        return "b"
    return CONSONANTS[ch]


def _delete_final_schwa(out, chars):
    """Drop the word-final inherent schwa (घर -> ghar, नेपाल -> nepaal) except
    when it is the only vowel (छ -> cha), after a written conjunct
    (गर्छ -> garcha), or after छ / य / ह (समय -> samaya). Loanword final
    conjuncts that drop it (पार्क -> paark) belong in LEXICON."""
    if not out or out[-1] != SCHWA:
        return
    if sum(1 for t in out if t in VOWELS) <= 1:
        return
    cons_pos = [i for i, c in enumerate(chars) if c in CONSONANTS]
    if cons_pos:
        last = cons_pos[-1]
        if chars[last] in ("छ", "य", "ह"):
            return
        if last > 0 and chars[last - 1] in (HALANTA, ANUSVARA, CHANDRABINDU):
            return  # conjunct or nasal+consonant cluster keeps it (अंक -> anka)
    out.pop()


# English letter names as Nepali speakers say them (retroflex T/D), for acronyms
_LETTER_NAMES = {
    "A": "e", "B": "b i", "C": "s i", "D": "D i", "E": "i", "F": "e ph",
    "G": "j i", "H": "e c", "I": "aa i", "J": "j e", "K": "k e", "L": "e l",
    "M": "e m", "N": "e n", "O": "o", "P": "p i", "Q": "k y u", "R": "aa r",
    "S": "e s", "T": "T i", "U": "y u", "V": "bh i", "W": "D a b l y u",
    "X": "e k s", "Y": "w aa i", "Z": "j e D",
}
_EN_DIGRAPHS = [("sh", "sh"), ("ch", "c"), ("th", "th"), ("ph", "ph"), ("ng", "ng"),
                ("ee", "i"), ("oo", "u"), ("ea", "i"), ("ai", "e"), ("ay", "e"),
                ("ou", "au"), ("ow", "au")]
_EN_LETTERS = {"a": "a", "b": "b", "c": "k", "d": "D", "e": "e", "f": "ph", "g": "g",
               "h": "h", "i": "i", "j": "j", "k": "k", "l": "l", "m": "m", "n": "n",
               "o": "o", "p": "p", "q": "k", "r": "r", "s": "s", "t": "T", "u": "a",
               "v": "bh", "w": "w", "x": "k s", "y": "y", "z": "j"}


def _spell(word: str):
    return " ".join(_LETTER_NAMES[c] for c in word.upper() if c in _LETTER_NAMES).split()


def _english_fallback(word: str):
    """Rough Nepali-accented reading of an English word not in the lexicon,
    so it is never silently dropped. Production could swap in a dictionary
    G2P (e.g. g2p_en/CMUdict mapped to this phone set)."""
    w, out, i = word.lower(), [], 0
    while i < len(w):
        for dg, ph in _EN_DIGRAPHS:
            if w.startswith(dg, i):
                out.extend(ph.split()); i += 2
                break
        else:
            if w[i] in _EN_LETTERS:
                out.extend(_EN_LETTERS[w[i]].split())
            i += 1
    return out


_PAUSE_LONG = set("।॥.?!|…")
_PAUSE_SHORT = set(",;:")
_STRIP = "\"'“”‘’()[]{}«»<>*#"
_INNER_SPLIT = re.compile(r"[-–—/\\]+")
_LATIN = "A-Za-z\u00C0-\u024F"
_SCRIPT_RUN = re.compile(rf"[{_LATIN}'’]+|[^{_LATIN}'’]+")


def _latin_word(word: str):
    """Short or vowel-less all-caps words are spelled (AI, OK, BBC); other
    words use the lexicon (Google has ~5.6k Latin entries) or the fallback."""
    letters = "".join(c for c in unicodedata.normalize("NFKD", word)
                      if not unicodedata.combining(c) and c.isascii() and c.isalpha())
    if not letters:
        return []
    if letters.isupper() and (len(letters) <= 3 or not set(letters) & set("AEIOU")):
        return _spell(letters)
    low = letters.lower()
    if low in LEXICON:
        return list(LEXICON[low])
    return _english_fallback(letters)


def g2p(text: str):
    """Phonemize raw or normalized text (normalize is idempotent and applied
    here, so digits/symbols are never silently dropped). '|' marks word
    boundaries; ',' and '.' are short/long pause tokens from punctuation."""
    tokens = []
    for raw in normalize(text).split():
        word, pause = raw, None
        while word:
            if word[-1] in _PAUSE_LONG or word[-1] in _PAUSE_SHORT:
                pause = "." if word[-1] in _PAUSE_LONG or pause == "." else ","
            elif word[-1] not in _STRIP:
                break
            word = word[:-1]
        word = word.lstrip(_STRIP)
        for piece in _INNER_SPLIT.split(word):
            for part in _SCRIPT_RUN.findall(piece):
                ph = _latin_word(part) if re.match(f"[{_LATIN}]", part) else _phonemize_word(part)
                if not ph:
                    continue
                if tokens and tokens[-1] not in (",", "."):
                    tokens.append("|")
                tokens.extend(ph)
        if pause and tokens:
            if tokens[-1] not in (",", "."):
                tokens.append(pause)
            elif pause == ".":
                tokens[-1] = "."
    return tokens


_EN_PHONES = {p for v in list(_LETTER_NAMES.values()) + [ph for _, ph in _EN_DIGRAPHS]
              + list(_EN_LETTERS.values()) for p in v.split()}

PHONEME_INVENTORY = sorted(
    set(CONSONANTS.values()) | set(NUKTA_MAP.values()) | set(IND_VOWELS.values())
    | set(MATRAS.values()) | {SCHWA, NASAL, "w", "b", "r", "rh", "h"} | _EN_PHONES
    | {t for toks in LEXICON.values() for t in toks}
)


# F5-TTS tokenizes text per character, so multi-letter tokens ("kh", "aa")
# must become one symbol each or "kh" collides with "k"+"h". Word boundary ->
# space; pauses keep their punctuation. Mapping is deterministic (sorted
# inventory -> Unicode private-use block) so train and inference agree.
_F5_SPECIAL = {"|": " ", ",": ", ", ".": ". "}
F5_SYMBOLS = {tok: chr(0xE000 + i) for i, tok in enumerate(PHONEME_INVENTORY)}


def encode_for_f5(tokens):
    return "".join(_F5_SPECIAL.get(t) or F5_SYMBOLS[t] for t in tokens).strip()


# Pronunciation respelling: phonemes written back as Devanagari, with a halanta
# on every consonant that has no vowel (कम -> कम्), one letter per sibilant
# (शर्मा -> सर्मा) and doubled sounds spelled out. A character model (IndicF5)
# already knows every symbol, so this feeds it our G2P without new embeddings.
_RESPELL_CONS = {v: k for k, v in CONSONANTS.items() if k not in "शषळ"}
_RESPELL_CONS.update({"sh": "श", "f": "फ़", "z": "ज़", "rh": "ढ़"})
_RESPELL_IND = {"a": "अ", "aa": "आ", "i": "इ", "u": "उ", "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ", "ri": "ऋ"}
_RESPELL_MATRA = {"a": "", "aa": "ा", "i": "ि", "u": "ु", "e": "े", "ai": "ै", "o": "ो", "au": "ौ", "ri": "ृ"}
_RESPELL_SPECIAL = {"|": " ", ",": ", ", ".": "। "}


def respell(tokens):
    # Google's lexicon doubles the first consonant of every medial cluster
    # (राम्रो -> raam.mro, पत्र -> pat.tra); that is predictable from the
    # spelling, so it is not written out (राम्म्रो would read as a long m).
    tokens = [t for i, t in enumerate(tokens)
              if not (i + 2 < len(tokens) and t in _RESPELL_CONS and tokens[i + 2] in _RESPELL_CONS
                      and tokens[i + 1] in (t, *[a for a, u in _UNASPIRATED.items() if u == t]))]
    out, bare = [], False  # bare: last symbol is a consonant still owed a vowel or halanta
    for t in tokens:
        if t in _RESPELL_SPECIAL or t not in PHONEME_INVENTORY:
            out.append(("्" if bare else "") + _RESPELL_SPECIAL.get(t, " "))
            bare = False
        elif t in _RESPELL_CONS:
            out.append(("्" if bare else "") + _RESPELL_CONS[t])
            bare = True
        elif t in _RESPELL_IND:
            out.append(_RESPELL_MATRA[t] if bare else _RESPELL_IND[t])
            bare = False
        elif t == NASAL:
            out.append(("्" if bare else "") + CHANDRABINDU)
            bare = False
    if bare:
        out.append("्")
    return unicodedata.normalize("NFC", " ".join("".join(out).split()))



# ---------------------------------------------------------------------------
# Learned decision layer. The rule path is expressed as a template of fixed
# tokens plus decision points; each decision's default is the hand rule, and a
# context model trained on the Google lexicon (pipeline/train_g2p.py) can
# override it. With no model file the output equals the hand rules exactly.
# Decision item: [kind, char_index, options, default]
# ---------------------------------------------------------------------------
_NASAL_OPTIONS = ["n~", "m", "n", "ng"]
# क्ष in Google's lexicon: k ch y (316), k ch (67), ch (64), ch y (63)
_KSA_OPTIONS = ["k ch y", "k ch", "ch y", "ch", "k s"]
_UNASPIRATED = {"kh": "k", "gh": "g", "ch": "c", "jh": "j", "Th": "T", "Dh": "D",
                "th": "t", "dh": "d", "ph": "p", "bh": "b"}
CONTEXT_ORDERS = [(3, 3), (2, 3), (3, 2), (2, 2), (1, 3), (3, 1), (1, 2), (2, 1),
                  (1, 1), (0, 2), (2, 0), (0, 1), (1, 0), (0, 0)]


def _load_model(path=_MODEL_PATH):
    """Returns (decisions, params, selector). decisions maps context key ->
    label (first-hit mode) or -> {label: count} (vote mode); selector holds
    compound-vs-rules statistics (see _choose_compound)."""
    if not os.path.exists(path):
        return {}, {}, {}
    import gzip
    import json
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return (data.get("decisions", {}), data.get("meta", {}).get("params", {}),
            data.get("selector", {}))


MODEL, MODEL_PARAMS, SELECTOR = _load_model()


def _rule_parts(word):
    """Templates of the rule path (_phonemize_word with use_lexicon=False)."""
    sp = _split_suffix(word, None)
    if sp:
        return _rule_parts(sp[0]) + _rule_parts(sp[1])
    return [_rule_template(word)]


def rules_confidence(word, target, max_combos=4096):
    """How sure the learned rules are where they disagree with `target`:
    'U' if no decision combination yields it, else a bucket of the weakest
    vote share among the decisions that would have to flip."""
    parts = _rule_parts(word)
    decs = []
    for items, chars in parts:
        for it in items:
            if isinstance(it, str):
                continue
            sc = _vote_scores(context_keys(it[0], chars, it[1]), MODEL,
                              MODEL_PARAMS.get("alpha", 0.5), MODEL_PARAMS.get("smooth", 5)) if MODEL else {}
            sc = {o: sc.get(o, 0.0) for o in it[2]}
            tot = sum(sc.values())
            share = {o: (v / tot if tot else (1.0 if o == it[3] else 0.0)) for o, v in sc.items()}
            pick = max(share, key=share.get) if tot else it[3]
            decs.append((it[2], pick, share))
    n = 1
    for opts, _, _ in decs:
        n *= len(opts)
    if n > max_combos:
        return "U"
    import itertools
    best = None
    for combo in itertools.product(*[d[0] for d in decs]):
        out, k = [], 0
        for items, _ in parts:
            m = sum(1 for it in items if not isinstance(it, str))
            out += _realize(items, list(combo[k:k + m]))
            k += m
        if out == target:
            flips = [d[2][d[1]] for c, d in zip(combo, decs) if c != d[1]]
            if best is None or len(flips) < len(best):
                best = flips
    if best is None:
        return "U"
    weakest = min(best) if best else 0.0
    return "c" + str(sum(weakest >= t for t in (0.6, 0.8, 0.95)))


def selector_keys(parts, comp, rules, conf="*", tag=""):
    a, b = parts
    d = max(-2, min(2, comp.count(SCHWA) - rules.count(SCHWA)))
    short = min(4, len(a), len(b))  # spurious splits have a tiny piece: छु+टेन, नि+काल
    keys = [f"{conf}|{d}|{short}|{a[-1]}|{b[0]}", f"{conf}|{d}|{short}|*|*", f"{conf}|{d}|*|*|*",
            f"*|{d}|{short}|{a[-1]}|{b[0]}", f"*|{d}|{short}|{a[-1]}|*", f"*|{d}|{short}|*|{b[0]}",
            f"*|{d}|{short}|*|*", f"*|{d}|*|*|*"]
    return [tag + k for k in keys]


def compound_candidate(parts):
    a, b = parts
    return list(LEXICON[a]) + (list(LEXICON[b]) if b in LEXICON else _phonemize_word(b, use_lexicon=False))


def _choose_compound(parts, word, tag=""):
    """Compound split vs learned rules. On held-out compounds the split alone is
    right 85%, rules 83%, at least one 94%; a cross-fitted selector keyed on the
    schwa difference and the junction letters picks between them."""
    comp = compound_candidate(parts)
    fallback = comp if not tag else None  # suffix words default to the rules
    stats, params = SELECTOR.get("stats"), SELECTOR.get("params", {})
    if not stats:
        return fallback or _phonemize_word(word, use_lexicon=False)
    rules = _phonemize_word(word, use_lexicon=False)
    if rules == comp:
        return comp
    for key in selector_keys(parts, comp, rules, rules_confidence(word, comp), tag):
        c = stats.get(key)
        if c and sum(c.values()) >= params["min_n"]:
            if tag:  # suffix words: the stem lookup must clearly win
                return comp if c.get("comp", 0) > c.get("rules", 0) * params["margin"] else rules
            if c.get("rules", 0) > c.get("comp", 0) * params["margin"]:
                return rules
            if c.get("comp", 0) >= c.get("rules", 0):
                return comp
    return fallback or rules


def _after(chars, i, seq):
    return "".join(chars[max(0, i - len(seq)):i]) == seq


# Consonants the lexicon sometimes realizes differently: (applies?, kind, alternate).
# English loans use dental t/d (कार्ड -> kaard); द्ध is often just dh; final ङ्ग is ng.
_ALTERNATES = {
    "ट": (lambda c, i: True, "dent", "t"),
    "ड": (lambda c, i: True, "dent", "d"),
    "द": (lambda c, i: c[i + 1:i + 3] == [HALANTA, "ध"], "dd", ""),
    "ग": (lambda c, i: _after(c, i, "ङ" + HALANTA), "ngg", ""),
}


def _nasal_item(chars, j):
    """ं ँ ः after a vowel. Visarga is silent in 70/74 lexicon words, sometimes
    doubles the next consonant (दुःख); final ं is sometimes 'm' (एवं -> ebam)."""
    nxt = chars[j + 1] if j + 1 < len(chars) else ""
    if chars[j] == VISARGA:
        options = ["", "h"]
        if nxt in CONSONANTS:
            options.append(_UNASPIRATED.get(CONSONANTS[nxt], CONSONANTS[nxt]))
        return ["visarga", j, options, ""]
    if nxt in CONSONANTS:
        return ["nasal", j, _NASAL_OPTIONS, NASAL]
    if not nxt:
        return ["nasal", j, ["n~", "m"], NASAL]
    return NASAL


def _rule_template(norm_word):
    w = norm_word
    jna = {m.start() + 2 for m in re.finditer("ज्ञ", w)}  # index of य after the rewrite
    for src, dst in _CONJUNCT_REWRITES:
        w = w.replace(src, dst)
    chars = list(w)
    n, items, i = len(chars), [], 0
    while i < n:
        ch = chars[i]
        if ch in CONSONANTS:
            j = i + 1
            ph = None
            k = len(items)  # a doubled consonant goes before it: d dh, not dh d
            if j < n and chars[j] == NUKTA:
                ph = NUKTA_MAP.get(ch, CONSONANTS[ch])
                items.append(ph)
                j += 1
            elif ch == "व":
                items.append(["va", i, ["b", "w"], _base_phoneme(ch, chars, i, j, None)])
                if j + 1 < n and chars[j] == HALANTA and chars[j + 1] in CONSONANTS:
                    items.insert(k, ["gem", i, ["", "b"], ""])  # कर्तव्य -> kartabby
            elif ch == "क" and chars[j:j + 2] == [HALANTA, "ष"]:
                items.append(["ksa", i, _KSA_OPTIONS, _KSA_OPTIONS[0]])
                j += 2
            elif ch == "ण":
                items.append(["Nn", i, ["N", "n"], "N"])
            elif ch == "ह":
                # silent or assimilated in the lexicon: सिंह -> sing, तेह्र -> tera, अपराह्न -> aparaanna
                opts = ["h", ""]
                if j + 1 < n and chars[j] == HALANTA and chars[j + 1] in CONSONANTS:
                    opts.append(CONSONANTS[chars[j + 1]])
                items.append(["h", i, opts, "h"])
            else:
                ph = CONSONANTS[ch]
                alt = _ALTERNATES.get(ch)
                if alt and alt[0](chars, i):
                    items.append([alt[1], i, [ph, alt[2]], ph])  # learned; default is the plain sound
                else:
                    items.append(ph)
            if j < n and chars[j] == HALANTA:
                if ph and j + 1 < n and chars[j + 1] in CONSONANTS:
                    dup = _UNASPIRATED.get(ph, ph)  # doubled aspirate is written d dh
                    items.insert(k, ["gem", i, ["", dup], dup if (ch, chars[j + 1]) in _GEMINATE else ""])
                i = j + 1
                continue
            if ph and j < n and chars[j] == "ृ":
                items.insert(k, ["gem", i, ["", _UNASPIRATED.get(ph, ph)], ""])  # अमृत -> ammrit
            if j < n and chars[j] in MATRAS:
                items.append(MATRAS[chars[j]])
                j += 1
            else:
                items.append(["schwa", i, ["a", ""], SCHWA])
            if i in jna:
                items.append(["jna", i, ["", NASAL], ""])  # ज्ञान -> gyaan~n in the lexicon
            while j < n and chars[j] in (ANUSVARA, CHANDRABINDU, VISARGA):
                items.append(_nasal_item(chars, j))
                j += 1
            i = j
            continue
        if ch in IND_VOWELS:
            items.append(IND_VOWELS[ch])
            j = i + 1
            while j < n and chars[j] in (ANUSVARA, CHANDRABINDU, VISARGA):
                items.append(_nasal_item(chars, j))
                j += 1
            i = j
            continue
        if ch in (ANUSVARA, CHANDRABINDU):
            items.append(_nasal_item(chars, i))
        elif ch == VISARGA:
            items.append("h")
        i += 1
    last = items[-1] if items else None
    if isinstance(last, list) and last[0] == "schwa" and w not in KEEP_FINAL_SCHWA:
        probe = _realize(items, None)
        trial = list(probe)
        _delete_final_schwa(trial, chars)
        if len(trial) < len(probe):
            last[3] = ""
    return items, chars


def _realize(items, choices):
    out, k = [], 0
    for it in items:
        if isinstance(it, str):
            out.append(it)
            continue
        tok = it[3] if choices is None else choices[k]
        k += 1
        out.extend(tok.split())
    return out


def context_keys(kind, chars, pos):
    """Context keys from widest to narrowest. Every key carries whether the
    decision is word-final (F) or medial (M), so narrow backoff contexts still
    know the single most important fact for schwa decisions."""
    s = "^" + "".join(chars) + "$"
    p = pos + 1
    where = "F" if all(c == NUKTA for c in chars[pos + 1:]) else "M"
    for left, right in CONTEXT_ORDERS:
        yield f"{kind}{where}|{left}|{right}|{s[max(0, p - left):p]}|{s[p]}|{s[p + 1:p + 1 + right]}"


def _vote(keys, model, alpha, smooth):
    """Combine all matching contexts: each votes its label distribution,
    weighted by alpha**rank (wider contexts first) and by count reliability."""
    scores = _vote_scores(keys, model, alpha, smooth)
    return max(scores, key=scores.get) if scores else None


def _vote_scores(keys, model, alpha, smooth):
    scores = {}
    for rank, key in enumerate(keys):
        dist = model.get(key)
        if not dist:
            continue
        total = sum(dist.values())
        w = (alpha ** rank) * total / (total + smooth) / total
        for label, n in dist.items():
            scores[label] = scores.get(label, 0.0) + n * w
    return scores


def _decide(items, chars, model=None, params=None):
    model = MODEL if model is None else model
    params = MODEL_PARAMS if params is None else params
    vote = params.get("mode") == "vote"
    choices = []
    for it in items:
        if isinstance(it, str):
            continue
        kind, pos, options, default = it
        choice = default
        if model:
            keys = context_keys(kind, chars, pos)
            if vote:
                hit = _vote(keys, model, params["alpha"], params["smooth"])
            else:
                hit = next((model[k] for k in keys if k in model), None)
            if hit is not None and hit in options:
                choice = hit
        choices.append(choice)
    return choices
