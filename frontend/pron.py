"""Pronunciation fixes for the TTS input: words the model says wrong, respelled the way it
says them right (found automatically by tools/mine_hard_words.py: data/pron_fixes.json).
Applied only to what the model reads, never to reference transcripts."""
import json
import os
import re

_FIXES = None


def fixes():
    global _FIXES
    if _FIXES is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pron_fixes.json")
        _FIXES = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _FIXES


def apply_fixes(text):
    f = fixes()
    if not f:
        return text
    return re.sub(r"[ऀ-ॿ]+", lambda m: _fix_word(m.group(0), f), text)


# case endings a fixed stem can carry (काठमाडौँमा, काठमाडौँको): the fix applies to the stem, the ending stays
_SUFFIXES = ("मा", "को", "का", "की", "बाट", "ले", "लाई", "सम्म", "तिर", "भित्र", "वासी", "माथि")


def _fix_word(w, f):
    if w in f:
        return f[w]
    for suf in _SUFFIXES:
        if w.endswith(suf) and w[:-len(suf)] in f:
            return f[w[:-len(suf)]] + suf
    return w


# Nepal pronunciation, written the way it is said, for the model input only. Much of the training audio
# is Indian Nepali (IndicVoices: Darjeeling, Sikkim, Assam), where श/ष are "sh" and व is "v"; in Nepal
# all three sibilants are "s", व starts a word as "b", क्ष is "chhy", ज्ञ is "gy", ण is "n", ृ is "ri".
# दशचोटि -> दसचोटि (das choti), विशेष -> बिसेस, शिक्षा -> सिछ्या, सर्वोच्च -> सर्बोच्च.
_NEPAL = [
    (re.compile(r"क्ष"), "छ्य"),
    (re.compile(r"ज्ञ"), "ग्य"),
    (re.compile(r"श(?!्र)"), "स"),          # keep श्र (श्री is said "shree")
    (re.compile(r"ष"), "स"),
    (re.compile(r"ण"), "न"),
    (re.compile(r"(?<![\u0900-\u097F])व"), "ब"),   # word-initial व
    (re.compile(r"र्व"), "र्ब"),
    (re.compile(r"ृ"), "्रि"),
]


def nepal_accent(text):
    for pat, rep in _NEPAL:
        text = pat.sub(rep, text)
    return text


# Spoken final "a" (schwa) after a plain final consonant: Nepali keeps it in खान (khaa-na), आज (aa-ja), गरेर, छैन,
# where Hindi-style reading drops it ("khaan", "aaj"); the model, trained partly on Indian Nepali, drops it too.
# Written out as halanta + अ (खान्अ, तर्अ, आज्अको). A plain अ (खानअ) held the final sound longer in 6 of 7 words
# (ops/final_vowel_cases6.json: खान 155 -> 172 ms, किन 142 -> 175 ms) but the user heard it as distorted (2026-10-23);
# the take picker now checks the held vowel instead (ops/v4_lab.py held_ms). ऽ (avagraha) lengthened the
# word but the model often hummed a nasal instead of the vowel (खानङ, तरं, literal MMS); ्अ gave the vowel without
# that (खान्अ heard खाना 3/4, तर्अ clean 4/4, final sound 60 -> 100 ms; tools/final_vowel.py rounds 1-4).
# Only where Nepali reliably keeps it: a G2P-wide rule (any word whose G2P ends in "a") also held nouns that drop it
# (बार, फाल) and the model added a stray nasal there; NepTTS-Bench pronunciation CER 4.11 -> 4.31.
_HOLD_WORDS = {"आज", "अब", "तर", "किन", "जब", "तब", "खान", "जान", "आउन", "पिउन", "बिहान", "बेलुकान"}
_HOLD_ENDINGS = ("ेर", "एर", "ैन")  # गरेर, मिलेर, भनेर, गएर; छैन, हुँदैन, लाग्दैन
_CONS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"


def hold_final_schwa(text):
    def held(w):
        if len(w) < 2 or w[-1] not in _CONS or w[-2] == "्":
            return w
        return w + "्अ" if w in _HOLD_WORDS or w.endswith(_HOLD_ENDINGS) else w

    def one(m):
        w = m.group(0)
        for suf in _SUFFIXES:  # आजको -> आजऽको: the stem keeps its spoken a before a case ending
            if w.endswith(suf) and len(w) > len(suf) + 1:
                stem = w[:-len(suf)]
                h = held(stem)
                if h != stem:
                    return h + suf
        return held(w)
    return re.sub(r"[ऀ-ॿ]+", one, text)
