"""Hard English words for the bilingual checkpoint (base v5, English in Latin letters), 2026-10-27.

A word whose spelling does not say how it sounds ("tough", "Wednesday", "machines") is written the way it sounds before
the model sees it, so the model's letter-to-sound habits land on the right sounds. Every entry was measured, not guessed:
the same sentences in 3-4 voices with and without the respelling, each take read by a phoneme recogniser
(wav2vec2-lv-60-espeak) against espeak's US pronunciation, kept only when the respelling gave at least 5 points fewer
wrong sounds (a spelling tester and per-word phoneme scoring). Long passages (6 kinds of text, 5 voices): 8.5% wrong
sounds as typed, 8.1% with this list.
Do not add a word here without measuring it: the base v4 list hurt base v5 ("honestly" written "onestly" came out
"wun-estly", "again" written "agen" came out "ay-jun", "design" written "dizine" came out "di-zeen").
  fix_english(text) -> text with listed English words respelled; Devanagari, case of the first letter, punctuation kept"""
import re

LEX = {
    "tough": "tuff", "rough": "ruff", "enough": "inuff", "says": "sez", "champagne": "shampain", "wednesday": "wenzday",
    "wednesdays": "wenzdays", "salmon": "sammon", "ensure": "enshoor", "castle": "cassel", "chaos": "kayos",
    "chef": "shef", "sign": "sine", "women": "wimmin", "colonel": "kernel", "busy": "bizzy", "often": "offen",
    "thorough": "thurro", "yacht": "yot", "minute": "minnit", "minutes": "minnits", "restaurant": "restront",
    "machine": "mesheen", "machines": "mesheens", "pretty": "pritty", "brochure": "broshoor", "knowledge": "nolij",
    "knowledgeable": "nolijable", "though": "thoh", "knew": "new", "parachute": "parashoot", "listen": "lissen",
    "listens": "lissens", "sure": "shoor", "stomach": "stumak", "once": "wunce", "comfortable": "kumfterbul",
    "foreign": "forin", "subtle": "suttul", "people": "peepul", "receipt": "reseet", "chicago": "shikago",
    "business": "bizness", "choir": "kwyer", "actually": "aktually", "meant": "mennt",
    # base v6 said "honestly" as typed with 16% of its sounds wrong; of 7 spellings, "onnist" was best (9.5%)
    "honest": "onnist", "honestly": "onnistly", "honesty": "onnisty",
}
_WORD = re.compile(r"[A-Za-z']+")


def fix_english(text, extra=None):
    lex = {**LEX, **(extra or {})}

    def one(m):
        w = m.group(0)
        r = lex.get(w.lower())
        if not r:
            return w
        return r[0].upper() + r[1:] if w[0].isupper() else r
    return _WORD.sub(one, text)
