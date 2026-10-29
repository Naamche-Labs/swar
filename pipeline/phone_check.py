"""Sound-level English check for picking takes (2026-10-27). Whisper forgives a wrong sound when it can still guess the
word ("hoon-estly" -> "honestly", "ma-chi-ne" -> "machine"), so a take Whisper hears perfectly can still sound wrong.
This scores the SOUNDS: espeak-ng's US pronunciation of the text against what a phoneme recogniser
(facebook/wav2vec2-lv-60-espeak-cv-ft) hears, after merging spellings the two write differently for the same sound
(flap t, the schwas, syllabic n/l, r-coloured vowels, cot/caught). Returns the share of sounds that are wrong.
Needs `phonemizer` and the espeak-ng binary; without them per() returns None and the caller falls back to Whisper.
Same scoring as our per-word phoneme check, where long passages in 5 voices read 7.7-8.5% wrong sounds."""
import re

_M = {}
DROP = re.compile(r"[ˈˌːˑ\s.|-]")
CANON = {"ɾ": "t", "ɐ": "ə", "ʌ": "ə", "ᵻ": "ɪ", "ɜ": "ə", "ɡ": "g", "ɒ": "ɑ", "o": "oʊ", "e": "eɪ", "r": "ɹ", "ʔ": "t",
         "ɔ": "ɑ", "oɪ": "ɔɪ"}
SPLIT = {"n̩": ["ə", "n"], "l̩": ["ə", "l"], "ɚ": ["ə", "ɹ"], "ɝ": ["ə", "ɹ"], "m̩": ["ə", "m"], "əl": ["ə", "l"],
         "ɑɹ": ["ɑ", "ɹ"], "ɔɹ": ["ɑ", "ɹ"], "oɹ": ["ɑ", "ɹ"], "ʊɹ": ["ʊ", "ɹ"], "ɛɹ": ["ɛ", "ɹ"], "ɪɹ": ["ɪ", "ɹ"],
         "aɪɚ": ["aɪ", "ə", "ɹ"], "aɪə": ["aɪ", "ə"], "iə": ["i", "ə"], "aʊɚ": ["aʊ", "ə", "ɹ"]}


def _canon(ph):
    out = []
    for p in ph:
        p = p.replace("ː", "")
        out += [CANON.get(q, q) for q in SPLIT.get(p, [p])]
    return out


def _load():
    if "asr" not in _M:
        try:
            import torch
            from phonemizer import phonemize  # noqa: F401
            from transformers import AutoModelForCTC, AutoProcessor
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
            _M.update(proc=AutoProcessor.from_pretrained(name), asr=AutoModelForCTC.from_pretrained(name).to(dev).eval(), dev=dev)
        except Exception as e:  # noqa: BLE001
            print("phone_check unavailable:", e, flush=True)
            _M["asr"] = None
    return _M["asr"]


def ref_phones(text):
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    t = re.sub(r"[^A-Za-z' ,.?!]", " ", text)
    p = phonemize(t, language="en-us", backend="espeak", separator=Separator(phone=" ", word=" | ", syllable=""), strip=True,
                  preserve_punctuation=False, with_stress=False)
    return _canon([x for x in DROP.sub(" ", p).split() if x])


def hyp_phones(path):
    import librosa
    import torch
    w = librosa.load(path, sr=16000)[0]
    if len(w) < 1600:
        return []
    with torch.no_grad():
        x = _M["proc"](w, sampling_rate=16000, return_tensors="pt").input_values.to(_M["dev"])
        ids = _M["asr"](x).logits.argmax(-1)
    return _canon([x for x in DROP.sub(" ", _M["proc"].batch_decode(ids)[0]).split() if x])


def _ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]


def per(text, path):
    """share of the text's English sounds said wrong in `path`, or None when the checker is not installed"""
    if not _load():
        return None
    r = ref_phones(text)
    return _ed(r, hyp_phones(path)) / max(1, len(r))
