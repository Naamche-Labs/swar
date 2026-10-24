# nepa-newa-text-frontend

Devanagari → phone-token frontend for Nepali TTS. eSpeak-free, audit-traced,
lexicon-first.

---

## ⚠️ Important: this G2P encodes **Kathmandu Newari-influenced Nepali**

This is the most important thing to know before you use this. We document Kathmandu
Nepali as analysed in **Khatiwada (2009),** *Journal of the International Phonetic
Association* 39(3) — the standard reference IPA description of the dialect spoken
in the Kathmandu Valley, which has substantial Newari (Tibeto-Burman) substrate
influence on consonants and vowels.

**Concrete consequences any TTS built on this frontend will have:**

| Devanagari | This G2P emits | Pan-Nepali / Indo-Aryan tradition |
|---|---|---|
| `च`        | `/ts/`         | `/tʃ/` (English "ch")            |
| `ज`        | `/dz/`         | `/dʒ/` (English "j")             |
| `छ`        | `/tsʰ/`        | `/tʃʰ/`                          |
| `झ`        | `/dzʱ/`        | `/dʒʱ/`                          |
| `व`        | `/w/`          | `/v/` in some traditions         |
| `ि` / `ी`   | merged → `i`   | distinguished short / long       |

If you say `चीन` (China) as **"chīn"** and the model produces **"tsīn"**, that is
the policy doing what it says on the tin, not a model defect. If your target
audience is Hill / Eastern Nepali speakers, English-influenced urban Indo-Aryan
speech, or Hindi-style pronunciation, **this is not the right frontend** — fork it
and rewrite the relevant phone mappings, or wait for a future v2 with a dialect
flag.

For Kathmandu Newari Nepali, BS / urban speech in the Valley, formal radio /
television Nepali (which broadly follows the Kathmandu standard), and ASR-style
literal-grapheme-to-IPA pipelines, the policy is correct and well-attested.

---

## Clear-Standard Profile

The default package remains the Kathmandu Newari-influenced frontend above.
The optional `real_nepali` package provides a clear-standard production profile
used by the Piper/TTS experiments. It preserves the base frontend and rewrites
target phones such as `ts/tsh/dz/dzh` to `ch/chh/j/jh` for voices where that
matches the intended listener target.

Use it explicitly:

```python
from real_nepali import g2p

g2p.phonemize_text("चार छ आज", profile="real_nepali_v0.2")
```

Do not treat `real_nepali` as a silent replacement for the default dialect
policy; it is a named profile.

---

## Why this exists

eSpeak NG's Nepali support is unreliable for production TTS. Existing Indic G2Ps
either lump Nepali into Hindi rules (wrong /tʃ/ / /v/ / vowel length defaults) or
require manual phoneme markup. We needed:

1. A lexicon-first G2P with a clear linguistic anchor (Khatiwada 2009)
2. Per-word audit trace so failures can be traced to lexicon vs rule fallback
3. A phone inventory that's a clean subset of any standard IPA-based scheme
4. No eSpeak runtime dependency
5. Fast, deterministic, side-effect-free

This package delivers all five.

## Phonological research the package implements

### Phone inventory (65 audible + 4 special)

Following Khatiwada 2009 with extensions for gemination, retroflex aspirates, and
loan phones common in modern Kathmandu Nepali:

- **6 oral vowels** — `i e ax aa o u`
- **5 nasal vowels** (phonemic per Khatiwada) — `in en axn aan un`
- **13 diphthongs** (oral + nasal) — `axj axw aaj aaw oj ew ow / axjn axwn aajn aawn ojn own`
- **16 stops** (4 places × 4 phonations) — `p ph b bh / t th d dh / tx txh dx dxh / k kh g gh`
- **4 affricates** — `ts tsh dz dzh`  *(see dialect note above)*
- **5 nasals** — `m n ng ny nx`
- **6 sonorants/fricatives** — `r s h l y w`
- **4 careful** — `sh sx ny nx` *(orthographic distinctions retained for careful style)*
- **2 loan** — `f z`
- **Geminates** — `<phone>:` for any consonant
- **Specials** — `_` (PAD), `^` (BOS), `$` (EOS), `#` (BLANK)

Full table in [`data/frontend/phones.tsv`](data/frontend/phones.tsv) with IPA
correspondences in [`data/frontend/ipa_map.tsv`](data/frontend/ipa_map.tsv).

### Phonological policy decisions baked in

These are the policy calls we made and their rationale (from research recorded in
the lexicon and minimal-pair tests):

1. **/ts/ for च, /dz/ for ज** — Khatiwada documents these as the dialectal norm
   in Kathmandu Nepali. The /tʃ/ / /dʒ/ realisations exist in the same speakers'
   careful speech but `/ts/`-`/dz/` are the unmarked production targets.
2. **No phonemic vowel length** — Khatiwada notes the `ि` / `ी` distinction is
   orthographic-only in Kathmandu Nepali. Both map to `i`. Same for `ु` / `ू` → `u`.
3. **Schwa is `/ə/` rendered as `ax`** — to keep ASCII-friendly phone tokens.
   Final-schwa deletion follows standard Nepali rules (covered by lexicon entries
   for the affected words).
4. **Aspirated retroflex contrast preserved** — `tx` / `txh` / `dx` / `dxh` are
   distinct phones in our inventory because Khatiwada and minimal-pair listening
   tests both show contrast (e.g. टुक्रा / ठुक्रा).
5. **Nasal vowels are full phonemes** — not nasal-+-/n/ sequences, because Nepali
   contrasts pairs like हास / हाँस minimally on nasalisation alone.
6. **Gemination is encoded** — many Nepali words contain true geminate consonants
   (सक्कल, गद्दा), and minimal pairs distinguish them from singletons (सकल / सक्कल).

### Coverage and benchmark

- **Lexicon size:** 54K seed entries from Google language-resources (CC-BY-4.0)
  + curated additions in `gold_lexicon.tsv`
- **Coverage on natural Nepali production text:** ~74-99% words hit the lexicon
  cleanly, ~11% via suffix-strip, ~15% via rule fallback (dependent on text source)
- **NepTTS-Bench minimal-pair contrast preservation:** **99.5%** — for 149 minimal
  pairs designed to test the contrasts in our inventory, the G2P emits distinct
  phone sequences for the two members of the pair in 99.5% of cases. The remaining
  0.5% are cases where the lexicon collapses two valid spellings to the same
  underlying form.

The minimal pairs are in [`data/frontend/minimal_pairs.tsv`](data/frontend/minimal_pairs.tsv);
the bench source is at [`data/external/neptts_bench/`](data/external/neptts_bench/).

## What this gives you

- **G2P** — Devanagari word → list of project phones
- **Tokenization** — sentence chunking, word splitting, danda handling
- **Normalization** — numerals, abbreviations, punctuation
- **Prosody chunker** — clause-level phrasing for natural pause placement
- **Audit trace** — every phone tagged with its source so failures are debuggable

## Install

```bash
git clone https://github.com/Ampixa/nepa-newa-text-frontend.git
cd nepa-newa-text-frontend
pip install -e .
```

(Sibling-relative `data/` lookup means `pip install -e .` is what you want for
now; PyPI distribution will package the data files in-tree.)

## Usage

```python
from nepali_frontend.g2p import phonemizer as ph

words = ph.phonemize_text("नेपाल हाम्रो देश हो।")
for w in words:
    print(w.text, "->", w.phones, "(source:", w.source + ")")

# नेपाल -> ['n', 'e', '.', 'p', 'aa', 'l']           (lexicon)
# हाम्रो -> ['h', 'aa', 'm', '.', 'm', 'r', 'o']     (lexicon)
# देश   -> ['d', 'e', 's']                            (lexicon)
# हो    -> ['h', 'o']                                 (lexicon)
```

The `Word` objects expose:
- `.text` — original Devanagari token
- `.phones` — list of phone strings
- `.source` — `'lexicon'`, `'suffix_strip'`, or `'g2p_rule'`

The `'.'` token marks syllable boundary inside a word; `'|'` marks word boundary
inside a multi-word phrase.

Single-word lookup with full trace:

```python
r = ph.phonemize_word("पश्चिम")
print(r.phones)   # ['p', 'ax', 's', '.', 'ts', 'i', 'm']
print(r.source)   # 'lexicon'
```

## The audit trace — why it matters

Every phone has a tagged source. When a TTS built on this frontend mispronounces
a word, you can immediately see whether the fault is:

| source | meaning | fix path |
|---|---|---|
| `lexicon`        | direct hit in the curated lexicon | edit `gold_lexicon.tsv` |
| `suffix_strip`   | base form found after stripping a known inflection | check the suffix rule, or add the inflected form to the lexicon |
| `g2p_rule`       | rule-based fallback (Akshara → phone) | check `akshara_cases.tsv`, possibly add rule |

This converts G2P bugs from "the model sounds wrong somewhere" into specific TSV
rows you can edit.

## Trained TTS using this frontend

A multi-speaker Piper-VITS Nepali TTS trained with this G2P:

- **[`ampixa/nepali-voices-v0`](https://huggingface.co/ampixa/nepali-voices-v0)** —
  419 speakers, ~22h training audio, ONNX-exportable for ARM/CPU inference

A 5-voice clean cold-start follow-up will land at `ampixa/nepali-tts-v1` once
training completes.

## What's NOT in scope for this version

- Other Nepali dialects (Hill, Eastern, Western) — would need a different policy
  table and a different lexicon
- Code-switched Nepali-English text — Latin-script tokens are dropped silently
- Sanskrit-loaded literary register — handled per the lexicon if entries exist,
  no special rules
- Hindi text — wrong /tʃ/, /v/, vowel length defaults

## Files

| path | what |
|---|---|
| `nepali_frontend/` | Python package |
| `data/frontend/phones.tsv` | canonical 65-phone inventory |
| `data/frontend/gold_lexicon.tsv` | manually curated word→phones (highest confidence) |
| `data/frontend/candidates_lexicon.tsv` | seed lexicon from Google language-resources |
| `data/frontend/ipa_map.tsv` | project phones ↔ IPA correspondence |
| `data/frontend/akshara_cases.tsv` | rule-based fallback cases (Akshara → phone) |
| `data/frontend/minimal_pairs.tsv` | curated minimal pairs for evaluation |
| `data/frontend/text_norm_cases.tsv` | text-normalisation rules (numerals, abbreviations) |
| `data/frontend/prosody_cases.tsv` | prosody chunker test cases |
| `data/external/google_ne_lr/` | upstream lexicon (CC-BY-4.0) — see `NOTICE.md` |
| `data/external/neptts_bench/` | NepTTS-Bench sentence list (CC-BY-4.0) |
| `tests/` | pytest unit tests |

## License

**MIT** for code in `nepali_frontend/` and curated TSVs in `data/frontend/`.

Upstream third-party data in `data/external/` carries upstream licenses
(CC-BY-4.0 for both the Google language-resources lexicon and NepTTS-Bench
sentences). See each subdirectory's `NOTICE.md` for attribution requirements.

## Citation

If you use this frontend in research or production, please cite both the
phonological source and this package:

```
@article{khatiwada2009nepali,
  title   = {Nepali},
  author  = {Khatiwada, R.},
  journal = {Journal of the International Phonetic Association},
  volume  = {39}, number = {3}, pages = {373--380},
  year    = {2009},
}

@misc{nepali_text_frontend_2026,
  title  = {nepa-newa-text-frontend: Devanagari G2P for Kathmandu Nepali TTS},
  author = {Ampixa},
  year   = {2026},
  url    = {https://github.com/Ampixa/nepa-newa-text-frontend}
}
```
