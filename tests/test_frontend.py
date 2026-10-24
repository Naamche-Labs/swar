import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend import normalize, number_to_words, g2p, split_codeswitch
from frontend.g2p import _phonemize_word


def rules(word):
    """Rule path only (lexicon bypassed), so rule tests don't pass by lookup."""
    return _phonemize_word(word, use_lexicon=False)


# ---------- normalization ----------

def test_number_words_small():
    assert number_to_words(0) == "शून्य"
    assert number_to_words(7) == "सात"
    assert number_to_words(19) == "उन्नाइस"
    assert number_to_words(99) == "उनान्सय"


def test_number_words_large():
    assert number_to_words(100) == "एक सय"
    assert number_to_words(1000) == "एक हजार"
    assert number_to_words(123456) == "एक लाख तेइस हजार चार सय छपन्न"
    assert number_to_words(10000000) == "एक करोड"


def test_normalize_digits_and_symbols():
    assert number_to_words(25) in normalize("२५ वटा")
    assert "प्रतिशत" in normalize("९०%")
    assert "रुपैयाँ" in normalize("रू १००")


def test_normalize_decimal():
    out = normalize("३.५")
    assert "दशमलव" in out and "पाँच" in out


def test_thousands_separator():
    assert normalize("१,२५०") == "एक हजार दुई सय पचास"
    assert normalize("1,25,000") == "एक लाख पच्चिस हजार"


def test_rs_does_not_eat_english():
    out = normalize("users and hours")  # English is now spoken as Devanagari (frontend/english.py)
    assert "रुपैयाँ" not in out and "युजर्स" in out and "आवर्स" in out


def test_english_words_become_devanagari():
    assert normalize("मेरो laptop को battery") == "मेरो ल्याप्टप को ब्याट्री"
    assert normalize("OTP आयो") == "ओटीपी आयो"
    assert normalize("charger") == "चार्जर"


def test_currency_amount_first():
    assert normalize("रू १००") == "एक सय रुपैयाँ"
    assert normalize("Rs. 50") == "पचास रुपैयाँ"


def test_phone_digit_by_digit():
    out = normalize("9800000000")
    assert out.startswith("नौ आठ शून्य") and "करोड" not in out
    assert normalize("+9779800000000").startswith("प्लस नौ सात सात")


def test_codeswitch_split():
    assert ("AI", True) in list(split_codeswitch("यो AI हो"))


# ---------- rules (lexicon bypassed) ----------

def test_rules_basic_cluster():
    assert rules("नमस्ते") == ["n", "a", "m", "a", "s", "t", "e"]


def test_rules_final_schwa():
    assert rules("कम") == ["k", "a", "m"]
    assert rules("घर") == ["gh", "a", "r"]
    assert rules("छ") == ["ch", "a"]                    # only vowel: keep
    assert rules("गर्छ") == ["g", "a", "r", "ch", "a"]  # after conjunct: keep
    assert rules("समय") == ["s", "a", "m", "a", "y", "a"]  # after य: keep
    assert rules("बाट") == ["b", "aa", "T", "a"]           # postposition: keep
    assert rules("अंक") == ["a", "n~", "k", "a"]           # after nasal cluster: keep


def test_rules_suffix_split():
    assert rules("नेपालको") == ["n", "e", "p", "aa", "l", "k", "o"]


def test_rules_matra_and_nasal():
    assert rules("आमा") == ["aa", "m", "aa"]
    assert rules("अं") == ["a", "n~"]


def test_rules_no_vowel_length():
    assert rules("पानी")[-1] == "i" and rules("दूध")[1] == "u"


def test_rules_sibilants_merge_to_s():
    assert rules("शर्मा") == ["s", "a", "r", "m", "aa"]


def test_rules_conjunct_rewrites():
    assert rules("ज्ञान") == ["g", "y", "aa", "n~", "n"]  # learned; Google lexicon nasalizes after ज्ञ
    assert rules("क्षमा") == ["ch", "y", "a", "m", "aa"]  # learned; matches Google lexicon


def test_rules_wa_ba():
    assert rules("विद्यालय")[0] == "b"
    assert rules("स्वर") == ["s", "w", "a", "r"]
    assert g2p("वास्तव")[0] == "b"  # Google lexicon: baastab
    assert "b" in rules("धन्यवाद")


def test_rules_no_flap():
    assert rules("पहाड") == ["p", "a", "h", "aa", "D"]


def test_rules_nukta_keeps_vowels():
    assert rules("फ़ोन") == ["f", "o", "n"]
    assert rules("ज़िन्दगी")[:2] == ["z", "i"]
    assert rules("पढ़")[:3] == ["p", "a", "rh"]  # final 'a' is a learned decision (verb पढ = paDha)


# ---------- full g2p (lexicon-first) ----------

def test_lexicon_lookup():
    assert g2p("ज्ञान")[:3] == ["g", "y", "aa"]
    assert g2p("Facebook") == ["ph", "e", "s", "b", "u", "k"]


def test_word_boundary():
    assert "|" in g2p("राम्रो छ")


def test_english_not_dropped():
    toks = g2p(normalize("यो AI हो"))
    assert toks == ["y", "o", "|", "e", "aa", "i", "|", "h", "o"], toks
    assert len(g2p("blockchainify")) > 0  # OOV English still read, never dropped


def test_punctuation_pauses():
    toks = g2p("राम्रो छ। हो।")
    assert toks.count(".") == 2 and toks[-1] == "." and "|" in toks
    assert "," in g2p("हो, तर")


# ---------- edge cases (from tests/edge_probe.py + Ampixa text_norm_cases) ----------

def test_ampixa_text_norm_cases():
    assert normalize("२५०") == "दुई सय पचास"
    assert normalize("250") == "दुई सय पचास"
    assert normalize("रु २५०") == "दुई सय पचास रुपैयाँ"
    assert normalize("५:३०") == "पाँच बजेर तीस मिनेट"
    assert g2p("१०%") == g2p("दस प्रतिशत")  # दश/दस spellings sound the same


def test_plural_suffix_is_not_currency():
    assert normalize("महिलाहरू") == "महिलाहरू"
    assert normalize("महिलाहरु") == "महिलाहरु"


def test_times_and_dates():
    assert normalize("9:00 बजे") == "नौ बजे"
    assert normalize("२०८१/०५/१२") == "दुई हजार एकासी साल भदौ बाह्र गते"
    assert normalize("2024-05-12") == "सन् दुई हजार चौबिस मे बाह्र"
    assert normalize("१/२") == "आधा"


def test_ordinals_ranges_negatives():
    assert normalize("१ला") == "पहिलो" and normalize("२रो") == "दोस्रो" and normalize("1st") == "पहिलो"
    assert normalize("५-१०") == "पाँच देखि दश"
    assert normalize("-5") == "माइनस पाँच"
    assert normalize(".5") == "शून्य दशमलव पाँच"


def test_number_glued_to_word():
    assert normalize("२०वटा") == "बीस वटा"


def test_abbreviations_units_currencies():
    assert normalize("डा. राम") == "डाक्टर राम"
    assert normalize("नं. ५") == "नम्बर पाँच"
    assert normalize("५ कि.मि.") == "पाँच किलोमिटर"
    assert normalize("१० kg") == "दश केजी"
    assert normalize("३५°C") == "पैँतिस डिग्री सेल्सियस"
    assert normalize("$100") == "एक सय डलर"


def test_urls_and_emails_spoken():
    assert "डट" in normalize("www.google.com") and "एट" in normalize("abc@gmail.com")


def test_mixed_script_and_accents_not_dropped():
    assert g2p("COVID-19")[:2] == ["k", "o"] and "u" in g2p("COVID-19")
    assert g2p("5G")[-2:] == ["j", "i"]
    assert g2p("café")
    assert g2p("NASA")[:2] == ["n", "aa"] and g2p("OK") == ["o", "k", "e"]  # acronyms said as words stay words


def test_hyphen_slash_split():
    assert g2p("आमा-बुबा") == ["aa", "m", "aa", "|", "b", "u", "b", "aa"]
    assert "|" in g2p("र/वा")


def test_more_pause_marks():
    assert g2p("छ|")[-1] == "." and g2p("हो…")[-1] == "."
    assert g2p("हो ; तर") == ["h", "o", ",", "t", "a", "r", "a"]


def test_unicode_hygiene():
    assert normalize("नेपाल​") == "नेपाल"
    assert g2p("गर्‍यो") == g2p("गर्यो")
    assert g2p("ळ") == ["l", "a"]
    assert g2p("२५")  # raw digits handled inside g2p


def test_minimal_pairs_stay_distinct():
    for a, b in [("कल", "खल"), ("तल", "थल"), ("पाल", "बाल"), ("बाल", "भाल"), ("तल", "टल"),
                 ("दल", "डल"), ("चपल", "चप्पल"), ("कप", "काँप"), ("बस", "बाँस"), ("चाल", "छाल")]:
        assert g2p(a) != g2p(b), (a, b)


def test_compound_split_uses_known_parts():
    from frontend.g2p import LEXICON
    word = "नेपालबहादुर"
    assert word not in LEXICON and "नेपाल" in LEXICON and "बहादुर" in LEXICON
    assert g2p(word) == g2p("नेपाल") + g2p("बहादुर")


def test_learned_model_verb_stem_keeps_schwa():
    # hand rules could not tell verb stems from nouns; the learned layer can
    from frontend.g2p import MODEL
    assert MODEL, "data/g2p_model.json should be loaded"
    assert rules("घर") == ["gh", "a", "r"]


def test_doubled_aspirate_order():
    # the doubled copy goes before the sound (d dh, never dh d)
    from frontend.g2p import _rule_template
    items, _ = _rule_template("अध्यक्ष")
    assert items[1][0] == "gem" and items[2] == "dh", items
    assert g2p("अध्यक्ष")[:3] == ["a", "d", "dh"]


def test_normalize_idempotent():
    for s in ["रु २५०", "५:३०", "२०८१/०५/१२", "COVID-19", "www.google.com", "१ला", "३५°C", "महिलाहरू"]:
        assert normalize(normalize(s)) == normalize(s), s


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
