"""Generate a phonetically-balanced Nepali recording script for the exact-clone
path. The target speaker reads these aloud (~15-20 min, quiet room, phone mic)
to produce fine-tuning data for a near 1-to-1 voice clone.

Selects a sentence subset that covers the full G2P phoneme inventory and
reports coverage. Output: review/recording_script.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend import g2p, normalize
from frontend.g2p import PHONEME_INVENTORY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everyday Nepali sentences chosen to span consonants, vowels, clusters,
# nasals, retroflex/dental, affricates, numbers, names and English loanwords.
POOL = [
    "नमस्ते, मेरो नाम के हो भनेर सोध्नुभयो?",
    "आज काठमाडौँमा मौसम धेरै राम्रो छ।",
    "मलाई नेपाली खाना, विशेष गरी दालभात तरकारी मन पर्छ।",
    "हामी बिहान सबेरै उठेर घुम्न जान्छौँ।",
    "यो किताब पढेपछि मैले धेरै कुरा सिकेँ।",
    "उनको छोरा र छोरी दुवै विद्यालय जान्छन्।",
    "पहाड र हिमालको दृश्य साँच्चै मनमोहक छ।",
    "कृपया ढोका बिस्तारै बन्द गर्नुहोस्।",
    "त्यो पसलमा सयौँ थरीका सामान पाइन्छ।",
    "ज्ञान र क्षमताले मात्र मानिस ठूलो हुन्छ।",
    "वर्षाको पानीले खेतबारी हरियाली भयो।",
    "बुबाआमाले हामीलाई सधैँ माया गर्नुहुन्छ।",
    "रू एक हजार दुई सय पचास तिर्नुपर्‍यो।",
    "नब्बे प्रतिशत विद्यार्थी परीक्षामा उत्तीर्ण भए।",
    "कम्प्युटर र इन्टरनेटले संसार नजिक बनायो।",
    "मोबाइल फोनमा धेरै उपयोगी एप छन्।",
    "सिंहको गर्जन जंगलभरि गुन्जियो।",
    "अंकगणित र विज्ञान मेरा मनपर्ने विषय हुन्।",
    "भोलि बिहान हामी गाउँतिर जाने योजना छ।",
    "हिजो साँझ आकाशमा राम्रो इन्द्रेणी देखियो।",
    "चिया पिउँदै हामीले लामो कुराकानी गर्‍यौँ।",
    "ठूलो स्वरमा नबोल्नुहोस्, बच्चा सुतिरहेको छ।",
    "डाक्टरले बिरामीलाई औषधि खान सल्लाह दिए।",
    "फूलको बगैँचामा रंगीचंगी पुतली उडिरहेका छन्।",
    "घरको छानामा घाम लाग्दा न्यानो हुन्छ।",
    "ऋतुअनुसार हाम्रो खानपान पनि फेरिन्छ।",
    "औँलामा औँठी लगाएर उनी मुस्कुराइन्।",
    "प्रधानमन्त्रीले राष्ट्रलाई सम्बोधन गर्नुभयो।",
    "सडकमा सावधानीपूर्वक हिँड्नुहोस्।",
    "झरनाको शीतल पानीले मन प्रफुल्ल भयो।",
]


def phoneme_set(tok_list):
    s = set(tok_list)
    s.discard("|")
    return s


def run():
    inventory = set(PHONEME_INVENTORY)
    covered, chosen = set(), []
    # greedy: add sentences that bring new phonemes first
    remaining = list(POOL)
    while remaining:
        best, best_new = None, -1
        for s in remaining:
            new = len(phoneme_set(g2p(normalize(s))) - covered)
            if new > best_new:
                best, best_new = s, new
        chosen.append(best)
        covered |= phoneme_set(g2p(normalize(best)))
        remaining.remove(best)
        if covered >= inventory and best_new == 0:
            break
    missing = inventory - covered
    os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
    out = os.path.join(ROOT, "review", "recording_script.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("NEPALI VOICE-CLONE RECORDING SCRIPT\n")
        f.write("Read each line clearly at a natural pace. Quiet room, phone/laptop mic is fine.\n")
        f.write("Pause ~1s between lines. Re-record any line you stumble on.\n")
        f.write("=" * 60 + "\n\n")
        for i, s in enumerate(chosen, 1):
            f.write(f"{i:2d}. {s}\n\n")
    pct = 100 * len(covered & inventory) / len(inventory)
    print(f"wrote {len(chosen)} sentences -> {out}")
    print(f"phoneme coverage: {len(covered & inventory)}/{len(inventory)} ({pct:.0f}%)")
    if missing:
        print(f"uncovered phonemes (rare; add a sentence later): {sorted(missing)}")


if __name__ == "__main__":
    run()
