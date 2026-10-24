"""Build the listening page with all audio embedded as data URIs."""
import base64
import json
import os
import sys

sys.path.insert(0, "/Users/arjanchaudharyy/nepali-tts")
from frontend.romanize import romanize  # noqa: E402

D = "/Users/arjanchaudharyy/nepali-tts/samples/compare"
OUT = "/private/tmp/claude-501/-Users-arjanchaudharyy/ca45a637-50d9-46b4-a2f9-b06c6ee8f31c/scratchpad/nepali_voice_check.html"
S = json.load(open(f"{D}/sentences.json"))
CAT = {"phonological_minimal_pairs": "Minimal pair", "question_intonation": "Question", "emotion": "Emotion",
       "contrastive_stress": "Stress", "robustness": "Tongue twister", "phrases": "Short phrase",
       "homographs": "Homograph", "newspaper_ambiguities": "Headline"}
SYS = [("ours", "Ours", "F5 · IndicF5 → Nepali"), ("tingting", "TingTing Subina", "best published system"),
       ("elevenlabs", "ElevenLabs v3", "commercial")]
EXTRA = [("long", "Long paragraph", "नेपाल हिमालको काखमा रहेको सुन्दर देश हो। यहाँ सगरमाथासहित संसारका आठ सर्वोच्च शिखरमध्ये आठवटा छन्। हरेक वर्ष हजारौँ पर्यटक हिमाल चढ्न, जङ्गल सफारी गर्न र यहाँको संस्कृति हेर्न आउँछन्। काठमाडौं उपत्यकाका पुराना दरबार र मन्दिरहरू युनेस्कोको विश्व सम्पदा सूचीमा परेका छन्।"),
         ("names", "Place names", "आज विराटनगर, भरतपुर र धनगढीबाट आएका प्रतिनिधिहरूले सिंहदरबारमा प्रधानमन्त्रीसँग भेट गरे।"),
         ("numbers", "Numbers", "यो वर्ष २०८१ सालमा कुल १२,५०० विद्यार्थीले परीक्षा दिए, जसमध्ये ८७ प्रतिशत उत्तीर्ण भए।"),
         ("codeswitch", "English mixed in", "मेरो laptop को battery छिटो सकिन्छ, त्यसैले म charger सधैँ बोकेर हिँड्छु।")]


def src(name):
    return "data:audio/mpeg;base64," + base64.b64encode(open(f"{D}/mp3/{name}.mp3", "rb").read()).decode()


def player(name, label, sub, mine=False):
    return (f'<div class="take{" mine" if mine else ""}"><div class="who"><b>{label}</b><span>{sub}</span></div>'
            f'<audio controls preload="none" src="{src(name)}"></audio></div>')


rows = []
for k, v in S.items():
    takes = "".join(player(f"{k}_{s}", lab, sub, s == "ours") for s, lab, sub in SYS)
    rows.append(f'<article class="item"><header><span class="tag">{CAT[v["cat"]]}</span>'
                f'<p class="np" lang="ne">{v["text"]}</p><p class="rom">{romanize(v["text"])}</p></header>'
                f'<div class="takes">{takes}</div></article>')

extras = []
for k, lab, text in EXTRA:
    takes = (player(f"extra_{k}_female_news", "Ours", "female voice", True)
             + player(f"extra_{k}_male_wiki", "Ours", "male voice", True))
    extras.append(f'<article class="item"><header><span class="tag">{lab}</span>'
                  f'<p class="np long" lang="ne">{text}</p></header><div class="takes two">{takes}</div></article>')

# (name, Devanagari-only MMS, XLS-R, ours) mean-file CER %
SCORES = [("Ours (female prompt)", 6.14, 9.41, True), ("TingTing Subina", 13.39, 16.30, False),
          ("Gemini", 15.31, 18.12, False), ("ElevenLabs v3", 15.70, 14.44, False),
          ("TingTing Asmita", 16.71, 20.66, False), ("Piper", 18.68, 17.75, False),
          ("Edge TTS Sagar", 19.95, 15.53, False)]
mx = 22.0
bars = []
for name, mms, xl, mine in SCORES:
    def bar(v, cls):
        if v is None:
            return f'<div class="bar {cls} na"><span>no score</span></div>'
        return f'<div class="bar {cls}"><i style="width:{100 * v / mx:.1f}%"></i><span>{v:.1f}%</span></div>'
    bars.append(f'<div class="srow{" mine" if mine else ""}"><div class="sname">{name}</div>'
                f'{bar(xl, "xl")}{bar(mms, "mms")}</div>')

clones = []
if os.path.exists(f"{D}/clones.json"):
    for c in json.load(open(f"{D}/clones.json")):
        i = c["i"]
        takes = (player(f"clone{i}_prompt", "Their voice clip", "the only input (one short recording)")
                 + player(f"clone{i}_ours", "Our clone", "new sentence, never heard", True)
                 + player(f"clone{i}_real", "The real person", "saying that sentence"))
        clones.append(f'<article class="item"><header><span class="tag">Unseen speaker {i + 1}</span>'
                      f'<p class="np long" lang="ne">{c["text"]}</p><p class="rom">{romanize(c["text"])}</p></header><div class="takes">{takes}</div></article>')

mine = []
AJ = "/Users/arjanchaudharyy/nepali-tts/samples/arjan/texts.json"
if os.path.exists(AJ):
    mine.append('<article class="item"><header><span class="tag">Your recording (the only input)</span>'
                '<p class="np long" lang="ne">विराटनगर अत्यन्त राम्रो ठाउँ छ। विराटनगरका मान्छेहरू, विराटनगरको खानपिन, मासुभात, मिठाई, नजिकैको जोगबनी, सबै राम्रो छ।</p>'
                f'<p class="rom">{romanize("विराटनगर अत्यन्त राम्रो ठाउँ छ। विराटनगरका मान्छेहरू, विराटनगरको खानपिन, मासुभात, मिठाई, नजिकैको जोगबनी, सबै राम्रो छ।")}</p></header>'
                f'<div class="takes one">{player("arjan_prompt", "Aarjan", "picked automatically from your 70 s recording")}</div></article>')
    for k, text in json.load(open(AJ)).items():
        takes = (player(f"arjan_{k}_auto", "New", "auto-picked clip + your pace", True)
                 + player(f"arjan_{k}_old", "Before", "your read intro clip, its slower pace"))
        mine.append(f'<article class="item"><header><span class="tag">{k}</span>'
                    f'<p class="np long" lang="ne">{text}</p><p class="rom">{romanize(text)}</p></header><div class="takes two">{takes}</div></article>')

html = open(os.path.join(os.path.dirname(OUT), "listen_template.html"), encoding="utf-8").read()
html = html.replace("{{ROWS}}", "\n".join(rows)).replace("{{EXTRAS}}", "\n".join(extras)).replace("{{BARS}}", "\n".join(bars)).replace("{{CLONES}}", "\n".join(clones)).replace("{{MINE}}", "\n".join(mine))
open(OUT, "w", encoding="utf-8").write(html)
print(OUT, round(len(html) / 1e6, 2), "MB")
