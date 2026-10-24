"""Does the bilingual checkpoint fix English? Pure English and mixed lines, two voices (podcast host A, studio female),
three ways: current model + Nepali-letter English (today), bilingual + Nepali-letter English, bilingual + Latin English.
Usage (L4): python ops/english_test.py OUT"""
import json
import os
import sys

sys.path[:0] = ["pipeline", "tools", "."]
from clone import patch_f5_chunker, set_pace, to_model_text  # noqa: E402
from f5_tts.api import F5TTS  # noqa: E402
from frontend import normalize  # noqa: E402

T = {"en_thanks": "Thank you so much for your support. See you tomorrow.",
     "en_show": "Hello everyone, welcome to the show. Today we are going to talk about technology in Nepal.",
     "en_weather": "The weather in Kathmandu is sunny today, with a light breeze in the afternoon.",
     "en_story": "I was really nervous at first, but honestly, it turned out to be the best day of my life.",
     "mix_office": "आज office मा meeting धेरै लामो भयो, अब घर गएर rest गर्नुपर्छ।",
     "mix_order": "Customer service मा call गर्दा उनीहरूले भने, your order will be delivered by Friday."}
V = {"podcast": ("/home/podcast/pieces/pod_080.wav", "इट्स वे अफ लिभिङ हो नि त यो त हाम्रो रिलिजन स्पेसिफिक भन्दा पनि वे अफ लिभिङ त्यो एनिमेस्टिक नेचर कस्तो मनपर्दै गयो के मलाई", 1.037),
     "studio": ("/home/data/ref/bench/female_news.wav", "यी हुन् गान्तोक, मङ्गन, सिङताम, रम्फू, जोरथाङ, नयाँबजार, गेजिङ र नाम्ची।", 1.04)}
RUNS = [("today", "/home/ema/nepali_accent_8000.pt", "0"), ("bi_deva", "/home/ema/nepali_bi_ema.pt", "0"),
        ("bi_latin", "/home/ema/nepali_bi_ema.pt", "1")]
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
patch_f5_chunker()
cur, m, rows = None, None, []
for run, ck, latin in RUNS:
    if ck != cur:
        m = F5TTS(model="F5TTS_Base", ckpt_file=ck, vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
        m.ema_model.float()
        cur = ck
    os.environ["ENGLISH_LATIN"] = latin
    for vn, (pw, pt, spd) in V.items():
        rt = normalize(pt)
        for k, t in T.items():
            gen = to_model_text(t, "chars")
            set_pace(rt, pw)
            p = f"{out}/{run}_{vn}_{k}.wav"
            m.infer(ref_file=pw, ref_text=rt, gen_text=gen, file_wave=p, nfe_step=32, seed=0, cfg_strength=2.0, speed=spd,
                    remove_silence=True)
            os.system(f"ffmpeg -loglevel error -y -i {p} -b:a 128k {p[:-4]}.mp3")
            rows.append({"run": run, "voice": vn, "k": k, "text": t, "model_text": gen})
            print(run, vn, k, "|", gen[:70], flush=True)
json.dump(rows, open(f"{out}/rows.json", "w"), ensure_ascii=False, indent=1)
print("ENGLISH_TEST_DONE")
