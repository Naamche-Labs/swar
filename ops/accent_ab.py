import sys, json, os
sys.path[:0] = ["/home/nepali-tts/tools", "/home/nepali-tts/pipeline", "/home/nepali-tts"]
from f5_tts.api import F5TTS
from clone import speed_for
from frontend import normalize
from frontend.pron import nepal_accent
L = ["उसले दशचोटि फोन गर्‍यो तर कसैले उठाएन।", "विशेष गरी शिक्षा क्षेत्रमा धेरै विकास भएको छ।",
     "सर्वोच्च अदालतले कृषि मन्त्रालयलाई आदेश दियो।", "ज्ञान र क्षमताले मात्र मानिस ठूलो हुन्छ।",
     "शनिबार बिहान हामी शिवपुरी जाने विचार गर्दैछौँ।", "भाषा र संस्कृतिको संरक्षण हाम्रो कर्तव्य हो।"]
refs = json.load(open("/home/data/ref/bench/refs.json"))
tts = F5TTS(model="F5TTS_Base", ckpt_file="/home/ema/nepali_accent_8000.pt", vocab_file="/home/models/IndicF5/checkpoints/vocab.txt"); tts.ema_model.float()
os.makedirs("/home/out/accent_ab", exist_ok=True)
for voice, key in (("female", "female_news"), ("male", "male_wiki")):
    r = refs[key]
    for i, t in enumerate(L):
        g = normalize(t)
        for tag, x in (("std", g), ("nepal", nepal_accent(g))):
            tts.infer(ref_file=r["audio"], ref_text=r["text"], gen_text=x, file_wave=f"/home/out/accent_ab/{voice}_{i}_{tag}.wav",
                      remove_silence=True, seed=0, nfe_step=16, speed=speed_for(x, r.get("speed", 1.0)))
json.dump(L, open("/home/out/accent_ab/lines.json", "w"), ensure_ascii=False)
print("AB_DONE")
