import sys, json, os
sys.path[:0] = ["/home/nepali-tts/tools", "/home/nepali-tts/pipeline", "/home/nepali-tts"]
from f5_tts.api import F5TTS
from clone import speed_for, to_model_text, patch_f5_chunker, set_pace
from frontend import normalize
L = ["Meeting cancel भयो, सबैलाई email मा update पठाइदिनुहोस् है।",
     "मेरो phone को battery low छ, charger छ भने दिनुस् न।",
     "Actually, यो project को deadline next week सम्म extend भयो।",
     "आज office पछि हामी Thamel को एउटा cafe मा coffee खान जाने plan छ।",
     "Online class मा teacher ले assignment दिनुभयो, Sunday सम्म submit गर्नुपर्छ।",
     "Bro, तिमीले त्यो new movie हेर्यौ? Trailer त एकदमै awesome थियो!"]
refs = json.load(open("/home/data/ref/bench/refs.json"))
v = json.load(open("/home/out/arjan3/voice.json"))
voices = {"female": (refs["female_news"]["audio"], refs["female_news"]["text"], refs["female_news"].get("speed", 1.0)),
          "male": (refs["male_wiki"]["audio"], refs["male_wiki"]["text"], refs["male_wiki"].get("speed", 1.0)),
          "aarjan": (v["ref"], v["ref_text"], v["speed"])}
patch_f5_chunker()
tts = F5TTS(model="F5TTS_Base", ckpt_file="/home/ema/nepali_accent_8000.pt", vocab_file="/home/models/IndicF5/checkpoints/vocab.txt"); tts.ema_model.float()
os.makedirs("/home/out/cs_demo", exist_ok=True)
for name, (ra, rt, sp) in voices.items():
    set_pace(normalize(rt), ra)
    for i, t in enumerate(L):
        g = to_model_text(t, "chars")
        tts.infer(ref_file=ra, ref_text=normalize(rt), gen_text=g, file_wave=f"/home/out/cs_demo/{name}_{i}.wav", remove_silence=True, seed=0, nfe_step=16, speed=speed_for(g, sp))
json.dump(L, open("/home/out/cs_demo/lines.json", "w"), ensure_ascii=False)
print("CS_DEMO_DONE")
