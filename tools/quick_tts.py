"""Quick zero-shot synthesis with an F5 checkpoint: quick_tts.py CKPT VOCAB REF_WAV REF_TEXT OUT_DIR"""
import sys, os, json
from f5_tts.api import F5TTS
ckpt, vocab, ref_wav, ref_text, out = sys.argv[1:6]
arch = sys.argv[6] if len(sys.argv) > 6 else "F5TTS_Base"
os.makedirs(out, exist_ok=True)
tts = F5TTS(model=arch, ckpt_file=ckpt, vocab_file=vocab)
tts.ema_model.float()  # IndicF5 overflows to NaN in the default fp16 inference
S = ["नमस्ते, म नेपाली बोल्ने कम्प्युटर हुँ।",
     "आज काठमाडौंको मौसम धेरै राम्रो छ, तर बेलुका पानी पर्न सक्छ।",
     "नेपालको संविधान अनुसार सबै नागरिक कानुनको दृष्टिमा समान छन्।",
     "मलाई चिया र मःम खान धेरै मन पर्छ।",
     "विद्यालयका विद्यार्थीहरूले राष्ट्रिय गान गाए।"]
meta = []
for i, s in enumerate(S):
    p = f"{out}/sent_{i:03d}.wav"
    tts.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=s, file_wave=p, remove_silence=True, seed=0)
    meta.append({"wav": p, "text": s})
json.dump(meta, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)
print("done", len(meta))
