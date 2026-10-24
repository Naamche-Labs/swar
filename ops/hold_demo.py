"""Ear check for the spoken-final-a hold: the user's sentences, hold off vs on, podcast voice (seed 0 and 1).
Usage (L4): python ops/hold_demo.py OUT"""
import json
import os
import sys

sys.path[:0] = ["pipeline", "tools", "."]
from clone import patch_f5_chunker, set_pace, to_model_text  # noqa: E402
from f5_tts.api import F5TTS  # noqa: E402
from frontend import normalize  # noqa: E402

S = ["हिजो बेलुका हामी साथीहरू मिलेर खाना खान गयौँ, एकदमै रमाइलो भयो।",
     "आजको खाना एकदमै मीठो थियो।",
     "आज बिहान म अलि ढिलो उठेँ, अनि सिधै office गएँ।",
     "मलाई यो कुरा गर्न मन लाग्दैन, तर गर्नुपर्छ।",
     "अब के गर्ने? तिमी किन आएनौ?",
     "घर गएर आराम गर्छु, भोलि भेटौँला।"]
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
PW = "/home/podcast/pieces/pod_080.wav"
PT = normalize("इट्स वे अफ लिभिङ हो नि त यो त हाम्रो रिलिजन स्पेसिफिक भन्दा पनि वे अफ लिभिङ त्यो एनिमेस्टिक नेचर कस्तो मनपर्दै गयो के मलाई")
patch_f5_chunker()
m = F5TTS(model="F5TTS_Base", ckpt_file=os.environ.get("F5_CKPT", "/home/ema/nepali_accent_8000.pt"), vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
m.ema_model.float()
rows = []
for i, t in enumerate(S):
    for h in ("0", "1"):
        os.environ["SCHWA_HOLD"] = h
        gen = to_model_text(t, "chars")
        for seed in (0, 1):
            set_pace(PT, PW)
            p = f"{out}/s{i}_hold{h}_seed{seed}.wav"
            m.infer(ref_file=PW, ref_text=PT, gen_text=gen, file_wave=p, nfe_step=32, seed=seed, cfg_strength=2.0, speed=1.037,
                    remove_silence=True)
            os.system(f"ffmpeg -loglevel error -y -i {p} -b:a 128k {p[:-4]}.mp3")
        rows.append({"i": i, "text": t, "hold": h, "model_text": gen})
json.dump(rows, open(f"{out}/rows.json", "w"), ensure_ascii=False, indent=1)
print("HOLD_DEMO_DONE")
