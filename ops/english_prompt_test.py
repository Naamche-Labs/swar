"""English in the speaker's own English: find the podcast host's English stretches (Whisper language ID, English
transcript in Latin letters), use them as prompts for pure-English lines on the bilingual model (Latin in, Latin
prompt, the setting it was trained on), and score: Whisper English WER, accent classifier, UTMOS, voice match.
Usage (L4): python ops/english_prompt_test.py OUT"""
import glob
import json
import os
import re
import sys

import librosa
import numpy as np
import torch

sys.path[:0] = ["pipeline", "tools", "."]
from clone import patch_f5_chunker, set_pace  # noqa: E402
from f5_tts.api import F5TTS  # noqa: E402
from mos import utmos  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).cuda().eval()
lang_ids = {t: proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|en|>", "<|ne|>", "<|hi|>")}


def p_english(x):
    f = proc(x, sampling_rate=16000, return_tensors="pt").input_features.half().cuda()
    with torch.no_grad():
        lg = wm(input_features=f, decoder_input_ids=torch.tensor([[wm.config.decoder_start_token_id]]).cuda()).logits[0, -1]
    pr = torch.softmax(lg[list(lang_ids.values())].float(), -1)
    return float(pr[0])


def transcribe_en(x):
    f = proc(x, sampling_rate=16000, return_tensors="pt").input_features.half().cuda()
    with torch.no_grad():
        ids = wm.generate(f, language="english", task="transcribe")
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


cands = []
for p in sorted(glob.glob("/home/podcast/pieces/pod_*.wav")):
    x = librosa.load(p, sr=16000)[0]
    if not 4 <= len(x) / 16000 <= 11:
        continue
    pe = p_english(x)
    if pe > 0.6:
        cands.append({"wav": p, "p_en": pe, "text": transcribe_en(x), "dur": len(x) / 16000})
cands.sort(key=lambda c: -c["p_en"])
print("english pieces:", len(cands), flush=True)
for c in cands[:6]:
    print(f"  {os.path.basename(c['wav'])} p_en {c['p_en']:.2f} {c['dur']:.1f}s | {c['text'][:80]}", flush=True)
T = {"en_thanks": "Thank you so much for your support. See you tomorrow.",
     "en_show": "Hello everyone, welcome to the show. Today we are going to talk about technology in Nepal.",
     "en_weather": "The weather in Kathmandu is sunny today, with a light breeze in the afternoon.",
     "en_story": "I was really nervous at first, but honestly, it turned out to be the best day of my life."}
patch_f5_chunker()
m = F5TTS(model="F5TTS_Base", ckpt_file="/home/ema/nepali_bi_ema.pt", vocab_file="/home/models/IndicF5/checkpoints/vocab.txt")
m.ema_model.float()
rows = []
for ci, c in enumerate(cands[:3]):
    for k, t in T.items():
        for seed in (0, 1):
            set_pace(c["text"], c["wav"])
            p = f"{out}/enprompt{ci}_{k}_s{seed}.wav"
            m.infer(ref_file=c["wav"], ref_text=c["text"], gen_text=t, file_wave=p, nfe_step=32, seed=seed, cfg_strength=2.0,
                    remove_silence=True)
            x = librosa.load(p, sr=16000)[0]
            h = transcribe_en(x)
            w = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
            from eval_heldout import ed
            rows.append({"prompt": ci, "k": k, "seed": seed, "file": p, "heard": h, "wer": ed(w(t), w(h)) / len(w(t)),
                         "utmos": float(utmos(x))})
    rs = [r for r in rows if r["prompt"] == ci]
    print(f"PROMPT {ci}: WER {100 * np.mean([r['wer'] for r in rs]):.1f}  UTMOS {np.mean([r['utmos'] for r in rs]):.2f} | "
          f"{rs[0]['heard'][:60]}", flush=True)
json.dump({"cands": cands[:6], "rows": rows}, open(f"{out}/enprompt.json", "w"), ensure_ascii=False, indent=1)
print("ENPROMPT_DONE")
