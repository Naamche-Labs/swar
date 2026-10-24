"""NepTTS-Bench audio from AI4Bharat Indic Parler-TTS (Apache-2.0; Nepali speaker "Amrita", its only one).
Text as given by the bench (its own tokenizer handles Devanagari and English). Runs in /home/venv_parler.
Usage: /home/venv_parler/bin/python tools/gen_indic_parler.py"""
import json
import os

import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

m = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").cuda().eval()
tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
dtok = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
desc = ("Amrita speaks at a moderate pace with a clear, natural and expressive voice. "
        "The recording is of very high quality, with the speaker's voice sounding clear and very close up.")
d = dtok(desc, return_tensors="pt").to("cuda")
out = "/home/data/bench/runs/indic_parler_amrita"
os.makedirs(out, exist_ok=True)
for x in json.load(open("/home/data/bench/sentences.json")):
    if x["category"] == "natural_speech":
        continue
    p = f"{out}/{x['sent_id']}.wav"
    if os.path.exists(p):
        continue
    t = tok(x["text_devanagari"], return_tensors="pt").to("cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        a = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=t.input_ids,
                       prompt_attention_mask=t.attention_mask)
    sf.write(p, a.cpu().numpy().squeeze(), m.config.sampling_rate)
print("PARLER_DONE")
