"""Clone a voice and speak Nepali text (zero-shot, no studio).

Give it a short reference clip (5-15 s, quiet room) + its transcript and
Nepali text; it speaks the text in that voice.

Best results: record the user reading one sentence we show them (--enroll prints
one), so the transcript is exact. Without --ref_text the clip is transcribed with
MMS, which costs about +3 pp CER on held-out speakers (F5 leans on an exact prompt
transcript). Runs on the GPU box after
training. Text input must match how the checkpoint was trained:
  --input chars     normalized Devanagari (train_*_chars.csv)
  --input respell   our G2P written back as Devanagari (train_*_respell.csv)

Use a calm, clean read clip as the reference: on NepTTS-Bench an annoyed
conversational prompt gave 35% CER and a calm read one 12% with the same model.

Backends: openf5 (primary, f5-tts API) and chatterbox (MIT challenger,
chatterbox-tts multilingual API, Hindi/Devanagari base + Nepali LoRA).

Usage:
    python clone.py --enroll                       # a sentence for the user to read aloud
    python clone.py --model openf5 --ckpt model.pt --vocab vocab.txt --input chars \
        --ref me.wav --ref_text "मेरो नाम ..." --text "नमस्ते" --out out.wav
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend import g2p, normalize
from frontend.g2p import respell

SHORT_CHARS, SHORT_SPEED = 15, 0.5  # NepTTS-Bench: mean-file CER 11.8% -> 10.0% (0.3: 10.2%)


def to_model_text(text, mode):
    from frontend.pron import apply_fixes
    if mode == "chars":  # Nepal pronunciation on by default (user A/B, 2026-10-22); NEPAL_ACCENT=0 turns it off
        # mined respellings first (keyed by standard spelling), then Nepal pronunciation, then English words
        fixed = apply_fixes(normalize(text, english=False))
        if os.environ.get("SCHWA_HOLD", "1") == "1":  # spoken final a: खान -> खान्अ, आज -> आज्अ (frontend.pron)
            from frontend.pron import hold_final_schwa
            fixed = hold_final_schwa(fixed)
        # ENGLISH_LATIN=1 (bilingual checkpoints, trained on LibriTTS-R English): English stays in Latin letters
        # so the model reads it as English instead of Nepali-letter English
        latin = os.environ.get("ENGLISH_LATIN", "0") == "1"
        return normalize(fixed, english=not latin, nepal=os.environ.get("NEPAL_ACCENT", "1") == "1")
    return respell(g2p(normalize(text)))


CHUNK_SECONDS = 10.0  # most training clips are 5-10 s; longer single generations slur words


def nepali_chunks(text, chars_per_sec, max_sec=CHUNK_SECONDS):
    """Split at sentence ends (। ? ! .), then at commas/semicolons if still too long, and
    pack whole pieces up to max_sec of speech at the prompt's pace. F5's own chunker
    splits only on Latin/Chinese punctuation, so a Nepali paragraph reached the model as
    one long generation."""
    import re
    limit = max(20, int(chars_per_sec * max_sec))
    pieces = []
    for sent in re.split(r"(?<=[।॥?!.])\s*", text.strip()):
        if len(sent) <= limit:
            pieces.append(sent)
        else:
            pieces += [p for p in re.split(r"(?<=[,;:])\s*", sent)]
    chunks, cur = [], ""
    for p in (p.strip() for p in pieces):
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


_PACE = {"cps": None}


def set_pace(ref_text, ref_wav):
    """Prompt speaking rate in characters per second, used to size chunks."""
    import soundfile as sf
    _PACE["cps"] = max(4.0, len(ref_text) / max(1.0, sf.info(ref_wav).duration))


def patch_f5_chunker():
    """Make F5's infer_process use nepali_chunks. Pace comes from set_pace(); without
    it, it is estimated from F5's byte budget (ref_bytes/ref_sec * (22 - ref_sec)),
    taking ~3 bytes per Devanagari letter and a ~14 s remaining window."""
    import f5_tts.infer.utils_infer as U
    if getattr(U.chunk_text, "nepali", False):
        return

    def chunk_text(text, max_chars=135):
        return nepali_chunks(text, chars_per_sec=_PACE["cps"] or max(4.0, max_chars / 3 / 14))
    chunk_text.nepali = True
    U.chunk_text = chunk_text


def speed_for(text, speed=1.0):
    """F5 slows text under 10 *bytes* to avoid rushed short phrases; a Devanagari
    letter is 3 bytes, so Nepali never triggers it. Same idea, counted in characters;
    `speed` (the speaker's pace from prepare_voice) applies to everything else."""
    return SHORT_SPEED * speed if len(text) < SHORT_CHARS else speed


def synth_openf5(a, ref_text, gen_text):
    from f5_tts.api import F5TTS
    patch_f5_chunker()
    set_pace(ref_text, a.ref)
    tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=a.ckpt, vocab_file=a.vocab)
    tts.ema_model.float()  # IndicF5-derived weights overflow to NaN in fp16 inference
    tts.infer(ref_file=a.ref, ref_text=ref_text, gen_text=gen_text, file_wave=a.out,
              remove_silence=True, speed=speed_for(gen_text, getattr(a, "speed", 1.0)))


def synth_chatterbox(a, gen_text):
    # API per resemble-ai/chatterbox README; load the Nepali LoRA per the
    # finetuning repo used in training (configs/chatterbox_nepali.json).
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    wav = model.generate(gen_text, language_id="hi", audio_prompt_path=a.ref)
    torchaudio.save(a.out, wav, model.sr)


def _speech_islands(w, sr, top_db=35, bridge=0.35):
    import librosa
    isl = []
    for a, b in librosa.effects.split(w, top_db=top_db, frame_length=2048, hop_length=256):
        if isl and (a - isl[-1][1]) / sr < bridge:
            isl[-1][1] = b
        else:
            isl.append([int(a), int(b)])
    return isl


def prepare_voice(upload, out_dir, transcripts=None, lm=None, min_sec=4.0, max_sec=12.0):
    """Turn a whole upload (30 s or more of someone talking) into an F5 prompt, no training.

    1. cut it at pauses into 3-12 s pieces;
    2. transcribe each piece (or take `transcripts`, one per piece, when known);
    3. prompt = the piece whose ECAPA voice embedding is closest to the average voice of the
       whole upload (the most typical sound of the person), preferring 6-12 s pieces;
    4. pace = letters per second over the whole upload. F5 copies the prompt's pace, and one
       piece can be slower than the person usually talks (a read intro: 7.1 letters/s vs 9.7-11.6
       when talking freely), so speed = upload pace / prompt pace (clipped to 0.8-1.35).
    Returns {"ref": wav, "ref_text": text, "speed": float, "pieces": [...]}."""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    os.makedirs(out_dir, exist_ok=True)
    sr = 24000
    w, _ = librosa.load(upload, sr=sr)
    pieces, cur = [], None
    for a, b in _speech_islands(w, sr):
        if cur and (b - cur[0]) / sr <= max_sec:
            cur[1] = b
        else:
            if cur:
                pieces.append(cur)
            cur = [a, b]
    if cur:
        pieces.append(cur)
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.expanduser("~/.cache/ecapa"))
    out = []
    for i, (a, b) in enumerate(pieces):
        a, b = max(0, a - int(0.15 * sr)), min(len(w), b + int(0.15 * sr))
        if (b - a) / sr < 2.5:
            continue
        p = os.path.join(out_dir, f"piece_{i:02d}.wav")
        sf.write(p, w[a:b], sr)
        x = librosa.resample(w[a:b], orig_sr=sr, target_sr=16000)
        with torch.no_grad():
            e = torch.nn.functional.normalize(spk.encode_batch(torch.tensor(x).unsqueeze(0).to(spk.device)).squeeze(), dim=-1)
        text = transcripts[len(out)] if transcripts else transcribe(p, lm)
        speech = sum(bb - aa for aa, bb in _speech_islands(w[a:b], sr)) / sr
        out.append({"wav": p, "dur": (b - a) / sr, "speech": speech, "text": text, "emb": e.cpu().numpy()})
    if not out:
        raise ValueError("no speech found in the upload")
    centroid = np.mean([o["emb"] for o in out], 0)
    centroid /= np.linalg.norm(centroid)
    for o in out:
        o["typical"] = float(o["emb"] @ centroid)
        o["score"] = o["typical"] - (0.0 if min_sec + 2 <= o["dur"] <= max_sec else 0.05 if o["dur"] >= min_sec else 0.2)
    best = max(out, key=lambda o: o["score"])
    letters = lambda t: len(normalize(t).replace(" ", ""))
    upload_rate = sum(letters(o["text"]) for o in out) / sum(o["dur"] for o in out)
    prompt_rate = letters(best["text"]) / best["dur"]
    speed = float(min(1.35, max(0.8, upload_rate / prompt_rate)))
    return {"ref": best["wav"], "ref_text": best["text"], "speed": speed, "upload_rate": upload_rate,
            "prompt_rate": prompt_rate, "pieces": [{k: v for k, v in o.items() if k != "emb"} for o in out]}


def enroll_sentence(seed=None):
    """A calm, everyday sentence for the user to read as their voice prompt."""
    import random
    from make_recording_script import POOL
    ok = [x for x in POOL if 25 <= len(x) <= 80]
    return random.Random(seed).choice(ok)


def transcribe(wav, lm=None):
    """Fallback prompt transcript: MMS-1b-all (npi), the best of 5 Nepali ASRs we tried
    on spontaneous prompts; with a KenLM word model (tools/build_lm.py) if given."""
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True).eval()
    w, _ = librosa.load(wav, sr=16000)
    with torch.no_grad():
        lg = asr(**proc(w, sampling_rate=16000, return_tensors="pt")).logits[0]
    if lm:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        from build_lm import decoder
        return decoder(lm).decode(torch.log_softmax(lg.float(), -1).numpy(), beam_width=64)
    return proc.decode(lg.argmax(-1))


def run(a):
    if a.upload:
        v = prepare_voice(a.upload, a.out + ".voice", lm=a.lm)
        a.ref, a.ref_text, a.speed = v["ref"], v["ref_text"], v["speed"]
        print(f"prompt: {a.ref} ({a.ref_text}); pace {v['upload_rate']:.1f} letters/s overall, "
              f"{v['prompt_rate']:.1f} in the prompt -> speed {a.speed:.2f}")
    if not a.ref_text:
        a.ref_text = transcribe(a.ref, a.lm)
        print(f"no --ref_text: transcribed the clip as \"{a.ref_text}\" (an exact transcript is better: see --enroll)")
    if a.model == "openf5":
        synth_openf5(a, to_model_text(a.ref_text, a.input), to_model_text(a.text, a.input))
    else:
        synth_chatterbox(a, normalize(a.text))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["openf5", "chatterbox"], default="openf5")
    ap.add_argument("--ckpt", help="F5 checkpoint (.pt/.safetensors)")
    ap.add_argument("--vocab", help="F5 vocab.txt matching the checkpoint")
    ap.add_argument("--input", choices=["chars", "respell"], default="chars")
    ap.add_argument("--enroll", action="store_true", help="print a sentence to record as the voice prompt")
    ap.add_argument("--upload", help="a longer recording (30 s+) of the person: picks the prompt and pace itself")
    ap.add_argument("--ref", help="reference voice clip")
    ap.add_argument("--ref_text", help="transcript of the reference clip (omit to transcribe it with MMS)")
    ap.add_argument("--lm", help="KenLM binary for transcribing the clip (tools/build_lm.py)")
    ap.add_argument("--text", help="Nepali text to speak")
    ap.add_argument("--out", default="out.wav")
    a = ap.parse_args()
    if a.enroll:
        print(enroll_sentence())
    elif not ((a.ref or a.upload) and a.text):
        ap.error("--ref or --upload, and --text, are required (or use --enroll)")
    else:
        run(a)
