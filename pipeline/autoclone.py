"""Drag-and-drop voice cloning, fully automatic: any number of clips (10 s to 5 min each, any format) in,
the person's voice saying new text out. No training, no hand-made transcripts.

Enrol (once per voice, ~1 s per 10 s of audio on the RTX box):
  1. decode every upload to 24 kHz mono, cut at pauses into 3.5-10 s pieces (long talk without pauses is
     split at its quietest 10 ms frame);
  2. keep one speaker: ECAPA embeddings, average-linkage clusters at 0.62 similarity, the cluster with the
     most speech wins (WhatsApp forwards and replies mix voices; speaker A's 11 notes held two people);
  3. transcribe each piece with up to four recognisers (Devanagari-only MMS, Whisper-large-v3, IndicConformer
     CTC and RNNT, Kriti Telephony RNNT and CTC; the last two run in their own venvs) and keep the medoid: the
     hypothesis closest to all the others after folding spellings speech does not distinguish. Confidence =
     1 - its mean distance to the others. F5 copies whatever the prompt transcript says, so a wrong
     transcript is the main way a clone goes wrong;
  4. prompts = the K best pieces by typical voice (ECAPA to the speaker centroid) + transcript confidence,
     5-10 s long, from different uploads where possible; pace = her median letters/s over confident pieces.
Say (per line):
  candidates = K prompts x CFG {2.0, 2.5} x seeds; every take is read back by Devanagari MMS and scored by
  pronunciation CER against the text; takes within 3 points of the best CER compete on ECAPA similarity to
  the speaker's pieces not used as that prompt. (Choosing by similarity alone let "खाना" come out as "खान".)
Usage (GPU box):
  python pipeline/autoclone.py enroll VOICE_DIR clip1.ogg clip2.m4a ...
  python pipeline/autoclone.py say VOICE_DIR "नमस्ते, तपाईंलाई कस्तो छ?" out.wav [--takes 12]
English-only lines can go to F5-TTS v1 (ENGLISH_MODEL=1): its weights are CC-BY-NC, demo use only."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "pipeline"), ROOT]
SR = 24000
CKPT = os.environ.get("F5_CKPT", "/home/models/nepali_accent_8000.pt")
VOCAB = os.environ.get("F5_VOCAB", "/home/models/IndicF5/checkpoints/vocab.txt")
CONFORMER_PY = os.environ.get("CONFORMER_PY", "/home/venv_conf/bin/python")
KRITI_PY = os.environ.get("KRITI_PY", "/home/venv_nemo/bin/python")
KRITI_ONNX = os.environ.get("KRITI_ONNX", "")  # folder from ops/export_kriti_onnx.py: Kriti in-process, no NeMo
# BigVGAN-v2 fine-tuned on F5's mels (56k steps): real speech round trip UTMOS 3.57 / timbre 5.92 vs Vocos 3.07 / 6.53,
# TTS output UTMOS 3.60 vs 3.54. VOCODER=vocos goes back to F5's stock vocoder.
# bilingual checkpoint (accent_8000 + LibriTTS-R English): pure-English lines are said from the speaker's own English
# speech (Latin prompt + Latin text, the setting it was trained in). Podcast host test: accent classifier "native-like"
# 31% (her real English 33%, Nepali-letter English 0%), Whisper WER 15% per take before picking.
BILINGUAL = os.environ.get("BILINGUAL", "1" if "_bi" in os.path.basename(CKPT) else "0") == "1"
# English model (2026-10-29): with F5_EN_CKPT set (swar_v6_ema.pt, trained further on 96 h of English), a sentence
# with no Devanagari in it goes to that model with English kept in Latin letters. Every sentence with Devanagari, pure
# Nepali or mixed, takes exactly the old path (same model, same settings, byte-identical output). Nine voices x five
# English sentences, 4 takes: wrong sounds 20.4% -> 8.6%, wrong words 1.9% -> 1.6%.
F5_EN_CKPT = os.environ.get("F5_EN_CKPT", "")
VOCODER = os.environ.get("VOCODER", "vocos")  # BigVGAN path for studio presets; on WhatsApp clones it scored lower (UTMOS 3.00 vs 3.12)
BIGVGAN_SRC = os.environ.get("BIGVGAN_SRC", "/home/BigVGAN")
_M = {}


class BigVGANDecoder:
    """Drop-in for F5's Vocos object: F5 calls vocoder.decode(mel) -> (batch, samples)."""
    def __init__(self, ckpt):
        import torch
        sys.path.insert(0, BIGVGAN_SRC)
        import bigvgan
        from env import AttrDict
        h = AttrDict(json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))))
        self.g = bigvgan.BigVGAN(h, use_cuda_kernel=False)
        self.g.load_state_dict(torch.load(ckpt, map_location="cpu")["generator"])
        self.g.remove_weight_norm()
        self.g = self.g.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def decode(self, mel):
        import torch
        with torch.no_grad():
            return self.g(mel.to(next(self.g.parameters()).device).float()).squeeze(1)


def _lazy(name):
    """Models load once per process and only when a step needs them."""
    if name in _M:
        return _M[name]
    import torch
    if name == "spk":
        from speechbrain.inference.speaker import EncoderClassifier
        _M[name] = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.expanduser(
            "~/.cache/ecapa"), run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    elif name == "mms":
        from rescore_deva import DevaMMS
        _M[name] = DevaMMS()
    elif name == "whisper":
        from transformers import pipeline
        _M[name] = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0,
                            torch_dtype=torch.float16)
    elif name == "f5":
        from f5_tts.api import F5TTS
        from clone import patch_f5_chunker
        patch_f5_chunker()
        _M[name] = F5TTS(model="F5TTS_Base", ckpt_file=CKPT, vocab_file=VOCAB)
        _M[name].ema_model.float()  # IndicF5 weights give NaN / noise in fp16
        if VOCODER != "vocos" and os.path.exists(VOCODER):
            _M[name].vocoder = BigVGANDecoder(VOCODER)
    elif name == "f5en":
        from f5_tts.api import F5TTS
        from clone import patch_f5_chunker
        patch_f5_chunker()
        _M[name] = F5TTS(model="F5TTS_Base", ckpt_file=F5_EN_CKPT, vocab_file=VOCAB)
        _M[name].ema_model.float()
    elif name == "eng":
        from f5_tts.api import F5TTS
        from huggingface_hub import snapshot_download
        d = os.path.join(snapshot_download("SWivid/F5-TTS", allow_patterns=["F5TTS_v1_Base/*"]), "F5TTS_v1_Base")
        _M[name] = F5TTS(model="F5TTS_v1_Base", ckpt_file=f"{d}/model_1250000.safetensors", vocab_file=f"{d}/vocab.txt")
    return _M[name]


class Kriti:
    """Kriti Telephony kept loaded in its own venv; read(paths) -> {wav: {"rnnt", "ctc"}}. Commercial-safe, unlike
    MMS-1b-all (CC-BY-NC), so it is the read-back check in `say`; MMS stays as the fallback and in research evals."""
    def __init__(self):
        self.p = subprocess.Popen([KRITI_PY, os.path.join(ROOT, "tools", "asr_kriti.py"), "--serve"], text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  env={**os.environ, "NUMBA_CUDA_USE_NVIDIA_BINDING": "1"})
        for line in self.p.stdout:
            if line.startswith("READY"):
                break
        else:
            raise RuntimeError("Kriti worker failed to start")

    def read(self, paths):
        self.p.stdin.write(json.dumps(paths) + "\n")
        self.p.stdin.flush()
        for line in self.p.stdout:
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        raise RuntimeError("Kriti worker died")


class KritiInProcess:
    """Same interface as Kriti, backed by the ONNX export (pipeline/kriti_onnx.py): no NeMo, same process, so it
    works on a ZeroGPU Space. It has only the CTC head, so "rnnt" carries the CTC text too."""
    def __init__(self, d):
        from kriti_onnx import KritiONNX
        self.k = KritiONNX(d)

    def read(self, paths):
        return {p: {"rnnt": t, "ctc": t} for p, t in zip(paths, self.k.read(paths))}


def kriti():
    if "kriti" not in _M:
        try:
            _M["kriti"] = (KritiInProcess(KRITI_ONNX) if KRITI_ONNX else Kriti() if os.path.exists(KRITI_PY) else None)
        except Exception as e:  # noqa: BLE001
            print("Kriti unavailable, falling back to MMS:", e, flush=True)
            _M["kriti"] = None
    return _M["kriti"]


def read_back(paths):
    """What a recogniser hears in each take (Kriti CTC: closest to the letters said, no language-model smoothing)."""
    k = kriti()
    if k:
        r = k.read(paths)
        return [r[p]["ctc"] for p in paths]
    return [_lazy("mms")(p) for p in paths]


def emb(wav_or_path):
    import librosa
    import torch
    x = librosa.load(wav_or_path, sr=16000)[0] if isinstance(wav_or_path, str) else wav_or_path
    spk = _lazy("spk")
    with torch.no_grad():
        e = spk.encode_batch(torch.tensor(x).unsqueeze(0).to(spk.device)).squeeze()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def whisper_lid_en(path):
    """P(English) from Whisper-large-v3 language ID, and its English transcript when that is likely."""
    import librosa
    import torch
    wh = _lazy("whisper")
    x = librosa.load(path, sr=16000)[0]
    f = wh.feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_features.to(wh.model.device, wh.model.dtype)
    tok = wh.tokenizer
    ids = [tok.convert_tokens_to_ids(t) for t in ("<|en|>", "<|ne|>", "<|hi|>")]
    with torch.no_grad():
        lg = wh.model(input_features=f, decoder_input_ids=torch.tensor([[wh.model.config.decoder_start_token_id]],
                                                                       device=wh.model.device)).logits[0, -1]
    p_en = float(torch.softmax(lg[ids].float(), -1)[0])
    text = whisper_en(path) if p_en > 0.6 else ""
    return p_en, text


def whisper_en(path):
    import librosa
    x = librosa.load(path, sr=16000)[0]
    return _lazy("whisper")({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "english", "task": "transcribe"})["text"].strip()


def wer_en(ref, hyp):
    import re
    from eval_heldout import ed
    w = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
    r = w(ref)
    return ed(r, w(hyp)) / max(1, len(r))


def letters(t):
    from frontend import normalize
    return len(normalize(t).replace(" ", ""))


def speech_sec(w, sr):
    from clone import _speech_islands
    return sum(b - a for a, b in _speech_islands(w, sr)) / sr


def pcer(ref, hyp):
    from eval_heldout import ed
    from pron_cer import fold
    r = fold(ref)
    return ed(r, fold(hyp)) / max(1, len(r))


def cut_pieces(path, out_dir, tag, max_sec=10.0, min_sec=3.5):
    """One upload -> 3.5-10 s pieces at pauses (prep_voiceA's cutter)."""
    import librosa
    import soundfile as sf
    from clone import _speech_islands
    w, _ = librosa.load(path, sr=SR)  # ffmpeg backend: ogg/opus, m4a, mp3, wav
    w = w / max(1e-4, np.abs(w).max()) * 0.9
    rms = librosa.feature.rms(y=w, frame_length=2048, hop_length=240)[0]

    def split(a, b):
        if (b - a) / SR <= max_sec:
            return [[a, b]]
        lo, hi = a // 240 + int(3 * SR / 240), b // 240 - int(3 * SR / 240)
        k = lo + int(np.argmin(rms[lo:hi])) if hi > lo else (a + b) // 2 // 240
        return split(a, k * 240) + split(k * 240, b)
    cur, segs = None, []
    for a, b in _speech_islands(w, SR, top_db=30, bridge=0.25):
        if cur and (b - cur[0]) / SR <= max_sec:
            cur[1] = b
        else:
            segs += split(*cur) if cur else []
            cur = [a, b]
    segs += split(*cur) if cur else []
    out = []
    for a, b in segs:
        a, b = max(0, a - int(0.12 * SR)), min(len(w), b + int(0.12 * SR))
        if (b - a) / SR < min_sec:
            continue
        p = os.path.join(out_dir, f"{tag}_{len(out):03d}.wav")
        sf.write(p, w[a:b], SR)
        out.append({"wav": p, "src": os.path.basename(path), "start": a / SR, "dur": (b - a) / SR,
                    "speech": speech_sec(w[a:b], SR)})
    return out


def main_speaker(pieces, thr=0.62):
    """Average-linkage clusters over ECAPA similarity; the cluster with the most speech is the voice."""
    from scipy.cluster.hierarchy import fcluster, linkage
    E = np.stack([p["emb"] for p in pieces])
    if len(pieces) < 3:
        return [True] * len(pieces)
    Z = linkage(E, method="average", metric="cosine")
    lab = fcluster(Z, t=1 - thr, criterion="distance")
    best = max(set(lab), key=lambda c: sum(p["speech"] for p, l in zip(pieces, lab) if l == c))
    return [l == best for l in lab]


def external_asr(paths, only=("kriti", "conf")):
    """Recognisers that need their own environment, run as subprocesses; each is skipped when its venv is missing
    or the model is not accessible (IndicConformer is gated on Hugging Face). {wav: {"kriti_rnnt": .., ...}}"""
    res = {p: {} for p in paths}
    for tag, py, script, env in (("kriti", KRITI_PY, "asr_kriti.py", {"NUMBA_CUDA_USE_NVIDIA_BINDING": "1"}),
                                 ("conf", CONFORMER_PY, "asr_conformer.py", {})):
        if tag not in only or not os.path.exists(py):
            continue
        with tempfile.TemporaryDirectory() as d:
            json.dump(paths, open(f"{d}/in.json", "w"))
            r = subprocess.run([py, os.path.join(ROOT, "tools", script), f"{d}/in.json", f"{d}/out.json"],
                               capture_output=True, text=True, env={**os.environ, **env})
            if r.returncode:
                print(f"{tag} recogniser skipped:", r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "", flush=True)
                continue
            for w, hy in json.load(open(f"{d}/out.json")).items():
                res[w].update({f"{tag}_{m}": t for m, t in hy.items()})
    return res


def consensus(hyps):
    """Kriti Telephony RNNT when present (on speaker A's WhatsApp notes it matched hand transcripts nearly word for
    word where MMS and Whisper were garbled), else the medoid of the rest after spelling folds. Confidence = 1 - mean
    distance to the other hypotheses (Whisper left out of that when Kriti is there: it writes Hindi-like Nepali)."""
    hyps = {k: v.strip() for k, v in hyps.items() if v and v.strip()}
    if not hyps:
        return "", 0.0, None
    if len(hyps) == 1:
        k, v = next(iter(hyps.items()))
        return v, 0.3, k
    if "kriti_rnnt" in hyps:
        others = [v for k, v in hyps.items() if k not in ("kriti_rnnt", "whisper")] or [hyps.get("whisper", "")]
        return hyps["kriti_rnnt"], float(1 - np.mean([min(1.0, pcer(hyps["kriti_rnnt"], o)) for o in others])), "kriti_rnnt"
    d = {k: np.mean([min(1.0, pcer(v, o)) for kk, o in hyps.items() if kk != k]) for k, v in hyps.items()}
    order = ["conf_rnnt", "conf_ctc", "mms", "whisper"]  # ties go to the stronger Nepali recogniser
    k = min(d, key=lambda k: (round(d[k], 3), order.index(k) if k in order else 9))
    return hyps[k], float(1 - d[k]), k


def enroll(voice_dir, uploads, k_prompts=3, exclude=None, one_speaker=None):
    """uploads -> VOICE_DIR/voice.json. `exclude`: piece wavs (or callables on a piece) kept out of prompts and
    of the similarity reference (used by the replicate test to hold real speech out)."""
    import librosa
    os.makedirs(f"{voice_dir}/pieces", exist_ok=True)
    pieces = []
    for i, u in enumerate(uploads):
        pieces += cut_pieces(u, f"{voice_dir}/pieces", f"u{i:02d}")
    if not pieces:
        raise ValueError("no speech found in the uploads")
    for p in pieces:
        p["emb"] = emb(p["wav"])
    # the uploader may know better: one person recorded months apart on different phones scores 0.51-0.60 across
    # notes (Speaker S), the same range as two different people (speaker A vs the other voice, 0.52-0.71)
    one = os.environ.get("ONE_SPEAKER") == "1" if one_speaker is None else one_speaker
    keep = [True] * len(pieces) if one else main_speaker(pieces)
    mains = [p["wav"] for p, k in zip(pieces, keep) if k]
    ext = {w: {f"kriti_{m}": t for m, t in hy.items()} for w, hy in kriti().read(mains).items()} if kriti() else {}
    if os.path.exists(CONFORMER_PY):
        for w, hy in external_asr(mains, only=("conf",)).items():
            ext.setdefault(w, {}).update(hy)
    mms = _lazy("mms") if not ext or os.environ.get("CROSSCHECK_MMS") == "1" else None
    wh = _lazy("whisper") if not ext else None  # Kriti made Whisper redundant (Hindi-like Nepali, loops)
    from mos import utmos
    for p, k in zip(pieces, keep):
        p["main"] = bool(k)
        if not k:
            continue
        x = librosa.load(p["wav"], sr=16000)[0]
        hy = {}
        if mms:
            hy["mms"] = mms(p["wav"])
        if wh:
            hy["whisper"] = wh({"raw": x, "sampling_rate": 16000}, generate_kwargs={
                "language": "nepali", "task": "transcribe", "no_repeat_ngram_size": 4})["text"]
        hy.update(ext.get(p["wav"], {}))
        p["hyps"] = hy
        p["text"], p["confidence"], p["by"] = consensus(hy)
        p["utmos"] = float(utmos(x))
        p["rate"] = letters(p["text"]) / max(0.5, p["speech"]) if p["text"] else 0.0
        if BILINGUAL and 4.0 <= p["dur"] <= 11.0:  # the person's own English, for pure-English lines
            p["p_en"], p["en_text"] = whisper_lid_en(p["wav"])
    return select(voice_dir, pieces, uploads, k_prompts, exclude)


def select(voice_dir, pieces, uploads, k_prompts=3, exclude=None):
    """Prompts and pace from analysed pieces (re-run cheaply with a different hold-out)."""
    main = [p for p in pieces if p["main"]]
    held = {p["wav"] for p in main if exclude and any((e(p) if callable(e) else e == p["wav"]) for e in exclude)}
    pool = [p for p in main if p["wav"] not in held]
    cent = np.mean([p["emb"] for p in pool], 0)
    cent /= np.linalg.norm(cent)
    for p in main:
        p["typical"] = float(np.asarray(p["emb"]) @ cent)
    confident = [p for p in pool if p["confidence"] >= 0.6] or pool
    rate = float(np.median([p["rate"] for p in confident if p["rate"] > 0]))
    cand = [p for p in pool if 4.5 <= p["dur"] <= 10.5 and p["text"]]
    for p in cand:
        p["prompt_score"] = p["typical"] + 0.6 * p["confidence"] + 0.05 * p["utmos"] - (0.1 if p["dur"] < 5.5 else 0)
    chosen, used = [], set()
    for p in sorted(cand, key=lambda p: -p["prompt_score"]):  # one per upload first, then the rest
        if len(chosen) < k_prompts and p["src"] not in used:
            chosen.append(p), used.add(p["src"])
    for p in sorted(cand, key=lambda p: -p["prompt_score"]):
        if len(chosen) < k_prompts and p not in chosen:
            chosen.append(p)
    prompts = [{"wav": p["wav"], "text": p["text"], "confidence": p["confidence"],
                "speed": float(min(1.3, max(0.75, rate / max(1.0, p["rate"]))))} for p in chosen]
    en = sorted([p for p in pool if p.get("p_en", 0) > 0.8 and p.get("en_text") and 4.5 <= p["dur"] <= 10.5],
                key=lambda p: -(p["p_en"] + p["typical"]))[:2]
    en_prompts = [{"wav": p["wav"], "text": p["en_text"], "speed": 1.0} for p in en]
    v = {"uploads": [os.path.basename(u) for u in uploads], "rate": rate, "prompts": prompts, "en_prompts": en_prompts,
         "held_out": sorted(held),
         "pieces": [{k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in p.items()} for p in pieces]}
    json.dump(v, open(f"{voice_dir}/voice.json", "w"), ensure_ascii=False, indent=1)
    print(f"ENROLLED {voice_dir}: {len(main)}/{len(pieces)} pieces of the main speaker, pace {rate:.1f} letters/s, "
          f"prompts " + ", ".join(f"{os.path.basename(p['wav'])} (conf {p['confidence']:.2f})" for p in prompts), flush=True)
    return v


def _is_english(text):
    import re
    return not re.search(r"[ऀ-ॿ]", text)


def _nat(path):
    """Predicted naturalness (UTMOS) of a take."""
    import librosa
    from mos import utmos
    return float(utmos(librosa.load(path, sr=16000)[0]))


def _score(c):
    """Voice match first, naturalness as a nudge: Speaker S's "ठिक छ? All the best..." won on similarity while
    sounding bad. 0.1 per UTMOS point ~ the spread of similarity between good takes."""
    return c["sim"] + 0.1 * (c.get("nat", 3.0) - 3.0)


def _err(c):
    """pronunciation error used to pick takes: CER (Nepali) or Whisper word error (English), plus a penalty that
    shuts out an English take whose sounds are clearly worse than the best one's (phone_check)"""
    return c["cer"] + c.get("pen", 0.0)


def say(voice_dir, text, *args, **kw):
    """`_say`, with all-English sentences sent to the English model when F5_EN_CKPT is set (see F5_EN_CKPT above)."""
    global BILINGUAL
    if not (F5_EN_CKPT and _is_english(text)):
        return _say(voice_dir, text, *args, **kw)
    saved = (_lazy("f5"), BILINGUAL, os.environ.get("ENGLISH_LATIN"))
    _M["f5"], BILINGUAL, os.environ["ENGLISH_LATIN"] = _lazy("f5en"), True, "1"
    try:
        return _say(voice_dir, text, *args, **kw)
    finally:
        _M["f5"], BILINGUAL = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop("ENGLISH_LATIN", None)
        else:
            os.environ["ENGLISH_LATIN"] = saved[2]


def _say(voice_dir, text, out_wav, takes=12, nfe=16, keep_all=False, pace=None, prompts=None):
    """Generate `takes` candidates and keep the best (pronunciation first, then voice match). `pace` (letters/s)
    overrides the speaker's median pace, e.g. to match one real clip."""
    import shutil
    import time
    t0 = time.time()
    from clone import cross_script_speed, set_pace, speed_for, to_model_text
    from frontend import normalize
    from frontend.romanize import romanize
    v = json.load(open(f"{voice_dir}/voice.json"))
    main = [p for p in v["pieces"] if p["main"] and p["wav"] not in set(v["held_out"])]
    english = _is_english(text) and os.environ.get("ENGLISH_MODEL") == "1"
    # base v5 (663 h English) says English well from a Nepali prompt too: every English test of 2026-10-26/27 used the
    # voice's Nepali prompt with English text, so a voice without English clips no longer falls back to the Nepali path
    own_en = _is_english(text) and BILINGUAL and (bool(v.get("en_prompts")) or os.environ.get("EN_FROM_NE_PROMPT", "1") == "1")
    models = [("eng", _lazy("eng"))] if english else [("ours_en", _lazy("f5"))] if own_en else [("ours", _lazy("f5"))]
    use = [pr for pr in (v.get("en_prompts") or v["prompts"] if own_en else v["prompts"]) if not prompts or os.path.basename(pr["wav"]) in prompts]
    # English in the speaker's own voice follows the text harder: long passages in 5 voices, CFG 3.0 vs 2.0 gave 7.7% vs
    # 8.1% wrong sounds at the same naturalness (UTMOS 3.88 vs 3.87); 4.0 had fewer still but sounded flatter (3.78)
    cfgs = (3.0, 2.5) if own_en else (2.0, 2.5)
    plan = [(pr, m, cfg, seed) for pr in use for m in models for cfg in cfgs for seed in range(4)]
    plan = sorted(plan, key=lambda x: (x[3], x[2]))[:max(1, takes)]  # spread takes over prompts first
    work = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out_wav)) or ".")
    cands = []
    for i, (pr, (mname, model), cfg, seed) in enumerate(plan):
        if mname == "ours_en":
            from frontend.english_fix import fix_english
            gen = fix_english(text)
        else:
            gen = text if mname == "eng" else to_model_text(text, "chars")
        rt = romanize(pr["text"]) if mname == "eng" else normalize(pr["text"]) if mname != "ours_en" or not _is_english(pr["text"]) else pr["text"]
        set_pace(rt, pr["wav"])
        base_sp = pr["speed"] * (pace / v["rate"] if pace else 1.0)
        # English text said from a Nepali prompt: length by syllables, not bytes (clone.cross_script_speed)
        sp = cross_script_speed(gen, rt, base_sp) if mname == "ours_en" and not _is_english(rt) else speed_for(gen, base_sp)
        p = f"{work}/take_{i:02d}.wav"
        model.infer(ref_file=pr["wav"], ref_text=rt, gen_text=gen, file_wave=p, nfe_step=nfe, seed=seed, cfg_strength=cfg,
                    speed=sp, remove_silence=True)
        import soundfile as sf
        if sf.info(p).duration < 0.3:   # silence trimming can leave nothing; ECAPA then crashes on an empty clip
            continue
        # a short upload can leave one piece only: compare with it rather than with nothing
        others = [q["emb"] for q in main if q["wav"] != pr["wav"]] or [q["emb"] for q in main]
        ref = np.mean(others, 0)
        ref /= np.linalg.norm(ref)
        cands.append({"file": p, "prompt": os.path.basename(pr["wav"]), "model": mname, "cfg": cfg, "seed": seed,
                      "sim": float(emb(p) @ ref), "heard": "", "cer": 0.0, "nat": _nat(p)})
    for c in cands:  # English takes: what Whisper understands, as word error
        if c["model"] == "ours_en":
            c["heard"] = whisper_en(c["file"])
            c["cer"] = wer_en(text, c["heard"])
    en = [c for c in cands if c["model"] == "ours_en"]
    if en and os.environ.get("PHONE_CHECK", "1") == "1":
        # Whisper forgives "hoon-estly" as "honestly": among the takes it hears right, keep the ones whose SOUNDS
        # are right too (pipeline/phone_check.py); the rest of the pick (voice match, naturalness) is unchanged
        from phone_check import per
        for c in en:
            c["per"] = per(text, c["file"])
        if all(c["per"] is not None for c in en):
            wfloor = min(c["cer"] for c in en)
            ok = [c for c in en if c["cer"] <= wfloor + 0.03]
            pfloor = min(c["per"] for c in ok)
            for c in en:   # a take with clearly more wrong sounds is pushed out of the competition (see _err)
                if c not in ok or c["per"] > pfloor + 0.03:
                    c["pen"] = 1.0
    ours = [c for c in cands if c["model"] == "ours"]
    for c, h in zip(ours, read_back([c["file"] for c in ours]) if ours else []):  # one batched read-back per line
        c["heard"], c["cer"] = h, pcer(normalize(text), h)
    if not cands:
        raise RuntimeError(f"every take came out silent for: {text[:60]}")
    floor = min(_err(c) for c in cands)
    best = max([c for c in cands if _err(c) <= floor + 0.03], key=_score)
    shutil.copy(best["file"], out_wav)
    if not keep_all:
        shutil.rmtree(work, ignore_errors=True)
    best = {**best, "file": out_wav, "n": len(cands), "mean_sim": float(np.mean([c["sim"] for c in cands])),
            **({"cands": cands} if keep_all else {})}
    best["seconds"] = time.time() - t0
    print(f"SAID {os.path.basename(out_wav)}: sim {best['sim']:.3f} (mean {best['mean_sim']:.3f}) pron-CER "
          f"{100 * best['cer']:.1f} in {best['seconds']:.1f} s | heard: {best['heard'][:70]}", flush=True)
    return best


def say_long(voice_dir, text, out_wav, takes=8, pace=None, prompt=None):
    """Paragraphs or essays: one sentence at a time, joined with 0.35 s pauses inside a paragraph and 0.7 s between
    paragraphs. ONE prompt for the whole piece: picking each sentence's best take independently switched between
    prompts from notes recorded months apart, and the voice and pace changed every line (Speaker S's advice piece).
    All sentences are made with every prompt, the prompt with the best mean score wins (or `prompt`, a piece file
    name), and each sentence keeps its best take within it. Short text goes straight to `say`."""
    import re
    import shutil
    import soundfile as sf
    paras = [[s.strip() for s in re.findall(r"[^।?!.]+[।?!.]*", p) if s.strip(" ।?!.")] for p in text.split("\n") if p.strip()]
    for para in paras:  # a 1-3 word fragment ("ठिक छ?") alone comes out rushed and off-voice: join it to the next
        i = 0
        while i < len(para) - 1:
            if letters(para[i]) < 12:
                para[i:i + 2] = [para[i] + " " + para[i + 1]]
            else:
                i += 1
    if sum(len(p) for p in paras) <= 1:
        return say(voice_dir, text, out_wav, takes=takes, pace=pace)
    base = os.path.splitext(out_wav)[0]
    sent = [(pi, si, s) for pi, para in enumerate(paras) for si, s in enumerate(para)]
    res = [say(voice_dir, s, f"{base}_s{pi:02d}{si:02d}.wav", takes=takes, pace=pace, keep_all=True,
               prompts=[prompt] if prompt else None) for pi, si, s in sent]

    def pick(r, pr):
        cs = [c for c in r["cands"] if c["prompt"] == pr] or r["cands"]   # every take of this prompt came out empty
        floor = min(_err(c) for c in cs)
        return max([c for c in cs if _err(c) <= floor + 0.03], key=_score)
    names = sorted({c["prompt"] for r in res for c in r["cands"]})
    score = {pr: float(np.mean([_score(pick(r, pr)) - 0.5 * pick(r, pr)["cer"] for r in res])) for pr in names}
    chosen = prompt if prompt in names else max(score, key=score.get)
    print("SAY_LONG prompt scores", {k: round(v, 3) for k, v in score.items()}, "->", chosen, flush=True)
    audio, rows = [], []
    for (pi, si, s), r in zip(sent, res):
        c = pick(r, chosen)
        w, sr = sf.read(c["file"])
        audio += [w, np.zeros(int(sr * (0.35 if si < len(paras[pi]) - 1 else 0.7)))]
        rows.append({**c, "text": s})
        shutil.rmtree(os.path.dirname(c["file"]), ignore_errors=True)
        os.remove(r["file"])
    sf.write(out_wav, np.concatenate(audio[:-1]), sr)
    return {"file": out_wav, "sim": float(np.mean([r["sim"] for r in rows])), "cer": float(np.mean([r["cer"] for r in rows])),
            "n": sum(r["n"] for r in res), "heard": " ".join(r["heard"] for r in rows), "sentences": len(rows),
            "prompt": chosen, "rows": [{k: r[k] for k in ("text", "sim", "cer", "nat", "heard")} for r in rows]}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "enroll":
        enroll(sys.argv[2], sys.argv[3:])
    elif cmd == "say":
        n = int(sys.argv[sys.argv.index("--takes") + 1]) if "--takes" in sys.argv else 12
        say_long(sys.argv[2], sys.argv[3], sys.argv[4], takes=n)
    else:
        sys.exit(__doc__)
