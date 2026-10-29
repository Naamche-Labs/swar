"""Zero-shot eval on held-out speakers: intelligibility (MMS CER) + voice similarity.

For each held-out speaker: one real clip is the voice prompt, two other clips of
the same speaker give the target sentences. The real target recordings are
scored the same way, as the ceiling (system "real").

Usage (GPU box):
    python tools/eval_heldout.py build                      # writes /home/data/eval/items.json once
    python tools/eval_heldout.py run NAME CKPT [VOCAB]      # synthesize + score one checkpoint
    python tools/eval_heldout.py real                       # score the real recordings
"""
import hashlib
import json
import os
import re
import sys
import unicodedata

EVAL = "/home/data/eval"
MANIFEST = "/home/data/nepali_tts/manifest.jsonl"
HOLDOUT = "/home/data/ref/holdout_speakers.json"
IVF5_VOCAB = "/home/models/IndicF5/checkpoints/vocab.txt"
from mos import _auto_device  # noqa: E402

DEV = _auto_device()


def build():
    held = set(json.load(open(HOLDOUT)))
    by = {}
    for line in open(MANIFEST, encoding="utf-8"):
        m = json.loads(line)
        if m["speaker"] in held:
            by.setdefault(m["speaker"], []).append(m)
    items = []
    for spk in sorted(by):
        clips = sorted(by[spk], key=lambda m: hashlib.md5(m["audio"].encode()).hexdigest())
        refs = [c for c in clips if 4 <= c["dur"] <= 10 and c["snr"] >= 25] or [c for c in clips if 3 <= c["dur"] <= 12]
        if not refs:
            continue
        ref = refs[0]
        targets = [c for c in clips if c is not ref and 3 <= c["dur"] <= 15][:2]
        for i, t in enumerate(targets):
            items.append({"id": f"{len(items):03d}", "speaker": spk, "ref_wav": ref["audio"], "ref_text": ref["text"],
                          "text": t["text"], "real_wav": t["audio"]})
    os.makedirs(EVAL, exist_ok=True)
    json.dump(items, open(f"{EVAL}/items.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(items)} items from {len({i['speaker'] for i in items})} held-out speakers")


def norm(s):
    if os.environ.get("DIGITS_AS_WORDS", "1") == "1" and re.search(r"[0-9०-९]", s):
        # ASR writes numbers as digits ("24 घण्टामा") while targets are spelled out;
        # expand them with our own normalizer so correct speech is not counted as wrong
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from frontend import normalize
        s = normalize(s)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[।॥.,?!;:\"'()\-–—]", " ", s)
    return " ".join(s.split())


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def score(name, wavs):
    """wavs: {item id: wav path}. Writes EVAL/NAME/scores.json and prints a summary."""
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from speechbrain.inference.speaker import EncoderClassifier

    items = json.load(open(f"{EVAL}/items.json"))
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                         ignore_mismatched_sizes=True).to(DEV).eval()
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV},
                                         savedir="/home/hf_cache/ecapa")

    def emb(path):
        w, _ = librosa.load(path, sr=16000)
        with torch.no_grad():
            return spk.encode_batch(torch.tensor(w).unsqueeze(0).to(DEV)).squeeze()

    from mos import utmos
    E = N = 0
    sims, moss, rows = [], [], []
    for it in items:
        p = wavs.get(it["id"])
        if not p or not os.path.exists(p):
            continue
        w, _ = librosa.load(p, sr=16000)
        inp = {k: v.to(DEV) for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            hyp = norm(proc.decode(asr(**inp).logits.argmax(-1)[0]))
        ref = norm(it["text"])
        e = ed(ref, hyp)
        E, N = E + e, N + len(ref)
        # similarity to the speaker's *other* real clip (not the prompt itself)
        sim = float(torch.nn.functional.cosine_similarity(emb(p), emb(it["real_wav"] if name != "real" else it["ref_wav"]), dim=0))
        sims.append(sim)
        moss.append(utmos(w))
        rows.append({"id": it["id"], "cer": e / max(1, len(ref)), "sim": sim, "mos": moss[-1], "hyp": hyp, "ref": ref})
    res = {"name": name, "n": len(rows), "cer": 100 * E / max(1, N), "sim": sum(sims) / max(1, len(sims)),
           "mos": sum(moss) / max(1, len(moss)), "rows": rows}
    os.makedirs(f"{EVAL}/{name}", exist_ok=True)
    json.dump(res, open(f"{EVAL}/{name}/scores.json", "w"), ensure_ascii=False, indent=1)
    print(f"{name}: CER {res['cer']:.2f}%  voice-sim {res['sim']:.3f}  UTMOS {res['mos']:.2f}  (n={res['n']})")
    return res


def text_converter():
    """TEXT_MODE=respell feeds the model our G2P written back as Devanagari
    (frontend.g2p.respell), matching a model trained on train_respell.csv."""
    mode = os.environ.get("TEXT_MODE", "chars")
    if mode == "nepal":  # Nepal pronunciation respelling (frontend.pron.nepal_accent) + pron fixes
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from frontend.pron import apply_fixes, nepal_accent
        return lambda s: nepal_accent(apply_fixes(s))
    if mode == "fix":  # data/pron_fixes.json respellings (tools/mine_hard_words.py)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from frontend.pron import apply_fixes
        return apply_fixes
    if mode != "respell":
        return lambda s: s
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend import g2p, normalize
    from frontend.g2p import respell
    return lambda s: respell(g2p(normalize(s)))


def auto_ref_text():
    """REF_TEXT=asr: transcribe the prompt clip with MMS (npi) instead of using the gold
    transcript, as a real user would only have the audio. F5's own fallback is Whisper,
    which is weak on Nepali."""
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                         ignore_mismatched_sizes=True).to(DEV).eval()
    cache = {}
    lm = None
    if os.environ.get("REF_TEXT") == "asr_lm":  # Devanagari-only + KenLM (tools/build_lm.py)
        from build_lm import decoder
        from rescore_deva import DevaMMS
        lm, mask = decoder(os.environ.get("LM", "/home/lm/ne4.bin"), 0.3, 2.0), DevaMMS.latin_mask(proc)

    def f(it):
        if it["ref_wav"] not in cache:
            w, _ = librosa.load(it["ref_wav"], sr=16000)
            inp = {k: v.to(DEV) for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
            with torch.no_grad():
                lg = asr(**inp).logits[0]
            cache[it["ref_wav"]] = proc.decode(lg.argmax(-1)) if lm is None else \
                lm.decode(torch.log_softmax(lg.float() + mask, -1).cpu().numpy(), beam_width=64)
        return cache[it["ref_wav"]]
    return f


def run(name, ckpt, vocab=IVF5_VOCAB, arch=os.environ.get("F5_MODEL", "F5TTS_Base")):
    from f5_tts.api import F5TTS
    items = json.load(open(f"{EVAL}/items.json"))
    out = f"{EVAL}/{name}"
    os.makedirs(out, exist_ok=True)
    tts = F5TTS(model=arch, ckpt_file=ckpt, vocab_file=vocab)
    tts.ema_model.float()  # IndicF5-derived weights overflow in fp16 inference
    # inference knobs for sweeps (F5 defaults: nfe 32, cfg 2.0, sway -1.0, speed 1.0)
    knobs = {"nfe_step": int(os.environ.get("NFE", 32)), "cfg_strength": float(os.environ.get("CFG", 2.0)),
             "sway_sampling_coef": float(os.environ.get("SWAY", -1.0)), "speed": float(os.environ.get("SPEED", 1.0))}
    conv = text_converter()
    ref_text = auto_ref_text() if os.environ.get("REF_TEXT", "").startswith("asr") else (lambda it: it["ref_text"])
    from enhance import enhance  # REF_ENHANCE=dfn cleans the prompt clip first
    wavs = {}
    for it in items:
        p = f"{out}/{it['id']}.wav"
        if not os.path.exists(p):
            tts.infer(ref_file=enhance(it["ref_wav"], os.environ.get("REF_ENHANCE")), ref_text=conv(ref_text(it)), gen_text=conv(it["text"]), file_wave=p,
                      remove_silence=True, seed=0, **knobs)
        wavs[it["id"]] = p
    del tts
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score(name, wavs)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "rescore":  # re-score an existing run dir (e.g. after adding a metric)
        items = json.load(open(f"{EVAL}/items.json"))
        n = sys.argv[2]
        src = (lambda it: it["real_wav"]) if n == "real" else (lambda it: f"{EVAL}/{n}/{it['id']}.wav")
        score(n, {it["id"]: src(it) for it in items})
    elif cmd == "real":
        items = json.load(open(f"{EVAL}/items.json"))
        score("real", {it["id"]: it["real_wav"] for it in items})
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or []))
