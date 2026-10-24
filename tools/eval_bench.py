"""NepTTS-Bench (ampixa/neptts-bench) head-to-head: our model vs the published systems.

The 205 designed sentences (minimal pairs, questions, robustness, ...) are
synthesized in the flagship voice and scored with the same MMS-1b-all CER as
tools/eval_heldout.py. The baseline systems' published audio is re-scored with
the identical scorer, so the comparison does not depend on their normalization.
Their own numbers (baselines.json) are per-file mean CER; we report that too.

Usage (GPU box, after /home/data/bench/sentences.json is downloaded):
    python tools/eval_bench.py run NAME CKPT          # TEXT_MODE=respell for respell models
    python tools/eval_bench.py baselines              # re-score published systems once
    python tools/eval_bench.py table                  # print everything scored so far
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_heldout import DEV, IVF5_VOCAB, ed, norm, text_converter  # noqa: E402

BENCH = "/home/data/bench"
OUT = f"{BENCH}/runs"
SYSTEMS = ["tingting_subina", "tingting_asmita", "elevenlabs", "gemini", "piper", "edge_tts/sagar"]


def sentences():
    s = json.load(open(f"{BENCH}/sentences.json"))
    return [x for x in s if x["category"] != "natural_speech"]


def bench_ref():
    """REF=<key> picks a calm read clip from /home/data/ref/bench/refs.json (female_news,
    male_wiki, ...); default is the shortest 5-10 s flagship clip."""
    key = os.environ.get("REF")
    if key:
        return json.load(open("/home/data/ref/bench/refs.json"))[key]
    return flagship_ref()


def speed_for(text, speed):
    """F5 slows down text under 10 *bytes* (speed 0.3) to avoid rushed short phrases,
    but a Devanagari letter is 3 bytes, so Nepali never triggers it. SHORT_CHARS /
    SHORT_SPEED apply the same idea in characters."""
    n = int(os.environ.get("SHORT_CHARS", 0))
    return float(os.environ.get("SHORT_SPEED", 0.5)) if n and len(text) < n else speed


def flagship_ref():
    """Shortest flagship clip of 5-10 s (F5 prompts work best short and clean)."""
    import soundfile as sf
    ref = json.load(open("/home/data/ref/flagship/reference.json"))
    clips = [(sf.info(c["audio"]).duration, c) for c in ref["clips"]]
    ok = [x for x in clips if 5 <= x[0] <= 10] or clips
    return min(ok, key=lambda x: x[0])[1]


def asr():
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                           ignore_mismatched_sizes=True).to(DEV).eval()
    return proc, model


def score(name, wavs):
    """wavs: {sent_id: path}. Writes OUT/NAME.json."""
    import librosa
    import torch
    from mos import utmos
    proc, model = asr()
    text = {x["sent_id"]: x["text_devanagari"] for x in sentences()}
    E = N = 0
    rows = []
    for sid, p in sorted(wavs.items()):
        w, _ = librosa.load(p, sr=16000)
        inp = {k: v.to(DEV) for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            hyp = norm(proc.decode(model(**inp).logits.argmax(-1)[0]))
        ref = norm(text[sid])
        e = ed(ref, hyp)
        E, N = E + e, N + len(ref)
        rows.append({"id": sid, "cer": e / max(1, len(ref)), "mos": utmos(w), "hyp": hyp, "ref": ref})
    res = {"name": name, "n": len(rows), "cer": 100 * E / max(1, N), "mos": sum(r["mos"] for r in rows) / max(1, len(rows)),
           "mean_file_cer": 100 * sum(r["cer"] for r in rows) / max(1, len(rows)), "rows": rows}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{name.replace('/', '_')}.json", "w"), ensure_ascii=False, indent=1)
    print(f"{name}: CER {res['cer']:.2f}%  mean-file CER {res['mean_file_cer']:.2f}%  UTMOS {res['mos']:.2f}  (n={res['n']})")
    return res


class Reranker:
    """Best-of-N: synthesize N seeds, keep the take a *different* ASR (not the MMS
    scorer) transcribes closest to the text, so the pick is not tuned to the metric."""
    def __init__(self, model_id=os.environ.get("RERANK_ASR", "gagan3012/wav2vec2-xlsr-nepali")):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEV).eval()

    def cer(self, path, text):
        import librosa
        import torch
        w, _ = librosa.load(path, sr=16000)
        inp = {k: v.to(DEV) for k, v in self.proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            hyp = norm(self.proc.decode(self.model(**inp).logits.argmax(-1)[0]))
        ref = norm(text)
        return ed(ref, hyp) / max(1, len(ref))


def run(name, ckpt, vocab=IVF5_VOCAB):
    from f5_tts.api import F5TTS
    tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt, vocab_file=vocab)
    tts.ema_model.float()  # IndicF5-derived weights overflow in fp16 inference
    knobs = {"nfe_step": int(os.environ.get("NFE", 32)), "cfg_strength": float(os.environ.get("CFG", 2.0)),
             "sway_sampling_coef": float(os.environ.get("SWAY", -1.0)), "speed": float(os.environ.get("SPEED", 1.0))}
    conv, ref = text_converter(), bench_ref()
    # our text front end (numbers, English words -> Devanagari), as the product runs; FRONTEND=0 = raw text
    # (runs before 2026-10-22 were raw: English in Latin letters and digits reached the model)
    if os.environ.get("FRONTEND", "1") == "1":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from frontend import normalize
        nep = os.environ.get("NEPAL_ACCENT", "1") == "1"
        conv = (lambda c: lambda t: c(normalize(t, nepal=nep)))(conv)
        if os.environ.get("FIXES", "0") == "1":  # mined fixes + spoken final-a hold, as pipeline/clone.to_model_text
            from frontend.pron import apply_fixes, hold_final_schwa
            hold = os.environ.get("SCHWA_HOLD", "1") == "1"
            conv = (lambda c: lambda t: c((hold_final_schwa if hold else (lambda x: x))(apply_fixes(normalize(t, english=False)))))(conv)
    out = f"{OUT}/{name}"
    os.makedirs(out, exist_ok=True)
    n_best = int(os.environ.get("BEST_OF", 1))
    rr = Reranker() if n_best > 1 else None
    wavs = {}
    for x in sentences():
        p = f"{out}/{x['sent_id']}.wav"
        if not os.path.exists(p):
            gen = conv(x["text_devanagari"])
            takes = []
            for seed in range(n_best):
                q = p if n_best == 1 else f"{out}/{x['sent_id']}_s{seed}.wav"
                tts.infer(ref_file=ref["audio"], ref_text=conv(ref["text"]), gen_text=gen, file_wave=q,
                          remove_silence=True, seed=seed, **{**knobs, "speed": speed_for(gen, knobs["speed"])})
                takes.append(q)
            if rr:
                best = min(takes, key=lambda q: rr.cer(q, x["text_devanagari"]))
                os.replace(best, p)
        wavs[x["sent_id"]] = p
    del tts
    return score(name, wavs)


def baselines():
    from huggingface_hub import hf_hub_download
    ids = {x["sent_id"] for x in sentences()}
    for sysname in SYSTEMS:
        wavs = {}
        for sid in sorted(ids):
            try:
                wavs[sid] = hf_hub_download("ampixa/neptts-bench", f"audio/{sysname}/{sid}.mp3",
                                            repo_type="dataset", local_dir=BENCH)
            except Exception:
                pass  # not every system covers every sentence
        score(sysname, wavs)


def table():
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            r = json.load(open(f"{OUT}/{f}"))
            print(f"{r['name']:28s} CER {r['cer']:6.2f}%  mean-file {r['mean_file_cer']:6.2f}%  UTMOS {r.get('mos', 0):.2f}  n={r['n']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(*sys.argv[2:4])
    elif cmd == "baselines":
        baselines()
    elif cmd == "table":
        table()
