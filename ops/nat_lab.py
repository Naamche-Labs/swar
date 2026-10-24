"""Naturalness push (2026-10-24): Indic-Speak Srijana beats us on NepTTS-Bench UTMOS (3.97 vs 3.82) and OmniVoice on
FLEURS UTMOS (2.49 vs 2.39). Settings are chosen on dev sets that share nothing with the tests (Rasa test sentences for
the bench, FLEURS validation speakers for FLEURS); each test is then run once with the chosen setting, one take.
Knobs: prompt voice (bench), prompt cleaning (FLEURS, DeepFilterNet3), NFE, CFG, output post-filter.
Usage (box, from /home/nepali-tts):
  python ops/nat_lab.py fdev-build                         # FLEURS validation split -> /home/data/fleurs_dev/items.json
  python ops/nat_lab.py dev   NAME VOICE NFE CFG POST      # 60 Rasa sentences, UTMOS + CER
  python ops/nat_lab.py bench NAME VOICE NFE CFG POST      # NepTTS-Bench -> /home/data/bench/runs/NAME
  python ops/nat_lab.py fl    NAME SET ENH NFE CFG POST    # SET dev|test, ENH none|dfn|dfn6|dfn12|dfn18
  python ops/nat_lab.py boot  VOICE                        # read-style prompt in a preset voice -> boot_VOICE/best.json
  python ops/nat_lab.py pick  VOICE                        # choose among boot's takes on other train sentences
      (BOOT_TAG=_slow: separate boot dir; PICK_SET=short: choose on data/short_sel.json phrases)
DEVSET=short: data/short_dev.json. NSEED=n: n seeds per item. TAKES=N: best of N seeds by Kriti read-back (the product's rule). VOICE: a refs.json key (female_news ...) or a data/voices.json name (maya ...). POST: none|dfn|norm|dfnnorm."""
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path[:0] = ["tools", "pipeline", "."]
from clone import patch_f5_chunker, set_pace, speed_for, to_model_text  # noqa: E402
from frontend import normalize  # noqa: E402

CK = os.environ.get("F5_CKPT", "/home/ema/podvoices_ema.pt")
VOCAB = "/home/models/IndicF5/checkpoints/vocab.txt"
_tts = {}


def model():
    if not _tts:
        from f5_tts.api import F5TTS
        patch_f5_chunker()
        _tts["m"] = F5TTS(model="F5TTS_Base", ckpt_file=CK, vocab_file=VOCAB)
        _tts["m"].ema_model.float()
    return _tts["m"]


def voice(key):
    if "::" in key:  # WAV::TRANSCRIPT, e.g. another clip of a preset voice
        w, t = key.split("::", 1)
        return w, normalize(t), float(os.environ.get("SPD", 1.0))
    refs = json.load(open("/home/data/ref/bench/refs.json"))
    if key in refs:
        r = refs[key]
        return r["audio"], normalize(r["text"]), r.get("speed", 1.0) * float(os.environ.get("SPD", 1.0))
    v = {x["name"]: x for x in json.load(open("data/voices.json"))["voices"]}[key]
    return v["prompt"], normalize(v["prompt_text"]), float(os.environ.get("SPD", 1.0))


def post(p, how):
    """Output post-filter, in place. dfn: DeepFilterNet3 denoise; norm: speech RMS to -20 dBFS (peak-limited)."""
    if "dfn" in how:
        from enhance import _dfn
        _dfn(p, p + ".tmp.wav")
        os.replace(p + ".tmp.wav", p)
    if "norm" in how:
        w, sr = sf.read(p, dtype="float32")
        rms = np.sqrt(np.mean(w[np.abs(w) > 0.01] ** 2) + 1e-9)
        w = w * min(10 ** (-20 / 20) / rms, 0.97 / (np.abs(w).max() + 1e-9))
        sf.write(p, w, sr)


def gen(text, ref_wav, ref_text, spd, p, nfe, cfg, how, pace=False, seed=0):
    if os.path.exists(p):
        return p
    g = to_model_text(text, "chars")
    if pace:
        set_pace(ref_text, ref_wav)
    takes = int(os.environ.get("TAKES", 1))
    if takes == 1:
        model().infer(ref_file=ref_wav, ref_text=ref_text, gen_text=g, file_wave=p, remove_silence=True, seed=seed,
                      nfe_step=nfe, cfg_strength=cfg, speed=speed_for(g, spd))
    else:  # as the product does (autoclone.say): keep the take our own ASR (Kriti, not the MMS scorer) reads back best
        import autoclone as A
        fs = [f"{p}.s{k}.wav" for k in range(takes)]
        for k, f in enumerate(fs):
            model().infer(ref_file=ref_wav, ref_text=ref_text, gen_text=g, file_wave=f, remove_silence=True, seed=k,
                          nfe_step=nfe, cfg_strength=cfg, speed=speed_for(g, spd))
        cer = [A.pcer(normalize(text), h) for h in A.read_back(fs)]
        os.replace(fs[cer.index(min(cer))], p)
        for f in fs:
            if os.path.exists(f):
                os.remove(f)
    if how != "none":
        post(p, how)
    return p


def score_dev(name, pairs):
    from eval_heldout import ed, norm
    from mos import utmos
    from rescore_deva import DevaMMS
    asr = DevaMMS()
    rows = []
    for text, p in pairs:
        ref, hyp = norm(normalize(text, english=False)), norm(asr(p))
        rows.append({"cer": 100 * ed(ref, hyp) / max(1, len(ref)), "mos": utmos(librosa.load(p, sr=16000)[0])})
    m = {k: float(np.mean([r[k] for r in rows])) for k in ("cer", "mos")} | {"n": len(rows), "name": name}
    os.makedirs("/home/data/natlab", exist_ok=True)
    json.dump(m, open(f"/home/data/natlab/{name}.json", "w"))
    print(f"NAT {name:34s} CER {m['cer']:5.2f}  UTMOS {m['mos']:.3f}  n={m['n']}", flush=True)


def fdev_build():
    """tools/eval_fleurs.build on the validation split (speakers disjoint from test in FLEURS)."""
    import eval_fleurs as E
    import huggingface_hub
    orig = huggingface_hub.hf_hub_download
    huggingface_hub.hf_hub_download = lambda repo, f, **k: orig(repo, f.replace("test-00000", "validation-00000"), **k)
    E.F = "/home/data/fleurs_dev"
    E.build()


def fl(name, split, enh, nfe, cfg, how):
    import eval_fleurs as E
    from enhance import enhance
    E.F = "/home/data/fleurs" if split == "test" else "/home/data/fleurs_dev"
    out = f"{E.F}/runs/{name}"
    os.makedirs(out, exist_ok=True)
    wavs = {}
    for it in json.load(open(f"{E.F}/items.json")):
        ref = enhance(it["ref_wav"], None if enh == "none" else enh)
        p = f"{out}/{it['id']}.wav"
        if not os.path.exists(p):
            set_pace(it["ref_text"], ref)
            model().infer(ref_file=ref, ref_text=it["ref_text"], gen_text=it["text"], file_wave=p, remove_silence=True,
                          seed=0, nfe_step=nfe, cfg_strength=cfg, speed=speed_for(it["text"]))
            if how != "none":
                post(p, how)
        wavs[it["id"]] = p
    _tts.clear()
    E.score(name, wavs)


def boot(key, n_sent=6, seeds=4):
    """Read-style prompt in a preset voice: the model reads FLEURS train-split sentences (not dev, not test) in the
    preset's voice; the take with the lowest CER (ties: higher UTMOS) becomes the new prompt. The podcast prompts are fast
    casual talk, and F5 copies that pace into read text (dev CER 6.4-7.2 vs 4.6 for the studio prompt)."""
    import random
    from eval_heldout import ed, norm
    from mos import utmos
    from rescore_deva import DevaMMS
    rw, rt, spd = voice(key)
    import glob
    import pyarrow.parquet as pq
    fl = glob.glob("/home/hf_cache/hub/datasets--google--fleurs/snapshots/*/parquet-data/ne_np/train-*.parquet")[0]
    texts = sorted({r["raw_transcription"].strip() for r in pq.read_table(fl, columns=["raw_transcription"]).to_pylist()})
    texts = [normalize(t) for t in texts if 90 <= len(t) <= 130 and not any(c.isascii() and c.isalpha() for c in t)]
    random.Random(0).shuffle(texts)
    d = f"/home/data/natlab/boot_{key}{os.environ.get('BOOT_TAG', '')}"
    os.makedirs(d, exist_ok=True)
    json.dump(texts[:n_sent], open(f"{d}/texts.json", "w"), ensure_ascii=False)
    asr, best = DevaMMS(), None
    for i, t in enumerate(texts[:n_sent]):
        for s in range(seeds):
            p = f"{d}/{i}_{s}.wav"
            g = to_model_text(t, "chars")
            model().infer(ref_file=rw, ref_text=rt, gen_text=g, file_wave=p, remove_silence=True, seed=s, nfe_step=32,
                          cfg_strength=2.0, speed=speed_for(g, spd))
            ref, hyp = norm(normalize(t, english=False)), norm(asr(p))
            c, m = 100 * ed(ref, hyp) / max(1, len(ref)), utmos(librosa.load(p, sr=16000)[0])
            print(f"BOOT {key} {i}_{s} CER {c:5.2f} UTMOS {m:.2f}", flush=True)
            if best is None or (c, -m) < (best[0], -best[1]):
                best = (c, m, p, t)
    json.dump({"wav": best[2], "text": best[3], "cer": best[0], "utmos": best[1]}, open(f"{d}/best.json", "w"), ensure_ascii=False)
    print("BOOT_BEST", key, best, flush=True)


def pick(key, n_eval=12):
    """Choose among boot()'s takes: each error-free take is tried as the prompt on n_eval other FLEURS train sentences
    (one take each); keep the lowest mean CER among prompts whose clones average UTMOS >= 4.0."""
    import glob
    import pyarrow.parquet as pq
    from eval_heldout import ed, norm
    from mos import utmos
    from rescore_deva import DevaMMS
    asr = DevaMMS()
    d = f"/home/data/natlab/boot_{key}{os.environ.get('BOOT_TAG', '')}"
    log = [l.split() for l in open(os.environ.get("BOOT_LOG", "/home/nat4.log")) if l.startswith(f"BOOT {key} ")]
    cands = [f"{d}/{x[2]}.wav" for x in log if float(x[4]) == 0.0]
    fl = glob.glob("/home/hf_cache/hub/datasets--google--fleurs/snapshots/*/parquet-data/ne_np/train-*.parquet")[0]
    texts = sorted({r["raw_transcription"].strip() for r in pq.read_table(fl, columns=["raw_transcription"]).to_pylist()})
    texts = [normalize(t) for t in texts if 60 <= len(t) <= 140 and not any(c.isascii() and c.isalpha() for c in t)]
    import random
    random.Random(1).shuffle(texts)
    used = set(json.load(open(f"{d}/texts.json"))) if os.path.exists(f"{d}/texts.json") else set()
    ev = [t for t in texts if t not in used][:n_eval]
    if os.environ.get("PICK_SET") == "short":  # short phrases (data/short_sel.json, not the short dev set): where fast
        ev = json.load(open("data/short_sel.json"))  # prompts fail; the bench is 94% phrases under 30 letters
    src = {}
    for x in log:
        src[f"{d}/{x[2]}.wav"] = x[2]
    res = []
    for c in cands:
        i = int(os.path.basename(c).split("_")[0])
        t = json.load(open(f"{d}/texts.json"))[i]
        rows = []
        for j, e in enumerate(ev):
            p = f"{d}/eval{os.environ.get('PICK_SET', '')}_{os.path.basename(c)[:-4]}_{j}.wav"
            gen(e, c, normalize(t), 1.0, p, 16, 2.0, "none")
            ref, hyp = norm(normalize(e, english=False)), norm(asr(p))
            rows.append((100 * ed(ref, hyp) / max(1, len(ref)), utmos(librosa.load(p, sr=16000)[0])))
        cer, mos = float(np.mean([r[0] for r in rows])), float(np.mean([r[1] for r in rows]))
        res.append({"wav": c, "text": t, "cer": cer, "utmos": mos})
        print(f"PICK {key} {os.path.basename(c)} CER {cer:5.2f} UTMOS {mos:.3f}", flush=True)
    ok = [r for r in res if r["utmos"] >= 4.0] or res
    best = min(ok, key=lambda r: r["cer"])
    json.dump(best, open(f"{d}/pick{os.environ.get('PICK_SET', '')}.json", "w"), ensure_ascii=False)
    print("PICK_BEST", key, best, flush=True)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "boot":
        boot(a[0])
    elif cmd == "pick":
        pick(a[0])
    elif cmd == "fdev-build":
        fdev_build()
    elif cmd == "dev":
        name, key, nfe, cfg, how = a[0], a[1], int(a[2]), float(a[3]), a[4]
        rw, rt, spd = voice(key)
        items = json.load(open("/home/data/rasa_test/items.json"))
        if os.environ.get("DEVSET") == "short":  # 40 complete short phrases written for this (data/short_dev.json):
            # the bench's short phrases broke on a fast prompt ("नमस्कार" -> "कर") while full sentences looked fine
            items = [{"id": f"short_{i:02d}", "text": t} for i, t in enumerate(json.load(open("data/short_dev.json")))]
        else:
            items = [x for x in items if x["style"] != "PROPER NOUN"][::2][:60]
        d = f"/home/data/natlab/{name}"
        os.makedirs(d, exist_ok=True)
        nseed = int(os.environ.get("NSEED", 1))  # NSEED=n: every item n times (seeds 0..n-1), to average out noise
        pairs = [(x["text"], gen(x["text"], rw, rt, spd, f"{d}/{x['id']}{f'_r{k}' if k else ''}.wav", nfe, cfg, how, seed=k))
                 for x in items for k in range(nseed)]
        _tts.clear()
        score_dev(name, pairs)
    elif cmd == "bench":
        from eval_bench import sentences
        name, key, nfe, cfg, how = a[0], a[1], int(a[2]), float(a[3]), a[4]
        rw, rt, spd = voice(key)
        d = f"/home/data/bench/runs/{name}"
        os.makedirs(d, exist_ok=True)
        for x in sentences():
            gen(x["text_devanagari"], rw, rt, spd, f"{d}/{x['sent_id']}.wav", nfe, cfg, how)
        print("BENCH_DONE", name, len(os.listdir(d)), flush=True)
    elif cmd == "fl":
        fl(a[0], a[1], a[2], int(a[3]), float(a[4]), a[5])
