"""v4 listening fixes (user ear check of preset v2/v3, 2026-10-23): full English sounds broken, pace sounds like
a newsreader, no pause after a full stop ("...the show. Today..."), "customer" said wrong, male voice disliked
(the speaker, not the training). Usage on a GPU box (KRITI_ONNX=/home/kriti_onnx, F5_CKPT=...):
  python ops/v4_lab.py male OUT                   # rank FLEURS ne_np male speakers (CC-BY 4.0) for a new male voice
  python ops/v4_lab.py render OUT NAME:REF_WAV:REF_TEXT_FILE ...   # demo lines, one sentence at a time
  python ops/v4_lab.py customer OUT REF_WAV REF_TEXT_FILE          # spellings of "customer" in the mixed line
"""
import glob
import io
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402
from clone import patch_f5_chunker, set_pace, speed_for, to_model_text  # noqa: E402
from frontend import normalize  # noqa: E402

T = json.load(open(os.environ.get("TEXTS", "data/vote_texts.json")))
FLEURS = glob.glob("/home/hf_cache/hub/datasets--google--fleurs/snapshots/*/parquet-data/ne_np/")
HELD_MIN = 120  # ms; खान said "khan" measured 60-110, held 150+
PAUSE = {"।": 0.38, ".": 0.38, "?": 0.45, "!": 0.42}


def sentences(text):
    """Split at sentence ends; F5 runs straight through a full stop inside one generation."""
    out = [s.strip() for s in re.findall(r"[^।?!.]+[।?!.]*", text) if s.strip(" ।?!.")]
    i = 0
    while i < len(out) - 1:  # a short fragment alone comes out rushed: join it to the next
        if A.letters(out[i]) < 12:
            out[i:i + 2] = [out[i] + " " + out[i + 1]]
        else:
            i += 1
    return out


def take(m, ref_wav, ref_text, gen, p, seed, spd=1.0):
    set_pace(ref_text, ref_wav)
    m.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=gen, file_wave=p, nfe_step=32, seed=seed, cfg_strength=2.0,
            speed=speed_for(gen, spd), remove_silence=True)
    return p


_MMS = {}


def held_words(s):
    """{written word: index of the letter whose vowel the frontend holds}: खान -> 2, मिलेर -> 3, and inside a case ending
    too, आजको -> 1 (user heard "aaj" in आजको lines, which the bare-word check never measured)."""
    from frontend.pron import hold_final_schwa
    out = {}
    for w in re.findall(r"[ऀ-ॿ]+", normalize(s, english=False)):
        h = hold_final_schwa(w)
        if h != w and "्अ" in h:
            out[w] = h.index("्अ") - 1
    return out


def held_ms(path, s):
    """Shortest held final sound (ms) over the sentence's held words: MMS emissions, forced alignment of the written
    sentence, time from the word's last letter to the next sound (tools/final_vowel.py). A swallowed vowel ("khan" for
    खान) is short. MMS is research-only (CC-BY-NC): used to pick demo takes, not in the product path."""
    import librosa
    import torch
    import torchaudio.functional as AF
    words = held_words(s)
    if not words:
        return None
    if not _MMS:
        from rescore_deva import DevaMMS
        _MMS["m"] = DevaMMS()
    mms = _MMS["m"]
    vocab = mms.proc.tokenizer.get_vocab()
    pad = mms.proc.tokenizer.pad_token_id
    w = librosa.load(path, sr=16000)[0]
    inp = {k: v.to(mms.model.device) for k, v in mms.proc(w, sampling_rate=16000, return_tensors="pt").items()}
    with torch.no_grad():
        lg = mms.model(**inp).logits[0] + mms.mask
    text = re.sub(r"\s+", " ", re.sub(r"[^ऀ-ॿ ]", " ", normalize(s))).strip()  # English words in Devanagari, as said
    chars = [c for c in ("|" if c == " " else c for c in text) if c in vocab]
    try:
        al, _ = AF.forced_align(torch.log_softmax(lg.float(), -1).unsqueeze(0).cpu(), torch.tensor([[vocab[c] for c in chars]]), blank=pad)
    except Exception:
        return None
    starts, prev = [], None
    for f, a in enumerate(al[0].tolist()):
        if a != pad and a != prev:
            starts.append(f)
        prev = a if a != pad else None
    toks, out, pos = "".join(chars).split("|"), [], 0
    for t in toks:
        last = pos + words.get(t, len(t) - 1)
        if t in words and last + 1 < len(starts):
            out.append((starts[last + 1] - starts[last]) * 20)
        pos += len(t) + 1
    return min(out) if out else None


def best_sentence(m, ref_wav, ref_text, s, stem, seeds=int(os.environ.get("SEEDS", "8")), english_latin=True, spd=1.0):
    """Best of `seeds` takes: pronunciation first (Kriti for Nepali/mixed, Whisper WER for pure English), then UTMOS."""
    english = A._is_english(s)
    mode = os.environ.get("EN_MODE", "latin")  # latin: English text, Nepali prompt as is; roman: prompt text romanized too
    gen = s if (english and mode in ("latin", "roman")) else to_model_text(s, "chars")
    if english and mode == "roman":
        from frontend.romanize import romanize
        ref_text = romanize(ref_text)
    if english and mode == "boot":  # the voice's own English (made from Nepali-letter English) as an English prompt
        ref_wav, ref_text = boot_prompt(m, ref_wav, ref_text, os.path.dirname(stem))
    cands = [{"file": take(m, ref_wav, ref_text, gen, f"{stem}_s{k}.wav", k, spd)} for k in range(seeds)]
    if english:
        for c in cands:
            c["heard"] = A.whisper_en(c["file"])
            c["err"] = A.wer_en(s, c["heard"])
    else:
        for c, h in zip(cands, A.read_back([c["file"] for c in cands])):
            c["heard"], c["err"] = h, A.pcer(normalize(s), h)
    for c in cands:
        c["nat"] = A._nat(c["file"])
    if not english and held_words(s):  # user: खान heard as "khan"; drop takes that swallow a held vowel
        for c in cands:
            c["held"] = held_ms(c["file"], s)
        ok = [c for c in cands if c["held"] is not None and c["held"] >= HELD_MIN]
        if ok:
            cands = ok
        else:
            cands = [max(cands, key=lambda c: c["held"] or 0)]
    floor = min(c["err"] for c in cands)
    b = max([c for c in cands if c["err"] <= floor + 0.03], key=lambda c: c["nat"])
    for f in glob.glob(f"{stem}_s*.wav"):
        if f != b["file"]:
            os.remove(f)
    return {**b, "gen": gen, "mean_err": float(np.mean([c["err"] for c in cands])), "held": b.get("held")}


BOOT_LINES = ["Hello, my name is Swar, and I am really happy to talk with you today.",
              "We started this project because we wanted a voice that sounds like home."]
_BOOT = {}


def boot_prompt(m, ref_wav, ref_text, out):
    """English prompt in the same voice, from our own model only: say BOOT_LINES in Nepali-letter English (best of 6 by
    Whisper WER, then UTMOS), keep the Latin text as its transcript."""
    if ref_wav in _BOOT:
        return _BOOT[ref_wav]
    parts, sr = [], 24000
    for i, line in enumerate(BOOT_LINES):
        gen = to_model_text(line, "chars")
        cs = [{"file": take(m, ref_wav, ref_text, gen, f"{out}/boot_{os.path.basename(ref_wav)[:-4]}_{i}_{k}.wav", k)}
              for k in range(6)]
        for c in cs:
            c["wer"], c["nat"] = A.wer_en(line, A.whisper_en(c["file"])), A._nat(c["file"])
        floor = min(c["wer"] for c in cs)
        b = max([c for c in cs if c["wer"] <= floor + 0.05], key=lambda c: c["nat"])
        w, sr = sf.read(b["file"])
        parts += [w, np.zeros(int(0.3 * sr))]
        print("BOOT", line[:30], "wer", round(b["wer"], 2), "nat", round(b["nat"], 2), flush=True)
    p = f"{out}/boot_{os.path.basename(ref_wav)}"
    sf.write(p, np.concatenate(parts[:-1]), sr)
    _BOOT[ref_wav] = (p, " ".join(BOOT_LINES))
    return _BOOT[ref_wav]


def speak(m, ref_wav, ref_text, text, out_wav, **kw):
    parts, rows, sr = [], [], 24000
    for i, s in enumerate(sentences(text)):
        r = best_sentence(m, ref_wav, ref_text, s, out_wav[:-4] + f"_{i}", **kw)
        w, sr = sf.read(r["file"])
        os.remove(r["file"])
        parts += [w, np.zeros(int(sr * PAUSE.get(s[-1], 0.3)))]
        rows.append({k: r[k] for k in ("gen", "heard", "err", "mean_err", "nat", "held")})
    sf.write(out_wav, np.concatenate(parts[:-1]), sr)
    return rows


def f0_median(x, sr):
    import librosa
    f = librosa.yin(x, fmin=60, fmax=400, sr=sr)
    v = librosa.feature.rms(y=x)[0][:len(f)] > 0.02
    return float(np.median(f[:len(v)][v])) if v.any() else 0.0


def male(out):
    """FLEURS has gender but no speaker ids: keep clean male clips, group them by voice (ECAPA), best clip per voice."""
    import librosa
    import pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True)
    rows = []
    for split in ("train", "validation", "test"):
        t = pq.read_table(glob.glob(FLEURS[0] + f"{split}-*.parquet")[0],
                          columns=["id", "audio", "raw_transcription", "gender", "num_samples"]).to_pylist()
        rows += [r for r in t if r["gender"] == 0 and 5.5 * 16000 <= r["num_samples"] <= 11 * 16000]
    print("male clips 5.5-11 s:", len(rows), flush=True)
    seen, cands = set(), []
    for r in rows:
        if r["raw_transcription"] in seen:
            continue
        seen.add(r["raw_transcription"])
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        x = x.mean(1) if x.ndim > 1 else x
        x16 = (librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x).astype(np.float32)
        f0 = f0_median(x16, 16000)
        if not 75 <= f0 <= 150:
            continue
        from mos import utmos
        cands.append({"id": r["id"], "text": r["raw_transcription"], "f0": f0, "nat": float(utmos(x16)), "x": x16})
    cands.sort(key=lambda c: -c["nat"])
    top = cands[:160]
    print("pitch-checked:", len(cands), "UTMOS top:", [round(c["nat"], 2) for c in top[:10]], flush=True)
    E = np.stack([A.emb(c["x"]) for c in top])
    from scipy.cluster.hierarchy import fcluster, linkage
    lab = fcluster(linkage(E, "average", metric="cosine"), t=1 - 0.6, criterion="distance")
    groups = {}
    for c, g in zip(top, lab):
        groups.setdefault(int(g), []).append(c)
    voices = sorted(groups.values(), key=lambda g: -g[0]["nat"])[:10]
    meta = []
    for i, g in enumerate(voices):
        c = g[0]
        name = f"m{i:02d}"
        sf.write(f"{out}/{name}.wav", c["x"], 16000)
        open(f"{out}/{name}.txt", "w").write(c["text"])
        meta.append({"name": name, "fleurs_id": c["id"], "clips": len(g), "f0": round(c["f0"]), "utmos": round(c["nat"], 2),
                     "text": c["text"]})
        print(meta[-1], flush=True)
    json.dump(meta, open(f"{out}/male.json", "w"), ensure_ascii=False, indent=1)


def render(out, specs, keys=None):
    from f5_tts.api import F5TTS
    patch_f5_chunker()
    os.makedirs(out, exist_ok=True)
    m = F5TTS(model="F5TTS_Base", ckpt_file=A.CKPT, vocab_file=A.VOCAB)
    m.ema_model.float()
    res = json.load(open(f"{out}/render.json")) if os.path.exists(f"{out}/render.json") else {}
    for spec in specs:
        name, wav, txt = spec.split(":")
        rt = open(txt).read().strip()
        rt = rt if A._is_english(rt) else normalize(rt)  # an English prompt keeps its English transcript
        spd = float(os.environ.get("SPEED", "1.0"))
        for k in keys or T:
            res[f"{name}/{k}"] = speak(m, wav, rt, T[k], f"{out}/{name}_{k}.wav", spd=spd)
            r = res[f"{name}/{k}"]
            print(f"{name:10s} {k:11s} err {np.mean([x['err'] for x in r]) * 100:5.1f} UTMOS "
                  f"{np.mean([x['nat'] for x in r]):.2f} | {' / '.join(x['heard'] for x in r)[:90]}", flush=True)
        json.dump(res, open(f"{out}/render.json", "w"), ensure_ascii=False, indent=1)


def customer(out, wav, txt):
    """The mixed line with different spellings of "customer", 6 takes each, Kriti read-back of the first word."""
    from f5_tts.api import F5TTS
    patch_f5_chunker()
    os.makedirs(out, exist_ok=True)
    m = F5TTS(model="F5TTS_Base", ckpt_file=A.CKPT, vocab_file=A.VOCAB)
    m.ema_model.float()
    rt = normalize(open(txt).read().strip())
    line = "कस्टमर सर्भिस मा कल गर्दा उनीहरूले भने।"
    res = {}
    for tag, w in [("ksTmr", "कस्टमर"), ("ksTamr", "कस्टम्अर"), ("kasTomar", "कस्टोमर"), ("kaSTmr", "कष्टमर"),
                   ("kustmr", "कुस्टमर"), ("latin", "Customer")]:
        gen = line.replace("कस्टमर", w)
        gen = gen if tag == "latin" else normalize(gen, english=False)
        files = [take(m, wav, rt, gen, f"{out}/{tag}_{k}.wav", k) for k in range(6)]
        heard = A.read_back(files)
        first = [h.split()[0] if h.split() else "" for h in heard]
        res[tag] = {"input": gen, "first_word": first, "ok": sum(f.startswith("कस्टम") for f in first)}
        print(tag, res[tag], flush=True)
    json.dump(res, open(f"{out}/customer.json", "w"), ensure_ascii=False, indent=1)


def accent(dirs):
    """Native-English likelihood (CommonAccent ECAPA, as tools/accent_score.py) of every en_* clip under each dir."""
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier
    clf = EncoderClassifier.from_hparams("Jzuluaga/accent-id-commonaccent_ecapa", savedir="/home/hf_cache/accent",
                                         run_opts={"device": "cuda"})
    labels = clf.hparams.label_encoder.decode_ndim(list(range(16)))
    native = {"us", "england", "canada", "australia", "newzealand", "ireland", "scotland"}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*_en_*.wav")):
            with torch.no_grad():
                pr = torch.softmax(clf.classify_file(f)[0][0], -1).cpu().numpy()
            top = labels[int(pr.argmax())]
            print(f"ACCENT {f.replace('/home/v4/', ''):30s} native {100 * sum(pr[i] for i, l in enumerate(labels) if l in native):5.1f}%"
                  f"  indian {100 * pr[labels.index('indian')]:5.1f}%  top {top}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "male":
        male(sys.argv[2])
    elif cmd == "render":
        render(sys.argv[2], sys.argv[3:], keys=os.environ.get("KEYS", "").split(",") if os.environ.get("KEYS") else None)
    elif cmd == "accent":
        accent(sys.argv[2:])
    elif cmd == "customer":
        customer(*sys.argv[2:5])
