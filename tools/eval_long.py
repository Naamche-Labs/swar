"""Long-text test: 40 paragraphs of 3 unseen Rasa test sentences (WIKI/NEWS/BOOK, one
voice each), 20-45 s of speech. Compares F5's stock chunker (splits only on Latin/Chinese
punctuation, never on ।) with our sentence-aware one (pipeline/clone.py).

Usage: CHUNK=stock|nepali python tools/eval_long.py run NAME CKPT ; python tools/eval_long.py table
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/nepali-tts/pipeline")
from eval_heldout import DEV, IVF5_VOCAB, ed, norm  # noqa: E402
from eval_rasa import PROMPT, R  # noqa: E402

L = "/home/data/long_test"


def paragraphs():
    items = [it for it in json.load(open(f"{R}/items.json")) if it["style"] in ("WIKI", "NEWS", "BOOK")]
    out = []
    for v in ("Female", "Male"):
        its = [it for it in items if it["voice"] == v]
        for i in range(0, len(its) - 2, 3):
            out.append({"id": f"{v[0]}{i // 3:02d}", "voice": v, "text": " ".join(x["text"] for x in its[i:i + 3])})
    return out


def run(name, ckpt, vocab=IVF5_VOCAB):
    import librosa
    import torch
    from f5_tts.api import F5TTS
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    import clone
    from mos import utmos
    if os.environ.get("CHUNK", "nepali") == "nepali":
        clone.patch_f5_chunker()
    import f5_tts.infer.utils_infer as U
    seen = []
    orig = U.chunk_text
    U.chunk_text = lambda t, max_chars=135: seen.append(orig(t, max_chars)) or seen[-1]
    tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt, vocab_file=vocab)
    tts.ema_model.float()
    refs = json.load(open("/home/data/ref/bench/refs.json"))
    out = f"{L}/runs/{name}"
    os.makedirs(out, exist_ok=True)
    paras = paragraphs()
    n_chunks = {}
    for p in paras:
        ref = refs[PROMPT[p["voice"]]]
        clone.set_pace(ref["text"], ref["audio"])
        f = f"{out}/{p['id']}.wav"
        if not os.path.exists(f):
            tts.infer(ref_file=ref["audio"], ref_text=ref["text"], gen_text=p["text"], file_wave=f,
                      remove_silence=True, seed=0)
            n_chunks[p["id"]] = len(seen[-1])
    del tts
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                         ignore_mismatched_sizes=True).to(DEV).eval()
    rows = []
    for p in paras:
        w, _ = librosa.load(f"{out}/{p['id']}.wav", sr=16000)
        inp = {k: v.to(DEV) for k, v in proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            hyp = norm(proc.decode(asr(**inp).logits.argmax(-1)[0]))
        ref = norm(p["text"])
        rows.append({"id": p["id"], "e": ed(ref, hyp), "n": len(ref), "sec": len(w) / 16000, "mos": utmos(w),
                     "chunks": n_chunks.get(p["id"]), "hyp": hyp, "ref": ref})
    res = {"name": name, "cer": 100 * sum(r["e"] for r in rows) / sum(r["n"] for r in rows),
           "mos": sum(r["mos"] for r in rows) / len(rows), "sec": sum(r["sec"] for r in rows) / len(rows),
           "chunks": sum(r["chunks"] or 0 for r in rows) / len(rows), "rows": rows}
    json.dump(res, open(f"{L}/runs/{name}.json", "w"), ensure_ascii=False, indent=1)
    show(res)


def show(r):
    print(f"{r['name']:24s} CER {r['cer']:5.2f}%  UTMOS {r['mos']:.2f}  avg {r['sec']:.1f} s  chunks/para {r['chunks']:.1f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:4])
    else:
        for f in sorted(glob.glob(f"{L}/runs/*.json")):
            show(json.load(open(f)))
