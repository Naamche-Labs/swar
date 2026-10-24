"""Public zero-shot cloning test on Google FLEURS Nepali (ne_np test, CC-BY 4.0).

FLEURS speakers were recorded in Nepal (read Wikipedia sentences), unlike our
IndicVoices-R/Rasa training data (Indian Nepali), so this is also an accent check.
FLEURS has no speaker ids: test recordings are grouped into speakers by ECAPA
embeddings (average-linkage clustering, cosine distance 0.6, within gender). Per speaker,
one 4-12 s clip is the prompt (gold transcript) and up to 8 other sentences are
targets; their real recordings are scored the same way (ceiling).

Scores: Devanagari-only MMS CER (tools/rescore_deva.py), ECAPA similarity to the
real recording, UTMOS.

Usage (GPU box):
    python tools/eval_fleurs.py build
    python tools/eval_fleurs.py run NAME CKPT
    python tools/eval_fleurs.py real
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_heldout import DEV, IVF5_VOCAB, ed, norm  # noqa: E402

F = "/home/data/fleurs"


def build():
    import librosa
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import AgglomerativeClustering
    from speechbrain.inference.speaker import EncoderClassifier
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend import normalize
    p = hf_hub_download("google/fleurs", "parquet-data/ne_np/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(p).to_pylist()
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV},
                                         savedir="/home/hf_cache/ecapa")
    os.makedirs(f"{F}/wav", exist_ok=True)
    clips = []
    for r in rows:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=False)
        dur = len(a) / sr
        if not 2 <= dur <= 15:
            continue
        w24 = librosa.resample(a, orig_sr=sr, target_sr=24000)
        path = f"{F}/wav/{r['id']}_{len(clips)}.wav"
        sf.write(path, w24, 24000)
        w16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
        with torch.no_grad():
            e = torch.nn.functional.normalize(spk.encode_batch(torch.tensor(w16).unsqueeze(0).to(DEV)).squeeze(), dim=-1)
        clips.append({"path": path, "dur": dur, "gender": r["gender"], "text": normalize(r["raw_transcription"].strip()),
                      "emb": e.cpu().numpy()})
    items = []
    import collections
    for g in sorted({c["gender"] for c in clips}):
        cs = [c for c in clips if c["gender"] == g]
        lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=float(os.environ.get("DIST", 0.6))).fit_predict(np.stack([c["emb"] for c in cs]))
        print(g, "cluster sizes", sorted(collections.Counter(lab).values(), reverse=True)[:25], flush=True)
        for k in sorted(set(lab)):
            group = sorted([c for c, l in zip(cs, lab) if l == k], key=lambda c: c["path"])
            refs = [c for c in group if 4 <= c["dur"] <= 12]
            if len(group) < 2 or not refs:
                continue
            ref = refs[0]
            for t in [c for c in group if c is not ref and c["text"] != ref["text"]][:int(os.environ.get("PER", 8))]:
                items.append({"id": f"{len(items):03d}", "speaker": f"{g}_{k}", "ref_wav": ref["path"],
                              "ref_text": ref["text"], "text": t["text"], "real_wav": t["path"]})
    json.dump(items, open(f"{F}/items.json", "w"), ensure_ascii=False, indent=1)
    print(len(clips), "clips,", len(items), "items from", len({i["speaker"] for i in items}), "speakers")


def score(name, wavs):
    import librosa
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from mos import utmos
    from rescore_deva import DevaMMS
    items = json.load(open(f"{F}/items.json"))
    asr = DevaMMS()
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV},
                                         savedir="/home/hf_cache/ecapa")

    def emb(p):
        w, _ = librosa.load(p, sr=16000)
        with torch.no_grad():
            return spk.encode_batch(torch.tensor(w).unsqueeze(0).to(DEV)).squeeze()
    rows = []
    for it in items:
        p = wavs.get(it["id"])
        if not p or not os.path.exists(p):
            continue
        ref, hyp = norm(it["text"]), norm(asr(p))
        other = it["real_wav"] if name != "real" else it["ref_wav"]
        w, _ = librosa.load(p, sr=16000)
        rows.append({"id": it["id"], "e": ed(ref, hyp), "n": len(ref), "hyp": hyp, "ref": ref, "mos": utmos(w),
                     "sim": float(torch.nn.functional.cosine_similarity(emb(p), emb(other), dim=0))})
    n = max(1, len(rows))
    res = {"name": name, "n": len(rows), "cer": 100 * sum(r["e"] for r in rows) / max(1, sum(r["n"] for r in rows)),
           "mean_file_cer": 100 * sum(r["e"] / max(1, r["n"]) for r in rows) / n,
           "sim": sum(r["sim"] for r in rows) / n, "mos": sum(r["mos"] for r in rows) / n, "rows": rows}
    os.makedirs(f"{F}/runs", exist_ok=True)
    json.dump(res, open(f"{F}/runs/{name}.json", "w"), ensure_ascii=False, indent=1)
    print(f"fleurs {name:24s} CER {res['cer']:5.2f}  mean-file {res['mean_file_cer']:5.2f}  sim {res['sim']:.3f}  "
          f"UTMOS {res['mos']:.2f}  (n={res['n']})", flush=True)


def run(name, ckpt, vocab=IVF5_VOCAB):
    from f5_tts.api import F5TTS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
    from clone import patch_f5_chunker, set_pace, speed_for
    patch_f5_chunker()
    tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt, vocab_file=vocab)
    tts.ema_model.float()
    out = f"{F}/runs/{name}"
    os.makedirs(out, exist_ok=True)
    wavs = {}
    for it in json.load(open(f"{F}/items.json")):
        p = f"{out}/{it['id']}.wav"
        if not os.path.exists(p):
            set_pace(it["ref_text"], it["ref_wav"])
            tts.infer(ref_file=it["ref_wav"], ref_text=it["ref_text"], gen_text=it["text"], file_wave=p,
                      remove_silence=True, seed=0, speed=speed_for(it["text"]))
        wavs[it["id"]] = p
    del tts
    score(name, wavs)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "run":
        run(*sys.argv[2:4])
    elif cmd == "score-dir":  # audio made elsewhere (e.g. tools/gen_omnivoice.py): NAME DIR
        score(sys.argv[2], {it["id"]: f"{sys.argv[3]}/{it['id']}.wav" for it in json.load(open(f"{F}/items.json"))})
    elif cmd == "real":
        score("real", {it["id"]: it["real_wav"] for it in json.load(open(f"{F}/items.json"))})
