"""Find the Nepali words our TTS mispronounces, and fix them in the text front end.

mine:  synthesize every FLEURS Nepali train+dev sentence (human recordings from Nepal, in no
       training set) with one calm prompt; transcribe ours and the human audio with the
       Devanagari-only MMS; a word is HARD when we get it wrong in >= 2 of its occurrences
       (>= 60%) while the human recordings of the same sentences get it right (<= 25%).
fix:   for each hard word, synthesize it in a carrier sentence spelled several ways (our G2P
       respelling, vowel-length / nasal / sibilant / व-ब / final-halanta variants) and keep the
       spelling that MMS reads back as the original word most often (3 seeds); it must beat
       the original spelling. Writes data/pron_fixes.json, which frontend.pron applies to the
       TTS input only (never to reference texts).

Usage (GPU box): python tools/mine_hard_words.py mine CKPT OUT_DIR
                 python tools/mine_hard_words.py fix CKPT OUT_DIR
"""
import collections
import difflib
import io
import json
import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from eval_heldout import IVF5_VOCAB, ed, norm  # noqa: E402

REF = "/home/data/ref/bench/refs.json"


def word_errors(ref, hyp):
    r, h = ref.split(), hyp.split()
    bad = set()
    for op, i1, i2, _, _ in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if op != "equal":
            bad |= set(range(i1, i2))
    return [(r[i], i in bad) for i in range(len(r))]


def tts_model(ckpt):
    from f5_tts.api import F5TTS
    from clone import patch_f5_chunker
    patch_f5_chunker()
    t = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt, vocab_file=IVF5_VOCAB)
    t.ema_model.float()
    return t


def mine(ckpt, out):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from clone import set_pace, speed_for
    from frontend import normalize
    from rescore_deva import DevaMMS
    os.makedirs(f"{out}/ours", exist_ok=True)
    os.makedirs(f"{out}/human", exist_ok=True)
    rows = []
    for split in ("train", "validation"):
        p = hf_hub_download("google/fleurs", f"parquet-data/ne_np/{split}-00000-of-00001.parquet", repo_type="dataset")
        rows += pq.read_table(p).to_pylist()
    ref = json.load(open(REF))["female_news"]
    tts = tts_model(ckpt)
    set_pace(ref["text"], ref["audio"])
    items, seen = [], set()
    max_sent = int(os.environ.get("MAX_SENT", 1000))  # ~14 s per long FLEURS sentence at NFE 32 on an L4
    for r in rows:
        if r["id"] not in seen and len(seen) >= max_sent:
            continue
        text = normalize(r["raw_transcription"].strip())
        hp = f"{out}/human/{r['id']}_{len(items)}.wav"
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=False)
        sf.write(hp, a, sr)
        op = f"{out}/ours/{r['id']}.wav"
        if r["id"] not in seen and not os.path.exists(op):
            tts.infer(ref_file=ref["audio"], ref_text=ref["text"], gen_text=text, file_wave=op, remove_silence=True,
                      seed=0, speed=speed_for(text), nfe_step=16)  # NFE 16 = NFE 32 on the bench (6.07 vs 6.48)
        seen.add(r["id"])
        items.append({"id": r["id"], "text": text, "ours": op, "human": hp})
    del tts
    asr = DevaMMS()
    stats = collections.defaultdict(lambda: [0, 0, 0, 0])
    done_ours = set()
    for it in items:
        ref_n = norm(it["text"])
        if it["id"] not in done_ours:  # one synthesis per sentence id, several human readings
            for w, bad in word_errors(ref_n, norm(asr(it["ours"]))):
                stats[w][0] += bad
                stats[w][1] += 1
            done_ours.add(it["id"])
        for w, bad in word_errors(ref_n, norm(asr(it["human"]))):
            stats[w][2] += bad
            stats[w][3] += 1
    hard = [{"word": w, "ours": s[:2], "human": s[2:]} for w, s in stats.items()
            if s[1] >= 2 and s[0] / s[1] >= 0.6 and s[3] >= 1 and s[2] / s[3] <= 0.25]
    hard.sort(key=lambda x: -x["ours"][0])
    tot = [sum(s[i] for s in stats.values()) for i in range(4)]
    print(f"words: ours wrong {tot[0]}/{tot[1]} ({100 * tot[0] / tot[1]:.1f}%), human wrong {tot[2]}/{tot[3]} "
          f"({100 * tot[2] / tot[3]:.1f}%); hard words {len(hard)}", flush=True)
    json.dump({"hard": hard, "items": items}, open(f"{out}/hard_words.json", "w"), ensure_ascii=False, indent=1)


SWAPS = [("ि", "ी"), ("ी", "ि"), ("ु", "ू"), ("ू", "ु"), ("ं", "ँ"), ("ँ", "ं"), ("ं", "न्"), ("श", "स"), ("ष", "स"),
         ("स", "श"), ("व", "ब"), ("ब", "व"), ("ण", "न"), ("क्ष", "छ्य"), ("ज्ञ", "ग्य")]


def variants(w):
    from frontend import g2p
    from frontend.g2p import respell
    out = {w}
    try:
        out.add(respell(g2p(w)).strip())
    except Exception:
        pass
    for a, b in SWAPS:
        if a in w:
            out.add(w.replace(a, b, 1))
    if w.endswith("्"):
        out.add(w[:-1])
    elif "क" <= w[-1] <= "ह":
        out.add(w + "्")
    return [v for v in out if v]


def fix(ckpt, out):
    from clone import set_pace, speed_for
    from rescore_deva import DevaMMS
    hard = json.load(open(f"{out}/hard_words.json"))["hard"]
    ref = json.load(open(REF))["female_news"]
    tts = tts_model(ckpt)
    set_pace(ref["text"], ref["audio"])
    asr = DevaMMS()
    carriers = ["म {} भन्छु।", "उनले {} भने।", "यो {} हो।"]
    fixes, log = {}, []
    os.makedirs(f"{out}/fix", exist_ok=True)
    for h in hard:
        w = h["word"]
        scores = {}
        for v in variants(w):
            hit = 0.0
            for k, c in enumerate(carriers):
                p = f"{out}/fix/tmp.wav"
                text = c.format(v)
                tts.infer(ref_file=ref["audio"], ref_text=ref["text"], gen_text=text, file_wave=p, remove_silence=True,
                          seed=k, speed=speed_for(text))
                # did the listener hear the word as written originally? (exact word, else closeness)
                heard = norm(asr(p)).split()
                hit += 1.0 if w in heard else max(0.0, 1 - min(ed(w, x) for x in heard) / len(w)) * 0.5 if heard else 0.0
            scores[v] = hit / len(carriers)
        best = max(scores, key=scores.get)
        log.append({"word": w, "scores": scores, "best": best})
        if best != w and scores[best] > scores[w] + 0.2:
            fixes[w] = best
        print(f"{w}: original {scores[w]:.2f} -> {best} {scores[best]:.2f}", flush=True)
    json.dump(log, open(f"{out}/fix_log.json", "w"), ensure_ascii=False, indent=1)
    json.dump(fixes, open(f"{out}/pron_fixes.json", "w"), ensure_ascii=False, indent=1)
    print("FIXES", len(fixes), "of", len(hard))


if __name__ == "__main__":
    {"mine": mine, "fix": fix}[sys.argv[1]](sys.argv[2], sys.argv[3])
