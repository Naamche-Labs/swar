"""Nepali + English (code-switch) test. data/codeswitch_sentences.json (60, used while building
frontend/english.py) and data/codeswitch_heldout.json (30, written before any output was heard:
the honest number). Reference = our normalized text (English in Devanagari as Nepali speakers say
it); scored with the Devanagari-only MMS. RAW=1 feeds the English words to the model in Latin
letters (the old behaviour), same reference.
Usage: python tools/eval_codeswitch.py NAME CKPT [dev|heldout]"""
import json
import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from eval_heldout import IVF5_VOCAB, ed, norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_dir(name, which="heldout"):
    """Score audio made elsewhere (e.g. OmniVoice) in /home/data/codeswitch/{name}_{which}."""
    import librosa
    from frontend import normalize
    from mos import utmos
    from rescore_deva import DevaMMS
    items = json.load(open(f"{ROOT}/data/codeswitch_{'sentences' if which == 'dev' else 'heldout'}.json"))
    asr, E, N, m = DevaMMS(), 0, 0, []
    for it in items:
        p = f"/home/data/codeswitch/{name}_{which}/{it['id']}.wav"
        t = norm(normalize(it["text"]))
        E, N = E + ed(t, norm(asr(p))), N + len(t)
        m.append(utmos(librosa.load(p, sr=16000)[0]))
    print(f"CS {name:20s} {which:8s} CER {100 * E / N:5.2f}  UTMOS {sum(m) / len(m):.2f}  (n={len(items)})", flush=True)


def main(name, ckpt, which="heldout"):
    import librosa
    from f5_tts.api import F5TTS
    from clone import patch_f5_chunker, set_pace, speed_for
    from frontend import normalize
    from frontend import english
    from mos import utmos
    from rescore_deva import DevaMMS
    items = json.load(open(f"{ROOT}/data/codeswitch_{'sentences' if which == 'dev' else 'heldout'}.json"))
    ref = json.load(open("/home/data/ref/bench/refs.json"))["female_news"]
    out = f"/home/data/codeswitch/{name}_{which}"
    os.makedirs(out, exist_ok=True)
    patch_f5_chunker()
    tts = F5TTS(model=os.environ.get("F5_MODEL", "F5TTS_Base"), ckpt_file=ckpt, vocab_file=IVF5_VOCAB)
    tts.ema_model.float()
    set_pace(ref["text"], ref["audio"])
    raw = os.environ.get("RAW") == "1"
    for it in items:
        p = f"{out}/{it['id']}.wav"
        if raw:
            keep = english.english_to_devanagari
            english.english_to_devanagari = lambda t: t
            g = normalize(it["text"])
            english.english_to_devanagari = keep
        else:
            g = normalize(it["text"])
        if not os.path.exists(p):
            tts.infer(ref_file=ref["audio"], ref_text=ref["text"], gen_text=g, file_wave=p, remove_silence=True, seed=0,
                      speed=speed_for(g))
    del tts
    asr = DevaMMS()
    E = N = 0
    m = []
    for it in items:
        p = f"{out}/{it['id']}.wav"
        t = norm(normalize(it["text"]))
        E, N = E + ed(t, norm(asr(p))), N + len(t)
        m.append(utmos(librosa.load(p, sr=16000)[0]))
    print(f"CS {name:20s} {which:8s} CER {100 * E / N:5.2f}  UTMOS {sum(m) / len(m):.2f}  (n={len(items)})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "score-dir":
        score_dir(*sys.argv[2:4])
    else:
        main(*sys.argv[1:4])
