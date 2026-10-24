"""Do questions sound like questions? Final pitch movement on NepTTS-Bench question sentences.

For each clip: F0 (pyin) over voiced frames; rise = median F0 of the last 25% of voiced frames minus
the median of the whole clip, in semitones. Nepali yes/no and tag questions (... छ र?) usually end
higher; statements end lower. Reported per system for the question_intonation category and, as a
contrast, for statements (phonological minimal pairs ending in । ).
Usage: python tools/question_rise.py RUN_OR_SYSTEM [...]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rise(path):
    import librosa
    w, _ = librosa.load(path, sr=16000)
    f0, v, _ = librosa.pyin(w, fmin=70, fmax=450, sr=16000, frame_length=1024)
    f0 = f0[v & ~np.isnan(f0)]
    if len(f0) < 20:
        return None
    st = 12 * np.log2(f0 / 100)
    return float(np.median(st[-max(5, len(st) // 4):]) - np.median(st))


if __name__ == "__main__":
    from rescore_bench import audio_for
    from eval_bench import sentences
    cats = {x["sent_id"]: (x["category"], x["text_devanagari"].strip()) for x in sentences()}
    res = {}
    for name in sys.argv[1:]:
        a = audio_for(name)
        q = [rise(p) for s, p in a.items() if cats[s][0] == "question_intonation" and cats[s][1].endswith("?")]
        st = [rise(p) for s, p in a.items() if cats[s][0] == "phonological_minimal_pairs" and cats[s][1].endswith("।")][:30]
        q, st = [x for x in q if x is not None], [x for x in st if x is not None]
        res[name] = {"question_rise": float(np.mean(q)), "statement_rise": float(np.mean(st)), "n_q": len(q), "n_s": len(st)}
        print(f"QR {name:26s} questions {np.mean(q):+.2f} st (n={len(q)})  statements {np.mean(st):+.2f} st  "
              f"contrast {np.mean(q) - np.mean(st):+.2f}", flush=True)
    json.dump(res, open("/home/data/bench/question_rise.json", "w"), indent=1)
