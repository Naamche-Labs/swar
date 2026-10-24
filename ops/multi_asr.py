import sys, json, glob, torch, librosa
sys.path.insert(0, "/home/nepali-tts/tools")
from ref_asr_bench import transcriber
segs = sorted(glob.glob("/home/data/ref/arjan_segs/[0-9][0-9].wav"))
W = {p: librosa.load(p, sr=16000)[0] for p in segs}
res = {p: {} for p in segs}
for mid in ["sumanpaudel1997/nepali-asr-whisper-turbo", "sumanpaudel1997/nepali-asr-indicwav2vec", "gagan3012/wav2vec2-xlsr-nepali", "openai/whisper-large-v3"]:
    try:
        f = transcriber(mid) if "openai" not in mid else None
        if f is None:
            from transformers import pipeline
            p = pipeline("automatic-speech-recognition", model=mid, device=0, torch_dtype=torch.float16)
            f = lambda w, l: p({"raw": w, "sampling_rate": 16000}, generate_kwargs={"language": l, "task": "transcribe"})["text"]
            for s in segs:
                res[s]["whisper-v3 ne"] = f(W[s], "nepali"); res[s]["whisper-v3 en"] = f(W[s], "english")
        else:
            for s in segs: res[s][mid.split("/")[1]] = f(W[s])
        del f; torch.cuda.empty_cache()
    except Exception as e:
        print("FAIL", mid, type(e).__name__, str(e)[:100], flush=True)
for s in segs:
    print("=====", s[-6:]); [print(f"  {k:28s} {v}") for k, v in res[s].items()]
json.dump(res, open("/home/data/ref/arjan_segs/multi_asr.json", "w"), ensure_ascii=False, indent=1)
