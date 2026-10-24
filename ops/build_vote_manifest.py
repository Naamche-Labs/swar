"""Fill space_vote/clips for the listening test.
Each RUN is system:voice:pattern, where pattern has {k} for the line key (e.g. ~/Downloads/v3_voices/f_v2_{k}.mp3).
Every clip is converted to 96 kbps MP3 and loudness-matched (-23 LUFS) so volume never gives a system away.
TEXTS is a JSON file {key: text}. BEST lists the systems for the best-voice tab as female=a,b;male=c,d.
Usage: python ops/build_vote_manifest.py OUT TEXTS BEST RUN [RUN ...]"""
import glob
import json
import os
import subprocess
import sys

out, texts, best, *runs = sys.argv[1:]
T = json.load(open(texts))
os.makedirs(out, exist_ok=True)
clips = []
for r in runs:
    system, voice, pat = r.split(":", 2)
    os.makedirs(f"{out}/{system}", exist_ok=True)
    for k, t in T.items():
        src = glob.glob(os.path.expanduser(pat.format(k=k)))
        if not src:
            continue
        f = f"{system}/{k}.mp3"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src[0], "-af", "loudnorm=I=-23:TP=-2", "-ar", "24000",
                        "-b:a", "96k", f"{out}/{f}"], check=True)
        clips.append({"id": f"{system}/{k}", "system": system, "voice": voice, "text": t, "file": f})
B = {v: s.split(",") for v, s in (x.split("=") for x in best.split(";") if x)}
json.dump({"clips": clips, "best": B}, open(f"{out}/manifest.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(clips)} clips from {len(runs)} systems -> {out}/manifest.json")
