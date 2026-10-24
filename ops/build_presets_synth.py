"""Preset voices for a public release, from model-made audio only: the demo voices Maya and Aarav are podcast speakers
(used with permission), and a public repo must not carry their raw recordings. Each preset is enrolled from the
model's own read-style takes in that voice (ops/nat_lab.py boot: FLEURS-train sentences, CER from MMS read-back),
keeping the highest-UTMOS takes among those with CER <= MAX_CER (3.5). Output: OUT/<name>/ (voice.json + pieces) and OUT/voices.json.
Usage (GPU box): KRITI_ONNX=/home/kriti_onnx python ops/build_presets_synth.py OUT name:BOOT_LOG [name:BOOT_LOG ...]"""
import json
import os
import sys

sys.path[:0] = ["pipeline", "tools", "."]
import autoclone as A  # noqa: E402

GENDER = {"maya": "female", "aarav": "male"}
N = 5
out, specs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
meta = {}
for spec in specs:
    name, log = spec.split(":")
    d = f"/home/data/natlab/boot_{name}"
    rows = [l.split() for l in open(log) if l.startswith(f"BOOT {name} ")]  # BOOT name i_s CER x UTMOS y
    ok = sorted((-float(r[6]), f"{d}/{r[2]}.wav") for r in rows if float(r[4]) <= float(os.environ.get("MAX_CER", 3.5)))
    wavs = [w for _, w in ok][:N]
    A.enroll(f"{out}/{name}", wavs, one_speaker=True)
    meta[name] = {"gender": GENDER.get(name, "voice"), "style": "read", "source": "model-made takes only"}
    print("PRESET", name, len(wavs), flush=True)
json.dump(meta, open(f"{out}/voices.json", "w"), indent=1)
print("PRESETS_DONE")
