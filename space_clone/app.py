"""Hugging Face Space (ZeroGPU): drop 20-30 s of someone talking, type Nepali, English or both, hear it in their voice.
Same pipeline as app/cloner.py (pipeline/autoclone.py), made to fit a ZeroGPU Space: one process, Kriti from ONNX
(KRITI_ONNX, no NeMo), weights pulled from a private model repo (MODEL_REPO, read with the HF_TOKEN secret).
GPU work runs inside @spaces.GPU functions; outside a Space the decorator is a no-op.
Preset voices: MODEL_REPO/presets/<name>/ (voice.json + pieces, built by ops/build_presets.py from data/voices.json).
English is spoken from Nepali-letter English (frontend/english.py), the way the user preferred by ear; BILINGUAL=0 so
the speaker's own English prompts are not used for English lines.
Files: deploy with ops/deploy_space_clone.sh (copies pipeline/, frontend/, tools/ and data/ next to this file)."""
import hashlib
import json
import os
import sys

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "pipeline"), os.path.join(HERE, "tools"), HERE]

try:
    import spaces
    gpu = spaces.GPU
except ImportError:  # local run
    def gpu(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

if os.environ.get("MODEL_REPO"):
    from huggingface_hub import snapshot_download
    d = snapshot_download(os.environ["MODEL_REPO"], token=os.environ.get("HF_TOKEN"))
    os.environ.setdefault("F5_CKPT", os.path.join(d, os.environ.get("MODEL_FILE", "podvoices_ema.pt")))
    # English sentences go to swar_v6_ema.pt when the repo has it (pipeline/autoclone.py, F5_EN_CKPT)
    if os.path.exists(os.path.join(d, os.environ.get("EN_MODEL_FILE", "swar_v6_ema.pt"))):
        os.environ.setdefault("F5_EN_CKPT", os.path.join(d, os.environ.get("EN_MODEL_FILE", "swar_v6_ema.pt")))
    os.environ.setdefault("PRESETS_DIR", os.path.join(d, "presets"))
    os.environ.setdefault("F5_VOCAB", os.path.join(d, "vocab.txt"))
    os.environ.setdefault("KRITI_ONNX", os.path.join(d, "kriti_onnx"))
os.environ.setdefault("KRITI_PY", "/nonexistent")
os.environ.setdefault("CONFORMER_PY", "/nonexistent")
os.environ.setdefault("BILINGUAL", "0")
import autoclone as A  # noqa: E402

VOICES = os.environ.get("VOICES_DIR", "/tmp/voices")
os.makedirs(VOICES, exist_ok=True)


def _presets():
    """{label: voice id} for the preset voices; their folders are copied next to the uploaded voices."""
    import shutil
    src, out = os.environ.get("PRESETS_DIR", ""), {}
    if not src or not os.path.isdir(src):
        return out
    meta = json.load(open(os.path.join(src, "voices.json"))) if os.path.exists(os.path.join(src, "voices.json")) else {}
    for name in sorted(os.listdir(src)):
        if os.path.isfile(os.path.join(src, name, "voice.json")):
            if not os.path.exists(f"{VOICES}/{name}"):
                shutil.copytree(os.path.join(src, name), f"{VOICES}/{name}")
            out[f"{name.capitalize()} ({meta.get(name, {}).get('gender', 'voice')})"] = name
    return out


PRESETS = _presets()


@gpu(duration=120)
def _enroll(vd, paths, same):
    return A.enroll(vd, paths, one_speaker=same)


@gpu(duration=120)
def _say(vd, text, out, takes):
    return A.say_long(vd, text, out, takes=takes)


def do_enroll(files, consent, same):
    if not consent:
        raise gr.Error("Please confirm the speaker agreed to be cloned.")
    if not files:
        raise gr.Error("Drop at least one clip (20 seconds or more of talking is best).")
    paths = sorted(f if isinstance(f, str) else f.name for f in files)
    vid = hashlib.sha1(("".join(open(p, "rb").read(1 << 20).hex()[:64] for p in paths) + str(same)).encode()).hexdigest()[:10]
    vd = f"{VOICES}/{vid}"
    v = json.load(open(f"{vd}/voice.json")) if os.path.exists(f"{vd}/voice.json") else _enroll(vd, paths, same)
    main = [p for p in v["pieces"] if p["main"]]
    sec = sum(p["speech"] for p in main)
    msg = f"Voice ready: {sec:.0f} s of speech." + ("  Tip: 20 seconds or more gives a closer voice." if sec < 15 else "")
    if len(v["pieces"]) > len(main):
        msg += " Some parts sounded like another person and were left out (tick the box if it is all one person)."
    return vid, msg


def do_say(vid, text, quality):
    if not vid:
        raise gr.Error("Make the voice first (step 1).")
    if not text.strip():
        raise gr.Error("Type something to say.")
    if len(text) > 600:
        raise gr.Error("Please keep it under 600 characters here.")
    takes = {"Fast": 1, "Best": 4}[quality]
    out = f"{VOICES}/{vid}/say_{hashlib.sha1((text + quality).encode()).hexdigest()[:8]}.wav"
    b = _say(f"{VOICES}/{vid}", text, out, takes)
    return out, f"Voice match {b['sim']:.2f}"


with gr.Blocks(title="Nepali voice cloning") as demo:
    gr.Markdown("## Nepali voice cloning\n1. Drop 20-30 seconds of one person talking (a voice note is fine).  "
                "2. Type Nepali, English or both.  3. Listen.  Only clone a voice with that person's permission.")
    vid = gr.State("")
    if PRESETS:
        with gr.Row():
            pick = gr.Dropdown(list(PRESETS), label="Or pick one of our voices", value=None)
            pinfo = gr.Textbox(label="Voice", interactive=False)
        pick.change(lambda k: (PRESETS.get(k, ""), f"Using {k}." if k else ""), [pick], [vid, pinfo])
    with gr.Row():
        files = gr.File(file_count="multiple", file_types=["audio"], label="Clips of the voice")
        with gr.Column():
            consent = gr.Checkbox(label="The speaker agreed to have their voice cloned")
            same = gr.Checkbox(label="Every clip is this one person", value=True)
            make = gr.Button("1. Make the voice", variant="primary")
            info = gr.Textbox(label="Voice", interactive=False, lines=2)
    text = gr.Textbox(label="2. Text", value="नमस्ते! आज office मा meeting धेरै लामो भयो।", lines=4)
    quality = gr.Radio(["Fast", "Best"], value="Best", label="Quality")
    go = gr.Button("3. Speak", variant="primary")
    audio = gr.Audio(label="Result", type="filepath")
    note = gr.Textbox(label="Check", interactive=False)
    make.click(do_enroll, [files, consent, same], [vid, info])
    go.click(do_say, [vid, text, quality], [audio, note])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
