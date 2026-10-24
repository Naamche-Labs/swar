"""Figures for the arXiv paper (numbers from EXPERIMENTS.md, 2026-10-23). Usage: python make_figs.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OURS, OTHER = "#1f4e79", "#9aa7b4"

# 1. NepTTS-Bench pronunciation CER
sys_ = [("Swar (final)", 2.92), ("OmniVoice", 4.32), ("Swar, Maya voice", 4.52), ("Indic-Speak Srijana", 8.12), ("Indic Parler-TTS", 9.51), ("Indic-Speak Sagar", 10.53),
        ("TingTing Subina", 11.25), ("ElevenLabs v3", 12.91), ("Gemini", 15.03), ("Piper", 17.00)]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
names, vals = zip(*sys_[::-1])
ax.barh(names, vals, color=[OURS if n.startswith("Swar") else OTHER for n in names], height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.25, i, f"{v:.1f}", va="center", fontsize=7)
ax.set_xlabel("Pronunciation CER (%), lower is better")
ax.set_xlim(0, 19.5)
fig.tight_layout()
fig.savefig("fig_bench.pdf")

# 2. Held final vowel duration
words = ["khān(a)", "āj(a)", "kin(a)", "jān(a)", "gaer(a)"]
written = [108, None, None, 92, None]
hal = [155, 172, 142, 112, 208]
plain = [172, 192, 175, 152, 218]
fig, ax = plt.subplots(figsize=(3.4, 1.7))
x = range(len(words))
w = 0.26
ax.bar([i - w for i in x], [v or 0 for v in written], w, color="#c9cfd6", label="as written")
ax.bar(list(x), hal, w, color=OURS, label="+ halanta+a (used)")
ax.bar([i + w for i in x], plain, w, color=OTHER, label="+ plain a (heard as distorted)")
ax.set_xticks(list(x), words)
ax.set_ylabel("held final sound (ms)")
ax.axhline(120, color="#888", lw=0.6, ls="--")
ax.text(-0.55, 126, "take check (120 ms)", fontsize=6, ha="left", color="#555")
ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig("fig_vowel.pdf")

# 3. English routes: objective scores vs the listener
routes = ["English text,\nNepali prompt", "English text,\nown English prompt", "Nepali-letter,\nold spellings",
          "Nepali-letter,\nreviewed spellings"]
# best-take Whisper WER (%) and UTMOS, means over the runs in EXPERIMENTS.md: bilingual model with two Rasa prompts
# (6 takes); podcast-voices model, own English prompt, best of 12 (4 voices); old spellings, best of 4 (4 voices);
# reviewed spellings, best of 8 (5 voices)
wer = [59.5, 5.8, 0.3, 0.0]
utm = [3.52, 4.18, 3.99, 4.12]
heard = ["unusable", "rejected", "broken", "accepted"]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
cols = ["#c9cfd6", OTHER, OTHER, OURS]
ax.bar(range(4), utm, color=cols, width=0.6)
for i, (u, h, e) in enumerate(zip(utm, heard, wer)):
    ax.text(i, u + 0.05, f"{h}\nWER {e:.1f}%", ha="center", fontsize=6)
ax.set_xticks(range(4), routes, fontsize=6)
ax.set_ylabel("UTMOS")
ax.set_ylim(3.0, 4.75)
fig.tight_layout()
fig.savefig("fig_english.pdf")
print("figures written")
