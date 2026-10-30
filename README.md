# Swar स्वर: open Nepali voice cloning

Swar speaks Nepali the way it is spoken in Nepal. Give it 20 to 30 seconds of one person talking (a WhatsApp voice note is enough) and it speaks in their voice: Nepali, English, and Nepali mixed with English. Built by [Naamche Labs](https://naamchelabs.com/models/swar).

- **Try it in your browser, no setup:** [Swar Playground](https://naamchelabs.com/playground). Nine ready voices, or clone your own from 30 seconds of speech. Free to try, nothing to install.
- **Run it yourself:** [Colab notebook](https://colab.research.google.com/github/Naamche-Labs/swar/blob/main/notebooks/Swar_Clone_Colab.ipynb) on a free T4 GPU. Upload a clip, or pick a ready-made voice (Maya or Aarav).
- **Weights:** [huggingface.co/Aarjan/swar](https://huggingface.co/Aarjan/swar)
- **Cloning benchmark:** [github.com/Naamche-Labs/nepali-voice-cloning-bench](https://github.com/Naamche-Labs/nepali-voice-cloning-bench)

## Results

**[NepTTS-Bench](https://github.com/Ampixa/neptts-bench)**, the public Nepali TTS benchmark: 205 sentences built to trip a TTS up (minimal pairs, questions, tongue twisters, short phrases), one take per system, scored with the benchmark's own recogniser (Meta MMS). *Pronunciation error* ignores spelling differences Nepali speech does not make (श/ष/स, ि/ी, and so on), so it counts the mistakes a listener hears. Lower is better.

| System | Pronunciation error | Character error |
|---|---|---|
| **Swar** | **2.92%** | **6.19%** |
| OmniVoice (same voice prompt) | 4.32% | 6.68% |
| Indic-Speak Srijana | 8.12% | 10.25% |
| IndicF5 (the model Swar started from) | 9.37% | 12.82% |
| Indic Parler-TTS | 9.51% | 11.70% |
| ElevenLabs v3 | 12.91% | 15.70% |
| Gemini | 15.03% | 15.31% |
| Piper | 17.00% | 18.68% |

Swar rows use the studio voice (AI4Bharat Rasa, CC BY 4.0). With the benchmark's official toolkit (neptts-eval), Swar's Maya voice has the highest SCOREQ naturalness score in the benchmark's published results (4.57; ElevenLabs v3 4.45) and the studio voice the lowest Whisper round-trip error (0.329; Gemini 0.371). Both are automatic scores.

**[Nepali Voice Cloning Benchmark](https://github.com/Naamche-Labs/nepali-voice-cloning-bench)** (our open benchmark: four voices recorded in Nepal by Google FLEURS, one short clip each, 64 distinct prompt + sentence pairs, identical inputs for every system; v1.1 scoring, clean track):

| | Errors (CER) ↓ | Voice match ↑ |
|---|---|---|
| **Swar** | **4.72%** | **0.697** |
| OmniVoice | 5.08% | 0.683 |
| IndicF5 | 8.30% | 0.669 |

**Nepali mixed with English** (30 held-out sentences such as "मेरो laptop को battery…", written before any output was heard): Swar 5.6% character errors, OmniVoice 11.4%, IndicF5 14.6%.

## Checkpoints

| File | Use it for | Notes |
|---|---|---|
| `podvoices_ema.pt` | **The default.** Nepali, and Nepali with English words. | Every number above is this model. |
| `swar_v6_ema.pt` | English sentences. | Trained further with 96 h of English (LibriTTS-R). The cloner uses it automatically for sentences with no Nepali letters when it is present (`F5_EN_CKPT`); everything with Devanagari still goes to the default model, unchanged. Across the nine voices, English wrong sounds 20.4% → 8.6%. On its own it is weaker at pure Nepali (NepTTS-Bench pronunciation error 4.04%), which is why it is not the default. |

## What is here

- `frontend/`: text normalisation and pronunciation: Nepali written the way it is spoken in Nepal, English the way Nepali speakers say it, and `english_fix.py`, a short list of hard English words (tough, Wednesday, honestly…) respelled only where measurement showed it helps.
- `pipeline/`: data preparation and `autoclone.py`, the cloner: it turns a casual voice note into a clean prompt with a transcript from Kriti (our Nepali speech recogniser), makes several takes, and keeps the one whose pronunciation checks out before it looks at voice match. English said in a voice cloned from Nepali speech is paced by syllables, so no words are dropped.
- `tools/`, `ops/`: evaluation and training scripts as they were run. `EXPERIMENTS.md`: the full log, including what did not work.
- `notebooks/`: the Colab cloner. `space_clone/`: the same app as a Hugging Face Space.
- `docs/paper/arxiv/`: paper sources.

## Model

F5-TTS Base initialised from [AI4Bharat IndicF5](https://huggingface.co/ai4bharat/IndicF5) (MIT), trained on open Nepali speech (Rasa, IndicVoices-R, FLEURS, OpenSLR 43 and 54), English (LibriTTS-R) and podcast speech used with the speakers' permission. The ready-made voices are built only from model-made audio; no podcast recording is released.

## Use it responsibly

Only clone a voice with that person's permission. Do not use Swar to impersonate anyone, to deceive, or to commit fraud.

## License

Code: MIT (`LICENSE`). Weights: CC BY-SA 4.0 (see the model card on Hugging Face).
