# Swar: open Nepali voice cloning

Swar speaks Nepali the way it is spoken in Nepal, and clones a voice from 20 to 30 seconds of someone talking (a WhatsApp voice note is enough). It also reads English and Nepali mixed with English. Built by [Naamche Labs](https://github.com/Naamche-Labs).

- **Try it free:** [open the Colab notebook](https://colab.research.google.com/github/Naamche-Labs/swar/blob/main/notebooks/Swar_Clone_Colab.ipynb) (free T4 GPU), drop in a clip or pick one of our voices, Maya or Aarav.
- **Model:** [huggingface.co/Aarjan/swar](https://huggingface.co/Aarjan/swar)
- **Listening test:** [swar-vote.vercel.app](https://swar-vote.vercel.app): can you tell Swar from a real person?
- **Paper:** `docs/paper/arxiv/` (arXiv link coming).

## Results

NepTTS-Bench, 205 sentences, one take, scored with the benchmark's recogniser (MMS-1b-all). Lower is better for the error columns.

| System | Character error | Pronunciation error | UTMOS |
|---|---|---|---|
| Swar, studio voice | **6.19** | **2.92** | 3.82 |
| Swar, Maya voice | 7.45 | 4.52 | **4.09** |
| OmniVoice (same prompt) | 6.68 | 4.32 | 3.93 |
| Indic-Speak Srijana | 10.25 | 8.12 | 3.97 |
| ElevenLabs v3 | 15.70 | 12.91 | |
| Piper | 18.68 | 17.00 | |

Cloning unseen speakers (FLEURS, 82 items): Swar CER 4.55, speaker similarity 0.691, UTMOS 2.67; OmniVoice 5.30, 0.681, 2.49. Mixed Nepali and English (30 held-out sentences): CER 5.6 against 12.1 for OmniVoice. UTMOS and NepaliMOS are automatic predictors, not listeners; the public listening test above is the human check.

## What is here

- `frontend/`: text normalisation and pronunciation layers (Nepal Nepali spellings, held final vowels, English written the way Nepali speakers say it).
- `pipeline/`: data preparation and `autoclone.py`, the cloner: enrolment from casual audio, take selection that checks pronunciation before voice match.
- `tools/`, `ops/`: evaluation and training scripts as they were run. `EXPERIMENTS.md`: the full log, including what did not work.
- `notebooks/`: the Colab cloner. `space_clone/`: the same app as a Hugging Face Space. `site_vote/`: the listening test.
- `docs/paper/arxiv/`: paper sources.

## Model

F5-TTS Base initialised from [IndicF5](https://huggingface.co/ai4bharat/IndicF5) (MIT), trained on open Nepali speech (Rasa, IndicVoices-R, FLEURS, OpenSLR 43/54) plus English (LibriTTS-R) and podcast speech used with the speakers' permission. The preset voices Maya and Aarav are built only from model-made audio; no podcast recording is released.

## Use it responsibly

Only clone a voice with that person's permission. Do not use Swar to impersonate anyone or to deceive.

## License

Code: MIT (`LICENSE`). Model weights: see the model card on Hugging Face.
