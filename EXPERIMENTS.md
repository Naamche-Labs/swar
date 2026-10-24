# Experiment log

Times in IST. Held-out eval = 100 items, 50 unseen speakers (IndicVoices-R, spontaneous speech); one real clip is the voice prompt, two other real sentences are the targets. CER via MMS-1b-all (npi), voice similarity via ECAPA. Real recordings score CER 22.85 / sim 0.686 (ASR struggles with spontaneous speech, so synthetic CER can beat real; sim is the real ceiling).

## 2026-10-19 Pilot: chars, IndicF5 init
F5TTS_Base, lr 3e-5, 38,400 frames/batch, warmup 2k, 18 epochs (~21.9k updates), 129.65 h, bf16, 1x RTX PRO 6000. ~3 h, ~₹650.

| Model | CER % | Sim |
|---|---|---|
| IndicF5 untouched | 15.81 | 0.649 |
| 5k | 13.01 | 0.676 |
| 10k | 12.22 | 0.676 |
| 15k | 12.23 | 0.677 |
| 20k | 12.82 | 0.678 |

Plateau after 10k. Known failure: lists of foreign names (item 022, 76% CER).

## 2026-10-21 Research notes
- IndicF5 paper (arXiv 2505.20693): lr 5e-5, 150k steps, 48k warmup, char vocab; fine-tune on target language only beats mixing English; longer training helps data-poor setups.
- NepTTS-Bench (ampixa): 365 sentences; reports per-file mean CER (MMS, XLS-R, Chirp2). Best MMS CER among baselines: TingTing Subina 0.181. Oshara XTTS-v2 Nepali claims MMS 0.199.
- More open Nepali data: OpenSLR 43 (CC-BY-SA, studio, 18 spk), SLR143, SLR54 (CC-BY-SA, ~157k utts ASR), Common Voice (CC0, small), Chitwan (CC0, 1 h).

## 2026-10-21 Respelling A/B (running)
Same clips and hyperparameters as the pilot, but the text is our G2P output written back as Devanagari (`frontend.g2p.respell`): halanta on every vowelless consonant, one sibilant, no lexicon gemination before medial clusters. All symbols already in IndicF5's vocab (no new embeddings). Checkpoints every 2.5k.

## 2026-10-21 Data v2 prep
- Rasa 1-3 s clips added back (`ops/prep_short.py`, from cached parquet since Rasa is gated): 6,402 clips, 3.5 h, mostly names, brands and short phrases. Manifest `manifest_short.jsonl`.
- IndicVoices Nepali (CC-BY 4.0, same family as IndicVoices-R, raw ASR-grade): 74 parquet shards, 28 GB, downloading. Must drop held-out speakers and eval texts, and dedupe against IndicVoices-R.
- Eval box (L4) set up so scoring never fights training for GPU memory.

## 2026-10-21 Finding: pilot trained on normalized, not spoken, transcripts
IndicVoices-R has `text` (normalized spelling) and `verbatim` (what was said). They differ in 25,585 of 36,124 kept clips (71%): e.g. written बङ्गाल / spoken बेङ्गाल, त्यसको / तेसको, रेसिपी / रिसिपी. The pilot used `text`, so it learned a noisy spelling-to-sound map. `ops/add_verbatim.py` maps clip index -> verbatim (0 index mismatches over 36,124) into `manifest_v.jsonl`.

Our bench scorer reproduces NepTTS-Bench's published MMS numbers (TingTing Subina 18.21 vs 18.09 published mean-file CER; Asmita 23.34 vs 23.37), so head-to-head numbers are comparable.

## 2026-10-21 v2 (running, box 2)
Pilot recipe unchanged (lr 3e-5, 38,400 frames, warmup 2k, 18 epochs), data changed only: verbatim IVR transcripts + all train-speaker Rasa + 6,402 Rasa 1-3 s clips. 59,722 clips, 131.7 h after F5 filtering. `ops/build_v2.py`, `ops/train_v2.sh`.

## 2026-10-21 IndicVoices (raw) prepared
463 h / 1,076 speakers raw; all 50 held-out speakers present (excluded). Filters: any annotation tag, held-out speaker, eval sentence, duplicate of an IVR clip (speaker + normalized text), 3-20 s, SNR >= 15 dB. Kept 106,351 clips, 260.3 h, 1,025 speakers (`ops/prep_indicvoices.py`, `manifest_iv.jsonl`, verbatim text).
Bandwidth check (energy share above 4 kHz): Conversation (57.3 h) is all telephone band (p90 = 0.0) -> drop. Read 23.7 h + Extempore 179.3 h kept for v3. Audio is 16 kHz (no 8-12 kHz band), so v3 needs a quality check (UTMOS now in both evals, `tools/mos.py`).

## 2026-10-21 NepTTS-Bench, first look: short phrases fail
Same MMS scorer for everyone (205 designed sentences; mean-file CER is the bench's own headline metric):

| System | Corpus CER % | Mean-file CER % |
|---|---|---|
| TingTing Subina | 14.44 | 18.21 |
| ElevenLabs | 17.44 | 20.56 |
| TingTing Asmita | 17.47 | 23.34 |
| Edge TTS Sagar | 20.49 | 26.77 |
| ours, pilot chars 10k (flagship ref_04) | 27.18 | 35.21 |

Median file CER for ours is 15.8%; the mean is wrecked by 1-3 word phrases (नमस्ते -> 0.6 s of mush, CER 300%). Cause: F5's `infer_batch_process` slows text under 10 *bytes* to speed 0.3, and a Devanagari letter is 3 bytes, so Nepali short phrases are never slowed. Also the default flagship prompt (ref_04) is an annoyed line. Testing: calm Rasa NEWS/WIKI prompts x a character-based short-text slowdown (`SHORT_CHARS`, `SHORT_SPEED` in `tools/eval_bench.py`).

## 2026-10-21 Bug: stray quotes merged CSV rows
F5's `prepare_csv_wavs.py` reads `audio|text` with Python's csv module, so a `"` in a transcript opens a quoted field that swallows later rows. The pilot's train_chars.csv had 280 quoted rows: that is where its 1,059 "missing" Rasa clips went, and roughly 140 pilot clips were trained with merged, wrong text (0.3%). The respell CSV has no quotes (respell drops them), so the A/B is unaffected apart from that caveat. v2's first start had the same bug (lost 3.8 h); it was killed and restarted on a clean list (`build_v2.py` now strips quotes): v2 = 59,722 clips / 135.54 h, all kept.

## 2026-10-21 v3 (running, H200)
v2 data + IndicVoices Read/Extempore = 138,151 clips / 338.6 h. Same lr/batch, 10 epochs (~32.2k updates), checkpoints every 5k. `ops/train_v3.sh`.

## 2026-10-21 Bench: the prompt voice was the problem
Pilot chars 10k, NepTTS-Bench 205 sentences, same MMS scorer as the baselines:

| Prompt | Corpus CER % | Mean-file CER % | UTMOS |
|---|---|---|---|
| flagship ref_04 (annoyed CONV line) | 27.18 | 35.21 | 1.57 |
| Rasa Male WIKI 6.3 s (calm read) | **8.68** | **11.83** | **3.70** |
| TingTing Subina (best published) | 14.44 | 18.21 | - |
| ElevenLabs v3 | 17.44 | 20.56 | - |

With a calm read prompt the pilot already beats every published system on intelligibility. Held-out (spontaneous prompts) for reference: chars10k 11.93 / 0.678 / UTMOS 2.77; respell 2.5k 13.31 / 0.676 / 2.76 (pilot 5k was 13.01).
Second eval box (L4, 151.185.52.19) added: bench inference sweep with the male_wiki prompt.

## 2026-10-21 v3 diverged (NaN at update 793), restarted with a NaN guard
The first v3 run (H200) hit loss=nan at update 793 and the stock F5 trainer kept stepping on NaN weights for ~16k updates (~2 h of H200 wasted; caught only when its 10k checkpoint scored 100% CER with full-scale noise output). Checks: no non-finite/silent audio in all 138,151 clips; no extreme text/audio ratios. v2 (same recipe, RTX PRO 6000) and respell show no NaN. `ops/patch_nan_guard.py` makes the trainer skip and log (NONFINITE_LOSS / NONFINITE_GRAD) any non-finite step; the run monitor now alarms on any loss=nan. v3 restarted from scratch with the guard.
Held-out: v2 10k 11.90 / 0.679 (pilot 10k 11.93 / 0.678): saturated metric, bench decides.
Bench knob: NFE 64 no gain (12.29 vs 11.83 mean-file at NFE 32).

## 2026-10-21 Results so far, and the H200 instability
- Respell vs chars (same clips): held-out 5k 12.77 vs 13.01, 10k 12.24 vs 11.93. No real difference on held-out; bench pending.
- v2 (verbatim + short clips) vs pilot on bench, paired over 205 sentences: v2 worse on 44, better on 34, mean +2.1 pp, 95% bootstrap CI [-0.4, +4.9]: not significant. Most of the gap is 1-3 word phrases where MMS falls back to Latin letters ("sa may bityo"), so the scorer is noisy there.
- Knobs on bench (male_wiki, pilot 10k): NFE 64 12.29, CFG 2.5 12.31 vs default 11.83: keep defaults.
- v3 on H200 with the guard: non-finite gradients at updates 791, 1125, 1330 (skipped). The same v3 data on an RTX PRO 6000 (box 1, `nepali_v3rtx`) ran 1,000 updates with none, in the same data order: the instability is the H200, not the data. Box 1 first crashed on a full disk (400 GB): deleted full checkpoints already kept as EMA, and HF arrow caches.
- Best-of-3 with an independent reranker ASR (gagan3012/wav2vec2-xlsr-nepali; MMS still scores) running on eval box 2.
- **Respell A/B verdict: chars win.** Bench (male_wiki, short slowdown): respell 10k 8.57 / 11.15 mean-file vs chars 10k 7.94 / 9.98; held-out equal. IndicF5's character model already handles Nepali spelling-to-sound; our G2P respelling adds nothing. Keep chars input.
- Prompt voice x short slowdown (pilot 10k, bench mean-file CER / UTMOS): male_wiki 11.83 / 3.70; +slow 0.5 9.98 / 3.75; +slow 0.3 10.17; **female_news +0.5 8.74 / 3.77** (best); male_news +0.5 11.60 / 3.56. CFG 1.5 11.86 (no gain). Prompt choice is worth ~3 pp: ship with a curated calm read prompt per voice.
- v3 5k (H200): held-out 12.09 / 0.672; bench 10.74 (male_wiki). Pilot 5k/15k/20k being benched for a same-step curve.

## 2026-10-21 Third test: Rasa test split with a human ceiling
`tools/eval_rasa.py`: 200 unseen Rasa test sentences (20 per voice x style: PROPER NOUN, WIKI, NEWS, BOOK, CONV), texts not in training; prompts = the calm bench prompts per voice. The human recordings are scored the same way, so this is the first test with a true ceiling for CER and UTMOS. Also `tools/rescore_bench.py`: XLS-R second-opinion CER on every bench run (biased for best-of-3, which was picked with that ASR).
Same-step bench curve: pilot 5k 10.41 vs v3 5k 10.74 (male_wiki): no difference yet.

## 2026-10-21 H200 is numerically unreliable for this model: dropped
v3 (H200) 15k scored 31.4% on the bench while its held-out score was normal: 47/205 outputs were full-scale noise (rms 1.0). The 15k EMA weights have 0 non-finite tensors, and the same sentences regenerate normally on an L4 (rms 0.06-0.10). Together with ~20 skipped non-finite gradient steps in 6k updates (none on RTX PRO 6000 with the same data order), the H200 (Hopper) kernels are unstable for this model, in training and inference. H200 paused; v3 continues cleanly on RTX (box 1, `nepali_v3rtx`). H200 v3 checkpoints 5k/10k/15k kept on eval box 1 for reference only.

Second-opinion ASR (XLS-R, gagan3012/wav2vec2-xlsr-nepali) on the bench agrees with MMS: ours (pilot 10k, female_news) 9.41 vs ElevenLabs 14.44, Edge Sagar 15.53, TingTing Subina 16.30, Piper 17.75, Gemini 18.12, TingTing Asmita 20.66 (mean-file CER %).

Rasa test ceiling (human recordings): CER 10.15 / mean-file 15.54 / UTMOS 3.72; by style PROPER NOUN 37.4, WIKI 7.5, NEWS 14.9, BOOK 11.7, CONV 6.2. The ASR cannot read names, so proper-noun CER mostly measures the ASR.
Knobs: global speed 0.9 10.34, sway -0.5 13.27 (worse); defaults stay.
- Best-of-3 with the XLS-R reranker: MMS mean-file 10.37 vs 9.98 single take: no gain, dropped.
- Rasa test (pilot 10k, calm prompts): CER 9.58 (human 10.15), mean-file 16.25 (human 15.54), sim 0.755, UTMOS 3.52 (human 3.72). By style mean-file: PROPER NOUN 49.8 (human 37.4), WIKI 7.3 (7.5), NEWS 11.5 (14.9), BOOK 6.5 (11.7), CONV 6.1 (6.2). At or above human everywhere except names and naturalness: **proper nouns are the weak spot.**
- Eval box 2 now runs the Rasa test on every new v3 checkpoint (`ops/rasa_auto.sh`). Listening page built from `ops/make_listen.py`.

## 2026-10-21 Long sentences: F5 never splits Nepali at ।
User report: longer sentences mispronounce Nepali words. Cause found in F5 `utils_infer.chunk_text`: it splits only on `; : , . ! ?` and Chinese punctuation, never on the danda, so a Nepali paragraph is cut only at commas and long sentences reach the model as single 14 s+ generations, past the 5-10 s clips it mostly trained on. Fix: `pipeline/clone.py` `nepali_chunks` (split at । ॥ ? ! ., then , ; : if still long; pack to <= 10 s at the prompt's own pace from `set_pace`) patched into F5 via `patch_f5_chunker()`. Measured with `tools/eval_long.py`: 40 paragraphs of 3 unseen Rasa test sentences, stock vs Nepali chunking.
Also running: `tools/clone_audit.py` (ECAPA + WavLM-SV, real-vs-real ceiling, impostor floor, EER acceptance) and REF_TEXT=asr (prompt transcribed by MMS instead of gold) on held-out.

## 2026-10-21 Cloning audit: at the verifier ceiling
50 unseen speakers, 100 items (`tools/clone_audit.py`, results/clone_audit.json):

| Verifier | real vs real (same person) | clone vs real | impostor | EER | clones accepted (real pairs accepted) |
|---|---|---|---|---|---|
| ECAPA | 0.686 | 0.678 (pilot 10k) | 0.191 | 2.0% | 97% (98%) |
| WavLM-SV | 0.953 | 0.959 (pilot 10k) | 0.778 | 9.0% | 91% (91%) |

v3rtx 5k/10k identical within noise (ECAPA 97/96%, WavLM 90/93%). Zero-shot cloning of unseen voices is as convincing to two independent verifiers as the person's own second recording. Remaining headroom is naturalness and pronunciation, not identity.
Bench at 10k: v3rtx 10.99, v3lr (lr 5e-5) 10.18, v3 H200-trained (regenerated on L4) 11.01, pilot 9.98: the bench no longer separates these models.

## 2026-10-21 More results; polish stage started
- Real-world prompt (REF_TEXT=asr, prompt transcribed by MMS): held-out CER 15.08 vs 11.93 with the gold transcript; similarity unchanged (0.677). F5 leans on an exact prompt transcript: next fix.
- Long text, stock F5 chunking (40 paragraphs, ~25 s each): CER 16.20, UTMOS 3.00 (vs ~8 CER on single sentences).
- v3rtx: 10k Rasa test mean-file 14.93 (pilot 16.25, human 15.54), names 45.7 (49.8), UTMOS 3.59 (3.52); 15k bench 9.73 (best male_wiki score so far); held-out 15k 11.66.
- v3lr (lr 5e-5) 15k bench 11.65 vs 9.73 at lr 3e-5: stopped.
- **Polish** (box 2): from v3rtx 15k EMA (as safetensors; F5's finetune loader needs model_state_dict in .pt files) at lr 1e-5, warmup 300, 8 epochs (~5.9k updates) on read speech only: Rasa all styles + IndicVoices Read, 34,652 clips / 70.6 h. `ops/train_polish.sh`.
- **Long text, Nepali chunking: CER 16.20 -> 6.64** (40 paragraphs, ~25 s, pilot 10k; 1.8 -> 3.1 chunks per paragraph; UTMOS 3.00 -> 2.99). Splitting at the danda is the fix for the user-reported long-sentence mispronunciations. On by default in `pipeline/clone.py`.
- Many residual long-text "errors" are the ASR writing numbers as digits (24 घण्टामा for चौबिस घन्टामा). Scorers now expand digits in the ASR output with our normalizer (`DIGITS_AS_WORDS=1`, default; earlier numbers in this log were scored without it). Rescored: stock 15.70 -> Nepali chunking **5.40**.
- Prompt-transcript ASR on the 50 held-out prompt clips (spontaneous speech): MMS 28.2% CER (best), IndicWav2Vec (sumanpaudel1997) 30.5, Whisper-turbo Nepali 35.4, XLS-R gagan 37.5, kiranpantha whisper-large-v3-nepali 100 (broken output). No drop-in beats MMS. Product fix: an enrollment sentence the user reads (exact transcript); Kriti (our IndicConformer) as fallback for uploaded clips, later (needs the NeMo env).
- Polish 1k/2k: held-out 11.32/11.45, bench 10.07 (start 11.66 / 9.73): flat so far.
- Polish finished (5,928 updates): held-out 1k-4k 11.32/11.45/11.51/11.02; bench 2k 9.61, 3k 10.43 (noise band +-0.5).
- **Names run** (box 2): polish recipe from v3rtx 15k with the 865 Rasa PROPER NOUN train clips repeated 4x (37,247 rows, 73.4 h). Judged on the Rasa test PROPER NOUN column (pilot 49.8, v3rtx 10k 45.7, human 37.4).
- v3rtx finished (32,200 updates): held-out last 11.49, bench last 10.65; 30k bench 10.84. Box 1 paused.
- Names run finished (6.2k updates): held-out 1k-4k 11.52/11.70/11.55/11.48; bench 1k 9.34, 2k 9.82, 3k 10.12, 4k 9.65. Box 2 paused. Rasa-test PROPER NOUN column pending (note: 40 items, v3rtx 5k/10k/15k gave 46.4/45.7/50.3, so +-4 is noise).
- Names run on the Rasa test: PROPER NOUN 50.0 (4k) / 49.3 (6k) vs 45-51 for v3/polish: **oversampling names does not help.** Humans score 37.4 there, so much of the names gap is the ASR.
- Final Rasa table (mean-file / corpus CER, UTMOS): every model 14.2-17.0 / 8.2-9.5 vs human 15.5 / 10.2; UTMOS 3.53-3.62 vs human 3.72.

## 2026-10-21 End of day summary
- **What measurably helped today, all at inference:** calm read prompt (bench mean-file 35.2 -> 11.8), character-based short-text slowdown (11.8 -> 10.0), female_news prompt (8.7), Nepali sentence chunking for long text (paragraph CER 15.7 -> 5.4).
- **What did not:** G2P respelling input, verbatim transcripts, lr 5e-5, NFE 64, CFG 1.5/2.5, sway -0.5, global speed 0.9, best-of-3 reranking, names oversampling. 3.5x more data (v3, 338 h) and the read-speech polish are within noise of the pilot on all three ASR-based tests: these metrics are saturated (at or past the human recordings).
- **Cloning:** two verifiers accept zero-shot clones of unseen speakers at the same rate as genuine recordings (97% ECAPA, 91% WavLM).
- **Open problems:** naturalness (UTMOS 3.55 vs 3.72 human; needs human listening, not ASR), names, and prompts without a transcript (MMS-transcribed prompts cost +3 pp CER; fix = read-aloud enrollment sentence, Kriti as fallback).
- Recommended checkpoint for now: `nepali_names_6000` (bench 9.34 male_wiki, best) or `nepali_polish_4000` (bench 9.59, Rasa corpus CER 8.16 best); all within noise. EMA copies on eval box 1 (/home/ema), full checkpoints on boxes 1/2.
- All GPUs paused. Results JSONs in results/.

## 2026-10-22 The names gap was the scorer
MMS-1b-all's Nepali head shares its vocabulary with Latin letters and switches script mid-word on names and English words ("तानपुरा" -> "tan pूra", "सुप्रिम इन्डस्ट्रिज" -> "suprem industries"), and it does so more on synthetic audio than on the human recordings. `tools/rescore_deva.py` masks the 26 Latin tokens out of the CTC logits so it must spell in Devanagari; same mask for every system and the humans.
Rasa test, Devanagari-only MMS (mean-file CER %; corpus CER in brackets):

| System | All | PROPER NOUN | WIKI | NEWS | BOOK | CONV |
|---|---|---|---|---|---|---|
| Human recordings | 10.93 (8.09) | 22.1 | 6.1 | 8.7 | 11.6 | 6.2 |
| polish_4000 | **8.95 (6.33)** | **21.9** | 5.5 | 7.1 | 5.9 | 4.3 |
| names_6000 | 9.49 (6.87) | 22.2 | 5.9 | 7.9 | 6.6 | 4.8 |
| names_4000 | 9.51 (6.78) | 23.6 | 6.0 | 7.7 | 5.8 | 4.4 |

With a scorer that cannot switch script, our names are at the human level (21.9 vs 22.1) and every other style is at or below the human recordings. The old 45-55 vs 37 "names gap" was MMS writing our names in Latin letters. Names are no longer an open problem by this measure.

NepTTS-Bench with Devanagari-only MMS (205 sentences, mean-file / corpus CER %):

| System | Mean-file | Corpus |
|---|---|---|
| Ours, pilot 10k + female_news + slowdown | **6.14** | 6.34 |
| Ours, names_6000 + male_wiki + slowdown | 6.78 | 5.96 |
| Ours, pilot 10k + male_wiki + slowdown | 7.19 | 6.40 |
| TingTing Subina | 13.39 | 12.08 |
| Gemini | 15.31 | 12.20 |
| ElevenLabs v3 | 15.70 | 13.79 |
| TingTing Asmita | 16.71 | 14.49 |
| Piper | 18.68 | 16.39 |
| Edge TTS Sagar | 19.95 | 16.45 |

Every system gains from the fix (script switching hurts them all), ours most: the gap to the best published system goes from 2.1x to 2.2x on mean-file CER.

Prompt transcription with a KenLM 4-gram (Nepali Wikipedia + Sakonii news/web text, 8.4M sentences, no IndicVoices text): 50 held-out prompt clips, greedy MMS 25.38% CER (with digit expansion) -> best LM 23.40% (alpha 0.3, beta 2.0); higher LM weight hurts (alpha 1.2: 31-32%). The formal written text does not match spontaneous speech well. Finer grid running.

## 2026-10-22 Cloning a real user (Aarjan): prompt choice and pace matter, training does not
User report: the clone "doesn't sound like me", English words garbled, pacing off. 70 s upload (read intro + free talk).
- **Per-speaker fine-tune is not the lever** (and does not scale for an open release): polish_4000 fine-tuned 600 updates on 4 of his pieces (43 s) moved ECAPA similarity to his held-out speech 0.643 -> 0.664. Matches arXiv 2610.19774 (long-prompt TTS: SIM flat from 30 s to 30 min; per-speaker FT ~+0.01).
- **Pace**: F5 copies the prompt's pace. His read intro was 7.1 letters/s, his free talk 9.7-11.6, so clones from the intro were slow (7.5-8.3).
- **Prompt choice**: the read intro was his least typical piece (ECAPA to the upload's mean voice 0.815 vs 0.905-0.939 for free talk).
- `pipeline/clone.py prepare_voice` (`--upload`): split the upload at pauses, prompt = piece closest to the upload's mean voice, speed = upload pace / prompt pace. On 6 test lines vs the read-intro prompt: pace 7.5-8.3 -> 9.9-11.3 (his average 10.1), similarity to held-out pieces higher on 5/6 (mean ~0.65 -> ~0.70), CER similar.
- **English words**: `frontend/english.py` (loanword list, all-caps letter names, CMUdict -> Nepali sounds, romanized-Nepali fallback) wired into `normalize`. Code-switch line CER 37 -> 8-14.
- Accent stage running: polish_4000 + OpenSLR 54/43 (Nepal-recorded, 70.7 h, 545 speakers) + Rasa (43 h), 5 epochs.

## 2026-10-22 Voice match without per-user training (tools/voice_match.py)
3 FLEURS Nepal speakers (>= 8 items), upload = 5 of their recordings joined (~50 s), 24 target sentences vs their real recordings. polish_4000:

| Variant | CER % | ECAPA sim to real | pitch level diff (st) | pitch movement ratio | UTMOS |
|---|---|---|---|---|---|
| auto (prepare_voice) | **6.8** | 0.676 | 3.92 | 1.29 | 2.28 |
| best of 3 by voice | 7.7 | **0.685** | 2.89 | 2.07 | 2.34 |
| 20 s prompt (2 pieces) | 40.8 | 0.647 | 4.14 | 1.66 | 2.20 |
| kNN-VC on top, whole upload | 14.1 | 0.610 | 3.14 | 0.97 | 2.10 |

Long prompts break F5 (outside training), kNN-VC lowers similarity and intelligibility, best-of-3 is +0.009 sim for 3x compute: none adopted. Pitch: clones sit 2.23 st from the upload's median pitch, real sentences 2.35 st: pitch level is copied. pyin is unreliable on FLEURS audio (octave jumps, movement ratios 0.5-3.5); a neural pitch tracker is needed for per-speaker pitch claims.

## 2026-10-22 English inside Nepali, OmniVoice head-to-head, duration units
- **Code-switch** (`tools/eval_codeswitch.py`, 30 held-out mixed sentences written before any output was heard; female_news prompt, polish_4000): English in Latin letters (old) CER 43.94 -> `frontend/english.py` 6.42, UTMOS 3.73 -> 3.70. English spelling = the Devanagari form Nepali writers use most (word counts over 8.4M sentences, data/ne_wordfreq.tsv.gz) among CMUdict-derived candidates; loanword list for collisions with native words (plan/प्लेन, shirt/शर्त).
- **OmniVoice** (k2-fsa, 0.6B, 581k h, 600+ languages incl. npi; weights not commercially licensed), same prompts and scorers:

| Test | OmniVoice | Ours (polish_4000 / pilot+female_news) |
|---|---|---|
| NepTTS-Bench Devanagari-MMS mean-file / corpus CER | 6.68 / 6.31 | 6.14 / 6.34 |
| FLEURS (4 Nepal speakers, 82 items) CER | 5.30 | 5.21 |
| FLEURS ECAPA sim to real | 0.681 | 0.695 |
| FLEURS UTMOS | 2.49 | 2.33 |

A tie on pronunciation, ours slightly ahead on voice similarity, OmniVoice ahead on predicted naturalness. Naturalness is the gap to close.
- **Duration units** (100 held-out pairs, predicted target duration = prompt duration x unit ratio): UTF-8 bytes (F5) 28% off, chars 28%, letters 28%, G2P phonemes 29%, syllables 30%. The error is speaking-rate variation, not the unit: F5's rule stays.
- Accent stage finished (8,580 updates, 0 non-finite steps); evaluation running on box 1.

## 2026-10-22 Official NepTTS-Bench naturalness predictor (NepaliMOS): a tie
`neptts-eval` (Ampixa/neptts-bench) public NepaliMOS checkpoint, same checkpoint for every system (205 bench sentences):
ours (pilot 10k, female_news) 2.94, ElevenLabs 2.95, OmniVoice 2.98, TingTing Subina 2.99, Piper 3.05. The leaderboard's 3.4-3.5 values come from the authors' full (unreleased) predictor; with the public one all five are within 0.11 (the bench paper treats such gaps as ties). SCOREQ: ours 4.16 (ElevenLabs 4.45 on the leaderboard). Getting the eval running needed a torchaudio shim (s3prl imports removed APIs; sitecustomize in venv_ne).
Naturalness cannot be settled by predictors: next step is blind human ratings.

## 2026-10-22 Accent stage (OpenSLR 54/43 Nepal speech + Rasa) results
polish_4000 + 8,580 updates (5 epochs, lr 1e-5) on 82,496 Nepal-recorded clips (70.7 h, 545 speakers) + Rasa (43 h). Box 1, same hardware for the comparison:

| Model | FLEURS CER / sim / UTMOS | Bench (Deva MMS) mean-file | Rasa (Deva) mean-file / names |
|---|---|---|---|
| polish_4000 | 5.30 / 0.697 / 2.36 | - | 8.95 / 21.9 |
| accent_4000 | 5.30 / 0.697 / 2.37 | 6.73 | 9.19 / 23.5 |
| accent_8000 | **4.80** / 0.697 / 2.38 | 6.48 | 9.28 / 23.6 |

Nepal-speaker intelligibility -0.5 pp; similarity unchanged; Indian-studio Rasa +0.3 pp. A small, real shift toward Nepal speech, not a large accent change: zero-shot accent comes mostly from the prompt.

## 2026-10-22 The naturalness bottleneck is the mel representation (Vocos), not the model
Copy-synthesis of 40 real held-out Rasa test clips (UTMOS): original 3.88, Vocos (F5's mel + vocoder) 3.14, BigVGAN-v2 24 kHz 100-band on its own mel 3.80. Vocos-mel -> BigVGAN bridges: filterbank pinv 3.14 (same as Vocos), least squares 1.27, learned 2.6M conv bridge 2.82. The information is missing from the Vocos mel itself (HTK filterbank, no norm), so no vocoder swap can recover it: the TTS has to predict BigVGAN's mel. `ops/train_bigvgan.sh`: accent_8000 fine-tuned on BigVGAN mel targets (F5 supports it), lr 3e-5, 3 epochs. Loss 2.0 -> 0.6 in 800 updates; at 1k updates UTMOS on 3 sentences 2.22 vs 3.66 (Vocos model): still adapting.

## 2026-10-22 Compute: steps and CPU
accent_8000, NepTTS-Bench female_news, Devanagari MMS mean-file CER / UTMOS: NFE 32 6.48 / -, NFE 16 6.07 / 3.80, NFE 8 7.32 / 3.81. NFE 16 is free; NFE 8 costs ~1 pp.
CPU only (8 threads, server CPU, fp32), 9.3 s of speech: NFE 32 68.8 s (RTF 7.4), NFE 16 34.4 s (RTF 3.7), NFE 8 17.3 s (RTF 1.9). Not real time on CPU yet; next: Apple-silicon GPU (MPS), half precision / ONNX, few-step distillation.

## 2026-10-22 BigVGAN-mel stage: stopped
accent_8000 fine-tuned 5,475 updates on BigVGAN-v2 mel targets (lr 3e-5). UTMOS on 40 bench sentences (NFE 16): 2k 2.21, 4k 2.46, last 2.56 vs the Vocos model 3.80; bench Devanagari CER 9.43 (4k) vs 6.07. Not clipping (output -6/-12 dB: 2.54). F5's BigVGAN copy-synthesis path is fine (3.79 vs 3.85 originals). The model learns the wider-range mel slowly; matching the Vocos model would take an F5-scale run. And our Vocos TTS output already scores UTMOS 3.80 (real recordings 3.88, OmniVoice 3.93): audio fidelity is not where we lose. Stopped; prosody next.

## 2026-10-22 First human preference test (1 native rater, 51 blind pairs, accent_8000 audio made without the frontend)
Ours preferred vs ElevenLabs 71% (same 6%, theirs 24%), vs TingTing Subina 53/0/47, vs OmniVoice 18/35/47. Losses: questions and emotion (sent_116, sent_120), news headlines (171, 172), and the English-mixed line, whose audio had been made without our frontend (bench runs fed raw text; fixed: FRONTEND=1 default in eval_bench). Mispronunciation flags: ours 4, TingTing 2, OmniVoice 1. Rating page v2 adds NepTTS-Bench-style 1-5 MOS with a human-recording anchor.

## 2026-10-22 Human MOS (1 native rater, NepTTS-Bench-style 1-5, 17 sentences per system, loudness-matched)
Real person (Rasa recordings) 3.75 [3.20-4.30], **ours (accent_8000, frontend on) 3.12** [2.75-3.49], OmniVoice 3.06, Piper 3.06, TingTing Subina 2.82, ElevenLabs v3 2.76. Gap to human: 0.63.

## 2026-10-22 Prosody and emotion
- Question intonation (`tools/question_rise.py`, final-quarter F0 minus clip median, NepTTS-Bench question vs statement sentences): ours questions -0.39 st / statements -2.94 (contrast +2.54); OmniVoice +0.27 / -1.35 (+1.62); TingTing Subina -1.10 / -2.96 (+1.85); ElevenLabs -4.92 / -6.02 (+1.10). We separate questions from statements most, but our questions do not rise at the end.
- **Training transcripts have no punctuation**: IndicVoices-R and IndicVoices (140k clips) 0% "?" / "।" / ","; ~30% contain a question word. Only Rasa (19k) is punctuated (8.5% "?", 86% "।"). `tools/restore_punct.py`: MMS forced alignment -> commas at pauses >= 0.30 s, danda at >= 0.65 s after a final verb, "?" at the end when a question word co-occurs with a final F0 rise >= 1 st. Running on IndicVoices-R (4 shards).
- **Emotion presets** (`ops/emotion_presets.py`): Rasa has both voices in HAPPY/SAD/ANGER/FEAR/SURPRISE/DISGUST/CONV/NEWS; 14 presets (/home/data/ref/emotion). emotion2vec+ large, probability of the intended emotion (2 lines per emotion, so indicative): neutral prompt 0.20, emotion preset 0.42, real Rasa emotional recordings 0.34. Surprise 0.01 -> 1.00, anger 0.49 -> 0.58; sad/fear are barely recognised even on real recordings (0.16/0.08).
- Best-of-4 seeds (NFE 16, = cost of 2 x NFE 32) picked by per-file NepaliMOS: UTMOS 3.77 vs 3.80 single take, CER 5.90 vs 6.07: per-utterance NepaliMOS is too weak a picker. Rejected.

## 2026-10-22 Clone vs real studio recording (tools/match_test.py), and preset pace
Rasa test, 160 unseen sentences (no proper nouns), the studio speakers' real recordings vs clones of the same sentence from the calm presets. MFCC-DTW timbre distance (relative index, not literature MCD), log-F0 correlation and RMSE along the DTW path, pace (speech duration ratio), ECAPA sim, UTMOS.

| System | timbre dist | F0 corr | F0 RMSE st | pace off | sim | UTMOS (F / M) |
|---|---|---|---|---|---|---|
| real | 0 | 1 | 0 | 0% | 1 | 3.68 (3.76 / 3.60) |
| real through Vocos (pipeline ceiling) | 6.1 | 0.995 | 0.32 | 0% | 0.987 | **3.06** (3.00 / 3.11) |
| accent_8000 clone | 42.0 | 0.455 | 5.38 | 14% | 0.781 | 3.44 (3.43 / 3.44) |
| + preset pace calibration, NFE 16 | 41.7 | **0.493** | **5.02** | 13% | 0.780 | **3.54 (3.60 / 3.49)** |

The user hears the clones as artificial, even the male voice; the Vocos round trip alone loses 0.6 UTMOS on real speech. Female clones were 12% slower than the real speaker (male 2% faster): the female_news preset reads at 8.0 letters/s vs the speaker's median 9.33 (male_wiki 9.53 vs 10.28). refs.json now carries "speed" = speaker median / preset pace (female 1.166, male 1.078). Rasa (Deva) mean-file CER 9.28 -> 8.77. Adopted.
Box 1 disk filled (400 GB) and killed the punct stage at update 2000; cleaned (spk_aarjan, nepali_bigvgan, v3rtx, old accent checkpoints) and resumed from model_last; keep_last_n 3.

## 2026-10-22 Vocoder: the user hears "AI distortion, vibration, music" on the Vocos round trip
Plan agreed: vocoder first (until a real recording round-trips unchanged), then the acoustic model.
BigVGAN-v2 (24 kHz, 100 band) fine-tuned with F5/Vocos mels as generator input (ops/patch_bigvgan.py; pretrained G+D, lr 1.35e-5 inherited, batch 4, Rasa + 15k IndicVoices-R). Copy-synthesis UTMOS on 60 Rasa test clips (real 3.79, Vocos 3.07): step 2k 2.10, 4k 2.80 (timbre distance 16.0 -> 12.3, Vocos 6.5). Rising; continuing.

## 2026-10-22 Nepal pronunciation layer
User: "दशचोटि is wrong". In Nepal श/ष/स are all "s", व starts words as "b", क्ष is "chhy", ज्ञ "gy", ण "n"; the IndicVoices speakers (India) use "sh"/"v". `frontend/pron.nepal_accent` respells the model input (not reference texts): दशचोटि -> दसचोटि, विशेष -> बिसेस, शिक्षा -> सिछ्या, सर्वोच्च -> सर्बोच्च, कृषि -> क्रिसि (श्र kept). On in clone.py (NEPAL_ACCENT=1); eval TEXT_MODE=nepal. A/B clips for the user on the match page.

## 2026-10-22 Punctuation stage: rejected
accent_8000 + 10,148 updates (lr 1e-5) on 203 h with punctuation restored from audio in IndicVoices-R (548 "?" of 34k clips, commas at pauses >= 0.30 s), Rasa, OpenSLR with sentence ends. vs accent_8000: bench Deva CER 6.75 vs 6.07; FLEURS 5.06 / 0.702 vs 4.80 / 0.697; code-switch 6.24 / UTMOS 3.78 vs 6.49 / 3.74; clone-vs-real F0 corr 0.491 vs 0.493, UTMOS 3.54 vs 3.54; question final rise -1.80 st vs -0.39 (contrast with statements +1.22 vs +2.54): questions got flatter. Commas at hesitation pauses likely taught the wrong prosody. Not adopted; accent_8000 stays.

## 2026-10-22 English layer v2 and mined fixes
- User: "cafe" read like "coffee", "awesome" as "aasam". Causes: CMU's American "cuh-FAY" (कफे) and the frequency picker choosing असम/आसाम (the place) for awesome.
- Lookup order now: data/loanwords.tsv (curated, ~330 everyday words as said in Nepal) > built-in list > Google's Nepali lexicon Latin entries (5.6k, pronunciation as spoken in Nepal with syllable breaks, converted to Devanagari) > all-caps letters > CMUdict candidates picked by corpus frequency, never a word in the native Nepali lexicon > romanized-Nepali rules.
- Halanta follows syllables: inside onsets/codas yes, across a syllable break only after nasals, liquids and s (कम्प्लीट, इन्टरनेट, मोस्टली vs फिडब्याक, ल्यापटप, डेडलाइन).
- Held-out code-switch CER 6.49 (new spellings as reference), UTMOS 3.74.
- tools/mine_hard_words.py on 1,000 FLEURS sentences (ours vs human, Devanagari MMS): word error ours 13.2%, human 16.9%; 29 hard words, 6 respellings that the model reads back correctly (data/pron_fixes.json: सुझाव -> सुझाब, न्यूनतम -> न्युनतम्, ...). Order in clone.to_model_text: fixes -> Nepal pronunciation -> English.
- Vocoder fine-tune copy-synthesis UTMOS: 6k 3.05, **8k 3.18 (passes Vocos 3.07)**, timbre distance 10.1 (Vocos 6.5). Match page now has "real through the new vocoder" and "clone, new vocoder" columns (g_05008000).
- Vocoder 10k 3.18 / timbre 9.49, 12k 3.29 / 9.14. User: the new vocoder changes the voice's character (sounds more mature/serious or younger). Matches the timbre distance (9.1 vs Vocos 6.5). Restarted from 12k with lambda_melloss 15 -> 45 to anchor the spectral envelope.
- Aarjan clone v4 (accent_8000, prepare_voice on the 70 s upload, all text layers, NFE 16), same mel decoded by Vocos and by BigVGAN 12k, on the match page.

## 2026-10-22 Per-preset speaker fine-tunes: no gain
accent_8000 + 12 epochs (lr 1e-5) on one Rasa speaker each (Female ~2.6k updates, Male similar). Clone vs real (pace-calibrated): female items F timbre 47.73 / UTMOS 3.56 (base 47.85 / 3.60); male items 35.60 / 3.52 (base 35.63 / 3.49); F0 corr 0.485/0.479 vs 0.493. The base model already knows these speakers; the remaining distance is the acoustic model's generation, not speaker familiarity. (Caveat: the timbre/F0 distance to a real take of the same sentence includes legitimate prosody differences; there is no second human take to give a human-vs-human floor.)

## 2026-10-22 Speaker B (29 s WhatsApp voice note) clone
Opus 20 kbps, <0.2% of energy above 4 kHz. resemble-enhance (CPU; its torch has no Blackwell kernels) full enhancement: UTMOS of the note 1.86 -> 2.27. Clones (accent_8000, prepare_voice, all text layers, NFE 32, Vocos), 9 lines: from the enhanced note sim 0.679 to his other real piece / UTMOS 3.47; from the note as sent 0.624 / 3.20; his two real pieces 0.752. Page: Speaker B Voice Clone artifact (private).
Vocoder (lambda_melloss 45 from 12k): 14k UTMOS 3.34, timbre distance 8.07 (12k: 3.29 / 9.14).
- Speaker B v2 (ops/gen_Speaker B2.py): per line, 18-36 candidates (2 prompt pieces x note as sent / resemble-enhance light / full x CFG 2.0/2.5 x 3 seeds; English lines also on F5-TTS v1 Base with a romanized prompt transcript), keep the highest ECAPA similarity to the piece not used as prompt (English: Whisper WER <= 0.15 first). Best per line 0.648-0.773 (mean single take 0.62-0.63; his real pieces 0.752). English-only lines: F5 v1 wins on similarity (0.736-0.773) at WER 0-0.08. F5-TTS pretrained weights are CC-BY-NC (Emilia data): English routing is demo-only until a commercial English path exists. Disk on box 2 filled (old polish/names checkpoints removed).
- Speaker B v3: user found clones too deep and too fast. Pitch matches (his note median 106 Hz; clones 102-110 Hz; brightness similar) and clones carry LESS energy under 150 Hz than his note (12-27% vs 32-40%), so "deep" is not pitch or bass. Pace was the real gap: his pieces 9.0 / 12.7 letters/s (mean 10.9), clones 11.1-18.5 because the best-matching prompt piece is his faster one. Re-rendered winners with speed = 10.9 / clone rate (0.78-0.95), English x0.93: 12.5-13.7 letters/s (within his range). The meta speed "set from the whole upload" should be applied after prompt choice in clone.prepare_voice too (it is: speed = upload/prompt pace), but the per-line search picked the faster piece without it.

## 2026-10-22 Taking the NepTTS-Bench error down
- Where our 6.1% comes from (old run): 41% on sentences where every system fails (spelling variants, spacing, ASR limits), 20% on sentences another system gets right.
- `tools/pron_cer.py`: pronunciation CER, standard CER after folding spellings Nepali speech does not distinguish (श/ष/स, ी/ि, ू/ु, ँ/ं, nasal+halanta/anusvara, व/ब, ण/न, ृ/रि, spacing, word-final halanta) and converting English words in references to Devanagari with frontend.english, identically for every system.
- Full current pipeline (fixes -> Nepal pronunciation -> English layer, pace-calibrated female_news, NFE 16), accent_8000: standard 6.15 (old 6.14), **pronunciation 3.13 (old 3.98)**; TingTing Subina 13.39 / 11.25, ElevenLabs 15.70 / 12.91, Gemini 15.31 / 15.03, Piper 18.68 / 17.00. Standard CER does not move because Nepal pronunciation is written स for श by the ASR.
- Best-of-4 by an independent ASR (XLS-R): 6.23 / 3.51, worse: rejected again.
- Per category (pronunciation CER): ours minimal pairs 3.0 (others 10.6-16.6), robustness 18.3 (others 27.5-42.1, pilot 14.5) is the largest remaining block (tongue twisters); real slips left: ड/ढ, त/थ, ध/द, gemination (कत्था), and sent_174 where every take is heard as "339" for उनन्तिस.

## 2026-10-22 Speaker A (8 WhatsApp notes, 285 s, consent given) and a local page
Two speakers among the 11 notes (ECAPA: within group A 0.72-0.95, across 0.52-0.71); group A cloned. Fast casual speech: the two ASRs agree on only 0-71% of each piece, so prompts are the 3 best-agreeing pieces with hand-corrected transcripts; long talk without pauses is split at the quietest 10 ms frame (pieces <= 10 s). resemble-enhance on CPU is 7.5 min per note, so only prompts were enhanced. Search: 3 prompts x {as sent, enhanced} x CFG {2, 2.5} x 2 seeds (English also on F5 v1), her median pace 11.5 letters/s. Best per line: 0.72-0.84 ECAPA to the centroid of her other pieces (her own pieces vs the rest: 0.859); every winner used a prompt as sent, none enhanced. Local self-contained HTML (samples/voice_new/voice_clone.html, not in git). Third-party voice samples are now git-ignored.

## 2026-10-22 Automatic cloner (no hand steps) and the replicate test
User: "can a normal person drag and drop a 10 s to 5 min clip and get this?" and "make the clone say exactly what she says, that is how we find out if it is right".
- `pipeline/autoclone.py` (enroll / say) does every step the speaker A clone needed by hand: cut uploads at pauses (3.5-10 s, quietest-frame split for nonstop talk), keep one speaker (ECAPA average-linkage clusters at 0.62, the cluster with most speech), transcribe, pick 3 prompts (typical voice + transcript confidence, one per upload first), set her median pace, and per line pick the best of N takes.
- Take choice is now pronunciation first, voice second: each take is read back by Devanagari MMS; takes within 3 points of the best pronunciation CER compete on ECAPA similarity. User heard "खाना" said as "खान" in the speaker A np_story clip, which was chosen on similarity alone; the text going in was right (खाना).
- Transcripts: Kriti Telephony (Naamche, Aarjan/kriti-telephony, NeMo, /home/venv_nemo) on her WhatsApp notes is close to the hand transcripts where MMS and Whisper are garbled (piece 004: Kriti "यसपालि त यो सिनेरीमा तेरो गल्ती थिएन लज्जै थियो थियो हैन" vs hand "यसपालि त यो सिनेरियोमा तेरो गल्ती थिएन, लज्जा थियो थियो हैन"; MMS "यसपालित यो सिनेरेमा तेरो बल्तै थिरेन"). It is now the primary transcript; MMS (and IndicConformer when its files are accessible: still 403 for this token) cross-check it as the confidence. IndicConformer hook in tools/asr_conformer.py (/home/venv_conf, onnxruntime CPU).
- `app/cloner.py`: Gradio drag-and-drop page (clips in, consent box, text, Fast/Good/Best = 1/4/12 takes).
- Replicate test (ops/replicate_A.py, ops/replicate_pace.py): all 11 notes in as received. Speaker step kept 36 of 42 pieces and dropped the 2 notes from the other person by itself. 6 held-out pieces (one per note, most confident transcripts, never prompts), clone says the same words, best of 12. Voice match clone vs that real clip: mean 0.767 (her held-out clips vs each other: 0.771); pitch real/clone 262/270 Hz; ASR-misheard letters her 23% / clone 5% (her casual speech is harder to recognise than the clone). Pace: the cloner uses her median 12.6 letters/s but her clips run 9.7-15.8; rendering each clone at its own clip's pace matched within 3% (similarity 0.746, about the same).
- Speaker A np_story re-rendered with the pronunciation-first pick: "खाना खान" read back correctly, similarity 0.836 (was 0.824 with the swallowed syllable).
- Local pages (not in git): samples/voice_new/replicate/replicate.html, samples/voice_new/voice_clone.html.

## 2026-10-22 Competitors: Indic-Speak and Indic Parler-TTS
Two open Indic TTS models on NepTTS-Bench (the 205 scored sentences; natural_speech skipped as for every system), scored exactly like our full_s0 row: Devanagari-only MMS CER (`tools/rescore_deva.py bench NAME`, mean per file), pronunciation CER (`tools/pron_cer.py`), and mean UTMOS (`tools/utmos_runs.py`, new; UTMOS22 strong at 16 kHz). Per-category summary: `results/deva/bench_competitors_2026-10-22.json` (`tools/bench_summary.py`).

| System | Standard CER | Pronunciation CER | UTMOS | n |
| --- | --- | --- | --- | --- |
| Ours, full_s0 (female_news prompt) | **6.15** | **3.13** | 3.81 | 205 |
| Indic-Speak, Srijana (female preset) | 10.25 | 8.12 | **3.97** | 205 |
| Indic-Speak, Sagar (male preset) | 12.37 | 10.53 | 3.81 | 205 |
| Indic Parler-TTS, Amrita | 11.70 | 9.51 | 3.26 | 205 |
| (earlier) OmniVoice, female_news | 6.68 | | | |
| (earlier) TingTing Subina | 13.39 | 11.25 | | |
| (earlier) ElevenLabs v3 | 15.70 | 12.91 | | |

- Indic-Speak (bodhan-ai/indic-speak, Llama-3.2-3B speech LM + SNAC codes + Vocos, Indic Open Model License v1.0, Nepali is a "preview" language). Its model card says there is no voice cloning, so it could not be run with our female_news prompt; only its two Nepali preset voices. `tools/gen_indic_speak.py` uses the repo's own inference.py with the card's defaults (temperature 0.6, top_p 0.9, top_k 50), seed 0, Devanagari text as given. Venv /home/venv_ispeak reuses /home/venv's torch 2.11 and transformers 5.8.1 via a .pth file, plus `snac`. No sentence failed to produce audio. Sagar ran away on 6 sentences (sent_042, 047, 050, 103, 126, 172 hit the 2520-token limit, 30.7 s of audio for a short line); those count as errors in its CER, which is the model's real behaviour.
- Indic Parler-TTS (ai4bharat/indic-parler-tts, Apache-2.0), speaker Amrita (its only Nepali voice), description "moderate pace, clear, natural and expressive, very high quality, close up", seed 0. `tools/gen_indic_parler.py` in /home/venv_parler (transformers 4.46.1, torch 2.8). The earlier 401 was only the missing HF_TOKEN.
- Per category (pronunciation CER, ours / Indic-Speak Srijana / Indic-Speak Sagar / Parler): minimal pairs (149) 3.0 / 6.8 / 10.6 / 7.7; robustness (10) 6.4 / 17.2 / 22.3 / 20.3; question intonation 1.7 / 12.4 / 7.4 / 22.0; emotion 3.2 / 12.7 / 13.1 / 9.3; phrases 2.7 / 15.7 / 7.3 / 15.7. Ours is lowest in every category on pronunciation CER except newspaper ambiguities (only 4 sentences), where Indic-Speak Sagar is slightly lower (4.3 vs 4.8).
- Takeaway: both are clearly better than TingTing and ElevenLabs on this bench, but still make 2.6x (Srijana) to 3.4x (Sagar) our pronunciation errors. Indic-Speak Srijana scores higher UTMOS than us (3.97 vs 3.81), so on clean-sounding audio it is the one to watch; Parler sounds noticeably worse by UTMOS (3.26). UTMOS is an English-trained acoustic score, not a Nepali listener test.

## 2026-10-22 Speaker S (3 WhatsApp notes, 155 s, months apart) through the automatic cloner
- Speaker step wrongly dropped 2 of his 3 notes: his notes score 0.51-0.60 against each other (within a note 0.67-0.74), the same range as two different people in speaker A's set (0.52-0.71). Channel and room differences look like a different voice to ECAPA. Fix: `one_speaker` / ONE_SPEAKER=1 and an app checkbox "every clip is this one person"; the automatic filter stays the default and the app says what it left out.
- Kriti is now the transcriber and the take checker everywhere in the product path (persistent worker, tools/asr_kriti.py --serve). MMS-1b-all (CC-BY-NC) and Whisper are out of the product path; MMS stays in research evals. Kriti read his real held-out clips back at 1-5% CER (MMS on speaker A's: ~23%).
- Replicate (4 held out, all notes kept): clone vs his real clip 0.598, his clips vs each other 0.630 (speaker A: 0.767 vs 0.771). Hardest pair 0.47: a note whose recording differs from the prompts. Pace-matched pass 0.636.
- Time per sentence, 12 takes, NFE 16, GPU shared with vocoder training and two other jobs: 8-25 s (first call 40-50 s with model loading).
- Essay (12 sentences, 78 s, best of 8): sim 0.727 mean. 10 short lines (Nepali, English, mixed; all on our model): mean 0.664.
- 30 s upload test (speaker A, 3 windows clear of the held-out pieces, best of 12, same 6 held-out texts): clone vs her real clip 0.738 / 0.696 / 0.620 vs 0.767 from the full 5 min. A 30 s clip works but quality depends on the clip: the 0.620 window was slow, careful talk (9.1 letters/s vs her usual 12.6).

## 2026-10-23 Preset v3 (bilingual base + studio speaker + 1500 English clips, 40 epochs)
| model | Rasa CER | mean-file | UTMOS | match F0corr | match sim | match UTMOS |
|---|---|---|---|---|---|---|
| accent8k (base) | 8.23 | 14.78 | 3.62 | 0.493 | 0.780 | 3.54 |
| preset2_female | 8.16 | 14.46 | 3.67 | 0.442 | 0.778 | 3.60 |
| preset3_female | 8.13 | 15.28 | 3.67 | 0.472 | 0.778 | 3.62 |
| preset2_male | 8.08 | 14.59 | 3.64 | 0.418 | 0.771 | 3.57 |
| preset3_male | 8.60 | 15.79 | 3.61 | 0.421 | 0.771 | 3.53 |
Read: female v3 is about level with v2 (better pitch match, worse long-file CER). Male v3 is slightly worse than v2.
Full studio fine-tunes have reached a plateau: none beats the base on pitch correlation. The next gains must come from
prompts and selection, not more epochs. Ear check: ops/female_demo.py with DEMO_REF and DEMO_RUNS (v2 vs v3).

## 2026-10-23 CPU speed (is the free Hugging Face CPU Space usable for cloning?)
F5 Base size model, 2 CPU threads (like the free Space's 2 vCPU), Mac: 8.5 s of audio took 115.6 s at NFE 32 and 57.9 s
at NFE 16, about 14x and 7x slower than real time. One take per sentence, before speaker-check or read-back.
Verdict: the free CPU tier only works as a toy (about 1 min per sentence, one user at a time). The cloner needs ZeroGPU
(HF Pro) or a paid GPU. The voting site is fine on free CPU (it only plays clips).

## 2026-10-23 Hugging Face hosting needs Pro
Creating a Gradio Space on account Aarjan (free plan) returned 402: "Static Spaces are free for everyone, but hosting
Gradio and Docker Spaces on free cpu-basic requires a PRO subscription." So the voting site (space_vote/) and the
cloner both need HF Pro (~$9/mo), which also unlocks ZeroGPU for the cloner. Without Pro, the voting site must be
rewritten as a static page plus an outside vote store.

## 2026-10-23 Can Kriti and F5 share one process (needed for ZeroGPU)?
Not as is. Kriti needs the AI4Bharat NeMo fork (multilingual tokenizer, language_id) with old pins (numpy 1.26,
huggingface-hub 0.23). Stock nemo_toolkit 2.4 imports beside F5-TTS (torch 2.7.1, numpy 2.2), but restoring
kriti_telephony.nemo fails with KeyError 'dir' (the fork's tokenizer config). Options for the Space: (a) port the fork's
multilingual tokenizer loader onto stock NeMo; (b) export Kriti's CTC head to ONNX and run it without NeMo;
(c) ask the uploader to type or confirm the transcript. (b) is the cleanest; it needs one GPU-box session to export.
Done (b): ops/export_kriti_onnx.py exports Kriti's CTC head to ONNX (493 MB) on CPU; pipeline/kriti_onnx.py runs it
with a numpy log-mel frontend and no NeMo. Same process as F5-TTS works. Transcripts are identical to NeMo's CTC on the
5 clips tested (the box results for the v2 demo lines, word for word). Mac CPU ~0.5 s per clip. Note it is the CTC
head; the product's read-back used RNNT text (very close: बलुका vs बेलुका was the only difference on the test line).

## 2026-10-23 Cloner under Space conditions (L4, one process, no NeMo, Kriti ONNX)
30.6 s upload (podcast host A, 4 pieces) -> voice in 23.5 s. Lines, best of 4 takes: story 10.2 s (sim 0.74, CER 5.7),
आजको/काठमाडौँ line 9.7 s (sim 0.74, CER 4.7), mixed office line 11.1 s (sim 0.80, CER 4.3); 1 take: 2.5-6.5 s.
Pure English from a Nepali-only 30 s clip: no English prompt found, Nepali-letter path, CER 21 ("see you tomorrow" weak).
So a ZeroGPU Space is workable: about 10 s per sentence at best of 4. Clips: ~/Downloads/space_selftest/ (not in git).

## 2026-10-23 v4 listening fixes (user ear check of preset v2/v3)
User: female v3 good; male v3 voice itself disliked (the speaker, not the training); full English "very broken"; pace
like a newsreader; no pause after a full stop ("...the show. Today..."); "customer" wrong in both voices.
- Cause of two of these: ops/female_demo.py made each line in one generation (F5 runs through a full stop) and wrote
  English in Nepali letters. The cloner (autoclone.say_long) already split sentences; the demo did not.
- `ops/v4_lab.py` (L4, nepali_bi_ema; preset3 is on box 2, no RTX PRO free all evening): one sentence at a time,
  0.38 s after । and ., 0.45 after ?. Female prompts: news (old) vs two Rasa CONV clips.
- Full English, 2 lines x 2 prompts, Whisper WER of the best of 4: Nepali-letter English 0.0; English letters with the
  Nepali prompt 45-76 (garbage: "Chinese, Japanese, Chinese"); English letters + romanized prompt transcript 0-28;
  self-made English prompt (the voice says 2 English lines in Nepali letters first) 0-33. CommonAccent native-like
  score is 40.5-45.1% for every variant (no signal). English needs the ear; cross-lingual prompting does not work on
  this checkpoint.
- "customer": 6 spellings x 6 takes; Kriti hears कस्टमर in all Devanagari spellings (6/6 each), English letters 2/6.
  The read-back cannot see what the user hears (same as the held final vowel).
- New male candidates: FLEURS ne_np (CC-BY 4.0) male clips 5.5-11 s, pitch 75-150 Hz (512), UTMOS top 160, ECAPA
  clusters at 0.6: 8 voices (UTMOS of the real clip 2.92-3.61). Rendered 4 lines each zero-shot.
- Listening page: ~/Downloads/v4_listen/Swar_v4_Listening.html (local, not in git).

## 2026-10-23 User answers on the v4 page, and fixes
- User: female A (news clip) is good for news; pick B or C for conversation (C: Rasa CONV 01918+02068; B and C tie,
  pron error 4.4 vs 4.7 on one line, UTMOS 3.55 vs 3.61, C has the longer 8.6 s prompt). English: way 1 (Nepali-letter
  English) with the news voice is the only good one, but "technology" and "tomorrow" are wrong. "customer": कस्टम्अर
  take 1. Male: assistant shortlists, user decides. User still heard खान as "khan" in A news np_story.
- English "sh": our tables spell it स the way Nepali writers do (show -> सो, shopping -> सपिङ) and the model says "so".
  `frontend/english._keep_sh`: S/SH phones and स/श letters matched in order, SH -> श when the counts agree (शो, शपिङ,
  श्योर, स्टेशन). Curated: technology टेक्नोलोजी, tomorrow टुमोरो, customer कस्टम्अर.
- Held final vowel, ops/final_vowel_cases6.json (8 takes, news prompt): plain अ beats halanta+अ in 6 of 7 words:
  खान 155 -> 172 ms, आज 172 -> 192, गएर 208 -> 218, किन 142 -> 175 (word kept 25% -> 62%), जान 112 -> 152, तर 185/182,
  आजको 85/85. Conversational prompt: खान 148/145 ms but word kept 50% -> 100%. `hold_final_schwa` now appends अ.

## 2026-10-23 podcast A podcasts (used with permission): speaker survey
4 files (2 podcast A, podcast C, podcast D), 48 kHz Opus, 368 min of speech in 3206 pieces
(`ops/podcast_survey.py`). One male voice runs through all four files (9 ECAPA clusters with centroid similarity
0.6-0.9, median pitch 150-172 Hz): the host, 180 min. Guests: v03 male 143 Hz (16 min, cleanest, UTMOS of best piece
3.37), v04 female, v05 male. UTMOS is low everywhere (1.5-2.6 cluster means) and DeepFilterNet3 moves it by only
+0.0-0.2, so it is not noise: fast, laughing, overlapping talk scores low on an English read-speech MOS.

## 2026-10-23 Podcast voices: 8 presets in one model (in progress)
User: v5 A news sounded distorted, the previous one better; C conversation good; all 4 male podcast voices good, keep
them all with names; wants 4 female voices too, all in one model with zero-shot cloning. New permitted podcasts:
podcast B (2 hosts) and four interview shows (one main woman each).
- Distortion: no clipping or loudness change between v4 and v5 clips (peaks -4.9 / -5.6 dB); the only change was the
  held-vowel spelling, so it is back to halanta+अ, and the demo take picker (ops/v4_lab.py) now measures the held
  vowel (MMS forced alignment, held_ms >= 120 ms) over 8 takes instead of trusting the read-back.
- Surveys: Thriving hosts 203 Hz (28.8 min) and 240 Hz (5.3 min); interviews 44-60 min per woman, UTMOS 2.2-3.2.
- `ops/build_podvoices.py`: 10 voices, per-voice relative cleanliness cut (worst 30% out, floor 1.6), ECAPA
  typicality >= 0.55, Kriti CTC confidence >= 0.80 (`KritiONNX.read_conf`, new), pace 6-22 letters/s, 10 held out per
  voice, 45 min cap: host 44.9 min, guest1 6.8, guest2 1.1, podAf 3.6, thrA 12.9, thrB 1.2, p4a 26.9, p4b 19.1, p4c 11.6,
  p4d 17.8; replay FLEURS ne_np 1500 + LibriTTS-R dev.clean 1500. 4193 rows.
- Kriti ONNX ran on CPU on the boxes: onnxruntime asked for TensorRT (absent) and could not find cuDNN 9; importing
  torch first (it ships libcudnn.so.9) lets the CUDA provider start.
- `ops/train_podvoices.sh` on the L4: nepali_bi_ema, lr 2e-5, 12 epochs, 8000-frame batches x 2 accumulation (16000
  OOMs on 22 GB), 184 updates/epoch at ~1.2 s. Eval: `ops/eval_podvoices.py` (held-out pieces: similarity to the
  real piece, Kriti CER, UTMOS; plus 24 unseen FLEURS speakers for zero-shot).
- Result (model_last = update 2000 of 2208; the update-2000 save broke on a full disk, model_last was intact).
  Held-out pieces, 8 per voice, one take, base nepali_bi_ema -> podvoices fine-tune (similarity / pron CER / UTMOS):
  host 0.626/1.9/3.11 -> 0.636/2.2/3.11; guest1 0.753/3.6/3.54 -> 0.760/2.1/3.52; guest2 0.738/1.4/3.11 -> 0.744/2.0/3.07;
  podAf 0.727/3.7/3.11 -> 0.736/4.8/3.09; thrA 0.697/2.7/3.95 -> 0.698/3.3/3.88; thrB 0.684/5.3/3.78 -> 0.689/5.5/3.93;
  p4a 0.725/3.3/3.76 -> 0.729/3.2/3.89; p4b 0.671/3.3/3.93 -> 0.678/2.9/3.85; p4c 0.628/6.3/3.93 -> 0.627/5.1/3.94;
  p4d 0.746/4.1/3.18 -> 0.752/3.7/3.19. Unseen FLEURS speakers (zero-shot): 0.650/6.4 -> 0.652/6.3.
  Read: similarity +0.005 on average, CER mixed, UTMOS level, zero-shot not hurt. As with the studio presets, the
  base already clones these voices; the fine-tune is kept as the one model (it is never worse) but is not the lever.
- Voices chosen for the demo (data/voices.json, names are ours): female asha (Rasa CONV), maya (Thriving host A),
  nisha (p4a), sabina (p4b); male aarav and bibek (podcast A host, two clips), kiran (guest1), sujan (guest2).
- User: "There should be an extra a in aaja", heard "aaj" in some lines. The demo take check only measured bare
  held words, never a held stem inside a case ending (आजको, the word in np_weather and np_today). `held_words` now
  returns the held letter's index (आजको -> ज), so आजको is measured too; 8 takes (16 for mix_office). Chosen takes:
  held sound 120-180 ms in 22 of 24 lines (sabina and sujan mix_office 100 ms, the shortest of आज/अब/गएर).

## 2026-10-23 English stage: real English speech with Nepali-letter text
User: full English is best as Nepali-letter English but "can be improved" (accent). `ops/build_en_deva.py`: 4000
LibriTTS-R dev.clean clips with their text run through the inference frontend (English -> Devanagari), plus the
podcast-voices set (8193 rows); from podvoices_ema, lr 2e-5, 6 epochs (1866 updates, A30, 1.65 updates/s).
Frontend bugs found on the way: an all-caps word with an apostrophe crashed the letter speller; "it's" became
इटको (possessive) instead of इट्स (contractions now come from CMUdict).
Eval, 4 voices x 12 lines (10 English, 2 mixed), best of 4: English WER best 0.3 -> 0.4, mean take 4.1 -> 4.9,
UTMOS 3.99 -> 3.98; mixed CER 6.2 -> 5.7, UTMOS 3.76 -> 3.79. No measurable change; intelligibility was already
near perfect, the open question is accent, so it goes to a blind A/B by ear (~/Downloads/en_test).
- User: all still bad, "everyone" said "eveeonee" (एभरीवन read with Nepali sounds). Nepali-letter English cannot carry
  English sounds. New path, no training: English text + a prompt of the SAME speaker speaking English, found in the
  podcasts by Whisper language ID (`ops/en_prompts.py`; maya, nisha, sabina fluent, UTMOS 4.0-4.3; kiran one clip;
  aarav and sujan none). v4_lab render keeps an English prompt's transcript in Latin (it was being converted).
  10 English lines, best of 4: podvoices best WER 8-11% vs nepali_bi 14-27%; best of 12 on podvoices: maya 4.4%,
  nisha 4.3%, sabina 10.4%, kiran 3.9%; UTMOS 4.28-4.31 for the women (Nepali-letter English: ~4.0). One line fails
  for every voice ("bought a beautiful new jacket"). Page: ~/Downloads/en_test/Swar_English_v2.html.
- User on the English-prompt path: "Maya ... can't understand properly, Nisha also very bad ... older was better",
  same for all voices. So Whisper WER on English-prompt output (4%) did not match the ear; back to Nepali-letter
  English. Two changes tested: (1) 27 Nepali-English spellings in data/loanwords.tsv (everyone एभ्रिवान; bought was
  बाट "from", that was दात "tooth", jacket was जैकेट); (2) a fine-tune adding each voice's own English pieces
  (Whisper LID > 0.8; 344 rows, all from the women: thrA 25, thrB 4, p4a 97, p4b 75, ...; the podcast A men have none)
  written in Nepali letters (`ops/build_own_en.py`, 846 updates on an A100-80GB).
  10 English lines x 5 voices, best of 8, new spellings: podvoices mean-take WER maya 2.3 / nisha 0.6 / sabina 1.0 /
  kiran 7.0 / aarav 3.0, UTMOS 4.22/4.26/4.19/3.94/3.77 (old spellings, best of 4: mean-take 4.1, UTMOS 3.99);
  own-English fine-tune: 2.2/0.4/1.2/7.7/3.1, UTMOS 4.22/4.28/4.20/3.93/3.79. The spellings are the lever; the
  fine-tune changes nothing measurable and is not kept.

## 2026-10-23 English spelling review: the 2000 most common English words
The spelling fix was the lever that worked (above), so the 2000 most common English words (wordfreq) were reviewed by
hand against the Nepali-English way of saying them. Rules in frontend/english.py: "o" before r is ओ (more मोर, was मर
"die"; four, before, north); final z after a consonant or "er" is स (years इयर्स, others अदर्स, things थिङ्स);
-ed after t/d and -es after a hissing sound keep their vowel (needed नीडेड, decided डिसाइडेड, changes चेन्जेज);
k/p/f before s/sh/t/ch are joined (इक्स्पेक्ट, एक्टिभ, सेफ्टी; feedback फिडब्याक and football still split); a table
spelling that splits a cluster CMUdict says has no vowel gets the halanta (सेक्शन, पिक्चर्स); English "sh" kept as श.
339 hand fixes in data/loanwords.tsv, e.g. former was फार्मर "farmer", wrong रङ "colour", short शर्ट "shirt", program
प्रोगग्राम, public पबब्लिक, china चाइन. Tests 40/40 (users युजर्स, laptop ल्याप्टप updated).
- Render with the reviewed spellings (data/en_eval2_texts.json: 10 lines, 6 new with words like excited, honestly,
  easier, provides, complicated), podvoices_ema, best of 8, Nepali-letter English: best-take WER 0.0 for all five
  voices (jacket line now right), mean take maya 1.9 / nisha 1.2 / sabina 1.5 / aarav 1.6 / kiran 8.7, UTMOS 4.25 /
  4.31 / 4.25 / 3.81 / 3.99. Page: ~/Downloads/en_test/Swar_English_v4.html.

## 2026-10-23 Final model numbers (paper Draft 1)
Same protocol for all (ops/bench_final.py: full frontend incl. the 2000-word English review, female_news preset with
pace, NFE 16, seed 0): NepTTS-Bench standard / pronunciation CER / UTMOS: podvoices (final) 6.19 / **2.92** / 3.82;
accent_8000 6.23 / 3.35 / 3.81; nepali_bi 6.45 / 3.26 / 3.78. Code-switch held-out (30): final 5.61, accent 5.70 (UTMOS
3.81 / 3.80). FLEURS (82 items): final CER 4.58 / sim 0.694 / UTMOS 2.39, accent 4.47 / 0.692 / 2.36. Paper:
docs/paper/swar.tex Draft 1 (copy ~/Downloads/Swar_Paper_Draft1.pdf).
- Deploy prep (no deploy: HF Pro not approved): space_clone now loads podvoices_ema.pt (MODEL_FILE), offers the eight
  preset voices from MODEL_REPO/presets (ops/build_presets.py: prompt + 4 typical pieces per voice through the normal
  enrolment; say() needs more than one piece to score a take), BILINGUAL=0 so English uses Nepali-letter English.
  Deploy order once Pro is on: build presets + upload podvoices_ema.pt and presets/ to Aarjan/swar from a
  box, then ops/deploy_space_clone.sh and the voting Space (space_vote/clips already holds the 8 voices).

## 2026-10-24 Naturalness push: FLEURS won, NepTTS-Bench split by voice
Goal: beat Indic-Speak Srijana on NepTTS-Bench UTMOS (3.97 vs our 3.82) and OmniVoice on FLEURS UTMOS (2.49 vs 2.39)
without losing the error lead. `ops/nat_lab.py`. Settings chosen on dev sets that share nothing with the tests: 60 Rasa
test sentences (long), 40 hand-written short phrases (data/short_dev.json, 4 seeds each), FLEURS *validation*
speakers (25 items, 4 speakers). Each test then run once with the chosen setting, one take, seed 0.
- **FLEURS** (dev): cleaning the prompt with DeepFilterNet3 raises UTMOS 2.49 -> 2.79 but costs CER (+0.7) and sim
  (-0.009); capping the removal at 18 dB (`tools/enhance.py dfn18`) gives UTMOS 2.70, sim 0.694 (=), CER +0.2. NFE 32
  and output denoising did not help. **Test, once: CER 4.55 / sim 0.691 / UTMOS 2.67** (before 4.58 / 0.694 / 2.39;
  OmniVoice 5.30 / 0.681 / 2.49; the real recordings score UTMOS 2.24). First on all three.
- **Bench voice** (long dev, CER / UTMOS): female_news 4.63 / 3.55, female_wiki 5.49 / 3.19, asha 5.70 / 3.26, maya
  7.24 / 3.95, nisha 6.90 / 3.89, sabina 7.13 / 3.95, aarav 8.56 / 3.36, kiran 9.08 / 3.52. Podcast prompts are fast
  casual talk; a *read-style prompt* (the model reads FLEURS-train sentences in the preset voice, take picked on other
  train sentences) helps: sabina 5.40 / 4.16, maya 5.77 / 4.08, nisha 5.94 / 4.07.
- Bench with the read-style Sabina prompt: standard 9.82 / pronunciation 6.57 / UTMOS **4.08**. Beats Srijana on all
  three (10.25 / 8.12 / 3.97), but more than doubles our studio voice's pronunciation errors. Cause: the outputs are
  exactly 20% shorter (F5 copies prompt pace), and the bench is 193/205 phrases under 30 letters (under 15 letters:
  2.5 -> 8.8). Short-phrase dev, 4 seeds (CER / UTMOS): studio 10.3 / 3.84; sabina 18.5 / 4.06 (pace 0.8: 16.9);
  nisha 16.4 / 4.07; maya 14.3 / 4.05. Every brighter voice costs 40-80% more short-phrase errors for +0.2 UTMOS.
- Did not help: best-of-4 by Kriti read-back (studio long dev CER 4.63 -> 5.14), NFE 32, CFG 2.5, slower pace,
  resemble-enhance on outputs (long UTMOS +0.24 but short +0.0, CER +0.3-1.1), DeepFilterNet or loudness on outputs
  (+0.01-0.04), silence padding (UTMOS flat or lower), the pre-fine-tune checkpoints with the Sabina prompt.
- Verdict: per voice (as the bench lists Indic-Speak Srijana and Sagar separately), the lowest errors are Swar's studio
  voice and the highest UTMOS is Swar's Sabina voice; no single voice leads on both.
- Follow-up, same day (box 545912): **slow read-style prompts** (made at pace 0.75, chosen on 40 other short phrases,
  data/short_sel.json) do not help: short dev sabina 16.7 / maya 14.6 (studio 10.3). **Short-phrase fine-tune**
  (`ops/build_short.py`: podcast pieces cut at word boundaries by MMS forced alignment, kept if Kriti reads them back,
  1,144 clips / 21.5 min, plus the podvoices rows; 1,680 updates from podvoices_ema, lr 1e-5): short dev studio 10.6,
  sabina 19.4, maya 15.4; long dev sabina 5.26 / 4.14; FLEURS dev CER 3.39 -> 4.26. Not kept.
- **The short-phrase gap is a scorer effect, at least in part.** The same short-dev audio read by Kriti (pronunciation
  CER): studio 4.57, sabina 4.42, maya 4.09, nisha 4.25, i.e. no gap; MMS: 10.3 / 18.5 / 14.3 / 16.4. A third ASR
  (IndicConformer) was not available (gated; only the README is cached). Kriti has a language model that can fill in
  expected words, so neither ASR settles it; a listener does. NepTTS-Bench's official scorer is also mms-1b-all, so the
  bench numbers stand as they are.
- Men, short dev (seed 0; MMS CER / UTMOS / Kriti pron-CER): studio male_wiki 14.8 / 3.88 / 4.8, male_news 15.6 / 3.77
  / 4.7, aarav 23.9 / 3.50 / 5.0, bibek 25.0 / 3.54 / 5.7, sujan 31.2 / 3.54 / 10.4, kiran 41.3 / 3.36 / 24.7 (Kiran
  really fails short phrases). Blind pick page for the user (local, third-party voices): ~/Downloads/Swar_Voice_Pick.html.
- **Demo voices picked by ear** (blind, 12 short phrases): women Maya 8, Nisha 4, studio 0, Sabina 0; men Aarav 5,
  Sujan 5, Bibek 2, Kiran 0, studio 0. Demo voices: Maya (read-style prompt) and Aarav (tie broken by Kriti
  short-phrase error 5.0 vs 10.4). Neither studio voice won a single phrase, so MMS's short-phrase penalty on the
  podcast voices did not match the listener.
- **Free Hugging Face hardware estimate** (2 CPU threads, no GPU, final model): load 4 s; 2 s phrase 117 s at NFE 16 /
  58 s at NFE 8; 10 s sentence 181 s / 90 s. One take only; enrolment and best-of-N takes would add more.
- **Maya on NepTTS-Bench** (read-style prompt, one take): 7.45 / 4.52 / UTMOS **4.09**. **OmniVoice scored fully for
  the first time** (our studio prompt, run earlier): pronunciation CER 4.32, UTMOS 3.93. Correction: the paper's "a
  third of the next system's pronunciation errors" compared against Srijana (8.12); the next system is OmniVoice, so it
  is about two thirds (2.92 / 4.32). Paper and docs/launch-copy.md corrected.
- Voting site live (Vercel, free): https://swar-vote.vercel.app (human-or-AI test: 16 Rasa sentences, real recording vs
  Maya/Aarav, plus 12 English/mixed Swar lines). Votes in a private Vercel Blob store; `site_vote/fetch_votes.mjs`.
- Final model uploaded to the private HF repo Aarjan/swar (podvoices_ema.pt).
- **NepaliMOS, final models** (public checkpoint, 205 bench sentences): Piper 3.05, TingTing Subina 2.99, Gemini 2.98,
  OmniVoice 2.98, Swar studio 2.96, ElevenLabs 2.95, TingTing Asmita 2.93, Sabina 2.91, Maya 2.88. Paired per-sentence
  differences are small but consistent (studio minus Piper -0.088, 95% -0.101 to -0.076). Not a win; and the predictor
  ranks the most error-prone system (Piper) first and the listener's pick (Maya) last.
- **Colab cloner** (notebooks/Swar_Clone_Colab.ipynb; app bundle in the HF repo's app/, ops/release_bundle.sh; presets
  from model-made audio only, ops/build_presets_synth.py): tested in a fresh venv on the box: download 24 s, voice from a
  10 s FLEURS upload 24 s, Fast/Best sentence 1-11 s on an A100, preset match Maya 0.91 / Aarav 0.89. Found and fixed:
  f5-tts pin 1.1.7 (conflicts with gradio, and not the version the model was built with: 1.1.22), and a crash when an
  upload yields one piece (autoclone.say compared against an empty list).
