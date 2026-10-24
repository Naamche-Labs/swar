#!/bin/bash
# bench fixes on chars10k: calm reference voice x short-text slowdown
export HF_HOME=/home/hf_cache PYTHONUNBUFFERED=1
cd /home/nepali-tts
C=/home/ema/chars_model_10000.pt
b() { env "$@" /home/venv/bin/python tools/eval_bench.py run "$NAME" $C 2>&1 | grep -E "CER|Error" | tail -2; }
NAME=c10k_malewiki            b REF=male_wiki
NAME=c10k_malewiki_short15_05 b REF=male_wiki SHORT_CHARS=15 SHORT_SPEED=0.5
NAME=c10k_malewiki_short15_03 b REF=male_wiki SHORT_CHARS=15 SHORT_SPEED=0.3
NAME=c10k_femnews_short15_05  b REF=female_news SHORT_CHARS=15 SHORT_SPEED=0.5
NAME=c10k_malenews_short15_05 b REF=male_news SHORT_CHARS=15 SHORT_SPEED=0.5
echo BQ_DONE
