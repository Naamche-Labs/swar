"""Export Kriti Telephony's CTC head to ONNX plus what pipeline/kriti_onnx.py needs (Nepali vocab slice, mel filterbank).
Needs the AI4Bharat NeMo fork @8dce88c (CPU is enough; on macOS drop triton from requirements/requirements.txt).
Usage: python ops/export_kriti_onnx.py OUT_DIR"""
import json
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
m = EncDecHybridRNNTCTCBPEModel.restore_from(hf_hub_download("Aarjan/kriti-telephony", "kriti_telephony.nemo"),
                                             strict=False, map_location="cpu").eval()
ne = m.tokenizer.tokenizers_dict["ne"]
json.dump({"offset": m.tokenizer.token_id_offset["ne"], "blank": m.ctc_decoder.num_classes_with_blank - 1,
           "pieces": [ne.tokenizer.id_to_piece(i) for i in range(ne.vocab_size)]},
          open(f"{out}/kriti_ne_vocab.json", "w"), ensure_ascii=False)
np.save(f"{out}/kriti_melfb.npy", m.preprocessor.featurizer.fb.squeeze(0).numpy())
m.cur_decoder = "ctc"
m.export(f"{out}/kriti_ctc.onnx")
print("EXPORTED", out)
