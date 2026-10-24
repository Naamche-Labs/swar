"""IndicF5 model.safetensors -> F5-TTS finetune checkpoint (strict-load verified)."""
import sys, torch
from safetensors.torch import load_file, save_file
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
src, vocab, dst = sys.argv[1:4]
sd = load_file(src)
ema = {k.replace("ema_model._orig_mod.", "ema_model."): v for k, v in sd.items()}
voc = {k.replace("vocoder._orig_mod.", ""): v for k, v in sd.items() if k.startswith("vocoder.")}
ema = {k: v for k, v in ema.items() if "mel_spec" not in k and not k.startswith("vocoder.")}
ema["initted"] = torch.tensor(True)
ema["step"] = torch.tensor(0)
vmap, vsize = get_tokenizer(vocab, "custom")
model = CFM(transformer=DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False,
                            conv_layers=4, pe_attn_head=1, text_num_embeds=vsize, mel_dim=100),
            mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
                                 target_sample_rate=24000, mel_spec_type="vocos"), vocab_char_map=vmap)
from ema_pytorch import EMA
e = EMA(model, include_online_model=False)
missing, unexpected = e.load_state_dict(ema, strict=False)
missing = [m for m in missing if "mel_spec" not in m]
print("vocab", vsize, "| missing", missing[:5], len(missing), "| unexpected", unexpected[:5], len(unexpected))
assert not missing and not unexpected
save_file(ema, dst)
save_file(voc, dst.replace(".safetensors", "_vocoder.safetensors"))  # IndicF5 ships its own vocos weights
print("wrote", dst)
