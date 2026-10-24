"""Patch F5's trainer: skip any step whose loss or gradient norm is not finite, and log it.
v3 went NaN at update 793 and the stock trainer kept stepping on NaN for 16k updates."""
import re
import sys

T = sys.argv[1] if len(sys.argv) > 1 else "/home/venv/lib/python3.10/site-packages/f5_tts/model/trainer.py"
s = open(T).read()
if "NONFINITE" in s:
    print("already patched"); sys.exit()
s = s.replace('''                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
''', '''                    if not torch.isfinite(loss):
                        print(f"NONFINITE_LOSS update={global_update} mel_lengths={mel_lengths.tolist()} "
                              f"text_lens={[len(t) for t in text_inputs]}", flush=True)
                        self.optimizer.zero_grad()
                        continue
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        gn = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        if gn is not None and not torch.isfinite(gn):
                            print(f"NONFINITE_GRAD update={global_update} mel_lengths={mel_lengths.tolist()}", flush=True)
                            self.optimizer.zero_grad()
                            continue
''')
assert "NONFINITE_LOSS" in s, "pattern not found"
if not re.search(r"^import torch$", s, re.M):
    s = "import torch\n" + s
open(T, "w").write(s)
print("patched", T)
