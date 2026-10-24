"""Kriti Telephony CTC head as ONNX, no NeMo: runs in the same process as F5-TTS (needed for a ZeroGPU Space, where a
side process cannot reach the GPU). Built by ops/export_kriti_onnx.py from the AI4Bharat NeMo fork; the log-mel
frontend here reproduces NeMo's FilterbankFeatures (preemph 0.97, 25 ms Hann window, 10 ms hop, n_fft 512, 80 mels,
log guard 2^-24, per-feature normalisation). Output: greedy CTC over the Nepali slice of the multilingual vocab.
Usage: KritiONNX(model_dir).read([wav, ...]) -> [text, ...]"""
import json
import os

import numpy as np


class KritiONNX:
    def __init__(self, d, providers=None):
        try:  # torch ships libcudnn.so.9; loading it first lets onnxruntime's CUDA provider start (else: CPU)
            import torch
            torch.backends.cudnn.version()
        except Exception:
            pass
        import onnxruntime as ort
        # TensorRT is listed on GPU builds but not installed on our boxes; asking for it made onnxruntime fall back to CPU
        avail = [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
        self.s = ort.InferenceSession(os.path.join(d, "kriti_ctc.onnx"), providers=providers or avail)
        v = json.load(open(os.path.join(d, "kriti_ne_vocab.json")))
        self.off, self.pieces, self.blank = v["offset"], v["pieces"], v["blank"]
        self.cols = list(range(self.off, self.off + len(self.pieces))) + [self.blank]
        self.fb = np.load(os.path.join(d, "kriti_melfb.npy")).astype(np.float32)  # (80, 257)
        self.win = np.hanning(400).astype(np.float32)  # torch.hann_window(400, periodic=False)

    def features(self, w):
        w = np.concatenate([w[:1], w[1:] - 0.97 * w[:-1]]).astype(np.float32)
        n_fft, hop = 512, 160
        win = np.zeros(n_fft, np.float32)
        win[(n_fft - 400) // 2:(n_fft - 400) // 2 + 400] = self.win
        x = np.pad(w, n_fft // 2, mode="constant")
        n = 1 + (len(x) - n_fft) // hop
        frames = np.lib.stride_tricks.as_strided(x, (n, n_fft), (x.strides[0] * hop, x.strides[0])) * win
        p = (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).astype(np.float32)
        f = np.log(self.fb @ p.T + 2 ** -24)  # (80, frames)
        L = len(w) // hop + 1
        f = f[:, :L]
        f = (f - f.mean(1, keepdims=True)) / (f.std(1, ddof=1, keepdims=True) + 1e-5)
        return f.astype(np.float32)

    def read(self, paths):
        return [t for t, _ in self.read_conf(paths)]

    def read_conf(self, paths):
        """(text, confidence): confidence = mean probability of the chosen piece over the frames that emit one."""
        import librosa
        out = []
        for p in paths:
            f = self.features(librosa.load(p, sr=16000)[0])
            lp = self.s.run(None, {"audio_signal": f[None], "length": np.array([f.shape[1]], np.int64)})[0][0][:, self.cols]
            pr = np.exp(lp - lp.max(-1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            ids = pr.argmax(-1)
            toks, prev, conf = [], None, []
            for t, i in enumerate(ids):
                if i != len(self.pieces):
                    conf.append(pr[t, i])
                    if i != prev:
                        toks.append(self.pieces[i])
                prev = i
            out.append(("".join(toks).replace("▁", " ").strip(), float(np.mean(conf)) if conf else 0.0))
        return out
