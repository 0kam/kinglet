"""Kinglet: kinglet-sized bioacoustic embedding and classification models distilled from Perch 2.0.

Since v0.3 a model is two files: a trunk ONNX mapping 5-s windows at 32 kHz mono to "embedding"
(batch, 1536; a projection into the Perch 2.0 embedding space) and "feature" (batch, d; the trunk
feature), plus a regional head pack (.npz, see kinglet.pack) folding the full 14,795-class Perch vocabulary onto
<region species> + <sound groups such as human_voice / vehicle / geophony> + "other".
Since v0.4 `Kinglet.run` and `embedding_from_feature` return the embedding L2-normalised (unit
length; pass normalize=False for the network's raw output, about 8-12x as long as a Perch 2.0
embedding and varying in length between windows).
Scores are softmax probabilities (an exact pack reads the full 14,795-way softmax, a compact pack
approximates the out-of-pack mass with its "other" row); a group's score is the sum of its member
probabilities. Prebuilt packs ship for Japan; `kinglet build-pack` folds any species list.
A trunk comes in two forms with the same outputs: the power-input trunk (input "power", the
power spectrogram, which Kinglet computes with numpy's FFT; the faster form for Python) and the
waveform-input trunk (input "waveform" (batch, 160000) float32 in [-1, 1]; preprocessing inside the
graph as a DFT matrix product, for runtimes without numpy). Kinglet accepts either.
v0.2 single-file models (logits in the graph) are not supported by this version.
"""

import subprocess
from pathlib import Path

import numpy as np

from .pack import load_pack

SR = 32_000
WIN = 5 * SR
N_FFT, HOP = 1024, 320
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic, as torch.hann_window

__version__ = "0.4.0"


def decode(path: str | Path, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Any audio file -> mono float32 waveform at 32 kHz (requires ffmpeg on PATH)."""
    res = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"ffmpeg failed on {path}: {res.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(res.stdout, dtype=np.float32)


def windows(audio: np.ndarray, hop_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """(start times in s, stacked 5 s windows); short audio is zero-padded to one window."""
    if len(audio) < WIN:
        audio = np.pad(audio, (0, WIN - len(audio)))
    hop = int(hop_s * SR)
    starts = np.arange(0, len(audio) - WIN + 1, hop)
    return starts / SR, np.stack([audio[s : s + WIN] for s in starts])


def power_spectrogram(wins: np.ndarray) -> np.ndarray:
    """(n, samples) float32 windows -> (n, 1 + samples // 320, 513) float32 power spectrogram
    (501 frames for a 5-s window): the torchaudio Spectrogram(n_fft=1024, hop_length=320, power=2)
    the models were trained on - centred frames with reflect padding, periodic Hann window, one-sided
    FFT. One window at a time (cache-friendly); the FFT runs in float64, which numpy computes faster
    than float32."""
    out = np.empty((len(wins), 1 + wins.shape[1] // HOP, N_FFT // 2 + 1), np.float32)
    for i, x in enumerate(wins):
        x = np.pad(x, N_FFT // 2, mode="reflect")
        z = np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * HANN)
        np.add(np.square(z.real), np.square(z.imag), out=out[i], casting="same_kind")
    return out


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm: inner-product search then ranks as cosine similarity does."""
    return emb / np.linalg.norm(emb, axis=-1, keepdims=True)


def embedding_from_feature(model_path: str | Path, feature: np.ndarray, scale: float = 1.0,
                           zero_point: int = 0, normalize: bool = True) -> np.ndarray:
    """Trunk feature (n, d) -> Perch-space embedding (n, 1536) on the host.

    The embedding is one linear layer on top of the trunk feature, so a device that cannot afford
    that layer (a microcontroller build logs the int8 feature, 1 byte per dimension) loses nothing:
    this applies the layer stored in the same tier's trunk ONNX. For quantised features pass the
    feature tensor's `scale` and `zero_point`. The result is L2-normalised as `Kinglet.run` returns it;
    normalize=False gives the layer's raw output. Needs the `onnx` package (pip install kinglet-bio[mcu]).
    """
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(model_path)).graph
    node = next(n for n in g.node if "embedding" in n.output)
    assert node.op_type == "Gemm" and node.input[0] == "feature", \
        "not a v0.3 split model (expected embedding = Gemm(feature))"
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer if i.name in node.input}
    W, b = init[node.input[1]], init[node.input[2]]
    h = (np.asarray(feature, np.float32) - zero_point) * scale
    assert h.ndim == 2 and h.shape[1] == W.shape[1], f"feature must be (n, {W.shape[1]}), got {h.shape}"
    emb = h @ W.T + b
    return l2_normalize(emb) if normalize else emb


class Kinglet:
    """Trunk ONNX + head pack. `classes` rows: (ebird_code, scientific_name, japanese_name);
    `groups` maps group name -> operating threshold (or None if not calibrated)."""

    def __init__(self, model_path: str | Path, pack_path: str | Path, threads: int = 4):
        import onnxruntime as ort

        opt = ort.SessionOptions()
        opt.intra_op_num_threads, opt.inter_op_num_threads = threads, 1
        self.sess = ort.InferenceSession(str(model_path), opt, providers=["CPUExecutionProvider"])
        assert [o.name for o in self.sess.get_outputs()] == ["embedding", "feature"], \
            "not a v0.3 split model (expected outputs: embedding, feature)"
        self.input = self.sess.get_inputs()[0].name
        assert self.input in ("power", "waveform"), f"unexpected model input {self.input!r}"
        p = load_pack(pack_path)
        n = p["n_species"]
        self.W, self.b, self.n_species = np.asarray(p["W"], np.float32), p["b"], n
        self.cols = p.get("cols")  # exact pack: full-head column of every pack row but "other"
        self.classes = list(zip([str(c) for c in p["codes"][:n]],
                                [str(s) for s in p["scientific_name"]],
                                [str(j) for j in p["japanese_name"]]))
        self.group_cols = p["group_cols"]
        self.groups = p["group_thresholds"]

    def run(self, wins: np.ndarray, batch: int = 64,
            normalize: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
        """(n, 160000) float32 windows -> (embeddings (n, 1536), species scores (n, n_species),
        {group: scores (n,)}): softmax probabilities over the pack rows (exact pack: over the full
        head, read at the pack columns); the background/"other" channel is not returned.
        Embeddings are L2-normalised unless normalize=False (the trunk's raw "embedding" output)."""
        embs, probs = [], []
        for i in range(0, len(wins), batch):
            x = wins[i : i + batch]
            e, h = self.sess.run(None, {self.input: power_spectrogram(x) if self.input == "power" else x})
            lg = h @ self.W.T + self.b
            p = np.exp(lg - lg.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            if self.cols is not None:
                p = p[:, self.cols]
            embs.append(e)
            probs.append(p)
        emb, p = np.concatenate(embs), np.concatenate(probs)
        if normalize:
            emb = l2_normalize(emb)
        gscores = {g: p[:, cols].sum(axis=1) for g, cols in self.group_cols.items()}
        return emb, p[:, : self.n_species], gscores
