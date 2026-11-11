"""Kinglet: kinglet-sized bioacoustic embedding and classification models distilled from Perch 2.0.

Since v0.3 a model is two files: a trunk ONNX (preprocessing included) mapping waveform
(batch, 160000) float32 in [-1, 1] at 32 kHz mono to "embedding" (batch, 1536; a projection into
the Perch 2.0 embedding space) and "feature" (batch, d; the trunk feature), plus a regional head
pack (.npz, see kinglet.pack) folding the full 14,795-class Perch vocabulary onto
<region species> + <sound groups such as human_voice / vehicle / geophony> + "other".
Scores are softmax probabilities over the pack rows; a group's score is the sum of its member
probabilities. Prebuilt packs ship for Japan; `kinglet build-pack` folds any species list.
v0.2 single-file models (logits in the graph) are not supported by this version.
"""

import subprocess
from pathlib import Path

import numpy as np

from .pack import load_pack

SR = 32_000
WIN = 5 * SR

__version__ = "0.3.1"


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


def embedding_from_feature(model_path: str | Path, feature: np.ndarray, scale: float = 1.0,
                           zero_point: int = 0) -> np.ndarray:
    """Trunk feature (n, d) -> Perch-space embedding (n, 1536) on the host.

    The embedding is one linear layer on top of the trunk feature, so a device that cannot afford
    that layer (a microcontroller build logs the int8 feature, 1 byte per dimension) loses nothing:
    this applies the layer stored in the same tier's trunk ONNX. For quantised features pass the
    feature tensor's `scale` and `zero_point`. Needs the `onnx` package (pip install kinglet-bio[mcu]).
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
    return h @ W.T + b


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
        p = load_pack(pack_path)
        n = p["n_species"]
        self.W, self.b, self.n_species = p["W"], p["b"], n
        self.classes = list(zip([str(c) for c in p["codes"][:n]],
                                [str(s) for s in p["scientific_name"]],
                                [str(j) for j in p["japanese_name"]]))
        self.group_cols = p["group_cols"]
        self.groups = p["group_thresholds"]

    def run(self, wins: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray, dict]:
        """(n, 160000) float32 windows -> (embeddings (n, 1536), species scores (n, n_species),
        {group: scores (n,)}): softmax probabilities over the pack rows; the trailing
        background/"other" channel is dropped from species scores."""
        embs, probs = [], []
        for i in range(0, len(wins), batch):
            e, h = self.sess.run(None, {"waveform": wins[i : i + batch]})
            lg = h @ self.W.T + self.b
            p = np.exp(lg - lg.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            embs.append(e)
            probs.append(p)
        emb, p = np.concatenate(embs), np.concatenate(probs)
        gscores = {g: p[:, cols].sum(axis=1) for g, cols in self.group_cols.items()}
        return emb, p[:, : self.n_species], gscores
