"""Kinglet: kinglet-sized bioacoustic embedding and classification models distilled from Perch 2.0.

A model is a single ONNX file (preprocessing included): waveform (batch, 160000) float32 in [-1, 1]
at 32 kHz mono -> "embedding" (batch, 1536; a projection into the Perch 2.0 embedding space) and
"logits" (batch, n_classes + 1), whose last channel is a background/"other" class distilled from
the teacher's full-vocabulary softmax; scores are softmax probabilities over all channels with the
background column dropped. Class list ships as <model>.classes.csv next to the model.
"""

import csv
import subprocess
from pathlib import Path

import numpy as np

SR = 32_000
WIN = 5 * SR

__version__ = "0.2.0"


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


class Kinglet:
    """ONNX-runtime wrapper. `classes` rows: (ebird_code, scientific_name, japanese_name)."""

    def __init__(self, model_path: str | Path, threads: int = 4):
        import onnxruntime as ort

        model_path = Path(model_path)
        opt = ort.SessionOptions()
        opt.intra_op_num_threads, opt.inter_op_num_threads = threads, 1
        self.sess = ort.InferenceSession(str(model_path), opt, providers=["CPUExecutionProvider"])
        with model_path.with_suffix(".classes.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.classes = [(r["ebird_code"], r["scientific_name"], r["japanese_name"]) for r in rows]

    def run(self, wins: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """(n, 160000) float32 windows -> (embeddings (n, 1536), species scores (n, n_classes)):
        softmax probabilities with the trailing background channel dropped."""
        embs, scores = [], []
        for i in range(0, len(wins), batch):
            e, lg = self.sess.run(None, {"waveform": wins[i : i + batch]})
            assert lg.shape[1] == len(self.classes) + 1, "model/classes.csv mismatch"
            embs.append(e)
            p = np.exp(lg - lg.max(axis=1, keepdims=True))
            scores.append((p / p.sum(axis=1, keepdims=True))[:, :-1])
        return np.concatenate(embs), np.concatenate(scores)
