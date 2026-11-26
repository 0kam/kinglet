# Written with Claude (Anthropic) under the author's direction; see README.md (Tests and demonstration data).
"""Offline tests for decoding, 5-s windows and `kinglet analyze` (ONNX trunk and decoder mocked).
Run: python -m unittest discover tests"""

import contextlib
import csv
import io
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import kinglet
from kinglet import SR, WIN, cli, pack


class Windows(unittest.TestCase):
    def test_full_windows_only(self):
        audio = np.arange(12 * SR, dtype=np.float32)
        starts, wins = kinglet.windows(audio)
        np.testing.assert_array_equal(starts, [0.0, 5.0])  # the last 2 s do not fill a window
        np.testing.assert_array_equal(wins, [audio[:WIN], audio[WIN : 2 * WIN]])

    def test_hop(self):
        starts, wins = kinglet.windows(np.arange(10 * SR, dtype=np.float32), hop_s=2.5)
        np.testing.assert_array_equal(starts, [0.0, 2.5, 5.0])
        self.assertEqual(wins[1, 0], 2.5 * SR)

    def test_short_audio_is_zero_padded(self):
        audio = np.ones(SR, np.float32)
        starts, wins = kinglet.windows(audio)
        np.testing.assert_array_equal(starts, [0.0])
        self.assertEqual(wins.shape, (1, WIN))
        self.assertTrue((wins[0, :SR] == 1).all() and (wins[0, SR:] == 0).all())


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class Decode(unittest.TestCase):
    def test_samples_at_32khz(self):
        pcm = np.random.default_rng(0).integers(-32768, 32768, SR // 10)
        with tempfile.TemporaryDirectory() as d:
            write_wav(Path(d) / "a.wav", pcm, SR)
            a = kinglet.decode(Path(d) / "a.wav")
        self.assertEqual(a.dtype, np.float32)
        np.testing.assert_array_equal(a, pcm / 32768)  # 16-bit PCM -> [-1, 1), no resampling

    def test_resampled_to_32khz(self):
        t = np.arange(48_000) / 48_000
        with tempfile.TemporaryDirectory() as d:
            write_wav(Path(d) / "a.wav", np.round(16000 * np.sin(2 * np.pi * 1000 * t)), 48_000)
            a = kinglet.decode(Path(d) / "a.wav")
        self.assertAlmostEqual(len(a), SR, delta=SR // 1000)
        self.assertEqual(np.argmax(np.abs(np.fft.rfft(a[: SR // 2]))) * 2, 1000)  # 1 kHz tone, 2-Hz bins

    def test_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.wav").write_bytes(b"not audio")
            with self.assertRaises(RuntimeError):
                kinglet.decode(Path(d) / "x.wav")


class FirstSamplesSession:
    """onnxruntime.InferenceSession stand-in: feature = the first 4 samples of each window, embedding constant."""

    def __init__(self, *args, **kwargs):
        pass

    def get_inputs(self):
        return [SimpleNamespace(name="waveform")]

    def get_outputs(self):
        return [SimpleNamespace(name="embedding"), SimpleNamespace(name="feature")]

    def run(self, _, feeds):
        x = feeds["waveform"]
        return np.full((len(x), 1536), 3.0, np.float32), x[:, :4]


class Analyze(unittest.TestCase):
    def test_detections_csv(self):
        """Identity head over 4 classes, so window probabilities are the softmax of its first 4 samples:
        3 species, one group member ("Speech", human_voice threshold 0.3); exact pack."""
        probs = np.array([[0.60, 0.30, 0.05, 0.05],   # species 1 and 2 pass 0.1; species 3 does not
                          [0.12, 0.04, 0.04, 0.80],   # species 1; human_voice passes its threshold
                          [0.30, 0.28, 0.26, 0.16]])  # three species pass 0.1; --top-k 2 keeps two
        audio = np.zeros(17 * SR, np.float32)  # three windows; the last 2 s are dropped
        audio[np.arange(3)[:, None] * WIN + np.arange(4)] = np.log(probs)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.savez(d / "hp.npz", W=np.eye(4, dtype=np.float32), b=np.zeros(4, np.float32),
                     labels=np.array(["Aus bus", "Bus cus", "Cus dus", "Speech"]), meta="{}")
            pack.build_pack(d / "hp.npz", [{"ebird_code": "aaa1", "scientific_name": "Aus bus", "japanese_name": "A"},
                                           {"ebird_code": "bbb1", "scientific_name": "Bus cus"},
                                           {"ebird_code": "ccc1", "scientific_name": "Cus dus"}],
                            d / "p.npz", groups={"human_voice": ["Speech"]}, group_thresholds={"human_voice": 0.3})
            out = io.StringIO()
            with mock.patch("onnxruntime.InferenceSession", FirstSamplesSession), \
                    mock.patch.object(cli, "decode", return_value=audio), \
                    mock.patch("sys.argv", ["kinglet", "analyze", str(d / "rec.wav"), "--model", "toy.onnx",
                                            "--pack", str(d / "p.npz"), "--out", str(d / "det.csv"), "--top-k", "2",
                                            "--embeddings", str(d / "emb")]), \
                    contextlib.redirect_stdout(out):
                cli.main()
            with (d / "det.csv").open() as f:
                rows = list(csv.reader(f))
            emb = np.load(d / "emb" / "rec.npz")
        self.assertEqual(rows, [
            ["file", "start_s", "end_s", "label", "scientific_name", "japanese_name", "score"],
            ["rec.wav", "0.0", "5.0", "aaa1", "Aus bus", "A", "0.600"],
            ["rec.wav", "0.0", "5.0", "bbb1", "Bus cus", "", "0.300"],
            ["rec.wav", "5.0", "10.0", "aaa1", "Aus bus", "A", "0.120"],
            ["rec.wav", "5.0", "10.0", "human_voice", "", "", "0.8000"],
            ["rec.wav", "10.0", "15.0", "aaa1", "Aus bus", "A", "0.300"],
            ["rec.wav", "10.0", "15.0", "bbb1", "Bus cus", "", "0.280"]])
        self.assertIn("6 rows (species top 2/window >= 0.1; groups: human_voice >= 0.3)", out.getvalue())
        np.testing.assert_array_equal(emb["t_start"], [0.0, 5.0, 10.0])
        self.assertEqual(emb["embedding"].shape, (3, 1536))
        np.testing.assert_allclose(np.linalg.norm(emb["embedding"].astype(np.float32), axis=1), 1, atol=1e-3)

    def test_gather(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("b.WAV", "a.flac", "notes.txt", "sub/c.mp3"):
                (d / name).parent.mkdir(exist_ok=True)
                (d / name).touch()
            self.assertEqual(cli.gather([d, Path("x.ogg")]),
                             [d / "a.flac", d / "b.WAV", d / "sub/c.mp3", Path("x.ogg")])


if __name__ == "__main__":
    unittest.main()
