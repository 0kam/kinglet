"""Offline test of the FFT frontend: power_spectrogram against a direct DFT, and Kinglet.run on a power-input
toy model against the same model with the spectrogram inside the graph (as a waveform-input trunk computes it,
a DFT matrix product). Run: python -m unittest discover tests"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import kinglet
from kinglet import HANN, HOP, N_FFT, Kinglet, pack

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # optional dependency (kinglet-bio[mcu])
    onnx = None

SAMPLES = 16 * HOP  # a short signal: 17 frames


def reference(x: np.ndarray) -> np.ndarray:
    """Power spectrogram by definition, float64: centred frames (reflect padding), periodic Hann, DFT."""
    x = np.pad(x.astype(np.float64), N_FFT // 2, mode="reflect")
    frames = np.stack([x[t * HOP : t * HOP + N_FFT] for t in range(1 + (len(x) - N_FFT) // HOP)])
    k = np.arange(N_FFT // 2 + 1)[:, None] * np.arange(N_FFT)[None, :]
    return np.abs(frames * HANN @ np.exp(-2j * np.pi * k / N_FFT).T) ** 2


class PowerSpectrogramTest(unittest.TestCase):
    def test_matches_the_definition(self):
        rng = np.random.default_rng(0)
        w = rng.uniform(-1, 1, (3, SAMPLES)).astype(np.float32)
        p = kinglet.power_spectrogram(w)
        self.assertEqual((p.shape, p.dtype), ((3, 17, N_FFT // 2 + 1), np.float32))
        for a, x in zip(p, w):
            r = reference(x)
            self.assertLess(np.abs(a - r).max() / r.max(), 1e-6)
        self.assertEqual(kinglet.power_spectrogram(np.zeros((1, kinglet.WIN), np.float32)).shape, (1, 501, 513))


@unittest.skipUnless(onnx, "onnx not installed")
class PowerInputRunTest(unittest.TestCase):
    def toy(self, path: Path, waveform: bool):
        """feature = Gemm(log(mean over frames of power + 1e-6)), embedding = Gemm(feature); the waveform-input
        variant computes power in the graph: reflect pad, framing (Gather), DFT matrix (MatMul), re^2 + im^2."""
        rng = np.random.default_rng(1)
        nb = N_FFT // 2 + 1
        inits = [numpy_helper.from_array(rng.normal(size=(4, nb)).astype(np.float32) / 10, "wf"),
                 numpy_helper.from_array(rng.normal(size=(6, 4)).astype(np.float32), "we"),
                 numpy_helper.from_array(np.array([1e-6], np.float32), "eps")]
        nodes = [helper.make_node("ReduceMean", ["power"], ["m"], axes=[1], keepdims=0),
                 helper.make_node("Add", ["m", "eps"], ["me"]),
                 helper.make_node("Log", ["me"], ["l"]),
                 helper.make_node("Gemm", ["l", "wf"], ["feature"], transB=1),
                 helper.make_node("Gemm", ["feature", "we"], ["embedding"], transB=1)]
        n_frames = 1 + SAMPLES // HOP
        if waveform:
            k = 2 * np.pi * np.arange(N_FFT)[None, :] * np.arange(nb)[:, None] / N_FFT
            dft = (np.concatenate([np.cos(k), -np.sin(k)]) * HANN).T.astype(np.float32)  # (1024, 1026)
            idx = np.arange(n_frames)[:, None] * HOP + np.arange(N_FFT)[None, :]
            inits += [numpy_helper.from_array(np.array([0, N_FFT // 2, 0, N_FFT // 2]), "pads"),
                      numpy_helper.from_array(idx, "idx"), numpy_helper.from_array(dft, "dft"),
                      *[numpy_helper.from_array(np.array([v]), n) for n, v in
                        (("s0", 0), ("s1", nb), ("s2", 2 * nb), ("ax", 2))]]
            nodes = [helper.make_node("Pad", ["waveform", "pads"], ["padded"], mode="reflect"),
                     helper.make_node("Gather", ["padded", "idx"], ["frames"], axis=1),
                     helper.make_node("MatMul", ["frames", "dft"], ["z"]),
                     helper.make_node("Mul", ["z", "z"], ["z2"]),
                     helper.make_node("Slice", ["z2", "s0", "s1", "ax"], ["re2"]),
                     helper.make_node("Slice", ["z2", "s1", "s2", "ax"], ["im2"]),
                     helper.make_node("Add", ["re2", "im2"], ["power"]), *nodes]
            inp = helper.make_tensor_value_info("waveform", TensorProto.FLOAT, ["n", SAMPLES])
        else:
            inp = helper.make_tensor_value_info("power", TensorProto.FLOAT, ["n", n_frames, nb])
        graph = helper.make_graph(nodes, "toy", [inp],
                                  [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["n", 6]),
                                   helper.make_tensor_value_info("feature", TensorProto.FLOAT, ["n", 4])], inits)
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), path)

    def test_power_input_matches_waveform_input(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(2)
            labels = ["Aus bus", "Bus cus", "Cus dus"]
            np.savez(d / "hp.npz", W=rng.normal(size=(3, 4)).astype(np.float16), b=np.zeros(3, np.float32),
                     labels=np.array(labels), meta="{}")
            pack.build_pack(d / "hp.npz", [{"ebird_code": "aaa", "scientific_name": "Aus bus"},
                                           {"ebird_code": "ccc", "scientific_name": "Cus dus"}], d / "p.npz", groups={})
            self.toy(d / "wave.onnx", waveform=True)
            self.toy(d / "power.onnx", waveform=False)
            wins = rng.normal(0, 0.1, (5, SAMPLES)).astype(np.float32)
            wave, power = Kinglet(d / "wave.onnx", d / "p.npz"), Kinglet(d / "power.onnx", d / "p.npz")
            self.assertEqual((wave.input, power.input), ("waveform", "power"))
            (ew, pw, _), (ep, pp, _) = wave.run(wins, batch=2), power.run(wins, batch=2)
            np.testing.assert_allclose(ep, ew, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(pp, pw, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
