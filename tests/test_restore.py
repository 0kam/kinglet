"""Offline test for embedding_from_feature on a toy split model. Run: python -m unittest discover tests"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import kinglet

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # optional dependency (kinglet-bio[mcu])
    onnx = None


@unittest.skipUnless(onnx, "onnx not installed")
class RestoreTest(unittest.TestCase):
    def test_matches_the_graph_embedding(self):
        import onnxruntime as ort

        rng = np.random.default_rng(0)
        W, b = rng.normal(size=(6, 4)).astype(np.float32), rng.normal(size=6).astype(np.float32)
        graph = helper.make_graph(
            [helper.make_node("Relu", ["waveform"], ["feature"]),
             helper.make_node("Gemm", ["feature", "w", "b"], ["embedding"], transB=1)],
            "toy", [helper.make_tensor_value_info("waveform", TensorProto.FLOAT, ["n", 4])],
            [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["n", 6]),
             helper.make_tensor_value_info("feature", TensorProto.FLOAT, ["n", 4])],
            [numpy_helper.from_array(W, "w"), numpy_helper.from_array(b, "b")])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "toy.onnx"
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
            onnx.save(model, path)
            emb, feat = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
                None, {"waveform": rng.normal(size=(5, 4)).astype(np.float32)})
            np.testing.assert_allclose(kinglet.embedding_from_feature(path, feat), emb, atol=1e-5)
            # int8 feature as a microcontroller logs it: exact up to the rounding of the feature itself
            scale, zp = float(feat.max()) / 255, -128
            q = np.clip(np.round(feat / scale) + zp, -128, 127).astype(np.int8)
            restored = kinglet.embedding_from_feature(path, q, scale, zp)
            self.assertLessEqual(np.abs(restored - emb).max(), 0.5 * scale * np.abs(W).sum(1).max() + 1e-5)
            with self.assertRaises(AssertionError):
                kinglet.embedding_from_feature(path, feat[:, :3])


if __name__ == "__main__":
    unittest.main()
