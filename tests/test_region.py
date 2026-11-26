# Written with Claude (Anthropic) under the author's direction; see README.md (Tests and demonstration data).
"""Offline tests for build-pack (eBird API and ONNX trunk mocked). Run: python -m unittest discover tests"""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from kinglet import Kinglet, cli, pack

KEY = "secret-ebird-key"
TAXA = {
    "aaa1": {"speciesCode": "aaa1", "sciName": "Aus bus", "category": "species"},
    "ccc1": {"speciesCode": "ccc1", "sciName": "Cus novus", "category": "species"},  # not in vocab
    "duck1": {"speciesCode": "duck1", "sciName": "Anas sp.", "category": "spuh"},
}


def fake_ebird(spplist: list[str], calls: list):
    """urlopen stand-in serving a region list and the taxonomy of the codes in TAXA."""

    def urlopen(req, timeout=None):
        calls.append(req)
        url = urllib.parse.urlsplit(req.full_url)
        if url.path.startswith("/v2/product/spplist/"):
            body = spplist
        else:
            codes = urllib.parse.parse_qs(url.query)["species"][0].split(",")
            body = [TAXA.get(c, {"speciesCode": c, "sciName": f"Genus {c}", "category": "species"})
                    for c in codes if c != "gone1"]
        return io.BytesIO(json.dumps(body).encode())

    return urlopen


class FeatureSession:
    """onnxruntime.InferenceSession stand-in for a trunk whose feature is its input."""

    def __init__(self, *args, **kwargs):
        pass

    def get_inputs(self):
        return [SimpleNamespace(name="waveform")]

    def get_outputs(self):
        return [SimpleNamespace(name="embedding"), SimpleNamespace(name="feature")]

    def run(self, _, feeds):
        x = feeds["waveform"]
        return np.ones((len(x), 1536), np.float32), x  # the embedding is not under test; non-zero for L2 normalisation


def run_cli(*argv: str) -> str:
    out = io.StringIO()
    with mock.patch("sys.argv", ["kinglet", *argv]), contextlib.redirect_stdout(out):
        cli.main()
    return out.getvalue()


class FetchRegionSpecies(unittest.TestCase):
    def test_happy_path(self):
        calls = []
        with mock.patch("urllib.request.urlopen", fake_ebird(["aaa1", "duck1", "ccc1", "gone1"], calls)):
            rows = pack.fetch_region_species("US-CA", KEY)
        self.assertEqual(rows, [{"ebird_code": "aaa1", "scientific_name": "Aus bus"},
                                {"ebird_code": "ccc1", "scientific_name": "Cus novus"},
                                {"ebird_code": "gone1", "scientific_name": ""}])
        self.assertEqual(calls[0].full_url, "https://api.ebird.org/v2/product/spplist/US-CA")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c.get_header("X-ebirdapitoken") == KEY for c in calls))

    def test_chunked_taxonomy(self):
        codes = [f"sp{i:03d}" for i in range(450)]
        calls = []
        with mock.patch("urllib.request.urlopen", fake_ebird(codes, calls)):
            rows = pack.fetch_region_species("CR", KEY)
        self.assertEqual([r["ebird_code"] for r in rows], codes)
        self.assertEqual(rows[449]["scientific_name"], "Genus sp449")
        sizes = [len(urllib.parse.parse_qs(urllib.parse.urlsplit(c.full_url).query)["species"][0].split(","))
                 for c in calls[1:]]
        self.assertEqual(sizes, [200, 200, 50])


class CompactPackScores(unittest.TestCase):
    def test_hand_computed_scores(self):
        """Compact pack, Eq. (S3) of the paper: p_k = exp(z_k) / (sum over pack rows exp(z_j) + exp(o)); "other" not returned."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "compact.npz"
            # rows: two species, one group member, "other"; feature (1, 0) gives logits log(2, 1, 1, 4)
            np.savez(path, W=np.array([[np.log(2), 0], [0, 0], [0, 0], [np.log(4), 0]], np.float32),
                     b=np.zeros(4, np.float32), codes=np.array(["aaa1", "bbb1", "Speech", "other"]), n_species=2,
                     scientific_name=np.array(["Aus bus", "Bus cus"]), japanese_name=np.array(["", ""]),
                     groups=json.dumps({"human_voice": ["Speech"]}), group_thresholds=json.dumps({"human_voice": 0.1}),
                     meta="{}")
            with mock.patch("onnxruntime.InferenceSession", FeatureSession):
                m = Kinglet("toy.onnx", path)
        _, scores, groups = m.run(np.array([[1, 0], [0, 1]], np.float32))
        np.testing.assert_allclose(scores, [[2 / 8, 1 / 8], [1 / 4, 1 / 4]], rtol=1e-6)
        np.testing.assert_allclose(groups["human_voice"], [1 / 8, 1 / 4], rtol=1e-6)
        self.assertEqual(m.classes, [("aaa1", "Aus bus", ""), ("bbb1", "Bus cus", "")])
        self.assertEqual(m.groups, {"human_voice": 0.1})


class BuildPackRegion(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        groups, _ = pack.default_groups()
        labels = ["Aus bus", "Genus zzz"] + [m for ms in groups.values() for m in ms]
        rng = np.random.default_rng(0)
        self.headpack = self.tmp / "hp.npz"
        np.savez(self.headpack, W=rng.normal(size=(len(labels), 4)).astype(np.float32),
                 b=np.zeros(len(labels), np.float32), labels=np.array(labels),
                 calib_h=rng.normal(size=(40, 4)).astype(np.float32),
                 calib_is_holdout=np.arange(40) % 2 == 0, meta=json.dumps({}))

    def test_no_group_rows(self):
        out = self.tmp / "species_only.npz"
        pack.build_pack(self.headpack, [{"ebird_code": "aaa1", "scientific_name": "Aus bus"}], out, groups={})
        p = pack.load_pack(out)
        self.assertEqual([str(c) for c in p["codes"]], ["aaa1", "other"])
        self.assertEqual(p["groups"], {})

    def test_region_builds_pack_and_species_csv(self):
        out = self.tmp / "my_region.npz"
        with mock.patch("urllib.request.urlopen", fake_ebird(["aaa1", "duck1", "ccc1"], [])), \
                mock.patch.dict(os.environ, {"EBIRD_API_KEY": KEY}):
            stdout = run_cli("build-pack", "--headpack", str(self.headpack), "--region", "US-CA",
                             "--out", str(out))
        self.assertNotIn(KEY, stdout)
        self.assertIn("WARNING: 1 species not in the Perch vocabulary (dead rows, never detected): ccc1", stdout)
        with (self.tmp / "my_region.species.csv").open() as f:
            self.assertEqual(list(csv.DictReader(f)), [{"ebird_code": "aaa1", "scientific_name": "Aus bus"},
                                                        {"ebird_code": "ccc1", "scientific_name": "Cus novus"}])
        p = pack.load_pack(out)
        self.assertEqual(p["n_species"], 2)
        self.assertEqual(p["meta"]["unmatched"], ["ccc1"])

    def test_exact_pack_is_the_full_softmax(self):
        # pack rows deliberately out of vocabulary order; the two unmatched species are dead rows
        species = [{"ebird_code": "ccc1", "scientific_name": "Cus novus"},
                   {"ebird_code": "aaa1", "scientific_name": "Aus bus"},
                   {"ebird_code": "ddd1", "scientific_name": "Dus novus"}]
        pack.build_pack(self.headpack, species, self.tmp / "exact.npz")
        run_cli("build-pack", "--headpack", str(self.headpack), "--species", str(self._csv(species)),
                "--out", str(self.tmp / "compact.npz"), "--compact")
        hp = np.load(self.headpack)
        labels = [str(n) for n in hp["labels"]]
        h = 3 * np.random.default_rng(1).normal(size=(9, 4)).astype(np.float32)
        lg = h.astype(np.float64) @ hp["W"].astype(np.float64).T + hp["b"]
        ref = np.exp(lg - lg.max(1, keepdims=True))
        ref /= ref.sum(1, keepdims=True)
        with mock.patch("onnxruntime.InferenceSession", FeatureSession):
            exact, compact = Kinglet("toy.onnx", self.tmp / "exact.npz"), Kinglet("toy.onnx", self.tmp / "compact.npz")
        _, scores, groups = exact.run(h)
        np.testing.assert_allclose(scores[:, 1], ref[:, labels.index("Aus bus")], rtol=1e-5, atol=1e-9)
        self.assertTrue((scores[:, [0, 2]] == 0).all())
        for g, members in pack.default_groups()[0].items():
            np.testing.assert_allclose(groups[g], ref[:, [labels.index(m) for m in members]].sum(1),
                                       rtol=1e-5, atol=1e-9)
        # the compact pack keeps its own layout (listed rows + "other") and the same outputs
        self.assertIsNone(compact.cols)
        self.assertEqual(len(compact.W), len(pack.load_pack(self.tmp / "compact.npz")["codes"]))
        _, c_scores, c_groups = compact.run(h)
        self.assertEqual(c_scores.shape, scores.shape)
        self.assertEqual(c_groups.keys(), groups.keys())

    def _csv(self, rows: list[dict]) -> Path:
        path = self.tmp / "species.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, ["ebird_code", "scientific_name"])
            w.writeheader()
            w.writerows(rows)
        return path

    def test_compact_equals_exact_when_other_is_exact(self):
        # one class outside the pack: its logit is the exact out-of-pack logsumexp and is linear in the
        # feature, so an unpenalised ridge recovers it and the compact pack must give the exact scores
        groups = pack.default_groups()[0]
        labels = [str(n) for n in np.load(self.headpack)["labels"]]
        species = [{"ebird_code": "aaa1", "scientific_name": "Aus bus"},
                   {"ebird_code": "ddd1", "scientific_name": "Dus novus"}]  # "Genus zzz" stays out of the pack
        pack.build_pack(self.headpack, species, self.tmp / "exact.npz", groups=groups)
        report = pack.build_pack(self.headpack, species, self.tmp / "compact.npz", groups=groups,
                                 ridge_lambda=0.0, compact=True)
        self.assertLess(report["prob_err_max"], 1e-5)
        p = pack.load_pack(self.tmp / "compact.npz")
        hp = np.load(self.headpack)
        np.testing.assert_allclose(p["W"][-1], hp["W"][labels.index("Genus zzz")], atol=1e-4)
        np.testing.assert_array_equal(p["W"][0], hp["W"][labels.index("Aus bus")])  # listed rows are sliced
        self.assertEqual(p["b"][1], pack.DEAD_BIAS)  # unmatched species: dead row
        h = 3 * np.random.default_rng(2).normal(size=(9, 4)).astype(np.float32)
        with mock.patch("onnxruntime.InferenceSession", FeatureSession):
            exact, compact = Kinglet("toy.onnx", self.tmp / "exact.npz"), Kinglet("toy.onnx", self.tmp / "compact.npz")
        (_, s_exact, g_exact), (_, s_compact, g_compact) = exact.run(h), compact.run(h)
        np.testing.assert_allclose(s_compact, s_exact, atol=1e-5)
        self.assertTrue((s_compact[:, 1] == 0).all())
        for g in groups:
            np.testing.assert_allclose(g_compact[g], g_exact[g], atol=1e-5)

    def test_missing_key(self):
        env = {k: v for k, v in os.environ.items() if k != "EBIRD_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(SystemExit) as cm:
            run_cli("build-pack", "--headpack", str(self.headpack), "--region", "CR",
                    "--out", str(self.tmp / "x.npz"))
        self.assertIn("https://ebird.org/api/keygen", str(cm.exception.code))

    def test_species_and_region_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as cm:
            run_cli("build-pack", "--headpack", str(self.headpack), "--species", "s.csv",
                    "--region", "CR", "--out", str(self.tmp / "x.npz"))
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("not allowed with argument", err.getvalue())


if __name__ == "__main__":
    unittest.main()
