"""Offline tests for build-pack --region (eBird API mocked). Run: python -m unittest discover tests"""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

import numpy as np

from kinglet import cli, pack

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
