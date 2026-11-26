# Written with Claude (Anthropic) under the author's direction; see README.md (Tests and demonstration data).
"""End-to-end check on the demonstration recordings: build the German exact pack from the released Kinglet-S
v0.4.0 head, run `kinglet analyze` on demo/audio and compare with demo/expected_detections.csv.
Needs the two release files in demo/models/ (see README.md) and ffmpeg; skips without them.
Run: python -m unittest discover tests"""

import contextlib
import csv
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kinglet import cli

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
SPECIES = ROOT / "paper/results/regionlist_de.csv"  # the paper's German list (region_lists.py)
MODEL, HEADPACK = DEMO / "models/kinglet-s-v0.4.0.power.onnx", DEMO / "models/kinglet-s-v0.4.0.headpack.npz"
SCORE_TOL = 0.01  # scores may differ in the last digits between CPUs and onnxruntime builds


def run_cli(*argv: str) -> None:
    with mock.patch("sys.argv", ["kinglet", *argv]), contextlib.redirect_stdout(io.StringIO()):
        cli.main()


def read(path: Path) -> dict:
    with path.open() as f:
        return {tuple(r[:6]): float(r[6]) for r in list(csv.reader(f))[1:]}


@unittest.skipUnless(MODEL.exists() and HEADPACK.exists(),
                     f"demo model files not found: download {MODEL.name} and {HEADPACK.name} into demo/models/ "
                     "(see README.md, 'Tests and demonstration data')")
@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class Demo(unittest.TestCase):
    def test_matches_expected_detections(self):
        with tempfile.TemporaryDirectory() as d:
            pack, out = Path(d) / "de.npz", Path(d) / "detections.csv"
            run_cli("build-pack", "--headpack", str(HEADPACK), "--species", str(SPECIES), "--out", str(pack))
            run_cli("analyze", str(DEMO / "audio"), "--model", str(MODEL), "--pack", str(pack), "--out", str(out))
            got = read(out)
        expected = read(DEMO / "expected_detections.csv")
        self.assertEqual(sorted(got), sorted(expected))
        for row, score in expected.items():
            self.assertAlmostEqual(got[row], score, delta=SCORE_TOL, msg=row)


if __name__ == "__main__":
    unittest.main()
