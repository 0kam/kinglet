# Written with Claude (Anthropic) under the author's direction; see README.md (Tests and demonstration data).
"""Offline tests of the scoring functions the paper's results rest on (paper/): class-mean average precision,
the site-weighted AP of the WABAD site bootstrap, the paired species bootstrap, and the false-detection curves
read at equal recall. They need the paper environment (pandas, scikit-learn, scipy, torch) and skip without it.
Run: python -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
try:
    from sklearn.metrics import average_precision_score

    from common.scoring import cmap
    from evaluate.stats import revision_stats as rs
    reason = None
except ImportError as e:  # pandas, scikit-learn, scipy
    rs = None
    reason = f"paper environment not installed ({e})"
try:
    import torch

    from evaluate.species_lists import opensetcurve as oc
    oc_reason = None
except ImportError as e:  # also torch
    oc = None
    oc_reason = f"paper environment not installed ({e})"


@unittest.skipIf(rs is None, reason)
class AveragePrecision(unittest.TestCase):
    def test_cmap_hand_computed(self):
        s = np.array([0.9, 0.8, 0.7, 0.6])
        y = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
        # species 1: hits at ranks 1 and 3 -> (1 + 2/3) / 2; species 2 has no positive and is left out;
        # species 3: hit at rank 4 -> 1/4
        self.assertAlmostEqual(cmap(np.stack([s, s, s], 1), y), ((1 + 2 / 3) / 2 + 1 / 4) / 2)

    def test_weighted_ap_is_ap_with_repeated_sites(self):
        rng = np.random.default_rng(0)
        n, n_sites = 300, 6
        s = np.round(rng.random(n), 1)  # many ties
        site = rng.integers(0, n_sites, n)
        y = rng.random(n) < 0.3 + 0.4 * s
        mult = np.array([[1] * n_sites, [2, 0, 1, 3, 0, 1], [0] * n_sites], float)  # float, as bootstrap() builds it
        ap = rs.weighted_ap(s, y, site, mult)
        self.assertAlmostEqual(ap[0], average_precision_score(y, s))
        rep = np.repeat(np.arange(n), mult[1, site].astype(int))  # every window repeated as often as its site was drawn
        self.assertAlmostEqual(ap[1], average_precision_score(y[rep], s[rep]))
        self.assertTrue(np.isnan(ap[2]))  # no site drawn: undefined

    def test_bootstrap_is_paired(self):
        teacher = np.random.default_rng(1).random(40)
        ratio, diff = rs.boot_ratio(teacher + 0.1, teacher, np.random.default_rng(2))
        self.assertEqual(diff.shape, (rs.N_BOOT,))
        np.testing.assert_allclose(diff, 0.1)  # same species drawn for both models in every resample
        ratio, _ = rs.boot_ratio(2 * teacher, teacher, np.random.default_rng(3))
        np.testing.assert_allclose(ratio, 2)
        self.assertEqual(rs.ci(np.arange(1001)), [25.0, 975.0])


@unittest.skipIf(oc is None, oc_reason)
class FalseDetectionCurves(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.s = rng.uniform(-9, 4, (500, 7))  # logit scores
        self.y = rng.random((500, 7)) < 0.15
        self.cl = rng.integers(0, 3, 500)
        self.h = oc.Hist(3)
        for i in range(0, 500, 128):  # streamed in chunks, as in score_set
            self.h.add(self.s[i : i + 128], self.y[i : i + 128], self.cl[i : i + 128])

    def test_curve_counts(self):
        for sel in (np.ones(3, bool), np.array([False, True, True])):
            w = sel[self.cl]
            s, y = self.s[w], self.y[w]
            c = oc.curve(self.h, sel)
            for i in (0, 500, 1200, 1500, oc.NB - 2):
                t = oc.EDGES[i]
                self.assertAlmostEqual(c["recall"][i], (s[y] >= t).mean())
                self.assertAlmostEqual(c["fp"][i], (s[~y] >= t).sum() / len(s))
                self.assertAlmostEqual(c["any"][i], (np.where(y, -np.inf, s) >= t).any(1).mean())

    def test_operating_point(self):
        t = np.log(0.1 / 0.9)
        self.assertAlmostEqual(oc.EDGES[1200], t)  # p = 0.1 on a grid edge
        r = oc.op(self.h, np.ones(3, bool))
        f = self.s > t
        self.assertAlmostEqual(r["fp_per_window"], (f & ~self.y).sum() / 500)
        self.assertAlmostEqual(r["recall"], (f & self.y).sum() / self.y.sum())
        self.assertAlmostEqual(r["p_any_fp"], (f & ~self.y).any(1).mean())

    def test_at_recall_interpolates(self):
        c = {"recall": np.array([1.0, 0.8, 0.5, 0.2]), "fp": np.array([4.0, 2.0, 1.0, 0.5]),
             "any": np.array([0.9, 0.6, 0.3, 0.1])}
        np.testing.assert_allclose(oc.at_recall(c, 0.65), (1.5, 0.45))
        np.testing.assert_allclose(oc.at_recall(c, 0.8), (2.0, 0.6))
        self.assertTrue(np.isnan(oc.at_recall(c, 0.1)).all())  # recall not reached: undefined

    def test_logit_from_logp(self):
        lp = torch.log(torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
        np.testing.assert_allclose(oc.logit_from_logp(lp).numpy(), np.log([1 / 9, 1, 9]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
