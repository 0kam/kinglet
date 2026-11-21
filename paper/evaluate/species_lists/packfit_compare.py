# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""packfit_v1: how closely does the compact output layer (head pack) follow the exact full fold?

A head pack approximates the exact full-vocabulary fold: species / sound-group rows slice exactly
from the full head, and only the "other" logit - the out-of-pack logsumexp - is a ridge fit on the
calibration features inside the headpack bundle (kinglet.pack.build_pack). The bundle tested here
is <ckpt_dir>/headpack_focal.npz (export_headpack.py), calibrated on focal validation windows only.

Readout (label-free, public audio the calibration never saw): on the 60,158 WABAD windows compare
pack vs exact fold for each tier (B/S/XS) and list regime -
  JP    the released 317-species Japan list on all windows (foreign list),
  SITE  each site's annotated species list on that site's windows, pooled over the 72 sites.
R1 = fire@0.1 disagreement (any species p > 0.1), R2 = mean |p_other(pack) - p_other(exact)|.

Needs the WABAD trunk features: evaluate/embeddings/embed_student.py --feature
--out DISTILL_ROOT/packfit_v1/feat_wabad_<tier>, and the v0.3.1 Japan packs in RELEASE_ASSETS
(only their sound-group definitions are read).

Usage:
  python evaluate/species_lists/packfit_compare.py [--gpu]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, DISTILL_ROOT, RELEASE_ASSETS, RESULTS  # noqa: E402
from kinglet.pack import RIDGE_LAMBDA, build_pack  # noqa: E402
from teacher.embed_perch import perch_classes  # noqa: E402

TIERS = {"B": ("g3conf_v1/regnety_016_full", "b"), "S": ("g3conf_v1/efficientnet_lite0_full", "s"),
         "XS": ("g2xcl_v1/mobilenetv4_conv_small_full", "xs")}
THR = 0.1  # CLI operating point
CHUNK = 4096


class Bundle:
    """Calibration side of a headpack: full-vocabulary logits of the calibration features and the
    normal equations of the ridge fit (shared by every species list)."""

    def __init__(self, path: Path, W: torch.Tensor, b: torch.Tensor):
        hp = np.load(path, allow_pickle=False)
        assert np.array_equal(hp["W"], W.cpu().numpy().astype(np.float16)), f"{path}: another head"
        self.meta = json.loads(str(hp["meta"]))
        self.h = torch.from_numpy(hp["calib_h"].astype(np.float32)).to(W.device)
        self.is_ho = torch.from_numpy(hp["calib_is_holdout"]).to(W.device)
        self.lg = self.h @ W.t() + b
        hf = self.h[~self.is_ho].double()
        self.ht = torch.cat([hf, torch.ones(len(hf), 1, dtype=torch.float64, device=W.device)], 1)
        self.chol = torch.linalg.cholesky(
            self.ht.t() @ self.ht + RIDGE_LAMBDA * torch.eye(self.ht.shape[1], dtype=torch.float64,
                                                             device=W.device))

    def other_row(self, in_cols: torch.Tensor) -> torch.Tensor:
        """[w_o, b_o] (f32): ridge fit of the exact out-of-pack logsumexp on the fit features."""
        z = out_logsumexp(self.lg[~self.is_ho], in_cols).double()
        return torch.cholesky_solve((self.ht.t() @ z)[:, None], self.chol)[:, 0].float()


def out_logsumexp(lg: torch.Tensor, in_cols: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.logsumexp(lg[i : i + CHUNK].index_fill(1, in_cols, -torch.inf), 1)
                      for i in range(0, len(lg), CHUNK)])


class Tally:
    """Pack-vs-exact-fold agreement, accumulated over windows."""

    def __init__(self):
        self.s = dict.fromkeys(("n", "disagree", "pack_only", "exact_only", "fire_exact", "other_abs",
                                "prob_abs", "prob_n", "logit_abs", "logit_bias"), 0.0)
        self.prob_max = 0.0

    def add(self, lg: torch.Tensor, h: torch.Tensor, in_cols: torch.Tensor, n_sp: int, wb: torch.Tensor):
        """lg/h: full-vocabulary logits and trunk features of the windows; in_cols: live species
        columns (the first n_sp) then group-member columns; wb: the pack's other row.
        As in build_pack, the reference is the exact fold of the full softmax. A list may map two
        codes to one class (JP: rinphe1/rinphe2); the pack then holds that row twice, the exact
        fold gives both the class probability."""
        lin = lg[:, in_cols]
        z = out_logsumexp(lg, in_cols)
        o = h @ wb[:-1] + wb[-1]
        log_z = torch.cat([torch.logsumexp(lg[i : i + CHUNK], 1) for i in range(0, len(lg), CHUNK)])
        pe = torch.exp(torch.cat([lin, z[:, None]], 1) - log_z[:, None])
        pp = torch.softmax(torch.cat([lin, o[:, None]], 1), 1)
        fe, fp = (pe[:, :n_sp] > THR).any(1), (pp[:, :n_sp] > THR).any(1)
        err = (pp[:, :-1] - pe[:, :-1]).abs()
        for k, v in {"n": len(lg), "disagree": (fe != fp).sum(), "pack_only": (fp & ~fe).sum(),
                     "exact_only": (fe & ~fp).sum(), "fire_exact": fe.sum(),
                     "other_abs": (pp[:, -1] - pe[:, -1]).abs().sum(), "prob_abs": err.sum(),
                     "prob_n": err.numel(), "logit_abs": (o - z).abs().sum(),
                     "logit_bias": (o - z).sum()}.items():
            self.s[k] += float(v)
        self.prob_max = max(self.prob_max, float(err.max()))

    def result(self) -> dict:
        s, n = self.s, self.s["n"]
        return {"windows": int(n), "R1_fire_disagree": s["disagree"] / n, "R2_other_abs_err": s["other_abs"] / n,
                "pack_only_fire": s["pack_only"] / n, "exact_only_fire": s["exact_only"] / n,
                "fire_exact": s["fire_exact"] / n, "prob_err_mean": s["prob_abs"] / s["prob_n"],
                "prob_err_max": self.prob_max, "other_logit_abs_err": s["logit_abs"] / n,
                "other_logit_bias": s["logit_bias"] / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if args.gpu else "cpu"
    arm = "focal"

    man = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    site = man["site"].to_numpy()
    site_codes = {s: sorted({c for x in man.loc[site == s, "labels"] for c in x}) for s in np.unique(site)}
    pidx = {c["species"]: c["perch_idx"] for c in perch_classes(sorted({c for v in site_codes.values() for c in v}))}

    out = {"arm": arm, "tiers": {}}
    for tier, (ckdir, tag) in TIERS.items():
        ckdir = DISTILL_ROOT / ckdir
        hp = np.load(ckdir / f"headpack_{arm}.npz", allow_pickle=False)
        W = torch.from_numpy(hp["W"].astype(np.float32)).to(dev)
        b = torch.from_numpy(hp["b"].astype(np.float32)).to(dev)
        lab_idx = {str(n): i for i, n in enumerate(hp["labels"])}
        # sound groups as defined in the v0.3.1 release packs
        rel = np.load(RELEASE_ASSETS / f"kinglet-{tag}-v0.3.1.pack-jp.npz")
        groups = json.loads(str(rel["groups"]))
        midx = [lab_idx[m] for ms in groups.values() for m in ms]

        def in_cols(vidx) -> tuple[torch.Tensor, int]:
            live = [int(i) for i in vidx if i >= 0]
            assert not set(live) & set(midx), "species rows overlap group members"
            return torch.tensor(live + midx, device=dev), len(live)

        # WABAD windows: recomputed trunk features, checked against the cached embeddings
        ck = torch.load(ckdir / "best.pt", map_location="cpu", weights_only=False)
        f = np.load(DISTILL_ROOT / f"packfit_v1/feat_wabad_{tier}/part-0000.npz")
        cached = np.load(DISTILL_ROOT / f"emb_wabad_{tier}/part-0000.npz")
        assert f["ids"].tolist() == cached["ids"].tolist() == man["window_id"].astype(str).tolist()
        h = torch.from_numpy(f["feat"].astype(np.float32)).to(dev)
        emb = h.cpu() @ ck["model"]["emb_head.weight"].t() + ck["model"]["emb_head.bias"]
        cos = torch.nn.functional.cosine_similarity(emb, torch.from_numpy(cached["emb"].astype(np.float32)))
        assert cos.mean() >= 0.999, f"{tier}: recomputed features do not reproduce the cached embeddings"
        lg = torch.cat([h[i : i + CHUNK] @ W.t() + b for i in range(0, len(h), CHUNK)])

        jp = ck["cfg"]["classes"]
        jp_cols, jp_n = in_cols([c["perch_idx"] for c in jp])
        bd = Bundle(ckdir / f"headpack_{arm}.npz", W, b)
        res = {"feature_check_cos_mean": float(cos.mean()), "feature_check_cos_min": float(cos.min())}
        wb = bd.other_row(jp_cols)
        jp_t, ho_t, site_t = Tally(), Tally(), Tally()
        jp_t.add(lg, h, jp_cols, jp_n, wb)
        ho_t.add(bd.lg[bd.is_ho], bd.h[bd.is_ho], jp_cols, jp_n, wb)
        for s, codes in site_codes.items():
            cols, n_sp = in_cols([pidx[c] for c in codes])
            m = torch.from_numpy(site == s).to(dev)
            site_t.add(lg[m], h[m], cols, n_sp, bd.other_row(cols))
        res[arm] = {"calib": bd.meta, "JP": jp_t.result(), "SITE": site_t.result(),
                    "own_holdout_JP": ho_t.result()}
        print(tier, arm, json.dumps({k: {m: round(v, 5) for m, v in res[arm][k].items()}
                                     for k in ("JP", "SITE", "own_holdout_JP")}), flush=True)

        # bug check: the fit here is kinglet.pack.build_pack
        assert [str(c) for c in rel["codes"][: len(jp)]] == [c["species"] for c in jp]
        with tempfile.TemporaryDirectory() as tmp:
            rep = build_pack(ckdir / f"headpack_{arm}.npz",
                             [{"ebird_code": c["species"], "vocab_index": c["perch_idx"]} for c in jp],
                             Path(tmp) / "pack.npz", groups=groups, compact=True)
            ref = np.load(Path(tmp) / "pack.npz")
            row = wb.cpu().numpy()
            assert np.allclose(row[:-1], ref["W"][-1], atol=1e-4) and abs(row[-1] - ref["b"][-1]) < 1e-3, tier
        ho = res[arm]["own_holdout_JP"]
        assert abs(ho["R1_fire_disagree"] - rep["fire@0.1_disagree"]) < 5e-4, (ho, rep)  # a few borderline windows

        out["tiers"][tier] = res
        del lg, h, bd
    dst = RESULTS / f"packfit_v1_{arm}.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
