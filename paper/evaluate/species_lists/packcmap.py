# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""packcmap_v1 + packother_v1: what the compact pack costs on WABAD site lists, and why.

For each tier (released v0.3.2 headpack bundle) and each of the 72 WABAD sites, a compact pack is built from the
site's annotated species (species rows + released sound-group rows + ridge "other" on the bundle's focal fit
windows = kinglet build-pack --compact; packfit_compare.Bundle) and applied to that site's windows.

packcmap_v1: site-masked score matrices (a species scores 0 on windows of sites whose list lacks it) for the pack and
  for the exact full-vocabulary fold -> WABAD cmAP and pooled AUPRC, and per-site pooled AUPRC.
  Bug check: the unmasked exact-fold cmAP reproduces wabad_v1b.
packother_v1 (diagnostic): other-logit error e = o_pack - z_exact (z = exact out-of-pack logsumexp) per site list on
  the bundle's focal holdout windows vs on the site's WABAD windows, and a refit of "other" on WABAD windows of the
  sites of the other parity (never shipped) scored the same way.

packrecall_diag_v1: recall and precision of the annotated occurrences at the default threshold 0.1, with the
  site-masked scores of the exact fold and of the pack (key `at_threshold`).

  python evaluate/species_lists/packcmap.py        # -> results/packcmap_v1.json (all three)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, DISTILL_ROOT, RELEASE_ASSETS, RESULTS  # noqa: E402
from common.scoring import cmap  # noqa: E402
from evaluate.species_lists.packfit_compare import Bundle, out_logsumexp  # noqa: E402
from kinglet.pack import RIDGE_LAMBDA  # noqa: E402
from teacher.embed_perch import perch_classes  # noqa: E402

TIERS = {"B": "b", "S": "s", "XS": "xs"}
THR = 0.1
PAPER_CMAP = {"B": 0.283, "S": 0.269, "XS": 0.259}  # wabad_v1b (benchmarks.csv, rounded)
torch.set_num_threads(8)


def ridge(h: torch.Tensor):
    """Cholesky factor of the ridge normal equations on features h (as kinglet.pack / Bundle)."""
    ht = torch.cat([h.double(), torch.ones(len(h), 1, dtype=torch.float64)], 1)
    return ht, torch.linalg.cholesky(ht.t() @ ht + RIDGE_LAMBDA * torch.eye(ht.shape[1], dtype=torch.float64))


def solve(ht, chol, z):
    return torch.cholesky_solve((ht.t() @ z.double())[:, None], chol)[:, 0].float()


def fold(lg: torch.Tensor, h: torch.Tensor, live: torch.Tensor, in_cols: torch.Tensor, wb: torch.Tensor):
    """Exact and pack probabilities of the listed species, and the other-logit error e = o - z."""
    z = out_logsumexp(lg, in_cols)
    o = h @ wb[:-1] + wb[-1]
    pe = torch.exp(lg[:, live] - torch.logsumexp(lg, 1, keepdim=True))
    pp = torch.softmax(torch.cat([lg[:, in_cols], o[:, None]], 1), 1)[:, : len(live)]
    return pe, pp, o - z


def summary(e: torch.Tensor, pe: torch.Tensor, pp: torch.Tensor) -> np.ndarray:
    fe, fp = (pe > THR).any(1), (pp > THR).any(1)
    return np.array([len(e), float(e.sum()), float(e.abs().sum()), float((fe & ~fp).sum()), float((fp & ~fe).sum()),
                     float(fe.sum())])


def rates(a: np.ndarray) -> dict:
    n = a[0]
    return {"windows": int(n), "other_logit_bias": a[1] / n, "other_logit_abs_err": a[2] / n,
            "exact_only_fire": a[3] / n, "pack_only_fire": a[4] / n, "fire_exact": a[5] / n}


def main():
    man = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    site = man["site"].to_numpy()
    sites = sorted(np.unique(site))
    parity = {s: k % 2 for k, s in enumerate(sites)}
    species = sorted({c for x in man["labels"] for c in x})
    sp_i = {c: j for j, c in enumerate(species)}
    pidx = np.array([c["perch_idx"] for c in perch_classes(species)])
    y = np.zeros((len(man), len(species)), np.int8)
    for i, labs in enumerate(man["labels"]):
        y[i, [sp_i[c] for c in labs]] = 1
    site_sp = {s: sorted({sp_i[c] for x in man.loc[site == s, "labels"] for c in x}) for s in sites}
    out = {"species": len(species), "no_perch_class": int((pidx < 0).sum()), "tiers": {}}

    for tier, tag in TIERS.items():
        hp_path = RELEASE_ASSETS / f"kinglet-{tag}-v0.3.2.headpack.npz"
        hp = np.load(hp_path)
        W, b = torch.from_numpy(hp["W"].astype(np.float32)), torch.from_numpy(hp["b"].astype(np.float32))
        lab = {str(n): i for i, n in enumerate(hp["labels"])}
        rel = np.load(RELEASE_ASSETS / f"kinglet-{tag}-v0.3.2.pack-jp.npz")
        midx = [lab[m] for ms in json.loads(str(rel["groups"])).values() for m in ms]
        bd = Bundle(hp_path, W, b)
        h_ho, lg_ho = bd.h[bd.is_ho], bd.lg[bd.is_ho]
        f = np.load(DISTILL_ROOT / f"packfit_v1/feat_wabad_{tier}/part-0000.npz")
        assert f["ids"].tolist() == man["window_id"].astype(str).tolist()
        h = torch.from_numpy(f["feat"].astype(np.float32))
        lg = torch.cat([h[i : i + 4096] @ W.t() + b for i in range(0, len(h), 4096)])  # 60k x 14,795 f32, ~3.6 GB
        # unmasked exact fold on every species (the paper's protocol) - bug check
        live_all = torch.from_numpy(np.maximum(pidx, 0))
        ex_all = torch.cat([torch.exp(lg[i : i + 4096][:, live_all]
                                      - torch.logsumexp(lg[i : i + 4096], 1, keepdim=True)) for i in range(0, len(h), 4096)])
        ex_all[:, torch.from_numpy(pidx < 0)] = 0.0
        unmasked = cmap(ex_all.numpy(), y)
        del ex_all
        par = np.array([parity[s] for s in site])
        refit = {p: ridge(h[torch.from_numpy(par != p)]) for p in (0, 1)}  # fit on the OTHER parity's windows
        s_pack, s_exact, s_refit = (np.zeros(y.shape, np.float32) for _ in range(3))
        acc = {k: np.zeros(6) for k in ("wabad_shipped", "focal_holdout_shipped", "wabad_refit")}
        per_site = []
        for s in sites:
            m = torch.from_numpy(site == s)
            cols = [j for j in site_sp[s] if pidx[j] >= 0]
            live = torch.tensor(pidx[cols])
            in_cols = torch.cat([live, torch.tensor(midx)])
            wb = bd.other_row(in_cols)
            pe, pp, e = fold(lg[m], h[m], live, in_cols, wb)
            acc["wabad_shipped"] += summary(e, pe, pp)
            pe_ho, pp_ho, e_ho = fold(lg_ho, h_ho, live, in_cols, wb)
            acc["focal_holdout_shipped"] += summary(e_ho, pe_ho, pp_ho)
            ht, chol = refit[parity[s]]
            other_win = torch.from_numpy(par != parity[s])
            wb_r = solve(ht, chol, out_logsumexp(lg[other_win], in_cols))
            _, pr, e_r = fold(lg[m], h[m], live, in_cols, wb_r)
            acc["wabad_refit"] += summary(e_r, pe, pr)
            rows = np.flatnonzero(site == s)
            s_exact[np.ix_(rows, cols)] = pe.numpy()
            s_pack[np.ix_(rows, cols)] = pp.numpy()
            s_refit[np.ix_(rows, cols)] = pr.numpy()
            ys = y[np.ix_(rows, site_sp[s])].ravel()
            ap = {k: float(average_precision_score(ys, a[np.ix_(rows, site_sp[s])].ravel()))
                  for k, a in (("exact", s_exact), ("pack", s_pack), ("refit", s_refit))}
            per_site.append({"site": s, **ap})
        ps = pd.DataFrame(per_site)
        res = {"bug_check_unmasked_exact_cmAP": unmasked, "paper_cmAP": PAPER_CMAP[tier]}
        for k, a in (("exact", s_exact), ("pack", s_pack), ("refit_diagnostic", s_refit)):
            res[f"masked_cmAP_{k}"] = cmap(a, y)
            res[f"masked_pooled_auprc_{k}"] = float(average_precision_score(y.ravel(), a.ravel()))
        # packrecall_diag_v1: annotated occurrences detected at the default threshold, site-masked scores
        res["at_threshold"] = {k: {"recall": float(((a > THR) & (y == 1)).sum() / y.sum()),
                                   "precision": float(((a > THR) & (y == 1)).sum() / (a > THR).sum())}
                               for k, a in (("exact", s_exact), ("pack", s_pack))}
        for k in ("pack", "refit"):
            d = ps[k] - ps["exact"]
            res[f"site_auprc_{k}_minus_exact"] = {"median": float(d.median()), "min": float(d.min()),
                                                  "max": float(d.max()), "sites_worse": int((d < 0).sum())}
        res["site_auprc_median"] = {k: float(ps[k].median()) for k in ("exact", "pack", "refit")}
        res["packother"] = {k: rates(v) for k, v in acc.items()}
        assert abs(unmasked - PAPER_CMAP[tier]) < 0.003, (tier, unmasked)
        out["tiers"][tier] = res
        print(tier, json.dumps(res, default=float), flush=True)
        del lg, bd, s_pack, s_exact, s_refit
    dst = RESULTS / "packcmap_v1.json"
    dst.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", dst)


if __name__ == "__main__":
    main()
