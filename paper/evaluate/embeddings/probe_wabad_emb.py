# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""wabad_embprobe_v1 stage 2: frozen-embedding readouts, Perch teacher vs kinglet students.

Species filter: >= MIN_POS positive windows and >= 2 recordings on WABAD.
(a) few-shot exemplar retrieval: per species, seed and k in {1, 5}: sample k positive windows,
    drop ALL windows of the exemplar recordings, rank the remaining windows by max cosine to
    the exemplars, AP over the remaining positives (skip seeds that leave none). 10 seeds.
(b) linear probe: one recording-disjoint 70/30 split (seed 0); per species a one-vs-rest
    logistic regression (liblinear, C=1) on standardized embeddings; cmAP over test windows.

A model name NAME_feat reads the trunk feature saved next to NAME's embedding
(embed_student.py --feature) instead of the 1536-d embedding (xsabl_v1 own-space readouts).

Output: results/wabad_embprobe_results.json (--out to write elsewhere; entries are merged by name)

Usage: python evaluate/embeddings/probe_wabad_emb.py [--models teacher B S XS XS_int8pub XS0_feat] [--jobs 8] [--out FILE]
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, RESULTS  # noqa: E402
from common.config import DISTILL_ROOT as DD  # noqa: E402
from common.scoring import load_parts  # noqa: E402

EMB_DIRS = {"teacher": DATA_ROOT / "wabad/emb_test", "B": DD / "emb_wabad_B",
            "S": DD / "emb_wabad_S", "XS": DD / "emb_wabad_XS",
            # mcuemb_v2: int8 TFLite trunk feature of the int8 model (ptqpub_v1), projected on the host
            # (embed_student.py --tflite)
            "XS_int8pub": DD / "emb_wabad_XS_int8pub"}
# xsvar_v1 / xsabl_v1 (selectable with --models, not in the defaults): the released XS again with its
# trunk feature (packfit_v1 extraction), XS seeds 1-2, class-loss-only XS, detached-head XS
# (g2xcl_v1:H1d); *_ridge = the trunk feature mapped to Perch space by ridge_emb_map.py
XSVAR_DIRS = {"XS0": DD / "packfit_v1/feat_wabad_XS", "XS0_ridge": DD / "xsvar_v1/emb_wabad_XS0_ridge",
              **{f"{n}{r}": DD / f"xsvar_v1/emb_wabad_{n}{r}"
                 for n in ("XSs1", "XSs2", "XScls", "XSdet") for r in ("", "_ridge")}}
ALL_DIRS = EMB_DIRS | XSVAR_DIRS
MIN_POS = 10
KS = (1, 5)
SEEDS = 10


def retrieval(emb_n: np.ndarray, pos_idx: np.ndarray, recs: np.ndarray, rng) -> dict[int, float]:
    """One species, one seed: AP per k (np.nan when no eval positives remain)."""
    out = {}
    for k in KS:
        ex = rng.choice(pos_idx, size=k, replace=False)
        keep = ~np.isin(recs, np.unique(recs[ex]))
        y = np.zeros(len(recs), bool)
        y[pos_idx] = True
        if not (y & keep).any():
            out[k] = np.nan
            continue
        s = (emb_n[keep] @ emb_n[ex].T).max(1)
        out[k] = average_precision_score(y[keep], s)
    return out


def load_emb(name: str, ids: list) -> np.ndarray:
    """f32 embeddings (or, for NAME_feat, trunk features) of one model in WABAD window order."""
    base, key = (name.removesuffix("_feat"), "feat") if name.endswith("_feat") else (name, "emb")
    e = load_parts(ALL_DIRS[base], keys=("ids", key))
    assert e["ids"].tolist() == ids, f"{name}: window order mismatch"
    return e[key].astype(np.float32)


def fit_probe(X_tr, y_tr, X_te):
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    clf.fit(X_tr, y_tr)
    return clf.decision_function(X_te)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(EMB_DIRS),
                    choices=list(ALL_DIRS) + [f"{n}_feat" for n in ALL_DIRS])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS / "wabad_embprobe_results.json")
    args = ap.parse_args()

    meta = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    recs = meta["window_id"].str.rsplit("_", n=1).str[0].to_numpy()
    sp_all = sorted({c for x in meta["labels"] for c in x})
    Y = {c: np.zeros(len(meta), bool) for c in sp_all}
    for i, labs in enumerate(meta["labels"]):
        for c in labs:
            Y[c][i] = True
    species = [c for c in sp_all if Y[c].sum() >= MIN_POS and len(np.unique(recs[Y[c]])) >= 2]
    print(f"{len(species)} species pass the filter (>= {MIN_POS} pos windows, >= 2 recordings)")

    # one recording-disjoint 70/30 split for the probe (seed 0)
    rng0 = np.random.default_rng(0)
    urec = np.unique(recs)
    test_rec = set(rng0.choice(urec, size=int(0.3 * len(urec)), replace=False))
    te = np.isin(recs, list(test_rec))
    probe_sp = [c for c in species if Y[c][~te].any() and Y[c][te].any()]
    print(f"{len(probe_sp)} species have positives in both probe splits")

    res_path = args.out
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name in args.models:
        emb = load_emb(name, meta["window_id"].tolist())
        emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        ret = {k: [] for k in KS}
        for c in species:
            pos_idx = np.flatnonzero(Y[c])
            aps = [retrieval(emb_n, pos_idx, recs, np.random.default_rng(zlib.crc32(f"{c}:{s}".encode())))
                   for s in range(SEEDS)]
            for k in KS:
                ret[k].append(np.nanmean([a[k] for a in aps]))

        mu, sd = emb[~te].mean(0), emb[~te].std(0) + 1e-6
        X_tr, X_te = (emb[~te] - mu) / sd, (emb[te] - mu) / sd
        scores = Parallel(n_jobs=args.jobs)(
            delayed(fit_probe)(X_tr, Y[c][~te], X_te) for c in probe_sp)
        probe_ap = [average_precision_score(Y[c][te], s) for c, s in zip(probe_sp, scores)]

        results[name] = {
            **{f"retrieval_k{k}_mAP": round(float(np.nanmean(ret[k])), 4) for k in KS},
            "probe_cmAP": round(float(np.mean(probe_ap)), 4),
            "n_species_retrieval": len(species), "n_species_probe": len(probe_sp),
        }
        print(name, results[name], flush=True)
        res_path.parent.mkdir(exist_ok=True)
        res_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
