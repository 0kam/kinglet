# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""probe_l2_post_v1: teacher-trained probe applied to student embeddings with L2 normalisation (and raw or
rescaled embeddings) instead of the teacher's per-dimension standardisation.

Same species, recording/group-disjoint 70/30 split (seed 0) and probe (liblinear, C=1) as
compat_wabad_emb.py (WABAD) and eval_nonbird.py (non-bird sets). Per set and preprocessing
(std = teacher standardisation as before, l2 = unit-norm embeddings, no standardisation,
raw = embeddings as they are, as the Perch Hoplite agile classifier uses them; rev2_rawprobe_v1,
rawscaled = raw times one global constant per model, the teacher's mean training-split norm over the model's; rev2_rawscale_v1):
  own_<m>    : probe trained and tested in model m's own space (std uses m's own statistics)
  cross_<m>  : probe trained on the teacher, applied unchanged to m's test embeddings
The std rows recompute the published numbers as the bug check.

Also per set: mean L2 norm of each model's embeddings, and per preprocessing the number of liblinear fits
that stopped at max_iter (ConvergenceWarning).

Output: results/probe_l2_post.json (--out for another file)
Usage: python evaluate/embeddings/probe_l2_post.py [--sets wabad watkins anuraset humbugdb reefset] [--jobs 16] [--preps std l2 raw rawscaled]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, RESULTS  # noqa: E402
from evaluate.embeddings.probe_wabad_emb import MIN_POS, load_emb  # noqa: E402
from evaluate.nonbird.eval_nonbird import DATA, MODELS, NON_SPECIES, fit_apply, load_units, norm  # noqa: E402


def wabad():
    meta = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    groups = meta["window_id"].str.rsplit("_", n=1).str[0].to_numpy()
    labels = list(meta["labels"])
    emb = {m: load_emb(m, meta["window_id"].tolist()) for m in MODELS}
    return emb, labels, groups


def nonbird(name: str):
    units, labels, groups = load_units(name, pd.read_parquet(DATA / name / "manifest.parquet"))
    return {m: units[m]["emb"] for m in MODELS}, labels, groups


def fit_apply_conv(X_tr, y_tr, X_tests: list) -> tuple[list, bool]:
    """eval_nonbird.fit_apply, plus whether liblinear stopped at max_iter."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        sc = fit_apply(X_tr, y_tr, X_tests)
    return sc, any(issubclass(x.category, ConvergenceWarning) for x in w)


def readouts(emb: dict, labels: list, groups: np.ndarray, jobs: int, preps: list) -> dict:
    all_lab = sorted({c for x in labels for c in x})
    Y = {c: np.array([c in x for x in labels]) for c in all_lab}
    species = [c for c in all_lab if c not in NON_SPECIES and Y[c].sum() >= MIN_POS
               and len(np.unique(groups[Y[c]])) >= 2]
    rng0 = np.random.default_rng(0)
    ug = np.unique(groups)
    te = np.isin(groups, list(set(rng0.choice(ug, size=int(0.3 * len(ug)), replace=False))))
    probe_sp = [c for c in species if Y[c][~te].any() and Y[c][te].any()]

    def cmap(scores):
        return round(float(np.mean([average_precision_score(Y[c][te], s) for c, s in zip(probe_sp, scores)])), 4)

    out = {"n_species_probe": len(probe_sp),
           "mean_l2_norm": {m: round(float(np.linalg.norm(emb[m], axis=1).mean()), 4) for m in MODELS}}
    for prep in preps:
        if prep == "std":
            t = emb["teacher"]
            mu, sd = t[~te].mean(0), t[~te].std(0) + 1e-6
            X = {m: (emb[m] - mu) / sd for m in MODELS}  # teacher statistics (cross-space)
            own = {m: (emb[m] - emb[m][~te].mean(0)) / (emb[m][~te].std(0) + 1e-6) for m in MODELS}
        elif prep == "l2":
            X = own = {m: norm(emb[m]) for m in MODELS}
        elif prep == "raw":
            X = own = emb
        else:  # rawscaled
            t_norm = np.linalg.norm(emb["teacher"][~te], axis=1).mean()
            X = own = {m: emb[m] * (t_norm / np.linalg.norm(emb[m][~te], axis=1).mean()) for m in MODELS}
        # slice once: arrays built inside the generator are copied per dispatched task (OOM on ReefSet)
        X_tr, X_te = X["teacher"][~te], [X[m][te] for m in MODELS]
        cross = Parallel(n_jobs=jobs)(delayed(fit_apply_conv)(X_tr, Y[c][~te], X_te) for c in probe_sp)
        nonconv = {"teacher": sum(f for _, f in cross)}
        cross = [s for s, _ in cross]
        r = {f"cross_{m}": cmap([s[i] for s in cross]) for i, m in enumerate(MODELS)}
        for m in MODELS[1:]:
            O_tr, O_te = own[m][~te], [own[m][te]]
            sc = Parallel(n_jobs=jobs)(delayed(fit_apply_conv)(O_tr, Y[c][~te], O_te) for c in probe_sp)
            nonconv[m] = sum(f for _, f in sc)
            r[f"own_{m}"] = cmap([s[0][0] for s in sc])
        r["own_teacher"] = r.pop("cross_teacher")
        t0 = r["own_teacher"]
        r |= {f"ratio_{k}": round(v / t0, 3) for k, v in r.items() if k != "own_teacher"}
        r["fits_at_max_iter"] = nonconv
        out[prep] = r
        print(prep, r, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=["wabad", "watkins", "anuraset", "humbugdb", "reefset"])
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--preps", nargs="+", default=["std", "l2"], choices=["std", "l2", "raw", "rawscaled"],
                    help="WABAD's std numbers are archived in wabad_embprobe/embcompat results (--preps l2 saves hours)")
    ap.add_argument("--out", type=Path, default=RESULTS / "probe_l2_post.json")
    args = ap.parse_args()
    out_path = args.out
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for s in args.sets:
        print(s, flush=True)
        results[s] = results.get(s, {}) | readouts(*(wabad() if s == "wabad" else nonbird(s)), args.jobs, args.preps)
        out_path.write_text(json.dumps(results, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
