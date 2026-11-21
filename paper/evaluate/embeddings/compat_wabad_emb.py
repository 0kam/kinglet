# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""wabad_embcompat_v1: cross-space compatibility between Perch teacher and student embeddings.

Same windows, split, species filter and seeds as probe_wabad_emb.py. Readouts per student:
(a) field alignment: per-window cos(teacher, student) over all 60,158 WABAD windows
(b) cross-probe: logistic trained on teacher train-split embeddings (teacher standardization),
    applied unchanged to the student's test-split embeddings; cmAP vs teacher->teacher
(c) cross-retrieval: exemplars from one space, gallery in the other (k in {1,5}, 10 seeds)

Output: results/wabad_embcompat_results.json (--models / --out restrict the students and redirect
the output, e.g. for xsvar_v1 / xsabl_v1; the teacher baseline is always recomputed)

Usage: python evaluate/embeddings/compat_wabad_emb.py [--models B S XS XS_int8pub] [--out FILE]
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, RESULTS  # noqa: E402
from evaluate.embeddings.probe_wabad_emb import ALL_DIRS, EMB_DIRS, KS, MIN_POS, SEEDS, fit_probe, load_emb  # noqa: E402


def cross_retrieval(src_n, dst_n, pos_idx, recs, rng):
    out = {}
    for k in KS:
        ex = rng.choice(pos_idx, size=k, replace=False)
        keep = ~np.isin(recs, np.unique(recs[ex]))
        y = np.zeros(len(recs), bool)
        y[pos_idx] = True
        if not (y & keep).any():
            out[k] = np.nan
            continue
        out[k] = average_precision_score(y[keep], (dst_n[keep] @ src_n[ex].T).max(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    students = [n for n in EMB_DIRS if n != "teacher"]
    ap.add_argument("--models", nargs="+", default=students, choices=[n for n in ALL_DIRS if n != "teacher"])
    ap.add_argument("--out", type=Path, default=RESULTS / "wabad_embcompat_results.json")
    args = ap.parse_args()
    names = ["teacher"] + args.models

    meta = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    recs = meta["window_id"].str.rsplit("_", n=1).str[0].to_numpy()
    sp_all = sorted({c for x in meta["labels"] for c in x})
    Y = {c: np.zeros(len(meta), bool) for c in sp_all}
    for i, labs in enumerate(meta["labels"]):
        for c in labs:
            Y[c][i] = True
    species = [c for c in sp_all if Y[c].sum() >= MIN_POS and len(np.unique(recs[Y[c]])) >= 2]

    rng0 = np.random.default_rng(0)
    urec = np.unique(recs)
    te = np.isin(recs, list(set(rng0.choice(urec, size=int(0.3 * len(urec)), replace=False))))
    probe_sp = [c for c in species if Y[c][~te].any() and Y[c][te].any()]

    emb, emb_n = {}, {}
    for name in names + sorted({n.split("_int8")[0] for n in args.models if "_int8" in n} - set(names)):
        emb[name] = load_emb(name, meta["window_id"].tolist())
        emb_n[name] = emb[name] / np.linalg.norm(emb[name], axis=1, keepdims=True)

    # teacher-trained probes, shared by the baseline and every cross evaluation
    t = emb["teacher"]
    mu, sd = t[~te].mean(0), t[~te].std(0) + 1e-6
    X_tr = (t[~te] - mu) / sd
    tests = {name: (emb[name][te] - mu) / sd for name in names}
    scored = Parallel(n_jobs=8)(
        delayed(lambda ytr, Xs: [fit_probe(X_tr, ytr, X) for X in Xs])(
            Y[c][~te], list(tests.values())) for c in probe_sp)

    results = {}
    for name in args.models:
        cos = (emb_n["teacher"] * emb_n[name]).sum(1)
        res = {"field_cos_mean": round(float(cos.mean()), 4),
               "field_cos_median": round(float(np.median(cos)), 4),
               "field_cos_p10": round(float(np.percentile(cos, 10)), 4)}
        if "_int8" in name:  # mcuemb_v2: what the int8 feature path costs against the fp32 tier
            cos = (emb_n[name.split("_int8")[0]] * emb_n[name]).sum(1)
            res |= {"fp32_cos_mean": round(float(cos.mean()), 4),
                    "fp32_cos_p10": round(float(np.percentile(cos, 10)), 4),
                    "fp32_cos_min": round(float(cos.min()), 4)}
        for direction, (src, dst) in {"t2s": ("teacher", name), "s2t": (name, "teacher")}.items():
            ret = {k: [] for k in KS}
            for c in species:
                pos_idx = np.flatnonzero(Y[c])
                aps = [cross_retrieval(emb_n[src], emb_n[dst], pos_idx, recs,
                                       np.random.default_rng(zlib.crc32(f"{c}:{s}".encode())))
                       for s in range(SEEDS)]
                for k in KS:
                    ret[k].append(np.nanmean([a[k] for a in aps]))
            res |= {f"xretrieval_{direction}_k{k}_mAP": round(float(np.nanmean(ret[k])), 4) for k in KS}
        idx = list(tests).index(name)
        res["xprobe_cmAP"] = round(float(np.mean(
            [average_precision_score(Y[c][te], s[idx]) for c, s in zip(probe_sp, scored)])), 4)
        results[name] = res
        print(name, res, flush=True)
    results["teacher_probe_cmAP"] = round(float(np.mean(
        [average_precision_score(Y[c][te], s[0]) for c, s in zip(probe_sp, scored)])), 4)
    print("teacher->teacher probe baseline:", results["teacher_probe_cmAP"])
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
