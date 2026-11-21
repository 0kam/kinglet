# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""nonbird_v1: teacher vs Kinglet B/S/XS on public non-bird sets (rows nonbird_head_v1, nonbird_emb_v1
of the public experiment log).

Units: clip sets (watkins, humbugdb, reefset) -> one unit per clip: embedding = mean of its 5-s
window embeddings (Burns et al. 2025), class score = max over its windows. anuraset -> one unit per
5-s window of the 1-min recordings, labelled with every species whose strong-label box overlaps it.

(1) embedding, WABAD protocol (probe_wabad_emb.py / compat_wabad_emb.py): retrieval k=1,5 (10 draws,
    the exemplars' groups removed), one-vs-rest logistic probe (liblinear, C=1) on standardised
    embeddings with a group-disjoint 70/30 split (seed 0); cross-space retrieval (t2s, s2t) and the
    teacher-trained probe applied unchanged to student embeddings.
(2) published protocols, as bug checks: watkins/humbugdb BEANS split, multiclass logistic regression
    on standardised embeddings, test accuracy (Perch 2.0 paper Table 5); reefset k=16 per class,
    5 draws, logistic regression on L2-normalised embeddings, macro one-vs-rest ROC-AUC
    (Burns et al. 2025). Also the teacher-trained classifier applied to student embeddings.
(3) head: dataset classes with a Perch 2.0 class (classes.json) and >= HEAD_MIN_POS positive units:
    per-class AP of the teacher logit (and teacher softmax) vs the student's exact-fold softmax.

Output: results/nonbird_v1.json
Usage: python evaluate/nonbird/eval_nonbird.py [--sets watkins humbugdb anuraset reefset]
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
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DISTILL_ROOT, NONBIRD_ROOT, RELEASE_ASSETS, RESULTS  # noqa: E402
from evaluate.embeddings.compat_wabad_emb import cross_retrieval  # noqa: E402
from evaluate.embeddings.probe_wabad_emb import KS, MIN_POS, SEEDS, fit_probe, retrieval  # noqa: E402

DATA = NONBIRD_ROOT
EMB = DISTILL_ROOT / "nonbird_v1"
RELIABILITY = RELEASE_ASSETS / "kinglet-v0.3.2.reliability.csv"
MODELS = ["teacher", "B", "S", "XS"]
STUDENTS = MODELS[1:]
NON_SPECIES = {"non-mosquito", "others"}  # humbugdb BEANS labels that are not a species
HEAD_MIN_POS = 5
REEF_K, REEF_DRAWS = 16, 5


def anuraset_labels(ids: np.ndarray, t_start: np.ndarray) -> list[list[str]]:
    """Species whose strong-label box [on, off) overlaps the window [t, t + 5) (WABAD rule)."""
    boxes = {}
    for f in (DATA / "anuraset/strong").rglob("*.txt"):
        rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]  # 2 rows carry a trailing tab
        boxes[f.stem] = [(float(a), float(b), c.strip().rsplit("_", 1)[0]) for a, b, c in rows]
    return [sorted({c for a, b, c in boxes[r] if a < t + 5 and b > t}) for r, t in zip(ids, t_start)]


def load_units(name: str, man: pd.DataFrame) -> tuple[dict, list[list[str]], np.ndarray]:
    """Per model {emb, scores...} at unit level, the unit labels and the unit groups."""
    z = {m: dict(np.load(EMB / f"{name}_{m}.npz")) for m in MODELS}
    ids, t0 = z["teacher"]["ids"], z["teacher"]["t_start"]
    for m in STUDENTS:
        assert (z[m]["ids"] == ids).all() and np.allclose(z[m]["t_start"], t0), f"{name}/{m}: windows differ"
    if name == "anuraset":
        group = dict(zip(man["item_id"], man["group"]))
        return ({m: {k: v.astype(np.float32) for k, v in z[m].items() if k in ("emb", "logit", "softmax")}
                 for m in MODELS}, anuraset_labels(ids, t0), np.array([group[r] for r in ids]))
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    assert ids[starts].tolist() == man["item_id"].astype(str).tolist(), f"{name}: item order differs from manifest"
    n = np.diff(np.r_[starts, len(ids)])[:, None]
    units = {}
    for m in MODELS:
        u = {"emb": np.add.reduceat(z[m]["emb"].astype(np.float64), starts, axis=0) / n}
        for k in ("logit", "softmax"):
            if k in z[m] and z[m][k].shape[1]:
                u[k] = np.maximum.reduceat(z[m][k], starts, axis=0)
        units[m] = {k: v.astype(np.float32) for k, v in u.items()}
    units["_field_cos"] = {m: z[m]["emb"].astype(np.float32) for m in MODELS}  # window level, for the bug check
    units["_first"] = {m: {"emb": z[m]["emb"][starts].astype(np.float32)} for m in MODELS}  # first 5 s only
    return units, [[c] for c in man["label"]], man["group"].astype(str).to_numpy()


def fit_apply(X_tr, y_tr, X_tests: list) -> list:
    """fit_probe (liblinear, C=1) fitted once and applied to several test matrices."""
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(X_tr, y_tr)
    return [clf.decision_function(X) for X in X_tests]


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def embedding_readouts(units: dict, Y: dict, groups: np.ndarray, species: list[str], jobs: int) -> dict:
    """WABAD protocol: within-space retrieval/probe per model, cross-space per student."""
    rng0 = np.random.default_rng(0)
    ug = np.unique(groups)
    te = np.isin(groups, list(set(rng0.choice(ug, size=int(0.3 * len(ug)), replace=False))))
    probe_sp = [c for c in species if Y[c][~te].any() and Y[c][te].any()]
    emb = {m: units[m]["emb"] for m in MODELS}
    emb_n = {m: norm(e) for m, e in emb.items()}
    out = {"n_species_retrieval": len(species), "n_species_probe": len(probe_sp),
           "n_units": len(groups), "n_groups": len(ug), "n_test_units": int(te.sum())}

    def ret(src, dst):
        r = {k: [] for k in KS}
        for c in species:
            pos = np.flatnonzero(Y[c])
            aps = [(retrieval(emb_n[src], pos, groups, rng) if src == dst else
                    cross_retrieval(emb_n[src], emb_n[dst], pos, groups, rng))
                   for rng in (np.random.default_rng(zlib.crc32(f"{c}:{s}".encode())) for s in range(SEEDS))]
            for k in KS:
                r[k].append(np.nanmean([a[k] for a in aps]))
        return {k: float(np.nanmean(v)) for k, v in r.items()}

    t = emb["teacher"]
    mu, sd = t[~te].mean(0), t[~te].std(0) + 1e-6
    X_tr = (t[~te] - mu) / sd
    xtests = [(emb[m][te] - mu) / sd for m in MODELS]  # teacher standardisation for the cross probe
    xscored = Parallel(n_jobs=jobs)(delayed(fit_apply)(X_tr, Y[c][~te], xtests) for c in probe_sp)
    for m in MODELS:
        r = ret(m, m)
        e = emb[m]
        mu_m, sd_m = e[~te].mean(0), e[~te].std(0) + 1e-6
        E_tr, E_te = (e[~te] - mu_m) / sd_m, (e[te] - mu_m) / sd_m  # once: per-task copies blow up RAM
        scores = [s[0] for s in xscored] if m == "teacher" else Parallel(n_jobs=jobs)(  # teacher: same fit
            delayed(fit_probe)(E_tr, Y[c][~te], E_te) for c in probe_sp)
        res = {f"retrieval_k{k}_mAP": round(r[k], 4) for k in KS}
        res["probe_cmAP"] = round(float(np.mean([average_precision_score(Y[c][te], s)
                                                 for c, s in zip(probe_sp, scores)])), 4)
        if m != "teacher":
            for d, (src, dst) in {"t2s": ("teacher", m), "s2t": (m, "teacher")}.items():
                res |= {f"xretrieval_{d}_k{k}_mAP": round(v, 4) for k, v in ret(src, dst).items()}
            i = MODELS.index(m)
            res["xprobe_cmAP"] = round(float(np.mean([average_precision_score(Y[c][te], s[i])
                                                      for c, s in zip(probe_sp, xscored)])), 4)
        out[m] = res
        print(m, res, flush=True)
    out["chance_cmAP_probe"] = round(float(np.mean([Y[c][te].mean() for c in probe_sp])), 4)
    for m in STUDENTS:
        t_, s_ = out["teacher"], out[m]
        rat = {k: round(s_[k] / t_[k], 3) for k in ("retrieval_k1_mAP", "retrieval_k5_mAP", "probe_cmAP")}
        rat["xretrieval_t2s_k1_mAP"] = round(s_["xretrieval_t2s_k1_mAP"] / t_["retrieval_k1_mAP"], 3)
        rat["xretrieval_s2t_k1_mAP"] = round(s_["xretrieval_s2t_k1_mAP"] / t_["retrieval_k1_mAP"], 3)
        rat["xprobe_cmAP"] = round(s_["xprobe_cmAP"] / t_["probe_cmAP"], 3)
        rat["mean_within"] = round(float(np.mean([rat["retrieval_k1_mAP"], rat["retrieval_k5_mAP"],
                                                  rat["probe_cmAP"]])), 3)
        out[f"ratio_{m}"] = rat
    return out


def beans_accuracy(units: dict, labels: np.ndarray, split: np.ndarray) -> dict:
    """BEANS classification protocol: train split -> test accuracy (multiclass logistic regression)."""
    tr, te = split == "train", split == "test"
    t = units["teacher"]["emb"]
    mu, sd = t[tr].mean(0), t[tr].std(0) + 1e-6
    clf_t = LogisticRegression(max_iter=2000).fit((t[tr] - mu) / sd, labels[tr])
    out = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_classes": len(np.unique(labels))}
    for m in MODELS:
        e = units[m]["emb"]
        mu_m, sd_m = e[tr].mean(0), e[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000).fit((e[tr] - mu_m) / sd_m, labels[tr])
        out[m] = {"accuracy": round(float((clf.predict((e[te] - mu_m) / sd_m) == labels[te]).mean()), 4)}
        if m != "teacher":
            out[m]["x_accuracy"] = round(float((clf_t.predict((e[te] - mu) / sd) == labels[te]).mean()), 4)
    return out


def reef_auc(units: dict, labels: np.ndarray) -> dict:
    """Burns et al. 2025: k examples per class, rest held out, macro one-vs-rest ROC-AUC, 5 draws."""
    cls = [c for c in np.unique(labels) if (labels == c).sum() > REEF_K]
    keep = np.isin(labels, cls)
    lab = labels[keep]
    emb = {m: norm(units[m]["emb"][keep]) for m in MODELS}
    res = {m: [] for m in MODELS} | {f"x_{m}": [] for m in STUDENTS}
    for d in range(REEF_DRAWS):
        rng = np.random.default_rng(d)
        tr = np.zeros(len(lab), bool)
        for c in cls:
            tr[rng.choice(np.flatnonzero(lab == c), REEF_K, replace=False)] = True
        fits = {}
        for m in MODELS:
            fits[m] = LogisticRegression(max_iter=2000).fit(emb[m][tr], lab[tr])
            res[m].append(roc_auc_score(lab[~tr], fits[m].predict_proba(emb[m][~tr]), multi_class="ovr",
                                        labels=fits[m].classes_))
        for m in STUDENTS:
            res[f"x_{m}"].append(roc_auc_score(lab[~tr], fits["teacher"].predict_proba(emb[m][~tr]),
                                               multi_class="ovr", labels=fits["teacher"].classes_))
    return {"k": REEF_K, "draws": REEF_DRAWS, "n_classes": len(cls),
            **{m: round(float(np.mean(v)), 4) for m, v in res.items()}}


def head_readout(units: dict, Y: dict, cls: list[dict], reliability: pd.DataFrame) -> dict:
    mapped = [c for c in cls if c["perch_idx"] >= 0]
    col = {c["label"]: i for i, c in enumerate(mapped)}  # column order of perch_idx.json
    present = [c for c in cls if c["label"] in Y]
    rows = []
    for c in mapped:
        y = Y.get(c["label"])
        if y is None or y.sum() < HEAD_MIN_POS:
            continue
        j = col[c["label"]]
        r = {"label": c["label"], "sci": c["sci"], "perch_idx": c["perch_idx"], "n_pos": int(y.sum()),
             "prevalence": round(float(y.mean()), 5),
             "n_train_recordings": int(reliability.loc[c["perch_idx"], "n_train_recordings"]),
             "teacher_logit": average_precision_score(y, units["teacher"]["logit"][:, j]),
             "teacher_softmax": average_precision_score(y, units["teacher"]["softmax"][:, j])}
        r |= {m: average_precision_score(y, units[m]["softmax"][:, j]) for m in STUDENTS}
        rows.append({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    out = {"n_classes_in_data": len(present),
           "n_mapped": sum(c["perch_idx"] >= 0 for c in present),
           "n_mapped_with_train_recordings": sum(
               int(reliability.loc[c["perch_idx"], "n_train_recordings"]) > 0 for c in present if c["perch_idx"] >= 0),
           "n_scored": len(rows), "min_pos": HEAD_MIN_POS}
    if not rows:
        return out
    df = pd.DataFrame(rows)
    out["macro_AP"] = {k: round(float(df[k].mean()), 4) for k in ["prevalence", "teacher_logit", "teacher_softmax", *STUDENTS]}
    for m in STUDENTS:
        out[f"ratio_{m}"] = {
            "macro_AP_vs_teacher_logit": round(float(df[m].mean() / df["teacher_logit"].mean()), 3),
            "macro_AP_vs_teacher_softmax": round(float(df[m].mean() / df["teacher_softmax"].mean()), 3),
            "chance_corrected_vs_teacher_logit": round(float((df[m] - df["prevalence"]).mean()
                                                             / (df["teacher_logit"] - df["prevalence"]).mean()), 3),
            "median_per_class_ratio": round(float((df[m] / df["teacher_logit"]).median()), 3)}
    out["per_class"] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=["watkins", "humbugdb", "anuraset", "reefset"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--beans-only", action="store_true", help="recompute only the BEANS block of existing results")
    args = ap.parse_args()
    reliability = pd.read_csv(RELIABILITY).set_index("vocab_index")
    out_path = RESULTS / "nonbird_v1.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.sets:
        man = pd.read_parquet(DATA / name / "manifest.parquet")
        cls = json.loads((DATA / name / "classes.json").read_text())
        units, labels, groups = load_units(name, man)
        win = units.pop("_field_cos", {m: units[m]["emb"] for m in MODELS})
        first = units.pop("_first", None)
        if name in ("watkins", "humbugdb"):
            lab, split = man["label"].to_numpy(), man["split"].to_numpy()
            beans = beans_accuracy(units, lab, split)
            # bug-check diagnosis: BEANS crops clips to 3 s, so also score the first 5-s window alone
            beans["first_window_only"] = beans_accuracy(first, lab, split)
            print("beans", beans, flush=True)
            if args.beans_only:
                results[name]["beans"] = beans
                out_path.write_text(json.dumps(results, indent=1))
                continue
        if args.beans_only:
            continue
        res = {"field_cos_mean": {m: round(float((norm(win["teacher"]) * norm(win[m])).sum(1).mean()), 4)
                                  for m in STUDENTS}}
        print(name, res, flush=True)
        all_lab = sorted({c for x in labels for c in x})
        Y = {c: np.array([c in x for x in labels]) for c in all_lab}
        species = [c for c in all_lab if c not in NON_SPECIES and Y[c].sum() >= MIN_POS
                   and len(np.unique(groups[Y[c]])) >= 2]
        res["embedding"] = embedding_readouts(units, Y, groups, species, args.jobs)
        if name in ("watkins", "humbugdb"):
            res["beans"] = beans
        if name == "reefset":
            res["reef_k16"] = reef_auc(units, man["label"].to_numpy())
            print("reef_k16", res["reef_k16"], flush=True)
        res["head"] = head_readout(units, Y, cls, reliability)
        print("head", {k: v for k, v in res["head"].items() if k != "per_class"}, flush=True)
        results[name] = res
        out_path.write_text(json.dumps(results, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
