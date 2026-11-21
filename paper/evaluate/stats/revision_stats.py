# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""revstats_v1: statistics of the Kinglet paper, from the cached per-window scores.

Every score comes from the caches behind the paper (figures/build_figure_data.load_set: Perch 2.0
class scores, exact-fold probabilities of B/S/XS, BirdNET logits). The one addition is the teacher's
log-sum-exp over all 14,795 classes, computed by re-running Perch 2.0 (the same GPU build as the cached
teacher runs) with teacher/embed_perch.py (revstats_lse.sh -> DISTILL_ROOT/revstats_lse); NBP, whose cached
teacher run used the CPU build, with the CPU build. Items:

  bootstrap   paired species bootstrap of student/teacher cmAP per benchmark; WABAD site bootstrap
  agreement   per-species AP correlation with the teacher; site-level r
  groups      between-group differences of the training-recording group ratios; 'none' intervals per tier
  softmax     teacher scored by its full softmax
  common      cmAP on species that have a class in every model
  ordering    tier ordering per benchmark; teacher WABAD cmAP at full precision
  windows     WABAD window-label rule and the share of labels resting on short overlaps
  mcurate     per-species AP change of XS under the emulated 31,250 Hz capture
  continents  per-site student/teacher ratio by continent
  siteprobe   linear probe with a site-disjoint split
  threshold   recall and precision at p > 0.1
  nbp         facts about BirdSet NBP

Usage: python evaluate/stats/revision_stats.py [item ...]   # default: all; results merge into results/revision_stats.json
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT as ROOT  # noqa: E402
from common.config import DISTILL_ROOT, FIGURE_DATA, RESULTS, WABAD_SRC  # noqa: E402
from common.scoring import CACHE, SETS, load_parts  # noqa: E402
from figures.build_figure_data import BENCHMARKS, LATE_BENCHMARKS, MODELS, RELIABILITY, load_set  # noqa: E402

OUT = RESULTS / "revision_stats.json"
DATA = FIGURE_DATA
LSE_DIR = DISTILL_ROOT / "revstats_lse"
ALL_SETS = (*BENCHMARKS, *LATE_BENCHMARKS)  # NBP: birdsetnbp_v1; the group table keeps the eight
TEST_SETS = ("NES", "PER", "SNE", "SSW", "UHH", "NBP")  # BirdSet sets first scored after the models were frozen
TIERS = ("B", "S", "XS")
N_BOOT = 2000
THRESH = 0.1
GROUPS = {"high": ">=100", "medium": "10-99", "low": "1-9", "softlabel_only": "none"}


def ci(x) -> list[float]:
    return [round(float(v), 4) for v in np.nanpercentile(x, [2.5, 97.5])]


@lru_cache(maxsize=None)
def bench(name: str) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """load_set plus 'teacher_softmax': log of the teacher's full-vocabulary softmax (cached logit - lse)."""
    scores, y, meta = load_set(name)
    r = load_parts(LSE_DIR / name, keys=("ids", "logits", "lse"))
    assert r["ids"].tolist() == meta.index.tolist(), f"{name}: re-run window order"
    cached = load_parts(ROOT / SETS[name][-1], keys=("logits",))["logits"].astype(np.float32)
    assert np.array_equal(np.isnan(cached), np.isnan(r["logits"]))
    d = np.nanmax(np.abs(cached - r["logits"].astype(np.float32)))
    # GPU runs differ by numeric noise: <= 0.0078 (one float16 quantum at 8 <= |logit| < 16) on most sets,
    # up to 0.0195 on POW/HSN; NBP (CPU build both times) is identical
    assert d <= 0.025, f"{name}: re-run logits differ from the cache by {d}"
    codes = cache_codes(name)
    tcodes = [c["species"] for c in json.loads((ROOT / SETS[name][-1] / "classes.json").read_text())]
    logp = cached[:, [tcodes.index(c) for c in codes]] - r["lse"][:, None]
    scores["teacher_softmax"] = np.nan_to_num(logp, nan=np.nanmin(logp) - 1)
    return scores, y, meta


def cache_codes(name: str) -> list[str]:
    return np.load(CACHE / f"{MODELS['B']}_{name}.npz", allow_pickle=True)["codes"].tolist()


@lru_cache(maxsize=None)
def per_species() -> pd.DataFrame:
    """One row per (benchmark, annotated species): positives, training-recording group, AP of every model."""
    tier = pd.read_csv(RELIABILITY).set_index("vocab_index")["reliability_tier"]
    rows = []
    for name in ALL_SETS:
        scores, y, _ = bench(name)
        codes = cache_codes(name)
        perch = {c["species"]: c["perch_idx"] for c in json.loads((ROOT / SETS[name][-1] / "classes.json").read_text())}
        has = {m: set(codes) - set(np.load(CACHE / f"{c}_{name}.npz", allow_pickle=True)["unmatched"].tolist())
               for m, c in MODELS.items()}
        has["teacher"] = {c for c in codes if perch[c] is not None and perch[c] >= 0}
        for j in np.flatnonzero(y.any(0)):
            c = codes[j]
            rows.append({"benchmark": name, "species": c, "positives": int(y[:, j].sum()),
                         "group": tier[perch[c]] if c in has["teacher"] else "no_teacher_class",
                         "in_all_models": all(c in h for h in has.values())}
                        | {m: average_precision_score(y[:, j], s[:, j]) for m, s in scores.items()})
    df = pd.DataFrame(rows)
    reg = pd.read_csv(DATA / "benchmarks.csv").set_index(["benchmark", "model"])["cmAP"]
    for (b, m), v in reg.items():  # bug check (i)
        assert abs(df.loc[df["benchmark"] == b, m].mean() - v) < 5.1e-5, (b, m)
    return df


def boot_ratio(student: np.ndarray, teacher: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    pick = rng.integers(0, len(teacher), (N_BOOT, len(teacher)))  # the same species for both models
    s, t = student[pick].mean(1), teacher[pick].mean(1)
    return s / t, s - t


def weighted_ap(s: np.ndarray, yj: np.ndarray, site: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """AP of one species for every row of `mult` (resample x site multiplicity), i.e. as if each window were
    repeated as often as its site was drawn. Tied scores are one threshold, as in sklearn."""
    order = np.argsort(-s, kind="stable")
    ss, ys, st = s[order], yj[order], site[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]  # last position of each tie group
    gend = ends[np.searchsorted(ends, np.arange(len(ss)))]
    g = np.unique(gend[ys])  # thresholds at which recall changes
    b = np.searchsorted(g, np.arange(len(ss)))  # windows ranked within threshold g count from g on
    k = mult.shape[1]

    def counts(idx):  # (len(g), sites): windows of each site ranked at or above each threshold
        return np.bincount(b[idx] * k + st[idx], minlength=(len(g) + 1) * k).reshape(-1, k)[:-1].cumsum(0)

    n, tp = counts(slice(None)) @ mult.T, counts(ys) @ mult.T
    prec = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.diff(tp, axis=0, prepend=0) * prec).sum(0) / tp[-1]  # nan: species not drawn


def bootstrap() -> dict:
    """Paired species bootstrap of the student/teacher cmAP ratio and difference per benchmark and tier;
    the mean ratio over the five test-only BirdSet sets (species resampled within each set); WABAD bootstrap over
    sites (sites drawn with replacement, cmAP over the species present in the resample)."""
    df, rng, out = per_species(), np.random.default_rng(0), {"species": {}}
    for name in ALL_SETS:
        d = df[df["benchmark"] == name]
        for m in TIERS:
            ratio, diff = boot_ratio(d[m].to_numpy(), d["teacher"].to_numpy(), rng)
            out["species"][f"{name} {m}"] = {
                "n_species": len(d), "ratio": round(d[m].mean() / d["teacher"].mean(), 4), "ratio_ci": ci(ratio),
                "diff": round(d[m].mean() - d["teacher"].mean(), 4), "diff_ci": ci(diff)}
    for m in TIERS:
        sets = [df[df["benchmark"] == n] for n in TEST_SETS]
        boots = np.mean([boot_ratio(d[m].to_numpy(), d["teacher"].to_numpy(), rng)[0] for d in sets], axis=0)
        out["species"][f"test6_mean {m}"] = {"ratio": round(np.mean([d[m].mean() / d["teacher"].mean() for d in sets]), 4),
                                             "ratio_ci": ci(boots)}
    scores, y, meta = bench("WABAD")
    sites, site = np.unique(meta["site"].to_numpy(), return_inverse=True)
    mult = np.vstack([np.ones(len(sites)),
                      np.random.default_rng(1).multinomial(len(sites), np.full(len(sites), 1 / len(sites)), N_BOOT)])
    cm = {}
    for m in ("teacher", *TIERS):
        aps = np.array([weighted_ap(scores[m][:, j], y[:, j], site, mult) for j in np.flatnonzero(y.any(0))])
        reg = pd.read_csv(DATA / "benchmarks.csv").set_index(["benchmark", "model"]).loc[("WABAD", m), "cmAP"]
        assert abs(np.mean(aps[:, 0]) - df.loc[df["benchmark"] == "WABAD", m].mean()) < 1e-9  # bug check (ii)
        assert abs(np.mean(aps[:, 0]) - reg) < 5.1e-5
        cm[m] = np.nanmean(aps[:, 1:], axis=0)
    out["wabad_sites"] = {m: {"ratio": round(df.loc[df["benchmark"] == "WABAD", m].mean()
                                             / df.loc[df["benchmark"] == "WABAD", "teacher"].mean(), 4),
                              "ratio_ci": ci(cm[m] / cm["teacher"]), "diff_ci": ci(cm[m] - cm["teacher"])}
                          for m in TIERS}
    out["wabad_sites"]["teacher_cmAP_ci"] = ci(cm["teacher"])
    return out


def agreement() -> dict:
    """Correlation of per-species AP with the teacher's (Pearson r, Spearman rho), per benchmark and pooled
    over all species-benchmark pairs; site-level r of pooled AUPRC from wabad_per_site.csv."""
    df, out = per_species(), {"species": {}}
    for scope, d in [(n, df[df["benchmark"] == n]) for n in ALL_SETS] + [("pooled", df)]:
        out["species"][scope] = {m: {"r": round(pearsonr(d[m], d["teacher"])[0], 3),
                                     "rho": round(spearmanr(d[m], d["teacher"])[0], 3)}
                                 for m in (*TIERS, "birdnet_v3", "birdnet_v24")} | {"n_species": len(d)}
    site = pd.read_csv(DATA / "wabad_per_site.csv")
    out["sites"] = {m: {"r": round(pearsonr(site[m], site["teacher"])[0], 3),
                        "diff_median": round((site[m] - site["teacher"]).median(), 4),
                        "diff_range": [round((site[m] - site["teacher"]).min(), 4),
                                       round((site[m] - site["teacher"]).max(), 4)]}
                    for m in (*TIERS, "birdnet_v3", "birdnet_v24")}
    return out


def groups() -> dict:
    """Training-recording groups (species-benchmark pairs pooled over the eight benchmarks): bootstrap intervals of the
    difference in student/teacher ratio between each group and the >=100 group (groups resampled
    independently); and the per-tier intervals of the 'none' group as tier_stratified.csv holds them."""
    df, rng, out = per_species(), np.random.default_rng(0), {"diff_vs_high": {}}
    df = df[df["benchmark"].isin(BENCHMARKS)]  # the group table covers the eight benchmarks without NBP
    ts = pd.read_csv(DATA / "tier_stratified.csv")
    for m in TIERS:
        boots = {}
        for gname in GROUPS:
            d = df[df["group"] == gname]
            boots[gname] = boot_ratio(d[m].to_numpy(), d["teacher"].to_numpy(), rng)[0]
            row = ts[(ts["scope"] == "pooled") & (ts["tier"] == gname) & (ts["model"] == m)].iloc[0]
            assert abs(d[m].mean() / d["teacher"].mean() - row["ratio"]) < 5e-5  # bug check (vi)
        for gname in ("medium", "low", "softlabel_only"):
            d, h = df[df["group"] == gname], df[df["group"] == "high"]
            out["diff_vs_high"][f"{GROUPS[gname]} {m}"] = {
                "diff": round(d[m].mean() / d["teacher"].mean() - h[m].mean() / h["teacher"].mean(), 4),
                "ci": ci(boots[gname] - boots["high"])}
    none = ts[ts["tier"] == "softlabel_only"]
    out["none_group_ci"] = {f"{r.scope} {r.model}": [round(r.ratio, 3), round(r.ratio_lo, 3), round(r.ratio_hi, 3)]
                            for r in none.itertuples()}
    out["none_group_n"] = int((df["group"] == "softlabel_only").sum())
    out["none_group_benchmarks"] = sorted(df.loc[df["group"] == "softlabel_only", "benchmark"].unique())
    return out


def softmax() -> dict:
    """Teacher read as its full 14,795-class softmax (the students' training target) instead of its class
    scores: cmAP, pooled AUPRC and the student/teacher ratios under both readings."""
    df, out = per_species(), {}
    for name in ALL_SETS:
        scores, y, _ = bench(name)
        d = df[df["benchmark"] == name]
        pooled = {m: average_precision_score(y.ravel(), scores[m].ravel()) for m in ("teacher", "teacher_softmax", *TIERS)}
        out[name] = {"cmAP_raw": round(d["teacher"].mean(), 4), "cmAP_softmax": round(d["teacher_softmax"].mean(), 4),
                     "pooled_raw": round(pooled["teacher"], 4), "pooled_softmax": round(pooled["teacher_softmax"], 4)}
        for m in TIERS:
            out[name][m] = {"cmAP_ratio_raw": round(d[m].mean() / d["teacher"].mean(), 4),
                            "cmAP_ratio_softmax": round(d[m].mean() / d["teacher_softmax"].mean(), 4),
                            "pooled_ratio_raw": round(pooled[m] / pooled["teacher"], 4),
                            "pooled_ratio_softmax": round(pooled[m] / pooled["teacher_softmax"], 4)}
    for m in TIERS:
        out[f"test6_mean {m}"] = {k: round(np.mean([out[n][m][k] for n in TEST_SETS]), 4)
                                  for k in ("cmAP_ratio_raw", "cmAP_ratio_softmax")}
    return out


def common() -> dict:
    """cmAP on the species that have a class in every compared model (teacher, B/S/XS, both BirdNETs)."""
    df, out = per_species(), {}
    for name in ALL_SETS:
        d = df[df["benchmark"] == name]
        c = d[d["in_all_models"]]
        out[name] = {"n_species": len(d), "n_common": len(c)} | {
            m: {"cmAP_all": round(d[m].mean(), 4), "cmAP_common": round(c[m].mean(), 4)}
            for m in ("teacher", *TIERS, "birdnet_v3", "birdnet_v24")} | {
            f"{m}_ratio_common": round(c[m].mean() / c["teacher"].mean(), 4) for m in TIERS}
    return out


def ordering() -> dict:
    """Whether cmAP and pooled AUPRC fall from B to S to XS on each benchmark; teacher WABAD cmAP."""
    bm = pd.read_csv(DATA / "benchmarks.csv")
    out = {}
    for metric in ("cmAP", "pooled_auprc"):
        p = bm.pivot(index="benchmark", columns="model", values=metric)
        out[metric] = {b: bool(p.loc[b, "B"] > p.loc[b, "S"] > p.loc[b, "XS"]) for b in ALL_SETS}
        out[f"{metric}_exceptions"] = {b: {m: round(p.loc[b, m], 4) for m in TIERS}
                                       for b, ok in out[metric].items() if not ok}
    df = per_species()
    t = df.loc[df["benchmark"] == "WABAD", "teacher"].mean()
    out["teacher_wabad_cmAP"] = round(float(t), 6)
    out["teacher_wabad_cmAP_3dp"] = f"{t:.3f}"
    out["printed"] = {"Table 1 (table1_tiers.csv -> build_docx)": "0.291", "Table S1 (si_tables.md)": "0.290"}
    return out


def wabad_boxes() -> pd.DataFrame:
    """WABAD annotations with eBird codes, by the rule of data_prep/prepare_wabad.py."""
    from data_prep.prepare_wabad import SYNONYMS, TAXONOMY

    tax = pd.read_csv(TAXONOMY)
    tax = tax[tax["CATEGORY"] == "species"]
    sci2code = dict(zip(tax["SCI_NAME"], tax["SPECIES_CODE"]))
    ann = pd.read_csv(WABAD_SRC / "Pooled annotations.csv", encoding="latin-1")
    ann["code"] = ann["Species"].map(lambda s: SYNONYMS.get(s) or sci2code[s])
    return ann


def windows() -> dict:
    """Rule (data_prep/prepare_wabad.py): a window [5k, 5k + 5) gets a species if any of its boxes [begin, end)
    overlaps it by any positive amount; no minimum overlap; trailing partial windows of a recording are dropped.
    Here: how many window labels rest only on short overlaps (the longest overlap of the species' boxes)."""
    ann = wabad_boxes()
    man = pd.read_parquet(ROOT / SETS["WABAD"][0])
    b, e = ann["Begin_Time_(s)"].to_numpy(), ann["End_Time_(s)"].to_numpy()
    k_lo = (b // 5).astype(int)
    k_hi = ((np.maximum(e, b + 1e-6) - 1e-9) // 5).astype(int)
    rep = k_hi - k_lo + 1
    a = ann.loc[ann.index.repeat(rep)].copy()
    a["k"] = np.concatenate([np.arange(lo, hi + 1) for lo, hi in zip(k_lo, k_hi)])
    a["overlap"] = np.minimum(a["End_Time_(s)"], 5 * a["k"] + 5) - np.maximum(a["Begin_Time_(s)"], 5 * a["k"])
    a["window_id"] = a["Recording"].str[:-4] + "_" + a["k"].map("{:02d}".format)
    a = a[a["window_id"].isin(set(man["window_id"]))]
    lab = a.groupby(["window_id", "code"])["overlap"].max()
    n_labels = int(man["labels"].map(len).sum())
    assert len(lab) == n_labels, (len(lab), n_labels)
    dur = ann["End_Time_(s)"] - ann["Begin_Time_(s)"]
    return {"rule": "any positive overlap of an annotation box [begin, end) with the window [5k, 5k+5); "
                    "a box ending exactly at 5k does not reach window k; no minimum overlap; a box spanning a "
                    "boundary labels both windows; the trailing partial window of each recording is discarded",
            "n_window_labels": n_labels,
            "share_longest_overlap_below_s": {str(t): round(float((lab < t).mean()), 4) for t in (0.05, 0.1, 0.25, 0.5)},
            "box_duration_median_s": round(float(dur.median()), 3)}


def mcurate() -> dict:
    """Per-species AP change of fp32 XS on WABAD when the 31,250 Hz capture is read as 32 kHz
    (mcurate_v1b rate arm minus the plain exact-fold scores), against the species' median annotated frequency."""
    base = np.load(CACHE / f"{MODELS['XS']}_WABAD.npz", allow_pickle=True)
    rate = np.load(CACHE / "mcurate_b_rate_WABAD.npz", allow_pickle=True)
    _, y, meta = bench("WABAD")
    assert base["ids"].tolist() == rate["ids"].tolist() == meta.index.tolist()
    codes = base["codes"].tolist()
    keep = np.flatnonzero(y.any(0))
    ap = {k: np.array([average_precision_score(y[:, j], z["scores"][:, j]) for j in keep])
          for k, z in (("plain", base), ("rate", rate))}
    reg = json.loads((RESULTS / "mcurate_v1b.json").read_text())
    assert abs(ap["plain"].mean() - reg["baseline"]["cmAP"]) < 1e-9 and abs(ap["rate"].mean() - reg["rate"]["cmAP"]) < 1e-9
    ann = wabad_boxes()
    ann["centre"] = (ann["Low_Freq_(Hz)"] + ann["High_Freq_(Hz)"]) / 2
    freq = ann.groupby("code")[["centre", "High_Freq_(Hz)"]].median()
    d = pd.DataFrame({"species": [codes[j] for j in keep], "positives": y[:, keep].sum(0),
                      "plain": ap["plain"], "rate": ap["rate"]})
    d["delta"] = d["rate"] - d["plain"]
    d = d.join(freq, on="species")
    out = {}
    for scope, x in (("all", d), (">=10 positives", d[d["positives"] >= 10])):
        q = pd.qcut(x["centre"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
        out[scope] = {"n_species": len(x), "median": round(x["delta"].median(), 4), "mean": round(x["delta"].mean(), 4),
                      "p5": round(x["delta"].quantile(0.05), 4), "p95": round(x["delta"].quantile(0.95), 4),
                      "share_drop_gt_0.05": round(float((x["delta"] < -0.05).mean()), 4),
                      "share_gain_gt_0.05": round(float((x["delta"] > 0.05).mean()), 4),
                      "spearman_delta_vs_centre_freq": [round(v, 3) for v in spearmanr(x["delta"], x["centre"], nan_policy="omit")],
                      "spearman_delta_vs_high_freq": [round(v, 3) for v in spearmanr(x["delta"], x["High_Freq_(Hz)"], nan_policy="omit")],
                      "median_delta_by_centre_quartile": x.groupby(q, observed=True)["delta"].median().round(4).to_dict(),
                      "largest_drops": x.nsmallest(10, "delta")[["species", "positives", "plain", "rate", "delta", "centre"]]
                      .round(4).to_dict("records")}
    return out


CONTINENT = {"Asia": ("Indonesia", "Taiwan", "Vietnam", "China (mainland)"),
             "Europe": ("Spain", "France", "Scotland", "Greece", "Germany", "Poland", "Cyprus", "Sweden", "Ukraine"),
             "Africa": ("Cameroon", "Burkina Faso", "Uganda", "Guinea-Bissau"),
             "North and Central America": ("Canada", "Mexico", "USA", "Costa Rica", "Guatemala", "Dominican Republic"),
             "South America": ("Brazil", "Colombia", "Argentina"),
             "Oceania": ("New Caledonia", "Hawaii (USA)", "New Zealand")}


def continents() -> dict:
    """Student/teacher ratio of per-site pooled AUPRC by continent; Asia vs the rest with a
    bootstrap interval over sites. Cyprus counted as Europe, Hawai'i as Oceania."""
    site = pd.read_csv(DATA / "wabad_per_site.csv")
    cont = {c: k for k, cs in CONTINENT.items() for c in cs}
    site["continent"] = site["country"].map(cont)
    assert site["continent"].notna().all()
    rng, out = np.random.default_rng(0), {}
    asia = site["continent"] == "Asia"
    for m in TIERS:
        r = site[m] / site["teacher"]
        a, o = r[asia].to_numpy(), r[~asia].to_numpy()
        boot = (a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1) - o[rng.integers(0, len(o), (N_BOOT, len(o)))].mean(1))
        out[m] = {"by_continent": {k: {"n_sites": int(len(g)), "mean_ratio": round(g.mean(), 4), "median_ratio": round(g.median(), 4)}
                                   for k, g in r.groupby(site["continent"])},
                  "asia_minus_rest_mean": round(a.mean() - o.mean(), 4), "ci": ci(boot)}
    out["asia_sites"] = site.loc[asia, "site"].tolist()
    return out


def fit_probes(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, c: float = 1.0) -> np.ndarray:
    """One-vs-rest logistic regression for all species at once: the objective of probe_wabad_emb.fit_probe
    (liblinear, L2, C = 1, intercept as a penalised constant feature) minimised to convergence by L-BFGS on the GPU.
    The objective is a sum over species, so the joint minimum is each species' own. Returns test decision values."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aug = lambda x: torch.cat([torch.from_numpy(x).double(), torch.ones(len(x), 1, dtype=torch.float64)], 1).to(dev)  # noqa: E731
    xt, y = aug(x_tr), torch.from_numpy(y_tr).double().to(dev)
    w = torch.zeros(xt.shape[1], y.shape[1], dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=5000, history_size=50, tolerance_grad=1e-7, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        z = xt @ w
        f = c * (torch.nn.functional.softplus(z) - y * z).sum() + 0.5 * (w * w).sum()
        f.backward()
        return f

    opt.step(loss)
    assert w.grad.abs().max() < 1e-4, float(w.grad.abs().max())
    return (aug(x_te) @ w).detach().cpu().numpy()


def siteprobe(split_seeds=(0, 1, 2)) -> dict:
    """wabad_embprobe_v1 linear probe (same species filter, standardisation and objective) with a
    site-disjoint 70/30 split instead of a recording-disjoint one; three random site splits, and the record
    split on the same species as a control. Bug check: the record split of the teacher reproduces probe cmAP
    0.4991 (up to liblinear's stopping tolerance)."""
    from evaluate.embeddings.probe_wabad_emb import EMB_DIRS, MIN_POS

    meta = pd.read_parquet(ROOT / "wabad/test.parquet")
    recs = meta["window_id"].str.rsplit("_", n=1).str[0].to_numpy()
    site = meta["site"].to_numpy()
    sp_all = sorted({c for x in meta["labels"] for c in x})
    Y = {c: np.zeros(len(meta), bool) for c in sp_all}
    for i, labs in enumerate(meta["labels"]):
        for c in labs:
            Y[c][i] = True
    species = [c for c in sp_all if Y[c].sum() >= MIN_POS and len(np.unique(recs[Y[c]])) >= 2]

    def split(units, seed):
        u = np.unique(units)
        return np.isin(units, list(set(np.random.default_rng(seed).choice(u, size=int(0.3 * len(u)), replace=False))))

    def probe(emb, te):
        sp = [c for c in species if Y[c][~te].any() and Y[c][te].any()]
        mu, sd = emb[~te].mean(0), emb[~te].std(0) + 1e-6
        yy = np.stack([Y[c] for c in sp], 1)
        s = fit_probes((emb[~te] - mu) / sd, yy[~te], (emb[te] - mu) / sd)
        return {c: average_precision_score(yy[te, j], s[:, j]) for j, c in enumerate(sp)}

    rec_split, site_splits = {}, {seed: {} for seed in split_seeds}
    for m in ("teacher", *TIERS):
        e = load_parts(EMB_DIRS[m], keys=("ids", "emb"))
        assert e["ids"].tolist() == meta["window_id"].tolist()
        emb = e["emb"].astype(np.float32)
        rec_split[m] = probe(emb, split(recs, 0))
        for seed in split_seeds:
            site_splits[seed][m] = probe(emb, split(site, seed))
        print("siteprobe", m, flush=True)
    mean = lambda d, sp: float(np.mean([d[c] for c in sp]))  # noqa: E731
    res = {"record_split_teacher_cmAP": round(mean(rec_split["teacher"], rec_split["teacher"]), 4)}
    assert abs(res["record_split_teacher_cmAP"] - 0.4991) < 2e-3  # bug check (iv)
    res["record_split"] = {f"{m}_ratio": round(mean(rec_split[m], rec_split[m]) / res["record_split_teacher_cmAP"], 4)
                           for m in TIERS}
    for seed, d in site_splits.items():
        sp = list(d["teacher"])
        both = [c for c in sp if c in rec_split["teacher"]]  # same species under the record split, as a control
        res[f"seed{seed}"] = {"n_species": len(sp), "teacher_cmAP": round(mean(d["teacher"], sp), 4)} | {
            f"{m}_ratio": round(mean(d[m], sp) / mean(d["teacher"], sp), 4) for m in TIERS} | {
            "record_split_same_species": {"n_species": len(both), "teacher_cmAP": round(mean(rec_split["teacher"], both), 4)}
            | {f"{m}_ratio": round(mean(rec_split[m], both) / mean(rec_split["teacher"], both), 4) for m in TIERS}}
    for k in ("teacher_cmAP", *(f"{m}_ratio" for m in TIERS)):
        res[f"mean_{k}"] = round(float(np.mean([res[f"seed{s_}"][k] for s_ in split_seeds])), 4)
    return res


def threshold() -> dict:
    """Detections at probability > 0.1 (students: exact-fold probabilities; teacher: its full softmax) of
    the benchmark's species: micro recall over annotated window-species pairs and precision over detections."""
    counts = {}
    for name in ALL_SETS:
        scores, y, _ = bench(name)
        for m in ("teacher_softmax", *TIERS):
            fired = scores[m] > (np.log(THRESH) if m == "teacher_softmax" else THRESH)
            counts[name, m] = np.array([(fired & y).sum(), fired.sum(), y.sum(), len(y)])
    out = {}
    for scope, sets in [(n, (n,)) for n in ALL_SETS] + [("test6_pooled", TEST_SETS), ("unused7_pooled", ("WABAD", *TEST_SETS))]:
        out[scope] = {}
        for m in ("teacher_softmax", *TIERS):
            tp, fired, pos, n = sum(counts[s, m] for s in sets)
            out[scope][m] = {"recall": round(tp / pos, 4), "precision": round(tp / fired, 4) if fired else None,
                             "detections_per_window": round(fired / n, 4)}
    return out


def nbp() -> dict:
    """Facts about BirdSet NBP, checked on 2026-11-13 (Hugging Face API, revision 'data' of
    DBD-research-group/BirdSet, and the download manifest DOWNLOADS/MANIFEST.md)."""
    return {"on_hugging_face": "yes: NBP/NBP_test5s_shard_0001.tar.gz (21 MB) and NBP_metadata_test_5s.parquet",
            "test_5s": "539 clips of 1.0-5.0 s (median 5 s; NIPS4Bplus annotations of NIPS4B recordings), 51 species, "
                       "every clip carries at least one label",
            "history": "it was left out of birdsetfull_v1 only because the BirdSet download had not fetched it; no "
                       "technical or methodological exclusion was recorded",
            "status": "now scored for all six models (birdsetnbp_v1; benchmarks.csv); never used in development, so it "
                      "is an unused benchmark and is included in every item of this file except Table S2 (groups)"}


ITEMS = {f.__name__: f for f in (nbp, ordering, windows, continents, agreement, groups, common, softmax,
                                 bootstrap, threshold, mcurate, siteprobe)}

if __name__ == "__main__":
    for item in sys.argv[1:] or ITEMS:
        value = ITEMS[item]()
        print(item, json.dumps(value, default=float)[:2000], flush=True)
        res = json.loads(OUT.read_text()) if OUT.exists() else {}  # re-read: items may run in parallel
        OUT.write_text(json.dumps(res | {item: value}, indent=1, default=float))
