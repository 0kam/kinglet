# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Further statistics of the Kinglet paper on existing scores, histograms and features (nothing retrained).

  ssw        rev2_ssw_v1        foreign-list false detections without SSW, per test set, SSW shares
                                (from opensetcurve_v1's per-cluster histograms, DISTILL_ROOT/opensetcurve_v1/hists.npz)
  crlist     rev2_crlist_v1     listed share of annotated occurrences per list; Costa Rican list false detections
                                per species for the teacher, S and XS (full output layer)
  jpspecies  rev2_jpspecies_v1  student/teacher AP ratio, Japanese-list species vs the others (revision_stats)
  mcupack    rev2_mcupack_v1    the microcontroller output layer (317 species + other, no sound groups) of the
                                board model, in floating point on the host: Japan list and WABAD site lists

Operating point as in the species-list table: each model's false detections per window at the teacher's recall at p > 0.1.

Usage: python evaluate/stats/revision2_stats.py <item> [...]   ->  results/rev2_<item>_v1.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evaluate.species_lists.opensetcurve as oc  # noqa: E402
from common.config import DATA_ROOT, RELEASE_ASSETS, RESULTS, jp_list  # noqa: E402
from common.config import DISTILL_ROOT as DD  # noqa: E402
from evaluate.species_lists.opensetcurve import OUT, Hist, at_recall, curve, logit_from_logp, op, to_bins  # noqa: E402

torch.set_num_threads(4)  # opensetcurve sets 8 on import
RES = RESULTS
N_BOOT = 2000
T01 = oc.OPS["p>0.1"]
US_SETS = ("POW", "HSN", "SNE", "SSW")


def boot_R(hists: dict, name: str, sel: np.ndarray, models: list[str], rng) -> dict:
    """The opensetcurve.summarize cluster bootstrap of R = FP(model) / FP(teacher) at the teacher's recall at
    p > 0.1, on the clusters `sel`; point estimate and 95% interval per model."""
    th = hists[(name, "teacher")]
    r0 = op(th, sel)["recall"]
    fpt0 = at_recall(curve(th, sel), r0)[0]
    out = {m: {"R": at_recall(curve(hists[(name, m)], sel), r0)[0] / fpt0, "boot": []} for m in models}
    idx = np.flatnonzero(sel)
    for _ in range(N_BOOT):
        w = np.bincount(rng.choice(idx, len(idx)), minlength=len(sel)).astype(float)

        def wcurve(h):
            def top(a):
                return np.cumsum((a * w[:, None]).sum(0)[::-1])[::-1][1:]
            return {"recall": top(h.pos) / (h.npos * w).sum(), "fp": top(h.neg) / (h.nwin * w).sum(),
                    "any": top(h.wmax) / (h.nwin * w).sum()}
        r = (th.ops["p>0.1"][:, 1] * w).sum() / (th.npos * w).sum()
        fpt = at_recall(wcurve(th), r)[0]
        for m in models:
            out[m]["boot"].append(at_recall(wcurve(hists[(name, m)]), r)[0] / fpt)
    res = {"teacher_recall": r0, "teacher_fp_per_window": fpt0}
    for m, e in out.items():
        v = np.array(e["boot"])
        res[m] = {"R": e["R"], "fp_per_window": e["R"] * fpt0,
                  "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
                  "nan_share": float(np.isnan(v).mean())}
    return res


def selections() -> tuple[list[str], dict]:
    names, out_of = oc.clusters()
    return names, {k: np.isin(names, sorted(v)) for k, v in out_of.items()}


def set_of(n: str) -> str:
    return "WABAD" if n.startswith("WABAD:") else n


# ---------------------------------------------------------------------------------------------- rev2_ssw_v1
def ssw() -> dict:
    hists, _ = oc.load_hists()
    names, sel = selections()
    full = [f"{t}_exact" for t in oc.TIERS]
    is_ssw = np.array(names) == "SSW"
    out = {"bug_check": {}, "no_ssw": {}, "per_set": {}, "ssw_share": {}}
    # bug check: the published bootstrap, same seed and list order (JP, DE, CR) as opensetcurve.summarize
    rng = np.random.default_rng(0)
    pub = json.loads((RES / "opensetcurve_v1.json").read_text()) | {
        k: v for k, v in json.loads((RES / "regionlists_v1.json").read_text()).items() if k != "checks"}
    for name in ("JP", "DE", "CR"):
        models = sorted(m for (n, m) in hists if n == name and m != "teacher")
        b = boot_R(hists, name, sel[name], models, rng)
        for m in models:
            p = pub[name]["models"][m]["R_vs_teacher_at_teacher_recall"]
            assert abs(b[m]["R"] - p["R"]) < 1e-9 and np.allclose(b[m]["ci95"], p["ci95"], atol=1e-9), (name, m)
        out["bug_check"][name] = {m: {"R": round(b[m]["R"], 4), "ci95": [round(x, 4) for x in b[m]["ci95"]]}
                                  for m in full} | {"teacher_fp_per_window": round(b["teacher_fp_per_window"], 4),
                                                    "teacher_recall": round(b["teacher_recall"], 4)}
    out["bug_check"]["reproduced"] = "all R and ci95 of opensetcurve_v1.json / regionlists_v1.json to 1e-9"
    for name in ("JP", "DE", "CR"):
        s = sel[name]
        # (a) SSW excluded
        b = boot_R(hists, name, s & ~is_ssw, full, np.random.default_rng(0))
        out["no_ssw"][name] = {"clusters": int((s & ~is_ssw).sum())} | b
        # (b) per test set (WABAD sites pooled as one set)
        per = {}
        for set_name in dict.fromkeys(set_of(n) for n in np.array(names)[s]):
            ss = s & np.array([set_of(n) == set_name for n in names])
            th = hists[(name, "teacher")]
            o = op(th, ss)
            if not o["recall"]:  # no listed occurrence, or the teacher finds none at p > 0.1: no operating point
                per[set_name] = {"windows": o["windows"], "listed_occurrences": o["positives"],
                                 "teacher_fp_per_window_at_0.1": o["fp_per_window"],
                                 "note": "undefined: teacher recall at p>0.1 is 0" if o["positives"] else
                                 "undefined: no annotated occurrence of a listed species"}
                continue
            fpt = at_recall(curve(th, ss), o["recall"])[0]
            e = {"clusters": int(ss.sum()), "windows": o["windows"], "listed_occurrences": o["positives"],
                 "teacher_recall": o["recall"], "teacher_fp_per_window": fpt}
            for m in full:
                c = curve(hists[(name, m)], ss)
                e[m] = {"fp_per_window": at_recall(c, o["recall"])[0],
                        "R": at_recall(c, o["recall"])[0] / fpt if fpt > 0 else float("nan"),
                        "max_recall": float(c["recall"].max())}
            per[set_name] = e
        mean = {}
        for m in full:
            v = np.array([e[m]["R"] if m in e else np.nan for e in per.values()])
            ok = np.isfinite(v)
            mean[m] = {"unweighted_mean_R": float(v[ok].mean()), "median_R": float(np.median(v[ok])),
                       "sets_used": int(ok.sum()), "sets_undefined": [k for k, e in per.items()
                                                                      if m not in e or not np.isfinite(e[m]["R"])]}
        # (b') per set at the pooled operating point: each model at the grid threshold whose pooled recall is
        # closest to the teacher's pooled recall at p > 0.1; defined on every set, sums to the pooled numbers
        r0 = op(hists[(name, "teacher")], s)["recall"]
        pooled_thr = {}
        for m in ("teacher", *full):
            c = curve(hists[(name, m)], s)
            i = int(np.argmin(np.abs(c["recall"] - r0)))
            pooled_thr[m] = {"edge": i, "threshold_p": float(1 / (1 + np.exp(-oc.EDGES[i]))),
                             "pooled_recall": float(c["recall"][i]), "pooled_fp_per_window": float(c["fp"][i])}
        at_pooled = {}
        for set_name in dict.fromkeys(set_of(n) for n in np.array(names)[s]):
            ss = s & np.array([set_of(n) == set_name for n in names])
            e = {"windows": int(hists[(name, "teacher")].nwin[ss].sum())}
            for m in ("teacher", *full):
                h, i = hists[(name, m)], pooled_thr[m]["edge"]
                e[m] = {"fp": int(h.neg[ss][:, i + 1:].sum()), "hits": int(h.pos[ss][:, i + 1:].sum())}
            e["listed_occurrences"] = int(hists[(name, "teacher")].npos[ss].sum())
            for m in full:
                e[m]["R"] = e[m]["fp"] / e["teacher"]["fp"] if e["teacher"]["fp"] else float("nan")
            e["teacher_fp_share"] = e["teacher"]["fp"] / pooled_thr["teacher"]["pooled_fp_per_window"] / int(
                hists[(name, "teacher")].nwin[s].sum())
            at_pooled[set_name] = e
        mean_p = {}
        for m in full:
            v = np.array([e[m]["R"] for e in at_pooled.values()])
            mean_p[m] = {"unweighted_mean_R": float(np.nanmean(v)), "median_R": float(np.nanmedian(v)),
                         "sets_used": int(np.isfinite(v).sum())}
        out["per_set"][name] = {"sets": per, "mean": mean,
                                "at_pooled_threshold": {"thresholds": pooled_thr, "sets": at_pooled, "mean": mean_p}}
        # (c) SSW shares
        th = hists[(name, "teacher")]
        tot, sw = op(th, s), op(th, s & is_ssw)
        fp_all = th.ops["p>0.1"][s, 0].sum()
        fp_ssw = th.ops["p>0.1"][s & is_ssw, 0].sum()
        out["ssw_share"][name] = {"windows": [sw["windows"], tot["windows"], sw["windows"] / tot["windows"]],
                                  "listed_occurrences": [sw["positives"], tot["positives"],
                                                         sw["positives"] / tot["positives"]],
                                  "teacher_fp_at_0.1": [int(fp_ssw), int(fp_all), fp_ssw / fp_all]}
    return out


# ---------------------------------------------------------------------------------------------- rev2_crlist_v1
def manifest(set_name: str) -> pd.DataFrame:
    from common.scoring import SETS
    m, id_col = SETS[set_name][:2]
    man = pd.read_parquet(DATA_ROOT / m)
    man["_id"] = man[id_col].astype(str)
    man["_cluster"] = ("WABAD:" + man["site"].astype(str)) if set_name == "WABAD" else set_name
    return man


def heads() -> dict:
    out = {}
    for tier, tag in oc.TIERS.items():
        hp = np.load(RELEASE_ASSETS / f"kinglet-{tag}-v0.3.2.headpack.npz")
        out[tier] = (torch.from_numpy(hp["W"].astype(np.float32)), torch.from_numpy(hp["b"].astype(np.float32)))
    return out


def crlist() -> dict:
    hists, _ = oc.load_hists()
    names, sel = selections()
    lists = oc.lists()
    out = {"listed_share": {}}
    # (1) annotated occurrences (window-species labels) of listed species, on each list's clusters
    occ = {k: [0, 0] for k in lists}
    mans = {s: manifest(s) for s in oc.PUBLIC_SETS}
    for s, man in mans.items():
        for name, (codes, _) in lists.items():
            keep = man["_cluster"].isin(np.array(names)[sel[name]])
            labs = man.loc[keep, "labels"]
            cs = set(codes)
            occ[name][0] += int(sum(len(x) for x in labs))
            occ[name][1] += int(sum(sum(c in cs for c in x) for x in labs))
    for name, (tot, li) in occ.items():
        npos = int(hists[(name, "teacher")].npos[sel[name]].sum())
        assert li == npos, (name, li, npos)  # bug check: the histograms' positives
        out["listed_share"][name] = {"annotated_occurrences": tot, "listed": li, "listed_share": li / tot,
                                     "list_size": len(lists[name][0]), "clusters": int(sel[name].sum())}
    # (2) Costa Rican list, per species: teacher, S, XS (full output layer)
    codes, vidx = lists["CR"]
    s = sel["CR"]
    r = op(hists[("CR", "teacher")], s)["recall"]
    models = {"teacher": "teacher", "S": "S_exact", "XS": "XS_exact"}
    k_edge, thr_info = {}, {}
    for m, hk in models.items():
        c = curve(hists[("CR", hk)], s)
        i = int(np.argmin(np.abs(c["recall"] - r)))
        k_edge[m] = i + 1  # scores in bins >= i + 1, i.e. >= EDGES[i]
        thr_info[m] = {"threshold_logit": float(oc.EDGES[i]), "threshold_p": float(1 / (1 + np.exp(-oc.EDGES[i]))),
                       "recall_at_threshold": float(c["recall"][i]), "fp_per_window_at_threshold": float(c["fp"][i]),
                       "fp_per_window_interpolated": at_recall(c, r)[0]}
    hd = heads()
    cl_sets = sorted({set_of(n) for n in np.array(names)[s]})
    n_sp = len(codes)
    fp_match = {m: {st: np.zeros(n_sp, np.int64) for st in cl_sets} for m in models}
    fp01 = {m: np.zeros(n_sp, np.int64) for m in models}
    pos = {st: np.zeros(n_sp, np.int64) for st in cl_sets}
    nwin = {st: 0 for st in cl_sets}
    ci = {c: j for j, c in enumerate(codes)}
    dead = vidx < 0
    for st in cl_sets:
        man = mans[st]
        keep = man["_cluster"].isin(np.array(names)[s]).to_numpy()
        y = np.zeros((len(man), n_sp), bool)
        for i, labs in enumerate(man["labels"]):
            y[i, [ci[c] for c in labs if c in ci]] = True
        t = np.load(OUT / f"teacher_{st}.npz")
        assert t["ids"].tolist() == man["_id"].tolist()
        tcol = {int(v): j for j, v in enumerate(t["vidx"])}
        tc = np.array([tcol.get(int(v), 0) for v in vidx])
        feats = {}
        for tier in ("S", "XS"):
            f = (np.load(DD / f"packfit_v1/feat_wabad_{tier}/part-0000.npz") if st == "WABAD"
                 else np.load(OUT / f"feat_{st}_{tier}.npz"))
            assert f["ids"].tolist() == man["_id"].tolist()
            feats[tier] = f["feat"]
        nwin[st] = int(keep.sum())
        pos[st] += y[keep].sum(0)
        tl_all = torch.from_numpy(t["logits"].astype(np.float32)) - torch.from_numpy(t["logz"])[:, None]
        for i in range(0, len(man), oc.CHUNK):  # chunks exactly as opensetcurve.score_set, then the kept windows
            rows = slice(i, i + oc.CHUNK)
            k = keep[rows]
            if not k.any():
                continue
            yy = y[rows][k]
            sc = {"teacher": logit_from_logp(tl_all[rows][:, tc].double()).numpy()}
            for tier in ("S", "XS"):
                W, b = hd[tier]
                lg = torch.from_numpy(feats[tier][rows]).float() @ W.t() + b
                logz = torch.logsumexp(lg, 1, keepdim=True)
                sc[tier] = logit_from_logp((lg[:, np.maximum(vidx, 0)] - logz).double()).numpy()
            for m, v in sc.items():
                v = v[k]
                v[:, dead] = -np.inf
                fp_match[m][st] += ((to_bins(v) >= k_edge[m]) & ~yy).sum(0)
                fp01[m] += ((v > T01) & ~yy).sum(0)
        del feats, t
    # bug check: p > 0.1 counts reproduce the histogram op points; matched-edge counts the curve
    nw = sum(nwin.values())
    chk = {}
    for m, hk in models.items():
        o = op(hists[("CR", hk)], s)
        tot = sum(v.sum() for v in fp_match[m].values())
        chk[m] = {"fp01_mine_vs_hist": [int(fp01[m].sum()), int(hists[("CR", hk)].ops["p>0.1"][s, 0].sum())],
                  "fp_at_edge_mine_vs_hist": [int(tot), round(thr_info[m]["fp_per_window_at_threshold"] * nw)]}
        assert o["windows"] == nw
        assert abs(fp01[m].sum() / nw / o["fp_per_window"] - 1) < 1e-3, (m, chk[m])
        assert abs(tot / nw / thr_info[m]["fp_per_window_at_threshold"] - 1) < 1e-3, (m, chk[m])
    out["cr_bug_check"] = {"windows": nw, "counts": chk, "fp_per_window_at_0.1": {m: fp01[m].sum() / nw for m in models},
                           "listed_occurrences": int(sum(p.sum() for p in pos.values()))}
    out["cr_operating_point"] = {"teacher_recall_at_0.1": r} | thr_info
    tot_fp = {m: sum(fp_match[m].values()) for m in models}
    us = [st for st in cl_sets if st in US_SETS]
    sci = pd.read_csv(RESULTS / "regionlist_cr.csv").set_index("ebird_code")
    for tier in ("S", "XS"):
        inc = tot_fp[tier] - tot_fp["teacher"]
        order = np.argsort(-inc)
        top = []
        for j in order[:10]:
            top.append({"ebird_code": codes[j],
                        "name": str(sci.loc[codes[j]].get("scientific_name", "")) if codes[j] in sci.index else "",
                        "fp_teacher": int(tot_fp["teacher"][j]), f"fp_{tier}": int(tot_fp[tier][j]),
                        "increase": int(inc[j]),
                        f"fp_{tier}_by_set": {st: int(fp_match[tier][st][j]) for st in cl_sets if fp_match[tier][st][j]},
                        "fp_teacher_by_set": {st: int(fp_match["teacher"][st][j]) for st in cl_sets
                                              if fp_match["teacher"][st][j]},
                        "annotated_windows_by_set": {st: int(pos[st][j]) for st in cl_sets if pos[st][j]},
                        "annotated_in_us_sets": bool(sum(pos[st][j] for st in us) > 0)})
        net = int(inc.sum())
        pos_inc = int(inc[inc > 0].sum())
        top10 = int(inc[order[:10]].sum())
        ann_us = np.array([sum(pos[st][j] for st in us) > 0 for j in range(n_sp)])
        out[f"cr_{tier}"] = {
            "fp_total": {"teacher": int(tot_fp["teacher"].sum()), tier: int(tot_fp[tier].sum())},
            "net_increase": net, "sum_of_positive_increases": pos_inc, "top10_increase": top10,
            "top10_share_of_net": top10 / net if net else None,
            "increase_by_set": {st: int(fp_match[tier][st].sum() - fp_match["teacher"][st].sum()) for st in cl_sets},
            "increase_from_species_annotated_in_us_sets": int(inc[ann_us].sum()),
            "increase_from_species_not_annotated_anywhere_here": int(inc[pos_total_zero(pos, cl_sets)].sum()),
            "species_with_fp_increase": int((inc > 0).sum()), "top10": top}
    out["cr_sets"] = {st: {"windows": nwin[st], "listed_occurrences": int(pos[st].sum())} for st in cl_sets}
    return out


def pos_total_zero(pos: dict, sets: list[str]) -> np.ndarray:
    return sum(pos[st] for st in sets) == 0


# ---------------------------------------------------------------------------------------------- rev2_jpspecies_v1
def jpspecies() -> dict:
    import evaluate.stats.revision_stats as rs

    df = rs.per_species().copy()
    jp = {int(c["perch_idx"]) for c in jp_list()
          if c["perch_idx"] is not None and c["perch_idx"] >= 0}
    pidx = {}
    for name in rs.ALL_SETS:
        for c in json.loads((rs.ROOT / rs.SETS[name][-1] / "classes.json").read_text()):
            pidx[(name, c["species"])] = c["perch_idx"]
    df["perch_idx"] = [pidx[(b, s)] for b, s in zip(df["benchmark"], df["species"])]
    df["in_jp"] = [p is not None and p >= 0 and int(p) in jp for p in df["perch_idx"]]
    assert len(df) == 1694, len(df)
    fig = pd.read_csv(rs.DATA / "species_ap.csv")  # bug check: the Fig. 3c pairs
    for m in ("teacher", *rs.TIERS):
        assert abs(fig[m].mean() - df[m].mean()) < 1e-4, m
    out = {"jp_classes": len(jp)}
    scopes = {"all 9 benchmarks (Fig. 3c)": df,
              "7 benchmarks not used in development": df[~df["benchmark"].isin(("POW", "HSN"))],
              "WABAD only": df[df["benchmark"] == "WABAD"],
              ">= 100 training recordings (Table S2 group), 9 benchmarks": df[df["group"] == "high"]}
    for scope, d in scopes.items():
        g = {"jp": d[d["in_jp"]], "other": d[~d["in_jp"]]}
        res = {"n_pairs": {k: len(v) for k, v in g.items()},
               "n_species": {k: int(v["species"].nunique()) for k, v in g.items()},
               "jp_pairs_by_benchmark": g["jp"]["benchmark"].value_counts().to_dict(),
               "median_positives": {k: float(v["positives"].median()) for k, v in g.items()},
               "teacher_mean_ap": {k: float(v["teacher"].mean()) for k, v in g.items()}}
        rng = np.random.default_rng(0)
        for m in rs.TIERS:
            bj = rs.boot_ratio(g["jp"][m].to_numpy(), g["jp"]["teacher"].to_numpy(), rng)[0]
            bo = rs.boot_ratio(g["other"][m].to_numpy(), g["other"]["teacher"].to_numpy(), rng)[0]
            rj = g["jp"][m].mean() / g["jp"]["teacher"].mean()
            ro = g["other"][m].mean() / g["other"]["teacher"].mean()
            res[m] = {"ratio_jp": round(rj, 4), "ratio_jp_ci": rs.ci(bj), "ratio_other": round(ro, 4),
                      "ratio_other_ci": rs.ci(bo), "diff_jp_minus_other": round(rj - ro, 4), "diff_ci": rs.ci(bj - bo)}
        out[scope] = res
    return out


# ---------------------------------------------------------------------------------------------- rev2_mcupack_v1
def mcupack() -> dict:
    from sklearn.metrics import average_precision_score

    from common.scoring import cmap
    from evaluate.species_lists.packcmap import fold
    from evaluate.species_lists.packfit_compare import Bundle

    torch.set_num_threads(4)  # packcmap sets 8 on import
    out = {"bug_check": {}}
    hists, _ = oc.load_hists()
    names, sel = selections()
    cl_index = {n: i for i, n in enumerate(names)}
    jp_codes, jp_vidx = oc.lists()["JP"]
    live = jp_vidx >= 0
    dep = torch.load(DD / "xs_pub_deploy/deploy_state.pt", map_location="cpu", weights_only=False)
    Wd, bd = np.asarray(dep["W"], np.float32), np.asarray(dep["b"], np.float32)
    pk = np.load(DD / "xs_pub_deploy/pack_jp_focal.npz")
    assert Wd.shape == (318, 1280) and [str(c) for c in dep["codes"][:317]] == jp_codes
    assert int(pk["n_species"]) == 317 and json.loads(str(pk["groups"])) == {}
    same = np.allclose(Wd, pk["W"]) and np.allclose(bd, pk["b"])
    diff_rows = np.flatnonzero(~(np.isclose(Wd, pk["W"]).all(1) & np.isclose(bd, pk["b"])))
    hp_path = RELEASE_ASSETS / "kinglet-xs-v0.3.2.headpack.npz"
    hp = np.load(hp_path)
    W = torch.from_numpy(hp["W"].astype(np.float32))
    b = torch.from_numpy(hp["b"].astype(np.float32))
    bundle = Bundle(hp_path, W, b)
    wb = bundle.other_row(torch.tensor(jp_vidx[live])).numpy()
    out["bug_check"]["board_layer"] = {
        "deploy_state_equals_pack_jp_focal": bool(same), "rows_differing": diff_rows.tolist(),
        "dead_species": [jp_codes[j] for j in np.flatnonzero(~live)],
        "species_rows_vs_full_head_maxabs": float(np.abs(Wd[:317][live] - hp["W"][jp_vidx[live]].astype(np.float32)).max()),
        "other_row_rebuilt_from_v0.3.2_bundle_maxabs_W": float(np.abs(wb[:-1] - Wd[-1]).max()),
        "other_row_rebuilt_abs_b": float(abs(wb[-1] - bd[-1]))}
    assert np.abs(wb[:-1] - Wd[-1]).max() < 1e-4 and abs(wb[-1] - bd[-1]) < 1e-3
    rel = np.load(RELEASE_ASSETS / "kinglet-xs-v0.3.2.pack-jp.npz")
    Wr, br, nr = torch.from_numpy(rel["W"]), torch.from_numpy(rel["b"]), int(rel["n_species"])
    Wd_t, bd_t = torch.from_numpy(Wd), torch.from_numpy(bd)

    # (a) Japan list on the eight benchmarks
    hm, hc = Hist(len(names)), Hist(len(names))
    for st in oc.PUBLIC_SETS:
        man = manifest(st)
        ci = {c: j for j, c in enumerate(jp_codes)}
        y = np.zeros((len(man), 317), bool)
        for i, labs in enumerate(man["labels"]):
            y[i, [ci[c] for c in labs if c in ci]] = True
        cl = man["_cluster"].map(cl_index).to_numpy()
        f = (np.load(DD / "packfit_v1/feat_wabad_XS/part-0000.npz") if st == "WABAD"
             else np.load(OUT / f"feat_{st}_XS.npz"))
        assert f["ids"].tolist() == man["_id"].tolist()
        for i in range(0, len(man), oc.CHUNK):
            rows = slice(i, i + oc.CHUNK)
            h = torch.from_numpy(f["feat"][rows]).float()
            for hist, (Wp, bp, n) in ((hm, (Wd_t, bd_t, 317)), (hc, (Wr, br, nr))):
                pl = h @ Wp.t() + bp
                s = logit_from_logp((pl[:, :n] - torch.logsumexp(pl, 1, keepdim=True)).double()).numpy()
                s[:, ~live] = -np.inf
                hist.add(s, y[rows], cl[rows])
        print(st, "done", flush=True)
    hists[("JP", "XS_mcu")] = hm
    hists[("JP", "XS_pack_recomputed")] = hc
    ref = hists[("JP", "XS_pack")]
    out["bug_check"]["computer_pack_recomputed_vs_hists"] = {
        "ops_equal": bool(np.array_equal(ref.ops["p>0.1"], hc.ops["p>0.1"])),
        "neg_hist_max_abs_count_diff": int(np.abs(ref.neg - hc.neg).max()),
        "pos_hist_max_abs_count_diff": int(np.abs(ref.pos - hc.pos).max())}
    b_ = boot_R(hists, "JP", sel["JP"], ["XS_exact", "XS_pack", "XS_pack_recomputed", "XS_mcu"],
                np.random.default_rng(0))
    out["japan"] = b_ | {"op_p>0.1": {m: op(hists[("JP", m)], sel["JP"]) for m in
                                      ("teacher", "XS_exact", "XS_pack", "XS_mcu")}}
    out["japan"]["p_any_fp_at_teacher_recall"] = {
        m: at_recall(curve(hists[("JP", m)], sel["JP"]), b_["teacher_recall"])[1]
        for m in ("teacher", "XS_exact", "XS_pack", "XS_mcu")}

    # (b) WABAD site lists, packcmap_v1 protocol with the board recipe (no sound-group rows)
    from teacher.embed_perch import perch_classes
    man = manifest("WABAD")
    site = man["site"].to_numpy()
    sites = sorted(np.unique(site))
    species = sorted({c for x in man["labels"] for c in x})
    sp_i = {c: j for j, c in enumerate(species)}
    pidx = np.array([c["perch_idx"] for c in perch_classes(species)])
    y = np.zeros((len(man), len(species)), np.int8)
    for i, labs in enumerate(man["labels"]):
        y[i, [sp_i[c] for c in labs]] = 1
    site_sp = {s: sorted({sp_i[c] for x in man.loc[site == s, "labels"] for c in x}) for s in sites}
    lab = {str(n): i for i, n in enumerate(hp["labels"])}
    midx = [lab[m] for ms in json.loads(str(rel["groups"])).values() for m in ms]
    f = np.load(DD / "packfit_v1/feat_wabad_XS/part-0000.npz")
    h = torch.from_numpy(f["feat"].astype(np.float32))
    lg = torch.cat([h[i : i + 4096] @ W.t() + b for i in range(0, len(h), 4096)])
    arms = ("exact", "computer", "mcu")
    sc = {k: np.zeros(y.shape, np.float32) for k in arms}
    fires = {k: np.zeros(3) for k in ("computer", "mcu")}  # exact-only, compact-only, exact fires
    per_site = []
    for s in sites:
        m = torch.from_numpy(site == s)
        cols = [j for j in site_sp[s] if pidx[j] >= 0]
        lv = torch.tensor(pidx[cols])
        rows = np.flatnonzero(site == s)
        for arm, in_cols in (("computer", torch.cat([lv, torch.tensor(midx)])), ("mcu", lv)):
            pe, pp, _ = fold(lg[m], h[m], lv, in_cols, bundle.other_row(in_cols))
            sc["exact"][np.ix_(rows, cols)] = pe.numpy()
            sc[arm][np.ix_(rows, cols)] = pp.numpy()
            fe, fc = (pe > 0.1).any(1), (pp > 0.1).any(1)
            fires[arm] += [float((fe & ~fc).sum()), float((fc & ~fe).sum()), float(fe.sum())]
        ys = y[np.ix_(rows, site_sp[s])].ravel()
        per_site.append({"site": s} | {k: float(average_precision_score(ys, sc[k][np.ix_(rows, site_sp[s])].ravel()))
                                       for k in arms})
    ps = pd.DataFrame(per_site)
    sitelists = {f"masked_cmAP_{k}": cmap(sc[k], y) for k in arms} | {
        f"masked_pooled_auprc_{k}": float(average_precision_score(y.ravel(), sc[k].ravel())) for k in arms}
    for k in ("computer", "mcu"):
        d = ps[k] - ps["exact"]
        sitelists[f"cmAP_change_{k}"] = sitelists[f"masked_cmAP_{k}"] - sitelists["masked_cmAP_exact"]
        sitelists[f"site_auprc_{k}_minus_exact"] = {"median": float(d.median()), "min": float(d.min()),
                                                    "max": float(d.max()), "sites_worse": int((d < 0).sum())}
        a = fires[k]
        sitelists[f"fires_{k}"] = {"exact_only_windows": a[0] / len(man), "compact_only_windows": a[1] / len(man),
                                   "exact_fire_windows": a[2] / len(man), "missed_share_of_exact": a[0] / a[2]}
    pub = json.loads((RES / "packcmap_v1.json").read_text())["tiers"]["XS"]
    out["bug_check"]["packcmap_xs"] = {"exact": [sitelists["masked_cmAP_exact"], pub["masked_cmAP_exact"]],
                                       "computer": [sitelists["masked_cmAP_computer"], pub["masked_cmAP_pack"]]}
    assert abs(sitelists["masked_cmAP_exact"] - pub["masked_cmAP_exact"]) < 1e-6
    assert abs(sitelists["masked_cmAP_computer"] - pub["masked_cmAP_pack"]) < 1e-6
    out["wabad_site_lists"] = sitelists
    return out


ITEMS = {"ssw": ssw, "crlist": crlist, "jpspecies": jpspecies, "mcupack": mcupack}

if __name__ == "__main__":
    for item in sys.argv[1:] or ITEMS:
        res = ITEMS[item]()
        dst = RES / f"rev2_{item}_v1.json"
        dst.write_text(json.dumps(res, indent=1, default=float))
        print("wrote", dst, flush=True)
