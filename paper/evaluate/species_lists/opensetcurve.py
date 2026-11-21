# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""opensetcurve_v1 + regionlists_v1: false detections vs recall of listed species on the eight benchmarks.

Lists: JP (released 317 Japan classes), DE and CR (region_lists.py). Models:
  teacher       Perch 2.0 full softmax read at the listed classes (JP: cache openset317_perchfold_*, the paper's
                table; DE/CR: opensetcurve_extract.py --part teacher, logits - logZ)
  <T>_exact     Kinglet tier T, softmax of the full 14,795-class head (v0.3.2 headpack W, b) on the trunk features
  <T>_pack      JP: the released v0.3.2 pack-jp; DE/CR: a compact pack (species + released sound-group rows + a
                ridge-fitted "other", packfit_compare.Bundle on the v0.3.2 headpack = kinglet build-pack --compact)
  v24           BirdNET V2.4 confidence logits (JP only, caches of openset_public.py)
                (BirdNET+ V3.0 rows are added by evaluate/accuracy/birdnetv3rel.py, birdnetv3rel_v2)
A false detection = a listed species scoring above the threshold in a window where it is not annotated; recall =
share of annotated occurrences of listed species above the threshold. Scores are binned on a logit grid (step 0.01,
p = 0.1 on an edge) per cluster (72 WABAD sites, 7 BirdSet sets), so curves, matched-recall points and the cluster
bootstrap all come from the histograms. Everything streams in chunks; no (windows x 14,795) matrix is kept.

  python evaluate/species_lists/opensetcurve.py --lists JP     # Japan list only (no teacher pass needed)
  python evaluate/species_lists/opensetcurve.py                # -> results/opensetcurve_v1.json,
                                        #    results/regionlists_v1.json, figure_data/openset_curves.csv
  python evaluate/species_lists/opensetcurve.py --from-hists   # re-summarise DISTILL_ROOT/opensetcurve_v1/hists.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, DISTILL_ROOT, FIGURE_DATA, RELEASE_ASSETS, RESULTS, jp_list  # noqa: E402
from common.scoring import CACHE, SETS  # noqa: E402
from evaluate.species_lists.openset_public import PUBLIC_SETS  # noqa: E402
from evaluate.species_lists.packfit_compare import Bundle  # noqa: E402

OUT = DISTILL_ROOT / "opensetcurve_v1"
CSV = FIGURE_DATA / "openset_curves.csv"
TIERS = {"B": "b", "S": "s", "XS": "xs"}
LO, STEP_L, NB = float(np.log(0.1 / 0.9)) - 12.0, 0.01, 2602  # edges LO + 0.01 k; p = 0.1 is edge 1200
EDGES = LO + STEP_L * np.arange(NB - 1)
OPS = {"p>0.1": float(np.log(0.1 / 0.9)), "p>0.25": float(np.log(0.25 / 0.75))}
EUROPE = {"France", "Germany", "Greece", "Poland", "Scotland", "Spain", "Sweden", "Ukraine", "Cyprus"}
NEOTROPICS = {"Argentina", "Brazil", "Colombia", "Costa Rica", "Dominican Republic", "Guatemala", "Mexico"}
CHUNK = 4096
torch.set_num_threads(8)
np.seterr(divide="ignore")  # log(0) = -inf is the intended score of a silent class


def lists() -> dict:
    """list -> (codes, Perch index per code; -1 = dead row)."""
    jp = jp_list()
    out = {"JP": ([c["species"] for c in jp], np.array([c["perch_idx"] for c in jp]))}
    for r in ("de", "cr"):
        df = pd.read_csv(RESULTS / f"regionlist_{r}.csv")
        out[r.upper()] = (df.ebird_code.tolist(), df.vocab_index.to_numpy())
    return out


def clusters() -> tuple[list[str], dict]:
    sites = pd.read_csv(FIGURE_DATA / "wabad_per_site.csv")
    names = [f"WABAD:{s}" for s in sites.site] + [s for s in PUBLIC_SETS if s != "WABAD"]
    country = dict(zip(names[: len(sites)], sites.country)) | {
        "POW": "USA", "HSN": "USA", "NES": "Colombia", "PER": "Peru", "SNE": "USA", "SSW": "USA", "UHH": "Hawaii (USA)"}
    out_of = {"JP": set(names),
              "DE": {n for n in names if country[n] not in EUROPE},
              "CR": {n for n in names if country[n] not in NEOTROPICS and n not in ("NES", "PER")}}
    return names, out_of


def logit_from_logp(logp: torch.Tensor) -> torch.Tensor:
    return logp - torch.log1p(-torch.exp(logp.clamp(max=-1e-7)))


def to_bins(s: np.ndarray) -> np.ndarray:
    return np.floor(np.clip((s - LO) / STEP_L, -1, NB)).astype(np.int64).clip(-1, NB - 2) + 1  # -inf -> bin 0


class Hist:
    """Per cluster: histograms of positive scores, negative scores and per-window max negative score."""

    def __init__(self, n_cl: int):
        self.pos, self.neg, self.wmax = (np.zeros((n_cl, NB), np.int64) for _ in range(3))
        self.nwin, self.npos = np.zeros(n_cl, np.int64), np.zeros(n_cl, np.int64)
        self.ops = {k: np.zeros((n_cl, 3), np.int64) for k in OPS}  # fp, hits, windows with fp

    def add(self, s: np.ndarray, y: np.ndarray, cl: np.ndarray):
        s = np.where(np.isfinite(s), s, -np.inf).astype(np.float64)
        n_cl = len(self.nwin)
        b = to_bins(s)
        idx = cl[:, None] * NB + b
        self.pos += np.bincount(idx[y], minlength=n_cl * NB).reshape(n_cl, NB)
        self.neg += np.bincount(idx[~y], minlength=n_cl * NB).reshape(n_cl, NB)
        sneg = np.where(y, -np.inf, s).max(1)
        self.wmax += np.bincount(cl * NB + to_bins(sneg), minlength=n_cl * NB).reshape(n_cl, NB)
        self.nwin += np.bincount(cl, minlength=n_cl)
        self.npos += np.bincount(np.repeat(cl, y.sum(1)), minlength=n_cl)
        for k, t in OPS.items():
            f = s > t
            self.ops[k] += np.stack([np.bincount(cl, (f & ~y).sum(1), n_cl), np.bincount(cl, (f & y).sum(1), n_cl),
                                     np.bincount(cl, (sneg > t).astype(float), n_cl)], 1).astype(np.int64)


def curve(h: Hist, sel: np.ndarray) -> dict:
    """Pooled over the selected clusters: recall, FP/window, P(any FP) at every grid edge."""
    def top(a):  # count of scores >= EDGES[i]
        return np.cumsum(a[sel].sum(0)[::-1])[::-1][1:]
    nwin, npos = h.nwin[sel].sum(), h.npos[sel].sum()
    return {"recall": top(h.pos) / npos, "fp": top(h.neg) / nwin, "any": top(h.wmax) / nwin}


def at_recall(c: dict, r: float) -> tuple[float, float]:
    if not (c["recall"].min() <= r <= c["recall"].max()):
        return float("nan"), float("nan")
    rr = c["recall"][::-1]
    return float(np.interp(r, rr, c["fp"][::-1])), float(np.interp(r, rr, c["any"][::-1]))


def op(h: Hist, sel: np.ndarray, rule: str = "p>0.1") -> dict:
    fp, hits, anyfp = h.ops[rule][sel].sum(0)
    nwin, npos = h.nwin[sel].sum(), h.npos[sel].sum()
    return {"fp_per_window": fp / nwin, "p_any_fp": anyfp / nwin, "recall": hits / npos if npos else None,
            "windows": int(nwin), "positives": int(npos)}


def score_set(set_name: str, lst: dict, packs: dict, heads: dict, cl_index: dict,
              teacher_pass: bool) -> tuple[dict, dict]:
    """Histograms of every (list, model) on one set, accumulated chunk by chunk; plus bug-check numbers."""
    from teacher.embed_perch import perch_classes

    manifest, id_col = SETS[set_name][:2]
    man = pd.read_parquet(DATA_ROOT / manifest)
    ids = man[id_col].astype(str).tolist()
    cl = np.array([cl_index[f"WABAD:{s}"] for s in man.site]) if set_name == "WABAD" else np.full(
        len(man), cl_index[set_name])
    ys = {}
    for name, (codes, _) in lst.items():
        ci = {c: j for j, c in enumerate(codes)}
        y = np.zeros((len(man), len(codes)), bool)
        for i, labs in enumerate(man["labels"]):
            y[i, [ci[c] for c in labs if c in ci]] = True
        ys[name] = y
    hists, checks = {}, {}

    def add(name: str, model: str, s: np.ndarray, rows: slice):
        hists.setdefault((name, model), Hist(len(cl_index))).add(s, ys[name][rows], cl[rows])

    # cached JP scorers (the paper's table)
    cached = {}
    for m, key in (("teacher", "perchfold"), ("v24", "v24")):
        z = np.load(CACHE / f"openset317_{key}_{set_name}.npz")
        assert z["ids"].tolist() == ids and z["codes"].tolist() == lst["JP"][0]
        cached[m] = z["scores"]
    if teacher_pass:  # the region lists need the teacher's full-softmax pass
        t = np.load(OUT / f"teacher_{set_name}.npz")
        assert t["ids"].tolist() == ids
        tcol = {int(v): j for j, v in enumerate(t["vidx"])}
        tl_all = torch.from_numpy(t["logits"].astype(np.float32)) - torch.from_numpy(t["logz"])[:, None]
    feats = {}
    for tier in TIERS:
        f = (np.load(DISTILL_ROOT / f"packfit_v1/feat_wabad_{tier}/part-0000.npz") if set_name == "WABAD"
             else np.load(OUT / f"feat_{set_name}_{tier}.npz"))
        assert f["ids"].tolist() == ids, f"{set_name}/{tier}: window order"
        feats[tier] = torch.from_numpy(f["feat"])
    paper = {}  # the paper's cached exact-fold scores on the set species (bug check)
    for tier, run in {"B": "g3r_regnety_016", "S": "g3r_efficientnet_lite0", "XS": "g2r_full"}.items():
        pc = np.load(CACHE / f"distill_{run}_{set_name}.npz")
        assert pc["ids"].tolist() == ids
        pidx = np.array([c["perch_idx"] for c in perch_classes(pc["codes"].tolist())])
        paper[tier] = (pc["scores"][:, pidx >= 0], torch.from_numpy(pidx[pidx >= 0]))
    err = {k: 0.0 for k in ("teacher_pass_vs_cache_maxabs_p", *(f"{t}_exact_vs_paper_maxabs_p" for t in TIERS))}
    fires = {t: 0 for t in TIERS}
    for i in range(0, len(man), CHUNK):
        rows = slice(i, i + CHUNK)
        for m, sc in cached.items():
            s = sc[rows].astype(np.float64)
            add("JP", m, np.log(s) - np.log1p(-s) if m == "teacher" else s, rows)
        for name, (_, vidx) in lst.items() if teacher_pass else ():
            s = logit_from_logp(tl_all[rows][:, [tcol.get(int(v), 0) for v in vidx]].double()).numpy()
            s[:, vidx < 0] = -np.inf
            if name == "JP":  # bug check only; the JP teacher row is the paper's cache
                err["teacher_pass_vs_cache_maxabs_p"] = max(err["teacher_pass_vs_cache_maxabs_p"], float(
                    np.abs(1 / (1 + np.exp(-s)) - cached["teacher"][rows]).max()))
            else:
                add(name, "teacher", s, rows)
        for tier in TIERS:
            h = feats[tier][rows].float()
            W, b = heads[tier]
            lg = h @ W.t() + b
            logz = torch.logsumexp(lg, 1, keepdim=True)
            ps, pidx = paper[tier]
            err[f"{tier}_exact_vs_paper_maxabs_p"] = max(err[f"{tier}_exact_vs_paper_maxabs_p"], float(
                np.abs(torch.exp(lg[:, pidx] - logz).numpy() - ps[rows]).max()))
            for name, (_, vidx) in lst.items():
                dead = torch.from_numpy(vidx < 0)
                s = logit_from_logp((lg[:, np.maximum(vidx, 0)] - logz).double())
                s[:, dead] = -np.inf
                add(name, f"{tier}_exact", s.numpy(), rows)
                Wp, bp, n_sp = packs[(name, tier)]
                pl = h @ Wp.t() + bp
                s = logit_from_logp((pl[:, :n_sp] - torch.logsumexp(pl, 1, keepdim=True)).double())
                s[:, dead] = -np.inf
                add(name, f"{tier}_pack", s.numpy(), rows)
                if name == "JP":
                    fires[tier] += int((s > OPS["p>0.1"]).sum())
    checks |= err
    for tier in TIERS:  # the released JP pack applied here vs the cached ONNX pack scores (opensetpub_v1b)
        z = np.load(CACHE / f"openset317_kinglet032_{TIERS[tier]}_{set_name}.npz")
        checks[f"{tier}_pack_fires_mine_vs_onnx"] = [fires[tier], int((z["scores"] > 0.1).sum())]
    return hists, checks


def load_hists() -> tuple[dict, dict]:
    z = np.load(OUT / "hists.npz")
    hists = {}
    for k in z.files:
        if k.count("|") != 2:
            continue
        n, m, a = k.split("|")
        h = hists.setdefault((n, m), Hist(len(z["clusters"])))
        if a.startswith("op"):
            h.ops[a[2:]] = z[k]
        else:
            setattr(h, a, z[k])
    checks = (json.loads(str(z["checks"])) if "checks" in z.files  # older file: checks from the last summary
              else json.loads((RESULTS / "opensetcurve_v1.json").read_text())["checks"])
    return hists, checks


def merge(a: Hist, b: Hist) -> Hist:
    for k in ("pos", "neg", "wmax", "nwin", "npos"):
        setattr(a, k, getattr(a, k) + getattr(b, k))
    for k in OPS:
        a.ops[k] += b.ops[k]
    return a


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lists", nargs="+", default=["JP", "DE", "CR"], help="JP alone needs no teacher pass")
    ap.add_argument("--sets", nargs="+", default=PUBLIC_SETS, help="subset = screening only (outputs not written)")
    ap.add_argument("--from-hists", action="store_true", help="re-summarise the saved histograms (hists.npz)")
    args = ap.parse_args()
    lst = {k: v for k, v in lists().items() if k in args.lists}
    names, out_of = clusters()
    if args.from_hists:
        hists, checks = load_hists()
        return summarize(hists, names, out_of, checks)
    cl_index = {n: i for i, n in enumerate(names)}
    heads, packs = {}, {}
    for tier, tag in TIERS.items():
        hp_path = RELEASE_ASSETS / f"kinglet-{tag}-v0.3.2.headpack.npz"
        hp = np.load(hp_path)
        W, b = torch.from_numpy(hp["W"].astype(np.float32)), torch.from_numpy(hp["b"].astype(np.float32))
        heads[tier] = (W, b)
        rel = np.load(RELEASE_ASSETS / f"kinglet-{tag}-v0.3.2.pack-jp.npz")
        packs[("JP", tier)] = (torch.from_numpy(rel["W"]), torch.from_numpy(rel["b"]), int(rel["n_species"]))
        live = lst["JP"][1] >= 0
        assert np.allclose(rel["W"][:317][live], hp["W"][lst["JP"][1][live]].astype(np.float32), atol=1e-6)
        lab = {str(n): i for i, n in enumerate(hp["labels"])}
        midx = [lab[m] for ms in json.loads(str(rel["groups"])).values() for m in ms]
        bundle = Bundle(hp_path, W, b)
        for name in set(lst) - {"JP"}:  # compact pack = kinglet build-pack --compact with the released groups
            vidx = lst[name][1]
            live_idx = vidx[vidx >= 0]
            assert len(set(live_idx)) == len(live_idx) and not set(live_idx) & set(midx)
            wb = bundle.other_row(torch.tensor(list(live_idx) + midx))
            rows = np.concatenate([np.maximum(vidx, 0), midx])
            Wp = torch.cat([W[rows], wb[None, :-1]])
            bp = torch.cat([b[rows], wb[-1:]])
            bp[: len(vidx)][torch.from_numpy(vidx < 0)] = -1e9
            packs[(name, tier)] = (Wp, bp, len(vidx))
        del bundle
    hists, checks = {}, {}
    teacher_pass = set(lst) != {"JP"} and all((OUT / f"teacher_{s}.npz").exists() for s in args.sets)
    print("teacher pass for the region lists:", teacher_pass)
    for set_name in args.sets:
        hs, ck = score_set(set_name, lst, packs, heads, cl_index, teacher_pass)
        checks[set_name] = ck
        print(set_name, json.dumps(ck), flush=True)
        for k, h in hs.items():
            hists[k] = merge(hists[k], h) if k in hists else h
    if list(args.sets) != list(PUBLIC_SETS):
        return
    np.savez_compressed(OUT / "hists.npz", **{f"{n}|{m}|{a}": getattr(h, a) for (n, m), h in hists.items()
                                              for a in ("pos", "neg", "wmax", "nwin", "npos")},
                        **{f"{n}|{m}|op{k}": h.ops[k] for (n, m), h in hists.items() for k in OPS},
                        clusters=np.array(names), edges=EDGES, checks=json.dumps(checks))
    summarize(hists, names, out_of, checks)


def summary(hists: dict, names: list[str], out_of: dict, checks: dict) -> tuple[dict, list]:
    """Per list: operating points, matched-recall points and the cluster bootstrap; and the curve rows."""
    rng = np.random.default_rng(0)
    res, rows = {"checks": checks}, []
    for name in [n for n in ("JP", "DE", "CR") if any(k[0] == n for k in hists)]:
        sel_names = [n for n in names if n in out_of[name]]
        sel = np.isin(names, sel_names)
        models = sorted(m for (n, m) in hists if n == name)
        has_t = (name, "teacher") in hists
        targets = {"0.3": 0.3, "0.4": 0.4, "0.5": 0.5}
        if has_t:
            targets = {"teacher_recall@0.1": op(hists[(name, "teacher")], sel)["recall"]} | targets
        block = {"clusters": len(sel_names), "models": {}}
        for m in models:
            h = hists[(name, m)]
            c = curve(h, sel)
            e = {"p>0.1": op(h, sel)}
            if m == "v24":
                e["p>0.25"] = op(h, sel, "p>0.25")
            for k, r in targets.items():
                fp, anyfp = at_recall(c, r)
                e[f"at_recall_{k}"] = {"recall": r, "fp_per_window": fp, "p_any_fp": anyfp}
            e["max_recall"] = float(c["recall"].max())
            block["models"][m] = e
            for i in range(0, NB - 1, 5):
                if c["recall"][i] > 0:
                    rows.append((name, m, round(EDGES[i], 3), 1 / (1 + np.exp(-EDGES[i])), c["recall"][i], c["fp"][i],
                                 c["any"][i]))
        # cluster bootstrap of R = FP(model) / FP(teacher) at the teacher's recall at p > 0.1
        idx = np.flatnonzero(sel)
        boot = {m: [] for m in models if m != "teacher"} if has_t else {}
        for _ in range(2000 if has_t else 0):
            cnt = np.bincount(rng.choice(idx, len(idx)), minlength=len(names))
            th = hists[(name, "teacher")]
            w = cnt.astype(float)

            def wcurve(h):
                def top(a):
                    return np.cumsum((a * w[:, None]).sum(0)[::-1])[::-1][1:]
                return {"recall": top(h.pos) / (h.npos * w).sum(), "fp": top(h.neg) / (h.nwin * w).sum(),
                        "any": top(h.wmax) / (h.nwin * w).sum()}
            r = (th.ops["p>0.1"][:, 1] * w).sum() / (th.npos * w).sum()
            fpt = at_recall(wcurve(th), r)[0]
            for m in boot:
                boot[m].append(at_recall(wcurve(hists[(name, m)]), r)[0] / fpt)
        for m, v in boot.items():
            v = np.array(v)
            pt = block["models"][m]["at_recall_teacher_recall@0.1"]["fp_per_window"] / block["models"]["teacher"][
                "at_recall_teacher_recall@0.1"]["fp_per_window"]
            block["models"][m]["R_vs_teacher_at_teacher_recall"] = {
                "R": pt, "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
                "nan_share": float(np.isnan(v).mean())}
        res[name] = block
        print(name, json.dumps({m: {"op": {k: round(v, 4) for k, v in e["p>0.1"].items() if isinstance(v, float)},
                                    "R": e.get("R_vs_teacher_at_teacher_recall")}
                                for m, e in block["models"].items()}, default=str), flush=True)
    return res, rows


def summarize(hists: dict, names: list[str], out_of: dict, checks: dict):
    res, rows = summary(hists, names, out_of, checks)
    (RESULTS / "opensetcurve_v1.json").write_text(json.dumps({k: res[k] for k in ("checks", "JP")}, indent=1,
                                                                  default=float))
    if "DE" in res:
        (RESULTS / "regionlists_v1.json").write_text(json.dumps({k: res[k] for k in ("checks", "DE", "CR")},
                                                                     indent=1, default=float))
    pd.DataFrame(rows, columns=["list", "model", "threshold_logit", "threshold_prob", "recall", "fp_per_window",
                                "p_any_fp"]).to_csv(CSV, index=False, float_format="%.6g")
    print("wrote", RESULTS / "opensetcurve_v1.json", RESULTS / "regionlists_v1.json", "and", CSV)


if __name__ == "__main__":
    main()
