# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""BirdNET+ V3.0 (Zenodo record 24162518) on WABAD and the eight BirdSet test sets.

Same windows and metrics as every other model (common.scoring.evaluate_set). Two protocols per run (RUNS):
  crops3    each 5-s window = mean logit of the three overlapping 3-s crops 0-3, 1-4, 2-5 s (as BirdNET V2.4)
  5s        the whole 5-s window as one input
  --run v1  caches v3rel (crops3) and v3rel5s (5s); species matched by scientific name (common.scoring.label_index
            with BIRDNET_OVERRIDES["v3"]); input of the crop-aggregation diagnostic (stage diag)
  --run v2  the paper's numbers: caches v3relc5s (5s, main) and v3relc (crops3); species matched by the label file's
            ebird_code column, the v1 rule only where the code is not in the file
Logits = the input of the graph's final Sigmoid (added as an output). A digitally silent input gives NaN (the
official BirdNET+ code maps it to confidence 0, i.e. nothing detected): a silent crop is left out of the crop mean
(e.g. the zero padding at the end of a short clip); an input or window with no non-silent crop scores MISSING + 1
(below every real score, above species the model does not have). Both are counted.

One pass per set writes, per protocol, <proto>_<set>.npz on the set's species (benchmarks) and
openset317_<proto>_<set>.npz on the 317-species Japanese list (stress test), both in common.scoring.CACHE.

  python evaluate/accuracy/birdnetv3rel.py check [--gpu]        # bug checks: (a) md5, (b) crop path vs onnxruntime
  python evaluate/accuracy/birdnetv3rel.py score --run v1 [--gpu]
  python evaluate/accuracy/birdnetv3rel.py score --run v2 [--gpu]
  python evaluate/accuracy/birdnetv3rel.py report --run v2     # -> results/birdnetv3rel_v2.json and the BirdNET+
                                                               #    rows (list JP) of figure_data/openset_curves.csv
  python evaluate/accuracy/birdnetv3rel.py diag --gpu --sets POW HSN NBP --run v1  # cmAP under other crop aggregations
--gpu: onnxruntime-gpu (CUDA execution provider, TF32 off) from --ort-path, which goes first on sys.path.
The model files go in DATA_ROOT/birdnet_v3/models_release_24162518 (see common/config.py).
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, FIGURE_DATA, RESULTS, jp_list  # noqa: E402
from common.scoring import (BIRDNET_OVERRIDES, CACHE, MISSING, SETS, evaluate_set, label_index,  # noqa: E402
                            set_codes)
from teacher.embed_perch import SR, decode, windows  # noqa: E402

REL_DIR = DATA_ROOT / "birdnet_v3/models_release_24162518"
REL_MODEL = REL_DIR / "BirdNET+_V3.0_Global_12K_FP32.onnx"
REL_LABELS = REL_DIR / "BirdNET+_V3.0_Global_12K_Labels.csv"
MD5 = {REL_MODEL.name: "550fbfc3d4c1750026c5d588611197bf", REL_LABELS.name: "bb25847af8a581e1382081382f01e762"}
RUNS = {"v1": {"protos": {"v3rel": "crops3", "v3rel5s": "5s"}, "main": "v3rel", "match": "name"},
        "v2": {"protos": {"v3relc5s": "5s", "v3relc": "crops3"}, "main": "v3relc5s", "match": "code"}}
OUT = RESULTS / "birdnetv3rel_v1.json"  # set per run in main()
CURVES = FIGURE_DATA / "openset_curves.csv"
BENCH_SETS = ("WABAD", "POW", "HSN", "NES", "PER", "SNE", "SSW", "UHH", "NBP")
OPENSET_SETS = BENCH_SETS[:-1]  # the 381,911 windows of the stress test
PROTOS = tuple(RUNS["v1"]["protos"])  # set per run in main()
MATCH = "name"  # set per run in main()
RUNS_MAIN = [RUNS["v1"]["main"]]  # set per run in main()
CHUNK, BATCH = 512, 128  # recordings per decode chunk, 5-s windows per model call (GPU-efficient)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jp_codes() -> list[str]:
    return [c["species"] for c in jp_list()]


def label_sci(path: Path) -> list[str]:
    """Scientific name per output, in output order."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert df.iloc[:, 0].tolist() == list(range(len(df))), "labels not in output order"
    return df["scientific_name"].astype(str).tolist()


def model_index(net, codes: list[str], match: str) -> tuple[np.ndarray, np.ndarray]:
    """eBird code -> output index (-1 = no class) and whether it came from the label file's ebird_code column.
    match 'name': common.scoring.label_index; 'code': the ebird_code column first, label_index only for codes not in
    the column."""
    by_name = label_index(net.sci, codes, BIRDNET_OVERRIDES["v3"])
    if match == "name":
        return by_name, np.zeros(len(codes), bool)
    by_code = np.array([net.code_pos.get(c, -1) for c in codes])
    return np.where(by_code >= 0, by_code, by_name), by_code >= 0


class Net:
    """BirdNET+ V3.0 ONNX as a logit function on (B, samples) at 32 kHz; CPU or CUDA."""

    def __init__(self, path: Path, labels: Path, gpu: bool, threads: int):
        import onnx
        import onnxruntime as ort

        m = onnx.load(str(path))
        outs = {o.name for o in m.graph.output}
        sig = [n for n in m.graph.node if n.op_type == "Sigmoid" and n.output[0] in outs]
        assert len(sig) == 1, [n.output[0] for n in sig]
        self.prob_name, self.logit_name = sig[0].output[0], sig[0].input[0]
        m.graph.output.append(onnx.helper.make_tensor_value_info(self.logit_name, onnx.TensorProto.FLOAT, None))
        opt = ort.SessionOptions()
        opt.intra_op_num_threads, opt.inter_op_num_threads = threads, 1
        prov = ([("CUDAExecutionProvider", {"use_tf32": 0, "cudnn_conv_algo_search": "DEFAULT"})] if gpu else []) + [
            "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(m.SerializeToString(), opt, providers=prov)
        assert self.sess.get_providers()[0] == ("CUDAExecutionProvider" if gpu else "CPUExecutionProvider")
        self.inp = self.sess.get_inputs()[0].name
        self.sci = label_sci(labels)
        codes = pd.read_csv(labels)["ebird_code"]
        self.code_pos = {c: i for i, c in enumerate(codes) if isinstance(c, str)}
        assert len(self.code_pos) == codes.notna().sum(), "duplicate ebird_code"
        self.n_nan = self.n_silent_windows = 0  # silent model inputs; windows without any non-silent input
        self.lock = threading.Lock()

    def raw(self, x: np.ndarray) -> np.ndarray:
        """(B, samples) -> logits (B, classes); NaN rows for digitally silent inputs."""
        prob, logit = self.sess.run([self.prob_name, self.logit_name], {self.inp: np.ascontiguousarray(x, np.float32)})
        assert logit.shape == (len(x), len(self.sci)), logit.shape
        ok = np.isfinite(logit)
        assert (ok.all(1) | ~ok.any(1)).all(), "NaN in part of a row"
        live = ok.all(1)
        assert np.allclose(1 / (1 + np.exp(-logit[live][:2].astype(np.float64))), prob[live][:2], atol=1e-5)
        with self.lock:
            self.n_nan += int((~live).sum())
        return logit

    def silent_to_missing(self, logit: np.ndarray) -> np.ndarray:
        bad = np.isnan(logit).all(1)
        with self.lock:
            self.n_silent_windows += int(bad.sum())
        return np.where(bad[:, None], MISSING + 1, logit)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(B, samples) -> logits, one input per row."""
        return self.silent_to_missing(self.raw(x))

    def crops3(self, x: np.ndarray) -> np.ndarray:
        """(B, 5 s) -> mean logit of the 0-3 / 1-4 / 2-5 s crops (silent crops left out)."""
        c = np.concatenate([x[:, s * SR : (s + 3) * SR] for s in range(3)])
        lg = self.raw(c).reshape(3, len(x), -1)
        n = (~np.isnan(lg[:, :, 0])).sum(0)[:, None]
        return self.silent_to_missing(np.where(n > 0, np.nansum(lg, 0) / np.maximum(n, 1), np.nan))


def caches(name: str, set_name: str) -> list[Path]:
    return [CACHE / f"{name}_{set_name}.npz", CACHE / f"openset317_{name}_{set_name}.npz"]


def score_set(net: Net, set_name: str, protos: dict[str, str], decoders: int = 8, limit: int = 0) -> dict:
    """One pass over a set; protos = {cache name: 'crops3' | '5s'}. Returns {(name, 'bench'|'jp'): cache dict}."""
    manifest, id_col, off_col, max_s, hop, _ = SETS[set_name]
    man = pd.read_parquet(DATA_ROOT / manifest)
    rows = list(zip(man[id_col].astype(str), man["path"], man[off_col] if off_col else [0.0] * len(man)))[
        : limit or None]
    lists = {"bench": set_codes(set_name), "jp": jp_codes()}
    idx = {k: model_index(net, c, MATCH)[0] for k, c in lists.items()}
    cols = np.unique(np.concatenate([i[i >= 0] for i in idx.values()]))
    pos = {k: np.searchsorted(cols, np.maximum(i, 0)) for k, i in idx.items()}
    pool, prefetch, runners = ThreadPoolExecutor(decoders), ThreadPoolExecutor(1), ThreadPoolExecutor(2)

    def run(x):  # two batches in flight, so host-side work overlaps the GPU (onnxruntime releases the GIL)
        return [(net.crops3(x) if how == "crops3" else net(x))[:, cols].astype(np.float32) for how in protos.values()]

    def load(chunk):
        return list(pool.map(lambda r: decode(r[1], max_s, r[2]), chunk))

    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    nxt, meta, out, t0 = prefetch.submit(load, chunks[0]), [], {n: [] for n in protos}, time.time()
    for k, chunk in enumerate(chunks):
        audio = nxt.result()
        if k + 1 < len(chunks):
            nxt = prefetch.submit(load, chunks[k + 1])
        wins = []
        for (rid, _, _), a in zip(chunk, audio):
            assert a is not None, f"decode failed: {rid}"
            starts, w = windows(a, hop)
            meta += [(rid, i, s) for i, s in enumerate(starts)]
            wins += list(w)
        for r in runners.map(run, [np.stack(wins[b : b + BATCH]) for b in range(0, len(wins), BATCH)]):
            for n, lg in zip(protos, r):
                out[n].append(lg)
        if k % 20 == 0 or k + 1 == len(chunks):
            print(f"[{set_name}] {min((k + 1) * CHUNK, len(rows))}/{len(rows)} recordings, {len(meta)} windows, "
                  f"{(time.time() - t0) / 60:.1f} min, silent inputs {net.n_nan}, silent windows/inputs "
                  f"{net.n_silent_windows}", flush=True)
    res = {}
    for n in protos:
        lg = np.concatenate(out[n])
        for k, codes in lists.items():
            res[(n, k)] = {"ids": np.array([m[0] for m in meta]), "window_idx": np.array([m[1] for m in meta], np.int32),
                           "t_start": np.array([m[2] for m in meta], np.float32),
                           "scores": np.where(idx[k][None] >= 0, lg[:, pos[k]], MISSING).astype(np.float32),
                           "codes": np.array(codes),
                           "unmatched": np.array([c for c, i in zip(codes, idx[k]) if i < 0], dtype=str)}
    return res


def save(res: dict, set_name: str):
    for (n, k), r in res.items():
        dst = caches(n, set_name)[0 if k == "bench" else 1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst.with_suffix(".tmp.npz"), **r)
        dst.with_suffix(".tmp.npz").rename(dst)


# ------------------------------------------------------------------------------------------------ bug checks

def check(args) -> dict:
    out = {"md5": {f: md5(REL_DIR / f) for f in MD5}}
    assert out["md5"] == MD5, out["md5"]  # (a)
    # (b) 32 real WABAD windows: crops3 through this code (CPU, and the GPU provider when used) vs a direct
    # onnxruntime CPU call on the unmodified graph. The direct call gives float32 probabilities, which can be inverted
    # to logits precisely only where p is neither saturated (logit < 8: error < 2e-4) nor tiny (the Sigmoid kernel
    # has an absolute error ~1e-7, so logit > -9); probabilities are compared everywhere.
    import onnxruntime as ort

    man = pd.read_parquet(DATA_ROOT / SETS["WABAD"][0]).iloc[::1880][:32]
    x = np.stack([windows(decode(p, 0, 0.0), 5.0)[1][0] for p in man["path"]])
    crops = np.concatenate([x[:, s * SR : (s + 3) * SR] for s in range(3)]).astype(np.float32)
    direct = ort.InferenceSession(str(REL_MODEL), providers=["CPUExecutionProvider"])
    p = direct.run(["predictions"], {direct.get_inputs()[0].name: crops})[0].astype(np.float64)
    ref = np.log(p) - np.log1p(-p)
    out["b_crops3_vs_direct"] = {"windows": len(x), "outputs": [o.name for o in direct.get_outputs()]}
    for gpu in sorted({False, args.gpu}):
        rel = Net(REL_MODEL, REL_LABELS, gpu, args.threads)
        out["release_outputs"] = {"prob": rel.prob_name, "logit": rel.logit_name, "n_classes": len(rel.sci)}
        per_crop = rel(crops).astype(np.float64)
        mine = rel.crops3(x)
        inv = ((ref > -9) & (ref < 8)).reshape(3, len(x), -1).all(0)
        out["b_crops3_vs_direct"][rel.sess.get_providers()[0]] = {
            "max_abs_prob_diff": float(np.abs(1 / (1 + np.exp(-per_crop)) - p).max()),
            "max_abs_logit_diff_crops3": float(np.abs(mine - ref.reshape(3, len(x), -1).mean(0))[inv].max()),
            "share_invertible": float(inv.mean()),
            "top1_equal": float((mine.argmax(1) == ref.reshape(3, len(x), -1).mean(0).argmax(1)).mean())}
        del rel
    print("(b)", out["b_crops3_vs_direct"], flush=True)
    return out


# ------------------------------------------------------------------------------------------------ report

def per_species(caches_by_model: dict[str, str]) -> pd.DataFrame:
    """AP per (benchmark, annotated species) for the teacher, B/S/XS and the given BirdNET caches;
    in_all_models over every model's coverage."""
    from sklearn.metrics import average_precision_score

    import figures.build_figure_data as bfd

    rows = []
    for name in BENCH_SETS:
        saved = dict(bfd.MODELS)
        bfd.MODELS.update(caches_by_model)
        try:
            scores, y, _ = bfd.load_set(name)
        finally:
            bfd.MODELS.clear()
            bfd.MODELS.update(saved)
        codes = np.load(CACHE / f"{bfd.MODELS['B']}_{name}.npz")["codes"].tolist()
        perch = {c["species"]: c["perch_idx"]
                 for c in json.loads((DATA_ROOT / SETS[name][-1] / "classes.json").read_text())}
        has = {m: set(codes) - set(np.load(CACHE / f"{c}_{name}.npz")["unmatched"].tolist())
               for m, c in (saved | caches_by_model).items()}
        has["teacher"] = {c for c in codes if perch[c] is not None and perch[c] >= 0}
        for j in np.flatnonzero(y.any(0)):
            rows.append({"benchmark": name, "species": codes[j], "in_all_models": all(codes[j] in h for h in has.values())}
                        | {m: average_precision_score(y[:, j], s[:, j]) for m, s in scores.items()})
    return pd.DataFrame(rows)


def species_readouts(df: pd.DataFrame) -> dict:
    from scipy.stats import pearsonr, spearmanr

    out = {"agreement_pooled": {"n_pairs": len(df), "r": round(pearsonr(df["birdnet_v3"], df["teacher"])[0], 3),
                                "rho": round(spearmanr(df["birdnet_v3"], df["teacher"])[0], 3)},
           "common": {}}
    for name in BENCH_SETS:
        d = df[df["benchmark"] == name]
        c = d[d["in_all_models"]]
        out["common"][name] = {"n_species": len(d), "n_common": len(c)} | {
            m: {"cmAP_all": round(d[m].mean(), 4), "cmAP_common": round(c[m].mean(), 4)}
            for m in ("teacher", "B", "S", "XS", "birdnet_v3", "birdnet_v24")}
    return out


def openset_hist(name: str):
    """Japanese-list histograms of one cached BirdNET+ scorer on the eight benchmarks (opensetcurve rules)."""
    from evaluate.species_lists.opensetcurve import Hist, clusters

    names, _ = clusters()
    cl_index = {n: i for i, n in enumerate(names)}
    codes = jp_codes()
    ci = {c: j for j, c in enumerate(codes)}
    h = Hist(len(names))
    for s in OPENSET_SETS:
        manifest, id_col = SETS[s][:2]
        man = pd.read_parquet(DATA_ROOT / manifest)
        z = np.load(CACHE / f"openset317_{name}_{s}.npz")
        assert z["ids"].tolist() == man[id_col].astype(str).tolist() and z["codes"].tolist() == codes
        y = np.zeros((len(man), len(codes)), bool)
        for i, labs in enumerate(man["labels"]):
            y[i, [ci[c] for c in labs if c in ci]] = True
        cl = (np.array([cl_index[f"WABAD:{x}"] for x in man.site]) if s == "WABAD" else np.full(len(man), cl_index[s]))
        for i in range(0, len(man), 4096):
            h.add(z["scores"][i : i + 4096].astype(np.float64), y[i : i + 4096], cl[i : i + 4096])
    return h


def report(args) -> dict:
    from sklearn.metrics import average_precision_score

    from evaluate.species_lists.opensetcurve import clusters, load_hists, summary

    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    # benchmarks: evaluate_set on each protocol, plus WABAD per-site AUPRC
    teacher = pd.read_csv(FIGURE_DATA / "benchmarks.csv").query("model == 'teacher'").set_index("benchmark")["cmAP"]
    res["benchmarks"] = {}
    for proto in PROTOS:
        block = {}
        for s in BENCH_SETS:
            r = dict(np.load(caches(proto, s)[0]))
            ev = evaluate_set(s, r, proto)
            annotated = set(c for x in pd.read_parquet(DATA_ROOT / SETS[s][0])["labels"] for c in x)
            block[s] = {k: ev[k] for k in ("cmAP", "pooled_auprc", "auroc", "n_species", "n_windows", "all_finite")} | {
                "cmAP_vs_teacher": ev["cmAP"] / teacher[s], "n_unmatched_codes": len(ev["unmatched"]),
                "n_unmatched_annotated": len(set(ev["unmatched"]) & annotated), "unmatched_annotated":
                sorted(set(ev["unmatched"]) & annotated)}
            if s == "WABAD":
                meta = pd.read_parquet(DATA_ROOT / SETS[s][0]).set_index("window_id").loc[r["ids"]]
                y = np.array([[c in x for c in r["codes"]] for x in meta["labels"]])
                site = meta["site"].to_numpy()
                block["WABAD_per_site_auprc"] = {
                    n: float(average_precision_score(y[site == n][:, y[site == n].any(0)].ravel(),
                                                     r["scores"][site == n][:, y[site == n].any(0)].ravel()))
                    for n in np.unique(site)}
        res["benchmarks"][proto] = block
        print(proto, {s: round(block[s]["cmAP"], 4) for s in BENCH_SETS}, flush=True)
    # species matching on each benchmark: by the label file's ebird_code (v2), by the fallback / name rule, none
    net = Net.__new__(Net)
    net.sci, lab = label_sci(REL_LABELS), pd.read_csv(REL_LABELS)
    net.code_pos = {c: i for i, c in enumerate(lab["ebird_code"]) if isinstance(c, str)}
    by_name_key = "mapping_diagnostic_ebird_code" if MATCH == "name" else "mapping"
    res[by_name_key] = {}
    for s in (*BENCH_SETS, "JP317"):
        codes = jp_codes() if s == "JP317" else sorted(
            set(c for x in pd.read_parquet(DATA_ROOT / SETS[s][0])["labels"] for c in x))
        idx, via_code = model_index(net, codes, MATCH)
        name_idx = model_index(net, codes, "name")[0]
        alt = np.array([net.code_pos.get(c, -1) for c in codes])
        res[by_name_key][s] = {
            "annotated" if s != "JP317" else "listed": len(codes), "matched": int((idx >= 0).sum()),
            "matched_by_ebird_code": int(via_code.sum()), "matched_by_name_rule": int(((idx >= 0) & ~via_code).sum()),
            "unmatched": [c for c, i in zip(codes, idx) if i < 0],
            "class_differs_from_name_rule": sorted(
                f"{c}: {lab.scientific_name[n] if n >= 0 else '-'} -> {lab.scientific_name[i] if i >= 0 else '-'}"
                for c, n, i in zip(codes, name_idx, idx) if n != i) if MATCH == "code" else sorted(
                f"{c}: {lab.scientific_name[n] if n >= 0 else '-'} / {lab.scientific_name[a_] if a_ >= 0 else '-'}"
                for c, n, a_ in zip(codes, name_idx, alt) if n != a_)}
    if MATCH == "code":  # bug check: where both rules pick the same class, the v2 columns equal v1's
        res["bug_check_same_class_equals_v1"] = {}
        for s in BENCH_SETS:
            for new, old in zip(PROTOS, ("v3rel5s", "v3rel")):
                a_, b_ = np.load(caches(new, s)[0]), np.load(caches(old, s)[0])
                codes = a_["codes"].tolist()
                same = model_index(net, codes, "code")[0] == model_index(net, codes, "name")[0]
                same &= model_index(net, codes, "name")[0] >= 0
                assert a_["ids"].tolist() == b_["ids"].tolist()
                d = float(np.abs(a_["scores"][:, same] - b_["scores"][:, same]).max())
                res["bug_check_same_class_equals_v1"][f"{s} {new}"] = {"columns": int(same.sum()), "max_abs_diff": d}
                assert d < 1e-3, (s, new, d)
    # species agreement with the teacher and common-species cmAP (Appendix S2), each protocol in turn
    res["species"] = {proto: species_readouts(per_species({"birdnet_v3": proto})) for proto in PROTOS}
    # Japanese list on the eight benchmarks (Fig. S3, Table S3): the BirdNET+ histograms of each protocol
    hists, checks = load_hists()
    names, out_of = clusters()
    jp = {k: v for k, v in hists.items() if k[0] == "JP"}
    res["openset_JP"] = {}
    rows = None
    for proto in PROTOS:
        jp[("JP", "v3")] = openset_hist(proto)
        out, r = summary(jp, names, out_of, checks)
        res["openset_JP"][proto] = out["JP"]["models"]["v3"]
        if proto == RUNS_MAIN[0]:
            rows = [x for x in r if x[1] == "v3"]
    res["openset_JP"]["v3"] = res["openset_JP"][RUNS_MAIN[0]]  # the entry the figure/table builders read (main)
    cur = pd.read_csv(CURVES)
    new = pd.DataFrame(rows, columns=cur.columns)
    pd.concat([cur[~((cur["list"] == "JP") & (cur["model"] == "v3"))], new]).to_csv(CURVES, index=False,
                                                                                   float_format="%.6g")
    OUT.write_text(json.dumps(res, indent=1, default=float))
    return res


def diag(args) -> dict:
    """How the crop aggregation moves cmAP (run v1 caches). One pass per set over the three 3-s crops and the 5-s
    window; per-crop logits aggregated as mean logit (main protocol), max logit, mean probability, and each crop
    alone. Silent crops left out as in Net.crops3; a silent single crop scores MISSING + 1."""
    net = Net(REL_MODEL, REL_LABELS, args.gpu, args.threads)
    out = {}
    for s in args.sets:
        r0 = dict(np.load(caches("v3rel", s)[0]))
        idx = label_index(net.sci, r0["codes"].tolist(), BIRDNET_OVERRIDES["v3"])
        man = pd.read_parquet(DATA_ROOT / SETS[s][0]).set_index(SETS[s][1]).loc[r0["ids"]]
        crops, five = [], []
        for i in range(0, len(man), BATCH):
            x = np.stack([windows(decode(p_, 0, 0.0), 5.0)[1][0] for p_ in man["path"].iloc[i : i + BATCH]])
            c = np.concatenate([x[:, k * SR : (k + 3) * SR] for k in range(3)])
            crops.append(net.raw(c).reshape(3, len(x), -1)[:, :, np.maximum(idx, 0)])
            five.append(net(x)[:, np.maximum(idx, 0)])
        crops, five = np.concatenate(crops, 1).astype(np.float64), np.concatenate(five).astype(np.float64)
        with np.errstate(all="ignore"):  # silent crops (NaN) left out, as in Net.crops3
            variants = {"mean_logit (main)": np.nanmean(crops, 0), "max_logit": np.nanmax(crops, 0),
                        "mean_prob": np.log(np.nanmean(1 / (1 + np.exp(-crops)), 0)),
                        **{f"crop_{k}-{k + 3}s": crops[k] for k in range(3)}, "whole_5s": five}
        variants = {k: np.nan_to_num(v, nan=MISSING + 1) for k, v in variants.items()}
        assert np.allclose(variants["mean_logit (main)"][:, idx >= 0], r0["scores"][:, idx >= 0], atol=1e-3)
        out[s] = {}
        for k, v in variants.items():
            ev = evaluate_set(s, r0 | {"scores": np.where(idx[None] >= 0, v, MISSING).astype(np.float32)}, k)
            out[s][k] = {"cmAP": ev["cmAP"], "pooled_auprc": ev["pooled_auprc"]}
        print(s, {k: round(v["cmAP"], 4) for k, v in out[s].items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("check", "score", "report", "diag"))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--ort-path", default=None, help="directory holding onnxruntime-gpu (put first on sys.path)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sets", nargs="+", default=list(BENCH_SETS))
    ap.add_argument("--run", choices=list(RUNS), default="v1")
    args = ap.parse_args()
    global OUT, PROTOS, MATCH
    OUT, PROTOS, MATCH = RESULTS / f"birdnetv3rel_{args.run}.json", tuple(RUNS[args.run]["protos"]), RUNS[args.run]["match"]
    RUNS_MAIN[0] = RUNS[args.run]["main"]
    if args.ort_path:
        import torch  # noqa: F401  loads the CUDA / cuDNN libraries of the torch wheels for onnxruntime-gpu

        sys.path.insert(0, args.ort_path)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.stage == "check":
        res["bug_checks"] = res.get("bug_checks", {}) | check(args)
    elif args.stage == "score":
        net = Net(REL_MODEL, REL_LABELS, args.gpu, args.threads)
        for s in args.sets:
            if all(p.exists() for n in PROTOS for p in caches(n, s)):
                continue
            save(score_set(net, s, RUNS[args.run]["protos"]), s)
        res.setdefault("silent", {})[" ".join(args.sets)] = {"silent_model_inputs": net.n_nan,
                                                             "windows_or_inputs_without_sound": net.n_silent_windows}
    elif args.stage == "diag":
        res["crop_aggregation_diagnostic_post_hoc"] = diag(args)
    else:
        res = report(args)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
