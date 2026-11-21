# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Build the small CSV files behind the figures and tables from the score caches and result files.

Reads (set the locations in paper/config.yaml): the per-model score caches of the benchmarks
(common.scoring.CACHE), the Perch 2.0 teacher outputs on the benchmarks (DATA_ROOT), the WABAD site metadata
(WABAD_SRC/Metadata.csv), the embeddings, checkpoints and microcontroller files under DISTILL_ROOT, the released
reliability table (RELEASE_ASSETS) and the result files in paper/results.
Writes to paper/figure_data:
  benchmarks.csv         cmAP and pooled AUPRC per benchmark x model
  wabad_per_site.csv     pooled AUPRC per WABAD site x model, restricted to the site species list
  embcompat_cos_hist.csv histogram of per-window cos(teacher, student) on WABAD
  speed.csv              end-to-end CPU throughput per model, laptop and thread count (speed kit)
  table1_tiers.csv       Table 1: backbone, parameters, GMACs, throughput and accuracy per model
  table3_mcu.csv         Table S6: microcontroller measurements (which int8 model each value is from)
  mcu_power_trace.csv    Fig. 4b: board power over four inference windows (1 ms means)
  fig1_spectrogram.csv   Fig. 1: log-mel spectrogram of one WABAD window (8-bit, every second frame)
  tier_stratified.csv    accuracy relative to the teacher by number of training recordings of the class
  species_ap.csv         AP per annotated species and benchmark for the teacher and the three students

Every recomputed headline number is asserted against the stored result. BirdNET+ V3.0 = the final release
(Zenodo 24162518), each 5-s window as one input, species matched by the label file's ebird_code (cache v3relc5s).

Usage: python figures/build_figure_data.py [table ...]   (no argument: build all tables)
table1_tiers loads the checkpoints with PyTorch (CPU); fig1_spectrogram needs torchaudio.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_ROOT, DISTILL_ROOT, FIGURE_DATA, RELEASE_ASSETS, RESULTS, WABAD_SRC  # noqa: E402
from common.scoring import CACHE, SETS, cmap, load_parts  # noqa: E402

WABAD_META = WABAD_SRC / "Metadata.csv"
MODELS = {"B": "distill_g3r_regnety_016", "S": "distill_g3r_efficientnet_lite0",
          "XS": "distill_g2r_full", "birdnet_v3": "v3relc5s", "birdnet_v24": "v24"}
BENCHMARKS = ("WABAD", "POW", "HSN", "NES", "PER", "SNE", "SSW", "UHH")
# scored after the analyses built on the eight above (tierstrat_v1, table1_tiers, revstats_v1);
# joins benchmarks.csv only, as a set not used in development (birdsetnbp_v1)
LATE_BENCHMARKS = ("NBP",)
# also used as gates while the training recipe was developed; every other set was first scored after the
# models were frozen
DEV_SETS = ("POW", "HSN")


def load_set(name: str) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Scores per model (teacher first), label matrix and window metadata for one benchmark."""
    manifest, id_col, *_, emb_dir = SETS[name]
    teacher = load_parts(DATA_ROOT / emb_dir, keys=("ids", "logits"))
    teacher_codes = [c["species"] for c in json.loads((DATA_ROOT / emb_dir / "classes.json").read_text())]
    scores, codes = {}, None
    for model, cache_name in MODELS.items():
        r = np.load(CACHE / f"{cache_name}_{name}.npz", allow_pickle=True)
        codes = codes or r["codes"].tolist()
        assert r["ids"].tolist() == teacher["ids"].tolist(), f"{name}/{model}: window order"
        assert r["codes"].tolist() == codes, f"{name}/{model}: species order"
        scores[model] = r["scores"]
    # teacher logits may cover a superset of the benchmark species (POW/HSN share one class list)
    t = teacher["logits"].astype(np.float32)[:, [teacher_codes.index(c) for c in codes]]
    # species without a Perch class score below everything, as in the teacher runs
    scores = {"teacher": np.nan_to_num(t, nan=np.nanmin(t) - 1)} | scores
    meta = pd.read_parquet(DATA_ROOT / manifest).set_index(id_col).loc[teacher["ids"]]
    y = np.array([[c in x for c in codes] for x in meta["labels"]])
    return scores, y, meta


def benchmarks() -> pd.DataFrame:
    registered = json.loads((DISTILL_ROOT / "perch_teacher_birdset7.json").read_text())
    registered["WABAD"] = json.loads((DISTILL_ROOT / "perch_teacher_wabad.json").read_text())
    registered["NBP"] = json.loads((DISTILL_ROOT / "perch_teacher_nbp.json").read_text())
    rows = []
    for name in (*BENCHMARKS, *LATE_BENCHMARKS):
        scores, y, _ = load_set(name)
        for model, s in scores.items():
            rows.append({"benchmark": name, "model": model, "development_set": name in DEV_SETS, "n_windows": len(y),
                         "n_species": int(y.any(0).sum()), "cmAP": cmap(s, y),
                         "pooled_auprc": average_precision_score(y.ravel(), s.ravel())})
        t = rows[-len(scores)]
        assert abs(t["cmAP"] - registered[name]["cmAP"]) < 5e-4, (name, t, registered[name])
        print(name, {r["model"]: round(r["cmAP"], 4) for r in rows[-len(scores):]}, flush=True)
    return pd.DataFrame(rows)


def wabad_per_site() -> pd.DataFrame:
    scores, y, meta = load_set("WABAD")
    site = meta["site"].to_numpy()
    rows = []
    for name in np.unique(site):
        m = site == name
        sp = y[m].any(0)
        rows.append({"site": name, "n_windows": int(m.sum()), "n_species": int(sp.sum())} | {
            model: average_precision_score(y[m][:, sp].ravel(), s[m][:, sp].ravel())
            for model, s in scores.items()})
    df = pd.DataFrame(rows)
    info = pd.read_csv(WABAD_META, encoding="latin-1").rename(columns={
        "Site ID": "site", "Recording location": "country", "Biome": "biome",
        "Latitude": "lat", "Longitude": "lon"})
    info["site"] = info["site"].str.rstrip("*")
    df = df.merge(info[["site", "country", "biome", "lat", "lon"]], on="site", validate="1:1")
    assert len(df) == 72 and df[["lat", "lon"]].notna().all().all()
    old = pd.read_csv(DISTILL_ROOT / "wabad_per_site.csv").set_index("site").loc[df["site"]]
    # the stored per-site file is from wabad_v1; wabad_v1b changed the teacher and student scores of
    # 52 species (mapping fix), so only BirdNET V2.4 is expected to reproduce it; BirdNET+ V3.0 is checked
    # against birdnetv3rel_v2
    assert np.abs(df["birdnet_v24"].to_numpy() - old["v24"].to_numpy()).max() < 1e-3
    rel = json.loads((RESULTS / "birdnetv3rel_v2.json").read_text())["benchmarks"][MODELS["birdnet_v3"]][
        "WABAD_per_site_auprc"]
    assert np.abs(df["birdnet_v3"].to_numpy() - df["site"].map(rel).to_numpy()).max() < 1e-9
    return df


def embcompat_cos_hist() -> pd.DataFrame:
    from evaluate.embeddings.probe_wabad_emb import EMB_DIRS

    def unit(d: Path) -> np.ndarray:
        e = load_parts(d, keys=("emb",))["emb"].astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    registered = json.loads((RESULTS / "wabad_embcompat_results.json").read_text())
    teacher = unit(EMB_DIRS["teacher"])
    edges = np.linspace(0, 1, 101)
    rows = []
    for tier in ("B", "S", "XS"):
        cos = (teacher * unit(EMB_DIRS[tier])).sum(1)
        assert abs(cos.mean() - registered[tier]["field_cos_mean"]) < 1e-3, tier
        assert cos.min() >= 0, f"{tier}: cos below histogram range ({cos.min():.3f})"
        rows.append(pd.DataFrame({"tier": tier, "cos_lo": edges[:-1], "n": np.histogram(cos, edges)[0]}))
    return pd.concat(rows)


# laptop (CPU name, the `machine` column of speed.csv) -> label of its speed-kit result file (speedmulti_v1)
SPEED_MACHINES = {"i7-8565U": "i7-8565U", "M4 Pro": "M4-Pro", "Ryzen AI 7 PRO 350": "Ryzen-AI-7-PRO-350"}


def speed() -> pd.DataFrame:
    """End-to-end CPU throughput (x real time; decode + segmentation + model, what a user waits for) of the six
    models as normally used, from the speed kit (compute_speed/speed_kit, speedmulti_v1): BirdNET V2.4 and BirdNET+
    V3.0 on consecutive 3-s segments, Kinglet with the full output layer read at the Japanese list. One row per
    model, laptop and thread count; the Intel Core i7-8565U laptop (4 threads) is the reference of Table 1 and
    Fig. 2a. Secondary rows (overlap, 5-s, compact layer, convolution DFT) stay in the kit files (Appendix S4).

    The bug checks of speedmulti_v1 are asserted here. psutil cannot read the AC state of the i7-8565U laptop
    (plugged_in None); the run note records AC.
    """
    keys = {"teacher": "perch", "birdnet_v3": "birdnet_v3", "birdnet_v24": "birdnet_v24",
            "B": "kinglet_b", "S": "kinglet_s", "XS": "kinglet_xs"}
    rows = []
    for machine, label in SPEED_MACHINES.items():
        res = json.loads((RESULTS / f"speedkit_{label}.json").read_text())
        info, runs = res["machine"], res["runs"]
        assert info["load_before"] < 10 and info["power"]["battery"]["plugged_in"] is not False, machine
        assert all(v == "ok" for f in res["model_files"].values() for v in (f.values() if isinstance(f, dict) else [f]))
        assert all(r["status"] == "ok" and max(r["xrt_runs"]) / min(r["xrt_runs"]) < 1.2
                   and r["xrt"] < res["decode"]["xrt"] for r in runs), machine
        xrt = {(r["model"], r["threads"]): r["xrt"] for r in runs}
        rows += [{"model": m, "machine": machine, "threads": t, "xrt_e2e": xrt[k, t]}
                 for t in sorted({t for _, t in xrt}) for m, k in keys.items()]
    return pd.DataFrame(rows)


# tier -> (backbone, checkpoint under DISTILL_ROOT, GMACs per window registered in count_macs_v1)
TIERS = {"B": ("RegNetY-016", "g3conf_v1/regnety_016_full", 2.09),
         "S": ("EfficientNet-Lite0", "g3conf_v1/efficientnet_lite0_full", 0.49),
         "XS": ("MobileNetV4-Conv-S", "g2xcl_v1/mobilenetv4_conv_small_full", 0.23)}


def gmacs(net, x) -> float:
    """Multiply-accumulates (1e9) of the convolution and linear layers for one input."""
    import torch

    total = [0]

    def hook(m, _, out):  # multiplies per output element: Cin/groups * kh * kw, or in_features
        total[0] += out.numel() * (m.weight[0].numel() if isinstance(m, torch.nn.Conv2d) else m.in_features)

    hooks = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        net(x)
    for h in hooks:
        h.remove()
    return total[0] / 1e9


def table1_tiers() -> pd.DataFrame:
    """One row per model: size (parameters of the released trunk ONNX = backbone + embedding head), compute,
    end-to-end throughput (speed.csv) and accuracy (benchmarks.csv)."""
    import torch
    from evaluate.accuracy.score_student import Distilled

    bench, xrt = pd.read_csv(FIGURE_DATA / "benchmarks.csv"), pd.read_csv(FIGURE_DATA / "speed.csv").query(
        "machine == 'i7-8565U' and threads == 4").set_index("model")["xrt_e2e"]  # the reference laptop
    cmap_ = bench.pivot(index="benchmark", columns="model", values="cmAP")
    rel = cmap_.div(cmap_["teacher"], axis=0).drop(index=["WABAD", *LATE_BENCHMARKS])  # the seven BirdSet sets
    dev, rel = rel.loc[list(DEV_SETS)], rel.drop(index=list(DEV_SETS))  # development gates / the five test-only sets
    wabad = bench[bench["benchmark"] == "WABAD"].set_index("model")
    rows = []
    for model in ("teacher", *TIERS):
        row = {"model": "Perch 2.0" if model == "teacher" else f"Kinglet {model}"}
        if model == "teacher":
            # "an EfficientNet-B3 [...] with 12 million parameters" on 500 x 128 log-mel frames
            # (arXiv:2508.04665, section 2.2). The paper states no compute cost and the released CPU
            # model is one XLA-compiled module that cannot be counted here, so the GMACs stay empty.
            row |= {"backbone": "EfficientNet-B3", "params_M": 12.0}
        if model in TIERS:
            backbone, ckpt, registered = TIERS[model]
            net = Distilled(DISTILL_ROOT / ckpt / "best.pt", "cpu").model
            n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6  # noqa: E731
            g = gmacs(net.backbone, torch.zeros(1, 1, 128, 501))
            assert abs(g / registered - 1) < 0.03, (model, g, registered)
            row |= {"backbone": backbone, "params_M": n(net.backbone) + n(net.emb_head),  # the released trunk ONNX
                    "gmacs_per_window": g}  # counted on the checkpoint
        row |= {"xrt_e2e": xrt[model], "hours_per_1000h": 1000 / xrt[model],
                "wabad_cmAP": wabad.loc[model, "cmAP"], "wabad_pooled_auprc": wabad.loc[model, "pooled_auprc"]}
        if model in TIERS:
            row |= {"wabad_cmAP_vs_teacher": wabad.loc[model, "cmAP"] / wabad.loc["teacher", "cmAP"],
                    "birdset5_cmAP_vs_teacher_mean": rel[model].mean(), "birdset5_cmAP_vs_teacher_min": rel[model].min(),
                    "birdset5_cmAP_vs_teacher_max": rel[model].max(), "birdset_dev_cmAP_vs_teacher_mean": dev[model].mean()}
        rows.append(row)
    return pd.DataFrame(rows)


def table3_mcu() -> pd.DataFrame:
    """Microcontroller measurements, each with the int8 model it was taken on. Timing, memory and the
    test-window check come from the board log of the int8 model built from public audio only (ptqpub_v1),
    power from its three 60-s runs; accuracy rows from the same int8 model on the host (ptqpub_v1, mcuemb_v2)
    and the host emulation of the capture rate (mcurate_v1b). Files under DISTILL_ROOT/xs_pub_deploy."""
    dep_pub = DISTILL_ROOT / "xs_pub_deploy"
    uart, vela = (dep_pub / "fw_xs_pub_uart.log").read_text(), (dep_pub / "xs_pub_vela.log").read_text()
    arena = int(re.search(r"arena used: (\d+) bytes", uart).group(1))
    sram, flash = (float(re.search(rf"Total {k} used\s+([\d.]+) KiB", vela).group(1)) for k in ("SRAM", "Off-chip Flash"))
    golden_dp = {int(x) / 1000 for x in re.findall(r"max \|dp\| (\d+) milli", uart)}
    assert uart.count("[golden] PASS") == 2 and len(golden_dp) == 1
    dev16 = np.frombuffer(bytes.fromhex(re.search(r"first16 ([0-9a-f]{32})", uart).group(1)), np.uint8).astype(int)
    host16 = np.load(dep_pub / "golden_emb_int8.npy")[:16].astype(int) % 256
    clock = int(re.search(r"pll (\d+) Hz", uart).group(1))  # the firmware prints ms truncated; use cycles
    mel, npu = (round(1000 * np.median([int(x) for x in re.findall(pat, uart)]) / clock)
                for pat in (r"mel (\d+) cyc", r"npu\+cpu (\d+) cyc"))
    rate = int(re.search(r"-> (\d+) Hz", uart).group(1))
    # power from the 1-ms traces at full precision (the logs round mA to 2 decimals, which moves the
    # energy per window by up to 0.5 mJ): steady state = mean over ten whole windows from 7 s on, after
    # start-up; baseline = level between inferences as mcu/measure_power.py defines it (median of the
    # 10-ms smoothed trace); window period = 160,000 samples at 31,250 Hz
    period = 160_000 / 31_250
    traces = [np.load(dep_pub / f"power_xspub_5v_60s_r{i}.npz")["trace_ma"] for i in (1, 2, 3)]
    steady_ma = [float(tr[7000 : 7000 + round(10 * period * 1000)].mean()) for tr in traces]
    base_ma = [float(np.median(np.convolve(tr, np.ones(10) / 10, mode="same"))) for tr in traces]
    steady = [5.0 * s for s in steady_ma]
    energy = [(s - b) * 5.0 * period for s, b in zip(steady_ma, base_ma)]  # mJ above the level between inferences, per window
    logged = [float(re.search(r"steady-state mean over \d+ windows: [\d.]+ mA \(([\d.]+) mW\)", (dep_pub / f"power_xspub_5v_60s_r{i}.log").read_text()).group(1)) for i in (1, 2, 3)]
    assert all(abs(a - b) < 0.06 for a, b in zip(steady, logged)), (steady, logged)
    ptq = json.loads((RESULTS / "ptqpub_v1.json").read_text())
    probe = json.loads((RESULTS / "wabad_embprobe_results.json").read_text())
    compat = json.loads((RESULTS / "wabad_embcompat_results.json").read_text())
    i8, fp = probe["XS_int8pub"] | compat["XS_int8pub"], probe["XS"] | compat["XS"]
    ratios = [i8[k] / fp[k] for k in fp if k.endswith(("mAP", "cmAP")) or k == "field_cos_mean"]
    mcurate = json.loads((RESULTS / "mcurate_v1b.json").read_text())
    assert (mel, npu, rate, period) == (108, 366, 31250, 5.12) and 110.0 < min(steady) <= max(steady) < 110.5
    pub, board = "public-only int8 (ptqpub_v1)", "public-only int8 on the board"
    fw = "firmware (does not depend on the model)"
    size_mb = (dep_pub / "kinglet-xs_pub-jp-int8.tflite").stat().st_size / 1e6
    vela_mb = (dep_pub / "kinglet-xs_pub-jp-int8_vela.tflite").stat().st_size / 1e6
    rng = lambda v, f: f.format(min(v)) if f.format(min(v)) == f.format(max(v)) else f"{f.format(min(v))}-{f.format(max(v))}"  # noqa: E731
    rows = [
        ("sampling rate of the audio interface", rate, "Hz", fw, "fw_xs_pub_uart.log"),
        ("window period (160,000 samples)", period, "s", fw, "160,000 / capture rate; NPU onsets in power_xspub_5v_60s_r*.npz"),
        ("spectrogram, per window", mel, "ms", fw, "fw_xs_pub_uart.log"),
        ("network, per window", npu, "ms", board, "fw_xs_pub_uart.log"),
        ("processing time / audio duration", round((mel + npu) / (1000 * period), 3), "", board, "derived"),
        ("mean board power", rng(steady, "{:.1f}"), "mW", board, "power_xspub_5v_60s_r*.npz (5.0 V, 60 s x 3)"),
        ("board power between inferences", rng([5.0 * b for b in base_ma], "{:.1f}"), "mW", board, "same traces"),
        ("inference energy above the level between inferences, per window", rng(energy, "{:.1f}"), "mJ", board, "derived from the same traces"),
        ("inference share of board energy", rng([100 * e / (s * period) for e, s in zip(energy, steady)], "{:.1f}"), "%", board, "derived"),
        ("8-bit model file before compilation", round(size_mb, 2), "MB", pub, "kinglet-xs_pub-jp-int8.tflite"),
        ("compiled model file written to the board", round(vela_mb, 2), "MB", board, "kinglet-xs_pub-jp-int8_vela.tflite"),
        ("working memory reported by the compiler", round(sram, 1), "KiB", pub, "xs_pub_vela.log"),
        ("flash memory reported by the compiler", round(flash, 1), "KiB", pub, "xs_pub_vela.log"),
        ("memory reserved for the network", arena, "B", board, "fw_xs_pub_uart.log"),
        ("board vs computer, 8-bit model, test window: largest difference in probability", golden_dp.pop(), "", board, "fw_xs_pub_uart.log"),
        ("board vs computer, 8-bit model, test window: largest difference in the first 16 feature bytes", int(np.abs(dev16 - host16).max()), "quantisation steps", board, "fw_xs_pub_uart.log"),
        ("8-bit vs floating-point model: agreement on the top class (2,048 WABAD windows)", ptq["parity_wabad_2048"]["public_only"]["top1_agreement"], "", pub, "ptqpub_v1.json"),
        ("8-bit vs floating-point model: windows whose detections differ", 100 * ptq["parity_wabad_2048"]["public_only"]["window_fire_disagree"], "%", pub, "ptqpub_v1.json"),
        ("recovered embedding vs floating-point embedding, cosine similarity (mean)", i8["fp32_cos_mean"], "", pub, "wabad_embcompat_results.json"),
        ("retrieval and classifier performance relative to floating-point XS", rng([100 * r for r in ratios], "{:.1f}"), "%", pub, "mcuemb_v2"),
        ("WABAD cmAP at the board's sampling rate relative to 32 kHz", round(mcurate["cmAP_rate_over_baseline"], 3), "", "fp32 XS on the host (mcurate_v1b)", "mcurate_v1b.json"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "unit", "measured_on", "source"])


def mcu_power_trace() -> pd.DataFrame:
    """Board power (5.0 V source meter, 1 ms means) over four inference windows, int8 model of ptqpub_v1, run 2."""
    z = np.load(DISTILL_ROOT / "xs_pub_deploy/power_xspub_5v_60s_r2.npz")
    assert abs(float(z["sps"]) / 1e5 - 1) < 1e-3 and int(z["mv"]) == 5000  # no dropped samples
    t = np.arange(len(z["trace_ma"])) / 1000.0
    keep = (t >= 6.5) & (t < 24.0)
    df = pd.DataFrame({"t_s": t[keep] - 6.5, "mW": z["trace_ma"][keep] * 5.0})
    assert 109.5 < df["mW"].mean() < 111.0
    return df


def fig1_spectrogram() -> pd.DataFrame:
    """Log-mel (the student front end: 128 bands, 10 ms hop) of one WABAD window with two annotated
    species (common cuckoo and song thrush, site BIAL), scaled to 0-255; rows = mel bands, low first."""
    import torch
    import torchaudio
    from teacher.embed_perch import SR, decode
    from train.backbones import MEL

    man = pd.read_parquet(DATA_ROOT / SETS["WABAD"][0]).set_index("window_id")
    row = man.loc["BIAL_20210521_051901_09"]
    assert sorted(row["labels"]) == ["comcuc", "sonthr1"]
    x = torch.from_numpy(decode(row["path"], 0, 0.0).copy())[:160_000]
    m = torch.log(torchaudio.transforms.MelSpectrogram(sample_rate=SR, **MEL)(x) + 1e-6).numpy()[:, ::2]
    lo, hi = np.percentile(m, [1, 99.5])
    return pd.DataFrame(np.clip(255 * (m - lo) / (hi - lo), 0, 255).round().astype(int))


RELIABILITY = RELEASE_ASSETS / "kinglet-v0.3.2.reliability.csv"


def tier_stratified() -> pd.DataFrame:
    """tierstrat_v1: per-species AP grouped by how many training recordings the class had (the released
    reliability tiers; softlabel_only = never positive in the training audio). Mean AP per model and the
    student/teacher ratio of means, pooled over the eight benchmarks (species-set pairs) and for WABAD
    alone, with a 95% bootstrap interval over species.

    The tiers are those of the released table (release/export_reliability.py), which assigns training species to
    classes by the rule the benchmarks use and counts each recording once."""
    tier = pd.read_csv(RELIABILITY).set_index("vocab_index")["reliability_tier"]
    registered = pd.read_csv(FIGURE_DATA / "benchmarks.csv").set_index(["benchmark", "model"])["cmAP"]
    per_species = []
    for name in BENCHMARKS:
        scores, y, _ = load_set(name)
        codes = np.load(CACHE / f"{MODELS['B']}_{name}.npz", allow_pickle=True)["codes"].tolist()
        perch = {c["species"]: c["perch_idx"] for c in json.loads((DATA_ROOT / SETS[name][-1] / "classes.json").read_text())}
        d = pd.DataFrame({"benchmark": name, "species": codes, "positives": y.sum(0)})
        d["tier"] = [tier[perch[c]] if perch[c] is not None and perch[c] >= 0 else "no_teacher_class" for c in codes]
        d = d[d["positives"] > 0].copy()
        for model in ("teacher", *TIERS):
            d[model] = [average_precision_score(y[:, j], scores[model][:, j]) for j in d.index]
            assert abs(d[model].mean() - registered[name, model]) < 5.1e-5, (name, model)  # the file has 4 decimals
        per_species.append(d)
    per_species = pd.concat(per_species, ignore_index=True)
    rng, rows = np.random.default_rng(0), []
    for scope, d in (("pooled", per_species), ("WABAD", per_species[per_species["benchmark"] == "WABAD"])):
        for t in ("high", "medium", "low", "softlabel_only", "no_teacher_class"):
            g = d[d["tier"] == t]
            pick = rng.integers(0, len(g), (2000, len(g)))  # the same species for student and teacher
            teacher = g["teacher"].to_numpy()
            for model in TIERS:
                student = g[model].to_numpy()
                lo, hi = np.percentile(student[pick].mean(1) / teacher[pick].mean(1), [2.5, 97.5])
                rows.append({"scope": scope, "tier": t, "n_species": len(g), "model": model,
                             "median_positive_windows": g["positives"].median(), "teacher_mean_ap": teacher.mean(),
                             "student_mean_ap": student.mean(), "ratio": student.mean() / teacher.mean(),
                             "ratio_lo": lo, "ratio_hi": hi})
    return pd.DataFrame(rows)


def species_ap() -> pd.DataFrame:
    """AP of the teacher and of B/S/XS for every annotated species of every benchmark (species-benchmark pairs,
    NBP included); the rows behind revision_stats.json 'agreement' (pooled Pearson r per student)."""
    rows = []
    for name in (*BENCHMARKS, *LATE_BENCHMARKS):
        scores, y, _ = load_set(name)
        for j in np.flatnonzero(y.any(0)):
            rows.append({"benchmark": name, "positives": int(y[:, j].sum())}
                        | {m: average_precision_score(y[:, j], scores[m][:, j]) for m in ("teacher", *TIERS)})
    df = pd.DataFrame(rows)
    registered = json.loads((RESULTS / "revision_stats.json").read_text())["agreement"]["species"]["pooled"]
    assert len(df) == registered["n_species"]
    for m in TIERS:
        assert abs(np.corrcoef(df[m], df["teacher"])[0, 1] - registered[m]["r"]) < 1e-3, m
    return df


if __name__ == "__main__":
    tables = {fn.__name__: fn for fn in (wabad_per_site, embcompat_cos_hist, benchmarks, speed, table1_tiers,
                                         table3_mcu, mcu_power_trace, fig1_spectrogram, tier_stratified, species_ap)}
    for name in sys.argv[1:] or tables:  # optional arguments: rebuild only the named tables
        tables[name]().to_csv(FIGURE_DATA / f"{name}.csv", index=False, float_format="%.4f")
        print("wrote", name, flush=True)
