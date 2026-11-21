# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Build the Supporting Information tables S1-S9 (Markdown) from the figure data and the result files.

Usage: python figures/build_si_tables.py  ->  writes figures/si_tables.md
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import FIGURE_DATA as DATA, RESULTS  # noqa: E402

HERE = Path(__file__).resolve().parent

MODEL_NAMES = {
    "teacher": "Perch 2.0",
    "B": "Kinglet B",
    "S": "Kinglet S",
    "XS": "Kinglet XS",
    "birdnet_v3": "BirdNET+ V3.0",
    "birdnet_v24": "BirdNET V2.4",
}
TIERS = ("B", "S", "XS")
S1_SETS = ["WABAD", "NES", "PER", "SNE", "SSW", "UHH", "NBP", "POW", "HSN"]  # NBP: birdsetnbp_v1
DEV_SETS = {"POW", "HSN"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load(name):
    return json.loads((RESULTS / name).read_text())


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def set_label(s):
    return s + "†" if s in DEV_SETS else s


def ci(point, lo_hi, fmt="{:.2f}", sep="–"):
    return f"{fmt.format(point)} ({fmt.format(lo_hi[0])}{sep}{fmt.format(lo_hi[1])})"


def table_s1():
    """Accuracy of all models; for the students the cmAP ratio to the teacher with its species-bootstrap interval
    and the ratio when the teacher is scored by its full softmax (revstats_v1)."""
    by = {(r["benchmark"], r["model"]): r for r in read_csv(DATA / "benchmarks.csv")}
    rs = load("revision_stats.json")
    out = []
    for s in S1_SETS:
        r0 = by[(s, "teacher")]
        for m, name in MODEL_NAMES.items():
            r = by[(s, m)]
            ratio = soft = ""
            if m in TIERS:
                b = rs["bootstrap"]["species"][f"{s} {m}"]
                ratio = ci(b["ratio"], b["ratio_ci"])
                soft = f'{rs["softmax"][s][m]["cmAP_ratio_softmax"]:.2f}'
            elif m == "teacher":
                soft = f'cmAP {rs["softmax"][s]["cmAP_softmax"]:.4f}'
            out.append([set_label(s) if m == "teacher" else "",
                        f'{int(r0["n_windows"]):,} / {int(r0["n_species"]):,}' if m == "teacher" else "",
                        name, f'{float(r["cmAP"]):.4f}', f'{float(r["pooled_auprc"]):.4f}', ratio, soft])
    for m in TIERS:
        b = rs["bootstrap"]["species"][f"test6_mean {m}"]
        out.append(["Six held-out BirdSet sets, mean", "", MODEL_NAMES[m], "", "", ci(b["ratio"], b["ratio_ci"]),
                    f'{rs["softmax"][f"test6_mean {m}"]["cmAP_ratio_softmax"]:.2f}'])
    return md_table(["Benchmark", "Windows / species", "Model", "cmAP", "Pooled AUPRC",
                     "cmAP ratio to Perch 2.0 (95% CI)", "Ratio, teacher scored by softmax"], out)


def table_s2():
    rows = [r for r in read_csv(DATA / "tier_stratified.csv") if r["scope"] == "pooled"]
    diffs = load("revision_stats.json")["groups"]["diff_vs_high"]
    labels = {"high": "≥ 100", "medium": "10–99", "low": "1–9",
              "softlabel_only": "0", "no_teacher_class": "no class in Perch 2.0"}
    keys = {"medium": "10-99", "low": "1-9", "softlabel_only": "none"}
    out = []
    for r in rows:
        d = diffs.get(f'{keys[r["tier"]]} {r["model"]}') if r["tier"] in keys else None
        out.append([labels[r["tier"]], r["n_species"], r["model"],
                    f'{float(r["teacher_mean_ap"]):.3f}', f'{float(r["student_mean_ap"]):.3f}',
                    ci(float(r["ratio"]), (float(r["ratio_lo"]), float(r["ratio_hi"]))),
                    ci(d["diff"], d["ci"], "{:+.3f}", " to ") if d else ""])
    return md_table(["Training recordings", "Species-benchmark pairs", "Model", "Teacher mean AP", "Student mean AP",
                     "Ratio (95% CI)", "Ratio difference from ≥ 100 (95% CI)"], out)


def table_s3():
    """Species lists on foreign soundscapes (opensetcurve_v1, regionlists_v1; BirdNET+ V3.0 = final release,
    birdnetv3rel_v2)."""
    jp = load("opensetcurve_v1.json")["JP"]["models"] | {"v3": load("birdnetv3rel_v2.json")["openset_JP"]["v3"]}
    lists = [("Japan (317)", jp),
             ("Germany (496)", load("regionlists_v1.json")["DE"]["models"]),
             ("Costa Rica (949)", load("regionlists_v1.json")["CR"]["models"])]
    names = [("teacher", "Perch 2.0"), ("B_exact", "Kinglet B, full"), ("S_exact", "Kinglet S, full"),
             ("XS_exact", "Kinglet XS, full"), ("B_pack", "Kinglet B, compact"), ("S_pack", "Kinglet S, compact"),
             ("XS_pack", "Kinglet XS, compact"), ("v3", "BirdNET+ V3.0"), ("v24", "BirdNET V2.4")]
    out = []
    for label, models in lists:
        first = True
        for k, name in names:
            if k not in models:
                continue
            m = models[k]
            p, eq = m["p>0.1"], m["at_recall_teacher_recall@0.1"]
            r = m.get("R_vs_teacher_at_teacher_recall")
            out.append([label if first else "", name, f'{p["fp_per_window"]:.3f}', f'{100 * p["p_any_fp"]:.1f}',
                        f'{100 * p["recall"]:.0f}', f'{eq["fp_per_window"]:.3f}',
                        ci(r["R"], r["ci95"]) if r else "1"])
            first = False
    return md_table(["List", "Model / output layer", "False / window (p > 0.1)", "Windows with ≥ 1 false (%)",
                     "Recall (%)", "False / window at the teacher's recall", "Ratio to Perch 2.0 (95% CI)"], out)


def table_s4():
    """Taxa without training audio (nonbird_v1)."""
    d = load("nonbird_v1.json")
    sets = [("watkins", "Watkins (marine mammals)"), ("anuraset", "AnuraSet (frogs)"),
            ("humbugdb", "HumBugDB (mosquitoes)"), ("reefset", "ReefSet (coral reefs)")]
    out = []
    for key, name in sets:
        e, h = d[key]["embedding"], d[key]["head"]
        t = e["teacher"]
        ap = h.get("macro_AP", {})  # absent where no class maps to the vocabulary
        out.append([name, "Perch 2.0", str(h["n_scored"]), f'{ap["teacher_logit"]:.3f}' if h["n_scored"] else "", "", "",
                    f'{t["retrieval_k1_mAP"]:.3f}', f'{t["retrieval_k5_mAP"]:.3f}', f'{t["probe_cmAP"]:.3f}', "", "", ""])
        if h["n_scored"]:
            out.append(["", "Perch 2.0, with its probabilities", "", f'{ap["teacher_softmax"]:.3f}'] + [""] * 8)
        for m in TIERS:
            r = e[f"ratio_{m}"]
            head = ([f'{ap[m]:.3f}', f'{ap[m] / ap["teacher_logit"]:.2f}', f'{ap[m] / ap["teacher_softmax"]:.2f}']
                    if h["n_scored"] else ["", "", ""])
            out.append(["", MODEL_NAMES[m], "", *head, f'{r["retrieval_k1_mAP"]:.2f}', f'{r["retrieval_k5_mAP"]:.2f}',
                        f'{r["probe_cmAP"]:.2f}', f'{r["mean_within"]:.2f}', f'{r["xretrieval_t2s_k1_mAP"]:.2f}',
                        f'{r["xprobe_cmAP"]:.2f}'])
    return md_table(["Dataset", "Model", "Classes scored", "Output layer: mean AP", "Ratio to teacher (class scores)",
                     "Ratio to teacher (probabilities)", "Retrieval, 1 example",
                     "Retrieval, 5 examples", "Linear classifier", "Mean of the three",
                     "Teacher queries, student archive", "Teacher classifier, without refitting"], out)


def table_s5():
    rows = read_csv(DATA / "wabad_per_site.csv")
    out = []
    for r in sorted(rows, key=lambda r: (r["country"], r["site"])):
        out.append([r["site"], r["country"], r["n_windows"], r["n_species"]]
                   + [f'{float(r[m]):.3f}' for m in MODEL_NAMES])
    return md_table(["Site", "Country", "Windows", "Species"] + list(MODEL_NAMES.values()), out)


def table_s6():
    rows = read_csv(DATA / "table3_mcu.csv")
    return md_table(["Quantity", "Value", "Unit"], [[r["quantity"], r["value"], r["unit"]] for r in rows])


def table_s7():
    """Computation and parameters of the distributed models (count_macs_v2: Kinglet with the FFT front end)."""
    d = load("count_macs_v2.json")
    g = lambda x: f"{x / 1e9:.3f}"  # noqa: E731
    mp = lambda x: f"{x / 1e6:.2f}"  # noqa: E731
    t = d["teacher"]
    # trainable parameters of backbone + embedding layer as in Table 1 (batch-norm scale and shift included,
    # running statistics excluded); count_macs_v1 lists the same parts with batch norm folded into the convolutions
    params = {r["model"]: float(r["params_M"]) for r in read_csv(DATA / "table1_tiers.csv")}
    out = [["Perch 2.0", g(t["macs"]["frontend"]), g(t["macs"]["backbone"]), "0", g(t["macs"]["label_head"]), "—",
            g(t["macs"]["total"]), mp(t["params"]["backbone"]), mp(t["params"]["label_head"]), "—"]]
    for m in TIERS:
        s = d["students"][m.lower()]["fft_frontend"]
        out.append([MODEL_NAMES[m], g(s["macs"]["frontend"]), g(s["macs"]["backbone"]), g(s["macs"]["embedding_head"]),
                    g(s["macs"]["class_head_full"]), f'{s["macs"]["class_head_jp_pack"] / 1e9:.4f}',
                    g(s["macs"]["total_exact"]), f'{params[MODEL_NAMES[m]]:.2f}',
                    mp(s["params"]["class_head_full"]), mp(s["params"]["class_head_jp_pack"])])
    return md_table(["Model", "Spectrogram", "Backbone", "Embedding layer", "Full output layer",
                     "Compact output layer (Japan)", "Total, full output layer",
                     "Parameters: backbone and embedding (M)", "Full output layer (M)", "Compact, Japan (M)"], out)


def table_s8():
    """XS run-to-run variation and loss ablation (xsvar_v1 / xsabl_v1)."""
    s, probe, compat = load("xsvar_v1_summary.json"), load("xsvar_v1_embprobe.json"), load("xsvar_v1_embcompat.json")
    teacher = {r["benchmark"]: float(r["cmAP"]) for r in read_csv(DATA / "benchmarks.csv") if r["model"] == "teacher"}
    six = [x for x in S1_SETS[1:] if x not in DEV_SETS]  # the six BirdSet sets not used during development
    t_probe = compat["teacher_probe_cmAP"]
    runs = [("XS0", "XS", "Released XS (seed 0)"), ("XSs1", "XSs1", "Seed 1"), ("XSs2", "XSs2", "Seed 2"),
            ("XScls", "XScls", "Class loss only"), ("XSdet", "XSdet", "Embedding loss only on the backbone")]
    out = []
    for k, emb, label in runs:
        r, e, m = s[k], compat[emb], compat[f"{k}_ridge"]
        out.append([label, f'{r["WABAD"]:.3f}', f'{sum(r[x] / teacher[x] for x in six) / len(six):.3f}',
                    f'{probe[f"{k}_feat"]["retrieval_k1_mAP"]:.3f}', f'{probe[f"{k}_feat"]["probe_cmAP"]:.3f}',
                    f'{e["field_cos_median"]:.3f}', f'{m["field_cos_median"]:.3f}',
                    f'{100 * e["xprobe_cmAP"] / t_probe:.0f}', f'{100 * m["xprobe_cmAP"] / t_probe:.0f}'])
    return md_table(["Run", "WABAD cmAP", "BirdSet ratio", "Retrieval k = 1, features", "Classifier, features",
                     "Cosine to teacher, embedding layer", "Cosine to teacher, linear map",
                     "Teacher-trained classifier (%), embedding layer", "Teacher-trained classifier (%), linear map"],
                    out)


# label of the speed-kit result file (speedmulti_v1) -> laptop CPU shown in the table
SPEED_MACHINES = {"i7-8565U": "i7-8565U", "M4-Pro": "M4 Pro", "Ryzen-AI-7-PRO-350": "Ryzen AI 7 PRO 350"}
SPEED_ROWS = {"perch": "Perch 2.0", "birdnet_v24": "BirdNET V2.4", "birdnet_v3": "BirdNET+ V3.0",
              "kinglet_b": "Kinglet B", "kinglet_s": "Kinglet S", "kinglet_xs": "Kinglet XS",
              "birdnet_v24_overlap": "BirdNET V2.4, overlapping segments", "birdnet_v3_5s": "BirdNET+ V3.0, 5-s windows",
              "kinglet_b_pack": "Kinglet B, compact output layer", "kinglet_s_pack": "Kinglet S, compact output layer",
              "kinglet_xs_pack": "Kinglet XS, compact output layer"}


def table_s9():
    """End-to-end speed (x real time) per machine and thread count (speedmulti_v1 speed kit)."""
    fmt = lambda x: f"{x:.1f}" if x < 100 else f"{x:,.0f}"  # noqa: E731
    cols, decode = {}, {}
    for machine, name in SPEED_MACHINES.items():
        res = load(f"speedkit_{machine}.json")
        for r in res["runs"]:
            cols.setdefault((name, r["threads"]), {})[r["model"]] = r["xrt"]
            decode[name, r["threads"]] = res["decode"]["xrt"]
    keys = sorted(cols, key=lambda k: (list(SPEED_MACHINES.values()).index(k[0]), k[1]))
    rows = [[label] + [fmt(cols[k][m]) for k in keys] for m, label in SPEED_ROWS.items()]
    rows.append(["Decoding (ffmpeg)"] + [fmt(decode[k]) for k in keys])
    return md_table(["Model"] + [f"{n} ({t})" for n, t in keys], rows)


def main():
    parts = {"S1": table_s1(), "S2": table_s2(), "S3": table_s3(), "S4": table_s4(), "S5": table_s5(),
             "S6": table_s6(), "S7": table_s7(), "S8": table_s8(), "S9": table_s9()}
    text = "\n\n".join(f"<!-- TABLE {k} -->\n{v}" for k, v in parts.items())
    (HERE / "si_tables.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
