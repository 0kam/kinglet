# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Per-species summary and figures of the collected (downloaded) training audio.

Input:
  - <audio_root>/metadata/unified_metadata_v4.parquet (downloaded, deduplicated unified
    metadata from build_unified_metadata.py; rows with a non-empty exclude_reason, i.e. a
    different taxon without the Japanese population, are left out)
  - <audio_root>/audio/macaulay/metadata/ml_metadata.parquet (adds the ML star rating ml_rating)
  - data_prep/species/ebird_frequency.csv (eBird checklist reporting frequency in Japan, output
    of get_ebird_frequency.py)
  - data_prep/species/species_list.csv (eBird-matched entries)

Output (data_prep/species/dataset_summary/):
  - species_dataset_summary.csv        all per-species counts
  - fig1_ebird_coverage.{png,pdf}      comparison with the eBird reporting frequency
  - fig2_species_profile.{png,pdf}     breakdown of the top species (source/region/type/quality)
  - fig3_overview.{png,pdf}            breakdown of the whole dataset
  - fig4_all_species_heatmap.{png,pdf} recording counts of all species

Definitions:
  - Country: the country_code column (made by build_unified_metadata.py with country_codes
    below). XC/ML use the country name entered by the recordist, iNat uses offline reverse
    geocoding of the coordinates (reverse_geocoder: nearest GeoNames place with population
    >= 1000). Country name -> ISO code by majority vote of the same reverse geocoding.
    Russia east of 90 deg E counts as NE Asia.
  - Vocalization type: XC type / ML behaviors classified as song, call, song+call, other.
    iNat has no annotation.
  - Quality: XC A = high, B = mid, C = low (D/E are not downloaded). ML mean star rating
    >= 4 high, >= 3 mid, < 3 low, no ratings = unrated. iNat is unrated.

Usage:
  python summarize_dataset.py [--top-n 50]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import reverse_geocoder as rg
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collect"))
from utils import audio_path, get_target_species, load_config  # noqa: E402

STEP_DIR = Path(__file__).resolve().parent
OUT_DIR = STEP_DIR / "dataset_summary"

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae",
    "axes.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Categorical palette (validated: adjacent CVD dE >= 9.1 on light surface)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#d9d8d4"
TEXT2 = "#52514e"

SOURCES = {  # unified `source` value -> display label
    "xeno-canto": "Xeno-canto",
    "inat": "iNat Sounds 2024",
    "inat-api": "iNaturalist API",
    "macaulay": "Macaulay Library",
}
REGIONS = ["Japan", "NE Asia", "S & SE Asia", "Europe", "Americas", "Other"]
VTYPES = ["song", "call", "song+call", "other", "unannotated"]
QUALITY = ["high", "mid", "low", "unrated"]
QUALITY_COLORS = ["#1c5cab", "#3987e5", "#86b6ef", GRAY]  # ordinal blue ramp + gray
VTYPE_COLORS = PALETTE[:4] + [GRAY]

NE_ASIA = {"KR", "KP", "CN", "TW", "HK", "MO", "MN"}
S_SE_ASIA = {"TH", "VN", "MY", "SG", "ID", "PH", "KH", "LA", "MM", "BN", "TL",
             "IN", "NP", "BD", "LK", "BT", "PK", "MV"}
EUROPE = set("""AL AD AT BY BE BA BG HR CY CZ DK EE FO FI FR DE GI GR GG HU IS IE IM
IT JE XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO SM RS SK SI ES SJ SE CH UA GB VA AX
RU""".split())
AMERICAS = set("""US CA MX GT BZ SV HN NI CR PA CU JM HT DO PR BS BB TT AG DM GD KN LC
VC AW CW BQ SX MF BL KY TC VG VI AI MS GP MQ BM GL PM AR BO BR CL CO EC GY PY PE SR UY
VE GF FK""".split())

SONG_WORDS = ("song", "duet", "singing")
UNANNOTATED = {"", "uncertain", "?", "unknown"}
ML_BEHAVIOR_ONLY = {"flying", "foraging or eating", "courtship", "display", "or copulation",
                    "feeding young", "carrying food", "preening", "nest building", "flock"}


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------
def vocalization_class(text: str) -> str:
    """Classify XC type / ML behaviors (comma-separated) as song/call/song+call/other/unannotated."""
    tokens = [t.strip().lower() for t in (text or "").split(",")]
    tokens = [t for t in tokens if t not in UNANNOTATED and t not in ML_BEHAVIOR_ONLY]
    has_song = any(w in t for t in tokens for w in SONG_WORDS)
    has_call = any("call" in t for t in tokens)
    if has_song and has_call:
        return "song+call"
    if has_song:
        return "song"
    if has_call:
        return "call"
    return "other" if tokens else "unannotated"


def quality_class(df: pd.DataFrame) -> pd.Series:
    q = pd.Series("unrated", index=df.index)
    xc = df["source"] == "xeno-canto"
    q[xc & (df["quality"] == "A")] = "high"
    q[xc & (df["quality"] == "B")] = "mid"
    q[xc & (df["quality"] == "C")] = "low"
    rated = (df["source"] == "macaulay") & (df["ml_rating_count"] > 0)
    q[rated & (df["ml_rating"] >= 4)] = "high"
    q[rated & (df["ml_rating"] >= 3) & (df["ml_rating"] < 4)] = "mid"
    q[rated & (df["ml_rating"] < 3)] = "low"
    return q


def country_codes(df: pd.DataFrame) -> pd.Series:
    """ISO country code: the country name for XC/ML, else the coordinates."""
    has_xy = df["latitude"].notna() & df["longitude"].notna()
    xy = df.loc[has_xy, ["latitude", "longitude"]].astype(float)
    geo = pd.Series(
        [r["cc"] for r in rg.search(list(xy.itertuples(index=False, name=None)), mode=2)],
        index=xy.index,
    )
    name = df["country"].fillna("").str.strip()
    # Country name -> ISO code by majority vote of geocoded coordinates
    named = pd.DataFrame({"name": name[has_xy], "cc": geo})
    named = named[named["name"] != ""]
    name2cc = named.groupby("name")["cc"].agg(lambda s: s.value_counts().index[0])
    agree = (named["cc"] == named["name"].map(name2cc)).mean()
    print(f"  country name vs geocoded coordinates agree: {agree:.1%} ({len(named):,} rows)")

    cc = name.map(name2cc)
    cc = cc.fillna(geo)
    return cc.fillna("")


def region_of(cc: pd.Series, lon: pd.Series) -> pd.Series:
    r = pd.Series("Other", index=cc.index)
    r[cc.isin(EUROPE)] = "Europe"
    r[cc.isin(AMERICAS)] = "Americas"
    r[cc.isin(S_SE_ASIA)] = "S & SE Asia"
    r[cc.isin(NE_ASIA) | ((cc == "RU") & (lon >= 90))] = "NE Asia"
    r[cc == "JP"] = "Japan"
    return r


def load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = pd.read_parquet(audio_path(cfg, "metadata/unified_metadata_v4.parquet"))
    excluded = rec["exclude_reason"] != ""
    print(f"Excluded from training (exclude_reason): {excluded.sum():,} rows")
    rec = rec[~excluded].reset_index(drop=True)
    ml = pd.read_parquet(
        audio_path(cfg, cfg["macaulay"]["metadata_dir"]) / "ml_metadata.parquet",
        columns=["recording_id", "ml_rating", "ml_rating_count"],
    ).drop_duplicates("recording_id")
    ml["ml_rating_count"] = pd.to_numeric(ml["ml_rating_count"], errors="coerce").fillna(0)
    rec = rec.merge(ml, on="recording_id", how="left")

    print("Deriving country / region / vocalization type / quality...")
    rec["cc"] = rec["country_code"]
    rec["region"] = region_of(rec["cc"], rec["longitude"].astype(float))
    types = rec["vocalization_type"].fillna("")
    rec["vtype"] = types.map({t: vocalization_class(t) for t in types.unique()})
    rec["qclass"] = quality_class(rec)

    # Checklist species listed twice (native part A + introduced part B) share one eBird code
    species = get_target_species(cfg)[
        ["ebird_species_code", "japanese_name", "scientific_name"]
    ].drop_duplicates("ebird_species_code")
    freq = pd.read_csv(STEP_DIR / "ebird_frequency.csv")[
        ["ebird_species_code", "frequency_annual_mean", "residence_status"]
    ].drop_duplicates("ebird_species_code")
    species = species.merge(freq, on="ebird_species_code", how="left")
    return rec, species


def summarize(rec: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    def counts(col: str, levels: list[str], prefix: str) -> pd.DataFrame:
        t = pd.crosstab(rec["ebird_species_code"], rec[col]).reindex(columns=levels, fill_value=0)
        return t.add_prefix(prefix)

    by_source = pd.crosstab(rec["ebird_species_code"], rec["source"]).reindex(
        columns=list(SOURCES), fill_value=0)
    by_source.columns = ["n_" + c.replace("-", "_") for c in by_source.columns]
    table = pd.concat([
        rec.groupby("ebird_species_code").size().rename("n_total"),
        by_source,
        counts("region", REGIONS, "n_region_"),
        counts("vtype", VTYPES, "n_type_"),
        counts("qclass", QUALITY, "n_quality_"),
    ], axis=1)
    table.columns = [c.lower().replace(" & ", "_").replace(" ", "_").replace("+", "_")
                     for c in table.columns]
    rated = table[["n_quality_high", "n_quality_mid", "n_quality_low"]].sum(axis=1)
    table["clean_rate"] = (table["n_quality_high"] / rated).where(rated > 0)

    out = species.merge(table, left_on="ebird_species_code", right_index=True, how="left")
    num = out.columns.drop(["ebird_species_code", "japanese_name", "scientific_name",
                            "frequency_annual_mean", "residence_status", "clean_rate"])
    out[num] = out[num].fillna(0).astype(int)
    out["ebird_rank"] = out["frequency_annual_mean"].rank(ascending=False, method="first")
    return out.sort_values("ebird_rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}")


def stacked_share(ax, labels, table: pd.DataFrame, colors, title: str) -> None:
    """100% stacked horizontal bars (per row), with thin white gaps between segments."""
    share = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_numpy()
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for j, col in enumerate(table.columns):
        ax.barh(y, share[:, j], left=left, height=0.78, color=colors[j], label=col,
                edgecolor="white", linewidth=0.6)
        left += share[:, j]
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.set_title(title, loc="left", fontsize=9)
    legend_above_title(ax, ncol=2)


def legend_above_title(ax, ncol: int) -> None:
    """Legend above the title (borderaxespad, in font-size units, leaves room for the title)."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=ncol, frameon=False,
              fontsize=6.5, handlelength=1, borderaxespad=3.3, columnspacing=0.8)


def fig1_ebird_coverage(summary: pd.DataFrame) -> dict:
    d = summary[summary["frequency_annual_mean"] > 0].copy()
    d["pct"] = d["frequency_annual_mean"] * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.28})

    # (a) scatter: eBird frequency vs recordings
    y = d["n_total"].clip(lower=0.6)
    gap = (d["pct"] >= 1) & (d["n_total"] < 30)  # common in Japan but scarce recordings
    ax1.scatter(d["pct"][~gap], y[~gap], s=14, color=PALETTE[0], alpha=0.75,
                edgecolors="white", linewidths=0.4, zorder=3)
    ax1.scatter(d["pct"][gap], y[gap], s=22, color=PALETTE[1], edgecolors="white",
                linewidths=0.6, zorder=4, label="≥1% of checklists and <30 recordings")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axhline(100, color=TEXT2, lw=0.6, zorder=1)
    ax1.text(d["pct"].min(), 115, "100 recordings", color=TEXT2, fontsize=7)
    # Top-5 species; label sides chosen so near-equal pairs (bulbul/crow, sparrow/crow) split
    for right, (_, r) in zip([True, True, False, False, True],
                             d.nsmallest(5, "ebird_rank").iterrows()):
        ax1.annotate(r["japanese_name"], (r["pct"], r["n_total"]), fontsize=6.5,
                     xytext=(4 if right else -4, 0), textcoords="offset points",
                     ha="left" if right else "right", va="center")
    gap_names = ", ".join(f"{r.japanese_name} {r.n_total}" for r in
                          d[gap].sort_values("n_total").itertuples())
    ax1.legend(loc="upper right", frameon=False, fontsize=7, handletextpad=0.2)
    ax1.text(0, -0.2, f"Orange: {gap_names}", transform=ax1.transAxes, fontsize=7,
             color=TEXT2, va="top")
    ax1.set_yticks([0.6, 1, 10, 100, 1000, 10000], ["0", "1", "10", "100", "1k", "10k"])
    ax1.set_xlabel("eBird reporting frequency in Japan (% of checklists)")
    ax1.set_ylabel("Downloaded recordings (all sources)")
    ax1.set_title(f"(a) Recordings vs. eBird frequency ({len(d)} species reported in Japan)",
                  loc="left", fontsize=9)
    ax1.grid(True, which="major", color="#ecebe8", lw=0.5, zorder=0)

    # (b) detection-weighted coverage curve
    thresholds = np.unique(np.round(np.logspace(0, 4, 200)).astype(int))
    w = d["frequency_annual_mean"] / d["frequency_annual_mean"].sum()
    series = {
        "All sources, weighted by eBird frequency": (d["n_total"], w),
        "Recorded in Japan, weighted by eBird frequency": (d["n_region_japan"], w),
        "All sources, unweighted (species count)": (d["n_total"], pd.Series(1 / len(d), d.index)),
    }
    stats = {}
    for (lab, (n, weight)), c in zip(series.items(), PALETTE):
        cov = [weight[n >= t].sum() * 100 for t in thresholds]
        ax2.plot(thresholds, cov, color=c, lw=2, label=lab)
        stats[lab] = {t: weight[n >= t].sum() * 100 for t in (10, 50, 100, 500)}
    for lab, c in zip(series, PALETTE):
        v = stats[lab][100]
        ax2.plot(100, v, "o", ms=5, color=c, mec="white", mew=1.5, zorder=4)
        ax2.annotate(f"{v:.0f}%", (100, v), xytext=(5, 3), textcoords="offset points",
                     fontsize=7, color="#0b0b0b")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Minimum recordings per species (N)")
    ax2.set_ylabel("Coverage (%) by species with ≥ N recordings")
    ax2.set_title("(b) Share of eBird detections covered", loc="left", fontsize=9)
    ax2.legend(frameon=False, fontsize=7, loc="lower left")
    ax2.grid(True, which="major", color="#ecebe8", lw=0.5)
    save(fig, "fig1_ebird_coverage")
    return stats


def fig2_species_profile(summary: pd.DataFrame, top_n: int) -> None:
    d = summary.nsmallest(top_n, "ebird_rank").iloc[::-1]
    labels = [f"{r.japanese_name} ({r.frequency_annual_mean * 100:.1f}%)" for r in d.itertuples()]
    y = np.arange(len(d))
    fig, axes = plt.subplots(
        1, 5, figsize=(13, 0.16 * top_n + 1.6), sharey=True,
        gridspec_kw={"width_ratios": [1.4, 1, 1, 1, 1], "wspace": 0.12},
    )

    ax = axes[0]
    ax.barh(y, d["n_total"], height=0.72, color=PALETTE[0], label="All recordings")
    ax.scatter(d["n_region_japan"].clip(lower=0.8), y, marker="D", s=10, color=PALETTE[1],
               zorder=3, label="Recorded in Japan")
    ax.set_xscale("log")
    ax.set_xlim(0.8, d["n_total"].max() * 1.5)
    ax.axvline(100, color=TEXT2, lw=0.6)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_title("Recordings (log)", loc="left", fontsize=9)
    legend_above_title(ax, ncol=2)

    src = d[["n_" + s.replace("-", "_") for s in SOURCES]]
    src.columns = list(SOURCES.values())
    stacked_share(axes[1], labels, src, PALETTE, "Source")
    reg = d[[f"n_region_{r.lower().replace(' & ', '_').replace(' ', '_')}" for r in REGIONS]]
    reg.columns = REGIONS
    stacked_share(axes[2], labels, reg, PALETTE, "Recording region")
    vt = d[[f"n_type_{v.replace('+', '_')}" for v in VTYPES]]
    vt.columns = VTYPES
    stacked_share(axes[3], labels, vt, VTYPE_COLORS, "Vocalization type")
    ql = d[[f"n_quality_{q}" for q in QUALITY]]
    ql.columns = ["high (XC A / ML ≥4★)", "mid (XC B / ML 3★)", "low (XC C / ML <3★)", "unrated"]
    stacked_share(axes[4], labels, ql, QUALITY_COLORS, "Recording quality")
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Top {top_n} species by eBird reporting frequency in Japan "
                 "(label: Japanese name, % of checklists)", x=0.02, ha="left", y=1.0,
                 fontsize=10)
    save(fig, "fig2_species_profile")


def fig3_overview(rec: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4),
                             gridspec_kw={"hspace": 0.75, "wspace": 0.45})
    src_labels = [f"{v}\n{rec[rec.source == k].shape[0]:,} rec. / "
                  f"{rec[rec.source == k].ebird_species_code.nunique()} spp."
                  for k, v in SOURCES.items()]

    # (a) absolute recordings by source x region
    ax = axes[0, 0]
    t = pd.crosstab(rec["source"], rec["region"]).reindex(index=list(SOURCES), columns=REGIONS)
    left = np.zeros(len(t))
    yy = np.arange(len(t))[::-1]
    for j, r in enumerate(REGIONS):
        ax.barh(yy, t[r] / 1000, left=left, height=0.7, color=PALETTE[j], label=r,
                edgecolor="white", linewidth=0.8)
        left += t[r].to_numpy() / 1000
    ax.set_yticks(yy, src_labels, fontsize=7)
    ax.set_xlabel("Recordings (thousands)")
    ax.set_title("(a) Recordings by source and recording region", loc="left", fontsize=9)
    legend_above_title(ax, ncol=3)

    # (b) top countries, Japan emphasized
    ax = axes[0, 1]
    names = rec.groupby("cc")["country"].agg(
        lambda s: s[s.str.len() > 0].value_counts().index[0] if (s.str.len() > 0).any() else "")
    by_cc = rec["cc"].replace("", np.nan).value_counts()
    top = by_cc.head(15)
    if "JP" not in top.index:
        top = pd.concat([top, by_cc[["JP"]]])
    top = top[::-1]
    colors = [PALETTE[1] if cc == "JP" else "#9ec5f4" for cc in top.index]
    ax.barh(np.arange(len(top)), top / 1000, height=0.7, color=colors)
    rank = by_cc.index.get_loc("JP") + 1
    ticks = [f"{names.get(c) or c}" + (f" (#{rank})" if c == "JP" else "") for c in top.index]
    ax.set_yticks(np.arange(len(top)), ticks, fontsize=7)
    ax.annotate(f"{by_cc['JP']:,}", (by_cc["JP"] / 1000, 0), xytext=(3, 0),
                textcoords="offset points", va="center", fontsize=7)
    ax.set_xlabel("Recordings (thousands)")
    ax.set_title("(b) Top 15 recording countries + Japan", loc="left", fontsize=9)

    # (c)(d) vocalization type / quality by source
    for ax, col, levels, colors, title in (
        (axes[1, 0], "vtype", VTYPES, VTYPE_COLORS, "(c) Vocalization type annotation"),
        (axes[1, 1], "qclass", QUALITY, QUALITY_COLORS,
         "(d) Quality (high = XC A / ML ≥4★)"),
    ):
        t = pd.crosstab(rec["source"], rec[col]).reindex(index=list(SOURCES)[::-1],
                                                          columns=levels, fill_value=0)
        stacked_share(ax, list(SOURCES.values())[::-1], t, colors, title)
        ax.set_yticks(np.arange(len(t)), list(SOURCES.values())[::-1], fontsize=7)
    save(fig, "fig3_overview")


def fig4_heatmap(summary: pd.DataFrame) -> None:
    d = summary.copy()
    rows = {
        "All sources": "n_total",
        "Xeno-canto": "n_xeno_canto",
        "iNat Sounds 2024": "n_inat",
        "iNaturalist API": "n_inat_api",
        "Macaulay Library": "n_macaulay",
        "Recorded in Japan": "n_region_japan",
        "Song (incl. song+call)": None,
        "Call (incl. song+call)": None,
        "High quality": "n_quality_high",
    }
    mat = np.vstack([
        d[c] if c else (d["n_type_song"] if "Song" in k else d["n_type_call"]) + d["n_type_song_call"]
        for k, c in rows.items()
    ])
    bounds = [0, 1, 10, 50, 100, 500, 1e9]
    cmap = ListedColormap([GRAY, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    norm = BoundaryNorm(bounds, cmap.N)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.08})
    x = np.arange(len(d))
    ax0.bar(x, d["frequency_annual_mean"].fillna(0) * 100, width=1.0, color=PALETTE[0])
    ax0.set_yscale("log")
    ax0.set_ylim(0.005, 100)
    ax0.set_ylabel("eBird freq.\n(% checklists)", fontsize=7)
    ax1.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax1.set_yticks(np.arange(len(rows)), list(rows), fontsize=7)
    ax1.set_xlabel(f"{len(d)} species (eBird-matched checklist species), "
                   "sorted by eBird reporting frequency in Japan")

    f = d["frequency_annual_mean"].fillna(0) * 100
    for lo, hi, name in ((1, 1e9, "≥1%"), (0.1, 1, "0.1–1%"), (1e-9, 0.1, "<0.1%"),
                         (-1, 1e-9, "not reported")):
        idx = np.where((f >= lo) & (f < hi))[0]
        if len(idx) == 0:
            continue
        for ax in (ax0, ax1):
            ax.axvline(idx[0] - 0.5, color="#0b0b0b", lw=0.6)
        ax0.text(idx.mean(), 1.04, f"{name}: {len(idx)} spp.", ha="center", va="bottom",
                 fontsize=7, transform=ax0.get_xaxis_transform())
    ax1.set_xticks([])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax0, ax1],
                      fraction=0.02, pad=0.01, ticks=[0.5, 5.5, 30, 75, 300, 5e8])
    cb.ax.set_yticklabels(["0", "1–9", "10–49", "50–99", "100–499", "≥500"], fontsize=7)
    cb.set_label("Recordings", fontsize=7)
    save(fig, "fig4_all_species_heatmap")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-n", type=int, default=50, help="species in fig2")
    args = parser.parse_args()

    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec, species = load(cfg)
    summary = summarize(rec, species)
    summary.to_csv(OUT_DIR / "species_dataset_summary.csv", index=False, encoding="utf-8-sig")

    print("Plotting...")
    stats = fig1_ebird_coverage(summary)
    fig2_species_profile(summary, args.top_n)
    fig3_overview(rec, summary)
    fig4_heatmap(summary)

    # Headline numbers
    s = summary
    print(f"\nRecordings: {len(rec):,} / species with >=1: {(s.n_total > 0).sum()} / {len(s)}")
    print(f"Recorded in Japan: {(rec.region == 'Japan').sum():,} rec., "
          f"{(s.n_region_japan > 0).sum()} spp.")
    for lab, v in stats.items():
        print(f"  {lab}: " + ", ".join(f">={t}: {x:.1f}%" for t, x in v.items()))
    for lo, name in ((0.01, ">=1%"), (0.001, ">=0.1%")):
        c = s[s.frequency_annual_mean >= lo]
        print(f"eBird {name}: {len(c)} spp., >=100 rec: {(c.n_total >= 100).sum()}, "
              f">=100 JP rec: {(c.n_region_japan >= 100).sum()}, "
              f"min: {c.nsmallest(1, 'n_total')[['japanese_name', 'n_total']].values.tolist()}")
    print(rec["region"].value_counts().to_string())
    print(rec["vtype"].value_counts().to_string())
    print(pd.crosstab(rec["source"], rec["qclass"]).to_string())


if __name__ == "__main__":
    main()
