# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Build the manifest of the Japanese training species: split each species' recordings into a
held-out validation set and a training pool (Appendix S1).

Species: eBird Japan report frequency >= MIN_FREQ (0.1 %). Per species, the recordings are shuffled
with seed 0 and grouped by site (coordinates rounded to 3 decimals; a recording without
coordinates is its own group). Groups are taken in shuffled order into the validation set until it
holds n_val = min(8, n // 5) recordings (a group that would push it above 2 n_val stays in the pool),
so that recordings of one session never fall on both sides. The first POOL_MAX recordings of the
rest form the training pool (LARGER_POOL_MAX for the eight species with a larger pool, Appendix S1).

Inputs: <audio_root>/metadata/unified_metadata_v3.parquet (collect/build_unified_metadata.py) and
data_prep/species/dataset_summary/species_dataset_summary.csv (species/summarize_dataset.py).
Output: <data_root>/manifest.parquet
  rec_id, species, source, file_path, split (val|pool), rank (order within the species, from 0)

Usage: python build_manifest.py [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
from common import config  # noqa: E402

UNIFIED = config.AUDIO_ROOT / "metadata/unified_metadata_v3.parquet"
SUMMARY = PAPER / "data_prep/species/dataset_summary/species_dataset_summary.csv"

# eight species with a larger pool, Appendix S1 (eBird codes)
LARGER_POOL_SPECIES = {"eurcoo", "mallar3", "graher1", "bcnher", "eurwig", "gnwtea", "gadwal", "falduc"}
MIN_FREQ = 0.001
POOL_MAX = 256
LARGER_POOL_MAX = 1024
VAL_MAX = 8


def split_by_site(rec: pd.DataFrame, pool_max) -> pd.DataFrame:
    """rec_id, species, latitude, longitude, ... -> manifest with split (val|pool) and rank.

    pool_max(species) is the pool cap of that species.
    """
    site = rec["latitude"].round(3).astype(str) + "," + rec["longitude"].round(3).astype(str)
    rec = rec.assign(group=site.where(rec["latitude"].notna() & rec["longitude"].notna(),
                                      rec["rec_id"]))
    parts = []
    for sp, g in rec.groupby("species"):
        g = g.sample(frac=1, random_state=0).reset_index(drop=True)
        n_val, val_groups = min(VAL_MAX, len(g) // 5), set()
        sizes, n_taken = g["group"].value_counts(), 0
        for grp in g["group"].unique():  # groups in shuffled order of first appearance
            if n_taken >= n_val:
                break
            if n_taken + sizes[grp] <= 2 * n_val:  # large groups stay whole in the pool
                val_groups.add(grp)
                n_taken += sizes[grp]
        val = g[g["group"].isin(val_groups)].assign(split="val")
        pool = g[~g["group"].isin(val_groups)].assign(split="pool").iloc[: pool_max(sp)]
        val["rank"], pool["rank"] = range(len(val)), range(len(pool))
        parts += [val, pool]
    return pd.concat(parts, ignore_index=True).drop(columns=["latitude", "longitude"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=config.DATA_ROOT)
    args = ap.parse_args()

    summary = pd.read_csv(SUMMARY)
    species = set(summary.loc[summary["frequency_annual_mean"] >= MIN_FREQ, "ebird_species_code"])
    assert LARGER_POOL_SPECIES <= species, LARGER_POOL_SPECIES - species

    rec = pd.read_parquet(UNIFIED, columns=["recording_id", "ebird_species_code", "source",
                                            "file_path", "latitude", "longitude"])
    rec = rec[rec["ebird_species_code"].isin(species) & rec["file_path"].fillna("").ne("")]
    rec = rec.rename(columns={"recording_id": "rec_id", "ebird_species_code": "species"})
    man = split_by_site(rec, lambda sp: LARGER_POOL_MAX if sp in LARGER_POOL_SPECIES else POOL_MAX)

    args.out.mkdir(parents=True, exist_ok=True)
    man.to_parquet(args.out / "manifest.parquet", index=False)
    per_sp = man[man["split"] == "pool"].groupby("species").size()
    print(f"species: {man['species'].nunique()} (with data) / {len(species)} selected")
    print(f"recordings: {len(man):,} (val {(man['split'] == 'val').sum():,})")
    print(f"pool per species: min {per_sp.min()}, median {per_sp.median():.0f}, "
          f"<16: {(per_sp < 16).sum()}, <128: {(per_sp < 128).sum()}")
    print(per_sp[sorted(set(per_sp.index) & species)].head(12).to_string())


if __name__ == "__main__":
    main()
