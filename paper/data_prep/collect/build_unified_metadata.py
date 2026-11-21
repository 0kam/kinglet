# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Merge the metadata of all sources into unified metadata of the downloaded files.

Input (under common.config.AUDIO_ROOT):
  - audio/xeno-canto/metadata/<--xc-metadata ...> (one or more files, concatenated)
  - audio/inat-sounds/annotations/inat_metadata.parquet
  - audio/inat-api/metadata/inat_api_metadata.parquet
  - audio/macaulay/metadata/ml_metadata.parquet

Output (a new file per version; an existing version is never overwritten):
  - metadata/unified_metadata_<version>.parquet + .json (input hashes, counts, rules)

Steps:
  1. Take the downloaded recordings of each source (re-applying the download filters)
  2. Normalise file_path
  3. Build the Macaulay file_path from the extracted files
  4. Keep the unified schema (UNIFIED_SCHEMA) columns + the derived columns below
  5. Coordinate validation -> coord_source
  6. country_code / is_japan: summarize_dataset.country_codes (country name for XC/ML, reverse
     geocoding of the coordinates for iNat); is_japan = country_code == "JP"
  7. taxon_match / exclude_reason: flag recordings of a different IOC taxon that does not contain
     the Japanese population, and exclude them from training for species whose Japanese-
     population taxon has at least MIN_JAPANESE_TAXON recordings (rows are kept)
  8. File existence check (optional)

Usage:
  python build_unified_metadata.py --version v4 --xc-metadata xc_metadata.parquet [--verify] [--dry-run]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "species"))
from summarize_dataset import NE_ASIA, S_SE_ASIA, country_codes  # noqa: E402
from utils import (  # noqa: E402
    PROJECT_ROOT,
    UNIFIED_SCHEMA,
    audio_path,
    get_target_species,
    load_config,
    load_metadata,
    save_metadata,
)

STEP_DIR = Path(__file__).resolve().parent
DERIVED_COLUMNS = ["coord_source", "country_code", "taxon_match", "exclude_reason"]

# Source-metadata codes that are not target codes. The Macaulay metadata was collected with a
# species list that mapped Riparia chinensis to plamar1 (African Plain Martin); those rows are
# relabelled to gytmar1 and then flagged as a foreign taxon.
RELABELED_CODES = {"plamar1": "gytmar1"}

# Labels that also cover a different IOC taxon which does not contain the Japanese
# population. Per eBird code:
#   foreign_xc: XC taxa (gen sp) of the foreign taxon -> XC subspecies that nevertheless
#               belong to the Japanese population (kept as "match")
#   native_cc:  countries where the Japanese-population taxon occurs (including zones
#               shared with the foreign taxon). Used only for Macaulay/iNat rows, whose
#               labels are eBird/Clements concepts without the IOC taxon: rows geocoded
#               to any other country are "foreign", rows without a country "unknown".
PACIFIC = {"AU", "NZ", "PG", "NC", "SB", "VU", "FJ", "GU", "MP", "PW", "FM", "TL"}
FOREIGN_TAXA = {
    "cintit13": {  # Japanese Tit = IOC Parus minor; P. cinereus s.s. is S/SE Asian
        "foreign_xc": {"Parus cinereus": set()},
        "native_cc": NE_ASIA | {"JP", "RU", "VN", "LA", "MM", "TH"},
    },
    "scathr8": {  # checklist species = Z. major (Amami) + Z. dauma iriomotensis (Iriomote)
        "foreign_xc": {"Zoothera dauma": {"iriomotensis"}},  # Iriomote population
        "native_cc": {"JP"},
    },
    "whgpig1": {  # checklist species = T. permagnus (Ryukyu); T. formosae = Taiwan/Philippines
        "foreign_xc": {"Treron formosae": set()},
        "native_cc": {"JP"},
    },
    "taibeg1": {  # checklist species = A. serrirostris + A. fabalis middendorffii
        "foreign_xc": {"Anser fabalis": {"middendorffii"}},  # Middendorff's Bean Goose
        "native_cc": NE_ASIA | {"JP", "RU"},
    },
    "categr2": {  # checklist species = Eastern Cattle Egret; B. ibis = Western (Africa/Europe/Americas)
        "foreign_xc": {"Bubulcus ibis": set()},
        "native_cc": NE_ASIA | S_SE_ASIA | PACIFIC | {"JP", "RU"},
    },
    "gytmar1": {  # checklist species = R. chinensis; R. paludicola = Africa
        "foreign_xc": {"Riparia paludicola": set()},
        "native_cc": NE_ASIA | S_SE_ASIA | {"JP", "AF"},
    },
}
# Foreign-taxon rows are excluded only when the Japanese-population taxon has at least
# this many recordings; otherwise they stay as the best available proxy. 10 is the
# lowest coverage tier of the recording-count summary below.
MIN_JAPANESE_TAXON = 10
# Japan bounding box (config.yaml japan_bounds); used only for swapped lat/lon detection.
JAPAN_LAT = (24.0, 46.0)
JAPAN_LON = (122.0, 154.0)


def load_xc(cfg: dict, metadata_files: list[str]) -> pd.DataFrame:
    """XC: recordings passing the download filter (quality <= C, CC licence) with a file_path (files concatenated)."""
    meta_dir = audio_path(cfg, cfg["xeno_canto"]["metadata_dir"])
    df = pd.concat([load_metadata(meta_dir / f) for f in metadata_files], ignore_index=True)
    df["xc_taxon"] = df["xc_gen"] + " " + df["xc_sp"]

    # XC (IOC) lumps Masked Bunting (eBird bkfbun2, Emberiza personata) into
    # E. spodocephala, so its recordings arrive under Black-faced Bunting bkfbun1. Relabel ssp
    # personata anywhere, and Japanese recordings without ssp (Hokkaido/Honshu in the
    # breeding season, where only personata breeds).
    ssp = df["xc_subspecies"].fillna("").str.strip().str.lower()
    masked = (df["xc_taxon"] == "Emberiza spodocephala") & (
        ssp.isin(["personata", "masked bunting"]) | ((df["country"] == "Japan") & (ssp == ""))
    )
    df.loc[masked, ["ebird_species_code", "scientific_name", "japanese_name"]] = [
        "bkfbun2", "Emberiza personata", "アオジ",  # Japanese name of Masked Bunting (data value, from the checklist)
    ]
    print(f"  XC: relabelled {masked.sum()} Emberiza spodocephala personata / Japan -> bkfbun2")

    # Apply download filter
    quality_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    df["_qnum"] = df["quality"].map(quality_order)
    cc_mask = df["license"].str.contains("creativecommons", na=False)
    dl_mask = cc_mask & (df["_qnum"] <= 3)
    df = df[dl_mask].drop(columns=["_qnum"])

    # Only include files with file_path set
    df = df[df["file_path"].notna() & (df["file_path"] != "")]

    # Deduplicate by recording_id (same XC recording appearing multiple times)
    n_before = len(df)
    df = df.drop_duplicates(subset=["recording_id"], keep="first")
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  XC: removed {n_dedup} duplicate recording_ids")

    print(f"  XC: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def load_inat_s3(cfg: dict) -> pd.DataFrame:
    """iNat Sounds 2024: all rows (all downloaded)."""
    df = load_metadata(
        audio_path(cfg, cfg["inat_sounds"]["annotations_dir"]) / "inat_metadata.parquet"
    )

    # Normalize relative paths to absolute
    audio_root = Path(cfg["audio_root"])
    mask = ~df["file_path"].str.startswith("/")
    df.loc[mask, "file_path"] = df.loc[mask, "file_path"].apply(
        lambda p: str(audio_root / p)
    )

    print(f"  iNat S3: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def load_inat_api(cfg: dict) -> pd.DataFrame:
    """iNat API: recordings that are not duplicates and have a file_path."""
    df = load_metadata(
        audio_path(cfg, cfg["inat_api"]["metadata_dir"]) / "inat_api_metadata.parquet"
    )

    # Exclude duplicates with iNat Sounds 2024
    if "inat_api_is_duplicate_sounds2024" in df.columns:
        df = df[df["inat_api_is_duplicate_sounds2024"] != True]  # noqa: E712

    # Only include files with file_path set
    df = df[df["file_path"].notna() & (df["file_path"] != "")]

    # Deduplicate by sound_id: same audio file can appear in multiple observations
    # for different species. Keep the first occurrence.
    n_before = len(df)
    df = df.drop_duplicates(subset=["recording_id"], keep="first")
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  iNat API: removed {n_dedup} duplicate recording_ids (same sound, different obs)")

    print(f"  iNat API: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def load_macaulay(cfg: dict) -> pd.DataFrame:
    """Macaulay: map the metadata to the extracted files."""
    meta_path = audio_path(cfg, cfg["macaulay"]["metadata_dir"]) / "ml_metadata.parquet"
    df = load_metadata(meta_path)

    # Scan actual files on disk
    audio_dir = audio_path(cfg, cfg["macaulay"]["audio_dir"])
    file_map = {}  # asset_id -> file_path

    if audio_dir.exists():
        for species_dir in audio_dir.iterdir():
            if not species_dir.is_dir():
                continue
            for f in species_dir.iterdir():
                if f.is_file() and f.name.startswith("ml_"):
                    # ml_{asset_id}.{ext} -> asset_id
                    stem = f.stem  # ml_{asset_id}
                    asset_id = stem[3:]  # remove "ml_"
                    file_map[asset_id] = str(f)

    print(f"  Macaulay files on disk: {len(file_map)}")

    # Map asset_id to file_path
    df["ml_asset_id_str"] = df["ml_asset_id"].astype(str)
    df["file_path"] = df["ml_asset_id_str"].map(file_map).fillna("")
    df = df[df["file_path"] != ""]
    df = df.drop(columns=["ml_asset_id_str"])

    # Deduplicate by recording_id (same asset appearing in multiple metadata rows)
    n_before = len(df)
    df = df.drop_duplicates(subset=["recording_id"], keep="first")
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  Macaulay: removed {n_dedup} duplicate recording_ids")

    print(f"  Macaulay: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def verify_files(df: pd.DataFrame, sample_size: int = 1000) -> tuple[int, int]:
    """Check file existence on a sample."""
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    exists = 0
    missing = 0
    missing_examples = []

    for _, row in sample.iterrows():
        p = Path(row["file_path"])
        if p.exists():
            exists += 1
        else:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(str(p))

    return exists, missing, missing_examples


def validate_coordinates(lat: pd.Series, lon: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (lat, lon, coord_source) with invalid pairs set to NA.

    Japan-range lat/lon exchanged
    -> "swapped"; out of range or exactly (0, 0) -> NA, "invalid"; either value NA ->
    "missing"; otherwise "source" (as in the source metadata).
    """
    lat = pd.to_numeric(lat, errors="coerce").astype("float64")
    lon = pd.to_numeric(lon, errors="coerce").astype("float64")
    swapped = lat.between(*JAPAN_LON) & lon.between(*JAPAN_LAT)
    lat, lon = lat.where(~swapped, lon), lon.where(~swapped, lat)
    missing = lat.isna() | lon.isna()
    invalid = ~missing & (~lat.between(-90, 90) | ~lon.between(-180, 180) | ((lat == 0) & (lon == 0)))
    lat, lon = lat.mask(invalid), lon.mask(invalid)
    status = pd.Series("source", index=lat.index)
    status[swapped] = "swapped"
    status[invalid] = "invalid"
    status[missing] = "missing"
    return lat, lon, status


def taxon_match(df: pd.DataFrame) -> pd.Series:
    """"foreign" / "match" / "unknown" for FOREIGN_TAXA codes, "not_assessed" otherwise.

    XC rows use the XC (IOC) taxon; other sources use country_code vs native_cc. A
    recording made in Japan is always "match" (e.g. a Yaeyama tit uploaded as P. cinereus).
    """
    out = pd.Series("not_assessed", index=df.index)
    for code, rule in FOREIGN_TAXA.items():
        sub = df[df["ebird_species_code"] == code]
        xc = sub["source"] == "xeno-canto"
        keep_ssp = sub["xc_taxon"].map(rule["foreign_xc"])  # NaN unless a foreign XC taxon
        foreign_xc = xc & pd.Series(
            [isinstance(keep, set) and ssp not in keep for keep, ssp in zip(keep_ssp, sub["xc_subspecies"])],
            index=sub.index,
        )
        cc = sub["country_code"]
        foreign_other = ~xc & (cc != "") & ~cc.isin(rule["native_cc"])
        out[sub.index] = "match"
        out[sub.index[(foreign_xc | foreign_other) & (cc != "JP")]] = "foreign"
        out[sub.index[~xc & (cc == "")]] = "unknown"
    return out


def exclude_reason(df: pd.DataFrame) -> pd.Series:
    """"foreign_taxon" for foreign rows of codes with >= MIN_JAPANESE_TAXON matching rows."""
    n_match = df[df["taxon_match"] == "match"].groupby("ebird_species_code").size()
    enough = df["ebird_species_code"].map(n_match).fillna(0) >= MIN_JAPANESE_TAXON
    return pd.Series("", index=df.index).mask((df["taxon_match"] == "foreign") & enough, "foreign_taxon")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build unified metadata")
    parser.add_argument("--version", required=True, help="output version, e.g. v4 (never overwritten)")
    parser.add_argument("--xc-metadata", nargs="+", default=["xc_metadata.parquet"],
                        help="XC metadata file(s) in xeno_canto.metadata_dir, concatenated")
    parser.add_argument("--verify", action="store_true", help="Verify file existence (sample)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving")
    args = parser.parse_args()

    cfg = load_config()
    base = audio_path(cfg, cfg["unified_metadata"])
    out_path = base.with_name(f"{base.stem}_{args.version}.parquet")
    if out_path.exists() and not args.dry_run:
        parser.error(f"{out_path} is a frozen release; choose a new --version")

    print("Loading source metadata...")
    dfs = [load_xc(cfg, args.xc_metadata)]
    for loader in [load_inat_s3, load_inat_api, load_macaulay]:
        dfs.append(loader(cfg))

    # Select unified schema columns (+ XC taxon, used for taxon_match and then dropped)
    columns = UNIFIED_SCHEMA + ["xc_taxon", "xc_subspecies"]
    result = pd.concat([df.reindex(columns=columns) for df in dfs], ignore_index=True)

    # Check for cross-source duplicate recording_ids (should be rare after per-source dedup)
    dup_mask = result["recording_id"].duplicated(keep="first")
    n_dups = dup_mask.sum()
    if n_dups > 0:
        print(f"\nCross-source duplicates: {n_dups} (keeping first)")
        result = result[~dup_mask].reset_index(drop=True)

    relabel = result["ebird_species_code"].isin(list(RELABELED_CODES))
    result["ebird_species_code"] = result["ebird_species_code"].replace(RELABELED_CODES)
    off_list = set(result["ebird_species_code"]) - set(get_target_species(cfg)["ebird_species_code"])
    assert not off_list, f"codes not in species_list: {sorted(off_list)}"

    print("Deriving coordinates / country / taxon match...")
    result["latitude"], result["longitude"], result["coord_source"] = validate_coordinates(
        result["latitude"], result["longitude"]
    )
    result["country_code"] = country_codes(result).to_numpy()
    result["is_japan"] = result["country_code"] == "JP"
    result["taxon_match"] = taxon_match(result)
    result["exclude_reason"] = exclude_reason(result)
    result = result[UNIFIED_SCHEMA + DERIVED_COLUMNS]

    flagged = result[result["taxon_match"] != "not_assessed"]
    taxon_table = pd.crosstab(
        [flagged["ebird_species_code"], flagged["source"]],
        flagged["taxon_match"] + flagged["exclude_reason"].map(lambda r: f" ({r})" if r else ""),
    )
    print(f"\nTaxon match (FOREIGN_TAXA codes; excluded if >= {MIN_JAPANESE_TAXON} match):")
    print(taxon_table.to_string())

    print(f"\n{'='*60}")
    print(f"Unified metadata: {len(result)} recordings")
    print(f"Species: {result['ebird_species_code'].nunique()}")
    print(f"Japan recordings (country_code == JP): {result['is_japan'].sum()}")
    print(f"Excluded from training (exclude_reason): {(result['exclude_reason'] != '').sum()}")

    # Source breakdown
    print(f"\nSource breakdown:")
    for src, grp in result.groupby("source"):
        print(f"  {src}: {len(grp)} recordings, {grp['ebird_species_code'].nunique()} species")

    # Species coverage distribution
    species_counts = result.groupby("ebird_species_code").size()
    print(f"\nRecordings per species:")
    print(f"  Mean: {species_counts.mean():.1f}")
    print(f"  Median: {species_counts.median():.0f}")
    print(f"  Min: {species_counts.min()}")
    print(f"  Max: {species_counts.max()}")

    # Tier distribution
    tier_counts = Counter()
    for n in species_counts:
        if n < 10:
            tier_counts["< 10"] += 1
        elif n < 50:
            tier_counts["10-49"] += 1
        elif n < 100:
            tier_counts["50-99"] += 1
        elif n < 500:
            tier_counts["100-499"] += 1
        else:
            tier_counts["500+"] += 1

    print(f"\nSpecies by recording count:")
    for bucket in ["< 10", "10-49", "50-99", "100-499", "500+"]:
        print(f"  {bucket:>8}: {tier_counts.get(bucket, 0)} species")

    if args.verify:
        print(f"\nVerifying file existence (sample)...")
        exists, missing, examples = verify_files(result)
        total = exists + missing
        print(f"  Sample: {exists}/{total} exist ({exists/total*100:.1f}%)")
        if examples:
            print(f"  Missing examples:")
            for ex in examples:
                print(f"    {ex}")

    if args.dry_run:
        print("\n(dry-run: not saving)")
        return

    save_metadata(result, out_path)
    inputs = [
        *(audio_path(cfg, cfg["xeno_canto"]["metadata_dir"]) / f for f in args.xc_metadata),
        audio_path(cfg, cfg["inat_sounds"]["annotations_dir"]) / "inat_metadata.parquet",
        audio_path(cfg, cfg["inat_api"]["metadata_dir"]) / "inat_api_metadata.parquet",
        audio_path(cfg, cfg["macaulay"]["metadata_dir"]) / "ml_metadata.parquet",
        PROJECT_ROOT / cfg["species_list"],
    ]
    git = ["git", "-C", str(STEP_DIR)]
    commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(git + ["status", "--porcelain", "--", "."], capture_output=True, text=True)
    sidecar = {
        "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # None outside a git checkout
        "code_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "code_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "output": {str(out_path): sha256_file(out_path)},
        "rules": {
            "coord_source": "source / swapped (Japan-range lat,lon exchanged) / invalid (out of range or (0,0) -> NA)"
                            " / missing",
            "country_code": "summarize_dataset.country_codes: XC/ML country name (ISO by majority vote of geocoded"
                            " coordinates), else reverse-geocoded coordinates (iNat)",
            "is_japan": "country_code == JP",
            "relabeled_codes": RELABELED_CODES,
            "xc_bkfbun2": "XC Emberiza spodocephala ssp personata (anywhere) or without ssp in Japan -> bkfbun2",
            "taxon_match": "build_unified_metadata.FOREIGN_TAXA: XC by XC taxon (+ssp), others by country_code;"
                           " not_assessed for other codes",
            "exclude_reason": f"foreign_taxon if taxon_match == foreign and the code has >= {MIN_JAPANESE_TAXON}"
                              " taxon_match == match rows",
        },
        "counts": {
            "rows": len(result),
            "species": int(result["ebird_species_code"].nunique()),
            "by_source": result["source"].value_counts().to_dict(),
            "is_japan": int(result["is_japan"].sum()),
            "relabeled": int(relabel.sum()),
            "coord_source": result["coord_source"].value_counts().to_dict(),
            "taxon_match": result["taxon_match"].value_counts().to_dict(),
            "exclude_reason": result.loc[result["exclude_reason"] != "", "ebird_species_code"]
            .value_counts().to_dict(),
        },
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path} (+ .json)")


if __name__ == "__main__":
    main()
