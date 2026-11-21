# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Analyse the iNat Sounds 2024 annotations and fetch the audio.

Usage:
  # Annotations only (download the JSON files and match species)
  python collect_inat.py --annotations-only

  # Download audio (tar.gz from S3; needs the aws cli)
  python collect_inat.py --download
"""

import argparse
import json
import sys
import tarfile
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    get_target_species,
    is_in_japan,
    audio_path,
    load_config,
    save_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="iNat Sounds 2024 data collection")
    parser.add_argument("--annotations-only", action="store_true",
                        help="analyse annotations only (no audio download)")
    parser.add_argument("--download", action="store_true",
                        help="download audio from S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list what would be downloaded")
    return parser.parse_args()


# ── Annotation download and parsing ──────────────────────────────

def download_annotations(cfg: dict) -> dict:
    """Download and extract the train/val annotation JSON files."""
    import urllib.request

    inat_cfg = cfg["inat_sounds"]
    ann_dir = audio_path(cfg, inat_cfg["annotations_dir"])
    ann_dir.mkdir(parents=True, exist_ok=True)

    annotations = {}

    for split, url in inat_cfg["annotation_urls"].items():
        json_path = ann_dir / f"{split}.json"
        tar_path = ann_dir / f"{split}.json.tar.gz"

        if json_path.exists():
            print(f"  {split}.json already exists, loading...")
        else:
            print(f"  Downloading {split} annotations...")
            urllib.request.urlretrieve(url, str(tar_path))

            print(f"  Extracting {split}.json.tar.gz...")
            with tarfile.open(str(tar_path), "r:gz") as tar:
                tar.extractall(path=str(ann_dir))
            tar_path.unlink(missing_ok=True)

        with open(json_path) as f:
            annotations[split] = json.load(f)

        print(f"  {split}: loaded {len(annotations[split].get('annotations', []))} annotations")

    return annotations


def build_inat_taxonomy_map(annotations: dict) -> pd.DataFrame:
    """Build the category (species) table from the iNat annotation JSON."""
    all_categories = {}

    for split, data in annotations.items():
        for cat in data.get("categories", []):
            cat_id = cat["id"]
            if cat_id not in all_categories:
                all_categories[cat_id] = {
                    "inat_category_id": cat_id,
                    "inat_scientific_name": cat.get("name", ""),
                    "inat_common_name": cat.get("common_name", ""),
                    "inat_supercategory": cat.get("supercategory", ""),
                    "inat_kingdom": cat.get("kingdom", ""),
                    "inat_phylum": cat.get("phylum", ""),
                    "inat_class": cat.get("class", ""),
                    "inat_order": cat.get("order", ""),
                    "inat_family": cat.get("family", ""),
                }

    return pd.DataFrame(all_categories.values())


def match_species(inat_cats: pd.DataFrame, target_species: pd.DataFrame) -> pd.DataFrame:
    """Match iNat categories to our species list."""
    # Normalise scientific names (genus + species epithet, lower case)
    def normalize(name: str) -> str:
        parts = str(name).strip().split()
        if len(parts) >= 2:
            return f"{parts[0].lower()} {parts[1].lower()}"
        return str(name).lower().strip()

    inat_cats = inat_cats.copy()
    inat_cats["_inat_key"] = inat_cats["inat_scientific_name"].apply(normalize)

    target = target_species.copy()
    # Try both scientific_name (checklist 8th ed.) and ebird_sciname
    target["_key_sci"] = target["scientific_name"].apply(normalize)
    target["_key_ebird"] = target["ebird_sciname"].apply(normalize)

    # First match on scientific_name
    merged = inat_cats.merge(
        target, left_on="_inat_key", right_on="_key_sci", how="inner",
        suffixes=("", "_target"),
    )

    # Retry unmatched iNat categories on ebird_sciname
    matched_inat_ids = set(merged["inat_category_id"])
    unmatched = inat_cats[~inat_cats["inat_category_id"].isin(matched_inat_ids)]

    if len(unmatched) > 0:
        retry = unmatched.merge(
            target, left_on="_inat_key", right_on="_key_ebird", how="inner",
            suffixes=("", "_target"),
        )
        if len(retry) > 0:
            merged = pd.concat([merged, retry], ignore_index=True)

    # Drop duplicates
    merged = merged.drop_duplicates(subset="inat_category_id", keep="first")

    return merged


def build_license_map(annotations: dict) -> dict:
    """Build an ID -> name map from the licenses list of the annotation JSON."""
    license_map = {}
    for _split, data in annotations.items():
        for lic in data.get("licenses", []):
            license_map[lic["id"]] = lic.get("url", lic.get("name", ""))
    return license_map


def build_audio_lookup(annotations: dict) -> dict:
    """Index the audio entries of the annotation JSON by ID."""
    lookup = {}
    license_map = build_license_map(annotations)

    for split, data in annotations.items():
        for audio in data.get("audio", []):
            audio_id = audio["id"]
            lic_id = audio.get("license")
            lookup[audio_id] = {
                "split": split,
                "file_name": audio.get("file_name", ""),
                "latitude": audio.get("latitude"),
                "longitude": audio.get("longitude"),
                "date": audio.get("date", ""),
                "duration_sec": audio.get("duration"),
                "license": license_map.get(lic_id, str(lic_id) if lic_id is not None else ""),
            }
    return lookup


def analyze_annotations(cfg: dict) -> pd.DataFrame:
    """Main annotation analysis."""
    print("=" * 60)
    print("iNat Sounds 2024 — Annotation Analysis")
    print("=" * 60)

    # 1. Download annotations
    print("\n[1/4] Downloading annotations...")
    annotations = download_annotations(cfg)

    # 2. Build the iNat category (species) table
    print("\n[2/4] Building iNat taxonomy table...")
    inat_cats = build_inat_taxonomy_map(annotations)

    # Birds only
    birds = inat_cats[inat_cats["inat_class"] == "Aves"]
    print(f"  Total iNat categories: {len(inat_cats)}")
    print(f"  Birds (Aves): {len(birds)}")

    # 3. Match species
    print("\n[3/4] Matching with target species list...")
    target_species = get_target_species(cfg)
    matched = match_species(birds, target_species)
    print(f"  Matched species: {len(matched)}/{len(target_species)} "
          f"({len(matched)/len(target_species)*100:.1f}%)")

    # 4. Build recording metadata
    print("\n[4/4] Building recording metadata...")
    audio_lookup = build_audio_lookup(annotations)

    # category_id -> species row
    cat_to_species = {}
    for _, row in matched.iterrows():
        cat_to_species[row["inat_category_id"]] = row

    # Build recording metadata from the annotations (recording-category pairs)
    records = []
    for split, data in annotations.items():
        for ann in data.get("annotations", []):
            cat_id = ann["category_id"]
            if cat_id not in cat_to_species:
                continue

            audio_id = ann["audio_id"]
            audio_info = audio_lookup.get(audio_id, {})
            species_info = cat_to_species[cat_id]

            lat = audio_info.get("latitude")
            lon = audio_info.get("longitude")

            records.append({
                "recording_id": f"inat:{audio_id}",
                "source": "inat",
                "ebird_species_code": species_info["ebird_species_code"],
                "scientific_name": species_info["scientific_name"],
                "japanese_name": species_info["japanese_name"],
                "latitude": lat,
                "longitude": lon,
                "country": "",  # no country in the iNat annotations
                "is_japan": is_in_japan(lat, lon, cfg) if lat and lon else False,
                "duration_sec": audio_info.get("duration_sec"),
                "sample_rate": None,  # no sample rate in the iNat JSON
                "quality": "",
                "license": audio_info.get("license", ""),
                "file_path": "",  # set after download
                "vocalization_type": "",
                # iNat-specific
                "inat_audio_id": audio_id,
                "inat_category_id": cat_id,
                "inat_split": split,
                "inat_file_name": audio_info.get("file_name", ""),
                "inat_date": audio_info.get("date", ""),
            })

    df = pd.DataFrame(records)

    # Save metadata
    inat_cfg = cfg["inat_sounds"]
    metadata_path = audio_path(cfg, inat_cfg["annotations_dir"]) / "inat_metadata.parquet"
    save_metadata(df, metadata_path)

    # ── Summary report ──
    print("\n" + "=" * 60)
    print("iNat Sounds 2024 — Analysis Report")
    print("=" * 60)

    total_recordings = len(df)
    species_found = df["ebird_species_code"].nunique() if total_recordings > 0 else 0
    print(f"Matched species: {species_found}/{len(target_species)} "
          f"({species_found/len(target_species)*100:.1f}%)")
    print(f"Total recordings for matched species: {total_recordings}")

    if total_recordings > 0:
        # Per split
        for split_name, split_df in df.groupby("inat_split"):
            print(f"  {split_name}: {len(split_df)} recordings, "
                  f"{split_df['ebird_species_code'].nunique()} species")

        # Japan recordings
        japan_count = df["is_japan"].sum()
        japan_species = df[df["is_japan"]]["ebird_species_code"].nunique()
        print(f"\nJapan recordings: {japan_count} ({japan_species} species)")

        # Recordings per species
        per_species = df.groupby("ebird_species_code").size()
        print(f"\nRecordings per species:")
        print(f"  Mean:   {per_species.mean():.1f}")
        print(f"  Median: {per_species.median():.0f}")
        print(f"  Min:    {per_species.min()}")
        print(f"  Max:    {per_species.max()}")

    # Unmatched species
    matched_codes = set(matched["ebird_species_code"]) if len(matched) > 0 else set()
    unmatched_species = target_species[~target_species["ebird_species_code"].isin(matched_codes)]
    print(f"\nUnmatched species: {len(unmatched_species)}/{len(target_species)}")
    if len(unmatched_species) <= 30:
        for _, row in unmatched_species.iterrows():
            print(f"  {row['scientific_name']} ({row['japanese_name']})")

    return df


# ── Audio download ────────────────────────────────────────

def download_audio(cfg: dict, args):
    """Download and extract the audio from S3 and keep the matched species only."""
    import subprocess

    inat_cfg = cfg["inat_sounds"]
    raw_dir = audio_path(cfg, inat_cfg["raw_dir"])
    filtered_dir = audio_path(cfg, inat_cfg["filtered_dir"])
    metadata_path = audio_path(cfg, inat_cfg["annotations_dir"]) / "inat_metadata.parquet"

    if not metadata_path.exists():
        print("Error: metadata not found. Run --annotations-only first.")
        sys.exit(1)

    df = pd.read_parquet(str(metadata_path))

    # ── Dry run ──
    if args.dry_run:
        print(f"\n[DRY RUN] iNat Sounds 2024 Download")
        print(f"  Total recordings: {len(df)}")
        print(f"  Species: {df['ebird_species_code'].nunique()}")
        print(f"  S3 files to download:")
        s3_bucket = inat_cfg["s3_bucket"]
        s3_prefix = inat_cfg["s3_prefix"]
        for split in ["train", "val"]:
            split_cnt = len(df[df["inat_split"] == split])
            print(f"    s3://{s3_bucket}/{s3_prefix}/{split}.tar.gz "
                  f"({split_cnt} recordings)")
        print(f"  Estimated download: ~81 GB (train) + ~11 GB (val)")
        print(f"  Output: {filtered_dir}")
        return

    # Find the aws cli (also next to the Python interpreter)
    aws_cmd = shutil.which("aws")
    if aws_cmd is None:
        # also look in the bin directory of the environment
        venv_aws = Path(sys.executable).parent / "aws"
        if venv_aws.exists():
            aws_cmd = str(venv_aws)
        else:
            print("Error: aws cli is required.")
            print("  Install: pip install awscli")
            print("  Or:      sudo apt install awscli")
            sys.exit(1)
    print(f"Using aws cli: {aws_cmd}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Download from S3 (aws cli, no credentials needed)
    s3_bucket = inat_cfg["s3_bucket"]
    s3_prefix = inat_cfg["s3_prefix"]

    for split in ["train", "val"]:
        tar_name = f"{split}.tar.gz"
        tar_path = raw_dir / tar_name
        s3_url = f"s3://{s3_bucket}/{s3_prefix}/{tar_name}"

        if not tar_path.exists():
            print(f"Downloading {s3_url}...")
            result = subprocess.run(
                [aws_cmd, "s3", "cp", "--no-sign-request", s3_url, str(tar_path)],
            )
            if result.returncode != 0:
                # remove the partial file so that a rerun retries
                tar_path.unlink(missing_ok=True)
                print(f"  Download failed for {tar_name}")
                continue
        else:
            print(f"  {tar_name} already exists ({tar_path.stat().st_size / 1e9:.1f} GB)")

        # Extract (a marker file prevents re-extraction)
        extract_done_marker = raw_dir / f".{split}_extracted"
        if not extract_done_marker.exists():
            print(f"Extracting {tar_name} (this may take a while)...")
            try:
                with tarfile.open(str(tar_path), "r:gz") as tar:
                    tar.extractall(path=str(raw_dir))
                # write the success marker
                extract_done_marker.write_text("done")
            except Exception as e:
                print(f"  Extraction failed for {tar_name}: {e}")
                print(f"  Delete {tar_path} and re-run to retry.")
                continue
        else:
            print(f"  {split} already extracted")

    # Copy the files of matched species to filtered/
    print("\nFiltering matched species files...")

    copied = 0
    skipped = 0
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        src_name = row.get("inat_file_name", "")
        split = row.get("inat_split", "")
        species_code = row["ebird_species_code"]
        recording_id = row["inat_audio_id"]

        if not src_name or not split:
            continue

        dst_dir = filtered_dir / species_code
        dst_path = dst_dir / f"inat_{recording_id}.wav"

        if dst_path.exists():
            skipped += 1
            continue

        src_path = raw_dir / src_name
        if not src_path.exists():
            errors += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(dst_path))
        copied += 1

    print(f"\nResults: copied={copied}, skipped(exists)={skipped}, errors={errors}")

    # Write file paths into the metadata
    df["file_path"] = df.apply(
        lambda r: str(
            Path(inat_cfg["filtered_dir"]) / r["ebird_species_code"]
            / f"inat_{r['inat_audio_id']}.wav"
        ),
        axis=1,
    )
    save_metadata(df, metadata_path)


# ── Main ────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config()

    if args.annotations_only:
        analyze_annotations(cfg)
    elif args.download:
        download_audio(cfg, args)
    else:
        print("Specify --annotations-only or --download")
        print("  --annotations-only: download & analyze annotation JSONs")
        print("  --download:         download audio from S3 (after annotations)")
        sys.exit(1)


if __name__ == "__main__":
    main()
