# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Collect Macaulay Library audio metadata and download the audio.

Uses the Macaulay Library Search API of the Cornell Lab (no authentication) to collect bird
sounds submitted to eBird.

Usage:
  # Metadata only (all species)
  python collect_macaulay.py --metadata-only

  # Test: 5 species only
  python collect_macaulay.py --metadata-only --limit 5

  # Download audio (after collecting metadata)
  python collect_macaulay.py --download

  # Test: download 3 species only
  python collect_macaulay.py --download --limit 3

  # Dry run (list what would be downloaded)
  python collect_macaulay.py --download --dry-run

  # Maximum number of recordings per species (default: unlimited)
  python collect_macaulay.py --metadata-only --max-per-species 500
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

from utils import (
    audio_path,
    get_target_species,
    is_in_japan,
    load_config,
    load_metadata,
)

SEARCH_API = "https://search.macaulaylibrary.org/api/v1/search"
PER_PAGE = 100  # API maximum
PAGE_MAX_RETRIES = 3  # maximum retries per page


def parse_args():
    parser = argparse.ArgumentParser(description="Macaulay Library data collection")
    parser.add_argument("--metadata-only", action="store_true",
                        help="collect metadata only (no audio download)")
    parser.add_argument("--download", action="store_true",
                        help="download audio (metadata must exist)")
    parser.add_argument("--limit", type=int, default=0,
                        help="for testing: maximum number of species")
    parser.add_argument("--max-per-species", type=int, default=0,
                        help="maximum recordings per species (0 = unlimited)")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list what would be downloaded")
    parser.add_argument("--min-rating", type=float, default=0.0,
                        help="minimum rating for download (0-5, default: 0)")
    return parser.parse_args()


# ── Metadata collection ──────────────────────────────────────────

def fetch_species_metadata(
    taxon_code: str,
    rate_limit: float = 1.0,
    max_recordings: int = 0,
) -> tuple[list[dict], bool]:
    """Fetch all metadata pages for one species.

    Returns:
        (items, success): the records fetched, and whether all pages were fetched.
        After an error success is False and the species is retried on the next run.
    """
    import urllib.request
    import urllib.parse

    all_items = []
    cursor = None
    page = 0

    while True:
        params = {
            "taxonCode": taxon_code,
            "mediaType": "audio",
            "count": PER_PAGE,
        }
        if cursor:
            params["initialCursorMark"] = cursor

        url = f"{SEARCH_API}?{urllib.parse.urlencode(params)}"

        # Per-page retries
        data = None
        for attempt in range(PAGE_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "bird-bioacoustics-research/1.0"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                break
            except Exception as e:
                print(f"  Error fetching {taxon_code} page {page} "
                      f"(attempt {attempt + 1}/{PAGE_MAX_RETRIES}): {e}")
                if attempt < PAGE_MAX_RETRIES - 1:
                    time.sleep(rate_limit * (2 ** attempt) + 1)

        if data is None:
            # All retries failed -> return as incomplete
            return all_items, False

        results = data.get("results", {})
        content = results.get("content", [])

        if not content:
            break

        all_items.extend(content)
        page += 1

        # Per-species cap
        if max_recordings > 0 and len(all_items) >= max_recordings:
            all_items = all_items[:max_recordings]
            break

        next_cursor = results.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor
        time.sleep(rate_limit)

    return all_items, True


def parse_ml_item(item: dict, species_row: pd.Series, cfg: dict) -> dict:
    """Convert one item of the ML API response to the unified format."""
    asset_id = str(item.get("assetId", ""))
    lat = _safe_float(item.get("latitude"))
    lon = _safe_float(item.get("longitude"))
    rating = _safe_float(item.get("rating")) or 0.0

    return {
        "recording_id": f"ml:{asset_id}",
        "source": "macaulay",
        "ebird_species_code": species_row["ebird_species_code"],
        "scientific_name": species_row["scientific_name"],
        "japanese_name": species_row["japanese_name"],
        "latitude": lat,
        "longitude": lon,
        "country": _extract_country(item.get("locationLine2", "")),
        "is_japan": is_in_japan(lat, lon, cfg),
        "duration_sec": None,  # not in the API response
        "sample_rate": None,
        "quality": "",
        "license": item.get("licenseType", ""),
        "file_path": "",
        "vocalization_type": item.get("behaviors") or "",
        # ML-specific fields
        "ml_asset_id": asset_id,
        "ml_media_url": item.get("mediaUrl", ""),
        "ml_rating": rating,
        "ml_rating_count": item.get("ratingCount", "0"),
        "ml_location": item.get("location", ""),
        "ml_obs_date": item.get("obsDttm", ""),
        "ml_user": item.get("userDisplayName", ""),
        "ml_checklist_id": item.get("eBirdChecklistId", ""),
        "ml_source": item.get("source", ""),
    }


def _safe_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extract_country(location_line2: str) -> str:
    """Country name from locationLine2 ('Prefecture, Country')."""
    if not location_line2:
        return ""
    parts = location_line2.split(",")
    return parts[-1].strip() if parts else ""


def _save_metadata_atomic(df: pd.DataFrame, metadata_path: Path) -> None:
    """Save a Parquet file atomically (tmp + rename)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metadata_path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(tmp), compression="snappy")
    tmp.rename(metadata_path)


def collect_metadata(cfg: dict, args) -> pd.DataFrame:
    """Collect ML metadata for all target species."""
    ml_cfg = cfg["macaulay"]
    rate_limit = ml_cfg["rate_limit_sec"]
    max_per_species = args.max_per_species

    species_df = get_target_species(cfg)
    if args.limit > 0:
        species_df = species_df.head(args.limit)

    print(f"Collecting Macaulay Library metadata for {len(species_df)} species...")
    if max_per_species > 0:
        print(f"  Max per species: {max_per_species}")

    # Progress file (for resuming)
    metadata_dir = audio_path(cfg, ml_cfg["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    progress_path = metadata_dir / "collection_progress.json"
    metadata_path = metadata_dir / "ml_metadata.parquet"

    progress = {"completed_species": [], "species_counts": {}}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    completed_species = set(progress.get("completed_species", []))
    species_counts = progress.get("species_counts", {})

    # Load existing metadata
    existing_records = []
    if metadata_path.exists():
        existing_df = load_metadata(metadata_path)
        existing_records = existing_df.to_dict("records")
        print(f"  Resuming: {len(completed_species)} species done, "
              f"{len(existing_records)} recordings collected")

    all_records = existing_records
    no_hit_species = []
    error_species = []

    # Signal handling
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\nForce exit")
            sys.exit(1)
        shutdown_requested = True
        print("\n\nShutdown requested — saving progress...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    remaining = [
        (idx, row) for idx, row in species_df.iterrows()
        if row["ebird_species_code"] not in completed_species
    ]

    skipped = len(species_df) - len(remaining)
    if skipped > 0:
        print(f"  Skipping {skipped} already-completed species")

    for idx, row in tqdm(remaining, desc="Species", unit="sp"):
        if shutdown_requested:
            break

        sp_code = row["ebird_species_code"]
        items, success = fetch_species_metadata(
            sp_code, rate_limit, max_per_species,
        )

        if not items and success:
            # fetched without error but no recordings
            no_hit_species.append({
                "species_code": sp_code,
                "scientific_name": row["scientific_name"],
                "japanese_name": row["japanese_name"],
            })
        elif not success:
            # interrupted by an API error -> keep the partial data but do not mark complete
            error_species.append(sp_code)
            if items:
                print(f"  {sp_code}: API error after {len(items)} records "
                      f"(will retry next run)")

        if items:
            for item in items:
                parsed = parse_ml_item(item, row, cfg)
                all_records.append(parsed)

        species_counts[sp_code] = len(items)

        # Mark only successful species complete (errors are retried next run)
        if success:
            completed_species.add(sp_code)

        # Save progress (Parquet and JSON, each atomically)
        df = pd.DataFrame(all_records)
        if len(df) > 0:
            _save_metadata_atomic(df, metadata_path)
        _save_progress(progress_path, {
            "completed_species": sorted(completed_species),
            "species_counts": species_counts,
        })

        time.sleep(rate_limit)

    # Final save
    df = pd.DataFrame(all_records)
    if len(df) > 0:
        _save_metadata_atomic(df, metadata_path)
        print(f"\nSaved {len(df)} rows → {metadata_path}")

    # ── Summary report ──
    print(f"\n{'='*60}")
    print("Macaulay Library Metadata Collection Report")
    print(f"{'='*60}")
    total_species_found = df["ebird_species_code"].nunique() if len(df) > 0 else 0
    total_recordings = len(df)
    print(f"Total species with recordings: {total_species_found}/{len(species_df)}")
    print(f"Total recordings: {total_recordings}")

    if len(df) > 0:
        japan_count = df["is_japan"].sum()
        print(f"Japan recordings: {japan_count} ({japan_count/total_recordings*100:.1f}%)")

        per_species = df.groupby("ebird_species_code").size()
        print(f"\nRecordings per species:")
        print(f"  Mean:   {per_species.mean():.1f}")
        print(f"  Median: {per_species.median():.0f}")
        print(f"  Min:    {per_species.min()}")
        print(f"  Max:    {per_species.max()}")

        # Rating distribution
        if "ml_rating" in df.columns:
            ratings = pd.to_numeric(df["ml_rating"], errors="coerce")
            print(f"\nRating distribution:")
            print(f"  Mean: {ratings.mean():.2f}")
            for threshold in [1, 2, 3, 4]:
                n = (ratings >= threshold).sum()
                print(f"  ≥{threshold}: {n} ({n/total_recordings*100:.1f}%)")

        # Vocalization types
        voc = df["vocalization_type"].value_counts().head(10)
        if len(voc) > 0:
            print(f"\nVocalization types (top 10):")
            for v, cnt in voc.items():
                print(f"  {v or '(unknown)'}: {cnt}")

    if no_hit_species:
        print(f"\nSpecies with NO recordings ({len(no_hit_species)}):")
        for sp in no_hit_species[:20]:
            print(f"  {sp['scientific_name']} ({sp['japanese_name']})")
        if len(no_hit_species) > 20:
            print(f"  ... and {len(no_hit_species) - 20} more")

    if error_species:
        print(f"\nSpecies with API errors ({len(error_species)}, will retry):")
        for sp in error_species[:10]:
            print(f"  {sp}")
        if len(error_species) > 10:
            print(f"  ... and {len(error_species) - 10} more")

    if shutdown_requested:
        print(f"\n  Interrupted — {len(completed_species)}/{len(species_df)} species done.")
        print(f"  Run again to continue.")
        sys.exit(1)

    # Exit non-zero if species with errors remain (for retrying)
    if error_species:
        print(f"\n  {len(error_species)} species had API errors — run again to retry.")
        sys.exit(2)

    return df


# ── Audio download ────────────────────────────────────────

def _ml_audio_path(cfg: dict, species_code: str, asset_id: str) -> Path:
    """Storage path of one ML audio file."""
    audio_dir = audio_path(cfg, cfg["macaulay"]["audio_dir"])
    return audio_dir / species_code / f"ml_{asset_id}.mp3"


async def download_one(
    session: aiohttp.ClientSession,
    row: dict,
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    max_retries: int = 3,
) -> dict:
    """Download one recording."""
    async with semaphore:
        asset_id = row["ml_asset_id"]
        media_url = row["ml_media_url"]
        species_code = row["ebird_species_code"]

        if not media_url:
            return {"asset_id": asset_id, "status": "no_url"}

        out_path = _ml_audio_path(cfg, species_code, asset_id)
        if out_path.exists():
            return {"asset_id": asset_id, "status": "exists", "path": str(out_path)}

        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_status = "error"
        for attempt in range(max_retries):
            try:
                async with session.get(
                    media_url,
                    headers={"User-Agent": "bird-bioacoustics-research/1.0"},
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        break
                    last_status = f"http_{resp.status}"
                    if resp.status == 429:
                        # Rate limited — longer backoff
                        wait = rate_limit * (4 ** attempt) + 5
                        await asyncio.sleep(wait)
                        continue
                    if resp.status < 500:
                        return {"asset_id": asset_id, "status": last_status}
            except Exception as e:
                last_status = f"error: {e}"

            if attempt < max_retries - 1:
                await asyncio.sleep(rate_limit * (2 ** attempt) + 1)
        else:
            return {"asset_id": asset_id, "status": last_status}

        try:
            out_path.write_bytes(content)
            await asyncio.sleep(rate_limit)
            return {"asset_id": asset_id, "status": "ok", "path": str(out_path)}
        except Exception as e:
            out_path.unlink(missing_ok=True)
            return {"asset_id": asset_id, "status": f"error: {e}"}


async def download_species_batch(
    session: aiohttp.ClientSession,
    species_records: list[dict],
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
) -> list[dict]:
    """Download the recordings of one species as a batch."""
    tasks = [
        download_one(session, rec, cfg, semaphore, rate_limit)
        for rec in species_records
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    return results


def _save_progress(progress_path: Path, data: dict):
    """Save the progress file atomically."""
    tmp = progress_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(progress_path)


async def download_all(cfg: dict, args):
    """Download all recordings listed in the metadata."""
    ml_cfg = cfg["macaulay"]
    metadata_dir = audio_path(cfg, ml_cfg["metadata_dir"])
    metadata_path = metadata_dir / "ml_metadata.parquet"

    if not metadata_path.exists():
        print("Error: metadata not found. Run --metadata-only first.")
        sys.exit(1)

    df = load_metadata(metadata_path)
    print(f"Loaded {len(df)} recordings from metadata")

    # Rating filter
    if args.min_rating > 0:
        before = len(df)
        df["_rating"] = pd.to_numeric(df["ml_rating"], errors="coerce").fillna(0)
        df = df[df["_rating"] >= args.min_rating].drop(columns=["_rating"])
        print(f"Rating filter (≥{args.min_rating}): {before} → {len(df)} recordings")

    # Group by species
    species_groups = df.groupby("ebird_species_code")
    species_list = sorted(species_groups.groups.keys())
    if args.limit > 0:
        species_list = species_list[:args.limit]

    total_recordings = sum(len(species_groups.get_group(sp)) for sp in species_list)

    # ── Dry run ──
    if args.dry_run:
        print(f"\n[DRY RUN] Would download {total_recordings} recordings "
              f"for {len(species_list)} species")
        est_gb = total_recordings * 2.5 / 1024  # ~2.5 MB/file MP3
        print(f"  Estimated storage: ~{est_gb:.0f} GB")
        top10 = df[df["ebird_species_code"].isin(species_list)] \
            .groupby("ebird_species_code").size().nlargest(10)
        print(f"\nTop 10 species by recording count:")
        for sp, cnt in top10.items():
            print(f"  {sp}: {cnt}")
        return

    dl_cfg = ml_cfg["download"]
    max_concurrent = dl_cfg["max_concurrent"]
    rate_limit = dl_cfg["rate_limit_sec"]
    semaphore = asyncio.Semaphore(max_concurrent)

    # Progress file
    progress_path = metadata_dir / "download_progress.json"
    progress = {"files": {}, "completed_species": []}
    if progress_path.exists():
        with open(progress_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "files" in raw:
            progress = raw
        else:
            progress["files"] = raw

    all_ok_results = progress["files"]
    completed_species = set(progress["completed_species"])

    remaining_species = [sp for sp in species_list if sp not in completed_species]
    skipped_species = len(species_list) - len(remaining_species)
    remaining_recordings = sum(
        len(species_groups.get_group(sp)) for sp in remaining_species
    )

    print(f"\nDownloading {total_recordings} recordings "
          f"for {len(species_list)} species")
    print(f"  Concurrent: {max_concurrent}, Rate limit: {rate_limit}s")
    if skipped_species > 0:
        print(f"  Resuming: {skipped_species} species already completed, "
              f"{len(all_ok_results)} files tracked")
        print(f"  Remaining: {remaining_recordings} recordings "
              f"for {len(remaining_species)} species")
    print()

    if not remaining_species:
        print("All species already completed!")
        return

    # Signal handling
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\nForce exit (progress already saved)")
            sys.exit(1)
        shutdown_requested = True
        print("\n\nShutdown requested — finishing current species, saving progress...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    total_stats = {"ok": 0, "exists": 0, "error": 0, "http_error": 0}
    connector = aiohttp.TCPConnector(limit=max_concurrent, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(remaining_species, desc="Species", unit="sp")
        for sp_idx, sp_code in enumerate(pbar):
            if shutdown_requested:
                print(f"\nStopping after {sp_idx} species (signal received)")
                break

            sp_df = species_groups.get_group(sp_code)
            records = sp_df.to_dict("records")
            pbar.set_postfix_str(f"{sp_code} ({len(records)} recs)")

            try:
                results = await download_species_batch(
                    session, records, cfg, semaphore, rate_limit,
                )
            except Exception as e:
                print(f"\n  Error processing {sp_code}: {e}")
                total_stats["error"] += len(records)
                _save_progress(progress_path, {
                    "files": all_ok_results,
                    "completed_species": sorted(completed_species),
                })
                continue

            sp_ok = 0
            sp_skipped = 0
            for r in results:
                status = r["status"]
                if status == "ok":
                    total_stats["ok"] += 1
                    sp_ok += 1
                elif status == "exists":
                    total_stats["exists"] += 1
                    sp_ok += 1
                elif status == "no_url":
                    # a missing URL cannot be retried -> counts as done
                    sp_skipped += 1
                elif status.startswith("http_"):
                    total_stats["http_error"] += 1
                else:
                    total_stats["error"] += 1

                if status in ("ok", "exists") and "path" in r:
                    all_ok_results[r["asset_id"]] = r["path"]

            # The species is complete when ok + skipped covers every record
            if sp_ok + sp_skipped == len(records):
                completed_species.add(sp_code)
            if sp_skipped > 0:
                total_stats["no_url"] = total_stats.get("no_url", 0) + sp_skipped

            _save_progress(progress_path, {
                "files": all_ok_results,
                "completed_species": sorted(completed_species),
            })

    # Summary
    print(f"\n{'='*60}")
    print("Macaulay Library Download Results")
    print(f"{'='*60}")
    for status, cnt in sorted(total_stats.items()):
        print(f"  {status}: {cnt}")
    print(f"  Total tracked files: {len(all_ok_results)}")
    print(f"  Completed species: {len(completed_species)}/{len(species_list)}")

    # Write file paths into the metadata (all rows, before filtering)
    if all_ok_results:
        full_df = load_metadata(metadata_path)
        full_df["file_path"] = full_df["ml_asset_id"].apply(
            lambda x: all_ok_results.get(str(x), "")
        )
        _save_metadata_atomic(full_df, metadata_path)

    if shutdown_requested:
        print(f"\n  Download interrupted — run again to continue.")
        sys.exit(1)
    if len(completed_species) < len(species_list):
        incomplete = len(species_list) - len(completed_species)
        print(f"\n  {incomplete} species incomplete — run again to retry.")
        sys.exit(2)


# ── Main ────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config()

    if args.metadata_only:
        collect_metadata(cfg, args)
    elif args.download:
        asyncio.run(download_all(cfg, args))
    else:
        print("Specify --metadata-only or --download")
        print("  --metadata-only: collect metadata from ML Search API")
        print("  --download:      download audio files (after metadata)")
        sys.exit(1)


if __name__ == "__main__":
    main()
