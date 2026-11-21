# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Collect Xeno-canto metadata and download the audio.

Requires a Xeno-canto API v3 key, read from the environment variable XC_API_KEY or from the
file named by XC_API_KEY_FILE (no key is shipped).

Usage:
  # Metadata only (all species)
  python collect_xc.py --metadata-only

  # Test: 5 species only
  python collect_xc.py --metadata-only --limit 5

  # Download audio (after collecting metadata)
  python collect_xc.py --download

  # Keep the original MP3 files (no WAV conversion, saves storage)
  python collect_xc.py --download --format mp3

  # Test: download 3 species only
  python collect_xc.py --download --limit 3

  # Dry run (list what would be downloaded)
  python collect_xc.py --download --dry-run

  # Write a re-collection to a separate metadata version and download its recordings
  # (the progress file is per version as well)
  python collect_xc.py --metadata-only --metadata-file xc_metadata_v2.parquet
  python collect_xc.py --download --metadata-file xc_metadata_v2.parquet \
      --previous-metadata xc_metadata.parquet  # uploads after the previous collection: Japan and new query names only

Query names: the checklist (8th ed.) name, the eBird name and XC_EXTRA_NAMES (IOC names) are all
queried, and recordings are deduplicated by XC ID.
"""

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

from utils import (
    audio_file_path,
    audio_path,
    get_target_species,
    is_in_japan,
    load_api_key,
    load_config,
    load_metadata,
    save_metadata,
)

# Extra XC (IOC) names queried per eBird code, on top of the checklist 8th-ed and
# eBird v2024 names. XC follows IOC, so a Japanese population that IOC ranks as its
# own species (or places in another genus) is invisible to both of our names.
# The list holds the IOC names with recordings in Japan, Korea or the Russian Far East
# that neither of our names queries. Foreign IOC names (foreign IOC taxa that Clements lumps into a
# Japanese species, e.g. Anas carolinensis) are deliberately NOT added.
# Riparia chinensis needs no entry: it is the eBird name of gytmar1 in species_list.csv.
XC_EXTRA_NAMES = {
    "cintit13": ["Parus minor"],  # Japanese Tit; XC P. cinereus has no Japanese birds
    "scathr8": ["Zoothera major"],  # Amami Thrush (ssp. major of the 8th ed)
    "vartit1": ["Sittiparus owstoni"],  # Owston's Tit (ssp. owstoni of the 8th ed)
    "whgpig1": ["Treron permagnus"],  # Ryukyu Green Pigeon; XC T. formosae = Taiwan/Philippines
    "blfowl1": ["Bubo blakistoni"],  # Blakiston's Fish Owl: XC genus Bubo (we query Ketupa)
    "saugul2": ["Chroicocephalus saundersi"],  # Saunders's Gull: XC genus Chroicocephalus (we query Saundersilarus)
    "origre": ["Chloris kittlitzi"],  # Bonin Greenfinch, an issf of origre in eBird v2024
    "taibeg1": ["Anser serrirostris"],  # Tundra Bean Goose (ssp. serrirostris of the 8th ed)
    "demcra1": ["Grus virgo"],  # Demoiselle Crane: XC genus Grus (we query Anthropoides)
    "categr2": ["Bubulcus coromandus"],  # Eastern Cattle Egret; XC B. ibis = Western only
}


def xc_query_names(rows: pd.DataFrame) -> list[str]:
    """Names to query for one eBird code: checklist + eBird + extra XC names, deduplicated."""
    code = rows["ebird_species_code"].iloc[0]
    names = [*rows["scientific_name"], *rows["ebird_sciname"], *XC_EXTRA_NAMES.get(code, [])]
    return list(dict.fromkeys(names))


def parse_args():
    parser = argparse.ArgumentParser(description="Xeno-canto data collection")
    parser.add_argument("--metadata-only", action="store_true",
                        help="collect metadata only (no audio download)")
    parser.add_argument("--download", action="store_true",
                        help="download audio (metadata must exist)")
    parser.add_argument("--limit", type=int, default=0,
                        help="for testing: maximum number of species")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing metadata")
    parser.add_argument("--format", choices=["wav", "mp3"], default="mp3",
                        help="output format (mp3 = keep original file, wav = convert)")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list what would be downloaded")
    parser.add_argument("--metadata-file", default="xc_metadata.parquet",
                        help="metadata file name inside metadata_dir (use a new name for a re-collection)")
    parser.add_argument("--previous-metadata",
                        help="download: metadata name of the previous collection. Recordings uploaded after it "
                             "(XC ID > its maximum ID) are kept only if from Japan or under an XC name "
                             "not queried before")
    return parser.parse_args()


def metadata_paths(cfg: dict, args) -> tuple[Path, Path]:
    """(metadata, download progress) paths; a non-default metadata name gets its own progress file."""
    meta_dir = audio_path(cfg, cfg["xeno_canto"]["metadata_dir"])
    metadata_path = meta_dir / args.metadata_file
    if args.metadata_file == "xc_metadata.parquet":
        return metadata_path, meta_dir / "download_progress.json"
    return metadata_path, meta_dir / f"{metadata_path.stem}_download_progress.json"


# ── Metadata collection ──────────────────────────────────────────

def fetch_species_metadata(
    scientific_name: str,
    api_key: str,
    api_url: str,
    per_page: int = 500,
    rate_limit: float = 1.0,
    max_retries: int = 3,
) -> list[dict]:
    """Fetch all metadata pages for one name; failed pages are retried with exponential backoff."""
    import urllib.request
    import urllib.parse

    all_recordings = []
    page = 1

    while True:
        params = urllib.parse.urlencode({
            "query": f'sp:"{scientific_name}"',
            "key": api_key,
            "per_page": per_page,
            "page": page,
        })
        url = f"{api_url}?{params}"

        data = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:
                print(f"  Error fetching {scientific_name} page {page} (attempt {attempt + 1}): {e}")
                time.sleep(rate_limit * (2**attempt) + 1)
        if data is None:
            break

        recordings = data.get("recordings", [])
        if not recordings:
            break

        all_recordings.extend(recordings)

        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break

        page += 1
        time.sleep(rate_limit)

    return all_recordings


def parse_xc_recording(rec: dict, species_row: pd.Series, cfg: dict) -> dict:
    """Convert one recording of the XC API response to the unified format."""
    xc_id = str(rec.get("id", ""))
    lat = _safe_float(rec.get("lat"))
    lon = _safe_float(rec.get("lon"))

    # Parse the sample rate ("44100", "48000 (Hz)", ...)
    sr_raw = rec.get("smp", "")
    sample_rate = _parse_sample_rate(sr_raw)

    # Duration (s)
    length_raw = rec.get("length", "0:00")
    duration_sec = _parse_duration(length_raw)

    return {
        "recording_id": f"xc:{xc_id}",
        "source": "xeno-canto",
        "ebird_species_code": species_row["ebird_species_code"],
        "scientific_name": species_row["scientific_name"],
        "japanese_name": species_row["japanese_name"],
        "latitude": lat,
        "longitude": lon,
        "country": rec.get("cnt", ""),
        "is_japan": is_in_japan(lat, lon, cfg),
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "quality": rec.get("q", ""),
        "license": rec.get("lic", ""),
        "file_path": "",  # set at download
        "vocalization_type": rec.get("type", ""),
        # XC-specific fields
        "xc_id": xc_id,
        "xc_url": rec.get("url", ""),
        "xc_file_url": rec.get("file", ""),
        "xc_file_name": rec.get("file-name", ""),
        "xc_sono_url": rec.get("sono", {}).get("small", "") if isinstance(rec.get("sono"), dict) else "",
        "xc_gen": rec.get("gen", ""),
        "xc_sp": rec.get("sp", ""),
        "xc_subspecies": rec.get("ssp", ""),
        "xc_recordist": rec.get("rec", ""),
        "xc_date": rec.get("date", ""),
        "xc_time": rec.get("time", ""),
        "xc_loc": rec.get("loc", ""),
        "xc_rmk": rec.get("rmk", ""),
        "xc_also": rec.get("also", []),
    }


def _safe_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_sample_rate(sr_raw: str) -> int | None:
    """Parse a sample-rate string."""
    if not sr_raw:
        return None
    # "44100", "48000 (Hz)", ...
    import re
    m = re.search(r"(\d+)", str(sr_raw))
    return int(m.group(1)) if m else None


def _parse_duration(length_str: str) -> float:
    """Convert an XC duration string "m:ss" or "h:mm:ss" to seconds."""
    if not length_str:
        return 0.0
    parts = length_str.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return 0.0


def collect_metadata(cfg: dict, args) -> pd.DataFrame:
    """Collect XC metadata for all target species."""
    api_key = load_api_key("XC")
    xc_cfg = cfg["xeno_canto"]
    api_url = xc_cfg["api_url"]
    per_page = xc_cfg["per_page"]
    rate_limit = xc_cfg["rate_limit_sec"]

    species_df = get_target_species(cfg)
    if args.limit > 0:
        species_df = species_df.head(args.limit)

    print(f"Collecting XC metadata for {len(species_df)} species...")

    # Load existing metadata (for --resume)
    metadata_path, _ = metadata_paths(cfg, args)
    existing_species = set()
    existing_records = []
    if args.resume and metadata_path.exists():
        existing_df = load_metadata(metadata_path)
        existing_species = set(existing_df["ebird_species_code"].unique())
        existing_records = existing_df.to_dict("records")
        print(f"  Resuming: {len(existing_species)} species already collected")

    all_records = existing_records
    no_hit_species = []

    # Species listed twice (native and introduced) share one eBird code and are queried once
    groups = species_df.groupby("ebird_species_code", sort=False)
    for sp_code, rows in tqdm(groups, total=groups.ngroups, desc="Species"):
        if sp_code in existing_species:
            continue
        row = rows.iloc[0]

        # Query the checklist name, the eBird name and the extra XC names (all of them, regardless of hits)
        recordings = {}
        for name in xc_query_names(rows):
            for rec in fetch_species_metadata(name, api_key, api_url, per_page, rate_limit):
                recordings.setdefault(rec.get("id"), rec)
            time.sleep(rate_limit)

        if not recordings:
            no_hit_species.append({
                "species_code": sp_code,
                "scientific_name": row["scientific_name"],
                "japanese_name": row["japanese_name"],
            })
            continue

        for rec in recordings.values():
            all_records.append(parse_xc_recording(rec, row, cfg))

    # To DataFrame
    df = pd.DataFrame(all_records)

    # xc_also is a list -> JSON string (for Parquet)
    if "xc_also" in df.columns:
        df["xc_also"] = df["xc_also"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else str(x)
        )

    # Save
    save_metadata(df, metadata_path)

    # ── Summary report ──
    print("\n" + "=" * 60)
    print("XC Metadata Collection Report")
    print("=" * 60)
    total_species_found = df["ebird_species_code"].nunique() if len(df) > 0 else 0
    total_recordings = len(df)
    print(f"Total species with recordings: {total_species_found}/{len(species_df)}")
    print(f"Total recordings: {total_recordings}")

    if len(df) > 0:
        # Quality distribution
        print(f"\nQuality distribution:")
        for q, cnt in df["quality"].value_counts().sort_index().items():
            print(f"  {q}: {cnt} ({cnt/total_recordings*100:.1f}%)")

        # Japan / overseas
        japan_count = df["is_japan"].sum()
        print(f"\nJapan recordings: {japan_count} ({japan_count/total_recordings*100:.1f}%)")
        print(f"Overseas recordings: {total_recordings - japan_count}")

        # Recordings per species
        per_species = df.groupby("ebird_species_code").size()
        print(f"\nRecordings per species:")
        print(f"  Mean:   {per_species.mean():.1f}")
        print(f"  Median: {per_species.median():.0f}")
        print(f"  Min:    {per_species.min()}")
        print(f"  Max:    {per_species.max()}")

        # 10 species with the fewest recordings
        bottom10 = per_species.nsmallest(10)
        print(f"\nSpecies with fewest recordings:")
        for sp_code, cnt in bottom10.items():
            sp_row = species_df[species_df["ebird_species_code"] == sp_code]
            if len(sp_row) > 0:
                jp = sp_row.iloc[0]["japanese_name"]
                print(f"  {sp_code}: {cnt} recordings ({jp})")

    if no_hit_species:
        print(f"\nSpecies with NO recordings ({len(no_hit_species)}):")
        for sp in no_hit_species:
            print(f"  {sp['scientific_name']} ({sp['japanese_name']})")

    return df


# ── Audio download ────────────────────────────────────────

def filter_for_download(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filter the recordings to download (quality, licence)."""
    xc_cfg = cfg["xeno_canto"]
    dl_filter = xc_cfg["download_filter"]

    quality_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "": 6}
    min_q = dl_filter["min_quality"]
    max_q_val = quality_order.get(min_q, 3)

    # Quality filter
    df = df.copy()
    df["_q_val"] = df["quality"].map(quality_order).fillna(6)
    filtered = df[df["_q_val"] <= max_q_val].copy()
    filtered = filtered.drop(columns=["_q_val"])

    # Licence filter
    allowed = dl_filter["allowed_licenses"]
    if allowed:
        license_mask = filtered["license"].apply(
            lambda lic: any(pat in str(lic) for pat in allowed)
        )
        filtered = filtered[license_mask]

    before = len(df)
    after = len(filtered)
    print(f"Download filter: {before} → {after} recordings "
          f"(quality ≤ {min_q}, CC license)")

    return filtered


async def download_one(
    session: aiohttp.ClientSession,
    row: dict,
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    out_ext: str = "mp3",
    max_retries: int = 3,
) -> dict | None:
    """Download one recording, retrying on HTTP errors."""
    async with semaphore:
        xc_id = row["xc_id"]
        file_url = row["xc_file_url"]
        species_code = row["ebird_species_code"]

        # XC withholds the file of species threatened by the cage-bird trade
        # ("file" is null -> NaN), e.g. Oriolus chinensis, Leiothrix lutea
        if not isinstance(file_url, str) or not file_url:
            return {"xc_id": xc_id, "status": "no_file_url"}

        # Output path
        out_path = audio_file_path(
            cfg, "xeno-canto", species_code, f"xc_{xc_id}", ext=out_ext,
        )
        if out_path.exists():
            return {"xc_id": xc_id, "status": "exists", "path": str(out_path)}

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if file_url.startswith("//"):
            file_url = "https:" + file_url

        last_status = "error"
        for attempt in range(max_retries):
            try:
                async with session.get(file_url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        break
                    last_status = f"http_{resp.status}"
                    # retry 5xx, fail at once on 4xx
                    if resp.status < 500:
                        return {"xc_id": xc_id, "status": last_status}
            except Exception as e:
                last_status = f"error: {e}"

            # Wait before retrying (exponential backoff)
            if attempt < max_retries - 1:
                await asyncio.sleep(rate_limit * (2 ** attempt) + 1)
        else:
            # All retries failed
            return {"xc_id": xc_id, "status": last_status}

        try:
            if out_ext == "mp3":
                out_path.write_bytes(content)
            else:
                # MP3 -> WAV conversion
                mp3_path = out_path.with_suffix(".mp3")
                mp3_path.write_bytes(content)
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3_path),
                     "-acodec", "pcm_s16le", str(out_path)],
                    capture_output=True, timeout=60,
                )
                mp3_path.unlink(missing_ok=True)
                if result.returncode != 0:
                    out_path.unlink(missing_ok=True)
                    return {"xc_id": xc_id, "status": "ffmpeg_error"}

            await asyncio.sleep(rate_limit)
            return {"xc_id": xc_id, "status": "ok", "path": str(out_path)}

        except Exception as e:
            out_path.unlink(missing_ok=True)
            out_path.with_suffix(".mp3").unlink(missing_ok=True)
            return {"xc_id": xc_id, "status": f"error: {e}"}


async def download_species_batch(
    session: aiohttp.ClientSession,
    species_records: list[dict],
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    out_ext: str,
) -> list[dict]:
    """Download the recordings of one species as a batch."""
    tasks = [
        download_one(session, rec, cfg, semaphore, rate_limit, out_ext)
        for rec in species_records
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
    return results


def _save_progress(progress_path: Path, data: dict):
    """Save the progress file atomically."""
    tmp = progress_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(progress_path)


async def download_all(cfg: dict, args):
    """Download all recordings listed in the metadata, one species at a time."""
    xc_cfg = cfg["xeno_canto"]
    metadata_path, progress_path = metadata_paths(cfg, args)

    if not metadata_path.exists():
        print("Error: metadata not found. Run --metadata-only first.")
        sys.exit(1)

    df = load_metadata(metadata_path)
    filtered = filter_for_download(df, cfg)
    if args.previous_metadata:
        prev = load_metadata(metadata_path.with_name(args.previous_metadata))
        prev_names = set(prev["xc_gen"] + " " + prev["xc_sp"])
        late = filtered["xc_id"].astype(int) > prev["xc_id"].astype(int).max()
        new_name = ~(filtered["xc_gen"] + " " + filtered["xc_sp"]).isin(prev_names)
        filtered = filtered[~late | (filtered["country"] == "Japan") | new_name]
        print(f"  Uploaded after {args.previous_metadata}: Japan or new XC names only → {len(filtered)} recordings")

    out_ext = args.format

    # Group by species
    species_groups = filtered.groupby("ebird_species_code")
    species_list = sorted(species_groups.groups.keys())
    if args.limit > 0:
        species_list = species_list[:args.limit]

    total_recordings = sum(len(species_groups.get_group(sp)) for sp in species_list)

    # ── Dry run ──
    if args.dry_run:
        print(f"\n[DRY RUN] Would download {total_recordings} recordings "
              f"for {len(species_list)} species")
        print(f"  Format: {out_ext}")
        if out_ext == "mp3":
            est_gb = total_recordings * 2.5 / 1024
        else:
            est_gb = total_recordings * 12 / 1024
        print(f"  Estimated storage: ~{est_gb:.0f} GB")
        print(f"\nTop 10 species by recording count:")
        top10 = filtered.groupby("ebird_species_code").size().nlargest(10)
        for sp, cnt in top10.items():
            print(f"  {sp}: {cnt}")
        return

    # Dependency check
    if out_ext == "wav":
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode != 0:
            print("Error: ffmpeg is required for WAV conversion. "
                  "Install it or use --format mp3")
            sys.exit(1)

    dl_cfg = xc_cfg["download"]
    max_concurrent = dl_cfg["max_concurrent"]
    rate_limit = dl_cfg["rate_limit_sec"]
    semaphore = asyncio.Semaphore(max_concurrent)

    # Progress file (downloaded files + completed species)
    progress = {"files": {}, "completed_species": []}
    if progress_path.exists():
        with open(progress_path) as f:
            raw = json.load(f)
        # Also accept a plain {xc_id: path} dict
        if isinstance(raw, dict) and "files" in raw:
            progress = raw
        else:
            progress["files"] = raw

    all_ok_results = progress["files"]
    completed_species = set(progress["completed_species"])

    # Skip completed species
    # (a completed species is processed again if the metadata gained recordings not yet downloaded)
    untracked = filtered[~filtered["xc_id"].isin(all_ok_results) & filtered["xc_file_url"].notna()]
    grown = set(untracked["ebird_species_code"])
    remaining_species = [sp for sp in species_list if sp not in completed_species or sp in grown]
    skipped_species = len(species_list) - len(remaining_species)
    remaining_recordings = sum(
        len(species_groups.get_group(sp)) for sp in remaining_species
    )

    print(f"\nDownloading {total_recordings} recordings "
          f"for {len(species_list)} species")
    print(f"  Format: {out_ext}, Concurrent: {max_concurrent}")
    if skipped_species > 0:
        print(f"  Resuming: {skipped_species} species already completed, "
              f"{len(all_ok_results)} files tracked")
        print(f"  Remaining: {remaining_recordings} recordings "
              f"for {len(remaining_species)} species")
    print(f"  Existing files will be skipped")
    print()

    if not remaining_species:
        print("All species already completed!")
        return

    # Signal handling (Ctrl+C saves progress and exits)
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

    total_stats = {"ok": 0, "exists": 0, "error": 0, "http_error": 0, "no_file_url": 0}
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
                    session, records, cfg, semaphore, rate_limit, out_ext,
                )
            except Exception as e:
                print(f"\n  Error processing {sp_code}: {e}")
                print(f"  Skipping to next species...")
                total_stats["error"] += len(records)
                # Save progress, then continue
                _save_progress(progress_path, {
                    "files": all_ok_results,
                    "completed_species": sorted(completed_species),
                })
                continue

            # Tally results
            sp_ok = 0
            for r in results:
                status = r["status"]
                if status == "ok":
                    total_stats["ok"] += 1
                    sp_ok += 1
                elif status == "exists":
                    total_stats["exists"] += 1
                    sp_ok += 1
                elif status == "no_file_url":  # not downloadable; nothing to retry
                    total_stats["no_file_url"] += 1
                    sp_ok += 1
                elif status.startswith("http_"):
                    total_stats["http_error"] += 1
                else:
                    total_stats["error"] += 1

                if status in ("ok", "exists") and "path" in r:
                    all_ok_results[r["xc_id"]] = r["path"]

            # A species is complete when every file is ok/exists (or has no file URL)
            if sp_ok == len(records):
                completed_species.add(sp_code)

            # Save progress after every species
            _save_progress(progress_path, {
                "files": all_ok_results,
                "completed_species": sorted(completed_species),
            })

    # Summary
    print(f"\n{'='*60}")
    print("XC Download Results")
    print(f"{'='*60}")
    for status, cnt in sorted(total_stats.items()):
        print(f"  {status}: {cnt}")
    print(f"  Total tracked files: {len(all_ok_results)}")
    print(f"  Completed species: {len(completed_species)}/{len(species_list)}")

    # Write file paths into the metadata
    if all_ok_results:
        df["file_path"] = df["xc_id"].apply(lambda x: all_ok_results.get(x, ""))
        save_metadata(df, metadata_path)

    # Exit non-zero if species remain incomplete (the shell driver retries)
    if shutdown_requested:
        print(f"\n  Download interrupted — run again to continue.")
        sys.exit(1)
    if len(completed_species) < len(species_list):
        incomplete = len(species_list) - len(completed_species)
        print(f"\n  {incomplete} species incomplete (HTTP errors etc.) — "
              f"run again to retry.")
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
        print("  --metadata-only: collect metadata from XC API")
        print("  --download:      download audio files (after metadata)")
        sys.exit(1)


if __name__ == "__main__":
    main()
