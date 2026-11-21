# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""eBird reporting frequency of the target species in Japan.

Reads the eBird checklist reporting frequency (detection frequency) for Japan and derives a
frequency score per species, used to separate common species from vagrants.

Two methods:

Method 1: bar-chart TSV (recommended, most accurate)
  Parses the histogram TSV downloaded from the eBird website: detection frequency in 48
  periods per year (4 per month), which also gives the seasonal pattern.

Method 2: eBird API v2 (automatic, approximate)
  Fetches the Japanese species list and recent observations from the API and estimates a
  frequency. Based on 30 days of observations, so it is seasonally biased. Needs an eBird API
  key, read from the environment variable EBIRD_API_KEY or from the file named by
  EBIRD_API_KEY_FILE (no key is shipped).

Usage:
  # Method 1: from the TSV file (recommended)
  python get_ebird_frequency.py --from-tsv ebird_JP_barchart.tsv

  # Method 2: from the API
  python get_ebird_frequency.py --from-api

  # Only print the TSV download URL
  python get_ebird_frequency.py --show-url

Output (default): data_prep/species/ebird_frequency.csv
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collect"))
from utils import get_target_species, load_api_key, load_config  # noqa: E402

# ── Constants ──

EBIRD_API_BASE = "https://api.ebird.org/v2"
BARCHART_URL = "https://ebird.org/barchartData"
REGION_CODE = "JP"
OUTPUT_PATH = Path(__file__).resolve().parent / "ebird_frequency.csv"

# Labels of the 48 periods (4 per month)
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
PERIOD_LABELS = [f"{m}-{w}" for m in MONTH_NAMES for w in range(1, 5)]

# Month groups for the seasonal classification (1-indexed)
SEASON_MONTHS = {
    "breeding": {4, 5, 6, 7, 8},       # Apr-Aug
    "winter": {11, 12, 1, 2, 3},        # Nov-Mar
    "spring_passage": {3, 4, 5},        # Mar-May
    "autumn_passage": {8, 9, 10, 11},   # Aug-Nov
}


def ebird_api_headers(api_key: str) -> dict:
    """eBird API v2 request headers."""
    return {"x-ebirdapitoken": api_key}


# ── Method 1: bar-chart TSV ──

def get_barchart_download_url(
    region: str = REGION_CODE,
    byr: int = 1900,
    eyr: int = 2026,
) -> str:
    """Download URL of the eBird bar-chart TSV."""
    params = {
        "r": region,
        "bmo": 1,
        "emo": 12,
        "byr": byr,
        "eyr": eyr,
        "fmt": "tsv",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{BARCHART_URL}?{qs}"


def parse_barchart_tsv(tsv_path: str | Path) -> pd.DataFrame:
    """Parse an eBird bar-chart TSV file.

    TSV layout:
      - a header line (parsed by position, column names unused)
      - a "Sample Size" line with the number of checklists
      - then: species name + detection frequency (0.0-1.0) in 48 periods

    Returns:
        DataFrame of detection frequencies per species
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    raw = tsv_path.read_text(encoding="utf-8")
    lines = raw.strip().split("\n")

    if len(lines) < 3:
        raise ValueError("Malformed TSV file (too few lines)")

    # Header line
    header = lines[0].split("\t")

    # Find the Sample Size line (within the first few lines)
    sample_sizes = None
    data_start = 1
    for i, line in enumerate(lines[1:6], start=1):
        cols = line.split("\t")
        if cols and "sample size" in cols[0].lower():
            # Numeric columns
            sample_sizes = []
            for v in cols[1:49]:
                try:
                    sample_sizes.append(int(float(v)))
                except (ValueError, IndexError):
                    sample_sizes.append(0)
            data_start = i + 1
            break

    # Species rows
    records = []
    for line in lines[data_start:]:
        cols = line.split("\t")
        if not cols or not cols[0].strip():
            continue

        species_name = cols[0].strip()

        # In the eBird TSV the species name is followed by the 48 period frequencies
        freqs = []
        for v in cols[1:49]:
            v = v.strip()
            if v == "" or v == "—" or v == "-":
                freqs.append(0.0)
            else:
                try:
                    freqs.append(float(v))
                except ValueError:
                    freqs.append(0.0)

        # Pad with zeros to 48 periods
        while len(freqs) < 48:
            freqs.append(0.0)

        records.append({"species_name": species_name, "freqs": freqs})

    if not records:
        raise ValueError("No species rows found in the TSV")

    # To DataFrame
    rows = []
    for rec in records:
        row = {"species_name": rec["species_name"]}
        for j, label in enumerate(PERIOD_LABELS):
            row[label] = rec["freqs"][j]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Annual mean frequency and statistics
    freq_cols = PERIOD_LABELS
    df["frequency_annual_mean"] = df[freq_cols].mean(axis=1)
    df["frequency_max"] = df[freq_cols].max(axis=1)
    df["frequency_min"] = df[freq_cols].min(axis=1)
    df["n_periods_detected"] = (df[freq_cols] > 0).sum(axis=1)

    print(f"Bar-chart TSV: {len(df)} species parsed")
    if sample_sizes:
        total_checklists = sum(sample_sizes)
        print(f"Total checklists: {total_checklists:,}")

    return df


def classify_residence_status(row: pd.Series) -> str:
    """Residence status from the monthly frequency pattern.

    Returns:
        resident / summer / winter / passage / vagrant / undetected
    """
    freq_cols = PERIOD_LABELS
    freqs = row[freq_cols].values

    if all(f == 0 for f in freqs):
        return "undetected"

    annual_mean = row["frequency_annual_mean"]

    # Very rare -> vagrant
    if annual_mean < 0.001 and row["n_periods_detected"] <= 4:
        return "vagrant"

    # Monthly mean frequency (4 periods each)
    monthly_means = []
    for m in range(12):
        start = m * 4
        month_freqs = freqs[start : start + 4]
        monthly_means.append(sum(month_freqs) / 4)

    # Mean frequency in the breeding and wintering seasons
    breeding_freq = sum(monthly_means[m - 1] for m in SEASON_MONTHS["breeding"]) / len(
        SEASON_MONTHS["breeding"]
    )
    winter_freq = sum(monthly_means[m - 1] for m in SEASON_MONTHS["winter"]) / len(
        SEASON_MONTHS["winter"]
    )

    max_monthly = max(monthly_means)
    if max_monthly == 0:
        return "undetected"

    # Detected all year -> resident
    n_months_detected = sum(1 for m in monthly_means if m > max_monthly * 0.1)
    if n_months_detected >= 10:
        return "resident"

    # Concentrated in the breeding season -> summer visitor
    if breeding_freq > winter_freq * 3 and breeding_freq > 0.001:
        return "summer"

    # Concentrated in winter -> winter visitor
    if winter_freq > breeding_freq * 3 and winter_freq > 0.001:
        return "winter"

    # Otherwise, only a few months -> passage migrant
    if n_months_detected <= 6:
        return "passage"

    return "resident"


# ── Method 2: eBird API v2 ──

def api_get_species_list(api_key: str, region: str = REGION_CODE) -> list[str]:
    """eBird API: species list of a region."""
    url = f"{EBIRD_API_BASE}/product/spplist/{region}"
    resp = requests.get(url, headers=ebird_api_headers(api_key), timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_get_recent_observations(
    api_key: str,
    region: str = REGION_CODE,
    back: int = 30,
) -> list[dict]:
    """eBird API: recent observations."""
    url = f"{EBIRD_API_BASE}/data/obs/{region}/recent"
    params = {"back": back}
    resp = requests.get(
        url, headers=ebird_api_headers(api_key), params=params, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def frequency_from_api(api_key: str) -> pd.DataFrame:
    """Estimate frequencies from the eBird API.

    1. Species list of Japan (/product/spplist/JP)
    2. Observations of the last 30 days
    3. Number of observations per species
    """
    print("eBird API: fetching the species list of Japan...")
    jp_species = api_get_species_list(api_key)
    print(f"  → {len(jp_species)} species")
    time.sleep(1)

    print("eBird API: fetching observations of the last 30 days...")
    observations = api_get_recent_observations(api_key, back=30)
    print(f"  → {len(observations)} observations")

    # Observation count and individuals per species
    species_counts: dict[str, dict] = {}
    for obs in observations:
        code = obs.get("speciesCode", "")
        if not code:
            continue
        if code not in species_counts:
            species_counts[code] = {
                "obs_count": 0,
                "total_individuals": 0,
                "comName": obs.get("comName", ""),
                "sciName": obs.get("sciName", ""),
            }
        species_counts[code]["obs_count"] += 1
        how_many = obs.get("howMany")
        if how_many is not None:
            species_counts[code]["total_individuals"] += how_many

    # To DataFrame
    max_obs = max((v["obs_count"] for v in species_counts.values()), default=1)

    rows = []
    for code in jp_species:
        info = species_counts.get(code, {})
        obs_count = info.get("obs_count", 0)
        rows.append({
            "ebird_species_code": code,
            "scientific_name_ebird": info.get("sciName", ""),
            "common_name_ebird": info.get("comName", ""),
            "observation_count_30d": obs_count,
            "frequency_score": obs_count / max_obs if max_obs > 0 else 0,
            "in_recent_30d": obs_count > 0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("observation_count_30d", ascending=False).reset_index(drop=True)

    return df


# ── Merge ──

def merge_with_target_species(
    freq_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Merge the frequency data with the target species list."""
    cfg = load_config()
    target = get_target_species(cfg)

    if method == "tsv":
        # Extract the scientific and Japanese names from the TSV species names for matching
        freq_df = freq_df.copy()
        freq_df["sci_extracted"] = freq_df["species_name"].apply(
            lambda x: _extract_scientific_name(x) if isinstance(x, str) else ""
        )
        freq_df["jpn_extracted"] = freq_df["species_name"].apply(
            lambda x: _extract_japanese_name(x) if isinstance(x, str) else ""
        )

        # Match on the scientific name (scientific_name -> sci_extracted)
        merged = target.merge(
            freq_df,
            left_on="scientific_name",
            right_on="sci_extracted",
            how="left",
        )

        # Retry unmatched species on ebird_sciname
        unmatched_mask = merged["species_name"].isna()
        if unmatched_mask.sum() > 0:
            unmatched_codes = merged.loc[unmatched_mask, "ebird_species_code"].values
            retry = target[target["ebird_species_code"].isin(unmatched_codes)].merge(
                freq_df,
                left_on="ebird_sciname",
                right_on="sci_extracted",
                how="left",
                suffixes=("", "_retry"),
            )
            for _, row in retry[retry["species_name"].notna()].iterrows():
                idx = merged.index[
                    merged["ebird_species_code"] == row["ebird_species_code"]
                ]
                if len(idx) > 0:
                    for col in freq_df.columns:
                        if col in row.index:
                            merged.loc[idx[0], col] = row[col]

        # Retry the remaining species on the Japanese name
        unmatched_mask = merged["species_name"].isna()
        if unmatched_mask.sum() > 0:
            unmatched_codes = merged.loc[unmatched_mask, "ebird_species_code"].values
            retry2 = target[target["ebird_species_code"].isin(unmatched_codes)].merge(
                freq_df,
                left_on="japanese_name",
                right_on="jpn_extracted",
                how="left",
                suffixes=("", "_retry2"),
            )
            for _, row in retry2[retry2["species_name"].notna()].iterrows():
                idx = merged.index[
                    merged["ebird_species_code"] == row["ebird_species_code"]
                ]
                if len(idx) > 0:
                    for col in freq_df.columns:
                        if col in row.index:
                            merged.loc[idx[0], col] = row[col]

        n_matched = merged["species_name"].notna().sum()
        print(f"Species matched: {n_matched}/{len(target)} "
              f"({n_matched/len(target)*100:.1f}%)")

        # Residence status
        freq_cols = [c for c in PERIOD_LABELS if c in merged.columns]
        if freq_cols:
            merged["residence_status"] = merged.apply(
                lambda r: classify_residence_status(r) if pd.notna(r.get("species_name")) else "undetected",
                axis=1,
            )

        # Output columns
        output_cols = [
            "ebird_species_code", "scientific_name", "japanese_name",
            "ebird_common_name", "frequency_annual_mean", "frequency_max",
            "n_periods_detected", "residence_status",
        ]
        output_cols = [c for c in output_cols if c in merged.columns]
        result = merged[output_cols].copy()
        result = result.sort_values("frequency_annual_mean", ascending=False, na_position="last")

    elif method == "api":
        merged = target.merge(
            freq_df,
            left_on="ebird_species_code",
            right_on="ebird_species_code",
            how="left",
        )
        output_cols = [
            "ebird_species_code", "scientific_name", "japanese_name",
            "ebird_common_name", "frequency_score",
            "observation_count_30d", "in_recent_30d",
        ]
        output_cols = [c for c in output_cols if c in merged.columns]
        result = merged[output_cols].copy()
        result = result.sort_values("frequency_score", ascending=False, na_position="last")

    result = result.reset_index(drop=True)
    return result


def _extract_scientific_name(text: str) -> str:
    """Scientific name from a species-name string.

    Formats:
      - 'Common Name (Scientific name)' (English TSV)
      - '<Japanese name> (<em class="sci">Genus species</em>)' (Japanese TSV, with HTML tags)
    """
    # Scientific name inside the HTML tag
    m = re.search(r'<em[^>]*>([A-Z][a-z]+ [a-z]+[^<]*)</em>', text)
    if m:
        return m.group(1).strip()
    # Scientific name in parentheses
    m = re.search(r"\(([A-Z][a-z]+ [a-z]+)\)", text)
    if m:
        return m.group(1)
    return ""


def _extract_japanese_name(text: str) -> str:
    """Japanese name from a '<Japanese name> (<em...>...</em>)' string of the Japanese TSV."""
    # The part before the parenthesis / HTML tag is the Japanese name
    m = re.match(r'^(.+?)\s*[\(<]', text)
    if m:
        name = m.group(1).strip()
        return name
    return text.strip()


# ── Output ──

def print_summary(df: pd.DataFrame, method: str) -> None:
    """Print a summary of the result."""
    total = len(df)

    if method == "tsv":
        freq_col = "frequency_annual_mean"
        has_data = df[freq_col].notna() & (df[freq_col] > 0)
    else:
        freq_col = "frequency_score"
        has_data = df[freq_col].notna() & (df[freq_col] > 0)

    n_with_data = has_data.sum()
    n_no_data = total - n_with_data

    print("\n" + "=" * 70)
    print("eBird frequency summary")
    print("=" * 70)
    print(f"Species:      {total}")
    print(f"With data:    {n_with_data} ({n_with_data/total*100:.1f}%)")
    print(f"Without data: {n_no_data} ({n_no_data/total*100:.1f}%)")

    if method == "tsv" and "residence_status" in df.columns:
        print("\n── Residence status ──")
        status_counts = df["residence_status"].value_counts()
        for status, count in status_counts.items():
            print(f"  {status:12s}: {count:4d}")

    # Top 20
    top = df[has_data].head(20)
    print("\n── Top 20 by frequency ──")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row.get("japanese_name", "")
        sci = row.get("scientific_name", "")
        freq = row[freq_col]
        freq_str = f"{freq:.4f}" if pd.notna(freq) else "N/A"
        status = row.get("residence_status", "")
        status_str = f" [{status}]" if status else ""
        print(f"  {i:2d}. {name} ({sci}) freq={freq_str}{status_str}")

    # Bottom 20 (with data)
    bottom = df[has_data].tail(20)
    print("\n── Bottom 20 by frequency (species with data) ──")
    for i, (_, row) in enumerate(bottom.iterrows(), 1):
        name = row.get("japanese_name", "")
        sci = row.get("scientific_name", "")
        freq = row[freq_col]
        freq_str = f"{freq:.6f}" if pd.notna(freq) else "N/A"
        status = row.get("residence_status", "")
        status_str = f" [{status}]" if status else ""
        print(f"  {i:2d}. {name} ({sci}) freq={freq_str}{status_str}")

    # Species not found
    if n_no_data > 0:
        print(f"\n── No eBird data ({n_no_data} species) ──")
        no_data = df[~has_data]
        for _, row in no_data.head(30).iterrows():
            name = row.get("japanese_name", "")
            sci = row.get("scientific_name", "")
            print(f"  - {name} ({sci})")
        if n_no_data > 30:
            print(f"  ... and {n_no_data - 30} more")


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save the result as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_path} ({len(df)} rows)")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="eBird reporting frequency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-tsv",
        type=str,
        metavar="TSV_FILE",
        help="parse an eBird bar-chart TSV file",
    )
    group.add_argument(
        "--from-api",
        action="store_true",
        help="fetch from eBird API v2 (approximate)",
    )
    group.add_argument(
        "--show-url",
        action="store_true",
        help="only print the TSV download URL",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_PATH),
        help="output CSV path (default: data_prep/species/ebird_frequency.csv)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=REGION_CODE,
        help="eBird region code (default: JP)",
    )

    args = parser.parse_args()

    # Print the TSV download URL
    if args.show_url:
        url = get_barchart_download_url(region=args.region)
        print("Downloading the eBird bar-chart data:")
        print()
        print("1. Log in to eBird: https://ebird.org/home")
        print("2. Open this URL in the browser:")
        print(f"   {url}")
        print("3. Save the downloaded TSV file")
        print("4. Parse it with this script:")
        print(f"   python {Path(__file__).name} --from-tsv <saved file>")
        return

    # Method 1: TSV
    if args.from_tsv:
        print(f"Method 1: bar-chart TSV ({args.from_tsv})")
        freq_df = parse_barchart_tsv(args.from_tsv)
        result = merge_with_target_species(freq_df, method="tsv")
        print_summary(result, method="tsv")
        save_results(result, Path(args.output))
        return

    # Method 2: API
    if args.from_api:
        try:
            api_key = load_api_key("EBIRD")
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Get a key at https://ebird.org/api/keygen, or use the TSV method:", file=sys.stderr)
            print(f"  python {Path(__file__).name} --show-url", file=sys.stderr)
            sys.exit(1)

        print(f"Method 2: eBird API v2 (region: {args.region})")
        freq_df = frequency_from_api(api_key)
        result = merge_with_target_species(freq_df, method="api")
        print_summary(result, method="api")
        save_results(result, Path(args.output))
        return


if __name__ == "__main__":
    main()
