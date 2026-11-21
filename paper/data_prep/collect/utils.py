# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Shared helpers for the training-audio collection scripts.

Loading the configuration and species list, the Japan bounding-box test, metadata I/O and
audio file paths. All audio and metadata live under `common.config.AUDIO_ROOT`; the paths in
config.yaml are relative to it.

API keys are not shipped. The Xeno-canto key is read from the environment variable
XC_API_KEY, or from the file named by XC_API_KEY_FILE (see `load_api_key`).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import AUDIO_ROOT, PAPER  # noqa: E402

PROJECT_ROOT = PAPER
STEP_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    """Read config.yaml and add the audio root (`audio_root`)."""
    cfg_path = STEP_DIR / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["audio_root"] = str(AUDIO_ROOT)
    return cfg


def audio_path(cfg: dict, relative: str) -> Path:
    """Resolve a path relative to the audio root."""
    return Path(cfg["audio_root"]) / relative


def load_species_list(cfg: dict | None = None) -> pd.DataFrame:
    """Read species_list.csv."""
    if cfg is None:
        cfg = load_config()
    csv_path = PROJECT_ROOT / cfg["species_list"]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


def get_target_species(cfg: dict | None = None) -> pd.DataFrame:
    """Return only the species matched to eBird (ebird_species_code, scientific_name, ebird_sciname, ...)."""
    df = load_species_list(cfg)
    matched = df[df["ebird_matched"] == True].copy()  # noqa: E712
    cols = [
        "species_num", "scientific_name", "japanese_name",
        "ebird_sciname", "ebird_common_name", "ebird_species_code",
        "order", "family",
    ]
    return matched[cols].reset_index(drop=True)


def is_in_japan(lat: float | None, lon: float | None, cfg: dict | None = None) -> bool:
    """Whether a latitude/longitude lies inside the Japan bounding box."""
    if lat is None or lon is None:
        return False
    if cfg is None:
        cfg = load_config()
    bounds = cfg["japan_bounds"]
    return (
        bounds["lat_min"] <= lat <= bounds["lat_max"]
        and bounds["lon_min"] <= lon <= bounds["lon_max"]
    )


def save_metadata(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(path), compression="snappy")
    print(f"Saved {len(df)} rows → {path}")


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file."""
    return pq.read_table(str(path)).to_pandas()


def audio_file_path(
    cfg: dict,
    source: str,
    species_code: str,
    recording_id: str,
    ext: str = "wav",
) -> Path:
    """Storage path of one audio file.

    Example: <audio_root>/audio/xeno-canto/wav/leucig1/xc_12345.wav
    """
    if source == "xeno-canto":
        base = cfg["xeno_canto"]["wav_dir"]
    elif source == "inat":
        base = cfg["inat_sounds"]["filtered_dir"]
    elif source == "inat-api":
        base = cfg["inat_api"]["audio_dir"]
    elif source == "macaulay":
        base = cfg["macaulay"]["audio_dir"]
    else:
        raise ValueError(f"Unknown source: {source}")

    return audio_path(cfg, base) / species_code / f"{recording_id}.{ext}"


def load_api_key(prefix: str = "XC") -> str:
    """Read an API key from env `<prefix>_API_KEY`, else from the file named by `<prefix>_API_KEY_FILE`."""
    key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    if key:
        return key
    key_file = os.environ.get(f"{prefix}_API_KEY_FILE")
    if key_file:
        return Path(key_file).expanduser().read_text().strip()
    raise RuntimeError(
        f"No API key: set {prefix}_API_KEY to the key, or {prefix}_API_KEY_FILE to a file containing it."
    )


# ── Unified metadata schema ──

UNIFIED_SCHEMA = [
    "recording_id",       # "xc:12345", "inat:67890"
    "source",             # "xeno-canto" / "inat" / "inat-api" / "macaulay"
    "ebird_species_code",
    "scientific_name",
    "japanese_name",
    "latitude",
    "longitude",
    "country",
    "is_japan",
    "duration_sec",
    "sample_rate",
    "quality",            # XC quality A-E (empty for other sources)
    "license",
    "file_path",          # path of the audio file
    "vocalization_type",  # song / call / alarm, etc.
]
