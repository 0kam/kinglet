# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Locations of data, intermediate files and outputs for the scripts under paper/.

Every path can be set in paper/config.yaml (lines of the form `key: value`) or through an
environment variable of the same name in upper case; the defaults live under paper/_work/.

  data_root       benchmark manifests, teacher outputs on the benchmarks and score caches
  distill_root    teacher targets for training, checkpoints, deployment files, analysis caches
  audio_root      collected training audio and taxonomy files
  downloads       downloaded dataset archives
  wabad_src       the unpacked WABAD download (Zenodo record 17293588: Metadata.csv, annotations, audio)
  nonbird_root    the four datasets of taxa other than birds
  release_assets  the released model files (ONNX networks, output layers, reliability table)
  perch_dir       the Perch 2.0 SavedModel directory (CPU variant, Kaggle perch_v2_cpu/1)
  perch_gpu_dir   the Perch 2.0 GPU variant (Kaggle perch_v2/2; same weights), used with --gpu
"""
import json
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
FIGURE_DATA = PAPER / "figure_data"
_DEFAULTS = {"data_root": "_work/data", "distill_root": "_work/distill", "audio_root": "_work/audio",
             "downloads": "_work/downloads", "wabad_src": "_work/wabad_17293588", "nonbird_root": "_work/nonbird",
             "release_assets": "_work/release_assets", "perch_dir": "_work/perch_v2", "perch_gpu_dir": "_work/perch_v2_gpu"}


def _settings():
    cfg = dict(_DEFAULTS)
    f = PAPER / "config.yaml"
    if f.exists():
        for line in f.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in cfg and value.strip():
                cfg[key.strip()] = value.strip().strip("'\"")
    for key in cfg:
        cfg[key] = os.environ.get(key.upper(), cfg[key])
    return {k: (PAPER / v if not Path(v).expanduser().is_absolute() else Path(v).expanduser()) for k, v in cfg.items()}


_S = _settings()
DATA_ROOT, DISTILL_ROOT, AUDIO_ROOT = _S["data_root"], _S["distill_root"], _S["audio_root"]
DOWNLOADS, NONBIRD_ROOT, RELEASE_ASSETS = _S["downloads"], _S["nonbird_root"], _S["release_assets"]
PERCH_DIR, PERCH_GPU_DIR, WABAD_SRC = _S["perch_dir"], _S["perch_gpu_dir"], _S["wabad_src"]


def jp_list():
    """The 317 Japanese species of the paper: [{'species': eBird code, 'perch_idx': row of the Perch 2.0 class list, or -1}]."""
    return json.loads((PAPER / "data_prep/jp_list.json").read_text())
