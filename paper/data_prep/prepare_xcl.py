# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Build the XCL training manifest and extract its audio (Appendix S1).

BirdSet XCL (528,422 Xeno-canto recordings of 9,734 species, downloaded to <downloads>/birdset_XCL)
becomes the global training pool:

  1. remove recordings (by XC number) already in the Japanese manifest or the XCM manifest,
  2. split val/pool per species by recording site (build_manifest.split_by_site,
     val <= 8, pool <= --cap per species; the paper used cap 500),
  3. stream-extract the selected files from the 98 tar shards (resumable,
     prepare_birdset.extract) to <data_root>/birdset/xcl/.

Usage:  python prepare_xcl.py [--cap 500] [--workers 6] [--manifest-only]
Output: <data_root>/birdset/xcl_manifest.parquet + <data_root>/birdset/xcl/XC*.ogg
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config  # noqa: E402
from data_prep.build_manifest import split_by_site  # noqa: E402
from data_prep.prepare_birdset import extract  # noqa: E402

ROOT = config.DATA_ROOT
XCL = config.DOWNLOADS / "birdset_XCL/XCL"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=int, default=500, help="max pool recordings per species")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--manifest-only", action="store_true", help="plan + stats, no extraction")
    args = ap.parse_args()

    xcl = pd.read_parquet(XCL / "XCL_metadata.parquet")
    xcl["rec_id"] = xcl["filepath"].map(lambda p: Path(p).stem)
    have = set(pd.read_parquet(ROOT / "manifest.parquet")["rec_id"].astype(str))
    have |= set(pd.read_parquet(ROOT / "birdset/train_manifest.parquet")["rec_id"].astype(str))
    rec = (xcl[~xcl["rec_id"].isin(have)].drop_duplicates("rec_id")
           .rename(columns={"ebird_code": "species", "lat": "latitude", "long": "longitude"}))
    print(f"XCL {len(xcl):,} -> {len(rec):,} recs / {rec['species'].nunique():,} species "
          f"after dedupe vs JP+XCM", flush=True)

    dst = ROOT / "birdset/xcl"
    man = split_by_site(rec[["rec_id", "species", "latitude", "longitude", "filepath", "length"]],
                        lambda sp: args.cap)
    man["source"] = "xcl"
    man["file_path"] = man["filepath"].map(lambda p: str(dst / Path(p).name))
    win = (man["length"].clip(upper=60) // 5).clip(lower=1).sum()
    print(f"manifest: {len(man):,} recs ({int((man['split'] == 'val').sum()):,} val), "
          f"~{win / 1e6:.1f}M windows", flush=True)
    if args.manifest_only:
        return

    dst.mkdir(parents=True, exist_ok=True)
    names = set(man["filepath"].map(lambda p: Path(p).name))
    with Pool(args.workers) as pool:
        n = sum(pool.map(extract, [(t, names, dst) for t in sorted(XCL.glob("XCL_shard_*.tar.gz"))]))
    print(f"extracted {n:,} new files", flush=True)

    man = man[man["file_path"].map(lambda p: Path(p).exists())]
    out = ROOT / "birdset/xcl_manifest.parquet"
    man.drop(columns=["filepath", "length"]).to_parquet(out, index=False)
    per = man[man["split"] == "pool"].groupby("species").size()
    print(f"wrote {out}: {len(man):,} recs, {man['species'].nunique():,} species, "
          f"pool min {per.min()} median {per.median():.0f} max {per.max()}", flush=True)


if __name__ == "__main__":
    main()
