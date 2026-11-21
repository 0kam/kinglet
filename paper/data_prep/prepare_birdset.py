# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Prepare the BirdSet soundscape test sets and the XCM focal training recordings.

- Test: unpack <S>_test5s_shard_*.tar.gz and write the multi-label targets of each 5-s window
  (ebird_code_multilabel) to <data_root>/birdset/<S>_test.parquet (window_id, path, labels).
- Training: stream-extract from the XCM tar files only the recordings of the species that occur in
  the given test sets, split them into val/pool with build_manifest.split_by_site and write
  <data_root>/birdset/train_manifest.parquet (pool cap 256 per species; rec_id = XC number).
  The training data of the paper used the default --sets POW,HSN; the other test sets
  (NBP, NES, PER, SNE, SSW, UHH) are prepared with --test-only.

Input: the BirdSet downloads under <downloads>/birdset_<S>/ (download/download_birdset.sh).

Usage: python prepare_birdset.py [--sets POW,HSN] [--workers 6] [--test-only]
"""

import argparse
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config  # noqa: E402
from data_prep.build_manifest import POOL_MAX, split_by_site  # noqa: E402

DL = config.DOWNLOADS
OUT = config.DATA_ROOT / "birdset"


def extract(args: tuple[Path, set[str], Path]) -> int:
    """Read a tar once and write the members whose base name is in names to dst/<basename>."""
    tar_path, names, dst = args
    n = 0
    with tarfile.open(tar_path, "r|gz") as tar:
        for m in tar:
            base = Path(m.name).name
            if m.isfile() and base in names and not (dst / base).exists():
                f = tar.extractfile(m)
                tmp = dst / (base + ".tmp")
                tmp.write_bytes(f.read())
                tmp.rename(dst / base)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", default="POW,HSN")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--test-only", action="store_true",
                    help="stop after writing <S>_test.parquet (leave xcm/ and train_manifest.parquet unchanged)")
    args = ap.parse_args()
    sets = args.sets.split(",")
    species = set()
    for s in sets:
        src, dst = DL / f"birdset_{s}" / s, OUT / f"{s}_test5s"
        dst.mkdir(parents=True, exist_ok=True)
        meta = pd.read_parquet(src / f"{s}_metadata_test_5s.parquet")
        want = set(meta.index)
        with Pool(args.workers) as pool:
            pool.map(extract, [(t, want, dst) for t in sorted(src.glob(f"{s}_test5s_shard_*.tar.gz"))])
        test = pd.DataFrame({"window_id": meta.index, "path": [str(dst / f) for f in meta.index],
                             "labels": [list(x) for x in meta["ebird_code_multilabel"]]})
        assert all(Path(p).exists() for p in test["path"]), f"{s}: missing test audio"
        test.to_parquet(OUT / f"{s}_test.parquet", index=False)
        sp = {c for x in test["labels"] for c in x}
        species |= sp
        print(f"{s}: {len(test)} windows, {len(sp)} species, "
              f"{sum(len(x) == 0 for x in test['labels'])} empty", flush=True)

    if args.test_only:
        return

    xcm_dir = DL / "birdset_XCM/XCM"
    xcm = pd.read_parquet(xcm_dir / "XCM_metadata.parquet")
    xcm = xcm[xcm["ebird_code"].isin(species)]
    dst = OUT / "xcm"
    dst.mkdir(exist_ok=True)
    names = set(xcm["filepath"].map(lambda p: Path(p).name))
    with Pool(args.workers) as pool:
        n = sum(pool.map(extract, [(t, names, dst) for t in sorted(xcm_dir.glob("XCM_shard_*.tar.gz"))]))
    print(f"XCM: extracted {n} new files for {len(species)} species", flush=True)

    rec = pd.DataFrame({
        "rec_id": xcm["filepath"].map(lambda p: Path(p).stem),
        "species": xcm["ebird_code"],
        "source": "xcm",
        "file_path": xcm["filepath"].map(lambda p: str(dst / Path(p).name)),
        "latitude": xcm["lat"],
        "longitude": xcm["long"],
    })
    rec = rec[rec["file_path"].map(lambda p: Path(p).exists())].drop_duplicates("rec_id")
    missing = species - set(rec["species"])
    man = split_by_site(rec, lambda sp: POOL_MAX)
    man.to_parquet(OUT / "train_manifest.parquet", index=False)
    per_sp = man[man["split"] == "pool"].groupby("species").size()
    print(f"train: {len(man):,} recordings, {man['species'].nunique()} species "
          f"(no XCM audio: {sorted(missing)}), pool min {per_sp.min()} median {per_sp.median():.0f}")


if __name__ == "__main__":
    main()
