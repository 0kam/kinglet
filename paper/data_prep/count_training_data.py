# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Count the training and validation recordings and windows of Appendix S1.

Reads the three training manifests (train_distill.SOURCES) and the window index of the teacher parts
(<distill_root>/<source>/part-*.npz `ids`, one row per 5-s window, the parts used by training as
listed by train_distill.parts_index). A recording is "listed" when its manifest split is "pool" and
an "entry" when it has at least one window. The same Xeno-canto recording can appear in the Japanese
manifest (rec_id "xc:N") and in the BirdSet manifests (rec_id "XCN"); `canonical` maps both to "XCN"
to count distinct recordings and second copies.

Usage: python count_training_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_distill import SOURCES, parts_index  # noqa: E402

SOURCE_NAMES = {"xcl": "XCL", "xcm": "XCM", "xeno-canto": "Xeno-canto", "inat": "iNaturalist",
                "inat-api": "iNaturalist", "macaulay": "Macaulay Library"}


def canonical(rec_id: str) -> str:
    return "XC" + rec_id[3:] if rec_id.startswith("xc:") else rec_id


def main():
    man = pd.concat([pd.read_parquet(p, columns=["rec_id", "species", "source", "split"]).assign(part_src=s)
                     for s, p in SOURCES.items()], ignore_index=True)
    man["rec_id"] = man["rec_id"].astype(str)
    pool, val = man[man["split"] == "pool"], man[man["split"] == "val"]

    win = {"train": 0, "val": 0}
    with_win = {"train": set(), "val": set()}  # (source dir, rec_id) with at least one window
    win_per_rec = {}  # (source dir, rec_id) -> training windows
    for part in parts_index(0, SOURCES):
        src = part["npz"].parent.name
        ids = np.load(part["npz"])["ids"].astype(str)
        for which in ("train", "val"):
            sel = ids[part[which]]
            win[which] += len(sel)
            with_win[which].update((src, i) for i in sel)
        for i in ids[part["train"]]:
            win_per_rec[(src, i)] = win_per_rec.get((src, i), 0) + 1

    print("Listed training recordings by source:")
    for name, n in pool["source"].map(SOURCE_NAMES).value_counts().items():
        print(f"  {name:18s} {n:9,d}")
    print(f"  {'total':18s} {len(pool):9,d}")
    print(f"  Macaulay share     {(pool['source'] == 'macaulay').mean():9.1%}")

    keys = list(zip(pool["part_src"], pool["rec_id"]))
    has = np.array([k in with_win["train"] for k in keys])
    entries = pool[has]
    print(f"Listed training recordings without a window: {(~has).sum():,}")
    print(f"Training entries (with windows): {len(entries):,}")

    canon = entries["rec_id"].map(canonical)
    second = canon.duplicated(keep="first")
    dup = entries[canon.isin(canon[second])]
    print(f"Second copies: {second.sum():,} (sources of all copies: {dup['source'].value_counts().to_dict()})")
    dup_win = sum(win_per_rec[(s, r)] for s, r in zip(entries.loc[second, "part_src"], entries.loc[second, "rec_id"]))
    print(f"Windows of the second copies: {dup_win:,} ({dup_win / win['train']:.1%} of the training windows)")
    print(f"Distinct training recordings: {canon.nunique():,} of {entries['species'].nunique():,} species")
    print(f"Training windows per epoch: {win['train']:,} ({win['train'] / 1e6:.2f} million); "
          f"of the distinct recordings: {win['train'] - dup_win:,} ({(win['train'] - dup_win) / 1e6:.2f} million)")

    vkeys = list(zip(val["part_src"], val["rec_id"]))
    vhas = np.array([k in with_win["val"] for k in vkeys])
    vcanon = val.loc[vhas, "rec_id"].map(canonical)
    print(f"Validation recordings: {len(val):,} listed, {vhas.sum():,} with windows, {vcanon.nunique():,} distinct")
    print(f"Validation windows: {win['val']:,} ({win['val'] / 1e6:.2f} million)")
    print(f"Recordings in both the training and the validation set: {len(set(canon) & set(vcanon)):,}")


if __name__ == "__main__":
    main()
