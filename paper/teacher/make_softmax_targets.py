# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Full-vocabulary softmax targets of the Perch 2.0 teacher for every training window.

Reruns the teacher on the stored int16 audio windows (part-NNNN.raw.npy from teacher/prepare_mels.py --store audio)
of the Japanese and BirdSet XCM training sources and writes f16 probability targets aligned with the rows of the
teacher's part-NNNN.npz:

  DISTILL_ROOT/softmax_full/<source>/part-NNNN.npy   (n_rows, 14795) f16

Full vectors are stored because the teacher's prototype label head gives a diffuse softmax that top-k cannot
capture. The XCL source gets its targets directly from teacher/embed_perch.py --softmax-full-out.
Shardable and resumable (existing outputs are skipped):

  CUDA_VISIBLE_DEVICES="" python teacher/make_softmax_targets.py --shard 0 3
  TF_FORCE_GPU_ALLOW_GROWTH=true python teacher/make_softmax_targets.py --gpu --reverse

--gpu uses the Perch 2.0 GPU variant: same weights as the CPU variant up to numeric noise (max |logit difference|
0.9, mean 0.005 on a checked part). --reverse walks the job list backwards so a GPU worker can sweep toward the
forward CPU shards; writes are per-process temporary files and an atomic rename.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DISTILL_ROOT, PERCH_DIR, PERCH_GPU_DIR  # noqa: E402

SOURCES = ("emb_train60", "emb_xcm60")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"),
                    help="process parts with global_index %% N == K")
    ap.add_argument("--gpu", action="store_true", help="use the Perch 2.0 GPU variant, batch 128")
    ap.add_argument("--reverse", action="store_true", help="walk the shard's job list backwards")
    args = ap.parse_args()
    k, n = args.shard
    batch = 128 if args.gpu else 64

    import tensorflow as tf  # after CUDA_VISIBLE_DEVICES is set by the caller
    fn = tf.saved_model.load(str(PERCH_GPU_DIR if args.gpu else PERCH_DIR)).signatures["serving_default"]

    jobs = [(src, npz) for src in SOURCES for npz in sorted((DISTILL_ROOT / src).glob("part-*.npz"))
            if npz.with_suffix(".raw.npy").exists()]
    jobs = jobs[k::n]
    if args.reverse:
        jobs = jobs[::-1]
    print(f"shard {k}/{n}: {len(jobs)} parts", flush=True)
    for src, npz in jobs:
        dst = DISTILL_ROOT / "softmax_full" / src / (npz.stem + ".npy")
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = np.load(npz.with_suffix(".raw.npy"), mmap_mode="r")
        n_rows = len(np.load(npz)["ids"])
        assert len(raw) >= n_rows, f"{npz}: raw {len(raw)} < npz {n_rows}"
        t0, out = time.time(), None
        for i in range(0, n_rows, batch):
            x = raw[i : i + batch].astype(np.float32) / 32768.0
            nb = len(x)
            if nb < batch:  # pad to the compiled batch shape (XLA recompiles per shape)
                x = np.pad(x, ((0, batch - nb), (0, 0)))
            lab = fn(inputs=tf.constant(x))["label"].numpy()[:nb]
            lab -= lab.max(1, keepdims=True)
            p = np.exp(lab) / np.exp(lab).sum(1, keepdims=True)
            if out is None:
                out = np.zeros((n_rows, p.shape[1]), np.float16)
            out[i : i + batch] = p
        tmp = dst.parent / f"{dst.stem}.tmp{os.getpid()}.npy"
        np.save(tmp, out)
        tmp.rename(dst)
        print(f"{src}/{npz.stem}: {n_rows} rows, {time.time() - t0:.0f}s, median p_max {np.median(out.max(1)):.3f}",
              flush=True)
    print(f"=== SHARD {k}/{n} DONE ===", flush=True)


if __name__ == "__main__":
    main()
