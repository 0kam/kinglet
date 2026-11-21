# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""nonbird_v1: embeddings + class scores of the Perch 2.0 teacher or a Kinglet student on a manifest.

Windows and decode follow teacher.embed_perch (ffmpeg -> mono 32 kHz; first MAX_S = 60 s of each file; 5-s windows at t = 0, 5, ...,
only complete windows; a file shorter than 5 s is zero-padded to one window). Scores are taken at
the Perch vocabulary indices in --perch-idx (a JSON list):
  teacher: `logit` (the raw class logit, as in the paper's benchmarks) and `softmax` (softmax over
           all 14,795 logits, sliced)
  student: `softmax`, the exact fold (softmax over the full-vocabulary head, sliced), via
           evaluate.accuracy.score_student.Distilled - the same code path as the benchmark region fold.

Output <out>.npz: ids, window_idx, t_start, emb float16 (N, 1536), logit / softmax float32 (N, K).

Usage: python evaluate/nonbird/embed_nonbird.py --model B --manifest m.parquet --perch-idx idx.json --out x.npz
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DISTILL_ROOT as DD  # noqa: E402
from common.config import PERCH_DIR, PERCH_GPU_DIR  # noqa: E402
from teacher.embed_perch import decode, windows  # noqa: E402

STUDENTS = {"B": DD / "g3conf_v1/regnety_016_full/best.pt",
            "S": DD / "g3conf_v1/efficientnet_lite0_full/best.pt",
            "XS": DD / "g2xcl_v1/mobilenetv4_conv_small_full/best.pt"}
CHUNK = 256
GPU_MB = 1800
MAX_S = 60.0  # clips longer than this: only the first 60 s are windowed (bounds memory and time)


def teacher_fn(idx: np.ndarray, gpu: bool, batch: int):
    """Perch 2.0 GPU or CPU build: same weights up to numeric noise (teacher/make_softmax_targets.py)."""
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):  # hard cap on the GPU memory TF takes
        tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MB)])
    tf.config.threading.set_intra_op_parallelism_threads(12)
    fn = tf.saved_model.load(str(PERCH_GPU_DIR if gpu else PERCH_DIR)).signatures["serving_default"]

    def run(x: np.ndarray) -> dict:
        n = len(x)
        o = fn(inputs=tf.constant(np.pad(x, ((0, batch - n), (0, 0)))))  # compiled batch shape
        lg = o["label"].numpy()[:n].astype(np.float64)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return {"emb": o["embedding"].numpy()[:n], "logit": lg[:, idx], "softmax": p[:, idx]}
    return run


def student_fn(name: str, idx: np.ndarray, gpu: bool):
    import torch

    from evaluate.accuracy.score_student import Distilled

    m = Distilled(STUDENTS[name], "cuda" if gpu and torch.cuda.is_available() else "cpu")
    assert m.full, "needs a full-vocabulary checkpoint"
    m.fold_idx = torch.from_numpy(idx).to(m.device)
    m.fold_ok = torch.ones(len(idx), device=m.device)

    def run(x: np.ndarray) -> dict:
        s = m(x)  # softmax over all classes, sliced at idx
        return {"emb": m.emb.cpu().numpy(), "softmax": s}
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=["teacher", *STUDENTS])
    ap.add_argument("--manifest", type=Path, required=True, help="parquet with columns item_id, path")
    ap.add_argument("--perch-idx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--gpu", action="store_true", help="default CPU")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    if args.out.exists():
        print("already done", args.out)
        return

    idx = np.array(json.loads(args.perch_idx.read_text()), dtype=np.int64)
    run = teacher_fn(idx, args.gpu, args.batch) if args.model == "teacher" else student_fn(args.model, idx, args.gpu)
    man = pd.read_parquet(args.manifest)
    rows = list(zip(man["item_id"].astype(str), man["path"]))
    pool = ThreadPoolExecutor(args.decoders)
    meta, out = [], {}
    for c in range(0, len(rows), CHUNK):
        chunk = rows[c:c + CHUNK]
        wins = []
        for (rid, path), a in zip(chunk, pool.map(lambda r: decode(r[1], MAX_S), chunk)):
            assert a is not None, f"decode failed: {rid} ({path})"
            starts, w = windows(a, 5.0)
            meta += [(rid, i, s) for i, s in enumerate(starts)]
            wins += list(w)
        for b in range(0, len(wins), args.batch):
            for k, v in run(np.stack(wins[b:b + args.batch])).items():
                out.setdefault(k, []).append(v.astype(np.float16 if k == "emb" else np.float32))
        print(f"{args.model}: {min(c + CHUNK, len(rows))}/{len(rows)} items, {len(meta)} windows", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array([m[0] for m in meta]), window_idx=np.array([m[1] for m in meta], np.int32),
             t_start=np.array([m[2] for m in meta], np.float32),
             **{k: np.concatenate(v) for k, v in out.items()})
    tmp.rename(args.out)
    print("saved", args.out, len(meta), "windows")


if __name__ == "__main__":
    main()
