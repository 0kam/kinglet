# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Run the Perch 2.0 teacher (TF SavedModel) on 5-s windows of recordings and store embeddings and logits.

Audio is decoded by ffmpeg to mono 32 kHz (mp3/m4a/wav alike). Windows start at t = 0, hop, 2 hop, ...
(t + 5 s <= length); a recording shorter than 5 s is zero-padded to one window. --max-seconds reads only the start.

Output <out>/: part-NNNN.npz (PART_SIZE recordings each, written atomically, existing parts are skipped)
  recs (str, the part's recording IDs), ids (str, recording ID per window), window_idx, t_start,
  emb float32 (N, 1536), logits float16 (N, C: in classes.json order), lse float32 (N, log-sum-exp of all
  14,795 logits; full-vocabulary softmax = exp(logit - lse)), failed (str, recordings that failed to decode)
  classes.json: the species (eBird codes) and their Perch class index (-1 if none)
--softmax-full-out also writes the full-vocabulary softmax per part (training targets of the XCL source).

Usage (DATA_ROOT, DISTILL_ROOT: see common/config.py):
  python teacher/embed_perch.py --manifest $DATA_ROOT/wabad/test.parquet --id-col window_id --path-col path \
      --classes-json WABAD_classes.json --out $DATA_ROOT/wabad/emb_test --hop 5 --gpu --batch 128
  python teacher/embed_perch.py --manifest $DATA_ROOT/birdset/xcl_manifest.parquet --out $DISTILL_ROOT/emb_xcl60 \
      --max-seconds 60 --softmax-full-out $DISTILL_ROOT/softmax_full/emb_xcl60 --gpu --batch 128
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PAPER, PERCH_DIR, PERCH_GPU_DIR  # noqa: E402

SR = 32_000
WIN = 5 * SR
PART_SIZE = 500
SPECIES_LIST = PAPER / "data_prep/species/species_list.csv"


def decode(path: str, max_seconds: float, offset: float = 0.0) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(offset), "-i", path]
    if max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True)
    except (TypeError, OSError):  # missing or unreadable path
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    return np.frombuffer(res.stdout, dtype=np.float32)


def windows(audio: np.ndarray, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(starts in s, stacked windows)."""
    if len(audio) < WIN:
        audio = np.pad(audio, (0, WIN - len(audio)))
    hop = int(hop_s * SR)
    starts = np.arange(0, len(audio) - WIN + 1, hop)
    return starts / SR, np.stack([audio[s : s + WIN] for s in starts])


# eBird v2024 code -> Perch class (labels.csv name) where the eBird 2021 code / name rule picks a silent class.
# All four are 2024 splits whose Japanese taxon was an issf of an eBird 2021 species (the pre-split parent). Perch
# names classes by iNat 2024; the new species' class has no eBird code (iNat-only training) and stays silent on the
# species' own recordings. Evidence: hit = the class is Perch's top class (max over the 5-s windows of the first
# 20 s) on 20 focal training recordings of the species (Japanese first; JP = how many are Japanese); in
# parentheses: within Perch's top 5.
PERCH_OVERRIDES = {
    "bkfbun2": "Emberiza spodocephala",  # (blfbun1): E. personata 0/20 (0/20) -> 6/20 (15/20), 20 JP
    "categr2": "Ardea ibis",  # (categr, iNat name): A. coromanda 0/20 (0/20) -> 8/20 (13/20), 0 JP
    "rinphe2": "Phasianus colchicus",  # (rinphe): P. versicolor 0/20 (3/20) -> 10/20 (15/20), 20 JP
    "veggul1": "Larus argentatus",  # (hergul): L. vegae 0/20 (7/20) -> 9/20 (16/20), 7 JP
}


def perch_classes(codes: list[str], overrides: dict[str, str] = PERCH_OVERRIDES) -> list[dict]:
    """eBird code -> Perch logit index (PERCH_OVERRIDES, then the eBird 2021 code, then the scientific name).
    Scientific names come from the Japanese species list, else the eBird taxonomy v2024 (as for BirdNET)."""
    from common.scoring import sci_names  # imports this module, hence not at the top

    ebird = pd.read_csv(PERCH_DIR / "assets/perch_v2_ebird_classes.csv")["ebird2021"].tolist()
    names = pd.read_csv(PERCH_DIR / "assets/labels.csv").iloc[:, 0].tolist()
    sci = sci_names()
    out = []
    for c in codes:
        idx = names.index(overrides[c]) if c in overrides else ebird.index(c) if c in ebird else (
            names.index(sci[c]) if sci.get(c) in names else -1)
        out.append({"species": c, "perch_idx": idx})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, default=None,
                    help="manifest whose species define the class list (default: --manifest)")
    ap.add_argument("--classes-json", type=Path, default=None,
                    help="explicit class list (a classes.json); overrides --train-manifest")
    ap.add_argument("--offset-col", default=None,
                    help="column with a start offset in s; decodes --max-seconds from there")
    ap.add_argument("--id-col", default="rec_id")
    ap.add_argument("--path-col", default="file_path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seconds", type=float, default=0, help="0 = whole file")
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16, help="TF intra-op threads")
    ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--gpu", action="store_true",
                    help="use the Perch 2.0 GPU variant (same weights as the CPU variant up to numeric noise); "
                         "batch 128 recommended")
    ap.add_argument("--softmax-full-out", type=Path, default=None,
                    help="also write the full-vocabulary softmax per part (<dir>/part-NNNN.npy, (rows, 14795) f16)")
    args = ap.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    fn = tf.saved_model.load(str(PERCH_GPU_DIR if args.gpu else PERCH_DIR)).signatures["serving_default"]

    args.out.mkdir(parents=True, exist_ok=True)
    if args.classes_json:
        codes = [c["species"] for c in json.loads(args.classes_json.read_text())]
    else:
        codes = sorted(pd.read_parquet(args.train_manifest or args.manifest)["species"].unique())
    classes = perch_classes(codes)
    (args.out / "classes.json").write_text(json.dumps(classes, indent=1))
    cls_idx = np.array([c["perch_idx"] for c in classes])
    print(f"classes: {len(classes)}, unmapped to Perch: "
          f"{[c['species'] for c in classes if c['perch_idx'] < 0]}", flush=True)

    man = pd.read_parquet(args.manifest)
    ids, paths = man[args.id_col].astype(str).tolist(), man[args.path_col].tolist()
    offsets = man[args.offset_col].tolist() if args.offset_col else [0.0] * len(ids)
    n_parts = (len(ids) + PART_SIZE - 1) // PART_SIZE
    pool = ThreadPoolExecutor(args.decoders)
    t0, n_win = time.time(), 0

    def embed(batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nb = len(batch)
        x = np.stack(batch)
        if args.gpu and nb < args.batch:  # pad to the compiled shape (XLA recompiles per shape)
            x = np.pad(x, ((0, args.batch - nb), (0, 0)))
        o = fn(inputs=tf.constant(x))
        logits = o["label"].numpy()[:nb]
        sel = np.where(cls_idx[None, :] >= 0, logits[:, np.maximum(cls_idx, 0)], np.nan)
        return o["embedding"].numpy()[:nb], sel.astype(np.float16), logits

    if args.softmax_full_out:
        args.softmax_full_out.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        dst = args.out / f"part-{p:04d}.npz"
        sm_dst = args.softmax_full_out / f"part-{p:04d}.npy" if args.softmax_full_out else None
        lo, hi = p * PART_SIZE, min((p + 1) * PART_SIZE, len(ids))
        if dst.exists() and (sm_dst is None or sm_dst.exists()):
            assert np.load(dst)["recs"].tolist() == ids[lo:hi], f"{dst} does not match manifest"
            continue
        audio = pool.map(lambda a: decode(*a), [(p_, args.max_seconds, o) for p_, o in
                                                 zip(paths[lo:hi], offsets[lo:hi])])
        meta, wins, failed = [], [], []
        for rid, a in zip(ids[lo:hi], audio):
            if a is None:
                failed.append(rid)
                continue
            starts, w = windows(a, args.hop)
            meta += [(rid, i, s) for i, s in enumerate(starts)]
            wins += list(w)
        embs, logits, lses, probs = [], [], [], []
        for b in range(0, len(wins), args.batch):
            e, lg, full = embed(wins[b : b + args.batch])
            embs.append(e)
            logits.append(lg)
            top = full.max(1, keepdims=True)
            lses.append((top[:, 0] + np.log(np.exp(full - top).sum(1))).astype(np.float32))
            if sm_dst is not None:
                full -= full.max(1, keepdims=True)
                pr = np.exp(full)
                probs.append((pr / pr.sum(1, keepdims=True)).astype(np.float16))
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            recs=np.array(ids[lo:hi]),
            ids=np.array([m[0] for m in meta]),
            window_idx=np.array([m[1] for m in meta], dtype=np.int32),
            t_start=np.array([m[2] for m in meta], dtype=np.float32),
            emb=np.concatenate(embs) if embs else np.zeros((0, 1536), np.float32),
            logits=np.concatenate(logits) if logits else np.zeros((0, len(classes)), np.float16),
            lse=np.concatenate(lses) if lses else np.zeros(0, np.float32),
            failed=np.array(failed, dtype=str),
        )
        tmp.rename(dst)
        if sm_dst is not None:
            tmp2 = sm_dst.parent / f"{sm_dst.stem}.tmp{os.getpid()}.npy"
            np.save(tmp2, np.concatenate(probs) if probs else np.zeros((0, 14795), np.float16))
            tmp2.rename(sm_dst)
        n_win += len(wins)
        rate = n_win / (time.time() - t0)
        print(f"part {p + 1}/{n_parts}: {len(wins)} windows, {len(failed)} failed, "
              f"{rate:.1f} win/s, elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    (args.out / "DONE").write_text(json.dumps({"parts": n_parts, "recordings": len(ids)}))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
