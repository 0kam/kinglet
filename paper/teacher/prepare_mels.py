# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Precompute student inputs aligned row by row with the teacher's output parts.

For each part-NNNN.npz written by teacher/embed_perch.py, decode the same recordings with the same decode/windows
code and store the same 5-s windows (rows in the teacher's order; recordings the teacher failed to decode are
skipped). A part is skipped if its output exists (resumable). Run after the teacher part exists.

  --store mel (default)  part-NNNN.npy      log-mels, float16 (n_windows, 128, 501)
  --store audio          part-NNNN.raw.npy  int16 waveform windows (n_windows, 160000), the input of
                                            teacher/make_softmax_targets.py and of training-time front ends

Mel: torchaudio MelSpectrogram(32 kHz, n_fft 1024, hop 320, n_mels 128) = train.backbones.MEL, log(x + 1e-6).

Usage (DATA_ROOT, DISTILL_ROOT: see common/config.py):
  python teacher/prepare_mels.py --emb-dir $DISTILL_ROOT/emb_train60 --manifest $DATA_ROOT/manifest.parquet
  python teacher/prepare_mels.py --emb-dir $DISTILL_ROOT/emb_xcm60 --manifest $DATA_ROOT/birdset/train_manifest.parquet
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher.embed_perch import PART_SIZE, SR, decode, windows  # noqa: E402
from train.backbones import MEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emb-dir", type=Path, required=True, help="teacher output dir (part-NNNN.npz)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--id-col", default="rec_id")
    ap.add_argument("--path-col", default="file_path")
    ap.add_argument("--max-seconds", type=float, default=60)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--decoders", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--store", choices=["mel", "audio"], default="mel")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, **MEL)
    man = pd.read_parquet(args.manifest)
    ids, paths = man[args.id_col].astype(str).tolist(), man[args.path_col].tolist()
    pool = ThreadPoolExecutor(args.decoders)

    for npz_path in sorted(args.emb_dir.glob("part-*.npz")):
        dst = npz_path.with_suffix(".raw.npy" if args.store == "audio" else ".npy")
        if dst.exists():
            continue
        z = np.load(npz_path)
        p = int(npz_path.stem.split("-")[1])
        lo = p * PART_SIZE
        assert z["recs"].tolist() == ids[lo : lo + len(z["recs"])], f"{npz_path} does not match manifest"
        failed = set(z["failed"].tolist())
        audio = pool.map(lambda pth: decode(pth, args.max_seconds), paths[lo : lo + len(z["recs"])])
        meta, mels = [], []
        for rid, a in zip(z["recs"].tolist(), audio):
            if rid in failed:  # the teacher failed this recording; keep the rows aligned by skipping it
                continue
            assert a is not None, f"{rid}: decoded for the teacher but not now"
            starts, w = windows(a, args.hop)
            meta += [(rid, i) for i in range(len(starts))]
            if args.store == "audio":
                mels.append(np.clip(w * 32767, -32768, 32767).astype(np.int16))
                continue
            with torch.no_grad():
                m = torch.log(mel(torch.from_numpy(w)) + 1e-6)
            mels.append(m.numpy().astype(np.float16))
        assert [m[0] for m in meta] == z["ids"].tolist() and \
               [m[1] for m in meta] == z["window_idx"].tolist(), f"{npz_path}: window order differs"
        shape = (0, 160000) if args.store == "audio" else (0, MEL["n_mels"], 501)
        out = np.concatenate(mels) if mels else np.zeros(shape, np.int16 if args.store == "audio" else np.float16)
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, out)
        tmp.rename(dst)
        print(f"{dst.name}: {out.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
