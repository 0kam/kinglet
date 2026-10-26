"""kinglet analyze: batch-process audio files on a laptop CPU.

Writes one detections CSV (file, start_s, end_s, ebird_code, scientific_name, japanese_name, score)
with every window x species whose sigmoid score >= --min-score, and optionally one .npz of
embeddings per audio file (ids, t_start, embedding) for downstream probes / similarity search.

Usage:
  kinglet analyze recordings/ --model kinglet-s.onnx --out detections.csv
  kinglet analyze a.wav b.flac --model kinglet-xs.onnx --min-score 0.5 --embeddings emb/
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from . import Kinglet, decode, windows

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aac", ".opus"}


def gather(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTS) if p.is_dir() else [p]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="kinglet", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze", help="detect species / extract embeddings from audio files")
    an.add_argument("audio", nargs="+", type=Path, help="audio files and/or directories")
    an.add_argument("--model", type=Path, required=True, help="Kinglet .onnx (classes.csv alongside)")
    an.add_argument("--out", type=Path, default=Path("kinglet_detections.csv"))
    an.add_argument("--top-k", type=int, default=3, help="keep the k best species per window")
    an.add_argument("--min-score", type=float, default=0.0,
                    help="additionally drop rows below this sigmoid score (uncalibrated; ranks are "
                         "meaningful, absolute values are not yet)")
    an.add_argument("--hop", type=float, default=5.0, help="window hop in seconds")
    an.add_argument("--threads", type=int, default=4)
    an.add_argument("--embeddings", type=Path, default=None, help="directory for per-file .npz embeddings")
    args = ap.parse_args()

    model = Kinglet(args.model, threads=args.threads)
    files = gather(args.audio)
    if not files:
        sys.exit("no audio files found")
    if args.embeddings:
        args.embeddings.mkdir(parents=True, exist_ok=True)
    t0, audio_s, n_det = time.time(), 0.0, 0
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "start_s", "end_s", "ebird_code", "scientific_name", "japanese_name", "score"])
        for k, path in enumerate(files, 1):
            a = decode(path)
            audio_s += len(a) / 32_000
            starts, wins = windows(a, args.hop)
            emb, scores = model.run(wins)
            top = np.argsort(-scores, axis=1)[:, : args.top_k]
            for i, row in enumerate(top):
                for j in row:
                    if scores[i, j] < args.min_score:
                        break
                    code, sci, ja = model.classes[j]
                    w.writerow([path.name, f"{starts[i]:.1f}", f"{starts[i] + 5.0:.1f}", code, sci,
                                ja, f"{scores[i, j]:.3f}"])
                    n_det += 1
            if args.embeddings:
                np.savez(args.embeddings / (path.stem + ".npz"), t_start=starts,
                         embedding=emb.astype(np.float16))
            el = time.time() - t0
            print(f"[{k}/{len(files)}] {path.name}: {len(wins)} windows | total {audio_s / 3600:.2f} h "
                  f"in {el:.0f} s ({audio_s / el:.0f}x realtime)", flush=True)
    print(f"{n_det} rows (top {args.top_k}/window, score >= {args.min_score}) -> {args.out}")


if __name__ == "__main__":
    main()
