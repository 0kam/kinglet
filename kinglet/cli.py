"""kinglet CLI: analyze audio on a laptop CPU; build regional head packs.

kinglet analyze
  Writes one detections CSV (file, start_s, end_s, label, scientific_name, japanese_name, score)
  with every window x species whose score >= --min-score, plus one row per sound group
  (human_voice, vehicle, ...) whose score passes that group's own operating threshold
  (groups without a calibrated threshold are skipped; see MODEL_CARD.md). Optionally one .npz
  of embeddings per audio file (t_start, embedding) for downstream probes / similarity search.

kinglet build-pack
  Folds the full-vocabulary head (the .headpack.npz release asset) onto your species list:
  a CSV with columns ebird_code, scientific_name (+ optional japanese_name, vocab_index).

Usage:
  kinglet analyze recordings/ --model kinglet-s.onnx --pack kinglet-s.pack-jp.npz --out det.csv
  kinglet build-pack --headpack kinglet-s.headpack.npz --species my_region.csv --out pack_my.npz
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import Kinglet, decode, windows
from .pack import build_pack, read_species_csv

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aac", ".opus"}


def gather(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTS) if p.is_dir() else [p]
    return out


def analyze(args) -> None:
    model = Kinglet(args.model, args.pack, threads=args.threads)
    files = gather(args.audio)
    if not files:
        sys.exit("no audio files found")
    if args.embeddings:
        args.embeddings.mkdir(parents=True, exist_ok=True)
    live_groups = {g: t for g, t in model.groups.items() if t is not None}
    t0, audio_s, n_det = time.time(), 0.0, 0
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "start_s", "end_s", "label", "scientific_name", "japanese_name", "score"])
        for k, path in enumerate(files, 1):
            a = decode(path)
            audio_s += len(a) / 32_000
            starts, wins = windows(a, args.hop)
            emb, scores, gscores = model.run(wins)
            top = np.argsort(-scores, axis=1)[:, : args.top_k]
            for i, row in enumerate(top):
                for j in row:
                    if scores[i, j] < args.min_score:
                        break
                    code, sci, ja = model.classes[j]
                    w.writerow([path.name, f"{starts[i]:.1f}", f"{starts[i] + 5.0:.1f}", code, sci,
                                ja, f"{scores[i, j]:.3f}"])
                    n_det += 1
                for g, thr in live_groups.items():
                    if gscores[g][i] >= thr:
                        w.writerow([path.name, f"{starts[i]:.1f}", f"{starts[i] + 5.0:.1f}", g, "",
                                    "", f"{gscores[g][i]:.4f}"])
                        n_det += 1
            if args.embeddings:
                np.savez(args.embeddings / (path.stem + ".npz"), t_start=starts,
                         embedding=emb.astype(np.float16))
            el = time.time() - t0
            print(f"[{k}/{len(files)}] {path.name}: {len(wins)} windows | total {audio_s / 3600:.2f} h "
                  f"in {el:.0f} s ({audio_s / el:.0f}x realtime)", flush=True)
    print(f"{n_det} rows (species top {args.top_k}/window >= {args.min_score}; groups: "
          f"{', '.join(f'{g} >= {t}' for g, t in live_groups.items()) or 'none'}) -> {args.out}")


def build(args) -> None:
    report = build_pack(args.headpack, read_species_csv(args.species), args.out)
    print(json.dumps(report, indent=1))
    if report["unmatched"]:
        print(f"WARNING: {len(report['unmatched'])} species not in the Perch vocabulary "
              f"(dead rows, never detected): {', '.join(report['unmatched'])}")
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="kinglet", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyze", help="detect species / sound groups, extract embeddings")
    an.add_argument("audio", nargs="+", type=Path, help="audio files and/or directories")
    an.add_argument("--model", type=Path, required=True, help="Kinglet trunk .onnx")
    an.add_argument("--pack", type=Path, required=True, help="regional head pack .npz")
    an.add_argument("--out", type=Path, default=Path("kinglet_detections.csv"))
    an.add_argument("--top-k", type=int, default=3, help="keep at most the k best species per window")
    an.add_argument("--min-score", type=float, default=0.1,
                    help="drop species rows below this softmax probability; 0.1 is the operating "
                         "point used by the Perch ecosystem and in the Kinglet evaluations")
    an.add_argument("--hop", type=float, default=5.0, help="window hop in seconds")
    an.add_argument("--threads", type=int, default=4)
    an.add_argument("--embeddings", type=Path, default=None, help="directory for per-file .npz embeddings")
    an.set_defaults(fn=analyze)

    bp = sub.add_parser("build-pack", help="fold the full-vocabulary head onto a species list")
    bp.add_argument("--headpack", type=Path, required=True, help=".headpack.npz release asset")
    bp.add_argument("--species", type=Path, required=True,
                    help="CSV: ebird_code, scientific_name (+ japanese_name, vocab_index)")
    bp.add_argument("--out", type=Path, required=True, help="output pack .npz")
    bp.set_defaults(fn=build)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
