# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Export the build-pack bundle (headpack) for a full-vocabulary Kinglet checkpoint.

The split deployment is trunk ONNX (release/export_onnx.py --split) + a regional output layer
("pack"). `kinglet build-pack` folds the full head onto any species list; the ridge fit of the
pack's "other" row needs (a) the full head and (b) trunk features of calibration audio. This
script bundles both, so pack building needs no access to the training audio:

  <ckpt_dir>/headpack_focal.npz
    W (14795, d) f16, b (14795,) f32   effective linear head over the whole Perch vocabulary
    labels (14795,) str                Perch 2.0 label names (scientific names + FSD50K classes)
    calib_h (N, d) f16                 trunk features of focal validation windows, subsampled
    calib_is_holdout (N,) bool         recording-disjoint ~20% held out for the fit-quality report
    meta json                          ckpt path, backbone, counts, calib source

This is the calibration of the released output layers (packfit_v1); the release files
kinglet-<tier>-v0.3.2.headpack.npz are copies of these bundles.

Usage:
  python evaluate/species_lists/export_headpack.py --ckpt DISTILL_ROOT/g3conf_v1/regnety_016_full/best.pt [more ckpts]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import PERCH_DIR  # noqa: E402
from train.train_distill import SOURCES, Student, load_rows, parts_index  # noqa: E402

FIT_CAP = 24_000
HOLDOUT_CAP = 8_000


def effective_head(model: Student) -> tuple[np.ndarray, np.ndarray]:
    """The class head as one linear map: weights (C, d) and biases (C,)."""
    lh = model.logit_head
    return lh.weight.detach().numpy(), lh.bias.detach().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = pd.read_csv(PERCH_DIR / "assets/labels.csv").iloc[:, 0].to_numpy(dtype=np.str_)

    jobs = {}
    for p in args.ckpt:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        assert cfg.get("softmax_full"), f"{p}: not a full-vocabulary checkpoint"
        model = Student(cfg["backbone"], cfg["n_out"], head=cfg.get("head", "full"))
        model.load_state_dict(ck["model"])
        W, b = effective_head(model)
        assert W.shape[0] == len(labels), (W.shape, len(labels))
        jobs[p] = {"model": model.eval().to(dev), "cfg": cfg, "W": W, "b": b,
                   "fit": [], "ho": []}

    def consume(mel: np.ndarray, which: str, stride: int):
        mel = mel[::stride]
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for i in range(0, len(mel), 256):
                x0 = torch.from_numpy(np.ascontiguousarray(mel[i : i + 256])).float().to(dev)
                for j in jobs.values():
                    x = ((x0 - j["cfg"]["mel_mu"]) / j["cfg"]["mel_sd"]).unsqueeze(1)
                    j[which].append(j["model"](x)[0].float().cpu().numpy().astype(np.float16))

    # focal val windows: a recording-disjoint 20% of the val recordings -> holdout pool
    for part in parts_index(0, SOURCES):
        if part["val"].any():
            mel, _, _, ids = load_rows(part, "val")
            ho = np.array([zlib.crc32(str(i).encode()) % 5 == 0 for i in ids.tolist()])
            consume(mel[~ho], "fit", stride=8)
            consume(mel[ho], "ho", stride=8)

    rng = np.random.default_rng(0)
    for p, j in jobs.items():
        fit, ho = np.concatenate(j["fit"]), np.concatenate(j["ho"])
        fit = fit[rng.choice(len(fit), min(FIT_CAP, len(fit)), replace=False)]
        ho = ho[rng.choice(len(ho), min(HOLDOUT_CAP, len(ho)), replace=False)]
        h = np.concatenate([fit, ho])
        is_ho = np.zeros(len(h), bool)
        is_ho[len(fit):] = True
        dst = p.parent / "headpack_focal.npz"
        np.savez(dst, W=j["W"].astype(np.float16), b=j["b"].astype(np.float32), labels=labels,
                 calib_h=h, calib_is_holdout=is_ho,
                 meta=json.dumps({"ckpt": str(p), "backbone": j["cfg"]["backbone"],
                                  "d": int(j["W"].shape[1]), "n_fit": len(fit), "n_holdout": len(ho),
                                  "calib": "focal"}))
        print(f"{dst}: W {j['W'].shape} f16, calib {len(fit)}+{len(ho)}, "
              f"{dst.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
