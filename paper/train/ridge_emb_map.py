# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Post-hoc linear map from a student's trunk feature to the Perch embedding space.

Needed for the class-loss-only ablation (head clsonly: its emb_head was never trained), and fitted
the same way for the other students as controls (loss ablation xsabl_v1, Appendix S1). One ridge regression per student (closed form,
alpha = 1, with intercept) from the trunk feature h (1280-d for XS) to the L2-normalised Perch 2.0
embedding, fitted on focal VALIDATION windows of the distillation data (manifest split "val";
never a benchmark): the recording-disjoint 80% of the val recordings (crc32(rec_id) % 5 != 0, the
split of evaluate/species_lists/export_headpack.py --focal-only), every 8th window. Fit quality is reported on the held-out
20% (same stride) as mean cos(mapped, teacher), next to the trained emb_head's cos on the same
windows. The map is then applied to the WABAD trunk features saved by evaluate/embeddings/embed_student.py --feature.

Output per student: <out>/part-0000.npz (ids, window_idx, t_start, emb f16), fit.json, DONE.

Usage:
  python ridge_emb_map.py --ckpt A/best.pt B/best.pt --feat <distill_root>/emb_A <distill_root>/emb_B \
      --out <distill_root>/emb_A_ridge <distill_root>/emb_B_ridge [--parts 2]   # --parts: smoke test
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_distill import SOURCES, Student, load_rows, parts_index  # noqa: E402

ALPHA = 1.0
STRIDE = 8


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (Y - my))
    return W.astype(np.float32), (my - mx @ W).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--feat", type=Path, nargs="+", required=True,
                    help="embed_student.py --feature output on WABAD, one per --ckpt")
    ap.add_argument("--out", type=Path, nargs="+", required=True)
    ap.add_argument("--parts", type=int, default=0, help="cap parts per source (smoke test)")
    args = ap.parse_args()
    assert len(args.ckpt) == len(args.feat) == len(args.out)
    if all((o / "DONE").exists() for o in args.out):
        print("already done")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    jobs = []
    for p in args.ckpt:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = Student(cfg["backbone"], cfg["n_out"], head=cfg.get("head", "full"))
        model.load_state_dict(ck["model"])
        jobs.append({"model": model.eval().to(dev), "cfg": cfg, "fit": [], "ho": [], "ho_head": []})

    teacher = {"fit": [], "ho": []}
    for part in parts_index(args.parts, SOURCES):
        if not part["val"].any():
            continue
        mel, emb, _, ids = load_rows(part, "val")
        ho = np.array([zlib.crc32(str(i).encode()) % 5 == 0 for i in ids.tolist()])
        for which, m in (("fit", ~ho), ("ho", ho)):
            mm = mel[m][::STRIDE]
            teacher[which].append(unit(emb[m][::STRIDE].astype(np.float32)))
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                for i in range(0, len(mm), 256):
                    x0 = torch.from_numpy(np.ascontiguousarray(mm[i : i + 256])).float().to(dev)
                    for j in jobs:
                        h, pe, _ = j["model"](((x0 - j["cfg"]["mel_mu"]) / j["cfg"]["mel_sd"]).unsqueeze(1))
                        j[which].append(h.float().cpu().numpy())
                        if which == "ho":
                            j["ho_head"].append(pe.float().cpu().numpy())
    t_fit, t_ho = np.concatenate(teacher["fit"]), np.concatenate(teacher["ho"])
    print(f"val windows: fit {len(t_fit)}, holdout {len(t_ho)}", flush=True)

    for p, f, o, j in zip(args.ckpt, args.feat, args.out, jobs):
        h_fit, h_ho = np.concatenate(j["fit"]), np.concatenate(j["ho"])
        W, b = ridge(h_fit, t_fit, ALPHA)
        stats = {"ckpt": str(p), "head": j["cfg"].get("head"), "alpha": ALPHA, "stride": STRIDE,
                 "n_fit": len(h_fit), "n_holdout": len(h_ho),
                 "fit_cos_ridge": float((unit(h_fit @ W + b) * t_fit).sum(1).mean()),
                 "holdout_cos_ridge": float((unit(h_ho @ W + b) * t_ho).sum(1).mean()),
                 "holdout_cos_trained_head": float((unit(np.concatenate(j["ho_head"])) * t_ho).sum(1).mean())}
        z = np.load(f / "part-0000.npz")
        o.mkdir(parents=True, exist_ok=True)
        np.savez(o / "part-0000.npz", ids=z["ids"], window_idx=z["window_idx"], t_start=z["t_start"],
                 emb=(z["feat"].astype(np.float32) @ W + b).astype(np.float16))
        (o / "fit.json").write_text(json.dumps(stats, indent=1))
        (o / "DONE").touch()
        print(o, stats, flush=True)


if __name__ == "__main__":
    main()
