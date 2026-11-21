# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Bug check for the Kinglet rows of speed_kit.py: are the float32 head logits finite and equal to a float64 recomputation?

numpy on macOS (Accelerate) can print "divide by zero / overflow / invalid value encountered in matmul" at the head
matmul. This runs the same trunk + head on 64 synthetic windows, counts the warnings, and compares the float32 result
with float64. Usage, in the kit folder: .venv/bin/python check_head.py [models_dir] > results/check_head_<label>.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

import speed_kit as k
from kinglet import Kinglet

models = Path(sys.argv[1]) if len(sys.argv) > 1 else k.HERE / "models"
wins = k.segments(np.random.default_rng(0).normal(0, 0.05, k.SR * 320).astype(np.float32), 5.0)  # 64 windows
power = k.power_spectrogram(wins)
out = {"numpy": np.__version__, "blas": np.show_config(mode="dicts")["Build Dependencies"]["blas"].get("name")}
for tier in ("b", "s", "xs"):
    m = Kinglet(models / f"kinglet-{tier}-{k.POWER}.power.onnx", models / f"kinglet-{tier}-v0.3.2.pack-jp.npz", threads=4)
    _, h = m.sess.run(None, {"power": power})
    for head, (W, b) in {"exact": k.exact_head(models / f"kinglet-{tier}-v0.3.2.headpack.npz", m)[:2],
                         "pack": (m.W, m.b)}.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lg32 = h @ W.T + b
        lg64 = h.astype(np.float64) @ W.T.astype(np.float64) + b.astype(np.float64)
        live = b > -1e8  # rows with a -1e9 bias are dead (unmatched species) and never win
        out[f"{tier}_{head}"] = {"warnings": len(caught), "finite": bool(np.isfinite(lg32).all()),
                                 "max_abs_diff_live": float(np.abs(lg32 - lg64)[:, live].max()),
                                 "max_abs_logit_live": float(np.abs(lg64[:, live]).max())}
print(json.dumps(out, indent=1))
