# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""mcurate_v1b: what the MCU's 31,250 Hz capture costs when the model reads it as 32 kHz.

The Grove Vision AI V2 PDM path delivers 31,250 samples/s (measured at boot, mcu/BUILD.md) and the
firmware feeds them to the 32 kHz model unchanged: every sound is 2.4% higher and faster, and a
160,000-sample window spans 5.12 s. Emulated on the host for the XS checkpoint on WABAD (wabad_v1
protocol: 5-s windows, the 1,189-species list, exact fold of the full-vocabulary softmax).

WABAD windows are contiguous 5-s cuts of each recording, so the MCU window is real audio:
window k + the first 0.12 s of window k+1 (or the last 0.12 s of window k-1 in front when there
is no successor), 163,840 samples at 32 kHz, resampled to 31,250 Hz = exactly 160,000 samples.
The control is the cached wabad_v1 score of the plain window.

  python mcu/mcurate.py    # -> results/mcurate_v1b.json (+ score cache mcurate_b_rate_WABAD.npz;
                           #    delete it to recompute the embedding cosine)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_ROOT, DISTILL_ROOT, RESULTS  # noqa: E402
from common.scoring import BIRDNET_OVERRIDES, SETS, evaluate_set, infer, set_codes  # noqa: E402
from evaluate.accuracy.score_student import Distilled  # noqa: E402
from teacher.embed_perch import decode, windows  # noqa: E402

CKPT = DISTILL_ROOT / "g2xcl_v1/mobilenetv4_conv_small_full/best.pt"
BASELINE = "distill_g2r_full"  # wabad_v1 XS score cache
SR, MCU_SR, WIN = 32_000, 31_250, 160_000
EXTRA = WIN * SR // MCU_SR - WIN  # 3,840 samples: the 0.12 s a 160,000-sample MCU window adds


def neighbour(path: str, d: int) -> Path:
    stem, k = path[:-4].rsplit("_", 1)
    return Path(f"{stem}_{int(k) + d:0{len(k)}d}.wav")


def extended(x: np.ndarray, path: str) -> tuple[np.ndarray, str]:
    """5.12 s of contiguous audio holding the window: (WIN + EXTRA samples, which neighbour)."""
    fit = lambda a: np.pad(a, (0, EXTRA))[:EXTRA]  # noqa: E731  (ffmpeg may be a sample short)
    if neighbour(path, 1).exists():
        return np.concatenate([x, fit(decode(str(neighbour(path, 1)), EXTRA / SR))]), "next"
    if neighbour(path, -1).exists():
        return np.concatenate([fit(decode(str(neighbour(path, -1)), 0, 5.0 - EXTRA / SR)), x]), "prev"
    return np.concatenate([x, x[: -EXTRA - 1 : -1]]), "mirror"


class McuRate:
    """The student fed MCU-rate audio. infer() hands over the windows in manifest order."""

    tag = Distilled.tag

    def __init__(self, model: Distilled, paths: list[str], resample: bool = True):
        self.m, self.sci, self.paths, self.resample = model, model.sci, paths, resample
        self.i, self.cos, self.case, self.pool = 0, [], [], ThreadPoolExecutor(8)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ext, case = zip(*self.pool.map(extended, x, self.paths[self.i : self.i + len(x)]))
        self.i += len(x)
        self.case += case
        ext = np.stack(ext)
        if not self.resample:  # bug check: cut back to the window's own 5 s
            return self.m(np.stack([e[:WIN] if c != "prev" else e[EXTRA:] for e, c in zip(ext, case)]))
        self.m(x)
        e_plain = self.m.emb
        r = torchaudio.functional.resample(torch.from_numpy(ext), SR, MCU_SR).numpy()
        assert r.shape[1] == WIN
        out = self.m(r)
        self.cos.append(torch.cosine_similarity(e_plain, self.m.emb).cpu().numpy())
        return out


def main():
    BIRDNET_OVERRIDES.setdefault(Distilled.tag, {})
    model = Distilled(CKPT, "cuda" if torch.cuda.is_available() else "cpu")
    codes = set_codes("WABAD")
    model.set_region(codes)
    base = infer(model, BASELINE, "WABAD", codes, 8)  # cached
    res = {"baseline": evaluate_set("WABAD", base, BASELINE)}
    man = pd.read_parquet(DATA_ROOT / SETS["WABAD"][0])
    paths = man["path"].tolist()
    assert base["ids"].tolist() == man["window_id"].tolist()

    # bug check: the extended window cut back to its own 5 s reproduces the cached scores
    # (first 256 windows + the first 64 that take their extra audio from the previous window)
    prev = [i for i, p in enumerate(paths) if not neighbour(p, 1).exists()][:64]
    for name, rows in (("first256", list(range(256))), ("prev64", prev)):
        x = np.stack([windows(decode(paths[i], 0, 0.0), 5.0)[1][0] for i in rows])
        chk = McuRate(model, [paths[i] for i in rows], resample=False)
        dp = float(np.abs(np.concatenate([chk(x[i : i + 32]) for i in range(0, len(x), 32)]) - base["scores"][rows]).max())
        assert dp < 0.005, (name, dp)
        res[f"cutback_max_dp_{name}"] = dp

    arm = McuRate(model, paths)
    res["rate"] = evaluate_set("WABAD", infer(arm, "mcurate_b_rate", "WABAD", codes, 8), "mcurate_b_rate")
    if arm.cos:
        cos = np.concatenate(arm.cos)
        assert arm.i == len(paths) == len(cos)
        res["emb_cos_rate_vs_plain"] = {"mean": float(cos.mean()), "median": float(np.median(cos)),
                                        "p10": float(np.percentile(cos, 10)), "min": float(cos.min())}
        res["neighbour_share"] = pd.Series(arm.case).value_counts(normalize=True).round(4).to_dict()
    for k in ("baseline", "rate"):
        res[k] = {m: res[k][m] for m in ("n_windows", "cmAP", "pooled_auprc", "auroc", "n_species")}
    res["cmAP_rate_over_baseline"] = res["rate"]["cmAP"] / res["baseline"]["cmAP"]
    print(json.dumps(res, indent=1))
    (RESULTS / "mcurate_v1b.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
