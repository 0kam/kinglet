# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""fftfront_v1: parity and trunk timing of the FFT frontend (numpy rfft + power-input body ONNX).

parity (default): on the 2,048 ptqpub_v1 / mcuboard_v1 WABAD parity windows (wabad/test.parquet rows
  linspace(0, n-1, 2048), first 5 s of each file) compares, per tier,
    F  power-input body ONNX (DISTILL_ROOT/fftfront_v1/kinglet-<tier>.power.onnx, export_onnx.py --split
       --mel-impl power) fed by kinglet.power_spectrogram,
    R  the released v0.3.2 waveform trunk ONNX (conv-DFT inside the graph),
    T  the torch pipeline (evaluate.accuracy.score_student.Distilled: torchaudio mel + checkpoint model, CPU fp32),
  through the same exact Japan pack (kinglet build-pack exact fold of the released headpack; the torch feature
  goes through the same head, so only the frontend differs). F and R scores come from Kinglet.run.
  First the numpy power spectrogram is checked against torchaudio on random and real audio.
--diagnose: why the released conv-DFT trunk misses the embedding-cosine gate (float32 DFT angles in its weights):
  basis error of the released conv weights, log-mel difference on the worst window, and the trunk re-exported with a
  float64 basis (export_onnx.py, DISTILL_ROOT/fftfront_v1/kinglet-<tier>.convdft-f64basis.onnx) vs F.
--timing: trunk only on the speed kit's synthetic 10-min WAV (compute_speed/speed_kit/bench.wav, written by
  speed_kit.py; 120 windows, batches of 64): released conv-DFT
  trunk vs numpy FFT + body (and the FFT alone), median of --reps passes, per thread count.

Usage:
  python release/fftfront.py                     # -> results/fftfront_v1.json ["parity"]
  python release/fftfront.py --diagnose
  python release/fftfront.py --timing --threads 4 16
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_ROOT, PAPER, RELEASE_ASSETS, RESULTS  # noqa: E402
from common.config import DISTILL_ROOT as DD  # noqa: E402

OUT_DIR = DD / "fftfront_v1"
REL = RELEASE_ASSETS
KIT_WAV = PAPER / "compute_speed/speed_kit/bench.wav"  # the speed kit's synthetic 10-min WAV
RESULT = RESULTS / "fftfront_v1.json"
TIERS = {"b": "g3conf_v1/regnety_016_full", "s": "g3conf_v1/efficientnet_lite0_full",
         "xs": "g2xcl_v1/mobilenetv4_conv_small_full"}
N_PARITY = 2048
GATE = {"emb_cos_min": 0.9999, "top1_jp_agree": 0.999, "det_pair_disagree": 0.001}


def parity_windows() -> np.ndarray:
    """(2048, 160000) float32, cached in DISTILL_ROOT/fftfront_v1 (= the windows of mcu/export_int8.wabad_mels)."""
    cache = OUT_DIR / "parity_windows.npy"
    if cache.exists():
        return np.load(cache)
    import pandas as pd

    from teacher.embed_perch import decode, windows

    paths = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")["path"]
    paths = paths.iloc[np.linspace(0, len(paths) - 1, N_PARITY).astype(int)].tolist()
    with ThreadPoolExecutor(8) as ex:
        w = np.stack(list(ex.map(lambda p: windows(decode(p, 0), 5.0)[1][0], paths)))
    np.save(cache, w)
    return w


def exact_pack(tier: str) -> Path:
    """Exact Japan pack (kinglet build-pack default) of the released v0.3.2 compact pack's species: each
    species row is pinned to the headpack row it copies (vocab_index), dead rows stay dead."""
    out = OUT_DIR / f"pack-jp-exact-{tier}.npz"
    if out.exists():
        return out
    from kinglet.pack import build_pack, load_pack

    p = load_pack(REL / f"kinglet-{tier}-v0.3.2.pack-jp.npz")
    hp = np.load(REL / f"kinglet-{tier}-v0.3.2.headpack.npz")
    row = {r.tobytes(): i for i, r in enumerate(hp["W"].astype(np.float32))}
    n = p["n_species"]
    species = []
    for k in range(n):
        i = row.get(p["W"][k].astype(np.float32).tobytes())
        assert (i is None) == bool(p["b"][k] < -1e8), "pack row neither a headpack row nor dead"
        species.append({"ebird_code": str(p["codes"][k]), "japanese_name": str(p["japanese_name"][k]),
                        "scientific_name": str(p["scientific_name"][k]) if i is not None else "",
                        "vocab_index": "" if i is None else i})
    build_pack(REL / f"kinglet-{tier}-v0.3.2.headpack.npz", species, out, groups=p["groups"],
               group_thresholds=p["group_thresholds"])
    return out


def spectrogram_check(wins: np.ndarray) -> dict:
    import torch
    import torchaudio

    from kinglet import power_spectrogram

    spec = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=320, power=2.0)
    rng = np.random.default_rng(0)
    res = {}
    for name, x in (("random_normal", rng.normal(0, 0.1, (64, 160000)).astype(np.float32)),
                    ("random_uniform_fullscale", rng.uniform(-1, 1, (64, 160000)).astype(np.float32)),
                    ("wabad_parity", wins)):
        mx_abs, mx_rel, n = 0.0, 0.0, 0
        for i in range(0, len(x), 64):
            a = power_spectrogram(x[i : i + 64])
            r = spec(torch.from_numpy(x[i : i + 64])).numpy().transpose(0, 2, 1)
            assert a.shape == r.shape == (len(r), 501, 513) and a.dtype == np.float32
            d = np.abs(a - r)
            mx_abs = max(mx_abs, float(d.max()))
            mx_rel = max(mx_rel, float((d.max((1, 2)) / r.max((1, 2))).max()))  # relative to the window's peak
            n += len(r)
        res[name] = {"windows": n, "max_abs_diff": mx_abs, "max_diff_rel_to_window_peak": mx_rel}
    return res


def head_scores(h: np.ndarray, m) -> tuple[np.ndarray, np.ndarray]:
    """Trunk feature -> (Japan-list species probabilities as Kinglet.run, full-vocab top-1)."""
    lg = h @ m.W.T + m.b
    p = np.exp(lg - lg.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p[:, m.cols][:, : m.n_species], lg[:, :-1].argmax(1)  # last full-head row is the dead row


def compare(a: dict, b: dict) -> dict:
    ea, eb = a["emb"], b["emb"]
    cos = (ea * eb).sum(1) / np.linalg.norm(ea, axis=1) / np.linalg.norm(eb, axis=1)
    top1 = (a["p"].argmax(1) == b["p"].argmax(1)).mean()
    det = ((a["p"] >= 0.1) != (b["p"] >= 0.1)).mean()
    r = {"emb_cos_min": float(cos.min()), "emb_cos_mean": float(cos.mean()),
         "top1_jp_agree": float(top1), "top1_full_agree": float((a["top1_full"] == b["top1_full"]).mean()),
         "det_pair_disagree": float(det), "det_pairs_differing": int(((a["p"] >= 0.1) != (b["p"] >= 0.1)).sum()),
         "max_abs_diff": {k: float(np.abs(a[k] - b[k]).max()) for k in ("emb", "feature", "p")}}
    r["pass"] = bool(r["emb_cos_min"] >= GATE["emb_cos_min"] and r["top1_jp_agree"] >= GATE["top1_jp_agree"]
                     and r["det_pair_disagree"] <= GATE["det_pair_disagree"])
    return r


def parity(threads: int) -> dict:
    import torch

    from kinglet import Kinglet, power_spectrogram
    from evaluate.accuracy.score_student import Distilled

    torch.set_num_threads(threads)
    wins = parity_windows()
    res = {"windows": len(wins), "spectrogram_vs_torchaudio": spectrogram_check(wins), "gate": GATE, "tiers": {}}
    print(json.dumps(res["spectrogram_vs_torchaudio"], indent=1), flush=True)
    for tier, ck in TIERS.items():
        pack = exact_pack(tier)
        paths = {"F": OUT_DIR / f"kinglet-{tier}.power.onnx", "R": REL / f"kinglet-{tier}-v0.3.2.onnx"}
        out = {}
        for k, path in paths.items():
            m = Kinglet(path, pack, threads=threads)
            emb, p, _ = m.run(wins, normalize=False)  # raw, as the parity was measured (kinglet >= 0.4 normalises)
            feats = [m.sess.run(["feature"], {m.input: power_spectrogram(x) if m.input == "power" else x})[0]
                     for x in np.array_split(wins, len(wins) // 64)]
            h = np.concatenate(feats)
            p2, top1 = head_scores(h, m)
            assert np.abs(p - p2).max() < 1e-6, "head_scores differs from Kinglet.run"
            out[k] = {"emb": emb, "feature": h, "p": p, "top1_full": top1}
        d = Distilled(DD / ck / "best.pt", "cpu")
        embs, feats = [], []
        with torch.no_grad():
            for i in range(0, len(wins), 64):
                x = torch.from_numpy(wins[i : i + 64])
                mel = ((torch.log(d.mel(x) + 1e-6) - d.cfg["mel_mu"]) / d.cfg["mel_sd"]).unsqueeze(1)
                h, e, _ = d.model(mel)
                embs.append(e.numpy())
                feats.append(h.numpy())
        h = np.concatenate(feats)
        p, top1 = head_scores(h, m)
        out["T"] = {"emb": np.concatenate(embs), "feature": h, "p": p, "top1_full": top1}
        res["tiers"][tier] = {"ckpt": ck, "F_vs_R": compare(out["F"], out["R"]), "F_vs_T": compare(out["F"], out["T"]),
                              "R_vs_T (reference floor)": compare(out["R"], out["T"])}
        print(tier, json.dumps(res["tiers"][tier], indent=1), flush=True)
    res["all_pass"] = all(t[c]["pass"] for t in res["tiers"].values() for c in ("F_vs_R", "F_vs_T"))
    return res


def embeddings(sess, wins: np.ndarray) -> np.ndarray:
    from kinglet import power_spectrogram

    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(["embedding"], {name: power_spectrogram(x) if name == "power" else x})[0]
                           for x in np.array_split(wins, len(wins) // 64)])


def diagnose(threads: int) -> dict:
    import onnx
    import onnxruntime as ort
    import torchaudio
    from onnx import numpy_helper

    from kinglet import HANN, power_spectrogram

    wins = parity_windows()
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    k64 = 2 * np.pi * np.arange(1024)[None, :] * np.arange(513)[:, None] / 1024
    fb = torchaudio.functional.melscale_fbanks(513, 0.0, 16000, 128, 32000, norm=None, mel_scale="htk").numpy()
    res = {}
    for tier in TIERS:
        path = REL / f"kinglet-{tier}-v0.3.2.onnx"
        dft = next(numpy_helper.to_array(i) for i in onnx.load(str(path)).graph.initializer
                   if list(i.dims) == [1026, 1, 1024])[:, 0].astype(np.float64)
        sess = {k: ort.InferenceSession(str(p), opt, providers=["CPUExecutionProvider"]) for k, p in
                (("F", OUT_DIR / f"kinglet-{tier}.power.onnx"), ("R", path),
                 ("R_f64basis", OUT_DIR / f"kinglet-{tier}.convdft-f64basis.onnx"))}
        e = {k: embeddings(s, wins) for k, s in sess.items()}
        cos = {k: (e[k] * e["F"]).sum(1) / np.linalg.norm(e[k], axis=1) / np.linalg.norm(e["F"], axis=1)
               for k in ("R", "R_f64basis")}
        w = int(cos["R"].argmin())
        x = wins[w].astype(np.float64)
        frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, 512, mode="reflect"), 1024)[::320]
        z = frames @ dft.T  # released conv weights, float64 arithmetic: isolates the weights from ORT's arithmetic
        mel_r = (z[:, :513] ** 2 + z[:, 513:] ** 2) @ fb
        mel_f = power_spectrogram(wins[w : w + 1])[0].astype(np.float64) @ fb
        dl = np.abs(np.log(mel_r + 1e-6) - np.log(mel_f + 1e-6))
        res[tier] = {"released_dft_weight_max_abs_error_vs_float64_basis":
                     float(np.abs(dft - np.concatenate([np.cos(k64), -np.sin(k64)]) * HANN).max()),
                     "worst_window": w, "worst_window_rms": float(np.sqrt((x ** 2).mean())),
                     "worst_window_cos_R_vs_F": float(cos["R"][w]),
                     "worst_window_logmel_max_abs_diff_R_weights_vs_F": float(dl.max()),
                     "worst_window_logmel_cells_diff_gt_0.01": float((dl > 0.01).mean()),
                     "worst_window_min_mel_power_F": float(mel_f.min()),
                     "windows_cos_lt_0.9999": {k: int((c < 0.9999).sum()) for k, c in cos.items()},
                     "cos_min_vs_F": {k: float(c.min()) for k, c in cos.items()}}
        print(tier, res[tier], flush=True)
    return res


def timing(threads: list[int], reps: int) -> dict:
    import onnxruntime as ort

    from kinglet import power_spectrogram, windows

    import wave

    with wave.open(str(KIT_WAV)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    wins = windows(a)[1]
    res = {"audio_s": len(a) / 32000, "windows": len(wins), "batch": 64, "reps": reps,
           "onnxruntime": ort.__version__, "numpy": np.__version__, "load_avg_1min_before": os.getloadavg()[0], "runs": {}}

    def med(fn):
        fn()
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            walls.append(time.perf_counter() - t0)
        return round(statistics.median(walls), 4)

    for th in threads:
        opt = ort.SessionOptions()
        opt.intra_op_num_threads, opt.inter_op_num_threads = th, 1
        for tier in TIERS:
            conv = ort.InferenceSession(str(REL / f"kinglet-{tier}-v0.3.2.onnx"), opt, providers=["CPUExecutionProvider"])
            body = ort.InferenceSession(str(OUT_DIR / f"kinglet-{tier}.power.onnx"), opt,
                                        providers=["CPUExecutionProvider"])
            batches = [wins[i : i + 64] for i in range(0, len(wins), 64)]
            r = {"convdft_trunk_s": med(lambda: [conv.run(None, {"waveform": b}) for b in batches]),
                 "fft_plus_body_s": med(lambda: [body.run(None, {"power": power_spectrogram(b)}) for b in batches]),
                 "fft_only_s": med(lambda: [power_spectrogram(b) for b in batches])}
            r["speedup"] = round(r["convdft_trunk_s"] / r["fft_plus_body_s"], 3)
            res["runs"][f"{tier}_t{th}"] = r
            print(tier, th, r, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--threads", type=int, nargs="+", default=[8])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    if args.timing:
        res["trunk_timing"] = timing(args.threads, args.reps)
    elif args.diagnose:
        res["conv_dft_diagnosis"] = diagnose(args.threads[0])
    else:
        res["parity"] = parity(args.threads[0])
    RESULT.write_text(json.dumps(res, indent=1))
    print(f"wrote {RESULT}")


if __name__ == "__main__":
    main()
