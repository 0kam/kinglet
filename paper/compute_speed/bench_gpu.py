# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""gpuspeed_v1: GPU throughput of Perch 2.0 and Kinglet B/S/XS (v0.3.2 networks) on one machine.

Model-only rows: 5-s windows of the 10-min speed-kit file (speed_kit/bench.wav, written if missing) resident on the
GPU, outputs left on the GPU; each rep runs max(8, 2048 // batch) batches cycling over the resident
windows, 5 reps after warm-up, closed by torch.cuda.synchronize (Kinglet) or a host fetch of the last
output (Perch; the stream is in order). A cell with max/min rep wall >= 1.2 is rerun up to 3 times.
End-to-end rows: ffmpeg decode + windowing + host-to-GPU + inference + outputs to host for the same
file, at the batch each model uses in our pipelines (Perch 128 = teacher/embed_perch.py --gpu, Kinglet 32 =
common.scoring.BATCH); CPU-load dependent.

Kinglet = PyTorch eager fp32 (framework defaults): torchaudio FFT mel + log + normalisation + backbone
+ emb head + full 14,795-class layer + softmax (checkpoint loaded with score_student.Distilled).
Perch 2.0 = its GPU SavedModel (Kaggle perch_v2/2, config PERCH_GPU_DIR) as distributed. TF and torch run in
separate processes (--part); each part merges its rows into --out. Bug check: the GPU embedding of each tier is
compared with the released trunk ONNX (RELEASE_ASSETS) on CPU.

Usage (one machine label per JSON; compute_speed/gpuspeed_run.sh runs the paper's sequence):
  python compute_speed/bench_gpu.py --part perch --machine desktop
  python compute_speed/bench_gpu.py --part kinglet --machine desktop [--variant channels_last|compile]
  python compute_speed/bench_gpu.py --part launch --machine desktop   # batch-1 diagnostic: GPU ops vs host enqueue
(secondary Kinglet rows: channels_last memory format, or torch.compile per batch shape; key kinglet_<variant>)
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DISTILL_ROOT, PERCH_GPU_DIR, RELEASE_ASSETS, RESULTS  # noqa: E402
from compute_speed.speed_kit.speed_kit import write_wav  # noqa: E402
from teacher.embed_perch import decode, windows  # noqa: E402

WAV = Path(__file__).resolve().parent / "speed_kit/bench.wav"  # the speed kit's synthetic 10-min file
CKPTS = {"B": DISTILL_ROOT / "g3conf_v1/regnety_016_full/best.pt",
         "S": DISTILL_ROOT / "g3conf_v1/efficientnet_lite0_full/best.pt",
         "XS": DISTILL_ROOT / "g2xcl_v1/mobilenetv4_conv_small_full/best.pt"}
ONNX = {"B": "b", "S": "s", "XS": "xs"}  # released v0.3.2 trunk ONNX
BATCHES = (1, 16, 64, 256)
E2E_BATCH = {"perch": 128, "kinglet": 32}
RESIDENT = 512  # windows kept on the GPU (bench.wav's 120 windows, tiled); a multiple of every batch
REPS, MAX_RATIO, RETRIES = 5, 1.2, 3


def host_state() -> dict:
    """Load average (recorded with every cell)."""
    return {"loadavg": [round(x, 2) for x in os.getloadavg()]}


def timed_cell(run_rep, n_windows: int) -> dict:
    """5 reps of run_rep() (it must block until the GPU is done); rerun the cell if max/min >= 1.2."""
    for attempt in range(1, RETRIES + 1):
        state0 = host_state()
        walls = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            run_rep()
            walls.append(time.perf_counter() - t0)
        ratio = max(walls) / min(walls)
        if ratio < MAX_RATIO:
            break
        print(f"  rep spread {ratio:.2f} >= {MAX_RATIO}, rerun ({attempt}/{RETRIES})", flush=True)
    wps = n_windows / statistics.median(walls)
    return {"windows_per_s": round(wps, 1), "rtf": round(5.0 * wps, 1), "n_windows_per_rep": n_windows,
            "rep_walls_s": [round(w, 4) for w in walls], "max_min_ratio": round(ratio, 3),
            "spread_ok": ratio < MAX_RATIO, "attempts": attempt, "host_before": state0, "host_after": host_state()}


def bench_perch(xs: np.ndarray, audio_s: float) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpu = tf.config.list_physical_devices("GPU")
    assert gpu, "TensorFlow sees no GPU"
    tf.config.experimental.set_memory_growth(gpu[0], True)
    fn = tf.saved_model.load(str(PERCH_GPU_DIR)).signatures["serving_default"]
    info = tf.sysconfig.get_build_info()
    out = {"versions": {"tensorflow": tf.__version__, "tf_cuda_build": info.get("cuda_version"),
                        "tf_cudnn_build": info.get("cudnn_version"),
                        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled()},
           "model": "Perch 2.0 GPU SavedModel perch_v2/2 (fp32)", "model_only": {}}
    with tf.device("/GPU:0"):
        res = tf.constant(np.resize(xs, (RESIDENT, xs.shape[1])))
    for b in BATCHES:
        bs = resident_batches(res, b)
        try:
            o = fn(inputs=bs[0])  # warm-up: XLA compiles this batch shape
            fin = all(bool(np.isfinite(o[k].numpy()).all()) for k in ("label", "embedding"))
            tf.config.experimental.reset_memory_stats("GPU:0")
            cell = timed_cell(partial(perch_rep, fn, bs), len(bs) * b)
        except tf.errors.ResourceExhaustedError as e:
            out["model_only"][str(b)] = {"oom": str(e).splitlines()[0]}
            print(f"perch b={b}: OOM", flush=True)
            continue
        cell |= {"finite": fin, "peak_mem_mb": round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20)}
        out["model_only"][str(b)] = cell
        print(f"perch b={b}: {cell['windows_per_s']} win/s, RTF {cell['rtf']}, ratio {cell['max_min_ratio']}, "
              f"peak {cell['peak_mem_mb']} MB", flush=True)

    b = E2E_BATCH["perch"]

    def e2e():
        x = windows(decode(str(WAV), 0), 5.0)[1]
        for i in range(0, len(x), b):
            xb = x[i:i + b]
            nb = len(xb)
            if nb < b:  # pad to the compiled shape, as embed_perch --gpu does
                xb = np.pad(xb, ((0, b - nb), (0, 0)))
            o = fn(inputs=tf.constant(xb))
            o["label"].numpy()[:nb], o["embedding"].numpy()[:nb]

    e2e()  # warm-up (ffmpeg cold start, compile of batch 128)
    out["end_to_end"] = timed_cell(e2e, int(audio_s / 5)) | {"batch": b, "audio_s": audio_s}
    print(f"perch e2e b={b}: RTF {out['end_to_end']['rtf']}", flush=True)
    return out


def resident_batches(res, b: int) -> list:
    """max(8, 2048 // b) batches cycling over the resident windows (RESIDENT is a multiple of b)."""
    return [res[(i * b) % RESIDENT:(i * b) % RESIDENT + b] for i in range(max(8, 2048 // b))]


def perch_rep(fn, bs: list):
    for x in bs:
        o = fn(inputs=x)
    o["label"][0, 0].numpy()  # host fetch of the last output closes the rep (in-order stream)


def kinglet_rep(fwd, bs: list):
    import torch

    for x in bs:
        fwd(x)
    torch.cuda.synchronize()


def bench_kinglet(xs: np.ndarray, audio_s: float, variant: str) -> dict:
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    out = {"versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "onnxruntime": ort.__version__,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark}}
    resident = torch.from_numpy(np.resize(xs, (RESIDENT, xs.shape[1]))).to("cuda")
    for tier, ckpt in CKPTS.items():
        out[tier] = bench_tier(tier, ckpt, resident, xs[::15][:8], audio_s, variant)
        torch.cuda.empty_cache()
    return out


def kinglet_fwd(ckpt: Path, variant: str):
    """(fwd, model): audio (B, 160000) on the GPU -> (embedding, full-vocab softmax), fp32."""
    import torch

    from evaluate.accuracy.score_student import Distilled

    d = Distilled(ckpt, "cuda")
    assert not d.cfg.get("frontend") and d.full, "expects the v0.3.2 legacy-mel full-vocab checkpoints"
    model, mel, mu, sd = d.model, d.mel, d.cfg["mel_mu"], d.cfg["mel_sd"]
    channels_last = variant == "channels_last"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    @torch.no_grad()
    def fwd(x):  # fp32: no autocast (Distilled.__call__ would use bf16 on CUDA)
        m = ((torch.log(mel(x) + 1e-6) - mu) / sd).unsqueeze(1)
        if channels_last:
            m = m.contiguous(memory_format=torch.channels_last)
        _, emb, logits = model(m)
        return emb, torch.softmax(logits, 1)

    if variant == "compile":  # one graph per batch shape; compiled during the warm-up passes
        fwd = torch.compile(fwd, dynamic=False)
    return fwd, model


def bench_tier(tier: str, ckpt: Path, resident, check: np.ndarray, audio_s: float, variant: str) -> dict:
    """One Kinglet tier: bug check (ii) on the 8 `check` windows, model-only cells, end-to-end."""
    import onnxruntime as ort
    import torch

    fwd, model = kinglet_fwd(ckpt, variant)

    # bug check (ii): GPU embedding vs the released ONNX on CPU (onnxruntime, 4 threads)
    opt = ort.SessionOptions()
    opt.intra_op_num_threads, opt.inter_op_num_threads = 4, 1
    sess = ort.InferenceSession(str(RELEASE_ASSETS / f"kinglet-{ONNX[tier]}-v0.3.2.onnx"), opt,
                                providers=["CPUExecutionProvider"])
    e_cpu = sess.run(["embedding"], {"waveform": check})[0]
    e_gpu, p_gpu = (t.cpu().numpy() for t in fwd(torch.from_numpy(check).cuda()))
    cos = (e_cpu * e_gpu).sum(1) / np.linalg.norm(e_cpu, axis=1) / np.linalg.norm(e_gpu, axis=1)
    row = {"ckpt": str(ckpt.relative_to(DISTILL_ROOT)), "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "cos_vs_onnx_cpu": [round(float(c), 6) for c in cos], "cos_ok": bool((cos > 0.999).all()),
           "softmax_rowsum_max_dev": float(np.abs(p_gpu.sum(1) - 1).max()), "model_only": {}}
    print(f"{tier}: cos vs ONNX CPU min {cos.min():.6f}", flush=True)

    for b in BATCHES:
        bs = resident_batches(resident, b)
        for _ in range(3):  # warm-up (cuDNN algorithm choice, allocator)
            e, p = fwd(bs[0])
        torch.cuda.synchronize()
        fin = bool(torch.isfinite(e).all() and torch.isfinite(p).all())
        torch.cuda.reset_peak_memory_stats()
        cell = timed_cell(partial(kinglet_rep, fwd, bs), len(bs) * b) | {
            "finite": fin, "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}
        row["model_only"][str(b)] = cell
        print(f"{tier} b={b}: {cell['windows_per_s']} win/s, RTF {cell['rtf']}, "
              f"ratio {cell['max_min_ratio']}, peak {cell['peak_mem_mb']} MB", flush=True)

    b = E2E_BATCH["kinglet"]

    def e2e():
        x = windows(decode(str(WAV), 0), 5.0)[1]
        for i in range(0, len(x), b):
            e, p = fwd(torch.from_numpy(x[i:i + b]).cuda())
            e.cpu().numpy(), p.cpu().numpy()

    e2e()
    row["end_to_end"] = timed_cell(e2e, int(audio_s / 5)) | {"batch": b, "audio_s": audio_s}
    print(f"{tier} e2e b={b}: RTF {row['end_to_end']['rtf']}", flush=True)
    return row


def launch_b1(xs: np.ndarray) -> dict:
    """Why batch 1 is slow in eager mode: per b=1 forward, the number of GPU ops and their summed device time
    (torch.profiler, 20 forwards) against the host time to enqueue one forward (200 forwards, no sync).
    Host time >> device time means the cell is bound by kernel launches on the CPU, not by the GPU."""
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.set_num_threads(4)
    x = torch.from_numpy(xs[:1]).cuda()
    out = {}
    for tier, ckpt in CKPTS.items():
        fwd, _ = kinglet_fwd(ckpt, "eager")
        for _ in range(20):
            fwd(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                fwd(x)
            torch.cuda.synchronize()
        dev = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
        t0 = time.perf_counter()
        for _ in range(200):
            fwd(x)
        enq = (time.perf_counter() - t0) / 200
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / 200
        out[tier] = {"gpu_ops_per_forward": sum(e.count for e in dev) / 20,
                     "gpu_ms_per_forward": round(sum(e.self_device_time_total for e in dev) / 20 / 1e3, 3),
                     "host_enqueue_ms_per_forward": round(enq * 1e3, 3), "wall_ms_per_forward": round(wall * 1e3, 3),
                     "host": host_state()}
        print(tier, out[tier], flush=True)
        del fwd
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["perch", "kinglet", "launch"], required=True)
    ap.add_argument("--machine", required=True, help="machine label stored in the JSON")
    ap.add_argument("--variant", choices=["eager", "channels_last", "compile"], default="eager",
                    help="Kinglet runtime variant; non-eager rows are secondary (key kinglet_<variant>)")
    ap.add_argument("--out", type=Path, default=RESULTS / "gpuspeed_v1.json")
    args = ap.parse_args()

    write_wav(WAV)  # no-op when the speed kit already wrote it

    audio = decode(str(WAV), 0)
    xs = windows(audio, 5.0)[1]
    audio_s = len(xs) * 5.0
    print(f"{WAV.name}: {len(xs)} windows ({audio_s / 60:.0f} min)", flush=True)
    t0 = time.time()
    decode(str(WAV), 0)  # ffmpeg cold start
    decode_only = timed_cell(lambda: windows(decode(str(WAV), 0), 5.0), len(xs))  # the e2e rows' CPU share
    if args.part == "perch":
        key, part = "perch", bench_perch(xs, audio_s)
    elif args.part == "launch":
        key, part = "kinglet_launch_b1", launch_b1(xs)
    else:
        key = "kinglet" if args.variant == "eager" else f"kinglet_{args.variant}"
        part = bench_kinglet(xs, audio_s, args.variant)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    assert res.get("machine", args.machine) == args.machine, f"{args.out} holds {res['machine']}"
    cpu = next(ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
               if ln.startswith("model name"))
    res |= {"machine": args.machine, "gpu": smi, "cpu": cpu,
            "python": platform.python_version(), "wav": WAV.name, "resident_windows": RESIDENT,
            "batches": list(BATCHES), "e2e_batch": E2E_BATCH, "reps": REPS, "max_min_ratio": MAX_RATIO}
    res[key] = part | {"decode_only": decode_only, "date": time.strftime("%Y-%m-%d %H:%M"),
                       "elapsed_s": round(time.time() - t0)}
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out} [{key}]")


if __name__ == "__main__":
    main()
