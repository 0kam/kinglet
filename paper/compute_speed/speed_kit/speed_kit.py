# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Kinglet paper speed kit: end-to-end CPU throughput of Perch 2.0, BirdNET and Kinglet on this machine.

One command (see README.md):  python speed_kit.py --label <CPU name>
  1. downloads the public models into models/ (checksums verified; Perch 2.0 via kagglehub; the Kinglet files from
     RELEASE_BASE_URL below, or copied into models/ by hand),
  2. writes a synthetic 10-min 32 kHz 16-bit WAV (seeded noise; no recordings involved),
  3. times every model end to end -- ffmpeg decode + segmenting + model, per pass -- with 4 threads and with all
     physical cores: one warm-up pass, then --reps measured passes (median reported). Each (model, threads) runs in
     its own Python process, so a failing model cannot stop the others and TensorFlow's thread pools start clean,
  4. writes results/speedkit_<label>.json (machine description + every pass).
Nothing identifying is recorded: no hostname, user name or paths.

Models, as distributed (the paper's Appendix S4 protocol):
  perch          Perch 2.0 CPU SavedModel (TensorFlow), 5-s windows, batch 8
Main rows use each model as it is normally used; *_5s / *_overlap / *_convdft / *_pack rows are secondary.
  birdnet_v3     BirdNET+ V3.0 (final release, Zenodo 24162518), FP32 ONNX (ONNX Runtime), as its analyze.py runs
                 it by default: consecutive 3-s chunks (no overlap, last one zero-padded), batch 16 (analyze.py's
                 default file is the FP16 ONNX; the paper scores the FP32 file, timed here)
  birdnet_v3_5s  the same file on 5-s windows, one input per window, batch 32 (the paper's accuracy evaluation)
  birdnet_v24    BirdNET V2.4 FP32 TFLite (LiteRT) as BirdNET-Analyzer runs it by default: consecutive
                 non-overlapping 3-s segments, resampled to 48 kHz, 48 segments per call
  birdnet_v24_overlap  three overlapping 3-s crops (0-3, 1-4, 2-5 s) per 5-s window, 48 crops per call (the paper's
                 accuracy evaluation; 1.8x the passes)
  kinglet_b/s/xs Kinglet v0.3.2 networks with the FFT frontend (fftfront_v1) and the exact fold, the default on
                 computers: power spectrogram by numpy FFT (power_spectrogram below = kinglet.power_spectrogram),
                 power-input trunk ONNX (.power.onnx), the full 14,795-class head (released .headpack.npz) and a
                 softmax over all classes, read at the Japanese list (317 species and the sound-group members);
                 implemented here from the model files, as kinglet's Kinglet.run with an exact pack
  kinglet_*_convdft  the same with the released v0.3.2 waveform-input trunk ONNX, whose graph computes the
                 spectrogram as a DFT matrix product (conv-DFT, 0.526 GMACs per 5-s window)
  kinglet_*_pack the FFT frontend with the compact Japanese pack (464 rows, ridge "other" row; for microcontrollers)
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import wave
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = 32000
AUDIO_S = 600.0
# >>> SET THIS: base URL of the released Kinglet model files (the .onnx / .npz files listed in FILES below).
# Files already present in models/ are used after the sha256 check, so instead of setting the URL the files can
# be copied into models/ by hand.
RELEASE_BASE_URL = "<URL of the release files>"
KINGLET = RELEASE_BASE_URL
POWER = "v0.3.3"  # power-input trunks of the FFT frontend, exported from the v0.3.2 checkpoints
KINGLET_POWER = RELEASE_BASE_URL
V3_FILE = "BirdNET+_V3.0_Global_12K_FP32.onnx"
V3_URL = f"https://zenodo.org/records/24162518/files/{V3_FILE}?download=1"
V24_URL = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip?download=1"
PERCH_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
# file -> (url or None for the V2.4 zip member, sha256) -- the exact files behind the paper's numbers
FILES = {
    V3_FILE: (V3_URL, "e78c3b50e224cd79d549d3c3b5efde2436acd1d3fbd72694f2a0b8547fa18685"),  # md5 550fbfc3... as Zenodo
    "BirdNET_v2.4_audio-model-fp32.tflite":
        (None, "55f3e4055b1a13bfa9a2452731d0d34f6a02d6b775a334362665892794165e4c"),
    "kinglet-b-v0.3.2.onnx": (f"{KINGLET}/kinglet-b-v0.3.2.onnx",
                              "851abefdd1983fe9db22f809beae2718a9e914c6b9f348cfa3cb611f3c8afe25"),
    "kinglet-b-v0.3.2.pack-jp.npz": (f"{KINGLET}/kinglet-b-v0.3.2.pack-jp.npz",
                                     "9bd6a67a4bf112153085e6235267fb9306a26380191cd0fa38d11f8155f3a1a0"),
    "kinglet-s-v0.3.2.onnx": (f"{KINGLET}/kinglet-s-v0.3.2.onnx",
                              "0fa1a1b3e2f0f32fe5fce25653acd5ab01ebf593629df8eee985beadb5f8af42"),
    "kinglet-s-v0.3.2.pack-jp.npz": (f"{KINGLET}/kinglet-s-v0.3.2.pack-jp.npz",
                                     "5eabd6312440c613f7f9e967bd898aff8a35bab031049648a851e56d12bc16e6"),
    "kinglet-xs-v0.3.2.onnx": (f"{KINGLET}/kinglet-xs-v0.3.2.onnx",
                               "6b81be3d01aa1cc7c3bc7747a783a47351cdb8ec6c470df409e67fbb5eec6e75"),
    "kinglet-xs-v0.3.2.pack-jp.npz": (f"{KINGLET}/kinglet-xs-v0.3.2.pack-jp.npz",
                                      "a2d6b7933fb36ff9ca6cb21040a2fc44961ba1459a93fe8cc5d0f4105186efa0"),
    f"kinglet-b-{POWER}.power.onnx": (f"{KINGLET_POWER}/kinglet-b-{POWER}.power.onnx",
                                     "ea2f8a2347c1346ea38c1678a404a007eaf01089315989911bcbcac08a20a345"),
    f"kinglet-s-{POWER}.power.onnx": (f"{KINGLET_POWER}/kinglet-s-{POWER}.power.onnx",
                                     "d37a4251d6fcac6654c7700971184b7b51758544d264aa479b74449f039d9b7e"),
    f"kinglet-xs-{POWER}.power.onnx": (f"{KINGLET_POWER}/kinglet-xs-{POWER}.power.onnx",
                                      "034e65f0d25760d4ee3c80b29edc738d0e1722b3a18d67e19496dbf987c8386f"),
    "kinglet-b-v0.3.2.headpack.npz": (f"{KINGLET}/kinglet-b-v0.3.2.headpack.npz",
                                      "e2f4ceeca9fda2e203f5a4bc9a94aaa34b13863b9e4f4d1f3968faac9e904b74"),
    "kinglet-s-v0.3.2.headpack.npz": (f"{KINGLET}/kinglet-s-v0.3.2.headpack.npz",
                                      "d1b7017c5ee8517293b9c47793a6e9f2c775d8af177aed61fb161467f5361d0c"),
    "kinglet-xs-v0.3.2.headpack.npz": (f"{KINGLET}/kinglet-xs-v0.3.2.headpack.npz",
                                       "52cc4ba828345f35b320494830ab53e938d26a7a253879aaf2a507d4d8c1ae2a"),
}
PERCH_SHA256 = {"saved_model.pb": "1301ce8cef8ff4625f9447d9849d880f0dfa5fa9e41c13ec3274a4b0084c7f17",
                "variables/variables.index": "29c49db4f95727b8c4c22522cf8bb8c77e2eb930ce5b5b43e3a00317d479da1d"}
MODELS = ("perch", "birdnet_v3", "birdnet_v24", "kinglet_b", "kinglet_s", "kinglet_xs",
          "birdnet_v3_5s", "birdnet_v24_overlap", "kinglet_b_convdft", "kinglet_s_convdft", "kinglet_xs_convdft",
          "kinglet_b_pack", "kinglet_s_pack", "kinglet_xs_pack")
PACKAGES = ("tensorflow", "ai-edge-litert", "onnxruntime", "numpy", "scipy", "kagglehub", "psutil", "imageio-ffmpeg")


# ---------------------------------------------------------------- helpers shared by parent and workers

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_exe() -> tuple[str, str]:
    """ffmpeg on PATH (as in the paper), else the static build shipped by imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg"), "system"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg"


def decode(path: Path, ffmpeg: str):
    """Audio file -> mono float32 at 32 kHz (same ffmpeg call as the paper's embed_perch.decode)."""
    import numpy as np

    res = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-ss", "0", "-i", str(path), "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(res.stdout, dtype=np.float32)


def segments(a, seconds: float):
    """Consecutive non-overlapping segments; the last partial one is zero-padded."""
    import numpy as np

    n = int(seconds * SR)
    return np.pad(a, (0, -len(a) % n)).reshape(-1, n)


def power_spectrogram(wins):
    """Same computation as kinglet.power_spectrogram, kept in the kit: (n, samples) float32 -> (n, 1 + samples //
    320, 513) float32, torchaudio Spectrogram(n_fft=1024, hop_length=320, power=2): centred frames with reflect padding,
    periodic Hann window, one-sided FFT, one window at a time, FFT in float64."""
    import numpy as np

    n_fft, hop = 1024, 320
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    out = np.empty((len(wins), 1 + wins.shape[1] // hop, n_fft // 2 + 1), np.float32)
    for i, x in enumerate(wins):
        x = np.pad(x, n_fft // 2, mode="reflect")
        z = np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * hann)
        np.add(np.square(z.real), np.square(z.imag), out=out[i], casting="same_kind")
    return out


def timed(fn, reps: int) -> dict:
    """One unmeasured warm-up pass, then `reps` measured passes (median, as bench_speed.py)."""
    fn()
    walls, cpu0 = [], time.process_time()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - t0)
    wall = statistics.median(walls)
    return {"xrt": AUDIO_S / wall, "wall_s_per_run": round(wall, 3), "xrt_runs": [round(AUDIO_S / w, 1) for w in walls],
            "cpu_s_per_wall_s": round((time.process_time() - cpu0) / sum(walls), 2)}


# ---------------------------------------------------------------- worker: one model at one thread count

def build(model: str, threads: int, models: Path, perch_dir: str):
    """-> (fn(batch) -> output, segment seconds, batch size, expected output width)."""
    import numpy as np

    if model == "perch":
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        sig = tf.saved_model.load(perch_dir).signatures["serving_default"]
        return (lambda b: sig(inputs=tf.constant(b))["label"].numpy()), 5.0, 8, 14795
    if model.startswith("birdnet_v3"):
        import onnxruntime as ort

        opt = ort.SessionOptions()
        opt.intra_op_num_threads, opt.inter_op_num_threads = threads, 1
        sess = ort.InferenceSession(str(models / V3_FILE), opt, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        fn = lambda b: sess.run(["predictions"], {name: b})[0]  # noqa: E731
        return (fn, 5.0, 32, 11725) if model.endswith("_5s") else (fn, 3.0, 16, 11725)
    if model.startswith("birdnet_v24"):
        from ai_edge_litert.interpreter import Interpreter
        from scipy.signal import resample_poly

        it = Interpreter(model_path=str(models / "BirdNET_v2.4_audio-model-fp32.tflite"), num_threads=threads)
        inp, outp = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

        def invoke(crops):  # (n, 3 s at 32 kHz) -> logits; resampled to 48 kHz per call as in the paper
            x = resample_poly(crops, 3, 2, axis=1).astype(np.float32)
            it.resize_tensor_input(inp, x.shape)
            it.allocate_tensors()
            it.set_tensor(inp, x)
            it.invoke()
            return it.get_tensor(outp)

        if not model.endswith("_overlap"):
            return invoke, 3.0, 48, 6522
        return (lambda b: invoke(np.concatenate([b[:, s * SR : (s + 3) * SR] for s in range(3)]))
                .reshape(3, len(b), -1).mean(0)), 5.0, 16, 6522
    if model.startswith("kinglet_"):
        from kinglet import Kinglet

        tier = model.split("_")[1]
        convdft = model.endswith("_convdft")
        trunk = f"kinglet-{tier}-v0.3.2.onnx" if convdft else f"kinglet-{tier}-{POWER}.power.onnx"
        m = Kinglet(models / trunk, models / f"kinglet-{tier}-v0.3.2.pack-jp.npz", threads=threads)
        if model.endswith("_pack"):  # compact pack: softmax over its own rows ("other" last)
            W, bias, cols = m.W, m.b, None
        else:
            W, bias, cols = exact_head(models / f"kinglet-{tier}-v0.3.2.headpack.npz", m)

        def run(b):  # = Kinglet.run (kinglet with the FFT frontend), one batch of 64 windows
            emb, h = m.sess.run(None, {"waveform": b} if convdft else {"power": power_spectrogram(b)})
            lg = h @ W.T + bias
            p = np.exp(lg - lg.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            if cols is not None:
                p = p[:, cols]
            _groups = {g: p[:, c].sum(axis=1) for g, c in m.group_cols.items()}  # same work as Kinglet.run
            return p[:, : m.n_species]

        return run, 5.0, 64, m.n_species
    raise ValueError(model)


def exact_head(headpack: Path, m):
    """Full head (fp32, plus one dead row for unmatched species) and the full-head column of every row of the
    compact pack but "other": compact species and member rows are exact copies of full-head rows."""
    import numpy as np

    hp = np.load(headpack)
    W = np.vstack([hp["W"], np.zeros((1, hp["W"].shape[1]), hp["W"].dtype)]).astype(np.float32)
    bias = np.append(hp["b"].astype(np.float32), np.float32(-1e9))
    row = {r.tobytes(): i for i, r in enumerate(W[:-1])}
    dead = len(W) - 1
    cols = np.array([row.get(r.tobytes(), dead) for r in m.W[:-1].astype(np.float32)])
    unmatched = cols == dead
    assert (m.b[:-1][unmatched] < -1e8).all() and unmatched.sum() <= 5, "pack and headpack do not match"
    return W, bias, cols


def worker(args):
    fn, seg_s, batch, width = build(args.worker, args.threads[0], Path(args.models_dir), args.perch_dir)
    wav, ffmpeg = HERE / "bench.wav", ffmpeg_exe()[0]

    def e2e():
        x = segments(decode(wav, ffmpeg), seg_s)
        for i in range(0, len(x), batch):
            out = fn(x[i : i + batch])
        return out

    out = e2e()
    assert out.shape[-1] == width, (out.shape, width)  # the right model is loaded
    r = timed(e2e, args.reps)
    r |= {"segment_s": seg_s, "batch": batch, "passes_per_10min": len(segments(decode(wav, ffmpeg), seg_s))
          * (3 if args.worker.endswith("_overlap") else 1)}
    print("RESULT " + json.dumps(r), flush=True)


# ---------------------------------------------------------------- parent: models, audio, machine, runs

def download(url: str, dst: Path):
    if url.startswith("<"):
        raise RuntimeError("RELEASE_BASE_URL is not set in speed_kit.py; set it or copy the file into models/")
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  downloading {dst.name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "speed-kit"})
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)


def needed(name: str, wanted: list[str]) -> bool:
    """Whether a model file is used by any of the models to be timed."""
    prefix = {"birdnet_v3": "BirdNET+", "birdnet_v24": "BirdNET_v2.4", "kinglet_b": "kinglet-b-",
              "kinglet_s": "kinglet-s-", "kinglet_xs": "kinglet-xs-"}
    return any(name.startswith(p) for m in wanted for k, p in prefix.items() if m == k or m.startswith(k + "_"))


def fetch_models(models: Path, perch_dir: str | None, wanted: list[str]) -> tuple[dict, str | None]:
    """Download and verify the files of the models to be timed -> ({name: 'ok' | error}, Perch SavedModel directory)."""
    models.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, (url, digest) in FILES.items():
        if not needed(name, wanted):
            continue
        p = models / name
        try:
            if not p.exists():
                if url is None:  # BirdNET V2.4 FP32 TFLite: one member of the Zenodo zip
                    z = models / "BirdNET_v2.4_tflite.zip"
                    if not z.exists():
                        download(V24_URL, z)
                    with zipfile.ZipFile(z) as zf:
                        member = next(n for n in zf.namelist() if n.endswith("audio-model.tflite"))
                        p.write_bytes(zf.read(member))
                    z.unlink()
                else:
                    download(url, p)
            got = sha256(p)
            if got != digest:
                p.unlink()
                raise RuntimeError(f"sha256 mismatch ({got[:12]}...), file removed; rerun to download again")
            status[name] = "ok"
        except Exception as e:  # keep going: the models that are available still get timed
            status[name] = f"{type(e).__name__}: {e}"
            print(f"  {name}: {status[name]}")
    if "perch" not in wanted:
        return status, perch_dir
    try:
        if not perch_dir:
            import kagglehub

            print("  locating Perch 2.0 (kagglehub; downloads ~400 MB on first use) ...", flush=True)
            perch_dir = kagglehub.model_download(PERCH_HANDLE)
        status["perch"] = {f: ("ok" if sha256(Path(perch_dir) / f) == d else "sha256 differs from the paper's copy")
                           for f, d in PERCH_SHA256.items()}
    except Exception as e:
        status["perch"] = f"{type(e).__name__}: {e}"
        print(f"  perch: {status['perch']}")
    return status, perch_dir


def write_wav(p: Path):
    import numpy as np

    if p.exists():
        return
    a = np.random.default_rng(0).normal(0, 0.05, int(SR * AUDIO_S)).astype(np.float32)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=30).stdout.strip()
    except Exception:
        return ""


def cpu_name() -> str:
    s = platform.system()
    if s == "Linux":
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    if s == "Darwin":
        return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if s == "Windows":
        import winreg

        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    return platform.processor()


def read(p: str) -> str | None:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def machine_info() -> dict:
    import psutil

    s = platform.system()
    info = {"cpu": cpu_name(), "arch": platform.machine(), "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(), "memory_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "os": platform.platform(), "python": platform.python_version(),
            "packages": {}, "date": datetime.now().astimezone().isoformat(timespec="seconds")}
    for p in PACKAGES:
        try:
            info["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["packages"][p] = None
    try:
        import kinglet

        info["packages"]["kinglet"] = kinglet.__version__
    except ImportError:
        info["packages"]["kinglet"] = None
    exe, src = ffmpeg_exe()
    info["ffmpeg"] = {"source": src, "version": run([exe, "-version"]).splitlines()[0][:80]}
    bat = psutil.sensors_battery()
    power = {"battery": None if bat is None else {"percent": round(bat.percent), "plugged_in": bat.power_plugged}}
    if s == "Linux":
        info["os_release"] = platform.freedesktop_os_release().get("PRETTY_NAME") if hasattr(
            platform, "freedesktop_os_release") else None
        info["wsl"] = "microsoft" in platform.release().lower()
        flags = next((ln for ln in Path("/proc/cpuinfo").read_text().splitlines() if ln.startswith("flags")), "")
        info["isa"] = sorted(set(flags.split()) & {"avx2", "fma", "avx512f", "avx512_vnni", "avx512_bf16",
                                                   "avx_vnni", "amx_tile", "amx_bf16", "amx_int8"})
        c = "/sys/devices/system/cpu/cpu0/cpufreq/"
        power |= {"governor": read(c + "scaling_governor"), "epp": read(c + "energy_performance_preference"),
                  "platform_profile": read("/sys/firmware/acpi/platform_profile"),
                  "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
                  "boost": read("/sys/devices/system/cpu/cpufreq/boost")}
    elif s == "Darwin":
        info["isa"] = [ln.split(":")[0].rsplit(".", 1)[-1] for ln in run(["sysctl", "hw.optional.arm"]).splitlines()
                       if ln.endswith(": 1")]
        info["perf_levels"] = run(["sysctl", "-n", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"]).split()
        power["pmset"] = [ln.strip() for ln in run(["pmset", "-g"]).splitlines()
                          if re.search(r"powermode|lowpowermode|highpowermode", ln)]
    elif s == "Windows":
        power["scheme"] = run(["powercfg", "/getactivescheme"])
    info["power"] = power
    info["load_before"] = psutil.cpu_percent(interval=2.0)  # % of all CPUs busy just before the runs
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="short machine name for the output file, e.g. i7-8565U, M4-Pro, Ryzen-AI-7-PRO-350 (default: CPU name)")
    ap.add_argument("--note", default="", help="free text recorded in the JSON, e.g. the power / performance mode")
    ap.add_argument("--reps", type=int, default=5, help="measured passes per run (median reported); default 5")
    ap.add_argument("--threads", type=int, nargs="*", help="thread counts (default: 4 and all physical cores)")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    ap.add_argument("--models-dir", default=str(HERE / "models"))
    ap.add_argument("--perch-dir", help="an existing perch_v2_cpu/1 SavedModel directory (skips kagglehub)")
    ap.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    import psutil

    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label or cpu_name()).strip("-")[:40]
    phys = psutil.cpu_count(logical=False) or os.cpu_count()
    threads = sorted(set(args.threads or [4, phys]))
    out = HERE / "results" / f"speedkit_{label}.json"
    out.parent.mkdir(exist_ok=True)
    log = out.with_suffix(".log")
    print(f"[1/4] machine: {cpu_name()}, {phys} physical cores; threads {threads}")
    info = machine_info()
    if info["power"]["battery"] and not info["power"]["battery"]["plugged_in"]:
        print("  WARNING: running on battery - please plug in the charger (recorded in the JSON)")
    if info["load_before"] > 10:
        print(f"  WARNING: {info['load_before']:.0f}% CPU already busy - close other programs (recorded)")
    print("[2/4] models")
    status, perch_dir = fetch_models(Path(args.models_dir), args.perch_dir, args.models)
    print("[3/4] synthetic 10-min WAV + decode")
    wav = HERE / "bench.wav"
    write_wav(wav)
    ffmpeg = ffmpeg_exe()[0]
    res = {"kit": "speedkit_v2", "label": label, "note": args.note, "machine": info, "model_files": status,
           "reps": args.reps, "audio_s": AUDIO_S, "decode": timed(lambda: decode(wav, ffmpeg), args.reps), "runs": []}
    print(f"  decode {res['decode']['xrt']:.0f}x real time")
    home = os.path.expanduser("~")  # replaced by "~" in the stored error messages and log
    print(f"[4/4] timing {len(args.models)} models x {len(threads)} thread settings (Perch takes longest)")
    for th in threads:
        # BLAS / OpenMP pools (numpy in the Kinglet pack, scipy resampling) limited to the same thread count
        env = os.environ | {v: str(th) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
        env |= {"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2"}
        for model in args.models:
            row = {"model": model, "threads": th}
            t0 = time.time()
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", model, "--threads", str(th),
                   "--reps", str(args.reps), "--models-dir", args.models_dir, "--perch-dir", str(perch_dir or "")]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=3600)
                line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
                if line:
                    row |= {"status": "ok"} | json.loads(line[7:])
                else:
                    row |= {"status": "error", "error": p.stderr.replace(home, "~")[-1500:]}
                with log.open("a", encoding="utf-8") as f:
                    f.write(f"=== {model} t{th}\n{p.stdout}{p.stderr}".replace(home, "~"))
            except Exception as e:
                row |= {"status": "error", "error": f"{type(e).__name__}: {e}".replace(home, "~")}
            res["runs"].append(row)
            out.write_text(json.dumps(res, indent=1))  # partial results survive an interruption
            shown = f"{row['xrt']:8.1f}x real time" if row["status"] == "ok" else "FAILED (see JSON/log)"
            print(f"  {model:22s} t{th:<3d} {shown}  ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'model':22s} " + " ".join(f"{'t' + str(t):>9s}" for t in threads))
    for model in args.models:
        cells = []
        for t in threads:
            r = next(r for r in res["runs"] if r["model"] == model and r["threads"] == t)
            cells.append(f"{r['xrt']:9.1f}" if r["status"] == "ok" else f"{'FAIL':>9s}")
        print(f"{model:22s} " + " ".join(cells))
    print(f"\nwrote {out}\nKeep this JSON file (and the .log next to it if any row FAILED).")


if __name__ == "__main__":
    main()
