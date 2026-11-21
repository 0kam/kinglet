# Speed kit (speedkit_v2)

Times Perch 2.0, BirdNET V2.4, BirdNET+ V3.0 and Kinglet B/S/XS **end to end** (decode of an audio file ->
segmenting -> model output), each model as distributed. This is the laptop protocol of Appendix S4 in one command;
the paper's laptop rows (Table 1, Fig. 2a, Table S9) are `results/speedkit_i7-8565U.json`,
`results/speedkit_M4-Pro.json` and `results/speedkit_Ryzen-AI-7-PRO-350.json` (plus the secondary runs
`speedkit_Ryzen-AI-7-PRO-350-t8.json` and `check_head_M4-Pro.json`).

- The models are downloaded on first use and checked against the SHA-256 of the files behind the paper's numbers:
  Perch 2.0 = Kaggle `google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1` (via kagglehub);
  BirdNET V2.4 = the FP32 TFLite of Zenodo record 15050749; BirdNET+ V3.0 = the final FP32 ONNX of Zenodo record
  24162518 (`BirdNET+_V3.0_Global_12K_FP32.onnx`); Kinglet = the v0.3.2 trunk ONNX, full-vocabulary output layer
  (`.headpack.npz`) and Japanese pack (`.pack-jp.npz`) of each tier, and the power-spectrogram-input trunks
  `kinglet-{b,s,xs}-v0.3.3.power.onnx` (FFT frontend, exported from the same checkpoints).
- The audio is a synthetic 10-min WAV (32 kHz, 16 bit, seeded noise); no recordings are used.
- Each model runs with 4 threads and with as many threads as physical cores: one warm-up pass, then 5 measured
  passes (median reported), each (model, threads) in its own Python process.
- Main rows use each model as it is normally used:
  - BirdNET V2.4 (`birdnet_v24`): consecutive non-overlapping 3-s segments, as BirdNET-Analyzer by default.
  - BirdNET+ V3.0 (`birdnet_v3`): consecutive non-overlapping 3-s chunks (last one zero-padded), 16 per call, as
    its `analyze.py` by default; the kit times the FP32 ONNX used for the paper's accuracy numbers (the default file
    of `analyze.py` is the FP16 ONNX).
  - Kinglet (`kinglet_b/s/xs`): FFT frontend (numpy FFT power spectrogram fed to the `.power.onnx` trunk) and the
    exact fold, the default on computers (full 14,795-class layer + softmax, read at the Japanese list).
- Secondary rows: `birdnet_v24_overlap` (three overlapping 3-s crops per 5-s window, as in the paper's accuracy
  evaluation; 1.8x the model calls), `birdnet_v3_5s` (one 5-s window per input, 32 per call, as in the paper's
  accuracy evaluation), `kinglet_*_convdft` (released v0.3.2 waveform-input trunk, which computes the spectrogram
  as a DFT matrix product inside the ONNX graph), `kinglet_*_pack` (FFT frontend + the compact Japanese pack of
  464 rows).
- Output: one file `results/speedkit_<label>.json`. No host name, user name or path is recorded (only CPU name,
  core counts, memory, OS, Python and package versions, ffmpeg version and power state).

Needs: internet access on first use (about 1.4 GB; Zenodo can take 10-20 min), about 4 GB of disk, 20-60 min.

## Model files (set once)

The Kinglet files are fetched from `RELEASE_BASE_URL`, a placeholder constant near the top of `speed_kit.py`:

```python
RELEASE_BASE_URL = "<URL of the release files>"
```

Set it to the address under which the release files are published (each file is fetched as
`RELEASE_BASE_URL/<file name>`), or skip the download by copying the files into a folder `models/` next to
`speed_kit.py`; files found there are used after the SHA-256 check. The files are the ones listed in `FILES` in
`speed_kit.py` (`kinglet-{b,s,xs}-v0.3.2.{onnx,headpack.npz,pack-jp.npz}`, `kinglet-{b,s,xs}-v0.3.3.power.onnx`).
If neither is done, the Kinglet rows are reported as `FAILED` and the other models are still timed.

## Before measuring

1. Connect the charger (a run on battery prints a warning and is recorded in the JSON).
2. Close other programs (browser, sync clients, IDE); place the computer where it can cool.
3. Check the power mode (Windows: Settings -> System -> Power; vendor performance modes on gaming laptops;
   macOS: low power mode off) and record it with `--note`.

## Install and run

[uv](https://docs.astral.sh/uv/) provides Python 3.10; no other installation is needed. The `kinglet` package is
installed from this archive (its root is three levels above this folder).

Linux / macOS:

```bash
cd paper/compute_speed/speed_kit
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt ../../..
.venv/bin/python speed_kit.py --label i7-8565U --note "AC, powersave + balance_performance"
```

Windows (PowerShell):

```powershell
cd paper\compute_speed\speed_kit
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt ..\..\..
.venv\Scripts\python.exe speed_kit.py --label Ryzen-AI-7-PRO-350 --note "AC, best performance"
```

The label names the output file (default: the CPU name); use a different label for each machine and operating
system, e.g. `M4-Pro` on the MacBook. ffmpeg from PATH is used when installed, otherwise the static build of
imageio-ffmpeg.

## During and after the run

- One progress line per row (e.g. `perch  t4  10.6x real time`); Perch takes longest.
- A model that fails does not stop the others (its row is `FAILED`; details in `results/speedkit_<label>.log`,
  with the home directory replaced by `~`).
- `check_head.py [models_dir]` repeats the Kinglet output-layer computation on 64 synthetic windows and compares
  float32 with float64 (finite values, numpy warnings); the paper ran it on the M4 Pro
  (`results/check_head_M4-Pro.json`).
- Clean-up: delete the folder (Perch 2.0 also stays in `~/.cache/kagglehub`).

## Notes

- Versions: Python 3.10, TensorFlow 2.21.0, ai-edge-litert 2.3.0, ONNX Runtime 1.23.2 and NumPy 2.2.6 as in the
  paper; SciPy 1.15.3, kagglehub, psutil and imageio-ffmpeg are fixed by the kit (not recorded in the paper's
  environment).
- Each model is timed in its own process, and the BLAS / OpenMP pools of numpy and SciPy are limited to the same
  thread count. BirdNET+ V3.0 is used through its distributed `predictions` output.
- The Kinglet exact fold and the FFT power spectrogram are computed inside the kit from the released files (same
  computation as `kinglet.Kinglet.run` with an exact pack and `kinglet.power_spectrogram`).
- Options: `--threads 4 8`, `--models perch kinglet_b` (only these models; only their files are downloaded),
  `--reps 5`, `--perch-dir <perch_v2_cpu/1 directory>` (an existing Perch 2.0 copy; skips kagglehub),
  `--models-dir`.
