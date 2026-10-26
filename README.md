# Kinglet 🐦

**Kinglet-sized bioacoustic embedding and classification models, distilled from
[Perch 2.0](https://github.com/google-research/perch).**

Kinglets are among the smallest birds in the northern hemisphere. Kinglet models are among the
smallest bioacoustic models: a single ONNX file (~25 MB, preprocessing included) that processes
**months of field recordings in hours on an ordinary laptop CPU** — no GPU, no Python ML stack —
while staying close to Perch 2.0 on Japanese soundscape benchmarks. Each model outputs both species
scores and a 1536-d embedding projected into the Perch 2.0 embedding space.

## Models (preview)

| model | backbone | size | speed* | classes |
|---|---|---|---|---|
| Kinglet-S (preview1) | EfficientNet-Lite0 | 27 MB | ~1,300x realtime | 317 Japanese bird species |
| Kinglet-XS (preview1) | MobileNetV4-Conv-S | 24 MB | ~2,400x realtime | 317 Japanese bird species |

\* 4 CPU threads, log-mel preprocessing included (Ryzen 9950X3D; expect roughly half on a typical
laptop). Download from [GitHub Releases](../../releases); the `.classes.csv` must sit next to the
`.onnx`. **Preview checkpoints**: final models will replace these after recipe selection.

## Quickstart

```bash
pip install kinglet-bio   # needs ffmpeg on PATH
kinglet analyze recordings/ --model kinglet-s-preview1_efficientnet_lite0.onnx --out detections.csv
```

Or from Python / any ONNX runtime:

```python
from kinglet import Kinglet, decode, windows
m = Kinglet("kinglet-s-preview1_efficientnet_lite0.onnx")
starts, wins = windows(decode("field_recording.wav"))   # 5 s windows at 32 kHz mono
embeddings, scores = m.run(wins)                         # (n, 1536), (n, 317) sigmoid scores
```

## Model I/O

- Input `waveform`: float32 `(batch, 160000)` — 5 s, 32 kHz, mono, amplitude in [-1, 1]
- Output `embedding`: `(batch, 1536)` — projection into the Perch 2.0 embedding space
  (mean cosine ~0.9 to the teacher on held-out soundscapes)
- Output `logits`: `(batch, n_classes)` — pre-sigmoid; class order in `<model>.classes.csv`
  (`index, ebird_code, scientific_name, japanese_name, logit_bg_p99, score_bg_p99,
  logit_bg_p999, score_bg_p999`). Raw sigmoid scores run high (the distillation teacher's logits are
  uncalibrated), so a single global threshold does not work. Use the per-species background
  thresholds instead: `logit > logit_bg_p99` corresponds to roughly a 1% false-positive rate per
  5 s window on bird-free field audio (BirdVox-DCASE-20k, hasbird=0; p999 ~ 0.1%). The bundled CLI
  reports the top-k species per window.

## License

[Apache License 2.0](LICENSE). Kinglet models are distilled from Perch 2.0 (Apache-2.0, Google);
see [NOTICE](NOTICE) and [MODEL_CARD.md](MODEL_CARD.md) for training-data provenance.
