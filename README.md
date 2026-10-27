# Kinglet 🐦

**Kinglet-sized bioacoustic embedding and classification models, distilled from
[Perch 2.0](https://github.com/google-research/perch).**

Kinglets are among the smallest birds in the northern hemisphere. Kinglet models are among the
smallest bioacoustic models: a single ONNX file (~25–55 MB, preprocessing included) that processes
**months of field recordings in hours on an ordinary laptop CPU** — no GPU, no Python ML stack —
while staying close to Perch 2.0 on Japanese soundscape benchmarks. Each model outputs both species
scores and a 1536-d embedding projected into the Perch 2.0 embedding space.

## Models (v0.1.0)

| model | backbone | size | speed* | Kasumigaura macro AUPRC** |
|---|---|---|---|---|
| Kinglet-B | RegNetY-016 | 52 MB | ~660x realtime | 0.721 |
| Kinglet-S | EfficientNet-Lite0 | 27 MB | ~1,350x realtime | 0.708 |
| Kinglet-XS | MobileNetV4-Conv-S | 24 MB | ~2,400x realtime | 0.701 |

All models cover 317 Japanese bird species and are **score-calibrated**: a fixed threshold of 0.5
corresponds to roughly a ≤1% false-positive rate per 5 s window on bird-free field audio for most
species (details in [MODEL_CARD.md](MODEL_CARD.md)).

\* 4 CPU threads, log-mel preprocessing included (Ryzen 9950X3D; expect roughly half on a typical
laptop). \** Perch 2.0 (the ~47x-realtime teacher) scores 0.737; the 95% CI of the paired
difference includes 0 for all three models. Download from [GitHub Releases](../../releases); the
`.classes.csv` must sit next to the `.onnx`. The `.gemm-mel` variants trade a slightly larger file
for a mel frontend that is faster on some ARM CPUs — on x86, use the default files.

## Quickstart

```bash
pip install kinglet-bio   # needs ffmpeg on PATH
kinglet analyze recordings/ --model kinglet-s-v0.1.0.onnx --out detections.csv
```

The CLI keeps, per 5 s window, the top-3 species scoring at least 0.5 (both adjustable with
`--top-k` / `--min-score`), and can dump embeddings with `--embeddings emb/`.

Or from Python / any ONNX runtime:

```python
from kinglet import Kinglet, decode, windows
m = Kinglet("kinglet-s-v0.1.0.onnx")
starts, wins = windows(decode("field_recording.wav"))   # 5 s windows at 32 kHz mono
embeddings, scores = m.run(wins)                         # (n, 1536), (n, 317) sigmoid scores
```

## Model I/O

- Input `waveform`: float32 `(batch, 160000)` — 5 s, 32 kHz, mono, amplitude in [-1, 1]
- Output `embedding`: `(batch, 1536)` — projection into the Perch 2.0 embedding space
  (mean cosine ~0.86–0.90 to the teacher on held-out soundscapes)
- Output `logits`: `(batch, n_classes)` — pre-sigmoid; class order in `<model>.classes.csv`
  (`index, ebird_code, scientific_name, japanese_name, logit_bg_p99, score_bg_p99,
  logit_bg_p999, score_bg_p999`). Scores are background-calibrated at training time, so
  `sigmoid(logit) > 0.5` works as a global default threshold. The `*_bg_*` columns are per-species
  score quantiles measured on bird-free field audio (BirdVox-DCASE-20k, hasbird=0) — use them
  instead of 0.5 when you need a stricter, per-species false-positive guarantee.

## License

[Apache License 2.0](LICENSE). Kinglet models are distilled from Perch 2.0 (Apache-2.0, Google);
see [NOTICE](NOTICE) and [MODEL_CARD.md](MODEL_CARD.md) for training-data provenance.
