# Kinglet Model Card (v0.1.0)

## Overview

Kinglet models are CPU-first bioacoustic models for Japanese birds, knowledge-distilled from
Perch 2.0 (Google, Apache-2.0). Each model maps a 5 s / 32 kHz mono waveform to (a) a 1536-d
embedding trained to match the Perch 2.0 embedding (cosine loss) and (b) sigmoid scores over 317
Japanese bird species (binary cross-entropy against the teacher's calibrated sigmoid outputs; no
human labels).

| tier | backbone | params | speed (4 CPU threads, preprocessing incl.) |
|---|---|---|---|
| Kinglet-B | RegNetY-016 | 12.0 M | ~660x realtime |
| Kinglet-S | EfficientNet-Lite0 | 5.7 M | ~1,350x realtime |
| Kinglet-XS | MobileNetV4-Conv-S | 4.9 M | ~2,400x realtime |

## Training data (distillation inputs; audio is not redistributed)

- Focal recordings of 317 Japanese species (~63k recordings, first 60 s each):
  Xeno-canto (CC licenses incl. BY-NC variants), iNaturalist (CC licenses),
  Macaulay Library (Cornell Lab research license)
- BirdSet XCM (~13k recordings; Xeno-canto-derived)

Targets are Perch 2.0 outputs only. No evaluation-set labels were used for training or selection.
Unlike preview1, v0.1.0 uses **no unlabeled PAM audio** as distillation input (it did not help).

## Score calibration

Perch 2.0's raw logits carry a large species-dependent background offset (median sigmoid ~0.55 on
bird-free audio), which preview1 inherited. v0.1.0 subtracts, from the teacher's logits during
distillation, each species' 99th-percentile logit measured on ~250k unlabeled Japanese soundscape
windows. Within-species ranking is unchanged; absolute scores become usable:

- On bird-free field audio (1,500 BirdVox-DCASE-20k clips, hasbird=0, UK soundscapes), the overall
  background score median is **0.09** (preview1: ~0.65).
- The fixed threshold **score > 0.5** gives a <1% false-positive rate per 5 s window for ~93% of
  species (288–296 of 317, depending on the tier).
- For the remaining species (mostly rare seabirds/island endemics with few training recordings),
  use the per-species `score_bg_p99` / `score_bg_p999` columns in `classes.csv` as thresholds.

## Evaluation (Japanese PAM benchmarks; higher is better)

| | Kasumigaura macro AUPRC | Suishin clip AP | Fuji 5 s AP | OKEON top-1 |
|---|---|---|---|---|
| Perch 2.0 (teacher, ~47x realtime) | 0.737 | 0.588 | 0.398 | 0.849 |
| BirdNET V2.4 | 0.534 | 0.385 | 0.306 | 0.940* |
| Kinglet-B v0.1.0 | 0.721 | 0.558 | 0.379 | 0.863 |
| Kinglet-S v0.1.0 | 0.708 | 0.519 | 0.358 | 0.869 |
| Kinglet-XS v0.1.0 | 0.701 | 0.501 | 0.341 | 0.863 |

On Kasumigaura the 95% CI of the paired difference vs Perch 2.0 includes 0 for all three tiers.
Mean embedding cosine to the teacher on these soundscape domains: 0.86–0.90 (B > S > XS),
essentially equal to the cosine on held-out focal recordings — no measurable domain gap.

\* OKEON detections were sourced with BirdNET V2.4, biasing this comparison in its favor.

## Known issues / limitations (v0.1.0)

- 317 Japanese bird species only; no other taxa (frogs, insects, mammals score as spurious birds).
- ~20–30 species (mostly data-poor seabirds) still exceed score 0.5 on ~1% of bird-free windows;
  use their `classes.csv` quantile columns for strict filtering.
- Trained and evaluated on Japanese soundscapes; performance elsewhere is untested.
- Embeddings approximate Perch 2.0 (cosine ~0.9); for embedding-critical analyses prefer
  Kinglet-B, or Perch itself when compute allows.

## License and provenance

Apache-2.0 (see LICENSE, NOTICE). Distilled from Perch 2.0; some training audio carries
non-commercial Creative Commons licenses and the Macaulay Library research license — following the
precedent of Perch itself, the released weights are licensed Apache-2.0 with this provenance
disclosed. Contact: Ryotaro Okamoto.
