# Kinglet Model Card (preview1)

## Overview

Kinglet models are CPU-first bioacoustic models for Japanese birds, knowledge-distilled from
Perch 2.0 (Google, Apache-2.0). Each model maps a 5 s / 32 kHz mono waveform to (a) a 1536-d
embedding trained to match the Perch 2.0 embedding (cosine loss) and (b) sigmoid scores over 317
Japanese bird species (binary cross-entropy against the teacher's sigmoid outputs; no human labels).

## Training data (distillation inputs; audio is not redistributed)

- Focal recordings of 317 Japanese species (~63k recordings, first 60 s each):
  Xeno-canto (CC licenses incl. BY-NC variants), iNaturalist (CC licenses),
  Macaulay Library (Cornell Lab research license)
- BirdSet XCM (~13k recordings; Xeno-canto-derived)
- Unlabeled Japanese passive acoustic monitoring audio (~350 h) from monitoring networks
  (used as distillation inputs only; preview1 includes it, see Known issues)

Targets are Perch 2.0 outputs only. No evaluation-set labels were used for training or selection.

## Evaluation (Japanese PAM benchmarks; higher is better)

| | Kasumigaura macro AUPRC | Suishin clip AP | Fuji 5 s AP | OKEON top-1 |
|---|---|---|---|---|
| Perch 2.0 (teacher, ~47x realtime) | 0.737 | 0.588 | 0.398 | 0.849 |
| BirdNET V2.4 | 0.534 | 0.385 | 0.306 | 0.940* |
| Kinglet-S preview1 | 0.686 | 0.509 | 0.347 | 0.832 |
| Kinglet-XS preview1 | 0.643 | 0.394 | 0.320 | 0.799 |

\* OKEON detections were sourced with BirdNET V2.4, biasing this comparison in its favor.

## Known issues / limitations (preview1)

- Scores are not threshold-calibrated; use as rankings.
- 317 Japanese bird species only; no other taxa.
- preview1 was trained with naively mixed unlabeled PAM audio, which slightly depresses recall on
  night wetland soundscapes; the final release will use an improved recipe.
- Trained and evaluated on Japanese soundscapes; performance elsewhere is untested.

## License and provenance

Apache-2.0 (see LICENSE, NOTICE). Distilled from Perch 2.0; some training audio carries
non-commercial Creative Commons licenses and the Macaulay Library research license — following the
precedent of Perch itself, the released weights are licensed Apache-2.0 with this provenance
disclosed. Contact: Ryotaro Okamoto.
