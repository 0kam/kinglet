# Kinglet Model Card (v0.2.0)

## Overview

Kinglet models are CPU-first bioacoustic models for Japanese birds, knowledge-distilled from
Perch 2.0 (Google, Apache-2.0). Each model maps a 5 s / 32 kHz mono waveform to (a) a 1536-d
embedding trained to match the Perch 2.0 embedding (cosine loss) and (b) logits over 317 Japanese
bird species plus one background channel, trained with soft-label cross-entropy against the
teacher's full-vocabulary (14,795-class) softmax folded onto those 318 channels (no human labels).
Scores are softmax probabilities; the evaluated operating threshold is **0.1**, the same operating
point used by the Perch ecosystem.

| tier | backbone | params | speed (4 CPU threads, preprocessing incl.) |
|---|---|---|---|
| Kinglet-B | RegNetY-016 | 12.0 M | ~660x realtime |
| Kinglet-S | EfficientNet-Lite0 | 5.7 M | ~1,350x realtime |
| Kinglet-XS | MobileNetV4-Conv-S | 4.9 M | ~2,400x realtime |

## What changed vs v0.1.0: softmax distillation with a background channel

v0.1.0 scored each species with an independent calibrated sigmoid. That calibration held on
bird-free audio, but on field audio full of non-target sound the union of 317 independent sigmoids
gave the probability mass no sink: on a 188-recording subalpine deployment, Kinglet-B v0.1.0 at
score 0.5 reported 261 of 317 species (Perch 2.0 at its own operating point: 34), including
implausible seabirds. v0.2.0 instead distills the teacher's softmax, folding its 14,795-class
vocabulary onto the 317 Japanese species plus an explicit background/"other" channel that absorbs
everything else, exactly as the softmax does for Perch itself.

Open-set false-positive behavior, measured per tier on 250,997 Japanese PAM windows
("teacher-quiet" = windows where the teacher gives every Japanese species < 0.1; 92.2% of windows):

| | P(any detection on a teacher-quiet window) | worst single class |
|---|---|---|
| Kinglet-B v0.2.0 @0.1 | 2.0% | 0.3% |
| Kinglet-S v0.2.0 @0.1 | 1.7% | 0.4% |
| Kinglet-XS v0.2.0 @0.1 | 1.6% | 0.4% |
| (v0.1.0 XS @0.5) | ~100% | ~98% |

And on the labeled evaluation sets, counting detections of species outside the complete annotation
vocabulary as confirmed false positives (XS tier shown; all 317 classes scored):

| set (unit) | Perch 2.0 @0.1 | v0.1.0 @0.5 | v0.2.0 @0.1 |
|---|---|---|---|
| Suishin (confirmed FP / clip) | 0.61 | 30.2 | 0.45 |
| Fuji (confirmed FP / window) | 0.07 | 8.61 | 0.05 |
| Kasumigaura night (fires / window) | 0.32 | 8.30 | 0.28 |

v0.2.0 fires like the teacher (if anything slightly more conservatively); the v0.1.0
hundreds-of-species failure mode is gone.

## Training data (distillation inputs; audio is not redistributed)

- Focal recordings of 317 Japanese species (~63k recordings, first 60 s each):
  Xeno-canto (CC licenses incl. BY-NC variants), iNaturalist (CC licenses),
  Macaulay Library (Cornell Lab research license)
- BirdSet XCM (~13k recordings; Xeno-canto-derived)

Targets are Perch 2.0 outputs only (embeddings + folded softmax). No evaluation-set labels were
used for training or selection. No unlabeled PAM audio is used as distillation input.

## Scoring

Scores are softmax probabilities over 318 channels; the released CLI / Python wrapper drops the
background column and reports the 317 species. The fixed threshold **0.1** is the operating point
used in all evaluations below. `classes.csv` now has only `index, ebird_code, scientific_name,
japanese_name` — the v0.1.0 per-species background-quantile columns are superseded by the
background channel.

## Evaluation (Japanese PAM benchmarks; higher is better)

| | Kasumigaura macro AUPRC | Suishin clip AP | Fuji 5 s AP | OKEON top-1 |
|---|---|---|---|---|
| Perch 2.0 (teacher, ~47x realtime) | 0.737 | 0.588 | 0.398 | 0.849 |
| BirdNET V2.4 | 0.534 | 0.385 | 0.306 | 0.940* |
| Kinglet-B v0.2.0 | 0.722 | 0.572 | 0.380 | 0.832 |
| Kinglet-S v0.2.0 | 0.686 | 0.537 | 0.364 | 0.829 |
| Kinglet-XS v0.2.0 | 0.698 | 0.519 | 0.360 | 0.830 |

Versus v0.1.0: Kinglet-B improves on all three AP benchmarks (v0.1.0: 0.721 / 0.558 / 0.379) and
its Kasumigaura paired difference vs Perch 2.0 is not significant (Δmacro −0.015, 95% CI
[−0.048, +0.018]). S and XS improve on Suishin and Fuji; on Kasumigaura XS is −0.003 and S is
−0.022 vs v0.1.0 (see known issues). OKEON top-1 is ~3.5 pt below v0.1.0 on all tiers — the cost
of the shared trunk also learning the harder softmax task.

Mean embedding cosine to the teacher on held-out focal validation: 0.880 (B), 0.864 (S),
0.829 (XS); v0.1.0 measured 0.899 / 0.883 / 0.856.

\* OKEON detections were sourced with BirdNET V2.4, biasing this comparison in its favor.

## Known issues / limitations (v0.2.0)

- Kinglet-S regressed on falcated duck (`falduc`, ヨシガモ) on the Kasumigaura benchmark
  (AP 0.806 → 0.698), which pulls its 8-species macro below v0.1.0 (0.686 vs 0.708) even though
  its pooled AUPRC on the same benchmark improved (0.757 vs 0.743). The drop is recipe-systematic
  (XS shows the same direction). If falcated duck matters to your survey, prefer Kinglet-B.
- OKEON top-1 is ~3.5 pt below v0.1.0 on all tiers.
- v0.2.0 is slightly more conservative than the teacher: at the same 0.1 threshold it detects
  somewhat fewer species/windows than Perch 2.0.
- 317 Japanese bird species only. Other taxa (frogs, insects, mammals) are now mostly absorbed by
  the background channel instead of surfacing as spurious birds, but this is not guaranteed.
- Trained and evaluated on Japanese soundscapes; performance elsewhere is untested.
- Embeddings approximate Perch 2.0; for embedding-critical analyses prefer Kinglet-B, or Perch
  itself when compute allows.

## License and provenance

Apache-2.0 (see LICENSE, NOTICE). Distilled from Perch 2.0; some training audio carries
non-commercial Creative Commons licenses and the Macaulay Library research license — following the
precedent of Perch itself, the released weights are licensed Apache-2.0 with this provenance
disclosed. Contact: Ryotaro Okamoto.
