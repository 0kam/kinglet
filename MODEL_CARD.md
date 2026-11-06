# Kinglet Model Card (v0.3.0)

## Overview

Kinglet models are CPU-first bioacoustic models knowledge-distilled from Perch 2.0 (Google,
Apache-2.0). Since v0.3 a model is a **trunk** plus a **regional head pack** ("distill once, fold
anywhere"): the trunk ONNX maps a 5 s / 32 kHz mono waveform to (a) a 1536-d embedding trained to
match the Perch 2.0 embedding and (b) a trunk feature; the pack is a linear head folding the full
**14,795-class Perch vocabulary** (iNat2024 taxa + FSD50K sound classes) onto your region's
species, a set of sound groups (human voice, vehicles, geophony, ...), and an explicit
background/"other" channel. Scores are softmax probabilities over the pack rows; the evaluated
species operating threshold is **0.1**, the operating point used by the Perch ecosystem.

| tier | backbone | params (trunk) | speed (4 CPU threads, trunk+JP pack, preprocessing incl.) |
|---|---|---|---|
| Kinglet-B | RegNetY-016 | 11.7 M | ~600x realtime |
| Kinglet-S | EfficientNet-Lite0 | 5.3 M | ~1,250x realtime |
| Kinglet-XS | MobileNetV4-Conv-S | 4.5 M | ~2,200x realtime |

Speed vs the v0.2 single-file models on the same harness: B −1.6%, S −3.4%, XS −1%.

## What changed vs v0.2.0: full-vocabulary distillation + regional folding

v0.2 distilled the teacher's softmax already folded onto 317 Japanese species + background, so the
student only ever knew Japan. v0.3 distills the **unfolded 14,795-class softmax** (teacher targets
only, no human labels; training data grew from ~76k to ~460k recordings by adding BirdSet XCL) and
folds at deployment time. Folding a pack takes seconds on a laptop (`kinglet build-pack`) and needs
no audio: species rows slice exactly from the full head; the "other" row is a ridge fit of the exact
out-of-pack logsumexp on calibration trunk features shipped with the release.

On the Japanese benchmarks this costs nothing measurable (all tiers within the pre-registered
−0.01 of v0.2, see Evaluation), while adding:

- **any-region deployment**: fold the same trunk onto any species list. On BirdSet soundscapes
  (regions never seen by our pipeline, scored with folded packs): Powdermill (48 spp, dawn chorus)
  cmAP B 0.484 / S 0.463 / XS 0.445; High Sierra (21 spp) 0.518 / 0.501 / 0.487. Teacher folded
  the same way: 0.540 / 0.527. BirdNET V2.4: 0.576 / 0.498. Kinglet-B reaches ~94% of the folded
  teacher's mean; capacity ordering is monotone (XS < S < B on every region metric).
- **sound groups** (below), including a zero-training human-voice detector for privacy masking.

## Sound groups

The vocabulary's 198 FSD50K sound classes (+ Homo sapiens) ship pre-assigned to 7 groups:
`human_voice` (20 classes), `vehicle` (24), `geophony` (15: rain, wind, thunder, water, fire),
`music` (35), `gunshot` (1; fireworks/explosion are deliberately kept out as classic confusers),
`domestic_animal` (10), `other_anthrophony` (76). 18 classes that overlap bird/wildlife species or
are source-ambiguous onomatopoeia stay ungrouped. A group's score is the sum of its members'
softmax probabilities; every pack carries the group rows.

**human_voice is calibrated** (threshold 0.005 ≈ FPR 1% on voice-free Japanese PAM windows,
measured on the XS tier with LibriSpeech positives mixed into field audio):

| TPR at the 0.005 threshold | clean speech | SNR 0 dB | −6 dB | −12 dB |
|---|---|---|---|---|
| Kinglet-XS v0.3.0 | 1.00 | 0.98 | 0.95 | 0.74 |
| Perch 2.0 (same protocol) | 1.00 | 0.93 | 0.87 | 0.72 |

On held-out field windows the shipped packs flag 1.6% (B), 1.6% (S), 2.9% (XS) of windows at
0.005 — an upper bound on the false-mask rate, since field audio contains genuine voice and
over-masking is the safe failure direction for privacy. The other six groups ship **without** an
operating threshold (scores are exposed in the Python API; the CLI skips uncalibrated groups) —
calibrate against your own deployment before operational use.

## Files

Per tier: `kinglet-<t>-v0.3.0.onnx` (trunk; `.gemm-mel` variant for ARM), `.pack-jp.npz`
(prebuilt Japan pack: 317 species + 181 group rows + other), `.headpack.npz` (full head +
calibration features, only needed for `kinglet build-pack`). Shared:
`kinglet-v0.3.0.species-jp.csv` (the JP species list with authoritative vocabulary indices),
`kinglet-v0.3.0.reliability.csv` (per-class reliability tiers, below).
v0.2 single-file models (logits in the graph) are not supported by the v0.3 package.

```
kinglet analyze recordings/ --model kinglet-s-v0.3.0.onnx --pack kinglet-s-v0.3.0.pack-jp.npz
kinglet build-pack --headpack kinglet-s-v0.3.0.headpack.npz --species my_region.csv --out my.npz
```

## Per-class reliability tiers

Distillation quality per class tracks how often the class was actually heard during training.
Counting training recordings per vocabulary class (name-matched via the eBird 2024 taxonomy;
258 training species totalling 1.9% of recordings could not be matched, so counts are slight
undercounts):

| tier | training recordings | vocabulary classes | of the 317 JP classes |
|---|---|---|---|
| high | ≥ 100 | 996 | 245 |
| medium | 10–99 | 4,439 | 65 |
| low | 1–9 | 4,056 | 5 |
| softlabel_only | 0 | 5,304 | 1 (shtalb: dead row) |

`softlabel_only` classes were never positive in training audio — the student knows them only
through the teacher's soft probabilities on other sounds. Treat their detections as weak signals;
`kinglet-v0.3.0.reliability.csv` has the per-class table.

## Evaluation (Japanese PAM benchmarks; higher is better)

| | Kasumigaura macro AUPRC | Suishin clip AP | Fuji 5 s AP | OKEON top-1 |
|---|---|---|---|---|
| Perch 2.0 (teacher, ~47x realtime) | 0.737 | 0.588 | 0.398 | 0.849 |
| BirdNET V2.4 | 0.534 | 0.385 | 0.306 | 0.940* |
| Kinglet-B v0.3.0 | 0.714 | 0.582 | 0.389 | 0.832 |
| Kinglet-S v0.3.0 | 0.698 | 0.568 | 0.382 | 0.834 |
| Kinglet-XS v0.3.0 | 0.714 | 0.568 | 0.379 | 0.826 |
| (Kinglet-B v0.2.0) | 0.722 | 0.572 | 0.380 | 0.832 |

All tiers pass the pre-registered gates (JP cores within −0.01 of v0.2 per tier; S and XS improve
on every AP benchmark; B's Kasumigaura −0.008 is inside the paired CI). Mean embedding cosine to
the teacher on held-out focal validation: 0.871 (B), 0.868 (S), 0.842 (XS); v0.2 measured
0.880 / 0.864 / 0.829.

Open-set false positives, per tier on 250,997 Japanese PAM windows ("teacher-quiet" = the teacher
gives every Japanese species < 0.1; 92.2% of windows), scored through the shipped packs:

| | P(any detection on a teacher-quiet window) @0.1 | worst single class |
|---|---|---|
| Kinglet-B v0.3.0 | 1.5% | 0.4% |
| Kinglet-S v0.3.0 | 1.4% | 0.3% |
| Kinglet-XS v0.3.0 | 1.3% | ≤1% |

Pack-vs-exact-fold equivalence (the ridge "other" row vs computing the full 14,795-way softmax):
mean per-species probability error ≤ 4e-5, detections at 0.1 disagree on ≤ 0.8% of windows.

\* OKEON detections were sourced with BirdNET V2.4, biasing this comparison in its favor.

## Known issues / limitations (v0.3.0)

- OKEON top-1 remains ~3.5 pt below v0.1.0 on all tiers (unchanged from v0.2, as expected).
- The ridge "other" row is an approximation: on rare bird-dense windows a single species
  probability can shift by up to ~0.4 vs the exact fold (detection flips at 0.1 on ~0.8% of
  windows). If a study needs exact folding, score the full head from the headpack directly.
- human_voice TPR was measured on the XS tier; B/S share the recipe and their folded scores
  behave consistently, but their TPR curves were not separately measured. Mixing used additive
  synthetic mixes (LibriSpeech + field audio); real-world distant/reverberant speech below
  −12 dB SNR will be missed — the mask reduces, not eliminates, privacy exposure.
- Species outside your pack's list fold into "other" and are never reported; build the pack with
  every species you may encounter.
- Region folding is evaluated on two BirdSet soundscapes (Powdermill, High Sierra); elsewhere the
  claim rests on the teacher's own generality plus the reliability tiers.
- Embeddings approximate Perch 2.0; for embedding-critical analyses prefer Kinglet-B, or Perch
  itself when compute allows.

## License and provenance

Apache-2.0 (see LICENSE, NOTICE). Distilled from Perch 2.0; some training audio carries
non-commercial Creative Commons licenses and the Macaulay Library research license — following the
precedent of Perch itself, the released weights are licensed Apache-2.0 with this provenance
disclosed. Training additionally uses BirdSet XCL (Xeno-canto-derived). Audio is not
redistributed. Contact: Ryotaro Okamoto.
