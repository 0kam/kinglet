# Kinglet Model Card (v0.3.2)

## Overview

Kinglet models are CPU-first bioacoustic models knowledge-distilled from Perch 2.0 (Google,
Apache-2.0). Since v0.3 a model is a **trunk** plus a **regional head pack** ("distill once, fold
anywhere"): the trunk ONNX maps a 5 s / 32 kHz mono waveform to (a) a 1536-d embedding trained to
match the Perch 2.0 embedding and (b) a trunk feature; the pack is a linear head folding the full
**14,795-class Perch vocabulary** (iNat2024 taxa + FSD50K sound classes) onto your region's
species, a set of sound groups (human voice, vehicles, geophony, ...), and an explicit
background/"other" channel. Scores are softmax probabilities over the pack rows; the evaluated
species operating threshold is **0.1**, the operating point used by the Perch ecosystem.

| tier | backbone | params (trunk) | laptop, end to end* | laptop, model only* | workstation, end to end** |
|---|---|---|---|---|---|
| Kinglet-B | RegNetY-016 | 11.7 M | 142x realtime | 147x | 515x |
| Kinglet-S | EfficientNet-Lite0 | 5.3 M | 253x | 270x | 967x |
| Kinglet-XS | MobileNetV4-Conv-S | 4.5 M | 399x | 416x | 1,456x |
| *Perch 2.0 (teacher)* | | | 10.6x | 10.7x | 46.8x |
| *BirdNET V2.4* | | | 36.9x | 37.1x | 115x |
| *BirdNET+ V3* | | | 31.3x | 31.4x | 149x |

\* Intel Core i7-8565U, 4 threads, one session for all six models on a 10-minute 32 kHz 16-bit
WAV; end to end = decoding + windowing + model (trunk + JP pack) on every pass, model only =
windows already in memory. \** Same measurement on a Ryzen 9 9950X3D, 4 threads. The ratio to
Perch 2.0 depends on the machine (13–38x on the laptop, 11–31x on the workstation).

Speed vs the v0.2 single-file models on the same harness: B −1.6%, S −3.4%, XS −1%.

## What changed in v0.3.2

The trunks, the species rows of every pack and the sound-group thresholds are identical to v0.3.1,
and v0.3.1 packs keep working. Rebuilt or added:

- **packs and headpacks** are rebuilt: the calibration features behind the "other" row are
  held-out validation recordings of the public training corpora. On the WABAD windows the v0.3.1
  and v0.3.2 Japan packs differ on "any species ≥ 0.1" in 0.8–1.0% of windows.
- **the `music` group is removed**: it proved unreliable. Its classes fold into "other"
  (v0.3.1 packs still report a `music` score — ignore it).
- `kinglet.embedding_from_feature` restores the Perch-space embedding from a stored trunk feature,
  and `kinglet build-pack --region` fetches a regional species list.
- The sound-group and open-set measurements in this card are now made on public data.

## What changed vs v0.2.0: full-vocabulary distillation + regional folding

v0.2 distilled the teacher's softmax already folded onto 317 Japanese species + background, so the
student only ever knew Japan. v0.3 distills the **unfolded 14,795-class softmax** (teacher targets
only, no human labels; training data grew from ~76k to ~460k recordings by adding BirdSet XCL) and
folds at deployment time. Folding a pack takes seconds on a laptop (`kinglet build-pack`) and needs
no audio: species rows slice exactly from the full head; the "other" row is a ridge fit of the exact
out-of-pack logsumexp on calibration trunk features shipped with the release.

Folding at deployment time adds:

- **any-region deployment**: fold the same trunk onto any species list. On BirdSet soundscapes
  (regions never seen by our pipeline, scored with folded packs): Powdermill (48 spp, dawn chorus)
  cmAP B 0.484 / S 0.463 / XS 0.445; High Sierra (21 spp) 0.518 / 0.501 / 0.487. Teacher folded
  the same way: 0.540 / 0.527. BirdNET V2.4: 0.576 / 0.498. Kinglet-B reaches ~94% of the folded
  teacher's mean; capacity ordering is monotone (XS < S < B on every region metric).
- **sound groups** (below), including a human-voice flag that needs no extra training.

## Sound groups

Of the vocabulary's 198 FSD50K sound classes (+ Homo sapiens), 146 ship pre-assigned to 6 groups:
`human_voice` (20 classes), `vehicle` (24), `geophony` (15: rain, wind, thunder, water, fire),
`gunshot` (1; fireworks/explosion are deliberately kept out as classic confusers),
`domestic_animal` (10), `other_anthrophony` (76). 18 classes that overlap bird/wildlife species or
are source-ambiguous onomatopoeia stay ungrouped, and the 35 musical-instrument classes are left
out since the `music` group was removed in v0.3.2. A group's score is the sum of its members'
softmax probabilities; every pack carries the group rows.

**human_voice flags clear voices only.** The shipped threshold is 0.005, a fixed operating point
that flags 0.6% of the 60,158 WABAD windows. Measured on public data — LibriSpeech speech embedded
in WABAD soundscape windows at a given signal-to-noise ratio (B / S / XS):

| | clean speech | SNR 0 dB | −6 dB | −12 dB |
|---|---|---|---|---|
| detected at 0.005 | 1.00 / 1.00 / 1.00 | 0.56 / 0.58 / 0.68 | 0.41 / 0.42 / 0.49 | 0.24 / 0.27 / 0.28 |

Of the 352 WABAD windows in which two independent speech detectors (PANNs CNN14 and Silero VAD)
both find speech, the flag catches 47–49% at 0.005. Lowering the threshold to about 0.002 raises
that to 78–80% and flags 2.5% of all windows. **Do not use it as the only privacy mask**: at the
shipped threshold it misses about half of the audible voices. Where privacy matters, add a
dedicated voice activity detector — in the same embedded-speech test Silero VAD (default 0.5)
detected 0.94 / 0.73 / 0.47 at 0 / −6 / −12 dB while flagging 0.7% of the WABAD windows.

**The other groups** use fixed thresholds of the same kind. Measured on public data with the
v0.3.2 packs — FSD50K *eval* clips carrying a member label (uploader-disjoint from the training
source; 59–400 clips per group), clean and embedded at 0 dB in WABAD soundscape windows
(B / S / XS):

| group | threshold | detected, clean | detected, SNR 0 dB | WABAD windows flagged |
|---|---|---|---|---|
| gunshot | 0.0003 | 0.98 / 0.98 / 0.98 | 0.73 / 0.68 / 0.59 | 0.3–0.4% |
| domestic_animal | 0.003 | 0.84 / 0.78 / 0.82 | 0.56 / 0.53 / 0.50 | 0.3–0.4% |
| geophony (provisional) | 0.1 | 0.64 / 0.59 / 0.49 | 0.12 / 0.10 / 0.06 | 0.1% |
| vehicle (provisional) | 0.05 | 0.69 / 0.67 / 0.68 | 0.14 / 0.13 / 0.11 | 0.1% |

All thresholds are fixed operating points set during development, not confidence levels.
**geophony and vehicle are provisional features with limited performance**: they catch events
that dominate the window and miss most events no louder than the background (6–14% at 0 dB).
WABAD carries no labels for these sounds, so the flagged share mixes true events and false alarms.
Treat group rows as flags for review, not verdicts — loud impulsive events can co-flag
gunshot/vehicle — and re-tune the thresholds against your own deployment. `other_anthrophony` is
a catch-all and ships without a threshold (score exposed in the Python API; the CLI skips it).

## Files

Per tier: `kinglet-<t>-v0.3.2.onnx` (trunk; `.gemm-mel` variant for ARM), `.pack-jp.npz`
(prebuilt Japan pack: 317 species + 146 group rows + other), `.headpack.npz` (full head +
calibration features, only needed for `kinglet build-pack`). Shared:
`kinglet-v0.3.2.species-jp.csv` (the JP species list with authoritative vocabulary indices),
`kinglet-v0.3.2.reliability.csv` (per-class reliability tiers, below).
v0.2 single-file models (logits in the graph) are not supported by the v0.3 package.

```
kinglet analyze recordings/ --model kinglet-s-v0.3.2.onnx --pack kinglet-s-v0.3.2.pack-jp.npz
kinglet build-pack --headpack kinglet-s-v0.3.2.headpack.npz --species my_region.csv --out my.npz
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
`kinglet-v0.3.2.reliability.csv` has the per-class table.

## Evaluation on public benchmarks (higher is better)

Class-mean average precision (cmAP) on [WABAD](https://zenodo.org/records/17293588) (60,158 5-s
windows, 1,189 species, 72 sites) and the seven [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet)
test sets, scored by folding the full-vocabulary output onto each benchmark's species list. No
benchmark recording was used in training, and no benchmark label for model selection.

| | WABAD | POW | HSN | NES | PER | SNE | SSW | UHH |
|---|---|---|---|---|---|---|---|---|
| Perch 2.0 (teacher) | 0.279 | 0.540 | 0.527 | 0.404 | 0.229 | 0.350 | 0.473 | 0.383 |
| Kinglet-B v0.3.2 | 0.272 | 0.484 | 0.518 | 0.400 | 0.233 | 0.360 | 0.481 | 0.363 |
| Kinglet-S v0.3.2 | 0.259 | 0.463 | 0.501 | 0.391 | 0.215 | 0.338 | 0.459 | 0.353 |
| Kinglet-XS v0.3.2 | 0.249 | 0.445 | 0.487 | 0.385 | 0.207 | 0.323 | 0.452 | 0.348 |

Capacity ordering is monotone (XS < S < B) on all eight benchmarks. On WABAD, pooled AUPRC is
0.317 (B) / 0.301 (S) / 0.297 (XS) against 0.314 for Perch 2.0, and per site Kinglet-B stays within
±0.06 AUPRC of Perch 2.0 at all 72 sites (71 within ±0.05; median difference −0.008, S −0.018,
XS −0.026). The largest gap, POW, is the densest set (dawn chorus, 2.8 species per window).

Embedding compatibility with Perch 2.0 on the WABAD windows: mean cosine between the two models'
embeddings of the same window 0.868 (B) / 0.861 (S) / 0.834 (XS). Few-shot retrieval (k = 1, 5)
with Perch exemplars against a Kinglet archive reaches 99–100% (B), 97–98% (S), 95% (XS) of Perch's
own retrieval mAP, and 93–99% in the reverse direction. A linear probe trained on Perch embeddings
and applied unchanged to Kinglet embeddings keeps 93% (B) / 92% (S) / 86% (XS) of its cmAP;
refitting it on Kinglet embeddings recovers 100% / 100% / 97%.

Open-set false detections: the 317-species Japan list scored on the same eight test sets (381,911
windows), where almost no listed species occurs (69 of them are annotated in WABAD, 0–5 in each
BirdSet set), so nearly every bird is outside the list. A false detection is a listed species above
the threshold that is not annotated in that window. Kinglet is scored through the v0.3.2 Japan
pack, Perch 2.0 through its full-vocabulary softmax sliced to the list.

| | false detections per window | windows with any | recall on annotated listed species (WABAD) |
|---|---|---|---|
| Perch 2.0, p > 0.1 | 0.034 | 3.3% | 0.40 |
| Kinglet-B, p > 0.1 | 0.027 | 2.7% | 0.40 |
| Kinglet-S, p > 0.1 | 0.027 | 2.7% | 0.38 |
| Kinglet-XS, p > 0.1 | 0.021 | 2.0% | 0.36 |
| BirdNET+ V3, confidence > 0.1 / > 0.25 | 0.026 / 0.011 | 2.4% / 1.0% | 0.62 / 0.51 |
| BirdNET V2.4, confidence > 0.1 / > 0.25 | 0.017 / 0.004 | 1.6% / 0.4% | 0.23 / 0.15 |

These are single operating points, not a ranking. The rate depends on the avifauna: for Kinglet it
runs from under 0.002 per window (PER) to 0.05–0.07 (UHH), and close relatives of listed species
account for much of it (Goldcrest fires on 1–3% of the Sierra Nevada windows, where Golden-crowned
Kinglet lives). A real but unannotated call counts as a false detection for every model alike.
BirdNET has 313 (V3) and 299 (V2.4) of the 317 classes; a missing class cannot fire.

Mean embedding cosine to the teacher on held-out focal validation recordings: 0.871 (B),
0.868 (S), 0.842 (XS); v0.2 measured 0.880 / 0.864 / 0.829.

Pack-vs-exact-fold equivalence (the ridge "other" row vs computing the full 14,795-way softmax):
on held-out focal validation recordings (7,171 windows) the mean per-species probability error is
≤ 7e-5 and detections at 0.1 disagree on 1.9–2.3% of windows. On the 60,158 WABAD windows, "any
species ≥ 0.1" differs between pack and exact fold on 1.7–2.0% of windows with the Japan pack (a
foreign list there), and on 5.5–5.8% with packs built for each site's own species list — almost
always the pack missing a window the exact fold flags (about one in ten of the windows the exact
fold flags). The public benchmark results above use the exact fold.

## Known issues / limitations (v0.3.2)

- The ridge "other" row is an approximation: on bird-dense windows a single species probability
  can shift by up to ~0.5 vs the exact fold, and with a pack for the local species list about one
  in ten windows that the exact fold would flag at 0.1 falls below the threshold (see Evaluation).
  If a study needs exact folding, score the full head from the headpack directly.
- human_voice catches clear voices and misses faint ones: about half of the audible voices in
  WABAD pass undetected at the shipped threshold (see Sound groups). It reduces privacy exposure
  but does not remove it; use a dedicated voice activity detector when that matters.
- Group TPRs were measured on FSD50K eval clips (the closest public match to the training
  distribution), not on field recordings of the events themselves. geophony and vehicle are
  provisional features with limited performance: they miss faint events by design.
- Species outside your pack's list fold into "other" and are never reported; build the pack with
  every species you may encounter.
- Region folding is evaluated on WABAD (72 sites) and seven BirdSet soundscape sets; for regions
  and taxa they do not cover, the claim rests on the teacher's own generality plus the
  reliability tiers.
- Embeddings approximate Perch 2.0; for embedding-critical analyses prefer Kinglet-B, or Perch
  itself when compute allows.

## License and provenance

Apache-2.0 (see LICENSE, NOTICE). Distilled from Perch 2.0; some training audio carries
non-commercial Creative Commons licenses and the Macaulay Library research license — following the
precedent of Perch itself, the released weights are licensed Apache-2.0 with this provenance
disclosed. Training additionally uses BirdSet XCL (Xeno-canto-derived). Audio is not
redistributed. Contact: Ryotaro Okamoto.
