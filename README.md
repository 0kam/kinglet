# Kinglet 🐦

**Kinglet-sized bioacoustic embedding and classification models, distilled from
[Perch 2.0](https://github.com/google-research/perch).**

Kinglets are among the smallest birds in the northern hemisphere. Kinglet models are among the
smallest bioacoustic models: a trunk ONNX (~22–53 MB, preprocessing included) plus a ~2 MB regional
head pack that process **months of field recordings in hours on an ordinary laptop CPU** — no GPU,
no Python ML stack — while staying close to Perch 2.0 on public soundscape benchmarks from 72 sites
worldwide. Each model outputs species scores for **any regional species list**, sound-group scores
(human voice, vehicles, rain/wind, ...), and a 1536-d embedding in the Perch 2.0 embedding space,
so search indices and classifiers built on Perch embeddings keep working.

## Models (v0.3.1)

| model | backbone | trunk size | speed on a laptop* | 1,000 h of audio* | WABAD cmAP** | BirdSet cmAP vs Perch 2.0*** |
|---|---|---|---|---|---|---|
| Kinglet-B | RegNetY-016 | 51 MB | 142x realtime | 7.0 h | 0.272 (97%) | 90–103% |
| Kinglet-S | EfficientNet-Lite0 | 26 MB | 253x realtime | 4.0 h | 0.259 (93%) | 86–97% |
| Kinglet-XS | MobileNetV4-Conv-S | 22 MB | 399x realtime | 2.5 h | 0.249 (89%) | 82–96% |
| *Perch 2.0 (teacher)* | | | 10.6x realtime | 94 h | 0.279 | |

Since v0.3 the models are distilled from the teacher's **full 14,795-class vocabulary** and folded
to a region at deployment time ("distill once, fold anywhere"): a prebuilt Japan pack (317 species)
ships with the release, and `kinglet build-pack` folds the same trunk onto **any species list** in
seconds, no audio needed. Scores are **softmax probabilities with an explicit background channel**;
use a fixed species threshold of **0.1** (the Perch operating point). Every pack also carries
**sound groups** with operating thresholds (since v0.3.1): a calibrated `human_voice` detector
(threshold 0.005) for privacy masking before sharing field audio, calibrated `gunshot` and
`domestic_animal` detectors, and `geophony` / `vehicle` flags for rain-and-traffic noise screening.
The last two are **provisional features with limited performance** — hints for review, not
detections (details in [MODEL_CARD.md](https://github.com/0kam/kinglet/blob/main/MODEL_CARD.md)).

\* End to end — audio decoding, windowing and the model — on 4 CPU threads of a business laptop
(Intel Core i7-8565U), 10-minute 32 kHz WAV. On the same laptop BirdNET V2.4 runs at 37x and
BirdNET+ V3 at 31x realtime. \** Class-mean average precision on
[WABAD](https://zenodo.org/records/17293588) (60,158 5-s windows, 1,189 species, 72 sites on six
continents), in parentheses relative to Perch 2.0; per site, Kinglet-B stays within ±0.06 AUPRC of
Perch 2.0 at all 72 sites. \*** Range over the seven [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet)
test sets (POW, HSN, NES, PER, SNE, SSW, UHH). Both benchmarks are scored by folding the
full-vocabulary output onto each benchmark's species list; none of their recordings were used in
training. Download from [GitHub Releases](https://github.com/0kam/kinglet/releases).
The `.gemm-mel` trunk variants trade a slightly larger file for a mel frontend that is faster on
some ARM CPUs — on x86, use the default files.

## Quickstart

```bash
pip install kinglet-bio   # needs ffmpeg on PATH
kinglet analyze recordings/ --model kinglet-s-v0.3.1.onnx --pack kinglet-s-v0.3.1.pack-jp.npz --out detections.csv
```

The CLI keeps, per 5 s window, the top-3 species scoring at least 0.1 (adjustable with
`--top-k` / `--min-score`), adds one row per calibrated sound group that passes its own threshold
(`human_voice` windows are what you mute before sharing audio), and can dump embeddings with
`--embeddings emb/`.

Deploying outside Japan (or with your own species list)? Fold a pack from the `.headpack.npz`
release asset — the CSV needs `ebird_code, scientific_name` columns:

```bash
kinglet build-pack --headpack kinglet-s-v0.3.1.headpack.npz --species my_region.csv --out my_region.npz
```

Or let eBird supply the list for any region (free key at https://ebird.org/api/keygen, via `--ebird-key`
or `EBIRD_API_KEY`); it is saved as `us-ca.species.csv` to edit and rerun with `--species`:

```bash
kinglet build-pack --headpack kinglet-s-v0.3.1.headpack.npz --region US-CA --out us-ca.npz
```

Or from Python / any ONNX runtime:

```python
from kinglet import Kinglet, decode, windows
m = Kinglet("kinglet-s-v0.3.1.onnx", "kinglet-s-v0.3.1.pack-jp.npz")
starts, wins = windows(decode("field_recording.wav"))   # 5 s windows at 32 kHz mono
embeddings, scores, groups = m.run(wins)                # (n, 1536), (n, 317), {group: (n,)}
mask = groups["human_voice"] >= m.groups["human_voice"] # windows to mute before sharing
```

## Model I/O

- Input `waveform`: float32 `(batch, 160000)` — 5 s, 32 kHz, mono, amplitude in [-1, 1]
- Output `embedding`: `(batch, 1536)` — projection into the Perch 2.0 embedding space. On unseen
  soundscapes (WABAD) the mean cosine to the Perch 2.0 embedding of the same window is 0.87 (B) /
  0.86 (S) / 0.83 (XS). Mixing the two models in few-shot retrieval (Perch exemplars, Kinglet
  archive, or the reverse) keeps 93–100% of Perch's own retrieval mAP; a linear classifier trained
  on Perch embeddings applied unchanged keeps 93% (B) / 92% (S) / 86% (XS) of its cmAP, and
  97–100% once refitted on Kinglet embeddings.
- Output `feature`: `(batch, d)` — the trunk feature; scores = `softmax(feature @ W.T + b)` with
  `W, b` from the pack `.npz`, background column dropped (this is what `Kinglet.run` does).
  Row order: pack `codes` (species, then group-member classes, then "other").
- v0.2 single-file models (`logits` in the graph) are not supported by the v0.3 package — use the
  v0.2 release assets with `kinglet-bio==0.2.0` if you need them.

## License

[Apache License 2.0](https://github.com/0kam/kinglet/blob/main/LICENSE). Kinglet models are distilled from Perch 2.0 (Apache-2.0, Google);
see [NOTICE](https://github.com/0kam/kinglet/blob/main/NOTICE) and [MODEL_CARD.md](https://github.com/0kam/kinglet/blob/main/MODEL_CARD.md) for training-data provenance.
