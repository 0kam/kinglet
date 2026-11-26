# Kinglet 🐦

**Kinglet-sized bioacoustic embedding and classification models, distilled from
[Perch 2.0](https://github.com/google-research/perch).**

Kinglets are among the smallest birds in the northern hemisphere. Kinglet models are among the
smallest bioacoustic models: a trunk ONNX (~22–53 MB, preprocessing included) plus a regional
head pack (26–38 MB exact, ~2 MB compact) that process **months of field recordings in hours on an ordinary laptop CPU** — no GPU,
no Python ML stack — while staying close to Perch 2.0 on public soundscape benchmarks from 72 sites
worldwide. Each model outputs species scores for **any regional species list**, sound-group scores
(human voice, vehicles, rain/wind, ...), and a 1536-d embedding in the Perch 2.0 embedding space,
so search indices and classifiers built on Perch embeddings keep working.

## Models (v0.4.0)

| model | backbone | trunk size | speed on a laptop* | 1,000 h of audio* | WABAD cmAP** | BirdSet cmAP vs Perch 2.0*** |
|---|---|---|---|---|---|---|
| Kinglet-B | RegNetY-016 | 51 MB | 158x realtime | 6.3 h | 0.283 (97%) | 89–103% |
| Kinglet-S | EfficientNet-Lite0 | 26 MB | 307x realtime | 3.3 h | 0.269 (93%) | 86–97% |
| Kinglet-XS | MobileNetV4-Conv-S | 22 MB | 565x realtime | 1.8 h | 0.259 (89%) | 82–96% |
| *Perch 2.0 (teacher)* | | | 10.7x realtime | 93 h | 0.291 | |

Since v0.3 the models are distilled from the teacher's **full 14,795-class vocabulary** and folded
to a region at deployment time ("distill once, fold anywhere"): a prebuilt Japan pack (317 species)
ships with the release, and `kinglet build-pack` folds the same trunk onto **any species list** in
seconds, no audio needed. Scores are **softmax probabilities with an explicit background channel**;
use a fixed species threshold of **0.1** (the Perch operating point). Every pack also carries
**sound groups** with operating thresholds (since v0.3.1): a `human_voice` flag (threshold 0.005)
that catches clear voices — about half of the audible voices in the WABAD soundscapes, so not a
privacy mask on its own — `gunshot` and `domestic_animal` flags, and `geophony` / `vehicle` flags
for rain-and-traffic noise screening.
The last two are **provisional features with limited performance** — hints for review, not
detections (details in [MODEL_CARD.md](https://github.com/0kam/kinglet/blob/main/MODEL_CARD.md)).

\* End to end — audio decoding, windowing and the model with the exact Japan pack — on 4 CPU
threads of a 2018 business laptop (Intel Core i7-8565U), 10-minute 32 kHz WAV. On the same laptop
BirdNET V2.4 runs at 66x and BirdNET+ V3.0 at 34x realtime (both on consecutive 3-s segments, as
used by default). Over three laptops (this one, an Apple M4 Pro and an AMD Ryzen AI 7 PRO 350),
Kinglet-B, -S and -XS ran 8–15x, 17–29x and 21–53x faster than Perch 2.0; per-machine values are in
[MODEL_CARD.md](https://github.com/0kam/kinglet/blob/main/MODEL_CARD.md).

\*\* Class-mean average precision on [WABAD](https://zenodo.org/records/17293588) (60,158 5-s
windows, 1,189 species, 72 sites on six continents), in parentheses relative to Perch 2.0; per site,
Kinglet-B stays within ±0.06 AUPRC of Perch 2.0 at all 72 sites.

\*\*\* Range over the seven [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet)
test sets (POW, HSN, NES, PER, SNE, SSW, UHH); POW and HSN served as development gates, on the
other five the ranges are 95–103% / 92–97% / 90–96%.

Both benchmarks are scored by folding the full-vocabulary output onto each benchmark's species
list; none of their recordings were used in training.

Download from [GitHub Releases](https://github.com/0kam/kinglet/releases). Each tier comes in three
trunk forms with the same outputs: `.power.onnx` takes the power spectrogram, which the package
computes with numpy's FFT (use this one from Python); `.onnx` takes the waveform and computes the
spectrogram inside the graph (for runtimes without numpy); `.gemm-mel.onnx` is a waveform variant
whose mel frontend is faster on some ARM CPUs.

## Quickstart

```bash
pip install kinglet-bio   # needs ffmpeg on PATH
kinglet build-pack --headpack kinglet-s-v0.4.0.headpack.npz --species kinglet-v0.4.0.species-jp.csv --out jp.npz
kinglet analyze recordings/ --model kinglet-s-v0.4.0.power.onnx --pack jp.npz --out detections.csv
```

`build-pack` writes an **exact** pack by default: scores are the full 14,795-way softmax read at
the listed species, at almost the same speed. `--compact` writes the ~2 MB approximate pack
(ridge-fitted "other" row; it misses about one in ten exact-fold detections with a site-specific
list, see MODEL_CARD.md) for microcontrollers and low-memory devices; the prebuilt
`.pack-jp.npz` is such a compact pack.

The CLI keeps, per 5 s window, the top-3 species scoring at least 0.1 (adjustable with
`--top-k` / `--min-score`), adds one row per calibrated sound group that passes its own threshold
(`human_voice` marks windows with a clear voice), and can dump embeddings with
`--embeddings emb/`.

Deploying outside Japan (or with your own species list)? Fold a pack from the same `.headpack.npz`
— the CSV needs `ebird_code, scientific_name` columns:

```bash
kinglet build-pack --headpack kinglet-s-v0.4.0.headpack.npz --species my_region.csv --out my_region.npz
```

Or let eBird supply the list for any region (free key at https://ebird.org/api/keygen, via `--ebird-key`
or `EBIRD_API_KEY`); it is saved as `us-ca.species.csv` to edit and rerun with `--species`:

```bash
kinglet build-pack --headpack kinglet-s-v0.4.0.headpack.npz --region US-CA --out us-ca.npz
```

Or from Python / any ONNX runtime:

```python
from kinglet import Kinglet, decode, windows
m = Kinglet("kinglet-s-v0.4.0.power.onnx", "jp.npz")
starts, wins = windows(decode("field_recording.wav"))   # 5 s windows at 32 kHz mono
embeddings, scores, groups = m.run(wins)                # (n, 1536) unit length, (n, 317), {group: (n,)}
voice = groups["human_voice"] >= m.groups["human_voice"] # windows with a clear human voice
```

## Model I/O

- Input `waveform`: float32 `(batch, 160000)` — 5 s, 32 kHz, mono, amplitude in [-1, 1]; or, for the
  `.power.onnx` trunks, `power`: float32 `(batch, 501, 513)` — the power spectrogram of that window,
  `kinglet.power_spectrogram(waveform)` (1024-point FFT, hop 320, periodic Hann, centred frames with
  reflect padding). Same network and outputs; in Python the FFT is faster than the waveform trunk's
  in-graph DFT matrix product. `Kinglet` accepts either trunk.
- Output `embedding`: `(batch, 1536)` — projection into the Perch 2.0 embedding space. Since v0.4.0
  `Kinglet.run`, `kinglet.embedding_from_feature` and `kinglet analyze --embeddings` return it
  **L2-normalised** (unit length). The network's raw output (the ONNX `embedding` tensor, or
  `normalize=False`) is about 8–12 times as long as a Perch 2.0 embedding and varies in length
  between windows: ranked by raw inner product, as the Perch Hoplite search index ranks, Kinglet
  retrieval on WABAD lost up to 8.5 points of its ratio to Perch, and classifiers fitted on raw
  Kinglet embeddings with a fixed penalty kept 86–91% of Perch's cmAP instead of 96–101%. Unit
  length removes both; cosine similarities do not change. On unseen soundscapes (WABAD) the mean
  cosine to the Perch 2.0 embedding of the same window is 0.87 (B) / 0.86 (S) / 0.83 (XS). Mixing
  the two models in few-shot retrieval (Perch exemplars, Kinglet archive, or the reverse) keeps
  93–100% of Perch's own retrieval mAP; a linear classifier trained on L2-normalised Perch
  embeddings and applied unchanged to Kinglet embeddings keeps 95–100% of its cmAP. When you
  combine the two models (a shared archive, a classifier trained on Perch), L2-normalise the
  Perch 2.0 embeddings too.
- Output `feature`: `(batch, d)` — the trunk feature; scores = `softmax(feature @ W.T + b)` with
  `W, b` from the pack `.npz`, background column dropped (this is what `Kinglet.run` does).
  Row order: pack `codes` (species, then group-member classes, then "other"). An exact pack
  stores the full head as `W, b` and the full-head column of each pack row in `cols`; its scores
  are `softmax(feature @ W.T + b)[:, cols]`.
- A device that stores the trunk feature instead of the embedding (e.g. a microcontroller build
  logging the int8 feature, 1 byte per dimension) loses nothing: the embedding is one linear layer
  on top of the feature, and `kinglet.embedding_from_feature` applies it on the host
  (`pip install kinglet-bio[mcu]`):

  ```python
  import kinglet
  emb = kinglet.embedding_from_feature("kinglet-xs-v0.4.0.power.onnx", feature_int8, scale, zero_point)  # (n, 1536)
  ```

  `scale` and `zero_point` are the quantisation parameters of the device model's feature tensor
  (omit both for float features). Measured for Kinglet-XS with an int8 trunk on the 60,158 WABAD
  windows: cosine 0.990 between the restored and the fp32 embedding, and few-shot retrieval,
  linear-probe and cross-space readouts all within 1.5% of the fp32 model.
- v0.2 single-file models (`logits` in the graph) are not supported by the v0.3 package — use the
  v0.2 release assets with `kinglet-bio==0.2.0` if you need them.

## Tests and demonstration data

The unit tests need no model download and run in a few seconds (`onnx` from the `mcu` extra builds the toy
models; ffmpeg on PATH for the decoding tests):

```bash
pip install -e ".[mcu]"
python -m unittest discover tests -v
```

They check, against hand-computed values or exact identities: the FFT front end against a direct DFT
(`test_frontend.py`); decoding, 5-s windows and the `kinglet analyze` CSV, with the 0.1 threshold, the top-k
rule, sound-group thresholds and the embedding files (`test_analyze.py`); `kinglet build-pack`, the exact pack
against the full softmax, and the compact pack against Eq. (S3) of the paper's Supporting Information and against the exact pack when its
"other" row is exact (`test_region.py`); the embedding recovered from a stored feature, float and int8, and
L2 normalisation (`test_restore.py`); and the scoring functions of `paper/` (class-mean average precision,
the site-weighted AP of the site bootstrap, the paired species bootstrap, false detections at equal recall;
`test_paper_scoring.py`, skipped unless `pandas scikit-learn scipy torch` are installed). GitHub Actions runs
them on every push (`.github/workflows/tests.yml`).

**Demonstration.** `demo/audio/` holds three 20-s excerpts of WABAD soundscapes from Germany and Poland
(source, licence and changes: `demo/ATTRIBUTION.md`) and `demo/annotations.csv` the experts' annotations of
them. From a clean clone (about 140 MB download):

```bash
git clone -b paper-code https://github.com/0kam/kinglet.git && cd kinglet
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[mcu]"                          # needs ffmpeg on PATH
mkdir -p demo/models
for f in kinglet-s-v0.4.0.power.onnx kinglet-s-v0.4.0.headpack.npz; do
  curl -L -o demo/models/$f https://github.com/0kam/kinglet/releases/download/v0.4.0/$f
done
kinglet build-pack --headpack demo/models/kinglet-s-v0.4.0.headpack.npz \
  --species paper/results/regionlist_de.csv --out demo/models/de.npz      # exact pack, 493 German species
kinglet analyze demo/audio --model demo/models/kinglet-s-v0.4.0.power.onnx \
  --pack demo/models/de.npz --out demo_detections.csv
python -m unittest discover tests -p test_demo.py -v   # reruns both steps, compares with demo/expected_detections.csv
```

`demo_detections.csv` should hold the 19 rows of `demo/expected_detections.csv` (scores may differ in the last
digit between machines; the test allows 0.01). Next to `demo/annotations.csv`: in the Polish excerpt Common
Chiffchaff and Song Thrush are detected and Common Chaffinch is missed; in `SCHF` Song Thrush is detected in all
four windows and European Robin in two of its three, Eurasian Blackcap is missed, and Eurasian Blackbird and Great
Bittern are reported without an annotation; in `SCHG` Eurasian Skylark is detected in three of four windows,
Common Nightingale is missed and Greylag Goose is reported without an annotation. The excerpts show how to run
the tool, not its accuracy, which the paper measures on all of WABAD and BirdSet.

The package (`kinglet/`), the tests, the files in `demo/` and the test workflow were written with a large
language model (Claude, Anthropic; October–November 2026) under the author's direction, as was the code in
`paper/` (see `paper/README.md`). The Python files of the package, of the tests and of `paper/` and the test
workflow each carry a note to that effect.

## License

[Apache License 2.0](https://github.com/0kam/kinglet/blob/main/LICENSE). Kinglet models are distilled from Perch 2.0 (Apache-2.0, Google);
see [NOTICE](https://github.com/0kam/kinglet/blob/main/NOTICE) and [MODEL_CARD.md](https://github.com/0kam/kinglet/blob/main/MODEL_CARD.md) for training-data provenance.
