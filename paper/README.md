# Code and results behind the Kinglet paper

This folder holds the code, the result files and the figure data behind the paper "Kinglet: distilled
bioacoustic models compatible with Perch 2.0 for laptops and microcontrollers". The rest of the repository
is the `kinglet` package that runs the released models; the scripts here use it (`import kinglet`).

All code in this folder was written with a large language model (Claude, Anthropic; models Fable 5,
Fable 5.2 and Opus 5.5, October–November 2026) under the author's direction and was checked by the author;
each file carries a note to that effect.

## Contents

| Folder | What it holds |
|---|---|
| `common/` | `config.py` (all locations, read from `config.yaml`) and `scoring.py` (benchmark definitions, scoring, cmAP) |
| `data_prep/` | species list, collection of the training recordings, BirdSet / WABAD / non-bird preparation, `jp_list.json` (the 317 Japanese species) |
| `teacher/` | Perch 2.0 outputs: embeddings and logits (`embed_perch.py`), full-vocabulary probability targets (`make_softmax_targets.py`), stored spectrograms (`prepare_mels.py`) |
| `train/` | backbones, the distillation training loop and its drivers, the linear map used in the loss ablation |
| `evaluate/accuracy/` | scoring of Kinglet, BirdNET V2.4 and BirdNET+ V3.0 on WABAD and BirdSet |
| `evaluate/species_lists/` | compact output layers and the species-list analyses (Japan, Germany, Costa Rica) |
| `evaluate/embeddings/` | embedding extraction, retrieval and linear classifiers, compatibility with the Perch 2.0 space |
| `evaluate/nonbird/` | the four datasets of taxa other than birds |
| `evaluate/stats/` | bootstrap intervals and the other statistics reported in the text and the Supporting Information |
| `release/` | export of the released files (ONNX networks, FFT front end, reliability table) |
| `compute_speed/` | multiply-accumulate counts, the laptop speed kit, the GPU benchmark, the check of the released Perch 2.0 model |
| `mcu/` | 8-bit model for the microcontroller, sampling-rate emulation, board tests, power measurement, firmware |
| `figures/` | `build_figure_data.py` (figure CSVs and Table 1 data), `build_si_tables.py` (Tables S1–S9), `R/` (one script per figure), `photo/` |
| `results/` | the result files (JSON, CSV) that the tables, figures and numbers of the paper are read from |
| `figure_data/` | the small CSV files the figures and tables are drawn from |
| `log/experiments_public.csv` | the log of experiments on public data: for each run the change, the hypothesis and decision rule written before the run, the result and the outcome |

Experiment names such as `wabad_v1b` or `packcmap_v1` in comments and result files are rows of
`log/experiments_public.csv`. The three laptops of the speed measurements are named by their CPU
(i7-8565U, M4 Pro, Ryzen AI 7 PRO 350); "desktop" is the workstation (Ryzen 9 9950X3D, RTX 5090) used for
the GPU benchmark.

## Setting the paths

Copy `config.example.yaml` to `config.yaml` and set the locations (relative paths are relative to `paper/`;
each key can also be set by an environment variable of the same name in upper case):

| Key | Content |
|---|---|
| `data_root` | benchmark manifests, teacher outputs on the benchmarks, score caches |
| `distill_root` | teacher targets for training, checkpoints, deployment files, analysis caches |
| `audio_root` | the collected training recordings and taxonomy files |
| `downloads` | downloaded dataset archives (BirdSet) |
| `wabad_src` | the unpacked WABAD download (Zenodo record 17293588) |
| `nonbird_root` | Watkins (BEANS), AnuraSet, HumBugDB and ReefSet |
| `release_assets` | the released model files (ONNX networks, output layers, packs, reliability table) |
| `perch_dir`, `perch_gpu_dir` | the Perch 2.0 SavedModels (CPU and GPU versions, from Kaggle) |

Every script takes its locations from `common/config.py`; none contains an absolute path. Run Python
scripts from `paper/` (`python <folder>/<script>.py`); shell drivers find `paper/` themselves.

## Environments

- **Main** (Python 3.12): torch 2.10, torchaudio 2.10, timm 1.0.30, numpy 2.4, pandas 3.0, scikit-learn 1.8,
  onnxruntime 1.31, tensorflow 2.20 (Perch 2.0), kagglehub, and the `kinglet` package of this repository
  (`pip install -e .` at the repository root). Used for everything below unless stated otherwise.
  Training and the extraction of teacher outputs need a GPU; the figure data, the tables and most analyses
  run on a CPU from the stored caches.
- **8-bit conversion**: a separate environment with litert-torch 0.8.0 (torch 2.9.1 CPU, torchao 0.14.1) for the
  conversion stage of `mcu/export_int8.py`; the Vela compiler (ethos-u-vela 5.2.0) for the Ethos-U55 NPU.
- **Speed kit** (Python 3.10): `compute_speed/speed_kit/requirements.txt`; runs on Linux, macOS and Windows.
- **Firmware and power**: the Seeed Grove Vision AI Module V2 SDK, the Arm GNU toolchain 13.2, and the
  `ppk2-api` package for the Nordic Power Profiler Kit II (see the instructions in `mcu/`).
- **Figures**: R 4.3 with ggplot2, dplyr, tidyr, patchwork, jsonlite, ragg and jpeg.

## Pipeline

The steps in order; each folder's scripts describe their arguments in their docstrings.

1. **Data preparation** (`data_prep/`): build the species list and select the 317 Japanese training species;
   collect the Japanese recordings (xeno-canto, iNaturalist, Macaulay Library); prepare the BirdSet training
   (XCM, XCL) and test sets, WABAD and the non-bird datasets.
2. **Teacher outputs** (`teacher/`): run Perch 2.0 on the training windows (embeddings, full-vocabulary
   probability targets) and on every benchmark window; store the spectrograms the students train on.
3. **Training** (`train/`): distil the three students (B = RegNetY-016, S = EfficientNet-Lite0,
   XS = MobileNetV4-Conv-S), plus two more seeds of XS and the loss ablations.
4. **Scoring** (`evaluate/accuracy/`): score the students, Perch 2.0, BirdNET V2.4 and BirdNET+ V3.0 on WABAD and
   the eight BirdSet test sets; every model is scored on the teacher's 5-s windows and the scores are cached.
5. **Analyses** (`evaluate/species_lists/`, `evaluate/embeddings/`, `evaluate/nonbird/`, `evaluate/stats/`):
   compact output layers and species lists, embeddings, other taxa, bootstrap intervals and the remaining
   statistics.
6. **Release files** (`release/`): export the ONNX networks (waveform and power-spectrum inputs), the output
   layers and packs (`kinglet build-pack`), and the reliability table.
7. **Computation and speed** (`compute_speed/`): count multiply-accumulates and parameters on the distributed
   graphs; measure end-to-end speed on the laptops with the speed kit; GPU throughput.
8. **Microcontroller** (`mcu/`): build and quantise the 8-bit XS model, compile it for the NPU, build and flash
   the firmware, compare board and computer outputs, emulate the board's sampling rate, measure power.
9. **Figures and tables** (`figures/`):

       python figures/build_figure_data.py     # figure_data/*.csv (some tables need the caches of steps 2-8)
       python figures/build_si_tables.py       # figures/si_tables.md (Tables S1-S9)
       Rscript figures/R/fig-02-accuracy-speed.R   # one script per figure; writes figures/fig-*.pdf / .png

   `build_si_tables.py` and the R scripts read only `results/` and `figure_data/`, so they run without any of
   the large intermediate files.

## Where each result comes from

| Result in the paper | Script(s) | Result file(s) |
|---|---|---|
| Table 1 (size, compute, speed, accuracy) | `figures/build_figure_data.py` (`table1_tiers`), `compute_speed/count_macs.py`, `compute_speed/speed_kit/`, `evaluate/stats/revision_stats.py` | `figure_data/table1_tiers.csv`, `benchmarks.csv`, `speed.csv`; `results/count_macs_v1.json`, `count_macs_v2.json`, `revision_stats.json` |
| Fig. 1 | `figures/build_figure_data.py` (`fig1_spectrogram`), `figures/R/fig-01-concept.R` | `figure_data/fig1_spectrogram.csv` |
| Fig. 2a, accuracy vs speed; §3.1 | `evaluate/accuracy/` (`score_student.py`, `score_birdnet_v24.py`, `birdnetv3rel.py`), `build_figure_data.py` (`benchmarks`, `speed`), `figures/R/fig-02-accuracy-speed.R` | `figure_data/benchmarks.csv`, `speed.csv`; `results/distill_g3r_*_results.json`, `distill_g2r_full_results.json`, `birdnetv3rel_v2.json`, `speedkit_i7-8565U.json` |
| Fig. 2b, ratios with intervals; Table S1 | `evaluate/stats/revision_stats.py` (`bootstrap`, `softmax`) | `results/revision_stats.json`, `figure_data/benchmarks.csv` |
| §3.2 species and sites; Figs S1–S2; Table S5 | `build_figure_data.py` (`species_ap`, `wabad_per_site`), `revision_stats.py` | `figure_data/species_ap.csv`, `wabad_per_site.csv`; `results/revision_stats.json` |
| §3.2 training recordings per class; Table S2 | `build_figure_data.py` (`tier_stratified`), `release/export_reliability.py`, `revision_stats.py` | `figure_data/tier_stratified.csv`, `results/revision_stats.json` |
| §3.2 Japanese-list species | `evaluate/stats/revision2_stats.py` (`jpspecies`) | `results/rev2_jpspecies_v1.json` |
| §3.3 compute | `compute_speed/count_macs.py`, `release/fftfront.py`, `compute_speed/perch_release_check/` | `results/count_macs_v1.json`, `count_macs_v2.json`, `fftfront_v1.json`; Table S7 |
| §3.3 laptop speeds; Table S9; Appendix S4 | `compute_speed/speed_kit/` (`speed_kit.py`, `check_head.py`) | `results/speedkit_i7-8565U.json`, `speedkit_M4-Pro.json`, `speedkit_Ryzen-AI-7-PRO-350.json`, `speedkit_Ryzen-AI-7-PRO-350-t8.json`, `check_head_M4-Pro.json` |
| Appendix S4, GPU | `compute_speed/bench_gpu.py`, `compute_speed/gpuspeed_run.sh` | `results/gpuspeed_v1.json` |
| §3.4 compact vs full output layer; Appendix S2 | `evaluate/species_lists/` (`export_headpack.py`, `packfit_compare.py`, `packcmap.py`) | `results/packfit_v1_focal.json`, `packcmap_v1.json` |
| §3.4 species lists on foreign recordings; Table S3; Figs S3–S4 | `evaluate/species_lists/` (`openset_public.py`, `opensetcurve_extract.py`, `opensetcurve.py`, `region_lists.py`, `teacher_chain.sh`, `opensetrel.py`), `evaluate/accuracy/birdnetv3rel.py`, `figures/R/fig-S3-*.R`, `fig-S4-*.R` | `results/opensetcurve_v1.json`, `regionlists_v1.json`, `regionlist_de.csv`, `regionlist_cr.csv`, `opensetrel_v1.json`, `birdnetv3rel_v2.json`; `figure_data/openset_curves.csv` |
| §3.4 SSW share, Costa Rican list, microcontroller layer | `evaluate/stats/revision2_stats.py` (`ssw`, `crlist`, `mcupack`) | `results/rev2_ssw_v1.json`, `rev2_crlist_v1.json`, `rev2_mcupack_v1.json` |
| §3.5 embeddings; Fig. 3a,b; Appendix S3 | `evaluate/embeddings/` (`embed_student.py`, `run_embprobe_extract.sh`, `probe_wabad_emb.py`, `compat_wabad_emb.py`, `probe_l2_post.py`, `rev2_ipretr.py`), `build_figure_data.py` (`embcompat_cos_hist`), `figures/R/fig-03-embeddings.R` | `results/wabad_embprobe_results.json`, `wabad_embcompat_results.json`, `probe_l2_post.json`, `rev2_rawprobe_v1*.json`, `rev2_rawscale_v1*.json`, `rev2_ipretr_v1.json`; `figure_data/embcompat_cos_hist.csv` |
| §3.6 other taxa; Fig. 3c; Table S4 | `data_prep/prepare_nonbird.py`, `evaluate/nonbird/` (`run_nonbird.sh`, `embed_nonbird.py`, `eval_nonbird.py`) | `results/nonbird_v1.json` |
| §3.7 microcontroller; Fig. 4; Table S6; Appendix S5 | `mcu/` (`export_int8.py`, `mcurate.py`, `board_pcm_test.py`, `measure_power.py`, `gen_mel_assets.py`, firmware), `build_figure_data.py` (`table3_mcu`, `mcu_power_trace`), `figures/R/fig-04-mcu.R` | `results/ptqpub_v1.json`, `mcuboard_v1.json`, `mcurate_v1b.json`, `wabad_emb*_results.json` (`XS_int8pub`); `figure_data/table3_mcu.csv`, `mcu_power_trace.csv` |
| Appendix S1, XS seeds and loss ablation; Table S8 | `train/` (`run_xsvar.sh`, `ridge_emb_map.py`), `evaluate/embeddings/` | `results/xsvar_v1_summary.json`, `xsvar_v1_embprobe.json`, `xsvar_v1_embcompat.json`, `distill_xsvar_*_results.json`, `distill_g2r_detach_results.json` |
| Appendix S2, scoring rules and reference models | `common/scoring.py`, `evaluate/accuracy/birdnetv3rel.py`, `revision_stats.py` | `results/birdnetv3rel_v1.json`, `birdnetv3rel_v2.json`, `revision_stats.json` |
| Tables S1–S9 | `figures/build_si_tables.py` | `figures/si_tables.md` |
| Data availability: log of hypotheses | — | `log/experiments_public.csv` |

## What cannot be redistributed

- **Training recordings under a research licence.** About 1.7% of the training recordings come from the
  Macaulay Library under a licence for research use; they are not included and must be requested from the
  Library. The xeno-canto API needs a personal key. Without these recordings the models can be retrained only
  approximately; the released models are provided as files (`release_assets`).
- **Large intermediate files.** The teacher targets for training (about 86 GB of probabilities and about 515 GB
  of teacher outputs), the stored spectrograms, the score caches and the checkpoints are too large for this
  archive. They are regenerated from the public audio and Perch 2.0 by steps 1–4. Everything needed to rebuild
  the tables and figures from the stored results (`results/`, `figure_data/`) is included.
