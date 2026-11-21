# Perch 2.0 release check

Question (Appendix S4): does any public Perch 2.0 release, official or third-party, provide the linear class
head trained alongside the prototype head, or another cheap head?

Answer: no.

Official releases:
- Kaggle `google/bird-vocalization-classifier`: `perch_v2` (version 2, GPU) and `perch_v2_cpu` (version 1) are
  the only Perch 2.0 instances. Their `variables.data` files are byte-identical.
- The only 14,795-sized tensors are bias [14795], protop_kernel [14795, 4] and prototypes
  [14795, 1536, 4] (`heads_protopnet_logits_*`). No (1536, 14795) tensor exists.
- Outputs: embedding, label, spatial_embedding [16, 4, 1536], spectrogram.
- `label` is the ProtoPNet head (google-research/perch, chirp/models/heads.py, ProtoPNetHead):
  cosine similarity of the 64 spatial embeddings to the prototypes, max over positions,
  times relu(protop_kernel), summed, plus bias.
  `recompute.py` reproduces `label` from `spatial_embedding` and the checkpoint (max abs difference 3.8e-6).
- The Kaggle model card states that "the species classification head adds an additional 91 million
  parameters". The prototype head has 90.97 M parameters; a linear head would have 22.7 M.

Third-party derivatives (29 Hugging Face repositories, BirdNET-Analyzer / birdnet, BirdNET-Go,
bioacoustics-model-zoo, Kaggle datasets, GitHub search) fall into four groups:
- full conversions (about 410 MB, same parameters);
- quantised versions;
- embedding-only exports with no head (e.g. sammlapp/Perch_v2_headless, 11.2 M parameters,
  output `embedding` only);
- regional subsets of the prototype head (tphakala/Perch-v2-Models; Japan version
  label [799], largest head tensor [1536, 3196]).

One linear approximation was tried and not released (pengowray/oversample, research/perch.md:
top class right in 5 of 22 windows).

Not checked: Kaggle notebooks and write-ups (they need authentication / JavaScript), BirdCLEF 2026
working notes, Raven / Koogu / birdnetlib.

Scripts (run with `python -I` on a downloaded SavedModel directory or ONNX file):
- `listvars.py`: variables and shapes
- `sigs.py`: signatures and outputs
- `recompute.py`: recomputes `label`
- `onnxinfo.py`: ONNX outputs and largest tensors
