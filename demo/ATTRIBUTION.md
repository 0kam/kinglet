# Demonstration recordings: source, licence and changes

The three recordings in `audio/` are excerpts of soundscapes from WABAD, the World Annotated Bird Acoustic
Dataset for Passive Acoustic Monitoring:

> Pérez-Granados, C., Morant, J., Darras, K., et al. (2025). WABAD: A World Annotated Bird Acoustic Dataset for
> Passive Acoustic Monitoring. Zenodo. https://doi.org/10.5281/zenodo.17293588

| File | Source recording (WABAD site, archive) | Site |
|---|---|---|
| `SCHF_20230422_045610_0-20s.flac` | `Recordings/SCHF_20230422_045610.wav` in `SCHF.zip` | Schorfheide-Chorin, Germany |
| `SCHG_20160501_034000_0-20s.flac` | `Recordings/SCHG_20160501_034000.wav` in `SCHG.zip` | Schorfheide-Chorin, Germany |
| `BIAL_20210420_071504_0-20s.flac` | `Recordings/BIAL_20210420_071504.wav` in `BIAL.zip` | Białowieża forest district, Poland |

**Licence.** The Zenodo record's licence field is Creative Commons Attribution 4.0 International (CC BY 4.0);
the record's description states Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
These excerpts and `annotations.csv` are redistributed under the stricter of the two: credit the dataset as
above and use them for non-commercial purposes only. They are not covered by the Apache-2.0 licence of the
rest of this repository.

**Changes.** Seconds 0–20 of each recording, converted from 48 kHz to 32 kHz mono, 16-bit FLAC, file metadata
removed. `annotations.csv` holds the expert annotations of these 20 s from WABAD's `Pooled annotations.csv`
(times clipped to the excerpt; eBird codes added from `paper/results/regionlist_de.csv`).

**Selection.** Chosen from the annotations alone, before running any model: per site, the first 20-s excerpt
(in recording order, starting at a multiple of 20 s) with two or three annotated species, all on the German
species list, and at least one annotation in each of its four 5-s windows.

`expected_detections.csv` is the output of `kinglet analyze` with Kinglet-S v0.4.0 and the exact German pack
(see README.md, "Tests and demonstration data").
