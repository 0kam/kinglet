# Dataset downloads

Locations come from `paper/config.yaml` (`downloads`, `wabad_src`, `nonbird_root`).

| Data | How |
|---|---|
| BirdSet test sets (HSN, NBP, NES, PER, POW, SNE, SSW, UHH), XCM, XCL | `bash download_birdset.sh <SUBSET>` (Hugging Face `DBD-research-group/BirdSet`, revision `data`) |
| WABAD (Zenodo 17293588) | `bash download_zenodo.sh wabad` |
| The four non-bird sets (Table S4) | manual, see below |

`zenodo_fetch.py RECORD_ID DEST [FILE ...]` is the chunked, md5-verified Zenodo downloader used by
`download_zenodo.sh`; it can also fetch the three Zenodo records below.

## Non-bird datasets (manual)

Unpack each into `<nonbird_root>/<set>/` so that the paths read by `../prepare_nonbird.py` exist,
then run `python -I prepare_nonbird.py` (the archives are third-party data: `-I` keeps Python from
importing modules out of the data folders).

| Set | Source | Files expected under `<nonbird_root>/<set>/` |
|---|---|---|
| `watkins` | Watkins Marine Mammal Sound Database as packaged by the BEANS benchmark (Hagiwara et al. 2023; https://github.com/earthspecies/beans, `scripts/watkins.py`) | `data/annotations.csv` and `data/<species>/<clip>.wav` |
| `humbugdb` | HumBugDB, Zenodo 4904800 (CC BY 4.0): `humbugdb_neurips_2021_{1,2,3,4}.zip` and `neurips_2021_zenodo_0_0_1.csv` | `neurips_2021_zenodo_0_0_1.csv`, `audio/<id>.wav` |
| `anuraset` | AnuraSet, Zenodo 8342596 (CC BY 4.0): `raw_data.zip`, `strong_labels.zip`, `species.csv` | `species.csv`, `raw/**/*.wav`, `strong/**/*.txt` |
| `reefset` | ReefSet v1.0, Zenodo 11071202 (CC BY 4.0): `ReefSet_v1.0.zip` | `data/ReefSet_v1.0/reefset_annotations.json`, `data/ReefSet_v1.0/full_dataset/*` |
