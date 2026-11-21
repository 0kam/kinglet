# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Manifests and Perch 2.0 class mapping for the four public non-bird sets (Table S4).

Data under <nonbird_root>/<set>/ (download/README.md; downloaded archives are untrusted: run with python -I).
Per set writes <set>/manifest.parquet (item_id, path, group, label, split) and <set>/classes.json
([{label, sci, perch_idx}], perch_idx = row of Perch 2.0 labels.csv matched by exact scientific
name, -1 if absent). `group` = the source recording used for group-disjoint splits and for removing
the exemplar's recording in retrieval; `split` = the BEANS train/valid/test split where one exists.

  watkins   BEANS (Hagiwara et al. 2023) archive; 31 species after BEANS drops Weddell seal
            (2 clips); group = the 5-character tape prefix of the Watkins record number.
  humbugdb  Zenodo 4904800 (CC BY 4.0); BEANS labels (species with <= 100 clips -> 'others',
            no species -> 'non-mosquito'); group = original file name.
  anuraset  Zenodo 8342596 (CC BY 4.0) raw 1-min recordings + strong labels; items = recordings,
            window labels are built in eval_nonbird.py; group = recording.
  reefset   Zenodo 11071202 (CC BY 4.0) ReefSet v1.0, 1.88-s clips; the metadata has no source
            recording, so group = clip (neighbouring clips of one recording can fall on both sides).

Usage: python -I prepare_nonbird.py [--sets watkins humbugdb anuraset reefset]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config  # noqa: E402

ROOT = config.NONBIRD_ROOT
PERCH_NAMES = pd.read_csv(config.PERCH_DIR / "assets/labels.csv").iloc[:, 0].tolist()

# Watkins (BEANS label) -> scientific name (Watkins Marine Mammal Sound Database species list)
WATKINS_SCI = {
    "Atlantic_Spotted_Dolphin": "Stenella frontalis", "Bearded_Seal": "Erignathus barbatus",
    "Beluga,_White_Whale": "Delphinapterus leucas", "Bottlenose_Dolphin": "Tursiops truncatus",
    "Bowhead_Whale": "Balaena mysticetus", "Clymene_Dolphin": "Stenella clymene",
    "Common_Dolphin": "Delphinus delphis", "False_Killer_Whale": "Pseudorca crassidens",
    "Fin,_Finback_Whale": "Balaenoptera physalus", "Frasers_Dolphin": "Lagenodelphis hosei",
    "Grampus,_Rissos_Dolphin": "Grampus griseus", "Harp_Seal": "Pagophilus groenlandicus",
    "Humpback_Whale": "Megaptera novaeangliae", "Killer_Whale": "Orcinus orca",
    "Leopard_Seal": "Hydrurga leptonyx", "Long-Finned_Pilot_Whale": "Globicephala melas",
    "Melon_Headed_Whale": "Peponocephala electra", "Minke_Whale": "Balaenoptera acutorostrata",
    "Narwhal": "Monodon monoceros", "Northern_Right_Whale": "Eubalaena glacialis",
    "Pantropical_Spotted_Dolphin": "Stenella attenuata",
    "Rough-Toothed_Dolphin": "Steno bredanensis", "Ross_Seal": "Ommatophoca rossii",
    "Short-Finned_Pacific_Pilot_Whale": "Globicephala macrorhynchus",
    "Southern_Right_Whale": "Eubalaena australis", "Sperm_Whale": "Physeter macrocephalus",
    "Spinner_Dolphin": "Stenella longirostris", "Striped_Dolphin": "Stenella coeruleoalba",
    "Walrus": "Odobenus rosmarus", "Weddell_Seal": "Leptonychotes weddellii",
    "White-beaked_Dolphin": "Lagenorhynchus albirostris",
    "White-sided_Dolphin": "Lagenorhynchus acutus",
}


# ReefSet labels naming a species that has a Perch 2.0 class (the other species labels -
# mycbon, pomamb, holocentrus, midshipman, grouper - have none)
REEF_SCI = {"bioph_megnov": "Megaptera novaeangliae"}


def beans_split(df: pd.DataFrame) -> pd.Series:
    """BEANS scripts/{watkins,humbugdb}.py: stratified 6:2:2 with random_state=42."""
    tr, vt = train_test_split(df, test_size=0.4, random_state=42, shuffle=True, stratify=df["label"])
    va, te = train_test_split(vt, test_size=0.5, random_state=42, shuffle=True, stratify=vt["label"])
    split = pd.Series("", index=df.index)
    split[tr.index], split[va.index], split[te.index] = "train", "valid", "test"
    return split


def classes(labels: list[str], sci: dict[str, str]) -> list[dict]:
    return [{"label": c, "sci": sci.get(c), "perch_idx": PERCH_NAMES.index(sci[c]) if sci.get(c) in PERCH_NAMES else -1}
            for c in labels]


def watkins():
    d = ROOT / "watkins/data"
    df = pd.read_csv(d / "annotations.csv")
    df = df[df["species"] != "Weddell_Seal"].reset_index(drop=True)  # as BEANS
    df = pd.DataFrame({"item_id": df["path"].str.replace("/", "__"), "path": [str(d / p) for p in df["path"]],
                       "group": df["path"].str.split("/").str[1].str[:5], "label": df["species"]})
    df["split"] = beans_split(df)
    return df, classes(sorted(df["label"].unique()), WATKINS_SCI)


def humbugdb():
    d = ROOT / "humbugdb"
    df = pd.read_csv(d / "neurips_2021_zenodo_0_0_1.csv")
    lab = df["species"].fillna("non-mosquito")
    vc = lab.value_counts()
    lab = lab.replace(list(vc[vc <= 100].index), "others")  # as BEANS
    df = pd.DataFrame({"item_id": df["id"].astype(str), "path": [str(d / f"audio/{i}.wav") for i in df["id"]],
                       "group": df["name"].astype(str), "label": lab})
    df["split"] = beans_split(df)
    return df, classes(sorted(df["label"].unique()), {})  # no mosquito species in the Perch vocabulary


def anuraset():
    d = ROOT / "anuraset"
    sp = pd.read_csv(d / "species.csv")
    paths = sorted((d / "raw").rglob("*.wav"))
    df = pd.DataFrame({"item_id": [p.stem for p in paths], "path": [str(p) for p in paths]})
    df["group"], df["label"], df["split"] = df["item_id"], "", ""
    strong = {p.stem for p in (d / "strong").rglob("*.txt")}
    assert set(df["item_id"]) >= strong, f"{len(strong - set(df['item_id']))} labelled recordings without audio"
    df = df[df["item_id"].isin(strong)].reset_index(drop=True)
    return df, classes(sp["CODE"].tolist(), dict(zip(sp["CODE"], sp["SPECIES"])))


def reefset():
    d = ROOT / "reefset/data/ReefSet_v1.0"
    ann = pd.DataFrame(json.loads((d / "reefset_annotations.json").read_text()))
    df = pd.DataFrame({"item_id": ann["id"].astype(str), "path": [str(d / "full_dataset" / f) for f in ann["file_name"]],
                       "group": ann["id"].astype(str),  # no source-recording field in the metadata
                       "label": ann["label"], "split": ""})
    return df, classes(sorted(df["label"].unique()), REEF_SCI)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=["watkins", "humbugdb", "anuraset", "reefset"])
    args = ap.parse_args()
    for s in args.sets:
        df, cls = globals()[s]()
        missing = [p for p in df["path"] if not Path(p).exists()]
        assert not missing, f"{s}: {len(missing)} missing files, e.g. {missing[:2]}"
        df.to_parquet(ROOT / s / "manifest.parquet", index=False)
        (ROOT / s / "classes.json").write_text(json.dumps(cls, indent=1))
        (ROOT / s / "perch_idx.json").write_text(json.dumps([c["perch_idx"] for c in cls if c["perch_idx"] >= 0]))
        print(f"{s}: {len(df)} items, {df['group'].nunique()} groups, {len(cls)} classes, "
              f"{sum(c['perch_idx'] >= 0 for c in cls)} mapped to Perch", flush=True)


if __name__ == "__main__":
    main()
