# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Per-class training coverage of the released Kinglet models: kinglet-v<version>.reliability.csv.

For every class of the Perch 2.0 vocabulary, the number of distinct training recordings whose
labelled species is that class, and a tier (high >= 100, medium 10-99, low 1-9, softlabel_only 0).
The labels are not used in training; the table says how often a class was the subject of a
recording the student heard.

A training species is assigned to a class by the rule the benchmarks use (teacher.embed_perch.perch_classes:
override table, eBird 2021 code, scientific name); a recording listed in several training lists counts once.

  python release/export_reliability.py [OUT]   # default RELEASE_ASSETS/kinglet-v0.3.2.reliability.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PERCH_DIR, RELEASE_ASSETS  # noqa: E402
from common.scoring import EBIRD_TAXONOMY  # noqa: E402
from teacher.embed_perch import perch_classes  # noqa: E402
from train.train_distill import SOURCES  # noqa: E402

TIERS = {"softlabel_only": 0, "low": 1, "medium": 10, "high": 100}  # lower bounds


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RELEASE_ASSETS / "kinglet-v0.3.2.reliability.csv"
    man = pd.concat(pd.read_parquet(p)[["rec_id", "species", "split"]] for p in SOURCES.values())
    man = man[man["split"] == "pool"]
    # the lists spell Xeno-canto ids "xc:123" and "XC123"; one spelling, so a recording counts once
    man["rec"] = man["rec_id"].astype(str).str.replace(r"^(?:xc:|XC)(\d+)$", r"xc\1", regex=True)
    codes = sorted(man["species"].unique())
    man["vocab_index"] = man["species"].map(dict(zip(codes, (c["perch_idx"] for c in perch_classes(codes)))))
    lost = man[man["vocab_index"] < 0]
    print(f"{man['rec'].nunique()} training recordings of {len(codes)} species; without a class: "
          f"{lost['species'].nunique()} species, {lost['rec'].nunique()} recordings")
    n = man[man["vocab_index"] >= 0].drop_duplicates(["rec", "vocab_index"]).groupby("vocab_index").size()

    df = pd.DataFrame({"label": pd.read_csv(PERCH_DIR / "assets/labels.csv").iloc[:, 0]})
    tax = pd.read_csv(EBIRD_TAXONOMY, usecols=["SPECIES_CODE", "SCI_NAME"]).drop_duplicates("SCI_NAME")
    code21 = pd.read_csv(PERCH_DIR / "assets/perch_v2_ebird_classes.csv")["ebird2021"]
    # eBird 2024 code of the class name; the eBird 2021 code Perch lists where the name is not in the 2024 taxonomy
    df["ebird_code"] = df["label"].map(tax.set_index("SCI_NAME")["SPECIES_CODE"]).fillna(
        code21.where(code21 != "no_ebird_code"))
    df["n_train_recordings"] = n.reindex(df.index, fill_value=0)
    df["reliability_tier"] = pd.cut(df["n_train_recordings"], [*TIERS.values(), float("inf")], right=False,
                                    labels=list(TIERS)).astype(str)
    assert df["n_train_recordings"].sum() == man[man["vocab_index"] >= 0].drop_duplicates(["rec", "vocab_index"]).shape[0]
    print(df["reliability_tier"].value_counts().reindex(list(TIERS)[::-1]).to_dict())
    df.to_csv(out, index_label="vocab_index")
    print("wrote", out)


if __name__ == "__main__":
    main()
