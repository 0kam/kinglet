# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Convert WABAD (Zenodo 17293588) into an evaluation set of 5-s windows.

The Recordings/*.wav of each site zip (1 min to several hours) are cut into 5-s window wavs, and the
boxes of "Pooled annotations.csv" become window labels: the species of every box that overlaps the
window (any overlap). Only complete 5-s intervals from the start are used (the trailing remainder is
dropped), so the 1.5 % of boxes that fall only on a remainder are not evaluated (plus one box of
BERB_20130206_061400 at t = 5616 s in a 1200-s recording, an error in the source data). Scientific
names map to eBird v2024 codes (SYNONYMS resolves 29 genus changes). The output has the format of the
BirdSet test sets:

  <data_root>/wabad/win5s/<SITE>/<rec>_<k>.wav   (cut at the native sample rate)
  <data_root>/wabad/test.parquet                 (window_id, path, labels, site)

Input: the WABAD download in `wabad_src` of paper/config.yaml (download/download_zenodo.sh wabad) and
<audio_root>/taxonomy/ebird_taxonomy_v2024.csv.

Usage: python prepare_wabad.py [--src DIR] [--workers 6]
"""

import argparse
import sys
import tempfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config  # noqa: E402

ROOT = config.DATA_ROOT

WIN = 5.0
# WABAD scientific name -> eBird v2024 species code, where the direct SCI_NAME match fails
# (genus reassignments / lumps between the WABAD taxonomy and Clements v2024)
SYNONYMS = {
    "Acanthis cabaret": "redpol1",          # redpoll lump
    "Accipiter haplochrous": "necgos1",     # Tachyspiza haplochroa
    "Anurolimnas viridis": "ruccra1",       # Rufirallus viridis
    "Brachypodius melanocephalos": "blhbul1",
    "Bubulcus ibis": "categr1",             # Ardea ibis
    "Campethera nivosa": "buswoo1",         # Pardipicus nivosus
    "Chrysococcyx lucidus": "shbcuc1",      # Chalcites lucidus
    "Columba larvata": "lemdov2",           # Aplopelia larvata
    "Corvus monedula": "eurjac",            # Coloeus monedula
    "Dicaeum agile": "thbflo1",             # Pachyglossa agilis
    "Dicrurus divaricatus": "fotdro5",      # lumped in Dicrurus adsimilis
    "Empidonax occidentalis": "wesfly",     # Western Flycatcher lump
    "Eopsaltria flaviventris": "yebrob1",   # Cryptomicroeca flaviventris
    "Habia fuscicauda": "rtatan1",          # Driophlox fuscicauda
    "Herpsilochmus sellowi": "caaant1",     # Radinopsyche sellowi
    "Lanius corvinus": "yebshr1",           # Corvinella corvina
    "Mirafra rufocinnamomea": "flalar1",    # Amirafra rufocinnamomea
    "Neocossyphus fraseri": "rufthr1",      # Stizorhina fraseri
    "Oreolais rufogularis": "butapa1",      # Apalis rufogularis
    "Phaeomyias murina": "y01204",          # Nesotriccus murinus
    "Phyllomyias nigrocapillus": "blctyr1",  # Tyranniscus nigrocapillus
    "Phylloscartes ophthalmicus": "mfbtyr1",  # Pogonotriccus ophthalmicus
    "Platysteira castanea": "chweye1",      # Dyaphorophyia castanea
    "Psilorhinus morio": "brnjay",          # Cyanocorax morio
    "Reinwardtipicus validus": "orbwoo1",   # Chrysocolaptes validus
    "Streptopelia chinensis": "spodov",     # Spilopelia chinensis
    "Streptopelia senegalensis": "laudov1",  # Spilopelia senegalensis
    "Telophorus multicolor": "macbus2",     # Chlorophoneus multicolor
    "Trachyphonus purpuratus": "yebbar5",   # Trachylaemus purpuratus
}
TAXONOMY = config.AUDIO_ROOT / "taxonomy/ebird_taxonomy_v2024.csv"


def slice_zip(args: tuple[Path, Path]) -> list[tuple[str, str, str, int]]:
    """One site zip -> 5-s window wavs; returns (site, rec, window_id, window_idx) rows."""
    zip_path, out_root = args
    site = zip_path.stem
    dst = out_root / site
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        wavs = [n for n in z.namelist() if n.startswith("Recordings/") and n.lower().endswith(".wav")]
        for name in sorted(wavs):
            rec = Path(name).name
            audio, sr = sf.read(z.extract(name, tmp), dtype="int16")
            if audio.ndim > 1:
                audio = audio[:, 0]
            step = int(WIN * sr)
            for k in range(max(1, len(audio) // step)):
                wid = f"{Path(rec).stem}_{k:02d}"
                p = dst / f"{wid}.wav"
                if not p.exists():
                    tmp_p = p.with_suffix(".tmp.wav")
                    sf.write(tmp_p, audio[k * step:(k + 1) * step], sr, format="WAV", subtype="PCM_16")
                    tmp_p.rename(p)
                rows.append((site, rec, wid, k))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(config.WABAD_SRC))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    src, out_root = Path(args.src), ROOT / "wabad/win5s"

    tax = pd.read_csv(TAXONOMY)
    sci2code = dict(zip(tax[tax["CATEGORY"] == "species"]["SCI_NAME"], tax[tax["CATEGORY"] == "species"]["SPECIES_CODE"]))
    ann = pd.read_csv(src / "Pooled annotations.csv", encoding="latin-1")
    ann["code"] = ann["Species"].map(lambda s: SYNONYMS.get(s) or sci2code[s])

    zips = sorted(src.glob("*.zip"))
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.imap_unordered(slice_zip, [(z, out_root) for z in zips]) for r in rs]
    win = pd.DataFrame(rows, columns=["site", "rec", "window_id", "k"]).sort_values(
        ["site", "window_id"], ignore_index=True)
    assert win["window_id"].is_unique, "duplicate window ids across sites"

    # window labels: any overlap between the box [b, e) and the window [5k, 5k+5)
    ann["k_lo"] = (ann["Begin_Time_(s)"] // WIN).astype(int)
    ann["k_hi"] = ((ann["End_Time_(s)"].clip(lower=ann["Begin_Time_(s)"] + 1e-6) - 1e-9) // WIN).astype(int)
    lab = ann.loc[ann.index.repeat(ann["k_hi"] - ann["k_lo"] + 1)]
    lab["k"] = lab.groupby(level=0).cumcount() + lab["k_lo"]
    lab = lab.drop_duplicates(["Recording", "k", "code"])
    by_win = lab.groupby(["Recording", "k"])["code"].agg(sorted)
    win["labels"] = [by_win.get((r, k), []) for r, k in zip(win["rec"], win["k"])]
    win["path"] = [str(out_root / s / f"{w}.wav") for s, w in zip(win["site"], win["window_id"])]

    orphan = set(zip(lab["Recording"], lab["k"])) - set(zip(win["rec"], win["k"]))
    win[["window_id", "path", "labels", "site"]].to_parquet(ROOT / "wabad/test.parquet", index=False)
    sp = {c for x in win["labels"] for c in x}
    print(f"WABAD: {len(win)} windows, {win['site'].nunique()} sites, {len(sp)} species, "
          f"{sum(len(x) == 0 for x in win['labels'])} empty windows, "
          f"{len(orphan)} annotation windows without audio: {sorted(orphan)[:5]}")


if __name__ == "__main__":
    main()
