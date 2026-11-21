# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Shared scoring core for the public benchmarks (WABAD and the eight BirdSet test sets).

Every model is scored on the same 5-s windows as the Perch 2.0 teacher (teacher/embed_perch.py: same decode and
windows; evaluate_set asserts that the (recording ID, window_idx) order equals the teacher's when the teacher's
output directory is complete). Species are matched to a model's classes by scientific name (label_index); a species
the model does not have scores MISSING and stays in the average.

Cache layout: CACHE/<model name>_<set>.npz with ids, window_idx, t_start, scores (windows x species), codes
(the set's annotated eBird codes, sorted) and unmatched (codes without a class). infer() skips a cached set.

Metrics (evaluate_set): cmAP = mean over annotated species of the average precision across all windows (cmap);
macro ROC-AUC over annotated species; pooled AUPRC over all window-species pairs.

BirdNET V2.4 (class V24) is scored by evaluate/accuracy/score_birdnet_v24.py.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import AUDIO_ROOT, DATA_ROOT  # noqa: E402
from teacher.embed_perch import SPECIES_LIST, SR, decode, windows  # noqa: E402

CACHE = DATA_ROOT / "birdnet_v3/cache_v2"
# BirdNET V2.4 FP32 TFLite model as installed by the `birdnet` package (model-fp32.tflite, labels/en_us.txt)
V24_DIR = DATA_ROOT / "birdnet_v3/models_v24"
EBIRD_TAXONOMY = AUDIO_ROOT / "taxonomy/ebird_taxonomy_v2024.csv"
BIRDSET = ("POW", "HSN", "NES", "PER", "SNE", "SSW", "UHH", "NBP")
# set -> (manifest, id column, offset column, max seconds, hop, teacher output dir), all under DATA_ROOT
SETS = {
    **{s: (f"birdset/{s}_test.parquet", "window_id", None, 0, 5.0, f"birdset/emb_{s}_test") for s in BIRDSET},
    "WABAD": ("wabad/test.parquet", "window_id", None, 0, 5.0, "wabad/emb_test"),
}
MISSING = -1e9
CHUNK = 256
BATCH = 32


def load_parts(d: Path, allow_partial: bool = False,
               keys=("ids", "window_idx", "t_start", "emb", "logits")) -> dict[str, np.ndarray]:
    """Concatenate the given arrays over the part-*.npz files of a teacher or student output directory."""
    assert allow_partial or (d / "DONE").exists(), f"{d} is incomplete (no DONE)"
    parts = [np.load(p) for p in sorted(d.glob("part-*.npz")) if ".tmp" not in p.name]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def cmap(scores: np.ndarray, y: np.ndarray) -> float:
    """Class-mean average precision over the species with at least one positive window."""
    keep = y.sum(0) > 0
    return float(np.mean([average_precision_score(y[:, j], scores[:, j]) for j in np.where(keep)[0]]))


class V24:
    """BirdNET V2.4 as a logit function on 5-s windows at 32 kHz."""
    tag = "v24"  # key of BIRDNET_OVERRIDES

    def __init__(self, threads: int):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=str(V24_DIR / "model-fp32.tflite"), num_threads=threads)
        self.inp, self.outp = self.it.get_input_details()[0]["index"], self.it.get_output_details()[0]["index"]
        self.labels = (V24_DIR / "labels/en_us.txt").read_text().splitlines()
        self.sci = [s.split("_", 1)[0] for s in self.labels]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(B, 5 s at 32 kHz) -> mean logits of the 0-3/1-4/2-5 s crops at 48 kHz (B, 6522)."""
        from scipy.signal import resample_poly

        crops = np.concatenate([x[:, s * SR : (s + 3) * SR] for s in range(3)])
        crops = resample_poly(crops, 3, 2, axis=1).astype(np.float32)
        self.it.resize_tensor_input(self.inp, crops.shape)
        self.it.allocate_tensors()
        self.it.set_tensor(self.inp, crops)
        self.it.invoke()
        return self.it.get_tensor(self.outp).reshape(3, len(x), -1).mean(0)


def sci_names() -> pd.Series:
    """eBird code -> scientific name (species_list first, then eBird taxonomy v2024)."""
    sl = pd.read_csv(SPECIES_LIST).drop_duplicates("ebird_species_code")
    tax = pd.read_csv(EBIRD_TAXONOMY, usecols=["SPECIES_CODE", "SCI_NAME"])
    return pd.concat([sl.set_index("ebird_species_code")["ebird_sciname"],
                      tax.set_index("SPECIES_CODE")["SCI_NAME"]]).groupby(level=0).first()


# eBird v2024 code -> label scientific name, per model, where the name rule picks a silent class or none.
# Evidence: hit = the class is the model's top class (max over the 5-s windows of the first 20 s) on 20 focal
# training recordings of the species (Japanese recordings first; JP = how many are Japanese). Changed only when the
# class is taxonomically justified and hit rises by >= 25 points.
BIRDNET_OVERRIDES = {
    "v3": {
        # the Japanese taxon was an issf of this eBird 2021 species (pre-split parent); its own class stays silent
        "bkfbun2": "Emberiza spodocephala",  # E. personata 0/20 -> 7/20 (20 JP)
        "categr2": "Ardea ibis",  # A. coromanda 0/20 -> 6/20 (0 JP)
        "rinphe2": "Phasianus colchicus",  # P. versicolor 0/20 -> 11/20 (20 JP)
        # two "Little Ringed Plover" classes; the eBird v2024 name picks the weaker one
        "lirplo": "Charadrius dubius",  # Thinornis dubius 2/20 -> 14/20 (1 JP)
        # kept: cintit13 Parus cinereus 16/20 vs Parus minor 0/20 (20 JP); veggul1 Larus vegae 1/20 vs
        # L. argentatus 4/20 (gain below 25 points)
    },
    "v24": {
        # pre-split parent (eBird 2021); V2.4 has no class for the new species
        "bkfbun2": "Emberiza spodocephala",  # none -> 12/20 (20 JP)
        "rinphe2": "Phasianus colchicus",  # none -> 9/20 (20 JP)
        "ryurob3": "Larvivora komadori",  # none -> 15/20 (20 JP)
        "veggul1": "Larus argentatus",  # none -> 14/20 (7 JP)
        # IOC Japanese Tit; eBird v2024 lumps it into Parus cinereus, whose V2.4 class (Cinereous Tit) is silent
        "cintit13": "Parus minor",  # P. cinereus 1/20 -> 16/20 (20 JP)
    },
}


def label_index(model_sci: list[str], codes: list[str], overrides: dict[str, str]) -> np.ndarray:
    """eBird code -> model output index by scientific name (`overrides` = BIRDNET_OVERRIDES[model] first, then
    the eBird v2024 name, then the BirdNET V2.4 synonym in species_list), last a label equal to the code
    itself (the student region fold labels a code without a known name by the code); -1 if none matches."""
    sci = sci_names()
    syn = (pd.read_csv(SPECIES_LIST).dropna(subset=["birdnet_sciname"]).drop_duplicates("ebird_species_code")
           .set_index("ebird_species_code")["birdnet_sciname"])
    pos = {s: i for i, s in enumerate(model_sci)}
    return np.array([pos.get(overrides.get(c), pos.get(sci.get(c), pos.get(syn.get(c), pos.get(c, -1))))
                     for c in codes])


def set_codes(name: str) -> list[str]:
    """The annotated eBird codes of a benchmark, sorted."""
    return sorted({c for x in pd.read_parquet(DATA_ROOT / SETS[name][0])["labels"] for c in x})


def infer(model, name: str, set_name: str, codes: list[str], decoders: int) -> dict:
    """Score every window of a set on the model's classes for `codes` (cached as CACHE/<name>_<set>.npz)."""
    dst = CACHE / f"{name}_{set_name}.npz"
    if dst.exists():
        return dict(np.load(dst))
    manifest, id_col, off_col, max_s, hop, _ = SETS[set_name]
    man = pd.read_parquet(DATA_ROOT / manifest)
    rows = list(zip(man[id_col].astype(str), man["path"], man[off_col] if off_col else [0.0] * len(man)))
    idx = label_index(model.sci, codes, BIRDNET_OVERRIDES[model.tag])
    pool, prefetch = ThreadPoolExecutor(decoders), ThreadPoolExecutor(1)

    def load(chunk):
        return list(pool.map(lambda r: decode(r[1], max_s, r[2]), chunk))

    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    nxt, meta, scores, t0 = prefetch.submit(load, chunks[0]), [], [], time.time()
    for k, chunk in enumerate(chunks):
        audio = nxt.result()
        if k + 1 < len(chunks):
            nxt = prefetch.submit(load, chunks[k + 1])
        wins = []
        for (rid, _, _), a in zip(chunk, audio):
            assert a is not None, f"decode failed: {rid}"
            starts, w = windows(a, hop)
            meta += [(rid, i, s) for i, s in enumerate(starts)]
            wins += list(w)
        for b in range(0, len(wins), BATCH):
            lg = model(np.stack(wins[b : b + BATCH]))
            scores.append(np.where(idx[None, :] >= 0, lg[:, np.maximum(idx, 0)], MISSING))
        print(f"[{name} {set_name}] {min((k + 1) * CHUNK, len(rows))}/{len(rows)} recordings, "
              f"{len(meta)} windows, {(time.time() - t0) / 60:.1f} min", flush=True)
    out = {"ids": np.array([m[0] for m in meta]), "window_idx": np.array([m[1] for m in meta], np.int32),
           "t_start": np.array([m[2] for m in meta], np.float32),
           "scores": np.concatenate(scores).astype(np.float32), "codes": np.array(codes),
           "unmatched": np.array([c for c, i in zip(codes, idx) if i < 0], dtype=str)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst.with_suffix(".tmp.npz"), **out)
    dst.with_suffix(".tmp.npz").rename(dst)
    return out


def evaluate_set(set_name: str, r: dict, model_name: str) -> dict:
    """cmAP, macro ROC-AUC and pooled AUPRC of one cached score set (`model_name` is kept for the callers)."""
    s = r["scores"]
    res = {"n_windows": len(s), "all_finite": bool(np.isfinite(s).all()),
           "unmatched": r["unmatched"].tolist()}
    teacher_dir = DATA_ROOT / SETS[set_name][5]
    res["checked_against_perch_windows"] = (teacher_dir / "DONE").exists()
    if res["checked_against_perch_windows"]:
        perch = load_parts(teacher_dir, keys=("ids", "window_idx"))
        assert r["ids"].tolist() == perch["ids"].tolist(), f"{set_name}: windows differ from Perch"
        assert r["window_idx"].tolist() == perch["window_idx"].tolist(), f"{set_name}: window_idx differ"
    codes = r["codes"].tolist()
    manifest, id_col = SETS[set_name][:2]
    meta = pd.read_parquet(DATA_ROOT / manifest).set_index(id_col).loc[r["ids"]]
    y = np.array([[c in x for c in codes] for x in meta["labels"]], dtype=int)
    pos = np.where(y.sum(0) > 0)[0]
    res |= {"cmAP": cmap(s, y), "n_species": len(pos),
            "auroc": float(np.mean([roc_auc_score(y[:, j], s[:, j]) for j in pos])),
            "pooled_auprc": float(average_precision_score(y.ravel(), s.ravel()))}
    return res
