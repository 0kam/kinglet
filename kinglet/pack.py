"""Regional head packs: build one from a headpack bundle, load one for scoring.

A v0.3 model is a trunk ONNX (waveform -> embedding + trunk feature h) plus a head pack:
a linear map folding the full 14,795-class Perch vocabulary onto
<region species> + <sound-group member classes> + "other". Species rows slice exactly from the
full head; the "other" row is a ridge fit of the exact out-of-pack logsumexp on calibration
features shipped inside the headpack (closed form, seconds on a laptop). Scores are softmax
probabilities over the pack rows; group scores are sums of member probabilities.

Pack npz fields: W (R, d) f32, b (R,) f32, codes (R,) str (species codes, member class names,
"other" last), n_species, scientific_name/japanese_name (n_species,), groups json,
group_thresholds json, meta json (build report included).

A species list comes from a CSV or, for any eBird region, from the eBird API (free key).
"""

import csv
import json
import urllib.parse
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np

RIDGE_LAMBDA = 1.0
DEAD_BIAS = -1.0e9  # species absent from the vocabulary keep an exact-0 probability row
CHUNK = 2048
EBIRD_API = "https://api.ebird.org/v2"
EBIRD_CHUNK = 200  # species codes per taxonomy request (keeps the URL short)


def default_groups() -> tuple[dict[str, list[str]], dict[str, float | None]]:
    """The shipped FSD50K class -> group mapping and per-group operating thresholds."""
    d = json.loads(resources.files("kinglet").joinpath("groups.json").read_text())
    return d["groups"], d["thresholds"]


def load_pack(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    pack = {k: z[k] for k in z.files}
    for k in ("groups", "group_thresholds", "meta"):
        pack[k] = json.loads(str(pack[k]))
    pack["n_species"] = int(pack["n_species"])
    codes = [str(c) for c in pack["codes"]]
    pack["group_cols"] = {g: np.array([codes.index(m) for m in members])
                          for g, members in pack["groups"].items()}
    return pack


def read_species_csv(path: str | Path) -> list[dict]:
    """Rows need ebird_code + scientific_name (+ optional japanese_name, vocab_index).
    vocab_index pins the Perch vocabulary row exactly; otherwise scientific_name is matched
    against the vocabulary (exact match only — synonyms become dead rows, listed in the report)."""
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def _ebird_get(path: str, key: str):
    req = urllib.request.Request(f"{EBIRD_API}/{path}", headers={"X-eBirdApiToken": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_region_species(region: str, key: str) -> list[dict]:
    """eBird species list of a region (US-CA, CR, AU-NSW, ...) as ebird_code + scientific_name rows.
    Names come from the current eBird taxonomy; the Perch vocabulary is older, so renamed or split
    species become dead rows (listed in the build report). Non-species taxa (spuhs, hybrids, forms)
    are dropped; codes the taxonomy does not resolve are kept with an empty name (dead rows too)."""
    codes = _ebird_get(f"product/spplist/{urllib.parse.quote(region)}", key)
    tax = {}
    for i in range(0, len(codes), EBIRD_CHUNK):
        q = urllib.parse.urlencode({"fmt": "json", "species": ",".join(codes[i : i + EBIRD_CHUNK])},
                                   safe=",")
        tax |= {t["speciesCode"]: t for t in _ebird_get(f"ref/taxonomy/ebird?{q}", key)}
    return [{"ebird_code": c, "scientific_name": tax.get(c, {}).get("sciName", "")}
            for c in codes if tax.get(c, {}).get("category", "species") == "species"]


def build_pack(headpack: str | Path, species: list[dict], out: str | Path,
               groups: dict[str, list[str]] | None = None,
               group_thresholds: dict[str, float | None] | None = None,
               ridge_lambda: float = RIDGE_LAMBDA) -> dict:
    """Fold the full head onto a species list; returns the build report (also stored in meta)."""
    hp = np.load(headpack, allow_pickle=False)
    W, b, labels = hp["W"].astype(np.float32), hp["b"].astype(np.float32), hp["labels"]
    lab_idx = {str(n): i for i, n in enumerate(labels)}
    if groups is None:
        groups, dft = default_groups()
        group_thresholds = group_thresholds or dft
    group_thresholds = group_thresholds or {}

    vidx, unmatched = [], []
    for r in species:
        if r.get("vocab_index") not in (None, ""):
            vidx.append(int(r["vocab_index"]))
        else:
            i = lab_idx.get(r["scientific_name"], -1)
            vidx.append(i)
            if i < 0:
                unmatched.append(r["ebird_code"])
    vidx = np.array(vidx)
    members = [m for ms in groups.values() for m in ms]
    assert len(set(members)) == len(members), "a class may belong to one group only"
    midx = np.array([lab_idx[m] for m in members])  # shipped groups always resolve
    overlap = set(vidx[vidx >= 0]) & set(midx)
    assert not overlap, f"species rows overlap group members: {sorted(overlap)}"

    in_mask = np.zeros(len(labels), bool)
    in_mask[vidx[vidx >= 0]] = True
    in_mask[midx] = True
    out_idx = np.flatnonzero(~in_mask)

    # exact out-of-pack logsumexp on the calibration features, chunked (W alone is ~40 MB)
    h = hp["calib_h"].astype(np.float32)
    is_ho = hp["calib_is_holdout"]
    n_sp = len(vidx)
    cols = np.concatenate([np.where(vidx >= 0, vidx, 0), midx])
    z = np.empty(len(h), np.float32)
    p_in = np.empty((len(h), len(cols)), np.float32)  # exact folded probs for the report
    for i in range(0, len(h), CHUNK):
        lg = h[i : i + CHUNK] @ W.T + b
        mx = lg.max(1, keepdims=True)
        e = np.exp(lg - mx)
        z[i : i + CHUNK] = np.log(e[:, out_idx].sum(1)) + mx[:, 0]
        p_in[i : i + CHUNK] = e[:, cols] / e.sum(1, keepdims=True)
    p_in[:, :n_sp][:, vidx < 0] = 0.0

    ht = np.hstack([h[~is_ho], np.ones((int((~is_ho).sum()), 1))]).astype(np.float64)
    A = ht.T @ ht + ridge_lambda * np.eye(ht.shape[1])
    wb = np.linalg.solve(A, ht.T @ z[~is_ho].astype(np.float64))
    w_o, b_o = wb[:-1].astype(np.float32), float(wb[-1])

    d = W.shape[1]
    Wp = np.zeros((len(cols) + 1, d), np.float32)
    bp = np.full(len(cols) + 1, DEAD_BIAS, np.float32)
    Wp[:-1], bp[:-1] = W[cols], b[cols]
    Wp[:n_sp][vidx < 0], bp[:n_sp][vidx < 0] = 0.0, DEAD_BIAS
    Wp[-1], bp[-1] = w_o, b_o

    # holdout report in probability space (where the operating threshold lives)
    hh, pe = h[is_ho], p_in[is_ho]
    lg = hh @ Wp.T + bp
    e = np.exp(lg - lg.max(1, keepdims=True))
    pp = (e / e.sum(1, keepdims=True))[:, :-1]
    fire_pack, fire_full = (pp[:, :n_sp] > 0.1).any(1), (pe[:, :n_sp] > 0.1).any(1)
    report = {
        "n_species": n_sp, "unmatched": unmatched, "n_group_rows": len(midx),
        "holdout_windows": int(is_ho.sum()),
        "prob_err_mean": float(np.abs(pp - pe).mean()), "prob_err_max": float(np.abs(pp - pe).max()),
        "fire@0.1_disagree": float((fire_pack != fire_full).mean()),
    }

    codes = ([r["ebird_code"] for r in species] + members + ["other"])
    np.savez(Path(out), W=Wp, b=bp, codes=np.array(codes), n_species=n_sp,
             scientific_name=np.array([r.get("scientific_name", "") for r in species]),
             japanese_name=np.array([r.get("japanese_name", "") for r in species]),
             groups=json.dumps(groups), group_thresholds=json.dumps(group_thresholds),
             meta=json.dumps({"headpack_meta": json.loads(str(hp["meta"])),
                              "ridge_lambda": ridge_lambda} | report))
    return report
