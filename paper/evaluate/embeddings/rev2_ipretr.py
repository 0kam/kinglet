# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""rev2_ipretr_v1: WABAD retrieval ranked by raw inner product,
as the Perch Hoplite search index does (metric 'IP'), instead of cosine similarity.

Same species, exemplar draws (seeds), k in {1, 5}, max over exemplars and AP as probe_wabad_emb.py
(within space) and compat_wabad_emb.py (across spaces). BUG CHECK: with L2-normalised inputs the
script reproduces the published cosine numbers (wabad_embprobe_results.json / wabad_embcompat_results.json).

Output: results/rev2_ipretr_v1.json
Usage: python evaluate/embeddings/rev2_ipretr.py
"""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, RESULTS  # noqa: E402
from evaluate.embeddings.compat_wabad_emb import cross_retrieval  # noqa: E402
from evaluate.embeddings.probe_wabad_emb import KS, MIN_POS, SEEDS, load_emb  # noqa: E402

MODELS = ["teacher", "B", "S", "XS"]


def main():
    meta = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    recs = meta["window_id"].str.rsplit("_", n=1).str[0].to_numpy()
    sp_all = sorted({c for x in meta["labels"] for c in x})
    Y = {c: np.zeros(len(meta), bool) for c in sp_all}
    for i, labs in enumerate(meta["labels"]):
        for c in labs:
            Y[c][i] = True
    species = [c for c in sp_all if Y[c].sum() >= MIN_POS and len(np.unique(recs[Y[c]])) >= 2]
    raw = {m: load_emb(m, meta["window_id"].tolist()) for m in MODELS}
    variants = {"cos": {m: e / np.linalg.norm(e, axis=1, keepdims=True) for m, e in raw.items()}, "ip": raw}

    def mean_ap(src, dst):
        ret = {k: [] for k in KS}
        for c in species:
            pos_idx = np.flatnonzero(Y[c])
            aps = [cross_retrieval(src, dst, pos_idx, recs, np.random.default_rng(zlib.crc32(f"{c}:{s}".encode())))
                   for s in range(SEEDS)]
            for k in KS:
                ret[k].append(np.nanmean([a[k] for a in aps]))
        return {f"k{k}": round(float(np.nanmean(ret[k])), 4) for k in KS}

    out = {"n_species": len(species)}
    for v, E in variants.items():
        r = {f"own_{m}": mean_ap(E[m], E[m]) for m in MODELS}
        for m in MODELS[1:]:
            r[f"t2s_{m}"] = mean_ap(E["teacher"], E[m])
            r[f"s2t_{m}"] = mean_ap(E[m], E["teacher"])
        out[v] = r
        print(v, r, flush=True)
    (RESULTS / "rev2_ipretr_v1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
