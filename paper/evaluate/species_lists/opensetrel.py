# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""opensetrel_v1: which species the Japan pack falsely detects on foreign soundscapes.

Reads the cached scores of opensetpub_v1b (v0.3.2 Japan pack, p > 0.1) and reports, per tier,
(a) how many goldcrest false detections on SNE fall in windows annotated with the golden-crowned
kinglet, and (b) how the pooled false detections are spread over the 317 listed species.

  python evaluate/species_lists/opensetrel.py        # -> results/opensetrel_v1.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, RESULTS  # noqa: E402
from common.scoring import CACHE, SETS  # noqa: E402
from evaluate.species_lists.openset_public import PUBLIC_SETS  # noqa: E402


def main():
    ref = json.loads((RESULTS / "openset_public_results.json").read_text())
    out = {}
    for tier in ("b", "s", "xs"):
        per_species, sne = 0, None
        for name in PUBLIC_SETS:
            z = np.load(CACHE / f"openset317_kinglet032_{tier}_{name}.npz")
            man = pd.read_parquet(DATA_ROOT / SETS[name][0])
            assert z["ids"].tolist() == man[SETS[name][1]].tolist(), f"{tier}/{name}: window order"
            ci = {c: j for j, c in enumerate(z["codes"].tolist())}
            labeled = np.zeros(z["scores"].shape, bool)
            for i, labs in enumerate(man["labels"]):
                labeled[i, [ci[c] for c in labs if c in ci]] = True
            fp = (z["scores"] > 0.1) & ~labeled
            per_species = per_species + fp.sum(0)
            if name == "SNE":
                gold, kinglet = fp[:, ci["goldcr1"]], np.array(["gockin" in list(labs) for labs in man["labels"]])
                sne = {"goldcr1_fp_windows": int(gold.sum()), "of_them_gockin": int((gold & kinglet).sum()),
                       "share": round(float((gold & kinglet).sum() / gold.sum()), 4),
                       "gockin_base_rate": round(float(kinglet.mean()), 4), "goldcr1_fp_rate": round(float(gold.mean()), 4)}
        r = ref[f"kinglet032_{tier} @p>0.1"]
        # bug checks: the counts of opensetpub_v1b are reproduced
        assert per_species.sum() == r["pooled"]["fp"], tier
        assert abs(sne["goldcr1_fp_rate"] - r["sets"]["SNE"]["worst_fp"]["goldcr1"]) < 1e-4, tier
        order, codes = np.argsort(-per_species), z["codes"].tolist()
        out[tier] = {"sne": sne, "pooled_fp": int(per_species.sum()), "species_with_fp": int((per_species > 0).sum()),
                     "top5_share": round(float(per_species[order[:5]].sum() / per_species.sum()), 4),
                     "top10_share": round(float(per_species[order[:10]].sum() / per_species.sum()), 4),
                     "top10": [(codes[j], int(per_species[j])) for j in order[:10]]}
        print(tier, json.dumps(out[tier]))
    (RESULTS / "opensetrel_v1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
