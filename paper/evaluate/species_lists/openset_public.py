# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""opensetpub_v1b: false detections of a Japan-list recogniser on public soundscapes.

Every model scores the 317 Japan classes - the released Japan pack - on WABAD and the seven
BirdSet test sets, where almost no listed species occurs: a detection of a species that is not
annotated in the window is a false positive. Models and operating rules:

  kinglet032_*   : released v0.3.2 trunks + Japan pack, fires at softmax prob > 0.1 (CLI default)
  perchfold      : Perch 2.0 full-vocabulary softmax sliced at the Japan classes, prob > 0.1
  v24            : BirdNET V2.4 on the Japan list, confidence > 0.1 and > 0.25
                   (classes it does not have cannot fire - coverage is in the output)

Scores are cached (DATA_ROOT/birdnet_v3/cache_v2/openset317_<model>_<set>.npz) and read again by
opensetcurve.py and opensetrel.py; the table is rebuilt from the caches:

  python evaluate/species_lists/openset_public.py --models kinglet032_xs perchfold --sets POW HSN   # screening
  python evaluate/species_lists/openset_public.py                                                   # everything

Output: results/openset_public_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, PERCH_DIR, PERCH_GPU_DIR, RELEASE_ASSETS, RESULTS, jp_list  # noqa: E402
from common.scoring import BIRDNET_OVERRIDES, SETS, V24, infer, label_index, sci_names  # noqa: E402

PUBLIC_SETS = ("WABAD", "POW", "HSN", "NES", "PER", "SNE", "SSW", "UHH")
COUNTS = ("windows", "fires", "fp", "windows_with_fp", "fp_unlisted", "hits", "positives")
BATCH_TF = 32  # fixed XLA batch; pad the tail


class PerchFolded:
    """Perch teacher as a 317-class scorer: full-vocab softmax folded to the Japan classes."""

    tag = "student"  # classes are eBird codes already (same convention as Distilled)

    def __init__(self, gpu: bool = False):
        import tensorflow as tf

        self._tf = tf  # GPU build of the same weights (teacher/embed_perch.py --gpu)
        self.fn = tf.saved_model.load(str(PERCH_GPU_DIR if gpu else PERCH_DIR)).signatures[
            "serving_default"]
        classes = jp_list()
        self.idx = np.array([c["perch_idx"] for c in classes])
        self.ok = self.idx >= 0
        codes = [c["species"] for c in classes]
        sci = sci_names()
        self.sci = [sci.get(c, c) for c in codes]
        self.labels = [f"{s}_{c}" for s, c in zip(self.sci, codes)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), len(self.sci)), np.float32)
        for i in range(0, len(x), BATCH_TF):
            b = x[i : i + BATCH_TF].astype(np.float32)
            nb = len(b)
            if nb < BATCH_TF:
                b = np.pad(b, ((0, BATCH_TF - nb), (0, 0)))
            lab = self.fn(inputs=self._tf.constant(b))["label"].numpy()[:nb]
            lab -= lab.max(1, keepdims=True)
            p = np.exp(lab) / np.exp(lab).sum(1, keepdims=True)
            out[i : i + nb][:, self.ok] = p[:, self.idx[self.ok]]
        return out


class Shipped:
    """Released Kinglet tier + Japan pack as a 317-class scorer (the probabilities the CLI reports)."""

    tag = "student"  # classes are eBird codes already

    def __init__(self, tier: str, threads: int):
        from kinglet import Kinglet

        self.m = Kinglet(RELEASE_ASSETS / f"kinglet-{tier}-v0.3.2.onnx",
                         RELEASE_ASSETS / f"kinglet-{tier}-v0.3.2.pack-jp.npz", threads=threads)
        sci = sci_names()
        self.sci = [sci.get(code, code) for code, *_ in self.m.classes]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.m.run(x)[1]


def logit(c: float) -> float:  # BirdNET confidence -> the logit space its scores are cached in
    return float(np.log(c / (1 - c)))


def models(threads: int, perch_gpu: bool) -> dict:
    """name -> (constructor, [(operating rule, threshold in the cached score space), ...])."""
    birdnet = [("c>0.1", logit(0.1)), ("c>0.25", logit(0.25))]
    return {
        **{f"kinglet032_{t}": (lambda t=t: Shipped(t, threads), [("p>0.1", 0.1)]) for t in ("b", "s", "xs")},
        "perchfold": (lambda: PerchFolded(perch_gpu), [("p>0.1", 0.1)]),
        "v24": (lambda: V24(threads), birdnet),
    }


def tally(fired: np.ndarray, labeled: np.ndarray) -> dict:
    """Counts behind every rate, so sets can be pooled by summing."""
    fp = fired & ~labeled
    t = dict(zip(COUNTS, (len(fired), int(fired.sum()), int(fp.sum()), int(fp.any(1).sum()),
                          int(fp[:, ~labeled.any(0)].sum()),  # species never annotated in the set
                          int((fired & labeled).sum()), int(labeled.sum()))))
    assert t["fp"] + t["hits"] == t["fires"] and t["fp_unlisted"] <= t["fp"]
    return t


def rates(t: dict) -> dict:
    return {"fp_per_window": round(t["fp"] / t["windows"], 4),
            "p_any_fp": round(t["windows_with_fp"] / t["windows"], 4),
            "fp_unlisted_per_window": round(t["fp_unlisted"] / t["windows"], 4),
            "recall_listed": round(t["hits"] / t["positives"], 4) if t["positives"] else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--sets", nargs="+", default=PUBLIC_SETS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--perch-gpu", action="store_true", help="Perch on the GPU build (same weights)")
    args = ap.parse_args()
    BIRDNET_OVERRIDES.setdefault("student", {})
    codes317 = [c["species"] for c in jp_list()]
    ci = {c: j for j, c in enumerate(codes317)}
    out_path = RESULTS / "openset_public_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    registry = models(args.threads, args.perch_gpu)
    for name in args.models or registry:
        build, rules = registry[name]
        model = build()
        mapped = int((label_index(model.sci, codes317, BIRDNET_OVERRIDES[model.tag]) >= 0).sum())
        for set_name in args.sets:
            z = infer(model, f"openset317_{name}", set_name, codes317, args.decoders)
            man = pd.read_parquet(DATA_ROOT / SETS[set_name][0])
            assert z["ids"].tolist() == man[SETS[set_name][1]].tolist(), f"{name}/{set_name}: window order"
            assert z["codes"].tolist() == codes317
            labeled = np.zeros(z["scores"].shape, bool)
            for i, labs in enumerate(man["labels"]):
                labeled[i, [ci[c] for c in labs if c in ci]] = True
            for rule, thresh in rules:
                fired = z["scores"] > thresh
                assert not fired[:, np.isin(codes317, z["unmatched"])].any()  # absent classes never fire
                t = tally(fired, labeled)
                fp_rate = (fired & ~labeled).mean(0)
                worst = {codes317[j]: round(float(fp_rate[j]), 4)
                         for j in np.argsort(-fp_rate)[:5] if fp_rate[j] > 0}
                results.setdefault(f"{name} @{rule}", {"classes_mapped": mapped, "sets": {}})["sets"][set_name] = (
                    t | rates(t) | {"listed_species_in_set": int(labeled.any(0).sum()), "worst_fp": worst})
                print(f"{name:10s} @{rule:6s} {set_name:6s} {rates(t)} worst {worst}", flush=True)
    for key, r in results.items():  # pooled over the sets scored so far, windows as the unit
        pooled = {k: sum(s[k] for s in r["sets"].values()) for k in COUNTS}
        r["pooled"] = pooled | rates(pooled) | {"sets": sorted(r["sets"])}
        print(f"{key:18s} pooled over {len(r['sets'])} sets: {rates(pooled)}")
    out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
