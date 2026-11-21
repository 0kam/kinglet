# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Score a distilled student (Kinglet B, S or XS) on WABAD and the BirdSet test sets.

The checkpoint (train/train_distill.py) is wrapped as a model with `.sci`/`.labels`/`__call__` (audio (B, 5 s at
32 kHz) -> class scores), so common.scoring.infer/evaluate_set handle windowing, eBird code -> class mapping,
caching and the metrics. The scores are the full output layer's softmax over the 14,795 Perch 2.0 classes, read
at the Perch class of each annotated species of the set (region fold, embed_perch.perch_classes); a species
without a Perch class scores 0. Scores are cached as common.scoring.CACHE/<name>_<set>.npz.

Usage (DISTILL_ROOT: see common/config.py):
  python evaluate/accuracy/score_student.py --ckpt $DISTILL_ROOT/g3conf_v1/regnety_016_full/best.pt \
      --name distill_g3r_regnety_016 --sets WABAD POW HSN
Output: results/<name>_results.json (one dict per set, merged into an existing file) and the score cache.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DISTILL_ROOT, RESULTS  # noqa: E402
from common.scoring import BIRDNET_OVERRIDES, SETS, evaluate_set, infer, sci_names, set_codes  # noqa: E402
from teacher.embed_perch import SR  # noqa: E402


class Distilled:
    """A full-vocabulary student checkpoint; scores = its softmax at the Perch index of each class code.

    The checkpoint takes the normalised log-mel (train.backbones.MEL, log(x + 1e-6), standardised by the training
    mean and SD stored in the checkpoint). Classes default to the 317 Japanese training species; set_region()
    switches them to a benchmark's species."""
    tag = "student"  # no name overrides: classes are eBird codes already

    def __init__(self, ckpt: Path, batch_device: str):
        import torchaudio

        from train.backbones import MEL
        from train.train_distill import Student

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = state["cfg"]
        self.full = bool(self.cfg.get("softmax_full"))
        assert self.full, "expects a full-vocabulary checkpoint"
        self.model = Student(self.cfg["backbone"], self.cfg["n_out"],
                             head=self.cfg.get("head", "full")).to(batch_device).eval()
        self.model.load_state_dict(state["model"])
        self.device = batch_device
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, **MEL).to(batch_device)
        self._fold([c["perch_idx"] for c in self.cfg["classes"]], [c["species"] for c in self.cfg["classes"]])

    def _fold(self, perch_idx: list[int], codes: list[str]):
        pidx = np.array(perch_idx)
        self.fold_idx = torch.from_numpy(np.where(pidx >= 0, pidx, 0)).to(self.device)
        self.fold_ok = torch.from_numpy((pidx >= 0).astype(np.float32)).to(self.device)
        sci = sci_names()
        self.sci = [sci.get(c, c) for c in codes]
        self.labels = [f"{s}_{c}" for s, c in zip(self.sci, codes)]
        return pidx

    def set_region(self, codes: list[str]):
        """Region fold: the class axis becomes `codes`, scores = the full softmax at each code's Perch index."""
        from teacher.embed_perch import perch_classes

        pidx = self._fold([c["perch_idx"] for c in perch_classes(codes)], codes)
        print(f"region fold: {len(codes)} codes, {int((pidx >= 0).sum())} mapped to Perch classes")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(x).to(self.device)
            m = ((torch.log(self.mel(xt) + 1e-6) - self.cfg["mel_mu"]) / self.cfg["mel_sd"]).unsqueeze(1)
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                _, emb, logits = self.model(m)
            self.emb = emb.float()  # Perch-space embedding of the last batch (mcu/mcurate.py)
            out = torch.softmax(logits.float(), 1)[:, self.fold_idx] * self.fold_ok
        return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", required=True, help="model name for caches / result files")
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    ap.add_argument("--decoders", type=int, default=4)
    args = ap.parse_args()

    BIRDNET_OVERRIDES.setdefault(Distilled.tag, {})
    model = Distilled(args.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt.resolve()
    ckpt = str(ckpt.relative_to(DISTILL_ROOT.resolve()) if ckpt.is_relative_to(DISTILL_ROOT.resolve()) else ckpt.name)
    out = RESULTS / f"{args.name}_results.json"
    # merge: scoring one more set must not drop the sets already scored under this name
    res = json.loads(out.read_text()) if out.exists() else {}
    assert res.get("ckpt", ckpt) == ckpt, f"{out} holds scores of {res['ckpt']}"
    res |= {"ckpt": ckpt, "epoch_val_cos": torch.load(args.ckpt, map_location="cpu", weights_only=False)["val_cos"]}
    for s in args.sets:
        model.set_region(set_codes(s))
        r = infer(model, args.name, s, set_codes(s), args.decoders)
        res[s] = evaluate_set(s, r, args.name)
        print(s, json.dumps({k: v for k, v in res[s].items() if k != "unmatched"}), flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
