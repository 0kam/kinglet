# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Score BirdNET V2.4 (FP32 TensorFlow Lite, CPU) on WABAD and the BirdSet test sets.

Each 5-s window is cut into the 3-s crops 0-3, 1-4 and 2-5 s, resampled to 48 kHz, and scored as the mean logit
(common.scoring.V24). Species are matched by scientific name (common.scoring.label_index with
BIRDNET_OVERRIDES["v24"]). Scores are cached as common.scoring.CACHE/v24_<set>.npz.

Usage:
  CUDA_VISIBLE_DEVICES="" python evaluate/accuracy/score_birdnet_v24.py --sets WABAD POW HSN
Output: results/birdnet_v24_results.json (one dict per set, merged into an existing file) and the score cache.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import RESULTS  # noqa: E402
from common.scoring import SETS, V24, evaluate_set, infer, set_codes  # noqa: E402

NAME = "birdnet_v24_zeroshot"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    ap.add_argument("--threads", type=int, default=5, help="inference intra-op threads")
    ap.add_argument("--decoders", type=int, default=1)
    args = ap.parse_args()

    model = V24(args.threads)
    out = RESULTS / "birdnet_v24_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for s in args.sets:
        res[s] = evaluate_set(s, infer(model, model.tag, s, set_codes(s), args.decoders), NAME)
        print(s, json.dumps({k: v for k, v in res[s].items() if k != "unmatched"}), flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
