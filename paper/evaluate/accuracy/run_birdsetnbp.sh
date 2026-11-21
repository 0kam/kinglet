#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# birdsetnbp_v1: BirdSet NBP (NIPS4Bplus test_5s) through the same path as birdsetfull_v1.
# Requires data_prep/prepare_birdset.py --test-only --sets NBP.
# 1) Perch 2.0 teacher embeddings + logits for the set species (CPU; 539 windows)
# 2) Kinglet B/S/XS (results/<name>_results.json is merged, not overwritten)  3) BirdNET V2.4 (CPU)
# BirdNET+ V3.0 is scored for all benchmarks by evaluate/accuracy/birdnetv3rel.py (score --run v1 / v2).
set -e
PAPER="$(cd "$(dirname "$0")/../.." && pwd)"
cfg() { python -c "import sys; sys.path.insert(0, sys.argv[1]); import common.config as c; print(getattr(c, sys.argv[2]))" "$PAPER" "$1"; }
ROOT="$(cfg DATA_ROOT)"
DD="$(cfg DISTILL_ROOT)"
TMP=$(mktemp -d)
cd "$PAPER"

# the annotated species of a benchmark manifest as a class list for the teacher
classes() {
    python - "$1" "$2" <<'EOF'
import json, sys
import pandas as pd
m = pd.read_parquet(sys.argv[1])
sp = sorted({c for x in m["labels"] for c in x})
json.dump([{"species": c} for c in sp], open(sys.argv[2], "w"))
print(sys.argv[1], len(sp), "species")
EOF
}

# Kinglet B, S, XS (full output layer, region fold)
students() {
    python evaluate/accuracy/score_student.py --ckpt "$DD/g3conf_v1/regnety_016_full/best.pt" \
        --name distill_g3r_regnety_016 --sets "$@"
    python evaluate/accuracy/score_student.py --ckpt "$DD/g3conf_v1/efficientnet_lite0_full/best.pt" \
        --name distill_g3r_efficientnet_lite0 --sets "$@"
    python evaluate/accuracy/score_student.py --ckpt "$DD/g2xcl_v1/mobilenetv4_conv_small_full/best.pt" \
        --name distill_g2r_full --sets "$@"
}

S=NBP
if [ ! -f "$ROOT/birdset/emb_${S}_test/DONE" ]; then
    classes "$ROOT/birdset/${S}_test.parquet" "$TMP/${S}_classes.json"
    python teacher/embed_perch.py --manifest "$ROOT/birdset/${S}_test.parquet" \
        --id-col window_id --path-col path --classes-json "$TMP/${S}_classes.json" \
        --out "$ROOT/birdset/emb_${S}_test" --hop 5 --threads 8
fi
students $S
CUDA_VISIBLE_DEVICES="" python evaluate/accuracy/score_birdnet_v24.py --sets $S
echo "birdsetnbp_v1 chain DONE"
