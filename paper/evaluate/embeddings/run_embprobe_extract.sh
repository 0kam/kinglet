#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# wabad_embprobe_v1 stage 1: student embeddings on the WABAD windows (GPU)
#   bash evaluate/embeddings/run_embprobe_extract.sh
set -e
PAPER="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PAPER"
read -r ROOT DD < <(python -c 'from common.config import DATA_ROOT, DISTILL_ROOT; print(DATA_ROOT, DISTILL_ROOT)')
python evaluate/embeddings/embed_student.py --ckpt "$DD/g3conf_v1/regnety_016_full/best.pt" \
    --manifest "$ROOT/wabad/test.parquet" --out "$DD/emb_wabad_B"
python evaluate/embeddings/embed_student.py --ckpt "$DD/g3conf_v1/efficientnet_lite0_full/best.pt" \
    --manifest "$ROOT/wabad/test.parquet" --out "$DD/emb_wabad_S"
python evaluate/embeddings/embed_student.py --ckpt "$DD/g2xcl_v1/mobilenetv4_conv_small_full/best.pt" \
    --manifest "$ROOT/wabad/test.parquet" --out "$DD/emb_wabad_XS"
echo "embprobe extraction DONE"
