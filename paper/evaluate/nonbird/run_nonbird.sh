#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# nonbird_v1: embeddings + class scores of the teacher and Kinglet B/S/XS on the public non-bird sets
# (run data_prep/prepare_nonbird.py before, eval_nonbird.py after). TF capped at 1.8 GB, batch 16/32.
#   bash evaluate/nonbird/run_nonbird.sh
set -e
PAPER="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PAPER"
read -r P DD < <(python -c 'from common.config import NONBIRD_ROOT, DISTILL_ROOT; print(NONBIRD_ROOT, DISTILL_ROOT)')
O="$DD/nonbird_v1"
for s in watkins anuraset humbugdb reefset; do
    python evaluate/nonbird/embed_nonbird.py --model teacher --gpu --batch 16 --manifest "$P/$s/manifest.parquet" \
        --perch-idx "$P/$s/perch_idx.json" --out "$O/${s}_teacher.npz" 2>&1 | grep -E "saved|done|Error|error" || true
    for m in B S XS; do
        python evaluate/nonbird/embed_nonbird.py --model $m --gpu --manifest "$P/$s/manifest.parquet" \
            --perch-idx "$P/$s/perch_idx.json" --out "$O/${s}_$m.npz" 2>&1 | grep -E "saved|done|Error|error" || true
    done
done
for s in watkins anuraset humbugdb reefset; do for m in teacher B S XS; do test -f "$O/${s}_$m.npz"; done; done
echo EMBED_DONE
