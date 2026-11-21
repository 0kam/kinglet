#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# revstats_v1: Perch 2.0 teacher re-run on the benchmarks to store the per-window log-sum-exp over all
# 14,795 classes (teacher/embed_perch.py key lse), read by revision_stats.py. Same manifests and class lists
# as the cached teacher runs, and the same build: GPU for WABAD and BirdSet POW..UHH, CPU for NBP.
#   bash evaluate/stats/revstats_lse.sh
set -e
export TF_FORCE_GPU_ALLOW_GROWTH=true
PAPER="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PAPER"
read -r ROOT DD < <(python -c 'from common.config import DATA_ROOT, DISTILL_ROOT; print(DATA_ROOT, DISTILL_ROOT)')
for S in WABAD POW HSN NES PER SNE SSW UHH NBP; do
    if [ $S = WABAD ]; then M="$ROOT/wabad/test.parquet"; C="$ROOT/wabad/emb_test/classes.json"
    else M="$ROOT/birdset/${S}_test.parquet"; C="$ROOT/birdset/emb_${S}_test/classes.json"; fi
    if [ $S = NBP ]; then BUILD=""; else BUILD="--gpu --batch 128"; fi
    [ -f "$DD/revstats_lse/$S/DONE" ] || python teacher/embed_perch.py --manifest "$M" --id-col window_id --path-col path \
        --classes-json "$C" --out "$DD/revstats_lse/$S" --hop 5 $BUILD
done
echo ALL DONE
