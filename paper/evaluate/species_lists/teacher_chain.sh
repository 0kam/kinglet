#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# opensetcurve_v1 / regionlists_v1: Perch 2.0 pass over the eight benchmarks for the region lists
# (opensetcurve_extract.py --part teacher, GPU build), then the full summary (opensetcurve.py, CPU).
# The student features (--part students) must exist already.
#   bash evaluate/species_lists/teacher_chain.sh
PAPER="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$(cd "$PAPER" && python -c 'from common.config import DISTILL_ROOT; print(DISTILL_ROOT / "opensetcurve_v1")')"
export TF_FORCE_GPU_ALLOW_GROWTH=true
cd "$PAPER"
python evaluate/species_lists/opensetcurve_extract.py --part teacher --decoders 8 || exit 1
CUDA_VISIBLE_DEVICES="" python evaluate/species_lists/opensetcurve.py > "$LOG_DIR/summary_full.log" 2>&1
echo "TEACHER CHAIN DONE"
