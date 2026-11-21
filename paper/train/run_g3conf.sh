#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# g3conf_v1: train B (regnety_016) and S (efficientnet_lite0) with the full recipe on shared batches,
# 15 epochs, then score both on POW and HSN (full-vocabulary scoring). XS comes from run_g2xcl.sh.
# The remaining benchmarks are scored by evaluate/accuracy/run_wabad.sh, run_birdsetfull.sh, run_birdsetnbp.sh.
# Needs the teacher outputs of teacher/ (parts, log-mels, <distill_root>/softmax_full).
set -eu
PAPER="$(cd "$(dirname "$0")/.." && pwd)"
DD=$(PYTHONPATH="$PAPER" python -c "from common import config; print(config.DISTILL_ROOT)")
cd "$PAPER"
echo "=== training g3conf_v1 (B + S, shared batches, 15 ep) $(date) ==="
python train/train_distill.py --out "$DD/g3conf_v1" --backbone regnety_016@full efficientnet_lite0@full \
    --softmax-full "$DD/softmax_full" --epochs 15 --resume
for M in regnety_016 efficientnet_lite0; do
  echo "=== scoring $M on POW/HSN $(date) ==="
  python evaluate/accuracy/score_student.py --ckpt "$DD/g3conf_v1/${M}_full/best.pt" \
      --name "distill_g3r_$M" --sets POW HSN
done
echo "=== ALL DONE $(date) ==="
