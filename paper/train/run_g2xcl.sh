#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# g2xcl_v1: train XS (mobilenetv4_conv_small) with the full recipe (@full, the released XS) and the
# loss-ablation variant whose class head is detached from the backbone (@detach), on shared batches,
# 15 epochs, then score both on POW and HSN (full-vocabulary scoring).
# The remaining benchmarks are scored by evaluate/accuracy/run_wabad.sh, run_birdsetfull.sh, run_birdsetnbp.sh.
# Needs the teacher outputs of teacher/ (parts, log-mels, <distill_root>/softmax_full).
set -eu
PAPER="$(cd "$(dirname "$0")/.." && pwd)"
DD=$(PYTHONPATH="$PAPER" python -c "from common import config; print(config.DISTILL_ROOT)")
M=mobilenetv4_conv_small
cd "$PAPER"
echo "=== training g2xcl_v1 ($M @full + @detach, shared batches, 15 ep) $(date) ==="
python train/train_distill.py --out "$DD/g2xcl_v1" --backbone $M@full $M@detach \
    --softmax-full "$DD/softmax_full" --epochs 15 --resume
for ARM in full detach; do
  echo "=== scoring $ARM on POW/HSN $(date) ==="
  python evaluate/accuracy/score_student.py --ckpt "$DD/g2xcl_v1/${M}_$ARM/best.pt" \
      --name "distill_g2r_$ARM" --sets POW HSN
done
echo "=== ALL DONE $(date) ==="
