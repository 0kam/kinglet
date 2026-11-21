#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# gpuspeed_v1: the GPU throughput run of Appendix S4 (bench_gpu.py), one machine, all parts in sequence.
# Writes results/gpuspeed_v1.json (replaced). Usage: bash compute_speed/gpuspeed_run.sh [machine label]
PAPER="$(cd "$(dirname "$0")/.." && pwd)"
MACHINE="${1:-desktop}"
cd "$PAPER"
export OMP_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
rm -f results/gpuspeed_v1.json
python compute_speed/bench_gpu.py --part perch --machine "$MACHINE" 2>&1 | grep -v cuda_timer
python compute_speed/bench_gpu.py --part kinglet --machine "$MACHINE"
python compute_speed/bench_gpu.py --part kinglet --machine "$MACHINE" --variant channels_last
TORCHINDUCTOR_COMPILE_THREADS=4 timeout 2400 python compute_speed/bench_gpu.py --part kinglet --machine "$MACHINE" --variant compile
python compute_speed/bench_gpu.py --part launch --machine "$MACHINE"  # batch-1 diagnostic (key kinglet_launch_b1)
