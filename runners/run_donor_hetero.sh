#!/bin/bash
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for DS in pathvqa_open_train slake_open_train kvasir_open; do
  python3 src/training_methods/head_donor_hetero.py --domain $DS --threads 8 --seeds 3
done
echo "DONOR_HETERO_DONE $(date -u +%H:%M:%S)"
