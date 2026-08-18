#!/bin/bash
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python3 -X faulthandler src/training_methods/head_eval_bce.py --stage curve --threads 4 \
  --widths 128 256 1024 --fracs 0.1 0.25 0.5 1.0 \
  --out results/cascade_methods/artifacts/head_curve_bce_2026-08-18.json \
  && echo "HEAD_CURVE_DONE $(date -u +%H:%M:%S)" || echo "HEAD_CURVE_FAILED"
