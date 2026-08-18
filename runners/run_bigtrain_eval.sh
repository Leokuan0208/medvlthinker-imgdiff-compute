#!/bin/bash
# Does 2.2x the training data reach EVAL?
# The data x capacity curve predicted +0.024 to +0.036 CV from the last doubling. The frozen head
# uses 31,498 of 68,539 judged rows; this fits on all 60,384 (radimagenet excluded, it is an eval
# cell) and runs the same one-shot eval, with the SAME arms so the comparison is like-for-like.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python3 src/training_methods/head_eval_bce.py --stage eval --threads 8 --seeds 8 --bigtrain \
  --out results/cascade_methods/artifacts/head_eval_bigtrain_2026-08-18.json \
  && echo "BIGTRAIN_EVAL_DONE $(date -u +%H:%M:%S)" || echo "BIGTRAIN_EVAL_FAILED"
