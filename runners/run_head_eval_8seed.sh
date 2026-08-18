#!/bin/bash
# THE ONE-SHOT EVAL at the deployed 8-seed readout, 8 threads.
#
# Thread count is the trigger for the SIGSEGVs: round 1 ran 74 configs at 7 threads clean, every
# crash was at 14/16/32, and a full 3-arm smoke test at 8 threads (including hidden=1024) completed
# without incident.  8 threads it is -- slower, and it finishes.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python3 -X faulthandler src/training_methods/head_eval_bce.py --stage eval --threads 8 --seeds 8 \
  --out results/cascade_methods/artifacts/head_eval_bce_2026-08-18.json \
  && echo "HEAD_EVAL_8SEED_DONE $(date -u +%H:%M:%S)" || echo "HEAD_EVAL_8SEED_FAILED"
python3 -X faulthandler src/training_methods/head_eval_bce.py --stage curve --threads 8 \
  --out results/cascade_methods/artifacts/head_curve_bce_2026-08-18.json \
  && echo "HEAD_CURVE_DONE $(date -u +%H:%M:%S)" || echo "HEAD_CURVE_FAILED"
