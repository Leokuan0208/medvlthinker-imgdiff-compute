#!/bin/bash
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python3 -X faulthandler src/training_methods/head_domain_curve.py --threads 8 --seeds 3 \
  && echo "DONOR_OK $(date -u +%H:%M:%S)" || echo "DONOR_FAILED"
