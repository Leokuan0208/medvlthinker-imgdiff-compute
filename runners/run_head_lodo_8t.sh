#!/bin/bash
# Leave-one-dataset-out at 8 threads (the count that does not segfault), run alongside the 8-seed
# eval.  Answers "is the head's skill domain-specific" on four datasets, which is the question the
# RadImageNet negative raised and which matters more to the roadmap than round 2's refinement.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python3 -X faulthandler src/training_methods/head_lodo.py --threads 8 --seeds 3 \
  && echo "HEAD_LODO_DONE $(date -u +%H:%M:%S)" || echo "HEAD_LODO_FAILED"
python3 -X faulthandler src/training_methods/head_domain_curve.py --threads 8 --seeds 3 \
  && echo "HEAD_DOMAIN_CURVE_DONE $(date -u +%H:%M:%S)" || echo "HEAD_DOMAIN_CURVE_FAILED"
