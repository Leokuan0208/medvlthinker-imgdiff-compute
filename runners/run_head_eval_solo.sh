#!/bin/bash
# THE ONE-SHOT EVAL, ALONE ON THE MACHINE.
#
# Every SIGSEGV so far (round 2 x4, the first eval attempt) happened while another large CPU-BLAS
# job was running; round 1 completed 74 configs when it had the box to itself, including the same
# hidden=1024 config that now crashes.  The fault is concurrency, not size or thread count, so the
# fix is to stop running these jobs in parallel.  One job, many threads, no neighbours.
#
# Arms are run as SEPARATE PROCESSES so each starts from a clean allocator and a crash costs one
# arm rather than the whole measurement.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
export MALLOC_ARENA_MAX=4

python3 src/training_methods/head_eval_bce.py --stage eval --threads 32 --seeds 8 \
  --out results/cascade_methods/artifacts/head_eval_bce_2026-08-18.json
rc=$?
echo "EVAL rc=$rc"
[ $rc -eq 0 ] && echo "HEAD_EVAL_NOW_DONE $(date -u +%H:%M:%S)" || echo "HEAD_EVAL_FAILED"

python3 src/training_methods/head_eval_bce.py --stage curve --threads 32 \
  --out results/cascade_methods/artifacts/head_curve_bce_2026-08-18.json \
  && echo "HEAD_CURVE_NOW_DONE $(date -u +%H:%M:%S)" || echo "HEAD_CURVE_FAILED"
