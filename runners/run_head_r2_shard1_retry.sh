#!/bin/bash
# Shard 1 of round 2 died with SIGSEGV inside threaded BLAS while shards 0 and 2 ran the same-sized
# configs without incident -- a race, not a size limit.  Retried alone at 4 threads.  The journal
# makes this safe: anything shard 1 had already finished is skipped.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for attempt in 1 2 3; do
  python3 src/training_methods/head_sweep.py --stage main2 --threads 4 --shard 1 --nshard 3 \
    >> logs/head_sweep_main2_r_s1.log 2>&1 && { echo "R2S1_OK attempt $attempt"; exit 0; }
  echo "R2S1 attempt $attempt died; retrying (journal resumes what completed)"
  sleep 10
done
echo "R2S1_GAVE_UP"
