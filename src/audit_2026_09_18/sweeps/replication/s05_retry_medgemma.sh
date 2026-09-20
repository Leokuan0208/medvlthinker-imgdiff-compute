#!/bin/bash
# retry wrapper: the CPU head fit segfaults under load (CLAUDE.md landmine). Retry up to 5x at 2 threads.
cd /data/dan/audit_2026-09-18/tmp/replication
for i in 1 2 3 4 5; do
  echo "=== ATTEMPT $i $(date) ===" >> refit/medgemma_L24.log
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python3 s03_refit_worker.py --generator medgemma --layer 24 --seeds 2 --threads 2 >> refit/medgemma_L24.log 2>&1
  if [ -f refit/medgemma_L24.meta.json ]; then echo "OK attempt $i" >> refit/medgemma_L24.log; break; fi
done
