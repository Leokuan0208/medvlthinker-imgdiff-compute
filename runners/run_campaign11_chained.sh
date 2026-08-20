#!/bin/bash
# Everything else, chained. Waits for the architecture sweep (its winner feeds best_arch_full_eval)
# and for the GEMeX image pull (its cell feeds four jobs). Both are already running.
set -u
cd ~/medvlthinker-imgdiff-compute
echo "waiting for the architecture transfer sweep ..."
until [ -s results/cascade_methods/artifacts/head_arch_transfer_2026-08-19.json ] && \
      grep -q "wrote results" logs/sv_arch_transfer_sweep.log 2>/dev/null; do sleep 120; done
echo "arch sweep done $(date -u +%H:%M:%S)"
echo "waiting for the GEMeX cell (image pull) ..."
for i in $(seq 1 720); do
  [ -s /data/dan/dataset/gemex_cell/gemex_open.json ] && break
  sleep 120
done
echo "gemex cell present: $([ -s /data/dan/dataset/gemex_cell/gemex_open.json ] && echo yes || echo NO-timed-out) $(date -u +%H:%M:%S)"
python3 src/reporting/supervisor.py --queue runners/campaign11.json --retries 1
echo "CAMPAIGN11_DONE $(date -u +%H:%M:%S)"
