#!/bin/bash
# Stages 2..N, GATED on round 2 actually being complete.
#
# The previous version advanced whenever its shards EXITED -- including by SIGSEGV -- so a crashed
# shard would have silently produced a partial round 2, and the one-shot eval would then have
# measured a "CV winner" chosen from an incomplete search.  This waits on the journals containing
# the expected number of configs, with a timeout, and refuses to continue if they never arrive.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
WANT=$(python3 -c "
import sys;sys.argv=['x'];sys.path.insert(0,'src/training_methods')
import head_sweep as h;print(len(h.grid_main2()))" 2>/dev/null | tail -1)
echo "round 2 expects $WANT configs"

deadline=$(( $(date +%s) + 14400 ))
while :; do
  have=$(cat results/cascade_methods/artifacts/_head_sweep_journal_main2_*.jsonl 2>/dev/null | grep -c cv_sel_eff || true)
  [ "${have:-0}" -ge "$WANT" ] && { echo "round 2 COMPLETE: $have/$WANT"; break; }
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "ABORT: round 2 stalled at ${have:-0}/$WANT after 4h -- refusing to run the eval on a partial search"
    exit 1
  fi
  running=$(ps -eo args | grep -c '[h]ead_sweep.py --stage main2' || true)
  if [ "${running:-0}" -eq 0 ]; then
    echo "no round-2 workers alive at ${have:-0}/$WANT; relaunching the gap"
    for i in 0 1 2; do
      python3 src/training_methods/head_sweep.py --stage main2 --threads 8 --shard $i --nshard 3 \
        >> logs/head_sweep_main2_r_s${i}.log 2>&1 &
    done
    wait
  fi
  sleep 60
done

echo "=== round 3 (top-8, 4 seeds) ==="
for i in 0 1 2; do
  python3 src/training_methods/head_sweep.py --stage final --threads 8 --shard $i --nshard 3 \
    --seeds 4 > logs/head_sweep_final_r_s${i}.log 2>&1 &
done
wait
echo "HEAD_SWEEP_ROUNDS_DONE $(date -u +%H:%M:%S)"

export HEAD_SWEEP_THREADS=24 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24
echo "=== leave-one-dataset-out ==="
python3 src/training_methods/head_lodo.py --threads 24 --seeds 3 && echo "HEAD_LODO_DONE"
echo "=== donor curve ==="
python3 src/training_methods/head_domain_curve.py --threads 24 --seeds 3 && echo "HEAD_DOMAIN_CURVE_DONE"
echo "=== ONE-SHOT EVAL ==="
python3 src/training_methods/head_eval_bce.py --stage eval --threads 24 --seeds 8 && echo "HEAD_EVAL_BCE_DONE"
echo "=== data x capacity ==="
python3 src/training_methods/head_eval_bce.py --stage curve --threads 24 && echo "HEAD_CURVE_DONE"
echo "HEAD_PIPELINE_DONE $(date -u +%H:%M:%S)"
