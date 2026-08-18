#!/bin/bash
# ONE consolidated CPU pipeline, replacing the four separately-chained runners that depended on
# each other's log markers.  That arrangement was fragile: when three round-2 shards segfaulted the
# chain still emitted its DONE marker, and everything downstream would have run on a partial round.
#
# The segfault itself was a launcher bug, fixed here: OMP_NUM_THREADS was being set from inside
# Python AFTER torch had loaded, which OpenMP ignores, so six shards each span one thread per core
# -- 6x oversubscription on 48 cores.  Round 1 survived it; round 2's hidden=1024 matrices did not.
# HEAD_SWEEP_THREADS is now exported before Python starts and head_sweep.py honours it at import.
#
# Fewer, fatter shards: 3 x 14 threads instead of 6 x 7.  Journals make every stage resumable, so
# the round-2 config that did complete is not recomputed.
set -u
cd ~/medvlthinker-imgdiff-compute
NSH=3
export HEAD_SWEEP_THREADS=14 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14

run_shards () {   # $1 = stage, $2... = extra args
  local stage="$1"; shift
  local pids=()
  for i in $(seq 0 $((NSH-1))); do
    python3 src/training_methods/head_sweep.py --stage "$stage" --threads 14 \
      --shard "$i" --nshard "$NSH" "$@" > "logs/head_sweep_${stage}_r_s${i}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "$(date -u +%H:%M:%S) stage=$stage finished (worst rc=$rc)"
  return 0
}

echo "=== round 2 (axes on the bce winner) ==="; run_shards main2
echo "=== round 3 (top-8, 4 seeds) ===";        run_shards final --seeds 4
echo "HEAD_SWEEP_ROUNDS_DONE $(date -u +%H:%M:%S)"

echo "=== leave-one-dataset-out ==="
python3 src/training_methods/head_lodo.py --threads 40 --seeds 3
echo "HEAD_LODO_DONE $(date -u +%H:%M:%S)"

echo "=== donor curve (RadImageNet) ==="
python3 src/training_methods/head_domain_curve.py --threads 40 --seeds 3
echo "HEAD_DOMAIN_CURVE_DONE $(date -u +%H:%M:%S)"

echo "=== ONE-SHOT EVAL of the CV-selected head ==="
python3 src/training_methods/head_eval_bce.py --stage eval --threads 40 --seeds 8
echo "HEAD_EVAL_BCE_DONE $(date -u +%H:%M:%S)"

echo "=== data x capacity diagnostic ==="
python3 src/training_methods/head_eval_bce.py --stage curve --threads 40
echo "HEAD_PIPELINE_DONE $(date -u +%H:%M:%S)"
