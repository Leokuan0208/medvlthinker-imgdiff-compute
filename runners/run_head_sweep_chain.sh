#!/bin/bash
# Head sweep, three rounds, chained.  Round 1 (already running) sweeps every axis on the deployed
# base.  Round 2 re-runs the representation/architecture/optimisation axes on ROUND 1'S WINNER,
# because round 1 explored them on top of the bt objective that round 1 itself refuted.  Round 3
# re-runs the top-8 at 4 seeds so that differences smaller than the ~0.005 seed spread are not
# mistaken for real ones.  Eval is never read by any of this.
cd ~/medvlthinker-imgdiff-compute
NSH=6

echo "waiting for round 1 (74 configs) ..."
until [ "$(grep -h 'sel_eff=' logs/head_sweep_main_s*.log 2>/dev/null | wc -l)" -ge 74 ]; do sleep 60; done
echo "round 1 complete at $(date -u +%H:%M:%S)"

for i in $(seq 0 $((NSH-1))); do
  python3 src/training_methods/head_sweep.py --stage main2 --threads 7 --shard $i --nshard $NSH \
    > logs/head_sweep_main2_s${i}.log 2>&1 &
done
wait
echo "round 2 complete at $(date -u +%H:%M:%S)"

for i in $(seq 0 $((NSH-1))); do
  python3 src/training_methods/head_sweep.py --stage final --threads 7 --shard $i --nshard $NSH \
    --seeds 4 > logs/head_sweep_final_s${i}.log 2>&1 &
done
wait
echo "HEAD_SWEEP_CHAIN_DONE at $(date -u +%H:%M:%S)"
