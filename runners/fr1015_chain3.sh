#!/usr/bin/env bash
# fr1015_chain3.sh -- wait for the stage-1/2 chain (fr1015_chain.sh) to EXIT, then run stage 3
# (per-benchmark-temperature full-resolution pools: generate, judge, extract) on both GPUs.
# Usage: nohup bash fr1015_chain3.sh <pid_of_fr1015_chain.sh> > logs/fr1015_chain3.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] stage 1+2 chain exited; starting stage 3"
for g in 0 1; do
  nohup python3 src/reporting/supervisor.py --queue $WT/runners/fr1015_stage3_gpu$g.json \
    > logs/fr1015_stage3_supervisor_gpu$g.log 2>&1 &
done
wait
echo "[$(date +%T)] stage 3 done"
