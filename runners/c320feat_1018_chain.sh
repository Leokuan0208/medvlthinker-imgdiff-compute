#!/usr/bin/env bash
# c320feat_1018_chain.sh -- after the N=16 chain EXITS, run the cap320-feature extraction queues on both GPUs.
# Usage: nohup bash c320feat_1018_chain.sh <pid_t16_chain> > logs/c320feat_1018_chain.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 120; done; done
echo "[$(date +%T)] N=16 chain exited; starting cap320 feature extraction"
python3 src/reporting/supervisor.py --queue $WT/runners/c320feat_1018_gpu0.json > logs/c320feat_1018_supervisor_gpu0.log 2>&1 &
python3 src/reporting/supervisor.py --queue $WT/runners/c320feat_1018_gpu1.json > logs/c320feat_1018_supervisor_gpu1.log 2>&1 &
wait
echo "[$(date +%T)] cap320 feature extraction queues exited"
