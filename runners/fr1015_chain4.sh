#!/usr/bin/env bash
# fr1015_chain4.sh -- wait for the stage-3 chain to EXIT, then run stage 4 (comparability cells).
# Usage: nohup bash fr1015_chain4.sh <pid_of_fr1015_chain3.sh> > logs/fr1015_chain4.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] stage 3 chain exited; starting stage 4"
for g in 0 1; do
  nohup python3 src/reporting/supervisor.py --queue $WT/runners/fr1015_stage4_gpu$g.json \
    > logs/fr1015_stage4_supervisor_gpu$g.log 2>&1 &
done
wait
echo "[$(date +%T)] stage 4 done"
