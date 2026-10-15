#!/usr/bin/env bash
# fr1015_chain.sh -- wait until BOTH stage-1 supervisors (generate + judge) have exited -- whether
# their jobs succeeded or failed, so a failed producer cannot deadlock this chain (CLAUDE.md §0) --
# then run stage 2 (extraction) on both GPUs. Always runs from the repo root.
# Usage: nohup bash fr1015_chain.sh <pid_gpu0_stage1> <pid_gpu1_stage1> > logs/fr1015_chain.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  echo "[$(date +%T)] stage-1 supervisor $pid exited"
done
echo "[$(date +%T)] stage 1 done; status:"
python3 src/reporting/supervisor.py --queue $WT/runners/fr1015_gpu0.json --status
python3 src/reporting/supervisor.py --queue $WT/runners/fr1015_gpu1.json --status
for g in 0 1; do
  nohup python3 src/reporting/supervisor.py --queue $WT/runners/fr1015_stage2_gpu$g.json \
    > logs/fr1015_stage2_supervisor_gpu$g.log 2>&1 &
done
wait
echo "[$(date +%T)] stage 2 done"
python3 src/reporting/supervisor.py --queue $WT/runners/fr1015_stage2_gpu0.json --status
python3 src/reporting/supervisor.py --queue $WT/runners/fr1015_stage2_gpu1.json --status
