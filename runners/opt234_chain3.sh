#!/usr/bin/env bash
# CPU scoring chain (2026-10-22): each scorer runs as soon as its inputs exist; a failed scorer never blocks the rest.
set -u
M=/home/jamesyang/medvlthinker-imgdiff-compute; WT=$M/.claude/worktrees/exp-2026-10-15; L=$M/logs
cd $WT; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
say(){ echo "[$(date '+%F %T')] $*" >> $L/opt234_chain3.log; }
waitp(){ while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
( waitp 3893800; say "chain1 done -> per-modality T, temperature policies with MedGemma, first baseline compare"
  python3 -u src/exp_2026_10/temp_modality.py > logs/temp_modality_2026-10-22.log 2>&1; say "temp_modality rc=$?"
  python3 -u src/exp_2026_10/temp_labelfree.py > logs/temp_labelfree_rerun_2026-10-22.log 2>&1; say "temp_labelfree rerun rc=$?"
  python3 -u src/exp_2026_10/baselines_compare.py > logs/baselines_compare_after_chain1.log 2>&1; say "baselines_compare (USC, 32B) rc=$?" ) &
( waitp 3870658; waitp 3870657; python3 -u src/exp_2026_10/refit_summary.py > logs/refit_summary_2026-10-22.log 2>&1; say "refit_summary rc=$?" ) &
( waitp 3893800; waitp 3912106; python3 -u src/exp_2026_10/baselines_compare.py > logs/baselines_compare_final.log 2>&1; say "baselines_compare (all) rc=$?" ) &
wait; say "chain3 DONE"
