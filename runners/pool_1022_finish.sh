#!/usr/bin/env bash
# CPU scorers for runners/pool_1022.json (2026-10-22): each runs as soon as the GPU jobs it reads have FINISHED
# (produced or failed), so a failed job never blocks a scorer. Re-runs are safe: every scorer rebuilds its artifact.
set -u
M=/home/jamesyang/medvlthinker-imgdiff-compute; WT=$M/.claude/worktrees/exp-2026-10-15; L=$M/logs; S=$WT/runners/pool_1022.state
cd $WT; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
say(){ echo "[$(date '+%F %T')] $*" >> $L/pool_1022_finish.log; }
waitj(){ for j in "$@"; do while [ ! -f $S/$j.rc ]; do sleep 120; done; done; }
X="python3 $WT/src/exp_2026_10/build_xjudge_inputs.py"
( waitj modality; python3 -u src/exp_2026_10/temp_modality.py > logs/temp_modality_2026-10-22.log 2>&1; say "temp_modality rc=$?" ) &
( waitj xj_lf0 xj_lf1; $X merge >> $L/pool_1022_finish.log 2>&1
  python3 -u src/exp_2026_10/temp_labelfree.py > logs/temp_labelfree_medgemma_2026-10-22.log 2>&1; say "temp_labelfree (MedGemma labels) rc=$?" ) &
( waitj usc_T07 usc_Tmap xj_g32_0 xj_g32_1 loraver_f0 loraver_f1; $X merge >> $L/pool_1022_finish.log 2>&1
  python3 -u src/exp_2026_10/baselines_compare.py > logs/baselines_compare_mid.log 2>&1; say "baselines_compare (USC, 32B, LoRA) rc=$?" ) &
wait
waitj $(python3 -c "import json;print(' '.join(j['name'] for j in json.load(open('$WT/runners/pool_1022.json'))))")
$X merge >> $L/pool_1022_finish.log 2>&1
python3 -u src/exp_2026_10/temp_labelfree.py > logs/temp_labelfree_final_2026-10-22.log 2>&1; say "temp_labelfree final rc=$?"
python3 -u src/exp_2026_10/temp_modality.py > logs/temp_modality_final_2026-10-22.log 2>&1; say "temp_modality final rc=$?"
python3 -u src/exp_2026_10/baselines_compare.py > logs/baselines_compare_final.log 2>&1; say "baselines_compare final rc=$?"
say "finish DONE"
