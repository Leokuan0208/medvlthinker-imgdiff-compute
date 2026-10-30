#!/usr/bin/env bash
# Layer study chain (2026-10-22): wait for the extraction supervisor pids given as args, verify every cache, fit 12 layers, fuse.
set -u
M=/home/jamesyang/medvlthinker-imgdiff-compute; WT=$M/.claude/worktrees/exp-2026-10-15; L=$M/logs; D=/data/dan/feats_deep_2026-10-21
cd $WT
say(){ echo "[$(date '+%F %T')] $*" >> $L/layer_study_chain.log; }
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
miss=$(python3 -c "
import json,os
J=[j for g in (0,1) for f in ('runners/deep_1021_gpu%d.json'%g,'runners/deep_1021b_gpu%d.json'%g) for j in json.load(open(f))]
print(' '.join(sorted({j['expect'] for j in J if not os.path.exists(j['expect'])})))")
if [ -n "$miss" ]; then say "STOP: deep caches missing -> $miss"; echo "FAILED missing caches" > $L/layer_study_chain.status; exit 1; fi
say "all deep caches present; fitting 12 layers on 4 CPU queues"
for q in 0 1 2 3; do python3 src/reporting/supervisor.py --queue $WT/runners/layerfit_1022_q$q.json --retries 1 > $L/layerfit1022_supervisor_q$q.log 2>&1 & done
wait
n=$(ls $M/ckpts/train/layer_study_2026-10-22/done_L*.json 2>/dev/null | wc -l); say "layer fits done: $n / 12"
[ "$n" = 12 ] || { say "STOP: not every layer fitted"; echo "FAILED fits $n/12" > $L/layer_study_chain.status; exit 1; }
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u src/exp_2026_10/layer_fusion.py > $WT/logs/layer_fusion_2026-10-22.log 2>&1
say "fusion rc=$?"; echo "DONE" > $L/layer_study_chain.status
