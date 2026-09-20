#!/bin/bash
# Cross-family re-judge (audit 2026-09-20). Same script, same prompt, same decoding as the labels of record
# (src/labeling/run_judge.py, default judge MedVLThinker-32B) -- ONLY the judge model changes.
# Takes the project's per-GPU lock so it cannot collide with the auto-campaign.
set -u
ROOT=/home/jamesyang/medvlthinker-imgdiff-compute
WT=$ROOT/.claude/worktrees/audit-2026-09-18
J=/data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots
SNAP=$(ls -d $J/*/ | head -1)
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
echo "[$(date -u +%F\ %T)] judge=$SNAP"
flock -w 600 $ROOT/logs/.gpu0.lock env CUDA_VISIBLE_DEVICES=0 python3 -u $WT/src/labeling/run_judge.py \
  --judge_model "$SNAP" --tp 1 --gpu_mem 0.90 \
  --preds /data/dan/audit_2026-09-18/tmp/me/xjudge/train_lingshu7b_medgemma27b.jsonl
echo "[$(date -u +%F\ %T)] exit=$?"
