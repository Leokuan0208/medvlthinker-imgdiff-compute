#!/usr/bin/env bash
# fr1015_chain6.sh -- after stage 5 EXITS: the Wasserstein Equilibrium Decoding baseline (arXiv 2605.18313;
# src/exp_2026_10/wed/wed_select.py, authors' code imported) on our full-resolution T=0.7 pools for the
# benchmarks its paper reports (PathVQA-open, VQA-RAD-open, yes/no excluded) plus SLAKE-open.
# Picks go to /data/dan/exp_2026_10/wed/. Usage: nohup bash fr1015_chain6.sh <pid_chain5> > logs/fr1015_chain6.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
CK=/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b
O=/data/dan/exp_2026_10/wed; mkdir -p $O
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] stage 5 exited; starting WED"
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
( flock -w 172800 logs/.gpu0.lock env CUDA_VISIBLE_DEVICES=0 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --gpu \
    --pool $CK/ckpt_pathvqa_open_lingshu7bFR_sc8.jsonl --out $O/wed_pathvqa_open_FR.jsonl > logs/fr1015_wed_pathvqa.log 2>&1 ) &
( for c in vqa_rad_open slake_open; do
    flock -w 172800 logs/.gpu1.lock env CUDA_VISIBLE_DEVICES=1 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --gpu \
      --pool $CK/ckpt_${c}_lingshu7bFR_sc8.jsonl --out $O/wed_${c}_FR.jsonl > logs/fr1015_wed_${c}.log 2>&1
  done ) &
wait
echo "[$(date +%T)] stage 6 (WED) done"; wc -l $O/*.jsonl
