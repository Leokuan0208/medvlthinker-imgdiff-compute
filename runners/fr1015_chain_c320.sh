#!/usr/bin/env bash
# fr1015_chain_c320.sh -- stages 4-6 REVISED for the cap320 final method (2026-10-16; the full-resolution
# pools did not improve the method). Runs in order, each wait on process EXIT:
#   4c  comparability cells at cap320: pools, judge (MedVLThinker-32B), features
#   5c  MedGemma-27B-it labels for every cap320 candidate the final evaluation can pick (both image halves),
#       reusing the 2026-09-20 audit labels where the same (cell, question, answer) was already judged
#   6c  WED (arXiv 2605.18313) on the cap320 T=0.7 pools of PathVQA-open, VQA-RAD-open, SLAKE-open
# Usage: nohup bash fr1015_chain_c320.sh > logs/fr1015_chain_c320.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
CK=/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b
SV="python3 src/reporting/supervisor.py --queue"
echo "[$(date +%T)] stage 4c starting"
for g in 0 1; do $SV $WT/runners/fr1015_stage4c_gpu$g.json > logs/fr1015_stage4c_supervisor_gpu$g.log 2>&1 & done
wait
echo "[$(date +%T)] stage 4c done; stage 5c starting"
python3 $WT/src/exp_2026_10/build_xjudge_c320.py build
SNAP=$(ls -d /data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots/*/ | head -1)
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
for g in 0 1; do
  flock -w 172800 logs/.gpu$g.lock env CUDA_VISIBLE_DEVICES=$g python3 -u src/labeling/run_judge.py \
    --judge_model "$SNAP" --tp 1 --gpu_mem 0.90 \
    --preds /data/dan/exp_2026_10/xjudge_c320/todo_part$g.jsonl > logs/fr1015_xjudge_c320_gpu$g.log 2>&1 &
done
wait
python3 $WT/src/exp_2026_10/build_xjudge_c320.py merge
echo "[$(date +%T)] stage 5c done; stage 6c (WED) starting"
O=/data/dan/exp_2026_10/wed; mkdir -p $O
( flock -w 172800 logs/.gpu0.lock env CUDA_VISIBLE_DEVICES=0 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --gpu --cap cap320 \
    --pool $CK/ckpt_pathvqa_open_lingshu7b_sc8.jsonl --out $O/wed_pathvqa_open_c320.jsonl > logs/fr1015_wed_c320_pathvqa.log 2>&1 ) &
( for c in vqa_rad_open slake_open; do
    flock -w 172800 logs/.gpu1.lock env CUDA_VISIBLE_DEVICES=1 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --gpu --cap cap320 \
      --pool $CK/ckpt_${c}_lingshu7b_sc8.jsonl --out $O/wed_${c}_c320.jsonl > logs/fr1015_wed_c320_${c}.log 2>&1
  done ) &
wait
echo "[$(date +%T)] stage 6c done"; wc -l $O/*c320*.jsonl
