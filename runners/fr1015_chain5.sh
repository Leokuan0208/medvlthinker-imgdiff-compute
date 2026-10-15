#!/usr/bin/env bash
# fr1015_chain5.sh -- after stage 4 EXITS: build the cross-family judge's work list from every
# full-resolution pool, judge it with MedGemma-27B-it on both GPUs (same run_judge.py, prompt and
# decoding as the judge of record; only the model differs), then join the two label halves.
# Usage: nohup bash fr1015_chain5.sh <pid_of_fr1015_chain4.sh> > logs/fr1015_chain5.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] stage 4 exited; building the MedGemma-27B work list"
python3 $WT/src/exp_2026_10/build_xjudge_fr.py || exit 1
SNAP=$(ls -d /data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots/*/ | head -1)
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
for g in 0 1; do
  flock -w 172800 logs/.gpu$g.lock env CUDA_VISIBLE_DEVICES=$g python3 -u src/labeling/run_judge.py \
    --judge_model "$SNAP" --tp 1 --gpu_mem 0.90 \
    --preds /data/dan/exp_2026_10/xjudge/fr_medgemma27b_part$g.jsonl > logs/fr1015_xjudge_gpu$g.log 2>&1 &
done
wait
cat /data/dan/exp_2026_10/xjudge/fr_medgemma27b_part0.judge.jsonl /data/dan/exp_2026_10/xjudge/fr_medgemma27b_part1.judge.jsonl \
  > /data/dan/exp_2026_10/xjudge/fr_medgemma27b.judge.jsonl
echo "[$(date +%T)] stage 5 done: $(wc -l < /data/dan/exp_2026_10/xjudge/fr_medgemma27b.judge.jsonl) labels"
