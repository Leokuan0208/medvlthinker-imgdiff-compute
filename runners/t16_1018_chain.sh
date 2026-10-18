#!/usr/bin/env bash
# t16_1018_chain.sh -- after both N=16 generation queues EXIT: MedGemma-27B labels for every N=16 candidate not
# already labelled, then merge. Usage: nohup bash t16_1018_chain.sh <pid_sup_gpu0> <pid_sup_gpu1> > logs/t16_1018_chain.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 120; done; done
echo "[$(date +%T)] N=16 queues exited"
python3 src/reporting/supervisor.py --queue $WT/runners/t16_1018_gpu0.json --status
python3 src/reporting/supervisor.py --queue $WT/runners/t16_1018_gpu1.json --status
python3 $WT/src/exp_2026_10/build_xjudge_sc16.py build || exit 1
SNAP=$(ls -d /data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots/*/ | head -1)
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
for g in 0 1; do
  flock -w 172800 logs/.gpu$g.lock env CUDA_VISIBLE_DEVICES=$g python3 -u src/labeling/run_judge.py \
    --judge_model "$SNAP" --tp 1 --gpu_mem 0.90 \
    --preds /data/dan/exp_2026_10/xjudge_sc16/todo_part$g.jsonl > logs/t16_1018_xjudge_gpu$g.log 2>&1 &
done
wait
python3 $WT/src/exp_2026_10/build_xjudge_sc16.py merge
echo "[$(date +%T)] N=16 cross-family labels done"
