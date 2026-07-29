#!/bin/bash
# Judge the MATCHED-PROMPT Lingshu-32B reasoning answers with the SAME judge the headline uses:
# src/labeling/run_judge.py (neutral MedVLThinker/Qwen2.5-32B grader, judge_ok on modal_pred vs gold).
# Identical invocation to runners/run_judge_think_open.sh (tp=2, greedy, max_tokens=2) so the matched
# arm is scored apples-to-apples with the direct and the unmatched-reasoning arms.
# Writes ckpt_<ds>_lingshu32b_think_matched.judge.jsonl next to each matched dump; resumes via done-set.
cd ~/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
T=ckpts/openvqa/strong_lingshu_think_matched
for attempt in 1 2 3; do
  echo "=== judge-matched attempt $attempt ($(date)) ==="
  timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --preds \
    $T/ckpt_slake_open_lingshu32b_think_matched.jsonl \
    $T/ckpt_vqa_rad_open_lingshu32b_think_matched.jsonl \
    $T/ckpt_pathvqa_open_lingshu32b_think_matched.jsonl > logs/judge_think_matched.log 2>&1
  rc=$?
  if grep -q "^DONE judge" logs/judge_think_matched.log; then echo "judge-matched OK (rc=$rc)"; exit 0; fi
  echo "judge-matched attempt $attempt did not finish (rc=$rc); retrying"
  sleep 20
done
echo "judge-matched FAILED after 3 attempts"; exit 1
