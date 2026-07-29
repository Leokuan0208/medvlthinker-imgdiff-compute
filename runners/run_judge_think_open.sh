#!/bin/bash
# Judge the Lingshu-32B-THINK open-text answers with the SAME judge pipeline the method uses for its
# open-text accuracy: src/labeling/run_judge.py (neutral MedVLThinker/Qwen2.5-32B grader, judge_ok on
# modal_pred vs gold). Writes ckpt_<ds>_lingshu32b_think.judge.jsonl next to each think dump.
# tp=2, greedy, max_tokens=2 -> fast; timeout -s KILL 3600 + one retry; resumes via done-set.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
T=ckpts/openvqa/strong_lingshu_think
for attempt in 1 2; do
  echo "=== judge-think attempt $attempt ($(date)) ==="
  timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --preds \
    $T/ckpt_slake_open_lingshu32b_think.jsonl \
    $T/ckpt_vqa_rad_open_lingshu32b_think.jsonl \
    $T/ckpt_pathvqa_open_lingshu32b_think.jsonl > logs/judge_think_open.log 2>&1
  rc=$?
  if grep -q "^DONE judge" logs/judge_think_open.log; then echo "judge-think OK (rc=$rc)"; exit 0; fi
  echo "judge-think attempt $attempt did not finish (rc=$rc); retrying"
done
echo "judge-think FAILED after 2 attempts"; exit 1
