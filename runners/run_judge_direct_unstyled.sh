#!/bin/bash
# Judge the UNSTYLED-DIRECT arm with the SAME judge the headline uses (run_judge.py, judge_ok on modal_pred
# vs gold, neutral MedVLThinker/Qwen2.5-32B grader). Identical invocation to runners/run_judge_think_open.sh.
cd ~/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
T=ckpts/openvqa/strong_lingshu_direct_unstyled
for attempt in 1 2 3; do
  echo "=== judge-unstyled attempt $attempt ($(date)) ==="
  timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --preds \
    $T/ckpt_slake_open_lingshu32b_direct_unstyled.jsonl \
    $T/ckpt_vqa_rad_open_lingshu32b_direct_unstyled.jsonl \
    $T/ckpt_pathvqa_open_lingshu32b_direct_unstyled.jsonl > logs/judge_direct_unstyled.log 2>&1
  rc=$?
  if grep -q "^DONE judge" logs/judge_direct_unstyled.log; then echo "judge-unstyled OK (rc=$rc)"; exit 0; fi
  echo "judge-unstyled attempt $attempt did not finish (rc=$rc); retrying"
  sleep 20
done
echo "judge-unstyled FAILED after 3 attempts"; exit 1
