#!/bin/bash
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --preds \
  ckpts/openvqa/cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b.jsonl \
  ckpts/openvqa/strong_lingshu/ckpt_pathvqa_open_lingshu32b.jsonl
echo "JUDGE_PATHVQA_DONE"
