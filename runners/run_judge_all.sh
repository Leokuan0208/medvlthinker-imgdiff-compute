#!/bin/bash
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --preds \
  ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.jsonl \
  ckpts/openvqa/cheap_lingshu7b/ckpt_vqa_rad_open_lingshu7b.jsonl \
  ckpts/openvqa/strong_lingshu/ckpt_slake_open_lingshu32b.jsonl \
  ckpts/openvqa/strong_lingshu/ckpt_vqa_rad_open_lingshu32b.jsonl \
  ckpts/openvqa/cheap/ckpt_slake_open_7b_t0.jsonl \
  ckpts/openvqa/cheap/ckpt_vqa_rad_open_7b_t0.jsonl
echo "JUDGE_DONE"
