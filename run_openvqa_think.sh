#!/bin/bash
# 32B-THINK strong leg for the open-ended cascade (bigger 7B->32B gap + expensive strong leg that
# justifies self-consistency's K-sample cost). TP=2 (both GPUs, only job running).
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
M32=/data/dan/weights/MedVLThinker-32B-RL_m23k
# pathvqa_open dropped: long descriptive answers are unscoreable by exact-match (7B-nt acc 0.058)
for ds in slake_open vqa_rad_open; do
  CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa.py --model_path $M32 --tag 32b_think \
    --dataset $ds --n_samples 1 --temp 0 --think --ckpt_dir ckpts/openvqa/strong --tp 2 \
    --gpu_mem 0.90 --max_model_len 4096 --max_tokens 512
done
echo "OPENVQA_THINK_DONE"
