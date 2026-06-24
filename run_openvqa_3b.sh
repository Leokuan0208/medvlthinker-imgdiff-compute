#!/bin/bash
# 3B (weak) cheap leg for a BIG-GAP cascade vs the existing 32B strong leg.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
M3=/data/dan/weights/MedVLThinker-3B-RL_m23k
for ds in slake_open vqa_rad_open; do
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path $M3 --tag 3b_t0 \
    --dataset $ds --n_samples 1 --temp 0 --ckpt_dir ckpts/openvqa/cheap3b --tp 1 --max_model_len 4096
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path $M3 --tag 3b_sc8 \
    --dataset $ds --n_samples 8 --temp 0.7 --ckpt_dir ckpts/openvqa/cheap3b --tp 1 --max_model_len 4096
done
echo "OPENVQA_3B_DONE"
