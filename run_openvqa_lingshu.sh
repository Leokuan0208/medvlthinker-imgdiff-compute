#!/bin/bash
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
LP="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/"
for ds in slake_open vqa_rad_open; do
  CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa.py --model_path "$LP" --tag lingshu32b \
    --dataset $ds --n_samples 1 --temp 0 --ckpt_dir ckpts/openvqa/strong_lingshu --tp 2 \
    --gpu_mem 0.90 --max_model_len 4096
done
echo "OPENVQA_LINGSHU_DONE"
