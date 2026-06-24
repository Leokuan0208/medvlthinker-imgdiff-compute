#!/bin/bash
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
for ds in slake_open vqa_rad_open; do
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path "$L7" --tag lingshu7b \
    --dataset $ds --n_samples 1 --temp 0 --ckpt_dir ckpts/openvqa/cheap_lingshu7b --tp 1 --max_model_len 4096
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path "$L7" --tag lingshu7b_sc8 \
    --dataset $ds --n_samples 8 --temp 0.7 --ckpt_dir ckpts/openvqa/cheap_lingshu7b --tp 1 --max_model_len 4096
done
echo "OPENVQA_LINGSHU7B_DONE"
