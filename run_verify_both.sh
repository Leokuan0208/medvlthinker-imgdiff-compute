#!/bin/bash
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
# Lingshu-7B self-verify (calibrated cascade)
for ds in slake_open vqa_rad_open; do
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa_verify.py --model_path "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/" --dataset $ds \
    --pred_dir ckpts/openvqa/cheap_lingshu7b --pred_tag lingshu7b --ckpt_dir ckpts/openvqa/cheap_lingshu7b --tag lingshu7b_verify --tp 1
done
# MedVLThinker-7B self-verify (miscalibrated cascade)
for ds in slake_open vqa_rad_open; do
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa_verify.py --model_path /data/dan/weights/MedVLThinker-7B-RL_m23k --dataset $ds \
    --pred_dir ckpts/openvqa/cheap --pred_tag 7b_t0 --ckpt_dir ckpts/openvqa/cheap --tag 7b_verify --tp 1
done
echo "VERIFY_DONE"
