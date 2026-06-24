#!/bin/bash
# Open-ended medical VQA cascade inference: 7B confidence + 7B self-consistency (GPU0), 32B strong (GPU1).
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
M7=/data/dan/weights/MedVLThinker-7B-RL_m23k
M32=/data/dan/weights/MedVLThinker-32B-RL_m23k
DSETS="slake_open vqa_rad_open pathvqa_open"
mkdir -p ckpts/openvqa/cheap ckpts/openvqa/strong logs

# GPU1: 32B strong answer (temp 0), TP=1
( for ds in $DSETS; do
    CUDA_VISIBLE_DEVICES=1 python3 src/labeling/run_openvqa.py --model_path $M32 --tag 32b_t0 \
      --dataset $ds --n_samples 1 --temp 0 --ckpt_dir ckpts/openvqa/strong --tp 1 \
      --gpu_mem 0.90 --max_model_len 4096
  done ) > logs/openvqa_32b.log 2>&1 &
P32=$!

# GPU0: 7B confidence (temp 0, n1) then self-consistency (temp 0.7, n8)
( for ds in $DSETS; do
    CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path $M7 --tag 7b_t0 \
      --dataset $ds --n_samples 1 --temp 0 --ckpt_dir ckpts/openvqa/cheap --tp 1 --max_model_len 4096
    CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path $M7 --tag 7b_sc8 \
      --dataset $ds --n_samples 8 --temp 0.7 --ckpt_dir ckpts/openvqa/cheap --tp 1 --max_model_len 4096
  done ) > logs/openvqa_7b.log 2>&1 &
P7=$!

wait $P32 $P7
echo "OPENVQA_ALL_DONE"
