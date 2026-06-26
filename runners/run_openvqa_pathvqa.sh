#!/bin/bash
# PathVQA-open (3rd dataset; exact-match fails it -> needs the LLM-judge). n=1500 sample for speed.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
# GPU1: Lingshu-32B strong
( CUDA_VISIBLE_DEVICES=1 python3 src/labeling/run_openvqa.py --model_path "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/" --tag lingshu32b \
    --dataset pathvqa_open --n_samples 1 --temp 0 --n 1500 --ckpt_dir ckpts/openvqa/strong_lingshu --tp 1 --gpu_mem 0.90 --max_model_len 4096
) > logs/pathvqa_l32.log 2>&1 &
P1=$!
# GPU0: Lingshu-7B cheap (t0 + sc8)
( CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/" --tag lingshu7b \
    --dataset pathvqa_open --n_samples 1 --temp 0 --n 1500 --ckpt_dir ckpts/openvqa/cheap_lingshu7b --tp 1 --max_model_len 4096
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py --model_path "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/" --tag lingshu7b_sc8 \
    --dataset pathvqa_open --n_samples 8 --temp 0.7 --n 1500 --ckpt_dir ckpts/openvqa/cheap_lingshu7b --tp 1 --max_model_len 4096
) > logs/pathvqa_l7.log 2>&1 &
P2=$!
wait $P1 $P2
echo "PATHVQA_DONE"
