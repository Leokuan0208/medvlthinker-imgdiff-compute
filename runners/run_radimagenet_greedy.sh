#!/bin/bash
# Lingshu-7B greedy on radimagenet_open -- the one missing arm for promoting it to a 4th open-text
# eval cell.  Everything else (best-of-8 + judge, T=0.4 pool, best-of-32, Lingshu-32B direct + judge)
# already exists on disk.  Same flags as runners/run_openvqa_lingshu7b.sh.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
CUDA_VISIBLE_DEVICES=1 python3 src/labeling/run_openvqa.py --model_path "$L7" --tag lingshu7b \
  --dataset radimagenet_open --n_samples 1 --temp 0 --ckpt_dir ckpts/openvqa/cheap_lingshu7b \
  --tp 1 --max_model_len 4096
echo "RADIMAGENET_GREEDY_DONE"
