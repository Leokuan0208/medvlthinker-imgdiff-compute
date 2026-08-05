#!/usr/bin/env bash
# HF full-adapter pairwise on pathvqa, restricted to the PRE-REGISTERED 500-question
# subsample (numpy default_rng(0).choice(1500, 500, replace=False), sorted).
# Registered before any HF pathvqa number was computed: the full HF round-robin over all
# 1500 pathvqa questions costs ~3 GPU-hours at the measured 3.3 rows/s.
# Waits for the slake/vqa_rad HF run on the same GPU to exit, then starts.
set -u
cd ~/medvlthinker-imgdiff-compute
GPU=$1; SHARD=$2; WAITFOR=$3
export PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=$GPU
while pgrep -f "realpairwise_clean_hf.py --dataset ${WAITFOR}" > /dev/null; do sleep 30; done
echo "[hf-queue gpu$GPU] starting pathvqa shard $SHARD $(date)"
python3 src/training_methods/realpairwise_clean_hf.py --dataset pathvqa_open \
  --subsample 500 --subsample_seed 0 \
  --shard "$SHARD" --nshard 2 --batch 16 >> logs/rpc_hf_pathvqa_s${SHARD}.log 2>&1
echo "[hf-queue gpu$GPU] pathvqa shard $SHARD rc=$? $(date)"
