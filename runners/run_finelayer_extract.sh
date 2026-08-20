#!/bin/bash
# Finer layer sweep. The deployed head reads layer 21, chosen from a FOUR-point grid (7/14/21/28)
# where 21 beat 28 by 0.017 CV -- so the optimum is somewhere in 15..27 and was never located.
# Layers 10,12,16,18,20,22,24,26 fill that gap on the train pool and the three original eval cells.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
LAYERS="10 12 16 18 20 22 24 26"
CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/extract_generator_hidden.py \
  --model_path "$L7" --mode generator --split eval --layers $LAYERS \
  --stem_tag finelayer --out feats_hidden > logs/finelayer_eval.log 2>&1
echo "FINELAYER_EVAL_DONE $(date -u +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/extract_generator_hidden.py \
  --model_path "$L7" --mode generator --split train --layers $LAYERS \
  --stem_tag finelayer --out feats_hidden > logs/finelayer_train.log 2>&1
echo "FINELAYER_TRAIN_DONE $(date -u +%H:%M:%S)"
