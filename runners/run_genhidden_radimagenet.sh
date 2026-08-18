#!/bin/bash
# Layer-{7,14,21,28} generator-frame hidden states for radimagenet_open as an EVAL cell.
# Same script, same prompt, same max_pixels as the frozen feats_hidden caches -- only --eval_ds and
# --stem_tag differ, so the features are directly comparable to the existing three cells.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/extract_generator_hidden.py \
  --model_path "$L7" --mode generator --split eval \
  --eval_ds radimagenet_open --stem_tag radimagenet --out feats_hidden
echo "GENHIDDEN_RADIMAGENET_DONE"
