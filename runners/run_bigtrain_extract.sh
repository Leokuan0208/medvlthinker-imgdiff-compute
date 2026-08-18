#!/bin/bash
# Extract generator-frame hidden states for EVERY judged training candidate, not the 46% the frozen
# cache holds.
#
# WHY.  The head trains on 31,498 of the 68,539 judged rows already on disk.  That subsample exists
# because build_train_rows reproduces a draw specified for the LoRA VERIFIER (train_config.json's
# matched composition, max_train=10364) which the head inherited and nobody revisited.  The
# data x capacity curve says this costs real accuracy: the last data doubling was worth +0.024 to
# +0.036 CV sel_eff at every width, against +0.004 for going 918k -> 7.3M parameters.
#
# radimagenet_open is EXCLUDED: it is serving as an eval cell (radimagenet_cell_2026-08-18.json) and
# must not enter training.  Its value as a fifth DOMAIN is measured separately and cleanly by
# head_domain_curve.py, which splits it by image into donor and held-out halves.
#
# Sharded across both GPUs.  ~60k rows at ~0.35 s/row is ~3 hours wall clock at nshard 2.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"

for sh in 0 1; do
  CUDA_VISIBLE_DEVICES=$sh python3 src/training_methods/extract_generator_hidden.py \
    --model_path "$L7" --mode generator --split train \
    --all_train --exclude_ds radimagenet_open \
    --stem_tag bigtrain --shard $sh --nshard 2 --out feats_hidden \
    > logs/bigtrain_extract_s${sh}.log 2>&1 &
done
wait
echo "BIGTRAIN_EXTRACT_DONE $(date -u +%H:%M:%S)"
