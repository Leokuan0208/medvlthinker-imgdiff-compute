#!/usr/bin/env bash
# ATTACK 2 arm A -- score the option-branch candidates with the EXISTING clean open-text verifier.
# BATCH=1 THROUGHOUT: null test N4 measured max |batch - batch1| = 0.0614 (mean 0.0101, 1/60 item
# argmax flips) with left-padded batching, which is ~8x this project's stated numerics tolerance,
# so batching is INVALID here and --batch_tokens 0 is mandatory.
# Queues politely: waits until a GPU has >= MINFREE GiB free before loading anything.
# HF transformers only (vLLM drops visual LoRA modules).  Launch from the repo root.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/dan/hf_cache
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-1}"
MINFREE="${MINFREE:-26}"
TAG="${TAG:-zeroshot}"
ADAPTER="${ADAPTER:-ckpts/train/lora_verifier_disjoint}"
CELLS="${CELLS:-VQA_RAD_closed,PATH_VQA_closed,MedXpertQA-MM,PMC_VQA}"

CUDA_VISIBLE_DEVICES="$GPU" python3 src/cascade_methods/unified_pipeline_score.py \
  --adapter "$ADAPTER" --tag "$TAG" --cells "$CELLS" \
  --batch_tokens 0 --wait_s 43200 --min_free_gib "$MINFREE"
