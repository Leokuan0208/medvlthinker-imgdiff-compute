#!/bin/bash
# Attack B -- everything the ADAPTED arm needs, in order, on one GPU.
#   $1 = seed tag (s0)   $2 = gpu id
# Waits for training, merges the LoRA into full weights, PROVES the merge + vLLM serving are faithful,
# generates the open-text arm in the SAME configuration as the base control, judges + scores it with
# the FROZEN verifier, then runs the MedEvalKit MCQ suite.  Every stage resumable.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
S="${1:-s0}"; GPU="${2:-1}"
ADAPT="ckpts/train/lora_cheapleg_${S}"
MERGED="ckpts/train/merged_cheapleg_${S}"
TAG="adapt7b_${S}"
L="logs/cheapleg_adapt_chain_${S}.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

say "waiting for training $ADAPT ..."
until [ -s "$ADAPT/adapter_model.safetensors" ] && [ -s "$ADAPT/train_config.json" ]; do sleep 60; done
say "training done"

if [ ! -s "$MERGED/merge_report.json" ]; then
  say ">> merge"
  CUDA_VISIBLE_DEVICES=$GPU python3 src/training_methods/merge_cheapleg_lora.py \
    --adapter "$ADAPT" --out "$MERGED" >> "$L" 2>&1 || { say "MERGE FAILED"; exit 1; }
fi

if [ ! -s "results/cascade_methods/artifacts/_cheapleg_merge_validate_${S}.json" ]; then
  say ">> merge validation (HF adapter vs HF merged vs vLLM merged)"
  CUDA_VISIBLE_DEVICES=$GPU python3 src/training_methods/cheapleg_merge_validate.py \
    --adapter "$ADAPT" --merged "$MERGED" --n 60 \
    --out "results/cascade_methods/artifacts/_cheapleg_merge_validate_${S}.json" >> "$L" 2>&1
  say "merge validation rc=$?"
fi

say ">> open-text generation (same config as the base control: cap320, tp=1, gpu_mem 0.35)"
bash runners/run_cheapleg_open_gen.sh "$PWD/$MERGED" "$TAG" "$GPU" 0.35 >> "logs/cheapleg_open_${TAG}.log" 2>&1
bash runners/run_cheapleg_arm_finish.sh "$TAG" "$GPU"
bash runners/run_cheapleg_mcq.sh "$PWD/$MERGED" "cheapleg_${TAG}" "$GPU"
say "CHEAPLEG_ADAPT_CHAIN_DONE $S"
