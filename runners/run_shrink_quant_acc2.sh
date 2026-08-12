#!/bin/bash
# ATTACK 3 (2026-08-12, RE-LAUNCH) -- accuracy of a QUANTISED Lingshu-32B strong leg vs its
# MATCHED bf16 control.
#
# The 2026-08-12 03:29 attempt (runners/run_shrink_quant_acc.sh) wrote 2,545 error rows per arm:
# QuantHFVLM never set `crop_to_patches`, which drv.HFVLM._encode reads on every item.  Fixed in
# src/cascade_methods/shrink_quantised_strong_leg.py; the poisoned checkpoints were MOVED to
# ckpts/_failed_shrink_quant_20260812/ (never deleted) so the resume logic cannot pick them up.
#
# Both arms run the SAME driver, the SAME MedEvalKit items/prompts/metrics, the SAME greedy
# decoding and the SAME batch size.  Only the weight representation differs, so every delta is
# attributable to quantisation alone.  Per-dataset and per-item resumable.
#
#   $1 = arm (bf16 | nf4)   $2 = gpu id   $3 = datasets   $4 = batch size
#
# GPU etiquette: the arm is PINNED to one card via SHRINK_GPU and never touches the other.
# bf16 needs ~64 GiB so it takes a whole card; nf4 needs ~20 GiB and leaves ~60 GiB free.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/jamesyang/pylibs_attack3

ARM="${1:?arm}"; GPU="${2:?gpu}"; DS="${3:-VQA_RAD,SLAKE,PATH_VQA}"; BS="${4:-4}"
# GPU="auto" leaves SHRINK_GPU unset, which makes the driver call its own wait_for_vram():
# it polls until a card has cfg["need_mb"] free and only then loads.  Use auto whenever another
# round holds a card -- the bf16 arm needs ~62 GiB of weights and must never be squeezed in
# next to someone else's job.
if [ "$GPU" != "auto" ]; then export SHRINK_GPU="$GPU"; fi
L="logs/shrink_quant_acc2_${ARM}.log"

echo "=== ATTACK3 acc arm=$ARM gpu=$GPU datasets=$DS bs=$BS $(date) ===" >> "$L"
for try in 1 2 3; do
  /data/dan/medeval_venv/bin/python src/cascade_methods/shrink_quantised_strong_leg.py \
    --stage acc --configs "$ARM" --datasets "$DS" --batch_size "$BS" >> "$L" 2>&1 && break
  echo "RETRY $ARM try=$try $(date)" >> "$L"; sleep 60
done
echo "SHRINK_QUANT_ACC2_DONE $ARM $(date)" >> "$L"
