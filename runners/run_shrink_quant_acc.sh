#!/bin/bash
# ATTACK 3 -- accuracy of a QUANTISED Lingshu-32B strong leg vs its MATCHED bf16 control.
# Both arms go through the SAME driver, the SAME items, the SAME greedy decoding; only the
# weight representation differs, so every delta is attributable to quantisation alone.
# Waits for the VRAM stage to finish first so this round never runs two GPU jobs against
# the other rounds at once.  Resumable per dataset (metrics.json) and per item (gen.jsonl).
set -u
cd ~/medvlthinker-imgdiff-compute
export PYTHONPATH=/home/jamesyang/pylibs_attack3
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs/shrink_quant_acc_2026-08-12.log

while pgrep -f "shrink_quantised_strong_leg.py --stage vram" >/dev/null; do sleep 120; done

for ARM in nf4 bf16; do
  echo "=== acc arm=$ARM $(date) ===" >> "$L"
  /data/dan/medeval_venv/bin/python src/cascade_methods/shrink_quantised_strong_leg.py \
    --stage acc --configs "$ARM" --datasets VQA_RAD,SLAKE --batch_size 8 >> "$L" 2>&1
done
echo "SHRINK_QUANT_ACC_DONE $(date)" >> "$L"
