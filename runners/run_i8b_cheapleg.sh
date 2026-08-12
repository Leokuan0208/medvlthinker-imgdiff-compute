#!/bin/bash
# ATTACK A -- Lingshu-I-8B as the cascade's cheap leg, plus its MATCHED Lingshu-7B control.
#   $1 = arm (i8b | base7b)   $2 = gpu id   $3 = batch size (default 32)
#
# Both arms go through THE SAME driver (src/cascade_methods/i8b_cheapleg_eval.py), which imports
# MedEvalKit's dataset classes / prompts / cal_metrics verbatim and supplies an HF-transformers
# model object.  MedEvalKit itself is NOT modified.  vLLM cannot serve Lingshu-I-8B at all
# (see logs/i8b_vllm_try.log), so the stored vLLM numbers cannot be the control -- hence the
# matched base7b arm.
#
# Datasets are ordered SMALLEST-EFFECTIVE-FIRST so the load-bearing PathVQA signal lands early
# and PMC_VQA (33,430 items, 3x everything else combined) runs last.
# Every stage is resumable: the driver appends per-item JSONL and skips a dataset whose
# metrics.json already exists.  GPU etiquette: waits for free VRAM, never kills another process.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TORCHDYNAMO_DISABLE=1
ARM="${1:-i8b}"; GPU="${2:-0}"; BS="${3:-32}"
L="logs/i8b_cheapleg_${ARM}.log"
NEED_MB=26000

wait_mem(){ local t0=$SECONDS
  while :; do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ $((81920-used)) -ge "$NEED_MB" ] && return 0
    [ $((SECONDS-t0)) -gt 21600 ] && { echo "WAIT_MEM_TIMEOUT gpu$GPU" >> "$L"; return 1; }
    sleep 60
  done; }

echo "=== $ARM on gpu $GPU bs=$BS $(date) ===" >> "$L"
for DS in PATH_VQA SLAKE VQA_RAD MedXpertQA-MM PMC_VQA; do
  if [ -s "ckpts/i8b_cheapleg/${ARM}/${DS}/metrics.json" ]; then
    echo "SKIP(done) $ARM $DS" >> "$L"; continue; fi
  for try in 1 2 3; do
    wait_mem || break
    echo ">> RUN $ARM $DS try=$try $(date)" >> "$L"
    CUDA_VISIBLE_DEVICES=$GPU /data/dan/medeval_venv/bin/python \
      src/cascade_methods/i8b_cheapleg_eval.py --arm "$ARM" --datasets "$DS" --batch_size "$BS" \
      >> "$L" 2>&1 && break
    echo "RETRY $ARM $DS" >> "$L"; sleep 60
  done
done
echo "I8B_CHEAPLEG_SUITE_DONE $ARM $(date)" >> "$L"
