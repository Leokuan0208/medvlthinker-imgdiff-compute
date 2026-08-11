#!/bin/bash
# ATTACK 2 (MCQ-TTA) generation, tp=1 on a SINGLE dedicated card.
#
# WHY tp=1.  vLLM charges OTHER processes' resident memory against its own
# `gpu_memory_utilization` budget, so with a neighbour holding 38.6 GB on card 1 a tp=2 Lingshu-32B
# cannot get a single KV block (measured: it died with "No available memory for the cache blocks"
# at util 0.50 at 14:55).  Card 0 is completely free (81,024 MiB), and the MCQ prompts here are
# short, so the 32B fits on one card with a capped context -- which is exactly the lever the vendor
# wrapper documents ("MAX_MODEL_LEN caps the KV cache so a 32B fits at tp=1").
#
# ⚠️ The deployed baseline dump was produced at tp=2.  Whether tp=1 changes the greedy argmax is
# NOT assumed -- it is exactly what pre-registered null test N3 measures (identity view must
# reproduce the stored per-cell accuracy to within 0.005 on the same ids, or the run is invalid).
#
# nohup, never tmux.  Resumable per-item JSONL with a per-batch error guard.  Repo root only.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs/mcq_tta_gen.log
CARD=${CARD:-0}
NEED_FREE=${NEED_FREE:-74000}
GPU_MEM=${GPU_MEM:-0.95}
MML=${MML:-16384}

echo "MCQ_TTA_TP1 START $(date) card=$CARD need=$NEED_FREE util=$GPU_MEM mml=$MML" >> "$L"
for attempt in 1 2 3 4 5 6 7 8; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CARD")
  if [ "$free" -lt "$NEED_FREE" ]; then
    echo "card $CARD only ${free} MiB free, waiting (attempt $attempt) $(date)" >> "$L"; sleep 120; continue
  fi
  echo "=== tp=1 launch on card $CARD, ${free} MiB free (attempt $attempt) $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES="$CARD" /data/dan/medeval_venv/bin/python \
    src/cascade_methods/mcq_tta_generate.py --stage A --tp 1 --gpu_mem "$GPU_MEM" \
    --max_model_len "$MML" --batch 250 \
    --cells PMC_VQA,VQA_RAD_closed,SLAKE_closed,MedXpertQA-MM,PATH_VQA_closed >> "$L" 2>&1 \
    && { echo "MCQ_TTA_GEN_OK $(date)" >> "$L"; break; } \
    || { echo "MCQ_TTA_GEN_RETRY attempt=$attempt $(date)" >> "$L"; sleep 60; }
done

CUDA_VISIBLE_DEVICES="$CARD" /data/dan/medeval_venv/bin/python src/cascade_methods/mcq_tta_cost.py \
  --n 20 --reps 2 --tp 1 --gpu_mem "$GPU_MEM" >> logs/mcq_tta_cost.log 2>&1 \
  && echo "MCQ_TTA_COST_OK $(date)" >> "$L" || echo "MCQ_TTA_COST_FAIL $(date)" >> "$L"
