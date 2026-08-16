#!/bin/bash
# run_output_bias_gen.sh -- ATTACK 1 (2026-08-17) generation driver.
#   ./runners/run_output_bias_gen.sh <GPU> <SHARD> <NSHARD>
# One vLLM engine load per attempt; the python half is resumable per item, so an engine death
# (exit 17) or an OOM just restarts and continues.  Waits for free VRAM before every load, never
# kills another process.  nohup, never tmux.  Launched from the repo root.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
G="${1:-0}"; S="${2:-0}"; N="${3:-1}"
LOG="logs/output_bias_gen_g${G}_s${S}of${N}.log"
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0

NEED_MB=45000     # Lingshu-7B bf16 weights ~16 GB + KV at gpu_mem 0.85 on an 80 GB card
for ATTEMPT in $(seq 1 12); do
  # --- re-check free VRAM immediately before EVERY model load (contention guard) ---------------
  for W in $(seq 1 120); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G")
    if [ "$FREE" -ge "$NEED_MB" ]; then break; fi
    echo "[wait] GPU $G free=${FREE}MiB < ${NEED_MB}MiB, queueing ($W) $(date -Is)" >> "$LOG"
    sleep 60
  done
  echo "=== attempt $ATTEMPT GPU $G shard $S/$N free=${FREE}MiB $(date -Is) ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$G" python3 src/cascade_methods/output_bias_gen.py \
      --arms id train swap cf_blank cf_na \
      --cells PMC_VQA MedXpertQA-MM SLAKE_closed VQA_RAD_closed PATH_VQA_closed PMC_TRAIN \
      --shard "$S" --nshard "$N" >> "$LOG" 2>&1
  RC=$?
  echo "=== attempt $ATTEMPT exit $RC $(date -Is) ===" >> "$LOG"
  if grep -q "OUTPUT_BIAS_GEN_DONE" "$LOG"; then
    echo "OUTPUT_BIAS_GEN_ALLDONE g${G}s${S}of${N} $(date -Is)" >> "$LOG"; exit 0
  fi
  sleep 20
done
echo "OUTPUT_BIAS_GEN_GAVEUP g${G}s${S}of${N} $(date -Is)" >> "$LOG"
