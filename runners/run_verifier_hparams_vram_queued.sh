#!/usr/bin/env bash
# KNOB 3 -- VRAM + batch-1 latency for the verifier scoring-resolution ladder (7 rungs).
# Launch from the repo root:  bash runners/run_verifier_hparams_vram_queued.sh <GPU>
#
# The (d) convention of vram_testtime_2026-08-11.json is board-used MINUS a pre-run baseline,
# so it is only meaningful on an EXCLUSIVE card -- free space is not enough, a co-tenant's
# allocations land in the same board reading.  This wrapper therefore waits for the card to be
# essentially EMPTY (not merely to have room), never kills anything, and gives up rather than
# fighting for the GPU.  If it gives up, the rung is reported "not measured" -- never estimated.
set -u
cd "$(dirname "$0")/.." || exit 1
GPU="${1:-1}"
IDLE_MIB="${IDLE_MIB:-1024}"        # board `used` must fall below this = exclusive card
MAX_WAIT="${MAX_WAIT:-14400}"       # 4 h, then give up cleanly

waited=0
while true; do
  USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$USED" -le "$IDLE_MIB" ] && break
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "GIVING UP: GPU $GPU still has ${USED} MiB resident after $((MAX_WAIT/3600)) h."
    echo "VRAM_LADDER_NOT_MEASURED $(date -Is)"
    exit 3
  fi
  echo "  waiting for an EXCLUSIVE GPU $GPU (${USED} MiB resident, need <= ${IDLE_MIB})  $(date -Is)"
  sleep 60; waited=$((waited+60))
done

echo "=== GPU $GPU exclusive (${USED} MiB resident) -- starting VRAM ladder $(date -Is) ==="
HF_HOME=/data/dan/hf_cache OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU" \
  python3 src/cascade_methods/verifier_hparams_vram.py \
  2>&1 | grep -vE "FutureWarning|UserWarning|^  import pynvml|^  self.setter|Loading checkpoint"
echo "=== VRAM LADDER DONE $(date -Is) ==="
