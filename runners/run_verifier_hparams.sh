#!/usr/bin/env bash
# KNOB 3 -- verifier scoring-resolution ladder.  Launch from the repo root.
#   bash runners/run_verifier_hparams.sh <GPU> <MAXPIXELS> [<MAXPIXELS> ...]
# Arms run SEQUENTIALLY on the given card; every arm is resumable (per-triple JSONL).
set -u
cd "$(dirname "$0")/.." || exit 1
GPU="$1"; shift
ADAPTER="${ADAPTER:-ckpts/train/lora_verifier_disjoint}"
TAGPFX="${TAGPFX:-}"
NEED_MIB="${NEED_MIB:-30000}"
for PX in "$@"; do
  # RE-CHECK FREE VRAM IMMEDIATELY BEFORE EVERY MODEL LOAD; queue rather than oversubscribe,
  # and never kill anything. Gives up after 6 h rather than fighting for the card.
  waited=0
  while true; do
    FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$FREE" -ge "$NEED_MIB" ] && break
    [ "$waited" -ge 21600 ] && { echo "GIVING UP: GPU $GPU only ${FREE} MiB free after 6 h"; exit 1; }
    echo "  waiting for GPU $GPU (${FREE} MiB free, need ${NEED_MIB})"; sleep 120; waited=$((waited+120))
  done
  echo "=== GPU $GPU  max_pixels $PX  adapter $ADAPTER  free ${FREE} MiB  $(date -Is) ==="
  HF_HOME=/data/dan/hf_cache OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU" \
    python3 src/cascade_methods/verifier_hparams_score.py --max_pixels "$PX" \
      --adapter "$ADAPTER" ${TAGPFX:+--tag "${TAGPFX}px${PX}"} \
    2>&1 | grep -vE "FutureWarning|UserWarning|^  import pynvml|^  self.setter|Loading checkpoint"
done
echo "=== ALL DONE GPU $GPU $(date -Is) ==="
