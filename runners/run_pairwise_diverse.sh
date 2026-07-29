#!/usr/bin/env bash
# COMPOUNDING experiment: score DIVERSE candidate pools with the REAL pairwise verifier.
# Each dataset is a self-guarded GPU run (timeout -s KILL 9000) with ONE retry; resumable.
# tp=1, GPU0. Launch from repo root. Appends to logs/gpu_experiments.log.
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gpu_experiments.log
DATASETS=(vqa_rad_open pathvqa_open slake_open pmc_content)

run_one() {
  local ds="$1"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') pairwise_diverse START ds=$ds (attempt $2) ===" >> "$LOG"
  timeout -s KILL 9000 env PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/pairwise_verifier_diverse.py --dataset "$ds" >> "$LOG" 2>&1
  return $?
}

for ds in "${DATASETS[@]}"; do
  run_one "$ds" 1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') pairwise_diverse ds=$ds attempt1 FAILED rc=$rc -> RETRY ===" >> "$LOG"
    sleep 10
    run_one "$ds" 2
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "=== $(date '+%Y-%m-%d %H:%M:%S') pairwise_diverse ds=$ds RETRY FAILED rc=$rc (continuing) ===" >> "$LOG"
    fi
  fi
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') pairwise_diverse ds=$ds DONE rc=$rc ===" >> "$LOG"
done
echo "=== $(date '+%Y-%m-%d %H:%M:%S') pairwise_diverse ALL DATASETS COMPLETE ===" >> "$LOG"
