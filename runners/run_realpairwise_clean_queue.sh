#!/usr/bin/env bash
# Chained queue for the clean real-pairwise replication. Launched from the repo root.
# GPU arg $1, then a list of "script:dataset" jobs.
set -u
cd ~/medvlthinker-imgdiff-compute
GPU=$1; shift
export PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=$GPU
PY=/data/dan/medeval_venv/bin/python
# wait for any pairwise job still holding this GPU
while pgrep -f "realpairwise_clean_gpu.py --dataset pathvqa_open --shard ${GPU}" > /dev/null; do sleep 20; done
for job in "$@"; do
  s="${job%%:*}"; d="${job##*:}"
  echo "[queue gpu$GPU] $s $d  $(date)"
  $PY src/training_methods/$s --dataset $d >> logs/rpc_queue_gpu${GPU}.log 2>&1
  echo "[queue gpu$GPU] finished $s $d rc=$?  $(date)"
done
echo "[queue gpu$GPU] ALL DONE $(date)"
