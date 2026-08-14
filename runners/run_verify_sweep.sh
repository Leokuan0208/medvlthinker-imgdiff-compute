#!/bin/bash
# $1 = gpu id, $2 = shard, $3 = nshard
# Score sweep candidates with the FROZEN incumbent LoRA verifier under HF transformers
# (vLLM drops visual.* LoRA modules -- CLAUDE.md landmine -- so HF only).
# Resumable: the score cache is append-only, keyed by (ds, idx, exact answer text).
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
NEED_GIB=16
for attempt in $(seq 1 200); do
  for w in $(seq 1 240); do
    FREE_GIB=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1") / 1024 ))
    if [ "$FREE_GIB" -ge "$NEED_GIB" ]; then break; fi
    echo "verify waiting for gpu$1: ${FREE_GIB}GiB free ($(date +%H:%M:%S))"; sleep 30
  done
  echo "=== verify attempt $attempt gpu$1 shard$2 (${FREE_GIB}GiB free) $(date) ==="
  CUDA_VISIBLE_DEVICES=$1 python3 src/cascade_methods/decoding_sweep_verify.py \
      --work ckpts/openvqa/decoding_sweep/verifier_work.json \
      --cache ckpts/openvqa/decoding_sweep/vscore_cache \
      --shard "$2" --nshard "$3"
  rc=$?
  if [ $rc -eq 0 ]; then echo "VERIFY_SHARD$2_ALL_DONE"; break; fi
  echo "--- verify exit $rc, waiting 30s then resuming ---"; sleep 30
done
