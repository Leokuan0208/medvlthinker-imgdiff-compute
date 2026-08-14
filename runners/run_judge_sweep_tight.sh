#!/bin/bash
# Tight-memory variant of runners/run_judge_sweep.sh, for when the GPUs are partly occupied.
# The shared runner is left untouched because sibling branches use it.
#
# WHY THIS EXISTS. run_judge_sweep.sh demands 37 GiB free on BOTH GPUs before it will start. On
# 2026-08-14 the machine settled at 55 GiB free on GPU0 and 34 GiB on GPU1 -- 3 GiB short -- and the
# judge would have waited indefinitely. Rather than evict anything, the requirement was recomputed
# from the model instead of inherited:
#
#   MedVLThinker-32B (Qwen2.5-32B) bf16 weights on disk = 63 GB  -> 31.5 GB/GPU at tp=2
#   KV per token = 64 layers x 8 KV heads x 128 dim x 2 (K,V) x 2 B = 0.262 MB -> 0.131 MB/GPU
#   at max_model_len 2048 that is 0.268 GB/GPU per sequence
#
# so a TEXT-ONLY judge emitting max_tokens=2 needs ~32 GB of weights and well under 2 GB of KV.
# 33 GiB is therefore a real floor with headroom, not a gamble, and it leaves the co-tenant alone.
# Identical to the shared runner in every other respect: same judge, same inputs, resumable by idx.
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
NEED_GIB=33
TOTAL_GIB=79
for attempt in $(seq 1 200); do
  for w in $(seq 1 240); do
    F0=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) / 1024 ))
    F1=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1) / 1024 ))
    MIN=$F0; [ "$F1" -lt "$MIN" ] && MIN=$F1
    if [ "$MIN" -ge "$NEED_GIB" ]; then break; fi
    echo "judge waiting: free gpu0=${F0} gpu1=${F1} GiB, need ${NEED_GIB} on both ($(date +%H:%M:%S))"
    sleep 25
  done
  USE=$((MIN - 1)); [ "$USE" -gt 60 ] && USE=60
  MEM=$(python3 -c "print(round($USE/$TOTAL_GIB, 3))")
  echo "=== judge attempt $attempt mem=$MEM (min free ${MIN}GiB) $(date) ==="
  CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py \
      --preds ckpts/openvqa/decoding_sweep/judgein_slake_open.jsonl \
              ckpts/openvqa/decoding_sweep/judgein_vqa_rad_open.jsonl \
              ckpts/openvqa/decoding_sweep/judgein_pathvqa_open.jsonl \
      --tp 2 --gpu_mem "$MEM" --max_model_len 2048
  rc=$?
  if [ $rc -eq 0 ]; then echo "JUDGE_ALL_DONE"; break; fi
  echo "--- judge exit $rc, waiting 25s then resuming ---"
  sleep 25
done
