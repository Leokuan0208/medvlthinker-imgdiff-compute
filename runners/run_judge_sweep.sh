#!/bin/bash
# Judge the decoding-sweep candidate strings with THE PROJECT'S OWN judge:
# src/labeling/run_judge.py -- MedVLThinker-32B (Qwen2.5-32B backbone, a NEUTRAL text-only grader,
# not the Lingshu model under test), temperature 0, Yes/No logit comparison. Resumable by idx.
# tp=2: ~32 GiB of weights per GPU, so it needs ~36 GiB free on BOTH GPUs. Waits rather than
# oversubscribing, and sizes the pool to whatever is actually free.
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
NEED_GIB=37
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
  USE=$((MIN - 3)); [ "$USE" -gt 60 ] && USE=60
  MEM=$(python3 -c "print(round($USE/$TOTAL_GIB, 3))")
  echo "=== judge attempt $attempt mem=$MEM (min free ${MIN}GiB) $(date) ==="
  CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py \
      --preds ckpts/openvqa/decoding_sweep/judgein_slake_open.jsonl \
              ckpts/openvqa/decoding_sweep/judgein_vqa_rad_open.jsonl \
              ckpts/openvqa/decoding_sweep/judgein_pathvqa_open.jsonl \
      --tp 2 --gpu_mem "$MEM"
  rc=$?
  if [ $rc -eq 0 ]; then echo "JUDGE_ALL_DONE"; break; fi
  echo "--- judge exit $rc, waiting 25s then resuming ---"
  sleep 25
done
