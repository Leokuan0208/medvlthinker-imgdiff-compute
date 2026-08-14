#!/bin/bash
# $1 = gpu id, $2 = shared settings json (both workers may share one queue; claims are atomic)
# Polite, adaptive, resumable generation runner for a HEAVILY SHARED GPU.
#  - never oversubscribes: waits for room, then sizes the vLLM pool to what is actually free
#  - never kills another process
#  - in tight memory it drops CUDA graphs and shrinks max_model_len (prompts are ~470 tokens)
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
GPU=$1
SETTINGS=$2
NEED_GIB=17            # 7B bf16 weights are 15.6 GiB; below this nothing can run
TOTAL_GIB=79

for attempt in $(seq 1 400); do
  for w in $(seq 1 240); do
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    FREE_GIB=$((FREE_MIB / 1024))
    if [ "$FREE_GIB" -ge "$NEED_GIB" ]; then break; fi
    echo "waiting for gpu$GPU: ${FREE_GIB}GiB free < ${NEED_GIB}GiB needed ($(date +%H:%M:%S))"
    sleep 20
  done
  USE_GIB=$((FREE_GIB - 1))
  if [ "$USE_GIB" -gt 40 ]; then USE_GIB=40; fi
  MEM=$(python3 -c "print(round($USE_GIB/$TOTAL_GIB, 3))")
  EXTRA=""
  if [ "$USE_GIB" -lt 22 ]; then EXTRA="--enforce_eager --max_model_len 1024"; fi
  echo "=== attempt $attempt gpu$GPU mem=$MEM (${FREE_GIB}GiB free) $EXTRA $(date) ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 src/cascade_methods/decoding_sweep_gen.py \
      --settings "$SETTINGS" --out ckpts/openvqa/decoding_sweep --gpu_mem "$MEM" $EXTRA
  rc=$?
  if [ $rc -eq 0 ]; then echo "SWEEP_GPU${GPU}_ALL_DONE"; break; fi
  echo "--- exit $rc, waiting 20s then resuming ---"
  sleep 20
done
