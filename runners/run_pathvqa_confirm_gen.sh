#!/bin/bash
# ATTACK D / PART 1 -- CONFIG-SHIFT REPLICATION of the PathVQA-open +0.0269 win.
#
# Round 1 generated every 32B open-text arm at tensor_parallel_size=2, gpu_memory_utilization=0.70
# (runners/run_openstrong_queue.sh).  Its null test N3 FAILED on PathVQA-open (0.0080 deviation)
# and diagnosed vLLM decode nondeterminism ACROSS SERVING CONFIGURATIONS as the cause.
# The +0.0269 effect is only ~3x that drift, so the effect must be shown to survive a config shift.
#
# This runner regenerates BOTH the matched greedy control AND the N=8 pools at
#     tensor_parallel_size = 1   (vs 2)      <- changes every matmul reduction order
#     gpu_memory_utilization = 0.92 (vs 0.70) <- changes KV blocks -> changes batching/scheduling
# with the SAME generation seeds 0/1/2, so config is the only thing that moves.
# Everything else (prompt, cap320, max_tokens, temp, item universe) is the same script.
#
# Lingshu-32B is ~62.3 GiB of bf16 weights; 0.92 * 81 GiB = 74.5 GiB leaves ~12 GiB of KV, which
# is enough at max_model_len 4096 with 64-item chunks.  Falls back to tp=2/gpu_mem 0.90 on failure.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
CK=ckpts/openvqa/strong_lingshu_bo
NEED_MIB=76800          # 75 GiB free on the target card

wait_for_gpu() {   # $1 = device id
  while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1)
    [ "$f" -ge "$NEED_MIB" ] && return
    sleep 30
  done
}

run_one() {   # tag nsamples temp seed nlimit
  local tag=$1 ns=$2 tp=$3 sd=$4 nl=$5
  local out=$CK/ckpt_pathvqa_open_${tag}.jsonl
  for attempt in $(seq 1 6); do
    if [ -f "$out" ]; then
      have=$(wc -l < "$out")
      if [ "$have" -ge "$nl" ]; then echo "SKIP $tag (have $have >= $nl)"; return 0; fi
    fi
    wait_for_gpu 0
    echo "=== launch $tag N=$ns T=$tp seed=$sd tp=1 gpu_mem=0.92 (attempt $attempt) ==="
    CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/openstrong_gen.py \
      --dataset pathvqa_open --ckpt_dir $CK --tp 1 --gpu_mem 0.92 --n $nl \
      --n_samples $ns --temp $tp --seed $sd --tag $tag && return 0
    echo "  attempt $attempt failed (tp=1); trying tp=2 gpu_mem 0.90 fallback"
    CUDA_VISIBLE_DEVICES=0,1 python3 src/cascade_methods/openstrong_gen.py \
      --dataset pathvqa_open --ckpt_dir $CK --tp 2 --gpu_mem 0.90 --n $nl \
      --n_samples $ns --temp $tp --seed $sd --tag $tag && return 0
    sleep 60
  done
  echo "GIVE UP $tag"; return 1
}

# matched greedy control FIRST, in the new config
run_one c2_n1     1 0   0 1500
for sd in 0 1 2; do
  run_one c2_bo8_s$sd 8 0.7 $sd 1500
done
echo "PATHVQA_CONFIRM_GEN_DONE"
