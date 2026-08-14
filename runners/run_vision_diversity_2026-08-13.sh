#!/usr/bin/env bash
# SWEEP 3 -- vision-axis candidate diversity. Generation phase only (GPU).
#
# Both arms -- the 12-view portfolio (k=3 samples each) AND the 24-sample iid control at the BASE
# view -- are produced by THIS script in THIS session, same vLLM serving config (tp=1, bf16,
# seed 0, same max_model_len), so the +-0.008 open-text reproducibility caveat cannot contaminate
# the portfolio-vs-iid contrast. Nothing here is compared to a stored number from another config.
#
# SHARED MACHINE. Other sessions are running on both GPUs. This script:
#   * never kills another process;
#   * takes a SMALL fixed slice of VRAM (default 0.30 of an 80GB card ~ 24 GiB, enough for a 7B
#     plus a short-sequence KV cache) instead of the usual 0.85;
#   * waits for that slice to be free, and retries rather than oversubscribing.
# Items are restricted to the EXACT canonical endpoint (slake 645 / vqa_rad 200 / pathvqa 1500)
# via data/visdiv/idx_*.json, extracted from the published sc8 dumps.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 DIVERSITY_GPU_OK=1
mkdir -p logs ckpts/openvqa/visdiv
GPU="${VISDIV_GPU:-1}"
MEM="${VISDIV_MEM:-0.30}"
NEED_MIB=$(python3 -c "print(int($MEM*81920)+4096)")   # slice + headroom

wait_free () {   # wait until GPU $GPU has NEED_MIB free (cap ~4h); never preempts anyone
  for _ in $(seq 1 480); do
    used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    free=$((81920 - used))
    [ "$free" -ge "$NEED_MIB" ] && return 0
    sleep 30
  done
  echo "TIMEOUT waiting for ${NEED_MIB}MiB on GPU$GPU"; return 1
}

for ds in slake_open vqa_rad_open pathvqa_open; do
  for attempt in 1 2 3 4 5 6 7 8; do
    wait_free || exit 1
    echo "=== $(date '+%F %T') START $ds attempt $attempt on GPU$GPU (mem=$MEM) ==="
    CUDA_VISIBLE_DEVICES="$GPU" python3 src/cascade_methods/vision_diversity_gen.py \
      --dataset "$ds" --idx_file "data/visdiv/idx_${ds}.json" \
      --k 3 --k_iid 24 --tp 1 --chunk 32 --seed 0 --gpu_mem "$MEM" \
      >> "logs/visdiv_gen_${ds}.log" 2>&1
    if grep -q "GEN_DONE" "logs/visdiv_gen_${ds}.log"; then
      echo "=== $(date '+%F %T') DONE $ds ==="; break
    fi
    echo "=== $(date '+%F %T') RETRY $ds (engine start failed / interrupted) ==="
    sleep 60
  done
done
echo "VISDIV_GEN_ALL_DONE"
