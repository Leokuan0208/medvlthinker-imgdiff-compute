#!/usr/bin/env bash
# run_vram_levers_quant.sh -- ATTACK 4: bitsandbytes footprint of the UNIFIED 7B pipeline.
# ONE MODEL PER PROCESS (the 08-12 retraction's lesson), and each arm waits for a card with enough
# free VRAM rather than oversubscribing a shared GPU.  Never kills another process.
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=/data/dan/hf_cache
export PYTHONPATH=/home/jamesyang/.pylibs_vram

wait_for_gpu() {                       # $1 = GiB needed; echoes a gpu index once one has that free
  local need=$1 tries=0
  while [ $tries -lt 720 ]; do
    while read -r idx free; do
      if [ "$(( free / 1024 ))" -ge "$need" ]; then echo "$idx"; return 0; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',')
    sleep 20; tries=$((tries+1))
  done
  return 1
}

# per-arm requirement: the measured peak (c) of that arm plus headroom, so a cheap nf4 arm is not
# blocked behind a bf16-sized reservation on a contended card.
need_for() { case "$1" in nf4|nf4_skipvisual) echo 14;; int8) echo 18;; *) echo 26;; esac; }

for arm in nf4 int8 nf4_skipvisual bf16_control; do
  out="results/cascade_methods/artifacts/_vram_levers_parts/levers_quant_${arm}.json"
  if [ -s "$out" ]; then echo "[skip] $arm already done"; continue; fi
  g=$(wait_for_gpu "$(need_for "$arm")") || { echo "[$arm] no GPU freed up in time -- NOT ATTEMPTED"; continue; }
  echo "[$arm] launching on gpu $g at $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$g python3 src/cascade/vram_levers.py --part quant --quant_arm "$arm" \
      --suffix "_${arm}" >> "logs/vram_levers_quant_${arm}_2026-08-12.log" 2>&1
  echo "[$arm] exit=$? at $(date +%H:%M:%S)"
done
echo "ALL QUANT ARMS DONE"
