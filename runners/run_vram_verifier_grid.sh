#!/usr/bin/env bash
# run_vram_verifier_grid.sh -- ATTACK 4, open half: the remaining (scheme x cap) verifier arms.
# Each waits for a card with room rather than oversubscribing; never kills another process.
# Resumable: each arm appends per-item JSONL and skips ids already scored.
#
#   nf4_skipvisual_cap1280  keeps the VISION TOWER in bf16 while 4-bit-ing the LM -- the mechanism
#                           question, because the verifier's documented failure mode is visual
#                           grounding on short answers.
#   nf4_cap320              both levers at once -- the smallest-config point, and the interaction test.
#
# int8 is deliberately NOT run on the open half: nf4 is the strictly more aggressive scheme, so if
# nf4 is free here, int8 is free a fortiori, and GPU time on a contended machine is better spent on
# the two arms above.  Stated in the artifact rather than left as a silent omission.
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=/data/dan/hf_cache
export PYTHONPATH=/home/jamesyang/.pylibs_vram
LIMIT=${LIMIT:-200}

wait_for_gpu() {
  local need=$1 tries=0
  while [ $tries -lt 900 ]; do
    while read -r idx free; do
      if [ "$(( free / 1024 ))" -ge "$need" ]; then echo "$idx"; return 0; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',')
    sleep 20; tries=$((tries+1))
  done
  return 1
}

run_arm() {                            # $1 arm  $2 quant  $3 cap  $4 extra-flags  $5 need-GiB
  local arm=$1 q=$2 cap=$3 extra=$4 need=$5
  local n; n=$(cat "ckpts/vram_levers/verifier_grid/${arm}/scores_"*.jsonl 2>/dev/null | wc -l)
  if [ "$n" -ge $((LIMIT * 3)) ]; then echo "[skip] $arm already has $n rows"; return 0; fi
  local g; g=$(wait_for_gpu "$need") || { echo "[$arm] no GPU freed -- NOT ATTEMPTED"; return 1; }
  echo "[$arm] launching on gpu $g at $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$g python3 src/cascade/vram_verifier_grid.py --arm "$arm" --quant "$q" \
      --cap "$cap" --limit "$LIMIT" $extra >> "logs/vram_vgrid_${arm}_2026-08-12.log" 2>&1
  echo "[$arm] exit=$? at $(date +%H:%M:%S)"
}

run_arm nf4_skipvisual_cap1280 nf4 1280 "--skip_visual" 16 &
P1=$!
sleep 60
run_arm nf4_cap320 nf4 320 "" 16 &
P2=$!
wait $P1 $P2
echo "VERIFIER GRID EXTRA ARMS DONE"
