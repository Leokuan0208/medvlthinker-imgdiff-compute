#!/usr/bin/env bash
# Attack 3 (COST-FLOOR) rule-2 corroboration measurement.
# NVML energy is only valid on an UNCONTENDED card, so this waits for a clean window
# (util < 8% and >= 30 GiB free, sustained over 3 consecutive 60 s polls) before launching.
# Launch from the repo root, with nohup.  Never tmux.
cd ~/medvlthinker-imgdiff-compute || exit 1
MAXWAIT=${MAXWAIT:-14400}
T0=$(date +%s)
STREAK=0; PICK=""
while true; do
  NOW=$(date +%s); [ $((NOW-T0)) -gt "$MAXWAIT" ] && { echo "TIMEOUT waiting for a clean GPU"; exit 2; }
  CAND=""
  while IFS=, read -r idx used util; do
    idx=$(echo "$idx"|tr -d ' '); used=$(echo "$used"|tr -d ' MiB'); util=$(echo "$util"|tr -d ' %')
    free=$((81920-used))
    if [ "$util" -lt 8 ] && [ "$free" -gt 30720 ]; then CAND=$idx; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
  if [ -n "$CAND" ] && [ "$CAND" = "$PICK" ]; then STREAK=$((STREAK+1)); else STREAK=1; PICK=$CAND; fi
  echo "$(date +%H:%M:%S) cand=${CAND:-none} streak=$STREAK"
  if [ -n "$CAND" ] && [ "$STREAK" -ge 3 ]; then break; fi
  sleep 60
done
echo "=== clean window on GPU $PICK at $(date) ==="
for REP in 1 2; do
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=$PICK python3 src/cascade_methods/cost_floor_measure.py \
      --phase vllm --prefix_caching on --n 20 --warmup 4 --rep $REP \
      >> logs/cost_floor_vllm_on.log 2>&1 || echo "vllm rep$REP FAILED"
done
HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=$PICK python3 src/cascade_methods/cost_floor_measure.py \
    --phase vllm --prefix_caching off --n 20 --warmup 4 --rep 1 \
    >> logs/cost_floor_vllm_off.log 2>&1 || echo "vllm-nopc FAILED"
for REP in 1 2; do
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=$PICK python3 src/cascade_methods/cost_floor_measure.py \
      --phase hf --n 20 --warmup 4 --rep $REP \
      >> logs/cost_floor_hf.log 2>&1 || echo "hf rep$REP FAILED"
done
echo "=== measurement done at $(date) ==="
