#!/usr/bin/env bash
# run_closed_as_open_chain.sh -- BUILD 3 GPU chain, polite on a SHARED pair of A100s.
#
# Every stage waits until its target card actually has the VRAM it needs before loading, and retries
# on OOM instead of fighting another user's process.  Never kills anything.
#
#   nohup bash runners/run_closed_as_open_chain.sh > logs/closed_as_open_chain_2026-08-16.log 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
export TORCHDYNAMO_DISABLE=1

free_mib () { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits; }

# wait_free <gpu> <need_mib> <max_wait_s>
wait_free () {
  local g=$1 need=$2 max=${3:-7200} t=0
  while [ "$(free_mib "$g")" -lt "$need" ]; do
    if [ "$t" -ge "$max" ]; then echo "  !! gpu$g never freed ${need} MiB in ${max}s"; return 1; fi
    sleep 30; t=$((t+30))
  done
  echo "  gpu$g has $(free_mib "$g") MiB free (needed $need) $(date -Is)"
  return 0
}

# try_gpu <need_mib> -> echoes 0 or 1 or nothing
pick_gpu () {
  local need=$1
  for g in 0 1; do
    if [ "$(free_mib "$g")" -ge "$need" ]; then echo "$g"; return 0; fi
  done
  return 1
}

NEED_SCORE=26000     # HF Lingshu-7B bf16 + LoRA + activations, measured headroom

# ---------------------------------------------------------------- 1. verifier scoring (HF, 1 card)
for attempt in $(seq 1 40); do
  g=$(pick_gpu "$NEED_SCORE") || { echo "[score] no card with ${NEED_SCORE} MiB, waiting ($attempt)"; sleep 60; continue; }
  echo "=== [score] attempt $attempt on gpu$g $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=$g python3 src/cascade_methods/closed_as_open_score.py \
      --arms openPRJ_s8 openMEK_s8 closedD_s8
  rc=$?
  echo "=== [score] rc=$rc $(date -Is) ==="
  [ $rc -eq 0 ] && break
  sleep 60
done

echo "CLOSED_AS_OPEN_CHAIN_SCORE_STAGE_EXIT $(date -Is)"
