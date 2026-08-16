#!/usr/bin/env bash
# run_closed_as_open_openfull.sh -- BUILD 3 POST-HOC arm: the open-form prompts at FULLRES.
#
# ⚠️ POST-HOC.  Added after the pre-registered primary endpoint had already been read and settled at
# 0 of 3 cells.  These arms are NOT eligible for the verifier claim and are labelled exploratory
# everywhere they appear.
#
# WHY.  The round's only positive is that dropping MedEvalKit's "Please output 'yes' or 'no'"
# instruction gains +0.041 [+0.031,+0.051] on PATH_VQA_closed -- but every open arm ran at cap320,
# so "change the prompt" is confounded with "drop the resolution".  These two arms hold the prompt
# change and move the resolution back to the deployed fullres, which separates them.
#
# Greedy, n=1, 3 cells = 4,449 items per arm.  Waits for a card, never evicts a co-tenant.
#   nohup bash runners/run_closed_as_open_openfull.sh > logs/closed_as_open_openfull_2026-08-16.log 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
export TORCHDYNAMO_DISABLE=1
NEED=42000
free_mib () { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits; }
for attempt in $(seq 1 60); do
  G=""
  for w in $(seq 1 480); do
    for g in 0 1; do
      if [ "$(free_mib $g)" -ge "$NEED" ]; then G=$g; break; fi
    done
    [ -n "$G" ] && break
    echo "openfull waiting for ${NEED} MiB on one card ($(date +%H:%M:%S))"
    sleep 30
  done
  [ -z "$G" ] && { echo "openfull: never got a card"; exit 1; }
  echo "=== openfull attempt $attempt on gpu$G $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=$G python3 src/cascade_methods/closed_as_open_gen.py \
      --arms openMEK_g_full openPRJ_g_full
  rc=$?
  echo "=== openfull rc=$rc $(date -Is) ==="
  [ $rc -eq 0 ] && { echo "OPENFULL_DONE"; break; }
  sleep 30
done
