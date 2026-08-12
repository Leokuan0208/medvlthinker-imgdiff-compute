#!/usr/bin/env bash
# Queue-then-run the vision-token extraction.  Both A100s were full when this round started
# (75.1 / 72.7 GiB used), so this polls until one GPU has enough headroom for a 7B bf16 forward
# (measured 23.42 GiB process peak, artifacts/vram_testtime_2026-08-11.json) and only then runs.
# It never kills anything and never oversubscribes.  Resumable at the SPLIT level: a split whose
# .npz already exists is skipped.
set -u
cd ~/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache
export TOKENIZERS_PARALLELISM=false
NEED_MIB=${NEED_MIB:-26000}
OUT=feats_vision
mkdir -p "$OUT" logs

pick_gpu() {
  while true; do
    for g in 0 1; do
      free=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$free" -ge "$NEED_MIB" ]; then echo "$g"; return 0; fi
    done
    sleep 60
  done
}

run_one() {  # $1 = split, $2 = ablate, $3 = extra args
  local split=$1 abl=$2 extra=${3:-}
  local tag=""; [ "$abl" != "none" ] && tag="_$abl"
  local f="$OUT/vis_${split}${tag}.npz"
  if [ -f "$f" ]; then echo "[skip] $f exists"; return 0; fi
  local g; g=$(pick_gpu)
  echo "[run] split=$split ablate=$abl on GPU $g at $(date -Is)"
  CUDA_VISIBLE_DEVICES=$g python3 src/training_methods/extract_vision_tokens.py \
      --split "$split" --ablate "$abl" --out "$OUT" $extra \
      >> "logs/vision_tokens_${split}${tag}.log" 2>&1
  echo "[done] split=$split ablate=$abl rc=$? at $(date -Is)"
}

run_one eval  none  "--verify_causal 1"
run_one train none  ""
run_one eval  blank ""
run_one eval  noise ""
echo "[ALL DONE] $(date -Is)"
