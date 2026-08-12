#!/usr/bin/env bash
# Re-extract the LANGUAGE-SIDE generator hidden states for the EVAL pool with the image destroyed.
# This is the image-ablation control for the language-side bar (arm L) -- it answers "was the
# language side ever vision-blind?", which is the premise attack 1 rests on.
#
# Two shards, one per GPU, exactly as the published cache was built (generator_eval_s{0,1}of2), so
# genframe_data's shard reassembly works unchanged.  Queues rather than oversubscribes; never kills.
# Resumable at the SHARD level: a shard whose .npz exists is skipped.
set -u
cd ~/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache
export TOKENIZERS_PARALLELISM=false
NEED_MIB=${NEED_MIB:-26000}
ABL=${ABL:-noise}
OUT=feats_hidden_${ABL}
mkdir -p "$OUT" logs

wait_gpu() {  # $1 = gpu index
  while true; do
    free=$(nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -ge "$NEED_MIB" ] && return 0
    sleep 60
  done
}

run_shard() {  # $1 = shard, $2 = gpu
  local s=$1 g=$2
  local f="$OUT/generator_eval_s${s}of2.npz"
  if [ -f "$f" ]; then echo "[skip] $f exists"; return 0; fi
  wait_gpu "$g"
  echo "[run] shard=$s ablate=$ABL on GPU $g at $(date -Is)"
  CUDA_VISIBLE_DEVICES=$g python3 -u src/training_methods/extract_generator_hidden_ablated.py \
      --ablate "$ABL" --mode generator --split eval --shard "$s" --nshard 2 --out "$OUT" \
      >> "logs/genhidden_${ABL}_s${s}.log" 2>&1
  echo "[done] shard=$s rc=$? at $(date -Is)"
}

run_shard 0 0 &
P0=$!
run_shard 1 1 &
P1=$!
wait $P0 $P1
echo "[ALL DONE] $(date -Is)"
