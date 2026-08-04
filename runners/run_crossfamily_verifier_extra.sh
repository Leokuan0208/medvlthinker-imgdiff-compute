#!/bin/bash
# Extra families for the cross-family verifier sweep (more points for the de-correlation law).
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 CROSSFAM_GPU_OK=1
mkdir -p logs/crossfam
H=/data/dan/hf_cache/hub
QOQ="$H/models--ddvd233--QoQ-Med-VL-7B/snapshots/aedfc42114c42fed85e5eef91466131e9597d750/"
CHI=$(ls -d $H/models--manglu3935--Chiron-o1-8B/snapshots/*/ | head -1)

run() {
  local gpu=$1 tag=$2 path=$3 tp=$4; shift 4
  for ds in vqa_rad_open slake_open pathvqa_open; do
    for attempt in 1 2; do
      CUDA_VISIBLE_DEVICES=$gpu timeout 5400 python3 src/cascade_methods/crossfamily_verifier_gpu.py \
        --dataset $ds --model_path "$path" --tag $tag --tp $tp "$@" \
        >> logs/crossfam/${tag}_${ds}.log 2>&1 && break
      echo "[retry] $tag $ds attempt $attempt failed" >> logs/crossfam/${tag}_${ds}.log
    done
  done
  echo "DONE_$tag"
}

run 0 qoqmed7b     "$QOQ" 1 &
run 1 chiron_o1_8b "$CHI" 1 &
wait
echo "CROSSFAM_EXTRA_DONE"
