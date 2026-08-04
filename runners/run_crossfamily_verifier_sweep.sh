#!/bin/bash
# Cross-family zero-shot verifier sweep over the FIXED Lingshu-7B sc8 pools.
# 7/8B models run two-at-a-time (one per GPU); Lingshu-32B runs last on tp=2.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 CROSSFAM_GPU_OK=1
mkdir -p logs/crossfam

H=/data/dan/hf_cache/hub
L7="$H/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
L32="$H/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/"
QW7="$H/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"
IV3="$H/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9/"
MVT="/data/dan/weights/MedVLThinker-7B-RL_m23k"
HUA="/data/dan/weights/HuatuoGPT-Vision-7B"
MG4="$H/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408/"

run() {  # gpu tag path tp extra
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

# ---- pass 1: two 7B models in parallel
run 0 lingshu7b_zs "$L7"  1 &
run 1 qwen25vl7b   "$QW7" 1 &
wait
# ---- pass 2
run 0 mvt7b        "$MVT" 1 &
run 1 internvl3_8b "$IV3" 1 &
wait
# ---- pass 3: extra medical families (either may be unsupported by vLLM -> logged, skipped)
run 0 huatuo7b     "$HUA" 1 &
run 1 medgemma4b   "$MG4" 1 &
wait
# ---- pass 4: same-family scale reference, tp=2
CUDA_VISIBLE_DEVICES=0,1 bash -c 'true'
run 0,1 lingshu32b "$L32" 2
echo "CROSSFAM_SWEEP_DONE"
