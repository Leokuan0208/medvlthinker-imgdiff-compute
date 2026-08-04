#!/bin/bash
# Cross-family zero-shot verifier sweep, PART 2: the remaining families.
# InternVL-family chat templates concatenate content as a plain string -> --content_format string
# (with 'auto' every score comes back null; the harness now aborts loudly on that).
# HuatuoGPT-Vision-7B (model_type llava_qwen2, no remote code) is NOT loadable by this vLLM build
# and is deliberately not attempted here -- see logs/crossfam/huatuo7b_*.log.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 CROSSFAM_GPU_OK=1
mkdir -p logs/crossfam
H=/data/dan/hf_cache/hub
IV3="$H/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9/"
MG4="$H/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408/"
QOQ="$H/models--ddvd233--QoQ-Med-VL-7B/snapshots/aedfc42114c42fed85e5eef91466131e9597d750/"
CHI=$(ls -d $H/models--manglu3935--Chiron-o1-8B/snapshots/*/ | head -1)
L32="$H/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/"

run() {
  local gpu=$1 tag=$2 path=$3 tp=$4; shift 4
  for ds in vqa_rad_open slake_open pathvqa_open; do
    for attempt in 1 2; do
      CUDA_VISIBLE_DEVICES=$gpu timeout 7200 python3 src/cascade_methods/crossfamily_verifier_gpu.py \
        --dataset $ds --model_path "$path" --tag $tag --tp $tp "$@" \
        >> logs/crossfam/${tag}_${ds}.log 2>&1 && break
      echo "[retry] $tag $ds attempt $attempt failed" >> logs/crossfam/${tag}_${ds}.log
      sleep 20
    done
  done
  echo "DONE_$tag"
}

# NOTE: the --gpu_mem values below are the ones that actually ran. With the default 0.88 a second
# engine on the same GPU aborts with "Free memory on device ... is less than desired GPU memory
# utilization". If a vLLM job is killed hard, check for an orphaned `VLLM::EngineCore` process
# holding VRAM (ps aux | grep VLLM::EngineCore) before relaunching.
run 0 internvl3_8b "$IV3" 1 --content_format string &
run 1 medgemma4b   "$MG4" 1 --gpu_mem 0.55 &
wait
run 0 qoqmed7b     "$QOQ" 1 --gpu_mem 0.55 &
run 1 chiron_o1_8b "$CHI" 1 --content_format string --gpu_mem 0.85 &
wait
run 0,1 lingshu32b "$L32" 2 --gpu_mem 0.88
echo "CROSSFAM_SWEEP2_DONE"
