#!/bin/bash
# Fine-layer features for the THREE NEW CELLS.  The finelayer sweep can otherwise only measure
# transfer among the four training domains -- and the whole point of choosing a layer on transfer
# is to choose it on the cells where the head actually fails (omnimed -0.0489, vqamed -0.0259).
# Two GPUs, split by cell size so both finish together.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
LAYERS="10 12 16 18 20 22 24 26"
run () {  # $1=gpu  $2=cell
  CUDA_VISIBLE_DEVICES=$1 python3 src/training_methods/extract_generator_hidden.py \
    --model_path "$L7" --mode generator --split eval --eval_ds "$2" --layers $LAYERS \
    --stem_tag "finelayer_$2" --out feats_hidden > "logs/finelayer_$2.log" 2>&1
  echo "FINELAYER_${2}_DONE $(date -u +%H:%M:%S)"
}
( run 0 kvasir_x1_open; run 0 vqamed_open ) &
( run 1 omnimed_open ) &
wait
echo "FINELAYER_NEWCELLS_ALL_DONE $(date -u +%H:%M:%S)"
