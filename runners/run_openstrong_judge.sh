#!/bin/bash
# ATTACK 1 (OPEN-STRONG) -- explode the N-sample pools and judge every candidate with THE SAME
# judge as every published open-text cell (src/labeling/run_judge.py, MedVLThinker-32B text-only).
# Usage:  bash runners/run_openstrong_judge.sh <tag> [<tag> ...]     e.g.  l32_n1 l32_bo8_s0
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
CK=ckpts/openvqa/strong_lingshu_bo
NEED_MIB=58368

FILES=()
for tag in "$@"; do
  for ds in slake_open vqa_rad_open pathvqa_open; do
    SRC=$CK/ckpt_${ds}_${tag}.jsonl
    [ -f "$SRC" ] || { echo "MISSING $SRC"; continue; }
    if [ "$tag" = "l32_n1" ]; then
      FILES+=("$SRC")                                  # N=1: judge the single answer directly
    else
      python3 src/cascade_methods/explode_sc_for_judge.py "$SRC"
      FILES+=("${SRC%.jsonl}_scexploded.jsonl")
    fi
  done
done
echo "judging ${#FILES[@]} files"

while true; do
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  [ "$f0" -ge "$NEED_MIB" ] && [ "$f1" -ge "$NEED_MIB" ] && break
  sleep 30
done
CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --gpu_mem 0.70 --preds "${FILES[@]}"
echo "OPENSTRONG_JUDGE_DONE"
