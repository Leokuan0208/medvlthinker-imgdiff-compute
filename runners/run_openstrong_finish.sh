#!/bin/bash
# ATTACK 1 (OPEN-STRONG) -- everything after generation: explode -> judge -> verifier-score.
# Waits for the generation queue, then runs the whole chain unattended.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
CK=ckpts/openvqa/strong_lingshu_bo
NEED2=58368     # 57 GiB on EACH card for the tp=2 judge
NEED1=25600     # 25 GiB on ONE card for the HF verifier

until grep -q "OPENSTRONG_QUEUE_DONE" logs/openstrong_queue.log; do sleep 30; done
echo "=== generation queue finished; exploding ==="

FILES=()
for ds in slake_open vqa_rad_open pathvqa_open; do
  FILES+=("$CK/ckpt_${ds}_l32_n1.jsonl")
  for sd in 0 1 2; do
    SRC=$CK/ckpt_${ds}_l32_bo8_s${sd}.jsonl
    [ -f "$SRC" ] || continue
    python3 src/cascade_methods/explode_sc_for_judge.py "$SRC"
    FILES+=("${SRC%.jsonl}_scexploded.jsonl")
  done
done

echo "=== judging ${#FILES[@]} files (same judge as every published open-text cell) ==="
while true; do
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  [ "$f0" -ge "$NEED2" ] && [ "$f1" -ge "$NEED2" ] && break
  sleep 30
done
CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --gpu_mem 0.70 --preds "${FILES[@]}"
echo "=== judge done; verifier scoring (HF, never vLLM) ==="

pick1() {
  while true; do
    for g in 0 1; do
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      if [ "$f" -ge "$NEED1" ]; then echo $g; return; fi
    done
    sleep 30
  done
}
for sd in 0 1 2; do
  [ -f "$CK/ckpt_pathvqa_open_l32_bo8_s${sd}.jsonl" ] || continue
  g=$(pick1)
  echo "=== verifier-scoring seed $sd on GPU$g ==="
  CUDA_VISIBLE_DEVICES=$g python3 src/cascade_methods/openstrong_score.py \
      --tag l32_bo8_s${sd} --datasets slake_open vqa_rad_open pathvqa_open
done
echo "OPENSTRONG_FINISH_DONE"
