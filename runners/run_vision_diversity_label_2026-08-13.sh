#!/usr/bin/env bash
# SWEEP 3 -- labelling pipeline: explode -> judge -> verifier-score.
#
#  judge  = src/labeling/run_judge.py, MedVLThinker-32B (a NEUTRAL text LLM, not Lingshu) -- the
#           SAME judge that produced the frozen endpoint's `sl` labels via
#           runners/run_verifier_disjoint_retrain.sh step 3. Not reinvented.
#  score  = the CLEAN disjoint LoRA verifier under HF transformers (never vLLM: vLLM 0.9.0.1 drops
#           all 192 visual.* LoRA modules). The scorer always sees the ORIGINAL image at fullres,
#           identical for every arm, so the varied factor stays purely on the generator side.
#
# SHARED MACHINE: waits for capacity, never kills another process, takes a modest VRAM slice.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
DS_ALL="slake_open vqa_rad_open pathvqa_open"

# ---- 1. explode (CPU) ----------------------------------------------------------------------
for ds in $DS_ALL; do
  python3 src/cascade_methods/vision_diversity_explode.py "ckpts/openvqa/visdiv/gen_${ds}.jsonl"
done

# ---- 2. judge (needs both GPUs: 32B tp=2) ---------------------------------------------------
NEED=""
for ds in $DS_ALL; do
  E="ckpts/openvqa/visdiv/gen_${ds}_scexploded.jsonl"; JU="${E%.jsonl}.judge.jsonl"
  e=$(wc -l < "$E"); j=0; [ -f "$JU" ] && j=$(wc -l < "$JU")
  [ "$j" -lt "$e" ] && NEED="$NEED $E"
done
if [ -n "$NEED" ]; then
  for _ in $(seq 1 480); do    # wait for ~150 GiB across the pair (32B bf16 = 64 GiB + KV)
    f0=$((81920 - $(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)))
    f1=$((81920 - $(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)))
    [ "$f0" -ge 45000 ] && [ "$f1" -ge 45000 ] && break
    sleep 30
  done
  echo "=== $(date '+%F %T') JUDGE$NEED ==="
  CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --gpu_mem 0.55 \
    --preds $NEED > logs/visdiv_judge.log 2>&1
  echo "=== $(date '+%F %T') JUDGE rc=$? ==="
fi

# ---- 3. verifier score (HF, batch-1 to match the incumbent convention exactly) --------------
# ~30k unique (idx,answer) pairs at ~3.3 it/s is the pipeline bottleneck, so split by dataset
# across both GPUs when the second one has room; otherwise do everything on GPU1, sequentially.
free_mib () { echo $((81920 - $(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits))); }

for _ in $(seq 1 480); do
  [ "$(free_mib 1)" -ge 30000 ] && break
  sleep 30
done

echo "=== $(date '+%F %T') SCORE start (GPU1 free=$(free_mib 1) MiB, GPU0 free=$(free_mib 0) MiB) ==="
# batch-1 forward passes leave the GPU badly under-utilised, so run several INDEPENDENT shards.
# Sharding partitions the (idx,answer) work list only -- each shard is the identical batch-1 code
# path, so merged scores are bit-identical to an unsharded run.
SC="python3 src/cascade_methods/vision_diversity_score.py"
PIDS=""
if [ "$(free_mib 0)" -ge 40000 ]; then
  echo "    4 shards: 2 on GPU1, 2 on GPU0"
  for sh in 0 1; do
    CUDA_VISIBLE_DEVICES=1 $SC --datasets $DS_ALL --shard $sh --nshard 4 \
      > "logs/visdiv_score_s${sh}of4.log" 2>&1 & PIDS="$PIDS $!"
  done
  for sh in 2 3; do
    CUDA_VISIBLE_DEVICES=0 $SC --datasets $DS_ALL --shard $sh --nshard 4 \
      > "logs/visdiv_score_s${sh}of4.log" 2>&1 & PIDS="$PIDS $!"
  done
else
  echo "    2 shards, both on GPU1 (GPU0 busy)"
  for sh in 0 1; do
    CUDA_VISIBLE_DEVICES=1 $SC --datasets $DS_ALL --shard $sh --nshard 2 \
      > "logs/visdiv_score_s${sh}of2.log" 2>&1 & PIDS="$PIDS $!"
  done
fi
for p in $PIDS; do wait "$p"; done
echo "=== $(date '+%F %T') SCORE done ==="
echo "VISDIV_LABEL_ALL_DONE"
