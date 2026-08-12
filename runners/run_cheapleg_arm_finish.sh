#!/bin/bash
# Attack B -- turn ONE arm's raw open-text generations into a scored transfer dump.
#   $1 = tag (base7b | adapt7b_s0 | ...)   $2 = gpu id
# Steps, each resumable and skipped when already complete:
#   1. explode the 8-sample pools into unique (item, answer) judge rows
#   2. judge them with THE SAME judge as every published open-text number
#      (MedVLThinker-32B, text-only, src/labeling/run_judge.py) -- also judges the temp-0 greedy file
#   3. score every candidate with the FROZEN incumbent verifier under HF transformers
#      (ckpts/train/lora_verifier_disjoint; NEVER vLLM -- it drops the 192 visual.* LoRA modules)
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
TAG="$1"; GPU="$2"
CKD="ckpts/openvqa/cheapleg_${TAG}"
L="logs/cheapleg_finish_${TAG}.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

wait_mem(){ local need="$1" t0=$SECONDS
  while :; do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ $((81920-used)) -ge "$need" ] && return 0
    [ $((SECONDS-t0)) -gt 10800 ] && { say "WAIT_MEM_TIMEOUT gpu$GPU"; return 1; }
    sleep 60
  done; }

say "=== finish arm $TAG on gpu $GPU ==="
NEED=""
for DS in slake_open vqa_rad_open pathvqa_open; do
  SRC="$CKD/ckpt_${DS}_${TAG}_sc8.jsonl"; EXP="${SRC%.jsonl}_scexploded.jsonl"
  [ -s "$SRC" ] || { say "MISSING $SRC -- abort"; exit 1; }
  [ -s "$EXP" ] || python3 src/cascade_methods/explode_sc_for_judge.py "$SRC" >> "$L" 2>&1
  J="${EXP%.jsonl}.judge.jsonl"; E=$(wc -l < "$EXP"); Jn=0; [ -f "$J" ] && Jn=$(wc -l < "$J")
  [ "$Jn" -ge "$E" ] || NEED="$NEED $EXP"
  # the temp-0 greedy arm is judged too, so the true greedy accuracy is measurable (the pipeline's
  # "greedy" is the sc8 MODAL prediction, which is a different quantity)
  G="$CKD/ckpt_${DS}_${TAG}.jsonl"; GJ="${G%.jsonl}.judge.jsonl"
  if [ -s "$G" ]; then Gn=$(wc -l < "$G"); GJn=0; [ -f "$GJ" ] && GJn=$(wc -l < "$GJ")
    [ "$GJn" -ge "$Gn" ] || NEED="$NEED $G"; fi
done
if [ -n "$NEED" ]; then
  say ">> JUDGE$NEED"
  for try in 1 2 3; do
    # MedVLThinker-32B is ~64 GB of bf16 weights: tp=1 needs ~75 GB free on the device.
    wait_mem 75000 || break
    CUDA_VISIBLE_DEVICES=$GPU python3 src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 \
      --preds $NEED >> "$L" 2>&1 && break
    say "JUDGE retry $try"; sleep 120
  done
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 20
fi
for DS in slake_open vqa_rad_open pathvqa_open; do
  EXP="$CKD/ckpt_${DS}_${TAG}_sc8_scexploded.jsonl"; J="${EXP%.jsonl}.judge.jsonl"
  E=$(wc -l < "$EXP"); Jn=0; [ -f "$J" ] && Jn=$(wc -l < "$J")
  [ "$Jn" -ge "$E" ] || { say "ABORT: judge incomplete $DS ($Jn/$E)"; exit 1; }
done
say "judge complete"

OUT="ckpts/cheapleg/scores_${TAG}"
if [ -s "$OUT/transfer_dump_pathvqa_open_lingshu7b.json" ]; then say "SKIP(score done)"; else
  for try in 1 2 3; do
    wait_mem 26000 || break
    say ">> SCORE with the FROZEN verifier (HF)"
    CUDA_VISIBLE_DEVICES=$GPU python3 src/training_methods/cheapleg_score_open.py \
      --gen_dir "$CKD" --tag "$TAG" --out_dir "$OUT" >> "$L" 2>&1 && break
    say "SCORE retry $try"; sleep 120
  done
fi
say "CHEAPLEG_ARM_FINISH_DONE $TAG"
