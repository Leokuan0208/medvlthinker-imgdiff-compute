#!/bin/bash
# ATTACK 2 -- FINISH the pre-registered trained arms B0 (option-only) and B (unified).
# The 2026-08-12 07:18 artifact reports them as "NOT MEASURED -- no adapter exists" because both
# A100s were held by three concurrent rounds.  Both cards are idle now, so the two arms run in
# PARALLEL, one per card, each pinned with CUDA_VISIBLE_DEVICES so they cannot collide.
#
#   setsid nohup bash runners/run_unified_armB_finish.sh >/dev/null 2>&1 &
#
# GPU etiquette: each arm is pinned to ONE card and the trainer's own wait_room() polls for a
# sustained 26 GiB free block on that card before touching the weights.  Nothing is ever killed.
# nohup/setsid, never tmux.  Launch from the repo root.
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO"
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="$REPO/logs/unified_armB_finish.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }

# ---- one arm end-to-end on one pinned card -----------------------------------------------------
arm(){   # <gpu> <out_dir> <max_open> <max_option> <seed> <tag> <do_open_cells>
  local G="$1" OUTD="$2" MO="$3" MOPT="$4" SEED="$5" TAG="$6" DOOPEN="$7"
  export CUDA_VISIBLE_DEVICES="$G"
  if [ ! -s "$REPO/$OUTD/adapter_model.safetensors" ]; then
    say ">> [gpu$G] TRAIN $OUTD (open=$MO option=$MOPT seed=$SEED)"
    timeout -s KILL 30000 python3 src/training_methods/train_unified_verifier.py \
      --seed "$SEED" --out_dir "$OUTD" --max_open "$MO" --max_option "$MOPT" --deadline_s 25200 \
      >> "$REPO/logs/unified_train_$(basename "$OUTD").log" 2>&1
    say "   [gpu$G] TRAIN $OUTD rc=$?"
  else
    say "SKIP(train done) $OUTD"
  fi
  [ -s "$REPO/$OUTD/adapter_model.safetensors" ] || { say "ABORT[gpu$G]: no adapter for $OUTD"; return 1; }

  # cheapest-decisive cells first, so a deadline still buys the answer
  for CELLS in "PATH_VQA_closed,VQA_RAD_closed" "MedXpertQA-MM" "PMC_VQA"; do
    say ">> [gpu$G] SCORE $TAG [$CELLS]"
    timeout -s KILL 21600 python3 src/cascade_methods/unified_pipeline_score.py \
      --adapter "$OUTD" --tag "$TAG" --cells "$CELLS" --deadline_s 18000 \
      >> "$REPO/logs/unified_score_${TAG}.log" 2>&1
    say "   [gpu$G] SCORE $TAG [$CELLS] rc=$?"
  done

  if [ "$DOOPEN" = "1" ]; then
    for DS in slake_open vqa_rad_open pathvqa_open; do
      [ -s "$REPO/$OUTD/transfer_dump_${DS}_lingshu7b.json" ] && { say "SKIP(open done) $DS"; continue; }
      say ">> [gpu$G] SCORE open $DS for $TAG"
      timeout -s KILL 14400 python3 src/training_methods/verifier_transfer_eval.py \
        --adapter "$OUTD" --datasets "$DS" \
        >> "$REPO/logs/unified_open_${TAG}.log" 2>&1
      say "   [gpu$G] SCORE open $DS rc=$?"
    done
  fi
  say "=== [gpu$G] $TAG DONE ==="
}

say "=== FINISH ARM B0 (gpu0) + ARM B (gpu1) START ==="
arm 0 ckpts/train/lora_verifier_optiononly_s0 0     10364 0 optiononly_s0 0 &
P0=$!
sleep 90                     # stagger the two CPU data builds so they do not fight for RAM
arm 1 ckpts/train/lora_verifier_unified_s0   10364 10364 0 unified_s0    1 &
P1=$!
wait $P0; say "gpu0 branch rc=$?"
wait $P1; say "gpu1 branch rc=$?"
say "=== FINISH ARM B0 + ARM B ALL DONE ==="
