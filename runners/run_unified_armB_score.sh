#!/bin/bash
# ATTACK 2 phase 2 -- SCORING only, with the priority order that buys the decisive answers first.
# Replaces the scoring half of runners/run_unified_armB_finish.sh once both adapters exist.
#
#   gpu0 = arm B0 (option-only): the four option cells, cheapest-decisive first
#   gpu1 = arm B  (unified):     the OPEN pool FIRST (the interference measurement, which is the
#                                only genuinely new endpoint of this phase), then the option cells
#
#   setsid nohup bash runners/run_unified_armB_score.sh >/dev/null 2>&1 &
#
# Every stage is resumable: unified_pipeline_score.py skips already-written (item, candidate) keys
# and verifier_transfer_eval.py is skipped when its dump already exists.  nohup, never tmux.
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO"
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="$REPO/logs/unified_armB_score.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }

score_opts(){ # <gpu> <out_dir> <tag> <cells>
  local G="$1" OUTD="$2" TAG="$3" CELLS="$4"
  say ">> [gpu$G] SCORE $TAG [$CELLS]"
  CUDA_VISIBLE_DEVICES="$G" timeout -s KILL 25200 python3 \
    src/cascade_methods/unified_pipeline_score.py --adapter "$OUTD" --tag "$TAG" \
    --cells "$CELLS" --deadline_s 21600 --wait_s 3600 --min_free_gib 26 \
    >> "$REPO/logs/unified_score_${TAG}.log" 2>&1
  say "   [gpu$G] SCORE $TAG [$CELLS] rc=$?"
}

score_open(){ # <gpu> <out_dir> <tag>
  local G="$1" OUTD="$2" TAG="$3"
  for DS in slake_open vqa_rad_open pathvqa_open; do
    [ -s "$REPO/$OUTD/transfer_dump_${DS}_lingshu7b.json" ] && { say "SKIP(open done) $TAG $DS"; continue; }
    say ">> [gpu$G] SCORE open $DS for $TAG"
    CUDA_VISIBLE_DEVICES="$G" timeout -s KILL 14400 python3 \
      src/training_methods/verifier_transfer_eval.py --adapter "$OUTD" --datasets "$DS" \
      >> "$REPO/logs/unified_open_${TAG}.log" 2>&1
    say "   [gpu$G] SCORE open $DS rc=$?"
  done
}

B0=ckpts/train/lora_verifier_optiononly_s0
BU=ckpts/train/lora_verifier_unified_s0

say "=== PHASE 2 SCORING START ==="
(
  score_opts 0 "$B0" optiononly_s0 "PATH_VQA_closed,VQA_RAD_closed"
  score_opts 0 "$B0" optiononly_s0 "MedXpertQA-MM"
  score_opts 0 "$B0" optiononly_s0 "PMC_VQA"
  score_open 0 "$B0" optiononly_s0          # the transfer control (amendment 5 diagnostic)
  say "=== [gpu0] arm B0 DONE ==="
) &
P0=$!
(
  score_open 1 "$BU" unified_s0             # FIRST: the interference measurement
  score_opts 1 "$BU" unified_s0 "PATH_VQA_closed,VQA_RAD_closed"
  score_opts 1 "$BU" unified_s0 "MedXpertQA-MM"
  score_opts 1 "$BU" unified_s0 "PMC_VQA"
  say ">> [gpu1] VRAM probe of the OPTION BRANCH (batch=1, the deployed serving shape)"
  CUDA_VISIBLE_DEVICES=1 timeout -s KILL 3600 python3 \
    src/cascade_methods/unified_pipeline_score.py --adapter "$BU" --tag unified_s0 \
    --cells "PATH_VQA_closed,VQA_RAD_closed,MedXpertQA-MM,PMC_VQA" --vram_probe 10 \
    >> "$REPO/logs/unified_vram_unified_s0.log" 2>&1
  say "   [gpu1] VRAM probe rc=$?"
  say "=== [gpu1] arm B DONE ==="
) &
P1=$!
wait $P0; say "gpu0 rc=$?"
wait $P1; say "gpu1 rc=$?"
say "=== PHASE 2 SCORING ALL DONE ==="
