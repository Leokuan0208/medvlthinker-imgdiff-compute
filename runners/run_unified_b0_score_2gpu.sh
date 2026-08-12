#!/bin/bash
# ATTACK 2 -- score arm B0 across BOTH A100s the moment its adapter lands (amendment 6).
#
# Arm B was stopped (amendment 6), so the second card is free.  41,226 option forwards are split by
# CELL across the two cards -- different cells write different JSONL files, so the two processes
# never touch the same file -- and the cross-format transfer (arm B0 on the OPEN pool) follows.
#
#   setsid nohup bash runners/run_unified_b0_score_2gpu.sh >/dev/null 2>&1 &
#
# Resumable everywhere: unified_pipeline_score.py skips already-written (item, candidate) keys and
# verifier_transfer_eval.py is skipped when its dump exists.  nohup, never tmux.  Repo root.
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO"
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="$REPO/logs/unified_b0_score_2gpu.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
B0=ckpts/train/lora_verifier_optiononly_s0
PARENT=2562663      # the run_unified_armB_finish.sh master, verified by command line before any kill

say "=== waiting for $B0/adapter_model.safetensors ==="
while [ ! -s "$REPO/$B0/adapter_model.safetensors" ]; do
  pgrep -f "train_unified_verifier.py .*lora_verifier_optiononly_s0" >/dev/null || {
    sleep 90
    [ -s "$REPO/$B0/adapter_model.safetensors" ] || { say "ABORT: arm B0 trainer gone, no adapter"; exit 1; }
    break
  }
  sleep 60
done
say "adapter present"

# ---- retire the parent runner so it does not also start scoring on gpu0 -------------------------
if ps -p "$PARENT" -o args= 2>/dev/null | grep -q run_unified_armB_finish; then
  say "retiring parent runner $PARENT (its training stage is complete)"
  pkill -KILL -P "$PARENT" 2>/dev/null || true
  kill -KILL "$PARENT" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "unified_pipeline_score.py --adapter $B0" 2>/dev/null || true
  sleep 10
fi

score(){ # <gpu> <cells>
  local G="$1" C="$2"
  say ">> [gpu$G] SCORE optiononly_s0 [$C]"
  CUDA_VISIBLE_DEVICES="$G" timeout -s KILL 25200 python3 \
    src/cascade_methods/unified_pipeline_score.py --adapter "$B0" --tag optiononly_s0 \
    --cells "$C" --deadline_s 21600 --wait_s 2400 --min_free_gib 26 \
    >> "$REPO/logs/unified_score_optiononly_s0_gpu${G}.log" 2>&1
  say "   [gpu$G] SCORE [$C] rc=$?"
}
open1(){ # <gpu> <datasets...>
  local G="$1"; shift
  for DS in "$@"; do
    [ -s "$REPO/$B0/transfer_dump_${DS}_lingshu7b.json" ] && { say "SKIP(open done) $DS"; continue; }
    say ">> [gpu$G] OPEN transfer $DS"
    CUDA_VISIBLE_DEVICES="$G" timeout -s KILL 14400 python3 \
      src/training_methods/verifier_transfer_eval.py --adapter "$B0" --datasets "$DS" \
      >> "$REPO/logs/unified_open_optiononly_s0_gpu${G}.log" 2>&1
    say "   [gpu$G] OPEN $DS rc=$?"
  done
}

# cheapest-decisive cells first on gpu1; the single most expensive cell alone on gpu0
( score 1 "PATH_VQA_closed,VQA_RAD_closed"
  score 1 "MedXpertQA-MM"
  open1 1 slake_open vqa_rad_open
  say ">> [gpu1] VRAM probe of the OPTION BRANCH (batch=1, the deployed serving shape)"
  CUDA_VISIBLE_DEVICES=1 timeout -s KILL 3600 python3 \
    src/cascade_methods/unified_pipeline_score.py --adapter "$B0" --tag optiononly_s0 \
    --cells "PATH_VQA_closed,VQA_RAD_closed,MedXpertQA-MM,PMC_VQA" --vram_probe 8 \
    >> "$REPO/logs/unified_vram_optiononly_s0.log" 2>&1
  say "   [gpu1] VRAM probe rc=$?"
  say "=== gpu1 DONE ===" ) &
P1=$!
( score 0 "PMC_VQA"
  open1 0 pathvqa_open
  say "=== gpu0 DONE ===" ) &
P0=$!
wait $P1; say "gpu1 rc=$?"
wait $P0; say "gpu0 rc=$?"
say "=== ARM B0 SCORING ALL DONE ==="
bash runners/run_unified_armB_analyse.sh >> "$M" 2>&1
say "=== ANALYSIS DONE ==="
