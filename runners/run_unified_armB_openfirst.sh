#!/bin/bash
# ATTACK 2 -- re-prioritise arm B's scoring the moment its adapter lands.
#
# run_unified_armB_finish.sh scores arm B's four OPTION cells first and its OPEN pool last.  The
# open pool is the only genuinely NEW endpoint of this phase (does unifying COST anything on the
# format the verifier was built for), and arm B's training runs ~4.7 h, so "last" may mean "never".
# This watcher waits for arm B's adapter, retires ONLY the gpu1 branch of the parent runner (the
# gpu0 branch, arm B0, is left completely alone), and re-runs arm B's scoring OPEN-FIRST.
#
#   setsid nohup bash runners/run_unified_armB_openfirst.sh >/dev/null 2>&1 &
#
# It kills nothing it did not verify by command line, and nothing belonging to another user.
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO"
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="$REPO/logs/unified_armB_openfirst.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
GPU1_SUBSHELL=2574513
BU=ckpts/train/lora_verifier_unified_s0

say "=== watcher armed; waiting for $BU/adapter_model.safetensors ==="
while [ ! -s "$REPO/$BU/adapter_model.safetensors" ]; do
  # if arm B's trainer died without an adapter, stop waiting rather than spin forever
  pgrep -f "train_unified_verifier.py .*lora_verifier_unified_s0" >/dev/null || {
    sleep 60
    [ -s "$REPO/$BU/adapter_model.safetensors" ] || { say "ABORT: arm B trainer gone, no adapter"; exit 1; }
    break
  }
  sleep 60
done
say "adapter present"

# ---- retire ONLY the gpu1 branch of the parent runner -------------------------------------------
if ps -p "$GPU1_SUBSHELL" -o cmd= 2>/dev/null | grep -q run_unified_armB_finish; then
  KIDS=$(pgrep -P "$GPU1_SUBSHELL" || true)
  say "retiring gpu1 branch pid=$GPU1_SUBSHELL kids=[$KIDS]"
  for k in $KIDS; do pkill -TERM -P "$k" 2>/dev/null || true; kill -TERM "$k" 2>/dev/null || true; done
  kill -TERM "$GPU1_SUBSHELL" 2>/dev/null || true
  sleep 20
  for k in $KIDS; do kill -KILL "$k" 2>/dev/null || true; done
  kill -KILL "$GPU1_SUBSHELL" 2>/dev/null || true
  sleep 10
else
  say "gpu1 branch already gone; nothing to retire"
fi
pkill -KILL -f "unified_pipeline_score.py --adapter $BU" 2>/dev/null || true
sleep 15

say ">> [gpu1] OPEN pool first (the interference measurement)"
for DS in slake_open vqa_rad_open pathvqa_open; do
  [ -s "$REPO/$BU/transfer_dump_${DS}_lingshu7b.json" ] && { say "SKIP(open done) $DS"; continue; }
  CUDA_VISIBLE_DEVICES=1 timeout -s KILL 14400 python3 \
    src/training_methods/verifier_transfer_eval.py --adapter "$BU" --datasets "$DS" \
    >> "$REPO/logs/unified_open_unified_s0.log" 2>&1
  say "   [gpu1] open $DS rc=$?"
done

for CELLS in "PATH_VQA_closed,VQA_RAD_closed" "MedXpertQA-MM" "PMC_VQA"; do
  say ">> [gpu1] SCORE unified_s0 [$CELLS]"
  CUDA_VISIBLE_DEVICES=1 timeout -s KILL 25200 python3 \
    src/cascade_methods/unified_pipeline_score.py --adapter "$BU" --tag unified_s0 \
    --cells "$CELLS" --deadline_s 21600 --wait_s 3600 --min_free_gib 26 \
    >> "$REPO/logs/unified_score_unified_s0.log" 2>&1
  say "   [gpu1] SCORE unified_s0 [$CELLS] rc=$?"
done

say ">> [gpu1] VRAM probe of the OPTION BRANCH (batch=1, the deployed serving shape)"
CUDA_VISIBLE_DEVICES=1 timeout -s KILL 3600 python3 \
  src/cascade_methods/unified_pipeline_score.py --adapter "$BU" --tag unified_s0 \
  --cells "PATH_VQA_closed,VQA_RAD_closed,MedXpertQA-MM,PMC_VQA" --vram_probe 10 \
  >> "$REPO/logs/unified_vram_unified_s0.log" 2>&1
say "   [gpu1] VRAM probe rc=$?"
say "=== [gpu1] arm B DONE ==="
