#!/bin/bash
# ATTACK A -- GPU handover.
#
# Three arms were launched on two GPUs: i8b (13-tile, GPU0), i8b_1tile + base7b (GPU1).
# The 13-tile arm exists ONLY to settle whether transformers' InternVLProcessor default
# (crop_to_patches=True -> 13 crops, ~3.3k prompt tokens) or the checkpoint's own saved
# preprocessor_config.json (crop_to_patches=false -> 1 crop, ~0.3k tokens) is the right
# inference configuration.  One full cell (PATH_VQA, 3,362 closed items) settles that.
#
# So: as soon as the 13-tile arm finishes PATH_VQA, stop it and give GPU0 to base7b -- the
# MATCHED CONTROL, which is the bottleneck (1.3 it/s while sharing GPU1) and which every
# reported delta depends on.  base7b is resumable per item, so the move costs at most one
# unflushed batch plus a reload.
#
# Only this session's own processes are touched.  Nothing belonging to the concurrent round
# is killed.
set -u
cd ~/medvlthinker-imgdiff-compute
L=logs/i8b_handover.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

say "waiting for the 13-tile arm to finish PATH_VQA ..."
until [ -s ckpts/i8b_cheapleg/i8b/PATH_VQA/metrics.json ]; do sleep 30; done
say "13-tile PATH_VQA done -- its sensitivity job is complete"

# stop ONLY the 13-tile arm (its runner loop and its python child)
pkill -f "run_i8b_cheapleg.sh i8b 0" 2>/dev/null
pkill -f "i8b_cheapleg_eval.py --arm i8b --datasets" 2>/dev/null
sleep 20
say "13-tile arm stopped; GPU0 free"

# move the matched control to the now-free GPU0
pkill -f "run_i8b_cheapleg.sh base7b 1" 2>/dev/null
pkill -f "i8b_cheapleg_eval.py --arm base7b" 2>/dev/null
sleep 20
say "restarting base7b on GPU0 (resumes from its per-item JSONL)"
nohup bash runners/run_i8b_cheapleg.sh base7b 0 32 >> logs/i8b_suite_base7b_outer.log 2>&1 &
say "HANDOVER_DONE"
