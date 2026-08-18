#!/bin/bash
# Judge the radimagenet_open Lingshu-7B GREEDY arm -- the always-7B baseline for the new 4th cell.
# Same judge model and same tp as every other judge run in this project (run_judge_all.sh /
# run_judge_kvasir.sh), so the new cell's labels are produced by the same instrument as the pool's.
# CHAINED: waits for the layer-hidden extraction on GPU 0 to release the card before taking tp=2.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

echo "waiting for GENHIDDEN_RADIMAGENET_DONE ..."
until grep -q "GENHIDDEN_RADIMAGENET_DONE" logs/genhidden_radimagenet.log 2>/dev/null; do sleep 30; done
echo "extraction released GPU 0 at $(date -u +%H:%M:%S); starting judge"

CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 --preds \
  ckpts/openvqa/cheap_lingshu7b/ckpt_radimagenet_open_lingshu7b.jsonl
echo "JUDGE_RADIMAGENET_GREEDY_DONE"
