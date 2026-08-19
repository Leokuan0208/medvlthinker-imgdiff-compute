#!/bin/bash
# Quilt-VQA re-test, chained behind the GPU campaigns so it never contends for a card.
set -u
cd ~/medvlthinker-imgdiff-compute
echo "waiting for campaign2 to release the GPUs ..."
until grep -q "QUEUE COMPLETE" logs/campaign2_supervisor.log 2>/dev/null; do sleep 120; done
echo "campaign2 done at $(date -u +%H:%M:%S); starting the quilt re-test"
python3 src/reporting/supervisor.py --queue runners/campaign4.json --retries 1
echo "CAMPAIGN4_DONE $(date -u +%H:%M:%S)"
