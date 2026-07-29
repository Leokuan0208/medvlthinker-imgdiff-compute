#!/bin/bash
# Phase 3 of the decisive-experiment pipeline: wait for the in-flight matched-B generation to finish, then
# generate the unstyled-DIRECT arm and judge all three new arms. Separate file (not an edit of the running
# chain script) because bash reads a script incrementally -- editing a script while it executes can corrupt
# its read offset.
cd ~/medvlthinker-imgdiff-compute || exit 1
mkdir -p logs

# ---- wait for matched-B generation (bounded 2h; marker-based, no pgrep self-match) ----
w=0
while [ $w -lt 7200 ]; do
  grep -q "MATCHED2_GEN_ALL_DONE" logs/think_matched2_master.log 2>/dev/null && break
  sleep 30; w=$((w+30))
done
echo "matched-B generation settled after ${w}s ($(date))"

bash runners/run_openvqa_direct_unstyled.sh > logs/direct_unstyled_master.log 2>&1
echo "unstyled-direct generation exit=$? ($(date))"

bash runners/run_judge_think_matched.sh   > logs/judge_matched_master.log 2>&1
echo "judge matched-A exit=$? ($(date))"
bash runners/run_judge_think_matched2.sh  > logs/judge_matched2_master.log 2>&1
echo "judge matched-B exit=$? ($(date))"
bash runners/run_judge_direct_unstyled.sh > logs/judge_unstyled_master.log 2>&1
echo "judge unstyled-direct exit=$? ($(date))"

echo "MATCHED_PHASE3_ALL_DONE ($(date))"
