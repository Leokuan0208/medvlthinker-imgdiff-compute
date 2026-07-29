#!/bin/bash
# Serialize the whole decisive-experiment GPU pipeline on the two shared A100s (only one tp=2 job may hold
# the GPUs at a time): wait for the in-flight matched-A generation, then generate matched-B (trace-preserving
# matched prompt), then judge BOTH arms with the headline's judge. Every stage is already individually
# guardrailed (per-chunk timeout+retry, checkpoint/resume); this file only orders them.
cd ~/medvlthinker-imgdiff-compute || exit 1
mkdir -p logs

# ---- 1. wait for the matched-A generation to be COMPLETE (bounded: 2h) ----
# NB: do NOT poll `pgrep -f run_openvqa_think_matched.sh` -- any observer process whose command line
# merely CONTAINS that string (a monitor, a grep) self-matches and the wait never ends. Wait on the
# runner's own completion marker plus the absence of a live generation process instead.
w=0
while [ $w -lt 7200 ]; do
  grep -q "MATCHED_GEN_ALL_DONE" logs/think_matched_master.log 2>/dev/null && break
  pgrep -f "[p]ython3 src/labeling/run_openvqa.py" >/dev/null || { [ $w -gt 120 ] && break; }
  sleep 30; w=$((w+30))
done
echo "matched-A generation settled after ${w}s ($(date))"

# ---- 2. matched-B generation ----
bash runners/run_openvqa_think_matched2.sh > logs/think_matched2_master.log 2>&1
echo "matched-B generation exit=$? ($(date))"

# ---- 3. the unstyled-DIRECT arm (4th cell of the style x reasoning 2x2) ----
bash runners/run_openvqa_direct_unstyled.sh > logs/direct_unstyled_master.log 2>&1
echo "unstyled-direct generation exit=$? ($(date))"

# ---- 4. judge every new arm with the SAME judge as the headline ----
bash runners/run_judge_think_matched.sh   > logs/judge_matched_master.log 2>&1
echo "judge matched-A exit=$? ($(date))"
bash runners/run_judge_think_matched2.sh  > logs/judge_matched2_master.log 2>&1
echo "judge matched-B exit=$? ($(date))"
bash runners/run_judge_direct_unstyled.sh > logs/judge_unstyled_master.log 2>&1
echo "judge unstyled-direct exit=$? ($(date))"

echo "MATCHED_CHAIN_ALL_DONE ($(date))"
