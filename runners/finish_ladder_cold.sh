#!/bin/bash
# 2026-08-14 COLD temperature ladder: label -> score -> analyse.
# Generation is done by runners/run_ladder_cold_greedy.sh (T=0, n=1) and runners/run_sweep_adaptive.sh
# (the n=8 rungs) over artifacts/_decoding_ladder_cold_settings.json.
#
# Every stage below is the EXISTING sweep machinery, unchanged, pointed at the new setting names:
#   decoding_sweep_prepare.py   content-addressed judge inputs + verifier work list (new strings only)
#   run_judge_sweep_tight.sh    the project's own judge (MedVLThinker-32B, temp 0), memory-polite
#   run_verify_sweep.sh         the FROZEN incumbent LoRA verifier under HF transformers (never vLLM)
#   decoding_ladder_cold.py     this round's analysis -> artifacts/decoding_ladder_cold_2026-08-14.json
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

RUNGS="T07r,T03r,T05r,T04,T02,T01,T005,T00"

echo "=== phase 2: judge [$RUNGS] $(date) ==="
python3 src/cascade_methods/decoding_sweep_prepare.py \
        --judge_settings "$RUNGS" --verify_settings "$RUNGS" || exit 1
bash runners/run_judge_sweep_tight.sh > logs/ladcold_judge.log 2>&1
grep -aq "JUDGE_ALL_DONE" logs/ladcold_judge.log || { echo "JUDGE DID NOT COMPLETE"; exit 1; }
echo "=== judge done $(date) ==="

echo "=== phase 3: verifier [$RUNGS] $(date) ==="
python3 src/cascade_methods/decoding_sweep_prepare.py \
        --judge_settings "$RUNGS" --verify_settings "$RUNGS" || exit 1
N=$(python3 -c "import json;print(len(json.load(open('ckpts/openvqa/decoding_sweep/verifier_work.json'))))")
echo "  verifier work list: $N slots"
if [ "$N" -gt 0 ]; then
  bash runners/run_verify_sweep.sh 0 0 2 > logs/ladcold_verify_s0.log 2>&1 &
  V0=$!
  bash runners/run_verify_sweep.sh 1 1 2 > logs/ladcold_verify_s1.log 2>&1 &
  V1=$!
  wait $V0 $V1
fi
echo "=== verifier done $(date) ==="

echo "=== phase 4: analysis $(date) ==="
# NULL TEST 3 first: the SAME analysis code, pointed at the PREVIOUS round's pools and control, must
# reproduce the 2026-08-13 published deltas -- otherwise the new code, not the new rungs, moved things.
OMP_NUM_THREADS=1 python3 src/cascade_methods/decoding_ladder_cold.py --control T07 \
    --out results/cascade_methods/artifacts/_decoding_ladder_cold_nulltest3_prior_round.json \
    > logs/ladcold_nulltest3.log 2>&1
OMP_NUM_THREADS=1 python3 src/cascade_methods/decoding_ladder_cold.py > logs/ladcold_analysis.log 2>&1
tail -70 logs/ladcold_analysis.log
echo "LADDER_COLD_DONE $(date)"
