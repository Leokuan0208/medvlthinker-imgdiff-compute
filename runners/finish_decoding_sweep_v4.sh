#!/bin/bash
# Resume the 2026-08-14 decoding-sweep completion from PHASE 2 (judge), using the tight-memory judge
# runner. Generation is already finished: 18/42 runs complete = all 6 PRIORITY settings at 3 seeds
# (T03, T07 control, T13, minp01, rp105, rp11). Phase 1 is therefore skipped entirely.
#
# Why v3 exists: two orphaned VLLM::EngineCore processes from this session's own generation workers
# were still holding 26 GiB (GPU0) and 45.8 GiB (GPU1) after their parents were stopped, leaving
# 34 GiB free on GPU1 -- 3 GiB below the shared judge runner's 37 GiB floor. Terminating them was not
# available, so the judge is instead sized to what is actually free (see run_judge_sweep_tight.sh,
# which derives the 33 GiB floor from the model's real weight and KV footprint).
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

PRIORITY="T07,rp11,rp105,minp01,T03,T13,T05,T10"

echo "=== phase 2: judge [$PRIORITY] $(date) ==="
python3 src/cascade_methods/decoding_sweep_prepare.py --include_deployed \
        --judge_settings "$PRIORITY" --verify_settings "$PRIORITY" || exit 1
bash runners/run_judge_sweep_tight.sh > logs/decsweep4_judge.log 2>&1
grep -aq "JUDGE_ALL_DONE" logs/decsweep4_judge.log || { echo "JUDGE DID NOT COMPLETE"; exit 1; }
echo "=== judge done $(date) ==="
python3 src/cascade_methods/decoding_sweep_report.py > logs/decsweep4_report_cov.log 2>&1 || true

echo "=== phase 3: verifier [$PRIORITY] $(date) ==="
python3 src/cascade_methods/decoding_sweep_prepare.py --include_deployed \
        --judge_settings "$PRIORITY" --verify_settings "$PRIORITY" || exit 1
N=$(python3 -c "import json;print(len(json.load(open('ckpts/openvqa/decoding_sweep/verifier_work.json'))))")
echo "  verifier work list: $N slots"
if [ "$N" -gt 0 ]; then
  bash runners/run_verify_sweep.sh 0 0 2 > logs/decsweep4_verify_s0.log 2>&1 &
  V0=$!
  bash runners/run_verify_sweep.sh 1 1 2 > logs/decsweep4_verify_s1.log 2>&1 &
  V1=$!
  wait $V0 $V1
fi
echo "=== verifier done $(date) ==="

echo "=== phase 4: final report $(date) ==="
python3 src/cascade_methods/decoding_sweep_report.py          > logs/decsweep4_report.log   2>&1
python3 src/cascade_methods/decoding_sweep_dual_currency.py   > logs/decsweep4_dual.log     2>&1
python3 src/cascade_methods/decoding_sweep_currency_audit.py  > logs/decsweep4_currency.log 2>&1
python3 src/cascade_methods/decoding_sweep_paraphrase_test.py > logs/decsweep4_para.log     2>&1
python3 src/cascade_methods/decoding_sweep_ceiling.py         > logs/decsweep4_ceiling.log  2>&1
python3 src/cascade_methods/decoding_sweep_prereg_outcomes.py > logs/decsweep4_prereg.log   2>&1
python3 src/cascade_methods/decoding_sweep_finalize.py        > logs/decsweep4_finalize.log 2>&1
echo "DECODING_SWEEP_V4_DONE $(date)"
