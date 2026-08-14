#!/bin/bash
# Self-completing tail of the 2026-08-13 decoding sweep.
# The 32B judge could not be scheduled inside the session's GPU window (two sibling jobs held both
# A100s). It is left running and polite; when it finishes, this script scores the frozen verifier on
# the pre-registered selection tier and regenerates the artifact. Everything is resumable, so a
# re-run costs only what is missing.
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

echo "=== waiting for the judge to finish (started separately, resumable) ==="
for i in $(seq 1 2000); do
  grep -aq "JUDGE_ALL_DONE" logs/decsweep_judge.log 2>/dev/null && break
  sleep 30
done
grep -aq "JUDGE_ALL_DONE" logs/decsweep_judge.log 2>/dev/null || { echo "judge never completed; stopping"; exit 1; }
echo "=== judge done $(date) ==="

# Regenerate the work lists now that new judge labels exist.
python3 src/cascade_methods/decoding_sweep_prepare.py --include_deployed \
        --verify_settings T07,rp11,minp01,T13,T03 || exit 1

# Frozen verifier, HF only, two shards.
bash /tmp/run_verify_sweep.sh 0 0 2 > logs/decsweep_verify_s0.log 2>&1 &
V0=$!
bash /tmp/run_verify_sweep.sh 1 1 2 > logs/decsweep_verify_s1.log 2>&1 &
V1=$!
wait $V0 $V1
echo "=== verifier done $(date) ==="

python3 src/cascade_methods/decoding_sweep_report.py
echo "DECODING_SWEEP_PIPELINE_DONE $(date)"
