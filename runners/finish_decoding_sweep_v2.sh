#!/bin/bash
# Chained completion of the 2026-08-13 decoding sweep, 2026-08-14 session.
#
# The 2026-08-13 run left the grid at 9/42 complete runs (5 settings, 1-2 seeds each) because two
# sibling jobs held both A100s. Both GPUs are now free, so this finishes the grid:
#   phase 1  generation of the 33 missing runs         (already launched separately, we wait on it)
#   phase 2  judge  -- coverage tier for EVERY setting (tp=2, both GPUs)
#   phase 3  verifier -- selection tier, in the PRE-REGISTERED order, one group at a time so that
#            an early stop still leaves whole, usable groups rather than a half-scored setting
#   phase 4  report + dual-currency + currency audit
#
# Every stage is content-addressed and resumable; a re-run costs only what is missing.
# Nothing is ever killed: the judge/verify runners wait for GPU room.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

# ---------------- phase 1: wait for generation ----------------
echo "=== phase 1: waiting for generation workers $(date) ==="
for i in $(seq 1 3000); do
  # NB: `grep -c` exits 1 on zero matches, so `|| echo 0` would emit "0\n0" and break `[ -ge ]`.
  if grep -aq "SWEEP_GPU0_ALL_DONE" logs/decsweep2_gen_gpu0.log 2>/dev/null \
  && grep -aq "SWEEP_GPU1_ALL_DONE" logs/decsweep2_gen_gpu1.log 2>/dev/null; then break; fi
  # also stop waiting if no generation process is alive any more
  if ! pgrep -f "decoding_sweep_gen.py" > /dev/null 2>&1; then
    if ! pgrep -f "run_sweep_adaptive.sh" > /dev/null 2>&1; then
      echo "generation workers exited $(date)"; break
    fi
  fi
  sleep 30
done
echo "=== generation phase over $(date) ==="
python3 - <<'PY'
import json, os
DS={'slake_open':645,'vqa_rad_open':200,'pathvqa_open':1500}
S=json.load(open('results/cascade_methods/artifacts/_decoding_sweep_settings.json'))
SW='ckpts/openvqa/decoding_sweep'; ok=[]
for s in S:
    t=s['tag']
    good=all((sum(1 for l in open(f'{SW}/ckpt_{d}_{t}.jsonl') if l.strip()) if os.path.exists(f'{SW}/ckpt_{d}_{t}.jsonl') else 0)==n for d,n in DS.items())
    if good: ok.append(t)
print(f"COMPLETE RUNS: {len(ok)}/{len(S)} -> {ok}")
PY

# ---------------- phase 2: judge every generated pool ----------------
echo "=== phase 2: judge $(date) ==="
python3 src/cascade_methods/decoding_sweep_prepare.py --include_deployed || exit 1
bash runners/run_judge_sweep.sh > logs/decsweep2_judge.log 2>&1
grep -aq "JUDGE_ALL_DONE" logs/decsweep2_judge.log || { echo "JUDGE DID NOT COMPLETE"; exit 1; }
echo "=== judge done $(date) ==="

# coverage-tier report is already meaningful now
python3 src/cascade_methods/decoding_sweep_report.py > logs/decsweep2_report_coverage.log 2>&1 || true

# ---------------- phase 3: verifier, pre-registered groups ----------------
# Group 1 = the five settings already reported on 2026-08-13 PLUS rp105, the mechanistic ladder for
#           the repetition-penalty axis (rp11 was pre-registered as a HARM check and is the only
#           setting that won under the judge; rp105 completes that axis and is not an outcome pick).
# Groups 2/3 follow the pre-registered fallback order.
G1="T07,rp11,rp105,minp01,T03,T13"
G2="T10,T05,minp005"
G3="topk20,topk50,topp09,topp095,T13minp010"
for GRP in "$G1" "$G2" "$G3"; do
  echo "=== phase 3: verifier group [$GRP] $(date) ==="
  python3 src/cascade_methods/decoding_sweep_prepare.py --include_deployed --verify_settings "$GRP" || exit 1
  N=$(python3 -c "import json;print(len(json.load(open('ckpts/openvqa/decoding_sweep/verifier_work.json'))))")
  echo "  work list: $N slots"
  if [ "$N" -gt 0 ]; then
    bash runners/run_verify_sweep.sh 0 0 2 > logs/decsweep2_verify_s0.log 2>&1 &
    V0=$!
    bash runners/run_verify_sweep.sh 1 1 2 > logs/decsweep2_verify_s1.log 2>&1 &
    V1=$!
    wait $V0 $V1
  fi
  echo "=== verifier group done $(date) ==="
  python3 src/cascade_methods/decoding_sweep_report.py > "logs/decsweep2_report.log" 2>&1 || true
  python3 src/cascade_methods/decoding_sweep_dual_currency.py > "logs/decsweep2_dual.log" 2>&1 || true
done

# ---------------- phase 4: final analyses ----------------
echo "=== phase 4: final report $(date) ==="
python3 src/cascade_methods/decoding_sweep_report.py > logs/decsweep2_report.log 2>&1
python3 src/cascade_methods/decoding_sweep_dual_currency.py > logs/decsweep2_dual.log 2>&1
python3 src/cascade_methods/decoding_sweep_currency_audit.py > logs/decsweep2_currency.log 2>&1
echo "DECODING_SWEEP_V2_DONE $(date)"
