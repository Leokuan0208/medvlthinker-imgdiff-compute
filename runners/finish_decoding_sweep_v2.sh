#!/bin/bash
# Chained completion of the 2026-08-13 decoding sweep, 2026-08-14 session.
#
# The 2026-08-13 run left the grid at 9/42 complete runs (5 settings, 1-2 seeds each) because two
# sibling jobs held both A100s. Both GPUs freed up, so this finishes it.
#
# SCOPE DECISION (recorded because it shapes what the artifact can claim):
#   generation is cheap and, crucially, records `oks_em` per slot -- so EXACT-MATCH coverage endpoints
#   (oracle@8, distinct count, token audit, EM capture-recapture) cost NO GPU and are reported for
#   EVERY setting in the 14-setting grid. The JUDGE is the expensive stage (~95k new strings for the
#   full grid, ~5 h) and the VERIFIER after it, so those two are scoped to a PRIORITY set:
#       T07 (control), rp11, rp105, minp01, T03, T13
#   chosen as: the 5 settings already reported on 2026-08-13, all lifted to 3 seeds, PLUS rp105, which
#   completes the repetition-penalty ladder -- rp11 was pre-registered as a HARM check and turned out
#   to be the only setting that won under the judge, so the ladder is a mechanism control, not an
#   outcome-chasing pick. Depth (3 seeds, both currencies) beats breadth here because the endpoint is
#   SELECTED accuracy and seed sd is ~0.004-0.008, the same size as the effects being argued about.
#
# Every stage is content-addressed and resumable. Nothing is ever killed; the runners wait for GPU room.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

PRIORITY="T07,rp11,rp105,minp01,T03,T13"

# ---------------- phase 1: wait for generation ----------------
echo "=== phase 1: waiting for generation workers $(date) ==="
for i in $(seq 1 3000); do
  # NB: `grep -c` exits 1 on zero matches, so `|| echo 0` would emit "0\n0" and break `[ -ge ]`.
  if grep -aq "SWEEP_GPU0_ALL_DONE" logs/decsweep2_gen_gpu0.log 2>/dev/null \
  && grep -aq "SWEEP_GPU1_ALL_DONE" logs/decsweep2_gen_gpu1.log 2>/dev/null; then break; fi
  if ! pgrep -f "decoding_sweep_gen.py" > /dev/null 2>&1 \
  && ! pgrep -f "run_sweep_adaptive.sh" > /dev/null 2>&1; then
    echo "generation workers exited $(date)"; break
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

# EM-currency breadth needs no GPU -- emit it now, for every setting that generated.
python3 src/cascade_methods/decoding_sweep_report.py > logs/decsweep2_report_em.log 2>&1 || true
echo "=== EM-currency breadth written $(date) ==="

# ---------------- phase 2: judge, PRIORITY settings only ----------------
echo "=== phase 2: judge [$PRIORITY] $(date) ==="
python3 src/cascade_methods/decoding_sweep_prepare.py --include_deployed \
        --judge_settings "$PRIORITY" --verify_settings "$PRIORITY" || exit 1
bash runners/run_judge_sweep.sh > logs/decsweep2_judge.log 2>&1
grep -aq "JUDGE_ALL_DONE" logs/decsweep2_judge.log || { echo "JUDGE DID NOT COMPLETE"; exit 1; }
echo "=== judge done $(date) ==="
python3 src/cascade_methods/decoding_sweep_report.py > logs/decsweep2_report_cov.log 2>&1 || true

# ---------------- phase 3: verifier, PRIORITY settings ----------------
echo "=== phase 3: verifier [$PRIORITY] $(date) ==="
python3 src/cascade_methods/decoding_sweep_prepare.py --include_deployed \
        --judge_settings "$PRIORITY" --verify_settings "$PRIORITY" || exit 1
N=$(python3 -c "import json;print(len(json.load(open('ckpts/openvqa/decoding_sweep/verifier_work.json'))))")
echo "  verifier work list: $N slots"
if [ "$N" -gt 0 ]; then
  bash runners/run_verify_sweep.sh 0 0 2 > logs/decsweep2_verify_s0.log 2>&1 &
  V0=$!
  bash runners/run_verify_sweep.sh 1 1 2 > logs/decsweep2_verify_s1.log 2>&1 &
  V1=$!
  wait $V0 $V1
fi
echo "=== verifier done $(date) ==="

# ---------------- phase 4: final analyses ----------------
echo "=== phase 4: final report $(date) ==="
python3 src/cascade_methods/decoding_sweep_report.py        > logs/decsweep2_report.log   2>&1
python3 src/cascade_methods/decoding_sweep_dual_currency.py > logs/decsweep2_dual.log     2>&1
python3 src/cascade_methods/decoding_sweep_currency_audit.py> logs/decsweep2_currency.log 2>&1
python3 src/cascade_methods/decoding_sweep_paraphrase_test.py > logs/decsweep2_para.log   2>&1
python3 src/cascade_methods/decoding_sweep_ceiling.py       > logs/decsweep2_ceiling.log  2>&1
python3 src/cascade_methods/decoding_sweep_prereg_outcomes.py > logs/decsweep2_prereg.log 2>&1
python3 src/cascade_methods/decoding_sweep_finalize.py      > logs/decsweep2_finalize.log 2>&1
echo "DECODING_SWEEP_V2_DONE $(date)"
