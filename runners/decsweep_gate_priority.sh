#!/bin/bash
# Gate: wait until the 9 PRIORITY generation runs are complete, then stop generation so the
# already-running completion pipeline can advance to the judge (which needs both GPUs).
#
# Stopping generation is safe and loses nothing: decoding_sweep_gen.py writes per-item JSONL and
# resumes from the last completed line, and every partial pool is refused by the analysis loaders
# (load_pool is strict about row counts), so a half-written run can never reach a reported number.
# The remaining breadth runs can be resumed later by relaunching runners/run_sweep_adaptive.sh.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
PRI="T07_s2 rp11_s2 rp105_s0 rp105_s1 rp105_s2 minp01_s1 minp01_s2 T03_s2 T13_s2"

for i in $(seq 1 2000); do
  if python3 - "$PRI" <<'PY'
import json, os, sys
DS={'slake_open':645,'vqa_rad_open':200,'pathvqa_open':1500}
SW='ckpts/openvqa/decoding_sweep'
def ok(t):
    for d,n in DS.items():
        f=f'{SW}/ckpt_{d}_{t}.jsonl'
        c=sum(1 for l in open(f) if l.strip()) if os.path.exists(f) else 0
        if c<n: return False
    return True
sys.exit(0 if all(ok(t) for t in sys.argv[1].split()) else 1)
PY
  then
    echo "PRIORITY_RUNS_COMPLETE $(date)"
    break
  fi
  # if generation has already died out on its own, stop waiting
  if ! pgrep -f "decoding_sweep_gen.py" >/dev/null 2>&1 \
  && ! pgrep -f "run_sweep_adaptive.sh" >/dev/null 2>&1; then
    echo "GENERATION_ENDED_EARLY $(date)"; break
  fi
  sleep 45
done

echo "stopping generation workers so the judge can have both GPUs $(date)"
pkill -f run_sweep_adaptive.sh   >/dev/null 2>&1
sleep 2
pkill -f decoding_sweep_gen.py   >/dev/null 2>&1
sleep 5
echo "gen procs remaining: $(pgrep -f decoding_sweep_gen.py | wc -l)"
python3 - <<'PY'
import json, os
DS={'slake_open':645,'vqa_rad_open':200,'pathvqa_open':1500}
S=json.load(open('results/cascade_methods/artifacts/_decoding_sweep_settings.json'))
SW='ckpts/openvqa/decoding_sweep'; ok=[]
for s in S:
    t=s['tag']
    if all((sum(1 for l in open(f'{SW}/ckpt_{d}_{t}.jsonl') if l.strip()) if os.path.exists(f'{SW}/ckpt_{d}_{t}.jsonl') else 0)==n for d,n in DS.items()):
        ok.append(t)
print(f"COMPLETE RUNS AT GATE: {len(ok)}/{len(S)}")
print(sorted(ok))
PY
echo "GATE_DONE $(date)"
