#!/bin/bash
# Poll until BOTH generation workers of the 2026-08-14 cold ladder report done, assert every pool file
# is complete, then run the label -> score -> analyse pipeline. Polling (not `wait`) so this survives
# being started from a different shell than the generators.
cd /home/jamesyang/medvlthinker-imgdiff-compute
for i in $(seq 1 720); do
  if grep -aq "SWEEP_GPU0_ALL_DONE" logs/ladcold_gen_gpu0.log 2>/dev/null \
     && grep -aq "SWEEP_GPU1_ALL_DONE" logs/ladcold_gen_gpu1.log 2>/dev/null; then
    echo "generation complete $(date)"
    python3 - <<'PY'
import os
SW = "ckpts/openvqa/decoding_sweep"
EXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}
bad = []
for tag in ["T00", "T005", "T01", "T02", "T03r", "T04", "T05r", "T07r"]:
    for s in range(3):
        for ds, n in EXP.items():
            f = f"{SW}/ckpt_{ds}_{tag}_s{s}.jsonl"
            got = sum(1 for l in open(f) if l.strip()) if os.path.exists(f) else -1
            if got != n:
                bad.append((f, got, n))
print("INCOMPLETE:", bad if bad else "none -- all 72 pool files complete")
PY
    bash runners/finish_ladder_cold.sh
    exit 0
  fi
  sleep 60
done
echo "TIMED OUT waiting for generation"
