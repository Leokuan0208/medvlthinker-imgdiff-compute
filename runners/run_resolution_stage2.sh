#!/bin/bash
# SWEEP 2 -- stage 2 driver: wait until the two decisive caps (the deployed cap320 control and
# `native` = the qwen_vl_utils default the MCQ arms already run at) have all four arms on disk,
# then stop generation and run the labelling stage, so the endpoint table exists even if the
# lower rungs of the ladder never get a card on this contended box.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
L=logs/resolution_stage2_2026-08-13.log
GEN=logs/resolution_open_gen_2026-08-13.log
say(){ echo "[$(date +%F\ %T)] $*" >> "$L"; }
say "STAGE2 WAITING for cap320 + native blocks"
DEADLINE=$(( $(date +%s) + ${MAXWAIT:-9000} ))
# Completeness is checked on the FILES, not on log lines: a resumed run re-emits a "DONE ... in
# 0.0 min" line for every arm it skips, so counting log lines double-counts and fired this stage
# early once. 2 caps x 4 arms x 3 cells = 24 files, each at its cell's full row count.
count_ready(){
  python3 - <<'PY'
import os
SW = os.path.expanduser("~/medvlthinker-imgdiff-compute/ckpts/openvqa/resolution_sweep")
NEXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}
n = 0
CAPTAGS = os.environ.get("READY_ARMS", "cap320:t0,s0,s1,s2 native:t0,s0,s1")
for spec in CAPTAGS.split():
    cap, tags = spec.split(":")
    for tag in tags.split(","):
        for ds, k in NEXP.items():
            p = os.path.join(SW, f"ckpt_{ds}_{cap}_{tag}.jsonl")
            if os.path.exists(p) and sum(1 for l in open(p) if l.strip()) >= k:
                n += 1
print(n)
PY
}
while :; do
  n=$(count_ready)
  [ "$n" -ge "${READY_N:-21}" ] && { say "both blocks complete ($n/${READY_N:-21} complete arm files)"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { say "deadline hit with $n/${READY_N:-21} complete arm files -- proceeding"; break; }
  sleep 60
done
say "stopping generation to free the card"
pkill -f "run_resolution_open_gen.sh" 2>/dev/null
sleep 5
pkill -f "resolution_open_generate.py" 2>/dev/null
sleep 25
say "launching label stage"
NULLTEST=${NULLTEST:-200} VBATCH=${VBATCH:-8} VERIF_MIB=${VERIF_MIB:-30000} \
  JUDGE_MIB=${JUDGE_MIB:-72000} bash runners/run_resolution_label.sh >> "$L" 2>&1
say "label stage rc=$?"
say "STAGE2 DONE"
