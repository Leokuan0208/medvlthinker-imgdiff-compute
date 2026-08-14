#!/bin/bash
# run_coadapt_T04_driver.sh -- the whole co-adaptation round, end to end, resumable at every stage.
#
# PRE-REGISTRATION: results/cascade_methods/artifacts/_coadapt_verifier_prereg_2026-08-14.json
#
#   stage 1  T=0.4 TRAIN candidate pools           runners/run_coadapt_T04_trainpools.sh
#   stage 2  explode + judge-preload + judge       runners/run_coadapt_T04_judge.sh
#   stage 3  train + score, 2 workers, 1 per GPU   runners/run_coadapt_T04_trainscore.sh
#   stage 4  the artifact is rewritten after EVERY completed seed, so a kill always leaves a
#            complete, honest design at whatever seed count was reached.
#
# SEQUENCING (the pre-registration's, verbatim): finish ALL FOUR ARMS at 3 seeds first, write the
# artifact, and only then deepen towards >= 10.  Worker seed lists are ordered so seeds 0,1,2 land
# first.
#
#   setsid nohup bash runners/run_coadapt_T04_driver.sh >/dev/null 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
M="$REPO/logs/coadapt_T04_master.log"
IDX="$REPO/data/disjoint_split"
POOLD="$REPO/ckpts/openvqa/cheap_lingshu7b_T04"
say(){ echo "$(date -u +%H:%M:%S) [driver] $*" >> "$M"; }
say "=== DRIVER START ==="

pools_complete(){
  for DS in slake_open_train vqa_rad_open_train pathvqa_open_train kvasir_open radimagenet_open; do
    W=$(python3 -c "import json;print(len(json.load(open('$IDX/idx_${DS}.json'))))")
    H=0; [ -f "$POOLD/ckpt_${DS}_lingshu7bT04_sc8.jsonl" ] && H=$(wc -l < "$POOLD/ckpt_${DS}_lingshu7bT04_sc8.jsonl")
    [ "$H" -ge "$W" ] || return 1
  done
  return 0
}

# ---- stage 1: wait for generation (launched separately) or run it ------------------------------
waited=0
while ! pools_complete; do
  if ! pgrep -f run_coadapt_T04_trainpools.sh >/dev/null; then
    say "generation not running and pools incomplete -> (re)launching stage 1"
    bash runners/run_coadapt_T04_trainpools.sh >> "$M" 2>&1
  fi
  sleep 60; waited=$((waited+60))
  if [ "$waited" -ge 28800 ]; then say "ABORT: pools still incomplete after 8h"; exit 1; fi
done
say "stage 1 complete: all five T04 train pools present"

# ---- stage 2: judge ----------------------------------------------------------------------------
if [ -f "$POOLD/.judge_done" ]; then
  say "SKIP stage 2 (judge already done)"
else
  bash runners/run_coadapt_T04_judge.sh >> "$M" 2>&1 && touch "$POOLD/.judge_done" \
    || { say "ABORT: judge stage failed"; exit 1; }
fi
say "stage 2 complete"

# ---- stage 3: train + score, one worker per GPU ------------------------------------------------
setsid nohup bash runners/run_coadapt_T04_trainscore.sh 0 0 2 4 6 8 >/dev/null 2>&1 &
setsid nohup bash runners/run_coadapt_T04_trainscore.sh 1 1 3 5 7 9 >/dev/null 2>&1 &
say "stage 3 workers launched (gpu0: seeds 0 2 4 6 8 | gpu1: seeds 1 3 5 7 9)"

# ---- stage 4: refresh the artifact whenever a new seed lands -----------------------------------
WANT=$(python3 -c "import json;print(len(json.load(open('$REPO/ckpts/openvqa/decoding_sweep/verifier_work_coadapt.json'))))")
n_done(){ python3 - "$WANT" <<'PY'
import glob, json, os, sys
want = int(sys.argv[1]); n = 0
for d in sorted(glob.glob(os.path.expanduser("~/medvlthinker-imgdiff-compute/ckpts/train/lora_verifier_T04_s*"))):
    tag = os.path.basename(d)
    if not os.path.exists(os.path.join(d, "adapter_model.safetensors")):
        continue
    s = set()
    for f in glob.glob(os.path.expanduser(
            f"~/medvlthinker-imgdiff-compute/ckpts/openvqa/decoding_sweep/vscore_{tag}_shard*.jsonl")):
        for l in open(f):
            if l.strip():
                try:
                    r = json.loads(l); s.add((r["ds"], r["idx"], r["ans"]))
                except Exception:
                    pass
    if len(s) >= want:
        n += 1
print(n)
PY
}
LAST=0; waited=0
while true; do
  N=$(n_done)
  if [ "$N" -gt "$LAST" ]; then
    say ">> $N complete seed(s); refreshing the artifact"
    python3 src/cascade_methods/coadapt_verifier.py >> "$REPO/logs/coadapt_T04_analyse.log" 2>&1
    say "artifact refresh rc=$? at $N seeds"
    LAST="$N"
  fi
  if ! pgrep -f run_coadapt_T04_trainscore.sh >/dev/null; then
    say "workers finished; final artifact at $LAST seed(s)"
    break
  fi
  sleep 120; waited=$((waited+120))
  if [ "$waited" -ge 172800 ]; then say "driver watchdog stop after 48h"; break; fi
done
say "=== DRIVER DONE ==="
echo "COADAPT_T04_DRIVER_DONE"
