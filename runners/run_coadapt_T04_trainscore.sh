#!/bin/bash
# run_coadapt_T04_trainscore.sh -- train the CO-ADAPTED verifiers on the T=0.4 pools and score
# arms C and D with each of them.  $1 = gpu id, $2.. = the seeds this worker owns.
#
# PRE-REGISTRATION: results/cascade_methods/artifacts/_coadapt_verifier_prereg_2026-08-14.json
#
# TRAINING DISCIPLINE -- THE POOL TEMPERATURE IS THE ONLY VARIABLE.  The trainer is
# src/training_methods/run_lora_verifier_disjoint.py UNCHANGED, at its incumbent defaults
# (lora_r 16 / alpha 32 / dropout 0.05, q,k,v,o,gate,up,down _proj, Lingshu-7B, max_pixels 1003520,
# lr 1e-4, bs 2 x accum 8, 1 epoch, cap_div 1, level L1, composition matching ON, max_train 10364).
# It is pointed at the T=0.4 pools purely through its own VERIF_CK / VERIF_TAG environment hooks, so
# not one line of the incumbent recipe is edited.
#
# ⚠️ ckpts/train/lora_verifier_disjoint -- the artifact of record for every published number -- is
# NEVER written to.  Each seed gets its own ckpts/train/lora_verifier_T04_s<seed>.
#
# Arms C and D SHARE ONE adapter per seed: one scoring pass covers the union of the T04 and T07r
# candidate strings (ckpts/openvqa/decoding_sweep/verifier_work_coadapt.json, 21,104 pairs).
#
# HF transformers only for adapter scoring (vLLM 0.9.0.1 silently drops all 192 visual.* modules).
#
#   setsid nohup bash runners/run_coadapt_T04_trainscore.sh 0 0 2 4 6 8 >/dev/null 2>&1 &
#   setsid nohup bash runners/run_coadapt_T04_trainscore.sh 1 1 3 5 7 9 >/dev/null 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
GPU="$1"; shift
SEEDS="$*"
POOLD="$REPO/ckpts/openvqa/cheap_lingshu7b_T04"
SWEEP="$REPO/ckpts/openvqa/decoding_sweep"
WORK="$SWEEP/verifier_work_coadapt.json"
M="$REPO/logs/coadapt_T04_master.log"
say(){ echo "$(date -u +%H:%M:%S) [gpu$GPU] $*" >> "$M"; }
say "=== TRAIN/SCORE WORKER START seeds=$SEEDS ==="

wait_gpu(){   # $1 = GiB needed; sustained-free, never kills a co-tenant
  local need="$1" streak=0 waited=0 free
  while true; do
    free=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU") / 1024 ))
    if [ "$free" -ge "$need" ]; then streak=$((streak+1)); [ "$streak" -ge 3 ] && return 0
    else streak=0; fi
    if [ "$waited" -ge 43200 ]; then say "ABORT: gpu never freed ${need}GiB in 12h"; return 1; fi
    sleep 30; waited=$((waited+30))
  done
}

n_work(){ python3 -c "import json;print(len(json.load(open('$WORK'))))"; }
n_scored(){ python3 - "$1" <<'PY'
import glob, json, sys
d = set()
for f in glob.glob(sys.argv[1] + "_shard*.jsonl"):
    for l in open(f):
        if l.strip():
            try:
                r = json.loads(l); d.add((r["ds"], r["idx"], r["ans"]))
            except Exception:
                pass
print(len(d))
PY
}

WANT=$(n_work)
for S in $SEEDS; do
  DIR="ckpts/train/lora_verifier_T04_s${S}"
  ADAPTER="$REPO/$DIR/adapter_model.safetensors"

  # ---- train -----------------------------------------------------------------------------------
  if [ -s "$ADAPTER" ]; then
    say "SKIP(train done) seed $S"
  else
    for attempt in 1 2 3; do
      wait_gpu 60 || exit 1
      say ">> TRAIN seed $S -> $DIR (attempt $attempt)"
      timeout -s KILL 25200 env CUDA_VISIBLE_DEVICES="$GPU" \
        VERIF_CK="ckpts/openvqa/cheap_lingshu7b_T04" VERIF_TAG="lingshu7bT04" \
        python3 src/training_methods/run_lora_verifier_disjoint.py \
          --level L1 --out_dir "$DIR" --seed "$S" --deadline_s 21600 \
          >> "$REPO/logs/coadapt_T04_train_s${S}.log" 2>&1
      say "TRAIN seed $S rc=$?"
      [ -s "$ADAPTER" ] && break
      sleep 30
    done
    [ -s "$ADAPTER" ] || { say "ABORT: no adapter for seed $S"; exit 1; }
    # record the one variable that was changed, so the artifact can never lose its provenance
    python3 - "$REPO/$DIR/train_config.json" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["pool_temperature"] = 0.4
c["pool_dir"] = "ckpts/openvqa/cheap_lingshu7b_T04"
c["pool_tag"] = "lingshu7bT04"
c["incumbent_pool_temperature"] = 0.7
c["incumbent_adapter"] = "ckpts/train/lora_verifier_disjoint"
json.dump(c, open(p, "w"), indent=1)
PY
  fi

  # ---- score arms C and D (one pass, shared adapter) ---------------------------------------------
  CACHE="ckpts/openvqa/decoding_sweep/vscore_lora_verifier_T04_s${S}"
  HAVE=$(n_scored "$REPO/$CACHE")
  if [ "$HAVE" -ge "$WANT" ]; then say "SKIP(score done $HAVE/$WANT) seed $S"; continue; fi
  for attempt in $(seq 1 20); do
    wait_gpu 20 || exit 1
    say ">> SCORE seed $S ($HAVE/$WANT, attempt $attempt)"
    timeout -s KILL 21600 env CUDA_VISIBLE_DEVICES="$GPU" python3 \
      src/cascade_methods/decoding_sweep_verify.py --work "$WORK" --cache "$CACHE" \
      --adapter "$DIR" --shard 0 --nshard 1 \
      >> "$REPO/logs/coadapt_T04_score_s${S}.log" 2>&1
    rc=$?
    HAVE=$(n_scored "$REPO/$CACHE")
    say "SCORE seed $S rc=$rc -> $HAVE/$WANT"
    [ "$HAVE" -ge "$WANT" ] && break
    sleep 30
  done
  [ "$HAVE" -ge "$WANT" ] || { say "ABORT: scoring incomplete for seed $S ($HAVE/$WANT)"; exit 1; }
  say "SEED $S COMPLETE (train + score)"
done
say "=== TRAIN/SCORE WORKER DONE seeds=$SEEDS ==="
echo "COADAPT_T04_TRAINSCORE_DONE_gpu$GPU"
