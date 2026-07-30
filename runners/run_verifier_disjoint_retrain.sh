#!/bin/bash
# Retrain the open-text answer verifier on a STRICTLY DISJOINT split, then re-measure the open-text arm.
#
# WHY: the deployed verifier (ckpts/train/lora_verifier_pooled4) was trained on 67-73% of the items it
# is evaluated on (results/cascade_methods/artifacts/verifier_validity_2026-07-29.json), so the open-text
# arm's accuracy gain is inflated ~1.8x. This produces an UNCONTAMINATED number.
#
# STAGES (each skip-if-done, each under `timeout -s KILL`, never loops forever):
#   1  generate  Lingshu-7B sc8 candidates for the OFFICIAL TRAIN splits of SLAKE / VQA-RAD / PathVQA,
#                restricted to the image-disjoint allowlists from build_disjoint_verifier_split.py.
#                Same generator + flags as the eval-set candidates (run_openvqa.py --n_samples 8
#                --temp 0.7 --cap cap320 --max_model_len 4096), so the training candidate distribution
#                matches the eval one.
#   2  explode   one judge row per unique (question, answer) pair
#   3  judge     the SAME judge as the headline: MedVLThinker-32B via src/labeling/run_judge.py (tp=2)
#   4  train     two clean verifiers, composition-matched to the contaminated one (10364 examples,
#                894 SLAKE / 4973 PathVQA / 522 VQA-RAD / 3975 Kvasir):
#                  L1  image-disjoint (no eval image, no eval item)
#                  L2  L1 + no eval question TEXT at all (conservative bound)
#   5  score     re-score the SAME candidate pools with each clean adapter (verifier_transfer_eval.py);
#                candidates and judge labels are verifier-independent, so nothing is regenerated
#   6  measure   clean vs contaminated table with paired-bootstrap 95% CIs -> the artifact
#
# GPU ETIQUETTE: waits until BOTH GPUs are free (another job may be running); training and scoring use
# tp=1 on GPU0. Launch detached:
#   setsid nohup bash runners/run_verifier_disjoint_retrain.sh >/dev/null 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
CKD="$REPO/ckpts/openvqa/cheap_lingshu7b"
IDX="$REPO/data/disjoint_split"
M="$REPO/logs/verif_disjoint_master.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
say "=== DISJOINT VERIFIER RETRAIN START ==="

# ---- wait for BOTH GPUs to be free (cap 8h, never loop forever) -------------------------------
waited=0
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  [ "${used:-99999}" -lt 2000 ] && break
  if [ "$waited" -ge 28800 ]; then say "ABORT: GPUs still busy after 8h (${used} MiB)"; exit 1; fi
  sleep 60; waited=$((waited+60))
done
say "GPUs free (${used} MiB) after ${waited}s"

# ---- 1. generate sc8 candidates for the disjoint TRAIN pools ----------------------------------
for DS in slake_open_train vqa_rad_open_train pathvqa_open_train; do
  OUT="$CKD/ckpt_${DS}_lingshu7b_sc8.jsonl"
  WANT=$(python3 -c "import json;print(len(json.load(open('$IDX/idx_${DS}.json'))))")
  HAVE=0; [ -f "$OUT" ] && HAVE=$(wc -l < "$OUT")
  if [ "$HAVE" -ge "$WANT" ]; then say "SKIP(gen done $HAVE/$WANT) $DS"; continue; fi
  say ">> GEN $DS ($HAVE/$WANT done)"
  timeout -s KILL 5400 env CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py \
    --model_path "$L7" --tag lingshu7b_sc8 --dataset "$DS" --n_samples 8 --temp 0.7 \
    --ckpt_dir ckpts/openvqa/cheap_lingshu7b --tp 1 --max_model_len 4096 \
    --idx_file "$IDX/idx_${DS}.json" >> "$REPO/logs/verif_disjoint_gen.log" 2>&1
  HAVE=0; [ -f "$OUT" ] && HAVE=$(wc -l < "$OUT")
  say "GEN $DS rc=$? -> $HAVE/$WANT rows"
  [ "$HAVE" -ge "$WANT" ] || { say "ABORT: generation incomplete for $DS"; exit 1; }
done

# ---- 2. explode for the judge -----------------------------------------------------------------
for DS in slake_open_train vqa_rad_open_train pathvqa_open_train; do
  SRC="$CKD/ckpt_${DS}_lingshu7b_sc8.jsonl"; EXP="${SRC%.jsonl}_scexploded.jsonl"
  if [ -s "$EXP" ]; then say "SKIP(explode done) $DS"; continue; fi
  python3 src/cascade_methods/explode_sc_for_judge.py "$SRC" >> "$M" 2>&1
done

# ---- 3. judge (same judge as the headline: MedVLThinker-32B, tp=2) ----------------------------
NEED=""
for DS in slake_open_train vqa_rad_open_train pathvqa_open_train; do
  EXP="$CKD/ckpt_${DS}_lingshu7b_sc8_scexploded.jsonl"; JUD="${EXP%.jsonl}.judge.jsonl"
  E=$(wc -l < "$EXP"); Jn=0; [ -f "$JUD" ] && Jn=$(wc -l < "$JUD")
  if [ "$Jn" -lt "$E" ]; then NEED="$NEED $EXP"; else say "SKIP(judge done $Jn/$E) $DS"; fi
done
if [ -n "$NEED" ]; then
  say ">> JUDGE$NEED"
  timeout -s KILL 7200 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 \
    --preds $NEED >> "$REPO/logs/verif_disjoint_judge.log" 2>&1
  say "JUDGE rc=$?"
  pkill -9 -f VLLM::EngineCore 2>/dev/null; sleep 30
fi
for DS in slake_open_train vqa_rad_open_train pathvqa_open_train; do
  EXP="$CKD/ckpt_${DS}_lingshu7b_sc8_scexploded.jsonl"; JUD="${EXP%.jsonl}.judge.jsonl"
  E=$(wc -l < "$EXP"); Jn=0; [ -f "$JUD" ] && Jn=$(wc -l < "$JUD")
  [ "$Jn" -ge "$E" ] || { say "ABORT: judge incomplete for $DS ($Jn/$E)"; exit 1; }
done

# ---- 4/5. train + score, for both disjointness levels -----------------------------------------
#   The two levels are INDEPENDENT, so each gets its own GPU and they run concurrently: tp=1 per job,
#   one 7B (+LoRA) per device, no contention. Halves wall time vs running them back to back.
run_level(){
  local LVL="$1" DIR="$2" GPU="$3"
  if [ -s "$REPO/$DIR/adapter_model.safetensors" ]; then say "SKIP(train done) $LVL"; else
    say ">> TRAIN $LVL -> $DIR (gpu $GPU)"
    # --deadline_s < the timeout so the adapter is always SAVED rather than killed mid-epoch
    timeout -s KILL 21600 env CUDA_VISIBLE_DEVICES="$GPU" python3 \
      src/training_methods/run_lora_verifier_disjoint.py --level "$LVL" --out_dir "$DIR" \
      --deadline_s 18000 >> "$REPO/logs/verif_disjoint_train_${LVL}.log" 2>&1
    say "TRAIN $LVL rc=$?"
    [ -s "$REPO/$DIR/adapter_model.safetensors" ] || { say "ABORT: no adapter for $LVL"; return 1; }
  fi
  for DS in slake_open vqa_rad_open pathvqa_open; do
    if [ -s "$REPO/$DIR/transfer_dump_${DS}_lingshu7b.json" ]; then say "SKIP(score done) $LVL/$DS"; continue; fi
    say ">> SCORE $LVL/$DS (gpu $GPU)"
    timeout -s KILL 10800 env CUDA_VISIBLE_DEVICES="$GPU" python3 \
      src/training_methods/verifier_transfer_eval.py --adapter "$DIR" --datasets "$DS" \
      >> "$REPO/logs/verif_disjoint_score_${LVL}.log" 2>&1
    say "SCORE $LVL/$DS rc=$?"
  done
}
run_level L1 ckpts/train/lora_verifier_disjoint    0 || say "L1 FAILED" &
P_L1=$!
run_level L2 ckpts/train/lora_verifier_disjoint_l2 1 || say "L2 FAILED" &
P_L2=$!
wait $P_L1; wait $P_L2
say "both levels finished"

# ---- 6. measure --------------------------------------------------------------------------------
say ">> MEASURE"
python3 src/training_methods/verifier_disjoint_measure.py >> "$REPO/logs/verif_disjoint_measure.log" 2>&1
say "MEASURE rc=$?"
say "=== DISJOINT VERIFIER RETRAIN DONE ==="
