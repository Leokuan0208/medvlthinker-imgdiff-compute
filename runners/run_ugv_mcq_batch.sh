#!/usr/bin/env bash
# UGV MCQ-as-generation: re-run the datasets fixed in run_mcq_generate_verify.py.
# tp=1, GPU0 only, ONE dataset at a time (memory cgroup ~244GB -> small --chunk). Never run two concurrently.
set -u
cd "$(dirname "$0")/.."   # repo root
export HF_HOME=/data/dan/hf_cache
export CUDA_VISIBLE_DEVICES=0
LING="lingshu-medical-mllm/Lingshu-7B"
MVT="/data/dan/weights/MedVLThinker-7B-RL_m23k"
LORA="ckpts/train/lora_verifier_pooled4"
LOGDIR="logs/ugv_mcq"; mkdir -p "$LOGDIR"
COMMON="--verifier_lora $LORA --n_samples 8 --temp 0.7 --cap cap320 --tp 1 --chunk 16"

run () {  # args: model dataset mode tag ckpt_dir nmax
  local model="$1" ds="$2" mode="$3" tag="$4" dir="$5" nmax="$6"
  local log="$LOGDIR/${tag}_${ds}_${mode}.log"
  echo "=== $(date '+%F %T') START $tag $ds $mode (n=$nmax) -> $log ==="
  python3 src/labeling/run_mcq_generate_verify.py --model_path "$model" \
    --dataset "$ds" --mode "$mode" --tag "$tag" --ckpt_dir "$dir" --n "$nmax" \
    $COMMON > "$log" 2>&1 \
    && echo "=== $(date '+%F %T') DONE  $tag $ds $mode ===" \
    || echo "=== $(date '+%F %T') FAIL  $tag $ds $mode (see $log) ==="
}

# --- Lingshu-7B CONTENT (options hidden -> generate answer) ---
run "$LING" PMC_VQA       content lingshu7b ckpts/mcq_gen_verify/lingshu7b 2000
run "$LING" MedXpertQA-MM content lingshu7b ckpts/mcq_gen_verify/lingshu7b 2000
run "$LING" PATH_VQA      content lingshu7b ckpts/mcq_gen_verify/lingshu7b 2000
# --- Lingshu-7B LETTER (options shown -> answer with letter) ---
run "$LING" PMC_VQA       letter  lingshu7b ckpts/mcq_gen_verify/lingshu7b 2000
run "$LING" MedXpertQA-MM letter  lingshu7b ckpts/mcq_gen_verify/lingshu7b 2000
# --- MedVLThinker-7B CONTENT (cross-family) ---
run "$MVT"  PMC_VQA       content mvt7b     ckpts/mcq_gen_verify/mvt7b     2000
run "$MVT"  PATH_VQA      content mvt7b     ckpts/mcq_gen_verify/mvt7b     2000

echo "=== $(date '+%F %T') ALL UGV MCQ RUNS COMPLETE ==="
