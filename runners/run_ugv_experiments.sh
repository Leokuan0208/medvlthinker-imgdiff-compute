#!/bin/bash
# UGV (Unified Generative Verifier) GPU experiments — the decisive test for the unified method.
# Waits for the OmniMed pipeline to finish, then samples N generations per MCQ item (cheap model) and scores
# each with the EXISTING trained verifier, in options-hidden (content) and options-shown (letter) modes. This
# is what unblocks "does the unified verifier work on MCQ-as-generation?" — if yes, one method covers the
# whole suite. tp=1 (cheap model), no NCCL. After generation, re-runs the offline unified analysis.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/ugv_experiments.log
STRONG=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_strong.log
VER=ckpts/train/lora_verifier_pooled4
PY=/data/dan/medeval_venv/bin/python
echo "UGV experiments queued $(date)" > "$L"

# wait for the whole OmniMed pipeline (cheap + strong chunked) to finish so the GPUs are free
until grep -q "OMNIMED_STRONG_DONE" "$STRONG" 2>/dev/null; do sleep 300; done
echo "OmniMed pipeline done; UGV experiments begin $(date)" >> "$L"

run_ugv(){ # MODELPATH TAG CKPTDIR DATASET MODE NLIMIT
  local MP=$1 TAG=$2 CK=$3 DS=$4 MODE=$5 NL=$6
  local out="$CK/${TAG}_${DS}_${MODE}.json"
  if [ -f "$out" ]; then echo "  SKIP $TAG $DS $MODE (done) $(date)" >> "$L"; return; fi
  echo "  >> UGV $TAG $DS mode=$MODE n=$NL $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES=0 $PY src/labeling/run_mcq_generate_verify.py \
    --model_path "$MP" --verifier_lora "$VER" --dataset "$DS" --tag "${TAG}_${DS}_${MODE}" \
    --mode "$MODE" --n_samples 8 --temp 0.7 --cap cap320 --ckpt_dir "$CK" --tp 1 --n "$NL" >> "$L" 2>&1 \
    && echo "  OK $TAG $DS $MODE $(date)" >> "$L" || echo "  FAIL $TAG $DS $MODE $(date)" >> "$L"
}

# --- DECISIVE TEST: Lingshu-7B, options-hidden generation (the fully-unified mode) on the MCQ benchmarks ---
for ds in PMC_VQA MedXpertQA-MM VQA_RAD PATH_VQA SLAKE; do
  NL=100000; [ "$ds" = "PMC_VQA" ] && NL=3000   # subsample the 33k PMC set for tractability
  run_ugv lingshu-medical-mllm/Lingshu-7B lingshu7b ckpts/mcq_gen_verify/lingshu7b "$ds" content "$NL"
done
# --- CONTRAST: options-shown (letter) mode on two benchmarks (content-vs-letter comparison) ---
run_ugv lingshu-medical-mllm/Lingshu-7B lingshu7b ckpts/mcq_gen_verify/lingshu7b PMC_VQA   letter 3000
run_ugv lingshu-medical-mllm/Lingshu-7B lingshu7b ckpts/mcq_gen_verify/lingshu7b MedXpertQA-MM letter 100000
# --- CROSS-FAMILY robustness: MedVLThinker-7B, content mode, 3 MCQ benchmarks (Lingshu-trained verifier transfer) ---
for ds in PMC_VQA MedXpertQA-MM VQA_RAD; do
  NL=100000; [ "$ds" = "PMC_VQA" ] && NL=3000
  run_ugv /data/dan/weights/MedVLThinker-7B-RL_m23k mvt7b ckpts/mcq_gen_verify/mvt7b "$ds" content "$NL"
done

# --- real batch-1 latency for every leg (replaces the amortized proxy) ---
echo ">> batch-1 latency measurement $(date)" >> "$L"
env CUDA_VISIBLE_DEVICES=0 $PY src/cascade_methods/measure_batch1_latency_unified.py >> "$L" 2>&1 \
  && echo "  OK latency $(date)" >> "$L" || echo "  FAIL latency $(date)" >> "$L"

# --- re-run the offline unified analysis now that MCQ-generation data exists ---
echo ">> re-run unified analysis with MCQ-generation data $(date)" >> "$L"
$PY src/cascade_methods/unified_router.py >> "$L" 2>&1 || echo "  (router re-run needs a flag to include MCQ-gen; inspect)" >> "$L"
echo "UGV_EXPERIMENTS_DONE $(date)" >> "$L"
