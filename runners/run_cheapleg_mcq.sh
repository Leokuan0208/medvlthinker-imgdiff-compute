#!/bin/bash
# Attack B -- MedEvalKit MCQ/closed evaluation for ONE arm of the cheap-leg probe.
#   $1 = model path (base snapshot, or the MERGED adapted checkpoint)
#   $2 = output tag  -> MedEvalKit/eval_results_<tag>
#   $3 = gpu id
# The command is byte-for-byte runners/run_full_matrix_medeval.sh's runjob() with tp=1, so the
# adapted arm and its matched base control are produced by the SAME harness invocation, differing
# only in --model_path.  MedEvalKit itself is NOT modified (protected dependency).
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO/MedEvalKit"
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com \
       TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
MP="$1"; TAG="$2"; GPU="$3"
OUT="eval_results_${TAG}"
L="$REPO/logs/cheapleg_mcq_${TAG}.log"
for DS in SLAKE VQA_RAD PATH_VQA MedXpertQA-MM PMC_VQA; do
  if find "$OUT" -path "*/$DS/*" -name "results.json" 2>/dev/null | grep -q .; then
    echo "SKIP(done) $OUT $DS $(date)" >> "$L"; continue; fi
  echo ">> RUN $OUT $DS $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES="$GPU" tensor_parallel_size=1 /data/dan/medeval_venv/bin/python eval.py \
    --eval_datasets "$DS" --datasets_path hf --output_path "$OUT/{}" \
    --model_name "Qwen2.5-VL" --model_path "$MP" \
    --seed 42 --cuda_visible_devices "$GPU" --tensor_parallel_size 1 --use_vllm "True" \
    --max_new_tokens 2048 --max_image_num 6 --temperature 0 --top_p 0.0001 \
    --repetition_penalty 1 --reasoning "False" --use_llm_judge "False" \
    --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 \
    >> "$L" 2>&1 && echo "OK $OUT $DS $(date)" >> "$L" || echo "FAIL $OUT $DS $(date)" >> "$L"
done
echo "CHEAPLEG_MCQ_DONE $TAG $(date)" >> "$L"
