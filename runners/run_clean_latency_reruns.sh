#!/bin/bash
# (b) Clean-latency re-run of the GPU-contended legs identified by the 2026-07-02 latency-comparability audit.
# Contended cells: iv3_8b's 5 benchmarks (measured co-resident w/ a Lingshu-7B engine) + the 4 Qwen PATH_VQA legs
# (regenerated in that same contended session). This re-runs them SERIALLY, AFTER the full matrix completes, so
# there is no co-resident engine -> clean, matched-condition latency_s. Accuracy is reproduced deterministically
# (seed 42, temp 0), so each dataset either completes clean or keeps its prior (valid-accuracy) result -> no loss.
# tp is kept at each family's established basis: iv3_8b tp=1 (matches iv3_38b InternVL setup), Qwen PATH_VQA tp=2.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/clean_latency.log
M=/home/jamesyang/medvlthinker-imgdiff-compute/logs/full_matrix.log
echo "CLEAN-LATENCY RERUN queued $(date)" > "$L"

# wait for the full matrix to finish (frees the GPUs; guarantees serial / no contention)
until grep -q "FULL_MATRIX_DONE" "$M" 2>/dev/null; do sleep 120; done
echo "matrix done; clean re-runs begin $(date)" >> "$L"

# --- Phase-1 GAP: IV3-38B MedXpert failed under MAX_MODEL_LEN=16384 (its prompts are ~20k tokens > 16384).
#     Re-run direct AND reasoning with a 24000 cap (fits ~20k prompt, still <= the ~24288-token KV budget at tp=2). ---
run "0,1" 2 InternVL OpenGVLab/InternVL3-38B eval_results_iv3_38b MedXpertQA-MM 24000
echo ">> RERUN eval_results_iv3_38b_reason MedXpertQA-MM (reasoning, mml=24000) $(date)" >> "$L"
env CUDA_VISIBLE_DEVICES="0,1" tensor_parallel_size=2 MAX_MODEL_LEN=24000 /data/dan/medeval_venv/bin/python eval.py \
  --eval_datasets "MedXpertQA-MM" --datasets_path "hf" --output_path "eval_results_iv3_38b_reason/{}" --model_name "InternVL" --model_path "OpenGVLab/InternVL3-38B" \
  --seed 42 --cuda_visible_devices "0,1" --tensor_parallel_size 2 --use_vllm "True" --max_new_tokens 2048 \
  --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning "True" --use_llm_judge "False" \
  --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1 \
  && echo "OK eval_results_iv3_38b_reason MedXpertQA-MM $(date)" >> "$L" || echo "FAIL eval_results_iv3_38b_reason MedXpertQA-MM $(date)" >> "$L"

run(){  # GPUS TP MODEL_NAME MODEL_PATH OUT DATASET [MAX_MODEL_LEN]
  local G=$1 TP=$2 MN=$3 MP=$4 OUT=$5 DS=$6 MML=${7:-}
  echo ">> RERUN $OUT $DS tp=$TP ${MML:+mml=$MML} $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES="$G" tensor_parallel_size="$TP" ${MML:+MAX_MODEL_LEN=$MML} /data/dan/medeval_venv/bin/python eval.py \
    --eval_datasets "$DS" --datasets_path "hf" --output_path "$OUT/{}" --model_name "$MN" --model_path "$MP" \
    --seed 42 --cuda_visible_devices "$G" --tensor_parallel_size "$TP" --use_vllm "True" --max_new_tokens 2048 \
    --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning "False" --use_llm_judge "False" \
    --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1 \
    && echo "OK $OUT $DS $(date)" >> "$L" || echo "FAIL $OUT $DS $(date)" >> "$L"
}

# --- iv3_8b (tp=1) : the 5 contended benchmarks ---
for DS in MMMU-Medical-val PMC_VQA SLAKE VQA_RAD MedXpertQA-MM; do
  run 0 1 InternVL OpenGVLab/InternVL3-8B eval_results_iv3_8b "$DS"
done
# --- the 4 Qwen PATH_VQA legs (tp=2, matches each family's other datasets) ---
run "0,1" 2 Qwen2.5-VL lingshu-medical-mllm/Lingshu-7B  eval_results_lingshu7b_full  PATH_VQA
run "0,1" 2 Qwen2.5-VL lingshu-medical-mllm/Lingshu-32B eval_results_lingshu32b_full PATH_VQA
run "0,1" 2 Qwen2.5-VL /data/dan/weights/MedVLThinker-7B-RL_m23k  eval_results_mvt7b  PATH_VQA
run "0,1" 2 Qwen2.5-VL /data/dan/weights/MedVLThinker-32B-RL_m23k eval_results_mvt32b PATH_VQA

# --- regenerate the cascade tables with now-clean latency for the affected families ---
echo "=== clean-latency cascade tables ===" >> "$L"
cd /home/jamesyang/medvlthinker-imgdiff-compute
python3 src/cascade_methods/cascade_all_families.py Lingshu     eval_results_lingshu7b_full eval_results_lingshu32b_full >> "$L" 2>&1
python3 src/cascade_methods/cascade_all_families.py MedVLThinker eval_results_mvt7b         eval_results_mvt32b         >> "$L" 2>&1
python3 src/cascade_methods/cascade_all_families.py InternVL3   eval_results_iv3_8b         eval_results_iv3_38b        >> "$L" 2>&1
echo "CLEAN_LATENCY_DONE $(date)" >> "$L"
