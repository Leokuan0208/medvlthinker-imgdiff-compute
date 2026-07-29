#!/bin/bash
# OmniMedVQA full re-run with BOTH fixes: (a) the OmniMedVQA.cal_metrics .get() parser fix (already in the
# gitignored MedEvalKit), and (b) the NCCL fix below. At tp=2 the ~89k OmniMed data load kept rank0 busy long
# enough to trip rank1's NCCL heartbeat monitor -> the process group aborted and the run hung. Disabling the
# heartbeat monitor + extending the timeouts lets the long data load complete. Strong legs run FIRST so the
# NCCL fix is verified early (that is the exact configuration that hung).
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
# --- NCCL fix: don't let the long OmniMed data load trip the distributed watchdog ---
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 TORCH_NCCL_TRACE_BUFFER_SIZE=0 NCCL_TIMEOUT=7200
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_rerun.log
F=/home/jamesyang/medvlthinker-imgdiff-compute/logs/clean_latency.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
echo "OMNIMED RERUN queued $(date)" > "$L"

# wait for the clean-latency fixup to finish (frees the GPUs)
until grep -q "CLEAN_LATENCY_DONE" "$F" 2>/dev/null; do sleep 120; done
echo "fixup done; OmniMed re-runs begin $(date)" >> "$L"

run(){ # GPUS TP MODEL_NAME MODEL_PATH OUT [MAX_MODEL_LEN]
  local G=$1 TP=$2 MN=$3 MP=$4 OUT=$5 MML=${6:-}
  # skip only if a COMPLETE result exists (results.json is written only on success)
  if find "$OUT" -path "*/OmniMedVQA/results.json" 2>/dev/null | grep -q .; then echo "SKIP(done) $OUT $(date)" >> "$L"; return; fi
  echo ">> OMNI $OUT tp=$TP $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES="$G" tensor_parallel_size="$TP" ${MML:+MAX_MODEL_LEN=$MML} /data/dan/medeval_venv/bin/python eval.py \
    --eval_datasets "OmniMedVQA" --datasets_path "$OMNI" --output_path "$OUT/{}" --model_name "$MN" --model_path "$MP" \
    --seed 42 --cuda_visible_devices "$G" --tensor_parallel_size "$TP" --use_vllm "True" --max_new_tokens 2048 \
    --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning "False" --use_llm_judge "False" \
    --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1 \
    && echo "OK $OUT $(date)" >> "$L" || echo "FAIL $OUT $(date)" >> "$L"
}

# --- STRONG legs first (tp=2 — the config that hung; verifies the NCCL fix early) ---
run "0,1" 2 Qwen2.5-VL lingshu-medical-mllm/Lingshu-32B         eval_results_lingshu32b_full
run "0,1" 2 Qwen2.5-VL /data/dan/weights/MedVLThinker-32B-RL_m23k eval_results_mvt32b
run "0,1" 2 InternVL   OpenGVLab/InternVL3-38B                  eval_results_iv3_38b 16384
# --- cheap legs (tp=1 — no NCCL; parser fix covers their earlier crash) ---
run 0 1 Qwen2.5-VL lingshu-medical-mllm/Lingshu-7B          eval_results_lingshu7b_full
run 0 1 Qwen2.5-VL /data/dan/weights/MedVLThinker-7B-RL_m23k  eval_results_mvt7b
run 0 1 InternVL   OpenGVLab/InternVL3-8B                    eval_results_iv3_8b

echo "OMNIMED_RERUN_DONE $(date)" >> "$L"
