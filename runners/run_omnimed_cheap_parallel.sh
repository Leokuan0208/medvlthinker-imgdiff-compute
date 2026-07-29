#!/bin/bash
# Parallel cheap OmniMed across BOTH GPUs. Assumes Lingshu-7B is already running (orphaned) on GPU0.
# Runs MVT-7B on GPU1 now (parallel), then IV3-8B on GPU0 once Lingshu-7B frees. Writes OMNIMED_CHEAP_DONE
# to the shared log so the strong-chunked driver can proceed. tp=1 => no NCCL, legs fit on one GPU each.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_cheap.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
LING_METRICS="eval_results_lingshu7b_full/{}/OmniMedVQA/metrics.json"
echo "PARALLEL cheap: MVT-7B on GPU1 now; IV3-8B on GPU0 after Lingshu-7B $(date)" >> "$L"

leg(){ # GPU MN MP OUT
  local G=$1 MN=$2 MP=$3 OUT=$4
  local m="$OUT/{}/OmniMedVQA/metrics.json"
  if [ -f "$m" ]; then echo "SKIP(done) $OUT $(date)" >> "$L"; return; fi
  echo ">> OMNI-CHEAP-P $OUT gpu=$G $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES=$G tensor_parallel_size=1 /data/dan/medeval_venv/bin/python eval.py \
    --eval_datasets OmniMedVQA --datasets_path "$OMNI" --output_path "$OUT/{}" \
    --model_name "$MN" --model_path "$MP" --seed 42 --cuda_visible_devices "$G" \
    --tensor_parallel_size 1 --use_vllm True --max_new_tokens 2048 --max_image_num 6 \
    --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning False --use_llm_judge False \
    --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1 \
    && echo "OK $OUT $(date)" >> "$L" || echo "FAIL $OUT $(date)" >> "$L"
}

# MVT-7B on GPU1 in parallel with the already-running Lingshu-7B (GPU0)
leg 1 Qwen2.5-VL /data/dan/weights/MedVLThinker-7B-RL_m23k eval_results_mvt7b &
MVT_PID=$!

# wait for Lingshu-7B (GPU0) to finish (its Python writes metrics.json), then run IV3-8B on GPU0
until [ -f "$LING_METRICS" ]; do sleep 60; done
echo "Lingshu-7B done (metrics.json seen); starting IV3-8B on GPU0 $(date)" >> "$L"
leg 0 InternVL OpenGVLab/InternVL3-8B eval_results_iv3_8b

wait "$MVT_PID"
echo "OMNIMED_CHEAP_DONE $(date)" >> "$L"
