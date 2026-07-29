#!/bin/bash
# CHEAP FIX: the orphaned Lingshu-7B OmniMed leg died at ~43/45 without writing output (caught by a broad
# eval.py cleanup during the strong-leg work). Re-run it on GPU0 now; run IV3-8B on GPU1 once MVT-7B (still
# running there) finishes; then signal OMNIMED_CHEAP_DONE so the chunked strong driver proceeds. tp=1, no NCCL.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_cheap.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
MVT_METRICS="eval_results_mvt7b/{}/OmniMedVQA/metrics.json"
echo "CHEAP FIX $(date): re-run Lingshu-7B (GPU0); IV3-8B (GPU1 after MVT-7B); MVT-7B still running on GPU1" >> "$L"

leg(){ # GPU MN MP OUT
  local G=$1 MN=$2 MP=$3 OUT=$4
  local m="$OUT/{}/OmniMedVQA/metrics.json"
  if [ -f "$m" ]; then echo "SKIP(done) $OUT $(date)" >> "$L"; return; fi
  echo ">> OMNI-CHEAP-FIX $OUT gpu=$G $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES="$G" tensor_parallel_size=1 /data/dan/medeval_venv/bin/python eval.py \
    --eval_datasets OmniMedVQA --datasets_path "$OMNI" --output_path "$OUT/{}" --model_name "$MN" --model_path "$MP" \
    --seed 42 --cuda_visible_devices "$G" --tensor_parallel_size 1 --use_vllm True --max_new_tokens 2048 \
    --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning False --use_llm_judge False \
    --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1 \
    && echo "OK $OUT $(date)" >> "$L" || echo "FAIL $OUT $(date)" >> "$L"
}

# GPU0: re-run Lingshu-7B now (background)
leg 0 Qwen2.5-VL lingshu-medical-mllm/Lingshu-7B eval_results_lingshu7b_full &
LING_PID=$!

# GPU1: once MVT-7B (running) writes its metrics.json, run IV3-8B there
until [ -f "$MVT_METRICS" ]; do sleep 60; done
echo "MVT-7B done (metrics.json); starting IV3-8B on GPU1 $(date)" >> "$L"
leg 1 InternVL OpenGVLab/InternVL3-8B eval_results_iv3_8b

wait "$LING_PID"
echo "OMNIMED_CHEAP_DONE $(date)" >> "$L"
