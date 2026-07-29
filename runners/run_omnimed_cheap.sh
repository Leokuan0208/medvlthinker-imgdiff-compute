#!/bin/bash
# SAFE OmniMed progress: the three CHEAP legs at tp=1 (no NCCL -> no hang class; 7B/8B fit easily).
# Parser fix (OmniMedVQA cal_metrics .get()) is already in place. Strong (32B/38B, tp=2) legs are
# deferred pending a robust approach (they hit an intermittent tp=2 NCCL tail-hang; tp=1 OOMs on the 32B).
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_cheap.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
echo "OMNIMED CHEAP (tp=1) queued $(date)" > "$L"

run(){ # MODEL_NAME MODEL_PATH OUT
  local MN=$1 MP=$2 OUT=$3
  if find "$OUT" -path "*/OmniMedVQA/results.json" 2>/dev/null | grep -q .; then echo "SKIP(done) $OUT $(date)" >> "$L"; return; fi
  echo ">> OMNI-CHEAP $OUT tp=1 $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES=0 tensor_parallel_size=1 /data/dan/medeval_venv/bin/python eval.py \
    --eval_datasets "OmniMedVQA" --datasets_path "$OMNI" --output_path "$OUT/{}" --model_name "$MN" --model_path "$MP" \
    --seed 42 --cuda_visible_devices 0 --tensor_parallel_size 1 --use_vllm True --max_new_tokens 2048 \
    --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning False --use_llm_judge False \
    --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1 \
    && echo "OK $OUT $(date)" >> "$L" || echo "FAIL $OUT $(date)" >> "$L"
}

run Qwen2.5-VL lingshu-medical-mllm/Lingshu-7B            eval_results_lingshu7b_full
run Qwen2.5-VL /data/dan/weights/MedVLThinker-7B-RL_m23k   eval_results_mvt7b
run InternVL   OpenGVLab/InternVL3-8B                      eval_results_iv3_8b

echo "OMNIMED_CHEAP_DONE $(date)" >> "$L"
