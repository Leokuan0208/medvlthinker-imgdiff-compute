#!/bin/bash
# Robust SEQUENTIAL cheap OmniMed. Concurrent legs OOM the container memory-cgroup during the ~200GB/batch
# image preprocessing (confirmed: cgroup OOM-kill, anon-rss 245GB). ONE leg fits (the earlier single run
# reached 43/45), so run strictly one-at-a-time on GPU0 with per-leg retry x3, skip-if-done. tp=1, no NCCL.
# Signals OMNIMED_CHEAP_DONE so the chunked strong driver proceeds.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_BATCH_SIZE=256   # a 2000-image OmniMed batch OOMs the container memcg (~245GB); 256 => ~31GB, safe
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_cheap.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
echo "CHEAP SEQUENTIAL (retry, GPU0 one-at-a-time) $(date)" >> "$L"

leg(){ # MN MP OUT
  local MN=$1 MP=$2 OUT=$3 m="$3/{}/OmniMedVQA/metrics.json" a=0
  if [ -f "$m" ]; then echo "SKIP(done) $OUT $(date)" >> "$L"; return; fi
  while [ $a -lt 3 ]; do
    a=$((a+1)); echo ">> OMNI-CHEAP-SEQ $OUT attempt $a $(date)" >> "$L"
    env CUDA_VISIBLE_DEVICES=0 tensor_parallel_size=1 /data/dan/medeval_venv/bin/python eval.py \
      --eval_datasets OmniMedVQA --datasets_path "$OMNI" --output_path "$OUT/{}" --model_name "$MN" --model_path "$MP" \
      --seed 42 --cuda_visible_devices 0 --tensor_parallel_size 1 --use_vllm True --max_new_tokens 2048 \
      --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning False --use_llm_judge False \
      --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1
    if [ -f "$m" ]; then echo "OK $OUT (attempt $a) $(date)" >> "$L"; return; fi
    echo "FAIL $OUT attempt $a — retry $(date)" >> "$L"; sleep 15
  done
  echo "GAVE UP $OUT $(date)" >> "$L"
}

leg Qwen2.5-VL lingshu-medical-mllm/Lingshu-7B            eval_results_lingshu7b_full
leg Qwen2.5-VL /data/dan/weights/MedVLThinker-7B-RL_m23k   eval_results_mvt7b
leg InternVL   OpenGVLab/InternVL3-8B                      eval_results_iv3_8b

echo "OMNIMED_CHEAP_DONE $(date)" >> "$L"
