#!/bin/bash
# FAST + robust parallel cheap OmniMed. batch=256 keeps each leg at ~9GB RAM, so two legs run at once
# (GPU0 + GPU1) with no cgroup OOM. Per-leg timeout (auto-kills a hang) + retry x4 + skip-if-done (resumable).
# Writes OMNIMED_CHEAP_DONE so the strong chunked driver proceeds.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_BATCH_SIZE=256
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_cheap.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
TO=21600   # 6h/leg hard backstop (a leg is ~2-3h); the watchdog kills a stalled leg much sooner

leg(){ # GPU MN MP OUT
  local G=$1 MN=$2 MP=$3 OUT=$4 m="$4/{}/OmniMedVQA/metrics.json" a=0
  if [ -f "$m" ]; then echo "SKIP(done) $OUT $(date)" >> "$L"; return; fi
  while [ $a -lt 4 ]; do
    a=$((a+1)); echo ">> OMNI-CHEAP-PAR $OUT gpu=$G attempt $a $(date)" >> "$L"
    timeout -s KILL $TO env CUDA_VISIBLE_DEVICES="$G" tensor_parallel_size=1 /data/dan/medeval_venv/bin/python eval.py \
      --eval_datasets OmniMedVQA --datasets_path "$OMNI" --output_path "$OUT/{}" --model_name "$MN" --model_path "$MP" \
      --seed 42 --cuda_visible_devices "$G" --tensor_parallel_size 1 --use_vllm True --max_new_tokens 2048 \
      --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning False --use_llm_judge False \
      --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1
    if [ -f "$m" ]; then echo "OK $OUT (attempt $a) $(date)" >> "$L"; return; fi
    echo "FAIL/HANG $OUT attempt $a — retry $(date)" >> "$L"; sleep 15
  done
  echo "GAVE UP $OUT $(date)" >> "$L"
}

echo "CHEAP PARALLEL (batch=256, 2 GPUs, timeout+retry) $(date)" >> "$L"
# two 7B legs concurrently on the two GPUs, then the 8B on GPU0
leg 0 Qwen2.5-VL lingshu-medical-mllm/Lingshu-7B            eval_results_lingshu7b_full &
leg 1 Qwen2.5-VL /data/dan/weights/MedVLThinker-7B-RL_m23k   eval_results_mvt7b &
wait
leg 0 InternVL   OpenGVLab/InternVL3-8B                      eval_results_iv3_8b
echo "OMNIMED_CHEAP_DONE $(date)" >> "$L"
