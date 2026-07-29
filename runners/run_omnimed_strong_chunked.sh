#!/bin/bash
# OmniMed STRONG legs (32B/38B) via CHUNKED tp=2 — option (A).
# Each leg is split into N resumable chunks (MedEvalKit native --num_chunks/--chunk_idx, sharded by
# idx%N). A tp=2 tail-hang then costs ~1/N of the run, not the whole thing, and is retried on a clean GPU.
# We deliberately DO NOT disable the NCCL heartbeat monitor: we WANT it to abort a stuck collective fast so
# the chunk fails quickly and we retry it. When all N chunks are present, MedEvalKit auto-aggregates
# (cal_metrics over the full 89k -> results.json + metrics.json). Output path uses the literal "{}" dir the
# rest of the pipeline reads from: eval_results_*/{}/OmniMedVQA/.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_BATCH_SIZE=256   # a 2000-image OmniMed batch OOMs the container memcg (~245GB); 256 => ~31GB, safe
N=6
MAX_RETRY=3
CHUNK_TIMEOUT=10800   # 3h/chunk backstop (a chunk is ~1-1.5h); primary hang-recovery is the NCCL monitor abort
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_strong.log
CHEAP=/home/jamesyang/medvlthinker-imgdiff-compute/logs/omnimed_cheap.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
echo "OMNIMED STRONG CHUNKED (N=$N, tp=2) queued $(date)" > "$L"

cleanup_gpu(){
  pkill -9 -f "eval.py.*OmniMedVQA" 2>/dev/null || true
  pkill -9 -f "spawn_main.*multiprocessing-fork" 2>/dev/null || true
  sleep 12
}

run_chunk(){ # OUT MN MP MML idx
  local OUT=$1 MN=$2 MP=$3 MML=$4 idx=$5
  local dir="$OUT/{}/OmniMedVQA"
  local rf="$dir/results_${idx}.json"
  if [ -f "$rf" ]; then echo "  SKIP chunk $idx (results_${idx}.json exists) $(date)" >> "$L"; return 0; fi
  local attempt=0
  while [ $attempt -lt $MAX_RETRY ]; do
    attempt=$((attempt+1))
    echo "  >> chunk $idx/$N attempt $attempt $(date)" >> "$L"
    timeout -s KILL $CHUNK_TIMEOUT env CUDA_VISIBLE_DEVICES=0,1 tensor_parallel_size=2 ${MML:+MAX_MODEL_LEN=$MML} \
      /data/dan/medeval_venv/bin/python eval.py \
      --eval_datasets OmniMedVQA --datasets_path "$OMNI" --output_path "$OUT/{}" \
      --model_name "$MN" --model_path "$MP" --seed 42 --cuda_visible_devices 0,1 \
      --tensor_parallel_size 2 --use_vllm True --max_new_tokens 2048 --max_image_num 6 \
      --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning False --use_llm_judge False \
      --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 \
      --num_chunks $N --chunk_idx $idx >> "$L" 2>&1
    if [ -f "$rf" ]; then echo "  OK chunk $idx (attempt $attempt) $(date)" >> "$L"; return 0; fi
    echo "  HANG/FAIL chunk $idx attempt $attempt — cleaning GPU + retrying $(date)" >> "$L"
    cleanup_gpu
  done
  echo "  GAVE UP chunk $idx after $MAX_RETRY attempts $(date)" >> "$L"; return 1
}

run_leg(){ # OUT MN MP MML
  local OUT=$1 MN=$2 MP=$3 MML=$4
  local dir="$OUT/{}/OmniMedVQA"
  if [ -f "$dir/metrics.json" ]; then echo ">> LEG $OUT already done (metrics.json) $(date)" >> "$L"; return; fi
  echo ">> LEG $OUT (N=$N chunks, MML=${MML:-none}) $(date)" >> "$L"
  cleanup_gpu
  local idx
  for idx in $(seq 0 $((N-1))); do run_chunk "$OUT" "$MN" "$MP" "$MML" "$idx"; done
  # aggregation backstop: if all N chunks present but metrics.json missing, regenerate the last chunk to trigger it
  local nres; nres=$(ls "$dir"/results_*.json 2>/dev/null | wc -l)
  if [ ! -f "$dir/metrics.json" ] && [ "$nres" -eq "$N" ]; then
    echo "  forcing aggregation (all $N chunks present, metrics.json missing) $(date)" >> "$L"
    rm -f "$dir/results_$((N-1)).json"
    run_chunk "$OUT" "$MN" "$MP" "$MML" $((N-1))
  fi
  if [ -f "$dir/metrics.json" ]; then
    echo "OK LEG $OUT -> acc $(grep -oE '\"acc\"[: ]*[0-9.]+' "$dir/metrics.json" | head -1) $(date)" >> "$L"
  else echo "FAIL LEG $OUT (incomplete chunks) $(date)" >> "$L"; fi
}

run_single(){ # OUT MN MP MML DS DPATH  (a single non-chunked benchmark, retry-guarded)
  local OUT=$1 MN=$2 MP=$3 MML=$4 DS=$5 DPATH=$6
  local m="$OUT/{}/$DS/metrics.json"
  if [ -f "$m" ]; then echo ">> SINGLE $DS $OUT already done $(date)" >> "$L"; return; fi
  echo ">> SINGLE $DS $OUT (MML=${MML:-none}) $(date)" >> "$L"; cleanup_gpu
  local attempt=0
  while [ $attempt -lt $MAX_RETRY ]; do
    attempt=$((attempt+1)); echo "  >> $DS attempt $attempt $(date)" >> "$L"
    timeout -s KILL $CHUNK_TIMEOUT env CUDA_VISIBLE_DEVICES=0,1 tensor_parallel_size=2 ${MML:+MAX_MODEL_LEN=$MML} \
      /data/dan/medeval_venv/bin/python eval.py \
      --eval_datasets "$DS" --datasets_path "$DPATH" --output_path "$OUT/{}" \
      --model_name "$MN" --model_path "$MP" --seed 42 --cuda_visible_devices 0,1 \
      --tensor_parallel_size 2 --use_vllm True --max_new_tokens 2048 --max_image_num 6 \
      --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning False --use_llm_judge False \
      --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1
    if [ -f "$m" ]; then echo "  OK $DS $OUT $(date)" >> "$L"; return; fi
    echo "  FAIL $DS attempt $attempt — cleanup + retry $(date)" >> "$L"; cleanup_gpu
  done
  echo "  GAVE UP $DS $OUT $(date)" >> "$L"
}

# wait for the cheap legs to free the GPUs (strong legs need both GPUs for tp=2)
until grep -q "OMNIMED_CHEAP_DONE" "$CHEAP" 2>/dev/null; do sleep 120; done
echo "cheap legs done; strong chunked legs begin $(date)" >> "$L"

# fill the one missing baseline cell first (quick): IV3-38B MedXpert DIRECT (no-think) @ cap 24000
run_single eval_results_iv3_38b InternVL OpenGVLab/InternVL3-38B 24000 MedXpertQA-MM hf

run_leg eval_results_lingshu32b_full Qwen2.5-VL lingshu-medical-mllm/Lingshu-32B ""
run_leg eval_results_mvt32b          Qwen2.5-VL /data/dan/weights/MedVLThinker-32B-RL_m23k ""
run_leg eval_results_iv3_38b         InternVL   OpenGVLab/InternVL3-38B 16384

echo "OMNIMED_STRONG_DONE $(date)" >> "$L"
