#!/bin/bash
# SWEEP 2 -- MCQ half, the two cells the 2026-08-13 session could not reach: PATH_VQA and
# MedXpertQA-MM at cap320 against a matched 12,845,056 control generated in the SAME session.
#
# WHY THIS EXISTS SEPARATELY FROM run_resolution_mcq_ladder.sh.  That runner drives every cell with
# one set of engine parameters (MAX_MODEL_LEN 16384, --max_image_num 6, GPU_MEM_UTIL 0.30).  On
# 2026-08-14 that combination was diagnosed as the real cause of the "PATH_VQA always fails"
# symptom the previous session attributed to a co-tenant: vLLM's own profiler reported
#   model weights 15.57GiB + PyTorch activation peak 17.23GiB > gpu_memory_utilization(0.40)x79.14
#   -> "No available memory for the cache blocks"
# i.e. the job could not fit its own activation peak, co-tenant or not.  The activation peak is set
# by MAX_MODEL_LEN x max_image_num, and the two cells need very different values:
#
#   PATH_VQA      1 image/item, px_max 1,109,658  -> <=560 vision tokens, 4096 context is ample
#   MedXpertQA-MM up to 6 images, px_max 9,107,712 -> 46,816 vision tokens on the worst item,
#                 so it genuinely needs 16384 and cannot be shrunk without truncating inputs
#
# So PATH_VQA runs here at a context that fits beside a co-tenant, and MedXpert is attempted only
# when a card is empty enough for the full 16384.  MedEvalKit is NEVER modified: the resolution is
# applied through its own CAP_MAX_PIXELS env lever and the invocation is otherwise copied verbatim
# from runners/run_resolution_mcq_ladder.sh so the arms stay comparable.
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO/MedEvalKit"
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com \
       TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
L="$REPO/logs/resolution_mcq_pathvqa_2026-08-14.log"

DS=${DS:-PATH_VQA}
CAPS=${CAPS:-"12845056 250880"}
MML=${MML:-4096}
IMGN=${IMGN:-2}
GMU=${GMU:-0.40}
NEED_MIB=${NEED_MIB:-34000}

say(){ echo "[$(date +%F\ %T)] $*" >> "$L"; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
   | awk -v n="$NEED_MIB" -F', *' '{ if ($2-$3 >= n) { print $1; exit } }'; }

say "PATHVQA/MEDXPERT RESOLUTION PAIR START ds=$DS caps=[$CAPS] mml=$MML imgn=$IMGN gmu=$GMU"
for PX in $CAPS; do
  OUT="eval_results_res7b_px${PX}"
  if find "$OUT" -path "*/$DS/*" -name "results.json" 2>/dev/null | grep -q .; then
    say "SKIP(done) px=$PX $DS"; continue; fi
  for ATTEMPT in 1 2 3 4 5 6; do
    if find "$OUT" -path "*/$DS/*" -name "results.json" 2>/dev/null | grep -q .; then break; fi
    G=""; T=0
    # bounded wait: never oversubscribe, never kill a tenant, but do not hang the session either
    while [ -z "$G" ] && [ "$T" -lt 3600 ]; do
      G=$(pick_gpu); [ -z "$G" ] && { say "waiting for $NEED_MIB MiB"; sleep 120; T=$((T+120)); }
    done
    [ -z "$G" ] && { say "GIVE UP px=$PX $DS: no card with $NEED_MIB MiB in 3600s"; break; }
    say ">> px=$PX $DS on GPU $G (attempt $ATTEMPT)"
    env CUDA_VISIBLE_DEVICES="$G" tensor_parallel_size=1 CAP_MAX_PIXELS="$PX" \
        GPU_MEM_UTIL="$GMU" MAX_MODEL_LEN="$MML" \
        /data/dan/medeval_venv/bin/python eval.py \
        --eval_datasets "$DS" --datasets_path hf --output_path "$OUT/{}" \
        --model_name "Qwen2.5-VL" --model_path "lingshu-medical-mllm/Lingshu-7B" \
        --seed 42 --cuda_visible_devices "$G" --tensor_parallel_size 1 --use_vllm "True" \
        --max_new_tokens 2048 --max_image_num "$IMGN" --temperature 0 --top_p 0.0001 \
        --repetition_penalty 1 --reasoning "False" --use_llm_judge "False" \
        --judge_model_type openai --judge_model None --api_key None --base_url None \
        --test_times 1 >> "$L" 2>&1 \
      && { say "OK px=$PX $DS"; break; } || { say "FAIL px=$PX $DS attempt=$ATTEMPT"; sleep 120; }
  done
done
say "PATHVQA/MEDXPERT RESOLUTION PAIR DONE ds=$DS"
