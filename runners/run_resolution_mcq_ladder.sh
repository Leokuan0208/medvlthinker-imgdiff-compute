#!/bin/bash
# SWEEP 2 -- MCQ half: a resolution LADDER on the MedEvalKit (macro-8) track.
#
# WHY A FRESH CONTROL IS RUN AT THE DEFAULT RESOLUTION TOO.  The 2026-07-01 dumps
# (eval_results_lingshu7b_{full,cap320}) are a valid PAIR with each other -- same session, same
# tp=2, same vLLM -- and resolution_mcq_paired.py already uses them at full n.  Everything this
# script adds is generated at tp=1 in THIS session, so it gets its OWN default-resolution control
# arm and no ladder delta is ever taken against the July dumps (+-0.008 serving-config caveat).
#
# Cells: SLAKE, VQA_RAD, PATH_VQA, MedXpertQA-MM.  PMC-VQA (33,430) is excluded from the ladder --
# it already has the two-point answer at full n from the July pair -- and is added last if time
# allows.  MedEvalKit is NEVER modified: the cap is applied through its own CAP_MAX_PIXELS env
# lever and the invocation is copied from runners/run_full_matrix_medeval.sh.
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO/MedEvalKit"
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com \
       TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
L="$REPO/logs/resolution_mcq_ladder_2026-08-13.log"
NEED_MIB=${NEED_MIB:-42000}
DS_LIST=${DS_LIST:-"SLAKE,VQA_RAD,PATH_VQA,MedXpertQA-MM"}
CAPS=${CAPS:-"12845056 250880 501760 1003520 62720"}

say(){ echo "[$(date +%F\ %T)] $*" >> "$L"; }
# Pinned to a card so this job and the open-text generation sweep (pinned to the other) cannot
# race each other into the same free VRAM -- they did once, and one engine core was OOM-killed.
PIN_GPU=${PIN_GPU:-any}
pick_gpu(){ nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
   | awk -v n="$NEED_MIB" -v g="$PIN_GPU" -F', *' \
     '{ if ((g=="any" || $1==g) && $2-$3 >= n) { print $1; exit } }'; }

say "MCQ RESOLUTION LADDER START  caps=[$CAPS] cells=$DS_LIST"
for PX in $CAPS; do
  OUT="eval_results_res7b_px${PX}"
  for DS in $(echo "$DS_LIST" | tr ',' ' '); do
    if find "$OUT" -path "*/$DS/*" -name "*.json" 2>/dev/null | grep -q .; then
      say "SKIP(done) px=$PX $DS"; continue; fi
    # RETRY: the co-tenant on this box grows between the free-VRAM check and the model load, and
    # has OOM-killed this job twice (12:55:44, 13:05:12 -- both "Process <foreign> has 46.62 GiB").
    # NEED_MIB is therefore set well above GPU_MEM_UTIL*80GB to leave a cushion, and each job is
    # retried rather than skipped. No foreign process is ever killed.
    for ATTEMPT in 1 2 3 4 5 6; do
    if find "$OUT" -path "*/$DS/*" -name "*.json" 2>/dev/null | grep -q .; then break; fi
    G=""
    while [ -z "$G" ]; do G=$(pick_gpu); [ -z "$G" ] && { say "waiting for GPU ($NEED_MIB MiB)"; sleep 120; }; done
    say ">> px=$PX $DS on GPU $G (attempt $ATTEMPT)"
    env CUDA_VISIBLE_DEVICES="$G" tensor_parallel_size=1 CAP_MAX_PIXELS="$PX" \
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}" MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384} \
        /data/dan/medeval_venv/bin/python eval.py \
        --eval_datasets "$DS" --datasets_path hf --output_path "$OUT/{}" \
        --model_name "Qwen2.5-VL" --model_path "lingshu-medical-mllm/Lingshu-7B" \
        --seed 42 --cuda_visible_devices "$G" --tensor_parallel_size 1 --use_vllm "True" \
        --max_new_tokens 2048 --max_image_num 6 --temperature 0 --top_p 0.0001 \
        --repetition_penalty 1 --reasoning "False" --use_llm_judge "False" \
        --judge_model_type openai --judge_model None --api_key None --base_url None \
        --test_times 1 >> "$L" 2>&1 \
      && { say "OK px=$PX $DS"; break; } || { say "FAIL px=$PX $DS attempt=$ATTEMPT"; sleep 180; }
    done
  done
done
say "MCQ RESOLUTION LADDER DONE"
