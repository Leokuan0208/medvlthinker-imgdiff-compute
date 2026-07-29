#!/bin/bash
# Full 3-family x 7-benchmark MedEvalKit matrix (cascade vs always-big), resumable.
# Families: Lingshu 7B/32B, MedVLThinker 7B/32B, InternVL3 8B/38B.
# Suite (Lingshu paper Table 6): MMMU-Med, VQA-RAD, SLAKE, PathVQA, PMC-VQA, MedXpertQA-MM, OmniMedVQA(~89k, full).
# Order: complete IV3-38B on the 6-suite + IV3-8B PathVQA -> reasoning passes (MMMU/MedXpert) -> OmniMedVQA marathon.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
L=/home/jamesyang/medvlthinker-imgdiff-compute/logs/full_matrix.log
RLOG=/home/jamesyang/medvlthinker-imgdiff-compute/logs/lingshu32b_reason_mmmu.log
OMNI=/data/dan/dataset/medevalkit/OmniMedVQA_unpacked
echo "FULL MATRIX START $(date)" >> "$L"

# wait for the in-flight Lingshu-32B MMMU reasoning test to release the GPUs
until grep -qE "REASON_MMMU_OK|REASON_MMMU_FAIL" "$RLOG" 2>/dev/null; do sleep 30; done
echo "GPUs free; matrix begins $(date)" >> "$L"

# runjob: GPUS TP MODEL_NAME MODEL_PATH OUT DATASET REASONING DPATH [MAX_MODEL_LEN]
runjob(){
  local G="$1" TP="$2" MN="$3" MP="$4" OUT="$5" DS="$6" RE="$7" DP="$8" MML="${9:-}"
  if find "$OUT" -path "*/$DS/*" -name "*.json" 2>/dev/null | grep -q .; then
    echo "SKIP(done) $OUT $DS re=$RE $(date)" >> "$L"; return; fi
  echo ">> RUN $OUT $DS re=$RE tp=$TP $(date)" >> "$L"
  env CUDA_VISIBLE_DEVICES="$G" tensor_parallel_size="$TP" ${MML:+MAX_MODEL_LEN=$MML} /data/dan/medeval_venv/bin/python eval.py \
    --eval_datasets "$DS" --datasets_path "$DP" --output_path "$OUT/{}" --model_name "$MN" --model_path "$MP" \
    --seed 42 --cuda_visible_devices "$G" --tensor_parallel_size "$TP" --use_vllm "True" --max_new_tokens 2048 \
    --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --reasoning "$RE" --use_llm_judge "False" \
    --judge_model_type openai --judge_model None --api_key None --base_url None --test_times 1 >> "$L" 2>&1 \
    && echo "OK $OUT $DS re=$RE $(date)" >> "$L" || echo "FAIL $OUT $DS re=$RE $(date)" >> "$L"
}

L7="Qwen2.5-VL lingshu-medical-mllm/Lingshu-7B eval_results_lingshu7b_full"
L32="Qwen2.5-VL lingshu-medical-mllm/Lingshu-32B eval_results_lingshu32b_full"
M7="Qwen2.5-VL /data/dan/weights/MedVLThinker-7B-RL_m23k eval_results_mvt7b"
M32="Qwen2.5-VL /data/dan/weights/MedVLThinker-32B-RL_m23k eval_results_mvt32b"
I8="InternVL OpenGVLab/InternVL3-8B eval_results_iv3_8b"
I38="InternVL OpenGVLab/InternVL3-38B eval_results_iv3_38b"

echo "== PHASE 1: complete IV3 on the 6-benchmark suite (direct) ==" >> "$L"
for DS in MMMU-Medical-val VQA_RAD SLAKE PATH_VQA PMC_VQA MedXpertQA-MM; do
  runjob "0,1" 2 $I38 "$DS" False hf 16384
done
runjob 0 1 $I8 PATH_VQA False hf

echo "== PHASE 2: strong-leg REASONING passes (MMMU + MedXpert) -> think-tier test ==" >> "$L"
runjob "0,1" 2 Qwen2.5-VL lingshu-medical-mllm/Lingshu-32B eval_results_lingshu32b_reason MMMU-Medical-val True hf
runjob "0,1" 2 Qwen2.5-VL lingshu-medical-mllm/Lingshu-32B eval_results_lingshu32b_reason MedXpertQA-MM True hf
runjob "0,1" 2 Qwen2.5-VL /data/dan/weights/MedVLThinker-32B-RL_m23k eval_results_mvt32b_reason MMMU-Medical-val True hf
runjob "0,1" 2 Qwen2.5-VL /data/dan/weights/MedVLThinker-32B-RL_m23k eval_results_mvt32b_reason MedXpertQA-MM True hf
runjob "0,1" 2 InternVL OpenGVLab/InternVL3-38B eval_results_iv3_38b_reason MMMU-Medical-val True hf 16384
runjob "0,1" 2 InternVL OpenGVLab/InternVL3-38B eval_results_iv3_38b_reason MedXpertQA-MM True hf 16384

echo "== PHASE 3: OmniMedVQA (~89k, full) marathon: cheap legs then strong ==" >> "$L"
# ensure the unzip has finished
while pgrep -f "unzip -o OmniMedVQA" >/dev/null 2>&1; do sleep 60; done
runjob 0 1 $L7 OmniMedVQA False "$OMNI"
runjob 0 1 $M7 OmniMedVQA False "$OMNI"
runjob 0 1 $I8 OmniMedVQA False "$OMNI"
runjob "0,1" 2 $L32 OmniMedVQA False "$OMNI"
runjob "0,1" 2 $M32 OmniMedVQA False "$OMNI"
runjob "0,1" 2 $I38 OmniMedVQA False "$OMNI" 16384

echo "FULL_MATRIX_DONE $(date)" >> "$L"
