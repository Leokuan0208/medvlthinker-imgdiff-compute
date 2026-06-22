#!/bin/bash
# Re-run each model's THINK tier with its OWN NATIVE reasoning prompt (vs the foreign MedVLThinker <think> prompt).
# Native recipes from each model's training code/paper (workflow w2lcgd0sy). $1=n per benchmark, $2=ckpt suffix.
set -u; cd ~/medvlthinker-imgdiff-compute; export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
N=${1:-4000}; SUF=${2:-think_native}
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
LINGSHU_INSTR='Answer with the option'"'"'s letter from the given choices and put the letter in one "\boxed{}".'
QOQ_INSTR='You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.'
CHIRON_INSTR="Let's reason step-by-step to answer the above question."
MEDGEMMA_INSTR="Reason step by step, then state your final answer as 'Final Answer: X' where X is the correct option letter."
echo "=== Lingshu-32B NATIVE (empty system + boxed user instr) $(date +%H:%M) ==="
python3 src/labeling/run_vlm_eval.py --model_path lingshu-medical-mllm/Lingshu-32B --tag lingshu32b --arm think --no_system --user_instr "$LINGSHU_INSTR" --cap fullres --max_tokens 2048 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/lingshu32b/$SUF --tp 2 --gpu_mem 0.90 > logs/nt_lingshu_$SUF.log 2>&1
echo "=== QoQ-32B NATIVE (DRPO <think>+boxed user instr) $(date +%H:%M) ==="
python3 src/labeling/run_vlm_eval.py --model_path ddvd233/QoQ-Med-VL-32B --tag qoq32b --arm think --no_system --user_instr "$QOQ_INSTR" --cap fullres --max_tokens 2048 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/qoq32b/$SUF --tp 2 --gpu_mem 0.90 > logs/nt_qoq_$SUF.log 2>&1
echo "=== Chiron-8B NATIVE (step-by-step + '### The final answer is:') $(date +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path manglu3935/Chiron-o1-8B --tag chiron8b_think --think --think_instr "$CHIRON_INSTR" --max_tokens 1024 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/chiron8b/$SUF --tp 1 --gpu_mem 0.85 --max_images 4 --max_model_len 12288 > logs/nt_chiron_$SUF.log 2>&1
echo "=== MedGemma-27B NATIVE (medical system + 'Final Answer: X') $(date +%H:%M) ==="
python3 src/labeling/run_peer_eval.py --model_path google/medgemma-27b-it --tag medgemma27b_think --think --system "You are a helpful medical assistant." --think_instr "$MEDGEMMA_INSTR" --max_tokens 1024 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/medgemma27b/$SUF --tp 2 --gpu_mem 0.90 --max_side 896 --max_images 4 > logs/nt_medgemma_$SUF.log 2>&1
echo "=== NATIVE_THINK_DONE ($SUF) $(date +%H:%M) ==="
