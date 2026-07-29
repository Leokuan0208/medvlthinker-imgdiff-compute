#!/bin/bash
# Re-measure big-THINK batch-1 latency/energy with each model's NATIVE prompt (so measured gen == applied gen;
# fixes the Lingshu extrapolation bug where gen=3 native was costed via a fit on foreign gen 70-407). Then
# regenerate all analyses/charts/tables/record. medvlthinker unchanged (its think tier is already native).
set -u; cd ~/medvlthinker-imgdiff-compute; export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1; N=6
LINGSHU_INSTR='Answer with the option'"'"'s letter from the given choices and put the letter in one "\boxed{}".'
QOQ_INSTR='You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.'
CHIRON_INSTR="Let's reason step-by-step to answer the above question."
MEDGEMMA_INSTR="Reason step by step, then state your final answer as 'Final Answer: X' where X is the correct option letter."
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
rm -rf ckpts/acc_gen/lingshu/lat/big_think ckpts/acc_gen/qoq/lat/big_think ckpts/acc_gen/chiron/lat/big_think ckpts/acc_gen/medgemma/lat/big_think
echo "=== Lingshu big-think native batch-1 $(date +%H:%M) ==="
python3 src/labeling/run_vlm_eval.py --model_path lingshu-medical-mllm/Lingshu-32B --tag lat --arm think --no_system --user_instr "$LINGSHU_INSTR" --cap fullres --batch1 --max_tokens 2048 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/lingshu/lat/big_think --tp 2 --gpu_mem 0.90 > logs/natlat_lingshu.log 2>&1
echo "=== QoQ big-think native batch-1 ==="
python3 src/labeling/run_vlm_eval.py --model_path ddvd233/QoQ-Med-VL-32B --tag lat --arm think --no_system --user_instr "$QOQ_INSTR" --cap fullres --batch1 --max_tokens 2048 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/qoq/lat/big_think --tp 2 --gpu_mem 0.90 > logs/natlat_qoq.log 2>&1
echo "=== Chiron big-think native batch-1 ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path manglu3935/Chiron-o1-8B --tag lat --think --think_instr "$CHIRON_INSTR" --batch1 --max_tokens 1024 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/chiron/lat/big_think --tp 1 --gpu_mem 0.85 --max_images 4 --max_model_len 12288 > logs/natlat_chiron.log 2>&1
echo "=== MedGemma big-think native batch-1 ==="
python3 src/labeling/run_peer_eval.py --model_path google/medgemma-27b-it --tag lat --think --system "You are a helpful medical assistant." --think_instr "$MEDGEMMA_INSTR" --batch1 --max_tokens 1024 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/medgemma/lat/big_think --tp 2 --gpu_mem 0.90 --max_side 896 --max_images 4 > logs/natlat_medgemma.log 2>&1
echo "=== regenerate analyses + charts + record $(date +%H:%M) ==="
python3 src/cascade_methods/compare_native_think.py > results/cascade_methods/artifacts/native_think_compare.txt 2>&1
{ for f in medvlthinker lingshu qoq chiron medgemma; do python3 src/cascade_methods/acc_allmethods.py --family $f 2>/dev/null; done; } > results/cascade_methods/artifacts/acc_allmethods_all.txt 2>&1
{ for f in medvlthinker lingshu qoq chiron medgemma; do python3 src/cascade_methods/acc_2size.py --family $f 2>/dev/null; done; } > results/cascade_methods/artifacts/acc_2size_all.txt 2>&1
python3 src/cascade_methods/make_master_charts.py > logs/regen_charts2.log 2>&1
python3 src/cascade_methods/make_full_record.py >> logs/regen_charts2.log 2>&1
echo "=== NATLAT_REGEN_DONE $(date +%H:%M) ==="
