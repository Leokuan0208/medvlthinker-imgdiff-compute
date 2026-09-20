set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
echo "##### B3: seqlogprob nullity across sc8 dumps (first line of each) #####"
for f in $(ls ckpts/openvqa/cheap_lingshu7b/*_sc8.jsonl ckpts/openvqa/cheap_lingshu7b/*_sc16.jsonl 2>/dev/null | head -20); do
  v=$(head -1 "$f" | python3 -c "import sys,json; d=json.loads(sys.stdin.readline()); print('seqlogprob=',d.get('seqlogprob'),'keys=',sorted(d.keys()))" 2>/dev/null)
  echo "  $(basename $f) :: $v"
done
echo
echo "##### B3b: does ANY openvqa dump carry per-candidate logprobs? #####"
grep -l '"seqlogprobs"\|"logprobs"\|"cand_logprob"\|"per_cand_logprob"' -r ckpts/openvqa/ 2>/dev/null | head -5
echo "  (empty above = none)"
echo
echo "##### B7: 32B open-text dumps #####"
for d in strong_lingshu strong_lingshu_bo strong verifier32b judgeval strong_lingshu_direct_unstyled; do
  echo "--- ckpts/openvqa/$d"; ls ckpts/openvqa/$d 2>/dev/null | head -25
done
echo
echo "##### B9: claude_judge #####"
ls -la results/cascade_methods/claude_judge/ 2>/dev/null
echo
echo "##### B8: image-ablation scripts + artifacts #####"
ls -la src/training_methods/extract_generator_hidden_ablated.py src/training_methods/langside_image_dependence_cv.py 2>/dev/null
ls results/cascade_methods/artifacts/ | grep -i -E "ablat|langside|image_dep|noimg|visual"
echo
echo "##### B2: July LoRA verifier coverage #####"
ls results/cascade_methods/artifacts/ | grep -i -E "free_head|crossfam|verifier32b|bestofn"
echo
echo "##### B5/B6: probe-weighted vote / greedy-included #####"
grep -rln "greedy_included\|greedy-included\|weighted_major\|weighted vote\|score_weighted" src/ results/cascade_methods/ 2>/dev/null | head
echo "  (empty = none)"
