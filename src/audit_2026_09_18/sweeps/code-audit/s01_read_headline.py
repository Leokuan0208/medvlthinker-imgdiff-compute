import json,os
ART="/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
for f in ["head_final_stack_PVFIXED_2026-09-13.json","head_final_stack_qwen_2026-09-13.json","head_final_stack_medgemma_ALL8_2026-09-16.json"]:
    a=json.load(open(os.path.join(ART,f))); print("=====",f); print({k:v for k,v in a.items() if k!="cells"})
    for c,v in a["cells"].items(): print(c, v)
