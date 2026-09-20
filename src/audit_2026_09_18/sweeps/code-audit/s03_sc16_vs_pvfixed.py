import json,os
ART="/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
a=json.load(open(f"{ART}/coverage_sc16_ci_ALL_2026-09-16.json"))
p=json.load(open(f"{ART}/head_final_stack_PVFIXED_2026-09-13.json"))
print({k:v for k,v in a.items() if k!="cells"})
print("cell | n_matched | n_images | greedy(sc16art) greedy(PVFIXED) | at8-greedy(sc16art) [ci] | pooled_ens-greedy(PVFIXED) | n(PVFIXED)")
m8=[];mp=[]
for c,v in a["cells"].items():
    pv=p["cells"][c]
    print(c, v["n_matched_questions"], v["n_images"], round(v["greedy"],4), round(pv["greedy"],4), round(v["at8_minus_greedy"]["delta"],4), [round(x,4) for x in v["at8_minus_greedy"]["ci"]], v["at8_minus_greedy"]["verdict"], round(pv["pooled_ens_minus_greedy"],4), pv["n_questions"],
          "| 16-8", round(v["budget_16_minus_8"]["delta"],4), [round(x,4) for x in v["budget_16_minus_8"]["ci"]], v["budget_16_minus_8"]["verdict"])
    m8.append(v["at8_minus_greedy"]["delta"]); mp.append(pv["pooled_ens_minus_greedy"])
import numpy as np
print("macro at8-greedy (sc16 artifact):", np.mean(m8), " PVFIXED pooled_ens macro:", np.mean(mp))
