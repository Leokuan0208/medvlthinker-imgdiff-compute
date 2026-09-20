#!/usr/bin/env python3
"""s08: check every number the docs quote for the replication claims against the artifacts. READ-ONLY."""
import json, os, glob
import numpy as np
A = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
FOUR = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open"]
ARMS = ["deployed_4dom_L21ish", "pooled_singlelayer", "pooled_ens", "pooled_ens_sc"]

print("### macro over the FOUR matched benchmarks, every lingshu/qwen artifact, every arm")
for f in sorted(glob.glob(os.path.join(A, "head_final_stack_*.json")) +
                glob.glob(os.path.join(A, "repro_*.json")) +
                glob.glob(os.path.join(A, "head_second_generator_*.json"))):
    a = json.load(open(f))
    cs = a.get("cells", {})
    if not cs:
        continue
    have4 = [c for c in FOUR if c in cs]
    row = {}
    for arm in ARMS:
        k = arm + "_minus_greedy"
        if all(k in cs[c] for c in have4) and len(have4) == 4:
            row[arm] = round(float(np.mean([cs[c][k] for c in have4])), 6)
    m8 = a.get("macro", {})
    print(" %-52s gen=%-16s cells=%d  macro4=%s" % (
        os.path.basename(f), a.get("generator", "lingshu"), len(cs),
        json.dumps(row)))
    print("      macro8/own = %s  beats=%s" % (
        json.dumps({k: round(v, 6) for k, v in m8.items()}), json.dumps(a.get("beats_greedy", {}))))
