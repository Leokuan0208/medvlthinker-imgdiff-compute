#!/usr/bin/env python3
"""compare_native_think.py - native-prompt think vs foreign-prompt think vs no-think, per family.
Tests how much of the cross-family 'think hurts' was a PROMPT artifact (foreign MedVLThinker <think>) vs real.
Reads existing no-think + old (foreign) think + new native-think labels. No new inference."""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np; from collections import defaultdict
from harness import ALL6, ALL5
BAB = {"PMC-VQA": "PMC", "SLAKE": "SLAKE", "VQA-RAD": "VQARAD", "PathVQA": "PathV", "MMMU": "MMMU",
       "MedXpert-Reasoning": "MX-R", "MedXpert-Understanding": "MX-U"}
FAM = {  # nt dir, foreign-think dir, native-think dir
  "lingshu": ("ckpts/acc_gen/lingshu32b/nothink_cap320", "ckpts/acc_gen/lingshu32b/think_fullres", "ckpts/acc_gen/lingshu32b/think_native"),
  "qoq": ("ckpts/acc_gen/qoq32b/nothink_cap320", "ckpts/acc_gen/qoq32b/think_fullres", "ckpts/acc_gen/qoq32b/think_native"),
  "chiron": ("ckpts/acc_gen/chiron8b/nt", "ckpts/acc_gen/chiron8b/think", "ckpts/acc_gen/chiron8b/think_native"),
  "medgemma": ("ckpts/acc_gen/medgemma27b/nt", "ckpts/acc_gen/medgemma27b/think", "ckpts/acc_gen/medgemma27b/think_native")}
def loadarm(d):
    o = defaultdict(dict)
    for f in glob.glob(os.path.join(d, "ckpt_*.jsonl")):
        m = re.match(r"ckpt_(.+?)_", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); o[m.group(1)][r["idx"]] = r
    return o
out = {}
for fam, (ntd, ftd, nvd) in FAM.items():
    nt, ft, nv = loadarm(ntd), loadarm(ftd), loadarm(nvd)
    print(f"\n######## {fam.upper()}: no-think vs FOREIGN-think vs NATIVE-think ########")
    print(f"  {'benchmark':<22}{'no-think':>9}{'foreign':>9}{'native':>9}{'nat-frgn':>10}{'nat-nt':>9}{'gen_nat':>9}")
    perf = {}
    for ds in ALL6:
        ii = sorted(set(nt.get(ds, {})) & set(ft.get(ds, {})) & set(nv.get(ds, {}))) if nv.get(ds) else []
        if not ii: continue
        a_nt = np.mean([nt[ds][i]["ok"] for i in ii]); a_ft = np.mean([ft[ds][i]["ok"] for i in ii]); a_nv = np.mean([nv[ds][i]["ok"] for i in ii])
        g_nv = np.mean([nv[ds][i].get("gen_tokens", 0) for i in ii])
        perf[ds] = (a_nt, a_ft, a_nv, len(ii))
        print(f"  {ds:<22}{a_nt:>9.3f}{a_ft:>9.3f}{a_nv:>9.3f}{a_nv-a_ft:>+10.3f}{a_nv-a_nt:>+9.3f}{g_nv:>9.0f}")
    for lab, names in [("POOLED ALL-5", ALL5), ("POOLED ALL-6", ALL6)]:
        w = [(d, perf[d]) for d in names if d in perf]
        if not w: continue
        tot = sum(p[3] for _, p in w)
        ant = sum(p[0] * p[3] for _, p in w) / tot; aft = sum(p[1] * p[3] for _, p in w) / tot; anv = sum(p[2] * p[3] for _, p in w) / tot
        print(f"  {lab:<22}{ant:>9.3f}{aft:>9.3f}{anv:>9.3f}{anv-aft:>+10.3f}{anv-ant:>+9.3f}")
        out.setdefault(fam, {})[lab] = {"no_think": ant, "foreign_think": aft, "native_think": anv}
json.dump(out, open("results/cascade_methods/native_think_compare.json", "w"), indent=1)
print("\n-> results/cascade_methods/native_think_compare.json")
print("READ: native>foreign => the foreign prompt was hurting (artifact). native<no-think still => real over-thinking.")
