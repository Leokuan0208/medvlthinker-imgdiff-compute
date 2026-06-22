#!/usr/bin/env python3
"""
overthink_generalize.py - does ACC's core premise ("no-think >= think on perception VQA") generalize
ACROSS VLM architectures? Compares no-think vs think accuracy per model per competent-4 benchmark,
aligned by idx. Cross-family cached models (InternVL2.5-8B, Phi-3.5-V) + MedVLThinker 7B/32B reference.
Run from repo root:  python3 src/cascade_methods/overthink_generalize.py
"""
import os, glob, json, re; import numpy as np
from collections import defaultdict
COMP = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
def load(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"ckpt_*{tag}*.jsonl")):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out
# (model label, no-think dir/tag, think dir/tag)
MODELS = [
    ("InternVL2.5-8B (cross-fam)", "ckpts/peer/internvl25_8b", "internvl25_8b", "ckpts/acc_gen/internvl25_8b_think", "internvl25_8b_think"),
    ("Phi-3.5-V (cross-fam)",      "ckpts/peer/phi35v", "phi35v", "ckpts/acc_gen/phi35v_think", "phi35v_think"),
    ("MedVLThinker-32B (ref)",     "ckpts/gate_32b_modes/nothink_fullres", "nothink_norag", "ckpts/gate_32b", "think_norag"),
    ("MedVLThinker-7B (ref)",      "ckpts/gate_7b_prune/cap320", "nothink_norag", "ckpts/gate_7b_think", "think_norag"),
]
print(f"{'model':<28}{'benchmark':<10}{'nothink':>8}{'think':>8}{'Δ(th-nt)':>9}{'n':>6}")
print("-" * 70)
for label, ntd, ntt, thd, tht in MODELS:
    nt = load(ntd, ntt); th = load(thd, tht)
    if not nt or not th: print(f"{label:<28} (labels missing: nt={bool(nt)} th={bool(th)})"); continue
    pooled_nt, pooled_th = [], []
    for ds in COMP:
        idx = sorted(set(nt.get(ds, {})) & set(th.get(ds, {})))
        if not idx: continue
        a_nt = np.array([nt[ds][i]["ok"] for i in idx]); a_th = np.array([th[ds][i]["ok"] for i in idx])
        pooled_nt += list(a_nt); pooled_th += list(a_th)
        d = a_th.mean() - a_nt.mean()
        flag = "  <- think hurts" if d < -0.005 else ("  think helps" if d > 0.005 else "")
        print(f"{label:<28}{ds:<10}{a_nt.mean():>8.3f}{a_th.mean():>8.3f}{d:>+9.3f}{len(idx):>6}{flag}")
    if pooled_nt:
        pnt, pth = np.array(pooled_nt), np.array(pooled_th)
        print(f"{label:<28}{'POOLED':<10}{pnt.mean():>8.3f}{pth.mean():>8.3f}{pth.mean()-pnt.mean():>+9.3f}{len(pnt):>6}  ***")
    print("-" * 70)
print("\nACC premise generalizes if think <= no-think (Δ<=0) on perception across architectures.")
