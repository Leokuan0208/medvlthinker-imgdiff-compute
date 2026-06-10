#!/usr/bin/env python3
"""
cascade_check.py - THE fork decider. Do 7B and 32B fail on DIFFERENT questions
(complementary -> a cascade can beat 32B-alone) or the same ones (32B just
dominates -> 'use 32B' is the honest answer)?

Reads 7B labels from ckpts/gate_7b_v2 (4 arms) and 32B from ckpts/gate_32b.
Per question:
  a7_best = best of 7B's 4 arms (gives 7B every advantage)
  a7_cheap = 7B nothink_norag (the realistic cheap default a cascade would use)
  a32 = 32B think_norag
Reports, per dataset and pooled:
  (1) oracle{7B-best,32B} vs LUCK FLOOR  -> real complementarity? (z>2)
  (2) gain of oracle over 32B-alone      -> does routing beat just using 32B?
  (3) the four-cell breakdown            -> where the complementarity (if any) lives
"""
import json, glob, os, re, numpy as np
from collections import defaultdict
np.random.seed(42)

ARMS7 = ["nothink_norag","think_norag","think_rag_StatPearls","think_rag_Textbooks"]
DATASETS = ["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA"]
B = 5000

def load(ckdir, arms):
    pat = re.compile(r"ckpt_(.+?)_(" + "|".join(arms) + r")_s\dof\d\.jsonl$")
    d = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(os.path.join(ckdir,"*.jsonl"))):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r=json.loads(l); d[m.group(1)][m.group(2)][r["idx"]]=r["ok"]
    return d
lab7  = load("ckpts/gate_7b_v2", ARMS7)
lab32 = load("ckpts/gate_32b", ["think_norag"])

print("="*72); print("CASCADE COMPLEMENTARITY: 7B vs 32B"); print("="*72)
pool = {"a7":[], "a7c":[], "a32":[]}
for name in DATASETS:
    if name not in lab32 or "think_norag" not in lab32[name]:
        print(f"\n{name}: 32B not found — skip"); continue
    idx = sorted(set.intersection(set(lab32[name]["think_norag"]),
                                  *[set(lab7[name][a]) for a in ARMS7]))
    if not idx: print(f"\n{name}: no overlapping idx"); continue
    a7  = np.array([max(lab7[name][a][i] for a in ARMS7) for i in idx])
    a7c = np.array([lab7[name]["nothink_norag"][i] for i in idx])
    a32 = np.array([lab32[name]["think_norag"][i] for i in idx])
    pool["a7"]+=a7.tolist(); pool["a7c"]+=a7c.tolist(); pool["a32"]+=a32.tolist()

    oracle = (a7 | a32).mean()
    # four-cell: who gets each question
    both  = ((a7==1)&(a32==1)).mean()
    only7 = ((a7==1)&(a32==0)).mean()
    only32= ((a7==0)&(a32==1)).mean()
    neither=((a7==0)&(a32==0)).mean()
    def luck():
        c1=a7.copy(); c2=a32.copy(); np.random.shuffle(c1); np.random.shuffle(c2)
        return (c1|c2).mean()
    sims=[luck() for _ in range(B)]; mu,sd=np.mean(sims),np.std(sims)
    z=(oracle-mu)/sd if sd else 0.0
    print(f"\n### {name}  (n={len(idx)})")
    print(f"     7B best-of-4 = {a7.mean():.3f}   7B cheap(nothink) = {a7c.mean():.3f}   32B = {a32.mean():.3f}")
    print(f"     cell split:  both={both:.3f}  only-7B={only7:.3f}  only-32B={only32:.3f}  neither={neither:.3f}")
    print(f"     oracle(7B|32B) = {oracle:.3f}   luck floor = {mu:.3f} +/- {sd:.3f}   (z = {z:+.1f})")
    print(f"     >> gain of oracle over 32B-alone = {oracle-a32.mean():+.3f}")
    print(f"     >> 7B rescues {only7*len(idx):.0f} Qs the 32B got WRONG ({only7:.1%})")
    if z>2 and (oracle-a32.mean())>0.01:
        print(f"     >>> COMPLEMENTARY: a cascade can beat 32B-alone")
    elif (oracle-a32.mean())<=0.01:
        print(f"     >>> 32B DOMINATES: cascade ~= just use 32B (only-7B too small)")
    else:
        print(f"     >>> weak/ambiguous")

# pooled
a7=np.array(pool["a7"]); a7c=np.array(pool["a7c"]); a32=np.array(pool["a32"])
oracle=(a7|a32).mean()
def luckp():
    c1=a7.copy(); c2=a32.copy(); np.random.shuffle(c1); np.random.shuffle(c2)
    return (c1|c2).mean()
sims=[luckp() for _ in range(B)]; mu,sd=np.mean(sims),np.std(sims); z=(oracle-mu)/sd if sd else 0.0
only7=((a7==1)&(a32==0)).mean()
print("\n"+"="*72+"\nPOOLED (3 datasets)\n"+"="*72)
print(f"     7B best-of-4={a7.mean():.3f}  7B cheap={a7c.mean():.3f}  32B={a32.mean():.3f}")
print(f"     oracle={oracle:.3f}  luck floor={mu:.3f}+/-{sd:.3f}  (z={z:+.1f})")
print(f"     >> oracle gain over 32B-alone = {oracle-a32.mean():+.3f}")
print(f"     >> 7B rescues {only7:.1%} of questions the 32B got wrong")
print("\n" + "="*72)
print("READ: 'only-32B' large + 'only-7B' tiny => 32B dominates, cascade adds little")
print("=> the honest result is 'use 32B'. 'only-7B' nontrivial AND oracle>32B+0.01")
print("=> genuine complementarity, a cost-saving cascade (cheap 7B default, escalate")
print("hard Qs to 32B) is a real method. z<2 => even cross-model redundant.")
