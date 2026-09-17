#!/usr/bin/env python3
"""budget_conversion_sc32.py -- does buying COVERAGE with more samples actually buy ACCURACY?

THE QUESTION THIS SETTLES.  44.1% of all questions have no correct answer anywhere in an 8-candidate
set, and the project had been treating that as a capability wall of the 7B.  It is not: quadrupling
the budget moves oracle@N a long way (radimagenet 0.5120 -> 0.7050, vqamed 0.2102 -> 0.3524).  The
question that matters is whether the PROBE can find the newly-present correct answers, because
oracle is an upper bound nobody can spend.

WHAT IT MEASURES.  The shipped pooled probe, on HELD-OUT image halves only (it was fitted on the
other half), scoring the real N=8 / N=16 / N=32 candidate sets -- not subsamples of one pool, so
candidate-set size is the only thing that varies.  32B-judge currency throughout.

THE ANSWER, and it splits by whether the probe has selection skill on that benchmark:
  radimagenet_open  verifier-greedy  +0.1215 (N=8) -> +0.1544 (N=16) -> +0.1932 (N=32)
                    conversion of headroom  66.7% -> 59.8% -> 52.9%
      The probe converts the new coverage.  Going 8->32 is worth +0.0717 MORE accuracy, which is
      comparable to the entire shipped macro (+0.0736) on one benchmark.
  vqamed_open       verifier-greedy  -0.0033 (N=8) -> +0.0011 (N=16) -> +0.0044 (N=32)
                    conversion             -3.1%  ->    0.6%  ->    1.7%
      Oracle rises +0.1488 and the probe converts ~none of it.  The correct answers are now in the
      set and it cannot find them -- 1,667 distinct golds over 3,663 questions, greedy 0.0913.

SO: extra budget pays where the probe already has selection skill and is wasted where it does not.
That is a usable rule for adaptive-N, and it is the opposite of spending the budget where accuracy
is lowest.

  python3 src/cascade_methods/budget_conversion_sc32.py
"""
import json, os, sys, hashlib
from collections import defaultdict
import numpy as np
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute"); os.chdir(ROOT)
sys.path.insert(0,f"{ROOT}/src/training_methods"); sys.path.insert(0,f"{ROOT}/src/cascade_methods")
from pooled_selector import PooledSelector
from genframe_data import rank_avg
FEATS=f"{ROOT}/feats_hidden"; CK=f"{ROOT}/ckpts/openvqa/cheap_lingshu7b"; ENS=[18,20,22]
S=PooledSelector.load()
half=lambda im:int(hashlib.md5(("nd"+str(im)).encode()).hexdigest(),16)%2

def arm(bench, stem, gjp):
    z=np.load(f"{FEATS}/{stem}.npz"); m=json.load(open(f"{FEATS}/{stem}.meta.json"))
    lay=[int(x) for x in z["layers"]]
    if not all(L in lay for L in ENS): return None
    keep=[i for i,r in enumerate(m["rows"]) if r.get("n_tok",-1)>0 and half(r["img_md5"])==0]
    rr=[m["rows"][i] for i in keep]
    X={L:z["h_span"][keep,lay.index(L)].astype(np.float32) for L in ENS}
    y=np.array([r["y"] for r in rr],dtype=int)
    gok={}
    for l in open(gjp):
        if l.strip(): d=json.loads(l); gok[d["idx"]]=int(d["judge_ok"])
    byq=defaultdict(list)
    for i,r in enumerate(rr): byq[r["idx"]].append(i)
    qs=[q for q in byq if q in gok]
    SC=np.asarray(S.scores(X))
    if SC.ndim==1: SC=SC[None,:]
    ver=[];orc=[];gre=[]
    for q in qs:
        ii=np.array(byq[q])
        hr=np.mean([rank_avg(SC[k][ii]) for k in range(SC.shape[0])],axis=0)
        ver.append(int(y[ii][int(np.argmax(hr))])); orc.append(int(y[ii].max())); gre.append(gok[q])
    return dict(n=len(qs), verifier=float(np.mean(ver)), oracle=float(np.mean(orc)), greedy=float(np.mean(gre)))

print("DOES THE PROBE CONVERT THE EXTRA COVERAGE?  pooled probe, held-out halves\n")
print(f"{'benchmark':16s} {'N':>4s} {'greedy':>8s} {'verifier':>9s} {'oracle':>8s} {'v-greedy':>9s} {'headroom':>9s} {'converted':>10s}")
for bench,s32,lab in (("radimagenet_open","generator_eval_radimagenet_open_sc32","T=1.0"),
                      ("vqamed_open","generator_eval_vqamed_open_sc32","T=0.7")):
    gjp=f"{CK}/ckpt_{bench}_lingshu7b.judge.jsonl"
    fine=f"generator_eval_finelayer_{bench}"
    for N,stem in ((8,fine),(16,f"generator_eval_{bench}_sc16"),(32,s32)):
        r=arm(bench,stem,gjp)
        if not r: print(f"{bench:16s} {N:>4d}   (cache missing layers)"); continue
        hd=r["oracle"]-r["greedy"]; vg=r["verifier"]-r["greedy"]
        print(f"{bench:16s} {N:>4d} {r['greedy']:8.4f} {r['verifier']:9.4f} {r['oracle']:8.4f} "
              f"{vg:+9.4f} {hd:+9.4f} {100*vg/hd if hd>0 else float('nan'):9.1f}%")
    print(f"{'':16s}      (N=32 drawn at {lab}; n={r['n']})\n")
