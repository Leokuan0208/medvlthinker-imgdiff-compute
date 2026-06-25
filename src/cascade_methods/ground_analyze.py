#!/usr/bin/env python3
"""Analyze SLAKE grounding: does spatial SELF-CONSISTENCY predict box correctness (IoU), i.e. does a
verifiable/structured output ESCAPE the free-text luck floor? Parses Lingshu boxes (normalized or absolute),
computes IoU vs gold, and compares greedy / random-sample / SC-medoid / oracle@8, plus the AUROC of
SC-agreement (mean pairwise IoU) predicting correct@0.5. Run: python3 src/cascade_methods/ground_analyze.py"""
import json, os, re, numpy as np
from sklearn.metrics import roc_auc_score
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
F=os.path.join(ROOT,"ckpts/ground/slake_lingshu7b.jsonl")
def parse_box(s, W, H):
    nums=re.findall(r"-?\d+\.?\d*", s.replace(","," "))
    if len(nums)<4: return None
    v=[float(x) for x in nums[:4]]
    if max(v)<=1.5:                       # normalized 0-1
        v=[v[0]*W, v[1]*H, v[2]*W, v[3]*H]
    x1,y1,x2,y2=v
    if x2<x1: x1,x2=x2,x1
    if y2<y1: y1,y2=y2,y1
    return [max(0,x1),max(0,y1),min(W,x2),min(H,y2)]
def iou(a,b):
    if a is None or b is None: return 0.0
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
rows=[json.loads(l) for l in open(F) if l.strip()]
print(f"n={len(rows)} grounding targets")
def agg(rows, thr=0.5, kind=None):
    rows=[r for r in rows if (kind is None or r["kind"]==kind)]
    g_iou=[]; rnd=[]; medoid=[]; oracle=[]; sc=[]; corr=[]
    for r in rows:
        W,H=r["W"],r["H"]; gold=r["gold"]
        gb=parse_box(r["greedy_raw"],W,H); g_iou.append(iou(gb,gold))
        sb=[parse_box(x,W,H) for x in r["sample_raws"]]
        ious=[iou(b,gold) for b in sb]
        oracle.append(max(ious) if ious else 0.0)
        rnd.append(ious[0] if ious else 0.0)
        # SC: mean pairwise IoU per sample; medoid = max mean-IoU-to-others
        n=len(sb); pair=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j: pair[i,j]=iou(sb[i],sb[j])
        meaniou=pair.sum(1)/max(n-1,1)
        mi=int(np.argmax(meaniou)); medoid.append(ious[mi] if ious else 0.0)
        sc.append(float(meaniou.mean()))          # overall agreement for this item
        corr.append(1 if (max(ious) if ious else 0)>=thr else 0)  # recoverable-correct for AUROC target
    return dict(n=len(rows), greedy=np.mean(np.array(g_iou)>=thr), greedy_iou=np.mean(g_iou),
                random=np.mean(np.array(rnd)>=thr), medoid=np.mean(np.array(medoid)>=thr),
                medoid_iou=np.mean(medoid), oracle=np.mean(np.array(oracle)>=thr),
                sc=np.array(sc), correct_greedy=np.array([1 if x>=thr else 0 for x in g_iou]))
for thr in (0.5, 0.3):
    print(f"\n===== IoU threshold {thr} =====")
    print(f"{'subset':<10} {'n':>4} {'greedy':>7} {'random':>7} {'SC-medoid':>9} {'oracle@8':>9} | {'greedyIoU':>9} {'medoidIoU':>9}")
    for kind in (None,"organ","abn"):
        a=agg(rows,thr,kind); lab={None:"ALL","organ":"organ","abn":"abnormality"}[kind]
        print(f"{lab:<10} {a['n']:>4} {a['greedy']:>7.3f} {a['random']:>7.3f} {a['medoid']:>9.3f} {a['oracle']:>9.3f} | {a['greedy_iou']:>9.3f} {a['medoid_iou']:>9.3f}")
    # KEY: does SC-agreement predict greedy correctness? (structured-output analog of free-text SC)
    a=agg(rows,thr,None)
    sc=a['sc']; cg=a['correct_greedy']
    if len(set(cg))>1:
        au=roc_auc_score(cg, sc)
        print(f"  AUROC: SC-agreement (mean pairwise IoU) predicts greedy-correct@{thr}: {au:.3f}  (free-text SC was luck-floored ~0.5-0.55)")
print("\nREAD: if SC-medoid >> greedy toward oracle AND SC-agreement AUROC >> 0.6, structured outputs")
print("      ESCAPE the free-text luck floor -> a novel positive (selection works when output is verifiable).")
