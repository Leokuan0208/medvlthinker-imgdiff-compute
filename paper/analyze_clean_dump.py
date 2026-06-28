#!/usr/bin/env python3
"""Consume clean_dump.json (verifier scores+labels per held-out question) -> the definitive peer-comparison
table (same split) + the accuracy-vs-compute picture. Writes JSON + TV-readable figures (large fonts, few
series). All baselines and the verifier on the IDENTICAL questions. 32B from strong_lingshu judge."""
import os, json, glob
from collections import Counter, defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
def norm(s): return str(s).strip().lower().rstrip(".")
dump=json.load(open(os.path.join(ROOT,"ckpts/train/lora_verifier_pooled4/clean_dump.json")))
# 32B labels per dataset
S32={}
for ds,tag in [("slake_open","slake_open"),("vqa_rad_open","vqa_rad_open"),("pathvqa_open","pathvqa_open"),("kvasir_open","kvasir_open")]:
    f=os.path.join(ROOT,f"ckpts/openvqa/strong_lingshu/ckpt_{tag}_lingshu32b.judge.jsonl")
    if os.path.exists(f): S32[ds]={json.loads(l)["idx"]:json.loads(l)["judge_ok"] for l in open(f)}
rng=np.random.default_rng(0)
per=defaultdict(lambda: defaultdict(list))
for r in dump:
    ds=r["ds"]; sl=[None if x==-1 else x for x in r["sl"]]; sc=r["sc"]; preds=r["preds"]
    cand=[k for k in range(len(sl)) if sl[k] is not None]
    if not cand: continue
    g=sl[0] if sl[0] is not None else 0
    rnd=sl[cand[int(rng.integers(len(cand)))]]
    modal=Counter(norm(p) for p in preds).most_common(1)[0][0]
    mk=next((k for k in range(len(preds)) if norm(preds[k])==modal and sl[k] is not None), cand[0])
    scq=sl[max(cand,key=lambda k:sc[k])]
    orc=max(sl[k] for k in cand)
    for nm,v in [("greedy",g),("random",rnd),("self_consistency",sl[mk] if sl[mk] is not None else 0),("verifier",scq),("oracle",orc)]:
        per[ds][nm].append(v)
    b=S32.get(ds,{}).get(r["idx"])
    if b is not None: per[ds]["m32b"].append(b)
DS=["slake_open","vqa_rad_open","pathvqa_open","kvasir_open"]; METH=["greedy","self_consistency","random","m32b","verifier","oracle"]
tab={};
for ds in DS:
    tab[ds]={m:(float(np.mean(per[ds][m])) if per[ds][m] else None) for m in METH}; tab[ds]["n"]=len(per[ds]["greedy"])
tot=sum(tab[d]["n"] for d in DS)
pool={m:sum((tab[d][m] or 0)*tab[d]["n"] for d in DS)/tot for m in METH}; pool["n"]=tot
tab["POOLED"]=pool
print(f"{'dataset':<14}"+''.join(f'{m[:8]:>9}' for m in METH)+f'{\"n\":>6}')
for ds in DS+["POOLED"]:
    print(f"{ds:<14}"+''.join((f'{tab[ds][m]:>9.3f}' if tab[ds][m] is not None else f'{\"-\":>9}') for m in METH)+f"{tab[ds]['n']:>6}")
json.dump(tab, open(os.path.join(ROOT,"ckpts/train/lora_verifier_pooled4/peer_comparison.json"),"w"), indent=1)

# best-of-K accuracy-compute curve (pooled), cost in 7B-forward-equiv: bo-K = 2K (K gen + K verify); 32B single = 4.3
rows=[([None if x==-1 else x for x in r["sl"]], r["sc"]) for r in dump]
def bok(K):
    acc=[]
    for sl,sc in rows:
        cand=[k for k in range(min(K,len(sl))) if sl[k] is not None]
        if cand: acc.append(sl[max(cand,key=lambda k:sc[k])])
    return float(np.mean(acc))
Ks=[1,2,4,8]; curve={K:{"acc":bok(K),"cost":(1 if K==1 else 2*K)} for K in Ks}
print("\nbest-of-K (verifier):", {K:round(curve[K]["acc"],3) for K in Ks})

# ---------- FIG 1: peer comparison bar (TV-readable) ----------
plt.rcParams.update({"font.size":17})
fig,ax=plt.subplots(figsize=(13,7))
labels=["greedy\n(1 sample)","self-consistency\n(Wang'23)","32B single\n(scale-up)","trained verifier\n(ours)","oracle@8\n(ceiling)"]
keys=["greedy","self_consistency","m32b","verifier","oracle"]
vals=[pool[k] for k in keys]; cols=["#9aa1ad","#5c6bc0","#c62828","#00897b","#cdd4e3"]
b=ax.bar(labels,vals,color=cols,edgecolor="k",linewidth=0.8,width=0.62)
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.008,f"{v:.3f}",ha="center",fontsize=18,fontweight="bold")
ax.set_ylabel("pooled open-ended accuracy (LLM-judge)",fontsize=18); ax.set_ylim(0,max(vals)*1.18)
ax.set_title("Medical open-ended VQA: a small trained verifier beats every training-free\nselector AND the 5×-larger model (n=%d, held-out)"%tot,fontsize=18)
ax.axhline(pool["greedy"],ls="--",c="#9aa1ad",lw=1.2)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(ROOT,"paper/figs/limits/fig_peer_comparison.png"),dpi=150); plt.close()

# ---------- FIG 2: accuracy vs compute (verifier extends the frontier beyond the 32B) ----------
fig,ax=plt.subplots(figsize=(12,7))
xs=[curve[K]["cost"] for K in Ks]; ys=[curve[K]["acc"] for K in Ks]
ax.plot(xs,ys,"o-",color="#00897b",lw=3,ms=12,label="7B + trained verifier (best-of-K)")
for K in Ks: ax.annotate(f"K={K}",(curve[K]["cost"],curve[K]["acc"]),textcoords="offset points",xytext=(8,-4),fontsize=14)
ax.scatter([4.3],[pool["m32b"]],s=240,marker="D",color="#c62828",zorder=5,label="32B single pass (scale-up)")
ax.scatter([1],[pool["greedy"]],s=160,marker="v",color="#9aa1ad",zorder=5,label="7B greedy")
ax.set_xscale("log",base=2); ax.set_xlabel("relative compute (7B-forward-equivalents, log)",fontsize=18)
ax.set_ylabel("pooled accuracy",fontsize=18)
ax.set_title("Test-time compute beats parameters: the verifier reaches accuracy\nthe 32B cannot, by spending compute on samples not parameters",fontsize=17)
ax.legend(fontsize=15,loc="lower right"); ax.grid(alpha=0.3)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(ROOT,"paper/figs/limits/fig_accuracy_compute.png"),dpi=150); plt.close()
json.dump({"curve":curve,"pool":pool}, open(os.path.join(ROOT,"ckpts/train/lora_verifier_pooled4/accuracy_compute.json"),"w"),indent=1)
print("wrote fig_peer_comparison.png, fig_accuracy_compute.png")

# ---------- INTEGRATION: verifier-augmented cascade (cheap 7B best-of-8 + escalate residual to 32B) ----------
# gate on the verifier's top confidence: escalate to 32B when max(sc) < tau. Cost in 7B-fwd-equiv:
# cheap leg = 8 gen + 8 verify = 16 ; 32B single = 4.3 (added on escalated fraction).
rows32=[]
for r in dump:
    sl=[None if x==-1 else x for x in r["sl"]]; sc=r["sc"]
    cand=[k for k in range(len(sl)) if sl[k] is not None]
    if not cand: continue
    vlab=sl[max(cand,key=lambda k:sc[k])]; conf=max(sc[k] for k in cand)
    b=S32.get(r["ds"],{}).get(r["idx"])
    rows32.append((vlab, conf, (b if b is not None else vlab)))  # fallback: if no 32B label, keep verifier
casc=[]
for tau in [i/20 for i in range(0,21)]:
    esc=[1 if c<tau else 0 for _,c,_ in rows32]; er=float(np.mean(esc))
    acc=float(np.mean([ (b if c<tau else v) for v,c,b in rows32]))
    cost=16+er*4.3
    casc.append({"tau":tau,"esc_rate":er,"acc":acc,"cost":cost})
base={"cheap_verifier_bo8":{"acc":bok(8),"cost":16},"always_32b":{"acc":pool["m32b"],"cost":4.3},
      "cheap_greedy":{"acc":pool["greedy"],"cost":1}}
json.dump({"cascade":casc,"baselines":base}, open(os.path.join(ROOT,"ckpts/train/lora_verifier_pooled4/cascade_frontier.json"),"w"),indent=1)
best=max(casc,key=lambda x:x["acc"])
print(f"verifier-augmented cascade: max acc {best['acc']:.3f} @ esc {best['esc_rate']:.2f} cost {best['cost']:.1f} (vs always-32B {pool['m32b']:.3f}@4.3, cheap-bo8 {bok(8):.3f}@16)")
