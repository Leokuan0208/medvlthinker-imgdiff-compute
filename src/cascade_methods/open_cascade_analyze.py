#!/usr/bin/env python3
"""
open_cascade_analyze.py - OPEN-ENDED medical VQA cascade: is semantic SELF-CONSISTENCY a better
routing signal than confidence here, where the MCQ degeneracy (that capped everything at ~0.6 AUROC)
is removed? Reads ckpts/openvqa/{cheap,strong}. Compares routing signals for the 7B->32B cascade:
  confidence  : 7B temp-0 sequence logprob (the MCQ-era best signal; Chow/Jitkrittum-style deferral)
  self-consist: 7B agreement across K=8 temp-0.7 samples (Wang et al. ICLR'23 signal, degenerate in MCQ)
  n_distinct  : # distinct answers among the K samples (inverse consistency)
Reports, per dataset + pooled:
  (1) AUROC of each signal for predicting CHEAP-WRONG and RECOVERABLE (7B wrong & 32B right);
  (2) honest accuracy-vs-escalation frontier for the confidence gate vs the self-consistency gate,
      with the self-consistency cheap leg = majority vote over the K samples (Wang self-consistency).
Scoring is the runner's normalized match. Validated baseline = confidence deferral (the gate the
project's MCQ work and Jitkrittum NeurIPS'23 establish as the one to beat).
"""
import os, json, glob, re, string
import numpy as np
def _norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation)); return re.sub(r"\s+", " ", s).strip()
def sc_at_K(preds, K):
    from collections import Counter
    c = Counter(_norm(p) for p in preds[:K]); top = c.most_common(1)[0][1]; return top / K, len(c)
def token_f1(pred, gold):  # partial-credit scorer (rigor check vs exact-match)
    p, g = set(_norm(pred).split()), set(_norm(gold).split())
    if not p or not g: return 0.0
    ov = len(p & g)
    if ov == 0: return 0.0
    prec, rec = ov/len(p), ov/len(g); return 2*prec*rec/(prec+rec)
import sys as _s
# pathvqa_open/kvasir_open have long answers -> judge-only (exact-match uninformative); opt in via flags
DSETS = ["slake_open", "vqa_rad_open"] + (["pathvqa_open"] if "--pathvqa" in _s.argv else []) \
        + (["kvasir_open"] if "--kvasir" in _s.argv else [])
if "--lingshu" in _s.argv:                          # cross-family strong leg = Lingshu-32B
    STRONG = "ckpts/openvqa/strong_lingshu"; STRONG_TAG = "lingshu32b"
else:
    STRONG = "ckpts/openvqa/strong"; STRONG_TAG = "32b_think" if ("--think" in _s.argv) else "32b_t0"
if "--cheap_l7" in _s.argv:                          # same-family: Lingshu-7B cheap leg
    CHEAP = "ckpts/openvqa/cheap_lingshu7b"; CTAG = "lingshu7b"; T0TAG = "lingshu7b"; SC8TAG = "lingshu7b_sc8"
elif "--cheap3b" in _s.argv:
    CHEAP = "ckpts/openvqa/cheap3b"; CTAG = "3b"; T0TAG = "3b_t0"; SC8TAG = "3b_sc8"
else:
    CHEAP = "ckpts/openvqa/cheap"; CTAG = "7b"; T0TAG = "7b_t0"; SC8TAG = "7b_sc8"
def load(p):
    m = {}
    if not os.path.exists(p): return m
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True); s = np.zeros(len(cnt)); np.add.at(s, inv, ranks); ranks = (s/cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

JUDGE = "--judge" in _s.argv
def apply_judge(d, jpath):
    """override modal_ok with LLM-judge_ok (robustness vs exact-match)."""
    if not os.path.exists(jpath): return d
    j = {r["idx"]: r["judge_ok"] for r in (json.loads(l) for l in open(jpath) if l.strip())}
    for i, r in d.items():
        if i in j: r["modal_ok"] = j[i]
    return d
def build(ds):
    t0 = load(f"{CHEAP}/ckpt_{ds}_{T0TAG}.jsonl"); sc = load(f"{CHEAP}/ckpt_{ds}_{SC8TAG}.jsonl"); st = load(f"{STRONG}/ckpt_{ds}_{STRONG_TAG}.jsonl")
    if JUDGE:
        t0 = apply_judge(t0, f"{CHEAP}/ckpt_{ds}_{T0TAG}.judge.jsonl")
        st = apply_judge(st, f"{STRONG}/ckpt_{ds}_{STRONG_TAG}.judge.jsonl")
    idx = sorted(set(t0) & set(sc) & set(st))
    rows = []
    for i in idx:
        rows.append(dict(ds=ds, idx=i,
            cheap_ok=t0[i]["modal_ok"],                       # 7B temp-0 answer correctness
            sc_ok=sc[i]["modal_ok"],                          # 7B self-consistency majority answer correctness
            strong_ok=st[i]["modal_ok"],
            conf=(t0[i].get("seqlogprob") if t0[i].get("seqlogprob") is not None else 0.0),
            selfcons=sc[i]["self_consistency"], ndist=sc[i]["n_distinct"], preds=sc[i].get("preds", []),
            cheap_gen=(t0[i].get("gen_tokens") or 8), strong_gen=(st[i].get("gen_tokens") or 0),
            cheap_f1=token_f1(t0[i].get("modal_pred",""), t0[i]["gold"]),
            strong_f1=token_f1(st[i].get("modal_pred",""), st[i]["gold"])))
    return rows

# --- cost model (FLOPs = 2N(P+G); P fixed ~ cap320 prompt; latency: SC samples decode in PARALLEL ~ 1 pass) ---
N0, N1, PFIX = (3.0e9 if CTAG == "3b" else 7.6e9), 33e9, 420.0
def cost_frontier(rows, gate_score, cheap_ok_key, K):
    """Return list of (flops, latency, acc) over escalation thresholds. K = #cheap samples for the gate."""
    g = np.asarray(gate_score, float)
    ok_cheap = np.array([r[cheap_ok_key] for r in rows]); ok_strong = np.array([r["strong_ok"] for r in rows])
    cflop = np.array([2*N0*(PFIX+r["cheap_gen"]) for r in rows]); sflop = np.array([2*N1*(PFIX+r["strong_gen"]) for r in rows])
    # latency proxy (s): cheap ~ 0.15 + 0.01*gen ; strong-think ~ 0.4 + 0.03*gen (decode-bound). SC cheap ~ 1 pass (parallel).
    clat = np.array([0.15+0.01*r["cheap_gen"] for r in rows]); slat = np.array([0.4+0.03*r["strong_gen"] for r in rows])
    base_flop = K*cflop                                   # K cheap passes always (gate cost)
    pts = []
    for t in np.quantile(g, np.linspace(0, 1, 51)):
        esc = g >= t
        flops = (base_flop + np.where(esc, sflop, 0)).sum()
        lat = (clat + np.where(esc, slat, 0)).mean()       # SC cheap latency ~ 1 pass (parallel sampling)
        acc = np.where(esc, ok_strong, ok_cheap).mean()
        pts.append((float(flops), float(lat), float(esc.mean()), float(acc)))
    return pts
def min_cost_at(pts, target, idx):
    ok = [p for p in pts if p[3] >= target-1e-9]; return min(ok, default=None, key=lambda p: p[idx])
def kcurve(rows):
    """AUROC of self-consistency at K samples for predicting cheap-wrong (cost scales with K)."""
    cw = [1-r["cheap_ok"] for r in rows]
    print("  self-consistency AUROC (cheap-wrong) vs #samples K:  ", end="")
    for K in [2, 3, 5, 8]:
        if all(len(r["preds"]) >= K for r in rows):
            scK = [-sc_at_K(r["preds"], K)[0] for r in rows]
            print(f"K={K}:{auroc(scK, cw):.3f}  ", end="")
    print()

def report(rows, label):
    if not rows: print(f"[{label}] no data yet"); return
    cw = [1-r["cheap_ok"] for r in rows]; rec = [int(r["cheap_ok"]==0 and r["strong_ok"]==1) for r in rows]
    # escalate-scores: higher => more likely wrong
    s_conf = [-r["conf"] for r in rows]           # low logprob => escalate
    s_sc   = [-r["selfcons"] for r in rows]        # low consistency => escalate
    s_nd   = [r["ndist"] for r in rows]            # many distinct => escalate
    print(f"\n[{label}] n={len(rows)}  7B-acc={np.mean([r['cheap_ok'] for r in rows]):.3f}  "
          f"SC-acc={np.mean([r['sc_ok'] for r in rows]):.3f}  32B-acc={np.mean([r['strong_ok'] for r in rows]):.3f}  "
          f"cheap-wrong={np.mean(cw):.3f}  recoverable={np.mean(rec):.3f}")
    print(f"  AUROC predict CHEAP-WRONG :  confidence={auroc(s_conf,cw):.3f}   self-consistency={auroc(s_sc,cw):.3f}   n_distinct={auroc(s_nd,cw):.3f}")
    print(f"  AUROC predict RECOVERABLE :  confidence={auroc(s_conf,rec):.3f}   self-consistency={auroc(s_sc,rec):.3f}   n_distinct={auroc(s_nd,rec):.3f}")
    cf1 = np.mean([r["cheap_f1"] for r in rows]); sf1 = np.mean([r["strong_f1"] for r in rows])
    print(f"  TOKEN-F1 (partial credit, rigor check):  cheap={cf1:.3f}  strong={sf1:.3f}  gap={sf1-cf1:+.3f}  "
          f"(vs exact-match gap {np.mean([r['strong_ok'] for r in rows])-np.mean([r['cheap_ok'] for r in rows]):+.3f})")
    if any(r.get("preds") for r in rows): kcurve(rows)
    return dict(label=label, n=len(rows), cheap_acc=float(np.mean([r['cheap_ok'] for r in rows])),
                sc_acc=float(np.mean([r['sc_ok'] for r in rows])), strong_acc=float(np.mean([r['strong_ok'] for r in rows])),
                auroc_cw=dict(conf=auroc(s_conf,cw), selfcons=auroc(s_sc,cw), ndist=auroc(s_nd,cw)),
                auroc_rec=dict(conf=auroc(s_conf,rec), selfcons=auroc(s_sc,rec), ndist=auroc(s_nd,rec)))

def frontier(rows, gate, cheap_ok_key):
    """escalate by `gate` score (higher=escalate); cheap leg correctness = cheap_ok_key. Return (esc,acc) sweep."""
    g = np.array(gate); ok_cheap = np.array([r[cheap_ok_key] for r in rows]); ok_strong = np.array([r["strong_ok"] for r in rows])
    pts = []
    for t in np.quantile(g, np.linspace(0, 1, 51)):
        esc = g >= t
        acc = np.where(esc, ok_strong, ok_cheap).mean()
        pts.append((float(esc.mean()), float(acc)))
    return pts
def min_esc_at(pts, target):
    ok = [(e, a) for (e, a) in pts if a >= target-1e-9]; return min(ok, default=(None, None), key=lambda x: x[0])

def main():
    REC = {ds: build(ds) for ds in DSETS}
    allrows = [r for ds in DSETS for r in REC[ds]]
    print("="*92); print("OPEN-ENDED MEDICAL VQA CASCADE — self-consistency vs confidence routing"); print("="*92)
    OUT = {}
    for ds in DSETS:
        OUT[ds] = report(REC[ds], ds)
    OUT["POOLED"] = report(allrows, "POOLED (all open)")
    # frontier comparison (pooled): confidence gate (cheap=t0) vs self-consistency gate (cheap=SC majority)
    if allrows:
        strong = np.mean([r["strong_ok"] for r in allrows])
        f_conf = frontier(allrows, [-r["conf"] for r in allrows], "cheap_ok")
        f_sc   = frontier(allrows, [-r["selfcons"] for r in allrows], "sc_ok")
        print("\nFRONTIER (pooled): min escalation to reach a target accuracy (lower=better routing)")
        for tname, T in [("7B-acc+2pt", np.mean([r['cheap_ok'] for r in allrows])+0.02),
                         ("midpoint", (np.mean([r['cheap_ok'] for r in allrows])+strong)/2),
                         ("strong-2pt", strong-0.02)]:
            ec, _ = min_esc_at(f_conf, T); es, _ = min_esc_at(f_sc, T)
            sc_s = f"{es*100:.0f}%" if es is not None else "—"; cf_s = f"{ec*100:.0f}%" if ec is not None else "—"
            print(f"  @acc>={T:.3f} ({tname:<12}):  confidence-gate esc={cf_s:<6}  self-consistency-gate esc={sc_s}")
        OUT["frontier"] = dict(confidence=f_conf, self_consistency=f_sc, strong_acc=float(strong))
        # cost-aware comparison (with the expensive strong leg): FLOPs and latency at matched accuracy
        cheap = np.mean([r['cheap_ok'] for r in allrows])
        cf = cost_frontier(allrows, [-r["conf"] for r in allrows], "cheap_ok", K=1)        # confidence gate, 1 cheap pass
        cs = cost_frontier(allrows, [-r["selfcons"] for r in allrows], "sc_ok", K=8)       # self-consistency gate, K=8
        print(f"\nCOST-AWARE FRONTIER (strong leg = 32B-think). cheap={cheap:.3f} strong={strong:.3f}.")
        print(f"  min LATENCY / min FLOPs at matched accuracy (self-consistency cheap leg = K=8 parallel samples):")
        for tname, T in [("midpoint", (cheap+strong)/2), ("strong-3pt", strong-0.03), ("strong-1pt", strong-0.01)]:
            cl = min_cost_at(cf, T, 1); sl = min_cost_at(cs, T, 1); cflo = min_cost_at(cf, T, 0); sflo = min_cost_at(cs, T, 0)
            def fmt(p, i, unit): return (f"{p[i]:.2f}{unit}" if (i==1 and p) else (f"{p[i]/1e12:.0f}TF" if p else "—")) if p else "—"
            print(f"  @acc>={T:.3f} ({tname:<10}): latency conf={fmt(cl,1,'s')} vs SC={fmt(sl,1,'s')}   |   "
                  f"FLOPs conf={fmt(cflo,0,'')} vs SC={fmt(sflo,0,'')}   "
                  f"(esc conf={cl[2]*100:.0f}% vs SC={sl[2]*100:.0f}%)" if (cl and sl) else f"  @acc>={T:.3f}: unreachable")
        OUT["cost_frontier"] = dict(confidence_K1=cf, selfcons_K8=cs)
    os.makedirs("results/cascade_methods/artifacts", exist_ok=True)
    json.dump(OUT, open("results/cascade_methods/artifacts/open_cascade.json", "w"), indent=1)
    print("\n-> results/cascade_methods/artifacts/open_cascade.json")
    print("\nREAD: if self-consistency AUROC >> confidence AND >> the ~0.6 MCQ ceiling, the open-ended setting")
    print("breaks the routing wall -> a genuinely better cascade gate (the publishable novel claim).")
if __name__ == "__main__":
    main()
