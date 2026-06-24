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
import os, json, glob
import numpy as np
CHEAP = "ckpts/openvqa/cheap"; STRONG = "ckpts/openvqa/strong"
# pathvqa_open excluded: long descriptive answers unscoreable by exact-match (7B-nt acc 0.058)
DSETS = ["slake_open", "vqa_rad_open"]
import sys as _s
STRONG_TAG = "32b_think" if ("--think" in _s.argv) else "32b_t0"
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

def build(ds):
    t0 = load(f"{CHEAP}/ckpt_{ds}_7b_t0.jsonl"); sc = load(f"{CHEAP}/ckpt_{ds}_7b_sc8.jsonl"); st = load(f"{STRONG}/ckpt_{ds}_{STRONG_TAG}.jsonl")
    idx = sorted(set(t0) & set(sc) & set(st))
    rows = []
    for i in idx:
        rows.append(dict(ds=ds, idx=i,
            cheap_ok=t0[i]["modal_ok"],                       # 7B temp-0 answer correctness
            sc_ok=sc[i]["modal_ok"],                          # 7B self-consistency majority answer correctness
            strong_ok=st[i]["modal_ok"],
            conf=(t0[i].get("seqlogprob") if t0[i].get("seqlogprob") is not None else 0.0),
            selfcons=sc[i]["self_consistency"], ndist=sc[i]["n_distinct"]))
    return rows

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
    os.makedirs("results/cascade_methods", exist_ok=True)
    json.dump(OUT, open("results/cascade_methods/open_cascade.json", "w"), indent=1)
    print("\n-> results/cascade_methods/open_cascade.json")
    print("\nREAD: if self-consistency AUROC >> confidence AND >> the ~0.6 MCQ ceiling, the open-ended setting")
    print("breaks the routing wall -> a genuinely better cascade gate (the publishable novel claim).")
if __name__ == "__main__":
    main()
