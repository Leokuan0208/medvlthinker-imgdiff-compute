#!/usr/bin/env python3
"""
bidirectional_stability.py - test the BIDIRECTIONAL visual-stability gate premise.

The deployed margin gate keeps HIGH-margin samples cheap. But a high-margin sample can be
confidently WRONG because the 7B can't resolve fine visual detail -> its answer FLIPS across
resolution (visually fragile). If those fragile-but-confident samples are (a) more often wrong and
(b) recoverable by the 32B, then ESCALATING them (which the margin gate misses) ADDS accuracy.
Combined with the rescue-DOWN (keep stable low-margin cheap), this is a bidirectional gate that
could move accuracy UP and compute DOWN simultaneously.

This script tests the PREMISE only (no cost yet): among gate-KEEP-cheap samples, do visually
fragile ones have lower cheap-acc and higher 32B-fix-rate than stable ones? And does adding a
fragility-escalation arm raise the cascade's accuracy frontier? Offline from existing ckpts.
"""
import os, json, pickle
import numpy as np

PRUNE = "ckpts/gate_7b_prune"; FULLRES = "ckpts/gate_7b_vllm"; STRONG = "ckpts/gate_32b"
_R = pickle.load(open("ckpts/router_margin.pkl", "rb")); GATE, TAU = _R["gate"], _R["tau"]
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres"]

def load_jsonl(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def cap_file(cap, ds):
    return (os.path.join(FULLRES, f"ckpt_{ds}_nothink_norag.jsonl") if cap == "fullres"
            else os.path.join(PRUNE, cap, f"ckpt_{ds}_nothink_norag.jsonl"))
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

def build(ds):
    caps = {c: load_jsonl(cap_file(c, ds)) for c in CAPS}
    strong = load_jsonl(os.path.join(STRONG, f"ckpt_{ds}_think_norag.jsonl"))
    rows = []
    for i in sorted(caps["cap320"]):
        if i not in strong or any(i not in caps[c] for c in CAPS): continue
        p320 = caps["cap320"][i]["pred"]
        n_agree = sum(int(caps[c][i]["pred"] == p320) for c in CAPS if c != "cap320")  # 0..4
        rows.append(dict(ds=ds, idx=i, margin=margin(caps["cap320"][i]),
                         ok320=caps["cap320"][i]["ok"], ok32=strong[i]["ok"], n_agree=n_agree))
    return rows

ROWS = [r for ds in COMP4 for r in build(ds)]
mg = np.array([[r["margin"]] for r in ROWS], dtype=np.float32)
PROBA = GATE.predict_proba(mg)[:, 1]
ESC_DEP = PROBA < TAU
OK320 = np.array([r["ok320"] for r in ROWS]); OK32 = np.array([r["ok32"] for r in ROWS])
NAG = np.array([r["n_agree"] for r in ROWS])

def acc_of(group_mask):
    g = group_mask
    return float(OK320[g].mean()) if g.any() else float("nan")
def fixrate(group_mask):  # of the cheap-WRONG in group, fraction the 32B gets right
    w = group_mask & (OK320 == 0)
    return float(OK32[w].mean()) if w.any() else float("nan")

def main():
    keep = ~ESC_DEP   # gate keeps cheap (high margin)
    print("=" * 84)
    print("BIDIRECTIONAL premise: among gate-KEEP-cheap samples, fragile vs stable")
    print("=" * 84)
    print(f"  gate keeps cheap: {keep.sum()}/{len(ROWS)}")
    for thr, lab in [(4, "fully stable (n_agree=4)"), (3, "n_agree>=3"), (2, "n_agree>=2")]:
        stable = keep & (NAG >= thr); fragile = keep & (NAG < thr)
        print(f"\n  split at {lab}:  stable={stable.sum()}  fragile={fragile.sum()}")
        print(f"    cheap-acc   stable={acc_of(stable):.3f}   fragile={acc_of(fragile):.3f}   "
              f"(escalate fragile helps if fragile cheap-acc is low)")
        print(f"    32B-fix on cheap-wrong  stable={fixrate(stable):.3f}   fragile={fixrate(fragile):.3f}   "
              f"(escalate fragile helps if fix-rate high)")
        # net accuracy gain if we escalate the fragile-kept set
        f = fragile
        gain = float((OK32[f] - OK320[f]).sum()) / len(ROWS) if f.any() else 0.0
        print(f"    NET acc gain from escalating this fragile-kept set: {gain:+.4f}  "
              f"(over n={len(ROWS)}; escalates +{f.mean()*100:.1f}% of samples)")

    # ---- build bidirectional frontier vs margin frontier (in-sample, accuracy ceiling check) ----
    print("\n" + "=" * 84)
    print("FRONTIER: can bidirectional reach HIGHER accuracy than the margin gate's max?")
    print("=" * 84)
    parity = 0.6451
    # margin frontier: sweep tau -> (esc, acc); record max acc
    taus = np.linspace(0.27, 0.55, 57)
    best_m = max(((PROBA < t).mean(), np.where(PROBA < t, OK32, OK320).mean()) for t in taus)
    max_acc_m = max(np.where(PROBA < t, OK32, OK320).mean() for t in taus)
    print(f"  margin gate MAX achievable acc (any tau): {max_acc_m:.4f}")
    # bidirectional: escalate if (proba<t1) OR (n_agree<=k)  [adds fragility-escalation arm]
    best = (1.0, 0.0, None)
    for t in taus:
        for k in [0, 1, 2, 3]:
            esc = (PROBA < t) | (NAG <= k)
            acc = np.where(esc, OK32, OK320).mean()
            if acc > best[1] or (acc == best[1] and esc.mean() < best[0]):
                best = (esc.mean(), acc, (t, k))
    print(f"  bidirectional MAX achievable acc: {best[1]:.4f} at esc={best[0]:.3f} (tau={best[2][0]:.3f}, n_agree<= {best[2][1]})")
    print(f"  always-strong parity = {parity:.4f}   (does bidir exceed margin's ceiling? "
          f"{'YES +%.4f' % (best[1]-max_acc_m) if best[1] > max_acc_m else 'no'})")

    out = dict(max_acc_margin=float(max_acc_m), max_acc_bidir=float(best[1]),
               bidir_esc=float(best[0]), bidir_params=dict(tau=float(best[2][0]), k=int(best[2][1])))
    os.makedirs("results/cascade_methods", exist_ok=True)
    json.dump(out, open("results/cascade_methods/bidirectional_stability.json", "w"), indent=1)
    print("\n-> results/cascade_methods/bidirectional_stability.json")

if __name__ == "__main__":
    main()
