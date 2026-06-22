#!/usr/bin/env python3
"""
peer_router.py - does a CHEAP router actually capture the cross-family complementarity?

Loads MedVLThinker-7B (cap320) + cross-family peers (InternVL2.5-8B, Phi-3.5-V) + 32B (reference),
aligned by idx on competent-4. Evaluates realistic routers with HONEST held-out CV and FLOPs cost:
  - always-7B / always-peer / always-32B(parity) / ORACLE union (ceiling)
  - learned per-model P(correct) router from robust features {7B confidence + pairwise AGREEMENT
    flags + benchmark id} (NOT peer logprobs, which are noisy); route argmax. 2-leg & 3-leg.
  - interpretable router: if models AGREE -> that answer; if DISAGREE -> 7B unless its margin < tau
    (tau picked on the train split), then peer.
FLOPs per forward = 2*N*(P+G); peers ~no-think (G~2). Compares cascade FLOPs vs always-32B-think.
Run from repo root:  python3 src/cascade_methods/peer_router.py
"""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

COMP = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
N7, N_IV, N_PHI, N32 = 7.6e9, 8.0e9, 4.2e9, 33.0e9   # param counts (approx) for FLOPs
def load_arm(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"ckpt_*{tag}*.jsonl")):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out

c7 = load_arm("ckpts/gate_7b_prune/cap320", "nothink_norag")
c32 = load_arm("ckpts/gate_32b", "think_norag")
iv = load_arm("ckpts/peer/internvl25_8b", "internvl25_8b")
phi = load_arm("ckpts/peer/phi35v", "phi35v")

rows = []
for ds in COMP:
    idx = sorted(set(c7[ds]) & set(c32[ds]) & set(iv[ds]) & set(phi[ds]))
    for i in idx:
        s = signals_from_logprobs(c7[ds][i].get("opt_logprobs"))
        rows.append(dict(ds=ds, ok7=c7[ds][i]["ok"], okiv=iv[ds][i]["ok"], okphi=phi[ds][i]["ok"], ok32=c32[ds][i]["ok"],
                         p7=c7[ds][i]["pred"], piv=iv[ds][i]["pred"], pphi=phi[ds][i]["pred"],
                         m=s["margin"], e=s["entropy"], t=s["top1prob"], g=s["gini"], pm=s["prob_margin"],
                         P7=float(c7[ds][i].get("gen_tokens") or 2)))
A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
ok7, okiv, okphi, ok32 = [A[k].astype(int) for k in ["ok7", "okiv", "okphi", "ok32"]]
ds = A["ds"]; n = len(rows)
ag_7iv = (A["p7"] == A["piv"]).astype(float); ag_7phi = (A["p7"] == A["pphi"]).astype(float); ag_ivphi = (A["piv"] == A["pphi"]).astype(float)
oneh = np.column_stack([(ds == d).astype(float) for d in COMP])
CONF = np.column_stack([A["m"], A["e"], A["t"], A["g"], A["pm"]]).astype(float)

print(f"n={n}  baselines: 7B={ok7.mean():.3f} InternVL={okiv.mean():.3f} Phi={okphi.mean():.3f} 32B(parity)={ok32.mean():.3f}")
print(f"oracle unions: 7B|IV={np.maximum(ok7,okiv).mean():.3f}  7B|IV|Phi={np.maximum.reduce([ok7,okiv,okphi]).mean():.3f}")

def route_eval(models_ok, feats, name, fwd_flops):
    """Learned per-model P(correct) router via 5-fold OOF; route to argmax predicted-correct."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    strat = ok7 + 2 * okiv
    P = np.zeros((n, len(models_ok)))
    for tr, te in skf.split(feats, strat):
        for j, y in enumerate(models_ok):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            if y[tr].sum() in (0, len(tr)):
                P[te, j] = y[tr].mean(); continue
            clf.fit(feats[tr], y[tr]); P[te, j] = clf.predict_proba(feats[te])[:, 1]
    pick = P.argmax(1)
    routed = np.choose(pick, models_ok)
    # FLOPs: every routed query runs ALL legs in this set (run-then-pick); cost-matched to no-think
    flops = sum(fwd_flops) * n
    flops32 = 2 * N32 * (685 + 391) * n   # always-32B-think
    sw = (pick != 0).mean()
    print(f"  {name:<34} acc={routed.mean():.3f}  routed-away-from-7B={sw*100:.0f}%  FLOPs={flops/flops32*100:.0f}% of always-32B-think")
    return routed.mean()

# feature set: robust (7B confidence + agreement flags + benchmark)
F2 = np.column_stack([CONF, ag_7iv, oneh])                       # 2-leg: 7B + InternVL
F3 = np.column_stack([CONF, ag_7iv, ag_7phi, ag_ivphi, oneh])    # 3-leg: 7B + InternVL + Phi
f7 = 2 * N7 * (388 + 2); fiv = 2 * N_IV * (388 + 2); fphi = 2 * N_PHI * (388 + 2)
print(f"\n--- learned per-model router (robust features; honest 5-fold OOF) ---")
print(f"  always-7B (no routing)             acc={ok7.mean():.3f}")
route_eval([ok7, okiv], F2, "2-leg router {7B, InternVL}", [f7, fiv])
route_eval([ok7, okiv, okphi], F3, "3-leg router {7B, InternVL, Phi}", [f7, fiv, fphi])

# interpretable: agree->trust; disagree->7B unless margin<tau then peer (tau on train split)
def interp_router(okA, okB, pA, pB, agree, name, fwd):
    skf = StratifiedKFold(5, shuffle=True, random_state=1); out = np.zeros(n)
    for tr, te in skf.split(CONF, ok7):
        best_t, best_a = 0.0, -1
        for t in np.quantile(A["m"][tr], np.linspace(0, 1, 40)):
            # train acc of this rule
            use_peer = (agree[tr] == 0) & (A["m"][tr] < t)
            acc = np.where(use_peer, okB[tr], okA[tr]).mean()
            if acc > best_a: best_a, best_t = acc, t
        use_peer = (agree[te] == 0) & (A["m"][te] < best_t)
        out[te] = np.where(use_peer, okB[te], okA[te])
    flops32 = 2 * N32 * (685 + 391) * n
    print(f"  {name:<34} acc={out.mean():.3f}  FLOPs={sum(fwd)*n/flops32*100:.0f}% of always-32B-think")
print(f"\n--- interpretable router (agree->7B/peer same; disagree->7B if confident else peer) ---")
interp_router(ok7, okiv, A["p7"], A["piv"], ag_7iv, "agree/conf {7B, InternVL}", [f7, fiv])
print(f"\nReference: always-32B-think acc={ok32.mean():.3f} at 100% FLOPs. Router cost is run-then-pick (all legs).")
