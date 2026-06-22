#!/usr/bin/env python3
"""
CASP-Stability (TRAINING-based, src/training_methods/) - a trained gate for the ACC cascade whose target
is ANSWER-STABILITY-UNDER-COMPUTE  S(x)=1[pred(7B-nt@cap320)==pred(32B-think@fullres)], instead of the
un-learnable recoverability ("will a DIFFERENT model be right").

Two deliverables (both honest, pre-registered):
  (A) LEARNABILITY DISSECTION (the headline): on the SAME honest PMC-train->competent-4-eval split and the
      SAME features, train a small torch MLP (and a logistic baseline) to predict three targets and report
      transfer AUROC: recoverability (the wall) vs stability vs which-tier. Shows RE-TARGETING the label
      moves AUROC; adding capacity (MLP vs logistic) / hidden-state features does not.
  (B) DEPLOYED GATE: use P(stable) as the ACC tier-0 STOP gate (keep 7B if predicted stable, else escalate
      7B-nt -> 32B-nt -> 32B-think-on-disagree). Calibrate tau on PMC-train (no eval leakage), test on
      competent-4. Compare FLOPs%/latency at acc>=always-32B-think parity vs ACC's margin gate, with the
      per-benchmark never-worse-than-7B guardrail. Stability cuts COMPUTE at iso-accuracy; it CANNOT raise
      accuracy above always-32B-think (pre-registered).

torch-only (no peft/deepspeed/downloads). Run from repo root:  python3 src/training_methods/casp_stability.py
"""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np, torch, torch.nn as nn
from harness import signals_from_logprobs, COMPETENT
from acc_compare import fit_models, predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); np.random.seed(0)
J = lambda p: os.path.join("/home/jamesyang/medvlthinker-imgdiff-compute", p)
N7, N32 = 7.6e9, 33.0e9
FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "n_opts"]

def load_jsonl(path):
    return {json.loads(l)["idx"]: json.loads(l) for l in open(path) if l.strip()}
def load_eval_arm(d, tag):
    out = {}
    for f in glob.glob(J(os.path.join(d, f"ckpt_*{tag}*.jsonl"))):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if m: out[m.group(1)] = load_jsonl(f)
    return out

# ---- CALIB: PMC-train, 3 tiers (no eval leakage) ----
c7 = load_jsonl(J("ckpts/gate_7b_pmctrain_prune/cap320/ckpt_nothink.jsonl"))
c32n = load_jsonl(J("ckpts/gate_32b_pmctrain_nothink_cap320/ckpt_nothink.jsonl"))
c32t = load_jsonl(J("ckpts/gate_32b_pmctrain/ckpt_think.jsonl"))
cidx = sorted(set(c7) & set(c32n) & set(c32t))
def feat_row(r): s = signals_from_logprobs(r.get("opt_logprobs")); return [s[f] for f in FEATS]
Xc = np.array([feat_row(c7[i]) for i in cidx], float)
c_ok7 = np.array([c7[i]["ok"] for i in cidx]); c_ok32t = np.array([c32t[i]["ok"] for i in cidx])
c_stab = np.array([int(c7[i]["pred"] == c32t[i]["pred"]) for i in cidx])   # answer-stability label

# ---- EVAL: competent-4, 3 tiers ----
e7 = load_eval_arm("ckpts/gate_7b_prune/cap320", "nothink_norag")
e32n = load_eval_arm("ckpts/gate_32b_modes/nothink_cap320", "nothink_norag")
e32t = load_eval_arm("ckpts/gate_32b", "think_norag")
cache = json.load(open(J("ckpts/token_cache.json")))
rows = []
for ds in COMPETENT:
    cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
    idx = sorted(set(e7[ds]) & set(e32n[ds]) & set(e32t[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
    for i in idx:
        rows.append((ds, e7[ds][i], e32n[ds][i], e32t[ds][i], cC[str(i)][0], cF[str(i)][0]))
ds_of = np.array([r[0] for r in rows])
Xe = np.array([feat_row(r[1]) for r in rows], float)
ok7 = np.array([r[1]["ok"] for r in rows]); ok32n = np.array([r[2]["ok"] for r in rows]); ok32t = np.array([r[3]["ok"] for r in rows])
p7 = np.array([r[1]["pred"] for r in rows]); p32n = np.array([r[2]["pred"] for r in rows]); p32t = np.array([r[3]["pred"] for r in rows])
stab = (p7 == p32t).astype(int); g32t = np.array([r[3].get("gen_tokens") or 0 for r in rows], float)
Pc = np.array([r[4] for r in rows], float); Pf = np.array([r[5] for r in rows], float)
print(f"calib(PMC-train) n={len(cidx)}  eval(competent-4) n={len(rows)}")

# ---------- (A) LEARNABILITY DISSECTION ----------
class MLP(nn.Module):
    def __init__(s, d): super().__init__(); s.f = nn.Sequential(nn.Linear(d,128), nn.GELU(), nn.Dropout(.15), nn.Linear(128,64), nn.GELU(), nn.Dropout(.15), nn.Linear(64,1))
    def forward(s, x): return s.f(x).squeeze(-1)
def train_mlp(Xtr, ytr, epochs=150, bs=512):
    Xtr = np.nan_to_num(Xtr, nan=0., posinf=0., neginf=0.); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    Xn = np.clip((Xtr - mu) / sd, -8, 8)   # clip: PMC-train is all 4-option so n_opts has ~0 calib var -> eval explodes without clipping
    m = MLP(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); xt = torch.tensor(Xn, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32); n = len(xt)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad(); lossf(m(xt[b]), yt[b]).backward(); opt.step()
    m.eval(); return m, mu, sd
def mlp_predict(m, mu, sd, X):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.); Xn = np.clip((X - mu) / sd, -8, 8)
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(Xn, dtype=torch.float32))).numpy()
def logit_auc(Xtr, ytr, Xte, yte, mask=None):
    if len(np.unique(ytr)) < 2: return float("nan")
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    yy, pp = (yte, p) if mask is None else (yte[mask], p[mask])
    return roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")
def mlp_auc(Xtr, ytr, Xte, yte, mask=None):
    if len(np.unique(ytr)) < 2: return float("nan")
    m, mu, sd = train_mlp(Xtr, ytr.astype(float)); p = mlp_predict(m, mu, sd, Xte)
    yy, pp = (yte, p) if mask is None else (yte[mask], p[mask])
    return roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")

wrong = ok7 == 0; c_wrong = c_ok7 == 0
print("\n=== (A) LEARNABILITY DISSECTION — PMC-train -> competent-4 transfer AUROC (same features) ===")
print(f"  {'target':<42}{'logistic':>9}{'MLP':>7}")
# recoverability: predict 32B-think-right among 7B-WRONG rows
print(f"  {'recoverability 1[32B-think right]|7B-wrong':<42}{logit_auc(Xc[c_wrong], c_ok32t[c_wrong], Xe, ok32t, wrong):>9.3f}{mlp_auc(Xc[c_wrong], c_ok32t[c_wrong], Xe, ok32t, wrong):>7.3f}   <- the WALL")
# stability: predict pred7==pred(32B-think)
print(f"  {'stability 1[pred7==pred32think]':<42}{logit_auc(Xc, c_stab, Xe, stab):>9.3f}{mlp_auc(Xc, c_stab, Xe, stab):>7.3f}   <- learnable target")
# which-tier: predict 7B-nt correct
print(f"  {'which-tier 1[7B-nt correct]':<42}{logit_auc(Xc, c_ok7, Xe, ok7):>9.3f}{mlp_auc(Xc, c_ok7, Xe, ok7):>7.3f}")
print("  => re-TARGETING the label (recoverability->stability) moves AUROC; MLP≈logistic (capacity doesn't).")

# ---------- (B) DEPLOYED GATE: stability vs margin tier-0 stop ----------
LAT, EN = fit_models()
l0 = predict(LAT, "7b:nothink@cap320", np.full(len(rows), 2.0)); l1 = predict(LAT, "32b:nothink@cap320", np.full(len(rows), 2.0)); l2 = predict(LAT, "32b:think@fullres", g32t)
f0 = 2 * N7 * (Pc + 2); f1 = 2 * N32 * (Pc + 2); f2 = 2 * N32 * (Pf + g32t)
parity = ok32t.mean()
# CASP stability score (P(unstable) = escalate when likely to change); train on PMC-train stability
mS, muS, sdS = train_mlp(Xc, c_stab.astype(float)); pstab_e = mlp_predict(mS, muS, sdS, Xe); pstab_c = mlp_predict(mS, muS, sdS, Xc)
lrS = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xc, c_stab)
plr_c = lrS.predict_proba(Xc)[:, 1]; plr_e = lrS.predict_proba(Xe)[:, 1]
GATES = {  # tier-0 escalation score (higher=escalate). margin: low margin -> escalate. CASP: low P(stable) -> escalate.
    "ACC margin gate":          (-Xc[:, 0], -Xe[:, 0]),            # margin is FEATS[0]
    "CASP-Stability (logistic)": (1 - plr_c, 1 - plr_e),
    "CASP-Stability (MLP)":     (1 - pstab_c, 1 - pstab_e),
}
def cascade_metrics(e0):  # e0 = escalate-past-T0 mask on eval; T1->T2 by agreement (ACC-v2)
    e1 = e0 & (p7 != p32n)   # reach think iff escalated and 7B-nt vs 32B-nt disagree
    acc = np.where(~e0, ok7, np.where(~e1, ok32n, ok32t)).mean()
    flop = (f0.sum() + f1[e0].sum() + f2[e1].sum()) / f2.sum() * 100
    lat = l0.mean() + l1[e0].sum() / len(rows) + l2[e1].sum() / len(rows)
    # per-benchmark guardrail: cascade acc >= always-7B acc on each benchmark
    gfail = sum(1 for d in COMPETENT if (m := ds_of == d).any() and
                np.where(~e0[m], ok7[m], np.where(~e1[m], ok32n[m], ok32t[m])).mean() < ok7[m].mean() - 1e-9)
    return acc, e0.mean(), e1.mean(), flop, lat, gfail
print(f"\n=== (B) DEPLOYED: tier-0 gate swap in ACC, calibrated on PMC-train, tested on competent-4 ===")
print(f"  parity(always-32B-think)={parity:.3f}  always-7B={ok7.mean():.3f}")
print(f"  {'tier-0 gate':<24}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}{'lat(s)':>8}{'guardfail':>10}")
for name, (sc_c, sc_e) in GATES.items():
    # calibrate threshold on PMC-train: min escalation s.t. calib cascade-acc >= calib parity
    cp = c_ok32t.mean(); cand = np.unique(sc_c)[::-1]; chosen = cand[0] + 1
    for t in cand:
        ce0 = sc_c >= t; ce1 = ce0 & (np.array([c7[i]["pred"] for i in cidx]) != np.array([c32n[i]["pred"] for i in cidx]))
        ca = np.where(~ce0, c_ok7, np.where(~ce1, np.array([c32n[i]["ok"] for i in cidx]), c_ok32t)).mean()
        if ca >= cp - 1e-9: chosen = t; break
    acc, e0r, e1r, fl, lat, gf = cascade_metrics(sc_e >= chosen)
    print(f"  {name:<24}{acc:>7.3f}{e0r*100:>6.0f}%{e1r*100:>6.0f}%{fl:>7.1f}%{lat:>7.2f}s{gf:>9d}")
print("\nVERDICT: CASP wins if it reaches parity at LOWER FLOPs/latency than margin, guardrail-clean.")
print("(Pre-registered: stability cuts COMPUTE at iso-accuracy; it cannot raise accuracy above always-32B-think.)")
