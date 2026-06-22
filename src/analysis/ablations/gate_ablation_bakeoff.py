#!/usr/bin/env python3
"""
fbe_and_signals.py -- Offline router bake-off on saved cascade data. No inference.

PART A (7B only): per-benchmark 7B accuracy, margin diagnostics, temperature fit,
  frozen-gate escalation provenance, and the TRUE live-run escalation rate.
PART B (uses 32B answers): faithful CP-Router/FBE + plain conformal + single-signal
  gates + temp-scaled margin, each compared to the margin gate AT THE SAME escalation
  count (frontier-matched), with bootstrap CIs of the accuracy difference.

Every method is a function of the 7B option distribution + the 32B answer; none sees
the 32B before deciding. Per Jitkrittum (2023) a tie is expected; we run it to make
the CP-Router comparison faithful and to show the margin gate is Pareto-best.
"""
import glob, json, math, os, pickle
import numpy as np

# ---- paths / config -------------------------------------------------------
CAL_PATH    = "ckpts/gate_7b_pmctrain/ckpt_nothink.jsonl"                    # 7B no-think, PMC-VQA TRAIN (calibration)
CHEAP_GLOB  = "ckpts/gate_7b_prune/cap320/ckpt_*_nothink_norag.jsonl"  # deployed cheap leg (cap320)
STRONG_GLOB = "ckpts/gate_32b/ckpt_*_think_norag.jsonl"                # 32B, full coverage
LIVE_OUT    = "ckpts/rt_cascade_cap320.jsonl"                                      # rt_cascade.py output (escalate/ok/final)
GATE_PKL    = "ckpts/router_margin.pkl"
BUDGET   = 0.631        # deployed escalation rate
TAU_SET  = 1            # CP-Router: route cheap iff |prediction set| <= TAU_SET
N_BOOT   = 5000
COMPETENT = {"SLAKE", "PMC-VQA", "PathVQA", "VQA-RAD"}
# ---------------------------------------------------------------------------

def norm(s):                       # "PMC-VQA" / "pmc_vqa" -> "pmcvqa"
    return "".join(ch for ch in str(s).lower() if ch.isalnum())
COMP_NORM = {norm(c) for c in COMPETENT}

def parse_probs(d):
    letters = list(d.keys())
    lp = np.array([float(d[k]) for k in letters])
    p = np.exp(lp - lp.max()); p = p / p.sum()
    return dict(zip(letters, p))

def bench_of(path):
    return os.path.basename(path).split("_")[1]      # ckpt_<BENCH>_...

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def signals(pd_):
    pr = sorted(pd_.values(), reverse=True)
    p1 = pr[0]; p2 = pr[1] if len(pr) > 1 else 0.0
    pv = np.array(list(pd_.values()))
    ent = float(-np.sum(pv[pv > 0] * np.log(pv[pv > 0])))
    return {"margin": p1 - p2, "top1": p1, "entropy": ent}

def escalate_at_budget(u, budget):
    return u >= np.quantile(u, 1.0 - budget)

# ---- conformal (LAC score) ------------------------------------------------
def conf_qhat(cal_scores, alpha):
    n = len(cal_scores); k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else np.sort(cal_scores)[k - 1]

def set_size(pd_, qhat):
    thr = -np.inf if qhat == np.inf else (1 - qhat)
    return int(np.sum(np.array(list(pd_.values())) >= thr))

def fbe(sizes, beta=3.0):
    _, c = np.unique(sizes, return_counts=True); pi = c / c.sum()
    Hf = -sum(p * math.log(p) for p in pi if p > 0)
    p1 = float(np.mean(sizes == 1)); pne = 1 - p1
    xlx = lambda x: x * math.log(x) if x > 0 else 0.0
    return beta * Hf + (-(xlx(p1) + xlx(pne)))

# ---- temperature scaling (1 scalar, NLL on calibration) -------------------
def fit_T(cal, grid=np.linspace(0.3, 5.0, 95)):
    bestT, best = 1.0, np.inf
    for T in grid:
        nll = 0.0
        for pd_, g in cal:
            if g not in pd_: continue
            z = np.array([math.log(max(v, 1e-12)) / T for v in pd_.values()])
            z -= z.max(); pz = np.exp(z); pz /= pz.sum()
            nll -= math.log(max(pz[list(pd_).index(g)], 1e-12))
        if nll < best: best, bestT = nll, T
    return bestT

def temp_margin(pd_, T):
    z = np.array([math.log(max(v, 1e-12)) / T for v in pd_.values()])
    z -= z.max(); pz = np.exp(z); pz /= pz.sum()
    s = np.sort(pz)[::-1]
    return s[0] - (s[1] if len(s) > 1 else 0.0)

# ---- load -----------------------------------------------------------------
cal = [(parse_probs(r["opt_logprobs"]), r["gold"]) for r in load_jsonl(CAL_PATH)]
cal_scores = np.array([1 - pd_.get(g, 0.0) for pd_, g in cal])

ev = []
for p in sorted(glob.glob(CHEAP_GLOB)):
    b = bench_of(p)
    for r in load_jsonl(p):
        pd_ = parse_probs(r["opt_logprobs"])
        ev.append({"idx": r["idx"], "bench": b, "gold": r["gold"], "ok": int(r["ok"]),
                   "cheap_pred": max(pd_, key=pd_.get), "probs": pd_, **signals(pd_)})

# ================= PART A : 7B-only diagnostics ===========================
print("=" * 64 + "\nPART A — 7B-only diagnostics\n" + "=" * 64)
print(f"calibration: {len(cal)} rows   eval: {len(ev)} rows\n")
print(f"{'benchmark':<22}{'n':>6}{'7B acc':>9}")
accs = {}
for b in sorted({e["bench"] for e in ev}):
    sub = [e["ok"] for e in ev if e["bench"] == b]
    accs[b] = float(np.mean(sub))
    print(f"{b:<22}{len(sub):>6}{accs[b]:>9.4f}{'' if b in COMPETENT else '   (excluded)'}")
comp_ok = [e["ok"] for e in ev if e["bench"] in COMPETENT]
print(f"\ncompetent-4 micro acc = {np.mean(comp_ok):.4f}   "
      f"macro = {np.mean([accs[b] for b in COMPETENT]):.4f}")

mar_c   = np.array([e["margin"] for e in ev if e["bench"] in COMPETENT])
mar_all = np.array([e["margin"] for e in ev])
T = fit_T(cal)
print(f"\nmargin on competent-4: mean={mar_c.mean():.3f}   "
      f"{int(BUDGET*100)}th pct = {np.quantile(mar_c, BUDGET):.3f}")
print(f"fitted temperature T = {T:.2f}  "
      f"({'over-confident' if T > 1.05 else 'calibrated' if T > 0.95 else 'under-confident'})")

# frozen-gate escalation provenance (resolves tau=0.426 vs 0.631 headline)
try:
    R = pickle.load(open(GATE_PKL, "rb")); tau = float(R["tau"])
    print(f"\nfrozen gate: signal={R.get('signal')!r}  trained_on={R.get('trained_on')!r}  "
          f"gate={repr(R.get('gate'))[:40]}")
    print(f"frozen gate (tau={tau:.3f}, assuming raw-margin threshold): "
          f"escalates {(mar_c < tau).mean():.3f} of competent-4  |  "
          f"{(mar_all < tau).mean():.3f} of all-8220   (Table 2 headline = 0.631)")
except Exception as e:
    print(f"\n[gate pickle not read: {e}]")

# ground-truth escalation rate from the LIVE run
if os.path.exists(LIVE_OUT):
    live = load_jsonl(LIVE_OUT)
    dsets = {}
    for r in live:
        dsets.setdefault(r.get("dataset", "?"), []).append(int(r["escalate"]))
    comp_live = [v for d, vs in dsets.items() if norm(d) in COMP_NORM for v in vs]
    print(f"\nLIVE run ({LIVE_OUT}, n={len(live)}): overall escalation = "
          f"{np.mean([int(r['escalate']) for r in live]):.3f}")
    if comp_live:
        print(f"   competent-4 escalation (live, ground truth) = {np.mean(comp_live):.3f}")
    for d in sorted(dsets):
        print(f"     {d:<24} esc={np.mean(dsets[d]):.3f}  (n={len(dsets[d])})")
else:
    print(f"\n[{LIVE_OUT} not found — skipping live escalation provenance]")

# ================= PART B : routed bake-off ===============================
if not glob.glob(STRONG_GLOB):
    print("\n" + "=" * 64 + f"\nPART B skipped: no 32B files match {STRONG_GLOB}")
else:
    strong = {}                         # idx -> 32B correctness (its opt_logprobs are empty; need only ok)
    for p in glob.glob(STRONG_GLOB):
        for r in load_jsonl(p):
            strong[r["idx"]] = int(r["ok"])
    use  = [e for e in ev if e["bench"] in COMPETENT and e["idx"] in strong]
    miss = sum(1 for e in ev if e["bench"] in COMPETENT and e["idx"] not in strong)
    if miss:
        print(f"\n[warn] {miss} competent-4 queries missing from 32B files (idx mismatch)")

    cheap_ok  = np.array([e["ok"] for e in use])
    strong_ok = np.array([strong[e["idx"]] for e in use])
    margin_u  = -np.array([e["margin"] for e in use])      # uncertainty = -margin (high => escalate)

    def routed(esc):
        return np.where(esc, strong_ok, cheap_ok).astype(float)

    def margin_at_n(n_esc):                                # margin gate escalating exactly n_esc queries
        esc = np.zeros(len(margin_u), bool)
        if n_esc > 0:
            esc[np.argsort(margin_u)[::-1][:n_esc]] = True
        return routed(esc)

    def boot(corr_m, corr_ref):
        rng = np.random.default_rng(0); n = len(corr_m); d = np.empty(N_BOOT)
        for b in range(N_BOOT):
            i = rng.integers(0, n, n); d[b] = corr_m[i].mean() - corr_ref[i].mean()
        return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)

    print("\n" + "=" * 64 + f"\nPART B — routed bake-off, competent-4, n={len(use)}\n" + "=" * 64)
    rescue = ((cheap_ok == 0) & (strong_ok == 1)).mean()
    brk    = ((cheap_ok == 1) & (strong_ok == 0)).mean()
    print(f"cheap-only acc={cheap_ok.mean():.4f}   strong-only acc={strong_ok.mean():.4f}")
    print(f"rescuable={rescue:.4f}  breakable={brk:.4f}   "
          f"(escalate-all net = {rescue - brk:+.4f}; oracle escalates only the {rescue:.3f} rescuable)")

    n63 = round(BUDGET * len(use))
    cm63 = margin_at_n(n63)
    print(f"\nEvery method is compared to the MARGIN GATE escalating the SAME #queries (frontier-matched).")
    print(f"{'method':<26}{'esc%':>7}{'acc':>9}{'Δ vs margin@esc':>17}{'95% CI':>22}  verdict")
    print(f"{'margin (deployed @63%)':<26}{BUDGET*100:>6.1f}%{cm63.mean():>9.4f}{'reference':>17}{'':>22}")

    rows = []
    sig_u = {"top1":                 -np.array([e["top1"] for e in use]),
             "entropy":               np.array([e["entropy"] for e in use]),
             "temp_margin(T=%.1f)" % T: -np.array([temp_margin(e["probs"], T) for e in use])}
    for name, u in sig_u.items():
        esc = escalate_at_budget(u, BUDGET); n = int(esc.sum())
        c = routed(esc); m, lo, hi = boot(c, margin_at_n(n))
        rows.append((name, esc.mean(), c.mean(), m, lo, hi))

    alphas = np.round(np.linspace(0.02, 0.98, 97), 3)
    sizes_by_a = {a: np.array([set_size(e["probs"], conf_qhat(cal_scores, a)) for e in use]) for a in alphas}
    a_bud  = min(alphas, key=lambda a: abs((sizes_by_a[a] > TAU_SET).mean() - BUDGET))
    a_star = max(alphas, key=lambda a: fbe(sizes_by_a[a]))
    for tag, a in [("conformal@budget α=%.2f" % a_bud, a_bud),
                   ("CP-Router/FBE α*=%.2f" % a_star, a_star)]:
        esc = sizes_by_a[a] > TAU_SET; n = int(esc.sum())
        c = routed(esc); m, lo, hi = boot(c, margin_at_n(n))
        rows.append((tag, esc.mean(), c.mean(), m, lo, hi))

    for name, r, a, m, lo, hi in rows:
        v = "WIN" if (lo > 0 or hi < 0) else "tie"
        print(f"{name:<26}{r*100:>6.1f}%{a:>9.4f}{m:>+17.4f}{('[%+.4f, %+.4f]' % (lo, hi)):>22}  {v}")
    print(f"\nBonferroni at {len(rows)} comparisons -> demand ~{0.05/len(rows):.4f} per test before a WIN.")
    print("Each Δ isolates whether the method picks a BETTER escalation set than the margin gate at\n"
          "the same compute. A tie across the board is the predicted, paper-friendly outcome.")
