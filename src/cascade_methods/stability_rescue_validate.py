#!/usr/bin/env python3
"""
stability_rescue_validate.py - HONEST validation of the Visual-Stability RESCUE gate.

Mechanism: deployed gate escalates every low-margin sample. RESCUE keeps a low-margin sample on
the CHEAP model if its 7B-nt answer is invariant across resolution caps (visually stable) -> those
samples are usually already right, so escalating them wastes the 32B. Training-free, VLM-specific.

Protocol mirrors the DEPLOYED gate exactly (freeze on PMC-VQA train, test on held-out competent-4):
  TRAIN = pmc_vqa_train sample: 7B-nt @ caps {80,160,320,640}  + 32B-think labels.
  TEST  = MedVLThinker-Eval competent-4 (PMC,SLAKE,VQA-RAD,PathVQA), never seen during freeze.
Stability is defined over caps present in BOTH splits (no fullres on train) so the feature is
identical at freeze and test time. We also try CHEAPER stability sets (fewer extra passes) and an
eval-only 50/50 CV (tests whether the extra param overfits).
margin = logprob_top1 - logprob_top2 (deployed def). Deployed tau = 0.4264.
"""
import os, re, json
import numpy as np

PRUNE = "ckpts/gate_7b_prune"
STRONG_EVAL = "ckpts/gate_32b"
TRAIN_PRUNE = "ckpts/gate_7b_pmctrain_prune"
TRAIN_STRONG = "ckpts/gate_32b_pmctrain/ckpt_think.jsonl"
TAU = 0.4264123185919304
COMPETENT4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
CAPS4 = ["cap80", "cap160", "cap320", "cap640"]      # present in BOTH splits

def load_jsonl(path):
    m = {}
    if not os.path.exists(path): return m
    for l in open(path):
        if l.strip():
            r = json.loads(l); m[r["idx"]] = r
    return m

def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

# stability feature sets: which OTHER caps to compare against cap320 (extra cheap passes needed)
STAB_SETS = {
    "cap160 only (1 extra)":           ["cap160"],
    "cap640 only (1 extra)":           ["cap640"],
    "cap160+cap640 (2 extra,near)":    ["cap160", "cap640"],
    "cap80+cap160 (2 extra,cheap)":    ["cap80", "cap160"],
    "cap80+160+640 (3 extra)":         ["cap80", "cap160", "cap640"],
}

def build_eval():
    recs = {}
    for ds in COMPETENT4:
        caps = {c: load_jsonl(os.path.join(PRUNE, c, f"ckpt_{ds}_nothink_norag.jsonl")) for c in CAPS4}
        strong = load_jsonl(os.path.join(STRONG_EVAL, f"ckpt_{ds}_think_norag.jsonl"))
        idx = set(caps["cap320"])
        for c in CAPS4: idx &= set(caps[c])
        idx &= set(strong)
        for i in sorted(idx):
            recs.setdefault(ds, []).append(_mk(i, ds, caps, strong))
    return [r for ds in COMPETENT4 for r in recs[ds]]

def build_train():
    caps = {c: load_jsonl(os.path.join(TRAIN_PRUNE, c, "ckpt_nothink.jsonl")) for c in CAPS4}
    strong = load_jsonl(TRAIN_STRONG)
    idx = set(caps["cap320"])
    for c in CAPS4: idx &= set(caps[c])
    idx &= set(strong)
    return [_mk(i, "pmctrain", caps, strong) for i in sorted(idx)]

def _mk(i, ds, caps, strong):
    p320 = caps["cap320"][i]["pred"]
    preds = {c: caps[c][i]["pred"] for c in CAPS4}
    agree = {c: int(preds[c] == p320) for c in CAPS4 if c != "cap320"}
    return dict(idx=i, ds=ds, margin=margin(caps["cap320"][i]),
                ok320=caps["cap320"][i]["ok"], ok32=strong[i]["ok"], agree=agree)

def n_agree(r, stab_caps):  # how many of the chosen extra caps agree with cap320
    return sum(r["agree"][c] for c in stab_caps)

def cascade(rows, esc_mask):
    final = np.array([(r["ok32"] if e else r["ok320"]) for r, e in zip(rows, esc_mask)])
    return float(np.mean(esc_mask)), float(np.mean(final))

# ---------- mechanism diagnostic ----------
def diagnostic(rows, stab_caps):
    K = len(stab_caps)
    low = [r for r in rows if r["margin"] < TAU]
    stable = [r for r in low if n_agree(r, stab_caps) == K]
    unstable = [r for r in low if n_agree(r, stab_caps) < K]
    def acc(g): return float(np.mean([r["ok320"] for r in g])) if g else float("nan")
    def fix(g):  # of the cheap-wrong, how many 32B fixes (recoverability)
        w = [r for r in g if r["ok320"] == 0]
        return float(np.mean([r["ok32"] for r in w])) if w else float("nan")
    print(f"    among margin<tau (n={len(low)}): stable={len(stable)} unstable={len(unstable)}")
    print(f"      cheap-acc  stable={acc(stable):.3f}  unstable={acc(unstable):.3f}   "
          f"(rescue is justified if stable cheap-acc >> unstable)")
    print(f"      32B-fix-rate on the cheap-WRONG: stable={fix(stable):.3f}  unstable={fix(unstable):.3f}   "
          f"(rescue is safe if 32B rarely fixes the stable ones)")

# ---------- frozen-on-train -> test honest comparison ----------
def fit_margin_to_target(train, A_target):
    """min train-esc s.t. train-acc >= A_target, over tau."""
    taus = np.quantile([r["margin"] for r in train], np.linspace(0.01, 0.99, 99))
    best = None
    for t in taus:
        e, a = cascade(train, [r["margin"] < t for r in train])
        if a >= A_target - 1e-9 and (best is None or e < best[0]): best = (e, a, t)
    return best  # (esc, acc, tau)

def fit_rescue_to_target(train, stab_caps, A_target):
    """min train-esc s.t. train-acc >= A_target, over (tau, k). Rule: escalate iff margin<tau AND n_agree<k."""
    K = len(stab_caps)
    taus = np.quantile([r["margin"] for r in train], np.linspace(0.01, 0.99, 99))
    best = None
    for k in range(1, K + 1):          # keep cheap if n_agree >= k ; k=K means require full stability
        for t in taus:
            mask = [(r["margin"] < t) and (n_agree(r, stab_caps) < k) for r in train]
            e, a = cascade(train, mask)
            if a >= A_target - 1e-9 and (best is None or e < best[0]): best = (e, a, t, k)
    return best  # (esc, acc, tau, k)

def main():
    train = build_train(); test = build_eval()
    print("=" * 80)
    print(f"Visual-Stability RESCUE — honest validation  (train n={len(train)} PMC-VQA-train, "
          f"test n={len(test)} competent-4)")
    print("=" * 80)

    # baseline: deployed margin gate at frozen tau on TEST
    base_e, base_a = cascade(test, [r["margin"] < TAU for r in test])
    allcheap = float(np.mean([r["ok320"] for r in test])); allstrong = float(np.mean([r["ok32"] for r in test]))
    print(f"TEST always-cheap acc={allcheap:.4f}  always-strong acc={allstrong:.4f}")
    print(f"TEST DEPLOYED margin@tau=0.4264 : esc={base_e:.3f}  acc={base_a:.4f}\n")

    # train target = train accuracy of the deployed gate at its frozen tau (apples to apples)
    tr_e, tr_a = cascade(train, [r["margin"] < TAU for r in train])
    A_target = tr_a
    print(f"TRAIN deployed-gate acc (target to match) = {A_target:.4f}  (train esc={tr_e:.3f})\n")

    print("MECHANISM on TRAIN (PMC-VQA-train) — is the rescue signal even present here?")
    diagnostic(train, ["cap160", "cap640"])
    print("MECHANISM on TEST (competent-4):")
    diagnostic(test, ["cap160", "cap640"])

    # ---------- PARAMETER-FREE rescue: frozen tau, add full-stability filter (NO new tunable) ----------
    print("\nPARAMETER-FREE RESCUE  (tau frozen at 0.4264; keep cheap if ALL extra caps agree). "
          "Zero overfitting risk:")
    print(f"  {'stability set':<32}{'esc':>8}{'acc':>8}{'Δesc':>9}{'Δacc':>9}")
    print(f"  {'DEPLOYED (no rescue)':<32}{base_e:>8.3f}{base_a:>8.4f}{0.0:>+9.3f}{0.0:>+9.4f}")
    for name, sc in STAB_SETS.items():
        K = len(sc)
        e, a = cascade(test, [(r["margin"] < TAU) and (n_agree(r, sc) < K) for r in test])
        print(f"  {name:<32}{e:>8.3f}{a:>8.4f}{e-base_e:>+9.3f}{a-base_a:>+9.4f}")

    print("\nFROZEN-ON-TRAIN -> TEST  (params chosen ONLY on PMC-VQA train, then applied to held-out):")
    print(f"  {'stability set':<32}{'esc':>8}{'acc':>8}{'Δesc vs margin':>16}{'frozen (tau,k)':>18}")
    # margin re-fit to the same target on train (sanity; ~ deployed tau)
    fm = fit_margin_to_target(train, A_target)
    em, am = cascade(test, [r["margin"] < fm[2] for r in test])
    print(f"  {'margin gate (1 param)':<32}{em:>8.3f}{am:>8.4f}{0.0:>+16.3f}{('tau=%.3f'%fm[2]):>18}")
    results = {"margin": dict(esc=em, acc=am, tau=fm[2])}
    for name, sc in STAB_SETS.items():
        fr = fit_rescue_to_target(train, sc, A_target)
        if fr is None:
            print(f"  {name:<32}{'—':>8}{'(no train soln)':>8}"); continue
        e, a = cascade(test, [(r["margin"] < fr[2]) and (n_agree(r, sc) < fr[3]) for r in test])
        print(f"  {name:<32}{e:>8.3f}{a:>8.4f}{e-em:>+16.3f}{('tau=%.3f,k=%d'%(fr[2],fr[3])):>18}")
        results[name] = dict(esc=e, acc=a, tau=fr[2], k=fr[3], extra=sc)

    # ---------- eval-only 50/50 CV (does the extra param overfit within-distribution?) ----------
    print("\nEVAL 50/50 CROSS-VALIDATION (fit on half A, test on half B; mean over 6 seeds):")
    sc = ["cap80", "cap160", "cap640"]
    margin_e, margin_a, resc_e, resc_a = [], [], [], []
    for seed in range(6):
        rng = np.random.RandomState(seed); order = rng.permutation(len(test))
        A = [test[i] for i in order[:len(test)//2]]; B = [test[i] for i in order[len(test)//2:]]
        At = cascade(A, [r["margin"] < TAU for r in A])[1]
        fmg = fit_margin_to_target(A, At); frs = fit_rescue_to_target(A, sc, At)
        if fmg is None or frs is None: continue
        me, ma = cascade(B, [r["margin"] < fmg[2] for r in B])
        re_, ra = cascade(B, [(r["margin"] < frs[2]) and (n_agree(r, sc) < frs[3]) for r in B])
        margin_e.append(me); margin_a.append(ma); resc_e.append(re_); resc_a.append(ra)
    print(f"  margin  : esc={np.mean(margin_e):.3f}±{np.std(margin_e):.3f}  acc={np.mean(margin_a):.4f}")
    print(f"  rescue  : esc={np.mean(resc_e):.3f}±{np.std(resc_e):.3f}  acc={np.mean(resc_a):.4f}  "
          f"(Δesc={np.mean(resc_e)-np.mean(margin_e):+.3f} at acc Δ={np.mean(resc_a)-np.mean(margin_a):+.4f})")
    results["eval_cv"] = dict(margin_esc=float(np.mean(margin_e)), margin_acc=float(np.mean(margin_a)),
                              rescue_esc=float(np.mean(resc_e)), rescue_acc=float(np.mean(resc_a)))
    os.makedirs("results/cascade_methods", exist_ok=True)
    json.dump(results, open("results/cascade_methods/stability_rescue_validate.json", "w"), indent=1)
    print("\n-> results/cascade_methods/stability_rescue_validate.json")

if __name__ == "__main__":
    main()
