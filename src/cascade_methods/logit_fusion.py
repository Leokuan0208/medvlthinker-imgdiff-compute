#!/usr/bin/env python3
"""
logit_fusion.py -- OFFLINE logit-LEVEL fusion of Lingshu-7B + Lingshu-32B on the MCQ benchmarks,
using the per-OPTION posteriors already saved in the checkpoint dumps
(ckpts/gate_lingshu{7b,32b}_mcq/*.jsonl, schema key `opt_logprobs` = letter->logprob).

THE QUESTION (Exp 1): does FULL-POSTERIOR (logit-level) fusion beat always-32B on MORE MCQ
cells than the existing DECISION-level fusion F3 (which certifies a win only on PMC-VQA, +0.0135)?

No inference -- everything is computed from saved dumps. Every hyperparameter (F11 mixing weight
lambda, F6 contrastive-decoding alpha, F3 isotonic calibrators) is HELD-OUT via 5-fold cross-fit
(fit on 4/5, predict on the held-out 1/5, pool the held-out predictions). Paired bootstrap CIs
of the accuracy delta vs always-32B and vs F3.

METHODS:
  always-7B / always-32B          reference (each model's own argmax over its option posterior)
  F3  decision-level conf-advantage (RECOMPUTED here on the SAME subsample for apples-to-apples):
         agree -> shared answer; disagree -> take the leg with higher isotonic-calibrated P(correct).
  F11 full-posterior product-of-experts (log-opinion pool), per-slice reliability-weighted:
         pred = argmax_o [ lambda*logp32(o) + (1-lambda)*logp7(o) ], lambda in [0,1] held-out per slice
         (lambda->1 = trust 32B, lambda->0 = trust 7B, 0.5 = equal PoE). Also a fixed lambda=0.5 variant.
  F6  contrastive decoding (7B = amateur, 32B = expert), per-slice held-out alpha:
         among options plausible under 32B (p32(o) >= beta*max p32), pred = argmax_o [logp32(o) - alpha*logp7(o)].
         alpha=0 recovers always-32B, so held-out alpha reveals WHERE subtracting the 7B amateur helps vs hurts.

Launch from repo root:  python3 src/cascade_methods/logit_fusion.py
"""
import json, os
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
D7   = os.path.join(ROOT, "ckpts/gate_lingshu7b_mcq")
D32  = os.path.join(ROOT, "ckpts/gate_lingshu32b_mcq")
OUT  = os.path.join(ROOT, "results/cascade_methods/artifacts/logit_fusion.json")

SLICES = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
          "MedXpert-Reasoning", "MedXpert-Understanding"]
BROAD4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]      # perception/closed sets (method scope)

K = 5                       # cross-fit folds
LAMBDA_GRID = [round(x, 2) for x in np.linspace(0, 1, 11)]      # F11 mixing weight
ALPHA_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]            # F6 contrastive strength
BETA_PLAUS  = 0.1                                              # F6 plausibility floor (standard CD)
N_BOOT = 4000
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- data loading
def load_slice(b):
    r7  = {json.loads(l)["idx"]: json.loads(l) for l in open(f"{D7}/ckpt_{b}_nothink_norag.jsonl")}
    r32 = {json.loads(l)["idx"]: json.loads(l) for l in open(f"{D32}/ckpt_{b}_nothink_norag.jsonl")}
    idxs = sorted(set(r7) & set(r32))
    rows = []
    for i in idxs:
        a, b_ = r7[i], r32[i]
        assert str(a["gold"]) == str(b_["gold"]), f"gold mismatch @ {b} idx {i}"
        rows.append(dict(idx=i, gold=str(a["gold"]),
                         lp7=a["opt_logprobs"], lp32=b_["opt_logprobs"],
                         pred7=a["pred"], pred32=b_["pred"],
                         ok7=int(a["ok"]), ok32=int(b_["ok"])))
    return rows


def candidate_lps(lp7, lp32):
    """Union candidate set (spurious high letters are naturally suppressed in the argmax).
    Missing letter in a model -> floor = (that model's min present logprob) - 5.0."""
    cand = sorted(set(lp7) | set(lp32))
    f7  = min(lp7.values())  - 5.0
    f32 = min(lp32.values()) - 5.0
    v7  = np.array([lp7.get(o,  f7)  for o in cand], float)
    v32 = np.array([lp32.get(o, f32) for o in cand], float)
    return cand, v7, v32


# ----------------------------------------------------------------------------- methods (return pred letter)
def pred_poe(lp7, lp32, lam):
    cand, v7, v32 = candidate_lps(lp7, lp32)
    s = lam * v32 + (1.0 - lam) * v7
    return cand[int(np.argmax(s))]


def pred_cd(lp7, lp32, alpha, beta=BETA_PLAUS):
    cand, v7, v32 = candidate_lps(lp7, lp32)
    thr = np.max(v32) + np.log(beta)                  # p32(o) >= beta*max p32
    mask = v32 >= thr
    score = np.where(mask, v32 - alpha * v7, -1e18)
    return cand[int(np.argmax(score))]


def conf_top1(lp):
    """calibrated-confidence signal = top-1 softmax prob over the option posterior."""
    v = np.array(list(lp.values()), float)
    v = v - v.max()
    p = np.exp(v); p /= p.sum()
    return float(p.max())


# ----------------------------------------------------------------------------- isotonic (for F3), no sklearn dep needed
def isotonic_fit(x, y):
    """Pool-adjacent-violators; returns (sorted_x, fitted_y) step function."""
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, float)[order]; ys = np.asarray(y, float)[order]
    w = np.ones_like(ys)
    yv = ys.copy(); wv = w.copy(); idx = list(range(len(ys)))
    # PAV
    vals = list(yv); wts = list(wv); bnds = [[i] for i in range(len(ys))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i+1] + 1e-12:
            nv = (vals[i]*wts[i] + vals[i+1]*wts[i+1]) / (wts[i]+wts[i+1])
            nw = wts[i] + wts[i+1]
            vals[i] = nv; wts[i] = nw; bnds[i] += bnds[i+1]
            del vals[i+1]; del wts[i+1]; del bnds[i+1]
            if i > 0: i -= 1
        else:
            i += 1
    fx = []; fy = []
    for v, b in zip(vals, bnds):
        for j in b:
            fx.append(xs[j]); fy.append(v)
    return np.array(fx), np.array(fy)


def isotonic_predict(fx, fy, xq):
    return np.interp(xq, fx, fy, left=fy[0], right=fy[-1])


# ----------------------------------------------------------------------------- cross-fit driver
def folds(n):
    idx = np.arange(n); RNG.shuffle(idx)
    return [idx[i::K] for i in range(K)]


def crossfit_predictions(rows):
    """Return dict method -> np.array(ok per sample), held-out, aligned to rows order."""
    n = len(rows)
    gold = [r["gold"] for r in rows]
    fold_ids = folds(n)
    ok = {m: np.zeros(n, int) for m in
          ["always7", "always32", "F11_fixed", "F11_rw", "F6_cd", "F3_confadv"]}
    lam_used = []; alpha_used = []
    for te in fold_ids:
        tr = np.array(sorted(set(range(n)) - set(te.tolist())))
        # ---- F11 held-out lambda: maximise calibration accuracy ----
        best_lam, best_acc = 1.0, -1
        for lam in LAMBDA_GRID:
            acc = np.mean([pred_poe(rows[i]["lp7"], rows[i]["lp32"], lam) == gold[i] for i in tr])
            if acc > best_acc + 1e-12 or (abs(acc-best_acc) <= 1e-12 and abs(lam-1.0) < abs(best_lam-1.0)):
                best_acc, best_lam = acc, lam
        lam_used.append(best_lam)
        # ---- F6 held-out alpha ----
        best_a, best_aacc = 0.0, -1
        for a in ALPHA_GRID:
            acc = np.mean([pred_cd(rows[i]["lp7"], rows[i]["lp32"], a) == gold[i] for i in tr])
            if acc > best_aacc + 1e-12 or (abs(acc-best_aacc) <= 1e-12 and a < best_a):
                best_aacc, best_a = acc, a
        alpha_used.append(best_a)
        # ---- F3 isotonic calibrators on train ----
        c7 = np.array([conf_top1(rows[i]["lp7"])  for i in tr])
        c32= np.array([conf_top1(rows[i]["lp32"]) for i in tr])
        y7 = np.array([rows[i]["ok7"]  for i in tr], float)
        y32= np.array([rows[i]["ok32"] for i in tr], float)
        fx7, fy7   = isotonic_fit(c7, y7)
        fx32, fy32 = isotonic_fit(c32, y32)
        # ---- predict held-out ----
        for i in te:
            r = rows[i]
            ok["always7"][i]  = int(r["pred7"]  == gold[i])
            ok["always32"][i] = int(r["pred32"] == gold[i])
            ok["F11_fixed"][i]= int(pred_poe(r["lp7"], r["lp32"], 0.5) == gold[i])
            ok["F11_rw"][i]   = int(pred_poe(r["lp7"], r["lp32"], best_lam) == gold[i])
            ok["F6_cd"][i]    = int(pred_cd(r["lp7"], r["lp32"], best_a) == gold[i])
            # F3 decision-level conf-advantage
            if r["pred7"] == r["pred32"]:
                pf = r["pred7"]
            else:
                p7c  = isotonic_predict(fx7, fy7, conf_top1(r["lp7"]))
                p32c = isotonic_predict(fx32, fy32, conf_top1(r["lp32"]))
                pf = r["pred7"] if p7c > p32c else r["pred32"]
            ok["F3_confadv"][i] = int(pf == gold[i])
    return ok, dict(lambda_mean=float(np.mean(lam_used)), lambda_folds=lam_used,
                    alpha_mean=float(np.mean(alpha_used)), alpha_folds=alpha_used)


def boot_delta(ok_a, ok_b):
    """paired bootstrap mean(a-b): (delta, lo95, hi95)."""
    d = ok_a.astype(float) - ok_b.astype(float)
    n = len(d)
    means = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------------------- run
def main():
    result = {"protocol": dict(cross_fit_folds=K, lambda_grid=LAMBDA_GRID, alpha_grid=ALPHA_GRID,
                               beta_plaus=BETA_PLAUS, n_boot=N_BOOT,
                               data="ckpts/gate_lingshu{7b,32b}_mcq (per-option opt_logprobs, held-out subsample)",
                               offline=True),
              "per_slice": {}, "pooled": {}}
    all_ok = {}
    for b in SLICES:
        rows = load_slice(b)
        ok, hyp = crossfit_predictions(rows)
        all_ok[b] = ok
        entry = dict(n=len(rows),
                     acc7=float(ok["always7"].mean()), acc32=float(ok["always32"].mean()),
                     hyper=hyp, methods={})
        for m in ["F11_fixed", "F11_rw", "F6_cd", "F3_confadv"]:
            acc = float(ok[m].mean())
            d32, lo32, hi32 = boot_delta(ok[m], ok["always32"])
            df3, lof3, hif3 = boot_delta(ok[m], ok["F3_confadv"])
            entry["methods"][m] = dict(acc=acc,
                                       d_vs_32b=d32, d_vs_32b_lo=lo32, d_vs_32b_hi=hi32,
                                       certified_beat_32b=bool(lo32 > 0),
                                       d_vs_F3=df3, d_vs_F3_lo=lof3, d_vs_F3_hi=hif3)
        result["per_slice"][b] = entry
        print(f"\n===== {b}  (n={len(rows)}) =====")
        print(f"  acc7={entry['acc7']:.4f}  acc32={entry['acc32']:.4f}   "
              f"F11_rw lam~{hyp['lambda_mean']:.2f}  F6 alpha~{hyp['alpha_mean']:.2f}")
        for m in ["F3_confadv", "F11_fixed", "F11_rw", "F6_cd"]:
            e = entry["methods"][m]
            star = " *CERT" if e["certified_beat_32b"] else ""
            print(f"  {m:11s} acc={e['acc']:.4f}  d32={e['d_vs_32b']:+.4f} "
                  f"[{e['d_vs_32b_lo']:+.4f},{e['d_vs_32b_hi']:+.4f}]{star}")

    # ---- pooled ----
    def pool(names, tag):
        cat = {m: np.concatenate([all_ok[b][m] for b in names]) for m in
               ["always7", "always32", "F11_fixed", "F11_rw", "F6_cd", "F3_confadv"]}
        e = dict(slices=names, n=int(len(cat["always32"])),
                 acc7=float(cat["always7"].mean()), acc32=float(cat["always32"].mean()), methods={})
        for m in ["F11_fixed", "F11_rw", "F6_cd", "F3_confadv"]:
            d32, lo32, hi32 = boot_delta(cat[m], cat["always32"])
            df3, lof3, hif3 = boot_delta(cat[m], cat["F3_confadv"])
            e["methods"][m] = dict(acc=float(cat[m].mean()),
                                   d_vs_32b=d32, d_vs_32b_lo=lo32, d_vs_32b_hi=hi32,
                                   certified_beat_32b=bool(lo32 > 0),
                                   d_vs_F3=df3, d_vs_F3_lo=lof3, d_vs_F3_hi=hif3)
        result["pooled"][tag] = e
        print(f"\n===== POOLED {tag} (n={e['n']}) acc32={e['acc32']:.4f} =====")
        for m in ["F3_confadv", "F11_fixed", "F11_rw", "F6_cd"]:
            x = e["methods"][m]; star = " *CERT" if x["certified_beat_32b"] else ""
            print(f"  {m:11s} acc={x['acc']:.4f}  d32={x['d_vs_32b']:+.4f} "
                  f"[{x['d_vs_32b_lo']:+.4f},{x['d_vs_32b_hi']:+.4f}]{star}")
    pool(SLICES, "all7")
    pool(BROAD4, "broad4")
    pool([b for b in SLICES if b != "MMMU"], "excl_mmmu")

    # ---- certified-cell tally (the headline: does logit fusion beat 32B on MORE cells than F3?) ----
    tally = {}
    for m in ["F3_confadv", "F11_fixed", "F11_rw", "F6_cd"]:
        cells = [b for b in SLICES if result["per_slice"][b]["methods"][m]["certified_beat_32b"]]
        tally[m] = dict(n_certified=len(cells), cells=cells)
    result["certified_cells_vs_32b"] = tally
    print("\n===== CERTIFIED beat-32B cells (lower 95% CI > 0) =====")
    for m, t in tally.items():
        print(f"  {m:11s}: {t['n_certified']} cells  {t['cells']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
