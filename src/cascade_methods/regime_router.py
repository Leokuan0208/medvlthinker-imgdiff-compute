#!/usr/bin/env python3
"""regime_router.py -- turn the regime finding into a decision, and test it OUT OF SAMPLE.

WHAT IS BEING ROUTED.  Not an abstention: every branch returns an answer.  The router decides how to
SPEND, per question:
    route "select"  -> draw the 8 samples and take the frozen head's pick   (8x generator compute)
    route "greedy"  -> return the model's own greedy answer                 (1x)

WHY THERE IS ANYTHING TO ROUTE (free_signal_bakeoff / regime_detector, both 2026-08-21):
    in-domain      head - greedy = +0.039 to +0.078
    out-of-domain  head - greedy = -0.026 to -0.049
So on half the cells the 8x spend BUYS NEGATIVE ACCURACY.  A router that merely skips those is
cheaper and more accurate at once -- if the regime is detectable.

DETECTORS THAT ORDERED THE CELLS (regime_detector_2026-08-21.json, 7 cells):
    knn_train       spearman -0.821, pooled per-question AUROC 0.296 (i.e. 0.704 with the sign
                    flipped, the strongest per-question signal measured)
    domclf_maxprob  spearman +0.857, pooled per-question AUROC 0.667
Refuted: maha_pca64 (+0.214), vocab_coverage (+0.571), head_spread (-0.429), sc_entropy (0.000).

TWO HONESTY CONTROLS THIS SCRIPT ENFORCES, because without them the result is not interpretable:

 1. LEAVE-ONE-CELL-OUT.  The threshold is fitted on six cells and applied to the seventh, rotated.
    A threshold read off all seven and then scored on all seven is a fit, not a measurement.
 2. WITHIN-CELL vs BETWEEN-CELL.  A pooled AUROC over cells can be driven entirely by cells
    differing in BOTH detector level and outcome, with zero discriminative power inside any cell
    (Simpson's paradox).  Both are reported: the between-cell number licenses routing whole
    DATASETS, the within-cell number licenses routing individual QUESTIONS, and they are different
    claims with different deployment stories.

  python3 src/cascade_methods/regime_router.py
"""
import json, os
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
NPZ = os.path.join(ART, "regime_detector_2026-08-21_perquestion.npz")
OUT = os.path.join(ART, "regime_router_2026-08-21.json")
DETS = {"knn_train": -1.0, "domclf_maxprob": +1.0}   # sign: higher => trust the head


def auroc(score, lab):
    lab = np.asarray(lab).astype(int)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return float("nan")
    o = np.argsort(score); r = np.empty(len(score), float)
    r[o] = np.arange(1, len(score) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_paired(a, b, clusters=None, nboot=10000, seed=20260821):
    """Paired bootstrap, CLUSTERED BY CELL when clusters is given.

    CORRECTION 2026-08-21: the first version resampled questions i.i.d. and returned
    routed-vs-greedy +0.0133 [+0.0109,+0.0156] WIN.  That interval is anticonservative by roughly
    8x, because essentially all of the variance in this experiment is BETWEEN cells -- the router's
    behaviour is near-constant within a cell and swings from a 0.07 to a 0.96 select rate across
    them.  Resampling cells instead gives +0.0133 [-0.0064,+0.0334], a TIE.  An i.i.d. interval on
    clustered data is the same mistake the clustered-CI audit fixed elsewhere in this project.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(a))]
    else:
        clusters = np.asarray(clusters)
        groups = [np.where(clusters == c)[0] for c in np.unique(clusters)]
    d = np.empty(nboot)
    for i in range(nboot):
        s = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        d[i] = a[s].mean() - b[s].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(a.mean() - b.mean()), "ci": [float(lo), float(hi)],
            "verdict": "WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE",
            "resampled": "cells" if clusters is not None else "questions_iid"}


def macro_boot(A, B, cells, nboot=10000, seed=1):
    """Bootstrap of the MACRO (equal weight per cell) delta, resampling cells."""
    rng = np.random.default_rng(seed)
    d = np.empty(nboot)
    for i in range(nboot):
        k = rng.integers(0, len(cells), len(cells))
        d[i] = np.mean([A[cells[j]] for j in k]) - np.mean([B[cells[j]] for j in k])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(np.mean([A[c] for c in cells]) - np.mean([B[c] for c in cells])),
            "ci": [float(lo), float(hi)],
            "verdict": "WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE", "resampled": "cells"}


def main():
    z = np.load(NPZ)
    cells = sorted({k.split("|")[0] for k in z.files})
    D = {c: {k.split("|")[1]: z[k] for k in z.files if k.startswith(c + "|")} for c in cells}
    print(f"{len(cells)} cells: {cells}\n")

    art = {"title": "Routing select-vs-greedy on a regime detector, leave-one-cell-out",
           "date": "2026-08-21", "no_fabricated_numbers": True,
           "arms": {"always_greedy": "1x compute", "always_select": "8x compute",
                    "routed": "8x only where the detector says the head has support"},
           "detectors": {}}

    for det, sgn in DETS.items():
        # ---- within-cell discriminative power, per cell -----------------------------------
        within = {}
        for c in cells:
            lab = (D[c]["head_ok"] > D[c]["greedy_ok"]).astype(int)
            oth = (D[c]["greedy_ok"] > D[c]["head_ok"]).astype(int)
            m = (lab + oth) > 0
            within[c] = auroc(sgn * D[c][det][m], lab[m]) if m.sum() > 20 else float("nan")
        wv = np.array([v for v in within.values() if np.isfinite(v)])

        # ---- leave-one-cell-out routing ----------------------------------------------------
        rows, tot_acc, tot_sel, tot_n, cid = [], [], [], 0, []
        for held in cells:
            tr = [c for c in cells if c != held]
            # fit one global threshold on the six training cells
            xs = np.concatenate([sgn * D[c][det] for c in tr])
            hs = np.concatenate([D[c]["head_ok"] for c in tr])
            gs = np.concatenate([D[c]["greedy_ok"] for c in tr])
            best_t, best_v = None, -1e9
            for q in np.linspace(1, 99, 99):
                t = np.percentile(xs, q)
                v = np.where(xs >= t, hs, gs).mean()
                if v > best_v:
                    best_v, best_t = v, t
            x = sgn * D[held][det]
            take_head = x >= best_t
            acc = np.where(take_head, D[held]["head_ok"], D[held]["greedy_ok"])
            rows.append({"cell": held, "n": int(len(acc)),
                         "select_rate": float(take_head.mean()),
                         "routed": float(acc.mean()),
                         "always_select": float(D[held]["head_ok"].mean()),
                         "always_greedy": float(D[held]["greedy_ok"].mean()),
                         "within_cell_auroc": within[held]})
            tot_acc.append(acc); tot_sel.append(take_head); tot_n += len(acc)
            cid.append(np.full(len(acc), held))
            r = rows[-1]
            print(f"  [{det:15}] {held:17} n{r['n']:6}  sel {r['select_rate']:.2f}  "
                  f"routed {r['routed']:.4f}  select {r['always_select']:.4f}  "
                  f"greedy {r['always_greedy']:.4f}  withinAUROC {r['within_cell_auroc']:.3f}")
        A = np.concatenate(tot_acc)
        S = np.concatenate([D[c]["head_ok"] for c in cells])
        G = np.concatenate([D[c]["greedy_ok"] for c in cells])
        SR = np.concatenate(tot_sel); CID = np.concatenate(cid)
        # macro = equal weight per cell, the project's reporting convention
        macro = {k: float(np.mean([r[k] for r in rows]))
                 for k in ("routed", "always_select", "always_greedy", "select_rate")}
        art["detectors"][det] = {
            "sign": sgn, "per_cell": rows,
            "within_cell_auroc_mean": float(np.nanmean(wv)),
            "within_cell_auroc_range": [float(np.nanmin(wv)), float(np.nanmax(wv))],
            "macro": macro,
            "pooled": {"routed": float(A.mean()), "always_select": float(S.mean()),
                       "always_greedy": float(G.mean()), "select_rate": float(SR.mean())},
            "pooled_deltas_CELL_CLUSTERED": {
                "routed_vs_greedy": boot_paired(A, G, CID),
                "routed_vs_select": boot_paired(A, S, CID)},
            "pooled_deltas_iid_ANTICONSERVATIVE": {
                "routed_vs_greedy": boot_paired(A, G),
                "routed_vs_select": boot_paired(A, S)},
            "macro_deltas": {
                "routed_vs_greedy": macro_boot({r["cell"]: r["routed"] for r in rows},
                                               {r["cell"]: r["always_greedy"] for r in rows},
                                               [r["cell"] for r in rows]),
                "routed_vs_select": macro_boot({r["cell"]: r["routed"] for r in rows},
                                               {r["cell"]: r["always_select"] for r in rows},
                                               [r["cell"] for r in rows])},
            "compute_x_pooled": float(1 + 7 * SR.mean()),
            "compute_x_macro": float(1 + 7 * macro["select_rate"])}
        a = art["detectors"][det]
        print(f"  [{det:15}] MACRO routed {macro['routed']:.4f} vs select "
              f"{macro['always_select']:.4f} vs greedy {macro['always_greedy']:.4f} "
              f"| select rate {macro['select_rate']:.2f} ~ {a['compute_x_macro']:.2f}x compute")
        for nm, dd in a["macro_deltas"].items():
            print(f"  [{det:15}] MACRO {nm:18} {dd['delta']:+.4f} "
                  f"[{dd['ci'][0]:+.4f},{dd['ci'][1]:+.4f}] {dd['verdict']} (cells resampled)")
        print(f"  [{det:15}] within-cell AUROC mean {a['within_cell_auroc_mean']:.3f} "
              f"range [{a['within_cell_auroc_range'][0]:.3f},{a['within_cell_auroc_range'][1]:.3f}]"
              f"  <- 0.5 means NO per-question power\n")
        json.dump(art, open(OUT, "w"), indent=1)

    best = max(DETS, key=lambda d: art["detectors"][d]["macro"]["routed"])
    b = art["detectors"][best]
    mg = b["macro_deltas"]["routed_vs_greedy"]; ms = b["macro_deltas"]["routed_vs_select"]
    art["VERDICT"] = (
        f"{best}: macro routed {b['macro']['routed']:.4f}; vs always-greedy {mg['delta']:+.4f} "
        f"[{mg['ci'][0]:+.4f},{mg['ci'][1]:+.4f}] {mg['verdict']}; vs always-select "
        f"{ms['delta']:+.4f} [{ms['ci'][0]:+.4f},{ms['ci'][1]:+.4f}] {ms['verdict']}, at "
        f"{b['compute_x_macro']:.2f}x compute against 8x. With 7 cells the ACCURACY claim is a "
        f"tie in both directions; what survives is the COMPUTE claim -- same accuracy as "
        f"always-select for roughly half its sampling cost -- and that is arithmetic, not "
        f"statistics.")
    print(f"=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
