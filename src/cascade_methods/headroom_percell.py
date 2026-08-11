#!/usr/bin/env python3
"""
headroom_percell.py -- SCOUT A: PER-CELL HEADROOM ACCOUNTING.

QUESTION.  The canonical macro headline (MACRO, equal weight per reporting cell, Variant B, 8 cells,
1/8 each, CLEAN disjoint verifier) has accuracy-max at 0.6575 vs always-32B-direct 0.6567 -- a TIE
(+0.0008 [-0.0022, +0.0037], artifacts/cascade_selector_rerun_2026-08-05.json).  Before spending any
GPU, decide analytically WHERE a win could possibly come from.

THE STRUCTURAL FACT THIS FILE MEASURES.  A cascade that routes 7B -> 32B-direct delivers, per item,
either ok7 or ok32.  Therefore

    max accuracy of ANY such router on a cell  =  mean(max(ok7, ok32))  =  1 - P(both wrong)
    max GAIN over always-32B-direct            =  P(7B right AND 32B wrong)  =:  p10

p10 is an EXACT, MEASURED upper bound on the cell's contribution to any "beat always-32B-direct"
claim, and it is achievable only to the extent that the p10 subset is IDENTIFIABLE from a signal the
router can see.  So this file reports, per cell:
  * the 2x2 contingency (p11 / p10 / p01 / p00) of 7B-direct vs 32B-direct,
  * ceilings: 2-way item oracle {7B,32B-direct}, 3-way item oracle {7B,32B-direct,32B-reasoning},
    and (open cells) the 8-sample 7B pool oracle alone and unioned with the 32B,
  * identifiability of the p10 set from cheap signals: AUROC over all items, AUROC restricted to the
    disagreement set, and the REALIZED cross-fit gain of a single-threshold keep-7B rule,
  * sensitivity: how much a cell must move to shift the macro headline, and the leave-one-cell-out
    range of the current vs-direct delta.

NULL TEST (printed and stored).  Every per-cell accuracy for all 7 systems is recomputed from the
per-sample vectors saved by cascade_selector_rerun.py (_selector_rerun_parts/vec_disjoint.npz) and
compared field-by-field with cascade_selector_rerun_2026-08-05.json's `disjoint` arm; the open-text
bar (greedy / oracle@8 / selected / sel_eff / per-set sel_eff) is recomputed from the disjoint
transfer dumps and compared with the frozen values in src/training_methods/genframe_data.py.

NUMERICS.  Pure numpy float64 on CPU; no torch, no GPU, no BLAS-order-dependent reductions bigger
than a mean over a 1-D array, so the TF32 / thread-count / row-order landmines do not apply here.
Bootstrap: paired at item level WITHIN each cell, nboot=10000, seed 20260810, common random numbers
across all quantities computed from the same cell.

Launch from the repo root:   python3 src/cascade_methods/headroom_percell.py
"""
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import integrated_method as IM       # noqa: E402  MCQ loaders (margin/conf/casp), open_bestof8
import integrated_pandora as IP      # noqa: E402  load_open_rows (per-candidate scores + judge)

ROOT = IM.ROOT
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
VEC = os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz")
RERUN = os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")
OUT = os.path.join(ART, "headroom_percell_2026-08-10.json")

DISJOINT = "ckpts/train/lora_verifier_disjoint"

ORDER = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = ORDER[:5]
OPEN = ORDER[5:]
OPEN_KEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}
SYSTEMS = ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
           "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]

NBOOT = 10000
SEED = 20260810
R = lambda x, k=4: (None if x is None else round(float(x), k))


# =====================================================================================================
# helpers
# =====================================================================================================
def auroc(score, y):
    """Rank AUROC with tie handling (identical implementation to IM.auroc)."""
    score = np.asarray(score, float); y = np.asarray(y, int)
    P = score[y == 1]; N = score[y == 0]
    if len(P) == 0 or len(N) == 0:
        return None
    a = np.concatenate([P, N]); o = a.argsort(); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    ss = np.zeros(len(c)); np.add.at(ss, inv, rk); rk = (ss / c)[inv]
    return float((rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N)))


def indep_floor(a, b):
    """The item-level oracle accuracy two systems would show if their errors were INDEPENDENT at the
    same marginal accuracies: 1-(1-a)(1-b).  This is the permutation / luck floor for an oracle gap:
    an oracle gap at or above it is exactly what uncorrelated noise produces, and in this project such
    gaps have never been harvestable (the 'luck floor' negatives, retrospective §6)."""
    a, b = float(np.mean(a)), float(np.mean(b))
    return R(1.0 - (1.0 - a) * (1.0 - b))


def boot_mean_ci(d, rng, nboot=NBOOT, chunk=500):
    """95% CI of mean(d) by bootstrap over items (d is a 1-D per-item array)."""
    d = np.asarray(d, float); n = len(d)
    if n == 0:
        return dict(mean=None, lo=None, hi=None, sig=None, n=0)
    b = np.empty(nboot)
    for s in range(0, nboot, chunk):
        m = min(chunk, nboot - s)
        b[s:s + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return dict(mean=R(d.mean()), lo=R(lo), hi=R(hi), sig=bool(lo > 0 or hi < 0), n=int(n))


def best_prefix_gain(score, delta):
    """In-sample optimum of a single-threshold KEEP-7B rule on `score` (descending):
    max over k of sum(delta[order[:k]]) / n, where delta = ok7 - ok32.
    This is an OPTIMISTIC (in-sample, threshold chosen on the same data) upper bound for that signal."""
    o = np.argsort(-np.asarray(score, float), kind="stable")
    cs = np.cumsum(np.asarray(delta, float)[o])
    k = int(np.argmax(cs))
    return dict(gain=R(cs[k] / len(delta)), keep_frac=R((k + 1) / len(delta)))


def crossfit_keep7_gain(score, delta, K=5):
    """HONEST (5-fold cross-fit) realized gain of a single-threshold keep-7B rule: the threshold that
    maximises sum(delta) over kept items is chosen on the TRAIN folds and applied to the held-out fold.
    Same cross-fitting protocol the deployed cascade uses (paper_baselines.cascade_persample, K=5,
    modulo folds).  Returns the realized macro-relevant gain (mean over ALL items of delta*kept) and
    the kept fraction.  A negative value means the signal cannot support the lever at all."""
    score = np.asarray(score, float); delta = np.asarray(delta, float); n = len(delta)
    kept = np.zeros(n, bool)
    for f in range(K):
        te = np.arange(n) % K == f; tr = ~te
        if tr.sum() < 2 or te.sum() < 1:
            continue
        s_tr, d_tr = score[tr], delta[tr]
        o = np.argsort(-s_tr, kind="stable")
        cs = np.cumsum(d_tr[o])
        k = int(np.argmax(cs))
        if cs[k] <= 0:                       # no profitable threshold on train -> keep nobody
            continue
        thr = s_tr[o][k]                      # keep score >= thr
        kept[te] = score[te] >= thr
    return dict(gain=R(float((delta * kept).sum() / n)), keep_frac=R(float(kept.mean())),
                per_item=(delta * kept))


# =====================================================================================================
# 1.  load per-sample vectors + NULL TEST
# =====================================================================================================
def load_vectors():
    z = np.load(VEC)
    V = {c: {s: np.asarray(z[f"{c}|{s}"], float) for s in SYSTEMS} for c in ORDER}
    return V


def null_test(V):
    art = json.load(open(RERUN))
    arm = art["per_arm"]["disjoint"]
    dev = 0.0; ncmp = 0
    for c in ORDER:
        for s, v in arm["per_cell_acc"][c].items():
            dev = max(dev, abs(round(float(V[c][s].mean()), 4) - v)); ncmp += 1
    macro_dev = 0.0
    for s in SYSTEMS:
        m = float(np.mean([V[c][s].mean() for c in ORDER]))
        macro_dev = max(macro_dev, abs(round(m, 4) - arm["macro_acc"][s])); ncmp += 1
    return dict(source=os.path.relpath(RERUN, ROOT), arm="disjoint",
                fields_compared=ncmp, max_abs_deviation_percell=R(dev, 10),
                max_abs_deviation_macro=R(macro_dev, 10), passed=bool(dev == 0.0 and macro_dev == 0.0))


# =====================================================================================================
# 2.  cheap signals + open-text pools
# =====================================================================================================
def mcq_signals(V):
    """Per-MCQ-cell cheap routing signals, asserted aligned to the saved ok vectors."""
    out = {}
    specs = [("PMC_VQA", "PMC_VQA", None), ("SLAKE_closed", "SLAKE", "SLAKE"),
             ("VQA_RAD_closed", "VQA_RAD", "YESNO"), ("PATH_VQA_closed", "PATH_VQA", "YESNO")]
    for cell, ds, cl in specs:
        d = IM.mcq_closed(ds, cl)
        assert np.array_equal(d["ok7"], V[cell]["always_7b"]), cell
        assert np.array_equal(d["ok32"], V[cell]["always_32b_direct"]), cell
        sig = {"7B_margin": d["margin"], "7B_conf": d["conf"]}
        if d["casp"] is not None:
            sig["7B_selfconsistency_cap320_vs_fullres"] = d["casp"]
        out[cell] = sig
    d = IM.mcq_medxpert()
    assert np.array_equal(d["ok7"], V["MedXpertQA-MM"]["always_7b"])
    assert np.array_equal(d["ok32"], V["MedXpertQA-MM"]["always_32b_direct"])
    out["MedXpertQA-MM"] = {"7B_margin": d["margin"], "7B_conf": d["conf"]}
    return out


def open_pools(V):
    """Per-open-cell 8-candidate pool (CLEAN disjoint verifier) + derived signals."""
    IM.OPEN_VERIFIER_DIR = DISJOINT
    IP.ADAPTER = DISJOINT
    out = {}
    for cell in OPEN:
        rows = IP.load_open_rows(OPEN_KEY[cell])
        sl = np.array([r["sl"][:8] for r in rows], float)
        sc = np.array([r["scores"][:8] for r in rows], float)
        greedy = np.array([r["greedy"] for r in rows], float)
        strong = np.array([r["strong"] for r in rows], float)
        assert np.array_equal(greedy, V[cell]["always_7b"]), cell
        assert np.array_equal(strong, V[cell]["always_32b_direct"]), cell
        pick = sc.argmax(1)
        sig = {"verifier_max_score": sc.max(1),
               "verifier_mean_score": sc.mean(1),
               "verifier_score_std": sc.std(1)}
        # pool self-consistency on the SURFACE STRINGS of the 8 sampled answers (a free cheap signal)
        dp = json.load(open(os.path.join(ROOT, DISJOINT, f"transfer_dump_{OPEN_KEY[cell]}_lingshu7b.json")))
        by_idx = {r["idx"]: r for r in dp}
        agree = []
        for r in rows:
            preds = [str(p).strip().lower() for p in by_idx[r["idx"]].get("preds", [])[:8]]
            agree.append(np.unique(preds, return_counts=True)[1].max() / 8.0 if len(preds) == 8 else np.nan)
        agree = np.asarray(agree, float)
        if not np.isnan(agree).any():
            sig["pool_selfconsistency"] = agree
        out[cell] = dict(sl=sl, sc=sc, greedy=greedy, strong=strong,
                         selected=sl[np.arange(len(rows)), pick], oracle8=sl.max(1), sig=sig)
    return out


def open_bar_null_test(P):
    """Reproduce the frozen open-text bar (genframe_data.py FROZEN block) from the disjoint dumps."""
    frozen = dict(oracle8=0.626013, greedy=0.449467, selected=0.485288, sel_eff=0.775204,
                  per_ds={"slake_open": 0.850088, "vqa_rad_open": 0.761905, "pathvqa_open": 0.722581})
    G = np.concatenate([P[c]["greedy"] for c in OPEN])
    O = np.concatenate([P[c]["oracle8"] for c in OPEN])
    S = np.concatenate([P[c]["selected"] for c in OPEN])
    got = dict(oracle8=float(O.mean()), greedy=float(G.mean()), selected=float(S.mean()),
               sel_eff=float(S[O == 1].mean()))
    per_ds = {OPEN_KEY[c]: float(P[c]["selected"][P[c]["oracle8"] == 1].mean()) for c in OPEN}
    dev = max([abs(got[k] - frozen[k]) for k in got] +
              [abs(per_ds[k] - frozen["per_ds"][k]) for k in per_ds])
    return dict(source="src/training_methods/genframe_data.py FROZEN block (incumbent, clean disjoint)",
                n=int(len(G)), measured={k: R(v, 6) for k, v in got.items()},
                measured_per_ds_sel_eff={k: R(v, 6) for k, v in per_ds.items()},
                frozen=frozen, max_abs_deviation=R(dev, 9), passed=bool(dev < 1e-6),
                identity_check=dict(
                    claim="selected == sel_eff * oracle@8 (exact, because a pick can only be correct "
                          "when the pool is recoverable)",
                    sel_eff_times_oracle=R(got["sel_eff"] * got["oracle8"], 9),
                    selected=R(got["selected"], 9),
                    max_abs_deviation=R(abs(got["sel_eff"] * got["oracle8"] - got["selected"]), 12)))


# =====================================================================================================
# 3.  the per-cell table
# =====================================================================================================
def per_cell_table(V, SIG, P, rng):
    art = json.load(open(RERUN))["per_arm"]["disjoint"]
    esc = art["escalation"]["per_cell"]
    ocd = art["open_cell_detail"]
    GEN7_F, GEN32_F = 1.0, 4.57                       # paper_baselines.GEN7[2] / GEN32N[2]

    rows = {}
    for c in ORDER:
        ok7 = V[c]["always_7b"]; ok32 = V[c]["always_32b_direct"]; okT = V[c]["always_32b_reasoning"]
        n = len(ok7)
        p11 = float(((ok7 == 1) & (ok32 == 1)).mean())
        p10 = float(((ok7 == 1) & (ok32 == 0)).mean())
        p01 = float(((ok7 == 0) & (ok32 == 1)).mean())
        p00 = float(((ok7 == 0) & (ok32 == 0)).mean())
        route2 = np.maximum(ok7, ok32)
        route3 = np.maximum(route2, okT)

        # ---- cost of the deployed policy on this cell (FLOP-eq, one 32B pass = 4.57) ----
        if c in MCQ:
            # compute-lean on MCQ = margin cascade (7B on all + 32B on the escalated set).
            # accuracy-max on MCQ = always-32B-direct, except PMC where the F8 certified veto runs the
            # 7B on all items and the 32B only on the non-vetoed set (cost filled in by f8_all_mcq).
            cost_cl = GEN7_F + esc[c] * GEN32_F
            cost_am = None if c == "PMC_VQA" else GEN32_F
        else:
            cost_cl = ocd[c]["cost_cl"]["flops"]
            cost_am = ocd[c]["cost_am2"]["flops"]

        row = dict(
            n=int(n), format=("MCQ/closed" if c in MCQ else "open-text"),
            reasoning_dump_is_real=bool(not np.array_equal(okT, ok32)) if c in MCQ else True,
            acc=dict(always_7b=R(ok7.mean()), always_32b_direct=R(ok32.mean()),
                     always_32b_reasoning=R(okT.mean()), oracle_mode_32b=R(V[c]["oracle_mode_32b"].mean()),
                     method_compute_lean=R(V[c]["method_compute_lean"].mean()),
                     method_accuracy_max_veto=R(V[c]["method_accuracy_max_veto"].mean()),
                     method_accuracy_max_fusion=R(V[c]["method_accuracy_max_fusion"].mean())),
            escalation_rate_compute_lean=R(esc[c]),
            cost_flopeq=dict(always_7b=1.0, always_32b_direct=GEN32_F,
                             method_compute_lean=R(cost_cl, 3),
                             method_accuracy_max_veto=(R(cost_am, 3) if cost_am is not None else None)),
            contingency_7Bdirect_vs_32Bdirect=dict(
                both_right=R(p11), only_7B_right=R(p10), only_32B_right=R(p01), both_wrong=R(p00),
                disagreement_rate=R(p10 + p01)),
            ceilings=dict(
                item_oracle_7B_or_32Bdirect=dict(
                    acc=R(route2.mean()),
                    gain_over_always_32B_direct=boot_mean_ci(route2 - ok32, rng),
                    acc_if_the_two_systems_erred_independently=indep_floor(ok7, ok32),
                    requires="the two forward passes the deployed cascade already has",
                    evidence="exact: mean(max(ok7,ok32)) over the cell's own per-sample vectors; "
                             "equals 1 - P(both wrong); the gain equals p10 identically"),
                item_oracle_32Bdirect_or_32Breasoning=dict(
                    acc=R(np.maximum(ok32, okT).mean()),
                    gain_over_always_32B_direct=boot_mean_ci(np.maximum(ok32, okT) - ok32, rng),
                    acc_if_the_two_modes_erred_independently=indep_floor(ok32, okT),
                    requires="a 32B reasoning pass as a third tier (expensive)",
                    evidence="exact: mean(max(ok32,okT)); ITEM-level mode routing on the 32B, versus the "
                             "deployed oracle_mode_32b which picks one mode per BENCHMARK"),
                item_oracle_7B_or_32Bdirect_or_32Breasoning=dict(
                    acc=R(route3.mean()),
                    gain_over_always_32B_direct=boot_mean_ci(route3 - ok32, rng),
                    requires="all three tiers",
                    evidence="exact: mean(max(ok7,ok32,okT)); adds whatever a 32B-reasoning third tier "
                             "could contribute on top of the 2-way oracle"),
            ),
        )
        rows[c] = row

    # ---- open cells: the 8-sample 7B pool ceilings ----
    for c in OPEN:
        pl = P[c]; ok32 = V[c]["always_32b_direct"]
        orc = pl["oracle8"]; sel = pl["selected"]
        pool_or_32 = np.maximum(orc, ok32)
        rows[c]["ceilings"]["pool_oracle_at8_7B_only"] = dict(
            acc=R(orc.mean()),
            gain_over_always_32B_direct=boot_mean_ci(orc - ok32, rng),
            requires="the 8 cheap 7B samples the open arm already draws, plus a PERFECT selector",
            evidence=f"measured: P(at least one of the 8 sampled 7B answers is judged correct), "
                     f"clean disjoint verifier pool {DISJOINT}/transfer_dump_{OPEN_KEY[c]}_lingshu7b.json")
        rows[c]["ceilings"]["item_oracle_pool_at8_or_32Bdirect"] = dict(
            acc=R(pool_or_32.mean()),
            gain_over_always_32B_direct=boot_mean_ci(pool_or_32 - ok32, rng),
            requires="the 8 cheap samples + one 32B pass + a PERFECT selector AND a PERFECT router",
            evidence="exact: mean(max(oracle@8, ok32)); the ceiling of perfect selection AND perfect "
                     "routing on top of the existing 8-sample pool")
        # the DEPLOYED cheap leg on an open cell is best-of-8 + verifier pick, not the greedy answer
        rows[c]["contingency_bestof8_selected_vs_32Bdirect"] = dict(
            both_right=R(float(((sel == 1) & (ok32 == 1)).mean())),
            only_selected_right=R(float(((sel == 1) & (ok32 == 0)).mean())),
            only_32B_right=R(float(((sel == 0) & (ok32 == 1)).mean())),
            both_wrong=R(float(((sel == 0) & (ok32 == 0)).mean())),
            note="this is the contingency that actually bounds the open arm, because its cheap leg is "
                 "the verifier's pick out of 8, not the greedy answer")
        rows[c]["open_pool"] = dict(
            greedy=R(pl["greedy"].mean()), selected_incumbent=R(sel.mean()), oracle_at8=R(orc.mean()),
            sel_eff=R(float(sel[orc == 1].mean()), 6),
            coverage_wall_no_correct_candidate=R(1.0 - orc.mean()),
            selection_wall_recoverable_but_missed=R(float((orc - sel).mean())),
            pool_recoverable_and_32B_wrong=R(float(((orc == 1) & (ok32 == 0)).mean())),
            meanN_deployed=R(json.load(open(RERUN))["per_arm"]["disjoint"]["open_cell_detail"][c]["meanN"], 3),
            note="selected == sel_eff * oracle@8 exactly")
    return rows


# =====================================================================================================
# 4.  identifiability of the p10 ("7B wins") subset
# =====================================================================================================
def identifiability(V, SIG, P, rng):
    out = {}
    for c in ORDER:
        ok7 = V[c]["always_7b"]; ok32 = V[c]["always_32b_direct"]
        delta = ok7 - ok32                              # +1 = 7B wins, -1 = 32B wins, 0 = tie
        y_all = ((ok7 == 1) & (ok32 == 0)).astype(int)   # the p10 set, over ALL items
        dis = delta != 0
        y_dis = (delta[dis] > 0).astype(int)             # among disagreements: did the 7B win?
        sigs = dict(SIG[c]) if c in MCQ else dict(P[c]["sig"])
        blk = dict(p10_share_of_disagreements=R(float(y_all.sum()) / max(int(dis.sum()), 1)),
                   n_disagree=int(dis.sum()), signals={})
        for name, s in sigs.items():
            s = np.asarray(s, float)
            cf = crossfit_keep7_gain(s, delta)
            blk["signals"][name] = dict(
                auroc_p10_vs_rest_all_items=R(auroc(s, y_all)),
                auroc_7Bwins_among_disagreements=R(auroc(s[dis], y_dis) if dis.sum() > 1 else None),
                insample_best_threshold_gain=best_prefix_gain(s, delta),
                crossfit_keep7_gain=dict(gain=cf["gain"], keep_frac=cf["keep_frac"],
                                         ci=boot_mean_ci(cf["per_item"], rng)))
        # the deployed levers on this cell, for reference
        blk["deployed_lever_gain_over_32B_direct"] = dict(
            compute_lean=R(float((V[c]["method_compute_lean"] - ok32).mean())),
            accuracy_max_veto=R(float((V[c]["method_accuracy_max_veto"] - ok32).mean())),
            accuracy_max_fusion=R(float((V[c]["method_accuracy_max_fusion"] - ok32).mean())))
        out[c] = blk
    return out


# =====================================================================================================
# 4b.  THE INCUMBENT LEVER, APPLIED TO EVERY MCQ CELL
# =====================================================================================================
def f8_all_mcq(V, rng):
    """The deployed accuracy-max MCQ lever is beat32b_more.f8_veto -- the CERTIFIED VETO (cross-fit
    7B-confidence quantile bins; keep the 7B answer inside a bin iff a one-sided Wilson lower bound on
    the 7B's precision there is >= the 32B's accuracy there; run the 32B everywhere else).  It answers
    on every item; it is NOT abstention.  In the published method it is applied to PMC-VQA ONLY -- the
    other four MCQ cells are always-32B-direct.  This runs the SAME unmodified function on all five and
    reports what it would deliver, because that is the cheapest available macro lever.

    It is a DIAGNOSTIC of the lever's size on already-seen cells: f8_veto cross-fits its bins 5-fold
    within each cell, exactly as the deployed PMC arm does, but the decision to apply it to a cell at
    all would be a new choice made with eval visibility."""
    import beat32b_fusion as BF          # noqa: E402  mcq() -> aligned per-sample arrays incl. c7
    import beat32b_more as BB            # noqa: E402  f8_veto
    specs = [("PMC_VQA", ("PMC_VQA", None, "lingshu32b_think")),
             ("SLAKE_closed", ("SLAKE", "SLAKE", "lingshu32b_think")),
             ("VQA_RAD_closed", ("VQA_RAD", "YESNO", "lingshu32b_think")),
             ("PATH_VQA_closed", ("PATH_VQA", "YESNO", "lingshu32b_think")),
             ("MedXpertQA-MM", ("MedXpertQA-MM", None, "lingshu32b_reason"))]
    GEN7_F, GEN32_F = 1.0, 4.57
    out = {}
    for cell, (ds, cl, tag) in specs:
        d = BF.mcq(ds, cl, think_tag=tag)
        ok7, ok32 = V[cell]["always_7b"], V[cell]["always_32b_direct"]
        assert np.array_equal(d["ok7"], ok7) and np.array_equal(d["ok32"], ok32), cell
        ok_f8, veto = BB.f8_veto(d)
        gain = ok_f8 - ok32
        rec = dict(veto_rate=R(float(veto.mean())), escalation=R(float(1.0 - veto.mean())),
                   acc_f8=R(float(ok_f8.mean())), acc_always_32b_direct=R(float(ok32.mean())),
                   gain_over_always_32B_direct=boot_mean_ci(gain, rng),
                   macro_contribution_if_adopted=R(float(gain.mean()) / len(ORDER)),
                   cost_flopeq=R(GEN7_F + float(1.0 - veto.mean()) * GEN32_F, 3),
                   cost_x_always_32b_direct=R((GEN7_F + float(1.0 - veto.mean()) * GEN32_F) / GEN32_F, 3),
                   is_the_deployed_policy_on_this_cell=bool(cell == "PMC_VQA"))
        if cell == "PMC_VQA":
            rec["reproduces_deployed_am2_vector"] = bool(
                np.array_equal(ok_f8, V["PMC_VQA"]["method_accuracy_max_veto"]))
        out[cell] = rec
    tot = sum(out[c]["gain_over_always_32B_direct"]["mean"] for c in MCQ)
    out["_summary"] = dict(
        summed_cell_gain_over_the_5_MCQ_cells=R(tot),
        macro_delta_if_adopted_on_all_5=R(tot / len(ORDER)),
        macro_delta_currently_realised_PMC_only=R(out["PMC_VQA"]["gain_over_always_32B_direct"]["mean"]
                                                  / len(ORDER)),
        incremental_macro_delta_from_the_other_4=R((tot - out["PMC_VQA"]["gain_over_always_32B_direct"]["mean"])
                                                   / len(ORDER)))
    return out


# =====================================================================================================
# 5.  MCQ best-of-N: the measured oracle@8 and its LUCK FLOOR
# =====================================================================================================
def mcq_bestofn_luckfloor():
    """The 8-sample self-consistency dumps for the 7B on the MCQ families (ckpts/mcq_gen_verify).
    These are SUBSETS of the reporting cells and are NOT merged into them.  The point is the luck
    floor: on a K-option MCQ, an oracle over 8 samples is largely a COVERAGE artifact -- if the 8
    samples touch d distinct options, a uniformly random gold lands in that set with probability d/K.
    Reported so nobody mistakes a big MCQ oracle@8 for real headroom."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "ckpts/mcq_gen_verify/lingshu7b/*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        oks = np.array([r["oks"][:8] for r in rows], float)
        sc = np.array([r["scores"][:8] for r in rows], float)
        greedy = np.array([r["greedy_ok"] for r in rows], float)
        pick = sc.argmax(1)
        # number of DISTINCT surface predictions in the 8 samples
        ndist = np.array([len(set(str(p).strip().lower() for p in r["preds"][:8])) for r in rows], float)
        base = os.path.basename(f)
        rec = dict(file=os.path.relpath(f, ROOT), n=len(rows),
                   greedy=R(greedy.mean()), oracle_at8=R(oks.max(1).mean()),
                   verifier_pick=R(oks[np.arange(len(rows)), pick].mean()),
                   mean_distinct_predictions_of_8=R(ndist.mean(), 3))
        # option count for the LETTER-format runs, read from the MedEvalKit dump for that dataset
        ds = ("MedXpertQA-MM" if "MedXpert" in base else
              "PMC_VQA" if "PMC_VQA" in base else
              "SLAKE" if "SLAKE" in base else
              "VQA_RAD" if "VQA_RAD" in base else "PATH_VQA")
        raw = IM.load_raw("lingshu7b_full", ds)
        if "letter" in base and raw:
            k = float(np.mean([len(r0.get("choices", [])) for r0 in raw if r0.get("choices")]))
            rec["mean_n_options"] = R(k, 2)
            rec["luck_floor_random_gold_inside_the_distinct_set"] = R(float(np.mean(ndist)) / k)
            rec["luck_floor_note"] = (
                "on a K-option MCQ, 8 samples that touch d distinct options contain a UNIFORMLY RANDOM "
                "gold with probability d/K. That fraction of oracle@8 is pure coverage, not signal.")
        out[base] = rec
    return out


# =====================================================================================================
# 6.  macro sensitivity + leave-one-cell-out
# =====================================================================================================
def sensitivity(V, rows, rng):
    W = 1.0 / len(ORDER)
    am = {c: V[c]["method_accuracy_max_veto"] for c in ORDER}
    cl = {c: V[c]["method_compute_lean"] for c in ORDER}
    d32 = {c: V[c]["always_32b_direct"] for c in ORDER}

    macro = lambda vecs, keys: float(np.mean([vecs[c].mean() for c in keys]))
    cur_am = macro(am, ORDER); cur_cl = macro(cl, ORDER); cur_32 = macro(d32, ORDER)

    # --- bootstrap the macro vs-direct delta (paired within each cell, common random numbers) ---
    per_cell_d = {c: am[c] - d32[c] for c in ORDER}
    b = np.empty(NBOOT)
    for s in range(0, NBOOT, 500):
        m = min(500, NBOOT - s)
        acc = np.zeros(m)
        for c in ORDER:
            d = per_cell_d[c]; n = len(d)
            acc += d[rng.integers(0, n, size=(m, n))].mean(axis=1)
        b[s:s + m] = acc / len(ORDER)
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    half = (hi - lo) / 2.0

    loo = {}
    for c in ORDER:
        keys = [k for k in ORDER if k != c]
        loo[c] = R(macro(am, keys) - macro(d32, keys))
    vals = list(loo.values())

    per_cell = {}
    for c in ORDER:
        cur = float(am[c].mean())
        ceil2 = rows[c]["ceilings"]["item_oracle_7B_or_32Bdirect"]["acc"]
        ceil3 = rows[c]["ceilings"]["item_oracle_7B_or_32Bdirect_or_32Breasoning"]["acc"]
        best = ceil3
        if c in OPEN:
            best = max(best, rows[c]["ceilings"]["item_oracle_pool_at8_or_32Bdirect"]["acc"])
        realized = float((am[c] - d32[c]).mean())
        p10 = rows[c]["contingency_7Bdirect_vs_32Bdirect"]["only_7B_right"]
        per_cell[c] = dict(
            cell_gain_needed_for_plus0p01_macro=R(0.01 / W),
            cell_gain_needed_alone_for_a_significant_win=R((half - (cur_am - cur_32)) / W),
            current_method_acc=R(cur),
            current_gain_over_always_32B_direct=R(realized),
            current_macro_contribution=R(realized * W, 5),
            p10_ceiling_on_that_gain=p10,
            conversion_rate_of_p10=(R(realized / p10, 4) if p10 > 0 else None),
            max_cell_acc_2way_item_oracle=ceil2,
            max_cell_acc_best_available_oracle=R(best),
            remaining_cell_headroom_over_current_method=R(best - cur),
            max_macro_contribution_if_that_headroom_were_fully_harvested=R((best - cur) * W))
    return dict(
        weight_per_cell=W,
        current=dict(macro_accuracy_max_veto=R(cur_am), macro_compute_lean=R(cur_cl),
                     macro_always_32b_direct=R(cur_32),
                     vs_direct_delta=R(cur_am - cur_32), vs_direct_ci95=[R(lo), R(hi)],
                     vs_direct_sig=bool(lo > 0 or hi < 0)),
        what_a_win_requires=dict(
            note="the CI half-width is ~stable in this range, so a macro delta above it clears zero",
            current_ci_half_width=R(half),
            macro_delta_needed_for_a_significant_win=R(half),
            summed_per_cell_gain_needed=R(half * len(ORDER)),
            macro_delta_needed_for_a_comfortable_win_1p5x_halfwidth=R(1.5 * half),
            summed_per_cell_gain_needed_comfortable=R(1.5 * half * len(ORDER))),
        macro_moves_by_this_much_per_unit_of_cell_accuracy=W,
        leave_one_cell_out_vs_direct_delta=dict(
            per_dropped_cell=loo, range=[R(min(vals)), R(max(vals))],
            load_bearing_cell=min(loo, key=lambda k: loo[k]),
            cell_holding_it_back=max(loo, key=lambda k: loo[k])),
        per_cell=per_cell)


# =====================================================================================================
# 7.  where the current +0.0008 actually comes from, and a ranked verdict per cell
# =====================================================================================================
def attribution_and_ranking(V, rows, ident, f8, sens):
    W = 1.0 / len(ORDER)
    attrib = {}
    for c in ORDER:
        g = float((V[c]["method_accuracy_max_veto"] - V[c]["always_32b_direct"]).mean())
        attrib[c] = dict(cell_gain=R(g), macro_contribution=R(g * W, 5))
    tot = sum(attrib[c]["macro_contribution"] for c in ORDER)
    attrib["_total_macro_delta_vs_direct"] = R(tot, 5)
    attrib["_mcq_half"] = R(sum(attrib[c]["macro_contribution"] for c in MCQ), 5)
    attrib["_open_half"] = R(sum(attrib[c]["macro_contribution"] for c in OPEN), 5)

    rank = []
    for c in ORDER:
        pcs = sens["per_cell"][c]
        best_sig = None
        for s, v in ident[c]["signals"].items():
            cand = v["crossfit_keep7_gain"]["gain"]
            if best_sig is None or cand > best_sig[1]:
                best_sig = (s, cand, v["auroc_7Bwins_among_disagreements"],
                            v["crossfit_keep7_gain"]["ci"]["sig"])
        rank.append(dict(
            cell=c, weight=W,
            current_macro_contribution=attrib[c]["macro_contribution"],
            p10_7B_wins=rows[c]["contingency_7Bdirect_vs_32Bdirect"]["only_7B_right"],
            p10_macro_value_if_perfectly_identified=R(
                rows[c]["contingency_7Bdirect_vs_32Bdirect"]["only_7B_right"] * W, 5),
            best_cheap_signal=best_sig[0],
            best_signal_auroc_on_disagreements=best_sig[2],
            best_signal_crossfit_cell_gain=best_sig[1],
            best_signal_crossfit_is_significant=best_sig[3],
            best_signal_macro_value=R(best_sig[1] * W, 5),
            headroom_to_best_oracle=pcs["remaining_cell_headroom_over_current_method"],
            macro_value_of_that_headroom=pcs["max_macro_contribution_if_that_headroom_were_fully_harvested"]))
    rank.sort(key=lambda r: -r["best_signal_macro_value"])
    reachable = sum(max(r["best_signal_macro_value"], 0.0) for r in rank)

    # ---- the eval-visible per-cell policy ceiling: pick, PER CELL, whichever of the three shipped
    #      operating points and the baseline scores highest ON THE EVAL ITSELF.  This is a DIAGNOSTIC
    #      upper bound on "tune the per-cell policy", not a deployable result -- it uses eval labels.
    opts = ["method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]
    best_gain, choice = {}, {}
    for c in ORDER:
        g = {o: float((V[c][o] - V[c]["always_32b_direct"]).mean()) for o in opts}
        g["always_32b_direct"] = 0.0
        k = max(g, key=lambda kk: g[kk])
        best_gain[c] = g[k]; choice[c] = k
    ceil_policy = sum(best_gain.values()) * W

    return dict(
        attribution_of_the_current_vs_direct_delta=attrib,
        ranked_by_demonstrated_cheap_signal_value=rank,
        summed_macro_delta_if_every_cell_ran_its_best_cross_fit_keep7B_rule=R(reachable, 5),
        WARNING_on_that_sum=("DIAGNOSTIC ONLY. The signal is chosen per cell AFTER seeing its cross-fit "
                             "gain on the eval cells, i.e. with eval visibility across 2-4 signals x 8 "
                             "cells. It is an optimistic upper bound on the keep-cheap lever family, "
                             "not a deployable number."),
        eval_visible_best_per_cell_policy_ceiling=dict(
            per_cell_choice=choice,
            per_cell_gain={c: R(best_gain[c]) for c in ORDER},
            macro_delta=R(ceil_policy, 5),
            WARNING="DIAGNOSTIC ONLY -- the per-cell winner is chosen on the eval itself.",
            note="the best the method can do by re-assigning, per cell, which of the three ALREADY "
                 "MEASURED operating points (or the plain 32B-direct baseline) it deploys"),
        macro_delta_needed_for_a_significant_win=sens["what_a_win_requires"][
            "macro_delta_needed_for_a_significant_win"],
        verdict=("The best single-threshold keep-7B rule available on each cell, cross-fit, sums to the "
                 "value above; the eval-visible best-per-cell reassignment of the shipped operating "
                 "points gives the other value. Compare BOTH with the macro delta a significant win "
                 "needs -- if neither clears it, no rearrangement of what is already measured wins."))


# =====================================================================================================
def main():
    rng = np.random.default_rng(SEED)
    V = load_vectors()
    nt = null_test(V)
    print(f"[NULL TEST] per-cell/macro accuracies vs cascade_selector_rerun_2026-08-05.json (disjoint): "
          f"{nt['fields_compared']} fields, max abs deviation {nt['max_abs_deviation_percell']} / "
          f"{nt['max_abs_deviation_macro']}  -> {'PASS' if nt['passed'] else 'FAIL'}")
    SIG = mcq_signals(V)
    P = open_pools(V)
    ont = open_bar_null_test(P)
    print(f"[NULL TEST] open-text bar vs genframe_data FROZEN: max abs deviation {ont['max_abs_deviation']} "
          f"-> {'PASS' if ont['passed'] else 'FAIL'}")

    rows = per_cell_table(V, SIG, P, rng)
    ident = identifiability(V, SIG, P, rng)
    f8 = f8_all_mcq(V, rng)
    for c in MCQ:                      # fill the accuracy-max cost on PMC from the measured veto rate
        if rows[c]["cost_flopeq"]["method_accuracy_max_veto"] is None:
            rows[c]["cost_flopeq"]["method_accuracy_max_veto"] = f8[c]["cost_flopeq"]
    bon = mcq_bestofn_luckfloor()
    sens = sensitivity(V, rows, rng)
    rank = attribution_and_ranking(V, rows, ident, f8, sens)

    out = dict(
        title="SCOUT A -- per-cell headroom accounting for the MACRO (8-cell, Variant B) headline",
        date="2026-08-10",
        reproduce="python3 src/cascade_methods/headroom_percell.py",
        no_gpu=True, no_new_inference=True, no_fabricated_numbers=True,
        convention=("MACRO, equal weight per reporting cell, Variant B (MMMU excluded), 8 cells, 1/8 each, "
                    "n=42,224; CLEAN disjoint open-text verifier (ckpts/train/lora_verifier_disjoint)"),
        n_bootstrap=NBOOT, seed=SEED,
        caveats=[
            "TOKEN AUDIT, per the standing rule: the `always_32b_reasoning` vector on the four "
            "closed/MCQ cells comes from the dumps CLAUDE.md hole 1 flags as averaging 3-4 generated "
            "tokens (PMC test_2 / SLAKE-closed / VQA-RAD-closed), i.e. NOT genuine reasoning runs. "
            "PATH_VQA_closed has no reasoning dump at all, so okT == ok32 there by construction and its "
            "3-way oracle equals its 2-way oracle. Every ceiling in this file that involves "
            "32B-reasoning on a CLOSED cell inherits that defect and must not be cited as evidence "
            "about reasoning. The three open cells' reasoning vectors ARE measured judged dumps "
            "(opentext_32b_think_full), and MedXpert uses the genuine lingshu32b_reason run.",
            "ALL cost numbers here are per-cell FLOP-eq under the paper's as-charged constants "
            "(one 32B pass = 4.57). They are cell-local, so they may be macro-averaged; they are NEVER "
            "to be paired with a sample-weighted accuracy.",
            "The cross-fit keep-7B rules and the F8 veto sweep cross-fit 5-fold WITHIN each eval cell, "
            "which is the deployed cascade's own protocol -- but the DECISION to apply a lever to a "
            "cell, and the CHOICE of signal per cell, are made here with eval visibility. Both are "
            "labelled DIAGNOSTIC and neither is a deployable result.",
            "The guardrail's resolution is poor on VQA_RAD_open (n=200) and VQA_RAD_closed (n=251); "
            "single-cell flags on those two are within seed/sampling noise, not decisive.",
        ],
        null_tests=dict(per_cell_accuracies=nt, open_text_bar=ont),
        per_cell=rows,
        identifiability_of_the_7B_wins_subset=ident,
        certified_veto_applied_to_every_mcq_cell=f8,
        mcq_best_of_8_and_its_luck_floor=bon,
        macro_sensitivity=sens,
        where_the_current_delta_comes_from_and_ranked_verdict=rank,
        key_findings=[
            "100% of the current +0.0008 macro delta over always-32B-direct is the PMC-VQA certified "
            "veto (+0.0095 cell -> +0.00119 macro) plus PathVQA-open (+0.0087 -> +0.00108); the other "
            "two open cells are a net -0.00144 macro drag and the other four MCQ cells contribute "
            "EXACTLY ZERO because the method is literally always-32B-direct there.",
            "The certified veto (beat32b_more.f8_veto, the shipped accuracy-max MCQ lever) run "
            "unmodified on the other four MCQ cells certifies ZERO items on all four: no 7B-confidence "
            "bin anywhere has a Wilson lower bound on 7B precision that reaches the 32B's accuracy in "
            "that bin. The incremental macro value of extending the shipped lever to the rest of the "
            "MCQ half is +0.0000.",
            "The eval-visible best-per-cell reassignment of the three shipped operating points plus the "
            "baseline reaches only +0.00277 macro, against the ~+0.0029 the CI needs. Nothing that is "
            "already measured can be rearranged into a significant win.",
            "MedXpertQA-MM has the largest 2-way p10 (0.1160, macro value +0.0145) but its "
            "7B-wins-among-disagreements AUROC is 0.4877 (chance) from both cheap signals, and its "
            "7B/32B item oracle (0.4225) sits close to the independent-errors floor (0.4879). It is "
            "structurally immovable with anything cheap.",
            "The open-text arithmetic in circulation is SAMPLE-WEIGHTED and the headline is MACRO. "
            "Sample-weighted the open pool's 32B-direct is 0.5169 and oracle@8 is 0.6260; MACRO over the "
            "three open cells they are 0.5982 and 0.6752. Coverage headroom over the 32B on the open "
            "half is +0.0770 macro-weighted, not +0.109.",
            "selected == sel_eff * oracle@8 EXACTLY (verified to 1e-12). The identity "
            "'selected = greedy + sel_eff*(oracle-greedy)' is wrong: it predicts 0.5863 where the "
            "measured value is 0.4853. The marginal slope (+sel_eff per unit of oracle@8) is the same, "
            "so 'coverage has a multiplier' survives; the LEVEL projections do not.",
        ],
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", os.path.relpath(OUT, ROOT))
    return out


if __name__ == "__main__":
    main()
