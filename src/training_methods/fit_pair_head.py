#!/usr/bin/env python3
"""fit_pair_head.py -- STAGE 2: evaluate the PRE-REGISTERED pairwise contrast head.

Reads the configuration chosen by pairhead_cv.py (train-split CV only), refits it on the full
disjoint training pool at >=10 seeds, scores all within-question candidate pairs of the 2345
eval questions from CACHED generator-frame vectors, aggregates to a ranking, and reports the
frozen endpoint against every published control.

  python3 src/training_methods/fit_pair_head.py \
      --pre data/verifarch/pairhead_cv.json \
      --out results/cascade_methods/artifacts/verifarch_pairhead_2026-08-04.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genframe_data as G
import pairhead_lib as P

R6 = lambda x: (round(float(x), 6) if isinstance(x, (int, float, np.floating)) else x)


def pack(r, extra=None):
    o = {"sel_eff": R6(r["sel_eff"]), "acc": R6(r["acc"]),
         "per_ds": {d: R6(r["per_ds"][d]["sel_eff"]) for d in G.EVAL_DS},
         "contested_sel_eff": R6(r["contested"]["sel_eff"]),
         "contested_n": r["contested"]["n"]}
    if extra:
        o.update(extra)
    return o


def boot(a, b, r, nboot, mask=None):
    x = G.paired_bootstrap(a, b, rec=r["rec"], nboot=nboot, mask=mask)
    return {"d_sel_eff": R6(x["d_sel_eff"]), "ci": [R6(v) for v in x["d_sel_eff_ci"]],
            "d_acc": R6(x["d_acc"]), "acc_ci": [R6(v) for v in x["d_acc_ci"]],
            "n_stratum": x["n_stratum"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="data/verifarch/pairhead_cv.json")
    ap.add_argument("--ptw", default="data/verifarch/pointwise_seeds_gpu.json")
    ap.add_argument("--ptw_scores", default="data/verifarch/pointwise_seed_scores_gpu.npy")
    ap.add_argument("--out", default="results/cascade_methods/artifacts/verifarch_pairhead_2026-08-04.json")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--nboot", type=int, default=10000)
    A = ap.parse_args()
    t0 = time.time()
    items = G.load_items()
    art = {"what": "A PAIRWISE CONTRAST HEAD g(h_i,h_j) over cached generator-frame hidden states: "
                   "head-to-head comparison of best-of-8 candidates at zero extra forward passes, "
                   "because the per-candidate vectors are the ones the deployed pointwise head "
                   "already computes.",
           "date": "2026-08-05", "nboot": A.nboot, "device": str(P.DEV)}

    # ------------------------------------------------------------------ 1. NULL TEST
    nt = G.null_test()
    art["null_test"] = {"protocol": "this harness pointed at the incumbent's own stored per-candidate "
                                    "scores in ckpts/train/lora_verifier_disjoint/transfer_dump_*.json; "
                                    "pick = argmax over the 8 slots, first-index tie-break.",
                        "measured": {k: R6(v) if not isinstance(v, dict) else {kk: R6(vv) for kk, vv in v.items()}
                                     for k, v in nt["measured"].items()},
                        "published": nt["published"],
                        "abs_deviation": {k: float(f"{v:.3g}") for k, v in nt["abs_deviation"].items()},
                        "max_abs_deviation": float(f"{nt['max_abs_deviation']:.3g}"),
                        "pass": nt["pass"]}
    print("NULL TEST pass =", nt["pass"], "max abs dev =", nt["max_abs_deviation"], flush=True)
    assert nt["pass"], "NULL TEST FAILED -- stop"

    # ------------------------------------------------------------------ 2. DISJOINTNESS
    art["disjointness"] = P.independent_disjointness()
    print("DISJOINT:", art["disjointness"]["verdict"], flush=True)

    # ------------------------------------------------------------------ 3. controls
    inc = G.sel_eff(G.incumbent_scores(), items)
    ctl = G.control_scores(items)
    sc = G.sel_eff(ctl["self_consistency"], items)
    rp = G.random_pick(items)
    art["controls"] = {
        "greedy_acc": R6(inc["greedy"]), "oracle@8": R6(inc["oracle"]),
        "random_pick": {"acc": R6(rp["acc"]), "sel_eff": R6(rp["sel_eff"])},
        "self_consistency": pack(sc),
        "incumbent_lora_verifier_THE_BAR": pack(inc, {"cand_auroc": R6(G.cand_auroc(G.incumbent_scores(), items))}),
        "strata": {"n_items": inc["n"], "n_recoverable": inc["n_recoverable"],
                   "n_contested_recoverable": inc["contested"]["n"],
                   "n_unanimous_recoverable": inc["contested"]["unanimous_n"],
                   "note": "the unanimous stratum is scored 1.0 by every selector by construction"}}

    # ------------------------------------------------------------------ 4. pointwise bar, 12 seeds
    ptw = json.load(open(os.path.join(G.ROOT, A.ptw)))
    Sp = np.load(os.path.join(G.ROOT, A.ptw_scores))
    kp = [tuple(k) for k in ptw["keys"]]
    Zp = (Sp - Sp.mean(1, keepdims=True)) / (Sp.std(1, keepdims=True) + 1e-12)
    ptw_ens = {kp[i]: float(v) for i, v in enumerate(Zp.mean(0))}
    r_ptw = G.sel_eff(ptw_ens, items)
    art["pointwise_head_bar"] = {
        "config": ptw["config"], "n_seeds": ptw["seed_spread"]["n"],
        "seed_spread_sel_eff": {k: R6(v) for k, v in ptw["seed_spread"].items() if k != "n"},
        "seed0_vs_published": {k: R6(v) for k, v in
                               (ptw.get("seed0_vs_published") or ptw.get("seed0_reproduction")).items()},
        "numerics_note": ptw.get("numerics_note", ""),
        "per_seed_sel_eff": [R6(ptw["seeds"][str(s)]["sel_eff"]) for s in range(ptw["seed_spread"]["n"])],
        "seed_ensemble_zmean": pack(r_ptw, {"cand_auroc": R6(G.cand_auroc(ptw_ens, items)),
                                            "vs_incumbent": boot(r_ptw["got"], inc["got"], r_ptw, A.nboot),
                                            "guardrail_clean": G.guardrail_clean(r_ptw, inc)}),
        "note": "the published 0.795640 is a single seed-0 fit of this same config; the seed spread "
                "below is what a single seed hides."}

    # ------------------------------------------------------------------ 5. pre-registered pair head
    pre = json.load(open(os.path.join(G.ROOT, A.pre)))
    CFG, AGG = pre["PREREGISTERED"]["config"], pre["PREREGISTERED"]["aggregation"]
    art["preregistration"] = {"source": A.pre, "config": CFG, "aggregation": AGG,
                              "cv_sel_eff_train_only": R6(pre["PREREGISTERED"]["cv_sel_eff"]),
                              "cv_protocol": pre["cv_protocol"],
                              "train_pairs": pre["train"]}
    print("PREREGISTERED:", CFG, AGG, flush=True)

    tr = G.load_candidates("train", mode="generator", layers=CFG["layers"],
                           pooling=("last", "span"), order="concat")
    ev = G.load_candidates("eval", mode="generator", layers=CFG["layers"],
                           pooling=("last", "span"), order="concat")
    Xtr = P.base_matrix(tr, CFG["layers"], CFG["pooling"])
    Xev = P.base_matrix(ev, CFG["layers"], CFG["pooling"])
    pos, neg, pf = P.train_pairs(tr)
    fit_rows = np.unique(np.concatenate([pos, neg]))
    mu, sd = P.standardize(Xtr, fit_rows)
    Xtr_d, Xev_d = P.to_dev(Xtr, mu, sd), P.to_dev(Xev, mu, sd)
    del Xtr, Xev

    groups = [[c.row for c in q.cands] for q in ev.questions]
    sizes = [len(g) for g in groups]
    A_, B_, Q_, IA_, IB_ = P.build_query_pairs(groups)
    rev0 = {(int(A_[t]), int(B_[t])): t for t in range(len(A_))}
    print(f"eval: {len(groups)} questions, {sum(sizes)} distinct candidates, "
          f"{len(A_)//2} unordered pairs, mean {np.mean([k*(k-1)/2 for k in sizes]):.2f}/question", flush=True)

    def score_from_mats(mats, agg):
        out = {}
        for qi, q in enumerate(ev.questions):
            s = P.aggregate(mats[qi], agg)
            for ci, c in enumerate(q.cands):
                out[(q.ds, q.idx, c.na)] = float(s[ci])
        return out

    per_seed, GL = [], []
    for s in range(A.seeds):
        t = time.time()
        m = P.fit_pair_head(Xtr_d, pos, neg, CFG, seed=s)
        gl = P.pair_logits(m, Xev_d, A_, B_, CFG["antisym"])
        GL.append(gl / (gl.std() + 1e-12))
        mats = P.logits_to_matrices(gl, Q_, IA_, IB_, sizes)
        r = G.sel_eff(score_from_mats(mats, AGG), items)
        per_seed.append(r)
        print(f"seed {s}: sel_eff={r['sel_eff']:.6f} acc={r['acc']:.6f} "
              f"contested={r['contested']['sel_eff']:.6f} ({time.time()-t:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()

    v = np.array([r["sel_eff"] for r in per_seed])
    art["pair_head_seeds"] = {
        "n_seeds": A.seeds, "aggregation": AGG,
        "per_seed_sel_eff": [R6(x) for x in v],
        "mean": R6(v.mean()), "sd": R6(v.std(ddof=1)), "min": R6(v.min()), "max": R6(v.max()),
        "per_seed_contested": [R6(r["contested"]["sel_eff"]) for r in per_seed]}

    # ---- seed ENSEMBLE (the deployable number): mean of scale-normalized pair logits
    glm = np.mean(GL, 0)
    mats_ens = P.logits_to_matrices(glm, Q_, IA_, IB_, sizes)
    ens_scores = {a: score_from_mats(mats_ens, a) for a in P.AGGS}
    head = ens_scores[AGG]
    r_head = G.sel_eff(head, items)

    art["HEADLINE_pair_head_seed_ensemble"] = pack(r_head, {
        "aggregation": AGG, "n_seeds": A.seeds,
        "cand_auroc": R6(G.cand_auroc(head, items)),
        "vs_incumbent_0p775204": boot(r_head["got"], inc["got"], r_head, A.nboot),
        "vs_pointwise_head_ensemble": boot(r_head["got"], r_ptw["got"], r_head, A.nboot),
        "vs_self_consistency": boot(r_head["got"], sc["got"], r_head, A.nboot),
        "contested_vs_incumbent": boot(r_head["got"], inc["got"], r_head, A.nboot, mask=r_head["contested_mask"]),
        "contested_vs_pointwise_head_ensemble": boot(r_head["got"], r_ptw["got"], r_head, A.nboot,
                                                     mask=r_head["contested_mask"]),
        "guardrail_clean_vs_incumbent": G.guardrail_clean(r_head, inc),
        "guardrail_clean_vs_pointwise": G.guardrail_clean(r_head, r_ptw),
        "ensemble_rule": "per-seed pair logits divided by their global sd (the arch-antisymmetric "
                         "logit vector has mean exactly 0), averaged over seeds, then aggregated."})

    # ---- every aggregation rule on the ensemble (DIAGNOSTIC: only AGG was pre-registered)
    aggtab = {}
    for a in P.AGGS:
        ra = G.sel_eff(ens_scores[a], items)
        aggtab[a] = pack(ra, {"vs_incumbent": boot(ra["got"], inc["got"], ra, A.nboot),
                              "preregistered": bool(a == AGG),
                              "comparisons_per_question": (R6(np.mean([P.n_knockout_comparisons(k) for k in sizes]))
                                                           if a == "knockout"
                                                           else R6(np.mean([k * (k - 1) / 2 for k in sizes])))})
    art["aggregation_table"] = aggtab

    # ---- THE DECISIVE MECHANISTIC TEST: how much of the learned comparison is ADDITIVE?
    # Any antisymmetric G decomposes uniquely as G = (theta_i - theta_j) + Resid with
    # theta_i = mean_j G[i,j] and Resid row-summing to zero. The additive part is a POINTWISE
    # scorer (aggregating it IS the 'logit_sum' rule). Whatever the head learned that a
    # pointwise head could not express lives entirely in Resid. If Resid carries no selection
    # signal, "comparative" is a costume and the pointwise bar is the ceiling of this feature set.
    add_var, res_var, sc_add, sc_res = 0.0, 0.0, {}, {}
    for qi, q in enumerate(ev.questions):
        Gm, k = mats_ens[qi], len(q.cands)
        th = Gm.mean(1) if k > 1 else np.zeros(1)
        Add = th[:, None] - th[None, :]
        Res = Gm - Add
        np.fill_diagonal(Res, 0.0)
        off = ~np.eye(k, dtype=bool)
        add_var += float((Add[off] ** 2).sum()); res_var += float((Res[off] ** 2).sum())
        rs = Res.sum(1) if k > 1 else np.zeros(1)          # identically 0 by construction
        rn = (np.abs(Res)[off].sum() if k > 1 else 0.0)
        for ci, c in enumerate(q.cands):
            sc_add[(q.ds, q.idx, c.na)] = float(th[ci])
            sc_res[(q.ds, q.idx, c.na)] = float(np.sign(Res[ci]).sum() if k > 1 else 0.0)
        _ = rs, rn
    r_add = G.sel_eff(sc_add, items)
    r_res = G.sel_eff(sc_res, items)
    art["additive_decomposition"] = {
        "definition": "G[i,j] = (theta_i - theta_j) + Resid, theta_i = mean_j G[i,j]; the first term "
                      "is exactly a pointwise scorer, the second is everything a pointwise head "
                      "cannot express.",
        "variance_share_additive": R6(add_var / max(add_var + res_var, 1e-12)),
        "variance_share_residual": R6(res_var / max(add_var + res_var, 1e-12)),
        "additive_part_alone": pack(r_add, {"vs_incumbent": boot(r_add["got"], inc["got"], r_add, A.nboot),
                                            "note": "identical to the 'logit_sum' aggregation up to a "
                                                    "positive per-question scale factor k"}),
        "residual_part_alone": pack(r_res, {"note": "sign-sum of the non-additive residual; a random-pick "
                                                    "level number here means the comparison carries no "
                                                    "selection signal beyond a pointwise ranking",
                                            "random_pick_sel_eff": R6(rp["sel_eff"])})}
    print(f"  additive share={art['additive_decomposition']['variance_share_additive']:.4f} "
          f"additive-only sel_eff={r_add['sel_eff']:.6f} residual-only={r_res['sel_eff']:.6f}", flush=True)

    # ---- IN-DOMAIN CONTAMINATED DIAGNOSTIC: is the negative about TRANSFER, or is the
    # non-additive structure simply not present in these features?  Fit the same pair head
    # INSIDE eval with image-grouped 5-fold CV (shares images and question distribution with
    # the test set -> optimistic bound, NEVER deployable) and re-measure the additive share.
    efold = G.eval_folds(5, items)
    ev_pos, ev_neg, ev_qi = [], [], []
    for qi, q in enumerate(ev.questions):
        p = [c.row for c in q.cands if c.y == 1]
        n = [c.row for c in q.cands if c.y == 0]
        for a in p:
            for b in n:
                ev_pos.append(a); ev_neg.append(b); ev_qi.append(qi)
    ev_pos, ev_neg, ev_qi = np.array(ev_pos), np.array(ev_neg), np.array(ev_qi)
    pfold = efold[ev_qi]
    glid = np.zeros(len(A_))
    for f in range(5):
        tr_m = pfold != f
        mm = P.fit_pair_head(Xev_d, ev_pos[tr_m], ev_neg[tr_m], CFG, seed=0)
        te = np.where(efold[Q_] == f)[0]
        glid[te] = P.pair_logits(mm, Xev_d, A_[te], B_[te], CFG["antisym"])
        del mm
    torch.cuda.empty_cache()
    mats_id = P.logits_to_matrices(glid, Q_, IA_, IB_, sizes)
    av, rv_ = 0.0, 0.0
    for qi, q in enumerate(ev.questions):
        Gm, k = mats_id[qi], len(q.cands)
        if k < 2:
            continue
        th = Gm.mean(1)
        Add = th[:, None] - th[None, :]
        Res = Gm - Add
        off = ~np.eye(k, dtype=bool)
        av += float((Add[off] ** 2).sum()); rv_ += float((Res[off] ** 2).sum())
    r_id = G.sel_eff(score_from_mats(mats_id, AGG), items)
    art["indomain_diagnostic_CONTAMINATED"] = pack(r_id, {
        "protocol": "the SAME pre-registered pair head fit INSIDE the eval set with 5-fold "
                    "image-grouped CV (folds from genframe_data.eval_folds), single seed. Shares "
                    "images and question distribution with the test set: OPTIMISTIC BOUND, not "
                    "comparable to the clean arms and NOT deployable.",
        "n_eval_pairs": int(len(ev_pos)),
        "variance_share_additive": R6(av / max(av + rv_, 1e-12)),
        "why": "separates 'the comparative structure does not transfer from the disjoint pool' from "
               "'the comparative structure is not in these features at all'."})
    print(f"  IN-DOMAIN (contaminated) sel_eff={r_id['sel_eff']:.6f} "
          f"additive share={art['indomain_diagnostic_CONTAMINATED']['variance_share_additive']:.4f}",
          flush=True)

    # ---- DIAGNOSTIC variant table: pair encoding x antisymmetry (only the CV winner is
    # pre-registered; the rest are reported so the design space is on record, and to measure
    # position bias, which killed naive pairwise verifiers before).
    var = {}
    for inp in ["concat", "diff", "full"]:
        for anti in ["arch", "augment"]:
            for hid in ([CFG["hidden"], 0] if inp == "diff" and anti == "arch" else [CFG["hidden"]]):
                cfg = {**CFG, "inp": inp, "antisym": anti, "hidden": hid}
                tag = f"{inp}/{anti}/h{hid}"
                gls = []
                for s in range(3):
                    mm = P.fit_pair_head(Xtr_d, pos, neg, cfg, seed=s)
                    g = P.pair_logits(mm, Xev_d, A_, B_, anti)
                    gls.append(g / (g.std() + 1e-12))
                    del mm
                torch.cuda.empty_cache()
                gm = np.mean(gls, 0)
                mt = P.logits_to_matrices(gm, Q_, IA_, IB_, sizes)
                rv = G.sel_eff(score_from_mats(mt, AGG), items)
                asy = [abs(gm[t] + gm[rev0[(int(B_[t]), int(A_[t]))]]) / 2
                       for t in range(0, len(A_), 7) if (int(B_[t]), int(A_[t])) in rev0]
                var[tag] = pack(rv, {
                    "n_seeds": 3, "preregistered": bool(inp == CFG["inp"] and anti == CFG["antisym"]
                                                        and hid == CFG["hidden"]),
                    "antisymmetry_violation_mean_abs": R6(np.mean(asy)),
                    "vs_pointwise_ens": boot(rv["got"], r_ptw["got"], rv, A.nboot)})
                print(f"  VARIANT {tag}: sel_eff={rv['sel_eff']:.6f} "
                      f"asym_viol={var[tag]['antisymmetry_violation_mean_abs']:.4f}", flush=True)
    var["_note"] = ("3 seeds each, seed-ensembled, at the pre-registered layer/pooling/capacity and "
                    "aggregation. DIAGNOSTIC: only the pre-registered cell is a headline. "
                    "'diff/arch/h0' is the DEGENERACY CONTROL -- linear in h_i-h_j is algebraically "
                    "a pointwise scorer w.h_i - w.h_j, so it must behave like one. "
                    "'antisymmetry_violation' = mean |g(i,j)+g(j,i)|/2, exactly 0 for antisym='arch' "
                    "by construction and a direct measure of POSITION BIAS for antisym='augment'.")
    art["variant_table_DIAGNOSTIC"] = var

    # ------------------------------------------------------------------ 6. fusions
    fus = {}
    combos = {
        "FUSE_pair+incumbent": [head, G.incumbent_scores()],
        "FUSE_pair+pointwise": [head, ptw_ens],
        "FUSE_pair+pointwise+incumbent": [head, ptw_ens, G.incumbent_scores()],
        "REF_FUSE_pointwise+incumbent": [ptw_ens, G.incumbent_scores()],
    }
    ref_fuse_got = None
    for name, parts in combos.items():
        f = G.rank_fuse(*parts, items=items, ranker=G.rank_avg)
        rf = G.sel_eff(f, items)
        if name == "REF_FUSE_pointwise+incumbent":
            ref_fuse_got = rf["got"]
        fus[name] = pack(rf, {"ranker": "rank_avg",
                              "vs_incumbent": boot(rf["got"], inc["got"], rf, A.nboot),
                              "vs_pointwise_ens": boot(rf["got"], r_ptw["got"], rf, A.nboot),
                              "contested_vs_incumbent": boot(rf["got"], inc["got"], rf, A.nboot,
                                                             mask=rf["contested_mask"]),
                              "guardrail_clean_vs_incumbent": G.guardrail_clean(rf, inc)})
        print(f"  {name}: sel_eff={rf['sel_eff']:.6f} contested={rf['contested']['sel_eff']:.6f}", flush=True)
    for name in combos:
        if name.startswith("FUSE"):
            rf = G.sel_eff(G.rank_fuse(*combos[name], items=items, ranker=G.rank_avg), items)
            fus[name]["vs_seed_ensembled_pointwise_plus_incumbent"] = boot(rf["got"], ref_fuse_got, rf, A.nboot)
    # the DEPLOYED fusion (0.806540) is single-seed pointwise + incumbent; give its seed
    # distribution so "seed-ensembling the pointwise head improves the deployed fusion" is
    # measured, not asserted.
    sf = []
    for s in range(Sp.shape[0]):
        one = {kp[i]: float(Sp[s, i]) for i in range(len(kp))}
        rs = G.sel_eff(G.rank_fuse(one, G.incumbent_scores(), items=items, ranker=G.rank_avg), items)
        sf.append(rs["sel_eff"])
    sf = np.array(sf)
    fus["_deployed_fusion_single_seed_distribution"] = {
        "per_seed_sel_eff": [R6(x) for x in sf], "mean": R6(sf.mean()), "sd": R6(sf.std(ddof=1)),
        "min": R6(sf.min()), "max": R6(sf.max()),
        "published_deployed_fusion": 0.806540,
        "note": "rank_avg fusion of ONE pointwise seed with the incumbent, the published deployed "
                "recipe, repeated over 12 seeds. Compare with REF_FUSE_pointwise+incumbent, which "
                "fuses the 12-seed ENSEMBLE instead."}
    art["fusions"] = fus

    # ------------------------------------------------------------------ 7. mechanism diagnostics
    diag = {}
    # (a) antisymmetry of the deployed head (exactly 0 by construction when antisym='arch')
    asym = []
    for t in range(0, len(A_), 7):
        u = rev0.get((int(B_[t]), int(A_[t])))
        if u is not None:
            asym.append(abs(glm[t] + glm[u]))
    diag["antisymmetry_residual_mean_abs"] = R6(np.mean(asym) if asym else 0.0)
    # (b) agreement with the pointwise head
    ph = np.array([head[(q.ds, q.idx, c.na)] for q in ev.questions for c in q.cands])
    pp = np.array([ptw_ens[(q.ds, q.idx, c.na)] for q in ev.questions for c in q.cands])
    pi = []
    for q in ev.questions:
        s8 = np.array(q.inc_scores, float)
        for c in q.cands:
            pi.append(float(np.mean([s8[k] for k in c.slots])))
    pi = np.array(pi)
    rk = lambda x: np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
    diag["spearman_pair_vs_pointwise"] = R6(np.corrcoef(rk(ph), rk(pp))[0, 1])
    diag["spearman_pair_vs_incumbent"] = R6(np.corrcoef(rk(ph), rk(pi))[0, 1])
    diag["same_pick_rate_pair_vs_pointwise"] = R6(np.mean(r_head["picks"] == r_ptw["picks"]))
    diag["same_pick_rate_pair_vs_incumbent"] = R6(np.mean(r_head["picks"] == inc["picks"]))
    for nm, other in [("pointwise", r_ptw), ("incumbent", inc)]:
        po = np.maximum(r_head["got"], other["got"])
        diag[f"pair_oracle_ceiling_vs_{nm}"] = R6(po[r_head["rec"] == 1].mean())
    art["diagnostics"] = diag

    # ------------------------------------------------------------------ 8. cost
    art["cost"] = {
        "extra_forward_passes_per_question_beyond_the_8_generations": 0,
        "explanation": "the per-candidate generator-frame vector is produced by the SAME forward "
                       "pass the deployed pointwise head already needs (1 per DISTINCT candidate, "
                       "mean 3.81 of 8 after dedup by normalized answer). The pairwise head adds "
                       "only MLP evaluations over those cached vectors.",
        "mlp_evaluations_per_question_round_robin": R6(np.mean([k * (k - 1) / 2 for k in sizes])),
        "mlp_evaluations_per_question_round_robin_max": int(max(k * (k - 1) / 2 for k in sizes)),
        "mlp_evaluations_per_question_knockout": R6(np.mean([P.n_knockout_comparisons(k) for k in sizes])),
        "ordered_forward_evals_per_comparison": 2 if CFG["antisym"] == "arch" else 1,
        "head_parameters": int(sum(p.numel() for p in P.PairHead(
            Xtr_d.shape[1], CFG["inp"], CFG["hidden"]).parameters())),
        "features_shared_with_deployed_pointwise_head": True,
        "contrast_with_the_shelved_pairwise_win": "the 2026-07 measured pairwise win "
            "(artifacts/pairwise_verifier_gpu.json) needed 28 REAL A-vs-B forward passes per "
            "question and was shelved for that reason; this head needs none.",
        "minutes_total": None}

    art["cost"]["minutes_total"] = round((time.time() - t0) / 60, 1)
    op = os.path.join(G.ROOT, A.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(art, open(op, "w"), indent=1, default=float)
    print("\nwrote", op, flush=True)
    print("HEADLINE sel_eff =", art["HEADLINE_pair_head_seed_ensemble"]["sel_eff"], flush=True)


if __name__ == "__main__":
    main()
