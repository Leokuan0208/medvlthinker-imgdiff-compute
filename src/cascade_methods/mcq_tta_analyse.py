#!/usr/bin/env python3
"""
mcq_tta_analyse.py -- ATTACK 2 analysis half: null test N3, the luck-floor control N4, the
aggregation rules, the cross-fit gate, the paired item-level bootstrap and the macro integration.

Every grader is MedEvalKit's OWN grader, imported read-only (judge_multi_choice / judge_judgement /
judge_close_end_vqa).  Verified 2026-08-10: those three functions reproduce the stored `correct`
field on 33430/33430 + 2000/2000 + 251/251 + 3362/3362 + 836/836 rows, so the TTA arm and the
deployed always-32B-direct vectors are graded by literally the same function.

    python3 src/cascade_methods/mcq_tta_analyse.py
"""
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mcq_tta as M  # noqa: E402

os.environ.setdefault("api_key", "none")
os.environ.setdefault("base_url", "http://localhost:1/v1")
os.environ.setdefault("judge_model", "none")
os.environ.setdefault("judge_model_type", "openai")
sys.path.insert(0, M.MEK)
from utils.utils import judge_multi_choice, judge_judgement, judge_close_end_vqa  # noqa: E402

ART = M.ART
STORED_ACC = {"PMC_VQA": 0.5518, "SLAKE_closed": 0.8589, "VQA_RAD_closed": 0.8526,
              "PATH_VQA_closed": 0.8891, "MedXpertQA-MM": 0.3065}
N3_TOL = 0.005


# ===============================================================================================
# loading
# ===============================================================================================
def load_rows(cell, stage="A"):
    p = os.path.join(M.CKPT, f"{cell}_stage{stage}.jsonl")
    by = {}
    if not os.path.exists(p):
        return by
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        by.setdefault(r["i"], {})[r["v"]] = r          # last write wins (resume-safe)
    return by


def option_logprobs(row, labels):
    """Renormalised log-probs over `labels` from the first generated token's top-20 logprobs.
    Missing labels get min(observed) - 1.0 (pre-registered floor).  Returns (vec, all_found)."""
    lp = row.get("first_logprobs") or {}
    got = {}
    for tok, v in lp.items():
        t = str(tok).strip().lower()
        if t in labels:
            got[t] = max(v, got.get(t, -1e9))
    if not got:
        return None, False
    floor = min(got.values()) - 1.0
    vec = np.array([got.get(l, floor) for l in labels], float)
    vec = vec - (np.max(vec) + math.log(np.sum(np.exp(vec - np.max(vec)))))   # log-softmax
    return vec, len(got) == len(labels)


def norm_free(resp):
    """SLAKE_closed normalisation: MedEvalKit's own parse_response, then the same stripping
    judge_close_end_vqa applies."""
    from utils.utils import parse_response
    return parse_response(str(resp)).replace("\n", "").replace(".", "").strip()


# ===============================================================================================
# per-cell aggregation
# ===============================================================================================
def aggregate_cell(cell, items, by, kmax=M.K):
    """Return dict of per-item arrays, all aligned to the cell's deployed vector index order."""
    stored = None
    n = len(items)
    out = dict(
        ok_id_raw=np.zeros(n), ok_id_lp=np.zeros(n),
        ok_lp={k: np.zeros(n) for k in range(1, kmax + 1)},
        ok_mv={k: np.zeros(n) for k in range(1, kmax + 1)},
        nviews=np.zeros(n, int), full_cover=np.zeros(n), gen_toks=np.zeros(n),
        agree_id=np.zeros(n), lp_per_view=[None] * n, decided=[None] * n)
    fmt = items[0]["fmt"]
    for j, it in enumerate(items):
        vs = by.get(it["i"], {})
        if 0 not in vs:
            out["nviews"][j] = 0
            continue
        views = [vs[k] for k in sorted(vs)]
        out["nviews"][j] = len(views)
        out["gen_toks"][j] = np.mean([v["gen_toks"] for v in views])
        r0 = views[0]
        if fmt == "mcq":
            nopt = it["choices"] and len(it["choices"])
            labels = [chr(ord("a") + i) for i in range(nopt)]
            out["ok_id_raw"][j] = judge_multi_choice(it["choices"], it["answer"], r0["response"])
            mats, cov = [], True
            for v in views:
                vec, full = option_logprobs(v, labels)
                cov = cov and full
                if vec is None:
                    vec = np.full(nopt, -math.log(nopt))
                oos = v["orig_of_slot"] or list(range(nopt))
                back = np.empty(nopt)
                for s in range(nopt):
                    back[oos[s]] = vec[s]
                mats.append(back)
            out["full_cover"][j] = float(cov)
            out["lp_per_view"][j] = mats
            for k in range(1, kmax + 1):
                kk = min(k, len(mats))
                mean = np.mean(np.stack(mats[:kk]), axis=0)
                idx = int(np.argmax(mean))
                out["ok_lp"][k][j] = judge_multi_choice(
                    it["choices"], it["answer"], chr(ord("A") + idx) + ".")
                votes = [int(np.argmax(m)) for m in mats[:kk]]
                cnt = np.bincount(votes, minlength=nopt).astype(float)
                cnt = cnt + 1e-6 * mean            # ties -> higher mean logprob
                out["ok_mv"][k][j] = judge_multi_choice(
                    it["choices"], it["answer"], chr(ord("A") + int(np.argmax(cnt))) + ".")
            out["ok_id_lp"][j] = out["ok_lp"][1][j]
            out["decided"][j] = int(np.argmax(mats[0]))
        elif fmt == "judge":
            labels = ["yes", "no"]
            out["ok_id_raw"][j] = judge_judgement(it["answer"], r0["response"])
            mats, cov = [], True
            for v in views:
                vec, full = option_logprobs(v, labels)
                cov = cov and full
                if vec is None:
                    vec = np.array([-math.log(2)] * 2)
                mats.append(vec)
            out["full_cover"][j] = float(cov)
            out["lp_per_view"][j] = mats
            for k in range(1, kmax + 1):
                kk = min(k, len(mats))
                mean = np.mean(np.stack(mats[:kk]), axis=0)
                out["ok_lp"][k][j] = judge_judgement(it["answer"], labels[int(np.argmax(mean))] + ".")
                votes = [int(np.argmax(m)) for m in mats[:kk]]
                cnt = np.bincount(votes, minlength=2).astype(float) + 1e-6 * mean
                out["ok_mv"][k][j] = judge_judgement(it["answer"], labels[int(np.argmax(cnt))] + ".")
            out["ok_id_lp"][j] = out["ok_lp"][1][j]
        else:                                        # SLAKE_closed: free single word / phrase
            out["ok_id_raw"][j] = judge_close_end_vqa(str(it["answer"]), r0["response"])
            strs = [norm_free(v["response"]) for v in views]
            raws = [v["response"] for v in views]
            clp = [float(v.get("cum_logprob") or 0.0) for v in views]
            for k in range(1, kmax + 1):
                kk = min(k, len(views))
                best, bestkey = None, None
                for s in set(strs[:kk]):
                    idx = [t for t in range(kk) if strs[t] == s]
                    key = (len(idx), np.mean([clp[t] for t in idx]))
                    if bestkey is None or key > bestkey:
                        bestkey, best = key, raws[idx[0]]
                out["ok_lp"][k][j] = judge_close_end_vqa(str(it["answer"]), best)
                out["ok_mv"][k][j] = out["ok_lp"][k][j]
            out["ok_id_lp"][j] = out["ok_lp"][1][j]
            out["full_cover"][j] = 1.0
    return out


# ===============================================================================================
# statistics
# ===============================================================================================
def boot_delta(a, b, nboot=M.NBOOT, seed=M.SEED_BOOT):
    """Paired item-level bootstrap of mean(a)-mean(b) on one cell."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(nboot, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def boot_macro(meth, base, order, nboot=M.NBOOT, seed=M.SEED_BOOT):
    """Paired item-level bootstrap of the macro (equal weight per cell) difference."""
    rng = np.random.default_rng(seed)
    dm = np.zeros(nboot)
    for c in order:
        a = np.asarray(meth[c], float); b = np.asarray(base[c], float)
        n = len(a)
        idx = rng.integers(0, n, size=(nboot, n))
        dm += (a[idx].mean(1) - b[idx].mean(1)) / len(order)
    point = float(np.mean([np.mean(meth[c]) - np.mean(base[c]) for c in order]))
    lo, hi = np.percentile(dm, [2.5, 97.5])
    return point, float(lo), float(hi)


def crossfit_gate(margin, ok1, okK, K=5, seed=0):
    """Spend K>1 only where the 32B's own margin is below a threshold fitted on the OTHER folds.
    Returns (per-item ok, per-item spent-K-flag).  Every item is answered; the gate chooses how
    many forward passes to spend, never whether to answer (CRITICAL RULE 6)."""
    n = len(ok1)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, K, size=n)
    ok = np.zeros(n); spend = np.zeros(n)
    cand = np.unique(np.quantile(margin[np.isfinite(margin)], np.linspace(0, 1, 51)))
    cand = np.concatenate([[-np.inf], cand, [np.inf]])
    for f in range(K):
        te = fold == f; tr = ~te
        if tr.sum() < 10 or te.sum() < 1:
            continue
        best, btau = -1.0, -np.inf
        for t in cand:
            g = margin[tr] < t
            acc = np.where(g, okK[tr], ok1[tr]).mean()
            if acc > best:
                best, btau = acc, t
        g = margin[te] < btau
        ok[te] = np.where(g, okK[te], ok1[te]); spend[te] = g
    return ok, spend


def letter_distribution(cell, items, agg):
    """MECHANISM DIAGNOSTIC for the permutable cells: does averaging over cyclic rotations move the
    predicted-letter distribution TOWARD the gold distribution (a real position-bias correction) or
    AWAY from it, toward uniform (destruction of a learned, informative prior over the benchmark's
    own answer key)?  Also splits accuracy by whether the K rotations agree."""
    if cell not in M.PERMUTABLE:
        return dict(applicable=False)
    n = len(items)
    nopt = len(items[0]["choices"])
    gold = np.zeros(nopt); pid = np.zeros(nopt); pen = np.zeros(nopt)
    stable = np.zeros(n)
    for j, it in enumerate(items):
        mats = agg["lp_per_view"][j]
        if not mats:
            continue
        gi = ord(str(it["answer"]).strip().upper()) - ord("A")
        if 0 <= gi < nopt:
            gold[gi] += 1
        v0 = int(np.argmax(mats[0])); en = int(np.argmax(np.mean(np.stack(mats), axis=0)))
        pid[v0] += 1; pen[en] += 1
        stable[j] = float(len(set(int(np.argmax(m)) for m in mats)) == 1)
    g, a, b = gold / n, pid / n, pen / n
    tv = lambda p: float(0.5 * np.abs(p - g).sum())
    ok0 = agg["ok_lp"][1]; okK = agg["ok_lp"][M.K]
    out = dict(applicable=True, n=n,
               gold_letter_rate=g.tolist(), identity_letter_rate=a.tolist(),
               ensemble_letter_rate=b.tolist(),
               tv_identity_vs_gold=tv(a), tv_ensemble_vs_gold=tv(b),
               tv_uniform_vs_gold=tv(np.full(nopt, 1.0 / nopt)),
               moves_toward_gold=bool(tv(b) < tv(a)),
               order_stable_rate=float(stable.mean()))
    for lab, mask in [("all_rotations_agree", stable > 0), ("rotations_disagree", stable == 0)]:
        if mask.sum():
            out[lab] = dict(n=int(mask.sum()), identity=float(ok0[mask].mean()),
                            ensemble=float(okK[mask].mean()),
                            delta=float(okK[mask].mean() - ok0[mask].mean()))
    return out


# ===============================================================================================
# N4 -- luck-floor control (permutation-label shuffle)
# ===============================================================================================
def n4_luck_floor(cell, items, agg, nluck=M.NLUCK, seed=M.SEED_LUCK):
    """Recombine the K arms after RANDOMLY re-assigning which un-permutation map is applied.
    Applies to the two cells that have a permutation map (PMC_VQA, MedXpertQA-MM); for the other
    three there is no map, so the discriminating control is the equal-K temperature arm instead."""
    if cell not in M.PERMUTABLE:
        return dict(applicable=False,
                    reason="no permutation map on this cell (no explicit option list); the "
                           "discriminating control here is the equal-K temperature arm")
    rng = np.random.default_rng(seed)
    real = agg["ok_lp"][M.K]
    n = len(items)
    accs = np.zeros(nluck)
    per_item_lp = agg["lp_per_view"]
    for b in range(nluck):
        ok = np.zeros(n)
        for j, it in enumerate(items):
            mats = per_item_lp[j]
            if not mats:
                ok[j] = agg["ok_id_raw"][j]
                continue
            nopt = len(mats[0])
            # SHUFFLE the un-permutation: re-rotate each view's already-un-permuted vector by a
            # random extra shift, i.e. apply the WRONG map, then average as usual.
            sh = rng.integers(0, nopt, size=len(mats))
            stack = np.stack([np.roll(m, int(s)) for m, s in zip(mats, sh)])
            idx = int(np.argmax(stack.mean(0)))
            ok[j] = judge_multi_choice(it["choices"], it["answer"], chr(ord("A") + idx) + ".")
        accs[b] = ok.mean()
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return dict(applicable=True, real_acc=float(real.mean()), null_mean=float(accs.mean()),
                null_lo=float(lo), null_hi=float(hi), nluck=nluck,
                delta_vs_null=float(real.mean() - accs.mean()),
                rejected=bool(real.mean() > hi))


# ===============================================================================================
# main
# ===============================================================================================
def run():
    items_all = M.build_items()
    sub = set(M.pmc_subsample_ids())
    z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"), allow_pickle=True)

    res = dict(title="ATTACK 2 (MCQ-TTA) -- results", date=M.DATE,
               preregistration=os.path.basename(M.PREREG),
               bar=dict(always_32b_direct_macro=0.6567, macro_delta_needed=M.BAR_MACRO_DELTA,
                        summed_mcq_gain_needed=M.BAR_SUM_MCQ))
    cells_done, per_cell, vec_tta, vec_base, vec_id = [], {}, {}, {}, {}
    n3 = {}

    for cell in M.CELLS:
        items = items_all[cell]
        if cell == "PMC_VQA":
            items = [r for r in items if r["i"] in sub]
        by = load_rows(cell, "A")
        if not by:
            continue
        have = [r for r in items if r["i"] in by and 0 in by[r["i"]]]
        if len(have) < len(items):
            print(f"[{cell}] INCOMPLETE: {len(have)}/{len(items)} items have the identity view; "
                  f"restricting the cell to the completed items and SAYING SO")
        items = have
        if not items:
            continue
        agg = aggregate_cell(cell, items, by)
        cells_done.append(cell)

        # ---- N3: the identity view must reproduce the deployed baseline on the same ids -------
        pos = {r["i"]: k for k, r in enumerate(items)}
        base = np.array([z[f"{cell}|always_32b_direct"][r["i"]] for r in items], float)
        idraw = agg["ok_id_raw"]
        n3[cell] = dict(n=len(items),
                        identity_rerun_acc=float(idraw.mean()),
                        stored_baseline_acc_same_ids=float(base.mean()),
                        stored_baseline_acc_full_cell=float(z[f"{cell}|always_32b_direct"].mean()),
                        published_full_cell=STORED_ACC[cell],
                        abs_dev_vs_same_ids=float(abs(idraw.mean() - base.mean())),
                        per_item_agreement=float((idraw == base).mean()),
                        passed=bool(abs(idraw.mean() - base.mean()) <= N3_TOL),
                        mean_gen_toks=float(agg["gen_toks"].mean()),
                        full_option_coverage_rate=float(agg["full_cover"].mean()),
                        mean_views=float(agg["nviews"].mean()))

        # ---- arms -----------------------------------------------------------------------------
        margin = np.array([M_ for M_ in _stored_margin(cell, items)], float)
        alwaysK = agg["ok_lp"][M.K]
        gated_seeds = []
        for s in range(10):
            g, spend = crossfit_gate(margin, idraw, alwaysK, seed=s)
            gated_seeds.append((g.mean(), spend.mean(), g))
        gm = float(np.mean([a for a, _, _ in gated_seeds]))
        gsd = float(np.std([a for a, _, _ in gated_seeds]))
        gspend = float(np.mean([b for _, b, _ in gated_seeds]))
        gated_ok = np.mean(np.stack([g for _, _, g in gated_seeds]), axis=0)  # seed-averaged (diag)
        gated_ok_seed0 = gated_seeds[0][2]

        d_alwaysK = boot_delta(alwaysK, base)
        d_gated = boot_delta(gated_ok_seed0, base)
        per_cell[cell] = dict(
            n=len(items), n_full_cell=int(len(z[f"{cell}|always_32b_direct"])),
            baseline_same_ids=float(base.mean()),
            identity_rerun=float(idraw.mean()),
            identity_logprob_argmax=float(agg["ok_id_lp"].mean()),
            K_curve_logprob={k: float(agg["ok_lp"][k].mean()) for k in range(1, M.K + 1)},
            K_curve_majority={k: float(agg["ok_mv"][k].mean()) for k in range(1, M.K + 1)},
            alwaysK_delta_vs_baseline=dict(delta=d_alwaysK[0], lo=d_alwaysK[1], hi=d_alwaysK[2],
                                           sig=bool(d_alwaysK[1] > 0 or d_alwaysK[2] < 0),
                                           label="DIAGNOSTIC: eval-visible upper bound"),
            gated_delta_vs_baseline=dict(delta=d_gated[0], lo=d_gated[1], hi=d_gated[2],
                                         sig=bool(d_gated[1] > 0 or d_gated[2] < 0),
                                         label="DEPLOYABLE: 5-fold cross-fit gate on the 32B's own margin"),
            gated_seed_mean=gm, gated_seed_sd=gsd, gated_spend_rate=gspend,
            n4=n4_luck_floor(cell, items, agg),
            letter_distribution=letter_distribution(cell, items, agg))
        vec_tta[cell] = alwaysK
        vec_base[cell] = base
        vec_id[cell] = idraw

    res["N3_identity_control"] = n3
    res["per_cell"] = per_cell
    res["cells_completed"] = cells_done

    # ---- macro integration ---------------------------------------------------------------------
    # A cell with no TTA data falls back to the deployed baseline vector, which is exactly what the
    # method does there (on 4 of 5 MCQ cells the deployed method IS always-32B-direct).  So a
    # partial run still yields a VALID macro for "TTA on the completed cells, baseline elsewhere";
    # `cells_completed` says which is which and no cell is ever silently dropped.
    if cells_done:
        openc = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
        meth, bas = {}, {}
        for c in M.CELLS:
            if c in cells_done:
                meth[c] = vec_tta[c]; bas[c] = vec_base[c]
            else:
                v = np.asarray(z[f"{c}|always_32b_direct"], float)
                meth[c] = v; bas[c] = v
        for c in openc:
            meth[c] = np.asarray(z[f"{c}|method_accuracy_max_veto"], float)
            bas[c] = np.asarray(z[f"{c}|always_32b_direct"], float)
        p, lo, hi = boot_macro(meth, bas, M.MACRO8)
        res["macro"] = dict(
            weighting="equal weight per reporting cell (8 cells, 1/8 each), Variant B",
            cells_with_tta=cells_done,
            cells_falling_back_to_baseline=[c for c in M.CELLS if c not in cells_done],
            note="PMC_VQA is restricted to the pre-registered n=6000 subsample on BOTH sides "
                 "(fully paired). The 3 open cells carry the deployed accuracy-max arm from "
                 "_selector_rerun_parts/vec_disjoint.npz unchanged; this attack does not touch them.",
            baseline_macro_same_pool=float(np.mean([np.mean(bas[c]) for c in M.MACRO8])),
            method_macro=float(np.mean([np.mean(meth[c]) for c in M.MACRO8])),
            delta=p, lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0), nboot=M.NBOOT, seed=M.SEED_BOOT,
            summed_mcq_gain=float(sum(np.mean(vec_tta[c]) - np.mean(vec_base[c]) for c in cells_done)),
            arm="TTA always-K (DIAGNOSTIC, eval-visible upper bound)")
    # ---- verdict against EVERY pre-registered criterion --------------------------------------
    v = dict()
    v["N1"] = "PASS (max abs deviation 0.0) -- see artifacts/mcq_tta_nulltests_2026-08-10.json"
    v["N2"] = "PASS (100% byte-equality on all 5 cells) -- same file"
    v["N3"] = {c: dict(abs_dev=n3[c]["abs_dev_vs_same_ids"], passed=n3[c]["passed"],
                       mean_gen_toks=n3[c]["mean_gen_toks"]) for c in n3}
    v["N3_overall"] = ("PASS" if all(n3[c]["passed"] for c in n3) else "FAIL") if n3 else "NOT RUN"
    v["N4"] = {c: per_cell[c]["n4"] for c in per_cell}
    if per_cell:
        sK = sum(per_cell[c]["alwaysK_delta_vs_baseline"]["delta"] for c in per_cell)
        sG = sum(per_cell[c]["gated_delta_vs_baseline"]["delta"] for c in per_cell)
        v["summed_mcq_gain_alwaysK_DIAGNOSTIC"] = sK
        v["summed_mcq_gain_gated_DEPLOYABLE"] = sG
        v["KILL_ii_alwaysK_below_bar"] = dict(
            threshold=M.BAR_SUM_MCQ, value=sK, fires=bool(sK < M.BAR_SUM_MCQ),
            meaning="the always-K policy is the EVAL-VISIBLE UPPER BOUND; if it is below the bar "
                    "the gated cross-fit policy cannot reach it either, so the attack is dead and "
                    "no Stage B temperature control is run (pre-registered trigger).")
        v["stage_B_triggered"] = bool(sK >= M.BAR_SUM_MCQ)
        v["pmc_extension_triggered"] = bool(
            per_cell.get("PMC_VQA", {}).get("alwaysK_delta_vs_baseline", {}).get("delta", -1) >= 0.01)
        v["guardrail_per_cell"] = {c: dict(
            delta=per_cell[c]["gated_delta_vs_baseline"]["delta"],
            worse=bool(per_cell[c]["gated_delta_vs_baseline"]["hi"] < 0)) for c in per_cell}
    # ---- macro-weighted cost of the gated policy (labelled; never paired with sample-weighted acc)
    costf = os.path.join(ART, "mcq_tta_cost_2026-08-10.json")
    if per_cell and os.path.exists(costf):
        cj = json.load(open(costf))
        r_meas = float(cj["ratios_measured"]["energy"])
        R32 = 4.57
        base_cell = {c: R32 for c in M.CELLS}
        opencost = {"SLAKE_open": 13.97, "VQA_RAD_open": 17.30, "PATH_VQA_open": 10.31}
        gc = {c: R32 * ((1 - per_cell[c]["gated_spend_rate"])
                        + per_cell[c]["gated_spend_rate"] * 4.0) for c in per_cell}
        gm = {c: R32 * ((1 - per_cell[c]["gated_spend_rate"])
                        + per_cell[c]["gated_spend_rate"] * r_meas) for c in per_cell}
        for c in M.CELLS:
            gc.setdefault(c, R32); gm.setdefault(c, R32)
        allc = dict(gc); allc.update(opencost)
        allm = dict(gm); allm.update(opencost)
        res["cost"] = dict(
            weighting="MACRO (equal weight per cell, 8 cells) -- this MUST NOT be paired with a "
                      "sample-weighted accuracy, and the accuracy reported above is macro.",
            baseline_always_32b_direct_macro_flopeq=float(np.mean([R32] * 5 + [R32] * 3)),
            method_macro_flopeq_as_charged=float(np.mean([allc[c] for c in M.MACRO8])),
            method_macro_flopeq_measured_ratio=float(np.mean([allm[c] for c in M.MACRO8])),
            measured_K4_over_K1_energy_ratio=r_meas,
            per_cell_gated_spend_rate={c: per_cell[c]["gated_spend_rate"] for c in per_cell},
            open_cell_costs_note="the 3 open cells carry the deployed accuracy-max arm's costs "
                                 "(13.97 / 17.30 / 10.31 FLOP-eq) unchanged; this attack does not "
                                 "touch them.",
            provenance="as-charged = model (K x R32, no sharing); measured-ratio = the NVML/wall-clock "
                       "ratio from artifacts/mcq_tta_cost_2026-08-10.json applied to the K>1 fraction.")
    res["verdict"] = v
    res["priors_zero_gpu"] = dict(
        file="artifacts/mcq_tta_2026-08-10_pilot.json",
        note="two independent zero-GPU priors were measured BEFORE this run: (a) a 2-view "
             "prompt-form ensemble on the 32B gains nothing by any cheap combiner (summed -0.0042 "
             "by confidence, exactly 0.0 by a cross-fit logistic) against a 2-view ORACLE of "
             "+0.0484; (b) the cyclic-permutation MAJORITY ensemble on Lingshu-32B on MMMU-Medical "
             "(n=145, the EXCLUDED cell) is -0.0069 [-0.0483,+0.0345] against an oracle of +0.1448. "
             "Both were negative-leaning; the pre-registered run was executed anyway because its "
             "primary aggregation averages the per-option POSTERIOR rather than voting on argmaxes.")
    json.dump(res, open(M.OUT, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "per_cell"}, indent=1, default=float)[:4000])
    print("wrote", M.OUT)
    return res


_MARGIN_CACHE = {}


def _stored_margin(cell, items):
    """The deployed 32B's own first-token margin, from the read-only baseline dump."""
    if cell not in _MARGIN_CACHE:
        tag = {"PMC_VQA": "PMC_VQA", "SLAKE_closed": "SLAKE", "VQA_RAD_closed": "VQA_RAD",
               "PATH_VQA_closed": "PATH_VQA", "MedXpertQA-MM": "MedXpertQA-MM"}[cell]
        raw = M.load_stored(tag)
        if cell == "SLAKE_closed":
            raw = [r for r in raw if r["answer_type"] == "CLOSED"]
        elif cell in ("VQA_RAD_closed", "PATH_VQA_closed"):
            raw = [r for r in raw if str(r["answer"]).lower() in ("yes", "no")]
        _MARGIN_CACHE[cell] = [float(r.get("margin") or 0.0) for r in raw]
    m = _MARGIN_CACHE[cell]
    return [m[r["i"]] for r in items]


if __name__ == "__main__":
    run()
