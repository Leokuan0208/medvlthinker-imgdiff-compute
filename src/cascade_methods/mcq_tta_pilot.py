#!/usr/bin/env python3
"""
mcq_tta_pilot.py -- ATTACK 2, ZERO-GPU PILOT: a 2-view PROMPT-FORM test-time ensemble on
Lingshu-32B, measurable entirely from dumps that already exist on disk.

WHAT THE TWO VIEWS ARE.  MedEvalKit/eval_results_lingshu32b_full   (`Answer with the option's
letter from the given choices directly.`)  and  .../eval_results_lingshu32b_think  (`Answer with
the option's letter ... and put the letter in one "\\boxed{}".`).  Both are DIRECT arms: the token
audit below reports 3.0-3.3 mean generated tokens in every cell of both arms, exactly as CLAUDE.md
records ("MedEvalKit's --reasoning True flag only appends 'put the letter in \\boxed{}', which is
not a reasoning prompt").  So this is a genuine prompt-FORM perturbation of the same greedy decode,
not a think-vs-direct contrast, and it must never be cited as one.

WHY IT IS RUN.  It is the cheapest possible probe of Attack 2's mechanism and it costs no GPU.  It
is NOT the pre-registered experiment: it has K=2 (not 4), the perturbation is an instruction-format
change (not a content-preserving cyclic option permutation), and the dumps store only the argmax
letter plus the first token's top-1 confidence and margin -- NOT the per-option posterior -- so the
pre-registered primary aggregation (mean of un-permuted per-option log-probs) is NOT computable
here.  Everything measurable here is a SELECTION rule between two views.

    python3 src/cascade_methods/mcq_tta_pilot.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mcq_tta as M  # noqa: E402

OUT = os.path.join(M.ART, "mcq_tta_2026-08-10_pilot.json")
CELLS = [("PMC_VQA", "PMC_VQA", None), ("SLAKE_closed", "SLAKE", "SLAKE"),
         ("VQA_RAD_closed", "VQA_RAD", "YESNO"), ("MedXpertQA-MM", "MedXpertQA-MM", None)]


def load(tag, ds):
    p = f"{M.MEK}/eval_results_lingshu32b_{tag}/{{}}/{ds}/results.json"
    return json.load(open(p))


def boot(a, b, nboot=M.NBOOT, seed=M.SEED_BOOT):
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(nboot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(a.mean() - b.mean()), lo=float(lo), hi=float(hi),
                sig=bool(lo > 0 or hi < 0))


def crossfit_logistic(X, y, ok_a, ok_b, K=5, seeds=10):
    """Best DEPLOYABLE 2-view combiner: pick view A or B from cheap signals only, fitted on the
    other folds.  Reported as a seed-averaged number with sd, because a single fold partition is
    not a result in this project (seed spread ~0.021 exceeds most architectural effects)."""
    from sklearn.linear_model import LogisticRegression
    accs = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        fold = rng.integers(0, K, size=len(y))
        ok = np.zeros(len(y))
        for f in range(K):
            te = fold == f; tr = ~te
            if tr.sum() < 20 or te.sum() < 1:
                continue
            if len(np.unique(y[tr])) < 2:
                ok[te] = ok_a[te]; continue
            m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            pick_b = m.predict(X[te]) == 1
            ok[te] = np.where(pick_b, ok_b[te], ok_a[te])
        accs.append(ok.mean())
    return float(np.mean(accs)), float(np.std(accs)), float(np.min(accs)), float(np.max(accs))


def position_bias():
    """ZERO-GPU diagnostic: how large is the answer-position error mode the permutation arm targets?
    Read from the deployed baseline dump only (eval_results_lingshu32b_full), no new numbers."""
    import re
    from collections import defaultdict
    out = {}
    for cell, tag in [("PMC_VQA", "PMC_VQA"), ("MedXpertQA-MM", "MedXpertQA-MM")]:
        d = load("full", tag)
        n = len(d)
        acc = defaultdict(list); gold = defaultdict(int)
        for r in d:
            p = re.sub(r"[^a-z0-9]", "", str(r.get("response", "")).strip().lower()).upper()
            acc[p].append(1.0 if r["correct"] else 0.0)
            gold[str(r["answer"]).upper()] += 1
        nerr = sum(1 for r in d if not r["correct"])
        letters = sorted(gold)
        out[cell] = dict(
            n=n, overall_acc=float(np.mean([1.0 if r["correct"] else 0.0 for r in d])),
            per_predicted_letter={L: dict(pred_rate=len(acc.get(L, [])) / n,
                                          gold_rate=gold[L] / n,
                                          acc_when_predicted=float(np.mean(acc[L])) if acc.get(L) else None)
                                  for L in letters},
            total_variation_pred_vs_gold=float(0.5 * sum(
                abs(len(acc.get(L, [])) / n - gold[L] / n) for L in letters)),
            share_of_errors_made_while_predicting_A=float(
                sum(1 for r in d if re.sub(r"[^a-z0-9]", "", str(r["response"]).strip().lower()).upper() == "A"
                    and not r["correct"]) / max(1, nerr)))
    out["reading"] = (
        "On PMC_VQA the 32B answers 'A' on 23.3% of items where 'A' is gold 13.2% of the time, and "
        "it is right only 27.6% of the time when it says 'A' against 64.3% / 68.8% when it says "
        "B / C. 37.5% of ALL its errors are made while predicting A. That is the structured, "
        "answer-position-shaped error mode a cyclic-permutation ensemble is supposed to cancel, and "
        "it is why Attack 2 was worth running. It is a DIAGNOSTIC of headroom, NOT a result: an "
        "A-preference could equally be a fallback-under-uncertainty habit, in which case averaging "
        "over rotations moves the error rather than removing it.")
    return out


def mmmu_permutation_prior():
    """ZERO-GPU: the ONLY existing in-repo measurement of Attack 2's exact mechanism on
    Lingshu-32B.  results/cascade_methods/artifacts/mmmu_perm_{32b,7b}.json (produced by
    MedEvalKit/mmmu_perm_eval.py, an n=150 CONTAMINATION AUDIT on MMMU-Medical, the cell Variant B
    EXCLUDES) evaluated every cyclic option shift per question and stored the per-shift prediction.
    Nobody ever combined those shifts into an ensemble.  We do that here.

    CAVEATS, all load-bearing: n=145 multiple-choice questions; MMMU-Medical is the excluded cell
    and Lingshu-7B's 0.84 there is the known contamination anomaly; the prompt path is MMMU's own
    construct_prompt, not get_multiple_choice_prompt; and the dumps store only the parsed letter,
    so this is MAJORITY VOTE -- the pre-registered PRIMARY aggregation (mean of un-permuted
    per-option log-probs) is strictly more informative and is NOT computable here.  A prior, not a
    result on a reporting cell."""
    from collections import Counter
    out = {}
    for tag, f in [("Lingshu-32B", "mmmu_perm_32b.json"), ("Lingshu-7B", "mmmu_perm_7b.json")]:
        d = json.load(open(os.path.join(M.ART, f)))
        byq = {}
        for r in d["per_eval"]:
            if r["qtype"] != "multiple-choice":
                continue
            byq.setdefault(r["id"], {})[r["shift"]] = r
        ok0, oke, oko, stable, pred_a = [], [], [], [], []
        for q in sorted(byq):
            sh = byq[q]; k = len(sh)
            if 0 not in sh:
                continue
            gi = ord(str(sh[0]["gold"]).upper()) - ord("A")
            votes = []
            for s, r in sh.items():
                p = str(r["pred"]).strip().upper()
                votes.append((ord(p) - ord("A") + s) % k
                             if (len(p) == 1 and "A" <= p <= chr(ord("A") + k - 1)) else None)
            v0 = votes[0]
            cnt = Counter([v for v in votes if v is not None])
            if cnt:
                top = max(cnt.values()); cands = [c for c, n in cnt.items() if n == top]
                pick = v0 if v0 in cands else sorted(cands)[0]
            else:
                pick = None
            ok0.append(float(v0 == gi)); oke.append(float(pick == gi))
            oko.append(float(any(v == gi for v in votes)))
            stable.append(float(len(cnt) == 1))
            pred_a.append(float(str(sh[0]["pred"]).strip().upper() == "A"))
        ok0 = np.array(ok0); oke = np.array(oke); oko = np.array(oko)
        stab = np.array(stable, float); pa = np.array(pred_a, float)
        def sub(mask):
            if mask.sum() == 0:
                return None
            return dict(n=int(mask.sum()), identity=float(ok0[mask].mean()),
                        ensemble=float(oke[mask].mean()),
                        delta=float(oke[mask].mean() - ok0[mask].mean()))
        out[tag] = dict(n_mc_questions=len(ok0),
                        identity_shift0=float(ok0.mean()),
                        permutation_majority_ensemble=float(oke.mean()),
                        oracle_over_shifts=float(oko.mean()),
                        delta_ensemble_vs_identity=boot(oke, ok0),
                        delta_oracle_vs_identity=boot(oko, ok0),
                        order_instability_rate=float(1.0 - stab.mean()),
                        breakdown=dict(
                            all_shifts_agree=sub(stab > 0),
                            shifts_disagree=sub(stab == 0),
                            identity_answered_A=sub(pa > 0),
                            identity_did_not_answer_A=sub(pa == 0)))
    out["reading"] = (
        "On Lingshu-32B the cyclic-permutation majority ensemble does NOT beat the identity view "
        "(-0.0069, CI spans zero) even though the ORACLE over shifts is +0.1448 -- the same shape "
        "as the 2-view prompt-form pilot above and as this project's recurring luck floor: a large "
        "combinable gap that no frozen-model combiner harvests. It is a PRIOR for the pre-registered "
        "run and NOT a substitute: n=145 on the EXCLUDED cell, and majority vote discards the "
        "per-option posterior that the pre-registered primary aggregation averages. "
        "MECHANISM, and it is the informative part: 35.9% of Lingshu-32B's MMMU questions are "
        "ORDER-UNSTABLE (the K rotations do not all give the same un-permuted answer), but on that "
        "unstable subset the ensemble is 0.365 vs the identity view's 0.385, and on the subset where "
        "the identity view answered 'A' -- the bias direction the attack targets -- the ensemble is "
        "0.596 vs 0.638, i.e. it HURTS exactly where it was supposed to help. That is evidence for "
        "the alternative hypothesis named in the position-bias diagnostic: order instability here is "
        "a symptom of the model not knowing the answer, not a correctable positional prior, so "
        "averaging over rotations MOVES the error rather than removing it.")
    return out


def run():
    res = dict(
        title="ATTACK 2 zero-GPU pilot -- 2-view PROMPT-FORM ensemble on Lingshu-32B",
        date=M.DATE,
        views=dict(A="eval_results_lingshu32b_full  (bare-letter instruction)",
                   B="eval_results_lingshu32b_think (\\boxed{} instruction)"),
        caveat="NOT the pre-registered experiment: K=2 not 4; the perturbation is an instruction "
               "FORMAT change, not a content-preserving cyclic option permutation; and the dumps "
               "store only the argmax letter + top-1 conf/margin, so the pre-registered primary "
               "aggregation (mean of un-permuted per-option log-probs) is NOT computable here. "
               "Every combiner below is a SELECTION rule between two views.",
        token_audit_note="both arms must be DIRECT (3-4 generated tokens); reported per cell.",
        position_bias_diagnostic=position_bias(),
        mmmu_permutation_prior=mmmu_permutation_prior(),
        cells={})
    summed = {}
    for cell, ds, filt in CELLS:
        a, b = load("full", ds), load("think", ds)
        if filt == "SLAKE":
            keep = [i for i, r in enumerate(a) if r["answer_type"] == "CLOSED"]
        elif filt == "YESNO":
            keep = [i for i, r in enumerate(a) if str(r["answer"]).lower() in ("yes", "no")]
        else:
            keep = list(range(min(len(a), len(b))))
        A = [a[i] for i in keep]; B = [b[i] for i in keep]
        okA = np.array([1.0 if r["correct"] else 0.0 for r in A])
        okB = np.array([1.0 if r["correct"] else 0.0 for r in B])
        cA = np.array([float(r.get("conf") or 0) for r in A])
        cB = np.array([float(r.get("conf") or 0) for r in B])
        mA = np.array([float(r.get("margin") or 0) for r in A])
        mB = np.array([float(r.get("margin") or 0) for r in B])
        agree = np.array([str(r1["response"]).strip().lower()[:8] == str(r2["response"]).strip().lower()[:8]
                          for r1, r2 in zip(A, B)], float)
        by_conf = np.where(agree > 0, okA, np.where(cA >= cB, okA, okB))
        by_marg = np.where(agree > 0, okA, np.where(mA >= mB, okA, okB))
        oracle2 = np.maximum(okA, okB)
        y = (okB > okA).astype(int)                    # 1 => view B is the one to take
        X = np.column_stack([cA, cB, mA, mB, agree, cA - cB, mA - mB])
        lm, lsd, lmin, lmax = crossfit_logistic(X, y, okA, okB)
        res["cells"][cell] = dict(
            n=len(A),
            mean_gen_toks=dict(viewA=float(np.mean([r["gen_toks"] for r in A])),
                               viewB=float(np.mean([r["gen_toks"] for r in B]))),
            acc=dict(viewA=float(okA.mean()), viewB=float(okB.mean()),
                     agree_rate=float(agree.mean()),
                     pick_by_conf=float(by_conf.mean()), pick_by_margin=float(by_marg.mean()),
                     crossfit_logistic_seedmean=lm, crossfit_logistic_seedsd=lsd,
                     crossfit_logistic_seedrange=[lmin, lmax],
                     oracle_of_2=float(oracle2.mean())),
            delta_vs_viewA=dict(
                viewB=boot(okB, okA), pick_by_conf=boot(by_conf, okA),
                pick_by_margin=boot(by_marg, okA), oracle_of_2=boot(oracle2, okA)),
            selection_efficiency=float((by_conf.mean() - okA.mean()) /
                                       max(1e-9, oracle2.mean() - okA.mean())))
        for k, v in [("viewB", okB), ("pick_by_conf", by_conf), ("pick_by_margin", by_marg),
                     ("crossfit_logistic", np.array([lm])), ("oracle_of_2", oracle2)]:
            d = (lm - okA.mean()) if k == "crossfit_logistic" else float(v.mean() - okA.mean())
            summed[k] = summed.get(k, 0.0) + d
    res["summed_gain_over_4_cells"] = summed
    res["bar"] = dict(summed_mcq_gain_needed_over_5_cells=M.BAR_SUM_MCQ,
                      note="the pilot covers 4 of the 5 MCQ cells (no PATH_VQA think dump exists), "
                           "so its summed gain is not directly comparable to the 5-cell bar; it is "
                           "reported to show the SIGN and the ORDER OF MAGNITUDE.")
    res["verdict"] = (
        "A 2-view prompt-form ensemble on the 32B, combined by any cheap frozen signal, does NOT "
        "gain: the confidence rule and the cross-fit logistic combiner both sit at or below the "
        "single-view baseline, while the 2-view ORACLE is large. The MCQ selection wall reappears "
        "in a new place. This is a PRIOR for the pre-registered permutation arm, not a substitute "
        "for it: the permutation arm AVERAGES a bias-cancelling family rather than SELECTING "
        "between views, which is exactly the distinction that has to be settled empirically.")
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    run()
