#!/usr/bin/env python3
"""coverage_diagnosis2.py -- SCOUT B part 2, 2026-08-10.

Part 1 (coverage_diagnosis.py) established the shape. Part 2 nails down the decisive
numbers with CIs and mines every remaining free dump for a coverage lever:

  A. THE MULTIPLIER, measured: when coverage is ADDED, how often does the selector
     convert it?  (items recoverable at 8 but NOT at 4, incumbent + deployed selector)
  B. oracle@N out to N=32 on vqa_rad_open (judge-labelled sc32 dump) -- saturation.
  C. TEXT-side portfolio complementarity, JUDGE-labelled: temp-1.0 and think-mode
     8-sample pools (ckpts/openvqa/cheap_lingshu7b_scale) vs the endpoint pool.
  D. IMAGE-side complementarity: cap80 / cap160 greedy answers
     (ckpts/openvqa/cheap_lingshu7b_perturb) vs the cap320 pool. EXACT-MATCH labels
     only -> a strict LOWER BOUND on rescue.
  E. laterality deep-dive on the no-coverage subset.
  F. capture-recapture ceiling on "questions reachable by iid sampling at all".

No GPU. nboot=10000, seed 20260810.
Out: results/cascade_methods/artifacts/coverage_diagnosis2_2026-08-10.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))
from training_methods import genframe_data as G  # noqa: E402

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/coverage_diagnosis2_2026-08-10.json")
SCDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
SCALEDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b_scale")
PERTDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b_perturb")
DEPLOYED = os.path.join(ROOT, "ckpts/train/selector_ens8_scaled")
EVAL_DS = G.EVAL_DS
NBOOT = 10000
SEED = 20260810
LAT = {"right", "left", "both", "bilateral"}


def norm(s):
    return str(s).strip().lower()


def load_judged(path_base):
    """{idx -> {preds, y(judge), gold, question, exact}} for any *_scexploded.judge.jsonl pair."""
    jd = {json.loads(l)["idx"]: int(json.loads(l)["judge_ok"])
          for l in open(path_base + "_scexploded.judge.jsonl")}
    out = {}
    for l in open(path_base + ".jsonl"):
        r = json.loads(l)
        lab, first = {}, {}
        for s, a in enumerate(r["preds"]):
            na = norm(a)
            if na not in lab:
                lab[na] = jd[f"{r['idx']}#{s}"]
        out[str(r["idx"])] = {"preds": list(r["preds"]),
                              "y": [lab[norm(a)] for a in r["preds"]],
                              "exact": [int(v) for v in r["oks"]],
                              "gold": r.get("gold"), "question": r.get("question")}
    return out


def boot_mean_diff(a, b, nboot=NBOOT, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, n, n)
        d[k] = a[s].mean() - b[s].mean()
    return {"delta": float(a.mean() - b.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def boot_rate(x, nboot=NBOOT, seed=SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    d = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    return {"rate": float(x.mean()), "n": int(n),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


# ======================================================================================
# A. THE MULTIPLIER
# ======================================================================================
def multiplier(items):
    SL = np.array([[int(v) for v in it["sl"]] for it in items], int)
    ds = np.array([EVAL_DS.index(it["ds"]) for it in items])
    selectors = {"incumbent_clean_disjoint": G.incumbent_scores()}
    dep = {}
    for d in ["slake", "vqa_rad", "pathvqa"]:
        for it in json.load(open(os.path.join(DEPLOYED, f"transfer_dump_{d}_open_lingshu7b.json"))):
            dep[(it["ds"], it["idx"])] = list(it["scores"])
    selectors["deployed_ens8_scaled"] = dep

    out = {}
    for name, sc in selectors.items():
        r = G.sel_eff(sc, items)
        got = r["got"]
        res = {"sel_eff_all": r["sel_eff"], "selected": r["acc"], "oracle": r["oracle"],
               "per_ds_sel_eff": {d: r["per_ds"][d]["sel_eff"] for d in EVAL_DS}}
        # strata of ADDED coverage
        strata = {}
        for k0, k1 in [(1, 2), (2, 4), (4, 8), (1, 8)]:
            r0 = (SL[:, :k0].max(1) == 1)
            r1 = (SL[:, :k1].max(1) == 1)
            base = r0
            added = r1 & ~r0
            strata[f"recoverable_at_{k0}"] = {
                "n": int(base.sum()),
                "conversion_rate": float(got[base].mean()) if base.sum() else None}
            strata[f"ADDED_by_samples_{k0+1}..{k1}"] = dict(
                boot_rate(got[added]),
                share_of_all_items=float(added.mean()),
                interpretation="fraction of NEWLY-covered questions the selector actually converts")
        res["conversion_of_added_coverage"] = strata
        # marginal ratio dSelected/dOracle over the prefix curve
        S = np.array([list(v) for v in
                      (sc[(it["ds"], it["idx"])] for it in items)], float)
        curve = []
        for k in range(1, 9):
            pk = np.argmax(S[:, :k], 1)
            curve.append({"N": k,
                          "oracle": float((SL[:, :k].max(1) == 1).mean()),
                          "selected": float(SL[np.arange(len(items)), pk].mean())})
        res["prefix_curve"] = curve
        o1, o8 = curve[0]["oracle"], curve[7]["oracle"]
        s1, s8 = curve[0]["selected"], curve[7]["selected"]
        o4, s4 = curve[3]["oracle"], curve[3]["selected"]
        o7, s7 = curve[6]["oracle"], curve[6]["selected"]
        res["realized_multiplier"] = {
            "N1->N8": {"d_oracle": o8 - o1, "d_selected": s8 - s1, "ratio": (s8 - s1) / (o8 - o1)},
            "N4->N8": {"d_oracle": o8 - o4, "d_selected": s8 - s4, "ratio": (s8 - s4) / (o8 - o4)},
            "N7->N8 (marginal)": {"d_oracle": o8 - o7, "d_selected": s8 - s7,
                                  "ratio": (s8 - s7) / (o8 - o7)},
            "brief_assumed_ratio": 0.810627,
            "note": "selected = oracle x sel_eff is an EXACT identity, so d(selected)/d(oracle) "
                    "= sel_eff + oracle * d(sel_eff)/d(oracle); the second term is strongly "
                    "negative because added coverage arrives in diluted pools.",
        }
        # per-cell conversion of the 5..8 increment
        pc = {}
        for j, d in enumerate(EVAL_DS):
            m = ds == j
            added = m & (SL[:, :8].max(1) == 1) & ~(SL[:, :4].max(1) == 1)
            base = m & (SL[:, :4].max(1) == 1)
            pc[d] = {"added_n": int(added.sum()),
                     "added_conversion": float(got[added].mean()) if added.sum() else None,
                     "base_n": int(base.sum()),
                     "base_conversion": float(got[base].mean()) if base.sum() else None}
        res["per_cell_added_vs_base_conversion"] = pc
        out[name] = res
    return out


# ======================================================================================
# B. oracle@N to 32 (vqa_rad_open, judge)
# ======================================================================================
def oracle32(items):
    d = load_judged(os.path.join(SCALEDIR, "ckpt_vqa_rad_open_lingshu7b_sc32"))
    ks = sorted(d)
    Y = np.array([d[k]["y"][:32] for k in ks], int)
    c = Y.sum(1)
    curve = []
    for n in range(1, 33):
        p = []
        for ci in c:
            if ci == 0:
                p.append(0.0)
            elif 32 - ci < n:
                p.append(1.0)
            else:
                v = 1.0
                for t in range(n):
                    v *= (32 - ci - t) / (32 - t)
                p.append(1.0 - v)
        curve.append(float(np.mean(p)))
    return {"cell": "vqa_rad_open", "n": len(ks), "source": "cheap_lingshu7b_scale/ckpt_vqa_rad_open_lingshu7b_sc32 (judge-labelled)",
            "oracle_at_N_1..32": [round(v, 6) for v in curve],
            "oracle@8": curve[7], "oracle@16": curve[15], "oracle@32": curve[31],
            "gain_8_to_16": curve[15] - curve[7], "gain_16_to_32": curve[31] - curve[15],
            "always_32b_direct_this_cell": 0.6000,
            "n_correct_of_32_hist": dict(sorted(Counter(c.tolist()).items())),
            "share_zero_correct_of_32": float((c == 0).mean())}


# ======================================================================================
# C. TEXT-side alternatives, judge-labelled (temp 1.0, think), vqa_rad_open
# ======================================================================================
def text_side(items):
    end = {str(it["idx"]): it for it in items if it["ds"] == "vqa_rad_open"}
    out = {"cell": "vqa_rad_open", "n": len(end),
           "note": "all three arms are 8-sample pools on the SAME 200 questions with JUDGE labels; "
                   "the endpoint arm is cap320/temp0.7 (the deployed config)."}
    arms = {}
    for tag, lbl in [("t10", "temp 1.0 (endpoint is temp 0.7)"),
                     ("think", "reasoning-mode sampling")]:
        d = load_judged(os.path.join(SCALEDIR, f"ckpt_vqa_rad_open_lingshu7b_{tag}"))
        ks = [k for k in end if k in d]
        base_rec = np.array([1 if 1 in end[k]["sl"] else 0 for k in ks])
        arm_rec = np.array([1 if 1 in d[k]["y"][:8] else 0 for k in ks])
        union = ((base_rec + arm_rec) > 0).astype(int)
        arms[tag] = {
            "label": lbl, "n": len(ks),
            "oracle@8_this_arm": float(arm_rec.mean()),
            "oracle@8_endpoint_arm": float(base_rec.mean()),
            "delta_vs_endpoint": boot_mean_diff(arm_rec, base_rec),
            "oracle@16_UNION_of_both_arms": float(union.mean()),
            "union_gain_over_endpoint": boot_mean_diff(union, base_rec),
            "rescued_of_endpoint_no_coverage": {
                "n_no_coverage": int((base_rec == 0).sum()),
                "n_rescued": int(((base_rec == 0) & (arm_rec == 1)).sum()),
                "rate": float(arm_rec[base_rec == 0].mean())},
        }
    # comparison anchor: an INDEPENDENT iid 8-sample redraw (first 8 of sc16) on the same cell
    d16 = load_judged(os.path.join(SCDIR, "ckpt_vqa_rad_open_lingshu7b_sc16"))
    ks = [k for k in end if k in d16]
    base_rec = np.array([1 if 1 in end[k]["sl"] else 0 for k in ks])
    iid_rec = np.array([1 if 1 in d16[k]["y"][:8] else 0 for k in ks])
    union = ((base_rec + iid_rec) > 0).astype(int)
    arms["iid_redraw_first8_of_sc16"] = {
        "label": "CONTROL: a second iid 8-sample draw at the SAME temp/prompt/resolution",
        "n": len(ks), "oracle@8_this_arm": float(iid_rec.mean()),
        "oracle@8_endpoint_arm": float(base_rec.mean()),
        "delta_vs_endpoint": boot_mean_diff(iid_rec, base_rec),
        "oracle@16_UNION_of_both_arms": float(union.mean()),
        "union_gain_over_endpoint": boot_mean_diff(union, base_rec),
        "rescued_of_endpoint_no_coverage": {
            "n_no_coverage": int((base_rec == 0).sum()),
            "n_rescued": int(((base_rec == 0) & (iid_rec == 1)).sum()),
            "rate": float(iid_rec[base_rec == 0].mean())},
    }
    out["arms"] = arms
    return out


# ======================================================================================
# D. IMAGE-side complementarity (resolution), exact-match LOWER BOUND
# ======================================================================================
def image_side(items):
    out = {"labels": "EXACT-MATCH (`oks`) only -- the cap80/cap160 dumps were never judged. "
                     "Exact match is strictly stronger than the judge, so every number here is a "
                     "LOWER BOUND on the judge-measured rescue.",
           "config": "cap80 = HIGH_PX/16, cap160 = HIGH_PX/8, endpoint pool = cap320 = HIGH_PX/4 "
                     "(src/labeling/run_openvqa.py:52, runners/run_openvqa_lingshu7b.sh); the "
                     "cap80/cap160 dumps are GREEDY (n_samples=1, temp 0), one answer each.",
           "per_cell": {}}
    sc8 = {ds: {} for ds in EVAL_DS}
    for ds in ["vqa_rad_open", "pathvqa_open"]:
        base = {}
        for l in open(os.path.join(SCDIR, f"ckpt_{ds}_lingshu7b_sc8.jsonl")):
            r = json.loads(l)
            base[str(r["idx"])] = r
        end = {str(it["idx"]): it for it in items if it["ds"] == ds}
        rows = {}
        for cap in ["cap80", "cap160"]:
            p = os.path.join(PERTDIR, f"ckpt_{ds}_lingshu7b_{cap}.jsonl")
            if not os.path.exists(p):
                continue
            d = {}
            for l in open(p):
                r = json.loads(l)
                d[str(r["idx"])] = r
            ks = [k for k in end if k in d and k in base]
            new_ans = exact_ok = rescue = 0
            base_exact_cov = 0
            nocov = 0
            nocov_new = nocov_rescue = 0
            for k in ks:
                pool = set(norm(a) for a in base[k]["preds"])
                a = norm(d[k]["preds"][0])
                ok = int(d[k]["oks"][0])
                bexact = int(max(base[k]["oks"]) == 1)
                base_exact_cov += bexact
                new_ans += int(a not in pool)
                exact_ok += ok
                rescue += int(ok == 1 and bexact == 0)
                if 1 not in end[k]["sl"]:            # judge-defined no-coverage
                    nocov += 1
                    nocov_new += int(a not in pool)
                    nocov_rescue += int(ok == 1)
            rows[cap] = {
                "n": len(ks),
                "share_answer_NOT_in_the_8_pool (new candidate)": new_ans / len(ks),
                "greedy_exact_acc_at_this_cap": exact_ok / len(ks),
                "pool_exact_oracle@8_at_cap320": base_exact_cov / len(ks),
                "exact_match_rescues_of_exact_no_coverage": rescue,
                "exact_match_rescue_rate_of_exact_no_coverage":
                    rescue / max(len(ks) - base_exact_cov, 1),
                "judge_no_coverage_subset": {
                    "n": nocov,
                    "share_answer_NOT_in_pool": nocov_new / max(nocov, 1),
                    "n_exact_match_correct (LOWER-BOUND rescue)": nocov_rescue,
                    "lower_bound_rescue_rate": nocov_rescue / max(nocov, 1)},
            }
        out["per_cell"][ds] = rows
    return out


# ======================================================================================
# E. laterality deep-dive
# ======================================================================================
def laterality(items):
    sc = {ds: {} for ds in EVAL_DS}
    for ds in EVAL_DS:
        for l in open(os.path.join(SCDIR, f"ckpt_{ds}_lingshu7b_sc8.jsonl")):
            r = json.loads(l)
            sc[ds][str(r["idx"])] = r
    rows = []
    for it in items:
        g = sc[it["ds"]][str(it["idx"])]
        gold = norm(g["gold"])
        gtok = set(re.findall(r"[a-z0-9]+", gold))
        ptoks = [set(re.findall(r"[a-z0-9]+", norm(a))) for a in it["preds"]]
        pool = set().union(*ptoks) if ptoks else set()
        is_lat = bool(gtok & LAT) or bool(pool & LAT)
        gold_lat = gtok & LAT
        rows.append({"ds": it["ds"], "rec": int(1 in it["sl"]), "is_lat": int(is_lat),
                     "gold_has_lat": int(bool(gold_lat)),
                     # did the pool produce the OPPOSITE laterality but nothing correct?
                     "pool_has_lat": int(bool(pool & LAT)),
                     "gold_lat_in_pool": int(bool(gold_lat and (gold_lat <= pool))),
                     "nonlat_gold_tokens_in_pool": int(bool((gtok - LAT) & pool)) if (gtok - LAT) else -1})
    R = rows
    lat = [r for r in R if r["is_lat"]]
    nolat = [r for r in R if not r["is_lat"]]
    latno = [r for r in lat if r["rec"] == 0]
    out = {
        "definition": "a question is 'laterality' if the gold answer or any pooled answer contains "
                      "one of {right,left,both,bilateral}",
        "n_laterality": len(lat), "share_of_pool": len(lat) / len(R),
        "oracle@8_laterality": float(np.mean([r["rec"] for r in lat])),
        "oracle@8_non_laterality": float(np.mean([r["rec"] for r in nolat])),
        "delta": boot_mean_diff([r["rec"] for r in lat], [r["rec"] for r in nolat])
        if len(lat) == len(nolat) else None,
        "laterality_no_coverage": {
            "n": len(latno),
            "share_of_all_no_coverage": len(latno) / sum(1 for r in R if r["rec"] == 0),
            "gold_laterality_token_present_in_pool": float(np.mean([r["gold_lat_in_pool"] for r in latno])),
            "note": "if the gold laterality word appears in the pool but the item is still "
                    "un-recoverable, the miss is in the REST of the answer, not the side.",
        },
        "per_cell": {},
    }
    for d in EVAL_DS:
        L = [r for r in R if r["ds"] == d and r["is_lat"]]
        N = [r for r in R if r["ds"] == d and not r["is_lat"]]
        out["per_cell"][d] = {
            "n_lat": len(L), "share": len(L) / max(len([r for r in R if r["ds"] == d]), 1),
            "oracle@8_lat": float(np.mean([r["rec"] for r in L])) if L else None,
            "oracle@8_nonlat": float(np.mean([r["rec"] for r in N])) if N else None,
            "n_lat_no_coverage": sum(1 for r in L if r["rec"] == 0),
        }
    return out


# ======================================================================================
# F. capture-recapture ceiling
# ======================================================================================
def ceiling(items):
    out = {"method": "Lincoln-Petersen / Chao two-sample: run A = the endpoint 8-sample pool, "
                     "run B = the INDEPENDENT sc16 16-sample pool, both JUDGE-labelled. "
                     "'Captured' = at least one correct answer in the run. Heterogeneous per-item "
                     "detection probability biases LP DOWNWARD, so this is a LOWER BOUND on the "
                     "share of questions reachable by iid sampling at any N.",
           "per_cell": {}}
    end = defaultdict(dict)
    for it in items:
        end[it["ds"]][str(it["idx"])] = it
    tots = defaultdict(float)
    for ds in EVAL_DS:
        d16 = load_judged(os.path.join(SCDIR, f"ckpt_{ds}_lingshu7b_sc16"))
        ks = [k for k in end[ds] if k in d16]
        A = np.array([1 if 1 in end[ds][k]["sl"] else 0 for k in ks])
        B = np.array([1 if 1 in d16[k]["y"][:16] else 0 for k in ks])
        nA, nB, nAB = int(A.sum()), int(B.sum()), int((A & B).sum())
        lp = nA * nB / nAB if nAB else float("nan")
        out["per_cell"][ds] = {
            "n": len(ks), "captured_A_8samples": nA, "captured_B_16samples": nB,
            "captured_both": nAB, "captured_either": int(((A + B) > 0).sum()),
            "LP_estimated_reachable_questions": lp,
            "LP_estimated_reachable_share": lp / len(ks),
            "observed_either_share": float(((A + B) > 0).mean()),
            "unreachable_share_upper_bound": 1 - lp / len(ks),
        }
        tots["n"] += len(ks); tots["lp"] += lp
        tots["either"] += ((A + B) > 0).sum()
    out["macro_LP_reachable_share"] = float(np.mean(
        [out["per_cell"][d]["LP_estimated_reachable_share"] for d in EVAL_DS]))
    out["macro_observed_either_share"] = float(np.mean(
        [out["per_cell"][d]["observed_either_share"] for d in EVAL_DS]))
    out["always_32b_direct_macro_open"] = 0.5982
    return out


def main():
    items = G.load_items()
    nt = G.null_test()
    res = {"title": "SCOUT B part 2 -- the coverage multiplier measured, and every free lever mined",
           "date": "2026-08-10", "no_gpu": True, "no_fabricated_numbers": True,
           "nboot": NBOOT, "seed": SEED,
           "null_test": {"pass": nt["pass"], "max_abs_deviation": nt["max_abs_deviation"]},
           "A_multiplier": multiplier(items),
           "B_oracle_to_32": oracle32(items),
           "C_text_side_alternatives": text_side(items),
           "D_image_side_resolution": image_side(items),
           "E_laterality": laterality(items),
           "F_ceiling": ceiling(items)}
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
    return res


if __name__ == "__main__":
    main()
