#!/usr/bin/env python3
"""coverage_diagnosis.py -- SCOUT B, 2026-08-10.

Is the open-text coverage ceiling (oracle@8 = 0.626013 on the 2,345-item endpoint) a
DIVERSITY failure or a CAPABILITY failure -- and does the round's central arithmetic
("selected ~= greedy + sel_eff*(oracle-greedy)", i.e. coverage has a multiplier) actually
hold on the real data?

No GPU. Reads only existing dumps:
  ckpts/train/lora_verifier_disjoint/transfer_dump_*.json   (THE endpoint, judge labels)
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl   (== the endpoint pool)
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc16.jsonl  (INDEPENDENT 16-sample draw)
  ...both with *_scexploded.judge.jsonl giving judge labels at 100% coverage.

Everything is computed with the frozen metric in src/training_methods/genframe_data.py.

Run:  PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 src/cascade_methods/coverage_diagnosis.py
Out:  results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json
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

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json")
SCDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
EVAL_DS = G.EVAL_DS
RNG_SEED = 20260810
NBOOT = 10000

# always-32B-direct per open cell, verbatim from
# artifacts/cascade_selector_rerun_2026-08-05.json : per_arm.disjoint.per_cell_acc
DIRECT32B = {"slake_open": 0.8186, "vqa_rad_open": 0.6000, "pathvqa_open": 0.3760}
DIRECT32B_MACRO_OPEN = 0.5982   # per_arm.disjoint.open_only.macro_cells.always_32b_direct
DIRECT32B_MACRO8 = 0.6567       # per_arm.disjoint.macro_acc.always_32b_direct
METHOD_AM_MACRO8 = 0.6575       # per_arm.disjoint.macro_acc.method_accuracy_max_veto


def norm(s):
    return str(s).strip().lower()


# ======================================================================================
# 0. load
# ======================================================================================
def load_sc(ds, tag):
    """{idx -> {preds:[...], y:[...judge...], gold, question, greedy_ok? }} for an sc dump."""
    base = os.path.join(SCDIR, f"ckpt_{ds}_lingshu7b_{tag}")
    jd = {}
    for l in open(base + "_scexploded.judge.jsonl"):
        r = json.loads(l)
        jd[r["idx"]] = int(r["judge_ok"])
    out = {}
    for l in open(base + ".jsonl"):
        r = json.loads(l)
        lab = {}
        for s, a in enumerate(r["preds"]):
            na = norm(a)
            if na not in lab:
                k = f"{r['idx']}#{s}"
                if k not in jd:
                    raise KeyError(f"missing judge for {k} in {base}")
                lab[na] = jd[k]
        out[str(r["idx"])] = {
            "preds": list(r["preds"]),
            "y": [lab[norm(a)] for a in r["preds"]],
            "gold": r.get("gold"),
            "question": r.get("question"),
            "exact_oks": [int(v) for v in r["oks"]],
        }
    return out


# ======================================================================================
# 1. NULL TESTS
# ======================================================================================
def null_tests(items):
    nt = G.null_test()
    # the sc8 dump must reconstruct the endpoint pool + labels exactly
    devs = {}
    for ds, short in zip(EVAL_DS, ["slake", "vqa_rad", "pathvqa"]):
        sc8 = load_sc(ds, "sc8")
        n_pred_mismatch = n_lab_mismatch = n = 0
        for it in items:
            if it["ds"] != ds:
                continue
            n += 1
            r = sc8[str(it["idx"])]
            if [str(x) for x in it["preds"]] != [str(x) for x in r["preds"]]:
                n_pred_mismatch += 1
            if [int(x) for x in it["sl"]] != r["y"]:
                n_lab_mismatch += 1
        devs[ds] = {"n": n, "pred_mismatch": n_pred_mismatch, "judge_label_mismatch": n_lab_mismatch}
    return {
        "frozen_metric_null_test": {"pass": nt["pass"], "max_abs_deviation": nt["max_abs_deviation"],
                                    "measured": nt["measured"]},
        "sc8_dump_is_the_endpoint_pool": devs,
        "note": ("sc8 preds reproduce the transfer-dump pool item-for-item and the judge labels "
                 "reconstructed from *_scexploded.judge.jsonl reproduce transfer-dump `sl` exactly; "
                 "the sc8 file's own `oks` field is EXACT-MATCH correctness and does NOT equal the "
                 "judge label -- never mix the two."),
    }


# ======================================================================================
# 2. THE ARITHMETIC
# ======================================================================================
def arithmetic(items):
    """selected = oracle x sel_eff is an IDENTITY; the brief's additive form is not."""
    inc = G.incumbent_scores()
    r = G.sel_eff(inc, items)
    got, rec = r["got"], r["rec"]
    greedy = r["greedy"]

    def cell(mask):
        o = float(rec[mask].mean())
        a = float(got[mask].mean())
        se = float(got[mask & (rec == 1)].mean())
        return o, a, se

    dsi = r["ds_index"]
    rows = []
    for j, ds in enumerate(EVAL_DS):
        m = dsi == j
        o, a, se = cell(m)
        gr = float(np.mean([it["greedy_ok"] for it, k in zip(items, m) if k]))
        rows.append({"cell": ds, "n": int(m.sum()), "greedy": gr, "oracle@8": o,
                     "selected": a, "sel_eff": se,
                     "pred_multiplicative_oracle_x_seleff": o * se,
                     "err_multiplicative": a - o * se,
                     "pred_brief_additive": gr + se * (o - gr),
                     "err_brief_additive": a - (gr + se * (o - gr))})
    o, a, se = cell(np.ones(len(items), bool))
    rows.append({"cell": "POOLED", "n": len(items), "greedy": greedy, "oracle@8": o,
                 "selected": a, "sel_eff": se,
                 "pred_multiplicative_oracle_x_seleff": o * se, "err_multiplicative": a - o * se,
                 "pred_brief_additive": greedy + se * (o - greedy),
                 "err_brief_additive": a - (greedy + se * (o - greedy))})

    # --- across N: prefix subpools of the SAME 8-sample pool, incumbent scores restricted
    byN = []
    S = np.array([list(it["scores"]) for it in items], float)
    SL = np.array([[int(v) for v in it["sl"]] for it in items], int)
    for k in range(1, 9):
        pk = np.argmax(S[:, :k], axis=1)
        gk = SL[np.arange(len(items)), pk]
        rk = (SL[:, :k].max(axis=1) == 1).astype(int)
        pur = SL[:, :k].mean(axis=1)
        byN.append({"N": k, "oracle": float(rk.mean()), "selected": float(gk.mean()),
                    "sel_eff": float(gk[rk == 1].mean()),
                    "purity_given_recoverable": float(pur[rk == 1].mean()),
                    "random_pick_acc": float(pur.mean())})
    return {"per_cell": rows, "prefix_pool_curve": byN}


# ======================================================================================
# 3. ORACLE@N from the independent 16-sample draw
# ======================================================================================
def oracleN(items):
    end = defaultdict(dict)
    for it in items:
        end[it["ds"]][str(it["idx"])] = it
    out = {"per_cell": {}, "macro": {}, "sample_weighted": {}}
    curves16, curves8, keys = {}, {}, {}
    for ds in EVAL_DS:
        sc16 = load_sc(ds, "sc16")
        sc8 = load_sc(ds, "sc8")
        ks = sorted(set(sc16) & set(end[ds]))
        keys[ds] = ks
        Y16 = np.array([sc16[k]["y"][:16] for k in ks], int)
        Y8 = np.array([sc8[k]["y"][:8] for k in ks], int)
        curves16[ds] = Y16
        curves8[ds] = Y8
        # expected oracle@n over random n-subsets of the 16, closed form (hypergeometric)
        c = Y16.sum(axis=1)                       # #correct of 16
        exp = []
        for n in range(1, 17):
            # P(at least one correct in a random n-subset) = 1 - C(16-c, n)/C(16, n)
            p = []
            for ci in c:
                if ci == 0:
                    p.append(0.0)
                elif 16 - ci < n:
                    p.append(1.0)
                else:
                    num = 1.0
                    for t in range(n):
                        num *= (16 - ci - t) / (16 - t)
                    p.append(1.0 - num)
            exp.append(float(np.mean(p)))
        out["per_cell"][ds] = {
            "n": len(ks),
            "oracle_at_N_subset_expectation_1..16": [round(v, 6) for v in exp],
            "oracle@8_sc16run": exp[7], "oracle@16_sc16run": exp[15],
            "delta_16_minus_8": exp[15] - exp[7],
            "oracle@8_endpoint_judge": float(np.mean([1 if 1 in end[ds][k]["sl"] else 0 for k in ks])),
            "union_endpoint8_plus_sc16_oracle@24": float(np.mean(
                [1 if (1 in end[ds][k]["sl"] or Y16[i].max() == 1) else 0 for i, k in enumerate(ks)])),
            "greedy": float(np.mean([end[ds][k]["greedy_ok"] for k in ks])),
            "always_32b_direct": DIRECT32B[ds],
        }
    ns = {ds: out["per_cell"][ds]["n"] for ds in EVAL_DS}
    tot = sum(ns.values())
    for field in ["oracle@8_sc16run", "oracle@16_sc16run", "oracle@8_endpoint_judge",
                  "union_endpoint8_plus_sc16_oracle@24", "greedy", "always_32b_direct"]:
        out["macro"][field] = float(np.mean([out["per_cell"][d][field] for d in EVAL_DS]))
        out["sample_weighted"][field] = float(
            sum(out["per_cell"][d][field] * ns[d] for d in EVAL_DS) / tot)
    out["macro"]["oracle_at_N_1..16"] = [
        float(np.mean([out["per_cell"][d]["oracle_at_N_subset_expectation_1..16"][i]
                       for d in EVAL_DS])) for i in range(16)]
    return out, curves16, curves8, keys


# ======================================================================================
# 4. DIVERSITY vs CAPABILITY on the no-coverage subset
# ======================================================================================
def ent(counts):
    p = np.array(counts, float)
    p = p / p.sum()
    p = p[p > 0]
    if len(p) == 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def diversity_vs_capability(items, curves16, keys):
    nd = np.array([len(set(norm(a) for a in it["preds"])) for it in items])
    rec = np.array([1 if 1 in it["sl"] else 0 for it in items])
    dsi = np.array([EVAL_DS.index(it["ds"]) for it in items])
    modal_share = np.array([max(Counter(norm(a) for a in it["preds"]).values()) / 8.0 for it in items])
    entr = np.array([ent(list(Counter(norm(a) for a in it["preds"]).values())) for it in items])

    hist_all = dict(sorted(Counter(nd.tolist()).items()))
    hist_rec = dict(sorted(Counter(nd[rec == 1].tolist()).items()))
    hist_no = dict(sorted(Counter(nd[rec == 0].tolist()).items()))

    # cross-run test: is a no-coverage item recoverable in the INDEPENDENT 16-sample draw?
    end = defaultdict(dict)
    for it in items:
        end[it["ds"]][str(it["idx"])] = it
    cross = {}
    tot = defaultdict(int)
    for ds in EVAL_DS:
        ks = keys[ds]
        Y16 = curves16[ds]
        pos = {k: i for i, k in enumerate(ks)}
        n_no = n_rescued = n_rescued_nd1 = n_no_nd1 = 0
        n_rescued_ndhi = n_no_ndhi = 0
        for it in [x for x in items if x["ds"] == ds]:
            k = str(it["idx"])
            if k not in pos:
                continue
            if 1 in it["sl"]:
                continue
            n_no += 1
            d1 = len(set(norm(a) for a in it["preds"])) == 1
            dhi = len(set(norm(a) for a in it["preds"])) >= 6
            r16 = Y16[pos[k]].max() == 1
            n_rescued += int(r16)
            n_no_nd1 += int(d1)
            n_rescued_nd1 += int(d1 and r16)
            n_no_ndhi += int(dhi)
            n_rescued_ndhi += int(dhi and r16)
        cross[ds] = {
            "n_no_coverage_at_8": n_no,
            "rescued_by_independent_16": n_rescued,
            "rescue_rate": n_rescued / max(n_no, 1),
            "unanimous_nd1_subset": {"n": n_no_nd1, "rescued": n_rescued_nd1,
                                     "rescue_rate": n_rescued_nd1 / max(n_no_nd1, 1)},
            "high_diversity_nd>=6_subset": {"n": n_no_ndhi, "rescued": n_rescued_ndhi,
                                            "rescue_rate": n_rescued_ndhi / max(n_no_ndhi, 1)},
        }
        tot["n"] += n_no
        tot["r"] += n_rescued
        tot["n1"] += n_no_nd1
        tot["r1"] += n_rescued_nd1
        tot["nh"] += n_no_ndhi
        tot["rh"] += n_rescued_ndhi

    strata = {}
    for lo, hi, name in [(1, 1, "nd=1 (one answer 8x)"), (2, 3, "nd=2-3"),
                         (4, 5, "nd=4-5"), (6, 8, "nd=6-8 (near-max exploration)")]:
        m = (nd >= lo) & (nd <= hi)
        strata[name] = {"n": int(m.sum()), "share_of_all": float(m.mean()),
                        "oracle@8": float(rec[m].mean()),
                        "n_no_coverage": int((m & (rec == 0)).sum()),
                        "share_of_no_coverage": float((m & (rec == 0)).sum() / (rec == 0).sum())}
    return {
        "n_distinct_hist_all": hist_all,
        "n_distinct_hist_recoverable": hist_rec,
        "n_distinct_hist_no_coverage": hist_no,
        "mean_n_distinct": {"all": float(nd.mean()), "recoverable": float(nd[rec == 1].mean()),
                            "no_coverage": float(nd[rec == 0].mean())},
        "mean_modal_share": {"all": float(modal_share.mean()),
                             "recoverable": float(modal_share[rec == 1].mean()),
                             "no_coverage": float(modal_share[rec == 0].mean())},
        "mean_normalized_entropy": {"all": float(entr.mean()),
                                    "recoverable": float(entr[rec == 1].mean()),
                                    "no_coverage": float(entr[rec == 0].mean())},
        "strata_by_n_distinct": strata,
        "cross_run_rescue": cross,
        "cross_run_rescue_pooled": {
            "n_no_coverage_at_8": tot["n"], "rescued_by_independent_16": tot["r"],
            "rescue_rate": tot["r"] / max(tot["n"], 1),
            "unanimous_nd1": {"n": tot["n1"], "rescued": tot["r1"],
                              "rescue_rate": tot["r1"] / max(tot["n1"], 1)},
            "high_diversity_nd>=6": {"n": tot["nh"], "rescued": tot["rh"],
                                     "rescue_rate": tot["rh"] / max(tot["nh"], 1)},
        },
        "_arrays": {"nd": nd, "rec": rec, "dsi": dsi},
    }


# ======================================================================================
# 5. SIGNATURE of the no-coverage subset
# ======================================================================================
LAT = {"right", "left", "both", "bilateral", "r", "l"}
YESNO = {"yes", "no"}


def tokset(s):
    return set(re.findall(r"[a-z0-9]+", norm(s)))


def f1(a, b):
    A, B = tokset(a), tokset(b)
    if not A or not B:
        return 0.0
    i = len(A & B)
    if i == 0:
        return 0.0
    p, r = i / len(A), i / len(B)
    return 2 * p * r / (p + r)


def signature(items, sc_gold):
    rec = np.array([1 if 1 in it["sl"] else 0 for it in items])
    out = {}
    rows = []
    for it in items:
        g = sc_gold[it["ds"]][str(it["idx"])]
        gold = str(g["gold"])
        preds = [str(a) for a in it["preds"]]
        nw_gold = len(re.findall(r"[a-z0-9]+", norm(gold)))
        nw_pred = float(np.mean([len(re.findall(r"[a-z0-9]+", norm(a))) for a in preds]))
        best_f1 = max(f1(a, gold) for a in preds)
        pool_tokens = set().union(*[tokset(a) for a in preds])
        gold_tok = tokset(gold)
        gold_recall = len(gold_tok & pool_tokens) / max(len(gold_tok), 1)
        lat = any(tokset(a) & LAT for a in preds) or bool(gold_tok & LAT)
        yn = (norm(gold) in YESNO) or all(norm(a).rstrip(".") in YESNO for a in preds)
        rows.append({"ds": it["ds"], "rec": int(1 in it["sl"]), "gold_words": nw_gold,
                     "pred_words": nw_pred, "best_f1": best_f1, "gold_tok_recall": gold_recall,
                     "laterality": int(lat), "yesno": int(yn),
                     "nd": len(set(norm(a) for a in preds))})
    R = rows
    def agg(sel, name):
        S = [r for r in R if sel(r)]
        if not S:
            return None
        return {"n": len(S),
                "mean_gold_words": float(np.mean([r["gold_words"] for r in S])),
                "mean_pred_words": float(np.mean([r["pred_words"] for r in S])),
                "share_gold_<=3_words": float(np.mean([r["gold_words"] <= 3 for r in S])),
                "mean_best_token_f1_vs_gold": float(np.mean([r["best_f1"] for r in S])),
                "share_best_f1_zero (NO gold token anywhere in pool)": float(
                    np.mean([r["best_f1"] == 0 for r in S])),
                "share_best_f1_>=0.5 (NEAR MISS)": float(np.mean([r["best_f1"] >= 0.5 for r in S])),
                "mean_gold_token_recall_over_pool": float(np.mean([r["gold_tok_recall"] for r in S])),
                "share_laterality": float(np.mean([r["laterality"] for r in S])),
                "share_yesno": float(np.mean([r["yesno"] for r in S])),
                "mean_n_distinct": float(np.mean([r["nd"] for r in S]))}
    out["all"] = agg(lambda r: True, "all")
    out["recoverable"] = agg(lambda r: r["rec"] == 1, "rec")
    out["no_coverage"] = agg(lambda r: r["rec"] == 0, "no")
    out["per_cell_no_coverage"] = {d: agg(lambda r, d=d: r["rec"] == 0 and r["ds"] == d, d)
                                   for d in EVAL_DS}
    out["per_cell_recoverable"] = {d: agg(lambda r, d=d: r["rec"] == 1 and r["ds"] == d, d)
                                   for d in EVAL_DS}
    # laterality / yesno slice: coverage inside them
    lat_m = np.array([r["laterality"] for r in R], bool)
    yn_m = np.array([r["yesno"] for r in R], bool)
    out["slices"] = {
        "laterality": {"n": int(lat_m.sum()), "share_of_pool": float(lat_m.mean()),
                       "oracle@8": float(rec[lat_m].mean())},
        "yesno": {"n": int(yn_m.sum()), "share_of_pool": float(yn_m.mean()),
                  "oracle@8": float(rec[yn_m].mean())},
        "other": {"n": int((~lat_m & ~yn_m).sum()),
                  "oracle@8": float(rec[~lat_m & ~yn_m].mean())},
    }
    out["_rows"] = R
    return out


# ======================================================================================
# 6. what would it take -- the corrected projection
# ======================================================================================
def projection(items, arith, orc):
    """At FIXED sel_eff, selected = oracle x sel_eff. Solve for the oracle that ties/beats
    always-32B-direct per open cell and on the 8-cell macro."""
    per = {r["cell"]: r for r in arith["per_cell"]}
    rows = {}
    for ds in EVAL_DS:
        se = per[ds]["sel_eff"]
        o = per[ds]["oracle@8"]
        need = DIRECT32B[ds] / se
        rows[ds] = {"sel_eff": se, "oracle@8": o, "selected_now": per[ds]["selected"],
                    "always_32b_direct": DIRECT32B[ds],
                    "oracle_needed_to_TIE_32b_direct_at_fixed_sel_eff": need,
                    "oracle_gap": need - o,
                    "achievable_within_oracle@16": need <= orc["per_cell"][ds]["oracle@16_sc16run"],
                    "oracle@16_sc16run": orc["per_cell"][ds]["oracle@16_sc16run"],
                    "oracle@inf_upper_bound_union24": orc["per_cell"][ds]["union_endpoint8_plus_sc16_oracle@24"]}
    macro_now = float(np.mean([rows[d]["selected_now"] for d in EVAL_DS]))
    rows["_macro_open"] = {
        "selected_now": macro_now, "always_32b_direct": DIRECT32B_MACRO_OPEN,
        "gap": DIRECT32B_MACRO_OPEN - macro_now,
        "macro8_effect_of_closing_the_open_gap": (DIRECT32B_MACRO_OPEN - macro_now) * 3 / 8,
    }
    return rows


# ======================================================================================
def main():
    items = G.load_items()
    res = {"title": "SCOUT B -- coverage diagnosis: does raising oracle@N have a multiplier, "
                    "and is the 8-sample coverage ceiling diversity or capability?",
           "date": "2026-08-10", "no_gpu": True, "no_fabricated_numbers": True,
           "seed": RNG_SEED, "nboot": NBOOT,
           "sources": {
               "endpoint": "ckpts/train/lora_verifier_disjoint/transfer_dump_{slake,vqa_rad,pathvqa}_open_lingshu7b.json",
               "metric": "src/training_methods/genframe_data.py (frozen)",
               "sc8": "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl (+ _scexploded.judge.jsonl)",
               "sc16": "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc16.jsonl (+ _scexploded.judge.jsonl) -- INDEPENDENT draw",
               "baselines": "results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05.json per_arm.disjoint",
           }}

    res["null_tests"] = null_tests(items)
    res["arithmetic"] = arith = arithmetic(items)
    orc, curves16, curves8, keys = oracleN(items)
    res["oracle_at_N"] = orc
    dvc = diversity_vs_capability(items, curves16, keys)
    arrs = dvc.pop("_arrays")
    res["diversity_vs_capability"] = dvc
    sc_gold = {ds: load_sc(ds, "sc8") for ds in EVAL_DS}
    sig = signature(items, sc_gold)
    sig.pop("_rows")
    res["signature"] = sig
    res["projection"] = projection(items, arith, orc)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
    return res


if __name__ == "__main__":
    r = main()
    import pprint
    print("\n=== ARITHMETIC ===")
    for row in r["arithmetic"]["per_cell"]:
        print(f"{row['cell']:14s} n={row['n']:5d} greedy={row['greedy']:.4f} oracle={row['oracle@8']:.4f} "
              f"sel_eff={row['sel_eff']:.4f} selected={row['selected']:.6f} | "
              f"mult_pred={row['pred_multiplicative_oracle_x_seleff']:.6f} err={row['err_multiplicative']:+.2e} | "
              f"brief_pred={row['pred_brief_additive']:.6f} err={row['err_brief_additive']:+.4f}")
    print("\n=== PREFIX-POOL CURVE (same 8-pool, incumbent scores) ===")
    for b in r["arithmetic"]["prefix_pool_curve"]:
        print(b)
