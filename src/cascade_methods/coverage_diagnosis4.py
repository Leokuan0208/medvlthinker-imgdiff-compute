#!/usr/bin/env python3
"""coverage_diagnosis4.py -- SCOUT B part 4, 2026-08-10.

PER-SAMPLE RESCUE YIELD: given a question the endpoint 8-pool does not cover, what is the
probability that ONE more sample covers it -- as a function of where that sample comes from?

  * one more IID sample at the deployed config (cap320 / temp 0.7 / base prompt)
  * one GREEDY sample from a DIFFERENT TEXT config  (few-shot-5 prompt; JUDGE-labelled;
    ckpts/openvqa/cheap_lingshu7b/ckpt_{slake,pathvqa}_open_lingshu7b_fs5)
  * one GREEDY sample at a DIFFERENT RESOLUTION     (cap80 / cap160; EXACT-MATCH only)
  * the 15-way prompt x temperature portfolio        (EXACT-MATCH only)

This is the fair currency: everything is normalised to "per extra generated sample", so a
1.88x-cost portfolio cannot look better just by spending more.

Appends to results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))
from training_methods import genframe_data as G  # noqa: E402

ART = os.path.join(ROOT, "results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json")
SCDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
SCALEDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b_scale")
PERTDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b_perturb")
DIVDIR = os.path.join(ROOT, "ckpts/openvqa/diverse")
EVAL_DS = G.EVAL_DS
NBOOT = 10000
SEED = 20260810


def norm(s):
    return str(s).strip().lower()


def boot_rate(x, nboot=NBOOT, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"rate": None, "n": 0, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    n = len(x)
    d = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    return {"rate": float(x.mean()), "n": int(n),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    items = G.load_items()
    end = defaultdict(dict)
    for it in items:
        end[it["ds"]][str(it["idx"])] = it

    out = {"question": "per ONE extra generated sample, P(covers a question the deployed 8-pool "
                       "does not cover). Normalised per sample so a bigger budget cannot win by "
                       "spending more.",
           "iid_baseline": {}, "alternative_configs": {}, "notes": []}

    # ---------------- IID baseline: the 8th sample of the deployed pool -------------------
    SL = np.array([[int(v) for v in it["sl"]] for it in items], int)
    ds_i = np.array([EVAL_DS.index(it["ds"]) for it in items])
    for j, d in enumerate(EVAL_DS):
        m = ds_i == j
        no7 = m & (SL[:, :7].max(1) == 0)
        resc = (SL[no7, 7] == 1).astype(int)
        out["iid_baseline"][d] = dict(
            boot_rate(resc),
            what="P(sample #8 is correct | samples 1..7 all wrong), JUDGE labels, deployed config")
    no7 = SL[:, :7].max(1) == 0
    out["iid_baseline"]["POOLED"] = dict(boot_rate((SL[no7, 7] == 1).astype(int)),
                                         what="pooled over the 3 cells")
    out["iid_baseline"]["MACRO"] = float(np.mean(
        [out["iid_baseline"][d]["rate"] for d in EVAL_DS]))

    # ---------------- few-shot-5 (DIFFERENT TEXT CONFIG), JUDGE labels -------------------
    fs = {}
    for ds in ["slake_open", "pathvqa_open"]:
        p = os.path.join(SCDIR, f"ckpt_{ds}_lingshu7b_fs5")
        if not os.path.exists(p + ".jsonl"):
            continue
        jd = {str(json.loads(l)["idx"]): int(json.loads(l)["judge_ok"])
              for l in open(p + ".judge.jsonl")}
        pr = {str(json.loads(l)["idx"]): json.loads(l)["modal_pred"] for l in open(p + ".jsonl")}
        ks = [k for k in end[ds] if k in jd]
        nocov = [k for k in ks if 1 not in end[ds][k]["sl"]]
        resc = np.array([jd[k] for k in nocov], int)
        newans = np.array([int(norm(pr[k]) not in set(norm(a) for a in end[ds][k]["preds"]))
                           for k in nocov], int)
        fs[ds] = {"label": "few-shot-5 prompt, GREEDY, 1 sample, JUDGE-labelled",
                  "n_matched": len(ks),
                  "standalone_acc_on_all_items": float(np.mean([jd[k] for k in ks])),
                  "rescue_of_endpoint_no_coverage": dict(boot_rate(resc)),
                  "share_answer_new_to_the_pool": float(newans.mean())}
    out["alternative_configs"]["few_shot_5_prompt_JUDGE"] = fs

    # ---------------- vqa_rad-only judge-labelled arms, per-sample ------------------------
    va = {}
    for tag, lbl, nsamp in [("t10", "temp 1.0 pool (8 samples)", 8),
                            ("think", "reasoning-mode pool (8 samples)", 8)]:
        base = os.path.join(SCALEDIR, f"ckpt_vqa_rad_open_lingshu7b_{tag}")
        jd = {json.loads(l)["idx"]: int(json.loads(l)["judge_ok"])
              for l in open(base + "_scexploded.judge.jsonl")}
        rows = {}
        for l in open(base + ".jsonl"):
            r = json.loads(l)
            lab = {}
            for s, a in enumerate(r["preds"]):
                na = norm(a)
                if na not in lab:
                    lab[na] = jd[f"{r['idx']}#{s}"]
            rows[str(r["idx"])] = [lab[norm(a)] for a in r["preds"]]
        ks = [k for k in end["vqa_rad_open"] if k in rows]
        nocov = [k for k in ks if 1 not in end["vqa_rad_open"][k]["sl"]]
        Y = np.array([rows[k][:nsamp] for k in nocov], int)
        # per-sample yield: P(a single random sample of this arm is correct | endpoint no-coverage)
        per_sample = Y.mean(1)
        # whole-arm yield: P(at least one of the 8 is correct | endpoint no-coverage)
        whole = (Y.max(1) == 1).astype(int)
        va[tag] = {"label": lbl,
                   "per_sample_rescue": dict(boot_rate(per_sample)),
                   "whole_8_sample_arm_rescue": dict(boot_rate(whole)),
                   "n_no_coverage": len(nocov)}
    # iid control on the same cell: first 8 of the independent sc16
    base = os.path.join(SCDIR, "ckpt_vqa_rad_open_lingshu7b_sc16")
    jd = {json.loads(l)["idx"]: int(json.loads(l)["judge_ok"])
          for l in open(base + "_scexploded.judge.jsonl")}
    rows = {}
    for l in open(base + ".jsonl"):
        r = json.loads(l)
        lab = {}
        for s, a in enumerate(r["preds"]):
            na = norm(a)
            if na not in lab:
                lab[na] = jd[f"{r['idx']}#{s}"]
        rows[str(r["idx"])] = [lab[norm(a)] for a in r["preds"]]
    ks = [k for k in end["vqa_rad_open"] if k in rows]
    nocov = [k for k in ks if 1 not in end["vqa_rad_open"][k]["sl"]]
    Y = np.array([rows[k][:8] for k in nocov], int)
    va["iid_control_first8_of_sc16"] = {
        "label": "CONTROL: independent iid 8-sample draw, SAME config",
        "per_sample_rescue": dict(boot_rate(Y.mean(1))),
        "whole_8_sample_arm_rescue": dict(boot_rate((Y.max(1) == 1).astype(int))),
        "n_no_coverage": len(nocov)}
    out["alternative_configs"]["vqa_rad_open_JUDGE_arms"] = va

    # ---------------- resolution, EXACT-MATCH lower bound ---------------------------------
    rs = {}
    for ds in ["vqa_rad_open", "pathvqa_open"]:
        rs[ds] = {}
        for cap in ["cap80", "cap160"]:
            p = os.path.join(PERTDIR, f"ckpt_{ds}_lingshu7b_{cap}.jsonl")
            if not os.path.exists(p):
                continue
            d = {str(json.loads(l)["idx"]): json.loads(l) for l in open(p)}
            ks = [k for k in end[ds] if k in d]
            nocov = [k for k in ks if 1 not in end[ds][k]["sl"]]
            resc = np.array([int(d[k]["oks"][0]) for k in nocov], int)
            rs[ds][cap] = dict(boot_rate(resc),
                               label=f"{cap} GREEDY, 1 sample, EXACT-MATCH label (LOWER BOUND)")
    out["alternative_configs"]["resolution_EXACTMATCH_lower_bound"] = rs

    # ---------------- 15-way prompt x temp portfolio, EXACT-MATCH -------------------------
    dv = {}
    for ds in EVAL_DS:
        p = os.path.join(DIVDIR, f"ckpt_{ds}_lingshu7b_div.jsonl")
        if not os.path.exists(p):
            continue
        d = {str(json.loads(l)["idx"]): json.loads(l) for l in open(p)}
        # exact-match currency on BOTH sides so the comparison is internally consistent
        b8 = {}
        for l in open(os.path.join(SCDIR, f"ckpt_{ds}_lingshu7b_sc8.jsonl")):
            r = json.loads(l)
            b8[str(r["idx"])] = r
        ks = [k for k in d if k in b8]
        nocov = [k for k in ks if max(b8[k]["oks"]) == 0]
        Y = np.array([d[k]["oks"] for k in nocov], int)
        dv[ds] = {"label": "5-prompt x 3-temp portfolio, M=15, EXACT-MATCH labels on both sides",
                  "n_matched": len(ks), "n_exactmatch_no_coverage_at_8": len(nocov),
                  "per_sample_rescue": dict(boot_rate(Y.mean(1))),
                  "whole_15_sample_arm_rescue": dict(boot_rate((Y.max(1) == 1).astype(int))),
                  "iid_per_sample_control_EXACTMATCH": None}
        # matched iid control in the SAME currency: P(sample 8 exact-correct | 1..7 all wrong)
        E = np.array([b8[k]["oks"] for k in ks], int)
        no7 = E[:, :7].max(1) == 0
        dv[ds]["iid_per_sample_control_EXACTMATCH"] = dict(boot_rate((E[no7, 7] == 1).astype(int)))
    out["alternative_configs"]["prompt_temp_portfolio_EXACTMATCH"] = dv

    out["notes"] = [
        "The four alternative-config families are NOT all on the same label currency. "
        "few_shot_5 and vqa_rad_open_JUDGE_arms use the endpoint's JUDGE labels and are directly "
        "comparable to iid_baseline. resolution_* and prompt_temp_portfolio_* use EXACT-MATCH on "
        "both sides of their own comparison, so they are internally consistent but NOT comparable "
        "to the judge-labelled rows.",
        "pathvqa_open's diverse dump covers only 178 of the 1,500 endpoint questions.",
        "A per-sample rescue rate is P(this one sample is correct | the deployed 8-pool covers "
        "nothing). It is the honest unit: an M=15 portfolio that rescues 20% of no-coverage "
        "questions using 15 samples is worse per sample than an 8-sample arm that rescues 15%.",
    ]

    art = json.load(open(ART))
    art["part4_per_sample_rescue_yield"] = out
    json.dump(art, open(ART, "w"), indent=1, default=float)
    print("wrote", ART)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
