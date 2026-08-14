#!/usr/bin/env python3
"""decoding_sweep_dual_currency.py -- the SELECTED endpoint under BOTH grading currencies.

The 2026-08-13 sweep found one CI-clean SELECTED win (repetition_penalty=1.10, +0.0068) in the
project's frozen JUDGE currency, while the SAME setting is a CI-clean LOSS on oracle@8 in EXACT-MATCH
currency (-0.0132). The currency audit (decoding_sweep_currency_audit.py) showed the whole judge gain
is an increase in judge-minus-EM disagreement, driven by a compositional shift toward longer answers.

This script closes the loop by scoring the IDENTICAL verifier picks under both currencies:

    SELECTED_judge = judge label of the slot the frozen verifier picked   (the frozen endpoint)
    SELECTED_em    = EM    label of the SAME slot, same picks             (the robustness endpoint)

Picks come from genframe_data.sel_eff(...)["picks"] -- the frozen routine, not a re-implementation --
so the two currencies differ ONLY in the label applied to the picked slot. A win that is a genuine
generation improvement should survive both; a win that is a grading artifact will not.

Outputs results/cascade_methods/artifacts/_decoding_sweep_dual_currency.json
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G                       # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (                   # noqa: E402
    load_judge, load_vscores, load_pool, boot, DS, SWEEP, NBOOT, SEED)

ap = argparse.ArgumentParser()
ap.add_argument("--control", default="T07")
ap.add_argument("--settings", default="T03,T05,T07,T10,T13,minp005,minp01,rp105,rp11,"
                                      "topk20,topk50,topp09,topp095,T13minp010")
ap.add_argument("--out", default="results/cascade_methods/artifacts/_decoding_sweep_dual_currency.json")
A = ap.parse_args()

lab, vsc, ref = load_judge(), load_vscores(), G.load_items()


def seed_tags(setting):
    out = []
    for s in range(8):
        tag = f"{setting}_s{s}"
        if os.path.exists(os.path.join(SWEEP, f"ckpt_{DS[0]}_{tag}.jsonl")):
            out.append(tag)
    return out


def one_seed(tag):
    """Returns per-item picked-slot labels in BOTH currencies, or None if the pool/labels are short."""
    pool = load_pool(tag, strict=False)
    if pool is None:
        return None
    items, em_sl, miss_j, miss_v = [], [], 0, 0
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        sl, sc = [], []
        for a in r["preds"]:
            y = lab.get((it["ds"], it["idx"], G.norm(a)))
            if y is None:
                miss_j += 1; y = 0
            sl.append(int(y))
            k = (it["ds"], it["idx"], a)
            if k not in vsc:
                miss_v += 1
            sc.append(vsc.get(k, G.MISSING_SCORE))
        items.append({"ds": it["ds"], "idx": it["idx"], "sl": sl, "scores": sc,
                      "preds": r["preds"], "gold": r.get("gold", ""),
                      "greedy_ok": int(lab.get((it["ds"], it["idx"], G.norm(r["modal_pred"])), 0)),
                      "gen_tokens_all": r.get("gen_tokens_all", [])})
        em_sl.append(list(r["oks_em"]))
    # SELECTED needs a complete verifier pass AND a complete judge pass. Defaulting a missing judge
    # label to 0 (an earlier version did) silently deflates the judge currency; skipping unlabelled
    # slots instead biases toward duplicated/short candidates. Refuse the pool outright.
    if miss_v or miss_j:
        return {"incomplete_verifier": miss_v, "missing_judge": miss_j}
    res = G.sel_eff({(it["ds"], it["idx"]): it["scores"] for it in items}, items=items)
    picks = np.asarray(res["picks"], int)
    got_j = np.asarray(res["got"], int)
    got_e = np.array([em_sl[i][picks[i]] for i in range(len(items))], int)
    rec_j = np.asarray(res["rec"], int)
    rec_e = np.array([1 if 1 in x else 0 for x in em_sl], int)
    tok_pick = np.array([items[i]["gen_tokens_all"][picks[i]]
                         if items[i]["gen_tokens_all"] else 0 for i in range(len(items))], float)
    return {"got_judge": got_j, "got_em": got_e, "rec_judge": rec_j, "rec_em": rec_e,
            "picks": picks, "tok_picked": tok_pick,
            "ds_index": np.asarray(res["ds_index"], int),
            "sel_eff_judge": res["sel_eff"], "selected_judge": res["acc"],
            "oracle_judge": res["oracle"], "missing_judge": miss_j}


data = {}
for st in A.settings.split(","):
    seeds = {}
    for tag in seed_tags(st):
        r = one_seed(tag)
        if r is None or "incomplete_verifier" in r:
            if isinstance(r, dict):
                print(f"  [incomplete, excluded] {tag}: "
                      f"{r['missing_judge']} unjudged / {r['incomplete_verifier']} unscored slots")
            continue
        seeds[tag] = r
    if seeds:
        data[st] = seeds

C = A.control
out = {"title": "SELECTED under BOTH grading currencies, identical verifier picks",
       "why": __doc__.strip().split("\n\n")[1],
       "picks_from": "genframe_data.sel_eff(...)['picks'] -- the frozen routine",
       "currencies": {"judge": "MedVLThinker-32B judge (the frozen, deployed metric)",
                      "em": "normalised exact match, score_em() from run_openvqa.py"},
       "nboot": NBOOT, "bootstrap_seed": SEED, "per_setting": {}}

for st, seeds in data.items():
    out["per_setting"][st] = {
        "n_seeds": len(seeds), "seeds": sorted(seeds),
        "SELECTED_judge": float(np.mean([v["got_judge"].mean() for v in seeds.values()])),
        "SELECTED_judge_sd": float(np.std([v["got_judge"].mean() for v in seeds.values()], ddof=1))
                             if len(seeds) > 1 else 0.0,
        "SELECTED_em": float(np.mean([v["got_em"].mean() for v in seeds.values()])),
        "SELECTED_em_sd": float(np.std([v["got_em"].mean() for v in seeds.values()], ddof=1))
                          if len(seeds) > 1 else 0.0,
        "oracle_judge": float(np.mean([v["rec_judge"].mean() for v in seeds.values()])),
        "oracle_em": float(np.mean([v["rec_em"].mean() for v in seeds.values()])),
        "sel_eff_judge": float(np.mean([v["sel_eff_judge"] for v in seeds.values()])),
        "sel_eff_em": float(np.mean([v["got_em"].sum() / max(v["rec_em"].sum(), 1)
                                     for v in seeds.values()])),
        "mean_tokens_of_PICKED_slot": float(np.mean([v["tok_picked"].mean() for v in seeds.values()])),
    }

if C in data:
    cj = np.mean([v["got_judge"] for v in data[C].values()], axis=0)
    ce = np.mean([v["got_em"] for v in data[C].values()], axis=0)
    out["deltas_vs_control"] = {}
    for st, seeds in data.items():
        if st == C:
            continue
        sj = np.mean([v["got_judge"] for v in seeds.values()], axis=0)
        se = np.mean([v["got_em"] for v in seeds.values()], axis=0)
        dj, de = boot(sj, cj), boot(se, ce)
        out["deltas_vs_control"][st] = {
            "SELECTED_judge_delta": dj, "SELECTED_em_delta": de,
            "AGREE": bool(dj["verdict"] == de["verdict"]),
            "currency_conflict": bool((dj["verdict"] == "WIN" and de["verdict"] == "LOSS")
                                      or (dj["verdict"] == "LOSS" and de["verdict"] == "WIN")),
            "sign_flip_point_estimate": bool(np.sign(dj["delta"]) != np.sign(de["delta"])),
        }

os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1, default=float)
print(f"wrote {A.out}\n")
print(f"{'setting':12s} {'seeds':>5s} {'SEL_judge':>10s} {'SEL_em':>9s} {'orc_j':>7s} {'orc_em':>7s} "
      f"{'seff_j':>7s} {'seff_em':>7s} {'tok_pick':>8s}")
for st, b in out["per_setting"].items():
    print(f"{st:12s} {b['n_seeds']:5d} {b['SELECTED_judge']:10.6f} {b['SELECTED_em']:9.6f} "
          f"{b['oracle_judge']:7.4f} {b['oracle_em']:7.4f} {b['sel_eff_judge']:7.4f} "
          f"{b['sel_eff_em']:7.4f} {b['mean_tokens_of_PICKED_slot']:8.2f}")
if "deltas_vs_control" in out:
    print(f"\nvs control {C}:")
    for st, b in out["deltas_vs_control"].items():
        j, e = b["SELECTED_judge_delta"], b["SELECTED_em_delta"]
        print(f"  {st:12s} judge {j['delta']:+.5f} [{j['lo']:+.5f},{j['hi']:+.5f}] {j['verdict']:5s} | "
              f"em {e['delta']:+.5f} [{e['lo']:+.5f},{e['hi']:+.5f}] {e['verdict']:5s}"
              + ("   <<< CURRENCY CONFLICT" if b["currency_conflict"] else ""))
