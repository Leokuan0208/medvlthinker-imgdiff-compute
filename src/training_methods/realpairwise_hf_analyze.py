#!/usr/bin/env python3
"""realpairwise_hf_analyze.py -- score the HF (full-adapter) real-pairwise arm.

The HF arm is the ENGINE-MATCHED one: same stack, same adapter (vision tower included),
same max_pixels as the incumbent bar 0.775204. Its pathvqa half runs on a PRE-REGISTERED
500-question subsample (rng(0).choice(1500,500)), so this script evaluates every arm --
including the incumbent -- on exactly the COVERED item subset, paired item by item. It
never mixes a full-pool incumbent number with a subsample pairwise number.

  python3 src/training_methods/realpairwise_hf_analyze.py
"""
import os, sys, json, argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402
from src.training_methods.realpairwise_clean_analyze import (  # noqa: E402
    load_ordered, build_matrices, borda_scores, copeland_wins, knockout_pick_det,
    knockout_pick_stoch, ART)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hf")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--subsample", type=int, default=500,
                    help="the PRE-REGISTERED pathvqa subsample size actually run (0 = full pool)")
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ART, "verifarch_realpairwise_hf_2026-08-05.json"))
    ap.add_argument("--teacher", default=os.path.join(ART, "realpairwise_teacher_pmatrix_hf_2026-08-05.jsonl"))
    A = ap.parse_args()

    nt = G.null_test()
    print(f"[null] pass={nt['pass']} max_abs_dev={nt['max_abs_deviation']:.3e}", flush=True)
    if not nt["pass"]:
        sys.exit("NULL TEST FAILED")

    all_items = G.load_items()
    P, meta = load_ordered(A.tag)

    # The pathvqa half was run on the PRE-REGISTERED subsample only. A unanimous (k==1)
    # question needs no pairs, so pair-coverage alone would silently admit unanimous
    # pathvqa questions that were never in the sample -- and those score 1.0 for every
    # selector, which would inflate the pathvqa cell. Enforce the registered draw itself.
    pv = [it for it in all_items if it["ds"] == "pathvqa_open"]
    sel = set(int(i) for i in np.sort(np.random.default_rng(A.subsample_seed).choice(
        len(pv), min(A.subsample, len(pv)), replace=False))) if A.subsample else None
    pv_keep = {(pv[i]["ds"], pv[i]["idx"]) for i in sel} if sel is not None else None

    covered = []
    for it in all_items:
        if pv_keep is not None and it["ds"] == "pathvqa_open" and (it["ds"], it["idx"]) not in pv_keep:
            continue
        k = len(set(G.norm(a) for a in it["preds"]))
        exp = k * (k - 1) // 2
        pm = P.get((it["ds"], str(it["idx"])), {})
        got = sum(1 for v in pm.values() if 0 in v and 1 in v)
        if got == exp:
            covered.append(it)
    print(f"[cov] {len(covered)}/{len(all_items)} items fully covered by the {A.tag} arm", flush=True)
    per_ds_n = defaultdict(int)
    for it in covered:
        per_ds_n[it["ds"]] += 1
    print(f"      per set: {dict(per_ds_n)}", flush=True)

    rows, cov, bias = build_matrices(covered, P)
    print(f"[bias] first-slot win rate={bias.get('first_slot_win_rate'):.4f} "
          f"order disagreement={bias.get('order_disagreement_rate'):.4f} "
          f"mean|p_o0-(1-p_o1)|={bias.get('mean_abs_order_gap'):.4f}", flush=True)

    inc = {(it["ds"], it["idx"]): list(it["scores"]) for it in covered}
    base = G.sel_eff(inc, covered)
    print(f"\n=== ALL numbers below are on the SAME {len(covered)} covered items ===")
    print(f"  incumbent on covered subset: sel_eff={base['sel_eff']:.6f} "
          f"(full-pool value 0.775204)  contested n={base['contested']['n']}", flush=True)

    arms = {}

    def report(name, sc=None, picks=None, note=""):
        r = G.sel_eff(sc, covered, picks=picks)
        b = G.paired_bootstrap(r["got"], base["got"], rec=r["rec"], nboot=A.nboot, seed=0)
        bc = G.paired_bootstrap(r["got"], base["got"], nboot=A.nboot, seed=0,
                                mask=base["contested_mask"])
        o = {"name": name, "note": note, "sel_eff": r["sel_eff"], "acc": r["acc"],
             "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
             "contested_sel_eff": r["contested"]["sel_eff"],
             "d_sel_eff": b["d_sel_eff"], "d_sel_eff_ci": b["d_sel_eff_ci"],
             "d_contested": bc["d_sel_eff"], "d_contested_ci": bc["d_sel_eff_ci"],
             "guardrail_clean": G.guardrail_clean(r, base),
             "sig": bool(b["d_sel_eff_ci"][0] > 0 or b["d_sel_eff_ci"][1] < 0)}
        print(f"  {name:30s} sel_eff={r['sel_eff']:.6f} d={b['d_sel_eff']:+.6f} "
              f"[{b['d_sel_eff_ci'][0]:+.4f},{b['d_sel_eff_ci'][1]:+.4f}] "
              f"{'SIG' if o['sig'] else 'n.s.':4s} cont={r['contested']['sel_eff']:.6f} "
              f"guard={'clean' if o['guardrail_clean'] else 'DIRTY'} | "
              f"{o['per_ds']['slake_open']:.4f}/{o['per_ds']['vqa_rad_open']:.4f}/"
              f"{o['per_ds']['pathvqa_open']:.4f}", flush=True)
        return o

    sd = {}
    for arm in ("avg", "o0", "o1"):
        bd = {(x["it"]["ds"], x["it"]["idx"]): list(borda_scores(x["S"][arm])) for x in rows}
        cw = {(x["it"]["ds"], x["it"]["idx"]): list(copeland_wins(x["S"][arm])) for x in rows}
        sd[f"borda_{arm}"] = bd
        sd[f"copelandwins_{arm}"] = cw
        arms[f"borda_{arm}"] = report(f"borda ({arm})", bd)
        arms[f"copeland_pure_{arm}"] = report(f"copeland/round-robin ({arm})", cw)
    for arm in ("avg", "o0", "o1"):
        pk = np.array([knockout_pick_det(x["S"][arm], list(range(8)))[0] for x in rows])
        arms[f"knockout_det_{arm}"] = report(f"knockout_det ({arm})", picks=pk)

    effs = []
    for s in range(A.seeds):
        rng = np.random.default_rng(9000 + s)
        pk = np.array([knockout_pick_stoch(x["S"]["avg"], list(range(8)), rng)[0] for x in rows])
        effs.append(G.sel_eff(None, covered, picks=pk)["sel_eff"])
    arms["knockout_stoch"] = {"seeds": A.seeds, "sel_eff_mean": float(np.mean(effs)),
                              "sel_eff_sd": float(np.std(effs, ddof=1)),
                              "sel_eff_range": [float(np.min(effs)), float(np.max(effs))],
                              "sel_eff_per_seed": [float(x) for x in effs]}
    print(f"  {'knockout_stoch (10 seeds)':30s} mean={np.mean(effs):.6f} "
          f"sd={np.std(effs, ddof=1):.6f} range=[{np.min(effs):.6f},{np.max(effs):.6f}]", flush=True)

    print("\n=== fusion with the incumbent (rank_avg, parameter-free) ===")
    for k in ("borda_avg", "copelandwins_avg"):
        f = G.rank_fuse(inc, sd[k], items=covered, ranker=G.rank_avg)
        arms[f"fuse_inc_{k}"] = report(f"FUSE rank_avg inc+{k}", f)

    ctl = G.control_scores(covered)
    arms["self_consistency"] = report("self-consistency", ctl["self_consistency"])
    arms["incumbent"] = {"name": "incumbent on covered subset", "sel_eff": base["sel_eff"],
                         "acc": base["acc"],
                         "per_ds": {d: base["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
                         "contested_sel_eff": base["contested"]["sel_eff"]}
    arms["random_pick"] = G.random_pick(covered)

    # discordant-pair discrimination, engine-matched
    n = 0; okp = 0.0; oki = 0.0
    for x in rows:
        it = x["it"]
        yd = [max([it["sl"][s] for s in x["slots"][nn]]) for nn in x["na"]]
        sc = [float(np.mean([it["scores"][s] for s in x["slots"][nn]])) for nn in x["na"]]
        for a in range(x["k"]):
            for b in range(a + 1, x["k"]):
                if yd[a] == yd[b] or yd[a] < 0 or yd[b] < 0:
                    continue
                g, bb = (a, b) if yd[a] == 1 else (b, a)
                n += 1
                p = x["M"]["avg"][g, bb]
                okp += 1 if p > 0.5 else (0.5 if p == 0.5 else 0)
                oki += 1 if sc[g] > sc[bb] else (0.5 if sc[g] == sc[bb] else 0)
    disc = {"n_discordant_pairs": n, "real_pairwise_prefers_correct": okp / max(n, 1),
            "incumbent_pointwise_prefers_correct": oki / max(n, 1)}
    print(f"\n[discordant pairs n={n}] real pairwise (HF) "
          f"{disc['real_pairwise_prefers_correct']:.4f} vs incumbent pointwise "
          f"{disc['incumbent_pointwise_prefers_correct']:.4f}", flush=True)

    npd = float(np.mean([x["k"] * (x["k"] - 1) / 2 for x in rows]))
    out = {"generated": "2026-08-05", "engine": "HuggingFace + PeftModel (FULL adapter, "
           "192 visual lora_A modules present) -- matches the stack that produced the bar",
           "script": "src/training_methods/realpairwise_hf_analyze.py",
           "null_test": nt, "verdict_files": meta,
           "covered_items": len(covered), "covered_per_ds": dict(per_ds_n),
           "pathvqa_subsample": {"n": 500, "seed": 0, "pre_registered": True,
                                 "rule": "np.random.default_rng(0).choice(1500,500,replace=False)"},
           "coverage": cov, "position_bias": bias,
           "incumbent_on_covered_subset": base["sel_eff"],
           "discordant_pair_discrimination": disc, "arms": arms,
           "cost": {"roundrobin_forwards_per_question_both_orders": 2 * npd,
                    "mean_unordered_pairs_per_question": npd}}
    json.dump(out, open(A.out, "w"), indent=1)
    print(f"\n-> {A.out}", flush=True)

    with open(A.teacher, "w") as fh:
        for x in rows:
            it = x["it"]
            fh.write(json.dumps({
                "ds": it["ds"], "idx": it["idx"], "na": x["na"],
                "slots": [x["slots"][nn] for nn in x["na"]],
                "y": [max([it["sl"][s] for s in x["slots"][nn]]) for nn in x["na"]],
                "inc_score": [float(np.mean([it["scores"][s] for s in x["slots"][nn]])) for nn in x["na"]],
                "P_avg": [[round(float(v), 6) for v in r] for r in x["M"]["avg"]],
                "P_o0": [[round(float(v), 6) for v in r] for r in x["M"]["o0"]],
                "P_o1": [[round(float(v), 6) for v in r] for r in x["M"]["o1"]]}) + "\n")
    print(f"-> TEACHER (HF, full adapter): {A.teacher}", flush=True)


if __name__ == "__main__":
    main()
