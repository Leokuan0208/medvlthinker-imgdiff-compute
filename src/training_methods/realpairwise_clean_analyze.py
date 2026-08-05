#!/usr/bin/env python3
"""realpairwise_clean_analyze.py -- score the CLEAN real-pairwise verdicts against the
CURRENT bar (incumbent sel_eff 0.775204 on the 2345-question disjoint pool).

Reads ckpts/pairwise_clean/ordered_*.jsonl (one row per ORDERED forward pass, written by
realpairwise_clean_gpu.py), debiases position, expands the distinct-candidate preference
matrix onto the 8 slots (identical normalized strings -> p = 0.5, verbatim old rule), and
evaluates every aggregator through genframe_data's frozen metric.

Aggregators
  borda            score_i = sum_j p_i>j                       pointwise-free, continuous
  copeland_pure    wins (p>0.5, ties 0.5), tie-break Borda      pointwise-free
  copeland_det     VERBATIM active_comparison_verifier.copeland_pick_det -- ties broken by
                   the POINTWISE score, i.e. it inherits the incumbent. Reported, flagged.
  knockout_det     index-seeded single-elimination, winner = argmax p, tie -> lower index
  knockout_stoch   the original stochastic bracket, >= 10 seeds, mean/sd/range
Order arms: order-0 only, order-1 only, and the averaged (debiased) matrix -- reported
separately so position bias is visible rather than assumed away.

  python3 src/training_methods/realpairwise_clean_analyze.py
"""
import os, sys, json, glob, math, argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402
from src.training_methods.realpairwise_clean_gpu import distinct_cands  # noqa: E402

VDIR = os.path.join(ROOT, "ckpts/pairwise_clean")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")


# --------------------------------------------------------------------------- load
def load_pointwise_control(tag="disjoint"):
    """{(ds, idx, na) -> pyes} from the ENGINE-MATCHED vLLM pointwise control, or None."""
    files = sorted(glob.glob(os.path.join(VDIR, f"pointwise_*_{tag}.jsonl")))
    if not files:
        return None, {"files": []}
    S, n = {}, 0
    for f in files:
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            S[(r["ds"], int(r["idx"]) if str(r["idx"]).lstrip("-").isdigit() else r["idx"], r["na"])] = r["pyes"]
            n += 1
    return S, {"files": [os.path.basename(f) for f in files], "n_rows": n}


def load_ordered(tag="disjoint"):
    """(ds, idx) -> {(ai, bi): {order: p_first}}  + error/row counts."""
    P = defaultdict(dict)
    n_rows = n_err = 0
    files = sorted(glob.glob(os.path.join(VDIR, f"ordered_*_{tag}*.jsonl")))
    for f in files:
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            n_rows += 1
            if r.get("p_first") is None:
                n_err += 1
                continue
            key = (r["ds"], str(r["idx"]))
            P[key].setdefault((int(r["ai"]), int(r["bi"])), {})[int(r["order"])] = float(r["p_first"])
    return P, {"files": [os.path.basename(f) for f in files], "n_rows": n_rows, "n_error_rows": n_err}


# --------------------------------------------------------------- matrices per item
def build_matrices(items, P):
    """For each item: distinct-candidate preference matrices in three order arms, plus the
    8x8 slot expansion of each. Returns per-item dict, coverage stats, position-bias stats."""
    out = []
    cov = {"items": 0, "items_complete": 0, "pairs_expected": 0, "pairs_present": 0,
           "pairs_both_orders": 0}
    bias = {"n": 0, "first_slot_wins": 0.0, "disagree": 0, "mean_abs_gap": 0.0,
            "mean_p_first": 0.0}
    for it in items:
        key = (it["ds"], str(it["idx"]))
        na, slots, text = distinct_cands(it["preds"])
        k = len(na)
        pm = P.get(key, {})
        M = {arm: np.full((k, k), 0.5) for arm in ("avg", "o0", "o1")}
        exp = k * (k - 1) // 2
        got = both = 0
        for a in range(k):
            for b in range(a + 1, k):
                d = pm.get((a, b))
                if not d:
                    continue
                got += 1
                p0 = d.get(0)                  # P(a preferred) when a listed first
                p1 = d.get(1)                  # P(b preferred) when b listed first
                pa_o1 = None if p1 is None else 1.0 - p1
                if p0 is not None and pa_o1 is not None:
                    both += 1
                    pa = 0.5 * (p0 + pa_o1)
                    bias["n"] += 1
                    bias["first_slot_wins"] += (1.0 if p0 > 0.5 else 0.0) + (1.0 if p1 > 0.5 else 0.0)
                    bias["mean_p_first"] += p0 + p1
                    bias["disagree"] += int((p0 > 0.5) != (pa_o1 > 0.5))
                    bias["mean_abs_gap"] += abs(p0 - pa_o1)
                else:
                    pa = p0 if p0 is not None else pa_o1
                for arm, v in (("avg", pa), ("o0", p0 if p0 is not None else pa),
                               ("o1", pa_o1 if pa_o1 is not None else pa)):
                    M[arm][a, b] = v
                    M[arm][b, a] = 1.0 - v
        cov["items"] += 1
        cov["items_complete"] += int(got == exp)
        cov["pairs_expected"] += exp
        cov["pairs_present"] += got
        cov["pairs_both_orders"] += both
        # 8x8 slot expansion: identical normalized strings -> 0.5 (verbatim old rule)
        d_of_slot = np.array([na.index(G.norm(a)) for a in it["preds"]], dtype=int)
        S = {arm: M[arm][np.ix_(d_of_slot, d_of_slot)] for arm in M}
        for arm in S:
            np.fill_diagonal(S[arm], 0.5)
        out.append({"it": it, "na": na, "slots": slots, "text": text, "k": k,
                    "d_of_slot": d_of_slot, "M": M, "S": S})
    if bias["n"]:
        n2 = 2.0 * bias["n"]
        bias["first_slot_win_rate"] = bias["first_slot_wins"] / n2
        bias["mean_p_first"] = bias["mean_p_first"] / n2
        bias["order_disagreement_rate"] = bias["disagree"] / bias["n"]
        bias["mean_abs_order_gap"] = bias["mean_abs_gap"] / bias["n"]
    return out, cov, bias


# ------------------------------------------------------------------- aggregators
def borda_scores(S):
    """score_i = sum_{j!=i} p_i>j over the 8 slots."""
    n = S.shape[0]
    return (S.sum(1) - np.diag(S))


def copeland_wins(S):
    n = S.shape[0]
    w = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            p = S[i, j]
            if p > 0.5:
                w[i] += 1
            elif p < 0.5:
                w[j] += 1
            else:
                w[i] += 0.5; w[j] += 0.5
    return w


def logit(s):
    s = min(max(float(s), 1e-6), 1 - 1e-6)
    return math.log(s / (1 - s))


def copeland_pick_det(S, pointwise):
    """VERBATIM active_comparison_verifier.copeland_pick_det: ties broken by the POINTWISE
    score then by index. Inherits the incumbent -- flagged wherever reported."""
    w = copeland_wins(S)
    z = [logit(s) for s in pointwise]
    return max(range(len(w)), key=lambda k: (w[k], z[k], -k))


def knockout_pick_det(S, order):
    """Single elimination over `order`; winner = i if p_i>j > 0.5 else j (tie -> first)."""
    alive = list(order)
    ncmp = 0
    while len(alive) > 1:
        nxt = []
        for k in range(0, len(alive) - 1, 2):
            i, j = alive[k], alive[k + 1]
            ncmp += 1
            nxt.append(i if S[i, j] >= 0.5 else j)
        if len(alive) % 2 == 1:
            nxt.append(alive[-1])
        alive = nxt
    return alive[0], ncmp


def knockout_pick_stoch(S, order, rng):
    """VERBATIM shape of active_comparison_verifier.knockout_pick: stochastic BT outcome."""
    alive = list(order)
    ncmp = 0
    while len(alive) > 1:
        nxt = []
        for k in range(0, len(alive) - 1, 2):
            i, j = alive[k], alive[k + 1]
            ncmp += 1
            nxt.append(i if rng.random() < S[i, j] else j)
        if len(alive) % 2 == 1:
            nxt.append(alive[-1])
        alive = nxt
    return alive[0], ncmp


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="disjoint")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ART, "verifarch_realpairwise_clean_2026-08-04.json"))
    ap.add_argument("--teacher", default=os.path.join(ART, "realpairwise_teacher_pmatrix_2026-08-05.jsonl"))
    A = ap.parse_args()

    # ---------------- protocol rule 1: null test BEFORE anything new
    nt = G.null_test()
    print(f"[null] pass={nt['pass']}  max_abs_dev={nt['max_abs_deviation']:.3e}", flush=True)
    if not nt["pass"]:
        sys.exit("NULL TEST FAILED -- stopping, per protocol rule 1.")

    items = G.load_items()
    inc = G.incumbent_scores()
    base = G.sel_eff(inc, items)
    P, meta = load_ordered(A.tag)
    rows, cov, bias = build_matrices(items, P)
    print(f"[cov] items={cov['items']} complete={cov['items_complete']} "
          f"pairs {cov['pairs_present']}/{cov['pairs_expected']} both-orders={cov['pairs_both_orders']}",
          flush=True)
    print(f"[bias] first-slot win rate={bias.get('first_slot_win_rate'):.4f} "
          f"order disagreement={bias.get('order_disagreement_rate'):.4f} "
          f"mean |p_o0-(1-p_o1)|={bias.get('mean_abs_order_gap'):.4f}", flush=True)

    def report(name, picks_or_scores, is_picks=False, note=""):
        r = (G.sel_eff(None, items, picks=picks_or_scores) if is_picks
             else G.sel_eff(picks_or_scores, items))
        b = G.paired_bootstrap(r["got"], base["got"], rec=r["rec"], nboot=A.nboot, seed=0)
        bc = G.paired_bootstrap(r["got"], base["got"], nboot=A.nboot, seed=0,
                                mask=base["contested_mask"])
        out = {"name": name, "note": note,
               "sel_eff": r["sel_eff"], "acc": r["acc"],
               "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
               "contested_sel_eff": r["contested"]["sel_eff"],
               "contested_n": r["contested"]["n"],
               "d_sel_eff": b["d_sel_eff"], "d_sel_eff_ci": b["d_sel_eff_ci"],
               "d_acc": b["d_acc"], "d_acc_ci": b["d_acc_ci"],
               "d_contested": bc["d_sel_eff"], "d_contested_ci": bc["d_sel_eff_ci"],
               "guardrail_clean": G.guardrail_clean(r, base),
               "sig": bool(b["d_sel_eff_ci"][0] > 0 or b["d_sel_eff_ci"][1] < 0)}
        sg = "SIG" if out["sig"] else "n.s."
        print(f"  {name:34s} sel_eff={r['sel_eff']:.6f}  d={b['d_sel_eff']:+.6f} "
              f"[{b['d_sel_eff_ci'][0]:+.4f},{b['d_sel_eff_ci'][1]:+.4f}] {sg:4s} "
              f"cont={r['contested']['sel_eff']:.6f}  guard={'clean' if out['guardrail_clean'] else 'DIRTY'} "
              f"| {out['per_ds']['slake_open']:.4f}/{out['per_ds']['vqa_rad_open']:.4f}/"
              f"{out['per_ds']['pathvqa_open']:.4f}", flush=True)
        return out, r

    arms = {}
    print("\n=== aggregators (bar: incumbent sel_eff 0.775204) ===", flush=True)

    score_dicts = {}
    for arm in ("avg", "o0", "o1"):
        bd, cw = {}, {}
        for x in rows:
            key = (x["it"]["ds"], x["it"]["idx"])
            bd[key] = list(borda_scores(x["S"][arm]))
            cw[key] = list(copeland_wins(x["S"][arm]))
        score_dicts[f"borda_{arm}"] = bd
        score_dicts[f"copelandwins_{arm}"] = cw
        arms[f"borda_{arm}"] = report(f"borda ({arm})", bd)[0]
        arms[f"copeland_pure_{arm}"] = report(f"copeland_pure ({arm})", cw,
                                              note="wins; argmax tie-break by slot index")[0]

    # copeland_det -- verbatim old tie-break by the pointwise score (INHERITS INCUMBENT)
    for arm in ("avg",):
        picks = np.array([copeland_pick_det(x["S"][arm], x["it"]["scores"]) for x in rows], dtype=int)
        arms[f"copeland_det_{arm}"] = report(f"copeland_det ({arm}) [pw tiebreak]", picks, is_picks=True,
                                             note="VERBATIM old aggregator: Copeland wins, ties broken by "
                                                  "the INCUMBENT pointwise score -- not pointwise-free")[0]

    # knockout, deterministic, over slots in index order
    for arm in ("avg", "o0", "o1"):
        picks, costs = [], []
        for x in rows:
            p, c = knockout_pick_det(x["S"][arm], list(range(8)))
            picks.append(p); costs.append(c)
        arms[f"knockout_det_{arm}"] = report(f"knockout_det slots ({arm})", np.array(picks), is_picks=True,
                                             note=f"cost {np.mean(costs):.2f} comparisons/question over 8 slots")[0]
        arms[f"knockout_det_{arm}"]["cost_pairs_per_q"] = float(np.mean(costs))

    # knockout over DISTINCT candidates (the cheap, sensible version)
    picks, costs = [], []
    for x in rows:
        p, c = knockout_pick_det(x["M"]["avg"], list(range(x["k"])))
        picks.append(x["slots"][x["na"][p]][0]); costs.append(c)
    arms["knockout_det_distinct"] = report("knockout_det distinct (avg)", np.array(picks), is_picks=True,
                                           note=f"cost {np.mean(costs):.2f} comparisons/question")[0]
    arms["knockout_det_distinct"]["cost_pairs_per_q"] = float(np.mean(costs))

    # knockout, stochastic (the ORIGINAL), >= 10 seeds
    effs, accs, gots = [], [], []
    for s in range(A.seeds):
        rng = np.random.default_rng(9000 + s)
        picks = np.array([knockout_pick_stoch(x["S"]["avg"], list(range(8)), rng)[0] for x in rows], dtype=int)
        r = G.sel_eff(None, items, picks=picks)
        effs.append(r["sel_eff"]); accs.append(r["acc"]); gots.append(r["got"])
    got_ens = (np.mean(gots, axis=0) >= 0.5).astype(int)   # majority-vote ensemble over seeds
    b = G.paired_bootstrap(got_ens, base["got"], rec=base["rec"], nboot=A.nboot, seed=0)
    arms["knockout_stoch"] = {"name": "knockout_stoch (original)", "seeds": A.seeds,
                              "sel_eff_mean": float(np.mean(effs)), "sel_eff_sd": float(np.std(effs, ddof=1)),
                              "sel_eff_range": [float(np.min(effs)), float(np.max(effs))],
                              "sel_eff_per_seed": [float(x) for x in effs],
                              "acc_mean": float(np.mean(accs)),
                              "ensemble_majority_d_sel_eff": b["d_sel_eff"],
                              "ensemble_majority_d_sel_eff_ci": b["d_sel_eff_ci"],
                              "note": "VERBATIM old knockout: winner sampled with prob p (Bernoulli), not argmax"}
    print(f"  {'knockout_stoch (10 seeds)':34s} mean={np.mean(effs):.6f} sd={np.std(effs, ddof=1):.6f} "
          f"range=[{np.min(effs):.6f},{np.max(effs):.6f}]", flush=True)

    # ---------------- fusion with the incumbent
    print("\n=== fusion with the incumbent (rank_avg, parameter-free) ===", flush=True)
    for k in ("borda_avg", "copelandwins_avg"):
        f = G.rank_fuse(inc, score_dicts[k], items=items, ranker=G.rank_avg)
        arms[f"fuse_inc_{k}"] = report(f"FUSE rank_avg inc+{k}", f)[0]

    # ---------------- controls
    print("\n=== controls ===", flush=True)
    ctl = G.control_scores(items)
    arms["incumbent"] = report("incumbent (the bar)", ctl["incumbent"])[0]
    arms["self_consistency"] = report("self-consistency", ctl["self_consistency"])[0]
    rp = G.random_pick(items)
    print(f"  {'random-pick (closed form)':34s} sel_eff={rp['sel_eff']:.6f}", flush=True)

    # ---------------- THE ENGINE-MATCHED POINTWISE CONTROL (decisive for attribution)
    pc, pcmeta = load_pointwise_control(A.tag)
    if pc:
        pcd = {}
        for it in items:
            na, slots, text = distinct_cands(it["preds"])
            pcd[(it["ds"], it["idx"])] = [pc.get((it["ds"], it["idx"], G.norm(a)), np.nan)
                                          for a in it["preds"]]
        miss = sum(1 for v in pcd.values() for x in v if isinstance(x, float) and np.isnan(x))
        print(f"\n=== engine-matched pointwise control (vLLM, LM-only LoRA) "
              f"[{pcmeta['n_rows']} rows, {miss} missing slots] ===", flush=True)
        arms["pointwise_control_vllm"] = report("POINTWISE control (vLLM engine)", pcd)[0]
        arms["pointwise_control_vllm"]["meta"] = pcmeta
        # pairwise vs the ENGINE-MATCHED bar, not just the HF bar
        pcr = G.sel_eff(pcd, items)
        for k in ("borda_avg", "copelandwins_avg"):
            r = G.sel_eff(score_dicts[k], items)
            b = G.paired_bootstrap(r["got"], pcr["got"], rec=r["rec"], nboot=A.nboot, seed=0)
            arms[f"{k}_vs_engine_matched_pointwise"] = {
                "name": f"{k} vs vLLM pointwise control", "d_sel_eff": b["d_sel_eff"],
                "d_sel_eff_ci": b["d_sel_eff_ci"],
                "sig": bool(b["d_sel_eff_ci"][0] > 0 or b["d_sel_eff_ci"][1] < 0)}
            print(f"  {k} vs ENGINE-MATCHED pointwise: d={b['d_sel_eff']:+.6f} "
                  f"[{b['d_sel_eff_ci'][0]:+.4f},{b['d_sel_eff_ci'][1]:+.4f}]", flush=True)
    else:
        print("\n!! no engine-matched pointwise control found -- attribution incomplete", flush=True)

    # ---------------- discordant-pair discrimination (the mechanism read)
    pw_ok = pw_n = inc_ok = 0
    pd_list = []
    for x in rows:
        it = x["it"]
        yd = [max([it["sl"][s] for s in x["slots"][n]]) for n in x["na"]]
        sd = [float(np.mean([it["scores"][s] for s in x["slots"][n]])) for n in x["na"]]
        for a in range(x["k"]):
            for b_ in range(a + 1, x["k"]):
                if yd[a] == yd[b_] or yd[a] < 0 or yd[b_] < 0:
                    continue
                good, bad = (a, b_) if yd[a] == 1 else (b_, a)
                p = x["M"]["avg"][good, bad]
                pd_list.append(float(p))
                pw_n += 1
                pw_ok += 1 if p > 0.5 else (0.5 if p == 0.5 else 0)
                inc_ok += 1 if sd[good] > sd[bad] else (0.5 if sd[good] == sd[bad] else 0)
    disc = {"n_discordant_pairs": pw_n,
            "real_pairwise_prefers_correct": pw_ok / max(pw_n, 1),
            "incumbent_pointwise_prefers_correct": inc_ok / max(pw_n, 1),
            "mean_p_correct_over_incorrect": float(np.mean(pd_list)) if pd_list else None,
            "sd_p": float(np.std(pd_list)) if pd_list else None}
    print(f"\n[discordant pairs n={pw_n}] real pairwise {disc['real_pairwise_prefers_correct']:.4f} "
          f"vs incumbent pointwise {disc['incumbent_pointwise_prefers_correct']:.4f}", flush=True)

    # ---------------- NEAR-TIE stratum (the stratum the OLD work claimed +0.050 on)
    inc_slots = np.array([it["scores"] for it in items], dtype=float)
    top2 = np.sort(inc_slots, axis=1)[:, -2:]
    neartie = (top2[:, 1] - top2[:, 0]) < 0.1
    nt_mask = neartie & (base["rec"] == 1)
    print(f"\n=== near-tie stratum (incumbent top1-top2 < 0.1), n={int(nt_mask.sum())} recoverable ===",
          flush=True)
    nt_arms = {}
    for k in ("borda_avg", "copelandwins_avg"):
        r = G.sel_eff(score_dicts[k], items)
        b = G.paired_bootstrap(r["got"], base["got"], nboot=A.nboot, seed=0, mask=nt_mask)
        nt_arms[k] = {"sel_eff": float(r["got"][nt_mask].mean()),
                      "d_vs_incumbent": b["d_sel_eff"], "ci": b["d_sel_eff_ci"]}
        print(f"  {k:22s} near-tie sel_eff={r['got'][nt_mask].mean():.6f} "
              f"d={b['d_sel_eff']:+.6f} [{b['d_sel_eff_ci'][0]:+.4f},{b['d_sel_eff_ci'][1]:+.4f}]", flush=True)
    nt_arms["incumbent"] = {"sel_eff": float(base["got"][nt_mask].mean())}
    nt_arms["n"] = int(nt_mask.sum())
    print(f"  {'incumbent':22s} near-tie sel_eff={base['got'][nt_mask].mean():.6f}", flush=True)

    # ---------------- cost
    npairs_distinct = float(np.mean([x["k"] * (x["k"] - 1) / 2 for x in rows]))
    ndist = float(np.mean([x["k"] for x in rows]))
    cost = {"generations_per_question": 8,
            "incumbent_pointwise_extra_forwards_per_question": 8.0,
            "incumbent_pointwise_extra_forwards_dedup": ndist,
            "mean_distinct_candidates": ndist,
            "roundrobin_unordered_pairs_per_question": npairs_distinct,
            "roundrobin_forwards_per_question_both_orders": 2 * npairs_distinct,
            "knockout_distinct_forwards_per_question_both_orders":
                2 * float(np.mean([x["k"] - 1 for x in rows])),
            "total_ordered_forwards_run": meta["n_rows"] - meta["n_error_rows"]}
    print(f"\n[cost] round-robin {2*npairs_distinct:.2f} forwards/q (both orders) vs pointwise "
          f"{ndist:.2f} (dedup) / 8 (slots)", flush=True)

    out = {"generated": "2026-08-05", "script": "src/training_methods/realpairwise_clean_analyze.py",
           "what": "CLEAN GPU replication of the REAL pairwise verifier on the current 2345-question "
                   "disjoint pool, with the clean disjoint-trained adapter.",
           "null_test": nt,
           "verdict_files": meta, "coverage": cov, "position_bias": bias,
           "discordant_pair_discrimination": disc,
           "near_tie_stratum": nt_arms,
           "arms": arms, "cost": cost,
           "bar": {"incumbent_sel_eff": base["sel_eff"], "incumbent_per_ds":
                   {d: base["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
                   "incumbent_contested": base["contested"]["sel_eff"],
                   "random_pick_sel_eff": rp["sel_eff"], "oracle_acc": base["oracle"],
                   "greedy_acc": base["greedy"]},
           "NOT_COMPARABLE_TO": {
               "old_numbers": {"pointwise": 0.783, "knockout": 0.849, "roundrobin": 0.859,
                               "n": 578, "adapter": "ckpts/train/lora_verifier_pooled4 (CONTAMINATED)",
                               "pool": "ckpts/mcq_gen_verify/lingshu7b/*_sc8.jsonl",
                               "max_pixels": 250880},
               "why": "different pool, different n, different (contaminated) adapter, different image "
                      "resolution. The old 0.783/0.849/0.859 triple must never be quoted beside the "
                      "numbers in this file as if they were the same measurement."}}
    json.dump(out, open(A.out, "w"), indent=1)
    print(f"\n-> {A.out}", flush=True)

    # ---------------- teacher file for the cached-feature siblings
    with open(A.teacher, "w") as fh:
        for x in rows:
            it = x["it"]
            sl_d = [max([it["sl"][s] for s in x["slots"][n]]) for n in x["na"]]
            inc_d = [float(np.mean([it["scores"][s] for s in x["slots"][n]])) for n in x["na"]]
            fh.write(json.dumps({
                "ds": it["ds"], "idx": it["idx"], "na": x["na"],
                "slots": [x["slots"][n] for n in x["na"]],
                "y": sl_d, "inc_score": inc_d,
                "P_avg": [[round(float(v), 6) for v in r] for r in x["M"]["avg"]],
                "P_o0": [[round(float(v), 6) for v in r] for r in x["M"]["o0"]],
                "P_o1": [[round(float(v), 6) for v in r] for r in x["M"]["o1"]],
            }) + "\n")
    print(f"-> TEACHER (per-item pairwise P matrices, keyed (ds,idx,na) to join "
          f"genframe_data rows): {A.teacher}", flush=True)


if __name__ == "__main__":
    main()
