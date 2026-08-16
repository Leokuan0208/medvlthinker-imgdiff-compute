#!/usr/bin/env python3
"""selfcons_suite.py -- ATTACK 2: training-free SELF-CONSISTENCY swept across all eight macro cells.

Pre-registration: artifacts/self_consistency_suite_2026-08-17_preregistration.json (written first).

CPU ONLY.  No GPU job, no new generation.  Every pool already exists:
  MCQ cells     ckpts/closed_as_open_mcq/     (T=0.4 n=8 + matched in-session greedy, fullres)
  closed cells  ckpts/closed_as_open/         (T=0.4 n=8 + matched in-session greedy, cap320)
  open cells    ckpts/openvqa/decoding_sweep/ (T=0.4 n=8 x 3 seeds + matched in-session T=0 x 3 seeds)

Every delta subtracts a greedy control generated in the SAME session and engine as the sampled pool
it is compared against -- never a stored published number -- which is what the project's +-0.008
regeneration caveat requires.

  python3 src/cascade_methods/selfcons_suite.py --stage mcq
  python3 src/cascade_methods/selfcons_suite.py --stage closed
  python3 src/cascade_methods/selfcons_suite.py --stage open
  python3 src/cascade_methods/selfcons_suite.py --stage head      # needs feats_hidden (4.4 GB, CPU)
  python3 src/cascade_methods/selfcons_suite.py --stage finalize
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import selfcons_lib as S                                                   # noqa: E402
import closed_as_open_lib as L                                             # noqa: E402


# =============================================================================================
# helpers shared by the cell stages
# =============================================================================================
def per_item_arms(pools, greedy_labels, ns=S.NS):
    """pools: list over items of (classes, labels).  Returns {arm -> per-item np array}."""
    n = len(pools)
    out = {f"vote@{N}": np.zeros(n) for N in ns}
    out.update({f"vote@{N}_orderfree": np.zeros(n) for N in ns})
    out.update({f"vote@{N}_prefix": np.zeros(n) for N in ns})
    out["oracle@8"] = np.zeros(n)
    out["greedy"] = np.asarray(greedy_labels, float)
    for i, (cls, lab) in enumerate(pools):
        for N in ns:
            out[f"vote@{N}"][i] = S.vote_exact(cls, lab, N)
            out[f"vote@{N}_orderfree"][i] = S.vote_exact_orderfree(cls, lab, N)
            out[f"vote@{N}_prefix"][i] = S.vote_prefix(cls, lab, N)
        out["oracle@8"][i] = S.oracle_exact(lab, 8)
    out["random_slot"] = out["vote@1"]          # C(8,1) == a single random draw, by construction
    return out


def cell_block(arms, extra=None):
    """Deltas of every vote@N against the matched greedy control, plus descriptives."""
    g = arms["greedy"]
    blk = {"n": int(len(g)),
           "greedy_matched_control": S.R(g.mean()),
           "random_slot_floor_N1": S.R(arms["vote@1"].mean()),
           "oracle@8": S.R(arms["oracle@8"].mean())}
    for N in [2, 4, 8]:
        blk[f"vote@{N}"] = S.R(arms[f"vote@{N}"].mean())
        blk[f"vote@{N}_orderfree"] = S.R(arms[f"vote@{N}_orderfree"].mean())
        blk[f"vote@{N}_prefix"] = S.R(arms[f"vote@{N}_prefix"].mean())
    blk["ROBUSTNESS_slot_order"] = {
        "why": "an n=8 vLLM pool with a fixed seed is NOT slot-exchangeable on 6 of 14 pools "
               "(_selfcons_parts/slot_exchangeability.json). Three tie-break / subset conventions are "
               "therefore reported, and the spread between them is the size of the position artefact.",
        **{f"N={N}": {"index_tiebreak": S.R(arms[f"vote@{N}"].mean()),
                      "orderfree_tiebreak": S.R(arms[f"vote@{N}_orderfree"].mean()),
                      "first_N_slots_prefix": S.R(arms[f"vote@{N}_prefix"].mean()),
                      "max_minus_min": S.R(max(arms[f"vote@{N}"].mean(),
                                               arms[f"vote@{N}_orderfree"].mean(),
                                               arms[f"vote@{N}_prefix"].mean())
                                           - min(arms[f"vote@{N}"].mean(),
                                                 arms[f"vote@{N}_orderfree"].mean(),
                                                 arms[f"vote@{N}_prefix"].mean()))}
           for N in [2, 4, 8]}}
    blk["vs_matched_greedy"] = {f"N={N}": S.paired_boot(arms[f"vote@{N}"], g) for N in [2, 4, 8]}
    blk["vs_matched_greedy_ORDERFREE"] = {f"N={N}": S.paired_boot(arms[f"vote@{N}_orderfree"], g)
                                          for N in [2, 4, 8]}
    blk["vs_matched_greedy_PREFIX"] = {f"N={N}": S.paired_boot(arms[f"vote@{N}_prefix"], g)
                                       for N in [2, 4, 8]}
    blk["vs_random_slot_floor"] = {f"N={N}": S.paired_boot(arms[f"vote@{N}"], arms["vote@1"])
                                   for N in [2, 4, 8]}
    blk["greedy_vs_random_slot_floor"] = S.paired_boot(g, arms["vote@1"])
    blk["N5_exchangeability_vote@2_minus_random_slot"] = S.R(
        arms["vote@2"].mean() - arms["vote@1"].mean(), 8)
    if extra:
        blk.update(extra)
    return blk


# =============================================================================================
# STAGE mcq -- PMC_VQA + MedXpertQA-MM
# =============================================================================================
MCQ_CELLS = ["PMC_VQA", "MedXpertQA-MM"]
MCQ_CK = os.path.join(ROOT, "ckpts/closed_as_open_mcq")


def mcq_load(cell, arm):
    p = os.path.join(MCQ_CK, f"gen_{cell}_{arm}.jsonl")
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[int(r["i"])] = r
    return out


def letter_of(response, k):
    """First-branch letter rule, byte-identical to closed_as_open_mcq_sc.letter_of."""
    alphas = [chr(ord("a") + i) for i in range(k)]
    r = L.parse_response(response).strip().lower().replace("\n", "")
    head = r.split(".")[0].split(":")[-1].strip()
    if head in alphas:
        return head
    m = re.match(r"^\(?([a-z])\)?\b", r)
    if m and m.group(1) in alphas:
        return m.group(1)
    return None


def stage_mcq(A):
    import mcq_tta as M
    items = M.build_items()
    choices = {c: {r["i"]: [str(x) for x in r["choices"]] for r in items[c]} for c in MCQ_CELLS}

    #: NULL TEST N2 -- the grader copy must reproduce MedEvalKit's stored `correct` field row by row
    g2 = L.mcq_grader_null_test({c: [[str(x) for x in r["choices"]] for r in items[c]]
                                 for c in MCQ_CELLS})
    print("N2 MCQ grader null test:", json.dumps(g2), flush=True)

    out = {"null_test_N2_mcq_grader": g2, "cells": {}}
    for cell in MCQ_CELLS:
        t0 = time.time()
        g = mcq_load(cell, "mcq_g")
        s = mcq_load(cell, "mcq_s8")
        idxs = sorted(set(g) & set(s))
        ch = choices[cell]

        def harness(i, gold, resp):
            return int(bool(L.judge_multi_choice([c.lower() for c in ch[i]], str(gold), resp)))

        pools_j, pools_e, gj, ge = [], [], [], []
        golds, nd, unpar_g, unpar_s, ndistinct_hist = [], [], 0, 0, Counter()
        for i in idxs:
            gold = str(g[i]["gold"]).strip().lower()
            k = int(g[i]["k"])
            golds.append(gold)
            lg = letter_of(g[i]["preds"][0], k)
            unpar_g += int(lg is None)
            ge.append(int(lg == gold))
            gj.append(harness(i, gold, g[i]["preds"][0]))
            cls, lab_e, lab_j = [], [], []
            for p in s[i]["preds"]:
                x = letter_of(p, k)
                if x is None:
                    unpar_s += 1
                    cls.append(("RAW", L.norm_text(p)))
                else:
                    cls.append(x)
                lab_e.append(int(x == gold))
                lab_j.append(harness(i, gold, p))
            pools_e.append((cls, lab_e))
            pools_j.append((cls, lab_j))
            nd.append(len(set(cls)))
            ndistinct_hist[len(set(cls))] += 1

        arms_j = per_item_arms(pools_j, gj)
        arms_e = per_item_arms(pools_e, ge)

        # ---- RANDOM-GOLD LUCK FLOOR: gold re-drawn i.i.d. from this cell's own gold marginal
        gmarg = Counter(golds)
        ntot = len(golds)
        gmarg = {a: v / ntot for a, v in gmarg.items()}
        luck = {"gold_letter_marginal": {a: S.R(v, 4) for a, v in sorted(gmarg.items())}}
        # greedy under random gold
        pred_marg = Counter()
        for i in idxs:
            x = letter_of(g[i]["preds"][0], int(g[i]["k"]))
            pred_marg[x if x is not None else "UNPARSED"] += 1
        luck["greedy_pred_letter_marginal"] = {str(a): S.R(v / ntot, 4)
                                               for a, v in sorted(pred_marg.items(), key=str)}
        luck["greedy_random_gold"] = S.R(sum(gmarg.get(a, 0.0) * v / ntot
                                             for a, v in pred_marg.items()))
        for N in [1, 2, 4, 8]:
            acc_v, acc_o = 0.0, 0.0
            for (cls, _lab) in pools_e:
                dv = S.vote_class_dist(cls, N)
                acc_v += sum(gmarg.get(c, 0.0) * p for c, p in dv.items())
                do = S.distinct_class_prob(cls, N)
                acc_o += sum(gmarg.get(c, 0.0) * p for c, p in do.items())
            luck[f"vote@{N}_random_gold"] = S.R(acc_v / len(pools_e))
            luck[f"oracle@{N}_random_gold"] = S.R(acc_o / len(pools_e))
        luck["how_to_read"] = ("gold letters re-drawn i.i.d. from this cell's OWN empirical gold "
                              "marginal, independent of the item. An arm that does not clear its own "
                              "random-gold floor has produced no evidence of skill on this cell. "
                              "PMC-VQA gold is 73.6% B+C [pmcvqa_answer_bias_audit_2026-08-11.json], "
                              "which is exactly why this control is mandatory here.")
        luck["oracle@8_vs_its_own_random_gold_floor"] = S.R(
            arms_e["oracle@8"].mean() - luck["oracle@8_random_gold"])
        luck["vote@8_vs_its_own_random_gold_floor"] = S.R(
            arms_e["vote@8"].mean() - luck["vote@8_random_gold"])

        blk_j = cell_block(arms_j)
        blk_e = cell_block(arms_e)
        out["cells"][cell] = {
            "n": len(idxs),
            "pool": "T=0.4, n=8, fullres, deployed MedEvalKit MCQ prompt (in-session)",
            "greedy_control": "T=0, same engine/session/prompt/resolution (mcq_g)",
            "published_always_7b": S.PUBLISHED_ALWAYS_7B[cell][0],
            "published_grader": S.PUBLISHED_ALWAYS_7B[cell][1],
            "in_session_greedy_minus_published": S.R(blk_j["greedy_matched_control"] -
                                                     S.PUBLISHED_ALWAYS_7B[cell][0]),
            "judge_currency_harness_judge_multi_choice": blk_j,
            "em_currency_letter_em": blk_e,
            "RANDOM_GOLD_LUCK_FLOOR_letter_em": luck,
            "diagnostics": {
                "unparsed_letter_rate_greedy": S.R(unpar_g / len(idxs), 5),
                "unparsed_letter_rate_sampled": S.R(unpar_s / (8 * len(idxs)), 5),
                "mean_distinct_letters_in_pool": S.R(float(np.mean(nd)), 4),
                "distinct_letter_hist": dict(sorted(ndistinct_hist.items())),
                "contested_frac": S.R(float(np.mean([x >= 2 for x in nd])), 5),
            },
        }
        print(f"[{cell}] n={len(idxs)} judge greedy={blk_j['greedy_matched_control']:.4f} "
              f"vote@8={blk_j['vote@8']:.4f} d={blk_j['vs_matched_greedy']['N=8']['delta']:+.4f} "
              f"{blk_j['vs_matched_greedy']['N=8']['verdict']}  ({time.time()-t0:.0f}s)", flush=True)
        # keep the per-item vectors for the macro / permutation stages
        np.savez(os.path.join(S.PARTS, f"vec_{cell}.npz"),
                 **{f"j_{k}": v for k, v in arms_j.items()},
                 **{f"e_{k}": v for k, v in arms_e.items()})
    os.makedirs(S.PARTS, exist_ok=True)
    print("wrote", S.dump_part("mcq.json", S.rj(out)))


# =============================================================================================
# STAGE closed -- SLAKE_closed / VQA_RAD_closed / PATH_VQA_closed
# =============================================================================================
def stage_closed(A):
    #: NULL TEST N3 -- the closed-grader copy must reproduce the stored `correct` field row by row
    g1 = L.grader_null_test()
    print("N3 closed grader null test:", json.dumps(g1), flush=True)
    out = {"null_test_N3_closed_grader": g1, "cells": {}}
    os.makedirs(S.PARTS, exist_ok=True)
    for cell in L.CELLS:
        t0 = time.time()
        jm = L.judge_map(cell)
        g = L.load_gen(cell, "closedD_g")
        s = L.load_gen(cell, "closedD_s8")
        idxs = sorted(set(g) & set(s))
        pools = {"judge": [], "em_harness": [], "em_repaired": []}
        gl = {"judge": [], "em_harness": [], "em_repaired": []}
        nd, unpar = [], 0
        for i in idxs:
            gold = g[i]["gold"]
            gp = g[i]["preds"][0]
            gl["judge"].append(int(jm[(i, L.norm_text(gp))]))
            gl["em_harness"].append(L.em_harness(cell, gold, gp))
            ok, up = L.em_repaired(cell, gold, gp)
            gl["em_repaired"].append(ok)
            unpar += up
            cls, lj, lh, lr = [], [], [], []
            for p in s[i]["preds"]:
                na = L.norm_text(p)
                cls.append(na)
                lj.append(int(jm[(i, na)]))
                lh.append(L.em_harness(cell, gold, p))
                ok, up = L.em_repaired(cell, gold, p)
                lr.append(ok)
                unpar += up
            pools["judge"].append((cls, lj))
            pools["em_harness"].append((cls, lh))
            pools["em_repaired"].append((cls, lr))
            nd.append(len(set(cls)))
        rec = {"n": len(idxs),
               "pool": "T=0.4, n=8, cap320, DEPLOYED MedEvalKit prompt (closedD_s8, in-session)",
               "greedy_control": "T=0, cap320, same prompt/session (closedD_g)",
               "published_always_7b": S.PUBLISHED_ALWAYS_7B[cell][0],
               "published_grader": S.PUBLISHED_ALWAYS_7B[cell][1],
               "published_resolution": "fullres -- the matched control here is cap320, so the "
                                       "in-session greedy differs from the published cell by the "
                                       "resolution term reported below",
               "diagnostics": {"mean_distinct_answers_in_pool": S.R(float(np.mean(nd)), 4),
                               "distinct_hist": dict(sorted(Counter(nd).items())),
                               "contested_frac": S.R(float(np.mean([x >= 2 for x in nd])), 5),
                               "unparsed_rate_em_repaired": S.R(unpar / (9 * len(idxs)), 6)}}
        vecs = {}
        for cur in ["judge", "em_harness", "em_repaired"]:
            arms = per_item_arms(pools[cur], gl[cur])
            rec[f"{cur}_currency"] = cell_block(arms)
            if cur == "em_harness":
                rec["in_session_greedy_minus_published"] = S.R(
                    float(np.mean(gl[cur])) - S.PUBLISHED_ALWAYS_7B[cell][0])
            for k, v in arms.items():
                vecs[f"{cur[:1] if cur!='em_repaired' else 'r'}_{k}"] = v
        out["cells"][cell] = rec
        np.savez(os.path.join(S.PARTS, f"vec_{cell}.npz"), **vecs)
        b = rec["judge_currency"]
        print(f"[{cell}] n={len(idxs)} judge greedy={b['greedy_matched_control']:.4f} "
              f"vote@8={b['vote@8']:.4f} d={b['vs_matched_greedy']['N=8']['delta']:+.4f} "
              f"{b['vs_matched_greedy']['N=8']['verdict']}  ({time.time()-t0:.0f}s)", flush=True)
    print("wrote", S.dump_part("closed.json", S.rj(out)))


# =============================================================================================
# STAGE open -- SLAKE_open / VQA_RAD_open / PATH_VQA_open at T=0.4, three generation seeds
# =============================================================================================
OPEN_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
OPEN_CELL = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open",
             "pathvqa_open": "PATH_VQA_open"}


def stage_open(A):
    from src.training_methods import genframe_data as G
    from src.cascade_methods.decoding_sweep_analyse import load_judge, load_pool

    #: NULL TEST N1 -- the frozen open-text metric
    items = G.load_items()
    r = G.sel_eff(G.incumbent_scores(), items)
    n1 = {"sel_eff": r["sel_eff"], "oracle@8": r["oracle"], "greedy": r["greedy"],
          "n": r["n"], "n_recoverable": int(r["rec"].sum()),
          "published": {"sel_eff": 0.775204, "oracle@8": 0.626013, "greedy": 0.449467,
                        "n": 2345, "n_recoverable": 1468}}
    n1["max_abs_deviation"] = max(abs(n1[k] - n1["published"][k])
                                  for k in ["sel_eff", "oracle@8", "greedy"])
    n1["pass"] = bool(n1["max_abs_deviation"] <= 1e-6 and n1["n"] == 2345
                      and n1["n_recoverable"] == 1468)
    #: kept as a STRING because the artifact writer rounds floats to 6 dp, which would print a
    #: 3.6e-07 deviation as 0.0 and overstate the agreement.
    n1["max_abs_deviation_exact"] = repr(n1["max_abs_deviation"])
    n1["tolerance"] = 1e-6
    print("N1 frozen open metric:", json.dumps(n1), flush=True)

    lab = load_judge()
    ref = items
    out = {"null_test_N1_frozen_open_metric": n1, "seeds": ["s0", "s1", "s2"], "cells": {}}
    os.makedirs(S.PARTS, exist_ok=True)

    per_seed = {}          # (seed) -> {ds -> {arm -> per-item vector}}
    for sd in ["s0", "s1", "s2"]:
        pg = load_pool(f"T00_{sd}", strict=False)
        ps = load_pool(f"T04_{sd}", strict=False)
        assert pg is not None and ps is not None, sd
        per_seed[sd] = {}
        for ds in OPEN_DS:
            rows = [it for it in ref if it["ds"] == ds]
            pools_j, pools_e, gj, ge, nd = [], [], [], [], []
            for it in rows:
                idx = it["idx"]
                rg = pg[(ds, idx)]
                gp = rg["preds"][0]
                gj.append(int(lab[(ds, idx, G.norm(gp))]))
                ge.append(int(rg["oks_em"][0]))
                rs = ps[(ds, idx)]
                cls, lj, le = [], [], []
                for k, p in enumerate(rs["preds"]):
                    na = G.norm(p)
                    cls.append(na)
                    lj.append(int(lab[(ds, idx, na)]))
                    le.append(int(rs["oks_em"][k]))
                pools_j.append((cls, lj))
                pools_e.append((cls, le))
                nd.append(len(set(cls)))
            per_seed[sd][ds] = {"judge": per_item_arms(pools_j, gj),
                                "em": per_item_arms(pools_e, ge),
                                "n_distinct": np.asarray(nd, float)}
        print(f"[open] seed {sd} done", flush=True)

    for ds in OPEN_DS:
        cell = OPEN_CELL[ds]
        rec = {"n": int(len(per_seed["s0"][ds]["judge"]["greedy"])),
               "pool": "T=0.4, n=8, cap320, SYS_OPEN prompt, 3 generation seeds (decoding_sweep)",
               "greedy_control": "T=0.0 n=1, SAME session/engine/prompt/resolution, 3 seeds (T00)",
               "published_always_7b": S.PUBLISHED_ALWAYS_7B[cell][0],
               "published_grader": S.PUBLISHED_ALWAYS_7B[cell][1],
               "diagnostics": {"mean_distinct_answers_in_pool":
                               S.R(float(np.mean([per_seed[s][ds]["n_distinct"].mean()
                                                  for s in per_seed])), 4)}}
        for cur in ["judge", "em"]:
            # seed-averaged per-item vectors (the estimand is the expectation over generation seeds)
            arms = {}
            for k in per_seed["s0"][ds][cur]:
                arms[k] = np.mean([per_seed[s][ds][cur][k] for s in per_seed], axis=0)
            blk = cell_block(arms)
            blk["per_seed"] = {
                f"vote@{N}": {"mean": S.R(float(np.mean([per_seed[s][ds][cur][f"vote@{N}"].mean()
                                                         for s in per_seed]))),
                              "sd": S.R(float(np.std([per_seed[s][ds][cur][f"vote@{N}"].mean()
                                                      for s in per_seed], ddof=1))),
                              "values": [S.R(per_seed[s][ds][cur][f"vote@{N}"].mean())
                                         for s in ["s0", "s1", "s2"]]}
                for N in [2, 4, 8]}
            blk["per_seed"]["greedy"] = {
                "mean": S.R(float(np.mean([per_seed[s][ds][cur]["greedy"].mean() for s in per_seed]))),
                "sd": S.R(float(np.std([per_seed[s][ds][cur]["greedy"].mean() for s in per_seed],
                                       ddof=1))),
                "values": [S.R(per_seed[s][ds][cur]["greedy"].mean()) for s in ["s0", "s1", "s2"]]}
            rec[f"{cur}_currency"] = blk
            if cur == "judge":
                rec["in_session_greedy_minus_published"] = S.R(
                    blk["greedy_matched_control"] - S.PUBLISHED_ALWAYS_7B[cell][0])
            np.savez(os.path.join(S.PARTS, f"vec_{cell}_{cur}.npz"), **arms)
        b = rec["judge_currency"]
        print(f"[{cell}] n={rec['n']} judge greedy={b['greedy_matched_control']:.4f} "
              f"vote@8={b['vote@8']:.4f} d={b['vs_matched_greedy']['N=8']['delta']:+.4f} "
              f"{b['vs_matched_greedy']['N=8']['verdict']}", flush=True)
        out["cells"][cell] = rec

    # pooled 2345 view, for direct comparison with decoding_ladder_cold's pooled MODAL_VOTE
    pooled = {}
    for cur in ["judge", "em"]:
        arms = {}
        for k in per_seed["s0"]["slake_open"][cur]:
            arms[k] = np.concatenate([np.mean([per_seed[s][ds][cur][k] for s in per_seed], axis=0)
                                      for ds in OPEN_DS])
        pooled[cur] = cell_block(arms)
        np.savez(os.path.join(S.PARTS, f"vec_OPEN_POOLED_{cur}.npz"), **arms)
    pooled["cross_check_vs_decoding_ladder_cold_2026-08-14"] = {
        "their_T04_MODAL_VOTE_judge": 0.46012793176972283,
        "our_vote@8_judge": pooled["judge"]["vote@8"],
        "their_T04_MODAL_VOTE_em": 0.4578535891968727,
        "our_vote@8_em": pooled["em"]["vote@8"],
        "their_T00_greedy_judge": 0.4626865671641791,
        "our_greedy_judge": pooled["judge"]["greedy_matched_control"],
        "note": "their MODAL_VOTE grades the label of the pool's modal_pred STRING; ours grades the "
                "earliest slot of the modal CLASS. Identical when labels are per normalised answer "
                "(the judge currency). Any residual is printed here, not hidden."}
    out["POOLED_2345"] = pooled
    print("wrote", S.dump_part("open.json", S.rj(out)))


# =============================================================================================
# STAGE head -- SELF-CONSISTENCY vs THE FREE HEAD, on the pool where the head is actually scored
# =============================================================================================
def stage_head(A):
    """The head-vs-vote head-to-head must be on ONE pool.  The head is scored only on the FROZEN
    T=0.7 2345-item pool (feats_hidden), so that is the pool used.  The T=0.4 hybrid uses the
    incumbent LoRA verifier, which IS scored on those pools."""
    from src.training_methods import genframe_data as G
    from src.training_methods.genframe_selector import FrozenSelector, score_eval_pool
    from src.cascade_methods.decoding_sweep_analyse import load_judge, load_pool, load_vscores

    os.makedirs(S.PARTS, exist_ok=True)
    out = {}

    # ---------------------------------------------------------------- A. frozen T=0.7 pool
    sel = FrozenSelector.load()
    ev = G.load_candidates("eval", sel.recipe["mode"], layers=[sel.recipe["layer"]],
                           pooling=(sel.recipe["pooling"],), order=sel.recipe["row_order"])
    X = sel.standardize(ev.matrix(sel.recipe["pooling"], sel.recipe["layer"]))
    Lg = sel.head_logits(X, standardized=True)                     # (8 seeds, n_rows)
    items = G.load_items()
    qmap = {(q.ds, q.idx): q for q in ev.questions}

    fusion_scores, head_scores, inc_scores = {}, {}, {}
    for it in items:
        q = qmap[(it["ds"], it["idx"])]
        rows = np.asarray(q.slot_rows)
        hr = np.mean([G.rank_avg(Lg[s][rows]) for s in range(Lg.shape[0])], axis=0)
        inc = np.asarray(it["scores"], float)
        head_scores[(it["ds"], it["idx"])] = hr
        inc_scores[(it["ds"], it["idx"])] = inc
        fusion_scores[(it["ds"], it["idx"])] = G.rank_avg(inc) + G.rank_avg(hr)

    #: null test: the reloaded selector must reproduce the frozen published endpoint
    rf = G.sel_eff({k: list(v) for k, v in fusion_scores.items()}, items)
    rh = G.sel_eff({k: list(v) for k, v in head_scores.items()}, items)
    ri = G.sel_eff({k: list(v) for k, v in inc_scores.items()}, items)
    n6 = {"fusion_sel_eff": rf["sel_eff"], "fusion_acc": rf["acc"],
          "published_fusion_sel_eff": 0.810627, "published_fusion_acc": 0.507463,
          "head_only_sel_eff": rh["sel_eff"], "head_only_acc": rh["acc"],
          "incumbent_sel_eff": ri["sel_eff"], "incumbent_acc": ri["acc"],
          "published_incumbent_sel_eff": 0.775204, "published_incumbent_acc": 0.485288}
    n6["max_abs_deviation"] = max(abs(rf["sel_eff"] - 0.810627), abs(rf["acc"] - 0.507463),
                                  abs(ri["sel_eff"] - 0.775204), abs(ri["acc"] - 0.485288))
    n6["pass"] = bool(n6["max_abs_deviation"] <= 1e-6)
    print("N6 frozen selector reload:", json.dumps(n6), flush=True)
    out["null_test_N6_frozen_selector_reload"] = n6

    # per-item arms on the frozen pool.  The frozen transfer dumps carry the 32B JUDGE label per
    # slot (it["sl"]) and the greedy arm's judge label (it["greedy_ok"]); they do NOT carry a
    # slot-wise EM label, so the frozen-pool block is judge-only by construction and says so.
    lab = load_judge()
    ds_of = [it["ds"] for it in items]
    n = len(items)
    pools, greedy = [], []
    for it in items:
        pools.append(([G.norm(p) for p in it["preds"]], [int(x) for x in it["sl"]]))
        greedy.append(int(it["greedy_ok"]))
    arms = per_item_arms(pools, greedy)
    for name, sc in [("head_only", head_scores), ("fusion", fusion_scores),
                     ("incumbent_LoRA", inc_scores)]:
        v = np.zeros(n)
        for i, it in enumerate(items):
            v[i] = it["sl"][int(np.argmax(sc[(it["ds"], it["idx"])]))]
        arms[name] = v
    # HYBRID: vote shortlist, selector breaks the tie
    for shortlist in ["tiebreak", "top2"]:
        for name, sc in [("head_only", head_scores), ("fusion", fusion_scores)]:
            v = np.zeros(n)
            for i, it in enumerate(items):
                cls, lb = pools[i]
                v[i] = hybrid_label(cls, lb, sc[(it["ds"], it["idx"])], shortlist)
            arms[f"hybrid_{shortlist}_{name}"] = v
    frozen = {"pool": "FROZEN T=0.7 8-sample pool, 2345 items (the pool the head is scored on)",
              "currency": "32B LLM judge (the frozen endpoint); EM is NOT available slot-wise on this "
                          "pool, so the EM column here is 'not measured' -- the dual-currency read is "
                          "at T=0.4 in the open stage",
              "arms": {}, "vs_greedy": {}, "vs_vote@8": {}, "vs_head_only": {}}
    order = ["greedy", "vote@1", "vote@2", "vote@4", "vote@8",
             "vote@2_orderfree", "vote@4_orderfree", "vote@8_orderfree",
             "vote@2_prefix", "vote@4_prefix", "vote@8_prefix", "head_only", "fusion",
             "incumbent_LoRA", "hybrid_tiebreak_head_only", "hybrid_top2_head_only",
             "hybrid_tiebreak_fusion", "hybrid_top2_fusion", "oracle@8"]
    for k in order:
        frozen["arms"][k] = S.R(arms[k].mean())
    for k in order:
        if k == "greedy":
            continue
        frozen["vs_greedy"][k] = S.paired_boot(arms[k], arms["greedy"])
        frozen["vs_vote@8"][k] = S.paired_boot(arms[k], arms["vote@8"])
        frozen["vs_head_only"][k] = S.paired_boot(arms[k], arms["head_only"])
    frozen["per_cell"] = {}
    for ds in OPEN_DS:
        m = np.asarray([d == ds for d in ds_of])
        frozen["per_cell"][OPEN_CELL[ds]] = {
            "n": int(m.sum()),
            **{k: S.R(arms[k][m].mean()) for k in order},
            "vote@8_vs_greedy": S.paired_boot(arms["vote@8"][m], arms["greedy"][m]),
            "head_only_vs_greedy": S.paired_boot(arms["head_only"][m], arms["greedy"][m]),
            "head_only_vs_vote@8": S.paired_boot(arms["head_only"][m], arms["vote@8"][m]),
            "hybrid_tiebreak_head_only_vs_vote@8": S.paired_boot(
                arms["hybrid_tiebreak_head_only"][m], arms["vote@8"][m]),
            "hybrid_tiebreak_head_only_vs_head_only": S.paired_boot(
                arms["hybrid_tiebreak_head_only"][m], arms["head_only"][m]),
        }
    out["A_frozen_T07_pool_head_vs_vote"] = frozen
    np.savez(os.path.join(S.PARTS, "vec_FROZEN_T07.npz"), **arms)

    # ---------------------------------------------------------------- B. T=0.4 pools, LoRA hybrid
    vsc = load_vscores()
    b = {"pool": "T=0.4, n=8, cap320, 3 generation seeds",
         "selector": "the INCUMBENT LoRA verifier (ckpts/train/lora_verifier_disjoint), the only "
                     "selector scored on these pools. The free head has NEVER been scored on a "
                     "T=0.4 pool -- no hidden-state cache exists for them. NOT MEASURED, not "
                     "estimated.",
         "selector_cost_flopeq": "2.186187 per question prefix-shared "
                                 "[central_table_2026-08-16.json COST_TABLE], so unlike the vote this "
                                 "arm is NOT free",
         "cells": {}, "pooled": {}}
    seedsets = {}
    for sd in ["s0", "s1", "s2"]:
        ps = load_pool(f"T04_{sd}", strict=False)
        pg = load_pool(f"T00_{sd}", strict=False)
        seedsets[sd] = (ps, pg)
    acc = {}
    for cur in ["judge", "em"]:
        percell = {}
        for ds in OPEN_DS:
            rows = [it for it in items if it["ds"] == ds]
            stack = []
            for sd in ["s0", "s1", "s2"]:
                ps, pg = seedsets[sd]
                a = {"greedy": [], "vote@8": [], "lora": [], "hybrid_tiebreak": [],
                     "hybrid_top2": []}
                miss = 0
                for it in rows:
                    idx = it["idx"]
                    rs, rg = ps[(ds, idx)], pg[(ds, idx)]
                    cls = [G.norm(p) for p in rs["preds"]]
                    if cur == "judge":
                        lb = [int(lab[(ds, idx, c)]) for c in cls]
                        gk = int(lab[(ds, idx, G.norm(rg["preds"][0]))])
                    else:
                        lb = [int(x) for x in rs["oks_em"]]
                        gk = int(rg["oks_em"][0])
                    sc = np.asarray([vsc.get((ds, idx, p), G.MISSING_SCORE) for p in rs["preds"]],
                                    float)
                    miss += int(sum(1 for p in rs["preds"] if (ds, idx, p) not in vsc))
                    a["greedy"].append(gk)
                    a["vote@8"].append(S.vote_exact(cls, lb, 8))
                    a["lora"].append(lb[int(np.argmax(sc))])
                    a["hybrid_tiebreak"].append(hybrid_label(cls, lb, sc, "tiebreak"))
                    a["hybrid_top2"].append(hybrid_label(cls, lb, sc, "top2"))
                stack.append({k: np.asarray(v, float) for k, v in a.items()})
                if miss:
                    print(f"  !! {ds} {sd} {cur}: {miss} unscored slots", flush=True)
            avg = {k: np.mean([s[k] for s in stack], axis=0) for k in stack[0]}
            percell[ds] = avg
        acc[cur] = percell
        b["cells"][cur] = {}
        for ds in OPEN_DS:
            avg = percell[ds]
            b["cells"][cur][OPEN_CELL[ds]] = {
                "n": int(len(avg["greedy"])),
                **{k: S.R(v.mean()) for k, v in avg.items()},
                "vote@8_vs_greedy": S.paired_boot(avg["vote@8"], avg["greedy"]),
                "lora_vs_greedy": S.paired_boot(avg["lora"], avg["greedy"]),
                "hybrid_tiebreak_vs_vote@8": S.paired_boot(avg["hybrid_tiebreak"], avg["vote@8"]),
                "hybrid_tiebreak_vs_lora": S.paired_boot(avg["hybrid_tiebreak"], avg["lora"]),
                "hybrid_top2_vs_lora": S.paired_boot(avg["hybrid_top2"], avg["lora"]),
            }
        cat = {k: np.concatenate([percell[ds][k] for ds in OPEN_DS]) for k in percell[OPEN_DS[0]]}
        b["pooled"][cur] = {**{k: S.R(v.mean()) for k, v in cat.items()},
                            "vote@8_vs_greedy": S.paired_boot(cat["vote@8"], cat["greedy"]),
                            "lora_vs_greedy": S.paired_boot(cat["lora"], cat["greedy"]),
                            "lora_vs_vote@8": S.paired_boot(cat["lora"], cat["vote@8"]),
                            "hybrid_tiebreak_vs_lora": S.paired_boot(cat["hybrid_tiebreak"],
                                                                     cat["lora"]),
                            "hybrid_tiebreak_vs_vote@8": S.paired_boot(cat["hybrid_tiebreak"],
                                                                       cat["vote@8"]),
                            "hybrid_top2_vs_lora": S.paired_boot(cat["hybrid_top2"], cat["lora"])}
    out["B_T04_pools_lora_hybrid"] = b
    print("wrote", S.dump_part("head.json", S.rj(out)))


# =============================================================================================
# STAGE baseline -- IS THE PUBLISHED OPEN-CELL "ALWAYS-7B GREEDY" ACTUALLY A GREEDY NUMBER?
# =============================================================================================
def stage_baseline(A):
    """Found while running ATTACK 2: on the frozen open pool, `vote@8 - greedy` came out EXACTLY
    0.0000 with a zero-width CI on SLAKE_open.  That can only happen if the two arms are the same
    object.  This stage measures what the frozen `greedy_ok` field actually is, and what the true
    temperature-0 greedy decode of the same items scores, from the SAME June session and engine."""
    from src.training_methods import genframe_data as G
    items = G.load_items()
    CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")

    out = {"question": "the round brief and CLAUDE.md both call 0.7364 / 0.4650 / 0.3240 the "
                       "'always-7B GREEDY' cells and charge them 1.0 FLOP-eq. Are they greedy?",
           "how_this_was_found": "vote@8 minus greedy came out EXACTLY 0.0000 with a zero-width "
                                 "bootstrap CI on SLAKE_open (645/645 items identical), which is "
                                 "impossible for two independently generated arms.",
           "cells": {}}

    # (1) what IS greedy_ok?  agreement with the pool's modal-of-8 label, and with slot 0
    tot_modal = tot_slot0 = tot = 0
    for ds in OPEN_DS:
        rows = [it for it in items if it["ds"] == ds]
        am = a0 = 0
        for it in rows:
            cls = [G.norm(p) for p in it["preds"]]
            cnt = Counter(cls)
            first = {}
            for k, c in enumerate(cls):
                first.setdefault(c, k)
            win = min(cnt.items(), key=lambda kv: (-kv[1], first[kv[0]]))[0]
            am += int(it["sl"][first[win]] == it["greedy_ok"])
            a0 += int(it["sl"][0] == it["greedy_ok"])
        tot_modal += am
        tot_slot0 += a0
        tot += len(rows)
        out["cells"][OPEN_CELL[ds]] = {
            "n": len(rows),
            "published_always_7b_greedy": S.PUBLISHED_ALWAYS_7B[OPEN_CELL[ds]][0],
            "mean_greedy_ok_field": S.R(float(np.mean([it["greedy_ok"] for it in rows]))),
            "greedy_ok_equals_modal_of_8_label": f"{am}/{len(rows)} = {am/len(rows):.4f}",
            "greedy_ok_equals_slot0_label": f"{a0}/{len(rows)} = {a0/len(rows):.4f}"}
    out["pooled_greedy_ok_equals_modal_of_8"] = f"{tot_modal}/{tot} = {tot_modal/tot:.4f}"
    out["pooled_greedy_ok_equals_slot0"] = f"{tot_slot0}/{tot} = {tot_slot0/tot:.4f}"

    # (2) the TRUE temperature-0 greedy decode of the same items, same June session and engine
    have_all = True
    for ds in OPEN_DS:
        p = os.path.join(CK, f"ckpt_{ds}_lingshu7b.judge.jsonl")
        if not os.path.exists(p):
            have_all = False
            continue
        jm = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                jm[r["idx"]] = int(r["judge_ok"])
        rows = [it for it in items if it["ds"] == ds]
        miss = sum(1 for it in rows if it["idx"] not in jm)
        t0 = np.asarray([jm.get(it["idx"], 0) for it in rows], float)
        gk = np.asarray([it["greedy_ok"] for it in rows], float)
        c = out["cells"][OPEN_CELL[ds]]
        c["TRUE_temperature0_greedy_acc"] = S.R(float(t0.mean()))
        c["n_items_without_a_t0_label_scored_0"] = int(miss)
        c["TRUE_greedy_minus_published_modal_of_8"] = S.rj(S.paired_boot(t0, gk))
        c["t0_source"] = f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.judge.jsonl " \
                         "(June 2026, T=0 decode of the same items, same 32B judge, same session " \
                         "and engine as the sc8 pool -- so this is a MATCHED control, not a " \
                         "cross-session comparison)"
    if have_all:
        allt0, allgk = [], []
        for ds in OPEN_DS:
            jm = {}
            for line in open(os.path.join(CK, f"ckpt_{ds}_lingshu7b.judge.jsonl"), encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    jm[r["idx"]] = int(r["judge_ok"])
            rows = [it for it in items if it["ds"] == ds]
            allt0 += [jm.get(it["idx"], 0) for it in rows]
            allgk += [it["greedy_ok"] for it in rows]
        a, b = np.asarray(allt0, float), np.asarray(allgk, float)
        out["POOLED_2345"] = {"true_t0_greedy": S.R(float(a.mean())),
                              "published_greedy_ok_modal_of_8": S.R(float(b.mean())),
                              "difference": S.rj(S.paired_boot(a, b))}
        # MACRO consequence: 5 cells are unchanged (their published value is a real greedy decode),
        # 3 cells move by the measured per-cell difference.  Bootstrapped by resampling items inside
        # each of the three moving cells; the five fixed cells contribute a constant 0.
        pairs = {}
        for ds in OPEN_DS:
            jm = {}
            for line in open(os.path.join(CK, f"ckpt_{ds}_lingshu7b.judge.jsonl"), encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    jm[r["idx"]] = int(r["judge_ok"])
            rows = [it for it in items if it["ds"] == ds]
            pairs[OPEN_CELL[ds]] = (np.asarray([jm.get(it["idx"], 0) for it in rows], float),
                                    np.asarray([it["greedy_ok"] for it in rows], float))
        mb3 = S.macro_boot(pairs)
        out["MACRO8_consequence"] = {
            "how": "five of the eight cells are untouched (their published value IS a real greedy "
                   "decode); the three open cells move by the measured per-cell difference. The "
                   "macro is the equal-weight mean over eight cells, so the shift is "
                   "(sum of the three per-cell differences) / 8, bootstrapped by resampling items "
                   "inside each of the three moving cells.",
            "macro3_delta_over_the_three_open_cells": S.R(mb3["macro_delta"]),
            "macro8_shift": S.R(mb3["macro_delta"] * 3.0 / 8.0),
            "macro8_shift_ci": [S.R(mb3["ci"][0] * 3.0 / 8.0), S.R(mb3["ci"][1] * 3.0 / 8.0)],
            "verdict": mb3["verdict"],
            "stated_macro_baseline": S.MACRO_BASELINE,
            "corrected_macro_baseline_true_greedy": S.R(S.MACRO_BASELINE + mb3["macro_delta"] * 3 / 8),
            "and_it_is_cheaper": "the corrected baseline is a REAL greedy decode on all eight cells, "
                                 f"so it costs 1.0 FLOP-eq, against "
                                 f"{S.R((5 * 1.0 + 3 * S.FLOPEQ_BY_N[8]) / 8)} for the published one.",
            "read_as_an_intervention": "stated the other way round: DELETING self-consistency from the "
                                       "three open cells of the published baseline is the cheapest "
                                       "intervention this round found -- it moves the macro UP and "
                                       "cuts baseline compute by a third."}

    out["CODE_CITATIONS_the_repo_already_says_so_in_three_places"] = [
        "src/cascade_methods/gen_slake_open_bestofN.py:138  greedy_ok = int(aj[i].get(modal_norm, 0)) "
        "-- the field is WRITTEN as the modal-of-8 label",
        "src/cascade_methods/gen_slake_open_bestofN.py:163  '# self-consistency modal (dump greedy_ok)'",
        "src/cascade_methods/verifier_n_scaling.py:173  '# NB: this is the MODAL-of-8 answer, i.e. SC@8'",
        "src/cascade_methods/inference_params_verify.py:125  'greedy_ok (= modal-of-8, the frozen "
        "metric's definition)'",
    ]
    out["CONSEQUENCE_accuracy"] = ("on the three open cells the 'always-7B greedy' baseline IS "
                                   "self-consistency@8. Attack 2 asks whether self-consistency beats "
                                   "always-7B greedy; on 3 of the 8 cells the incumbent baseline was "
                                   "already self-consistency@8, so that comparison was self-referential "
                                   "until this round supplied a real greedy control.")
    out["CONSEQUENCE_cost"] = {
        "stated_baseline_cost": "1.0 FLOP-eq on all eight cells",
        "measured_cost_of_modal_of_8": S.R(S.FLOPEQ_BY_N[8]),
        "corrected_macro8_baseline_cost": S.R((5 * 1.0 + 3 * S.FLOPEQ_BY_N[8]) / 8),
        "arithmetic": "five cells at 1.0 plus three cells at the measured 2.369969 = 1.513738",
        "source": S.FLOPEQ_SOURCE,
        "read": "the baseline every arm in this round is measured against is not a 1.0 FLOP-eq "
                "baseline. It is a 1.51 FLOP-eq baseline under the project's own measured generation "
                "cost, because three of its eight cells already pay for eight samples."}
    out["WHAT_IS_NOT_CLAIMED"] = ("this does NOT say any published DELTA is wrong. Every arm in the "
                                  "open-cell chain (verifier, head, oracle) was compared against the "
                                  "same greedy_ok field, so the deltas are internally consistent. What "
                                  "is wrong is the LABEL on the baseline and the 1.0 charged for it.")
    p = S.dump_part("baseline_defect.json", S.rj(out))
    print(json.dumps(S.rj(out), indent=1)[:3000])
    print("wrote", p)


def hybrid_label(cls, lab, scores, shortlist):
    """Vote-shortlist + selector tie-break, at N=8 (the full pool).

    shortlist='tiebreak': the shortlist is the set of classes tied at the maximum vote count.  When
        the plurality is unique the hybrid IS the plain vote -- the selector never overrides a real
        plurality.
    shortlist='top2': the shortlist is every class whose vote count is one of the top TWO distinct
        counts, so the selector gets a say whenever the pool is contested.
    The selector then picks the shortlisted SLOT with the highest score; first-index tie-break.
    """
    cnt = Counter(cls)
    counts = sorted(set(cnt.values()), reverse=True)
    keep = ({c for c, v in cnt.items() if v == counts[0]} if shortlist == "tiebreak"
            else {c for c, v in cnt.items() if v in counts[:2]})
    idxs = [k for k in range(len(cls)) if cls[k] in keep]
    if len(keep) == 1:
        first = {}
        for k in range(len(cls)):
            first.setdefault(cls[k], k)
        return lab[first[next(iter(keep))]]
    best = max(idxs, key=lambda k: (scores[k], -k))
    return lab[best]


# =============================================================================================
# STAGE finalize
# =============================================================================================
#: which grader each currency column actually is, per cell.  The eight published cells that sum to
#: macro 0.5971 are NOT graded by one rule -- MCQ by judge_multi_choice, closed by MedEvalKit's own
#: closed grader, open by the 32B judge -- so the column has to be named per cell or the table lies.
CURRENCY_NAME = {
  "primary": {"PMC_VQA": "MedEvalKit judge_multi_choice (= the published grader)",
              "MedXpertQA-MM": "MedEvalKit judge_multi_choice (= the published grader)",
              "SLAKE_closed": "32B LLM judge (NOT the published grader; see the em column)",
              "VQA_RAD_closed": "32B LLM judge (NOT the published grader; see the em column)",
              "PATH_VQA_closed": "32B LLM judge (NOT the published grader; see the em column)",
              "SLAKE_open": "32B LLM judge (= the published grader)",
              "VQA_RAD_open": "32B LLM judge (= the published grader)",
              "PATH_VQA_open": "32B LLM judge (= the published grader)"},
  "em": {"PMC_VQA": "letter_em (length-neutral; unparsed -> wrong)",
         "MedXpertQA-MM": "letter_em (length-neutral; unparsed -> wrong)",
         "SLAKE_closed": "MedEvalKit judge_close_end_vqa = THE PUBLISHED GRADER of this cell",
         "VQA_RAD_closed": "MedEvalKit judge_judgement = THE PUBLISHED GRADER of this cell",
         "PATH_VQA_closed": "MedEvalKit judge_judgement = THE PUBLISHED GRADER of this cell",
         "SLAKE_open": "normalised exact match", "VQA_RAD_open": "normalised exact match",
         "PATH_VQA_open": "normalised exact match"},
}


def stage_finalize(A):
    P = lambda f: json.load(open(os.path.join(S.PARTS, f)))
    mcq, closed, open_ = P("mcq.json"), P("closed.json"), P("open.json")
    head = P("head.json") if os.path.exists(os.path.join(S.PARTS, "head.json")) else None
    slots = (P("slot_exchangeability.json")
             if os.path.exists(os.path.join(S.PARTS, "slot_exchangeability.json")) else None)
    basedef = (P("baseline_defect.json")
               if os.path.exists(os.path.join(S.PARTS, "baseline_defect.json")) else None)

    # ---- per-item vectors, one primary-currency pair per cell
    vec = {}
    for cell in MCQ_CELLS:
        z = np.load(os.path.join(S.PARTS, f"vec_{cell}.npz"))
        vec[cell] = {"primary": {k[2:]: z[k] for k in z.files if k.startswith("j_")},
                     "em": {k[2:]: z[k] for k in z.files if k.startswith("e_")}}
    for cell in L.CELLS:
        z = np.load(os.path.join(S.PARTS, f"vec_{cell}.npz"))
        vec[cell] = {"primary": {k[2:]: z[k] for k in z.files if k.startswith("j_")},
                     "em": {k[2:]: z[k] for k in z.files if k.startswith("e_")},
                     "em_repaired": {k[2:]: z[k] for k in z.files if k.startswith("r_")}}
    for ds in OPEN_DS:
        cell = OPEN_CELL[ds]
        zj = np.load(os.path.join(S.PARTS, f"vec_{cell}_judge.npz"))
        ze = np.load(os.path.join(S.PARTS, f"vec_{cell}_em.npz"))
        vec[cell] = {"primary": {k: zj[k] for k in zj.files},
                     "em": {k: ze[k] for k in ze.files}}

    # ---- the 8-cell table
    table, macro, guard = {}, {}, {}
    for cur in ["primary", "em"]:
        rows, pairs_by_N = {}, {N: {} for N in [1, 2, 4, 8]}
        for cell in S.CELL_ORDER:
            v = vec[cell][cur]
            row = {"n": int(len(v["greedy"])),
                   "currency": CURRENCY_NAME[cur][cell],
                   "published_always_7b": S.PUBLISHED_ALWAYS_7B[cell][0],
                   "published_grader": S.PUBLISHED_ALWAYS_7B[cell][1],
                   "in_session_greedy": S.R(v["greedy"].mean()),
                   "random_slot_floor": S.R(v["vote@1"].mean())}
            pairs_by_N[1][cell] = (v["vote@1"], v["greedy"])
            row["delta@1_random_single_draw"] = S.rj(S.paired_boot(v["vote@1"], v["greedy"]))
            for N in [2, 4, 8]:
                bt = S.paired_boot(v[f"vote@{N}"], v["greedy"])
                row[f"vote@{N}"] = S.R(v[f"vote@{N}"].mean())
                row[f"delta@{N}"] = S.rj(bt)
                row[f"delta@{N}_orderfree"] = S.rj(S.paired_boot(v[f"vote@{N}_orderfree"],
                                                                 v["greedy"]))
                row[f"delta@{N}_prefix"] = S.rj(S.paired_boot(v[f"vote@{N}_prefix"], v["greedy"]))
                pairs_by_N[N][cell] = (v[f"vote@{N}"], v["greedy"])
            rows[cell] = row
        table[cur] = rows
        macro[cur] = {}
        guard[cur] = {}
        for N in [1, 2, 4, 8]:
            mb = S.macro_boot(pairs_by_N[N])
            if N == 1:
                mb["note"] = ("N=1 is a SINGLE RANDOM DRAW from the T=0.4 pool -- the random-pick "
                              "floor. It is the left end of the curve and is not a vote.")
                mb["macro_accuracy_if_baseline_is_0.5971"] = S.R(S.MACRO_BASELINE + mb["macro_delta"])
                macro[cur]["N=1"] = mb
                guard[cur]["N=1"] = {"CI_clean_losses":
                                     [c for c in S.CELL_ORDER
                                      if rows[c]["delta@1_random_single_draw"]["verdict"] == "LOSS"],
                                     "CI_clean_wins":
                                     [c for c in S.CELL_ORDER
                                      if rows[c]["delta@1_random_single_draw"]["verdict"] == "WIN"]}
                guard[cur]["N=1"]["guardrail_clean"] = not guard[cur]["N=1"]["CI_clean_losses"]
                continue
            mb["orderfree"] = S.rj(S.macro_boot(
                {c: (vec[c][cur][f"vote@{N}_orderfree"], vec[c][cur]["greedy"])
                 for c in S.CELL_ORDER}))
            mb["prefix_first_N_slots"] = S.rj(S.macro_boot(
                {c: (vec[c][cur][f"vote@{N}_prefix"], vec[c][cur]["greedy"])
                 for c in S.CELL_ORDER}))
            mb["macro_accuracy_if_baseline_is_0.5971"] = S.R(S.MACRO_BASELINE + mb["macro_delta"])
            mb["sum_of_per_cell_deltas"] = S.R(sum(rows[c][f"delta@{N}"]["delta"]
                                                   for c in S.CELL_ORDER))
            macro[cur][f"N={N}"] = mb
            losses = [c for c in S.CELL_ORDER if rows[c][f"delta@{N}"]["verdict"] == "LOSS"]
            wins = [c for c in S.CELL_ORDER if rows[c][f"delta@{N}"]["verdict"] == "WIN"]
            guard[cur][f"N={N}"] = {"CI_clean_losses": losses, "CI_clean_wins": wins,
                                    "guardrail_clean": bool(not losses),
                                    "n_cells_win": len(wins), "n_cells_loss": len(losses)}

    # ---- permutation null over the (cell x N) grid: the vote replaced by a random slot
    rng = np.random.default_rng(S.PSEED)
    obs, nullmax = {}, np.zeros(S.NPERM)
    for cell in S.CELL_ORDER:
        v = vec[cell]["primary"]
        for N in [2, 4, 8]:
            obs[f"{cell}@{N}"] = S.R(v[f"vote@{N}"].mean() - v["greedy"].mean())
    # the null arm is the random-slot arm (vote@1), whose per-item value is fixed; the permutation
    # resamples WHICH slot each item returns by drawing a slot label at random -> Bernoulli(vote@1)
    for b in range(S.NPERM):
        mx = -9.0
        for cell in S.CELL_ORDER:
            v = vec[cell]["primary"]
            p = v["vote@1"]
            gm = float(v["greedy"].mean())
            for _N in [2, 4, 8]:
                draw = (rng.random(len(p)) < p).astype(float)   # independent per (cell, N)
                mx = max(mx, float(draw.mean()) - gm)
        nullmax[b] = mx
    # z-standardised variant: the raw max-of-grid is dominated by the SMALLEST cell (VQA_RAD_closed,
    # n=251), whose random-slot arm is simply noisier, so the same test is repeated on per-cell
    # z-scores where every cell contributes on the same scale.
    rng2 = np.random.default_rng(S.PSEED)
    sd_null, null_draw = {}, {}
    for cell in S.CELL_ORDER:
        v = vec[cell]["primary"]
        p = v["vote@1"]
        D = np.empty(S.NPERM)
        for b in range(S.NPERM):
            D[b] = (rng2.random(len(p)) < p).astype(float).mean() - float(v["greedy"].mean())
        sd_null[cell] = float(D.std(ddof=1))
        null_draw[cell] = D
    #: the null is NOT centred at zero -- a random slot is genuinely worse than greedy -- so both the
    #: observed and the null are centred on the null's own mean before standardising.  (An earlier
    #: pass took |D| of an uncentred null and produced a meaningless null_max_z of 4.6.)
    mu_null = {c: float(null_draw[c].mean()) for c in S.CELL_ORDER}
    obs_z = {f"{cell}@{N}": S.R((obs[f"{cell}@{N}"] - mu_null[cell]) / sd_null[cell])
             for cell in S.CELL_ORDER for N in [2, 4, 8]}
    nullmax_z = np.max(np.stack([(null_draw[c] - mu_null[c]) / sd_null[c] for c in S.CELL_ORDER]),
                       axis=0)
    bestz = max(obs_z.values())
    permnull_z = {"statistic": "max over the 24 (cell x N) combinations of "
                               "(arm - greedy - mean of that cell's random-slot null) / sd of that "
                               "null; i.e. how many random-slot standard deviations above a random "
                               "pick the vote sits, on one scale for every cell",
                  "per_cell_null_mean_random_slot_minus_greedy": {c: S.R(mu_null[c])
                                                                  for c in S.CELL_ORDER},
                  "per_cell_null_sd": {c: S.R(sd_null[c]) for c in S.CELL_ORDER},
                  "observed_z_per_combination": obs_z,
                  "observed_best": max(obs_z, key=obs_z.get), "observed_best_z": S.R(bestz),
                  "null_max_z_mean": S.R(float(nullmax_z.mean())),
                  "null_max_z_p95": S.R(float(np.percentile(nullmax_z, 95))),
                  "p_value_best_of_grid": S.R(float((nullmax_z >= bestz).mean())),
                  "significant_at_0.05": bool(float((nullmax_z >= bestz).mean()) < 0.05)}

    best = max(obs.values())
    permnull = {"nperm": S.NPERM, "seed": S.PSEED,
                "z_standardised_variant": permnull_z,
                "what_needs_no_correction": "the PRE-SPECIFIED comparison is the 8-cell MACRO, a "
                                            "single number; and PMC_VQA@8 was already measured and "
                                            "published on 2026-08-16 before this grid existed "
                                            "(_closed_as_open_parts/mcq_self_consistency.json), so "
                                            "neither is a best-of-24 selection. This correction "
                                            "guards the reader against the OTHER 23 cells.",
                "null": "the vote's pick is replaced by a uniformly random slot from the SAME pool "
                        "(one Bernoulli draw per item at the item's random-slot rate); the statistic "
                        "is the max over the 24 (cell x N) combinations of (arm - matched greedy)",
                "n_combinations": len(obs), "observed_delta_per_combination": obs,
                "observed_best": max(obs, key=obs.get), "observed_best_delta": S.R(best),
                "null_max_mean": S.R(float(nullmax.mean())),
                "null_max_p95": S.R(float(np.percentile(nullmax, 95))),
                "p_value_best_of_grid": S.R(float((nullmax >= best).mean())),
                "significant_at_0.05": bool(float((nullmax >= best).mean()) < 0.05)}

    # ---- cost
    cost = {"unit": "1.0 FLOP-eq = one always-7B greedy answer",
            "self_consistency_charges_generation_only":
                "no verifier forward, no head forward, no 32B call -- the arm's whole cost is N "
                "samples",
            "MEASURED_flopeq_by_N_vllm_default_prefix_caching_on":
                {str(k): S.R(v) for k, v in S.FLOPEQ_BY_N.items()},
            "source": S.FLOPEQ_SOURCE,
            "same_measurement_flag_forced_on": {str(k): S.R(v)
                                                for k, v in S.FLOPEQ_BY_N_CACHE_ON.items()},
            "control_prefix_caching_OFF_validates_the_instrument":
                {str(k): S.R(v) for k, v in S.FLOPEQ_BY_N_CACHE_OFF.items()},
            "control_note": "with caching off every term scales as exactly N, reproducing the "
                            "project's as-charged 8.0 convention to 0.3% -- that is what validates "
                            "the instrument",
            "N8_with_precomputed_image_embeddings": S.R(S.FLOPEQ_N8_VISION_EMBEDS),
            "N8_with_precomputed_image_embeddings_source": S.FLOPEQ_N8_VISION_EMBEDS_SOURCE,
            "caveat_geometry": "the ratio was measured on the open-text cap320 geometry. Mean prefill "
                               "differs per cell (275-868 tokens, "
                               "_closed_as_open_parts/prefill_cost.json) and the ratio has NOT been "
                               "separately measured on the MCQ or closed cells. Applied uniformly "
                               "here as the best available MEASUREMENT, and flagged as such.",
            "macro8_cost_per_N": {f"N={N}": S.R(S.FLOPEQ_BY_N[N]) for N in [1, 2, 4, 8]}}

    # ---- currency divergence: judge vs EM, per cell per N (the project's artefact tripwire)
    div = {"why": "every time the judge and exact match have diverged sharply in this project it has "
                  "indicated an artefact rather than a gain, so the gap is reported per cell per N "
                  "rather than left to be discovered.",
           "per_cell": {}, "worst_abs_gap": 0.0, "worst_cell": None}
    for cell in S.CELL_ORDER:
        r = {}
        for N in [2, 4, 8]:
            jd = table["primary"][cell][f"delta@{N}"]["delta"]
            ed = table["em"][cell][f"delta@{N}"]["delta"]
            r[f"N={N}"] = {"primary_delta": S.R(jd), "em_delta": S.R(ed),
                           "gap": S.R(jd - ed),
                           "sign_agreement": bool((jd >= 0) == (ed >= 0))}
            if abs(jd - ed) > div["worst_abs_gap"]:
                div["worst_abs_gap"] = S.R(abs(jd - ed))
                div["worst_cell"] = f"{cell} N={N}"
        r["currencies"] = [CURRENCY_NAME["primary"][cell], CURRENCY_NAME["em"][cell]]
        div["per_cell"][cell] = r
    div["read"] = ("the two currencies agree in sign on every cell/N pair where the flag below is "
                   "true. Where they do not, the effect is at or below the noise floor of both.")
    div["n_sign_disagreements"] = sum(
        0 if div["per_cell"][c][f"N={N}"]["sign_agreement"] else 1
        for c in S.CELL_ORDER for N in [2, 4, 8])

    # ---- IS THE N-CURVE EVEN IDENTIFIED?  (it is not, on the non-exchangeable pools)
    ident = {"why": "selfcons_slots.py measured that 6 of 17 pools are NOT slot-exchangeable. On such "
                    "a pool 'N=2' is ambiguous: a random 2-subset of the 8-fan and the FIRST 2 slots "
                    "of the fan are different experiments. The spread between the three conventions "
                    "is reported per cell per N; where it exceeds the effect, the N point is NOT "
                    "IDENTIFIED and no claim is made at that N.",
             "per_cell": {}}
    worst = 0.0
    for cell in S.CELL_ORDER:
        r = {}
        for N in [2, 4, 8]:
            ds = [table["primary"][cell][f"delta@{N}"]["delta"],
                  table["primary"][cell][f"delta@{N}_orderfree"]["delta"],
                  table["primary"][cell][f"delta@{N}_prefix"]["delta"]]
            sp = S.R(max(ds) - min(ds))
            worst = max(worst, sp)
            r[f"N={N}"] = {"random_subset_index_tiebreak": S.R(ds[0]),
                           "random_subset_orderfree_tiebreak": S.R(ds[1]),
                           "first_N_slots_of_the_fan": S.R(ds[2]),
                           "spread": sp,
                           "sign_flips_across_conventions": bool(min(ds) < 0 < max(ds)),
                           "identified": bool(sp <= abs(ds[0]) or sp < 0.002)}
        ident["per_cell"][cell] = r
    ident["worst_spread_over_all_cells_and_N"] = S.R(worst)
    ident["headline"] = ("on PMC_VQA at N=2 the delta is -0.0190 for a random 2-subset of the fan and "
                         "+0.0193 for the first 2 slots of the same fan -- a SIGN FLIP of 0.038 from "
                         "nothing but which two samples you get. N=8 uses the whole pool and is "
                         "invariant to all three conventions, so it is the only budget at which this "
                         "cell's number means anything.")

    # ---- the pre-registered decision rule, applied
    mcq_cells = ["PMC_VQA", "MedXpertQA-MM"]
    open_cells = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
    verdict = {"rule": "from the preregistration: the split is CONFIRMED iff (a) >=1 MCQ cell is a "
                       "CI-clean WIN for the vote at some N, (b) no open cell is a CI-clean WIN at "
                       "that N, and (c) the head beats the vote on the open cells on the same pool.",
               "per_N": {}}
    for N in [2, 4, 8]:
        mw = [c for c in mcq_cells if table["primary"][c][f"delta@{N}"]["verdict"] == "WIN"]
        ow = [c for c in open_cells if table["primary"][c][f"delta@{N}"]["verdict"] == "WIN"]
        ol = [c for c in open_cells if table["primary"][c][f"delta@{N}"]["verdict"] == "LOSS"]
        verdict["per_N"][f"N={N}"] = {"mcq_CI_clean_wins": mw, "open_CI_clean_wins": ow,
                                      "open_CI_clean_losses": ol,
                                      "condition_a": bool(mw), "condition_b": bool(not ow)}
    if head:
        hv = head["A_frozen_T07_pool_head_vs_vote"]["vs_vote@8"]["head_only"]
        verdict["condition_c_head_beats_vote_on_the_open_cells"] = {
            "pool": "FROZEN T=0.7 2345-item open pool (the pool the head is scored on)",
            "head_only_minus_vote@8": S.rj(hv), "holds": bool(hv["verdict"] == "WIN")}
    verdict["OUTCOME"] = (
        "CONFIRMED" if (verdict["per_N"]["N=8"]["condition_a"]
                        and verdict["per_N"]["N=8"]["condition_b"]
                        and head is not None
                        and verdict["condition_c_head_beats_vote_on_the_open_cells"]["holds"])
        else "PARTIAL_OR_REFUTED -- read per_N")

    art = {
      "title": "ATTACK 2 -- training-free SELF-CONSISTENCY (majority vote) across all eight macro "
               "cells at N = 1, 2, 4, 8",
      "date": S.DATE,
      "preregistration": "results/cascade_methods/artifacts/"
                         "self_consistency_suite_2026-08-17_preregistration.json",
      "code": ["src/cascade_methods/selfcons_lib.py", "src/cascade_methods/selfcons_suite.py",
               "src/cascade_methods/selfcons_prereg.py"],
      "no_fabricated_numbers": "every field is computed in this run from the named checkpoint dumps. "
                               "Numbers imported from other artifacts are labelled with their file.",
      "not_abstention": "every arm returns an answer for every question; an unparseable string is "
                        "graded WRONG, never withheld.",
      "zero_new_generation": "no GPU job was run. Every pool and every matched greedy control already "
                             "existed on disk.",
      "numerics": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
                   "device": "CPU only", "nboot": S.NBOOT, "bootstrap_seed": S.BSEED,
                   "nperm": S.NPERM, "permutation_seed": S.PSEED},
      "NULL_TESTS": {
        "N1_frozen_open_metric": open_["null_test_N1_frozen_open_metric"],
        "N2_mcq_grader": mcq["null_test_N2_mcq_grader"],
        "N3_closed_grader": closed["null_test_N3_closed_grader"],
        "N4_vote_machinery": vote_machinery_null(),
        "N5_exchangeability_vote@2_equals_random_slot": {
            cell: table["primary"][cell]["vote@2"] - table["primary"][cell]["random_slot_floor"]
            for cell in S.CELL_ORDER},
      },
      "VERDICT_against_the_preregistered_decision_rule": verdict,
      "IDENTIFIABILITY_of_the_N_curve": ident,
      "THE_TABLE_primary_currency": table["primary"],
      "THE_TABLE_em_diagnostic": table["em"],
      "CURRENCY_DIVERGENCE_judge_vs_em": div,
      "MACRO8": macro,
      "GUARDRAIL": guard,
      "PERMUTATION_NULL_over_the_cell_x_N_grid": permnull,
      "COST": cost,
      "PER_CELL_DETAIL": {"mcq": mcq["cells"], "closed": closed["cells"], "open": open_["cells"],
                          "open_pooled_2345": open_["POOLED_2345"]},
    }
    if head:
        head["READ_THIS_FIRST_the_frozen_pool_greedy_arm_is_itself_modal_of_8"] = (
            "on the FROZEN T=0.7 pool the 'greedy' arm is the dump's greedy_ok field, which this "
            "round measured to be the MODAL-OF-8 label on 2339/2345 items "
            "(_selfcons_parts/baseline_defect.json). The row 'vote@8 vs greedy' on that pool is "
            "therefore a TIE-BREAK CONVENTION difference, not an effect, and must not be read as "
            "'self-consistency loses on the open cells' -- that statement is supported instead by the "
            "T=0.4 block below and by the open stage, both of which use a REAL temperature-0 control. "
            "Everything else in this section (head vs vote, hybrid vs head) compares two arms to each "
            "other and is unaffected.")
        art["HEAD_VS_SELF_CONSISTENCY"] = head
    if slots:
        art["POOL_IS_NOT_SLOT_EXCHANGEABLE"] = slots
    if basedef:
        art["THE_PUBLISHED_OPEN_CELL_BASELINE_IS_ALREADY_SELF_CONSISTENCY_AT_8"] = basedef
    p = os.path.join(S.ART, "self_consistency_suite_2026-08-17.json")
    json.dump(S.rj(art), open(p, "w"), indent=1, ensure_ascii=False)
    print("wrote", p)
    # console summary
    print("\n=== PRIMARY CURRENCY, delta vs MATCHED in-session greedy ===")
    print(f"{'cell':<18}{'n':>7}{'greedy':>9}" + "".join(f"{'N='+str(N):>22}" for N in [2, 4, 8]))
    for cell in S.CELL_ORDER:
        r = table["primary"][cell]
        s = f"{cell:<18}{r['n']:>7}{r['in_session_greedy']:>9.4f}"
        for N in [2, 4, 8]:
            d = r[f"delta@{N}"]
            s += f"  {d['delta']:+.4f} {d['verdict']:<5}"
        print(s)
    for N in [2, 4, 8]:
        m = macro["primary"][f"N={N}"]
        print(f"MACRO N={N}: {m['macro_delta']:+.5f} [{m['ci'][0]:+.5f},{m['ci'][1]:+.5f}] "
              f"{m['verdict']}   cost {S.FLOPEQ_BY_N[N]:.3f} FLOP-eq   "
              f"guardrail_clean={guard['primary'][f'N={N}']['guardrail_clean']}")


def vote_machinery_null():
    """N4 -- the vote machinery, on synthetic pools with known answers."""
    out = {}
    # (a) weights sum to 1 at every N: guaranteed by the enumeration; assert via a class distribution
    cls = ["a", "a", "b", "c", "b", "a", "d", "b"]
    dev = 0.0
    for N in [1, 2, 4, 8]:
        d = S.vote_class_dist(cls, N)
        dev = max(dev, abs(sum(d.values()) - 1.0))
    out["a_subset_weights_sum_to_1_max_dev"] = dev
    # (b) at N = M the exact expectation equals the plain deterministic vote
    lab = [1, 1, 0, 0, 0, 1, 0, 0]
    first = {}
    for k, c in enumerate(cls):
        first.setdefault(c, k)
    cnt = Counter(cls)
    win = min(cnt.items(), key=lambda kv: (-kv[1], first[kv[0]]))[0]
    out["b_N_eq_M_equals_plain_vote"] = abs(S.vote_exact(cls, lab, 8) - lab[first[win]])
    # (c) at N=1 it equals the plain per-item mean of the slot labels
    out["c_N_eq_1_equals_slot_mean"] = abs(S.vote_exact(cls, lab, 1) - float(np.mean(lab)))
    # (d) tie-break: a 1-1 tie returns the EARLIER slot
    out["d_two_way_tie_returns_earlier_slot"] = abs(S.vote_exact(["x", "y"], [0, 1], 2) - 0.0)
    out["max_abs_deviation"] = max(v for k, v in out.items() if k.startswith(("a_", "b_", "c_", "d_")))
    out["pass"] = bool(out["max_abs_deviation"] <= 1e-12)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["mcq", "closed", "open", "head", "baseline", "finalize"])
    A = ap.parse_args()
    os.makedirs(S.PARTS, exist_ok=True)
    {"mcq": stage_mcq, "closed": stage_closed, "open": stage_open, "head": stage_head,
     "baseline": stage_baseline, "finalize": stage_finalize}[A.stage](A)
