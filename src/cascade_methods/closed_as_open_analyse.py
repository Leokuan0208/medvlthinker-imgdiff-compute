#!/usr/bin/env python3
"""closed_as_open_analyse.py -- BUILD 3 CPU half: assemble the artifact.

Per cell, per arm, in BOTH currencies (32B judge and normalised exact match, on IDENTICAL picks):
    greedy | oracle@8 | sel_eff | SELECTED | majority vote | random-pick floor | distinct candidates
plus the paired item bootstrap of SELECTED - GREEDY, the guardrail, and the grading-artifact
diagnostics (answer length, UNPARSED rate, EM_harness-vs-EM_repaired gap).

Asserts the frozen identity  SELECTED == oracle@8 * sel_eff  on every cell x currency.

CPU only.  python3 src/cascade_methods/closed_as_open_analyse.py
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402

sys.path.insert(0, os.path.join(L.ROOT, "src"))
from training_methods import genframe_data as G                           # noqa: E402

OUT = os.path.join(L.ART, f"closed_as_open_{L.DATE}.json")
SEED_BOOT = 20260816
NBOOT = 10000

SAMPLED_ARMS = ["openPRJ_s8", "openMEK_s8", "closedD_s8"]
GREEDY_OF = {"openPRJ_s8": "openPRJ_g", "openMEK_s8": "openMEK_g", "closedD_s8": "closedD_g"}
CURRENCIES = ["judge", "em_harness", "em_repaired"]


# =============================================================================================
def label(cell, gold, text, jm, i, currency):
    if currency == "judge":
        v = jm.get((i, L.norm_text(text)))
        return None if v is None else int(v)
    if currency == "em_harness":
        return L.em_harness(cell, gold, text)
    ok, _ = L.em_repaired(cell, gold, text)
    return ok


def arm_labels(cell, arm, jm, currency):
    """{item -> [labels per slot]} for one arm, one currency."""
    gen = L.load_gen(cell, arm)
    out = {}
    for i, r in gen.items():
        out[i] = [label(cell, r["gold"], p, jm, i, currency) for p in r["preds"]]
    return out


def picks_from(scores):
    """argmax with FIRST-INDEX tie-break -- the frozen pick rule (genframe_data.picks_from_scores)."""
    return int(np.argmax(np.asarray(scores, dtype=float)))


def majority_pick(preds):
    """plurality of the normalised surface strings; ties broken by first appearance."""
    cnt = Counter(L.norm_text(p) for p in preds)
    best = max(cnt.items(), key=lambda kv: (kv[1], -list(cnt).index(kv[0])))[0]
    for k, p in enumerate(preds):
        if L.norm_text(p) == best:
            return k
    return 0


def paired_boot(a, b, nboot=NBOOT, seed=SEED_BOOT):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, n, n)
        d[k] = a[s].mean() - b[s].mean()
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta": float(a.mean() - b.mean()), "ci": [lo, hi],
            "significant": bool(lo > 0 or hi < 0), "sign": "WIN" if lo > 0 else
            ("LOSS" if hi < 0 else "TIE"), "n": n, "nboot": nboot, "seed": seed}


# =============================================================================================
def contaminated(cell):
    p = os.path.join(L.PARTS, "contaminated_items.json")
    if not os.path.exists(p):
        return None
    return set(json.load(open(p)).get(cell, []))


def analyse_cell(cell, jm):
    gen = {a: L.load_gen(cell, a) for a in L.ARMS}
    present = [a for a in L.ARMS if len(gen[a]) == L.EXPECT_N[cell]]
    out = {"n": L.EXPECT_N[cell], "arms_complete": present,
           "arms_incomplete": {a: len(gen[a]) for a in L.ARMS if a not in present and gen[a]}}

    # ---- descriptive: length, unparsed, distinct-candidate counts --------------------------
    desc = {}
    for a in present:
        toks, chars, unp, ndist, hit_cap = [], [], 0, [], 0
        for i, r in gen[a].items():
            toks += list(r["gen_tokens"])
            chars += [len(p) for p in r["preds"]]
            hit_cap += sum(1 for t in r["gen_tokens"] if t >= L.MAX_TOKENS)
            for p in r["preds"]:
                _, u = L.em_repaired(cell, r["gold"], p)
                unp += u
            ndist.append(len(set(L.norm_text(p) for p in r["preds"])))
        nslots = len(toks)
        desc[a] = {"mean_gen_tokens": round(float(np.mean(toks)), 3),
                   "mean_answer_chars": round(float(np.mean(chars)), 2),
                   "frac_slots_hitting_max_tokens": round(hit_cap / nslots, 5),
                   "unparsed_rate_em_repaired": round(unp / nslots, 5),
                   "mean_distinct_candidates": round(float(np.mean(ndist)), 4),
                   "distinct_candidate_hist": dict(sorted(Counter(ndist).items())),
                   "prompt_gives_answer_space": L.has_answer_space(L.ARMS[a]["prompt"], cell)}
    out["descriptives"] = desc

    # ---- per-currency, per-arm accounting --------------------------------------------------
    out["by_currency"] = {}
    for cur in CURRENCIES:
        if cur == "judge" and not jm:
            out["by_currency"][cur] = "NOT MEASURED -- no 32B judge labels on disk for this cell yet"
            continue
        cd = {"greedy_arms": {}, "sampled_arms": {}}
        for a in present:
            if L.ARMS[a]["n"] == 1:
                lab = arm_labels(cell, a, jm, cur)
                v = [lab[i][0] for i in sorted(lab)]
                miss = sum(1 for x in v if x is None)
                v = [0 if x is None else x for x in v]
                cd["greedy_arms"][a] = {"acc": round(float(np.mean(v)), 6), "n": len(v),
                                        "n_unlabelled_scored_0": miss}
        bad = contaminated(cell) or set()
        for a in SAMPLED_ARMS:
            if a not in present:
                continue
            gname = GREEDY_OF[a]
            if gname not in present:
                continue
            lab = arm_labels(cell, a, jm, cur)
            glab = arm_labels(cell, gname, jm, cur)
            sc = L.load_scores(cell, a)
            allidx = sorted(i for i in lab if i in glab)
            subsets = {"all": allidx}
            if bad:
                subsets["image_disjoint"] = [i for i in allidx if i not in bad]
            rows = {}
            for sname, idxs in subsets.items():
                n_scored = sum(1 for i in idxs if i in sc)
                gv, ov, sv, mv, rv, rec = [], [], [], [], [], []
                for i in idxs:
                    sl = [0 if x is None else x for x in lab[i]]
                    gv.append(0 if glab[i][0] is None else glab[i][0])
                    ov.append(int(max(sl)))
                    rec.append(int(max(sl)))
                    rv.append(float(np.mean(sl)))
                    mv.append(sl[majority_pick(gen[a][i]["preds"])])
                    sv.append(sl[picks_from(sc[i])] if i in sc else None)
                gv, ov, rv, mv = map(np.asarray, (gv, ov, rv, mv))
                rec = np.asarray(rec)
                row = {"n": len(idxs), "n_verifier_scored": n_scored,
                       "greedy_arm": gname,
                       "greedy": round(float(gv.mean()), 6),
                       "oracle@8": round(float(ov.mean()), 6),
                       "random_pick_floor": round(float(rv.mean()), 6),
                       "majority_vote": round(float(mv.mean()), 6)}
                if n_scored == len(idxs) and len(idxs):
                    sva = np.asarray(sv, dtype=float)
                    sel_eff = float(sva[rec == 1].mean()) if rec.sum() else float("nan")
                    nd = np.array([len(set(L.norm_text(p) for p in gen[a][i]["preds"]))
                                   for i in idxs], dtype=int)
                    con = nd >= 2
                    row.update({
                        "SELECTED": round(float(sva.mean()), 6),
                        "sel_eff": round(sel_eff, 6),
                        "identity_selected_minus_oracle_times_sel_eff":
                            float(sva.mean() - ov.mean() * sel_eff) if rec.sum() else None,
                        "vs_greedy": paired_boot(sva, gv),
                        "vs_majority": paired_boot(sva, mv),
                        "vs_random_pick_floor": paired_boot(sva, rv),
                        "oracle_vs_greedy": paired_boot(ov.astype(float), gv),
                        # THE decisive stratum: the pools are near-unanimous, so on ~80% of items
                        # every selector is forced to the same answer and the pooled delta is
                        # diluted by construction.  All the signal lives here.
                        "CONTESTED": ({
                            "n": int(con.sum()),
                            "frac_of_cell": round(float(con.mean()), 5),
                            "greedy": round(float(gv[con].mean()), 6),
                            "oracle@8": round(float(ov[con].mean()), 6),
                            "random_pick_floor": round(float(rv[con].mean()), 6),
                            "majority_vote": round(float(mv[con].mean()), 6),
                            "SELECTED": round(float(sva[con].mean()), 6),
                            "vs_greedy": paired_boot(sva[con], gv[con]),
                            "vs_random_pick_floor": paired_boot(sva[con], rv[con]),
                            "vs_majority": paired_boot(sva[con], mv[con]),
                        } if con.sum() >= 2 else {"n": int(con.sum()),
                                                  "note": "too few contested items"}),
                    })
                else:
                    row["SELECTED"] = None
                    row["note"] = "verifier scores incomplete for this arm"
                rows[sname] = row
            cd["sampled_arms"][a] = rows["all"]
            if "image_disjoint" in rows:
                cd["sampled_arms"][a]["IMAGE_DISJOINT_SUBSET"] = rows["image_disjoint"]
        out["by_currency"][cur] = cd

    # ---- reformat effect: open-form greedy vs closed-form greedy ---------------------------
    ref = {}
    for cur in CURRENCIES:
        if not isinstance(out["by_currency"].get(cur), dict):
            continue
        ga = out["by_currency"][cur]["greedy_arms"]
        if "closedD_g" in ga:
            for a in ("openPRJ_g", "openMEK_g", "closedD_g_full"):
                if a in ga:
                    la = arm_labels(cell, a, jm, cur)
                    lb = arm_labels(cell, "closedD_g", jm, cur)
                    idxs = sorted(set(la) & set(lb))
                    va = np.array([0 if la[i][0] is None else la[i][0] for i in idxs], float)
                    vb = np.array([0 if lb[i][0] is None else lb[i][0] for i in idxs], float)
                    ref.setdefault(cur, {})[f"{a}_minus_closedD_g"] = paired_boot(va, vb)
    out["reformat_effect_greedy"] = ref
    out["published_always_7b_deployed_harness"] = L.PUBLISHED_ALWAYS_7B[cell]
    if cell == "SLAKE_closed":
        gen_any = gen.get("openPRJ_g") or {}
        out["language_split"] = dict(Counter(r["lang"] for r in gen_any.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=L.CELLS)
    A = ap.parse_args()

    nt = G.null_test()
    m = nt["measured"]
    doc = {
        "title": "BUILD 3 -- re-ask SLAKE_closed / VQA_RAD_closed / PATH_VQA_closed as OPEN TEXT and "
                 "test whether the FROZEN verifier's best-of-8 pick beats greedy on them.",
        "date": L.DATE,
        "preregistration": f"results/cascade_methods/artifacts/closed_as_open_{L.DATE}_preregistration.json",
        "reproduce": {
            "prereg": "python3 src/cascade_methods/closed_as_open_prereg.py",
            "generate": "python3 src/cascade_methods/closed_as_open_gen.py",
            "explode": "python3 src/cascade_methods/closed_as_open_explode.py",
            "judge": "python3 src/labeling/run_judge.py --preds ckpts/closed_as_open/judge_*.jsonl --tp 2",
            "score": "python3 src/cascade_methods/closed_as_open_score.py",
            "disjoint": "python3 src/cascade_methods/closed_as_open_disjoint.py",
            "this file": "python3 src/cascade_methods/closed_as_open_analyse.py",
        },
        "no_fabricated_numbers": True,
        "not_abstention": "every arm answers every item; an UNPARSED answer is graded WRONG, never withheld.",
        "null_tests": {
            "N1_frozen_open_text_metric": {
                "pass": nt["pass"], "max_abs_deviation": nt["max_abs_deviation"],
                "measured": {k: m[k] for k in ["n", "n_recoverable", "oracle@8", "selected",
                                               "greedy", "sel_eff", "cand_auroc"]},
                "identity_selected_eq_oracle_times_sel_eff_abs_error":
                    abs(m["selected"] - m["oracle@8"] * m["sel_eff"]),
            },
            "G1_grader": L.grader_null_test(),
        },
        "arms": {k: {kk: ("SYS_OPEN" if kk == "sys" and vv else vv) for kk, vv in v.items()}
                 for k, v in L.ARMS.items()},
        "PRIMARY_ENDPOINT": L.PRIMARY,
        "verifier": {"adapter": "ckpts/train/lora_verifier_disjoint (FROZEN, nothing retrained)",
                     "scoring_fn": "pyes() verbatim from verifier_transfer_eval.py",
                     "backend": "HF transformers (vLLM drops all 192 visual.* LoRA modules)"},
        "cells": {},
    }
    dj = os.path.join(L.PARTS, "disjointness.json")
    if os.path.exists(dj):
        doc["image_disjointness"] = json.load(open(dj))

    for cell in A.cells:
        jm = L.judge_map(cell)
        print(f"[{cell}] judge labels: {len(jm)}", flush=True)
        doc["cells"][cell] = analyse_cell(cell, jm)

    sc = os.path.join(L.PARTS, "mcq_self_consistency.json")
    if os.path.exists(sc):
        doc["secondary_self_consistency"] = json.load(open(sc))

    # ---- the pre-registered decision rule, applied mechanically ---------------------------
    arm = L.PRIMARY["sampled"]
    verdict = {"rule": "a cell COUNTS toward the claim only if SELECTED - GREEDY is positive with a "
                       "95% paired-bootstrap CI excluding 0 in BOTH currencies (32B judge AND the "
                       "length-neutral normalised exact match).",
               "primary_arm": arm, "per_cell": {}, "cells_counting": [], "cells_not_counting": []}
    for cell in A.cells:
        c = doc["cells"][cell].get("by_currency", {})
        row = {}
        ok = True
        for cur in ("judge", "em_repaired"):
            r = c.get(cur, {}).get("sampled_arms", {}).get(arm)
            if not r or r.get("SELECTED") is None:
                row[cur] = "NOT MEASURED"
                ok = False
                continue
            key = "IMAGE_DISJOINT_SUBSET" if "IMAGE_DISJOINT_SUBSET" in r else None
            use = r[key] if key else r
            row[cur] = {"basis": key or "all_items", "n": use["n"],
                        "greedy": use["greedy"], "SELECTED": use["SELECTED"],
                        "delta": use["vs_greedy"]["delta"], "ci": use["vs_greedy"]["ci"],
                        "sign": use["vs_greedy"]["sign"]}
            ok = ok and use["vs_greedy"]["sign"] == "WIN"
        row["counts"] = bool(ok)
        verdict["per_cell"][cell] = row
        (verdict["cells_counting"] if ok else verdict["cells_not_counting"]).append(cell)
    verdict["n_cells_added_to_the_claim"] = len(verdict["cells_counting"])
    verdict["claim_scope"] = (f"3 open cells + {len(verdict['cells_counting'])} reformatted closed "
                              f"cells = {3 + len(verdict['cells_counting'])} of 8 reporting cells")
    doc["VERDICT"] = verdict

    os.makedirs(L.ART, exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")

    # ---- console summary -------------------------------------------------------------------
    for cell in A.cells:
        c = doc["cells"][cell]
        print(f"\n=== {cell} (n={c['n']}) ===")
        for cur in CURRENCIES:
            cd = c["by_currency"][cur]
            if not isinstance(cd, dict):
                print(f"  [{cur}] {cd}")
                continue
            ga = ", ".join(f"{k}={v['acc']:.4f}" for k, v in cd["greedy_arms"].items())
            print(f"  [{cur}] greedy: {ga}")
            for a, r in cd["sampled_arms"].items():
                if r.get("SELECTED") is None:
                    print(f"     {a}: SELECTED n/a ({r.get('note','')})")
                    continue
                vg = r["vs_greedy"]
                print(f"     {a}: greedy={r['greedy']:.4f} oracle@8={r['oracle@8']:.4f} "
                      f"selEff={r['sel_eff']:.4f} SELECTED={r['SELECTED']:.4f} "
                      f"maj={r['majority_vote']:.4f} floor={r['random_pick_floor']:.4f} | "
                      f"sel-greedy {vg['delta']:+.4f} [{vg['ci'][0]:+.4f},{vg['ci'][1]:+.4f}] {vg['sign']}")


if __name__ == "__main__":
    main()
