#!/usr/bin/env python3
"""closed_as_open_skill.py -- BUILD 3: the metric that makes the reformatted CLOSED cells comparable
to the three OPEN cells the frozen verifier was actually built for.

WHY sel_eff IS NOT THAT METRIC.  On a yes/no cell the 8-sample pool is near-unanimous (mean 1.17-1.24
distinct strings), so a RANDOM pick already scores ~0.93 of oracle.  A sel_eff of 0.93 on such a cell
therefore means NO SKILL AT ALL, while the same 0.93 on the open cells would be extraordinary.  This
project has already retracted a claim for mistaking coverage for signal; reporting sel_eff across
these two regimes without renormalising would repeat it.

    SKILL = (SELECTED - random_pick_floor) / (oracle@8 - random_pick_floor)

0 = the verifier is worth exactly a coin flip among the candidates; 1 = oracle.  The denominator is
the only headroom a selector can possibly convert, because coverage is fixed by the generator.

Both currencies (32B judge, length-neutral EM) on IDENTICAL picks.  The image-disjoint subset is used
wherever the cell shares images with the verifier's training pool (SLAKE_closed, VQA_RAD_closed).

CPU only.  python3 src/cascade_methods/closed_as_open_skill.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402
from closed_as_open_analyse import contaminated, paired_boot, picks_from  # noqa: E402

sys.path.insert(0, os.path.join(L.ROOT, "src"))
from training_methods import genframe_data as G                           # noqa: E402

METRIC = ("SKILL = (SELECTED - random_pick_floor) / (oracle@8 - random_pick_floor): the fraction of "
          "the headroom ABOVE A RANDOM PICK that the verifier converts. 0 = no skill, 1 = oracle. "
          "sel_eff is NOT comparable across these cells: a near-unanimous pool makes a RANDOM pick "
          "score near oracle by construction, so a high sel_eff on a yes/no cell is COVERAGE, not "
          "signal.")


def open_cell_reference():
    """The three OPEN cells, frozen deployed pools -- the regime where the verifier demonstrably works."""
    items = G.load_items()
    ok = np.array([it["sl"] for it in items], dtype=float)
    picks = np.argmax(G._slot_scores(G.incumbent_scores(), items), axis=1)
    sel = ok[np.arange(len(ok)), picks]
    oracle, floor = ok.max(1), ok.mean(1)
    greedy = np.array([it["greedy_ok"] for it in items], dtype=float)
    con = np.array([len(set(G.norm(a) for a in it["preds"])) >= 2 for it in items])
    skill = (sel.mean() - floor.mean()) / (oracle.mean() - floor.mean())
    skill_c = (sel[con].mean() - floor[con].mean()) / (oracle[con].mean() - floor[con].mean())
    return {"n": int(len(ok)), "greedy": round(float(greedy.mean()), 6),
            "random_pick_floor": round(float(floor.mean()), 6),
            "SELECTED": round(float(sel.mean()), 6), "oracle@8": round(float(oracle.mean()), 6),
            "skill": round(float(skill), 6), "contested_frac": round(float(con.mean()), 6),
            "contested_skill": round(float(skill_c), 6),
            "SELECTED_vs_random_pick_floor": paired_boot(sel, floor),
            "source": "src/training_methods/genframe_data.py load_items()+incumbent_scores() (FROZEN)"}


def closed_cell(cell, arm=None, greedy_arm=None):
    arm = arm or L.PRIMARY["sampled"]
    greedy_arm = greedy_arm or L.PRIMARY["greedy"]
    bad = contaminated(cell) or set()
    jm = L.judge_map(cell)
    gen, gg, sc = L.load_gen(cell, arm), L.load_gen(cell, greedy_arm), L.load_scores(cell, arm)
    idxs = sorted(i for i in gen if i in gg and i in sc and i not in bad)
    row = {"basis": "image_disjoint_subset" if bad else "all_items", "arm": arm,
           "greedy_arm": greedy_arm}
    for cur in ("judge", "em_repaired"):
        def lab(i, p):
            if cur == "judge":
                v = jm.get((i, L.norm_text(p)))
                return 0 if v is None else int(v)
            return L.em_repaired(cell, gen[i]["gold"], p)[0]
        O = np.array([[lab(i, p) for p in gen[i]["preds"]] for i in idxs], dtype=float)
        gv = np.array([lab(i, gg[i]["preds"][0]) for i in idxs], dtype=float)
        sel = O[np.arange(len(idxs)), [picks_from(sc[i]) for i in idxs]]
        oracle, floor = O.max(1), O.mean(1)
        con = np.array([len(set(L.norm_text(p) for p in gen[i]["preds"])) >= 2 for i in idxs])
        skill = float((sel.mean() - floor.mean()) / (oracle.mean() - floor.mean()))
        skill_c = float((sel[con].mean() - floor[con].mean()) /
                        (oracle[con].mean() - floor[con].mean())) if con.sum() else float("nan")
        row[cur] = {"n": len(idxs), "greedy": round(float(gv.mean()), 6),
                    "random_pick_floor": round(float(floor.mean()), 6),
                    "SELECTED": round(float(sel.mean()), 6),
                    "oracle@8": round(float(oracle.mean()), 6), "skill": round(skill, 6),
                    "contested_frac": round(float(con.mean()), 6),
                    "contested_skill": round(skill_c, 6),
                    "SELECTED_vs_random_pick_floor": paired_boot(sel, floor)}
    return row


def pool_degeneracy():
    """WHY the skill is zero: the reformat never created the regime it was meant to create.

    The hypothesis under test was CANDIDATE PROVENANCE -- sampled candidates should be the regime the
    verifier works in.  Every sampled arm here IS sampled from the generator.  What the reformat was
    supposed to change, and did not, is the CANDIDATE SET: removing the answer space from the prompt
    leaves the pool exactly as narrow, because a yes/no question has a binary answer whether or not
    the prompt says so.  Diversity, not provenance, is what distinguishes the two regimes.
    """
    out = {"note": "distinct normalised answer strings produced across the WHOLE cell (all items, all "
                   "8 slots), and the fraction of items whose 8-sample pool holds >= 2 distinct "
                   "strings. The open cells the verifier was built for are contested on 73.6% of "
                   "items; these cells are contested on 16-24% and emit a handful of strings in total.",
           "open_cell_reference_contested_frac": None, "cells": {}}
    items = G.load_items()
    out["open_cell_reference_contested_frac"] = round(float(np.mean(
        [len(set(G.norm(a) for a in it["preds"])) >= 2 for it in items])), 6)
    out["open_cell_reference_distinct_strings_cellwide"] = int(len(
        {G.norm(a) for it in items for a in it["preds"]}))
    for cell in L.CELLS:
        row = {}
        for arm in ("closedD_s8", "openMEK_s8", "openPRJ_s8"):
            gen = L.load_gen(cell, arm)
            allp = [L.norm_text(p) for r in gen.values() for p in r["preds"]]
            nd = [len({L.norm_text(p) for p in r["preds"]}) for r in gen.values()]
            row[arm] = {"n_slots": len(allp),
                        "distinct_strings_cellwide": int(len(set(allp))),
                        "mean_distinct_per_item": round(float(np.mean(nd)), 4),
                        "contested_frac": round(float(np.mean([x >= 2 for x in nd])), 6),
                        "prompt_gives_answer_space": L.has_answer_space(L.ARMS[arm]["prompt"], cell)}
        out["cells"][cell] = row
    return out


def main():
    out = {"metric": METRIC,
           "basis": "primary arm openPRJ_s8; image-disjoint subset wherever the cell shares images "
                    "with the verifier's training pool",
           "open_cell_reference": open_cell_reference(),
           "POOL_DEGENERACY_why_the_skill_is_zero": pool_degeneracy(),
           "closed_cells": {}}
    ref = out["open_cell_reference"]
    print(f"OPEN-CELL REFERENCE n={ref['n']} floor={ref['random_pick_floor']:.4f} "
          f"SELECTED={ref['SELECTED']:.4f} oracle={ref['oracle@8']:.4f} SKILL={ref['skill']:+.4f}")
    for cell in L.CELLS:
        out["closed_cells"][cell] = closed_cell(cell)
        for cur in ("judge", "em_repaired"):
            r = out["closed_cells"][cell][cur]
            pb = r["SELECTED_vs_random_pick_floor"]
            print(f"  {cell:16s} [{cur:11s}] n={r['n']:4d} floor={r['random_pick_floor']:.4f} "
                  f"SEL={r['SELECTED']:.4f} oracle={r['oracle@8']:.4f} SKILL={r['skill']:+.4f} | "
                  f"SEL-floor {pb['delta']:+.4f} [{pb['ci'][0]:+.4f},{pb['ci'][1]:+.4f}] {pb['sign']}")
    os.makedirs(L.PARTS, exist_ok=True)
    p = os.path.join(L.PARTS, "skill_vs_floor.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
