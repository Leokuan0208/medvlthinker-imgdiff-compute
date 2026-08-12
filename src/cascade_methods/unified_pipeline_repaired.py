#!/usr/bin/env python3
"""unified_pipeline_repaired.py -- ATTACK 2: re-run every option-branch comparison against a
REPAIRED-GRADER baseline, because the deployed PMC-VQA grader is defective and the defect is exactly
the size of this round's only positive result.

WHAT HAPPENED.  The option branch delivers a candidate INDEX, so it is graded `pick == gold`.  The
baselines come from the MedEvalKit dumps' `correct` field.  On PMC-VQA those two graders DISAGREE on
69 of the 6,000 scored items (65 the harness calls wrong that are right, 4 the other way): the model
answers "C:" and MedEvalKit/utils/utils.py:111-112 reduces that to the empty string
(`response.split('.')[0].split(':')[-1]`), then falls through to difflib similarity against the four
option TEXTS and picks something else.  Independently found and fully characterised the same day by
results/cascade_methods/artifacts/pmcvqa_grader_defect_2026-08-12.json (upstream defect, MedEvalKit
NOT modified).

So an arm that is graded `pick == gold` is being compared against a baseline graded by a broken
extractor, and it collects +0.0102 on PMC for free.  This script re-grades BOTH legs with ONE rule --
the first letter character of the response, required to be in range; yes/no by prefix -- so the
comparison is paired and self-consistent, and reports every option-branch delta under BOTH graders.

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline_repaired.py --tag zeroshot
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

LETTER_RE = re.compile(r"([A-Za-z])")
DUMPS = {"always_7b": "eval_results_lingshu7b_full", "always_32b_direct": "eval_results_lingshu32b_full"}


def parse_pick(resp, k):
    t = str(resp).strip()
    if k == 2:                                   # yes/no cells
        s = re.sub(r"[^a-z]", "", t.lower())
        return 0 if s.startswith("yes") else (1 if s.startswith("no") else None)
    m = LETTER_RE.search(t)
    if not m:
        return None
    p = ord(m.group(1).upper()) - 65
    return p if 0 <= p < k else None


def repaired_vectors(cell, rows):
    out = {}
    for sys_name, d in DUMPS.items():
        raw = json.load(open(f"{U.MEK}/{d}/{{}}/{U._DSNAME[cell]}/results.json"))
        v = []
        for r in rows:
            p = parse_pick(raw[r["src"]].get("response", ""), len(r["cands"]))
            v.append(0.0 if p is None else float(int(p == r["gold"])))
        out[sys_name] = np.array(v, float)
    return out


def run(tag):
    z = np.load(U.VEC_NPZ, allow_pickle=True)
    work = U.build_worklist()
    out = {"tag": tag, "grader_rule": "first letter character of the response, required in range; "
                                      "yes/no by prefix. Unparseable -> scored wrong (never dropped).",
           "cross_check": "results/cascade_methods/artifacts/pmcvqa_grader_defect_2026-08-12.json "
                          "reports the same defect on the FULL PMC cell: always_7b 0.542656 -> "
                          "0.552647 (+351/-17) and always_32b_direct 0.551780 -> 0.569339 (+622/-35), "
                          "a 0.7568 pp DIFFERENTIAL in the 32B's favour.",
           "cells": {}}
    for cell in U.OPTION_CELLS:
        rows = work[cell]
        sc = U.load_scores(cell, tag)
        keep = [j for j in range(len(rows))
                if sc.get(rows[j]["i"]) is not None
                and len(sc[rows[j]["i"]]) == len(rows[j]["cands"])]
        if not keep:
            out["cells"][cell] = {"status": "not measured"}
            continue
        krows = [rows[j] for j in keep]
        rep = repaired_vectors(cell, krows)
        idx = [r["i"] for r in krows]
        h7 = np.array([z[f"{cell}|always_7b"][i] for i in idx], float)
        h32 = np.array([z[f"{cell}|always_32b_direct"][i] for i in idx], float)
        okv = np.array([int(int(np.argmax([sc[r["i"]][c] for c in range(len(r["cands"]))]))
                            == r["gold"]) for r in krows], float)
        raw7 = json.load(open(f"{U.MEK}/{DUMPS['always_7b']}/{{}}/{U._DSNAME[cell]}/results.json"))
        p7 = [parse_pick(raw7[r["src"]].get("response", ""), len(r["cands"])) for r in krows]
        fus = {}
        for lam in (0.1, 1.0):
            v = []
            for a, r in enumerate(krows):
                s = np.array([sc[r["i"]][c] for c in range(len(r["cands"]))], float)
                if p7[a] is not None:
                    s[p7[a]] += lam
                v.append(float(int(np.argmax(s)) == r["gold"]))
            fus[lam] = np.array(v, float)
        out["cells"][cell] = {
            "n": len(krows),
            "always_7b_harness_grader": float(h7.mean()),
            "always_7b_repaired_grader": float(rep["always_7b"].mean()),
            "always_32b_direct_harness_grader": float(h32.mean()),
            "always_32b_direct_repaired_grader": float(rep["always_32b_direct"].mean()),
            "grader_disagreement_items_7b": int((rep["always_7b"] != h7).sum()),
            "grader_disagreement_items_32b": int((rep["always_32b_direct"] != h32).sum()),
            "armA_verifier_over_options": float(okv.mean()),
            "armA_vs_7b_HARNESS_grader": U.paired_boot(okv, h7),
            "armA_vs_7b_REPAIRED_grader": U.paired_boot(okv, rep["always_7b"]),
            "armA_vs_32b_REPAIRED_grader": U.paired_boot(okv, rep["always_32b_direct"]),
            "fusion_lambda_0p1": float(fus[0.1].mean()),
            "fusion_lambda_0p1_vs_7b_HARNESS": U.paired_boot(fus[0.1], h7),
            "fusion_lambda_0p1_vs_7b_REPAIRED": U.paired_boot(fus[0.1], rep["always_7b"]),
            "fusion_lambda_1p0_equals_always_7b_repaired": float(fus[1.0].mean()),
            "fusion_lambda_1p0_vs_7b_REPAIRED": U.paired_boot(fus[1.0], rep["always_7b"]),
        }
        out["cells"][cell]["_vectors_for_macro"] = {
            "armA": okv, "fus01": fus[0.1], "fus10": fus[1.0],
            "rep7": rep["always_7b"], "rep32": rep["always_32b_direct"],
            "h7": h7, "h32": h32}
    cells = [c for c in U.OPTION_CELLS if "_vectors_for_macro" in out["cells"].get(c, {})]
    if cells:
        V = {c: out["cells"][c].pop("_vectors_for_macro") for c in cells}
        m = lambda k: float(np.mean([V[c][k].mean() for c in cells]))  # noqa: E731
        out["four_cell_macro_option_branch_only"] = {
            "cells": cells,
            "always_7b_harness": m("h7"), "always_7b_REPAIRED": m("rep7"),
            "always_32b_direct_harness": m("h32"), "always_32b_direct_REPAIRED": m("rep32"),
            "armA_verifier_over_options": m("armA"),
            "fusion_lambda_0p1": m("fus01"),
            "fusion_lambda_1p0_GLOBAL_CROSSFIT_CHOICE": m("fus10"),
            "fusion_lambda_1p0_vs_7b_REPAIRED": U.macro_boot(
                {c: V[c]["fus10"] for c in cells}, {c: V[c]["rep7"] for c in cells}),
            "fusion_lambda_0p1_vs_7b_REPAIRED": U.macro_boot(
                {c: V[c]["fus01"] for c in cells}, {c: V[c]["rep7"] for c in cells}),
            "fusion_lambda_1p0_vs_7b_HARNESS_the_INFLATED_comparison": U.macro_boot(
                {c: V[c]["fus10"] for c in cells}, {c: V[c]["h7"] for c in cells}),
            "reading": "the cross-fit GLOBAL lambda is 1.0, and at lambda >= 1.0 the generator's own "
                       "answer can never be overturned (verifier scores live in [0,1]), so the "
                       "fusion has DEGENERATED to always-7B. Against the harness grader that "
                       "degenerate arm still looks like a win; against a repaired grader it is not."}
    else:
        for c in out["cells"]:
            out["cells"][c].pop("_vectors_for_macro", None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zeroshot")
    a = ap.parse_args()
    r = run(a.tag)
    p = os.path.join(U.PARTS, f"repaired_grader_{a.tag}.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print("wrote", p)
