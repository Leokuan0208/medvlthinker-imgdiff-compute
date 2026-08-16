#!/usr/bin/env python3
"""hole17_seeds.py -- fold-assignment stability of the macro refit (10 seeds) and the knock-on
effect on the SHIPPED accuracy-max arm.

WHY 10 SEEDS.  Nothing here is trained by SGD, so the only stochastic input is the cross-fit fold
assignment.  The repo's canonical assignment is i % 5 (seed None); 10 random assignments are run so
the refit delta is reported as mean / sd / range rather than a single fold draw.

WHY THE ACCURACY-MAX ARM MOVES ON COST ONLY.  In method_final_mmmu_corrected.add_v2_vectors the
accuracy-max arm is always-32B-direct on 4 MCQ cells, the F8 certified veto on PMC, and the F10 L2D
rejector over a FIXED best-of-8 pick on the 3 open cells -- so tau and lambda enter its ACCURACY
nowhere.  lambda enters its COST, because am2_cost = cost_pandora(meanN, esc_F10) reuses the
compute-lean Pandora draw count.  That knock-on is computed here exactly, not estimated.

  nohup env OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_seeds.py \
        > logs/hole17/seeds.log 2>&1 &
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import hole17_run as R
import hole17_data as HD

ANCHORS = [("macro_iso_acc", "macro", "iso_acc"), ("macro_iso_cost", "macro", "iso_cost"),
           ("pooled_iso_acc", "pooled", "iso_acc"), ("pooled_iso_cost", "pooled", "iso_cost")]
OUT = os.path.join(R.ART, "_hole17_seeds.jsonl")


def run_seed(D, seed):
    fa = {k: R.folds(D.n[k], R.K_OUTER, seed) for k in R.ORDER_B}
    res = R.heldout_pass(D, fa)
    inc = R.summarise(res["incumbent"], D.n)
    nes = R.nested_pass(D, fa, ANCHORS, inner_seed=(seed if seed is not None else 0) + 11)
    rec = dict(seed=("canonical_modulo" if seed is None else seed),
               incumbent={q: inc[q] for q in ("macro_acc", "macro_cost", "pooled_acc", "pooled_cost")},
               incumbent_meanN={k: float(res["incumbent"][k]["N"].mean()) for k in R.OPEN_B})
    rec["arms"] = {}
    for lab, _, _ in ANCHORS:
        s = R.summarise(nes[lab], D.n)
        rec["arms"][lab] = dict(
            macro_acc=s["macro_acc"], macro_cost=s["macro_cost"],
            pooled_acc=s["pooled_acc"], pooled_cost=s["pooled_cost"],
            d_macro_acc=s["macro_acc"] - inc["macro_acc"],
            d_macro_cost=s["macro_cost"] - inc["macro_cost"],
            meanN={k: float(nes[lab][k]["N"].mean()) for k in R.OPEN_B},
            per_cell_esc=s["per_cell_esc"])
    return rec


def accuracy_max_knockon(meanN_inc, meanN_new):
    """am2_cost = meanN * 2.0 + esc_F10 * 4.57 on the 3 open cells; the 5 MCQ cells are unchanged.
    esc_F10 and every accuracy are read from the published artifact -- they do not depend on lambda."""
    S = json.load(open(os.path.join(R.ART, "_selector_rerun_parts", "summary_disjoint.json")))
    det = S["open_cell_detail"]
    per_cell = {}
    for k in R.OPEN_B:
        esc10 = det[k]["am2_esc"]
        per_cell[k] = dict(esc_F10=esc10,
                           cost_published=det[k]["cost_am2"]["flops"],
                           cost_recomputed_inc=meanN_inc[k] * HD.C_CHEAP + esc10 * HD.C_STRONG,
                           cost_refit=meanN_new[k] * HD.C_CHEAP + esc10 * HD.C_STRONG,
                           meanN_inc=meanN_inc[k], meanN_refit=meanN_new[k])
    mcq_cost = {"PMC_VQA": None}
    # the 5 MCQ cells' am2 costs, verbatim from the published artifact
    macro_pub = S["cost_macro"]["method_accuracy_max_veto"]["flops"]
    delta_macro = sum((per_cell[k]["cost_refit"] - per_cell[k]["cost_recomputed_inc"]) for k in R.OPEN_B) / 8.0
    return dict(per_cell=per_cell, published_macro_cost=macro_pub,
                macro_cost_after_refit=macro_pub + delta_macro,
                delta_macro_cost=delta_macro,
                x_direct_before=macro_pub / R.DIRECT_FLOPS,
                x_direct_after=(macro_pub + delta_macro) / R.DIRECT_FLOPS,
                accuracy_unchanged_reason="accuracy-max is always-32B-direct on 4 MCQ cells, F8 on PMC "
                                          "and F10 over a FIXED best-of-8 pick on the 3 open cells; "
                                          "tau and lambda enter none of those accuracy vectors.")


if __name__ == "__main__":
    D = R.Data()
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["seed"])
            except Exception: pass
    with open(OUT, "a") as fh:
        for s in [None] + list(range(10)):
            key = "canonical_modulo" if s is None else s
            if key in done: continue
            t = time.time()
            try:
                rec = run_seed(D, s)
            except Exception:
                import traceback; rec = dict(seed=key, error=traceback.format_exc()[-800:])
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print("seed %s  %.1fs" % (key, time.time() - t), flush=True)
    # knock-on on the shipped accuracy-max arm, canonical fold assignment
    recs = [json.loads(l) for l in open(OUT)]
    can = [r for r in recs if r.get("seed") == "canonical_modulo"][0]
    ko = {lab: accuracy_max_knockon(can["incumbent_meanN"], can["arms"][lab]["meanN"])
          for lab in can["arms"]}
    json.dump(ko, open(os.path.join(R.ART, "_hole17_amax_knockon.json"), "w"), indent=1, default=float)
    print(json.dumps({l: dict(dmacro=ko[l]["delta_macro_cost"], xb=ko[l]["x_direct_before"],
                              xa=ko[l]["x_direct_after"]) for l in ko}, indent=1))
