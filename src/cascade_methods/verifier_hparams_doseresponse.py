#!/usr/bin/env python3
"""verifier_hparams_doseresponse.py -- KNOB 3: why the pooled ladder LOOKS like a step, and
what the resolution dose-response actually is once the confound is removed.

THE CONFOUND IN THE POOLED LADDER
---------------------------------
`max_pixels` is a CAP, not a resolution.  Qwen's smart_resize only ever SHRINKS, so an image
whose native size already sits below the cap is rendered byte-identically at that rung and at
the deployed 1,003,520.  The fraction of items the cap actually BINDS on therefore changes
from rung to rung -- 99.7% at 62,720 down to 15.2% at 501,760 -- and the pooled per-rung delta
is (per-affected-item damage) x (fraction affected).  A pooled curve that looks like a sharp
STEP between 376,320 and 501,760 is then ambiguous: it could be a threshold in the model, or
it could be the cap simply ceasing to bind on most of the pool.

THE FIX
-------
Restrict to a FIXED item set -- the items whose rendering changes even at the highest
sub-deployed rung, i.e. the largest images -- and read every rung on that same set.  Those
items are re-rendered at every rung, so the comparison is a genuine dose-response in vision
tokens with the binding fraction held at 1.0 throughout.

THE FREE PLACEBO THAT VALIDATES THE SPLIT
-----------------------------------------
The complement (items rendered identically) is a placebo measured inside the same comparison:
the verifier input is byte-identical there, so its delta must be exactly 0.  It is -- 0 score
differences over every identical-rendering slot at every rung -- which is what licenses
attributing the whole pooled effect to the changed-rendering stratum.

CPU only.   python3 src/cascade_methods/verifier_hparams_doseresponse.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
from src.training_methods import genframe_data as G          # noqa: E402
from verifier_hparams_analyze import load_arm, slot_scores    # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
CTRL = 1003520
RUNGS = [62720, 125440, 250880, 376320, 501760, 1003520, 12845056]
NBOOT = 10000
BSEED = 20260815
#: the fixed stratum is defined by the highest SUB-deployed rung, so every rung re-renders it.
STRATUM_DEF_PX = 501760


def main():
    items = G.load_items()
    d = np.load(os.path.join(PARTS, "em_slots.npz"))
    em, judge = d["em"], d["judge"]

    A, V = {}, {}
    for px in RUNGS:
        a = load_arm(px, "")
        if a is None or a["n_scored"] < 8965:
            print(f"px{px} incomplete -- abort")
            return
        A[px] = a
        sq, _ = slot_scores(a["scores"], items)
        r = G.sel_eff(sq, items)
        picks = r["picks"]
        V[px] = {"got_j": np.asarray(r["got"], int),
                 "got_e": np.array([em[i, picks[i]] for i in range(len(items))], int),
                 "rec": np.asarray(r["rec"], int)}

    key = [(it["ds"], it["idx"]) for it in items]
    pc = A[CTRL]["patch"]

    out = {"_what": "removes the BINDING-FRACTION confound from the pooled resolution ladder.",
           "_code": "src/cascade_methods/verifier_hparams_doseresponse.py",
           "_stratum_definition": f"items whose rendering differs from the deployed {CTRL} even "
                                  f"at {STRATUM_DEF_PX} -- i.e. the images larger than "
                                  f"{STRATUM_DEF_PX} px, which every rung re-renders."}

    # ---- 1. how often does the cap actually bind? ------------------------------------
    bind = {}
    for px in RUNGS:
        if px == CTRL:
            continue
        ch = np.array([1 if A[px]["patch"][k] != pc[k] else 0 for k in key], int)
        bind[str(px)] = {"n_items_rendering_changed_vs_deployed": int(ch.sum()),
                         "frac_binding": float(ch.mean()),
                         "mean_vision_tokens": A[px]["geometry"]["mean_vision_tokens"]}
    out["1_the_cap_binds_on_a_different_fraction_at_every_rung"] = {
        "by_max_pixels": bind,
        "_read": "the pooled per-rung delta is (damage per affected item) x (fraction affected). "
                 "Between 376,320 and 501,760 the binding fraction collapses from 69.8% to 15.2%, "
                 "which is on its own enough to produce an apparent STEP with no threshold in the "
                 "model at all.",
        "deployed_mean_vision_tokens": A[CTRL]["geometry"]["mean_vision_tokens"],
    }

    # ---- 2. the fixed-stratum dose-response ------------------------------------------
    ch = np.array([1 if A[STRATUM_DEF_PX]["patch"][k] != pc[k] else 0 for k in key], int)
    rec = V[CTRL]["rec"] == 1
    sub = (ch == 1) & rec
    idx = np.where(sub)[0]
    m = len(idx)
    rng = np.random.default_rng(BSEED)
    rows = {}
    for px in RUNGS:
        vt = float(np.mean([A[px]["patch"][k] for i, k in enumerate(key) if sub[i]]) / 4.0)
        if px == CTRL:
            rows[str(px)] = {"mean_vision_tokens_on_stratum": vt, "is_control": True,
                             "is_trained_resolution": True,
                             "d_sel_eff_judge": 0.0, "d_sel_eff_em": 0.0}
            continue
        r = {"mean_vision_tokens_on_stratum": vt, "is_control": False,
             "is_trained_resolution": False}
        for cur, gk in (("judge", "got_j"), ("em", "got_e")):
            a, b = V[px][gk], V[CTRL][gk]
            obs = float(a[idx].mean() - b[idx].mean())
            bs = np.empty(NBOOT)
            for k2 in range(NBOOT):
                j = idx[rng.integers(0, m, m)]
                bs[k2] = a[j].mean() - b[j].mean()
            r[f"d_sel_eff_{cur}"] = obs
            r[f"d_sel_eff_{cur}_ci"] = [float(np.percentile(bs, 2.5)),
                                        float(np.percentile(bs, 97.5))]
            r[f"d_sel_eff_{cur}_excludes_0"] = bool(np.percentile(bs, 2.5) > 0
                                                    or np.percentile(bs, 97.5) < 0)
        rows[str(px)] = r
    out["2_fixed_stratum_dose_response"] = {
        "n_items_in_stratum": int(ch.sum()),
        "n_recoverable_in_stratum": int(m),
        "nboot": NBOOT, "seed": BSEED,
        "by_max_pixels": rows,
        "_read": "on ONE fixed set of items, re-rendered at every rung, the damage is MONOTONE "
                 "and GRADED in vision tokens and saturates before the trained resolution is "
                 "reached. It is a dose-response, not a step, and the trained rung is not a peak.",
    }

    # ---- 3. the placebo that licenses the split --------------------------------------
    plac = {}
    for px in RUNGS:
        if px == CTRL:
            continue
        same = np.array([1 if A[px]["patch"][k] == pc[k] else 0 for k in key], int)
        nslot = int(sum(len(it["preds"]) for i, it in enumerate(items) if same[i] == 1))
        ndiff = int(sum(1 for i, it in enumerate(items) if same[i] == 1 for a_ in it["preds"]
                        if abs(A[px]["scores"].get((it["ds"], it["idx"], a_), 0.0) -
                               A[CTRL]["scores"].get((it["ds"], it["idx"], a_), 0.0)) > 0.0))
        mm = (same == 1) & rec
        plac[str(px)] = {"n_slots_identical_rendering": nslot,
                         "n_score_differences": ndiff,
                         "n_recoverable": int(mm.sum()),
                         "d_sel_eff_judge": (float(V[px]["got_j"][mm].mean() -
                                                   V[CTRL]["got_j"][mm].mean())
                                             if mm.sum() else None)}
    out["3_identical_rendering_placebo"] = {
        "by_max_pixels": plac,
        "_read": "EXACT zeros everywhere: an identically-rendered image gives a bit-identical "
                 "score. The numerical noise floor of this whole round is therefore 0, and the "
                 "entire pooled effect lives in the changed-rendering stratum.",
        "_note": "the comparison is exact equality of the stored float, not a tolerance.",
    }

    json.dump(out, open(os.path.join(PARTS, "doseresponse.json"), "w"), indent=1, default=float)
    print(f"stratum: {int(ch.sum())} items, {m} recoverable")
    for px in RUNGS:
        r = rows[str(px)]
        if r.get("is_control"):
            print(f"  px{px:>9}  vis_tok {r['mean_vision_tokens_on_stratum']:7.1f}   CONTROL "
                  f"(= trained resolution)")
        else:
            print(f"  px{px:>9}  vis_tok {r['mean_vision_tokens_on_stratum']:7.1f}   "
                  f"judge {r['d_sel_eff_judge']:+.6f} "
                  f"[{r['d_sel_eff_judge_ci'][0]:+.6f},{r['d_sel_eff_judge_ci'][1]:+.6f}]   "
                  f"EM {r['d_sel_eff_em']:+.6f} "
                  f"[{r['d_sel_eff_em_ci'][0]:+.6f},{r['d_sel_eff_em_ci'][1]:+.6f}]")
    print(f"\nwrote {PARTS}/doseresponse.json")


if __name__ == "__main__":
    main()
