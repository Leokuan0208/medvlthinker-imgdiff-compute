#!/usr/bin/env python3
"""integrate_prereg2.py -- supplementary TRAIN-ONLY pre-registration, run BEFORE the eval
stage, reusing the cross-fitted train scores integrate_prereg.py already wrote (no refits).

  Q4  should the free self-consistency count (how many of the 8 samples produced this
      answer) be a MEMBER of the deployed rank fusion?  It costs nothing, so if train CV
      says yes it belongs in the headline rather than in the diagnostics.
  Q5  would adding a GRADER-frame head as a second member help?  It is not free
      (+1 forward pass per distinct candidate), so this only decides whether an
      "accuracy-max at extra cost" variant is worth measuring at all.

  python3 src/training_methods/integrate_prereg2.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G          # noqa: E402
import integrate_lib as IL         # noqa: E402
import cheapcontrast as CC         # noqa: E402
from integrate_prereg import ens_train, pool_rank, CFG, GRADER_CFG  # noqa: E402


def main():
    pre = json.load(open(os.path.join(IL.PARTS, "prereg.json")))
    conv = pre["PREREGISTERED_DECISION"]["ensembling_convention"]
    k = int(pre["PREREGISTERED_DECISION"]["n_seeds"])
    Sg = np.load(os.path.join(IL.PARTS, "prereg_cv_gen.npy"))
    Sr = np.load(os.path.join(IL.PARTS, "prereg_cv_grader.npy"))

    tr = G.load_candidates("train", "generator", layers=[], pooling=())
    qidx = IL.train_question_index(tr)
    mult = CC.pool_multiplicity(sorted({r["ds"] for r in tr.rows}))
    m = np.array([mult[(r["ds"], r["idx"], r["na"])] for r in tr.rows], float)

    a = ens_train(Sg[:k], qidx, conv)
    b = ens_train(Sr[:Sr.shape[0]], qidx, conv)
    ra, rb, rm = pool_rank(a, qidx), pool_rank(b, qidx), pool_rank(m, qidx)

    S = IL.train_sel_eff
    res = {
        "what": "supplementary train-only pre-registration (Q4 self-consistency member, "
                "Q5 grader-frame member). No eval label is read.",
        "readout_used": {"convention": conv, "k": k},
        "Q4_self_consistency_member": {
            "head_ens_alone": S(ra, qidx),
            "self_consistency_alone": S(rm, qidx),
            "head_ens + self_consistency (rank_avg)": S(ra + rm, qidx),
        },
        "Q5_grader_frame_member": {
            "head_ens_alone": S(ra, qidx),
            "grader_head_alone": S(rb, qidx),
            "head_ens + grader_head (rank_avg)": S(ra + rb, qidx),
            "cost_note": "a grader-frame member needs +1 VLM forward pass per DISTINCT "
                         "candidate (mean 3.81/question); it is NOT free like Q4.",
        },
    }
    q4 = res["Q4_self_consistency_member"]
    res["Q4_DECISION"] = ("include" if q4["head_ens + self_consistency (rank_avg)"] >
                          q4["head_ens_alone"] else "exclude")
    q5 = res["Q5_grader_frame_member"]
    res["Q5_DECISION"] = ("worth measuring as a paid variant"
                          if q5["head_ens + grader_head (rank_avg)"] > q5["head_ens_alone"]
                          else "not worth its cost")
    IL.jdump(res, os.path.join(IL.PARTS, "prereg2.json"))
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
