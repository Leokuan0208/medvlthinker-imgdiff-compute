#!/usr/bin/env python3
"""verifier_hparams_guardrail.py -- KNOB 3: per-set GUARDRAIL confidence intervals for every rung
of the verifier scoring-resolution ladder, in BOTH currencies, against the in-session control.

THE GUARDRAIL is this project's standing rule: a change must never be worse on any single
benchmark, because pooled wins routinely hide per-benchmark damage.  A bare per-set delta is not
enough to act on, though -- vqa_rad_open has only 200 open items (126 recoverable under the judge,
110 under EM), so ONE item there is worth 0.0079-0.0091 of sel_eff and a "flag" can be a single
sample.  Every flag is therefore reported with a paired item bootstrap CI and with the value of a
single item, so the reader can see whether a flag is a finding or a coin flip.

Both currencies on IDENTICAL picks.  CPU only.

    python3 src/cascade_methods/verifier_hparams_guardrail.py
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


def main():
    items = G.load_items()
    d = np.load(os.path.join(PARTS, "em_slots.npz"))
    em, judge = d["em"], d["judge"]

    V = {}
    for px in RUNGS:
        a = load_arm(px, "")
        if a is None or a["n_scored"] < 8965:
            print(f"px{px} incomplete -- skipped")
            continue
        sq, _ = slot_scores(a["scores"], items)
        r = G.sel_eff(sq, items)
        picks = r["picks"]
        V[px] = {"got_j": np.asarray(r["got"], int),
                 "got_e": np.array([em[i, picks[i]] for i in range(len(items))], int),
                 "rec_j": np.asarray(r["rec"], int),
                 "ds_index": np.asarray(r["ds_index"], int)}
    rec_em = (em.max(axis=1) == 1).astype(int)

    out = {}
    for px in RUNGS:
        if px == CTRL or px not in V:
            continue
        row = {}
        for j, ds in enumerate(G.EVAL_DS):
            cell = {}
            for cur, gk, rk in (("judge", "got_j", V[CTRL]["rec_j"]),
                                ("em", "got_e", rec_em)):
                m = (V[CTRL]["ds_index"] == j) & (rk == 1)
                idx = np.where(m)[0]
                n = len(idx)
                a, b = V[px][gk], V[CTRL][gk]
                obs = float(a[idx].mean() - b[idx].mean())
                rng = np.random.default_rng(BSEED)
                bs = np.empty(NBOOT)
                for k in range(NBOOT):
                    s = idx[rng.integers(0, n, n)]
                    bs[k] = a[s].mean() - b[s].mean()
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                cell[cur] = {"d_sel_eff": round(obs, 6), "ci": [round(lo, 6), round(hi, 6)],
                             "n_recoverable": int(n),
                             "excludes_0": bool(lo > 0 or hi < 0),
                             "one_item_worth": round(1.0 / n, 6)}
            row[ds] = cell
        out[str(px)] = row
        f = [f"{ds}/{c}" for ds in G.EVAL_DS for c in ("judge", "em")
             if row[ds][c]["d_sel_eff"] < 0]
        sig = [f"{ds}/{c}" for ds in G.EVAL_DS for c in ("judge", "em")
               if row[ds][c]["d_sel_eff"] < 0 and row[ds][c]["excludes_0"]]
        print(f"px{px:>9}  negative flags: {len(f)}/6  of which CI-excludes-0: "
              f"{len(sig)}  {sig}")

    out["_meta"] = {
        "_what": "per-set guardrail deltas vs the IN-SESSION 1,003,520 control, both currencies, "
                 "with paired item bootstrap CIs.",
        "_code": "src/cascade_methods/verifier_hparams_guardrail.py",
        "nboot": NBOOT, "seed": BSEED,
        "_why_one_item_worth_is_printed": "vqa_rad_open has 200 open items (126 recoverable under "
                                          "the judge, 110 under EM), so a single item moves its "
                                          "sel_eff by ~0.008-0.009. A guardrail flag smaller than "
                                          "that is one sample, not a finding.",
    }
    json.dump(out, open(os.path.join(PARTS, "guardrail_cis.json"), "w"), indent=1, default=float)
    print(f"\nwrote {PARTS}/guardrail_cis.json")


if __name__ == "__main__":
    main()
