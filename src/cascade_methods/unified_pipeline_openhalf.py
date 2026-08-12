#!/usr/bin/env python3
"""unified_pipeline_openhalf.py -- ATTACK 2: DOES UNIFYING COST ANYTHING ON THE OPEN BRANCH?

The 2026-08-12 07:18 artifact could only answer this trivially ("the sampled branch is the incumbent
arm unchanged") because no unified adapter existed.  Arm B trains ONE adapter on BOTH branches'
candidate sets, so the question becomes a real measurement: put the unified adapter back on the
frozen 2,345-item open-text pool and compare it with the incumbent, item for item.

  incumbent   ckpts/train/lora_verifier_disjoint          sel_eff 0.775204 (frozen bar)
  arm B       ckpts/train/lora_verifier_unified_s0        same recipe, same 10,364 open examples,
                                                          PLUS 10,364 option examples
  arm B0      ckpts/train/lora_verifier_optiononly_s0     option examples ONLY -- the format-specific
                                                          option verifier evaluated on the OTHER
                                                          format, i.e. the transfer control

The metric is the single frozen definition in src/training_methods/genframe_data.py.  Nothing is
redefined here.  The comparison is exactly paired: every adapter ranks the SAME candidate lists with
the SAME judge labels, so greedy / oracle are identical by construction and this script ASSERTS it.

CPU only (the GPU scoring is verifier_transfer_eval.py, which writes the dumps this reads).
    python3 src/cascade_methods/unified_pipeline_openhalf.py --tags unified_s0,optiononly_s0
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_methods"))
import unified_pipeline as U  # noqa: E402
import genframe_data as G  # noqa: E402

DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]


def scores_from(adapter):
    """{(ds, idx): [8 scores]} from an adapter's transfer dumps; None if any dump is missing."""
    out = {}
    for d in DS:
        p = os.path.join(U.ROOT, adapter, f"transfer_dump_{d}_lingshu7b.json")
        if not os.path.exists(p):
            return None, f"missing {p}"
        for r in json.load(open(p)):
            out[(r["ds"], r["idx"])] = list(r["scores"])
    return out, None


def dump_index(adapter):
    out = {}
    for d in DS:
        p = os.path.join(U.ROOT, adapter, f"transfer_dump_{d}_lingshu7b.json")
        for r in json.load(open(p)):
            out[(r["ds"], r["idx"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="unified_s0,optiononly_s0")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=U.SEED_BOOT)
    A = ap.parse_args()

    items = G.load_items()
    inc = G.incumbent_scores()
    base = G.sel_eff(inc, items)
    nt = G.null_test()
    assert nt["pass"], f"frozen-metric null test FAILED: {nt}"

    out = {"frozen_metric": "src/training_methods/genframe_data.py (single definition)",
           "null_test_of_the_metric": nt,
           "incumbent_bar": {"adapter": "ckpts/train/lora_verifier_disjoint",
                             "sel_eff": base["sel_eff"], "acc": base["acc"],
                             "greedy": base["greedy"], "oracle": base["oracle"],
                             "n": base["n"], "n_recoverable": base["n_recoverable"],
                             "per_ds": {d: base["per_ds"][d]["sel_eff"] for d in DS},
                             "contested": base["contested"],
                             "cand_auroc": G.cand_auroc(inc, items)},
           "arms": {}}

    for tag in A.tags.split(","):
        adapter = f"ckpts/train/lora_verifier_{tag}"
        sc, err = scores_from(adapter)
        if sc is None:
            out["arms"][tag] = {"adapter": adapter, "status": "NOT MEASURED", "why": err}
            continue
        # ---- pairing assertions: same items, same pools, same judge labels ---------------------
        di = dump_index(adapter)
        miss = [(it["ds"], it["idx"]) for it in items if (it["ds"], it["idx"]) not in di]
        bad_sl = sum(1 for it in items if (it["ds"], it["idx"]) in di
                     and list(di[(it["ds"], it["idx"])]["sl"]) != list(it["sl"]))
        bad_pr = sum(1 for it in items if (it["ds"], it["idx"]) in di
                     and [G.norm(x) for x in di[(it["ds"], it["idx"])]["preds"]]
                     != [G.norm(x) for x in it["preds"]])
        r = G.sel_eff(sc, items)
        assert abs(r["greedy"] - base["greedy"]) < 1e-12, "greedy must be identical (same pools)"
        assert abs(r["oracle"] - base["oracle"]) < 1e-12, "oracle must be identical (same pools)"
        bs = G.paired_bootstrap(r["got"], base["got"], rec=base["rec"],
                                nboot=A.nboot, seed=A.seed)
        bc = G.paired_bootstrap(r["got"], base["got"], rec=base["rec"], nboot=A.nboot,
                                seed=A.seed, mask=base["contested_mask"])
        out["arms"][tag] = {
            "adapter": adapter,
            "what_it_is": ("the UNIFIED verifier: one adapter, 10,364 open + 10,364 option examples"
                           if tag.startswith("unified") else
                           "the OPTION-ONLY verifier evaluated on the OPEN format (transfer control)"),
            "pairing_assertions": {"items_missing_from_dump": len(miss),
                                   "items_with_different_judge_labels": bad_sl,
                                   "items_with_different_candidate_pools": bad_pr,
                                   "greedy_identical": True, "oracle_identical": True},
            "sel_eff": r["sel_eff"], "acc": r["acc"], "n": r["n"],
            "n_recoverable": r["n_recoverable"],
            "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in DS},
            "contested": r["contested"],
            "cand_auroc": G.cand_auroc(sc, items),
            "vs_incumbent": {"d_sel_eff": bs["d_sel_eff"], "ci": bs["d_sel_eff_ci"],
                             "sig": not (bs["d_sel_eff_ci"][0] <= 0 <= bs["d_sel_eff_ci"][1]),
                             "d_acc": bs["d_acc"], "acc_ci": bs["d_acc_ci"],
                             "acc_sig": not (bs["d_acc_ci"][0] <= 0 <= bs["d_acc_ci"][1]),
                             "nboot": A.nboot, "seed": A.seed},
            "vs_incumbent_contested_stratum": {"d_sel_eff": bc["d_sel_eff"],
                                               "ci": bc["d_sel_eff_ci"],
                                               "sig": not (bc["d_sel_eff_ci"][0] <= 0
                                                           <= bc["d_sel_eff_ci"][1]),
                                               "n_stratum": bc["n_stratum"]},
            "guardrail_never_worse_than_incumbent_on_any_set":
                bool(G.guardrail_clean(r, base)),
            "guardrail_per_set_delta": {d: r["per_ds"][d]["sel_eff"] - base["per_ds"][d]["sel_eff"]
                                        for d in DS},
        }

    os.makedirs(U.PARTS, exist_ok=True)
    p = os.path.join(U.PARTS, "open_half_trained.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
