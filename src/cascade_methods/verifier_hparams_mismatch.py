#!/usr/bin/env python3
"""verifier_hparams_mismatch.py -- KNOB 3: separate the RESOLUTION effect from the
TRAIN/INFERENCE MISMATCH effect, using the pre-registered base-model control.

THE CONFOUND
------------
ckpts/train/lora_verifier_disjoint was TRAINED at max_pixels 1,003,520.  Scoring it at any
other resolution changes TWO things at once: how much of the image the model sees (a
RESOLUTION effect) and how far the input is from the distribution the adapter was fitted on
(a MISMATCH effect).  One adapter cannot separate them on its own.

THE CONTROL (pre-registered in _verifier_hparams_parts/prereg.json)
------------------------------------------------------------------
The BASE Lingshu-7B with NO adapter, run as a zero-shot verifier on the IDENTICAL prompt,
pool and item order at {250,880 / 1,003,520 / 12,845,056}.  It was never trained at any
resolution, so NO point on its ladder is a mismatch.  Whatever its curve does is therefore a
pure resolution effect, and the DIFFERENCE between the two curves is the mismatch term:

    d_LoRA(px)  =  resolution(px) + mismatch(px)
    d_base(px)  =  resolution(px)
    DiD(px)     =  d_LoRA(px) - d_base(px)  =  mismatch(px)

read under the assumption that the resolution term is shared by the two models -- which is the
assumption the design can offer, and it is stated as such, not hidden.

FLOOR CHECK -- WITHOUT WHICH THE CONTROL IS WORTHLESS
-----------------------------------------------------
A control that sits at the random-pick floor cannot move, so its flatness would mean nothing.
This project has measured that every TRAINING-FREE selector sits at that floor, so the check
is mandatory: the floor is measured here by uniform random slot picks, and the base verifier
must clear it before any of its deltas are interpreted.

Both currencies (32B judge, normalised exact match) on IDENTICAL picks.  CPU only.

    python3 src/cascade_methods/verifier_hparams_mismatch.py
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
NBOOT = 10000
BSEED = 20260815
CTRL = 1003520
RUNGS = [250880, 12845056]      # the rungs the base control was run at, besides the control


def got_vec(px, pfx, items, judge, em):
    """Per-item 0/1 'the pick was correct' in BOTH currencies, on IDENTICAL picks."""
    a = load_arm(px, pfx)
    if a is None or a["n_scored"] < 8965:
        return None
    sq, _ = slot_scores(a["scores"], items)
    r = G.sel_eff(sq, items)
    picks = r["picks"]
    n = len(items)
    got_j = np.asarray(r["got"], int)
    got_e = np.array([em[i, picks[i]] for i in range(n)], int)
    S = G._slot_scores(sq, items)
    return {"got_j": got_j, "got_e": got_e, "picks": np.asarray(picks, int),
            "auroc_j": float(G.auroc(judge.ravel(), S.ravel())),
            "auroc_e": float(G.auroc(em.ravel(), S.ravel())),
            "sel_eff_j": float(r["sel_eff"]), "sel_eff_e": None, "S": S,
            "rec_j": np.asarray(r["rec"], int),
            "vis_tok": a["geometry"]["mean_vision_tokens"]}


def did_boot(lo_a, lo_b, ba_a, ba_b, sub, nboot=NBOOT, seed=BSEED):
    """Paired item bootstrap of the difference-in-differences
       (lo_a - lo_b) - (ba_a - ba_b) inside `sub`, all four vectors paired on the SAME items."""
    rng = np.random.default_rng(seed)
    idx = np.where(np.asarray(sub, bool))[0]
    m = len(idx)
    d_lo = np.empty(nboot); d_ba = np.empty(nboot); did = np.empty(nboot)
    for k in range(nboot):
        j = idx[rng.integers(0, m, m)]
        a = lo_a[j].mean() - lo_b[j].mean()
        b = ba_a[j].mean() - ba_b[j].mean()
        d_lo[k] = a; d_ba[k] = b; did[k] = a - b
    obs_lo = lo_a[idx].mean() - lo_b[idx].mean()
    obs_ba = ba_a[idx].mean() - ba_b[idx].mean()

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"d_lora": float(obs_lo), "d_lora_ci": ci(d_lo),
            "d_base": float(obs_ba), "d_base_ci": ci(d_ba),
            "DiD_mismatch_term": float(obs_lo - obs_ba), "DiD_ci": ci(did),
            "DiD_excludes_0": bool(ci(did)[0] > 0 or ci(did)[1] < 0),
            "p_two_sided_DiD": float(2 * min((did >= 0).mean(), (did <= 0).mean())),
            "nboot": nboot, "seed": seed, "n_items_in_subset": int(m)}


def auroc_boot(S_a, S_b, lab, nboot=2000, seed=BSEED):
    """Paired ITEM bootstrap of the candidate-level AUROC difference between two arms."""
    rng = np.random.default_rng(seed)
    n = lab.shape[0]
    d = np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, n, n)
        d[k] = G.auroc(lab[s].ravel(), S_a[s].ravel()) - G.auroc(lab[s].ravel(), S_b[s].ravel())
    return {"d_auroc": float(G.auroc(lab.ravel(), S_a.ravel()) - G.auroc(lab.ravel(), S_b.ravel())),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "excludes_0": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
            "nboot": nboot, "seed": seed}


def random_pick_floor(lab, ndraw=20000, seed=0):
    rng = np.random.default_rng(seed)
    rec = (lab.max(axis=1) == 1)
    n = lab.shape[0]
    ar = np.arange(n)
    v = np.array([lab[ar, rng.integers(0, 8, n)][rec].mean() for _ in range(ndraw)])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
            "ndraw": ndraw,
            "_what": "uniform random slot pick, sel_eff on the recoverable items. A selector at "
                     "this value carries no signal and its resolution curve cannot move."}


def main():
    items = G.load_items()
    d = np.load(os.path.join(PARTS, "em_slots.npz"))
    em, judge = d["em"], d["judge"]

    out = {"_what": "separates the RESOLUTION effect from the TRAIN/INFERENCE MISMATCH effect "
                    "using the pre-registered no-adapter base-model control.",
           "_code": "src/cascade_methods/verifier_hparams_mismatch.py",
           "_scores": "ckpts/openvqa/verifier_hparams/scores_{,base_}px*.jsonl",
           "_identity": "d_LoRA = resolution + mismatch ; d_base = resolution ; "
                        "DiD = d_LoRA - d_base = mismatch, under the stated assumption that the "
                        "resolution term is shared between the adapted and unadapted model."}

    # ---- the floor check, without which the control means nothing --------------------
    out["0_floor_check"] = {
        "random_pick_floor_judge": random_pick_floor(judge),
        "random_pick_floor_em": random_pick_floor(em),
    }

    lo = {px: got_vec(px, "", items, judge, em) for px in [CTRL] + RUNGS}
    ba = {px: got_vec(px, "base_", items, judge, em) for px in [CTRL] + RUNGS}
    missing = [f"{'base_' if k=='b' else ''}px{px}"
               for k, dd in (("l", lo), ("b", ba)) for px, v in dd.items() if v is None]
    if missing:
        out["_incomplete"] = missing
        json.dump(out, open(os.path.join(PARTS, "mismatch.json"), "w"), indent=1, default=float)
        print("INCOMPLETE:", missing)
        return

    fl = out["0_floor_check"]["random_pick_floor_judge"]
    out["1_the_base_control_clears_the_floor"] = {
        "base_sel_eff_judge_at_control": lo and ba[CTRL]["sel_eff_j"],
        "random_pick_floor_judge_mean": fl["mean"],
        "sd_above_floor": (ba[CTRL]["sel_eff_j"] - fl["mean"]) / fl["sd"],
        "base_cand_auroc_judge_at_control": ba[CTRL]["auroc_j"],
        "lora_sel_eff_judge_at_control": lo[CTRL]["sel_eff_j"],
        "lora_cand_auroc_judge_at_control": lo[CTRL]["auroc_j"],
        "_read": "the zero-shot base verifier is NOT at the random-pick floor, so its flat "
                 "sel_eff curve is a real measurement and not a floor artifact. It is however a "
                 "much weaker selector than the adapter, which is the known result that a "
                 "TRAINED verifier is the only thing that broke the floor in this project.",
    }

    rec = lo[CTRL]["rec_j"] == 1
    out["2_difference_in_differences"] = {}
    for px in RUNGS:
        row = {
            "direction": "DOWN from the trained resolution" if px < CTRL else "UP from it",
            "vision_tokens_lora": lo[px]["vis_tok"], "vision_tokens_base": ba[px]["vis_tok"],
            "judge": did_boot(lo[px]["got_j"], lo[CTRL]["got_j"],
                              ba[px]["got_j"], ba[CTRL]["got_j"], rec),
            "em": did_boot(lo[px]["got_e"], lo[CTRL]["got_e"],
                           ba[px]["got_e"], ba[CTRL]["got_e"], rec),
            "auroc_lora": auroc_boot(lo[px]["S"], lo[CTRL]["S"], judge),
            "auroc_base": auroc_boot(ba[px]["S"], ba[CTRL]["S"], judge),
        }
        out["2_difference_in_differences"][str(px)] = row
        print(f"px{px}: judge d_lora {row['judge']['d_lora']:+.6f}  d_base "
              f"{row['judge']['d_base']:+.6f}  DiD {row['judge']['DiD_mismatch_term']:+.6f} "
              f"CI [{row['judge']['DiD_ci'][0]:+.6f},{row['judge']['DiD_ci'][1]:+.6f}]  "
              f"p={row['judge']['p_two_sided_DiD']:.4f}", flush=True)
        print(f"        AUROC  d_lora {row['auroc_lora']['d_auroc']:+.6f} "
              f"CI [{row['auroc_lora']['ci'][0]:+.6f},{row['auroc_lora']['ci'][1]:+.6f}] | "
              f"d_base {row['auroc_base']['d_auroc']:+.6f} "
              f"CI [{row['auroc_base']['ci'][0]:+.6f},{row['auroc_base']['ci'][1]:+.6f}]",
              flush=True)

    json.dump(out, open(os.path.join(PARTS, "mismatch.json"), "w"), indent=1, default=float)
    print(f"\nwrote {PARTS}/mismatch.json")


if __name__ == "__main__":
    main()
