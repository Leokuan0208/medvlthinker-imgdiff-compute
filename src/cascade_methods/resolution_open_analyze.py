#!/usr/bin/env python3
"""resolution_open_analyze.py -- SWEEP 2: the open half's endpoint table, per resolution.

For every (cap, seed) arm produced by resolution_open_generate.py this computes, on the frozen
2,345-question endpoint:

    greedy_t0     temperature-0 single decode, judge-labelled  (the TRUE greedy arm)
    pool_modal    the modal answer of the 8 samples -- this is what the published pool calls
                  "greedy" (verifier_transfer_eval.py:g uses sc8's modal_pred), reported under its
                  own name so the two are never conflated
    oracle@8      P(at least one of the 8 is judged correct)
    selected      the deployed clean disjoint LoRA verifier's argmax pick is correct
    sel_eff       mean(pick correct | pool recoverable)   -- the frozen definition
    identity      selected == oracle@8 * sel_eff is asserted, not assumed

plus a Lincoln-Petersen capture-recapture ceiling PER CAP (using this cap's own independent seeds
as the two capture occasions -- the same estimator as coverage_diagnosis2.ceiling, which used the
8- and 16-sample pools), the laterality stratum, per-cell guardrails, and a paired item bootstrap
of every cap against the cap320 control generated in the SAME session.

    python3 src/cascade_methods/resolution_open_analyze.py
"""
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
os.makedirs(OUT, exist_ok=True)
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
CAPS = [("cap80", 62720), ("cap160", 125440), ("cap320", 250880), ("cap640", 501760),
        ("fullres", 1003520), ("native", 12845056)]
CONTROL = "cap320"                     # the DEPLOYED generator resolution
NBOOT, BSEED = 10000, 20260813

# the project's canonical laterality regex (src/training_methods/visverif_lib.py:150)
from src.training_methods.visverif_lib import LATERAL  # noqa: E402


def norm(s):
    return str(s).strip().lower()


#: the endpoint's own per-cell item counts -- an arm short of these is INCOMPLETE and is dropped
#: rather than analysed, so a half-written file can never enter a delta.
NEXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}


def load_arm(cap, tag):
    """{ds: {idx: row}} for one arm, or None if the arm is not complete."""
    out = {}
    for ds in DS:
        p = os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{tag}.jsonl")
        if not os.path.exists(p):
            return None
        d = {}
        for l in open(p):
            if l.strip():
                try:
                    r = json.loads(l)
                    d[r["idx"]] = r
                except Exception:
                    pass
        if len(d) < NEXP[ds]:
            return None
        out[ds] = d
    return out


def boot_delta(a, b, nboot=NBOOT, seed=BSEED):
    """paired item bootstrap of mean(b) - mean(a) over a common item order."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) == 0:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    s = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    J = json.load(open(os.path.join(SWEEP, "judge_cache.json")))
    vp = os.path.join(SWEEP, "verifier_score_cache.json")
    if os.path.exists(vp):
        V = json.load(open(vp))
    else:
        print("WARNING: no verifier_score_cache.json -- run the label stage first. Falling back "
              "to the stored deployed transfer-dump scores, which cover the deployed pool only, "
              "so any arm's selected/sel_eff would be scored on partial coverage. Aborting.")
        raise SystemExit(2)

    # ---- the canonical item order: the deployed endpoint, taken from the transfer dumps --------
    order = []
    for ds, nm in [("slake_open", "slake"), ("vqa_rad_open", "vqa_rad"), ("pathvqa_open", "pathvqa")]:
        p = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint",
                         f"transfer_dump_{nm}_open_lingshu7b.json")
        for r in json.load(open(p)):
            order.append((ds, r["idx"]))
    assert len(order) == 2345, len(order)

    def arm_stats(arm, name):
        """per-item vectors on the canonical order; None where the arm lacks the item."""
        rec, sel, orc, modal, nd, nmiss = [], [], [], [], [], 0
        n_unlabelled_slots = n_unscored_slots = n_slots = 0
        gold, ques, cands = [], [], []
        for ds, idx in order:
            r = arm[ds].get(idx)
            if r is None:
                nmiss += 1
                rec.append(0); sel.append(0); orc.append(0); modal.append(0); nd.append(0)
                gold.append(""); ques.append(""); cands.append([])
                continue
            preds = r["preds"]
            y = [J.get(f"{ds}|{idx}|{norm(a)}") for a in preds]
            sc = [V.get(f"{ds}|{idx}|{a}") for a in preds]
            n_slots += len(preds)
            n_unlabelled_slots += sum(1 for v in y if v is None)
            n_unscored_slots += sum(1 for v in sc if v is None)
            # matches src/training_methods/verifier_transfer_eval.py exactly: an unlabelled slot is
            # EXCLUDED from the oracle max and counts as WRONG if it is the one the verifier picks;
            # an unscored slot is pushed to the bottom of the ranking.
            yv = [0 if v is None else int(v) for v in y]
            svv = [-1e9 if v is None else float(v) for v in sc]
            k = int(np.argmax(svv))
            labelled = [int(v) for v in y if v is not None]
            orc.append(int(max(labelled) == 1) if labelled else 0)
            sel.append(int(yv[k] == 1))
            rec.append(int(max(labelled) == 1) if labelled else 0)
            mk = next((i for i, a in enumerate(preds) if norm(a) == norm(r["modal_pred"])), 0)
            modal.append(yv[mk])
            nd.append(len({norm(a) for a in preds}))
            gold.append(str(r["gold"])); ques.append(str(r["question"])); cands.append(preds)
        return dict(name=name, rec=np.array(rec), sel=np.array(sel), orc=np.array(orc),
                    modal=np.array(modal), nd=np.array(nd), n_missing=nmiss,
                    gold=gold, ques=ques, cands=cands,
                    n_slots=n_slots, n_unlabelled_slots=n_unlabelled_slots,
                    n_unscored_slots=n_unscored_slots)

    def summarize(S, dsmask):
        r, s, o = S["rec"], S["sel"], S["orc"]
        m = dsmask
        rr = (r == 1) & m
        se = float(s[rr].mean()) if rr.sum() else float("nan")
        # CONTESTED = recoverable AND the pool is not one unanimous answer (>=2 distinct), the
        # sensitive secondary endpoint; identical definition to genframe_data.sel_eff's con mask.
        con = rr & (S["nd"] >= 2)
        una = rr & (S["nd"] == 1)
        return dict(n=int(m.sum()), n_recoverable=int(rr.sum()),
                    oracle8=round(float(o[m].mean()), 6),
                    selected=round(float(s[m].mean()), 6),
                    pool_modal=round(float(S["modal"][m].mean()), 6),
                    sel_eff=round(se, 6),
                    contested_n=int(con.sum()),
                    contested_sel_eff=round(float(s[con].mean()), 6) if con.sum() else None,
                    unanimous_n=int(una.sum()),
                    mean_distinct_candidates=round(float(S["nd"][m].mean()), 4))

    dsmask = {ds: np.array([d == ds for d, _ in order]) for ds in DS}
    allm = np.ones(len(order), bool)

    res = {"_meta": {}, "by_cap": {}, "null_test": {}, "capture_recapture": {},
           "strata": {}, "vs_control": {}}

    arms = {}
    for cap, px in CAPS:
        for tag in ["s0", "s1", "s2"]:
            A = load_arm(cap, tag)
            if A is not None:
                arms[(cap, tag)] = arm_stats(A, f"{cap}_{tag}")
        G = load_arm(cap, "t0")
        if G is not None:
            arms[(cap, "t0")] = arm_stats(G, f"{cap}_t0")

    # ---- per-cap tables --------------------------------------------------------------------
    for cap, px in CAPS:
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms]
        if not seeds:
            continue
        row = {"max_pixels": px, "vision_token_budget": px // (28 * 28), "seeds": seeds,
               "per_seed": {}, "mean_sd": {}, "per_cell": {}}
        for t in seeds:
            row["per_seed"][t] = summarize(arms[(cap, t)], allm)
            A = arms[(cap, t)]
            jmiss = A["n_unlabelled_slots"] / max(A["n_slots"], 1)
            vmiss = A["n_unscored_slots"] / max(A["n_slots"], 1)
            row["per_seed"][t]["coverage_of_labels"] = {
                "items_missing_from_arm": A["n_missing"], "n_candidate_slots": A["n_slots"],
                "slots_without_a_judge_label": A["n_unlabelled_slots"],
                "frac_slots_without_a_judge_label": round(jmiss, 6),
                "slots_without_a_verifier_score": A["n_unscored_slots"],
                "frac_slots_without_a_verifier_score": round(vmiss, 6),
                "RELIABLE": bool(jmiss <= 0.01 and vmiss <= 0.01),
                "_rule": "an unlabelled slot is excluded from oracle@8 and counts as wrong if the "
                         "verifier picks it, so incomplete labelling biases this arm DOWNWARD. "
                         "Any arm with RELIABLE false must not be quoted as a measurement."}
        for q in ["oracle8", "selected", "sel_eff", "pool_modal"]:
            v = np.array([row["per_seed"][t][q] for t in seeds], float)
            row["mean_sd"][q] = {"mean": round(float(v.mean()), 6),
                                 "sd": round(float(v.std(ddof=1)), 6) if len(v) > 1 else None,
                                 "per_seed": [round(float(x), 6) for x in v]}
        # identity check: selected == oracle8 * sel_eff
        idm = max(abs(row["per_seed"][t]["selected"]
                      - row["per_seed"][t]["oracle8"] * row["per_seed"][t]["sel_eff"])
                  for t in seeds)
        row["identity_selected_eq_oracle_x_seleff_max_abs_err"] = round(float(idm), 9)
        for ds in DS:
            row["per_cell"][ds] = {
                q: {"mean": round(float(np.mean([summarize(arms[(cap, t)], dsmask[ds])[q]
                                                 for t in seeds])), 6),
                    "per_seed": [round(summarize(arms[(cap, t)], dsmask[ds])[q], 6) for t in seeds]}
                for q in ["oracle8", "selected", "sel_eff", "pool_modal"]}
        if (cap, "t0") in arms:
            A = arms[(cap, "t0")]
            row["greedy_t0"] = {"all": round(float(A["modal"].mean()), 6),
                                **{ds: round(float(A["modal"][dsmask[ds]].mean()), 6) for ds in DS},
                                "_what": "temperature-0, n=1 decode, judge-labelled. NOT the same "
                                         "quantity as pool_modal."}
        # measured token geometry actually used at this cap
        gm = {}
        for ds in DS:
            p = os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{seeds[0]}.jsonl")
            vt, pt, gt = [], [], []
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    vt.append(r.get("vision_px_tokens", -1))
                    pt.append(r.get("prompt_tokens", -1))
                    gt.append(float(np.mean(r["gen_tokens_all"])))
            gm[ds] = {"mean_vision_tokens": round(float(np.mean(vt)), 2),
                      "mean_prompt_tokens": round(float(np.mean(pt)), 2),
                      "mean_gen_tokens_per_sample": round(float(np.mean(gt)), 3), "n": len(vt)}
        row["measured_token_geometry"] = gm
        res["by_cap"][cap] = row

    # ---- capture-recapture ceiling per cap (LP over two INDEPENDENT seeds) ------------------
    for cap, px in CAPS:
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms]
        if len(seeds) < 2:
            continue
        per_ds = {}
        for ds in DS:
            m = dsmask[ds]
            ests = []
            for t1, t2 in itertools.combinations(seeds, 2):
                A = arms[(cap, t1)]["orc"][m].astype(bool)
                B = arms[(cap, t2)]["orc"][m].astype(bool)
                nA, nB, nAB = int(A.sum()), int(B.sum()), int((A & B).sum())
                if nAB:
                    ests.append(min(1.0, (nA * nB / nAB) / int(m.sum())))
            per_ds[ds] = {"n": int(m.sum()),
                          "LP_reachable_share_mean": round(float(np.mean(ests)), 6),
                          "LP_per_pair": [round(float(x), 6) for x in ests],
                          "observed_union_share": round(float(np.mean(
                              np.any([arms[(cap, t)]["orc"][m].astype(bool) for t in seeds], axis=0)
                          )), 6),
                          "oracle8_mean": round(float(np.mean(
                              [arms[(cap, t)]["orc"][m].mean() for t in seeds])), 6)}
        res["capture_recapture"][cap] = {
            "max_pixels": px, "per_cell": per_ds,
            "macro_open3_LP_ceiling": round(float(np.mean(
                [per_ds[d]["LP_reachable_share_mean"] for d in DS])), 6),
            "macro_open3_oracle8": round(float(np.mean([per_ds[d]["oracle8_mean"] for d in DS])), 6),
            "_method": "Lincoln-Petersen with THIS cap's own independent generation seeds as the "
                       "two capture occasions ('captured' = >=1 judged-correct answer in that "
                       "seed's 8-sample pool), averaged over the 3 seed pairs. Same estimator as "
                       "src/cascade_methods/coverage_diagnosis2.ceiling, which used the 8- and "
                       "16-sample pools instead. Heterogeneous per-item detection probability "
                       "biases LP DOWNWARD, so it is a LOWER bound on the iid-reachable share AT "
                       "THIS RESOLUTION. It is distribution-specific: the +0.0091 macro bound in "
                       "the retrospective is the cap320 distribution's bound, not a universal one."}

    # ---- laterality + length strata, arm-invariant masks ------------------------------------
    ref = arms.get((CONTROL, "s0"))
    if ref is not None:
        goldlen = np.array([len(g.split()) for g in ref["gold"]])
        lat_qg = np.array([bool(LATERAL.search(q)) or bool(LATERAL.search(g))
                           for q, g in zip(ref["ques"], ref["gold"])])
        masks = {"laterality_question_or_gold": lat_qg,
                 "short3_not_laterality": (goldlen <= 3) & ~lat_qg,
                 "gold_1word": goldlen == 1}
        for cap, px in CAPS:
            seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms]
            if not seeds:
                continue
            res["strata"][cap] = {}
            for nm, msk in masks.items():
                vals = []
                for t in seeds:
                    S = arms[(cap, t)]
                    rr = (S["rec"] == 1) & msk
                    vals.append(float(S["sel"][rr].mean()) if rr.sum() else float("nan"))
                orcs = [float(arms[(cap, t)]["orc"][msk].mean()) for t in seeds]
                res["strata"][cap][nm] = {
                    "n_items": int(msk.sum()),
                    "sel_eff_mean": round(float(np.mean(vals)), 6),
                    "sel_eff_per_seed": [round(v, 6) for v in vals],
                    "oracle8_mean": round(float(np.mean(orcs)), 6)}
        res["strata"]["_mask_note"] = (
            "masks are built from QUESTION and GOLD only (never from the candidates), so they are "
            "ARM-INVARIANT and the same items are compared at every cap. The project's published "
            "laterality stratum (visverif_lib.strata) also ORs in the candidate texts, which would "
            "move the mask with the arm and make caps incomparable.")

    # ---- every cap vs the cap320 control, paired, seed-matched ------------------------------
    for cap, px in CAPS:
        if cap == CONTROL:
            continue
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms and (CONTROL, t) in arms]
        if not seeds:
            continue
        blk = {"max_pixels": px, "seeds_paired": seeds, "per_metric": {}, "guardrail": {}}
        for q, fld in [("oracle8", "orc"), ("selected", "sel"), ("pool_modal", "modal")]:
            ds_, cis = [], []
            for t in seeds:
                d, ci = boot_delta(arms[(CONTROL, t)][fld], arms[(cap, t)][fld])
                ds_.append(d); cis.append(ci)
            blk["per_metric"][q] = {
                "delta_mean_over_seeds": round(float(np.mean(ds_)), 6),
                "delta_per_seed": [round(x, 6) for x in ds_],
                "ci95_per_seed": [[round(c[0], 6), round(c[1], 6)] for c in cis],
                "all_seeds_ci_exclude_zero": bool(all(c[0] > 0 or c[1] < 0 for c in cis)),
                "sign_consistent": bool(len(set(np.sign(ds_))) == 1)}
        # sel_eff delta, bootstrapped inside the union-recoverable stratum (paired items)
        se_d, se_ci = [], []
        for t in seeds:
            a, b = arms[(CONTROL, t)], arms[(cap, t)]
            m = (a["rec"] == 1) & (b["rec"] == 1)
            d, ci = boot_delta(a["sel"][m], b["sel"][m])
            se_d.append(d); se_ci.append(ci)
        blk["per_metric"]["sel_eff_on_jointly_recoverable"] = {
            "delta_mean_over_seeds": round(float(np.mean(se_d)), 6),
            "delta_per_seed": [round(x, 6) for x in se_d],
            "ci95_per_seed": [[round(c[0], 6), round(c[1], 6)] for c in se_ci],
            "all_seeds_ci_exclude_zero": bool(all(c[0] > 0 or c[1] < 0 for c in se_ci)),
            "_note": "restricted to items recoverable in BOTH arms so the conditioning set is "
                     "identical; the unconditional sel_eff per arm is in by_cap."}
        # MANIPULATION CHECK: did changing max_pixels actually change the candidate distribution?
        jac, agree, nn = [], 0, 0
        for t in seeds:
            a, b = arms[(CONTROL, t)], arms[(cap, t)]
            for i in range(len(order)):
                A = {norm(x) for x in a["cands"][i]}
                B = {norm(x) for x in b["cands"][i]}
                if not A and not B:
                    continue
                jac.append(len(A & B) / max(len(A | B), 1))
                agree += int(a["modal"][i] == b["modal"][i]
                             and norm(a["cands"][i][0] if a["cands"][i] else "")
                             == norm(b["cands"][i][0] if b["cands"][i] else ""))
                nn += 1
        blk["manipulation_check"] = {
            "mean_pool_jaccard_vs_control": round(float(np.mean(jac)), 6) if jac else None,
            "n_item_seed_pairs": nn,
            "_read": "Jaccard over the NORMALIZED answer sets of the two 8-sample pools for the "
                     "same item and the same seed. 1.0 would mean the resolution change did "
                     "nothing; anything well below 1.0 shows the treatment reached the candidate "
                     "distribution, which is what this attack is trying to move."}
        for ds in DS:
            m = dsmask[ds]
            dd = [float(arms[(cap, t)]["sel"][m].mean() - arms[(CONTROL, t)]["sel"][m].mean())
                  for t in seeds]
            blk["guardrail"][ds] = {"delta_selected_mean": round(float(np.mean(dd)), 6),
                                    "per_seed": [round(x, 6) for x in dd],
                                    "worse_than_control": bool(np.mean(dd) < 0),
                                    "within_seed_spread": bool(abs(np.mean(dd)) < np.std(dd, ddof=1)
                                                               if len(dd) > 1 else True)}
        res["vs_control"][cap] = blk

    res["_meta"] = {
        "control": CONTROL,
        "n_items": 2345,
        "judge": "src/labeling/run_judge.py -- MedVLThinker-32B (Qwen2.5-32B backbone), text-only, "
                 "the project's existing judge; labels cached by (ds, idx, normalized answer) and "
                 "seeded from the stored deployed judge files, so identical answer strings carry "
                 "byte-identical labels across arms.",
        "verifier": "ckpts/train/lora_verifier_disjoint (clean disjoint LoRA), HF transformers, "
                    "bf16, max_pixels 1,003,520, batch 1 -- HELD FIXED. This sweep moves the "
                    "GENERATOR only.",
        "nboot": NBOOT, "bootstrap_seed": BSEED,
    }
    json.dump(res, open(os.path.join(OUT, "open_generator_resolution.json"), "w"), indent=1)
    print(json.dumps({c: res["by_cap"][c]["mean_sd"] for c in res["by_cap"]}, indent=1))
    print("wrote", os.path.join(OUT, "open_generator_resolution.json"))


if __name__ == "__main__":
    main()
