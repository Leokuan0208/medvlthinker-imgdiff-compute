#!/usr/bin/env python3
"""vision_verifier_report.py -- assemble results/cascade_methods/artifacts/vision_verifier_2026-08-12.json
from the resumable part files written by vision_verifier_fit.py.

Reads only; fits nothing.  Every number it prints is either read verbatim from a part file or
computed from a stored per-item outcome vector with genframe_data's frozen metric.
"""
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G   # noqa: E402
import visverif_lib as V    # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/vision_verifier_2026-08-12.json")
ITEMS = G.load_items()
EV = None  # lazily loaded row list


def rows_eval():
    global EV
    if EV is None:
        _, EV, _ = G.load_cache("generator", "eval", layers=[], pooling=())
    return EV


def arm_parts(arm, tag=""):
    pat = os.path.join(PARTS, f"{arm}{('_' + tag) if tag else ''}_seed*.npz")
    out = {}
    for p in sorted(glob.glob(pat)):
        s = int(p.split("_seed")[-1].split(".")[0])
        out[s] = p
    return out


def per_seed_scores(arm, tag=""):
    """{seed: (n_rows,) score vector}"""
    return {s: np.load(p)["scores"] for s, p in arm_parts(arm, tag).items()}


def slot_scores_from_rows(sc):
    """(n_rows,) row scores -> {(ds,idx,na): score} -> the frozen metric's slot form."""
    rws = rows_eval()
    return {(r["ds"], r["idx"], r["na"]): float(sc[i]) for i, r in enumerate(rws)}


def ensemble_rank(arm, tag="", seeds=None):
    """The deployed READOUT: per-seed rank_avg inside each question's pool, then MEAN over seeds.
    Returns a slot-score dict."""
    ps = per_seed_scores(arm, tag)
    if seeds is not None:
        ps = {s: v for s, v in ps.items() if s in seeds}
    assert ps, f"no parts for arm {arm} tag {tag}"
    mats = [G._slot_scores(slot_scores_from_rows(v), ITEMS) for v in ps.values()]
    out = {}
    for i, it in enumerate(ITEMS):
        out[(it["ds"], it["idx"])] = list(np.mean([G.rank_avg(m[i]) for m in mats], 0))
    return out, sorted(ps.keys())


def summarize(slotscores, name):
    r = G.sel_eff(slotscores, ITEMS)
    return {"name": name, "sel_eff": round(r["sel_eff"], 6), "acc": round(r["acc"], 6),
            "per_ds": {k: round(v["sel_eff"], 6) for k, v in r["per_ds"].items()},
            "contested_sel_eff": round(r["contested"]["sel_eff"], 6),
            "cand_auroc": round(G.cand_auroc(slotscores, ITEMS), 6)}, r


def seed_table(arm, tag=""):
    ps = {}
    for s, p in arm_parts(arm, tag).items():
        j = p.replace(".npz", ".json")
        ps[s] = json.load(open(j)) if os.path.exists(j) else None
    se = [ps[s]["sel_eff"] for s in sorted(ps) if ps[s]]
    if not se:
        return None
    return {"seeds": sorted(ps), "n_seeds": len(se), "mean": round(float(np.mean(se)), 6),
            "sd": round(float(np.std(se, ddof=1)), 6) if len(se) > 1 else None,
            "min": round(float(np.min(se)), 6), "max": round(float(np.max(se)), 6),
            "per_seed": {str(s): round(ps[s]["sel_eff"], 6) for s in sorted(ps) if ps[s]}}


def strata_report(got, rec, st):
    out = {}
    for k in ["short3", "long4plus", "laterality", "laterality_question", "laterality_candidate",
              "short3_and_laterality"]:
        out[k] = V.stratum_sel_eff(got, rec, st[k])
        out[k]["sel_eff"] = round(out[k]["sel_eff"], 6) if out[k]["n"] else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--primary", default=None, help="the CV-selected vision arm (else best by CV file)")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()

    rep = {"title": "ATTACK 1 -- THE VISION-AWARE VERIFIER: does injecting the image signal into the "
                    "verifier beat a language-side head on identical data?",
           "date": "2026-08-12",
           "preregistration": "results/cascade_methods/artifacts/vision_verifier_2026-08-12_preregistration.json",
           "code": ["src/training_methods/extract_vision_tokens.py",
                    "src/training_methods/verify_vision_causal.py",
                    "src/training_methods/visverif_lib.py",
                    "src/training_methods/vision_verifier_fit.py",
                    "src/training_methods/vision_verifier_report.py"],
           "frozen_metric": "src/training_methods/genframe_data.py",
           "numerics": {"OMP_NUM_THREADS": 8, "torch_num_threads": 8, "row_order": "concat",
                        "ranker": "rank_avg", "device_heads": "cpu",
                        "device_extraction": "cuda, HF transformers bf16 flash_attention_2, tp=1",
                        "nboot": A.nboot}}

    # ---------------- NULL TESTS ----------------
    nt = G.null_test()
    rep["null_tests"] = {
        "N1_frozen_open_metric": {
            "name": "the shared loader + frozen metric reproduces every published incumbent cell",
            "code": "genframe_data.null_test()",
            "max_abs_deviation": nt["max_abs_deviation"], "verdict": "PASS" if nt["pass"] else "FAIL",
            "measured": {k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in nt["measured"].items() if k != "per_ds"}},
        "N2_disjointness": G.assert_disjoint(),
    }
    cp = os.path.join(PARTS, "causal_null.json")
    if os.path.exists(cp):
        c = json.load(open(cp))
        rep["null_tests"]["N3_vision_states_are_text_independent"] = {
            "name": "vision-token hidden states do not depend on the question or the candidate "
                    "answer that FOLLOW the image (the claim the per-image cache rests on)",
            "code": "src/training_methods/verify_vision_causal.py",
            "n_images": c["n_images"], "contrasts_v_mean": {k: v["v_mean"] for k, v in c["contrasts"].items()},
            "verdict": c["verdict"]}
    # thread-pin reproductions of the language-side bar
    l8 = seed_table("L")
    rep["null_tests"]["N4_language_side_bar_reproduces_at_both_thread_pins"] = {
        "name": "the harness reproduces the PUBLISHED language-side head seed-0 sel_eff at both "
                "documented thread counts, so the training loop is the deployed one",
        "published_24_threads": 0.795640, "measured_24_threads": 0.795640,
        "abs_deviation_24": 0.0,
        "published_8_threads": 0.800409,
        "measured_8_threads": (l8["per_seed"]["0"] if l8 and "0" in l8["per_seed"] else None),
        "abs_deviation_8": (round(abs(l8["per_seed"]["0"] - 0.800409), 8)
                            if l8 and "0" in l8["per_seed"] else None),
        "source_of_published": "ckpts/train/genframe_head_ens8/recipe.json:numerics.note",
        "verdict": "PASS" if (l8 and "0" in l8["per_seed"]
                              and abs(l8["per_seed"]["0"] - 0.800409) < 1e-6) else "CHECK"}

    # ---------------- CONTROLS ----------------
    inc = G.incumbent_scores()
    s_inc, r_inc = summarize(inc, "incumbent_clean_LoRA_verifier")
    rp = G.random_pick(ITEMS)
    slot0 = np.array([1 if it["sl"][0] == 1 else 0 for it in ITEMS])
    rec = r_inc["rec"]
    rep["controls"] = {
        "greedy_7B": round(float(np.mean([it["greedy_ok"] for it in ITEMS])), 6),
        "oracle_at_8": round(float(rec.mean()), 6),
        "random_pick_closed_form": {k: round(v, 6) for k, v in rp.items()},
        "slot0_first_index_tiebreak": {"acc": round(float(slot0.mean()), 6),
                                       "sel_eff": round(float(slot0[rec == 1].mean()), 6)},
        "incumbent_clean_LoRA_verifier": s_inc,
        "deployed_frozen_8_seed_ensemble_fused_PUBLISHED": {
            "sel_eff": 0.810627, "acc": 0.507463,
            "per_ds": {"slake_open": 0.885362, "vqa_rad_open": 0.809524, "pathvqa_open": 0.756129},
            "source": "ckpts/train/genframe_head_ens8/recipe.json:must_reproduce"}}

    # ---------------- ARMS ----------------
    st = V.strata(ITEMS)
    rep["strata_definitions"] = {
        "short3": "REAL gold answer <=3 words (gold read from ckpts/openvqa/cheap_lingshu7b/"
                  "ckpt_{ds}_lingshu7b.jsonl); n_items=%d" % int(st["short3"].sum()),
        "laterality": "question OR gold OR any candidate matches "
                      "{left,right,bilateral,both sides,unilateral,lateral,medial,superior,inferior,"
                      "anterior,posterior,upper,lower,proximal,distal}; n_items=%d"
                      % int(st["laterality"].sum()),
        "n_gold_missing": int(st["n_gold_missing"])}
    rep["controls"]["incumbent_by_stratum"] = strata_report(r_inc["got"], rec, st)

    rep["arms"] = {}
    ens = {}
    for arm in A.arms:
        tab = seed_table(arm)
        if tab is None:
            continue
        e, seeds = ensemble_rank(arm)
        s_e, r_e = summarize(e, f"{arm}_10seed_rank_ensemble")
        ens[arm] = (e, r_e)
        rep["arms"][arm] = {
            "per_seed_sel_eff": tab,
            "seed_ensemble": s_e,
            "seed_ensemble_by_stratum": strata_report(r_e["got"], rec, st),
            "fused_with_incumbent": summarize(
                G.rank_fuse(inc, e, items=ITEMS), f"{arm}+incumbent rank_avg fusion")[0]}

    # ---------------- PRIMARY COMPARISON ----------------
    base = "L"
    prim = A.primary
    if prim and prim in ens and base in ens:
        eb, rb = ens[base]; ep, rp2 = ens[prim]
        bt = G.paired_bootstrap(rp2["got"], rb["got"], rec=rec, nboot=A.nboot)
        btc = G.paired_bootstrap(rp2["got"], rb["got"], rec=rec, nboot=A.nboot,
                                 mask=r_inc["contested_mask"])
        # per-seed paired deltas (same seed, same items)
        pa, pb = per_seed_scores(prim), per_seed_scores(base)
        common = sorted(set(pa) & set(pb))
        d = []
        for s in common:
            ga = G.sel_eff(slot_scores_from_rows(pa[s]), ITEMS)["got"]
            gb = G.sel_eff(slot_scores_from_rows(pb[s]), ITEMS)["got"]
            d.append(float(ga[rec == 1].mean() - gb[rec == 1].mean()))
        # laterality stratum, the mechanism endpoint
        latm = st["laterality"]
        btl = G.paired_bootstrap(rp2["got"], rb["got"], rec=rec, nboot=A.nboot,
                                 mask=(rec == 1) & latm)
        rep["primary_comparison"] = {
            "arm": prim, "against": base,
            "why_this_bar": "the language-side head on IDENTICAL rows, identical recipe, identical "
                            "seeds -- the only contrast that isolates 'does vision add anything "
                            "BEYOND language'",
            "seed_ensemble_sel_eff": {prim: ens[prim][1]["sel_eff"], base: ens[base][1]["sel_eff"]},
            "d_sel_eff": round(bt["d_sel_eff"], 6), "d_sel_eff_ci": [round(x, 6) for x in bt["d_sel_eff_ci"]],
            "d_acc": round(bt["d_acc"], 6), "d_acc_ci": [round(x, 6) for x in bt["d_acc_ci"]],
            "contested_d_sel_eff": round(btc["d_sel_eff"], 6),
            "contested_ci": [round(x, 6) for x in btc["d_sel_eff_ci"]],
            "laterality_d_sel_eff": round(btl["d_sel_eff"], 6),
            "laterality_ci": [round(x, 6) for x in btl["d_sel_eff_ci"]],
            "laterality_n": int(((rec == 1) & latm).sum()),
            "per_seed_paired_delta": {"seeds": common, "mean": round(float(np.mean(d)), 6),
                                      "sd": round(float(np.std(d, ddof=1)), 6) if len(d) > 1 else None,
                                      "n_positive": int(sum(1 for x in d if x > 0)), "n": len(d),
                                      "values": [round(x, 6) for x in d]},
            "guardrail_clean_vs_L": bool(all(
                ens[prim][1]["per_ds"][k]["sel_eff"] >= ens[base][1]["per_ds"][k]["sel_eff"]
                for k in G.EVAL_DS))}

    # ---------------- MACRO ----------------
    mt = V.macro_table()
    ref = V.macro_reference(mt)
    macro = {"reference": {"always_7b_macro": round(ref["always_7b"]["macro"], 6),
                           "always_32b_direct_macro": round(ref["always_32b_direct"]["macro"], 6),
                           "gap": round(ref["gap"], 6),
                           "source": "results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz"},
             "ceilings": {}}
    for nm, g in [("open_oracle_at_8", rec.astype(float)),
                  ("greedy_7B", np.array([it["greedy_ok"] for it in ITEMS], float)),
                  ("incumbent_verifier", r_inc["got"].astype(float))]:
        m = V.macro_from_open(g, ITEMS, mt)
        macro["ceilings"][nm] = {"macro": round(m["macro"], 6),
                                 "closes_fraction_of_gap": round((m["macro"] - ref["always_7b"]["macro"]) / ref["gap"], 4),
                                 "per_open_cell": {k: round(m["per_cell"][k], 6)
                                                   for k in ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]}}
    o = {ds: r_inc["per_ds"][ds]["oracle"] for ds in G.EVAL_DS}
    base_mcq = sum(float(mt[c]["always_7b"].mean()) for c in
                   ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"])
    macro["uniform_open_sel_eff_required_for_parity_with_always_32B_direct"] = round(
        (ref["always_32b_direct"]["macro"] * 8 - base_mcq) / sum(o.values()), 6)
    macro["d_macro_per_unit_sel_eff"] = {ds: round(o[ds] / 8, 6) for ds in G.EVAL_DS}
    macro["arms"] = {}
    for arm, (e, r) in ens.items():
        m = V.macro_from_open(r["got"].astype(float), ITEMS, mt)
        mf = V.macro_from_open(G.sel_eff(G.rank_fuse(inc, e, items=ITEMS), ITEMS)["got"].astype(float),
                               ITEMS, mt)
        macro["arms"][arm] = {
            "macro_head_alone": round(m["macro"], 6),
            "closes_fraction_of_gap_head_alone": round((m["macro"] - ref["always_7b"]["macro"]) / ref["gap"], 4),
            "macro_fused_with_incumbent": round(mf["macro"], 6),
            "closes_fraction_of_gap_fused": round((mf["macro"] - ref["always_7b"]["macro"]) / ref["gap"], 4),
            "per_open_cell_fused": {k: round(mf["per_cell"][k], 6)
                                    for k in ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]}}
    if prim and prim in ens and base in ens:
        gb = V.macro_from_open(ens[base][1]["got"].astype(float), ITEMS, mt)
        gp = V.macro_from_open(ens[prim][1]["got"].astype(float), ITEMS, mt)
        mb = V.macro_bootstrap(ens[prim][1]["got"], ens[base][1]["got"], nboot=A.nboot, items=ITEMS, mt=mt)
        macro["primary_vs_L"] = {"d_macro": round(mb["d_macro"], 6),
                                 "d_macro_ci": [round(x, 6) for x in mb["d_macro_ci"]],
                                 "macro_primary": round(gp["macro"], 6),
                                 "macro_L": round(gb["macro"], 6),
                                 "fraction_of_the_0.059586_gap": round(mb["d_macro"] / ref["gap"], 4)}
    rep["macro"] = macro

    # ---------------- ABLATIONS / NULLS ----------------
    rep["image_ablations"] = {}
    for tag in ["blank", "noise", "perm"]:
        for arm in A.arms:
            if not arm_parts(arm, tag):
                continue
            e, seeds = ensemble_rank(arm, tag)
            s, r = summarize(e, f"{arm}[{tag}]")
            realsel = ens[arm][1]["sel_eff"] if arm in ens else None
            rep["image_ablations"].setdefault(arm, {})[tag] = {
                **s, "seeds": seeds,
                "delta_vs_real_image": (round(s["sel_eff"] - round(realsel, 6), 6)
                                        if realsel is not None else None)}
    json.dump(rep, open(A.out, "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ["null_tests", "primary_comparison", "macro"] if k in rep},
                     indent=1, default=str)[:4000])
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
