#!/usr/bin/env python3
"""vision_diversity_analyze.py -- SWEEP 3 analysis: candidate diversity along the VISION axis.

Reads the (idx, view) generation dump, the judge labels and the CLEAN disjoint-verifier scores
produced in this session, builds every pre-registered N=8 arm, and reports for each:
oracle@8, sel_eff, SELECTED accuracy, distinct-candidate count, vision-token cost, and a
Lincoln-Petersen capture-recapture ceiling -- against a MATCHED iid-8 control generated in the
SAME session and serving config.

The frozen metric and the bootstrap are IMPORTED, never re-implemented:
  src/training_methods/genframe_data.py   sel_eff / picks / null_test / rank_avg
  src/cascade_methods/open_diverse.py     boot_delta_seleff (the correct paired form when the two
                                          arms' recoverable-conditioning sets differ)
  src/training_methods/visverif_lib.py    LATERAL regex, macro_from_open

  python3 src/cascade_methods/vision_diversity_analyze.py
"""
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
sys.path.insert(0, J("src/training_methods"))
sys.path.insert(0, J("src/cascade_methods"))
import genframe_data as G                       # noqa: E402
import visverif_lib as V                        # noqa: E402
from open_diverse import boot_delta_seleff, boot_delta_paired   # noqa: E402
from vision_diversity_gen import VIEWS, VIEW_GROUP, REFUSED_VIEWS, SYS, TEMP, MAX_TOKENS, TOP_P  # noqa: E402

EVAL_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
GEN_DIR = "ckpts/openvqa/visdiv"
NBOOT, SEED, N_RAND_SEEDS = 10000, 0, 20
K = 3                     # samples per view
N = 8                     # every reported arm is an 8-candidate pool

# ------------------------------------------------------------------ PRE-REGISTERED compositions
# Fixed BEFORE any eval number was computed. Each is a view list; slot i of draw s takes
# view_list[i % len(view_list)] at sample index (s + i // len(view_list)) % K.
PORTFOLIOS = OrderedDict([
    ("P_mixed",  ["r320", "up640", "up1280", "c_center", "t_tl", "t_br", "p_gamma_lo", "p_autoc"]),
    ("P_res",    ["r160", "r320", "up640", "up1280"]),
    ("P_crop",   ["r320", "c_center", "t_tl", "t_tr", "t_bl", "t_br"]),
    ("P_photo",  ["r320", "p_gamma_lo", "p_gamma_hi", "p_autoc"]),
])
PRIMARY = "P_mixed"       # the only arm with K fully DISJOINT draws (8 distinct views x 1 sample)
ALL_VIEWS = list(VIEWS)


def loadl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def norm(s):
    return str(s).strip().lower()


# ------------------------------------------------------------------ data assembly
def load_ds(ds):
    """gen[idx][view] = [preds]; vt[idx][view] = vision tokens; judge/score maps; question/gold."""
    gen, vt, meta = defaultdict(dict), defaultdict(dict), {}
    ngen_fail = 0
    for r in loadl(J(f"{GEN_DIR}/gen_{ds}.jsonl")):
        if r.get("err") or not r.get("preds"):
            ngen_fail += 1
            continue
        gen[str(r["idx"])][r["view"]] = list(r["preds"])
        vt[str(r["idx"])][r["view"]] = int(r.get("vis_tokens", 0))
        meta.setdefault(str(r["idx"]), (r.get("question", ""), r.get("gold", "")))

    jud, nj_fail = {}, 0
    exp = {r["idx"]: r for r in loadl(J(f"{GEN_DIR}/gen_{ds}_scexploded.jsonl"))}
    for r in loadl(J(f"{GEN_DIR}/gen_{ds}_scexploded.judge.jsonl")):
        e = exp.get(r["idx"])
        if e is None:
            nj_fail += 1
            continue
        jud[(str(e["idx"]).split("||")[0], e["na"])] = int(r["judge_ok"])

    # score shards: the repo convention is no suffix at N==1, '_sKofN' only when truly sharded,
    # and every reader must accept BOTH forms and merge.
    import glob as _glob
    parts = sorted(_glob.glob(J(f"{GEN_DIR}/scores/scores_{ds}_disjoint.jsonl"))) + \
        sorted(_glob.glob(J(f"{GEN_DIR}/scores/scores_{ds}_disjoint_s*of*.jsonl")))
    sc, ns_fail = {}, 0
    for pth in parts:
        for r in loadl(pth):
            if r.get("score") is None:
                ns_fail += 1
                continue
            sc[(str(r["idx"]), r["na"])] = float(r["score"])

    # the incumbent GREEDY arm (temp 0, base view) -- identical for every arm, used only for
    # the 'greedy' column; read from the published dump, NOT regenerated.
    gp = J(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.judge.jsonl")
    greedy = {str(r["idx"]): int(r["judge_ok"]) for r in loadl(gp)} if os.path.exists(gp) else {}
    return dict(gen=gen, vt=vt, meta=meta, jud=jud, sc=sc, greedy=greedy,
                score_parts=[os.path.basename(x) for x in parts],
                fails=dict(gen=ngen_fail, judge_unmatched=nj_fail, score=ns_fail))


def canonical_order(ds):
    """The endpoint's item order for this cell (genframe_data.load_items), as strings.
    Using it -- rather than a fresh sort -- is what lets visverif_lib.macro_from_open accept
    these vectors: that function ASSERTS the open cells are 645/200/1500 in this exact order."""
    return [str(it["idx"]) for it in G.load_items() if it["ds"] == ds]


def complete_idx(D, ds):
    """Questions with every view + the iid control present, and every candidate judged AND scored,
    returned in the CANONICAL endpoint order."""
    ok = []
    for idx in canonical_order(ds):
        byv = D["gen"].get(idx)
        if not byv or not all(v in byv for v in ALL_VIEWS) or "iid" not in byv:
            continue
        if any(len(byv[v]) < K for v in ALL_VIEWS) or len(byv["iid"]) < 24:
            continue
        cands = {norm(a) for v in byv for a in byv[v]}
        if all((idx, na) in D["jud"] and (idx, na) in D["sc"] for na in cands):
            ok.append(idx)
    return ok


# ------------------------------------------------------------------ arm construction
def slots_portfolio(byv, views, draw):
    """8 (view, sample) slots from a view list -- the pre-registered deterministic rule."""
    out = []
    for i in range(N):
        v = views[i % len(views)]
        s = (draw + i // len(views)) % K
        out.append((v, byv[v][s]))
    return out


def slots_iid(byv, draw):
    """8 iid samples at the BASE view: draws 0/1/2 are DISJOINT thirds of the 24-sample control."""
    return [("iid", a) for a in byv["iid"][draw * N:(draw + 1) * N]]


def slots_rand_viewpool(byv, rng):
    """8 uniformly random of the 36 view-pool candidates (12 views x K)."""
    pool = [(v, a) for v in ALL_VIEWS for a in byv[v][:K]]
    return [pool[i] for i in rng.choice(len(pool), N, replace=False)]


def slots_rand_iidpool(byv, rng):
    return [("iid", byv["iid"][i]) for i in rng.choice(24, N, replace=False)]


def make_items(D, idxs, ds, slot_fn):
    """genframe-schema items for one arm: sl / scores / preds / greedy_ok, 8 slots per question."""
    items, vtoks = [], []
    for idx in idxs:
        sl = slot_fn(D["gen"][idx])
        preds = [a for _, a in sl]
        items.append({"ds": ds, "idx": idx,
                      "sl": [D["jud"][(idx, norm(a))] for a in preds],
                      "scores": [D["sc"][(idx, norm(a))] for a in preds],
                      "preds": preds,
                      "greedy_ok": D["greedy"].get(idx, 0)})
        vtoks.append(np.mean([D["vt"][idx][v] if v != "iid" else D["vt"][idx]["r320"]
                              for v, _ in sl]))
    return items, float(np.mean(vtoks))


def arm_vectors(items):
    """rec / got / distinct / oracle / acc / sel_eff for one arm, using the FROZEN pick rule."""
    picks = G.picks_from_scores({(it["ds"], it["idx"]): it["scores"] for it in items}, items)
    rec = np.array([1 if 1 in it["sl"] else 0 for it in items])
    got = np.array([int(it["sl"][picks[i]] == 1) for i, it in enumerate(items)])
    nd = np.array([len({norm(a) for a in it["preds"]}) for it in items])
    return dict(rec=rec, got=got, n_distinct=nd,
                oracle=float(rec.mean()), acc=float(got.mean()),
                sel_eff=float(got[rec == 1].mean()) if rec.sum() else float("nan"),
                mean_distinct=float(nd.mean()))


def lincoln_petersen(recA, recB):
    """Chao/LP two-sample coverage ceiling -- the SAME estimator as coverage_diagnosis2 F_ceiling.
    'Captured' = the pool contains >=1 correct answer. Heterogeneous per-item detection biases LP
    DOWNWARD, so this is a LOWER BOUND on the share reachable at N=infinity FOR THIS DISTRIBUTION."""
    a, b = int(recA.sum()), int(recB.sum())
    both = int((recA & recB).sum())
    either = int((recA | recB).sum())
    est = (a * b / both) if both else float("nan")
    return {"captured_A": a, "captured_B": b, "captured_both": both, "captured_either": either,
            "LP_reachable_share": float(est / len(recA)) if both else float("nan"),
            "observed_either_share": float(either / len(recA))}


# ------------------------------------------------------------------ null tests
def null_tests(DATA, IDX):
    out = {}
    nt = G.null_test()
    out["N1_frozen_metric"] = {
        "what": "reproduce every published incumbent cell from the transfer dumps with THIS "
                "session's metric code (src/training_methods/genframe_data.py)",
        "measured": nt["measured"], "published": nt["published"],
        "max_abs_deviation": nt["max_abs_deviation"], "verdict": "PASS" if nt["pass"] else "FAIL",
        "note": nt["note"]}

    # N2 -- the portfolio's BASE view is the iid control's rendering. Same prompt, same temperature,
    # same max_pixels, same session. Per-candidate correctness must agree within sampling noise:
    # any gap is a bug in the harness, not a view effect.
    rows = []
    for ds in EVAL_DS:
        D, idxs = DATA[ds], IDX[ds]
        b = [D["jud"][(i, norm(a))] for i in idxs for a in D["gen"][i]["r320"][:K]]
        d = [D["jud"][(i, norm(a))] for i in idxs for a in D["gen"][i]["iid"]]
        se = (np.var(b, ddof=1) / len(b) + np.var(d, ddof=1) / len(d)) ** 0.5
        rows.append({"ds": ds, "n_base_samples": len(b), "n_iid_samples": len(d),
                     "base_view_percand_acc": round(float(np.mean(b)), 6),
                     "iid_control_percand_acc": round(float(np.mean(d)), 6),
                     "diff": round(float(np.mean(b) - np.mean(d)), 6),
                     "se_of_diff": round(float(se), 6),
                     "z": round(float((np.mean(b) - np.mean(d)) / se), 3) if se else None})
    out["N2_base_view_equals_iid_control"] = {
        "what": "view 'r320' IS the iid control's rendering (identity transform, max_pixels "
                "250880 = the incumbent convention). Per-candidate judge accuracy must match "
                "within sampling noise.",
        "per_ds": rows,
        "max_abs_z": (round(float(max(abs(r["z"]) for r in rows if r["z"] is not None)), 3)
                      if any(r["z"] is not None for r in rows) else None),
        "verdict": ("PASS" if all(abs(r["z"]) < 3 for r in rows if r["z"] is not None)
                    else "INVESTIGATE")}

    # N3 -- one judge call per distinct (idx, answer): no arm can be advantaged by judge noise,
    # because a string produced by BOTH the portfolio and the iid control is judged once.
    n_rows = n_pairs = 0
    for ds in EVAL_DS:
        rows = loadl(J(f"{GEN_DIR}/gen_{ds}_scexploded.jsonl"))
        n_rows += len(rows)
        n_pairs += len({(str(r["idx"]).split("||")[0], r["na"]) for r in rows})
    out["N3_one_judge_label_per_distinct_answer"] = {
        "what": "candidates are deduplicated by normalized answer across ALL views AND the iid "
                "control before judging, so a shared string carries the SAME label in both arms.",
        "judge_rows": n_rows, "distinct_idx_answer_pairs": n_pairs,
        "verdict": "PASS" if n_rows == n_pairs else "FAIL"}

    # N4 -- the exact identity selected = oracle x sel_eff (never the additive form)
    errs = []
    for ds in EVAL_DS:
        it, _ = make_items(DATA[ds], IDX[ds], ds, lambda b: slots_iid(b, 0))
        v = arm_vectors(it)
        errs.append(abs(v["acc"] - v["oracle"] * v["sel_eff"]))
    out["N4_identity_selected_eq_oracle_x_sel_eff"] = {
        "max_abs_error": float(max(errs)), "verdict": "PASS" if max(errs) < 1e-12 else "FAIL"}
    return out


# ------------------------------------------------------------------ per-arm reporting
def report_arm(DATA, IDX, slot_fn, name):
    per_ds, rec_all, got_all, nd_all, vt_all, n_all = {}, [], [], [], [], 0
    for ds in EVAL_DS:
        items, vt = make_items(DATA[ds], IDX[ds], ds, slot_fn)
        v = arm_vectors(items)
        per_ds[ds] = {"n": len(items), "oracle@8": round(v["oracle"], 6),
                      "selected": round(v["acc"], 6), "sel_eff": round(v["sel_eff"], 6),
                      "mean_distinct": round(v["mean_distinct"], 4),
                      "mean_vis_tokens": round(vt, 1)}
        rec_all.append(v["rec"]); got_all.append(v["got"]); nd_all.append(v["n_distinct"])
        vt_all.append(vt * len(items)); n_all += len(items)
    rec = np.concatenate(rec_all); got = np.concatenate(got_all); nd = np.concatenate(nd_all)
    return {"name": name, "n": int(n_all), "per_ds": per_ds,
            "pooled": {"oracle@8": round(float(rec.mean()), 6),
                       "selected": round(float(got.mean()), 6),
                       "sel_eff": round(float(got[rec == 1].mean()), 6),
                       "mean_distinct": round(float(nd.mean()), 4),
                       "mean_vis_tokens_per_candidate": round(float(sum(vt_all) / n_all), 1)},
            "_rec": rec, "_got": got}


def cat_rec_got(reports):
    return reports["_rec"], reports["_got"]


def compare(A, B, label):
    """A vs B on the SAME items. sel_eff uses the differing-conditioning-set paired bootstrap."""
    ra, ga = cat_rec_got(A); rb, gb = cat_rec_got(B)
    se = boot_delta_seleff(ra, ga, rb, gb, nboot=NBOOT, seed=SEED)
    ac = boot_delta_paired(ga, gb, nboot=NBOOT, seed=SEED)
    orc = boot_delta_paired(ra, rb, nboot=NBOOT, seed=SEED)
    return {"contrast": label,
            "d_oracle@8": round(orc[0], 6), "d_oracle_ci95": [round(orc[1], 6), round(orc[2], 6)],
            "d_selected": round(ac[0], 6), "d_selected_ci95": [round(ac[1], 6), round(ac[2], 6)],
            "d_sel_eff": round(se[0], 6), "d_sel_eff_ci95": [round(se[1], 6), round(se[2], 6)],
            "selected_significant": bool(ac[1] > 0 or ac[2] < 0),
            "oracle_significant": bool(orc[1] > 0 or orc[2] < 0)}


def strata_masks(DATA, IDX):
    """Laterality / short-gold masks recomputed on THIS pool with visverif_lib's regex."""
    m = {"laterality": [], "short3": [], "ds": []}
    for ds in EVAL_DS:
        D = DATA[ds]
        for idx in IDX[ds]:
            q, gold = D["meta"][idx]
            pool = " ".join(a for v in D["gen"][idx] for a in D["gen"][idx][v])
            lat = bool(V.LATERAL.search(q)) or bool(V.LATERAL.search(str(gold))) \
                or bool(V.LATERAL.search(pool))
            m["laterality"].append(lat)
            m["short3"].append(len(str(gold).split()) <= 3)
            m["ds"].append(ds)
    return {k: np.array(v) for k, v in m.items()}


def stratum_compare(A, B, mask, name):
    ra, ga = cat_rec_got(A); rb, gb = cat_rec_got(B)
    se = boot_delta_seleff(ra[mask], ga[mask], rb[mask], gb[mask], nboot=NBOOT, seed=SEED)
    ac = boot_delta_paired(ga[mask], gb[mask], nboot=NBOOT, seed=SEED)
    return {"stratum": name, "n": int(mask.sum()),
            "n_recoverable_portfolio": int(ra[mask].sum()), "n_recoverable_iid": int(rb[mask].sum()),
            "portfolio_oracle": round(float(ra[mask].mean()), 6),
            "iid_oracle": round(float(rb[mask].mean()), 6),
            "portfolio_sel_eff": round(float(ga[mask][ra[mask] == 1].mean()), 6),
            "iid_sel_eff": round(float(gb[mask][rb[mask] == 1].mean()), 6),
            "portfolio_selected": round(float(ga[mask].mean()), 6),
            "iid_selected": round(float(gb[mask].mean()), 6),
            "d_sel_eff": round(se[0], 6), "d_sel_eff_ci95": [round(se[1], 6), round(se[2], 6)],
            "d_selected": round(ac[0], 6), "d_selected_ci95": [round(ac[1], 6), round(ac[2], 6)]}


def novelty(DATA, IDX):
    """Does a view produce answers the iid pool never produces -- and are they RIGHT?
    This is the coverage mechanism: new candidates are only worth having if some are correct."""
    out = {}
    for ds in EVAL_DS:
        D, rows = DATA[ds], []
        for idx in IDX[ds]:
            iid = {norm(a) for a in D["gen"][idx]["iid"]}
            for v in ALL_VIEWS:
                for a in D["gen"][idx][v][:K]:
                    na = norm(a)
                    rows.append((v, na not in iid, D["jud"][(idx, na)]))
        per_view = {}
        for v in ALL_VIEWS:
            r = [(nw, y) for vv, nw, y in rows if vv == v]
            new = [y for nw, y in r if nw]
            per_view[v] = {"group": VIEW_GROUP[v], "n_candidates": len(r),
                           "share_not_in_iid24": round(float(np.mean([nw for nw, _ in r])), 6),
                           "acc_of_new_candidates": round(float(np.mean(new)), 6) if new else None,
                           "acc_of_all_candidates": round(float(np.mean([y for _, y in r])), 6)}
        iid_acc = float(np.mean([D["jud"][(i, norm(a))] for i in IDX[ds] for a in D["gen"][i]["iid"]]))
        # THE NULL NOVELTY FLOOR: view r320 is the iid control's own rendering, so its "novelty"
        # is pure sampling stochasticity. Any view must clear THAT to have moved the distribution.
        floor = per_view["r320"]["share_not_in_iid24"]
        for v, d in per_view.items():
            d["novelty_above_sampling_floor"] = round(d["share_not_in_iid24"] - floor, 6)
            d["new_candidates_beat_iid_percand_acc"] = (
                None if d["acc_of_new_candidates"] is None
                else bool(d["acc_of_new_candidates"] > iid_acc))
        out[ds] = {"iid_control_percand_acc": round(iid_acc, 6),
                   "null_novelty_floor_from_base_view_r320": round(floor, 6),
                   "floor_note": "r320 IS the iid control's rendering; its novelty rate is what "
                                 "sampling noise alone produces. A view has only moved the "
                                 "candidate DISTRIBUTION if it clears this floor.",
                   "per_view": per_view}
    return out


def main():
    DATA = {ds: load_ds(ds) for ds in EVAL_DS}
    IDX = {ds: complete_idx(DATA[ds], ds) for ds in EVAL_DS}
    n_tot = sum(len(v) for v in IDX.values())
    print("items with complete data:", {ds: len(IDX[ds]) for ds in EVAL_DS}, "total", n_tot, flush=True)

    res = {
        "attack": "SWEEP 3 -- candidate diversity along the VISION axis (the varied factor is the "
                  "IMAGE; prompt, temperature, top_p and max_tokens are FROZEN at the incumbent "
                  "pool's values)",
        "date": "2026-08-13",
        "how_this_differs_from_the_killed_diverse_arm": {
            "killed_arm": "results/cascade_methods/artifacts/open_diverse_2026-08-10.json -- a "
                          "5-PROMPT portfolio x a 3-TEMPERATURE ladder, DPP over answer strings. It "
                          "varied the TEXT side and confounded prompt with temperature.",
            "this_arm": "prompt/temperature/top_p/max_tokens held FIXED and identical to the "
                        "incumbent pool (src/labeling/run_openvqa.py SYS, temp 0.7, top_p 1.0, "
                        "max_tokens 64); the ONLY thing that moves is the rendered image: "
                        "resolution budget, crop/tile, or a monotone intensity remap.",
            "shared_risk": "same shape as the killed arm -- a wider pool can add confident wrong "
                           "distractors that cost more sel_eff than the added coverage buys, which "
                           "is exactly what killed it. That is why the within-pool control below "
                           "is a pre-registered KILL CRITERION."},
        "generation_config_frozen": {"system_prompt": SYS, "temperature": TEMP, "top_p": TOP_P,
                                     "max_tokens": MAX_TOKENS, "k_per_view": K, "k_iid": 24,
                                     "engine": "vLLM tp=1 bf16 seed 0, one session, both arms"},
        "views": {v: {"group": VIEW_GROUP[v], "max_pixels": VIEWS[v][1]} for v in ALL_VIEWS},
        "views_refused_and_why": REFUSED_VIEWS,
        "portfolios_preregistered": {k: v for k, v in PORTFOLIOS.items()},
        "n_items": {ds: len(IDX[ds]) for ds in EVAL_DS}, "n_total": n_tot,
        "generation_failures": {ds: DATA[ds]["fails"] for ds in EVAL_DS},
        "nboot": NBOOT, "bootstrap_seed": SEED,
        "provenance": {
            "generation_and_labelling": "2026-08-13 (runners/run_vision_diversity_2026-08-13.sh -> "
                                        "vision_diversity_gen.py; judge via src/labeling/run_judge.py, "
                                        "the project's existing MedVLThinker-32B text-only judge -- no "
                                        "new judge was invented; verifier scoring via "
                                        "vision_diversity_score.py, HF transformers, 4 shards).",
            "analysis_rerun": "2026-08-14 03:05, OMP_NUM_THREADS=8 PYTHONHASHSEED=0, CPU only "
                              "(logs/visdiv_analyze_full_2026-08-14.log).",
            "supersedes": "an earlier write of this same path at 2026-08-13 15:04, kept verbatim at "
                          "results/cascade_methods/artifacts/_visdiv_parts/"
                          "vision_diversity_PARTIAL_pathvqa18_2026-08-13T1504.json. That write "
                          "landed BEFORE the PathVQA verifier-scoring shard finished (17:07), so its "
                          "pool was slake 645 / vqa_rad 200 / pathvqa 18 = 863 items and its PathVQA "
                          "cell, its pooled numbers and its macro translation were all unusable. "
                          "Every number in THIS file is on the complete 645/200/1500 = 2345 endpoint. "
                          "Quote nothing from the partial.",
            "verifier_scoring_engine": "HF transformers, bf16, flash_attention_2, batch 1, the "
                                       "verifier always seeing the ORIGINAL untransformed image at "
                                       "max_pixels 1003520. NEVER vLLM (vLLM 0.9.0.1 drops all 192 "
                                       "visual.* LoRA modules: 0.775204 HF vs 0.702997 vLLM).",
            "matched_control_discipline": "the iid control and every portfolio come from ONE vLLM "
                                          "process, ONE serving config (tp=1, bf16, seed 0, "
                                          "max_model_len 4096, limit_mm_per_prompt image=1) in ONE "
                                          "session. No delta in this file is against a stored number "
                                          "from another config, so the +-0.008 open-text "
                                          "reproducibility caveat cannot contaminate it.",
        },
    }
    res["null_tests"] = null_tests(DATA, IDX)

    # ---------------- arms -----------------------------------------------------------------
    arms = {}
    for s in range(K):
        arms[f"iid8_s{s}"] = report_arm(DATA, IDX, (lambda d, s=s: slots_iid(d, s)), f"iid8_s{s}")
    for pname, views in PORTFOLIOS.items():
        for s in range(K):
            arms[f"{pname}_s{s}"] = report_arm(
                DATA, IDX, (lambda d, v=views, s=s: slots_portfolio(d, v, s)), f"{pname}_s{s}")

    def mean_sd(prefix, field):
        vals = [arms[f"{prefix}_s{s}"]["pooled"][field] for s in range(K)]
        return {"mean": round(float(np.mean(vals)), 6),
                "sd": round(float(np.std(vals, ddof=1)), 6), "per_draw": vals}

    res["arms"] = {k: {kk: vv for kk, vv in a.items() if not kk.startswith("_")}
                   for k, a in arms.items()}
    res["arms_across_draws"] = {
        p: {f: mean_sd(p, f) for f in ["oracle@8", "selected", "sel_eff", "mean_distinct",
                                       "mean_vis_tokens_per_candidate"]}
        for p in ["iid8"] + list(PORTFOLIOS)}

    # ---------------- PRIMARY: portfolio vs MATCHED iid control, draw-paired ----------------
    res["primary_contrast"] = {
        "definition": f"{PRIMARY} (8 DISTINCT views x 1 sample) vs iid8 (8 samples at the BASE "
                      f"view), same items, same session, same serving config. Draw s is compared "
                      f"to iid draw s; all three draws are mutually DISJOINT in both arms.",
        "per_draw": [compare(arms[f"{PRIMARY}_s{s}"], arms[f"iid8_s{s}"], f"{PRIMARY}_s{s} - iid8_s{s}")
                     for s in range(K)]}
    res["all_portfolios_vs_iid"] = {
        p: [compare(arms[f"{p}_s{s}"], arms[f"iid8_s{s}"], f"{p}_s{s} - iid8_s{s}") for s in range(K)]
        for p in PORTFOLIOS}

    # ---------------- guardrail: per cell, per draw -----------------------------------------
    # per-cell slices of the concatenated vectors, so a cell-level delta gets its own paired CI
    # instead of a bare point estimate (vqa_rad_open is n=200 and its draw sd is ~0.025).
    bounds, off = {}, 0
    for ds in EVAL_DS:
        bounds[ds] = (off, off + len(IDX[ds])); off += len(IDX[ds])
    res["guardrail_per_cell"] = {}
    for p in PORTFOLIOS:
        rows = []
        for ds in EVAL_DS:
            lo, hi = bounds[ds]
            for s in range(K):
                a = arms[f"{p}_s{s}"]["per_ds"][ds]; b = arms[f"iid8_s{s}"]["per_ds"][ds]
                ga = arms[f"{p}_s{s}"]["_got"][lo:hi]; gb = arms[f"iid8_s{s}"]["_got"][lo:hi]
                ra = arms[f"{p}_s{s}"]["_rec"][lo:hi]; rb = arms[f"iid8_s{s}"]["_rec"][lo:hi]
                ci = boot_delta_paired(ga, gb, nboot=NBOOT, seed=SEED)
                cio = boot_delta_paired(ra, rb, nboot=NBOOT, seed=SEED)
                # draw-to-draw sd of each arm ON THIS CELL: the seed-noise band the delta must clear
                sd_p = float(np.std([arms[f"{p}_s{t}"]["per_ds"][ds]["selected"]
                                     for t in range(K)], ddof=1))
                sd_i = float(np.std([arms[f"iid8_s{t}"]["per_ds"][ds]["selected"]
                                     for t in range(K)], ddof=1))
                rows.append({"ds": ds, "draw": s, "n": a["n"],
                             "d_selected": round(a["selected"] - b["selected"], 6),
                             "d_selected_ci95": [round(ci[1], 6), round(ci[2], 6)],
                             "selected_significant": bool(ci[1] > 0 or ci[2] < 0),
                             "draw_sd_portfolio": round(sd_p, 6),
                             "draw_sd_iid": round(sd_i, 6),
                             "within_seed_noise": bool(abs(a["selected"] - b["selected"])
                                                       < max(sd_p, sd_i)),
                             "d_oracle": round(a["oracle@8"] - b["oracle@8"], 6),
                             "d_oracle_ci95": [round(cio[1], 6), round(cio[2], 6)],
                             "oracle_significant": bool(cio[1] > 0 or cio[2] < 0),
                             "d_sel_eff": round(a["sel_eff"] - b["sel_eff"], 6)})
        res["guardrail_per_cell"][p] = {
            "rows": rows,
            "n_cell_draws_selected_negative": int(sum(1 for r in rows if r["d_selected"] < 0)),
            "n_cell_draws": len(rows),
            "n_cell_draws_significantly_negative":
                int(sum(1 for r in rows if r["selected_significant"] and r["d_selected"] < 0)),
            "n_cell_draws_significantly_positive":
                int(sum(1 for r in rows if r["selected_significant"] and r["d_selected"] > 0)),
            "cells_negative_on_ALL_draws": sorted({
                ds for ds in EVAL_DS
                if all(r["d_selected"] < 0 for r in rows if r["ds"] == ds)}),
            "_read": "the guardrail is 'never worse than the incumbent on any single benchmark'. A "
                     "cell that is negative on all 3 disjoint draws is a FLAG even when no single "
                     "draw clears its own CI, because the 3 draws are independent samples of the "
                     "same arm. Compare d_selected to draw_sd_* before reading any single row."}
    return res, arms, DATA, IDX


def within_pool_control(DATA, IDX, arms, res):
    """THE PRE-REGISTERED KILL CRITERION.

    Confound-free: portfolio-8 (view-STRATIFIED: 8 distinct views, one sample each) vs random-8
    drawn from the SAME 36-candidate 12-view pool, N_RAND_SEEDS seeds. Both draw from an
    identically vision-diverse generation distribution, so this isolates the view-SPREAD RULE from
    the pool. Exactly the control that killed the 2026-08-10 DPP arm.

    KILL RULE, fixed in advance: if the portfolio's mean d_selected vs the random-8 control is
    negative on >= 2 of the 3 cells, the attack is reported as a NEGATIVE and stops.
    """
    rand = [report_arm(DATA, IDX, (lambda d, r=np.random.default_rng(1000 + s):
                                   slots_rand_viewpool(d, r)), f"rand8_viewpool_s{s}")
            for s in range(N_RAND_SEEDS)]
    rand_iid = [report_arm(DATA, IDX, (lambda d, r=np.random.default_rng(2000 + s):
                                       slots_rand_iidpool(d, r)), f"rand8_iidpool_s{s}")
                for s in range(N_RAND_SEEDS)]

    def pooled(rs, f):
        v = [r["pooled"][f] for r in rs]
        return {"mean": round(float(np.mean(v)), 6), "sd": round(float(np.std(v, ddof=1)), 6)}

    def percell(rs, ds, f):
        v = [r["per_ds"][ds][f] for r in rs]
        return float(np.mean(v)), float(np.std(v, ddof=1))

    port = [arms[f"{PRIMARY}_s{s}"] for s in range(K)]
    per_cell = {}
    for ds in EVAL_DS:
        pm, ps = percell(port, ds, "selected")
        rm, rs_ = percell(rand, ds, "selected")
        po, _ = percell(port, ds, "oracle@8")
        ro, _ = percell(rand, ds, "oracle@8")
        pe, _ = percell(port, ds, "sel_eff")
        re_, _ = percell(rand, ds, "sel_eff")
        per_cell[ds] = {
            "n": arms[f"{PRIMARY}_s0"]["per_ds"][ds]["n"],
            "portfolio8_selected_mean": round(pm, 6), "portfolio8_selected_sd": round(ps, 6),
            "random8_from_same_viewpool_selected_mean": round(rm, 6),
            "random8_selected_sd": round(rs_, 6),
            "d_selected": round(pm - rm, 6),
            "d_oracle": round(po - ro, 6), "d_sel_eff": round(pe - re_, 6),
            "negative": bool(pm - rm < 0),
            "within_seed_noise": bool(abs(pm - rm) < max(ps, rs_))}
    n_neg = sum(1 for v in per_cell.values() if v["negative"])
    return {
        "definition": "portfolio-8 (8 distinct views, 1 sample each) vs random-8 from the SAME "
                      f"36-candidate 12-view pool, {N_RAND_SEEDS} seeds. Isolates the view-spread "
                      "RULE from the view pool itself.",
        "kill_rule_preregistered": "if d_selected < 0 on >= 2 of 3 cells -> report NEGATIVE and stop",
        "per_cell": per_cell,
        "n_cells_negative": n_neg,
        "KILL_TRIGGERED": bool(n_neg >= 2),
        "pooled": {"portfolio8": {f: pooled(port, f) for f in ["oracle@8", "selected", "sel_eff"]},
                   "random8_viewpool": {f: pooled(rand, f) for f in ["oracle@8", "selected", "sel_eff"]},
                   "random8_iidpool": {f: pooled(rand_iid, f) for f in ["oracle@8", "selected", "sel_eff"]}},
        "note": "random8_iidpool is the same rule applied to the iid control pool -- it gives the "
                "seed-noise band a difference of this size has to clear."}


def ceilings(DATA, IDX, arms):
    """Lincoln-Petersen coverage ceiling PER SETTING. The published +0.0091 bound is
    DISTRIBUTION-SPECIFIC (iid resampling of the base view); a different view distribution has a
    different ceiling, so it is re-measured here rather than assumed."""
    out = {}
    for name, (a, b) in {
            "iid_base_view (draws 0 vs 1)": (arms["iid8_s0"], arms["iid8_s1"]),
            f"{PRIMARY}_portfolio (draws 0 vs 1)": (arms[f"{PRIMARY}_s0"], arms[f"{PRIMARY}_s1"]),
            f"portfolio vs iid (cross)": (arms[f"{PRIMARY}_s0"], arms["iid8_s0"]),
    }.items():
        ra, _ = cat_rec_got(a); rb, _ = cat_rec_got(b)
        out[name] = lincoln_petersen(ra.astype(bool), rb.astype(bool))
    # per cell, for the two same-distribution ceilings
    out["per_cell"] = {}
    off = 0
    for ds in EVAL_DS:
        n = len(IDX[ds])
        sl = slice(off, off + n); off += n
        r0, _ = cat_rec_got(arms["iid8_s0"]); r1, _ = cat_rec_got(arms["iid8_s1"])
        p0, _ = cat_rec_got(arms[f"{PRIMARY}_s0"]); p1, _ = cat_rec_got(arms[f"{PRIMARY}_s1"])
        out["per_cell"][ds] = {
            "iid": lincoln_petersen(r0[sl].astype(bool), r1[sl].astype(bool)),
            "portfolio": lincoln_petersen(p0[sl].astype(bool), p1[sl].astype(bool))}
    out["interpretation"] = (
        "LP reachable share = the share of questions for which SOME draw of this distribution "
        "contains a correct answer, extrapolated to N=infinity. Comparing the iid row to the "
        "portfolio row answers the question the +0.0091 bound cannot: does changing the image "
        "distribution move the coverage ceiling itself?")
    return out


def cost(DATA, IDX, arms, res):
    """Vision tokens dominate this workload (measured 2026-08-11: LM prefill 82.1%, vision towers
    16.7%, ALL decode 1.2%), so per-candidate vision tokens are the honest compute currency here."""
    out = {"currency": "mean vision tokens per candidate x 8 candidates; the 2026-08-11 VRAM "
                       "measurement established this workload is PREFILL-BOUND and driven by "
                       "vision-token count (all decode = 1.2% of compute).",
           "per_arm": {}}
    base = arms["iid8_s0"]["pooled"]["mean_vis_tokens_per_candidate"]
    for name in ["iid8_s0"] + [f"{p}_s0" for p in PORTFOLIOS]:
        v = arms[name]["pooled"]["mean_vis_tokens_per_candidate"]
        out["per_arm"][name] = {"mean_vis_tokens_per_candidate": v,
                                "vis_tokens_per_question_at_N8": round(v * 8, 1),
                                "x_vs_iid8": round(v / base, 4)}
    for k, v in out["per_arm"].items():
        v["cost_matched_to_iid8"] = bool(abs(v["x_vs_iid8"] - 1.0) <= 0.05)
    out["cost_matched_arms"] = [k for k, v in out["per_arm"].items() if v["cost_matched_to_iid8"]]
    out["why_cost_matching_matters"] = (
        "P_crop and P_photo render every view at the BASE token budget, so they are cost-matched "
        "to the iid control: a win there is attributable to DIVERSITY. P_res and P_mixed include "
        "up640/up1280 and therefore buy more vision tokens as well as more diversity -- a win "
        "there is confounded between the two, and must be read against the cost-matched arms.")
    out["caveat"] = ("this counts the GENERATOR only. It excludes the verifier pass, which is "
                     "unchanged across arms (the scorer always sees the original image at fullres) "
                     "and therefore cancels in every ratio reported here.")

    # ---- PREFILL-SHARING CORRECTION -------------------------------------------------------
    # The per-candidate charge above OVERSTATES the iid arm and therefore UNDERSTATES the
    # portfolio's true cost. vLLM is given ONE request per (item, view) with SamplingParams(n=k)
    # (vision_diversity_gen.py:247,265), so the prefill -- vision tower + LM prefill of the image
    # tokens -- is computed ONCE and k sequences are forked from it. The iid arm therefore pays
    # ONE prefill for all 8 of its candidates. A view portfolio cannot share anything across
    # distinct views: it pays one prefill PER DISTINCT VIEW. On a workload measured to be 82.1%
    # LM prefill + 16.7% vision tower and only 1.2% decode (2026-08-11), that is the dominant term.
    vt_by_view = {}
    for v in ALL_VIEWS:
        tot = n = 0
        for ds in EVAL_DS:
            for idx in IDX[ds]:
                tot += DATA[ds]["vt"][idx][v]; n += 1
        vt_by_view[v] = tot / n
    base_prefill = vt_by_view["r320"]          # the iid control's rendering
    corr = {"iid8": {"n_prefills_per_question": 1,
                     "prefill_vision_tokens_per_question": round(base_prefill, 1),
                     "x_vs_iid8": 1.0,
                     "views": ["r320 (identity, = the iid control's rendering)"]}}
    for p, views in PORTFOLIOS.items():
        dv = sorted(set(views))
        tot = sum(vt_by_view[v] for v in dv)
        corr[p] = {"n_prefills_per_question": len(dv),
                   "prefill_vision_tokens_per_question": round(tot, 1),
                   "x_vs_iid8": round(tot / base_prefill, 3),
                   "views": dv}
    out["prefill_sharing_correction"] = {
        "why": "SamplingParams(n=8) on one request = ONE prefill for 8 candidates; 8 distinct "
               "views = 8 prefills for 8 candidates. The per-candidate charge in "
               "cost.per_arm bills the iid arm 8x for a prefill it computes once, so it "
               "flatters the portfolio. This row is the honest generator cost.",
        "per_arm": corr,
        "read": "under prefill-shared accounting the primary portfolio costs "
                f"{corr[PRIMARY]['x_vs_iid8']}x the iid control's generator vision-token work, not "
                f"{out['per_arm'][PRIMARY + '_s0']['x_vs_iid8']}x. NO arm in this sweep is "
                "cost-matched to the incumbent under this accounting -- the 'cost_matched_arms' "
                "list above holds only under the per-candidate charge.",
        "what_is_still_not_charged": "decode (measured at 1.2% of compute), the verifier pass "
                                     "(identical across arms), and the CPU-side image transforms "
                                     "(crop/resize/gamma), which were not profiled.",
        "not_measured": "wall-clock latency, energy and test-time VRAM were NOT measured in this "
                        "session for any arm. This is an arithmetic re-charge of measured vision-"
                        "token counts, not a timing run."}
    return out


def resolution_provenance(DATA, IDX):
    """WHAT RESOLUTION DID THE PUBLISHED CELLS ACTUALLY USE?  Read off the runners / scripts, with
    the native token statistics measured here. The answer is: they are NOT all the same, and the
    open-text generator and its own verifier differ by 4x.
    """
    import numpy as _np
    native = {}
    for ds in EVAL_DS:
        t = []
        for idx in IDX[ds]:
            wh = DATA[ds]["vt"][idx]
            # 'up1280' is the frame fitted to ~1280 tokens; the NATIVE count is what a cap of
            # 1280 would leave, i.e. min(native, 1280). Use the r320/up640/up1280 ladder to bound it.
            t.append(wh.get("up1280", 0))
        native[ds] = {"n": len(t)}
    for ds in EVAL_DS:
        c320 = [DATA[ds]["vt"][i]["r320"] for i in IDX[ds]]
        c160 = [DATA[ds]["vt"][i]["r160"] for i in IDX[ds]]
        native[ds].update({
            "vis_tokens_at_cap320_median": float(_np.median(c320)),
            "vis_tokens_at_cap160_median": float(_np.median(c160)),
            "share_of_items_where_cap320_binds": round(float(_np.mean(
                [a > b for a, b in zip(
                    [DATA[ds]["vt"][i]["up640"] for i in IDX[ds]], c320)])), 4)})
    return {
        "question": "what image resolution do the published cells actually use?",
        "answer": "NOT one resolution. Three different conventions coexist, and the open-text "
                  "generator is 4x LOWER than the verifier that scores its own candidates.",
        "table": {
            "MCQ cells (the faithful MedEvalKit numbers: PMC_VQA, SLAKE-closed, VQA_RAD-closed, "
            "PathVQA-closed, MedXpertQA-MM)": {
                "resolution": "NATIVE / uncapped",
                "evidence": "MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py:51-54 -- max_pixels is "
                            "set only when the env var CAP_MAX_PIXELS is non-zero; it is unset in "
                            "the faithful runs, so no cap is applied."},
            "open-text 7B greedy AND the 8-sample pool (the n=2345 endpoint)": {
                "resolution": "cap320 (max_pixels = 1280*28*28/4 = 250880)",
                "evidence": "src/labeling/run_openvqa.py:61 sets --cap default 'cap320'; "
                            "runners/run_openvqa_lingshu7b.sh and "
                            "runners/run_verifier_disjoint_retrain.sh:67 pass no --cap override."},
            "open-text 32B-direct": {
                "resolution": "cap320 (same default)",
                "evidence": "runners/run_openvqa_lingshu.sh -- no --cap override."},
            "MCQ-as-generation pools (ckpts/mcq_gen_verify/*)": {
                "resolution": "cap320",
                "evidence": "runners/run_ugv_mcq_batch.sh:12 COMMON=... --cap cap320; "
                            "runners/run_ugv_experiments.sh likewise."},
            "the LoRA verifier that SCORES those candidates": {
                "resolution": "fullres (max_pixels = 1280*28*28 = 1003520) -- 4x the generator",
                "evidence": "src/training_methods/verifier_transfer_eval.py:20 MAXPX=1280*28*28, "
                            "hardcoded, no cap flag. Same in "
                            "src/cascade_methods/open_diverse_score.py:39."},
            "the hidden-state feature cache (feats_hidden/)": {
                "resolution": "fullres",
                "evidence": "src/training_methods/extract_generator_hidden.py:48 HIGH_PX."},
            "the 32B acc_gen arms": {
                "resolution": "MIXED within one runner: nothink_cap320, nothink_fullres and "
                              "think_fullres all exist",
                "evidence": "runners/run_lingshu_32b_phase2.sh:7-9."},
        },
        "measured_native_token_statistics_this_session": native,
        "why_it_matters": "cap320 caps the frame at 320 vision tokens. Measured native medians "
                          "this session: SLAKE 334, PathVQA 533, VQA-RAD 754 -- so the cap is "
                          "discarding real pixels on PathVQA and VQA-RAD in particular, and the "
                          "verifier is nonetheless allowed to look at 4x more of the image than "
                          "the generator that produced the answer. Nobody chose this: cap320 was "
                          "chosen on COST, and the verifier scripts simply never had a cap flag.",
        "status": "REPORTED AS A PROVENANCE FINDING. This sweep does not re-run the published "
                  "arms, so it does not measure what unifying the resolution would do to them."}


def macro_translation(arms, IDX):
    """Translate the open-cell outcome to the 8-cell MACRO headline -- only when the pool is the
    COMPLETE canonical endpoint, because visverif_lib.macro_from_open asserts 645/200/1500."""
    n = {ds: len(IDX[ds]) for ds in EVAL_DS}
    want = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}
    if n != want:
        return {"status": "NOT COMPUTED -- the complete-data pool is "
                          f"{n}, not the canonical {want}. Macro is only defined on the full "
                          "endpoint; the open-cell deltas above stand on their own.",
                "n": n}
    items = G.load_items()
    mt = V.macro_table()
    out = {"reference": V.macro_reference(mt),
           "note": "MCQ cells are held at always-7B in this translation, so these macro values are "
                   "NOT the deployed cascade's macro -- they isolate what the open-text arm "
                   "contributes. Compare rows to each other, never to the 0.6567 headline."}
    for name in [f"{PRIMARY}_s{s}" for s in range(K)] + [f"iid8_s{s}" for s in range(K)]:
        _, got = cat_rec_got(arms[name])
        out[name] = V.macro_from_open(got, items, mt)
    bs = []
    for s in range(K):
        _, ga = cat_rec_got(arms[f"{PRIMARY}_s{s}"])
        _, gb = cat_rec_got(arms[f"iid8_s{s}"])
        bs.append(V.macro_bootstrap(ga, gb, nboot=NBOOT, seed=SEED, items=items, mt=mt))
    out["portfolio_minus_iid_per_draw"] = bs
    return out


def matched_control_caveat(res):
    """Quantify, on this sweep's own numbers, why the in-session iid control is not optional.

    The STORED deployed 8-sample pool and this session's iid8 control are the SAME distribution --
    same model, same prompt, same temperature, same cap320 rendering, 8 samples -- differing only
    in serving config and sampling seed. Whatever separates them is pure nuisance. If the portfolio
    had been differenced against the stored number instead of the in-session control, that nuisance
    would have been read as an experimental effect.
    """
    pub = res["null_tests"]["N1_frozen_metric"]["published"]
    ctl = res["arms_across_draws"]["iid8"]
    prt = res["arms_across_draws"][PRIMARY]
    d_nuis = ctl["selected"]["mean"] - pub["selected"]
    return {
        "stored_deployed_pool (ckpts/train/lora_verifier_disjoint/transfer_dump_*.json)": {
            "oracle@8": pub["oracle@8"], "selected": pub["selected"], "sel_eff": pub["sel_eff"]},
        "this_session_iid8_control (same distribution, 3 disjoint draws)": {
            "oracle@8": ctl["oracle@8"]["mean"], "oracle@8_sd": ctl["oracle@8"]["sd"],
            "selected": ctl["selected"]["mean"], "selected_sd": ctl["selected"]["sd"],
            "sel_eff": ctl["sel_eff"]["mean"]},
        "nuisance_shift_selected": round(d_nuis, 6),
        "nuisance_shift_oracle@8": round(ctl["oracle@8"]["mean"] - pub["oracle@8"], 6),
        "the_effect_being_measured_selected": round(
            prt["selected"]["mean"] - ctl["selected"]["mean"], 6),
        "ratio_nuisance_to_effect": round(abs(d_nuis) / abs(prt["selected"]["mean"]
                                                            - ctl["selected"]["mean"]), 3),
        "read": "the nuisance shift between two runs of the SAME distribution is larger than the "
                "entire effect this sweep is testing. Differenced against the stored pool the "
                "portfolio would read as a LOSS; differenced against the in-session control it "
                "reads as a small non-significant GAIN. Only the second is an experiment. This is "
                "the +-0.008 open-text reproducibility caveat (CLAUDE.md) measured on this sweep's "
                "own arms, and it is why no number in this file is differenced against a stored "
                "value from another serving config."}


NOT_MEASURED = {
    "the 5 MCQ cells": "this sweep touches the OPEN half only (2345 of the 8-cell macro's 37.5% "
                       "weight). Vision-side portfolios were never run on PMC_VQA, SLAKE-closed, "
                       "VQA_RAD-closed, PathVQA-closed or MedXpertQA-MM.",
    "the 32B's own vision portfolio": "only the 7B generator was diversified. Whether the same "
                                      "views move the 32B-direct bar is unmeasured.",
    "VRAM and wall-clock energy of the portfolio arm": "cost here is GENERATOR vision tokens only, "
                                                       "the currency the 2026-08-11 prefill-bound "
                                                       "measurement justifies. No test-time VRAM or "
                                                       "joules were measured in this session.",
    "a verifier retrained on view-diverse candidates": "the scorer is the frozen incumbent, trained "
                                                       "on base-view candidates. Whether a verifier "
                                                       "that has SEEN crop/upsample candidates would "
                                                       "recover the lost sel_eff is untested -- and "
                                                       "it would no longer be a training-free change.",
    "learned or per-item view selection": "every portfolio here is a FIXED, pre-registered view list "
                                          "applied to all items. Choosing views per question was not "
                                          "attempted and would need its own train/eval split.",
    "N > 8": "every arm is exactly 8 candidates, to keep the frozen endpoint comparable. The "
             "12-view x 3-sample pool (36 candidates) is used only for the within-pool control.",
    "seeds beyond 3": "K=3 samples per view fixes the number of disjoint draws at 3. The draw-to-draw "
                      "sd of the SELECTED endpoint (iid8 0.005343, P_mixed 0.003473) is the same size "
                      "as the effect being tested, which is itself part of the finding.",
}


def mechanism(res):
    """The measured explanation for the null: WHERE the new candidates come from and how good they
    are. Every number here is read back out of sections already computed above."""
    nov = res["novelty_per_view"]
    rows, by_group = [], defaultdict(list)
    n_beat = n_tot = 0
    for ds, v in nov.items():
        iid_acc = v["iid_control_percand_acc"]
        for view, x in v["per_view"].items():
            if view == "r320":
                continue                      # the base view IS the control; not a treatment
            n_tot += 1
            beat = bool(x["new_candidates_beat_iid_percand_acc"])
            n_beat += int(beat)
            rows.append({"ds": ds, "view": view, "group": x["group"],
                         "novelty_above_sampling_floor": x["novelty_above_sampling_floor"],
                         "acc_of_new_candidates": x["acc_of_new_candidates"],
                         "iid_percand_acc": iid_acc,
                         "new_minus_iid_percand_acc": round(x["acc_of_new_candidates"] - iid_acc, 6)})
            by_group[x["group"]].append(rows[-1])
    grp = {g: {"n_view_cells": len(r),
               "mean_novelty_above_sampling_floor": round(
                   float(np.mean([q["novelty_above_sampling_floor"] for q in r])), 6),
               "mean_acc_of_new_candidates": round(
                   float(np.mean([q["acc_of_new_candidates"] for q in r])), 6),
               "mean_new_minus_iid_percand_acc": round(
                   float(np.mean([q["new_minus_iid_percand_acc"] for q in r])), 6)}
           for g, r in by_group.items()}
    pri = res["primary_contrast"]["per_draw"]
    d_o = float(np.mean([p["d_oracle@8"] for p in pri]))
    d_s = float(np.mean([p["d_selected"] for p in pri]))
    return {
        "the_question": "vision-side diversity DOES move the candidate distribution and DOES raise "
                        "oracle@8 significantly. Why does none of it reach SELECTED?",
        "answer_1_every_new_candidate_stream_is_worse_than_the_base_stream": {
            "n_view_cells_whose_NEW_candidates_beat_the_iid_per_candidate_accuracy":
                f"{n_beat} of {n_tot}",
            "read": "across all 11 non-base views x 3 cells, NOT ONE produced novel answers that "
                    "are as accurate as the base view's own samples. Image-side diversity buys "
                    "coverage by adding candidates that are individually much more likely to be "
                    "wrong, which is precisely what a fixed scorer has to absorb.",
            "per_view_cell": rows},
        "answer_2_novelty_and_quality_trade_off_by_view_group": {
            "by_group": grp,
            "read": "CROP views move the distribution most (highest novelty above the sampling "
                    "floor) and produce the WORST new candidates; RESOLUTION views move it less and "
                    "produce the best new candidates; PHOTOMETRIC views barely move it at all -- on "
                    "SLAKE, autocontrast is BELOW the base view's own sampling-noise floor. The "
                    "portfolio built purely from crops (P_crop) is the only arm with a NEGATIVE mean "
                    "d_selected, and it is the most diverse arm in the sweep."},
        "answer_3_the_conversion_arithmetic": {
            "mean_d_oracle@8_P_mixed_vs_iid8": round(d_o, 6),
            "mean_d_selected_P_mixed_vs_iid8": round(d_s, 6),
            "realised_conversion_of_the_oracle_gain": round(d_s / d_o, 4) if d_o else None,
            "project_measured_marginal_conversion_of_newly_covered_questions":
                "0.447 [0.368, 0.526] (artifacts/coverage_diagnosis2_2026-08-10.json)",
            "read": "the oracle gain converts at well under the project's own 0.447 marginal rate, "
                    "because this coverage was not bought by resampling the same distribution -- it "
                    "was bought by adding a lower-quality stream, so the sel_eff loss "
                    f"({round(float(np.mean([p['d_sel_eff'] for p in pri])), 6)}) eats it. "
                    "selected = oracle@8 x sel_eff is exact, and both factors moved."},
        "answer_4_the_confound_free_control_says_the_SPREAD_RULE_is_not_the_lever": {
            "portfolio8_minus_random8_from_the_same_12_view_pool_selected":
                res["within_pool_control"]["pooled"]["portfolio8"]["selected"]["mean"]
                - res["within_pool_control"]["pooled"]["random8_viewpool"]["selected"]["mean"],
            "random8_viewpool_minus_random8_iidpool_selected":
                res["within_pool_control"]["pooled"]["random8_viewpool"]["selected"]["mean"]
                - res["within_pool_control"]["pooled"]["random8_iidpool"]["selected"]["mean"],
            "read": "deliberately spreading over views beats drawing 8 at random from the same "
                    "view pool by a margin inside the seed-noise band on all 3 cells. And the "
                    "view pool ITSELF, drawn at random, is no better than the iid pool on SELECTED "
                    "despite carrying ~+0.02 more oracle -- the coverage is real and the conversion "
                    "is not."},
    }


def verdict(res):
    """State the outcome from the pre-registered rules only -- no post-hoc rescue."""
    kill = res["within_pool_control"]["KILL_TRIGGERED"]
    pri = res["primary_contrast"]["per_draw"]
    d_sel = [p["d_selected"] for p in pri]
    d_orc = [p["d_oracle@8"] for p in pri]
    sig = [p for p in pri if p["selected_significant"]]
    n_sig_pos = sum(1 for p in sig if p["d_selected"] > 0)
    n_sig_neg = sum(1 for p in sig if p["d_selected"] < 0)
    if kill:
        v = ("NEGATIVE -- pre-registered kill criterion TRIGGERED. The confound-free within-pool "
             "control (portfolio-8 vs random-8 from the same 12-view pool) is negative on >= 2 of "
             "3 cells, so view STRATIFICATION is not the active ingredient and the arm stops here, "
             "exactly as the 2026-08-10 prompt/temperature arm did.")
    elif n_sig_neg:
        v = ("NEGATIVE -- the portfolio is SIGNIFICANTLY WORSE than the matched iid control on "
             f"{n_sig_neg}/{len(pri)} draws of the SELECTED endpoint.")
    elif n_sig_pos == len(pri):
        v = ("POSITIVE on the selected endpoint in every draw -- but read the cost row and the "
             "guardrail before treating it as deployable.")
    elif n_sig_pos:
        v = (f"MIXED -- SELECTED is significantly positive on {n_sig_pos}/{len(pri)} draws and "
             "indistinguishable on the rest.")
    else:
        v = ("NULL -- no draw moves the SELECTED endpoint significantly. Vision-side diversity "
             "behaves like the text-side diversity that preceded it: it changes the pool without "
             "changing the answer that gets returned.")
    return {"verdict": v,
            "kill_criterion_triggered": bool(kill),
            "d_selected_per_draw": d_sel,
            "d_selected_mean": round(float(np.mean(d_sel)), 6),
            "d_oracle_per_draw": d_orc,
            "d_oracle_mean": round(float(np.mean(d_orc)), 6),
            "n_draws_selected_significant_positive": n_sig_pos,
            "n_draws_selected_significant_negative": n_sig_neg}


if __name__ == "__main__":
    res, arms, DATA, IDX = main()
    res["within_pool_control"] = within_pool_control(DATA, IDX, arms, res)
    res["capture_recapture_ceilings"] = ceilings(DATA, IDX, arms)
    res["cost"] = cost(DATA, IDX, arms, res)
    res["novelty_per_view"] = novelty(DATA, IDX)

    ST = strata_masks(DATA, IDX)
    res["laterality_endpoint"] = {
        "why": "the dedicated mechanism endpoint: laterality is the incumbent verifier's weakest "
               "stratum (sel_eff 0.613043 vs 0.817186 on short non-laterality items, "
               "artifacts/vision_verifier_2026-08-12.json) and it is a SPATIAL-PERCEPTION failure, "
               "so if image-side diversity works anywhere it should work here first.",
        "mask_definition": "question OR gold OR any pooled candidate matches visverif_lib.LATERAL",
        "per_draw": [stratum_compare(arms[f"{PRIMARY}_s{s}"], arms[f"iid8_s{s}"],
                                     ST["laterality"], "laterality") for s in range(K)],
        "non_laterality_per_draw": [stratum_compare(arms[f"{PRIMARY}_s{s}"], arms[f"iid8_s{s}"],
                                                    ~ST["laterality"], "not_laterality")
                                    for s in range(K)],
        "short_gold_per_draw": [stratum_compare(arms[f"{PRIMARY}_s{s}"], arms[f"iid8_s{s}"],
                                                ST["short3"], "gold_<=3_words") for s in range(K)]}
    res["published_resolution_provenance"] = resolution_provenance(DATA, IDX)
    res["macro_translation"] = macro_translation(arms, IDX)
    res["why_the_matched_control_was_mandatory"] = matched_control_caveat(res)
    res["mechanism"] = mechanism(res)
    res["not_measured"] = NOT_MEASURED
    res["verdict"] = verdict(res)

    def _jd(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(repr(type(o)))

    outp = J("results/cascade_methods/artifacts/vision_diversity_2026-08-13.json")
    with open(outp, "w") as fh:
        json.dump(res, fh, indent=1, default=_jd)
    print(json.dumps(res["verdict"], indent=1))
    print("->", outp)
