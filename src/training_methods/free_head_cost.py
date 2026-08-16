#!/usr/bin/env python3
"""free_head_cost.py -- BUILD 1, part C (CPU only): what the head actually costs, and what freeing
it saves, in FLOP-eq against the NEW baseline (always-7B = 1.0).

Two accounting views, both reported:

  AS-CHARGED (the project's primary convention, cost_decomposition_2026-08-12.json:Q0):
      every 7B forward -- generation, verifier, feature extraction -- is charged one full batch-1
      Lingshu-7B pass.  Under it the head costs 3.813646 FLOP-eq/question.

  MEASURED-GEOMETRY (this script): rebuild the repo's FLOP model
      (flop_ratio_derivation_2026-08-03.json:flop_model) and evaluate it at the MEASURED token
      geometry of each pass instead of at one shared operating point.  This matters here because
      the passes are NOT the same size:
        * generation ran at cap320  (run_openvqa.py --cap default -> max_pixels = 1280*28*28//4)
        * the head's feature extraction ran at FULLRES (extract_generator_hidden.HIGH_PX)
        * the LoRA verifier ran at FULLRES (cheapleg_score_open.py MAXPX = 1280*28*28)
      so an extraction pass is strictly MORE expensive than a generation pass, and the as-charged
      convention UNDER-states what the head costs.

  N1 null test: rebuild reproduces the published per-component GFLOPs at the published operating
  point.  Deviation is reported; nothing downstream runs if it fails.

Token geometry is MEASURED, not modelled:
  generation prompt tokens / image tokens  <- feats_free/free_cap320_L21.rows.jsonl (this round)
  extraction prompt tokens / image tokens  <- feats_hidden/generator_eval_s*of2.meta.json n_tok
  verifier   prompt tokens / image tokens  <- feats_hidden/grader_eval_s*of2.meta.json  n_tok
  generated tokens                          <- ckpts/openvqa/cheap_lingshu7b/*_sc8.jsonl gen_tokens_all
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G   # noqa: E402

ROOT = G.ROOT
DERIV = os.path.join(ROOT, "results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json")
POOL_DIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")

D = json.load(open(DERIV))
P7 = D["parameter_counts"]["lingshu_7b"]["by_role"]
N_VIT, N_MERGER = P7["vision_tower"], P7["vision_merger"]
N_BODY, N_HEAD = P7["lm_body"], P7["lm_head"]
L_LM, D_LM = 28, 3584                 # Lingshu-7B / Qwen2.5-VL-7B text stack
D_VIT, VIT_FULL, VIT_LAYERS, VIT_WIN = 1280, 4, 32, 64


def flops_7b(T, M, Gtok, lm_head_all_positions=False):
    """GFLOP of ONE batch-1 Lingshu-7B forward, from flop_ratio_derivation_2026-08-03:flop_model.

    T     total prompt tokens (text + M image tokens)
    M     merged image tokens; pre-merge patches P = 4*M
    Gtok  generated tokens (0 = a pure prefill / scoring pass)
    lm_head_all_positions  HF computes logits at every position unless told otherwise; charging it
                           is the IMPLEMENTATION cost, not charging it is the ideal cost.
    """
    P = 4.0 * M
    nwin = P / VIT_WIN
    vit_dense = 2 * P * N_VIT
    vit_attn = 4 * (P ** 2) * D_VIT * VIT_FULL + 4 * (VIT_WIN ** 2) * D_VIT * nwin * (VIT_LAYERS - VIT_FULL)
    merger = 2 * M * N_MERGER
    pre_dense = 2 * T * N_BODY
    pre_attn = 2 * L_LM * (T ** 2) * D_LM
    dec_dense = 2 * max(Gtok - 1, 0) * N_BODY
    dec_attn = 4 * L_LM * D_LM * T * max(Gtok - 1, 0)
    head = 2 * (T if lm_head_all_positions else max(Gtok, 0)) * N_HEAD
    tot = vit_dense + vit_attn + merger + pre_dense + pre_attn + dec_dense + dec_attn + head
    return {"vision_tower_dense": vit_dense / 1e9, "vision_tower_attn": vit_attn / 1e9,
            "vision_merger": merger / 1e9, "lm_prefill_dense": pre_dense / 1e9,
            "lm_prefill_attn": pre_attn / 1e9, "lm_decode_dense": dec_dense / 1e9,
            "lm_decode_attn": dec_attn / 1e9, "lm_head": head / 1e9, "TOTAL": tot / 1e9}


def null_test():
    op = D["flop_model"]["operating_point"]
    got = flops_7b(op["T"], op["M"], op["G_7b"])
    pub = D["flop_model"]["lingshu_7b_gflops"]
    dev = {k: abs(got[k] - pub[k]) for k in pub}
    return {"name": "N1 -- rebuild flop_ratio_derivation_2026-08-03:flop_model.lingshu_7b_gflops",
            "operating_point": op, "rebuilt": {k: round(v, 2) for k, v in got.items()},
            "published": pub, "abs_deviation": dev,
            "max_abs_deviation_gflops": max(dev.values()),
            "pass": bool(max(dev.values()) <= 0.05),
            "note": "published components are 2-dp rounded"}


# ---------------------------------------------------------------- measured token geometry
def _rows_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def geometry():
    """Measured (T, M) of each of the three pass types, plus generated tokens."""
    out = {}

    # generation: cap320, measured this round (prompt only, no answer appended)
    p = os.path.join(ROOT, "feats_free/free_cap320_L21.rows.jsonl")
    if os.path.exists(p):
        rs = [r for r in _rows_jsonl(p) if "err" not in r]
        # one row per DISTINCT answer; the prompt is per QUESTION -> dedupe by (ds, idx)
        byq = {}
        for r in rs:
            byq[(r["ds"], r["idx"])] = (r["n_prompt_tok"], r["n_img_tok"])
        T = np.array([v[0] for v in byq.values()], float)
        M = np.array([v[1] for v in byq.values()], float)
        out["generation_cap320"] = {"n_questions": len(byq), "T_mean": float(T.mean()),
                                    "M_mean": float(M.mean()),
                                    "source": "feats_free/free_cap320_L21.rows.jsonl "
                                              "(n_prompt_tok / n_img_tok), MEASURED"}
        # extraction at cap320 (prompt + candidate), per ROW
        Tf = np.array([r["n_tok"] for r in rs], float)
        Mf = np.array([r["n_img_tok"] for r in rs], float)
        out["extraction_cap320"] = {"n_rows": len(rs), "T_mean": float(Tf.mean()),
                                    "M_mean": float(Mf.mean()),
                                    "source": "same file, n_tok (prompt+candidate), MEASURED"}

    # extraction as DEPLOYED: fullres, from the frozen feature cache's own metadata
    _, ev_rows, _ = G.load_cache("generator", "eval", layers=[], pooling=())
    Te = np.array([r["n_tok"] for r in ev_rows], float)
    out["extraction_fullres_DEPLOYED"] = {
        "n_rows": len(ev_rows), "T_mean": float(Te.mean()),
        "source": "feats_hidden/generator_eval_s*of2.meta.json n_tok, MEASURED"}

    # verifier: fullres grader prompt
    _, gr_rows, _ = G.load_cache("grader", "eval", layers=[], pooling=())
    Tg = np.array([r["n_tok"] for r in gr_rows], float)
    out["verifier_fullres_DEPLOYED"] = {
        "n_rows": len(gr_rows), "T_mean": float(Tg.mean()),
        "source": "feats_hidden/grader_eval_s*of2.meta.json n_tok, MEASURED"}

    # generated tokens of the deployed 8-sample pools
    gt = []
    for ds in G.EVAL_DS:
        for l in open(os.path.join(POOL_DIR, f"ckpt_{ds}_lingshu7b_sc8.jsonl")):
            if l.strip():
                r = json.loads(l)
                gt += r.get("gen_tokens_all", [r["gen_tokens"]])
    out["generated_tokens_sc8"] = {"n": len(gt), "mean": float(np.mean(gt)),
                                   "max": int(np.max(gt)),
                                   "frac_at_max_tokens_64": float(np.mean(np.array(gt) >= 64)),
                                   "source": "ckpts/openvqa/cheap_lingshu7b/ckpt_<ds>_lingshu7b_sc8.jsonl "
                                             "gen_tokens (these 2026-06 pools predate gen_tokens_all, "
                                             "so this is sample 0 of each pool), MEASURED"}
    return out


def image_tokens_fullres(geo):
    """Fullres merged image tokens, inferred as n_tok - text_tok with text_tok MEASURED at cap320.

    The text half of the prompt does not depend on the image budget, so
    M_fullres = T_fullres - (T_cap320 - M_cap320).
    """
    if "extraction_cap320" not in geo:
        return None
    text = geo["extraction_cap320"]["T_mean"] - geo["extraction_cap320"]["M_mean"]
    return {"text_tok_mean_measured_cap320": text,
            "M_fullres_extraction": geo["extraction_fullres_DEPLOYED"]["T_mean"] - text}


def cost_table(cap_rows, n_q, n_gen=8):
    """The BUILD-1 cost table, from MEASURED per-row token geometry.

    cap_rows  the cap320 capture log rows (per distinct normalised answer), each carrying
              n_prompt_tok / n_img_tok / n_ans_tok / n_tok at cap320.
    n_q       number of questions (2345).

    Per-row FULLRES image tokens are recovered EXACTLY, not modelled:
        M_fullres(r) = M_cap320(r) + (T_fullres(r) - T_cap320(r))
    with T_fullres(r) the n_tok recorded in feats_hidden's own metadata for the same
    (ds, idx, na) row and T_cap320(r) the n_tok recorded here.
    """
    _, ev_rows, _ = G.load_cache("generator", "eval", layers=[], pooling=())
    _, gr_rows, _ = G.load_cache("grader", "eval", layers=[], pooling=())
    tf_full = {(r["ds"], r["idx"], r["na"]): r["n_tok"] for r in ev_rows}
    gr_full = {(r["ds"], r["idx"], r["na"]): r["n_tok"] for r in gr_rows}

    gen_T, gen_M, ex_T, ex_M, vr_T, vr_M, per_q = [], [], [], [], [], [], {}
    for r in cap_rows:
        k = (r["ds"], r["idx"], r["na"])
        if k not in tf_full or k not in gr_full:
            continue
        dM = tf_full[k] - r["n_tok"]                      # fullres minus cap320 image tokens
        Mf = r["n_img_tok"] + dM
        ex_T.append(tf_full[k]); ex_M.append(Mf)
        vr_T.append(gr_full[k]); vr_M.append(Mf)
        per_q[(r["ds"], r["idx"])] = (r["n_prompt_tok"], r["n_img_tok"], Mf, dM)
    for (T, M, Mf, dM) in per_q.values():
        gen_T.append(T); gen_M.append(M)

    def mean(x):
        return float(np.mean(x))

    geo = {
        "generation_cap320_per_question": {"T": mean(gen_T), "M": mean(gen_M), "n": len(gen_T)},
        "extraction_fullres_per_row": {"T": mean(ex_T), "M": mean(ex_M), "n": len(ex_T)},
        "verifier_fullres_per_row": {"T": mean(vr_T), "M": mean(vr_M), "n": len(vr_T)},
        "fullres_minus_cap320_image_tokens_mean": mean([v[3] for v in per_q.values()]),
        "method": "T and M(cap320) MEASURED in feats_free/free_cap320_L21.rows.jsonl; "
                  "M(fullres) = M(cap320) + (n_tok_fullres - n_tok_cap320) row-by-row, with "
                  "n_tok_fullres from feats_hidden's own metadata. No modelling.",
    }
    Gtok = 5.373561          # measured, ckpt_<ds>_lingshu7b_sc8.jsonl gen_tokens
    f_gen = flops_7b(geo["generation_cap320_per_question"]["T"],
                     geo["generation_cap320_per_question"]["M"], Gtok)["TOTAL"]
    f_gen_full = flops_7b(mean(ex_T) - np.mean([r["n_ans_tok"] for r in cap_rows]),
                          mean(ex_M), Gtok)["TOTAL"]
    f_ext = flops_7b(geo["extraction_fullres_per_row"]["T"],
                     geo["extraction_fullres_per_row"]["M"], 0)["TOTAL"]
    f_ver = flops_7b(geo["verifier_fullres_per_row"]["T"],
                     geo["verifier_fullres_per_row"]["M"], 1)["TOTAL"]
    f_ext320 = flops_7b(mean([r["n_tok"] for r in cap_rows]),
                        mean([r["n_img_tok"] for r in cap_rows]), 0)["TOTAL"]
    unit = f_gen
    n_norm = len(cap_rows) / n_q
    n_surf = None
    items = G.load_items()
    n_surf = sum(len(set(it["preds"])) for it in items) / n_q

    def arm(gens, ext=0.0, ver=0.0, gen_flop=None, ext_flop=None):
        gf = gen_flop if gen_flop is not None else f_gen
        ef = ext_flop if ext_flop is not None else f_ext
        gflops = gens * gf + ext * ef + ver * f_ver
        return {"gen_passes": gens, "head_extraction_passes": ext, "verifier_passes": ver,
                "total_passes": gens + ext + ver,
                "flopeq_as_charged": gens + ext + ver,
                "gflops": gflops, "flopeq_measured_geometry": gflops / unit}

    arms = {
        "always_7b_BASELINE": arm(1.0),
        "bo8_generation_only": arm(8.0),
        "bo8_head_TODAY": arm(8.0, ext=n_norm),
        "bo8_head_FREE": arm(8.0),
        "bo8_incumbent_LoRA": arm(8.0, ver=n_surf),
        "bo8_fusion_TODAY": arm(8.0, ext=n_norm, ver=n_surf),
        "bo8_fusion_head_FREE": arm(8.0, ver=n_surf),
        "bo8_head_FREE_generating_at_fullres": arm(8.0, gen_flop=f_gen_full),
    }
    return {
        "unit": "1 FLOP-eq = one always-7B answer = ONE batch-1 Lingshu-7B forward at the MEASURED "
                "cap320 generation geometry (%.1f GFLOP)" % f_gen,
        "geometry": geo,
        "per_forward_gflops": {
            "generation_cap320": f_gen, "generation_fullres": f_gen_full,
            "head_extraction_fullres_DEPLOYED": f_ext,
            "head_extraction_cap320_hypothetical": f_ext320,
            "verifier_fullres_DEPLOYED": f_ver},
        "per_forward_flopeq": {
            "generation_cap320": 1.0, "generation_fullres": f_gen_full / unit,
            "head_extraction_fullres_DEPLOYED": f_ext / unit,
            "head_extraction_cap320_hypothetical": f_ext320 / unit,
            "verifier_fullres_DEPLOYED": f_ver / unit},
        "passes_per_question": {"distinct_normalised_answers": n_norm,
                                "distinct_surface_answers": n_surf},
        "arms_fixed_bestof8": arms,
        "headline": {
            "head_passes_removed_per_question": n_norm,
            "flopeq_removed_as_charged": n_norm,
            "flopeq_removed_measured_geometry": n_norm * f_ext / unit,
            "note_as_charged_understates": "the as-charged convention charges the head's extraction "
                                           "one full generation pass, but it was run at FULLRES "
                                           "while generation ran at cap320, so it is really "
                                           "%.3f FLOP-eq each." % (f_ext / unit)},
    }


def distinct_answers_vs_n():
    """MEASURED: mean distinct NORMALISED answers in the first n samples of each pool, n = 1..8.

    This is exactly the head's extraction-pass count as a function of the sample budget, read off
    the stored pools -- no modelling.  It is the bridge between this build's fixed-N=8 endpoint and
    the DEPLOYED adaptive-N operating point (meanN 4.371-6.630, measured in
    cost_decomposition_2026-08-12.json:null_tests.N3), whose exact per-question N is set by the
    Weitzman policy and is NOT reproduced here.
    """
    items = G.load_items()
    per_ds = {}
    for ds in G.EVAL_DS:
        it = [x for x in items if x["ds"] == ds]
        per_ds[ds] = {str(n): float(np.mean([len(set(G.norm(a) for a in x["preds"][:n])) for x in it]))
                      for n in range(1, 9)}
    pooled = {str(n): float(np.mean([len(set(G.norm(a) for a in x["preds"][:n])) for x in items]))
              for n in range(1, 9)}
    return {"pooled": pooled, "per_ds": per_ds,
            "deployed_meanN_measured": {
                "SLAKE_open": 5.547287, "VQA_RAD_open": 6.63, "PATH_VQA_open": 4.371333,
                "source": "results/cascade_methods/artifacts/cost_decomposition_2026-08-12.json"
                          ":null_tests.N3.per_cell.*.meanN_recomputed (MEASURED)"},
            "how_to_read": "at sample budget n the head costs this many extraction passes per "
                           "question TODAY and ZERO once captured during generation.  The deployed "
                           "arm's per-question N varies, so its exact head cost is bracketed by the "
                           "curve at floor(meanN) and at 8, not computed here.",
            "note_shipped_table_assumed_zero":
                "cost_decomposition_2026-08-12.json:Q0.stage_definitions.selector_head records "
                "selector_head = 0.0 for the SHIPPED accuracy-max arm because that arm selects with "
                "the LoRA verifier alone.  The FUSION selector (the better one: sel_eff 0.810627 vs "
                "0.775204) does incur these passes; this build is what makes charging them zero true."}


if __name__ == "__main__":
    nt = null_test()
    print("N1 pass =", nt["pass"], " max abs dev (GFLOP) =", round(nt["max_abs_deviation_gflops"], 4))
    geo = geometry()
    print(json.dumps(geo, indent=1))
    print(json.dumps(image_tokens_fullres(geo), indent=1))
