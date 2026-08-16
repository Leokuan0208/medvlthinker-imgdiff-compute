#!/usr/bin/env python3
"""free_head_lib.py -- shared definitions for BUILD 1 (MAKE THE HEAD FREE), 2026-08-16.

The generator-frame head reads h_span at layer 21 of the UNMODIFIED base Lingshu-7B for every
distinct candidate.  Today that is a SEPARATE teacher-forced forward pass
(src/training_methods/extract_generator_hidden.py, mode=generator) costing 3.813646 passes per
question -- exactly the mean number of distinct normalised answers, 8943/2345.  The generator
already computed those states while sampling the candidate; this round tests whether capturing
them during generation reproduces the head's scores and picks.

Nothing here touches a GPU.  Two currencies, one metric module (genframe_data), no re-implemented
selection rule.

CURRENCIES
  judge  the 32B judge label of the picked slot -- the project's frozen endpoint (`sl` in the
         transfer dumps, identical to ckpt_*_sc8_scexploded.judge.jsonl).
  em     normalised exact match of the picked slot, `oks` in
         ckpts/openvqa/cheap_lingshu7b/ckpt_<ds>_lingshu7b_sc8.jsonl.  That field is
         run_openvqa.score(pred, gold), which is character-for-character
         decoding_sweep_gen.score_em -- the EM currency every dual-currency cell in this project
         uses.  Only compare EM to EM.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import genframe_data as G                      # noqa: E402
from genframe_selector import FrozenSelector   # noqa: E402

ROOT = G.ROOT
POOL_DIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
POOL_TAG = "lingshu7b_sc8"
SELDIR = os.path.join(ROOT, "ckpts/train/genframe_head_ens8")

# generation geometry of the DEPLOYED pools (runners/run_openvqa_lingshu7b.sh,
# runners/run_openvqa_pathvqa.sh -> src/labeling/run_openvqa.py defaults):
#   --cap cap320  =>  max_pixels = 1280*28*28 // 4 ;  min_pixels = 4*28*28
# the FEATURE CACHE was extracted at FULLRES (extract_generator_hidden.HIGH_PX = 1280*28*28).
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4}
GEN_CAP = "cap320"


# ------------------------------------------------------------------ EM currency
def em_slot_labels(items=None):
    """{(ds, idx): [8 EM labels]} aligned to the transfer dumps' `preds` order.

    Raises if any pool row is missing or if the pool's surface answers disagree with the dump's,
    so an EM/judge misalignment cannot pass silently.
    """
    items = items if items is not None else G.load_items()
    pool = {}
    for ds in G.EVAL_DS:
        p = os.path.join(POOL_DIR, f"ckpt_{ds}_{POOL_TAG}.jsonl")
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                pool[(ds, r["idx"])] = r
    out, nmis = {}, 0
    for it in items:
        k = (it["ds"], it["idx"])
        if k not in pool:
            raise KeyError(f"no generation row for {k}")
        r = pool[k]
        if list(r["preds"]) != list(it["preds"]):
            nmis += 1
        out[k] = [int(v) for v in r["oks"]]
    if nmis:
        raise ValueError(f"{nmis} items whose pool preds differ from the transfer dump preds")
    return out


def greedy_em(items=None):
    """EM label of the GREEDY (temperature-0, n=1) answer -- the always-7B baseline, EM currency."""
    items = items if items is not None else G.load_items()
    g = {}
    for ds in G.EVAL_DS:
        p = os.path.join(POOL_DIR, f"ckpt_{ds}_lingshu7b.jsonl")
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                g[(ds, r["idx"])] = int(r["oks"][0])
    return np.array([g[(it["ds"], it["idx"])] for it in items], dtype=int)


# ------------------------------------------------------------------ the endpoint
def endpoint(scores_by_question, items=None, em=None, name=""):
    """Both currencies on IDENTICAL picks, plus per-set guardrail cells.

    picks come from genframe_data.sel_eff (the frozen argmax / first-index rule); the two
    currencies differ ONLY in the label applied to the picked slot.
    """
    items = items if items is not None else G.load_items()
    em = em if em is not None else em_slot_labels(items)
    r = G.sel_eff(scores_by_question, items)
    picks = np.asarray(r["picks"], int)
    E = np.array([em[(it["ds"], it["idx"])] for it in items], dtype=int)
    got_e = E[np.arange(len(items)), picks]
    rec_e = (E.sum(1) > 0).astype(int)
    dsi = np.asarray(r["ds_index"], int)
    per = {}
    for j, ds in enumerate(G.EVAL_DS):
        m = dsi == j
        per[ds] = {
            "n": int(m.sum()),
            "sel_eff_judge": float(r["per_ds"][ds]["sel_eff"]),
            "selected_judge": float(r["per_ds"][ds]["acc"]),
            "oracle_judge": float(r["per_ds"][ds]["oracle"]),
            "sel_eff_em": float(got_e[m & (rec_e == 1)].mean()),
            "selected_em": float(got_e[m].mean()),
            "oracle_em": float(rec_e[m].mean()),
        }
    return {
        "name": name,
        "n": r["n"], "n_recoverable_judge": r["n_recoverable"],
        "n_recoverable_em": int(rec_e.sum()),
        "oracle_judge": r["oracle"], "oracle_em": float(rec_e.mean()),
        "greedy_judge": r["greedy"],
        "selected_judge": r["acc"], "sel_eff_judge": r["sel_eff"],
        "selected_em": float(got_e.mean()),
        "sel_eff_em": float(got_e[rec_e == 1].mean()),
        "identity_selected_eq_oracle_x_seleff_judge":
            float(abs(r["acc"] - r["oracle"] * r["sel_eff"])),
        "per_ds": per,
        "_picks": picks, "_got_judge": np.asarray(r["got"], int), "_got_em": got_e,
        "_rec_judge": np.asarray(r["rec"], int), "_rec_em": rec_e, "_ds_index": dsi,
    }


def vs_always_7b(a, items=None, nboot=10000, seed=0):
    """The ROUND'S endpoint: this selector vs the always-7B (greedy, temperature-0) baseline,
    per reporting cell and pooled, in BOTH currencies, with a paired item bootstrap.

    The new claim shape is 'a small trained verifier improves a 7B medical VLM by +X on N of 8
    benchmark cells', so the per-cell CI-clean count is the quantity of record.
    """
    items = items if items is not None else G.load_items()
    gj = np.array([it["greedy_ok"] for it in items], int)
    ge = greedy_em(items)
    dsi = a["_ds_index"]
    out = {"n_cells_of_8_covered": len(G.EVAL_DS),
           "note": "the other 5 macro cells (PMC_VQA, MedXpert, SLAKE/VQA-RAD/PathVQA closed) are "
                   "multiple-choice-as-presented and carry no open-text pool, so this selector does "
                   "not apply to them -- they are NOT counted as wins or losses here."}
    for cur, gbase in (("judge", gj), ("em", ge)):
        got = a[f"_got_{cur}"]
        cells, nclean = {}, 0
        for j, ds in enumerate(G.EVAL_DS):
            m = dsi == j
            bt = G.paired_bootstrap(got[m], gbase[m], rec=np.ones(int(m.sum()), int),
                                    nboot=nboot, seed=seed)
            v = ("WIN" if bt["d_acc_ci"][0] > 0 else "LOSS" if bt["d_acc_ci"][1] < 0 else "TIE")
            nclean += int(v == "WIN")
            cells[ds] = {"n": int(m.sum()), "always_7b": float(gbase[m].mean()),
                         "selector": float(got[m].mean()), "delta": bt["d_acc"],
                         "ci": bt["d_acc_ci"], "verdict": v}
        bt = G.paired_bootstrap(got, gbase, rec=np.ones(len(items), int), nboot=nboot, seed=seed)
        out[cur] = {"per_cell": cells, "n_cells_CI_clean_win": nclean,
                    "n_cells_CI_clean_loss": sum(1 for c in cells.values() if c["verdict"] == "LOSS"),
                    "pooled_always_7b": float(gbase.mean()), "pooled_selector": float(got.mean()),
                    "pooled_delta": bt["d_acc"], "pooled_ci": bt["d_acc_ci"],
                    "pooled_verdict": ("WIN" if bt["d_acc_ci"][0] > 0 else
                                       "LOSS" if bt["d_acc_ci"][1] < 0 else "TIE"),
                    "macro3_delta": float(np.mean([c["delta"] for c in cells.values()]))}
    return out


def compare(a, b, nboot=10000, seed=0):
    """Paired item bootstrap of a - b in BOTH currencies + the per-set guardrail flags."""
    out = {}
    for cur in ("judge", "em"):
        ga, gb = a[f"_got_{cur}"], b[f"_got_{cur}"]
        rec = a[f"_rec_{cur}"]
        bt = G.paired_bootstrap(ga, gb, rec=rec, nboot=nboot, seed=seed)
        clean = all(a["per_ds"][d][f"sel_eff_{cur}"] >= b["per_ds"][d][f"sel_eff_{cur}"]
                    for d in G.EVAL_DS)
        out[cur] = {
            "d_selected": bt["d_acc"], "d_selected_ci": bt["d_acc_ci"],
            "d_sel_eff": bt["d_sel_eff"], "d_sel_eff_ci": bt["d_sel_eff_ci"],
            "guardrail_clean_sel_eff": bool(clean),
            "per_ds_d_sel_eff": {d: a["per_ds"][d][f"sel_eff_{cur}"] - b["per_ds"][d][f"sel_eff_{cur}"]
                                 for d in G.EVAL_DS},
            "verdict": ("WIN" if bt["d_acc_ci"][0] > 0 else
                        "LOSS" if bt["d_acc_ci"][1] < 0 else "TIE"),
        }
    out["picks_changed"] = int((a["_picks"] != b["_picks"]).sum())
    out["nboot"] = nboot
    return out


# ------------------------------------------------------------------ selectors
def head_only_scores(H_by_row, ev_questions, sel: FrozenSelector, standardized=False):
    """{(ds, idx): 8 slot scores} for the HEAD ALONE (no incumbent, no fusion).

    The score is the 8-seed mean of the per-seed within-pool rank_avg -- i.e. exactly the
    `head_rank` term of the frozen fusion, used on its own.
    """
    X = H_by_row if standardized else sel.standardize(H_by_row)
    L = sel.head_logits(X, standardized=True)          # (8 seeds, n_rows)
    out = {}
    for q in ev_questions:
        rows = np.asarray(q.slot_rows)
        out[(q.ds, q.idx)] = list(np.mean([G.rank_avg(L[s][rows]) for s in range(L.shape[0])], axis=0))
    return out, L


def fusion_scores(H_by_row, ev_questions, sel: FrozenSelector, inc_scores, standardized=False):
    """The frozen deployed fusion: rank_avg(incumbent) + rank_avg(head_rank)."""
    hs, L = head_only_scores(H_by_row, ev_questions, sel, standardized)
    out = {}
    for q in ev_questions:
        k = (q.ds, q.idx)
        out[k] = list(G.rank_avg(np.asarray(inc_scores[k], float)) + G.rank_avg(np.asarray(hs[k], float)))
    return out, L


# ------------------------------------------------------------------ cost
#: one FLOP-eq = one full batch-1 Lingshu-7B forward on the cap320 open-text geometry
#: (results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json via
#:  cost_decomposition_2026-08-12.json:N2).
STAGE_SHARE_7B = {"vision": 0.253643, "lm_prefill": 0.734812, "decode": 0.011544}
PROMPT_TOK = 326.68        # cap320 open-text prompt (incl. 280.48 image tokens)
GEN_TOK_7B = 5.64
MEAN_ANS_TOK = 5.3         # brief; recomputed from the tokenizer in free_head_capture.py


def prefill_only_pass(extra_tok=0.0):
    """FLOP-eq of a PREFILL-ONLY 7B forward (vision + LM prefill, no decode), relative to one
    full generation pass.  `extra_tok` extends the prefill (the candidate appended to the prompt).
    """
    s = STAGE_SHARE_7B
    return s["vision"] + s["lm_prefill"] * (PROMPT_TOK + extra_tok) / PROMPT_TOK
