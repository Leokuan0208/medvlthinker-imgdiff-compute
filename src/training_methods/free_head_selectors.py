#!/usr/bin/env python3
"""free_head_selectors.py -- BUILD 1, part A (CPU only): HEAD-ONLY selection quality.

Answers deliverable 4 of the round: the frozen generator-frame head scores sel_eff 0.7956 alone
against the LoRA verifier's 0.7752, so DROPPING THE LoRA ENTIRELY may be both cheaper and better.
That has to be tested explicitly, in BOTH currencies, per set, with the guardrail and a paired
bootstrap -- not implied.

Arms (all on the SAME frozen 2345-question best-of-8 pool, identical picks rule):
  greedy              the always-7B baseline (temperature 0, n=1)
  random_pick         closed-form expectation of picking a uniform slot
  self_consistency    plurality of the 8 surface strings (training-free control)
  incumbent           the CLEAN disjoint LoRA verifier               3.823028 passes/question
  head_ens8           the frozen 8-seed generator-frame head ALONE   3.813646 passes/question
  fusion              rank_avg(incumbent) + rank_avg(head_rank)      7.636674 passes/question
  oracle@8            upper bound

Writes results/cascade_methods/artifacts/_free_head_parts/selectors.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G                      # noqa: E402
import free_head_lib as F                      # noqa: E402
from genframe_selector import FrozenSelector   # noqa: E402

OUT = os.path.join(G.ROOT, "results/cascade_methods/artifacts/_free_head_parts")
os.makedirs(OUT, exist_ok=True)


def main():
    items = G.load_items()
    em = F.em_slot_labels(items)
    sel = FrozenSelector.load(F.SELDIR)
    ev = G.load_candidates("eval", sel.recipe["mode"], layers=[sel.recipe["layer"]],
                           pooling=(sel.recipe["pooling"],), order=sel.recipe["row_order"])
    H = ev.matrix(sel.recipe["pooling"], sel.recipe["layer"])
    inc = G.incumbent_scores()

    hs, _ = F.head_only_scores(H, ev.questions, sel)
    fu, _ = F.fusion_scores(H, ev.questions, sel, inc)
    ctl = G.control_scores(items)

    arms = {}
    arms["incumbent"] = F.endpoint(inc, items, em, "incumbent LoRA verifier (clean/disjoint)")
    arms["head_ens8"] = F.endpoint(hs, items, em, "frozen 8-seed generator-frame head ALONE")
    arms["fusion"] = F.endpoint(fu, items, em, "deployed fusion: rank_avg(inc) + rank_avg(head)")
    arms["self_consistency"] = F.endpoint(ctl["self_consistency"], items, em, "plurality of 8 surface strings")

    # greedy (always-7B) and oracle, both currencies
    E = np.array([em[(it["ds"], it["idx"])] for it in items], int)
    gj = np.array([it["greedy_ok"] for it in items], int)
    ge = F.greedy_em(items)
    dsi = np.array([G.EVAL_DS.index(it["ds"]) for it in items], int)
    rp = G.random_pick(items)
    baselines = {
        "always_7b_greedy": {
            "selected_judge": float(gj.mean()), "selected_em": float(ge.mean()),
            "per_ds": {d: {"selected_judge": float(gj[dsi == j].mean()),
                           "selected_em": float(ge[dsi == j].mean())}
                       for j, d in enumerate(G.EVAL_DS)},
            "source_judge": "transfer dump greedy_ok",
            "source_em": "ckpts/openvqa/cheap_lingshu7b/ckpt_<ds>_lingshu7b.jsonl oks[0]"},
        "oracle_at_8": {
            "selected_judge": float(arms["incumbent"]["oracle_judge"]),
            "selected_em": float(arms["incumbent"]["oracle_em"]),
            "per_ds": {d: {"selected_judge": arms["incumbent"]["per_ds"][d]["oracle_judge"],
                           "selected_em": arms["incumbent"]["per_ds"][d]["oracle_em"]}
                       for d in G.EVAL_DS}},
        "random_pick_closed_form_judge": rp,
    }

    # per-seed head-only, to reconcile the round brief's "the head alone scores 0.7956"
    Lall = sel.head_logits(H)
    per_seed = []
    for s in range(Lall.shape[0]):
        sm = {}
        for q in ev.questions:
            rows = np.asarray(q.slot_rows)
            sm[(q.ds, q.idx)] = list(G.rank_avg(Lall[s][rows]))
        r = G.sel_eff(sm, items)
        per_seed.append(r["sel_eff"])
    head_seeds = {
        "per_seed_sel_eff_judge": [float(v) for v in per_seed],
        "mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)),
        "min": float(np.min(per_seed)), "max": float(np.max(per_seed)),
        "ensemble_of_8_sel_eff_judge": arms["head_ens8"]["sel_eff_judge"],
        "note": "the round brief quotes 0.7956 for the head alone; that is a SINGLE-SEED fit "
                "(freeze_selector.py records seed-0 sel_eff 0.795640 at the pinned thread count). "
                "The frozen 8-seed ENSEMBLE head alone is the number reported here."}

    cmp = {
        "head_vs_incumbent": F.compare(arms["head_ens8"], arms["incumbent"]),
        "head_vs_fusion": F.compare(arms["head_ens8"], arms["fusion"]),
        "fusion_vs_incumbent": F.compare(arms["fusion"], arms["incumbent"]),
        "head_vs_self_consistency": F.compare(arms["head_ens8"], arms["self_consistency"]),
    }

    # ---- cost: passes per question, and FLOP-eq against always-7B = 1.0 -------------------
    n_q = len(items)
    n_rows_distinct_norm = sum(len(set(G.norm(a) for a in it["preds"])) for it in items) / n_q
    n_rows_distinct_surf = sum(len(set(it["preds"])) for it in items) / n_q
    ans_extra = F.MEAN_ANS_TOK
    per_pass = F.prefill_only_pass(ans_extra)
    cost = {
        "convention": "as-charged PRIMARY (1 FLOP-eq = 1 full batch-1 Lingshu-7B forward on the "
                      "cap320 open-text geometry); the honest re-cost charges the scoring passes "
                      "prefill-only.",
        "always_7b_baseline_flopeq": 1.0,
        "distinct_normalised_answers_per_question": float(n_rows_distinct_norm),
        "distinct_surface_answers_per_question": float(n_rows_distinct_surf),
        "prefill_only_pass_flopeq": float(per_pass),
        "note_prefill_only": "vision 0.253643 + lm_prefill 0.734812 * (326.68 + %.1f)/326.68"
                             % ans_extra,
        "arms_fixed_bestof8": {},
    }
    for nm, extra in [("always_7b", 0.0), ("bo8_no_selector", 0.0),
                      ("bo8_incumbent", n_rows_distinct_surf),
                      ("bo8_head_today", n_rows_distinct_norm),
                      ("bo8_head_FREE", 0.0),
                      ("bo8_fusion_today", n_rows_distinct_surf + n_rows_distinct_norm),
                      ("bo8_fusion_head_free", n_rows_distinct_surf)]:
        gens = 1.0 if nm == "always_7b" else 8.0
        cost["arms_fixed_bestof8"][nm] = {
            "gen_passes": gens, "scoring_passes": float(extra),
            "total_passes": float(gens + extra),
            "flopeq_as_charged": float(gens + extra),
            "flopeq_honest": float(gens + extra * per_pass),
            "x_always_7b_as_charged": float(gens + extra),
        }

    out = {
        "title": "BUILD 1 part A -- HEAD-ONLY selection quality on the frozen best-of-8 open pool",
        "date": "2026-08-16", "no_gpu": True,
        "reproduce": "python3 src/training_methods/free_head_selectors.py",
        "null_test": G.null_test(),
        "selector_reload_verify": __import__("genframe_selector").verify(F.SELDIR),
        "features": "feats_hidden/generator_eval_s*of2.npz (FULLRES teacher-forced), layer 21, span",
        "currencies": {
            "judge": "32B judge label of the picked slot (frozen endpoint)",
            "em": "normalised exact match, run_openvqa.score == decoding_sweep_gen.score_em, "
                  "read from ckpts/openvqa/cheap_lingshu7b/ckpt_<ds>_lingshu7b_sc8.jsonl:oks"},
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in arms.items()},
        "baselines": baselines,
        "head_per_seed": head_seeds,
        "vs_always_7b": {k: F.vs_always_7b(v, items) for k, v in arms.items()},
        "comparisons": cmp,
        "cost": cost,
    }
    p = os.path.join(OUT, "selectors.json")
    json.dump(out, open(p, "w"), indent=1, default=float)

    print(f"NULL TEST pass={out['null_test']['pass']} maxdev={out['null_test']['max_abs_deviation']:.3e}")
    print(f"SELECTOR VERIFY pass={out['selector_reload_verify']['pass']} "
          f"maxdev={out['selector_reload_verify']['max_abs_deviation']:.3e}")
    print(f"\n{'arm':<20}{'selE_j':>9}{'sel_j':>9}{'selE_em':>9}{'sel_em':>9}   per-ds sel_eff (judge)")
    print(f"{'greedy(always-7B)':<20}{'-':>9}{gj.mean():>9.6f}{'-':>9}{ge.mean():>9.6f}")
    for k, a in arms.items():
        print(f"{k:<20}{a['sel_eff_judge']:>9.6f}{a['selected_judge']:>9.6f}"
              f"{a['sel_eff_em']:>9.6f}{a['selected_em']:>9.6f}   "
              + " ".join(f"{a['per_ds'][d]['sel_eff_judge']:.4f}" for d in G.EVAL_DS))
    print(f"{'oracle@8':<20}{'1.000000':>9}{arms['incumbent']['oracle_judge']:>9.6f}"
          f"{'1.000000':>9}{arms['incumbent']['oracle_em']:>9.6f}")
    print("\nidentity |selected - oracle*sel_eff| (judge):",
          {k: f"{a['identity_selected_eq_oracle_x_seleff_judge']:.3e}" for k, a in arms.items()})
    for k, c in cmp.items():
        print(f"\n{k}  picks_changed={c['picks_changed']}")
        for cur in ("judge", "em"):
            d = c[cur]
            print(f"   {cur:<6} dSELECTED {d['d_selected']:+.6f} [{d['d_selected_ci'][0]:+.6f},"
                  f"{d['d_selected_ci'][1]:+.6f}] {d['verdict']:<5} "
                  f"dsel_eff {d['d_sel_eff']:+.6f} [{d['d_sel_eff_ci'][0]:+.6f},"
                  f"{d['d_sel_eff_ci'][1]:+.6f}]  guardrail_clean={d['guardrail_clean_sel_eff']}")
    print("\nwrote", p)


if __name__ == "__main__":
    main()
