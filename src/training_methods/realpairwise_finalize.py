#!/usr/bin/env python3
"""realpairwise_finalize.py -- merge the two engine arms into the canonical artifact.

Writes results/cascade_methods/artifacts/verifarch_realpairwise_clean_2026-08-04.json with:
  * the null test and the independent disjointness proof,
  * the vLLM arm (complete 2345-item pool, plus its engine-matched pointwise control),
  * the HF full-adapter arm (complete slake+vqa_rad, pre-registered 500-question pathvqa
    subsample), evaluated against the incumbent on the identical covered items,
  * the explicit non-comparability notice for the old 0.783/0.849/0.859 triple,
  * cost, and the paths of the per-item pairwise teacher files.

  python3 src/training_methods/realpairwise_finalize.py
"""
import os, sys, json

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "verifarch_realpairwise_clean_2026-08-04.json")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    vllm = load(os.path.join(ART, "verifarch_realpairwise_vllm_2026-08-05.json"))
    hf = load(os.path.join(ART, "verifarch_realpairwise_hf_2026-08-05.json"))
    dis = load(os.path.join(ART, "realpairwise_disjointness_2026-08-05.json"))
    if vllm is None or hf is None:
        sys.exit("missing an arm; run realpairwise_clean_analyze.py and realpairwise_hf_analyze.py")

    hb = hf["arms"]["borda_avg"]
    hc = hf["arms"]["copeland_pure_avg"]
    hk = hf["arms"]["knockout_det_avg"]
    inc_cov = hf["incumbent_on_covered_subset"]

    out = {
        "generated": "2026-08-05",
        "title": "CLEAN GPU replication of the REAL pairwise verifier -- decontamination test",
        "question": "Does the prior real-pairwise selection win (pointwise 0.783 -> knockout 0.849 "
                    "-> round-robin 0.859) survive CLEAN disjoint-trained weights on the current "
                    "2345-question pool whose incumbent bar is sel_eff 0.775204?",
        "answer": "NO. Under conditions matched to the bar (same HuggingFace stack, same clean "
                  "adapter incl. vision tower, same max_pixels, same items, paired), real A-vs-B "
                  "forward passes are a NULL against the pointwise verifier -- and there is no "
                  "pointwise < knockout < round-robin ladder at all.",
        "protocol": {
            "null_test": vllm["null_test"],
            "disjointness_independently_reproved": dis,
            "seeds": "the method trains nothing, so it has no seed variance; the only stochastic "
                     "component (the original Bernoulli knockout bracket) was run with 10 seeds and "
                     "its mean/sd/range are reported in each arm",
            "statistics": "paired item-level bootstrap, nboot=10000, vs the incumbent's per-item "
                          "0/1 scores on the identical items",
            "eval_leakage": "none -- no configuration was fitted on eval. The aggregators (Borda, "
                            "Copeland/round-robin, knockout) and the rank_avg fusion are "
                            "parameter-free, and the pathvqa subsample was registered before any HF "
                            "pathvqa number was computed.",
            "harness_self_test": "Bradley-Terry preferences SIMULATED from the incumbent's own "
                                 "scores reproduce 0.775204 exactly through Borda, Copeland and "
                                 "knockout, so any deviation measured here is real comparative "
                                 "information and not an aggregation artifact.",
        },
        "arm_hf_full_adapter_ENGINE_MATCHED_HEADLINE": {
            "engine": hf["engine"],
            "covered_items": hf["covered_items"], "covered_per_ds": hf["covered_per_ds"],
            "pathvqa_subsample": hf["pathvqa_subsample"],
            "incumbent_on_the_same_items": inc_cov,
            "roundrobin_copeland": {"sel_eff": hc["sel_eff"], "d": hc["d_sel_eff"],
                                    "ci": hc["d_sel_eff_ci"], "sig": hc["sig"],
                                    "per_ds": hc["per_ds"], "contested": hc["contested_sel_eff"],
                                    "d_contested": hc["d_contested"],
                                    "ci_contested": hc["d_contested_ci"],
                                    "guardrail_clean": hc["guardrail_clean"]},
            "borda": {"sel_eff": hb["sel_eff"], "d": hb["d_sel_eff"], "ci": hb["d_sel_eff_ci"],
                      "sig": hb["sig"], "per_ds": hb["per_ds"],
                      "contested": hb["contested_sel_eff"], "d_contested": hb["d_contested"],
                      "ci_contested": hb["d_contested_ci"],
                      "guardrail_clean": hb["guardrail_clean"]},
            "knockout": {"sel_eff": hk["sel_eff"], "d": hk["d_sel_eff"], "ci": hk["d_sel_eff_ci"],
                         "sig": hk["sig"], "per_ds": hk["per_ds"],
                         "guardrail_clean": hk["guardrail_clean"]},
            "knockout_stochastic_10_seeds": hf["arms"]["knockout_stoch"],
            "fusion_with_incumbent": {k: hf["arms"][k] for k in hf["arms"] if k.startswith("fuse_")},
            "position_bias": hf["position_bias"],
            "discordant_pair_discrimination": hf["discordant_pair_discrimination"],
            "all_arms": hf["arms"],
        },
        "arm_vllm_complete_pool": {
            "engine": "vLLM 0.9.0.1 -- applies the LoRA to the LANGUAGE MODEL ONLY and DROPS all "
                      "192 visual.* LoRA modules (it logs this). The arm is therefore engine-"
                      "handicapped and is reported WITH its own engine-matched pointwise control.",
            "covers": "all 2345 items, all 19952 distinct pairs, both orders, 0 errors",
            "roundrobin_copeland": vllm["arms"]["copeland_pure_avg"],
            "borda": vllm["arms"]["borda_avg"],
            "knockout": vllm["arms"]["knockout_det_avg"],
            "knockout_stochastic_10_seeds": vllm["arms"]["knockout_stoch"],
            "ENGINE_MATCHED_POINTWISE_CONTROL": vllm["arms"].get("pointwise_control_vllm"),
            "pairwise_vs_engine_matched_pointwise": {
                k: v for k, v in vllm["arms"].items() if k.endswith("_vs_engine_matched_pointwise")},
            "position_bias": vllm["position_bias"],
            "discordant_pair_discrimination": vllm["discordant_pair_discrimination"],
            "near_tie_stratum": vllm.get("near_tie_stratum"),
            "all_arms": vllm["arms"],
        },
        "engine_finding": {
            "what": "vLLM's LM-only LoRA support materially degrades this verifier.",
            "same_adapter_same_prompt_same_pixels": {
                "HF_PeftModel_full_adapter": {"sel_eff": 0.7752043596730245,
                                              "cand_auroc": 0.8855921901711237},
                "vLLM_language_model_only": {
                    "sel_eff": vllm["arms"]["pointwise_control_vllm"]["sel_eff"],
                    "cand_auroc": 0.760242},
            },
            "score_agreement": {"pearson": 0.4711, "spearman": 0.6241, "mean_abs_diff": 0.3680},
            "implication": "Never score this verifier under vLLM and compare it to an HF-scored "
                           "number. Any future vLLM verifier arm must ship an engine-matched "
                           "control.",
        },
        "cost": {
            "generations_per_question": 8,
            "incumbent_pointwise_extra_forwards_per_question": "8 (or 3.81 deduplicated)",
            "round_robin_extra_forwards_per_question": vllm["cost"][
                "roundrobin_forwards_per_question_both_orders"],
            "knockout_extra_forwards_per_question": vllm["cost"][
                "knockout_distinct_forwards_per_question_both_orders"],
            "note": "Round-robin costs ~17.0 forward passes per question over and above the 8 "
                    "generations (8.51 unordered distinct pairs x 2 orders for position debias), "
                    "i.e. ~4.5x the deduplicated pointwise verifier, for a measured null. These are "
                    "full VLM forward passes over image+question+two answers; they share nothing "
                    "with the cached generator-frame features the pointwise head uses.",
            "total_forward_passes_run_this_round": {"vllm_pairwise": 39904,
                                                    "vllm_pointwise_control": 8943,
                                                    "hf_pairwise": hf["verdict_files"]["n_rows"]},
        },
        "teacher_files": {
            "hf_full_adapter": os.path.join(ART, "realpairwise_teacher_pmatrix_hf_2026-08-05.jsonl"),
            "vllm_complete_pool": os.path.join(ART, "realpairwise_teacher_pmatrix_2026-08-05.jsonl"),
            "schema": "one JSON object per question: ds, idx, na[k] (distinct normalized answers, "
                      "the SAME keys as genframe_data rows), slots[k], y[k], inc_score[k], and the "
                      "k x k matrices P_avg / P_o0 / P_o1 where P[a][b] = P(answer a is better than "
                      "answer b). Join on (ds, idx, na) to feats_hidden rows.",
            "use": "the real comparative signal, for measuring how much of it a cached-vector "
                   "pairwise head can recover (teacher/student). NOTE the teacher is a NULL "
                   "selector -- distilling it should not be expected to beat the pointwise head.",
        },
        "NOT_COMPARABLE_TO": vllm["NOT_COMPARABLE_TO"],
        "code": ["src/training_methods/realpairwise_clean_gpu.py",
                 "src/training_methods/realpairwise_clean_hf.py",
                 "src/training_methods/realpairwise_pointwise_control.py",
                 "src/training_methods/realpairwise_clean_analyze.py",
                 "src/training_methods/realpairwise_hf_analyze.py",
                 "src/training_methods/realpairwise_assert_disjoint.py",
                 "src/training_methods/realpairwise_finalize.py"],
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    print(f"\nHEADLINE (HF, engine-matched, n={hf['covered_items']} covered items):")
    print(f"  incumbent            {inc_cov:.6f}")
    print(f"  round-robin/Copeland {hc['sel_eff']:.6f}  d={hc['d_sel_eff']:+.6f} "
          f"{hc['d_sel_eff_ci']}  {'SIG' if hc['sig'] else 'n.s.'}")
    print(f"  knockout             {hk['sel_eff']:.6f}  d={hk['d_sel_eff']:+.6f} {hk['d_sel_eff_ci']}")
    print(f"  borda                {hb['sel_eff']:.6f}  d={hb['d_sel_eff']:+.6f} {hb['d_sel_eff_ci']}")


if __name__ == "__main__":
    main()
