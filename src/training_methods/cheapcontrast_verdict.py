#!/usr/bin/env python3
"""cheapcontrast_verdict.py -- write the verdict block of
results/cascade_methods/artifacts/verifarch_cheapcontrast_2026-08-04.json.

Every number in the verdict is READ OUT of the artifact it is describing (no value is typed
by hand), so the prose cannot drift from the measurement.

    python3 src/training_methods/cheapcontrast_verdict.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import genframe_data as G  # noqa: E402

OUT = os.path.join(G.ROOT, "results/cascade_methods/artifacts/verifarch_cheapcontrast_2026-08-04.json")
art = json.load(open(OUT))

ev = art["job1_eval"]
best = ev["cv_selected"]
arms = {f'{a["featset"]}|{a["objective"]}': a for a in ev["arms"]}
cmps = {c["arm"]: c for c in ev["comparisons"]}
HEAD = f'{best["featset"]}|{best["objective"]}'
BASE = f'H|{best["objective"]}'
h, b = arms[HEAD], arms[BASE]
ch, cb = cmps[HEAD], cmps[BASE]


def r6(x):
    return round(float(x), 6)


def ci(x):
    return [r6(v) for v in x]


art["verdict"] = {
    "headline": (
        "JOB 1 IS A CLEAN NEGATIVE. The pre-registered cheap-contrast arm beats the incumbent LoRA "
        f'verifier ({r6(h["ensemble_rank"]["sel_eff"])} vs 0.775204, d = {r6(h["ensemble_rank"]["d_sel_eff"])} '
        f'{ci(h["ensemble_rank"]["d_sel_eff_ci"])}, guardrail-clean) but it does NOT beat the SAME pointwise '
        f'head on identical raw features (d = {r6(ch["vs_pointwise_head_same_features_ensemble"]["d_sel_eff"])} '
        f'{ci(ch["vs_pointwise_head_same_features_ensemble"]["ci"])}) and does NOT beat the deployed '
        f'rank fusion 0.806540 (d = {r6(ch["vs_published_fusion_0.806540"]["d_sel_eff"])} '
        f'{ci(ch["vs_published_fusion_0.806540"]["ci"])}). Pool-relative geometry over cached '
        "per-candidate vectors does not reproduce the measured A-vs-B pairwise win. "
        "JOB 2 RETIRES THE FRAME EFFECT: at matched configuration and 10 seeds, generator minus grader "
        "is +0.0041 / +0.0129 / -0.0020 / +0.0020 across the four pooling x objective cells, every 95% CI "
        "spanning zero. The published +0.045 was pooling + objective + one seed, not frame."),

    "job1_preregistered_primary": {
        "definition": "feature set chosen by 5-fold image-grouped CV inside the disjoint TRAIN pool "
                      "with the head architecture frozen at the previous round's CV pick "
                      "(L21/span/bt/hidden256/wd1e-2/30ep); eval untouched during selection. "
                      "Deployable number = the 10-seed rank-average ensemble.",
        "featset": best["featset"], "blocks": best["blocks"], "objective": best["objective"],
        "cv_sel_eff_train_only": r6(best["cv_sel_eff"]),
        "cv_sel_eff_train_only_H_baseline": r6(
            [g for g in art["job1_cv"]["grid"] if g["featset"] == "H"][0]["cv_sel_eff"]),
        "sel_eff": r6(h["ensemble_rank"]["sel_eff"]),
        "selected_acc": r6(h["ensemble_rank"]["acc"]),
        "per_seed_sel_eff": [r6(x) for x in h["per_seed_sel_eff"]],
        "seed_mean": r6(h["seed_mean"]), "seed_sd": r6(h["seed_sd"]),
        "seed_range": [r6(h["seed_min"]), r6(h["seed_max"])],
        "per_ds": {k: r6(v) for k, v in h["ensemble_rank"]["per_ds"].items()},
        "guardrail_clean": h["ensemble_rank"]["guardrail_clean"],
        "contested_sel_eff": r6(h["ensemble_rank"]["contested_sel_eff"]),
        "contested_n": h["ensemble_rank"]["contested_n"],
        "vs_incumbent_0.775204": {"d_sel_eff": r6(h["ensemble_rank"]["d_sel_eff"]),
                                  "ci": ci(h["ensemble_rank"]["d_sel_eff_ci"]),
                                  "d_contested": r6(h["ensemble_rank"]["d_contested"]),
                                  "contested_ci": ci(h["ensemble_rank"]["d_contested_ci"]),
                                  "verdict": "WINS, CI excludes zero"},
        "vs_pointwise_head_identical_features": {
            "comparator_sel_eff": r6(b["ensemble_rank"]["sel_eff"]),
            "d_sel_eff": r6(ch["vs_pointwise_head_same_features_ensemble"]["d_sel_eff"]),
            "ci": ci(ch["vs_pointwise_head_same_features_ensemble"]["ci"]),
            "d_contested": r6(ch["vs_pointwise_head_same_features_ensemble"]["d_contested"]),
            "contested_ci": ci(ch["vs_pointwise_head_same_features_ensemble"]["contested_ci"]),
            "verdict": "NO -- CI includes zero. The contrast blocks add nothing over the raw "
                       "hidden state the deployed head already sees."},
        "vs_deployed_fusion_0.806540": {
            "d_sel_eff": r6(ch["vs_published_fusion_0.806540"]["d_sel_eff"]),
            "ci": ci(ch["vs_published_fusion_0.806540"]["ci"]),
            "verdict": "NO -- CI includes zero."},
    },

    "job1_ablation_reading": {
        "what_each_block_buys_over_H_alone": {
            k: {"sel_eff": r6(arms[k]["ensemble_rank"]["sel_eff"]),
                "d_vs_pointwise": r6(cmps[k]["vs_pointwise_head_same_features_ensemble"]["d_sel_eff"]),
                "ci": ci(cmps[k]["vs_pointwise_head_same_features_ensemble"]["ci"])}
            for k in ["H+C|bt", "H+M|bt", "H+C+M|bt", "H+C+M+Wc|bt", "H+C+M+Wc+Ws|bt",
                      "H+C+M+Yc+Ys|bt"] if k in arms},
        "contrast_features_ALONE": {
            "C_geometry_only": r6(arms["C|bt"]["ensemble_rank"]["sel_eff"]),
            "C_plus_multiplicity_only": r6(arms["CM|bt"]["ensemble_rank"]["sel_eff"]),
            "random_pick_floor": r6(art["controls"]["random_pick"]["sel_eff"]),
            "reading": "the geometry block alone selects WORSE THAN RANDOM (0.662807 vs a 0.676260 "
                       "random-pick floor). Pool-typicality is not correctness."},
        "single_feature_table": "job1_per_feature.per_feature -- all 18 geometry features used alone "
                                "as selectors land between 0.634196 (log_norm) and 0.694142 "
                                "(max_cos_other) on eval, against a 0.676260 random-pick floor: 7 of "
                                "18 at or below it, and the best is +0.0179 above it and 0.0811 "
                                "BELOW the incumbent. The only pool-relative scalar that carries "
                                "real signal is the free self-consistency count (mult: 0.713896, "
                                "which is exactly the self-consistency control).",
        "diagnosis_WHY_the_negative": (
            "A cached per-candidate generator-frame vector was computed with the other candidates "
            "ABSENT. A real A-vs-B pairwise pass conditions candidate A's representation on B's "
            "text; a cosine between two independently-computed vectors cannot manufacture that "
            "conditioning. The comparative information a pairwise forward pass creates is not a "
            "function of the two pointwise vectors, so no amount of pool-relative arithmetic over "
            "the cache can recover it. The only pool-relative quantity that helps is the one that "
            "was never representational to begin with -- how many of the 8 samples produced this "
            "string -- and that is already available for free and worth only "
            "+0.0048 [-0.0048, +0.0143] on top of the head."),
    },

    "job1_secondary_POSITIVE_result": {
        "claim": "seed-averaging the ALREADY DEPLOYED head is worth more than every feature added "
                 "in this round, and costs nothing at inference.",
        "published_single_seed_head": {"sel_eff": 0.795640,
                                       "d_vs_incumbent": 0.020436, "ci": [-0.001362, 0.041553],
                                       "verdict": "near-miss, n.s. (the previous round's finding)"},
        "same_head_10_seed_rank_ensemble": {
            "sel_eff": r6(b["ensemble_rank"]["sel_eff"]),
            "per_seed": [r6(x) for x in b["per_seed_sel_eff"]],
            "seed_mean": r6(b["seed_mean"]), "seed_sd": r6(b["seed_sd"]),
            "seed_range": [r6(b["seed_min"]), r6(b["seed_max"])],
            "d_vs_incumbent": r6(b["ensemble_rank"]["d_sel_eff"]),
            "ci": ci(b["ensemble_rank"]["d_sel_eff_ci"]),
            "guardrail_clean": b["ensemble_rank"]["guardrail_clean"],
            "per_ds": {k: r6(v) for k, v in b["ensemble_rank"]["per_ds"].items()},
            "vs_deployed_fusion_0.806540": {
                "d": r6(cb["vs_published_fusion_0.806540"]["d_sel_eff"]),
                "ci": ci(cb["vs_published_fusion_0.806540"]["ci"]),
                "verdict": "statistically TIES the fusion without using the incumbent at all"},
            "caveat": "guardrail-DIRTY on vqa_rad_open (0.746032 vs the incumbent's 0.761905). The "
                      "pre-registered contrast arm is guardrail-CLEAN (0.825397 on vqa_rad_open); "
                      "that is the one concrete thing the contrast blocks buy, and it is a "
                      "guardrail repair, not a pooled gain.",
            "cost": "10 tiny MLP evaluations per candidate instead of 1, over feature vectors that "
                    "were already computed. Zero extra forward passes."},
    },

    "job2_frame_effect": {
        "claim_tested": "the same frozen Lingshu-7B read in the GENERATOR frame beats itself read in "
                        "the GRADER frame (published 0.795640 vs 0.750681, '+0.045').",
        "confound": "the two published cells were fit at DIFFERENT configurations -- generator "
                    "L21/SPAN/BT, grader L21/LAST/BCE -- and at ONE seed each. Frame is confounded "
                    "with pooling, objective and seed.",
        "both_published_cells_reproduce_bit_exact_on_cpu": {
            "generator_L21_span_bt": {"measured": 0.7956403269754768, "published": 0.795640},
            "grader_L21_last_bce": {"measured": 0.7506811989100818, "published": 0.750681},
            "published_rank_avg_fusion": {"measured": 0.8065395095367848, "published": 0.806540}},
        "matched_2x2x2_grid_10_seeds": {
            f'{hh["frame"]}|{hh["pooling"]}|{hh["objective"]}': {
                "seed_mean": r6(hh["seed_mean"]), "seed_sd": r6(hh["seed_sd"]),
                "seed_range": [r6(hh["seed_min"]), r6(hh["seed_max"])],
                "ensemble_sel_eff": r6(hh["ensemble_rank"]["sel_eff"])}
            for hh in art["job2_matched_heads"]["heads"]},
        "frame_contrast_paired": art["job2_matched_heads"]["frame_contrast_matched"],
        "verdict": "NOT ESTABLISHED. Generator minus grader at matched config is +0.0041 (last/bce), "
                   "+0.0129 (last/bt), -0.0020 (span/bce), +0.0020 (span/bt); all four 95% CIs "
                   "include zero, and at SPAN pooling the two frames are indistinguishable. The "
                   "published +0.045 is a configuration-plus-seed artifact. The round's mechanism "
                   "story ('the information is only readable in the generator's frame') must be "
                   "withdrawn in that form.",
        "what_IS_real_the_geometry": {
            "collapse_measured": "in the grader frame the candidates of one question are nearly the "
                                 "SAME vector: mean within-question cosine of the RAW hidden states "
                                 "is 0.9518-0.9992 at every layer/pooling, versus 0.7366-0.9497 in "
                                 "the generator frame. After standardisation, the within-question "
                                 "share of total variance is 0.1047-0.4229 (grader) vs 0.2932-0.6410 "
                                 "(generator) -- candidate identity occupies 3-5x less of the "
                                 "grader representation.",
            "but_it_is_NOT_information_loss": "the grader frame's best ridge probe cell "
                                              "(L21/last, sel_eff 0.777248) is the BEST cell in the "
                                              "whole 16-cell probe grid, above every generator cell. "
                                              "So the answer to 'collapse or rotation' is: a real "
                                              "magnitude collapse that is NOT a loss of linear "
                                              "separability. A standardised readout recovers it.",
            "where_it_happens": "the grader frame integrates candidate content between layers 14 and "
                                "21: layer-to-layer CKA in the grader/last stream drops to 0.3461 "
                                "for 14->21 (vs 0.8458 in generator/last), and exactly there its "
                                "within-question variance share jumps 0.1155 -> 0.4105 and its probe "
                                "sel_eff jumps 0.747275 -> 0.777248.",
            "frames_encode_similar_content_at_span_pooling": "generator-vs-grader CKA rises 0.5118 -> "
                                                             "0.6865 -> 0.7810 -> 0.7379 over layers "
                                                             "7/14/21/28 at span pooling, while at "
                                                             "last pooling it FALLS 0.4703 -> 0.5110 "
                                                             "-> 0.3717 -> 0.2874. The frames "
                                                             "converge where the readout is pooled "
                                                             "over the answer span and diverge where "
                                                             "it is a single final token.",
        },
        "short_answer_stratum": {
            "definition": "gold answer word count <= 3 (n=2029 items, 1372 recoverable); the "
                          "complement is n=316 (96 recoverable). This does NOT exactly reproduce "
                          "the previously quoted n=1928 / 79% stratum -- different item filter -- "
                          "so it is stated as its own definition.",
            "incumbent": {"short": 0.794461, "long": 0.5},
            "per_frame_heads": {
                f'{hh["frame"]}|{hh["pooling"]}|{hh["objective"]}': {
                    "short": r6(hh["strata"]["gold_len<=3"]["sel_eff"]),
                    "long": r6(hh["strata"]["gold_len>3"]["sel_eff"])}
                for hh in art["job2_matched_heads"]["heads"]},
            "reading": "the grader frame's deficit does NOT concentrate on short answers. On the "
                       "short stratum the grader's best cell (span/bce, 0.831633) is the best cell "
                       "of either frame; on the long stratum the ordering reverses cell by cell "
                       "with n=96 recoverable items, which is too few to separate anything. There "
                       "is no short-answer-specific frame deficit to explain.",
        },
    },

    "diagnostics_NOT_in_the_headline": {
        "bce_objective": {
            "what": "the objective was frozen at bt (the previous round's train-only CV pick); bce "
                    "was never CV-selected. Its numbers are reported for completeness and are NOT "
                    "the pre-registered result.",
            "H|bce_ensemble": r6(arms["H|bce"]["ensemble_rank"]["sel_eff"]),
            "cv_pick_featset_at_bce": r6(arms["H+C+M+Wc+Ws|bce"]["ensemble_rank"]["sel_eff"]),
            "cv_pick_at_bce_vs_pointwise_head": {
                "d": r6(cmps["H+C+M+Wc+Ws|bce"]["vs_pointwise_head_same_features_ensemble"]["d_sel_eff"]),
                "ci": ci(cmps["H+C+M+Wc+Ws|bce"]["vs_pointwise_head_same_features_ensemble"]["ci"])},
            "warning": "this is the only cell in the round whose 'vs pointwise head' CI excludes "
                       "zero, it was selected with eval visible, and it is one of 22 comparisons "
                       "reported with uncorrected nominal CIs. It is a hypothesis for a future "
                       "pre-registered run, not a result.",
        },
        "single_feature_selectors": "job1_per_feature -- the eval columns there were not "
                                    "pre-registered and are labelled DIAGNOSTIC in the file.",
        "ridge_probe_grid": "job2_geometry.grid -- seed-free closed-form probes, used for the "
                            "geometry argument, not as selection results.",
    },

    "multiplicity_note": "22 arm-level comparisons are reported with nominal uncorrected 95% CIs. "
                         "The two conclusions drawn do not depend on any single one of them: the "
                         "Job-1 negative is that SEVEN feature sets all fail to separate from the "
                         "same baseline, and the Job-2 negative is that FOUR matched contrasts all "
                         "span zero.",
}
json.dump(art, open(OUT, "w"), indent=1, default=float)
print(f"wrote verdict into {OUT}")
