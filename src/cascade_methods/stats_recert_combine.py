#!/usr/bin/env python3
"""
ATTACK C -- combine the four re-certification parts into the single deliverable artifact
results/cascade_methods/artifacts/stats_recertification_2026-08-11.json

Every number here is copied verbatim from the four part artifacts in
results/cascade_methods/artifacts/_stats_recert/; nothing is recomputed and nothing is rounded
differently.  Each part carries its own NULL TEST and the combined artifact refuses to declare
`all_null_tests_passed` unless all four passed.

Launch from the repo root:  python3 src/cascade_methods/stats_recert_combine.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_recert_common import ART, CELLS_OPEN, META, jdump

OUT = os.path.join(ART, "stats_recertification_2026-08-11.json")


def L(name):
    return json.load(open(os.path.join(META, name)))


def main():
    p1, p2, p3, p4 = L("part1_oracle.json"), L("part2_cluster.json"), L("part3_bom.json"), L("part4_router.json")
    nulls = {"part1_oracle_noise": p1["null_test"], "part2_cluster_resampling": p2["null_test"],
             "part3_best_of_majority": p3["null_test"], "part4_pre_generation_router": p4["null_test"]}
    allpass = all(bool(n["passed"]) for n in nulls.values())

    # ---------------------------------------------------------------- part 2 headline table ----
    def sch(src, claim, s):
        return p2["results"][src]["schemes"][s]["all8"][claim]

    claims = ["method_accuracy_max_veto - always_32b_direct",
              "method_accuracy_max_veto - always_32b_reasoning",
              "method_compute_lean - always_32b_direct",
              "method_accuracy_max_fusion - always_32b_direct"]
    tbl = {}
    for src in ("disjoint", "ens8_scaled"):
        for c in claims:
            r = {s: sch(src, c, s) for s in ("item", "image", "cell")}
            tbl[f"{src} | {c}"] = dict(
                point=r["item"]["delta"],
                item=[r["item"]["lo"], r["item"]["hi"], r["item"]["verdict"]],
                image_cluster=[r["image"]["lo"], r["image"]["hi"], r["image"]["verdict"]],
                cell_cluster=[r["cell"]["lo"], r["cell"]["hi"], r["cell"]["verdict"]],
                ci_width_item=r["item"]["width"], ci_width_image=r["image"]["width"],
                ci_widening_image_over_item=round(r["image"]["width"] / r["item"]["width"], 3),
                ci_width_cell=r["cell"]["width"],
                verdict_changed_under_image=bool(r["item"]["verdict"] != r["image"]["verdict"]),
                verdict_changed_under_cell=bool(r["item"]["verdict"] != r["cell"]["verdict"]))

    P = p4["variants"]["prompt_text_only_PRIMARY"]

    res = {
        "title": "ATTACK C -- re-certifying this project's statistics against the four 2026 "
                 "corrections flagged by LITERATURE_UPDATE_2026-08-11.md",
        "date": "2026-08-11",
        "no_gpu": True, "no_new_inference": True,
        "reproduce": ["python3 src/cascade_methods/stats_recert_meta.py",
                      "python3 src/cascade_methods/stats_recert_p1_oracle.py 10000",
                      "python3 src/cascade_methods/stats_recert_p2_cluster.py 10000",
                      "python3 src/cascade_methods/stats_recert_p3_bom.py 10000",
                      "python3 src/cascade_methods/stats_recert_p4_router.py",
                      "python3 src/cascade_methods/stats_recert_combine.py"],
        "preregistration": "results/cascade_methods/artifacts/"
                           "stats_recertification_2026-08-11_preregistration.json "
                           "(written before any part-3 / part-4 number existed)",
        "all_null_tests_passed": allpass,
        "null_tests": nulls,

        "HEADLINE": {
            "part1_oracle_noise": (
                "THE CORRECTION DOES NOT APPLY TO US, AND IT STRENGTHENS OUR HEADROOM CLAIMS. "
                "Across 4,323 (item, answer) pairs independently re-judged by a second judge pass, "
                "the judge disagreed with itself on 0.10-0.20%%; of the pairs our pool called "
                "CORRECT the independent pass contradicted 0.00%%/0.57%%/0.00%%; and of the 120 "
                "LONE-correct samples (k=1 items -- the exact population arXiv:2607.03436 is about) "
                "ZERO were contradicted. Item-level: 1,319 of 1,319 confirmable recoverable items "
                "confirmed, 0 contradicted. So the raw oracle is NOT inflated by label noise. "
                "SAMPLING fragility is real but nearly self-cancelling at the pool level: "
                "E[oracle@8 on a FRESH 8] is 0.8742/0.6255/0.5065 vs the raw 0.8791/0.6300/0.5167, "
                "moving the perfect-selection ceiling from +0.0301 to +0.0276 -- and those "
                "per-cell differences are INSIDE our own +/-0.008/cell serving-config band, so the "
                "pool-level correction is NOT established. What IS established (immune to that "
                "caveat, since it compares groups inside the same replicate pool) is the "
                "COMPOSITION: an item recoverable via a single lucky sample replicates in a fresh "
                "pool only 53-58%% of the time, versus 99.7-99.8%% for k>=4 items; and items with "
                "NO correct answer in our pool still yield one 11-22%% of the time, so the "
                "'coverage wall' leaks in both directions."),
            "part2_cluster_resampling": (
                "NO CLAIM LOSES SIGNIFICANCE. Under image-cluster resampling the CIs widen by only "
                "6-19%% and every verdict is unchanged. Under the brutal cell-cluster bootstrap "
                "(the 8 reporting cells themselves are the sampling units) the vs-REASONING win "
                "SURVIVES (+0.0601 [+0.0072,+0.1285] disjoint; +0.0615 [+0.0068,+0.1330] "
                "ens8_scaled), the vs-DIRECT tie stays a tie, and only ONE verdict moves: "
                "compute-lean vs direct goes from a SIGNIFICANT LOSS to a TIE for ens8_scaled. "
                "The RouteGuard failure mode does not reproduce here -- because the cell that "
                "dominates our item count, PMC-VQA, has 1.15 questions per image."),
            "part3_best_of_majority": (
                "PRE-REGISTERED NEGATIVE, AND IT IS A LARGE ONE. Best-of-Majority filtering HURTS "
                "every arm it is applied to: the pre-registered primary BoM-c2 costs the incumbent "
                "verifier -0.0307 [-0.0490,-0.0143] sel_eff and the deployed fused selector "
                "-0.0511 [-0.0680,-0.0350], with per-cell guardrail losses in both. Every one of "
                "the five sweep variants is a significant loss for both selectors; nested CV "
                "grouped by image cluster confirms it. DIAGNOSIS: our verifier is already strictly "
                "better than answer frequency, so a frequency pre-filter can only delete "
                "correct-but-rare candidates it would have found. Pure majority vote scores "
                "sel_eff 0.7139 against the incumbent's 0.7752 and the deployed 0.8106."),
            "part4_pre_generation_router": (
                "THE STRUCTURAL THEOREM REPRODUCES ON OUR DATA, AND IT IS WORSE FOR US THAN THE "
                "LITERATURE SAYS: THE THING THAT BEATS OUR CASCADE ON COST IS A COIN FLIP. A "
                "prompt-only, 5-fold cross-fitted, image-cluster-grouped pre-generation router "
                "matches compute-lean's macro accuracy (0.6443) at 3.713 macro FLOP-eq versus "
                "compute-lean's 6.674 -- a 1.80x saving -- and matches the fusion arm at 4.140 vs "
                "7.766 (1.88x). BUT it is statistically INDISTINGUISHABLE from its within-cell "
                "permutation null at every budget (p = 0.105 / 0.403 / 0.876 at macro cost <= 2.0 / "
                "3.0 / 4.57), and a RANDOM-ALLOCATION FLOOR -- send a uniformly random fraction of "
                "every cell's traffic to the 32B, no model, no features, no training -- gets 1.74x "
                "and 1.85x of the same savings. The learned router's entire margin over coin-"
                "flipping is +0.0067 / +0.0030 / +0.000004 macro accuracy at those three budgets, "
                "and the permutation null says even that is not significant. Neither the router nor "
                "the coin flip can reach accuracy-max's 0.6575: any 7B/32B mixing policy is bounded "
                "above by always-32B-direct's 0.6567, so the cascade's +0.0008 comes from its "
                "open-arm best-of-N and NOT from routing."),
        },

        "part1_oracle_noise": p1,
        "part2_cluster_resampling": dict(
            cluster_stats=p2["cluster_stats"], cluster_definition=p2["cluster_definition"],
            headline_table=tbl, full=p2),
        "part3_best_of_majority": p3,
        "part4_pre_generation_router": p4,

        "effect_on_published_claims": {
            "STRENGTHENED": [
                "'+0.0615 [+0.0514,+0.0715] vs always-32B-with-reasoning' -- survives image-cluster "
                "resampling ([+0.0505,+0.0724]) AND the 8-unit cell bootstrap ([+0.0068,+0.1330]). "
                "This is now the most robust claim in the project.",
                "'the coverage wall is ~4.5x the selection wall' and the +0.0301 perfect-selection "
                "ceiling -- NOT inflated by label noise; the judge replicates at 99.8-99.9%% and "
                "contradicted 0 of 120 lone-correct samples.",
                "'the frozen 8-seed selector is worth +0.0354 pooled sel_eff' -- survives image "
                "clustering, [+0.0205,+0.0505] item vs [+0.0202,+0.0505] image.",
                "'identifiability, not the ceiling, is the binding limit on 7B-vs-32B routing' -- "
                "independently re-confirmed by part 4's permutation null: a prompt-only router has "
                "NO within-cell signal at all (p = 0.105 / 0.403 / 0.876), and a random-allocation "
                "floor captures ~97%% of its cost saving. The free upper bound of +0.0661 for "
                "perfect routing is reproduced exactly by part 4's null test.",
                "'selection efficiency 0.78-0.81 is a field constant' -- Best-of-Majority, the "
                "literature's proposed fix, makes it WORSE, not better.",
            ],
            "WEAKENED": [
                "The perfect-selection ceiling is +0.0301 raw but +0.0276 once the oracle is "
                "measured on an independent fresh pool, and +0.0113 under the most pessimistic "
                "bound (sampling correction applied only to already-recoverable items). Quote "
                "+0.0301 as the RAW pool oracle, not as 'the headroom a re-run would find'.",
                "'compute-lean loses significantly to always-32B-direct (-0.0091 [-0.0153,-0.0031])' "
                "-- for the ens8_scaled arm this becomes a TIE ([-0.0217,+0.0013]) when the 8 "
                "reporting cells are treated as the sampling units. The disjoint arm's loss "
                "survives. State which arm and which resampling unit.",
                "Any framing of the 8-sample coverage wall as a hard per-item property: items with "
                "no correct answer in our pool produce one in a fresh pool 11-22%% of the time.",
            ],
            "RETIRED": [],
            "NEW_THREAT_TO_A_PUBLISHED_COST_CLAIM": {
                "claim_threatened": "'Pareto-optimal' / 'non-dominated' language for the cascade's "
                                    "operating points on the macro cost-accuracy plane. "
                                    "(CLAUDE.md section 0 already RETIRED 'Pareto-dominates'; this "
                                    "threatens what was left.)",
                "finding": "Any policy that runs exactly ONE of the two models costs at most 4.57 "
                           "macro FLOP-eq, while the shipped cascade costs 6.674 (compute-lean), "
                           "7.766 (fusion) and 7.951 (accuracy-max), because a cascade pays the 7B "
                           "on every query before it can decide anything. Random allocation alone "
                           "-- no model, no features, no training -- reaches compute-lean's macro "
                           "accuracy at 3.829 FLOP-eq (1.74x cheaper) and the fusion arm's at "
                           "4.190 (1.85x cheaper).",
                "arithmetic_on_the_same_published_constants": (
                    "At the mixing rate that matches compute-lean's macro accuracy the expected "
                    "batch-1 latency is 0.2393*347 + 0.7607*665 = 589 ms and the expected energy "
                    "is 0.2393*45.8 + 0.7607*127.0 = 108 J, against compute-lean's measured 690.8 "
                    "ms parallel / 228.8 J (_selector_rerun_parts/summary_disjoint.json:cost_macro). "
                    "This is ARITHMETIC on the per-model constants, NOT an end-to-end measurement: "
                    "the router has never been run, so its latency and energy are NOT MEASURED."),
                "what_SURVIVES": "compute-lean and the fusion arm are dominated on macro cost at "
                                 "equal macro accuracy. accuracy-max is NOT: its 0.6575 is above "
                                 "the 0.6567 ceiling of every 7B/32B mixing policy, so it remains "
                                 "non-dominated -- but the part of it that is non-dominated is the "
                                 "open-arm best-of-N, not the routing.",
                "honest_reading": "This is a cost-axis finding about MACRO weighting on this "
                                  "benchmark mix. It does not touch the vs-reasoning latency and "
                                  "energy wins, and it is not a sample-weighted statement. It does "
                                  "mean that any future cost claim for compute-lean must be made "
                                  "against a random-allocation floor, not only against "
                                  "always-32B-direct."},
        },
        "caveats": [
            "Part 1's judge measurement is REPLICATION, not VALIDITY: it rules out stochastic label "
            "noise, not a systematically lenient grader.",
            "Part 1's independent pools (sc16 / sc32) were generated in a separate run; the "
            "project's own standing caveat is that a different serving configuration reproduces a "
            "cell to only +/-0.008. The pool-level sampling correction is inside that band and is "
            "reported as NOT established; the conditional (within-pool-B) results are not affected.",
            "Part 2's cell-cluster bootstrap has only 8 units. It is deliberately brutal and is a "
            "generalisation check, not a replacement for the item-level CI.",
            "Part 4's router is evaluated offline from frozen per-item correctness vectors. It has "
            "never been run end-to-end, so its accuracy is measured but its latency and energy are "
            "NOT MEASURED. Only FLOP-eq cost is reported, from the project's own constants.",
            "vqa_rad cells are n=200/251; per-cell guardrail flags there can be seed noise.",
        ],
        "sources": {
            "eval vectors": "results/cascade_methods/artifacts/_selector_rerun_parts/vec_{disjoint,ens8_scaled}.npz",
            "per-item clusters and prompts": "results/cascade_methods/artifacts/_stats_recert/meta_*.json "
                                             "(src/cascade_methods/stats_recert_meta.py)",
            "open-text 8-sample pools": "ckpts/train/lora_verifier_disjoint/transfer_dump_*.json",
            "independent replicate pools": "ckpts/openvqa/cheap_lingshu7b/ckpt_*_sc16.jsonl and "
                                           "ckpts/openvqa/cheap_lingshu7b_scale/ckpt_vqa_rad_open_lingshu7b_sc32.jsonl",
            "frozen selector (READ ONLY)": "ckpts/train/genframe_head_ens8/",
            "cost constants": "results/cascade_methods/artifacts/_selector_rerun_parts/summary_disjoint.json:cost_macro",
            "published ceilings": "results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json",
        },
    }
    jdump(res, OUT)
    print(f"all_null_tests_passed = {allpass}")
    for k, v in nulls.items():
        print(f"  {k:32s} passed={v['passed']}")


if __name__ == "__main__":
    main()
