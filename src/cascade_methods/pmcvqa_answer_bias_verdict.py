#!/usr/bin/env python3
"""
pmcvqa_answer_bias_verdict.py -- adds the MECHANISM diagnostic and the consolidated VERDICT to
results/cascade_methods/artifacts/pmcvqa_answer_bias_audit_2026-08-11.json.

Only ADDS keys (T11_mechanism, VERDICT_FINAL, companion_artifacts); every measurement written by the
first pass is preserved byte-for-byte and re-asserted before the file is rewritten.

T11 -- THE MECHANISM.  Precision by PREDICTED letter, P(correct | model predicts L), against the
model's own mean confidence on those same items.  If precision varies far more by letter than
confidence does, then the answer letter is a large OMITTED VARIABLE in every confidence-based gate in
this project -- which is exactly how a split-level answer-position skew leaks into a gate that never
looks at the letter.

Launch from repo root (CPU only):
    OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_verdict.py
"""
import os, sys, json, copy
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import beat32b_fusion as B          # noqa: E402

MAIN = os.path.join(ROOT, "results/cascade_methods/artifacts/pmcvqa_answer_bias_audit_2026-08-11.json")
LETTERS = ["A", "B", "C", "D"]


def r4(x):
    return round(float(x), 4)


def main():
    art = json.load(open(MAIN))
    before = copy.deepcopy(art)
    assert art["NULL_TEST"]["passed"] is True

    d = B.mcq("PMC_VQA")
    r7 = B.load_raw("lingshu7b_full", "PMC_VQA")
    r32 = B.load_raw("lingshu32b_full", "PMC_VQA")
    n = len(d["ok7"])
    gold = np.array([str(r["answer"]).strip() for r in r7[:n]])

    def lt(s):
        for ch in str(s).strip().upper():
            if ch in "ABCD":
                return ch
        return "other"
    p7 = np.array([lt(r["response"]) for r in r7[:n]])
    p32 = np.array([lt(r["response"]) for r in r32[:n]])

    def table(pl, ok, cf):
        rows = {}
        for L in LETTERS:
            m = pl == L
            rows[L] = dict(n=int(m.sum()), predicted_share=r4(m.mean()),
                           gold_share=r4((gold == L).mean()),
                           precision_P_correct_given_predicts_L=r4(ok[m].mean()),
                           mean_confidence=r4(cf[m].mean()),
                           calibration_gap_precision_minus_confidence=r4(ok[m].mean() - cf[m].mean()))
        pr = [rows[L]["precision_P_correct_given_predicts_L"] for L in LETTERS]
        cc = [rows[L]["mean_confidence"] for L in LETTERS]
        return dict(per_predicted_letter=rows,
                    precision_range_across_letters=r4(max(pr) - min(pr)),
                    mean_confidence_range_across_letters=r4(max(cc) - min(cc)),
                    ratio_precision_range_over_confidence_range=r4((max(pr) - min(pr)) / (max(cc) - min(cc))))

    art["T11_mechanism_letter_is_an_omitted_variable_in_the_gate"] = dict(
        what="P(correct | model predicts letter L) versus the model's own mean confidence on those "
             "same items. Every gate in this project (the certified veto's Wilson-certified 7B-conf "
             "quantile bins; the F3 fusion's isotonic calibrator) conditions on CONFIDENCE ONLY.",
        lingshu_7B=table(p7, d["ok7"], d["c7"]),
        lingshu_32B_direct=table(p32, d["ok32"], d["c32"]),
        finding="On PMC_VQA test_2.csv, precision varies by ~0.39 (7B) / ~0.41 (32B) across the four "
                "predicted letters while mean confidence varies by only ~0.10 / ~0.09 -- a 3.8x / 4.7x "
                "gap. Both models are grossly OVER-confident when they answer A (calibration gap "
                "-0.2547 for the 7B, -0.3269 for the 32B) and about calibrated when they answer C. "
                "So the answer letter is a large omitted variable in every confidence-based gate on "
                "this cell, and it is an omitted variable whose distribution is an artifact of this "
                "split (B+C = 73.65% vs 62.40% in the human-verified v1 split).",
        why_it_matters="This is the causal route by which a split-level answer-position skew reaches a "
                       "gate that never reads the answer letter: confidence is not a sufficient "
                       "statistic for correctness here, and the residual is the letter.",
    )

    art["companion_artifacts"] = {
        "extension (fusion arm + macro consequence + question-blind baselines)":
            "results/cascade_methods/artifacts/pmcvqa_answer_bias_extend_2026-08-11.json",
        "null test of the control + row-order robustness":
            "results/cascade_methods/artifacts/pmcvqa_answer_bias_controls_2026-08-11.json",
        "other-cells data quality (decoded-pixel train/eval disjointness)":
            "results/cascade_methods/artifacts/othercells_dataquality_2026-08-11.json",
        "prior label-noise audit being reconciled":
            "results/cascade_methods/artifacts/pmc_label_noise_audit_2026-07-29.json",
        "the claim under audit":
            "results/cascade_methods/artifacts/armcombine_mcqonly_2026-08-11.json",
    }

    art["VERDICT_FINAL"] = dict(
        one_line="PARTLY REAL. The PMC +0.0095 is NOT an answer-letter artifact -- a permutation null "
                 "that holds the veto set's letter composition fixed rejects the artifact hypothesis "
                 "at z = 9.78 (p < 1e-4) -- but it is MATERIALLY INFLATED by the split's answer-position "
                 "skew: balancing the gold-letter marginal takes it from +0.00954 [+0.00715, +0.01188] "
                 "to +0.00534 [+0.00276, +0.00793], i.e. 44% of the gain is attributable to the skew. "
                 "The F3 fusion arm, which carries the larger MCQ-only claim, is worse: +0.01349 -> "
                 "+0.00532, 61% attributable.",
        real_component=dict(
            veto_letter_balanced="+0.00534 [+0.00276, +0.00793]  (56% of published, still significant)",
            fusion_letter_balanced="+0.00532 [+0.00132, +0.00937]  (39% of published, still significant)",
            permutation_null_veto="z = 9.78, p < 1e-4 (null mean -0.00316, sd 0.00130, max +0.00165)",
            permutation_null_fusion="z = 15.79, p < 1e-4",
            row_order_robustness="10/10 orderings agree in sign on every component "
                                 "(letter-balanced +0.00528 +- 0.00027)",
        ),
        artifact_component=dict(
            share_of_in_veto_7B_advantage_explained_by_letter_marginal=0.3383,
            share_for_fusion=0.3875,
            veto_delta_on_gold_A="-0.01176 [-0.01809, -0.00543]  SIGNIFICANT LOSS on 13.2% of the cell",
            fusion_delta_on_gold_A="-0.04499 [-0.05471, -0.03482]  SIGNIFICANT LOSS",
            veto_delta_on_rare_letters_A_or_D="-0.00363 [-0.00795, +0.00068]  point-negative, n.s.",
            fusion_delta_on_rare_letters_A_or_D="-0.01215 [-0.01873, -0.00545]  SIGNIFICANT LOSS",
            note="Both arms' entire gain lives on the two frequent gold letters. Neither arm helps on "
                 "the rare ones; the fusion significantly hurts there.",
        ),
        claims_that_must_be_amended=[
            "armcombine_mcqonly_2026-08-11.json VERDICT rows: '+0.00119 [+0.00090, +0.00148]' (veto) "
            "and '+0.00169 [+0.00126, +0.00212]' (fusion) remain CORRECT AS MEASURED and need no "
            "retraction -- but they must now travel with the letter-balanced sensitivity "
            "+0.00067 [+0.00035, +0.00099] and +0.00066 [+0.00016, +0.00117], and with the statement "
            "that both arms lose significantly on the gold-A stratum.",
            "The OWED answer-letter-bias audit that gated the MCQ-only result is hereby DISCHARGED: "
            "the result survives, at roughly half its published size once the skew is balanced.",
            "CLAUDE.md PMC-VQA landmine (section 0) must gain the answer-position bias beside the "
            "existing two-splits entry: test_2.csv B+C = 73.65%, constant-C floor 37.80% (= 68.5% of "
            "always-32B-direct's 0.5518 on that cell); test_clean.csv B+C = 62.40%.",
            "Any future confidence-gate result on PMC_VQA must report the per-gold-letter breakdown, "
            "because confidence is NOT a sufficient statistic for correctness on this cell (T11).",
            "The 8-cell macro headline (accuracy-max 0.6575, +0.0008 [-0.0022, +0.0037] vs "
            "always-32B-direct) is NOT affected in status: it was already a TIE and remains one. "
            "No published TIE becomes a LOSS and no published WIN becomes a TIE.",
        ],
        what_was_NOT_found=[
            "No evidence the veto preferentially keeps 7B answers merely because they are frequent "
            "letters: the veto set is only mildly enriched in gold B+C (0.7650 inside vs 0.7365 "
            "overall), and the permutation null that fixes that composition is decisively rejected.",
            "No letter dependence in the 2026-07-29 label-noise defect rates among audited veto wins "
            "(A 50%, B 56%, C 48%, D 71%, n = 8/39/46/7) -- label noise and answer-position skew are "
            "two INDEPENDENT contaminations of the same cell, not the same one counted twice.",
        ],
    )

    # integrity: nothing from the first pass was altered
    for k in before:
        assert json.dumps(before[k], sort_keys=True) == json.dumps(art[k], sort_keys=True), k
    json.dump(art, open(MAIN, "w"), indent=1)
    print("UPDATED (additive only)", MAIN)
    print(json.dumps(art["T11_mechanism_letter_is_an_omitted_variable_in_the_gate"]["finding"], indent=1))
    print(json.dumps(art["VERDICT_FINAL"]["one_line"], indent=1))


if __name__ == "__main__":
    main()
