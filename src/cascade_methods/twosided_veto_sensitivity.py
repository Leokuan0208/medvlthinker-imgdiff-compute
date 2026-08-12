#!/usr/bin/env python3
"""
twosided_veto_sensitivity.py -- the two adversarial checks the ATTACK C result needs before its two
winning cells can be believed.  Merges its output into
results/cascade_methods/artifacts/twosided_veto_2026-08-11.json under `part6_sensitivities`.

Only two of the eight cells produced a positive nested arbitration gain:
    PMC_VQA        +0.01264 [+0.0095, +0.0158]
    PATH_VQA_open  +0.02633 [+0.0147, +0.0383]   (greedy 7B side) / +0.03647 (best-of-8 7B side)
Each has a specific, named way of being an artifact rather than accuracy.

S1  PMC_VQA vs the MedEvalKit colon-scoring defect (D1).  MedEvalKit's judge_multi_choice mis-scores a
    response of the form "C:" (utils.py:111-112), and the 32B writes "X:" on 61.1% of PMC items versus
    the 7B's 28.4%, so the canonical labels are biased toward the 7B on exactly this cell.  The arbiter
    is gated on ANSWER DISAGREEMENT, so it cannot fire on a "C." vs "C:" pair directly -- but the
    LABELS it is trained and scored on are still MedEvalKit's.  S1 re-runs the whole PMC arbiter with
    strict leading-option-letter labels for BOTH models (valid here: 33,429 of 33,430 responses begin
    with a valid option letter) and asks whether the win survives.

S2  The open cells vs the judge's known length/verbosity channel.  CLAUDE.md flags a live style/length
    grading channel in the open-text judge.  The top two-sided signal on two of the three open cells is
    literally `32B|answer_len` (AUROC 0.5677 / 0.7500).  S2 re-runs the open arbiters with EVERY
    length-derived feature deleted and asks whether the win survives on answer content alone.

Same protocol as the parent file: nested 5x5 CV, 10 fold seeds, paired item bootstrap nboot=10000,
threads pinned to 1, float64, fixed random_state.

Launch from the repo root:  python3 src/cascade_methods/twosided_veto_sensitivity.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import twosided_veto as T  # noqa: E402

R = T.R


def strict_pmc():
    """PMC_VQA rebuilt with strict leading-option-letter labels for both models."""
    d = T.mcq_cell("PMC_VQA", "PMC_VQA", None)

    def lead(x):
        s = str(x.get("response", "")).strip()
        return s[0].upper() if s and s[0].isalpha() else None

    d = dict(d)
    d["ok7"] = np.array([1.0 if lead(x) == x["answer"] else 0.0 for x in d["raw7"]])
    d["ok32"] = np.array([1.0 if lead(x) == x["answer"] else 0.0 for x in d["raw32"]])
    return d


def drop_length_features(d):
    d = dict(d)
    d["feats"] = {k: v for k, v in d["feats"].items() if "len" not in k.lower()}
    return d


def run_cell(d, arms, rng, label):
    per_seed, gbar = [], np.zeros(len(d["ok7"]))
    for s in range(T.NSEED):
        g, _, _ = T.nested_cv_cell(d, arms, T.SEED + 1000 * s)
        per_seed.append(float(g.mean()))
        gbar += g / T.NSEED
    ok7, ok32 = d["ok7"], d["ok32"]
    dec = ok7 != ok32
    p10 = float(((ok7 == 1) & (ok32 == 0)).mean())
    p01 = float(((ok7 == 0) & (ok32 == 1)).mean())
    T.log(f"  {label}: nested {np.mean(per_seed):+.5f}")
    return dict(
        acc=dict(always_7b=R(ok7.mean()), always_32b_direct=R(ok32.mean())),
        oracle_gain_p10=R(p10),
        base_rate_pi=R(p10 / (p10 + p01)) if (p10 + p01) > 0 else None,
        n_decisive=int(dec.sum()),
        features_used=sorted(d["feats"]),
        nested_gain_seed_mean=R(float(np.mean(per_seed)), 5),
        nested_gain_seed_sd=R(float(np.std(per_seed)), 5),
        nested_gain_seed_range=[R(float(np.min(per_seed)), 5), R(float(np.max(per_seed)), 5)],
        ci=T.boot_mean_ci(gbar, rng))


def main():
    rng = np.random.default_rng(T.SEED + 7)
    out = {}

    T.log("S1  PMC_VQA under a punctuation-robust scorer ...")
    base = T.mcq_cell("PMC_VQA", "PMC_VQA", None)
    strict = strict_pmc()
    s1 = dict(
        question="is PMC_VQA's +0.01264 arbitration win an artifact of MedEvalKit's colon mis-scoring?",
        scorer_a_medevalkit_judge_multi_choice=run_cell(base, T.ARMS, rng, "PMC medevalkit scorer"),
        scorer_b_strict_leading_option_letter=run_cell(strict, T.ARMS, rng, "PMC strict-letter scorer"),
        validity_of_scorer_b=dict(
            rows_whose_first_char_is_not_a_valid_option_letter=int(sum(
                1 for x in base["raw7"] + base["raw32"]
                if not (str(x.get("response", "")).strip()[:1].isalpha()
                        and str(x.get("response", "")).strip()[:1].upper() in "ABCD")),),
            of_total=int(len(base["raw7"]) + len(base["raw32"])),
            note="PMC-VQA responses are of the form 'X.' / 'X:' / 'X: <choice text>'; the leading "
                 "character is the model's option choice in all but one of 66,860 rows, so a strict "
                 "leading-letter match is a sound alternative scorer FOR THIS CELL.  It is a "
                 "SENSITIVITY, not a re-scoring of the paper: MedEvalKit's scorer remains the "
                 "canonical protocol and MedEvalKit was not modified."))
    a, b = s1["scorer_a_medevalkit_judge_multi_choice"], s1["scorer_b_strict_leading_option_letter"]
    s1["verdict"] = dict(
        gain_medevalkit=a["nested_gain_seed_mean"], gain_strict=b["nested_gain_seed_mean"],
        change=R(b["nested_gain_seed_mean"] - a["nested_gain_seed_mean"], 5),
        p10_medevalkit=a["oracle_gain_p10"], p10_strict=b["oracle_gain_p10"],
        model_gap_medevalkit=R(a["acc"]["always_32b_direct"] - a["acc"]["always_7b"]),
        model_gap_strict=R(b["acc"]["always_32b_direct"] - b["acc"]["always_7b"]),
        survives=bool(b["ci"]["lo"] is not None and b["ci"]["lo"] > 0))
    out["S1_pmc_scorer_sensitivity"] = s1

    T.log("S2  open cells with every length feature deleted ...")
    s2 = {}
    for variant in ("greedy", "bo8"):
        for cell in T.OPEN:
            d = T.open_cell(cell, variant)
            with_len = run_cell(d, T.OPEN_ARMS, rng, f"{cell}/{variant} with length")
            no_len = run_cell(drop_length_features(d), T.OPEN_ARMS, rng, f"{cell}/{variant} no length")
            s2[f"{cell}|{variant}"] = dict(
                with_length_features=with_len, without_length_features=no_len,
                change=R(no_len["nested_gain_seed_mean"] - with_len["nested_gain_seed_mean"], 5),
                survives=bool(no_len["ci"]["lo"] is not None and no_len["ci"]["lo"] > 0))
    out["S2_open_length_channel_sensitivity"] = dict(
        question="is the open-text arbitration gain riding the judge's known style/length grading "
                 "channel (CLAUDE.md: src/labeling/run_openvqa.py:26/27) rather than answer content?",
        dropped_features="every feature whose name contains 'len' (7B|pick_len, 7B|answer_len, "
                         "32B|answer_len, X|d_len)",
        per_cell=s2)

    path = T.OUT
    art = json.load(open(path))
    art["part6_sensitivities"] = dict(
        reproduce="python3 src/cascade_methods/twosided_veto_sensitivity.py",
        protocol="identical to part3: nested 5x5 CV, 10 fold seeds, paired item bootstrap nboot=10000",
        **out)
    with open(path, "w") as f:
        json.dump(art, f, indent=1)
    T.log(f"MERGED into {path}")
    print(json.dumps({k: (v.get("verdict") if k.startswith("S1") else
                          {kk: dict(with_len=vv["with_length_features"]["nested_gain_seed_mean"],
                                    no_len=vv["without_length_features"]["nested_gain_seed_mean"],
                                    survives=vv["survives"])
                           for kk, vv in v["per_cell"].items()})
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
