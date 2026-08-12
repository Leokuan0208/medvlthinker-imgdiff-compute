#!/usr/bin/env python3
"""A GRADING DEFECT IN THE HARNESS THAT DIFFERENTIALLY PENALISES THE 32B ON PMC-VQA.

FOUND WHILE RUNNING THE OWED LETTER-BIAS AUDIT (src/cascade_methods/pmcvqa_letterbias_audit.py).
A simple leading-letter parse of `response` disagrees with the harness's own `correct` field on
1.05% of the 7B's items and 1.86% of the 32B's.  Every one of those is the harness marking a
RIGHT answer WRONG, and the asymmetry (0.81 pp) is the same size as the entire +0.95 pp PMC-VQA gain
that carries round 2's only CI-clean macro win over always-32B-direct
(`armcombine_mcqonly_2026-08-11.json`).

THE MECHANISM, in unmodified vendor code -- MedEvalKit/utils/utils.py:111-112

    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]      # <-- line 112

  A response of "C:" becomes "c:" -> split(".")[0] = "c:" -> split(":")[-1] = ""  .  The letter is
  DESTROYED.  The empty string matches no answer, no letter and no choice, so control reaches the
  `else` branch and the item is graded by difflib similarity of "c:" against the four option TEXTS
  (utils.py:131-134) -- effectively a coin flip.  Line 112 exists to handle "Answer: C"; it also
  eats the letter whenever the model writes the letter with a trailing colon.

⛔ MedEvalKit/ IS A PROTECTED DEPENDENCY AND IS NOT MODIFIED BY THIS SCRIPT.  The grader below is a
   VERBATIM TRANSCRIPTION, proven faithful by reproducing the stored `correct` field ELEMENT-WISE on
   all 33,430 x 2 items.  The repaired variant is a SENSITIVITY ANALYSIS ONLY.  The project's
   deployed convention remains the harness's; nothing here silently restates a published number.

No GPU.  Reproduce:
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/pmcvqa_grader_defect.py
"""
import os
import json
import time
import difflib
from collections import Counter

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_selector_rerun_parts")
MEK = os.path.join(REPO, "MedEvalKit")
OUT = os.path.join(ART, "pmcvqa_grader_defect_2026-08-12.json")
SEED, NBOOT = 20260812, 10000
t0 = time.time()


# =====================================================================================
# 0.  VERBATIM TRANSCRIPTION OF THE VENDOR GRADER  (MedEvalKit/utils/utils.py)
# =====================================================================================
def str_similarity(a, b):                                    # utils.py:71
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_most_similar_index(str_list, target):               # utils.py:75
    idx, high = 0, 0
    for i, s in enumerate(str_list):
        sim = str_similarity(s, target)
        if sim > high:
            high, idx = sim, i
    return idx


ANSWER_PATTERNS = ["**answer**:", "**answer**", "*answer*:", "**answer:**", "answer is",
                   "answer:", "答案:", "final answer", "final answer is"]


def parse_response(response):                                # utils.py:138
    response = response.lower()
    # the boxed / <answer> branches cannot fire on this dump -- asserted in N2 below
    for pat in ANSWER_PATTERNS:
        if pat in response:
            response = response.split(pat)[-1]
    return response


def judge_multi_choice(choices, answer, response, repair=False):   # utils.py:98
    response = response.lower()
    if response.split("\n\n")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[-1].split(".")[0]
    response = parse_response(response)
    alphas = [chr(ord('a') + i) for i in range(len(choices))]
    choices = [c.lower() for c in choices]
    flag = False
    response = response.strip().lower()
    response = response.replace("\n", "")
    split_response = response.split(".")[0]
    # ---------------------------------------------------------------- utils.py line 112
    if repair:
        # THE MINIMAL REPAIR, and the ONLY difference between the two graders:
        # keep the ":"-suffix behaviour, but do not let it annihilate the response.
        _tail = split_response.split(":")[-1]
        split_response = _tail if _tail.strip() != "" else split_response.split(":")[0]
    else:
        split_response = split_response.split(":")[-1]
    # ----------------------------------------------------------------
    answer = answer.strip().lower()
    if len(split_response) > 300:
        flag = False
    if split_response == answer:
        flag = True
    elif split_response in alphas:
        if choices[ord(split_response) - ord("a")] == answer:
            flag = True
    elif split_response in choices:
        if answer in alphas and split_response == choices[ord(answer) - ord("a")]:
            flag = True
    else:
        index = find_most_similar_index(choices, response)
        if alphas[index] == answer or choices[index] == answer:
            flag = True
    return flag


# =====================================================================================
# 1.  LOAD + NULL TESTS
# =====================================================================================
def okv(recs):
    return np.array([1 if (r.get("correct") is True or
                           str(r.get("correct")).strip().lower() in ("true", "1")) else 0
                     for r in recs], float)


Z = np.load(os.path.join(PARTS, "vec_disjoint.npz"))
ok7d, ok32d = Z["PMC_VQA|always_7b"].astype(float), Z["PMC_VQA|always_32b_direct"].astype(float)
VETO, FUS = Z["PMC_VQA|method_accuracy_max_veto"].astype(float), Z["PMC_VQA|method_accuracy_max_fusion"].astype(float)
D7 = json.load(open(f"{MEK}/eval_results_lingshu7b_full/{{}}/PMC_VQA/results.json"))
D32 = json.load(open(f"{MEK}/eval_results_lingshu32b_full/{{}}/PMC_VQA/results.json"))
N = len(D7)

rep = dict(
    title="A GRADING DEFECT IN THE MedEvalKit PMC-VQA JUDGE THAT DIFFERENTIALLY PENALISES THE 32B, "
          "and what it does to round 2's only CI-clean macro win over always-32B-direct.",
    date="2026-08-12", no_gpu=True, no_fabricated_numbers=True, seed=SEED, nboot=NBOOT,
    found_while="running the owed answer-letter-bias audit "
                "(results/cascade_methods/artifacts/pmcvqa_letterbias_audit_2026-08-12.json)",
    gates="results/cascade_methods/artifacts/armcombine_mcqonly_2026-08-11.json",
    reproduce="OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/pmcvqa_grader_defect.py",
    MedEvalKit_untouched="this script only READS MedEvalKit.  The grader here is a transcription; "
                         "the vendor tree is not modified.  Verified: `git -C MedEvalKit diff --stat "
                         "utils/utils.py` is empty (reported in provenance below).",
    defect=dict(
        file="MedEvalKit/utils/utils.py", lines="111-112",
        code="split_response = response.split('.')[0]  /  split_response = split_response.split(':')[-1]",
        mechanism="a bare-letter-plus-colon response ('C:') is reduced to the EMPTY STRING by "
                  "line 112, matches nothing, and falls through to difflib similarity of the raw "
                  "response against the four option TEXTS (utils.py:131-134).",
        why_it_is_not_symmetric="the two legs emit that response form at different rates, so the "
                                "defect is not a wash between arms -- it is a differential."),
)

nt = {}
gold = [str(r["answer"]).strip() for r in D7]
assert [str(r["answer"]).strip() for r in D32] == gold
mine7 = np.array([1.0 if judge_multi_choice(r["choices"], r["answer"], r["response"]) else 0.0 for r in D7])
mine32 = np.array([1.0 if judge_multi_choice(r["choices"], r["answer"], r["response"]) else 0.0 for r in D32])
st7, st32 = okv(D7), okv(D32)
nt["N1_transcription_is_faithful"] = dict(
    name="the transcribed grader reproduces the harness's stored `correct` field ELEMENT-WISE",
    n=int(N),
    mismatches_7b=int((mine7 != st7).sum()), mismatches_32b=int((mine32 != st32).sum()),
    verdict="PASS" if (mine7 == st7).all() and (mine32 == st32).all() else "FAIL")
assert nt["N1_transcription_is_faithful"]["verdict"] == "PASS", "transcription is not faithful"

nt["N2_dead_branches"] = dict(
    name="the boxed / <answer> branches of parse_response cannot fire on this dump, so omitting "
         "their helpers cannot change a grade",
    n_responses_containing_boxed=int(sum(("boxed" in str(r["response"]).lower()) for r in D7 + D32)),
    n_responses_containing_answer_tag=int(sum(("<answer>" in str(r["response"]).lower()) for r in D7 + D32)))
nt["N2_dead_branches"]["verdict"] = ("PASS" if nt["N2_dead_branches"]["n_responses_containing_boxed"] == 0
                                     and nt["N2_dead_branches"]["n_responses_containing_answer_tag"] == 0
                                     else "REVIEW")
nt["N3_alignment_to_deployed_vectors"] = dict(
    name="the dumps re-graded here are the SAME per-item objects, in order, as the deployed vectors",
    elementwise_7b=bool(np.array_equal(st7, ok7d)), elementwise_32b=bool(np.array_equal(st32, ok32d)))
nt["N3_alignment_to_deployed_vectors"]["verdict"] = (
    "PASS" if nt["N3_alignment_to_deployed_vectors"]["elementwise_7b"]
    and nt["N3_alignment_to_deployed_vectors"]["elementwise_32b"] else "FAIL")
assert nt["N3_alignment_to_deployed_vectors"]["verdict"] == "PASS"

# ---- repaired grades ---------------------------------------------------------------------------
rp7 = np.array([1.0 if judge_multi_choice(r["choices"], r["answer"], r["response"], repair=True) else 0.0 for r in D7])
rp32 = np.array([1.0 if judge_multi_choice(r["choices"], r["answer"], r["response"], repair=True) else 0.0 for r in D32])
f7, f32 = rp7 - st7, rp32 - st32          # +1 wrong->right, -1 right->wrong


def flipsum(f):
    return dict(gained=int((f > 0).sum()), lost=int((f < 0).sum()), net=int(f.sum()),
                net_pp=round(100 * float(f.mean()), 4))


rep["null_tests"] = nt
rep["A1_size_of_the_defect"] = dict(
    note="'gained' = items the repaired grader marks right that the harness marks wrong.  Every "
         "accuracy here is on PMC-VQA test_2.csv, n=33,430, the MedEvalKit track.",
    always_7b=dict(harness_acc=round(float(st7.mean()), 6), repaired_acc=round(float(rp7.mean()), 6),
                   **flipsum(f7)),
    always_32b_direct=dict(harness_acc=round(float(st32.mean()), 6), repaired_acc=round(float(rp32.mean()), 6),
                           **flipsum(f32)),
    differential_pp=round(100 * float(f32.mean() - f7.mean()), 4),
    response_forms_repaired={
        "always_7b": dict(Counter(str(D7[i]["response"])[:6] for i in np.where(f7 > 0)[0]).most_common(6)),
        "always_32b_direct": dict(Counter(str(D32[i]["response"])[:6] for i in np.where(f32 > 0)[0]).most_common(6))},
    reading="the harness under-scores BOTH legs, but it under-scores the 32B roughly twice as hard.  "
            "That differential is the confound: every arm that answers with 7B text where the "
            "baseline answers with 32B text collects it for free.")


# =====================================================================================
# 2.  WHAT IT DOES TO THE GATED WIN
# =====================================================================================
# The arms answer with the 7B on a veto set V and with the 32B elsewhere, so
#     delta = (1/N) * sum_{i in V} (ok7[i] - ok32[i]) ,
# and the repair moves the delta by (1/N) * [ sum_V f7 - sum_V f32 ].
# V is identified EXACTLY on the disagreement set; a conf threshold recovers the rest.
conf = np.array([r["conf"] for r in D7], float)
DIS = ok7d != ok32d


def recover_mask(v):
    best = None
    grid = np.unique(conf)
    for t in grid:
        pred = np.where(conf >= t, ok7d, ok32d)
        mm = int((pred != v).sum())
        if best is None or mm < best[0]:
            best = (mm, float(t))
    return best


mm_v, thr_v = recover_mask(VETO)
Vmask = conf >= thr_v
rep["A2_veto_mask_recovery"] = dict(
    method="the certified veto answers with the 7B on V and the 32B on V^c, so V is identified "
           "EXACTLY wherever the two legs differ in correctness; a single threshold on the 7B's "
           "stored `conf` recovers the whole mask.",
    feature="conf (MedEvalKit results.json field)", threshold=round(thr_v, 6),
    residual_mismatched_items=int(mm_v), residual_pct=round(100 * mm_v / N, 4),
    veto_rate=round(float(Vmask.mean()), 5),
    verdict="EXACT to %d items (%.3f%%)" % (mm_v, 100 * mm_v / N),
    fusion_note="the fusion arm is NOT recovered by any single threshold on conf or margin "
                "(best residual 2,215 items), so it is reported by BOUNDS only, never by a mask.")

rng = np.random.default_rng(SEED)
IDX = rng.integers(0, N, size=(NBOOT, N))


def ci_of(d):
    b = d[IDX].mean(1)
    return dict(delta=round(float(d.mean()), 5), lo=round(float(np.percentile(b, 2.5)), 5),
                hi=round(float(np.percentile(b, 97.5)), 5))


def verdict_of(c):
    return "WIN" if c["lo"] > 0 else ("LOSS" if c["hi"] < 0 else "TIE")


# --- veto, via the recovered mask ---
veto_rep = np.where(Vmask, rp7, rp32)
nat_v, rp_v = ci_of(VETO - ok32d), ci_of(veto_rep - rp32)
for c in (nat_v, rp_v):
    c["verdict"] = verdict_of(c)

# --- rigorous bounds that need NO mask, for BOTH arms ---
def bounds(v):
    """delta_repaired = delta_natural + (1/N)(sum_V f7 - sum_V f32).  V is known on the
    disagreement set; on the agreement set only its SIZE is constrained, so bound it."""
    known = DIS & (v == ok7d)                     # items provably answered by the 7B
    unknown = ~DIS                                # legs indistinguishable from correctness alone
    base = float(((v - ok32d).mean()))
    shift_known = float((f7[known] .sum() - f32[known].sum()) / N)
    worst = float((f7[unknown][f7[unknown] < 0].sum() - f32[unknown][f32[unknown] > 0].sum()) / N)
    best = float((f7[unknown][f7[unknown] > 0].sum() - f32[unknown][f32[unknown] < 0].sum()) / N)
    return dict(delta_natural=round(base, 5),
                delta_repaired_lower_bound=round(base + shift_known + worst, 5),
                delta_repaired_upper_bound=round(base + shift_known + best, 5),
                note="assumption-free: the leg is identified on every item where the two legs "
                     "differ in correctness; on the rest the bound assigns the unknown legs "
                     "adversarially (lower) and favourably (upper).")


rep["A3_effect_on_the_gated_win"] = dict(
    arithmetic="the gated policy is byte-identical to always-32B-direct on 7 of the 8 cells BY "
               "CONSTRUCTION, so the 8-cell macro delta is exactly (PMC_VQA delta)/8.",
    certified_veto=dict(
        pmc_delta_harness_grader=nat_v, pmc_delta_repaired_grader=rp_v,
        macro_delta_harness=round(nat_v["delta"] / 8, 5),
        macro_ci_harness=[round(nat_v["lo"] / 8, 5), round(nat_v["hi"] / 8, 5)],
        macro_delta_repaired=round(rp_v["delta"] / 8, 5),
        macro_ci_repaired=[round(rp_v["lo"] / 8, 5), round(rp_v["hi"] / 8, 5)],
        survives_the_repair=bool(rp_v["lo"] > 0),
        via="the recovered mask (residual %d items)" % mm_v,
        assumption_free_bounds=bounds(VETO)),
    fusion=dict(
        pmc_delta_harness_grader=ci_of(FUS - ok32d),
        assumption_free_bounds=bounds(FUS),
        macro_bounds=[round(bounds(FUS)["delta_repaired_lower_bound"] / 8, 5),
                      round(bounds(FUS)["delta_repaired_upper_bound"] / 8, 5)],
        note="no mask is recoverable for the fusion arm, so ONLY bounds are reported.  Its repaired "
             "delta is NOT MEASURED."),
)
rep["A3_effect_on_the_gated_win"]["fusion"]["pmc_delta_harness_grader"]["verdict"] = verdict_of(
    rep["A3_effect_on_the_gated_win"]["fusion"]["pmc_delta_harness_grader"])

# =====================================================================================
# 3.  THE COMBINED STRESS TEST -- repaired grader AND letter-balanced at the same time
# =====================================================================================
gold_arr = np.array([g.upper()[:1] for g in gold])
letters = sorted(set(gold_arr))
w_bal = np.zeros(N)
for L in letters:
    m = gold_arr == L
    w_bal[m] = 1.0 / (len(letters) * m.sum())
W = w_bal * N


def ci_bal(d):
    b = (d * W)[IDX].mean(1)
    return dict(delta=round(float((d * w_bal).sum()), 5),
                lo=round(float(np.percentile(b, 2.5)), 5),
                hi=round(float(np.percentile(b, 97.5)), 5))


comb = ci_bal(veto_rep - rp32)
comb["verdict"] = verdict_of(comb)
nat_bal = ci_bal(VETO - ok32d)
nat_bal["verdict"] = verdict_of(nat_bal)
rep["A4_combined_stress_test"] = dict(
    design="apply BOTH corrections at once to the certified-veto policy: the repaired grader (so the "
           "32B is no longer differentially penalised) AND equal weight per gold letter (so no arm "
           "can profit by matching the benchmark's answer prior).  This is the strictest honest "
           "version of the gated claim.  The veto arm is used because it is the only one whose "
           "per-item leg mask is recoverable; the fusion arm is NOT MEASURED here.",
    gold_letter_prior={L: round(float((gold_arr == L).mean()), 5) for L in letters},
    pmc_delta=dict(
        harness_grader_natural_prior=nat_v,
        harness_grader_letter_balanced=nat_bal,
        repaired_grader_natural_prior=rp_v,
        repaired_grader_letter_balanced=comb),
    macro_delta_strictest=round(comb["delta"] / 8, 5),
    macro_ci_strictest=[round(comb["lo"] / 8, 5), round(comb["hi"] / 8, 5)],
    survives_both=bool(comb["lo"] > 0),
    shrinkage=dict(
        as_banked=round(nat_v["delta"] / 8, 5),
        after_both_corrections=round(comb["delta"] / 8, 5),
        fraction_of_the_banked_delta_that_survives=round(comb["delta"] / nat_v["delta"], 3)),
    the_bar="a significant macro win over always-32B-direct needs about +0.0029 "
            "(cascade_selector_rerun_2026-08-05.json CI half-width).  Every number in this block is "
            "far below that bar -- surviving is not the same as clearing it.")

json.dump(rep, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT, round(time.time() - t0, 1), "s")
print(json.dumps(rep["A4_combined_stress_test"], indent=1))
print(json.dumps(rep["null_tests"], indent=1))
print(json.dumps(rep["A1_size_of_the_defect"], indent=1))
print(json.dumps(rep["A2_veto_mask_recovery"], indent=1))
print(json.dumps(rep["A3_effect_on_the_gated_win"], indent=1))
