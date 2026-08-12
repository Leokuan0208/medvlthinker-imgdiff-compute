#!/usr/bin/env python3
"""OWED AUDIT -- answer-letter bias on PMC-VQA test_2.csv, and whether the round-2 MCQ-only win
survives letter-balancing.

WHY THIS EXISTS.  Round 2's only CI-clean macro win over always-32B-direct
(`armcombine_mcqonly_2026-08-11.json`: MCQ-half = the shipped certified veto, open half =
always-32B-direct, +0.00119 [+0.0009, +0.00148] at 0.977x FLOP-eq) is, by construction,
byte-identical to the baseline on 7 of the 8 cells.  100% of it is PMC_VQA -- which in the
MedEvalKit track is `test_2.csv`, 33,430 items, the split with ZERO published verification
(CLAUDE.md two-split landmine; docs/current/PMCVQA_PROVENANCE_2026-07-30.md).
`docs/current/BEAT32B_ROUND_2026-08-10.md:26` gates that win behind an answer-letter-bias audit and
quotes "B+C = 73.6%, 37.8% constant-C floor" WITHOUT NAMING AN ARTIFACT.  This script is that
artifact: it re-derives those two numbers from the harness dump, and then runs the test the gate is
actually about -- does the +0.0135 PMC_VQA gain survive equalising the gold-letter prior?

THE DECISIVE TEST.  If the shipped arm's advantage over always-32B-direct is that it shifts
predicted-letter mass toward the over-represented gold letters, then re-weighting items so each gold
letter carries equal weight will destroy the gain.  If the gain is a capability gain it will survive.
Both the per-gold-letter deltas and the letter-balanced delta are reported with paired bootstrap CIs.

No GPU.  Pure numpy over the stored harness dumps.
Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/pmcvqa_letterbias_audit.py
"""
import os
import re
import json
import time
from collections import Counter

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_selector_rerun_parts")
MEK = os.path.join(REPO, "MedEvalKit")
OUT = os.path.join(ART, "pmcvqa_letterbias_audit_2026-08-12.json")

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
BASE = "always_32b_direct"
ARMS = ["always_7b", "always_32b_direct", "always_32b_reasoning",
        "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]
SEED, NBOOT = 20260812, 10000
t0 = time.time()
rep = dict(
    title="OWED AUDIT -- answer-letter bias on PMC-VQA test_2.csv, and whether the round-2 "
          "MCQ-only macro win over always-32B-direct survives letter-balancing.",
    date="2026-08-12",
    gates="results/cascade_methods/artifacts/armcombine_mcqonly_2026-08-11.json",
    owed_by="results/cascade_methods/docs/current/BEAT32B_ROUND_2026-08-10.md:26 -- which quotes "
            "'B+C = 73.6%, 37.8% constant-C floor' with NO named artifact.  This file supplies it.",
    reproduce="OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/pmcvqa_letterbias_audit.py",
    no_gpu=True, no_fabricated_numbers=True, seed=SEED, nboot=NBOOT,
    numerics_pins=dict(OMP_NUM_THREADS="1", PYTHONHASHSEED="0",
                       tf32="not applicable -- pure numpy over stored 0/1 vectors and strings",
                       bootstrap="paired item-level, one shared index stream reused by every arm"),
)


# =====================================================================================
# 0.  LOAD  +  NULL TESTS
# =====================================================================================
def okv(recs):
    return np.array([1 if (r.get("correct") is True or
                           str(r.get("correct")).strip().lower() in ("true", "1")) else 0
                     for r in recs], float)


Z = np.load(os.path.join(PARTS, "vec_disjoint.npz"))
VEC = {a: Z[f"PMC_VQA|{a}"].astype(float) for a in ARMS}
N = len(VEC[BASE])

D7 = json.load(open(f"{MEK}/eval_results_lingshu7b_full/{{}}/PMC_VQA/results.json"))
D32 = json.load(open(f"{MEK}/eval_results_lingshu32b_full/{{}}/PMC_VQA/results.json"))
ok7, ok32 = okv(D7), okv(D32)

nt = {}
nt["N1_item_alignment"] = dict(
    name="the harness dump used for the letter analysis is the SAME per-item object, in the SAME "
         "order, as the deployed evaluation vectors -- asserted ELEMENT-WISE, not just in mean",
    source="MedEvalKit/eval_results_lingshu{7b,32b}_full/{}/PMC_VQA/results.json vs "
           "_selector_rerun_parts/vec_disjoint.npz",
    n_dump_7b=len(ok7), n_dump_32b=len(ok32), n_vec=int(N),
    elementwise_identical_7b=bool(np.array_equal(ok7, VEC["always_7b"])),
    elementwise_identical_32b=bool(np.array_equal(ok32, VEC[BASE])),
    acc_7b=round(float(ok7.mean()), 6), acc_32b=round(float(ok32.mean()), 6))
assert nt["N1_item_alignment"]["elementwise_identical_7b"], "7B dump does not align to the deployed vector"
assert nt["N1_item_alignment"]["elementwise_identical_32b"], "32B dump does not align to the deployed vector"
nt["N1_item_alignment"]["verdict"] = "PASS (exact, asserted -- the script raises otherwise)"

PUB = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))["per_arm"]["disjoint"]
pubcell = PUB["per_cell"]["method_accuracy_max_veto"]["PMC_VQA"] if isinstance(
    PUB.get("per_cell", {}).get("method_accuracy_max_veto", None), dict) else None
MQ = json.load(open(os.path.join(ART, "armcombine_mcqonly_2026-08-11.json")))
dev2 = max(abs(VEC[a].mean() - MQ["E7_crossfit_CI_foldseed_averaged"]["per_cell"]["PMC_VQA"]["acc"])
           for a in ["method_accuracy_max_fusion"])
nt["N2_reproduce_banked_arm_accuracies"] = dict(
    name="reproduce the banked PMC_VQA accuracy of the arm the gated win deploys",
    fusion_acc_mine=round(float(VEC["method_accuracy_max_fusion"].mean()), 6),
    fusion_acc_banked=MQ["E7_crossfit_CI_foldseed_averaged"]["per_cell"]["PMC_VQA"]["acc"],
    veto_acc_mine=round(float(VEC["method_accuracy_max_veto"].mean()), 6),
    max_abs_dev=float(f"{dev2:.3g}"), verdict="PASS" if dev2 < 1e-5 else "FAIL")

# ---- gold letters, predictions, option counts -------------------------------------------------
LET = re.compile(r"^\s*([A-Z])\b")


def gold_of(r):
    return str(r.get("answer", "")).strip().upper()[:1]


def pred_of(r):
    s = str(r.get("response", "")).strip().upper()
    m = LET.match(s)
    return m.group(1) if m else "?"


gold = np.array([gold_of(r) for r in D7])
p7 = np.array([pred_of(r) for r in D7])
p32 = np.array([pred_of(r) for r in D32])
nopt = np.array([len(r.get("choices", [])) for r in D7])
assert [gold_of(r) for r in D32] == list(gold), "gold letters differ between the two dumps"

nt["N3_gold_consistency"] = dict(
    name="the gold letter is identical between the 7B and the 32B dump for every item",
    verdict="PASS (asserted)", n=int(N))
nt["N4_parse"] = dict(
    name="the leading-letter parse of `response` reproduces the harness's own `correct` field",
    agreement_7b=round(float((( p7 == gold).astype(float) == ok7).mean()), 6),
    agreement_32b=round(float((((p32 == gold).astype(float)) == ok32).mean()), 6),
    unparsed_7b=int((p7 == "?").sum()), unparsed_32b=int((p32 == "?").sum()),
    note="a parse agreement below 1.0 means the harness's grader accepted responses this simple "
         "leading-letter regex does not; the letter analysis below uses the HARNESS's `correct` "
         "field for every accuracy number and the regex only for the PREDICTED-letter histograms.")
nt["N4_parse"]["verdict"] = ("PASS" if min(nt["N4_parse"]["agreement_7b"],
                                           nt["N4_parse"]["agreement_32b"]) > 0.99 else "REVIEW")
rep["null_tests"] = nt


# =====================================================================================
# 1.  THE GOLD-LETTER PRIOR OF test_2.csv   (re-derives the two quoted numbers)
# =====================================================================================
gc = Counter(gold)
letters = sorted(gc)
prior = {L: gc[L] / N for L in letters}
const = {L: float((gold == L).mean()) for L in letters}
best_const = max(const, key=lambda L: const[L])
bc = prior.get("B", 0.0) + prior.get("C", 0.0)

rep["A1_gold_letter_prior"] = dict(
    source="MedEvalKit/eval_results_lingshu7b_full/{}/PMC_VQA/results.json -- the `answer` field, "
           "n=33,430, i.e. PMC-VQA test_2.csv as loaded by unmodified vendor code "
           "(MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39)",
    n=int(N), counts={L: int(gc[L]) for L in letters},
    fraction={L: round(prior[L], 5) for L in letters},
    B_plus_C=round(bc, 5), B_plus_C_pct=round(100 * bc, 1),
    constant_letter_baseline_accuracy={L: round(const[L], 5) for L in letters},
    best_constant_letter=best_const,
    best_constant_letter_accuracy=round(const[best_const], 5),
    options_per_item=dict(Counter(nopt.tolist())),
    uniform_would_be=round(1.0 / len(letters), 5),
    max_deviation_from_uniform=round(max(abs(prior[L] - 1.0 / len(letters)) for L in letters), 5),
    QUOTED_IN_DOC=dict(
        claim="BEAT32B_ROUND_2026-08-10.md:26 -- 'B+C = 73.6%, 37.8% constant-C floor'",
        B_plus_C_quoted=0.736, B_plus_C_measured=round(bc, 5),
        constant_C_quoted=0.378, constant_C_measured=round(const.get("C", float("nan")), 5)),
)
_q = rep["A1_gold_letter_prior"]["QUOTED_IN_DOC"]
_q["B_plus_C_abs_dev"] = round(abs(_q["B_plus_C_quoted"] - _q["B_plus_C_measured"]), 5)
_q["constant_C_abs_dev"] = round(abs(_q["constant_C_quoted"] - _q["constant_C_measured"]), 5)
_q["verdict"] = ("CONFIRMED" if max(_q["B_plus_C_abs_dev"], _q["constant_C_abs_dev"]) <= 0.001
                 else "DOES NOT REPRODUCE -- the doc's parenthetical is not what the dump says")


# =====================================================================================
# 2.  PREDICTED-LETTER DISTRIBUTIONS  (is the cheap leg the biased one?)
# =====================================================================================
def hist(p):
    c = Counter(p)
    return {L: round(c.get(L, 0) / N, 5) for L in letters + (["?"] if (p == "?").any() else [])}


def tv(p):
    c = Counter(p)
    return round(0.5 * sum(abs(c.get(L, 0) / N - prior[L]) for L in letters), 5)


rep["A2_predicted_letter_distribution"] = dict(
    note="how much probability mass each leg puts on each letter, against the gold prior.  TV = "
         "total-variation distance to the gold prior; SMALLER means the leg's output distribution "
         "matches the benchmark's answer prior more closely.",
    gold_prior={L: round(prior[L], 5) for L in letters},
    always_7b=hist(p7), always_7b_TV_to_gold_prior=tv(p7),
    always_32b_direct=hist(p32), always_32b_direct_TV_to_gold_prior=tv(p32),
    reading="the certified veto imports the 7B's answer wherever it fires, so if the 7B's letter "
            "distribution were the better-matched one, a naive prior-matching story would predict "
            "the veto's gain.  The decisive test is A3, not this table.")


# =====================================================================================
# 3.  THE DECISIVE TEST -- does the gain survive equalising the gold-letter prior?
# =====================================================================================
rng = np.random.default_rng(SEED)
IDX = rng.integers(0, N, size=(NBOOT, N))
masks = {L: (gold == L) for L in letters}
w_bal = np.zeros(N)
for L in letters:
    w_bal[masks[L]] = 1.0 / (len(letters) * masks[L].sum())


def boot_pair(v, b):
    """paired item bootstrap of mean(v-b) under (i) the natural prior and (ii) letter-balanced."""
    d = v - b
    nat = d[IDX].mean(1)
    W = w_bal * N
    bal = (d * W)[IDX].mean(1)
    return nat, bal


def blk(name, v):
    b = VEC[BASE]
    d = v - b
    nat_pt = float(d.mean())
    bal_pt = float((d * w_bal).sum())
    nat, bal = boot_pair(v, b)
    per = {}
    for L in letters:
        m = masks[L]
        dl = d[m]
        bi = rng.integers(0, m.sum(), size=(2000, int(m.sum())))
        bs = dl[bi].mean(1)
        per[L] = dict(n=int(m.sum()), acc_arm=round(float(v[m].mean()), 5),
                      acc_direct=round(float(b[m].mean()), 5), delta=round(float(dl.mean()), 5),
                      lo=round(float(np.percentile(bs, 2.5)), 5),
                      hi=round(float(np.percentile(bs, 97.5)), 5))
        per[L]["verdict"] = ("WIN" if per[L]["lo"] > 0 else
                             ("LOSS" if per[L]["hi"] < 0 else "TIE"))
    return dict(
        arm=name, acc=round(float(v.mean()), 6),
        natural_prior=dict(delta=round(nat_pt, 5), lo=round(float(np.percentile(nat, 2.5)), 5),
                           hi=round(float(np.percentile(nat, 97.5)), 5)),
        letter_balanced=dict(delta=round(bal_pt, 5), lo=round(float(np.percentile(bal, 2.5)), 5),
                             hi=round(float(np.percentile(bal, 97.5)), 5)),
        shrinkage_natural_minus_balanced=round(nat_pt - bal_pt, 5),
        per_gold_letter=per)


for x in (rep["A1_gold_letter_prior"], rep["A2_predicted_letter_distribution"]):
    pass

rep["A3_does_the_gain_survive_letter_balancing"] = dict(
    design="every accuracy is the HARNESS's own `correct` field.  'natural prior' = the benchmark as "
           "shipped (each item weight 1/33430).  'letter-balanced' = each of the 4 gold letters "
           "carries weight 1/4, items inside a letter equally -- so an arm that wins only by putting "
           "more mask on the over-represented letters cannot profit.  Paired item-level bootstrap, "
           "nboot=10000, one shared index stream across every arm.",
    arms={a: blk(a, VEC[a]) for a in ARMS if a != BASE},
)


# =====================================================================================
# 4.  WHAT IT MEANS FOR THE GATED MACRO WIN
# =====================================================================================
def macro_delta(pmc_delta):
    """the MCQ-only policy is byte-identical to always-32B-direct on the other 7 cells, so the
    8-cell macro delta is exactly the PMC_VQA delta / 8."""
    return pmc_delta / 8.0


rows = []
for a, lab in (("method_accuracy_max_veto", "MCQ = certified veto (accuracy-max), OPEN = always-32B-direct"),
               ("method_accuracy_max_fusion", "MCQ = fusion (accuracy-max+), OPEN = always-32B-direct")):
    B = rep["A3_does_the_gain_survive_letter_balancing"]["arms"][a]
    rows.append(dict(
        policy=lab, arm=a,
        macro_delta_natural=round(macro_delta(B["natural_prior"]["delta"]), 5),
        macro_ci_natural=[round(macro_delta(B["natural_prior"]["lo"]), 5),
                          round(macro_delta(B["natural_prior"]["hi"]), 5)],
        macro_delta_letter_balanced=round(macro_delta(B["letter_balanced"]["delta"]), 5),
        macro_ci_letter_balanced=[round(macro_delta(B["letter_balanced"]["lo"]), 5),
                                  round(macro_delta(B["letter_balanced"]["hi"]), 5)],
        survives=bool(macro_delta(B["letter_balanced"]["lo"]) > 0)))

banked = MQ["VERDICT"]["rows"]
rep["A4_consequence_for_the_gated_win"] = dict(
    arithmetic="the gated policy is byte-identical to always-32B-direct on 7 of 8 cells BY "
               "CONSTRUCTION, so its 8-cell macro delta is exactly (PMC_VQA delta)/8.  Re-weighting "
               "PMC_VQA therefore re-weights the entire claim.",
    banked_rows_for_comparison=[dict(policy=r["policy"], delta=r["delta"], ci=r["ci"],
                                     flopeq_x=r["flopeq_x"]) for r in banked],
    rows=rows,
    cross_check=dict(
        note="the natural-prior macro delta recomputed here must reproduce the banked one",
        veto_mine=rows[0]["macro_delta_natural"], veto_banked=banked[0]["delta"],
        abs_dev=round(abs(rows[0]["macro_delta_natural"] - banked[0]["delta"]), 6),
        fusion_mine=rows[1]["macro_delta_natural"], fusion_banked=banked[1]["delta"],
        abs_dev_fusion=round(abs(rows[1]["macro_delta_natural"] - banked[1]["delta"]), 6)),
)

json.dump(rep, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT, round(time.time() - t0, 1), "s")
print(json.dumps(rep["A1_gold_letter_prior"], indent=1))
print(json.dumps(rep["A4_consequence_for_the_gated_win"]["rows"], indent=1))
for a, B in rep["A3_does_the_gain_survive_letter_balancing"]["arms"].items():
    print(a, "nat", B["natural_prior"], "bal", B["letter_balanced"])
