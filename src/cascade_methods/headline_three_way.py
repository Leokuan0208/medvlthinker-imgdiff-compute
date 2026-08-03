#!/usr/bin/env python3
"""
headline_three_way.py -- THE HEADLINE THAT HAD NEVER BEEN COMPUTED:
    8-cell MACRO  x  CLEAN (disjoint-trained) L1 verifier  x  MATCHED-PROMPT reasoning baseline,
priced with the two grounded cost constants derived/measured on 2026-08-03.

WHY THIS EXISTS.
  The project's surviving claim is
      accuracy-max-veto beats always-32B-WITH-REASONING by +0.0601 [+0.0498, +0.0703]
      at -87.7% batch-1 latency and -84.3% energy
      (macro_headline_clean_verifier_2026-07-30.json, column C).
  That number applies TWO corrections (macro re-weighting; de-contaminated verifier) while the
  reasoning baseline it is measured against is still the UNMATCHED one -- a prompt whose answer-style
  clause differs from the direct arm's, which matched_prompt_reasoning_2026-07-29.json showed costs
  the reasoning arm a large amount of measured accuracy on open text
  (pooled open: unmatched 0.3028 vs matched-A 0.4235 / matched-B 0.4192, direct 0.5168).
  So +0.0601 is an UPPER BOUND, not a measurement.  The three corrections have never been applied
  together.  This script applies all three, and re-prices the result.

WHAT IS SWAPPED, AND HOW.
  Exactly as in macro_headline_clean_verifier.py: an EXACT-PATH REDIRECT installed on builtins.open,
  so not one line of the scoring / aggregation / costing machinery is duplicated.  Two redirect
  layers are stacked:
    (1) VERIFIER  (inherited, unchanged): ckpts/train/lora_verifier_pooled4/transfer_dump_*.json
                  -> ckpts/train/lora_verifier_disjoint/...            (clean L1)
    (2) REASONING BASELINE (new, this file): both files of the open-text 32B reasoning dump
                  ckpts/openvqa/strong_lingshu_think/ckpt_<ds>_lingshu32b_think.jsonl        (gen_tokens -> COST)
                  ckpts/openvqa/strong_lingshu_think/ckpt_<ds>_lingshu32b_think.judge.jsonl  (judge_ok -> ACCURACY)
                  -> the corresponding files of ckpts/openvqa/strong_lingshu_think_matched2/  (arm B, primary)
                  or ...strong_lingshu_think_matched/                                          (arm A)
  Because BOTH files are redirected, accuracy and cost move together: the matched arms emit fewer
  tokens (86.4 / 45.8 / 101.5 vs 122.4 / 104.5 / 141.5 mean generated tokens), so honest_recosting
  charges the matched reasoning baseline LESS, which shrinks the efficiency claim as well as the
  accuracy claim.  Nothing is hand-edited: honest_recosting.measure_reasoning_lengths() reads the
  redirected dump.

ARM CHOICE (primary = arm B), justified from matched_prompt_reasoning_2026-07-29.json:
  * the artifact itself designates `decisive_arm = "reason_matched_B"`;
  * prompt_design.SYS_THINK_MATCHED2: "SYS_THINK's trigger sentences kept VERBATIM (trace fires) with
    its answer-style clause replaced by the direct arm's persona/style" -- i.e. the reasoning TRIGGER,
    the thing under test, is byte-identical to the published reasoning arm and ONLY the answer-style
    clause is matched to the direct arm.  Arm A re-orders the prompt so the trigger no longer opens
    with the model's documented verbatim lead-in ("You will solve a problem/request.").
  Arm A is computed and reported alongside in every table.

HONESTY NOTE ON WHAT ARM B IS.  Arm B's <think> trace fires on only 67.0% / 30.5% / 71.1% of
SLAKE/VQA-RAD/PathVQA-open items (matched artifact, interpretation.trap_1), so it is a MIXTURE of
reasoning and direct answering, and its accuracy is therefore HIGHER than a fully-reasoning arm's.
Using it as the baseline makes the baseline STRONGER, so the resulting margin is a LOWER bound on
"method vs a 32B that always reasons".  That is the conservative direction and it is the point.

MULTIPLE-CHOICE HALF -- see `mcq_reasoning_baseline_analysis` in the JSON.  Short version: NO MCQ cell
is re-run, and that is not an omission.  Of the 5 Variant-B multiple-choice cells only ONE
(MedXpert) has a genuine reasoning baseline; the matched-prompt MCQ study matched the DIRECT arm to
the reasoning arm, leaving the reasoning arm untouched, and found 0/9 trigger effects significant.

COST CONSTANTS (grounded, 2026-08-03):
  * R32 = 3.816 FLOP-equivalents per 32B forward (DERIVED: exact safetensors parameter counts +
    measured prompt geometry; band [3.734, 3.859]) replaces the underived literal 4.57.
    Patched through flop_ratio_impact.set_R32 -- the same patcher that already passes a reproduction
    gate at 4.57.  NOTE: this is not cosmetic for accuracy either -- the Pandora open-arm controller
    optimises FLOPs, so R32 enters the policy.  Both values are run.
  * Fixed best-of-8 open arm: 1305.3 ms / 316.7 J per question, MEASURED (Lingshu-7B, HF batch-1,
    cap320, real vqa_rad images, NVML, n=45 / 2 replicates), replacing the asserted 522.0 ms
    (2.5x too low) and the modelled 8*(GEN7+VER7) = 568.8 J (1.8x too high).

VALIDATION (the run aborts on failure).
  GATE 1  (contaminated, unmatched, R32=4.57) must reproduce macro_average_headline_2026-07-30.json
          exactly -- every accuracy level, delta, CI bound, escalation and cost ratio.
  GATE 2  (clean L1,     unmatched, R32=4.57) must reproduce macro_headline_clean_verifier_2026-07-30.json
          column C exactly.
  GATE 3  the matched open-cell reasoning accuracies loaded through the redirect must equal the
          matched_prompt_reasoning_2026-07-29.json per-dataset accuracies.
  Any drift is a wiring error, not a finding.

NO GPU, no new inference, no fabricated numbers.  Launch from the repo root:
    python3 src/cascade_methods/headline_three_way.py
"""
import os, sys, json, time, copy

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SRC = os.path.join(ROOT, "src/cascade_methods")
sys.path.insert(0, SRC)

# macro_headline_clean_verifier installs the builtins.open redirect at import and pulls in the whole
# unchanged pipeline (macro_average_headline / paper_baselines / integrated_method / honest_recosting).
import macro_headline_clean_verifier as MHC
import macro_average_headline as MAH
import paper_baselines as PB
import integrated_method as IM
import flop_ratio_impact as FRI            # set_R32: the already-gated FLOP-constant patcher
import honest_recosting as HRC

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/headline_three_way_2026-08-03.json")
MATCHED_ART = os.path.join(ROOT, "results/cascade_methods/artifacts/matched_prompt_reasoning_2026-07-29.json")
MEDEVAL_ART = os.path.join(ROOT, "results/cascade_methods/artifacts/medeval_matched_direct_2026-07-29.json")
PUB_MACRO = os.path.join(ROOT, "results/cascade_methods/artifacts/macro_average_headline_2026-07-30.json")
PUB_CLEAN = os.path.join(ROOT, "results/cascade_methods/artifacts/macro_headline_clean_verifier_2026-07-30.json")
FLOP_DERIV = os.path.join(ROOT, "results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json")

ORDER, MCQ_B, OPEN_B = MHC.ORDER, MHC.MCQ_B, MHC.OPEN_B
SYSTEMS, METHODS = MHC.SYSTEMS, MHC.METHODS
HEADLINE_BASELINES = MHC.HEADLINE_BASELINES

R_OLD, R_NEW = 4.57, 3.816
BO8_MS_MEAS, BO8_J_MEAS = 1305.3, 316.7        # MEASURED batched best-of-8, 2026-08-03
BO8_MS_OLD, BO8_J_OLD = IM.BO8["ms"], 8 * (PB.GEN7[1] + PB.VER7[1])   # 522.0 ms / 568.8 J

OPEN_DSKEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}
THINK_ARMS = {
    "unmatched": ("strong_lingshu_think", "lingshu32b_think"),
    "matched_A": ("strong_lingshu_think_matched", "lingshu32b_think_matched"),
    "matched_B": ("strong_lingshu_think_matched2", "lingshu32b_think_matched2"),
}
_REAL_OPEN = MHC._REAL_OPEN


# =================================================================================================
# 0.  the second redirect layer: the open-text 32B REASONING dump (accuracy file AND cost file)
# =================================================================================================
_THINK_ARM = {"name": "unmatched"}
_MHC_SET_ARM = MHC.set_arm


def _think_paths(arm, dskey):
    d, tag = THINK_ARMS[arm]
    base = os.path.join(ROOT, "ckpts/openvqa", d, f"ckpt_{dskey}_{tag}")
    return os.path.abspath(base + ".jsonl"), os.path.abspath(base + ".judge.jsonl")


def _set_arm_with_think(adapter):
    """MHC.set_arm (verifier redirect; it CLEARS the map) then stack the reasoning-arm redirect."""
    _MHC_SET_ARM(adapter)
    arm = _THINK_ARM["name"]
    if arm == "unmatched":
        return
    for dskey in OPEN_DSKEY.values():
        s_gen, s_judge = _think_paths("unmatched", dskey)
        d_gen, d_judge = _think_paths(arm, dskey)
        for s, d in ((s_gen, d_gen), (s_judge, d_judge)):
            if not os.path.exists(d):
                raise SystemExit(f"missing matched reasoning dump: {d}")
            MHC._REDIRECT[s] = d


MHC.set_arm = _set_arm_with_think       # compute_arm() resolves set_arm from module globals


def set_R32(r):
    """FRI.set_R32 patches paper_baselines/integrated_method/beat32b_fusion/pandora_controller/
    method_final_mmmu_corrected/honest_recosting with ONE value.  The PUBLISHED pipeline is not
    internally consistent -- it carries 4.57 in paper_baselines/integrated_method and 4.571 in
    honest_recosting.flop_ratio_32b_over_7b (both underived; retrospective 7 hole 14c).  To reproduce
    the published artifacts EXACTLY at R32=4.57 the honest_recosting value must stay 4.571; the
    corrected runs use the single derived value everywhere, which also removes that inconsistency."""
    FRI.set_R32(r)
    if r == R_OLD:
        HRC.verify_constants = FRI._HR_VERIFY        # restore the published 4.571 in the recost model


def compute(adapter, think_arm):
    _THINK_ARM["name"] = think_arm
    res, vecs = MHC.compute_arm(adapter)
    return res


# =================================================================================================
# 1.  GATE 3 -- the redirected reasoning accuracies must equal the matched-prompt artifact's
# =================================================================================================
def gate3_matched_accuracies():
    art = json.load(_REAL_OPEN(MATCHED_ART))
    got, diffs = {}, []
    for cell, dskey in OPEN_DSKEY.items():
        got[cell] = {}
        for arm in THINK_ARMS:
            _, jp = _think_paths(arm, dskey)
            ok = [json.loads(l)["judge_ok"] for l in _REAL_OPEN(jp) if l.strip()]
            a = round(float(np.mean(ok)), 4)
            got[cell][arm] = dict(n=len(ok), acc=a)
            key = {"unmatched": "acc_reason_unmatched", "matched_A": "acc_reason_matched_A",
                   "matched_B": "acc_reason_matched_B"}[arm]
            exp = art["per_dataset"][cell][key]
            if abs(a - exp) > 1e-4:
                diffs.append(dict(cell=cell, arm=arm, recomputed=a, artifact=exp))
    return got, diffs


# =================================================================================================
# 2.  the multiple-choice half -- worked out explicitly, from the dumps
# =================================================================================================
def mcq_analysis():
    """Which Variant-B multiple-choice cells even HAVE a reasoning baseline, and what the matched
    protocol implies for each.  Every number read from the dumps / the MedEvalKit matched artifact."""
    import honest_recosting as HR
    meas = HR.measure_reasoning_lengths()
    med = json.load(_REAL_OPEN(MEDEVAL_ART))
    cellrec = {c["cell"]: c for c in med["cells"] if c["family"] == "Lingshu-32B"}

    per = {}
    for k in MCQ_B:
        m = meas[k]
        gr = m["gen_reasoning"]["mean"] if m["gen_reasoning"] else None
        per[k] = dict(
            n=m["n"], reasoning_dump=m["reasoning_dump"], verdict=m["verdict"],
            gen_tok_direct_mean=round(m["gen_direct"]["mean"], 2),
            gen_tok_reasoning_mean=(round(gr, 2) if gr is not None else None),
            prediction_agreement_with_direct=m["agreement"],
            acc_reasoning=m.get("acc_reasoning"), acc_direct=m.get("acc_direct"))
    # the only genuine multiple-choice reasoning cell in the Variant-B pool
    mx = cellrec["MedXpert-ALL"]
    per["MedXpertQA-MM"]["matched_prompt_study"] = dict(
        source="medeval_matched_direct_2026-07-29.json (Lingshu-32B, MedXpert-ALL, n=2000)",
        what_was_matched="the DIRECT arm was re-run with the reasoning arm's prompt minus the trigger "
                         "clause; the REASONING arm was not re-run and is byte-identical to the one "
                         "used here",
        acc_reason=mx["acc"]["reason"], acc_direct_matched=mx["acc"]["direct_matched"],
        acc_direct_unmatched=mx["acc"]["direct_unmatched"],
        trigger_effect_matched=mx["delta_matched"], format_effect=mx["delta_format"],
        verdict=mx["verdict"])

    return dict(
        question="What should the MATCHED reasoning baseline be on the 5 Variant-B multiple-choice cells?",
        answer=("It is the SAME vector as the unmatched one, and no MCQ cell is re-run here. Two "
                "independent reasons, both verifiable in `per_cell` below: (1) On PMC-VQA, "
                "SLAKE-closed and VQA-RAD-closed the dump used as 'always-32B-with-reasoning' "
                "(MedEvalKit/eval_results_lingshu32b_think) is NOT a reasoning run at all -- mean "
                "generated tokens 3.09-4.26, prediction agreement with the plain direct run 92-94%. "
                "There is no reasoning behaviour there to prompt-match; the arm differs from direct "
                "only in ANSWER FORMAT, which is precisely the confound the matched protocol removes, "
                "and removing it would collapse those cells onto always-32B-direct. PathVQA-closed "
                "has no reasoning dump at all and is already imputed reasoning = direct. (2) MedXpert "
                "is the one genuine reasoning cell (320.3 generated tokens). For it the matched-prompt "
                "study matched the DIRECT arm to the reasoning arm and left the reasoning arm "
                "untouched, so the matched protocol changes the reasoning baseline by exactly zero "
                "there; the trigger effect it measured is +0.0035 [-0.0185, +0.0250], not significant "
                "(0/9 cells significant across the 3 families x 3 cells it covered)."),
        genuine_reasoning_cells_in_variant_b=["MedXpertQA-MM"] + OPEN_B,
        share_of_pool_with_genuine_reasoning=dict(
            n_items=int(sum(meas[k]["n"] for k in ["MedXpertQA-MM"] + OPEN_B)),
            n_pool=int(sum(meas[k]["n"] for k in ORDER)),
            frac=round(sum(meas[k]["n"] for k in ["MedXpertQA-MM"] + OPEN_B) /
                       sum(meas[k]["n"] for k in ORDER), 4)),
        per_cell=per,
        sensitivity_medxpert_direct_matched=dict(
            what="If the MedXpert DIRECT baseline is also swapped to its matched version "
                 "(0.3065 -> 0.3005), always_32b_direct moves by -0.0060 on that one cell.",
            macro_effect_on_always_32b_direct=round(
                (mx["acc"]["direct_matched"] - mx["acc"]["direct_unmatched"]) / len(ORDER), 5),
            status="DERIVED from the artifact's own cell accuracies; not applied to the headline, "
                   "because it changes the DIRECT baseline, not the reasoning baseline this item is about."),
        not_applied_and_why=(
            "Setting okT := ok32 on the three NOT-REASONED multiple-choice cells (the strictest reading "
            "of the matched protocol) is a FOURTH correction and is reported as a labelled point-estimate "
            "sensitivity in `strictest_reading_sensitivity` of final_headline, not as the headline."))


# =================================================================================================
# 3.  extraction helpers
# =================================================================================================
def delta(res, m, b, ax="all8", lab="macro_cells"):
    d = res["deltas"][m][b][ax][lab]
    return dict(delta=d["delta"], ci95=[d["lo"], d["hi"]], verdict=d["verdict"])


def cost_row(res, conv, lab, s):
    return dict(res["cost"][conv][lab][s])


def ratio(res, conv, lab, m, b):
    return dict(res["cost"][conv][lab][m][b]) if False else \
        dict(res["cost"]["method_vs_baseline_ratios"][conv][lab][m][b])


BO8_SLOPE_MS = round((BO8_MS_MEAS - (PB.GEN7[0] + PB.VER7[0])) / 7.0, 2)   # ms per EXTRA batched draw


def corrected_lat_par(res, lab="macro_cells"):
    """Re-price the BATCHED-latency axis for the open cells with a measured-anchored draw model.

    The repo charges a batched round of N cheap draws the latency of ONE draw
    (pandora_controller.cost_of: lat_bat = (GEN7+VER7) + esc*GEN32, 'N drops out of latency').
    The 2026-08-03 measurement refutes that: a batched best-of-8 round takes 1305.3 ms, not 522.0.
    Two-point linear model, DERIVED:  L(N) = 522.0 + 111.9*(N-1)  [N=1 anchor = GEN7+VER7 = 522.0 ms
    repo constant; N=8 anchor = 1305.3 ms MEASURED].  MCQ cells run a single forward and are
    unaffected.  ENERGY is deliberately NOT re-priced: Pandora's draws are sequential, so each is a
    genuine batch-1 forward at the repo's per-draw energy; the measured batch-8 energy saving
    (39.6 vs 71.1 J/draw) comes from batching, which an adaptive-N policy cannot use.
    """
    pc = res["cost"]["per_cell_honest_recost"]
    meanN = res["escalation"]["open_meanN"]
    n = {k: res["cell_n"][k] for k in ORDER}
    tot = sum(n.values())
    w = {k: (n[k] / tot if lab == "sample_weighted" else 1.0 / len(ORDER)) for k in ORDER}
    out = {}
    for s in SYSTEMS:
        old = sum(pc[k][s]["lat_par_ms"] * w[k] for k in ORDER)
        new = 0.0
        for k in ORDER:
            v = pc[k][s]["lat_par_ms"]
            if k in OPEN_B and s in METHODS:
                v = v + BO8_SLOPE_MS * max(0.0, meanN[k] - 1.0)
            new += v * w[k]
        out[s] = dict(lat_par_ms_repo=round(old, 1), lat_par_ms_corrected=round(new, 1))
    for m in METHODS:
        for b in HEADLINE_BASELINES:
            out.setdefault("ratios", {}).setdefault(m, {})[b] = dict(
                pct_repo=round(-100 * (1 - out[m]["lat_par_ms_repo"] / out[b]["lat_par_ms_repo"]), 1),
                pct_corrected=round(-100 * (1 - out[m]["lat_par_ms_corrected"] /
                                            out[b]["lat_par_ms_corrected"]), 1))
    out["model"] = dict(status="DERIVED (two-point linear)",
                        anchor_N1_ms=PB.GEN7[0] + PB.VER7[0], anchor_N1_status="repo constant (GEN7+VER7)",
                        anchor_N8_ms=BO8_MS_MEAS, anchor_N8_status="MEASURED 2026-08-03",
                        slope_ms_per_extra_draw=BO8_SLOPE_MS, open_cell_meanN=meanN)
    return out


NOT_REASONED_MCQ = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed"]


def strictest_reading(res, lab="macro_cells"):
    """FOURTH correction, POINT ESTIMATE ONLY (no CI: the MCQ per-item vectors are not returned by
    compute_arm).  On PMC-VQA / SLAKE-closed / VQA-RAD-closed the 'reasoning' dump does not reason
    (3.09-4.26 generated tokens, 92-94% prediction agreement with direct), so the strictest reading of
    the matched protocol replaces those cells' reasoning vector with the direct one."""
    ac = res["acc_cell"]
    n = res["cell_n"]; tot = sum(n[k] for k in ORDER)
    w = {k: (n[k] / tot if lab == "sample_weighted" else 1.0 / len(ORDER)) for k in ORDER}
    base = sum(ac[k]["always_32b_reasoning"] * w[k] for k in ORDER)
    adj = sum((ac[k]["always_32b_direct"] if k in NOT_REASONED_MCQ
               else ac[k]["always_32b_reasoning"]) * w[k] for k in ORDER)
    out = dict(status="POINT ESTIMATE (no CI)", weighting=lab,
               cells_replaced=NOT_REASONED_MCQ,
               reasoning_baseline_acc_as_is=round(base, 4),
               reasoning_baseline_acc_strictest=round(adj, 4),
               shift=round(adj - base, 4))
    for m in METHODS:
        out[m] = dict(delta_as_is=round(sum(ac[k][m] * w[k] for k in ORDER) - base, 4),
                      delta_strictest=round(sum(ac[k][m] * w[k] for k in ORDER) - adj, 4))
    return out


def bo8_recost(res, lab="macro_cells"):
    """Re-price the FIXED best-of-8 open arm with the MEASURED batched constants (the old 522 ms /
    568.8 J were an assertion and a model).  escalation and FLOP-eq are unchanged."""
    b = res["bo8_open_arm"][lab]
    e = b["escalation"]
    return dict(escalation=e,
                lat_par_ms_old_assumed=b["lat_par_ms"],
                lat_par_ms_measured=round(BO8_MS_MEAS + e * IM.GEN32N["ms"], 1),
                energy_j_old_modelled=b["energy_j"],
                energy_j_measured=round(BO8_J_MEAS + e * PB.GEN32N[1], 1),
                flops=b["flops"], arm_accuracy=b["arm_accuracy"])


# =================================================================================================
# 4.  RUN
# =================================================================================================
def run():
    t0 = time.time()
    P = print

    # ---- GATE 3 first (cheap) --------------------------------------------------------------------
    matched_acc, g3 = gate3_matched_accuracies()
    if g3:
        for d in g3:
            P("   ", d)
        raise SystemExit("GATE 3 FAILED: redirected reasoning accuracies != matched-prompt artifact")
    P("[gate3] matched-arm open-cell accuracies reproduce matched_prompt_reasoning_2026-07-29.json")

    mcqa = mcq_analysis()

    # ---- build every arm -------------------------------------------------------------------------
    arms = {}
    plan = [(R_OLD, "contaminated", "unmatched"), (R_OLD, "clean_L1", "unmatched"),
            (R_OLD, "clean_L1", "matched_B"), (R_OLD, "clean_L1", "matched_A"),
            (R_NEW, "contaminated", "unmatched"), (R_NEW, "clean_L1", "unmatched"),
            (R_NEW, "clean_L1", "matched_B"), (R_NEW, "clean_L1", "matched_A"),
            (R_NEW, "contaminated", "matched_B")]
    cur_r = None
    for r, ver, arm in plan:
        if r != cur_r:
            set_R32(r); cur_r = r
        t = time.time()
        adapter = dict((n, a) for n, a, _ in MHC.ARMS)[ver]
        arms[(r, ver, arm)] = compute(adapter, arm)
        P(f"[build] R32={r} verifier={ver:12s} reasoning={arm:10s}  {time.time()-t:5.1f}s")

    # ---- GATE 1 / GATE 2 -------------------------------------------------------------------------
    diffs1, n1 = MHC.validate_against_published(arms[(R_OLD, "contaminated", "unmatched")])
    if diffs1:
        for d in diffs1[:30]:
            P("   ", d)
        raise SystemExit("GATE 1 FAILED: contaminated+unmatched at R32=4.57 does not reproduce the "
                         "published macro artifact")
    P(f"[gate1] reproduces macro_average_headline_2026-07-30.json ({n1} fields)")

    pubc = json.load(_REAL_OPEN(PUB_CLEAN))
    c = arms[(R_OLD, "clean_L1", "unmatched")]
    diffs2 = []
    for lab, dd in pubc["accuracy_levels"]["clean_L1"].items():
        for s, v in dd.items():
            if c["acc_levels"][lab][s] != v:
                diffs2.append(dict(path=f"acc.{lab}.{s}", recomputed=c["acc_levels"][lab][s], published=v))
    for m in METHODS:
        for b in MHC.BASELINES:
            for ax in ("all8", "mcq_only", "open_only"):
                for lab in ("sample_weighted", "macro_cells", "macro_benchmarks_cellavg"):
                    g = c["deltas"][m][b][ax][lab]; e = pubc["deltas"]["clean_L1"][m][b][ax][lab]
                    for f in ("delta", "lo", "hi", "verdict"):
                        if g[f] != e[f]:
                            diffs2.append(dict(path=f"delta.{m}.{b}.{ax}.{lab}.{f}",
                                               recomputed=g[f], published=e[f]))
    if diffs2:
        for d in diffs2[:30]:
            P("   ", d)
        raise SystemExit("GATE 2 FAILED: clean_L1+unmatched at R32=4.57 does not reproduce column C")
    P("[gate2] reproduces macro_headline_clean_verifier_2026-07-30.json column C")

    # =============================================================================================
    # 5.  THE FOUR-COLUMN PROGRESSION  (each correction's own contribution, visible)
    # =============================================================================================
    COLS = [("col1_published_sample_weighted_contaminated_unmatched",
             (R_OLD, "contaminated", "unmatched"), "sample_weighted",
             "the published headline: pooled sample-weighted, contaminated verifier, unmatched reasoning"),
            ("col2_macro", (R_OLD, "contaminated", "unmatched"), "macro_cells",
             "+ correction 1: equal weight per reporting cell (8 cells, 1/8 each)"),
            ("col3_macro_plus_clean_L1", (R_OLD, "clean_L1", "unmatched"), "macro_cells",
             "+ correction 2: open-text verifier retrained on strictly disjoint data (L1)"),
            ("col4_macro_plus_clean_plus_matched_B", (R_OLD, "clean_L1", "matched_B"), "macro_cells",
             "+ correction 3: reasoning baseline prompt-matched (arm B). ALL THREE, old cost constants."),
            ("col4a_arm_A", (R_OLD, "clean_L1", "matched_A"), "macro_cells",
             "correction 3 with arm A instead of arm B (reported alongside)"),
            ("col5_all_three_plus_grounded_R32",
             (R_NEW, "clean_L1", "matched_B"), "macro_cells",
             "FINAL: all three corrections + the grounded FLOP constant R32=3.816 (this also moves "
             "accuracy slightly, because the Pandora open-arm controller optimises FLOPs)")]

    def col_block(key, lab):
        res = arms[key]
        r, ver, arm = key
        return dict(
            R32=r, verifier=ver, reasoning_arm=arm, weighting=lab,
            accuracy={s: res["acc_levels"][lab][s] for s in SYSTEMS},
            deltas_vs_reasoning={m: delta(res, m, "always_32b_reasoning", "all8", lab) for m in METHODS},
            deltas_vs_direct={m: delta(res, m, "always_32b_direct", "all8", lab) for m in METHODS},
            deltas_vs_oracle={m: delta(res, m, "oracle_mode_32b", "all8", lab) for m in METHODS},
            cost_honest_recost={s: cost_row(res, "honest_recost", lab, s) for s in SYSTEMS},
            cost_as_charged={s: cost_row(res, "as_charged", lab, s) for s in SYSTEMS},
            ratios_honest_recost={m: {b: ratio(res, "honest_recost", lab, m, b)
                                      for b in HEADLINE_BASELINES} for m in METHODS})

    progression = {name: dict(description=desc, **col_block(key, lab))
                   for name, key, lab, desc in COLS}

    # the headline claim, column by column
    HEAD_M, HEAD_B = "method_accuracy_max_veto", "always_32b_reasoning"
    headline_progression = []
    for name, key, lab, desc in COLS:
        res = arms[key]
        d = delta(res, HEAD_M, HEAD_B, "all8", lab)
        rt = ratio(res, "honest_recost", lab, HEAD_M, HEAD_B)
        headline_progression.append(dict(
            column=name, description=desc,
            method_acc=res["acc_levels"][lab][HEAD_M],
            reasoning_baseline_acc=res["acc_levels"][lab][HEAD_B],
            delta=d["delta"], ci95=d["ci95"], verdict=d["verdict"],
            flops_x=rt["flops_x"], lat_seq_pct=rt["lat_seq_pct"], lat_par_pct=rt["lat_par_pct"],
            energy_pct=rt["energy_pct"]))
    for i in range(1, len(headline_progression)):
        headline_progression[i]["delta_change_from_previous_column"] = round(
            headline_progression[i]["delta"] - headline_progression[i - 1]["delta"], 4)

    # every claim, all four corrections, all three pools
    all_claims = []
    for m in METHODS:
        for b in HEADLINE_BASELINES + ["always_7b"]:
            for ax, pool in (("all8", "all 8 cells"), ("mcq_only", "5 multiple-choice cells"),
                             ("open_only", "3 open-text cells")):
                row = dict(claim=f"{m} vs {b}", pool=pool, method=m, baseline=b, axis=ax)
                for name, key, lab, _ in COLS:
                    row[name] = delta(arms[key], m, b, ax, lab)
                a, z = row[COLS[0][0]], row[COLS[5][0]]
                row["net_col1_to_col5"] = round(z["delta"] - a["delta"], 4)
                row["effect_of_matched_reasoning_alone"] = round(
                    row[COLS[3][0]]["delta"] - row[COLS[2][0]]["delta"], 4)
                row["significance_change_col1_to_col5"] = (
                    f"{a['verdict']} -> {z['verdict']}" if a["verdict"] != z["verdict"]
                    else f"unchanged ({a['verdict']})")
                all_claims.append(row)

    # =============================================================================================
    # 6.  the FINAL headline, fully priced
    # =============================================================================================
    FIN = arms[(R_NEW, "clean_L1", "matched_B")]
    FIN_A = arms[(R_NEW, "clean_L1", "matched_A")]
    LAB = "macro_cells"

    def full_comparison(res, lab=LAB):
        out = {}
        for m in METHODS:
            out[m] = {}
            for b in HEADLINE_BASELINES:
                acc = delta(res, m, b, "all8", lab)
                out[m][b] = dict(
                    accuracy=acc,
                    accuracy_mcq_only=delta(res, m, b, "mcq_only", lab),
                    accuracy_open_only=delta(res, m, b, "open_only", lab),
                    flop_eq=dict(method=res["cost"]["honest_recost"][lab][m]["flops"],
                                 baseline=res["cost"]["honest_recost"][lab][b]["flops"],
                                 x=ratio(res, "honest_recost", lab, m, b)["flops_x"]),
                    latency_batch1_ms=dict(method=res["cost"]["honest_recost"][lab][m]["lat_seq_ms"],
                                           baseline=res["cost"]["honest_recost"][lab][b]["lat_seq_ms"],
                                           pct=ratio(res, "honest_recost", lab, m, b)["lat_seq_pct"]),
                    latency_batched_ms=dict(method=res["cost"]["honest_recost"][lab][m]["lat_par_ms"],
                                            baseline=res["cost"]["honest_recost"][lab][b]["lat_par_ms"],
                                            pct=ratio(res, "honest_recost", lab, m, b)["lat_par_pct"],
                                            caveat="rests on the 'N drops out of latency' batching "
                                                   "assumption that the 2026-08-03 best-of-8 "
                                                   "measurement refutes; prefer latency_batch1_ms"),
                    energy_j=dict(method=res["cost"]["honest_recost"][lab][m]["energy_j"],
                                  baseline=res["cost"]["honest_recost"][lab][b]["energy_j"],
                                  pct=ratio(res, "honest_recost", lab, m, b)["energy_pct"]))
        return out

    final = dict(
        definition=("8-cell macro average, open-text verifier trained on strictly disjoint data (L1), "
                    "open-text reasoning baseline prompt-matched (arm B), FLOP-eq at the derived "
                    "R32=3.816, reasoning baseline honestly re-costed per cell from its own measured "
                    "generation length."),
        weighting=LAB, verifier="clean_L1", reasoning_arm="matched_B", R32=R_NEW,
        accuracy_levels={s: FIN["acc_levels"][LAB][s] for s in SYSTEMS},
        comparison=full_comparison(FIN),
        comparison_arm_A=full_comparison(FIN_A),
        per_cell_accuracy=FIN["acc_cell"],
        escalation=FIN["escalation"],
        leave_one_cell_out_headline=FIN["deltas"][HEAD_M][HEAD_B]["all8"]["macro_cells_leave_one_out"],
        strictest_reading_sensitivity=strictest_reading(FIN),
        sample_weighted_for_reference=dict(
            accuracy={s: FIN["acc_levels"]["sample_weighted"][s] for s in SYSTEMS},
            comparison=full_comparison(FIN, "sample_weighted")))

    # FLOP band sensitivity (accuracy also moves, because the controller optimises FLOPs)
    band = json.load(_REAL_OPEN(FLOP_DERIV))["derived_ratio"]["recommended"]["band"]
    band_rows = {}
    for r in band:
        set_R32(r)
        res = compute("ckpts/train/lora_verifier_disjoint", "matched_B")
        band_rows[f"R32_{r}"] = dict(
            headline_delta=delta(res, HEAD_M, HEAD_B, "all8", LAB),
            flops_x=ratio(res, "honest_recost", LAB, HEAD_M, HEAD_B)["flops_x"],
            lat_seq_pct=ratio(res, "honest_recost", LAB, HEAD_M, HEAD_B)["lat_seq_pct"],
            energy_pct=ratio(res, "honest_recost", LAB, HEAD_M, HEAD_B)["energy_pct"])
    set_R32(R_NEW)

    # =============================================================================================
    # 7.  cost-constant provenance + the best-of-8 re-pricing
    # =============================================================================================
    cost_constants = dict(
        flop_ratio=dict(
            used=R_NEW, replaced=R_OLD, band=band, status="DERIVED",
            basis="exact safetensors parameter counts (8,292,166,656 / 33,452,718,336) + measured "
                  "prompt geometry (326.68 tokens, 280.48 image); the band spans the three operating "
                  "points actually run (cap320 open-text 3.816, cap320 MCQ 3.859, fullres MCQ 3.734)",
            source="results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json",
            why_4p57_rejected="32.0/7.0 name-plate, ~16.5% too high; it is approximately the "
                              "decode-only ratio (4.524) applied to a prefill-dominated workload",
            effect_on_accuracy_tested=("The Pandora open-arm controller selects lambda by FLOPs, so "
                                       "R32 can in principle enter the METHOD's accuracy. It was run "
                                       "at both values: every system's macro accuracy is IDENTICAL to "
                                       "4 decimals, so the constant is a pure cost correction here "
                                       "(see four_column_progression col4 vs col5).")),
        best_of_8=dict(
            latency_ms_measured=BO8_MS_MEAS, energy_j_measured=BO8_J_MEAS, status="MEASURED",
            replaced=dict(latency_ms=BO8_MS_OLD, latency_status="ASSERTED (GEN7+VER7, 'N drops out')",
                          energy_j=round(BO8_J_OLD, 1), energy_status="MODELLED (8 x (GEN7+VER7))"),
            basis="Lingshu-7B, HF batch-1, cap320, real vqa_rad images, NVML, n=45 over 2 replicates "
                  "(replicate means 1325.7 / 1289.0 ms); harness validated (single greedy gen 350.0 ms "
                  "vs canonical GEN7 347.1 ms, +0.8%)",
            where_it_applies="the FIXED best-of-8 open arm only. The macro pipeline's open cells run "
                             "Pandora ADAPTIVE-N, whose draws are inherently sequential "
                             "(draw -> verify -> draw), so its batch-1 latency (lat_seq) and its "
                             "energy already use genuine batch-1 per-draw constants and are NOT "
                             "affected. What the measurement invalidates is the lat_par / lat_bat "
                             "axis, which assumes N draws cost one draw's latency."),
        latency_axis_choice=("lat_seq_ms is used as the batch-1 latency axis in the headline because it "
                             "is unaffected by the refuted batching assumption. lat_par_ms is reported "
                             "next to it, labelled."))

    bo8 = {}
    for lab in ("sample_weighted", "macro_cells"):
        bo8[lab] = dict(
            clean_L1_matched_B=bo8_recost(FIN, lab),
            clean_L1_unmatched=bo8_recost(arms[(R_NEW, "clean_L1", "unmatched")], lab),
            contaminated_unmatched=bo8_recost(arms[(R_NEW, "contaminated", "unmatched")], lab))

    batched_latency_correction = dict(
        why=("The PUBLISHED sentence '+0.0601 at -87.7% latency and -84.3% energy' takes -87.7% from "
             "the lat_par (BATCHED) axis, not from batch-1 latency: "
             "macro_headline_clean_verifier_2026-07-30.json cost_side_by_side.honest_recost."
             "lat_par_pct.method_accuracy_max_veto.always_32b_reasoning column C = -87.7 (the "
             "batch-1 / lat_seq figure for the same cell is -73.6%). That axis rests on the "
             "'N drops out of latency' assumption the 2026-08-03 best-of-8 measurement refutes."),
        final_matched_B={lab: corrected_lat_par(FIN, lab) for lab in ("macro_cells", "sample_weighted")},
        published_column_C={lab: corrected_lat_par(arms[(R_OLD, "clean_L1", "unmatched")], lab)
                            for lab in ("macro_cells", "sample_weighted")})

    # =============================================================================================
    # 8.  the reasoning baseline itself: what matching did to it
    # =============================================================================================
    import honest_recosting as HR
    reason_arm_table = {}
    for arm in THINK_ARMS:
        _THINK_ARM["name"] = arm
        _set_arm_with_think(MHC.CONTAM_ADAPTER)
        meas = HR.measure_reasoning_lengths()
        reason_arm_table[arm] = {k: dict(
            n=meas[k]["n"], gen_tok_mean=round(meas[k]["gen_reasoning"]["mean"], 2),
            judged_acc=matched_acc[k][arm]["acc"]) for k in OPEN_B}
    _THINK_ARM["name"] = "unmatched"
    _set_arm_with_think(MHC.CONTAM_ADAPTER)

    art = json.load(_REAL_OPEN(MATCHED_ART))
    arm_choice = dict(
        primary="matched_B",
        justification=[
            "matched_prompt_reasoning_2026-07-29.json sets decisive_arm = 'reason_matched_B'.",
            "prompt_design.SYS_THINK_MATCHED2: the reasoning TRIGGER sentences are kept VERBATIM and "
            "only the answer-style clause is replaced by the direct arm's -- so the variable under "
            "test (the reasoning instruction) is unchanged from the published reasoning arm.",
            "Arm A re-orders the prompt so the persona precedes the trigger, perturbing the exact "
            "trigger string these models require to emit a <think> trace.",
            "Empirically the two arms agree closely (pooled open 0.4192 arm B vs 0.4235 arm A), so the "
            "choice is not load-bearing; arm A is reported alongside in every table."],
        prompts={a: art["prompts"][{"unmatched": "reason_unmatched", "matched_A": "reason_matched_A",
                                    "matched_B": "reason_matched_B"}[a]]["text"] for a in THINK_ARMS},
        trace_fired_fraction_arm_B={
            k: art["interpretation"]["trap_1_matched_reasoning_prompts_stop_the_reasoning"]
                  ["evidence_trace_conditional"][k]["trace_fired"]["frac"] for k in OPEN_B},
        direction_of_the_bias=("Arm B's trace fires on only 30.5-71.1% of items, so it is a MIXTURE of "
                               "reasoning and direct answering and scores HIGHER than a fully-reasoning "
                               "arm would. Using it as the baseline therefore makes the baseline "
                               "STRONGER and the reported margin a LOWER bound on 'method vs a 32B "
                               "that always reasons'."))

    # =============================================================================================
    # 9.  verdicts
    # =============================================================================================
    h_pub = headline_progression[0]; h_c3 = headline_progression[2]
    h_c4 = headline_progression[3]; h_fin = headline_progression[5]
    survives = h_fin["verdict"] == "WIN"
    shrink_abs = round(h_fin["delta"] - h_c3["delta"], 4)
    shrink_pct = round(100 * (1 - h_fin["delta"] / h_c3["delta"]), 1) if h_c3["delta"] else None

    verdict = dict(
        claim_audited="accuracy-max-veto vs always-32B-with-reasoning, 8-cell macro, all 8 cells",
        published_value_being_audited=dict(delta=0.0601, ci95=[0.0498, 0.0703],
                                           source="macro_headline_clean_verifier_2026-07-30.json column C"),
        reproduced_column_C=dict(delta=h_c3["delta"], ci95=h_c3["ci95"], verdict=h_c3["verdict"]),
        with_matched_reasoning_old_constants=dict(delta=h_c4["delta"], ci95=h_c4["ci95"],
                                                  verdict=h_c4["verdict"]),
        final_all_three_plus_grounded_R32=dict(delta=h_fin["delta"], ci95=h_fin["ci95"],
                                               verdict=h_fin["verdict"]),
        survives=survives,
        remains_significant=bool(h_fin["ci95"][0] > 0),
        shrinkage_from_matching_absolute=shrink_abs,
        shrinkage_from_matching_percent=shrink_pct,
        cost_of_the_final_claim=dict(
            flop_eq_x=h_fin["flops_x"], batch1_latency_pct=h_fin["lat_seq_pct"],
            batched_latency_pct_repo_assumption=h_fin["lat_par_pct"],
            batched_latency_pct_corrected=batched_latency_correction["final_matched_B"]["macro_cells"]
                ["ratios"][HEAD_M][HEAD_B]["pct_corrected"],
            energy_pct=h_fin["energy_pct"]),
        published_87p7_was_the_batched_axis=dict(
            published=-87.7,
            same_axis_recomputed_at_column_C=batched_latency_correction["published_column_C"]
                ["macro_cells"]["ratios"][HEAD_M][HEAD_B]["pct_repo"],
            same_axis_corrected_at_column_C=batched_latency_correction["published_column_C"]
                ["macro_cells"]["ratios"][HEAD_M][HEAD_B]["pct_corrected"],
            batch1_equivalent_at_column_C=h_c3["lat_seq_pct"]),
        band_sensitivity=band_rows)

    o = dict(
        title="THE THREE-WAY HEADLINE: 8-cell macro x clean L1 verifier x MATCHED reasoning baseline, "
              "priced with the 2026-08-03 grounded cost constants.",
        date="2026-08-03", reproduce="python3 src/cascade_methods/headline_three_way.py",
        no_gpu=True, no_fabricated_numbers=True, n_bootstrap=MHC.NBOOT, seed=MHC.SEED,
        what_is_new=("The three corrections (macro re-weighting, de-contaminated verifier, matched "
                     "reasoning prompt) had never been applied together. They are here, plus the "
                     "grounded FLOP and best-of-8 constants."),
        arm_choice=arm_choice,
        validation=dict(
            gate1="contaminated + unmatched at R32=4.57 reproduces macro_average_headline_2026-07-30.json "
                  f"({n1} fields, exact)",
            gate2="clean_L1 + unmatched at R32=4.57 reproduces macro_headline_clean_verifier_2026-07-30.json "
                  "column C (accuracy levels + every delta and CI bound, exact)",
            gate3="redirected matched judge files reproduce matched_prompt_reasoning_2026-07-29.json "
                  "per-dataset accuracies (exact)"),
        cost_constants=cost_constants,
        mcq_reasoning_baseline_analysis=mcqa,
        reasoning_baseline_by_arm=reason_arm_table,
        matched_open_cell_accuracies=matched_acc,
        headline_progression=headline_progression,
        four_column_progression=progression,
        all_claims_four_columns=all_claims,
        final_headline=final,
        best_of_8_repricing=bo8,
        batched_latency_correction=batched_latency_correction,
        verdict=verdict,
        caveats=[
            "The CIs resample ITEMS WITHIN each cell (paired, common random numbers) and recompute the "
            "macro average per replicate. They capture within-dataset sampling noise ONLY, not "
            "dataset-selection noise; the leave-one-cell-out range is the honest companion.",
            "Arm B is a MIXTURE of reasoning and direct answering (trace fires 30.5-71.1%); the margin "
            "against it is a LOWER bound on the margin against a 32B that always reasons.",
            "No multiple-choice cell is re-run. 4 of the 5 Variant-B multiple-choice cells have no "
            "genuine reasoning baseline at all (see mcq_reasoning_baseline_analysis); the reasoning "
            "framing is really only tested on MedXpert + the three open cells.",
            "The PathVQA judging correction (pathvqa_judge_audit.json) was hand-derived on the "
            "UNMATCHED reasoning answers and cannot be transferred to a different arm without "
            "re-labelling, so it is NOT stacked here.",
            "as-charged cost keeps the flat 10,521.6 ms / 2,001.9 J reasoning constant, which was "
            "measured at 98.3 generated tokens on the UNMATCHED arm. It is wrong for the matched arm "
            "(86.4 / 45.8 / 101.5 tokens). Use the honest_recost convention, which prices each cell "
            "from its own measured generation length; it is what the headline uses.",
        ],
        runtime_s=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(o, _REAL_OPEN(OUT, "w"), indent=2, default=str)
    console(o)
    P(f"\nwrote {os.path.relpath(OUT, ROOT)}   ({o['runtime_s']}s)")
    return o


def console(o):
    P = print
    W = 122
    P("\n" + "=" * W)
    P("FOUR-COLUMN PROGRESSION -- accuracy-max-veto vs always-32B-WITH-REASONING (8 cells)")
    P("=" * W)
    P(f"  {'column':<52}{'method':>8}{'reason':>8}{'delta':>9}{'CI95':>20}{'verd':>6}"
      f"{'FLOPx':>7}{'lat_b1%':>9}{'nrg%':>7}")
    for r in o["headline_progression"]:
        P(f"  {r['column']:<52}{r['method_acc']:>8.4f}{r['reasoning_baseline_acc']:>8.4f}"
          f"{r['delta']:>+9.4f}  [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]{r['verdict']:>6}"
          f"{r['flops_x']:>7.2f}{r['lat_seq_pct']:>9.1f}{r['energy_pct']:>7.1f}")
    P("\n" + "=" * W)
    P("FINAL HEADLINE -- macro x clean L1 x matched-B x R32=3.816, honest re-cost")
    P("=" * W)
    f = o["final_headline"]
    P("  accuracy:  " + "  ".join(f"{s}={v:.4f}" for s, v in f["accuracy_levels"].items()))
    for m, bb in f["comparison"].items():
        for b, d in bb.items():
            a = d["accuracy"]
            P(f"  {m:<28} vs {b:<22} d={a['delta']:+.4f} [{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}] "
              f"{a['verdict']:<5} FLOP {d['flop_eq']['x']:.3f}x  lat_b1 {d['latency_batch1_ms']['pct']:+.1f}%"
              f"  nrg {d['energy_j']['pct']:+.1f}%")
    P("\n" + "=" * W)
    v = o["verdict"]
    P(f"  SURVIVES: {v['survives']}   significant: {v['remains_significant']}")
    P(f"  +0.0601 -> {v['final_all_three_plus_grounded_R32']['delta']:+.4f} "
      f"{v['final_all_three_plus_grounded_R32']['ci95']}  "
      f"(matched-prompt correction alone: {v['shrinkage_from_matching_absolute']:+.4f}, "
      f"{v['shrinkage_from_matching_percent']}%)")
    P("=" * W)


if __name__ == "__main__":
    run()
