#!/usr/bin/env python3
"""
Matched-prompt reasoning-vs-direct comparison on the reasoning-heavy MedEvalKit benchmarks.

CONTEXT
-------
The published cross-family claim "reasoning helps on reasoning-heavy benchmarks" rested on
MedEvalKit `*_reason` dumps compared against `*_direct` dumps whose prompts differed in THREE
ways at once (reasoning trigger, the word "directly", and \\boxed{} vs bare letter). The
reasoning arm was therefore confounded with an answer-FORMAT change.

This script scores the repaired experiment: the reasoning arm is unchanged (its prompt is
byte-identical to the pre-revert local edit), and the direct arm was re-run with a prompt that
differs from it by EXACTLY the reasoning clause -- see `src/labeling/medeval_matched_prompt.py`
and `runners/run_medeval_direct_matched.sh`.

  reasoning : First reason step by step about the question and each option, then put the
              final answer letter from the given choices in one "\\boxed{}".
  direct    : Put the final answer letter from the given choices in one "\\boxed{}".

Every number is read from a real dump on disk. No GPU, no fabricated values.

WHAT IT REPORTS  (per family x benchmark cell)
  * acc_reason, acc_direct_matched, acc_direct_unmatched  -- all paired on item id
  * delta_matched   = reason - direct_matched    + paired-bootstrap 95% CI + exact McNemar p
  * delta_unmatched = reason - direct_unmatched  + paired-bootstrap 95% CI  (the published number)
  * mean generated tokens per arm (confirms the reasoning arm reasoned and the direct arm did not)
  * parse_ok per arm -- the \\boxed{} format is NEW for the direct arm, so extraction must be
    shown not to have degraded.  NB MedEvalKit's MMMU parser falls back to random.choice() on a
    parse miss, so a parse failure is invisible in `parsed_pred`; we recompute candidate
    detection to expose it.

Run:  python3 src/cascade_methods/medeval_matched_direct.py
"""

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEK = os.path.join(ROOT, "MedEvalKit")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/medeval_matched_direct_2026-07-29.json")

# paired bootstrap, same convention as src/cascade_methods/paper_baselines.py (NBOOT, seed)
NBOOT = 10000
RNG = np.random.default_rng(12345)

REASON_TAIL = ('First reason step by step about the question and each option, then put the '
               'final answer letter from the given choices in one "\\boxed{}".')
DIRECT_TAIL = 'Put the final answer letter from the given choices in one "\\boxed{}".'

# family -> arm -> eval_results dir.  `direct_unmatched` is the PUBLISHED direct arm
# (upstream "...directly." prompt); `direct_matched` is the new re-run.
FAMILIES = {
    "Lingshu-32B": {
        "reason": "eval_results_lingshu32b_reason",
        "direct_unmatched": "eval_results_lingshu32b_full",
        "direct_matched": "eval_results_lingshu32b_direct_matched",
    },
    "MedVLThinker-32B": {
        "reason": "eval_results_mvt32b_reason",
        "direct_unmatched": "eval_results_mvt32b",
        "direct_matched": "eval_results_mvt32b_direct_matched",
    },
    "InternVL3-38B": {
        "reason": "eval_results_iv3_38b_reason",
        "direct_unmatched": "eval_results_iv3_38b",
        "direct_matched": "eval_results_iv3_38b_direct_matched",
    },
}

ARMS = ["reason", "direct_matched", "direct_unmatched"]


# ------------------------------------------------------------------ stats

def paired_ci(a, b):
    """95% CI of mean(a-b) by paired bootstrap over items (a,b aligned 0/1 arrays)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n == 0:
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"), sig=False, n=0)
    boots = np.empty(NBOOT)
    for s in range(0, NBOOT, 1000):
        m = min(1000, NBOOT - s)
        boots[s:s + m] = d[RNG.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return dict(delta=round(float(d.mean()), 4), lo=round(lo, 4), hi=round(hi, 4),
                sig=bool(lo > 0 or hi < 0), n=n)


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on discordant pairs. a,b are 0/1 arrays (a=reason, b=direct)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    b01 = int(np.sum((a == 1) & (b == 0)))   # reason right, direct wrong
    c10 = int(np.sum((a == 0) & (b == 1)))   # reason wrong, direct right
    n = b01 + c10
    if n == 0:
        return dict(reason_only=b01, direct_only=c10, p=1.0)
    k = min(b01, c10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return dict(reason_only=b01, direct_only=c10, p=round(min(1.0, p), 6))


# ------------------------------------------------------------------ parsing diagnostics

_BOXED = re.compile(r"\\boxed\s*\{")


def boxed_content(text):
    """Last \\boxed{...} content with balanced braces; None if absent/unbalanced."""
    if not text:
        return None
    out = None
    for m in _BOXED.finditer(text):
        i = m.end()
        depth, buf = 1, []
        while i < len(text) and depth:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            i += 1
        if depth == 0:
            out = "".join(buf)
    return out


def letter_of(s, letters):
    """First standalone choice letter in s, else None."""
    if s is None:
        return None
    t = s.strip().strip(",.!?;:'()").upper()
    if t in letters:
        return t
    m = re.search(r"\b([" + "".join(letters) + r"])\b", t)
    return m.group(1) if m else None


def mmmu_parse_ok(response, all_choices, index2ans):
    """
    Reproduce MedEvalKit's MMMU candidate detection (utils/MMMU/eval_utils.py
    parse_multi_choice_response) WITHOUT its random.choice() fallback, so a genuine parse
    miss is visible. Returns True iff at least one candidate was found.
    NB `response` in the dump is already post-\\boxed-extraction (done in eval_val.run_model).
    """
    r = "" if response is None else response
    if "\\boxed" in r:
        r = boxed_content(r) or r
    r = r.strip()
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        r = r.strip(ch)
    r = " " + r + " "
    for c in all_choices:
        if f"({c})" in r:
            return True
    for c in all_choices:
        if f" {c} " in r:
            return True
    if len(r.split()) > 5:
        for _, ans in (index2ans or {}).items():
            if ans and ans.lower() in r.lower():
                return True
    return False


# ------------------------------------------------------------------ loaders

def _one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits


def load_mmmu(dirname):
    """id -> record, from <dir>/*/MMMU-Medical-val/<subject>/parsed_output.json"""
    recs = {}
    files = _one(os.path.join(MEK, dirname, "*", "MMMU-Medical-val", "*", "parsed_output.json"))
    for f in files:
        for s in json.load(open(f)):
            recs[s["id"]] = dict(
                ok=1 if s.get("judge") == "Correct" else 0,
                gen_toks=s.get("gen_toks"),
                question_type=s.get("question_type"),
                response=s.get("response"),
                parse_ok=mmmu_parse_ok(s.get("response"), s.get("all_choices") or [],
                                       s.get("index2ans") or {})
                if s.get("question_type") == "multiple-choice" else None,
                subject=os.path.basename(os.path.dirname(f)),
            )
    return recs, files


def load_medxpert(dirname):
    """id -> record, from <dir>/*/MedXpertQA-MM/results.json"""
    recs = {}
    files = _one(os.path.join(MEK, dirname, "*", "MedXpertQA-MM", "results.json"))
    for f in files:
        for s in json.load(open(f)):
            letters = [c.split(".")[0].strip().upper() for c in s.get("choices", [])]
            letters = [c for c in letters if len(c) == 1]
            bc = boxed_content(s.get("response") or "")
            recs[s["id"]] = dict(
                ok=1 if s.get("correct") else 0,
                gen_toks=s.get("gen_toks"),
                question_type=s.get("question_type"),
                medical_task=s.get("medical_task"),
                response=s.get("response"),
                boxed_present=bool(bc is not None),
                parse_ok=bool(letter_of(bc, letters) is not None) if bc is not None
                else bool(letter_of(s.get("response"), letters) is not None),
            )
    return recs, files


# ------------------------------------------------------------------ cells

def cells_for(bench, recs_by_arm):
    """(cell_name, filter_fn) list for a benchmark."""
    if bench == "MMMU":
        return [("MMMU", lambda r: True),
                ("MMMU-MCQonly", lambda r: r["question_type"] == "multiple-choice")]
    return [("MedXpert-ALL", lambda r: True),
            ("MedXpert-Reasoning", lambda r: r["question_type"] == "Reasoning"),
            ("MedXpert-Understanding", lambda r: r["question_type"] == "Understanding")]


REASONED_TOK_THRESHOLD = 50   # a bare letter (+\boxed wrapper) is <=10 tokens; >50 means it reasoned


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def median_or_none(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 1) if vals else None


def reasoned_frac(vals):
    """Fraction of items whose generation is long enough to BE reasoning.
    This is the check that an arm labelled 'direct' actually answered directly."""
    vals = [v for v in vals if v is not None]
    return round(float(np.mean([v > REASONED_TOK_THRESHOLD for v in vals])), 4) if vals else None


def frac_or_none(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def build():
    out = {
        "_meta": {
            "what": "Matched-prompt reasoning-vs-direct on MedEvalKit reasoning-heavy benchmarks "
                    "(MMMU-Medical-val, MedXpertQA-MM) for the 3 families with *_reason dumps.",
            "date": "2026-07-29",
            "reasoning_prompt_tail": REASON_TAIL,
            "direct_matched_prompt_tail": DIRECT_TAIL,
            "arms_differ_by": "exactly the leading reasoning clause "
                              "('First reason step by step about the question and each option, then ')",
            "patch": "src/labeling/medeval_matched_prompt.py (env-gated MEDEVAL_MATCHED_PROMPT=1; "
                     "monkeypatches utils.question_formats.get_multiple_choice_prompt and "
                     "utils.MMMU.data_utils.construct_prompt; MedEvalKit left byte-identical to upstream)",
            "runner": "runners/run_medeval_direct_matched.sh",
            "direct_unmatched_note": "the PUBLISHED direct arm, upstream prompt "
                                     "\"Answer with the option's letter from the given choices directly.\" "
                                     "-- differs from the reasoning arm in 3 ways (trigger, 'directly', bare vs \\boxed)",
            "settings_matched_to_reason_arm": {
                "seed": 42, "tensor_parallel_size": 2, "use_vllm": True, "max_new_tokens": 2048,
                "max_image_num": 6, "temperature": 0, "top_p": 0.0001, "repetition_penalty": 1,
                "datasets_path": "hf", "MAX_MODEL_LEN(IV3 only)": 16384,
                "source": "runners/run_full_matrix_medeval.sh (the runner that produced *_reason)",
            },
            "residual_caveat": "Only the MULTIPLE-CHOICE prompt branch is matched. MMMU-Medical-val has "
                               "5/150 'open' items that keep upstream's open-question strings and remain "
                               "format-unmatched; the MMMU-MCQonly cell (n=145) is fully matched.",
            "stats": {"paired_bootstrap_NBOOT": NBOOT, "rng_seed": 12345,
                      "mcnemar": "two-sided exact on discordant pairs"},
            "known_minor_unmatched_axis": "EVAL_BATCH_SIZE=250 for the direct-matched run vs the "
                                          "default 2000 for the *_reason run (OOM safety on TP=2). "
                                          "Affects MedXpert only (MMMU batches at <=30 per subject "
                                          "regardless); greedy temperature-0 decoding, so at most "
                                          "rare batch-composition tie-breaks. Orders of magnitude "
                                          "smaller than the prompt confound being repaired.",
            "no_gpu_this_script": True, "no_fabricated_numbers": True,
        },
        "dumps": {}, "pairing": {}, "cells": [], "missing": [], "verdict": {},
        "mmmu_open_item_audit": {
            "why": "The 5/150 MMMU 'open' items keep upstream's (still format-unmatched) strings. "
                   "If every arm scores them identically, the residual mismatch cannot affect any "
                   "delta and MMMU == MMMU-MCQonly in the numerator.",
            "per_family_arm_correct_out_of_5": {},
        },
    }

    loaded = {}
    for fam, arms in FAMILIES.items():
        loaded[fam] = {}
        for arm, d in arms.items():
            mm, mmf = load_mmmu(d)
            mx, mxf = load_medxpert(d)
            loaded[fam][arm] = {"MMMU": mm, "MedXpert": mx}
            out["dumps"][f"{fam}|{arm}"] = {
                "dir": d, "n_MMMU": len(mm), "n_MedXpert": len(mx),
                "MMMU_files": len(mmf), "MedXpert_files": len(mxf),
            }
            for bench, recs in (("MMMU", mm), ("MedXpert", mx)):
                if not recs:
                    out["missing"].append({"family": fam, "arm": arm, "benchmark": bench, "dir": d})
            if mm:
                op = [v for v in mm.values() if v["question_type"] != "multiple-choice"]
                out["mmmu_open_item_audit"]["per_family_arm_correct_out_of_5"][f"{fam}|{arm}"] = \
                    f"{sum(v['ok'] for v in op)}/{len(op)}"

    for fam in FAMILIES:
        for bench in ("MMMU", "MedXpert"):
            per = {a: loaded[fam][a][bench] for a in ARMS}
            have = [a for a in ARMS if per[a]]
            if "reason" not in have or "direct_matched" not in have:
                continue
            # pair on the id set of the REASON dump, intersected with every available arm
            ids = set(per["reason"].keys())
            for a in have:
                ids &= set(per[a].keys())
            ids = sorted(ids)
            out["pairing"][f"{fam}|{bench}"] = {
                "n_paired": len(ids),
                "n_per_arm": {a: len(per[a]) for a in have},
                "arms_available": have,
                "ids_identical_across_arms": all(set(per[a].keys()) == set(per["reason"].keys())
                                                 for a in have),
            }

            for cname, filt in cells_for(bench, per):
                sub = [i for i in ids if filt(per["reason"][i])]
                if not sub:
                    continue
                vec = {a: np.array([per[a][i]["ok"] for i in sub]) for a in have}
                cell = {
                    "family": fam, "benchmark": bench, "cell": cname, "n": len(sub),
                    "acc": {a: round(float(vec[a].mean()), 4) for a in have},
                    "mean_gen_toks": {a: mean_or_none([per[a][i]["gen_toks"] for i in sub])
                                      for a in have},
                    "median_gen_toks": {a: median_or_none([per[a][i]["gen_toks"] for i in sub])
                                        for a in have},
                    # fraction of items that actually reasoned (gen_toks > 50)
                    "reasoned_frac": {a: reasoned_frac([per[a][i]["gen_toks"] for i in sub])
                                      for a in have},
                    "parse_ok": {a: frac_or_none([per[a][i].get("parse_ok") for i in sub])
                                 for a in have},
                }
                if bench == "MedXpert":
                    cell["boxed_present"] = {
                        a: frac_or_none([per[a][i].get("boxed_present") for i in sub]) for a in have}

                cell["delta_matched"] = paired_ci(vec["reason"], vec["direct_matched"])
                cell["mcnemar_matched"] = mcnemar_exact(vec["reason"], vec["direct_matched"])
                if "direct_unmatched" in have:
                    cell["delta_unmatched"] = paired_ci(vec["reason"], vec["direct_unmatched"])
                    cell["mcnemar_unmatched"] = mcnemar_exact(vec["reason"], vec["direct_unmatched"])
                    cell["delta_shift_from_matching"] = round(
                        cell["delta_matched"]["delta"] - cell["delta_unmatched"]["delta"], 4)
                    cell["format_effect_direct_matched_minus_unmatched"] = round(
                        float(vec["direct_matched"].mean() - vec["direct_unmatched"].mean()), 4)
                # ---- is the "direct" arm actually direct? -------------------------------
                rf = cell["reasoned_frac"]
                cell["reason_arm_reasoned"] = (rf.get("reason") or 0) > 0.5
                cell["direct_matched_answered_directly"] = (rf.get("direct_matched") or 0) < 0.1
                cell["direct_unmatched_answered_directly"] = (
                    (rf.get("direct_unmatched") or 0) < 0.1 if "direct_unmatched" in rf else None)
                if not cell["direct_matched_answered_directly"]:
                    cell["arm_validity"] = (
                        "CONTAMINATED: the matched DIRECT arm reasoned on "
                        f"{100 * (rf.get('direct_matched') or 0):.1f}% of items -- the \\boxed{{}} "
                        "instruction alone induces reasoning in this model. delta_matched therefore "
                        "measures the MARGINAL value of the explicit reasoning trigger ON TOP of "
                        "format-induced reasoning, NOT reasoning-vs-no-reasoning.")
                else:
                    cell["arm_validity"] = ("clean: reasoning arm reasoned, matched direct arm "
                                            "answered directly")
                # 3-rung ladder of reasoning amount, when a truly-direct arm exists
                if "direct_unmatched" in have and cell["direct_unmatched_answered_directly"]:
                    cell["reasoning_ladder"] = {
                        "rung1_truly_direct(bare letter)": {
                            "acc": cell["acc"]["direct_unmatched"],
                            "mean_gen_toks": cell["mean_gen_toks"]["direct_unmatched"]},
                        "rung2_boxed_no_trigger": {
                            "acc": cell["acc"]["direct_matched"],
                            "mean_gen_toks": cell["mean_gen_toks"]["direct_matched"]},
                        "rung3_trigger_plus_boxed": {
                            "acc": cell["acc"]["reason"],
                            "mean_gen_toks": cell["mean_gen_toks"]["reason"]},
                    }
                dm = cell["delta_matched"]
                base = ("reasoning HELPS (CI-significant)" if dm["sig"] and dm["delta"] > 0
                        else "reasoning HURTS (CI-significant)" if dm["sig"] and dm["delta"] < 0
                        else "no significant effect")
                if not cell["direct_matched_answered_directly"]:
                    base = ("explicit-trigger MARGINAL effect: " + base
                            + " (direct arm already reasoned; not a reasoning-vs-none test)")
                cell["verdict"] = base
                out["cells"].append(cell)
    return out, loaded


def summarize(out):
    """Headline verdict over the primary (non-redundant) cells."""
    primary = [c for c in out["cells"] if c["cell"] in ("MMMU-MCQonly", "MedXpert-Reasoning",
                                                        "MedXpert-Understanding")]
    sig_pos = [c for c in primary if c["delta_matched"]["sig"] and c["delta_matched"]["delta"] > 0]
    sig_neg = [c for c in primary if c["delta_matched"]["sig"] and c["delta_matched"]["delta"] < 0]
    pt_pos = [c for c in primary if c["delta_matched"]["delta"] > 0]
    by_fam = defaultdict(list)
    for c in primary:
        by_fam[c["family"]].append(c)
    fam_verdict = {}
    for fam, cs in by_fam.items():
        sp = [c for c in cs if c["delta_matched"]["sig"] and c["delta_matched"]["delta"] > 0]
        sn = [c for c in cs if c["delta_matched"]["sig"] and c["delta_matched"]["delta"] < 0]
        fam_verdict[fam] = {
            "n_cells": len(cs),
            "sig_positive": len(sp), "sig_negative": len(sn),
            "point_positive": sum(1 for c in cs if c["delta_matched"]["delta"] > 0),
            "cells": {c["cell"]: c["delta_matched"]["delta"] for c in cs},
            "verdict": ("reasoning helps" if sp and not sn else
                        "reasoning hurts" if sn and not sp else
                        "mixed" if sp and sn else "no significant effect"),
        }
    out["verdict"] = {
        "primary_cells": len(primary),
        "sig_positive": len(sig_pos), "sig_negative": len(sig_neg),
        "point_positive": len(pt_pos),
        "per_family": fam_verdict,
        "mean_delta_shift_from_matching": round(float(np.mean(
            [c["delta_shift_from_matching"] for c in primary
             if "delta_shift_from_matching" in c])), 4) if primary else None,
    }
    return out


def console(out):
    m = out["_meta"]
    print("=" * 118)
    print("MATCHED-PROMPT reasoning-vs-direct  --  MedEvalKit reasoning-heavy benchmarks")
    print("=" * 118)
    print(f"reasoning tail : {m['reasoning_prompt_tail']}")
    print(f"direct   tail  : {m['direct_matched_prompt_tail']}")
    print()
    if out["missing"]:
        print("MISSING DUMPS (cells skipped):")
        for x in out["missing"]:
            print(f"  {x['family']:18s} {x['arm']:16s} {x['benchmark']:10s} {x['dir']}")
        print()
    hdr = (f"{'family':18s} {'cell':22s} {'n':>5s} {'reas':>6s} {'dirM':>6s} {'dirU':>6s} "
           f"{'d_match':>8s} {'95% CI':>18s} {'sig':>4s} {'d_unmat':>8s} {'shift':>7s} "
           f"{'tokR':>7s} {'tokM':>7s} {'rfM':>5s} {'pokM':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for c in out["cells"]:
        a = c["acc"]
        dm, du = c["delta_matched"], c.get("delta_unmatched", {})
        print(f"{c['family']:18s} {c['cell']:22s} {c['n']:5d} "
              f"{a.get('reason', float('nan')):6.3f} {a.get('direct_matched', float('nan')):6.3f} "
              f"{a.get('direct_unmatched', float('nan')):6.3f} "
              f"{dm['delta']:+8.4f} [{dm['lo']:+.4f},{dm['hi']:+.4f}] {'Y' if dm['sig'] else 'n':>4s} "
              f"{du.get('delta', float('nan')):+8.4f} {c.get('delta_shift_from_matching', float('nan')):+7.4f} "
              f"{c['mean_gen_toks'].get('reason') or float('nan'):7.1f} "
              f"{c['mean_gen_toks'].get('direct_matched') or float('nan'):7.1f} "
              f"{(c['reasoned_frac'].get('direct_matched') if c['reasoned_frac'].get('direct_matched') is not None else float('nan')):5.2f} "
              f"{(c['parse_ok'].get('direct_matched') if c['parse_ok'].get('direct_matched') is not None else float('nan')):6.3f}")
    print()
    v = out["verdict"]
    print("VERDICT (primary cells = MMMU-MCQonly, MedXpert-Reasoning, MedXpert-Understanding)")
    print(f"  {v['sig_positive']}/{v['primary_cells']} CI-significantly POSITIVE, "
          f"{v['sig_negative']}/{v['primary_cells']} CI-significantly NEGATIVE, "
          f"{v['point_positive']}/{v['primary_cells']} point-positive")
    sh = v["mean_delta_shift_from_matching"]
    print(f"  mean shift in delta caused by matching the prompt: "
          f"{('%+.4f' % sh) if sh is not None else 'n/a'}")
    for fam, fv in v["per_family"].items():
        print(f"  {fam:18s} {fv['verdict']:24s} sig+={fv['sig_positive']} sig-={fv['sig_negative']} "
              f"pt+={fv['point_positive']}/{fv['n_cells']}  {fv['cells']}")
    bad = sorted({c["family"] for c in out["cells"] if not c["direct_matched_answered_directly"]})
    if bad:
        print("\n  !! DIRECT-ARM CONTAMINATION (\\boxed{} alone induces reasoning): "
              + ", ".join(bad))
        print("     For these families delta_matched is the MARGINAL value of the explicit trigger,")
        print("     not a reasoning-vs-no-reasoning test.  rfM column = frac of direct-arm items >50 toks.")


if __name__ == "__main__":
    out, _ = build()
    out = summarize(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    console(out)
    print(f"\nwrote {OUT}")
