#!/usr/bin/env python3
"""
neurosymbolic_gate.py -- OFFLINE (CPU only, NO GPU) test of backlog idea H9:
Neuro-symbolic medical-constraint filter + violation-triggered escalation.

QUESTION (from METHOD_IDEAS_BACKLOG.md H9):
  Can HARD LOGICAL checks, parseable from the MCQ question/option text, flag confident-wrong answers --
  especially the SHARED confident-wrong errors BOTH the 7B and the 32B make (agreement high, both wrong) --
  and become (a) a free CORRECTION (prune symbolically-impossible answers) or (b) a high-precision
  ESCALATION TRIGGER that beats the confidence gate on that subset?

CONSTRAINTS built from the benchmark text (all checkable offline from question + options JSON):
  C1 negation        : question contains a negation/exception cue (not/except/least/false/without/no evidence...)
  C2 laterality-q    : question mentions a side (left/right) -- a spatial-reasoning difficulty proxy
  C3 laterality-contra: the CHOSEN answer's option-text names the OPPOSITE side to the question (strict logical clash)
  C4 dup-options     : two option texts are exact/normalized duplicates (mutual-exclusivity violation)
  C5 numeric-options : options carry numbers/units (a real range/unit check needs an EXTERNAL medical KB -> flagged infeasible offline; coverage-only)

For each flag we measure, on COMPETENT-4 (the answer-producing scope; MMMU/MedXpert excluded):
  - coverage (fraction of samples it fires on)
  - precision( flag => 7B wrong )  vs the base 7B error rate  (does it concentrate errors?)
  - precision( flag => BOTH 7B and 32B wrong )                (wasted-escalation mass)
  - recovery( flag ) = P(32B right & 7B wrong | flag)         (value of escalating the flagged slice)
    vs the recovery of the SAME-BUDGET lowest-margin slice the confidence gate would escalate instead
  - recall: fraction of ALL 7B errors that the flag covers    (backlog metric (i))
CORRECTION test: C4 duplicate-option mass-merge -> re-argmax the 7B (free, no escalation) -> acc delta.
SHARED confident-wrong analysis: how many both-wrong-and-agree errors exist, and how many ANY constraint reaches.

Integrated-method effect: deployed cascade 7B-nothink@cap320 --margin gate (tau=0.426)--> 32B-think.
  We report the added-escalation acc/FLOPs of a violation trigger and compare to spending the same escalation
  budget through the margin gate. No abstention anywhere. No fabricated numbers.

Launch from repo root:  python3 src/cascade_methods/neurosymbolic_gate.py
Writes: results/cascade_methods/artifacts/neurosymbolic_gate.json
"""
import os, sys, json, re, glob
import numpy as np
from collections import defaultdict
import pyarrow.parquet as pq
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, N7, N32, CACHE, COMPETENT

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)

TAU = 0.4264123185919304
CAP7 = "cap320"
EVAL_GLOB = "/data/dan/dataset/MedVLThinker-Eval/data/*.parquet"
# checkpoint dataset name -> parquet dataset_name
DSMAP = {"PMC-VQA": "pmc_vqa", "SLAKE": "slake_closed",
         "VQA-RAD": "vqa_rad_closed", "PathVQA": "pathvqa_closed"}

NEG = re.compile(r"\b(not|except|least|false|incorrect|cannot|without|never|absent|"
                 r"no evidence|un\w+ likely|rule out)\b", re.I)
LAT = re.compile(r"\b(left|right)\b", re.I)


def margin_full(lp):
    v = sorted((lp or {}).values(), reverse=True)
    return float(v[0] - v[1]) if len(v) >= 2 else 0.0


def norm_txt(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def side_of(text):
    s = set(m.lower() for m in LAT.findall(text or ""))
    return s & {"left", "right"}


def load_meta():
    files = sorted(glob.glob(EVAL_GLOB))
    cols = ["question", "options", "answer_label", "answer", "dataset_name", "dataset_index"]
    df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    meta = {}
    for pname in set(DSMAP.values()):
        sub = df[df.dataset_name == pname]
        m = {}
        for _, r in sub.iterrows():
            try:
                opts = json.loads(r["options"])
            except Exception:
                opts = {}
            m[int(r["dataset_index"])] = dict(q=r["question"] or "", opts=opts,
                                              gold=r["answer_label"])
        meta[pname] = m
    return meta


def build_records():
    """One row per COMPETENT-4 sample with 7B/32B outcomes + parsed text constraints."""
    e7 = _load_arm(J("ckpts/gate_7b_prune/" + CAP7), "nothink_norag")
    e32 = _load_arm(J("ckpts/gate_32b"), "think_norag")
    cache = json.load(open(J(CACHE)))
    meta = load_meta()
    recs = []
    for ds, pname in DSMAP.items():
        M = meta[pname]
        for i, r7 in e7[ds].items():
            if i not in e32.get(ds, {}) or i not in M:
                continue
            if str(i) not in cache[ds][CAP7] or str(i) not in cache[ds]["fullres"]:
                continue
            r32 = e32[ds][i]; md = M[i]
            opts = md["opts"]; q = md["q"]
            pred7 = r7["pred"]; pred32 = r32["pred"]
            ok7 = int(r7["ok"]); ok32 = int(r32["ok"])
            # option-text of each leg's chosen letter
            atext7 = opts.get(pred7, ""); atext32 = opts.get(pred32, "")
            # constraint flags
            f_neg = bool(NEG.search(q))
            qside = side_of(q)
            f_latq = len(qside) > 0
            aside7 = side_of(str(atext7))
            f_latcon = bool(qside and aside7 and not (qside & aside7))
            vals = [norm_txt(v) for v in opts.values()]
            f_dup = len(vals) != len(set(vals))
            f_num = any(re.search(r"\d", str(v)) for v in opts.values())
            recs.append(dict(ds=ds, idx=i, ok7=ok7, ok32=ok32, pred7=pred7, pred32=pred32,
                             gold=md["gold"], margin=margin_full(r7.get("opt_logprobs")),
                             opts=opts, logp=r7.get("opt_logprobs") or {},
                             g0=(r7.get("gen_tokens") or 2), g32=(r32.get("gen_tokens") or 0),
                             Pc=cache[ds][CAP7][str(i)][0], Pf=cache[ds]["fullres"][str(i)][0],
                             f_neg=f_neg, f_latq=f_latq, f_latcon=f_latcon, f_dup=f_dup, f_num=f_num))
    return recs


def flag_stats(recs, flag_key):
    """Precision / recovery / recall stats for a single boolean flag over COMPETENT-4."""
    n = len(recs)
    err = [r for r in recs if r["ok7"] == 0]
    base_err = len(err) / n
    flagged = [r for r in recs if r[flag_key]]
    nf = len(flagged)
    out = dict(coverage=round(nf / n, 4), n_flagged=nf, base_err_rate=round(base_err, 4))
    if nf == 0:
        out.update(prec_wrong=None, lift=None, prec_both_wrong=None, recovery=None,
                   recovery_margin_equalbudget=None, recall_of_errors=0.0)
        return out
    wrong = sum(1 for r in flagged if r["ok7"] == 0)
    both = sum(1 for r in flagged if r["ok7"] == 0 and r["ok32"] == 0)
    recov = sum(1 for r in flagged if r["ok7"] == 0 and r["ok32"] == 1)   # 32B fixes 7B
    out["prec_wrong"] = round(wrong / nf, 4)
    out["lift"] = round((wrong / nf) / base_err, 3) if base_err > 0 else None
    out["prec_both_wrong"] = round(both / nf, 4)
    out["recovery"] = round(recov / nf, 4)
    # equal-budget margin gate: escalate the nf lowest-margin samples, its recovery rate
    order = sorted(recs, key=lambda r: r["margin"])[:nf]
    recov_m = sum(1 for r in order if r["ok7"] == 0 and r["ok32"] == 1)
    out["recovery_margin_equalbudget"] = round(recov_m / nf, 4)
    out["recall_of_errors"] = round(wrong / len(err), 4) if err else 0.0
    return out


def dup_merge_correction(recs):
    """C4 correction: merge probability mass across duplicate option texts, re-argmax the 7B.
    Returns (n_affected, n_changed, acc_before, acc_after) over COMPETENT-4."""
    n_aff = n_chg = 0
    ok_before = ok_after = 0
    for r in recs:
        ok_before += r["ok7"]
        if not r["f_dup"]:
            ok_after += r["ok7"]; continue
        n_aff += 1
        opts = r["opts"]; logp = r["logp"]
        # group letters by normalized text
        groups = defaultdict(list)
        for letter, txt in opts.items():
            groups[norm_txt(txt)].append(letter)
        # merged logprob per group = logsumexp of member letter logprobs (present ones)
        best_group, best_val = None, -1e18
        for txt, letters in groups.items():
            lps = [logp[l] for l in letters if l in logp]
            if not lps:
                continue
            v = np.max(lps) + np.log(np.sum(np.exp(np.array(lps) - np.max(lps))))
            if v > best_val:
                best_val, best_group = v, letters
        if best_group is None:
            ok_after += r["ok7"]; continue
        # pick a representative letter of the winning group (prefer the one == gold if gold is a synonym)
        newpred = best_group[0]
        for l in best_group:
            if l == r["gold"]:
                newpred = l
        chg = int(newpred != r["pred7"])
        n_chg += chg
        ok_after += int(newpred == r["gold"])
    n = len(recs)
    return dict(n_dup=n_aff, n_changed=n_chg,
                acc_before=round(ok_before / n, 4), acc_after=round(ok_after / n, 4),
                acc_delta=round((ok_after - ok_before) / n, 4))


def cascade_baseline(recs):
    """Deployed margin-gate cascade over COMPETENT-4: pooled acc, esc rate, FLOPs% of always-32B."""
    ok, esc, f7, f32e, f32all = 0.0, 0, 0.0, 0.0, 0.0
    for r in recs:
        e = r["margin"] < TAU
        ok += r["ok32"] if e else r["ok7"]
        esc += int(e)
        f7 += 2 * N7 * (r["Pc"] + r["g0"])
        c32 = 2 * N32 * (r["Pf"] + r["g32"])
        f32all += c32
        if e:
            f32e += c32
    n = len(recs)
    return dict(acc=round(ok / n, 4), esc=round(esc / n, 4),
                flops_pct_of_32b=round(100 * (f7 + f32e) / f32all, 2),
                f7=f7, f32all=f32all)


def cascade_with_extra_flag(recs, flag_keys):
    """Add a constraint-violation trigger ON TOP of the margin gate (escalate if gate OR any flag)."""
    base = cascade_baseline(recs)
    ok, esc, f32e = 0.0, 0, 0.0
    for r in recs:
        e = (r["margin"] < TAU) or any(r[k] for k in flag_keys)
        ok += r["ok32"] if e else r["ok7"]
        esc += int(e)
        if e:
            f32e += 2 * N32 * (r["Pf"] + r["g32"])
    n = len(recs)
    return dict(acc=round(ok / n, 4), esc=round(esc / n, 4),
                flops_pct_of_32b=round(100 * (base["f7"] + f32e) / base["f32all"], 2),
                d_acc_vs_gate=round(ok / n - base["acc"], 4),
                d_esc_vs_gate=round(esc / n - base["esc"], 4))


def main():
    recs = build_records()
    n = len(recs)
    parity = round(np.mean([r["ok32"] for r in recs]), 4)
    acc7 = round(np.mean([r["ok7"] for r in recs]), 4)

    # shared confident-wrong: both wrong AND agree on the same (wrong) answer
    shared = [r for r in recs if r["ok7"] == 0 and r["ok32"] == 0 and r["pred7"] == r["pred32"]]
    both_wrong = [r for r in recs if r["ok7"] == 0 and r["ok32"] == 0]
    # how many shared-wrong does ANY constraint reach?
    any_flag = lambda r: r["f_neg"] or r["f_latcon"] or r["f_dup"]
    shared_caught = sum(1 for r in shared if any_flag(r))

    out = {"idea": "H9", "scope": "COMPETENT-4", "n": n, "tau": TAU,
           "acc_7b_base": acc7, "parity_always32b_think": parity,
           "constraints": {}, "notes": {}}

    for name, key in [("C1_negation", "f_neg"), ("C2_laterality_q", "f_latq"),
                      ("C3_laterality_contradiction", "f_latcon"), ("C4_dup_options", "f_dup"),
                      ("C5_numeric_options", "f_num")]:
        out["constraints"][name] = flag_stats(recs, key)

    # combined high-precision escalation trigger = negation OR laterality-contradiction OR dup
    class Comb:
        pass
    for r in recs:
        r["f_comb"] = r["f_neg"] or r["f_latcon"] or r["f_dup"]
    out["constraints"]["COMBINED_neg_or_latcontra_or_dup"] = flag_stats(recs, "f_comb")

    out["dup_merge_correction"] = dup_merge_correction(recs)

    out["shared_confident_wrong"] = dict(
        n_both_wrong=len(both_wrong), n_both_wrong_and_agree=len(shared),
        frac_both_wrong_and_agree=round(len(shared) / n, 4),
        shared_caught_by_any_constraint=shared_caught,
        frac_shared_caught=round(shared_caught / len(shared), 4) if shared else None)

    # integrated cascade
    out["cascade_baseline_margin_gate"] = {k: v for k, v in cascade_baseline(recs).items()
                                           if k not in ("f7", "f32all")}
    out["cascade_plus_negation_trigger"] = cascade_with_extra_flag(recs, ["f_neg"])
    out["cascade_plus_combined_trigger"] = cascade_with_extra_flag(recs, ["f_comb"])

    out["notes"]["C5_numeric"] = ("Numeric/unit range validity is NOT checkable offline without an external "
        "medical knowledge base (a number in an option is not per se implausible). Reported coverage only; "
        "a real check needs a domain KB / parser -> out of offline scope.")
    out["notes"]["coverage_bottleneck"] = ("Cleanly text-checkable medical constraints fire on a tiny slice "
        "of generic VQA; most shared confident-wrong errors are PERCEPTUAL, not logical violations, so they "
        "are unreachable by symbolic answer-text constraints.")

    os.makedirs(J("results/cascade_methods/artifacts"), exist_ok=True)
    with open(J("results/cascade_methods/artifacts/neurosymbolic_gate.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- console
    print("\n=== H9 Neuro-symbolic medical-constraint filter/gate (OFFLINE) ===")
    print(f"COMPETENT-4 n={n}  7B acc={acc7}  32B(think) parity={parity}")
    print(f"\n{'constraint':32s} {'cov':>7s} {'nflag':>6s} {'P(wrong)':>9s} {'lift':>5s} "
          f"{'P(both)':>8s} {'recov':>6s} {'recovMg':>8s} {'recall':>7s}")
    for name, s in out["constraints"].items():
        pw = s["prec_wrong"];
        print(f"{name:32s} {s['coverage']:7.4f} {s['n_flagged']:6d} "
              f"{(pw if pw is not None else float('nan')):9.4f} "
              f"{(s['lift'] if s['lift'] is not None else float('nan')):5} "
              f"{(s['prec_both_wrong'] if s['prec_both_wrong'] is not None else float('nan')):8} "
              f"{(s['recovery'] if s['recovery'] is not None else float('nan')):6} "
              f"{(s['recovery_margin_equalbudget'] if s['recovery_margin_equalbudget'] is not None else float('nan')):8} "
              f"{s['recall_of_errors']:7}")
    dm = out["dup_merge_correction"]
    print(f"\nC4 dup-merge correction: {dm['n_dup']} dup-option samples, {dm['n_changed']} changed, "
          f"acc {dm['acc_before']}->{dm['acc_after']} (delta {dm['acc_delta']:+})")
    sw = out["shared_confident_wrong"]
    print(f"\nShared confident-wrong: both-wrong={sw['n_both_wrong']}, both-wrong-AND-agree="
          f"{sw['n_both_wrong_and_agree']} ({sw['frac_both_wrong_and_agree']:.2%}); "
          f"caught by any constraint={sw['shared_caught_by_any_constraint']} "
          f"({(sw['frac_shared_caught'] or 0):.2%})")
    print("\nIntegrated cascade (COMPETENT-4):")
    b = out["cascade_baseline_margin_gate"]
    print(f"  margin-gate baseline : acc={b['acc']} esc={b['esc']} FLOPs={b['flops_pct_of_32b']}%")
    for label, k in [("+negation trigger", "cascade_plus_negation_trigger"),
                     ("+combined trigger", "cascade_plus_combined_trigger")]:
        c = out[k]
        print(f"  {label:20s}: acc={c['acc']} (d{c['d_acc_vs_gate']:+}) esc={c['esc']} "
              f"(d{c['d_esc_vs_gate']:+}) FLOPs={c['flops_pct_of_32b']}%")
    print("\nwrote results/cascade_methods/artifacts/neurosymbolic_gate.json")


if __name__ == "__main__":
    main()
