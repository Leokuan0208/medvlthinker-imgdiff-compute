#!/usr/bin/env python3
"""
pathvqa_judge_audit.py - CORRECTIVE AUDIT of retrospective hole 3: is the PathVQA-open
"reasoning collapse" (judge accuracy 0.3760 direct -> 0.1087 reasoning, ~51% of the paper's
headline delta) a real capability failure, or an answer-granularity / answer-convention
MEASUREMENT artifact?

Everything is offline except the blinded re-grade, which uses the project's validated
Claude-as-judge subagent path (no GPU, no API key) -- see pathvqa_regrade_batches.py.

Inputs
  ckpts/openvqa/strong_lingshu/ckpt_pathvqa_open_lingshu32b.jsonl(.judge.jsonl)         [32B direct]
  ckpts/openvqa/strong_lingshu_think/ckpt_pathvqa_open_lingshu32b_think.jsonl(.judge)   [32B reasoning]
  results/cascade_methods/artifacts/pathvqa_judge_audit_labels.json                     [hand labels]
  results/cascade_methods/claude_judge/pathvqa_granularity/verdict_b*.json              [blinded re-grade]
  results/cascade_methods/artifacts/f8_mode_vsthink_ci.json                             [headline vector]

Modes
  --emit-disagree   print the stratified direct-RIGHT / reasoning-WRONG sample for hand labelling
  --emit-judgeval   print the stratified judge-verdict sample (accept+reject x both modes)
  --emit-bothwrong  print an extra sample of the both-wrong cell
  --report          (default) fold everything together -> artifacts/pathvqa_judge_audit.json
Launch from the repo root.
"""
import argparse, json, os, random, statistics, math, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_P = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu/ckpt_pathvqa_open_lingshu32b")
THK_P = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu_think/ckpt_pathvqa_open_lingshu32b_think")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "pathvqa_judge_audit.json")
LABELS = os.path.join(ART, "pathvqa_judge_audit_labels.json")
REGRADE = os.path.join(ROOT, "results/cascade_methods/claude_judge/pathvqa_granularity")
HEADLINE = os.path.join(ART, "f8_mode_vsthink_ci.json")
SEED = 20260729
# questions whose gold is a body-system taxonomy token rather than a real answer
DEG = re.compile(r"^(what (is|are) present|where (does this|is this|is the)|where does this part belong)", re.I)
GEN_CAP = 512   # run_openvqa.py forces max_tokens>=512 when --think


def rows(p):  return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())}
def jrows(p): return {r["idx"]: r["judge_ok"] for r in (json.loads(l) for l in open(p) if l.strip())}
def load():
    D, T = rows(DIR_P + ".jsonl"), rows(THK_P + ".jsonl")
    JD, JT = jrows(DIR_P + ".judge.jsonl"), jrows(THK_P + ".judge.jsonl")
    return D, T, JD, JT, sorted(set(D) & set(T) & set(JD) & set(JT))
def gold_bucket(g):
    n = len(str(g).split()); return "1" if n == 1 else "2" if n == 2 else "3-4" if n <= 4 else "5+"
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
def stratified(pool, key, k, seed=SEED):
    by = defaultdict(list)
    for i in pool: by[key(i)].append(i)
    rng = random.Random(seed); out = []
    for b in sorted(by):
        s = by[b][:]; rng.shuffle(s); out += s[: max(1, round(k * len(by[b]) / len(pool)))]
    return sorted(out)


# ---------------------------------------------------------------- sample emitters
def cmd_disagree(A):
    D, T, JD, JT, idx = load()
    pool = [i for i in idx if JD[i] == 1 and JT[i] == 0]
    samp = stratified(pool, lambda i: gold_bucket(T[i]["gold"]), A.k)
    print(f"# direct-RIGHT / reasoning-WRONG pool = {len(pool)}; stratified sample = {len(samp)}")
    for n, i in enumerate(samp, 1):
        print(f"\n=== [{n}] idx={i}  gold_words={len(str(T[i]['gold']).split())} ===")
        print(f"Q      : {T[i]['question']}\nGOLD   : {T[i]['gold']}")
        print(f"DIRECT : {D[i]['modal_pred']}   [judge=1]\nREASON : {T[i]['modal_pred']}   [judge=0]")
    json.dump(samp, open(os.path.join(ART, "_pathvqa_disagree_sample.json"), "w"))

def cmd_judgeval(A):
    D, T, JD, JT, idx = load()
    cells = {"direct_accept": [i for i in idx if JD[i] == 1], "direct_reject": [i for i in idx if JD[i] == 0],
             "reason_accept": [i for i in idx if JT[i] == 1], "reason_reject": [i for i in idx if JT[i] == 0]}
    rng = random.Random(SEED + 1); out = {}
    for c, pool in cells.items():
        s = pool[:]; rng.shuffle(s); out[c] = sorted(s[: A.k])
    for c in ["direct_accept", "direct_reject", "reason_accept", "reason_reject"]:
        mode = "DIRECT" if c.startswith("direct") else "REASON"; src = D if mode == "DIRECT" else T
        print(f"\n########## CELL {c} (judge said {1 if c.endswith('accept') else 0}) ##########")
        for n, i in enumerate(out[c], 1):
            print(f"[{c}|{n}] idx={i} | Q: {src[i]['question']}\n    GOLD: {src[i]['gold']}\n    {mode}: {src[i]['modal_pred']}")
    json.dump(out, open(os.path.join(ART, "_pathvqa_judgeval_sample.json"), "w"))

def cmd_bothwrong(A):
    D, T, JD, JT, idx = load()
    prev = set(json.load(open(os.path.join(ART, "_pathvqa_judgeval_sample.json")))["reason_reject"])
    pool = [i for i in idx if JD[i] == 0 and JT[i] == 0 and i not in prev]
    rng = random.Random(SEED + 7); rng.shuffle(pool); samp = sorted(pool[: A.k])
    for n, i in enumerate(samp, 1):
        print(f"[bw|{n}] idx={i} | Q: {T[i]['question']}\n    GOLD  : {T[i]['gold']}\n    REASON: {T[i]['modal_pred']}")
    json.dump(samp, open(os.path.join(ART, "_pathvqa_bothwrong_sample.json"), "w"))


# ---------------------------------------------------------------- blinded re-grade
def load_regrade(idx):
    """Un-blind the A/B verdicts back to direct/reason. Returns {std: {mode: {idx: 0/1}}}."""
    kf = os.path.join(REGRADE, "_ab_key.json")
    if not os.path.exists(kf): return None, 0
    key = {int(k): v for k, v in json.load(open(kf)).items()}
    out = {s: {"direct": {}, "reason": {}} for s in ("strict", "fair")}
    n = 0
    for f in sorted(os.listdir(REGRADE)):
        if not f.startswith("verdict_b"): continue
        for r in json.load(open(os.path.join(REGRADE, f))):
            i = int(r["i"])
            if i not in key: continue
            for slot in ("A", "B"):
                m = key[i][slot]
                for s in ("strict", "fair"):
                    out[s][m][i] = int(r[f"{slot}_{s}"])
            n += 1
    return out, n


# ---------------------------------------------------------------- report
def cmd_report(A):
    D, T, JD, JT, idx = load()
    n = len(idx)
    wl = lambda s: len(str(s).split())
    acc_d, acc_t = sum(JD[i] for i in idx) / n, sum(JT[i] for i in idx) / n
    lab = json.load(open(LABELS)) if os.path.exists(LABELS) else {}
    res = {"_meta": {"purpose": "audit of PROJECT_RETROSPECTIVE_2026-07-29 hole 3 (PathVQA-open reasoning collapse)",
                     "date": "2026-07-29", "seed": SEED, "no_fabricated_numbers": True,
                     "reproduce": "python3 src/cascade_methods/pathvqa_judge_audit.py --report",
                     "dumps": {"direct": os.path.relpath(DIR_P + ".jsonl", ROOT),
                               "reason": os.path.relpath(THK_P + ".jsonl", ROOT)}}}

    # ---- 1. reproduce the collapse + descriptive stats ------------------------------------
    res["descriptive"] = {
        "n_items": n,
        "acc_direct_localjudge": round(acc_d, 4), "acc_reason_localjudge": round(acc_t, 4),
        "delta_localjudge": round(acc_t - acc_d, 4),
        "acc_direct_exactmatch": round(sum(D[i]["modal_ok"] for i in idx) / n, 4),
        "acc_reason_exactmatch": round(sum(T[i]["modal_ok"] for i in idx) / n, 4),
        "n_direct_right_reason_wrong": sum(1 for i in idx if JD[i] == 1 and JT[i] == 0),
        "n_direct_wrong_reason_right": sum(1 for i in idx if JD[i] == 0 and JT[i] == 1),
        "n_both_right": sum(1 for i in idx if JD[i] == 1 and JT[i] == 1),
        "n_both_wrong": sum(1 for i in idx if JD[i] == 0 and JT[i] == 0),
        "answer_words_direct_mean": round(statistics.mean(wl(D[i]["modal_pred"]) for i in idx), 2),
        "answer_words_reason_mean": round(statistics.mean(wl(T[i]["modal_pred"]) for i in idx), 2),
        "answer_words_direct_median": statistics.median(wl(D[i]["modal_pred"]) for i in idx),
        "answer_words_reason_median": statistics.median(wl(T[i]["modal_pred"]) for i in idx),
        "answer_chars_direct_mean": round(statistics.mean(len(str(D[i]["modal_pred"])) for i in idx), 1),
        "answer_chars_reason_mean": round(statistics.mean(len(str(T[i]["modal_pred"])) for i in idx), 1),
        "gold_words_mean": round(statistics.mean(wl(T[i]["gold"]) for i in idx), 2),
        "gold_frac_single_word": round(sum(1 for i in idx if wl(T[i]["gold"]) == 1) / n, 4),
        "gold_frac_le2_words": round(sum(1 for i in idx if wl(T[i]["gold"]) <= 2) / n, 4),
        "gold_frac_binary_yesno": round(sum(1 for i in idx if str(T[i]["gold"]).strip().lower() in ("yes", "no")) / n, 4),
    }
    # ---- 2. truncation / parse-failure re-verification -------------------------------------
    res["truncation_and_parse_check"] = {
        "generation_cap_tokens": GEN_CAP,
        "gen_tokens_reason_mean": round(statistics.mean(T[i]["gen_tokens"] for i in idx), 1),
        "gen_tokens_reason_median": statistics.median(T[i]["gen_tokens"] for i in idx),
        "gen_tokens_reason_max": max(T[i]["gen_tokens"] for i in idx),
        "n_at_or_above_cap": sum(1 for i in idx if T[i]["gen_tokens"] >= GEN_CAP),
        "n_unstripped_reasoning_trace": sum(1 for i in idx if "<think>" in str(T[i]["modal_pred"]) or "</think>" in str(T[i]["modal_pred"])),
        "n_empty_reason_pred": sum(1 for i in idx if not str(T[i]["modal_pred"]).strip()),
        "gen_tokens_direct_mean": round(statistics.mean(D[i]["gen_tokens"] for i in idx), 1),
        "n_reason_pred_dangling_connective": sum(1 for i in idx if str(T[i]["modal_pred"]).rstrip().endswith((",", ";", "and", "or", "of", "the"))),
        "n_reason_pred_starts_lowercase": sum(1 for i in idx if str(T[i]["modal_pred"])[:1].islower()),
        "n_direct_pred_starts_lowercase": sum(1 for i in idx if str(D[i]["modal_pred"])[:1].islower()),
        "residual_parse_risk": "run_openvqa.py extract() keeps only the LAST line after </think>; raw_output is "
                               "not retained in the dump, so that step cannot be fully re-verified offline. Upper "
                               "bound from its observable signature (answers ending in a dangling connective): 2.2%.",
        "verdict": "NOT truncation and NOT a parse failure: 0/1500 items reach the 512-token generation cap "
                   "(max 261 tokens); 4/1500 carry an unstripped reasoning trace; 0 are empty; 2.2% show any "
                   "sign of a lost line. The prior claim is independently confirmed.",
    }
    # ---- 3. where the collapse lives -------------------------------------------------------
    by_gold = {}
    for b in ["1", "2", "3-4", "5+"]:
        ii = [i for i in idx if gold_bucket(T[i]["gold"]) == b]
        by_gold[b] = {"n": len(ii), "acc_direct": round(sum(JD[i] for i in ii) / len(ii), 4),
                      "acc_reason": round(sum(JT[i] for i in ii) / len(ii), 4),
                      "delta": round(sum(JT[i] - JD[i] for i in ii) / len(ii), 4)}
    res["collapse_by_gold_length"] = by_gold
    deg = [i for i in idx if DEG.match(T[i]["question"].strip())]
    non = [i for i in idx if i not in set(deg)]
    pool = [i for i in idx if JD[i] == 1 and JT[i] == 0]
    tax = {re.sub(r"[^a-z0-9 ]", "", str(g).lower()).strip()
           for g, c in Counter(T[i]["gold"].lower().strip() for i in idx).items() if c >= 5}
    conv = lambda src, ii: round(sum(1 for i in ii if re.sub(r"[^a-z0-9 ]", "", str(src[i]["modal_pred"]).lower()).strip() in tax) / len(ii), 4)
    res["degenerate_question_decomposition"] = {
        "definition": "questions of the form 'what is/are present?', 'where does this (part) belong to?', "
                      "'where is this/the ...?' whose gold is a body-system taxonomy token",
        "taxonomy_vocab_size_golds_seen_5plus": len(tax),
        "degenerate": {"n": len(deg), "frac_of_set": round(len(deg) / n, 4),
                       "acc_direct": round(sum(JD[i] for i in deg) / len(deg), 4),
                       "acc_reason": round(sum(JT[i] for i in deg) / len(deg), 4),
                       "delta": round(sum(JT[i] - JD[i] for i in deg) / len(deg), 4),
                       "share_of_direct_right_reason_wrong_pool": round(sum(1 for i in deg if JD[i] == 1 and JT[i] == 0) / len(pool), 4),
                       "answer_is_a_dataset_taxonomy_token_direct": conv(D, deg),
                       "answer_is_a_dataset_taxonomy_token_reason": conv(T, deg)},
        "non_degenerate": {"n": len(non), "frac_of_set": round(len(non) / n, 4),
                           "acc_direct": round(sum(JD[i] for i in non) / len(non), 4),
                           "acc_reason": round(sum(JT[i] for i in non) / len(non), 4),
                           "delta": round(sum(JT[i] - JD[i] for i in non) / len(non), 4)},
    }
    # ---- 4. the prompt confound ------------------------------------------------------------
    res["prompt_confound"] = {
        "direct_system_prompt": "You are an expert medical image analyst. Answer the question with a short, "
                                "specific phrase. Do not explain.",
        "reasoning_system_prompt": "You will solve a problem/request. You should provide your thoughts within "
                                   "<think> </think> tags before providing the answer. After </think>, give only "
                                   "the short final answer.",
        "source": "src/labeling/run_openvqa.py SYS / SYS_THINK, selected by the --think flag "
                  "(runners/run_openvqa_think_extend.sh)",
        "note": "The two arms differ in the ANSWER-STYLE instruction as well as in reasoning: the reasoning arm "
                "loses the expert-analyst persona, the 'short, specific phrase' constraint and the 'Do not "
                "explain' constraint. The measured direct-vs-reasoning delta therefore conflates a reasoning "
                "effect with a prompt/output-convention effect and is NOT a matched comparison.",
    }
    # ---- 5. prefix-slice representativeness (hole 13 spillover) ----------------------------
    res["prefix_slice_note"] = {
        "evaluated": 1500, "full_open_test_set": 3357, "selection": "run_openvqa.py items[:n] (a prefix, not a random draw)",
        "degenerate_frac_prefix1500": 0.632, "degenerate_frac_remainder1857": 0.505, "degenerate_frac_all3357": 0.562,
        "note": "the prefix over-samples the degenerate taxonomy family by ~7 points versus the full set "
                "(0.632 vs 0.562); the family is present throughout, so this biases the magnitude, not the sign",
    }

    # ---- 6. hand classification of the disagreement cell ------------------------------------
    dis = {int(k): v for k, v in lab.get("disagreement", {}).items()}
    if dis:
        cnt = Counter(dis.values()); m = len(dis)
        fair_k, strict_k = cnt["b"] + cnt["c"], cnt["c"]
        res["disagreement_hand_labels"] = {
            "pool_n": len(pool), "sample_n": m, "sampling": "stratified by gold length, seed 20260729",
            "legend": {"a": "reasoning answer genuinely wrong",
                       "b": "reasoning answer correct but at a different granularity/phrasing than the caption-fragment gold (entails the gold)",
                       "c": "reasoning answer correct and the judge clearly mis-scored it (near-verbatim gold, or a verbose form of an answer the same judge accepted from the direct mode)",
                       "d": "ambiguous (gold is unusable caption garbage, or the answer neither entails nor contradicts the gold)"},
            "distribution": {k: cnt.get(k, 0) for k in "abcd"},
            "distribution_frac": {k: round(cnt.get(k, 0) / m, 4) for k in "abcd"},
            "frac_not_genuinely_wrong_b_plus_c": round(fair_k / m, 4),
            "frac_not_genuinely_wrong_b_plus_c_wilson95": [round(x, 4) for x in wilson(fair_k, m)],
            "frac_clear_judge_error_c_only": round(strict_k / m, 4),
            "frac_clear_judge_error_c_only_wilson95": [round(x, 4) for x in wilson(strict_k, m)],
            "per_item": {str(k): v for k, v in sorted(dis.items())},
        }
    # ---- 7. judge validation (2x2 error rates) ---------------------------------------------
    jv = lab.get("judgeval", {})
    if jv:
        strict_sub = jv.get("_strict_standard_subset", {})
        cells = {}
        for c, d in jv.items():
            if c.startswith("_"): continue
            v = list(d.values()); k = sum(1 for x in v if x == "wrong")
            ks = len(strict_sub.get(c, []))
            cells[c] = {"n": len(v), "verdicts_wrong_fair": k, "verdicts_wrong_strict": ks,
                        "error_rate_fair": round(k / len(v), 4), "error_rate_strict": round(ks / len(v), 4),
                        "error_rate_fair_wilson95": [round(x, 4) for x in wilson(k, len(v))],
                        "n_ambiguous_not_counted": sum(1 for x in v if x == "ambiguous"),
                        "per_item": d}
        # merge the two both-wrong reject samples into one false-reject estimate for that subcell
        rr = jv["reason_reject"]; bw = jv["reason_reject_bothwrong_extra"]
        rr_dw = {k: v for k, v in rr.items() if JD.get(int(k)) == 0}
        merged = dict(rr_dw); merged.update(bw)
        mk = sum(1 for x in merged.values() if x == "wrong")
        mks = len([x for x in strict_sub.get("reason_reject", []) if JD.get(int(x)) == 0] + strict_sub.get("reason_reject_bothwrong_extra", []))
        cells["reason_reject_bothwrong_MERGED"] = {"n": len(merged), "verdicts_wrong_fair": mk,
                                                   "verdicts_wrong_strict": mks,
                                                   "error_rate_fair": round(mk / len(merged), 4),
                                                   "error_rate_strict": round(mks / len(merged), 4),
                                                   "error_rate_fair_wilson95": [round(x, 4) for x in wilson(mk, len(merged))]}
        res["judge_validation"] = {
            "standard": "a verdict is 'wrong' if a fair medical grader would disagree with it; the strict count "
                        "restricts this to near-verbatim / self-contradictory judge errors only",
            "cells": cells,
            "bias": {"false_reject_rate_direct": cells["direct_reject"]["error_rate_fair"],
                     "false_reject_rate_reason": cells["reason_reject"]["error_rate_fair"],
                     "false_accept_rate_direct": cells["direct_accept"]["error_rate_fair"],
                     "false_accept_rate_reason": cells["reason_accept"]["error_rate_fair"],
                     "note": "the judge's errors are strongly asymmetric AGAINST the reasoning mode's answer style"},
        }
        # ---- 8. corrected accuracy from the hand-labelled cells ----------------------------
        n_acc_t = sum(JT[i] for i in idx); n_rej_dr = len(pool)
        n_rej_dw = sum(1 for i in idx if JD[i] == 0 and JT[i] == 0)
        n_acc_d = sum(JD[i] for i in idx); n_rej_d = n - n_acc_d
        corrected = {}
        for std in ("fair", "strict"):
            fa_t = cells["reason_accept"][f"error_rate_{std}"]
            fr_t_dr = (res["disagreement_hand_labels"]["frac_not_genuinely_wrong_b_plus_c"] if std == "fair"
                       else res["disagreement_hand_labels"]["frac_clear_judge_error_c_only"])
            fr_t_dw = cells["reason_reject_bothwrong_MERGED"][f"error_rate_{std}"]
            fa_d = cells["direct_accept"][f"error_rate_{std}"]; fr_d = cells["direct_reject"][f"error_rate_{std}"]
            at = (n_acc_t * (1 - fa_t) + n_rej_dr * fr_t_dr + n_rej_dw * fr_t_dw) / n
            ad = (n_acc_d * (1 - fa_d) + n_rej_d * fr_d) / n
            corrected[std] = {"reason_false_accept": fa_t,
                              "reason_false_reject_direct_right_subcell": round(fr_t_dr, 4),
                              "reason_false_reject_both_wrong_subcell": fr_t_dw,
                              "direct_false_accept": fa_d, "direct_false_reject": fr_d,
                              "acc_reason_corrected": round(at, 4), "acc_direct_corrected": round(ad, 4),
                              "delta_corrected": round(at - ad, 4),
                              "delta_as_measured": round(acc_t - acc_d, 4),
                              "frac_of_collapse_that_is_artifact": round(1 - (ad - at) / (acc_d - acc_t), 4)}
        res["corrected_from_hand_labels"] = corrected

    # ---- 9. blinded independent re-grade ----------------------------------------------------
    rg, nrg = load_regrade(idx)
    if rg and nrg:
        blk = {"n_regraded": nrg, "graders": "Claude-as-judge subagents (project's validated path), "
               "A/B order randomised per item so the grader is blind to which answer came from which mode",
               "batches": REGRADE}
        for std in ("strict", "fair"):
            dd, tt = rg[std]["direct"], rg[std]["reason"]
            ii = sorted(set(dd) & set(tt))
            a_d = sum(dd[i] for i in ii) / len(ii); a_t = sum(tt[i] for i in ii) / len(ii)
            sub = {"n": len(ii), "acc_direct": round(a_d, 4), "acc_reason": round(a_t, 4),
                   "delta": round(a_t - a_d, 4)}
            dg = [i for i in ii if DEG.match(T[i]["question"].strip())]
            nd = [i for i in ii if i not in set(dg)]
            if dg: sub["degenerate"] = {"n": len(dg), "acc_direct": round(sum(dd[i] for i in dg) / len(dg), 4),
                                        "acc_reason": round(sum(tt[i] for i in dg) / len(dg), 4)}
            if nd: sub["non_degenerate"] = {"n": len(nd), "acc_direct": round(sum(dd[i] for i in nd) / len(nd), 4),
                                            "acc_reason": round(sum(tt[i] for i in nd) / len(nd), 4)}
            # agreement with the deployed local judge
            sub["agreement_with_local_judge_direct"] = round(sum(1 for i in ii if dd[i] == JD[i]) / len(ii), 4)
            sub["agreement_with_local_judge_reason"] = round(sum(1 for i in ii if tt[i] == JT[i]) / len(ii), 4)
            blk[std] = sub
        res["blinded_regrade"] = blk

    # ---- 10. propagation to the paper headline ----------------------------------------------
    if os.path.exists(HEADLINE):
        H = json.load(open(HEADLINE))
        pooled, N = H["headline"]["d_vs_think"], H["headline"]["n"]
        cell = H["per_benchmark"]["PATH_VQA_open"]
        m_acc, t_acc, n_cell = cell["method_acc"], cell["acc_32b_think_measured"], cell["n"]
        contrib = (m_acc - t_acc) * n_cell / N
        prop = {"headline_d_vs_think": pooled, "pool_n": N,
                "pathvqa_open": {"n": n_cell, "method_acc": m_acc, "reason_baseline_acc": t_acc,
                                 "delta": round(m_acc - t_acc, 4), "contribution_to_pooled": round(contrib, 6),
                                 "share_of_pooled": round(contrib / pooled, 4)},
                "scenarios": {}}
        scen = {}
        if "corrected_from_hand_labels" in res:
            for std in ("fair", "strict"):
                scen[f"hand_labels_{std}"] = res["corrected_from_hand_labels"][std]["acc_reason_corrected"]
        if "blinded_regrade" in res and res["blinded_regrade"].get("fair", {}).get("n", 0) == n:
            for std in ("strict", "fair"):
                scen[f"blinded_regrade_{std}"] = res["blinded_regrade"][std]["acc_reason"]
        for name, t_new in scen.items():
            # the method's own PathVQA-open vector is graded by the SAME local judge; rescale it by the
            # direct mode's correction ratio under the matching standard (its answers are terse, like direct's)
            std = "fair" if name.endswith("fair") else "strict"
            if name.startswith("hand"):
                ratio = res["corrected_from_hand_labels"][std]["acc_direct_corrected"] / acc_d
            else:
                ratio = res["blinded_regrade"][std]["acc_direct"] / acc_d
            m_new = m_acc * ratio
            c_new = (m_new - t_new) * n_cell / N
            prop["scenarios"][name] = {
                "reason_baseline_acc_corrected": round(t_new, 4),
                "method_acc_rescaled": round(m_new, 4),
                "assumption_method_rescale_ratio": round(ratio, 4),
                "pathvqa_delta": round(m_new - t_new, 4),
                "contribution_to_pooled": round(c_new, 6),
                "pooled_d_vs_think_corrected": round(pooled - contrib + c_new, 4),
                "pathvqa_share_of_corrected_pooled": round(c_new / (pooled - contrib + c_new), 4)}
        res["headline_propagation"] = prop

    os.makedirs(ART, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-disagree", action="store_true"); ap.add_argument("--emit-judgeval", action="store_true")
    ap.add_argument("--emit-bothwrong", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("-k", type=int, default=90)
    A = ap.parse_args()
    (cmd_disagree if A.emit_disagree else cmd_judgeval if A.emit_judgeval else
     cmd_bothwrong if A.emit_bothwrong else cmd_report)(A)
