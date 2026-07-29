#!/usr/bin/env python3
"""
matched_prompt_reasoning.py -- THE DECISIVE EXPERIMENT: is the project's "chain-of-thought reasoning
HURTS perception open-text VQA" result a REASONING effect or a PROMPT/output-convention effect?

THE PROBLEM (results/cascade_methods/artifacts/pathvqa_judge_audit.json -> "prompt_confound").
The open-text direct-vs-reasoning comparison behind that claim was NOT matched.  src/labeling/run_openvqa.py
served the two arms different system prompts:
  direct     (SYS)        "You are an expert medical image analyst. Answer the question with a short,
                           specific phrase. Do not explain."
  reasoning  (SYS_THINK)  "You will solve a problem/request. You should provide your thoughts within
                           <think> </think> tags before providing the answer. After </think>, give only
                           the short final answer."
The reasoning arm LOSES the expert-analyst persona, the "short, specific phrase" constraint and the
"Do not explain" constraint, so the measured gap conflates reasoning with output convention.  Symptom:
on PathVQA-open 63.2% of questions have a body-system taxonomy token as gold, and the direct arm emits
such a token 75.5% of the time versus the reasoning arm's 15.1%.

THE FIX -- TWO matched reasoning arms, because the obvious one has a trap.
  A) SYS_THINK_MATCHED (--think_matched): the literal fix -- keep SYS's persona, "short, specific phrase"
     and "Do not explain", add the <think></think> instruction.  TRAP: it paraphrases the reasoning
     trigger, and these models only emit a trace when the trigger sentences appear VERBATIM (CLAUDE.md
     s8).  Measured: the trace fires on only ~1/4 of items, so arm A silently turns reasoning OFF instead
     of matching prompts -- it cannot attribute the effect.  Kept and reported because that fragility is
     itself a result.
  B) SYS_THINK_MATCHED2 (--think_matched2): THE DECISIVE ARM.  Keeps SYS_THINK's trigger sentences
     VERBATIM (trace still fires) and replaces only its answer-style clause ("give only the short final
     answer") with the direct prompt's persona + "short, specific phrase" + "Do not explain".  Reasoning
     is ON and the only difference from the direct arm is the reason-first instruction.
Both were run with Lingshu-32B over the EXACT evaluated idx sets the headline is scored on (SLAKE-open
645 / VQA-RAD-open 200 / PathVQA-open 1500, integrated_pandora.load_open_rows order) with everything else
identical to the unmatched think run (cap320, greedy n=1, max_tokens 512, tp=2, answer = text after the
LAST </think>), then judged through the SAME judge (src/labeling/run_judge.py, judge_ok).

WHAT THIS PRINTS/WRITES.
  1. per dataset + pooled-open: direct vs UNMATCHED vs MATCHED-A vs MATCHED-B judged accuracy, with
     paired-bootstrap 95% CIs for every contrast (NBOOT=10000, paper_baselines.paired_ci);
  2. gap decomposition: how much of the unmatched direct-minus-reasoning gap is prompt convention and how
     much survives as reasoning;
  3. diagnostics: reasoning-TRACE EMISSION RATE, taxonomy-token emission rate (the audit's definition,
     re-used verbatim) and answer-length / generation-token distributions per arm;
  4. VERDICT (from arm B): survives / survives partially / collapses, per dataset and pooled;
  5. PROPAGATION to the pooled headline: f8_mode_vsthink_ci's Variant-B accuracy-max-vs-always-32B-THINK
     delta (+0.0245) recomputed with the open-cell always-32B-THINK vector swapped unmatched -> matched,
     with a fresh paired-bootstrap CI.  Method vectors untouched; only the THINK baseline's prompt changed.
NO GPU, NO fabricated numbers -- every number comes from the judged dumps on disk.  Arms whose dumps are
absent/incomplete are skipped and listed in "arms_missing", so this can be run before the runs finish.
Launch from the repo root:  python3 src/cascade_methods/matched_prompt_reasoning.py
"""
import argparse, ast, json, os, re
from collections import Counter

import numpy as np

import paper_baselines as PB

ROOT = PB.ROOT
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "matched_prompt_reasoning_2026-07-29.json")
RUNNER = os.path.join(ROOT, "src/labeling/run_openvqa.py")

# arm -> (checkpoint dir, tag, the run_openvqa.py system-prompt constant it was served)
ARMS = {
    "direct":            ("ckpts/openvqa/strong_lingshu",                 "lingshu32b",                 "SYS"),
    "direct_unstyled":   ("ckpts/openvqa/strong_lingshu_direct_unstyled", "lingshu32b_direct_unstyled", "SYS_DIRECT_UNSTYLED"),
    "reason_unmatched":  ("ckpts/openvqa/strong_lingshu_think",           "lingshu32b_think",           "SYS_THINK"),
    "reason_matched_A":  ("ckpts/openvqa/strong_lingshu_think_matched",   "lingshu32b_think_matched",   "SYS_THINK_MATCHED"),
    "reason_matched_B":  ("ckpts/openvqa/strong_lingshu_think_matched2",  "lingshu32b_think_matched2",  "SYS_THINK_MATCHED2"),
}
BASE = "direct"                       # the headline's direct arm (styled prompt)
PRIMARY = "reason_matched_B"          # best matched-reasoning arm (styled convention, partial trace)
CLEAN = ("reason_unmatched", "direct_unstyled")   # the CLEAN reasoning isolation (fixed unstyled convention)
REASON_ARMS = ["reason_unmatched", "reason_matched_A", "reason_matched_B"]
# the style x reasoning 2x2: arm -> (output convention, reasoning instruction)
CELL = {"direct": ("styled", "off"), "direct_unstyled": ("unstyled", "off"),
        "reason_unmatched": ("unstyled", "on"), "reason_matched_A": ("styled", "on"),
        "reason_matched_B": ("styled", "on")}
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
NAME = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open", "pathvqa_open": "PATH_VQA_open"}
IDXFILES = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu_think_matched/idxfiles")
# the audit's degenerate-question family (gold is a body-system taxonomy token, not a real answer)
DEG = re.compile(r"^(what (is|are) present|where (does this|is this|is the)|where does this part belong)", re.I)
GEN_CAP = 512


# ------------------------------------------------------------------ loading
def sys_prompts():
    """Read the system-prompt constants straight out of run_openvqa.py so the reported prompts are the
    literal strings that were served (no transcription drift)."""
    want = {a[2] for a in ARMS.values()}
    out = {}
    for node in ast.parse(open(RUNNER).read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in want:
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


def _rows(path, key=None):
    if not os.path.exists(path):
        return None
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            out[r["idx"]] = int(r["judge_ok"]) if key == "judge" else r
    return out


def arm_files(arm, ds):
    d, tag, _ = ARMS[arm]
    p = os.path.join(ROOT, d, f"ckpt_{ds}_{tag}")
    return p + ".jsonl", p + ".judge.jsonl"


def evaluated_idx(ds):
    """The EXACT idx (and order) the headline scores the open cells on: integrated_pandora.load_open_rows.
    Cross-checked against the frozen allowlist the generation runners used."""
    import integrated_pandora as IP
    idx = [r["idx"] for r in IP.load_open_rows(ds)]
    allow = os.path.join(IDXFILES, f"{ds}_evaluated.json")
    if os.path.exists(allow):
        assert idx == json.load(open(allow)), f"{ds}: generation allowlist != load_open_rows idx order"
    return idx


def load_all():
    """{ds: {arm: (preds, judged_vec)}} over the evaluated idx, plus the list of arms that are complete
    on every dataset (an arm missing/short anywhere is dropped everywhere so all pools are comparable)."""
    data, status = {}, {}
    for ds in DS:
        idx = evaluated_idx(ds)
        data[ds] = {"idx": idx, "arms": {}}
        for arm in ARMS:
            pf, jf = arm_files(arm, ds)
            P, J = _rows(pf), _rows(jf, "judge")
            if P is None or J is None:
                status.setdefault(arm, []).append(f"{ds}: missing {'dump' if P is None else 'judge'}")
                continue
            miss = [i for i in idx if i not in P or i not in J]
            if miss:
                status.setdefault(arm, []).append(f"{ds}: {len(miss)}/{len(idx)} evaluated idx unjudged")
                continue
            data[ds]["arms"][arm] = (P, np.array([J[i] for i in idx], float))
    complete = [a for a in ARMS if all(a in data[ds]["arms"] for ds in DS)]
    return data, complete, status


# ------------------------------------------------------------------ diagnostics
def nrm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def tax_vocab(P, idx):
    """Audit definition, verbatim: normalized golds seen >=5x among the evaluated idx == the dataset's
    de-facto answer taxonomy (a closed label set the direct arm can pattern-match into)."""
    return {nrm(g) for g, c in Counter(P[i]["gold"].lower().strip() for i in idx).items() if c >= 5}


def gold_bucket(g):
    n = len(str(g).split())
    return "1" if n == 1 else "2" if n == 2 else "3-4" if n <= 4 else "5+"


def dist(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {}
    return dict(mean=round(float(np.mean(v)), 2), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), max=float(max(v)))


def arm_diag(P, idx, tax, deg):
    ans = [str(P[i]["modal_pred"]) for i in idx]
    words = [len(a.split()) for a in ans]
    gen = [P[i].get("gen_tokens") for i in idx if P[i].get("gen_tokens") is not None]
    d = dict(
        n=len(idx),
        taxonomy_token_rate=round(sum(1 for i in idx if nrm(P[i]["modal_pred"]) in tax) / len(idx), 4),
        answer_words=dist(words), answer_chars=dist([len(a) for a in ans]),
        gen_tokens=dist(gen),
        frac_at_gen_cap=round(sum(1 for g in gen if g >= GEN_CAP) / max(1, len(gen)), 4),
        frac_empty_answer=round(sum(1 for a in ans if not a.strip()) / len(idx), 4),
        frac_answer_over_15_words=round(sum(1 for w in words if w > 15) / len(idx), 4),
    )
    if deg:
        d["taxonomy_token_rate_degenerate_subset"] = round(
            sum(1 for i in deg if nrm(P[i]["modal_pred"]) in tax) / len(deg), 4)
    raw = [i for i in idx if "raw" in P[i]]
    if raw:   # --save_raw arms: the reasoning trace is directly checkable
        d["raw_retained_n"] = len(raw)
        d["reasoning_trace_rate"] = round(
            sum(1 for i in raw if any("</think>" in t for t in P[i]["raw"])) / len(raw), 4)
    return d


def subsets(idx, V, Pd, deg):
    """Where the (prompt or reasoning) effect lives: the audit's degenerate taxonomy family vs the rest,
    and gold-length buckets, for every available arm."""
    pos = {i: k for k, i in enumerate(idx)}

    def blk(ii):
        if not ii:
            return None
        sel = [pos[i] for i in ii]
        b = dict(n=len(ii), **{f"acc_{a}": round(float(V[a][sel].mean()), 4) for a in V})
        for a in V:
            if a != BASE:
                b[f"d_{a}_minus_direct"] = round(float(V[a][sel].mean() - V[BASE][sel].mean()), 4)
        return b

    dset = set(deg)
    out = dict(degenerate_taxonomy_family=blk(deg),
               non_degenerate=blk([i for i in idx if i not in dset]), by_gold_length={})
    for b in ["1", "2", "3-4", "5+"]:
        out["by_gold_length"][b] = blk([i for i in idx if gold_bucket(Pd[i]["gold"]) == b])
    return out


def trace_conditional(idx, V, P, arms):
    """For arms with --save_raw: judged accuracy SPLIT by whether the model actually emitted a reasoning
    trace, with the direct arm's accuracy on the same items for reference. This is a SELECTED comparison
    (the split is the model's own behaviour, not a randomisation), so it is diagnostic only -- it says
    where an arm's items landed, not what reasoning causes."""
    pos = {i: k for k, i in enumerate(idx)}
    out = {}
    for a in arms:
        if a == BASE or not any("raw" in P[a][i] for i in idx):
            continue
        fired = [i for i in idx if any("</think>" in t for t in P[a][i].get("raw", []))]
        nof = [i for i in idx if i not in set(fired)]
        blk = {}
        for lab, ii in (("trace_fired", fired), ("no_trace", nof)):
            if not ii:
                blk[lab] = None
                continue
            sel = [pos[i] for i in ii]
            blk[lab] = dict(n=len(ii), frac=round(len(ii) / len(idx), 4),
                            acc_this_arm=round(float(V[a][sel].mean()), 4),
                            acc_direct_same_items=round(float(V[BASE][sel].mean()), 4),
                            d_vs_direct=round(float(V[a][sel].mean() - V[BASE][sel].mean()), 4))
        out[a] = blk
    return out


# ------------------------------------------------------------------ core comparison
def ci(a, b):
    r = PB.paired_ci(np.asarray(a, float), np.asarray(b, float))
    return dict(delta=r["delta"], lo=r["lo"], hi=r["hi"], sig=r["sig"], n=r["n"])


def verdict(d_matched, ci_matched, d_unmatched):
    """SURVIVES: the matched reasoning arm is still significantly BELOW direct and keeps >=50% of the
    unmatched gap.  SURVIVES_PARTIALLY: still significantly below, but most of the gap was prompt.
    COLLAPSES: no longer significantly below direct."""
    if d_matched >= 0 or not ci_matched["sig"]:
        return "COLLAPSES"
    kept = (d_matched / d_unmatched) if d_unmatched else float("nan")
    return "SURVIVES" if kept >= 0.5 else "SURVIVES_PARTIALLY"


def two_by_two(V, arms):
    """The style x reasoning 2x2 and the effects that can be read off it.

      * reasoning_effect_at_unstyled = reason_unmatched - direct_unstyled
        The CLEAN reasoning isolation: identical output convention on both sides, and a direct prompt cannot
        accidentally start reasoning (whereas a 'matched reasoning' prompt can stop it).  THE DECISIVE NUMBER.
      * output_convention_effect_reasoning_off = direct - direct_unstyled
        What the direct arm gains purely from the persona + 'short, specific phrase' + 'Do not explain'
        wording, with reasoning off in both arms.
      * reasoning_effect_at_styled = reason_matched_B - direct  (trace-suppression caveat: see diagnostics)
      * original_unmatched_gap = reason_unmatched - direct  == the headline's direct-vs-reasoning gap, which
        the identity below splits into a convention part and a reasoning part exactly.
    """
    if not all(a in V for a in CLEAN) or BASE not in V:
        return None
    R, Du, Ds = V[CLEAN[0]], V[CLEAN[1]], V[BASE]
    o = dict(
        cells={a: dict(convention=CELL[a][0], reasoning=CELL[a][1], acc=round(float(V[a].mean()), 4))
               for a in arms},
        reasoning_effect_at_unstyled=dict(arms=f"{CLEAN[0]} - {CLEAN[1]}",
                                          delta=round(float(R.mean() - Du.mean()), 4), ci=ci(R, Du)),
        output_convention_effect_reasoning_off=dict(arms=f"{BASE} - {CLEAN[1]}",
                                                    delta=round(float(Ds.mean() - Du.mean()), 4), ci=ci(Ds, Du)),
        original_unmatched_gap=dict(arms=f"{CLEAN[0]} - {BASE}",
                                    delta=round(float(R.mean() - Ds.mean()), 4), ci=ci(R, Ds)))
    if PRIMARY in V:
        o["reasoning_effect_at_styled"] = dict(arms=f"{PRIMARY} - {BASE}",
                                               delta=round(float(V[PRIMARY].mean() - Ds.mean()), 4),
                                               ci=ci(V[PRIMARY], Ds),
                                               caveat="the styled reasoning prompt suppresses the <think> "
                                                      "trace on part of the set (diagnostics.reasoning_trace_rate)")
    gap = o["original_unmatched_gap"]["delta"]
    conv = o["output_convention_effect_reasoning_off"]["delta"]
    rsn = o["reasoning_effect_at_unstyled"]["delta"]
    o["identity_check"] = dict(
        statement="original_unmatched_gap == reasoning_effect_at_unstyled - output_convention_effect_reasoning_off",
        lhs=gap, rhs=round(rsn - conv, 4), residual=round(gap - (rsn - conv), 4))
    if gap < 0:                       # the headline gap is a LOSS for reasoning; attribute it
        o["attribution_of_the_unmatched_gap"] = dict(
            share_output_convention=round(conv / -gap, 4),
            share_reasoning=round(rsn / gap, 4),
            note="shares of the original direct-minus-reasoning gap (they sum to 1 by the identity above); "
                 "the output-convention share is the part the unmatched comparison wrongly charged to "
                 "reasoning. A NEGATIVE reasoning share means reasoning HELPED once the convention is held "
                 "fixed, and the whole apparent loss (plus more) was the prompt.")
    o["verdict"] = verdict(rsn, o["reasoning_effect_at_unstyled"]["ci"], gap)
    o["verdict_basis"] = ("reasoning_effect_at_unstyled -- the clean contrast (same output convention both "
                          "sides, reasoning instruction the only difference)")
    return o


def contrasts(V, arms):
    """Every contrast of interest on one aligned block of judged vectors."""
    out = {f"acc_{a}": round(float(V[a].mean()), 4) for a in arms}
    dun = float(V["reason_unmatched"].mean() - V[BASE].mean()) if "reason_unmatched" in V else None
    for a in arms:
        if a == BASE:
            continue
        d = float(V[a].mean() - V[BASE].mean())
        out[f"d_{a}_minus_direct"] = round(d, 4)
        out[f"ci_{a}_vs_direct"] = ci(V[a], V[BASE])
        if a in REASON_ARMS and a != "reason_unmatched" and dun is not None:
            out[f"d_{a}_minus_unmatched"] = round(float(V[a].mean() - V["reason_unmatched"].mean()), 4)
            out[f"ci_{a}_vs_unmatched"] = ci(V[a], V["reason_unmatched"])
            out[f"prompt_share_of_unmatched_gap_{a}"] = round((d - dun) / -dun, 4) if dun < 0 else None
            out[f"reasoning_share_of_unmatched_gap_{a}"] = round(d / dun, 4) if dun else None
            out[f"verdict_{a}"] = verdict(d, out[f"ci_{a}_vs_direct"], dun)
    tbt = two_by_two(V, arms)
    if tbt:
        out["two_by_two"] = tbt
        out["verdict"] = tbt["verdict"]                       # clean contrast wins
        out["verdict_arm"] = "reason_unmatched vs direct_unstyled (matched output convention)"
    elif PRIMARY in arms:
        out["verdict"] = out[f"verdict_{PRIMARY}"]
        out["verdict_arm"] = PRIMARY
    return out


def build(propagate=True):
    PB.RNG = np.random.default_rng(12345)          # deterministic CIs for this script's call order
    prompts = sys_prompts()
    data, arms, status = load_all()
    if BASE not in arms or not any(a in arms for a in REASON_ARMS):
        raise RuntimeError(f"nothing to compare: complete arms = {arms}; status = {status}")

    per_ds, vecs = {}, {}
    for ds in DS:
        idx = data[ds]["idx"]
        V = {a: data[ds]["arms"][a][1] for a in arms}
        P = {a: data[ds]["arms"][a][0] for a in arms}
        vecs[ds] = V
        tax = tax_vocab(P[BASE], idx)
        deg = [i for i in idx if DEG.match(str(P[BASE][i]["question"]).strip())]
        per_ds[NAME[ds]] = dict(
            dataset_key=ds, n=len(idx), **contrasts(V, arms),
            degenerate_subset_n=len(deg), taxonomy_vocab_size_golds_seen_5plus=len(tax),
            diagnostics={a: arm_diag(P[a], idx, tax, deg) for a in arms},
            subset_decomposition=subsets(idx, V, P[BASE], deg),
            trace_conditional=trace_conditional(idx, V, P, arms))

    pool = {a: np.concatenate([vecs[ds][a] for ds in DS]) for a in arms}
    pooled_open = dict(n=int(len(pool[BASE])), benchmarks=[NAME[d] for d in DS], **contrasts(pool, arms))

    out = dict(
        title="THE DECISIVE EXPERIMENT -- matched-prompt reasoning vs direct on open-text medical VQA "
              "(Lingshu-32B, same judge, same evaluated idx). Does 'reasoning hurts perception open-text "
              "VQA' survive when the reasoning arm keeps the direct arm's persona and answer-style "
              "constraints and differs ONLY by the reason-first instruction?",
        reproduce=["bash runners/run_openvqa_think_matched.sh    # arm A generation (chunked, tp=2 guarded)",
                   "bash runners/run_openvqa_think_matched2.sh   # arm B generation (decisive arm)",
                   "bash runners/run_judge_think_matched.sh ; bash runners/run_judge_think_matched2.sh",
                   "python3 src/cascade_methods/matched_prompt_reasoning.py",
                   "(runners/run_matched_prompt_chain.sh serializes B + both judges on the shared GPUs)"],
        no_gpu_this_script=True, no_fabricated_numbers=True, n_bootstrap=PB.NBOOT,
        arms_compared=arms, arms_missing=status, decisive_arm=PRIMARY,
        prompts={a: dict(constant=ARMS[a][2], text=prompts.get(ARMS[a][2]),
                         dir=ARMS[a][0], tag=ARMS[a][1]) for a in ARMS},
        prompt_design=dict(
            SYS_THINK_MATCHED="arm A -- the literal matched prompt: SYS's persona ('You are an expert medical "
                              "image analyst'), its answer-style constraint ('answer the question with a short, "
                              "specific phrase') and its no-explanation constraint, plus the <think></think> "
                              "reason-first instruction. It PARAPHRASES the reasoning trigger, and this model "
                              "family only emits a trace when the trigger sentences appear verbatim (CLAUDE.md "
                              "s8) -- see diagnostics.reasoning_trace_rate: reasoning is mostly OFF in this arm, "
                              "so it measures 'reasoning requested but not performed', not matched reasoning.",
            SYS_THINK_MATCHED2="arm B -- THE DECISIVE ARM: SYS_THINK's trigger sentences kept VERBATIM (trace "
                               "fires) with its answer-style clause 'After </think>, give only the short final "
                               "answer' replaced by the direct arm's persona + 'answer the question with a short, "
                               "specific phrase' + 'Do not explain'. Reasoning ON; the only difference from the "
                               "direct arm is the reason-first instruction. Ordering caveat: the persona sentence "
                               "sits after the trigger sentences (it must, or the trace stops firing), whereas in "
                               "SYS it comes first.",
            source_of_truth="src/labeling/run_openvqa.py, read by ast at runtime."),
        held_constant="model (Lingshu-32B, same snapshot), cap320, greedy temp=0 n_samples=1, "
                      "max_model_len 4096, max_tokens 512 (reasoning arms), tp=2, answer extraction (text "
                      "after the LAST </think>), evaluated idx sets, and the judge (run_judge.py judge_ok, "
                      "neutral MedVLThinker/Qwen2.5-32B grader).",
        per_dataset=per_ds,
        pooled_open=pooled_open,
        taxonomy_definition="normalized (lowercase, non-alphanumerics stripped) gold strings occurring >=5 "
                            "times among the dataset's evaluated idx; emission rate = fraction of items whose "
                            "normalized final answer is in that vocabulary. Identical definition to "
                            "pathvqa_judge_audit.py's 'answer_is_a_dataset_taxonomy_token_*'.",
        caveats=[
            "The direct arm's dump is the pre-existing one the headline uses (max_tokens 64; PathVQA generated "
            "at tp=1, SLAKE/VQA-RAD at tp=2); only the reasoning arms were re-generated. Any tp-related "
            "numerical drift is a pre-existing property of the headline, not introduced here.",
            "run_openvqa.py's extract() keeps the LAST line after the final </think>; both matched dumps were "
            "written with --save_raw so this step and the trace rate are re-verifiable offline.",
            "PathVQA-open's evaluated 1500 is a prefix of the 3357-item open test set and over-samples the "
            "degenerate taxonomy family (0.632 vs 0.562 overall) -- unchanged from the audit.",
            "Arm B changes the persona sentence's POSITION relative to SYS (it must follow the verbatim trigger "
            "sentences). Position, not content, is the residual difference from the direct prompt.",
        ])
    if propagate:
        out["headline_propagation"] = propagation(vecs, arms)
    os.makedirs(ART, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    console(out)
    print(f"\nwrote {OUT}")
    return out


# ------------------------------------------------------------------ headline propagation
def propagation(vecs, arms):
    """Recompute the pooled headline (f8_mode_vsthink_ci: FLOP-negative accuracy-max vs always-32B-THINK,
    Variant B) with the open-cell always-32B-THINK vector swapped UNMATCHED -> each MATCHED arm. The method
    vectors and every MCQ cell are untouched: the only change is which prompt the THINK baseline was served."""
    import method_final_mmmu_corrected as MFC
    import opentext_32b_think_full as OTF
    cells = PB.build_cells()
    MFC.add_v2_vectors(cells)
    meas = OTF.measured_open_think()                      # unmatched judged think = the headline's baseline
    ORDER = PB.MCQ_ORDER + PB.OPEN_ORDER
    MMMU = "MMMU-Medical-val"
    which = [a for a in REASON_ARMS if a in arms]
    okT = {}
    for name in PB.OPEN_ORDER:
        ds = PB.OPEN_KEY[name]
        assert cells[name]["n"] == meas[name]["n"] == len(vecs[ds][BASE])
        # sanity: the unmatched vector this script loaded IS the one the headline uses
        assert np.array_equal(np.asarray(meas[name]["okT"], float), vecs[ds]["reason_unmatched"]), name
        okT[name] = {a: vecs[ds][a] for a in which}

    def thk(k, a):
        return okT[k][a] if k in PB.OPEN_ORDER else np.asarray(cells[k]["okT"], float)

    def am2(k):
        return np.asarray(cells[k]["am2_ok"], float)

    POOLS = {"variant_b_mmmu_excluded": [k for k in ORDER if k != MMMU],
             "full_suite": ORDER, "open_only": PB.OPEN_ORDER}
    res = {}
    for lab, keys in POOLS.items():
        n = sum(cells[k]["n"] for k in keys)
        am = sum(float(am2(k).mean()) * cells[k]["n"] for k in keys) / n
        blk = dict(benchmarks=keys, n=int(n), method_acc=round(am, 4))
        A = np.concatenate([am2(k) for k in keys])
        for a in which:
            aT = sum(float(thk(k, a).mean()) * cells[k]["n"] for k in keys) / n
            T = np.concatenate([thk(k, a) for k in keys])
            c = PB.paired_ci(A, T)
            blk[f"think_{a}"] = dict(acc_32b_think=round(aT, 4), d_vs_think=round(am - aT, 4),
                                     ci95=[c["lo"], c["hi"]], sig=c["sig"])
        for a in which:
            if a != "reason_unmatched":
                blk[f"headline_shift_{a}"] = round(
                    blk[f"think_{a}"]["d_vs_think"] - blk["think_reason_unmatched"]["d_vs_think"], 4)
        res[lab] = blk
    res["per_open_cell"] = {
        k: dict(n=int(cells[k]["n"]), method_acc=round(float(am2(k).mean()), 4),
                **{f"think_{a}": round(float(okT[k][a].mean()), 4) for a in which},
                **{f"d_vs_think_{a}": round(float(am2(k).mean() - okT[k][a].mean()), 4) for a in which},
                **{f"ci_vs_think_{a}": PB.paired_ci(am2(k), okT[k][a])
                   for a in which if a == PRIMARY})
        for k in PB.OPEN_ORDER}
    res["note"] = ("Baseline row reproduced from f8_mode_vsthink_ci.json (Variant B: method 0.5836 vs "
                   "32B-think 0.5591, d=+0.0245, n=42224). Each 'think_<arm>' row replaces ONLY the three "
                   "open cells' always-32B-THINK vector with that arm's. The separate PathVQA judging "
                   "correction (+0.019..+0.0226 scenarios in pathvqa_judge_audit.json) was hand-derived on "
                   "the UNMATCHED reasoning answers and cannot be transferred to a different arm without "
                   "re-labeling, so it is NOT stacked on top of these numbers.")
    return res


# ------------------------------------------------------------------ console
def console(out):
    W = 124
    print("=" * W)
    print("MATCHED-PROMPT REASONING vs DIRECT -- open-text medical VQA (Lingshu-32B, same judge, same idx)")
    print("=" * W)
    for a in out["arms_compared"]:
        p = out["prompts"][a]
        print(f"  {a:<18} [{p['constant']}]\n    {p['text']}")
    if out["arms_missing"]:
        print(f"  arms skipped (incomplete): {json.dumps(out['arms_missing'])}")
    arms = [a for a in out["arms_compared"] if a != BASE]

    print("\n" + "=" * W)
    hdr = f"  {'dataset':<14}{'n':>6}{'direct':>9}" + "".join(f"{a.replace('reason_',''):>12}" for a in arms)
    print(hdr + f"{'d(B-dir)':>10}{'CI(B-dir)':>21}{'verdict':>21}")
    for k, r in list(out["per_dataset"].items()) + [("POOLED_OPEN", out["pooled_open"])]:
        line = f"  {k:<14}{r['n']:>6}{r['acc_direct']:>9.4f}" + "".join(f"{r['acc_'+a]:>12.4f}" for a in arms)
        if PRIMARY in arms:
            c = r[f"ci_{PRIMARY}_vs_direct"]
            line += (f"{r['d_'+PRIMARY+'_minus_direct']:>+10.4f}"
                     f"{('[%+.4f,%+.4f]' % (c['lo'], c['hi'])):>21}{r['verdict']:>21}")
        print(line)

    print("\n  matched-reasoning arms vs the direct arm (styled convention; trace-suppression caveat):")
    for k, r in list(out["per_dataset"].items()) + [("POOLED_OPEN", out["pooled_open"])]:
        for a in arms:
            if a not in REASON_ARMS or a == "reason_unmatched":
                continue
            print(f"    {k:<14}{a:<18} d_vs_direct {r['d_'+a+'_minus_direct']:+.4f}  "
                  f"d_vs_unmatched {r['d_'+a+'_minus_unmatched']:+.4f}  "
                  f"prompt-share {str(r.get('prompt_share_of_unmatched_gap_'+a)):>8}  "
                  f"reasoning-share {str(r.get('reasoning_share_of_unmatched_gap_'+a)):>8}  [{r['verdict_'+a]}]")

    if "two_by_two" in out["pooled_open"]:
        print("\n" + "=" * W)
        print("THE 2x2 (output convention x reasoning) -- the CLEAN reasoning isolation")
        print("=" * W)
        for k, r in list(out["per_dataset"].items()) + [("POOLED_OPEN", out["pooled_open"])]:
            t = r["two_by_two"]
            print(f"  {k}  (n={r['n']})")
            print(f"    {'cell':<20}{'convention':<12}{'reasoning':<11}{'acc':>9}")
            for a, c in t["cells"].items():
                print(f"    {a:<20}{c['convention']:<12}{c['reasoning']:<11}{c['acc']:>9.4f}")
            for key in ("reasoning_effect_at_unstyled", "output_convention_effect_reasoning_off",
                        "reasoning_effect_at_styled", "original_unmatched_gap"):
                if key in t:
                    e = t[key]; c = e["ci"]
                    print(f"    {key:<42}{e['delta']:>+9.4f}  CI[{c['lo']:+.4f},{c['hi']:+.4f}]  "
                          f"{'SIG' if c['sig'] else 'ns'}   ({e['arms']})")
            att = t.get("attribution_of_the_unmatched_gap")
            if att:
                print(f"    attribution of the original gap: output-convention "
                      f"{att['share_output_convention']:+.4f}  reasoning {att['share_reasoning']:+.4f}"
                      f"   (identity residual {t['identity_check']['residual']:+.6f})")
            print(f"    VERDICT: {t['verdict']}")

    print("\n  diagnostics -- reasoning-trace rate / taxonomy-token rate / answer words (mean|med|p90) / gen tokens:")
    for k, r in out["per_dataset"].items():
        print(f"    {k}  (taxonomy vocab {r['taxonomy_vocab_size_golds_seen_5plus']}, "
              f"degenerate-family n={r['degenerate_subset_n']})")
        for a in out["arms_compared"]:
            d = r["diagnostics"][a]; w = d["answer_words"]; g = d["gen_tokens"]
            tr = d.get("reasoning_trace_rate")
            extra = (f" deg-tax {d['taxonomy_token_rate_degenerate_subset']:.3f}"
                     if "taxonomy_token_rate_degenerate_subset" in d else "")
            print(f"      {a:<18} trace {('%.3f' % tr) if tr is not None else '  n/a'}   "
                  f"tax {d['taxonomy_token_rate']:.3f}{extra}   words "
                  f"{w['mean']:>5.1f}|{w['median']:>4.0f}|{w['p90']:>4.0f}   gen "
                  f"{g['mean']:>6.1f}|{g['median']:>5.0f}|{g['p90']:>5.0f}   >15w {d['frac_answer_over_15_words']:.3f}")

    print("\n  where the effect lives (acc " + " / ".join(out["arms_compared"]) + "):")
    for k, r in out["per_dataset"].items():
        for lab in ("degenerate_taxonomy_family", "non_degenerate"):
            b = r["subset_decomposition"][lab]
            if b:
                accs = " / ".join(f"{b['acc_'+a]:.4f}" for a in out["arms_compared"])
                print(f"    {k:<14}{lab:<28} n={b['n']:>5}  {accs}")

    print("\n  trace-conditional (DIAGNOSTIC ONLY -- split by the model's own behaviour, not randomised):")
    for k, r in out["per_dataset"].items():
        for a, blk in r.get("trace_conditional", {}).items():
            for lab, b in blk.items():
                if b:
                    print(f"    {k:<14}{a:<18}{lab:<12} n={b['n']:>5} ({b['frac']:.3f})  arm {b['acc_this_arm']:.4f}"
                          f"  direct-same-items {b['acc_direct_same_items']:.4f}  d {b['d_vs_direct']:+.4f}")
    if "headline_propagation" in out:
        print("\n" + "=" * W)
        print("PROPAGATION -- pooled headline (accuracy-max F8+F10 vs always-32B-THINK) under each THINK prompt")
        print("=" * W)
        hp = out["headline_propagation"]
        for lab in ("variant_b_mmmu_excluded", "full_suite", "open_only"):
            b = hp[lab]
            print(f"  {lab:<26} n={b['n']:>6}  method {b['method_acc']:.4f}")
            for a in [x for x in REASON_ARMS if f"think_{x}" in b]:
                v = b[f"think_{a}"]
                sh = b.get(f"headline_shift_{a}")
                print(f"    THINK {a:<18} {v['acc_32b_think']:.4f}  d {v['d_vs_think']:+.4f} "
                      f"CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] {'SIG' if v['sig'] else 'ns'}"
                      + (f"   (shift {sh:+.4f})" if sh is not None else "   (headline baseline)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_propagate", action="store_true", help="skip the pooled-headline recomputation")
    A = ap.parse_args()
    build(propagate=not A.no_propagate)
