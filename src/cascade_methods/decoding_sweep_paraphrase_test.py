#!/usr/bin/env python3
"""decoding_sweep_paraphrase_test.py -- WHY repetition_penalty wins under the judge and loses under EM.

MECHANISM UNDER TEST. vLLM applies repetition_penalty to tokens present in the PROMPT as well as the
output (vllm/model_executor/layers/utils.py: apply_penalties -> apply_repetition_penalties(logits,
prompt_mask, output_mask, ...)). The prompt contains the question. In these three open VQA sets the gold
answer very often REUSES a word of the question ('Which lung is affected?' -> 'right lung'). So
repetition_penalty > 1 should push the model away from quoting the question and toward SYNONYMS.

That predicts exactly the observed sign flip:
  * normalised exact match is LEXICAL -- a synonym scores 0                      -> EM falls
  * the 32B judge is SEMANTIC -- a synonym scores 1                             -> judge holds/rises
so the setting looks like a coverage WIN under the judge and a LOSS under EM without the model having
become any better at the task.

FALSIFIABLE PREDICTIONS, all measured here, all needing no GPU:
  P1  rp11 lowers token overlap between the PREDICTION and the QUESTION   (it stops quoting)
  P2  rp11 lowers token overlap between the PREDICTION and the GOLD       (it stops matching lexically)
  P3  the drop in P2 is concentrated on items whose GOLD SHARES a word with the QUESTION
      -- the sub-population the mechanism can act on. On golds that share NO question word there is
      nothing to penalise, so the effect should be much smaller or absent.
  P4  the judge-minus-EM coverage gap grows with the same stratification.
If P3 fails -- i.e. the effect is the same on both strata -- the prompt-token mechanism is NOT what is
happening and the explanation must be plain verbosity instead.

Outputs results/cascade_methods/artifacts/_decoding_sweep_paraphrase_test.json
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G                       # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (                   # noqa: E402
    load_judge, load_pool, boot, DS, SWEEP)

ap = argparse.ArgumentParser()
ap.add_argument("--control", default="T07")
ap.add_argument("--settings", default="T03,T05,T07,T10,T13,minp005,minp01,rp105,rp11,"
                                      "topk20,topk50,topp09,topp095,T13minp010")
ap.add_argument("--out", default="results/cascade_methods/artifacts/_decoding_sweep_paraphrase_test.json")
A = ap.parse_args()

lab, ref = load_judge(), G.load_items()
STOP = set("the a an is are of in on at this that image picture what which where "
           "does do did was were be been has have had it its to for from with and or "
           "how many much there here shown seen show see".split())


def toks(s):
    return set(w for w in G.norm(str(s)).split() if w and w not in STOP)


def seed_tags(st):
    return [f"{st}_s{s}" for s in range(8)
            if os.path.exists(os.path.join(SWEEP, f"ckpt_{DS[0]}_{st}_s{s}.jsonl"))]


def fully_judged(pool):
    """A pool missing judge labels must be refused. lab.get(..., 0) would score every unlabelled
    slot as WRONG and silently deflate that setting's judge oracle."""
    for it in ref:
        for a in pool[(it["ds"], it["idx"])]["preds"]:
            if lab.get((it["ds"], it["idx"], G.norm(a))) is None:
                return False
    return True


# ---- the stratifier: does the GOLD share a content word with the QUESTION? (fixed across settings) --
strat = {}
base = load_pool(f"{A.control}_s0", strict=False)
if base is None:
    sys.exit("control pool missing")
for it in ref:
    r = base[(it["ds"], it["idx"])]
    gq, gg = toks(r["question"]), toks(r.get("gold", ""))
    strat[(it["ds"], it["idx"])] = bool(gq & gg)
SHARED = np.array([strat[(it["ds"], it["idx"])] for it in ref], dtype=bool)


def measure(tag):
    pool = load_pool(tag, strict=False)
    if pool is None:
        return None
    if not fully_judged(pool):
        print(f"  [unjudged, excluded] {tag}")
        return None
    pq, pg, pj, jud, em, judonly = [], [], [], [], [], []
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        tq, tg = toks(r["question"]), toks(r.get("gold", ""))
        a_pq, a_pg, a_pj = [], [], []
        for a in r["preds"]:
            ta = toks(a)
            # NOTE both of these are normalised by the PREDICTION length, not the gold length.
            # Normalising by |gold| (an earlier version) is confounded: a longer answer trivially
            # recalls more gold tokens, so rp11 -- which is more verbose -- scored higher for free.
            a_pq.append(len(ta & tq) / max(len(ta), 1))
            a_pg.append(len(ta & tg) / max(len(ta), 1))
            a_pj.append(len(ta & tg) / max(len(ta | tg), 1))       # length-fair Jaccard
        pq.append(np.mean(a_pq)); pg.append(np.mean(a_pg)); pj.append(np.mean(a_pj))
        j = int(any(lab.get((it["ds"], it["idx"], G.norm(a)), 0) for a in r["preds"]))
        e = int(any(r["oks_em"]))
        jud.append(j); em.append(e); judonly.append(j - e)
    return {k: np.array(v, float) for k, v in
            dict(pred_question_overlap=pq, pred_gold_overlap=pg, pred_gold_jaccard=pj,
                 judge_oracle=jud, em_oracle=em, judge_only=judonly).items()}


data = {}
for st in A.settings.split(","):
    per = [measure(t) for t in seed_tags(st)]
    per = [p for p in per if p is not None]
    if per:
        data[st] = {k: np.mean([p[k] for p in per], axis=0) for k in per[0]}

out = {"title": "Does repetition_penalty win under the judge by PARAPHRASING instead of quoting?",
       "mechanism": __doc__.strip().split("MECHANISM UNDER TEST.")[1].split("FALSIFIABLE")[0].strip(),
       "stratifier": "GOLD shares >=1 content word with the QUESTION (stopword-filtered, computed once "
                     f"on the control pool {A.control}_s0 so it is identical across settings)",
       "n_gold_shares_question_word": int(SHARED.sum()),
       "POWER_WARNING_P3": "the gold-shares-a-question-word stratum holds only 92 of 2345 items, so each item moves the EM-oracle by 1/92 = 0.0109 and the P3 contrast is UNDERPOWERED. P3 is reported but is NOT evidence either way.",
       "n_gold_shares_no_question_word": int((~SHARED).sum()),
       "per_setting": {}}

for st, d in data.items():
    out["per_setting"][st] = {
        "pred_question_overlap": float(d["pred_question_overlap"].mean()),
        "pred_gold_overlap": float(d["pred_gold_overlap"].mean()),
        "pred_gold_jaccard": float(d["pred_gold_jaccard"].mean()),
        "judge_oracle@8": float(d["judge_oracle"].mean()),
        "em_oracle@8": float(d["em_oracle"].mean()),
        "judge_only_coverage": float(d["judge_only"].mean()),
        "by_stratum": {
            "gold_shares_question_word": {
                "n": int(SHARED.sum()),
                "pred_gold_overlap": float(d["pred_gold_overlap"][SHARED].mean()),
                "judge_oracle@8": float(d["judge_oracle"][SHARED].mean()),
                "em_oracle@8": float(d["em_oracle"][SHARED].mean()),
                "judge_only_coverage": float(d["judge_only"][SHARED].mean())},
            "gold_shares_none": {
                "n": int((~SHARED).sum()),
                "pred_gold_overlap": float(d["pred_gold_overlap"][~SHARED].mean()),
                "judge_oracle@8": float(d["judge_oracle"][~SHARED].mean()),
                "em_oracle@8": float(d["em_oracle"][~SHARED].mean()),
                "judge_only_coverage": float(d["judge_only"][~SHARED].mean())}}}

C = A.control
if C in data:
    c = data[C]
    out["deltas_vs_control"] = {}
    for st, d in data.items():
        if st == C:
            continue
        blk = {"P1_pred_question_overlap_delta": boot(d["pred_question_overlap"], c["pred_question_overlap"]),
               "P2_pred_gold_overlap_delta": boot(d["pred_gold_overlap"], c["pred_gold_overlap"]),
               "P2b_pred_gold_JACCARD_delta": boot(d["pred_gold_jaccard"], c["pred_gold_jaccard"]),
               "P3_em_oracle_delta_GOLD_SHARES_Q": boot(d["em_oracle"], c["em_oracle"], mask=SHARED),
               "P3_em_oracle_delta_GOLD_SHARES_NONE": boot(d["em_oracle"], c["em_oracle"], mask=~SHARED),
               "P4_judge_only_delta_GOLD_SHARES_Q": boot(d["judge_only"], c["judge_only"], mask=SHARED),
               "P4_judge_only_delta_GOLD_SHARES_NONE": boot(d["judge_only"], c["judge_only"], mask=~SHARED)}
        out["deltas_vs_control"][st] = blk

os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1, default=float)
print(f"wrote {A.out}\n")
print(f"{'setting':12s} {'pred~Q':>8s} {'pred~gold':>10s} {'judge@8':>8s} {'em@8':>8s} {'judgeonly':>10s}")
for st, b in out["per_setting"].items():
    print(f"{st:12s} {b['pred_question_overlap']:8.4f} {b['pred_gold_overlap']:10.4f} "
          f"{b['judge_oracle@8']:8.4f} {b['em_oracle@8']:8.4f} {b['judge_only_coverage']:10.4f}"
          f"  jacc {b['pred_gold_jaccard']:.4f}")
if "deltas_vs_control" in out:
    print(f"\nvs {C}  (P3: EM-oracle delta by stratum -- mechanism acts only where gold shares a question word)")
    for st, b in out["deltas_vs_control"].items():
        q, n = b["P3_em_oracle_delta_GOLD_SHARES_Q"], b["P3_em_oracle_delta_GOLD_SHARES_NONE"]
        p1, p2 = b["P1_pred_question_overlap_delta"], b["P2_pred_gold_overlap_delta"]
        print(f"  {st:12s} P1 {p1['delta']:+.4f} {p1['verdict']:5s} | P2 {p2['delta']:+.4f} {p2['verdict']:5s} "
              f"| P3 sharesQ {q['delta']:+.4f} {q['verdict']:5s}  none {n['delta']:+.4f} {n['verdict']:5s}")
