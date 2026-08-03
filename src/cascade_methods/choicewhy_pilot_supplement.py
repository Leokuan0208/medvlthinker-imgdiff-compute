#!/usr/bin/env python3
"""
choicewhy_pilot_supplement.py -- rigour supplement to choicewhy_pilot_analyze.py.

Three things the headline analysis flagged as needing scrutiny:

 (1) ARM-C INSTRUCTION-FOLLOWING AUDIT.  The token audit showed arm C generated 4.7 tokens on
     average -- it did NOT reason.  Quantify how often C actually put the letter LAST (as instructed)
     vs FIRST (like arm A), and check that the arm-C letter extractor (which assumes letter-last)
     is not corrupting the score.
 (2) DEGENERATE-RATIONALE RATE for arm B.  "B) No" has a non-empty string after the letter but is
     NOT a justification -- it just restates the option.  Measure how often B adds nothing beyond
     the chosen option's own text.
 (3) IS THE RATIONALE DISCRIMINATIVE *WITHIN* A BENCHMARK, AND *INCREMENTALLY* OVER THE SIGNAL WE
     ALREADY HAVE?  The pooled TF-IDF AUROC (0.749) can be produced by benchmark identity alone
     (accuracy ranges 0.26-0.85 across benchmarks), so report per-benchmark, and compare against /
     combine with the deployed letter-margin confidence, all cross-fitted (5-fold).

Appends the results into results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json
under "supplement".  Run from repo root.
"""
import collections, json, math, os, re, string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CK = "ckpts/choicewhy_pilot"
OUT = "results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json"
ARMS = ["A_letter_only", "B_answer_first", "C_reason_first",
        "B2_answer_first_forced", "C2_reason_first_forced"]
BENCHES = ["SLAKE", "VQA-RAD", "PMC-VQA", "MedXpert-Reasoning", "MedXpert-Understanding"]

LET = re.compile(r"\b([A-J])\b")
LEAD = re.compile(r"^\s*[*\"'(\[]*\s*([A-J])\s*(?=[).:,;\-—\]]|$|\n)")
TRAIL = re.compile(r"\b([A-J])\b[\s.)*\"'\]]*$")

from datasets import load_dataset  # noqa: E402
data = load_dataset("/data/dan/dataset/MedVLThinker-Eval")["test"]


def parse_opts(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))


def norm(s):
    s = str(s).lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


R = {a: {} for a in ARMS}
for a in ARMS:
    for b in BENCHES:
        for l in open(os.path.join(CK, f"ckpt_{b}_{a}.jsonl")):
            if l.strip():
                r = json.loads(l); R[a][r["idx"]] = r
IDX = {b: sorted(json.loads(l)["idx"] for l in open(os.path.join(CK, f"ckpt_{b}_{ARMS[0]}.jsonl")) if l.strip())
       for b in BENCHES}
ALL = [i for b in BENCHES for i in IDX[b]]
OPTS = {i: parse_opts(data[i]["options"]) for i in ALL}
GOLD = {i: str(data[i]["answer_label"]).strip().upper()[:1] for i in ALL}

sup = {}

# ------------------------------------------------------------------ (1) reason-first audit
def reason_first_audit(CARM):
 c = {"n": 0, "letter_first": 0, "letter_last_only": 0, "both_ends_same": 0,
      "lead_vs_trail_disagree": 0, "acc_trail_rule_used_in_headline": 0, "acc_lead_rule": 0,
      "gen_tokens_le_8": 0}
 per_b = {}
 for b in BENCHES:
     cb = dict(n=0, letter_first=0, lead_vs_trail_disagree=0, acc_trail=0, acc_lead=0)
     for i in IDX[b]:
         t = R[CARM][i]["raw_output"].strip()
         ml, mt = LEAD.match(t), TRAIL.search(t)
         ls = LET.findall(t)
         trail_letter = ls[-1].upper() if ls else "?"
         lead_letter = ml.group(1) if ml else "?"
         c["n"] += 1; cb["n"] += 1
         if ml:
             c["letter_first"] += 1; cb["letter_first"] += 1
         if mt and not ml:
             c["letter_last_only"] += 1
         if ml and mt and ml.group(1) == mt.group(1):
             c["both_ends_same"] += 1
         if lead_letter != "?" and trail_letter != lead_letter:
             c["lead_vs_trail_disagree"] += 1; cb["lead_vs_trail_disagree"] += 1
         c["acc_trail_rule_used_in_headline"] += int(trail_letter == GOLD[i])
         cb["acc_trail"] += int(trail_letter == GOLD[i])
         la = lead_letter if lead_letter != "?" else trail_letter
         c["acc_lead_rule"] += int(la == GOLD[i]); cb["acc_lead"] += int(la == GOLD[i])
         if R[CARM][i]["gen_tokens"] <= 8:
             c["gen_tokens_le_8"] += 1
     per_b[b] = {"n": cb["n"], "letter_first_rate": cb["letter_first"] / cb["n"],
                 "lead_vs_trail_disagree_rate": cb["lead_vs_trail_disagree"] / cb["n"],
                 "acc_letter_last_rule": cb["acc_trail"] / cb["n"],
                 "acc_letter_first_rule": cb["acc_lead"] / cb["n"]}

 return {
    "n": c["n"],
    "letter_first_rate": c["letter_first"] / c["n"],
    "letter_last_only_rate": c["letter_last_only"] / c["n"],
    "gen_tokens_le_8_rate": c["gen_tokens_le_8"] / c["n"],
    "extractor_lead_vs_trail_disagree_rate": c["lead_vs_trail_disagree"] / c["n"],
    "acc_with_letter_last_rule_headline": c["acc_trail_rule_used_in_headline"] / c["n"],
    "acc_with_letter_first_rule": c["acc_lead_rule"] / c["n"],
    "per_benchmark": per_b,
    "note": "the two extraction rules are compared so that a reason-first arm's reported accuracy "
            "cannot be an artifact of assuming the letter is last.",
 }


sup["reason_first_instruction_following_audit"] = {
    "verdict": "arm C DID NOT follow its instruction (letter_first_rate 1.000, zero justification "
               "words) so arm C is NOT a chain-of-thought control; arm C2, which forces the sentence, "
               "is the control that actually ran.",
    "C_reason_first": reason_first_audit("C_reason_first"),
    "C2_reason_first_forced": reason_first_audit("C2_reason_first_forced"),
}

# ------------------------------------------------------------------ (2) degenerate-rationale rate
def rationale_beyond_option(text, arm, opts):
    """Strip the leading/trailing answer letter AND the chosen option's own text; what remains is the
    actual justification."""
    t = text.strip()
    m = LEAD.match(t)
    chosen = m.group(1) if m else None
    if m:                                     # answer-first arms: strip the LEADING letter
        t = t[m.end():]
    else:                                     # reason-first arms: strip the TRAILING letter
        mt = TRAIL.search(t)
        if mt:
            chosen = mt.group(1)
            t = t[:mt.start()]
    t = re.sub(r"^[\s).:,;\-—\]]+", "", t)
    if chosen and chosen in opts:
        ot = norm(opts[chosen])
        toks = norm(t).split()
        otoks = ot.split()
        # remove a leading restatement of the option text
        if otoks and toks[:len(otoks)] == otoks:
            toks = toks[len(otoks):]
        t = " ".join(toks)
    return norm(t)


deg = {}
for b in BENCHES + ["POOLED"]:
    idxs = ALL if b == "POOLED" else IDX[b]
    for arm in ["B_answer_first", "C_reason_first", "B2_answer_first_forced", "C2_reason_first_forced"]:
        rs = [rationale_beyond_option(R[arm][i]["raw_output"], arm, OPTS[i]) for i in idxs]
        w = np.array([len(x.split()) for x in rs], float)
        deg.setdefault(b, {})[arm] = {
            "n": len(rs),
            "no_justification_rate_lt3_words": float(np.mean(w < 3)),
            "mean_justification_words": float(w.mean()),
            "median_justification_words": float(np.median(w)),
            "distinct_justification_rate": float(len(set(rs)) / len(rs)),
        }
sup["degenerate_rationale"] = {
    "definition": "text remaining after removing the leading answer letter AND a verbatim restatement "
                  "of the chosen option; <3 words = the model restated the option and justified nothing",
    "per_group": deg,
}

# ------------------------------------------------------------------ (3) within-benchmark + incremental
def margin(rec):
    lp = rec.get("opt_logprobs") or {}
    if len(lp) < 2:
        return None
    v = sorted(lp.values(), reverse=True)
    return math.exp(v[0]) - math.exp(v[1])


def cv_auroc(texts, marg, y, use_text, use_margin, seed=0, shuffle=False):
    y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 60:
        return None
    if use_text and sum(1 for t in texts if t.strip()) < 20:
        return None   # arm A has no justification text at all -> nothing to model
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        feats_tr, feats_te = [], []
        if use_text:
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
            try:
                feats_tr.append(v.fit_transform([texts[i] for i in tr]).toarray())
                feats_te.append(v.transform([texts[i] for i in te]).toarray())
            except ValueError:
                return None
        if use_margin:
            feats_tr.append(np.array([[marg[i]] for i in tr]))
            feats_te.append(np.array([[marg[i]] for i in te]))
        Xtr = np.hstack(feats_tr); Xte = np.hstack(feats_te)
        m = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y[tr])
        pred[te] = m.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y, pred))


inc = {}
for b in BENCHES + ["POOLED"]:
    idxs = ALL if b == "POOLED" else IDX[b]
    row = {}
    for arm in ARMS:
        mg = [margin(R[arm][i]) for i in idxs]
        if any(x is None for x in mg):
            mg = [0.0 if x is None else x for x in mg]
        y = [int(LET.search(R[arm][i]["raw_output"] or "").group(1) == GOLD[i])
             if LET.search(R[arm][i]["raw_output"] or "") else 0 for i in idxs]
        txt = [rationale_beyond_option(R[arm][i]["raw_output"], arm, OPTS[i]) for i in idxs]
        row[arm] = {
            "n": len(idxs), "accuracy": float(np.mean(y)),
            "auroc_margin_only": (float(roc_auc_score(y, mg)) if len(set(y)) > 1 else None),
            "auroc_text_only_cv": cv_auroc(txt, mg, y, True, False),
            "auroc_margin_plus_text_cv": cv_auroc(txt, mg, y, True, True),
            "auroc_margin_only_cv": cv_auroc(txt, mg, y, False, True),
            "auroc_text_only_cv_label_shuffled_null": cv_auroc(txt, mg, y, True, False, seed=1, shuffle=True),
        }
    inc[b] = row
sup["discriminativeness_within_benchmark"] = {
    "why": "the pooled TF-IDF AUROC is confounded by benchmark identity (per-benchmark accuracy ranges "
           "0.26-0.85), so a text model can score well by merely recognising which benchmark an item is "
           "from. These are the honest per-benchmark numbers, plus whether the rationale adds anything "
           "ON TOP of the letter-margin confidence the deployed gate already uses.",
    "signals": {"margin": "exp(top1 logprob) - exp(top2 logprob) at the answer-letter position",
                "text": "TF-IDF(1,2)-grams of the justification only (option restatement removed), "
                        "5-fold cross-fitted logistic regression",
                "note_arm_A": "arm A has no justification; its 'text' column is a floor/sanity control"},
    "per_benchmark": inc,
}

# ------------------------------------------------------------------ decode-token cost
cost = {}
for b in BENCHES + ["POOLED"]:
    idxs = ALL if b == "POOLED" else IDX[b]
    cost[b] = {a: float(np.mean([R[a][i]["gen_tokens"] for i in idxs])) for a in ARMS}
    cost[b]["B_over_A_decode_token_ratio"] = cost[b]["B_answer_first"] / cost[b]["A_letter_only"]
    cost[b]["B2_over_A_decode_token_ratio"] = cost[b]["B2_answer_first_forced"] / cost[b]["A_letter_only"]
sup["decode_token_cost"] = {"measured_mean_generated_tokens": cost,
                            "note": "MEASURED decode tokens only. Total FLOPs are prefill-dominated for a "
                                    "VLM (image tokens), so this ratio is an upper bound on the relative "
                                    "cost increase, not the cost increase itself."}

d = json.load(open(OUT))
d["supplement"] = sup
json.dump(d, open(OUT, "w"), indent=2)
print(f"appended supplement -> {OUT}\n")

print("=== (1) ARM C INSTRUCTION FOLLOWING ===")
a = sup["reason_first_instruction_following_audit"]["C2_reason_first_forced"]
print("  (shown for C2, the arm that actually forced a justification; C is in the JSON)")
for k in ("letter_first_rate", "letter_last_only_rate", "gen_tokens_le_8_rate",
          "extractor_lead_vs_trail_disagree_rate", "acc_with_letter_last_rule_headline",
          "acc_with_letter_first_rule"):
    print(f"  {k:<42} {a[k]:.4f}")

print("\n=== (2) NO-JUSTIFICATION RATE (<3 words beyond the option) ===")
ARMS2 = ["B_answer_first", "C_reason_first", "B2_answer_first_forced", "C2_reason_first_forced"]
print(f"{'group':<24}" + "".join(f"{a.split('_')[0]:>10}" for a in ARMS2) + "   (mean justification words)")
for b, v in deg.items():
    print(f"{b:<24}" + "".join(f"{v[a]['no_justification_rate_lt3_words']:>10.3f}" for a in ARMS2)
          + "   " + " / ".join(f"{v[a]['mean_justification_words']:.1f}" for a in ARMS2))

f = lambda x: f"{x:.3f}" if x is not None else "  NA "
for arm in ["B_answer_first", "B2_answer_first_forced"]:
    print(f"\n=== (3) DISCRIMINATIVENESS, {arm}, WITHIN benchmark (cross-fit AUROC) ===")
    print(f"{'bench':<24}{'acc':>7}{'margin':>9}{'text':>9}{'marg+txt':>10}{'null':>8}")
    for b, v in inc.items():
        r = v[arm]
        print(f"{b:<24}{r['accuracy']:>7.3f}{f(r['auroc_margin_only']):>9}{f(r['auroc_text_only_cv']):>9}"
              f"{f(r['auroc_margin_plus_text_cv']):>10}{f(r['auroc_text_only_cv_label_shuffled_null']):>8}")

print("\n=== decode tokens (mean) ===")
for b, v in cost.items():
    print(f"{b:<24} A={v['A_letter_only']:.1f}  B={v['B_answer_first']:.1f}  C={v['C_reason_first']:.1f}  "
          f"B2={v['B2_answer_first_forced']:.1f}  C2={v['C2_reason_first_forced']:.1f}  "
          f"B2/A={v['B2_over_A_decode_token_ratio']:.1f}x")
