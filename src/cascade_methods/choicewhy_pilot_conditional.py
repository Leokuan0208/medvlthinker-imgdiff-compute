#!/usr/bin/env python3
"""
choicewhy_pilot_conditional.py -- final addendum to the (choice)(why) Phase-1 gate.

Both reason-first arms only PARTLY complied (C never, C2 on ~10% of items) and the answer-first
arms only justify on some items, so the pooled numbers mix compliant and non-compliant items.
This script conditions on compliance:

  (a) C2 on the items where it genuinely put the letter LAST vs arm A on the SAME items
      (descriptive only -- the model chose when to reason, so this subset is self-selected).
  (b) B2 on the items where it genuinely produced a justification (>=3 words) vs arm A on the same
      items, same caveat.
  (c) THE DECISION-RELEVANT ONE: restricted to items that actually carry a justification, is the
      justification text discriminative of correctness, and is it INCREMENTAL over the letter-margin
      confidence the deployed gate already uses?  5-fold cross-fitted, with a label-shuffle null.

Appends to results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json under
"conditional_on_compliance".  Run from repo root.
"""
import json, math, os, re, string
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
RNG = np.random.default_rng(20260803)

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
BENCH_OF = {i: b for b in BENCHES for i in IDX[b]}
OPTS = {i: parse_opts(data[i]["options"]) for i in ALL}
GOLD = {i: str(data[i]["answer_label"]).strip().upper()[:1] for i in ALL}


def split_answer(text):
    """-> (letter, position in {'first','last','none'}, justification text)."""
    t = text.strip()
    m = LEAD.match(t)
    if m:
        rest = re.sub(r"^[\s).:,;\-—\]]+", "", t[m.end():])
        return m.group(1), "first", rest
    mt = TRAIL.search(t)
    if mt and mt.start() > 0:
        return mt.group(1), "last", t[:mt.start()]
    ls = LET.findall(t)
    return (ls[-1].upper() if ls else "?"), ("last" if ls else "none"), ""


def justification(i, arm):
    letter, pos, rest = split_answer(R[arm][i]["raw_output"])
    o = OPTS[i].get(letter, "")
    ot, toks = norm(o).split(), norm(rest).split()
    if ot and toks[:len(ot)] == ot:
        toks = toks[len(ot):]
    return letter, pos, " ".join(toks)


def ok(i, arm):
    letter, _, _ = split_answer(R[arm][i]["raw_output"])
    return int(letter == GOLD[i])


def paired_boot(idxs, a1, a2, n=10000):
    x = np.array([ok(i, a1) for i in idxs], float)
    y = np.array([ok(i, a2) for i in idxs], float)
    d = y - x
    bs = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"n": len(d), "acc_ref": float(x.mean()), "acc_arm": float(y.mean()),
            "delta": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


res = {}

# ---------------------------------------------------------------- (a) C2 genuinely reason-first
sel = [i for i in ALL if split_answer(R["C2_reason_first_forced"][i]["raw_output"])[1] == "last"
       and len(justification(i, "C2_reason_first_forced")[2].split()) >= 3]
res["a_C2_on_items_it_actually_reasoned_first"] = {
    "selection": "items where C2 put the letter LAST and wrote >=3 words of justification",
    "vs_arm_A_same_items": paired_boot(sel, "A_letter_only", "C2_reason_first_forced"),
    "per_benchmark_n": {b: sum(1 for i in sel if BENCH_OF[i] == b) for b in BENCHES},
    "CAVEAT": "SELF-SELECTED subset -- the model chose when to reason first, and it reasons on the "
              "items it finds hard, so this comparison is descriptive, NOT a causal estimate of the "
              "ordering effect.",
}

# ---------------------------------------------------------------- (b) B2 genuinely justified
selb = [i for i in ALL if len(justification(i, "B2_answer_first_forced")[2].split()) >= 3]
res["b_B2_on_items_it_actually_justified"] = {
    "selection": "items where B2 wrote >=3 words of justification after the letter",
    "vs_arm_A_same_items": paired_boot(selb, "A_letter_only", "B2_answer_first_forced"),
    "per_benchmark_n": {b: sum(1 for i in selb if BENCH_OF[i] == b) for b in BENCHES},
    "CAVEAT": "self-selected in the same way; descriptive.",
}

# ---------------------------------------------------------------- (c) discriminativeness where text exists
def margin(rec):
    lp = rec.get("opt_logprobs") or {}
    if len(lp) < 2:
        return 0.0
    v = sorted(lp.values(), reverse=True)
    return math.exp(v[0]) - math.exp(v[1])


def cv_auroc(texts, marg, y, use_text, use_margin, seed=0, shuffle=False):
    y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 60:
        return None
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        ftr, fte = [], []
        if use_text:
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
            try:
                ftr.append(v.fit_transform([texts[i] for i in tr]).toarray())
                fte.append(v.transform([texts[i] for i in te]).toarray())
            except ValueError:
                return None
        if use_margin:
            ftr.append(np.array([[marg[i]] for i in tr])); fte.append(np.array([[marg[i]] for i in te]))
        m = LogisticRegression(max_iter=3000).fit(np.hstack(ftr), y[tr])
        pred[te] = m.predict_proba(np.hstack(fte))[:, 1]
    return float(roc_auc_score(y, pred))


disc = {}
for arm in ["B_answer_first", "B2_answer_first_forced"]:
    disc[arm] = {}
    for b in BENCHES + ["POOLED"]:
        idxs = ALL if b == "POOLED" else IDX[b]
        keep = [i for i in idxs if len(justification(i, arm)[2].split()) >= 3]
        if len(keep) < 60:
            disc[arm][b] = {"n_with_justification": len(keep), "note": "too few to model"}
            continue
        txt = [justification(i, arm)[2] for i in keep]
        mg = [margin(R[arm][i]) for i in keep]
        y = [ok(i, arm) for i in keep]
        disc[arm][b] = {
            "n_with_justification": len(keep),
            "coverage_of_benchmark": len(keep) / len(idxs),
            "accuracy_on_these": float(np.mean(y)),
            "auroc_margin_only": (float(roc_auc_score(y, mg)) if len(set(y)) > 1 else None),
            "auroc_text_only_cv": cv_auroc(txt, mg, y, True, False),
            "auroc_margin_plus_text_cv": cv_auroc(txt, mg, y, True, True),
            "auroc_text_only_cv_shuffled_null": cv_auroc(txt, mg, y, True, False, seed=1, shuffle=True),
        }
res["c_discriminativeness_restricted_to_items_that_have_a_justification"] = {
    "why": "the fair test of 'can a verifier grade this': only items that actually carry a "
           "justification, and does the text beat / add to the letter-margin confidence.",
    "per_arm": disc,
}

d = json.load(open(OUT))
d["conditional_on_compliance"] = res
json.dump(d, open(OUT, "w"), indent=2)
print(f"appended conditional_on_compliance -> {OUT}\n")

for k in ("a_C2_on_items_it_actually_reasoned_first", "b_B2_on_items_it_actually_justified"):
    v = res[k]["vs_arm_A_same_items"]
    print(f"{k}\n   n={v['n']}  A={v['acc_ref']:.4f}  arm={v['acc_arm']:.4f}  "
          f"delta={v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
    print(f"   per-benchmark n: {res[k]['per_benchmark_n']}")

for arm, tbl in disc.items():
    print(f"\n=== (c) {arm}: items WITH a justification ===")
    print(f"{'bench':<24}{'n':>6}{'cov':>7}{'acc':>7}{'margin':>9}{'text':>9}{'m+t':>8}{'null':>8}")
    for b, r in tbl.items():
        if "auroc_margin_only" not in r:
            print(f"{b:<24}{r['n_with_justification']:>6}   (too few)")
            continue
        f = lambda x: f"{x:.3f}" if x is not None else "  NA "
        print(f"{b:<24}{r['n_with_justification']:>6}{r['coverage_of_benchmark']:>7.2f}"
              f"{r['accuracy_on_these']:>7.3f}{f(r['auroc_margin_only']):>9}{f(r['auroc_text_only_cv']):>9}"
              f"{f(r['auroc_margin_plus_text_cv']):>8}{f(r['auroc_text_only_cv_shuffled_null']):>8}")
