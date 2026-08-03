#!/usr/bin/env python3
"""
choicewhy_pilot_analyze.py -- PHASE 1 GATE analysis for the "(choice)(why)" program.

Reads the three-arm dumps written by src/labeling/run_choicewhy_pilot.py and answers, per benchmark
and pooled:

  1. ACCURACY per arm, paired on item idx, with paired-bootstrap 95% CIs for B-A and C-A.
     Scoring = the repo's exact-letter match against answer_label.  Reported three ways so a parse
     failure cannot masquerade as an accuracy loss:
       acc_strict       parse failure counts as wrong  (headline)
       acc_parsed_only  restricted to items where a letter was extracted
       acc_lenient      letter, else nearest-option-text fallback
  2. TOKEN AUDIT: mean/median/p90 generated tokens per arm; finish_reason=length rate.
  3. RATIONALE DISCRIMINATIVENESS (cheap proxy, no training): do correct-answer rationales differ
     from incorrect ones?  length AUROC, hedge-word rate, image-specific-detail rate, boilerplate
     rate (duplicate rationales), and a cross-fitted TF-IDF logistic-regression AUROC on the
     rationale TEXT ALONE, with a label-shuffle null.

Writes results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json.
Run from repo root:  python3 src/cascade_methods/choicewhy_pilot_analyze.py
"""
import argparse, collections, json, os, re, string
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", default="ckpts/choicewhy_pilot")
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json")
ap.add_argument("--boot", type=int, default=10000)
A = ap.parse_args()

ROOT = "/data/dan/dataset/MedVLThinker-Eval"
ARMS = ["A_letter_only", "B_answer_first", "C_reason_first",
        "B2_answer_first_forced", "C2_reason_first_forced"]
BENCHES = ["SLAKE", "VQA-RAD", "PMC-VQA", "MedXpert-Reasoning", "MedXpert-Understanding"]
RNG = np.random.default_rng(20260803)

SYS = {  # copied verbatim from run_choicewhy_pilot.py so the report is self-contained
    "A_letter_only": "Answer with only the correct option letter (e.g. 'A'). Do not explain.",
    "B_answer_first": ("Answer with the correct option letter first (e.g. 'A'), then explain in one or two short "
                       "sentences why that option is correct. Example: \"A. The mass is in the left lower lobe.\""),
    "C_reason_first": ("Explain in one or two short sentences why an option is correct, then answer with the "
                       "correct option letter last (e.g. 'A'). Example: \"The mass is in the left lower lobe. A.\""),
    "B2_answer_first_forced": ("Answer with the correct option letter first (e.g. 'A'), then, in exactly one sentence, state "
                               "the specific finding in the image that makes that option correct. Always give the sentence, "
                               "even when the answer is obvious. Example: \"A. The mass is in the left lower lobe.\""),
    "C2_reason_first_forced": ("First, in exactly one sentence, state the specific finding in the image that makes an option "
                               "correct, then answer with that option letter last (e.g. 'A'). Always give the sentence, "
                               "even when the answer is obvious. Example: \"The mass is in the left lower lobe. A.\""),
}

# ------------------------------------------------------------------ letter extraction
LET_FIRST = re.compile(r"\b([A-J])\b")                       # the repo's baseline rule (first standalone letter)
LEAD = re.compile(r"^\s*[*\"'(\[]*\s*([A-J])\s*(?=[).:,;\-\u2014\]]|$|\n)")
BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-J])")
MARK = re.compile(r"(?:answer|option|choice|correct|select)(?:\s+is)?\s*(?:letter)?\s*[:=]?\s*[*\"'(\[]*\s*([A-J])\b", re.I)


def extract(text, arm):
    """Return (letter, parse_ok, rule). Arm-appropriate rules; every rule hit is counted and reported."""
    t = text.strip()
    if "letter_only" in arm:
        m = LET_FIRST.search(t)
        return (m.group(1), 1, "first_standalone") if m else ("?", 0, "none")
    if "answer_first" in arm:
        m = LEAD.match(t)
        if m:
            return m.group(1), 1, "lead"
        b = BOXED.findall(t)
        if b:
            return b[-1].upper(), 1, "boxed"
        k = list(MARK.finditer(t))
        if k:
            return k[-1].group(1).upper(), 1, "marker"
        m = LET_FIRST.search(t)
        return (m.group(1), 1, "first_standalone") if m else ("?", 0, "none")
    # C_reason_first: the letter is meant to be LAST
    b = BOXED.findall(t)
    if b:
        return b[-1].upper(), 1, "boxed"
    k = list(MARK.finditer(t))
    if k:
        return k[-1].group(1).upper(), 1, "marker"
    ls = LET_FIRST.findall(t)
    if ls:
        return ls[-1].upper(), 1, "last_standalone"
    return "?", 0, "none"


def norm(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def lenient_letter(text, options):
    """Fallback when no letter parsed: nearest option by normalised token overlap / containment."""
    pn = norm(text)
    if not pn:
        return "?"
    best, bl = -1, "?"
    for L, o in options.items():
        on = norm(o)
        if not on:
            continue
        sc = len(set(pn.split()) & set(on.split())) + (2 if on in pn else 0) + (3 if pn == on else 0)
        if sc > best:
            best, bl = sc, L
    return bl if best > 0 else "?"


def strip_letter(text, arm):
    """Remove the answer token so the rationale text is graded on its own."""
    t = text.strip()
    if "answer_first" in arm:
        t = LEAD.sub("", t, count=1)
    else:
        t = re.sub(r"[\s.,;:*\"'()\[\]]*\b[A-J]\b[\s.)*\"'\]]*$", "", t)
    t = re.sub(r"\\boxed\{[^}]*\}", " ", t)
    return t.strip(" .,:;-\n\t")


# ------------------------------------------------------------------ load
from datasets import load_dataset  # noqa: E402
data = load_dataset(ROOT)["test"]


def parse_opts(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))


rows = {}   # arm -> bench -> idx -> record
for arm in ARMS:
    rows[arm] = {}
    for b in BENCHES:
        p = os.path.join(A.ckpt_dir, f"ckpt_{b}_{arm}.jsonl")
        d = {}
        if os.path.exists(p):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    d[r["idx"]] = r
        rows[arm][b] = d

# common idx per bench (paired analysis)
common = {b: sorted(set.intersection(*[set(rows[a][b]) for a in ARMS])) for b in BENCHES}
print({b: (len(common[b]), [len(rows[a][b]) for a in ARMS]) for b in BENCHES})

OPTS = {}
for b in BENCHES:
    for i in common[b]:
        OPTS[i] = parse_opts(data[i]["options"])

# ------------------------------------------------------------------ score
scored = {a: {} for a in ARMS}   # arm -> idx -> dict
for arm in ARMS:
    for b in BENCHES:
        for i in common[b]:
            r = rows[arm][b][i]
            letter, pk, rule = extract(r["raw_output"], arm)
            lz = letter if pk else lenient_letter(r["raw_output"], OPTS[i])
            scored[arm][i] = {
                "bench": b, "gold": r["gold"], "letter": letter, "parse_ok": pk, "rule": rule,
                "ok": int(pk and letter == r["gold"]),
                "ok_lenient": int(lz == r["gold"]),
                "gen_tokens": r["gen_tokens"], "finish": r.get("finish"),
                "rationale": strip_letter(r["raw_output"], arm), "raw": r["raw_output"],
            }


def acc(arm, idxs, key="ok"):
    return float(np.mean([scored[arm][i][key] for i in idxs])) if idxs else float("nan")


def paired_boot(idxs, a1, a2, key="ok", n=A.boot):
    x = np.array([scored[a1][i][key] for i in idxs], float)
    y = np.array([scored[a2][i][key] for i in idxs], float)
    d = y - x
    if len(d) == 0:
        return None
    bs = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_two_sided_sign": float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))}


# ------------------------------------------------------------------ rationale features
HEDGE = re.compile(r"\b(may|might|possibly|possible|appears?|appear to|likely|unlikely|seems?|suggest\w*|"
                   r"could|cannot|can't|unclear|difficult|uncertain|probably|consistent with|indicative of)\b", re.I)
DETAIL = re.compile(r"\b(left|right|upper|lower|anterior|posterior|lateral|medial|superior|inferior|"
                    r"lobe|lung|liver|kidney|brain|heart|spine|axial|sagittal|coronal|contrast|"
                    r"hyperintense|hypointense|opacity|lesion|mass|nodule|effusion|fracture|"
                    r"ventricle|cortex|artery|vein|stain|cell|nucleus|nuclei|tissue|margin)\b", re.I)


def rationale_stats(arm, idxs):
    rr = [scored[arm][i] for i in idxs]
    txt = [r["rationale"] for r in rr]
    okv = np.array([r["ok"] for r in rr], float)
    L = np.array([len(t.split()) for t in txt], float)
    H = np.array([len(HEDGE.findall(t)) / max(len(t.split()), 1) for t in txt], float)
    D = np.array([len(DETAIL.findall(t)) / max(len(t.split()), 1) for t in txt], float)

    def auroc(s, y):
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    cnt = collections.Counter(t.lower().strip() for t in txt)
    dup = sum(c for t, c in cnt.items() if c > 1 and t) / max(len(txt), 1)
    return {
        "n": len(rr),
        "mean_words": float(L.mean()), "median_words": float(np.median(L)),
        "empty_rationale_rate": float(np.mean([1.0 if not t.strip() else 0.0 for t in txt])),
        "distinct_rationale_rate": float(len(cnt) / max(len(txt), 1)),
        "duplicate_rationale_rate": float(dup),
        "words_correct": float(L[okv == 1].mean()) if (okv == 1).any() else None,
        "words_incorrect": float(L[okv == 0].mean()) if (okv == 0).any() else None,
        "hedge_rate_correct": float(H[okv == 1].mean()) if (okv == 1).any() else None,
        "hedge_rate_incorrect": float(H[okv == 0].mean()) if (okv == 0).any() else None,
        "detail_rate_correct": float(D[okv == 1].mean()) if (okv == 1).any() else None,
        "detail_rate_incorrect": float(D[okv == 0].mean()) if (okv == 0).any() else None,
        "auroc_length_vs_correct": auroc(L, okv),
        "auroc_hedge_vs_correct": auroc(-H, okv),
        "auroc_detail_vs_correct": auroc(D, okv),
    }


def tfidf_auroc(arm, idxs, seed=0, shuffle=False):
    """Cross-fitted (5-fold) TF-IDF + logistic-regression AUROC on rationale text ALONE.
    Cheap proxy for 'is there anything in the text a verifier could grade'. Not a verifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    txt = [scored[arm][i]["rationale"] for i in idxs]
    y = np.array([scored[arm][i]["ok"] for i in idxs])
    if len(set(y)) < 2 or len(y) < 50:
        return None
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(txt, y):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
        Xtr = v.fit_transform([txt[i] for i in tr]); Xte = v.transform([txt[i] for i in te])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        pred[te] = m.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y, pred))


# ------------------------------------------------------------------ assemble
out = {
    "question": "Does answering multiple choice as (choice)(why) preserve accuracy? PHASE 1 GATE.",
    "date": "2026-08-03",
    "model": "lingshu-medical-mllm/Lingshu-7B (cheap leg), greedy temperature=0, tp=1, fullres, max_tokens=320 for EVERY arm",
    "generator": "src/labeling/run_choicewhy_pilot.py",
    "analyzer": "src/cascade_methods/choicewhy_pilot_analyze.py",
    "dumps": A.ckpt_dir,
    "eval_set": "MedVLThinker-Eval (/data/dan/dataset/MedVLThinker-Eval); repo fixed_slice(seed=42) selection",
    "prompts": {"system_by_arm": SYS,
                "user_turn": "IDENTICAL across arms: images + question + '\\n' + 'K) option' lines "
                             "(byte-identical to src/labeling/run_32b_modes_vllm.py::build_prompt)"},
    "scoring": "exact letter match vs answer_label (the repo's MCQ scoring). acc_strict counts a parse "
               "failure as wrong; acc_parsed_only restricts to parsed items; acc_lenient adds a "
               "nearest-option-text fallback.",
    "item_set": {b: len(common[b]) for b in BENCHES},
    "arms": {}, "deltas": {}, "tokens": {}, "rationales": {}, "null_test": {}, "gate": {},
}

ALL = [i for b in BENCHES for i in common[b]]
GROUPS = {b: common[b] for b in BENCHES}
GROUPS["MedXpert-MM(both)"] = common["MedXpert-Reasoning"] + common["MedXpert-Understanding"]
GROUPS["POOLED-all-1488"] = ALL
GROUPS["POOLED-ex-MedXpert"] = common["SLAKE"] + common["VQA-RAD"] + common["PMC-VQA"]

for g, idxs in GROUPS.items():
    out["arms"][g] = {a: {"n": len(idxs),
                          "acc_strict": acc(a, idxs, "ok"),
                          "acc_lenient": acc(a, idxs, "ok_lenient"),
                          "parse_ok_rate": acc(a, idxs, "parse_ok"),
                          "acc_parsed_only": (float(np.mean([scored[a][i]["ok"] for i in idxs
                                                             if scored[a][i]["parse_ok"]]))
                                              if any(scored[a][i]["parse_ok"] for i in idxs) else None),
                          "rule_hits": dict(collections.Counter(scored[a][i]["rule"] for i in idxs)),
                          } for a in ARMS}
    dd = {f"{a}_minus_A_strict": paired_boot(idxs, ARMS[0], a, "ok") for a in ARMS[1:]}
    dd.update({f"{a}_minus_A_lenient": paired_boot(idxs, ARMS[0], a, "ok_lenient") for a in ARMS[1:]})
    if "B2_answer_first_forced" in ARMS and "C2_reason_first_forced" in ARMS:
        dd["ORDERING_CONTROL_B2_minus_C2_strict"] = paired_boot(
            idxs, "C2_reason_first_forced", "B2_answer_first_forced", "ok")
    dd["B_minus_A_strict"] = dd["B_answer_first_minus_A_strict"]
    dd["C_minus_A_strict"] = dd["C_reason_first_minus_A_strict"]
    out["deltas"][g] = dd

for g, idxs in GROUPS.items():
    out["tokens"][g] = {}
    for a in ARMS:
        t = np.array([scored[a][i]["gen_tokens"] for i in idxs], float)
        fin = collections.Counter(scored[a][i]["finish"] for i in idxs)
        out["tokens"][g][a] = {"mean": float(t.mean()), "median": float(np.median(t)),
                               "p90": float(np.percentile(t, 90)), "max": float(t.max()),
                               "finish_reason": dict(fin),
                               "truncated_rate": float(fin.get("length", 0) / len(idxs))}

for g, idxs in GROUPS.items():
    out["rationales"][g] = {}
    for a in ARMS[1:]:
        st = rationale_stats(a, idxs)
        st["tfidf_cv_auroc_text_only"] = tfidf_auroc(a, idxs)
        st["tfidf_cv_auroc_label_shuffled_null"] = tfidf_auroc(a, idxs, seed=1, shuffle=True)
        out["rationales"][g][a] = st
    out["rationales"][g]["A_letter_only"] = {"note": "arm A emits no rationale by construction"}

# ------------------------------------------------------------------ NULL TEST vs the existing dump
for b in BENCHES:
    f = f"ckpts/gate_lingshu7b_mcq/ckpt_{b}_nothink_norag.jsonl"
    if not os.path.exists(f):
        continue
    prev = {json.loads(l)["idx"]: json.loads(l) for l in open(f) if l.strip()}
    idxs = [i for i in common[b] if i in prev]
    if not idxs:
        continue
    out["null_test"][b] = {
        "n_shared": len(idxs),
        "published_dump_acc": float(np.mean([prev[i]["ok"] for i in idxs])),
        "arm_A_rerun_acc": acc(ARMS[0], idxs, "ok"),
        "pred_agreement": float(np.mean([prev[i]["pred"] == scored[ARMS[0]][i]["letter"] for i in idxs])),
        "source": f,
        "note": "same items, same system instruction, same greedy decode; only max_tokens differs "
                "(16 -> 320), which cannot change a greedy prefix that ends before 16 tokens.",
    }

# ------------------------------------------------------------------ GATE decision
best_arm, best_delta = None, -9
for a in ARMS[1:]:
    d = out["deltas"]["POOLED-all-1488"][f"{a}_minus_A_strict"]["delta"]
    if d > best_delta:
        best_arm, best_delta = a, d
ci = out["deltas"]["POOLED-all-1488"][f"{best_arm}_minus_A_strict"]["ci95"]
disc = out["rationales"]["POOLED-all-1488"][best_arm] if best_arm in out["rationales"]["POOLED-all-1488"] else {}
out["gate"] = {
    "rule": "go=true iff the best (choice)(why) arm loses <= 0.02 accuracy vs letter-only AND its "
            "rationales look discriminative",
    "best_arm": best_arm,
    "delta_vs_letter_only": best_delta,
    "ci95": ci,
    "loss_within_0.02": bool(best_delta >= -0.02),
    "loss_within_0.02_at_ci_lower": bool(ci[0] >= -0.02),
    "rationale_discriminative_proxy": {k: disc.get(k) for k in
                                       ("tfidf_cv_auroc_text_only", "tfidf_cv_auroc_label_shuffled_null",
                                        "auroc_length_vs_correct", "auroc_hedge_vs_correct",
                                        "distinct_rationale_rate", "mean_words")},
}

os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwrote {A.out}")

# ------------------------------------------------------------------ console summary
print("\n=== ACCURACY (exact letter, strict) ===")
print(f"{'group':<22}" + "".join(f"{a.split('_')[0]:>9}" for a in ARMS))
for g in GROUPS:
    r = out["arms"][g]
    print(f"{g:<22}" + "".join(f"{r[a]['acc_strict']:>9.4f}" for a in ARMS))
print("\n=== PAIRED DELTAS vs arm A (95% paired-bootstrap CI) ===")
for g in GROUPS:
    d = out["deltas"][g]
    print(f"  {g}")
    for a in ARMS[1:]:
        v = d[f"{a}_minus_A_strict"]
        print(f"     {a:<26} {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
    if "ORDERING_CONTROL_B2_minus_C2_strict" in d:
        v = d["ORDERING_CONTROL_B2_minus_C2_strict"]
        print(f"     {'B2 - C2 (ordering control)':<26} {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
print("\n=== PARSE OK ===")
for g in GROUPS:
    print(f"{g:<22}" + "".join(f"{out['arms'][g][a]['parse_ok_rate']:>9.4f}" for a in ARMS))
print("\n=== TOKENS (mean / median / p90) ===")
for g in GROUPS:
    print(f"{g:<22}" + "".join(f"{out['tokens'][g][a]['mean']:>6.1f}/{out['tokens'][g][a]['median']:>4.0f}/"
                               f"{out['tokens'][g][a]['p90']:>4.0f}" for a in ARMS))
print("\n=== NULL TEST (arm A vs existing gate_lingshu7b_mcq dump) ===")
for b, v in out["null_test"].items():
    print(f"{b:<24} dump={v['published_dump_acc']:.4f}  rerun={v['arm_A_rerun_acc']:.4f}  "
          f"pred_agree={v['pred_agreement']:.4f}  n={v['n_shared']}")
print("\n=== GATE ===")
print(json.dumps(out["gate"], indent=2))
