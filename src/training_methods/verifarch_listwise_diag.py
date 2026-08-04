#!/usr/bin/env python3
"""verifarch_listwise_diag.py -- WHY the objective swap cannot move selection.

Structural diagnostics on the same 2345 open-text eval questions, computed from the same on-disk
ground truth as verifarch_listwise.py.  Appends a "diagnostics" block to that artifact.

  D1  decomposition of the sel_eff denominator: which questions can a ranker actually win or lose?
  D2  label-noise floor: identical candidate STRINGS inside one question that carry different judge
      labels -- an irreducible ceiling no score function of the answer can beat.
  D3  where the correct answer sits in the incumbent's ranking when the incumbent picks wrong
      (a mis-ordering a ranking loss could repair vs no signal at all).
  D4  monotone invariance: how much of the pointwise->listwise difference is even expressible.
  D5  in-sample (label-leaking) ceiling of the feature space, pointwise vs listwise -- is the wall
      the objective, the features, or generalisation?

  python3 src/training_methods/verifarch_listwise_diag.py
"""
import json, os, re, string
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
N = 8
ART = J("results/cascade_methods/artifacts/verifarch_listwise_2026-08-04.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)

rows = []
for ds in DS:
    rows += json.load(open(J(f"ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json")))
sl = np.array([r["sl"] for r in rows], dtype=float)
V = np.array([r["scores"] for r in rows], dtype=float)
ds_of = np.array([r["ds"] for r in rows])
n_q = len(rows)
has = sl.max(1) > 0
mixed = (sl.sum(1) > 0) & (sl.sum(1) < N)
pick = V.argmax(1)
ok = sl[np.arange(n_q), pick]


def norm(s):
    s = str(s).lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


D = {}

# --- D1 decomposition of the sel_eff denominator ----------------------------------------------
allc = sl.sum(1) == N
D["D1_sel_eff_decomposition"] = {
    "n_questions": n_q,
    "n_recoverable_denominator_of_sel_eff": int(has.sum()),
    "n_all_correct_free_by_construction": int(allc.sum()),
    "n_mixed_label_the_only_winnable_stratum": int(mixed.sum()),
    "n_all_wrong_unwinnable": int((~has).sum()),
    "incumbent_sel_eff": float(ok[has].mean()),
    "incumbent_sel_eff_on_mixed_stratum": float(ok[mixed].mean()),
    "algebra": ("sel_eff = (n_all_correct + n_mixed*eff_mixed) / n_recoverable "
                f"= ({int(allc.sum())} + {int(mixed.sum())}*{ok[mixed].mean():.6f}) / {int(has.sum())}"),
    "sel_eff_units_per_mixed_item": float(1.0 / has.sum()),
    "note": ("Only the mixed stratum is winnable, so a +0.021 pooled sel_eff win (the 2-sigma detection "
             f"floor) requires flipping {0.021 * has.sum():.0f} of the {int(mixed.sum())} mixed items "
             f"net, i.e. +{0.021 * has.sum() / mixed.sum():.3f} on the mixed stratum."),
}

# --- D2 label-noise floor: same string, different judge label ---------------------------------
noisy_q, noisy_pairs, tot_dup_pairs = 0, 0, 0
noisy_mixed = 0
per_ds = defaultdict(lambda: [0, 0])
for i, r in enumerate(rows):
    by = defaultdict(list)
    for j, p in enumerate(r["preds"]):
        by[norm(p)].append(r["sl"][j])
    bad = False
    for s, labs in by.items():
        if len(labs) > 1:
            tot_dup_pairs += len(labs) * (len(labs) - 1) // 2
            if len(set(labs)) > 1:
                bad = True
                nb = sum(labs)
                noisy_pairs += nb * (len(labs) - nb)
    if bad:
        noisy_q += 1
        per_ds[r["ds"]][0] += 1
        if mixed[i]:
            noisy_mixed += 1
    per_ds[r["ds"]][1] += 1
D["D2_label_noise_identical_strings_different_judge_label"] = {
    "n_questions_with_an_inconsistently_labelled_duplicate_string": noisy_q,
    "frac_of_all_questions": noisy_q / n_q,
    "n_such_questions_inside_the_mixed_stratum": noisy_mixed,
    "frac_of_mixed_stratum": noisy_mixed / int(mixed.sum()),
    "n_inconsistent_duplicate_pairs": noisy_pairs,
    "n_duplicate_pairs_total": tot_dup_pairs,
    "per_dataset_questions_affected": {k: {"n": v[0], "of": v[1]} for k, v in per_ds.items()},
    "why_it_matters": ("On these questions two candidates are the SAME STRING with different judge "
                       "labels, so no function of (image, question, answer) -- pointwise, listwise or "
                       "pairwise -- can order them. On the affected mixed items the expected efficiency "
                       "of any such scorer is bounded below 1 by construction."),
}

# --- D3 where the correct answer sits when the incumbent picks wrong ---------------------------
order = np.argsort(-V, axis=1, kind="stable")
best_correct_rank = []
for i in np.flatnonzero(mixed & (ok == 0)):
    r = next(k for k in range(N) if sl[i, order[i, k]] == 1)
    best_correct_rank.append(r + 1)
c = Counter(best_correct_rank)
D["D3_rank_of_best_correct_when_incumbent_errs"] = {
    "n_mixed_errors": len(best_correct_rank),
    "rank_histogram": {str(k): c[k] for k in sorted(c)},
    "frac_at_rank_2": c[2] / len(best_correct_rank),
    "frac_at_rank_2_or_3": (c[2] + c[3]) / len(best_correct_rank),
    "mean_rank": float(np.mean(best_correct_rank)),
    "reading": ("A ranking loss can only help by promoting a correct candidate past the incumbent's "
                "top-1. If the mass sits at rank 2 the errors are near-ties a ranking objective could "
                "plausibly repair; if it is spread toward rank 8 the incumbent is not mis-ordering, it "
                "is unable to see the correct answer at all."),
}
# how near are those near-ties?
gap = []
for i in np.flatnonzero(mixed & (ok == 0)):
    top = V[i, order[i, 0]]
    bc = max(V[i, j] for j in range(N) if sl[i, j] == 1)
    gap.append(top - bc)
gap = np.array(gap)
D["D3b_score_gap_to_best_correct_on_errors"] = {
    "median": float(np.median(gap)), "mean": float(gap.mean()),
    "frac_gap_below_0.01": float((gap < 0.01).mean()), "frac_gap_below_0.1": float((gap < 0.1).mean()),
    "frac_gap_above_0.5": float((gap > 0.5).mean()),
}

# --- D4 monotone invariance --------------------------------------------------------------------
D["D4_monotone_invariance"] = {
    "statement": ("Selection is argmax WITHIN a question, so sel_eff is exactly invariant to any "
                  "strictly increasing global transform of the score. Any objective change that only "
                  "recalibrates -- which is most of what a pointwise->listwise swap does to a score "
                  "already fit on the same features -- moves selection by exactly 0. To move sel_eff "
                  "an objective must change the WITHIN-QUESTION ORDER on a mixed-label list."),
    "empirical_check_V_only_arms": ("the mlp_V_* arms in the main artifact: all four objectives on the "
                                    "single incumbent-score feature give delta 0.0000 with a degenerate "
                                    "CI, confirming the pipeline adds nothing when the features cannot "
                                    "express a reordering."),
    "n_mixed_lists_with_only_2_distinct_strings": int(sum(
        1 for i in np.flatnonzero(mixed) if len({norm(p) for p in rows[i]["preds"]}) == 2)),
    "n_mixed_lists": int(mixed.sum()),
}

# --- D5 in-sample (label-leaking) ceiling of the feature space ----------------------------------
# Rebuild the FULL / NOV feature matrices exactly as the main script does, then fit WITH the eval
# labels and score in-sample.  This is deliberately leaky: it upper-bounds what these features can
# express, so a low ceiling means the wall is the FEATURES, not the objective or generalisation.
def build_features():
    pool = {}
    for ds in DS:
        for line in open(J(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl")):
            if line.strip():
                r = json.loads(line)
                pool[(ds, r["idx"])] = r
    cols, names = [], []

    def add(nme, arr):
        names.append(nme); cols.append(np.asarray(arr, float).reshape(n_q, N))

    v = V
    vmo = np.zeros_like(v)
    for i in range(N):
        vmo[:, i] = np.delete(v, i, axis=1).max(1)
    vm, vs = v.mean(1, keepdims=True), v.std(1, keepdims=True) + 1e-9
    add("v", v); add("v_z", (v - vm) / vs); add("v_minus_maxother", v - vmo)
    add("v_minus_poolmean", v - vm)
    add("v_rank", np.argsort(np.argsort(-v, 1), 1) / (N - 1.0))
    add("v_pool_std", np.repeat(vs, N, 1)); add("v_pool_mean", np.repeat(vm, N, 1))
    vder = set(names)

    def tokset(s):
        return set(norm(s).split())

    def tokf1(a, b):
        A_, B_ = tokset(a), tokset(b)
        if not A_ or not B_:
            return 0.0
        return 2 * len(A_ & B_) / (len(A_) + len(B_))

    z = lambda: np.zeros((n_q, N))
    dupf, ismo, nch, nw, sm, sx, qo, yn = z(), z(), z(), z(), z(), z(), z(), z()
    for i, r in enumerate(rows):
        p = pool[(r["ds"], r["idx"])]
        pr = r["preds"]; nmv = [norm(x) for x in pr]; cc = Counter(nmv); b = max(cc.values())
        qt = tokset(p["question"])
        for j in range(N):
            dupf[i, j] = cc[nmv[j]] / N; ismo[i, j] = 1.0 * (cc[nmv[j]] == b)
            nch[i, j] = len(pr[j]); nw[i, j] = len(nmv[j].split())
            s = [tokf1(pr[j], pr[k]) for k in range(N) if k != j]
            sm[i, j] = np.mean(s); sx[i, j] = np.max(s)
            t = tokset(pr[j]); qo[i, j] = (len(t & qt) / len(t)) if t else 0.0
            yn[i, j] = 1.0 * (nmv[j] in ("yes", "no"))
    add("dupfrac", dupf); add("is_modal", ismo); add("n_chars", nch); add("n_words", nw)
    add("n_words_dev", nw - nw.mean(1, keepdims=True))
    add("sim_mean_to_others", sm); add("sim_max_to_others", sx); add("q_overlap", qo); add("is_yesno", yn)
    for nme, arr in [("pool_n_distinct", [pool[(r["ds"], r["idx"])]["n_distinct"] for r in rows]),
                     ("pool_self_consistency", [pool[(r["ds"], r["idx"])]["self_consistency"] for r in rows]),
                     ("pool_gen_tokens", [pool[(r["ds"], r["idx"])]["gen_tokens"] for r in rows]),
                     ("q_n_words", [len(norm(pool[(r["ds"], r["idx"])]["question"]).split()) for r in rows])]:
        add(nme, np.repeat(np.array(arr, float).reshape(-1, 1), N, 1))
    for d in DS:
        add(f"is_{d}", np.repeat((ds_of == d).reshape(-1, 1).astype(float), N, 1))
    return np.stack(cols, 2), names, vder


X, names, vder = build_features()
FS = {"V": [names.index("v")], "FULL": list(range(len(names))),
      "NOV": [i for i, n_ in enumerate(names) if n_ not in vder]}


def ceiling(fs, obj, epochs=3000, h=256):
    idx = FS[fs]
    Xa = X[:, :, idx]
    mu, sd = Xa.reshape(-1, len(idx)).mean(0), Xa.reshape(-1, len(idx)).std(0) + 1e-9
    xt = torch.tensor((Xa - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.tensor(sl, dtype=torch.float32, device=DEV)
    m = torch.tensor(mixed, device=DEV)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(len(idx), h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(epochs):
        opt.zero_grad()
        s = net(xt).squeeze(-1)
        if obj == "pointwise_bce":
            L = nn.functional.binary_cross_entropy_with_logits(s, yt)
        elif obj == "anypos":
            lp = torch.log_softmax(s[m], 1)
            L = -torch.logsumexp(lp.masked_fill(yt[m] < 0.5, -1e30), 1).mean()
        else:
            t = yt[m] / yt[m].sum(1, keepdim=True)
            L = -(t * torch.log_softmax(s[m], 1)).sum(1).mean()
        L.backward(); opt.step()
    with torch.no_grad():
        S = net(xt).squeeze(-1).cpu().numpy()
    p = S.argmax(1); o = sl[np.arange(n_q), p]
    return {"sel_eff_in_sample": float(o[has].mean()), "sel_eff_mixed_in_sample": float(o[mixed].mean())}


D["D5_in_sample_leaky_ceiling"] = {
    "protocol": ("a 2x256 MLP fit on the EVAL labels themselves and scored in-sample -- deliberately "
                 "leaky. It upper-bounds what these features can express at unlimited capacity, with "
                 "generalisation removed as a factor."),
    "oracle_at_8_sel_eff": 1.0,
    "incumbent_sel_eff": float(ok[has].mean()),
}
for fs in ["FULL", "NOV"]:
    for obj in ["pointwise_bce", "listnet", "anypos"]:
        D["D5_in_sample_leaky_ceiling"][f"{fs}_{obj}"] = ceiling(fs, obj)
        print("[D5]", fs, obj, D["D5_in_sample_leaky_ceiling"][f"{fs}_{obj}"], flush=True)

art = json.load(open(ART))
art["diagnostics"] = D
json.dump(art, open(ART, "w"), indent=1)
print(json.dumps({k: v for k, v in D.items() if k.startswith(("D1", "D2", "D3"))}, indent=1)[:4000])
print("\nappended diagnostics to", ART)
