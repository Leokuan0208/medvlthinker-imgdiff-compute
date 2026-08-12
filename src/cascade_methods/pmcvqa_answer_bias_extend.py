#!/usr/bin/env python3
"""
pmcvqa_answer_bias_extend.py -- extension of pmcvqa_answer_bias_audit.py.

Adds four things the first pass did not cover:

 E1  The SAME answer-letter decomposition applied to the **F3 confidence-advantage FUSION** arm, not
     just the certified veto.  This matters because the MCQ-only claim that actually carries the
     biggest number (armcombine_mcqonly_2026-08-11.json, +0.00169 [+0.00126, +0.00212] macro) is the
     FUSION arm, whose PMC cell delta is +0.01349 -- larger than the veto's +0.00954.

 E2  The MACRO CONSEQUENCE.  The MCQ-only policies are byte-identical to always-32B-direct on 7 of
     the 8 cells, so their macro delta is exactly (PMC cell delta)/8.  Recompute both macro deltas
     with the PMC cell delta replaced by its letter-balanced (macro-over-gold-letter) version, with a
     stratified paired bootstrap.  Clearly labelled as a SENSITIVITY, not a re-definition of the
     benchmark.

 E3  QUESTION-BLIND (image-blind) baselines for SLAKE / PathVQA / VQA-RAD -- the ReMedQA-genre check.
     For each eval item, predict the modal answer of that EXACT question text in the TRAIN split.
     This is the yes/no analogue of PMC's constant-letter floor and is the number a reviewer asks for
     once a 73.6% two-option concentration has been found in a sibling cell.

 E4  Language composition of MedEvalKit's SLAKE cells (the closed-cell answer distribution contains
     Chinese labels, so the cell is bilingual; quantify it).

 Also records, explicitly, that the gold x predicted-letter joint stratification computed in the
 first pass is a DEGENERATE control and must not be cited: gold == pred is exactly "the 7B is
 correct", so conditioning on that joint conditions on the outcome being measured.

Launch from repo root (CPU only):
    OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_extend.py
"""
import os, sys, json, glob, collections
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import beat32b_fusion as B          # noqa: E402
import beat32b_more as M            # noqa: E402

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/pmcvqa_answer_bias_extend_2026-08-11.json")
LETTERS = ["A", "B", "C", "D"]
NBOOT = 10000
SEED = 20260811


def r5(x):
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return x


def r4(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


def paired_boot(diff, nboot=NBOOT, seed=SEED, scale=1.0):
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(axis=1) * scale
    return dict(delta=r5(d.mean() * scale),
                ci=[r5(np.percentile(b, 2.5)), r5(np.percentile(b, 97.5))],
                sig=bool(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0), n=int(len(d)))


def macro_letter_boot(diff, strat, nboot=NBOOT, seed=SEED, scale=1.0):
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strat == L) for L in LETTERS]
    point = float(np.mean([d[g].mean() for g in groups])) * scale
    boots = np.empty(nboot)
    for i in range(nboot):
        boots[i] = np.mean([d[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) * scale
    return dict(delta=r5(point), ci=[r5(np.percentile(boots, 2.5)), r5(np.percentile(boots, 97.5))],
                sig=bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0))


def letter_marginal_acc(pred, gold):
    n = len(gold)
    q = collections.Counter(pred)
    p = collections.Counter(gold)
    return float(sum((q.get(L, 0) / n) * (p.get(L, 0) / n) for L in set(list(q) + list(p))))


def decompose(name, ok_arm, took7, ok7, ok32, gold, p7L, p32L):
    """Full letter decomposition of one MCQ arm on PMC."""
    n = len(ok_arm)
    diff = ok_arm - ok32
    per = {}
    for L in LETTERS:
        m = gold == L
        bl = paired_boot(diff[m])
        per[L] = dict(n=int(m.sum()), share=r4(m.mean()),
                      acc_7B=r5(ok7[m].mean()), acc_32B=r5(ok32[m].mean()), acc_arm=r5(ok_arm[m].mean()),
                      took7_rate=r4(took7[m].mean()),
                      delta=bl["delta"], ci95=bl["ci"], sig=bl["sig"],
                      contribution=r5(m.mean() * diff[m].mean()))
    # letter-marginal-only decomposition inside the override set
    ov = took7
    m7 = letter_marginal_acc(list(p7L[ov]), list(gold[ov]))
    m32 = letter_marginal_acc(list(p32L[ov]), list(gold[ov]))
    gap = float(ok7[ov].mean() - ok32[ov].mean())
    # permutation null holding letter composition fixed
    rng = np.random.default_rng(SEED)
    adv = ok7 - ok32
    obs = float((ov * adv).mean())
    groups = [np.flatnonzero(gold == L) for L in LETTERS]
    kc = [int(ov[g].sum()) for g in groups]
    stats = np.empty(10000)
    for b in range(10000):
        s = 0.0
        for g, k in zip(groups, kc):
            s += adv[g[rng.choice(len(g), size=k, replace=False)]].sum()
        stats[b] = s / n
    p = float((stats >= obs).mean())
    return dict(
        arm=name,
        overall=paired_boot(diff),
        override_rate=r4(ov.mean()),
        per_gold_letter=per,
        frequent_BC=paired_boot(diff[(gold == "B") | (gold == "C")]),
        rare_AD=paired_boot(diff[(gold == "A") | (gold == "D")]),
        LETTER_BALANCED_macro_over_gold_letters=macro_letter_boot(diff, gold),
        inside_override_set=dict(
            n=int(ov.sum()), acc_7B=r5(ok7[ov].mean()), acc_32B=r5(ok32[ov].mean()),
            acc_gap=r5(gap), marginal_only_acc_7B=r5(m7), marginal_only_acc_32B=r5(m32),
            marginal_only_gap=r5(m7 - m32),
            share_of_gap_explained_by_letter_marginal=r4((m7 - m32) / gap) if abs(gap) > 1e-9 else None),
        permutation_null_letter_composition_fixed=dict(
            observed=r5(obs), null_mean=r5(stats.mean()), null_sd=r5(stats.std()),
            null_max=r5(stats.max()), z=r4((obs - stats.mean()) / stats.std()),
            p_one_sided=r5(max(p, 1e-4)), nperm=10000),
    )


# --------------------------------------------------------------- E3/E4: the other cells
def question_blind_baseline():
    import pyarrow.parquet as pq
    res = {}

    def hf(name, tr_glob, te_glob):
        def rows(g):
            out = []
            for p in sorted(glob.glob(g)):
                t = pq.read_table(p, columns=["question", "answer"])
                qs, an = t.column("question").to_pylist(), t.column("answer").to_pylist()
                out += [(str(q).strip().lower(), str(a).strip().lower()) for q, a in zip(qs, an)]
            return out
        tr, te = rows(tr_glob), rows(te_glob)
        modal = {}
        byq = collections.defaultdict(collections.Counter)
        for q, a in tr:
            byq[q][a] += 1
        for q, c in byq.items():
            modal[q] = c.most_common(1)[0][0]
        glob_modal = collections.Counter(a for _, a in tr).most_common(1)[0][0]
        # restrict to the yes/no subset -- the "closed" cell our results table reports
        te_c = [(q, a) for q, a in te if a in ("yes", "no")]
        hit = sum(1 for q, a in te_c if modal.get(q, glob_modal) == a)
        cover = sum(1 for q, _ in te_c if q in modal)
        maj = collections.Counter(a for _, a in te_c).most_common(1)[0]
        return dict(n_train=len(tr), n_test=len(te), n_test_closed_yesno=len(te_c),
                    frac_test_questions_seen_verbatim_in_train=r4(cover / len(te_c)) if te_c else None,
                    QUESTION_BLIND_train_modal_answer_accuracy=r4(hit / len(te_c)) if te_c else None,
                    majority_class_floor=r4(maj[1] / len(te_c)) if te_c else None,
                    majority_class=maj[0] if te_c else None,
                    global_train_modal_answer=glob_modal)
    res["VQA_RAD"] = hf("VQA_RAD", "/data/dan/dataset/vqa_rad/data/train-*.parquet",
                        "/data/dan/dataset/vqa_rad/data/test-*.parquet")
    res["PATH_VQA"] = hf("PATH_VQA", "/data/dan/dataset/path_vqa/data/train-*.parquet",
                         "/data/dan/dataset/path_vqa/data/test-*.parquet")
    base = "/data/dan/dataset/medevalkit/SLAKE"
    tr = json.load(open(f"{base}/train.json"))
    te = json.load(open(f"{base}/test.json"))
    byq = collections.defaultdict(collections.Counter)
    for r in tr:
        byq[str(r["question"]).strip().lower()][str(r["answer"]).strip().lower()] += 1
    modal = {q: c.most_common(1)[0][0] for q, c in byq.items()}
    glob_modal = collections.Counter(str(r["answer"]).strip().lower() for r in tr).most_common(1)[0][0]
    te_c = [r for r in te if str(r.get("answer_type", "")).upper() == "CLOSED"]
    hit = sum(1 for r in te_c
              if modal.get(str(r["question"]).strip().lower(), glob_modal) == str(r["answer"]).strip().lower())
    cover = sum(1 for r in te_c if str(r["question"]).strip().lower() in modal)
    ac = collections.Counter(str(r["answer"]).strip().lower() for r in te_c)
    lang = collections.Counter(r.get("q_lang") for r in te_c)
    res["SLAKE"] = dict(n_train=len(tr), n_test=len(te), n_test_closed=len(te_c),
                        frac_test_questions_seen_verbatim_in_train=r4(cover / len(te_c)),
                        QUESTION_BLIND_train_modal_answer_accuracy=r4(hit / len(te_c)),
                        majority_class_floor=r4(ac.most_common(1)[0][1] / len(te_c)),
                        majority_class=ac.most_common(1)[0][0],
                        global_train_modal_answer=glob_modal,
                        E4_language_composition_of_closed_cell={k: r4(v / len(te_c)) for k, v in lang.items()},
                        E4_note="MedEvalKit evaluates BOTH SLAKE languages (no language filter in the "
                                "loader), so SLAKE_closed as this project reports it is bilingual.")
    return res


def main():
    out = dict(
        title="Extension of the PMC-VQA answer-letter bias audit: the FUSION arm, the macro "
              "consequence, question-blind baselines on the other three cells, and an explicit "
              "retraction of one degenerate control.",
        date="2026-08-11", no_gpu=True, no_fabricated_numbers=True,
        parent="results/cascade_methods/artifacts/pmcvqa_answer_bias_audit_2026-08-11.json",
        reproduce="OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_extend.py",
        numerics=dict(numpy=np.__version__, nboot=NBOOT, seed=SEED,
                      OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "unset")),
    )

    d = B.mcq("PMC_VQA")
    r7 = B.load_raw("lingshu7b_full", "PMC_VQA")
    r32 = B.load_raw("lingshu32b_full", "PMC_VQA")
    n = len(d["ok7"])
    gold = np.array([str(r["answer"]).strip() for r in r7[:n]])

    def as_letter(s):
        for ch in str(s).strip().upper():
            if ch in "ABCD":
                return ch
        return "other"
    p7L = np.array([as_letter(r["response"]) for r in r7[:n]])
    p32L = np.array([as_letter(r["response"]) for r in r32[:n]])
    ok7, ok32 = d["ok7"], d["ok32"]

    # ---- veto (re-derived here so both arms sit in one file) ----
    ok_veto, veto = M.f8_veto(d)
    # ---- F3 confidence-advantage fusion ----
    # B.confadv_fuse returns only the per-sample ok vector; replicate it verbatim here so the
    # "took the 7B" indicator is EXACT rather than inferred from ok (which is unidentifiable on
    # items where ok7 == ok32).  Asserted below to reproduce B.confadv_fuse bit-for-bit.
    ok_fuse = B.confadv_fuse(d)
    c7, c32, p7, p32 = d["c7"], d["c32"], d["p7"], d["p32"]
    ii = np.arange(n)
    dis = np.array([p7[i] != p32[i] for i in range(n)])
    took7_fuse_exact = np.zeros(n, bool)
    ok_fuse_repl = np.zeros(n)
    for f in range(B.K):
        te = ii % B.K == f
        tr = ~te
        pr7 = B._calibrate(c7[tr], ok7[tr], c7[te])
        pr32 = B._calibrate(c32[tr], ok32[tr], c32[te])
        take7 = (pr7 > pr32) & dis[te]
        took7_fuse_exact[te] = take7
        ok_fuse_repl[te] = np.where(take7, ok7[te], ok32[te])
    assert np.array_equal(ok_fuse_repl, ok_fuse), "fusion replication mismatch"
    out["NULL_TEST_both_arms"] = dict(
        published_reference=json.load(open(os.path.join(
            ROOT, "results/cascade_methods/artifacts/pmc_label_noise_audit_2026-07-29.json")))["reproduction"],
        recomputed=dict(acc_7b=r5(ok7.mean()), acc_32b_nt=r5(ok32.mean()),
                        veto_acc=r5(ok_veto.mean()), veto_delta=r5(ok_veto.mean() - ok32.mean()),
                        fusion_acc=r5(ok_fuse.mean()), fusion_delta=r5(ok_fuse.mean() - ok32.mean()),
                        n_win_fusion=int((ok_fuse > ok32).sum()), n_loss_fusion=int((ok_fuse < ok32).sum())),
        max_abs_deviation=r5(max(
            abs(round(float(ok7.mean()), 4) - 0.5427), abs(round(float(ok32.mean()), 4) - 0.5518),
            abs(round(float(ok_veto.mean()), 4) - 0.5613), abs(round(float(ok_fuse.mean()), 4) - 0.5653),
            abs(round(float(ok_fuse.mean() - ok32.mean()), 4) - 0.0135))),
    )

    out["E1_veto_arm"] = decompose("certified veto (F8, accuracy-max MCQ)", ok_veto, veto,
                                   ok7, ok32, gold, p7L, p32L)
    out["E1_fusion_arm"] = decompose("F3 confidence-advantage fusion (accuracy-max+ MCQ)", ok_fuse,
                                     took7_fuse_exact, ok7, ok32, gold, p7L, p32L)

    # ---- E2: macro consequence ----
    e2 = {}
    for key, arm, published_macro, published_ci in [
            ("MCQ-only certified veto", out["E1_veto_arm"], 0.00119, [0.0009, 0.00148]),
            ("MCQ-only fusion", out["E1_fusion_arm"], 0.00169, [0.00126, 0.00212])]:
        diff = (ok_veto if arm is out["E1_veto_arm"] else ok_fuse) - ok32
        e2[key] = dict(
            published_macro_delta=published_macro, published_macro_ci=published_ci,
            published_source="results/cascade_methods/artifacts/armcombine_mcqonly_2026-08-11.json",
            structure="byte-identical to always-32B-direct on 7 of 8 cells => macro delta == "
                      "(PMC cell delta)/8 exactly",
            pmc_cell_delta_as_published=arm["overall"],
            macro_delta_recomputed_as_pmc_over_8=paired_boot(diff, scale=1.0 / 8),
            SENSITIVITY_macro_delta_with_letter_balanced_PMC=macro_letter_boot(diff, gold, scale=1.0 / 8),
            interpretation="The SENSITIVITY row is NOT a re-definition of the benchmark. It answers: "
                           "how much of this macro win would remain if test_2.csv's gold-letter "
                           "marginal were uniform instead of B+C=73.65%?",
        )
    out["E2_macro_consequence"] = e2

    # ---- E3/E4 ----
    out["E3_E4_question_blind_baselines_other_cells"] = question_blind_baseline()

    # ---- retracted control ----
    out["RETRACTED_CONTROL"] = dict(
        which="T6_gain_by_7B_predicted_letter.macro_over_gold_x_pred_cells and "
              "T8...stratified_by_gold_x_7Bpred in pmcvqa_answer_bias_audit_2026-08-11.json",
        values_not_to_be_cited=dict(macro_over_gold_x_pred=-0.01725, permutation_z=-20.835),
        why="gold == 7B-predicted-letter is EXACTLY the event 'the 7B is correct'. Stratifying or "
            "permuting within the (gold x pred) joint therefore conditions on the outcome being "
            "measured: it forces equal weight on the 4 diagonal (7B-correct) cells and the 12 "
            "off-diagonal (7B-wrong) cells, a 4:12 reweighting of a 54:46 population. Both numbers "
            "are artefacts of that conditioning and are formally invalid as controls. The VALID "
            "controls are the gold-letter-only stratification and the gold-letter-only permutation "
            "null, both of which are reported and both of which SUPPORT the arm.",
    )

    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT)
    for k in ("NULL_TEST_both_arms", "E2_macro_consequence"):
        print(json.dumps(out[k], indent=1))


if __name__ == "__main__":
    main()
