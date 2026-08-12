#!/usr/bin/env python3
"""
pmcvqa_answer_bias_audit.py -- RETRACTION CHECK on the one MCQ cell that carries the project's
vs-always-32B-direct claim.

THE THREAT.  PMC-VQA `test_2.csv` (v2, 33,430 items -- the MedEvalKit-track split we report, with
ZERO published verification) has a heavily skewed gold-answer-letter marginal.  PMC_VQA is
load-bearing for the vs-32B-direct claim (leave-one-cell-out turns the macro delta negative without
it) and the certified veto's +0.0095 on PMC is 100% of the MCQ-side macro delta.  If that +0.0095 is
the veto riding an answer-letter / option-position prior rather than genuine 7B competence, a
published claim is contaminated.

WHAT THIS SCRIPT DOES (no GPU, no new inference; everything is recomputed from files on disk):

  NULL TEST   reproduce the published certified-veto numbers bit-for-bit from the same code path
              (beat32b_fusion.mcq + beat32b_more.f8_veto) and report max abs deviation vs the
              reproduction block stored in artifacts/pmc_label_noise_audit_2026-07-29.json.

  T1  Gold-letter marginals on BOTH PMC splits + the two train splits, straight from the CSVs.
  T2  Image-free floors: constant-letter, longest/shortest option, and what fraction of each
      model's measured accuracy they reproduce.
  T3  Letter-marginal-only expected accuracy for each model  (sum_L q_model(L) * p_gold(L)) and
      "skill above marginal" = acc - marginal_acc.  Applied globally AND inside the veto set.
      This is the decisive non-stochastic decomposition: if the 7B's advantage inside the veto set
      equals the difference of the two marginal-only accuracies, the gain is a letter prior.
  T4  Veto gain stratified by GOLD letter (per-letter delta + contribution + paired bootstrap CI).
  T5  Letter-balanced delta = macro over the 4 gold letters (1/4 each).  Under a uniform gold
      marginal a constant-letter policy scores exactly 0.25, so any surviving delta is NOT a letter
      prior.  Stratified paired bootstrap.
  T6  Veto gain stratified by the 7B's PREDICTED letter, and the delta reweighted to a uniform
      7B-predicted-letter marginal.  Also the frequent-letter (B/C) vs rare-letter (A/D) split.
  T7  Composition of the certified/veto set: gold- and predicted-letter marginals inside vs outside.
  T8  PERMUTATION NULL that holds the letter composition fixed: permute the veto flag WITHIN
      gold-letter strata (and within gold x 7B-pred cells), preserving the per-letter veto rate
      exactly, and recompute the delta.  If the observed delta sits far above this null, the veto
      is selecting ITEMS, not LETTERS.
  T9  Reconciliation with the 2026-07-29 label-noise audit (37% UNANSWERABLE + 9% BAD-GOLD among
      100 audited veto wins): join those classifications back to gold letters and ask whether the
      defect rate is letter-dependent.
  T10 Cheap data-quality quantification for the OTHER cells flagged by the literature sweep
      (SLAKE / VQA-RAD / PathVQA): majority-class ("trivially guessable") floors on the closed
      cells, and train<->eval question/image overlap where the splits are on disk.

NUMERICS PINNED: OMP_NUM_THREADS=1, numpy 1.26.4, row order = MedEvalKit results.json file order
(the f8_veto cross-fit folds are `arange(n) % 5`, so row order is load-bearing -- see the standing
caveat "row order +0.0041").  Bootstraps: paired item bootstrap, nboot=10000, seed pinned.

Launch from repo root (CPU only):
    OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_audit.py
"""
import os, sys, csv, json, collections
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import beat32b_fusion as B          # noqa: E402  (BF.mcq -- the published MCQ loader)
import beat32b_more as M            # noqa: E402  (M.f8_veto -- the published certified veto)

PMC_DIR = "/data/dan/dataset/medevalkit/PMC-VQA"
MEK = os.path.join(ROOT, "MedEvalKit")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/pmcvqa_answer_bias_audit_2026-08-11.json")
LETTERS = ["A", "B", "C", "D"]
NBOOT = 10000
SEED = 20260811


# ------------------------------------------------------------------ helpers
def r4(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


def r5(x):
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return x


def paired_boot(diff, nboot=NBOOT, seed=SEED):
    """Paired item bootstrap over a per-item difference vector."""
    d = np.asarray(diff, float)
    n = len(d)
    if n == 0:
        return dict(delta=None, ci=[None, None], n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    b = d[idx].mean(axis=1)
    return dict(delta=r5(d.mean()), ci=[r5(np.percentile(b, 2.5)), r5(np.percentile(b, 97.5))],
                n=int(n), sig=bool(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0))


def stratified_boot_macro(diff, strat, keys, nboot=NBOOT, seed=SEED):
    """Macro-over-strata mean with a stratified paired bootstrap (resample items within stratum)."""
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strat == k) for k in keys]
    groups = [g for g in groups if len(g) > 0]
    if not groups:
        return dict(delta=None, ci=[None, None])
    point = float(np.mean([d[g].mean() for g in groups]))
    boots = np.empty(nboot)
    for b in range(nboot):
        boots[b] = np.mean([d[g[rng.integers(0, len(g), len(g))]].mean() for g in groups])
    return dict(delta=r5(point), ci=[r5(np.percentile(boots, 2.5)), r5(np.percentile(boots, 97.5))],
                sig=bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0),
                n_strata=len(groups), n_per_stratum=[int(len(g)) for g in groups])


def letter_marginal_acc(pred_letters, gold_letters):
    """Expected accuracy of a predictor that keeps this letter MARGINAL but has zero item-level
    skill: sum_L q(L) * p(L).  Anything the model scores above this is item-level information."""
    n = len(gold_letters)
    if n == 0:
        return None
    q = collections.Counter(pred_letters)
    p = collections.Counter(gold_letters)
    return float(sum((q.get(L, 0) / n) * (p.get(L, 0) / n) for L in set(list(q) + list(p))))


def dist(items):
    n = len(items)
    c = collections.Counter(items)
    return {k: dict(n=int(v), frac=r4(v / n)) for k, v in sorted(c.items(), key=lambda t: -t[1])}, n


# ------------------------------------------------------------------ T1/T2: the CSVs
def read_pmc_csv(fname):
    """Return list of dicts with gold letter + the four option strings, from the raw CSV on disk."""
    p = os.path.join(PMC_DIR, fname)
    rd = csv.reader(open(p, encoding="utf-8"))
    hdr = next(rd)
    rows = []
    for row in rd:
        if len(row) == 10:      # v2 schema: index,Figure_path,Caption,Question,A,B,C,D,Answer,split
            gold, opts = row[8].strip(), row[4:8]
        elif len(row) == 8:     # v1 schema: Figure_path,Question,Answer,A,B,C,D,Answer_label
            gold, opts = row[7].strip(), row[3:7]
        else:
            continue
        rows.append(dict(gold=gold, opts=[o.strip() for o in opts]))
    return p, hdr, rows


def strip_prefix(opt, letter):
    """'  B: Left Anterior Descending ' -> 'Left Anterior Descending'."""
    s = opt.strip()
    for pre in (letter + ":", letter + ".", letter + ")", letter):
        if s.startswith(pre):
            return s[len(pre):].strip()
    return s


def csv_letter_audit(fname):
    path, hdr, rows = read_pmc_csv(fname)
    gold = [r["gold"] for r in rows]
    n = len(rows)
    cnt = collections.Counter(gold)
    per_letter = {L: dict(n=int(cnt.get(L, 0)), frac=r4(cnt.get(L, 0) / n)) for L in LETTERS}
    other = {k: int(v) for k, v in cnt.items() if k not in LETTERS}
    const = {L: r4(cnt.get(L, 0) / n) for L in LETTERS}
    maj_letter = max(LETTERS, key=lambda L: cnt.get(L, 0))
    # image-free text heuristics
    def heur(fn):
        hit = 0
        for r in rows:
            texts = [strip_prefix(o, L) for o, L in zip(r["opts"], LETTERS)]
            pick = LETTERS[fn(texts)]
            hit += int(pick == r["gold"])
        return r4(hit / n)
    longest = heur(lambda t: int(np.argmax([len(x) for x in t])))
    shortest = heur(lambda t: int(np.argmin([len(x) for x in t])))
    return dict(
        path=path, n_rows=int(n), header=hdr,
        gold_letter_distribution=per_letter,
        non_ABCD_gold=other,
        B_plus_C=r4((cnt.get("B", 0) + cnt.get("C", 0)) / n),
        constant_letter_accuracy=const,
        majority_letter=maj_letter,
        majority_letter_floor=const[maj_letter],
        longest_option_heuristic=longest,
        shortest_option_heuristic=shortest,
        uniform_random_floor=0.25,
    )


# ------------------------------------------------------------------ main
def main():
    out = {
        "title": "PMC-VQA answer-letter / option-position bias audit -- is the certified veto's "
                 "+0.0095 on PMC_VQA (100% of the MCQ-side macro delta, and the load-bearing cell "
                 "for the vs-always-32B-direct claim) genuine 7B competence or an answer-prior artifact?",
        "date": "2026-08-11",
        "attack": "Attack B (retraction check)",
        "no_gpu": True,
        "no_fabricated_numbers": True,
        "reproduce": "OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_audit.py",
        "numerics_pinned": dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "unset"),
                                numpy=np.__version__, python=sys.version.split()[0],
                                row_order="MedEvalKit results.json file order (f8_veto folds = arange(n) %% 5)",
                                nboot=NBOOT, seed=SEED),
    }

    # ---------------- loader provenance ----------------
    loader_src = open(os.path.join(MEK, "utils/PMC_VQA/PMC_VQA.py")).read().splitlines()
    line39 = loader_src[38].strip()
    out["loader_provenance"] = dict(
        file="MedEvalKit/utils/PMC_VQA/PMC_VQA.py",
        line_39=line39,
        hardcoded_split_verified=bool("test_2.csv" in line39),
        note="Verified by reading the file, not assumed. The MedEvalKit track evaluates test_2.csv only.",
    )

    # ---------------- T1/T2 ----------------
    out["T1_T2_split_letter_audit"] = {
        "test_2.csv (v2, the split WE report)": csv_letter_audit("test_2.csv"),
        "test_clean.csv (v1, human-verified, the internal-harness split)": csv_letter_audit("test_clean.csv"),
        "test.csv (v1 full test)": csv_letter_audit("test.csv"),
        "train_2.csv (v2 train)": csv_letter_audit("train_2.csv"),
    }

    # ---------------- load the evaluated cell ----------------
    d = B.mcq("PMC_VQA")
    r7 = B.load_raw("lingshu7b_full", "PMC_VQA")
    r32 = B.load_raw("lingshu32b_full", "PMC_VQA")
    n = len(d["ok7"])
    assert len(r7) == n == 33430, (len(r7), n)

    gold = np.array([str(r["answer"]).strip() for r in r7[:n]])
    ok7, ok32 = d["ok7"], d["ok32"]

    def as_letter(s):
        """First A-D letter emitted by the model; 'other' if the response is not a letter pick."""
        t = str(s).strip().upper()
        for ch in t:
            if ch in "ABCD":
                return ch
        return "other"
    p7L = np.array([as_letter(r["response"]) for r in r7[:n]])
    p32L = np.array([as_letter(r["response"]) for r in r32[:n]])

    # ---------------- NULL TEST ----------------
    ok_veto, veto = M.f8_veto(d)
    published = json.load(open(os.path.join(
        ROOT, "results/cascade_methods/artifacts/pmc_label_noise_audit_2026-07-29.json")))["reproduction"]
    got = dict(acc_7b=float(ok7.mean()), acc_32b_nt=float(ok32.mean()),
               veto_acc=float(ok_veto.mean()), veto_delta=float(ok_veto.mean() - ok32.mean()),
               veto_rate=float(veto.mean()),
               n_win_veto=int(((ok_veto > ok32)).sum()), n_loss_veto=int(((ok_veto < ok32)).sum()),
               n_pmc=int(n))
    devs = {}
    for k, v in got.items():
        if k in published:
            devs[k] = abs(round(v, 4) - published[k]) if isinstance(published[k], float) else abs(v - published[k])
    out["NULL_TEST"] = dict(
        what="Re-run the PUBLISHED certified veto (beat32b_more.f8_veto on beat32b_fusion.mcq('PMC_VQA')) "
             "and compare to the reproduction block stored in pmc_label_noise_audit_2026-07-29.json.",
        recomputed={k: r5(v) for k, v in got.items()},
        published_reference={k: published[k] for k in got if k in published},
        per_field_abs_deviation={k: r5(v) for k, v in devs.items()},
        max_abs_deviation=r5(max(devs.values())),
        passed=bool(max(devs.values()) < 1e-9),
    )

    # ---------------- T3: letter-marginal-only accuracy ----------------
    goldc = collections.Counter(gold)
    p_gold = {L: goldc.get(L, 0) / n for L in LETTERS}

    def marg_block(mask, label):
        g = gold[mask]; a7 = ok7[mask]; a32 = ok32[mask]
        m7 = letter_marginal_acc(list(p7L[mask]), list(g))
        m32 = letter_marginal_acc(list(p32L[mask]), list(g))
        dq7, _ = dist(list(p7L[mask])); dq32, _ = dist(list(p32L[mask])); dg, _ = dist(list(g))
        return dict(
            subset=label, n=int(mask.sum()),
            gold_letter_marginal={k: v["frac"] for k, v in dg.items()},
            pred_letter_marginal_7B={k: v["frac"] for k, v in dq7.items()},
            pred_letter_marginal_32B={k: v["frac"] for k, v in dq32.items()},
            acc_7B=r5(a7.mean()), acc_32B=r5(a32.mean()),
            marginal_only_acc_7B=r5(m7), marginal_only_acc_32B=r5(m32),
            skill_above_marginal_7B=r5(a7.mean() - m7), skill_above_marginal_32B=r5(a32.mean() - m32),
            acc_gap_7B_minus_32B=r5(a7.mean() - a32.mean()),
            marginal_only_gap=r5(m7 - m32),
            share_of_gap_explained_by_letter_marginal=(
                r4((m7 - m32) / (a7.mean() - a32.mean())) if abs(a7.mean() - a32.mean()) > 1e-9 else None),
        )

    allm = np.ones(n, bool)
    out["T3_letter_marginal_decomposition"] = dict(
        definition="marginal_only_acc = sum_L P(model predicts L) * P(gold is L)  -- the accuracy a "
                   "predictor with this model's letter marginal but ZERO item-level skill would get. "
                   "skill_above_marginal = measured accuracy - that. If the 7B's advantage inside the "
                   "veto set equals marginal_only_gap, the veto's gain IS a letter prior.",
        whole_cell=marg_block(allm, "all 33,430 items"),
        inside_veto_set=marg_block(veto, "veto set (7B kept)"),
        outside_veto_set=marg_block(~veto, "escalated to 32B"),
    )

    # ---------------- T4: gain stratified by GOLD letter ----------------
    diff = ok_veto - ok32
    per_gold = {}
    for L in LETTERS:
        m = gold == L
        bl = paired_boot(diff[m])
        per_gold[L] = dict(
            n=int(m.sum()), share_of_cell=r4(m.mean()),
            acc_7B=r5(ok7[m].mean()), acc_32B=r5(ok32[m].mean()), acc_veto=r5(ok_veto[m].mean()),
            veto_rate=r4(veto[m].mean()),
            delta_veto_minus_32B=bl["delta"], ci95=bl["ci"], sig=bl["sig"],
            contribution_to_overall_delta=r5(m.mean() * diff[m].mean()),
        )
    out["T4_gain_by_gold_letter"] = dict(
        overall=paired_boot(diff),
        per_gold_letter=per_gold,
        contributions_sum_check=r5(sum(v["contribution_to_overall_delta"] for v in per_gold.values())),
        frequent_BC=paired_boot(diff[(gold == "B") | (gold == "C")]),
        rare_AD=paired_boot(diff[(gold == "A") | (gold == "D")]),
    )

    # ---------------- T5: letter-balanced (macro-over-gold-letter) delta ----------------
    out["T5_letter_balanced_delta"] = dict(
        what="Macro over the 4 gold letters (1/4 each). Under a UNIFORM gold marginal a constant-letter "
             "policy scores exactly 0.25 and confers no advantage, so this is the letter-prior-removed delta.",
        macro_over_gold_letters=stratified_boot_macro(diff, gold, LETTERS),
        sample_weighted_for_contrast=paired_boot(diff),
    )

    # ---------------- T6: gain by the 7B's PREDICTED letter ----------------
    per_pred = {}
    for L in LETTERS + ["other"]:
        m = p7L == L
        if m.sum() == 0:
            continue
        bl = paired_boot(diff[m])
        per_pred[L] = dict(
            n=int(m.sum()), share=r4(m.mean()), veto_rate=r4(veto[m].mean()),
            acc_7B=r5(ok7[m].mean()), acc_32B=r5(ok32[m].mean()),
            delta_veto_minus_32B=bl["delta"], ci95=bl["ci"], sig=bl["sig"],
            contribution_to_overall_delta=r5(m.mean() * diff[m].mean()),
        )
    out["T6_gain_by_7B_predicted_letter"] = dict(
        per_predicted_letter=per_pred,
        macro_over_7B_predicted_letter=stratified_boot_macro(diff, p7L, LETTERS),
        pred_BC=paired_boot(diff[(p7L == "B") | (p7L == "C")]),
        pred_AD=paired_boot(diff[(p7L == "A") | (p7L == "D")]),
        note="If the veto only wins where the 7B named a FREQUENT letter (B/C) it is riding the prior; "
             "a win that survives on the RARE letters (A/D) is item-level competence.",
    )
    # joint macro over (gold letter x 7B-pred-is-gold-letter?) -- the strictest reweighting
    joint = np.array([f"{g}|{p}" for g, p in zip(gold, p7L)])
    jkeys = [k for k in sorted(set(joint)) if (joint == k).sum() >= 100]
    out["T6_gain_by_7B_predicted_letter"]["macro_over_gold_x_pred_cells"] = dict(
        cells_used=len(jkeys), min_cell_n=100,
        result=stratified_boot_macro(diff, joint, jkeys),
    )

    # ---------------- T7: composition of the veto set ----------------
    dg_in, _ = dist(list(gold[veto])); dg_out, _ = dist(list(gold[~veto]))
    dp_in, _ = dist(list(p7L[veto])); dp_out, _ = dist(list(p7L[~veto]))
    winm = ok_veto > ok32
    lossm = ok_veto < ok32
    dg_win, _ = dist(list(gold[winm])); dg_loss, _ = dist(list(gold[lossm]))
    out["T7_veto_set_composition"] = dict(
        veto_rate_overall=r4(veto.mean()),
        gold_letter_inside_veto={k: v["frac"] for k, v in dg_in.items()},
        gold_letter_outside_veto={k: v["frac"] for k, v in dg_out.items()},
        gold_letter_whole_cell={L: r4(p_gold[L]) for L in LETTERS},
        pred7_letter_inside_veto={k: v["frac"] for k, v in dp_in.items()},
        pred7_letter_outside_veto={k: v["frac"] for k, v in dp_out.items()},
        gold_letter_among_950_veto_WINS={k: v["frac"] for k, v in dg_win.items()},
        n_wins=int(winm.sum()),
        gold_letter_among_631_veto_LOSSES={k: v["frac"] for k, v in dg_loss.items()},
        n_losses=int(lossm.sum()),
        note="If the veto set / the win set were enriched for the frequent gold letters relative to the "
             "whole cell, the certificate would be selecting letters rather than items.",
    )

    # ---------------- T8: permutation null holding letter composition fixed ----------------
    rng = np.random.default_rng(SEED)
    NPERM = 10000
    adv = ok7 - ok32                      # per-item advantage of keeping the 7B
    obs = float((veto * adv).mean())

    def perm_null(strat_keys, strat_vec, nperm=NPERM):
        """Permute the veto flag WITHIN each stratum -> per-stratum veto rate preserved exactly."""
        groups = [np.flatnonzero(strat_vec == k) for k in strat_keys]
        groups = [g for g in groups if len(g) > 0]
        kcount = [int(veto[g].sum()) for g in groups]
        stats = np.empty(nperm)
        for b in range(nperm):
            s = 0.0
            for g, k in zip(groups, kcount):
                if k == 0:
                    continue
                sel = rng.choice(len(g), size=k, replace=False)
                s += adv[g[sel]].sum()
            stats[b] = s / n
        p = float((stats >= obs).mean())
        return dict(observed_delta=r5(obs), null_mean=r5(stats.mean()), null_sd=r5(stats.std()),
                    null_p975=r5(np.percentile(stats, 97.5)), null_max=r5(stats.max()),
                    p_value_one_sided=r5(max(p, 1.0 / nperm)),
                    p_value_floor=f"<{1.0/nperm}" if p == 0 else None,
                    z=r4((obs - stats.mean()) / stats.std()) if stats.std() > 0 else None)

    out["T8_permutation_null_letter_composition_fixed"] = dict(
        what="Randomly re-assign WHICH items are vetoed, holding the number of vetoed items fixed "
             "within each stratum. This preserves the veto set's letter composition exactly and "
             "destroys only the item-level selection. nperm=%d." % NPERM,
        statistic="mean over all 33,430 items of veto*(ok7-ok32)  ==  veto_acc - acc_32B",
        stratified_by_gold_letter=perm_null(LETTERS, gold),
        stratified_by_gold_x_7Bpred=perm_null(jkeys, joint),
        unstratified_for_contrast=perm_null(["all"], np.array(["all"] * n)),
    )

    # ---------------- T9: reconcile with the 2026-07-29 label-noise audit ----------------
    ws = json.load(open(os.path.join(ROOT, "results/cascade_methods/artifacts/_pmc_audit_worksheet.json")))
    cls = json.load(open(os.path.join(ROOT, "results/cascade_methods/artifacts/_pmc_audit_classifications.json")))
    rec = {}
    for grp, key in [("wins", "sample_wins"), ("losses", "sample_losses"),
                     ("control_agree_correct", "sample_control_agree_correct")]:
        by_letter = collections.defaultdict(collections.Counter)
        tot = collections.Counter()
        checked = 0
        for r in ws[key]:
            c = cls.get(r["item_id"], {}).get("class")
            if c is None:
                continue
            g = r["gold"]
            # verify the worksheet's gold letter against the loaded dump (integrity check)
            if r["row"] < n and gold[r["row"]] != g:
                rec.setdefault("_gold_mismatches", []).append(r["item_id"])
            by_letter[g][c] += 1
            tot[c] += 1
            checked += 1
        defective = lambda cc: cc["BAD-GOLD"] + cc["UNANSWERABLE"] + cc["MULTI-CORRECT"]
        rec[grp] = dict(
            n_audited=checked,
            overall_counts=dict(tot),
            overall_defect_rate=r4(defective(tot) / checked) if checked else None,
            per_gold_letter={L: dict(n=int(sum(by_letter[L].values())),
                                     counts=dict(by_letter[L]),
                                     defect_rate=r4(defective(by_letter[L]) / sum(by_letter[L].values()))
                                     if sum(by_letter[L].values()) else None)
                             for L in LETTERS if sum(by_letter[L].values()) > 0},
        )
    out["T9_reconcile_label_noise_audit"] = dict(
        source="results/cascade_methods/artifacts/pmc_label_noise_audit_2026-07-29.json "
               "(+ _pmc_audit_worksheet.json, _pmc_audit_classifications.json)",
        prior_finding="Among 100 audited veto WINS: 45 GENUINE, 37 UNANSWERABLE, 9 BAD-GOLD, "
                      "7 MULTI-CORRECT, 2 UNCLEAR -> 53% defective [0.4329, 0.6249]. Among 50 audited "
                      "veto LOSSES: 60% defective. Among 50 agree-correct controls: 28% defective.",
        recomputed_from_the_two_worksheets=rec,
        gold_letter_integrity_check="worksheet gold letters matched the loaded dump for every audited item"
        if "_gold_mismatches" not in rec else "MISMATCH -- see _gold_mismatches",
    )

    # ---------------- T10: the other cells ----------------
    other = {}
    for ds, tag, closed in [("SLAKE", "SLAKE_closed", "SLAKE"),
                            ("VQA_RAD", "VQA_RAD_closed", "YESNO"),
                            ("PATH_VQA", "PATH_VQA_closed", "YESNO")]:
        raw = B.load_raw("lingshu7b_full", ds)
        raw32 = B.load_raw("lingshu32b_full", ds)
        if raw is None:
            continue
        nn = min(len(raw), len(raw32))
        if closed == "SLAKE":
            idx = [i for i in range(nn) if raw[i].get("answer_type") == "CLOSED"]
        else:
            idx = [i for i in range(nn) if str(raw[i].get("answer", "")).strip().lower() in ("yes", "no")]
        ans = [str(raw[i]["answer"]).strip().lower() for i in idx]
        c = collections.Counter(ans)
        maj = max(c, key=c.get)
        floor = c[maj] / len(idx)
        a7 = np.mean([B.as_ok(raw[i]) for i in idx])
        a32 = np.mean([B.as_ok(raw32[i]) for i in idx])
        # question-text duplication INSIDE the eval cell
        qs = [str(raw[i].get("question", "")).strip().lower() for i in idx]
        qc = collections.Counter(qs)
        dup_items = sum(v for v in qc.values() if v > 1)
        other[tag] = dict(
            n=len(idx),
            answer_distribution={k: dict(n=int(v), frac=r4(v / len(idx))) for k, v in c.most_common(6)},
            majority_class_floor=r4(floor), majority_class=maj,
            acc_7B=r5(a7), acc_32B_direct=r5(a32),
            frac_of_32B_accuracy_reproduced_by_majority_class=r4(floor / a32),
            n_distinct_questions=len(qc),
            frac_items_whose_question_text_repeats_within_the_cell=r4(dup_items / len(idx)),
            most_common_questions=[[q[:90], int(v)] for q, v in qc.most_common(5)],
        )
    # SLAKE train<->test overlap (both splits are on disk)
    try:
        tr = json.load(open("/data/dan/dataset/medevalkit/SLAKE/train.json"))
        te = json.load(open("/data/dan/dataset/medevalkit/SLAKE/test.json"))
        tr_en = [r for r in tr if r.get("q_lang") == "en"]
        te_en = [r for r in te if r.get("q_lang") == "en"]
        tri = set(r["img_name"] for r in tr_en)
        tei = set(r["img_name"] for r in te_en)
        trq = set((str(r["question"]).strip().lower(), str(r["answer"]).strip().lower()) for r in tr_en)
        teq = [(str(r["question"]).strip().lower(), str(r["answer"]).strip().lower()) for r in te_en]
        n_qa_shared = sum(1 for q in teq if q in trq)
        # same image AND same question text
        trpair = set((r["img_name"], str(r["question"]).strip().lower()) for r in tr_en)
        n_img_q = sum(1 for r in te_en if (r["img_name"], str(r["question"]).strip().lower()) in trpair)
        other["SLAKE_train_eval_overlap"] = dict(
            source="/data/dan/dataset/medevalkit/SLAKE/{train,test}.json (en only)",
            n_train_en=len(tr_en), n_test_en=len(te_en),
            n_train_images=len(tri), n_test_images=len(tei),
            n_images_shared=len(tri & tei),
            frac_test_images_seen_in_train=r4(len(tri & tei) / len(tei)) if tei else None,
            frac_test_items_whose_EXACT_question_and_answer_appear_in_train=r4(n_qa_shared / len(teq)),
            frac_test_items_whose_image_AND_question_appear_in_train=r4(n_img_q / len(te_en)),
            note="This is a dataset property, NOT leakage into our method (our cheap/strong legs are "
                 "zero-shot on SLAKE eval). It bounds how much of the cell is answerable from a "
                 "train-set prior, which matters for the peer numbers we compare against.",
        )
    except Exception as e:                                     # pragma: no cover
        other["SLAKE_train_eval_overlap"] = dict(error=str(e))
    out["T10_other_cells_data_quality"] = other

    # ---------------- verdict ----------------
    v = out["T5_letter_balanced_delta"]["macro_over_gold_letters"]
    ins = out["T3_letter_marginal_decomposition"]["inside_veto_set"]
    pn = out["T8_permutation_null_letter_composition_fixed"]["stratified_by_gold_letter"]
    out["VERDICT"] = dict(
        letter_balanced_delta=v["delta"], letter_balanced_ci=v["ci"], letter_balanced_sig=v["sig"],
        share_of_7B_advantage_in_veto_set_explained_by_letter_marginal=ins[
            "share_of_gap_explained_by_letter_marginal"],
        permutation_null_z=pn["z"], permutation_null_p=pn["p_value_one_sided"],
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT)
    print(json.dumps(out["NULL_TEST"], indent=1))
    print(json.dumps(out["VERDICT"], indent=1))


if __name__ == "__main__":
    main()
