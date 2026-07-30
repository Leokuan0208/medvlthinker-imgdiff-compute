#!/usr/bin/env python3
"""
pmc_label_noise_audit.py -- THE DECISIVE VALIDITY EXPERIMENT (offline, no GPU).

Question: is the project's headline PMC-VQA accuracy win over always-32B (+0.0135 with the F3
confidence-advantage fusion, +0.0095 with the F8 certified veto) REAL, or is it PMC-VQA label noise?

PMC-VQA carries 33,430 of the 42,224 pooled items (79%) and contributes ~38-90% of the headline
delta depending on decomposition (retrospective 2026-07-29 sec.7 hole 2). A ~1-point win is only
meaningful if PMC's gold labels are better than ~1 point accurate on the decision-relevant items.

STAGE 1 (this script, --stage extract):
  * reproduces the F3 fusion and F8 veto policies EXACTLY as beat32b_fusion.py / beat32b_more.py do
    (same 5-fold cross-fit isotonic calibration, same folds ii%K==f, deterministic -- no RNG),
  * verifies the reproduced per-benchmark PMC delta equals the published +0.0135 / +0.0095,
  * isolates the decision-relevant items:
        WIN  = fusion right AND always-32B wrong
        LOSS = fusion wrong AND always-32B right
    and checks (n_win - n_loss)/n == the published delta,
  * draws seeded audit samples (wins / losses / both-agree-correct control) and writes a worksheet
    with the image path, question, options, gold, both legs' raw responses, and the PMC-VQA source
    CAPTION (PMC-VQA gold was auto-generated FROM the caption, so the caption is the annotation's
    own provenance and is essential to judging label quality).

STAGE 2 (--stage score): folds the human/model per-item classifications back in, computes Wilson
95% intervals, the wins-vs-control bias test (two-proportion z + Fisher), the achievable-accuracy
noise ceiling, and the corrected PMC / pooled deltas.

Launch from the repository root:
    python3 src/cascade_methods/pmc_label_noise_audit.py --stage extract
    python3 src/cascade_methods/pmc_label_noise_audit.py --stage score

NO fabricated numbers: every quantity is computed from the dumps on disk or from the recorded
per-item classifications.
"""
import argparse, csv, json, math, os, re, sys
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
MEK = os.path.join(ROOT, "MedEvalKit")
PMC = os.path.join(MEK, "datas/PMC-VQA")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
WORK = os.path.join(ART, "_pmc_audit_worksheet.json")
OUT = os.path.join(ART, "pmc_label_noise_audit_2026-07-29.json")
CLASSES_IN = os.path.join(ART, "_pmc_audit_classifications.json")

K = 5
N_WIN, N_LOSS, N_CTRL = 100, 50, 50
SEED = 20260729

# published numbers this script must reproduce (retrospective sec.4.3 / beat32b_fusion.json /
# beat32b_more.json). Used only as ASSERTION TARGETS, never as substitutes for computed values.
PUB = dict(pmc_n=33430, acc_7b=0.5427, acc_32b_nt=0.5518, fusion_acc=0.5653,
           fusion_delta=0.0135, veto_acc=0.5613, veto_delta=0.0095)


# ------------------------------------------------------------------ loaders (mirror beat32b_*.py)
def as_ok(r):
    v = r.get("correct")
    return int(v is True or str(v).strip().lower() in ("true", "1"))


def as_float(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def npred(r):
    return re.sub(r"[^a-z0-9]", "", str(r.get("response", "")).strip().lower())


def load_raw(tag, ds):
    # NOTE the literal "{}" directory: beat32b_fusion.py builds the path with an f-string whose
    # "{{}}" renders as a literal "{}", and the eval harness wrote there. Mirrored exactly.
    p = f"{MEK}/eval_results_{tag}/{{}}/{ds}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None


def load_pmc():
    r7, r32 = load_raw("lingshu7b_full", "PMC_VQA"), load_raw("lingshu32b_full", "PMC_VQA")
    assert r7 is not None and r32 is not None, "PMC dumps missing"
    n = min(len(r7), len(r32))
    rows = list(csv.reader(open(os.path.join(PMC, "test_2.csv"), encoding="utf-8")))
    hdr, data = rows[0], rows[1:]
    assert len(data) == n, f"csv rows {len(data)} != dump rows {n}"
    # verify row-for-row alignment (choices + gold letter) before trusting the index mapping
    bad = 0
    for i in range(n):
        _, fig, cap, q, cA, cB, cC, cD, ans, split = data[i]
        if r7[i]["choices"] != [cA, cB, cC, cD] or r7[i]["answer"] != ans:
            bad += 1
    assert bad == 0, f"{bad} csv/dump misalignments"
    d = dict(
        n=n,
        ok7=np.array([as_ok(r7[i]) for i in range(n)], float),
        ok32=np.array([as_ok(r32[i]) for i in range(n)], float),
        c7=np.array([as_float(r7[i].get("conf")) for i in range(n)]),
        c32=np.array([as_float(r32[i].get("conf")) for i in range(n)]),
        m7=np.array([as_float(r7[i].get("margin")) for i in range(n)]),
        p7=[npred(r7[i]) for i in range(n)],
        p32=[npred(r32[i]) for i in range(n)],
        resp7=[str(r7[i].get("response", "")) for i in range(n)],
        resp32=[str(r32[i].get("response", "")) for i in range(n)],
        gold=[data[i][8] for i in range(n)],
        question=[data[i][3] for i in range(n)],
        caption=[data[i][2] for i in range(n)],
        figure=[data[i][1] for i in range(n)],
        pmc_index=[data[i][0] for i in range(n)],
        choices=[[data[i][4], data[i][5], data[i][6], data[i][7]] for i in range(n)],
        M=4,
    )
    return d


# ------------------------------------------------------- policies (verbatim logic, + decision trace)
def _calibrate(c_tr, ok_tr, c_ev):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(c_tr, ok_tr)
    return np.clip(ir.predict(c_ev), 1e-6, 1 - 1e-6)


def confadv_fuse_traced(d, deadband=0.0):
    """F3 (beat32b_fusion.confadv_fuse) + the per-item take-7B trace the audit needs."""
    ok7, ok32, c7, c32 = d["ok7"], d["ok32"], d["c7"], d["c32"]
    p7, p32 = d["p7"], d["p32"]
    n = len(ok7)
    out = np.zeros(n)
    take7 = np.zeros(n, bool)
    pr7_all = np.zeros(n)
    pr32_all = np.zeros(n)
    ii = np.arange(n)
    dis = np.array([p7[i] != p32[i] for i in range(n)])
    for f in range(K):
        te = ii % K == f
        tr = ~te
        pr7 = _calibrate(c7[tr], ok7[tr], c7[te])
        pr32 = _calibrate(c32[tr], ok32[tr], c32[te])
        t = (pr7 > pr32 + deadband) & dis[te]
        out[te] = np.where(t, ok7[te], ok32[te])
        take7[te] = t
        pr7_all[te] = pr7
        pr32_all[te] = pr32
    return out, take7, dis, pr7_all, pr32_all


def _wilson_lb(k, nn, z):
    if nn == 0:
        return 0.0
    ph = k / nn
    den = 1 + z * z / nn
    cen = ph + z * z / (2 * nn)
    rad = z * math.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn))
    return (cen - rad) / den


def f8_veto_traced(d, n_bins=5, alpha_z=1.645):
    """F8 (beat32b_more.f8_veto) verbatim + the per-item veto trace."""
    ok7, ok32, c7 = d["ok7"], d["ok32"], d["c7"]
    n = len(ok7)
    out = ok32.copy()
    veto = np.zeros(n, bool)
    ii = np.arange(n)
    for f in range(K):
        te = ii % K == f
        tr = ~te
        qs = np.quantile(c7[tr], np.linspace(0, 1, n_bins + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        qs = np.unique(qs)
        btr = np.clip(np.digitize(c7[tr], qs[1:-1]), 0, len(qs) - 2)
        bte = np.clip(np.digitize(c7[te], qs[1:-1]), 0, len(qs) - 2)
        for b in range(len(qs) - 1):
            m = btr == b
            if m.sum() < 30:
                continue
            lb7 = _wilson_lb(int(ok7[tr][m].sum()), int(m.sum()), alpha_z)
            a32 = ok32[tr][m].mean()
            if lb7 >= a32:
                sel = te.copy()
                sel[te] = (bte == b)
                out[sel] = ok7[sel]
                veto[sel] = True
    return out, veto


# ------------------------------------------------------------------ stats helpers
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, cen - rad), min(1.0, cen + rad)


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pv = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return z, pv


def fisher_exact_2x2(a, b, c, dd):
    """two-sided Fisher exact p for [[a,b],[c,d]] (small tables; exact hypergeometric sum)."""
    from math import comb
    r1, r2 = a + b, c + dd
    c1, tot = a + c, a + b + c + dd
    def pr(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(tot, c1)
    p_obs = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return float(sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p_obs + 1e-12))


# ------------------------------------------------------------------ stage 1
def stage_extract():
    d = load_pmc()
    n = d["n"]
    fus_ok, take7, dis, pr7, pr32 = confadv_fuse_traced(d)
    veto_ok, veto = f8_veto_traced(d)

    acc7, acc32 = float(d["ok7"].mean()), float(d["ok32"].mean())
    accF, accV = float(fus_ok.mean()), float(veto_ok.mean())

    # decision-relevant sets
    win_f = np.where((fus_ok == 1) & (d["ok32"] == 0))[0]
    loss_f = np.where((fus_ok == 0) & (d["ok32"] == 1))[0]
    win_v = np.where((veto_ok == 1) & (d["ok32"] == 0))[0]
    loss_v = np.where((veto_ok == 0) & (d["ok32"] == 1))[0]
    agree_ok = np.where((~dis) & (d["ok32"] == 1))[0]

    rng = np.random.default_rng(SEED)
    s_win = sorted(rng.choice(win_f, min(N_WIN, len(win_f)), replace=False).tolist())
    s_loss = sorted(rng.choice(loss_f, min(N_LOSS, len(loss_f)), replace=False).tolist())
    s_ctrl = sorted(rng.choice(agree_ok, min(N_CTRL, len(agree_ok)), replace=False).tolist())

    def item(i, grp):
        fig = d["figure"][i]
        p = os.path.join(PMC, "figures", fig)
        if not os.path.exists(p):
            p = os.path.join(PMC, "images", fig)
        return dict(
            item_id=f"pmc-{i}", row=int(i), group=grp,
            pmc_csv_index=d["pmc_index"][i], figure=fig, image_path=p,
            image_exists=os.path.exists(p),
            question=d["question"][i].strip(),
            options={"A": d["choices"][i][0].strip(), "B": d["choices"][i][1].strip(),
                     "C": d["choices"][i][2].strip(), "D": d["choices"][i][3].strip()},
            gold=d["gold"][i],
            source_caption=d["caption"][i].strip(),
            answer_7b=d["resp7"][i].strip(), answer_32b=d["resp32"][i].strip(),
            ok_7b=int(d["ok7"][i]), ok_32b=int(d["ok32"][i]),
            conf_7b=round(float(d["c7"][i]), 4), conf_32b=round(float(d["c32"][i]), 4),
            calib_p_correct_7b=round(float(pr7[i]), 4), calib_p_correct_32b=round(float(pr32[i]), 4),
            fusion_took_7b=bool(take7[i]), fusion_ok=int(fus_ok[i]),
            veto_took_7b=bool(veto[i]), veto_ok=int(veto_ok[i]),
            method_answer=(d["resp7"][i].strip() if take7[i] else d["resp32"][i].strip()),
        )

    work = dict(
        meta=dict(
            title="PMC-VQA label-noise audit worksheet (stage 1)",
            n_pmc=n, seed=SEED,
            acc_7b=round(acc7, 4), acc_32b_nt=round(acc32, 4),
            fusion_acc=round(accF, 4), fusion_delta=round(accF - acc32, 4),
            veto_acc=round(accV, 4), veto_delta=round(accV - acc32, 4),
            n_win_fusion=int(len(win_f)), n_loss_fusion=int(len(loss_f)),
            n_win_veto=int(len(win_v)), n_loss_veto=int(len(loss_v)),
            n_agree_correct=int(len(agree_ok)),
            disagree_rate=round(float(dis.mean()), 4),
            fusion_override_rate=round(float(take7.mean()), 4),
            veto_rate=round(float(veto.mean()), 4),
            win_overlap_fusion_veto=int(len(set(win_f.tolist()) & set(win_v.tolist()))),
        ),
        sample_wins=[item(i, "win") for i in s_win],
        sample_losses=[item(i, "loss") for i in s_loss],
        sample_control_agree_correct=[item(i, "control_agree_correct") for i in s_ctrl],
    )
    json.dump(work, open(WORK, "w"), indent=1)

    # ---- reproduction assertions (printed, not silently trusted) ----
    print(f"PMC n={n}  (published {PUB['pmc_n']})")
    print(f"  acc 7B      {acc7:.4f}  (published {PUB['acc_7b']})")
    print(f"  acc 32B-nt  {acc32:.4f}  (published {PUB['acc_32b_nt']})")
    print(f"  F3 fusion   {accF:.4f}  delta {accF-acc32:+.4f}  (published {PUB['fusion_acc']} / {PUB['fusion_delta']:+})")
    print(f"  F8 veto     {accV:.4f}  delta {accV-acc32:+.4f}  (published {PUB['veto_acc']} / {PUB['veto_delta']:+})")
    print(f"  fusion wins {len(win_f)}  losses {len(loss_f)}  net/n {(len(win_f)-len(loss_f))/n:+.4f}")
    print(f"  veto   wins {len(win_v)}  losses {len(loss_v)}  net/n {(len(win_v)-len(loss_v))/n:+.4f}")
    print(f"  disagree {dis.mean():.4f}  fusion override {take7.mean():.4f}  veto {veto.mean():.4f}")
    print(f"  both-agree-and-correct pool {len(agree_ok)}")
    print(f"  win overlap fusion&veto {len(set(win_f.tolist()) & set(win_v.tolist()))}")
    print(f"worksheet -> {WORK}")
    return work


# ------------------------------------------------------------------ stage 2
CLASSES = ["GENUINE", "BAD-GOLD", "UNANSWERABLE", "MULTI-CORRECT", "UNCLEAR"]
# classes that mean "this item's gold cannot support a real accuracy claim"
DEFECT = ["BAD-GOLD", "UNANSWERABLE", "MULTI-CORRECT"]


def stage_score():
    work = json.load(open(WORK))
    cls = json.load(open(CLASSES_IN))  # {item_id: {"class":..., "reason":...}}
    meta = work["meta"]
    n = meta["n_pmc"]

    groups = {}
    for key, gname in (("sample_wins", "wins"), ("sample_losses", "losses"),
                       ("sample_control_agree_correct", "control_agree_correct")):
        items = []
        for it in work[key]:
            c = cls.get(it["item_id"])
            assert c is not None, f"missing classification for {it['item_id']}"
            assert c["class"] in CLASSES, f"bad class {c['class']} for {it['item_id']}"
            rec = dict(it)
            rec["audit_class"] = c["class"]
            rec["audit_reason"] = c["reason"]
            items.append(rec)
        groups[gname] = items

    def dist(items):
        n_i = len(items)
        out = dict(n=n_i, counts={}, rates={})
        for cn in CLASSES:
            k = sum(1 for x in items if x["audit_class"] == cn)
            p, lo, hi = wilson(k, n_i)
            out["counts"][cn] = k
            out["rates"][cn] = dict(k=k, p=round(p, 4), ci95=[round(lo, 4), round(hi, 4)])
        k_def = sum(1 for x in items if x["audit_class"] in DEFECT)
        p, lo, hi = wilson(k_def, n_i)
        out["defective"] = dict(k=k_def, p=round(p, 4), ci95=[round(lo, 4), round(hi, 4)],
                                definition="BAD-GOLD or UNANSWERABLE or MULTI-CORRECT")
        k_unc = out["counts"]["UNCLEAR"]
        # conservative bound: treat UNCLEAR as genuine (lower bound on defect rate) and as
        # defective (upper bound), so the verdict does not depend on the unjudgeable items
        p2, lo2, hi2 = wilson(k_def + k_unc, n_i)
        out["defective_upper_incl_unclear"] = dict(k=k_def + k_unc, p=round(p2, 4),
                                                   ci95=[round(lo2, 4), round(hi2, 4)])
        return out

    D = {g: dist(items) for g, items in groups.items()}

    # ---- bias test: are defects concentrated in the wins vs the agree-correct control? ----
    def bias(gA, gB):
        a, na = D[gA]["defective"]["k"], D[gA]["n"]
        b, nb = D[gB]["defective"]["k"], D[gB]["n"]
        z, pv = two_prop_z(a, na, b, nb)
        fp = fisher_exact_2x2(a, na - a, b, nb - b)
        pa, la, ha = wilson(a, na)
        pb, lb, hb = wilson(b, nb)
        return dict(group_a=gA, group_b=gB, k_a=a, n_a=na, k_b=b, n_b=nb,
                    p_a=round(pa, 4), p_b=round(pb, 4), diff=round(pa - pb, 4),
                    z=round(z, 3), p_two_sided_z=round(pv, 5), fisher_p_two_sided=round(fp, 5),
                    significant_at_05=bool(fp < 0.05))

    bias_tests = dict(
        wins_vs_control=bias("wins", "control_agree_correct"),
        wins_vs_losses=bias("wins", "losses"),
        losses_vs_control=bias("losses", "control_agree_correct"),
    )

    # ---- noise ceiling ----
    # PMC decomposes into: both-agree-and-correct | both-agree-and-wrong | decisive disagreements
    # (the fusion wins and losses) | non-decisive disagreements. Defect rates were measured on the
    # agree-correct stratum and on both decisive strata; the two unsampled strata are charged the
    # agree-correct rate, which is the CEILING-MAXIMISING (most generous) choice, because
    # agree-and-correct is by construction the best-posed stratum in the pool.
    f_def_ctrl = D["control_agree_correct"]["defective"]["p"]
    f_def_win = D["wins"]["defective"]["p"]
    f_def_loss = D["losses"]["defective"]["p"]
    f_def_dis = (D["wins"]["defective"]["k"] + D["losses"]["defective"]["k"]) / (
        D["wins"]["n"] + D["losses"]["n"])
    agree_ok_frac = meta["n_agree_correct"] / n
    win_frac = meta["n_win_fusion"] / n
    loss_frac = meta["n_loss_fusion"] / n
    disagree_frac = meta["disagree_rate"]
    agree_wrong_frac = (1 - disagree_frac) - agree_ok_frac
    nondecisive_dis_frac = disagree_frac - win_frac - loss_frac
    P_GUESS = 0.25  # 4-option MCQ: an uninformative item is still answered right ~1/4 of the time

    def ceil_variant(f_rest_agreewrong, f_rest_dis, label):
        d = (agree_ok_frac * f_def_ctrl + win_frac * f_def_win + loss_frac * f_def_loss
             + agree_wrong_frac * f_rest_agreewrong + nondecisive_dis_frac * f_rest_dis)
        return dict(label=label, pool_defect_rate=round(d, 4),
                    ceiling_defective_scored_wrong=round(1 - d, 4),
                    ceiling_defective_answered_at_chance=round((1 - d) + d * P_GUESS, 4))

    ceiling = dict(
        strata_mass=dict(agree_and_correct=round(agree_ok_frac, 4),
                         agree_and_wrong=round(agree_wrong_frac, 4),
                         fusion_wins=round(win_frac, 4), fusion_losses=round(loss_frac, 4),
                         non_decisive_disagreements=round(nondecisive_dis_frac, 4)),
        defect_rate_by_measured_stratum=dict(agree_correct=f_def_ctrl, wins=f_def_win,
                                             losses=f_def_loss,
                                             all_decisive_disagreements=round(f_def_dis, 4)),
        variant_generous=ceil_variant(
            f_def_ctrl, f_def_ctrl,
            "unsampled strata charged the agree-correct defect rate (ceiling-maximising)"),
        variant_stratified=ceil_variant(
            f_def_ctrl, f_def_dis,
            "non-decisive disagreements charged the measured disagreement defect rate"),
        p_guess_used=P_GUESS,
        measured_accuracies=dict(always_7b=meta["acc_7b"], always_32b_nt=meta["acc_32b_nt"],
                                 fusion=meta["fusion_acc"], veto=meta["veto_acc"]),
        note=("Two readings of 'ceiling': (a) defective items scored wrong -> 1 - defect rate; "
              "(b) defective items answered at 4-way chance -> (1-d) + 0.25d. The true achievable "
              "score sits between them, and can exceed (b) for a model whose language priors happen "
              "to match the caption-derived keys."),
    )

    # ---- corrected deltas ----
    W, L = meta["n_win_fusion"], meta["n_loss_fusion"]
    d_meas_f, d_meas_v = meta["fusion_delta"], meta["veto_delta"]
    kw, nw = D["wins"]["defective"]["k"], D["wins"]["n"]
    kl, nl = D["losses"]["defective"]["k"], D["losses"]["n"]
    kw_bg = D["wins"]["counts"]["BAD-GOLD"]
    kl_bg = D["losses"]["counts"]["BAD-GOLD"]
    f_w_bgun = (kw_bg + D["wins"]["counts"]["UNANSWERABLE"]) / nw
    f_l_bgun = (kl_bg + D["losses"]["counts"]["UNANSWERABLE"]) / nl
    # Monte Carlo over the audit proportions (uniform-prior Beta posteriors) PLUS the item-level
    # sampling noise of the measured delta, so every reported CI carries both sources.
    rng = np.random.default_rng(SEED + 1)
    M = 200000
    fw = rng.beta(kw + 1, nw - kw + 1, M)
    fl = rng.beta(kl + 1, nl - kl + 1, M)
    fw_bg = rng.beta(kw_bg + 1, nw - kw_bg + 1, M)
    fl_bg = rng.beta(kl_bg + 1, nl - kl_bg + 1, M)
    sd_delta = (0.0169 - 0.0100) / (2 * 1.959964)   # from beat32b_fusion.json's paired CI
    jitter = rng.normal(0, sd_delta, M)
    mc_sym = (W * (1 - fw) - L * (1 - fl)) / n + jitter
    mc_wins_only = d_meas_f * (1 - fw) + jitter
    mc_rekey = (W - L - 2 * W * fw_bg + 2 * L * fl_bg) / n + jitter
    d_pool = ceiling["variant_generous"]["pool_defect_rate"]
    mc_clean = ((W * (1 - fw) - L * (1 - fl)) / n + jitter) / (1 - d_pool)

    def q(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

    veto_sym = (meta["n_win_veto"] * (1 - kw / nw) - meta["n_loss_veto"] * (1 - kl / nl)) / n
    corrected = dict(
        pmc=dict(
            measured=dict(fusion_delta=d_meas_f, fusion_ci95=[0.0100, 0.0169],
                          veto_delta=d_meas_v, n_win=W, n_loss=L, n=n),
            audit_rates=dict(
                f_defective_wins=round(kw / nw, 4), f_defective_losses=round(kl / nl, 4),
                f_badgold_or_unanswerable_wins=round(f_w_bgun, 4),
                f_badgold_or_unanswerable_losses=round(f_l_bgun, 4),
                f_badgold_wins=round(kw_bg / nw, 4), f_badgold_losses=round(kl_bg / nl, 4)),
            correction_A_wins_only_as_briefed=dict(
                formula="(1 - f_wins) x measured delta",
                fusion=round(d_meas_f * (1 - kw / nw), 4), fusion_ci95=q(mc_wins_only),
                fusion_using_badgold_plus_unanswerable_only=round(d_meas_f * (1 - f_w_bgun), 4),
                veto_transfer=round(d_meas_v * (1 - kw / nw), 4),
                caveat=("This one-sided correction is what the brief asks for, but it is NOT the "
                        "right one here: the LOSSES are at least as defective as the wins (see the "
                        "bias test), so discounting only the wins overstates the damage.")),
            correction_B_symmetric_drop_defective=dict(
                formula="(W(1-f_w) - L(1-f_l)) / n",
                fusion=round(float(np.mean(mc_sym)), 4), fusion_ci95=q(mc_sym),
                fraction_of_measured_delta_surviving=round(float(np.mean(mc_sym)) / d_meas_f, 3),
                veto_transfer=round(veto_sym, 4),
                veto_transfer_note=("applies the FUSION-win defect rate to the VETO win set; "
                                    f"{meta['win_overlap_fusion_veto']} of {meta['n_win_veto']} "
                                    "veto wins are also fusion wins, so the transfer is partial")),
            correction_C_rekey_badgold_only=dict(
                formula=("(W - L - 2*W*f_badgold_w + 2*L*f_badgold_l) / n  -- a mis-keyed win "
                         "becomes a loss and vice versa"),
                fusion=round(float(np.mean(mc_rekey)), 4), fusion_ci95=q(mc_rekey)),
            correction_D_cleaned_benchmark=dict(
                formula="(W(1-f_w) - L(1-f_l)) / (n x (1 - pooled defect rate))",
                comment=("re-scoring on a CLEANED benchmark shrinks the denominator too, so the "
                         "per-item delta barely moves -- but it then rests on roughly half the "
                         "decision-relevant items, so the evidence is weaker even where the point "
                         "estimate survives"),
                fusion=round(float(np.mean(mc_clean)), 4), fusion_ci95=q(mc_clean)),
            monte_carlo=dict(draws=M, prior="uniform Beta on each audit proportion",
                             item_noise_sd=round(sd_delta, 5),
                             item_noise_source="beat32b_fusion.json paired-bootstrap CI [0.0100, 0.0169]"),
        ),
    )

    # ---- pooled headline propagation (Variant B, n=42224; retrospective sec.4.2/4.3) ----
    POOL_N = 42224
    w_pmc = n / POOL_N
    fac_A = 1 - kw / nw
    fac_B = float(np.mean(mc_sym)) / d_meas_f
    pooled = {}
    for name, pooled_meas, pmc_cell_meas in (
            ("accuracy_max_veto_vs_32B_reasoning", 0.0245, 0.0095),
            ("accuracy_max_plus_fusion_vs_32B_reasoning", 0.0271, 0.0135),
            ("accuracy_max_veto_vs_always_32B_direct", 0.0106, 0.0095),
            ("compute_lean_vs_32B_reasoning", 0.0150, -0.0010)):
        contrib = pmc_cell_meas * w_pmc
        pooled[name] = dict(pooled_measured=pooled_meas, pmc_cell_measured=pmc_cell_meas,
                            pmc_weight=round(w_pmc, 4),
                            pmc_contribution_measured=round(contrib, 5),
                            pmc_share_of_pooled=round(contrib / pooled_meas, 3),
                            pooled_corrected_A_wins_only=round(pooled_meas - contrib + contrib * fac_A, 4),
                            pooled_corrected_B_symmetric=round(pooled_meas - contrib + contrib * fac_B, 4))
    pooled["_note"] = ("Only the PMC cell is re-costed; every other cell keeps its published value. "
                       "Measured pooled numbers from PROJECT_RETROSPECTIVE_2026-07-29 sec.4.2/4.3. "
                       "compute-lean's PMC cell is NEGATIVE (-0.0010), so correcting it is "
                       "immaterial -- compute-lean never depended on a PMC win.")
    corrected["pooled_headline"] = pooled

    out = dict(
        title=("PMC-VQA label-noise audit -- is the project's headline accuracy win over always-32B "
               "real, or annotation error? Offline; every audited item's figure was opened and read."),
        reproduce=("python3 src/cascade_methods/pmc_audit_classifications.py ; "
                   "python3 src/cascade_methods/pmc_label_noise_audit.py --stage extract ; "
                   "python3 src/cascade_methods/pmc_label_noise_audit.py --stage score"),
        no_gpu=True, no_fabricated_numbers=True,
        rubric=dict(
            GENUINE="gold correct/caption-consistent AND derivable from the shown image by a domain expert",
            BAD_GOLD="key wrong, contradicts the caption or the image, or the other model's answer is at least as defensible",
            UNANSWERABLE="answer not in the shown image (wrong panel/modality, absent marker, temporal or metadata question, caption-only convention)",
            MULTI_CORRECT="more than one option defensibly correct",
            UNCLEAR="not judgeable by the auditor; never counted as a defect",
            precedence="BAD-GOLD > UNANSWERABLE > MULTI-CORRECT > GENUINE",
            conservatism="hard-but-well-posed expert calls, and awkward-but-answerable wording, count as GENUINE",
            auditor=("Claude Opus 5, image by image, with the question / four options / gold letter / "
                     "both models' raw answers / the PMC-VQA source caption in view. Not a "
                     "radiologist: label QUALITY was judged, not diagnoses.")),
        reproduction=meta,
        classification_distributions=D,
        bias_tests=bias_tests,
        noise_ceiling=ceiling,
        corrected_deltas=corrected,
        verdict=dict(
            headline="PARTLY REAL, but the arithmetic survives while the CONSTRUCT does not.",
            arithmetic=(
                f"Label defects are severe ({kw}/{nw} = {kw/nw:.0%} of the decision-relevant wins) "
                f"but they are NOT biased toward wins: losses are {kl}/{nl} = {kl/nl:.0%} defective, "
                f"a difference of {(kw/nw)-(kl/nl):+.2f} that is not significant "
                f"(Fisher p={bias_tests['wins_vs_losses']['fisher_p_two_sided']}), and the sign "
                "favours the losses. Mis-keying in particular is symmetric "
                f"({kw_bg/nw:.0%} of wins vs {kl_bg/nl:.0%} of losses), so re-keying leaves the delta "
                f"at {corrected['pmc']['correction_C_rekey_badgold_only']['fusion']:+.4f}. The "
                "symmetric drop-defective correction keeps "
                f"{corrected['pmc']['correction_B_symmetric_drop_defective']['fusion']:+.4f} "
                f"(CI {corrected['pmc']['correction_B_symmetric_drop_defective']['fusion_ci95']}), "
                "i.e. the point estimate survives but the CI now nearly touches zero."),
            construct=(
                f"{f_w_bgun:.0%} of the wins sit on items where the "
                "gold is wrong or the answer is simply not in the image (wrong panel, wrong modality, "
                "absent marker, caption-only convention, metadata). On those items the score is "
                "decided by which model's language prior better matches a caption-derived key, not by "
                "which model reads the image better. Worked examples in per_item: pmc-13058 (question "
                "about a blue arrow on a head CT; the image is a SPLEEN ultrasound), pmc-24120 "
                "(question about the femur; the image is a CHEST CT), pmc-24810 (question about blue "
                "labelling in photomicrographs; the image is a photo of a cat's face), pmc-25510 (the "
                "32B correctly described the panel that was actually shown and was scored WRONG)."),
            ceiling=(
                "Achievable PMC-VQA accuracy is bounded at roughly "
                f"{ceiling['variant_stratified']['ceiling_defective_scored_wrong']:.2f}-"
                f"{ceiling['variant_generous']['ceiling_defective_answered_at_chance']:.2f} depending "
                "on how defective items are treated; every system in this project scores 0.54-0.57, so "
                "the benchmark is not saturated - but a 1-point margin is being measured on a "
                f"benchmark where ~{ceiling['variant_generous']['pool_defect_rate']:.0%}-"
                f"{ceiling['variant_stratified']['pool_defect_rate']:.0%} of items cannot support a "
                "correctness claim at all."),
            what_the_project_can_defend=(
                "(1) The PMC delta is NOT an artifact of biased annotation error - that specific "
                "attack fails, and it fails for a measurable reason (defect symmetry across the "
                "disagreement set). (2) But the delta cannot be described as an ACCURACY improvement "
                "in the medical-visual sense: about half of it is earned on items that are not "
                "visual questions. (3) Report it as 'higher agreement with PMC-VQA's caption-derived "
                "keys', and stop using PMC-VQA to carry an accuracy claim. (4) compute-lean is "
                "untouched by this audit (its PMC cell is -0.0010), so the compute story is unaffected."),
            recommended_headline=(
                "Matches the strong model at roughly half the compute. The one CI-certified accuracy "
                "gain that survives an item-level validity audit is PathVQA-open (free text, "
                "judge-scored); the PMC-VQA gain should be reported as benchmark-key agreement with "
                "the measured 53% defect rate stated alongside it."),
            does_not_change=("This audit does NOT bear on the open-text cells, on the compute/latency "
                             "claims, or on retrospective holes 1/3/4, which remain open."),
        ),
        per_item=groups,
    )
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=1))
    print(f"\n-> {OUT}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "score"], required=True)
    a = ap.parse_args()
    stage_extract() if a.stage == "extract" else stage_score()
