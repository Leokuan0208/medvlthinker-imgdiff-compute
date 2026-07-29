#!/usr/bin/env python3
"""
combine_diverse_pairwise.py -- COMPOUNDING experiment analysis (CPU only, no GPU).

Tests whether the two just-validated levers COMPOUND:
  lever 1  DIVERSE generation (portfolio-prompted + temp-laddered, M=15) raises oracle@N but the
           POINTWISE verifier fails to convert it (distractor injection; esp. PMC).
  lever 2  a REAL PAIRWISE verifier beats pointwise on SELECTION (concentrated in near-ties).
HYPOTHESIS: pairwise SELECTION over the DIVERSE pool converts the diverse-gen oracle lift that the
pointwise verifier could not.

Builds the 2x2 SELECTION-ACCURACY table  {pointwise, pairwise} x {iid@8, diverse@15}, per dataset +
pooled, with bootstrap 95% CIs. Correctness = exact-match `oks` (map_correct) used UNIFORMLY across
all four cells (the diverse dumps only carry `oks`; the iid dumps carry both `oks` and judge `sl`, and
the iid `sl`-based numbers are reported separately as a cross-reference to Experiment 2).

Selection methods (both deterministic, over SLOTS, correctness = oks[selected slot]):
  pointwise = argmax(pointwise scores)                    (0 comparisons)
  pairwise  = Copeland / round-robin over the REAL pairwise verdicts   (noise-free pairwise ceiling)
              (identical-string pairs -> 0.5; this is the winning "pairwise" number in Experiment 2)
  knockout  = seed-averaged single-elimination over slots (M-1 comparisons; the cheap pairwise variant)

Inputs (matched idx set = the diverse-dump questions, which are a subset of the iid questions):
  diverse dump      ckpts/openvqa/diverse/ckpt_<DS>_lingshu7b_div.jsonl   (preds/oks/scores, M~15)
  diverse verdicts  ckpts/pairwise_diverse/verdicts_<DS>_lingshu7b_div.jsonl (ki,kj,p_ki_gt_kj)  [NEW GPU]
  iid dump          ckpts/mcq_gen_verify/lingshu7b/ckpt_<...>_sc8.jsonl    (preds/oks/sl/scores, 8)
  iid verdicts      ckpts/pairwise/verdicts_<DS>_lingshu7b.jsonl           (i,j,p_i_gt_j)  [Experiment 2]

Writes results/cascade_methods/artifacts/combine_diverse_pairwise.json
"""
import os, json, itertools
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)

DIVERSE_DUMP = {
    "vqa_rad_open": "ckpts/openvqa/diverse/ckpt_vqa_rad_open_lingshu7b_div.jsonl",
    "slake_open":   "ckpts/openvqa/diverse/ckpt_slake_open_lingshu7b_div.jsonl",
    "pathvqa_open": "ckpts/openvqa/diverse/ckpt_pathvqa_open_lingshu7b_div.jsonl",
    "pmc_content":  "ckpts/openvqa/diverse/ckpt_pmc_content_lingshu7b_div.jsonl",
}
IID_DUMP = {
    "vqa_rad_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_VQA_RAD_lingshu7b_VQA_RAD_content_content_sc8.jsonl",
    "slake_open":   "ckpts/mcq_gen_verify/lingshu7b/ckpt_SLAKE_lingshu7b_SLAKE_content_content_sc8.jsonl",
    "pathvqa_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_PATH_VQA_lingshu7b_content_sc8.jsonl",
    "pmc_content":  "ckpts/mcq_gen_verify/lingshu7b/ckpt_PMC_VQA_lingshu7b_content_sc8.jsonl",
}
DIVERSE_VERDICTS = "ckpts/pairwise_diverse/verdicts_{ds}_lingshu7b_div.jsonl"
IID_VERDICTS     = "ckpts/pairwise/verdicts_{ds}_lingshu7b.jsonl"      # only 3 datasets (no pmc)

DATASETS = ["vqa_rad_open", "slake_open", "pathvqa_open", "pmc_content"]
FULL_2x2 = ["vqa_rad_open", "slake_open", "pathvqa_open"]              # all 4 cells present (iid-pairwise exists)
NEARTIE = 0.10
KO_SEEDS = [0, 1, 2, 3, 4]
B_BOOT = 3000
RNG = np.random.default_rng(20260706)


def norm(s):
    return str(s).strip().lower()


# ------------------------------------------------------------------ loaders
def load_dump(path, need_sl=False):
    out = {}
    for l in open(J(path)):
        if not l.strip():
            continue
        r = json.loads(l)
        idx = str(r["idx"])
        preds = r["preds"]; oks = [int(x) for x in r["oks"]]; sc = r["scores"]
        if sc is None or any(s is None for s in sc):
            continue
        rec = dict(preds=preds, oks=oks, sc=[float(s) for s in sc])
        rec["modal_ok"] = int(r.get("modal_ok", r.get("greedy_ok", 0)))
        if need_sl and "sl" in r:
            rec["sl"] = [0 if x in (None, -1) else int(x) for x in r["sl"]]
        out[idx] = rec
    return out


def load_verdicts_str(path):
    """diverse verdicts keyed by (ki,kj)-normalized-string pair -> p_ki_gt_kj."""
    out = {}
    p = J(path)
    if not os.path.exists(p):
        return out
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l)
        out.setdefault(str(r["idx"]), {})[(r["ki"], r["kj"])] = float(r["p_ki_gt_kj"])
    return out


def load_verdicts_slot(path):
    """iid verdicts keyed by (i,j)-slot pair -> p_i_gt_j."""
    out = {}
    p = J(path)
    if not os.path.exists(p):
        return out
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l)
        out.setdefault(str(r["idx"]), {})[(int(r["i"]), int(r["j"]))] = float(r["p_i_gt_j"])
    return out


# ------------------------------------------------------------------ selection
def pf_diverse(keys, V):
    def pf(i, j):
        ki, kj = keys[i], keys[j]
        if ki == kj:
            return 0.5
        if (ki, kj) in V:
            return V[(ki, kj)]
        if (kj, ki) in V:
            return 1.0 - V[(kj, ki)]
        return None  # missing -> flagged by caller
    return pf


def pf_slot(V):
    def pf(i, j):
        if (i, j) in V:
            return V[(i, j)]
        if (j, i) in V:
            return 1.0 - V[(j, i)]
        return None
    return pf


def copeland_pick(n, sc, pf):
    wins = [0.0] * n
    missing = 0
    for i in range(n):
        for j in range(i + 1, n):
            p = pf(i, j)
            if p is None:
                p = 0.5; missing += 1
            if p > 0.5:
                wins[i] += 1
            elif p < 0.5:
                wins[j] += 1
            else:
                wins[i] += 0.5; wins[j] += 0.5
    best = max(range(n), key=lambda k: (wins[k], sc[k], -k))
    return best, missing


def knockout_pick(n, sc, pf, seed):
    rng = np.random.default_rng(9000 + seed)
    alive = list(range(n)); ncmp = 0
    while len(alive) > 1:
        nxt = []
        for k in range(0, len(alive) - 1, 2):
            i, j = alive[k], alive[k + 1]; ncmp += 1
            p = pf(i, j)
            if p is None:
                p = 0.5
            nxt.append(i if rng.random() < p else j)
        if len(alive) % 2 == 1:
            nxt.append(alive[-1])
        alive = nxt
    return alive[0], ncmp


def build_perq(dump, verdicts, is_diverse, ds):
    """Per question -> dict of method->selected-ok (+ oracle, present flag, neartie flag, cost).
    missing_total: count of distinct-answer pairs with NO verdict (coverage gate).
    `uid`=f"{ds}:{idx}" is a GLOBALLY-unique key (raw idx collides across datasets when pooled)."""
    perq = []
    missing_total = 0
    for idx, rec in dump.items():
        preds, oks, sc = rec["preds"], rec["oks"], rec["sc"]
        n = min(len(preds), len(oks), len(sc))
        preds, oks, sc = preds[:n], oks[:n], sc[:n]
        if n < 1:
            continue
        V = verdicts.get(idx, {})
        if is_diverse:
            keys = [norm(p) for p in preds]
            pf = pf_diverse(keys, V)
            n_uniq = len(set(keys))
        else:
            pf = pf_slot(V)
            n_uniq = len(set(norm(p) for p in preds))
        pw = int(np.argmax(sc))
        cp, miss = copeland_pick(n, sc, pf); missing_total += miss
        ko_oks = []
        for s in KO_SEEDS:
            j, _ = knockout_pick(n, sc, pf, s); ko_oks.append(oks[j])
        ss = sorted(sc, reverse=True)
        gap = ss[0] - ss[1] if len(ss) > 1 else ss[0]
        perq.append(dict(
            idx=idx,
            uid=f"{ds}:{idx}",
            pointwise=oks[pw],
            copeland=oks[cp],
            knockout=float(np.mean(ko_oks)),
            oracle=max(oks),
            modal_ok=rec.get("modal_ok", 0),
            present=1 if max(oks) == 1 else 0,
            neartie=1 if (gap < NEARTIE and max(oks) == 1) else 0,
            n_uniq=n_uniq,
            n_pairs_distinct=n_uniq * (n_uniq - 1) // 2,
        ))
    return perq, missing_total


# ------------------------------------------------------------------ metrics + bootstrap
def sel_acc(perq, key):
    return float(np.mean([q[key] for q in perq])) if perq else float("nan")


def sel_eff(perq, key):
    d = [q[key] for q in perq if q["present"]]
    return float(np.mean(d)) if d else float("nan")


def boot_ci(perq, fn, B=B_BOOT):
    if not perq:
        return (float("nan"), float("nan"), float("nan"))
    n = len(perq); idx = np.arange(n)
    vals = []
    for _ in range(B):
        s = RNG.choice(idx, n, replace=True)
        vals.append(fn([perq[i] for i in s]))
    return (float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_delta(perq_a, key_a, perq_b, key_b, paired_key=None, B=B_BOOT):
    """95% CI on mean(a[key_a]) - mean(b[key_b]). If the two perq lists share the same questions
    (paired), resample question indices jointly; else resample independently."""
    if paired_key is not None:
        # align by paired_key (must be globally unique, e.g. "uid"=ds:idx; raw "idx" collides across datasets)
        bmap = {q[paired_key]: q for q in perq_b}
        pairs = [(qa, bmap[qa[paired_key]]) for qa in perq_a if qa[paired_key] in bmap]
        n = len(pairs); ix = np.arange(n)
        base = np.mean([a[key_a] for a, _ in pairs]) - np.mean([b[key_b] for _, b in pairs])
        vals = []
        for _ in range(B):
            s = RNG.choice(ix, n, replace=True)
            va = np.mean([pairs[i][0][key_a] for i in s]); vb = np.mean([pairs[i][1][key_b] for i in s])
            vals.append(va - vb)
        return dict(delta=float(base), lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)), n=n)
    na, nb = len(perq_a), len(perq_b)
    base = np.mean([q[key_a] for q in perq_a]) - np.mean([q[key_b] for q in perq_b])
    ia, ib = np.arange(na), np.arange(nb)
    vals = []
    for _ in range(B):
        sa = RNG.choice(ia, na, replace=True); sb = RNG.choice(ib, nb, replace=True)
        vals.append(np.mean([perq_a[i][key_a] for i in sa]) - np.mean([perq_b[i][key_b] for i in sb]))
    return dict(delta=float(base), lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)),
                n_a=na, n_b=nb)


def cell_block(perq):
    out = {}
    for m in ["pointwise", "copeland", "knockout", "oracle", "modal_ok"]:
        mean, lo, hi = boot_ci(perq, lambda pq, k=m: sel_acc(pq, k))
        out[m] = dict(sel_acc=round(sel_acc(perq, m), 4), ci95=[round(lo, 4), round(hi, 4)],
                      sel_eff=round(sel_eff(perq, m), 4))
    out["n"] = len(perq)
    out["n_present"] = int(sum(q["present"] for q in perq))
    out["mean_uniq"] = round(float(np.mean([q["n_uniq"] for q in perq])), 2) if perq else None
    out["mean_pairs_distinct"] = round(float(np.mean([q["n_pairs_distinct"] for q in perq])), 2) if perq else None
    return out


# ------------------------------------------------------------------ main
def main():
    # load everything
    div_dump = {ds: load_dump(DIVERSE_DUMP[ds]) for ds in DATASETS}
    iid_dump = {ds: load_dump(IID_DUMP[ds], need_sl=True) for ds in DATASETS}
    div_verd = {ds: load_verdicts_str(DIVERSE_VERDICTS.format(ds=ds)) for ds in DATASETS}
    iid_verd = {ds: load_verdicts_slot(IID_VERDICTS.format(ds=ds)) for ds in DATASETS}

    # matched idx set: diverse questions restricted to those also present in iid dump
    perq_div, perq_iid = {}, {}
    coverage = {}
    for ds in DATASETS:
        common = [i for i in div_dump[ds] if i in iid_dump[ds]]
        dd = {i: div_dump[ds][i] for i in common}
        ii = {i: iid_dump[ds][i] for i in common}
        pdv, miss_dv = build_perq(dd, div_verd[ds], is_diverse=True, ds=ds)
        pii, miss_ii = build_perq(ii, iid_verd[ds], is_diverse=False, ds=ds)
        perq_div[ds] = pdv; perq_iid[ds] = pii
        # coverage: expected distinct-answer pairs vs verdict rows present
        exp_pairs = sum(q["n_pairs_distinct"] for q in pdv)
        got_pairs = sum(len(v) for i, v in div_verd[ds].items() if i in dd)
        coverage[ds] = dict(n_matched=len(common), diverse_missing_pairs=int(miss_dv),
                            iid_missing_pairs=int(miss_ii),
                            expected_distinct_pairs=int(exp_pairs), verdict_rows=int(got_pairs),
                            complete=bool(miss_dv == 0))
        print(f"[{ds:13s}] matched={len(common):4d}  diverse: uniq~{np.mean([q['n_uniq'] for q in pdv]):.1f} "
              f"pairs_scored={got_pairs} missing={miss_dv}  iid_pairwise={'yes' if iid_verd[ds] else 'NO'}",
              flush=True)

    RESULT = dict(
        note=("2x2 selection-accuracy {pointwise,pairwise}x{iid@8,diverse@15}. Correctness = exact-match "
              "oks (map_correct) uniformly across all cells (diverse dumps carry only oks). 'pairwise' = "
              "Copeland/round-robin over REAL pairwise verifier verdicts (Lingshu-7B pooled4 LoRA, "
              "pairwise prompt, both orders position-debiased). Matched idx = diverse questions (subset of "
              "iid). CIs = 3000-sample question bootstrap."),
        coverage=coverage,
        per_dataset={}, pooled_2x2={}, pooled_4ds={}, pmc={}, verdict={},
    )

    # ---- per dataset 2x2 ----
    for ds in DATASETS:
        RESULT["per_dataset"][ds] = dict(
            cell_pointwise_iid8=cell_block(perq_iid[ds])["pointwise"],
            cell_pairwise_iid8=(cell_block(perq_iid[ds])["copeland"] if iid_verd[ds] else None),
            cell_pointwise_diverse=cell_block(perq_div[ds])["pointwise"],
            cell_pairwise_diverse=cell_block(perq_div[ds])["copeland"],
            oracle_iid8=cell_block(perq_iid[ds])["oracle"],
            oracle_diverse=cell_block(perq_div[ds])["oracle"],
            iid8_full=cell_block(perq_iid[ds]),
            diverse_full=cell_block(perq_div[ds]),
        )

    # ---- pooled over the 3 datasets that have all 4 cells ----
    def pool(perq_map, dss):
        out = []
        for ds in dss:
            out.extend(perq_map[ds])
        return out

    pooled_div_3 = pool(perq_div, FULL_2x2)
    pooled_iid_3 = pool(perq_iid, FULL_2x2)
    A = cell_block(pooled_iid_3)["pointwise"]      # pointwise x iid@8
    Cc = cell_block(pooled_iid_3)["copeland"]      # pairwise  x iid@8
    Bb = cell_block(pooled_div_3)["pointwise"]     # pointwise x diverse@15
    Dd = cell_block(pooled_div_3)["copeland"]      # pairwise  x diverse@15
    RESULT["pooled_2x2"] = dict(
        datasets=FULL_2x2, n=len(pooled_div_3),
        table=dict(pointwise_iid8=A, pairwise_iid8=Cc, pointwise_diverse=Bb, pairwise_diverse=Dd),
        oracle_iid8=cell_block(pooled_iid_3)["oracle"], oracle_diverse=cell_block(pooled_div_3)["oracle"],
        knockout_diverse=cell_block(pooled_div_3)["knockout"],
        deltas=dict(
            # compounding tests (all vs pairwise-diverse D)
            D_minus_B_pairwiseDiv_vs_pointwiseDiv=boot_delta(pooled_div_3, "copeland", pooled_div_3, "pointwise", paired_key="uid"),
            D_minus_A_pairwiseDiv_vs_pointwiseIid=boot_delta(pooled_div_3, "copeland", pooled_iid_3, "pointwise", paired_key="uid"),
            # single-lever decomposition
            C_minus_A_pairwiseIid_vs_pointwiseIid=boot_delta(pooled_iid_3, "copeland", pooled_iid_3, "pointwise", paired_key="uid"),
            B_minus_A_pointwiseDiv_vs_pointwiseIid=boot_delta(pooled_div_3, "pointwise", pooled_iid_3, "pointwise", paired_key="uid"),
            D_minus_C_pairwiseDiv_vs_pairwiseIid=boot_delta(pooled_div_3, "copeland", pooled_iid_3, "copeland", paired_key="uid"),
        ),
    )

    # ---- 4-dataset pooled (incl pmc) for the cells that exist for all 4 (no iid-pairwise for pmc) ----
    pooled_div_4 = pool(perq_div, DATASETS)
    pooled_iid_4 = pool(perq_iid, DATASETS)
    RESULT["pooled_4ds"] = dict(
        datasets=DATASETS, n=len(pooled_div_4),
        pointwise_iid8=cell_block(pooled_iid_4)["pointwise"],
        pointwise_diverse=cell_block(pooled_div_4)["pointwise"],
        pairwise_diverse=cell_block(pooled_div_4)["copeland"],
        oracle_iid8=cell_block(pooled_iid_4)["oracle"],
        oracle_diverse=cell_block(pooled_div_4)["oracle"],
        deltas=dict(
            D_minus_B_pairwiseDiv_vs_pointwiseDiv=boot_delta(pooled_div_4, "copeland", pooled_div_4, "pointwise", paired_key="uid"),
            D_minus_A_pairwiseDiv_vs_pointwiseIid=boot_delta(pooled_div_4, "copeland", pooled_iid_4, "pointwise", paired_key="uid"),
            B_minus_A_pointwiseDiv_vs_pointwiseIid=boot_delta(pooled_div_4, "pointwise", pooled_iid_4, "pointwise", paired_key="uid"),
        ),
    )

    # ---- PMC deep dive (the headline conversion test) ----
    pmc_div = perq_div["pmc_content"]; pmc_iid = perq_iid["pmc_content"]
    RESULT["pmc"] = dict(
        n=len(pmc_div),
        pointwise_iid8=cell_block(pmc_iid)["pointwise"],
        pointwise_diverse=cell_block(pmc_div)["pointwise"],
        pairwise_diverse=cell_block(pmc_div)["copeland"],
        knockout_diverse=cell_block(pmc_div)["knockout"],
        oracle_iid8=cell_block(pmc_iid)["oracle"],
        oracle_diverse=cell_block(pmc_div)["oracle"],
        deltas=dict(
            pairwiseDiv_minus_pointwiseDiv=boot_delta(pmc_div, "copeland", pmc_div, "pointwise", paired_key="uid"),
            pairwiseDiv_minus_pointwiseIid=boot_delta(pmc_div, "copeland", pmc_iid, "pointwise", paired_key="uid"),
            pointwiseDiv_minus_pointwiseIid=boot_delta(pmc_div, "pointwise", pmc_iid, "pointwise", paired_key="uid"),
        ),
        conversion=dict(
            oracle_lift_diverse_minus_iid=round(cell_block(pmc_div)["oracle"]["sel_acc"] - cell_block(pmc_iid)["oracle"]["sel_acc"], 4),
            pointwise_converted=round(cell_block(pmc_div)["pointwise"]["sel_acc"] - cell_block(pmc_iid)["pointwise"]["sel_acc"], 4),
            pairwise_converted=round(cell_block(pmc_div)["copeland"]["sel_acc"] - cell_block(pmc_iid)["pointwise"]["sel_acc"], 4),
        ),
    )

    # ---- cost accounting (both-axes, honest) ----
    def mean_pairs(perq):
        return round(float(np.mean([q["n_pairs_distinct"] for q in perq])), 1)
    RESULT["cost"] = dict(
        note="both-axes cost. pairwise-diverse = 15 generations + Copeland comparisons (each = 2 debiased model calls). knockout = M-1 comparisons.",
        gen_samples=dict(iid8=8, diverse=15, extra_gen_factor=round(15/8, 2)),
        pointwise_verify_calls=dict(iid8=8, diverse=15),
        pairwise_copeland_mean_comparisons=dict(
            iid8=round(float(np.mean([q["n_pairs_distinct"] for q in pooled_iid_3])), 1),
            diverse_3ds=mean_pairs(pooled_div_3),
            diverse_pmc=mean_pairs(pmc_div),
        ),
        knockout_mean_comparisons=dict(diverse="~M-1 (14)"),
    )

    # ---- cross-reference: iid cells with judge sl (connect to Experiment 2) ----
    xref = {}
    for ds in FULL_2x2:
        pii_sl = []
        for q, rec in zip(perq_iid[ds], [iid_dump[ds][q["idx"]] for q in perq_iid[ds]]):
            sl = rec.get("sl")
            if sl is None:
                continue
            n = len(sl); preds = rec["preds"]; sc = rec["sc"]
            pw = int(np.argmax(sc[:n]))
            V = iid_verd[ds]; pf = pf_slot(V.get(q["idx"], {}))
            cp, _ = copeland_pick(n, sc[:n], pf)
            pii_sl.append(dict(idx=q["idx"], pointwise=sl[pw], copeland=sl[cp], present=1 if max(sl) == 1 else 0))
        if pii_sl:
            xref[ds] = dict(n=len(pii_sl),
                            pointwise_sl=round(np.mean([q["pointwise"] for q in pii_sl]), 4),
                            pairwise_sl=round(np.mean([q["copeland"] for q in pii_sl]), 4))
    # pooled sl
    allsl = []
    for ds in FULL_2x2:
        for q, rec in zip(perq_iid[ds], [iid_dump[ds][q["idx"]] for q in perq_iid[ds]]):
            sl = rec.get("sl")
            if sl is None:
                continue
            n = len(sl); sc = rec["sc"]
            pw = int(np.argmax(sc[:n])); pf = pf_slot(iid_verd[ds].get(q["idx"], {}))
            cp, _ = copeland_pick(n, sc[:n], pf)
            allsl.append((sl[pw], sl[cp]))
    if allsl:
        xref["pooled_3ds"] = dict(n=len(allsl),
                                  pointwise_sl=round(float(np.mean([a for a, _ in allsl])), 4),
                                  pairwise_sl=round(float(np.mean([b for _, b in allsl])), 4))
    RESULT["xref_iid_judge_sl"] = dict(
        note="iid@8 cells recomputed with JUDGE sl (not oks) to connect to Experiment 2 (pairwise_verifier_gpu.json). Diverse has no judge labels; primary 2x2 uses oks uniformly.",
        per_dataset=xref)

    # ---- verdict ----
    d_DB = RESULT["pooled_2x2"]["deltas"]["D_minus_B_pairwiseDiv_vs_pointwiseDiv"]
    d_DA = RESULT["pooled_2x2"]["deltas"]["D_minus_A_pairwiseDiv_vs_pointwiseIid"]
    d_CA = RESULT["pooled_2x2"]["deltas"]["C_minus_A_pairwiseIid_vs_pointwiseIid"]
    d_BA = RESULT["pooled_2x2"]["deltas"]["B_minus_A_pointwiseDiv_vs_pointwiseIid"]
    pmc_conv = RESULT["pmc"]["deltas"]["pairwiseDiv_minus_pointwiseDiv"]
    compounds = (d_DB["lo"] > 0) and (d_DA["delta"] >= max(d_CA["delta"], d_BA["delta"]) - 1e-9) and (d_DA["lo"] > 0)
    lines = [
        f"POOLED 2x2 (datasets={FULL_2x2}, n={RESULT['pooled_2x2']['n']}), selection accuracy [oks], 95% CI:",
        f"                 iid@8                         diverse@15",
        f"  pointwise   {A['sel_acc']:.4f} {A['ci95']}      {Bb['sel_acc']:.4f} {Bb['ci95']}",
        f"  pairwise    {Cc['sel_acc']:.4f} {Cc['ci95']}      {Dd['sel_acc']:.4f} {Dd['ci95']}",
        f"  oracle      {cell_block(pooled_iid_3)['oracle']['sel_acc']:.4f}                        {cell_block(pooled_div_3)['oracle']['sel_acc']:.4f}",
        "",
        f"COMPOUNDING deltas (paired bootstrap):",
        f"  pairwise-lever alone   C-A = {d_CA['delta']:+.4f}  CI[{d_CA['lo']:+.4f},{d_CA['hi']:+.4f}]",
        f"  diverse-lever alone    B-A = {d_BA['delta']:+.4f}  CI[{d_BA['lo']:+.4f},{d_BA['hi']:+.4f}]",
        f"  BOTH (pairwise+diverse) D-A = {d_DA['delta']:+.4f}  CI[{d_DA['lo']:+.4f},{d_DA['hi']:+.4f}]",
        f"  pairwise converts diverse  D-B = {d_DB['delta']:+.4f}  CI[{d_DB['lo']:+.4f},{d_DB['hi']:+.4f}]",
        "",
        f"PMC (n={RESULT['pmc']['n']}): oracle iid {RESULT['pmc']['oracle_iid8']['sel_acc']:.3f} -> diverse "
        f"{RESULT['pmc']['oracle_diverse']['sel_acc']:.3f} (lift +{RESULT['pmc']['conversion']['oracle_lift_diverse_minus_iid']:.3f}); "
        f"pointwise-div {RESULT['pmc']['pointwise_diverse']['sel_acc']:.3f} vs pointwise-iid "
        f"{RESULT['pmc']['pointwise_iid8']['sel_acc']:.3f}; pairwise-div "
        f"{RESULT['pmc']['pairwise_diverse']['sel_acc']:.3f}.",
        f"PMC pairwise-div vs pointwise-div: {pmc_conv['delta']:+.4f} CI[{pmc_conv['lo']:+.4f},{pmc_conv['hi']:+.4f}].",
        "",
        ("=> LEVERS COMPOUND: pairwise-over-diverse beats both single levers and the CI on D-B and D-A "
         "excludes 0." if compounds else
         "=> NO clean compounding: pairwise-over-diverse does NOT robustly beat both single levers "
         "(see D-B / D-A CIs)."),
    ]
    RESULT["verdict"] = dict(compounds=bool(compounds), lines=lines)

    for l in lines:
        print(l)

    outp = J("results/cascade_methods/artifacts/combine_diverse_pairwise.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(RESULT, open(outp, "w"), indent=1)
    print(f"\n[dump] {outp}")


if __name__ == "__main__":
    main()
