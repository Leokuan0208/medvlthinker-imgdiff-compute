#!/usr/bin/env python3
"""
diverse_measure_gpu.py -- OFFLINE measure for EXPERIMENT 1 (diverse GENERATION vs iid@8).

Compares, on EXACTLY the same questions/model/cap/verifier (matched design):
  * iid@8  : existing ckpts/mcq_gen_verify/lingshu7b/ckpt_<DS>_..._content_sc8.jsonl
             (Lingshu-7B, cap320, temp0.7, 8 iid samples, exact-match oks, pointwise pooled4 scores)
  * diverse@M : ckpts/openvqa/diverse/ckpt_<ds_open>_lingshu7b_div.jsonl produced by
             diversity_generate_gpu.py (5-prompt x 3-temp portfolio, M=15; SAME model/cap/verifier/scorer)

The diverse generator was RESTRICTED to the iid dump's idx set, so the two arms are on the SAME
questions.  Both arms' `oks` use the IDENTICAL map_correct scorer, both `scores` use the SAME pooled4
pointwise verifier -> apples-to-apples.

Reports, per dataset + pooled-over-datasets, the 3 things asked for (honestly):
  (A) ORACLE@N lift  : does diverse gen RAISE the coverage ceiling?
        oracle@8 iid  vs  oracle@8 diverse-DPP-8 (SAME budget N=8)  vs  oracle@M diverse-full (bigger budget)
  (B) VERIFIER best-of-N selection accuracy: does the lift CONVERT through the verifier?
        pick = argmax(pointwise score); sel_acc = mean sl[pick]; sel_eff = P(pick correct | correct present)
        iid  vs  diverse-DPP-8  vs  diverse-full
  (C) DISTRACTOR INJECTION: diverse gen can add confident-WRONG candidates the verifier mis-picks.
        - new-coverage conversion: on Qs where diverse newly covers a correct answer (iid missed), does
          the verifier actually pick it?
        - confident-distractor rate: P(verifier's argmax is wrong | a correct candidate exists), iid vs diverse.
  (D) COST (both axes): generation samples (8 vs M) and verify calls (8 vs M) per question.

Bootstrap 95% CIs (resample questions) on the headline deltas.  CPU only.  No fabricated numbers.
Launch from repo root:  python3 src/cascade_methods/diverse_measure_gpu.py
Writes: results/cascade_methods/artifacts/diverse_generation_gpu.json
"""
import os, json, importlib.util
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)

# reuse the generator's DPP farthest-first selection (identical distance metric as generation)
_spec = importlib.util.spec_from_file_location("dgg", J("src/cascade_methods/diversity_generate_gpu.py"))
_dgg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dgg)
farthest_first_select = _dgg.farthest_first_select

# ds_open (diverse ckpt) -> iid mcq dump
PAIRS = {
    "vqa_rad_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_VQA_RAD_lingshu7b_VQA_RAD_content_content_sc8.jsonl",
    "slake_open":   "ckpts/mcq_gen_verify/lingshu7b/ckpt_SLAKE_lingshu7b_SLAKE_content_content_sc8.jsonl",
    "pathvqa_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_PATH_VQA_lingshu7b_content_sc8.jsonl",
    "pmc_content":  "ckpts/mcq_gen_verify/lingshu7b/ckpt_PMC_VQA_lingshu7b_content_sc8.jsonl",
}
DIV_DIR = "ckpts/openvqa/diverse"
SEL_N = 8   # DPP down-selection budget (matched to iid N=8)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def oracle_curve(list_of_ok, maxk):
    """mean over questions of [any(ok[:k]) for k=1..maxk]. Questions shorter than k contribute their max."""
    cur = []
    for k in range(1, maxk + 1):
        cur.append(float(np.mean([1 if any(o[:k]) else 0 for o in list_of_ok])))
    return cur


def bo_n(rows_scores, rows_oks):
    """verifier best-of-N: pick=argmax(scores); return sel_acc, sel_eff, present, confident_distractor_rate."""
    accs, effs, cdist = [], [], []
    for sc, ok in zip(rows_scores, rows_oks):
        j = int(np.argmax(sc))
        accs.append(ok[j])
        if max(ok) == 1:
            effs.append(ok[j])
            cdist.append(1 if ok[j] == 0 else 0)   # correct existed but verifier picked wrong
    return (float(np.mean(accs)),
            float(np.mean(effs)) if effs else float("nan"),
            len(effs),
            float(np.mean(cdist)) if cdist else float("nan"))


def boot_delta(vals_a, vals_b, n_boot=2000, seed=0):
    """95% CI of mean(a)-mean(b) over PAIRED per-question values (resample questions)."""
    a = np.asarray(vals_a, float); b = np.asarray(vals_b, float)
    rng = np.random.default_rng(seed); n = len(a); d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d.append(a[idx].mean() - b[idx].mean())
    return float(np.mean(a) - np.mean(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyze_ds(ds, iid_path):
    div_path = os.path.join(DIV_DIR, f"ckpt_{ds}_lingshu7b_div.jsonl")
    if not (os.path.exists(div_path) and os.path.exists(J(iid_path))):
        return {"status": f"missing (div={os.path.exists(div_path)}, iid={os.path.exists(J(iid_path))})"}
    div = {str(r["idx"]): r for r in load_jsonl(div_path)}
    iid = {str(r["idx"]): r for r in load_jsonl(J(iid_path))}
    common = sorted(set(div) & set(iid))
    if not common:
        return {"status": "no common idx"}

    iid_oks, iid_sc, div_oks, div_sc = [], [], [], []
    dpp_oks, dpp_sc = [], []
    for k in common:
        ri, rd = iid[k], div[k]
        io = [int(x) for x in ri["oks"]][:8]; isc = list(ri["scores"])[:8]
        do = [int(x) for x in rd["oks"]];      dsc = list(rd["scores"])
        if len(io) < 1 or len(do) < 1 or isc is None or dsc is None: continue
        iid_oks.append(io); iid_sc.append(isc)
        div_oks.append(do); div_sc.append(dsc)
        sel = farthest_first_select(rd["preds"], SEL_N)      # DPP down-select N-of-M (answer-string distance)
        dpp_oks.append([do[i] for i in sel]); dpp_sc.append([dsc[i] for i in sel])

    n = len(iid_oks)
    M = int(np.median([len(o) for o in div_oks]))
    # (A) oracle curves
    or_iid = oracle_curve(iid_oks, 8)
    or_div = oracle_curve(div_oks, max(len(o) for o in div_oks))
    or_dpp = oracle_curve(dpp_oks, SEL_N)
    oracle8_iid = or_iid[-1]
    oracle8_dpp = or_dpp[-1]
    oracleM_div = or_div[-1]
    # per-question oracle indicators (for bootstrap) at matched budget 8 and full
    q_or8_iid = [1 if any(o[:8]) else 0 for o in iid_oks]
    q_or8_dpp = [1 if any(o) else 0 for o in dpp_oks]         # dpp lists have <=8
    q_orM_div = [1 if any(o) else 0 for o in div_oks]

    # (B) verifier best-of-N
    acc_iid, eff_iid, pres_iid, cd_iid = bo_n(iid_sc, iid_oks)
    acc_div, eff_div, pres_div, cd_div = bo_n(div_sc, div_oks)
    acc_dpp, eff_dpp, pres_dpp, cd_dpp = bo_n(dpp_sc, dpp_oks)
    # per-question bo-n correctness (for bootstrap)
    q_bo_iid = [io[int(np.argmax(isc))] for io, isc in zip(iid_oks, iid_sc)]
    q_bo_div = [do[int(np.argmax(dsc))] for do, dsc in zip(div_oks, div_sc)]
    q_bo_dpp = [do[int(np.argmax(dsc))] for do, dsc in zip(dpp_oks, dpp_sc)]

    # (C) distractor injection / new-coverage conversion
    new_cov, new_cov_conv = 0, 0        # diverse newly covers (iid missed) -> did verifier pick correct?
    lost_cov = 0                         # iid covered but diverse-full missed (coverage regression)
    for io, isc, do, dsc in zip(iid_oks, iid_sc, div_oks, div_sc):
        iid_hit = any(io[:8]); div_hit = any(do)
        if div_hit and not iid_hit:
            new_cov += 1
            if do[int(np.argmax(dsc))] == 1: new_cov_conv += 1
        if iid_hit and not div_hit:
            lost_cov += 1

    # bootstrap CIs on headline deltas
    d_or8, lo_or8, hi_or8 = boot_delta(q_or8_dpp, q_or8_iid)            # oracle@8 diverse-DPP vs iid
    d_orM, lo_orM, hi_orM = boot_delta(q_orM_div, q_or8_iid)           # oracle@M diverse vs iid@8
    d_bo_dpp, lo_bd, hi_bd = boot_delta(q_bo_dpp, q_bo_iid)            # bo-N diverse-DPP vs iid
    d_bo_div, lo_bv, hi_bv = boot_delta(q_bo_div, q_bo_iid)            # bo-N diverse-full vs iid

    return {
        "n_common": n, "M_diverse": M,
        "oracle": {
            "iid@8": round(oracle8_iid, 4), "diverse_dpp@8": round(oracle8_dpp, 4),
            "diverse_full@M": round(oracleM_div, 4),
            "curve_iid": [round(x, 4) for x in or_iid],
            "curve_diverse_full": [round(x, 4) for x in or_div],
            "curve_diverse_dpp8": [round(x, 4) for x in or_dpp],
        },
        "oracle_lift": {
            "dpp8_minus_iid8": round(d_or8, 4), "ci95": [round(lo_or8, 4), round(hi_or8, 4)],
            "fullM_minus_iid8": round(d_orM, 4), "ci95_full": [round(lo_orM, 4), round(hi_orM, 4)],
        },
        "verifier_bo_n": {
            "iid@8":        {"sel_acc": round(acc_iid, 4), "sel_eff": round(eff_iid, 4), "n_present": pres_iid,
                             "confident_distractor_rate": round(cd_iid, 4)},
            "diverse_dpp@8":{"sel_acc": round(acc_dpp, 4), "sel_eff": round(eff_dpp, 4), "n_present": pres_dpp,
                             "confident_distractor_rate": round(cd_dpp, 4)},
            "diverse_full@M":{"sel_acc": round(acc_div, 4), "sel_eff": round(eff_div, 4), "n_present": pres_div,
                             "confident_distractor_rate": round(cd_div, 4)},
            "delta_dpp8_minus_iid":  {"sel_acc": round(d_bo_dpp, 4), "ci95": [round(lo_bd, 4), round(hi_bd, 4)]},
            "delta_fullM_minus_iid": {"sel_acc": round(d_bo_div, 4), "ci95": [round(lo_bv, 4), round(hi_bv, 4)]},
        },
        "distractor_injection": {
            "n_new_coverage": new_cov,
            "n_new_coverage_verifier_converted": new_cov_conv,
            "new_coverage_conversion_rate": round(new_cov_conv / new_cov, 4) if new_cov else None,
            "n_lost_coverage_diverse_vs_iid8": lost_cov,
        },
        "cost": {"gen_samples_iid": 8, "gen_samples_diverse": M,
                 "verify_calls_iid": 8, "verify_calls_diverse": M,
                 "extra_gen_factor": round(M / 8.0, 2)},
        # keep the raw per-question arrays needed to re-pool without reloading
        "_pool": {"q_or8_iid": q_or8_iid, "q_or8_dpp": q_or8_dpp, "q_orM_div": q_orM_div,
                  "q_bo_iid": q_bo_iid, "q_bo_dpp": q_bo_dpp, "q_bo_div": q_bo_div,
                  "iid_oks": iid_oks, "iid_sc": iid_sc, "div_oks": div_oks, "div_sc": div_sc,
                  "dpp_oks": dpp_oks, "dpp_sc": dpp_sc},
    }


def pooled(per_ds):
    """Pool per-question arrays across datasets and recompute the headline numbers + CIs."""
    keys = ["q_or8_iid", "q_or8_dpp", "q_orM_div", "q_bo_iid", "q_bo_dpp", "q_bo_div"]
    agg = {k: [] for k in keys}
    iid_oks = []; iid_sc = []; div_oks = []; div_sc = []; dpp_oks = []; dpp_sc = []
    for ds, r in per_ds.items():
        if "_pool" not in r: continue
        for k in keys: agg[k].extend(r["_pool"][k])
        iid_oks += r["_pool"]["iid_oks"]; iid_sc += r["_pool"]["iid_sc"]
        div_oks += r["_pool"]["div_oks"]; div_sc += r["_pool"]["div_sc"]
        dpp_oks += r["_pool"]["dpp_oks"]; dpp_sc += r["_pool"]["dpp_sc"]
    n = len(agg["q_or8_iid"])
    if n == 0: return {"status": "no data"}
    acc_iid, eff_iid, pres_iid, cd_iid = bo_n(iid_sc, iid_oks)
    acc_div, eff_div, pres_div, cd_div = bo_n(div_sc, div_oks)
    acc_dpp, eff_dpp, pres_dpp, cd_dpp = bo_n(dpp_sc, dpp_oks)
    d_or8, lo8, hi8 = boot_delta(agg["q_or8_dpp"], agg["q_or8_iid"])
    d_orM, loM, hiM = boot_delta(agg["q_orM_div"], agg["q_or8_iid"])
    d_bd, lbd, hbd = boot_delta(agg["q_bo_dpp"], agg["q_bo_iid"])
    d_bv, lbv, hbv = boot_delta(agg["q_bo_div"], agg["q_bo_iid"])
    return {
        "n_common": n,
        "oracle": {"iid@8": round(np.mean(agg["q_or8_iid"]), 4),
                   "diverse_dpp@8": round(np.mean(agg["q_or8_dpp"]), 4),
                   "diverse_full@M": round(np.mean(agg["q_orM_div"]), 4)},
        "oracle_lift": {"dpp8_minus_iid8": round(d_or8, 4), "ci95": [round(lo8, 4), round(hi8, 4)],
                        "fullM_minus_iid8": round(d_orM, 4), "ci95_full": [round(loM, 4), round(hiM, 4)]},
        "verifier_bo_n": {
            "iid@8":        {"sel_acc": round(acc_iid, 4), "sel_eff": round(eff_iid, 4),
                             "confident_distractor_rate": round(cd_iid, 4)},
            "diverse_dpp@8":{"sel_acc": round(acc_dpp, 4), "sel_eff": round(eff_dpp, 4),
                             "confident_distractor_rate": round(cd_dpp, 4)},
            "diverse_full@M":{"sel_acc": round(acc_div, 4), "sel_eff": round(eff_div, 4),
                             "confident_distractor_rate": round(cd_div, 4)},
            "delta_dpp8_minus_iid":  {"sel_acc": round(d_bd, 4), "ci95": [round(lbd, 4), round(hbd, 4)]},
            "delta_fullM_minus_iid": {"sel_acc": round(d_bv, 4), "ci95": [round(lbv, 4), round(hbv, 4)]},
        },
    }


def verdict(pool):
    L = []
    if "oracle" not in pool:
        return ["no data"]
    o = pool["oracle"]; ol = pool["oracle_lift"]; v = pool["verifier_bo_n"]
    L.append(f"POOLED n={pool['n_common']}")
    L.append(f"ORACLE  iid@8={o['iid@8']:.3f}  diverse-DPP@8={o['diverse_dpp@8']:.3f} "
             f"(Δ{ol['dpp8_minus_iid8']:+.3f} CI{ol['ci95']})  diverse-full@M={o['diverse_full@M']:.3f} "
             f"(Δ{ol['fullM_minus_iid8']:+.3f} CI{ol['ci95_full']})")
    L.append(f"VERIF   iid bo-8 acc={v['iid@8']['sel_acc']:.3f} eff={v['iid@8']['sel_eff']:.3f}  | "
             f"diverse-DPP acc={v['diverse_dpp@8']['sel_acc']:.3f} (Δ{v['delta_dpp8_minus_iid']['sel_acc']:+.3f} "
             f"CI{v['delta_dpp8_minus_iid']['ci95']})  | diverse-full acc={v['diverse_full@M']['sel_acc']:.3f} "
             f"(Δ{v['delta_fullM_minus_iid']['sel_acc']:+.3f} CI{v['delta_fullM_minus_iid']['ci95']})")
    L.append(f"DISTRACTOR  confident-distractor-rate  iid={v['iid@8']['confident_distractor_rate']:.3f}  "
             f"diverse-full={v['diverse_full@M']['confident_distractor_rate']:.3f}")
    oracle_up = ol["fullM_minus_iid8"] > 0 and ol["ci95_full"][0] > 0
    dpp_up = ol["dpp8_minus_iid8"] > 0 and ol["ci95"][0] > 0
    conv = v["delta_fullM_minus_iid"]["sel_acc"] > 0 and v["delta_fullM_minus_iid"]["ci95"][0] > 0
    L.append("")
    L.append(f"=> oracle@8 (matched budget) raised by diverse gen: {'YES' if dpp_up else 'NO (CI incl 0)'}")
    L.append(f"=> oracle@M (extra budget) raised vs iid@8: {'YES' if oracle_up else 'NO (CI incl 0)'}")
    L.append(f"=> lift CONVERTS to verifier selection accuracy: {'YES' if conv else 'NO (verifier does not convert it / distractors offset)'}")
    return L


def main():
    print("=" * 100)
    print("EXPERIMENT 1  DIVERSE GENERATION vs iid@8  (matched: Lingshu-7B / cap320 / pooled4 verifier)")
    print("=" * 100)
    per_ds = {}
    for ds, iid in PAIRS.items():
        r = analyze_ds(ds, iid)
        per_ds[ds] = r
        if "oracle" in r:
            o = r["oracle"]; ol = r["oracle_lift"]; v = r["verifier_bo_n"]; di = r["distractor_injection"]
            print(f"\n### {ds}  n={r['n_common']}  M={r['M_diverse']}")
            print(f"  oracle:  iid@8={o['iid@8']:.3f}  diverse-DPP@8={o['diverse_dpp@8']:.3f} "
                  f"(Δ{ol['dpp8_minus_iid8']:+.3f} CI{ol['ci95']})  diverse-full@M={o['diverse_full@M']:.3f} "
                  f"(Δ{ol['fullM_minus_iid8']:+.3f})")
            print(f"  verif bo-N acc:  iid={v['iid@8']['sel_acc']:.3f}  dpp8={v['diverse_dpp@8']['sel_acc']:.3f} "
                  f"full={v['diverse_full@M']['sel_acc']:.3f}   eff: iid={v['iid@8']['sel_eff']:.3f} "
                  f"full={v['diverse_full@M']['sel_eff']:.3f}")
            print(f"  distractor: new-coverage Qs={di['n_new_coverage']} converted={di['n_new_coverage_verifier_converted']} "
                  f"({di['new_coverage_conversion_rate']}) | lost-coverage={di['n_lost_coverage_diverse_vs_iid8']}")
        else:
            print(f"\n### {ds}  {r.get('status')}")
    pool = pooled({k: v for k, v in per_ds.items() if "_pool" in v})
    vlines = verdict(pool)
    print("\n" + "=" * 100 + "\nVERDICT\n" + "=" * 100)
    for l in vlines: print("  " + l)

    # strip heavy per-question arrays before dumping (keep the analysis compact)
    for r in per_ds.values():
        r.pop("_pool", None)
    out = {"meta": {"design": "matched iid@8 (mcq_gen_verify lingshu7b) vs diverse portfolio M=15 "
                    "(diversity_generate_gpu), same model/cap/verifier/scorer; diverse restricted to iid idx set",
                    "select_n_dpp": SEL_N, "portfolio": _dgg.PROMPT_PORTFOLIO and [p[0] for p in _dgg.PROMPT_PORTFOLIO],
                    "temp_ladder": _dgg.TEMP_LADDER},
           "per_dataset": per_ds, "pooled": pool, "verdict": vlines}
    outp = J("results/cascade_methods/artifacts/diverse_generation_gpu.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1)
    print(f"\n[dump] {outp}")


if __name__ == "__main__":
    main()
