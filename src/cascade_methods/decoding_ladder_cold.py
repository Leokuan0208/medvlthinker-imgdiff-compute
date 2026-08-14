#!/usr/bin/env python3
"""decoding_ladder_cold.py -- the COLD extension of the temperature ladder (2026-08-14).

Pre-registration: results/cascade_methods/artifacts/_decoding_ladder_cold_prereg.json (written
2026-08-14T07:14:20Z, BEFORE any candidate string of any new rung existed).

What this measures, per temperature rung, on the frozen 2,345-item open pool:
  * distinct-candidate count and the UNANIMOUS fraction (pools with exactly 1 distinct answer)
  * oracle@8, sel_eff, SELECTED -- in BOTH grading currencies, on IDENTICAL verifier picks
  * the COLLAPSE POINT: SELECTED minus a single draw from the SAME pool (the verifier's contribution)
  * the N-interaction: SELECTED at N = 1, 2, 4, 8 by EXACT subsampling of the existing 8-pools

THE N-SUBSAMPLING IS EXACT, NOT MONTE CARLO.  Verifier scores are fixed, so given a subset of N of the
M drawn slots the pick is deterministic.  Rank the slots by (-score, index) -- the frozen first-index
tie-break -- and the slot of rank r is picked iff it is in the subset and none of the r better-ranked
slots are, with probability C(M-1-r, N-1) / C(M, N).  Summing over r gives 1 (hockey-stick identity),
which the script asserts.  E[correct] is then the label-weighted sum.  At N=1 this reduces to the plain
per-item mean of the slot labels, i.e. a single random draw from the same pool -- the baseline the
collapse point is defined against.

T = 0.0 is a special rung: vLLM 0.10.1.1 refuses n>1 with greedy sampling ("n must be 1 when using
greedy sampling, got 8."), so it is generated at n=1, three times with three seeds.  The frozen metric
handles a 1-slot pool correctly on its own (unfilled slots hold MISSING_SCORE, argmax lands on slot 0),
giving oracle = modal = SELECTED and sel_eff = 1.0 by construction.  Nothing is replicated to fake 8
draws; the 3-seed byte-identity rate is reported instead as the determinism measurement.

  python3 src/cascade_methods/decoding_ladder_cold.py
"""
import argparse, json, os, sys
from collections import defaultdict
from math import comb
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G                       # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (                   # noqa: E402
    load_judge, load_vscores, load_pool, boot, DS, SWEEP)

NBOOT, BSEED = 10000, 20260814

# tag -> (temperature, provenance).  Order is COLD -> HOT.
LADDER = [("T00", 0.0, "in_session_NEW (n=1, greedy)"),
          ("T005", 0.05, "in_session_NEW"),
          ("T01", 0.1, "in_session_NEW"),
          ("T02", 0.2, "in_session_NEW"),
          ("T03r", 0.3, "in_session_REGENERATED"),
          ("T04", 0.4, "in_session_NEW"),
          ("T05r", 0.5, "in_session_REGENERATED"),
          ("T07r", 0.7, "in_session_CONTROL")]
# shown for curve continuity only -- generated in the 2026-08-13/14 round, NOT regenerated here.
PRIOR = [("T03", 0.3), ("T05", 0.5), ("T07", 0.7), ("T10", 1.0), ("T13", 1.3)]
CONTROL = "T07r"


# ------------------------------------------------------------------ per-pool measurement
def pick_probs(M, N):
    """P(slot of rank r is the argmax of a uniformly random N-subset of M slots), r = 0..M-1."""
    if N > M:
        return None
    den = comb(M, N)
    p = np.array([comb(M - 1 - r, N - 1) / den if M - 1 - r >= N - 1 else 0.0 for r in range(M)])
    assert abs(p.sum() - 1.0) < 1e-12, f"pick_probs({M},{N}) sums to {p.sum()}"
    return p


def one_seed(tag, lab, vsc, ref):
    """Per-item arrays for one generated pool, or a dict describing why it was refused."""
    pool = load_pool(tag, strict=False)
    if pool is None:
        return {"refused": "pool absent or short"}
    n = len(ref)
    miss_j = miss_v = 0
    Ms, sl_j, sl_e, sc, toks, modal_j, modal_e, nd, golds = [], [], [], [], [], [], [], [], []
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        preds = r["preds"]
        Ms.append(len(preds))
        lj, le, ss = [], list(r["oks_em"]), []
        for a in preds:
            y = lab.get((it["ds"], it["idx"], G.norm(a)))
            if y is None:
                miss_j += 1; y = 0
            lj.append(int(y))
            if (it["ds"], it["idx"], a) not in vsc:
                miss_v += 1
            ss.append(vsc.get((it["ds"], it["idx"], a), G.MISSING_SCORE))
        sl_j.append(lj); sl_e.append(le); sc.append(ss)
        toks.append(list(r.get("gen_tokens_all", [0] * len(preds))))
        mi = preds.index(r["modal_pred"]) if r["modal_pred"] in preds else 0
        modal_j.append(int(lab.get((it["ds"], it["idx"], G.norm(r["modal_pred"])), 0)))
        modal_e.append(int(le[mi]))
        nd.append(len(set(G.norm(a) for a in preds)))
        golds.append(r.get("gold", ""))
    if miss_j or miss_v:
        return {"refused": f"{miss_j} unjudged / {miss_v} unscored slots"}

    # exact E[correct | N] and the pick distribution, per item
    NS = [1, 2, 4, 8]
    got = {c: {N: np.zeros(n) for N in NS} for c in ("judge", "em")}
    orc = {c: {N: np.zeros(n) for N in NS} for c in ("judge", "em")}
    tok_pick = np.zeros(n); pick8 = np.zeros(n, int)
    for i in range(n):
        M = Ms[i]
        order = sorted(range(M), key=lambda k: (-sc[i][k], k))       # frozen first-index tie-break
        lj = np.array([sl_j[i][k] for k in order], float)
        le = np.array([sl_e[i][k] for k in order], float)
        tk = np.array([toks[i][k] for k in order], float)
        kj, ke = int(sum(sl_j[i])), int(sum(sl_e[i]))
        for N in NS:
            Ne = min(N, M)                       # a pool of M draws cannot be subsampled beyond M
            p = pick_probs(M, Ne)
            got["judge"][N][i] = float(p @ lj); got["em"][N][i] = float(p @ le)
            orc["judge"][N][i] = 1.0 - (comb(M - kj, Ne) / comb(M, Ne) if M - kj >= Ne else 0.0)
            orc["em"][N][i] = 1.0 - (comb(M - ke, Ne) / comb(M, Ne) if M - ke >= Ne else 0.0)
        pick8[i] = order[0]
        tok_pick[i] = tk[0]

    ds_index = np.array([DS.index(it["ds"]) for it in ref], int)
    rec_j = np.array([1 if 1 in x else 0 for x in sl_j], int)
    rec_e = np.array([1 if 1 in x else 0 for x in sl_e], int)
    out = {"got": got, "oracle_n": orc, "rec_judge": rec_j, "rec_em": rec_e,
           "modal_judge": np.array(modal_j, float), "modal_em": np.array(modal_e, float),
           "n_distinct": np.array(nd, float), "M": np.array(Ms, int),
           "tok_all_mean": float(np.mean([t for tt in toks for t in tt])),
           "tok_pick": tok_pick, "pick8": pick8, "ds_index": ds_index,
           "golds": golds, "preds_modal_slot": None}
    # cross-check against the frozen metric: N=8 expectation must equal the frozen argmax pick
    fr = G.sel_eff({(ref[i]["ds"], ref[i]["idx"]): sc[i] for i in range(n)},
                   items=[{"ds": ref[i]["ds"], "idx": ref[i]["idx"], "sl": sl_j[i],
                           "preds": pool[(ref[i]["ds"], ref[i]["idx"])]["preds"],
                           "greedy_ok": modal_j[i]} for i in range(n)])
    out["frozen"] = {"selected": fr["acc"], "oracle": fr["oracle"], "sel_eff": fr["sel_eff"],
                     "greedy": fr["greedy"], "contested_n": fr["contested"]["n"],
                     "contested_sel_eff": fr["contested"]["sel_eff"],
                     "per_ds": {d: dict(fr["per_ds"][d]) for d in DS},
                     "identity_residual": abs(fr["acc"] - fr["oracle"] * fr["sel_eff"])}
    out["max_abs_dev_vs_frozen_pick"] = float(np.max(np.abs(got["judge"][8] - fr["got"])))
    out["contested_mask"] = fr["contested_mask"]
    return out


# ------------------------------------------------------------------ main
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/cascade_methods/artifacts/decoding_ladder_cold_2026-08-14.json")
A = ap.parse_args()

print("NULL TEST 1 ...", flush=True)
nt = G.null_test()
r0 = G.sel_eff(G.incumbent_scores())
print(f"  pass={nt['pass']} max_abs_deviation={nt['max_abs_deviation']:.3e} "
      f"identity_residual={abs(r0['acc']-r0['oracle']*r0['sel_eff']):.3e}", flush=True)

lab, vsc, ref = load_judge(), load_vscores(), G.load_items()
LAT = None
try:
    from src.training_methods.visverif_lib import LATERAL
    LAT = np.array([bool(LATERAL.search(str(it.get("gold", "")))) for it in ref], bool)
except Exception as e:
    print("  !! laterality mask unavailable:", e)

def seed_tags(setting):
    return [f"{setting}_s{s}" for s in range(3)
            if os.path.exists(os.path.join(SWEEP, f"ckpt_{DS[0]}_{setting}_s{s}.jsonl"))]

DATA, REFUSED = {}, {}
for tag, temp, prov in LADDER + [(t, v, "prior_round_NOT_regenerated") for t, v in PRIOR]:
    seeds = {}
    for st in seed_tags(tag):
        r = one_seed(st, lab, vsc, ref)
        if "refused" in r:
            REFUSED[st] = r["refused"]; print(f"  [refused] {st}: {r['refused']}", flush=True)
            continue
        seeds[st] = r
    if seeds:
        DATA[tag] = seeds
        print(f"  [ok] {tag}: {len(seeds)} seeds", flush=True)

if CONTROL not in DATA:
    sys.exit(f"CONTROL {CONTROL} not available -- refusing to report deltas against anything else")

TEMP = dict([(t, v) for t, v, _ in LADDER] + PRIOR)
PROV = dict([(t, p) for t, _, p in LADDER] + [(t, "prior_round_NOT_regenerated") for t, _ in PRIOR])


def avg(seeds, key, sub=None, N=None):
    if sub is None:
        return np.mean([v[key] for v in seeds.values()], axis=0)
    return np.mean([v[key][sub][N] for v in seeds.values()], axis=0)


out = {"title": "COLD extension of the decoding temperature ladder -- where the peak is, "
                "where the verifier goes inert, and what that costs",
       "date": "2026-08-14",
       "prereg": "results/cascade_methods/artifacts/_decoding_ladder_cold_prereg.json "
                 "(written 2026-08-14T07:14:20Z, before any rung existed)",
       "generator": "src/cascade_methods/decoding_sweep_gen.py (unchanged) over "
                    "artifacts/_decoding_ladder_cold_settings{,_T00}.json",
       "analysis": "src/cascade_methods/decoding_ladder_cold.py",
       "endpoint": "SELECTED accuracy on the frozen 2,345-item open pool (slake_open 645 / "
                   "vqa_rad_open 200 / pathvqa_open 1500), frozen incumbent LoRA verifier "
                   "(ckpts/train/lora_verifier_disjoint) scored under HF transformers, batch 1.",
       "control": f"{CONTROL} = T 0.7, the deployed setting, REGENERATED IN THIS SESSION. Every delta "
                  "is against it, never against a stored number.",
       "held_fixed": {"prompt": "SYS verbatim from src/labeling/run_openvqa.py", "image_cap": "cap320 "
                      "(max_pixels 250880)", "N": 8, "max_tokens": 64, "top_p": 1.0, "top_k": -1,
                      "min_p": 0.0, "repetition_penalty": 1.0,
                      "generation_seeds": [20260813, 20261813, 20262813]},
       "NULL_TEST_1": {"pass": bool(nt["pass"]), "max_abs_deviation": nt["max_abs_deviation"],
                       "measured": nt["measured"],
                       "identity_residual": abs(r0["acc"] - r0["oracle"] * r0["sel_eff"])},
       "nboot": NBOOT, "bootstrap_seed": BSEED,
       "pools_refused": REFUSED,
       "LADDER": {}}

for tag, seeds in DATA.items():
    gj8 = avg(seeds, "got", "judge", 8); ge8 = avg(seeds, "got", "em", 8)
    gj1 = avg(seeds, "got", "judge", 1); ge1 = avg(seeds, "got", "em", 1)
    mj = avg(seeds, "modal_judge"); me = avg(seeds, "modal_em")
    rj = avg(seeds, "rec_judge"); re_ = avg(seeds, "rec_em")
    b = {"temperature": TEMP[tag], "provenance": PROV[tag], "n_seeds": len(seeds),
         "seeds": sorted(seeds),
         "pool_size_M_mean": float(np.mean([v["M"].mean() for v in seeds.values()])),
         "distinct_of_8": float(np.mean([v["n_distinct"].mean() for v in seeds.values()])),
         "frac_items_1_distinct_UNANIMOUS": float(np.mean([(v["n_distinct"] == 1).mean()
                                                           for v in seeds.values()])),
         "mean_gen_tokens_all_slots": float(np.mean([v["tok_all_mean"] for v in seeds.values()])),
         "mean_gen_tokens_PICKED_slot": float(np.mean([v["tok_pick"].mean() for v in seeds.values()])),
         "oracle@8_judge": float(np.mean([v["frozen"]["oracle"] for v in seeds.values()])),
         "oracle@8_em": float(np.mean([v["rec_em"].mean() for v in seeds.values()])),
         "sel_eff_judge": float(np.mean([v["frozen"]["sel_eff"] for v in seeds.values()])),
         "sel_eff_em": float(np.mean([v["got"]["em"][8].sum() / max(v["rec_em"].sum(), 1)
                                      for v in seeds.values()])),
         "SELECTED_judge": float(gj8.mean()), "SELECTED_em": float(ge8.mean()),
         "SELECTED_judge_seed_sd": float(np.std([v["got"]["judge"][8].mean() for v in seeds.values()],
                                                ddof=1)) if len(seeds) > 1 else 0.0,
         "SELECTED_em_seed_sd": float(np.std([v["got"]["em"][8].mean() for v in seeds.values()],
                                             ddof=1)) if len(seeds) > 1 else 0.0,
         "MODAL_VOTE_judge": float(mj.mean()), "MODAL_VOTE_em": float(me.mean()),
         "RANDOM_SLOT_judge": float(gj1.mean()), "RANDOM_SLOT_em": float(ge1.mean()),
         "identity_residual_selected_eq_oracle_x_sel_eff":
             float(np.mean([v["frozen"]["identity_residual"] for v in seeds.values()])),
         "max_abs_dev_exact_N8_expectation_vs_frozen_argmax_pick":
             float(np.max([v["max_abs_dev_vs_frozen_pick"] for v in seeds.values()])),
         "per_cell": {}, }
    for j, d in enumerate(DS):
        m = seeds[sorted(seeds)[0]]["ds_index"] == j
        b["per_cell"][d] = {"n": int(m.sum()),
                            "SELECTED_judge": float(gj8[m].mean()), "SELECTED_em": float(ge8[m].mean()),
                            "oracle@8_judge": float(rj[m].mean()),
                            "sel_eff_judge": float(np.mean([v["frozen"]["per_ds"][d]["sel_eff"]
                                                            for v in seeds.values()]))}
    b["SELECTED_at_N"] = {c: {str(N): float(avg(seeds, "got", c, N).mean()) for N in (1, 2, 4, 8)}
                          for c in ("judge", "em")}
    b["oracle_at_N"] = {c: {str(N): float(avg(seeds, "oracle_n", c, N).mean()) for N in (1, 2, 4, 8)}
                        for c in ("judge", "em")}
    out["LADDER"][tag] = b

# ---------------------------------------------------------------- deltas vs the in-session control
C = DATA[CONTROL]
cj = avg(C, "got", "judge", 8); ce = avg(C, "got", "em", 8)
con_mask = np.mean([v["contested_mask"] for v in C.values()], axis=0) >= 0.5
out["contested_stratum_definition"] = ("recoverable under the CONTROL pool AND >=2 distinct candidate "
                                       "strings in the CONTROL pool -- fixed by the control so the "
                                       "same items are compared across rungs")
out["contested_n"] = int(con_mask.sum())
out["laterality_n"] = int(LAT.sum()) if LAT is not None else None
out["DELTAS_VS_CONTROL"] = {}
for tag, seeds in DATA.items():
    if tag == CONTROL:
        continue
    sj = avg(seeds, "got", "judge", 8); se = avg(seeds, "got", "em", 8)
    dj, de = boot(sj, cj, nboot=NBOOT, seed=BSEED), boot(se, ce, nboot=NBOOT, seed=BSEED)
    row = {"temperature": TEMP[tag], "provenance": PROV[tag],
           "judge": dj, "em": de,
           "CI_clean_in_BOTH": bool(dj["verdict"] == de["verdict"] and dj["sig"] and de["sig"]),
           "currency_conflict": bool((dj["verdict"] == "WIN" and de["verdict"] == "LOSS")
                                     or (dj["verdict"] == "LOSS" and de["verdict"] == "WIN")),
           "contested": {"judge": boot(sj, cj, mask=con_mask, nboot=NBOOT, seed=BSEED),
                         "em": boot(se, ce, mask=con_mask, nboot=NBOOT, seed=BSEED)},
           "per_cell_guardrail": {}}
    if LAT is not None and LAT.sum() > 0:
        row["laterality"] = {"judge": boot(sj, cj, mask=LAT, nboot=NBOOT, seed=BSEED),
                             "em": boot(se, ce, mask=LAT, nboot=NBOOT, seed=BSEED)}
    dsi = seeds[sorted(seeds)[0]]["ds_index"]
    for j, d in enumerate(DS):
        m = dsi == j
        row["per_cell_guardrail"][d] = {"judge": boot(sj, cj, mask=m, nboot=NBOOT, seed=BSEED),
                                        "em": boot(se, ce, mask=m, nboot=NBOOT, seed=BSEED)}
    row["guardrail_flags"] = [d for d in DS if row["per_cell_guardrail"][d]["judge"]["verdict"] == "LOSS"
                              or row["per_cell_guardrail"][d]["em"]["verdict"] == "LOSS"]
    out["DELTAS_VS_CONTROL"][tag] = row

# ---------------------------------------------------------------- the collapse point
out["COLLAPSE_POINT"] = {
    "definition_primary": "VC_single(T) = SELECTED(T) - RANDOM_SLOT(T): what best-of-8 + the frozen "
                          "verifier buys over ONE draw from the SAME pool at the SAME temperature "
                          "(exact expectation, not a sampled draw). Pre-registered as the primary "
                          "statistic before any rung was generated.",
    "definition_secondary": "VC_modal(T) = SELECTED(T) - MODAL_VOTE(T): the harder bar, against "
                            "majority vote over the same pool, which is itself a training-free selector.",
    "per_rung": {}}
for tag, seeds in DATA.items():
    sj8 = avg(seeds, "got", "judge", 8); se8 = avg(seeds, "got", "em", 8)
    sj1 = avg(seeds, "got", "judge", 1); se1 = avg(seeds, "got", "em", 1)
    mj = avg(seeds, "modal_judge"); me = avg(seeds, "modal_em")
    out["COLLAPSE_POINT"]["per_rung"][tag] = {
        "temperature": TEMP[tag],
        "distinct_of_8": out["LADDER"][tag]["distinct_of_8"],
        "frac_items_1_distinct": out["LADDER"][tag]["frac_items_1_distinct_UNANIMOUS"],
        "VC_single_judge": boot(sj8, sj1, nboot=NBOOT, seed=BSEED),
        "VC_single_em": boot(se8, se1, nboot=NBOOT, seed=BSEED),
        "VC_modal_judge": boot(sj8, mj, nboot=NBOOT, seed=BSEED),
        "VC_modal_em": boot(se8, me, nboot=NBOOT, seed=BSEED)}

# ---------------------------------------------------------------- N interaction vs the deployed arm
cn8j, cn8e = float(cj.mean()), float(ce.mean())
out["N_INTERACTION"] = {
    "question": "does a COLD pool at N=4 (or 2) match the DEPLOYED hot pool at N=8? If so the "
                "generation cost halves (or quarters) at parity.",
    "method": "EXACT subsampling of the already-generated 8-sample pools: given N of the M slots, the "
              "verifier's pick is deterministic, so E[correct] is a closed-form weighted sum over the "
              "slot ranks. No new generation, no Monte Carlo. Costs are per-question generated-token "
              "counts x N; prefill is charged once per sample because each sample re-reads the prompt.",
    "deployed_reference": {"setting": CONTROL, "N": 8, "SELECTED_judge": cn8j, "SELECTED_em": cn8e},
    "rows": []}
for tag, seeds in DATA.items():
    for N in (1, 2, 4, 8):
        sj = avg(seeds, "got", "judge", N); se = avg(seeds, "got", "em", N)
        dj, de = boot(sj, cj, nboot=NBOOT, seed=BSEED), boot(se, ce, nboot=NBOOT, seed=BSEED)
        out["N_INTERACTION"]["rows"].append({
            "setting": tag, "temperature": TEMP[tag], "N": N,
            "SELECTED_judge": float(sj.mean()), "SELECTED_em": float(se.mean()),
            "oracle_judge": float(avg(seeds, "oracle_n", "judge", N).mean()),
            "vs_deployed_N8_judge": dj, "vs_deployed_N8_em": de,
            "matches_deployed_in_BOTH": bool(dj["verdict"] != "LOSS" and de["verdict"] != "LOSS"),
            "gen_token_ratio_vs_deployed_N8": float(
                out["LADDER"][tag]["mean_gen_tokens_all_slots"] * min(N, out["LADDER"][tag]["pool_size_M_mean"])
                / (out["LADDER"][CONTROL]["mean_gen_tokens_all_slots"] * 8))})

os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1, default=float)
print(f"\nwrote {A.out}")

# ---------------------------------------------------------------- console table
print(f"\n{'rung':7s} {'T':>5s} {'sd':>4s} {'dist/8':>7s} {'unan':>6s} {'orc_j':>7s} {'seff_j':>7s} "
      f"{'SEL_j':>8s} {'SEL_em':>8s} {'modal_j':>8s} {'rand_j':>7s} {'tok':>6s}")
for tag in sorted(out["LADDER"], key=lambda t: (TEMP[t], t)):
    b = out["LADDER"][tag]
    print(f"{tag:7s} {b['temperature']:5.2f} {b['n_seeds']:4d} {b['distinct_of_8']:7.3f} "
          f"{b['frac_items_1_distinct_UNANIMOUS']:6.3f} {b['oracle@8_judge']:7.4f} "
          f"{b['sel_eff_judge']:7.4f} {b['SELECTED_judge']:8.5f} {b['SELECTED_em']:8.5f} "
          f"{b['MODAL_VOTE_judge']:8.5f} {b['RANDOM_SLOT_judge']:7.4f} "
          f"{b['mean_gen_tokens_all_slots']:6.2f}")
print(f"\nvs in-session control {CONTROL}:")
for tag in sorted(out["DELTAS_VS_CONTROL"], key=lambda t: (TEMP[t], t)):
    r = out["DELTAS_VS_CONTROL"][tag]
    j, e = r["judge"], r["em"]
    print(f"  {tag:7s} T={r['temperature']:4.2f}  judge {j['delta']:+.5f} [{j['lo']:+.5f},{j['hi']:+.5f}] "
          f"{j['verdict']:5s} | em {e['delta']:+.5f} [{e['lo']:+.5f},{e['hi']:+.5f}] {e['verdict']:5s}"
          + ("  <<< CURRENCY CONFLICT" if r["currency_conflict"] else "")
          + (f"  guardrail:{r['guardrail_flags']}" if r["guardrail_flags"] else ""))
print("\nCOLLAPSE (SELECTED - one draw from the same pool):")
for tag in sorted(out["COLLAPSE_POINT"]["per_rung"], key=lambda t: (TEMP[t], t)):
    r = out["COLLAPSE_POINT"]["per_rung"][tag]
    a, b_ = r["VC_single_judge"], r["VC_single_em"]
    print(f"  {tag:7s} T={r['temperature']:4.2f} dist={r['distinct_of_8']:5.2f} "
          f"judge {a['delta']:+.5f} [{a['lo']:+.5f},{a['hi']:+.5f}] {a['verdict']:5s} | "
          f"em {b_['delta']:+.5f} [{b_['lo']:+.5f},{b_['hi']:+.5f}] {b_['verdict']:5s}")
