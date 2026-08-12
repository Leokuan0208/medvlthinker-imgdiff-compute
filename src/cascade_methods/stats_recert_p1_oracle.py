#!/usr/bin/env python3
"""
ATTACK C / PART 1 -- THE ORACLE GAP NET OF RESAMPLING AND LABEL NOISE.

WHY.  LITERATURE_UPDATE_2026-08-11.md cites a 2026 result (arXiv:2607.03436) measuring that 12-36%
of a raw oracle@N gap is unharvestable label / resampling noise: an item is counted "recoverable"
only because one lucky sample happened to match a noisy gold.  EVERY headroom claim this project
makes is built on the RAW oracle -- the coverage wall (37.4% of questions have no correct answer
anywhere in the 8-sample pool) and the perfect-selection ceiling (+0.0301 macro).  If the
correction applies to us, the denominator of every headroom statement shrinks.

WHAT IS ON DISK THAT MAKES THIS MEASURABLE WITH ZERO GPU.
  pool A  the deployed 8-sample eval pool (ckpts/train/lora_verifier_disjoint/transfer_dump_*.json).
          Verified byte-identical to ckpts/openvqa/cheap_lingshu7b/ckpt_*_sc8.jsonl.
  pool B  ckpt_*_sc16.jsonl -- an INDEPENDENT 16-sample re-generation of the SAME eval items
          (checked: its first 8 predictions differ from pool A's on 340/645, 168/200 and
          1360/1500 items, so it is a fresh sampling run, not a superset).
  pool C  ckpt_vqa_rad_open_lingshu7b_sc32.jsonl -- an independent 32-sample run, vqa_rad_open only.
Each pool carries its OWN separate LLM-judge pass (`*_scexploded.judge.jsonl`), keyed by the
normalised prediction string.  So where pools A and B both judged the SAME (item, answer string),
we have two independent draws of the judge on identical input -- a direct label-noise measurement.

THE TWO CORRECTIONS, REPORTED SEPARATELY BECAUSE THEY ARE DIFFERENT THINGS.
  (1) LABEL noise   -- does an independent judge pass confirm the label that made this item
                       "recoverable"?  Measured on the exact population the correction is about:
                       the LONE correct sample of a k=1 item.
  (2) SAMPLING noise -- would a FRESH pool of 8 still contain a correct answer?  Computed EXACTLY
                       (hypergeometric over pool B's 16 labels), never simulated.

NULL TEST.  Rebuilding pool A's labels from its exploded judge file must reproduce the deployed
transfer dump's `sl` EXACTLY (max abs deviation 0), and hence the published oracle@8 of 0.8791 /
0.6300 / 0.5167 and the 37.4% coverage wall.

Launch from the repo root:  python3 src/cascade_methods/stats_recert_p1_oracle.py
Writes results/cascade_methods/artifacts/_stats_recert/part1_oracle.json
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_recert_common import (CELLS, CELLS_OPEN, META, NBOOT, OPEN_DS, SEED, ci, jdump,
                                 load_transfer, load_vec, norm, oracle_at_N, pool_labels)

PUB_ORACLE = {"slake_open": 0.8791, "vqa_rad_open": 0.6300, "pathvqa_open": 0.5167}
# the 5 MCQ cells' shipped accuracy-max values -- held fixed when the open arm is replaced by an
# oracle, exactly as coverage_diagnosis_2026-08-10.json does it
MCQ_ACCMAX = {"PMC_VQA": 0.5613, "SLAKE_closed": 0.8589, "VQA_RAD_closed": 0.8526,
              "PATH_VQA_closed": 0.8891, "MedXpertQA-MM": 0.3065}
PUB_PERFECT_SEL = 0.6867670542635659
PUB_DIRECT = 0.6567


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else NBOOT
    rng = np.random.default_rng(SEED)
    res = {"what": "ATTACK C part 1 -- the oracle gap net of resampling and label noise "
                   "(arXiv:2607.03436 correction)",
           "date": "2026-08-11", "n_bootstrap": nboot, "seed": SEED, "cells": {}}

    per_cell = {}
    for cell in CELLS_OPEN:
        ds = OPEN_DS[cell]
        rows = load_transfer(ds)
        idx = [r["idx"] for r in rows]
        A = np.array([[0 if x in (None, -1) else int(x) for x in r["sl"]] for r in rows], np.int8)
        predsA = [r["preds"] for r in rows]
        selA = np.array([r["sl"][int(np.argmax(r["scores"]))] for r in rows], np.int8)

        # ---- NULL TEST: rebuild pool A's labels from the exploded judge file -----------------
        labA, predA8, ansA = pool_labels(ds, "sc8")
        rebuilt = np.array([labA[i] for i in idx], np.int8)
        dev = int(np.max(np.abs(rebuilt - A)))

        labB, predB, ansB = pool_labels(ds, "sc16")
        labC, predC, ansC = (pool_labels(ds, "sc32") if ds == "vqa_rad_open" else ({}, {}, {}))

        k = A.sum(1)
        orc = (k >= 1).astype(np.int8)

        # ---------------- (1) LABEL noise: independent judge pass on the SAME answer string ----
        # every (item, normalised answer) judged by BOTH pool A's judge run and pool B's / C's
        agree = dis = 0
        a1_tot = a1_dis = 0                       # restricted to answers pool A called CORRECT
        lone_tot = lone_dis = 0                   # the lone correct sample of a k=1 item
        for n_, i in enumerate(idx):
            other = {}
            other.update(ansB.get(i, {}))
            other.update(ansC.get(i, {}))
            if not other:
                continue
            for a, la in ansA.get(i, {}).items():
                if a not in other:
                    continue
                lb = other[a]
                agree += (la == lb)
                dis += (la != lb)
                if la == 1:
                    a1_tot += 1
                    a1_dis += (lb != 1)
                    if k[n_] == 1 and norm(predsA[n_][int(np.argmax(A[n_]))]) == a:
                        lone_tot += 1
                        lone_dis += (lb != 1)
        judge = dict(n_pairs=int(agree + dis),
                     disagreement_rate=(round(dis / (agree + dis), 5) if agree + dis else None),
                     n_pairs_A_says_correct=int(a1_tot),
                     independent_judge_says_wrong=(round(a1_dis / a1_tot, 5) if a1_tot else None),
                     n_lone_correct_confirmable=int(lone_tot),
                     lone_correct_not_confirmed=(round(lone_dis / lone_tot, 5) if lone_tot else None))

        # item-level confirmation: is ANY of the item's A-correct answers confirmed independently?
        conf = np.full(len(idx), -1, np.int8)      # 1 confirmed / 0 contradicted / -1 unjudged
        for n_, i in enumerate(idx):
            if not orc[n_]:
                continue
            other = {}
            other.update(ansB.get(i, {}))
            other.update(ansC.get(i, {}))
            seen = [other[a] for a, la in ansA.get(i, {}).items() if la == 1 and a in other]
            conf[n_] = -1 if not seen else (1 if max(seen) == 1 else 0)

        # ---------------- (2) SAMPLING noise: EXACT P(a fresh 8 is recoverable) ---------------
        fresh = np.full(len(idx), np.nan)
        for n_, i in enumerate(idx):
            lb = labB.get(i)
            if lb is None:
                continue
            fresh[n_] = oracle_at_N(lb, 8)
        have = ~np.isnan(fresh)

        per_cell[cell] = dict(ds=ds, A=A, k=k, orc=orc, selA=selA, fresh=fresh, have=have,
                              conf=conf, n=len(idx))
        res["cells"][cell] = dict(
            n=len(idx), null_test_sl_max_abs_deviation=dev,
            raw_oracle_at_8=round(float(orc.mean()), 6),
            published_oracle_at_8=PUB_ORACLE[ds],
            oracle_abs_deviation=round(abs(float(orc.mean()) - PUB_ORACLE[ds]), 5),
            k_distribution={str(j): int((k == j).sum()) for j in range(9)},
            no_correct_anywhere=round(float((k == 0).mean()), 6),
            lone_correct_share_of_recoverable=round(float((k == 1).sum() / max(orc.sum(), 1)), 5),
            selected_acc=round(float(selA.mean()), 6),
            perfect_selection_headroom_this_cell=round(float(orc.mean() - selA.mean()), 6),
            headroom_sitting_on_lone_correct_items=round(
                float(((k == 1) & (selA == 0)).sum() / len(idx)), 6),
            judge_replication=judge,
            fresh_pool_coverage=dict(
                n_items_with_an_independent_16_sample_pool=int(have.sum()),
                E_oracle_at_8_on_a_FRESH_pool=round(float(np.nanmean(fresh)), 6),
                E_fresh_given_raw_recoverable=round(
                    float(np.nanmean(fresh[have & (orc == 1)])), 6),
                E_fresh_given_k_eq_1=round(float(np.nanmean(fresh[have & (k == 1)])), 6),
                E_fresh_given_k_ge_4=round(float(np.nanmean(fresh[have & (k >= 4)])), 6),
                E_fresh_given_k_eq_0=round(float(np.nanmean(fresh[have & (k == 0)])), 6)),
            item_level_independent_confirmation=dict(
                n_recoverable=int(orc.sum()),
                n_confirmable=int((conf >= 0).sum()),
                n_confirmed=int((conf == 1).sum()),
                n_contradicted=int((conf == 0).sum()),
                confirmation_rate_among_confirmable=(
                    round(float((conf == 1).sum() / max((conf >= 0).sum(), 1)), 5))))
        print(f"  {cell:14s} raw oracle {orc.mean():.4f} (pub {PUB_ORACLE[ds]}) sl_dev={dev}  "
              f"E[fresh@8] {np.nanmean(fresh):.4f}  E[fresh|k=1] {np.nanmean(fresh[have&(k==1)]):.4f}",
              flush=True)

    # ================= restated ceilings, with a paired item bootstrap ========================
    # Three oracle definitions, all on the SAME items:
    #   raw           the published oracle@8 on pool A
    #   label_corr    an item counts as recoverable only if an INDEPENDENT judge pass confirmed at
    #                 least one of the answers pool A called correct (unconfirmable items are
    #                 handled by the two bracketing conventions below)
    #   sampling_corr E[recoverable in a FRESH pool of 8], exact, from the independent 16-sample run
    def macro8(open_vals):
        return (sum(MCQ_ACCMAX.values()) + sum(open_vals)) / 8.0

    defs = {}
    for name in ("raw", "label_corr_optimistic", "label_corr_conservative", "sampling_corr",
                 "both_conservative"):
        vals, boots = [], []
        for cell in CELLS_OPEN:
            P = per_cell[cell]
            orc, conf, fresh, have = P["orc"], P["conf"], P["fresh"], P["have"]
            if name == "raw":
                v = orc.astype(float)
            elif name == "label_corr_optimistic":
                # unconfirmable recoverable items are given the benefit of the doubt
                v = np.where(conf == 0, 0.0, orc.astype(float))
            elif name == "label_corr_conservative":
                # unconfirmable recoverable items are discounted by the MEASURED confirmation rate
                r = (conf == 1).sum() / max((conf >= 0).sum(), 1)
                v = np.where(conf == 1, 1.0, np.where(conf == 0, 0.0,
                                                      np.where(orc == 1, r, 0.0)))
            elif name == "sampling_corr":
                v = np.where(have, fresh, orc.astype(float))
            else:
                r = (conf == 1).sum() / max((conf >= 0).sum(), 1)
                lab = np.where(conf == 1, 1.0, np.where(conf == 0, 0.0,
                                                        np.where(orc == 1, r, 0.0)))
                samp = np.where(have, fresh, orc.astype(float))
                v = lab * samp / np.maximum(orc.astype(float), 1e-12) * (orc == 1)
            vals.append(float(np.mean(v)))
            n = len(v)
            bw = rng.multinomial(n, np.full(n, 1.0 / n), size=nboot).astype(np.float64)
            boots.append((bw @ v) / n)
        m = macro8(vals)
        bd = (sum(boots) / 8.0) + sum(MCQ_ACCMAX.values()) / 8.0
        defs[name] = dict(per_cell={c: round(v, 6) for c, v in zip(CELLS_OPEN, vals)},
                          macro8_perfect_selection_ceiling=round(m, 6),
                          vs_always_32b_direct=ci(bd - PUB_DIRECT, m - PUB_DIRECT),
                          pooled_coverage_wall=round(
                              1 - sum(v * per_cell[c]["n"] for v, c in zip(vals, CELLS_OPEN))
                              / sum(per_cell[c]["n"] for c in CELLS_OPEN), 6))
    res["restated_ceilings"] = defs
    res["definitions"] = {
        "raw": "the published convention: an item is recoverable if ANY of the 8 pool samples was "
               "judged correct. This is what +0.0301 and the 37.4% coverage wall are built on.",
        "label_corr_optimistic": "recoverable UNLESS an independent judge pass contradicted every "
                                 "answer pool A called correct. Unconfirmable items keep their raw "
                                 "label -- an UPPER bound on the corrected oracle.",
        "label_corr_conservative": "confirmed items count 1, contradicted 0, and unconfirmable "
                                   "recoverable items are discounted by the MEASURED confirmation "
                                   "rate -- a LOWER bound on the corrected oracle.",
        "sampling_corr": "E[at least one correct in a FRESH pool of 8], computed exactly from the "
                         "independent 16-sample re-generation. Answers 'would a re-run harvest "
                         "this?', not 'is the label right?'.",
        "both_conservative": "label_corr_conservative multiplied by the sampling replication "
                             "probability. NOTE: because the MEASURED item-level confirmation rate "
                             "is 1.000 in all three cells (no recoverable item was contradicted by "
                             "the independent judge pass), the label term is a no-op and this "
                             "reduces to 'the sampling correction applied ONLY to items already "
                             "seen to be recoverable', i.e. it discards the fresh-pool coverage "
                             "that k=0 items gain. It is the most pessimistic bound available, not "
                             "the best estimate; `sampling_corr` is the unbiased one.",
        "IMPORTANT_serving_config_caveat": "the pool-level differences between pool A and pool B "
                                           "(-0.005 / -0.005 / -0.010 per cell) are INSIDE this "
                                           "project's own measured +/-0.008-per-cell "
                                           "serving-configuration reproducibility band, so the "
                                           "POOL-LEVEL sampling correction is NOT established. The "
                                           "CONDITIONAL results (E[fresh|k=1] vs E[fresh|k>=4]) "
                                           "compare two groups inside the SAME pool B and are "
                                           "immune to that caveat.",
        "IMPORTANT_judge_caveat": "the judge replication measured here is REPLICATION (the same "
                                  "grader, the same (question, gold, answer) input, an independent "
                                  "pass), NOT VALIDITY. It rules out stochastic label noise as a "
                                  "source of inflated headroom; it cannot rule out a SYSTEMATICALLY "
                                  "lenient grader, which would replicate perfectly and still be "
                                  "wrong.",
        "macro8_perfect_selection_ceiling": "the 5 MCQ cells held at the shipped accuracy-max "
                                            "values and the 3 open cells replaced by the oracle, "
                                            "exactly as coverage_diagnosis_2026-08-10.json does it "
                                            "(published raw value 0.6867670542635659, +0.0301)."}

    # -------------------------------------------------------------------------- NULL TEST -----
    devs = [res["cells"][c]["null_test_sl_max_abs_deviation"] for c in CELLS_OPEN]
    odev = max(res["cells"][c]["oracle_abs_deviation"] for c in CELLS_OPEN)
    cov = 1 - sum(per_cell[c]["orc"].sum() for c in CELLS_OPEN) / sum(per_cell[c]["n"] for c in CELLS_OPEN)
    raw_ceiling = defs["raw"]["macro8_perfect_selection_ceiling"]
    res["null_test"] = dict(
        what="pool A's labels rebuilt from its own exploded judge file must equal the deployed "
             "transfer dump's sl exactly, and must reproduce the published oracle@8, the 37.4% "
             "coverage wall and the +0.0301 perfect-selection ceiling.",
        sl_max_abs_deviation=int(max(devs)),
        oracle_max_abs_deviation=float(odev),
        pooled_coverage_wall=round(float(cov), 6), published_coverage_wall=0.374,
        perfect_selection_ceiling=raw_ceiling, published_perfect_selection_ceiling=PUB_PERFECT_SEL,
        ceiling_abs_deviation=round(abs(raw_ceiling - PUB_PERFECT_SEL), 6),
        passed=bool(max(devs) == 0 and odev <= 1e-4 and abs(cov - 0.374) <= 1e-3
                    and abs(raw_ceiling - PUB_PERFECT_SEL) <= 1e-3))
    jdump(res, os.path.join(META, "part1_oracle.json"))
    n = res["null_test"]
    print(f"[null] sl_dev={n['sl_max_abs_deviation']} oracle_dev={n['oracle_max_abs_deviation']} "
          f"coverage={n['pooled_coverage_wall']} ceiling={n['perfect_selection_ceiling']} "
          f"passed={n['passed']}")
    for k_, v in defs.items():
        d = v["vs_always_32b_direct"]
        print(f"  {k_:26s} ceiling {v['macro8_perfect_selection_ceiling']:.4f} "
              f"headroom {d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]  "
              f"coverage wall {v['pooled_coverage_wall']:.4f}")


if __name__ == "__main__":
    main()
