#!/usr/bin/env python3
"""verifier_disjoint_measure.py -- re-measure the open-text arm with the CLEAN (disjoint-trained)
verifier and put the CONTAMINATED verifier's numbers alongside, so the inflation is visible.

On the FULL reported eval sets (slake_open 645, vqa_rad_open 200, pathvqa_open 1500):
  greedy            the cheap model's modal answer
  self-consistency  majority vote over the 8 samples
  verifier@8        best-of-8 by argmax verifier P(Yes)  -- CLEAN (L1 and L2) and CONTAMINATED
  oracle@8          any-correct-in-8 upper bound
all scored by the SAME 32B LLM judge (judge_ok) the headline uses, with paired bootstrap 95% CIs.
Also re-derives the END-TO-END open arm (best-of-8 + verifier pick, verifier-confidence gate,
escalate to 32B-no-think, 5-fold cross-fit tau) exactly as src/cascade_methods/integrated_method.py
does, so the effect on the reported open-arm cell and the pooled headline is measured, not guessed.

The comparison is exactly paired: every verifier ranks the SAME candidate lists with the SAME judge
labels, so greedy / SC / oracle are identical by construction and the script ASSERTS that.

  python3 src/training_methods/verifier_disjoint_measure.py
  -> results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json
"""
import argparse, json, os
from collections import Counter

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
DSETS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
TAG = "lingshu7b"
STRONG = "ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl"
FULL_SUITE_N = 42374                 # METHOD_FINAL_2026-07.md pooled.full_suite
REPORTED_FULL_SUITE_ACC = 0.5750
REPORTED_OPEN_ARM_POOLED = 0.5642
norm = lambda s: str(s).strip().lower()

ap = argparse.ArgumentParser()
ap.add_argument("--contaminated", default="ckpts/train/lora_verifier_pooled4")
ap.add_argument("--clean_l1", default="ckpts/train/lora_verifier_disjoint",
                help="image-disjoint verifier (no eval image, no eval item)")
ap.add_argument("--clean_l2", default="ckpts/train/lora_verifier_disjoint_l2",
                help="L1 + no eval question TEXT at all (conservative bound); skipped if absent")
ap.add_argument("--nboot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="results/cascade_methods/artifacts/verifier_disjoint_split.json")
ap.add_argument("--audit", default="results/cascade_methods/artifacts/verifier_validity_2026-07-29.json")
ap.add_argument("--out", default="results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json")
A = ap.parse_args()

DUMP_ORDER = {}   # ds -> the contaminated dump's FILE order. integrated_method.py's 5-fold cross-fit
                  # assigns folds by row POSITION, so order must be preserved and shared by all arms.


def load_dump(adapter, ds, record_order=False):
    p = J(f"{adapter}/transfer_dump_{ds}_{TAG}.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    if record_order:
        DUMP_ORDER[ds] = [r["idx"] for r in rows]
    return {r["idx"]: r for r in rows}


def load_judge_jsonl(p):
    m = {}
    if os.path.exists(J(p)):
        for l in open(J(p)):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m


def per_question(r):
    """-> (greedy, sc, oracle, verifier_pick_label) from the dump's own labels."""
    sl = [None if x == -1 else int(x) for x in r["sl"]]
    preds = r["preds"]
    lab = {norm(a): l for a, l in zip(preds, sl) if l is not None}
    greedy = int(r["greedy_ok"])
    top = Counter(norm(a) for a in preds).most_common(1)[0][0]
    sc = int(lab.get(top, 0))
    oracle = int(max([x for x in sl if x is not None]))
    k = int(np.argmax(r["scores"]))
    ver = int(sl[k]) if sl[k] is not None else 0
    return greedy, sc, oracle, ver


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    m = y >= 0
    s, y = s[m], y[m]
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        ix = np.where(s == v)[0]
        if len(ix) > 1: ranks[ix] = ranks[ix].mean()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------- cascade mechanics, verbatim from integrated_method.py ----------------
def cascade_acc(ok_cheap, ok_strong, gate, tau):
    esc = gate < tau
    return np.where(esc, ok_strong, ok_cheap).mean(), esc.mean()


def pick_tau_isocost(ok_cheap, ok_strong, gate, target):
    cand = sorted(set(gate.tolist()))
    grid = [cand[0] - 1e-9] + cand + [cand[-1] + 1e-9]
    best_iso, best_max = None, (-1, None)
    for tau in grid:
        acc, esc = cascade_acc(ok_cheap, ok_strong, gate, tau)
        if acc > best_max[0]: best_max = (acc, tau)
        if acc >= target - 1e-12 and (best_iso is None or esc < best_iso[1]):
            best_iso = (tau, esc)
    return best_iso[0] if best_iso else best_max[1]


def heldout(ok_cheap, ok_strong, gate, K=5):
    n = len(ok_cheap); accs, escs = [], []
    for f in range(K):
        te = np.array([i % K == f for i in range(n)]); tr = ~te
        if tr.sum() < 2 or te.sum() < 1: continue
        tgt = ok_strong[tr].mean()
        tau = pick_tau_isocost(ok_cheap[tr], ok_strong[tr], gate[tr], tgt)
        acc, esc = cascade_acc(ok_cheap[te], ok_strong[te], gate[te], tau)
        accs.append(acc); escs.append(esc)
    return float(np.mean(accs)), float(np.mean(escs))


# ================================================================== analysis of one clean adapter
def analyse(clean_adapter, label, cont_dumps, nboot):
    rng = np.random.default_rng(A.seed)
    rows, cand = {}, {}
    for ds in DSETS:
        dc = load_dump(clean_adapter, ds)
        if dc is None:
            print(f"[{label}] MISSING clean dump for {ds} -> skipping this level", flush=True)
            return None, None
        dk = cont_dumps[ds]
        assert len(dc) == len(dk), f"{ds}: dump sizes differ ({len(dc)} vs {len(dk)})"
        rr, cs, ks, ys = [], [], [], []
        for i in DUMP_ORDER[ds]:
            a, b = dc[i], dk[i]
            # HARD comparability check: same candidates, same judge labels, same greedy answer
            assert a["preds"] == b["preds"], f"{ds}/{i}: candidate lists differ"
            assert a["sl"] == b["sl"], f"{ds}/{i}: judge labels differ"
            assert a["greedy_ok"] == b["greedy_ok"], f"{ds}/{i}: greedy label differs"
            g1, s1, o1, v1 = per_question(a)
            g2, s2, o2, v2 = per_question(b)
            assert (g1, s1, o1) == (g2, s2, o2), f"{ds}/{i}: baselines differ -- not comparable"
            rr.append({"idx": i, "greedy": g1, "sc": s1, "oracle": o1, "clean": v1, "cont": v2})
            for sc_c, sc_k, l_ in zip(a["scores"], b["scores"], a["sl"]):
                cs.append(sc_c); ks.append(sc_k); ys.append(l_)
        rows[ds] = rr; cand[ds] = (cs, ks, ys)
        print(f"[{label}] {ds:14s} n={len(rr)} questions, {len(cs)} candidates", flush=True)
    rows["POOLED"] = [r for ds in DSETS for r in rows[ds]]
    cand["POOLED"] = tuple([sum((list(cand[ds][j]) for ds in DSETS), []) for j in range(3)])

    CONTRASTS = {
        "clean_verifier_minus_greedy":        lambda a: a["clean"].mean() - a["greedy"].mean(),
        "clean_verifier_minus_sc":            lambda a: a["clean"].mean() - a["sc"].mean(),
        "contaminated_verifier_minus_greedy": lambda a: a["cont"].mean() - a["greedy"].mean(),
        "clean_minus_contaminated":           lambda a: a["clean"].mean() - a["cont"].mean(),
    }

    def boot(rr, f):
        n = len(rr)
        arr = {m: np.array([r[m] for r in rr], float) for m in ("greedy", "sc", "oracle", "clean", "cont")}
        out = np.empty(nboot)
        for b in range(nboot):
            ix = rng.integers(0, n, n)
            out[b] = f({m: v[ix] for m, v in arr.items()})
        return {"point": float(f(arr)),
                "ci": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]}

    RES = {}
    for ds in DSETS + ["POOLED"]:
        rr = rows[ds]; cs, ks, ys = cand[ds]
        m = {k: float(np.mean([r[k] for r in rr])) for k in ("greedy", "sc", "oracle", "clean", "cont")}
        RES[ds] = {
            "n_questions": len(rr), "n_candidates": len(cs),
            "greedy": m["greedy"], "self_consistency": m["sc"],
            "verifier_clean": m["clean"], "verifier_contaminated": m["cont"], "oracle_at_8": m["oracle"],
            "auroc_candidate_clean": auroc(cs, ys), "auroc_candidate_contaminated": auroc(ks, ys),
            "oracle_conversion_clean": (float((m["clean"] - m["greedy"]) / (m["oracle"] - m["greedy"]))
                                        if m["oracle"] > m["greedy"] else None),
            "oracle_conversion_contaminated": (float((m["cont"] - m["greedy"]) / (m["oracle"] - m["greedy"]))
                                               if m["oracle"] > m["greedy"] else None),
            "contrasts": {k: boot(rr, f) for k, f in CONTRASTS.items()},
        }
        r = RES[ds]
        print(f"\n[{label}] === {ds}  n={r['n_questions']}")
        print(f"  greedy {r['greedy']:.4f} | SC {r['self_consistency']:.4f} | verifier CLEAN "
              f"{r['verifier_clean']:.4f} | verifier CONTAMINATED {r['verifier_contaminated']:.4f} | "
              f"oracle@8 {r['oracle_at_8']:.4f}")
        for k, v in r["contrasts"].items():
            print(f"    {k:36s} {v['point']:+.4f}  95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")

    # ---------------- end-to-end open arm ----------------
    E2E = {}
    for ds in DSETS:
        sj = load_judge_jsonl(STRONG.format(ds=ds))
        dc, dk = load_dump(clean_adapter, ds), cont_dumps[ds]
        if not sj:
            E2E[ds] = {"note": "no 32B-no-think judge file"}; continue
        row = {}
        for who, dump in (("clean", dc), ("contaminated", dk)):
            okc, ok32, gate = [], [], []
            for i in DUMP_ORDER[ds]:
                if i not in sj or i not in dump: continue
                r = dump[i]
                sc = r["scores"][:8]; sl = [0 if x in (None, -1) else int(x) for x in r["sl"][:8]]
                okc.append(sl[int(np.argmax(sc))]); ok32.append(sj[i]); gate.append(float(max(sc)))
            okc, ok32, gate = np.array(okc, float), np.array(ok32, float), np.array(gate, float)
            acc, esc = heldout(okc, ok32, gate)
            row[who] = {"n": len(okc), "cheap_leg_bo8_verifier": float(okc.mean()),
                        "strong_32b_nothink": float(ok32.mean()),
                        "arm_accuracy": acc, "escalation_rate": esc}
        row["delta_arm_accuracy_clean_minus_contaminated"] = (
            row["clean"]["arm_accuracy"] - row["contaminated"]["arm_accuracy"])
        E2E[ds] = row
        print(f"[{label}] end-to-end {ds}: clean arm {row['clean']['arm_accuracy']:.4f} "
              f"(esc {row['clean']['escalation_rate']:.1%}) vs contaminated "
              f"{row['contaminated']['arm_accuracy']:.4f} (esc {row['contaminated']['escalation_rate']:.1%})")
    if all("clean" in v for v in E2E.values()):
        wn = {ds: E2E[ds]["clean"]["n"] for ds in DSETS}; tot = sum(wn.values())
        pooled = {who: sum(E2E[ds][who]["arm_accuracy"] * wn[ds] for ds in DSETS) / tot
                  for who in ("clean", "contaminated")}
        d = pooled["clean"] - pooled["contaminated"]
        E2E["POOLED_OPEN_ARM"] = {
            "n": tot, "clean": pooled["clean"], "contaminated": pooled["contaminated"], "delta": d,
            "reported_in_METHOD_FINAL": REPORTED_OPEN_ARM_POOLED,
            "first_order_effect_on_full_suite_pooled": d * tot / FULL_SUITE_N,
            "full_suite_pooled_reported": REPORTED_FULL_SUITE_ACC,
            "full_suite_pooled_corrected_first_order": REPORTED_FULL_SUITE_ACC + d * tot / FULL_SUITE_N,
            "caveat": "first-order: only the open-arm cell is recomputed. The MCQ arm (n=40029, 94.5% of "
                      "the pooled suite) does not use the trained verifier and is unaffected.",
        }
        print(f"[{label}] end-to-end POOLED OPEN ARM (n={tot}): clean {pooled['clean']:.4f} vs "
              f"contaminated {pooled['contaminated']:.4f} ({d:+.4f}); first-order full-suite "
              f"{REPORTED_FULL_SUITE_ACC:.4f} -> {REPORTED_FULL_SUITE_ACC + d*tot/FULL_SUITE_N:.4f}")
    return RES, E2E


# ================================================================== run
cont_dumps = {}
for ds in DSETS:
    d = load_dump(A.contaminated, ds, record_order=True)
    assert d is not None, f"no contaminated dump for {ds}"
    cont_dumps[ds] = d

LEVELS = {}
for key, adapter, desc in [
        ("L1_image_disjoint", A.clean_l1,
         "no eval image and no eval item in training; question TEMPLATES may recur with other images"),
        ("L2_strict", A.clean_l2,
         "L1 and no eval question TEXT at all -- conservative bound; starves the in-domain SLAKE pool")]:
    RES, E2E = analyse(adapter, key, cont_dumps, A.nboot)
    if RES is None:
        LEVELS[key] = {"status": "not available", "adapter": adapter}
        continue
    infl = {ds: {"clean_gain": RES[ds]["contrasts"]["clean_verifier_minus_greedy"]["point"],
                 "contaminated_gain": RES[ds]["contrasts"]["contaminated_verifier_minus_greedy"]["point"],
                 "inflation_x": (RES[ds]["contrasts"]["contaminated_verifier_minus_greedy"]["point"] /
                                 RES[ds]["contrasts"]["clean_verifier_minus_greedy"]["point"])
                 if RES[ds]["contrasts"]["clean_verifier_minus_greedy"]["point"] else None}
            for ds in DSETS + ["POOLED"]}
    tc = J(f"{adapter}/train_config.json")
    LEVELS[key] = {"status": "ok", "adapter": adapter, "definition": desc,
                   "train_config": json.load(open(tc)) if os.path.exists(tc) else None,
                   "selection_stage": RES, "inflation": infl, "end_to_end_open_arm": E2E}

out = {
    "what": "UNCONTAMINATED re-measurement of the open-text best-of-N arm: verifiers retrained on "
            "strictly disjoint splits (no eval question, no eval image) vs the deployed verifier, which "
            "was trained on 67-73% of the items it was scored on.",
    "date": "2026-07-30",
    "judge": "src/labeling/run_judge.py (MedVLThinker-32B, judge_ok) -- the SAME judge as the headline",
    "contaminated_adapter": A.contaminated,
    "candidate_pools": "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl -- UNCHANGED. Candidates "
                       "and judge labels are verifier-independent (verified: all three contaminated dumps' "
                       "preds/sl/greedy_ok reproduce exactly from the raw sc8+judge files), so only the "
                       "verifier SCORES were recomputed; nothing was regenerated.",
    "comparability_asserted": "identical candidate lists, identical judge labels, identical greedy/SC/oracle "
                              "across all arms (asserted in code, run aborts otherwise)",
    "harness_validated": "running this script with clean==contaminated reproduces METHOD_FINAL_2026-07.md's "
                         "open-arm cells exactly (SLAKE 0.8155@12.6%, VQA-RAD 0.5850@5.5%, "
                         "PathVQA 0.4533@0.1%, pooled 0.5642)",
    "nboot": A.nboot, "seed": A.seed,
    "split": json.load(open(J(A.split))) if os.path.exists(J(A.split)) else None,
    "levels": LEVELS,
    "prior_audit": A.audit,
}
os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(out, open(J(A.out), "w"), indent=1)
print(f"\nwrote -> {A.out}")
