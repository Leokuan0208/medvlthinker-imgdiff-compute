#!/usr/bin/env python3
"""choicewhy_eval_pool_inventory.py -- inventory the N=8 EVALUATION candidate pool the Phase-2 verifier
will be measured on, per arm and per benchmark.

Everything here is MEASURED from ckpts/choicewhy_pilot/ckpt_<bench>_<arm>_sc8.jsonl. No model is run.
Reported per cell:
  n_items, n_candidates, mean unique candidate STRINGS per item, mean distinct LETTERS per item
  candidate-level positive rate (the base rate a verifier faces at scoring time)
  self-consistency@8 (majority letter, ties broken by first occurrence -- the Phase-1 convention)
  oracle@8 (a correct letter is present among the 8) and the headroom between them
  the LETTER-DISAGREEMENT subset (items where the 8 samples do not all agree) -- the only population a
  selector can act on, which is where Phase 1 measured +0.44 of headroom

  python3 src/cascade_methods/choicewhy_eval_pool_inventory.py
"""
import argparse, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from choicewhy_common import ARM_NAME, extract, norm  # noqa: E402

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", default="ckpts/choicewhy_pilot")
ap.add_argument("--arms", nargs="+", default=["A", "B2"])
ap.add_argument("--benches", nargs="+",
                default=["SLAKE", "VQA-RAD", "PMC-VQA", "MedXpert-Reasoning", "MedXpert-Understanding"])
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_eval_pool_inventory.json")
A = ap.parse_args()


def cell(rows, an):
    n = len(rows)
    sc = orc = 0
    dis_n = dis_sc = dis_orc = 0
    uniq_str, uniq_let, cand_pos, cand_n = 0, 0, 0, 0
    for r in rows:
        lets, strs = [], set()
        for s in r["raw_outputs"]:
            t = s.strip()
            if not t:
                continue
            strs.add(norm(t))
            L, ok, _ = extract(t, an)
            lets.append(L if ok else "?")
            cand_n += 1
            cand_pos += int(ok and L == r["gold"])
        c = Counter(lets)
        maj = max(c.items(), key=lambda kv: (kv[1], -lets.index(kv[0])))[0] if lets else "?"
        ok_sc = int(maj == r["gold"])
        ok_or = int(r["gold"] in set(lets))
        sc += ok_sc
        orc += ok_or
        uniq_str += len(strs)
        uniq_let += len(set(lets))
        if len(set(lets)) > 1:
            dis_n += 1
            dis_sc += ok_sc
            dis_orc += ok_or
    d = lambda a, b: round(a / b, 4) if b else None
    return {"n_items": n, "n_candidates": cand_n,
            "candidate_pos_rate": d(cand_pos, cand_n),
            "mean_unique_candidate_strings": d(uniq_str, n),
            "mean_distinct_letters": d(uniq_let, n),
            "self_consistency_at8": d(sc, n), "oracle_at8": d(orc, n),
            "headroom": round((orc - sc) / n, 4) if n else None,
            "letter_disagreement": {
                "n_items": dis_n, "frac_of_items": d(dis_n, n),
                "self_consistency_at8": d(dis_sc, dis_n), "oracle_at8": d(dis_orc, dis_n),
                "headroom": round((dis_orc - dis_sc) / dis_n, 4) if dis_n else None}}


out = {"purpose": "inventory of the N=8 evaluation candidate pool for the (choice)(why) verifier",
       "date": "2026-08-03", "source": f"{A.ckpt_dir}/ckpt_<bench>_<arm>_sc8.jsonl "
                                       "(Lingshu-7B, n=8, temperature 0.7, seed 1234, max_tokens 320)",
       "arms": {}}
for arm in A.arms:
    an = ARM_NAME[arm]
    per, pooled = {}, []
    for b in A.benches:
        p = os.path.join(ROOT, A.ckpt_dir, f"ckpt_{b}_{an}_sc8.jsonl")
        if not os.path.exists(p):
            print(f"  !! missing {p}", flush=True)
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        per[b] = cell(rows, an)
        pooled += rows
    per["POOLED"] = cell(pooled, an)
    out["arms"][an] = per
    print(f"\n=== arm {an} ===")
    for b, c in per.items():
        print(f"  {b:24s} n={c['n_items']:5d} uniqStr={c['mean_unique_candidate_strings']:.2f} "
              f"letters={c['mean_distinct_letters']:.2f} candPos={c['candidate_pos_rate']:.4f} "
              f"SC@8={c['self_consistency_at8']:.4f} oracle@8={c['oracle_at8']:.4f} "
              f"head={c['headroom']:+.4f} | disagree n={c['letter_disagreement']['n_items']:5d} "
              f"({c['letter_disagreement']['frac_of_items']:.3f}) SC={c['letter_disagreement']['self_consistency_at8']} "
              f"oracle={c['letter_disagreement']['oracle_at8']} head={c['letter_disagreement']['headroom']}")

os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1)
print(f"\nwrote -> {A.out}")
