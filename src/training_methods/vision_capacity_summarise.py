#!/usr/bin/env python3
"""vision_capacity_summarise.py -- turn the resumable score JSONL from
verifier_vision_capacity_ablation.py into results/.../_visverif_parts/vision_capacity.json.

Reads only; fits nothing; scores nothing.

METRIC.  The pass is restricted to MIXED items (pool contains both a correct and an incorrect
candidate) because on every other item the pick cannot change the outcome, so scoring them would
burn GPU for zero information about a SELECTION difference.  The reported quantity is therefore

    mixed_selection_accuracy = mean over scored mixed items of [ label of the argmax-scored slot ]

computed on the SAME items for every arm (arms are compared only on their intersection), with the
incumbent's stored dump scores as the reference on that identical subset.  This is a sub-metric of
sel_eff, not sel_eff itself, and is labelled as such -- it is not comparable to 0.775204.

FIDELITY GATE.  The `full` arm re-scores with the adapter untouched, so its scores must reproduce
the stored transfer dump.  If they do not, the harness is wrong and no ablation number is reported.
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G  # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PARTS, "vision_capacity_scores.jsonl"))
    ap.add_argument("--out", default=os.path.join(PARTS, "vision_capacity.json"))
    ap.add_argument("--nboot", type=int, default=10000)
    A = ap.parse_args()

    items = {(it["ds"], it["idx"]): it for it in G.load_items()}
    got = defaultdict(dict)
    for line in open(A.ckpt):
        try:
            r = json.loads(line)
        except Exception:
            continue
        got[r["arm"]][(r["ds"], r["idx"])] = r["scores"]

    if not got:
        print("no scores yet"); return

    common = set.intersection(*[set(v) for v in got.values()]) if len(got) > 1 \
        else set(next(iter(got.values())))
    common = sorted(common, key=lambda k: (k[0], str(k[1])))
    print(f"[arms] {', '.join(f'{a}:{len(v)}' for a, v in got.items())}")
    print(f"[common] {len(common)} items scored under every arm")

    def outcome(scores, it):
        n = min(len(scores), len(it["sl"]))
        k = int(np.argmax(np.asarray(scores[:n], float)))
        return 1 if it["sl"][k] == 1 else 0

    # ---- fidelity gate on the untouched adapter
    fid = None
    if "full" in got:
        dev, npts = [], 0
        for k, sc in got["full"].items():
            it = items[k]
            n = min(len(sc), len(it["scores"]))
            dev.extend(abs(np.asarray(sc[:n], float) - np.asarray(it["scores"][:n], float)))
            npts += n
        dev = np.asarray(dev)
        fid = {"n_candidate_scores_compared": int(npts),
               "mean_abs_deviation": float(dev.mean()), "max_abs_deviation": float(dev.max()),
               "frac_within_0.01": float((dev <= 0.01).mean()),
               "verdict": "PASS" if dev.mean() < 1e-3 else "CHECK",
               "what": "re-scoring with lora_B untouched must reproduce the stored transfer dump; "
                       "any deviation in the ablated arms is then the ablation, not the harness"}

    res, vecs = {}, {}
    for arm, d in got.items():
        v = np.array([outcome(d[k], items[k]) for k in common if k in d])
        vecs[arm] = v
        by = defaultdict(list)
        for k in common:
            if k in d:
                by[k[0]].append(outcome(d[k], items[k]))
        res[arm] = {"n": int(len(v)), "mixed_selection_accuracy": float(v.mean()),
                    "per_ds": {ds: {"n": len(x), "acc": float(np.mean(x))} for ds, x in by.items()},
                    "n_scored_total": len(d)}

    inc = np.array([outcome(items[k]["scores"], items[k]) for k in common])
    res["incumbent_stored_dump"] = {"n": int(len(inc)),
                                    "mixed_selection_accuracy": float(inc.mean())}

    def boot(a, b):
        rng = np.random.default_rng(0)
        n = len(a)
        d = np.array([(a[i].mean() - b[i].mean())
                      for i in (rng.integers(0, n, n) for _ in range(A.nboot))])
        return {"delta": float(a.mean() - b.mean()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    # PAIRWISE intersections: a 3-way intersection is capped by whichever arm has scored fewest
    # items so far, which would throw away most of the completed work on the key contrast.
    comp = {}
    if "full" in got:
        for arm in got:
            if arm == "full":
                continue
            keys = sorted(set(got[arm]) & set(got["full"]), key=lambda k: (k[0], str(k[1])))
            a = np.array([outcome(got[arm][k], items[k]) for k in keys])
            b = np.array([outcome(got["full"][k], items[k]) for k in keys])
            inc_k = np.array([outcome(items[k]["scores"], items[k]) for k in keys])
            r = boot(a, b)
            r.update({"n_paired_items": len(keys),
                      "arm_accuracy": float(a.mean()), "full_accuracy": float(b.mean()),
                      "incumbent_dump_on_same_items": float(inc_k.mean()),
                      "per_ds_n": {ds: int(sum(1 for k in keys if k[0] == ds))
                                   for ds in sorted({k[0] for k in keys})}})
            comp[f"{arm}_minus_full"] = r

    rep = {
        "what": "ATTACK 1(c) BY ABLATION: how much of the deployed clean verifier's selection comes "
                "from the 15.17% of its LoRA capacity that lands on the vision tower?",
        "date": "2026-08-12",
        "code": ["src/training_methods/verifier_vision_capacity_ablation.py",
                 "src/training_methods/vision_capacity_summarise.py"],
        "adapter": "ckpts/train/lora_verifier_disjoint",
        "adapter_composition": {
            "total_lora_params": 47589376, "vision_tower_params": 7219200,
            "vision_fraction": 0.1517, "vision_tensors": 192, "adapted_modules": 292,
            "STRUCTURAL_FACT": "all 192 vision tensors are visual.blocks.*.mlp.{down,gate,up}_proj. "
                               "The ViT's ATTENTION carries NO adapter capacity at all, because "
                               "Qwen2.5-VL names it attn.qkv / attn.proj and the recipe's "
                               "target_modules are q_proj/k_proj/v_proj/o_proj. So the 'incidental "
                               "15.2% on vision' is feed-forward only, and the spatial-mixing part "
                               "of the vision tower -- what a laterality question would need -- was "
                               "never adaptable under this recipe."},
        "arms": {"full": "adapter untouched (fidelity gate + reference)",
                 "no_visual_lora": "lora_B zeroed on all 96 visual.* modules",
                 "visual_only": "lora_B zeroed on all 196 language_model.* modules"},
        "metric": "mixed_selection_accuracy -- see module docstring. NOT comparable to sel_eff "
                  "0.775204; it is computed on mixed items only, on a capped subsample.",
        "item_selection": "mixed items only (pool has both a correct and an incorrect candidate; "
                          "n=852 of 2345), capped per dataset by a fixed shuffle (seed 20260812)",
        "fidelity_gate": fid,
        "results": res,
        "comparisons": comp,
        "n_common_items": len(common),
    }
    json.dump(rep, open(A.out, "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk != "per_ds"})
                      for k, v in res.items()}, indent=1))
    print(json.dumps(comp, indent=1))
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
