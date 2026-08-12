#!/usr/bin/env python3
"""vram_verifier_grid_report.py -- turn the (scheme x cap) verifier re-scoring arms into the frozen
metric, paired against the DEPLOYED configuration on exactly the items each arm scored.

THE CONTROL IS FREE AND EXACT.  The deployed verifier configuration is bf16 @ max_pixels 1,003,520,
which is precisely what produced the stored transfer dumps -- and this run's harness reproduces those
dumps to 0.000e+00 (see nulltest). So the control arm needs no GPU: it is the stored scores,
restricted to the same item ids the treatment arm scored.  Pairing is therefore exact by
construction, not by assumption.

METRIC: src/training_methods/genframe_data.py -- the project's single definition of sel_eff, called
with `items=` restricted to the scored subset.  Bootstrap: genframe_data.paired_bootstrap,
nboot=10000, one shared seed.

    python3 src/cascade/vram_verifier_grid_report.py --out results/cascade_methods/artifacts/_vram_levers_parts/verifier_grid.json
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))
from training_methods import genframe_data as G                     # noqa: E402

GRID = os.path.join(ROOT, "ckpts/vram_levers/verifier_grid")
NBOOT, SEED = 10000, 20260812

ap = argparse.ArgumentParser()
ap.add_argument("--grid_dir", default=GRID)
ap.add_argument("--out", default=os.path.join(
    ROOT, "results/cascade_methods/artifacts/_vram_levers_parts/verifier_grid.json"))
A = ap.parse_args()

ALL = G.load_items()
BY_KEY = {(it["ds"], it["idx"]): it for it in ALL}


def load_arm(arm):
    d = os.path.join(A.grid_dir, arm)
    sc, meta = {}, {}
    for p in ("meta.json", "nulltest.json"):
        if os.path.exists(os.path.join(d, p)):
            meta[p.replace(".json", "")] = json.load(open(os.path.join(d, p)))
    for ds in G.EVAL_DS:
        f = os.path.join(d, f"scores_{ds}.jsonl")
        if not os.path.exists(f):
            continue
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                sc[(r["ds"], r["idx"])] = r["scores"]
    return sc, meta


def gold_map():
    """{(ds, idx) -> gold answer}, joined from the published open-text generation checkpoints.
       Used only to STRATIFY, never to score."""
    g = {}
    for ds in G.EVAL_DS:
        p = os.path.join(ROOT, f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                g[(ds, r["idx"])] = str(r.get("gold", ""))
    return g


GOLD = gold_map()
LATERAL = {"left", "right", "both", "bilateral", "left side", "right side", "both sides",
           "left lung", "right lung", "l", "r"}


def strata(items):
    """The project's own documented verifier failure mode is SHORT answers -- sel_eff 79% on <=3-word
       golds (n=1,928) vs 90% at 4-8 words -- and the named cases are laterality one-liners, i.e. a
       VISUAL GROUNDING failure. If cutting the verifier's image resolution hurts, it should hurt
       there first. These masks let the report say whether it does."""
    short, lat = [], []
    for it in items:
        gd = GOLD.get((it["ds"], it["idx"]), "")
        short.append(len(gd.split()) <= 3)
        lat.append(gd.strip().lower().rstrip(".") in LATERAL)
    return np.array(short), np.array(lat)


def evaluate(arm_scores, tag):
    """sel_eff of the arm and of the control on the SAME items, plus the paired bootstrap."""
    keys = [k for k in arm_scores if k in BY_KEY]
    items = [BY_KEY[k] for k in keys]
    # canonical order (slake, vqa_rad, pathvqa), matching genframe_data.DUMP_ORDER
    order = {d: i for i, d in enumerate(G.EVAL_DS)}
    idx = sorted(range(len(items)), key=lambda i: (order[items[i]["ds"]], str(items[i]["idx"])))
    items = [items[i] for i in idx]
    keys = [keys[i] for i in idx]
    treat = {k: arm_scores[k] for k in keys}
    ctrl = {k: list(BY_KEY[k]["scores"]) for k in keys}
    rt = G.sel_eff(treat, items=items)
    rc = G.sel_eff(ctrl, items=items)
    bs = G.paired_bootstrap(rt["got"], rc["got"], rec=rt["rec"], nboot=NBOOT, seed=SEED)
    bsc = G.paired_bootstrap(rt["got"], rc["got"], rec=rt["rec"], nboot=NBOOT, seed=SEED,
                             mask=rt["contested_mask"])
    n_flip = int(np.sum(np.asarray(rt["picks"]) != np.asarray(rc["picks"])))
    sh, la = strata(items)
    got_t, got_c, rec = np.asarray(rt["got"]), np.asarray(rc["got"]), np.asarray(rt["rec"])
    strat = {}
    for nm, m in (("gold_le_3_words", sh), ("gold_gt_3_words", ~sh), ("gold_is_laterality", la)):
        sub = m & (rec == 1)
        if sub.sum() >= 10:
            bsx = G.paired_bootstrap(got_t, got_c, rec=rec, nboot=NBOOT, seed=SEED, mask=sub)
            strat[nm] = dict(n_recoverable=int(sub.sum()),
                             sel_eff_arm=round(float(got_t[sub].mean()), 6),
                             sel_eff_control=round(float(got_c[sub].mean()), 6),
                             d=round(bsx["d_sel_eff"], 6),
                             ci95=[round(bsx["d_sel_eff_ci"][0], 6), round(bsx["d_sel_eff_ci"][1], 6)],
                             significant=not (bsx["d_sel_eff_ci"][0] <= 0 <= bsx["d_sel_eff_ci"][1]))
        else:
            strat[nm] = dict(n_recoverable=int(sub.sum()), note="too few items to report")
    per_ds = {}
    for ds in G.EVAL_DS:
        if ds in rt["per_ds"]:
            per_ds[ds] = dict(n=rt["per_ds"][ds]["n"],
                              sel_eff_arm=round(rt["per_ds"][ds]["sel_eff"], 6),
                              sel_eff_control=round(rc["per_ds"][ds]["sel_eff"], 6),
                              delta=round(rt["per_ds"][ds]["sel_eff"] - rc["per_ds"][ds]["sel_eff"], 6),
                              acc_arm=round(rt["per_ds"][ds]["acc"], 6),
                              acc_control=round(rc["per_ds"][ds]["acc"], 6))
    return dict(
        arm=tag, n_items=len(items), n_recoverable=int(rt["n_recoverable"]),
        sel_eff_arm=round(rt["sel_eff"], 6), sel_eff_control=round(rc["sel_eff"], 6),
        d_sel_eff=round(bs["d_sel_eff"], 6),
        d_sel_eff_ci95=[round(bs["d_sel_eff_ci"][0], 6), round(bs["d_sel_eff_ci"][1], 6)],
        sel_eff_significant=not (bs["d_sel_eff_ci"][0] <= 0 <= bs["d_sel_eff_ci"][1]),
        selected_acc_arm=round(rt["acc"], 6), selected_acc_control=round(rc["acc"], 6),
        d_selected_acc=round(bs["d_acc"], 6),
        d_selected_acc_ci95=[round(bs["d_acc_ci"][0], 6), round(bs["d_acc_ci"][1], 6)],
        selected_acc_significant=not (bs["d_acc_ci"][0] <= 0 <= bs["d_acc_ci"][1]),
        greedy=round(rt["greedy"], 6), oracle8=round(rt["oracle"], 6),
        contested=dict(n=int(rt["contested"]["n"]),
                       sel_eff_arm=round(rt["contested"]["sel_eff"], 6),
                       sel_eff_control=round(rc["contested"]["sel_eff"], 6),
                       d=round(bsc["d_sel_eff"], 6),
                       ci95=[round(bsc["d_sel_eff_ci"][0], 6), round(bsc["d_sel_eff_ci"][1], 6)],
                       significant=not (bsc["d_sel_eff_ci"][0] <= 0 <= bsc["d_sel_eff_ci"][1])),
        n_picks_changed=n_flip, pick_change_rate=round(n_flip / max(1, len(items)), 6),
        strata=strat, per_ds=per_ds,
        guardrail_cells_worse=[d for d, v in per_ds.items() if v["delta"] < 0],
        cand_auroc_arm=round(G.cand_auroc(treat, items=items), 6),
        cand_auroc_control=round(G.cand_auroc(ctrl, items=items), 6))


out = {"_what": ("ATTACK 4, open half: what the two VRAM levers (verifier max_pixels, weight "
                 "precision) do to the FROZEN best-of-8 selection metric on the 2,345-question "
                 "open-text pool. Closes the two holes named in "
                 "vram_levers_2026-08-12.json:not_measured -- 'ACCURACY of the resolution lever on "
                 "the OPEN-TEXT arm' and 'quantisation accuracy on the 3 OPEN cells'."),
       "_control": ("bf16 @ max_pixels 1,003,520 = THE DEPLOYED verifier configuration = the stored "
                    "transfer dumps themselves, restricted to the items each arm scored. This "
                    "harness reproduces those dumps to 0.000e+00 (nulltest), so the control is exact "
                    "and the pairing is by construction."),
       "_metric": "src/training_methods/genframe_data.py sel_eff / paired_bootstrap",
       "_stats": dict(nboot=NBOOT, seed=SEED,
                      note="paired item bootstrap; sel_eff resampled inside the recoverable stratum, "
                           "selected-accuracy over all items, exactly as genframe_data draws them"),
       "_published_bar_full_pool": dict(
           n=2345, sel_eff=0.775204, selected=0.485288, greedy=0.449467, oracle8=0.626013,
           per_ds=dict(slake_open=0.850088, vqa_rad_open=0.761905, pathvqa_open=0.722581),
           source="ckpts/train/lora_verifier_disjoint/transfer_dump_*_lingshu7b.json"),
       "arms": {}}

for arm in sorted(os.listdir(A.grid_dir)) if os.path.isdir(A.grid_dir) else []:
    d = os.path.join(A.grid_dir, arm)
    if not os.path.isdir(d):
        continue
    sc, meta = load_arm(arm)
    if not sc:
        out["arms"][arm] = dict(arm=arm, status="no scores on disk", meta=meta)
        continue
    r = evaluate(sc, arm)
    r["meta"] = meta
    out["arms"][arm] = r
    print(f"[{arm}] n={r['n_items']} sel_eff {r['sel_eff_control']:.4f} -> {r['sel_eff_arm']:.4f} "
          f"(d={r['d_sel_eff']:+.4f} {r['d_sel_eff_ci95']}) picks changed {r['n_picks_changed']}",
          flush=True)

# ------------------------------------------------------------------ cross-arm, exactly paired
# Each arm above is evaluated against the control on ITS OWN scored items, which is correct for that
# arm but NOT comparable across arms while they are at different completion levels (a delta computed
# on 400 items and one computed on 285 differ partly because the item mixes differ).  This block
# restricts every arm to the INTERSECTION of what all arms scored, so arm-vs-arm statements are on
# one item set.  When every arm has finished its seeded 600 the intersection IS that 600 and this
# block reproduces the per-arm block.
raw = {a: load_arm(a)[0] for a in out["arms"] if "sel_eff_arm" in out["arms"][a]}
if len(raw) >= 2:
    common = set.intersection(*[set(v) for v in raw.values()])
    common &= set(BY_KEY)
    if common:
        out["common_subset"] = {
            "_what": ("every arm restricted to the items ALL arms scored, so arm-vs-arm comparisons "
                      "are on one item set. Each arm is still compared to the control on those same "
                      "items."),
            "n_items": len(common),
            "arms": {a: evaluate({k: v for k, v in sc.items() if k in common}, a)
                     for a, sc in raw.items()}}
        for a, r in out["common_subset"]["arms"].items():
            r.pop("per_ds", None)
            print(f"  [common n={len(common)}] {a}: sel_eff {r['sel_eff_control']:.4f} -> "
                  f"{r['sel_eff_arm']:.4f} (d={r['d_sel_eff']:+.4f} {r['d_sel_eff_ci95']})", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(out, open(A.out, "w"), indent=1)
print(f"\nwrote -> {A.out}")
