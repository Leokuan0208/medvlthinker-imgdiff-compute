#!/usr/bin/env python3
"""decoding_sweep_judgepreload.py -- reuse the project's EXISTING judge labels for the sweep.

The judge (src/labeling/run_judge.py, MedVLThinker-32B, temperature 0) is a deterministic function of
(question, gold, candidate text). The same (ds, idx, normalized-answer) therefore has the same label in
every dump it appears in, regardless of which model or decoding setting produced the string. So every
candidate the sweep re-generates that was already judged in an earlier project dump is FREE.

This is only trustworthy if it reproduces the frozen labels, so the script VALIDATES the preload
against the transfer dumps' `sl` (the labels the published cells were computed from) and refuses to
write a cache that disagrees.

  python3 src/cascade_methods/decoding_sweep_judgepreload.py
"""
import json, os, sys
from collections import defaultdict
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G      # noqa: E402

SWEEP = os.path.join(ROOT, "ckpts/openvqa/decoding_sweep")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]

# TEST-split dumps of the three eval sets only. *_train is EXCLUDED: the train parquet index space
# collides numerically with the test one, so a train label could be silently attached to a test idx.
SRC = {
    "slake_open": ["cheap_lingshu7b/ckpt_slake_open_lingshu7b_sc8_scexploded",
                   "cheap_lingshu7b/ckpt_slake_open_lingshu7b_sc16_scexploded",
                   "diverse/ckpt_slake_open_lingshu7b_div_scexploded",
                   "cheapleg_base7b/ckpt_slake_open_base7b_sc8_scexploded"],
    "vqa_rad_open": ["cheap_lingshu7b/ckpt_vqa_rad_open_lingshu7b_sc8_scexploded",
                     "cheap_lingshu7b/ckpt_vqa_rad_open_lingshu7b_sc16_scexploded",
                     "diverse/ckpt_vqa_rad_open_lingshu7b_div_scexploded",
                     "cheapleg_base7b/ckpt_vqa_rad_open_base7b_sc8_scexploded"],
    "pathvqa_open": ["cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b_sc8_scexploded",
                     "cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b_sc16_scexploded",
                     "diverse/ckpt_pathvqa_open_lingshu7b_div_scexploded",
                     "cheapleg_base7b/ckpt_pathvqa_open_base7b_sc8_scexploded"],
}


def main():
    out = {"sources_used": defaultdict(list), "n_preloaded": {}, "internal_conflicts": {}}
    cache = {}                     # (ds, idx, na) -> label
    conflicted = set()
    conflicts = defaultdict(int)
    for ds in DS:
        for stem in SRC[ds]:
            pf = os.path.join(ROOT, "ckpts/openvqa", stem + ".jsonl")
            jf = os.path.join(ROOT, "ckpts/openvqa", stem + ".judge.jsonl")
            if not (os.path.exists(pf) and os.path.exists(jf)):
                continue
            jud = {}
            for l in open(jf):
                if l.strip():
                    r = json.loads(l); jud[r["idx"]] = int(r["judge_ok"])
            n = 0
            for l in open(pf):
                if not l.strip():
                    continue
                r = json.loads(l)
                if r["idx"] not in jud:
                    continue
                base = str(r["idx"]).split("#")[0]
                try:
                    idx = int(base)
                except ValueError:
                    continue
                k = (ds, idx, G.norm(r["modal_pred"]))
                v = jud[r["idx"]]
                if k in cache and cache[k] != v:
                    # Two earlier dumps disagree on the SAME (ds, idx, normalized answer). The judge is
                    # deterministic in (question, gold, text), so a disagreement means the inputs were
                    # not identical (e.g. a differently-worded question field). Do not guess: DROP the
                    # key entirely and let it be re-judged.
                    conflicted.add(k); conflicts[ds] += 1
                    continue
                cache[k] = v; n += 1
            out["sources_used"][ds].append({"stem": stem, "rows_used": n})
        out["n_preloaded"][ds] = sum(1 for k in cache if k[0] == ds)
        out["internal_conflicts"][ds] = conflicts[ds]

    for k in conflicted:
        cache.pop(k, None)
    out["n_conflicted_keys_dropped"] = len(conflicted)
    for ds in DS:
        out["n_preloaded"][ds] = sum(1 for k in cache if k[0] == ds)

    # ---- VALIDATION: must reproduce the frozen labels the published cells were computed from ----
    agree = disagree = absent = 0
    for it in G.load_items():
        for s, a in enumerate(it["preds"]):
            k = (it["ds"], it["idx"], G.norm(a))
            if k not in cache:
                absent += 1
            elif cache[k] == int(it["sl"][s]):
                agree += 1
            else:
                disagree += 1
    out["validation_against_frozen_transfer_dump_sl"] = {
        "agree": agree, "disagree": disagree, "not_in_preload": absent,
        "agreement_rate": agree / max(agree + disagree, 1),
        "coverage_of_deployed_pool": (agree + disagree) / max(agree + disagree + absent, 1)}
    ok = (disagree == 0)          # conflicted keys are dropped, not trusted
    out["pass"] = bool(ok)
    print(json.dumps({k: v for k, v in out.items() if k != "sources_used"}, indent=1, default=str))
    if not ok:
        print("\n!! PRELOAD REFUSED: it does not reproduce the frozen labels exactly.", flush=True)
        return 1
    for ds in DS:
        p = os.path.join(SWEEP, f"judgecache_preload_{ds}.jsonl")
        with open(p, "w") as fh:
            for (d, idx, na), v in cache.items():
                if d == ds:
                    fh.write(json.dumps({"idx": idx, "na": na, "judge_ok": v}) + "\n")
        print(f"wrote {p} ({out['n_preloaded'][ds]} labels)")
    json.dump(out, open(os.path.join(SWEEP, "judgepreload_report.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
