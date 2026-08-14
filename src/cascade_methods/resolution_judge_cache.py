#!/usr/bin/env python3
"""resolution_judge_cache.py -- SWEEP 2: judge-label bookkeeping for the generator sweep.

The project's judge (src/labeling/run_judge.py: MedVLThinker-32B, Qwen2.5-32B backbone, a NEUTRAL
third model, text-only) scores a triple (question, gold, answer).  It never sees the image, so a
label is a pure function of (dataset, item, answer TEXT) and is IDENTICAL no matter which
resolution arm produced that text.  This script therefore:

  build  -- reads every stored judge file in ckpts/openvqa/ plus every arm of the resolution sweep,
            builds the label cache keyed by (ds, idx, normalized answer), and writes ONE exploded
            judge-input file containing only the triples that have never been judged;
  merge  -- folds the new judge output back into the cache.

That keeps the judge pass proportional to NEW answer strings, not to caps x seeds, and it
guarantees the control arm and every swept arm are labelled by the same judge, in the same way,
with byte-identical labels wherever the text coincides.

  python3 src/cascade_methods/resolution_judge_cache.py build
  # ... run_judge.py on the emitted file ...
  python3 src/cascade_methods/resolution_judge_cache.py merge
"""
import glob
import json
import os
import sys

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
DEPLOYED = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
CACHE = os.path.join(SWEEP, "judge_cache.json")
TODO = os.path.join(SWEEP, "judge_todo.jsonl")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]


def norm(s):
    return str(s).strip().lower()


def _read(p):
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        if l.strip():
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def seed_cache():
    """labels already on disk from the deployed pool (greedy + sc8 + sc16 + train pools)."""
    cache = {}
    for ds in DS:
        # single-answer files: idx -> judge_ok, answer is that row's modal_pred
        for base in glob.glob(os.path.join(DEPLOYED, f"ckpt_{ds}*.jsonl")):
            if base.endswith(".judge.jsonl") or base.endswith(".audit.jsonl"):
                continue
            jf = base.replace(".jsonl", ".judge.jsonl")
            if not os.path.exists(jf):
                continue
            lab = {r["idx"]: r["judge_ok"] for r in _read(jf)}
            for r in _read(base):
                if r["idx"] in lab and "modal_pred" in r:
                    cache[f"{ds}|{r['idx']}|{norm(r['modal_pred'])}"] = int(lab[r["idx"]])
    return cache


def sweep_rows():
    """every (ds, idx, question, gold, answer) the resolution sweep needs a label for."""
    need = {}
    for f in sorted(glob.glob(os.path.join(SWEEP, "ckpt_*.jsonl"))):
        b = os.path.basename(f)
        ds = next((d for d in DS if b.startswith(f"ckpt_{d}_")), None)
        if ds is None:
            continue
        for r in _read(f):
            for a in r.get("preds", []):
                need[f"{ds}|{r['idx']}|{norm(a)}"] = (ds, r["idx"], r["question"], r["gold"], a)
    return need


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    os.makedirs(SWEEP, exist_ok=True)
    if mode == "build":
        cache = json.load(open(CACHE)) if os.path.exists(CACHE) else seed_cache()
        need = sweep_rows()
        new = [v for k, v in need.items() if k not in cache]
        with open(TODO, "w") as fh:
            for ds, idx, q, gold, a in new:
                fh.write(json.dumps({"idx": f"{ds}@@{idx}@@{norm(a)}", "question": q,
                                     "gold": gold, "modal_pred": a}) + "\n")
        json.dump(cache, open(CACHE, "w"))
        print(f"cache={len(cache)} needed={len(need)} NEW_TO_JUDGE={len(new)} -> {TODO}")
    elif mode == "merge":
        cache = json.load(open(CACHE))
        jf = TODO.replace(".jsonl", ".judge.jsonl")
        n = 0
        for r in _read(jf):
            cache[str(r["idx"]).replace("@@", "|")] = int(r["judge_ok"])
            n += 1
        json.dump(cache, open(CACHE, "w"))
        need = sweep_rows()
        miss = [k for k in need if k not in cache]
        print(f"merged {n}; cache={len(cache)}; still-missing={len(miss)}")
    else:
        raise SystemExit("mode must be build|merge")


if __name__ == "__main__":
    main()
