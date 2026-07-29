#!/usr/bin/env python3
"""
pathvqa_regrade_batches.py - build BLINDED re-grading batches for PathVQA-open (hole 3).

Each batch file holds N items: question, reference gold, and the two 32B answers (direct mode and
reasoning mode) presented as "A"/"B" in a per-item RANDOMISED order, so the grader cannot tell which
mode produced which answer (removes style/position bias). The key that maps A/B back to modes is
written separately and is NOT given to the graders.
  python3 src/cascade_methods/pathvqa_regrade_batches.py --size 100
Launch from the repo root.
"""
import json, os, random, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_P = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu/ckpt_pathvqa_open_lingshu32b.jsonl")
THK_P = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu_think/ckpt_pathvqa_open_lingshu32b_think.jsonl")
OUTD = os.path.join(ROOT, "results/cascade_methods/claude_judge/pathvqa_granularity")
SEED = 20260729

ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=100); A = ap.parse_args()
os.makedirs(OUTD, exist_ok=True)
D = {r["idx"]: r for r in (json.loads(l) for l in open(DIR_P) if l.strip())}
T = {r["idx"]: r for r in (json.loads(l) for l in open(THK_P) if l.strip())}
idx = sorted(set(D) & set(T))
rng = random.Random(SEED)
key, items = {}, []
for i in idx:
    flip = rng.random() < 0.5           # True -> A = reasoning
    a, b = (T[i]["modal_pred"], D[i]["modal_pred"]) if flip else (D[i]["modal_pred"], T[i]["modal_pred"])
    key[i] = {"A": "reason" if flip else "direct", "B": "direct" if flip else "reason"}
    items.append({"i": i, "question": T[i]["question"], "reference": T[i]["gold"], "A": a, "B": b})
json.dump(key, open(os.path.join(OUTD, "_ab_key.json"), "w"))
n = 0
for c0 in range(0, len(items), A.size):
    ch = items[c0:c0 + A.size]
    json.dump(ch, open(os.path.join(OUTD, f"batch_b{n:03d}.json"), "w"), indent=1)
    n += 1
print(f"{len(items)} items -> {n} batches of {A.size} in {OUTD}")
