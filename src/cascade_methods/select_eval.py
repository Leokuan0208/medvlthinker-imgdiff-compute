#!/usr/bin/env python3
"""Evaluate verifier-guided SELECTION over sc8 samples, scored by the LLM JUDGE (semantic).
Reconstructs per-sample judge labels from the exploded judge file, then compares selectors:
  greedy(t0) | SC-majority | <verifier>-argmax | per-sample-confidence-argmax | oracle@8.
Verifier files (optional, repeatable): {idx, preds[8], p_yes[8]} from run_openvqa_verify_persample.
Usage: python3 src/cascade_methods/select_eval.py <ckpt_dir> <ds> <tag> [verifier_label=verifier_file ...]"""
import json, os, sys, numpy as np
from collections import Counter
ckdir, ds, tag = sys.argv[1], sys.argv[2], sys.argv[3]
verifiers = {}
for a in sys.argv[4:]:
    lab, path = a.split("=", 1); verifiers[lab] = path
def load(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def norm(s): return str(s).strip().lower()

sc = load(f"{ckdir}/ckpt_{ds}_{tag}_sc8.jsonl")
# greedy/main judge (t0 or main file)
mainf = f"{ckdir}/ckpt_{ds}_{tag}.jsonl"
mainj = f"{ckdir}/ckpt_{ds}_{tag}.judge.jsonl"
main = load(mainf) if os.path.exists(mainf) else {}
gj = load(mainj) if os.path.exists(mainj) else {}
# per-sample judge: join exploded (idx#k -> answer) with exploded.judge (idx#k -> judge_ok)
expf = f"{ckdir}/ckpt_{ds}_{tag}_sc8_scexploded.jsonl"
expj = f"{ckdir}/ckpt_{ds}_{tag}_sc8_scexploded.judge.jsonl"
exp = {r["idx"]: r for r in (json.loads(l) for l in open(expf) if l.strip())}
jud = {r["idx"]: r["judge_ok"] for r in (json.loads(l) for l in open(expj) if l.strip())}
# answer(normed)->judge_ok per question
ans_judge = {}   # origidx -> {normed_answer: judge_ok}
for cid, r in exp.items():
    if cid not in jud: continue
    oi = cid.split("#")[0]; oi = int(oi) if oi.lstrip("-").isdigit() else oi
    ans_judge.setdefault(oi, {})[norm(r["modal_pred"])] = jud[cid]

def slot_labels(i):
    """per-slot judge_ok for the 8 samples of question i (None if unjudged)."""
    aj = ans_judge.get(i, {})
    return [aj.get(norm(a)) for a in sc[i]["preds"]]

idxs = [i for i in sc if i in ans_judge]
n = len(idxs)
print(f"{ds} [{tag}]: n={n} (judged)\n")

def acc_greedy():
    # greedy = main file modal_pred, judged in gj; fall back to slot of t0's modal answer
    vals = []
    for i in idxs:
        if i in gj: vals.append(gj[i]["judge_ok"])
        else:
            aj = ans_judge[i].get(norm(sc[i]["modal_pred"]))
            if aj is not None: vals.append(aj)
    return np.mean(vals), len(vals)
def acc_sc():
    v = [ans_judge[i].get(norm(sc[i]["modal_pred"])) for i in idxs]
    v = [x for x in v if x is not None]; return np.mean(v), len(v)
def acc_oracle():
    v = []
    for i in idxs:
        sl = [x for x in slot_labels(i) if x is not None]
        if sl: v.append(max(sl))
    return np.mean(v), len(v)
def acc_selector(pick):
    """pick(i)-> slot index; score by that slot's judge label."""
    v = []
    for i in idxs:
        k = pick(i); sl = slot_labels(i)
        if k is None or sl[k] is None:
            # fall back to any judged slot's majority? skip if unjudged
            continue
        v.append(sl[k])
    return np.mean(v), len(v)

g, ng = acc_greedy(); s, nsc = acc_sc(); o, no = acc_oracle()
print(f"{'greedy (t0)':<26} {g:.3f}  (n={ng})")
print(f"{'SC-majority':<26} {s:.3f}  (n={nsc})")
# load verifiers (pointwise p_yes -> argmax; listwise picked_answer -> matching slot)
vf = {lab: load(p) for lab, p in verifiers.items()}
for lab, vmap in vf.items():
    def pick(i, vmap=vmap):
        if i not in vmap: return None
        r = vmap[i]
        if "p_yes" in r: return int(np.argmax(r["p_yes"]))
        if "picked_answer" in r:                       # listwise: map picked answer to a slot
            pa = norm(r["picked_answer"])
            for k, a in enumerate(sc[i]["preds"]):
                if norm(a) == pa: return k
            return None
        return None
    a, na = acc_selector(pick)
    print(f"{'select['+lab+']':<26} {a:.3f}  (n={na})")
print(f"{'oracle@8':<26} {o:.3f}  (n={no})")
print(f"\nREAD: a verifier that beats SC-majority and approaches oracle@8 ESCAPES the majority trap.")
print(f"      greedy/SC are the bars to beat; oracle@8 is the ceiling for any training-free selector.")
