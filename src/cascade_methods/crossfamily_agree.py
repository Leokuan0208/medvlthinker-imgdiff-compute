#!/usr/bin/env python3
"""Cross-FAMILY agreement as a correctness signal (decorrelated errors vs within-model self-consistency).
Two independently-trained medical VLMs (MedVLThinker-7B, Lingshu-7B) on SLAKE-open. When they give the
SAME open-ended answer, is it almost always right? If agree->high accuracy AND covers many questions,
a cross-family consensus SELECTOR could escape the luck floor. Uses existing answers + LLM-judge labels."""
import json, os, numpy as np
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute"); J=lambda p:os.path.join(ROOT,p)
def load(p): return {r["idx"]:r for r in (json.loads(l) for l in open(J(p)) if l.strip())}
def norm(s):
    import re
    return re.sub(r"[^a-z0-9 ]","",str(s).strip().lower()).strip()
def agree(a,b):
    na,nb=norm(a),norm(b)
    if not na or not nb: return False
    if na==nb: return True
    # semantic-ish: one contained in the other (short medical answers)
    return na in nb or nb in na
ds="slake_open"
mv=load(f"ckpts/openvqa/cheap/ckpt_{ds}_7b_t0.jsonl"); mvj={r["idx"]:r["judge_ok"] for r in load(f"ckpts/openvqa/cheap/ckpt_{ds}_7b_t0.judge.jsonl").values()}
ls=load(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.jsonl"); lsj={r["idx"]:r["judge_ok"] for r in load(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.judge.jsonl").values()}
idx=[i for i in mv if i in ls and i in mvj and i in lsj]
n=len(idx)
mvacc=np.mean([mvj[i] for i in idx]); lsacc=np.mean([lsj[i] for i in idx])
print(f"SLAKE-open n={n}: MedVLThinker-7B acc={mvacc:.3f}  Lingshu-7B acc={lsacc:.3f}\n")
ag=[i for i in idx if agree(mv[i]["modal_pred"], ls[i]["modal_pred"])]
dis=[i for i in idx if i not in ag]
print(f"AGREE   n={len(ag)} ({len(ag)/n:.0%})  -> Lingshu acc on agree = {np.mean([lsj[i] for i in ag]):.3f}  MedVL acc = {np.mean([mvj[i] for i in ag]):.3f}")
print(f"DISAGREE n={len(dis)} ({len(dis)/n:.0%}) -> Lingshu acc = {np.mean([lsj[i] for i in dis]):.3f}  MedVL acc = {np.mean([mvj[i] for i in dis]):.3f}")
# consensus selector: if agree -> take agreed answer; if disagree -> take the stronger model's (Lingshu)
def consensus(i):
    if i in set(ag): return lsj[i]   # agreed answer (both ~same; use Lingshu's judge)
    return lsj[i]                      # disagree -> trust stronger model
csel=np.mean([consensus(i) for i in idx])
print(f"\nconsensus selector acc = {csel:.3f}  (vs Lingshu-only {lsacc:.3f})")
print(f"AGREEMENT as correctness signal: P(correct|agree)={np.mean([lsj[i] for i in ag]):.3f} vs P(correct|disagree)={np.mean([lsj[i] for i in dis]):.3f}")
# coverage-accuracy: answer only on agree
print(f"\nIf we AUTO-ANSWER only on cross-family agreement: coverage={len(ag)/n:.0%}, accuracy={np.mean([lsj[i] for i in ag]):.3f}")
print("READ: high P(correct|agree) AND high coverage => cross-family agreement is a strong selector (escapes luck floor).")
