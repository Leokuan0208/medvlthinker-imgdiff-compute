#!/usr/bin/env python3
"""Does the higher-capacity 32B fix the 7B's GENUINELY-UNKNOWN errors (oracle@8 wrong = beyond the
sampling luck floor)? If yes -> the missing ingredient is capacity/knowledge (a real, non-luck lever).
If the 32B also fails them -> genuinely hard/ambiguous, no knowledge source helps. Split by content_type
(knowledge vs perception). All consistent judge (this session): 7B exploded judge + 32B-free compare judge."""
import json, os, numpy as np
from collections import defaultdict
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute"); J=lambda p:os.path.join(ROOT,p)
ck="ckpts/openvqa/cheap_lingshu7b"; ds="slake_open"; tag="lingshu7b"
def norm(s): return str(s).strip().lower()
sc={r["idx"]:r for r in (json.loads(l) for l in open(J(f"{ck}/ckpt_{ds}_{tag}_sc8.jsonl")) if l.strip())}
exp={r["idx"]:r for r in (json.loads(l) for l in open(J(f"{ck}/ckpt_{ds}_{tag}_sc8_scexploded.jsonl")) if l.strip())}
jud={r["idx"]:r["judge_ok"] for r in (json.loads(l) for l in open(J(f"{ck}/ckpt_{ds}_{tag}_sc8_scexploded.judge.jsonl")) if l.strip())}
aj=defaultdict(dict)
for cid,r in exp.items():
    if cid in jud:
        oi=cid.split("#")[0]; oi=int(oi) if oi.lstrip("-").isdigit() else oi
        aj[oi][norm(r["modal_pred"])]=jud[cid]
# 32B-free consistent judge from the compare file
cj={}
for r in (json.loads(l) for l in open(J("ckpts/openvqa/strong_lingshu/ckpt_slake_open_compare_scexploded.judge.jsonl")) if l.strip()):
    if r["idx"].endswith("#free"):
        oi=r["idx"][:-5]; oi=int(oi) if oi.lstrip("-").isdigit() else oi
        cj[oi]=r["judge_ok"]
ctype={x["qid"]:x.get("content_type") for x in json.load(open("/data/dan/dataset/slake/test.json"))
       if x.get("answer_type")=="OPEN" and x.get("q_lang")=="en"}
KNOW={"KG","Abnormality"}
# build genuinely-unknown set
gu=[]; allq=[]
for i in sc:
    if i not in aj or i not in cj or i not in ctype: continue
    sl=[aj[i].get(norm(a)) for a in sc[i]["preds"]]; sl=[x for x in sl if x is not None]
    if not sl: continue
    oracle=max(sl); allq.append(i)
    if oracle==0: gu.append(i)
def grp(i): return "KNOWLEDGE" if ctype[i] in KNOW else "perception"
print(f"SLAKE-open n={len(allq)}; 7B genuinely-unknown (oracle@8 wrong) = {len(gu)} ({len(gu)/len(allq):.1%})\n")
for lab,subset in [("ALL genuinely-unknown",gu),
                   ("  KNOWLEDGE-type GU",[i for i in gu if grp(i)=="KNOWLEDGE"]),
                   ("  perception-type GU",[i for i in gu if grp(i)=="perception"])]:
    if not subset: continue
    fix=np.mean([cj[i] for i in subset])
    print(f"{lab:<24} n={len(subset):>3}  32B-free fixes {fix:.1%}")
# context: 32B overall + on 7B-correct
print()
print(f"context: 32B-free overall acc = {np.mean([cj[i] for i in allq]):.3f}")
sevenok=[i for i in allq if aj[i].get(norm(sc[i]['modal_pred']),0)==1]
print(f"         32B on 7B-already-correct = {np.mean([cj[i] for i in sevenok]):.3f}  (breakage if <1)")
print()
print("READ: high 32B-fix on KNOWLEDGE-type GU => capacity/knowledge is the lever (escalate knowledge Qs).")
print("      low 32B-fix on GU => genuinely hard; neither sampling NOR a bigger model helps (true ceiling).")
