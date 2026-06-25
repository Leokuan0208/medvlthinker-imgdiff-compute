#!/usr/bin/env python3
"""RAG feasibility: decompose SLAKE-open errors by content_type. For each type report greedy acc,
oracle@8 acc, and the GENUINELY-UNKNOWN fraction (neither greedy NOR any of 8 samples right = the
model lacks the fact, NOT sampling luck). If knowledge types (KG, Abnormality) have high
genuinely-unknown, external knowledge (RAG) has a real target. If errors are perception types or
luck-recoverable, RAG won't help. All from consistent exploded judge (this session)."""
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
ctype={}; trip={}
for x in json.load(open("/data/dan/dataset/slake/test.json")):
    if x.get("answer_type")=="OPEN" and x.get("q_lang")=="en":
        ctype[x["qid"]]=x.get("content_type"); trip[x["qid"]]=x.get("triple")
KNOW={"KG","Abnormality"}
rows=defaultdict(lambda:[0,0,0,0])  # type -> [n, greedy_ok, oracle_ok, genuinely_unknown]
for i in sc:
    if i not in aj or i not in ctype: continue
    t=ctype[i]
    sl=[aj[i].get(norm(a)) for a in sc[i]["preds"]]; sl=[x for x in sl if x is not None]
    if not sl: continue
    gok=aj[i].get(norm(sc[i]["modal_pred"]),0)
    orok=max(sl)
    r=rows[t]; r[0]+=1; r[1]+=gok; r[2]+=orok; r[3]+=(1-orok)
print(f"{'content_type':<13} {'n':>4} {'greedy':>7} {'oracle@8':>9} {'unknown%':>9}  {'class':>11}")
tot=[0,0,0,0]; ktot=[0,0,0,0]; ptot=[0,0,0,0]
for t in sorted(rows,key=lambda t:-rows[t][3]):
    n,g,o,u=rows[t]
    cls="KNOWLEDGE" if t in KNOW else "perception"
    print(f"{t:<13} {n:>4} {g/n:>7.3f} {o/n:>9.3f} {u/n:>9.1%}  {cls:>11}")
    for j in range(4): tot[j]+=rows[t][j]; (ktot if t in KNOW else ptot)[j]+=rows[t][j]
print("-"*60)
for lab,a in [("KNOWLEDGE(KG+Abn)",ktot),("perception(rest)",ptot),("ALL",tot)]:
    n,g,o,u=a; print(f"{lab:<18} n={n:>4} greedy={g/n:.3f} oracle@8={o/n:.3f} genuinely-unknown={u/n:.1%}")
print()
print("READ: high genuinely-unknown% in KNOWLEDGE types = RAG target (model lacks the fact, not luck).")
print("      if unknown% is similar across knowledge/perception, or dominated by perception, RAG won't help.")
