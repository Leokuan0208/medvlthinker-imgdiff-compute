#!/usr/bin/env python3
"""
triage_3way.py - the substantial new method: a unified 3-WAY TRIAGE for open-ended medical VLMs that
combines a cost cascade with safe abstention. Each query is routed by the CHEAP model's open-ended
confidence into one of three actions:
  ANSWER-cheap   (conf >= tau_high)          -- the cheap model is reliable here
  ESCALATE       (tau_low <= conf < tau_high)-- hand to the strong model (recover the fixable errors)
  ABSTAIN/refer  (conf < tau_low)            -- irreducibly hard: refer to a clinician
This dominates 2-way abstention (answer/abstain) when a real recovery gap exists, and degrades GRACEFULLY
to 2-way when it does not (e.g. Kvasir, where the strong model is no better). We trace the risk-coverage
frontier (auto-answered fraction vs error among answered) for 2-way vs 3-way, and report the escalation
(compute) cost of 3-way. Offline; LLM-judge labels. Emits paper/figs/open/fig_triage.png + JSON.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def load(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def judged(d, jp):
    if os.path.exists(jp):
        j = {r["idx"]: r["judge_ok"] for r in (json.loads(l) for l in open(jp) if l.strip())}
        for i, r in d.items():
            if i in j: r["modal_ok"] = j[i]
    return d
LC = "ckpts/openvqa/cheap_lingshu7b"; LS = "ckpts/openvqa/strong_lingshu"

def get(dss):
    conf, ccor, scor = [], [], []
    for ds in dss:
        c = judged(load(f"{LC}/ckpt_{ds}_lingshu7b.jsonl"), f"{LC}/ckpt_{ds}_lingshu7b.judge.jsonl")
        s = judged(load(f"{LS}/ckpt_{ds}_lingshu32b.jsonl"), f"{LS}/ckpt_{ds}_lingshu32b.judge.jsonl")
        for i in set(c) & set(s):
            conf.append(c[i].get("seqlogprob") or 0.0); ccor.append(c[i]["modal_ok"]); scor.append(s[i]["modal_ok"])
    return np.array(conf), np.array(ccor), np.array(scor)

def frontier_2way(conf, ccor):
    """answer top-c by conf with the cheap model; risk among answered."""
    order = np.argsort(-conf); cor = ccor[order]; n = len(cor); k = np.arange(1, n+1)
    cov = k / n; risk = 1 - np.cumsum(cor) / k
    return cov, risk
def frontier_3way(conf, ccor, scor):
    """For each coverage c (= answer the top-c by conf), choose the split a (answer-cheap top-a, escalate
    the band [a,c]) that MINIMISES risk. Escalated band answered by the strong model. Returns cov, best risk,
    escalation fraction at the best split."""
    order = np.argsort(-conf); cc = ccor[order]; ss = scor[order]; n = len(cc)
    cum_c = np.cumsum(cc); cum_s = np.cumsum(ss)            # prefix correct counts (sorted by conf desc)
    covs, risks, escs = [], [], []
    for c in range(1, n+1):
        # answered set = ranks [0,c). answer-cheap [0,a), escalate [a,c). minimise errors over a in [0,c].
        # correct(a) = cum_c[a-1] + (cum_s[c-1]-cum_s[a-1]); maximise -> pick best a.
        a_idx = np.arange(0, c+1)
        cheap_correct = np.concatenate([[0], cum_c[:c]])              # cum_c[a-1], a=0..c
        strong_correct_band = cum_s[c-1] - np.concatenate([[0], cum_s[:c]])
        total_correct = cheap_correct + strong_correct_band
        a_best = int(np.argmax(total_correct))
        risk = 1 - total_correct[a_best] / c
        covs.append(c/n); risks.append(risk); escs.append((c - a_best)/n)
    return np.array(covs), np.array(risks), np.array(escs)

def cov_at_risk(cov, risk, r):
    ok = np.where(risk <= r)[0]; return float(cov[ok[-1]]) if len(ok) else 0.0

SETS = {"SLAKE": ["slake_open"], "competent-3": ["slake_open", "vqa_rad_open", "pathvqa_open"],
        "all-4 (+Kvasir OOD)": ["slake_open", "vqa_rad_open", "pathvqa_open", "kvasir_open"]}
out = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, dss) in zip(axes, SETS.items()):
    conf, ccor, scor = get(dss)
    c2, r2 = frontier_2way(conf, ccor)
    c3, r3, e3 = frontier_3way(conf, ccor, scor)
    ca2 = {r: cov_at_risk(c2, r2, r) for r in (0.05, 0.10)}
    ca3 = {r: cov_at_risk(c3, r3, r) for r in (0.05, 0.10)}
    # escalation cost at the 5%-risk operating point of the 3-way
    idx5 = np.where(r3 <= 0.05)[0]; esc5 = float(e3[idx5[-1]]) if len(idx5) else 0.0
    out[name] = dict(n=len(conf), cheap_acc=float(ccor.mean()), strong_acc=float(scor.mean()),
        cov5_2way=ca2[0.05], cov5_3way=ca3[0.05], cov10_2way=ca2[0.10], cov10_3way=ca3[0.10],
        escalation_at_5pct=esc5)
    ax.plot(c2*100, r2*100, "-", c="#d62728", lw=1.8, label=f"2-way (answer/abstain) cov@5%={ca2[0.05]:.2f}")
    ax.plot(c3*100, r3*100, "-", c="#2ca02c", lw=2, label=f"3-way (+escalate) cov@5%={ca3[0.05]:.2f}")
    ax.axhline(5, ls="--", c="k", lw=0.7)
    ax.set_xlabel("coverage (% auto-answered)"); ax.set_ylabel("risk (% error among answered)")
    ax.set_ylim(0, 55); ax.set_title(f"{name}\ncheap {ccor.mean():.2f} / strong {scor.mean():.2f}", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    print(f"{name:<22} cheap={ccor.mean():.3f} strong={scor.mean():.3f} | cov@5%: 2way={ca2[0.05]:.2f} 3way={ca3[0.05]:.2f}"
          f" (+{ca3[0.05]-ca2[0.05]:+.2f}, escalate {esc5:.2f}) | cov@10%: 2way={ca2[0.10]:.2f} 3way={ca3[0.10]:.2f}")
fig.suptitle("3-way triage (answer / escalate / abstain) vs 2-way abstention — open-ended medical VLM (Lingshu-7B→32B, judge)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs("paper/figs/open", exist_ok=True)
fig.savefig("paper/figs/open/fig_triage.png", dpi=140); print("-> paper/figs/open/fig_triage.png")
json.dump(out, open("results/cascade_methods/artifacts/triage_3way.json", "w"), indent=1)
