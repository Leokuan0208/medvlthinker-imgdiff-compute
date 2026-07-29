#!/usr/bin/env python3
"""
selective_abstain.py - NEW METHOD (Direction 1): training-free SELECTIVE PREDICTION / safe abstention for
OPEN-ENDED medical VLMs. Thesis: medical VLMs *do* know when they are wrong, but only in the OPEN-ENDED
regime; the MCQ "uncertainty fails for clinical VQA" result (2606.16583) is a benchmark artifact. We turn
the open-ended self-confidence into a DEPLOYABLE risk-coverage / clinician-referral system and report AURC
+ coverage@target-risk, contrasting MCQ vs open for the SAME models.

Metrics (selective prediction):
  - risk-coverage curve: answer the top-c fraction by confidence; risk = error rate among answered.
  - AURC (area under risk-coverage, lower=better) and E-AURC (excess over the oracle ordering).
  - coverage@risk<=r : max fraction answerable at risk <= r (e.g. 5%, 10%).
  - detection AUROC (confidence vs own correctness) -- the signal quality.
Offline; uses LLM-judge labels for open-ended. Emits paper/figs/open/fig_selective.png + JSON.
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
def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); P = s[y == 1]; N = s[y == 0]
    if len(P) == 0 or len(N) == 0: return float("nan")
    a = np.concatenate([P, N]); o = a.argsort(); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a)+1)
    u, inv, c = np.unique(a, return_inverse=True, return_counts=True); ss = np.zeros(len(c)); np.add.at(ss, inv, rk); rk = (ss/c)[inv]
    return (rk[:len(P)].sum() - len(P)*(len(P)+1)/2) / (len(P)*len(N))
def risk_coverage(conf, correct):
    """conf higher=more confident. Returns coverage grid, risk, AURC, E-AURC, cov@{5,10,20}%-risk."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, int); n = len(conf)
    order = np.argsort(-conf)                      # most confident first
    cor = correct[order]; cum_correct = np.cumsum(cor); k = np.arange(1, n+1)
    risk = 1 - cum_correct / k; cov = k / n
    aurc = float(np.mean(risk))
    # oracle: order correct-first -> minimal achievable risk at each coverage
    oc = np.sort(correct)[::-1]; orisk = 1 - np.cumsum(oc) / k; eaurc = aurc - float(np.mean(orisk))
    def cov_at(r):
        ok = np.where(risk <= r)[0]
        return float(cov[ok[-1]]) if len(ok) else 0.0
    return cov, risk, aurc, eaurc, {0.05: cov_at(0.05), 0.10: cov_at(0.10), 0.20: cov_at(0.20)}, (1-correct.mean())

OPENC = "ckpts/openvqa/cheap_lingshu7b"; OPEN_MED = "ckpts/openvqa/cheap"
def open_signals(ckdir, main_tag, sc_tag, dss):
    conf, sc, cor = [], [], []
    for ds in dss:
        base = f"{ckdir}/ckpt_{ds}_{main_tag}.jsonl"; jf = f"{ckdir}/ckpt_{ds}_{main_tag}.judge.jsonl"
        if not (os.path.exists(base) and os.path.exists(jf)): continue   # require judge for honest scoring
        c = judged(load(base), jf)
        scf = f"{ckdir}/ckpt_{ds}_{sc_tag}.jsonl"; s8 = load(scf) if os.path.exists(scf) else {}
        for i, r in c.items():
            conf.append(r.get("seqlogprob") or 0.0)
            sc.append(s8[i]["self_consistency"] if i in s8 else 0.0)
            cor.append(r["modal_ok"])
    return np.array(conf), np.array(sc), np.array(cor)
def mcq_signals(ckdir, tmpl, dss):
    conf, cor = [], []
    for ds in dss:
        p = f"{ckdir}/{tmpl.format(ds=ds)}"
        if not os.path.exists(p): continue
        d = load(p)
        for r in d.values():
            conf.append(margin(r.get("opt_logprobs"))); cor.append(r["ok"])
    return np.array(conf), np.array(cor)

OPEN4 = ["slake_open", "vqa_rad_open", "pathvqa_open", "kvasir_open"]
MCQ4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
results = {}
# Lingshu-7B (native open-ended model = the deployable subject)
lc, ls, lcor = open_signals(OPENC, "lingshu7b", "lingshu7b_sc8", OPEN4)
cov, risk, aurc, eaurc, ca, base = risk_coverage(lc, lcor)
_, _, aurc_sc, eaurc_sc, ca_sc, _ = risk_coverage(ls, lcor)
lmc, lmcor = mcq_signals("ckpts/acc_gen/lingshu7b/cap320", "ckpt_{ds}_lingshu7b.jsonl", MCQ4)
mcov, mrisk, maurc, meaurc, mca, mbase = risk_coverage(lmc, lmcor)
results["lingshu7b"] = dict(open=dict(n=len(lcor), base_risk=base, det_auroc=auroc(-lc, 1-lcor),
    aurc_conf=aurc, eaurc_conf=eaurc, covatrisk_conf=ca, aurc_selfcons=aurc_sc, covatrisk_selfcons=ca_sc),
    mcq=dict(n=len(lmcor), base_risk=mbase, det_auroc=auroc(-lmc, 1-lmcor), aurc=maurc, eaurc=meaurc, covatrisk=mca))
# MedVLThinker-7B (RL-on-MCQ model = the clearest MCQ-artifact contrast); open run = original 3 datasets
mc, ms, mcor = open_signals(OPEN_MED, "7b_t0", "7b_sc8", ["slake_open", "vqa_rad_open", "pathvqa_open"])
mo_cov, mo_risk, mo_aurc, mo_eaurc, mo_ca, mo_base = risk_coverage(mc, mcor)
mmc, mmcor = mcq_signals("ckpts/gate_7b_prune/cap320", "ckpt_{ds}_nothink_norag.jsonl", MCQ4)
mm_cov, mm_risk, mm_aurc, mm_eaurc, mm_ca, mm_base = risk_coverage(mmc, mmcor)
results["medvlthinker7b"] = dict(open=dict(n=len(mcor), base_risk=mo_base, det_auroc=auroc(-mc, 1-mcor),
    aurc=mo_aurc, eaurc=mo_eaurc, covatrisk=mo_ca),
    mcq=dict(n=len(mmcor), base_risk=mm_base, det_auroc=auroc(-mmc, 1-mmcor), aurc=mm_aurc, eaurc=mm_eaurc, covatrisk=mm_ca))

for m, r in results.items():
    print(f"\n== {m} ==")
    for reg in ("mcq", "open"):
        d = r[reg]; print(f"  {reg:<5} n={d['n']:<5} base_risk={d['base_risk']:.3f}  detection_AUROC={d['det_auroc']:.3f}  "
              f"AURC={d.get('aurc',d.get('aurc_conf')):.3f}  E-AURC={d.get('eaurc',d.get('eaurc_conf')):.3f}")
        ca = d.get('covatrisk', d.get('covatrisk_conf')); print(f"        coverage@risk: 5%={ca[0.05]:.2f} 10%={ca[0.10]:.2f} 20%={ca[0.20]:.2f}")

# per-dataset open-ended risk-coverage (deployability where it matters; base accuracy varies by dataset)
DSLAB = {"slake_open": "SLAKE", "vqa_rad_open": "VQA-RAD", "pathvqa_open": "PathVQA", "kvasir_open": "Kvasir(GI)"}
perds = {}
for ds in OPEN4:
    cc, ss2, ccor = open_signals(OPENC, "lingshu7b", "lingshu7b_sc8", [ds])
    if len(ccor) == 0: continue
    cv, rk, au, eau, ca2, br = risk_coverage(cc, ccor)
    perds[ds] = (cv, rk, ca2, br, au); results.setdefault("lingshu7b_perdataset", {})[ds] = dict(
        base_risk=br, aurc=au, eaurc=eau, covatrisk=ca2, det_auroc=auroc(-cc, 1-ccor), n=len(ccor))
# DEPLOYED model = the strongest available (Lingshu-32B) self-abstaining: higher base acc -> more coverage
LS = "ckpts/openvqa/strong_lingshu"
print("\n== Lingshu-32B SELF-abstention (the DEPLOYED model: auto-answer the confident, refer the rest) ==")
for ds in OPEN4:
    s = judged(load(f"{LS}/ckpt_{ds}_lingshu32b.jsonl"), f"{LS}/ckpt_{ds}_lingshu32b.judge.jsonl") if os.path.exists(
        f"{LS}/ckpt_{ds}_lingshu32b.jsonl") else {}
    if not s: continue
    cor = np.array([r["modal_ok"] for r in s.values()]); cf = np.array([r.get("seqlogprob") or 0.0 for r in s.values()])
    cv, rk, au, eau, ca2, br = risk_coverage(cf, cor)
    results.setdefault("lingshu32b_deployed", {})[ds] = dict(base_acc=float(cor.mean()), det_auroc=auroc(-cf, 1-cor),
        aurc=au, covatrisk=ca2, n=len(cor))
    print(f"  {DSLAB[ds]:<11} base_acc={cor.mean():.3f} det_AUROC={auroc(-cf,1-cor):.3f} AURC={au:.3f} "
          f"cov@5%={ca2[0.05]:.2f} cov@10%={ca2[0.10]:.2f}")

# figure: money plot (detection AUROC MCQ vs open) + per-dataset open-ended risk-coverage
fig, (a2, a1) = plt.subplots(1, 2, figsize=(12.5, 4.6))
COL = {"slake_open": "#2ca02c", "vqa_rad_open": "#1f77b4", "pathvqa_open": "#ff7f0e", "kvasir_open": "#9467bd"}
for ds, (cv, rk, ca2, br, au) in perds.items():
    a1.plot(cv*100, rk*100, "-", c=COL[ds], lw=1.8, label=f"{DSLAB[ds]} (AURC {au:.2f}, cov@5%={ca2[0.05]:.2f})")
a1.axhline(5, ls="--", c="k", lw=0.7); a1.set_xlabel("coverage (% auto-answered)")
a1.set_ylabel("risk (% error among answered)"); a1.set_ylim(0, 60)
a1.set_title("Deployable abstention (Lingshu-7B, open-ended):\nrisk-coverage per dataset; refer the rest to a clinician")
a1.legend(fontsize=7.5, loc="upper left"); a1.grid(alpha=0.3)
labels = ["MedVLThinker-7B", "Lingshu-7B"]
mcqA = [results["medvlthinker7b"]["mcq"]["det_auroc"], results["lingshu7b"]["mcq"]["det_auroc"]]
opnA = [results["medvlthinker7b"]["open"]["det_auroc"], results["lingshu7b"]["open"]["det_auroc"]]
x = np.arange(2); w = 0.35
a2.bar(x-w/2, mcqA, w, color="#d62728", label="MCQ (margin)")
a2.bar(x+w/2, opnA, w, color="#2ca02c", label="open-ended (confidence)")
for i in range(2):
    a2.text(i-w/2, mcqA[i]+0.01, f"{mcqA[i]:.2f}", ha="center", fontsize=8)
    a2.text(i+w/2, opnA[i]+0.01, f"{opnA[i]:.2f}", ha="center", fontsize=8)
a2.axhline(0.5, ls=":", c="gray"); a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=9)
a2.set_ylim(0.5, 0.95); a2.set_ylabel("self-error-detection AUROC")
a2.set_title("The 'money plot': error-detection lifts\nfrom MCQ to open-ended (same model)"); a2.legend(fontsize=8)
fig.suptitle("Selective abstention for open-ended medical VLMs (training-free)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs("paper/figs/open", exist_ok=True)
fig.savefig("paper/figs/open/fig_selective.png", dpi=140); print("\n-> paper/figs/open/fig_selective.png")
json.dump(results, open("results/cascade_methods/artifacts/selective_abstain.json", "w"), indent=1, default=float)
