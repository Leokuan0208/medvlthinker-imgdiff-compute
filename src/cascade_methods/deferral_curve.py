#!/usr/bin/env python3
"""
deferral_curve.py - the canonical routing figure (reviewer-requested): cascade accuracy vs escalation
budget for RANDOM, the CONFIDENCE gate, and the ORACLE, on the same axes, for BOTH regimes:
  (A) MCQ  : MedVLThinker 7B-nt@cap320 -> 32B-think, competent-4 (PMC/SLAKE/VQA-RAD/PathVQA)
  (B) OPEN : Lingshu-7B -> Lingshu-32B, LLM-judge, SLAKE+VQA-RAD+PathVQA open-ended
The gap between CONFIDENCE and RANDOM (normalized by ORACLE-RANDOM) = how much routing signal is usable.
Reports AURC-style "routing efficiency" = area(confidence-random)/area(oracle-random). Offline.
Emits paper/figs/open/fig_deferral_curve.png + results/cascade_methods/deferral_curve.json.
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

def curves(cheap_ok, strong_ok, conf):
    co = np.array(cheap_ok, float); so = np.array(strong_ok, float); cf = np.array(conf, float); n = len(co)
    gain = so - co; B = np.linspace(0, 1, 21)
    def at(esc_mask): return np.where(esc_mask, so, co).mean()
    conf_c, orc_c, rnd_c = [], [], []
    conf_order = np.argsort(cf)            # escalate LOWEST confidence first
    orc_order = np.argsort(gain)[::-1]     # escalate highest-gain (recoverable) first
    rng = np.random.default_rng(0)
    for b in B:
        k = int(round(b * n))
        m1 = np.zeros(n, bool); m1[conf_order[:k]] = True; conf_c.append(at(m1))
        m2 = np.zeros(n, bool); m2[orc_order[:k]] = True; orc_c.append(at(m2))
        rs = []
        for _ in range(30):
            m3 = np.zeros(n, bool); m3[rng.permutation(n)[:k]] = True; rs.append(at(m3))
        rnd_c.append(np.mean(rs))
    conf_c, orc_c, rnd_c = map(np.array, (conf_c, orc_c, rnd_c))
    eff = np.trapz(conf_c - rnd_c, B) / np.trapz(orc_c - rnd_c, B)   # routing efficiency in [0,1]
    # DETECTION quality (selective prediction): does conf separate cheap-right from cheap-wrong?
    det = auroc(-cf, 1 - co)   # higher -conf ranks cheap-wrong; cw = 1-cheap_ok
    return B, conf_c, orc_c, rnd_c, eff, co.mean(), so.mean(), det

# (A) MCQ competent-4
COMP = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
co, so, cf = [], [], []
for ds in COMP:
    c = load(f"ckpts/gate_7b_prune/cap320/ckpt_{ds}_nothink_norag.jsonl"); s = load(f"ckpts/gate_32b/ckpt_{ds}_think_norag.jsonl")
    for i in set(c) & set(s):
        co.append(c[i]["ok"]); so.append(s[i]["ok"]); cf.append(margin(c[i].get("opt_logprobs")))
mcq = curves(co, so, cf)
# (B) open-ended Lingshu judge
co, so, cf = [], [], []
for ds in ["slake_open", "vqa_rad_open", "pathvqa_open"]:
    c = judged(load(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.jsonl"), f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.judge.jsonl")
    s = judged(load(f"ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.jsonl"), f"ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl")
    for i in set(c) & set(s):
        # cf = a CONFIDENCE score (higher = more confident); seqlogprob (mean logprob) is already that.
        co.append(c[i]["modal_ok"]); so.append(s[i]["modal_ok"]); cf.append(c[i].get("seqlogprob") or 0.0)
opn = curves(co, so, cf)

print("TWO DISTINCT AXES (the key honest distinction):")
print(f"  MCQ : DETECTION cheap-wrong AUROC={mcq[7]:.3f} | cascade HEADROOM (oracle-cheap)={mcq[2][-1]-mcq[5]:+.3f}"
      f" (cheap {mcq[5]:.3f}->strong {mcq[6]:.3f}); confidence routing eff (area/oracle)={mcq[4]:.3f}")
print(f"  OPEN: DETECTION cheap-wrong AUROC={opn[7]:.3f} | cascade HEADROOM (oracle-cheap)={opn[2][-1]-opn[5]:+.3f}"
      f" (cheap {opn[5]:.3f}->strong {opn[6]:.3f}); confidence routing eff (area/oracle)={opn[4]:.3f}")
print("READ: the ceiling-break is in DETECTION (0.6->0.87) — the cheap model's own error-detection. The")
print("realized CASCADE gain is recoverability/headroom-bounded (small in BOTH here), a model-pair property.")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ax, m, title in [(axes[0], mcq, "MCQ (MedVLThinker 7B→32B-think, competent-4)"),
                     (axes[1], opn, "Open-ended (Lingshu-7B→32B, LLM-judge)")]:
    B, conf_c, orc_c, rnd_c, eff, base, strong, det = m
    ax.plot(B*100, orc_c, "-", c="green", lw=2, label="oracle")
    ax.plot(B*100, conf_c, "o-", c="#1f77b4", ms=3, label="confidence gate")
    ax.plot(B*100, rnd_c, ":", c="gray", label="random")
    ax.fill_between(B*100, rnd_c, conf_c, color="#1f77b4", alpha=0.12)
    ax.set_xlabel("escalation budget (% routed to strong)"); ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.06, f"detection AUROC = {det:.2f}\ncascade headroom = {strong-base:+.03f}",
            transform=ax.transAxes, ha="center", fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="gray"))
axes[0].set_ylabel("cascade accuracy"); axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Deferral curves — detection (AUROC, the ceiling-break) is distinct from cascade gain "
             "(recoverability/headroom-bounded)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs("paper/figs/open", exist_ok=True)
fig.savefig("paper/figs/open/fig_deferral_curve.png", dpi=140); print("-> paper/figs/open/fig_deferral_curve.png")
json.dump({"mcq": {"detection_auroc": mcq[7], "headroom": mcq[6]-mcq[5], "cheap": mcq[5], "strong": mcq[6], "routing_eff": mcq[4]},
           "open": {"detection_auroc": opn[7], "headroom": opn[6]-opn[5], "cheap": opn[5], "strong": opn[6], "routing_eff": opn[4]}},
          open("results/cascade_methods/deferral_curve.json", "w"), indent=1)
