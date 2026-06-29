# Unified-method experiment log (open-text medical VQA) — running record
Goal: a unified test-time-compute method (cascade gate + verifier) that beats the strong 32B on the open-text
suite (accuracy + FLOPs; latency/energy where the family allows), across MedVLThinker, Lingshu, (InternVL3).
Datasets: SLAKE, VQA-RAD, PathVQA, Kvasir (in-dist) + RadImageNet (HELD-OUT OOD). Judge-based accuracy.
All inference captures: per-sample preds, judge label, margin/conf/seqlogprob (gate signals), gen_tokens(+all), lat_s.
Cost model: FLOPs=2N(P+G) from gen_tokens; latency/energy from measured batch-1 per-tier fits (per family).
Drive: keep main <80% (big data -> /data; no OmniMed).

## Phase 0 — Baselines (greedy / SC / oracle@8 / 32B), judge-based
See OPENTEXT_BASELINE.md. Lingshu complete (7B 5ds, 32B 4ds). MedVLThinker gap-fill RUNNING (7B kvasir/radimagenet
+ 32B all-5, instrumented). [TABLE TO FILL WHEN RUNS FINISH]

## Phase 1 — Best cascade gate (escalate cheap->strong)
Candidates: confidence/seqlogprob, margin(top1-top2), self-consistency(n_distinct), verifier-confidence(max sc),
CASP-stability(trained logistic on cheap signals), learned/MLP. Metric: acc-vs-escalation frontier vs always-32B.
[RESULTS TBD]

## Phase 2 — Best verifier setup (best-of-N selection)
Vary N in {2,4,8,(16)}; selection rule (argmax confirmed best > weighted-vote); which model generates the samples;
verifier base. Metric: gap-captured, acc, cost. [RESULTS TBD]

## Phase 3 — Integration (the two methods + variants)
M1 (router): detect MCQ vs open-text -> cascade vs verifier. (Open-text suite => verifier path mostly.)
M2 (unified pipeline): every prompt -> verifier-pick -> gate -> escalate; per-tier (7B-nt, 32B-nt, 32B-think for MedVLThinker).
Variants: unified trained module (verifier+gate jointly) vs separate; agreement on the verifier-picks of the two small models.
Metric: acc + FLOPs + latency + energy vs always-32B, in-dist + held-out OOD. [RESULTS TBD]

## Findings (append as they land)
- (pending baselines)
