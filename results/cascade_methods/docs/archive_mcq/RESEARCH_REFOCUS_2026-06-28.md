# Research refocus (2026-06-28): one method + peer baselines

## The focused thesis
**Test-time compute for medical VLMs: what actually helps.** Headline method = a small **trained LoRA
outcome verifier** for best-of-N selection on open-ended medical VQA (+ grounding). Efficiency companion =
ACC. The verifier is benchmarked against prestigious peers and beats them.

## Baseline landscape (full open-ended set, n=3545, LLM-judge; cheap=Lingshu-7B, strong=Lingshu-32B)
| dataset | greedy | self-consistency/majority | random | 32B single-pass (scale-up) | oracle@8 |
|---|---|---|---|---|---|
| SLAKE   | 0.722 | 0.736 | 0.720 | 0.819 | 0.879 |
| VQA-RAD | 0.420 | 0.465 | 0.441 | 0.600 | 0.630 |
| PathVQA | 0.295 | 0.324 | 0.294 | 0.376 | 0.517 |
| Kvasir  | 0.287 | 0.286 | 0.282 | 0.301 | 0.491 |
| POOLED  | 0.377 | 0.394 | 0.375 | 0.444 | 0.580 |
Trained verifier (held-out test split, n=1064): **pooled 0.501** (PathVQA 0.441, Kvasir 0.405, VQA-RAD 0.611, SLAKE 0.762).

## The novel finding (the hook)
- In the **general LLM** literature, trained-verifier best-of-N barely beats self-consistency
  (e.g., self-certainty BoN paper arXiv:2502.18581 NeurIPS'25; "optimal aggregation" 2510.13918).
- In **medical open-ended VQA, self-consistency essentially fails** (pooled 0.394 vs greedy 0.377;
  *below* greedy on Kvasir) — the **majority trap** (correct answer is a minority vote). Scaling to a 5×
  larger model only reaches 0.444.
- A **small trained verifier reaches 0.501** — beating every training-free selector AND the 32B —
  recovering 49% of the oracle gap. So medical VQA is a regime where a trained verifier is *essential*,
  not marginal. AUROC 0.924 (right vs wrong candidates).

## Peer baselines (provenance + why compared)
| peer | source / venue | why it's the right comparison | result vs ours |
|---|---|---|---|
| Greedy (single sample) | — | the deploy-default | ours +0.09 |
| Self-consistency / majority vote | Wang et al., ICLR 2023 | the standard training-free selector | ours ≫ (SC fails here) |
| Self-verification P(True) | Kadavath et al., 2022 (Anthropic) | the standard self-check baseline | ours > (P(True) luck-floored) |
| Self-certainty BoN | Zhao et al., NeurIPS 2025 (2502.18581) | recent training-free BoN SOTA | [needs logprobs; pending] |
| Generative verifier (GenRM) | Zhang et al., ICLR 2025 (2408.15240) | the trained-verifier family ours instantiates | ours = medical instance |
| Scale-up (32B single pass) | — | "just use a bigger model" | ours 0.501 > 0.444 |
| Cascade gates: FrugalGPT('23), ABC('24), CAR('25), CP-Router (Su'25), Jitkrittum L2D (NeurIPS'23) | various | ACC efficiency baselines (master_data.csv) | ACC frontier-best |

## Pending experiments (this session)
- clean_dump.json: verifier scores + judge labels + answers on the held-out split (for the fair same-split
  table + the verifier-augmented cascade frontier). RUNNING.
- Verifier-augmented open-ended cascade (cheap-7B-verifier-bo-N -> escalate residual to 32B): cost-accuracy
  frontier vs always-32B / cheap / no-verifier. OFFLINE once clean_dump lands.
- Cross-generator transfer (verifier on MedVLThinker-7B/3B samples): if GPU time permits.

## DATASET POLICY (locked, CORRECTED 2026-06-28 — applied consistently everywhere)
The "switch" between regimes is driven by ANSWER FORMAT + signal, not arbitrary selection. Per-benchmark
32B-think accuracy (master_data.csv): PMC 0.557 | SLAKE 0.764 | VQA-RAD 0.776 | PathVQA 0.673 |
MMMU-medical 0.688 | MedXpert-R 0.326 | MedXpert-U 0.385.

- **The consistent core thread (used in BOTH regimes): SLAKE, VQA-RAD, PathVQA** — competent AND have both
  MCQ and open-ended versions, so they bridge ACC (MCQ) and the verifier (open-ended).
- **ACC / efficiency (MCQ, closed-ended):** full 6-benchmark suite — PMC-VQA, SLAKE, VQA-RAD, PathVQA,
  MMMU-medical, MedXpert-MM. Report ALL-6 (all six), ALL-5 (drop MedXpert), competent-4 (PMC/SLAKE/VQA-RAD/PathVQA).
  - **MMMU-medical is COMPETENT (0.688), not near-chance.** But it is a *reasoning* benchmark where thinking
    HELPS (no-think 0.624 < think 0.688), unlike the perception sets — so it's in the tables but the
    over-thinking premise doesn't apply to it (state this honestly; ACC still routes it via the gate).
  - **MedXpert-MM** (0.33/0.39) is the genuinely near-chance/very-hard exam → excluded from headline efficiency claims.
  - **PMC-VQA** (0.557) is a core competent MCQ benchmark + the gate-calibration set; kept in ACC. It is
    MCQ-only, so it does NOT appear in the verifier work (no free-text answers to select among) — absence is
    by FORMAT, not a drop.
- **Verifier / accuracy (open-ended):** datasets with free-text answers — SLAKE, VQA-RAD, PathVQA (the core
  3) + Kvasir-VQA-x1 (OOD modality, GI endoscopy) + RadImageNet-VQA (held-out transfer). PMC-VQA and
  MMMU are MCQ-only ⇒ no open-ended best-of-N (excluded by format, not quality). MedXpert excluded (near-chance).
- **Grounding: SLAKE organs + MS-CXR** (have gold boxes).
- **Why MCQ→open-ended is necessary:** best-of-N selection is undefined on a single MCQ letter; §5.7 shows
  selection signal exists only open-ended (AUROC 0.6 MCQ → 0.87 open). The SAME 3 core datasets carry through.

## Definitions to include verbatim in the report (context the professor wants)
- **AUROC**: P(score(correct) > score(incorrect)); 0.5=random, 1.0=perfect. Rank statistic over all (sample,label) pairs.
- **FLOPs (cost)**: F = 2·N·(P+G), N=params, P=prompt tokens incl. vision (prefill), G=generated. PREFILL-INCLUSIVE. Normalized to always-32B-think=100%.
- **Latency**: batch-1 wall-clock per question, end-to-end (prefill+decode), measured isolated (measure_config.py). Canonical = native batch-1.
- **Energy**: NVML GPU power polled every 25ms, trapezoid-integrated over the query window (Joules).
- **Verifier architecture**: a LoRA adapter (rank-rxx) on Lingshu-7B (a 7B medical VLM); input = image+question+candidate; output = P(Yes) at the final token; trained by cross-entropy on judge/IoU labels. ~190MB adapter, base frozen.
- **LLM judge**: a strong neutral grader (MedVLThinker/Lingshu-32B) deciding if a free-text answer is semantically correct vs gold; chosen because exact-match is too brittle for free text. One consistent judge run.
- **Self-consistency (Wang et al., ICLR'23)**: sample N, majority-vote the answer. Training-free.
- **Self-verification P(True) (Kadavath et al., 2022)**: ask the model "is this answer correct?", use its Yes-probability. Training-free.
- **Self-certainty BoN (NeurIPS'25, 2502.18581)**: rank samples by the model's own output-distribution certainty. Training-free.
- **Generative verifier / GenRM (Zhang et al., ICLR'25, 2408.15240)**: train a verifier to score candidates for best-of-N. Ours is the medical-VQA + grounding instance.
- **Conformal prediction / CP-Router (Su et al., 2025)**: distribution-free uncertainty sets with coverage guarantees, used to route between a small LLM and a large reasoning model. A cascade-gate baseline.
- **FrugalGPT ('23) / ABC agreement ('24) / CAR ('25) / Jitkrittum L2D (NeurIPS'23)**: published cascade/routing baselines (in master_data.csv).
- **Oracle gap**: oracle@N − greedy, the unrealized headroom if you could always pick a correct sample.
- **Majority trap**: the correct answer is a minority vote among samples (74–90% of recoverable cases here), so majority voting picks the wrong one.
