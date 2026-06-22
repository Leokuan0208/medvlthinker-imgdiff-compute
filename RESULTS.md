# Results index — MedVLThinker efficiency cascade

## Canonical benchmark names (use everywhere)
| parquet `dataset_name` | our name | paper column | n (closed subset) |
|---|---|---|---|
| pmc_vqa         | PMC-VQA  | PMC      | 2000 |
| MMMU-medical    | MMMU     | MMMU     | 170  |
| MedXpertQA-MM   | MedX-M   | MedX-M   | 2000 (= our R 1446 + U 554) |
| pathvqa_closed  | PathVQA  | PathVQA  | 3362 |
| slake_closed    | SLAKE    | SLAKE    | 416  |
| vqa_rad_closed  | VQA-RAD  | VQA-Rad  | 272  |

## Result folders (current layout)
- ckpts/gate_7b_vllm/            7B no_think, full-res, EVAL    (all 6)
- ckpts/gate_7b_prune/cap{640,320,160,80}/  7B no_think, capped, EVAL  (all 6; MedXpert+MMMU now present)
- ckpts/gate_32b/               32B think, full-res, EVAL       (all 6)
- archive/single-model-routing/gate_7b_rag_axes/  (was ckpts/gate_7b_v2) 7B n~500 grid over think/nothink/RAG axes — KILLED RAG experiment, archived
- ckpts/gate_7b_pmctrain/       7B no_think, full-res, CALIB (PMC-train, 3000)
- ckpts/gate_7b_pmctrain_prune/cap{...}/  7B no_think, capped, CALIB (3000 each)
- ckpts/router_margin.pkl       frozen margin gate, tau=0.426

## NEW (2026-06-17/18) — cascade-methods research-loop artifacts
- ckpts/gate_32b_modes/{nothink_fullres,nothink_cap320,think_cap320}/  32B strong-leg ablation, EVAL (all 6).
    KEY FINDING: on the 4 competent perception sets, 32B NO-THINK >= 32B-think (SLAKE 0.841 vs 0.764,
    VQA-RAD 0.893 vs 0.776) at ~2 decode tok vs ~477 — thinking overthinks perception VQA.
- ckpts/gate_32b_pmctrain/                       32B think, full-res, CALIB (PMC-train, 3000) — strong-model labels.
- ckpts/gate_32b_pmctrain_nothink_{cap320,fullres}/  32B no_think CALIB (for the ACC tier-1 gate).
- ckpts/gate_7b_verify/, gate_7b_verify_cap80/, gate_7b_pmctrain_verify*/  P(True) self-verification passes.
- results/cascade_methods/latency_{7b,32b}.jsonl  REAL batch-1 per-config latencies
    (7B-nothink 0.18s | 32B-nothink@cap320 0.34s | 32B-think@fullres ~28s).

## File naming
- EVAL  files: ckpt_<benchmark>_<mode>_norag_s<k>of<N>.jsonl   (mode = nothink | think)
- CALIB files: ckpt_<mode>_s<k>of<N>.jsonl
- row schema: {idx, gold, pred, ok, parse_ok, opt_logprobs{A..}, gen_tokens, raw_output}

## Proposed clean layout (when we migrate, after the remaining runs land)
runs/eval/<model>_<mode>/<resolution>/     e.g. runs/eval/7b_nothink/cap320/
runs/calib/7b_nothink/<resolution>/        (PMC-train, for tau)
runs/artifacts/router_margin.pkl
  resolution in {fullres, cap640, cap320, cap160, cap80}

## Validation status (2026-06-10)
- Harness CORRECT: 7B-think and 32B-think both reproduce the paper (PMC, MedX-M) within ~1pt.
- Subsets = paper's CLOSED subsets (vqa_rad 272, slake 416, pathvqa 3362).
- no_think = a faster mode, ~+2-3pt on perception sets, ~flat on MedXpert.

## Known gaps (Task 3) — mostly RESOLVED as of 2026-06-18
- MMMU: now run on both models at all caps (170). [resolved]
- MedXpert: now present at all caps. [resolved]
- 7B think: still only PMC + MedXpert at n~500 (not needed for the cascade — the cheap leg is no-think).

## Cascade-methods research loop (2026-06-18) — see results/cascade_methods/README.md
Goal: beat the deployed margin gate on compute/latency at iso-accuracy. Outcome:
- The GATE is signal-limited & saturated: no training-free gate (confidence/conformal/learned/
  recoverability/self-verification) beats the margin gate in a way that is novel + real-efficiency-
  positive + guardrail-safe (recoverability AUROC ceiling ~0.6 from any cheap signal).
- WINNER = **Adaptive-Compute Cascade (ACC)**: confidence-gated 3-tier 7B-nothink@cap320 →
  32B-NOTHINK@cap320 → 32B-think@fullres. The fast no-think 32B tier resolves most escalations; slow
  think fires only on the reasoning residual. Honest held-out, real measured latency, at parity acc:
  **ALL-6 latency 20.0s→5.7s (−72%), FLOPs 81→55%; ALL-5 9.1s→0.28s (−97%), FLOPs 51→27%;
  guardrail-cleaner than the SOTA cascade.** Scope: 4 competent benchmarks (MMMU/MedXpert excluded,
  both near chance). Novelty: incremental-but-defensible systems contribution (closest prior art:
  CAR 2505.15154; see results/cascade_methods/METHOD_ACC.md). Reproduce: `python3 src/cascade_methods/acc.py`.
