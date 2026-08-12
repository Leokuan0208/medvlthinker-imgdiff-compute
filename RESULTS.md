# RESULTS — where every number lives

> **Updated 2026-07-29.** This file is an **index**, not a numbers source. Its job is to tell you which
> file to open for a given figure. The definitive account of the project — arc, method, results with
> CIs, negatives, holes, corrections — is
> **[`results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`](results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md)**.
>
> The lower half of this file (from "PART B") is the **2026-06 MedVLThinker checkpoint index**, kept
> verbatim as the historical record. It is still correct *for that era's checkpoints*; it is not the
> current result.

---

# PART A — the current results (2026-07, Lingshu + MedEvalKit)

## A.0 The one canonical headline

**Method:** a format-aware adaptive cascade, Lingshu-7B → Lingshu-32B, on the MedEvalKit suite.
**Pool:** "Variant B" — **MMMU excluded, 5 benchmarks / 8 cells / n = 42,224**.
**Baseline:** always-32B-with-reasoning = **0.5591 (measured)**.

| operating point | accuracy | compute (× one 32B pass) | latency (par.) | Δ vs 32B-reasoning, 95% CI | source artifact |
|---|---|---|---|---|---|
| compute-lean | 0.5741 | **0.49×** | 469 ms | **+0.0150 [+0.0107, +0.0192]** SIG | `f8_mode_vsthink_ci.json` |
| **accuracy-max** (veto + L2D) | 0.5836 | **0.93×** | 731 ms | **+0.0245 [+0.0216, +0.0274]** SIG | **`f8_mode_vsthink_ci.json` ← CANONICAL** |
| accuracy-max⁺ (fusion) | 0.5862 | 1.25× | 668 ms | +0.0271 [+0.0237, +0.0305] SIG | `opentext_32b_think_full.json` |

Other baselines: always-7B 0.5549 (1.00, 347 ms) · always-32B-direct 0.5729 (4.57, 665 ms) ·
oracle mode-select 32B 0.5730 (4.57, 860 ms — the fairest strong baseline, not deployable).

> ⚠️ **Five values circulate for the accuracy-max row** — `+0.0212`, `+0.0207`, `+0.0238`, `+0.0245`,
> `+0.0271`, `+0.0275`. They are the **same method** measured with a different **lever** (certified veto
> vs fusion), a different **pool** (MMMU kept / escalated / excluded), and **estimated vs measured**
> open-text think cells. **Only `+0.0245` is canonical.** Decode table: retrospective §10.3, and
> `INCONSISTENCIES.md` if a July section has been added there.
>
> ⚠️ **Do not write "Baselines (measured): 0.5632".** That figure's open-text cells were *estimates*;
> the measurement on 2026-07-08 moved it to **0.5594 (full suite) / 0.5591 (Variant B)**. Retrospective
> §10.2 X8.

## A.1 Where each current number is computed

| number | script | → artifact |
|---|---|---|
| **the canonical headline CI** | `src/cascade_methods/f8_mode_vsthink_ci.py` | `artifacts/f8_mode_vsthink_ci.json` |
| both knob settings, v1 (fusion) + v2 (veto+L2D) levers | `src/cascade_methods/method_final.py` | `artifacts/method_final.json`, `artifacts/method_final_v2.json` |
| MMMU Variants A/B × 3 modes, oracle-mode baseline, Pareto frontier | `src/cascade_methods/method_final_mmmu_corrected.py` | `artifacts/method_final_mmmu_corrected.json` |
| main baseline table + all paired bootstrap CIs | `src/cascade_methods/paper_baselines.py` | `artifacts/paper_baselines.json` |
| **measured** 32B-reasoning open-text accuracy (replaced the n=200 estimate) | `src/cascade_methods/opentext_32b_think_full.py` | `artifacts/opentext_32b_think_full.json` |
| the format-aware cascade scored vs always-32B-think | `src/cascade_methods/integrated_method.py` | `artifacts/integrated_method_vs_think.json` |
| adaptive-N (Weitzman) controller | `src/cascade_methods/pandora_controller.py`, `integrated_pandora.py` | `artifacts/pandora_controller.json`, `artifacts/integrated_pandora_opentext.json` |
| PMC fusion / certified veto / learning-to-defer levers | `src/cascade_methods/beat32b_fusion.py`, `beat32b_more.py` | `artifacts/beat32b_fusion.json`, `artifacts/beat32b_more.json` |
| speed levers (incl. prefill prefetch, −12.1% latency) | `src/cascade_methods/escalation_levers.py`, `escalation_more.py` | `artifacts/escalation_levers.json`, `artifacts/escalation_more.json` |
| cross-family generalization of the three findings | — | `artifacts/GENERALIZATION.md`, `artifacts/master_data.csv` |

**Launch discipline:** run everything as `python3 src/cascade_methods/<x>.py` **from the repo root** —
these modules use bare sibling imports and a hard-coded `ROOT`.
`paper_baselines.build_cells()` completes in ~42 s, CPU only, returning all 9 cells
(33,430 / 836 / 251 / 3,362 / 2,000 / 150 / 645 / 200 / 1,500 = 42,374).

## A.2 The evaluation contexts — never cross-multiply them

There are **three** distinct evaluation contexts in this repo (retrospective §9.3):

| context | harness | pool | n |
|---|---|---|---|
| **A. Faithful MedEvalKit** (the paper's suite) | `MedEvalKit/eval.py`, vLLM, seed 42 | 7 benchmarks | PMC 33,430 · SLAKE 2,094 · VQA-RAD 451 · PathVQA 6,719 · MMMU 150 · MedXpert 2,000 · OmniMed 88,996 |
| **B. Internal NGC harness** (the 5-family bake-off, PART B below) | `src/labeling/*`, `ckpts/acc_gen/` | 6 benchmarks | 8,220 total |
| **C. Custom open-text pipeline** (LLM-judged) | `run_openvqa.py` + `run_judge.py` | 5 open sets | 645 · 200 · 1,500 · 1,200 · 2,000 |

**The headline pool (n = 42,374 / 42,224) is a splice of A and C** — 40,029 MCQ cells from A plus 2,345
open cells from C. It is not a single harness run. Also: **MedEvalKit's open-half exact match is broken**
(gold `"CT"` vs response `"CT."` scores incorrect while ROUGE-1 ≈ 1.0), so only the `close` sub-metrics
of SLAKE / VQA-RAD / PathVQA are usable from that harness; **the open halves must be judged.**

## A.3 Current-era checkpoints (gitignored)

| path | what it holds |
|---|---|
| `ckpts/train/lora_verifier_pooled4` | **the deployed trained outcome verifier** (LoRA), + the other verifier training runs |
| `ckpts/openvqa/` | open-text generations (incl. `strong_lingshu`, `strong_lingshu_think`) + judge labels |
| `ckpts/ground/` | grounding / bounding-box outputs (SLAKE organs, MS-CXR chest X-ray) |
| `ckpts/pairwise/`, `ckpts/pairwise_diverse/` | real A-vs-B pairwise verifier verdicts (both orders) |
| `ckpts/acc_gen/` | the 5-family + peer-architecture mode dumps (incl. `internvl25_8b_think`, `phi35v_think`) |
| `ckpts/gate_lingshu7b_mcq/`, `ckpts/gate_lingshu32b_mcq/` | Lingshu MCQ signal dumps for the gate bake-off |
| `ckpts/peer/`, `ckpts/mcq_gen_verify/` | peer-model eval; MCQ generative-verification dumps |
| `MedEvalKit/eval_results_*/` | **the faithful evaluation dumps** — gitignored *vendor* territory; do not clean |
| `results/cascade_methods/artifacts/` | ~107 numeric `.json` outputs of the analysis scripts |

## A.4 Read the weaknesses before quoting anything

Retrospective §7 ranks **16 holes**. The three critical ones:

1. **The "always-32B-with-reasoning" baseline is a NO-reasoning run on ~90% of the pool, charged at
   reasoning cost.** The `..._think` dumps for PMC / SLAKE-closed / VQA-RAD-closed average 3–4 generated
   tokens; PathVQA-closed has no reasoning dump at all. Honest re-costing moves the cost headline from
   −95.5%/−96% to roughly **−72%/−74%**.
2. **89% of the headline delta comes from 2 of 8 cells** (PathVQA-open 51%, PMC-VQA 38%); against the
   deployable always-32B-direct baseline, four cells contribute **exactly 0.0000**. "Pareto-dominates"
   is a *sample-weighted* statement — the MCQ-only **macro** average is a significant *loss*
   (−0.0080 [−0.0138, −0.0025] vs oracle-mode).
3. **The open-text verifier scores items it was trained on** (70/30 grouped split, scored over the full
   sets) — about **31% inflation** measured on selection gain.

---

# PART B — the 2026-06 MedVLThinker record (historical, kept verbatim)

> This is the original MCQ efficiency result and its checkpoint index. Evaluation **context B**
> (internal NGC harness, 6 benchmarks, 8,220 samples). Still correct for those checkpoints; **not** the
> current result and **not** comparable to PART A.

**The era's headline.** Frozen margin gate **τ = 0.426**, cheap leg at **cap320**: parity with
always-32B (0.5718 vs 0.572) at **73.6%** of always-32B prefill-inclusive compute, 63.3% escalation.
Its successor, the 3-tier compute-configuration cascade (**ACC**: 7B-nothink@cap320 →
32B-NOTHINK@cap320 → 32B-think@fullres), at parity with always-32B-think: **ALL-6 latency 11.34 s →
2.27 s (−80%), FLOPs 100 → 52%, energy 6,318.8 J → 1,181.9 J (~5.3×)**; ALL-5 8.88 s → 0.44 s,
FLOPs 100 → 25%. Canonicalized in `INCONSISTENCIES.md` X1 → `artifacts/master_data.csv`.
Spec: `results/cascade_methods/docs/archive_2026-07/METHOD_ACC.md`. Reproduce: `python3 src/cascade_methods/acc.py`.
*Honest note:* ACC's agreement gate is prior art (Agreement-Based Cascading, arXiv 2407.02348).

## Canonical benchmark names (this era)
| parquet `dataset_name` | our name | paper column | n (closed subset) |
|---|---|---|---|
| pmc_vqa         | PMC-VQA  | PMC      | 2000 |
| MMMU-medical    | MMMU     | MMMU     | 170  |
| MedXpertQA-MM   | MedX-M   | MedX-M   | 2000 (= our R 1446 + U 554) |
| pathvqa_closed  | PathVQA  | PathVQA  | 3362 |
| slake_closed    | SLAKE    | SLAKE    | 416  |
| vqa_rad_closed  | VQA-RAD  | VQA-Rad  | 272  |

## Result folders
- ckpts/gate_7b_vllm/            7B no_think, full-res, EVAL    (all 6)
- ckpts/gate_7b_prune/cap{640,320,160,80}/  7B no_think, capped, EVAL  (all 6; MedXpert+MMMU present)
- ckpts/gate_32b/               32B think, full-res, EVAL       (all 6)
- archive/single-model-routing/gate_7b_rag_axes/  (was ckpts/gate_7b_v2) 7B n~500 grid over think/nothink/RAG axes — KILLED RAG experiment, archived
- ckpts/gate_7b_pmctrain/       7B no_think, full-res, CALIB (PMC-train, 3000)
- ckpts/gate_7b_pmctrain_prune/cap{...}/  7B no_think, capped, CALIB (3000 each)
- ckpts/router_margin.pkl       frozen margin gate, tau=0.426

## Cascade-methods research-loop artifacts (2026-06-17/18)
- ckpts/gate_32b_modes/{nothink_fullres,nothink_cap320,think_cap320}/  32B strong-leg ablation, EVAL (all 6).
    KEY FINDING: on the 4 competent perception sets, 32B NO-THINK >= 32B-think (SLAKE 0.841 vs 0.764,
    VQA-RAD 0.893 vs 0.776) at ~2 decode tok vs ~477 — thinking overthinks perception VQA.
    (Operative deltas at the deployed cap320: SLAKE +0.085, VQA-RAD +0.077 — `INCONSISTENCIES.md` X6.)
- ckpts/gate_32b_pmctrain/                       32B think, full-res, CALIB (PMC-train, 3000) — strong-model labels.
- ckpts/gate_32b_pmctrain_nothink_{cap320,fullres}/  32B no_think CALIB (for the ACC tier-1 gate).
- ckpts/gate_7b_verify/, gate_7b_verify_cap80/, gate_7b_pmctrain_verify*/  P(True) self-verification passes.
- results/cascade_methods/artifacts/latency_{7b,32b}.jsonl  REAL batch-1 per-config latencies
    (7B-nothink 0.18s | 32B-nothink@cap320 0.34s | 32B-think@fullres ~11s (batch-1; rt_cascade co-resident ~28s)).

## File naming
- EVAL  files: ckpt_<benchmark>_<mode>_norag_s<k>of<N>.jsonl   (mode = nothink | think)
- CALIB files: ckpt_<mode>_s<k>of<N>.jsonl
- row schema: {idx, gold, pred, ok, parse_ok, opt_logprobs{A..}, gen_tokens, raw_output}
- Single-shard runs carry **no** `_sKofN` suffix; only genuinely sharded runs are tagged.

## Validation status (2026-06-10)
- Harness CORRECT for this era: 7B-think and 32B-think both reproduce the paper (PMC, MedX-M) within ~1pt.
  *(Superseded 2026-07-01: this internal harness is **not** faithful to Lingshu's published numbers —
  MedEvalKit is. See PART A.)*
- Subsets = paper's CLOSED subsets (vqa_rad 272, slake 416, pathvqa 3362).
- no_think = a faster mode, ~+2-3pt on perception sets, ~flat on MedXpert.

## Known gaps — status
- MMMU: run on both models at all caps (170). [resolved 2026-06-18]
- MedXpert: present at all caps. [resolved 2026-06-18]
- 7B think: the full 8,220-row dumps exist (`ckpts/gate_7b_think/`, 2 shards). *(An earlier note here said
  "only PMC + MedXpert at n~500" — that was stale; corrected 2026-07-29.)*

## The era's two outcomes (see `results/cascade_methods/README.md`)
1. **The GATE is signal-limited & saturated.** No training-free gate (confidence / conformal / learned /
   recoverability / self-verification) beats the margin gate in a way that is simultaneously novel,
   real-efficiency-positive, and per-benchmark guardrail-safe. Recoverability tops out at ~0.6 AUROC.
   *This survived every later test and is now stated as the recoverability wall (16 mechanisms).*
2. **ACC** is the structural win — see the era headline above. *Scope: the 4 competent benchmarks
   (MMMU / MedXpert excluded, both near chance).* On Lingshu, ACC's slow think tier would fire ~0% of
   the time, so the final method does not deploy it; what carried forward is the **structure** and the
   compute-configuration idea, not the gate.
