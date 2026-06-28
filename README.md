# medvlthinker-imgdiff-compute

**Compute-efficient medical VQA via a two-model cascade.** A cheap **7B** VLM answers
most multiple-choice medical-VQA questions; a frozen **confidence-margin gate** (τ = 0.426)
escalates only the low-confidence ones to an expensive **32B** model. Serving the 7B at a
reduced image resolution ("cap320"), the cascade reaches **exact accuracy parity** with
always-using-32B (0.572 = 0.572) at ~**74%** of the always-32B compute (prefill-inclusive
FLOPs), and is never worse than always-7B on any of the six benchmarks. Both models are
`MedVLThinker-*-RL_m23k`. Target venue: **CVGIP 2026**.

> Full project context, glossary, environment, and the rules for working in this repo are in
> **`CLAUDE.md`** — read it before running or moving anything. Headline numbers there are
> ground truth; never fabricate or edit experimental numbers.

## Layout

All active code lives under `src/`, grouped by pipeline stage. **Always run scripts from the
repo root** (e.g. `python3 src/cascade/live_cascade.py`) — several resolve `ckpts/...` paths
relative to the launch directory.

```
src/
├── labeling/      run a model over a dataset -> per-sample JSONL checkpoints
│                  run_7b_hf_labeler.py, run_7b_vllm.py, run_7b_think_vllm.py,
│                  run_32b_hf.py, run_32b_vllm.py, run_pmctrain_vllm.py
├── sweep/         resolution / compute sweeps + the calibrated res×τ grid
│                  run_7b_prune_sweep.py, tokens_per_cap.py, grid_resolution_tau.py,
│                  cascade_resolution_sweep.py, cascade_heldout_frontier.py
├── gate/          train + freeze the deployed margin gate (-> ckpts/router_margin.pkl)
│                  train_margin_gate.py, refit_gate_tau_per_cap.py
├── cascade/       the LIVE co-resident cascade + real-time measurement
│                  live_cascade.py, measure_single_leg.py, measure_config.py,
│                  report_cascade_from_legs.py, analyze_live_cascade.py
├── cascade_methods/  RESEARCH LOOP (2026-06): offline harness + every cascade method tried,
│                  incl. the winner ACC. harness.py, acc.py, frontier_compare.py,
│                  metarouter_honest.py, sota_comparison.py, latency_estimate.py, diagnostics.py
├── analysis/
│   ├── cascade/   analyses of the live cascade (cost, complementarity, mechanism, energy)
│   └── ablations/ gate alternatives that LOST to the margin gate (conformal, learned, FBE)
├── reporting/     build the paper's accuracy/efficiency tables + harness validation
├── data_prep/     build eval subsets, sample the held-out PMC-VQA train split
└── legacy_retrieval/  retrieve.py — leftover from the killed RAG direction

archive/            killed directions, kept as the record of negative results:
                    image-difficulty/, old-gate-scripts/, single-model-routing/
MedRAG/, MedVLThinker/   dependency repos — DO NOT move or rename
ckpts/ logs/ data/ results/ feats/ feats_full/   gitignored data/checkpoints
```

## Headline pipeline

1. **Label** the eval sets and the held-out PMC-VQA train split with the 7B (cheap, no-think)
   and 32B (think) — `src/labeling/`.
2. **Sweep** image-resolution caps and calibrate the (resolution, τ) operating point —
   `src/sweep/` → cap320, τ = 0.426.
3. **Train + freeze** the margin gate on the clean PMC-VQA train labels — `src/gate/`
   → `ckpts/router_margin.pkl`.
4. **Run the live cascade** co-resident (7B on GPU0, 32B on GPU1) with real escalation and
   NVML power logging — `src/cascade/live_cascade.py` → `rt_cascade_cap320.jsonl`.
5. **Analyze / report** — `src/analysis/` and `src/reporting/`.

## Cascade-method research (2026-06) — current frontier

After the deployed margin gate (above), an autonomous research loop searched for a better cascade
**method**. Two outcomes (full account in `results/cascade_methods/README.md`):

- **The gate is signal-limited.** No training-free decision rule (confidence / conformal / learned /
  recoverability / self-verification) beats the margin gate in a way that is simultaneously novel,
  real-efficiency-positive, and per-benchmark guardrail-safe. ("Will the 32B fix it?" is ~0.6 AUROC
  from any cheap signal.) The deployed margin gate is essentially optimal among gates.

- **The win is structural — the Adaptive-Compute Cascade (ACC).** A confidence-gated 3-tier cascade
  over *compute configurations*: **7B-nothink@cap320 → 32B-NOTHINK@cap320 → 32B-think@fullres**. The
  big model's *fast* no-think mode (≈0.34s, vs ≈11s for think) is inserted as an intermediate tier,
  gated by its own logprob margin, so the slow reasoning pass fires only on the ~18% reasoning
  residual. Honest held-out eval with **real measured batch-1 latencies**, at accuracy parity with
  always-32B-think: **latency 11.34s→2.27s (−80%) on ALL-6, 8.88s→0.44s (−95%) on ALL-5; FLOPs 100→52% /
  100→25%; energy ~5×; and *cleaner* on the never-worse-than-7B guardrail.** Mechanism: thinking *overthinks*
  perception VQA (32B-no-think ≥ 32B-think on the competent benchmarks). Scope: the 4 competent
  benchmarks (MMMU/MedXpert excluded — both near chance). Reproduce: `python3 src/cascade_methods/acc.py`.
  Method spec + adversarial novelty check (incremental-but-defensible systems contribution; closest
  prior art CAR, arXiv 2505.15154): `results/cascade_methods/METHOD_ACC.md`.
