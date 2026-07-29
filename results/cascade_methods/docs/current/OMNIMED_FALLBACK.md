# OmniMedVQA — strong-leg fallback decision (2026-07-06)

## What happened
- **Cheap (7B/8B) OmniMed legs: RAN successfully and reproduce the paper.**
  - Lingshu-7B **0.827** (paper 0.829) · MedVLThinker-7B **0.625** · InternVL3-8B **0.785**.
  - This validates that our OmniMedVQA eval pipeline is faithful (Lingshu-7B matches the paper to 0.2 pt).
- **Strong (32B/38B) OmniMed legs: NOT run — persistent tp=2 infrastructure hang.**
  - At tensor-parallel=2 (required: a 32B does not fit tp=1 on 80 GB — 64 GB weights + multimodal activation OOMs), the 89k-image OmniMed run hits an intermittent **NCCL collective hang**: every chunk attempt stalls ~36 min and is auto-killed by the watchdog, then re-hangs. Confirmed across many attempts over ~2 days.
  - Tried and ruled out: chunked tp=2 + retry, `TORCH_NCCL_ENABLE_MONITORING=0`, `EVAL_BATCH_SIZE=256` (fixed the cgroup OOM but not the hang), 3 h/chunk timeout + auto-recovery. The hang recurs deterministically.

## Decision (why it's safe)
- **OmniMedVQA is a keep-cheap benchmark**: the cheap and strong models are essentially tied (paper Lingshu-7B 82.9 vs 32B 83.4 — a 0.5 pt gap). Our cheap 0.827 already matches. A cascade **keeps-cheap on OmniMed** (near-zero escalation), so the strong-leg number changes **no cascade conclusion**.
- Therefore: use the **paper's 32B OmniMed number (Lingshu 0.834)** as the always-strong reference baseline; the cascade on OmniMed = the cheap number at ~0% escalation. **No fabricated metrics.json is written to the results dir** — the strong OmniMed cell is simply reported as paper-reference + infra-limited, and our faithful cheap reproduction backs the fidelity claim.
- This unblocks the GPUs for the higher-value research (the UGV MCQ-as-generation experiment and the correlated-Pandora / diverse-generation follow-ups).

## Net status of the 3-family × 7-benchmark reproduction
- 6/7 benchmarks (MMMU, VQA-RAD, SLAKE, PathVQA, PMC-VQA, MedXpert): fully run cheap + strong, all 3 families (+ think tier). Faithful vs paper (see VERIFIED_FACTS / the eval-validity audit).
- OmniMedVQA (7th): cheap run + faithful; strong = paper-reference (tp=2 infra hang), keep-cheap so immaterial to the cascade.
