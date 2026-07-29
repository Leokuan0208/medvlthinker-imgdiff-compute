# Progress — June 27–28, 2026

> **Reconstructed 2026-07-02** from the commit log, `INCONSISTENCIES.md`, `docs/archive_mcq/RESEARCH_REFOCUS_2026-06-28.md`,
> and `docs/current/VERIFIED_FACTS.md` (this window predates the daily-log habit; filling the gap after the fact).
> All numbers are quoted from those sources — no new figures.

## Theme: consolidate, verify, and refocus the project onto one method + peer baselines

After the June 25–26 trained-verifier result, these two days were a **consolidation pass**, not new experiments.

### June 27 — consistency audit (`INCONSISTENCIES.md`)
Ran a 3-way audit (paper ↔ cross-docs ↔ ground-truth-from-`ckpts`) because the ACC efficiency headline
existed in **three conflicting forms** across the paper/README/CLAUDE/METHOD_ACC. Resolution: the canonical
numbers come from `results/cascade_methods/artifacts/master_data.csv` + `GROUND_TRUTH_NUMBERS.md`; forward-facing
docs were corrected to those, and the dated `progress_*.md` diaries were left as-written (history preserved,
with a pointer to canonical). Superseded ACC efficiency framings (e.g. "20.0s→5.7s / FLOPs 81→55%", and the
paper §5.1.1 "26.6s / 4.86s") were retired in favour of the `master_data.csv` values.

### June 28 — research refocus (`RESEARCH_REFOCUS_2026-06-28.md`) + verified-facts freeze (`VERIFIED_FACTS.md`)
- **The focused thesis:** *test-time compute for medical VLMs — what actually helps.* Headline method = a small
  **trained LoRA outcome verifier** for best-of-N selection on open-ended medical VQA (+ grounding). Efficiency
  companion = **ACC**. The verifier is to be benchmarked against prestigious peers and beat them.
- **Baseline landscape** (open-ended, n=3545, LLM-judge; cheap=Lingshu-7B, strong=Lingshu-32B): pooled greedy
  0.377, self-consistency 0.394, 32B single-pass 0.444, oracle@8 0.580. The **trained verifier** (held-out
  n=1064) reached **pooled 0.501** — above 32B single-pass, below the oracle ceiling (the selection headroom).
- **`VERIFIED_FACTS.md`** froze the per-benchmark MedVLThinker table (from `master_data.csv`) so later work
  builds only on sourced numbers: PMC 0.543/0.551/0.556, SLAKE 0.762/0.849/0.764 (no-think > think),
  VQA-RAD 0.761/0.853/0.776, PathVQA 0.641/0.661/0.672, MMMU 0.547/0.624/0.688 (reasoning: think > no-think),
  MedXpert near-chance for 7B (excluded from the headline).

**Standing conclusion end of June 28:** two-positive arc locked in — **ACC** (efficiency) + **trained verifier**
(accuracy), unified by the luck floor. The next window (June 29–30) executes the open-text unified-method search.
