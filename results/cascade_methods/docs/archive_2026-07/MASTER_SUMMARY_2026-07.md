> # ⚠️ SUPERSEDED (annotated 2026-07-29)
>
> This was the top-level consolidation **as of 2026-07-02** — the 2-tier MCQ-efficiency era. It is kept as
> the record of that state. It explicitly says the full 3-family x 7-benchmark matrix is "in progress"; that
> **finished 2026-07-03/05** (6 of 7 fully faithful + one cheap-faithful; OmniMed strong leg infra-blocked,
> see `OMNIMED_FALLBACK.md`). Everything after it — the format-aware router, the honest-baseline re-grounding,
> the MMMU exclusion, and the measured headline — is **not** in this file.
>
> **Current:** [`PROJECT_RETROSPECTIVE_2026-07-29.md`](PROJECT_RETROSPECTIVE_2026-07-29.md) (the definitive
> account) and `METHOD_FINAL_2026-07.md` (the method spec).
>
> *One thing this file got right and a later doc got wrong:* it already recorded that MMMU-7B shows
> Lingshu-7B-specific inflation and should be **excluded**. Three docs written 2026-07-07/08 banked it as a
> "+0.140 beat-the-32B win" anyway; the exclusion was re-taken on 2026-07-08 and is now final.

# Master Summary — Test-Time-Compute Cascade + Verifier for Medical VQA (2026-07)

Consolidated across the research program. Full running log: `UNIFIED_METHOD_EXPERIMENTS.md`. Open-text table:
`OPENTEXT_MASTER_TABLE.md`. All numbers from real checkpoints (no fabrication).

## The two settings

**A. Open-text medical VQA (free-text, LLM-judge graded).** Method = cheap VLM samples N answers → trained outcome
verifier picks best (best-of-N) → verifier-confidence gate escalates low-confidence to the strong VLM.

**B. MCQ medical VQA (Lingshu's published benchmarks, faithful MedEvalKit protocol).** Method = 2-tier cascade
(cheap 7B → strong 32B) with a margin gate. (CORRECTION 2026-07-02: Lingshu DOES reason when properly prompted —
gen_toks 3→174 with "reason step by step", →267 emitting <think> — so a 3-tier think cascade is viable across the
Qwen families; whether reasoning *helps accuracy* is being measured per reasoning-benchmark. The earlier "no think
mode" was a weak-prompt artifact; see progress/progress_July_01-02.md.)

## Headline results

**Open-text (A): the method BEATS the strong model on accuracy**, 3 families (Lingshu, MedVLThinker, InternVL3),
every dataset + held-out OOD (RadImageNet). Cross-architecture verifier transfer works.

**MCQ (B): the method MATCHES the strong model at large compute savings** (efficiency, faithful Lingshu eval):
| benchmark | Lingshu 2-tier vs 32B | MedVLThinker 2-tier vs 32B |
| PMC-VQA (33k) | match @ **−69% FLOPs / −33% latency** (9% esc) | match @ **−49% FLOPs** (29% esc) |
| SLAKE-closed | match @ −56% FLOPs / −22% latency (22%) | no win (7B weak) |
| VQA-RAD-closed | match @ −17% FLOPs (61% esc) | match @ −41% FLOPs (37%) |
| MedXpert-MM | no win (7B near-floor) | no win |
Win magnitude ~ (32B − 7B accuracy gap): small gap → big win. Efficiency generalizes cross-family where 7B competitive.

## Verdicts (settled)

1. **Gate: verifier-confidence (open-text) / margin (MCQ) is the best gate.** No trained gate beats it — CASP/CCPS,
   learned MLP/GBM, and the SOTA post-hoc recoverability gate (Jitkrittum Diff-Prob) all tie or lose, in a controlled
   swap and on the efficiency leg. Binding limit = the recoverability WALL (strong fixes only 6–26% of cheap errors,
   unpredictably). Trained gates beat WEAK gates (agreement/self-consistency) but not the confidence gate.
2. **Verifier is near its selection ceiling** (open-text): training tricks (data/epochs/ranking → per-answer AUROC
   0.90→0.93 but selection flat) and a bigger (zero-shot 32B) verifier don't raise selection. Ceiling is intrinsic
   grounding difficulty. **Binding limit is candidate quality** — more/cross-model candidates raise oracle +0.11–0.15.
3. **Faithful baseline validated** (MedEvalKit); the internal NGC harness is NOT faithful to Lingshu's paper.
4. **Judge trusted** — independent 2nd-judge agreement κ 0.85–0.96 + 100% exact-match anchors.
5. **Cost/latency/energy measured** (rare in the cascade literature — our differentiator); also report standard
   proxies (%-strong-calls, deferral curve, APGR/CPT) for comparability to FrugalGPT/RouteLLM.

## Best deployable configs
- **Open-text:** verifier best-of-N selection (N=2 = FLOP break-even vs the 32B) + verifier-confidence gate.
- **MCQ (Lingshu eval):** 2-tier 7B→32B + margin gate, per-benchmark iso-32B operating point; cap the cheap leg
  (cap320) where the domain tolerates it (PMC), full-res for radiology.

## Caveats / open
- MMMU-7B: Lingshu-7B-specific inflation (MVT-7B normal) → excluded from MCQ cascade claims.
- 3-tier think: viable across the Qwen families (Lingshu reasons when prompted — corrected 2026-07-02); whether it
  raises accuracy on the reasoning benchmarks (MMMU/MedXpert) is under test in the running matrix.
- RESOLVED since first draft: PATH_VQA now wired + run locally; open-ended judging done via Claude-as-judge (Max
  plan, no API key) instead of GPT-4.1. InternVL3-38B unblocked (KV-cache max_model_len fix).
- In progress (2026-07-02): full 3-family × 7-benchmark matrix (adds OmniMedVQA, the 7th Lingshu-suite benchmark;
  ~89k, multi-day) — cascade vs always-big across all three families.
