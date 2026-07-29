# Progress — July 1–2, 2026

> **Written 2026-07-02** from the commit log, `docs/current/MASTER_SUMMARY_2026-07.md`, the eval-protocol memory,
> and this session's work. July 2 is partly in-flight (the full 3-family matrix is running as this is written).

## July 1 — the faithful Lingshu MCQ baseline + the 2-tier cascade, and the exhaustive verdicts

### Faithful baseline (the correct protocol)
Our internal NGC harness is **not** faithful to Lingshu's paper; **MedEvalKit** (repo `./MedEvalKit`, isolated
`/data/dan/medeval_venv`, vllm 0.9.0.1) **is**. Recipe locked: medeval_venv + Qwen2_5_VL wrapper + Lingshu weights
+ `datasets_path=hf` + `use_vllm` + `TORCHDYNAMO_DISABLE` + `use_llm_judge=False` (MCQ). Result: Lingshu-32B
**MMMU 0.633 = paper 62.3 (exact)**; MedXpert/PMC/SLAKE-closed/VQA_RAD-closed reproduce paper. Both 7B and 32B
validated on SLAKE/PMC/MedXpert.

### The 2-tier cascade (headline MCQ efficiency)
Faithful 2-tier cascade (7B→32B, **margin gate**) matches always-32B accuracy at large FLOPs savings:
**PMC-VQA −69% FLOPs / −33% latency** (Lingshu, 33k), **SLAKE −56% / −22%**; generalizes cross-family
(MedVLThinker PMC −49%, VQA-RAD −41%). Win magnitude ≈ the (32B−7B) gap; no win where the 7B is weak
(SLAKE-MVT, MedXpert floor). Gate-signal variation: **margin > conf > cum_logprob**. APGR + CPT (RouteLLM-style)
metrics computed.

### Verdicts settled (gate / candidate / verifier)
- **Gate:** controlled swap (verifier+selection fixed) → verifier-confidence has the highest ADC + AUROC in both
  regimes; SOTA post-hoc **Jitkrittum Diff-Prob** and all simple signals are worse; trained gates only tie.
  **The confidence gate cannot be improved by substitution.** MCQ is saturated (7B≈32B, recoverability wall
  ~0.578) → "beat-the-strong" is an **open-text** phenomenon, not MCQ.
- **Verifier ceiling is intrinsic:** a 32B zero-shot verifier is *worse* than the trained 7B (0.355 vs 0.403) →
  task-training ≫ size; the cheap 7B verifier is near-optimal. Selection efficiency ~80% across answer lengths.
- **Candidate quality is the real limit:** a cross-model candidate pool raises oracle@N by **+0.11–0.15**.
- **MMMU-7B anomaly RESOLVED:** Lingshu-7B-specific (MedVLThinker-7B is normal 0.533 on the same eval), excluded
  from claims. The 32B **did** match paper (63.3 vs 62.3).

### `MASTER_SUMMARY_2026-07.md`
Consolidated the whole program: open-text (beats 32B, 3 families) + faithful MCQ cascade (matches 32B at
−17…−69% FLOPs). Honesty check: held-out-τ efficiency holds (PMC −57% Lingshu / −49% MVT FLOPs saved with a fair
threshold) → the deployable efficiency claim is honest.

> **⚠️ A July-1 claim that July-2 overturned:** a commit asserted *"Lingshu has no promptable think mode
> (gen_toks~3 even with CoT prompt) → 2-tier only."* **This was wrong** — see the July-2 correction below.

## July 2 — Claude-as-judge, project cleanup, the InternVL3 fix, and the reasoning correction

- **Claude-as-judge (no API key):** validated a Sonnet-5 subagent judge via the Max plan (100% exact-match anchor;
  the zero-overlap-correct cases are legitimate synonyms / bilingual Chinese matches). Unblocks the open-ended
  halves → VQA-RAD-32B 74.1% (paper 76.5), SLAKE-32B 85.0% (paper 89.2). Opus not needed for grading.
- **Project maintenance:** reorganized `results/cascade_methods/` into `docs/{current,archive_mcq}/` + `artifacts/`
  (updated 37 scripts' paths + all doc cross-links; rewrote the README index + STRUCTURE.md). Decluttered the repo
  root: daily logs → `progress/`, meeting decks → `meetings/` (dated).
- **InternVL3-38B fixed:** the "InternVL not supported" crash was MedEvalKit's `init_llm` masking the real error —
  `max_seq_len 65536 > KV cache 24288` at tp=2. Added a `MAX_MODEL_LEN` lever to the InternVL wrapper (=16384 for
  the 38B); smoke test passed. Also wired **OmniMedVQA** (the 7th benchmark in Lingshu's paper suite; ~88,996 Qs,
  of which 56,697 are RadImageNet) — unzipped, path-wired, gate-signal capture patched.
- **CORRECTION — Lingshu *does* reason when prompted.** An empirical probe (`lingshu_reason_probe.py`) showed
  Lingshu-7B goes from `gen_toks=3` (direct) to **174** with "reason step by step" and **267** emitting real
  `<think></think>` tags — on **both MCQ and open-text**. The earlier "no think mode" was a **weak-prompt
  artifact**: MedEvalKit's `--reasoning True` only appends *"put the letter in `\boxed{}`"*, which Lingshu answers
  directly. The paper confirms reasoning is trained-in (CoT + RL), not a toggle. Fixed the reasoning prompt in
  both `question_formats.py` (VQA-types + MedXpert) and `MMMU/data_utils.py` (MMMU has its own prompt path). So a
  **3-tier think cascade is viable across the Qwen-based families**, not MedVLThinker-only; whether reasoning
  *helps accuracy* is being measured per reasoning-benchmark (MMMU-Med, MedXpert-MM).
- **Full 3-family × 7-benchmark matrix launched** (`runners/run_full_matrix_medeval.sh`, resumable): completes
  InternVL3 on the suite, adds valid reasoning passes, and runs the full OmniMedVQA (~89k) marathon — cascade vs
  always-big across all three families. Multi-day; results reported per-family as they land.

**Eval-validity note:** all 7 benchmarks persist the gate metrics (margin/conf/gen_toks/latency); closed/MCQ =
exact-match, open halves = the validated Claude judge, OmniMedVQA = 4-option MCQ exact-match.
