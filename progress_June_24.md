# Session progress log — June 24, 2026 (visual-stability rescue, ACC-v3/v4, the open-ended ceiling-break)

> Continues from `progress_June_20-22.md`. This log records, in detail, everything done on **2026-06-24**:
> a fresh training-free-gate hunt that produced the Visual-Stability Rescue (real for the 2-tier deployed
> gate, but it does **not** improve ACC-v2); the validated ACC-v3 (confidence-tightened think gate) and the
> ACC-v4 resolution/reasoning dissociation; the gate-data-size answer; and the session's headline result —
> moving from MCQ to **open-ended** medical VQA, where the routing ceiling breaks (AUROC ~0.6 → ~0.87), with
> an exhaustive gate hunt, an LLM-judge robustness pass, a 3rd dataset, the paper §5.7, related-work, and the
> regenerated master charts. **No fabricated numbers** — every figure is read from real checkpoint output;
> reproduction pointers are inline. All work pushed to `main`.

---

## 2026-06-24 (phase 1) — Visual-Stability Rescue: real signal, but it does not improve ACC-v2

**Goal (user):** find a novel training-free model-cascade method, try all variations, don't stop.

- **The signal.** Among the cheap 7B-nt's would-escalate (low-margin) queries, *keep on the cheap model if
  its answer is invariant across an image-resolution ladder* (cap80/160/320/640) — it has "settled". Errors
  are **capacity-bound, not resolution-bound**: `resolution_tta_and_tier.py` shows 7B@fullres fixes only
  **12.6%** of cap320 errors vs the 32B's **45.2%**.
- **On the frozen deployed 2-tier gate (competent-4):** 32B-call **60%→20%**, prefill-inclusive backbone
  **69.5%→43.3%** at always-strong parity (0.6448), guardrail-clean; iso-accuracy frontier **44.1%→40.9%**
  backbone, ~half the 32B calls. Held-out eval-CV + paired bootstrap confirm.
  `[REPRO: resolution_consistency.py, stability_rescue_{validate,cost,bootstrap}.py;
  results/cascade_methods/NOVEL_METHOD_VISUAL_STABILITY.md]`
- **Negatives** (all offline): bidirectional "fragility-escalate" up-arm adds +0.0000; self-verify rescue and
  doubly-robust violate the SLAKE guardrail; multi-res vote marginal (+0.0034).
- **Integrated into ACC-v2 → HONEST NEGATIVE.** `acc_rescue_allfam.py` (3 Qwen families):
  rescue@tier-0 caps accuracy (wrong tier — tier-0 gates the *cheap* big-nt leg); rescue@think *appears* to
  win but the control (`control_think_signals.py`) shows resolution-stability is the **worst** think-skip
  signal — random 0.72 s, big-nt-confidence 0.60 s, even inverse-stability 0.93 s all beat it (1.07 s; ACC-v2
  1.70 s). The apparent win is generic think→big-nt rebalancing. **ACC-v2's big-no-think tier already subsumes
  the rescue's benefit.** `[REPRO: acc_rescue_allfam.py, think_rescue_mechanism.py, control_think_signals.py;
  RESCUE_INTO_ACCV2.md; figs paper/figs/rescue/]`
- Independent novelty check: route-**down** resolution rescue is novel-in-abstract (vs VisionThink, which
  routes **up**), but moot since it doesn't help ACC-v2.

## 2026-06-24 (phase 2) — ACC-v3 (validated), ACC-v4 (novel-but-marginal), gate-data-size

- **ACC-v3 — confidence-tightened think gate (the real improvement).** Fire think only when the two no-think
  models disagree **and** the big no-think model is itself unsure (m₁<τ₁). Honest 50/50 calib/test ×20 seeds:
  - **ALL-6 MedVLThinker beats both single-signal gates:** 2.61 s (v1) → **2.13 s**, 1371→1098 J, +acc,
    reaches always-32B-think parity **20/20 seeds** (ACC-v2 19/20).
  - **Perception pools:** confidence drives think→**0%**, halving compute (competent 1.55 s→0.23 s, 790→55 J).
  - Cross-family: Lingshu ALL-6 FLOPs 77.8%→48.6%; QoQ degenerate (no-op). Novelty: **incremental**
    (confidence=CAR + agreement=ABC, AND-combo). `[REPRO: acc_v3_confgate.py; figs paper/figs/rescue/table_*]`
- **Gate-data-size (user question): more PMC-VQA-train would NOT change the result.** Subsampling the 3000-row
  calibration, eval behaviour is **flat 2000→3000** (acc 0.6503→0.6502, threshold σ→0). The limit is the
  *perception-only distribution* of PMC-train, not the sample size. `[REPRO: gate_data_size.py]`
- **ACC-v4 — resolution/reasoning dissociation.** Medical **reasoning is resolution-insensitive** (think@cap320
  ≥ think@fullres on the benchmarks that reach the think tier: MMMU **0.688→0.712**, MedXpert ≈ same) while
  **perception is resolution-sensitive** (SLAKE 0.764→0.721, but no-think serves it). Running the think tier at
  cap320 saves ~28% think prefill — but the cascade-level effect is marginal (**ACC-v4 ≈ ACC-v3**: think fires
  ≤14% and is decode-bound). Novelty: the dissociation is incremental (Matryoshka M3/MQT, token-count axis);
  the medical-cascade composition is novel-but-marginal. Adaptive think-length is **dead** (the 32B commits its
  answer at the *end* of the trace — answer-marker median 0.99 over 8 220 traces). `[REPRO: acc_v4_lowres_think.py]`

## 2026-06-24 (phase 3) — new setting: OPEN-ENDED medical VQA (the headline)

**User:** change the setting for real novelty; GPU is free. Rationale: MCQ single-letter answers make every
routing signal degenerate (~0.6 ceiling); open-ended free-text revives them.

- **Inference** (`run_openvqa.py`, generative; 7B confidence temp-0, 7B self-consistency temp-0.7 K=8, 32B
  no-think & think): SLAKE-open (645) + VQA-RAD-open (200). PathVQA-open dropped initially (exact-match 0.058).
- **No within-family gap.** MedVLThinker **3B 0.457 ≈ 7B 0.419 ≈ 32B-nt 0.498 ≈ 32B-think 0.453** (token-F1
  confirms; think hurts). These MCQ-RL models generalize uniformly poorly to free-text → nothing to cascade to.
  `[REPRO: run_openvqa_3b.sh, open_cascade_analyze.py --cheap3b]`
- **Cross-family gap → the ceiling BREAKS.** With **Lingshu-32B** (strong open-ended medical model, 0.775) as
  the strong leg: MedVLThinker-7B 0.407 → 0.775 (token-F1 0.376→0.789), recoverable **0.385**. Routing AUROC
  (confidence, cheap-wrong): **MedVLThinker-7B→Lingshu-32B 0.735; Lingshu-7B→Lingshu-32B 0.866 / recover 0.804**
  — far above the ~0.6 MCQ ceiling. Cause = **4-option discreteness, not answer length** (open answers median
  **1–2 tokens**, yet AUROC ~0.87). `[REPRO: run_openvqa_{lingshu,lingshu7b}.sh, open_cascade_analyze.py --lingshu]`
- **Self-consistency is a conditional calibration rescue, not a better gate.** Cross-family (miscalibrated
  MedVLThinker-7B): SC > confidence (recoverability +0.043, bootstrap 95% CI [0.016, 0.069]); same-family
  (calibrated Lingshu-7B): **confidence wins** (0.866 vs 0.845). (This corrected an earlier overclaim.)
- **Exhaustive gate hunt — confidence is near-optimal.** On the calibrated cascade (bar = confidence 0.866/0.804):
  exact-SC 0.845, semantic-SC 0.806 (semantic clustering *hurts* short answers), semantic-entropy 0.807,
  mean-F1 0.844, self-verify **P(True) 0.755** (weakest), and the honest 20-seed-CV **fusion of all six ties at
  0.866** (+0.000). No signal beats confidence. `[REPRO: gate_search_open.py, run_openvqa_verify.py]`

## 2026-06-24 (phase 4) — hardening §5.7: LLM-judge, a 3rd dataset, related work

- **LLM-judge robustness** (`run_judge.py`: neutral MedVLThinker-32B text grader, *not* in the Lingshu cascade).
  Ceiling-break is **not a scoring artifact**: calibrated confidence AUROC under the judge **0.860 / 0.784** ≈
  exact-match 0.866 / 0.804; confidence still > self-consistency (0.835). Under the judge the cross-family SC
  margin *shrinks* (the judge credits MedVLThinker-7B's verbose answers, acc 0.41→0.52, partly fixing the
  miscalibration). `[REPRO: open_cascade_analyze.py --judge]`
- **3rd dataset — PathVQA-open** (n=1500). Exact-match collapses on its long descriptive answers (0.058), so it
  is **judge-only**: Lingshu-7B 0.343, Lingshu-32B 0.376; routing cheap-wrong AUROC **0.797** (still ≫ 0.6).
  **3-dataset pooled (n=2345, all judge-scored): confidence AUROC 0.846**, confidence ≥ self-consistency.
  Ceiling-break robust across SLAKE+VQA-RAD+PathVQA and three scorers (exact-match, token-F1, LLM-judge).
  `[REPRO: run_openvqa_pathvqa.sh, run_judge_pathvqa.sh, open_cascade_analyze.py --pathvqa]`
- **Independent novelty check:** closest prior art is the *text-LLM* semantic-agreement cascade (arXiv
  2509.21837, EMNLP'25; cross-model-ensemble greedy agreement, no medical/vision, no MCQ-degeneracy claim);
  ABC (2407.02348), Jitkrittum (NeurIPS'23). Genuinely unoccupied cell = **medical + VLM + open-ended cascade**
  + the *ceiling-is-discreteness* diagnostic → publishable applied contribution (CVGIP), **not** a new gate.

## 2026-06-24 (phase 5) — paper + master charts

- **Paper** (`paper/cvgip2026_draft.md`): new **§5.7** "From MCQ to open-ended: the routing ceiling is a
  benchmark artifact" (3 parts, 2 tables); §5.1.2 already carried ACC-v3/v4; abstract now reports **three
  findings** (added the MCQ-artifact result); §5.2 gets a forward-pointer; Conclusion nuanced ("why routing is
  hard here — and where it is not"); new Related-Work paragraph; Reproducibility index updated.
- **Master charts** (`make_master_charts.py`): regenerated all 5 MCQ-ACC charts + a **new Chart 6**
  (`paper/figs/master/fig_openended_ceiling.png`) — left: open-ended routing AUROC per dataset vs the 0.6 MCQ
  ceiling; right: the gate hunt (confidence ties fusion, beats all other signals). New JSONs
  `open_cascade_calib_judge.json` + `open_gate_search.json`; `MASTER_TABLES.md` got an open-ended section.

## 2026-06-24 (phase 6) — completeness pass + a new-method loop (user: "make the research as complete as
possible … then start another research loop to create another novel method that beats SOTA; GPU/network
free; non-core findings in a separate doc")

- **Four background literature/scouting agents** (cascade-domain gaps, learned baselines, datasets+models,
  new-method scouting). All non-core survey material is logged **separately** in
  `results/cascade_methods/AUXILIARY_RESEARCH.md` (gitignored), per the directive.
- **Master tables — LLM-judge column** (`make_master_charts.py`): open-ended model accuracy with exact-match
  vs LLM-judge columns (incl. Kvasir).
- **Ablations.** `open_ablations.py` (K-ablation: self-consistency 0.73→0.83 over K, never beats confidence
  0.846; routing efficiency 70/59/30% SLAKE/VQA-RAD/PathVQA). `deferral_curve.py` (random↔confidence↔oracle
  for MCQ+open) → **the key honest split: DETECTION** (cheap-wrong AUROC 0.66→0.85, the ceiling-break) is
  distinct from **cascade gain** (oracle−cheap headroom only +0.02 MCQ / +0.06 open) — the open-ended win is
  in error *detection*, not compute-saving. Bootstrap CI on the ceiling-break: 0.846, 95% CI [0.830, 0.862].
- **Theory backing** (`uniform_improver_diag.py`): the strong model breaks cheap-correct answers only 22%
  (MCQ) / 14% (open) → it is a near-**uniform improver** → per **Jitkrittum (2307.02764)** confidence is
  near-Bayes-optimal. Turns "the gate is saturated" into a *theory-predicted* result (§5.2).
- **4th modality — Kvasir-VQA-x1** (GI endoscopy, `prep_kvasir.py`, n=1200, LLM-judged): Lingshu-7B 0.302 ≈
  Lingshu-32B 0.301 (equal, zero cascade headroom) yet confidence cheap-wrong AUROC **0.749** (CI [0.72,0.78])
  — the ceiling-break holds on a fresh modality and cleanly shows detection ≠ cascade-gain. *(Caught/fixed a
  racy partial-judge-file read that had shown 0.84 / "32B worse" — always judge both legs fully first.)*
- **Reviewer-threat fixes (paper):** discrimination-vs-calibration + cite/distinguish CritiCal [2510.24505];
  over-thinking demoted to a *cited* mechanism (MME-CoT etc.); batch-1 single-stream scope + batched-serving
  caveat; related-work adds learned pre-generation routers (Hybrid-LLM, RouteLLM, Co-LLM, Cascade-Routing).
  Learned baseline *ran*: Jitkrittum Diff-01 (ties/marginal); cross-model agreement weak (0.586).
- **NEW METHOD — §5.8 "Knowing when to abstain"** (`selective_abstain.py`, `triage_3way.py`,
  `SELECTIVE_ABSTENTION.md`). Training-free **safe abstention / clinician-referral** for open-ended medical
  VLMs, engaging the verified 2026 SOTA **"Uncertainty Is Not a Safety Net for Clinical VQA"** [2606.16583]
  (MCQ-only, negative, ~0.72 ceiling). *Money plot:* self-error-detection AUROC lifts MCQ→open (MedVLThinker
  0.66→0.74, Lingshu 0.74→0.82). *Deployable:* SLAKE-open auto-answers **54% at ≤5% error** (det AUROC 0.89).
  *Honest:* deployability tracks competence (collapses on weak datasets — confirms the SOTA's nuance); the
  3-way answer/escalate/abstain triage is only marginally better (recoverability-bounded).

---

## Bottom line (2026-06-24)

1. **ACC-v3** — a real, validated, *incremental* improvement to the cascade method (in the paper).
2. **Open-ended breaks the MCQ routing ceiling** (DETECTION AUROC ~0.6 → ~0.87, robust across 4 datasets +
   3 scorers, theory-backed via Jitkrittum) — the headline, reframing the "gate is saturated" negative as a
   benchmark artifact. The win is in error *detection*, not cascade compute-saving (recoverability-bounded).
3. **No novel *gate*/signal beats confidence** — exhaustive across MCQ and open-ended (consistency, semantic
   entropy, self-verification, cross-model agreement, learned deferral, fusion all tie/lose). Every genuine
   contribution is in the **setting/application**, never a new signal.
4. **NEW (§5.8): safe abstention for open-ended medical VLMs** — a deployable selective-prediction system
   that engages and partially refutes a 2026 SOTA; honest about competence-dependence.

Docs: `results/cascade_methods/{NOVEL_METHOD_VISUAL_STABILITY, RESCUE_INTO_ACCV2, ACCV3_V4_AND_NOVELTY,
OPENENDED_CASCADE, DETAILED_TABLES, SELECTIVE_ABSTENTION, AUXILIARY_RESEARCH}.md`. Figures:
`paper/figs/{rescue,open,master}/`. All pushed to `main`.
