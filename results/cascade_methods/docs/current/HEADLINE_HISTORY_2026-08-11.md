# Headline history & superseded numbers — extracted from CLAUDE.md, 2026-08-11

> **Why this file exists.** CLAUDE.md is auto-loaded into every session's context. Its §0 had grown to
> 24 KB — 46% of the file — almost entirely accumulated correction banners recording numbers that are no
> longer canonical. That history is worth keeping (it is how this project avoids re-making retracted
> claims) but it does not need to be in every context window.
>
> **This file is the verbatim extraction.** Nothing was edited or dropped in the move. CLAUDE.md §0 now
> carries only the current canonical state plus a pointer here.
>
> **Canonical numbers live in** `PROJECT_RETROSPECTIVE_2026-07-29.md` (§4 results, §10 corrections log)
> and, for anything after 2026-08-05, `artifacts/cascade_selector_rerun_2026-08-05.json`.

---

## 0. Current status (2026-07-30) — read this first

> # ➜ THE ENTRY DOCUMENT IS THE RETROSPECTIVE
> **`results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`** (1,972 lines) is the
> definitive account of the whole project, 2026-06-17 → 2026-07-29: the arc and every pivot (§2), the
> method as it stands (§3), **the canonical results with CIs (§4)**, what generalizes (§5), a catalog of
> ~90 negative results (§6), **the honest holes (§7)**, what to do next (§8), practical notes + a
> codename glossary (§9), and **the corrections log (§10)**. Read it before anything else, and before
> quoting any number from any other document in this repo.
>
> Second-best single artifact: **`meetings/progress_report_professor_2026-07-27.html`** — the whole
> project in 13 source-cited sections.

### What the project is now

A **format-aware adaptive cascade** between **Lingshu-7B** and **Lingshu-32B**, evaluated on the
**MedEvalKit** harness (the harness that faithfully reproduces Lingshu's published numbers). It detects
**from the prompt text alone** whether a question is multiple-choice or open-ended and runs a different
policy for each, with one knob at two settings. Spec: `docs/current/METHOD_FINAL_2026-07.md` (mechanism
correct, numbers pre-date the 2026-07-08 measurement). Code: `src/cascade_methods/method_final.py`.
Prose: `paper/adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}` — **the deliverable**.

- **Multiple-choice arm:** 7B-direct → confidence-**margin** gate → 32B-**direct**.
- **Open-text arm:** 7B best-of-N (adaptive N, Weitzman optimal-search) → **trained LoRA outcome
  verifier** picks the best → escalate to 32B-direct on low verifier confidence.
- The 32B's *reasoning* mode is the **baseline**, never a deployed tier on Lingshu.

### The canonical headline (Variant B = MMMU excluded, 5 benchmarks / 8 cells / **n = 42,224**)

> **⚠️ RE-BASED 2026-07-30 — the metric is MACRO now, and the compute claim REVERSES.**
> The primary average is **equal weight per reporting cell (8 cells, 1/8 each)**, *not* sample-weighted,
> because sample-weighting let PMC-VQA (**79.2%** of items) speak for the method. The 5-benchmark macro
> is a **secondary robustness check only**. Reweighting: PMC **79.2% → 12.5%**; open-text arm
> **5.6% of items → 37.5% of weight**; closed/MCQ **94.4% → 62.5%**.
> Source (verbatim): **`artifacts/macro_average_headline_2026-07-30.json`**
> (`src/cascade_methods/macro_average_headline.py`). No measured value changed — only which weighting it
> belongs to.

**PRIMARY — macro over the 8 cells.** Baselines: always-32B-with-reasoning **0.5974**;
always-32B-direct **0.6567**; oracle-mode-32B **0.6573**; always-7B **0.5971**.

| setting | accuracy | compute (× one 32B pass) | lat. par./seq. | energy | Δ vs 32B-reasoning | Δ vs 32B-direct |
|---|---|---|---|---|---|---|
| **compute-lean** | 0.6600 | **1.196×** | 650 / 1,292 ms | 188 J | **+0.0626 [+0.0514, +0.0734]** SIG | +0.0033 [−0.0054, +0.0121] n.s. |
| **accuracy-max** (certified veto + learn-to-defer) | **0.6694** | **1.410×** | 691 / 1,334 ms | 206 J | **+0.0720 [+0.0614, +0.0824]** SIG ← *canonical* | **+0.0128 [+0.0056, +0.0200]** SIG |
| accuracy-max⁺ (fusion variant) | 0.6661 | **1.435×** | 665 / 1,350 ms | 209 J | **+0.0686 [+0.0582, +0.0790]** SIG | **+0.0094 [+0.0013, +0.0176]** SIG |

**⛔ "Pareto-dominates every fixed way of using the 32B" is RETIRED** (retrospective §10.1 C26). Under
equal weight **no operating point is compute-cheaper than always-32B-direct** — 1.196× / 1.410× /
1.435× (they were 0.492× / 0.932× / 1.250× sample-weighted). "Pareto-**optimal**" survives (the points
are non-dominated because they are more *accurate*); "Pareto-**dominates**" does not. Restrict
"dominates" to the always-32B-**with-reasoning** baseline — and even there, **not on FLOP-eq**.

**What survives, and what fails, at equal weight:**
- accuracy-max beats always-32B-direct **+0.0128 [+0.0056, +0.0200]** — but at **1.41×** its compute;
- compute-lean **neither** matches it on the multiple-choice half (**−0.0070 [−0.0126, −0.0017], a
  SIGNIFICANT LOSS**) **nor** stays cheap (**1.20×**);
- the one baseline the method still clearly beats is a **32B actually made to reason**: **+0.0720
  [+0.0614, +0.0824]** accuracy, **−89% latency, −87% energy** — though **not** fewer FLOP-eq (1.41× as
  charged, 1.13× honestly re-costed).

> **⚠️ NUANCE, required wherever cost is claimed.** Macro-averaging **cost** answers a *different*
> question from sample-weighted cost. Cost is **additive per query**, so on traffic resembling this suite
> the **~0.49×** saving is what you would actually pay; the macro number instead tests whether the saving
> **generalises across task types** — and it does not, it is **concentrated on the low-escalation cells**.
> **Report accuracy on macro and BOTH cost numbers, each labelled for what it means.** Never pair a macro
> accuracy with a sample-weighted cost.
> **The defensible joint claim:** *large latency and energy savings against a reasoning baseline; compute
> savings that are real but concentrated on low-escalation multiple-choice traffic rather than uniform.*
>
> **Mechanism:** escalation is wildly heterogeneous — PMC-VQA **8.45%**, SLAKE-cl 20.45%, PathVQA-cl
> 45.72%, VQA-RAD-cl 56.97%, MedXpert **89.60%**, SLAKE-open 15.81%, VQA-RAD-open 12.50%, PathVQA-open
> 35.67% — and PMC-VQA, the lowest-escalation cell, carried 79.2% of the sample-weighted average.
> Multiple-choice escalation: **16.22% → 44.24%** under equal weight.

**THE PREVIOUS CONVENTION (sample-weighted), kept for contrast only.** Source:
`artifacts/f8_mode_vsthink_ci.json` (2026-07-09) + `artifacts/opentext_32b_think_full.json`. Baseline
always-32B-with-reasoning 0.5591 (measured); always-32B-direct 0.5729; always-7B 0.5549. compute-lean
0.5741 at **0.492×** / 469 ms, **+0.0150 [+0.0107, +0.0192]**; accuracy-max 0.5836 at **0.932×** /
731 ms, **+0.0245 [+0.0216, +0.0274]**; fusion 0.5862 at 1.250× / 668 ms, +0.0271 [+0.0237, +0.0305].

**⛔ SUPERSEDED 2026-08-05 — the rule below still describes the *weighting*, but its VALUE is contaminated.
The clean vs-reasoning figure is +0.0601 [+0.0499, +0.0700] (deployed selector) / +0.0615 [+0.0514, +0.0715]
(frozen 8-seed selector), and the vs-direct claim is a TIE. See the RESOLVED box immediately below.**

**The canonicity rule is now: macro over 8 cells, Variant B, measured, veto lever ⇒ +0.0720
[+0.0614, +0.0824] vs the reasoning baseline.** **+0.0245 is "the sample-weighted equivalent"**, no
longer canonical. Older values circulate for the *same* operating point (+0.0212 / +0.0207 / +0.0238 /
+0.0275) — they differ by lever, pool, estimated-vs-measured open cells, **and now weighting (a fourth
axis)**. **Before quoting any headline value, read the decode table in retrospective §10.3.**

> # ⛔⛔ RESOLVED AGAINST US, 2026-08-05 — HOLE 4 IS CLOSED AND THE TABLE ABOVE IS CONTAMINATED
>
> **The disjoint-verifier retrain is DONE. It is no longer "in flight," and every open-text number in the
> table above was produced with the CONTAMINATED `lora_verifier_pooled4`** (confirmed in code — all three
> loaders defaulted to it — and by the contaminated arm reproducing
> `macro_average_headline_2026-07-30.json` exactly: 5,529 leaf fields, 0 differ).
> Source (verbatim): **`artifacts/cascade_selector_rerun_2026-08-05.json`**
> (`src/cascade_methods/cascade_selector_rerun.py`); the decontamination-only arm independently reproduces
> the earlier `artifacts/macro_headline_clean_verifier_2026-07-30.json` with max deviation 0.0.
>
> **The vs-32B-direct claim does not survive decontamination.** Macro, Variant B, 8 cells, nboot=10,000;
> baselines unchanged (7B 0.5971 · 32B-reasoning 0.5974 · 32B-direct 0.6567 · oracle-mode 0.6573):
>
> | arm | acc-max | vs reasoning | vs direct | compute-lean | vs direct |
> |---|---:|---|---|---:|---|
> | pooled4 (**contaminated**, the table above) | 0.6694 | +0.0720 WIN | **+0.0128 [+0.0056,+0.0200] WIN** | 0.6600 | +0.0033 TIE |
> | **disjoint** (clean verifier, deployed selector) | **0.6575** | **+0.0601 [+0.0499,+0.0700] WIN** | **+0.0008 [−0.0022,+0.0037] TIE** | 0.6443 | **−0.0124 LOSS** |
> | ens8_scaled (clean + frozen 8-seed selector) | 0.6590 | +0.0615 [+0.0514,+0.0715] WIN | +0.0023 [−0.0010,+0.0054] TIE | 0.6476 | **−0.0091 [−0.0153,−0.0031] LOSS** |
>
> **⇒ "accuracy-max beats always-32B-direct" is RETIRED. It is a TIE.** And **compute-lean vs direct goes
> from TIE to a SIGNIFICANT LOSS.** What survives is the **vs-reasoning** claim: **+0.0615 [+0.0514,
> +0.0715]**, with **−87.9% parallel latency and −84.3% energy** — and still **not** a FLOP-eq saving
> (macro acc-max **1.739×** direct as-charged, 1.395× honestly re-costed).
>
> **Attribution, on one shared bootstrap stream — decontamination is ~7× the selector and it is what
> flips the verdict.** Decontamination (pooled4 → clean) **−0.0119 [−0.0188, −0.0052]** macro, significant.
> Selection (clean → frozen 8-seed selector) **+0.0014 [−0.0003, +0.0032]**, **NOT significant**.
> Open-cell escalation moved 15.81/12.50/35.67% → **41.40/61.50/25.53%**: the open arm got expensive from
> *decontamination*, not from the selector.
>
> **⚠️ `ens8_scaled` carries one design decision that is NOT part of the measured recommendation.** The
> frozen selector's score is a *within-pool rank*, and `max(scores)` **is** the escalation gate — fed
> verbatim it takes 15 distinct values over 2,345 questions (vs the incumbent's 68) and the Weitzman
> controller collapses (SLAKE-open and VQA-RAD-open escalate 100% at meanN 0.0; PathVQA-open 0%). Its
> 0.746× compute is that collapse, **not a win — never quote it.** `ens8_scaled` quantile-matches the
> magnitudes onto the incumbent's scale (identical pick on all 2,345 items) so every gate feature is
> unchanged. **Pick `disjoint` as the conservative canonical** unless the scale-matching is separately
> justified.
>
> **Which selector is deployed is now nearly irrelevant to the headline** — see
> `docs/current/COMPARATIVE_VERIFIER_2026-08-05.md`: ~20 approaches converge on sel_eff 0.80–0.81, the
> seed spread (~0.021) exceeds every architectural effect, and **37.4% of questions have no correct answer
> anywhere in the 8-sample pool**. The coverage wall is ~4.5× the selection wall. Generator work outranks
> verifier work.

### The three findings that generalize (retrospective §5)

1. **Reasoning hurts perception; the reasoning-heavy "gain" is an ANSWER-FORMAT effect**
   *(re-derived 2026-07-29, reasoning half settled 2026-07-30 — see the correction notes below)*. On
   prompt- and resolution-matched arms, thinking is strictly worse than answering directly in **17/20**
   perception cells across 5 medical families — **14/20** with 95% CIs excluding zero, pooled
   **−0.0401 [−0.0456, −0.0347]** over **30,250** paired samples, **19/20** no better than +0.02 — and it
   reproduces at the same strength on arms that differ by nothing but the reasoning instruction. On
   reasoning-heavy benchmarks, **getting a reasoning-tuned model to emit a trace helps substantially, but
   the operative lever is the answer FORMAT (`\boxed{}`), not the reasoning instruction** — with the
   format matched, the trigger is worth ~nothing (**0/9** sub-cells CI-significant). Cost: 15–49× latency
   where a real think mode exists (MedVLThinker 49×, MedGemma 45×, QoQ 43×, Chiron 15×).
2. **Answer format determines whether routing signals work at all** — routing AUROC ~0.6 on 4-option
   MCQ vs ~0.87 on free text. The cause is option discreteness, not answer length.
3. **Training, not size, is the active ingredient in verification** — a trained 7B verifier beats or
   ties a zero-shot 32B verifier, confirmed three independent times.

> ### ⚠️ Finding 1 was corrected on 2026-07-29 — do not quote the old numbers
> Sources (use verbatim): **`results/cascade_methods/artifacts/finding1_corrected_2026-07-29.json`**
> and the audit **`.../finding1_prompt_matching_audit.json`**. Code:
> `src/cascade_methods/finding1_corrected.py`. Narrative: **retrospective §5.1, §10.1 C20–C25,
> §10.2 X15–X19, §10.5**.
> - **15/20 → 17/20.** The published arms were prompt-unmatched (and resolution-unmatched for
>   MedVLThinker). Three independent correction policies all give 17/20; **15/20 was the outlier**.
>   Two cells flip positive→negative (MedVLThinker PMC-VQA +0.0055 → −0.0075; Lingshu PMC-VQA
>   +0.0115 → −0.0425) — both on the internal harness, i.e. `test_clean.csv`, n = 2,000.
> - **WITHDRAWN — all 7 Lingshu-32B cells, both directions.** Its "native think" instruction
>   (`runners/run_native_think.sh:7`) is an answer-**format** string with no reasoning trigger (3.0
>   generated tokens). The repaired reasoning arm says perception 4/4 strictly negative, pooled
>   **−0.0866 [−0.0972, −0.0757]**, and reasoning **nothing**. ⇒ **Never cite Lingshu-32B as evidence
>   that reasoning helps.** Its **1.2× "think:no-think" ratio is not a reasoning ratio.**
> - **WITHDRAWN — QoQ-Med-VL-32B as reasoning evidence** (MMMU +0.071 → +0.012, CI spans zero;
>   MedXpert-Understanding significantly **negative**, −0.043, p = 0.022).
> - **MedGemma-27B on PathVQA is a real, significant exception** (+0.0413 [+0.0220, +0.0607], fully
>   matched) — the only perception cell where CoT genuinely helps.
> - **The open-text half of Finding 1 is PROVISIONAL.** `src/labeling/run_openvqa.py:26/27` has a live
>   style/length grading channel (the direct arm alone carries a persona and "short, specific phrase /
>   Do not explain"). A matched-prompt re-run is in flight. Do not quote the open-text
>   think-vs-direct delta (Δ = −0.154) until it lands. *(The **multiple-choice** matched re-run is now
>   complete — see the 2026-07-30 box below. The open-text one is a different run and is still outstanding.)*

> ### ⚠️ SETTLED 2026-07-30 — the reasoning half is an ANSWER-FORMAT effect (matched re-run COMPLETE, 6/6)
> Source (verbatim): **`results/cascade_methods/artifacts/medeval_matched_direct_2026-07-29.json`**
> (`src/labeling/medeval_matched_prompt.py` + `runners/run_medeval_direct_matched.sh`; **`MedEvalKit/`
> left byte-identical to upstream**). Narrative: **retrospective §5.1 box, §10.1 C27, §10.2 X22**.
> - **0/9 explicit-reasoning-TRIGGER effects are CI-significant** (9 sub-cells, n = 145 / 1,446 / 554,
>   paired on item id; 8/9 point-positive, mean delta shift from matching **−0.028**). **3/9 answer-FORMAT
>   effects ARE** significant. `parse_ok ≥ 0.9986` in every new arm (min over the 9 sub-cells; 1.000 in 6 of them) ⇒ not an extraction artifact.
> - Per cell, **published / format / trigger**: Lingshu MMMU +0.028 / −0.014 / +0.041 n.s. · MX-R
>   −0.004 / −0.008 / +0.004 n.s. · MX-U +0.000 / −0.002 / +0.002 n.s. · MVT-32B MMMU +0.103 / +0.062 /
>   +0.041 n.s. · MX-R +0.046 / **+0.046 SIG [+0.019, +0.072]** / +0.001 n.s. · MX-U +0.042 /
>   **+0.043 SIG** / −0.002 n.s. · IV3-38B MMMU +0.124 / **+0.090 SIG** / +0.035 n.s. · MX-R +0.035 /
>   +0.022 / +0.013 n.s. · MX-U +0.020 / +0.009 / +0.011 n.s.
> - **MECHANISM: asking for the answer in `\boxed{}` is itself a reasoning trigger.** MedVLThinker emits
>   **431–580** tokens on **99–100%** of items and InternVL3 **193–289** on **94–95%** with **no trigger
>   present**; Lingshu never does (3–4 tokens). The published deltas conflated "reasoning vs not" with
>   "boxed vs bare letter".
> - **DROP "a reasoning instruction improves accuracy on reasoning-heavy benchmarks."** **KEEP** the
>   weaker supported form: getting a reasoning-tuned model to *emit a trace* helps substantially
>   (MVT MMMU +0.103, MX-R +0.046; IV3 MMMU +0.124) — with the **format** named as the lever.
> - **Lingshu-32B must not be cited as reasoning evidence at all** (genuinely-reasoning vs
>   genuinely-direct: +0.041 MMMU n.s., ~0 on both MedXpert splits). Strengthens C22.
> - **The cascade's gated-reasoning tier keeps its FULL value** — the rung1→rung3 total is what a think
>   tier delivers; only the **attribution** changes.
> - **The honest substitute for the unobtainable clean contrast is the monotone ladder** (MVT MMMU-MCQonly:
>   **0.634 @ 2 tok → 0.697 @ 431 → 0.738 @ 580**).
> - **STANDING RULE:** any future think-vs-direct pair must be **format-matched AND token-audited**. A
>   "direct" arm that emits hundreds of tokens is not a direct arm.
> - **`MedEvalKit/` had two local edits — ✅ REVERTED 2026-07-29 23:36:53, verified clean 2026-08-10.**
>   `utils/question_formats.py` and `utils/MMMU/data_utils.py` added a reasoning trigger but **deleted**
>   the answer-format clause the direct arm still carries. Both worktree blobs are now **identical to
>   MedEvalKit's own HEAD** (`git diff` empty; blobs `045615ba…` / `143c024c…`), so the harness is back to
>   upstream and the matched-prompt re-run was produced against it.
>   **The dump-validity consequence stands and is permanent:** dumps written *before* the revert-era edits
>   (`eval_results_*_think`) are invalid as reasoning evidence, and the `*_reason` dumps made *while* the
>   edits were live reason but are **format-unmatched**. Judge any dump by its own date and its mean
>   generated tokens, never by the current state of these two files.
>   **`MedEvalKit/` is a protected dependency — do not modify it.** (Its worktree still shows unrelated
>   pre-existing churn — a modified `benchmarks.py` and deleted `datas/` fixtures — which is *not* ours
>   and must be left alone.)
> - **Lesson to apply going forward:** prompts are **not** persisted in the checkpoint rows (they live
>   only in `runners/*.sh` shell variables and module constants). **Persist the prompt in every new
>   checkpoint, and assert mean generated tokens on every think arm.**

### The two walls (the negative-results contribution)

- **Recoverability wall** — "will the strong model fix *this* error?" is ~0.5–0.6 AUROC from anything
  cheap. **16 independent mechanisms** hit it.
- **Selection wall** — a verifier converts only **74–82%** of oracle-of-N. **13 attempts** hit it,
  killed three ways (capacity, compounding, pre-filtering).
- Related: the **coverage** wall is 4.5× the selection wall (40.8% of questions have no correct answer
  anywhere in an 8-sample pool). Generator work outranks verifier work.

### Decisions and caveats that a new session must know

- **MMMU-Medical is EXCLUDED** (2026-07-08). Lingshu-7B scores 0.80 there vs its own published 54.0 and
  beats its own 32B; an adversarial audit (model identity PASS, image ablation, control model) concluded
  **genuine weights, consistent with train-set contamination outside our control**. Do **not** bank the
  MMMU keep-7B "+0.140 beat-the-32B win" — three July-7 docs did, and it was retracted. Retrospective
  §2.12, C12.
  **⚠️ 2026-07-30 — the exclusion RATIONALE inverts under macro.** It used to be argued as immaterial
  ("MMMU is 0.35% of the items, the headline moves −0.0005"). Under equal weight MMMU would carry
  **1/9 = 11.1%** of the headline, so excluding the contaminated cell is a **large and consequential
  decision** and must be re-argued **on contamination grounds alone**, with its size stated: macro-9 vs
  macro-8 vs always-32B-direct would be compute-lean **+0.0215 vs +0.0033**, accuracy-max **+0.0299 vs
  +0.0128**, fusion **+0.0269 vs +0.0094**.
- **The honest framing (§8.5), REPLACED 2026-07-30.** The old wording — *"matches the strong model at
  roughly half the compute, with a significant accuracy gain on two specific cells…"* — **fails on both
  halves at equal weight**: "matches" is a significant LOSS on the multiple-choice cells
  (−0.0070 [−0.0126, −0.0017]) and "half the compute" becomes **1.196×**. Use instead, verbatim from
  `artifacts/macro_average_headline_2026-07-30.json:honest_headline.sentence`:
  *"Under equal weight per benchmark cell (8 cells, 1/8 each), the accuracy-max setting beats
  always-32B-direct on accuracy by +0.0128 [+0.0056, +0.0200] — but it now costs 1.41× that baseline's
  compute, not less, and the compute-lean setting neither matches it on the multiple-choice half
  (−0.0070 [−0.0126, −0.0017], a significant loss) nor stays cheap (1.20× compute, up from 0.49×
  sample-weighted); the one baseline the method still clearly beats at equal weight is a 32B actually made
  to reason (+0.0720 [+0.0614, +0.0824] accuracy, −89% latency, −87% energy — though not fewer FLOP-eq:
  1.41× as charged, 1.13× honestly re-costed)."* Not a suite-wide accuracy advantage, and not a
  suite-wide compute advantage either.
- **⚠️ LANDMINE — there are TWO PMC-VQA splits in this project, one per evaluation track, and they are
  not comparable.** The **MedEvalKit/Lingshu track** (the paper) uses **`test_2.csv`** (v2, **33,430**
  items, **79%** of the Variant-B pool) — hard-coded in unmodified vendor code at
  `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`, and the split with **zero published verification**. The
  **internal-harness / MedVLThinker-Eval track** (the June cascade, and every `ckpts/acc_gen/` dump)
  uses **`test_clean.csv`** (v1, **2,000** items, **24.3%** of the 8,220 pool) — the authors' only
  human-verified split. **`test_clean` ∩ `test_2` = 6 items**, so never filter, average, or compare a
  number from one track against the other, and always quote a PMC-VQA number with its file name and row
  count. Full provenance (construction, validation, which dumps are which):
  **`results/cascade_methods/docs/current/PMCVQA_PROVENANCE_2026-07-30.md`**.
- **Known critical holes** (§7): the vs-reasoning baseline is a no-reasoning run on ~90% of the pool
  charged at reasoning cost (hole 1 — measured generated tokens are **3.0–3.3** on the four closed cells
  versus 104–320 on MedXpert and the three open cells); the open-text verifier scores items it was trained
  on, ~31% inflation (hole 4 — **the disjoint retrain is in flight and gates the open-text claim**); and
  new **hole 17**: the thresholds were tuned against a **pooled** objective but the report is now **macro**,
  and a macro-objective refit has not been done. These are open, not fixed.
  *(Hole 2's "89% of the headline delta comes from 2 of 8 cells" phrasing is **retired** — under equal
  weight every cell contributes exactly 1/8 of its own delta, so concentration is now reported as a
  **leave-one-cell-out range**. PathVQA-open is the load-bearing cell of every vs-reasoning/vs-direct
  claim: dropping it takes accuracy-max vs reasoning from +0.0720 to +0.0318.)*
- **Preservation risk — ✅ COMMITTED, ⚠️ NOT PUSHED, ⛔ AND THE EXPENSIVE INPUTS ARE IN NEITHER.**
  *(Was: "last commit `8cdefef` (2026-07-02), the whole Lingshu/July chain is untracked." That is fixed —
  the chain is committed. Re-verified 2026-08-10 after a reboot.)*
  - **Committed:** HEAD is `f3d0f82` (2026-08-05). `git fsck --full --strict` clean; all **734** tracked
    files re-hashed byte-identical to the index; working tree clean.
  - **⚠️ NOT PUSHED: `main` is 63 commits ahead of `origin/main` (`9829b71`)** — 554 files, +493,349 lines,
    i.e. the entire Lingshu/July/August chain, on **one disk**. `git push` is the top-priority chore.
  - **⛔ A push does NOT protect the inputs.** `results/` has **269 tracked files** (the artifacts and docs
    — the *numbers* travel), but **`ckpts/`, `feats_hidden/` and `logs/` have zero tracked files** and are
    gitignored. The reproduction chain is therefore **unbacked**: `feats_hidden/` (4.4 GB, regenerable only
    by an ~88-min dual-A100 pass, and only while the Lingshu-7B snapshot hash survives in
    `/data/dan/hf_cache/hub`), `ckpts/train/genframe_head_ens8/` (29.4 MB frozen selector) and
    `ckpts/train/lora_verifier_disjoint/` (190 MB adapter + the three transfer dumps that define the
    n=2,345 pool and the incumbent bar). Copy these to `/data` or external storage — a push will not.
  - **⚠️ `src/training_methods/freeze_selector.py` REWRITES `ckpts/train/genframe_head_ens8/`.** Never
    re-run it casually: a refit is a fresh seed draw, and `recipe.json` records that the CPU trainer's
    batch permutation is thread-count sensitive (seed-0 sel_eff **0.795640** at the pinned thread count vs
    **0.800409** at 8 threads). **The frozen `.pt` files are the artifact of record, not the recipe.**
- The earlier MedVLThinker work (margin gate τ=0.426, the 3-tier ACC cascade) is **historical**, not the
  live method — see §1 below.

---


---

## 1. The project in one paragraph

This is a **test-time-compute for medical Vision-Language Models** research project. A VLM takes an
**image + a text question** and returns a text answer — here the images are medical (radiology,
pathology, endoscopy) and the task is medical visual question answering (VQA), in **both** formats:
multiple-choice and open-ended free text. The thesis: **the accuracy–cost tension in medical VQA is not
a law, it is a consequence of spending test-time compute uniformly** — the same reasoning, the same
number of samples, the same model, on every question regardless of what the question needs. The method
is a cascade that spends selectively. **The deliverable is `paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf`**
(9 pages, IEEEtran). The original CVGIP 2026 target and its drafts are in `paper/archive/`.

**Live headline numbers → §0 above, and retrospective §4.** Do not quote numbers from this file.

### Historical headline (2026-06, MedVLThinker era — NOT the current result)

Kept because ~half the repo, all the June diaries, and `docs/archive_mcq/` are written against it.

- Models `MedVLThinker-7B/32B-RL_m23k`; a **frozen margin gate τ = 0.426** calibrated on a held-out
  PMC-VQA train sample (v1 `train.csv`, n = 3,000); cheap leg served at reduced resolution (**cap320**).
- Parity with always-32B (`0.5718` vs `0.572`) at **73.6%** of always-32B prefill-inclusive compute,
  63.3% escalation, over **6 benchmarks / 8,220 samples** (`progress/progress_June_17.md` §0).
- Its successor, the 3-tier **compute-configuration cascade** (ACC: 7B-direct@cap320 → 32B-direct@cap320
  → 32B-reasoning@fullres): at parity with always-32B-reasoning, **latency 11.34 s → 2.27 s (−80%),
  energy 6,318.8 J → 1,181.9 J (~5.3×), compute 100 → 52%** on the 6-benchmark pool
  (`artifacts/master_data.csv`, canonicalized in `INCONSISTENCIES.md` X1). Spec:
  `docs/current/METHOD_ACC.md`. Reproduce: `python3 src/cascade_methods/acc.py`.
- Its scope was the **four "competent" benchmarks** (PMC-VQA — here `test_clean.csv`, n = 2,000 —
  SLAKE, VQA-RAD, PathVQA); MMMU and MedXpert were excluded (both models near chance on MedXpert).
- **This era's evaluation harness is the internal NGC one, not MedEvalKit** — retrospective §9.3 lists
  three separate evaluation contexts (A faithful MedEvalKit / B internal 5-family bake-off / C custom
  open-text judged pipeline). **Never cross-multiply numbers across them.**

---
