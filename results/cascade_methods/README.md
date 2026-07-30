# Cascade-methods results — index

This folder holds every writeup and raw output of the test-time-compute research for medical VQA —
cheap VLM + gate/router + trained verifier + strong VLM, across both multiple-choice and open-ended
answers. All numbers come from real checkpoints; **no fabricated values** (CLAUDE.md critical rule 7).

> **➜ START HERE: [`docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`](docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md)**
> — the definitive account of the whole project (2026-06-17 → 2026-07-29): the arc and every pivot,
> the method, the results with CIs, ~90 negative results, the 17 honest holes, and the corrections log.
> **Read it before quoting a number from any other file in this folder.**

> **Two settled corrections propagated 2026-07-30** — the macro re-basing / cost reversal and the
> answer-format re-attribution of Finding 1's reasoning half. See the two ⚠️ seam boxes below.

> **Reorganized 2026-07-02.** Was a flat dump of ~108 mixed `.md`/`.json`/`.txt` files; now split
> into `docs/` (writeups), `artifacts/` (raw outputs), and `claude_judge/` (judge verdicts).

## Layout
```
docs/current/      canonical writeups + method specs — START HERE
docs/archive_mcq/  superseded MCQ-era research-loop writeups (kept as the negative-result record)
artifacts/         raw analysis outputs (.json/.txt/.jsonl/.csv) — gitignored, regeneratable
claude_judge/      Claude-as-judge verdicts (open-text grading, Max-plan subagents; committed)
METHOD_IDEAS_BACKLOG.md   68 cross-field ideas, §A–H
```
Scripts live in `src/cascade_methods/` (+ `src/cascade/measure_config.py`, `src/training_methods/`).
**Always run from the repo root**; scripts read/write `results/cascade_methods/artifacts/`.

## ⚠️ The numeric seam — read before using any table in `docs/current/`

The four big July-8 consolidation docs were written on **2026-07-08 at 06:0x**. **Three things happened
later that day and the next** and invalidated their headline rows:

1. the oracle-mode-32B baseline was added (`paper_baselines.json`);
2. the **MMMU contamination** audit → the decision to **exclude MMMU** ("Variant B", n = 42,224);
3. the **estimated** 32B-reasoning open-text cells were replaced with **measured** ones, moving the
   always-32B-reasoning baseline from **0.5632 → 0.5594 (full suite) / 0.5591 (Variant B)**;
4. and on 2026-07-09 the last rigor gap was closed with a CI (`f8_mode_vsthink_ci.json`);
5. and on **2026-07-30** the reporting metric itself changed — see the third seam immediately below.

**Sample-weighted values (the old convention):** accuracy-max **+0.0245, 95% CI [+0.0216, +0.0274],
n = 42,224, at 0.932× compute** (`artifacts/f8_mode_vsthink_ci.json`); compute-lean **+0.0150
[+0.0107, +0.0192]** at 0.492×.
Values of **+0.0212 / +0.0207 / +0.0238 / +0.0271 / +0.0275** are the *same method* under a different
lever, pool, or estimate-vs-measured convention — decode table in retrospective §10.3.
**Never write "Baselines (measured): 0.5632"** — that value's open cells were estimates.

## ⚠️ The third seam (2026-07-30): the metric is MACRO now, and the COST claim REVERSES

**Primary average = equal weight per reporting cell (8 cells, 1/8 each)**, not sample-weighted, because
sample-weighting let PMC-VQA (**79.2%** of items) speak for the method. Reweighting: PMC **79.2% →
12.5%**; open-text arm **5.6% of items → 37.5% of weight**; closed/MCQ **94.4% → 62.5%**. The
5-benchmark macro (1/5 each) is a **secondary robustness check only**. Source (verbatim):
**`artifacts/macro_average_headline_2026-07-30.json`** ← `src/cascade_methods/macro_average_headline.py`.

**Canonical today (macro, 8 cells):** accuracy-max **0.6694**, **+0.0720 [+0.0614, +0.0824]** vs
always-32B-with-reasoning and **+0.0128 [+0.0056, +0.0200]** vs always-32B-direct; compute-lean
**0.6600**, **+0.0626 [+0.0514, +0.0734]** / **+0.0033 [−0.0054, +0.0121] n.s.**; fusion **0.6661**,
**+0.0686** / **+0.0094 [+0.0013, +0.0176]**. Baselines macro: always-7B **0.5971**,
always-32B-with-reasoning **0.5974**, always-32B-direct **0.6567**, oracle-mode **0.6573**.
**The canonicity rule is now: macro over 8 cells, Variant B, measured, veto lever. +0.0245 is "the
sample-weighted equivalent".**

**⛔ "Pareto-dominates every fixed way of using the 32B" is RETIRED** (retrospective §10.1 C26). Under
equal weight **no operating point is compute-cheaper than always-32B-direct**: compute-lean **1.196×**,
accuracy-max **1.410×**, fusion **1.435×** FLOP-eq (were 0.492× / 0.932× / 1.250×). "Pareto-**optimal**"
survives — the points are non-dominated because they are more *accurate*, not cheaper. And compute-lean
is a **significant LOSS on the 5 multiple-choice cells: −0.0070 [−0.0126, −0.0017]** vs always-32B-direct
(−0.0080 [−0.0137, −0.0024] vs oracle-mode). Mechanism: escalation runs **8.45%** (PMC) to **89.60%**
(MedXpert), so MCQ escalation goes **16.22% → 44.24%** at equal weight.

> **NUANCE, required wherever cost is claimed.** Macro-averaging **cost** answers a **different question**
> from sample-weighted cost. Cost is additive per query, so on traffic resembling this suite the **~0.49×**
> saving is what you would actually pay; the macro number tests whether the saving **generalises across
> task types** — and it does not, it is concentrated on the low-escalation cells. **Report accuracy on
> macro and BOTH cost numbers, each labelled.** The defensible joint claim: *large latency and energy
> savings against a reasoning baseline; compute savings that are real but concentrated on low-escalation
> multiple-choice traffic rather than uniform.* What survives cleanly: vs a 32B **actually made to reason**,
> **+0.0720 accuracy at −89% latency / −87% energy** (honestly re-costed) — but **1.41× as charged /
> 1.13× honest on FLOP-eq**, i.e. not cheaper on compute there either.

**⏳ OPEN:** the **open-text accuracy claim is PROVISIONAL — a clean-verifier (disjoint-split) retrain is
in progress** (`artifacts/verifier_disjoint_split.json`) and will determine whether it is contaminated
(the verifier was trained on ~70% of its own evaluation items; retrospective §7 hole 4). The open arm
holds 37.5% of the macro weight and is the load-bearing cell of every headline delta.

## ⚠️ The second seam: **two PMC-VQA splits** — always name the file

The MedEvalKit/Lingshu docs report PMC-VQA on **`test_2.csv`** (v2, **33,430** items, **79%** of the
Variant-B pool, zero published verification, hard-coded at `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`).
The June MedVLThinker/internal-harness docs report it on **`test_clean.csv`** (v1, **2,000** items, the
authors' only human-verified split, **24.3%** of the 8,220 pool). **They intersect on 6 items** — never
compare, average, or filter across them. Every PMC-VQA number must carry its file name and row count.
Full provenance: **[`docs/current/PMCVQA_PROVENANCE_2026-07-30.md`](docs/current/PMCVQA_PROVENANCE_2026-07-30.md)**.

## `docs/current/` — the canonical writeups

| File | What it is | status |
|---|---|---|
| **`PROJECT_RETROSPECTIVE_2026-07-29.md`** | **★ START HERE** — the definitive account of the whole project. | **current** |
| **`PMCVQA_PROVENANCE_2026-07-30.md`** | **Read before quoting any PMC-VQA number** — how PMC-VQA is built (caption-only GPT-3.5 generation), how thinly it is validated, which splits exist, and which split each of this project's two evaluation tracks used. | **current** |
| `METHOD_FINAL_2026-07.md` | **The method spec** — every lever, every ablation. What you would re-implement from. | mechanism current; **numbers pre-seam** |
| `RESEARCH_RESULTS_2026-07.md` | The full results ledger (76 KB) — the experiment-by-experiment record. | mechanism current; **numbers pre-seam** |
| `TECHNICAL_REPORT_2026-07.md` | The technical walkthrough (the bridge from the paper to the spec). | mechanism current; **numbers pre-seam** |
| `OMNIMED_FALLBACK.md` | Why the OmniMedVQA strong leg is a documented keep-cheap fallback (NCCL hang). | current |
| `UNIFIED_METHOD_EXPERIMENTS.md` | Running experiment log for the open-text method. | current-as-log |
| `VERIFIED_FACTS.md` | Every load-bearing fact traced to its source — build only on these. | current-as-reference |
| `OPENTEXT_MASTER_TABLE.md` | 3-family open-text cascade master table (judge-graded). | current-as-reference |
| `OPENTEXT_BASELINE.md` | Open-text (judge-based) baseline — the comparison anchor. | current-as-reference |
| `METHODS_MASTER.md` | Was "single source of truth for the paper's Method section" (2026-06-29). | **superseded** by `METHOD_FINAL_2026-07.md` + the 2026-07-08 IEEE rewrite |
| `MASTER_SUMMARY_2026-07.md` | Top-level consolidation as of 2026-07-02 (2-tier MCQ efficiency era). | **superseded** by `METHOD_FINAL_2026-07.md` |
| `METHOD_ACC.md` | The Adaptive-Compute Cascade — the MedVLThinker-era structural win. | **historical** (survives as "the journey") |
| `METHOD_MATH.md` | ACC math: scores, expected cost, latency, energy. | **historical**, equations still used |
| `METHOD.md` | Compute-configuration cascade writeup (2026-06-17). | **historical** |
| `METHOD_deferral_router.md` | VADR final verdict — *"NOT novel, not a real win"*; self-labelled superseded. | **historical / negative result** |

## `docs/archive_mcq/` — superseded MCQ-era loop (record of what was tried & why it was dropped)
Findings/narrative: `FINDINGS.md`, `GROUND_TRUTH_NUMBERS.md`, `FULL_RECORD.md`, `MASTER_TABLES.md`,
`DETAILED_TABLES.md`, `2SIZE_VALIDATION.md`, `RESEARCH_REFOCUS_2026-06-28.md`, `AUXILIARY_RESEARCH.md`.
Novel-method attempts: `NOVEL_METHOD_FLD.md`, `NOVEL_METHOD_VISUAL_STABILITY.md`,
`ACCV3_V4_AND_NOVELTY.md`, `RESCUE_INTO_ACCV2.md`, `SELECTIVE_ABSTENTION.md` ⛔.
Open-ended / verifier phase: `OPENENDED_CASCADE.md`, `OPENENDED_SELECTION_LUCKFLOOR.md`,
`TRAINED_VERIFIER_RESULT.md`, `BOX_VERIFIER_RESULT.md`, `NEW_DIRECTIONS_2026-06-25.md`,
`RECOVERABILITY_IS_CAPACITY_BOUND.md`, `KNOWLEDGE_AUGMENTATION_FEASIBILITY.md`.

> ⛔ **`SELECTIVE_ABSTENTION.md` documents a permanently forbidden direction.** Abstention /
> reject-option / defer-to-human has been out of scope since 2026-07-07. That file is **historical
> record only** — never a live proposal. The method always answers.

## Bottom line

1. **The deliverable is a format-aware adaptive cascade** (Lingshu-7B → Lingshu-32B, evaluated on the
   faithful MedEvalKit harness). It detects multiple-choice vs open-ended **from the prompt text alone**
   and runs a different policy for each: an MCQ arm (margin gate → 32B in **direct** mode) and an
   open-text arm (7B best-of-N with adaptive N → **trained LoRA verifier** selects → escalate on low
   verifier confidence). One knob, two settings. **⚠️ 2026-07-30: "both using less compute than a single
   32B forward" was a SAMPLE-WEIGHTED claim and is retired** — at equal weight per cell the settings cost
   **1.196× / 1.410×** a single 32B forward. What holds: both beat always-32B-**with-reasoning** on
   accuracy at ~a tenth of its latency and energy; accuracy-max also beats always-32B-**direct** on
   accuracy (+0.0128 [+0.0056, +0.0200]) at 1.41× its compute.
   Spec: `docs/current/METHOD_FINAL_2026-07.md`. Code: `src/cascade_methods/method_final.py`.
   Paper: `paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf` (**the `.tex` carries the corrections; the
   PDF and `paper/figs_final/fig_pareto.pdf` do NOT and were deliberately not rebuilt**).
2. **Three findings that generalize** — (a) reasoning **hurts perception** across 5 families /
   3 architectures — **17/20 perception cells strictly negative**, 14/20 with 95% CIs excluding zero,
   pooled **−0.0401 [−0.0456, −0.0347]** on 30,250 paired samples *(re-derived 2026-07-29 from
   prompt-matched arms; the previously published 15/20 was the outlier — `artifacts/finding1_corrected_2026-07-29.json`,
   retrospective §5.1)*. On reasoning-heavy benchmarks the apparent gain is an **ANSWER-FORMAT effect**
   *(settled 2026-07-30, matched re-run complete 6/6 cells —
   `artifacts/medeval_matched_direct_2026-07-29.json`, retrospective §10.1 C27)*: with the format matched,
   the reasoning **instruction** is worth ~nothing (**0/9** sub-cells CI-significant) while the
   **`\boxed{}`** contrast is significant in **3/9** — **asking for `\boxed{}` is itself a reasoning
   trigger** (MedVLThinker emits 431–580 tokens, InternVL3 193–289, with no trigger present; Lingshu never
   does). Keep the weaker form: *getting a reasoning-tuned model to emit a trace helps substantially, via
   the answer format.* **Lingshu-32B must not be cited as reasoning evidence at all.** The cascade's
   gated-reasoning tier keeps its full value; only the attribution changes. The **open-text** version of
   the finding is still **provisional**;
   (b) **answer format** decides whether
   routing signals work at all (AUROC ~0.6 on 4-option MCQ vs ~0.87 on free text — the cause is option
   discreteness, not answer length); (c) **training, not size**, is the active ingredient in
   verification (a trained 7B verifier beats or ties a zero-shot 32B one, three independent times).
3. **Two walls, and they are the real contribution.** *Recoverability* — "will the strong model fix
   **this** error?" is ~0.5–0.6 AUROC from anything cheap; **16 independent mechanisms** hit it, so the
   confidence/margin gate is approximately optimal and cannot be improved by substitution. *Selection* —
   a verifier converts only **74–82%** of oracle-of-N; **13 attempts** hit it, killed by capacity,
   compounding and pre-filtering alike. And the **coverage** wall is 4.5× larger than the selection wall
   (40.8% of questions have no correct answer anywhere in an 8-sample pool), which is why generator work
   now outranks verifier work.
4. **Historically, ACC** (the 3-tier compute-configuration cascade) was the MedVLThinker-era structural
   win and is preserved in `docs/current/METHOD_ACC.md`. It is **not** the current method: on Lingshu its
   slow reasoning tier would fire ~0% of the time. What carried forward is the *structure* and the
   compute-configuration idea — not the gate, whose agreement rule turned out to be prior art
   (Agreement-Based Cascading, arXiv 2407.02348).
5. **Read the weaknesses.** Retrospective §7 ranks 17 holes, 3 critical: the vs-reasoning baseline is a
   no-reasoning run on ~90% of the pool charged at reasoning cost; the headline delta is concentrated in
   PathVQA-open (leave-one-cell-out takes accuracy-max vs reasoning from +0.0720 to +0.0318 when it is
   dropped — the old *"89% from 2 of 8 cells"* phrasing is **retired**, since at equal weight each cell
   contributes exactly 1/8 of its own delta); and the open-text verifier scores items it was trained on
   (~31% inflation — **retrain in flight, and it gates the open-text claim**). New **hole 17**: the
   thresholds were tuned against a *pooled* objective but the report is now *macro*, and a macro-objective
   refit has not been done.

## `artifacts/` (gitignored)

Raw outputs written by the analysis scripts (~109 `.json` files). **The headline chain:**
**`macro_average_headline_2026-07-30.json`** (**the canonical headline — macro levels, both cost
weightings, every delta with CI and leave-one-cell-out, and the `honest_headline` block**),
`f8_mode_vsthink_ci.json` (the sample-weighted headline CI), `honest_recosting_2026-07-29.json`,
`method_final.json` + `method_final_v2.json`, `method_final_mmmu_corrected.json` (**stale: Variant A,
sample-weighted, 32B-reasoning = 0.5628 *estimated* — and still the file `paper/make_ieee_figs.py` builds
the Pareto figure from, so figure and table now disagree; retrospective §7 hole 14**),
`paper_baselines.json`, `opentext_32b_think_full.json`.
**Finding 1's reasoning half:** `medeval_matched_direct_2026-07-29.json` is **canonical** for the
format-vs-trigger decomposition. **Open-text verifier contamination:** `verifier_disjoint_split.json`
(the in-flight retrain's design; `needs_generation`).
**Finding 1 (cross-family reasoning-vs-direct):** `finding1_corrected_2026-07-29.json` is **canonical**
(← `src/cascade_methods/finding1_corrected.py`); `finding1_prompt_matching_audit.json` is the audit that
triggered it. ⚠️ **`GENERALIZATION.md` / `generalization.json` are SUPERSEDED for Finding 1** (they carry the
old 15/20 count from prompt-unmatched arms; the `.md` is banner-marked, and `generalization.json` now carries a
`_SUPERSEDED_FINDING1_2026-07-29` key with a verbatim pre-correction copy in
`generalization_superseded_2026-07-08.json`). Findings 2 and 3 in them stand.
`reframe_vs_bigthink.json` and `master_data.csv` hold the **same unmatched** per-benchmark
`dThink_minus_nt` cells and are **not** annotated — use them for the cascade reframe, never for Finding 1.
Also: `master_data.csv` + `GENERALIZATION.md` (the 5-family matrix), `integrated_method_vs_think.json`,
`pandora_controller.json`, `beat32b_fusion.json`, `beat32b_more.json`, `escalation_levers.json`,
`robust_slice_routing.json`, `logit_fusion.json`, `verifier_32b_gpu.json`, `literature_raw.json`,
`allmethods_<family>.json`, `leaderboard_*.json`, `frontier_*.json`, `latency_{7b,32b,7b_think}.jsonl`,
`open_*.json`, and the saved `.txt` console dumps.
Regenerate via the scripts in `src/cascade_methods/` — see retrospective §4.6 for the number → script →
artifact table.
