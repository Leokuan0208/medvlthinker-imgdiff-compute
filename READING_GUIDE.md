# Reading Guide — how to understand this whole project, in order

> **Purpose.** There is a lot here: 7 root docs, a 1,972-line retrospective, 14 current writeups, 20
> archived ones, **13 dated progress diaries**, an IEEE paper, 3 HTML decks, a 68-idea backlog, and ~199
> Python files. This guide is the **reading order**: exactly which *section of which file* to read, what
> you'll learn, and why it matters.
>
> **Rewritten 2026-07-29.** The previous version stopped at June 26, pointed at a paper draft that has
> since moved to `paper/archive/`, and described the (now permanently forbidden) abstention work as part
> of the arc. If you have a bookmarked copy of the old ordering, discard it.
>
> **How the documents relate (mental model):**
> - **`results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md` = the definitive
>   account.** Everything else is either an excerpt of it, a source for it, or older than it.
> - `meetings/progress_report_professor_2026-07-27.html` = the **best single presentation** of the work.
> - `PROJECT_OVERVIEW.md` = the **plain-language 30-minute version**.
> - `paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf` = the **polished, publishable story**.
> - `progress/progress_*.md` = the **diaries** (chronological, messy, and the most *trustworthy* layer —
>   they routinely flag their own errors).
> - `CLAUDE.md` = the **briefing + the hard rules** (including the abstention prohibition).
> - `STRUCTURE.md` = the **map of the code**.
> - `results/cascade_methods/docs/` = the **deep-dive appendices** (`docs/current/` canonical,
>   `docs/archive_mcq/` superseded-but-preserved).
>
> **Three reading depths — pick one:**
> - 🟢 **45-minute skim** → Steps 0, 1, 2 only. (The whole arc + the headline + the honest caveats.)
> - 🟡 **Half-day, full understanding** → Steps 0–9 in order. (Recommended.)
> - 🔴 **Everything incl. code, negatives & proofs** → all steps, including Parts 4 and 5.

> ## ⚠️ Three things to know before you read a single number
> 0. **There are two different PMC-VQA splits in this project.** The MedEvalKit/Lingshu docs (the paper)
>    use **`test_2.csv`** (v2, 33,430 items, 79% of that pool, no published verification); the June
>    MedVLThinker/internal-harness docs use **`test_clean.csv`** (v1, 2,000 items, the authors' only
>    human-verified split, 24.3% of that pool). They overlap on **6 items**, so a PMC number from one is
>    never comparable to one from the other. Read
>    `results/cascade_methods/docs/current/PMCVQA_PROVENANCE_2026-07-30.md` before quoting any PMC-VQA
>    figure.
> 1. **There is a numeric seam at 2026-07-08 ~08:00.** The four big July-8-morning consolidation docs
>    (`TECHNICAL_REPORT`, `METHOD_FINAL`, `RESEARCH_RESULTS`, and the pre-2026-07-29 `PROJECT_OVERVIEW`)
>    were written *hours before* three things that changed the headline: the oracle-mode baseline, the
>    MMMU exclusion, and the replacement of **estimated** 32B-reasoning open-text cells with **measured**
>    ones. Read those four for **mechanism**, not for **values**.
> 2. **The canonical result is `+0.0245 [+0.0216, +0.0274]` at 0.93× compute, n = 42,224**
>    (`artifacts/f8_mode_vsthink_ci.json`). Values like +0.0212 / +0.0207 / +0.0238 / +0.0271 / +0.0275
>    are the *same method* under a different lever, pool, or estimate-vs-measured convention. Decode
>    table: retrospective §10.3.

---

## PART 0 — The definitive account (≈2–3 hr, or 30 min for §1–§4) 🟢

**Step 0 — `results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`.**
*Read:* §1 (what the project set out to do and how the question changed four times) → §2 (the arc, every
phase with its decisive number) → §3 (the method as it stands) → §4 (the results). Then, before quoting
anything anywhere: **§7 (the 16 holes)** and **§10 (the corrections log)**.
*Why:* it is the only document that is simultaneously complete, current and source-cited, and it
deliberately avoids the project's internal codenames (a glossary for the older docs is in §9.5).
**If you read only one file, read this one.**

---

## PART 1 — Orientation (≈45 min) 🟢

**Step 1 — `meetings/progress_report_professor_2026-07-27.html`** (open in a browser).
*Learn:* the whole project in 13 sections, every claim tied to a named artifact, plus a glossary.
*Why:* the best-presented artifact in the repo. *Caveat:* its slide-6 baseline table shows accuracy-max
at 5.70 compute-units (= 1.25×, the **fusion** variant) while its hero claim is 0.93× — two different
variants share one label. Trust the retrospective's §4 table over the deck where they differ.

**Step 2 — `PROJECT_OVERVIEW.md`, §1–§5 + §3a.**
*Learn:* the plain-language method — what a cascade is, what the two arms do, why the baseline choice was
the insight, and why MMMU is excluded. *Why:* it is the 30-minute version of Step 0, written for a
non-specialist, and it now carries the corrected numbers.

**Step 3 — `CLAUDE.md`, the CRITICAL RULES block + §0 + §3 (glossary).**
*Learn:* the seven standing rules (including the **permanent abstention prohibition** and the
no-fabricated-numbers rule), where the project stands, and every term you will meet — margin, τ,
escalation, recoverability vs detection, the luck floor, guardrail, Variant A/B, certified veto.
*Why:* nothing else parses without the vocabulary, and the rules are non-negotiable.

---

## PART 2 — The polished story: the paper (≈1 hr) 🟡

**Step 4 — `paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf`** (source `.tex` alongside; build with
`bash paper/build_ieee.sh`).
*Read:* the Abstract and Introduction (contributions **C1–C4**) first, then the Findings, Method, Main
Result and Limitations sections. *Why:* this is **the deliverable**, and — with the July-27 deck — the
only prose artifact on disk that carries the *corrected* numbers (Variant B, measured, CI'd). It uses no
codenames and its related-work section explicitly concedes that the multiple-choice gate is not novel.
*Note:* superseded drafts (`manuscript_final_2026-07.md`, `cvgip2026_draft.md`, …) are in
`paper/archive/` — see `paper/README.md` for the naming convention.

**Step 5 — `results/cascade_methods/docs/current/TECHNICAL_REPORT_2026-07.md`.**
*Learn:* the technical walkthrough of the same story, in more depth than the paper allows.
*Why:* the bridge from the paper to the specs. **Numbers pre-date the 2026-07-08 measurement** — read
for mechanism; take values from retrospective §4.

**Step 6 — `results/cascade_methods/docs/current/METHOD_FINAL_2026-07.md`.**
*Learn:* the full method spec, every lever, and the ablations behind each design choice.
*Why:* this is what you would re-implement from. Same numeric caveat as Step 5. Companions:
`METHOD_MATH.md` (cost/latency/energy equations, from the ACC era) and `OPENTEXT_MASTER_TABLE.md` /
`OPENTEXT_BASELINE.md` (the open-text anchors, still valid).

---

## PART 3 — The research story, in the order it happened (≈2–3 hr) 🟡

Read the diaries as a diary. You do **not** need every line — the pointers say which sections carry the
plot. The July files are where the *current* method was found; the June files are its prehistory.

**Step 7 — `progress/progress_July_04.md` → `progress_July_08.md`** (5 files; July 7 is 485 lines, the
biggest single day).
*Learn:* how the final method was actually found. The load-bearing moments: **July 4** — the gate
bake-off had been ranked by the *wrong proxy* (cascade quality tracks recoverability-AUROC, r ≈ +0.65,
not detection-AUROC, r ≈ −0.21), and the deterministic format-aware router works; **July 6** — the
execution marathon (~15 experiments: the adaptive-N controller, cross-model pooling, diverse generation,
the real pairwise verifier overturning its own simulation); **July 7** — *"we have been comparing against
the wrong 'strong' model"*, the measurement that reversed July 6's verdict and produced the final
headline; **July 8** — the MMMU contamination audit, the honest-baseline experiment, the last estimate
becoming measured, and the paper rewrite.
*Why:* this is where the decisions were made and where the reasoning is visible.

**Step 8 — `progress/progress_July_01-02.md`, `progress_July_03.md`, `progress_July_05.md`.**
*Learn:* the **faithful protocol** turning point (adopting MedEvalKit; Lingshu-32B MMMU 0.633 vs the
paper's 62.3, exact), the three-family matrix, and the infrastructure wall that produced the 68-idea
backlog. *Why:* everything numeric in the project is anchored here.

**Step 9 — `progress/progress_June_17.md` (§0 → §6 + Appendix A), then `June_20-22`, `June_24`,
`June_25-26`.** 🟢 for June 25-26.
*Learn:* the prehistory, and the four turning points that shaped everything after —
**June 17**: the gate is signal-limited *and* the strong leg was running in the wrong mode (reasoning
over-thinks perception) → the compute-configuration cascade; **June 20–22**: cross-family validation and
a cost-accounting bug that retired three earlier headline forms; **June 24**: the open-ended
ceiling-break (routing AUROC 0.6 → 0.87 — the multiple-choice ceiling was a *benchmark artifact*);
**June 25–26**: a trained verifier breaking the luck floor, the only thing in the project that did.
*Why:* the "Detailed narrative" section of `June_25-26.md` is the clearest example in the repo of how
research decisions were actually made — question → method → data → result → *why we moved next*.
*Note:* `June_27-28.md` and `June_29-30.md` are honestly labelled **reconstructions** (written
2026-07-02). There is **no diary for 2026-07-09** (the run that produced the current headline CI) or for
2026-07-10 → 07-27.

---

## PART 4 — The code (≈1 hr) 🔴

**Step 10 — `STRUCTURE.md`**, top-level map + the `src/` section you care about.
*Why:* the index for ~199 scripts. Use it as a lookup, not a read-through.

**Step 11 — the entry-point scripts, in this order.** Note there are **two disjoint code universes**: the
June multiple-choice era rooted at `harness.py` (imported by 39 files), and the July Lingshu chain (~10
files) which imports `harness.py` **zero** times.

*The live July chain (this is the current method):*
1. `src/cascade_methods/method_final.py` — **THE** single reproducible paper method.
2. `src/cascade_methods/paper_baselines.py` — the baseline table and every paired bootstrap CI.
3. `src/cascade_methods/integrated_method.py` — the format-aware cascade + the cost constants.
4. `src/cascade_methods/beat32b_fusion.py` / `beat32b_more.py` — the PMC fusion (`test_2.csv`), the certified veto, the
   learning-to-defer rule.
5. `src/cascade_methods/integrated_pandora.py` / `pandora_controller.py` — the adaptive-N controller.
6. `src/cascade_methods/f8_mode_vsthink_ci.py` — **produces the paper's headline CI.**
7. `src/cascade_methods/opentext_32b_think_full.py`, `method_final_mmmu_corrected.py` — the two 2026-07-08
   corrections.

*The June era + the trained methods:*
8. `src/cascade_methods/harness.py` — the shared offline evaluation harness.
9. `src/cascade_methods/acc.py` — the 3-tier compute-configuration cascade.
10. `src/training_methods/run_lora_verifier_open.py` — the **trained free-text verifier**.
11. `src/training_methods/run_lora_box_verifier.py` — the structured box-verifier.
12. `src/labeling/run_openvqa.py` + `run_judge.py` — how open-text samples are generated and scored.

**Step 12 — retrospective §4.6 ("Reproduction path") and §9.6 ("Code health").**
*Learn:* the exact script → artifact behind every headline number; and the honest state of the codebase
(three copies of the cost constants, `auroc` defined 12 times, ~66 orphan scripts, and the reason: fixes
were made in *new* files rather than propagated into the old ones).
*Why:* this is the bridge from "a result" to "the code that produced it".

---

## PART 5 — Deep-dive appendices (reference only) 🔴

Read the one that matches what you are digging into. **Note the folder split** introduced 2026-07-02 —
`docs/current/` is canonical, `docs/archive_mcq/` is the preserved superseded record.

| If you want the full detail on… | read |
|---|---|
| **anything at all, first** | `docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md` |
| **how PMC-VQA is built, how thinly it is validated, which splits exist and which we used** | `docs/current/PMCVQA_PROVENANCE_2026-07-30.md` — **read before quoting any PMC-VQA number** |
| the final method, fully specified | `docs/current/METHOD_FINAL_2026-07.md` |
| the full results ledger (76 KB) | `docs/current/RESEARCH_RESULTS_2026-07.md` |
| the technical walkthrough | `docs/current/TECHNICAL_REPORT_2026-07.md` |
| the open-text baselines & master table | `docs/archive_2026-07/OPENTEXT_BASELINE.md`, `docs/archive_2026-07/OPENTEXT_MASTER_TABLE.md` |
| load-bearing facts traced to source | `docs/archive_2026-07/VERIFIED_FACTS.md` |
| the OmniMedVQA infra fallback | `docs/current/OMNIMED_FALLBACK.md` |
| the running experiment log (58 KB) | `docs/archive_2026-07/UNIFIED_METHOD_EXPERIMENTS.md` |
| the ACC method (historical) | `docs/archive_2026-07/METHOD_ACC.md` |
| the equations / cost model | `docs/archive_2026-07/METHOD_MATH.md` (or `progress/progress_June_17.md` Appendix B) |
| the trained free-text verifier | `docs/archive_mcq/TRAINED_VERIFIER_RESULT.md` |
| the grounding box-verifier (SLAKE + MS-CXR) | `docs/archive_mcq/BOX_VERIFIER_RESULT.md` |
| why training-free selection is luck-floored | `docs/archive_mcq/OPENENDED_SELECTION_LUCKFLOOR.md` |
| why the 32B's edge is capacity, not fixable cheaply | `docs/archive_mcq/RECOVERABILITY_IS_CAPACITY_BOUND.md` |
| whether RAG could help (it can't) | `docs/archive_mcq/KNOWLEDGE_AUGMENTATION_FEASIBILITY.md` |
| a consolidated list of every MCQ-era result | `docs/archive_mcq/FINDINGS.md`, `MASTER_TABLES.md`, `FULL_RECORD.md` |
| the cross-family / 2-size validation | `docs/archive_mcq/2SIZE_VALIDATION.md` |
| the visual-stability rescue + ACC-v3/v4 | `docs/archive_mcq/NOVEL_METHOD_VISUAL_STABILITY.md`, `ACCV3_V4_AND_NOVELTY.md` |
| **~90 negative results, grouped by the principle that killed each** | retrospective §6 |
| the 68 cross-field idea backlog | `results/cascade_methods/METHOD_IDEAS_BACKLOG.md` ⚠️ **no per-idea status field — several entries have already been run and failed; cross-check retrospective §6 before proposing any of them** |
| the numeric audit trail | `INCONSISTENCIES.md` (2026-06-27, ACC era) + retrospective §10 |

`results/cascade_methods/README.md` indexes these too.

---

## The shortest honest summary of the whole project

If you internalize just this, you understand the project:

1. **The goal was efficiency; the result is a characterization.** A format-aware adaptive cascade
   (Lingshu-7B → Lingshu-32B) **matches the strong model at roughly half the compute**, with a
   significant accuracy gain on **two specific cells** — open-ended free text and PMC-VQA (`test_2.csv`) — and a
   measured account of why the remaining cells are unwinnable. That last part is the stronger
   contribution, and unlike +0.0245 it survives an honest re-costing.
2. **Every genuine positive came from changing *what* is being routed, never from improving the
   router.** Routing over **compute configurations** instead of models; **training** a small verifier
   instead of reading a frozen one; routing over **answer format** instead of using one unified gate;
   and finally **re-pricing the baseline** so the comparison is against the model a user would deploy.
3. **You cannot build a better training-free gate.** "Will the strong model fix *this* error?" is
   ~0.5–0.6 AUROC from anything cheap; **16 independent mechanisms** hit that wall. A companion wall
   bounds selection at 74–82% of oracle (**13 attempts**), and the *coverage* wall is 4.5× larger than
   the selection wall.
4. **Answer format, not model quality, decides whether routing signals work at all** — AUROC ~0.6 on
   4-option multiple choice vs ~0.87 on free text. Do not carry a "confidence is saturated" conclusion
   from a multiple-choice benchmark into a generative deployment.
5. **The habit that makes the record trustworthy is that it publishes its own refutations** — 19
   retracted or downgraded claims and 14 numeric corrections, logged in retrospective §10, including
   the ones that cost the project a headline.
