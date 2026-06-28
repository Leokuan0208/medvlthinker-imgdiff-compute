# Reading Guide — how to understand this whole project, in order

> **Purpose.** There are a lot of files here (a project-context file, 4 dated progress logs, a session
> report, a paper draft, 23 result writeups, 140 scripts). This guide is the **reading order**: exactly which
> *section of which file* to read, what you'll learn from it, and why it matters — so you can go from zero to
> understanding the whole research process and code without getting lost.
>
> **How the documents relate (mental model):**
> - `CLAUDE.md` = the **briefing** (what the project is + the rules). Start here.
> - `progress_June_*.md` = the **diary** (what happened, chronologically, with the messy detail).
> - the **Detailed narrative** section of `progress_June_25-26.md` = the **best narrative** of the most recent + most important phase.
> - `paper/cvgip2026_draft.md` = the **polished final story** (what we'd actually publish).
> - `STRUCTURE.md` = the **map of the code** (every file, one line each).
> - `results/cascade_methods/*.md` = the **deep-dive appendices** (read only when you want full detail on one result).
>
> **Three reading depths — pick one:**
> - 🟢 **30-minute skim** → Steps 1, 2, 9, 10 only. (You'll get the whole arc + the headline result.)
> - 🟡 **Half-day, full understanding** → Steps 1–14 in order. (Recommended.)
> - 🔴 **Everything incl. code & proofs** → all steps, including Part 4 and Part 5.
>
> Time estimates assume you are reading carefully, not speed-reading.

---

## PART 1 — Orientation: what is this project? (≈20 min) 🟢

**Step 1 — `CLAUDE.md`, §1 "The project in one paragraph" + §3 "Key terms (glossary)".**
*Learn:* what a medical VLM cascade is, the 7B→32B idea, the confidence gate, and every term you'll see
later (margin, τ, escalation, cap320, think/no-think, oracle gap, luck floor). *Why:* nothing else makes
sense without these terms. **If you read only one thing first, read this.**

**Step 2 — `CLAUDE.md`, §0 "Current status".**
*Learn:* where the project stands and its **two headline outcomes** — (1) the deployed margin gate is
essentially optimal among training-free gates, and (2) the genuine structural win is the **ACC** cascade.
This is the 1-page executive summary of the older half of the project. *Why:* it frames everything in the
progress logs.

**Step 3 — `CLAUDE.md`, §2 "How the project got to its current state".**
*Learn:* the project pivoted ~4 times (token-pruning → image-difficulty → single-model routing →
cross-model cascade). *Why:* explains why there are "dead" files in `archive/` and why the method is what it
is — negative results are part of the story here.

---

## PART 2 — The research story, in the order it happened (≈1.5–2 hr) 🟡

Read the progress logs as a diary. You do **not** need every line — the pointers below tell you which
sections carry the real plot.

**Step 4 — `progress_June_17.md`, §0 → §6** (skim §1; read §3 and §6 closely).
*Learn:* how the cascade loop started and found the **first genuine win**. The two turning points:
§3 ("the strong leg was running in the wrong mode" — the real lever) and §6 ("ACC: the genuine win").
*Why:* ACC is the project's main method; this is its origin story.

**Step 5 — `progress_June_17.md`, Appendix A (§A.2–A.4).**
*Learn:* how ACC was benchmarked against named SOTA cascade methods (FrugalGPT, CP-Router, AutoMix) and
the proof that **the win is the structure, not the gate** (§A.4). *Why:* this is the comparative evidence
behind outcome (1) in Step 2. (Appendix B is the math — skip unless you want equations; see Step 17.)

**Step 6 — `progress_June_20-22.md`, read "Standing conclusions" first, then skim the dated entries.**
*Learn:* ACC was validated across 5 model families and 3 architectures; a **cost-accounting bug was found
and fixed** (the honest numbers); a novel-method search returned a clean *negative*. *Why:* shows the
rigor/breadth pass and why the final numbers are trustworthy.

**Step 7 — `progress_June_24.md`, phase 3 (the headline) + phases 1–2.**
*Learn:* the **open-ended ceiling-break** — the single most important conceptual result of the middle
period: routing looks impossible on multiple-choice (AUROC ~0.6) only because MCQ is *discrete*; on
free-text the same signal jumps to ~0.87. Phases 1–2 cover the visual-stability rescue and ACC-v3/v4.
*Why:* this reframes the whole "routing is saturated" story and sets up the final phase.

**Step 8 — `progress_June_25-26.md` (all of it — only 70 lines).**
*Learn:* the most recent phase in brief — **Phase A** (the "luck floor": every training-free trick is
bounded) and **Phase B** (the trained verifier that finally breaks it). *Why:* this is the latest and
arguably best result; the next step explains it in depth.

**Step 9 — the **Detailed narrative** section of `progress_June_25-26.md` (read the whole file — it's the best single document).** 🟢
*Learn:* the deep, reasoned narrative of the final phase, written as *question → method → data → result →
**why we made that move***. It covers the seven luck-floor experiments and the three trained-verifier
positives (free-text answers + SLAKE boxes + the real MS-CXR chest-X-ray benchmark), with the bootstrap
confidence intervals. *Why:* if you want to understand *how research decisions were actually made* here,
this is the clearest example. Pairs with paper §5.9–§5.10 (Step 13).

---

## PART 3 — The polished story: the paper (≈1 hr) 🟡

Now read the paper, which synthesizes everything above into the publishable narrative.

*(The paper was restructured 2026-06-27 into a two-positive arc: ACC for efficiency + the trained verifier for accuracy, unified by the luck floor; the old §5.8 abstention section was removed.)*

**Step 10 — `paper/cvgip2026_draft.md`, Abstract + §1 Introduction (the "Contributions" list (i)–(v)).** 🟢
*Learn:* the entire project in ~2 pages and the five contributions. *Why:* the cleanest top-down summary.

**Step 11 — paper §3 "Setup" + §4 "Method I: the Adaptive-Compute Cascade (ACC)".**
*Learn:* the metrics/definitions (incl. the cost equation and the agreement-gate math) and the 3-tier ACC. *Why:* the formal version of Step 4.

**Step 12 — paper §5.1 (ACC results) + §5.2 (the luck floor) + §5.3 (open-ended ceiling-break).**
*Learn:* the efficiency win (−80% latency etc.), why no training-free gate/selector beats the baseline, and why open-ended evaluation matters. *Why:* the evidence for the first positive + the connective negative.

**Step 13 — paper §6 "Method II: a Trained Outcome Verifier".** 🟢
*Learn:* the verifier math (score, loss, best-of-N), the free-text result (49%), the box result (SLAKE + real MS-CXR), the AUROC-0.924 discrimination, the scaling curve, and the compute-beats-parameters Pareto. *Why:* the second, most novel positive; mirrors Steps 8–9.

**Step 14 — paper §7 "Discussion & Limitations" + §8 "Conclusion".**
*Learn:* honest scope (what is/ isn't novel), and the one-paragraph takeaway. *Why:* the boundaries of every claim.

---

## PART 4 — The code (≈1 hr) 🔴

**Step 15 — `STRUCTURE.md`, top-level map + the `src/` section that matches what you care about.**
*Learn:* where every file lives and what it does. *Why:* the index for all 140 scripts. Don't read it
end-to-end; use it as a lookup.

**Step 16 — Read these *entry-point* scripts (in this order) to see the method in code:**
1. `src/cascade_methods/harness.py` — the shared evaluation harness everything imports.
2. `src/cascade_methods/acc.py` — the core ACC cascade (the main method).
3. `src/training_methods/run_lora_verifier_open.py` — the trained free-text verifier (the §5.10 positive).
4. `src/training_methods/run_lora_box_verifier.py` — the structured box-verifier (grounding positive).
5. `src/labeling/run_openvqa.py` + `src/labeling/run_judge.py` — how samples are generated and scored.

*Why:* these ~5 files are the spine; the other ~135 are experiments/variants around them.

**Step 17 — paper "Reproducibility index" (near the end of the paper).**
*Learn:* the exact script + checkpoint behind **every** number in the paper. *Why:* this is the bridge from
"a result" to "the code that produced it" — use it whenever you want to re-run or verify something.

---

## PART 5 — Deep-dive appendices (reference only — read a file when you want full detail) 🔴

These live in `results/cascade_methods/`. Read the one that matches the result you're digging into:

| If you want the full detail on… | read |
|---|---|
| the ACC method, fully specified | `METHOD_ACC.md` |
| the equations / cost model behind ACC | `METHOD_MATH.md` (or `progress_June_17.md` Appendix B) |
| the trained free-text verifier | `TRAINED_VERIFIER_RESULT.md` |
| the grounding box-verifier (SLAKE + MS-CXR) | `BOX_VERIFIER_RESULT.md` |
| why training-free selection is luck-floored | `OPENENDED_SELECTION_LUCKFLOOR.md` |
| why the 32B's edge is capacity (not fixable cheaply) | `RECOVERABILITY_IS_CAPACITY_BOUND.md` |
| whether RAG could help | `KNOWLEDGE_AUGMENTATION_FEASIBILITY.md` |
| a consolidated list of every result | `FINDINGS.md`, `MASTER_TABLES.md`, `FULL_RECORD.md` |
| the cross-family / 2-size validation | `2SIZE_VALIDATION.md` |
| the visual-stability rescue + ACC-v3/v4 | `NOVEL_METHOD_VISUAL_STABILITY.md`, `ACCV3_V4_AND_NOVELTY.md` |

`results/cascade_methods/README.md` indexes these too.

---

## The shortest honest summary of the whole project

If you internalize just this, you understand the project:

1. **The goal is efficiency, not accuracy** — match the big model's accuracy for less compute.
2. **ACC is the method that works**: route each question across *compute configurations* of the same models
   (cheap 7B → big model's *fast* no-think mode → big model's slow think mode), firing the expensive tier
   only when needed. Big measured latency/energy/FLOPs savings at equal accuracy. (Steps 4, 11, 12.)
3. **You cannot build a better training-free *gate*** — every confidence/consistency/agreement signal hits
   the same "recoverability ceiling." This is a thoroughly-proven negative. (Steps 5, 7, 8, 12, 13.)
4. **But *training* breaks the ceiling**: a small trained verifier that picks the best of N sampled outputs
   recovers 40–77% of the unharvested "oracle gap" — for both text answers and bounding boxes, on a real
   chest-X-ray benchmark, statistically significant. This is the newest, strongest positive. (Steps 8, 9, 13.)
