# CLAUDE.md — Project Context for `medvlthinker-imgdiff-compute`

> ## ⚠️ CRITICAL RULES — READ FIRST, ALWAYS APPLY ⚠️
>
> **These seven rules override default behavior and apply to every task in this repo. Do not skip them.**
>
> 1. **Ask, don't assume.** If something is unclear, ask before writing a single line. Never make silent
>    assumptions about intent, architecture, or requirements.
> 2. **Simplest solution first.** Always implement the simplest thing that could work. Do not add
>    abstractions or flexibility that weren't explicitly requested.
> 3. **Don't touch unrelated code.** If a file or function is not directly part of the current task, do not
>    modify it, even if you think it could be improved.
> 4. **Flag uncertainty explicitly.** If you are not confident about an approach or technical detail, say so
>    before proceeding. Confidence without certainty causes more damage than admitting a gap.
> 5. **Suggest better ways.** I'm always open to ideas on better ways to do things — don't hesitate to
>    suggest a better approach, especially one with long-lasting impact over a tactical change.
> 6. **⛔ ABSTENTION IS PERMANENTLY FORBIDDEN** (made permanent 2026-07-07, `progress/progress_July_07.md`
>    §15). Never research, build, test, or propose abstention / reject-option / defer-to-human /
>    selective-prediction-as-a-method. **The method must always return an answer.** Reusing that
>    literature's *math* inside an answer-producing mechanism is fine (the "certified veto" keeps the
>    cheap model's answer — that is not abstention). Files from the June abstention work still exist in
>    `src/cascade_methods/` and `docs/archive_mcq/SELECTIVE_ABSTENTION.md`; they are **historical record
>    only** and must not be revived as a direction.
> 7. **No fabricated numbers — ever.** Every figure in any doc, paper, slide or site must come verbatim
>    from real experimental output, and must name the file it came from. If a number is needed and not
>    available, recompute it from checkpoints or say "not measured". Never invent, never interpolate
>    silently, never relabel an estimate as a measurement. (This rule was already in §7; it is promoted
>    here because it was violated by *labelling* — three docs printed "Baselines (measured): 0.5632"
>    when that value's open-text cells were estimates.)

> **What this file is.** This is a context/briefing file for Claude Code. When Claude Code
> opens this folder it reads `CLAUDE.md` automatically. The purpose of this file is to (1)
> explain what this research project is, (2) describe what is currently in the folder and what
> each piece does, (3) propose a cleaner folder structure, and (4) list the rules that must be
> followed so the reorganization does not break anything.
>
> **History — the original task was a *cleanup*** (move/rename files into `src/<stage>/`). That is
> **done**. The **rules & landmines** (§6/§7/§8) still apply to any file moves. The active work is a
> research program, summarized in §0.
>
> **⚠️ This file is a briefing, not a numbers source.** For any figure, read
> `results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md` (§4 for the results, §10
> for the corrections log). §0/§1 below give the shape of the project; the retrospective owns the values.
>
> **About the author.** The researcher (Leo / Li-Wen Kuan) is an engineering student who is
> relatively new to computer science. When you explain a command or a plan, be explicit and
> step-by-step; don't assume familiarity with shell, git internals, or Python packaging.

---

## 0. Current status (2026-07-29) — read this first

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

Source: `artifacts/f8_mode_vsthink_ci.json` (2026-07-09) + `artifacts/opentext_32b_think_full.json`.
Baseline **always-32B-with-reasoning = 0.5591 (measured)**; always-32B-direct 0.5729; always-7B 0.5549.

| setting | accuracy | compute (× one 32B pass) | latency (par.) | Δ vs 32B-reasoning, 95% CI |
|---|---|---|---|---|
| **compute-lean** | 0.5741 | **0.49×** | 469 ms | **+0.0150 [+0.0107, +0.0192]** SIG |
| **accuracy-max** (certified veto + learn-to-defer) | 0.5836 | **0.93×** | 731 ms | **+0.0245 [+0.0216, +0.0274]** SIG ← *canonical* |
| accuracy-max⁺ (fusion variant) | 0.5862 | 1.25× | 668 ms | +0.0271 [+0.0237, +0.0305] SIG |

**+0.0245 at 0.93× is the one canonical number.** It is the only value that is simultaneously the
FLOP-negative deployed configuration, MMMU-clean, measured (not estimated), and CI-certified. Four
older values circulate for the *same* operating point (+0.0212 / +0.0207 / +0.0238 / +0.0275) — they
differ by lever, pool, and estimated-vs-measured open cells. **Before quoting any `+0.02xx` value,
read the decode table in retrospective §10.3.**

### The three findings that generalize (retrospective §5)

1. **Reasoning hurts perception; on reasoning-heavy benchmarks it helps *some* families**
   *(re-derived 2026-07-29 — see the correction note below)*. On prompt- and resolution-matched arms,
   thinking is strictly worse than answering directly in **17/20** perception cells across 5 medical
   families — **14/20** with 95% CIs excluding zero, pooled **−0.0401 [−0.0456, −0.0347]** over
   **30,250** paired samples, **19/20** no better than +0.02 — and it reproduces at the same strength on
   arms that differ by nothing but the reasoning instruction. On reasoning-heavy benchmarks CoT helps
   MedVLThinker-32B / MedGemma-27B / InternVL3-38B but **not** Lingshu-32B or QoQ-Med-VL-32B: the
   reasoning-side gain is **model-dependent, not universal**. Cost: 15–49× latency where a real think
   mode exists (MedVLThinker 49×, MedGemma 45×, QoQ 43×, Chiron 15×).
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
>   think-vs-direct delta (Δ = −0.154) until it lands.
> - **`MedEvalKit/` has two local uncommitted edits** (`utils/question_formats.py:11`,
>   `utils/MMMU/data_utils.py:158`, mtime 2026-07-02) that added a reasoning trigger but **deleted** the
>   answer-format clause the direct arm still carries. Pre-edit `eval_results_*_think` dumps are invalid
>   as reasoning evidence; post-edit `*_reason` dumps reason but are format-unmatched.
>   **`MedEvalKit/` is a protected dependency — do not modify it**; the revert/repair is Leo's call.
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
- **The honest framing (§8.5)** is *"matches the strong model at roughly half the compute, with a
  significant accuracy gain on two specific cells — open-ended free text and PMC-VQA (`test_2.csv`) —
  plus a measured characterization of why the remaining cells are unwinnable."* Not a suite-wide
  accuracy advantage.
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
  charged at reasoning cost (hole 1); 89% of the headline delta comes from 2 of 8 cells (hole 2); the
  open-text verifier scores items it was trained on, ~31% inflation (hole 4). These are open, not fixed.
- **Preservation risk:** the last commit is `8cdefef` (2026-07-02). The entire Lingshu/July chain — the
  IEEE paper, the July diaries, `paper/figs_final/`, the July-27 deck, and every headline script under
  `src/cascade_methods/` — is **untracked**. Committing is the top-priority task (§8.1 item 2).
- The earlier MedVLThinker work (margin gate τ=0.426, the 3-tier ACC cascade) is **historical**, not the
  live method — see §1 below.

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

## 2. How the project got to its current state (why "dead" files exist)

This project **pivoted several times**. That history matters because the folder still contains
artifacts from abandoned directions. **Do not delete these** during cleanup — archive them.
They are the record of *why* the final method is what it is (negative results are a core part of
the paper). The arc, oldest to newest:

1. **Question-aware visual token pruning** (the original project name). Idea: drop unimportant
   image tokens to save compute. → did not yield a usable accuracy cliff.
2. **Image-difficulty-driven adaptive compute** (the `complexity` / `difficulty` / `lesion`
   files). Idea: spend more compute on "hard" images. → killed (correlations were the wrong
   sign / near zero).
3. **Single-model routing** across "axes" like think-vs-no-think and retrieval (RAG) — the
   `gate_probe`, `gate_rag`, `retrieve.py`, and `MedRAG/` artifacts. Idea: route *within one
   model*. → **definitively killed**: a "luck-floor" audit showed the oracle was ~29σ below
   the random-allocation floor, i.e. one model's confidence signals are mutually redundant and
   carry no routable signal.
4. **Cross-model 7B→32B cascade** (2026-06-17). The structural fix was to route *between two
   different models*, not within one. Produced the τ=0.426 margin gate (§1, historical).
5. **Compute-configuration cascade — "ACC"** (2026-06-17/22). The gate turned out to be
   un-improvable, but the *strong leg was running in the wrong mode*: reasoning **over-thinks**
   perception VQA. Route over compute configurations instead of models. Validated across 5 families
   (re-derived on prompt-matched arms 2026-07-29: **17/20** perception cells — see §0).
6. **Open-ended answers + a trained verifier** (2026-06-24/26). Every routing signal is degenerate on
   4-option MCQ (AUROC ~0.6) and works on free text (~0.87). Every *training-free* selector sits at the
   random-pick floor; a small **trained** LoRA verifier broke that floor — the only thing in the whole
   project that did.
7. **The faithful protocol** (2026-07-01). Abandoned the internal harness for **MedEvalKit**, which
   reproduces Lingshu's published numbers exactly (Lingshu-32B MMMU 0.633 vs paper 62.3). Primary
   family switched to **Lingshu**. Every claim is now anchored to a public baseline.
8. **The format-aware adaptive cascade + the honest baseline** (2026-07-04/08) — **the live work.**
   One router over answer format; and the realization that the project had been comparing itself
   against the *cheapest* way of running the big model rather than the way a user would deploy it.

So in the repo you will find four layers: **live July/Lingshu code** (`src/cascade_methods/` — the
~10-file headline chain: `method_final.py`, `paper_baselines.py`, `integrated_method.py`,
`beat32b_*.py`, `opentext_32b_think_full.py`, `f8_mode_vsthink_ci.py`), **June MCQ-era code** (rooted at
`src/cascade_methods/harness.py`, imported by 39 files — superseded as a source of headline numbers but
still the reproducibility anchor for `docs/archive_mcq/`), **dependency repos** (`MedRAG/`,
`MedVLThinker/`, `MedEvalKit/` — never move), and **archived dead directions** (`archive/`).

---

## 3. Key terms (glossary)

> A **codename glossary** (ACC, VADR, FALC, CASP/CCPS, F3/F8/F10, Pandora, G1–G8, H1–H9, Variant A/B)
> is in retrospective §9.5 — you need it to read the older documents.

- **VLM** — vision-language model; image + text in, text out.
- **VQA** — visual question answering. **Two formats**, and the distinction is load-bearing:
  *multiple-choice* (pick A/B/C/D) and *open-ended* (free text). Routing signals behave completely
  differently between them (retrospective §5.2).
- **7B / 32B** — model sizes (billions of parameters). 32B is the big, accurate, expensive one.
- **Cascade** — run the cheap model first; only call the expensive model on hard cases.
- **Gate / router** — the small decision rule that decides "escalate to 32B or not."
- **τ (tau)** — the confidence threshold the gate uses. Below threshold confidence → escalate.
- **Margin** — gap between the model's top-1 and top-2 answer probabilities; the confidence
  signal the gate thresholds.
- **Escalation rate** — fraction of questions handed up to the 32B.
- **cap320 / cap640 / fullres** — image-resolution budgets (a cap on pixels via `max_pixels`).
  Lower cap = fewer image tokens = cheaper. "cap320" is the chosen operating point.
- **Prefill-inclusive FLOPs** — honest compute accounting that includes the cost of reading the
  prompt+image, not just generating the answer. (An earlier decode-only estimate was too rosy.)
- **think / no-think** (also written *reasoning* / *direct*) — these models can emit a
  `<think>...</think>` reasoning trace or answer directly. **In the live Lingshu method BOTH legs run
  direct**; 32B-with-reasoning is only the *baseline*. (In the June MedVLThinker era the 32B leg ran
  think — that is why old docs say "the 32B runs think".)
- **RAG / retrieval** — pulling in external text (e.g. textbooks) to help answer. Part of a
  *killed* direction; `MedRAG/` and `retrieve.py` are its leftovers. Ruled out with zero GPU time: the
  32B fixes genuinely-unknown errors *equally* on knowledge (38%) and perception (36%) questions ⇒
  the deficit is capacity, not retrievable knowledge.
- **HistGBM / Conformal / FBE / CP-Router** — fancier gate alternatives that were tested and
  **lost** to the simple margin threshold. Their scripts are real results, kept as ablations.
- **HF vs vLLM** — two ways to run the models. **vLLM** is fast (~35× speedup) but hides true
  GPU memory use. **HF** (HuggingFace transformers) is slower but measures real VRAM, so it's
  used for the live cascade memory/energy measurement.
- **Verifier** — a small LoRA-fine-tuned model scoring `P(correct | image, question, candidate)`; used
  to pick the best of N sampled open-text answers. `ckpts/train/lora_verifier_pooled4`.
- **best-of-N / oracle-of-N** — sample N answers and keep one / the best possible one. The gap between
  them is the **selection wall**; the fraction of questions with *no* correct answer in the pool is the
  **coverage wall**.
- **Recoverability** — "will the strong model fix *this* cheap-model error?" ~0.6 AUROC ceiling; the
  binding limit of every gate. **Distinct from *detection*** ("is the cheap model wrong?", ~0.87 on open
  text). Gates must be ranked by recoverability — ranking by detection is a known, costly mistake here.
- **Luck floor** — the recurring shape of this project's negatives: a large oracle gap that no
  frozen-model signal can harvest, sometimes provably below a random-allocation permutation floor.
- **Guardrail** — "never worse than always-cheap on any single benchmark". Pooled wins routinely hide
  per-benchmark damage; the guardrail is what killed several of them.
- **Variant A / Variant B** — MMMU escalated / **MMMU excluded** (Variant B, n=42,224, is what the paper
  reports).
- **Certified veto** — keep the cheap answer where a Wilson lower bound on its precision beats the strong
  model's accuracy, so the strong model is never run there. **It is not abstention** — it answers.

---

## 4. Repository inventory — **see `STRUCTURE.md`**

`STRUCTURE.md` is the live per-file index (every directory + a one-line purpose for every script);
`results/cascade_methods/README.md` indexes the writeups and artifacts. This section used to duplicate
both and drifted badly, so it is now a pointer plus the handful of facts the landmines in §7 depend on.

**Always verify against the live tree before acting** — run a real listing and reconcile
(procedure in §6). Treat any inventory in any doc as "expected, confirm on disk".

### 4.1 The parts that matter for safety

- **`ckpts/` (gitignored, resumable).** Scripts write per-sample JSONL and resume from the last
  completed line. **Moving a checkpoint folder while a run could resume into it orphans that resume
  state.** As of 2026-07-29 it holds, besides the June MCQ dirs (`gate_7b_*`, `gate_32b*`, `pmctrain/`,
  `_legacy/`, `router_margin.pkl`, `token_cache.json`, `rt_cascade_cap320.jsonl`):
  `train/` (LoRA adapters, incl. `lora_verifier_pooled4` — the deployed verifier), `openvqa/`
  (open-text generations + judge labels), `ground/` (grounding/box outputs), `pairwise/` +
  `pairwise_diverse/`, `acc_gen/` (5-family + peer-architecture dumps), `peer/`,
  `gate_lingshu{7b,32b}_mcq/`, `mcq_gen_verify/`.
- **Checkpoint JSONL schema** (so you can read them without guessing): `idx, gold, pred, ok, parse_ok,
  opt_logprobs (letter→logprob dict), gen_tokens, latency_s, raw_output`. The live cascade JSONL uses
  `idx, dataset, escalate, ok, final` (+ in some versions `pred7, pred32, gold, margin, latency_s,
  energy_j, gen7, gen32`).
- **Shard-tag convention.** Single-shard runs carry **no `_sKofN` suffix**; only genuinely sharded runs
  are tagged (e.g. `gate_7b_think/` keeps `_s0of2`/`_s1of2`). Labelers write the tag only when `N>1`;
  every reader treats `(?:_s\d+of\d+)?` as optional, so both forms load and shards merge by `idx`.
- **Faithful-eval outputs live in `MedEvalKit/eval_results_*/`** — which is **gitignored vendor
  territory**. Every faithful MCQ number in the paper is read from there. Do not clean that directory.
- **`results/cascade_methods/artifacts/`** holds ~107 numeric `.json` outputs (gitignored,
  regeneratable). The headline chain is `method_final.json`, `method_final_v2.json`,
  `method_final_mmmu_corrected.json`, `paper_baselines.json`, `opentext_32b_think_full.json`,
  **`f8_mode_vsthink_ci.json`** (the canonical headline CI).
- **`archive/`, `docs/archive_mcq/`, `_legacy/`** are the record of negative results. Move, never delete.

### 4.2 Dead directions still on disk (keep, don't revive)

`archive/image-difficulty/`, `archive/old-gate-scripts/`, `archive/single-model-routing/` (incl. the
n≈500 RAG-axes grid), `src/legacy_retrieval/retrieve.py` + `MedRAG/` (killed RAG direction), and the
six **abstention** scripts still sitting in `src/cascade_methods/` (`selective_abstain.py`,
`abstain_calibration.py`, `triage_3way.py`, `deferral_curve.py`, `methods_deferral.py`,
`lingshu_deferral_apgr.py`) — historical record only, see CRITICAL RULE 6.

---

## 5. Repository structure — **see `STRUCTURE.md`**

The 2026-06-16 cleanup moved every active script out of the old flat `scripts/` into `src/<stage>/`.
That is done and `scripts/` no longer exists. The full tree used to be duplicated here; it now lives in
one place, **`STRUCTURE.md`**, which is kept current. Top-level shape, for orientation only:

```
src/{labeling,sweep,gate,cascade,cascade_methods,training_methods,analysis,reporting,data_prep,legacy_retrieval}
runners/            38 shell launchers (each cd's to the repo root)
progress/           13 dated daily diaries (June 17 -> July 8) — the primary narrative record
paper/              the IEEE deliverable + build scripts + figs_final/;  paper/archive/ = superseded drafts
meetings/           dated .html decks (the 2026-07-27 one is the best summary in the repo)
docx/               generated Word exports
results/cascade_methods/{docs/current,docs/archive_mcq,artifacts,claude_judge}
ckpts/ logs/ data/ feats*/          gitignored data & checkpoints
archive/            killed directions
MedRAG/ MedVLThinker/ MedEvalKit/   dependency repos — DO NOT MOVE/RENAME
```

**Naming conventions:** lowercase, `_`-separated, names that describe the *role* and read clearly
without the folder for context. Archived files kept their original names so they still match the
`logs/*.log` files keyed to them. The terse `src/cascade_methods/` names are **deliberately not
renamed** (the paper's repro index, the progress logs and ~36 imports reference them); `STRUCTURE.md`
is the descriptive index for them instead.

## 6. Safe cleanup procedure (follow this order)

The previous reorganizations on this project followed a strict "inspect first, move second"
discipline because a wrong move can orphan a resumable checkpoint or break an import. Replicate
that discipline:

1. **Inspect the real tree.** Run a listing and compare it to §4. Do not trust this file over
   what's on disk.
   ```bash
   cd ~/medvlthinker-imgdiff-compute
   { command -v tree >/dev/null 2>&1 && tree -L 2 -a -I '.git|__pycache__|*.pyc'; } \
     || find . -maxdepth 2 -not -path './.git/*' -not -path '*/__pycache__/*' -print | sort
   ```

2. **Find cross-references before renaming anything.** Renaming a file is only safe if every
   place that mentions it is updated too. Grep for imports and hard-coded paths:
   ```bash
   # which scripts import which (so a rename doesn't break an import)
   grep -rn "import\|from " --include=*.py src/ | grep -vi "^.*#"
   # hard-coded paths to weights / datasets / checkpoint dirs
   grep -rn "/data/dan\|ckpts\|CKPT_DIR\|gate_7b\|gate_32b\|router_margin" --include=*.py src/
   ```
   Anything that shows up here must be updated in the same change as the move/rename.

3. **Propose the full move/rename plan to Leo as a list, and wait for confirmation.** Do **not**
   write a blind `mv` script. Present: "these N files move here, these get renamed to X, here's
   every reference I'll update." Renaming files that other files import or that checkpoint
   folders are keyed to is exactly how a half-finished run gets orphaned.

4. **Prefer `git mv` over `mv`** if the file is tracked, so history is preserved. (Untracked,
   gitignored things like `ckpts/` use plain `mv` — but see the landmine about not moving live
   checkpoints at all.)

5. **Dry-run, then verify.** After moving, re-run the tree listing, then confirm nothing broke:
   the scripts still import cleanly (`python3 -c "import ast,sys; ast.parse(open(f).read())"`
   per file, or a real `--help`), and `git status` shows only the intended changes.

6. **Never `rm -rf` a whole directory** to "tidy." `ckpts/` holds resume state; `archive/` and
   `_legacy/` are the project's record of negative results. Move, don't delete.

---

## 7. Rules & landmines (read before moving anything)

- **`MedRAG/`, `MedVLThinker/` and `MedEvalKit/` are separate git repositories and live
  dependencies.** `retrieve.py` imports from `MedRAG`; the June eval stack uses `MedVLThinker`;
  **`MedEvalKit/` is the faithful harness every current paper number comes from, and its
  `eval_results_*/` output directories hold the primary evaluation dumps.** Moving or renaming any of
  them breaks imports for zero benefit. **Leave them at the root, untouched — and do not "clean"
  `MedEvalKit/eval_results_*`.**
- **`ckpts/` checkpoints are resumable and gitignored.** Scripts write per-sample JSONL and
  resume from the last completed line if restarted. **Moving a checkpoint folder while a run
  could resume into it orphans that resume state.** Do not relocate `ckpts/` contents as part of
  cleanup unless Leo explicitly says a run is finished. The `router_margin.pkl` gate artifact
  lives here and is the deployable result — handle with care.
- **`CKPT_DIR` and similar paths are relative to the launch directory, not the script.** Several
  scripts resolve `ckpts/...` against the current working directory at launch time. The
  invariant that keeps everything working is: **always run from the repo root**
  (`cd ~/medvlthinker-imgdiff-compute` first). If you move scripts into subfolders, do **not**
  also start running them from inside those subfolders — keep launching from the repo root
  (e.g. `python3 src/cascade/live_cascade.py`), or the relative checkpoint paths will point at the
  wrong place and silently re-run from scratch.
- **`.gitignore` keeps `ckpts/`, `logs/`, `data/`, `results/`, env backups, and the big
  dependency repos out of git.** The committed tree is meant to be **code only**. After any
  restructure, re-check `.gitignore` still matches the new paths, and before any commit, inspect
  the staged file list and scan for unexpectedly large files.
- **No fabricated numbers — ever.** This is a standing, non-negotiable rule for this project.
  Every figure in the paper, slides, or site must come verbatim from real experimental output.
  If a number is needed and not available, recompute it from checkpoints; never invent it.
  **Also never mislabel provenance:** an estimate is an estimate until it is measured, and a number
  copied by hand into a deck is not "read from an artifact". Both failure modes have happened here
  (retrospective §7 hole 14, §10.2 X8). See CRITICAL RULE 7.
- **The July/Lingshu work is not in git.** Last commit `8cdefef` (2026-07-02). 44 untracked `.py`
  files under `src/` include the entire live headline chain; the IEEE paper, the July diaries and the
  2026-07-27 deck are untracked; `results/` and `MedEvalKit/` are gitignored. **The method, its inputs
  and its outputs currently exist on one disk.** Do not delete or relocate anything untracked, and
  treat "commit the working tree" as the standing top-priority chore.
- **Code-delivery convention Leo uses:** brand-new files / standalone scripts are delivered as
  a heredoc (`cat > path << 'EOF' ... EOF`) so he can paste them whole. **Edits to existing
  files** are delivered as a plain code block (the snippet to change), which Leo applies himself
  — do **not** wrap an edit-to-an-existing-file in a heredoc that overwrites it. Do not append
  execution commands onto a file-creation heredoc.
- **Long-running jobs use `nohup`, never `tmux`.** Use checkpointed, resumable runs with
  per-sample error guards.
- **Two GPUs, shared storage.** The VM has dual A100 80GB GPUs (user `jamesyang`), a shared
  `/data` mount (weights + datasets live there, not in the repo), and a shared home. Code lives
  in the repo; **weights and datasets never go in the repo** — they stay under `/data/dan/...`.

---

## 8. Environment & where things live (for running, not editing)

- **Repo:** `~/medvlthinker-imgdiff-compute`
- **Weights (June era):** 7B at `/data/dan/weights/MedVLThinker-7B-RL_m23k`; 32B at
  `/data/dan/weights/MedVLThinker-32B-RL_m23k`. (A legacy `MedVLThinker-3B-RL_m23k` exists from
  the early phase.)
- **Weights (current era):** Lingshu-7B / Lingshu-32B and the peer families live in the HF cache,
  `/data/dan/hf_cache/hub/`. ⚠️ **The runners hard-code HuggingFace snapshot-hash paths** (e.g.
  `models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd4…`) — a cache refresh changes the hash
  and breaks them.
- **Eval data (June era):** `/data/dan/dataset/MedVLThinker-Eval` — 8,220 samples across the six
  benchmarks (`pmc_vqa`, `pathvqa_closed`, `slake_closed`, `vqa_rad_closed`, `MMMU-medical`,
  `MedXpertQA-MM`). Its `pmc_vqa` slice **is PMC-VQA `test_clean.csv`** (v1, 2,000 items, the
  human-verified split — verified 2,000/2,000 positionally). Train splits: PMC-VQA at
  `/data/dan/dataset/pmc_vqa_train`; others under `/data/dan/dataset/{vqa_rad, path_vqa, slake}`.
- **Eval data (current era):** `/data/dan/dataset/medevalkit/` — the MedEvalKit datasets. Faithful-run
  sizes: PMC-VQA 33,430 (**`test_2.csv`**, v2) · SLAKE 2,094 · VQA-RAD 451 · PathVQA 6,719 · MMMU 150 ·
  MedXpert 2,000 · OmniMedVQA 88,996. (`test_clean.csv` also sits in this directory — 418,686 bytes,
  2,000 rows — but MedEvalKit's loader cannot reach it; see the two-split landmine in §0.)
- **Inference:** vLLM (NGC container, ~35× faster) for bulk labeling; HuggingFace transformers
  for the live cascade VRAM/energy measurement. **Three** Python environments exist, only two were ever
  documented — the third is **`/data/dan/medeval_venv/bin/python`** (vLLM 0.9.0.1), used by 12 runner
  invocations and required for every MedEvalKit run. MedEvalKit recipe: the Qwen2_5_VL wrapper,
  `datasets_path=hf`, `TORCHDYNAMO_DISABLE`, seed 42.
- **Critical model quirk (MedVLThinker):** these models only emit a `<think>` trace when the system
  prompt is *exactly*: "You will solve a problem/request. You should provide your thoughts within
  `<think>` `</think>` tags before providing the answer." Without that exact prompt they answer
  directly. Don't paraphrase it.
- **Companion correction (Lingshu):** "Lingshu has no promptable reasoning mode" was asserted on
  2026-07-01 and **retracted within 24 hours** — it was a weak-prompt artifact. With a proper prompt,
  generated tokens go **3 → 174** (267 with real `<think>` tags). MedEvalKit's `--reasoning True` flag
  only appends *"put the letter in \boxed{}"*, which is **not** a reasoning prompt. This is the root of
  retrospective hole 1: the dumps named `..._think` for PMC (`test_2.csv`) / SLAKE-closed /
  VQA-RAD-closed average 3–4 generated tokens and are **not genuine reasoning runs**.
  **Extended 2026-07-29:** the *internal* (NGC-harness) Lingshu "native think" arm has the same defect —
  `runners/run_native_think.sh:7` passes only *"Answer with the option's letter … in one `\boxed{}`"*,
  and that arm generates **3.0** tokens across all 7 benchmarks. All 7 published Lingshu think-vs-direct
  cells are therefore **withdrawn**; the genuinely-reasoning replacement is
  `ckpts/acc_gen/lingshu32b/think_fullres` (150–259 tokens). Independently, `MedEvalKit`'s reason prompt
  is only a *reasoning* prompt because of two **local uncommitted edits** (§0) — upstream it was the
  direct arm's format string. **Before trusting any think/reasoning dump in this repo, check its mean
  generated tokens.**
- **Known infra wall:** Lingshu-32B / InternVL3-38B under MedEvalKit with `tp=2` on OmniMedVQA hangs
  deterministically in an NCCL collective. Every mitigation was tried and failed (retrospective §9.2);
  `tp=1` is impossible (64 GB weights on an 80 GB card). Do not re-litigate it without reading §9.2.

---

## 9. Quick "what is this file?" decision guide for cleanup

When you encounter a file and aren't sure where it belongs:

- Is it `MedRAG/`, `MedVLThinker/` or `MedEvalKit/`? → **dependency, leave at root, don't touch.**
- Is it a `.jsonl`, `.pkl`, or under `ckpts/`/`logs/`/`results/`? → **data/checkpoint, gitignored,
  do not move during a code cleanup unless told the run is done.**
- Does its name contain `complexity`, `difficulty`, `lesion`, `gate_probe`, `gate_rag`? →
  **killed direction → `archive/`.**
- Is it `retrieve.py` / RAG-related? → **legacy retrieval, keep for record (own subfolder).**
- Does its name contain `abstain`, `deferral_curve`, `triage`? → **the forbidden direction.
  Historical record only; do not run, extend, or cite as a live method (CRITICAL RULE 6).**
- Is it a live script? → it belongs under `src/<stage>/`; find it in **`STRUCTURE.md`**.
- Unsure? → **leave it, list it, and ask Leo.** Never guess-delete.

> **A meta-lesson from the retrospective (§9.6):** the failure mode in this repository is *not*
> sloppiness — it is that **corrections were made in new files instead of being propagated back into
> the old ones**. When you fix a number, fix it everywhere it appears, or add an explicit supersession
> banner to every stale copy. That is why §0 above names one canonical value and one canonical source.
