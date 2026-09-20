# Corrections applied to the documents, 2026-09-20

**Branch `audit-2026-09-18`, worktree `.claude/worktrees/audit-2026-09-18`. Nothing committed.**
Source of every new fact: `../AUDIT_2026-09-18.md` and the sweep reports in this directory.

**Rule followed (R1):** no number was changed unless it was re-opened and confirmed at the printed
precision in the artifact JSON under `results/cascade_methods/artifacts/` (or, for the shipped
recipe, `ckpts/train/genframe_head_pooled_ens_v2/recipe.json`, read-only). Where a printed number
could not be reproduced it was **marked, not replaced** — those rows say so.

Line numbers are **post-edit** (the line the new text starts on). `git diff` is the authority.

---

## Verification helper

Every artifact value below was read with a short script against the JSON on disk. The confirmations
that mattered most:

| artifact | key | value read |
|---|---|---|
| `free_signal_bakeoff_2026-08-21.json` | `cells.pathvqa_open.n_questions` / `arms_judge.*` | 3357 · greedy 0.3139708 · prior 0.2832887 · sc 0.2943104 · head 0.3452487 · oracle 0.4807864 |
| `free_signal_bakeoff_2026-08-21.json` | `cells.slake_open.arms_judge` | prior 0.7302326 · sc 0.7426357 · head 0.7720930 |
| `free_signal_bakeoff_2026-08-21.json` | `cells.vqa_rad_open.arms_judge` | prior 0.425 · sc 0.46 |
| `free_signal_bakeoff_2026-08-21.json` | Σ `cells.*.n_questions` | **36869** |
| `head_final_stack_PVFIXED_2026-09-13.json` | `macro` | 0.0182388 / 0.0729250 / 0.0736074 / 0.0719723; `beats_greedy` 6/8,6/8,6/8,7/8; `pooled_rows` 112770 |
| `head_final_stack_qwen_2026-09-13.json` | `macro.pooled_ens`, `pooled_rows`, `duplicate_training_rows_dropped` | 0.0820350 · 145085 · 77209 |
| `head_lobo_pooled_2026-08-25.json` | `macro`, `macro_breadth_gain`, `macro_own_data_gain` | 0.0278581 / 0.0222645 / 0.0707468 · −0.0055936 · 0.0484823 |
| `head_temp_ensemble_2026-08-30.json` | `macro_by_temperature`, `macro_spread_across_T` | 0.0392550 / 0.0539232 / 0.0773413 / 0.0710126 · 0.0380863 |
| `head_best_config_2026-08-24.json` | `macro_minus_greedy` | A 0.0181509 · B 0.0306096 · C 0.0281302 · D 0.0342734 (D−A = 0.0161) |
| `decomposition_2026-08-24.json` | `summary_T07` | skill 0.0387126 · penalty 0.0256802 · 7/8 |
| `head_price_from_lobo_2026-08-30.json` | `cells.*.curve.{0,100}.minus_greedy`, `summary.zero_shot_macro`, `seeds` | omnimed −0.0107599→0.0997534 · radimagenet 0.0149402→0.0577689 · gemex 0.0291604→0.0854701 · 0.0196395 · 3 |
| `head_pooled_alldomains_2026-08-24.json` | `macro`, `cells.pathvqa_open.n_questions` | 0.0213251 / 0.0777969 / 0.0564718 · **700** (pre-backfill) |
| `coverage_sc16_ci_ALL_2026-09-16.json` | `macro_budget_16_minus_8`, `VERDICT`, per-cell deltas | 0.0166667 · 4 WIN 0 LOSS 4 TIE · radimagenet 0.0328685, kvasir 0.0308618, omnimed 0.0461780, gemex 0.0359477 |
| `bestofn_vllm_2026-09-16.json` | `forward_tokens_per_question`, `measured`, `ratios_vs_greedy` | flopeq_shared 1.13 / as_charged 8.0 · 173.99→476.61 ms · 34.27→123.17 J · 2.739 / 3.594 |
| `em_rescore_pooled_probe_2026-09-18.json` | `macro.delta_*` | judge 0.0737201 [0.0607642, 0.0863763] WIN · EM 0.0047060 [−0.0077404, 0.0171103] TIE · strict EM 0.0103157 · token-F1 0.0156472 [0.0043852, 0.0265164] · benchmark-level judge CI [0.0259685, 0.1165120] |
| `xjudge_rescore_medgemma27b_2026-09-20.json` | `macro` | delta_A 0.0737201 · delta_B **0.0679145** · 7/8 positive, 5/8 significant |
| `judge_2x2_2026-09-20.json` | `macro`, `train_rows`, `train_label_agreement_AB` | A|A 0.0709617 · A|B 0.0658360 · B|A 0.0565232 · B|B 0.0803811 · 112770 · 0.9285182 |
| `replication_currency_2026-09-20.json` | `qwen.macro.probe_minus_greedy_*` | judge 0.0813701 [0.0708338, 0.0924411] · EM 0.0275887 · strict EM 0.0187278 · F1 0.0317042; greedy pathvqa 0.0714726, vqamed 0.0132817; `headroom.share_of_macro_from_lt0.10_cells` 0.0531802 |
| `replication_currency_2026-09-20.json` | `judge_family_s09.*.macro_P_judge1_given_em0_greedy` | lingshu 0.1389079 · qwen 0.1673580 · medgemma 0.1658507 |
| `pilot_cross_model_transfer_2026-09-18.json` | `macro` | A_zeroshot_minus_greedy **0.0451007** · ctrl 0.0737201 · ref qwen native 0.0820350 |
| `ckpts/.../genframe_head_pooled_ens_v2/recipe.json` (read-only) | `training_rows`, `measured_on_held_out_halves` | 112770 · macro 0.0736074 · reload_verified 0.0737 · four-domain baseline 0.0182388 |
| recomputed from `head_final_stack_PVFIXED_2026-09-13.json` | 4-cell equal-weight means | pooled_ens **0.0479487** · pooled_singlelayer 0.0487472 · pooled_ens_sc 0.0424015 |

Counted on disk 2026-09-20: `progress/*.md` = **24** (June 17 → August 17) · `runners/*.sh` = **147** ·
`results/cascade_methods/docs/current/*.md` = **30** · `meetings/*.html` = **8** ·
`PROJECT_RETROSPECTIVE_2026-07-29.md` = **2,672** lines.

---

## A. `results/cascade_methods/docs/current/OPENTEXT_FULL_RUNDOWN_2026-09-04.md`

| line | old text | new text | justification |
|---|---|---|---|
| 7–22 | *(none)* | supersession banner: PathVQA backfill, judge identity, four currencies, §5.6 withdrawn | `AUDIT_2026-09-18.md` §0, §2, §7 |
| 36–47 | *(none)* | §0 currency paragraph + "what *reads the frozen hidden states* means in the code as run" | `run_judge.py:21`; `extract_generator_hidden.py:437-439`, `:49`; audit §1, §4 |
| 55 | `questions … 35,012` | `36,869 (was 35,012 …)` | `free_signal_bakeoff_2026-08-21.json` Σ `cells.*.n_questions` = 36869 |
| 56 | `probe training rows … 108,126` | `112,770 (was 108,126 …)` | `head_final_stack_PVFIXED…json:pooled_rows` = 112770; `recipe.json:training_rows` = 112770 |
| 64 | PathVQA row `1,500 / 0.3427 / 0.3333 / 0.3260 / 0.3900 / 0.5167` | `3,357 / 0.3140 / 0.2833 / 0.2943 / 0.3452 / 0.4808` | `free_signal_bakeoff…:cells.pathvqa_open` |
| 65 | SLAKE `prior 0.7287 · sc 0.7395 · head 0.7690` | `0.7302 · 0.7426 · 0.7721` | `…:cells.slake_open.arms_judge` |
| 66 | VQA-RAD `prior 0.4350 · sc 0.4650` | `0.4250 · 0.4600` | `…:cells.vqa_rad_open.arms_judge` |
| 73 | *(none)* | note that the corrected rows still support "beats the prior on all eight" | recomputed from the same cells |
| 116 | *(none)* | §3.1 caveat: that artifact was never regenerated; PathVQA cell is n=700 | `head_pooled_alldomains_2026-08-24.json:cells.pathvqa_open.n_questions` = 700 |
| 121 | heading `+0.0168` | `+0.0161` | `head_best_config…:macro_minus_greedy` D−A = 0.034273−0.018151 |
| 126 | `MACRO +0.0199 / +0.0324 / +0.0297 / +0.0366` | `+0.0182 / +0.0306 / +0.0281 / +0.0343` (old in brackets) | `head_best_config_2026-08-24.json:macro_minus_greedy` |
| 135–149 | §3.3 table on `head_final_stack_2026-08-24.json`, `+0.0243/+0.0765/+0.0802/+0.0797`, SC `−0.0004` | re-based on `head_final_stack_PVFIXED_2026-09-13.json`: `+0.0182/+0.0729/+0.0736/+0.0720`, SC `−0.0016`, with beats-greedy column | `head_final_stack_PVFIXED…:macro`, `:beats_greedy` |
| 155 | *(none)* | §3.4 pointer to the CI-carrying 8→16 measurement (+0.0167, 4 WIN 4 TIE) | `coverage_sc16_ci_ALL_2026-09-16.json:macro_budget_16_minus_8`, `:VERDICT`, `:cells` |
| 160–174 | §3.5 `genframe_head_pooled_ens`, `108,126 rows`, `+0.0816`, `+0.0243` | `genframe_head_pooled_ens_v2`, `112,770`, `+0.0737`, `+0.0182`, plus the gitignored-README warning | `recipe.json` `measured_on_held_out_halves` + `training_rows` |
| 183 | LOBO `−0.0008` | `−0.0056` | `head_lobo_pooled_2026-08-25.json:macro_breadth_gain` |
| 200 | `mean +0.0403 … penalty +0.0256` | `+0.0387 … +0.0257` | `decomposition_2026-08-24.json:summary_T07` |
| 205–209 | `+0.0312 → +0.0304 → +0.0803`, breadth `−0.0008`, own data `+0.0500` | `+0.0279 → +0.0223 → +0.0707`, `−0.0056`, `+0.0485` | `head_lobo_pooled_2026-08-25.json:macro`, `macro_breadth_gain`, `macro_own_data_gain` |
| 212–217 | `−0.0004→+0.0986 · +0.0010→+0.0528 · +0.0224→+0.0857` | `−0.0108→+0.0998 · +0.0149→+0.0578 · +0.0292→+0.0855`, plus zero-shot macro `+0.0196` and the 3-seed warning | `head_price_from_lobo_2026-08-30.json:cells.*.curve`, `:summary.zero_shot_macro`, `:seeds` |
| 220–224 | `T=0.2 +0.0401 · 0.4 +0.0573 · 0.7 +0.0816 · 1.0 +0.0759 · spread 0.0416` | `+0.0393 · +0.0539 · +0.0773 · +0.0710 · spread 0.0381`, plus the +0.0063 vs the 0.0061 tie-break band | `head_temp_ensemble_2026-08-30.json:macro_by_temperature`, `:macro_spread_across_T` |
| 235–248 | §5.6 layer numbers presented as results | **UNREPRODUCIBLE banner; numbers kept and marked withdrawn, no replacements invented** | `docs-md_findings.json` S5.6; artifacts cover 5 of 8 benchmarks |
| 297 | "Second generator (Qwen) … in flight, 170,014 training rows" | landed: `+0.0820, 8/8, 145,085` rows, four currencies, same-family caveat | `head_final_stack_qwen_2026-09-13.json`; `replication_currency_2026-09-20.json:qwen.macro` |
| 298 | *(none)* | new row: **MedGemma third generator WITHDRAWN** | audit §3; `replication.md` |
| 309–327 | `+0.0802 … against +0.0243` | `+0.0736 … +0.0182`, "judge currency", plus the four-currency table | `head_final_stack_PVFIXED…`, `em_rescore…`, `xjudge_rescore…`, `judge_2x2…` |
| 341 | *(none)* | three added holes (extraction pass · judge-specific share · missing baselines) | audit §4, §2.4, §5 |

## B. `…/TRANSFER_WALL_2026-08-21.md`

| line | old text | new text | justification |
|---|---|---|---|
| 10–23 | banner listed only the GEMeX supersession | added the `AUDIT_2026-09-12.md` §5 list (**§1, §3, §5, §6, §10 superseded**) and three `AUDIT_2026-09-18.md` points | `AUDIT_2026-09-12.md` §5, quoted verbatim; `docs-md_findings.json` doc2 banner_check |
| 335–349 | §12 opened straight into the Qwen table | banner: not a cross-family replication (Lingshu **is** a Qwen2.5-VL fine-tune) + Qwen's four currencies | `replication_currency_2026-09-20.json:qwen.macro.probe_minus_greedy_*`, `:headroom` |
| 459–478 | §14 opened straight into the MedGemma argument | **⛔ WITHDRAWN banner** with the degeneration figures | audit §3; `replication.md`; `replication_currency_2026-09-20.json:degeneration_s12.medgemma` |
| 496–506 | `Lingshu, full protocol … +0.0433` | value **left as printed**, with a note that it does not reproduce (recomputed **+0.0479**) | recomputed 4-cell mean from `head_final_stack_PVFIXED…:cells.*.pooled_ens_minus_greedy`; alternates 0.0487 / 0.0424 also checked |

## C. `…/PRIOR_ART_PROBE_VERIFIER_2026-09-14.md`

| line | old text | new text | justification |
|---|---|---|---|
| 9–88 | *(none)* | dated **CORRECTIONS 2026-09-20** block: A (the three papers that break the residual claim, C1–C2), B (C3–C7 on DualRead/MedProb/CASE), C (dimension D is the design, not the run), D (judge name, +0.0736 judge currency, the other currencies) | `lit-vlm-med.md` V1 + "Corrections owed…" C1–C7; audit §4; `head_final_stack_PVFIXED…`; `em_rescore…`; `xjudge_rescore…` |
| 94 | `(D) reused from the generation pass at no extra forward pass` | same, with an inline ⚠️ pointer to §C | audit §4 |
| 155 | residual-claim blockquote | prefixed **⛔ WITHDRAWN**, left otherwise intact, with the three arXiv ids | `lit-vlm-med.md` V1-a/b/c |
| 171–184 | DualRead: "Qwen3-VL and MedGemma", five benchmarks, "our pooling", "it pays a teacher-forced replay rather than reading the generation pass" | Qwen3-VL-**2B**/MedGemma1.5-**4B** + GRPO; SLAKE-test 1,061 + **six** OOD named; pooling claim removed; CCG-AUC/BICR noted; "and so do we" | `lit-vlm-med.md` C3–C6 (paper lines 502, 505, 1303) |
| 194 | §4 item 2 "First application of generation-pass hidden-state verification…" | struck as overreach (C2); the "most thorough characterisation" half kept and strengthened | `lit-vlm-med.md` C2 |

## D. `…/AUDIT_2026-09-12.md`

| line | old text | new text | justification |
|---|---|---|---|
| 5–18 | *(none)* | banner: superseded by `AUDIT_2026-09-18.md`; its §5 corpus correction 35,012 is now **36,869**; its §5b "HTML rundown not updated" is itself stale | Σ `free_signal_bakeoff…:cells.*.n_questions` = 36869; `docs-html.md` FILE 3 |

## E. `meetings/*.html`

Each deck was re-parsed with `html.parser` after editing; `<section>` counts are unchanged and
`<div>` increased by exactly **one** (the banner) in each of the three decks that received one.

### `meetings/opentext_progress_2026-09-14.html` (the deck shown to the professor)

| line | old text | new text | justification |
|---|---|---|---|
| 184–217 | *(none)* | the one correction banner `<div>` (6 numbered items: judge name · currencies · 1.13× cost · "free"/extraction pass · MedGemma withdrawn · wall-slide scope) | audit §0–§5; `docs-html.md` F1-a…F1-g |
| 176 | `Lingshu-7B · best-of-8 · 32B judge` | `… MedVLThinker-32B judge` | `run_judge.py:21`; audit §1 |
| 229 | `+0.0736` / `6 / 8 benchmarks beaten` captions | `— judge currency; +0.0047 (a tie) under exact match` / `benchmarks beaten, judge currency` | `em_rescore_pooled_probe_2026-09-18.json:macro.delta_em` |
| 281 | `a 32B-judge label` | `a MedVLThinker-32B-judge label` | `run_judge.py:21` |
| 393 | figcaption | appended the vLLM measured costs and the extraction-pass correction | `bestofn_vllm_2026-09-16.json:forward_tokens_per_question`, `:measured`, `:ratios_vs_greedy`; audit §4 |
| 458 | `in 32B-judge currency` | `in MedVLThinker-32B-judge currency` | `run_judge.py:21` |
| 502 | cost-table "best-of-8 + probe" row: `8.00× / 1.99× / 2.95×` only | added `1.13× / 2.74× / 3.59×` **beside** them, both conventions named | `bestofn_vllm_2026-09-16.json` (flopeq_shared 1.13, 173.99→476.61 ms, 34.27→123.17 J) |
| 514 | "why the three differ" callout | added a paragraph naming the vLLM shared-prefill convention as the one that describes us | same artifact, its `why` and `harness` fields |
| 628 | MedGemma row label `— incomplete`, cell text | `— WITHDRAWN 20 Sep 2026` + the degeneration figures | audit §3 |
| 641 | "not a property of Lingshu, of medical fine-tuning, or of the Qwen language model" | withdrawn inline; Qwen's four currencies given instead | audit §3; `replication_currency_2026-09-20.json` |
| 666 | coverage-wall row | scope correction: 71.5 % / 21.2 % are the **2,345-question, 3-benchmark** study | `coverage_diagnosis_2026-08-10.json` title + fields 3 and 4 (`docs-html.md` F1-f) |
| 679 | selection-wall row | marked unsourced; recomputed headroom +0.1098; shipped-probe sel-eff 0.51 | `docs-html.md` F1-e; audit §5 |
| 735 | closing "in one line" callout | added a correction paragraph ("free" and "three generator families" do not survive; what does) | audit §0 items 3–4 |

### `meetings/shipped_method_2026-09-12.html`

| line | old text | new text | justification |
|---|---|---|---|
| 201–224 | *(none)* | the one correction banner `<div>` (5 items) | audit §0–§4 |
| 231 | cover stat `+0.0788` | `+0.0820` (with "was +0.0788") | `head_final_stack_qwen_2026-09-13.json:macro.pooled_ens` |
| 334 | `Macro +0.0788 … over 222,154 training rows` | `+0.0820 … 145,085 deduplicated rows` | same artifact: `macro.pooled_ens`, `pooled_rows`, `duplicate_training_rows_dropped` = 77209 |

### `meetings/progress_deck_2026-08-24.html`

| line | old text | new text | justification |
|---|---|---|---|
| 210–226 | *(none)* | the one correction banner `<div>`: no numeric error found, but the recipe is superseded (+0.0736, 6/8) and the judge is MedVLThinker-32B | `docs-html.md` FILE 4; `head_final_stack_PVFIXED…:macro.pooled_ens` |

### `meetings/opentext_rundown_2026-09-04.html`

| line | old text | new text | justification |
|---|---|---|---|
| 142 | `35,012 questions` · `judge currency` | `36,869 questions` · `MedVLThinker-32B judge currency` | Σ `free_signal_bakeoff…:cells.*.n_questions`; `run_judge.py:21` |
| 217 | `2,345 → 35,012` | `2,345 → 36,869` (old value kept in the label) | same |

*(Scope note: the brief limited this file to numbers listed in `docs-html.md`, which lists none —
that sweep checked it against `AUDIT_2026-09-12.md`'s list, where 35,012 was the **correct**
value. 35,012 has since been overtaken by the PathVQA backfill and is named as wrong in
`AUDIT_2026-09-18.md` / F5, so the two occurrences were corrected. No banner was added: this deck
is the best-maintained of the four and carries a thorough supersession banner already.)*

## F. `results/cascade_methods/README.md`

| line | old text | new text | justification |
|---|---|---|---|
| 96–140 | *(none)* | new section **"August–September 2026 (probe-verifier era)"**: 16 `docs/current/` files dated 2026-08-12 onward, one line each from their own titles, incl. `AUDIT_2026-09-18.md` and `NEW_DIRECTIONS_2026-09-20.md`; plus a table of the audit's artifacts | titles read from the files' own `# ` headings; artifact one-liners from each JSON's `title` field |
| 142 | `## docs/current/ — the canonical writeups` | `… (July/Lingshu cascade era)` + a pointer up to the new section | `docs-top.md` Finding R (25 of 31 `docs/current/` files unindexed) |

*(The brief said "the nine audit artifacts"; **eleven** files matched `*_2026-09-18.json` /
`*_2026-09-20.json` on disk by the end of this pass — `judge_consensus_2026-09-20.json` and
`pilot_lineage_alignment_2026-09-20.json` beyond the nine, the latter written by a concurrent
session while this pass ran. All eleven are listed, and `pilot_lineage_alignment.py` was added to
the `src/audit_2026_09_18/` entries in `STRUCTURE.md` and to that directory's `README.md` for the
same reason.)*

## G. `STRUCTURE.md`

| line | old text | new text | justification |
|---|---|---|---|
| 18–30 | *(none)* | coverage note: **401 of 600 `.py` under `src/` and 141 of 147 `runners/*.sh` are unlisted**; 5 named paths no longer exist | `docs-top.md` items 3–4 (counts taken at `main`) |
| 51 | `progress/ … 13 dated daily progress logs (June 17 → July 8)` | `24 … (June 17 → August 17)` | `ls progress/*.md` = 24; first `progress_June_17.md`, last `progress_August_17.md` |
| 54 | `runners/ 38 shell launchers` | `147 shell launchers` | `ls runners/*.sh` = 147 |
| 187–208 | *(none)* | new section **`src/audit_2026_09_18/`**, one line per top-level script (docstrings read with `ast.get_docstring`) + `sweeps/` | the scripts themselves |

## H. `src/audit_2026_09_18/README.md` — **created**

12-step run order (`em_rescore` → `dump_probe_scores` → `judge_length_bias` → `build_xjudge_preds`
→ `launch_medgemma27b.sh` [GPU] → `rescore_xjudge` → `build_xjudge_train_preds` →
`launch_medgemma27b_train.sh` [GPU] → `judge_2x2_refit` / `judge_consensus_refit` →
`pseudolabel_precision`, `template_stratified_gain`, `pilot_cross_model_transfer`), what each script
answers, inputs and output artifact names, the `/data/dan/audit_2026-09-18/tmp/` layout, the
**≤ 4 threads, load-dependent** rule, run-from-repo-root, and `sweeps/`.

## I. Entry docs — dated pointer blocks only

| file | line | change | justification |
|---|---|---|---|
| `README.md` | 3–8 | 6-line dated pointer to `AUDIT_2026-09-18.md`; "numbers here are cascade-era" | `docs-top.md` item 3B (zero mentions of the live method) |
| `RESULTS.md` | 3–9 | same | same |
| `INCONSISTENCIES.md` | 3–9 | same, pointing at the two September audits | same |
| `READING_GUIDE.md` | 3–8 | same | same |
| `READING_GUIDE.md` | 10–15 | `1,972-line` → **2,672**; `14 current writeups` → **30**; `13 dated progress diaries` → **24 (June 17 → August 17)**; `3 HTML decks` → **8**; `~199 Python files` → **600+** | counted on disk 2026-09-20 (`wc -l`, `ls`, `find src -name '*.py'`) |

---

## Deliberately NOT changed

| what | why |
|---|---|
| **`OPENTEXT_FULL_RUNDOWN` §5.6 layer values** (L18 +0.0275 … L22 +0.0277) | **Cannot be verified or recomputed.** The per-layer artifacts on disk were re-created 2026-09-13 covering only 5 of 8 benchmarks; L19 reads +0.0280 and L21 +0.0183 there, over a different benchmark set. Marked **UNREPRODUCIBLE**; no replacement invented (R1). |
| **`TRANSFER_WALL` §14 first table, Lingshu `+0.0433`** | Does not reproduce from the artifact (recomputed +0.0479) and no metric was found that yields +0.0433. Left as printed with a note, per the brief. |
| **09-14 deck's title "The Free Verifier", its five "free" claims, the cover's "0× extra cost", and the closing line** | Framing, explicitly reserved for the author. The problem is stated in the banner, in the figcaption and in a paragraph appended to the closing callout; no slide was removed or retitled. |
| **09-14 deck's "44.1 % / 12.1 % / 79.0 %" coverage percentages** | Not located in any artifact. Flagged in the scope-correction sentence rather than changed. |
| **09-14 deck's "+0.1650 / 0.686 / 0.579 / 0.997" selection-wall cluster** | Unsourced; recomputation gives a different headroom (+0.1098). Marked, not replaced. |
| **09-14 deck's "+0.0099 on the weak base"**, domain-breadth `+0.0087`, ensemble-width `−0.0011` | `docs-html.md` could not trace them to an artifact key and neither could this pass; they are plausible and unflagged there, so they were left alone rather than guessed at. |
| **`OPENTEXT_FULL_RUNDOWN` §3.1 table values** | They match `head_pooled_alldomains_2026-08-24.json` exactly; the artifact is pre-backfill. A caveat was added instead of changing figures that correctly report their own source. |
| **`OPENTEXT_FULL_RUNDOWN` §3.4 `+0.0147`** | Correct for `coverage_scaling_ALL_2026-09-01.json` (its VERDICT's eight values average to 0.014675). A pointer to the newer CI-carrying `+0.0167` was added beside it rather than overwriting a still-valid number from a different measurement. |
| **`ckpts/train/genframe_head_pooled_ens_v2/README.md`** (prints +0.0802/+0.0243) | Under `ckpts/`, out of scope for this pass and gitignored. Flagged in `OPENTEXT_FULL_RUNDOWN` §3.5; it is `AUDIT_2026-09-18.md` §9 item 3, for Leo. |
| **`literature/DOMAIN_GUIDE_2026-09-16.md`, `OPEN_EXPERIMENTS_2026-09-17.md`** (≥13 "same-family Lingshu-32B judge") | They exist only on the unmerged `worktree-lit-domain-package` branch, not in this worktree. `AUDIT_2026-09-18.md` §9 item 6. |
| **`PROJECT_OVERVIEW.md`, `CLAUDE.md`, `PROJECT_RETROSPECTIVE_2026-07-29.md`, the June/July docs, `progress/*`, `paper/`** | Out of scope for this pass. `CLAUDE.md` was corrected separately; the dated diaries are the historical record and are never rewritten. |
| **Anything under `src/` except the new `src/audit_2026_09_18/README.md`** | Out of scope. `freeze_pooled_selector.py` and `head_final_stack.py` already carry the audit's fixes as uncommitted working-tree changes. |

## One deviation from R4 worth naming

In the 09-14 deck, three of the corrections are new `<p>` elements appended **inside existing
`.call .cr` callout divs** (which already contain multiple `<p>` siblings and are styled for it),
rather than text inserted into an existing paragraph. `<section>` counts are unchanged and `<div>`
counts rose by exactly one (the banner) in every edited deck, as required; `<p>` rose by 2 in the
09-14 deck. Every file was re-parsed with `html.parser` after every edit.
