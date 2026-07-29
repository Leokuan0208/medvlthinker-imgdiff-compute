# Consistency audit + resolutions

> **Two eras, two sections.** Part 1 below is the original **2026-06-27** audit of the MedVLThinker/ACC
> numbers (X1–X10); its resolutions are still valid *for that era*. **Part 2 (added 2026-07-29)** covers the
> July / Lingshu era — the `+0.02xx` number family, the 0.5632 baseline mislabel, and the MMMU decision.
> The full corrections log (19 retracted claims + 14 numeric corrections) lives in
> `results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md` §10.

## PART 1 — the 2026-06-27 audit (MedVLThinker / ACC era)

Built from a 3-way audit (paper / cross-docs / ground-truth-from-`ckpts`). Each item: the conflict, the
**RESOLUTION** (canonical value), and the **evidence**. Source of truth = raw `ckpts/**/result.json` +
`results/cascade_methods/master_data.csv` + `GROUND_TRUTH_NUMBERS.md`. Forward-facing docs (paper, CLAUDE.md,
README, RESULTS, METHOD_ACC, results/cascade_methods/{README,NEW_DIRECTIONS,...}) are corrected to canonical.
Dated `progress_*.md` diaries are the historical record: left as-written, with a header pointer to the
canonical numbers (history is preserved, not rewritten).

## CRITICAL — numeric conflicts (resolved by data files)

### X1. ACC efficiency headline — 3 conflicting sets → use master_data.csv
- Superseded: 20.0s→5.7s (−72%), FLOPs 81→55%, energy 7049→1505J (rt_cascade methodology; CLAUDE§0, README, RESULTS, METHOD_ACC, results/README, paper §5.1/Abstract).
- Superseded: paper §5.1.1 table 26.6s / ACC-v2 4.86s.
- **RESOLUTION (canonical, master_data.csv, ALL-6, MedVLThinker, parity 0.5723):**
  always-32B-think **11.34s / 6318.8J / FLOPs 100%** → **Ours ACC-v2 2.27s / 1181.9J / FLOPs 52.0%**, acc 0.5693
  ⇒ **latency −80%, energy ~5.3× (−81%), FLOPs halved**. ALL-5: 8.88s/4915.9J → 0.44s/172.8J/FLOPs 24.9%.
- Evidence: `results/cascade_methods/master_data.csv` (June-24 regen, after the June-22 cost-methodology fix
  that replaced rt_cascade 0.072 s/think-token with batch-1-native 0.0288 s/token). The fix is the one the
  user flagged (`progress/progress_June_20-22.md` "Cost-methodology bug … found + fixed").

### X2. MS-CXR box-verifier — 0.248 / 0.184 are stale → 0.232 / 0.230
- Superseded: "0.230/0.248, oracle 0.313, 77/76%" (SESSION_REPORT, paper §5.10, progress/progress_June_25-26);
  "0.184/oracle 0.228/71%" (NEW_DIRECTIONS_2026-06-25).
- **RESOLUTION (n=435, IoU≥0.3):** greedy 0.041, SC-medoid 0.053, trained **0.232 (seed0) / 0.230 (seed1)**,
  oracle 0.285 ⇒ **78.3% / 77.4%**; bootstrap gain **+0.191, 95% CI [+0.152, +0.232]**; zero-shot 0.115 (30%).
- Evidence: `ckpts/train/lora_box_verifier_mscxr_{full,boot,s1}/result.json`. The 0.248/oracle-0.313 was a
  pre-coordinate-fix artifact (boxes in smart-resized space → wrong IoU → wrong oracle); not in any current file.

### X3. Free-text verifier "pooled-5" → pooled-4 (n=1064)
- Conflict: progress/progress_June_25-26 says "pooled-5 datasets"; everywhere else "pooled-4".
- **RESOLUTION:** trained on/eval over **4** datasets (SLAKE, PathVQA, VQA-RAD, Kvasir), n=1064; RadImageNet is
  the **5th, held-out transfer** set (+0.024, 13%). Evidence: `lora_verifier_pooled4/result.json` (4 ds).

### X4. Verifier gain +0.088 vs +0.116 — different baselines (both correct, label them)
- **RESOLUTION:** +0.088 = trained 0.501 − **greedy** 0.413 (49% of gap). +0.116 = best-of-8 − **single random
  sample (K=1)** 0.385, bootstrap CI [+0.092,+0.139]. State the baseline every time. Evidence: result.json + scaling_curve.json.

### X5. K=8 verifier 0.501 vs 0.417 — two different test pools
- **RESOLUTION:** 0.501 is the headline (n=1064, sc8 pool). 0.417 is from the sc16 run (n=1621, a *fresh
  larger* pool incl. all PathVQA). Not comparable. Paper uses the n=1064 K≤8 curve as canonical; K=16 is
  described qualitatively ("continues to rise, diminishing returns") to avoid cross-pool confusion. Evidence: scaling_curve{,16}.json.

### X6. Over-thinking delta (32B no-think vs think) — fullres vs cap320
- Conflict: SLAKE +7.7/VQA-RAD +11.7 (fullres-nt) vs +0.084/+0.077 (cap320-nt).
- **RESOLUTION:** ACC's T1 tier is **no-think@cap320**, so the operative deltas are cap320: SLAKE 0.849 vs
  0.764 (+0.085), VQA-RAD 0.853 vs 0.776 (+0.077). Note fullres-nt is even higher. Evidence: master_data.csv (always-big-nt vs always-big-think).

### X7. Energy reduction "~4×" → ~5×
- **RESOLUTION:** 6318.8/1181.9 = **5.35× (~5×, −81%)** ALL-6. Fix all "~4×" to "~5×". Evidence: master_data.csv.

### X8. ALL-6 parity accuracy 0.5723 / 0.572 / 0.5718
- **RESOLUTION:** 0.5723 = always-32B-think parity (master_data.csv). 0.5718 = the deployed 2-tier anchor
  accuracy (margin gate at τ=0.426) — a *different* system. Keep both, labeled distinctly. 0.572 = rounding of 0.5723.

### X9. SLAKE box "+0.09 / 40%" baseline mismatch; "below greedy" wrong for MS-CXR
- **RESOLUTION:** standardize gap-captured to **greedy** baseline. SLAKE: (0.255−0.197)/(0.343−0.197)=40%.
  Reframe SC-medoid as "at the luck floor (≈ random)" not "below greedy" (MS-CXR SC-medoid 0.053 > greedy 0.041).

### X10. Verification-Mirage arXiv id 2605.10850 vs 2606.10850
- **RESOLUTION:** **2605.10850** (confirmed via arxiv.org/abs/2605.10850). Fix the 2606 instance.

## MEDIUM — table-level (note in paper, use master_data.csv)
- FLOPs% normalization: MASTER/FULL_RECORD fix always-big-think=100% (canonical); DETAILED_TABLES uses a
  different denominator (145%). Use the 100%-normalized values.
- always-big-nt cost 0.231s/36.2% (master) vs 0.36s/45% (DETAILED) → use master.
- ACC-v1/v2 minor variants (0.5688 vs 0.5694 etc.) → use master_data.csv rows.

## STALE REFERENCES
- S1: `RECOVERABILITY_IS_CAPACITY_BOUND.md` cites `tmp/{repair_decomp,repair_complement,ladder_sim,maxconf_ensemble}.py` — do not exist. ACTION: update ref (the logic is described in the doc; mark scripts as not-retained) — flagged, low priority (gitignored doc).
- S2: `OPENENDED_SELECTION_LUCKFLOOR.md` cites `tmp/crossfamily_agree.py` → now `src/cascade_methods/crossfamily_agree.py`. Fix.
- S3: some diaries reference root `run_*.sh` (now `runners/`). Historical — leave, header pointer added.

## TERMINOLOGY — standardize in paper + forward docs
- T1: "luck floor" = the umbrella term for the selection/action/gate bound; "recoverability ceiling" = the gate-side instance; "majority trap" = the sub-mechanism. Use consistently.
- T2: method name = **ACC-v2 (agreement)** everywhere; drop the orphan "ACC-A" (STRUCTURE.md).
- T4: one term for cascade-FLOPs/always-big-think FLOPs = **FLOPs%** in tables, gloss "(= backbone%)" once.
- T5: **ALL-6 = 6 benchmarks** (MedXpert = 1 benchmark, 2 splits ⇒ 7 splits for per-split tables). Define once in Setup. ALL-5 = ALL-6 minus MedXpert. COMPETENT-4 = SLAKE/VQA-RAD/PathVQA/PMC.
- T7: dataset canonical names (RESULTS.md:3-12): MedXpertQA-MM, VQA-RAD, PathVQA, SLAKE, PMC-VQA, MMMU-medical, Kvasir-VQA-x1.

## POLICY
- Paper + CLAUDE.md + README + RESULTS + METHOD_ACC + results/{README,NEW_DIRECTIONS,GROUND_TRUTH_NUMBERS} → corrected to canonical.
- Dated progress_*.md diaries → historical; add one-line header pointer to GROUND_TRUTH_NUMBERS.md; not rewritten.
- New writing (reports, docx, html) → must match GROUND_TRUTH_NUMBERS.md. Double-check every number against it.

---

# PART 2 — the 2026-07 audit (Lingshu / MedEvalKit era), added 2026-07-29

Source of truth for this era = `results/cascade_methods/artifacts/*.json` (chiefly
`f8_mode_vsthink_ci.json`, `opentext_32b_think_full.json`, `method_final_mmmu_corrected.json`,
`paper_baselines.json`) + the retrospective §4/§10.

## Y1. THE `+0.02xx` NUMBER FAMILY — six values, one operating point

All six describe **accuracy-max versus always-32B-with-reasoning**. They differ on three orthogonal axes:
which **lever** (confidence-advantage fusion vs certified veto), which **pool** (MMMU kept / escalated /
excluded), and whether the open-text reasoning cells were **estimated or measured**.

| value | lever | pool | open reasoning cells | baseline | source |
|---|---|---|---|---|---|
| +0.0212 | veto + learning-to-defer (0.93×) | full suite, MMMU kept | estimated | 0.5632 | `method_final_v2.json` |
| +0.0207 | same | full suite, MMMU **escalated** (Variant A) | estimated | 0.5632 | `method_final_mmmu_corrected.json` |
| +0.0238 | fusion (1.25×) | full suite | estimated | 0.5632 | `method_final.json` |
| **+0.0245** | **veto + L2D (0.93×)** | **Variant B, n = 42,224** | **MEASURED** | **0.5591** | **`f8_mode_vsthink_ci.json`** |
| +0.0271 | fusion (1.25×) | Variant B | measured | 0.5591 | `opentext_32b_think_full.json` |
| (+0.0275) | fusion | full suite | measured | 0.5594 | `progress/progress_July_08.md` |

- **RESOLUTION (canonical): `+0.0245`, 95% CI [+0.0216, +0.0274], n = 42,224, at 0.93× FLOPs.** It is the
  only value that is simultaneously (a) the FLOP-negative deployed configuration, (b) MMMU-clean,
  (c) measured, and (d) CI-certified. Evidence: `f8_mode_vsthink_ci.json:headline` (2026-07-09).
- **Companion compute-lean family:** +0.0117 → +0.0123 (estimates) → **+0.0150 [+0.0107, +0.0192] (Variant B,
  measured)** / +0.0154 (full suite, measured).
- **When quoting, always state lever + pool + measured.** The `⁺` suffix is used in the root docs for the
  fusion variant precisely because it is **not** FLOP-negative (1.25×).

## Y2. The always-32B-with-reasoning baseline: 0.5632 was labelled "measured" and was not

- Superseded: "**Baselines (measured):** always-32B-think = 0.5632" — printed in `PROJECT_OVERVIEW.md`,
  `TECHNICAL_REPORT_2026-07.md` and `METHOD_FINAL_2026-07.md`. **That value's open-text cells were
  estimates.** This is the closest thing to a no-fabricated-numbers violation found in the tree: the number's
  provenance was fine, the *label* was wrong.
- **RESOLUTION: 0.5594 (full suite, n = 42,374) / 0.5591 (Variant B, n = 42,224).** Evidence:
  `opentext_32b_think_full.json`, `f8_mode_vsthink_ci.json`. Corrected in the root docs 2026-07-29;
  the three `docs/current/` files carry supersession banners instead of edits (mechanism preserved).
- Consequence: CLAUDE.md's critical-rules block now carries an explicit **"never mislabel provenance"** rule.

## Y3. Open-text 32B-reasoning accuracy: n=200 estimates → full-set measurements

| set | n | estimate (superseded) | **measured** | direct mode | Δ |
|---|---|---|---|---|---|
| SLAKE-open | 645 | 0.6236 | **0.6791** | 0.8186 | −0.1395 |
| VQA-RAD-open | 200 | 0.4800 | **0.5450** | 0.6000 | −0.0550 |
| PathVQA-open | 1,500 | 0.2460 | **0.1087** | 0.3760 | −0.2673 |
| pooled | 2,345 | ~0.387 | **0.3028** | 0.537 | — |

- **RESOLUTION:** use the measured column (`opentext_32b_think_full.json`, 2026-07-08). Any doc quoting
  "0.387 pooled", "SLAKE 0.700", "VQA-RAD 0.425" or "PathVQA 0.035" is on the n=200/set subsample.
- Also: "verifier bo-N **0.563** > 32B-nt 0.517 > 32B-think **0.370**" → **0.5727 > 0.5168 > 0.3028**.
- ⚠️ **Open caveat (retrospective hole 3):** the PathVQA-open collapse — half the headline — may be an
  answer-granularity artifact (the reasoning model gives more clinically substantive answers that miss
  PathVQA's caption-fragment gold strings). The independent Claude-judge cross-validation covers SLAKE and
  VQA-RAD **only, not PathVQA**. Unresolved.

## Y4. MMMU: excluded → banked → excluded. The exclusion is final.

- 2026-07-02 (`MASTER_SUMMARY_2026-07.md`): "MMMU-7B: Lingshu-7B-specific inflation → **excluded**."
- 2026-07-07/08 06:xx (`METHOD_FINAL`, `TECHNICAL_REPORT`, `PROJECT_OVERVIEW`): **banked** as a headline
  per-benchmark win, **+0.140** vs think / **+0.167** vs no-think. **These three docs regressed on a decision
  already taken.**
- 2026-07-08 §7 + the IEEE paper: **excluded entirely** after the adversarial audit — model identity PASS,
  image ablation DECISIVE (0.827 → 0.62 blank → 0.593 text-only), control model DECISIVE (untuned
  non-medical Qwen2.5-VL-7B scores 0.567 through the same harness). Verdict: genuine weights, consistent
  with train-set contamination outside our control.
- **RESOLUTION: report "Variant B" — MMMU excluded, 5 benchmarks / 8 cells / n = 42,224.** Effect on the
  sample-weighted headline is only **−0.0005** (MMMU is 0.35% of the pool) but the **macro** average must be
  corrected (+0.0777 → +0.0621). Do **not** bank MMMU keep-7B as a beat-the-32B win. Evidence:
  `mmmu_verify.json`, `method_final_mmmu_corrected.json`, `progress/progress_July_08.md` §7.

## Y5. Other July corrections worth carrying (full list: retrospective §10.2)

- **X7'/Y5a — MMMU reasoning gains.** `progress_July_03.md` records +0.034 / +0.107 / +0.120; those diary
  endpoints do not exist on disk. Canonical: **+0.027 (Lingshu) / +0.100 (MedVLThinker) / +0.120
  (InternVL3-38B)**, computed from `MedEvalKit/eval_results_*/{}/MMMU-Medical-val/*/parsed_output.json`.
- **Y5b — `RESULTS.md` said the 7B-think dumps are "only PMC + MedXpert at n~500".** Wrong: `ckpts/gate_7b_think/`
  holds the full **8,220** rows across all 6 benchmarks, in 2 shards. Verified by row count 2026-07-29.
- **Y5c — `README.md` said the 3-family × 7-benchmark matrix "is in progress".** It finished 2026-07-03/05
  (6 of 7 fully faithful + one cheap-faithful; OmniMed strong leg infra-blocked, `OMNIMED_FALLBACK.md`).
- **Y5d — the backlog holds 68 ideas**, not 56 (`RESEARCH_RESULTS_2026-07.md`).
- **Y5e — `harness.py` is imported by 39 files** (not ~36/45); there are **38** runners (not 23/40).
- **Y5f — the 32B/7B compute ratio 4.57 is an underived hard-coded literal.** No file derives it; an older
  math doc implies 4.34 from parameter counts. At 4.34 the accuracy-max ratio moves ~0.93× → ~0.95×, so the
  "compute-negative" claim has a 7% margin on an underived denominator. **Open** (retrospective hole 14c).

## Y6. UNRESOLVED contradictions — do not treat either side as settled

- **Y6a — is verifier confidence really the best open-text gate?** Two runs of the same regime disagree on
  the published deferral baseline: 2026-07-01 reports it at 0.3832 vs verifier-confidence 0.3923 ("we beat
  the SOTA"); `artifacts/gate_unified_bakeoff.json` (2026-07-04) reports **0.3965 vs 0.3901, +0.0062
  [+0.0040, +0.0086], winning on 100% of seeds** — same underlying pooled accuracies, **opposite sign**.
  The claim *"the gate question is settled"* is **not safe** until this is reconciled. (Retrospective hole 15.)
- **Y6b — best-of-8 cost accounting is internally inconsistent.** Latency treats verification as ONE forward
  for all N (522 = 347 + 175) while compute treats it as N forwards (16 = 8 × 2). Both cannot describe the
  same implementation, and no batch-8 measurement exists in the repo. Until measured, do not quote
  "best-of-N is latency-alive at 0.79×". (Retrospective hole 8.)
- **Y6c — the vs-reasoning cost claim.** The reasoning baseline is a **no-reasoning run on ~90% of the pool**
  charged at reasoning cost; honest re-costing turns −95.5%/−96% into ~−72%/−74%. (Retrospective hole 1.)

## Y7. Stale references — status of Part 1's S-items

- **S1** (`RECOVERABILITY_IS_CAPACITY_BOUND.md` → non-existent `tmp/*.py`): **still open.** The scripts were
  never retained; the logic is described in the doc. **WONTFIX** — mark the scripts as not-retained rather
  than inventing paths.
- **S2** (`OPENENDED_SELECTION_LUCKFLOOR.md` → `tmp/crossfamily_agree.py`): **still open.** Correct path is
  `src/cascade_methods/crossfamily_agree.py`.
- **S3** (diaries referencing root `run_*.sh`, now `runners/`): historical, leave as written.
- **New S4:** the Part-1 POLICY line says every dated diary gets a one-line header pointer to
  `GROUND_TRUTH_NUMBERS.md`. Only `progress_June_27-28.md` has one. **Partially executed.**

## Y8. POLICY (unchanged, extended)

- Forward-facing docs (`CLAUDE.md`, `README.md`, `RESULTS.md`, `READING_GUIDE.md`, `PROJECT_OVERVIEW.md`,
  `STRUCTURE.md`, `results/cascade_methods/README.md`) → **corrected to canonical** (done 2026-07-29).
- Large `docs/current/` writeups whose *mechanism* is correct but whose *numbers* pre-date the 2026-07-08
  seam → **annotated with a supersession banner, not rewritten** (done 2026-07-29): `TECHNICAL_REPORT_2026-07`,
  `METHOD_FINAL_2026-07`, `RESEARCH_RESULTS_2026-07`, `MASTER_SUMMARY_2026-07`, `METHODS_MASTER`,
  `METHOD_ACC`, `METHOD`, `METHOD_deferral_router`.
- Dated `progress_*.md` diaries → **historical, never rewritten.** They are the most trustworthy layer in the
  tree precisely because they were written in the moment and flag their own errors.
- New writing (reports, decks, docx, html) → must match the retrospective §4. **Double-check every number
  against the named artifact, and never hand-type a table you have not read from JSON** — the 2026-07-27 deck
  contains 116 hand-typed literals and zero JSON reads.
