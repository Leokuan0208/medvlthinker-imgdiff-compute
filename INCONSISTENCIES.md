# Consistency audit + resolutions (2026-06-27)

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
  user flagged (`progress_June_20-22.md` "Cost-methodology bug … found + fixed").

### X2. MS-CXR box-verifier — 0.248 / 0.184 are stale → 0.232 / 0.230
- Superseded: "0.230/0.248, oracle 0.313, 77/76%" (SESSION_REPORT, paper §5.10, progress_June_25-26);
  "0.184/oracle 0.228/71%" (NEW_DIRECTIONS_2026-06-25).
- **RESOLUTION (n=435, IoU≥0.3):** greedy 0.041, SC-medoid 0.053, trained **0.232 (seed0) / 0.230 (seed1)**,
  oracle 0.285 ⇒ **78.3% / 77.4%**; bootstrap gain **+0.191, 95% CI [+0.152, +0.232]**; zero-shot 0.115 (30%).
- Evidence: `ckpts/train/lora_box_verifier_mscxr_{full,boot,s1}/result.json`. The 0.248/oracle-0.313 was a
  pre-coordinate-fix artifact (boxes in smart-resized space → wrong IoU → wrong oracle); not in any current file.

### X3. Free-text verifier "pooled-5" → pooled-4 (n=1064)
- Conflict: progress_June_25-26 says "pooled-5 datasets"; everywhere else "pooled-4".
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
