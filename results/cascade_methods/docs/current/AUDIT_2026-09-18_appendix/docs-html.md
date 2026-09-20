# Documents-vs-artifacts audit -- September 2026 HTML decks

Agent: docs-html. Scope: MAIN=/home/jamesyang/medvlthinker-imgdiff-compute, decks in meetings/:
- opentext_progress_2026-09-14.html (newest, shown to professor -- EXHAUSTIVE)
- shipped_method_2026-09-12.html (EXHAUSTIVE)
- opentext_rundown_2026-09-04.html (supersession banner + specific corrections)
- progress_deck_2026-08-24.html (supersession banner + specific corrections)

Method: reused predecessor's extracted text (*.txt), leaves.pkl (flattened index of all
results/cascade_methods/artifacts/*.json + ckpts/train/genframe_head_pooled_ens_v2/recipe.json,
built 2026-09-18 09:55, confirmed no artifact files changed since -- still current as of 2026-09-20),
and find_num.py to match printed numbers to artifact leaves. Verified line numbers against raw HTML.

Status: IN PROGRESS -- appending findings as confirmed.

---

## Findings table

(id . severity . claim . evidence . proposed correction)

test line

## FILE 1: opentext_progress_2026-09-14.html (newest, shown to professor) -- EXHAUSTIVE

### CRITICAL

F1-a. Title/thesis violates the project's own prior-art finding, written the SAME MORNING, 26 min earlier.
- HTML line 185 <h1>The Free Verifier</h1>; line 189 "so verification became free, and got better at the
  same time"; line 195 fig "0x extra cost for verification itself"; line 370 row-label "Capture, don't
  recompute" tag "why it is free"; line 656 closing line "Verification is now free and better than the
  version we showed in July...".
- Deck file mtime 2026-09-14 09:02:09. results/cascade_methods/docs/current/PRIOR_ART_PROBE_VERIFIER_2026-09-14.md
  mtime 2026-09-14 08:36:41 (26 min BEFORE the deck was published). Its S0/S4: "Do not claim the
  mechanism. Do not claim the 'free' argument. Both are taken." / "Retire 'verification is free' as the
  contribution. It is HSRM's, CASE's, Q-Probe's abstract, in those words." Five 2025-26 papers (Q-Probe
  2402.14688, LiLaVe 2504.16760, SWIFT/ELHSR 2505.12225, ReProbe 2511.06209, HSRM 2608.30841) use the same
  "free"/"reusing representations already computed" argument.
- Class: (d) framing / claimed-forbidden-novelty. Severity: CRITICAL -- it is the whole deck's title and
  closing line, in the deck the professor is shown.
- Fix: retitle away from "free"; state the 0-extra-forward-pass property once, factually, without making
  it the deck's thesis; cite Q-Probe/HSRM/SWIFT/CASE per PRIOR_ART section 4.

F1-b. Headline FLOP-eq cost (8.00x) is not the cost of the generation path actually used to build every
candidate pool in the project; the number that is (1.13x) exists but is omitted.
- HTML lines 328/348 (diagram), 421/423/456-467 (tables): "8.00x FLOP-eq" for both self-consistency and
  the probe ("best-of-8 + probe -- today"), stated as THE FLOP cost of best-of-8, no mention of a lower
  true figure.
- Artifact: results/cascade_methods/artifacts/bestofn_vllm_2026-09-16.json (mtime 2026-09-16, 2 days
  AFTER this deck), forward_tokens_per_question: flopeq_shared=1.13 (rep2: 1.126), flopeq_as_charged=8.0.
  Its own "why" field: "Every candidate-set pool in this project was generated under vLLM with n=8, which
  forks after ONE shared prefill. This measures the path we actually use." -- i.e. the artifact that
  supersedes the 8.00x framing says outright that 8.00x is NOT the deployed path's cost; 1.13x is.
- Class: (b) stale by later regeneration, but still shown as of 2026-09-20 with no correction. Severity:
  CRITICAL -- this is the deck's central cost claim, off by ~7x from the number describing the actual
  serving stack.
- Fix: add a line stating the vLLM shared-prefill FLOP-eq is 1.13x, cite bestofn_vllm_2026-09-16.json,
  and reclassify "8.00x" as the HF-repeat-8 "as-charged" convention, not the deployed cost.

### HIGH

F1-c. Headline "+0.0736 / 6 of 8" is judge-currency only; not captioned at the point of claim; the
currency-matched picture is a TIE.
- HTML lines 193-194 (cover fig): "+0.0736" / "6 / 8 benchmarks beaten" -- no "judge currency" qualifier
  on the cover slide. Line 656 (closing line) repeats it, again uncaptioned. (Lines 245, 413 do say
  "32B-judge" elsewhere in the deck, but not attached to the headline figures.)
- Cross-check (independently re-run, script+log present):
  /home/jamesyang/.claude/jobs/37d73e6f/tmp/em-rescore/em_rescore_pooled_probe_2026-09-18.json:
  delta_em.macro = 0.00470598904248082, CI [-0.00774,+0.01711], verdict_image_clustered "TIE" (frozen 24
  probes, no refit). +0.0736 matches head_final_stack_PVFIXED_2026-09-13.json exactly in judge currency,
  but the same picks score only +0.0047 (TIE) under the project's own normalised exact match.
- Class: (d) framing/omission. Severity: HIGH.
- Fix: caption cover-slide fig and closing line "(32B-judge currency; +0.0047 [-0.0077,+0.0171] TIE under
  exact match)".

F1-d. The judge is never named, and "32B judge" next to "Lingshu-7B" invites the reader to assume it is a
Lingshu model. It is not.
- HTML lines 176 (eyebrow: "Lingshu-7B . best-of-8 . 32B judge"), 245 ("a 32B-judge label"), 413 ("in
  32B-judge currency").
- Code: src/labeling/run_judge.py line 21: judge_model default =
  "/data/dan/weights/MedVLThinker-32B-RL_m23k" -- the June-era MedVLThinker model, not Lingshu. Line 4
  docstring already says "NOT the model scored in the Lingshu cascade". Confirmed: of the 21 runner .sh
  scripts invoking run_judge.py, zero pass --judge_model. So every label in this project, including every
  number in this deck, comes from MedVLThinker-32B-RL_m23k, not Lingshu-32B.
- Class: (d) framing/misleading-by-omission. Severity: HIGH.
- Fix: name the judge explicitly at first use: "32B judge (MedVLThinker-32B-RL_m23k, not Lingshu-32B)".

### MEDIUM

F1-e. "Selection wall" cluster (oracle headroom, macro selection efficiency, random floor, self-fit
ceiling) does not match any located current artifact; one number looks conflated with an unrelated stat.
- HTML lines 606-609: "Mean oracle headroom above greedy is +0.1650; we convert +0.0736. Macro selection
  efficiency is 0.686 against a random-pick floor of 0.579. Fitting the same probe on the evaluation
  labels themselves reaches 0.997."
- The only artifact-backed "0.686" found anywhere in the project is GEMeX's domain-classifier confidence
  (OPENTEXT_FULL_RUNDOWN_2026-09-04.md S5.5; AUDIT_2026-09-12.md S5 correction "domclf confidence
  0.693->0.686") -- a single-benchmark OOD metric, not a macro selection efficiency over 8 benchmarks.
- My own recomputation of mean oracle headroom from head_final_stack_PVFIXED_2026-09-13.json (greedy) +
  coverage_scaling_ALL_2026-09-01.json (oracle@8 per cell) gives +0.1098, not +0.1650.
- Class: (c) unsourced / possible conflation. Severity: MEDIUM-HIGH.
- Fix: name the artifact for all four numbers, or mark as estimate pending recomputation.

F1-f. Coverage-wall percentages (71.5%, 21.2%) are reused from a stale, out-of-scope, 3-benchmark study
and presented as describing the current 8-benchmark, 36,869-question pool.
- HTML line 600-602: "Across all 36,869 questions, 44.1% have no correct answer anywhere in the
  8-candidate set -- 12.1% on SLAKE to 79.0% on VQA-Med. Not a diversity problem: on 71.5% of them not
  one gold token appears anywhere in the set, and tripling the budget rescues only 21.2%."
- results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json title: "SCOUT B -- COVERAGE
  DIAGNOSIS (open-text best-of-N, Lingshu-7B, n=2,345)" -- PathVQA-1500+SLAKE+VQA-RAD only, dated
  2026-08-10, before the 5 new benchmarks existed and before the PathVQA-truncation fix. Its field
  "4_they_are_not_near_misses": "71.5% of no-coverage questions share ZERO tokens with gold anywhere in
  the pool"; field "3_it_is_a_CAPABILITY...": "An INDEPENDENT 16-sample redraw (3x the total budget)
  rescues only 21.2% of them." Both match the deck exactly, but for a pool 1/16th the claimed size.
  44.1%/12.1%/79.0% were not located in any artifact in the time available.
- Class: (c) unsourced for the stated scope / likely (a) wrong when written. Severity: MEDIUM-HIGH.
- Fix: recompute 71.5%/21.2% on the current pool, or scope the sentence to the original 3 benchmarks.

F1-g. MedGemma row shown "incomplete/provisional" (4/8, +0.0248) has since been completed (2026-09-16, 2
days after this deck); the completed number complicates the "replicates across 3 families" framing.
- HTML lines 573-584 ("Treat the MedGemma row as provisional").
- results/cascade_methods/artifacts/head_final_stack_medgemma_ALL8_2026-09-16.json: best config
  (pooled_singlelayer) macro +0.0481, 8/8; but the "shipped" 3-layer-ensemble config used for the
  Lingshu/Qwen headlines (pooled_ens) is only +0.0447, 7/8 for MedGemma -- the same-recipe-as-shipped
  number is not 8/8. Picking whichever config is best per generator risks an apples-to-oranges
  comparison against Lingshu/Qwen's fixed pooled_ens recipe.
- Class: (b) stale (deck predates the completed run). Severity: MEDIUM.

### Terminology / novelty checks (09-14 deck)
- "cell", "MLP head", "WIN/TIE/LOSS" (tokens banned by TERMINOLOGY_2026-08-24.md): zero occurrences --
  clean. "pool"/"pools" occurs twice (lines 267, 438), both as "training pool"/"a disjoint pool", which
  the terminology doc explicitly permits ("candidate pool is also used").
- No explicit "novel"/"first"/"no one has" language found. The only forbidden-novelty issue is the
  pervasive "free" framing already covered under F1-a.

### Verified-clean (09-14 deck)
- Cover headline +0.0736 macro, 6/8 -- exact match to head_final_stack_PVFIXED_2026-09-13.json
  (macro.pooled_ens=0.07360735..., beats_greedy.pooled_ens="6/8").
- 8-benchmark table (lines 231-239): every n_questions/held-out/train-half count is an exact match to
  head_final_stack_PVFIXED_2026-09-13.json cells and to the recipe's rows_per_benchmark_half. Totals
  (36,869/18,452/18,417) are internally self-consistent (independently summed by hand).
- Selection-skill/penalty (line 604: +0.0387/+0.0257, "7 of 8") -- exact match to
  decomposition_2026-08-24.json summary_T07 (mean_selection_skill=0.038712..., mean_sampling_penalty=
  0.025680..., cells_with_positive_selection_skill="7/8"); this file already carries the PathVQA-fixed
  n=3,357.
- "What the extra data bought" ladder (lines 485-488: +0.0182/+0.0729/+0.0736/+0.0720, "6/8"x3+"7/8") --
  exact match to head_final_stack_PVFIXED_2026-09-13.json macro.* fields.
- SC-feature-once-pooled figure -0.0016 (line 557) -- exact: pooled_ens_sc (0.071972...) minus pooled_ens
  (0.073607...) = -0.0016351, i.e. correctly computed from the CURRENT PathVQA-fixed artifact (more
  current than AUDIT_2026-09-12's -0.0004, which itself predates the 09-13 PathVQA fix).
- "Both verifiers, identical questions" table (lines 420-425, 6 rows x 5 metrics + 3 cost cols, 48 cells)
  -- every value is an exact match to free_head_2026-08-16.json (baselines.always_7b_greedy,
  arms_all_feature_sources.incumbent/head_only::deployed/fusion::deployed,
  DELIVERABLE_4_head_only.arms_on_the_deployed_features.self_consistency_free_control, baselines.
  oracle_at_8) and bestofn_latency_energy_2026-08-03.json for FLOP/latency/energy. This table also
  correctly self-discloses (lines 441-444) that it still uses the truncated 1,500-q PathVQA, and explains
  why -- an accurate, well-hedged caveat.
- Cost-table latency/energy (360ms/55.9J greedy; 717ms/165.1J probe; 1,326ms/322.7J LoRA;
  1.99x/2.95x/3.68x/5.77x) -- exact match to bestofn_latency_energy_2026-08-03.json.
- GFLOP figures (5,692/45,538/86,435, 15.18x LoRA) -- exact match to free_head_2026-08-16.json
  DELIVERABLE_3_cost.arms_fixed_bestof8.

### Could not verify (09-14 deck)
- "44.1%/12.1%/79.0%" per-cell coverage-failure percentages (line 600) -- no matching artifact located.
- "+0.1650/0.686/0.579/0.997" selection-wall cluster (lines 606-609) -- see F1-e.
- "+0.0099 on the weak base" self-consistency figure (line 557 first half) -- closest is recipe.json's
  +0.0098 (0.0001 off); not resolved to an exact source in time available.
- Domain breadth "+0.0087 [-0.0014,+0.0200]" (543), regime-detection "no ordering" (547), ensemble-width
  "-0.0011" (551) -- not individually traced to a specific artifact key; plausible, not verified.

Counts (09-14 deck): ~85 distinct numeric claims checked; exact match: ~65 (incl. two full tables
cross-checked cell-by-cell, 78 cells total); stale-by-later-regeneration: 2 (F1-b, F1-g);
wrong/misapplied-scope: 1 (F1-f); unsourced/could-not-verify: ~8 (F1-e cluster + minor ablations);
framing violations: 1 pervasive ("free", 5+ occurrences) + 1 omission pattern (judge currency) + 1 naming
ambiguity (judge identity, 3 occurrences).

## FILE 2: meetings/shipped_method_2026-09-12.html -- EXHAUSTIVE

### CRITICAL

F2-a. Qwen replication number is stale: deck states +0.0788 / "8 of 8" / 222,154 training rows,
explicitly labelled "Updated 13 Sep"; the artifact this cites was itself superseded the SAME DAY.
- HTML line 207 (cover-slide fig): "+0.0788 ... beating its own greedy on 8 of 8". Line 310 (deep-dive
  callout): "Updated 13 Sep -- the 8-benchmark run finished. Macro +0.0788, beating Qwen's own greedy on
  8 of 8, over 222,154 training rows."
- results/cascade_methods/artifacts/head_second_generator_ALL8_2026-09-12.json (mtime 2026-09-13
  16:29:36) carries its own field "SUPERSEDED_2026-09-13": "Its training set was up to 4x duplicated...
  The train_rows 222,154 recorded here contains ~77,000 duplicate (ds, idx, candidate) rows -- the
  deduplicated count is 145,085. Superseded by head_final_stack_qwen_2026-09-13.json... The corrected
  macro is +0.0820 (8/8)."
- results/cascade_methods/artifacts/head_final_stack_qwen_2026-09-13.json (mtime 2026-09-13 16:26:54,
  i.e. BEFORE the superseded-notice was appended to the old file at 16:29:36) gives macro.pooled_ens =
  0.0820349630356359, beats_greedy.pooled_ens = "8/8", pooled_rows = 145085.
- The deck HTML file's own mtime is 2026-09-14 02:26:44 -- AFTER both artifacts above -- so the deck was
  last saved after the correction existed on disk, and still carries the pre-correction number.
- Class: (b) stale by later regeneration, worse because the deck's own save postdates the correction.
  Severity: CRITICAL -- appears on the COVER slide as a headline stat plus a deep-dive callout, both
  wrong at the same precision (+0.0788 vs correct +0.0820; 222,154 vs correct 145,085 rows). "8 of 8" is
  unaffected (correct either way).
- Fix: replace +0.0788 -> +0.0820 and 222,154 -> 145,085 in both locations (lines 207, 310); cite
  head_final_stack_qwen_2026-09-13.json.

### Terminology / novelty checks (09-12 deck)
- "cell", "MLP head" (literal), "WIN/TIE/LOSS": zero occurrences -- clean; deck consistently says
  "benchmark" and "the probe"/"probe verifier".
- "pool" occurs once (line 286): "a bigger pool, a colder pool" -- candidate pool, permitted usage.
- "free" occurs 4x: SVG diagram label "free" (line 224, describing the 0-extra-forward-pass scoring step
  only); line 275 "worth taking because it is free" (the layer-ensemble costing nothing, not the whole
  method); line 276 "shipping it would buy an inference-time dependency for free" (idiomatic, "for
  nothing"); line 286 "The probe is free; the 8 samples are not" -- this is the deck's OWN explicit
  caveat distinguishing the scoring step (free) from the real cost (the 8x sampling), which is materially
  more honest than the 09-14 deck's undifferentiated "The Free Verifier" framing. This deck predates
  PRIOR_ART_PROBE_VERIFIER_2026-09-14.md by 2 days, so class is (b)/not-yet-known rather than (a); given
  its own explicit hedge at line 286, severity: LOW (contrast noted, not flagged as a primary violation).
- No "novel"/"first" language found.

### Judge naming / currency
- No "Lingshu-32B" and no "32B judge" phrasing found; the deck says only "the judge" / "judging" generically
  (line 316) without naming a model at all -- so it does not misname the judge, but it also never
  discloses that the judge is NOT Lingshu-32B. "Judge currency throughout" is stated explicitly twice:
  line 261 (attached directly under the main 8-benchmark results table) and line 329 (closing "Standing
  rule"). The cover slide (line 205, "+0.0736 macro... 6 of 8 benchmarks") does NOT carry the qualifier
  inline, but the deck as a whole discloses currency far more consistently than the 09-14 deck.

### Verified-clean (09-12 deck)
- Cover headline +0.0736 macro, 6/8 (line 205) -- exact match, same source as F1 above.
- Full 8-benchmark performance table (line 258-259: PathVQA 1,623/0.3050/0.3580/+0.0530; SLAKE
  330/0.7121/0.7667/+0.0545; VQA-RAD 97/0.5052/0.4639/-0.0412; RadImageNet 1,004/0.3337/0.4592/+0.1255;
  Kvasir 5,152/0.2811/0.4022/+0.1211; OmniMed 4,461/0.5216/0.6783/+0.1567; VQA-Med 1,807/0.0913/0.0897/
  -0.0017; GEMeX 3,978/0.3997/0.5206/+0.1209; MACRO 18,452/--/--/ +0.0736) -- every one of these 36
  figures is an exact match to head_final_stack_PVFIXED_2026-09-13.json cells (n_questions, greedy,
  pooled_ens, pooled_ens_minus_greedy).
- "Four changes, measured one at a time" ladder (line 268-271: +0.0182 6/8; +0.0729 6/8; +0.0736 6/8
  SHIPPED; +0.0720 7/8, "+0.0098 before pooling, -0.0004 after") -- exact match to
  head_final_stack_PVFIXED_2026-09-13.json macro.* fields; the -0.0004 figure here matches the
  AUDIT_2026-09-12 S5-corrected value (unlike F1-line-557's -0.0016, which is the newer PVFIXED-derived
  figure -- both numbers are individually traceable and internally explained by their own dated context,
  not simply contradictory).
- "Training data did the work +0.0547 of the +0.0554 total" (line 274) and "ensemble... +0.0007" (line
  275) -- 0.0729-0.0182=0.0547 and 0.0736-00.0729=0.0007 both arithmetically exact from the table above.
- Leave-one-benchmark-out figures (line 296: "-0.0008" breadth-alone, "+0.0500" own-half) and two-loss
  diagnosis table (lines 290-293) are consistent in shape with (though not numerically identical to,
  since LOBO uses a different artifact) the briefing's cited head_lobo_pooled_2026-08-25.json breadth
  -0.0056/own-half +0.0485 -- NOT flagged as a mismatch since this deck's LOBO figures (-0.0008/+0.0500)
  are close but not identical to the briefing's cited numbers; I could not confirm a matching artifact
  key for -0.0008/+0.0500 specifically in the time available (see Could-not-verify).
- Qwen 4-benchmark provisional table (lines 303-307: Kvasir +0.1064, SLAKE +0.0727, PathVQA +0.0308,
  VQA-RAD +0.0000, MACRO +0.0525) is explicitly labelled provisional/superseded by the "Updated 13 Sep"
  paragraph immediately below it -- internally consistent structure, only the update itself is wrong
  (F2-a).
- PathVQA-truncation restatement (line 326: "+0.0802 -> +0.0736", "108,126 -> 112,770", "-0.0065") --
  matches the 09-04 rundown's own restatement table exactly (+0.0802/+0.0736, 108,126/112,770).
- Provenance/audit claims (line 315: "380,677 question-text comparisons... 1,366,424 judged-slot
  comparisons: zero mismatches... 175 of 182 checkable claims verified exact"; line 319-320: "thread
  count... macro spans 0.0033 on one arm") -- not independently re-run (would require re-executing the
  audit script), but internally consistent with AUDIT_2026-09-12.md's own account and not contradicted by
  anything found; not flagged.

### Could not verify (09-12 deck)
- LOBO figures "-0.0008" / "+0.0500" (line 296) -- plausible but not traced to an exact artifact key
  (closest located, head_lobo_pooled_2026-08-25.json, gives -0.0056/+0.0485 per the briefing, a
  different pair -- could be a different LOBO run/date; not resolved in time available).
- "175 of 182 checkable claims... verified exact" and "380,677 / 1,366,424" comparison counts (line 315)
  -- not independently reproduced.

Counts (09-12 deck): ~55 distinct numeric claims checked; exact match: ~50 (incl. one full 8-row x 4-col
table = 36 cells, cross-checked exactly); stale-by-later-regeneration: 1 (F2-a, appearing twice: cover +
callout); unsourced/could-not-verify: ~3; terminology: clean; framing: 1 LOW note (contrast with F1-a,
not a primary violation).

## FILE 3: meetings/opentext_rundown_2026-09-04.html

**Supersession banner: PRESENT and thorough.** Lines 163-179+: a boxed note.warn, "Restated 13 Sep 2026
-- read before quoting", explaining the PathVQA 1,500/3,357 truncation, citing
head_final_stack_PVFIXED_2026-09-13.json and coverage_scaling_ALL_2026-09-01.json by name, with a full
before/after table (four-domain +0.0243->+0.0182; pooled-single-layer +0.0765->+0.0729; shipped
+0.0802->+0.0736; +ensemble+SC +0.0797->+0.0720; training rows 108,126->112,770; reload-verified
+0.0816->+0.0737; doubling budget +0.0164->+0.0147; PathVQA's own share of that +0.0157->+0.0018).
File mtime 2026-09-14 02:25:49 -- i.e. this HTML was regenerated/patched AFTER
AUDIT_2026-09-12.md's own S5b adversarial recheck (written 2026-09-13, which explicitly said "the HTML
rundown... were NOT updated with these corrections"). That statement in AUDIT S5b is itself now STALE
for this file: the fix landed later the same window.

**AUDIT_2026-09-12.md S5 correction checklist, verified against the CURRENT file:**
| item | published (old) | correct | found in current HTML | status |
|---|---|---|---|---|
| corpus size | 30,912 | 35,012 | line 142 "35,012 questions"; line 217 "2,345 -> 35,012" | FIXED |
| beats greedy at k=0 | 6/8 | 5/8 | line 277 "beats greedy 5/8" (row 1 of 4) | FIXED |
| union loses to best single T | 6 of 7 | 6 of 8 | line 313 "loses to the best single temperature on 7 of 8" -- see note below | SEE NOTE |
| domclf confidence | 0.693 | 0.686 | line 166 "lowest domain-classifier confidence 0.686" | FIXED |
| greedy-anchored | +0.0007/ceiling+0.0022 | +0.0006/+0.0023 | line 316 "+0.0006; ceiling +0.0023" | FIXED |
| SC feature once pooled | -0.0005 | -0.0004 | line 291 "-0.0004 once pooled" | FIXED |
| GEMeX distinct golds | 4,316 | 4,315 | line 248 "4,315 distinct golds" | FIXED |
| answer-kind mean | +0.0054 | +0.0045 | line 318 "mean only +0.0045" | FIXED |
| onboarding cost | ~500 vs ~100 | ~100 (LOBO base) | line 342 heading "~100 labelled questions"; line 406 "~100 labelled questions" | FIXED |

Note on "union loses...7 of 8": AUDIT S5b (the adversarial recheck, written AFTER the main S5 table) says
the corrected value is actually "7 of 8" (not "6 of 8" as S5's own row implies: "PathVQA flipped when its
n went 1,500 -> 3,357"). The current HTML (line 313) says "7 of 8" -- so it matches the S5b-corrected
figure, not the S5-table figure. Verified via decomposition context this is consistent with the
PathVQA-fixed regime. FIXED (against the more authoritative S5b number).

**Conclusion for this file: all 9 AUDIT S5 correction items are present and correct in the current HTML.**
This deck is the best-maintained of the four.

**Other spot-checks (09-04 rundown):**
- Line 364: "A stale checkpoint invalidated every OmniMedVQA number... Corrected: -0.0489 -> -0.0193" --
  -0.0193 matches head_final_stack_PVFIXED_2026-09-13.json omnimed_open pooled_ens_minus_greedy exactly
  is NOT what's stored there (PVFIXED gives omnimed +0.1567, a different, later-recipe number) -- this
  -0.0193 is from an EARLIER recipe (single-domain/4-domain baseline), consistent with the deck's own
  framing of it as a specific historical bug-fix anecdote, not the shipped headline. Not flagged as
  inconsistent; different metric, correctly scoped.
- Terminology: "cell"/"MLP head"/"WIN/TIE/LOSS": zero occurrences (spot-checked via grep) -- clean.
- "Lingshu-32B" / "32B judge": zero occurrences (checked earlier, full-file grep) -- clean, no judge
  misnaming in this file.

**Could not verify (09-04 rundown):** did not re-derive every one of the ~60 additional per-benchmark
figures in this longer document (S1-S6 detail tables) against artifacts individually; the 9 AUDIT-listed
corrections and ~6 spot-checks above are all that were traced to source in the time available. No
contradictions found in what was checked.

Counts (09-04 rundown): 9/9 AUDIT S5 corrections verified fixed; ~10 additional spot-checked figures, all
exact; 0 stale/wrong found; terminology and judge-naming clean.

## FILE 4: meetings/progress_deck_2026-08-24.html

**Supersession banner: ABSENT.** No "restated"/"corrected"/"superseded"/"read before quoting" language
found anywhere in this file (grepped for all of those terms plus "updated"/"stale" -- zero hits). File
mtime 2026-09-13 12:40:10 (edited same day as the AUDIT correction pass, but before the PathVQA-fix
artifact regeneration at 14:41). This deck's headline framing (the T=0.7-only "4 wins, 2 ties, 2 losses"
recipe from decomposition_2026-08-24.json/free_signal_bakeoff_2026-08-21.json) PREDATES the
"pooled-training-across-all-8-benchmarks" recipe that produces the now-canonical +0.0736/6-of-8 headline
used in the other three decks -- this deck never cites that later recipe at all, so it is not internally
"wrong", but a reader encountering only this file (still on disk, no forward pointer) would not know a
materially different, later, better-performing recipe (pooled_ens) superseded the one shown here 3 days
after this deck's nominal date. This is a structural gap, not a numeric error.

**AUDIT_2026-09-12.md S5 correction checklist, verified against the CURRENT file:**
| item | published (old) | correct | found in current HTML | status |
|---|---|---|---|---|
| domclf confidence | 0.693 | 0.686 | line 254 "lowest domain-classifier confidence (0.686)" | FIXED |
| greedy-anchored | +0.0007/ceiling +0.0022 | +0.0006/+0.0023 | line 301 "+0.0006... at most +0.0023" | FIXED |
| answer-kind mean | +0.0054 | +0.0045 | line 303 "6/8 benchmarks... mean is +0.0045" | FIXED |
| GEMeX distinct golds | 4,316 | 4,315 | line 237 "4,315 distinct gold answers" | FIXED |
| union loses to best single T | 6 of 7 | -- | line 298 "worse on 7 of 8 benchmarks" | FIXED (matches 7/8) |
| corpus size | 30,912 | 35,012 | not stated as a single "corpus" figure; line 205 gives "36,869 questions" (the full-suite total, consistent with the other 3 decks) | N/A -- different framing, no discrepancy |
| beats greedy at k=0 | 6/8 | 5/8 | not present in this deck (this deck's own headline is "4 wins, 2 ties, 2 losses" on a different recipe, not a "beats greedy at k=0" doubling-curve claim) | N/A |
| SC feature once pooled | -0.0005/-0.0004 | -- | not present (this deck predates the pooled-SC-feature ablation) | N/A |
| onboarding cost | ~500/~100 | ~100 | not present (this deck predates LOBO onboarding-cost framing) | N/A |

**Conclusion for this file: the 5 correction items that apply to this deck's scope are all fixed; 4 of
the 9 AUDIT items are about a later recipe/analysis this deck never discusses, so they are not
applicable rather than missing.** No numeric violations found. The only issue is the missing forward
pointer/banner given the deck's headline recipe is now superseded.

**Other checks (08-24 deck):**
- Full 8-benchmark table (line 242) -- every value (n, greedy, answer-prior, self-consistency, verifier,
  delta, oracle@8) traced to decomposition_2026-08-24.json / free_signal_bakeoff_2026-08-21.json T=0.7
  cells; PathVQA row already reflects n=3,357 (post-fix) with greedy=0.3140 matching
  decomposition_2026-08-24.json cells.pathvqa_open."0.7".greedy exactly -- confirms this file, despite no
  banner, already carries corrected PathVQA data.
- Decomposition table (line 274-276: skill/penalty per benchmark, "7 of 8... mean +0.0383... +0.0257") --
  exact match to decomposition_2026-08-24.json summary_T07 field values (0.0383 here vs 0.0387 in the
  09-14/09-12 decks -- BOTH exist in different fields of the same artifact and are not a contradiction;
  not flagged).
- Terminology: "cell"/"MLP head"/"WIN/TIE/LOSS": zero occurrences -- clean (uses "benchmark", "head" is
  used generically a few times e.g. "the head can do" -- borderline but not the banned "MLP head"/"the
  head" pattern the TERMINOLOGY doc specifically flags as a first-mention issue; not flagged as a hard
  violation since "probe"/"MLP probe" is used at first mention, line 212).
- "Lingshu-32B" appears ONCE: line 224, "The ask: More open-text medical data, Lingshu-32B as the
  baseline, and make the MLP probe as good as it can be." This is a retrospective quote of an instruction
  Leo gave, referring to Lingshu-32B as an ACCURACY BASELINE/comparison point (a real, correctly-named
  Lingshu model used elsewhere in the project as the strong-arm baseline), not as the JUDGE. Not the same
  misnaming pattern as F1-d (which is about the judge specifically) -- noted for completeness, not flagged
  as a violation.

Counts (08-24 deck): 5/5 applicable AUDIT S5 corrections verified fixed (4 more are out of this deck's
scope, not applicable); ~40 additional figures spot-checked via the two main tables, all exact; 0
stale/wrong numeric findings; 1 structural finding (missing supersession banner, MEDIUM severity, no
numeric error but the deck's whole headline recipe is superseded and unflagged).

## Overall summary

**Ranked by severity across all 4 files:**
1. CRITICAL -- F1-a: 09-14 deck's title/thesis "The Free Verifier" + 5 "free" claims directly contradict
   the project's own PRIOR_ART_PROBE_VERIFIER_2026-09-14.md, written 26 minutes earlier the same day.
2. CRITICAL -- F1-b: 09-14 deck's headline cost claim "8.00x FLOP-eq" omits the 1.13x true cost of the
   actual vLLM shared-prefill generation path (bestofn_vllm_2026-09-16.json), a ~7x overstatement.
3. CRITICAL -- F2-a: 09-12 deck's Qwen replication headline (+0.0788, 222,154 rows) on the COVER slide is
   the exact number its own source artifact self-flags as SUPERSEDED the same day; correct is +0.0820,
   145,085 rows. Deck's file save postdates the correction.
4. HIGH -- F1-c: 09-14 deck's headline (+0.0736, 6/8) is judge-currency only and uncaptioned at point of
   use; true EM-currency result is +0.0047 [-0.0077,+0.0171], a TIE (independently re-verified,
   em-rescore/em_rescore_pooled_probe_2026-09-18.json).
5. HIGH -- F1-d: "32B judge" (09-14 deck, 3x) never names the model and, placed next to "Lingshu-7B",
   invites readers to assume Lingshu-32B. It is MedVLThinker-32B-RL_m23k (confirmed via
   src/labeling/run_judge.py default arg + all 21 calling runners passing no override).
6. MEDIUM-HIGH -- F1-e/F1-f: unsourced/likely-misapplied-scope "selection wall" and "coverage wall"
   headline numbers in the 09-14 deck (lines 600-609).
7. MEDIUM -- F1-g (MedGemma row now stale) and the 08-24 deck's missing supersession banner (structural,
   no numeric error).

**What is clean:** terminology (cell/MLP head/WIN-TIE-LOSS) is zero-violation across all 4 files; "pool"
usage is compliant everywhere; the 09-04 rundown is fully corrected against all 9 AUDIT_2026-09-12 S5
items and carries a thorough, well-sourced supersession banner; the two large side-by-side comparison
tables in the 09-14 and 09-12 decks (84 cells total) are byte-exact against their cited artifacts; the
08-24 deck's 5 applicable AUDIT corrections are all fixed even though it lacks a banner.

**Overall counts:** ~200 distinct numeric/factual claims checked across 4 files. Exact match: ~155.
Stale-by-later-regeneration: 5 (F1-b, F1-g, F2-a-cover, F2-a-callout, MedGemma-adjacent). Wrong or
misapplied-scope: 1 (F1-f). Unsourced/could-not-verify: ~14. Framing/terminology/novelty violations: 3
distinct patterns (the pervasive "free" claim, the uncaptioned judge-currency headline, the ambiguous
judge naming), concentrated entirely in the 09-14 deck -- the deck actually shown to the professor is the
least accurate of the four on these dimensions, despite being numerically the best-sourced on its core
tables.
