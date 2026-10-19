# R3 — final professor check (round 3)

Inputs: DECK_SPEC.md, out/slide_text.md (current), out/..._script.md (current), all 72 shots (06:27, viewed as 2×2 sheets in
review/r3/m00–m17.png), R_*.md / R2_*.md (not re-reported unless still open). Every number below is copied from the file
named next to it; "derived" marks my own arithmetic.

**Verdict.** The deck is in good shape: the R2 HIGH items (new-benchmark wording, cost accounting, answer prior per benchmark,
exposure, 32B vs Mirage) are fixed on the slides. **Three HIGH items remain.** One is a count contradiction on the headline
slide. One is a mislabelled training set that a professor will expose with a single question. One is an R2 fix that reached
the slides but not the script. Everything else is MED or LOW.

---

## HIGH

### H1 — Lenient EM "loses on 2" (slides 49, 62 + script 49, 62) vs "loses on 3" (slides 2, 50)
`final_eval_FINAL_cap320_T07.json` cells.*.all.ci_pick_minus_greedy_em shows three significant EM losses:
Kvasir-x1 −0.0381 [−0.0441, −0.0322], VQA-Med −0.0137 [−0.0227, −0.0049] and **GEMeX −0.0091 [−0.0162, −0.0021]**.
Slide 2 says "loses on 3 benchmarks" and slide 50 lists all three. Slide 49 ("loses on Kvasir-x1 and VQA-Med") and slide 62
("lenient EM small, loses on 2") leave out GEMeX. Because slide 62 is the limits slide, the omission also understates the harm.
- Slide 49, "lenient exact match" box: → "small gain; significant losses on Kvasir-x1, GEMeX, VQA-Med (GEMeX becomes a tie after Bonferroni ×8)".
- Slide 62, "judge-sensitive" card: "lenient EM small, loses on 2" → "lenient EM small, loses on 3 (Kvasir-x1, GEMeX, VQA-Med)".
- Script 49, last sentence: "and per benchmark it loses on Kvasir and VQA-Med" → "and per benchmark it loses significantly on three: Kvasir, GEMeX and VQA-Med".
- Script 62: "Exact match loses on Kvasir and VQA-Med." → "Exact match loses on Kvasir, GEMeX and VQA-Med; on Kvasir and GEMeX the golds are sentences."

### H2 — The "4 extra domains" are not general domains. Three of them are training splits of benchmarks we score, and slide 44's leave-out bar is optimistic on 4 of 8 folds, not 1
`numbers_review_part2.json` extra_domains.rows_by_domain = **pathvqa_open_train 22,390 · kvasir_open 5,562 · slake_open_train
2,207 · vqa_rad_open_train 1,339** (6,029 questions, 31,498 rows). The 4-domain probe in `head_lobo_pooled_2026-08-25.json` uses
31,439 rows, i.e. this same set. Consequences:
- The script calls them **"four general domains"** in four places: line 51 (slide 2 Q&A), slide 44, slide 62 and slide 63. That is wrong, and a professor who asks "which four?" will get the answer "the training splits of PathVQA, SLAKE, VQA-RAD, plus Kvasir-VQA".
- Slide 44's footnote flags only Kvasir-x1 ("keeps same-source Kvasir rows"). But the PathVQA, SLAKE and VQA-RAD leave-out folds also keep their own benchmark's **training split** (PathVQA lobo_rows 104,550 still include the 22,390 PathVQA-train rows). So 4 of the 8 leave-out folds are not zero-label.
- The conclusion **survives, and gets stronger**, on the four benchmarks with no training split in that set (derived from the same artifact, mean of `*_minus_greedy` over RadImageNet, OmniMedVQA, VQA-Med, GEMeX): 4-domain **+0.0147**, leave-out **+0.0033**, pooled **+0.0985**. On the four same-source benchmarks, all three settings are already about the same (+0.0410 / +0.0412 / +0.0430). So the whole pooling lever comes from benchmarks that had no labelled data before pooling. This is the cleanest evidence for "a new benchmark needs labels".

Fixes:
- Slide 44 bar label "4 training domains only" → "4-domain set only (PathVQA/SLAKE/VQA-RAD train splits + Kvasir-VQA)".
- Slide 44 footnote: "the Kvasir-x1 leave-out fold keeps same-source Kvasir rows (optimistic)" → "the PathVQA, SLAKE, VQA-RAD and Kvasir-x1 leave-out folds keep their own source's training split (optimistic); on the 4 benchmarks without one: 4-domain +0.0147, left out +0.0033, pooled +0.0985 (derived, same file)".
- Slide 26 "probe training data" box and slide 31 "training rows": "4 extra domains" → "4 extra sets (PathVQA/SLAKE/VQA-RAD training splits + Kvasir-VQA)".
- Script 44: "a probe trained only on four general domains" → "a probe trained only on the four-domain set — the training splits of PathVQA, SLAKE and VQA-RAD plus Kvasir-VQA". After "does no better on it than the four-domain probe" add: "On the four benchmarks that have no training split in that set, the gap is stark: left out +0.003, pooled +0.099."
- Script line 51 (slide 2 Q&A), script 62 and script 63: "four general domains" → "the four-domain set".
- Script 26: "plus about six thousand questions from four extra domains" → "plus about six thousand questions from the training splits of PathVQA, SLAKE and VQA-RAD and from Kvasir-VQA".

### H3 — The R2 #5 / shared-rule 5 fix is on slide 37 but not in the script
- Script 37: "it beats every training-free way of choosing, it ties a much larger untrained verifier" contradicts itself (a zero-shot 32B is training-free). It also contradicts slide 64, which says USC, EBA, likelihood and self-certainty were not run, and slide 57, where WED ties on VQA-RAD.
  → "it beats majority vote, the answer prior, random and longest pick, and self-verification; it ties a much larger untrained verifier; and against the published selector WED it wins on two benchmarks and ties on the third. Likelihood ranking and USC were not run."
- Script 38, first line: "Every untrained way of choosing among the samples fails." → "Every untrained way of choosing that we ran on these pools loses to greedy or to the probe."

---

## MED

### M1 — Script 35 compares across harnesses: "more than the sampling itself"
The 6.90 re-read figure is HF accounting. In that same accounting, sampling is 8 passes (`free_head_2026-08-16.json`, 14.90× total), so the re-read is **less** than the sampling. It is "more" only when set against the vLLM figure of 1.13×, which is exactly the comparison shared-rule 4 forbids.
→ Script 35: delete "more than the sampling itself"; the sentence then reads "…about 6.9 greedy passes' worth of arithmetic per question in the HuggingFace accounting. That figure comes from a different measurement setup, so I do not add the two into one total."

### M2 — Prepared answer missing: "Is the judge just rewarding longer answers? Why do EM and the judge disagree?"
The script answers this only with "pick-longest loses 0.029" (slide 11). The stronger evidence exists and is not used. Add it to the slide 50 Q&A:
> "No sign of a length reward. Regressing the judge label on EM, token-F1 and log-length over 81,554 held-out candidates gives a negative length coefficient (−0.157). Where the pick is not longer than greedy (72% of questions), the judge gains +0.0699 and EM gains +0.0296; where the pick is longer, the judge gains +0.1303 and EM loses −0.0648. About 32% of the upward judge flips have zero token overlap with the gold. Read by hand (36 sampled), most are correct against a verbose gold ('Oculus dexter' vs 'right eye'); a minority are judge errors ('2.5 cm' accepted for '11 to 20 millimeters'). September probe, image half A (AUDIT_2026-09-18.md §2.2, judge_length_bias_2026-09-20.json)."

### M3 — Prepared answer missing: "Does the 4-domain probe already see PathVQA?"
Answer from H2: yes, through the PathVQA/SLAKE/VQA-RAD training splits. Add it to the slide 44 Q&A, with the derived 4-benchmark numbers (+0.0147 / +0.0033 / +0.0985, head_lobo_pooled_2026-08-25.json cells.*).

---

## LOW

- **L1 — script line 50 (slide 2 Q&A):** "micro … (no CI computed)" contradicts slides 63 and 68, which show micro CIs. → "training judge +0.1097 [+0.1058, +0.1136], cross-family +0.0985 [+0.0944, +0.1025], lenient EM +0.0098 [+0.0064, +0.0131] (numbers_review_part3.json)".
- **L2 — script 53 "On screen":** it says "the prior's share per benchmark (top-right box)", but in the shot that box is bottom-left, under the table. → "(the orange box under the table)".
- **L3 — spec 12:** the script header promises an "On screen" cue for each slide. Content slides 9, 10, 11, 12, 16, 17, 19–23, 54, 56, 59, 60, 64, 65 still have none. The table slides need them most. Add one line each, e.g. 9 "the status column, top to bottom"; 19 "our row against the three above it"; 23 "column D, then our row"; 56 "the three tiers, then the orange box"; 59 "left table, then the 'not comparable' box"; 60 "group rows by the baseline column".
- **L4 — prepared answer, slide 51:** "How often does the probe change the answer?" The pick string differs from greedy on 22.9% (SLAKE) to 72.2% (VQA-Med) of questions; PathVQA 54.9%, OmniMed 52.0%, GEMeX 62.9% (numbers_review_part3.json pick_differs_from_greedy_string).

---

## Spec compliance (DECK_SPEC.md)
All 13 requirements and the standing rules are now **met**. The exceptions: req 1 terminology, where "four general domains" is a mislabel (H2); req 12, where on-screen cues are missing on some slides (L3); and the rule against judge-only numbers, which is satisfied. Temperature framing (req 8) and probes vs samples (req 9) are correct on 7, 33, 46, 65 and 69. No abstention framing anywhere. The cap320 re-extraction run that slides 47 and 65 call "running" is in fact running (logs/sv_c320eval_FINAL_cap320_C320.log, fold 0, layer 20 fitted at 06:30). If it finishes before the meeting, update slides 47, 65 and 23.

## Checked and fine (not re-reported)
Slide 8: macro bars 0.3936/0.4734/0.5586, so gap 0.0852 and wall 0.4414 ✓. Slide 7 chart matches head_temp_ensemble (2.03→5.12 distinct, 0.463→0.565) ✓. Worked example: ranks, 0.917/0.583, spleen −37.4/−7.2, "20 of 24" ✓. Slide 51: totals and 149 unavoidable harms ✓ (extra_numbers.json). Slide 57: n = 2,867/253/151 = 3,357−490 / 645−392 ✓. Slide 58: closed rows 2,246/2,123 ✓.
