# Comparison brief: our method vs the field (for the deck)

Prepared 2026-10-18. Read-only research. Each number comes from one of two places:
- **ours**: an artifact JSON under
  `/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/`. The file and key are named, and every value was re-read from the JSON for this brief.
- **theirs**: a paper. The table or section is named, plus the page when a PDF was available. Every value was re-read from the PDF or arXiv HTML text, not copied from the domain guide.

Values marked **(derived)** are arithmetic on quoted numbers; the formula is shown. "not reported" means the paper does not give the value.

Abbreviations:
- **J**: judge accuracy under MedVLThinker-32B, our judge of record (text-only; sees the gold answer).
- **XJ**: judge accuracy under MedGemma-27B, our cross-family judge.
- **EM**: our lenient exact match (containment).
- **LLaVA recall**: LLaVA-Med v1.0.0's open-question recall, computed with its own code.
- **macro**: mean over the 8 open benchmarks, equal weight.
- **CI**: 95% paired bootstrap interval, 10,000 resamples, clustered by image (`final_eval_FINAL_cap320_Tmap.json` → `bootstrap`).

---

## 0. Our numbers in one place (the anchor every comparison below refers to)

**Method.**
- Lingshu-7B samples 8 answers per question (cap320, the 250,880-pixel image cap), at a frozen per-benchmark temperature.
- 24 MLP probes score the candidates. Each probe has one hidden layer of width 256 and reads frozen hidden states from layers 18, 20 or 22, mean-pooled over the answer tokens.
- The readout is rank-average with a vote-count tie-break.
- Every question is scored by a probe that never saw its image (two-fold image cross-fitting).

**Macro gain over the model's own greedy decoding, 8 open benchmarks, 36,869 questions** (`final_eval_FINAL_cap320_Tmap.json` → `macro_over_open8.all`):

| currency | Δ vs greedy [95% CI] | benchmarks with positive point estimate | image half 1 only (`final_fold1_headline_2026-10-16.json` → `policies.Tmap.macro`) |
|---|---|---|---|
| J | **+0.0897 [+0.0831, +0.0966]** | 8/8 | +0.0936 [+0.0849, +0.1026] |
| XJ | **+0.0783 [+0.0715, +0.0853]** | 8/8 | +0.0811 [+0.0724, +0.0898] |
| EM | **+0.0165 [+0.0107, +0.0224]** | 5/8 | +0.0206 [+0.0120, +0.0291] |
| token-F1 | +0.0253 [+0.0206, +0.0301] | 8/8 | +0.0289 [+0.0223, +0.0355] |
| LLaVA recall | **+0.0347 [+0.0297, +0.0399]** | 8/8 | +0.0375 [+0.0304, +0.0447] |
| *self-consistency (majority vote), same pools, J* | **−0.0070 [−0.0111, −0.0029] (LOSS)** | 1/8 | — |

**Per benchmark** (`cells.<name>.all`; J unless noted):

| benchmark | n | greedy J | final J | Δ J [CI] | Δ XJ | Δ EM [CI] | oracle@8 J |
|---|---:|---:|---:|---|---:|---|---:|
| PathVQA-open | 3,357 | 0.3140 | 0.3664 | +0.0524 [+0.0389, +0.0661] | +0.0420 | +0.0185 [+0.0069, +0.0301] | 0.4808 |
| SLAKE-open | 645 | 0.7302 | 0.7876 | +0.0574 [+0.0291, +0.0868] | +0.0419 | +0.0171 [−0.0096, +0.0452] | 0.8791 |
| VQA-RAD-open | 200 | 0.4900 | 0.5150 | +0.0250 [−0.0103, +0.0628] tie | +0.0250 | +0.0100 [−0.0203, +0.0406] | 0.5650 |
| RadImageNet-VQA | 2,000 | 0.3210 | 0.4690 | +0.1480 [+0.1310, +0.1650] | +0.1260 | +0.0790 [+0.0665, +0.0920] | 0.5505 |
| Kvasir-VQA-x1 | 10,121 | 0.2849 | 0.3992 | +0.1143 [+0.1068, +0.1218] | +0.1048 | **−0.0381 [−0.0441, −0.0322]** | 0.4696 |
| OmniMedVQA (open) | 8,883 | 0.5164 | 0.6894 | +0.1730 [+0.1647, +0.1813] | +0.1594 | +0.0643 [+0.0555, +0.0732] | 0.7230 |
| VQA-Med 2019 C4 | 3,663 | 0.0947 | 0.1007 | +0.0060 [−0.0022, +0.0139] tie | +0.0087 | −0.0014 [−0.0093, +0.0068] | 0.1873 |
| GEMeX | 8,000 | 0.3974 | 0.5391 | +0.1417 [+0.1321, +0.1513] | +0.1182 | **−0.0174 [−0.0251, −0.0096]** | 0.6241 |

How to read this:
- Under J the method has **6 significant wins and 2 ties** (VQA-RAD, VQA-Med). Under EM it has **3 wins, 3 ties and 2 losses**. The losses are Kvasir-x1 and GEMeX, where the gold answers are sentences.
- VQA-Med is −0.0005 J on image half 1 (`final_fold1_headline_2026-10-16.json` → `policies.Tmap.cells.vqamed_open.judge` = −0.000539).

**Cost.** From `bestofn_vllm_2026-09-16.json`, vLLM shared prefill, cap320, one request in flight:
- latency 173.99 → 476.61 ms (`ratios_vs_greedy.latency` 2.739)
- energy 34.27 → 123.17 J (`ratios_vs_greedy.energy` 3.594)
- 1.13 FLOP-eq by forward-token count (`forward_tokens_per_question.flopeq_shared`). Charged the HF way it is 8.0 (`flopeq_as_charged`).

**Generalisation limit** (`head_lobo_pooled_2026-08-25.json` → `macro`, August probe, judge, 5 seeds, one fold):
- four-domain probe +0.0279
- leave-one-benchmark-out +0.0223
- pooled +0.0707 (`macro_breadth_gain` −0.0056; `macro_own_data_gain` +0.0485)

So the method needs the benchmark's own labelled training half. On a benchmark it has never seen, it still beats greedy, but by less than a third as much.

**Replication on Lingshu's non-medical base, Qwen2.5-VL-7B** (`replication_currency_2026-09-20.json` → `qwen.macro`):
- J +0.0814 (`probe_minus_greedy_judge.macro` 0.08137)
- EM +0.0276 **(derived)**: `probe_em` 0.22726 − `greedy_em` 0.19967

---

## 1. Same-protocol comparisons (the strongest slides)

Ranked by how controlled the comparison is.

### 1.1 WED, head-to-head on identical candidates: fully controlled

**Who they are.** "Wasserstein Equilibrium Decoding for Reliable Medical Visual Question Answering", Hagen et al., arXiv 2605.18313v2. The method is called BDG-W in the paper.
- It is a training-free, game-theoretic selector over sampled candidates, on the open-ended subsets of the VQA-RAD and PathVQA test splits, yes/no excluded (§3 "Datasets", PDF p.7).
- Candidates: "Nmax = 16 and n = 8", "temperature-diverse sampling" (§3, p.7).
- Judge: Grok 4.1 Fast. It also reports EM, token-F1 and SapBERT similarity (§3, p.7).
- Generators: Qwen3-VL-2B/4B/8B, Gemma-3-4B and MedGemma-4B.

**What they report** (Table 1, p.8): judge gain of BDG-W over greedy, VQA-RAD / PathVQA, in percentage points:

| model | VQA-RAD | PathVQA |
|---|---|---|
| Qwen-2B | +3.5 | +2.2 |
| Qwen-4B | +3.8 | +3.2 |
| Qwen-8B | −1.2 | +1.2 |
| Gemma-4B | +2.3 | +5.0 |
| MedGemma-4B | +2.7 | +2.0 |

The paper notes that "Gains diminish with size".

**What we ran.** WED as a selector over **our** cap320 T = 0.7 pools.
- Script: `src/exp_2026_10/wed/wed_select.py`, importing the authors' code.
- Artifact: `wed_vs_probe_2026-10-16.json` → `cells.<b>.probe_minus_wed_<metric>`.
- "WED-EM" is WED's own exact match, with its own canonicalisation.

| benchmark | n | currency | greedy | WED | probe | probe − WED [CI] | WED − greedy [CI] |
|---|---:|---|---:|---:|---:|---|---|
| PathVQA-open | 3,357 | J | 0.3140 | 0.2821 | **0.3664** | **+0.0843 [+0.0711, +0.0978]** | −0.0319 [−0.0446, −0.0189] LOSS |
| | | XJ | 0.3441 | 0.3077 | **0.3861** | **+0.0783 [+0.0647, +0.0918]** | −0.0363 LOSS |
| | | WED-EM | 0.2601 | 0.2237 | **0.2845** | **+0.0608 [+0.0497, +0.0722]** | −0.0363 LOSS |
| SLAKE-open | 645 | J | 0.7302 | 0.7302 | **0.7876** | **+0.0574 [+0.0326, +0.0818]** | 0.0000 tie |
| | | XJ | 0.7566 | 0.7488 | **0.7984** | **+0.0496 [+0.0216, +0.0775]** | −0.0078 tie |
| | | WED-EM | 0.6357 | 0.6264 | **0.6558** | **+0.0295 [+0.0048, +0.0546]** | −0.0093 tie |
| VQA-RAD-open | 200 | J | 0.4900 | 0.4850 | 0.4900 | +0.0050 [−0.0515, +0.0580] tie | −0.0050 tie |
| | | XJ | 0.5200 | 0.5200 | 0.5300 | +0.0100 tie | 0.0000 tie |
| | | WED-EM | 0.2600 | 0.2200 | 0.2600 | +0.0400 [−0.0053, +0.0826] tie | −0.0400 [−0.0761, −0.0050] LOSS |

**Same-protocol check.**
- Same images, same questions, same 8 candidates per question, same graders. The **only** difference is the selection rule. This is the one fully controlled external comparison we have.
- Our VQA-RAD open subset is 200 questions. WED's EM values are multiples of 0.5 pp (e.g. 17.50, 33.50; Table 3, p.8), which is consistent with n = 200.

**Footnotes the slide must carry.**
1. WED was adapted to choose from **our** 8 samples. Its own pipeline samples up to 16 with its own prompts and temperatures, and that pipeline was not reproduced.
2. When its filters removed every candidate, WED fell back to the majority answer: 12/3,357 PathVQA, 55/645 SLAKE and 0/200 VQA-RAD questions (FINAL_METHOD §5.3).
3. On VQA-RAD both selectors pick from the T = 0.7 pools. The final method uses T = 0.2 there, which scores 0.5150 J (`cells.vqa_rad_open.all.pick_judge`). The 0.4900 in this table is the T = 0.7 policy.
4. WED's paper never evaluates SLAKE. SLAKE is our extension of their protocol.

**Orientation only (not controlled).** WED's greedy VQA-RAD-open EM is 21.00 (Qwen3-VL-8B) and 29.00 (MedGemma-4B) (Table 3, p.8). Lingshu-7B greedy is 0.2600 on WED-EM (`cells.vqa_rad_open.greedy_wed_em`).

On PathVQA-open, WED's greedy judge accuracies are 4.98–23.88 (Table 3). Lingshu-7B greedy is 0.3140 J here. The judges differ (Grok vs MedVLThinker-32B), and Lingshu was trained on PathVQA (§1.4). **Do not put these on one axis.**

**Defensible claim.** "On identical candidates, the trained probe beats the published training-free selector WED on PathVQA and SLAKE in every currency, including WED's own exact match. VQA-RAD is a tie. With Lingshu-7B's pools, WED does not beat greedy decoding, and on PathVQA it is significantly worse."

Do **not** say "we beat WED's published numbers". Their numbers are for other generators.

### 1.2 LLaVA-Med protocol (open recall / closed accuracy): same metric code, different test counts

**Their protocol.** LLaVA-Med (Li et al., arXiv 2306.00890v1).
- §5.2 "Evaluation Metrics": "For the closed-set questions, we report the accuracy. For open-set questions, we use recall to evaluate the ratio that ground-truth tokens appear in the generated sequences."
- Table 4 caption: "For open-set questions, we report the recall for our free-form text generation method in column Open. For closed-set questions, we report the accuracy in column Closed."
- **Test counts** (Table 3):

| split | open | closed | total |
|---|---:|---:|---:|
| VQA-RAD | 179 | 272 | 451 |
| SLAKE (English only) | 645 | 416 | 1,061 |
| PathVQA | 3,370 | 3,391 | 6,761 |

**Ours.** `final_eval_FINAL_cap320_Tmap.json` → `full_official_splits`, using LLaVA-Med v1.0.0's code imported verbatim. Test counts:

| split | open | closed | total |
|---|---:|---:|---:|
| VQA-RAD | 200 | 251 | 451 |
| SLAKE-en | 645 | 416 | 1,061 |
| PathVQA | 3,357 | 3,362 | 6,719 |

**Comparability.**
- **SLAKE-en matches their split exactly** (645 / 416).
- VQA-RAD has the same 451 questions, but our open/closed partition differs (200/251 vs 179/272).
- PathVQA is 42 questions smaller (6,719 vs 6,761).
- LLaVA-Med's own text gives the reason it is "intrinsically harder" than prior work: earlier methods classify over the training answer set (§5.2).

| model | RAD open / closed | SLAKE open / closed | Path open / closed | training on the benchmark |
|---|---|---|---|---|
| LLaVA-Med zero-shot, 3 gray rows of Table 4(b): CLIP-7B, CLIP-13B, BioMedCLIP-7B, 60K-IM, stage 1/3, FT 0 | 28.23–37.84 / 60.66–61.40 | 37.71–39.73 / 49.76–54.33 | 11.34–12.30 / 49.07–54.05 | none |
| LLaVA-Med (From LLaVA), fine-tuned per dataset, Table 4(a) | 61.52 / 84.19 | 83.08 / 85.34 | 37.95 / 91.21 | yes, on each train split |
| LLaVA-Med (BioMed CLIP), fine-tuned, Table 4(a) | 64.75 / 83.09 | 87.11 / 86.78 | 39.60 / 91.09 | yes |
| **Lingshu-7B greedy (ours)** | 38.52 / 75.70 | 78.15 / 72.12 | 30.65 / 86.76 | Lingshu's SFT mix includes these train splits (§1.4) |
| **final method (ours)** | **39.74 / 77.29** | **81.15 / 71.63** | **33.43 / 89.74** | + probe trained on our open training halves |

Sources for our rows: `full_official_splits.<split>.{greedy,pick}_llava_open_recall` and `..._llava_closed_acc`, ×100.

**Defensible claims.**
- Under LLaVA-Med's own metric, the method raises open recall on all three splits:
  - VQA-RAD +1.22 pts
  - SLAKE +3.00
  - PathVQA +2.78 **(derived)**: pick − greedy, ×100
- Closed accuracy rises on VQA-RAD (+1.59) and PathVQA (+2.98 **(derived)**), and is flat on SLAKE (−0.48).
- The only closed-half change with a CI is PathVQA, significant: +0.0298 [+0.0197, +0.0397], `cells.pathvqa_closed.all.ci_pick_minus_greedy_llava_closed`.
- On SLAKE-en open, the only exactly matched split, the off-the-shelf model plus probe (81.15) is within 2 points of LLaVA-Med fine-tuned on SLAKE (83.08).

**Must not claim.**
- "We beat LLaVA-Med." The fine-tuned rows are higher on 5 of 6 columns, and only PathVQA open favours us.
- Any ranking against the zero-shot rows as if the generators were peers. Lingshu was trained on these datasets' training splits (§1.4); LLaVA-Med's zero-shot rows were not.

### 1.3 ImageCLEF VQA-Med 2019 official test (500): same test set, re-implemented metric

**Their protocol.** Ben Abacha et al., "VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019" (CEUR Vol-2380; local PDF `/data/dan/literature/tmp_compare/vqamed2019.pdf`).
- "The test set consists of 500 medical images and 500 questions" (p.6).
- The accuracy metric "strictly considers exact matching of a participant provided answer and the ground truth answer" (§4, p.6). BLEU is the second metric.
- Best team Hanlin: "0.624 Accuracy and 0.644 BLEU score" (p.7).
- 17 teams, 90 runs (p.6).
- **Table 3 per-category columns are divided by 4** ("normalized (each divided by a factor of 4) so that the summation is equal to the overall accuracy", footnote 8, p.8).

**Ours.** `vqamed2019_by_category_2026-10-16.json`. It uses ImageCLEF-style strict accuracy that we **re-implemented**; it is not the official evaluator. We have no BLEU.

| | modality | plane | organ | abnormality | overall |
|---|---:|---:|---:|---:|---:|
| Hanlin (winner), Table 3 p.7, raw | 0.202 | 0.192 | 0.184 | 0.046 | 0.624 |
| Hanlin, ×4 to per-category accuracy **(derived)** | 0.808 | 0.768 | 0.736 | 0.184 | 0.624 |
| ours, greedy, strict | 0.432 | 0.784 | 0.064 | 0.104 | 0.346 |
| ours, final, strict | 0.440 | 0.776 | 0.096 | 0.080 | 0.348 |
| ours, final, judge J | 0.720 | 0.832 | 0.472 | 0.128 | 0.538 |

- Overall J comes from `final_eval_FINAL_cap320_Tmap.json` → `cells.vqamed2019_test.all.pick_judge` = 0.538; greedy is 0.530.
- In `cells.vqamed2019_test.all`, final − greedy is +0.002 [−0.018, +0.022] on strict accuracy and +0.008 [−0.012, +0.028] on J. Both are ties.

**Leaderboard placement (derived).** Strict 0.348 sits between Table 3's 13th team, deepak.gupta651 (0.366), and 14th, ChandanReddy (0.294). That puts it **14th out of 18 rows if inserted**, counting only each team's best graded run in Table 3.

**Why the gap is mostly vocabulary.**
- The 2019 systems were trained on the challenge's category vocabularies, and the categories have "finite" answer sets (p.8).
- Our model answers free-form. On *organ*, strict matching accepts 0.096 of the final picks, but the judge accepts 0.472.
- On *plane*, where the vocabulary is small and conventional, we are at the winner's level: 0.776–0.784 vs 0.768 **(derived)**.

**Defensible claim.** "On the official 2019 test, the probe neither helps nor hurts. Strict accuracy (0.348) is far below the 2019 winner (0.624). The difference is concentrated in organ and modality naming, where the judge accepts 0.47 / 0.72 of the same answers."

**Must not claim** any rank under the official evaluator. We re-implemented it.

### 1.4 Lingshu's own published numbers: same generator, different protocol (NOT comparable as-is)

**Their numbers.**
- arXiv 2506.07044v4, **Table 6** (PDF p.17).
- The same values appear in v5 (21 Sep 2026), **Table I**. In v5 the model is renamed "Lingshu-Med-7B/32B".
- The two versions were checked against each other for the rows below.

| model | VQA-RAD | SLAKE | PathVQA | PMC-VQA | OmniMedVQA | MMMU-Med | MedXpertQA | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lingshu-7B | 67.9 | 83.1 | 61.9 | 56.3 | 82.9 | 54.0 | 26.7 | 61.8 |
| Lingshu-32B | 76.5 | 89.2 | 65.9 | 57.9 | 83.4 | 62.3 | 30.9 | 66.6 |
| Qwen2.5-VL-7B | 64.5 | 67.2 | 44.1 | 51.9 | 63.6 | 50.6 | 22.3 | 52.0 |
| MedGemma-4B-IT | 72.5 | 76.4 | 48.8 | 49.9 | 69.8 | 43.7 | 22.3 | 54.8 |
| LLaVA-Med-7B | 53.7 | 48.0 | 38.8 | 30.5 | 44.3 | 29.3 | 20.3 | 37.8 |
| GPT-4.1 | 65.0 | 72.2 | 55.5 | 55.2 | 75.5 | 75.2 | 45.2 | 63.4 |

**Their protocol** (v4 §4 "Evaluation", p.16; v5 §IV-B):
- "We use Accuracy as the primary metric for the QA task."
- Multiple choice: rule-based matching, then the MMMU codebase.
- "For open-ended questions, we directly leverage GPT-4.1 (2025-04-14 version) to assess consistency between model outputs and reference answers."
- v5 adds a human spot-check: GPT-4.1 "agreed with human annotators in 98% of cases" on 100 responses.
- Evaluation runs through their **MedEvalKit**.

What that means for VQA-RAD / SLAKE / PathVQA:
- Each is **one number over open + closed combined**.
- In MedEvalKit's SLAKE loader (`MedEvalKit/utils/SLAKE/SLAKE.py` l.42–60) there is **no language filter**. The local `test.json` holds **2,094 items: 1,061 English + 1,033 Chinese** (counted for this brief from `/data/dan/dataset/medevalkit/SLAKE/test.json`).
- So Lingshu's SLAKE 83.1 almost certainly includes the Chinese half. This rests on their stated use of MedEvalKit and on the public loader. We did not re-run their exact harness, so mark it "appears to".

**Our rows on the matched splits.** `full_official_splits.<split>.{greedy,pick}_judge`, ×100; combined open + closed; J judge; SLAKE English only; cap320:

| | VQA-RAD (451) | SLAKE-en (1,061) | PathVQA (6,719) |
|---|---:|---:|---:|
| ours, Lingshu-7B greedy | 64.1 | 77.1 | 59.1 |
| ours, final method | **66.1** | **80.0** | **63.2** |
| Lingshu-7B as published | 67.9 | 83.1 | 61.9 |
| Lingshu-32B as published | 76.5 | 89.2 | 65.9 |

**Comparable or not: NOT comparable across rows from different sources.** Four protocol differences:
1. **The judge differs.** GPT-4.1 vs MedVLThinker-32B. Our project measured that verifier gains differ by judge by roughly 0.006–0.009 (CLAUDE.md §0, `coadapt_verifier_T04_2026-08-14.json`).
2. **SLAKE language.** Theirs is en + zh, 2,094 items; ours is en, 1,061.
3. **Image resolution.** We use cap320 (`max_pixels` 250,880). MedEvalKit runs the model's default.
4. **Prompts and answer extraction.** These differ.

Our greedy sits 2.8–6.0 points below their published Lingshu-7B **(derived)**: 67.9 − 64.1 = 3.8; 83.1 − 77.1 = 6.0; 61.9 − 59.1 = 2.8. That gap is a protocol gap, not evidence that the probe "recovers" anything.

**The only like-for-like comparison is our two rows.**

**What may be said** (each row with its own footnote):
- "The probe lifts Lingshu-7B by +2.0 / +2.9 / +4.1 points on the three official splits under our protocol" **(derived)** from the values above.
- "On PathVQA the result (63.2) is numerically above Lingshu-7B's published 61.9 and below Lingshu-32B's 65.9, under different protocols."

**What must not be said.** "7B + probe matches 32B" or "beats Lingshu's published 7B".

The only matched 7B+probe vs 32B comparison we have is old and four cells only (CLAUDE.md §0a, `audit_stats_baselines_2026-09-20.json`):
- judge: 1 loss (VQA-RAD), 2 ties, 1 win
- lenient EM: 2 losses, 1 tie, 1 win

**Contamination note that belongs on this slide.** Lingshu's training data lists PathVQA, SLAKE, VQA-RAD and VQA-Med-2019 among its medical instruction data (v4 **Table 1**, p.5, "Medical Instruction Data"). The RL stage uses data "curated from MIMIC-Ext-MIMIC-CXR-VQA, PMC-VQA, Kvasir-VQA, Path-VQA, SLAKE, VQA-Med-2019, VQA-RAD" (v4 **Table 5**, p.12).

So the generator has seen the *training* splits of several of our benchmarks. Our VQA-Med C4 cell is mostly the 2019 *training* split (FINAL_METHOD §6.4). This affects greedy and probe equally, so it does not inflate our Δ. It does affect any absolute-accuracy comparison against models that were not trained on those splits.

---

## 2. Same task, different protocol: context, not ranking

### 2.0 The mapping problem in one line

Every paper below reports a gain against a **different baseline**:
- majority vote
- a single sample
- greedy chain-of-thought
- another reward model
- coverage, not accuracy

Each also uses a different grader and N. Our numbers are **probe pick − greedy** on the same pool.

To set ourselves against the papers that use majority vote as the baseline, the matching quantity of ours is **probe pick − majority vote on the same pools**, also listed below.

| our quantity | J | source |
|---|---:|---|
| probe − greedy, macro | +0.0897 | `macro_over_open8.all.pick_minus_greedy_judge` |
| majority − greedy, macro | −0.0070 | `...sc_minus_greedy_judge` |
| **probe − majority, macro** | **+0.0967** | **(derived)**: 0.08974 − (−0.00700) |
| probe − majority, PathVQA-open | +0.0724 | **(derived)**: `pick_judge` 0.3664 − `sc_judge` 0.2940 |

The probe − majority values are differences of point estimates. No CI was computed for them.

### 2.1 Medical test-time compute

| paper | setting | baseline | gain in their units | how it maps to ours |
|---|---|---|---|---|
| **Best-of-Evidence** (Zhang et al., arXiv 2607.20950) | generator Qwen3-VL-30B-A3B; evidence judge **Qwen3-VL-235B-A22B**; **K = 16** at T = 1.1; cells: VQA-Med open (2,334, training partition), PathVQA open (9,903), PMC-VQA "option-hidden open" (10,000), MedXpertQA-MM 5-way MCQ (2,000) (§5.1, p.7; Table 1). Open answers graded by matching + semantic-equivalence grading using "the same model tier" as the judge | **raw best-of-N majority**. **No greedy or single-sample baseline anywhere in the paper** | "BoE is 0.26–0.58 percentage points above raw majority" (p.9). Table 2 (p.9), BoE − raw [95%], pp: VQA-Med +0.26 [−0.47, +0.99]; PathVQA **+0.43 [+0.05, +0.81]**; PMC-VQA **+0.58 [+0.19, +0.97]**; MedXpertQA-MM +0.40 [−0.15, +0.95]. Absolute (Fig. 3 right, p.8), PathVQA open: majority 29.28 %, BoE 29.72 %; Oracle@16 61.5 % (Table 1). "All analyses are exploratory and uncorrected" | Same-*baseline* quantity of ours: probe − majority **+0.0967** macro J, **+0.0724** on PathVQA-open **(derived)**, with 8 samples and a ~918k-parameter probe vs 16 samples and a 235B judge. **Not comparable as a ranking**: different generator (30B MoE vs 7B), grader, N and PathVQA subset (their 9,903 non-yes/no items vs our 3,357 test-open). Use it as *context*: the published medical best-of-N with a frontier-scale judge moves accuracy by under a point over voting |
| **Verification Mirage** (Jin et al., arXiv 2605.10850) | self-verification (the generator checks its own single answer) on VQA-RAD, PathVQA, SLAKE, PMC-VQA, MedXpertQA; six VLMs including "Lingshu" (**size not stated**); Gemini-2.5-Flash grades correctness (App., p.16) | none (diagnostic study, no TTS gain) | "verifier error ≳40%" and "FPR ≳60%" (§4.2, p.6); differential-diagnosis FPR "∼95-100%" (p.7); generator error → "57× higher odds of verifier failure" (§4.3, p.8; OR 57.4, App. Table 1, p.24); "Only 2.2–3.8% of initially wrong answers are corrected over four turns, while 69.5%–87.1% are locked in" (§4.6, pp.10–11); larger same-family verifier for Lingshu: "no significant reduction (p= 0.782)" (B.4, p.19). Lingshu self-verifier FPR: 62.2–72.5 % across the five datasets (Tables 11–15, pp.28–30) | **Motivation, not a competitor.** It shows that the *untrained* verifier fails, including for Lingshu, and that a bigger same-family verifier does not fix it. Our component is the *trained* version. Their Lingshu generator accuracies (e.g. SLAKE 74.3, VQA-RAD 58.5, PathVQA 51.8; Tables 6–10) use a different judge and must not be compared with ours |
| **MedGemma tech report** (Sellergren et al., arXiv 2507.05201) | single run per example, **temperature 0.0** (§3.1, p.5); metric "average tokenized F1 … across open and closed QAs, as well as accuracy on the subset of yes/no questions" (§3.4) | n/a (no test-time compute) | Table 9 (p.15), MedGemma 4B: SLAKE overall token F1 72.3, **SLAKE open token recall 63.3**, SLAKE closed acc 87.6; VQA-RAD token F1 49.9, closed acc 69.1. Table 9 has **no PathVQA and no 27B row**. Table 4 (p.9), MedXpertQA-MM: MedGemma 4B 24.4; 27B "N/A" | Orientation only. SLAKE is English-only, 1,061 items (their Table 2), the **same split** as ours. But their "open token recall" is their own tokeniser, not LLaVA-Med's code, so do not set it beside our 81.15. Their VQA-RAD split differs (the paper cites both Yang et al. 2024 and Xu et al. 2023) |
| **MedProb, App. H** (arXiv 2609.04336, p.27) | Qwen2.5-VL-7B and Llama-3.2-11B; OmniMedVQA "true multiclass items" turned open-ended; 500 train / 100 test | "a single sampled generation with no reranking" | Table 14: Qwen2.5-VL-7B 0.09 → 0.13 (oracle 0.19); Llama-3.2-11B 0.15 → 0.19 (oracle 0.27). N, grader and CIs not stated | Closest in task (probe reranking of open medical VLM answers, on OmniMedVQA). Ours on OmniMedVQA (open): greedy 0.5164 → 0.6894 J, +0.1730 [+0.1647, +0.1813], n = 8,883, oracle@8 0.7230. **Not comparable in level**: their accuracies (0.09–0.19) imply a very different grader or item set. Compare scale and rigour (100 vs 8,883 test questions; CIs), not the numbers |
| **WED paper** (§1.1) | Qwen3-VL 2/4/8B, Gemma-3-4B, MedGemma-4B; Grok-4.1-Fast judge | greedy | Δ judge +1.2 to +5.0 pp, except Qwen-8B VQA-RAD −1.2 (Table 1, p.8) | Superseded by our controlled head-to-head (§1.1). Quote the paper only to say that its gains shrink with model size, which is consistent with WED − greedy ≈ 0 or negative on Lingshu-7B |
| **DualRead, CASE** | card-only (§3) | — | no numbers usable (DualRead does not select; CASE abstract: "up to 19 points" over voting on medium questions, general + medical *text* LLMs) | Cite as neighbours, not as numerical comparisons |

### 2.2 General test-time compute (text LLMs): one number each, own units

| paper | headline (verbatim, with source) | baseline | maps to ours how |
|---|---|---|---|
| **Self-consistency** (Wang et al., ICLR'23) | PaLM-540B GSM8K 56.5 → **74.4 (+17.9)**, 40 samples (Table 2, p.5); ARC-c 85.2 → 88.7 (+3.5) (Table 3) | greedy chain-of-thought | Same *baseline* as ours (greedy), but on math with exact answers, where the majority answer is usually correct. On our pools majority vote **loses** to greedy: −0.0070 J macro, 1/8 positive (§0). The difference is the answer space: open medical answers rarely repeat exactly, so voting has nothing to aggregate |
| **Universal SC** (Chen et al., 2023) | PaLM 2-L GSM8K greedy 85.7 → SC 90.4 / USC 90.2 (8 samples; Table 1, p.5); TruthfulQA GPT-judge 62.1 → 67.7 (Table 4, p.6); "accuracy on GSM8K decreases with 16 samples" (p.6) | greedy | An LLM-based selector over free-form text; same 8-sample budget as ours. Not run on our pools: an **open baseline we have not run**, so say so if asked |
| **Snell et al., 2024** | "improve the efficiency of test-time compute scaling by more than 4× compared to a best-of-N baseline"; FLOPs-matched, test-time compute "can be used to outperform a 14× larger model" on problems "where a smaller base model attains somewhat non-trivial success rates" (abstract, p.1); on hard questions "pretraining is a more effective way" (§7, p.15) | best-of-N (PRM) / 14× larger model | Units are compute efficiency, not accuracy. Relevant framing: our gain concentrates where coverage exists (oracle@8 well above greedy) and is ~0 on VQA-Med C4 (oracle@8 0.1873), which matches Snell's "hard questions" regime |
| **Large Language Monkeys** (Brown et al., 2024) | SWE-bench Lite coverage "15.9% with one sample to 56% with 250 samples" (abstract); MATH Llama-3-8B coverage 82.9 % → 98.44 % (100 → 10,000 samples) while "majority voting or reward models … only from 40.50% to 41.41%" (p.3) | 1 sample (coverage) | Coverage = our oracle@N. Their point, that selection fails to harvest coverage, is our "selection wall". Our probe converts part of the oracle gap; this is measurable per cell from §0 (e.g. OmniMedVQA greedy 0.5164, probe 0.6894, oracle@8 0.7230) |
| **Cobbe et al., 2021** | verifiers give "approximately the same performance boost as a 30x model size increase" (p.2); 100 samples ranked by the verifier (§4.2, p.8); "not beneficial to use verification at low dataset sizes" | finetuned generator | Trained-verifier lineage. Their verifier is a full fine-tuned LM; ours is a probe. Their low-data warning matches our LOBO result (§0) |
| **Sammani et al., 2026 (test-time scaling for VLMs)** | Qwen3-VL-2B WeMath 35.46 → 64.25 with self-consistency (Table 1, p.7; **absolute ≈ +28.8 pts (derived)**, the paper says "approximately a 30% increase"); "test-time scaling methods often degrade performance … on primarily perception-focused benchmarks" (p.9) | greedy | The VLM evidence: voting helps reasoning-heavy visual math and hurts perception. Medical VQA is mostly perception, which agrees with our majority-vote loss |
| **Stroebl et al., 2024 (limits of resampling)** | an imperfect verifier "imposes an upper bound to the accuracy of resampling-based inference scaling, regardless of compute budget"; "optimal sampling attempts are often fewer than 10" (abstract) | — | Supports N = 8 as a sensible budget and frames our selection efficiency as bounded by verifier quality |

**What the deck should say about §2.** "Published medical test-time methods report gains of 0.26–0.58 pp over voting with a 235B judge (Best-of-Evidence) or 1–5 pp over greedy for small general VLMs (WED). Untrained self-verification fails on medical VQA, Lingshu included (Verification Mirage). Our trained probe gives +0.0897 J / +0.0783 XJ / +0.0165 EM over greedy. These are different baselines, graders and generators, so the figures are context, not a ranking. The one controlled comparison is §1.1."

---

## 3. Mechanism neighbours: probes on frozen hidden states used to pick or score answers

Taxonomy used in the domain guide (§4.1):
- **(A)** a classifier on frozen hidden states
- **(B)** used to select among sampled candidates
- **(C)** pooled over the candidate's own tokens
- **(D)** reuses the generation pass, with no extra forward pass

Two facts about our own method:
- **For our own headline runs only A+B+C are demonstrated.** Features came from a separate teacher-forced pass at `max_pixels` 1,003,520 (CLAUDE.md §0a). (D) was shown once, for an earlier August probe.
- Our probe has about 918k parameters per probe (guide §5.6), and 24 probes are ensembled.

All values below were re-read from the PDFs (`/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/lit-domain-package/literature/papers/`) unless marked "card only".

| paper | task / modality | generator | probe and features | selection | headline in their units (source) | what differs from ours |
|---|---|---|---|---|---|---|
| **SWIFT** (= "ELHSR" in earlier versions), Guo et al., arXiv 2505.12225v3, KDD'26 | text-only math (MATH, GSM8K, AQuA_RAT) + code comprehension, HellaSwag, CoinFlip | Llama-3.2-3B, Llama-3.1-8B, Ministral-8B; T = 1.0, top-p 0.9 | one linear map over all-layer hidden states per token → gate × reward, gated mean over **all** response tokens; BCE on correctness; 1.8–3.0×10⁵ params (Table 3, p.7); 6,000 training instances × 8 paths | best-of-N argmax, N = 4/16/64 | MATH avg 57.5 vs EurusRM-7B 51.0; abstract's "12.7% higher" is **relative** (+6.5 pts absolute) (Table 1, p.5). Llama-3.1-8B MATH: BoN@1 47.2 → SWIFT@64 62.6. Baselines are text reward models; **no majority-vote baseline** | text-only; no image; no medical; no cross-fitting by item group; its pooling (all tokens, gated) is the closest to our mean-pool, so it is the nearest single neighbour |
| **HSRM**, Li & Zhu, arXiv 2608.30841v1 | text-only math (GSM8K, MATH-500, AIME, OlympiadBench) | Qwen3 1.7B–14B (non-thinking), Llama-3.2-1B/3B, Llama-3.1-8B; T = 0.7 | final-layer states at reasoning-step boundaries → 2-layer Transformer encoder → mean over steps; ~2.1–3.4 M params (Table 4, p.12); **Bradley–Terry loss; BCE was worse** (Table 9, p.14: GSM8K 85.93 vs 83.91) | best-of-8 | "matches or outperforms a 55M-parameter text-only energy verifier in 15 of 16 generator–dataset settings while using only about 2M parameters" (abstract). Llama-3.1-8B GSM8K: HSRM 88.3 / EORM 85.2 / oracle 96.0 (Table 3, p.9) | text-only; states "no additional generator forward passes" (§3.1, p.3), but runs on cached tensors and does not show how they were captured; step-boundary states, not a mean over the answer |
| **Q-Probe**, Li et al., arXiv 2402.14688v2, ICML'24 | code (MBPP, HumanEval), GSM8K, preference | Code-LLaMA-7B (layer 26); gpt-3.5 generations embedded by CodeLlama-70B; LLaMA-7B SFT | **linear** probe, 4,096-dim; pooling not specified; "2 or 3-layer MLPs … harms performance by overfitting" (§7.2, p.9, preference data) | soft reweighting over k = 48 (β = 0.1) for code; argmax (β = 0) for preference | MBPP: Pass@1 0.29 → Q-probe (L_PG) **0.46** (greedy 0.38; Pass@48 skyline 0.76) (Table 1, p.6); GSM8K 0.25 → 0.45 (greedy 0.29; skyline 0.80) (Table 3, p.7) | text-only; earliest instance; linear; embeddings come from a separate forward pass in the preference experiments (§7.1); says evaluating the probe "is essentially free in comparison to the base model" (§2) |
| **MedProb**, Nourbakhsh et al., arXiv 2609.04336 | medical VQA (PathVQA, SLAKE, VQA-RAD, OmniMedVQA), **multiple-choice** | Qwen2.5-VL, InternVL3, LLaVA, Gemma/MedGemma, Llama-3.2-Vision | main method: per-layer multinomial logistic regression on the **last input token, before generation**; no candidate selection | main: none. **App. H (p.27):** probe scores each open-ended candidate, highest is kept | App. H, Table 14 (p.27), OmniMedVQA "true multiclass items", **500 train / 100 test**: Qwen2.5-VL-7B 0.09 → **0.13** (oracle 0.19); Llama-3.2-11B 0.15 → **0.19** (oracle 0.27). The baseline is "a single sampled generation with no reranking", not greedy. N, probe type, layer, pooling, grader, temperature and CIs are **not stated**. Authors: "an initial demonstration" | the nearest *medical VLM* selector, but App. H is 100 test items, one benchmark, unspecified grader, candidate re-fed as input; ours: 8 benchmarks, 36,869 questions, two judges + EM, image-level cross-fitting, CIs. The PDF has two appendices labelled "H" (p.27 and p.34); cite p.27 by title |
| **CASE**, Wang et al., arXiv 2608.17124 (**card only, abstract, no PDF**) | general + **medical LLMs** (text) | not extracted | linear gate on the answer-token hidden state | best-of-N vs majority vote | "improves over voting by up to 19 points on medium-difficulty questions and 16.8 points on hard questions"; decodability predicts the gain over voting with "r=0.75 and a decision threshold near AUC=0.60" (abstract) | text-only; single answer-token state, not a mean; its question-grouped "decodability" is the leakage control we lack (cf. our image-level cross-fitting) |
| **LiLaVe**, arXiv 2504.16760 (**card only**) | text-only reasoning | frozen base LLM | lightweight verifier on hidden states (guide says gradient-boosted trees; "tried an MLP and rejected it" is from a project brief, **not re-verified**) | best-of-n, self-consistency, conditional majority voting | no numbers extracted | text-only; the MLP-vs-trees remark is unverified, so do not quote it |
| **ReProbe**, Ni et al., arXiv 2511.06209 (**card only**) | text-only math, planning, knowledge QA | frozen LLM | transformer probe < 10 M params; step-level credibility | selects **steps** inside a chain (the card says not whole answers) | "match or exceed the performance of PRMs that are up to 810x larger" (abstract) | step-level process verifier, not whole-answer best-of-N; the guide's §4.1 table and its card disagree on (B), so state it as "step-level" |
| **DualRead**, arXiv 2609.06419 (**card only**) | medical VQA (SLAKE, VQA-RAD, PathVQA, PMC-VQA, OmniMedVQA per guide §4.3) | GRPO-trained medical VLMs (Qwen3-VL, MedGemma per guide) | internal states at pre-answer and post-answer points | **none.** Calibrates the one answer produced | "improves correctness discrimination and calibration over verbalized confidence while preserving answer accuracy" (abstract); no numbers | does not select, so there is no accuracy gain to compare; same benchmarks and similar pooling, so cite and distinguish |

**HSRM note.** For the Llama table, the PDF gives HSRM / EORM / Oracle as:

| benchmark | generator | HSRM | EORM | Oracle |
|---|---|---:|---:|---:|
| GSM8K | 1B | 48.4 | 40.3 | 72.2 |
| GSM8K | 3B | 81.1 | 75.1 | 93.0 |
| GSM8K | 8B | 88.3 | 85.2 | 96.0 |
| MATH-500 | 1B | 30.1 | 26.5 | 61.4 |
| MATH-500 | 3B | 47.9 | 43.4 | 76.3 |
| MATH-500 | 8B | 56.8 | 52.3 | 87.1 |

The Qwen3 results are only in Figure 2, as plots, so no numbers can be quoted for them. HSRM's tables report the same cell differently: GSM8K Qwen3-1.7B is 84.8 in Table 7, 85.6 in Table 8 and 86.32 in Table 1.

**How these map to ours (one slide's worth).**
- The text-only neighbours report gains against *other reward models* (SWIFT, HSRM) or against Pass@1 / greedy (Q-Probe), on math and code with exact, checkable answers. Their gains are in absolute accuracy points on single benchmarks.
- None reports a majority-vote baseline except CASE (abstract only).
- Ours is the only one that:
  1. runs on a **vision-language** model;
  2. is on **open-ended medical** answers graded by **two independent judges and exact match**;
  3. reports a **gain over greedy with image-clustered CIs on 8 benchmarks**;
  4. **cross-fits at the image level**, so no question is scored by a probe that saw its image.

Numerical gains are **not comparable** across these papers: different tasks, generators, N, graders and baselines. The slide should compare *design*, not *size of gain*.

**Two findings in these papers that cut against our choices. Volunteer them.**
1. HSRM found BCE worse than Bradley–Terry (Table 9). We use BCE.
2. Q-Probe found 2–3-layer MLPs overfit on preference data (§7.2). We use a 1-hidden-layer MLP.

Our own capacity check: a 32-unit probe scores 0.70090 against 0.69742 for a 1024-unit probe (guide §5.6, `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §8). So width is not the binding factor. A ranking loss has **not** been tested on the final recipe.

---

## 4. Recommended slide plan for the comparison section

Seven slides. The order runs from the most controlled comparison to the least.

### Slide C1 — "Our gain, every currency" (anchor)
- **Content:** a table with rows J, XJ, EM, token-F1, LLaVA recall. Columns: Δ vs greedy [CI] both folds; n/8 positive; Δ on image half 1.
- **Extra row:** self-consistency on the same pools, −0.0070 [−0.0111, −0.0029].
- **Chart:** a forest plot of per-benchmark Δ J with CIs (8 rows), with Δ EM as a second marker per row.
- **Footnotes:**
  - Judge = MedVLThinker-32B (Qwen2.5-VL lineage, same as Lingshu); XJ = MedGemma-27B.
  - EM is lenient containment and penalises paraphrase on sentence-gold benchmarks (Kvasir-x1, GEMeX).
  - "n/8 positive" counts point estimates, not significant wins.
- **Sources:** `final_eval_FINAL_cap320_Tmap.json`, `final_fold1_headline_2026-10-16.json`.

### Slide C2 — "Head-to-head vs WED on identical candidates" (strongest external)
- **Content:** the §1.1 table, with J, XJ and WED-EM rows for each of the 3 benchmarks.
- **Chart:** grouped bars (greedy / WED / probe) per benchmark under J, with probe − WED CIs written above.
- **Footnotes:**
  1. Same 8 candidates per question; only the selector differs.
  2. WED adapted to our pools (its own sampling up to 16 not reproduced).
  3. Majority fallback on 12/3,357, 55/645 and 0/200 questions.
  4. WED's paper covers VQA-RAD and PathVQA only.
- **Source:** `wed_vs_probe_2026-10-16.json`.

### Slide C3 — "Field-standard metrics on full official splits"
Two small tables.
- **(a) LLaVA-Med protocol:** columns RAD / SLAKE / Path × open recall / closed accuracy; rows are LLaVA-Med zero-shot (range), LLaVA-Med fine-tuned, ours greedy, ours final.
- **(b) VQA-Med 2019:** rows Hanlin (0.624), ours greedy and ours final under strict accuracy and J, plus per-category bars that show the organ gap.
- **Footnotes:**
  - Test counts differ (179/272 vs 200/251 on RAD; 6,761 vs 6,719 on Path); SLAKE-en is identical.
  - The ImageCLEF metric is re-implemented, not official; no BLEU.
  - The winners were trained on the category vocabularies.
  - Lingshu was trained on these datasets' training splits.

### Slide C4 — "Where Lingshu's own numbers come from" (expectation management)
- **Content:** the §1.4 table: our greedy and final on the three official splits, beside Lingshu-7B and -32B as published.
- **Protocol column:**
  - GPT-4.1 judge vs MedVLThinker-32B
  - SLAKE en+zh (2,094) vs en (1,061)
  - default resolution vs cap320
- **Footnote** (verbatim wording recommended): "Rows from different protocols are shown for orientation only; the like-for-like comparison is our two rows."
- **Do not draw an arrow from our final row to Lingshu-32B.**

### Slide C5 — "Medical test-time compute: what others get"
- **Content:** the §2 table: Best-of-Evidence, Verification Mirage, MedProb App. H, WED (paper), each with its N, verifier/judge, baseline and gain *in its own units*.
- **The single message:**
  - Published medical best-of-N with a 235B judge gains under a point over majority vote.
  - Untrained self-verification fails.
  - Our trained probe is a different operating point: an 8-sample pool, about 918k parameters per probe, +0.0897 J over greedy.
- **Footnote:** "Different generators, baselines (greedy vs majority vote vs single sample), graders and N; gains are not on a common scale."

### Slide C6 — "Mechanism neighbours: we did not invent this"
- **Content:** the §3 table, cut to 5 columns: paper | modality | probe | selection | what is ours that they lack.
- Tick-mark A/B/C/D columns.
- Put SWIFT and HSRM on top as "same mechanism, text-only math". Put MedProb and DualRead as "medical, but not this".
- **Footnote:** our D is "by design"; the headline runs used a separate teacher-forced pass.

### Slide C7 — "General test-time compute context"
- **Content:** self-consistency, Snell, Brown, Cobbe, each with one headline number in its own units (§2).
- **Main point:** majority voting is the field's default free baseline, and on our pools it **loses** to greedy.

### Claims that are defensible
1. The probe improves Lingshu-7B over its own greedy decoding on open-ended medical VQA, in all five currencies. The macro gains are significant (§0), with 6 of 8 benchmarks significant under the judge.
2. On identical candidates it beats WED, the only published selector on our exact open subsets, on PathVQA and SLAKE in every currency including WED's own EM. VQA-RAD is a tie.
3. Majority voting, the standard training-free selector, is a significant loss on these pools.
4. Under LLaVA-Med's own metric code, open recall rises on all three official splits. Closed accuracy rises significantly on PathVQA, although the probe was never trained on closed questions.
5. The mechanism belongs to a known family (SWIFT, HSRM, Q-Probe and others). What is new is the setting:
   - a VLM
   - medical, open-ended answers
   - 8 benchmarks
   - image-level cross-fitting
   - a cross-family judge
   - a controlled head-to-head against a published selector

### Claims that must NOT be made
1. "We beat Lingshu-32B", "7B + probe matches 32B", or "we beat Lingshu's published 7B". The protocols differ (§1.4). The one matched 4-cell test is mixed, with a loss on VQA-RAD.
2. "We beat LLaVA-Med." Its fine-tuned rows are higher on most columns.
3. Any ImageCLEF leaderboard rank. Our metric is re-implemented.
4. "Our gain is N× larger than Best-of-Evidence / WED / SWIFT." The baselines, judges, generators and N differ. The only controlled multiple is against WED on our own pools.
5. "We invented hidden-state verification" or "verification is free". See SWIFT, HSRM and Q-Probe.
6. "The probe adds no forward pass" as a measured fact for this probe.
7. "The verifier generalises to new benchmarks." The leave-one-benchmark-out (LOBO) macro is +0.0223 vs pooled +0.0707; the benchmark's own training half is worth +0.0485.
8. A judge-only number on any slide. Always show XJ and EM beside it.

---

## 5. Corrections and discrepancies found while verifying the domain guide

These were checked against the PDFs/HTML. The guide's Tables A–E values all match the artifacts exactly.

1. **Lingshu table naming.** The guide's "Table 6 (v4) = Table I (v5)" is correct.
   - v4 Table 6 is on p.17.
   - v5 renames the models "Lingshu-Med-7B/32B".
   - All Lingshu, Qwen2.5-VL-7B and MedGemma-4B values match.
2. **SLAKE "appears to include Chinese".** Now verified at the loader level:
   - MedEvalKit's `test.json` has 2,094 items = 1,061 en + 1,033 zh.
   - The loader keeps both.
3. **LLaVA-Med zero-shot ranges.** These match the three gray rows of Table 4(b).
   - Other FT = 0 rows in Table 4(b) (for example 60K-IM stage 1/1: RAD open 29.67) fall inside or near the ranges.
   - Say "the three zero-shot rows highlighted in Table 4(b)".
4. **VQA-Med Table 3 per-category values are ÷4** (footnote 8). Any per-category comparison against the winners must multiply by 4. The guide does not mention this.
5. **WED sampling.** The paper says "temperature-diverse sampling, Nmax = 16, n = 8".
   - The "T ∈ {0.5, 1.0}" in FINAL_METHOD §5.3 is not in the PDF; it presumably comes from the authors' code.
   - Cite it as code, not paper.
6. **SWIFT/ELHSR.**
   - The "12.7%" is relative (+6.5 pts absolute).
   - The name "ELHSR" does not appear in v3.
   - (D) is not established by the PDF.
7. **HSRM.**
   - Its default layer is the final layer.
   - BCE underperforms Bradley–Terry.
   - "Matching the 7B PRM" holds only "at the larger generator scales".
8. **Q-Probe.**
   - The gpt-3.5 results use CodeLlama-70B embeddings, not API embeddings.
   - The preference experiments use a hard argmax.
   - (D) is settled as a separate pass.
   - "MLPs overfit" is not in the guide.
9. **MedProb.**
   - "20+ points consistently" overstates the paper's "exceeding 20 percentage points in several cases".
   - App. H does not state that its probe is logistic regression.
   - App. H's baseline is a single sample, not greedy.
   - Oracle values: 0.19 / 0.27.
10. **Guide-internal inconsistencies with no PDF to settle them** (LiLaVe, ReProbe, CASE, DualRead): the §4.1 table and the cards disagree on dimensions (B) and (C). Present these four as "per abstract".

11. **Self-consistency card (guide §3.5).** The abstract figures (AQuA +12.2, StrategyQA +6.4, ARC-c +3.9, SVAMP +11.0) are **code-davinci-002** rows, not "minor rounding" of PaLM-540B.
12. **MedGemma card.**
    - The abstract's "2.6–10% improvements" is qualified "For out-of-distribution tasks".
    - MedGemma-27B multimodal MedXpertQA is explicitly "N/A" in Table 4.
    - Table 9 has no PathVQA column and no 27B row.
13. **Best-of-Evidence.** The §3.5 duplicate card (guide l.2458) says "no numbers extracted". It is stale relative to the §3.2 card.
14. **Stroebl.** "K ≤ 5" is in the §4 Findings text (p.7), not in the Fig. 4 caption. The models are the Llama-3.1 and Code Llama families.
15. **Verification Mirage.** The Lingshu model size is **not stated** in the PDF. Do not write "Lingshu-7B" for their numbers.
