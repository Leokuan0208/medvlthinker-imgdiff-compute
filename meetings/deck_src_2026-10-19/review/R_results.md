# R_results — skeptical committee review (statistics & evaluation methodology)

Scope: opening (slides 1–3), Part V (48–54), Part VII (61–65), backup (66–71), with the method slides they depend on
(26, 31, 34, 35, 38, 46, 47, 58). Reviewer stance: "the most skeptical member of the committee".

Every number below is either quoted from a named artifact/key or **computed by me, read-only**, from
`/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json` (the per-question records behind
`final_eval_FINAL_cap320_T07.json`) with the scripts in this directory:
`rr_stats.py` → `rr_stats_out.json` (micro, Bonferroni, LOBO, per-fold, correlations; 10,000 image-clustered
resamples within benchmark, seed 12345 — so CIs differ from the artifact's in the 4th decimal),
`rr_lobo.py` (EM leave-one-benchmark-out CIs), `rr_rows.py` (training-row reconciliation from the
`feats_hidden/*.meta.json` caches). No GPU, nothing written to the repo.

---

## 0. What I checked and found CLEAN (say so if asked)

| check | result | evidence |
|---|---|---|
| Image cross-fitting is real in code | Yes. Fold f scores questions with `md5("nd"+img_md5)%2==f`; training keeps a row only if `(original-domain or half==1-f) and img_md5 not in evimg`, where `evimg` = every fold-f image of **every** scored cell (so same image across cells is covered). `img_md5` is a decoded-RGB-**pixel** md5 (AUDIT appendix DI-2), so re-encoding does not evade it. | `src/exp_2026_10/final_method_eval.py` lines with `tr = np.array([(Forig[i] or Fhalf[i] == 1 - f) and Fimg[i] not in evimg ...` |
| Training-row counts reproduce exactly | 115,934 (fold 0) and 116,000 (fold 1) reproduced from the cache metas; 261 / 377 `vqa_rad_open_train` rows dropped because their images are pixel-identical to images in `vqamed_open` (19 images), `vqa_rad_closed` (55) and `vqamed2019_test` (4). | `rr_rows.py`; artifact `folds.{0,1}.train_rows` |
| Exact-image overlap between the 4 extra domains and the 8 eval cells | Zero for 7 of 8 open cells; only the VQA-RAD-train ↔ VQA-Med overlaps above, all removed. Kvasir-VQA (976 imgs) ↔ Kvasir-VQA-x1 (2,860 imgs): 0 exact matches. | `rr_rows.py` |
| Judge labels consistent between greedy and candidate passes | Of 14,471 questions where pick and greedy are the same string, the judge label differs on 8; MedGemma on 0. | computed from records file |
| Bootstrap is paired and image-clustered; macro CI averages per-cell draws | Correct for a fixed set of 8 benchmarks. | `summarize()` in `final_method_eval.py` |
| Abstention / deferral / reject-option | Not presented anywhere as a method (grep of slide text + script for abstain/defer/reject/selective/fallback: no hits). WED's "majority fallback" is an answering fallback. | — |
| Half-1 vs half-0 | Fold-1 macro judge +0.0846 > fold-0 +0.0747 (EM +0.0165 vs +0.0076) — no sign that in-sample tuning inflated the headline. | `rr_stats_out.json` `judge.fold{0,1}_macro_pt`; `final_fold1_headline_2026-10-16.json` |

---

## 1. Issues

### HIGH — the claim is unsupported or wrong as stated

**H1 · Slide 58 (+ script §58, slides 26 and 31) — "the probe never trained on closed questions" is false.**
The FINAL run's `args.cells` includes `vqa_rad_closed, pathvqa_closed, slake_closed, vqamed2019_test`, and the
training loop (`for c in cells: ... fit_X[L].append(X07[L])`) adds the other image half of **every** cell. The
row reconciliation only reaches 115,934 / 116,000 if those cells are included: fold 0 adds 167 + 2,246 + 251 + 702
= 3,366 rows (VQA-RAD-closed, PathVQA-closed, SLAKE-closed, VQA-Med-2019-test), fold 1 adds 3,325 (`rr_rows.py`).
So the significant PathVQA-closed gain (+0.0297 [+0.0197, +0.0397]) comes from a probe that trained on 2,246 rows of
PathVQA-closed's other half. That is in-domain training, not transfer. The same wrong sentence is in
`FINAL_METHOD_2026-10-16.md` §5.1 ("The probe was trained only on open answers").
*Fix, slide 58 "claim":* "under LLaVA-Med's own scoring code the probe raises open recall on all three splits; closed
accuracy rises significantly on PathVQA (+0.0297 [+0.0197, +0.0397]) — the probe's training data included the other
image half of the closed questions (2,246 PathVQA-closed rows per fold), so this is in-domain, not transfer."
*Fix, slides 26 and 31 "training rows":* "the other image half of all 8 open benchmarks **and of the 4 comparability
cells (closed VQA-RAD/PathVQA/SLAKE, VQA-Med 2019 test)** + 4 extra domains: 115,934 / 116,000 rows."
*Script §58:* delete "even though the probe never trained on closed questions". *Docs:* add a correction banner to
FINAL_METHOD §5.1 (the CLAUDE.md meta-lesson: fix it everywhere).
*Does this affect the open-8 headline?* No leakage (image-disjoint), but the training-set description on slides 26/31
is wrong, and the open-8 numbers come from probes that also saw those ~3,300 rows.

**H2 · Slide 62 ("judge validated by humans on ~11% of questions") and slide 63 ("judge vs human agreement on 5 of 8
benchmarks"), script Q&A §2 — the human-validation claim is not what the evidence is.**
Source (`AUDIT_2026-09-18_appendix/stats-baselines.md` B9, `AUDIT_2026-09-18.md` §5): the "11.1%" is the share of the
September held-out questions that sit in benchmarks with *any* independent check (SLAKE / VQA-RAD / PathVQA). The checks
themselves are: `claude_judge/verdicts_lingshu32b/` — **LLM (Claude) verdicts**, SLAKE and VQA-RAD only, July pools;
`pathvqa_judge_audit.json` — 25 accepts + 25 rejects hand-checked (judge error 0.04, Wilson 95% [0.0071, 0.1954]) plus a
90-item disagreement pool (clear judge error 7.8% [0.0382, 0.1519]). None of it is on the shipped method's picks. I could
not confirm who produced the PathVQA "hand" labels — **flag: check before saying "human"**.
*Fix, slide 62:* "judge-sensitive — exact match gains are small and lose on 3 benchmarks; the main judge is also the label
source and shares Lingshu's lineage; **no human evaluation of this method's picks**. Earlier spot checks (July pools): an
LLM adjudicator on SLAKE/VQA-RAD and 50 + 90 hand-checked PathVQA items (judge error up to ~20%, Wilson upper bound);
nothing on the other 5 benchmarks."
*Fix, slide 63:* "human rating of the judge — none on the shipped picks; none at all on 5 of 8 benchmarks".
*Script Q&A:* replace "independent validation covers SLAKE, VQA-RAD and PathVQA, about 11% of questions" with the line above.

**H3 · Slide 35 title "generation dominates, the probe is negligible" — false for the system as run.**
The probe is negligible; the **re-read is not**. `free_head_2026-08-16.json` → `DELIVERABLE_3_cost`: as run, the
features cost 3.81 teacher-forced passes per question (`passes_per_question.distinct_normalised_answers` = 3.8136, on
2,345 questions of 3 benchmarks), each at full resolution = 1.81 greedy-forward equivalents
(`per_forward_flopeq.head_extraction_fullres_DEPLOYED` = 1.8095) → 6.90 FLOP-eq for the re-read alone;
`arms_fixed_bestof8.bo8_head_TODAY.flopeq_measured_geometry` = 14.90 vs `bo8_generation_only` = 8.0 (HF repeat-8
convention). Under vLLM's shared prefill, sampling 8 costs only 1.13× greedy (`bestofn_vllm_2026-09-16.json`
`forward_tokens_per_question.flopeq_shared`), so **as run the re-read is several times the cost of the sampling**.
On the 8 final benchmarks there are 4.42 distinct answers per question (162,885 rows / 36,869 questions, from the
feature-cache metas, `rr_rows.py`), so 3.81 understates the 8-benchmark re-read count. End-to-end: not measured.
*Fix, slide 35 title:* "What it costs: sampling is cheap with a shared prefill; the probe is negligible; the re-read, as
run, is not". *Body, "the re-read, as run":* "3.81 extra teacher-forced passes per question at full resolution
= 6.90 greedy-pass equivalents (free_head_2026-08-16.json, 3 benchmarks); 4.42 distinct answers per question on the 8
benchmarks. Capturing states inside generation removes it (equivalence shown for an earlier probe: 15/2,345 picks change)."
*Q&A:* "Total cost as run vs greedy? Not measured end to end. Components: sampling 1.13× FLOP-eq, 2.74× latency, 3.59×
energy (bestofn_vllm); re-read 6.90 FLOP-eq (free_head, 3 benchmarks)."

**H4 · Slide 52 and script §50/§52 — "VQA-Med (oracle 0.21) cannot gain — coverage limit" contradicts the deck's own
numbers.** VQA-Med's headroom (oracle@8 − greedy) is 0.1155 — 78% of SLAKE's 0.1488 and comparable to VQA-RAD's 0.140;
it has 453 recoverable questions (`extra_numbers.json` `cells.vqamed_open.recoverable`), and the probe fixes 139 while
breaking 153. The failure on VQA-Med (and VQA-RAD) is **selection**, not coverage — which is what slide 45 itself says
("the correct answer appears, the probe cannot find it"). Gain correlates with headroom r = 0.92 but with the oracle
level only r = 0.30 over the 8 benchmarks (`rr_stats_out.json` `corr_gain_headroom`, `corr_gain_oracle`; n = 8, descriptive).
Also "selection efficiency = pick ÷ oracle@8 = 0.81" is flattered by greedy: greedy ÷ oracle is already 0.65–0.83 on most
benchmarks (VQA-RAD scores 0.78 with a zero gain). The honest share is **headroom captured = (pick − greedy)/(oracle −
greedy): macro 0.44**, range −0.03 (VQA-Med) to 0.87 (OmniMedVQA) (`rr_stats_out.json` `gain_over_headroom`).
*Fix, slide 52 title:* "Where it helps: gain follows headroom — and selection fails on VQA-RAD and VQA-Med". *Body:*
"headroom captured (pick − greedy)/(oracle − greedy): PathVQA 0.31 · SLAKE 0.39 · VQA-RAD 0.00 · RadImageNet 0.70 ·
Kvasir-x1 0.62 · OmniMed 0.87 · VQA-Med −0.03 · GEMeX 0.66; macro 0.44." Replace "VQA-Med cannot gain — Snell's hard
questions" with "VQA-Med has headroom (453 recoverable questions) but the probe fixes 139 and breaks 153: a selection
failure." *Script §50:* replace "where a correct answer is rarely among the samples at all" with "where the probe cannot
find the correct sample even when there is one". *Script §52:* delete "there is nothing to gain".

**H5 · Slide 2 "What is new" box — "two independent judges".** The MedVLThinker-32B judge produced the **training labels**;
it cannot be an independent evaluator of a probe trained to imitate it. Only MedGemma-27B is independent of training.
The September 2×2 (`judge_2x2_2026-09-20.json`, AUDIT §2.4) measured the inflation: each probe scores best under its own
training judge; diagonal inflated by +0.005 to +0.024; off-diagonal (cross-judge) +0.0565 to +0.0658. Here, judge −
MedGemma = 0.0088, the size of the CLAUDE.md paraphrase-drift caveat (0.006–0.009).
*Fix, slide 2:* "...image-level cross-fitting, **a second judge from another model family that the probe never trained
on**, a controlled head-to-head...". Make the **cross-family judge the first number** on slide 2 / 49 / 65 or label the
first row "judge (MedVLThinker-32B — also the training-label source)".
*Script §2:* "strict exact string matching also improves, by 0.012" → "**lenient** exact match (equality or containment)
also improves, by 0.012" — 0.012 is the lenient scorer (`em` = run_openvqa.py lenient), not strict.

### MED — gaps a committee would hit

**M1 · Slides 34, 49 ("image half 1 is out of sample for every design choice made in the October round"), script §49.**
Not quite: the cap320-vs-full-resolution verdict (slide 47: +0.0790 vs +0.0745) is a macro over **both** folds
(`select_policy_cap320_vs_fullres_T07.json` `macro.cap320_minus_ref_greedy_judge` / `fullres_...`). Mitigation: it kept
the incumbent. Conversely, half 1 is cleaner than the slide says for **September** choices (layers 18/20/22 were the argmax
of 8 sets on the held-out half 0 — AUDIT §5; T = 0.7, width, BCE were all chosen on half 0).
*Fix, slide 49 note:* "image half 1 — out of sample for the September choices (made on half 0), the readout tie-break and
the temperature map; the October 'keep cap320' decision used both halves." Same edit on slide 34 and in
`final_fold1_headline_2026-10-16.json`'s title if it is quoted.

**M2 · Slides 2, 49, 65 — what "significant" generalises to.** All CIs treat the 8 benchmarks as fixed (resampling images
within benchmark). Treating benchmarks as the sampling unit (t, df = 7, over the 8 per-benchmark deltas): judge
[+0.0277, +0.1319], MedGemma [+0.0248, +0.1171], **EM [−0.0234, +0.0475] — a tie** (computed from `rr_stats_out.json`
cell points). And EM's macro significance rests on two benchmarks: drop OmniMedVQA → +0.0032 [−0.0071, +0.0132];
drop RadImageNet → +0.0034 [−0.0068, +0.0133] (`rr_lobo.py`). Judge LOBO stays in [+0.0683, +0.0918].
*Fix, slide 65 claim:* "... all significant **on these eight benchmarks** (image-clustered); with benchmarks as the unit
(n = 8) the judge gains stay significant and the exact-match gain does not." Add a backup row.

**M3 · Multiple comparisons (slides 49, 50, 65 "6 of 8 significant", "loses on 3").** 8 benchmarks × 5 currencies = 40
per-benchmark intervals, unadjusted. Bonferroni (`rr_stats_out.json` `cells.*.ci_bonf8/ci_bonf40`): the 6 judge wins
survive even ×40 (smallest lower bound SLAKE +0.0141); MedGemma-SLAKE does not at ×40 [−0.0046, +0.0912]; the **EM loss
on GEMeX is not significant after ×8** [−0.0191, +0.0006]; EM-PathVQA's win fails at ×40. Robust EM losses: Kvasir-x1 and
VQA-Med only.
*Fix:* footnote on slide 50: "Bonferroni over 8 benchmarks: judge 6 wins unchanged; exact-match losses: Kvasir-x1 and
VQA-Med (GEMeX becomes a tie)."

**M4 · Slide 62 "uncertainty in the intervals" — refit variance is understated, and template clustering is ignored.**
(a) The CIs condition on the fitted probes. The "~0.003" is a thread-count/seed refit floor (AUDIT §5 "0.0033 thread
floor"); variance over **training-set resampling or a different image-split salt** is not measured. (b) Questions are
clustered by image only, but RadImageNet has ~15 question templates and VQA-Med 18–20 (AUDIT appendix DI-6); errors
correlated within a template are not resampled together, so those cells' CIs are likely too narrow. Not measured.
*Fix, slide 62:* "CIs hold the trained probes and the image split fixed (refitting with other seeds/threads moves results
by up to ~0.003; a different split salt: not measured) and cluster by image, not by question template (not measured)."

**M5 · Slide 53 (memorisation) — the logic does not support "not memorisation".**
(i) September probe, half 0 only — not the shipped probe. (ii) Pooled micro across benchmarks: strata are confounded with
benchmark (novel templates and unseen golds come mostly from GEMeX/Kvasir, the big-gain sentence-gold sets). Within
PathVQA the pattern reverses: repeat-template +0.0669 vs novel +0.0178 [−0.0210, +0.0556] (`template_stratified_gain_2026-09-20.json`
`cells.pathvqa_open.{repeat,novel}.dA`; AUDIT §2.5 also gives gold-unseen +0.0123). (iii) The EM losses in the "unseen"
rows are equally explained by those strata being sentence-gold benchmarks where EM is invalid — so they are not evidence of
judge-specificity either. (iv) The decisive controls (image-shuffle, Hewitt–Liang control task) are not run (slide 63).
*Fix, slide 53 "reading":* "pooled over benchmarks the judge gain survives unseen templates and unseen answers, but
within PathVQA it does not (novel templates +0.018, tie); strata are confounded with benchmark. We claim 'not only an
answer prior', not 'not memorisation'; the image-shuffle control is owed."

**M6 · Slides 26 and 70 — contamination arguments.** (a) Our VQA-Med cell (3,663) "is mostly the 2019 training split"
(`FINAL_METHOD_2026-10-16.md` §6.4), and Lingshu's SFT includes VQA-Med-2019 training (slide 70, Lingshu Table 5) — the
deck never says the cell is training data of the generator. (b) "Exposure affects both arms" is an assumption: memorised
items have peaked sampling distributions, which can change greedy-vs-pick deltas. Not measured. (c) Near-duplicates: dHash
candidates not caught by pixel-md5 exist — SLAKE-train vs held-out 100, VQA-RAD-train 12, PathVQA-train 7, Kvasir-VQA 5
(AUDIT appendix DI-5, not visually verified).
*Fix, slide 26 "exposure":* "...VQA-Med 2019. **Our VQA-Med cell is mostly the 2019 training split** — data Lingshu was
fine-tuned on. Exposure enters both arms, but whether it changes the delta is not measured." *Slide 34/48:* "never saw its
image (exact pixel match; near-duplicate candidates by perceptual hash exist and were not resolved — AUDIT DI-5)."

**M7 · Slide 51 / Part VII — VQA-Med harm is not in the limits.** On VQA-Med the selector changes the answer on 72.2% of
questions (`rr_stats_out.json` `pick_differs_from_greedy`), breaks 44% of greedy's correct answers (153 vs 139 fixed), and
its EM loss is significant (−0.0137 [−0.0227, −0.0049]). Across all 8 the pick differs from greedy on 23–72% of questions.
*Fix, slide 62, new row:* "harms where selection fails — on VQA-Med it breaks 153 greedy-correct answers for 139 fixes (EM
loss significant); there is no per-benchmark guarantee of 'never worse than greedy'." (Report it as a limit only; the method
always answers, and no fix that withholds an answer should be proposed — CRITICAL RULE 6.)

**M8 · Slide 65 "what generalises: training — not size or prompting — makes a selector work".** The size evidence (slide 38)
is a **tie**: zero-shot Lingshu-32B verifier vs trained 7B verifier +0.005 [−0.023, +0.032], n = 600, July. A tie shows
training at 7B *matches* size, not that size does not work. *Fix:* "a trained 7B selector matches a zero-shot 32B
verifier (tie, n = 600, July) and beats prompting the model itself; selection gains follow headroom where the probe can
find the right sample."

**M9 · Slide 65 script — "beats the one published medical selector we could compare on identical candidates".** Drops the
qualifiers on the slide: 2 of 3 benchmarks (VQA-RAD tie), WED adapted to our pools, WED is training-free and ours uses
in-domain labels. *Fix, script §65:* "...and on identical candidates beats the training-free selector WED on PathVQA and
SLAKE, ties on VQA-RAD."

**M10 · Slide 54 (Qwen replication).** Add the EM CI and the protocol: September recipe, **single layer 20, 5 seeds, image
half 0 only, rank readout, same MedVLThinker-32B judge** (`replication_currency_2026-09-20.json` `qwen.layer/seeds`);
lenient EM +0.0276 [+0.0168, +0.0384], strict EM +0.0187 [+0.0076, +0.0298]; EM significant on only 3/8 benchmarks; Qwen
greedy is weak (macro judge 0.303) and after best-of-8 is still below plain Lingshu greedy on 4/8 (AUDIT §3).
*Fix, slide 54 title:* "A second training recipe of the same LM: Qwen2.5-VL-7B" (not "replication on a second generator").

### LOW

- **L1 · Slide 49 title** "significant gains in every currency" → "...in every currency (macro); per benchmark exact match
  loses on 2–3". **L2 · Macro vs micro:** add backup line — micro (question-weighted) judge +0.1097 [+0.1058, +0.1136],
  MedGemma +0.0985 [+0.0944, +0.1025], EM +0.0098 [+0.0064, +0.0131] (`rr_stats_out.json`); macro is the conservative
  choice for the judges and slightly favourable for EM. **L3 · What ships:** the evaluated estimator is two half-data probe
  sets (`ckpts/train/final_probe_2026-10-16/fold{0,1}`); a deployed probe would be trained on all images — say which.
  **L4 · Slide 51:** "unavoidable" harms = greedy right but no sample right; say so in the header. **L5 · Slide 45/46:**
  T = 0.7 beat 1.0 by +0.0063 at the edge of the tie band in September (AUDIT §5) — "project default" is the honest reason,
  as the slide already says.

---

## 2. Ten hardest committee questions — with sourced answers

1. **"Your probe is trained on the judge's labels and you report that judge first. Isn't +0.08 partly the probe learning the
   judge?"** — Yes, partly. The cross-family judge, which the probe never trained on, gives +0.0710 [+0.0622, +0.0797]
   (`final_eval_FINAL_cap320_T07.json` `macro_over_open8.all.pick_minus_greedy_xjudge`); the gap to +0.0798 (0.0088) is the
   size of the paraphrase-drift effect we measured before. In a September 2×2, each probe scored best under its own training
   judge, and the cross-judge gain was +0.0565 to +0.0658 (`judge_2x2_2026-09-20.json`). The number to quote is the
   cross-family one.
2. **"Has any human checked these answers?"** — Not on this method's picks. Earlier checks: an LLM adjudicator on SLAKE/VQA-RAD,
   and 50 + 90 hand-checked PathVQA items, judge error 0.04 with a Wilson upper bound of ~0.20 (`pathvqa_judge_audit.json`).
   Nothing at all on 5 of 8 benchmarks. A ~300-item stratified human rating is planned (NEW_DIRECTIONS D2), not done.
3. **"Exact match gains 0.012. Is that real or generalisable?"** — On these 8 benchmarks it is significant (image-clustered).
   It depends on OmniMedVQA and RadImageNet: drop either and it is a tie (+0.0032 [−0.0071, +0.0132]; +0.0034 [−0.0068,
   +0.0133]). With benchmarks as the unit (n = 8) the EM interval is [−0.0234, +0.0475]. On Kvasir-x1/GEMeX the golds are
   LLM-written sentences where EM rejects paraphrases (strict oracle@8 EM on Kvasir-x1 is 0.0004, AUDIT §2.2); without those
   two, EM macro is +0.0240 [+0.0120, +0.0356] (`rr_lobo.py`) — post hoc, so context only.
4. **"Your CIs hold the trained probes fixed. What is the real uncertainty?"** — Refitting with other seeds/thread counts moves
   macros by up to ~0.003 (AUDIT §5). Resampling the training set or the image split was not measured. Fold-to-fold
   difference (different probes, different questions) is +0.0747 vs +0.0846 — the same order as the CI width.
5. **"Were hyperparameters tuned on the test data?"** — September choices (layers, width, loss, T) were made on image half 0,
   which is fold 0 of the headline. Half 1 was never used to choose those; the October 'keep cap320' decision did use both
   halves, but kept the incumbent. Half 1 alone: +0.0846 [+0.0729, +0.0965] (`final_fold1_headline_2026-10-16.json`), larger
   than half 0.
6. **"Could the 4 extra training domains leak test images?"** — Exact (pixel-md5) overlaps exist only between VQA-RAD-train and
   VQA-Med / VQA-RAD-closed / VQA-Med-2019-test and are removed (261 / 377 rows per fold; reproduced exactly). Kvasir-VQA vs
   Kvasir-VQA-x1: zero exact matches. Perceptual-hash near-duplicates were flagged (SLAKE-train 100 candidates) and not
   resolved (AUDIT DI-5) — owed.
7. **"Questions repeat across images. Isn't the probe learning templates?"** — 55–100% of held-out questions on five
   benchmarks reuse a training template. Pooled, the gain is as large on novel templates, but within PathVQA it collapses
   (+0.067 repeat vs +0.018 novel, tie). And the CIs do not cluster by template. We can claim "not only an answer prior";
   the image-shuffle control that would settle it is not run.
8. **"What does it cost compared with greedy?"** — Sampling 8 with a shared prefill: 1.13× FLOPs, 2.74× latency, 3.59× energy
   (`bestofn_vllm_2026-09-16.json`). As run, re-reading each distinct answer adds 3.81 passes at full resolution = 6.90 greedy
   equivalents (`free_head_2026-08-16.json`); capturing states during generation removes that. End-to-end: not measured. vs
   Lingshu-32B greedy (earlier probe, 4 benchmarks, not cost-matched): judge 1 loss / 2 ties / 1 win; EM 2 losses.
9. **"Why does it fail on VQA-Med and VQA-RAD, and does it hurt there?"** — Not for lack of correct samples: VQA-Med has 453
   recoverable questions; the probe fixes 139 and breaks 153 (`extra_numbers.json`), and EM loses significantly. VQA-RAD: 17
   fixed, 17 broken. Selection fails there; headroom captured is −0.03 and 0.00. VQA-Med is also mostly the 2019 training
   split, which Lingshu was fine-tuned on.
10. **"Is the Qwen result an independent replication?"** — No. Qwen2.5-VL-7B is Lingshu's base model, the recipe is the September
    one (single layer, 5 seeds, half 0), and the judge is the same. Judge +0.0814 [+0.0708, +0.0924], lenient EM +0.0276
    [+0.0168, +0.0384]. MedGemma-4b was withdrawn (broken sampling). A genuinely different generator is owed.

---

## 3. Suggested replacement for the summary claim (slide 65)

> A 918k-parameter MLP probe on frozen Lingshu-7B hidden states, used as a best-of-8 verifier at T = 0.7, improves open-ended
> medical VQA over the model's own greedy answer on 8 benchmarks (36,869 questions, each scored by probes that never saw its
> image): **+0.0710 under a cross-family judge it was not trained on** (+0.0798 under the judge that supplied its training
> labels), **+0.0121 lenient exact match** — all significant on these benchmarks; with benchmarks as the unit, the judge gains
> hold and the exact-match gain does not. Under the judge it wins on 6 of 8 benchmarks and ties on 2, where selection — not
> coverage — fails. On identical candidates it beats the training-free selector WED on PathVQA and SLAKE and ties on VQA-RAD.
> No human evaluation of the picks; the as-run cost includes a re-read pass (6.90 greedy equivalents).
