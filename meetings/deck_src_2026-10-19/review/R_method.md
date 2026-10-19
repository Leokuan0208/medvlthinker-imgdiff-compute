# R_method — professor-style attack on Part I (vocabulary, slides 4–13) and Part III (method, slides 24–35)

Reviewer stance: a demanding deep-learning professor who reads code. Every claim below was checked against the code that
produced the headline (`src/exp_2026_10/final_method_eval.py`, `src/training_methods/extract_generator_hidden.py`,
`src/training_methods/head_sweep.py`, `src/training_methods/genframe_data.py:rank_avg`,
`src/audit_2026_09_18/em_rescore_pooled_probe.py`, `src/labeling/run_openvqa.py`, `src/labeling/run_judge.py`) and against
`final_eval_FINAL_cap320_T07.json`.

Numbers marked **[computed in review]** are not in any existing artifact. I derived them in this review from files on disk
(the file is named each time), and the derivation reproduces the artifact's `folds.*.train_rows` exactly
(115,934 / 116,000). If they go on a slide, cite the file named, or have them written into an artifact first.

Severity: **HIGH** = wrong or misleading as it stands · **MED** = a gap a professor would hit · **LOW** = polish.

---

## A. Findings, slide by slide

### 1. HIGH — Slide 31 (and 25, 69): the loss is *class-weighted* BCE, not the plain BCE written on the slide
**Code.** `head_sweep.fit → bce_step`: `pos = ytr.mean(); pw = (1-pos)/pos; nn.BCEWithLogitsLoss(pos_weight=pw)`.
So positives are up-weighted by w₊ = (1 − π)/π, where π is the positive rate of that fold's training rows.
**[computed in review]** from the `feats_hidden/generator_*_finelayer*.meta.json` rows that the fold filter keeps:
π = 0.2071 (fold 0) and 0.2085 (fold 1), so w₊ ≈ 3.83 and 3.80.
**Fix, slide 31 formula:**
$$\mathcal L=-\frac1{|B|}\sum_{j\in B}\Big[w_+\,c_j\log\sigma(s_j)+(1-c_j)\log\big(1-\sigma(s_j)\big)\Big],\qquad w_+=\frac{1-\pi}{\pi}\approx3.8$$
with the label "π = share of correct rows in the fold's training set (≈ 0.21)".
**Script, slide 31.** Replace "Standard optimiser settings;" with: "The loss is binary cross-entropy with the correct answers
up-weighted, because only about one training answer in five is correct. Each correct answer counts about 3.8 times. The
optimiser is AdamW with learning rate 10⁻³ and weight decay 10⁻², run for 30 epochs at batch size 256, with no early stopping."
Also change slide 25's "trained with BCE" to "class-weighted BCE", and slide 69's "BCE" to "BCE, pos_weight = (1−π)/π".

### 2. HIGH — Slide 30: "σ(s) ≈ P(this answer is correct | its hidden state)" and "score = logit of P(correct)" overclaim calibration
The weighted loss (item 1) shifts the logit by about log w₊. The probes are trained for 30 epochs with no early stopping.
On the deck's own worked example the *correct* answer "Large Bowel." gets logits from −1.451 to −8.712 across the 24 probes
(`worked_example_12231.json`), so σ(s) ≤ 0.19 for an answer that every probe ranks first or tied-first.
A professor who asks "so the probe thinks the right answer has a 0.03 % chance?" catches this immediately.
**Fix, slide 30 "reads as" box:** "σ(s) is a *score*, monotone in the probe's belief that the answer is correct. It is **not
a calibrated probability**: the loss up-weights correct answers, and only the order within a question is used."
Figure label: change "score = logit of P(correct)" to "score s (uncalibrated logit)".

### 3. HIGH — Slide 27: the deduplication rule is wrong
The slide says answers are "normalised (case, punctuation, articles)". **Code.** Candidate rows are keyed by
`norm(s) = str(s).strip().lower()` (`extract_generator_hidden.py:66`). Votes are counted with the same rule
(`em_rescore_pooled_probe.row_norm`, used in `final_method_eval.py`). Punctuation and articles are **kept**:
"Large Bowel." and "large bowel." merge, but "Large Bowel." and "Large Bowel" do not. (The article/punctuation normaliser
is the *exact-match grader* `em_norm`, a different function.) The model then re-reads the **surface form of the first
occurrence** in pool order (`r["ans"]`), with its case and trailing period. That is why the worked example's candidates are
"Large Bowel." etc.
**Fix, slide 27:** "deduplicate, keep votes — answers are merged if identical after lower-casing and trimming whitespace
(punctuation and articles are kept); the first-sampled spelling is the one re-read; its count vᵢ is kept."
**Also update the 4.4 source.** **[computed in review]** On the final T = 0.7 pools themselves: 162,885 candidate rows /
36,869 questions = 4.42 distinct answers per question (`feats_hidden/generator_eval_finelayer*.meta.json`). This replaces
the September `recipe.json` ratio (81,331 / 18,417).

### 4. HIGH — Slide 8 (and 13): "selected ÷ oracle@N = 0.81" contradicts the two bars on the same slide
On the slide, 0.473 / 0.559 = **0.85**. The 0.81 is the **mean of the eight per-benchmark ratios** (0.8077, NUMBERS_T07.md,
"selection efficiency macro (mean of pick/oracle over 8)"). **[computed in review]** The ratio of the macros is
0.4734 / 0.5586 = 0.8475 (`final_eval_FINAL_cap320_T07.json`, cells.*.all). A professor will divide the two numbers in
front of him.
**Fix, slide 8 box:** "selection efficiency (ours) = selected ÷ oracle@N, computed **per benchmark, then averaged**: 0.81.
(The ratio of the two macro bars, 0.473/0.559, is 0.85.)"
**Fix, glossary:** "selected ÷ oracle@N, per benchmark, then macro-averaged".

### 5. HIGH — Slide 11 (and 2, 34): the "exact match" definition is wrong, and the name misleads
**Code** (`em_rescore_pooled_probe.em_score`, verbatim from `run_openvqa.py`):
1. Normalise: lower-case, drop the words *the, a, an, is, are, of, in, on, at, this, image, picture*, strip punctuation.
2. Score correct if the strings are equal, **or if either one contains the other**, whether as a substring or as a word.

So "bowel" is scored correct against gold "large bowel". The slide's "the gold string is contained in the answer" gives
only one direction. "Cannot be fooled by paraphrase" holds, but containment *can* be gamed. Someone from QA (SQuAD) hears
"exact match" as equality after normalisation, and that is not this metric.
**Fix, slide 11 EM line:** "lenient match ('EM' in our tables) — after normalisation (lower-case, punctuation and a
few stop-words removed), correct if equal **or if either string contains the other**. Paraphrase cannot fool it, but a
partial answer contained in the gold can pass."
**Script, slide 2:** "strict exact string matching" → "a lenient string-matching score".
**Q&A:** "Strict exact match (equality only, `strict_em` in the same module) has not been computed for the final method —
not measured."

### 6. HIGH — Script slide 32: an unsourced number ("plus or minus eight, another's a tenth of that")
This breaks the no-fabricated-numbers rule, and the deck's own data contradicts it: in `worked_example_12231.json` every
score is negative.
**Replace with:** "On this very question, the spleen's score is −37.4 under one layer-18 probe and −7.2 under a layer-22
probe (worked_example_12231.json). The raw numbers are not comparable across probes, but the order is."
Then add the honest line from slide 43: "When we tested averaging raw, standardised or calibrated scores instead, all of
them tied with rank averaging, so ranks are the scale-free default, not a measured win."

### 7. HIGH — Slides 26, 31, 34: the training-data description leaves out a source
**Code.** `final_method_eval.py` fits on "every cell's T=0.7 cache". The FINAL run's `args.cells` includes the four
**comparability cells**: `vqa_rad_closed`, `pathvqa_closed`, `slake_closed` (closed-ended, mostly yes/no) and
`vqamed2019_test`. So their other image half is also training data.
**[computed in review]** These cells contribute 3,366 (fold 0) and 3,325 (fold 1) training rows
(`feats_hidden/generator_eval_finelayer_{vqa_rad_closed,pathvqa_closed,slake_closed,vqamed2019_test}.meta.json`, half filter).
The "4 extra domains" are not whole training splits either. They are the July LoRA-era matched subsample
(`generator_train_finelayer`): 6,029 questions / 31,498 rows (PathVQA-train 22,390, Kvasir-VQA 5,562, SLAKE-train 2,207,
VQA-RAD-train 1,339) **[computed in review, from generator_train_finelayer.meta.json]**. They are used **from both image
halves**, minus any image in the scored half (`Forig[i] or Fhalf[i]==1-f`).
**Fix, slide 31 "training rows":** "the opposite image half of the 8 open benchmarks **and of the 4 closed/VQA-Med-2019
comparison sets**, plus a fixed subsample of 4 extra domains (6,029 questions: Kvasir-VQA and the training splits of
PathVQA, SLAKE, VQA-RAD). 115,934 rows (fold 0), 116,000 (fold 1); any row whose image is in the scored half is removed."
Change slide 34's boxes and slide 26's footer line to match. Script slide 31: "the other half of the images of all eight
benchmarks, of the closed-question comparison sets, plus four extra domains."

### 8. MED — Slide 30 / 7: σ and z are each used for two different things
σ means the sigmoid and also the per-dimension std on the same slide (`(h̄ − μ) ⊘ σ`, σ^{(ℓ)}, then σ(s)). z means the
logits on slide 7 (p_i = e^{z_i/T}/…) and the standardised feature vector on slide 30. The code also adds ε = 10⁻⁶ to the std.
**Fix:** on slide 30 write $\tilde h_i^{(\ell)}=(\bar h_i^{(\ell)}-\mu^{(\ell)})\oslash(s^{(\ell)}+\epsilon)$ with
"s^{(ℓ)} = per-dimension std, ε = 10⁻⁶", and keep σ(·) for the sigmoid only. Then
$s_i^{(\ell,k)} = w_2^\top\mathrm{GELU}(W_1\tilde h_i^{(\ell)}+b_1)+b_2$, and rename the score letter if s clashes (e.g.
$f_{\ell,k}(\cdot)$). On slide 7, define "zᵢ = logit of token i" in a caption, or rename it uᵢ. Also check the KaTeX render
on slide 7: in the screenshot the exponent shows as "zi/T" with no visible subscript.

### 9. MED — Slide 32: n is never defined, the ranking direction is not stated, and the n = 1 edge case is not stated
**Fix, slide 32 caption:** "n = number of distinct answers for this question (1 ≤ n ≤ N); rank = 0-based position in
ascending score order, average for tied scores; for n = 1 the single answer is returned (code divides by max(n−1, 1))."
Add a second tie rule. The code takes `top[np.argmax(votes[top])]`, so if the vote counts also tie, the **first row** wins.
Rows are sorted by lower-cased answer string, so that is the alphabetical first. Say: "ties in rank → more votes; ties in
votes too → first in alphabetical order (arbitrary but deterministic)."

### 10. MED — Slide 33: the N = 4 panel shows a configuration that was never measured, and the box labels are garbled
No N = 4 result exists in DESIGN_EVIDENCE / NUMBERS_T07. The panels read "4 samples → ≤ n distinct answers", which is
backwards. The lines also run from every **sample** to the probes, but the probes score **distinct answers**.
**Fix:** relabel to "4 samples → n ≤ 4 distinct answers" (and n ≤ 8, n ≤ 16). Draw the lines from a deduplicated column,
or caption "lines = distinct answers after merging". Mark the N = 4 panel "(illustration — N = 4 not measured)".
**Script:** "with four samples every probe scores at most four answers (N = 4 itself we have not measured)".

### 11. MED — Slide 25: the figure hides two facts the professor needs
(a) Step 2 needs the image and question again: the re-read is over [system; image; question; yᵢ]. The figure only routes
the answers into the re-read box. **Fix:** draw a thin line from the input panel into box 2, labelled "image + question".
(b) The re-read runs at a different resolution, which is the deck's main protocol caveat. **Fix:** under "Re-read & tap",
add "HF pass at 1,003,520 px (generation: 250,880)".
(c) The legend "❄ frozen (never trained)" is wrong for Lingshu, which its authors trained. **Fix:** "❄ frozen (not trained
by us)". Use the same wording on slide 6.

### 12. MED — Slide 28 / script: say explicitly how the image reaches a probe that reads only answer tokens
Professor: "Your probe ignores the image tokens — so how is it image-aware?" Script slide 28: add after "…the formula at the
bottom": "The answer tokens come after the image and the question, and the model attends causally to everything before
them. So each answer vector already mixes in the image and the question; the probe reads them through context."
Also add the template tokens to the figure: there are `<|im_end|><|im_start|>assistant\n` tokens between the question and
the answer.
**Fix the "≈ 20" system-prompt count.** **[computed in review]** With the Lingshu tokenizer, the system text is 22 tokens,
and 30 tokens precede `<|vision_start|>` once the chat template is included. The slide's "≈ 20" has no source. Replace it
with "30 (incl. chat template)".
State the span precisely: **the answer's own tokens, including any trailing period, excluding the end-of-turn token**
(`s0 = len(ids) - n_ans_tok`; nothing is appended after the answer). I checked that "Large Bowel." tokenises identically
alone and after "assistant\n" (4 tokens), so the span boundary is exact.

### 13. MED — Script slide 28 promises evidence that the next slides do not contain
"the next slides show why we think it [resolution mismatch] does not matter" — slides 29–35 show only the
capture-vs-teacher-forcing test, which was **at the same resolution**. **Replace with:** "The direct test, re-extracting
these states at the generation resolution, is running now. Indirect evidence: when we regenerated the samples at full
resolution, so that generation and features matched, the method gave +0.0745 instead of +0.0790. Matching did not help,
but that test also changed the samples (FINAL_METHOD_2026-10-16.md, full-resolution row)."

### 14. MED — Slides 2, 25, 30, 33: "24 probes" — for evaluation there are 48 probes and no single deployable set
Each fold has its own 24 (`ckpts/train/final_probe_2026-10-16/fold0`, `fold1`). No probe set has been trained on all
images for deployment. Slide 30's source line also names the **September** probe (`genframe_head_pooled_ens_v2`), not the
final one.
**Fix:** slide 30 source → `ckpts/train/final_probe_2026-10-16/fold{0,1}` (recipe.json: "Linear(3584->256)-GELU-Linear(256->1)").
Add a Q&A (section B). Script slide 34: "…so we train two sets of 24 probes, one per fold."

### 15. MED — Slide 11: MedGemma-27B is described as "multimodal", but both judges run text-only with the same prompt
`run_judge.py` (used for both judges; MedGemma via `--judge_model`, `launch_medgemma27b_train.sh:13`) builds a
text-only chat with JUDGE_SYS. It runs at temperature 0 and labels **correct iff P(Yes) ≥ P(No) on the first token**.
**Fix, slide 11 table:** "MedGemma-27B-it | independent second grader, **used text-only, same prompt** | Gemma 3". Add
"both judges: text-only, see question + gold + candidate, label = P(Yes) ≥ P(No)".

### 16. MED — Slide 26 / script: "exposure cancels out" is asserted, not shown
A professor will say memorisation can *interact* with selection, so it need not cancel. The deck has the answer: the gains
are **larger** on the four benchmarks Lingshu was not trained on than on the four it was. Judge Δ: RadImageNet +0.1335,
OmniMedVQA +0.1604, GEMeX +0.1244, Kvasir-x1 +0.1143, versus PathVQA +0.0524, SLAKE +0.0574, VQA-RAD +0.0000,
VQA-Med −0.0038 (NUMBERS_T07.md per-benchmark table). **Script, slide 26:** replace "so that exposure cancels out" with
"so exposure affects both arms. And the gain is not coming from exposure: it is largest on the four benchmarks Lingshu
never trained on."

### 17. MED — Slide 9: ORM/PRM attributed to Snell 2024
Snell et al. use the terms, but the ORM/PRM distinction is from Uesato et al. 2022 ("Solving math word problems with
process- and outcome-based feedback") and Lightman et al. 2023 ("Let's Verify Step by Step"). **Fix source line:**
"ORM/PRM: Uesato et al. 2022; Lightman et al. 2023". Check these citations against the papers before adding them; they
are not in the PAPER_NOTES files.

### 18. MED — Slide 12: P(True) listed as an uncertainty method that "cannot pick"
P(True) is **per answer**, so it *can* rank candidates. It is exactly the self-verification selector on slide 9, which
failed in July. **Fix:** in the examples row, write "P(True) (per-answer — usable as a selector; that is slide 9's
self-verification row)".
Also pre-empt: "If calibration cannot reorder, why test Platt averaging on slide 43?" Answer: "Calibrating one probe cannot
reorder its own scores. Calibrating each of 24 probes *before averaging* can change the average's order. That is what
slide 43 tested, and it tied."

### 19. MED — Slide 34: half naming is inconsistent, and the macro-CI construction is not stated
The diagram says "half A / half B", the footnote says "Image half 1". In the code, half = md5(…) mod 2 ∈ {0, 1} and
fold f scores half f, so A = 0 and B = 1. **Fix:** "half A (= 0) / half B (= 1)" in the diagram, and "Image half B (= 1)
alone…" in the note.
Add to the macro + CI box: "each benchmark is resampled independently (10,000 image-level resamples). The macro CI is the
2.5/97.5 percentile of the average of the 8 resampled deltas. The probes and the fold split are held fixed."

### 20. MED — Slide 29: the gray bars are unlabelled
The +0.1008 and +0.1020 bars (layers 21 and 19) have no layer label, so the chart looks as if 18/20/22 had been compared
with nothing. **Fix:** label them "L21", "L19" (source `head_ens_width_2026-08-25.json`, DESIGN_EVIDENCE §2 row "which layers").

### 21. MED — Terms the field uses that are missing
* Rank averaging is **Borda-count rank aggregation** (with average ranks for ties). Add it to slide 13 and say it once in
  the script: "rank averaging, which is a Borda count".
* "8 seeds" is a **deep (seed) ensemble** (Lakshminarayanan et al. 2017), combined here with a layer ensemble. Add to the
  glossary.
* Cross-fitting = **2-fold grouped cross-validation / out-of-fold prediction (GroupKFold by image)**. Add "a.k.a." on
  slide 34; ML professors know the CV name better than the double-ML name.
* "Teacher forcing" usually names the RNN *training* technique. Here it means one parallel forward pass over the fixed
  prompt + answer, the same pass used for likelihood scoring. Glossary: "teacher-forced pass — one forward pass over
  prompt + a given answer (as in likelihood scoring); no sampling".
* z-scoring / standardisation is not in the glossary. Add "z-score: subtract the training mean and divide by the
  training std, per dimension".

### 22. LOW — Slide 6 / 28 / 29: hidden_states[28] is not the raw residual stream
In HF Qwen2, the last entry of `output_hidden_states` is taken **after the final RMSNorm**. Indices 1–27 are raw block
outputs. Our layers (18/20/22) are unaffected, but "ℓ = 28 = residual stream after block 28" is slightly wrong. **Fix
(footnote, slide 29):** "hidden_states[28] is post-final-norm in HF; our layers 18–22 are raw residual-stream outputs."

### 23. LOW — Slide 7: "each individual sample gets less accurate" (script) has no number on the slide
Either cite a per-sample accuracy by T, or soften to "and, typically, each individual sample is less accurate (not shown
here)". Also: mean distinct at T = 0.7 is 4.17 on slide 7 (held-out halves, head_temp_ensemble) and 4.4 on slide 27.
Add a word that they are different question sets, or use the 4.42 from item 3 on slide 27.
Add to the sampling box: "pure temperature sampling — no top-p, no top-k (run_openvqa.py SamplingParams: temperature,
max_tokens = 64, n)".

### 24. LOW — Script slide 35: "a few million multiply-adds per answer"
24 probes × 918,017 parameters ≈ 22 million multiply-adds per distinct answer (2 × that in FLOPs). **Fix:** "about
22 million multiply-adds per answer — still microseconds". Also add: "and the re-read could stop at block 22; we ran
all 28" (cost saving available, not measured).

### 25. LOW — Slide 2 / script slide 2: "24 such probes vote by rank"
"Vote" collides with majority vote, which the method beats. **Fix:** "the 24 probes' within-question ranks are averaged".

### 26. LOW — Slide 30: two notations for the same score
Slide 30 uses both $s^{(\ell,k)}(x,y)$ and $s_i^{(\ell,k)}$. Keep $s_i^{(\ell,k)}$, with "i indexes the distinct answers of
question x".

### 27. LOW — Slide 11: token-F1 is set-based
`token_f1` uses **sets** of words (`set(em_norm(...).split())`), not the multiset count SQuAD uses. One word in the
glossary: "token-F1 (word-set overlap)".

### 28. LOW — Script slide 34: "Cross-fitting doubles the scored sample, which is why exact match is now significant"
The method also changed since September. Half B alone is already EM-significant: +0.0165 [+0.0026, +0.0299]
(`final_fold1_headline_2026-10-16.json`, via NUMBERS_T07.md). **Fix:** "Cross-fitting doubles the scored sample; exact
match is significant on both halves pooled and on half B alone."

---

## B. Questions the slides + script + "If asked" do not answer — with the answer to give

**Q1. Is your loss plain BCE? Is σ(s) a probability?**
A: "Class-weighted BCE: correct answers are up-weighted by (1−π)/π, about 3.8, because about 21 % of training answers are
correct (head_sweep.fit, pos_weight; π computed from the fold's training rows). So σ(s) is a score, not a calibrated
probability; we use only the order within a question."

**Q2. How exactly are the eight answers deduplicated? Which spelling does the model re-read?**
A: "Lower-case and trim whitespace, nothing else (extract_generator_hidden.py norm). The first-sampled spelling is the one
re-read, and the judge labels that spelling. Votes are counted with the same rule."

**Q3. What exactly is in the pooled span? Is the end-of-turn token included?**
A: "Only the answer's own tokens, as the tokenizer splits the answer string, including a trailing period. The end-of-turn
token is not included, because nothing is appended after the answer. The span is the last n_ans_tok positions of the
sequence (extract_generator_hidden.py, generator mode)."

**Q4. How does a probe that reads only answer tokens see the image?**
A: "Through causal attention: every answer position attends to the image and question tokens before it. Explicit image or
question means added at most +0.0018, below refit noise (DESIGN_EVIDENCE §2)."

**Q5. Which probes would you actually deploy? You have two folds.**
A: "For evaluation there are two sets of 24, one per fold, each trained without the images it scores
(ckpts/train/final_probe_2026-10-16/fold0, fold1). Either set is a deployable model trained on half the benchmark images.
A set refit on all images has not been trained or measured."

**Q6. Your layers were chosen on the held-out half. Isn't the headline optimistically biased?**
A: "Partly, for half A. That is why we report half B alone, which was out of sample for every October choice: judge +0.0846
[+0.0729, +0.0965], MedGemma +0.0721 [+0.0601, +0.0841], EM +0.0165 [+0.0026, +0.0299]
(final_fold1_headline_2026-10-16.json). It is not smaller than the pooled +0.0798. The September layer choice (on half A)
is also within noise of every alternative (slide 29)."

**Q7. What is the training data, exactly?**
A: "Per fold: the opposite image half of the 8 open benchmarks and of the 4 comparison sets (3 closed-question splits +
VQA-Med 2019 test), plus a fixed 6,029-question subsample of 4 extra domains. 115,934 / 116,000 rows; any row whose image
is in the scored half is removed." (Item 7; the row breakdown was computed in this review.)

**Q8. Is the label circular? You train on the judge you evaluate with.**
A: "Partly, which is why we have the second judge. The 2×2 experiment trained on one judge and graded with the other: the
cross-judge gain is +0.057 to +0.066, and 7–30 % of a judge-currency gain is judge-specific (judge_2x2_2026-09-20.json, via
DESIGN_EVIDENCE §5)."

**Q9. How does the judge decide — sampling, or a threshold?**
A: "Deterministic: temperature 0, text-only, prompt in run_judge.py. Correct iff the first-token probability of 'Yes' is
at least that of 'No'. MedGemma-27B runs through the same script and prompt."

**Q10. How often is there nothing to choose — all eight samples the same?**
A: **[computed in review]** "On the final pools 16.3 % of all questions (6,026 / 36,869) have one distinct answer; per
benchmark from 0.7 % (VQA-Med) to 56.6 % (SLAKE) (feats_hidden/generator_eval_finelayer*.meta.json). There the method
returns that answer, which may differ from greedy."

**Q11. Is the greedy answer in the candidate set?**
A: "No. The eight candidates are samples only; greedy is a separate decode and the baseline. Adding greedy as a ninth
candidate has not been tested."

**Q12. Any top-p / top-k / repetition penalty in sampling?**
A: "SamplingParams sets temperature, max_tokens 64 and n only (run_openvqa.py:186), so it is plain temperature sampling
with vLLM's defaults (top-p 1, top-k off)." Caveat for the student: I have not confirmed that vLLM 0.9.0.1 ignores the
model's generation_config when SamplingParams is built by hand. If pressed, say "we did not set any; not separately
verified".

**Q13. What if the mean rank ties and the votes tie too?**
A: "The answer that comes first alphabetically (after lower-casing) wins: an arbitrary but deterministic rule
(final_method_eval.py, `top[np.argmax(votes[top])]`). Rank ties occurred on 0.58 % of questions for the September probe;
double ties: not measured."

**Q14. Do your CIs include the randomness of training the probes, or of the fold split?**
A: "No. The bootstrap holds both fixed and resamples images within each benchmark. Refit noise is about 0.003. Fold-split
variance: not measured. The macro CI averages eight independent per-benchmark resamples and is conditional on these eight
benchmarks."

**Q15. Micro vs macro — what is the question-weighted gain?**
A: **[computed in review]** "Weighting by question count instead of by benchmark: judge +0.1097, MedGemma +0.0985,
EM +0.0098 (Σ nᵢΔᵢ / 36,869 from cells.*.all in final_eval_FINAL_cap320_T07.json; no CI computed). We report macro so
the 10,121-question Kvasir set cannot dominate; micro is higher on the judges and slightly lower on EM."

**Q16. Was the image deduplication only exact?**
A: "Yes. Images are hashed on decoded RGB pixels, so exact duplicates across benchmarks fall in the same half and are
removed from training. Near-duplicates (re-encoded or resized copies) are not detected: not measured."

**Q17. Hyperparameters: learning rate, epochs, early stopping — how chosen?**
A: "No early stopping: fixed 30 epochs. Width, depth, dropout and weight decay came from a 147-configuration sweep with
5-fold image-grouped CV **inside the training pool** (four-domain, layer-21 era), so the test benchmarks were not used.
The learning rate for BCE was never swept, and neither was the number of seeds (DESIGN_EVIDENCE §3 caveats, slide 63)."

**Q18. Why mean pooling and not attention or max pooling?**
A: "Mean beats the last token at every layer from 14 up (DESIGN_EVIDENCE §2). Max and learned/attention pooling were not
tested; that is on the gap list."

**Q19. How many image tokens did the probe's feature pass actually see?**
A: "Up to 1,280 (max_pixels 1,003,520 / 784). Recorded only for some caches: PathVQA averages 485.4 image tokens, max
1,225 **[computed in review, feats_hidden/generator_eval_finelayer_pathvqa_open.npz n_img_tok]**. At generation, PathVQA
is capped at 320. Other benchmarks: not recorded in the cache."

**Q20. Is σ(s) at least useful as a confidence across questions (AUROC)?**
A: "Not evaluated for the final probe. Within-question AUROC and cross-question calibration are on the diagnostics gap
list (slide 63). We claim only selection accuracy."

---

## C. Checked and correct (no change needed)
* Layer indexing: `hs[L]` with `output_hidden_states`, index 0 = embeddings, so layer 18 = output of block 18. Matches the
  deck.
* Probe: Linear(3584, 256) → GELU → Dropout(0) → Linear(256, 1), 918,017 parameters; AdamW lr 1e-3, wd 1e-2, 30 epochs,
  batch 256; seeds 0–7 (`torch.manual_seed`); per-fold, per-layer, per-dimension training mean/std (+1e-6).
* Prompt in context = SYS_GEN, the generation system prompt; feature pass at max_pixels 1,003,520 (HIGH_PX), min 3,136.
* Readout: mean over 24 probes of `rank_avg` (average ranks for ties, ÷ max(n−1, 1)); argmax; vote tie-break.
  Worked-example ranks and means (0.917 / 0.583 / 0.000; 20 of 24 top) reproduce from `worked_example_12231.json`.
* Every distinct pool answer has a scored row: 0 unscored answers across six cells checked **[computed in review]**.
* Split: half = md5("nd" + pixel-md5) mod 2; fold f trains on half 1−f plus extra domains, minus every image of half f.
* Bootstrap: 10,000 image-level resamples per benchmark, paired pick − greedy, percentile CI.
* Slide 8 macro bars 0.394 / 0.473 / 0.559 and slide 7 curves (2.03 → 5.12 distinct; 0.463 → 0.565 oracle) match their
  artifacts.
* Slide 6 architecture facts match PAPER_NOTES_GH (ViT 32 layers, width 1280, window attention; 2-layer MLP merger →
  3584; 28 decoder blocks).
