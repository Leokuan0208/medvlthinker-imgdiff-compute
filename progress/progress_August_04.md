# Progress — August 4, 2026 (the `(choice)(why)` programme — a clean, decisive negative)

> **Follows `progress_August_03.md`.** One idea, taken from hypothesis to verdict in six hours across
> three phases, with a gate between each. The idea: **multiple-choice best-of-N fails because a bare
> letter `"A"` gives a verifier nothing to grade.** So make the model answer as **`(choice)(why)`** — the
> option letter *followed by* a one-sentence justification — and the verifier finally has text to score.
> The programme was designed so that it could be killed early and cheaply: **Phase 1** asks only whether
> the format costs accuracy (it does not, +0.0087); **Phase 2** builds a strictly disjoint train pool and
> trains two verifiers that differ *only* in what their candidates look like; **Phase 3** measures
> selection efficiency and runs the ablation that decides the question. The mechanism was **real in the
> generations** (2.74 distinct justifications per competing letter against 1.59 for bare letters) and the
> measurement was **negative** (selection efficiency 0.7751 against 0.7977 letter-only). The decisive
> control settled *why*: **cutting the justification off the same candidates and scoring the bare letter
> changes nothing (−0.0024, n.s.).** Conclusion: **the multiple-choice selection deficit is an
> INFORMATION deficit, not a FORMAT deficit.** Every number below is sourced to a named
> `choicewhy_*` artifact; nothing is fabricated. **Abstention remains permanently out of scope** and
> appears nowhere.

> **Clock note, stated so the record is exact.** Every artifact, log and the commit for this session
> carry a **2026-08-03 UTC** stamp — generation began 12:57 UTC and the commit (`6cf6abd`) landed at
> 19:19:03 UTC. The session is filed here as the **August 4 working day**, which is how it is referred
> to in the research log; nothing below depends on the label, and **all times quoted are the UTC
> timestamps on disk.**

---

## 1. Where the idea came from, and why it was worth a day

Two measured walls set this up, and both were established the previous day (`progress_August_03.md` §5):

1. **On open text, best-of-N stops paying because selection efficiency *decays* with N** — measured at
   **−0.076 per doubling** (`verifier_n_scaling_2026-08-03.json`). There is no crossover; accuracy peaks
   at N ≈ 5 and declines.
2. **On multiple choice, best-of-N was never even in the game.** The reason had always been asserted
   rather than measured: the candidates are single letters, so there is nothing for the verifier to read.

Today's programme measured assertion 2 and then tested the obvious fix. The premise turned out to be
*more* true than assumed. Across the whole 11,904-candidate evaluation pool, the letter-only arm
produces **19 distinct candidate strings in total**, and
`frac_no_gradeable_justification = 1.0` (`choicewhy_justification_stats.json`). The verifier is being
asked to rank eight copies of `"A"`, `"B"`, `"C"`, `"D"`.

The design rule inherited from July 30 and applied from the first line of Phase 2: **a verifier trained
on any item whose image or question appears in the evaluation set is worthless** — contamination
inflated this project's previous verifier gain by **2.9×**
(`build_choicewhy_mcq_split.py` docstring, citing `verifier_validity_2026-07-29.json`).

---

## 2. Phase 1 — the gate: does the format cost accuracy? (12:57 – 13:53)

`src/labeling/run_choicewhy_pilot.py` → `ckpts/choicewhy_pilot/`; analysis by
`choicewhy_pilot_analyze.py` (+ `_supplement`, `_bestofn`, `_conditional`) →
`choicewhy_pilot_2026-08-03.json` (13:37).

**Design.** Lingshu-7B, greedy (T = 0), tp = 1, full resolution, `max_tokens = 320` **for every arm**, so
no arm is truncated where another is not. The user turn — images, question, option block — is
**byte-identical across arms and byte-identical to the repo's existing dumps** in
`ckpts/gate_lingshu7b_mcq/`, which makes arm A a **null test against a known cell**. Item set n = 1,488
from the repo's fixed `fixed_slice(seed=42)`: SLAKE-closed 416 (full), VQA-RAD-closed 272 (full),
PMC-VQA 500 (the *same* seed-42 subsample as the existing dump), MedXpert-MM 300 (150 Reasoning +
150 Understanding).

**Five arms, differing only in the system message:**

| arm | instruction |
|---|---|
| **A** `letter_only` | the deployed baseline, verbatim: *"Answer with only the correct option letter (e.g. 'A'). Do not explain."* |
| **B** `answer_first` | letter, then 1–2 short sentences |
| **C** `reason_first` | 1–2 sentences, then the letter last (conventional CoT — the ordering control) |
| **B2** `answer_first_forced` | letter first, then **exactly one sentence** stating the specific image finding; *"Always give the sentence, even when the answer is obvious"* |
| **C2** `reason_first_forced` | same forcing, reason first |

**The null test passed.** Arm A reproduces the deployed dump: prediction agreement SLAKE 0.9976,
VQA-RAD 1.0000, PMC-VQA 0.9900, MedXpert-R 0.9733, MedXpert-U 0.9867.

**Accuracy (`acc_strict`, exact option-letter match):**

| group | n | A | B | C | **B2** | C2 |
|---|---:|---:|---:|---:|---:|---:|
| SLAKE | 416 | 0.8486 | 0.8534 | 0.8389 | **0.8534** | 0.8245 |
| VQA-RAD | 272 | 0.7574 | 0.7574 | 0.7500 | **0.7684** | 0.7463 |
| PMC-VQA | 500 | 0.6040 | 0.6020 | 0.6060 | 0.5940 | 0.5740 |
| MedXpert-R | 150 | 0.2600 | 0.2600 | 0.3133 | 0.2933 | 0.2933 |
| MedXpert-U | 150 | 0.2600 | 0.3000 | 0.2667 | 0.3133 | 0.2933 |
| **POOLED-all** | **1,488** | **0.6310** | 0.6358 | 0.6337 | **0.6398** | 0.6190 |
| POOLED-perception | 1,188 | 0.7247 | 0.7256 | 0.7205 | 0.7247 | 0.7012 |

**The two decisive deltas** (paired bootstrap):

| delta | pool | value | 95% CI | p |
|---|---|---:|---|---:|
| **B2 − A** (the format cost) | all-1,488 | **+0.0087** | [−0.0040, +0.0222] | 0.2032 |
| **C2 − A** (reason-first, perception) | 1,188 | **−0.0236** | [−0.0396, −0.0076] | **0.005** |
| B2 − C2 (ordering control) | 1,188 | +0.0236 | [+0.0101, +0.0379] | 0.0002 |
| B2 − A | 1,188 | 0.0000 | [−0.0143, +0.0143] | 1.0 |

Read carefully: **+0.0087 is not a gain** — it is *not significant*, and on the perception pool it is
exactly zero. The claim the gate makes is the weaker and correct one: **the format is free.** And
**−0.0236 is specifically reason-first on the perception pool**, not the all-1,488 pool (there it is
−0.0121 [−0.0276, +0.0034], n.s.). That is Finding 1 reappearing inside a format experiment: putting the
reasoning before the answer costs perception accuracy, putting it after does not.

**Gate verdicts, verbatim:** `decision: "GO"`, `best_arm: "B2_answer_first_forced"`; criterion 1
*"PASS -- decisively; the format is free, slightly positive"* (worst per-benchmark delta −0.01,
`parse_ok_rate` 1.0); criterion 2 *"PASS with a caveat -- the rationales are gradeable text and create
real candidate diversity, but the cheap text-only proxy adds ~0 over the deployed letter-margin"*.

**The ordering control was inconclusive as a mechanism test, and the artifact says so.** Arm C **never
complied** (`letter_first_rate` 1.000, 0.0 justification words — the model ignored the instruction and
answered first anyway); C2 complied on ~10% (`letter_last_only_rate` 0.0941). On the self-selected
compliant subset C2 − A = +0.0143 [−0.0429, +0.0714], n = 140. Verdict verbatim: *"INCONCLUSIVE as a
mechanism test; INTENT-TO-TREAT result is consistent with Finding 1."*

**Cost of the format:** mean generated tokens 3.00 (A) → **19.43** (B2), a **6.48×** decode cost; arm B
was 24.28 tokens (8.10×). This is the price the rest of the programme has to earn back.

### 2.1 The mechanism *was* real in the generations

This is the part that justified spending Phases 2 and 3. Sampling N = 8 at T = 0.7 (seed 1234) on SLAKE
+ PMC-VQA (n = 916) and restricting to items where the eight samples **do not all agree on the letter**:

| on letter-disagreement items | **A** letter-only | **B2** `(choice)(why)` |
|---|---:|---:|
| n items | 393 | 422 |
| mean distinct letters | 2.4097 | 2.4597 |
| mean distinct candidate strings | 3.7684 | **6.4479** |
| **mean distinct strings per competing letter** | **1.5910** | **2.7445** |
| self-consistency@8 | 0.4606 | 0.4550 |
| oracle@8 | 0.8702 | **0.8957** |
| **headroom (oracle − SC)** | 0.4097 | **0.4408** |

The format does exactly what it was supposed to do: for each competing option letter the model now
produces **2.74 textually distinct justifications** instead of **1.59** near-copies, and it raises the
oracle ceiling (+0.026) and the headroom a selector could capture (+0.031). Over the whole eval pool the
same effect is stark: distinct candidate strings **19 → 8,172**; residual justification words after
removing the chosen option's own text, 0.0 → 11.56; share of candidates with no gradeable justification
**1.000 → 0.318**.

> **A provenance gap, recorded rather than smoothed.** The final `/gate` block (with `decision: "GO"`) and
> the `/best_of_n_probe/conditional_on_letter_disagreement` block in the pilot artifact are **not written
> by any committed script** — `choicewhy_pilot_analyze.py:347` writes a different, simpler `gate` dict, and
> `grep -r "competing_letter"` matches only the JSON. Both were produced by a later ad-hoc step that was
> never committed. **The numbers are real** — 1.5910 and 2.7445 were independently recomputed from the raw
> `ckpts/choicewhy_pilot/*_sc8.jsonl` dumps and match to the last digit — but **the recipe is undocumented
> in-repo.** This is the same class of defect as C25 (prompts not persisted): a load-bearing number whose
> generator cannot be re-run. It should be folded into `choicewhy_pilot_analyze.py` before the number is
> cited anywhere.

### 2.2 The eval pool, inventoried before anything was trained (13:53)

`choicewhy_eval_pool_inventory.py` → `choicewhy_eval_pool_inventory.json`. N = 8, T = 0.7, seed 1234:

| bench | n | A uniq strings / letters | B2 uniq strings / letters | A sc@8 / oracle@8 | B2 sc@8 / oracle@8 | letter-disagree A → B2 |
|---|---:|---|---|---|---|---|
| SLAKE | 416 | 1.204 / 1.204 | 5.846 / 1.243 | 0.834 / 0.926 | 0.827 / 0.926 | 0.202 → 0.240 |
| VQA-RAD | 272 | 1.415 / 1.415 | 6.669 / 1.445 | 0.761 / 0.934 | 0.754 / 0.941 | 0.415 → 0.445 |
| PMC-VQA | 500 | 1.942 / 1.938 | 5.934 / 2.030 | 0.564 / 0.816 | 0.530 / 0.826 | 0.618 → 0.644 |
| MedXpert-R | 150 | 2.373 / 2.347 | 6.473 / 2.593 | 0.227 / 0.533 | 0.267 / 0.600 | 0.767 → 0.820 |
| MedXpert-U | 150 | 2.233 / 2.207 | 6.913 / 2.500 | 0.287 / 0.540 | 0.287 / 0.593 | 0.747 → 0.807 |
| **POOLED** | **1,488** | **1.712 / 1.706** | **6.197 / 1.807** | 0.614 / **0.812** | 0.603 / **0.829** | 0.493 → **0.529** |

Token audit on the candidates: A mean 3.00 (max 8); B2 mean 21.65, median 20, p90 40, **max 134** —
nowhere near the 320 cap, so **neither arm is truncated**. Note the two numbers that matter for
interpreting Phase 3: B2's oracle@8 is **higher** (0.8286 vs 0.8118) and its letter-disagreement rate is
**higher** (0.5289 vs 0.4926). The B2 pool is *richer and harder to convert*.

---

## 3. Phase 2 — a strictly disjoint train pool and two matched verifiers (14:15 – 18:01)

### 3.1 The split, proved in code (14:15)

`src/training_methods/build_choicewhy_mcq_split.py` → `choicewhy_mcq_split.json`. The MCQ analogue of
July 30's `build_disjoint_verifier_split.py`, reusing its method verbatim: images are compared by an
**md5 of the decoded RGB pixels** (not file bytes), so a re-encoded copy of the same image is still
caught; question identity is the triple *(dataset family, normalized question text, image pixel hash)*.

**The disjointness set is deliberately over-strict:** the **full MedVLThinker-Eval suite — all 8,220
items, 5,507 images, 6,538 distinct question texts** — not merely the 1,488 pilot items (1,116 images).
Train side, from the datasets' **official train splits** (seed 0): slake_closed 1,681 q / 445 img;
vqa_rad_closed 940 / 280; pmc_vqa 3,000 / 2,971; pathvqa_closed 2,500 / 1,577 → **8,121 questions /
5,273 images** (5,201 at L2 strictness).

Asserted in code, all zero: `image_md5_rgb_intersection 0`, `image_md5_rgb_sized_intersection 0`,
`item_triple_intersection 0`, `pilot_eval_image_intersection 0`, `L2_question_text_intersection 0`.
601 L1 question texts are shared **by design** (same wording, different images).

**MedXpertQA-MM has no public train split**, so its 20% share of the eval mix cannot be matched
in-domain; its quota is topped up from pathvqa_closed_train and **the shortfall is recorded** — exactly
as `run_lora_verifier_disjoint.py` records the RadImageNet top-ups. Final quotas: pathvqa 2,630 /
pmc_vqa 4,382 / slake 1,969 / vqa_rad 1,383 = **10,364 per arm**, with slake (−928) and vqa_rad (−511)
shortfalls redistributed. `composition_matched_across_arms: true`.

### 3.2 Candidate generation and the two verifiers (14:38 – 18:01)

`run_choicewhy_trainpool.py` generated the N = 8 pools over that disjoint pool with identical
sampling and prompts, asserting each staged image's md5 (`logs/choicewhy_trainpool_{A,B2}.log`,
14:38 / 14:44). `build_choicewhy_verifier_examples.py` (14:46) turned them into per-arm training sets:
label = `int(extracted letter == gold letter)` under the repo's own MCQ grader, one example per unique
normalized candidate string per question.

**Three adapters, everything held fixed except what the candidates look like.** Base
Lingshu-7B (Qwen2.5-VL), bf16, flash-attention-2; LoRA **r = 16, α = 32, dropout 0.05, bias none**,
targets `q,k,v,o,gate,up,down_proj`; objective = next-token CE on the single Yes/No continuation token;
AdamW lr 1e-4, bs 2, accum 8, 1 epoch, **5,182 steps**, seed 0, max_pixels 1,003,520. Every
hyperparameter is asserted `"same": true` against the reference `ckpts/train/lora_verifier_disjoint`.
The prompt (`VERIF_SYS` + `verifier_body`) is **identical across arms — only `candidate` differs.**

| adapter | n_examples | questions | images | pos_rate | mean candidate words | train min |
|---|---:|---:|---:|---:|---:|---:|
| **A** letter-only | 10,364 | 7,273 | 4,709 | 0.5783 | **1.00** | 95.1 |
| **B2** `(choice)(why)` | 10,364 | 5,823 | 4,194 | 0.6578 | **17.64** | 95.3 |
| **B2_posmatched** (control) | 10,364 | 5,770 | 4,173 | 0.5783 | 17.42 | 91.5 |

`skipped_examples: 0`, `early_stopped: false` for all three. The **posmatched** arm exists because B2's
positive rate drifted up to 0.6578; it re-matches the base rate to A's 0.5783 so that any Phase-3
difference cannot be attributed to label balance.

### 3.3 Two integrity checks run before measuring (14:52 – 14:53)

**Judge concordance** (`choicewhy_judge_concordance.py`, n = 600 per cell, MedVLThinker-32B-RL_m23k,
T = 0) asked whether the exact-letter grader could be swapped for the project's free-text judge. It
found an **arm-specific bias that would have favoured arm A**:

| cell | agreement | judge-no / exact-yes |
|---|---:|---:|
| eval A, full answer | 0.9967 | 0 |
| eval A, option-text only | **1.0000** | 0 |
| **eval B2, full answer** | **0.9650** | **17** |
| eval B2, option-text only | 0.9967 | 0 |
| train B2, full answer | 0.9817 | 9 |

The appended rationale can talk the judge **out of** a correct letter (worst cells: MedXpert-U 0.9138,
MedXpert-R 0.9437). Decision: keep the exact-letter grader — identical for every arm — and **do not**
substitute the free-text judge on MCQ.

**Justification statistics** (`choicewhy_justification_stats.py`) fixed the definition of "gradeable":
strip the answer token, remove the chosen option's own words, and count fewer than 3 remaining words as
no justification. Eval pooled: A 0.0 residual words / **1.000** no-justification; B2 **11.56** /
**0.318**. Train pooled: B2 11.40 / 0.3539. (Greedy Phase 1 had measured 0.4187 for B2 — sampling
*improves* compliance.)

---

## 4. Phase 3 — the measurement, and the ablation that decided it (18:25 – 19:15)

`choicewhy_score_candidates.py` dumped verifier P(Yes) per candidate for four scoring passes
(`cw_score_A` 18:25, `cw_score_B2` 18:49, `cw_score_B2lp` 19:08, `cw_score_B2pm` 19:09); then
`choicewhy_measure.py` → `choicewhy_measure_2026-08-03.json` (19:15).

**The target, defined precisely.** *Selection efficiency at fixed N = 8* =
**P(pick a correct candidate | a correct candidate is present)** = `mean(selector@8) / mean(oracle@8)`,
where the selector is argmax verifier-P(Yes) over the N candidates. Method copied verbatim from
`verifier_n_scaling.py` so the numbers are comparable to the open-text ones: oracle@N, verifier@N and
self-consistency@N are **exact expectations over all C(8,N) subsets** (255 per question, enumerated — no
Monte-Carlo); ties are resolved as the uniform random tie-break they actually are; CIs are a
**question-level non-parametric bootstrap, 4,000 resamples, paired** (the same resampled question
indices for every arm, so arms share sampling noise).

**Two null tests, both PASS:** the open-text harness reproduces the published clean cell to
`max_abs_diff` **3.4e−05** (tolerance 1e−4, n = 2,345), and the MCQ sc8 pool matches the Phase-2
inventory to **3.33e−05**.

### 4.1 The headline (n = 1,488)

| quantity | letter-only (verifier A) | **`(choice)(why)` (verifier B2)** | delta [95% CI] | sig |
|---|---:|---:|---|---|
| **selection efficiency @8** | **0.7977** [0.7742, 0.8203] | **0.7751** [0.7512, 0.7979] | **−0.0226 [−0.0433, −0.0024]** | **yes** |
| accuracy @8 | 0.6476 [0.6232, 0.6726] | 0.6423 [0.6177, 0.6668] | −0.0053 [−0.0216, +0.0108] | no |
| oracle @8 | 0.8118 | **0.8286** | +0.0168 [+0.0054, +0.0282] | yes |
| self-consistency @8 | 0.6159 (eff 0.7587) | 0.6062 (eff 0.7316) | — | — |
| confident-distractor rate @8 | 0.2023 | 0.2249 | *(open-text reference 0.2267)* | — |

**Decline slope per doubling of N** (least squares of selection efficiency on log₂N, N = 1…8):
letter-only **−0.0665** [−0.0737, −0.0592]; `(choice)(why)` **−0.0738** [−0.0812, −0.0662]; the
open-text reference is **−0.0761** [−0.0832, −0.0687]. Composition-matched on the perception pool
(n = 1,188): −0.0514 and −0.0578. **The multiple-choice decay is the same phenomenon as the open-text
decay, at the same rate**, and the richer format makes it very slightly worse, not better.

Context numbers from the same artifact: always-32B scores **0.6821**; the B2 verifier arm is
**−0.0399 [−0.0633, −0.0172]** against it; against the *deployed greedy* baseline the end-to-end arm is
+0.0112 [−0.0080, +0.0302], n.s. Cost at N = 8: **9.0808** 7B-forward-equivalents = **2.3772×** a single
always-32B forward (at the grounded R32 = 3.82 from August 3). The arm is 2.4× the price of the thing it
fails to beat.

### 4.2 The decisive ablation

Everything above is confounded: the B2 pool is *different* from the A pool. The ablation removes that
confound entirely. Take the **same B2 candidate pool**, truncate every candidate to its leading
`<letter><delimiter>` (regex `^(\s*[*"'(\[]*\s*[A-J]\s*[).:,;\-—\]]?)`, implemented as
`choicewhy_score_candidates.py --candidate_mode letter_prefix`), and score it with the **letter-only
verifier A**. Same items, same pool, same verifier. The only thing that changes is **whether the
verifier can read the justification.**

| n = 1,488 | accuracy @8 | selection efficiency @8 |
|---|---:|---:|
| verifier B2 — justification **visible** | 0.6423 | 0.7751 |
| verifier B2 — justification **cut off** | 0.6447 | 0.7780 |
| **difference (same pool)** | **−0.0024 [−0.0174, +0.0124]** | −0.0029 |

**Showing the verifier the entire justification is worth −0.0024, indistinguishable from zero.**
Corroborating: within-question ranking AUROC is **0.6375** (A) against **0.6433** (B2), a difference of
**+0.0058**; the lettercut variant scores 0.6412. The base-rate control is *worse*, not better
(posmatched selection efficiency 0.7682; `B2_posmatch − A` = −0.0295 [−0.0513, −0.0078]).

### 4.3 What the −0.0226 actually is — a pool effect, not a selector effect

This is the interpretation the artifact insists on, and the diary should not overstate it. The verifier
is **indifferent** to the justification (ablation −0.0024 n.s.; AUROC +0.0058). What moved is the
**denominator**: B2's oracle@8 is higher (0.8286 vs 0.8118) and its letter-disagreement rate is higher
(0.5289 vs 0.4926), which mechanically depresses `sel_eff = accuracy / oracle`. So `(choice)(why)`
makes the **pool harder to convert** while leaving the verifier's discrimination essentially unchanged.
It did not make the verifier worse; it made the ceiling higher and the verifier could not reach any more
of it.

### 4.4 The verdict, verbatim

> `answer`: *"NO. (choice)(why) does not give the verifier enough signal to improve selection. … a
> change of -0.0226 (95% CI [-0.0433, -0.0024]), i.e. significantly WORSE, not better. … The format did
> not fix the mechanism."*

> `the_decisive_control`: *"Cutting the justification off the SAME (choice)(why) candidates and scoring
> the bare letter with the letter-only verifier gives accuracy 0.6447 / sel_eff 0.7780, versus 0.6423 /
> 0.7751 when the verifier is shown the whole justification (difference -0.0024, CI [-0.0174, 0.0124]).
> Same pool, same items, the only difference is whether the verifier can read the justification -- and
> it is worth nothing."*

> `what_it_implies_about_the_selection_limit`: *"The deficit is not a text-availability problem. …
> Best-of-N on MCQ was never degenerate for lack of TEXT; it is limited because deciding which of two
> option letters is right IS the task, and a 7B verifier is no better at it than the 7B generator.
> Giving the same model more of its own words to read does not add information. The remaining lever is a
> selector with information the generator does not have, not a richer answer format."*

Committed as `6cf6abd` at 19:19:03: *"(choice)(why) does not rescue MCQ selection: the deficit is
information, not format."* 24 files, 17,015 insertions — 8 artifacts and 16 scripts.

---

## 5. What was learned

**The negative is clean, and that is its value.** Most negatives in this project were killed by a
confound, a cost model or a weighting. This one was killed by its own decisive control, on the same
pool, with the same verifier, in a design where the alternative hypothesis had every advantage: the
format was free on accuracy, the mechanism demonstrably fired in the generations (1.59 → 2.74 distinct
justifications per competing letter), the oracle ceiling genuinely rose (+0.017), the verifier was
retrained from scratch on a **provably disjoint** pool, and a base-rate control was run. It still
produced nothing.

**It converts a stated limit into a measured one.** The project has said for six weeks that the
selection wall is "intrinsic". Until today the multiple-choice half of that was an *assertion about
format*. It is now an *information* claim with an ablation behind it: a 7B verifier reading a 7B
generator's own justification has no more information than the generator had, because **on multiple
choice, deciding between two letters *is* the task.** The corollary is directional and is exactly what
the next experiment tests: **the remaining lever is a selector holding information the generator does
not** — a different model family, not a richer answer format.

**It also re-confirms Finding 1 from inside a format experiment.** Reason-first costs −0.0236
[−0.0396, −0.0076] on perception; answer-first costs nothing. Placing the reasoning *after* the answer
is free; placing it *before* is not.

---

## 6. Standing state and open questions

**Files produced today.** Artifacts: `choicewhy_pilot_2026-08-03.json`,
`choicewhy_eval_pool_inventory.json`, `choicewhy_mcq_split.json`, `choicewhy_verifier_examples.json`,
`choicewhy_judge_concordance.json`, `choicewhy_justification_stats.json`,
`choicewhy_build_2026-08-03.json`, `choicewhy_measure_2026-08-03.json`. Scripts: `choicewhy_common.py`,
`choicewhy_pilot_analyze.py`, `_supplement.py`, `_bestofn.py`, `_conditional.py`,
`choicewhy_eval_pool_inventory.py`, `choicewhy_justification_stats.py`, `choicewhy_judge_concordance.py`,
`choicewhy_build_artifact.py`, `choicewhy_score_candidates.py`, `choicewhy_measure.py`;
`src/labeling/run_choicewhy_{pilot,trainpool}.py`;
`src/training_methods/{build_choicewhy_mcq_split,build_choicewhy_verifier_examples,run_lora_verifier_choicewhy}.py`.
Checkpoints (gitignored): `ckpts/choicewhy_pilot/`, `ckpts/choicewhy_train/`,
`ckpts/choicewhy_judge_audit/`, `ckpts/train/lora_verifier_choicewhy_{A,B2,B2_posmatched}/`.

**Open questions:**

1. **The undocumented gate block (§2.1).** Two load-bearing blocks in the pilot artifact have no
   committed generator. Fold them into `choicewhy_pilot_analyze.py` so the 1.5910 / 2.7445 pair can be
   re-derived by running a script rather than by re-reading raw checkpoints.
2. **`POOLED-competent4` is mislabelled.** In `choicewhy_measure_2026-08-03.json` it has
   `n_items = 1188` and pools **three** benchmarks (SLAKE + VQA-RAD + PMC-VQA); PathVQA is not in this
   eval slice at all. The label should be corrected before the number is quoted.
3. **MedXpert's training quota is out-of-domain by construction.** It has no public train split, so 20%
   of the eval mix was topped up from PathVQA. The recorded shortfall means the verifier is weakest
   exactly where letter disagreement is highest (0.82 on MedXpert-R). Whether an in-domain MedXpert
   verifier would change the verdict is untested — but the ablation makes it unlikely, since the
   ablation is arm-internal.
4. **The programme cost 2.38× a single 32B forward to lose.** Any successor must clear that bar before
   accuracy is even discussed.
5. **The follow-up is already running, and has produced nothing yet.** At **05:57–06:00 UTC on
   2026-08-04** an uncommitted **cross-family zero-shot verifier sweep** was staged and launched —
   `src/cascade_methods/crossfamily_verifier_gpu.py` + `runners/run_crossfamily_verifier_sweep.sh` —
   testing today's corollary directly: score the **fixed** Lingshu-7B sc8 pools with verifiers from
   *other* families (Qwen2.5-VL-7B, MedVLThinker-7B, InternVL3-8B, HuatuoGPT-Vision-7B, MedGemma-4B,
   and Lingshu-32B as a same-family scale reference), holding the pool, the labels, the grader prompt,
   the image budget and the scoring rule fixed so the **only** variable is the verifier's weights. Its
   own docstring cites today's result as the motivation: *"the selection deficit has been shown to be an
   INFORMATION deficit, not a format deficit."* As of this entry the sweep is in pass 1
   (`logs/crossfam/`, three datasets in flight) and **no artifact exists** — **no result is claimed
   here.**
