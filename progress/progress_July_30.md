# Progress — July 30, 2026 (the correction pass, day 2 — the headline breaks)

> **Continues `progress_July_29.md` without a break** — the PMC-VQA audit worksheet was built at 23:35
> last night and the classification pass finished at 00:00 this morning. Yesterday's five audits each
> raised a doubt; today the doubts were **priced**, and the price was the headline. Six pieces of work:
> (1) the **PMC-VQA item-level validity audit** — 53% of the decision-relevant wins are defective, but
> **symmetrically with the losses**, so the arithmetic survives and the construct does not; (2) the
> **PMC-VQA provenance investigation**, which found the dataset is machine-written from captions with no
> image ever looked at, retracted a false claim this project made yesterday, and uncovered that **two
> different PMC-VQA splits have been used in two eras with nobody recording it**; (3) the **MedEvalKit
> matched direct-arm re-run** — 0/9 reasoning-trigger effects significant, 3/9 answer-format effects
> significant; (4) the **macro re-weighting**, which reverses the compute claim; (5) the **disjoint
> verifier retrain** — 2.9× inflation; and (6) the **three-way headline** combining macro with the clean
> verifier, which is where *"the method beats a single 32B forward"* died. The day closed with a
> 3,266-line comprehensive write-up. **Two process failures cost roughly nine hours and one wasted model
> load and are recorded in §8.** Every number is sourced to a named artifact; nothing is fabricated.
> **Abstention remains permanently out of scope** and appears nowhere.

---

## 1. The PMC-VQA item-level validity audit (`pmc_label_noise_audit_2026-07-29.json`, 00:00–00:04)

**Why.** PMC-VQA is **79.17%** of the MedEvalKit evaluation pool and the fusion cell there supplies the
bulk of the accuracy-max win. The obvious attack on that win — *"you are just exploiting annotation
noise"* — had never been tested. Scripts: `src/cascade_methods/pmc_audit_classifications.py`, then
`pmc_label_noise_audit.py --stage extract` and `--stage score`.

**Method.** 200 `test_2.csv` items opened **image by image**, classified against a fixed rubric with the
question, four options, gold letter, both models' raw answers and the PMC-VQA source caption in view.
Rubric precedence `BAD-GOLD > UNANSWERABLE > MULTI-CORRECT > GENUINE`; `UNCLEAR` never counted as a
defect. **Three strata — and the third is the whole experiment**: wins (n = 100, items where fusion
beats always-32B), losses (n = 50), and a **control** of 50 items where both models agree and are
correct. Without the control, a defect rate is uninterpretable. Auditor recorded verbatim: *"Claude
Opus 5, image by image … Not a radiologist: label QUALITY was judged, not diagnoses."*

Base pool: n = 33,430; 7B 0.5427; 32B-direct 0.5518; fusion 0.5653 (**Δ +0.0135 [+0.0100, +0.0169]**);
W = 1,969 wins, L = 1,518 losses; disagreement rate 0.3297.

| class | wins (n=100) | losses (n=50) | control (n=50) |
|---|---:|---:|---:|
| GENUINE | 45 | 20 | 36 |
| BAD-GOLD | 9 | 5 | 1 |
| UNANSWERABLE | 37 | 22 | 11 |
| MULTI-CORRECT | 7 | 3 | 2 |
| UNCLEAR | 2 | 0 | 0 |
| **defective** | **53% [0.4329, 0.6249]** | **60% [0.4618, 0.7239]** | **28% [0.1747, 0.4167]** |

**The three bias tests, and the decisive one:**

| test | difference | z | p (z) | **Fisher p** | significant |
|---|---:|---:|---:|---:|---|
| wins vs control | +0.25 | 2.903 | 0.00369 | 0.0051 | yes |
| **wins vs losses** | **−0.07** | −0.813 | 0.41626 | **0.48701** | **no** |
| losses vs control | +0.32 | 3.223 | 0.00127 | 0.00233 | yes |

**The attack fails, and fails for a measurable reason.** Decision-relevant disagreements are far more
defective than the agreement control (+0.25, +0.32, both significant) — but defects are **not biased
toward the wins**; the point difference actually *favours the losses*. Mis-keying in particular is
symmetric: **BAD-GOLD 9% of wins against 10% of losses.**

**Four correction models, and the one the brief asked for is the wrong one:**

| model | formula | fusion Δ | 95% interval |
|---|---|---:|---|
| **A** — discount the wins only (as briefed) | (1 − f_wins) × Δ | +0.0063 | [+0.0027, +0.0100] |
| **B** — symmetric drop-defective (**correct**) | (W(1−f_w) − L(1−f_l)) / n | **+0.0094** | **[+0.0004, +0.0183]** |
| **C** — re-key BAD-GOLD only | (W − L − 2W·f_bg,w + 2L·f_bg,l) / n | +0.0124 | [+0.0018, +0.0236] |
| **D** — cleaned benchmark (denominator shrinks) | B / (n(1−d)) | +0.0136 | [+0.0006, +0.0265] |

Because the losses are *at least as* defective as the wins, discounting only the wins **overstates** the
damage. Under the correct symmetric model **69.6%** of the measured delta survives — and the interval
now nearly touches zero. Propagated (PMC weight 0.7917): veto-vs-reasoning +0.0245 → +0.0222;
fusion-vs-reasoning +0.0271 → +0.0238; veto-vs-direct +0.0106 → +0.0083. **compute-lean is untouched**
(+0.0150 → +0.0152) because its PMC cell is −0.0010 — *compute-lean never depended on a PMC win.*

**But the construct is gone. 46% of the wins sit on items where the gold is wrong or the answer is
simply not in the shown image.** Four audited examples, verbatim from `per_item`:

- **`pmc-13058`** — the question asks about a blue arrow on a head CT; the image is a **spleen ultrasound**.
- **`pmc-24120`** — the question asks about the femur; the image is a **chest CT**.
- **`pmc-24810`** — the question asks about blue labelling in photomicrographs; the image is a **photo of a cat's face**.
- **`pmc-25510`** — the 32B **correctly described the panel that was actually shown**, and was scored wrong.

On those items the score is decided by which model's *language prior* better matches a caption-derived
key, not by which model reads the image better.

**The noise ceiling.** Stratum masses: agree-correct 0.4296, agree-wrong 0.2407, fusion wins 0.0589,
fusion losses 0.0454, non-decisive disagreements 0.2254. Pool defect rate **0.3093** (generous) to
**0.3709** (stratified) → achievable accuracy bounded at roughly **0.63–0.77** if defective items are
scored wrong, 0.72–0.77 at 4-way chance. Every system in this project scores **0.5427–0.5653**, so the
benchmark is *not* saturated — but a ~1-point margin is being measured where **31–37% of items cannot
support a correctness claim at all**.

**Verdict headline, verbatim:** *"PARTLY REAL, but the arithmetic survives while the CONSTRUCT does
not."* Recommended phrasing going forward: report the PMC number as **"higher agreement with PMC-VQA's
caption-derived answer keys"**, with the 53% defect rate stated alongside, and **stop using PMC-VQA to
carry an accuracy claim.** The audit explicitly *"does NOT bear on the open-text cells, on the
compute/latency claims"*.

---

## 2. The PMC-VQA provenance investigation (`PMCVQA_PROVENANCE_2026-07-30.md`, 00:28; committed 09:53)

The audit found *what* was wrong; this document found *why*, and turned up a second problem nobody was
looking for. Read-only, no GPU, 540 lines, every claim source-cited, re-measurements marked
**[measured 2026-07-30]**, unsourced claims marked **UNVERIFIED**.

### 2.1 The dataset is machine-written from captions, and no image was ever looked at

PMC-VQA is generated from ~1.6M PMC-OA figure/caption pairs; QA generation drew on **381K**
image-caption pairs, and **only the caption text** was fed to ChatGPT (MedVLThinker names GPT-3.5). The
generation prompt, verbatim: *"Ask 5 questions about the content and generate four options for each
question. The questions should be answerable with the information provided in the caption…"*

Filtering was entirely automatic: formatting → 1,497,808; a LLaMA-7B **text-only answerability** filter
(shuffle choices, infer 5×, dismiss if right ≥ 3/5) → 848,433; an image-dependency classifier trained
on **2,192 manual binary labels** at **81.77%** accuracy → final **226,946** QA over 149,075 images.
*(Measured discrepancy: the released v1 CSVs hold 176,948 + 50,000 = **226,948**, two rows more than the
paper states.)*

**The design consequence is exactly §1's defect taxonomy.** Nothing in the construction pipeline checks
image↔key consistency, so when multi-panel figures are cut, a caption-derived question travels to the
wrong panel — which is the cat's face, the spleen ultrasound and the chest CT.

**Validation, in full, is one sentence.** 50,000 items → "PMC-VQA-test-initial", then *"we manually
checked some test samples again, resulting in a small clean test set of 2000 samples"*, with the estimate
*"over 80% of cases in PMC-VQA-test can be retained"*. **The number of items inspected to yield the
2,000 is never stated**, so the 80% cannot be converted to a defect rate. Never reported in any version:
who verified, whether a clinician was involved (GitHub issue **#17**, open and unanswered), any
inter-annotator agreement, any human ceiling, or any verification of `test_2.csv` (issue **#11**, also
open). Across 15 issues, **none reports label errors**.

**The naming trap.** In the paper, "PMC-VQA-test" **is** the 2,000 verified set; the released files
**invert** it — `test.csv` is the 50,000 unverified pool and `test_clean.csv` is the 2,000 verified one.
**Rule adopted: never write "PMC-VQA test" unqualified; always write the file name and the row count.**

### 2.2 Two splits, two eras, nobody wrote it down

This is the finding, stated verbatim in the document: *"**We used both — in two eras — and nobody wrote
it down. That is the actual root cause of this investigation.**"*

| track | split | n | share of its own pool | verified |
|---|---|---:|---:|---|
| CVGIP cascade / margin gate (MedVLThinker-Eval, 8,220) | **`test_clean.csv`** | 2,000 | **24.3%** | **yes** |
| Lingshu-faithful (MedEvalKit, 44,694) | **`test_2.csv`** | 33,430 | 74.8% (78.3% excl. MedXpertQA) | **no** |

Evidence, both measured: `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39` hard-codes
`csv_path = .../test_2.csv` with no env var and no argument, and MedEvalKit is at upstream HEAD
`9b12e3b` with an empty diff on that file — **this was upstream's choice, not ours**. On the other side,
MedVLThinker-Eval's `pmc_vqa` slice matches `test_clean.csv` **2,000/2,000** on question, `answer_label`
and answer text, and `ckpts/gate_7b_prune/cap320/ckpt_PMC-VQA_nothink_norag.jsonl` golds match
**2,000/2,000**. Measured split relations: `test_clean ⊂ test.csv` 2,000/2,000; **`test_clean ∩ test_2`
= 6 items of 2,000.**

**Consequence for the cascade track: it was right, and better than we knew.** The briefing's *"the
headline leans on PMC-VQA at 79% of the pool"* is true **only of the MedEvalKit pool**. Reporting over
the 8,220-item MedVLThinker-Eval pool puts PMC at **24.3%** — a 3.3× reduction in exposure, achievable
with **existing dumps and no new inference**.

### 2.3 The retraction — this project falsifying its own document from yesterday

`PROJECT_RETROSPECTIVE_2026-07-29.md` hole 10 and row **X14** asserted, written yesterday, that
`test_clean.csv` *"is not on disk"* and *"has never been used anywhere in the repo"*. **Both are wrong**
[measured 2026-07-30]: the file exists in **two byte-identical copies** —
`/data/dan/dataset/medevalkit/PMC-VQA/test_clean.csv` (418,686 bytes, 2,000 data rows, mtime
2026-06-29 07:18) and `/data/dan/dataset/pmc_vqa_train/test_clean.csv` (md5
`6abfbcd088171c76a98911c5e7a8f5a0`) — and **the cascade track has been evaluating exactly those 2,000
items all along.** The retrospective's prescribed fix is therefore already satisfied for the cascade
track and outstanding only for MedEvalKit. *(Same statement, second error: the hard-code is at
`PMC_VQA.py:39`, not `:41`.)* **Root cause: the 2026-07-29 pass inferred the directory contents instead
of listing them.** Entered as **X20**, a correction to the corrections log itself.

### 2.4 The cross-split comparison, flagged rather than exploited

| source | split | rubric | rate |
|---|---|---|---:|
| authors (PMC11663219) | v1 test pool | criteria (i)–(iii), *estimated*, denominator unstated | ≲ 20% |
| **this audit** | **`test_2`** (never verified) | BAD-GOLD ∪ UNANSWERABLE ∪ MULTI-CORRECT | 28% control · 53% wins · 60% losses |
| ECCV 2026 precedent (arXiv 2607.00159) | E-VQA / InfoSeek | unsupported / ambiguous | 22% unsupported; 59% / 47% ambiguous |

*"These are not in conflict, and it is important not to present them as if they were."* Three things
differ at once: **split** (the authors' 80% is v1; only **8 of our 200** audited items are also in
test_clean), **rubric** (ours is stricter), and **conditioning** (53%/60% are conditioned on model
disagreement, which selects the hard items). The one comparable cell is our **28% control against their
≲20% estimate** — same direction, modestly worse, on a worse split.

A **second** cross-split comparison is flagged and explicitly *not* banked: the same models score 6–9
points higher on the clean split (Lingshu-7B **0.604** vs 0.5427; Lingshu-32B **0.640** vs 0.5518),
*"consistent with better keys, but confounded by the fact that v1 and v2 are different item populations,
not the same items re-keyed."* Labelled **suggestive, not probative**.

### 2.5 One line of inquiry closed permanently, and one power calculation

**Route A — filter the 33,430-row dumps down to the verified subset — is DEAD:** only **6 items** survive
the intersection. *"Close this line of inquiry permanently."* **Route B** — clean-split dumps that
already exist — is available at zero GPU cost, but the power problem is decisive: the published test_2
delta is +0.0135 with a half-width of 0.00345 at n = 33,430; 1/√n scaling gives half-width **0.0141 at
n = 2,000** and 0.0282 at n = 500, with the significance threshold at n = 2,183. **Nothing certifies on
the clean 500** — the headline policy `F3_confadv` scores 0.638 with **d = −0.002 [−0.032, +0.028]**;
*"the sign is not even stable."* Recorded as *"not a refutation"*, but *"the clean split currently
provides no positive evidence for the PMC win."* Any replication must be **pre-registered** with a null
as the expected outcome.

---

## 3. The MedEvalKit matched direct-arm re-run (`medeval_matched_direct_2026-07-29.json`, 11:25)

Yesterday's ranked re-run **#3**, and today's substantive GPU work. Runner
`runners/run_medeval_direct_matched.sh`; patch `src/labeling/medeval_matched_prompt.py`.

**The design decision that mattered.** MedEvalKit is a protected dependency, so instead of editing it
the patch is an **environment-gated monkeypatch** (`MEDEVAL_MATCHED_PROMPT=1`) over
`utils.question_formats.get_multiple_choice_prompt` and `utils.MMMU.data_utils.construct_prompt` —
**MedEvalKit is left byte-identical to upstream.**

**Three arms, so that format and trigger can be separated:**

| arm | prompt | measures |
|---|---|---|
| `reason` | reasoning trigger **+** `\boxed{}` | the published arm |
| `direct_matched` | `\boxed{}`, **no trigger** | — |
| `direct_unmatched` | upstream's *"Answer with the option's letter … directly."* | the published direct arm |

→ **`delta_matched` = the trigger effect** (reason − direct_matched); **`delta_format` = the format
effect** (direct_matched − direct_unmatched); `delta_unmatched` = the published total. 10,000-replicate
bootstrap, rng 12345, exact two-sided McNemar. Coverage **6/6 cells**, `missing []`.

**The nine primary sub-cells** (3 families × MMMU-MCQonly / MedXpert-Reasoning / MedXpert-Understanding):

| family | cell | n | **trigger Δ [CI]** | **format Δ [CI]** | published Δ |
|---|---|---:|---|---|---|
| Lingshu-32B | MMMU-MCQonly | 145 | +0.0414 [−0.0345, +0.1172] | −0.0138 [−0.0483, +0.0207] | +0.0276 n.s. |
| Lingshu-32B | MedXpert-R | 1,446 | +0.0041 [−0.0207, +0.0290] | −0.0076 [−0.0180, +0.0021] | −0.0035 n.s. |
| Lingshu-32B | MedXpert-U | 554 | +0.0018 [−0.0379, +0.0415] | −0.0018 [−0.0144, +0.0108] | 0.0000 n.s. |
| MedVLThinker-32B | MMMU-MCQonly | 145 | +0.0414 [−0.0138, +0.0966] | +0.0621 [−0.0071, +0.1310] | +0.1034 **sig** |
| MedVLThinker-32B | MedXpert-R | 1,446 | +0.0007 [−0.0221, +0.0228] | **+0.0456 [+0.0194, +0.0719]** | +0.0463 **sig** |
| MedVLThinker-32B | MedXpert-U | 554 | **−0.0018** [−0.0361, +0.0325] | **+0.0433 [+0.0054, +0.0830]** | +0.0415 **sig** |
| InternVL3-38B | MMMU-MCQonly | 145 | +0.0345 [−0.0138, +0.0897] | **+0.0897 [+0.0207, +0.1586]** | +0.1241 **sig** |
| InternVL3-38B | MedXpert-R | 1,446 | +0.0131 [−0.0090, +0.0353] | +0.0221 [0.0000, +0.0436] | +0.0353 **sig** |
| InternVL3-38B | MedXpert-U | 554 | +0.0108 [−0.0217, +0.0451] | +0.0090 [−0.0271, +0.0451] | +0.0199 n.s. |

**0 of 9 trigger effects are CI-significant** (8/9 point-positive, mean shift from matching **−0.0276**);
**3 of 9 format effects are** (MedVLThinker MX-R and MX-U, InternVL3 MMMU).

**The mechanism, from the token audit.** Mean generated tokens in the **direct_matched** arm — the arm
with no reasoning instruction in it at all:

- **Lingshu-32B: 3.0–4.4 tokens, `reasoned_frac` 0.0** → *"clean: reasoning arm reasoned, matched direct arm answered directly."*
- **MedVLThinker-32B: 417–580 tokens, `reasoned_frac` 0.9667–0.9979** → **CONTAMINATED**.
- **InternVL3-38B: 193–289 tokens, `reasoned_frac` 0.920–0.952** → **CONTAMINATED**.

**Asking for the answer in `\boxed{}` is itself a reasoning trigger.** The contaminated cells' own
`arm_validity` string says so: *"the \boxed{} instruction alone induces reasoning in this model.
delta_matched therefore measures the MARGINAL value of the explicit reasoning trigger ON TOP of
format-induced reasoning, NOT reasoning-vs-no-reasoning."*

**Verdict headline, verbatim:** *"Under a MATCHED prompt the explicit reasoning trigger is worth
~nothing: 0/9 primary cells are CI-significantly positive (0 negative). The published gains were carried
by the ANSWER FORMAT: requesting \boxed{} alone makes the reasoning-tuned families (MedVLThinker-32B,
InternVL3-38B) emit 280-580 reasoning tokens with no trigger present … Lingshu-32B never reasons without
the trigger and gains nothing when it does."*

**What is kept, and what is dropped.** DROP *"a reasoning instruction improves accuracy on
reasoning-heavy benchmarks"* (C27). KEEP the weaker supported form — *getting a reasoning-tuned model to
emit a trace* helps substantially (MedVLThinker MMMU +0.103, MX-R +0.046; InternVL3 MMMU +0.124) — **with
the answer format named as the operative lever.** Lingshu-32B must not be cited as reasoning evidence at
all, which *strengthens* yesterday's C22. **The cascade's gated-reasoning tier keeps its full value: the
rung1→rung3 total is what a think tier delivers; only the attribution changes.** The honest substitute
for the unobtainable clean contrast is the **monotone ladder**: MedVLThinker MMMU-MCQonly **0.634 @ 2
tokens → 0.697 @ 431 → 0.738 @ 580**.

**Standing rule adopted:** *"Any future think-vs-direct arm pair must be format-matched AND
token-audited: a 'direct' arm that emits hundreds of tokens is not a direct arm."*

> **Two integrity notes recorded with the artifact.** (a) `mmmu_open_item_audit`: all 9 (family × arm)
> combinations score **0/5** on the five format-unmatched MMMU "open" items, so MMMU ≡ MMMU-MCQonly in
> the numerator. (b) A known unmatched axis: `EVAL_BATCH_SIZE` **250 vs 2000** for the reason arm (OOM
> safety at tp = 2), affecting MedXpert only under greedy temperature-0 decoding.

> **⚠ An internal inconsistency in the artifact, flagged rather than silently fixed.** `verdict.headline`
> and `verdict.primary_cells` say **9**, and the per-family blocks sum to 9 — but the first
> `what_the_project_should_claim` bullet says *"0/7 cells CI-significant"*. **The 9 is correct**; the
> "0/7" is a stale denominator in that one string.

---

## 4. The macro re-weighting, and the compute-claim reversal (`macro_average_headline_2026-07-30.json`)

Yesterday's §3.3 observation, executed as a decision.
`src/cascade_methods/macro_average_headline.py`, 10,000 replicates, seed 20260730, Variant B
(n = 42,224, 8 cells / 5 benchmarks). *(The artifact was committed at 09:53 in `a5b7f35` and re-emitted
unchanged on 2026-08-03 02:56.)*

**The statement of the change, verbatim:** *"Macro-averaging moves PMC-VQA from 79.2% of the weight to
12.5%, the open-text arm from 5.6% of the items to 37.5% of the weight, and the closed/multiple-choice
arm from 94.4% to 62.5%. Every number below is a consequence of exactly that."*

**Accuracy levels move a lot, and in the same direction for everything:**

| system | sample-weighted | **macro (8 cells)** | shift |
|---|---:|---:|---:|
| always-7B | 0.5549 | 0.5971 | +0.0422 |
| always-32B-direct | 0.5729 | **0.6567** | +0.0838 |
| always-32B-reasoning | 0.5591 | 0.5974 | +0.0383 |
| oracle-mode-32B | 0.5730 | 0.6573 | +0.0843 |
| compute-lean | 0.5741 | 0.6600 | +0.0859 |
| accuracy-max-veto | 0.5836 | **0.6694** | +0.0858 |
| accuracy-max-fusion | 0.5862 | 0.6661 | +0.0799 |

**Verdicts that flip:**

| comparison | pool | sample-weighted | **macro** | change |
|---|---|---|---|---|
| **compute-lean vs 32B-direct** | MCQ (5) | −0.0015 TIE | **−0.0070 [−0.0126, −0.0017] LOSS** | **TIE → LOSS** |
| compute-lean vs oracle-mode | MCQ | −0.0016 TIE | **−0.0080 [−0.0137, −0.0024] LOSS** | TIE → LOSS |
| compute-lean vs 32B-direct | open | +0.0456 WIN | +0.0206 [−0.0009, +0.0423] TIE | WIN → TIE |
| veto vs 32B-reasoning | MCQ | +0.0101 WIN | +0.0043 [−0.0016, +0.0105] TIE | WIN → TIE |
| veto vs oracle-mode | MCQ | +0.0079 WIN | +0.0010 [−0.0006, +0.0025] TIE | WIN → TIE |
| veto vs 32B-reasoning | all 8 | +0.0245 WIN | **+0.0720 [+0.0614, +0.0824] WIN** | unchanged (≈3×) |
| veto vs 32B-direct | all 8 | +0.0107 WIN | +0.0128 [+0.0056, +0.0200] WIN | unchanged |

**And the compute claim reverses outright.** Escalation is wildly heterogeneous across cells — PMC-VQA
**8.45%**, SLAKE-closed 20.45%, VQA-RAD-closed 56.97%, PathVQA-closed 45.72%, MedXpert **89.60%** — and
the lowest-escalation cell carried 79% of the pooled average. Multiple-choice escalation goes
**16.22% → 44.24%**; all-8 escalation **16.89% → 35.65%**. FLOP-eq against always-32B-direct:

| operating point | sample-weighted | **macro** |
|---|---:|---:|
| compute-lean | 0.492× | **1.196×** |
| accuracy-max-veto | 0.932× | **1.410×** |
| accuracy-max-fusion | 1.250× | **1.435×** |

`cost.joint_claim.headline`, verbatim: *"Under equal weight per cell NO operating point is
compute-cheaper than always-32B-direct … The efficiency claim survives ONLY against a 32B actually made
to reason."* Latency and energy flip too: compute-lean against 32B-direct goes to −2.3% parallel latency
but **+94.3% sequential and +48.0% energy**.

**What this retires (C26).** *"The method Pareto-dominates every fixed way of using the 32B"* — the
paper's **title**, contribution C2, §5's main-result heading, and the one-line claim in `README.md`,
`PROJECT_OVERVIEW.md` and `READING_GUIDE.md`. The artifact states the distinction precisely:
*"'Pareto-optimal' survives; 'Pareto-DOMINATES' (strictly better on every axis) does NOT hold at equal
weight against always-32B-direct or oracle-mode-32B, and against always-32B-with-reasoning it holds on
accuracy / latency / energy but NOT on FLOP-eq."*

**A nuance that must travel with it**, and is easy to get wrong: macro **cost** answers a different
question from sample-weighted cost — cost is additive per query, so a deployment forecast wants the
sample-weighted number. **Report both, each labelled.** *(Also retired: the "89% of the headline delta
comes from 2 of 8 cells" phrasing is **meaningless under macro** and was replaced by leave-one-cell-out,
whose range here is [0.0225, 0.0732] with `PATH_VQA_open` carrying the claim and `SLAKE_closed` holding
it back. And the MMMU-exclusion rationale **inverts**: MMMU is 0.35% of items but 1/9 = 11.1% of macro
weight.)*

`docs_wording_changes` lists **14** entries; the reporting documents, `METHOD_FINAL_2026-07.md`, the
technical report and the retrospective were re-based between 11:47 and 12:04 and committed at 12:06
(`b5d4244`). The Pareto figure was rebuilt on macro at 12:30 (`dc3cede`, superseded version preserved as
`fig_pareto_superseded_2026-07-08.pdf`) and `fig_overthink` was stopped from asserting a retired claim
and made legible at full width at 12:46 (`d35c7b6`).

**Statistics note, recorded because it bounds every interval above:** the CIs cover **within-dataset**
noise only. Dataset-selection noise is *not* covered — the 8 cells are treated as fixed — and the
substitute diagnostic is the leave-one-cell-out range.

---

## 5. The disjoint verifier retrain (`verifier_disjoint_retrain_2026-07-30.json`, 16:03)

The verdict yesterday's validity audit never wrote. Scripts:
`src/training_methods/build_disjoint_verifier_split.py` → `run_lora_verifier_disjoint.py` →
`verifier_disjoint_measure.py`; 10,000 replicates, seed 0; judged by the **same** `run_judge.py`
(MedVLThinker-32B, `judge_ok`) as the headline.

> **Terminology, because it is easy to misread: "L1" is a de-contamination LEVEL, not a norm.**
> `L1_image_disjoint` = *"no eval image, no eval item; question TEMPLATES may recur"* — the **headline**
> level. `L2_strict` = L1 **plus** no eval question text at all — explicitly *"LOWER BOUND ONLY"*.

**Split design, and the alternative rejected with numbers.** Splitting the *evaluation* sets themselves
would discard **71.2% / 73.0% / 71.9%** of SLAKE-open / VQA-RAD-open / PathVQA-open, so the design
(`option_c_hybrid`) instead trains on the datasets' **official train splits** plus two out-of-domain
pools (Kvasir-open, RadImageNet-open): 16,621 L1 train items / 5,229 images (6,490 at L2), against an
eval side of 2,345 items / 528 images. Disjointness asserted in code, all zero:
`image_pixel_hash_intersection 0`, `question_item_intersection 0`, `question_id_intersection 0`,
`L2_question_text_intersection 0`; 184 L1 question texts shared **by design**. Images are compared by
**md5 of decoded RGB pixels**, so a re-encoded copy is still caught. Training composition matched to the
contaminated reference **exactly** (10,364 examples, 5,182 steps, LoRA r = 16 α = 32, lr 1e-4, seed 0).

**Harness validated first:** running the script with clean ≡ contaminated reproduces
`METHOD_FINAL_2026-07.md`'s open-arm cells exactly (SLAKE 0.8155 @ 12.6%, VQA-RAD 0.5850 @ 5.5%,
PathVQA 0.4533 @ 0.1%, pooled 0.5642).

**The result:**

| cell | n | greedy | **clean L1 gain [CI]** | contaminated gain [CI] | **inflation ×** |
|---|---:|---:|---|---|---:|
| SLAKE-open | 645 | 0.7364 | +0.0109 [−0.0171, +0.0388] **n.s.** | +0.0434 [+0.0155, +0.0714] | **4.00** |
| VQA-RAD-open | 200 | 0.4650 | +0.0150 [−0.0350, +0.0650] **n.s.** | +0.1100 [+0.0650, +0.1600] | **7.33** |
| PathVQA-open | 1,500 | 0.3240 | **+0.0493 [+0.0320, +0.0667] sig** | +0.1293 [+0.1100, +0.1487] | **2.62** |
| **POOLED** | **2,345** | 0.4495 | **+0.0358 [+0.0213, +0.0503]** | **+0.1041 [+0.0891, +0.1190]** | **2.9048** |

**The mechanistic result is the interesting one — contamination did not buy ranking ability, it bought
selection.** Candidate-level AUROC falls only **0.9433 → 0.8856** (L1) → 0.7960 (L2), while **oracle
conversion** — the share of the greedy→oracle-at-8 headroom actually captured — **collapses 0.5894 →
0.2029 → −0.0676**. Memorising the seen items is what turned a good *ranker* into a good *selector*.
**Report conversion, not AUROC, when claiming a selector works.**

**The efficiency consequence is larger than the accuracy consequence.** Because τ is chosen to *reach*
the strong leg's accuracy at minimum escalation, a weaker verifier is paid for in escalation, not
accuracy: escalation needed to hold 32B-direct parity goes **4.0% (contaminated) → 26.9% (clean L1) →
82.7% (L2)**. The open arm's accuracy lands at parity rather than above it: 0.5642 → **0.5143** against
always-32B-direct 0.5168, i.e. `beats_32B_no_think_clean_L1: false`. First-order effect on the
full-suite sample-weighted pooled accuracy is small — 0.5750 → 0.5722 — only because the multiple-choice
arm (94.5% of items) never touches the verifier.

**Verdict, verbatim:** *"YES at L1, at roughly ONE THIRD of its published magnitude … But it survives
ONLY on PathVQA (+0.0493 [+0.0320,+0.0667]); SLAKE (+0.0109) and VQA-RAD (+0.0150) both have CIs
spanning zero, so the pooled significance rests entirely on PathVQA, which is 64% of the open sample.
Under the strictest reading (L2) the gain is null (-0.0119 [-0.0277,+0.0034])."*

*(Why L1 and not L2 is the headline: L2 conflates de-contamination with distribution shift and
**understates** the verifier — it drops 7,306 of 9,903 PathVQA train items, taking candidate AUROC
0.868 → 0.700. L2 is reported as a lower bound a reviewer could reasonably prefer.)*

---

## 6. The three-way headline: macro **and** the clean verifier together (`macro_headline_clean_verifier_2026-07-30.json`, 16:25)

Each correction had only ever been applied alone. `macro_headline_clean_verifier.py` applies them
together on one bootstrap stream (10,000 replicates, seed 20260730, runtime 145.7 s), with a clean
isolation: **only the verifier's P(correct) scores on the three open cells were swapped** — candidates,
judge labels, greedy labels, 32B labels and all five multiple-choice cells are identical. Scores changed
on 630/645, 198/200 and 1,496/1,500 rows. Validation gate: the contaminated column reproduces
`macro_average_headline_2026-07-30.json` on **1,224 / 1,224 compared fields exactly**.

| system | A: published (sample-wtd, contaminated) | B: + macro | **C: + clean L1 (headline)** | D: + L2 |
|---|---:|---:|---:|---:|
| always-32B-direct | 0.5729 | 0.6567 | **0.6567** | 0.6567 |
| always-32B-reasoning | 0.5591 | 0.5974 | **0.5974** | 0.5974 |
| compute-lean | 0.5741 | 0.6600 | **0.6443** | 0.6409 |
| **accuracy-max-veto** | 0.5836 | 0.6694 | **0.6575** | 0.6548 |
| accuracy-max-fusion | 0.5862 | 0.6661 | 0.6503 | 0.6470 |

**The headline comparison, across the ladder:**

| accuracy-max-veto vs always-32B-direct | Δ [95% CI] | verdict |
|---|---|---|
| A — as published | +0.0107 [+0.0086, +0.0127] | **WIN** |
| B — macro only | +0.0128 [+0.0056, +0.0200] | **WIN** |
| **C — macro + clean verifier** | **+0.0008 [−0.0022, +0.0037]** | **TIE** |
| D — macro + L2 | −0.0019 [−0.0055, +0.0014] | TIE |

`does_accuracy_max_still_beat_32b_direct_under_macro_with_a_clean_verifier.answer`: **"NO -- it no longer
beats it"**. And the decomposition is unambiguous: `macro_reweighting_alone` **+0.0021**,
`clean_verifier_alone_under_macro` **−0.0120**, net **−0.0099** — *"Macro re-weighting alone slightly
HELPED this comparison; the clean verifier is what removed it."*

The rest of the column-C picture: compute-lean vs 32B-direct **−0.0124 [−0.0188, −0.0060] LOSS**; fusion
**−0.0063 [−0.0118, −0.0011] LOSS**; veto vs oracle-mode +0.0002 TIE. **Only `accuracy-max-veto` stays
on the frontier** — compute-lean and fusion are now flagged
`method_is_DOMINATED_by_baseline: true` against both 32B-direct and oracle-mode. Cost at C: veto is
**1.74×** the FLOP-eq of a single 32B forward, +16.7% parallel latency, **+149.6% sequential latency**,
**+101.0% energy**.

**What survives.** The vs-reasoning margin: **+0.0601 [+0.0498, +0.0703]** at −87.7% latency / −84.3%
energy (1.74× as-charged, **1.396×** honestly re-costed). The standalone finding that *reasoning mode is
actively harmful on free-text medical VQA* (PathVQA-open 0.1087 vs 0.3760; SLAKE-open 0.6791 vs 0.8186;
VQA-RAD-open 0.5450 vs 0.6000). The multiple-choice-half veto win **+0.0019 [+0.0014, +0.0024]**. And the
verifier is still a real ranker after de-contamination.

**What does not survive** — four claims, listed verbatim: *"The method beats a single 32B forward
pass."* · *"Compute-lean matches the strong model at half the compute."* · *"The open-text arm beats
always-32B-direct."* · *"The method Pareto-dominates the 32B baselines."*

**`one_line_if_only_one_number_is_allowed`, verbatim:** *"8-cell macro, clean verifier: accuracy-max
+0.0008 [-0.0022, +0.0037] vs always-32B-direct at 1.74x its compute -- a tie bought with more compute,
not a win."*

> **Flagged in the artifact and left open**: the macro × clean-verifier × **matched-reasoning**
> combination had *still* not been computed anywhere on disk, so **+0.0601 should be read as an upper
> bound.** That is the gap August 3 closed.

---

## 7. The SLAKE image-path bug — a near-miss, cleared (`slake_image_path_bug_audit_2026-07-30.json`, 09:58)

`src/training_methods/verifier_transfer_eval.py`'s `imgs_for(ds)` has **no `slake_open` branch**; its
`else` branch selects `base = path_vqa/data` for any dataset other than `vqa_rad_open`, so
`imgs_for('slake_open')` would have returned **PathVQA images keyed by PathVQA row index**. Finding that
in the file that evaluates the verifier, on the day the verifier's validity is the open question, is
alarming. Four checks, by `audit_slake_image_path_bug.py`:

1. **Reachability.** SLAKE-open qids are 11,934–12,991; PathVQA test-open has 3,357 rows, max index
   6,717 → **0 SLAKE qids would resolve** under the buggy path (the scorer skips missing keys). The
   actual dump has **645 rows**. The buggy path was *structurally incapable* of producing it.
2. **Provenance.** The real producer is `src/cascade_methods/gen_slake_open_bestofN.py`; 645 rows both
   sides, **scores identical 645, differing 0**.
3. **Image ablation.** The "real" condition reproduces the dump to `mean_abs_dev` **2.6e−06**, max
   5.0e−06, `frac_within_0.01 = 1.0`, while every wrong-image condition deviates by ~0.30–0.33.
4. **Siblings.** 51 scripts scanned; **no** script falls through without its own SLAKE loader.

Git history: the gap existed in the file's **first commit** (`c2db22c`) and in every revision since — *a
latent gap, not a regression*. `published_numbers_affected: "NONE"`. The branch was added anyway.

> **The lesson recorded with it:** when you find a bug in an evaluation path, the question is not *"is it
> wrong?"* but **"was it ever executed, and by what?"** — and a clean verdict deserves the same rigour as
> a damaging one.

---

## 8. Process failures — roughly nine hours and one wasted model load

Recorded because compute on a two-GPU box is the project's scarcest resource and both losses were
self-inflicted. **The log evidence is exact; the operator's reasoning at the moment of each kill is not
written down anywhere on disk and is not reconstructed here.**

### 8.1 A healthy run replaced on a stale progress signal — ~9 hours

`logs/verif_disjoint_master.log` records the disjoint-verifier retrain — **the run gating the entire
open-text claim** — starting **twice, 105 seconds apart** (00:45:01, 00:46:46). The first launcher had
produced no output because it was inside its silent GPU-wait loop. **Silence was read as stalling.**

The replacement then hit the second half of the same defect. Its readiness check was an **instantaneous**
free-memory reading:

```
00:52:47  GPUs free (26 MiB) after 360s
00:52:47  >> GEN slake_open_train (0/2976 done)
00:53:24  GEN slake_open_train rc=1 -> 0/2976 rows
00:53:24  ABORT: generation incomplete for slake_open_train
```

`logs/verif_disjoint_gen.log` gives the true state at that instant: `ValueError: Free memory on device
(46.74/79.14 GiB) on startup is less than desired GPU memory utilization (0.88, 69.64 GiB)`. Another
process still held ~32 GB. **The "free" reading was stale.**

**Cost:** the run did not restart until **09:56:52** — roughly **nine hours** of wall clock on the
project's highest-leverage open item. The fix is visible in the same log: the successful launcher
replaced the point check with a **sustained-free** check (`GPUs sustained-free (26 MiB, 4 consecutive
checks) after 5340s`), waited 89 minutes, and then ran cleanly end to end — generation, judging, two
LoRA trainings and six scoring passes — to `16:01:15 === DISJOINT VERIFIER RETRAIN DONE ===`.

> **Lesson.** A health check must be **sustained and debounced**, never a single instantaneous reading;
> and **silence is not stalling** — a long-running job that legitimately waits looks identical to a hung
> one unless it heartbeats. Require **N consecutive** passing observations before acting on any resource
> signal.

### 8.2 A diagnosed-and-fixed run killed

The InternVL3-38B × MedXpert cell of §3 failed twice (`logs/medeval_direct_matched_master.log`,
01:47:08–01:56:18, then `SKIPPED_AFTER_2_FAILURES`) with `The decoder prompt (length 20183) is longer
than the maximum model length of 16384` — one MedXpert item is ~20.2k tokens of image tiles. The
artifact's own incident note is blunt: this was **not** the documented NCCL hang, it was
**deterministic**, so the retry reproduced it exactly — *"16384 was our error."* **The fix already
existed on disk**: the corresponding `*_reason` arm had hit the same limit and been re-run at
`MAX_MODEL_LEN=24000` (`runners/run_clean_latency_reruns.sh:23`). 24000 was the value **matched to the
reason arm** all along.

Then the fix was applied — and the fixed run was replaced too. At **09:58:08** a relaunch began carrying
the fix (`max_seq_len=24000`, log line 9380), loading a 38 B model across two GPUs. At **10:03:29**,
**5 minutes 21 seconds later**, a fresh master started and re-ran the same cell from scratch, discarding
the in-progress load. The cell finally completed at ~11:25.

**Two distinct failures in one incident:** (a) an automatic retry policy that re-attempted a
**deterministic** error identically, twice, instead of failing fast; (b) a **correctly diagnosed and
correctly fixed** run destroyed before it could produce anything, so the fix had to be paid for twice.

> **Lesson.** Classify failures before retrying — a deterministic error should fail fast and escalate, a
> transient one should retry. Propagate a per-cell configuration (context length, timeout) from whichever
> arm first needed it, so a matched comparison inherits the setting that made its counterpart work. And
> once a run is diagnosed and relaunched with the fix, **let it run** — model-loading silence is not
> failure.

### 8.3 A rounded `parse_ok`

Not a compute loss, but the same species of error and worth the same care. In the first write-up of §3
the extraction-integrity check was stated as **`parse_ok = 1.000` in every new arm** — i.e. *"no answer
was ever unparseable, so the effect cannot be an extraction artifact."* Reading the artifact rather than
the rounded summary, `medeval_matched_direct_2026-07-29.json : cells[*].parse_ok` shows the
`direct_matched` arms at **exactly 1.0000 in 8 of the 9 primary sub-cells, with a minimum of 0.9986**
(InternVL3-38B, MedXpert-Reasoning, n = 1,446 — i.e. **two** unparsed items). Counting all three arms
per sub-cell, 7 of 9 are at exactly 1.0000.

The **conclusion is untouched** — it is still not an extraction artifact. But "1.000" is a *claim of
exactness the data does not support*, and a reader who later found the 0.9986 would have no way to tell
an honest rounding from a covered-up defect. Replaced by: *"`parse_ok ≥ 0.9986` in every new arm; exactly
1.0000 in 8 of the 9 primary sub-cells."*

> **Lesson.** Never round a diagnostic toward the answer you want. Report the **minimum** over sub-cells
> and the **count at the ceiling**, not a rounded mean. A sanity metric that reads exactly 1.000 should
> always be re-read at full precision before it is published.

---

## 9. The comprehensive write-up (`COMPREHENSIVE_WRITEUP_2026-07-30.md`, 19:27)

3,266 lines, assembled with **23 corrections applied at assembly time** — the definitive account of the
project, its corrections and what survives. It supersedes the scattered ledger/spec/diary/artifact set as
the single readable document, and it carries its own **register of what is unverified, unrecorded and
stale** (§11) — which is what commissioned August 3's work. Three entries in that register were flagged
as *never derived or measured anywhere*:

1. **the 4.57 FLOP ratio** — a hard-coded literal at twelve-plus sites, *"no file derives it"*;
2. **the 522 ms best-of-N latency** — *"asserted, not measured"*, with an energy model implying a
   physically impossible ~1,088 W against ~132 W measured;
3. **the macro × clean-verifier × matched-reasoning headline** — *"NOT computed anywhere on disk. The
   +0.0601 should be read as an upper bound."*

---

## 10. Standing state (end of 2026-07-30) and open questions

**Where the headline stands.** Under equal weight per reporting cell, with an uncontaminated verifier:
**accuracy-max ties a single 32B direct forward (+0.0008 [−0.0022, +0.0037]) at 1.74× its compute**;
compute-lean is a **significant loss** (−0.0124 [−0.0188, −0.0060]) at 1.46×. What survives is the
comparison against a 32B *made to reason*: **+0.0601 [+0.0498, +0.0703]** — flagged as an **upper
bound**, since the reasoning arm is still the prompt-unmatched one.

**What survives everything so far.** Finding 1 — reasoning is a net accuracy loss on perception-style
medical VQA (17/20 cells, pooled −0.0401 [−0.0456, −0.0347], n = 30,250) — untouched by either
correction, and now sharpened: the apparent *gains* on reasoning-heavy benchmarks are an **answer-format**
effect, not a reasoning effect.

**Corrections raised today:** **C26** (Pareto-dominates retired), **C27** (the reasoning-half dropped and
re-attributed to answer format), **X20** (the retrospective's own `test_clean.csv` claim falsified),
**X21** (every accuracy level and cost ratio re-labelled sample-weighted, with macro as primary),
**X22** (MedEvalKit reasoning gains re-attributed to `\boxed{}`).

**Open, in the order August 3 took them:**

1. **The three-way combination has still not been computed** — +0.0601 is an upper bound until macro ×
   clean verifier × matched reasoning baseline are applied on one bootstrap stream.
2. **`4.57` is used everywhere and derived nowhere.** Twelve-plus call sites, plus an incompatible
   `4.34` in two others.
3. **`522 ms` was never measured**, and its companion energy model is physically impossible.
4. **PathVQA-open still carries the surviving claim** (leave-one-cell-out drops it to +0.0225) and is
   still a non-random prefix judged by a judge validated on SLAKE and VQA-RAD but **not** PathVQA.
5. **Nothing has been repointed.** Every module still carries the superseded constants, so any figure
   regenerated today republishes them.
6. **R1–R9 from the provenance document are unactioned** — above all **R8**, an audit of `test_clean.csv`
   itself: *"Highest scientific value of anything on this list — nobody has published a defect rate for
   the split the whole field cites."*
7. **MedEvalKit's two local uncommitted edits** (2026-07-02) still **replace** rather than append the
   reasoning trigger. Whether to revert the dependency and re-run is an open decision.
