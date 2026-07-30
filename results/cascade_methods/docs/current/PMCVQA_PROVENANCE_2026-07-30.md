# PMC-VQA: how it is built, how it is validated, and which split we are allowed to claim on

**Date:** 2026-07-30 · **Scope:** provenance + split audit, read-only, no GPU, no new inference.
**Trigger:** the 200-item item-level label audit
(`results/cascade_methods/artifacts/pmc_label_noise_audit_2026-07-29.json`) measured a 28% defect rate
on agree-and-correct controls and 53–60% on the decision-relevant disagreement set, and asked whether
the benchmark — not the method — is the problem.

**Conventions used here.** Every factual claim carries its source: a paper section, a URL, or a
`path:line` on this machine. Claims I re-measured myself on disk today are marked **[measured
2026-07-30]**. Anything I could not establish from a source is marked **UNVERIFIED** and is not used
to support a conclusion. Cite the dataset by its peer-reviewed version (Communications Medicine,
DOI 10.1038/s43856-024-00709-2) or arXiv **v6** — never v1–v5, because the split nomenclature changed.

---

## 1. How PMC-VQA is built

### Plain language

PMC-VQA is not a hand-written medical exam. It is a **machine-written multiple-choice set produced
from figure captions**. Someone scraped ~1.6M figure/caption pairs out of open-access PubMed Central
articles, fed **the caption text only** to ChatGPT, and asked it to invent five questions with four
options each. **No image was ever looked at during question or answer generation.** Everything that
came after was automatic filtering by other models. So the "gold answer" is, by construction, the
answer that is consistent with *the caption* — not the answer a clinician would read off the picture.

That single design fact predicts the exact defect classes our audit found. When a multi-panel journal
figure is later cut into single-panel subfigures, a caption-derived question can travel to the wrong
panel — which is how you get "what does the blue arrow on the head CT show?" attached to a spleen
ultrasound (`pmc-13058`), "which bone is shown / femur" over an axial chest CT (`pmc-24120`), and a
photomicrograph-labelling question over a photo of a cat's face (`pmc-24810`) — all three in the
audit's `verdict.construct` field.

### Precisely, with citations

Paper: *"PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering"*, Xiaoman Zhang,
Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, Weidi Xie (SJTU / Shanghai AI Lab),
arXiv:2305.10415, v1 2023-05-17 → **v6 2024-09-08** (https://arxiv.org/abs/2305.10415). Peer-reviewed
as *"Development of a large-scale medical visual question-answering dataset"*, **Communications
Medicine**, Dec 2024, DOI 10.1038/s43856-024-00709-2, PMID 39709495, free full text
https://pmc.ncbi.nlm.nih.gov/articles/PMC11663219/ (cited below as **PMC11663219**).

1. **Source corpus.** PMC-OA, *"a comprehensive biomedical dataset comprising 1.6 million image-text
   pairs collected from PubMedCentral (PMC)'s OpenAccess subset"*; QA generation draws on *"381K
   image-caption pairs obtained from the first stage of the medical figure collection process"*
   (PMC11663219, Methods).
2. **Generation is caption-only.** *"we input the image captions of PMC-OA, and prompt ChatGPT to
   generate 5 question–answer pairs for each caption"* (PMC11663219, Methods). Verbatim prompt:
   > *"Ask 5 questions about the content and generate four options for each question. The questions
   > should be answerable with the information provided in the caption, and the four options should
   > include one correct and three incorrect options, with the position of the correct option
   > randomized."*

   MedVLThinker (arXiv:2508.02669, https://arxiv.org/html/2508.02669v1) names the generator more
   specifically: *"PMC-VQA was generated using GPT-3.5"*.
3. **Filtering — all automatic** (PMC11663219, Methods):
   - formatting → **1,497,808** QA pairs;
   - text-only answerability filter: *"we trained a question–answer model using LLaMA-7B with text
     data only and eliminated all questions that could be potentially answered by the language
     model"* — *"we will shuffle the choice list and do inference five times. The questions the
     language model can make the right choice three times out of five will be dismissed"* →
     **848,433**;
   - image-dependency classifier, trained on **2,192** manually annotated binary labels (1,752 train
     / 440 test), reported accuracy **81.77%** on that binary task, then applied for data cleaning;
   - final: **226,946** QA pairs over **149,075** images.
4. **Minor arithmetic discrepancy.** Released v1 `train.csv` (176,948) + `test.csv` (50,000) =
   226,948 — two rows more than the paper's 226,946 **[measured 2026-07-30]**. Immaterial, but it
   means the released files are not literally the counted artifact.

**The one design consequence that matters for us:** the gold key is a *caption-derived* key. An item
can be perfectly faithful to its caption and still be unanswerable from, or contradicted by, the
image that ships with it. Nothing in the construction pipeline checks image↔key consistency; the only
model that ever saw an image was the image-*dependency* classifier, which predicts "does this need an
image at all?" at 81.77% accuracy — not "is the key right?".

---

## 2. How it is validated

**Short answer: barely, and only on a 2,000-item subset — with no annotator identity, no agreement
statistic, and no human ceiling.**

The **entire** human-validation record for any test split is one sentence (PMC11663219, Methods):

> *"we randomly selected 50,000 image–question pairs to create an initial test set,
> PMC-VQA-test-initial. Additionally, **we manually checked some test samples again, resulting in a
> small clean test set of 2000 samples**"*.

Criteria, verbatim (PMC11663219): *"(i) whether questions are related to the image and can be
answered via images; (ii) whether the distractor choices in the candidate list are complex enough, to
avoid pure guessing from options; (iii) whether the image quality is good enough, dismissing the
figures which contain too many extra elements."*

Reported outcome, verbatim: *"During this verification procedure, **we have estimated that over 80% of
cases in PMC-VQA-test can be retained**."* This is the authors' own implicit defect estimate: **≲20%**
on the v1 test pool. Note the word *estimated*. It is not presented as a measurement, no per-criterion
breakdown is given, and **the number of items inspected to yield the 2,000 is never stated**, so the
80% cannot be converted into a defect rate on a defined denominator (**UNVERIFIED**, and unverifiable
from the published record).

What is **not** reported anywhere, in any version:

| Missing | Status |
|---|---|
| Who did the manual check; whether any clinician was involved | Never stated. GitHub issue #17, *"Inquiry About Manual Verification Process in PMC-VQA Dataset"* (JerrryNie, 2024-04-11, https://github.com/xiaoman-zhang/PMC-VQA/issues/17) asks exactly this, including whether *"the verification team consist[s] of experts from different medical specialties"* — **still open and unanswered**. |
| Inter-annotator agreement / κ | Not reported in any version; searched, not found (**UNVERIFIED**). |
| Human-performance ceiling on the clean set | Not reported (**UNVERIFIED**). |
| Any verification of the v2 split (`test_2.csv`) | **No statement exists either way.** Version 2 is absent from arXiv v6 and from the Nature version. |

**The authors do name our two biggest defect classes — they just never quantify them on a released
split** (PMC11663219, Discussion/Limitations): *"some of them can be answered correctly using
biomedical knowledge alone, i.e., without the need for a specific image"*; *"some questions in our
data rely on additional information in the caption that cannot be answered with only the corresponding
image"* (their example: *"How many patients were classified into the middle stage?"*); *"our data is
curated from academic papers, where there may be selective use of images to illustrate typical cases
or slices, along with additional annotations such as arrows to aid understanding"*.

So: the audit is not attacking an undefended claim. It is measuring, item by item, a failure mode the
dataset authors described in prose and never counted.

### 2.1 The naming trap (this is where most confusion in this repo came from)

In the paper, **"PMC-VQA-test" IS the 2,000-item verified set** and the 50,000-item set is
"PMC-VQA-test-initial": *"we randomly selected 50,000 image-question pairs to create an initial test
set, PMC-VQA-test-initial… Additionally, we manually checked some test samples again, resulting in a
small clean test set of 2,000 samples, which were manually verified for quality, **termed as
PMC-VQA-test**"* (arXiv 2305.10415v6). **The released file names invert this**: `test.csv` = the
50,000 unverified set; `test_clean.csv` = the 2,000 verified set. Paper results follow the paper's
naming: Table 3 (2,000 verified) MedVInT-TD 40.3% choice / 33.6% blanking; Table 5 (50,000 initial)
39.4% / 32.7%.

> **Rule for this project:** never write "PMC-VQA test" unqualified. Always write the **file name and
> the row count** (`test_clean.csv`, n=2,000 / `test_2.csv`, n=33,430).

**Inference, not verbatim fact:** that `test_clean.csv` (2,000 rows) *is* the paper's manually checked
set is an inference from (a) the paper's "clean test set of 2000 samples", (b) the file's name, and
(c) its exact 2,000 rows **[measured 2026-07-30]**. Nothing on disk states it: the local dataset
README says only *"`test_clean.csv`: metafile of test clean set"*, and the CSV has **no verification
or provenance column** — "clean" is expressed as a separate *file*, not a flag **[measured
2026-07-30]**. I rate the inference high-confidence but mark it **not verbatim-sourced**.

### 2.2 Version 2 ("noncompound images") is undocumented

The only description anywhere is the dataset-card line *"(**update** version-2: noncompound images)"*
on https://huggingface.co/datasets/xmcmic/PMC-VQA and https://huggingface.co/datasets/RadGenome/PMC-VQA,
mirrored in `/data/dan/dataset/medevalkit/PMC-VQA/README.md`. **There is no published verification and
no published accuracy for `test_2.csv`.** GitHub issue #11 (toggle1995, 2023-09-30,
https://github.com/xiaoman-zhang/PMC-VQA/issues/11) asks for v2 accuracy and confirms the paper's
numbers are v1 — **also open and unanswered**. PMC-VQA has 15 issues; **none reports label errors**;
issue #18 is a schema bug (`train_2.csv` lacks `Answer_label`), which is why the HF dataset viewer is
broken.

Whatever "noncompound" means operationally, it is a **re-cut**, not a clean-up: v2 splits compound
figures into single panels and regenerates the QA. That is the mechanism by which a caption-derived
question ends up over the wrong panel — i.e. v2 plausibly *increases* the unanswerable rate relative
to v1. (Mechanistic reading, **UNVERIFIED** — the authors never describe the v2 procedure.)

### 2.3 Measured relationships between the released splits

All **[measured 2026-07-30]** on `/data/dan/dataset/medevalkit/PMC-VQA/` with `csv.reader` (raw
`wc -l` over-counts: fields contain embedded newlines).

| File | Data rows | Version | Columns | Verified? |
|---|---:|---|---|---|
| `train.csv` | 176,948 | v1 | 8-col | no |
| `test.csv` | 50,000 | v1 | 8-col | no — paper's "PMC-VQA-test-initial" |
| **`test_clean.csv`** | **2,000** | v1 | `Figure_path, Question, Answer, Choice A–D, Answer_label` | **yes — the only manually checked split** |
| `train_2.csv` | 152,603 | v2 | 10-col | undocumented |
| **`test_2.csv`** | **33,430** | v2 | `index, Figure_path, Caption, Question, Choice A–D, Answer, split` | **no / undocumented** |

- `test_clean` ⊂ `test.csv`: **2,000 / 2,000** on (`Figure_path`, normalized `Question`).
- `test_clean` ∩ `train.csv` figures: **0** — properly held out of v1 train.
- **`test_clean` ∩ `test_2`: 6 items of 2,000** (base-stripped figure + normalized question). The two
  test sets are effectively disjoint populations, not two views of one benchmark.
- `test_2.split` is the constant `'test'` for all 33,430 rows — it carries no sub-split information.
- `test_2` gold-letter distribution is skewed: C 12,636 / B 11,984 / A 4,423 / D 4,387 (i.e. gold-A is
  13.2%), despite the generation prompt asking for randomized correct-option position.
- Residual non-clinical content is present in **all** splits: MDPI non-clinical journal figure names
  (`ijerph-`, `plants-`, `materials-`, `sensors-`, …) occur in **3.6%** of `test_clean` and **5.0%** of
  `test_2` by my regex. A parallel count with a different journal list gave 2.9% / 3.9%, so treat this
  as "a few percent, roughly half density in the verified split" — the exact rate is journal-list
  dependent and should not be quoted to two digits.

---

## 3. Which split we used, and whether that was the right choice

**We used both — in two eras — and nobody wrote it down. That is the actual root cause of this
investigation.**

| Track | PMC-VQA split | n | Share of its own pool | Verified? |
|---|---|---:|---:|---|
| CVGIP cascade / margin gate, via MedVLThinker-Eval (8,220) | **`test_clean.csv`** | 2,000 | **24.3%** (2,000/8,220) | **yes** |
| Lingshu-faithful / MedEvalKit (44,694) | **`test_2.csv`** | 33,430 | **74.8%** of 44,694; **78.3%** excl. MedXpertQA | **no** |

**Evidence that the MedEvalKit track is `test_2`.** `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`
hard-codes it:

```python
csv_path = os.path.join(dataset_path,"test_2.csv")
```

No env var, no argument; `test.csv` and `test_clean.csv` are unreachable through this class. Gold is
the `Answer` column verbatim (`:49`), which in `test_2` is already a letter; scoring is
`judge_multi_choice` (`MedEvalKit/utils/utils.py:98`) → plain micro accuracy (`PMC_VQA.py:83`).
`MedEvalKit/datas` is a symlink to `/data/dan/dataset/medevalkit`, so the audit's paths resolve to the
files above. And the whole file was evaluated, not a subset: `eval_results_lingshu7b_full/{}/PMC_VQA/
metrics.json` = `total 33430, right 18141, acc 0.5426562967394556`; `eval_results_lingshu32b_full` =
`total 33430, right 18446, acc 0.5517798384684415` — exactly the audit's `acc_7b 0.5427` /
`acc_32b_nt 0.5518` **[measured 2026-07-30]**.

**This was upstream's choice, not ours.** MedEvalKit remote is
`https://github.com/alibaba-damo-academy/MedEvalKit.git` at HEAD `9b12e3b`;
`git diff HEAD -- utils/PMC_VQA/PMC_VQA.py` is empty. Lingshu (arXiv:2506.07044v4) names the benchmark
*"PMC-VQA (v2; Zhang et al. 2024b)"*, uses MedEvalKit, and reports Lingshu-7B 56.3 / Lingshu-32B 57.9.
Our faithful reproduction lands on the same 33,430 rows at 0.5518, 2.7 pts under the published 57.9
(cause not investigated — **UNVERIFIED**, but not a split artifact since the split is confirmed
identical).

**Evidence that the cascade track is `test_clean`.** MedVLThinker's paper says only "the test set of
PMC-VQA", but its released eval set resolves the ambiguity empirically. `/data/dan/dataset/
MedVLThinker-Eval/data/*.parquet` = 8,220 rows (`pathvqa_closed` 3,362; **`pmc_vqa` 2,000**;
`MedXpertQA-MM` 2,000; `slake_closed` 416; `vqa_rad_closed` 272; `MMMU-medical` 170). Sorting the
`pmc_vqa` slice by `dataset_index` (0…1999) and comparing positionally against `test_clean.csv`
**[measured 2026-07-30]**: **2,000/2,000** on normalized `question`, **2,000/2,000** on `answer_label`,
**2,000/2,000** on normalized `answer` text. And our own dumps inherit that gold:
`ckpts/gate_7b_prune/cap320/ckpt_PMC-VQA_nothink_norag.jsonl` has 2,000 rows whose `gold` matches
`test_clean.Answer_label[idx]` **2,000/2,000**.

### Was it the right choice?

- **For the cascade track: yes, and better than we knew.** It runs on the only human-checked split,
  and PMC-VQA is under a quarter of that pool (`pathvqa_closed`, 3,362, is the largest cell). The
  briefing's framing — "the headline leans on PMC-VQA at 79% of the pool" — **is true only of the
  MedEvalKit pool** (33,430/42,694 excluding MedXpertQA = 78.3%; the audit records `pmc_weight
  0.7917` over 42,224). It is **not** true of the cascade pool.
- **For the MedEvalKit/Lingshu track: defensible but weak, and it must be labelled.** Using `test_2`
  is exactly what the shared harness and the published baseline do, so the numbers are *comparable*.
  But it is a split with **zero** published verification, it is ~78% of that pool, and it is the split
  the audit measured 53–60% decision-relevant defects on. Comparable-but-weak, not incomparable.
- **Gate calibration is clean either way** **[measured 2026-07-30]**: the calibration sample
  (`/data/dan/dataset/pmc_vqa_train/train_sample_3000.jsonl`, built by
  `src/data_prep/sample_pmcvqa_train_heldout.py:12` from v1 `train.csv` with an eval-question filter
  at lines 20–27/39) shares **0** figures with `test_clean` and has **0** figure+question duplicates;
  v1 `train.csv` shares **0** figures with v1 `test.csv`. 15 calibration questions match a
  `test_clean` question after aggressive normalization, but **0** share a figure — generic recurring
  stems ("What type of radiograph was taken?"), not item leakage. Cross-version: **0** of 176,948
  v1-train rows collide with a `test_2` item and **0** share a `test_2` source figure.

### 3.1 Two repo-doc claims are falsified — correct them

`results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md` (Hole 10, ~lines
1603–1606) states that `test_clean` *"is not on disk"* at `/data/dan/dataset/medevalkit/PMC-VQA/` and
*"has never been used anywhere in the repo"*; row **X14** (line 2079) repeats it. Both are wrong
**[measured 2026-07-30]**:

- `/data/dan/dataset/medevalkit/PMC-VQA/test_clean.csv` — 418,686 bytes, 2,000 data rows, mtime
  2026-06-29 07:18; second copy at `/data/dan/dataset/pmc_vqa_train/test_clean.csv` (md5
  `6abfbcd088171c76a98911c5e7a8f5a0`, byte-identical between the two locations).
- The entire cascade track already evaluates exactly those 2,000 items (§3 above).

So the retrospective's prescribed fix ("download `test_clean.csv`, re-run both legs on those ~2,000
items, ~20 min GPU") is **already satisfied for the cascade track** and outstanding **only** for the
MedEvalKit/Lingshu track. (Minor: the retrospective cites the hard-code at `PMC_VQA.py:41`; it is
line **39**.) *This document does not edit the retrospective — it is read-only. The correction should
be applied there in a separate, deliberate edit.*

Provenance of the local copies, for the record: all PMC-VQA CSVs came from HF dataset
`RadGenome/PMC-VQA` at revision **`b56ae594f794867893143b337b4118a835794647`** (recorded in
`/data/dan/dataset/pmc_vqa_train/.cache/huggingface/download/*.metadata` and in
`/data/dan/hf_cache/hub/datasets--RadGenome--PMC-VQA/refs/main`); `test_clean.csv` was fetched
separately on 2026-06-08 15:55:01 UTC, ~3 h after the other v1/v2 CSVs; the `medevalkit` copy was
populated 2026-06-29 07:15–07:23 by MedEvalKit's own `maybe_download_dataset` (`PMC_VQA.py:86–100`),
which `rmtree`s its `.cache` (`:99`) — hence no metadata there, but the md5s match. Integrity: the
git-blob-sha1 of the local `test_clean.csv` equals the hash in the HF metadata
(`3d7ccdefb395277c9d57f9684fd75320c3638bbd`) — not corrupted, not locally edited.

---

## 4. Is there a cleaner split, and can we use it?

**Yes — `test_clean.csv`, 2,000 rows, already on disk in two places. But you cannot get it by
filtering the existing `test_2` dumps, and at n=2,000 it is marginally underpowered for the effect
we are claiming.** Both halves of that answer are definitive.

### Route A — filter the 33,430-row dumps down to the verified subset: **dead.**

`test_clean` ∩ `test_2` = **6 items** (rows 5922, 5950, 6003, 11301, 28861, 29319; CSV `index` 65638,
65916, 66384, 128616, 1141890, 1146985) **[measured 2026-07-30]**. 515 of `test_clean`'s 1,440 unique
figures reappear (base-stripped) in `test_2`, and 1,162 `test_2` rows share a source figure — but the
QA pairs were **regenerated**, so item identity does not transfer. **Close this line of inquiry
permanently.**

### Route B — use the clean-split dumps that already exist: **available now, zero GPU.**

All idx-aligned to `test_clean`, golds verified positionally **[measured 2026-07-30]**:

| Dump | n | acc | rows with `opt_logprobs` |
|---|---:|---:|---:|
| `ckpts/gate_7b_prune/cap320/ckpt_PMC-VQA_nothink_norag.jsonl` | 2,000 | 0.5430 | **2,000** |
| `ckpts/gate_7b_prune/{cap80,cap160,cap640}/…` | 2,000 ea. | — | 2,000 ea. |
| `ckpts/gate_7b_vllm/ckpt_PMC-VQA_nothink_norag.jsonl` (fullres) | 2,000 | 0.5395 | **2,000** |
| `ckpts/gate_32b_modes/nothink_cap320/…` | 2,000 | 0.5510 | **2,000** |
| `ckpts/gate_32b_modes/nothink_fullres/…` | 2,000 | **0.5650** | **2,000** |
| `ckpts/gate_32b/ckpt_PMC-VQA_think_norag.jsonl` (think, fullres) | 2,000 | 0.5565 | only **659** |
| `ckpts/gate_32b_modes/think_cap320/…` | 2,000 | 0.5435 | only **672** |

And for the **Lingshu** family that carries the *current* headline, a paired 500-item subsample of the
same clean 2,000, both legs, full logprobs: `ckpts/gate_lingshu7b_mcq/ckpt_PMC-VQA_nothink_norag.jsonl`
acc **0.604**; `ckpts/gate_lingshu32b_mcq/…` acc **0.640**; identical `idx` set (paired bootstrap
valid); golds match `test_clean[idx]` 500/500 and `test_2[idx]` only 161/500. Structure: agree 349
(69.8%), agree-and-correct 253, disagree 151 (**30.2%**, vs the audit's `disagree_rate 0.3297` on
`test_2`), 7B-only-right 49, 32B-only-right 67, oracle 0.738. The subsample looks uniform over
0…1999 (mean 1053.4 vs 999.5; per-quartile 111/114/142/133 vs 125, all within ~2σ of
Binomial(500,0.25)) — **but no producer script exists** for it anywhere in `src/`, `runners/`, or
`logs/` (grep finds only the consumer `src/cascade_methods/logit_fusion.py:32-33`, plus
`STRUCTURE.md:433` and `RESULTS.md:85`), so its sampler and seed are **UNVERIFIED**.

**How many items would remain? 2,000 (MedVLThinker family) or 500 (Lingshu family) — not a filtered
subset of 33,430, but a separate, fully-evaluated population.** All 2,000 clean-split images resolve
locally in `/data/dan/dataset/medevalkit/PMC-VQA/images/` (0 missing), so even re-running through
MedEvalKit needs no download — only a one-line vendor patch at `PMC_VQA.py:39` plus a choice between
v1's `Answer` (text) and `Answer_label` (letter) as gold.

### The power problem — state it before running anything

The published `test_2` fusion delta is **+0.0135, paired-bootstrap CI [0.0100, 0.0169]**
(`results/cascade_methods/artifacts/beat32b_fusion.json`, `/per_benchmark/PMC_VQA/`), i.e. half-width
0.00345 at n=33,430 **[measured 2026-07-30]**. Scaling as 1/√n:

| n | CI half-width | can certify +0.0135? |
|---:|---:|---|
| 500 | 0.0282 | **no** |
| 2,000 | 0.0141 | **no — just barely short** |
| 2,183 | 0.0135 | threshold |

The scaling model is validated against real data: the observed F11_fixed CI on the clean 500 is
[−0.020, 0.036], half-width **0.028**, matching the predicted 0.0282
(`results/cascade_methods/artifacts/logit_fusion.json`, `per_slice."PMC-VQA"`).

Consistent with that, **nothing certifies on the clean 500** (same artifact, n=500, acc7 0.604 /
acc32 0.640): `F3_confadv` (the policy the headline uses) acc 0.638, **d = −0.002**, CI
[−0.032, 0.028]; `F11_fixed` acc 0.648, d = +0.008, CI [−0.020, 0.036]; `F6_cd` d = 0.000 (held-out
α = 0.0 in all 5 folds → contrastive decoding collapses to always-32B); `F11_rw` acc 0.630, d = −0.010.
**The sign is not even stable.** This is **not a refutation** — a +0.0135 effect is unfindable at
n=500 by construction — but it must be said plainly: **the clean split currently provides no positive
evidence for the PMC win.** Scaling the win/loss rates (`n_win_fusion` 1969 − `n_loss_fusion` 1518 =
451 net at n=33,430) gives **~27 net items at n=2,000** and **~6 at n=500**.

**Feasibility caveat if you want to audit the clean split itself:** `test_clean.csv` is v1 8-column and
has **no `Caption` column**, and v1 captions exist nowhere on disk. Joining base figure names to the
v2 CSVs recovers a caption for only **721 of 2,000** (36%) **[measured 2026-07-30]**. The
caption-provenance channel the `test_2` audit relied on is unavailable for ~64% of the clean split, so
a like-for-like defect-rate comparison cannot be done at full coverage from local data. Options:
scope the audit to the 721 caption-covered rows and say so, or fetch v1 captions from PMC OA (the
496 MB `oa_comm_use_file_list.csv` gives PMCID → article mapping).

---

## 5. How our measured defect rates compare

| Source | Split | Rubric | Rate |
|---|---|---|---|
| **Authors** (PMC11663219) | v1 test pool | criteria (i)–(iii) above; *estimated*, denominator unstated | *"over 80% … can be retained"* → **≲20% defective** |
| **Our audit** (`pmc_label_noise_audit_2026-07-29.json`) | **`test_2`** (never verified) | BAD-GOLD ∪ UNANSWERABLE ∪ MULTI-CORRECT, precedence-ordered, UNCLEAR never counted as a defect | **28%** agree-and-correct control (n=50) · **53%** fusion wins (n=100) · **60%** fusion losses (n=50) · **55.3%** all decisive disagreements |
| ECCV 2026 precedent (arXiv:2607.00159) | E-VQA / InfoSeek | unsupported answers; ambiguous questions | 22% unsupported (InfoSeek); **59% / 47%** ambiguous (E-VQA / InfoSeek) |

**These are not in conflict, and it is important not to present them as if they were.** Three things
differ at once: (a) **split** — the authors' 80% refers to v1; our audit is 200/200 in `test_2`
(only 8/200 also in `test_clean`; loader `src/cascade_methods/pmc_label_noise_audit.py:86`), a split
the authors never claimed to have checked; (b) **rubric** — ours counts unanswerable-from-image and
multi-defensible as defects, which is stricter than "can be retained"; (c) **conditioning** — 53%/60%
are conditioned on model disagreement, which selects for hard/ambiguous items by construction. The
one comparable cell is our **28% unconditioned-ish control** vs their **≲20% estimate** — same
direction, modestly worse, on a worse split.

**Qualitative corroboration from the papers we build on.** MedVLThinker (arXiv:2508.02669) — the
source of both our models and the 8,220-item pool — concedes it in its appendix: *"The PMC-VQA dataset
was generated automatically by GPT-3.5 from journal figures and captions. **Many of the questions may
be simplistic or flawed, and the answers might not always require deep reasoning (or could even be
incorrect in some cases).**"* Lingshu (arXiv:2506.07044) makes the general point: *"Many open-source
multimodal medical datasets are extracted automatically from scientific papers and thus often contain
noise and redundancy."*

**No independent quantitative audit of PMC-VQA label error exists** (searched; **UNVERIFIED** in the
sense that absence of evidence is not proof — but PMC-VQA's 15 GitHub issues contain no label-error
report, and no paper I found measures a PMC-VQA defect rate). DiN (CVPR 2025, arXiv:2503.18536) treats
medical-VQA "semantic noisy labels" as a modelling problem; whether it reports a PMC-VQA-specific
noise rate is **UNVERIFIED** (search summaries only).

**Two corroborating signals from our own numbers, both suggestive not probative.** (1) The same
models score **6–9 points higher** on the clean split than on `test_2`: Lingshu-7B 0.604 vs 0.5427,
Lingshu-32B 0.640 vs 0.5518 **[measured 2026-07-30]** — consistent with better keys, but confounded
by the fact that v1 and v2 are different item populations, not the same items re-keyed. (2) The
audit's noise-ceiling analysis bounds achievable `test_2` accuracy at roughly **0.63–0.77** depending
on how defective items are treated (`noise_ceiling` block), while every system in this project scores
0.54–0.57 — the benchmark is **not** saturated, but a ~1-point margin is being measured where
~31–37% of items cannot support a correctness claim at all.

---

## 6. What this means for the project's claims

Taking the audit's own arithmetic (`corrected_deltas` in the audit artifact) together with the
provenance above, the exposure is **narrower than the briefing implies, but the construct problem is
real and is not fixed by any split choice.**

1. **The "biased annotation error" attack on the PMC win fails, measurably.** Defects are *not*
   concentrated on the wins: wins 53% vs losses 60%, difference −0.07, Fisher p = 0.487 (not
   significant), sign favouring the losses; mis-keying specifically is symmetric (BAD-GOLD 9% of wins
   vs 10% of losses). Re-keying only the mis-keyed items leaves +0.0124; the symmetric
   drop-the-defective correction leaves **+0.0094, CI [0.0004, 0.0183]** — point estimate survives,
   CI nearly touches zero.
2. **But the delta cannot be called a medical-visual accuracy improvement.** 46% of the audited wins
   sit on items where the gold is wrong or the answer is simply not in the shown image. On those items
   the score is decided by **which model's language prior better matches a caption-derived key**, not
   by which model reads the image better. `pmc-25510` is the clean illustration: the 32B correctly
   described the panel that was actually shown and was scored **wrong**. This is exactly what §1
   predicts from caption-only generation — so it is a property of the benchmark's construction, not an
   accident of sampling.
3. **Therefore restate, don't retract.** The correct phrasing for any `test_2`-based PMC number is
   **"higher agreement with PMC-VQA's caption-derived answer keys"**, with the measured 53%
   decision-relevant defect rate stated alongside. Reserve the word *accuracy* for cells that survive
   an item-level validity check.
4. **Pooled leverage is the structural fix, and it is free.** Reporting the pooled headline over the
   MedVLThinker-Eval 8,220 pool — **fully covered by existing MVT dumps for all 7 slices** (MMMU 170,
   MedXpert-Reasoning 1,446, MedXpert-Understanding 554, PMC-VQA 2,000, PathVQA 3,362, SLAKE 416,
   VQA-RAD 272) — puts PMC at **24.3%** instead of **79.2%**, a ~3.3× reduction in the noisiest
   benchmark's weight, with **no new inference**. It also removes the "the whole result rests on the
   noisiest benchmark" objection outright.
5. **The compute story is untouched.** FLOPs/latency/energy claims do not depend on the gold key being
   right. And `compute-lean`'s PMC cell is **−0.0010** (negative), so correcting PMC cannot help or
   hurt it (audit `pooled_headline.compute_lean_vs_32B_reasoning`: pooled +0.0150 measured, +0.0152
   after symmetric correction). Likewise the open-text cells and retrospective holes 1/3/4 are out of
   scope of this audit.
6. **The split confusion itself is a reportable finding.** The project ran PMC-VQA on the human-curated
   2,000-item `test_clean` in the MedVLThinker era and silently moved to the 33,430-item v2 `test_2`
   in the MedEvalKit era, and the two overlap on **6 items**. The 28–60% defect rates apply to
   `test_2` **only** and must not be attributed retroactively to the MVT-era PMC numbers.

---

## 7. Recommendation — ranked

**R1. Restate the PMC claim and name the split everywhere. Cost: writing only. (Do this regardless of
everything else.)**
Replace "accuracy gain on PMC-VQA" with "higher agreement with PMC-VQA's caption-derived keys (53%
decision-relevant defect rate measured on `test_2`, n=200 audited)". In every table and sentence, write
the **file name + row count**. Add a two-sentence benchmark-provenance caveat with citations:
caption-only GPT-3.5 generation (PMC11663219 Methods), one undocumented 2,000-item human pass with a
*≲20% estimated* defect rate, no annotator identity (issue #17 open), no agreement statistic, and a v2
split the authors never described or scored (issue #11 open). Cite MedVLThinker's own appendix
admission — it is the strongest available third-party citation — and arXiv:2607.00159 (22% unsupported
/ 47–59% ambiguous on E-VQA/InfoSeek) so the audit reads as *consistent with the literature on
auto-built VQA benchmarks*, not as an outlier claim.
**This is the "change nothing but restate" option, and it is genuinely defensible on its own**, for
one reason: `test_2` via MedEvalKit is what Lingshu and the shared harness publish. Switching splits
unilaterally would forfeit comparability with every published baseline. What it buys: honesty, and
immunity to the exact reviewer attack the audit modelled. What it does not buy: any strengthening of
the evidence.

**R2. Re-report the pooled headline on the MedVLThinker-Eval 8,220 pool. Cost: CPU only, hours.**
All 7 slices are already covered by existing dumps. Cuts PMC's weight 79.2% → 24.3% and moves the
headline onto the human-verified PMC split. Buys the single biggest reduction in exposure per unit of
work, with zero new inference. **Pair with R1, not instead of it.**

**R3. Replicate the PMC fusion/veto comparison on the clean 2,000, MedVLThinker family, zero GPU.
Cost: CPU, a few hours.**
Cheap leg `ckpts/gate_7b_prune/cap320/…` (0.5430); strong leg `ckpts/gate_32b_modes/nothink_fullres/…`
(0.5650); both have `opt_logprobs` on all 2,000 rows and golds verified 2,000/2,000.
**Pre-register the expected outcome before running:** n=2,000 gives CI half-width ~0.0141 against a
+0.0135 effect (threshold n≈2,183), so **a non-significant result is the expected result and is not a
refutation**. Report the point estimate and CI either way. Buys the maximum-power clean-split evidence
obtainable without a GPU — and, if the point estimate lands positive, a genuinely stronger claim.
*Note the 32B **think** dumps on the clean split are logprob-poor (659/2,000 and 672/2,000), so any
confidence-based policy involving 32B-think is restricted to the no-think legs here.*

**R4. Correct the retrospective. Cost: minutes.**
`PROJECT_RETROSPECTIVE_2026-07-29.md` Hole 10 (~1603–1606) and row X14 (2079): `test_clean.csv` **is**
on disk in two locations, and the cascade track has been evaluating exactly those 2,000 items all
along. Also fix the line reference `PMC_VQA.py:41` → `:39`. Leaving a falsified "hole" in the canonical
retrospective is worse than having had the hole.

**R5. Record the provenance so this cannot recur. Cost: minutes.**
Add a short provenance note (STRUCTURE.md section, or a README beside the data) recording: HF revision
`b56ae594…`; the two download dates; the v1↔v2 file mapping (`test.csv`/`test_clean.csv`/`train.csv` =
v1 + `images/`; `test_2.csv`/`train_2.csv` = v2 + `figures/`); the measured **6-item** overlap; the
naming inversion; and the fact that **MedVLThinker-Eval's `pmc_vqa` == `test_clean.csv`**. The absence
of exactly this note is what caused the confusion.

**R6. Patch MedEvalKit locally to accept a split argument (`PMC_VQA.py:39`) and add `test_clean` as a
robustness *column*, not a replacement. Cost: ~20 min GPU + a small vendor diff.**
Keep `test_2` as the primary reported number for peer comparability; add the clean-split column beside
it. Requires deciding gold = `Answer_label` (letter) for the v1 format. Buys a two-split table, which
is the honest and strongest presentation of an accuracy claim on a noisy benchmark.

**R7. Extend Lingshu clean-split coverage 500 → 2,000. Cost: one small GPU job (1,500 items × 2
models, no-think MCQ with logprobs, reusing the `ckpts/gate_lingshu{7b,32b}_mcq/` schema).**
Brings the family that carries the *current* headline onto the clean split at maximum available n.
Still marginally underpowered (see R3), so its value is a credible point estimate + CI, not
certification. Also worth documenting the unknown 500-item sampler while you are there.

**R8. Audit the clean split itself (n≈200, same rubric). Cost: auditor time, no GPU.**
Highest scientific value of anything on this list — **nobody has published a defect rate for the split
the whole field cites** — and it directly tests the authors' *≲20%* estimate. But budget the caption
gap first: only **721/2,000** clean rows have a locally recoverable caption, so either scope to those
721 and say so, or fetch v1 captions from PMC OA via `oa_comm_use_file_list.csv`.

**R9. Housekeeping, cheap, prevents a future footgun.**
(a) Quarantine or label the un-attributed `MedEvalKit/eval_results/` (a **third** 33,430-row PMC run,
acc 0.5437) and `eval_results_32b/` — `STRUCTURE.md:~436` already flags them; any script globbing
`eval_results_*` could silently pick them up (the audit avoided this only by naming tags explicitly).
(b) Check v1-train ↔ `test_2` overlap at **PMC-article-ID** level, not figure-path level — I verified
0 figure-path and 0 figure collisions, but figure naming differs across versions so a string check can
return a spurious zero (**article-level check: not done, UNVERIFIED**).

**Explicitly do NOT do:** filter the 33,430-row dumps to the verified subset. **6 items survive.**
Record this as closed so it is not re-attempted.

---

### Loose ends, stated as loose ends

- **UNVERIFIED:** who verified the 2,000, and whether any clinician was involved (issue #17 open).
- **UNVERIFIED:** how many items were inspected to yield the 2,000 — so "over 80% retained" cannot be
  converted into a defect rate.
- **UNVERIFIED:** whether `test_2.csv` received any verification pass; no statement exists either way.
- **UNVERIFIED:** any inter-annotator agreement, κ, or human ceiling for PMC-VQA.
- **UNVERIFIED:** why upstream MedEvalKit chose v2 (`PMC_VQA.py:39` carries no comment); and whether
  the authors intended v1/v2 test as alternatives or independent benchmarks.
- **UNVERIFIED:** the defect rate of `test_clean` itself (no audit exists; only 36% caption-feasible
  locally).
- **UNVERIFIED:** the sampler/seed behind the 500-item Lingshu clean-split subsample (no producer
  script on disk).
- **UNVERIFIED:** the cause of the 2.7-pt gap between our Lingshu-32B reproduction (0.5518) and the
  published 57.9 — the split is confirmed identical, so it is not a split artifact.
- **Not re-derived here:** the +0.0135 delta and its CI are read from
  `results/cascade_methods/artifacts/beat32b_fusion.json`; the power figures are analytic 1/√n
  scalings of that published CI (validated against the observed n=500 CI), not fresh bootstraps.
