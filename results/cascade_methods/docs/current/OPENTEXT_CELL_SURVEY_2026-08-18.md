# Open-text cell survey — what can and cannot become a 4th..Nth cell
**2026-08-18.** Written while the RadImageNet extraction runs. Every access fact was verified
against the live HF API or the file on disk on this date, not recalled.

## 0. The constraint nobody had written down

**MedEvalKit ships exactly three open-text VQA loaders** — `SLAKE` (open split), `VQA_RAD` (open
split), `PATH_VQA` (open split). That *is* our three cells. Everything else it ships is
multiple-choice (`PMC_VQA`, `MedXpertQA`, `OmniMedVQA`, `MedFrameQA`), yes/no (`Radrestruct`) or
report generation (`IU_XRAY`, `CheXpert_Plus`, `MIMIC_CXR`). **There is no fourth cell to switch
on.** Any additional open-text cell is our own harness, so it will not be a
"MedEvalKit-faithful" number the way the current eight are. Say so in the paper.

## 1. Verdicts

| candidate | open-ended? | clean? | verdict |
|---|---|---|---|
| **radimagenet_open** (2,000 q / 1,000 img) | yes, 1–3 words | **yes — 0 rows in head train, 0 in LoRA** | ✅ **ADOPT** |
| kvasir_open (1,200 q / 1,052 img) | yes | ❌ 5,562 rows = 18% of the head's train pool | ⚠️ needs a refit without it |
| Kvasir-VQA-x1 official test (15,955 q / 4,058 img) | yes | ❌ **1,052 of its 4,058 images are our kvasir training images** | ⚠️ usable only on the ~3,006 disjoint images |
| Quilt-VQA (940 OPEN / 1,283) | **no — long form** | n/a | ❌ **REJECT for this arm** |
| ProbMed (57,132 q / 6,303 img) | **no — 100% yes/no** | n/a | ❌ reject for this arm, ✅ **gift for the prompt-bias result** |
| GEMeX | yes (dedicated open split) | n/a | ⛔ **not obtainable** — see §3 |
| Medical-Diff-VQA | yes | n/a | deferred (two-image questions; PhysioNet) |

## 2. Why the two rejections are rejections

**Quilt-VQA answers are an order of magnitude longer than ours.** Measured on
`quiltvqa_test_w_ans.json`:

| set | n | answer words: mean / median / p90 |
|---|---|---|
| Quilt-VQA `answer_type=OPEN` | 940 | **20.4 / 17.0 / 37.0** |
| Quilt-VQA `answer_type=CLOSED` | 343 | 15.6 / 13.0 / 27.0 |
| our slake_open gold | 645 | 1.72 / 1.0 / 3.0 |
| our pathvqa_open gold | 1,500 | 2.36 / 1.0 / 5.0 |

That is not the same task. It breaks four things at once: the judge prompt assumes a short phrase;
`max_tokens=64` truncates; the duplicate-collapse that turns 8 samples into ~3.8 distinct answers
will not happen when every sample is a unique 20-word paragraph; and "is this candidate correct"
stops being a clean binary, which is what `oracle@8` and `sel_eff` are built on. Adopting it means
a different judge and a different metric — a separate paper, not a fourth cell.

**ProbMed is 100% binary.** All 57,132 questions end in the literal string
`(please answer yes/no)`; the answer vocabulary is `{no: 29,129, yes: 28,003}`.

## 3. GEMeX is not blocked by credentialing — it is not released

`BoKelvin/GEMeX` **404s**. The only public repo, `BoKelvin/GEMeX-VQA`, contains four `.jsonl` files
of **19 rows each**. The GitHub (`Awenbocc/GEMeX-Project`) says "we release part of data". The
paper claims 151,025 images / 1,605,575 questions with a dedicated open-ended split, and the image
paths in the sample are MIMIC-CXR study IDs. **PhysioNet access does not help** — the annotations
themselves are not published. Getting it means emailing bokelvin.liu@connect.polyu.hk.

## 4. The ProbMed lead — for the OTHER result

ProbMed is useless to the open-text arm and close to ideal for the prompt-bias finding:

- **57,132 questions over 6,303 images**, every one carrying the answer-space instruction whose
  removal is worth **+0.0419** on PathVQA-closed at 1.000× compute
  (`output_bias_correct_2026-08-17.json`).
- Its answer key is **49.0% yes** — near-balanced, so the gold-balanced control that killed the
  rival PMC intervention is nearly free here.
- It spans modalities (CT / MRI / X-ray) and body parts, so it tests whether the effect is a
  PathVQA quirk or a property of how these harnesses are written.

If the yes-bias reproduces on ProbMed, the prompt result stops being one cell on one dataset.

## 5. What RadImageNet costs, and why it was free

Everything but the greedy arm was already on disk, generated in the June/July out-of-domain-pool
work and never spent as eval:

| arm | file | state |
|---|---|---|
| 7B best-of-8 + judge | `cheap_lingshu7b/..._sc8_scexploded.judge.jsonl` | 8,155 candidates ✅ |
| 7B T=0.4 pool + judge | `cheap_lingshu7b_T04/...` | 5,294 ✅ |
| 7B best-of-32 + judge | `cheap_lingshu7b_scale/...` | 19,242 ✅ |
| **Lingshu-32B direct + judge** | `strong_lingshu/..._lingshu32b_t0` | 2,000 ✅ |
| InternVL3-8B / 38B | `internvl3_*` | ✅ |
| 7B greedy | — | **run 2026-08-18** |
| layer-{7,14,21,28} features | — | **running 2026-08-18** |

Measured straight from the existing judge files: **Lingshu-32B direct 0.2890, 7B oracle@8 0.5120.**
The judge is not the reason the 32B is low — of the 1,422 answers it marks wrong, **only 3 (0.2%)
contain the gold string**, so this is not a paraphrase artifact. Sample errors are genuine:
*interstitial lung disease → "Lung cancer"*, *no pathology seen → "Pulmonary metastasis"*.

**And the head has never seen it** — 0 rows in `feats_hidden/generator_train_*`, 0 examples in
`lora_verifier_disjoint`. Applying the frozen 8-seed head is therefore a true out-of-domain
transfer test, which is a stronger claim than a 4th in-domain cell would have been.

## 6. Sources

Kvasir-VQA-x1 arXiv:2506.09958 · RadImageNet-VQA arXiv:2512.17396 · GEMeX arXiv:2411.16778 ·
OmniMedVQA arXiv:2402.09181 · ProbMed "Worse than Random?" (github.com/eric-ai-lab/ProbMed) ·
Quilt-VQA / Quilt-LLaVA (quilt1m.github.io) — note its licence forbids redistribution and
commercial use, and requires citing both Quilt-1M and Quilt-LLaVA.
