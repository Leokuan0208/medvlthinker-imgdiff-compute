# The open-text arm, 17 August – 4 September 2026: full rundown
**Every figure names the artifact it came from. Nothing here is estimated or interpolated.**

> ⚠️ **Audited 2026-09-12. Read `AUDIT_2026-09-12.md` first.** 175 of 182 checkable claims verified
> exact; the corrections below are applied in place, and the audit file is the supersession record.

> ## ⚠️ SUPERSEDED IN PART, 2026-09-20 — read `AUDIT_2026-09-18.md` before quoting anything here
> **Four things changed after this document was written, and every one of them is propagated below.**
> 1. **The 2026-09-13 PathVQA backfill** (PathVQA went from a 1,500-question prefix to the full 3,357)
>    regenerated most of the artifacts this file cites. The whole **+0.0802 family is now +0.0736**
>    (`head_final_stack_PVFIXED_2026-09-13.json`), and §1, §3.2, §3.3, §3.5, §4 row 3, §5.1, §5.2, §5.3,
>    §5.4 and §9 are corrected in place below. Corpus **35,012 → 36,869 questions**.
> 2. **The judge is MedVLThinker-32B, not Lingshu-32B** (`src/labeling/run_judge.py:21`, no runner
>    overrides it). It is text-only and sees the gold. Every number in this file is **judge currency**.
> 3. **Judge currency is not the only currency.** On identical picks over the same 18,452 held-out
>    questions the shipped probe is **+0.0737 [+0.0608, +0.0864]** under the judge of record,
>    **+0.0679** under a cross-family judge (MedGemma-27B-it), **+0.0156** token-F1 and **+0.0047
>    [−0.0077, +0.0171] (a TIE)** under lenient exact match
>    (`em_rescore_pooled_probe_2026-09-18.json`, `xjudge_rescore_medgemma27b_2026-09-20.json`).
> 4. **§5.6's layer numbers are unreproducible** and are marked as such in place — do not quote them.
>
> **See `AUDIT_2026-09-18.md` §2 (currencies), §3 (replication), §4 (the extraction pass), §5 (baselines).**

---

## 0. What the method is, in the field's terms

A lightweight **MLP probe** reads the frozen hidden states of an unmodified **Lingshu-7B** and acts
as a **best-of-N verifier**: the 7B samples 8 candidate answers, the probe scores each, the
top-scoring one is returned. It is compared against **greedy decoding**, an **answer-prior
baseline** (a counter over answer strings — no image, no hidden state), and **self-consistency**.

Terminology follows `TERMINOLOGY_2026-08-24.md`: *benchmark* not "cell", *probe* + *verifier* not
"MLP head", *candidate set* not "pool", *answer-prior baseline* not "string prior".

**Currency (added 2026-09-20).** Unless a line says otherwise, every accuracy in this document is
**judge currency**: the grader is **MedVLThinker-32B** (`src/labeling/run_judge.py:21` default; no
runner overrides it), it is **text-only** and it **sees the gold answer** — a paraphrase-equivalence
grader, not Lingshu-32B. Under lenient exact match the same picks are a tie (§9).

**What "reads the frozen hidden states" means in the code as run (added 2026-09-20).** The features
were **not** captured during generation. `src/training_methods/extract_generator_hidden.py:437-439`
runs a separate teacher-forced HuggingFace forward pass per (image, question, candidate), at
`max_pixels = 1,003,520`, while the candidates themselves were generated under vLLM at `cap320 =
250,880`. "Adds no forward pass" is a property of the **design**, not of what was run
(`AUDIT_2026-09-18.md` §4).

---

## 1. Where we started and where we are

| | 17 Aug | 4 Sep |
|---|---|---|
| open-ended benchmarks | 3 | **8** |
| questions | 2,345 | **36,869** (was 35,012 before the 2026-09-13 PathVQA backfill) |
| probe training rows | 31,439 | **112,770** (was 108,126 before the same backfill) |
| feature caches | 2 | 40+ (8 benchmarks × 4 temperatures, + fine-layer, + visual, + Qwen) |
| generators tested | 1 | 2 (Qwen2.5-VL-7B in flight) |

**Headline, incumbent frozen probe, full benchmarks** (`free_signal_bakeoff_2026-08-21.json`):

| benchmark | n | greedy | answer prior | self-consistency | **verifier** | oracle@8 |
|---|---:|---:|---:|---:|---:|---:|
| PathVQA | **3,357** | **0.3140** | **0.2833** | **0.2943** | **0.3452** | **0.4808** |
| SLAKE | 645 | 0.7302 | **0.7302** | **0.7426** | **0.7721** | 0.8791 |
| VQA-RAD | 200 | 0.4900 | 0.4250 | 0.4600 | 0.4650 | 0.6300 |
| RadImageNet | 2,000 | 0.3210 | 0.2825 | 0.3245 | 0.3295 | 0.5120 |
| Kvasir-x1 | 10,121 | 0.2849 | 0.2815 | 0.2699 | **0.3629** | 0.4696 |
| OmniMedVQA | 8,883 | **0.5164** | 0.4746 | 0.5162 | 0.4971 | 0.7007 |
| VQA-Med C4 | 3,663 | **0.0947** | 0.0459 | 0.0863 | 0.0688 | 0.2102 |
| GEMeX | 8,000 | 0.3974 | 0.3549 | 0.3794 | **0.4121** | 0.5864 |

The verifier beats the answer-prior baseline on **all eight**, including both benchmarks where it
loses to greedy. *(Rows corrected 2026-09-20 against `free_signal_bakeoff_2026-08-21.json` as it
stands today: PathVQA was regenerated on the full 3,357 questions, and SLAKE's and VQA-RAD's
answer-prior / self-consistency / verifier columns moved with the argsort tie-break fix of commit
`e021633`. The "beats the prior on all eight" statement still holds on the corrected rows.)*

---

## 2. Data built (4 new benchmarks)

| benchmark | questions | images | what it is | how the candidate set was kept clean |
|---|---:|---:|---|---|
| **Kvasir-x1** | 10,121 | 2,870 | GI endoscopy, disjoint test split | 1,052 burned frames + **136 perceptual near-duplicates** excluded, some identical to a training frame at Hamming distance 0 |
| **OmniMedVQA** | 8,883 | — | 7 modalities, modality & anatomy questions | RadImageNet source dropped entirely (it is one of our own evaluation benchmarks); disease-diagnosis types excluded as options-dependent |
| **VQA-Med C4** | 3,663 | — | ImageCLEF 2019 abnormality naming | 10 MedPix collisions excluded at build time |
| **GEMeX** | 8,000 | 3,514 | chest X-ray findings, free text | 21,312 images pulled from PhysioNet under our own credential; no image overlap with any evaluation pool |

**GEMeX is the least memorisable benchmark in the project**: 6,236 distinct questions, 4,315
distinct golds, top-10 gold coverage **12.7%** (RadImageNet 66.5%, SLAKE 41.9%, PathVQA 41.3%).

---

## 3. What worked

### 3.1 Retraining on all eight benchmarks — **+0.0565** (the largest lever)
`head_pooled_alldomains_2026-08-24.json`. The probe had never been trained on the four new
benchmarks — every number on them was zero-shot transfer. Fitting on the training half of all eight
(108,126 rows vs 31,439), strict by-image split, evaluated on held-out halves:

| benchmark | greedy | 4-domain | **pooled** | gain |
|---|---:|---:|---:|---:|
| PathVQA | 0.3214 | +0.0714 | +0.0886 | +0.0171 |
| SLAKE | 0.7121 | +0.0455 | +0.0545 | +0.0091 |
| VQA-RAD | 0.5052 | −0.0515 | −0.0309 | +0.0206 |
| RadImageNet | 0.3337 | +0.0189 | **+0.1215** | +0.1026 |
| Kvasir-x1 | 0.2811 | +0.0891 | +0.1219 | +0.0328 |
| OmniMedVQA | 0.5216 | −0.0105 | **+0.1556** | +0.1661 |
| VQA-Med | 0.0913 | −0.0138 | −0.0072 | +0.0066 |
| GEMeX | 0.3997 | +0.0216 | **+0.1184** | +0.0968 |
| **MACRO** | | **+0.0213** | **+0.0778** | **+0.0565** |

The gain is largest exactly where the probe had never seen the domain.

*(2026-09-20: this table still matches `head_pooled_alldomains_2026-08-24.json` exactly — but that
artifact was **never regenerated** after the PathVQA backfill, so its PathVQA row is the truncated
700-question held-out half and its 108,126 pooled rows are the pre-fix count. The numbers are
internally consistent; they are not on the same footing as §3.3's PVFIXED figures.)*

### 3.2 Multi-layer ensembling + self-consistency as an input feature — **+0.0161**
`head_best_config_2026-08-24.json`, end-to-end on full benchmarks, 5 seeds:

| | A: deployed | B: ensemble 18/20/22 | C: +SC feature | **D: both** |
|---|---:|---:|---:|---:|
| MACRO | **+0.0182** (was +0.0199 before the 2026-09-13 PathVQA backfill) | **+0.0306** (was +0.0324) | **+0.0281** (was +0.0297) | **+0.0343** (was +0.0366) |
| beats greedy | 5/8 | 7/8 | 7/8 | **7/8** |

*(D − A is **+0.0161**, not the +0.0168 in the heading of the pre-backfill version.)*

D flips three failures: VQA-RAD −0.0150→**+0.0300**, OmniMedVQA −0.0063→**+0.0220**,
RadImageNet +0.0010→+0.0125. Both components were found on a proxy metric and **both survived
end-to-end validation** — which the layer sweep did not.

### 3.3 The stack, and one component that vanishes
`head_final_stack_PVFIXED_2026-09-13.json`, identical held-out halves (18,452 questions), judge
currency. **This replaces `head_final_stack_2026-08-24.json`, which was fitted with PathVQA
truncated to a 1,500-question prefix; the old values are kept in brackets so the history reads.**

| arm | macro | beats greedy |
|---|---:|---:|
| four-domain, single layer | **+0.0182** (was +0.0243) | 6/8 |
| pooled, single layer | **+0.0729** (was +0.0765) | 6/8 |
| **pooled + layer ensemble** (shipped) | **+0.0736** (was +0.0802) | 6/8 |
| pooled + ensemble + SC | **+0.0720** (was +0.0797) | 7/8 |

The SC feature is worth +0.0098 from the four-domain base and **−0.0016 once pooled** (the
pre-backfill figure was −0.0004) — it was compensating for missing training data, not adding
independent signal. Not shipped.

### 3.4 Doubling the sample budget — **+0.0147**
`coverage_scaling_ALL_2026-09-01.json`, matched budget on identical questions, all eight:
OmniMedVQA +0.0460, GEMeX +0.0365, Kvasir-x1 +0.0316, RadImageNet +0.0299, PathVQA +0.0018,
VQA-Med +0.0055, SLAKE −0.0030, VQA-RAD −0.0309. **MACRO +0.0147, 6/8 positive.** (Restated 2026-09-13: PathVQA was +0.0157 on the truncated 700-question half and is +0.0018 on the full 1,623, which moves the macro from +0.0164.)
**Superseded 2026-09-16 by a measurement with intervals:** `coverage_sc16_ci_ALL_2026-09-16.json`
re-runs 8→16 on matched questions with an image-clustered bootstrap and gives **macro +0.0167,
4 WIN / 0 LOSS / 4 TIE** (RadImageNet +0.0329, Kvasir-x1 +0.0309, OmniMedVQA +0.0462, GEMeX +0.0359
are the wins). Quote that one; the row above carries no interval.

### 3.5 The shipped artifact
**`ckpts/train/genframe_head_pooled_ens_v2/`** (refitted 2026-09-13; it supersedes the
`genframe_head_pooled_ens` named here originally, which was fitted on 108,126 rows with PathVQA
truncated to 1,500 of 3,357 questions — both are kept on disk) — 24 heads (3 layers × 8 seeds),
**112,770 rows** (was 108,126), BCE, per-layer frozen standardizers. **Reload-verified: +0.0737
from disk** (was +0.0816) vs the incumbent recipe's **+0.0182** (was +0.0243) on the same halves.
Source for both: `recipe.json` in that directory, `measured_on_held_out_halves`. The incumbent
`genframe_head_ens8` is byte-intact; the freeze script refuses to target it. The artifact
**must not be evaluated on full benchmarks** — it has seen the other image half of each.

> ⚠️ The `README.md` inside `genframe_head_pooled_ens_v2/` still prints the pre-backfill
> +0.0802/+0.0243 and contradicts its own sibling `recipe.json`. It is gitignored and must be
> rewritten by hand (`AUDIT_2026-09-18.md` §9 item 3); the generating script was fixed on this
> branch so it can no longer hardcode those strings.

---

## 4. What did not work (with measured bounds)

| # | attempt | result | artifact |
|---|---|---|---|
| 1 | **Architecture** (7 variants) | plain `raw` wins at +0.0734; domadv +0.0558, poolnorm+raw +0.0527, poolnorm +0.0409, pca32 +0.0386, pca128 +0.0339, rankonly +0.0302 — **all six alternatives worse** | `head_arch_transfer_2026-08-19` |
| 2 | **Domain breadth**, budget-matched | mean slope +0.00868 **[−0.00143, +0.01997]** with the duplicated Kvasir source collapsed — includes zero | `head_domain_scaling_MERGED_2026-08-21` |
| 3 | **Breadth on unseen benchmarks (LOBO)** | **−0.0056** (was −0.0008 before the 2026-09-13 PathVQA backfill) | `head_lobo_pooled_2026-08-25` |
| 4 | **Union pools over temperature** | loses to the best single temperature on **7 of 8** | `mixed_temperature_2026-08-22` |
| 5 | **Pool pruning** (5 pruners × k∈{2,3,4,6}) | best fixed rule **−0.0000** | `pool_pruning_2026-08-24` |
| 6 | **Self-consistency as a scorer** | flat at the random floor, and **does not improve with N** (0.0933 at N=2, 0.0800 at N=16 while oracle climbs 0.156→0.418) | §4 of `TRANSFER_WALL_2026-08-21` |
| 7 | **Greedy-anchored veto** | +0.0006; **ceiling +0.0023** even with τ fitted in-sample | `greedy_anchored_2026-08-22` |
| 8 | **Regime detection** (6 detectors) | none order the benchmarks once GEMeX is added (knn −0.548, domclf +0.548) | `regime_detector_2026-08-21` |
| 9 | **Answer kind** | within-benchmark, long answers favour the verifier in 6/8 but mean only **+0.0045** | `answer_kind_2026-08-22` |
| 10 | **Train/deploy temperature matching** | no diagonal advantage; best cell is the *mismatched* one | `head_temp_matched_2026-08-24` |
| 11 | **Ensemble width** | shipped {18,20,22} +0.1051 already best; all-five +0.1040 | `head_ens_width_2026-08-25` |
| 12 | **Visual features** | +h_img **+0.0002**; the question-token control +0.0018 beats it | `head_visual_features_2026-09-01` |

---

## 5. Structural findings

### 5.1 The decomposition — the failures are the candidate set, not the ranker
`decomposition_2026-08-24.json`. `verifier − greedy = selection skill − sampling penalty`:
**selection skill is positive on 7 of 8 benchmarks** (mean **+0.0387**, was +0.0403 before the
2026-09-13 PathVQA backfill, against mean penalty **+0.0257**, was +0.0256).
VQA-RAD has +0.0244 of genuine skill and still loses because its penalty is 0.0494.

### 5.2 LOBO — the verifier is per-benchmark, not general
`head_lobo_pooled_2026-08-25.json`: four-domain **+0.0279** (was +0.0312 before the 2026-09-13
PathVQA backfill) → LOBO **+0.0223** (was +0.0304) → pooled **+0.0707** (was +0.0803).
**Breadth −0.0056** (was −0.0008)**, own data +0.0485** (was +0.0500). The method needs a labelled
split per benchmark. *(Breadth alone now buys slightly less than nothing, which strengthens rather
than weakens this section's conclusion.)*

### 5.3 The price of onboarding a benchmark — **~100 labelled questions**
`head_price_from_lobo_2026-08-30.json` (re-fitted on the corrected PathVQA, commit `e2aa6f0`), from
the deployable base. 5/8 already beat greedy at k=0 (OmniMedVQA crosses at k=50); the zero-shot
macro is **+0.0196**; the first hundred questions carry most of the gain (OmniMedVQA
**−0.0108→+0.0998**, was −0.0004→+0.0986; RadImageNet **+0.0149→+0.0578**, was +0.0010→+0.0528;
GEMeX **+0.0292→+0.0855**, was +0.0224→+0.0857). Then it flattens.
⚠️ The curve is **3 seeds**, against this project's own ≥10-seed rule (`AUDIT_2026-09-18.md` §5).

### 5.4 Temperature — the shipped verifier wants T=0.7
`head_temp_ensemble_2026-08-30.json`: T=0.2 **+0.0393** (was +0.0401 before the 2026-09-13 PathVQA
backfill), T=0.4 **+0.0539** (was +0.0573), **T=0.7 +0.0773** (was +0.0816), T=1.0 **+0.0710** (was
+0.0759). Unlike the layer effect, pooling did *not* flatten this (spread **0.0381**, was 0.0416).
⚠️ T=0.7 beats T=1.0 by **+0.0063**, at the edge of this project's own 0.0061 tie-break band
(`AUDIT_2026-09-18.md` §5) — the temperature choice is not separated from noise. A stronger verifier prefers
**more** diverse candidate sets because it can exploit the extra coverage.

### 5.5 GEMeX — the cleanest evidence of real verification
verifier +0.0148 [+0.0057, +0.0238] over greedy, and **+0.0573 [+0.0483, +0.0663] over the answer
prior** — where the prior itself *loses* to greedy by 0.0425. It is also the **most**
out-of-distribution benchmark (largest kNN distance 7.228, lowest domain-classifier confidence
0.686) and the verifier wins on it anyway, so OOD-ness does not predict where the method helps.

### 5.6 Layer choice — three metrics, three answers, all within ~0.013

> ⚠️ **UNREPRODUCIBLE — do not quote the per-layer numbers below (marked 2026-09-20).** The
> per-layer artifacts on disk today (`head_layer19_eval_2026-08-24.json`,
> `head_layer21_eval_2026-08-24.json`) were **re-created** on 2026-09-13 (commit `4b135ce`, pure
> additions) and cover only **5 of 8 benchmarks** — PathVQA, SLAKE and VQA-RAD are silently absent,
> because the backfill wrote them under a different cache stem than the layer-eval scripts read.
> On that reduced set L19 reads +0.0280 and L21 +0.0183, neither of which is the +0.0331/+0.0201
> printed below, and neither is computed over the same benchmarks. **A re-extraction is owed; no
> replacement values are invented here.** (`AUDIT_2026-09-18.md` §7, `docs-md_findings.json` S5.6.)

In-domain CV picks 20, the transfer proxy picks 18, end-to-end picks 19 (L18 +0.0275, L19 +0.0331,
L20 +0.0235, L21 +0.0201, L22 +0.0277 — **all five withdrawn per the banner above**). Deployed
layer 21 is worst of the five. With the **pooled** probe the spread collapses to 0.003 — pooling
makes layer choice nearly irrelevant. *(The qualitative conclusion — pooling flattens the layer
choice — is not what the withdrawn numbers rest on, but it has not been re-measured either.)*

---

## 6. Corrections to our own claims

1. **Stale checkpoint invalidated every OmniMedVQA number.** `run_openvqa.py` resumed on `idx`
   alone; the rebuild re-indexed 0..8882 over a different question set, so regeneration reported OK
   in 160 s having generated nothing. Only 17 of 5,000 questions matched. Corrected: −0.0489 →
   −0.0193, and the answer prior does *not* beat the verifier there. Fixed at source with a
   question-text guard.
2. **The router's "WIN" was my CI.** +0.0133 [+0.0109,+0.0156] i.i.d. → +0.0119 [−0.0032,+0.0297]
   clustered by benchmark. A tie.
3. **Mixed currency in the decomposition.** Exact-match for one term, judge for the others —
   inflated the sampling penalty ~8×. Caught before it reached a deck.
4. **Breadth confounded with volume.** k=6 trained on 6× the rows of k=1.
5. **Kvasir counted twice.** Same GI-endoscopy source as two "independent" domains, carrying the
   two largest slopes and the two largest spreads.
6. **The sampling conclusion drawn from the wrong two benchmarks.** "Doubling the budget buys
   nothing" came from VQA-RAD and VQA-Med — the two *lowest*-gain of the eight. Macro is **+0.0147**.
7. **The self-consistency sign-flip "with no exceptions"** — GEMeX is the exception.
8. **"The regime is detectable"** — did not survive one added benchmark.
9. **MedPix overlap**, 19 images between `vqa_rad_open_train` and VQA-Med. Immaterial (those
   questions score 0.0000 for both arms) but removed.
10. **The AdamW landmine** — `foreach=False` does not fix the segfault; thread count does.
11. **A stub counted as success** — numpy `int64` made `json.dump` raise after each layer's fit,
    leaving a 126-byte file the planner read as "done".

---

## 7. Infrastructure

- **A failed producer deadlocked its consumers** — a chain waited 21 h on an artifact whose
  producing job had already failed.
- **Fixed queues cannot keep GPUs busy** — the runnable set *grows* as jobs land. Replaced with a
  planner that reads the tree each wave.
- **The driver exited on an empty queue** — cost four idle GPU-days (recorded at runners/run_auto_campaign.sh:26, written from the incident). Now sleeps and re-plans.
- **A self-refilling planner is only as deep as its catalogue** — it later idled for more than a day because every
  job type it knew was finished.
- **Per-GPU `flock`** — vLLM reserves a fixed *fraction* of the card, so two jobs do not share.
- **GEMeX provenance** — the licence warning printed unconditionally, so a `--source physionet` run
  reported third-party re-hosting untrue of it.

---

## 8. In flight

| experiment | status |
|---|---|
| **Second generator (Qwen2.5-VL-7B)** | **LANDED 2026-09-13, and the row count here is wrong.** `head_final_stack_qwen_2026-09-13.json`: macro **+0.0820, 8/8**, over **145,085** deduplicated training rows (the 170,014/222,154 counts contained up to 4× duplicated `(ds, idx, candidate)` rows). Judge +0.0814, lenient EM +0.0276, strict EM +0.0187, token-F1 +0.0317 — a WIN in every currency (`replication_currency_2026-09-20.json`). **Caveat: Lingshu-7B *is* a Qwen2.5-VL-7B fine-tune, so this is a second training recipe over one language model, not a second family.** InternVL3-8B was tried first and is architecturally incompatible (its LM is also Qwen2.5-7B). |
| ~~Third generator (MedGemma-4b-it)~~ | ⛔ **WITHDRAWN 2026-09-20.** Its sampled candidate sets are broken — 100 % of sampled candidates ran to the 64-token cap and 58.5 % are chat-template garbage. On questions whose whole candidate set is clean the judge gain falls +0.0436 → +0.0096, and in strict EM the arm is −0.0751 (0/8). **There is currently no valid cross-family replication.** See `AUDIT_2026-09-18.md` §3 and `TRANSFER_WALL_2026-08-21.md` §14. |
| 32-sample pools | queued on the coverage-limited pair |
| Full 4×4 temperature matching | T=0.4 / T=1.0 training features extracted |

---

## 9. What I would claim today

**Positive.** A frozen 918k-parameter MLP probe reading one layer of an unmodified 7B beats greedy
decoding on 5 of 8 (4 of them significantly) open-ended medical VQA benchmarks and beats an answer-frequency baseline on
**all 8** — including GEMeX, where that baseline loses to greedy outright. Retrained on all eight
and rank-ensembled over three layers it reaches **+0.0736** macro (was +0.0802 before the
2026-09-13 PathVQA backfill) against **+0.0182** (was +0.0243) for the original recipe, **in judge
currency (MedVLThinker-32B, text-only, sees the gold)**, on 18,452 held-out questions.

**The currencies, on identical picks** (`em_rescore_pooled_probe_2026-09-18.json`,
`xjudge_rescore_medgemma27b_2026-09-20.json`, `judge_2x2_2026-09-20.json`) — this is the part the
sentence above does not say, and it must travel with it:

| currency | macro | verdict |
|---|---:|---|
| judge of record (MedVLThinker-32B) | **+0.0737 [+0.0608, +0.0864]** | WIN, 6/8 positive |
| cross-family judge (MedGemma-27B-it), same picks | **+0.0679** | 7/8 positive, 5/8 significant |
| a probe graded by a judge it was *not* trained on | **+0.0565 to +0.0658** | the honest cross-judge number |
| token-F1 | **+0.0156 [+0.0044, +0.0265]** | WIN (image-clustered) |
| lenient exact match | **+0.0047 [−0.0077, +0.0171]** | **TIE** |

Exact match is not a valid currency on the three benchmarks whose golds are sentences (oracle@8
strict EM on Kvasir-x1 is 0.0004), and the judge is mostly right but phrase-sensitive; the gap is
**not** verbosity (a pick-the-longest selector *loses* under the judge, −0.0294).

**Mechanism.** `verifier − greedy = selection skill − sampling penalty`. Skill is positive on 7/8;
failures are the candidate set, not the ranker.

**Deployment.** The method needs a labelled split per benchmark — breadth alone buys nothing — and
that split costs **~100 labelled questions** measured from the LOBO base (trained on the other
seven). TRANSFER_WALL's "~500" was measured from the four-domain base and is a different baseline,
not a contradiction.

**The honest hole.** We cannot predict *which* benchmarks the method helps. OOD distance doesn't do
it, answer kind doesn't, and no detector orders the benchmarks. Twelve documented dead ends bound
the alternatives, several with measured ceilings rather than mere nulls.

**Three more holes, added 2026-09-20** (`AUDIT_2026-09-18.md`): (i) the cost claim describes a
system that has not been run — the features come from a separate teacher-forced pass at a higher
image resolution than generation used (§0 above, audit §4); (ii) **7–30 % of the judge-currency
gain is judge-specific** (the 2×2 in audit §2.4), so +0.06 is the defensible cross-judge figure;
(iii) three baselines a reviewer asks first are missing or unreported — generator LoRA-SFT
(+0.0142 at 1× cost, on disk since 2026-08-11 and in no document), 7B+probe vs Lingshu-32B greedy
(loses 2, ties 1, wins 1 of the 4 measured), and the answer-string prior on the shipped probe's own
halves (recovers 25–30 % of the judge gain).
