# The open-text arm, 17 August – 4 September 2026: full rundown
**Every figure names the artifact it came from. Nothing here is estimated or interpolated.**

---

## 0. What the method is, in the field's terms

A lightweight **MLP probe** reads the frozen hidden states of an unmodified **Lingshu-7B** and acts
as a **best-of-N verifier**: the 7B samples 8 candidate answers, the probe scores each, the
top-scoring one is returned. It is compared against **greedy decoding**, an **answer-prior
baseline** (a counter over answer strings — no image, no hidden state), and **self-consistency**.

Terminology follows `TERMINOLOGY_2026-08-24.md`: *benchmark* not "cell", *probe* + *verifier* not
"MLP head", *candidate set* not "pool", *answer-prior baseline* not "string prior".

---

## 1. Where we started and where we are

| | 17 Aug | 4 Sep |
|---|---|---|
| open-ended benchmarks | 3 | **8** |
| questions | 2,345 | **30,912** |
| probe training rows | 31,439 | **108,126** |
| feature caches | 2 | 40+ (8 benchmarks × 4 temperatures, + fine-layer, + visual, + Qwen) |
| generators tested | 1 | 2 (Qwen2.5-VL-7B in flight) |

**Headline, incumbent frozen probe, full benchmarks** (`free_signal_bakeoff_2026-08-21.json`):

| benchmark | n | greedy | answer prior | self-consistency | **verifier** | oracle@8 |
|---|---:|---:|---:|---:|---:|---:|
| PathVQA | 1,500 | 0.3427 | 0.3333 | 0.3260 | **0.3900** | 0.5167 |
| SLAKE | 645 | 0.7302 | 0.7287 | 0.7395 | **0.7690** | 0.8791 |
| VQA-RAD | 200 | 0.4900 | 0.4350 | 0.4650 | 0.4650 | 0.6300 |
| RadImageNet | 2,000 | 0.3210 | 0.2825 | 0.3245 | 0.3295 | 0.5120 |
| Kvasir-x1 | 10,121 | 0.2849 | 0.2815 | 0.2699 | **0.3629** | 0.4696 |
| OmniMedVQA | 8,883 | **0.5164** | 0.4746 | 0.5162 | 0.4971 | 0.7007 |
| VQA-Med C4 | 3,663 | **0.0947** | 0.0459 | 0.0863 | 0.0688 | 0.2102 |
| GEMeX | 8,000 | 0.3974 | 0.3549 | 0.3794 | **0.4121** | 0.5864 |

The verifier beats the answer-prior baseline on **all eight**, including both benchmarks where it
loses to greedy.

---

## 2. Data built (4 new benchmarks)

| benchmark | questions | images | what it is | how the candidate set was kept clean |
|---|---:|---:|---|---|
| **Kvasir-x1** | 10,121 | 2,870 | GI endoscopy, disjoint test split | 1,052 burned frames + **136 perceptual near-duplicates** excluded, some identical to a training frame at Hamming distance 0 |
| **OmniMedVQA** | 8,883 | — | 7 modalities, modality & anatomy questions | RadImageNet source dropped entirely (it is one of our own evaluation benchmarks); disease-diagnosis types excluded as options-dependent |
| **VQA-Med C4** | 3,663 | — | ImageCLEF 2019 abnormality naming | 10 MedPix collisions excluded at build time |
| **GEMeX** | 8,000 | 3,514 | chest X-ray findings, free text | 21,312 images pulled from PhysioNet under our own credential; no image overlap with any evaluation pool |

**GEMeX is the least memorisable benchmark in the project**: 6,236 distinct questions, 4,316
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

### 3.2 Multi-layer ensembling + self-consistency as an input feature — **+0.0168**
`head_best_config_2026-08-24.json`, end-to-end on full benchmarks, 5 seeds:

| | A: deployed | B: ensemble 18/20/22 | C: +SC feature | **D: both** |
|---|---:|---:|---:|---:|
| MACRO | +0.0199 | +0.0324 | +0.0297 | **+0.0366** |
| beats greedy | 5/8 | 7/8 | 7/8 | **7/8** |

D flips three failures: VQA-RAD −0.0150→**+0.0300**, OmniMedVQA −0.0063→**+0.0220**,
RadImageNet +0.0010→+0.0125. Both components were found on a proxy metric and **both survived
end-to-end validation** — which the layer sweep did not.

### 3.3 The stack, and one component that vanishes
`head_final_stack_2026-08-24.json`, identical held-out halves:

| arm | macro |
|---|---:|
| four-domain, single layer | +0.0243 |
| pooled, single layer | +0.0765 |
| **pooled + layer ensemble** | **+0.0802** |
| pooled + ensemble + SC | +0.0797 |

The SC feature is worth +0.0098 from the four-domain base and **−0.0005 once pooled** — it was
compensating for missing training data, not adding independent signal. Not shipped.

### 3.4 Doubling the sample budget — **+0.0164**
`coverage_scaling_ALL_2026-09-01.json`, matched budget on identical questions, all eight:
OmniMedVQA +0.0460, GEMeX +0.0365, Kvasir-x1 +0.0316, RadImageNet +0.0299, PathVQA +0.0157,
VQA-Med +0.0055, SLAKE −0.0030, VQA-RAD −0.0309. **MACRO +0.0164, 6/8 positive.**

### 3.5 The shipped artifact
`ckpts/train/genframe_head_pooled_ens/` — 24 heads (3 layers × 8 seeds), 108,126 rows, BCE,
per-layer frozen standardizers. **Reload-verified: +0.0816 from disk** vs the incumbent recipe's
+0.0243 on the same halves. The incumbent `genframe_head_ens8` is byte-intact; the freeze script
refuses to target it. The artifact **must not be evaluated on full benchmarks** — it has seen the
other image half of each.

---

## 4. What did not work (with measured bounds)

| # | attempt | result | artifact |
|---|---|---|---|
| 1 | **Architecture** (7 variants) | plain `raw` wins at +0.0734; domadv +0.0558, poolnorm+raw +0.0527, poolnorm +0.0409, pca32 +0.0386, pca128 +0.0339, rankonly +0.0302 — **all six alternatives worse** | `head_arch_transfer_2026-08-19` |
| 2 | **Domain breadth**, budget-matched | mean slope +0.00868 **[−0.00143, +0.01997]** with the duplicated Kvasir source collapsed — includes zero | `head_domain_scaling_MERGED_2026-08-21` |
| 3 | **Breadth on unseen benchmarks (LOBO)** | **−0.0008** | `head_lobo_pooled_2026-08-25` |
| 4 | **Union pools over temperature** | loses to the best single temperature on **6 of 7** | `mixed_temperature_2026-08-22` |
| 5 | **Pool pruning** (5 pruners × k∈{2,3,4,6}) | best fixed rule **−0.0000** | `pool_pruning_2026-08-24` |
| 6 | **Self-consistency as a scorer** | flat at the random floor, and **does not improve with N** (0.0933 at N=2, 0.0800 at N=16 while oracle climbs 0.156→0.418) | §4 of `TRANSFER_WALL_2026-08-21` |
| 7 | **Greedy-anchored veto** | +0.0007; **ceiling +0.0022** even with τ fitted in-sample | `greedy_anchored_2026-08-22` |
| 8 | **Regime detection** (6 detectors) | none order the benchmarks once GEMeX is added (knn −0.548, domclf +0.548) | `regime_detector_2026-08-21` |
| 9 | **Answer kind** | within-benchmark, long answers favour the verifier in 6/8 but mean only **+0.0045** | `answer_kind_2026-08-22` |
| 10 | **Train/deploy temperature matching** | no diagonal advantage; best cell is the *mismatched* one | `head_temp_matched_2026-08-24` |
| 11 | **Ensemble width** | shipped {18,20,22} +0.1051 already best; all-five +0.1040 | `head_ens_width_2026-08-25` |
| 12 | **Visual features** | +h_img **+0.0002**; the question-token control +0.0018 beats it | `head_visual_features_2026-09-01` |

---

## 5. Structural findings

### 5.1 The decomposition — the failures are the candidate set, not the ranker
`decomposition_2026-08-24.json`. `verifier − greedy = selection skill − sampling penalty`:
**selection skill is positive on 7 of 8 benchmarks** (mean +0.0403 against mean penalty +0.0256).
VQA-RAD has +0.0244 of genuine skill and still loses because its penalty is 0.0494.

### 5.2 LOBO — the verifier is per-benchmark, not general
`head_lobo_pooled_2026-08-25.json`: four-domain +0.0312 → LOBO +0.0304 → pooled +0.0803.
**Breadth −0.0008, own data +0.0500.** The method needs a labelled split per benchmark.

### 5.3 The price of onboarding a benchmark — **~100 labelled questions**
`head_price_from_lobo_2026-08-30.json`, from the deployable base. 6/8 already beat greedy at k=0;
the first hundred questions carry most of the gain (OmniMedVQA −0.0004→+0.0986, RadImageNet
+0.0010→+0.0528, GEMeX +0.0224→+0.0857). Then it flattens.

### 5.4 Temperature — the shipped verifier wants T=0.7
`head_temp_ensemble_2026-08-30.json`: T=0.2 +0.0401, T=0.4 +0.0573, **T=0.7 +0.0816**, T=1.0 +0.0759.
Unlike the layer effect, pooling did *not* flatten this (spread 0.0416). A stronger verifier prefers
**more** diverse candidate sets because it can exploit the extra coverage.

### 5.5 GEMeX — the cleanest evidence of real verification
verifier +0.0148 [+0.0057, +0.0238] over greedy, and **+0.0573 [+0.0483, +0.0663] over the answer
prior** — where the prior itself *loses* to greedy by 0.0425. It is also the **most**
out-of-distribution benchmark (largest kNN distance 7.228, lowest domain-classifier confidence
0.693) and the verifier wins on it anyway, so OOD-ness does not predict where the method helps.

### 5.6 Layer choice — three metrics, three answers, all within ~0.013
In-domain CV picks 20, the transfer proxy picks 18, end-to-end picks 19 (L18 +0.0275, L19 +0.0331,
L20 +0.0235, L21 +0.0201, L22 +0.0277). Deployed layer 21 is worst of the five. With the **pooled**
probe the spread collapses to 0.003 — pooling makes layer choice nearly irrelevant.

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
   nothing" came from VQA-RAD and VQA-Med — the two *lowest*-gain of the eight. Macro is **+0.0164**.
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
- **The driver exited on an empty queue** — cost four idle GPU-days. Now sleeps and re-plans.
- **A self-refilling planner is only as deep as its catalogue** — it later idled 33 h because every
  job type it knew was finished.
- **Per-GPU `flock`** — vLLM reserves a fixed *fraction* of the card, so two jobs do not share.
- **GEMeX provenance** — the licence warning printed unconditionally, so a `--source physionet` run
  reported third-party re-hosting untrue of it.

---

## 8. In flight

| experiment | status |
|---|---|
| **Second generator (Qwen2.5-VL-7B)** | pipeline complete — 8 splits judged and extracted, 170,014 training rows; analysis running. The only test of whether this is a *method* or a fact about Lingshu. InternVL3-8B was tried first and is architecturally incompatible. |
| 32-sample pools | queued on the coverage-limited pair |
| Full 4×4 temperature matching | T=0.4 / T=1.0 training features extracted |

---

## 9. What I would claim today

**Positive.** A frozen 918k-parameter MLP probe reading one layer of an unmodified 7B beats greedy
decoding on 4 of 8 open-ended medical VQA benchmarks and beats an answer-frequency baseline on
**all 8** — including GEMeX, where that baseline loses to greedy outright. Retrained on all eight
and rank-ensembled over three layers it reaches **+0.0802** macro against **+0.0243** for the
original recipe.

**Mechanism.** `verifier − greedy = selection skill − sampling penalty`. Skill is positive on 7/8;
failures are the candidate set, not the ranker.

**Deployment.** The method needs a labelled split per benchmark — breadth alone buys nothing — and
that split costs **~100 labelled questions**.

**The honest hole.** We cannot predict *which* benchmarks the method helps. OOD distance doesn't do
it, answer kind doesn't, and no detector orders the benchmarks. Twelve documented dead ends bound
the alternatives, several with measured ceilings rather than mere nulls.
