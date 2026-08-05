# Comparative verification for best-of-8 selection — round synthesis (2026-08-05)

**Question this round asked.** Two things had worked on open-text selection, and nobody had combined
them. (1) **The generator's own frame**: a discriminative head reading the frozen Lingshu-7B's hidden
states under the model's *answering* prompt scored 0.795640 standalone and 0.806540 fused with the
incumbent verifier — the only positive in a programme of ~16 consecutive failures. (2) **Head-to-head
comparison**: real A-vs-B forward passes were the project's only *other* selection win, measured in
July at pointwise 0.783 → round-robin 0.859, and **shelved because round-robin costs 28 forward passes
per question**. The round's thesis was that if the comparative signal could be computed from the
**cached** per-candidate generator-frame vectors, the measured pairwise win would come for free — 28
tiny MLP evaluations over features the deployed pointwise head already computes, instead of 28 VLM
passes. Four architectures were built against that thesis, plus a clean replication of the pairwise
win itself.

**Answer, in one line.** **The thesis is dead, and it died at both ends.** All four comparative
architectures failed to beat a *pointwise* head on identical features, and the pairwise win they were
meant to amortise **does not survive decontamination** — on clean weights, on this pool, with the
engine matched to the bar, real A-vs-B forward passes are a **null** (0.797895–0.803158 against an
incumbent of 0.808421 on the same items, every point estimate negative). There was no win left to make
cheap. What the round *does* leave is a mechanism for why cached-vector comparison cannot work, the
retirement of the "generator frame" story that motivated it, and one free, deterministic, guardrail-clean
deployable change worth **+0.004087 [−0.004087, +0.012262] — not significant** — over what is already
deployed.

Nothing in this round abstains, defers or rejects. Every configuration always returns one of the 8
candidates.

---

## 0. Source tags

Every figure below carries a tag naming the artifact it was read from. All artifacts are under
`results/cascade_methods/artifacts/`.

| tag | artifact |
|---|---|
| `[AUDIT]` | `genframe_cache_audit_2026-08-05.json` |
| `[PAIRHEAD]` | `verifarch_pairhead_2026-08-04.json` |
| `[SETAWARE]` | `verifarch_setaware_2026-08-04.json`, `verifarch_setaware_cv_preregistration.json` |
| `[CHEAP]` | `verifarch_cheapcontrast_2026-08-04.json` |
| `[RPW]` | `verifarch_realpairwise_clean_2026-08-04.json`, `verifarch_realpairwise_hf_2026-08-05.json`, `verifarch_realpairwise_vllm_2026-08-05.json` |
| `[INT]` | `verifarch_integrated_2026-08-04.json` |
| `[HID]` | `verifarch_hidden_2026-08-04.json`, `verifarch_hidden_fusion_controls_2026-08-04.json`, `verifarch_hidden_generatorprompt_2026-08-04.json` |
| `[LIST]` | `verifarch_listwise_2026-08-04.json` |
| `[FEAT]` | `verifarch_features_2026-08-04.json` |
| `[ALIGN]` | `verifarch_alignment_2026-08-04.json` |
| `[JULY]` | `pairwise_verifier_gpu.json` (2026-07-06) |
| `[NSCALE]` | `verifier_n_scaling_2026-08-03.json` |

---

## 1. Shared ground truth

**Endpoint.** Selection efficiency at N=8, `sel_eff = (selected − greedy)/(oracle − greedy)` computed
over recoverable questions, on the open-text pool: **2,345 questions** (slake_open 645 / vqa_rad_open
200 / pathvqa_open 1,500), **1,468 recoverable**, **528 distinct images** `[AUDIT]`.

| quantity | value | source |
|---|---:|---|
| oracle@8 accuracy | 0.6260127931769722 | `[AUDIT]`, `[INT]` |
| greedy accuracy | 0.4494669509594883 | `[AUDIT]`, `[INT]` |
| incumbent selected accuracy | 0.48528784648187634 | `[AUDIT]`, `[INT]` |
| **incumbent selection efficiency** | **0.7752043596730245** | `[AUDIT]`, `[INT]` |
| incumbent candidate AUROC | 0.8855921901711237 | `[AUDIT]`, `[INT]` |
| incumbent per-set | 0.850088 / 0.761905 / 0.722581 | `[AUDIT]` |
| contested stratum denominator | **916** (contested ∧ recoverable) | `[AUDIT]` |
| quantum of sel_eff | 1/1468 = 0.000681 | `[INT]` |

The incumbent is `ckpts/train/lora_verifier_disjoint` — the **clean, disjoint-trained** verifier. The
contaminated `lora_verifier_pooled4` is used nowhere in this round except where a July number is
quoted and explicitly labelled non-comparable.

**Null test.** All five agents reproduced every published cell from the transfer dumps before running
anything: **max absolute deviation 3.5967302447481586e-07**, uniform across `[PAIRHEAD]`, `[SETAWARE]`,
`[CHEAP]`, `[RPW]`, `[INT]`. That deviation is exactly 6-decimal rounding of the published values.
Independently re-run while writing this document via `src/training_methods/genframe_data.py:null_test`
— same value.

**Trainer null test — stricter, and it caught a real error.** Two agents refit the *published head*
end-to-end in their own code on CPU at seed 0 and reproduced **0.7956403269754768** (dev 3.27e-07) and
the deployed fusion **0.8065395095367848** (dev 4.90e-07) `[INT]`, `[CHEAP]`. The integration agent's
**first** attempt omitted the train-µ/σ standardisation step (`fit_hidden_head.py:507`) and produced
**0.788147** — an 0.0075 error, larger than most effects in this round, caught by the test before
anything was built on it `[INT]`.

**Disjointness re-proved, not trusted.** Every arm asserted it in its own code on the md5 of *decoded
RGB pixels*: **3,457 train images vs 528 eval images, intersection 0**, 0 extraction failures in 31,498
train and 8,943 eval rows, both prompt frames `[AUDIT]`, `[INT]`, `[SETAWARE]`, `[CHEAP]`. The real-pairwise
arm asserted it against a **superset** of what the adapter could have seen — 5,229 train images vs 528
— also 0 `[RPW]`.

**Three conventions that are each larger than several effects being chased** `[AUDIT]`:

- **Row order is part of the config.** Identical config, identical seed 0: `order='concat'` gives
  **0.795640**, global-sorted order gives **0.799728** — a +0.0041 swing that flips the guardrail flag.
- **Name the ranker.** `rank_avg` (average ranks for ties) → **0.806540**; `rank_argsort` → **0.798365**.
  The published fusion is `rank_avg`.
- **Pick rule is `np.argmax`, first-index tie-break.** The stored `pick` field disagrees on 26/2345
  items → 0.774523. That ±0.0007 is tie-break sensitivity, not an effect.

And two numerics landmines discovered this round:

- **TF32 is on by default in this container** (NGC 25.09). With it on, the identical config and seed
  gave 0.786785 where CPU gives 0.795640, and 0.774523 where CPU gives 0.750681 — **an arithmetic
  artifact large enough to have manufactured a frame effect on its own** `[CHEAP]`. All numbers here
  were produced with TF32 forced off; residual GPU-vs-CPU deviation at seed 0 is then 0.0020 and 0.0054.
- **CPU thread count changes the SGD trajectory**: the same seed-0 config gives 0.795640 at the
  published thread count and 0.800409 at `torch.set_num_threads(8)` `[PAIRHEAD]`. A "bit-exact" *head*
  reproduction is device- and thread-conditional; the *metric* null test is not, because it reads
  stored scores.

---

## 2. Results — the whole ladder in one place

Selection efficiency on n=2,345 (1,468 recoverable); Δ is a paired item-level bootstrap, nboot=10,000.
Guardrail = never worse than the incumbent on any of the three sets. Cost = **full VLM forward passes
per question over and above the 8 generations**.

### 2.1 Controls and the prior ladder (all pre-dating this round)

| selector | sel_eff | Δ vs incumbent [95% CI] | contested (916) | per-set | guard | cost | source |
|---|---:|---|---:|---|:--:|---:|---|
| greedy (no selection) | — (acc 0.449467) | — | — | — | — | 0 | `[AUDIT]` |
| random pick | 0.676260 | — | — | — | — | 0 | `[AUDIT]` |
| zero-shot alignment, best encoder | 0.660763 | — | — | — | dirty | 0 | `[ALIGN]` |
| generator zero-shot P(Yes) | 0.705722 | −0.069482 [−0.090599, −0.048365] | 0.528384 | 0.811287/0.738095/0.623226 | dirty | 3.81 (der.) | `[HID]` |
| self-consistency (plurality of 8) | 0.713896 | −0.061308 [−0.083787, −0.038828] | 0.541485 | 0.837743/0.730159/0.620645 | dirty | 0 | `[AUDIT]` |
| grader-frame head, CV-selected (L21/last/bce) | 0.750681 | −0.024523 [−0.047003, −0.002044] | 0.600437 | 0.864198/0.785714/0.661935 | dirty | 3.81 (der.) | `[HID]` |
| 39-feature selector, 10-seed ens. | 0.7636 | −0.0116 [−0.0334, +0.0103] | — | — | dirty | 0 | `[FEAT]` |
| **incumbent LoRA verifier — THE BAR** | **0.775204** | — | 0.639738 | 0.850088/0.761905/0.722581 | — | 3.823028 | `[AUDIT]`, `[INT]` |
| best learning-to-rank arm (linear/FULL/anypos) | 0.776567 | +0.001362 [−0.006131, +0.008856] | n/a¹ | 0.844797/0.793651/0.723871 | — | 0 | `[LIST]` |
| generator-frame head, seed 0 | 0.795640 | +0.020436 [−0.001362, +0.041553] | 0.672489 | 0.871252/0.761905/0.745806 | clean | 3.813646 | `[HID]`, `[INT]` |
| **deployed fusion — rank_avg(inc, head seed 0)** | **0.806540** | **+0.031335 [+0.016349, +0.046322]** | 0.689956 | 0.883598/0.801587/0.750968 | clean | 7.636674 | `[AUDIT]`, `[HID]`, `[INT]` |
| oracle@8 | 1.000000 | — | 1.000000 | — | — | — | `[AUDIT]` |

*Cost provenance:* the deployed components' pass counts are measured in `[INT]` (incumbent 3.823028 per
distinct **surface** answer, head 3.813646 per distinct **normalized** answer, total 7.636674) and the
pairwise counts in `[RPW]` (round-robin 13.07 on the HF covered set / 17.02 on the complete pool,
knockout 5.63). Cells marked **(der.)** are arithmetic over those, not separately measured.

¹ `[LIST]` reports a contested stratum on a **different denominator** (its rates are integer multiples
of 1/907, not 1/916), so its contested number is not comparable to this column and is omitted rather
than converted.

### 2.2 This round's four comparative architectures

Every one of them is measured against **two** bars, and the second bar is the one that matters: a
*pointwise* head on the **identical** features, refit in the same harness at the same seed budget.

| # | architecture | sel_eff | Δ vs incumbent 0.775204 | Δ vs pointwise head, same features | Δ vs deployed 0.806540 | contested | per-set | guard | cost | source |
|---|---|---:|---|---|---|---:|---|:--:|---:|---|
| **A** | **Pairwise contrast head** over cached vectors, 12-seed ens., pre-registered `logit_sum` | 0.805177 | **+0.029973 [+0.007493, +0.052452]** SIG | +0.002044 [−0.012943, +0.017711] **n.s.** | below | 0.687773 | 0.876543/**0.746032**/0.762581 | **dirty** | 0 | `[PAIRHEAD]` |
| **B** | **Set-aware head** (centroid/listmax), 10-seed ens., pre-registered | 0.793597 | +0.018392 [−0.004087, +0.040191] **n.s.** | −0.010218 [−0.022480, +0.002044] **n.s., point NEGATIVE** | below | 0.669214 | 0.871252/**0.753968**/0.743226 | **dirty** | 0 (+1 tiny set MLP) | `[SETAWARE]` |
| **C** | **Pool-relative contrast features** H+C+M+Wc+Ws, 10-seed ens., pre-registered | 0.810627 | **+0.035422 [+0.013624, +0.057221]** SIG | +0.004768 [−0.010218, +0.020436] **n.s.** | +0.004087 [−0.014305, +0.022480] n.s. | 0.696507 | 0.871252/0.825397/0.763871 | **clean** | 0 | `[CHEAP]` |
| **D** | **Real A-vs-B forward passes**, clean adapter, HF engine-matched, 1,345 items — Borda | 0.803158 | (vs 0.808421 on same items) **−0.005263 [−0.020000, +0.009474]** n.s. | — | below | 0.641075 | 0.848325/0.777778/0.715953 | **dirty** | **13.07** | `[RPW]` |
| D′ | same, Copeland / round-robin | 0.797895 | −0.010526 [−0.025263, +0.004211] n.s. | — | below | 0.631478 | 0.844797/0.761905/0.712062 | dirty | 13.07 | `[RPW]` |
| D″ | same, deterministic knockout | 0.801053 | −0.007368 [−0.023158, +0.008421] n.s. | — | below | 0.637236 | 0.844797/0.769841/0.719844 | dirty | 5.63 | `[RPW]` |

**Matched pointwise comparators for A, B, C** (this is what kills all three):

| comparator | sel_eff | source |
|---|---:|---|
| 12-seed pointwise ensemble (z-mean), same features as A | 0.803134 | `[PAIRHEAD]` |
| 10-seed pointwise control (h128), same features as B | 0.803815 | `[SETAWARE]` |
| 10-seed pointwise H-only ensemble, same features as C | 0.805858 | `[CHEAP]` |

### 2.3 Fusions and the deployable candidates

| selector | sel_eff | Δ vs incumbent | Δ vs deployed 0.806540 | contested | per-set | guard | cost | source |
|---|---:|---|---|---:|---|:--:|---:|---|
| **rank_avg(inc, 8-seed head ens.) — THE RECOMMENDATION** | **0.810627** | **+0.035422 [+0.020436, +0.050409]** SIG | **+0.004087 [−0.004087, +0.012262] n.s.** | 0.696507 | 0.885362/0.809524/0.756129 | **clean vs both** | 7.64 | `[INT]` |
| rank_avg(inc, 10-seed pointwise h128 ens.) | 0.812670 | +0.037466 [+0.023161, +0.052452] | — | 0.699782 | 0.881834/0.801587/0.763871 | clean | 7.64 | `[SETAWARE]` |
| rank_avg(inc, pair head) | 0.811989 | +0.036785 [+0.022480, +0.051090] | — | 0.698690 | 0.885362/0.793651/0.761290 | clean | 7.64 | `[PAIRHEAD]` |
| rank_avg(inc, pair head, pointwise head) | 0.811989 | +0.036785 [+0.017711, +0.055858] | — | 0.698690 | 0.878307/0.769841/0.770323 | clean | 7.64 | `[PAIRHEAD]` |
| rank_avg(inc, pair head, pointwise) **minus** pairwise member | — | — | pair adds **+0.001362 [−0.009537, +0.011580]** n.s. | — | — | — | — | `[PAIRHEAD]` |
| rank_avg(inc, set-aware head) | 0.803134 | +0.027929 [+0.012943, +0.043597] | −0.003 | 0.684498 | 0.881834/0.809524/0.744516 | clean | 7.64 | `[SETAWARE]` |
| rank_avg(inc, pointwise, set-aware) — adding B **lowers** it | 0.807902 | +0.032698 | 0.812670 → 0.807902 | 0.692140 | 0.869489/0.761905/0.770323 | clean | 7.64 | `[SETAWARE]` |
| rank_avg(inc, real-pairwise Borda), 1,345 items | 0.808421 | **+0.000000 [−0.011579, +0.010526]** | — | 0.650672 | 0.851852/0.761905/0.735409 | dirty | 3.82+13.07 (der.) | `[RPW]`, `[INT]` |
| headline **+ real pairwise** as 4th member, 1,345 items | 0.846316 → **0.822105** | **−0.024211 [−0.040000, −0.008421]** SIG LOSS | — | — | — | — | +13.07 | `[INT]` |
| learned combiner, cross-fitted **on eval** (diagnostic) | 0.799728 | +0.024523 [+0.007493, +0.042234] | −0.010899 [−0.024523, +0.003406] | 0.679039 | 0.867725/0.746032/0.758710 | dirty | 7.64 | `[INT]` |

**A coincidence worth naming so nobody reads it as corroboration.** Four structurally different
selectors land on *exactly* **0.8106267029972752** — that is 1,190 of 1,468 items, and the quantum is
1/1468. They are: the recommendation `[INT]`; architecture C standalone `[CHEAP]`; C's H-only
comparator fused with the incumbent `[INT]`; and the 12-seed pointwise ensemble fused with the
incumbent `[PAIRHEAD]`. Identical integers, different selectors — and their bootstrap CIs against the
incumbent differ accordingly ([+0.013624, +0.057221] for C, [+0.020436, +0.050409] for the
recommendation), because the per-item score vectors are not the same. **Do not report these as
replications of one another.**

---

## 3. Did the clean real-pairwise replication hold? **No. This gates everything.**

The round was built on `[JULY]`: real (not simulated) A-vs-B verdicts gave selection efficiency
**pointwise 0.782609 → knockout 0.849 → round-robin 0.858696, Δ +0.0761 [+0.0362, +0.1159]**, with
+0.0498 [+0.0115, +0.0881] on near-ties, at 28 forward passes per question. That measurement used the
**contaminated `lora_verifier_pooled4`**, **n = 578**, the `ckpts/mcq_gen_verify/` pool, and cap320
(250,880 px) `[RPW]`. Four axes of difference from the current bar; it must never be quoted beside
0.775204.

**What was run to settle it** `[RPW]`. Two engine arms, 66,429 forward passes total, 0 errors:

- **HF + PeftModel with the FULL adapter** (all 192 visual LoRA modules present) — the stack that
  produced the 0.775204 bar. 17,582 ordered forwards over **1,345 items**: all 645 slake, all 200
  vqa_rad, and a **pre-registered** 500-question pathvqa subsample (`rng(0).choice(1500,500)`,
  registered before any HF pathvqa number was computed).
- **vLLM 0.9.0.1 over the complete 2,345-item pool** — all 19,952 distinct pairs, both orders, 39,904
  forwards, plus its **own engine-matched pointwise control** (8,943 forwards).

Prompt held **verbatim** from `src/cascade_methods/pairwise_verifier_score.py`; both orders scored and
averaged for position debias; identical normalized strings pinned to p=0.5 with no GPU call.

**Harness self-test before any conclusion.** Bradley-Terry preferences *simulated* from the incumbent's
own scores reproduce **0.775204 exactly** through Borda, Copeland and knockout `[RPW]`. So any deviation
below is real comparative information, not an aggregation artifact.

**Result — a null, and the ladder does not exist.** On the engine-matched arm the incumbent scores
**0.808421** on those same 1,345 items. Real pairwise: Borda 0.803158 (−0.005263), knockout 0.801053
(−0.007368), Copeland/round-robin 0.797895 (−0.010526) — **every point estimate negative, no CI
excluding zero**. The July ordering was pointwise < knockout < round-robin; here **round-robin is below
knockout**, i.e. *more comparisons make it slightly worse*. The contested stratum gives the identical
verdict: 0.641075 (Borda) / 0.637236 (knockout) / 0.631478 (Copeland) against the incumbent's 0.650672,
all negative, all n.s. `[RPW]` — the denominator there is **n = 521** of the covered items (derived: every
reported contested rate in that arm is an integer multiple of 1/521). Guardrail dirty for every
aggregator — round-robin never wins a single set.

**Two mechanisms, measured** `[RPW]`:

1. **Position bias is large and irreducible.** The first-listed answer wins **58.87%** of comparisons,
   the two orders **disagree on 25.12%** of pairs, and mean |p(o0) − (1−p(o1))| = 0.1705. Averaging
   both orders is what makes the arm even competitive: single-order arms are worse and two cross into
   significance — Copeland(o1 only) −0.016842 [−0.032632, −0.002105] **SIG**.
2. **Comparative discrimination is real but does not convert.** On the 1,733 discordant pairs (exactly
   one candidate correct) the pairwise verdicts prefer the correct answer **73.23%** of the time versus
   the incumbent's pointwise **72.04%** — pairwise is marginally *better* at the binary question, and
   still selects worse. Copeland/Borda discard the calibrated magnitude that argmax-over-p(yes)
   exploits. **The pairwise frame is marginally sharper per comparison and strictly lossier per
   aggregation.**

**Fusion contributes exactly zero.** rank_avg(incumbent, pairwise) = **0.808421**, d = **+0.000000**
[−0.011579, +0.010526] — identical to the incumbent to six decimals `[RPW]`. Compare the deployed
fusion's +0.031335 `[HID]`. Re-aggregated independently by the integration agent from the stored teacher
matrix, landing 1–3 quanta away with the same verdict `[INT]`.

**A consequential infrastructure finding, and it is a positive.** vLLM 0.9.0.1 applies a LoRA to the
**language model only** and silently drops all **192 `visual.*` LoRA modules**. Same adapter, same
prompt, same pixels, scored pointwise: HF gives sel_eff **0.775204** / AUROC **0.885592**; vLLM gives
**0.702997** / **0.760242**; score agreement pearson 0.4711, spearman 0.6241, mean|diff| 0.3680 `[RPW]`.
That is a **−0.072 sel_eff engine artifact**, three times most effects in this round. Against an HF bar
the vLLM pairwise arm looks like a −0.0688 catastrophe; against its **own** engine-matched control it is
+0.0034 [−0.0123, +0.0191] (Copeland) and +0.0082 [−0.0075, +0.0232] (Borda) — **the same null, confirmed
independently on the complete 2,345-item pool.**

> **STANDING RULE.** Never compare a vLLM-scored verifier number to an HF-scored one. Any future vLLM
> verifier arm must ship an engine-matched control.

**Honest limits of this test** `[RPW]`. (i) The HF arm covers 1,345 of 2,345 items; the full HF
round-robin would have cost ~3 GPU-hours and the subsample was pre-registered. (ii) The complete-pool
arm is engine-handicapped and is reported only against its own control. (iii) The prompt was held
verbatim, so this decontaminates the **existing** pairwise mechanism — it does not rule out that some
different pairwise prompt could work. (iv) The July result was **already guardrail-dirty**, losing on
pathvqa_open (0.6538 → 0.6154) `[JULY]`, which is visible in that artifact and was not reported at the
time.

---

## 4. Mechanism — the frame effect is **retired**, and what replaces it

The round's motivating story was: *the same frozen model read in the generator frame wins and in the
grader frame loses, therefore the information is only readable where the model would have produced the
answer.* Published contrast: **0.795640** (generator) vs **0.750681** (grader), "+0.045".

**Both cells are correct and both reproduce bit-exact on CPU** — generator L21/span/bt 0.7956403269754768
vs published 0.795640; grader L21/last/bce 0.7506811989100818 vs published 0.750681 `[CHEAP]`. **The
attribution is what is wrong.** The two cells were fit at *different configurations* — generator
L21/**span**/**bt**, grader L21/**last**/**bce** — at **one seed each**. Frame was confounded with
pooling, objective and seed.

**The matched test** `[CHEAP]`: full 2 frames × 2 poolings × 2 objectives grid, **10 seeds each**,
device-matched, TF32 off. Paired generator-minus-grader:

| pooling | objective | Δ generator − grader | 95% CI |
|---|---|---:|---|
| last | bce | +0.004087 | [−0.013624, +0.021798] |
| last | bt | +0.012943 | [−0.006131, +0.032016] |
| span | bce | −0.002044 | [−0.017030, +0.012943] |
| span | bt | +0.002044 | [−0.014986, +0.019074] |

**All four span zero, and at span pooling the two frames are indistinguishable.** The published +0.045
is pooling + objective + one seed. The claim *"the information is only readable in the frame where the
model would have produced the answer"* **must be withdrawn in that form.**

This was already visible in the 08-04 fusion controls and nobody read it: the *losing* grader head
(0.750681 standalone) fused with the incumbent to **0.799046** (+0.023842 [+0.008174, +0.040191],
guardrail-clean), and the per-benchmark grader head fused to **0.803815** `[HID]`. The standalone frame
gap was 0.0449; the **fused** gap was 0.0075. A 6× shrinkage under fusion is the signature of a
configuration artifact, not a representational one.

**What IS real is a geometric difference, and it is not information loss** `[CHEAP]`:

- **Collapse.** In the grader frame the candidates of one question are nearly the same vector: mean
  within-question cosine of the **raw** hidden states is **0.9518–0.9992** at every layer/pooling,
  versus **0.7366–0.9497** in the generator frame. After standardisation, the within-question share of
  total variance is **0.1047–0.4229** (grader) vs **0.2932–0.6410** (generator) — candidate identity
  occupies **3–5× less** of the grader representation.
- **But separability survives.** The grader frame's L21/last ridge probe scores **0.777248** — the
  **best of all 16 seed-free probe cells**, above every generator cell (best generator: L28/span
  0.776567) and above the incumbent's 0.775204. A standardised readout recovers the small direction
  perfectly. The answer to "collapse or rotation" is **a magnitude collapse that is not a loss of
  linear separability** — which is exactly why whitening or a bigger head is *not* the follow-up.
- **Where it happens.** The grader frame integrates candidate content between layers 14 and 21:
  layer-to-layer CKA in the grader/last stream drops to **0.3461** for 14→21 (vs **0.8458** in
  generator/last), and exactly there its within-question variance share jumps **0.1155 → 0.4105** and
  its probe sel_eff jumps **0.747275 → 0.777248**.
- **Frame is a readout-position effect.** Cross-frame CKA *rises* with depth at span pooling
  (0.5118 → 0.6865 → 0.7810 → 0.7379 over layers 7/14/21/28) and *falls* at last pooling
  (0.4703 → 0.5110 → 0.3717 → 0.2874). The frames converge where the readout is pooled over the answer
  span and diverge where it is a single final token.
- **No short-answer-specific deficit.** On the short stratum (gold ≤ 3 words, n=2,029 items / 1,372
  recoverable) the grader's best cell (span/bce, **0.831633**) is the best cell of **either** frame; the
  long stratum has 96 recoverable items and separates nothing.

**What survives from the 08-04 story, precisely.** The active ingredient is *a trained discriminative
head on frozen hidden states, standardised, fused parameter-free with the incumbent* — not the frame.
The falsification controls still hold and are still what makes the fusion credible: fusing a second
**generative opinion** into the incumbent makes it significantly **worse** — base zero-shot P(Yes)
−0.019755 [−0.035422, −0.004768], self-consistency count −0.019755 [−0.036785, −0.002725], random
−0.040872 [−0.059946, −0.021798] `[HID]`. Three non-head second scores go down, every head goes up.
The lever is the *different computation*, and the frame is not how it is chosen.

**And the mechanism that kills the round's thesis, stated three independent ways.** A cached
per-candidate vector was computed with the other candidates **absent**. A real A-vs-B pass conditions
candidate *A*'s representation on *B*'s text. No function of two independently-computed vectors can
manufacture that conditioning:

1. **Additivity (architecture A).** Any antisymmetric G decomposes uniquely as
   `G[i,j] = (θ_i − θ_j) + Resid`, and the first term *is* a pointwise scorer. Measured: **97.93%** of
   the learned matrix's off-diagonal variance is the additive term; ranking by the **residual alone**
   gives **0.679837** against a random-pick floor of **0.676260** `[PAIRHEAD]`. And it is not a transfer
   failure: refitting the same head **inside** eval with image-grouped folds — a contaminated,
   optimistic bound — gives 0.799728 and stays **97.63%** additive. The non-additive structure is not
   being lost; it is not there. Corroborating: the difference encoding `h_i − h_j`, flagged as the most
   important thing to test, is the **worst** encoding (0.773842, d = −0.029292 [−0.044959, −0.013624]
   vs the pointwise ensemble, a **significant loss**), and its linear degeneracy control `h0`
   (algebraically `w·h_i − w·h_j`, i.e. exactly pointwise) scores 0.772480 — indistinguishable, which
   is also a harness validation.
2. **Set-awareness imports noise (architecture B).** The set path is genuinely used — a context
   ablation with the same trained weights costs **−0.0388** (score each candidate alone) and **−0.0681**
   (pool size preserved, siblings swapped in from other questions), while the pointwise control is
   invariant by construction (0.0000/0.0000), validating the probe. **Using it is what costs.** At
   identical depth, parameter count and objective, **zeroing** the pooled context **improves** DeepSets
   (0.797684 vs 0.795640 ensemble; 0.787875 vs 0.782357 seed-mean), and the loss **grows with pool
   size**: set-aware minus pointwise is −0.0047 at 2–3 distinct candidates, **−0.0336** at 4–5, −0.0199
   at 6–8 — the opposite of a working set mechanism's signature `[SETAWARE]`. The pool is itself a
   random draw of 8 generations; conditioning on which siblings happened to be sampled imports sampling
   noise into a decision that was previously invariant to it.
3. **Pool geometry is not correctness (architecture C).** The 18-feature geometry block used **alone**
   selects **below random**: **0.662807** against a 0.676260 floor. All 18 features used alone land
   between 0.634196 (log-norm) and 0.694142 (max cosine to other candidates) — **7 of 18 at or below
   the random floor**, best +0.018 above it, 0.081 below the incumbent. "Is this candidate typical of
   the pool" is directly falsified as a proxy for correctness. The only pool-relative scalar carrying
   real signal is the trivial vote count (0.713896 — exactly the self-consistency control) `[CHEAP]`.

**Three routes, one conclusion: comparative information is created by joint encoding, and cannot be
recovered post-hoc from separate encodings.** Cheapening a pairwise verifier requires cheaper *forward
passes*, not cleverer heads over cached vectors — and §3 shows there is currently no pairwise win to
cheapen.

---

## 5. The deployable recommendation

**Recommendation: ship the deterministic seed-ensemble. Do not ship any comparative component.**

> `pick = argmax over the 8 slots of rank_avg( incumbent_score , mean-rank over 8 seeds of the frozen
> generator-frame head )`

| | sel_eff | acc | slake / vqa_rad / pathvqa | contested (916) | short ≤3w (1372) | long (96) |
|---|---:|---:|---|---:|---:|---:|
| incumbent | 0.775204 | 0.485288 | 0.850088 / 0.761905 / 0.722581 | 0.639738 | 0.794461 | 0.500000 |
| deployed fusion (published) | 0.806540 | 0.504904 | 0.883598 / 0.801587 / 0.750968 | 0.689956 | 0.825073 | 0.541667 |
| **RECOMMENDATION** | **0.810627** | **0.507463** | **0.885362 / 0.809524 / 0.756129** | **0.696507** | **0.828717** | **0.552083** |

All rows from `[INT]`.

- **vs incumbent**: d = **+0.035422 [+0.020436, +0.050409] SIG**; accuracy +0.022175 [+0.012793,
  +0.031557]; contested +0.056769 [+0.032751, +0.080786]; short-answer +0.034257 [+0.019679, +0.049563].
- **vs the deployed fusion — the comparison that matters**: d = **+0.004087 [−0.004087, +0.012262],
  NOT SIGNIFICANT**; accuracy +0.002559 [−0.002559, +0.007676]; contested +0.006550 [−0.006550,
  +0.019651].
- **Guardrail CLEAN against both**, and it is ≥ the deployed fusion on all three sets (+0.001764 /
  +0.007937 / +0.005161). It is the only arm in the entire round that is clean against both bars.

**Cost: zero extra forward passes over what is already deployed** `[INT]`. Total is **7.636674** full
VLM passes per question beyond the 8 generations — incumbent 3.823028 (one per distinct *surface*
answer; verified, 0 of 8,965 surface groups carry two different scores) plus head 3.813646 (one per
distinct *normalized* answer; candidates deduplicate from 8 to a mean of 3.81, histogram
{1:620, 2:342, 3:284, 4:209, 5:198, 6:173, 7:229, 8:290}). **The marginal cost of the change is 0** —
the same cached 3,584-d vector is scored by 8 tiny MLPs (918,529 params, ~1.8 MFLOP each) instead of 1,
i.e. ~10⁸ FLOP/question against ~10¹² for a single 7B pass. Storage: 8 × ~3.5 MB.

**State the gain correctly or it is an overstatement.** The deployed recipe is a **lottery**: re-running
`rank_avg(incumbent, one-seed head)` at 16 seeds gives mean **0.808200**, sd 0.003365, range
**[0.802452, 0.814033]**, and the published 0.806540 sits at its **37.5th percentile** `[INT]`.
Seed-ensembling is worth ~+0.010 to the head **alone** (0.793767 → 0.799728) but only **+0.001 to
+0.002 after fusion** (k=1 0.808200 → k=8 0.809264 → k=16 0.809946), because the fusion was already
averaging away part of that noise. The other disjoint 8-seed block of the same recipe scores
**0.807902**, so the deployable quantity carries ~0.003 of block-to-block spread. **What is
unambiguously bought is determinism** — a fixed, guardrail-clean artifact instead of a draw from
[0.8025, 0.8140] — at zero cost. Sell it as variance elimination, not as a new mechanism.

**The learned-combiner question is settled negatively** `[INT]`. Cross-fitted **on eval** with
image-disjoint folds — an advantage no deployable version could have — a learned combiner scores
**0.799728**, *below* the parameter-free fusion (−0.010899 [−0.024523, +0.003406]) and guardrail-dirty.
The eval-visible weight sweep peaks at exactly **w = 0.5**, the parameter-free point. There is nothing
for a combiner to learn. Adding self-consistency as a third member → 0.795640 (hurts, as train CV
predicted). Adding real pairwise as a fourth member on its covered items → **0.846316 → 0.822105,
d = −0.024211 [−0.040000, −0.008421], a significant LOSS**, for 13.07 extra VLM passes.

**Named fallback.** If the vqa_rad_open guardrail ever becomes binding, architecture C (pool-relative
contrast features) is the repair: 0.825397 on that set against the incumbent's 0.761905, guardrail-clean,
also zero extra forward passes — but it needs an extra cross-fitted stage and buys nothing pooled
(+0.004768 [−0.010218, +0.020436] over the same head on raw features) `[CHEAP]`.

**What this is NOT** `[INT]`, and the doc must say it: this is a **selection** endpoint on a 2,345-item
pool. It is **not** the open arm's end-to-end accuracy — escalation to the 32B is driven by the
selector's own confidence, so changing the selector changes the escalation set. It is **not** the macro
headline; translating requires re-running `src/cascade_methods/macro_average_headline.py`, which was not
done, and **no macro number is quoted anywhere in this document.** The pools differ too: 2,345 items
here versus the MedEvalKit open cells.

---

## 6. What this implies about the selection limit

**Be careful here. There is a real gain and there is a real wall, and both must be stated.**

**The real gain, and it is not nothing.** The bar moved 0.775204 → 0.806540 (deployed, +0.031335
[+0.016349, +0.046322], guardrail-clean) → 0.810627 (recommended, +0.035422 [+0.020436, +0.050409],
guardrail-clean against both bars), at **zero** forward passes beyond the incumbent's own. In accuracy
terms the open arm's pre-escalation answer goes 0.485288 → 0.504904 → **0.507463**, which closes
**32.85%** of the greedy→oracle gap against the incumbent's 20.29% `[INT]`. That is a genuine, measured,
reproducible improvement over a trained 7B LoRA judge, and it should not be buried under this round's
negatives.

**The real wall.** Counting this round, **twenty-plus distinct approaches** have now been tried:
six cross-family judges, scale, more samples, richer answers (`(choice)(why)`), diverse generation,
Dawid-Skene, bandits, portfolios, logit fusion, slice discovery, shrinkage, TTA, neuro-symbolic gates,
ranking/listwise **objectives** at two levels `[LIST]`, feature-based selectors `[FEAT]`, zero-shot
contrastive alignment `[ALIGN]`, discriminative heads `[HID]`, and now **four comparative
architectures plus a decontaminated replication of the one comparative win the project had**. The
comparative family is the one that was supposed to break the pattern; it did not.

What the accumulated evidence supports, stated at the strength it actually has:

1. **Comparative signal cannot be synthesised from independently-encoded cached per-candidate
   vectors.** This is *established*, by three independent routes with mechanisms (§4): 97.9%
   additivity, set-awareness importing sampling noise, geometry-alone below random. It is a mechanism,
   not an absence of evidence.
2. **The one real joint-encoding comparative implementation in this project is a null on clean
   weights.** Established for *this* prompt, *these* weights, *this* pool, at two engines (§3). It is
   **not** established that no pairwise prompt could work — that is a narrower negative and should be
   quoted as such.
3. **The objective is not the variable, the architecture is not the variable — the per-candidate
   evidence is.** `[LIST]` closed the objective at two levels; `[SETAWARE]` closes set-aware
   architectures over cached vectors; `[CHEAP]` closes pool-relative features. Together they bracket
   the question from three sides.
4. **What remains reachable on this pool is variance, not mechanism.** The single-fit seed range of any
   head is ~0.021 `[INT]`, `[PAIRHEAD]`, `[SETAWARE]`; the deployed fusion recipe's own 16-seed range is
   [0.802452, 0.814033]. **Both are larger than every architectural effect measured this round.** The
   honest reading is that selection on this pool sits at ~**0.80–0.81** and that the last few thousandths
   are seed control, not method.
5. **The guardrail statistic itself is at its resolution limit.** Guardrail cleanliness on this pool is
   a seed coin-flip: across the ten arms measured at 10 seeds each, clean-seed counts run **0, 1, 1, 3,
   3, 3, 6, 6, 7, 0** — set-aware prereg 0/10, pointwise control 1/10, published-bar config 3/10,
   `deepsets_noctx` 6/10, `bce` 7/10 — driven entirely by vqa_rad_open (n=200 items, **126
   recoverable**) `[SETAWARE]`. The published bar's guardrail-clean status did not survive
   seed-averaging. **Single-seed guardrail claims on this endpoint are unreliable** and should stop
   being made.

**And the thing that outranks all of it.** **37.4% of questions have no correct answer anywhere in the
8-sample pool** `[INT]`. Closing the *entire* remaining selection gap (sel_eff → 1.0) would reach
accuracy 0.626013; the remaining selection wall is **0.189373** of sel_eff while the coverage wall is
**~4.5×** that. And more samples do not route around it through the current selector: sel_eff **decays
at −0.076115 per doubling of N** (pooled) `[NSCALE]`. **Generator work outranks verifier work**, and
this round is the strongest evidence yet for that ordering — because it exhausted the verifier side of
the comparative hypothesis without moving the number.

**What would falsify this reading.** A selector that beats 0.810627 by more than +0.010 with ≥10 seeds,
guardrail-clean on all three sets, and pre-registered on train CV. Nothing in twenty attempts has done
it; that is a claim about the evidence, not a proof of impossibility.

---

## 7. Next experiments, ranked

1. **Convert the selection gain into a paper number — re-run the cascade end-to-end.** This is the only
   item that changes anything anyone outside this document reads. The recommendation changes the open
   arm's selector, which changes its confidence, which changes the escalation set, which changes the
   macro headline. Nothing here can be quoted in the paper until `macro_average_headline.py` is re-run
   with it. Cheap, decisive, and currently the blocker on the entire round's value. **Highest priority.**
2. **Freeze and version the 8-seed ensemble as an artifact.** 8 × 3.5 MB of head weights plus the train
   µ/σ vector, checked in beside `ckpts/train/lora_verifier_disjoint`. Right now the deployed selector
   is a *draw* from [0.8025, 0.8140] `[INT]`; freezing it is the entire deliverable of this round and
   costs an afternoon. Ship item 2 with item 1.
3. **Move to the coverage wall.** 37.4% unrecoverable `[INT]` against 0.189 of remaining selection
   `[INT]`, and sel_eff decays −0.076 per doubling `[NSCALE]`, so "more samples" is not the lever —
   **diversity at fixed N** is. Concretely: measure oracle@8 under temperature/nucleus sweeps and under
   prompt-perturbed decoding, with the selector held fixed. The endpoint is oracle@8, not sel_eff, and
   it is a generator experiment. This is where the headroom is.
4. **Fix the guardrail's resolution before running another architecture round.** vqa_rad_open has 126
   recoverable items, and clean-seed counts run 0/10 to 7/10 across arms `[SETAWARE]`. Either enlarge the open
   vqa_rad pool or replace the per-set guardrail with a seed-averaged per-set CI. Until this is done,
   "guardrail-clean" is not a reportable property of a method, and three arms in §2 are labelled with a
   coin flip.
5. **A single-pass joint-encoding selector — one forward pass over the *whole deduplicated candidate
   list* in one context.** §4 establishes that joint encoding is what creates comparative information
   and that caching cannot fake it; §3 establishes that *pairwise* joint encoding at 13 passes/question
   is a null. The untested cell is joint encoding of all K candidates at **1 pass/question**. Two priors
   against it: the pairwise null, and LLM listwise *prompting*'s known position bias (measured here at
   25.12% order disagreement `[RPW]`). Run it only with both orders / a shuffle control and a
   pre-registered config. **Ranked here, not higher, precisely because the evidence is against it.**
6. **Re-examine the July pairwise result's provenance rather than its accuracy.** `[JULY]` reported
   +0.0761 with a CI excluding zero on n=578, and it was already guardrail-dirty on pathvqa_open
   (0.6538 → 0.6154) — visible in the artifact, unreported at the time. The decontamination in §3 says
   the effect is gone; a short write-up of *how* a contaminated-weight, small-n, dirty-guardrail result
   became a load-bearing premise for a whole round is worth more to this project than another
   architecture.
7. **Persist the numerics environment in every artifact.** TF32 state, device, thread count, row order,
   ranker name, seed list. Two of those (TF32, threads) were discovered this round to move sel_eff by
   more than the effects being measured `[CHEAP]`, `[PAIRHEAD]`; two more (row order, ranker) by ~0.004
   and ~0.008 `[AUDIT]`. A "reproduces bit-exact" claim on a *trained* head is meaningless without them.
8. **Retire the teacher matrices as a distillation target.** `realpairwise_teacher_pmatrix_hf_2026-08-05.jsonl`
   and its vLLM twin are worth keeping as a *measurement instrument* (how much comparative signal is
   recoverable from cached vectors is a real question, and §4 answers it at ~2%), but **the teacher is a
   null selector** `[RPW]` — nobody should build a headline on top of it.

**Closed by this round — do not re-propose:** pairwise contrast heads over cached per-candidate vectors
(97.9% additive, `[PAIRHEAD]`); set-aware/listwise **architectures** over cached per-candidate vectors,
across DeepSets, raw-centroid and self-attention forms, 5 capacities, 4 objectives, 10 seeds each
(`[SETAWARE]`); pool-relative geometry features as a correctness proxy (below random alone, `[CHEAP]`);
learned combiners over {incumbent, head} (`[INT]`); and **the framing that generator-frame beats
grader-frame** (`[CHEAP]` — retired, see §4). Real A-vs-B pairwise with **this** prompt and these
weights is closed as a *deployable* component; the general pairwise question is narrowed, not closed.

---

## 8. Corrections this round makes to earlier documents

Per the repository's standing lesson — *corrections must be propagated, not filed in new files* — the
following supersede text in `VERIFIER_ARCHITECTURES_2026-08-04.md` and any doc quoting it.

1. **"The information is in the generator's representations but is only readable in the frame where the
   model would have produced the answer" is WITHDRAWN.** At matched pooling × objective × 10 seeds, all
   four generator-minus-grader contrasts span zero `[CHEAP]`. The published +0.045 was configuration and
   seed. See §4.
2. **"The +0.076 real-pairwise win, shelved for cost" is WITHDRAWN as a live result.** On clean weights,
   this pool, engine matched, it is a null with every point estimate negative `[RPW]`. The July numbers
   remain on record as a measurement on contaminated weights, n=578, a different pool and a different
   image budget — **never to be quoted beside 0.775204**.
3. **The published single-seed cells are seed draws, and two of them are extremes.** The generator head's
   0.795640 sits essentially at its own 12/16-seed mean (0.795413 `[PAIRHEAD]`, 0.793767 `[INT]`) — an
   honest draw — but the *same config* under the set-aware agent's harness has 0.795640 as the **maximum**
   of its 10 seeds (mean 0.792234) `[SETAWARE]`, and the deployed fusion's published 0.806540 sits at the
   **37.5th percentile** of its own 16-seed range `[INT]`. Quote seed-ensembles, not seeds.
4. **"Guardrail-clean" was reported at single-seed resolution and should not have been.** Clean-seed
   counts out of 10 range from **0/10 to 7/10** across the ten arms measured `[SETAWARE]`.
5. **vLLM-scored and HF-scored verifier numbers are not comparable** (−0.072 sel_eff, `[RPW]`). Any
   historical verifier number in this repo whose engine is unrecorded should be treated as unlabelled.

---

## 9. Artifacts and code

**Artifacts** (`results/cascade_methods/artifacts/`): `genframe_cache_audit_2026-08-05.json`,
`verifarch_pairhead_2026-08-04.json`, `verifarch_setaware_2026-08-04.json`,
`verifarch_setaware_cv_preregistration.json`, `verifarch_cheapcontrast_2026-08-04.json` (+
`_cheapcontrast_parts/`), `verifarch_realpairwise_clean_2026-08-04.json`,
`verifarch_realpairwise_hf_2026-08-05.json`, `verifarch_realpairwise_vllm_2026-08-05.json`,
`realpairwise_disjointness_2026-08-05.json`, `realpairwise_teacher_pmatrix_hf_2026-08-05.jsonl`,
`realpairwise_teacher_pmatrix_2026-08-05.jsonl`, `verifarch_integrated_2026-08-04.json` (+
`_integrate_parts/`). Prior-round artifacts cited: `verifarch_hidden_2026-08-04.json`,
`verifarch_hidden_fusion_controls_2026-08-04.json`, `verifarch_hidden_generatorprompt_2026-08-04.json`,
`verifarch_listwise_2026-08-04.json`, `verifarch_features_2026-08-04.json`,
`verifarch_alignment_2026-08-04.json`, `pairwise_verifier_gpu.json`,
`verifier_n_scaling_2026-08-03.json`.

**Code** (`src/training_methods/` unless noted): shared loader `genframe_data.py`; pairwise contrast
head `pairhead_lib.py`, `pairhead_cv.py`, `pairhead_cv2.py`, `fit_pair_head.py`, `pairhead_verdict.py`,
`pointwise_seeds.py`, `pointwise_seeds_gpu.py`; set-aware `verifarch_setaware.py`,
`verifarch_setaware_report.py`; cheap contrast `cheapcontrast.py`, `verifarch_cheapcontrast.py`,
`cheapcontrast_verdict.py`; real pairwise `realpairwise_clean_gpu.py`, `realpairwise_clean_hf.py`,
`realpairwise_pointwise_control.py`, `realpairwise_clean_analyze.py`, `realpairwise_hf_analyze.py`,
`realpairwise_assert_disjoint.py`, `realpairwise_finalize.py`, runners
`runners/run_realpairwise_clean_queue.sh`, `runners/run_realpairwise_hf_pathvqa.sh`; integration
`integrate_{lib,verify,cpuref,prereg,prereg2,eval,finalize}.py`.

**Caches (gitignored):** `feats_hidden/` (4.4 GB, 8 files; eval 8,943 rows / 2,345 questions / 528
images per mode; train 31,498 rows / 6,029 questions / 3,457 images per mode; 0 extraction failures)
`[AUDIT]`. Checkpoints: `ckpts/pairwise_clean/`, `ckpts/train/lora_verifier_disjoint`.

Nothing under `MedEvalKit/`, `MedVLThinker/` or `MedRAG/` was modified by this round (`MedEvalKit/`'s
two local uncommitted edits pre-date it — see `CLAUDE.md` §0). No abstention, deferral or reject-option
mechanism was proposed, built or evaluated anywhere in this round.
