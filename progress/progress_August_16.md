# Progress — 2026-08-16

**The round the objective changed.** The project stopped trying to match always-32B-direct. The new
baseline is **always-7B** (macro-8 0.5971, 1.0 FLOP-eq/question) and the new claim shape is "a small
trained verifier improves a 7B medical VLM by +X on N of 8 cells, at Y× the 7B's own compute". Three
builds ran against that objective; this entry records the assembly, the adversarial re-verification, and
the verdict.

Doc: `results/cascade_methods/docs/current/CHEAP_VERIFIER_ON_7B_2026-08-16.md`.

---

## 1. Verdict

**2 of 8 cells CI-clean under the 32B judge, 1 of 8 under normalised exact match, at 4.56× the 7B's
compute on the three cells the method touches and 2.33× macro-8** — down from 16.73× / 6.90× at the
start of the day. Y is not ~1, but the cost problem is now mostly solved and **the accuracy claim is the
thin part**: macro-8 is +0.0173 judge and **+0.0014 EM**.

---

## 2. What the day actually produced

Two structural defects were removed, and neither cost any accuracy.

- **BUILD 1 — the head is free.** The generator-frame head reads layer-21 `h_span` of the *unmodified
  base* model, which the generator already computed while sampling. It was being recomputed with a
  separate teacher-forced pass: 3.813646 passes/question. Capturing it out of `model.generate()`'s own
  per-step hidden states removes that entirely (6.893 → 2.07e-05 FLOP-eq).
- **BUILD 2 — the verifier's image prefix is shared.** `cheapleg_score_open.py:125` re-ran the whole
  vision tower and whole prefill once per candidate to score strings averaging 19.8 tokens. One shared
  prefill + N short tails: 7.465 → 2.186 FLOP-eq, −70.7%.
- **The costing itself was wrong in both directions.** vLLM shares the LM prefill across
  `SamplingParams(n=8)`, so 8 samples cost a **measured 2.370** FLOP-eq, not the as-charged 8.0 — while
  the head and verifier were *under*-charged at their real (fullres) geometry. The brief's "~11×" was
  never right; the honest starting point was **16.73×** per open question.

Full refactor vs the shipped fusion: **judge −0.001706 [−0.007676, +0.004264] TIE, EM +0.001706
[−0.004691, +0.008102] TIE**, 216/2345 picks changed. 3.67× cheaper for nothing.

---

## 3. The adversarial pass, and the one thing it caught

Everything was recomputed from the raw caches by `src/cascade_methods/central_table_verify.py`; the three
build artifacts were cross-checked, not trusted. Frozen null test **3.5967e-07**. Identity
`selected = oracle@8 × sel_eff` ≤ 1.11e-16 on every arm. My re-derivation of the verifier FLOP model
reproduced BUILD 2's 7.4653992685576815 / 2.186187155120831 to **deviation 0.0**. Permutation null (the
frozen head fed row-permuted features) sits **13.35 σ** below the real arm.

**The catch: `verifier_restructure_2026-08-16.json` has a row labelled `THE_PRIZE`** — head-only, head
captured free, vision shared via image embeds, **1.2027 FLOP-eq/open question, macro-8 accuracy
0.606173**. It is the most attractive number of the day and it is **judge-only**. The artifact reports
`vs_greedy7b_judge` with **no EM counterpart**. Computed here: head-only captured at cap320 is judge
**+0.045203 [+0.029424, +0.060554] WIN** and EM **−0.006823 [−0.022601, +0.008955]** with **all three
cells negative**; its EM `selected` (0.448188) is **below the always-7B EM baseline (0.455011)**.

A single-currency read would have shipped a method that is worse than greedy under exact match. This is
the paraphrase-drift exposure CLAUDE.md §0 already warns about, arriving through a new door — not a
newly *trained* verifier this time, but a newly *cheapened* one. **The fusion is the only selector that
survives dual currency**, which is why the right lever was making its head half free rather than deleting
its LoRA half.

---

## 4. The central table (always-7B baseline)

Best-of-8, T=0.7, frozen rank-fusion selector = prefix-shared LoRA verifier + free captured head.

| cell | n | 7B greedy (j/em) | 7B+verifier | Δ judge | Δ EM |
|---|---:|---|---|---|---|
| SLAKE_open | 645 | 0.7364 / 0.7628 | 0.7721 / 0.7736 | **+0.0357 [+0.0093, +0.0605] WIN** | +0.0109 [−0.0155, +0.0357] TIE |
| VQA_RAD_open | 200 | 0.4650 / 0.4250 | 0.5000 / 0.4000 | +0.0350 [−0.0100, +0.0850] TIE | −0.0250 [−0.0700, +0.0200] TIE |
| PATH_VQA_open | 1500 | 0.3240 / 0.3267 | 0.3920 / 0.3520 | **+0.0680 [+0.0500, +0.0853] WIN** | **+0.0253 [+0.0073, +0.0433] WIN** |
| pooled | 2345 | 0.4495 / 0.4550 | 0.5058 / 0.4721 | **+0.0563 [+0.0422, +0.0704] WIN** | **+0.0171 [+0.0030, +0.0311] WIN** |
| MACRO-8 | | 0.5970868 | 0.6144191 | **+0.017332** | **+0.001398** |

The other 5 cells are multiple-choice-as-presented and carry no open-text candidate pool.

---

## 5. Cost, and where it now sits

| configuration | open q | macro-8 |
|---|---:|---:|
| always-7B | 1.000 | 1.000 |
| deployed at round start | 16.728 | 6.898 |
| **this round (N=8, T=0.7)** | **4.556** | **2.334** |
| + T=0.4 at N=4 (PROJECTED) | 3.781 | 2.043 |
| + vision-embeds fix (accuracy NOT re-measured) | 3.389 | 1.896 |
| *control: prefix caching OFF* | 22.603 | 9.101 |

Re-derived here and matching BUILD 2 to 0.0:
`verifier FLOP-eq(n) = 1.9993 + 0.04887 × n_distinct`. **91.5% of the verifier is now the one shared
prefill and it does not shrink with N** — the whole remaining cost is one image+question prefill at
1,003,520 px, **4× the generator's own 250,880 px**, because no verifier script ever had a resolution
flag. Dropping to 250,880 is a CI-clean LOSS in both currencies on all three cells (judge −0.0102, EM
−0.0107) and is confounded with a train/inference mismatch. **501,760 px is a measured TIE** and is the
untested free rung.

---

## 6. Hyperparameters folded in

- **T = 0.4 is the peak in both currencies** and needs no retraining. T=0.4 at **N=4** is
  parity-or-better vs deployed T=0.7 at N=8 in both currencies (judge +0.00159 TIE, EM +0.00825 WIN) at
  **half** the generation FLOPs. Biggest remaining cost lever.
  ⚠️ **NOT MEASURED: T=0.4 together with the free head and the prefix-shared verifier.** The captured
  features and prefix scores exist only for the T=0.7 pools; the `T=0.4 at N=4` cost row is labelled
  PROJECTED for that reason.
- **Coarser veto binning** recorded but **out of scope** — the veto decides whether to call the 32B, so
  it cannot appear in a 7B-only claim. Kept for the paper's cascade section.

## 7. Two free non-verifier wins (reported separately, not folded into the verifier claim)

- **PATH_VQA_closed, prompt only:** dropping MedEvalKit's `"Please output 'yes' or 'no'"` is
  **+0.038370 judge / +0.038073 EM** vs a matched in-session fullres control, at **1.0 FLOP-eq**.
  Mechanistic: the instruction induces a **+0.069 yes-bias** against a 0.5402 gold base rate; removing it
  takes bias to +0.005 and specificity 0.7503 → 0.8655. Judge and EM agree to 0.0003 — not a grading
  artifact.
- **PMC_VQA self-consistency:** +0.0100 harness / +0.0132 letter-EM, training-free, on a **6,000-item
  subset** of the 33,430-item cell.

Best combined 7B-side system: **macro-8 0.620465 (+0.023379) at 2.505 FLOP-eq**.

---

## 8. Refuted this round

Re-asking SLAKE_closed / VQA_RAD_closed / PATH_VQA_closed as open text added **0 of 3 cells**. The
reformat **never changed the candidate set** — distinct answer strings per cell were VQA-RAD 5 (over
2,008 slots), PathVQA 9 (over 26,896), SLAKE 54, against 3,919 on the open cells; on PathVQA the open
prompt *narrows* the pool. **Candidate diversity, not candidate provenance, is what gates a verifier, and
the prompt does not control diversity on a binary question.** The claim stays at 3 cells.

---

## 9. Housekeeping

- Doc: `results/cascade_methods/docs/current/CHEAP_VERIFIER_ON_7B_2026-08-16.md`.
- New code: `src/cascade_methods/central_table_verify.py` (adversarial re-verification + table
  assembly). Artifact: `results/cascade_methods/artifacts/central_table_2026-08-16.json`.
- Round artifacts verified: `free_head_2026-08-16.json`, `shared_prefix_verifier_2026-08-16.json`,
  `closed_as_open_2026-08-16.json`, `verifier_restructure_2026-08-16.json`, plus the settled
  `decoding_ladder_cold_2026-08-14.json` and `veto_binning_2026-08-15.json`.
- **No GPU job ran in the assembly pass** — it is CPU numpy over stored caches. Both A100s were idle
  throughout and nothing was killed. `MedEvalKit/` untouched. `freeze_selector.py` **not** run. No visual
  LoRA scored under vLLM.
- ⚠️ Standing, unchanged: `ckpts/`, `feats_hidden/`, `feats_free/` and `logs/` have **zero tracked
  files**. `feats_free/` (496 MB of captured layer-21 states) is new today and is what BUILD 1's result
  is measured from; a `git push` does not protect it. Top-priority chore.

---

## 10. What this means for the direction

The **cost** half of the new objective is largely done and is mechanistic: two real defects removed for
3.67× at a TIE, plus a 3.38× costing error corrected. Remaining cost is one clearly-identified,
still-reducible term.

The **accuracy** half is thin. Under exact match the method is worth **+0.0014 macro-8** and is CI-clean
on **1 of 8 cells**. The paper should lead with the cost mechanism and report accuracy in both
currencies; leading with the judge number would not survive review, and the `THE_PRIZE` row is a live
example of how easily that happens.
