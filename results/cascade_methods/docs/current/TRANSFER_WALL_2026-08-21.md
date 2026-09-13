# The transfer wall: what closed it, what survived, and one correction to a result of my own
**2026-08-21** · every number below names the artifact it came from · no fabricated numbers

> ## ⚠️ SUPERSEDED IN PART BY GEMeX, 2026-08-22 — READ §10 BEFORE §3, §5 OR §6
> GEMeX (8,000 q chest X-ray, built after this document) **falsifies two claims made below**:
> the self-consistency sign-flip "at the domain boundary, with no exceptions" (§3) and the
> "regime is partly detectable" detector result (§5), which in turn removes the basis for §6's
> router. §10 carries the corrected picture. §1, §2, §4, §7 and §8 stand.

> **One-line summary (rewritten 2026-08-22 after GEMeX).** Five independent attempts to make the
> head domain-general all failed, and it still loses to plain greedy decoding on two of eight cells
> — but it WINS on four, including the one cell where answer-vocabulary memorisation is least
> available, so "per-domain" is too coarse and **we cannot yet say what separates the cells it helps
> from the cells it hurts.** Two things this document originally claimed are now falsified by
> GEMeX and corrected in §10: the self-consistency sign-flip at the domain boundary, and the
> detectability of the regime. The router of §6 was also, separately, an anticonservative CI I
> computed myself; clustered by cell it was already a tie before GEMeX removed its detector.

---

## 1. The wall, restated with the new cells

Frozen head, deployed readout (per-seed within-pool `rank_avg`, mean over 8 seeds), judge currency.
Source: `artifacts/free_signal_bakeoff_2026-08-21.json`.

| cell | head trained on it? | greedy | head | oracle@8 |
|---|---|---:|---:|---:|
| pathvqa_open | IN | 0.3140 | **0.3452** | 0.4808 |
| slake_open | IN | 0.7302 | **0.7690** | 0.8791 |
| vqa_rad_open | IN | 0.4900 | 0.4650 | 0.6300 |
| kvasir_x1_open | IN | 0.2849 | **0.3629** | 0.4696 |
| radimagenet_open | OUT | 0.3210 | 0.3295 | 0.5120 |
| omnimed_open | OUT | **0.5164** | 0.4971 | 0.7007 |
| vqamed_open | OUT | **0.0947** | 0.0688 | 0.2102 |

In domain the head is worth +0.031 to +0.078 over greedy. Out of domain it is worth −0.019 to
−0.026 — the 8× sampling spend **buys negative accuracy**.

## 2. Five closed routes

| route | result | artifact |
|---|---|---|
| **capacity** | h32 0.70090 ≈ h1024 0.69742 — irrelevant | `_head_sweep_journal_reg_s0of1.jsonl`, `_head_sweep_journal_main_s4of6.jsonl` |
| **regularisation** | does not rescue it; only data does | `_head_sweep_journal_reg_s0of1.jsonl` |
| **architecture** | all 6 alternatives transfer WORSE than plain | `head_arch_transfer_2026-08-19.json` |
| **more samples** | self-consistency is flat in N at the random floor | §4 below |
| **routing (accuracy)** | tie once the CI is clustered correctly | `regime_router_2026-08-21.json` |

Architecture, mean head-minus-string-prior over 4 held-out domains — **the plain baseline wins**:

| raw | domadv | poolnorm+raw | poolnorm | pca32 | pca128 | rankonly |
|---:|---:|---:|---:|---:|---:|---:|
| **+0.0734** | +0.0558 | +0.0527 | +0.0409 | +0.0386 | +0.0339 | +0.0302 |

Every targeted fix — within-pool normalisation, PCA compression, rank-only features,
domain-adversarial training — made transfer worse. The literature-motivated hypothesis that
"under distribution shift, structured and compressed features are more robust" is **refuted here**,
and `rankonly`, its strongest form, came last.

## 3. The discarded signal — and the sign flip

`extract_generator_hidden.py` **deduplicates the candidate pool** before storing features:
within-pool string duplication in the feature rows is **0.000**, while the raw dumps
(`ckpt_<cell>_lingshu7b_sc8.jsonl`, field `preds`) keep all 8 samples with their multiplicity.
The generator computed self-consistency, wrote it to disk, and feature extraction threw it away.
The head's entire input has always been `h_span` at layer 21.

Recovered (`free_signal_bakeoff_2026-08-21.json`), SC minus head:

| IN-domain | | OUT-of-domain | |
|---|---:|---|---:|
| pathvqa_open | −0.0640 LOSS | radimagenet_open | −0.0050 TIE |
| slake_open | −0.0295 LOSS | **omnimed_open** | **+0.0190 WIN** |
| vqa_rad_open | +0.0000 TIE | **vqamed_open** | **+0.0175 WIN** |
| kvasir_x1_open | −0.0930 LOSS | | |

**The sign flips exactly at the domain boundary, with no exceptions.** A training-free within-pool
count — no answer vocabulary, nothing to overfit — beats a trained 918k-parameter head precisely
where that head has no training data.

**Do not overstate it: out of domain neither beats greedy.** SC merely loses less (−0.0002 vs −0.0193
on omnimed). This diagnoses the wall; it does not break it.

## 4. Self-consistency does not scale with N

Fixed 225-question set from `ckpt_kvasir_open_lingshu7b_sc16.jsonl` (every one of the 16 samples
judged, so the question set is constant across N — the first cut of this varied the question set
with N and was discarded).

| N | SC acc | oracle@N | random floor |
|---:|---:|---:|---:|
| 2 | 0.0933 | 0.1558 | 0.0961 |
| 4 | 0.0958 | 0.2283 | 0.0945 |
| 8 | 0.0933 | 0.3236 | 0.0945 |
| 16 | 0.0800 | 0.4178 | 0.0953 |

Greedy on the same set: **0.1244**. SC sits at the random floor at every N while oracle climbs from
0.156 to 0.418. **Buying more samples cannot rescue the free signal**, which is why the planned
N=16 generation on the failing cells was cancelled rather than run.

## 5. The regime is partly detectable — dataset-level only

`artifacts/regime_detector_2026-08-21.json`, 7 cells:

| detector | cell Spearman | pooled per-question AUROC |
|---|---:|---:|
| **knn_train** | **−0.679** | 0.309 (0.691 sign-flipped) |
| **domclf_maxprob** | **+0.679** | 0.630 |
| vocab_coverage | +0.464 | 0.614 |
| head_spread | −0.536 | 0.540 |
| maha_pca64 | +0.179 | 0.508 |
| sc_entropy | +0.000 | 0.538 |

Mechanically sensible: omnimed, the worst cell, has the largest kNN distance (6.822 vs 2.6–5.4
in-domain) and the lowest domain-classifier confidence (0.754 vs 0.994–0.998).

**Caveats that must travel with these numbers.** With 7 cells, Spearman ±0.68 is p≈0.094
uncorrected and **not significant across six detectors tested**. Within-cell AUROC is only 0.564
[0.456, 0.686], so the signal orders **datasets, not questions**. And `domclf_maxprob`, the best
cell-orderer, **fails as a router** (−0.0134 LOSS vs always-select) — cell ordering does not imply
routing performance.

## 6. ⚠️ SUPERSEDED — this section reports a SEVEN-CELL RUN that no longer exists

> **Read `AUDIT_2026-09-12.md` §5 instead.** Every number in this section comes from a
> seven-benchmark `regime_router` fit made before GEMeX existed. That artifact was regenerated with
> eight benchmarks and the seven-cell values are **not re-derivable from anything on disk** — the run
> is gone. The current artifact reads pooled routed-vs-greedy **+0.0215 [+0.0181,+0.0250]** i.i.d.
> and **+0.0215 [−0.0087,+0.0587]** cell-clustered, against the +0.0133/+0.0119 quoted below.
> The *methodological* point — that an i.i.d. interval on benchmark-clustered data is
> anticonservative, and that correcting it turns this WIN into a TIE — still stands and is why the
> section is kept. The specific figures do not.

## 6. (superseded) CORRECTION TO A RESULT OF MY OWN — the router "WIN" was my CI, not the data

Leave-one-cell-out router on `knn_train` (`artifacts/regime_router_2026-08-21.json`), macro:

| arm | macro | compute |
|---|---:|---:|
| always-greedy | 0.3789 | 1× |
| always-select | 0.3893 | 8× |
| **routed** | **0.3908** | **4.33×** |

| comparison | question-level i.i.d. (WRONG) | cell-clustered (CORRECT) |
|---|---|---|
| routed vs greedy | +0.0133 [+0.0109, +0.0156] **WIN** | +0.0119 [−0.0032, +0.0297] **TIE** |
| routed vs select | −0.0004 [−0.0043, +0.0035] TIE | +0.0015 [−0.0174, +0.0209] TIE |

The i.i.d. interval is roughly **8× too narrow**. Essentially all variance here is *between* cells —
the select rate swings from 0.07 (omnimed) to 0.96 (slake) and is near-constant within a cell.
This is the same error the clustered-CI audit already fixed elsewhere in this project, reproduced.

**What survives is the compute claim:** the router matches always-select at ~54% of its sampling
cost. That is arithmetic, not statistics. **Both accuracy comparisons are ties**, and at n=7 cells
this experiment cannot establish an accuracy win in either direction.

## 7. Correction to a CLAUDE.md landmine — the segfault is THREAD COUNT, not `_foreach`

The standing note says the AdamW `_foreach` multi-tensor path segfaults and `foreach=False` fixes
it. **`foreach=False` does not fix it** — with it in place the crash simply moves to
`_single_tensor_adam` (`torch/optim/adam.py:535`, traced by `faulthandler` on
`head_finelayer.py`). The reproducible variable is **thread count**: 10 threads dies in ~30 s,
4 threads runs clean, and the same fit in isolation at 10 threads completes 30 epochs in 38.5 s.
So it is load-dependent, not path-dependent. **Every CPU head fit should run at ≤4–6 threads and
be sharded for parallelism rather than threaded.**

## 8. Two infrastructure failures worth not repeating

- **A failed producer deadlocks its consumers.** `run_campaign11_chained.sh` waited 21 hours on
  `gemex_open.json`, an artifact whose producing job had already FAILED. Nothing ran overnight.
  A chain must wait on "produced **or** failed", never on the artifact alone.
- **One corrupt input killed a completed 21,312-image pull.** `pixel_md5` let PIL's
  `OSError: image file is truncated` escape, discarding hours of download. Undecodable images are
  now dropped with a count, not raised.

## 9. What this leaves for the paper

The honest framing is **"a per-domain selector"**, not "a verifier". The open question is no longer
*can it be made general* — five routes say no — but **what a new domain costs**:
`head_newdomain_curve.py` is measuring the crossover k at which the head overtakes greedy on
omnimed and vqamed, with kvasir_x1 as the positive control. `head_domain_scaling.py` is separately
testing whether breadth (k of 7 domains) lifts transfer at all.


---

# 10. GEMeX (2026-08-22) — what it falsifies, and what is left standing

`artifacts/cell_gemex_open_2026-08-19.json`, `free_signal_bakeoff_2026-08-21.json`,
`regime_detector_2026-08-21.json`. 8,000 questions / 3,514 chest X-ray images, subsampled by image
from a 48,274-question PhysioNet build. Templating is the best in the project — 6,236 distinct
questions, 4,315 distinct golds, top-10 gold coverage **12.7%** — so answer-vocabulary memorisation
has the least to work with here of any cell we have.

| arm | accuracy | vs greedy |
|---|---:|---|
| always-7B greedy | 0.3974 | — |
| string prior | 0.3549 | **−0.0425 LOSS** |
| self-consistency | 0.3794 | −0.0180 LOSS |
| **frozen head** | **0.4121** | **+0.0148 [+0.0057, +0.0238] WIN** |
| oracle@8 | 0.5864 | — |

head vs string prior **+0.0573 [+0.0483, +0.0663] WIN**. This is the strongest evidence in the
project that the head performs real verification: on the cell where a counter over answer strings
has least to memorise, the counter **loses to greedy** while the head beats both.

## 10.1 FALSIFIED — the self-consistency sign-flip (§3)

§3 said the SC-minus-head sign flips "exactly at the domain boundary, with no exceptions". GEMeX is
out of training and the **head beats SC by +0.0328 [+0.0238, +0.0419]**. The exception exists.

| cell | domain | sc − head |
|---|---|---|
| pathvqa / slake / kvasir_x1 | IN | −0.0640 / −0.0295 / −0.0930 |
| vqa_rad | IN | +0.0000 |
| radimagenet | OUT | −0.0050 |
| omnimed | OUT | **+0.0190** |
| vqamed | OUT | **+0.0175** |
| **gemex** | **OUT** | **−0.0328** |

The true statement is narrower: **SC beats the head on omnimed and vqamed only.** Everything §3
concluded from a clean boundary must be re-derived; the raw measurements in §3 are unaffected.

## 10.2 FALSIFIED — "the regime is partly detectable" (§5), and §6's router with it

GEMeX is the **most** out-of-distribution cell measured by both surviving detectors — largest kNN
distance to the training rows (7.228, above omnimed's 6.822) and lowest domain-classifier confidence
(0.686, below omnimed's 0.754) — **and the head wins on it.** Distance from the training
distribution does not determine whether selection helps.

| detector | cell Spearman, 7 cells | cell Spearman, 8 cells |
|---|---:|---:|
| knn_train | −0.679 | **−0.548** |
| domclf_maxprob | +0.679 | **+0.548** |

The artifact's verdict is now *"NO detector orders the cells; the regime is not identifiable at
inference from anything tested."* One extra cell was enough — precisely the fragility §5 flagged
when it recorded that ±0.68 at n=7 is p≈0.094 uncorrected and not significant across six detectors.
It did not survive. **§6's router is therefore unsupported even as the compute claim**, because the
detector it routes on no longer orders the cells.

Re-run on all eight cells the router is not merely unsupported but **numerically worse than always
selecting**: macro routed 0.4107 vs always-select 0.4118 vs always-greedy 0.3972, routed-vs-select
−0.0011 [−0.0049,+0.0030] and routed-vs-greedy +0.0136 [−0.0095,+0.0384], both ties. The mechanism
of the failure is legible and damning: GEMeX has the largest kNN distance of any cell, so the router
sends **68% of it to greedy** (select rate 0.32) and throws away the +0.0148 win the head actually
had there. A detector that routes confidently in the wrong direction on the newest cell is worse
than no detector.

## 10.3 What is left standing

- **§1, §2, §4, §7, §8 are unaffected**: the wall itself, the five closed routes, SC's flatness in
  N, the thread-count segfault, the infrastructure failures.
- The head beats greedy on **4 of 8** cells (pathvqa +0.031, slake +0.039, kvasir_x1 +0.078,
  gemex +0.015), ties on 2 (vqa_rad −0.025, radimagenet +0.009) and loses on 2 (omnimed −0.019,
  vqamed −0.026).
- **"Per-domain" is too coarse and OOD-ness does not explain the split — and neither does answer
  kind.** `artifacts/answer_kind_2026-08-22.json` tested the obvious next hypothesis (the head
  loses where answers are short category labels and wins where they are descriptive free text)
  within cells, splitting each cell at its median mean-candidate-length so the cell's identity is
  held fixed. **Refuted.** Longer answers favour the head in 6/8 cells but the mean is only
  +0.0045, two cells go firmly the other way (pathvqa −0.0585 [−0.0971,−0.0175], vqa_rad −0.0910),
  and the cell-level Spearman is +0.238. The premise was also simply wrong on the facts: omnimed's
  mean candidate answer is **3.12 words**, LONGER than pathvqa's 1.99 and slake's 1.74, the two
  cells where the head wins most — short golds do not imply short candidates.
  **We do not currently know what separates the cells selection helps from the cells it hurts.**
  Two hypotheses have now been measured and both failed; do not add a third to a document without
  measuring it.
- The honest deployment statement is still the **price curve** (§9 pointer): ~500 labelled
  in-domain questions before the head does something a counter cannot, and on cells where sampling
  COVERAGE binds (vqamed, oracle@8 0.2102 against greedy 0.0947) no amount of head training helps.
