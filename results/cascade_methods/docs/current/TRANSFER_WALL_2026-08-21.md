# The transfer wall: what closed it, what survived, and one correction to a result of my own
**2026-08-21** · every number below names the artifact it came from · no fabricated numbers

> **One-line summary.** The head is a per-domain component. Five independent attempts to make it
> domain-general all failed, a training-free signal we had been discarding beats it on exactly the
> cells it was not trained on, and the one routing idea that survives is a **compute** claim, not an
> accuracy claim — the accuracy version was an artifact of an anticonservative CI I computed myself.

---

## 1. The wall, restated with the new cells

Frozen head, deployed readout (per-seed within-pool `rank_avg`, mean over 8 seeds), judge currency.
Source: `artifacts/free_signal_bakeoff_2026-08-21.json`.

| cell | head trained on it? | greedy | head | oracle@8 |
|---|---|---:|---:|---:|
| pathvqa_open | IN | 0.3427 | **0.3900** | 0.5167 |
| slake_open | IN | 0.7302 | **0.7690** | 0.8791 |
| vqa_rad_open | IN | 0.4900 | 0.4650 | 0.6300 |
| kvasir_x1_open | IN | 0.2849 | **0.3629** | 0.4696 |
| radimagenet_open | OUT | 0.3210 | 0.3295 | 0.5120 |
| omnimed_open | OUT | **0.3885** | 0.3396 | 0.5759 |
| vqamed_open | OUT | **0.0947** | 0.0688 | 0.2102 |

In domain the head is worth +0.039 to +0.078 over greedy. Out of domain it is worth −0.026 to
−0.049 — the 8× sampling spend **buys negative accuracy**.

## 2. Five closed routes

| route | result | artifact |
|---|---|---|
| **capacity** | h32 0.70090 ≈ h1024 0.69742 — irrelevant | `head_curve_bce_2026-08-18.json` |
| **regularisation** | does not rescue it; only data does | `head_reg_2026-08-18.json` |
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
| slake_open | −0.0295 LOSS | **omnimed_open** | **+0.0449 WIN** |
| vqa_rad_open | +0.0000 TIE | **vqamed_open** | **+0.0175 WIN** |
| kvasir_x1_open | −0.0930 LOSS | | |

**The sign flips exactly at the domain boundary, with no exceptions.** A training-free within-pool
count — no answer vocabulary, nothing to overfit — beats a trained 918k-parameter head precisely
where that head has no training data.

**Do not overstate it: out of domain neither beats greedy.** SC merely loses less (−0.004 vs −0.049
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
| **knn_train** | **−0.821** | 0.296 (0.704 sign-flipped) |
| **domclf_maxprob** | **+0.857** | 0.670 |
| vocab_coverage | +0.571 | 0.540 |
| head_spread | −0.429 | 0.517 |
| maha_pca64 | +0.214 | 0.485 |
| sc_entropy | +0.000 | 0.546 |

Mechanically sensible: omnimed, the worst cell, has the largest kNN distance (6.967 vs 2.6–3.3
in-domain) and the lowest domain-classifier confidence (0.767 vs 0.994–0.998).

**Caveats that must travel with these numbers.** With 7 cells, Spearman ±0.85 is p≈0.024
uncorrected and **not significant across six detectors tested**. Within-cell AUROC is only 0.551
[0.456, 0.686], so the signal orders **datasets, not questions**. And `domclf_maxprob`, the best
cell-orderer, **fails as a router** (−0.0134 LOSS vs always-select) — cell ordering does not imply
routing performance.

## 6. ⚠️ CORRECTION TO A RESULT OF MY OWN — the router "WIN" was my CI, not the data

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
