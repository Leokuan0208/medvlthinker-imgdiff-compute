# The transfer wall: what closed it, what survived, and one correction to a result of my own
**2026-08-21** · every number below names the artifact it came from · no fabricated numbers

> ## ⚠️ SUPERSEDED IN PART BY GEMeX, 2026-08-22 — READ §10 BEFORE §3, §5 OR §6
> GEMeX (8,000 q chest X-ray, built after this document) **falsifies two claims made below**:
> the self-consistency sign-flip "at the domain boundary, with no exceptions" (§3) and the
> "regime is partly detectable" detector result (§5), which in turn removes the basis for §6's
> router. §10 carries the corrected picture. §1, §2, §4, §7 and §8 stand.
>
> ## ⚠️ AND MORE BROADLY — `AUDIT_2026-09-12.md` §5: **treat §1, §3, §5, §6 and §10 as superseded**
> The banner above is insufficient and has been since 2026-09-12. `AUDIT_2026-09-12.md` §5 states
> plainly: *"TRANSFER_WALL_2026-08-21.md's banner is insufficient… Treat §1, §3, §5, §6 and §10 as
> superseded by this file."* That correction was never applied to this banner; it is applied now
> (2026-09-20). §10.3's own line "§1, §2, §4, §7, §8 are unaffected" is written against the older,
> narrower list and should be read with this one.
>
> ## ⚠️ 2026-09-20 — `AUDIT_2026-09-18.md` supersedes this file on three further points
> 1. **The judge is MedVLThinker-32B**, not Lingshu-32B (`src/labeling/run_judge.py:21`; no runner
>    overrides it). Every "32B judge" in this document means that model. It is text-only and sees
>    the gold.
> 2. **§14 (MedGemma, the "third generator from a different LM family") is WITHDRAWN** — see the
>    banner on that section.
> 3. **§12 (Qwen) stands numerically but is not a cross-family replication** — see the banner there.

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

---

## 11. The readout is not order-invariant (2026-09-13)

Found while chasing a **two-question** disagreement between `free_signal_bakeoff.py` and
`head_temperature_sweep.py` on `slake_open`: frozen head **0.768992** vs **0.772093**, a gap of
exactly 2/645. Both scripts compute the same quantity from the same frozen selector.

**Cause.** They read two different feature caches for the same benchmark —
`generator_eval_s{0,1}of2` filtered to slake, versus the per-benchmark `generator_eval_slake_open`.
The two hold an **identical multiset** of `(idx, answer, label)` (1,313 rows both; verified), but
**219 of the 645 questions list their candidates in a different row order**. `rank_avg` produces
integer ranks averaged over heads, so it ties exactly; `np.argmax` breaks a tie by taking the
first row. Which cache the analysis happened to open therefore decided the answer on the tied
questions. This is the "feature row order" landmine in CLAUDE.md §0 appearing in a *reported
number* rather than in a fit.

**Size of it** (`artifacts/tiebreak_2026-09-13.json`, all 8 benchmarks, frozen selector, T=0.7):

| | |
|---|---:|
| questions whose pick is decided by a tie | **2.62%** (0.6% slake → 3.5% omnimed) |
| ambiguity band — every tie broken best vs worst | **+0.0061** macro |
| spread across four deterministic tie-break rules | **+0.0010** macro |
| best rule (`selfcons`) minus status quo (`first_row`) | **+0.00018** |

Rules compared, all with a deterministic inner fallback: `lexical` (lowest normalised answer
string — uses no signal), `selfcons` (most-sampled candidate in the pool), `longest` (a verbosity
control), against `first_row` (the status quo). Per-benchmark clustered CIs on the best rule minus
`first_row` span zero on every benchmark; on four of them the best rule *is* `first_row`.

**Conclusion — a reproducibility fix, not an accuracy one.** No tie-break rule is worth adopting
for accuracy: the whole ambiguity is +0.0061 and no rule captures a significant share of it. But
`first_row` is not a rule, it is whichever cache was read, and that is what made two artifacts
disagree. The fix applied is the minimal one: **`free_signal_bakeoff.py` now reads the
per-benchmark cache for `slake_open` and `vqa_rad_open`** like every other analysis, so there is
one cache per benchmark and the ordering question does not arise. The shipped readout is
unchanged; the +0.0736 headline does not move.

**Standing caveat this adds.** Any two analyses that read the same rows from different caches can
differ by up to the ambiguity band (+0.0061 macro, and more on a single small benchmark) without
either being wrong. Quote the cache, not just the benchmark.

---

## 12. Pooled training replicates on a second generator, at the same size (2026-09-13)

> ⚠️ **2026-09-20 — the numbers in this section are verified exact, but the framing needs two
> additions** (`AUDIT_2026-09-18.md` §3, `replication_currency_2026-09-20.json`).
> **(a) It is not a cross-family replication.** Lingshu-7B *is* a Qwen2.5-VL-7B fine-tune, so this
> compares two training recipes over one language model. With MedGemma withdrawn (§14), **there is
> currently no valid cross-family replication in this project.**
> **(b) An independent refit reproduces `head_final_stack_qwen_2026-09-13.json` bitwise on all
> eight benchmarks, and Qwen is a WIN in every currency** — judge **+0.0814 [+0.0708, +0.0924]**,
> lenient EM **+0.0276**, strict EM **+0.0187**, token-F1 **+0.0317** (macro over the same eight
> held-out halves). Under EM only 3/8 benchmarks are individually significant, Qwen's greedy is
> weak on PathVQA (0.0715) and VQA-Med (0.0133) — though benchmarks with greedy < 0.10 contribute
> only 5.3 % of the macro — and after best-of-8 Qwen is still below plain Lingshu greedy on 4/8.
> **Qwen is the stronger of the two generators in every currency, which is worth stating: the
> currency collapse Lingshu shows under exact match does not repeat here.**

The largest lever we have — retraining the probe on the training half of all eight benchmarks
instead of the four July domains — was measured only on Lingshu. Running the **identical
implementation** on Qwen2.5-VL-7B (`head_final_stack.py --generator qwen`, artifact
`head_final_stack_qwen_2026-09-13.json`):

| arm | Lingshu-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| four-domain, single layer | +0.0182 (6/8) | +0.0227 (6/8) |
| pooled, single layer | +0.0729 (6/8) | +0.0814 (**8/8**) |
| pooled + layer ensemble | **+0.0736** (6/8) | **+0.0820** (**8/8**) |
| + self-consistency feature | +0.0720 (7/8) | +0.0783 (8/8) |
| **the pooled lever** (ensemble − four-domain) | **+0.0554** | **+0.0594** |

The lever is the same size on both generators, and on Qwen the pooled probe beats greedy on
**every one of the eight benchmarks** — where on Lingshu it loses on VQA-RAD and VQA-Med. Pooled
training is a property of the method, not of Lingshu or of medical finetuning.

**This corrects the previous Qwen number as a side effect.** `head_second_generator.py` built its
training set by concatenating four caches named `generator_train_qwen_{kvasir_open,
pathvqa_open_train, slake_open_train, vqa_rad_open_train}` — but each of those holds an
**overlapping mixture of all four domains**, not the one in its name (two are byte-identical,
1,365,111,910 bytes each). The same `(ds, idx, candidate)` entered training up to four times;
`train_rows: 222,154` is that inflation against a deduplicated **145,085**. Worth stating plainly:
the duplication was **depressing** the result, not inflating it — deduplicated, the macro goes
**+0.0788 → +0.0820**. The fix was to delete the sibling script's role rather than repair it:
`head_final_stack.py` is now parameterised over the generator, so one implementation produces both
rows of the table above.

**Read the small differences in that table against §13, not against zero.**

## 13. The reproducibility floor of the fit itself (2026-09-13)

> **⚠️ CORRECTED THE SAME DAY, BEFORE ANYTHING WAS BUILT ON IT.** The first version of this section
> claimed the fit is nondeterministic because of multithreaded float reduction order, on the
> strength of **two** runs that disagreed. Four more replicates settled it the other way: **five
> post-refactor runs at four threads are BITWISE IDENTICAL** — every arm, every benchmark, every
> digit. The fit is *deterministic*. The claim below is what the six runs actually support; the
> retracted version is kept in the commit history, not restated here.

Running `head_final_stack.py` six times on identical inputs — 112,770 pooled rows, 31,439 original,
59 leaking rows dropped, `--seeds 5` — gives **two** distinct answers, not six:

| arm | five runs (post-refactor) | one run (pre-refactor) | gap |
|---|---:|---:|---:|
| four-domain, single layer | +0.020079 | +0.018239 | +0.001840 |
| pooled, single layer | +0.071169 | +0.072925 | −0.001756 |
| pooled + ensemble | +0.070747 | **+0.073607** | −0.002861 |
| + self-consistency | +0.073946 | +0.071972 | +0.001974 |

The five agree **bitwise**; the sixth is the run that produced the shipped **+0.0736** headline,
made earlier the same day with `head_final_stack.py` as it stood *before* it was parameterised over
the generator. Two single-threaded runs are also bitwise identical to each other. So there is no
run-to-run randomness to measure: **something differs between the two code paths or their
invocation, and it moves the macro by up to 0.0029.** Every per-benchmark value differs, so it is
not one benchmark misbehaving.

**What this does and does not change.**

- The **direction** of the earlier caution stands, for a different reason. The shipped recipe was
  chosen over its runner-up by +0.0016, and on five of six runs `pooled_ens_sc` is the better arm,
  not `pooled_ens`. That choice is **not** robust and must not be reported as a win.
- The **magnitude** is real: ±0.0029 separates two code paths that were meant to be identical, and
  until the cause is found, no arm comparison below ~0.003 macro should be quoted from either.
- It does **not** license calling anything random. The fit reproduces exactly; a number from it is
  reproducible *given the same code and invocation*, which is the stronger position.
- The large effects are untouched — pooled training (+0.0554 Lingshu, +0.0594 Qwen) and the headline
  over greedy are ~20× this gap.

**It is the thread count.** Same code, same data, same seeds, one arm (`pooled_singlelayer`), only
the thread count varied:

| threads | macro | reproducible at that count? |
|---:|---:|---|
| 1 | +0.072615 | bitwise, over 2 runs |
| 2 | +0.074448 | — |
| 4 | +0.071169 | **bitwise, over 5 runs** |
| 8 | +0.071660 | — |
| *the pre-refactor run that shipped* | *+0.072925* | *thread count not recorded* |

**The spread across thread counts is 0.0033 — larger than the 0.0029 gap being explained, and it
contains the shipped run's value.** So the macro is a *deterministic function of the invocation*,
and the invocation was not written down. CLAUDE.md prices a thread-count change at +0.0048
elsewhere in the project; on this endpoint it is 0.0033, the same phenomenon.

**The refactor is innocent — confirmed, not assumed.** The pre-refactor code restored from git and
run at four threads is **bitwise identical to the current code at four threads**, on every arm and
every benchmark. So nothing about parameterising the script over the generator touched the
numerics, and the shipped run's +0.072925 is explained entirely by an invocation that was never
recorded.

Note what this does **not** excuse: within any single run all four arms are fitted at one thread
count, so an arm comparison *inside* a run is fair. What is not fair is comparing an arm from one
run against an arm from another run fitted at a different thread count — which is exactly how the
shipped recipe came to be preferred over its runner-up.

**The rule this produces.** Pin and record the thread count for anything that will be compared or
frozen. `head_final_stack.py` now stamps every artifact with `argv`, `--threads`,
`torch.get_num_threads()`, the OMP/MKL environment, torch and numpy versions and the git sha,
because the whole of this investigation was only necessary because an earlier run recorded none of
it.

**The methodological lesson, which is the part that generalises.** I published a causal claim
("threading is nondeterministic") from two samples, and a single-threaded control that was
consistent with it but did not discriminate between hypotheses. Four more samples reversed it. Two
runs cannot separate *nondeterminism* from *two deterministic code paths*; only replication can.

---

## 14. A third generator, from a different language-model family (2026-09-13)

> ## ⛔ WITHDRAWN 2026-09-20 — THIS ENTIRE SECTION RESTS ON BROKEN CANDIDATE SETS
> Verified independently by the 2026-09-18 audit (`AUDIT_2026-09-18.md` §3, `replication.md`,
> `replication_currency_2026-09-20.json`): **MedGemma-4b-it's *sampled* candidates are degenerate.**
> - **Every** sampled candidate hits `gen_tokens = 64`, the cap, on **every** benchmark — while
>   MedGemma's *greedy* decode stops at 4–10 tokens and Lingshu/Qwen samples hit the cap ≤ 0.64 %
>   of the time. The strings look like `"Lungs\n\n\nmodel\nLungs\n\nmodel\n…"`: the sampled arm did
>   not honour `<end_of_turn>` (`src/labeling/run_openvqa.py:180` passes no `stop_token_ids`).
> - **58.5 % of candidates are chat-template garbage; 92.6 % of questions contain at least one.**
>   The probe's picks average 16.9 words against greedy's 2.6.
> - On the questions whose whole candidate set is clean the judge gain falls **+0.0436 → +0.0096**;
>   in strict exact match the arm is **−0.0751, 0/8 positive**.
> - Separately, the "+0.0481, 8/8" figure that circulated is the **`pooled_singlelayer`** arm; the
>   **shipped** `pooled_ens` recipe gives **+0.0447, 7/8**, so the two generators were not compared
>   under the same recipe either.
>
> **Consequence: there is currently NO valid cross-family replication of the probe verifier.** The
> claim at the end of this section — "not a property of Lingshu, of medical finetuning, or of the
> Qwen language model" — **does not hold today.** The section is kept unrewritten as the record.
> It can only be revived by regenerating MedGemma's pools (with a 10-question stop-token check
> first), re-judging, re-extracting and refitting (`AUDIT_2026-09-18.md` §9 item 5).

§12's Qwen replication is weaker than it reads: **Lingshu-7B is a Qwen2.5-VL finetune**, so
Lingshu-vs-Qwen compares two *training recipes over one language model*. InternVL3-8B would not
have fixed that — a load probe confirms its LM is Qwen2.5-7B (hidden 3584). **MedGemma-4b-it**
does: Gemma 3 (hidden 2560, 34 layers) plus a SigLIP tower, medically trained.

Generated, judged with the same 32B judge, extracted and fitted through the same
`head_final_stack.py`. Ensemble layers **pre-registered before any probe was fitted**: Lingshu and
Qwen are both 28-layer so `[18,20,22]` transferred between them with no choice to make, but on 34
layers those indices sit at relative depth 0.529/0.588/0.647 against Lingshu's 0.643/0.714/0.786,
so the **depth-matched `[22,24,27]` is the primary** and `[18,20,22]` a declared secondary.

### The first reading was a data-volume artifact

| | training rows | macro over these 4 |
|---|---:|---:|
| MedGemma | 20,102 | **+0.0248** |
| Lingshu, *full* protocol | 112,770 | +0.0433 ⚠️ **does not reproduce — see note** |
| Qwen, *full* protocol | 145,085 | +0.0547 |

> ⚠️ **The Lingshu row does not reproduce from the artifact (flagged 2026-09-20; value left as
> printed rather than silently changed).** The equal-weight mean of `pooled_ens_minus_greedy` over
> the four benchmarks in `head_final_stack_PVFIXED_2026-09-13.json` is
> (0.052988 + 0.054545 − 0.041237 + 0.125498) / 4 = **+0.0479**, a 0.0046 gap from the +0.0433
> printed here. The same computation reproduces the Qwen row (+0.054689 → +0.0547) and MedGemma's
> +0.0248 **exactly**, so it is this row specifically. `pooled_singlelayer` (+0.0487) and
> `pooled_ens_sc` (+0.0424) do not match +0.0433 either. §13's own ±0.0029–0.0033 reproducibility
> floor is smaller than the gap. **Re-derive before quoting.**

MedGemma has no dedicated train-domain caches and only four benchmarks, so its probe saw **5.6×
less training data**. Family and data volume were confounded, and "three times weaker on Gemma"
would have been the wrong conclusion. Running the other two generators through **MedGemma's exact
protocol** — no train-domain cache, the same four benchmarks, trained only on their by-image train
halves — separates them:

| benchmark | MedGemma | Lingshu (matched) | Qwen (matched) |
|---|---:|---:|---:|
| PathVQA | +0.0246 | +0.0320 | +0.0302 |
| SLAKE | +0.0182 | +0.0152 | +0.0333 |
| VQA-RAD | −0.0103 | −0.0206 | −0.0619 |
| RadImageNet | +0.0667 | +0.1235 | +0.0697 |
| **macro** | **+0.0248** | **+0.0375** | **+0.0178** |
| training rows | 20,102 | 13,304 | 14,436 |
| distinct candidates / question | 6.58 | 4.36 | 4.73 |

**MedGemma lands between the two same-family generators.** The gap in the first table was the
training set, not the architecture. Note the row counts are *not* equalised — the protocol is
matched, but generators differ in how many distinct candidates they produce per question, and
MedGemma produces the most (6.58), so if anything this favours it slightly. All three lose on
VQA-RAD (n=97) under this matched protocol (Qwen −0.0619, `head_final_stack_qwen_matched_2026-09-13.json`).
*Corrected 2026-09-28 (CHECK_2026-09-28.md C5b):* it does **not** lose on every protocol — Qwen under the
full protocol gains **+0.0412** on VQA-RAD (pooled_ens, `head_final_stack_qwen_2026-09-13.json`).

**The pre-registered layer choice was also right, and can be said so because it was declared
first:** the depth-matched primary gives **+0.0248** against the absolute-matched secondary's
**+0.0185**.

### What the claim now is

The probe verifier is **not a property of Lingshu, of medical finetuning, or of the Qwen language
model**. On a different LM family, a different vision tower and a different pretraining corpus, the
same recipe produces an effect inside the range spanned by the two same-family generators.

**Caveat that travels with it:** four benchmarks, not eight, and a 4B generator whose absolute
accuracy is much lower on PathVQA (greedy 6.6% in judge currency against Lingshu's 31.4% — verified
to be real capability, not strict scoring: the judge rescues 85 exact-match misses and rejects 92
exact-match hits that the `contains` fallback had wrongly allowed).
