# Progress — August 5, 2026 (the comparative-verifier round — the premise itself was wrong)

> **Follows `progress_August_04.md`.** One thesis, five agents, seven hours, and a verdict that landed
> on the *premise* rather than on any of the four things built to test it. The thesis: the two things
> that had ever worked on open-text selection — **reading the generator's own frame** and **comparing
> candidates head-to-head** — had never been combined, and if the comparative signal could be computed
> from **cached** per-candidate vectors, the July pairwise win (+0.076 sel_eff) would arrive at *one*
> forward pass per candidate instead of 28. Four architectures were built against that thesis. **All
> four failed to beat a pointwise head on identical features.** And then the fifth arm — a clean
> replication of the pairwise win itself — found that **the win does not exist on decontaminated
> weights**. There was nothing to make cheap. What the day leaves is: a *mechanism* for why cached-vector
> comparison cannot work (three independent routes), the **retirement of the "generator frame" story**
> that motivated the whole round, one **infrastructure finding worth more than the round** (vLLM
> silently drops visual LoRA modules, −0.072 sel_eff), and one free, deterministic, guardrail-clean
> deployable change that is **not statistically distinguishable** from what is already deployed. Every
> number below is sourced to a named `verifarch_*` artifact; nothing is fabricated. **Abstention remains
> permanently out of scope** and appears nowhere.

> **Naming note, stated so the record is exact.** Five artifacts written today carry a **`_2026-08-04`**
> suffix — `verifarch_{setaware,cheapcontrast,pairhead,realpairwise_clean,integrated}_2026-08-04.json`
> — because the agents inherited the previous round's naming convention. Their mtimes are all
> **2026-08-05 15:36–17:18 UTC**, and that is what the timeline below uses. The three files stamped
> `_2026-08-05` (`genframe_cache_audit`, `realpairwise_{hf,vllm}`, `realpairwise_disjointness`) are
> correctly named. **Do not infer a date from a filename in this round.**

---

## 1. Where the thesis came from

Two prior results, both on the same 2,345-question open-text pool (slake_open 645 / vqa_rad_open 200 /
pathvqa_open 1,500; 1,468 recoverable; incumbent `lora_verifier_disjoint` at sel_eff **0.775204**):

1. **The generator frame** (`verifarch_hidden_generatorprompt_2026-08-04.json`, yesterday). A
   discriminative head on the frozen Lingshu-7B's hidden states read under the model's *answering*
   prompt scored **0.795640** standalone and **0.806540** fused with the incumbent — the only positive
   in ~16 consecutive failures. The story attached to it: *the same model in a judging frame loses
   (0.750681), so the information is only readable where the model would have produced the answer.*
2. **Real A-vs-B forward passes** (`pairwise_verifier_gpu.json`, 2026-07-06). Pointwise **0.782609** →
   knockout **0.849** → round-robin **0.858696**, Δ **+0.0761 [+0.0362, +0.1159]**. Shelved because
   round-robin costs 28 forward passes per question.

Nobody had combined them. The arithmetic was seductive: a candidate's generator-frame vector is already
computed for the deployed pointwise head, so all 28 pairs could be 28 *tiny MLP* evaluations. The whole
round was designed around that sentence.

**It rested on two assumptions, and today killed both.**

---

## 2. Ground pass — the shared harness (14:46)

Before anything was built, one agent audited `feats_hidden/` and wrote a shared loader, so five agents
would measure on one denominator. `genframe_cache_audit_2026-08-05.json`,
`src/training_methods/genframe_data.py`.

- **Cache is complete.** Eval **8,943 rows / 2,345 questions / 528 images**, all **18,760** candidate
  slots resolve, 0 rows match no slot, cache `y` agrees with the dump on **18,760/18,760**. Train
  **31,498 rows / 6,029 questions / 3,457 images**, 0 extraction failures.
- **Three facts the briefing had wrong or missing.** Train is 31,498 rows, *not* 15,749 (that is
  per-shard). `radimagenet_open` is **absent** from the train cache — the matched quota
  894+4,973+522+3,975 = 10,364 = `max_train`, so the deficit branch never fired. And **`yesno` is
  all-NaN in `mode='generator'`** — there is no Yes/No readout in that frame; using the column
  silently would have produced garbage.
- **Disjointness re-proved in the audit's own code**: train ∩ eval pixel-md5 = **0** (3,457 vs 528).
- **Strata frozen for everyone**: contested ∧ recoverable = **916**; unanimous ∧ recoverable = **552**,
  scored 1.0 by every selector by construction.

**Three conventions were pinned, and each is larger than several of the day's effects.** Row order is
part of the config (`order='concat'` 0.795640 vs global-sorted 0.799728, and the guardrail flag flips).
`rank_avg` → 0.806540 vs `rank_argsort` → 0.798365 — **always name the ranker**. Pick rule is
first-index `argmax`; the stored `pick` field disagrees on 26/2345 → 0.774523.

**Null test, five independent harnesses, one number:** max absolute deviation
**3.5967302447481586e-07** on every published cell — pure 6-dp rounding. Re-run once more while writing
this entry: same value.

---

## 3. The four architectures (15:36 – 16:29)

Each was pre-registered by image-grouped CV **inside the train split only**, run at ≥10 seeds, and
compared against **two** bars: the incumbent 0.775204, and — the one that matters — a *pointwise* head
on the **identical** features refit in the same harness at the same seed budget.

| arm | sel_eff | vs incumbent | **vs pointwise, same features** | guard | artifact |
|---|---:|---|---|:--:|---|
| **A** pairwise contrast head, 12-seed ens. | 0.805177 | +0.029973 [+0.007493, +0.052452] SIG | **+0.002044 [−0.012943, +0.017711] n.s.** | dirty | `verifarch_pairhead_*` |
| **B** set-aware (centroid/listmax), 10-seed | 0.793597 | +0.018392 [−0.004087, +0.040191] n.s. | **−0.010218 [−0.022480, +0.002044], point NEGATIVE** | dirty | `verifarch_setaware_*` |
| **C** pool-relative contrast features, 10-seed | 0.810627 | +0.035422 [+0.013624, +0.057221] SIG | **+0.004768 [−0.010218, +0.020436] n.s.** | clean | `verifarch_cheapcontrast_*` |
| **D** real A-vs-B, clean adapter, HF, Borda | 0.803158 | **−0.005263** vs 0.808421 on same items, n.s. | — | dirty | `verifarch_realpairwise_*` |

Matched pointwise comparators: 0.803134 (A), 0.803815 (B), 0.805858 (C).

**Three of them are the same negative with three different mechanisms — and each mechanism is measured,
not asserted.**

**A — the learned comparison is 97.93% additive.** Any antisymmetric `G` decomposes uniquely as
`(θ_i − θ_j) + Resid`, and the first term *is* a pointwise scorer. Measured: **97.93%** of the learned
matrix's off-diagonal variance is additive; ranking by the **residual alone** gives **0.679837** against
a random-pick floor of **0.676260**. It is not a transfer failure — refitting *inside* eval with
image-grouped folds (a contaminated optimistic bound) gives 0.799728 and stays **97.63%** additive.
And the pre-registration had already said so: train CV chose `logit_sum` as the aggregator, which is
*exactly* the additive projection `Σ_j G[i,j] = k·θ_i`. **The CV selected the pointwise reading of the
pairwise head without ever seeing eval.** The difference encoding `h_i − h_j` — flagged going in as the
most important thing to test — is the **worst** encoding (0.773842, a significant loss), and its linear
degeneracy control lands where algebra says it must (0.772480), which doubles as a harness validation.

**B — set-awareness imports sampling noise.** The heads are *not* ignoring the pool: a context ablation
with the same trained weights costs **−0.0388** (score each candidate alone) and **−0.0681** (siblings
swapped in from other questions), while the pointwise control is invariant by construction
(0.0000/0.0000), validating the probe. **Using the pool is what costs.** At identical depth, parameter
count and objective, *zeroing* the pooled context **improves** DeepSets, and the loss **grows with pool
size** (−0.0047 at 2–3 distinct candidates, **−0.0336** at 4–5, −0.0199 at 6–8) — the exact opposite of
a working set mechanism. The pool is a random draw of 8 generations; conditioning on which siblings
happened to be sampled is importing noise into a decision that was previously invariant to it.

**C — pool geometry is not correctness.** The 18-feature geometry block used **alone** selects **below
random**: **0.662807** against a 0.676260 floor. All 18 features used alone land between 0.634196 and
0.694142; **7 of 18 at or below the random floor**. Cosine to the pool centroid alone is 0.662807 —
"is this candidate typical of the pool" is directly falsified as a proxy for correctness. The one
pool-relative scalar with real signal is the trivial vote count (0.713896 — exactly the
self-consistency control).

**Three routes, one conclusion:** a cached per-candidate vector was computed with the other candidates
**absent**; a real A-vs-B pass conditions *A*'s representation on *B*'s text; **no function of two
independently-computed vectors can manufacture that conditioning.**

---

## 4. The arm that decided the day — real pairwise, decontaminated (15:12 – 16:25)

`verifarch_realpairwise_{clean_2026-08-04,hf_2026-08-05,vllm_2026-08-05}.json`. Two engine arms,
**66,429 forward passes, 0 errors**.

**Protocol first.** Disjointness re-proved against a **superset** of what the clean adapter could have
seen — 528 eval images vs 5,229 train images, pixel-md5 intersection **0**. Prompt held **verbatim**
from `src/cascade_methods/pairwise_verifier_score.py`; both orders scored and averaged. And a
**harness self-test**: Bradley-Terry preferences *simulated* from the incumbent's own scores reproduce
**0.775204 exactly** through Borda, Copeland and knockout — so any deviation measured is real
comparative information, not an aggregation artifact.

**The result.** On the engine-matched HF arm (1,345 items: all slake, all vqa_rad, a **pre-registered**
500-question pathvqa subsample), the incumbent scores **0.808421** on those same items:

| aggregator | sel_eff | Δ | cost (extra VLM passes/q) |
|---|---:|---|---:|
| Borda | 0.803158 | −0.005263 [−0.020000, +0.009474] n.s. | 13.07 |
| deterministic knockout | 0.801053 | −0.007368 [−0.023158, +0.008421] n.s. | 5.63 |
| Copeland / round-robin | 0.797895 | −0.010526 [−0.025263, +0.004211] n.s. | 13.07 |

**Every point estimate is negative, and the July ladder does not exist** — round-robin is *below*
knockout, i.e. more comparisons make it slightly worse, which is the signature of aggregating noise.
Contested stratum agrees (0.641075 / 0.637236 / 0.631478 vs 0.650672). Guardrail dirty for every
aggregator: round-robin never wins a single set. And **fusion contributes exactly zero** —
`rank_avg(incumbent, pairwise)` = **0.808421**, d = **+0.000000 [−0.011579, +0.010526]**, against the
deployed head fusion's +0.031335.

**Why.** (a) **Position bias is large and irreducible**: the first-listed answer wins **58.87%** of
comparisons, the two orders **disagree on 25.12%** of pairs, mean order gap 0.1705. Averaging both
orders is what makes the arm competitive at all; single-order arms drift into *significant* losses.
(b) **Comparative discrimination is real but does not convert**: on 1,733 discordant pairs the pairwise
verdicts prefer the correct answer **73.23%** of the time versus the incumbent's pointwise **72.04%** —
marginally *better* per comparison, and still selecting worse, because Copeland/Borda discard the
calibrated magnitude that argmax-over-`p(yes)` exploits. **Sharper per comparison, strictly lossier per
aggregation.**

**And an infrastructure finding that is worth more than the round.** **vLLM 0.9.0.1 applies a LoRA to
the language model only and silently drops all 192 `visual.*` LoRA modules.** Same adapter, same prompt,
same pixels, scored pointwise: HF+PeftModel **0.775204** / AUROC **0.885592**; vLLM **0.702997** /
**0.760242**; agreement pearson 0.4711, spearman 0.6241. That is a **−0.072 sel_eff engine artifact** —
three times most effects being chased today, and enough on its own to invert a verdict. It is why the
vLLM arm shipped its **own engine-matched pointwise control**, against which the pairwise arm is
+0.0034 [−0.0123, +0.0191] and +0.0082 [−0.0075, +0.0232] — **the same null, confirmed independently on
the complete 2,345-item pool.**

> **STANDING RULE, effective now:** never compare a vLLM-scored verifier number to an HF-scored one, and
> any future vLLM verifier arm must ship an engine-matched control.

**The July numbers are not refuted; they are re-scoped.** 0.783/0.849/0.859 was measured with the
**contaminated `lora_verifier_pooled4`**, **n=578**, the `ckpts/mcq_gen_verify/` pool, at cap320 — four
axes of difference. It must never be quoted beside 0.775204. It was also **already guardrail-dirty**,
losing on pathvqa_open 0.6538 → 0.6154 — visible in `pairwise_verifier_gpu.json` and unreported at the
time.

---

## 5. The frame effect is retired (16:12)

This is the day's other correction, and it lands on yesterday's headline story rather than on today's.

The published contrast was generator **0.795640** vs grader **0.750681**, "+0.045". **Both cells
reproduce bit-exact on CPU** (dev 3.27e-07 and 1.99e-07) — the numbers are right. **The attribution is
not.** The two cells were fit at *different configurations* — generator L21/**span**/**bt**, grader
L21/**last**/**bce** — at **one seed each**. Frame was confounded with pooling, objective and seed.

Matched 2 frames × 2 poolings × 2 objectives, **10 seeds each**, device-matched, TF32 off. Paired
generator-minus-grader: **+0.0041 [−0.0136, +0.0218]**, **+0.0129 [−0.0061, +0.0320]**,
**−0.0020 [−0.0170, +0.0129]**, **+0.0020 [−0.0150, +0.0191]**. **All four span zero, and at span
pooling the frames are indistinguishable.** The claim *"the information is only readable in the frame
where the model would have produced the answer"* is **withdrawn in that form**.

It was visible yesterday and nobody read it: the *losing* grader head fused with the incumbent to
**0.799046** (+0.023842, clean) and the per-benchmark grader head to **0.803815**. Standalone frame gap
0.0449; **fused** frame gap 0.0075. A 6× shrinkage under fusion is the signature of a configuration
artifact.

**What is real is geometry, and it is not information loss.** In the grader frame the candidates of one
question are nearly the same vector — mean within-question cosine of the **raw** states **0.9518–0.9992**
versus **0.7366–0.9497** in the generator frame, and after standardisation the within-question share of
variance is **0.1047–0.4229** vs **0.2932–0.6410**, i.e. candidate identity occupies **3–5× less** of the
grader representation. **But** the grader frame's L21/last ridge probe scores **0.777248** — the **best
of all 16 probe cells**, above every generator cell and above the incumbent. So it is a **magnitude
collapse that is not a loss of linear separability**, which is precisely why whitening or a bigger head
is *not* the follow-up. Where it happens is localisable: grader/last layer-to-layer CKA drops to
**0.3461** for 14→21 (vs 0.8458 generator/last), and exactly there its variance share jumps
0.1155 → 0.4105 and its probe jumps 0.747275 → 0.777248. Cross-frame CKA *rises* with depth at span
pooling and *falls* at last pooling. **"Frame" is a readout-position effect, not a difference in what
the model knows.**

**What survives from yesterday, precisely.** The active ingredient is a *trained discriminative head on
frozen hidden states, standardised, fused parameter-free with the incumbent* — not the frame. The
falsification controls still hold and are still what makes it credible: fusing a second **generative
opinion** makes the incumbent significantly **worse** (zero-shot P(Yes) −0.019755, self-consistency
−0.019755, random −0.040872).

---

## 6. The integration (17:18) — one deployable answer, honestly sized

`verifarch_integrated_2026-08-04.json`. Pre-registered on train CV only (readout = mean within-pool
rank at k=8; self-consistency member → exclude; grader-frame member → costs +3.81 passes).

> **RECOMMENDATION:** `rank_avg( incumbent , 8-seed rank ensemble of the frozen generator-frame head )`

| | sel_eff | acc | slake / vqa_rad / pathvqa | contested (916) |
|---|---:|---:|---|---:|
| incumbent | 0.775204 | 0.485288 | 0.850088 / 0.761905 / 0.722581 | 0.639738 |
| deployed fusion | 0.806540 | 0.504904 | 0.883598 / 0.801587 / 0.750968 | 0.689956 |
| **recommendation** | **0.810627** | **0.507463** | **0.885362 / 0.809524 / 0.756129** | **0.696507** |

vs incumbent **+0.035422 [+0.020436, +0.050409] SIG**; **vs the deployed fusion +0.004087 [−0.004087,
+0.012262] — NOT SIGNIFICANT**. It is **guardrail-clean against both**, and ≥ the deployed fusion on all
three sets — the only arm in the whole round that is. **Cost: zero extra forward passes** (total
7.636674/question, unchanged; the same cached 3,584-d vector is scored by 8 tiny MLPs instead of 1).

**And it must be sold as variance elimination, not as a mechanism.** The deployed recipe is a
**lottery**: re-run at 16 seeds it gives mean 0.808200, sd 0.003365, range **[0.802452, 0.814033]**, and
the published 0.806540 sits at its **37.5th percentile**. Seed-ensembling is worth ~+0.010 to the head
*alone* but only **+0.001 to +0.002 after fusion**, because the fusion was already averaging away part
of that noise. The other disjoint 8-seed block scores 0.807902. **What is unambiguously bought is
determinism** — a fixed artifact instead of a draw from [0.8025, 0.8140].

**The learned-combiner question closed negatively.** Cross-fitted **on eval** with image-disjoint folds —
an advantage no deployable version could have — a combiner scores **0.799728**, *below* the
parameter-free fusion, and guardrail-dirty. The eval-visible weight sweep peaks at exactly **w = 0.5**,
the parameter-free point. Adding real pairwise as a fourth member on its covered items:
**0.846316 → 0.822105, d = −0.024211 [−0.040000, −0.008421], a significant LOSS**, for 13.07 extra VLM
passes.

**Four selectors land on exactly 0.8106267029972752** (= 1,190 of 1,468; the quantum is 1/1468): the
recommendation, architecture C standalone, C's H-only comparator fused with the incumbent, and the
12-seed pointwise ensemble fused with the incumbent. **Identical integers, different selectors — not
replications of one another.**

---

## 7. Two numerics landmines, both found by accident

1. **TF32 is on by default in this container** (NGC 25.09). With it on, the identical config and seed
   gave **0.786785** where CPU gives 0.795640, and **0.774523** where CPU gives 0.750681. That is
   **larger than every effect in the round** — large enough to have manufactured the frame effect on its
   own. Every number reported today was produced with TF32 forced **off**; residual GPU-vs-CPU deviation
   at seed 0 is then 0.0020 and 0.0054.
2. **CPU thread count changes the SGD trajectory**: the same seed-0 config gives 0.795640 at the
   published thread count and **0.800409** at `torch.set_num_threads(8)`. A "bit-exact" reproduction of
   a *trained* head is device- and thread-conditional; the *metric* null test is not, because it reads
   stored scores.

Related, and it caught a real error before anything was built on it: the integration agent's first
trainer refit omitted the train-µ/σ standardisation (`fit_hidden_head.py:507`) and produced **0.788147**
— an **0.0075** error. The trainer null test rejected it.

---

## 8. What was learned

**The negative is clean, and it is a *premise* negative, which is rarer and more useful.** Most of this
project's negatives kill a method. Today's killed the *reason for building four methods*. The round was
designed to amortise a measured win onto cached features; the win turned out not to exist on clean
weights, and — independently — the cached features turned out to be incapable of carrying comparative
information anyway. **Either finding alone closes the round; having both, from different agents, on
different pools, at different engines, is what makes it safe to write down.**

**The comparative family is now bracketed from three sides.** `verifarch_listwise` closed the ranking
*objective* at two levels; `verifarch_setaware` closes set-aware *architectures* over cached vectors;
`verifarch_cheapcontrast` closes pool-relative *features*. Neither the objective nor the architecture is
the limiting variable — **the per-candidate evidence is.**

**Seed discipline is now the binding methodological constraint, not statistics.** The single-fit seed
range of any head on this pool is ~0.021, and the deployed fusion recipe's own 16-seed range is
[0.8025, 0.8140]. **Both are larger than every architectural effect measured today.** So is guardrail
cleanliness: across ten arms at 10 seeds each, clean-seed counts run **0/10 to 7/10** (set-aware prereg
0, pointwise control 1, published-bar config 3, `deepsets_noctx` 6, `bce` 7), driven entirely by
vqa_rad_open (n=200 items, **126 recoverable**). Single-seed guardrail claims on this endpoint should
stop being made.

**The selection limit, stated at the strength the evidence has.** The bar moved 0.775204 → 0.806540 →
0.810627, guardrail-clean, at zero extra passes — that is real and should not be buried. But it is
variance reduction on top of one mechanism, and twenty-plus distinct approaches now converge on
~**0.80–0.81** for within-question selection on this pool. Meanwhile **37.4% of questions have no
correct answer anywhere in the 8-sample pool**; the remaining selection wall is 0.189373 of sel_eff and
the coverage wall is **~4.5×** that, and sel_eff **decays −0.076115 per doubling of N**
(`verifier_n_scaling_2026-08-03.json`), so more samples do not route around it. **Generator work
outranks verifier work**, and today is the strongest evidence yet for that ordering — because it
exhausted the verifier side of the comparative hypothesis without moving the number.

---

## 9. Standing state and open questions

**Doc written:** `results/cascade_methods/docs/current/COMPARATIVE_VERIFIER_2026-08-05.md` — the round
synthesis, with a source tag on every figure and a §8 listing the five corrections this round makes to
`VERIFIER_ARCHITECTURES_2026-08-04.md`.

**Artifacts produced today:** `genframe_cache_audit_2026-08-05.json`,
`realpairwise_disjointness_2026-08-05.json`, `verifarch_setaware_cv_preregistration.json`,
`verifarch_setaware_2026-08-04.json`, `verifarch_cheapcontrast_2026-08-04.json` (+
`_cheapcontrast_parts/`), `verifarch_realpairwise_vllm_2026-08-05.json`,
`verifarch_realpairwise_hf_2026-08-05.json`, `verifarch_realpairwise_clean_2026-08-04.json`,
`realpairwise_teacher_pmatrix_{,hf_}2026-08-05.jsonl`, `verifarch_pairhead_2026-08-04.json`,
`verifarch_integrated_2026-08-04.json` (+ `_integrate_parts/`).

**Code:** `src/training_methods/genframe_data.py`; `pairhead_{lib,cv,cv2,verdict}.py`,
`fit_pair_head.py`, `pointwise_seeds{,_gpu}.py`; `verifarch_setaware{,_report}.py`;
`cheapcontrast.py`, `verifarch_cheapcontrast.py`, `cheapcontrast_verdict.py`;
`realpairwise_{clean_gpu,clean_hf,pointwise_control,clean_analyze,hf_analyze,assert_disjoint,finalize}.py`;
`integrate_{lib,verify,cpuref,prereg,prereg2,eval,finalize}.py`; runners
`run_realpairwise_clean_queue.sh`, `run_realpairwise_hf_pathvqa.sh`. Checkpoints (gitignored):
`ckpts/pairwise_clean/`. **Nothing under `MedEvalKit/`, `MedVLThinker/` or `MedRAG/` was modified**
(`MedEvalKit/`'s two local uncommitted edits pre-date this session).

**Open questions, in the order they should be answered:**

1. **The round's gain is not a paper number yet.** Changing the open arm's selector changes its
   confidence, which changes the escalation set, which changes the macro headline. `macro_average_headline.py`
   was **not** re-run and **no macro number is quoted anywhere** in today's doc. Until it is re-run,
   0.810627 is a selection-endpoint result on a 2,345-item pool and nothing more.
2. **Freeze the ensemble.** The deployed selector is currently a *draw* from [0.8025, 0.8140]. Eight
   head checkpoints (~3.5 MB each) plus the train µ/σ vector, versioned beside
   `ckpts/train/lora_verifier_disjoint`, is the entire deliverable of the day and costs an afternoon.
3. **The guardrail statistic is below its own resolution.** vqa_rad_open has 126 recoverable items and
   flips the flag on seed noise. Either enlarge that pool or replace the per-set guardrail with a
   seed-averaged per-set CI — otherwise "guardrail-clean" is not a reportable property.
4. **Untracked, as always.** Everything above is untracked; the last commit remains the July-era chain
   plus yesterday's `153eb98`. Committing is still the standing top-priority chore.
5. **How a contaminated, n=578, guardrail-dirty result became a load-bearing premise for a whole round**
   is worth a short write-up of its own. `pairwise_verifier_gpu.json` contained the pathvqa regression
   (0.6538 → 0.6154) the entire time.
6. **The teacher matrices should be kept as an instrument, not a target.**
   `realpairwise_teacher_pmatrix_hf_2026-08-05.jsonl` measures how much comparative signal is
   recoverable from cached vectors (§3 answers it at ~2%), but **the teacher is a null selector** —
   nobody should distil it into a headline.
