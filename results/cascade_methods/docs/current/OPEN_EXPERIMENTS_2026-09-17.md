# Open experiments — the backlog as of 2026-09-17

**Status: nothing here is running yet. One item is decided.**

- **§1.0 — regenerate the open-text pools at full resolution — is DECIDED** (Leo, 2026-09-17:
  *"worth running the inference again at full res since our goal is not speed anymore"*). It is not
  started, and it has a cheap CPU precursor (§0.9) that should run first because it sizes the job.
- Everything else is parked, written while Leo finishes reading the domain guide.

**Read §1.0 before starting anything else**, because regenerating the pools invalidates every cached
feature — some items are worth doing before it, and some are worth deferring until after. The
ordering section at the end says which.

Each item states: what the question is, why it is worth answering, **what it costs**, what it would
change if it came out either way, and where the inputs already are. Items are grouped by cost, then
ordered by value within the group, because the CPU-only tier can all be done without touching a GPU
queue.

Sources: the v2 domain guide (`literature/DOMAIN_GUIDE_2026-09-16.md`), its §8 response log, and the
review round at https://claude.ai/artifact/CFHKwaiy4qoq9fNn6wAiff

---

## Tier 0 — CPU only, inputs already on disk

### 0.1 Re-score the eight-benchmark probe under normalised exact match ⭐ highest value
**Question.** Does the +0.0736 headline survive when correctness is decided by exact match instead of
the Lingshu-32B judge?

**Why it matters more than anything else here.** The headline is **judge currency only**, and the
judge is the same model family as the generator. In August the head-only arm measured judge
**+0.0452 [+0.0294, +0.0606]** but exact match **−0.0068 [−0.0226, +0.0090], negative on all three
benchmarks** (`CHEAP_VERIFIER_ON_7B_2026-08-16.md` §6). Separately, a newly trained verifier gets a
free **+0.006–0.009** under the same-family judge before doing anything useful
(`coadapt_verifier_T04_2026-08-14.json`). Until this is run, every claim we make carries the
qualifier "under a Lingshu-32B judge", and a reviewer will find that qualifier before we do.

**Cost.** CPU, hours. Both label sets are already stored per candidate in the judged dumps.

**What each outcome means.** Survives → the claim becomes currency-independent and the single biggest
objection closes. Shrinks → we have measured our own judge bias and must lead with it; still
publishable, but the headline number changes.

---

### 0.2 Restate the cost claim on one serving path, with actual GFLOPs
**Question.** What does the method actually cost, stated once, consistently?

**Why.** Found in review (guide §2.4.6): the then-vs-now table mixed **three conventions across two
machines** — FLOP-eq charged at the HuggingFace `repeat-8` convention (8.00×), latency and energy
measured on the HuggingFace path (1.99×, 2.95×), while every candidate pool was actually generated
under vLLM. And the deeper point, raised 2026-09-17: **the FLOP ratio and the measured cost disagree
by ~3×**:

| accounting | greedy → best-of-8 | ratio |
|---|---|---:|
| dense-matmul FLOPs (forward tokens) | 5,831 → 6,589 GFLOPs | **1.13×** |
| wall-clock latency (vLLM, batch 1) | 174.0 → 476.6 ms | **2.74×** |
| energy (vLLM, batch 1) | 34.3 → 123.2 J | **3.59×** |

**So "1.13×" is a true statement about arithmetic and a false statement about cost.** Reporting it
alone would be a new error in the opposite direction to the one we just fixed.

**Cost.** Desk work, plus one re-measurement if we want the LoRA arm on the same path (the LoRA
cannot be scored under vLLM — it silently drops all 192 `visual.*` modules, CLAUDE.md §0 — so that
row may have to stay HF and be labelled as such).

**Deliverable.** One table, one serving path, actual GFLOPs beside every ratio, per the standing rule.

---

### 0.3 Why is best-of-8 2.74× the latency when it is 1.13× the FLOPs? 🔍 new, and genuinely open
**Question.** Where does the missing 2.4× go?

**Why it is interesting rather than housekeeping.** If the prefill is shared and the eight
continuations decode in a batch, the GPU should be doing barely more work than greedy — that is what
1.13× says. It is not what the clock says. Mean power also rises **197 W → 258 W**, so best-of-8 runs
the card *harder as well as longer*, which means the extra cost is real work, not idle waiting.
Candidate explanations, none verified: per-step scheduler and sampling overhead paid eight times;
KV-cache fork cost; decode being memory-bandwidth-bound so eight sequences do not batch as cleanly as
the FLOP model assumes; detokenisation. **Nothing in any artifact explains it.** (Commit `999fd68`
proposes the memory-bound-decode explanation; that is a hypothesis consistent with the numbers, not a
measurement — the decode/prefill split below is what would test it.)

**Why it matters beyond curiosity.** Our whole "verification is free" framing rests on a FLOP
argument. If the binding resource is bandwidth or per-step overhead rather than arithmetic, then the
honest cost story is a *latency and energy* story, and the FLOP number is the least relevant of the
three. That changes which axis we claim on (guide §1.6).

**Cost.** CPU + one GPU-hour of profiling. `bestofn_vllm_2026-09-16.json` has the harness; add an
N-sweep (1, 2, 4, 8, 16) and a decode/prefill split.

---

### 0.4 Calibrated score-average vs rank-average
**Question.** Is throwing away the probes' confidence magnitudes costing us accuracy?

**Why.** Raised in review (#4). The 24 probes are on incompatible scales, so we rank within the
candidate set and average ranks — which deliberately discards magnitude, so a probe that is
*certain* cannot outvote three that are lukewarm. The proper middle ground is to **calibrate first**
(z-score or Platt-scale each probe on the training split) and *then* average scores, so magnitudes
are genuinely comparable. **This is in no artifact.** Related: `rank_avg` itself was selected on
evaluation data (`OPENTEXT_CORRECTIONS_2026-08-19.md` §7 item 4), so it is not a settled choice.

**Cost.** CPU, hours. Cached features, frozen probes, no refit needed for the score-average variant.

**Caveat.** The whole tie-break ambiguity band is only +0.0061 macro (`tiebreak_2026-09-13.json`), so
size expectations accordingly — but it is cheap and it closes an obvious reviewer question.

---

### 0.5 Control task and selectivity
**Question.** Is the probe reading a real correctness signal, or does a 918k-parameter model simply
have enough capacity to fit anything?

**Why.** The probing literature's standard hygiene check (Hewitt & Liang 2019, arXiv:1909.03368):
refit the identical probe on **randomised labels** and report **selectivity** = real-task metric −
control-task metric. We have a permutation null on the *features* (13.35 σ) — a different test, it
randomises inputs rather than labels. A probing-literate reviewer will look for selectivity and not
find it.

**Cost.** CPU, a few hours of refits on cached features.

---

### 0.6 Within-question AUROC, or stop quoting AUROC
**Question.** Are our diagnostic AUROCs inflated by question-identity leakage?

**Why.** `OPENTEXT_CORRECTIONS_2026-08-19.md` §7 already flagged that `head_sweep.auroc` is "pooled
across questions, the wrong AUROC for a selection task", and it was never fixed. CASE
(arXiv:2608.17124) names the same problem from the other side and builds a leakage-free measure
around it, reporting that its "decodability" predicts whether hidden-state selection beats majority
voting at r = 0.75. Our *endpoint* metrics are safe — selection is an argmax inside one question —
but any AUROC we quote as evidence is not.

**Cost.** Near zero. Recompute within question, or delete the metric.

**Bonus.** If we implement decodability properly it becomes a *predictor* of where the method will
help — which is the open question §2.5 says we have never answered.

---

### 0.7 Row-capped fit — do the two biggest benchmarks dominate the probe?
**Question.** Does Kvasir-x1 (23,221 rows) drown out SLAKE (635 rows) in a way that hurts the small
benchmarks?

**Why.** Raised in review (#7). Per-dataset reweighting *was* tested and moved the result by
**0.00005** (`H_rwper_ds_bce` 0.68975 vs control 0.68980) — but that tested accuracy, not whether the
probe's learned rule is set by the largest benchmarks. VQA-RAD, with 365 training rows, is one of the
two benchmarks the method loses on.

**Cost.** CPU. Cap every benchmark at a fixed row count and refit.

---

### 0.8 Re-run the sweep winner under the pooled recipe
**Question.** We never revisited the hyperparameters after the biggest change to the method.

**Why.** The sweep's best cell is `R2_ep60_cos` — hidden **1024**, 60 epochs, cosine schedule — at CV
**0.70598 ± 0.00939**, against the deployed hidden-256 / 30-epoch / no-schedule configuration. It was
not chased because the gap is about one standard deviation. But those CV numbers are from the
**four-domain, layer-21 era**, and the method has since changed to pooled training over eight
benchmarks and three layers, with 3.6× the training rows. A wider probe may behave differently with
that much more data.

**Cost.** CPU, and it is a bigger fit (hidden 1024 × 24 probes).

**Caveat.** Pin and record the thread count — the shipped recipe beat its runner-up by +0.0016 and
the thread-count spread alone is 0.0033.

---

### 0.9 Measure image geometry on all eight benchmarks — the precursor to 1.0
**Question.** On how many of our images does the cap320 ceiling actually bind, and how many vision
tokens would fullres cost?

**Why it must come first.** The August resolution sweep measured geometry on the **three original
benchmarks only** — 2,345 of 36,869 questions. The five added in August (RadImageNet, Kvasir-x1,
OmniMedVQA, VQA-Med, GEMeX) are **80 % of the corpus and their image sizes have never been looked
at.** If their images are small, the cap never binds and regenerating them changes nothing; if they
are large, they are where the gain is. Right now we cannot size item 1.0 without this.

Known, for the three that were measured:

| benchmark | n | median px | images above cap320 | tokens @cap320 | @fullres | @native |
|---|---:|---:|---:|---:|---:|---:|
| SLAKE | 645 | 262,144 | 78.3 % | 244.0 | 616.5 | 671.6 |
| VQA-RAD | 200 | 590,850 | 85.0 % | 285.7 | 750.3 | 840.5 |
| PathVQA | 1,500 | 418,176 | 86.3 % | 285.5 | **487.3** | **487.3** |

**Cost.** CPU, minutes. Read image dimensions off disk; no model, no GPU.

---

### 0.10 Housekeeping: delete the vestigial `Dropout(0.0)`
It is the identity function and the only reason the output layer is indexed `f.3` rather than `f.2`.
Dropout was tested and lost (0.6785 → 0.67692 → 0.67465 at the deployed width). **Needs a checkpoint
migration**, since removing it renames state-dict keys — not a casual edit. Low value, do it when
touching that file for another reason.

---

## Tier 1 — needs GPU

### 1.0 ✅ DECIDED — regenerate the open-text pools at full resolution
**Leo, 2026-09-17: "worth running the inference again at full res since our goal is not speed
anymore."** This is a decision, not a candidate. What follows is scope, cost and the one thing that
would invalidate the result if we get it wrong.

#### The finding that prompted it

`cap320` is an inherited default from the **visual-token-pruning era** — the ladder
`{fullres:1, cap640:2, cap320:4, cap160:8, cap80:16}` divides `HIGH_PX = 1280×28×28`, and the
checkpoints it was written against are literally `ckpts/gate_7b_prune/cap320`. CLAUDE.md §9.5 files
it as *"the chosen **cheap-leg** operating point"* — a cascade-era rationale for making the cheap
model cheap so escalation to the 32B paid off. **There is no 32B leg any more.** The 32B is only the
judge.

Nobody is choosing it now. `src/labeling/run_openvqa.py:65` sets `--cap default="cap320"`, and the
September runners **do not pass `--cap` at all** — e.g. `auto_gpu1_wave51.json` generating
`gemex_open` and `kvasir_x1_open` at N=16 inherits it silently. Every pool across all eight
benchmarks, N=8 and N=16, is at 320 vision tokens by default.

**And it costs accuracy** (`resolution_sweep_2026-08-13.json`, 2,345 questions, generator-only,
verifier held fixed):

| cap | vision tokens | oracle@8 | selected | sel_eff | GFLOPs/candidate |
|---|---:|---:|---:|---:|---:|
| cap80 | 69.0 | 0.5925 | 0.4659 | 0.7863 | 1,909 |
| **cap320 — current** | 274.1 | 0.6154 | 0.4829 | 0.7847 | **5,693** |
| native (MedEvalKit default) | 568.1 | **0.6320** | **0.4878** | 0.7719 | 11,186 |

Native over cap320: **+0.0166 oracle@8, +0.0049 selected.** And on plain greedy decoding, with no
sampling involved: **judge 0.460554 → 0.486994 (+0.0264)** and **exact match 0.455011 → 0.472495
(+0.0175)**. A free +0.026 on the baseline is larger than several effects this project has spent
GPU-weeks on.

#### Use `--cap fullres`, not the MedEvalKit default

Three reasons, and this is the main design call in the item:

1. **It captures nearly all the benefit.** Fullres (1,003,520 px) reaches 616.5 of native's 671.6
   tokens on SLAKE (91.8 %), 750.3 of 840.5 on VQA-RAD (89.3 %), and on **PathVQA it *is* native** —
   487.3 either way, because no PathVQA image exceeds 1,003,520 px.
2. **It kills the train/deploy mismatch.** `extract_generator_hidden.py` already extracts the probe's
   features at `HIGH_PX = 1,003,520`. Generating at cap320 while extracting at fullres means the probe
   is trained on features from a resolution the generator never ran at. Generating at fullres aligns
   them for the first time.
3. **It is cheaper than native**, and the sweep's headline complaint — that the pipeline runs at
   *three* different resolutions — collapses to two.

⚠️ **The honest caveat: fullres has no accuracy point.** The table above measures `cap320` and
`native`; `fullres` was measured for **geometry only**. Choosing it interpolates. If we want the
number rather than the inference, add a fullres arm on the 2,345-question set first — that is cheap
and it is the difference between a measured claim and an assumed one.

#### Cost

**About 1.97× the generator compute** at native, somewhat less at fullres — *not* the 50× the pixel
ratio suggests, because the cap is a **ceiling**, not a resize: an image already under it is
untouched. Per candidate, 5,693 → 11,186 GFLOPs.

Two further cost items that are easy to forget and are probably larger than the generation itself:

- **Re-judging.** New resolution produces new answer strings, and judge labels are cached by
  `(ds, idx, normalised answer)`. Reuse will be partial at best — the sweep's null test N3 found that
  merely re-running the *same* config in a different serving config reproduced only 96.9 % of answer
  strings, and changing resolution will churn far more. Budget for re-judging most of
  ~36,869 × ~3.7 distinct candidates on a 32B.
- **Re-extraction and refit.** Every hidden-state feature cache is invalidated, and the probe must be
  refit from scratch. That is the full pipeline, not a patch.

#### ⚠️ The one thing that would invalidate this

**Regenerate a matched cap320 control in the same session.** This project's standing caveat is that
re-running an arm under a different serving configuration moves cells by **±0.008** — larger than the
effect we are chasing on `selected`. Comparing a freshly-generated fullres arm against the *stored*
cap320 pools would confound resolution with serving-configuration drift, and we would not be able to
tell them apart afterwards. The comparison must be fullres-vs-cap320 **both generated in the same
session, same vLLM build, same seeds.** This has bitten the project before; do not skip it.

#### Suggested sequence

1. **0.9 first** — image geometry on all eight benchmarks (CPU, minutes). If the five new benchmarks'
   images are mostly under the cap, the gain is concentrated in the three originals and the job
   shrinks dramatically.
2. Add a **fullres accuracy arm on the 2,345-question set**, against a same-session cap320 control, so
   the operating point is measured rather than interpolated.
3. Only then regenerate all eight at fullres, with the matched control, and re-judge, re-extract, refit.

---

### 1.1 Re-judge with MedGemma-27B — a judge from a different family ⭐ closes a reviewer objection
**Question.** Does the probe's gain survive a judge that is not a Lingshu?

**Why.** Our labels come from Lingshu-32B judging Lingshu-7B — a same-family judge, which the
LLM-as-a-judge literature flags for self-preference bias (guide §3.8). This is the single best answer
to that objection, and **the model is already downloaded**.

| candidate | family | as an independence test |
|---|---|---|
| **MedGemma-27B-it** | Gemma 3 + SigLIP | **best** — different LM, different vision tower, medically trained |
| InternVL3-38B | InternVL | independent, but shares the known `tp=2` NCCL hang (CLAUDE.md §8) |
| Qwen2.5-VL-32B-Instruct-AWQ | Qwen2.5-VL | weak — Lingshu *is* a Qwen2.5-VL finetune |

**Design.** Re-judge **one fixed candidate set** — GEMeX, the least memorisable benchmark and our
cleanest verifier win — and report the gain in **three currencies**: Lingshu judge, MedGemma judge,
exact match. Do not regenerate anything; re-judging stored candidates is the whole point.

**What each outcome means.** Survives all three → the objection closes permanently. Shrinks under the
cross-family judge → we have found a real bias and get to report it first. Both are publishable; the
current state, one same-family judge, is the only one that is not.

**Cost.** GPU. 8,000 questions × 9 candidates on a 27B. Check the tp=2 hang before committing.

---

### 1.2 Adaptive / dynamic sampling, using the controller that already exists ⭐ Leo's idea, and it has precedent
**Question.** Can we get the N=16 gain without paying N=16 everywhere?

**Why.** The 8 → 16 benefit is wildly uneven (`coverage_sc16_ci_ALL_2026-09-16.json`): OmniMedVQA
+0.0462, GEMeX +0.0359, RadImageNet +0.0329, Kvasir-x1 +0.0309 are all significant wins, while
PathVQA, SLAKE, VQA-RAD and VQA-Med are flat ties. A fixed 16 pays double on four benchmarks to buy
nothing.

**Two things make this better than it looks.**
- **The machinery exists.** The Weitzman "Pandora's box" optimal-stopping controller was built for
  this arm and measured at iso-accuracy for **11.74 against 16.0 FLOP-eq** — a 27 % cut — at 4.37–6.63
  mean draws instead of a fixed 8 (`weitzman_T04_2026-08-15.json`,
  `src/cascade_methods/weitzman_T04.py`). **It has never been run against the pooled probe.**
- **The stopping signal is free.** Weitzman needs a per-candidate value estimate to decide whether
  another draw is worth its price; the probe already produces exactly that, at no extra forward pass.

**⚠️ The thing that would kill it.** No detector we have tested can order benchmarks at inference
time, and the regime router that tried to route on OOD distance was **worse than always-selecting**
because it confidently routed the wrong way on the newest benchmark (guide §2.5 hole 3). Adaptive-N
keyed off a **within-question** signal — the probe's own scores as candidates arrive — is a different
and more defensible thing than a router keyed off **benchmark identity**. Respect that distinction or
this repeats a known failure.

**Cost.** Mostly CPU if it reuses the existing N=16 pools; GPU only if we want N=32 headroom.

**Start with:** the four benchmarks that gained, ceiling 16, against the pooled probe.

**Update 2026-09-18 — the design rule.** 1.3 (below) and the per-benchmark temperature table
(`head_temp_ensemble_2026-08-30.json`) agree: extra diversity pays only where the probe already has
selection skill. Benchmarks where it selects well prefer T = 1.0 and keep gaining at N = 32; the two
it loses prefer T = 0.2–0.4 and gain almost nothing from more samples. Key both N and T on a
within-question selection-skill signal, never on low accuracy. Guide §2.4.5 and §5.8b.

---

### 1.3 ✅ DONE (2026-09-17) — N=32 on the coverage-limited benchmarks
The question was whether more samples can rescue a **coverage** failure (VQA-Med: oracle@8 0.2102
against greedy 0.0913 — most candidate sets contain no correct answer at all).

**Answer: they rescue coverage but not accuracy, unless the probe can select.** Real N = 8/16/32
pools, shipped pooled probe, held-out halves, judge currency (`budget_conversion_sc32_2026-09-17.json`,
commit `999fd68`): VQA-Med's oracle rises 0.1987 → 0.3475 and verifier − greedy goes only
−0.0033 → +0.0044 (1.7 % of the headroom converted); RadImageNet goes +0.1215 → +0.1932 over greedy.
So the premise of the original framing ("if 32 helps only where coverage binds") came out the other
way round: 32 helps where **selection skill** exists. That is the signal 1.2 should key on. Guide
§2.4.5.

---

## What is NOT on this list, deliberately

- **Anything abstention-shaped** — reject option, defer-to-human, selective prediction as a method.
  Permanently out of scope (CLAUDE.md rule 6). Uncertainty *scoring* is fine; refusing to answer is
  not.
- **Chasing a better probe architecture.** Seven variants were tested and all six alternatives
  transferred worse than plain (`head_arch_transfer_2026-08-19.json`). The binding limits are the
  candidate set and identifiability, not the ranker.
- **More breadth in training data.** Leave-one-benchmark-out is **−0.0056**; a benchmark's own
  training half is **+0.0485**. Breadth alone buys nothing.

---

## Suggested order

**1.0 is decided and reorders the rest**, because regenerating the pools invalidates every cached
feature. Anything that reads the current caches should either run *before* the regeneration or be
deliberately deferred until *after* it — running it in between wastes the work.

1. **0.9** image geometry, all eight benchmarks — CPU, minutes, and it sizes 1.0. Do this first.
2. **0.1** exact-match re-scoring — it gates how every other result may be described, and it runs on
   the *current* pools. Worth doing now rather than waiting, so we have the dual-currency baseline to
   compare the regenerated arm against.
3. **0.6** within-question AUROC — near-zero cost, removes a wrong number.
4. **1.0** the fullres regeneration, staged as in its own section: measured fullres arm on the
   2,345-question set with a same-session cap320 control, *then* all eight.
5. **0.2 + 0.3** the cost restatement and the FLOPs-vs-latency gap. Deliberately after 1.0 — the
   whole cost table changes when the generator's resolution changes, so restating it first means
   doing it twice.
6. **1.1** the MedGemma judge — the only item that permanently closes a reviewer objection. Re-judging
   is already required by 1.0, so **fold the cross-family judge into that pass**: if we are paying to
   re-label the pools anyway, label them with both judges and get item 1.1 nearly free.
7. **0.4, 0.5, 0.7, 0.8** the probe-side experiments — all refit on the new features after 1.0.
8. **1.2** adaptive sampling — the most interesting *result* on the list, and the one most likely to
   be a contribution rather than a correction. Needs the new pools to be worth doing once.

**The one piece of leverage worth noticing:** items 1.0 and 1.1 both require a re-judging pass over
the whole corpus. Doing them as one pass — regenerate at fullres, then label every candidate with
*both* Lingshu-32B and MedGemma-27B — costs barely more than 1.0 alone and delivers the cross-family
judge result as a by-product.
