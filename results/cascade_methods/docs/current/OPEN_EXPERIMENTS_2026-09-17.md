# Open experiments — the backlog as of 2026-09-17

**Status: NOTHING HERE IS RUNNING.** This is a parked list, written while Leo finishes reading the
domain guide. Decide what to run *after* the read, not from this file.

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
the FLOP model assumes; detokenisation. **Nothing in any artifact explains it.**

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

### 0.9 Housekeeping: delete the vestigial `Dropout(0.0)`
It is the identity function and the only reason the output layer is indexed `f.3` rather than `f.2`.
Dropout was tested and lost (0.6785 → 0.67692 → 0.67465 at the deployed width). **Needs a checkpoint
migration**, since removing it renames state-dict keys — not a casual edit. Low value, do it when
touching that file for another reason.

---

## Tier 1 — needs GPU

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

---

### 1.3 N=32 on the coverage-limited benchmarks
Already partly in flight — `lingshu7b_sc32T07` extraction was running on GPU 1 for `vqamed_open` on
2026-09-16. The question is whether more samples can rescue a **coverage** failure (VQA-Med:
oracle@8 0.2102 against greedy 0.0913 — most candidate sets contain no correct answer at all). Pairs
naturally with 1.2: if 32 helps only where coverage binds, that is exactly the signal an adaptive
controller should key on.

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

## Suggested order, if we do nothing else

1. **0.1** exact-match re-scoring — it gates how every other result may be described.
2. **0.6** within-question AUROC — near-zero cost, removes a wrong number.
3. **0.2 + 0.3** the cost restatement and the FLOPs-vs-latency gap — one investigation, and it
   decides which axis we claim on.
4. **1.1** the MedGemma judge — the only item that permanently closes a reviewer objection.
5. **1.2** adaptive sampling — the most interesting *result* on the list, and the one most likely to
   be a contribution rather than a correction.
