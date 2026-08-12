# Progress — August 10, 2026 (round 1 against always-32B-direct — four attacks, no win, and my own premise was wrong)

> **Follows `progress_August_05.md`.** The instruction was blunt: *"the tie with 32B is not good enough,
> keep trying different methods, keep performing research loops until we get a positive result."* Target:
> beat **always-32B-direct** on the 8-cell macro (Variant B, equal weight, 1/8 each) with a CI excluding
> zero. Two scouts mapped the headroom, four pre-registered attacks went at it, and **0/4 won — three
> ties and one clean loss.** The round's real products are elsewhere: **two corrections to the strategic
> arithmetic I had written the previous turn** (one of them retires a formula this project had been
> reasoning from for weeks), **a reproducibility caveat larger than the entire published delta**, and
> **one robust cost result** — equal accuracy for a 1.49× compute cut. Every number below names its
> artifact. Abstention appears nowhere.

> **Bookkeeping note.** All artifacts are stamped `_2026-08-10` and were written 12:22–15:35 UTC on the
> 10th. The commit (`0183dd7`) landed **2026-08-11 14:25 UTC** because the round's synthesis agent was
> killed by a session limit after the four attacks had written their files. **This diary and
> `results/cascade_methods/docs/current/BEAT32B_ROUND_2026-08-10.md` were reconstructed from the
> artifacts on the 11th**, not written live. Nothing was recomputed by hand; where my circulating summary
> and an artifact disagree, §8 records it and the artifact wins.

---

## 1. The bar, stated exactly, before anything was built

Clean (decontaminated `lora_verifier_disjoint`) verifier, macro over 8 cells, n = 42,224. Source:
`cascade_selector_rerun_2026-08-05.json:per_arm.disjoint.macro_acc`, re-verified twice today —
`headroom_percell_2026-08-10.json:null_tests` reproduces all 63 per-cell fields to **0.0**, and
`cost_floor_2026-08-10.json:null_tests.N1` to **3.56e-05** (the published file's 4-dp rounding).

| system | macro |
|---|---:|
| always-7B | 0.5971 |
| always-32B-reasoning | 0.5974 |
| **always-32B-direct** | **0.6567 ← THE BAR** |
| oracle-mode-32B | 0.6573 |
| compute-lean | 0.6443 (−0.0124, a LOSS) |
| **shipped accuracy-max** | **0.6575 → +0.0008 [−0.0022, +0.0037] TIE**, at 1.740× direct as-charged |

**A significant win needs macro Δ ≥ +0.0029 — the CI half-width — i.e. a summed per-cell gain of
≈ +0.0236** (`headroom_percell:macro_sensitivity.what_a_win_requires`). Every attack was pre-registered
against that number and nothing else.

---

## 2. Scout A (12:36) — the ceiling is not the problem, identifiability is

`headroom_percell_2026-08-10.json`, `src/cascade_methods/headroom_percell.py`, zero GPU, zero new
inference.

A router between 7B-direct and 32B-direct can beat always-32B-direct on a cell by **at most
p10 = P(7B right ∧ 32B wrong)** — and that is an identity, not a loose bound: the item-level 2-way
oracle is exactly `1 − P(both wrong)`, whose gain over direct *equals* p10.

| cell | n | p10 | best keep-7B AUROC among disagreements | cross-fit keep-7B gain |
|---|---:|---:|---:|---|
| PMC_VQA | 33,430 | **.1143** | .5884 margin / .5806 conf | **+0.0105 [+0.0074, +0.0136] SIG** |
| SLAKE_closed | 836 | .0490 | .6391 | +0.0048 n.s. |
| VQA_RAD_closed | 251 | .0359 | .6420 | −0.0159 n.s. |
| PATH_VQA_closed | 3,362 | .0413 | .5737 | −0.0006 n.s. |
| MedXpertQA-MM | 2,000 | **.1160** | **.4877 = CHANCE** | −0.0035 n.s. |
| SLAKE_open | 645 | .0465 | .4655 | 0.0000 n.s. |
| VQA_RAD_open | 200 | .0500 | .3946 | 0.0000 n.s. |
| PATH_VQA_open | 1,500 | .0760 | .6345 | +0.0067 n.s. |

**Σ p10 = 0.529 ⇒ +0.0661 macro if perfectly identified. Realised today: +0.00083. A 1.3% conversion.**
The headroom is 22× what a win needs and **none of it is reachable by anything cheap.** MedXpert has the
largest p10 and its keep-7B AUROC is chance from both signals; its 7B/32B item oracle (0.4225) sits
close to the independent-errors floor (0.4879).

**Two facts that reframed the whole round:**

- **On four of the five MCQ cells — 50% of the macro weight — the shipped method IS always-32B-direct,
  exactly.** The certified veto certifies **zero** items on SLAKE-cl, VQA-RAD-cl, PathVQA-cl and
  MedXpert (veto rate 0.0000): no 7B-confidence bin anywhere has a Wilson lower bound on 7B precision
  that reaches the 32B's accuracy in that bin. Extending the shipped lever to all four is worth
  **+0.0000** macro.
- **Nothing already measured can be rearranged into a win.** The eval-visible best-per-cell reassignment
  of the three shipped operating points plus the baseline reaches **+0.00277**; the best cross-fit
  keep-7B rule per cell sums to **+0.00275** — both against a **+0.0029** bar, and **both are already
  cheating** (the per-cell winner chosen on the eval).

The entire current +0.0008 is **PMC_VQA's certified veto (+0.0095 cell → +0.00119 macro) plus
PathVQA-open (+0.0087 → +0.00108)**, minus **−0.00144** of drag from the other two open cells.
Leave-one-out range [−0.0004, +0.0024]: **PMC is load-bearing, VQA-RAD-open holds it back.**

---

## 3. Scout B (12:22–12:30) — and here my own arithmetic died

`coverage_diagnosis_2026-08-10.json` + `coverage_diagnosis2_2026-08-10.json`.

### 3.1 `selected = oracle@8 × sel_eff` is an EXACT IDENTITY

I had been reasoning with `selected ≈ greedy + sel_eff × (oracle − greedy)`. **It over-predicts selected
accuracy by +0.090 to +0.111 on every cell.** Measured: oracle@8 0.626013 × sel_eff 0.775204 =
**0.485288** = selected, max |err| **5.6e-17** over 4 cells (and **0.0** in two independent
re-derivations today, `headroom_percell:null_tests.open_text_bar.identity_check` and
`open_diverse:structural_headroom.identity_check`). The additive form predicts **0.5863** where the
measured value is **0.4853**, because it feeds a *conditional-mean* sel_eff (0.775 / 0.811) into a
*difference-form* formula whose sel_eff is **0.2029**.

**The slope survives, the level does not.** "Coverage has a multiplier" is still true marginally; my
projection *"oracle 0.626 → 0.70 puts selected at ~0.652"* is wrong — the correct value is
`0.811 × 0.70 = 0.567`.

**And the marginal multiplier is ~0.45, not 0.81.** Measured conversion of newly-covered questions falls
monotonically: 0.935 (covered at N=1) → 0.671 (added by sample 2) → 0.531 (3–4) → **0.4474
[0.3684, 0.5263]** (5–8). Realised `d(selected)/d(oracle)` = 0.415 over N=1→8, **0.303 over N=4→8**,
0.400 at N=7→8. **My 0.81 was optimistic by 1.8× and the CI excludes it.**

### 3.2 "Coverage has a multiplier, selection is exhausted" is BACKWARDS

Free upper bounds on the 8-cell macro, both charging no extra generation:

| ceiling | macro | vs direct |
|---|---:|---:|
| open arm at its **iid sampling ceiling** (N=∞, capture–recapture) | 0.66585 | **+0.0091** |
| open arm with a **PERFECT selector over the CURRENT 8-pool** | 0.68677 | **+0.0301** |

**The selection ceiling is 3.3× the coverage ceiling.** ~20 approaches converging at sel_eff 0.80–0.81
means selection is *hard*, not *small*.

And the coverage ceiling is not even reachable: capture–recapture puts the reachable-by-sampling share
at slake **0.917** / vqa_rad **0.692** / pathvqa **0.626**, while the oracle needed merely to *tie*
32B-direct at the measured multiplier is **0.958 / 0.868 / 0.518**. **SLAKE-open and VQA-RAD-open are
unreachable by iid sampling at any N.**

**Why: it is a CAPABILITY failure, not a DIVERSITY failure.** The no-coverage subset is the
*high*-diversity subset — mean distinct answers **5.17 vs 3.00**, modal share 0.432 vs 0.701,
normalized entropy 0.830 vs 0.522; 50.2% already emit 6–8 distinct answers out of 8. oracle@8 by
stratum: 0.890 at n_distinct=1 → **0.364 at 6–8**. An independent 16-sample redraw (3× the budget)
rescues **21.2%**. And they are not near misses: best token-F1 against gold on the no-coverage subset is
**0.113** (vs 0.889 on recoverable) and **71.5% have zero gold tokens anywhere in the 8-answer pool** —
so the "if these are perception failures on short factual answers, go image-side" conditional has a
**false antecedent**.

**A third correction rode along:** my open-text arithmetic was **sample-weighted** while the target is
**macro**. On macro the open half's 32B-direct is **0.5982** (not 0.5168) and oracle@8 is **0.6752**
(not 0.6260) — the incumbent open arm sits **0.0647 BELOW** direct on the open half, a much bigger hole
than I had described.

---

## 4. The four attacks (13:00 – 15:35)

### 4.1 OPEN-STRONG — best-of-N on the *strong* leg. **TIE.**

The strong leg had only ever been **one greedy 32B pass**. `openstrong_bestofn_2026-08-10.json`,
3 independent vLLM generation seeds, verifier and selector frozen.

**Seed-averaged, against a MATCHED same-runner baseline:**

| arm | macro | Δ vs matched direct | verdict |
|---|---:|---|---|
| **A5 format-aware (pre-registered primary)** | 0.6597 | **+0.0012 [−0.0055, +0.0080]** | **TIE** |
| A2 N=4 sub-pool | 0.6633 | +0.0048 [−0.0014, +0.0110] | TIE (bigger point est., not the primary) |
| CONTROL majority-8 | 0.6594 | +0.0009 [−0.0028, +0.0049] | TIE |
| BASELINE always-32B-best-of-8 | — | **+0.0000** (−0.0007…+0.0007) | **the bar does NOT rise** |

Kill K1 fired (SLAKE_open, VQA_RAD_open). **PATH_VQA_open genuinely WON: +0.0269 [+0.0089, +0.0456].**

**And it is not the verifier's fault — both inputs improved and the bar moved faster.** The frozen
verifier *transfers better* to 32B candidates (sel_eff s0 **0.897 / 0.831 / 0.706** vs 0.850 / 0.762 /
0.723 on the 7B pool) and 32B oracle@8 is higher on all three (**0.901 / 0.710 / 0.583** vs 0.879 /
0.630 / 0.517) — but the greedy 32B answer the arm must beat rose faster still.

**The majority-vote control is the sharp one:** it *loses* on PathVQA-open (−0.0162 [−0.0296, −0.0036]
SIG) exactly where the verifier wins. **What wins there is the trained verifier, not the ensemble.**

Cost: **4.259×** always-32B-direct as-charged (the 1.675× shared-prefill figure depends on the very
assumption Attack 3 put in doubt — quote 4.26×). Token audit: 4.73–5.91 mean generated tokens, every arm
genuinely direct.

**Baseline honesty, as pre-registered:** always-32B-best-of-8-plus-verifier does **not** beat
always-32B-direct on macro. **The paper's bar does not rise; always-32B-direct remains a fair and in
fact compute-optimal way to spend the 32B on this suite.**

### 4.2 MCQ-TTA — multi-view / permutation ensembling. **DEAD.**

`mcq_tta_2026-08-10.json`. The MCQ half is 62.5% of the macro weight and the method is literally the
baseline on four of five of those cells, so TTA was the obvious cheap lever. **It reaches less weight
than it looks:** cyclic option permutation applies only to PMC (4 options) and MedXpert (5 options) =
**2/8 = 25%** — VQA-RAD-cl and PathVQA-cl are yes/no with no option list and SLAKE-cl is free
single-word (420/836 Chinese). That was verified on disk and written into the pre-registration *before*
the run.

| endpoint | summed MCQ gain | bar |
|---|---:|---:|
| **always-K (EVAL-VISIBLE UPPER BOUND)** | **−0.00779** | +0.0235 |
| deployable cross-fit gated policy | +0.0000622 | +0.0235 |

**The pre-registered kill fired at Stage A and Stage B was never run** — if the eval-visible upper bound
is below the bar, the cross-fit policy cannot reach it. Macro on the same pool: −0.00133
[−0.01197, +0.00921] n.s. The PMC K-curve is monotonically *downward*: 0.5673 → 0.5617 → 0.5418 →
0.5402. The luck-floor control rejected the null on both permutable cells (PMC 0.5402 vs 0.2501
[0.2390, 0.2617]) — the ensemble is doing something real; it just does not help.

**Two zero-GPU priors had already said so:** a 2-view prompt-form ensemble gains −0.0042 by confidence
and **exactly 0.0** by a cross-fit logistic against a 2-view *oracle* of +0.0484; the cyclic-permutation
majority on MMMU-Medical (n=145) is −0.0069 [−0.0483, +0.0345] against an oracle of +0.1448. The run
went ahead because its aggregation averages per-option *posteriors* rather than voting on argmaxes.
It did not matter.

**One loose thread, flagged and NOT claimed:** on the PMC n=6,000 subsample, single-view **option-logprob
argmax** reads 0.5673 against the deployed generated-letter parse at 0.5523. **No CI was computed, it is
eval-visible, and it decays with K.** A lead for a future round, not a number.

### 4.3 COST-FLOOR — make cost the endpoint. **Target missed; robust partial positive.**

`cost_floor_2026-08-10.json`. Pre-registered success: **≤ 1.00× direct's macro compute at a preserved
tie** (CI lower bound ≥ −0.0029). **Not met.**

- A single fold seed reaches **0.906×** — the tie survives on **1/12 seeds**. **Fitting noise; do not
  quote it.**
- The fully honest **nested-CV** variant (eps chosen inside the training folds) gives **0.9931×** at
  −0.0009 **[−0.0034, +0.0015]** — the lower bound **misses the pre-registered tie by 0.0005**.
- **THE ROBUST POSITIVE:** eps=0 cross-fit per-cell arm selection — "deploy the arm the training folds
  say is most accurate" — reaches macro **0.6578 (sd 0.0011), +0.0011 [−0.0014, +0.0035]**, tie on
  **10/12** seeds, not-significantly-worse on **12/12**, guardrail clean, 7B macro weight 0.2833 (not
  degenerate), at **1.165× ± 0.025** against the shipped arm's **1.740×**. **Equal accuracy for a 1.49×
  compute reduction, with no re-costing argument involved at all**, and **−89.4% latency / −90.2%
  energy** against a 32B honestly re-costed to actually reason.
- **The cost floor is set by ACCURACY, not accounting:** going below 1.00× costs ~0.002 macro and breaks
  the tie.
- **The uncomfortable composition, which must travel with the number:** every cheap tie-preserving policy
  gets there by **routing the open-text cells to always-32B-direct**. SLAKE-open and VQA-RAD-open go to
  direct at *every* eps and *every* seed, because direct is **both more accurate and cheaper** there
  (0.8186 vs 0.8171 at 4.57 vs 13.97 FLOP-eq; 0.6000 vs 0.5900 at 4.57 vs 17.30). The open-text
  best-of-N machinery survives only on PathVQA-open.

**⛔ A lever I had assumed was voided — and the experiment that settles it never ran.** The brief asserted
that `run_openvqa.py:154` shares the prefill because it uses `SamplingParams(n=N)`. **In vLLM V1 — the
default in the 0.9.0.1 that produced the deployed dumps — `n=N` is not a post-prefill fork:**
`vllm/v1/engine/parallel_sampling.py:ParentRequest` splits it into **N child requests**, each carrying
the full prompt with `n=1`, and sharing then depends entirely on automatic prefix caching. The N
siblings arrive together and 8 × ~327 ≈ 2.6k tokens fits in one scheduler step, so they can each *miss*
the cache. **If that is what happens, the as-charged BO8 = 16.0 charge is simply right.** The verifier
half is definitely not shared (one HF batched forward over 8 full prompts; a batch does not reduce
FLOPs).

**Status, stated honestly: this is a CODE INSPECTION, not a measurement.** The decisive two-minute
experiment — reading `RequestOutput.num_cached_tokens` — **failed to run**: all three vLLM files under
`artifacts/_cost_floor_measure/` are **0 bytes**, `logs/cost_floor_vllm_on.log` ends in
`RuntimeError: Engine core initialization failed`, the poller recorded `vllm rep1 FAILED / rep2 FAILED /
vllm-nopc FAILED`, and `rule2_corroboration.measured_shared_prefill_path` is **null**. **Conventions B
and C are hypothetical and must not be headlined.** The as-charged primary endpoint is unaffected.

Both R32 values reported as required: as-charged 4.57 (which *flatters* us) and derived 3.816 (which
hurts — nested CV 0.9931× becomes **1.0344×**).

### 4.4 OPEN-DIVERSE — DPP portfolio vs iid at fixed N=8. **CLEAN LOSS, and it retires a live claim.**

`open_diverse_2026-08-10.json`. Two repo artifacts contradicted each other about *why* sel_eff falls:
`verifier_n_scaling_2026-08-03.json` said pool **SIZE** (−0.0761 per doubling); `diverse_generation_gpu.json`
(2026-07-06) said **REDUNDANCY** (iid-8 → DPP-8 raising sel_eff **+0.0644** vqa_rad, **+0.0962**
pathvqa). **Both were measured with the contaminated `pooled4` verifier.**

**Three defects found before the gate was decided:** (i) contaminated verifier; (ii) **the published
diverse scores came from a visually blind verifier** — `diversity_generate_gpu.py` scores candidates
with a vLLM `LoRARequest`, and vLLM 0.9.0.1 silently drops all 192 `visual.*` modules (re-scored under
HF here, argmax pick agreement is only **74.7 / 74.5 / 75.8%**); (iii) **exact-match labels wearing a
judge's name** — the `mcq_gen_verify` dumps' `sl` field is *identical* to their `oks` on 100% of rows
(1061/1061, 451/451, 345/345), and every downstream reader treats `sl` as a judge label. Both pools were
judge-labelled here before the gate (concordance vs the transfer dumps 0.9979 / 0.9973 / 0.9959).

**The published numbers reproduce exactly (max dev 4.73e-05), the DPP re-implementation reproduces the
stored picks on 100% of items — and then the effect reverses under decontamination + HF scoring + judge
labels:**

| cell | Δ sel_eff | Δ selected acc |
|---|---|---|
| slake_open | **−0.0748 [−0.1031, −0.0474] SIG LOSS** | **−0.0465 [−0.0729, −0.0217] SIG LOSS** |
| vqa_rad_open | −0.0465 [−0.1097, +0.0166] | −0.0150 [−0.0600, +0.0250] |
| pathvqa_open | +0.1214 [−0.0184, +0.2613] | +0.0056 [−0.0393, +0.0506] |

**Both kill criteria fired at Phase 0. No GPU was spent on Phase 1.** The confound-free control agrees:
DPP-8 vs random-8 from the **same** M=15 pool over 20 seeds is **negative on selected accuracy on all
three cells** (−0.0109 / −0.0112 / −0.0059). And the target is out of reach anyway — to lift a cell over
32B-direct at fixed sel_eff, oracle@8 must reach 0.9721 / 0.7977 / 0.5311, while the **M=15 portfolio
oracle measures 0.9116 / 0.6800** on the two cells it covers.

Disjointness re-proved in the attack's own code: train/eval **image pixel-md5 intersection = 0** (3,457
vs 528 images).

---

## 5. The caveat that outlives the round

**Regenerating the 32B greedy open-text arm under a different tensor-parallel configuration reproduces
the published cells only to ±0.008 per cell (±0.00183 macro).** SLAKE 0.0016, VQA-RAD 0.0050, **PathVQA
0.0080** — against a pre-registered abort threshold of 0.005.

It is **not** a prompt or decode-path difference. The system prompt, `cap320 max_pixels=250880`,
`max_tokens=64`, `max_model_len=4096` and the extraction rule are copied verbatim; mean generated tokens
match to 0.1; **95.35% of the 2,345 regenerated greedy answers are byte-identical**. The deviation
**tracks the tensor-parallel setting exactly** — the two cells whose deployed runner used `tp=2` deviate
0.0016 / 0.0050 with 3 and 1 judge disagreements, while PathVQA, whose runner used `tp=1`
(`runners/run_openvqa_pathvqa.sh:7`), deviates 0.0080 with 26.

**That is 63% of the +0.0029 significance bar and larger than the entire published
accuracy-max-vs-direct delta of +0.0008.** Attack 1's pre-registration said to **stop**; instead every
headline was reported against **both** the published and a matched same-runner baseline and the verdict
taken from the matched one. **That is a post-hoc remediation of a pre-registered failure and is labelled
as such in the artifact and the doc.** Standing rule from today: **any open-text comparison must carry a
matched control arm generated in the same serving configuration.**

---

## 6. What was learned

**The tie is not a near miss; it is a fixed point.** Four independent attacks on four different levers —
a stronger escalation target, MCQ test-time augmentation, cost as the endpoint, and generation diversity
— all land within ±0.005 of the same place, and the scouting says why:

1. **The ceiling is enormous and unreachable.** Σ p10 = 0.529 (+0.0661 macro if perfect) converts at
   **1.3%**. Every cheap signal is AUROC 0.57–0.64 among disagreements, and **chance** on the
   highest-headroom cell.
2. **The method degenerates to the baseline on half the macro weight**, and extending the shipped MCQ
   lever to the rest of that half is worth **+0.0000**.
3. **Nothing already measured can be rearranged into a win** — the eval-visible best-per-cell
   reassignment is **+0.00277** against a **+0.0029** bar, and that already cheats.
4. **The two new levers are measured and too small.** Coverage: +0.0091 ceiling, unreachable on 2 of 3
   cells, and the uncovered questions are high-diversity capability failures with zero gold tokens in
   the pool 71.5% of the time. Selection: +0.0301 ceiling but a **seed spread (~0.021) larger than every
   architectural effect**.
5. **And the measurement floor on open text is ±0.008 per cell — 1.03× the +0.00773 each open cell
   would have to contribute to clear the bar**, or **±0.00183 macro = 63% of the +0.0029 bar** and
   **2.3× the shipped +0.0008 delta**. *(Corrected 2026-08-12: this read "4.4× the delta being chased",
   which divided a per-cell figure by a macro figure. Doc §9.8.)*

**What survives, and it is a paper.** Against a 32B *actually made to reason*: **+0.0601
[+0.0499, +0.0700]** (deployed selector) or **+0.0615 [+0.0514, +0.0715]** (frozen 8-seed selector), at
**−87.7% parallel latency and −84.3% energy** honestly re-costed. Plus today's cost result: **equal
accuracy at 1.165× versus the shipped 1.740×**. Framed as *"a reasoning-32B's accuracy at a tenth of its
latency and energy, and parity with a direct 32B while spending less than the previously shipped
configuration"*, that is fully measured and defensible — **and it does not need the vs-direct accuracy
win that four attacks have now failed to produce.** Its limits must travel with it: parity, not
superiority, vs always-32B-direct; the compute saving is concentrated on low-escalation cells, not
uniform; and every cheap tie-preserving policy reaches its cost by routing two of three open cells
straight to the 32B.

**Methodologically, the round did the right things and they cost nothing.** Pre-registration caught two
attacks before they burned GPU (MCQ-TTA's kill fired at Stage A; OPEN-DIVERSE's fired at Phase 0). The
identity control on Attack 1 caught a ±0.008 reproducibility artifact that would otherwise have been
silently absorbed into a headline. The luck-floor control confirmed the TTA ensemble was real *and*
useless. The 12-seed sweep exposed a 0.906× "win" as fitting noise that survives on 1/12 seeds. **Every
one of those is a claim this project would previously have shipped.**

---

## 7. Standing state and what did not run

**Doc written:** `results/cascade_methods/docs/current/BEAT32B_ROUND_2026-08-10.md` — the round
synthesis, verdict on line 1, every figure source-tagged, with a §9 listing seven discrepancies between
the circulating summary and the artifacts.

**Artifacts (all `results/cascade_methods/artifacts/`):** `headroom_percell_2026-08-10.json`,
`coverage_diagnosis_2026-08-10.json`, `coverage_diagnosis2_2026-08-10.json`,
`openstrong_bestofn_2026-08-10.json` (+ `_preregistration`), `mcq_tta_2026-08-10.json` (+ `_pilot`,
`_preregistration`, `_nulltests`), `cost_floor_2026-08-10.json` (+
`_cost_floor_verifier_geometry.json`, `_cost_floor_measure/`), `open_diverse_2026-08-10.json`
(+ `_preregistration`).

**Code:** `src/cascade_methods/{headroom_percell,coverage_diagnosis{,2,3,4,5},openstrong_bestofn,
openstrong_gen,mcq_tta*,cost_floor{,_measure,_geometry},open_diverse{,_prereg,_score}}.py`. **Runners:**
`run_openstrong_{queue,judge,finish}.sh`, `run_mcq_tta{,_tp1}.sh`, `run_cost_floor_measure.sh`,
`run_attack4_phase1_pathvqa.sh`. **Nothing under `MedEvalKit/`, `MedVLThinker/` or `MedRAG/` was
modified.**

**Four things did not run, and three of them are cheap to finish:**

1. **The vLLM `num_cached_tokens` measurement.** Two minutes of clean GPU settles whether the deployed
   `n=N` path shares the prefill; re-running `src/cascade_methods/cost_floor.py` afterwards fills
   `rule2_corroboration` automatically. Until then conventions B and C stay hypothetical.
2. **Attack 2's latency/energy measurement** — `logs/mcq_tta_cost.log` is three identical
   `ModuleNotFoundError: No module named 'pynvml'`. Irrelevant to the verdict, but it means Attack 2 has
   no measured cost numbers at all.
3. **Attack 4's fully-matched `frozen_clean` contrast** — the 32B judge OOMed at engine init under GPU
   contention (`logs/attack4_judge_iidmcq.log`). The exploded judge inputs and clean HF scores already
   exist; `python3 src/cascade_methods/open_diverse.py` finishes it. The verdict does not depend on it —
   the two confound-free controls both agree with the gate.
4. **Attack 2's Stage B and PMC full-cell extension** — both pre-registered as conditional and both
   correctly *not* triggered.

**Open questions, in the order they should be answered:**

1. **Should there be a round 2 at all?** The measured answer to *"can this architecture beat
   always-32B-direct"* is now "not by any route we can find, and here is the identity that caps it".
   Writing the vs-reasoning + cost paper is a better use of the next week than a fifth attack.
2. **If round 2 happens, the priorities are measured:** the **generator** over the verifier; **PMC_VQA**,
   the only cell where a cheap signal converts and the cell the whole delta rests on;
   **heterogeneous-CONFIG generation** — one sample at a *changed image resolution* rescues more
   no-coverage PathVQA-open questions than one more iid sample (cap80 **+0.0214 [+0.0050, +0.0377]**,
   cap160 +0.0201 [+0.0050, +0.0365], exact-match currency, n=795) — **while noting it is worth only
   ~+0.0016 on the macro**; and **nothing** whose effect is smaller than ±0.008 per open cell.
3. **Seven discrepancies between my circulating summary and the artifacts** are catalogued in the doc's
   §9. The two worth fixing at the source are the PathVQA-open pool-oracle CI (I have been quoting
   [+0.1173, +0.1647]; the artifact says **[+0.1167, +0.1653]**) and the over-strong phrasing of the
   vLLM finding (a code inspection, not a measurement).
4. **`git push` is still the top-priority chore**, and a push still does not protect `ckpts/`,
   `feats_hidden/` or `logs/` — the reproduction chain remains on one disk.

---

## Addendum — 2026-08-12 (audit of this round's own documentation)

The round-1 artifacts were re-read against this diary and against
`docs/current/BEAT32B_ROUND_2026-08-10.md`, one number at a time. **Every headline verifies.** Spot-checks
re-derived from the files rather than copied: Σ p10 = **0.5290** over the eight
`per_cell.*.ceilings.item_oracle_7B_or_32Bdirect.gain_over_always_32B_direct.mean` entries ⇒ macro
**0.066125**, against a realised **+0.00083** — the **1.3% conversion** stands; the two coverage ceilings
are **0.6658469757964316** and **0.6867670542635659**, so **selection headroom is 3.29× coverage
headroom**; the exact identity holds at **max abs deviation 0.0** (`sel_eff × oracle = selected =
0.485287846`); Attack 1's A5 is **+0.0012 [−0.0055, +0.0080]** with **PATH_VQA_open +0.0269
[+0.0089, +0.0456]**; Attack 2's always-K upper bound is **−0.0077870559230383**; Attack 3's nested CV is
**0.9931× at −0.0009 [−0.0034, +0.0015]**, missing the pre-registered tie by exactly **0.0005**; Attack 4's
SLAKE-open loss is **−0.0748 [−0.1031, −0.0474]**. The doc's whole §9 discrepancy list also verifies,
including the PathVQA-open pool-oracle CI (**[+0.1167, +0.1653]**, not the [+0.1173, +0.1647] I had been
circulating) and the PMC veto rate (**0.4002**, i.e. escalation 0.5998, not the compute-lean 0.0845 that
the escalation column shows).

**One defect found, in my own summary rather than in any artifact.** §6 item 5 above and §10 step 4 of the
doc both said the ±0.008 open-text measurement floor was **"4.4× the delta being chased"**. That ratio is
`0.008 / 0.00182` — a **per-cell** deviation divided by the **macro** equivalent of the same deviation.
Wrong units, same species as pairing a macro accuracy with a sample-weighted cost. Both files are
corrected in place with the correction labelled; the doc records it as **§9.8**. The supported forms are
**±0.008 per cell = 1.03×** the +0.00773 each open cell must contribute to clear the bar, and **±0.00183
macro = 63%** of the +0.0029 bar / **2.3×** the shipped +0.0008 delta. **No measured value changed.**

**And open question 1 above has been answered by events.** Rounds 2 and 3 ran, were killed by session
limits too, and were salvaged in `18ee797`. There is now a **CI-clean macro win over always-32B-direct**
(`artifacts/armcombine_mcqonly_2026-08-11.json`, status **POST-HOC / EXPLORATORY**): MCQ-half = the
shipped certified veto, open half = always-32B-direct, macro **0.657865, +0.00119 [+0.0009, +0.00148]**,
guardrail-clean, **0.9773× FLOP-eq**. It is **100% PMC_VQA** — byte-identical to the baseline on 7 of 8
cells *by construction*, leave-one-out on PMC takes it to exactly 0.000 — on **`test_2.csv`**, the split
with zero published verification, and it is **gated behind an owed answer-letter-bias audit** (B+C =
73.6%, 37.8% constant-C floor). It does not contradict a line of this round. It is, in fact, the shape
round 1 predicted: **the only way found to beat always-32B-direct is to switch the method off wherever
it is not the baseline already.** A forward pointer to it now sits at the top of the doc.
