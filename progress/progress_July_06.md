# Progress — July 6, 2026

> Continues `progress_July_05.md` (which closed the OmniMed reproduction on the cheap side, took the
> keep-cheap strong-leg fallback, and seeded a cross-field method-idea backlog). This is the **execution
> marathon**: with the GPUs freed, ~15 experiments ran between ~11:43 and ~18:56 — a first offline round
> (three both-axes / candidate-quality tests, incl. the cycle's headline WIN), a UGV MCQ resolution, a
> second offline round of three honest negatives, two GPU passes (both positive), a five-experiment
> **selectability-wall battery** that re-grounds the best-of-N program, and two Pandora refinements.
> Every number below is copied from, or arithmetically derived from, the cited artifact under
> `results/cascade_methods/artifacts/`; all code under `src/cascade_methods/`. All figures land in the
> ledger `results/cascade_methods/docs/current/RESEARCH_RESULTS_2026-07.md` (written 18:56). The two
> GPU passes used real inference; everything else is CPU-only re-simulation over existing per-sample dumps.

## 0. Framing the day — fallback formalized + backlog expanded to 35

Two housekeeping artifacts bracket the morning and set up the experiment battery:

- **`OMNIMED_FALLBACK.md`** written 12:20 — formalizes yesterday's keep-cheap decision (strong OmniMed
  leg = paper-reference + infra-limited; no fabricated `metrics.json`). Reproduction = 6/7 faithful +
  OmniMed cheap-faithful.
- **`METHOD_IDEAS_BACKLOG.md` pass 2** (13:57) — +12 cross-field ideas → **35 total**, re-ranked. Pass 2
  targets the two *now-binding* limits (candidate quality / oracle@N; verifier near-tie selection). The
  new top-5-new: **C9** information-directed active-comparison verifier (TrueSkill μ,σ), **B5** Dawid–Skene
  truth inference, **A8** repulsive/semantic-guided diverse decoding, **C7** best-arm-identification bandit
  allocation, **B6** surprisingly-popular / Bayesian Truth Serum. Several of these get tested *today*.

## 1. Offline round 1 — three candidate-quality / controller tests (incl. the WIN)

### 1.1 Pandora's-Box adaptive controller (Weitzman) — the cycle's headline both-axes win

**What / why.** Backlog **C1**. `pandora_controller.py` implements the Weitzman (Econometrica 1979)
optimal-search rule as ONE controller that **unifies adaptive-N and the escalation gate**. Each "box" is
either "draw one more 7B sample" (cost 2.0 FLOP-eq = GEN7+VER7, reward = the verifier's calibrated
P(correct)) or "escalate to the 32B" (cost 4.57 FLOP-eq, deterministic reward = calibrated
P(strong-correct)); one exchange-rate knob λ yields BOTH a stop-drawing threshold and an escalation
threshold. **Honesty knob: Pandora's thresholds are held-out (5-fold cross-fit isotonic calibration, no
peek at correctness); the baselines' τ are swept on full data (optimistic "oracle-τ").** Cost model = the
project's canonical measured batch-1 model. Headline = per-domain-tuned aggregate, n-weighted over 11
(family × open-dataset) configs (`pandora_controller.json → SUMMARY_per_domain_tuned`).

| Target | Method | datasets covered | FLOPs | vs bo8 | energy (J) | meanN | esc | lat_seq (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **iso-bo8** (match cheap-ensemble acc) | **Pandora (held-out)** | 9/11 | **11.74** | **−27%** | **409.8** (**−28%**) | 5.38 | 21.6% | 2,951 |
| | adaptive-N (oracle-τ) | 11/11 | 13.01 | −19% | 459.5 | 6.31 | 8.7% | 3,350 |
| | verifier-bo8 + gate (oracle-τ) | 11/11 | 16.00 | 0% | 568.8 | 8.00 | 0.0% | 4,176 |
| **iso-strong** (match always-32B acc) | **Pandora (held-out)** | 11/11 | 6.32 | −61% | 222.6 | 3.03 | 5.5% | 1,619 |
| | adaptive-N (oracle-τ) | 11/11 | 10.84 | −32% | 383.6 | 5.30 | 5.4% | 2,802 |

**Headline.** At **iso-bo8 accuracy, held out**, Pandora reaches the target at **−27% FLOPs / −28% energy**
vs fixed best-of-8, and beats even the **optimistically-tuned** adaptive-N (−19%) and gate (0%) baselines
*despite being the only method whose thresholds are held-out.* (Reference costs: `bo8` = 16.0 FLOP-eq /
568.8 J; `always-32B` = 4.57 FLOP-eq / 127.0 J.)

**The reframe this forces (important, and it shapes the rest of the day).** On these open-text OOD
workloads the accuracy ceiling worth matching is the **cheap best-of-8 ensemble, not the 32B.** Pooled
Lingshu: bo8 = **0.414** vs always-32B = **0.331** — the strong model is a *weaker* accuracy target here, so
escalating to it buys little and the real lever is reaching bo8 accuracy more cheaply. That is why "iso-bo8"
is the headline row and "iso-strong" is a footnote (matching the 32B is easy — Pandora does it at 6.3
FLOPs — but uninteresting). **This "32B is a weak target" assumption is exactly what July 7 re-examines.**

**Honest caveats.** (i) On **2 of 11** configs Pandora's held-out frontier falls just short of bo8 parity
(covered 9/11 at iso-bo8) — where bo8 ≈ the oracle ceiling, a controller that stops early cannot always
match it. (ii) **Latency trade is real and adverse on one axis:** adaptive drawing is sequential
(draw→check→draw), so Pandora's `lat_seq` ≈ 2,951 ms, whereas a fixed bo8 can batch its 8 draws in parallel
(`lat_bat` ≈ 522 ms). Pandora wins FLOPs + energy but is **slower in batch-1 wall-clock than a batched
bo8** — the code reports both and flags this. (iii) Weitzman assumes independent box rewards;
within-question samples are correlated (re-sim over recorded draw order mitigates but doesn't eliminate).
Code: `src/cascade_methods/pandora_controller.py` · Artifact: `pandora_controller.json`.

### 1.2 Cross-model candidate pooling (generator portfolio / Markowitz) — pooling helps, allocation doesn't

**What / why.** Backlog **A2**. `generator_portfolio.py` treats each cheap generator {Lingshu-7B,
MedVLThinker-7B, InternVL3-8B} as a Markowitz "asset" (return = per-sample accuracy, covariance =
error-correlation φ) and, for a fixed budget B ∈ {2,4,8,16}, allocates B samples across generators to
maximise **oracle@B** (the #1 binding limit). Coverage = a with-replacement pass@k estimator applied
identically to all methods; the Markowitz allocation is fit on a train fold, scored held-out (5-fold CV).
Mean off-diagonal error-φ ≈ **0.52–0.56** (models fail on somewhat different questions, but errors are
still positively correlated).

Held-out oracle@B, pooled over the 3 all-generator datasets (kvasir + radimagenet + vqa_rad, n=3,400):

| B | single-best (Lingshu) | uniform pool | portfolio (Markowitz) | Δ portfolio vs single | Δ portfolio vs uniform |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.359 | 0.395 | 0.415 | **+0.056** | +0.020 |
| 4 | 0.421 | 0.481 | 0.486 | **+0.065** | +0.005 |
| 8 | 0.467 | 0.545 | 0.547 | **+0.080** | +0.002 |
| 16 | 0.498 | 0.600 | 0.603 | **+0.105** | +0.002 |

**Read.** Per-dataset, pooling 3 models beats the best single model by **+0.045 to +0.127 oracle** (e.g.
vqa_rad +0.108 @B=8, +0.127 @B=16); pooled **+0.08 (B=8) / +0.11 (B=16)** — a real, held-out lift of the
accuracy ceiling. **BUT the Markowitz-optimal allocation ≈ a naive uniform split** (Δ vs uniform = +0.002
to +0.02 pooled, and **negative** on vqa_rad: −0.003 @B=8, −0.005 @B=16). **The win is diversity/pooling,
not the clever allocation** — the same verdict the bandit test (§3.2) reaches independently.

**Honest caveats.** (i) Oracle-ceiling result; a trained verifier realises only ~74–82% of it (limit #2).
(ii) No temperature variants on disk → assets = the 3 models only. (iii) φ estimates are per-domain noisy;
a weak generator (InternVL3-8B on pathvqa is near-floor) can dilute the verifier's job.
Code: `generator_portfolio.py` · Artifact: `generator_portfolio.json`.

### 1.3 Diversity-maximized candidate selection (DPP/MMR) — a rate win, not a ceiling lift

**What / why.** Backlog **A1**, offline half. `diversity_candidates.py` tests whether diversity-aware
*selection* (greedy farthest-first / MMR ≈ DPP-MAP over answer-string distance) reaches the oracle answer
in fewer samples than iid draw-order. **Honest by construction:** reordering a FIXED set of 8 candidates
CANNOT raise oracle@8 — only diverse *generation* (the GPU job in §4.1) can. Pooled across all
families/datasets, n=15,057:

| Ordering | oracle@8 | AULC (area under oracle@k) | k_reach (mean #samples to first correct) |
|---|---:|---:|---:|
| draw (iid) | 0.412 | 0.350 | 2.20 |
| random subset | 0.412 | 0.350 | 2.20 |
| MMR (exact-string) | 0.412 | 0.365 | 1.91 |
| **MMR (token-Jaccard)** | 0.412 | **0.368** | **1.86** |

**Read.** oracle@8 is **identical (0.412) across all orderings — confirming no ceiling lift.** MMR reaches
the first correct answer with **k_reach 2.20 → 1.86 = −15.6% fewer samples** (a rate/efficiency win).
Redundancy headroom: on average only **4.67 of 8 candidates are distinct** (≈42% of iid samples are exact
duplicates), so an idealized perfectly-diverse *generator* could hit the same oracle@8 with ~4.7 samples
(up to **−42%** samples — an upper bound). The offline test proves only the −15% selection-rate win; the
ceiling-lift claim needs the GPU diverse-*generation* pass. Code: `diversity_candidates.py` · Artifact:
`diversity_candidates.json`.

### 1.4 Format-router re-score (Method C, `unified_router.py`)

Re-ran and re-scored the deterministic format-aware router (12:44) that dispatches MCQ/closed → margin
cascade, open → verifier best-of-8 cascade, scoring the whole pooled stream as ONE accuracy-vs-cost point
against always-strong. Confirms the July-4 result as the ledger's §1.1: matches-or-beats always-strong
accuracy on all 3 families (Δacc **+0.003 / +0.000 / +0.001**) at **48–82% of always-32B FLOPs** (30–59% if
image prefill is prefix-shared). This is the day's *deployable* efficiency anchor — and the selectability
battery (§5) ends by re-affirming the router, not best-of-N, as the deployable lever. Artifact:
`unified_router.json`.

## 2. UGV — single generative verifier, MCQ-as-generation — RESOLVED (negative for MCQ)

**What / why.** Backlog **B2** — the project's stated frontier: a Unified Generative Verifier that scores
MCQ options as *generated answers* through one grounding verifier (unify MCQ + open-text + boxes; attack
limit #2). The data-loader fix landed and the experiment ran on Lingshu-7B (+ MedVLThinker-7B) over PMC-VQA
and MedXpert (n=2,000 each), self-consistency N=8. **Two scoring modes:** `content` (options hidden → the
model must *generate* the answer string) vs `letter` (standard A/B/C/D letter-logprob); `strict` = strict
greedy parse. Code: `ugv_mcq_verdict.py` over `ckpts/mcq_gen_verify/` dumps · Artifact:
`ugv_mcq_verdict.json` · driver `runners/run_ugv_experiments.sh` (`logs/ugv_experiments.log`).

**Verdict — the no-router single generative verifier does NOT work for MCQ.**

| dataset | mode | greedy | verifier-boN | oracle-boN | verifier gain | AUROC(score vs ok) |
|---|---|---:|---:|---:|---:|---:|
| PMC-VQA | content | 0.132 | 0.140 | 0.300 | +0.009 | 0.793 |
| PMC-VQA | **letter** | **0.534** | **0.616** | 0.800 | **+0.082** | 0.540 |
| MedXpert | content | 0.499 | 0.494 | 0.800 | −0.004 | 0.498 |
| MedXpert | **letter** | **0.556** | 0.556 | 0.843 | −0.001 | 0.478 |

1. **Content-mode collapses MCQ accuracy.** Hiding the options and forcing free-text generation craters
   greedy accuracy: PMC-VQA **0.132 content vs 0.534 letter** (a ~0.40 drop); MedXpert 0.499 vs 0.556. The
   MCQ signal lives in the option set — discarding it to "unify" formats destroys what makes MCQ tractable.
2. **The verifier's gain on content-MCQ is negligible.** Pooled content-MCQ mean verifier-boN gain =
   **+0.0038** (strict), mean AUROC **0.696** — the generative verifier barely reranks its own weak content
   candidates.
3. **Letter-mode + verifier-boN is inconsistent / label-sensitive.** PMC letter gains **+0.082**
   (0.534→0.616) but MedXpert letter is **flat (−0.001)**, and the "gain" flips sign under the as-run
   (non-strict) parse (PMC letter as-run verifier_gain **−0.074**). Not a reliable lever.

**Conclusion.** The **router stays decisively the better method**: MCQ → letter + margin gate, open-text →
trained best-of-N verifier. The single generative verifier's genuine home is **open-text**. Backlog B2
closed as a *negative for MCQ*. (`holds = true` in the artifact = the router-is-better verdict holds.)

## 3. Offline round 2 — three honest negatives (the selection wall resists post-hoc tricks)

### 3.1 Active pairwise-comparison verifier (C9 / active-IDS) — simulated, NEGATIVE

`active_comparison_verifier.py` simulates pairwise verdicts from the pointwise scores
(P(i≻j)=σ(logit sᵢ−logit sⱼ); **no real pairwise dumps existed yet**). Result: active-IDS **cannot beat
pointwise-argmax** on selection (mean active−pointwise sel_eff = **−0.003**, `beats_pointwise=false`), and
the noise-free **round-robin ceiling equals pointwise (+0.000)** across all 3 families (Lingshu 0.806, MVT
0.800, IV3 0.741) — confirming the simulated pairwise preference **carries no information beyond the
pointwise score**. Its only edge is cost: active-IDS matches round-robin at ~6.2/28 comparisons ≈ 22% of a
full pairwise pass (Lingshu). **Honest limit flagged in the artifact:** you cannot manufacture comparative
signal by deriving pairwise preferences from pointwise scores — this needs a *real* pairwise forward pass.
That real pass runs in §4.2 and **overturns this**. Artifact: `active_comparison_verifier.json`.

### 3.2 Bandit / adaptive allocation (C7) — NEGATIVE

`bandit_allocation.py`: per-question adaptive allocation (Thompson-soft, UCB-E) of the sample budget across
the 3 generators, reward = verifier score, held-out 5-fold. Result: adaptive allocation **≈ fixed uniform
pooling** for oracle@B (best pooled held-out **Δ=+0.002** over B∈{2,4,8}); Thompson/UCB-E track uniform,
sometimes *losing* to exploration cost — same verdict as Markowitz (§1.2, Δ=+0.020). The de-biased
split-oracle ceiling shrinks to Δ=+0.052 and **inverts at B=8 (−0.026)** (single-arm concentration can't
match uniform's cross-arm coverage). Uniform pooling is already near-optimal; the verifier score is too
weak to capture what little per-question headroom exists (the selection wall again). Artifact:
`bandit_allocation.json`.

### 3.3 Unsupervised answer aggregation (Dawid–Skene, B5) — NEGATIVE

`dawid_skene_aggregate.py`: grouped one-coin Dawid–Skene EM over the pooled 3-generator answers (per-source
reliability, no labels). Result: guarded DS **≈ plain pooled majority (−0.013)**, stays **+0.132 below** the
trained verifier and **+0.373 below** oracle. WHY: unsupervised reliability tracks **self-agreement (~0.52),
not accuracy (~0.29)** — the generators are **confidently wrong**, so cross-source agreement carries no
correctness signal and can't break the majority trap. Artifact: `dawid_skene_aggregate.json`.

**Round-2 unified takeaway.** The candidate-quality / selection wall **resists post-hoc tricks** —
*simulated* pairwise re-ranking, adaptive allocation, and unsupervised aggregation all collapse to the
pointwise/uniform/majority baseline. What moves the numbers is **real new signal**: better candidates
(diverse generation, cross-model pooling), a **real** pairwise verifier, and the trained verifier.

## 4. GPU passes — real signal, both POSITIVE (the first ceiling-moving wins this cycle)

Both GPU passes attack a different binding wall with real inference (`logs/gpu_experiments.log`,
`ckpts/pairwise/`, `ckpts/mcq_gen_verify/`, diverse candidate dumps).

### 4.1 Diverse generation (attacks limit #1 — coverage / oracle@N) — POSITIVE

`diversity_generate_gpu.py` (generate) + `diverse_measure_gpu.py` (score): a portfolio of **5 prompt
personas** {base, anatomy, modality, differential, concise} × a **temperature ladder** {0.7, 1.0, 1.3},
M=15 draws, scored against iid@8 on the same model/cap/verifier/scorer, diverse set restricted to the iid
idx set. Pooled n=1,623 over 4 open sets (vqa_rad, slake, pathvqa, pmc-content).

| metric (pooled) | iid@8 | diverse-DPP@8 (matched budget) | diverse-full@M=15 (extra budget) |
|---|---:|---:|---:|
| **oracle** | 0.593 | **0.621** (Δ **+0.027**, CI [0.010, 0.043]) | **0.657** (Δ **+0.064**, CI [0.047, 0.080]) |
| verifier bo-N acc | 0.434 | 0.449 (Δ +0.014, CI [−0.003, 0.031]) | **0.459** (Δ **+0.025**, CI [0.008, 0.042] **SIG**) |
| verifier bo-N eff | 0.732 | 0.723 | 0.699 |
| confident-distractor rate | 0.268 | — | 0.301 |

**Read.** Diverse *generation* (unlike diversity *selection*, §1.3) **does raise the oracle ceiling** —
+0.027 at matched 8-sample budget and +0.064 at M=15 (both CIs exclude 0) — and this **converts to a
significant +0.025 verifier best-of-N accuracy** at **1.875× (=15/8) generation cost**. Clean on **VQA-RAD**
(oracle +0.045 CI [0.01, 0.08] → verifier acc +0.06 CI [0.02, 0.105], only 1 lost-coverage question). BUT on
**PMC-content the oracle lift is largest (+0.110)** yet the **pointwise verifier cannot convert it**:
selection efficiency *drops* 0.574 → 0.496 because the extra diverse draws inject confident-but-wrong
distractors (confident-distractor rate 0.426 → 0.504; 93 new-coverage questions, only 11 converted, 27
lost). **KEY: diverse generation shifts the binding limit from coverage (#1) to selection (#2)** — it buys
real oracle headroom, but cashing it in needs a stronger *selector*. Artifact: `diverse_generation_gpu.json`.

### 4.2 Real pairwise verifier (attacks limit #2 — the ~74–82% selection ceiling) — POSITIVE; overturns §3.1

`pairwise_verifier_score.py` dumps **REAL** A-vs-B verdicts (same Lingshu-7B + pooled4-LoRA as the
pointwise verifier, prompted pairwise, both orders averaged for position debias);
`active_comparison_verifier.py --real_verdicts_dir ckpts/pairwise` scores them. Pooled n=578 over 3 open
sets (vqa_rad, pathvqa, slake).

| selector (pooled) | sel_acc | sel_eff | cost (comparisons/q) |
|---|---:|---:|---:|
| pointwise-argmax (deployed) | 0.374 | 0.783 | 0 |
| knockout (real pairwise) | 0.405 | 0.849 | ~7 |
| **round-robin / Copeland (real pairwise)** | **0.410** | **0.859** | 28 |
| oracle@N (ceiling) | 0.478 | 1.000 | — |

**Read (overturns the simulation).** The **real** pairwise verifier **beats pointwise-argmax** — sel_acc
0.374 → 0.410 (Δ **+0.036**, CI [0.016, 0.055]) and sel_eff 0.783 → 0.859 (Δ **+0.076**, CI [0.036, 0.116]);
on the **near-ties** (n=261, exactly where pointwise loses) the gain is **+0.050** (CI [0.012, 0.088]).
**Knockout captures most of the win at ~7 comparisons/q** (0.405 — ~87% of the round-robin gain at ~25% of
the 28-comparison pass). Overall this **closes ~35% of the pointwise→oracle gap**. It directly overturns
§3.1's simulated parity: deriving P(i≻j) from pointwise scores cannot create signal the pointwise head
lacks, but a *real* pairwise forward pass carries genuine comparative information it does not. Artifact:
`pairwise_verifier_gpu.json`.

**The obvious next question these two wins pose:** they attack different walls and look complementary
(diverse gen lifts coverage → hands residual to selection; the real pairwise verifier is the stronger
selector limit #2 needs). Do they **compound**? That is the first question of §5.

## 5. Selectability-wall battery (five experiments) — the best-of-N program is characterized, not deployable

Five follow-on experiments to decide whether the open-text best-of-N verifier is the *deployable* method or
merely a *characterized* one. Together they establish that the **selectability wall is fundamental** and
the deployable efficiency lever is the **router**, not best-of-N. The 3 open sets (vqa_rad/slake/pathvqa)
use exact-match/judge `oks`; **PMC uses loose option-letter `oks`** (PMC-specific numbers below are
indicative only).

### 5.1 Compounding FAILS — diverse-generation and pairwise-selection do NOT stack

`combine_diverse_pairwise.py`: a 2×2 {pointwise, pairwise-Copeland over REAL A-vs-B verdicts} × {iid@8,
diverse@15}, pooled over 3 open sets (n=1023), paired 3000-sample question bootstrap.

| selector | iid@8 | diverse@15 |
|---|---:|---:|
| pointwise | 0.5191 | **0.5494** |
| pairwise (Copeland) | 0.5396 | 0.5376 |
| oracle | 0.6452 | 0.6813 |

Each lever alone beats the pointwise-iid baseline: **diverse-lever B−A = +0.0303** (CI [+0.0088,+0.0518],
sig), **pairwise-lever C−A = +0.0205** (CI [+0.0098,+0.0323], sig). **But they do NOT compound:**
pairwise-over-diverse (D=0.5376) is **≤** pointwise-over-diverse (B=0.5494) — `D−B = −0.0117` (CI
[−0.0283,+0.0049]) — and the both-levers gain `D−A = +0.0186` is **not significant** (CI [−0.0020,+0.0411]).
On PMC (n=600) diverse lifts the **oracle** +0.110 (0.505→0.615) but the selectors convert almost none of it
(pointwise-div 0.305 vs iid 0.290 → converted 0.015; pairwise-div 0.295 → converted 0.005). **Diversity buys
coverage, not selectability.** This resolves the §4 open "pairwise-over-diverse" question — negatively.
Artifact: `combine_diverse_pairwise.json`.

### 5.2 Distractor-filtering FAILS — no pre-filter beats plain diverse-pointwise

`distractor_filter.py`: 8 filters (drop-lone-confident, consensus, rarity, top-k-agreed) over the diverse
pool, using only candidate text + verifier score + cross-candidate agreement (never `oks`). Pooled-3ds
(n=1023): baseline a = unfiltered-diverse-pointwise **0.5494**, baseline b = iid@8-pointwise **0.5191**.
Best filter **rarity_log1p = 0.5601**: **vs a +0.0108 (CI [−0.0059,+0.0293], n.s.)**, vs b +0.0411 (CI
[+0.0235,+0.0596], sig). **No filter beats BOTH baselines** (`any_filter_beats_both=false`), and
rarity_log1p **sign-flips per dataset** vs diverse-pointwise (slake +0.0279, vqa_rad −0.015, pathvqa
−0.0225, pmc −0.0083). **Mechanism: the correct *new* answers diverse generation adds are themselves rare,
so a rarity/agreement signal cannot separate correct-rare from wrong-rare.** Artifact: `distractor_filter.json`.

### 5.3 Verifier CAPACITY does not break the wall — 32B-zeroshot ties 7B-trained

`verifier_32b_gpu.py` (GPU verdict dumps) + `verifier_32b_measure.py`. Pooled n=600 (vqa_rad, slake,
pmc_content), selecting over the Lingshu-7B diverse pool; oracle_distinct = 0.672.

| verifier | sel_acc | note |
|---|---:|---|
| 7B-trained (pooled4 LoRA) | 0.475 | current deployed |
| 7B-zeroshot (base) | 0.413 | capacity floor |
| 32B-zeroshot (base) | 0.480 | 7× capacity |

**32B-zeroshot vs 7B-trained = +0.005** (CI [−0.023,+0.032], **n.s.**) — a 7× bigger verifier does **not**
beat the small *trained* one. The pure-capacity contrast **32B-zeroshot vs 7B-zeroshot = +0.067** (CI
[+0.038,+0.095], sig) is real but small, and the 32B still leaves an **oracle→selection gap of 0.192**
(conversion 0.15). **The selectability ceiling is substantially FUNDAMENTAL, not a verifier-capacity
artifact** — a 32B recovers ~7 of the ~19 oracle-gap points. Artifact: `verifier_32b_gpu.json`.

### 5.4 End-to-end consolidation — best-of-N is Pareto-DOMINATED on FLOPs

`end_to_end_consolidation.py`. Pooled n=1023, strong=judge_ok, FLOPs in FLOP-eq (× one 7B forward). Acc
ladder: 7B-greedy 0.518 ≈ iid-bo8 0.519 < diverse-bo15 0.549 ≪ **always-32B 0.673**. Costs: **always-32B
F=4.57**, iid-bo8 **F=16**, diverse-bo15 **F=30**. Global FLOPs-Pareto envelope:

| operating point | FLOPs (FLOP-eq) | acc |
|---|---:|---:|
| 7B-greedy | 1.00 | 0.518 |
| iid→Pandora | 2.00 | 0.519 |
| iid→Pandora | 2.89 | 0.542 |
| iid→Pandora | 3.25 | 0.568 |
| always-32B | 4.57 | 0.673 |

**diverse-gen is NOT on the envelope** (`diverse_on_pareto_envelope=false`) — iid→Pandora Pareto-dominates
diverse→Pandora at every accuracy target (@0.55 iid F=3.3 vs div F=6.2; @0.65 iid F=8.0 vs div F=10.9), so
diverse-gen's 1.875× generation cost is never repaid. **The deployable envelope is
`greedy → 7B+Pandora → always-32B`; the win is Pandora/router, not diverse generation.** The crux the
artifact names: **open-text always-32B is CHEAP (F=4.57 = one 32B forward) AND more accurate than any 7B
best-of-N leg**, so escalation dominates spending budget on cheap draws. Artifact:
`end_to_end_consolidation.json`.

### 5.5 Latency re-examination — best-of-N is latency-ALIVE but still does NOT beat always-32B

`latency_reexamination.py`. Real measured batch-1 latency (HF, cap320, NVML): **GEN7 = 347.1 ms, VER7 =
175.5 ms, GEN32 = 665.0 ms**. A **parallel** best-of-N base costs only GEN7+VER7 = **522.6 ms = 0.79× the
single 32B forward (665 ms)** — the exact opposite of the FLOPs verdict (iid-bo8's 16 FLOPs = 3.5× the 32B's
4.57). **FLOPs-dominated yet latency-cheaper**, because batch-1 short-gen is overhead-bound (32B only ~1.9×
the 7B wall-clock, not 4.57×) and best-of-N parallelises N away. So best-of-N **survives on the
latency-Pareto envelope** as a real low-latency/lower-accuracy point (`latency_alive = YES`). **But it does
NOT beat always-32B** (`beats_always32 = NO`): the fixed-bo-N accuracy ceiling is ~0.549 (gated-diverse tops
~0.587 before it escalates), far below the 32B's 0.673, and matching 0.673 forces heavy escalation that
pushes parallel latency **above** 665 ms (iid-bo8+gate 1141 ms @esc93%; diverse-bo15+gate 1188 ms @esc100%).
**always-32B owns the high-accuracy corner because its open-text no-think generate (665 ms, F=4.57) is both
cheap and fast.** Artifact: `latency_reexamination.json`.

> **The load-bearing assumption, flagged here for July 7.** §5.4/§5.5 both hinge on "the strong open-text
> 32B forward is CHEAP and FAST (665 ms, F=4.57)". That 665 ms is the 32B **NO-THINK** generate. The
> artifact itself carries the scope caveat: *"this verdict is conditioned on a cheap, fast strong model; in
> an expensive or slow-strong regime (one strong forward ≫ N cheap draws, or latency-bound), the parallel
> best-of-N leg could still win — that regime is not measured here."* The paper's headline deployment-cost
> problem is the **32B-THINK** (~11 s / ~6 kJ), not the 665 ms no-think leg — which is exactly the
> re-grounding July 7 opens.

### 5.6 Re-grounding conclusion (end of the best-of-N program)

The best-of-N outcome-verifier direction is now **scientifically characterized but NOT the deployable method
in this setting.** Characterized: the **selectability wall is fundamental** — it resists compounding (5.1),
pre-filtering (5.2), and even 7× verifier capacity (5.3, only ~7 of ~19 oracle-gap points recovered). Not
deployable *here*: because the **Lingshu-32B strong leg is cheap on BOTH axes** (4.57 FLOP-eq, 665 ms
no-think), escalating to it dominates spending budget on more/diverser cheap draws (5.4) and best-of-N can't
beat it on latency either (5.5). **The deployable efficiency win is therefore the ROUTER** —
`greedy → 32B` (§1.4: 48–82% of always-32B FLOPs at parity) — **plus Pandora for tight compute budgets**
(F=2–3.3 for acc 0.52–0.57). **Scope caveat (carried, and the seed of July 7):** conditioned on a *cheap,
fast* strong model.

## 6. Offline Pandora refinements — two regime-limited additions

### 6.1 Correlated-Pandora (diversity-discounted stopping)

`pandora_correlated.py` extends the Weitzman rule so the marginal value of one more 7B draw is **discounted
by the diversity of answers already drawn** (c_eff = c_cheap / ρ, ρ = Simpson diversity of the drawn
predictions; entropy and novelty variants also run). Held-out, same 11 (family × open-dataset) configs as
§1.1.

| target | variant | FLOPs | meanN | latency (seq) | energy | datasets covered |
|---|---|---:|---:|---:|---:|---:|
| **iso-strong** | independent Pandora | 6.32 | 3.03 | 1,619 ms | 222.6 J | 11/11 |
| | **correlated (Simpson, primary)** | **5.32 (−16%)** | **2.52 (−17%)** | **1,357 ms (−16%)** | **187.0 J (−16%)** | 11/11 |
| **iso-bo8** | independent Pandora | 11.74 | 5.38 | 2,951 ms | 409.8 J | 9/11 |
| | correlated (Simpson) | 12.31 | 5.27 | 3,010 ms | 424.0 J | **7/11** |

**Read.** At **iso-strong**, diversity-discounted stopping is a **free ~16–17% cut** in meanN / FLOPs /
latency / energy at full 11/11 coverage (the novelty measure goes further, ~−19 to −20%). But at
**iso-bo8** (the §1.1 headline target) it does **NOT** help — coverage drops to 7/11 and FLOPs rise —
because stopping-on-agreement caps the attainable ceiling below the cheap best-of-8 ensemble. **A genuine
but regime-limited refinement** — use it for the (easy) 32B-parity operating point, not the cheap-ensemble
ceiling. Artifact: `pandora_correlated.json`.

### 6.2 Pandora × cross-model pooling combo

`pandora_pooling_combo.py` runs correlated-Pandora over a *pooled* candidate stream (round-robin /
residual-specialist across the 3 generators), on the 3 all-generator open sets. Pooling again **lifts the
oracle ceiling substantially — +0.113 to +0.150** (oracle@8 single-best → pool-8-each: kvasir 0.593→0.731,
radimagenet 0.512→0.625, vqa_rad 0.63→0.78) — **but there is no frontier gain**: at both iso-strong and
iso-bo8 the pooled-Pandora variants are **no cheaper** than single-model correlated-Pandora (iso-strong:
single-model correlated-Pandora ~2.92 FLOPs / ~1.42 meanN at 3/3 coverage, strictly under every pool variant
incl. pool-residual 3.15 FLOPs). **The selection wall (limit #2) blocks the conversion** — the extra oracle
coverage pooling buys is exactly what the pointwise verifier can't turn into selected accuracy (same
mechanism as §4.1 on PMC-content). **Single-model correlated-Pandora stays the best cheap-leg controller**
until a stronger selector (the §4.2 real pairwise verifier) is wired into the controller. Artifact:
`pandora_pooling_combo.json`.

## 7. Status at end of July 6

The ledger (`RESEARCH_RESULTS_2026-07.md`, 18:56) consolidates all of the above. Net for the cycle:

- **Validated positives:** format-router (§1.4, deployable −18…−52% FLOPs at parity), Pandora controller
  (§1.1, −27% FLOPs/energy at iso-bo8 held-out), cross-model pooling (§1.2, +0.05–0.13 oracle), diverse
  *generation* (§4.1, oracle +0.027/+0.064 → +0.025 verifier acc), real pairwise verifier (§4.2, +0.036
  sel_acc, overturns the simulation).
- **Honest negatives / characterizations:** UGV-for-MCQ (§2), the three round-2 post-hoc selectors (§3),
  and the five-experiment selectability battery (§5) — the selection wall is **fundamental** and best-of-N
  is **Pareto-dominated by escalation** *when the strong model is cheap+fast*.
- **The one assumption everything downstream rests on:** the strong open-text leg is the **665 ms no-think
  32B**. The paper's motivating expensive baseline is the **~11 s / ~6 kJ 32B-think**. Re-grounding the
  comparison against the *right* strong model is the July 7 investigation.
