> # ⚠️ NUMERICALLY SUPERSEDED — annotated 2026-07-29
>
> **The mechanism described in this document is correct. Its headline numbers are not.** It was written
> before three things that changed them: the **oracle-mode-32B baseline** (2026-07-08 08:18), the decision to
> **exclude MMMU** after the contamination audit ("Variant B", 08:24–09:43), and the replacement of the
> **estimated** 32B-reasoning open-text cells with **measured** ones (10:41), plus the headline CI computed
> 2026-07-09.
>
> **Canonical values (Variant B = MMMU excluded, n = 42,224, measured, CI-certified):**
> always-32B-with-reasoning baseline **0.5591**; compute-lean **0.5741, +0.0150 [+0.0107, +0.0192]** at 0.49x;
> **accuracy-max 0.5836, +0.0245 [+0.0216, +0.0274]** at 0.93x; accuracy-max-fusion 0.5862, +0.0271 at 1.25x.
> Sources: `artifacts/f8_mode_vsthink_ci.json`, `artifacts/opentext_32b_think_full.json`.
>
> **Also corrected since:** §7.7 takeaway #1 still states "compute-lean +0.0123 @ 0.49x; accuracy-max +0.0212
> @ 0.93x" as the deployable headline — read **+0.0150** and **+0.0245** with the CIs above. This file also
> says the backlog holds 56 ideas; it holds **68**.
>
> **Also corrected 2026-07-29 — the cross-family reasoning-vs-direct numbers in §5 (and every claim that
> "reasoning helps reasoning-heavy benchmarks across 5 families").** The think and direct arms were
> **prompt-unmatched**. Re-derived from the best-matched arms on disk
> (`artifacts/finding1_corrected_2026-07-29.json`; audit `artifacts/finding1_prompt_matching_audit.json`):
> perception is **17/20** strictly negative (14/20 CI-significant, pooled **−0.0401 [−0.0456, −0.0347]**,
> n = 30,250) — *stronger* than the 15/20 published; the reasoning half is **model-dependent, not
> universal** (12/15 point-positive, only **4/15** CI-significant, **1/15 significantly negative**);
> **all 7 Lingshu-32B cells and QoQ-Med-VL-32B's reasoning cells are WITHDRAWN**; **MedGemma-27B's
> PathVQA +0.0413 [+0.0220, +0.0607] is a real exception**; and the **open-text** think-vs-direct
> comparison is **provisional** pending a matched-prompt re-run. See §5.2 below and retrospective
> §5.1 / §10.1 C20–C25.
>
> **The definitive account is [`PROJECT_RETROSPECTIVE_2026-07-29.md`](PROJECT_RETROSPECTIVE_2026-07-29.md)**
> — §4 for the corrected results, §7 for this method's 16 known holes, §10.3 for the full decode of the
> `+0.02xx` number family. **Read this file for *how it works*, not for *what it scores*.**

# Method-research results ledger — 2026-07 cycle

> **What this is.** An honest, sourced ledger of the method-research results produced this cycle for the
> unified test-time-compute cascade (cheap 7B/8B medical VLM + trained best-of-N outcome verifier +
> confidence gate, cascading to a strong 32B/38B; scored on BOTH accuracy AND compute/latency/energy).
> This is **documentation, not a paper** — nothing here touches `paper/`.
>
> **No-fabricated-numbers rule.** Every figure below is copied from, or arithmetically derived from, an
> artifact/code file that was read. Provenance is cited inline. Source artifacts + code (all under
> `results/cascade_methods/artifacts/` and `src/cascade_methods/`):
> - `pandora_controller.json` ← `pandora_controller.py`
> - `diversity_candidates.json` ← `diversity_candidates.py`
> - `generator_portfolio.json` ← `generator_portfolio.py`
> - `unified_router.json` ← `unified_router.py`
> - `gate_unified_bakeoff.json` ← `gate_unified_bakeoff.py`
> - `open_bestofN_adaptive.json` (adaptive-N reference baseline)
> - `ugv_mcq_verdict.json` ← `ugv_mcq_verdict.py` (round-2 UGV resolution)
> - `active_comparison_verifier.json` ← `active_comparison_verifier.py`
> - `bandit_allocation.json` ← `bandit_allocation.py`
> - `dawid_skene_aggregate.json` ← `dawid_skene_aggregate.py`
> - `diverse_generation_gpu.json` ← `diverse_measure_gpu.py` (scores the GPU diverse-generation candidate dumps)
> - `pairwise_verifier_gpu.json` ← `active_comparison_verifier.py --real_verdicts_dir ckpts/pairwise` (over the
>   real A-vs-B verdicts dumped by `pairwise_verifier_score.py`)
> - `pandora_correlated.json` ← `pandora_correlated.py`
> - `pandora_pooling_combo.json` ← `pandora_pooling_combo.py`
> - `combine_diverse_pairwise.json` ← `combine_diverse_pairwise.py` (2×2 pairwise×diverse compounding; §2.6)
> - `distractor_filter.json` ← `distractor_filter.py` (§2.6)
> - `verifier_32b_gpu.json` ← `verifier_32b_measure.py` (scores the 32B/7B-zeroshot verdict dumps from `verifier_32b_gpu.py`; §2.6)
> - `end_to_end_consolidation.json` ← `end_to_end_consolidation.py` (§2.6)
> - `latency_reexamination.json` ← `latency_reexamination.py` (§2.6)
> - idea backlog: `results/cascade_methods/METHOD_IDEAS_BACKLOG.md`
> - reproduction fallback: `results/cascade_methods/docs/current/OMNIMED_FALLBACK.md`
>
> Most experiments this cycle were **OFFLINE / CPU-only** re-simulations over existing per-sample dumps
> (no GPU, no new inference). The two exceptions are the GPU passes in §2.3 (diverse-generation candidates +
> real pairwise A-vs-B verdicts) — **now RESOLVED**. The compounding pairwise-over-diverse run and the follow-on
> **selectability-wall battery** (compounding, distractor-filtering, 32B verifier capacity, end-to-end FLOPs +
> latency consolidation) are **RESOLVED in §2.6 (2026-07-06)** — the best-of-N program is now characterized and
> re-grounded (the deployable lever is the router, not best-of-N; see the §2.6 Re-grounding conclusion).

---

## 1. Validated results

### 1.1 Format-router (Method C) — a deterministic unified router, Pareto over always-32B

**What.** One deterministic router (`unified_router.py`) over a MIXED MCQ + open-text stream: detect each
item's answer format **from the prompt text only** (never the gold), dispatch MCQ/closed → logprob-margin
cascade, open → trained best-of-8 verifier cascade, then score the whole pooled workload as ONE
accuracy-vs-cost point against the always-strong (32B/38B) baseline. Operating point = honest 50/50
stratified calib/test over 20 seeds; thresholds picked on calib to hold pooled accuracy ≥ pooled
always-strong, evaluated on test. FLOPs = prefill-inclusive `2N(P+G)`; best-of-8 counted at the full 8×
cheap passes (conservative — the prefix-shared column is the deployable lower bound). Source:
`unified_router.json` (fields are fractions of the always-strong baseline).

| Family | pooled n (MCQ+open) | always-strong acc | router acc | Δacc | FLOPs (of 32B) | FLOPs prefix-shared | latency (proxy) | esc MCQ / open |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Lingshu** | 44,322 (39,422+4,900) | 0.546 | 0.549 | **+0.003** | **47.7%** (−52%) | 30.0% (−70%) | 65.7% | 7.6% / 0.0% |
| **MedVLThinker** | 42,822 (39,422+3,400) | 0.522 | 0.522 | **+0.000** | **60.0%** (−40%) | 47.3% (−53%) | 84.4% | 25.7% / 4.8% |
| **InternVL3** | 46,179 (39,422+6,757) | 0.512 | 0.514 | **+0.001** | **82.1%** (−18%) | 58.7% (−41%) | 57.3% | 39.6% / 10.7% |

**Read.** The router matches-or-beats always-strong accuracy on all three families (Δacc ≥ 0) while spending
**48–82%** of always-32B FLOPs (30–59% if the image prefill is prefix-shared across the 8 samples). The FLOPs
saving tracks how competitive the strong model is: Lingshu's 32B is barely ahead of its cheap leg → the router
keeps-cheap (7.6% MCQ escalation) → biggest saving; InternVL3's 38B is more competitive → more escalation
(39.6%) → smallest saving.

**Honest caveats.** (i) The latency column is an **amortized-batch `latency_s` proxy**, not a batch-1
wall-clock measurement (a real batch-1 latency GPU job is queued, per the code header); it is non-monotone with
FLOPs (InternVL3 shows lower latency-fraction than MedVLThinker despite higher FLOPs) because the two proxies
use different backbone latency ratios. (ii) Open-format MedEvalKit items are excluded (not double-counted) from
the MCQ pool and the open contribution is measured on the dedicated open-eval sets — this is logged, not hidden.
(iii) Δacc is within a few tenths of a point — the claim is **iso-accuracy at lower compute**, not an accuracy lift.

---

### 1.2 Pandora's-Box adaptive controller (Weitzman) — both-axes win at iso-bo8 accuracy

**What.** `pandora_controller.py` implements the Weitzman (Econometrica 1979) optimal-search rule as a single
controller that **unifies adaptive-N and the escalation gate**: each "box" is either "draw one more 7B sample"
(cost 2.0 FLOP-eq = GEN7+VER7, reward = the verifier's calibrated P(correct)) or "escalate to the 32B" (cost
4.57 FLOP-eq, deterministic reward = calibrated P(strong correct)). One exchange-rate knob λ yields BOTH a
stop-drawing threshold and an escalation threshold. **Thresholds are held-out (5-fold cross-fit isotonic
calibration; no peek at correctness); the baselines' τ are swept on full data (optimistic "oracle-τ").** Cost
model = the project's canonical measured batch-1 model. Headline = per-domain-tuned aggregate, n-weighted over
11 (family × open-dataset) configs (`pandora_controller.json → SUMMARY_per_domain_tuned`).

| Target | Method | datasets covered | FLOPs | vs bo8 | energy (J) | meanN | esc | lat_seq (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **iso-bo8** (match cheap-ensemble acc) | **Pandora (held-out)** | 9/11 | **11.74** | **−27%** | **409.8** (**−28%**) | 5.38 | 21.6% | 2,951 |
| | adaptive-N (oracle-τ) | 11/11 | 13.01 | −19% | 459.5 | 6.31 | 8.7% | 3,350 |
| | verifier-bo8 + gate (oracle-τ) | 11/11 | 16.00 | 0% | 568.8 | 8.00 | 0.0% | 4,176 |
| **iso-strong** (match always-32B acc) | **Pandora (held-out)** | 11/11 | 6.32 | −61% | 222.6 | 3.03 | 5.5% | 1,619 |
| | adaptive-N (oracle-τ) | 11/11 | 10.84 | −32% | 383.6 | 5.30 | 5.4% | 2,802 |

**Headline.** At **iso-bo8 accuracy, held out**, Pandora reaches the target at **−27% FLOPs / −28% energy** vs
fixed best-of-8, and beats even the **optimistically-tuned** adaptive-N (−19%) and gate (0%) baselines despite
being the only method whose thresholds are held-out. (Reference costs: `bo8` = 16.0 FLOP-eq / 568.8 J;
`always-32B` = 4.57 FLOP-eq / 127.0 J.)

**The reframe (important).** On these open-text OOD workloads the **accuracy ceiling worth matching is the
cheap best-of-8 ensemble, not the 32B.** Pooled Lingshu: bo8 = **0.414** vs always-32B = **0.331**; the strong
model is a *weaker* accuracy target here, so escalating to it buys little and the real lever is **reaching bo8
accuracy more cheaply.** This is why the "iso-bo8" row is the headline and "iso-strong" is a footnote: matching
the 32B is easy (Pandora does it at 6.3 FLOPs) but uninteresting.

**Honest caveats.** (i) On **2 of 11** configs Pandora's held-out frontier falls just short of bo8 parity
(covered 9/11 at iso-bo8) — where bo8 ≈ the oracle ceiling, a controller that stops early cannot always match
it. (ii) **Latency trade is real and adverse on one axis:** adaptive drawing is inherently sequential
(draw→check→draw), so Pandora's `lat_seq` ≈ 2,951 ms, whereas a fixed bo8 can batch its 8 draws in parallel
(`lat_bat` ≈ 522 ms). Pandora wins FLOPs + energy but is **slower in batch-1 wall-clock than a batched bo8** —
the code reports both and flags this. (iii) Weitzman assumes independent box rewards; within-question samples
are correlated (re-simulation over recorded draw order mitigates but does not eliminate this).

---

### 1.3 Cross-model candidate pooling (generator portfolio / Markowitz) — pooling helps, allocation doesn't

**What.** `generator_portfolio.py` treats each cheap generator {Lingshu-7B, MedVLThinker-7B, InternVL3-8B} as
a Markowitz "asset" (return = per-sample accuracy, covariance = error-correlation φ) and, for a fixed sample
budget B, allocates B samples across generators to maximise **oracle@B** (does a correct answer appear among
the drawn candidates — the project's #1 binding limit). Coverage = a with-replacement pass@k estimator applied
identically to all methods; the Markowitz allocation is fit on a train fold and scored held-out (5-fold CV).
Source: `generator_portfolio.json`. Mean off-diagonal error-φ ≈ **0.52–0.56** (models fail on somewhat
different questions, but errors are still positively correlated).

Held-out oracle@B, pooled over the 3 all-generator datasets (kvasir + radimagenet + vqa_rad, n=3,400):

| B | single-best (Lingshu) | uniform pool | portfolio (Markowitz) | Δ portfolio vs single | Δ portfolio vs uniform |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.359 | 0.395 | 0.415 | **+0.056** | +0.020 |
| 4 | 0.421 | 0.481 | 0.486 | **+0.065** | +0.005 |
| 8 | 0.467 | 0.545 | 0.547 | **+0.080** | +0.002 |
| 16 | 0.498 | 0.600 | 0.603 | **+0.105** | +0.002 |

**Read.** Per-dataset, pooling 3 models beats the best single model by **+0.045 to +0.127 oracle** (B=8–16;
e.g. vqa_rad +0.108 @B=8, +0.127 @B=16), pooled **+0.08 (B=8) / +0.11 (B=16)** — a real, held-out lift of the
accuracy ceiling. **But the Markowitz-optimal allocation ≈ a naive uniform split** (Δ vs uniform = +0.002 to
+0.02 pooled, and **negative** on vqa_rad: −0.003 @B=8, −0.005 @B=16). **The win is diversity/pooling, not the
clever allocation.**

**Honest caveats.** (i) This is an **oracle-ceiling** result; a trained verifier realises only ~74–82% of it
(the selection limit, §3). (ii) No temperature variants exist on disk → assets = the 3 models only (a real
portfolio would add temps/prompts). (iii) φ estimates are per-domain noisy; a weak generator can dilute the
verifier's job (InternVL3-8B on pathvqa is near-floor, so pathvqa is a 2-generator supplement only).

---

### 1.4 Diversity-maximized candidate selection (DPP/MMR) — a rate win, not a ceiling lift

**What.** `diversity_candidates.py` tests whether diversity-aware **selection** (greedy farthest-first /
MMR ≈ DPP-MAP over answer-string distance) reaches the oracle answer in fewer samples than iid draw-order.
**Honest by construction:** reordering a FIXED set of 8 candidates **cannot** raise oracle@8 — only diverse
*generation* (a GPU job, now **RESOLVED — §2.3(a)**) can. Source: `diversity_candidates.json`,
pooled across all families/datasets (n=15,057).

| Ordering | oracle@8 | AULC (area under oracle@k) | k_reach (mean #samples to first correct) |
|---|---:|---:|---:|
| draw (iid) | 0.412 | 0.350 | 2.20 |
| random subset | 0.412 | 0.350 | 2.20 |
| MMR (exact-string) | 0.412 | 0.365 | 1.91 |
| **MMR (token-Jaccard)** | 0.412 | **0.368** | **1.86** |

**Read.** oracle@8 is **identical (0.412) across all orderings — confirming no ceiling lift.** MMR selection
reaches the first correct answer with **k_reach 2.20 → 1.86 = −15.6% fewer samples** (a rate/efficiency win).
Redundancy headroom: on average only **4.67 of 8 candidates are distinct** (≈42% of iid samples are exact
duplicates), so an idealized perfectly-diverse *generator* could hit the same oracle@8 with ~4.7 samples
(up to **−42%** samples at fixed coverage — an upper bound, realizing it needs the GPU diverse-generation pass
and may inject confident-but-wrong distractors).

**Honest caveat.** The 42% "headroom" is an idealized upper bound over the existing answer set; the offline
test can only prove the −15% selection-rate win, not the ceiling lift. **→ Now resolved (§2.3(a)):** the GPU
diverse-*generation* pass **does** lift the ceiling (oracle +0.027 at matched budget / +0.064 at M=15) and, as
predicted here, injects confident-but-wrong distractors (worst on PMC-content).

---

### 1.5 Gate bake-off — cascade quality tracks recoverability-AUROC, not detection-AUROC

**What.** `gate_unified_bakeoff.py` benchmarks ~14 escalation gates in BOTH settings (MCQ: 7B-nt@cap320 logprobs
→ 32B-think; OPEN: Lingshu-7B bo8+verifier → 32B), all on existing checkpoints, trained gates using 5-fold
out-of-fold scores. Metrics: `AUROC_detect` (does the gate detect cheap errors), `AUROC_recover` (Jitkrittum
recoverability — can the score tell which cheap errors the strong model will fix), `ADC` (threshold-free area
under the deferral curve = cascade quality), `routing_eff` (APGR, 0–1). Deployed gates: MCQ = **margin**,
OPEN = **verifier-conf** (both are "confidence = P(correct)" signals). Source: `gate_unified_bakeoff.json`.

Selected gates, **MCQ competent-4** (n from checkpoints; cheap 0.622 / strong 0.645):

| gate | AUROC_detect | AUROC_recover | ADC | routing_eff |
|---|---:|---:|---:|---:|
| **margin (DEPLOYED)** | 0.661 | 0.587 | 0.6445 | 0.088 |
| 7B self-verify (AutoMix) | 0.643 | **0.614** | **0.6461** | 0.100 |
| maxprob (MSP/Chow) | 0.677 | 0.553 | 0.6432 | 0.076 |
| neg-entropy | 0.686 | 0.515 | 0.6413 | 0.062 |
| learned-RICH (fused) | **0.693** | 0.533 | 0.6430 | 0.075 |
| learned-gbm (logprob feats) | 0.681 | 0.506 | 0.6407 | 0.058 |

Selected gates, **OPEN pooled** (cheap 0.414 / strong 0.331):

| gate | AUROC_detect | AUROC_recover | ADC | routing_eff | vs deployed (ADC, 95% CI, P>0) |
|---|---:|---:|---:|---:|---|
| **verifier-conf (DEPLOYED)** | 0.853 | 0.434 | 0.3901 | 0.198 | — |
| Jitkrittum Diff-Prob | 0.744 | 0.452 | **0.3965** | 0.271 | +0.0062 [+0.0040,+0.0086] P=1.00 **SIG+** |
| EG-RC (ours) | 0.782 | 0.412 | 0.3931 | 0.232 | +0.0028 [+0.0006,+0.0048] P=0.99 **SIG+** |
| learned-gbm (verif+cheap) | **0.856** | 0.439 | 0.3899 | 0.197 | −0.0003 [−0.0013,+0.0007] P=0.28 |
| self-consistency | 0.690 | 0.434 | 0.3668 | −0.063 | −0.0227 [−0.026,−0.019] sig− |

**Read (the finding).** Cascade quality (ADC) **tracks recoverability-AUROC, not detection-AUROC.** On MCQ, the
highest-detection gates (learned-RICH 0.693, neg-entropy 0.686, maxprob 0.677) have **lower** ADC than the
deployed margin gate; the highest-ADC gate is 7B self-verify — the one with the highest *recoverability* AUROC
(0.614). The simple **verifier-P(correct) / margin confidence gate is the format-agnostic near-optimal deployable
gate**: no high-detection gate beats it on either setting. The only significant challengers are
**recoverability-aware** scores (Jitkrittum Diff-Prob = P(strong)−P(cheap), and our EG-RC), and even those win
only **+0.003 to +0.006 ADC** on OPEN-pooled — confirming the mechanism (ADC ← recovery signal) while showing
the practical edge is tiny.

**Honest caveats.** (i) `min_esc_parity` and the deferral-curve oracle are computed on the eval pool = optimistic
(flagged in the code). (ii) The recoverability-aware wins are statistically significant but operationally small
(≤0.006 ADC) and did not reproduce as a *detection* improvement — consistent with the recoverability wall (§3).

---

## 2. Round-2 resolutions + open

### 2.1 UGV — single generative-verifier, MCQ-as-generation — **RESOLVED (negative for MCQ)**

**What.** The unified generative-verifier (UGV, backlog B2 — the project's stated frontier) scores MCQ options
as *generated answers* through one generative grounding verifier (the idea: unify MCQ + open-text + boxes under a
single verifier, attacking the selection ceiling, §3 limit #2). The data-loader fix landed and the experiment ran
on Lingshu-7B (+ MedVLThinker-7B) over PMC-VQA and MedXpert (n=2,000 each), self-consistency N=8. **Two scoring
modes:** `content` (options hidden from the prompt → the model must *generate* the answer string) vs `letter`
(standard A/B/C/D letter-logprob); `strict` = strict greedy parse. Source: `ugv_mcq_verdict.json` ←
`ugv_mcq_verdict.py` (over `ckpts/mcq_gen_verify/` dumps).

**Verdict — the no-router single generative verifier does NOT work for MCQ.**

1. **Content-mode collapses MCQ accuracy.** Hiding the options and forcing free-text generation craters greedy
   accuracy: PMC-VQA strict greedy **0.132 content vs 0.534 letter** (Δ **−0.394**); MedXpert **0.499 content vs
   0.556 letter**. The MCQ signal lives in the option set — discarding it to "unify" formats destroys the very
   thing that makes MCQ tractable.
2. **The verifier's gain on content-MCQ is negligible.** Mean content-MCQ verifier-boN gain = **+0.004** (strict),
   AUROC ≈ **0.70** — the generative verifier barely reranks its own weak content candidates (PMC content strict
   +0.009; MedXpert −0.004).
3. **Letter-mode + verifier-bo-N is inconsistent / label-sensitive.** PMC letter strict verifier-boN gains
   **+0.082** (0.534→0.616), but MedXpert letter strict is **flat (−0.001)** — the "gain" doesn't transfer across
   datasets and flips sign under the as-run (non-strict) parse (PMC letter as-run verifier_gain **−0.074**). Not a
   reliable lever.

Selected numbers (Lingshu-7B, strict greedy parse, n=2,000/dataset):

| dataset | mode | greedy | verifier-boN | oracle-boN | verifier gain | AUROC(score vs ok) |
|---|---|---:|---:|---:|---:|---:|
| PMC-VQA | content | 0.132 | 0.140 | 0.300 | +0.009 | 0.793 |
| PMC-VQA | **letter** | **0.534** | **0.616** | 0.800 | **+0.082** | 0.540 |
| MedXpert | content | 0.499 | 0.494 | 0.800 | −0.004 | 0.498 |
| MedXpert | **letter** | **0.556** | 0.556 | 0.843 | −0.001 | 0.478 |

(pooled over all content-MCQ runs: mean verifier gain **+0.004**, mean AUROC **0.70**.)

**Conclusion.** The **router stays decisively the better method**: MCQ → letter + margin gate (§1.1/§1.5),
open-text → trained best-of-N verifier. The single generative verifier's genuine home is **open-text** (where it
delivers the §1.2/§1.5 wins), not MCQ. Backlog B2 is closed as a *negative for MCQ*.

### 2.2 Round 2 offline experiments — three honest negatives

Three further CPU-only re-simulations over existing dumps, each testing a post-hoc trick against the two walls
(§3); all three are honest negatives.

**(a) Active pairwise-comparison verifier (C9 / active-IDS)** — `active_comparison_verifier.json` ←
`active_comparison_verifier.py`. Simulates pairwise verdicts from the pointwise verifier scores
(P(i≻j)=σ(logit sᵢ−logit sⱼ); **no real pairwise dumps exist**). Result: active-IDS **cannot beat
pointwise-argmax** on selection accuracy (mean active−pointwise sel_eff = **−0.003**, `beats_pointwise=false`),
and the noise-free **round-robin ceiling equals pointwise (+0.000)** — confirming the simulated pairwise
preference carries **no information beyond the pointwise score**. Its only edge is **cost**: active-IDS
matches/dominates knockout selection (sel_eff 0.801 vs knockout 0.788) at ~**6.2/28 comparisons ≈ 22%** of a full
pairwise pass (Lingshu). **Honest limit → RESOLVED (§2.3(b)):** the real GPU pairwise pass **overturns this
simulated verdict** — a *real* A-vs-B verifier **beats** pointwise-argmax on selection (+0.036 sel_acc /
+0.076 eff). The parity seen here was an artifact of deriving pairwise preferences from the pointwise scores,
which cannot carry information the pointwise head lacks.

**(b) Bandit / adaptive allocation (C7)** — `bandit_allocation.json` ← `bandit_allocation.py`. Per-question
adaptive allocation (Thompson-soft, UCB-E) of the sample budget across the 3 generators, reward = verifier score,
held-out 5-fold. Result: adaptive allocation **≈ fixed uniform pooling** for oracle@B (best pooled held-out
**Δ=+0.002** over B∈{2,4,8}); Thompson/UCB-E track uniform, sometimes *losing* to exploration cost — same verdict
as Markowitz (§1.3, Δ=+0.020). The de-biased split-oracle ceiling shrinks to Δ=+0.052 and **inverts at B=8
(−0.026)** — single-arm concentration can't match uniform's cross-arm coverage. Uniform pooling is already
near-optimal; the verifier score is too weak to capture what little per-question headroom exists (the selection
wall again).

**(c) Unsupervised answer aggregation (Dawid–Skene)** — `dawid_skene_aggregate.json` ← `dawid_skene_aggregate.py`.
Grouped one-coin Dawid–Skene EM over the pooled 3-generator answers (per-source reliability, no labels). Result:
guarded DS **≈ plain pooled majority** (**−0.013**), stays **+0.132 below** the trained verifier and **+0.373
below** oracle. WHY: unsupervised reliability tracks **self-agreement (~0.52), not accuracy (~0.29)** — the
generators are **confidently wrong**, so cross-source agreement carries no correctness signal and can't break the
majority trap.

**Unified takeaway.** The candidate-quality / selection wall (§3, limits #1 & #2) **resists post-hoc tricks** —
*simulated* pairwise re-ranking, adaptive allocation, and unsupervised aggregation all collapse to the
pointwise/uniform/majority baseline. What **does** move the numbers is real new signal: **better candidates**
(diverse generation, cross-model pooling — §1.3/§1.4/§2.3(a)), a **real** pairwise verifier (§2.3(b), which
overturns the *simulated* parity above), and the **trained verifier**.

### 2.3 GPU experiments — **RESOLVED** (real signal, not simulation)

Both queued GPU passes have landed. Each attacks a different binding wall (§3) with real inference, and **both
succeed** — the first structural wins that move the accuracy *ceiling* this cycle.

**(a) Diverse generation (attacks limit #1 — the coverage / oracle@N ceiling) — POSITIVE.** `diverse_measure_gpu.py`
scores real diverse candidates (a portfolio of 5 prompt personas {base, anatomy, modality, differential, concise}
× a temperature ladder {0.7, 1.0, 1.3}, M=15 draws) against the iid@8 baseline on the **same**
model/cap/verifier/scorer, with the diverse set restricted to the iid idx set. Pooled n=1,623 over 4 open sets
(vqa_rad, slake, pathvqa, pmc-content). Source: `diverse_generation_gpu.json`.

| metric (pooled) | iid@8 | diverse-DPP@8 (matched budget) | diverse-full@M=15 (extra budget) |
|---|---:|---:|---:|
| **oracle** | 0.593 | **0.621** (Δ **+0.027**, CI [0.010, 0.043]) | **0.657** (Δ **+0.064**, CI [0.047, 0.080]) |
| verifier bo-N acc | 0.434 | 0.449 (Δ +0.014, CI [−0.003, 0.031]) | **0.459** (Δ **+0.025**, CI [0.008, 0.042] **SIG**) |
| verifier bo-N eff | 0.732 | 0.723 | 0.699 |
| confident-distractor rate | 0.268 | — | 0.301 |

**Read.** Diverse *generation* (unlike diversity *selection*, §1.4) **does raise the oracle ceiling** — +0.027 at
a matched 8-sample budget and +0.064 at the full M=15 (both CIs exclude 0) — and this **converts to a significant
+0.025 verifier best-of-N accuracy** at **1.875× (=15/8) generation cost**. It is **clean on VQA-RAD** (oracle
+0.045 CI [0.01, 0.08] → verifier acc +0.06 CI [0.02, 0.105], only 1 lost-coverage question). But on
**PMC-content the oracle lift is largest (+0.110)** yet the **pointwise verifier cannot convert it**: selection
*efficiency drops* 0.574 → 0.496 because the extra diverse draws inject confident-but-wrong distractors
(confident-distractor rate 0.426 → 0.504; 93 new-coverage questions, only 11 converted, 27 lost). **KEY takeaway:
diverse generation shifts the binding limit from coverage (#1) to selection (#2)** — it buys real oracle
headroom, but cashing it in needs a stronger *selector*, which is exactly what (b) delivers.

**(b) Real pairwise verifier (attacks limit #2 — the ~74–82% selection ceiling) — POSITIVE; overturns §2.2(a).**
`active_comparison_verifier.py --real_verdicts_dir ckpts/pairwise` scores **REAL** A-vs-B verdicts (the same
Lingshu-7B + pooled4-LoRA weights as the pointwise verifier, prompted pairwise, both orders averaged for
position debias — dumped by `pairwise_verifier_score.py`). Pooled n=578 over 3 open sets (vqa_rad, pathvqa,
slake). Source: `pairwise_verifier_gpu.json`.

| selector (pooled) | sel_acc | sel_eff | cost (comparisons/q) |
|---|---:|---:|---:|
| pointwise-argmax (deployed) | 0.374 | 0.783 | 0 |
| knockout (real pairwise) | 0.405 | 0.849 | ~7 |
| **round-robin / Copeland (real pairwise)** | **0.410** | **0.859** | 28 |
| oracle@N (ceiling) | 0.478 | 1.000 | — |

**Read (overturns the simulated verdict).** The **real** pairwise verifier **beats pointwise-argmax on
selection** — sel_acc 0.374 → 0.410 (Δ **+0.036**, CI [0.016, 0.055]) and sel_eff 0.783 → 0.859 (Δ **+0.076**,
CI [0.036, 0.116]); on the **near-ties** (n=261, exactly the case pointwise loses) the gain is **+0.050** (CI
[0.012, 0.088]). **Knockout captures most of the win at ~7 comparisons/q** (sel_acc 0.405 — ~87% of the
round-robin gain at ~25% of the full 28-comparison pass). Overall this **closes ~35% of the pointwise→oracle
gap** (0.374 → 0.410 of a 0.104 gap). It **directly overturns §2.2(a)'s simulated-pairwise parity**: the earlier
"no information beyond the pointwise score" verdict was a **simulation artifact** — deriving P(i≻j)=σ(logit sᵢ−
logit sⱼ) from the pointwise scores cannot create signal the pointwise head lacks, but a *real* pairwise forward
pass carries genuine comparative information it does not.

**Synthesis + current open experiment.** The two GPU wins are **complementary and attack different walls**:
diverse generation lifts the *coverage* ceiling (limit #1) but hands the residual to *selection* (limit #2); the
real pairwise verifier is precisely the stronger *selector* limit #2 needs. **The compounding experiment — the
real pairwise verifier run over the diverse-generation candidate sets ("pairwise-over-diverse") — is now RESOLVED
(§2.6): it does NOT compound** — pairwise-over-diverse (0.5376) ≤ pointwise-over-diverse (0.5494), and the
PMC-content oracle lift the hypothesis targeted stays unconverted (pairwise converts 0.005 of a +0.110 oracle
lift). Diversity buys coverage, not selectability.

### 2.4 Reproduction status (3 families × 7 benchmarks)
Per `OMNIMED_FALLBACK.md`:
- **6/7 benchmarks** (MMMU, VQA-RAD, SLAKE, PathVQA, PMC-VQA, MedXpert): fully run cheap + strong for all 3
  families (+ think tier), faithful vs paper.
- **OmniMedVQA (7th):** cheap (7B/8B) legs **ran and reproduce the paper** — Lingshu-7B **0.827** (paper 0.829),
  MedVLThinker-7B 0.625, InternVL3-8B 0.785. The **strong (32B/38B) leg is blocked** by a deterministic
  tp=2 NCCL collective hang (ruled out chunking, monitoring-off, larger batch, timeouts over ~2 days).
  **Decision:** OmniMed is a keep-cheap benchmark (paper cheap 82.9 vs strong 83.4, a 0.5-pt gap; our cheap
  0.827 already matches), so a cascade keeps-cheap at ~0% escalation and the missing strong number changes no
  cascade conclusion. Reported as **paper-reference (Lingshu 0.834) + infra-limited**; **no fabricated
  metrics.json** is written. Net: **6/7 faithful + OmniMed cheap-faithful + strong-fallback.**

### 2.5 Offline Pandora refinements — two regime-limited additions

Two further CPU-only re-simulations extend the §1.2 Pandora controller. Both are honest, sourced, and
**regime-limited** (they help on one operating target, not the §1.2 headline one).

**(a) Correlated-Pandora (diversity-discounted stopping)** — `pandora_correlated.py`. Extends the Weitzman rule
so the marginal value of one more 7B draw is **discounted by the diversity of the answers already drawn**
(c_eff = c_cheap / ρ, ρ = Simpson diversity of the drawn predictions — the primary measure; entropy and novelty
variants also run). Intuition: stop early when repeated draws are redundant. Source:
`pandora_correlated.json → SUMMARY_per_domain_tuned`, n-weighted over the same 11 (family × open-dataset) configs
as §1.2, held-out.

| target | variant | FLOPs | meanN | latency (seq) | energy | datasets covered |
|---|---|---:|---:|---:|---:|---:|
| **iso-strong** | independent Pandora | 6.32 | 3.03 | 1,619 ms | 222.6 J | 11/11 |
| | **correlated (Simpson, primary)** | **5.32 (−16%)** | **2.52 (−17%)** | **1,357 ms (−16%)** | **187.0 J (−16%)** | 11/11 |
| **iso-bo8** | independent Pandora | 11.74 | 5.38 | 2,951 ms | 409.8 J | 9/11 |
| | correlated (Simpson) | 12.31 | 5.27 | 3,010 ms | 424.0 J | **7/11** |

**Read.** At **iso-strong** (match always-32B), diversity-discounted stopping is a **free ~16–17% cut in meanN /
FLOPs / latency / energy at full 11/11 coverage** (the novelty measure goes further, ~−19 to −20%). But at
**iso-bo8** (the §1.2 headline target) it **does NOT help** — coverage *drops* to 7/11 and FLOPs *rise* — because
stopping-on-agreement caps the attainable ceiling below the cheap best-of-8 ensemble. **Verdict: a genuine but
regime-limited refinement** — use it when the target is the (easy) 32B-parity operating point, not the
cheap-ensemble ceiling.

**(b) Pandora × cross-model pooling combo** — `pandora_pooling_combo.py`. Combines the §1.3 finding (pooling 3
cheap generators lifts oracle@N) with the Pandora controller: run correlated-Pandora over a *pooled* candidate
stream (round-robin / residual-specialist across {Lingshu-7B, MedVLThinker-7B, InternVL3-8B}), scored over the 3
all-generator open sets (kvasir, radimagenet, vqa_rad). Source: `pandora_pooling_combo.json`.

**Read.** Pooling again **lifts the oracle ceiling substantially — +0.113 to +0.150** (oracle@8 single-best →
pool-8-each: kvasir 0.593→0.731, radimagenet 0.512→0.625, vqa_rad 0.63→0.78) — **but there is no frontier gain**:
at both iso-strong and iso-bo8 the pooled-Pandora variants are **no cheaper than single-model correlated-Pandora**
(iso-strong: single-model correlated-Pandora 2.92 FLOPs / 1.42 meanN at 3/3 coverage, strictly under every pool
variant, incl. pool-residual 3.15 FLOPs). **The selection wall (limit #2) blocks the conversion** — the extra
oracle coverage pooling buys is exactly the coverage the pointwise verifier can't turn into selected accuracy
(same mechanism as §2.3(a) on PMC-content). **Verdict: single-model correlated-Pandora stays the best cheap-leg
controller**; pooling adds candidate cost without a Pareto gain until a stronger selector (the §2.3(b) real
pairwise verifier) is wired into the controller.

### 2.6 Selectability-wall battery (2026-07-06) — five results that close the best-of-N program

Five follow-on experiments, run to decide whether the open-text best-of-N verifier direction is the
*deployable* method or merely a *characterized* one. Together they establish that the **selectability wall
is fundamental** and that the deployable efficiency lever is the **router**, not best-of-N. All figures are
copied from the cited artifacts; the 3 open sets (vqa_rad/slake/pathvqa) use exact-match/judge `oks`, PMC uses
**loose option-letter `oks`** (interpret PMC with caution — see caveat at the end).

**(1) Compounding FAILS — diverse-generation and pairwise-selection do not stack.** `combine_diverse_pairwise.py`
→ `combine_diverse_pairwise.json`. A 2×2 {pointwise, pairwise-Copeland over REAL A-vs-B verdicts} × {iid@8,
diverse@15}, selection accuracy [`oks`], pooled over the 3 open sets (n=1023), paired 3000-sample question
bootstrap.

| selector | iid@8 | diverse@15 |
|---|---:|---:|
| pointwise | 0.5191 | **0.5494** |
| pairwise (Copeland) | 0.5396 | 0.5376 |
| oracle | 0.6452 | 0.6813 |

Each lever alone beats the pointwise-iid baseline: **diverse-lever B−A = +0.0303** (CI [+0.0088, +0.0518], sig),
**pairwise-lever C−A = +0.0205** (CI [+0.0098, +0.0323], sig). **But they do NOT compound:** pairwise-over-diverse
(D=0.5376) is **≤** pointwise-over-diverse (B=0.5494) — `D−B = −0.0117` (CI [−0.0283, +0.0049]) — and the both-levers
gain over baseline `D−A = +0.0186` is **not significant** (CI [−0.0020, +0.0411]). On PMC (n=600) diverse lifts the
**oracle** +0.110 (0.505→0.615) but the selectors convert almost none of it: pointwise-div 0.305 vs pointwise-iid
0.290 (converted 0.015), pairwise-div 0.295 (converted 0.005). **Read: diversity buys coverage, not selectability;
the PMC oracle lift stays unconverted by either selector** (`compounds=false`). This resolves the §2.3 open
"pairwise-over-diverse" experiment — negatively.

**(2) Distractor-filtering FAILS — no pre-filter beats plain diverse-pointwise.** `distractor_filter.py` →
`distractor_filter.json`. Eight filters (drop-lone-confident, consensus, rarity, top-k-agreed) over the diverse
pool, using only candidate text + verifier score + cross-candidate agreement (never `oks`). Pooled-3ds (n=1023):
baseline a = unfiltered-diverse-pointwise **0.5494**, baseline b = iid@8-pointwise **0.5191**. Best filter
**rarity_log1p = 0.5601**: **vs a +0.0108 (CI [−0.0059, +0.0293], n.s.)**, vs b +0.0411 (CI [+0.0235, +0.0596],
sig). **No filter beats BOTH baselines** (`any_filter_beats_both=false`), and rarity_log1p **sign-flips per
dataset** vs diverse-pointwise (slake +0.0279, vqa_rad −0.015, pathvqa −0.0225, pmc −0.0083). **Mechanism: the
correct *new* answers diverse generation adds are themselves rare, so a rarity/agreement signal cannot separate
correct-rare from wrong-rare.**

**(3) Verifier CAPACITY does not break the wall — 32B-zeroshot ties 7B-trained.** `verifier_32b_gpu.py` (GPU
verdict dumps) + `verifier_32b_measure.py` → `verifier_32b_gpu.json`. Pooled n=600 (vqa_rad, slake, pmc_content),
selecting over the Lingshu-7B diverse pool; oracle_distinct = 0.672.

| verifier | sel_acc | note |
|---|---:|---|
| 7B-trained (pooled4 LoRA) | 0.475 | current deployed |
| 7B-zeroshot (base) | 0.413 | capacity floor |
| 32B-zeroshot (base) | 0.480 | 7× capacity |

**32B-zeroshot vs 7B-trained = +0.005** (CI [−0.023, +0.032], **n.s.**) — a 7× bigger verifier does **not** beat the
small *trained* one. The pure-capacity contrast **32B-zeroshot vs 7B-zeroshot = +0.067** (CI [+0.038, +0.095],
sig) is real but small, and the 32B still leaves an **oracle→selection gap of 0.192** (conversion 0.15). **Read:
the selectability ceiling is substantially FUNDAMENTAL, not a verifier-capacity artifact** — throwing a 32B at
selection buys ~7 points of the ~19-point gap.

**(4) End-to-end consolidation — best-of-N is Pareto-DOMINATED on FLOPs.** `end_to_end_consolidation.py` →
`end_to_end_consolidation.json`. Pooled n=1023, strong=judge_ok, FLOPs in FLOP-eq (× one 7B forward). Acc ladder:
7B-greedy 0.518 ≈ iid-bo8 0.519 < diverse-bo15 0.549 ≪ **always-32B 0.673**. Costs: **always-32B F=4.57**,
iid-bo8 **F=16**, diverse-bo15 **F=30**. The **global FLOPs-Pareto envelope**:

| operating point | FLOPs (FLOP-eq) | acc |
|---|---:|---:|
| 7B-greedy | 1.00 | 0.518 |
| iid→Pandora | 2.00 | 0.519 |
| iid→Pandora | 2.89 | 0.542 |
| iid→Pandora | 3.25 | 0.568 |
| always-32B | 4.57 | 0.673 |

**diverse-gen is NOT on the envelope** (`diverse_on_pareto_envelope=false`) — iid→Pandora Pareto-dominates
diverse→Pandora at every accuracy target (e.g. @0.55 iid F=3.3 vs div F=6.2; @0.65 iid F=8.0 vs div F=10.9), so
its 1.875× generation cost is never repaid. **The deployable envelope is `greedy → 7B+Pandora → always-32B`, and
the win is Pandora/router, not diverse generation.** (Recommendation in the artifact: max-acc → always-32B;
tight budget → iid→Pandora at F=2–3.3 for acc 0.52–0.57; diverse-gen → DO NOT deploy.)

**(5) Latency re-examination — best-of-N is latency-ALIVE but still does NOT beat always-32B.**
`latency_reexamination.py` → `latency_reexamination.json`. Real measured batch-1 latency (HF, cap320, NVML):
**GEN7 = 347.1 ms, VER7 = 175.5 ms, GEN32 = 665.0 ms**. A **parallel** best-of-N base costs only GEN7+VER7 =
**522.6 ms = 0.79× the single 32B forward (665 ms)** — the exact opposite of the FLOPs verdict, where iid-bo8's
16 FLOPs is 3.5× the 32B's 4.57 (**FLOPs-dominated yet latency-cheaper**, because batch-1 short-gen is
overhead-bound and best-of-N parallelises N away). So best-of-N **survives on the latency-Pareto envelope** as a
real low-latency/lower-accuracy point (`latency_alive = YES`). **But it does NOT beat always-32B**
(`beats_always32 = NO`): the fixed-bo-N accuracy ceiling is ~0.55 (gated-diverse envelope tops ~0.587 before it
escalates), far below the 32B's 0.673, and matching 0.673 forces heavy escalation that pushes parallel latency
back **above** 665 ms (iid-bo8+gate 1141 ms @esc93%; diverse-bo15+gate 1188 ms @esc100%). Note **Pandora is
penalized on this axis** (its adaptive draws are sequential, draw→check→draw), and plain **iid bo-N is not even on
the envelope** (dominated by diverse-bo15 at the same 522 ms). **always-32B owns the high-accuracy corner because
its open-text no-think generate (665 ms, F=4.57) is both cheap and fast.**

**Re-grounding (the conclusion of the best-of-N program).** The best-of-N outcome-verifier direction is now
**scientifically characterized but is NOT the deployable method in this setting.** Characterized: the
**selectability wall is fundamental** — it resists compounding (1), pre-filtering (2), and even 7× verifier
capacity (3, only ~7 of ~19 oracle-gap points recovered). Not deployable *here*: because the **Lingshu-32B strong
leg is cheap on BOTH axes** — **4.57 FLOP-eq** and **665 ms no-think** — escalating to it dominates spending budget
on more/diverser cheap draws (4), and best-of-N cannot beat it on latency either (5). **The deployable efficiency
win is therefore the ROUTER** — `greedy → 32B` (the §1.1 unified router: **48–82% of always-32B FLOPs at parity**)
— **plus Pandora for tight compute budgets** (F=2–3.3 for acc 0.52–0.57, §1.2/§4). **Caveat (scope, untested):**
this verdict is conditioned on a *cheap, fast* strong model; in an **expensive or slow-strong** regime (where one
strong forward costs ≫ N cheap draws or is latency-bound), the parallel best-of-N leg could still win — that
regime is not measured here. **Scoring caveat (carried over):** PMC-content `oks` is **loose option-letter
matching** (not exact-match/judge), so all PMC-specific numbers above are indicative only; the 3 open sets use
exact-match/judge `oks` and carry the load-bearing conclusions.

---

## 3. The two binding limits (and which ideas attack each)

The whole program is bounded by two walls (from the project's own analysis; see backlog header):

**Limit #1 — candidate quality / oracle@N ceiling.** oracle@8 ≫ verifier-bo8 everywhere (pooled Lingshu
oracle@8 0.513 vs bo8 0.414); the correct answer often simply isn't in the drawn set. This is where the accuracy
headroom lives.
- **Validated attacks:** §1.3 cross-model pooling **lifts the ceiling +0.05–0.13 oracle** (real, held-out) but
  Markowitz allocation ≈ uniform; §1.4 diversity **selection** buys a −15% sample-rate win but **cannot** lift
  the ceiling offline (needs diverse *generation*).
- **Resolved attack:** diverse-generation GPU pass (`diverse_measure_gpu.py`, **RESOLVED — §2.3(a)**) —
  **lifts the ceiling** (oracle +0.027 matched / +0.064 @M=15, converts to +0.025 verifier acc), and shifts the
  binding limit to selection (#2). **Queued:** boosting residual-specialist generator (backlog A3); speculative
  cascade with 32B-as-verifier (D1).

**Limit #2 — verifier selection ceiling (~74–82%).** Per-answer verifier AUROC is 0.90–0.93, but it loses
within-question near-ties, so it realises only ~74–82% of the oracle ceiling. §1.5 confirms the gate side is
already near-optimal (confidence/margin), so the lever is the *selector*, not the *escalator*.
- **Resolved attack:** §2.1 UGV single generative verifier (B2) — **negative for MCQ** (content-mode collapses
  accuracy, verifier gain +0.004); its home is open-text. Round-2 post-hoc selectors (§2.2: active-pairwise,
  bandit, Dawid–Skene) also fail to beat pointwise/uniform/majority.
- **Resolved attack:** real pairwise/knockout-tournament verifier (B1, GPU — **RESOLVED §2.3(b)**) — **beats
  pointwise +0.036 sel_acc / +0.076 eff**, closing ~35% of the pointwise→oracle gap and overturning the
  *simulated* failure (§2.2(a)). **Resolved:** compounding pairwise-over-diverse does NOT stack (§2.6 — the
  pairwise selector cannot convert the diverse-generation oracle lift; selectability wall is fundamental).
  **Queued:** semantic-cluster-then-verify (B3).

(Two further limits noted in the backlog — the **recoverability wall**, Jitkrittum AUROC ≈ 0.6, which §1.5
re-confirms explains why the confidence gate is near-optimal; and the **cost tension**, bo8 base cost = 2N cheap
forwards so FLOP break-even vs one 32B forward is N ≤ 2 — are the reason today's both-axes wins concentrate where
the strong model is weak/OOD, exactly the §1.2 open-text regime.)

---

## 4. Idea backlog pointer

Full living backlog: **`results/cascade_methods/METHOD_IDEAS_BACKLOG.md`** — **56 ideas** (initial 22 + re-rank →
pass-2 +12 = **35** §A–E; pass-3 +11 = **46** §F **[BEAT-32B]**; pass-3b +10 = **56** §G **★ESC**), each mapping a
mechanism from a related/unrelated field onto a concrete, testable variation, scored by both-axes potential ×
novelty × testability. §F (beat-32B fusion/routing) and §G (escalation-speed) are the two new axes exercised in §6.
**Top 5 to test first (§A–E):**

1. **Pandora's-Box adaptive controller** (Weitzman) — unifies adaptive-N + gate; **DONE this cycle → §1.2** (both-axes, −27% FLOPs held-out); **refinements → §2.5** (correlated-stopping −16–17% at iso-strong; pooling combo regime-limited).
2. **Diversity-maximized candidate set** (DPP/MMR) — attacks limit #1; **selection tested → §1.4**; diverse-*generation* GPU pass **DONE → §2.3(a)** (lifts oracle +0.027/+0.064, converts +0.025 verifier acc).
3. **Generator portfolio** (error-correlation / Markowitz) — attacks limit #1; **DONE → §1.3** (pooling wins, allocation ≈ uniform).
4. **Speculative cascade** (32B as a *verifier*, not generator) — the strongest structural both-axes lever; needs a GPU verify pass, then offline.
5. **Pairwise/knockout-tournament verifier** (PairJudge RM) — attacks the selection ceiling (limit #2);
   **simulated offline → §2.2(a) (can't beat pointwise); real GPU pass DONE → §2.3(b) — real pairwise BEATS pointwise (+0.036 sel_acc), overturning the simulation**.

Close runners-up: VOC/SPRT offline controllers with guarantees (C2/C3), and the **generative grounding verifier**
(B2 — the stated frontier, = the §2.1 UGV experiment, **resolved negative for MCQ this cycle**; open-text remains
its domain).

---

## 5. Reframe vs always-32B-THINK (2026-07-07, CPU re-costing on existing data)

> **The reframe.** The correct, honest baseline for the method is **always-32B-THINK** — the naive "just run the big
> *thinking* model on everything." The method is a **regime-adaptive 3-tier allocator**: MCQ → ACC (7B-nt → 32B-**nt**
> → 32B-**think**, gated so the slow think tier fires only on the reasoning residual); open-text → 7B best-of-N +
> trained verifier. **Think helps ONLY on reasoning** (MMMU, MedXpert); on perception **no-think ≥ think**, so the
> slow tier ~never fires. All numbers below are CPU re-costings of **existing** artifacts (no GPU). Machine-readable:
> `results/cascade_methods/artifacts/reframe_vs_bigthink.json`.
>
> **Two eval contexts, kept separate (do not cross-multiply):** (1) the **5-family ACC MCQ bake-off**
> (`master_data.csv`, NGC harness) carries the **measured batch-1 latency/energy** and the pooled efficiency; (2) the
> **faithful 3-family MedEvalKit eval** (`eval_results_*`) carries the **reasoning-regime think deltas**.

### 5.1 HEADLINE — method vs always-32B-THINK, per family × pool (measured batch-1)

Method row = `Ours (ACC-v2: agreement)`; baseline = `always-big-think [PARITY]`. Source: `master_data.csv`.

| family | pool | 32B-think acc | method acc | Δacc | parity? | 32B-think lat | method lat | Δlat | FLOPs% | 32B-think E | method E | think-esc% |
|---|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|
| **MVT** 7B→32B | ALL-6 | 0.5723 | 0.5693 | −0.003 | ✓ | 11.34 s | **2.27 s** | **−80%** | **52%** | 6319 J | 1182 J | 15% |
| | ALL-5 | 0.6463 | 0.6450 | −0.001 | ✓ | 8.88 s | **0.44 s** | **−95%** | 25% | 4916 J | 173 J | 2% |
| **Lingshu** 7B→32B | ALL-6 | 0.6611 | 0.6614 | **+0.000** | ✓(beats) | 0.32 s | 0.30 s | −8% | 49% | 113 J | 76 J | 0% |
| | ALL-5 | 0.7746 | 0.7726 | −0.002 | ✓ | 0.32 s | 0.25 s | −24% | 39% | 113 J | 61 J | 1% |
| **QoQ-Med** cheap→strong | ALL-6 | 0.4689 | 0.5095 | **+0.041** | ✓(beats) | 9.72 s | **0.12 s** | **−99%** | **9%** | 5382 J | 18 J | 0% |
| | ALL-5 | 0.5432 | 0.6048 | **+0.062** | ✓(beats) | 8.49 s | 0.12 s | −99% | 9% | 4692 J | 18 J | 0% |
| **Chiron** cheap→strong | ALL-6 | 0.5076 | 0.6023 | **+0.095** | ✓(beats) | 4.25 s | **0.20 s** | **−95%** | 19% | 1176 J | 17 J | 0% |
| | ALL-5 | 0.5926 | 0.7249 | **+0.132** | ✓(beats) | 3.66 s | 0.20 s | −95% | 20% | 1006 J | 17 J | 0% |
| **MedGemma** cheap→strong | ALL-6 | 0.5253 | 0.5219 | −0.003 | ✓ | 12.72 s | **3.37 s** | **−74%** | 68% | 6535 J | 1614 J | 20% |
| | ALL-5 | 0.5979 | 0.6028 | **+0.005** | ✓(beats) | 9.76 s | 0.18 s | −98% | 12% | 4990 J | 16 J | 0% |

**Read.** Always-32B-THINK is **Pareto-dominated on every family**: the method matches-or-beats its accuracy at
**9–68% of its FLOPs** and **8–99% lower latency / 33–99.7% lower energy**. On **3 of 5 families (QoQ, Chiron, and
marginally Lingshu) the big *thinking* model is actually LESS accurate than the method** (QoQ +0.041, Chiron +0.095) —
because think over-thinks the perception-dominated suite — *and* 1.2–45× slower. Think:no-think latency ratio per
family (ALL-6): **MVT 49×, MedGemma 45×, QoQ 43×, Chiron 15×, Lingshu 1.2×** (Lingshu has no real promptable think
mode — its "think" run ≈ no-think, which is why its latency win is small).

> **⚠️ Annotated 2026-07-29.** The `always-32B-think` column above uses each family's *native* think arm — the
> same arms the prompt-matching audit found unmatched, and for **Lingshu** and **QoQ** those arms were run at a
> different image resolution than the direct arm. Two specific repairs: (a) Lingshu's "think" arm generated
> **3.0 tokens** — it never reasoned, so **"Lingshu has no real promptable think mode" is wrong as an
> explanation; the prompt simply had no reasoning trigger** (`runners/run_native_think.sh:7`), and the **1.2×
> ratio is the ratio of two answer-format prompts, not of reasoning**; with a genuinely reasoning arm
> Lingshu's perception accuracy is *worse* (pooled −0.0866), which widens the method's margin rather than
> narrowing it. (b) These per-family cascade rows have **not** been recomputed against the repaired arms —
> treat them as an open item, not a claim. Retrospective §5.1, §10.1 C22.

### 5.2 By regime — where the two halves of the win come from

**Reasoning regime (MMMU, MedXpert) — think IS the accuracy target, and it's slow → gated-think 3-tier win.**
Faithful MedEvalKit think gains (`eval_results_*_reason` vs no-think): **MMMU +0.027 / +0.100 / +0.120**
(Lingshu / MVT / IV3); **MedXpert +0.00 / +0.045 / +0.031**. Reserving a *gated* think tier captures these at a
fraction of always-think cost (MedVLThinker, `src/cascade_methods` 3-tier):

> **⚠️ Corrected 2026-07-29 — put confidence intervals on those three MMMU numbers.** They reproduce, but
> **only two of three are significant**: **Lingshu-32B +0.0267 [−0.0467, +0.1000] (n = 150, NOT
> significant)**, MedVLThinker-32B **+0.100 [+0.027, +0.173]**, InternVL3-38B **+0.120 [+0.047, +0.193]**.
> Per MedXpert split, Lingshu is **−0.0035** (MX-R) and **+0.0000** (MX-U) — i.e. **Lingshu-32B shows no
> reasoning benefit anywhere**, on this harness or on the internal one, and must not be cited as evidence
> that reasoning helps. Two further caveats: (a) these `_reason` dumps are **format-unmatched** because of
> two local uncommitted `MedEvalKit` edits (retrospective §10.5) — they are corroboration, not matched
> evidence; the *pre-edit* `*_think` dumps are invalid outright (2.6–3.2 generated tokens); (b) the
> gated-think tier below is still sound as a *mechanism*, but on Lingshu it fires ~0% and buys ~0.
> Source: `artifacts/finding1_corrected_2026-07-29.json` → `medevalkit_external_corroboration`.

| dataset | 32B-think acc | 3-tier acc | →32B-nt | think fires | FLOPs% | lat/energy% | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| **MMMU** | 0.688 | 0.688 | 68% | 28% | **78%** | **~31%** | **WIN** (match think at ⅓ latency) |
| MedXpert-Reasoning | 0.326 | 0.326 | 100% | 92% | 143% | — | no win (7B near-floor → escalate ~all) |
| MedXpert-Understanding | 0.384 | 0.384 | 96% | 96% | 151% | — | no win |

MMMU is the clean gated-think win; MedXpert is the floor regime where no cascade saves compute (matches accuracy, no
efficiency gain — why the original ACC excluded it).

**Perception regime (SLAKE, VQA-RAD, PMC, PathVQA, OmniMed) — no-think ≥ think and ~35× faster → huge, think never
fires.** think Δ(think−nt) is negative-or-flat on perception (e.g. SLAKE: MVT −0.084, Chiron −0.108; VQA-RAD: all
−0.07 to −0.09). always-32B-THINK is a strictly dominated baseline here; the method routes to the fast no-think tiers
and **think escalation is ~0%** (ALL-5 pools above). Perception latency anchor: method 0.12–0.44 s vs always-think
8.5–12.7 s on the genuine-think families.

### 5.3 Router with the 32B-no-think MIDDLE TIER restored

The as-built unified router (`unified_router.json`) escalates MCQ to **32B-no-think** (it loads `eval_results_*_full`
first), so it already **matches always-32B-no-think** (Lingshu 0.549/47.7% FLOPs, MVT 0.522/60.0%, IV3 0.514/82.1%)
and already **avoids the think-tax on perception**. What it lacks vs the *correct* baseline is the **gated think top
tier** — it never fires think, so it leaves the reasoning gains on the table (per-benchmark MMMU MVT +0.100 / IV3
+0.120; MedXpert MVT +0.045 / IV3 +0.031). Restoring the full ACC 3-tier fires think on only the **5.45% reasoning
residual** of the MCQ stream (2 150 of 39 422 items), so the **pooled** MCQ accuracy recovered is small (**~+0.002–0.003**,
PMC's 33 430 items dominate) but the **per-benchmark MMMU/MedXpert parity-with-think is recovered**.

**Load-bearing "restore the middle tier" delta** (`METHOD_ACC.md` head-to-head, measured latency/energy, ALL-6):
inserting the 32B-**no-think** middle tier turns an escalate-everything-to-think cascade into the ACC —

| variant (ALL-6) | acc | think-esc | FLOPs% | latency | energy |
|---|---:|---:|---:|---:|---:|
| M2 escalate-to-think, **no nt-middle** (7B-think→32B-think) | 0.5725 | 86% | 105% | 29.8 s | 7049 J |
| M3 7B-think middle (7B-nt→7B-think→32B-think) | 0.5697 | 65% | 89% | 23.2 s | 5499 J |
| **M1 ACC — 32B-nt middle restored** | 0.5694 | **19%** | **55%** | **5.9 s** | **1505 J** |
| **M1b ACC + agreement gate** | 0.5710 | **14%** | **54%** | **4.86 s** | **1220 J** |

**Δ (restore nt-middle, M1 vs M2):** think-esc 86%→19%, FLOPs 105%→55%, latency **29.8→5.9 s (−80%)**, energy
7049→1505 J (−79%), at matched accuracy. This *is* the router's MCQ-arm upgrade.

### 5.4 Verifier / best-of-N / diverse-gen as cheap-tier boosters — value ≈ NIL for reducing think escalations

Do they cut cheap-tier error enough to fire think less often? **No, not in this structure.** (i) **MCQ verifier
degenerates** on single-letter answers (UGV content-mode collapses PMC 0.534→0.132; verifier gain +0.004) → cannot
boost the cheap MCQ tier → cannot reduce MCQ think-escalations (which are set by the genuine reasoning residual).
(ii) The levers that *do* work are **open-text-only**, and the open-text arm is **2-tier no-think** — best-of-N
*replaces* the think tier, so there is no think tier to cut. (iii) Every open-text lever is **Pareto-dominated on
FLOPs** by escalating to the cheap-fast strong leg (diverse-gen +0.027–0.064 oracle but off the FLOPs envelope;
cross-model pool +0.11–0.15 oracle but ~+0.002 conversion; real pairwise verifier +0.036 sel_acc but doesn't
compound) — the §2.6 re-grounding conclusion (deployable lever = the router, not best-of-N) holds. Their genuine
value is lifting the open-text **accuracy ceiling**, not reducing slow-think calls.

### 5.5 Blocked on GPU measurements (flagged, not computable here)

- **M1 — open-text 32B-THINK batch-1 latency/energy.** The open-text strong leg was only run **no-think** (measured
  gen32 = 665 ms). ⇒ the **open-text arm cannot be costed vs always-32B-THINK on latency/energy**; only FLOPs
  (N≤2 break-even) + accuracy are computable now (verifier matches/beats 32B-no-think; parallel bo-N ~522 ms < 665 ms).
- **M3 — InternVL3 8B/38B batch-1 latency/energy.** `master_data.csv` profiled only Lingshu-7B/32B and MVT-7B/32B.
  ⇒ **IV3 enters the reframe only via FLOPs proxy (router 82.1%) + accuracy + reason-dir think deltas**; its
  vs-always-38B-THINK latency/energy is unquantified.
- **ALL-7 / OmniMed strong-think.** The OmniMed 32B/38B leg is blocked (deterministic tp=2 NCCL hang). ⇒ the **pooled
  ALL-7 vs-always-32B-THINK number is not reportable**; **ALL-6 is the computable pool.** Conclusion is unchanged:
  OmniMed is keep-cheap (cheap Lingshu 0.827 ≈ paper strong 0.834), the method keeps-cheap at ~0% escalation, so only
  the pooled ALL-7 figure (not any verdict) is missing.

---

## 6. Final method (integrated router) + push results (2026-07-07)

> **The assembled method.** §5 established the correct baseline (always-32B-**THINK**) and the regime framing; §6
> assembles the project's proven pieces into a **single format-aware router** and pushes its accuracy/latency with the
> two new idea axes (backlog §F beat-32B, §G escalation). Full spec + ablations: **`METHOD_FINAL_2026-07.md`**.
> Machine-readable: `integrated_method_vs_think.json` ← `integrated_method.py`, `beat32b_fusion.json` ←
> `beat32b_fusion.py`, `escalation_levers.json` ← `escalation_levers.py`, `slake_open_bestofN.json`. CPU-only
> re-costings of existing dumps; measured batch-1 costs; all thresholds **held-out (5-fold cross-fit)**.

### 6.1 HEADLINE — integrated format router vs always-32B-THINK (n=42,374, faithful MedEvalKit)

**Router:** MCQ → 7B-nt + **margin** gate → **32B-no-think** (~16% esc), **keep-7B on MMMU**; open-text → **7B
best-of-8 + trained verifier** (verifier-conf gate) → 32B-no-think. Strong leg is **no-think everywhere** (think ≤
no-think on perception, §5). Source: `integrated_method_vs_think.json:pooled`.

| pool | n | method | 32B-think | **Δ vs think** | 32B-nt | Δ vs nt | esc | latency | lat saved vs think | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **full suite** (9) | 42,374 | **0.5750** | 0.5631 | **+0.0118** | 0.5732 | +0.0018 | 15.5% | **459.6 ms** | **−95.6%** | 2.538 |
| MCQ only (6) | 40,029 | 0.5756 | 0.5745 | +0.0011 | 0.5765 | −0.0009 | 16.2% | 454.5 ms | −95.7% | 1.738 |
| open only (3) | 2,345 | 0.5642 | 0.3698 | **+0.1943** | 0.5168 | +0.0473 | 4.0% | 548.3 ms | −94.8% | 16.181 |
| full-suite macro-avg | — | 0.6753 | 0.6063 | **+0.0690** | — | — | — | — | — | — |

**Read.** The router **matches-or-beats always-32B-THINK accuracy at −95.6% latency (460 ms vs 10,522 ms)**. The two
accuracy engines are the **open-text arm** (bo8+verifier beats even 32B-no-think, which beats 32B-think by +0.12…+0.21
because think over-thinks perception) and **MMMU keep-7B** (+0.140 vs think, the Lingshu-7B 0.80 anomaly). Perception
MCQ ties 32B-nt(≈think). Per-benchmark Δ vs think: PMC **+0.0014**, SLAKE-cl −0.012, VQA-RAD-cl −0.008, PathVQA-cl
−0.001, MedXpert −0.004, MMMU **+0.140**, SLAKE-o **+0.192**, VQA-RAD-o **+0.105**, PathVQA-o **+0.207**. **FLOPs are
mixed-honest:** the MCQ arm *saves* FLOPs (1.74 pooled vs 4.57); the open-text best-of-8 arm *costs* FLOPs (16 cheap
forwards) but buys the latency+accuracy win.

### 6.2 Gate bake-off — margin is the best MCQ gate (Correction #1; `integrated_method_vs_think.json:gate_comparison_mcq`)

The premise "CASP-stability or cross-model agreement beat the margin gate" **does not hold for Lingshu**. Pooled
perception-closed MCQ (n=37,879; cheap-7B-nt 0.5769, strong-32B-nt 0.5905):

| gate (KEEP=trust-7B) | detection AUROC | min esc to 32B-nt parity | note |
|---|---:|---:|:--|
| **margin (deployed)** | 0.7254 | **15.62%** | best deployable (cheap, continuous) |
| conf/MSP | 0.7318 | 20.26% | ~= AUROC but needs more escalation |
| CASP-stability | 0.7241 | 15.50% | **INERT** — 7B is 98.95% cap320-vs-full stable → collapses to margin |
| agreement (7B vs 32B) | 0.6565 | 19.96% | worst ranker **and** needs the 32B run (not cheap) |

Agreement is still an informative binary trust signal (P(7B ok\|agree)=0.687 vs \|disagree=0.326) but is dominated by
a continuous margin as a ranker. **Deployability order: margin > agreement > CASP.** Matches the repo's standing
finding: **no cheap gate beats margin on MCQ**.

### 6.3 Push 1 — beat-32B FUSION on PMC (`beat32b_fusion.json`; backlog §F)

Slice-gated router × decision fusion: per benchmark route to the **held-out-guardrail-certified** winner among
{always-32B-nt, keep-7B, calibrated **confidence-advantage** fusion}. Only **PMC-VQA** (fusion) and **MMMU** (keep-7B)
are certified non-32B; radiology/pathology closed sets keep 32B (guardrail — fusion hurts there).

- **PMC fusion (F3 conf-advantage ≡ 2-detector Chair-Varshney):** acc **0.5653** vs 32B-nt 0.5518 → **+0.0135, 95% CI
  [0.0100, 0.0169]**, n=33,430 held-out. Classic per-*slice*-reliability C-V collapses to always-32B (d=0.0) — the beat
  **requires per-sample confidence**. F5 double-reading: on the 33% disagreement set the **free** conf-advantage
  arbiter (0.412) beats both the 32B-nt (0.371) and the expensive 32B-think (0.387) arbiter → **think is a poor arbiter**.
- **Pooled (full suite, n=42,374):** method **0.5869** vs 32B-nt 0.5732 (**+0.0138**) vs 32B-**think** 0.5631
  (**+0.0238**); macro Δ vs think **+0.0739**. This **raises the beat vs always-32B-nt from the prior integrated
  method's +0.0017 → +0.0138, and vs 32B-think from +0.0118 → +0.0238**, by converting PMC from "matches 32B" to
  "beats 32B" — a **genuine broad-slice win, NOT the MMMU anomaly**.
- **Cost (honest):** the fusion cell runs both legs (FLOP 5.57 vs 4.57, +22% on PMC) → pooled FLOPs **5.751 (1.26×
  always-32B)**, latency 653 ms (still −93.8% vs think). It is an **accuracy lever, not a compute-saver** — you cannot
  cascade-away the 32B on a fusion slice. This is the **Pareto knob**: compute-lean cascade (§6.1, 0.575 @ FLOPs 2.54)
  ↔ accuracy-max fusion (0.587 @ FLOPs 5.75).

### 6.4 Push 2 — G8 parallel prefill prefetch (`escalation_levers.json`; backlog §G)

vs always-32B-**no-think** (665 ms) the MCQ arm is *slower* on three heavy-escalation cells (VQA-RAD-cl 726 ms,
MedXpert 943 ms, SLAKE-open 699 ms). **G8** runs the 32B image-prefill concurrently with the 7B pass (escalated leg =
max(cheap, prefill32)+decode32); measured φ=0.586 → prefill32=390 ms > cheap 347 ms, so the whole 7B pass hides under
the prefill on every MCQ escalation. **Zero accuracy change:** pooled latency **461.1 → 405.2 ms (−12.1%)**, and
**every slower cell flips** under always-32B-nt. FLOPs caveat: unconditional prefetch pays the 32B prefill on every
query (pooled 2.337 → 4.575 ≈ always-32B) — a latency-for-FLOPs trade, free only on an idle 2nd GPU; the **slice-gated**
variant (prefetch only where esc ≥ 0.40) keeps FLOPs 2.492 at 429.8 ms. **G5** (recoverability suppressor) and **G6**
(2-of-2 gate) are knobs, not free lunches (G5 ε\*=0.06 suppresses MedXpert for −0.0018 pooled / 943→347 ms; G6 no gain
— CASP 98.9% inert, no orthogonal 2nd MCQ signal). Combined G8+G5(ε\*): pooled 416.4 ms (−96.0% vs 32B-think) at
−0.0018 acc, no cell slower than always-32B-nt.

### 6.5 Push 3 — SLAKE-open best-of-8 verifier fill (`slake_open_bestofN.json`)

The one open-text cell that lacked a verifier dump is now filled: pooled4 verifier scored pointwise over the K=8 SC
candidates (n=645). **bo8+verifier 0.7798** vs greedy_t0 0.7302 (**+0.0496**) / self-consistency-modal 0.7364
(+0.0434); oracle@8 0.8791; 32B-nt 0.8186 (bo8 − 32B = −0.0388). With the verifier-conf gate the SLAKE-open cell hits
32B-nt parity at **~13% escalation vs ~53%** for the old greedy+seqlogprob fallback — a **~4× escalation cut**, and it
is what makes the final integrated SLAKE-open cell bo8+verifier (0.8155 @ 605 ms) rather than the greedy fallback.
**Caveat: SLAKE-open is IN-DOMAIN** for pooled4 (verifier trained on slake+pathvqa+kvasir+vqa_rad) — same as the
VQA-RAD-open / PathVQA-open cells.

### 6.6 Two corrections + the caveat ledger

- **Correction #1 (gate):** margin > agreement > CASP on Lingshu (§6.2). FALC's margin choice was correct.
- **Correction #2 (router):** a **router is required** — the MCQ margin gate has no open-text analog and the verifier
  is open-text-specific; a single unified gate (7B seqlogprob) is beaten by margin on MCQ and verifier-conf on open.
- **Caveats:** open-text 32B-**think** accuracy is **estimated** (32B-nt + measured modal think-delta); open-text
  (SLAKE/VQA-RAD/PathVQA) pooled-4 verifier is **in-domain**; **OmniMed-32B blocked** (NCCL hang, 7B-only, ALL-6 is the
  computable pool); fusion/best-of-N are latency+accuracy levers that **cost FLOPs** (fusion +22% on PMC; bo8 = 16
  forwards, break-even N ≤ 2); the MCQ margin cascade is the FLOP-saving part (1.74 pooled MCQ FLOPs vs 4.57).

---

## 7. Final-method suite (2026-07-07 evening) — F8+F10 fold + slice-structure / logit / escalation / quant probes

> **What §7 adds over §6.** §6 assembled the integrated router and pushed it with F3 fusion (accuracy-max mode
> **FLOP-POSITIVE at 1.25×**). §7 folds the two best pass-4 levers (**F8** certified veto, **F10** open-text L2D) into a
> single reproducible `method_final.py`, flipping the accuracy-max mode **FLOP-NEGATIVE**, and reports four honest
> negatives/re-costings that **sharpen** the scope rather than widen the claim. All OFFLINE (no new inference) except the
> two flagged re-costings; every figure is copied from the cited artifact. **Abstention is out of scope** for this whole
> program — the method always answers (backlog H3 abstain-to-human was excised); the F8 "certified veto" keeps the 7B
> answer, it does not defer to a human. New source artifacts + code (all under `artifacts/` and `src/cascade_methods/`):
> - `method_final.json` (v1) + `method_final_v2.json` (v2) ← `method_final.py`
> - `beat32b_more.json` ← `beat32b_more.py` (§F pass-4: F8/F7/F11/F10)
> - `escalation_more.json` ← `escalation_more.py` (§G pass-4: G7/G2/G4); `imagetoken_prune_gpu.json` (G4 feasibility)
> - `quantized_strong_leg.json` ← `quantized_strong_leg.py` (G3 INT4 re-cost)
> - `logit_fusion.json` ← `logit_fusion.py` (full-posterior MCQ fusion)
> - `robust_slice_routing.json` ← `robust_slice_routing.py` (§H pass-4: H4/H8/H2)

### 7.1 `method_final` v2 — F8 + F10 folded in → BOTH Pareto modes are FLOP-NEGATIVE (`method_final_v2.json`)

`method_final.py` writes v1 (the §6 F3-fusion config) **and** v2 (F8 replaces the accuracy-max PMC fusion cell; F10
replaces the open-text arm's parity-τ gate, shared by both modes). Held-out (5-fold), measured batch-1 cost constants
(GEN7 347 ms/1 FLOP, VER7 175 ms/1, BO8 522 ms/16, GEN32-nt 665 ms/4.57, GEN32-think 10 521.6 ms/4.57, FUSE-both
665 ms/5.57). Full suite n=42 374, sample-weighted:

| mode (v2) | acc | Δ vs 32B-think | Δ vs 32B-nt | esc | FLOP-eq | ×always-32B | lat_seq | lat_par | lat saved vs think |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **compute-lean** | 0.5754 | **+0.0123** | +0.0023 | 16.7% | **2.238** | **0.49×** | 577 ms | **468 ms** | **95.6%** (par) |
| **accuracy-max** | 0.5844 | **+0.0212** | +0.0112 | 64.0% | **4.246** | **0.93×** | 839 ms | 729 ms | 93.1% (par) |

**Read.** `both_modes_flop_negative = true`. The prior F3 accuracy-max ran **both legs on 100% of PMC** (5.57 FLOP/item)
→ full-suite **5.695 FLOP = 1.246× always-32B (FLOP-POSITIVE)**. Swapping in **F8's certified veto** (7B on all + 32B
only on the ~60% non-veto PMC cells = 3.74 FLOP/item) cuts full-suite FLOPs **5.695 → 4.246 (1.25× → 0.93×)**, flipping
the arm **FLOP-negative**, while **retaining 70.4% of F3's PMC beat** over 32B (PMC Δ vs 32B-nt +0.0135 → +0.0095, still
CI-certified above 32B) and cutting the PMC cell FLOPs 32.8%. **F10** lifts all three open cells (SLAKE_open +0.0109,
VQA_RAD_open +0.0100, PathVQA_open +0.0100 vs the prior parity-τ gate) and **repairs the SLAKE_open / VQA_RAD_open
losses** (both now ≥ 32B-nt), because the parity-τ gate targeted iso-32B *by design*. **v2 vs v1 compute-lean** is a
wash (+0.0005 acc, −0.006 FLOPs — it was already FLOP-negative at 0.49×); F8/F10's value is concentrated in the
accuracy-max mode. **Headline: a FLOP-negative Pareto knob at both ends** — compute-lean +0.0123 @ 0.49×, accuracy-max
+0.0212 @ 0.93×.

**Honest caveats** (`method_final_v2.json:data_gaps`): F8 captures MOST (not all) of F3's PMC beat (the trade buys the
FLOP-negative arm); the F10 open arm bills the cheap leg at Pandora's adaptive meanN(<8) draws while scoring the KEEP
leg on the best-of-8 verifier pick → kept-7B accuracy mildly optimistic vs a strict best-of-meanN; F10 routes on
7B-side features only (open dumps carry judge_ok, not the 32B answer text — no cross-model-agreement feature); 32B-THINK
open-text accuracy is **estimated** (judged 32B-nt + measured modal think-delta −0.195/−0.120/−0.130; PathVQA-closed has
no 32B-think dump → think = no-think); SLAKE/VQA-RAD/PathVQA-open are **in-domain** for the pooled4 verifier.

### 7.2 §F pass-4 (`beat32b_more.json`) — the MCQ accuracy-beat does NOT extend past PMC; only F10 (open-text) moves

Four combiners tested OFFLINE against F3 (champion: PMC +0.0135, gated off elsewhere). **Three MCQ combiners
independently confirm the recoverability wall — none certifies a new closed slice beyond PMC:**

| lever | pooled MCQ Δ vs 32B (n=40 029) | CI95 | NEW cells vs F3 | note |
|---|---:|---|:--|:--|
| **F8** certified weak-veto | **+0.0080** | [0.0061, 0.0099] | **none** | guardrail-safe, **0.885× FLOPs** (vs F3 1.22×); COST+SAFETY, not new acc |
| **F7** super-learner (GBM-rich) | +0.0136 | [0.0108, 0.0164] | **none** | GBM does **not** catastrophically overfit here (CALM-Fuse story not reproduced) |
| F7 super-learner (logistic-frugal) | +0.0110 | [0.0089, 0.0132] | none | reproduces PMC + MMMU |
| **F11** decision-level BMA (additive) | +0.0116 | [0.0089, 0.0142] | **none** | per-slice EM auto-gates strong slices; PoE ≈ additive; tiny <0.003 leaks (not hard-gate-safe) |

**F10 — learning-to-defer on OPEN-TEXT (the only lever on a new axis).** Learned team-objective rejector over 7B-side
open features (cross-fit): **PathVQA_open 0.462 = +0.086 CI[0.064,0.106] BEATS 32B-nt**; **SLAKE_open 0.8202 (+0.0016)**
and **VQA_RAD_open 0.605 (+0.005)** point-positive (CI spans 0 at n=200–645) — both **flipped from below-32B to
iso/above**, and F10 beats the deployed parity-τ gate on all three. Only CI-certified new open beat = PathVQA_open
(already won by the verifier cascade; F10 improves +0.0773 → +0.0860). **Verdict:** extends past PMC = **PARTIAL** — MCQ
**NO** (F8/F7/F11 confirm the wall; closed beat bounded to comparable-skill/de-correlated slices = PMC, radiology/
pathology/MedXpert correctly kept at 32B); open-text **YES in direction** (F10 removes the two residual open losses).

### 7.3 Logit-level fusion (`logit_fusion.json`) — NEGATIVE; full-posterior fusion does not extend the beat past F3

Does full per-option-logprob fusion of Lingshu-7B + Lingshu-32B beat always-32B on **more cells** than decision-level F3?
OFFLINE on the only dumps carrying the full option vector — `ckpts/gate_lingshu{7b,32b}_mcq` (300–500/slice). Four
combiners (F11_fixed / F11_reweighted log-opinion-pool / F6 contrastive-decoding / F3_confadv).

- **Every method certifies exactly ONE cell vs 32B, and it is MMMU** (F3 {MMMU}, F11_fixed {MMMU}, F11_rw {MMMU}, F6_cd
  **{} = 0 cells**) — a route-to-7B anomaly (acc7 0.853 ≫ acc32 0.624; F11_rw learns λ≈0.04 = all-7B), **not** a fusion
  win. On the 4 perception sets (broad4, n=1688) held-out fusion **collapses to λ≈1/α≈0** ("just use 32B"): F11_rw
  −0.0047 (n.s.), F6_cd exactly 0.0 (the 7B "amateur" is not uniformly worse → subtracting it is unsafe), F3 −0.0095.
- On the **PMC subsample (n≈500)** neither F3 nor F11 can certify a +0.0135-size effect (CI ~±0.03); F11 vs F3 is within
  noise. A power-matched full-posterior PMC test needs a **33k × 2-model GPU re-dump** (the MedEvalKit full dumps carry
  only top-1 conf/margin) → excluded by the SHORT-run guardrail. **Conclusion:** logit fusion does NOT extend the beat
  past PMC / past F3's cell coverage — a further independent confirmation of the fusion/recoverability wall.

### 7.4 Robust slice-routing (`robust_slice_routing.json`, §H pass-4) — H4/H8/H2 negative; 6th wall confirmation, guardrail validated

Three §H "remaining-headroom" ideas run OFFLINE to harden/extend the beat on the 6 Lingshu MCQ cells (n=40 029; baseline
F1-certified non-32B cells = PMC fusion +0.0135 and MMMU keep-7B +0.167, n=150 anomaly).

- **H4 learned error-slice discovery (Domino/Spotlight) — NO genuinely-new slice.** Over 8 DISCOVER/CONFIRM splits (106
  candidate slices, BH-FDR5 + Bonferroni, "genuinely-new" = a slice **inside an always-32B dataset**, vs a
  label-permutation null): mean BH-FDR5 survivors **0.25/split**; raw genuinely-new count (**1.62/split**) sits **below**
  the permutation null (mean 5.61, p95 15); no new slice recurs in > 5/8 splits (closest: MedXpert|bsys=Respiratory
  5/8). Discovery instead **re-finds the known PMC/MMMU wins as echoes** (PMC|wh=what, PMC|margmid, PMC|marghi all 8/8;
  MMMU|wh=other 7/8) **without being told they are special → validates the F1 hand-gate**. The beat does **not** extend
  past PMC/MMMU via slice structure → recorded as the **6th independent confirmation of the recoverability wall**.
- **H8 Bühlmann-Straub credibility shrinkage — overfit risk is REAL, but F1's existing guardrail already fixes it.**
  Naive point routing overfits thin slices: fine 61-slice family = **~7.5 held-out guardrail violations/split** (of
  ~24.5 deviating). Bühlmann shrinkage helps only marginally (**7.5 → 6.62**). The **simple CI lower-bound guardrail**
  (deviate iff discovery 95% LB > 0 — **exactly F1's existing rule**) drives fine-family violations to **0.25** and
  hand-family to **0.0** at a preserved pooled beat **+0.0117**. → credibility shrinkage does not beat the deployed
  guardrail; **H8's value is diagnostic** (confirms overfit + validates F1's CI-guardrail as the correct, sufficient fix).
- **H2 kNN neighborhood-recovery gate — 0/5 datasets beat the scalar margin gate.** Low-budget accuracy area loses to
  margin on all 5 (MedXpert 0.2684 vs 0.2746, PMC 0.5576 vs 0.5597, SLAKE 0.8577 vs 0.8598, VQA-RAD 0.8127 vs 0.8295,
  PathVQA 0.8768 vs 0.8792); `knn_beats_margin = false` everywhere → the MCQ escalation signal is intrinsically weak
  (~0.6 AUROC, the recoverability wall).

**Net:** the beat-always-32B is **robustly bounded to PMC + MMMU**; no automatic slice-structure method extends it;
thin-slice overfit is real and **F1's CI-lower-bound guardrail** (already deployed) is the validated fix. Guardrail-honest.

### 7.5 §G pass-4 escalation levers (`escalation_more.json`) — G4 is the one live speed lever; G7/G2 offline dead-ends

- **G7 semantic escalation cache — DEAD offline.** No image id/hash in the dumps → only question **text** is a key,
  conflating different images that share a templated question. Duplication ~0 on the escalation-heavy cells (MedXpert
  0.0, PMC 0.0014, VQA-RAD 0.0637); where high it is templated (SLAKE 0.815, PathVQA 0.372) with *different* images →
  unsafe cross-image reuse. Safe image-keyed cache is unmeasurable here → future work.
- **G2 early-exit — DATA-ABSENT for its productive mechanism.** Both legs emit **~3 generated tokens on every
  benchmark** (median 3, p90 3–7) → **prefill-bound**, generated-token halting saves ~0. The real lever (layer-depth
  early-exit, CALM/LayerSkip ~2–3×) needs intermediate-layer logits not in the dumps → GPU probe.
- **G4 32B image-token pruning — the single new lever with quantified offline headroom.** Analytical model (φ=0.586,
  prefill 390 ms / decode 275 ms, LAYER_RETAIN = 1−3/64 = 0.953, real token_cache image fractions): **projected
  per-escalation 32B latency/FLOPs −26% @ p=0.50 / −39% @ p=0.75** on the most image-token-rich cell (VQA-RAD-closed);
  **pooled cascade latency 459 → 432 ms @ p=0.50 (−5.9%)**. **Accuracy needs a GPU confirm.** Feasibility investigated
  and **deferred with a plan** (`imagetoken_prune_gpu.json`): the −26% FLOP projection stands (grounded by measured
  image-token fractions VQA-RAD 0.918 / SLAKE 0.844 / PathVQA 0.895 / PMC 0.852 / MMMU 0.644), but a correct impl is
  multi-hour (vLLM has no mid-network token-drop; Qwen2.5-VL 64-layer 3D M-RoPE needs position-id re-indexing) and the
  measured resolution-cap analog shows image-token reduction is **FREE on PMC (−0.001) but COSTS on radiology (SLAKE
  −0.017, VQA-RAD −0.040)** → a **per-benchmark radiology guardrail is mandatory**; a wrong impl would fabricate a
  lesion-safety accuracy number (no-fabrication rule). G4 stacks with the §6.4-validated G8 (prefetch) + G5 (suppressor).

### 7.6 Quantized (INT4) strong leg (`quantized_strong_leg.json`, G3) — a VRAM/energy win, NOT a FLOPs win

Re-cost the integrated cascade with an AWQ/GPTQ-INT4 32B strong leg (no pre-quantized Lingshu-32B loadable in vLLM;
`bench_int4_strong_leg.py` committed but a **HF-CDN outage stalled 2/6 AWQ shards** → latency is a composition-grounded
**projection**, INT4 accuracy from literature Δ ≈ −0.005…−0.010).

- **FLOPs — NO win under the repo's unit.** The repo's FLOP-eq is a **MAC count** (precision-independent): a 32B-INT4
  forward is still **4.57 FLOP-eq**, so method FLOPs are unchanged (full-suite 2.538, MCQ 1.738, open 16.181). Only under
  a *throughput-effective* accounting (~1.5 decode) does the strong leg get cheaper, and only on high-escalation MCQ
  cells (MedXpert 5.09→2.34, VQA-RAD-cl 3.60→1.85, PathVQA-cl 3.09→1.69; pooled 2.54→2.06). The **open-text arm is
  unchanged** (16.0→~16.0 — 97–98% best-of-8 cheap-7B forwards; the strong leg is 2–3%, so quantizing it is the wrong
  lever there).
- **Latency — only ~12%, not ~2.5×.** The strong leg is **32B-NO-THINK = PREFILL-bound** (measured decode 69 ms/tok, ~2
  gen tokens, decode ≈21% of 665 ms). AWQ-INT4 speeds **decode only** (~2–3× on A100), not compute-bound prefill nor the
  FP16 vision tower → **665 → 582.7 ms (0.876×, ≈12%)**. (INT4's decode win applies to always-32B-THINK's ~300 decode
  tokens, 10 521→~4 525 ms, but the method already beats that baseline by ~23× on latency → moot.)
- **Bottom line:** INT4 strong leg is worth adopting for **VRAM / energy / deployability** — it fits the 32B at **tp=1
  (~20 GB) on one GPU**, sidestepping the OmniMed tp=2 NCCL hang — and cleans up the MCQ FLOP story under a
  throughput-effective view, but it does **NOT** make the method uniformly FLOP-dominant vs always-32B-nt and does
  **NOT** materially change latency. **Quantization is a memory/energy lever, not a FLOPs lever, for this method.**

### 7.7 §7 takeaways

1. **The deployable method is `method_final.py` with a FLOP-negative Pareto knob at both ends** (compute-lean +0.0123 @
   0.49×; accuracy-max +0.0212 @ 0.93×), the accuracy-max mode flipped FLOP-negative by F8's cheaper certified veto.
2. **The MCQ accuracy-beat over always-32B is intrinsically bounded to PMC + MMMU** — confirmed 4 independent ways this
   session (F8/F7/F11 decision-level, logit-level fusion, and H4/H8/H2 slice structure). This **sharpens** the story.
3. **F1's CI-lower-bound guardrail is validated** as the correct, sufficient robustifier (H8); fancier actuarial
   shrinkage does not beat it.
4. **G4 image-token pruning** (projection −26% FLOP, needs a GPU radiology-safety confirm) and the **INT4 strong leg**
   (VRAM/energy/deployability, not FLOPs) are the two carried-forward projections.
5. **Abstention remains out of scope** — the method always answers; H3 (abstain-to-human) was excised, and F8's veto is
   an answer-producing gate (keeps the 7B answer), not a reject option.
