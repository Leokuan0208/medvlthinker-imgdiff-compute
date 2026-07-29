# METHOD IDEAS BACKLOG — cross-field mechanism transfers for the cascade + verifier

> ## ⛔ FORBIDDEN — ABSTENTION IS PERMANENTLY OUT OF SCOPE ⛔
> **Never propose, build, or test any ABSTENTION / reject-option / defer-to-human / selective-prediction-as-a-method
> idea in this project — a standing, non-negotiable user rule.** The method must ALWAYS produce an answer.
> You MAY reuse math from the selective-prediction/conformal domain only for answer-PRODUCING mechanisms
> (e.g. a certified veto that still answers). H3 (abstain-to-human) was excised. If unsure whether a mechanism
> counts as abstention, it does — skip it.

> **What this is.** A LIVING idea backlog for the unified test-time-compute method (cheap 7B medical VLM +
> trained best-of-N **outcome verifier** + confidence gate, cascading to a strong 32B; scored on BOTH accuracy
> AND latency/compute). Each entry maps a mechanism from a RELATED or (mostly) UNRELATED field onto a concrete,
> testable variation for our setup. **Append + re-rank; do not overwrite.** No fabricated experimental numbers —
> project numbers cited below are real (from `UNIFIED_METHOD_EXPERIMENTS.md` / `conference_2026-07.md`); every
> "expected effect" is an explicit HYPOTHESIS to be tested, not a result.
>
> **Entry schema:** name · source field + key paper (arXiv/citation) · mechanism · map to our cascade/verifier ·
> concrete testable variation · expected ACC effect / expected LATENCY-COMPUTE effect · how to test (OFFLINE on
> per-sample dumps vs a GPU job to queue) · novelty & risk.
>
> **The three binding limits every idea should be judged against (from our own analysis):**
> 1. **Candidate quality is the wall.** oracle@8 ≫ verifier-bo8 everywhere (Lingshu pooled 0.513 vs 0.414); a
>    cross-model candidate pool raises oracle by **+0.11–0.15**; sc32 adds +0.08–0.13. This is where the accuracy
>    headroom lives.
> 2. **Verifier selection efficiency ≈ 74–82 %** (per-answer AUROC 0.90–0.93 but it loses within-question near-ties).
> 3. **The recoverability wall** (Jitkrittum 2307.02764): the strong model fixes only 6–26 % of cheap errors, and
>    *which* is near-unlearnable (AUROC ≈ 0.6). The gate is already near-optimal (verifier-confidence / margin).
> 4. **Cost tension:** best-of-N base cost = 2N cheap forwards ⇒ FLOP break-even vs one 32B forward is **N ≤ 2**;
>    both-axes wins today only exist where the strong model is weak/OOD. At batch-1 the 32B is only ~1.9× the 7B
>    latency (bandwidth-bound), so latency ≠ FLOPs.
> Updated 2026-07-06 (pass 2: +12 cross-field ideas → **35 total**; re-ranked). Pass-2 ideas target the two now-binding
> limits (candidate quality / oracle@N; verifier near-tie selection) and are tagged NEW below.
>
> **Updated 2026-07-07 (pass 3: +11 cross-field ideas → 46 total).** Pass-3 targets a **NEW AXIS** set by the refined
> goal: a single Lingshu-7B→32B cascade that is **MORE ACCURATE than always-32B** (not just faster-at-parity). All
> pass-3 ideas are in the new **§F** and tagged **[BEAT-32B]**. They exploit the project's own complementarity facts
> (oracle-union > either member; 7B beats 32B on MMMU +0.167; error-φ ≈ 0.37–0.52 so the two legs fail on *different*
> items) by **fusing/arbitrating the 7B AND 32B outputs to exceed the strong member** — a mechanism family the existing
> 35 ideas (all oriented at *matching* the 32B or the cheap-ensemble ceiling more cheaply) do not cover. The hard
> constraint on this axis is the **recoverability wall** (per-sample "will the 32B fix the 7B?" AUROC ≈ 0.6 on MCQ,
> ≈ 0.87 on open-text): §F ideas are designed to *sidestep* it — route/fuse on **observable slices**, **calibrated
> confidence-advantage**, or **certified high-precision regions** (all higher-AUROC than per-sample recoverability),
> and concentrate the sample-level beat-32B claim on **open-text** where the wall is weakest. §F has its own
> re-rank + top-5-offline flag; the §A–E table below still governs the *efficiency/match* goal.
>
> **Updated 2026-07-07 (pass 3b: +10 cross-field ideas → 56 total; new §G).** Pass-3b targets yet another axis —
> **make escalation CHEAPER and RARER** so the cascade is faster (the FALC speed win collapses wherever the 7B
> escalates heavily: MedXpert 90%, open-text 53–91%, since one 32B forward = 4.57× a 7B / ~665 ms). All 10 are in the
> new **§G**, tagged **★ESC**, with their own re-rank + top-5-offline flag. Running total: 35 (§A–E) + 11 (§F) +
> 10 (§G) = **56 ideas**.
>
> **Updated 2026-07-07 (pass 4: +12 cross-field ideas → 68 total; new §H).** Pass-4 attacks the **remaining headroom
> GIVEN the walls are established** — every idea is deliberately chosen to NOT rely on a better per-sample
> *recoverability* signal or better *best-of-N selection* (both are walled), and instead sidesteps them by adapting the
> cheap model, routing on observable/neighborhood/learned slices, keeping-cheap on the futile mass, or adding external
> (symbolic / calibration / estimation) structure. Mechanisms + fields: test-time training (H1), retrieval/kNN gating
> (H2), learned error-slice discovery (H4), JIT tiered-compilation
> escalation (H5), cost-based query-planner routing (H6), network-QoS admission control (H7), actuarial credibility
> shrinkage (H8), neuro-symbolic answer constraints (H9), multicalibration (H10), real-time imprecise-computation
> scheduling (H11), active-testing guardrail certification (H12). All in new **§H**, tagged **◆PASS4**, with their own
> re-rank + top-5-offline flag. (Requested fields already covered elsewhere and NOT re-proposed: multi-agent debate →
> §F9; semantic edge-caching → §G7; anytime scheduling → §D4; conformal-risk UQ → §C5/§F8; triage cost-asymmetry → §E2.)
> Running total: 35 (§A–E) + 11 (§F) + 10 (§G) + 12 (§H) = **68 ideas**.

---

## TL;DR — re-ranked after pass 2 (35 ideas)

**Overall TOP 5 TO TEST FIRST** (both-axes potential × novelty × testability):

1. **Pandora's-Box adaptive controller** (C1, Weitzman reservation values) — one optimal rule unifying adaptive-N *and*
   the escalation gate; **15–35 % fewer generations at matched accuracy** (arXiv 2510.01394, confirmed). OFFLINE.
2. **Diversity-maximized candidate set** (A1, DPP/MMR) — attacks the #1 limit (raise oracle@N by covering the answer
   space instead of redundant iid samples). OFFLINE + small GPU.
3. **Information-directed active-comparison verifier** (C9, TrueSkill μ,σ) — **★NEW** — attacks the #2 limit (near-tie
   selection) AND cuts verify calls; strictly upgrades the blind knockout (B1). OFFLINE.
4. **Generator portfolio from the error-correlation matrix** (A2, Markowitz) — attacks #1 (oracle@budget) on dumps we
   already have. OFFLINE.
5. **Speculative cascade — 32B as a *verifier*** (D1, Narasimhan 2405.19261) — strongest both-axes structural lever;
   resolves the cost tension (escalation = one cheap verify forward). Small GPU pass, then OFFLINE sweep.

**★ TOP 5 NEW ideas this pass** (all cross-field; ranked among the 12 new; two also entered the overall top 5):

1. **C9 — Information-directed active-comparison verifier** (sequential analysis / TrueSkill / IDS) — limit #2,
   both-axes (fewer verify calls), OFFLINE — upgrades B1's blind knockout with uncertainty-aware ratings.
2. **B5 — Dawid–Skene truth inference** (crowdsourcing) — limit #2, **aggregate-don't-select**; OFFLINE on the
   multi-source per-sample dumps we already have.
3. **A8 — Repulsive / semantic-guided diverse decoding** (SemDiD/SVGD, arXiv 2506.23601) — limit #1, generation-time
   coverage that raises oracle@N *without* A1's M>N over-generation.
4. **C7 — Best-arm-identification bandit allocation** (pure-exploration bandits) — spend the N-sample budget adaptively
   across generators for oracle-coverage; both-axes, OFFLINE.
5. **B6 — Surprisingly-popular / Bayesian Truth Serum** (mechanism design) — limit #2, unsupervised; beats the
   confident-wrong "majority trap" without a trained verifier.

Next tier of new ideas: **A6** (NCL + bias-variance-diversity *headroom diagnostic* — measures whether limit #1 is even
beatable), **A7** (Quality-Diversity / MAP-Elites answer-niche illumination), **B7** (Kemeny/Borda rank aggregation over
verifier views), **B9** (Product-of-Experts soft fusion), **B8** (ECOC-robust MCQ verify), **A5** (fountain-code
coverage generation), **C8** (uncertainty-aware batch allocation). Incumbent runners-up: **C2/C3** (VOC/SPRT controllers
with guarantees) and **B2** (the generative grounding-verifier frontier).

---

## A. CANDIDATE-QUALITY / GENERATION LEVERS (attack the #1 binding limit: oracle@N)

### A1. Diversity-maximized candidate set (DPP / MMR)  ★TOP-5 (#2)
- **Source:** Recommenders / IR / diverse decoding. Kulesza & Taskar, *Determinantal Point Processes for ML*
  (FnT 2012); Carbonell & Goldstein *MMR* (SIGIR 1998); *Enhancing Diversity in LLMs via DPP* (OpenReview v74LJpeUYX).
- **Mechanism:** select a subset that is jointly high-quality AND mutually dissimilar (log-determinant / greedy MMR),
  penalizing redundancy so the set *covers* the option space rather than piling on near-duplicates.
- **Map to us:** our binding limit is oracle@N — *does the correct answer appear among the N?* N iid temperature
  samples are redundant. Replace them with a DIVERSE set (DPP/MMR over candidate embeddings, or diverse prompts /
  temperatures) to raise oracle@N at fixed N — more coverage per unit compute, then the verifier selects.
- **Concrete variation:** over-generate M > N, embed candidates, DPP/MMR-select N; measure oracle@N vs iid@N. Also a
  "diverse-prompt" generator (paraphrases; "focus on the left lung", "consider the modality first").
- **Expected ACC:** raises the ceiling the verifier picks from — the highest-headroom accuracy lever. **Latency/compute:**
  equal-or-lower N for the same oracle ⇒ can *reduce* cost while lifting accuracy (both-axes-plausible).
- **How to test:** OFFLINE if any dump has M > 8 candidates or candidates are embeddable (re-select from existing
  sc32 / cross-model dumps); GPU to generate diverse-prompt candidates fresh.
- **Novelty & risk:** high — coverage-as-diversity aimed at the exact binding limit. Risk: diversity can inject
  confident-but-wrong distractors that fool the verifier (selection-efficiency drop) — must measure oracle AND selection.

### A2. Generator portfolio from the error-correlation matrix (Markowitz)  ★TOP-5 (#3)
- **Source:** Portfolio theory / finance. Markowitz, *Portfolio Selection* (J. Finance 1952).
- **Mechanism:** maximize expected return for a risk budget by combining low/negatively-correlated assets.
- **Map to us:** the cross-model pool already gives **+0.11–0.15** oracle *because models fail on different questions*
  (low error correlation; we measure φ = 0.372 for 7B/32B). Formalize: choose the set of cheap generators
  {Lingshu-7B, MVT-7B, IV3-8B} × {temps} × {prompts} that maximizes pooled-oracle coverage under a compute budget,
  using the measured pairwise error-correlation matrix (submodular / mean-variance objective).
- **Concrete variation:** from dumps, build the per-question error-correlation matrix across generators; greedily
  select the budget-B portfolio maximizing union-coverage; compare oracle@budget vs best-single and vs uniform pool.
- **Expected ACC:** maximizes oracle@budget (the binding lever). **Latency/compute:** explicit budget constraint —
  spend the same FLOPs but on de-correlated sources.
- **How to test:** OFFLINE — we already have multi-model per-sample dumps (Lingshu/MVT/IV3) with judge labels.
- **Novelty & risk:** medium-high — turns the anecdotal "cross-model pool helps" into a principled construction tied
  to our own φ. Risk: correlation estimates are noisy per-domain; a weak model can dilute the verifier's job.

### A3. Boosting: residual-specialist generator (AdaBoost / gradient boosting)
- **Source:** Ensembles / boosting. Freund & Schapire, *AdaBoost* (1997); gradient boosting (Friedman 2001).
- **Mechanism:** train each new learner on the RESIDUAL — up-weight the examples the current ensemble gets wrong.
- **Map to us:** train a "specialist" cheap generator (LoRA on a 7B) on the **oracle-miss** set (questions where the
  base 7B's best-of-N never contains the answer) so it contributes COMPLEMENTARY candidates that raise oracle@N
  exactly where the base pool fails — a boosting round on the candidate *pool*, complementing A2's off-the-shelf pool.
- **Concrete variation:** identify oracle-miss questions from dumps; LoRA-tune a 7B on their gold answers (or
  hard-example-upweighted train data); add its samples to the pool; measure new oracle@N on held-out.
- **Expected ACC:** targets the precise residual the pool misses. **Latency/compute:** +one generator's samples.
- **How to test:** GPU (train specialist + generate); pooling + oracle recompute OFFLINE.
- **Novelty & risk:** medium — boosting the candidate pool rather than a classifier. Risk: overfits; needs
  train/test disjointness; the residual may be intrinsically ungroundable (same wall).

### A4. Two-tower recall→precision funnel reframing (recommender systems)
- **Source:** Recommenders. Two-tower retrieval + ranker; Covington et al. *YouTube recommendations* (RecSys 2016);
  the standard candidate-generation → ranking funnel.
- **Mechanism:** cheap recall stage maximizes candidate recall; expensive ranker maximizes precision; each stage is
  tuned to ITS OWN metric, not the end metric.
- **Map to us:** reframe the whole method as **recall (generator: maximize oracle@N) → rank (verifier: precision)**.
  Consequence: tune generation hyper-parameters (temp, N, prompt mix) to maximize *oracle@N on a dev set* (a recall
  objective) — NOT greedy accuracy — and tune the verifier separately. Explains why "greedy-optimal generator" is the
  wrong target for a best-of-N pipeline.
- **Concrete variation:** grid generation settings by oracle@N (recall) independent of the verifier; report a
  recall–precision funnel curve; pick N at the recall knee.
- **Expected ACC:** aligns each stage with the right metric ⇒ both improve. **Latency/compute:** N set at the knee.
- **How to test:** OFFLINE (recompute oracle@N under settings we have) + GPU for new settings.
- **Novelty & risk:** medium — a clean reframing that reorganizes the search space. Risk: mostly framing; gains
  bounded by achievable candidate diversity (couples to A1/A2).

### A5. Fountain / rateless (LT) codes → difficulty-adaptive coverage generation  ★NEW
- **Source:** Coding theory. Luby, *LT Codes* (FOCS 2002); MacKay, *Fountain codes* (IEE Proc. Comms. 2005); an LLM
  reliability-operators framing lists "rateless sampling" (arXiv 2605.09121).
- **Mechanism:** rateless codes emit a *limitless* stream of encoding symbols from a fixed source; the receiver decodes
  as soon as it has collected any slightly-super-critical subset — no pre-set rate, so the symbol count adapts to the
  (unknown) erasure rate.
- **Map to us:** treat difficulty as an unknown "erasure rate." Instead of fixed N, generate candidates as a rateless
  *stream*, "decode-complete" when a coverage test fires (verifier-score plateau, or semantic-agreement mass crosses a
  threshold). Easy questions decode after 1–2 draws; hard ones draw more — difficulty-adaptive N with a coding-theoretic
  stopping rule aimed at oracle-coverage (a coverage view, vs C1/C3's reward/likelihood views).
- **Concrete variation:** define a decode-success test on the growing set (top verifier score > θ AND runner-up gap > δ,
  OR cluster-mass > m); simulate stream-until-decode on scores[8]; report accuracy vs mean-N and vs fixed-N.
- **Expected ACC:** ≈ best-of-N at the covered ceiling. **Latency/compute:** fewer draws on easy Qs, more only where
  needed (both-axes, adaptive).
- **How to test:** OFFLINE (stopping-rule re-sim on existing scores[8] + oracle labels).
- **Novelty & risk:** high — a coding-theoretic coverage view of adaptive-N. Risk: the decode test is heuristic; the
  coding analogy is loose for discrete answers; correlated draws blunt the coverage argument.

### A6. Negative-correlation learning + bias–variance–diversity headroom diagnostic  ★NEW
- **Source:** Ensemble theory. Wood et al., *A Unified Theory of Diversity in Ensemble Learning* (JMLR 24(359), 2023);
  Krogh & Vedelsby ambiguity decomposition (NIPS 1995); Liu & Yao *Negative Correlation Learning* (1999); Page
  diversity-prediction theorem (2007).
- **Mechanism:** ensemble error = average member error − diversity (the ambiguity / bias-variance-diversity
  decomposition); NCL trains members with a penalty that de-correlates their errors, raising the diversity term.
- **Map to us:** two uses. (a) **Diagnostic (OFFLINE, high-value):** decompose our multi-generator dumps into
  bias/variance/diversity to *quantify how much oracle@N headroom the pool's diversity can buy* — tells us whether
  limit #1 is beatable by more/de-correlated generators or is bias-bound. (b) **Lever:** train the cheap generator LoRAs
  with an NCL-style error-de-correlation objective so misses are complementary (A2 *selects* off-the-shelf sources; A6
  *trains for* de-correlation).
- **Concrete variation:** compute the BVD decomposition across {Lingshu, MVT, IV3}×temps on dumps; then LoRA-train a 7B
  with an NCL penalty against the frozen base and measure the new diversity term + oracle@N on held-out.
- **Expected ACC:** the diagnostic bounds achievable oracle@N; NCL raises it where diversity (not bias) is the limiter.
  **Latency/compute:** diagnostic is free; the trained pool adds one generator's cost.
- **How to test:** OFFLINE (decomposition on dumps); GPU (NCL LoRA).
- **Novelty & risk:** medium-high — turns "cross-model helps" into a measured decomposition + a training objective.
  Risk: 0/1-loss diversity is label-distribution-dependent (Wood's caveat); NCL on generative outputs is non-standard.

### A7. Quality-Diversity / MAP-Elites — illuminate answer niches  ★NEW
- **Source:** Evolutionary computation. Mouret & Clune, *Illuminating search spaces / MAP-Elites* (arXiv 1504.04909);
  Lehman & Stanley novelty search; QD-for-LLM (arXiv 2605.09781).
- **Mechanism:** rather than optimize one point, maintain an archive of the best solution per behavioral *niche* —
  "illuminating" the whole behavior space (coverage-first, quality-per-niche).
- **Map to us:** define behavior descriptors for VQA candidates (chosen option; modality-focus; commit-vs-hedge;
  attended region) and generate to fill the niche archive → each plausible answer region is *represented* before the
  verifier selects; attacks oracle@N as coverage of behavior space (complements A1's dissimilarity and A2's portfolio).
- **Concrete variation:** re-bin existing candidates into behavior niches to measure current coverage vs oracle-miss;
  then QD-guided prompting ("answer as if the modality is X", "focus on region Y") to fill empty niches; measure new
  oracle@N.
- **Expected ACC:** raises oracle@N by covering under-represented answer regions. **Latency/compute:** archive-filling
  generation, cap at #niches (budget-bounded).
- **How to test:** OFFLINE (re-binning coverage audit); GPU (QD-guided generation).
- **Novelty & risk:** high — illumination/coverage framing new to medical-VQA best-of-N. Risk: hand-designed
  descriptors; niches may re-inject confident distractors (measure selection too).

### A8. Repulsive / semantic-guided diverse decoding at generation time (SVGD / SemDiD)  ★NEW-top-5
- **Source:** Diverse decoding. *SemDiD: Semantic-guided Diverse Decoding* (arXiv 2506.23601, improves oracle/coverage
  in the best-of-N regime); Stein self-repulsive dynamics (arXiv 2002.09070); determinantal beam search; Arithmetic
  Sampling (arXiv 2210.15458).
- **Mechanism:** inject a **repulsion force during decoding** (Stein/DPP/semantic) so the N samples spread across
  semantic space *as they are generated* — diversity by construction, not iid temperature.
- **Map to us:** A1 over-generates M>N then DPP-*selects*; A8 instead *generates* N already-spread candidates in one
  pass, raising oracle@N without paying for M-over-generation — the cheaper route to the same coverage lever.
- **Concrete variation:** decode N candidates with semantic repulsion (penalize similarity to already-drawn samples'
  embeddings); compare oracle@N and verifier-selected accuracy vs iid-temperature@N and vs A1's over-generate-then-DPP
  at equal N.
- **Expected ACC:** higher oracle@N per unit compute. **Latency/compute:** N draws, no M>N over-generation ⇒ cheaper
  than A1 for the same coverage (both-axes-plausible on the generation side).
- **How to test:** GPU (repulsive decoding) + OFFLINE oracle/selection scoring; partial OFFLINE proxy via existing
  diverse-prompt/temperature dumps.
- **Novelty & risk:** medium-high — generation-time coverage vs A1's post-hoc selection; not applied to medical VLM
  best-of-N. Risk: repulsion can push samples off-manifold (fluent-but-wrong), hurting selection; embedding choice matters.

---

## B. VERIFIER / SELECTION LEVERS (attack the #2 binding limit: 74–82 % selection efficiency)

### B1. Pairwise / knockout-tournament verifier  ★TOP-5 (#5)
- **Source:** Reward modeling. *PairJudge RM / Pairwise RM: Best-of-N with Knockout Tournament* (arXiv 2501.13007).
- **Mechanism:** compare candidates PAIRWISE (which is more correct?) and run a knockout tournament, instead of
  assigning arbitrary/inconsistent absolute scores — more stable, directly optimizes RANKING.
- **Map to us:** our failure is within-question near-ties where a wrong candidate outscores the right one under
  POINTWISE scoring. A pairwise verifier "(image, q, ansA, ansB) → A/B, which grounds better?" targets exactly this.
- **Concrete variation:** LoRA-train a pairwise head on our ~6k pairs re-formed as within-question A/B comparisons;
  run a knockout over the N candidates; compare selection accuracy vs pointwise argmax on the SAME dump candidates.
- **Expected ACC:** directly attacks the selection ceiling (0.74–0.82 → ?). **Latency/compute:** O(N)–O(N log N)
  verify calls (more than pointwise N) — a compute-for-accuracy trade on the verify side.
- **How to test:** GPU (train pairwise verifier + score pairs); the tournament bracket is OFFLINE.
- **Novelty & risk:** high — the project tried pointwise + a Bradley-Terry *auxiliary loss* (AUROC 0.90→0.93,
  selection flat) but never true pairwise-comparison *inference*. Risk: non-transitive comparisons; cost grows with N.

### B2. Generative / CoT grounding verifier with verification-time scaling (GenRM) — THE FRONTIER
- **Source:** Reward modeling. Zhang et al. *Generative Verifiers: Reward Modeling as Next-Token Prediction*
  (arXiv 2408.15240) — 16–40 % Best-of-N gains, supports CoT + majority-voted verification.
- **Mechanism:** the verifier emits a CoT critique then Yes/No as next-token prediction; can be sampled and
  majority-voted (verification-time scaling); jointly trained on generation + verification.
- **Map to us:** replace the pointwise P(Yes) LoRA verifier with a GENERATIVE one that reasons about grounding
  ("the image shows X, the answer claims Y, therefore…") before its verdict — and, crucially, it scores GENERATED
  answers uniformly, so ONE verifier handles MCQ (letter = degenerate answer) + open-text + box captions = the stated
  **unified generative verifier** frontier.
- **Concrete variation:** LoRA-train Lingshu-7B to emit a short grounding rationale + Yes/No on the same 6k pairs;
  best-of-N via majority-voted P(Yes) over K verifier samples; also run it as an MCQ option-reranker to test unification.
- **Expected ACC:** targets the selection-efficiency ceiling on near-ties; unification could add an MCQ accuracy lever.
  **Latency/compute:** the verifier now decodes a rationale (costlier per verify) — compute-for-selection trade.
- **How to test:** GPU (train + score); selection-rule and K-sweep OFFLINE.
- **Novelty & risk:** high — a generative grounding verifier for medical VQA + the MCQ/open/box unification is novel
  and is the project's frontier. Risk: prior evidence a bigger (zero-shot 32B) verifier did NOT help ⇒ the rationale
  may not add grounding; the ceiling may be intrinsic. This is the highest-upside, highest-uncertainty bet.

### B3. Semantic-cluster-then-verify (universal self-consistency × trained verifier)
- **Source:** Test-time scaling. *Universal Self-Consistency* (arXiv 2311.17311); *Atomic Self-Consistency* (2405.13131).
- **Mechanism:** cluster free-form candidates by MEANING (LLM-judged equivalence) before aggregating, so SC works for
  open-text and paraphrases don't split the vote.
- **Map to us:** the "majority trap" the project found is on RAW strings. Cluster the N candidates semantically first,
  then have the trained verifier score CLUSTERS (or vote within clusters) — collapsing paraphrases of the correct
  answer de-traps the near-ties (e.g. "Both" / "bilateral lungs" / "both lungs" become one cluster).
- **Concrete variation:** cluster the 8 candidates (embedding or judge-equivalence), aggregate verifier scores per
  cluster, pick the top cluster's best representative; compare vs pointwise argmax on dumps.
- **Expected ACC:** may lift selection efficiency by removing vote-splitting among correct paraphrases.
  **Latency/compute:** +cheap clustering step only.
- **How to test:** OFFLINE — cluster existing candidates (we have judge equivalence + can embed) and re-score.
- **Novelty & risk:** medium — USC-weighted-by-a-trained-verifier, not plain USC (which our SC finding suggests is
  weak alone). Risk: the project's SC-flavored aggregation HURT at string level; clustering must be semantic to differ.

### B4. Process-reward tree search for the reasoning regime (PRM)
- **Source:** Process reward models. VL PRM *TIM-PRM* (arXiv 2511.22998); medical reasoning RFT (Reason-RFT 2503.20752).
- **Mechanism:** a step-level verifier scores intermediate reasoning steps to guide beam/tree search — helps hard
  multi-step reasoning where outcome verification fails.
- **Map to us:** MedXpert / MMMU are the regime we call hopeless (outcome verifier + cascade can't help; 7B near-floor).
  A PROCESS verifier over the 32B-think trace could select among reasoning *paths* (step-level best-of-N / tree search),
  attacking the reasoning floor the outcome verifier can't touch.
- **Concrete variation:** generate multiple 32B-think traces on MMMU/MedXpert; train/apply a step verifier; tree-search
  select; compare vs single think pass.
- **Expected ACC:** potential gain on the untouched reasoning regime. **Latency/compute:** expensive (multiple think
  traces + step scoring) — a pure accuracy play for hard reasoning, not an efficiency play.
- **How to test:** GPU (generate traces + train PRM).
- **Novelty & risk:** high for medical VLM reasoning. Risk: hardest regime; data-hungry (needs step labels); may not
  beat the outcome verifier; highest cost.

### B5. Dawid–Skene truth inference — reliability-weighted aggregation (don't select)  ★NEW-top-5
- **Source:** Crowdsourcing truth inference. Dawid & Skene, *Maximum Likelihood Estimation of Observer Error-Rates*
  (J. Royal Stat. Soc. C, 1979); pairwise-co-occurrence identifiability (arXiv 1909.12325); LLM-ensemble DS (WISE,
  arXiv 2512.02405).
- **Mechanism:** EM jointly estimates each *source's* confusion matrix and the latent true label; sources are weighted
  by estimated reliability and their votes combined into a posterior over the answer — provably beats majority vote
  under heterogeneous reliability.
- **Map to us:** our selection ceiling is an *argmax* failure on near-ties. Treat the N candidates (across
  generators/temps/prompts) as noisy annotators, run DS-EM to get a *posterior over the answer* weighted by each
  source's learned reliability, then take the posterior mode → **aggregate instead of select**, directly de-noising
  near-ties. Unsupervised (no verifier labels), but composes with the verifier as one extra source.
- **Concrete variation:** OFFLINE, run DS-EM over the multi-source per-sample answers we already dump (Lingshu/MVT/IV3 ×
  temps); compare posterior-mode accuracy vs verifier-argmax and vs majority; use the correlated / co-occurrence DS
  variant for within-question correlation.
- **Expected ACC:** lifts selection on near-ties where a reliable minority is right. **Latency/compute:** +cheap EM only
  (no extra inference).
- **How to test:** OFFLINE — existing multi-source dumps + judge labels.
- **Novelty & risk:** high — crowd truth-inference for medical-VQA candidate aggregation is new; unsupervised. Risk: DS
  assumes conditional independence, but our samples are correlated ⇒ must use the correlated/co-occurrence variant or
  reliability is over-credited.

### B6. Surprisingly-popular / Bayesian Truth Serum selection  ★NEW-top-5
- **Source:** Mechanism design. Prelec, *A Bayesian Truth Serum for Subjective Data* (Science 2004); Prelec, Seung &
  McCoy, *A solution to the single-question crowd-wisdom problem* (Nature 2017, "surprisingly popular").
- **Mechanism:** ask each voter BOTH their answer and their *prediction of others'* answers; pick the answer that is
  **more popular than the crowd predicted** — provably recovers the truth even when the confident majority is wrong.
- **Map to us:** the project found a "majority trap" (confident-but-wrong consensus). Elicit, per sample, an answer AND
  a meta-prediction ("what fraction of runs will say X?"); select the surprisingly-popular option → attacks near-ties
  AND the majority trap **without a trained verifier**.
- **Concrete variation:** prompt for the meta-prediction (or approximate it from the model's own option-probability, or
  a second "predict the crowd" pass); compute actual-minus-predicted endorsement per option; select the max; compare vs
  majority and vs verifier-argmax on dumps.
- **Expected ACC:** targets exactly the confident-wrong-majority cases the verifier also misses. **Latency/compute:**
  +one cheap meta-prediction pass (≪ a 32B call).
- **How to test:** GPU (elicit meta-predictions) then OFFLINE selection sim; partial OFFLINE proxy from existing
  option-logprobs.
- **Novelty & risk:** high — mechanism-design selection, unsupervised, new to VQA best-of-N. Risk: a single model's
  meta-predictions may be miscalibrated/collapsed (mitigate with cross-model meta-prediction); needs option spread.

### B7. Rank aggregation / social choice over multiple verifier views (Kemeny / Borda / Condorcet)  ★NEW
- **Source:** Social choice. Kemeny-Young; Borda count; setwise Kemeny (arXiv 2304.14980); LLM rank-aggregation
  (arXiv 2406.11871).
- **Mechanism:** aggregate several noisy *rankings* into a consensus that minimizes total pairwise disagreement
  (Kemeny) or Borda-sums — robust to any single ranker's errors; Condorcet yields the pairwise-majority winner when one
  exists.
- **Map to us:** run the verifier K times under different *views* (grounding-prompt; option-order permutations to kill
  position bias; modality-focus), each producing a ranking of the N candidates; aggregate with Borda/Kemeny; pick the
  consensus top → de-noises the ranking that causes near-tie errors. Distinct from B1 (a single knockout) — this
  aggregates *multiple independent rankings*.
- **Concrete variation:** generate K verifier rankings per question (view perturbations); Borda- and Kemeny-aggregate;
  compare consensus-top accuracy vs single-view argmax and vs B1 knockout on dumps.
- **Expected ACC:** lifts selection by averaging out view-specific ranking noise (esp. position bias). **Latency/compute:**
  K verifier passes (compute-for-selection; Borda O(KN), Kemeny NP-hard but N small).
- **How to test:** OFFLINE if K verifier scorings exist / are cheap to generate; the aggregation is pure OFFLINE.
- **Novelty & risk:** medium-high — social-choice consensus over verifier views vs a single scorer. Risk: correlated
  views add little; Kemeny cost (mitigated by small N / Borda approximation).

### B8. Error-correcting output codes (ECOC) for robust MCQ verification  ★NEW
- **Source:** Coding theory × classification. Dietterich & Bakiri, *Solving Multiclass Problems via ECOC* (JAIR 1995,
  arXiv cs/9501101); integer-programming ECOC robustness (arXiv 2011.00144).
- **Mechanism:** encode each class as a codeword with large Hamming distance; answer a *battery of binary sub-questions*;
  decode to the nearest codeword — redundancy corrects a minority of wrong binary verdicts.
- **Map to us:** instead of scoring each option once (fragile on near-ties), pose a designed set of binary contrastive
  verify-queries ("is the answer in {A,C} vs {B,D}?", grouped by an ECOC codebook over the options/answer-clusters);
  decode the verifier's binary verdicts to the nearest option-codeword → error-correcting *selection* robust to
  individual verifier mistakes.
- **Concrete variation:** build an ECOC codebook over the N candidate answers (or option letters); derive the binary
  sub-comparisons from existing pairwise/group scores where possible, else a small GPU pass; decode by min-Hamming and
  compare selection accuracy vs pointwise argmax.
- **Expected ACC:** redundancy corrects the near-tie coin-flips that cap selection. **Latency/compute:** codeword-length
  binary verify calls (a compute-for-robustness trade; length is tunable).
- **How to test:** OFFLINE if binary sub-comparisons are derivable from current scores; else small GPU.
- **Novelty & risk:** medium-high — coding-theoretic redundancy for verifier selection is novel here. Risk: binary
  sub-questions may be correlated (breaks ECOC's independent-error premise); code-design overhead.

### B9. Product-of-Experts / logarithmic opinion-pool answer fusion  ★NEW
- **Source:** Probabilistic ML. Hinton, *Products of Experts* (Neural Comp. 2002); Genest & Zidek logarithmic opinion
  pools (1986); Jacobs et al. mixtures of experts (1991).
- **Mechanism:** fuse several experts' *distributions* multiplicatively (log-pool) rather than voting on hard labels —
  sharp where experts agree, and a reliability-weighted log-pool down-weights diffuse (uncertain) experts.
- **Map to us:** we dump per-candidate option-logprobs across generators. Instead of hard-vote/argmax, combine the
  *soft* per-option distributions via a reliability-weighted PoE (weights from held-out per-source accuracy) → a fused
  answer distribution that resolves near-ties by consensus *sharpness*, not a single score. Distinct from DS (B5,
  discrete confusion-matrix EM on hard labels) — B9 fuses continuous distributions.
- **Concrete variation:** OFFLINE, log-pool the per-source option-logprobs with learned temperatures/weights; take the
  fused argmax; compare vs majority, vs verifier-argmax, vs DS on the same dumps.
- **Expected ACC:** lifts near-tie selection via soft consensus. **Latency/compute:** +cheap fusion only.
- **How to test:** OFFLINE (we have opt_logprobs).
- **Novelty & risk:** medium — a principled soft-fusion baseline the project hasn't run. Risk: PoE over-sharpens on
  correlated experts (needs reliability weights / temperature); MCQ-only unless open-text gets a scoring proxy.

---

## C. GATE / STOPPING / COMPUTE-ALLOCATION LEVERS (both-axes controllers)

### C1. Pandora's-Box adaptive candidate+escalation controller  ★TOP-5 (#1)
- **Source:** Economics of information / optimal stopping. Weitzman, *Optimal Search for the Best Alternative*
  (Econometrica 1979); correlated variant Gergatsouli & Tzamos (NeurIPS 2023); LLM bridge *Optimal Stopping vs
  Best-of-N for Inference Time Optimization* (arXiv 2510.01394).
- **Mechanism:** each "box" has an inspection cost c and a reward distribution; open boxes in decreasing **reservation
  value** ζ (solving E[(V−ζ)⁺]=c), and STOP when the best-so-far reward exceeds the next box's ζ. Provably optimal
  sequential search; the LLM bridge reports 15–35 % fewer generations at matched Best-of-N accuracy.
- **Map to us:** treat each generation source as a box — "next 7B sample #k" (cost 2 forward-units, reward = verifier
  score / P(correct)) and "the 32B" (cost 4.57, reward = its correctness distribution). Compute ζ per source; sample
  the 7B adaptively, ESCALATE when the 32B's ζ dominates, STOP when the current best verifier score beats all ζ. One
  optimal rule that UNIFIES adaptive-N and the escalation gate.
- **Concrete variation:** fit the verifier-score reward distribution from dumps; compute ζ for "another 7B sample" and
  "the 32B"; simulate the Weitzman policy on per-sample scores[8] + escalation outcomes; report accuracy vs mean-N/FLOPs.
- **Expected ACC:** ≈ best-of-8 selection at the same accuracy target. **Latency/compute:** 15–35 % fewer generations
  ⇒ both-axes on the compute side, adaptive per-question.
- **How to test:** OFFLINE — we have per-sample scores[8] and escalate outcomes; the policy is a re-simulation.
- **Novelty & risk:** high — Pandora's Box as the candidate/escalation controller is a clean, principled unification of
  our two separate levers. Risk: Weitzman assumes independent box rewards; within-question samples are CORRELATED ⇒
  use the correlated-Pandora variant, and sample order is non-trivial (mitigated by re-sim over orders).

### C2. VOC controller / rational metareasoning  ★close runner-up
- **Source:** AI / bounded rationality. Russell & Wefald, *Principles of Metareasoning* (AIJ 1991); Horvitz (flexible
  computation); Zilberstein (anytime / metareasoning).
- **Mechanism:** at each step take the computation (sample-more / escalate / stop) with the max **value of
  computation** VOC = expected decision-quality gain − compute cost; the principled objective behind any gate.
- **Map to us:** unify the gate + adaptive-N + tier-choice as one VOC decision, using the verifier's score
  distribution to estimate P(another 7B sample or the 32B changes the answer). Sweeping λ (the accuracy↔compute
  exchange rate) traces the ENTIRE Pareto frontier in one framework — directly the two-axis deliverable.
- **Concrete variation:** estimate ΔAcc(next 7B sample) from the empirical oracle@N curve and ΔAcc(escalate) from the
  recoverability rate conditioned on verifier confidence; act = argmax(ΔAcc − λ·cost); sweep λ; compare the traced
  frontier to the fixed-τ gate.
- **Expected ACC / Latency-compute:** by construction traces the frontier; the value is a single principled knob
  spanning both axes and subsuming C1/C3.
- **How to test:** OFFLINE on dumps.
- **Novelty & risk:** medium-high — the "right" objective; unifies our levers. Risk: myopic VOC underestimates
  multi-step value; ΔAcc(escalate) uses the weak recoverability signal (but VOC degrades gracefully to always-escalate).

### C3. SPRT / sequential acceptance-sampling gate  ★close runner-up
- **Source:** Manufacturing SQC / statistics. Wald, *Sequential Analysis / SPRT* (1945, minimal expected sample size
  — Wald–Wolfowitz optimality); Dodge–Romig double/multiple sampling; LLM use *ConSol* (arXiv 2503.17587).
- **Mechanism:** accumulate evidence sample-by-sample; ACCEPT (answer cheap) when the log-likelihood ratio crosses
  upper bound A, ESCALATE when it crosses lower bound B, else keep sampling — minimizes expected N for fixed
  type-I/II error.
- **Map to us:** the "test" is H0 = the cheap answer is correct vs H1 = needs the 32B; evidence accumulates as
  per-sample verifier scores. Gives adaptive-N AND the escalation gate as ONE sequential test with error-rate
  GUARANTEES — a formally-grounded alternative to a static τ.
- **Concrete variation:** compute the log-LR from the verifier-score distributions (correct vs incorrect); set A,B
  from a target error; simulate on dumps; report accuracy vs mean-N and vs the fixed-τ gate.
- **Expected ACC:** match at controlled error. **Latency/compute:** fewer samples on easy Qs, early-escalate obvious
  hard Qs (both-axes) with a Wald-optimality argument.
- **How to test:** OFFLINE.
- **Novelty & risk:** high — a sequential-test gate with error guarantees is distinct from all static-threshold gates
  we (and the literature) benchmarked. Risk: needs calibrated per-sample likelihoods; Wald overshoot approximations.

### C4. Kelly / knapsack batch compute allocation
- **Source:** Economics of information / optimization. Kelly criterion (1956); budgeted knapsack.
- **Mechanism:** allocate a fixed budget across bets proportional to edge/uncertainty to maximize expected growth.
- **Map to us:** given a BATCH compute budget (real serving amortizes compute over a batch), allocate samples/
  escalations ACROSS questions to maximize total accuracy — spend where marginal ΔAcc-per-FLOP is highest
  (uncertain-but-recoverable), starve saturated questions. A different optimization than a per-question threshold.
- **Concrete variation:** from dumps estimate each question's marginal ΔAcc(next sample) and ΔAcc(escalate); solve a
  knapsack for the batch budget; compare vs uniform-N and vs the per-question gate at equal total compute.
- **Expected ACC:** higher accuracy at fixed total compute. **Latency/compute:** fixed budget by construction (both-axes).
- **How to test:** OFFLINE.
- **Novelty & risk:** medium-high — batch-level allocation matches real serving and differs from every per-question
  gate we tested. Risk: needs a per-question marginal-value estimate (recoverability-adjacent, weak) — though the
  allocation is fairly robust to ranking noise.

### C5. Conformal escalation with a distribution-free risk guarantee
- **Source:** Conformal prediction. Angelopoulos et al. *Conformal Risk Control* (arXiv 2208.02814); *Selective
  Conformal Risk Control* (arXiv 2512.12844); split-conformal (LAC/APS).
- **Mechanism:** calibrate a threshold so the retained (non-escalated) set has provably ≤ α error, distribution-free
  and finite-sample; conformal SETS have a coverage guarantee (correct answer ∈ set w.p. 1−α).
- **Map to us:** turn the verifier-confidence gate into a CONFORMAL gate — pick τ on a calibration split so the
  cheap-answered subset's error ≤ α with a guarantee, escalate the rest. Bonus: a conformal candidate SET =
  a guaranteed oracle bound the verifier then selects within (hit a target oracle@N with minimal N, with a certificate).
- **Concrete variation:** split-conformal calibrate τ on held-out dumps; verify empirical coverage at several α;
  report accuracy/escalation at guaranteed risk; build a conformal candidate set + verifier-select.
- **Expected ACC:** same curve, but now with a SAFETY guarantee (clinical deployability). **Latency/compute:**
  escalation set as small as the guarantee allows.
- **How to test:** OFFLINE (calibration + coverage check on dumps).
- **Novelty & risk:** medium — a guaranteed-risk cascade gate + guaranteed-oracle conformal candidate set is a strong
  "safe medical deployment" framing. Risk: coverage assumes exchangeability, so OOD (RadImageNet) breaks the guarantee
  — but quantifying that break is itself a publishable finding.

### C6. Particle-physics trigger menu + rate-based (budget) gate
- **Source:** Experimental HEP. CMS/ATLAS multilevel L1/HLT trigger; "trigger menus"; "prescaling" (Commissioning of
  the CMS HLT, arXiv 0908.1065; ATLAS L1 topo, 2105.01416).
- **Mechanism:** triggers are tuned to a fixed OUTPUT RATE (bandwidth budget), not a per-event threshold; "trigger
  menus" send different event types down different reconstruction paths; "prescaling" randomly downsamples high-rate
  streams to fit the budget.
- **Map to us:** (a) a RATE-BASED gate — escalate exactly the fraction that fits a compute/latency budget (batch-level
  Neyman–Pearson quantile on verifier confidence) rather than a fixed τ, guaranteeing throughput; (b) a TRIGGER MENU —
  per-question-type escalation paths (radiology → 32B fullres, pathology → cap320, MCQ → margin gate, open → verifier).
- **Concrete variation:** replace fixed τ with a budget-B quantile gate (escalate the B% least-confident); add typed
  routing using the deterministic MCQ/open detector + modality; simulate accuracy vs budget on dumps.
- **Expected ACC:** ≈ the τ-gate at equal escalation. **Latency/compute:** HARD budget guarantee by construction
  (deployability), plus per-type efficiency (cap where the domain tolerates it — PMC cap320 is free).
- **How to test:** OFFLINE.
- **Novelty & risk:** medium — the accuracy curve equals τ's (rate-gate is a re-parameterization), so novelty is the
  throughput GUARANTEE + the typed menu, not an accuracy lift. Best as a deployment-story component.

### C7. Best-arm identification / pure-exploration bandit budget allocation across generators  ★NEW-top-5
- **Source:** Pure-exploration bandits. Audibert & Bubeck BAI (COLT 2010); Karnin et al. *Sequential Halving*
  (ICML 2013); Russo *Simple Bayesian Algorithms for BAI* (arXiv 1602.08448); combinatorial pure exploration
  (arXiv 1706.01081); BAI-with-LLM-judges (arXiv 2601.21471).
- **Mechanism:** given a fixed sampling budget, adaptively allocate pulls across heterogeneous *arms* to identify the
  best (Sequential Halving / LUCB / successive elimination) with minimal samples — spend where reward is most
  uncertain/promising.
- **Map to us:** the N-sample budget is currently spread uniformly (iid). Treat each generator/temp/prompt as an arm
  with the verifier as noisy reward; allocate the budget adaptively — pull the arms most likely to *produce/contain* the
  correct candidate; **combinatorial** pure exploration maps to picking the arm-*set* maximizing oracle-coverage.
  Distinct from C1 (reservation-value stopping) and C4 (deterministic knapsack) — this is fixed-budget *adaptive
  allocation across sources*.
- **Concrete variation:** OFFLINE, simulate Sequential-Halving / LUCB over per-sample verifier scores across our
  multi-generator dumps; report accuracy@budget and oracle-coverage@budget vs uniform iid allocation.
- **Expected ACC:** more correct candidates found per budget. **Latency/compute:** fixed budget by construction
  (both-axes).
- **How to test:** OFFLINE (re-sim on existing multi-generator scores).
- **Novelty & risk:** medium-high — adaptive cross-generator allocation vs uniform best-of-N. Risk: arm rewards are
  correlated within a question; verifier-as-reward inherits the selection-noise ceiling.

### C8. Uncertainty-aware / Thompson batch-budget allocation across questions  ★NEW
- **Source:** Adaptive test-time compute. *Uncertainty-Aware Budget Allocation* (arXiv 2605.26849); *Plan-and-Budget*
  (arXiv 2505.16122); Thompson sampling (1933); allocation-probability test (arXiv 2111.00137).
- **Mechanism:** reallocate a fixed *batch* sampling budget by per-question uncertainty — uncertain questions get more
  draws, confident ones fewer — with Thompson-style posterior draws propagating estimate uncertainty into allocation.
- **Map to us:** real serving amortizes compute over a batch. Allocate the next sample to the question whose
  *posterior-sampled* marginal ΔAcc is highest → differs from C4 (Kelly/knapsack needs point estimates) by being
  uncertainty-aware, and from BEST-Route (E3) by acting at the batch level.
- **Concrete variation:** estimate each question's improvability posterior from the oracle@N curve + verifier spread;
  Thompson-allocate a batch budget; compare accuracy@total-compute vs uniform-N and vs C4/E3.
- **Expected ACC:** higher accuracy at fixed batch compute. **Latency/compute:** fixed batch budget (both-axes).
- **How to test:** OFFLINE (re-sim on dumps).
- **Novelty & risk:** medium — overlaps C4/E3; the fresh angle is Bayesian/uncertainty-aware allocation matching
  batched serving. Risk: rests on per-question marginal-value estimates (recoverability-adjacent, weak).

### C9. Information-directed active-comparison verifier (TrueSkill μ,σ)  ★NEW-top-5 (overall #3)
- **Source:** Sequential decision + rating. Russo & Van Roy *Information-Directed Sampling* (2018); Herbrich et al.
  *TrueSkill* (NIPS 2007); active ranking from pairwise comparisons (arXiv 1109.3701); comparative LLM-judge uncertainty
  (arXiv 2505.15240).
- **Mechanism:** maintain an uncertainty-aware rating (μ,σ) per candidate; run only the *most informative* pairwise
  comparisons (max information-per-comparison), and stop when the top candidate's rating provably dominates — near-ties
  get resolved, settled pairs are skipped.
- **Map to us:** B1's knockout runs O(N) *blind* comparisons; C9 upgrades it — schedule comparisons by information gain
  and stop early via σ, so we (a) resolve near-ties with an uncertainty-aware rating (attacks limit #2) and (b) spend
  *fewer* verify calls on already-settled candidates (both-axes on the verify side). The verifier's own score-variance
  seeds σ.
- **Concrete variation:** OFFLINE, simulate adaptive-comparison TrueSkill over our existing pairwise/pointwise verifier
  scores; report selection accuracy AND #comparisons vs B1 blind knockout and vs pointwise argmax; sweep the σ
  stop-threshold for the accuracy↔verify-cost frontier.
- **Expected ACC:** better near-tie resolution than pointwise / blind-knockout. **Latency/compute:** fewer comparisons
  than knockout at equal accuracy (both-axes).
- **How to test:** OFFLINE (re-sim on existing scores); GPU only if a pairwise comparator (B1) must be trained first.
- **Novelty & risk:** high — information-directed *scheduling* of an uncertainty-aware rating is distinct from every
  static verifier we ran and strictly upgrades B1 on both axes. Risk: needs a calibrated per-comparison uncertainty;
  non-transitive comparisons perturb the rating (TrueSkill tolerates some).

---

## D. STRUCTURAL / CASCADE-ARCHITECTURE LEVERS

### D1. Speculative cascade — 32B as a *verifier*, not a generator  ★TOP-5 (#4)
- **Source:** Hardware speculative execution × speculative decoding. Narasimhan et al. *Faster Cascades via
  Speculative Decoding* (arXiv 2405.19261); Leviathan et al. speculative decoding (2211.17192).
- **Mechanism:** the small model drafts; the large model VERIFIES in parallel and only overrides where its deferral
  rule fires — large-model quality at reduced latency, and the large model does a cheap verify pass, not a full generate.
- **Map to us (answer-level):** the 7B best-of-N pick is the DRAFT; the 32B runs a single VERIFY forward (score the
  draft ≈ 0.5× a 32B generation, one forward) instead of generating; accept if 32B-verify agrees, else the 32B
  generates. Escalation ACTION becomes cheap-verify-first, full-generate-only-on-reject.
- **Concrete variation:** run the 32B as a verifier over the 7B drafts (P(Yes) / agreement); sweep the accept-draft
  deferral rule; the accepted fraction pays only a verify, the rejected fraction pays a generate.
- **Expected ACC:** approaches always-32B (the 32B adjudicates). **Latency/compute:** 32B-verify ≪ 32B-generate and
  only the reject fraction pays generation ⇒ the strongest both-axes structural lever, especially where today's
  best-of-N loses FLOPs (the competitive-strong regime, e.g. VQA-RAD).
- **How to test:** GPU pass to get 32B-verify scores on 7B drafts (queue); the deferral-rule sweep is then OFFLINE.
- **Novelty & risk:** high — speculative cascades are not applied to medical VLMs, and using 32B-VERIFY as the
  escalation action (vs 32B-generate) is new to our pipeline and directly addresses the cost tension. Risk: for very
  short medical answers the 32B verify vs generate cost gap shrinks; measure the real per-forward cost.

### D2. Eager parallel escalation / branch-prediction latency hiding
- **Source:** Computer architecture. Branch prediction (2-bit saturating counters; perceptron predictor, Jiménez &
  Lin 2001); speculative execution.
- **Mechanism:** a branch predictor speculatively runs the likely path and squashes on misprediction — hides pipeline
  latency at the cost of wasted work.
- **Map to us:** a cheap query-only escalation predictor PRE-LAUNCHES the 32B in parallel with the 7B on queries
  predicted to escalate; if the 7B turns out confident, squash the 32B. Removes the serial 7B→32B dependency for
  escalated queries — a pure LATENCY win (paid for in speculative FLOPs).
- **Concrete variation:** train a query-only predictor (question embedding + image stats, available BEFORE generation)
  like a saturating-counter/perceptron; measure hit rate; model latency saved vs FLOPs wasted at several predictor
  operating points.
- **Expected ACC:** unchanged. **Latency:** large win (escalated answers ready immediately). **Compute:** worse
  (speculative waste) — an explicit latency-vs-compute knob distinct from all our FLOPs-only levers.
- **How to test:** OFFLINE for predictor accuracy + a latency model on measured per-tier latencies; GPU for real
  wall-clock.
- **Novelty & risk:** high — latency-axis-specific, under-explored in cascades (which optimize FLOPs). Risk: only
  helps latency, costs FLOPs; value depends entirely on the deployment SLA / whether the 2nd GPU is idle.

### D3. Attentional cascade / early-reject multi-stage (Viola–Jones)
- **Source:** Computer vision. Viola & Jones, *Rapid Object Detection with a Boosted Cascade* (CVPR 2001).
- **Mechanism:** a chain of increasingly expensive classifiers; each early stage cheaply REJECTS easy negatives
  (here: confidently answers easy questions) so expensive stages see only survivors; each stage tuned for high recall
  to a target pass-rate.
- **Map to us:** generalize the 2-tier cascade to an N-stage cheap→expensive cascade over COMPUTE CONFIGS
  (7B@cap160 → 7B@cap320 → 7B best-of-2 → 7B best-of-8 → 32B), each passing only unconfident survivors. Our ACC is a
  3-tier special case; Viola–Jones supplies the design principle (set each stage's threshold to a target pass rate).
- **Concrete variation:** order compute configs by cost; set each stage's confidence threshold to pass only the
  unconfident k%; simulate accuracy/cost on cap80/160/320 + best-of-N dumps.
- **Expected ACC:** match. **Latency/compute:** the cheapest config handles the bulk (leverages "cap320 free on PMC").
- **How to test:** OFFLINE (we have cap80/160/320 dumps + best-of-N scores).
- **Novelty & risk:** medium — a multi-stage resolution/N cascade generalizing ACC. Risk: many stages add gating
  overhead + threshold-tuning fragility; per-benchmark guardrails must stay clean.

### D4. Anytime / contract cascade under a latency deadline
- **Source:** Anytime algorithms / bounded rationality. Zilberstein & Russell, anytime algorithms and performance
  profiles (1996); contract algorithms.
- **Mechanism:** an anytime algorithm returns a valid answer at any interruption, with quality rising over time;
  performance profiles let a scheduler allocate time under a deadline.
- **Map to us:** make the cascade ANYTIME — always hold a current-best answer (7B greedy → best-of-N pick → 32B), so
  under a latency SLA it returns the best-so-far; use the verifier's per-stage performance profile to schedule.
- **Concrete variation:** build the performance profile (accuracy vs elapsed time) from measured per-tier latencies +
  accuracies; simulate deadline-constrained serving at several SLAs.
- **Expected ACC:** maximized-given-deadline. **Latency:** SLA-guaranteed by construction. **Compute:** deadline-bounded.
- **How to test:** OFFLINE (profile from measured latencies) + GPU for real interrupts.
- **Novelty & risk:** medium — anytime framing for VLM serving under deadlines (latency axis + deployability). Risk:
  largely a serving wrapper; gains are SLA-dependent.

### D5. Orthogonal confirmatory check (high-throughput drug screening cascade)
- **Source:** Drug discovery. HTS screening cascades; orthogonal/confirmatory assays; hit triage (Technology Networks
  "How Screening Cascades Work"; Dahlin et al. on HTS false positives).
- **Mechanism:** a primary screen (cheap, high-throughput, many false positives) → an ORTHOGONAL confirmatory assay
  that measures the same target by a DIFFERENT technology; agreement across orthogonal methods filters artifacts.
- **Map to us:** escalate a cheap answer not (only) to a bigger model but to an ORTHOGONAL cheap check — a different
  modality-focused prompt, a re-phrased question, or a different cheap model — and treat orthogonal AGREEMENT as
  confirmation, DISagreement as the escalation trigger. Reduces correlated-failure false-confidence more cheaply than
  the 32B. (Note: the project's plain cross-model *agreement* gate was high-precision/low-coverage; the new angle is
  orthogonal-by-CONSTRUCTION probes, e.g. modality-specific prompts, not a second full model.)
- **Concrete variation:** generate an orthogonal probe (e.g. "describe only the anatomy, then answer" vs the base
  prompt); use base⊥probe agreement as a cheap confirm/escalate signal; compare its deferral curve to verifier-conf.
- **Expected ACC:** confirm-cheap could match at lower escalation where orthogonal agreement is reliable.
  **Latency/compute:** an orthogonal 7B probe ≪ a 32B call.
- **How to test:** GPU (generate orthogonal probes) then OFFLINE gate sim; partly OFFLINE if probe-like dumps exist
  (we have cap-perturbed generations already — a resolution-orthogonal probe).
- **Novelty & risk:** medium — orthogonal-probe confirmation vs same-model self-consistency. Risk: prior visual-
  stability (resolution-orthogonal) probe was a WEAK gate (AUROC ~0.60) redundant with the verifier — new probes must
  be genuinely orthogonal to beat that.

---

## E. FRAMING / DIAGNOSTIC LEVERS (explain the walls; guide where to spend compute)

### E1. Successive-refinement / rate-distortion diagnosis of the cascade
- **Source:** Information theory. Equitz & Cover, *Successive Refinement of Information* (IEEE-IT 1991); Koshelev.
- **Mechanism:** a source is "successively refinable" iff describing it coarsely then refining incurs NO rate loss vs
  one-shot coding at each distortion level; refinability requires a Markov condition between the coarse and fine
  descriptions.
- **Map to us:** is the 7B answer a refinable coarse version of the 32B answer, or an unrelated guess? If refinable,
  cascading is compute-optimal; if not (7B and 32B answers are unrelated on the errors), the 7B pass is wasted and
  direct-routing is better. This reframes the RECOVERABILITY WALL as "the error set is not successively refinable."
- **Concrete variation:** on dumps, among 7B-wrong questions measure how often the 32B answer is a REFINEMENT (same
  category / finer detail / contains the 7B answer) vs a REPLACEMENT (containment + category analysis); correlate
  refinability with recoverability.
- **Expected ACC / Latency-compute:** a diagnostic — identifies which slices to cascade vs route-direct, and explains
  the wall; informs where the 7B pass is wasted compute.
- **How to test:** OFFLINE (answer-relationship analysis on dumps).
- **Novelty & risk:** medium — a principled lens on the wall + a router-vs-cascade decision rule. Risk: mostly
  analysis, not a new method; "bits" are hard to operationalize for discrete answers.

### E2. Clinical-triage cost-asymmetry + learned "red-flags"
- **Source:** Emergency medicine. START / Manchester triage; over-triage vs under-triage cost asymmetry.
- **Mechanism:** triage uses an ASYMMETRIC cost (under-triage — missing a sick patient — costs far more than
  over-triage) and hard "red-flag" rules that force escalation regardless of the score.
- **Map to us:** set the gate by an explicit COST MATRIX (missing a recoverable error vs an unnecessary 32B call)
  rather than iso-accuracy, and add learned/hand-crafted "red-flags" (question mentions a critical finding; a
  modality × question-type slice where the 7B is systematically wrong) that force escalation on top of the verifier gate.
- **Concrete variation:** sweep the cost-asymmetry ratio to trace a cost-aware frontier; mine dumps for high-precision
  always-escalate slices (modality × q-type where 7B is reliably wrong); layer them on the verifier-conf gate.
- **Expected ACC:** higher accuracy in high-stakes slices at controlled compute. **Latency/compute:** targeted, small
  extra escalation.
- **How to test:** OFFLINE (cost sweep + red-flag mining on dumps).
- **Novelty & risk:** medium — a natural, publishable medical framing (cost-sensitive + red-flag escalation on the
  verifier). Risk: red-flag patterns need enough per-slice support; the cost matrix is application-specific.

### E3. BEST-Route joint (model, sample-count) policy — strong baseline to adopt
- **Source:** LLM routing. Ding et al. *BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute*
  (arXiv 2506.22716) — routes each query to a (model, #responses) pair by difficulty, −60 % cost at <1 % drop.
- **Mechanism:** small-model-multi-sample beats big-model-single when cheaper; learn a per-query policy over
  {small×N, large×1}.
- **Map to us:** exactly our two knobs (which model, how many samples). Learn a per-query policy over {7B×N, 32B×1}
  minimizing cost at a quality bar — a principled unification we should at least run as a strong baseline / adopt.
- **Concrete variation:** from dumps, for each query find the cheapest (model, N) that answers it correctly; train a
  difficulty predictor to choose; evaluate the cost–accuracy frontier vs our gate.
- **Expected ACC:** match. **Latency/compute:** up to their −60 % where the 7B×N path suffices.
- **How to test:** OFFLINE (we already have 7B×N and 32B outcomes).
- **Novelty & risk:** low (published) — value is as a baseline / framework to fold in, not a novel contribution. Risk:
  the difficulty predictor rests on the weak recoverability signal (our known wall).

### E4. Self-truncation / early-stop best-of-N
- **Source:** Test-time scaling. *Self-Truncation Best-of-N (ST-BoN)* (arXiv 2503.01422); *Self-Calibration*
  (2503.00031).
- **Mechanism:** stop generating the N samples early using self-estimated confidence in early decoding — avoids full
  N generations.
- **Map to us:** verifier-score-gated stopping — stop drawing samples once a high-verifier-score candidate appears (or
  truncate obviously-bad partial generations). Cuts N adaptively; a lightweight sibling of C1/C3.
- **Concrete variation:** simulate "stop at first score > θ" over the per-candidate scores[8] on dumps; sweep θ; report
  accuracy vs mean-N.
- **Expected ACC:** ≈ best-of-8. **Latency/compute:** fewer samples (compute-side both-axes).
- **How to test:** OFFLINE (order-dependent stopping sim on scores[8]).
- **Novelty & risk:** low-medium — a known lever; our angle is verifier-score gating. Risk: sample order matters;
  early-stop can miss a late correct sample (bounded by the oracle wall).

---

## PRIORITIZATION TABLE (both-axes × novelty × testability)

**NEW pass-2 ideas are tagged ★. Re-ranked over all 35.**

| # | idea | attacks | both-axes? | novelty | test cost | rank |
|---|---|---|---|---|---|---|
| C1 | Pandora's-Box adaptive controller | cost+gate+N | **yes** (−15–35 % gens) | high | OFFLINE | **1** |
| A1 | Diversity-max candidate set (DPP/MMR) | oracle@N (limit #1) | acc↑, cost≈/↓ | high | OFFLINE+GPU | **2** |
| C9 ★ | Info-directed active-comparison verifier (TrueSkill) | selection (#2) | **yes** (fewer verifies) | high | OFFLINE | **3** |
| A2 | Generator portfolio (error-corr matrix) | oracle@N (limit #1) | acc↑ @budget | med-high | OFFLINE | **4** |
| D1 | Speculative cascade (32B as verifier) | cost tension | **yes** | high | GPU+OFFLINE | **5** |
| B5 ★ | Dawid–Skene reliability-weighted aggregation | selection (#2) | acc↑, cheap | high | OFFLINE | 6 |
| B1 | Pairwise/knockout verifier | selection ceiling (#2) | acc↑ | high | GPU+OFFLINE | 7 |
| A8 ★ | Repulsive/semantic diverse decoding (SemDiD) | oracle@N (#1) | acc↑, cost↓ | med-high | GPU+OFFLINE | 8 |
| C7 ★ | Best-arm-identification bandit allocation | oracle@N + alloc | **yes** (@budget) | med-high | OFFLINE | 9 |
| C2 | VOC metareasoning controller | cost+gate+N | **yes** (traces frontier) | med-high | OFFLINE | 10 |
| C3 | SPRT sequential gate | cost+gate+N | **yes** (+guarantee) | high | OFFLINE | 11 |
| B6 ★ | Surprisingly-popular / Bayesian Truth Serum | selection (#2) | acc↑ | high | GPU+OFFLINE | 12 |
| B2 | Generative grounding verifier (GenRM) | selection + unification (frontier) | acc↑ | high | GPU | 13 |
| A6 ★ | NCL + bias-variance-diversity diagnostic | oracle@N (#1) | acc↑ + diagnostic | med-high | OFFLINE+GPU | 14 |
| C4 | Kelly/knapsack batch allocation | cost @budget | **yes** | med-high | OFFLINE | 15 |
| A7 ★ | Quality-Diversity / MAP-Elites niches | oracle@N (#1) | acc↑ | high | OFFLINE+GPU | 16 |
| A3 | Boosting residual-specialist generator | oracle@N (#1) | acc↑ | med | GPU | 17 |
| B7 ★ | Rank aggregation / social choice (Kemeny/Borda) | selection (#2) | acc↑ | med-high | OFFLINE | 18 |
| C5 | Conformal escalation guarantee | gate safety | acc≈ +guarantee | med | OFFLINE | 19 |
| B9 ★ | Product-of-Experts soft answer fusion | selection (#2) | acc↑, cheap | med | OFFLINE | 20 |
| B3 | Semantic-cluster-then-verify | selection (#2) | acc↑, cheap | med | OFFLINE | 21 |
| B8 ★ | ECOC robust MCQ verification | selection (#2) | acc↑ | med-high | OFFLINE/GPU | 22 |
| A5 ★ | Fountain/rateless coverage generation | oracle@N + adaptive-N | **yes** | high | OFFLINE | 23 |
| D2 | Eager parallel escalation | latency axis | latency↑, FLOPs↓? | high | OFFLINE+GPU | 24 |
| D3 | Attentional multi-stage cascade | cost | acc≈, cost↓ | med | OFFLINE | 25 |
| A4 | Two-tower recall→precision reframe | oracle@N framing | both (via A1/A2) | med | OFFLINE | 26 |
| C8 ★ | Uncertainty-aware / Thompson batch allocation | cost @budget | **yes** | med | OFFLINE | 27 |
| D5 | Orthogonal confirmatory probe | gate cost | cost↓ | med | GPU+OFFLINE | 28 |
| E2 | Triage cost-asymmetry + red-flags | gate (high-stakes) | acc↑ targeted | med | OFFLINE | 29 |
| C6 | HEP trigger menu + rate gate | budget guarantee | throughput guarantee | med | OFFLINE | 30 |
| B4 | Process-reward tree search | reasoning floor | acc↑ (hard regime) | high | GPU (costly) | 31 |
| D4 | Anytime deadline cascade | latency axis | latency guarantee | med | OFFLINE+GPU | 32 |
| E1 | Successive-refinement diagnosis | explains the wall | diagnostic | med | OFFLINE | 33 |
| E3 | BEST-Route (model,N) policy | cost | match, cost↓ | low (baseline) | OFFLINE | 34 |
| E4 | Self-truncation early-stop bo-N | cost | cost↓ | low-med | OFFLINE | 35 |

**Why the new top 5:** the two limits drive it. Limit #1 (oracle@N) is held by **A1** (diversity coverage) and **A2**
(de-correlated portfolio) — the accuracy headroom, both OFFLINE on existing dumps. Limit #2 (near-tie selection) gains a
new #3 in **C9**: information-directed active-comparison with an uncertainty-aware (TrueSkill μ,σ) rating strictly
upgrades the blind knockout (B1) by resolving near-ties AND cutting verify calls, all OFFLINE. **C1** stays #1 (the
one principled controller unifying adaptive-N + the gate, 15–35 % fewer generations, OFFLINE) and **D1** stays top-5 as
the strongest structural both-axes lever (escalation = a cheap 32B verify). Just below: **B5** (Dawid–Skene aggregation
— *aggregate-don't-select*, OFFLINE on multi-source dumps) and **A8** (generation-time repulsive decoding — coverage
without over-generation). The best OFFLINE, both-limits-attacking pass-2 ideas are C9, B5, C7, A5 (all re-simulatable on
current dumps); the highest-upside-but-GPU ones are A8, B6, A6, A7.

---

## SOURCES (real papers/concepts cited above)
- GenRM: arXiv 2408.15240 · Faster Cascades via Speculative Decoding: arXiv 2405.19261 · Speculative decoding:
  arXiv 2211.17192 · PairJudge/Pairwise RM: arXiv 2501.13007 · Universal Self-Consistency: arXiv 2311.17311 ·
  Atomic Self-Consistency: arXiv 2405.13131 · TIM-PRM: arXiv 2511.22998 · Reason-RFT: arXiv 2503.20752
- Optimal Stopping vs Best-of-N: arXiv 2510.01394 · ST-BoN: arXiv 2503.01422 · Self-Calibration: arXiv 2503.00031 ·
  BEST-Route: arXiv 2506.22716 · ConSol (SPRT for reasoning): arXiv 2503.17587
- RouteLLM: arXiv 2406.18665 · Hybrid LLM: arXiv 2404.14618 · Jitkrittum recoverability: arXiv 2307.02764 ·
  CCPS: arXiv 2505.21772 · Self-REF: arXiv 2410.13284
- Conformal Risk Control: arXiv 2208.02814 · Selective Conformal Risk Control: arXiv 2512.12844
- Weitzman, *Optimal Search for the Best Alternative*, Econometrica 1979 · correlated Pandora: Gergatsouli & Tzamos,
  NeurIPS 2023 · Wald, *Sequential Analysis* 1945 (SPRT; Wald–Wolfowitz optimality) · Dodge–Romig acceptance sampling
- Russell & Wefald, *Principles of Metareasoning*, AIJ 1991 · Zilberstein & Russell, anytime algorithms 1996 ·
  Horvitz, flexible computation
- Markowitz, *Portfolio Selection*, J. Finance 1952 · Kelly criterion 1956
- Kulesza & Taskar, *DPPs for ML*, FnT 2012 · Carbonell & Goldstein, *MMR*, SIGIR 1998 · DPP-for-LLMs: OpenReview v74LJpeUYX
- Viola & Jones, *Boosted cascade*, CVPR 2001 · Freund & Schapire, AdaBoost 1997 · Friedman, gradient boosting 2001
- Covington et al., *Deep NN for YouTube Recommendations*, RecSys 2016 (two-tower funnel)
- Equitz & Cover, *Successive Refinement of Information*, IEEE-IT 1991
- CMS HLT commissioning: arXiv 0908.1065 · ATLAS L1 topo trigger: arXiv 2105.01416 (multilevel trigger / prescaling)
- HTS screening cascades / orthogonal confirmatory assays (hit-triage literature) · START / Manchester clinical triage
- Perceptron branch predictor: Jiménez & Lin, HPCA 2001

**Pass-2 (2026-07-06) sources — the 12 new ideas:**
- Coding theory: Luby, *LT Codes*, FOCS 2002 · MacKay, *Fountain codes*, IEE Proc. Comms. 2005 · rateless-sampling
  reliability operator: arXiv 2605.09121 · Dietterich & Bakiri, *ECOC for multiclass* (JAIR 1995, arXiv cs/9501101) ·
  IP-based robust ECOC: arXiv 2011.00144
- Ensemble diversity: Wood et al., *A Unified Theory of Diversity in Ensemble Learning*, JMLR 24(359) 2023 · Krogh &
  Vedelsby ambiguity decomposition (NIPS 1995) · Liu & Yao, *Negative Correlation Learning* 1999 · Page,
  diversity-prediction theorem 2007 · Hinton, *Products of Experts*, Neural Comp. 2002 · Genest & Zidek log opinion
  pools 1986 · Jacobs et al., mixtures of experts 1991
- Diverse decoding / QD: SemDiD: arXiv 2506.23601 · Stein self-repulsive dynamics: arXiv 2002.09070 · Arithmetic
  Sampling: arXiv 2210.15458 · MAP-Elites / illumination: arXiv 1504.04909 · QD-for-LLM: arXiv 2605.09781
- Crowd truth inference / mechanism design: Dawid & Skene, JRSS-C 1979 · pairwise-co-occurrence crowdsourcing:
  arXiv 1909.12325 · LLM-ensemble DS (WISE): arXiv 2512.02405 · Prelec, *Bayesian Truth Serum*, Science 2004 · Prelec,
  Seung & McCoy, *surprisingly popular*, Nature 2017
- Social choice: Kemeny-Young / Borda / Condorcet · setwise Kemeny: arXiv 2304.14980 · LLM rank-aggregation: arXiv 2406.11871
- Pure-exploration bandits: Audibert & Bubeck BAI (COLT 2010) · Karnin et al. *Sequential Halving* (ICML 2013) · Russo,
  *Simple Bayesian Algorithms for BAI*: arXiv 1602.08448 · combinatorial pure exploration: arXiv 1706.01081 ·
  BAI-with-LLM-judges: arXiv 2601.21471
- Allocation / rating: Uncertainty-Aware Budget Allocation: arXiv 2605.26849 · Plan-and-Budget: arXiv 2505.16122 ·
  allocation-probability test for Thompson: arXiv 2111.00137 · Russo & Van Roy, *Information-Directed Sampling* 2018 ·
  Herbrich et al., *TrueSkill*, NIPS 2007 · active ranking from pairwise comparisons: arXiv 1109.3701 · comparative
  LLM-judge uncertainty: arXiv 2505.15240

---

## F. BEAT-THE-32B / COMPLEMENTARITY-REALIZING FUSION LEVERS (pass 3 — attack a NEW axis: make the combined 7B+32B *strictly more accurate* than always-32B)  ★[BEAT-32B]

> **Why this section exists.** The refined goal is a cascade that is FASTER **and MORE ACCURATE** than always-Lingshu-32B.
> Best current method (FALC) *matches* the 32B (pooled 0.5711 vs 0.573) at −27% latency / −58% FLOPs but is not robustly
> more accurate. §A–E raise the *cheap-ensemble* ceiling or *match* the 32B cheaper; **none combines the 7B and 32B
> outputs to EXCEED the 32B.** Yet the project's own data says the headroom exists: **oracle-union(7B,32B) > 32B**;
> **7B beats 32B on MMMU by +0.167**; **error-φ ≈ 0.37–0.52** (the legs miss *different* questions). §F realizes that
> union. **The binding constraint is the recoverability wall** (per-sample "will 32B fix the 7B error?" AUROC ≈ 0.6 MCQ /
> 0.87 open-text). Every §F idea is chosen to *dodge* the wall: fuse/route on a **higher-AUROC** surrogate — an
> **observable slice**, a **calibrated cross-model confidence-advantage**, a **certified high-precision region**, or a
> **structural arbitration** — rather than on raw per-sample recoverability; and the sample-level beat-32B bet is
> concentrated on **open-text** (wall weakest). All "expected beat" figures are HYPOTHESES to test, not results.

### F1. Observable-slice specialization router (which expert owns which region)  ★[BEAT-32B] TOP-5-offline (#1)
- **Source:** Model fusion / MoE specialization. Shnitzer et al., *Fusing Models with Complementary Expertise* (FoE),
  arXiv 2310.01542 (ICLR 2024) + its Frugal-FoE extension; classic MoE region-specialization (Jacobs et al. 1991).
- **Mechanism:** different experts dominate different input regions; a router that sends each region to *its* best
  expert beats always-using-the-strongest, because the strong model is not uniformly best.
- **Map to us:** route by an **observable slice** — (dataset × answer-format × imaging-modality × question-type) — to
  the slice's *calibration-winner*, not always the 32B. 7B wins MMMU (+0.167) and may own finer sub-slices (a modality ×
  q-type cell) even inside datasets the 32B wins overall. **Crucially the slice is known a-priori from the prompt/image
  ⇒ NO recoverability wall** (that wall is *per-sample*; per-slice model-skill is high-AUROC and learnable). FALC already
  shows the with-MMMU macro beat (0.665 > 0.6541) with one coarse slice; F1 makes it a principled, guardrail-clean,
  finer-grained router.
- **Concrete variation:** on calibration, for every slice compute acc_7B vs acc_32B (with a significance/guardrail
  constraint: only assign the 7B where its slice-CI lower-bound ≥ 32B); apply held-out; report macro/pooled acc vs
  always-32B and the per-slice guardrail.
- **Expected ACC:** beats always-32B on macro wherever ≥1 slice is 7B-owned. **Latency/compute:** 7B-owned slices are
  *free* (cheap leg only) ⇒ both-axes.
- **How to test:** **OFFLINE** — per-slice argmax over existing per-benchmark dumps; the only choice is slice
  granularity + the guardrail rule.
- **Novelty & risk:** medium — the framing is standard (FoE/MoE) but the *observable-slice, wall-dodging* argument for
  medical VQA is the contribution. **Risk (honest):** the robust beat may ride mostly on the MMMU anomaly (n=150); the
  test is whether *non-MMMU* finer slices yield any certified 7B-owned cell. If not, F1 degrades to iso-32B (safe).

### F2. Chair–Varshney optimal two-detector decision fusion  ★[BEAT-32B] TOP-5-offline (#4)
- **Source:** Distributed detection / sensor fusion. Chair & Varshney, *Optimal Data Fusion in Multiple Sensor Detection
  Systems* (IEEE T-AES, 1986) — with local decisions + their reliabilities, the **provably-optimal** fuser is a
  likelihood-ratio test over the decision tuple, and it beats either local detector.
- **Mechanism:** weight each detector's vote by log[P(correct)/P(error)] (its reliability), sum the weighted votes,
  threshold — the Bayes-optimal fusion under conditional independence.
- **Map to us:** treat (7B-decision, 32B-decision) as two local detectors. From calibration build each leg's per-slice
  reliability P(correct | its answer); the C-V rule assigns each candidate answer a log-reliability-weighted score and
  picks the max. Because φ ≈ 0.37 (partly independent errors), the fused decision provably ≥ the 32B alone.
- **Concrete variation:** OFFLINE, estimate per-slice reliabilities on a calibration split; apply the C-V weighted-vote
  fuser over the 7B+32B (and optionally +7B-think) hard decisions; compare fused acc vs always-32B and vs majority.
- **Expected ACC:** provable small-but-positive beat where the two legs carry independent evidence. **Latency/compute:**
  runs on decisions already produced (no extra inference beyond the escalated 32B pass).
- **How to test:** **OFFLINE** (reliability tables + LRT on the agreement tuples).
- **Novelty & risk:** medium-high — a 40-yr-old provably-optimal fuser, unused in LLM cascades, applied to 7B/32B. **Risk:**
  C-V assumes conditional independence; our legs are correlated ⇒ use the correlated-noise C-V variant or per-slice
  reliabilities to avoid over-crediting; only 2 detectors bounds the gain.

### F3. Cross-model calibrated confidence-advantage arbitration  ★[BEAT-32B] TOP-5-offline (#3)
- **Source:** Confidence-advantage routing + sensor fusion. RACER *risk-aware calibrated routing* (arXiv 2603.06616);
  *Rational Tuning of LLM Cascades via Probabilistic Modeling* (arXiv 2501.09345, joint distribution of *calibrated*
  confidences); *Cross-Model Disagreement as a Label-Free Correctness Signal* (arXiv 2603.25450); meta-analysis
  inverse-variance weighting (classical).
- **Mechanism:** after per-model temperature/isotonic calibration, confidences become *comparable across models*; the
  **confidence advantage** (cal-conf of one model minus the other) is a valid selector — pick whichever model is more
  confident, or inverse-variance-combine, instead of always the bigger one.
- **Map to us:** on the *disagreement* set, take the answer of the leg with higher **calibrated** P(correct). This
  realizes exactly the 7B-confident-right ∩ 32B-wrong region (the mechanism behind the MMMU win) while ceding the
  32B-confident-right ∩ 7B-wrong region — a **different, higher-signal quantity than per-sample recoverability** (which
  is the un-learnable wall). Distinct from the deployed *margin gate* (which only decides escalate/not, never *overrides*
  the 32B with the 7B).
- **Concrete variation:** OFFLINE, fit per-leg calibrators on a split; on held-out disagreements select argmax
  cal-conf (sweep a dead-band so the 32B wins ties); report net flips (7B-rescues − 7B-regressions) and acc vs always-32B.
- **Expected ACC:** beats the 32B by the net-positive mass of the advantage region. **Latency/compute:** free (uses
  logprobs already produced); can even *reduce* escalation (keep confident-7B).
- **How to test:** **OFFLINE** (opt_logprobs on both legs already dumped).
- **Novelty & risk:** medium-high — calibrated cross-model *override* (not just gating) for medical VQA. **Risk:** the
  7B is often confidently-wrong ⇒ the net flip can be ≈0 or negative on MCQ; calibration quality is load-bearing; likely
  strongest on open-text/reasoning slices — measure net-flip per slice.

### F4. Product-of-Experts / log-opinion-pool fusion of the 7B AND 32B posteriors  ★[BEAT-32B]
- **Source:** Probabilistic ML. Hinton, *Products of Experts* (Neural Comp. 2002); Genest & Zidek, logarithmic opinion
  pools (1986). (Distinct from **B9**, which log-pools *cheap generators only*; F4 fuses **cheap + strong**.)
- **Mechanism:** multiply the two *calibrated* option-distributions (reliability-weighted log-pool) — sharp where the
  experts agree, and a confident, well-calibrated minority can overturn the other; beats the stronger member when errors
  are near-conditionally-independent.
- **Map to us:** fuse the 7B and 32B per-option logprobs with per-slice temperatures/weights (from held-out accuracy);
  argmax the fused distribution. Where the 7B independently "knows" the answer the 32B misses, the product flips it.
- **Concrete variation:** OFFLINE, log-pool opt_logprobs with learned (temp, weight) per leg/slice; compare fused-argmax
  acc vs always-32B, vs C-V (F2), vs additive BMA (F11) on MCQ dumps.
- **Expected ACC:** small beat on MCQ near-ties. **Latency/compute:** +cheap fusion only.
- **How to test:** **OFFLINE** (MCQ opt_logprobs).
- **Novelty & risk:** medium — a principled soft cheap+strong fuser the project hasn't run. **Risk:** PoE over-sharpens
  on correlated experts (needs reliability weights/temperature); MCQ-only unless open-text gets an option-scoring proxy.

### F5. Medical double-reading with disagreement-triggered arbitration  ★[BEAT-32B] TOP-5-offline (#2)
- **Source:** Radiology practice. Independent **double reading + arbitration** in screening mammography — two readers,
  a third arbitrates disagreements; raises sensitivity ~10–15% at ~flat specificity (CO-OPS trial, *Radiology* 2018;
  AI-as-second-reader, *Radiology*/*Sci Rep* 2024).
- **Mechanism:** two independent readers of *different* expertise catch *different* misses; agreement is high-precision,
  and arbitration of the disagreements is where the joint accuracy exceeds either reader alone.
- **Map to us:** run 7B-nt and 32B-nt as two independent readers. **Agree ⇒ accept the shared answer** (cheap, and
  agreement is very high-precision). **Disagree ⇒ invoke an ARBITER** — 32B-think, or a trained tie-break arbiter, or
  the outcome verifier — only on the small disagreement set. Beats always-32B-nt iff the arbiter, on disagreements,
  out-scores the plain 32B-nt (e.g. by sometimes siding with a correct 7B). This is the clinically-legible framing of the
  whole method.
- **Concrete variation:** OFFLINE, partition by 7B/32B-nt agree/disagree; on disagreements simulate each candidate
  arbiter from existing dumps (32B-**think**, verifier-pick, calibrated-conf F3); report pooled acc vs always-32B-nt and
  the escalation cost = disagreement rate × arbiter cost.
- **Expected ACC:** beat = (arbiter acc − 32B-nt acc) on the disagreement mass. **Latency/compute:** agreement mass is
  free; cost bounded by disagreement rate (measure it — likely 15–35%).
- **How to test:** **OFFLINE** — 7B, 32B-nt, and 32B-think dumps all exist.
- **Novelty & risk:** medium-high — double-reading→VLM-cascade mapping is a clean, publishable medical framing. **Risk
  (honest):** 32B-think ≤ 32B-nt on *perception* ⇒ think is a poor arbiter there; the arbiter must be a *trained*
  tie-breaker or restricted to reasoning slices. The beat lives or dies on arbiter quality on the disagreement set.

### F6. Contrastive-decoding cascade — the 7B as an amateur "negative expert"  ★[BEAT-32B]
- **Source:** Contrastive decoding. O'Brien & Lewis, *Contrastive Decoding Improves Reasoning* (arXiv 2309.09117);
  Li et al., *Contrastive Decoding* (arXiv 2210.15097); Liu et al., *DExperts* (arXiv 2105.03023).
- **Mechanism:** score = log p_expert − α·log p_amateur; subtracting the weaker model cancels the *shared, easy,
  surface-level* modes both models fall for and amplifies the expert's genuine edge — often beating the expert alone.
- **Map to us:** on escalated MCQ, re-rank the 32B's options by **(log p_32B − α·log p_7B)** instead of raw 32B logprob.
  Same-family, same-tokenizer (both Qwen2.5-VL-based) makes the logits directly subtractable. The 7B is the amateur; the
  subtraction removes the confident-shortcut errors the 32B inherits from the shared pretraining/family, netting above
  the 32B. A genuinely *different* combination (subtract, not average/vote).
- **Concrete variation:** OFFLINE, over paired 7B+32B opt_logprobs, sweep α (+ an amateur floor/plausibility mask as in
  CD); compare contrastive-argmax acc vs always-32B-argmax; also try the reverse (α<0) as a control.
- **Expected ACC:** beat where the 32B's errors are shared-with-7B surface modes. **Latency/compute:** free re-rank on
  the escalated set (both legs already run when escalating).
- **How to test:** **OFFLINE** (MCQ opt_logprobs; only escalated items need both legs).
- **Novelty & risk:** high — contrastive decoding as the *cascade combination rule* is new here. **Risk:** CD can over-
  penalize genuinely-easy-correct options (needs the plausibility mask); short single-token MCQ answers give CD little to
  work with; α is dataset-sensitive (fit per slice).

### F7. Frugal stacking / super-learner meta-combiner on calibrated features  ★[BEAT-32B]
- **Source:** Wolpert, *Stacked Generalization* (1992); van der Laan, Polley & Hubbard, *Super Learner* (2007) — the
  **oracle inequality**: the CV-weighted stack is asymptotically ≥ the best base learner.
- **Mechanism:** a low-capacity meta-learner over base predictions, weighted by out-of-fold performance, provably at
  least matches (and usually beats) the best single base.
- **Map to us:** a **frugal** stacker (logistic / shallow GBM) whose inputs are *calibrated, low-dimensional* features —
  {7B answer, 32B answer, both calibrated confidences, agreement flag, slice-id, margin} — outputs the final answer.
  **This is explicitly the fix for the failed CALM-Fuse:** hidden-state fusion overfit and *did not transfer*; the
  super-learner theory says a *frugal* combiner on calibrated features + strict CV transfers where high-capacity fusion
  did not. Composes F1/F2/F3 into one trained rule.
- **Concrete variation:** OFFLINE, 5-fold CV stack over the calibrated feature vector; report held-out acc vs always-32B
  and vs each single leg; ablate capacity (logistic → GBM → MLP) to show the *frugal* end transfers and the rich end
  overfits (the CALM-Fuse post-mortem).
- **Expected ACC:** provably ≥ best base; goal is a *significant* beat, not just ≥. **Latency/compute:** meta-model is ns.
- **How to test:** **OFFLINE** (all features exist on dumps).
- **Novelty & risk:** medium — stacking is standard, but the *frugal-transfers/rich-overfits* result on this cascade
  (given CALM-Fuse failed) is the contribution. **Risk:** even the frugal stack may only tie the 32B if the underlying
  advantage mass is tiny (recoverability wall); needs enough calibration data per slice.

### F8. Weak-model high-precision veto ("certified trust-region override")  ★[BEAT-32B] TOP-5-offline (#5)
- **Source:** Selective prediction / conformal. Chow's optimal-rejection rule (1970); split-conformal high-precision
  selection; weak-to-strong "when to trust the small model" (cf. Co-LLM defer, ACL 2024 / arXiv 2403.03870).
- **Mechanism:** override the strong model **only** inside a calibration-certified region where the weak model's
  precision provably exceeds the strong model's accuracy — a one-sided, guardrail-safe complementarity grab.
- **Map to us:** keep the 7B (veto the 32B) iff 7B-cal-conf > θ_high AND the 7B's *certified* precision in that
  (conf, slice) cell ≥ 32B accuracy there (split-conformal lower bound). Everywhere else, standard cascade. By
  construction the override only *adds* correct flips on calibration; the only risk is held-out slippage, bounded by the
  conformal guarantee. The safest, most deployable route to "beat-32B."
- **Concrete variation:** OFFLINE, mine (conf-bin × slice) cells on calibration for a conformal precision-≥-32B
  certificate; apply the veto held-out; report net flips + a distribution-free risk bound + escalation saved.
- **Expected ACC:** monotone-safe small beat (never below 32B by more than the conformal α). **Latency/compute:** vetoed
  items are 7B-only ⇒ *cheaper* than always-32B (both-axes).
- **How to test:** **OFFLINE** (calibration mining + coverage check on dumps).
- **Novelty & risk:** medium-high — a certified, one-sided weak-over-strong override with a safety guarantee is a strong
  "safe medical deployment" story. **Risk:** high-precision cells may be tiny/empty off-MMMU (beat → ~0 but never
  negative); exchangeability breaks OOD (RadImageNet) — quantifying that break is itself a finding.

### F9. Disagreement-set multi-agent debate / cross-examination  ★[BEAT-32B]
- **Source:** Du et al., *Improving Factuality and Reasoning via Multiagent Debate* (arXiv 2305.14325); Irving et al.,
  *AI safety via debate* (2018); *Inter-Cascade* interactive LLM cascade (arXiv 2509.22984, +6.35% overall acc);
  *Best-of-∞* complementary-error ensembling (arXiv 2509.21091).
- **Mechanism:** agents propose then critique/revise over rounds; structured disagreement surfaces and corrects
  errors, beating single agents.
- **Map to us:** only on the **small 7B/32B disagreement set**, run one reconsideration round — show the 32B the 7B's
  answer + rationale ("a colleague argues X because…; reconsider") and let it revise (optionally symmetric). Beats
  always-32B by flipping the 32B errors the 7B can *correctly* challenge, at cost bounded by the disagreement rate.
- **Concrete variation:** GPU — one 32B reconsideration pass conditioned on the 7B's answer over disagreements; sweep
  accept/revise; report acc vs always-32B and cost = disagreement-rate × 1 extra 32B forward. **OFFLINE proxy first:**
  does 32B-**think** already flip 32B-nt errors on the disagreement set? (a free upper-bound sanity check).
- **Expected ACC:** beat = corrected-32B-errors − newly-introduced-errors on disagreements. **Latency/compute:** +1 short
  32B pass on disagreements only.
- **How to test:** **GPU** (reconsideration pass); OFFLINE proxy via 32B-think flips.
- **Novelty & risk:** medium-high — disagreement-gated single-round debate as the *arbiter* of a cascade (vs full
  multi-round debate on everything) is a cost-aware novelty. **Risk:** debate can talk the 32B *out* of correct answers
  (sycophancy/anchoring to the 7B); needs a revise-only-if-more-confident guard; GPU-gated.

### F10. Learning-to-complement deferral, concentrated on open-text (where recoverability AUROC ≈ 0.87)  ★[BEAT-32B]
- **Source:** Learning-to-defer / complement. Mozannar & Sontag, *Consistent Estimators for Learning to Defer* (arXiv
  2006.01862, ICML 2020, the consistent multiclass surrogate); Mozannar et al., *Sample-Efficient Learning of Predictors
  that Complement Humans* (arXiv 2207.09584); Wilder, Horvitz & Kamar, *Learning to Complement Humans* (IJCAI 2020);
  *human-AI complementarity tight bounds* (arXiv 2605.08710).
- **Mechanism:** jointly train a *predictor + rejector* with a **consistent surrogate** so the *team* objective (not the
  strong model's) is minimized — provably realizes complementarity when *any* routing signal exists, unlike a threshold
  on a fixed confidence.
- **Map to us:** the recoverability wall is the reason per-sample MCQ routing can't beat the 32B — but **on open-text the
  wall is weak (AUROC ≈ 0.87)**, so a consistent L2D combiner over {7B-bo-N + verifier} vs {32B} can make the *team*
  exceed both members. Target the beat-32B claim exactly where the signal supports it (open-text), and let MCQ ride the
  safe §F1/F8 slice/veto levers.
- **Concrete variation:** OFFLINE, train a Mozannar-Sontag-style rejector on open-text dumps (features = 7B verifier
  conf, cross-model agreement, seqlogprob); score the joint 7B-or-32B team acc vs always-32B and vs the fixed-τ gate,
  held-out.
- **Expected ACC:** beat on open-text (the tractable-signal half); ≥ on MCQ. **Latency/compute:** defers only the
  hard fraction to the 32B (both-axes).
- **How to test:** **OFFLINE** (open-text per-sample dumps + judge labels).
- **Novelty & risk:** medium-high — a *consistent* L2D combiner (vs the project's threshold gates) that explicitly
  exploits the open-text-vs-MCQ recoverability split. **Risk:** L2D surrogates need enough deferral labels; the win is
  bounded by the open-text advantage mass; must not double-count best-of-N FLOPs (the §2.6 FLOPs caveat).

### F11. Calibrated confidence-weighted posterior averaging (BMA / ensemble-MOS) with per-slice skill  ★[BEAT-32B]
- **Source:** Ensemble weather forecasting. Raftery et al., *Using Bayesian Model Averaging to Calibrate Forecast
  Ensembles* (Monthly Weather Review, 2005) — the BMA mixture beats the best single member (~6% RMSE) by weighting
  calibrated members by skill; ensemble-MOS post-processing.
- **Mechanism:** average the members' *calibrated* predictive distributions weighted by training-period skill; the
  additive mixture reduces variance and de-biases, beating the best member — and (unlike PoE, F4) degrades gracefully if
  one member is bad.
- **Map to us:** the **additive** sibling of F2/F4 — mix the 7B and 32B calibrated option-posteriors with per-slice BMA
  weights (EM-fit on calibration), argmax. Robust where PoE would over-sharpen on correlated errors; a natural "combine
  a cheap-noisy and an expensive-accurate source to beat the expensive one" transfer from operational forecasting.
- **Concrete variation:** OFFLINE, EM-fit per-slice BMA weights over 7B/32B (optionally +7B-think) calibrated opt_logprobs;
  compare mixture-argmax acc vs always-32B, vs PoE (F4), vs C-V (F2).
- **Expected ACC:** beat where the 7B carries non-trivial slice skill; safer than F4 under correlation. **Latency/compute:**
  +cheap EM + fusion only.
- **How to test:** **OFFLINE** (MCQ opt_logprobs).
- **Novelty & risk:** medium — BMA is classical, but the cheap+strong medical-VQA transfer + the PoE-vs-BMA (multiplicative
  vs additive) contrast is the contribution. **Risk:** additive averaging can *dilute* a correct-but-outvoted 32B; skill
  weights need enough per-slice calibration data.

---

## F.RANK — beat-32B axis prioritization (pass 3)

**The §A–E table above still governs the *efficiency/match-the-32B* goal.** For the **refined beat-32B goal**, rank §F by
(beat-32B plausibility × novelty × testability × wall-dodging), and flag OFFLINE-testable:

| # | idea | how it dodges the recoverability wall | beat-32B route | test | rank |
|---|---|---|---|---|---|
| F1 | Observable-slice specialization router | routes on **observable slice** (high-AUROC), not per-sample recoverability | 7B-owned slices (MMMU + finer cells) | **OFFLINE** | **1** |
| F5 | Double-reading + disagreement arbitration | only the **disagreement set** is arbitrated (structural, not predicted) | arbiter > 32B on disagreements | **OFFLINE** | **2** |
| F3 | Calibrated confidence-advantage arbitration | uses **cross-model cal-conf advantage** (≠ recoverability signal) | 7B-confident-right ∩ 32B-wrong | **OFFLINE** | **3** |
| F2 | Chair–Varshney optimal decision fusion | provably-optimal fuser on **reliabilities**, not per-sample prediction | independent-evidence mass | **OFFLINE** | **4** |
| F8 | Certified high-precision weak-veto | **conformal certificate** per (conf×slice) cell — one-sided safe | certified 7B>32B cells | **OFFLINE** | **5** |
| F11 | Calibrated BMA posterior averaging | per-**slice** skill weights (observable) | additive cheap+strong mixture | OFFLINE | 6 |
| F4 | PoE / log-pool 7B+32B fusion | reliability-weighted (per-slice) | multiplicative cheap+strong | OFFLINE | 7 |
| F6 | Contrastive-decoding cascade (7B amateur) | subtracts **shared** modes (no recoverability needed) | expert-minus-amateur re-rank | OFFLINE | 8 |
| F7 | Frugal super-learner stack | calibrated low-dim features + CV (fixes CALM-Fuse) | provable ≥ best base | OFFLINE | 9 |
| F10 | Learning-to-complement (open-text) | targets **open-text** (wall weak, AUROC 0.87) | consistent L2D team | OFFLINE | 10 |
| F9 | Disagreement-set multi-agent debate | arbitrates only **disagreements** (structural) | 32B revises vs 7B challenge | GPU (+OFFLINE proxy) | 11 |

**★ TOP 5 OFFLINE-TESTABLE (beat-32B), test first:** **F1** (slice specialization — already part-evidenced by FALC's
with-MMMU macro 0.665 > 0.6541; test finer non-MMMU cells), **F5** (double-reading arbitration — 7B/32B-nt/32B-think
dumps already exist), **F3** (calibrated confidence-advantage override on opt_logprobs), **F2** (Chair–Varshney provably-
optimal decision fusion), **F8** (conformal high-precision weak-veto — safest/deployable). All five run on the
**existing per-sample dumps with zero new inference**; each attacks the beat-32B axis by a *different* wall-dodging
surrogate (slice / structure / cal-confidence / reliability / certificate), so they are complementary, not redundant.
**Honest meta-caveat:** the recoverability wall means the *realizable* beat-32B margin is small on MCQ (bounded by the
7B-right ∩ 32B-wrong mass); the biggest, most-honest headroom is **open-text (F10) + the MMMU-type observable slices
(F1)**. If §F yields only iso-32B on MCQ, that itself sharpens the paper's "match-cheaply on MCQ, beat on open-text +
owned-slices" story.

**SOURCES (pass 3 — §F):**
- Model fusion / MoE: Shnitzer et al., *Fusing Models with Complementary Expertise (FoE)* + Frugal-FoE, arXiv 2310.01542
  (ICLR 2024) · Jacobs et al., adaptive mixtures of local experts, 1991.
- Sensor / decision fusion: Chair & Varshney, *Optimal Data Fusion in Multiple Sensor Detection Systems*, IEEE T-AES 1986 ·
  Hinton, *Products of Experts*, Neural Comp. 2002 · Genest & Zidek, log opinion pools, 1986.
- Confidence-advantage routing: RACER, arXiv 2603.06616 · *Rational Tuning of LLM Cascades via Probabilistic Modeling*,
  arXiv 2501.09345 · *Cross-Model Disagreement as a Label-Free Correctness Signal*, arXiv 2603.25450 · Co-LLM /
  *Learning to Decode Collaboratively*, ACL 2024 (arXiv 2403.03870) · CITER token-routing, arXiv 2502.01976.
- Contrastive decoding: O'Brien & Lewis, *Contrastive Decoding Improves Reasoning*, arXiv 2309.09117 · Li et al.,
  *Contrastive Decoding*, arXiv 2210.15097 · Liu et al., *DExperts*, arXiv 2105.03023.
- Ensembles / stacking: Wolpert, *Stacked Generalization*, 1992 · van der Laan, Polley & Hubbard, *Super Learner*, 2007 ·
  Raftery et al., *BMA to Calibrate Forecast Ensembles*, Monthly Weather Review 2005.
- Learning-to-defer / complement: Mozannar & Sontag, *Consistent Estimators for Learning to Defer*, arXiv 2006.01862
  (ICML 2020) · Mozannar et al., *Predictors that Complement Humans*, arXiv 2207.09584 · Wilder, Horvitz & Kamar,
  *Learning to Complement Humans*, IJCAI 2020 · *Human-AI complementarity tight bounds*, arXiv 2605.08710.
- Debate / interactive cascade: Du et al., *Improving Factuality and Reasoning via Multiagent Debate*, arXiv 2305.14325 ·
  Irving et al., *AI safety via debate*, 2018 · *Inter-Cascade*, arXiv 2509.22984 · *Best-of-∞*, arXiv 2509.21091.
- Medical double-reading + arbitration: CO-OPS trial, *Radiology* 2018 · AI-as-second-reader, *Radiology* / *Sci Rep* 2024
  (double reading + arbitration raises sensitivity ~10–15% at ~flat specificity).
- Selective prediction / conformal veto: Chow, optimal rejection rule, 1970 · split-conformal high-precision selection.

---

## G. ESCALATION-COST & ESCALATION-RATE LEVERS — make the 7B→32B jump CHEAPER and RARER  ★ESC

> **Pass-3b (2026-07-07). A distinct axis from §A–F.** §A–E chase the two *accuracy walls*; §F (concurrent) chases
> *beating* the 32B. §G chases the **speed** the deployed cascade (FALC) actually rides on. FALC is faster than
> always-32B **only where escalation is rare**: pooled it wins (−27% latency / −58% FLOPs at ~20% escalation), but that
> win *collapses* on every slice where the 7B is weak and escalates heavily — MedXpert (90% esc), VQA-RAD-open (91%),
> all open-text (53–91%) — because **one 32B forward = 4.57× a 7B forward / ~665 ms**, and the latency crossover sits at
> ~48% MCQ / ~22% open escalation (`best_method_lingshu_medeval.json`). Two levers move this: **(i) make each
> escalation cheaper** — shrink the strong forward (G1–G4, G8, G9); **(ii) make escalation rarer** — fewer, more
> precise escalations at equal accuracy (G5–G7, G10). 32B-**think** is dropped throughout (16× slower, never beats
> no-think — measured). Cost anchors (real, from `best_method_lingshu_medeval.json` / `latency_reexamination.json`):
> **GEN7 347 ms / 1.0 FLOP-eq; VER7 175 ms; GEN32-nothink 665 ms / 4.57 FLOP-eq**. Every "expected effect" is a
> HYPOTHESIS to test, not a result. Tag: **★ESC**.

### G1. Token-level speculative decoding of the escalation forward (7B drafts, 32B verifies)  ★ESC-cheaper
- **Source:** speculative decoding — Leviathan et al. (arXiv 2211.17192), Chen et al. (2302.01318); speculative
  *cascades* Narasimhan et al. (2405.19261, = D1's cite) + Google "Speculative Cascades" (research.google blog, 2025).
- **Mechanism:** the small model drafts a block of tokens; the large model VERIFIES the whole block in ONE parallel
  forward and accepts the longest matching prefix — **lossless** large-model output at fewer large-model *steps*.
- **Map to us:** when escalation fires, don't let the 32B autoregress from scratch — have the **7B draft the answer and
  the 32B verify it**, so the 32B pays ~one prefill + a few verify steps, not a full serial decode. **Distinct from D1**
  (answer-level P(Yes) verify, *lossy* — may keep a wrong 7B answer): G1 reproduces the 32B answer *exactly*.
- **Concrete variation:** measure the 7B→32B token acceptance rate; model escalation latency = prefill +
  ⌈L/accept⌉·verify-step; compare to full 32B decode at the measured answer-length distribution per benchmark.
- **Expected ACC:** identical to always-32B (lossless). **Latency/compute:** cuts the *decode* part of the 665 ms — but
  medical answers are SHORT, so the win is bounded on MCQ/closed (prefill-bound) and largest on long open-text/reasoning.
- **How to test:** OFFLINE latency model IF per-token logprobs are dumped (acceptance ≈ 7B/32B top-token agreement);
  else a small GPU spec-decode pass. **Novelty & risk:** medium — spec-decoding the *escalation leg* specifically; risk:
  short answers ⇒ prefill-bound, so the decode-only saving is small unless paired with G4/G8/G9.

### G2. Depth-adaptive / early-exit strong leg (CALM / LayerSkip / Mixture-of-Depths)  ★ESC-cheaper
- **Source:** CALM (Schuster et al., 2207.07061, NeurIPS 2022, up to ×3, with a statistical guarantee); LayerSkip
  self-speculative (2404.16710); Mixture-of-Depths (Raposo et al., 2404.02258); early-exit survey (2509.05915).
- **Mechanism:** exit the transformer at an intermediate layer (or route tokens through a subset of layers) once
  per-token confidence is high — spend full depth only on the hard tokens.
- **Map to us:** run the 32B escalation forward with a confidence-based early-exit head; many short medical answers
  ("yes"/"left"/a letter) resolve in the first ⅓–½ of layers, so the *average* escalation costs a fraction of the full
  forward. LayerSkip additionally makes the 32B **self-speculative** (draft from early layers, verify with the rest) — a
  within-model version of G1 needing no 7B draft.
- **Concrete variation:** probe the 32B's intermediate-layer top-1 agreement with its final answer per dataset to bound
  the exit layer; re-cost escalation FLOPs/latency at the exit-layer distribution.
- **Expected ACC:** ≈ always-32B with a calibrated exit threshold (CALM gives a distortion guarantee). **Latency/compute:**
  cheaper escalation FLOPs+latency (both-axes on the strong leg).
- **How to test:** GPU (needs layer-wise logits / an exit head); a partial OFFLINE estimate from a one-off
  intermediate-layer probe. **Novelty & risk:** medium-high — early-exit on a *medical-VLM escalation tier*; risk: exit
  heads may need light training; the vision prefill is NOT shortened by this (pair with G4).

### G3. Quantized strong leg as the escalation target (AWQ / GPTQ INT4)  ★ESC-cheaper — OFFLINE re-cost
- **Source:** model compression — AWQ (Lin et al., 2306.00978, MLSys 2024 best paper, ~3.7× on 4090, VLM support
  VILA/LLaVA); GPTQ (Frantar et al., 2210.17323, 3.2–4.5×); quantized-LLM eval survey (2409.11055).
- **Mechanism:** 4-bit weight-only post-training quantization shrinks weights ~4× and speeds decode ~3–4× at typically
  <1% task-accuracy loss.
- **Map to us:** make the escalation target a **quantized 32B (AWQ INT4)**. The escalation forward drops from
  **4.57× → ~1.3–1.8×** a 7B forward and ~3× lower latency at ~0 accuracy loss, directly attacking the 665 ms / 4.57×
  that erodes the speed win. The latency crossover escalation-rate roughly *triples*, flipping MedXpert / VQA-RAD-open /
  open-text from "slower than always-32B" to "faster". It also unblocks the OmniMed 32B leg (the data-gap note itself
  lists quantization as a fix).
- **Concrete variation:** OFFLINE — recompute every per-benchmark cascade cost in `best_method_lingshu_medeval.json`
  with GEN32_q ≈ 4.57/3 FLOP-eq and ~665/3 ms; report the new pooled + per-benchmark faster/slower verdicts.
- **Expected ACC:** ≈ always-32B (published AWQ VLM drop <1%; **confirm on our suite, don't assume**). **Latency/compute:**
  the single biggest off-the-shelf per-escalation cost cut.
- **How to test:** OFFLINE cost re-simulation NOW; a GPU load of an AWQ-Lingshu-32B to CONFIRM the accuracy delta.
  **Novelty & risk:** low-mechanism / high-leverage — engineering, not novel, but it may lift the *entire* deployable
  frontier and is the cleanest "cheaper escalation" win. Risk: quant accuracy on hard medical reasoning must be *measured*.

### G4. Question-guided image-token pruning on the escalation forward (7B-attention-guided)  ★ESC-cheaper
- **Source:** visual-token pruning — FastV (Chen et al., 2403.06764), SparseVLM, QG-VTC "Question-Guided Visual Token
  Compression" (2504.00654), PyramidDrop, VisionZip.
- **Mechanism:** rank visual tokens by (text-conditioned) attention and drop the low-salience majority after a shallow
  layer — big VLM FLOPs cut at minimal VQA accuracy loss, because prefill over hundreds–thousands of image tokens
  dominates short-answer cost.
- **Map to us:** for short medical answers the 32B escalation cost is **prefill-bound** (image tokens ≫ answer tokens),
  so pruning the 32B's visual prefill is the highest-leverage cheaper-escalation lever. Reuse the **7B's cross-attention**
  (already computed on the cheap leg) to pick which image tokens the 32B keeps — a cascade-native, training-free
  relevance signal. This **revives the project's original killed direction** (visual-token pruning as a standalone
  accuracy lever) in a new role: shrinking only the *strong escalation forward*.
- **Concrete variation:** prune the 32B image tokens to k% using 7B-attention scores; sweep k; measure escalated-subset
  accuracy vs FLOPs; the token-count→FLOPs map already exists (`token_cache.json`).
- **Expected ACC:** ≈ always-32B at moderate pruning (VQA-tolerant per FastV/QG-VTC). **Latency/compute:** cuts the
  dominant prefill term of the 665 ms.
- **How to test:** GPU (prune + measure) + OFFLINE FLOPs re-cost. **Novelty & risk:** medium — 7B-attention-guided
  pruning of the *escalation* forward is new here; risk: medical fine detail (small lesions) may sit in low-salience
  tokens ⇒ per-benchmark guardrail check required.

### G5. Recoverability-aware escalation suppressor — keep-cheap on the predicted-futile set  ★ESC-rarer — OFFLINE
- **Source:** learning-to-defer / early abstention — "Cost-Saving LLM Cascades with Early Abstention" (2502.09054),
  Jitkrittum recoverability (2307.02764); clinical "futile-care"/low-yield-test triage.
- **Mechanism:** don't pay the expensive stage where it *won't* change the outcome — escalate only the cheap errors the
  strong model is likely to FIX; keep-cheap on the both-fail set (never abstain).
- **Map to us:** the speed win is erased exactly on the slices where escalation is **futile** — MedXpert (7B 0.26, 90%
  esc, but 32B also near-floor 0.31 ⇒ almost no escalation recovers) and the hardest open-text. Per-*item*
  recoverability is unlearnable (AUROC ~0.6, the wall), but per-*slice* recoverability is strong: suppress escalation
  where expected recovery ΔAcc ≤ λ·(escalation cost). Converts "escalate 90%, barely gain" into "escalate the
  recoverable minority only" — a large latency/FLOPs cut at ~0 accuracy loss on the futile slices.
- **Concrete variation:** OFFLINE — from 7B/32B per-sample correctness, estimate per-slice (dataset × margin-bin ×
  modality) recovery rate; escalate only when recovery·headroom > cost; report escalation-rate + Δacc vs the flat gate.
  On MedXpert this should cut esc 90% → the recoverable fraction for ≤ a fraction-of-a-point accuracy trade.
- **Expected ACC:** ≈ cascade (drops only escalations that were not recovering). **Latency/compute:** RARER escalation
  precisely where the cascade is currently *slower* than always-32B.
- **How to test:** OFFLINE on existing dumps. **Novelty & risk:** medium-high — the **inverse** of E2's force-escalate
  red-flags (a "don't-escalate" red-flag), aimed squarely at the named failure mode. Risk: slice recovery estimates
  need support; guardrail must ensure no benchmark dips below 7B.

### G6. Two-stage serial confirmatory gate (specificity-first escalation)  ★ESC-rarer — OFFLINE
- **Source:** diagnostic-testing theory — serial vs parallel testing (serial = confirm-before-treat: raises
  specificity, cuts false positives); acceptance sampling (Dodge–Romig double sampling).
- **Mechanism:** in SERIES, a positive must be confirmed by a second, more specific test before action — trades a
  little sensitivity for much higher specificity (far fewer false positives).
- **Map to us:** an escalation is a "positive." Today one signal (margin / verifier-conf) triggers the 32B. Require
  **two independent cheap signals to AGREE** before paying the 32B — e.g. margin-low AND 7B-self-verify-says-wrong (both
  already computed / cheap) — an AND-gate that raises escalation *specificity* → fewer wasted 32B calls at a controlled
  miss rate. **Distinct from C3** (SPRT likelihood accumulation over samples) and **D5** (a single orthogonal probe *as*
  the gate): G6 is a specificity-first *composition* of two existing gate signals.
- **Concrete variation:** OFFLINE — compose {margin, 7B self-verify P(True), seqlogprob} as an AND / calibrated 2-of-2
  gate; trace escalation-rate vs accuracy vs the single-signal gate on the bake-off dumps.
- **Expected ACC:** ≈ single-gate at fixed miss rate. **Latency/compute:** RARER escalation (fewer false-positive
  escalations) at +1 cheap verify (175 ms) ≪ a 32B call (665 ms).
- **How to test:** OFFLINE (the gate signals are in `gate_unified_bakeoff` dumps). **Novelty & risk:** medium —
  diagnostic serial-testing framing of gate composition; risk: the two signals are correlated (both confidence-derived)
  ⇒ specificity gain may be modest unless the second signal is genuinely orthogonal (pair with a G4-style resolution probe).

### G7. Semantic escalation cache / memoization (amortize escalation over a stream)  ★ESC-rarer — OFFLINE
- **Source:** systems — CPU cache / memoization; semantic LLM caching (GPTCache; "GPT Semantic Cache" 2411.05276,
  61–69% call reduction; SCALM); VLM encoder cache VLCache (2512.12977).
- **Mechanism:** reuse a stored answer for a semantically-equivalent query instead of recomputing — a cache hit never
  touches the model.
- **Map to us:** in a deployment STREAM (and within a benchmark) medical VQA has templated questions and repeated
  images. Cache 32B escalation answers keyed by (image-hash, question-embedding); a near-duplicate escalation hits the
  cache → the 32B fires **once per cluster, not per question**, lowering the *effective* escalation rate over the stream.
- **Concrete variation:** OFFLINE — cluster the eval stream by (image, question) similarity; measure near-duplicate rate
  and cache-hit accuracy (answer-reuse correctness) at several thresholds; report effective escalation-rate reduction.
- **Expected ACC:** ≈ cascade at a conservative threshold. **Latency/compute:** RARER *effective* escalation, amortized;
  near-zero added cost (an embedding + ANN lookup).
- **How to test:** OFFLINE. **Novelty & risk:** medium — caching the *escalation* tier specifically. Risk: **test-set
  leakage** if used to score a benchmark ⇒ frame strictly as a deployment-stream property, measure with WITHIN-stream
  dedup only (never reuse a test label); gains are distribution-dependent.

### G8. Speculative prefill prefetch for the escalation leg (precomputed / disaggregated prefill)  ★ESC-cheaper — OFFLINE
- **Source:** computer-architecture prefetching; disaggregated prefill/decode serving (Perplexity; PrefillShare
  2602.12029; FlowKV 2504.03775).
- **Mechanism:** start the expensive-but-deterministic work before it is known to be needed, so the critical path only
  pays the residual.
- **Map to us:** the 32B's **image prefill does not depend on the 7B output** (same image), so compute the 32B
  image-encode + prefill IN PARALLEL with the 7B pass; if escalation fires, the 32B pays only the (short) decode — the
  prefill latency is fully hidden. **Distinct from D2** (branch-predictor pre-launch of the *whole* 32B gated by a
  learned predictor): G8 unconditionally prefetches only the shareable, deterministic prefill (no predictor; the prefill
  is always valid if escalation fires), trading always-on prefill FLOPs for hidden latency — ideal when the 2nd GPU is idle.
- **Concrete variation:** OFFLINE latency model — split measured GEN32 (665 ms) into prefill vs decode; recost
  escalated-query wall-clock as max(7B, 32B-prefill) + 32B-decode; report latency saved vs FLOPs spent at several
  prefetch-fraction operating points (all vs gated-by-a-cheap-esc-predictor).
- **Expected ACC:** unchanged. **Latency/compute:** pure latency win on escalated queries; FLOPs cost = the prefetched
  prefills that didn't escalate.
- **How to test:** OFFLINE (latency re-cost) + GPU wall-clock to confirm. **Novelty & risk:** medium-high — latency-axis,
  zero-accuracy-risk; risk: wastes prefill FLOPs on non-escalations (bound it with a cheap escalation-likelihood
  predictor); value depends on the SLA / idle-GPU assumption.

### G9. Cross-tier vision-encoder / image-token reuse (encode the image once)  ★ESC-cheaper
- **Source:** VLM caching — VLCache "compute 2% vision tokens, reuse 98%" (2512.12977); LMCache multimodal (vLLM V1);
  cross-turn image-KV reuse.
- **Mechanism:** a vision encoder's output for an image can be cached and reused, bypassing costly re-encoding on later
  passes over the same image.
- **Map to us:** Lingshu-7B and -32B are BOTH Qwen2.5-VL-based, and Qwen2.5-VL keeps the SAME ~675M vision encoder
  across LLM sizes — **IF the two tiers share the ViT** (must verify), the image tokens computed on the cheap 7B leg can
  seed the 32B prefill, so escalation skips vision re-encoding entirely; even if only the ViT (not the projector) is
  shared, the patch features are reusable.
- **Concrete variation:** **VERIFY FIRST** (arch/config diff) that Lingshu-7B and -32B share the vision tower — this is
  an ASSUMPTION, flagged; if shared, cache 7B ViT features and feed the 32B prefill; measure escalation latency/FLOPs saved.
- **Expected ACC:** unchanged if the ViT is truly shared (bit-identical features); needs a numeric check otherwise.
  **Latency/compute:** removes the vision-encode cost from every escalation.
- **How to test:** OFFLINE arch/config diff to confirm sharing; GPU to wire feature reuse + measure. **Novelty & risk:**
  medium — cross-*tier* (not cross-turn) vision reuse; risk: Lingshu may have fine-tuned the ViT differently per size ⇒
  features not identical (VERIFY, don't assume); only the vision features are reusable — the LLM-side KV is NOT (different weights).

### G10. Cascade-aware distillation of the recoverable slice (shrink the escalation set)  ★ESC-rarer
- **Source:** knowledge distillation (Hinton et al., 2015); cascade/router-aware training; cascade confidence tuning
  "I Know What I Don't Know" (2502.19335).
- **Mechanism:** teach the cheap model the specific cases the expensive model would fix, so those cases stop needing the
  expensive model.
- **Map to us:** identify the **recoverable slice** (7B-wrong ∧ 32B-right) from dumps; LoRA-distill the 32B's answers on
  exactly those questions into the 7B. Post-distillation the 7B answers them correctly → they no longer trigger
  escalation → a **permanently smaller escalation set** at only 7B inference cost. **Distinct from A3** (boost the
  candidate POOL's oracle coverage): G10 transfers the STRONG answer into the cheap model to *remove* escalations.
- **Concrete variation:** GPU — LoRA-tune Lingshu-7B on (image, q, 32B-answer) over a held-out recoverable slice
  (train/test disjoint); re-measure 7B accuracy, the new escalation rate at parity, and cost.
- **Expected ACC:** ≈ cascade at a lower escalation rate. **Latency/compute:** RARER escalation, paid once at train time.
- **How to test:** GPU (distill) + OFFLINE re-cost. **Novelty & risk:** medium — cascade-aware targeted distillation;
  risk: overfitting / poor transfer to unseen questions (the recoverable slice may be idiosyncratic) ⇒ strict held-out
  to avoid a leakage illusion; the intrinsically-hard residual (MedXpert) won't distill.

### G.RANK — cheaper/rarer-escalation axis prioritization (pass 3b) + OFFLINE-testable flag

| ESC-rank | idea | lever | both-axes / effect | novelty | test cost | OFFLINE? |
|---:|---|---|---|---|---|:--:|
| **1** | **G5** Recoverability-aware escalation suppressor | rarer (futile set) | **yes** — rarer at ~0 acc; fixes the exact MedXpert/open-text speed erosion | med-high | OFFLINE | **✓** |
| **2** | **G3** Quantized strong leg (AWQ INT4) | cheaper | **yes** — 4.57×→~1.5× per escalation, ~3× faster; triples the crossover | low-mech/high-lev | OFFLINE re-cost (+GPU confirm) | **✓** |
| **3** | **G8** Speculative prefill prefetch | cheaper (latency) | latency↓ at **zero accuracy risk**; hides the 32B prefill | med-high | OFFLINE latency model (+GPU) | **✓** |
| **4** | **G6** Serial confirmatory 2-stage gate | rarer (specificity) | **yes** — cuts false-positive escalations | med | OFFLINE | **✓** |
| **5** | **G7** Semantic escalation cache | rarer (amortized) | **yes** — fires 32B once per cluster over a stream | med | OFFLINE | **✓** |
| 6 | **G1** Token-level spec-decode of escalation | cheaper (lossless) | latency↓; bounded on short answers | med | OFFLINE* / GPU | partial |
| 7 | **G4** 7B-attention image-token prune (strong fwd) | cheaper (prefill) | **yes** — cuts the dominant prefill term | med | GPU + OFFLINE re-cost | ✗ |
| 8 | **G2** Depth-adaptive / early-exit strong leg | cheaper | **yes** — fewer layers per escalation | med-high | GPU (+probe) | ✗ |
| 9 | **G9** Cross-tier vision-encoder reuse | cheaper (encode) | cheaper *if* ViT shared (verify) | med | OFFLINE check + GPU | partial |
| 10 | **G10** Distill the recoverable slice | rarer (train-time) | rarer, paid once | med | GPU | ✗ |

**Top-5 OFFLINE-testable (this axis): G5, G3, G8, G6, G7** — all re-simulatable on existing per-sample dumps +
measured cost constants, no new inference. **G5** is the highest-value single lever (it attacks the *named* failure —
the 90%-escalation slices where FALC turns slower than always-32B — at ~0 accuracy cost, fully offline). **G3** is the
biggest off-the-shelf per-escalation cost cut (offline re-cost now; a GPU AWQ load only to confirm the <1% accuracy
delta). **G8** is a zero-accuracy-risk latency win. **G6/G7** cut the escalation *rate* (specificity, amortized reuse).
GPU-gated but high-leverage: **G4** (prune the prefill-bound strong forward), **G2** (early-exit), **G10** (distill away
the recoverable escalations). **§G adds 10 ideas (G1–G10); backlog total = 35 (§A–E) + 11 (§F, BEAT-32B) + 10 (§G) = 56.**

**SOURCES (pass 3b — §G, escalation axis):**
- Speculative decoding / cascades: Leviathan et al. (arXiv 2211.17192) · Chen et al. (2302.01318) · Narasimhan et al.,
  *Faster Cascades via Speculative Decoding* (2405.19261) · Google Research, *Speculative Cascades* blog (2025) ·
  *Cluster, Route, Escalate* (2606.27457) · *I Know What I Don't Know: cascade confidence tuning* (2502.19335).
- Early-exit / conditional depth: Schuster et al., *Confident Adaptive Language Modeling / CALM* (2207.07061, NeurIPS
  2022) · LayerSkip (2404.16710) · Mixture-of-Depths, Raposo et al. (2404.02258) · Mixture-of-Recursions (2507.10524) ·
  early-exit survey (2509.05915).
- Quantization: AWQ, Lin et al. (2306.00978, MLSys 2024) · GPTQ, Frantar et al. (2210.17323) · quantized-LLM eval
  survey (2409.11055).
- Visual-token pruning: FastV, Chen et al. (2403.06764) · SparseVLM · QG-VTC (2504.00654) · PyramidDrop · VisionZip.
- Learning-to-defer / early abstention: *Cost-Saving LLM Cascades with Early Abstention* (2502.09054) · Jitkrittum et
  al. recoverability (2307.02764) · *Learning to Partially Defer for Sequences* (2502.01459).
- Diagnostic-testing theory: serial vs parallel testing (specificity/sensitivity trade; confirmatory serial testing) ·
  Dodge–Romig double/multiple acceptance sampling.
- Semantic caching / VLM reuse: GPTCache · *GPT Semantic Cache* (2411.05276) · SCALM · VLCache (2512.12977) · LMCache
  multimodal (vLLM V1) · cross-turn image-KV reuse.
- Prefill/decode disaggregation + prefetch: Perplexity disaggregated prefill/decode · PrefillShare (2602.12029) ·
  FlowKV (2504.03775) · DuetServe (2511.04791).
- Knowledge distillation: Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (2015, arXiv 1503.02531).

---

## H. REMAINING-HEADROOM LEVERS THAT SIDESTEP THE ESTABLISHED WALLS (pass 4 — no better recoverability signal, no better best-of-N selection)  ◆PASS4

> **Why this section exists.** Three walls are now *established*, not hypotheses: (i) the **recoverability wall** — the
> per-sample "will the 32B fix the 7B?" is ~unlearnable (AUROC ≈ 0.6 MCQ / ≈ 0.87 open-text), capping the MCQ beat to PMC;
> (ii) the **selectability wall** — bigger verifiers + filtering don't convert more of open-text best-of-N; (iii) the
> **cheap-strong wall** — the 32B is only 4.57× FLOPs / ~665 ms, so escalation-rate savings are modest. §A–G already
> mined "a better per-sample recovery signal" (dead) and "a better bo-N selector" (walled). **Pass-4's rule: every idea
> must attack the *remaining* headroom WITHOUT relying on either walled quantity.** They do so by four legal moves —
> (a) **adapt the cheap model** so it is simply better (no router needed); (b) route/fuse on an **observable /
> neighborhood / learned slice** or **external symbolic/calibration structure** (higher-AUROC than per-sample
> recoverability by construction); (c) keep-cheap on the futile mass instead of escalating in vain; (d) improve the
> **serving/estimation/certification** machinery the §F/§G program silently assumes. All figures are HYPOTHESES to test.
> Cost anchors reused from §G (real): GEN7 347 ms / 1.0 FLOP-eq; VER7 175 ms; GEN32-nothink 665 ms / 4.57 FLOP-eq. Tag: **◆PASS4**.

### H1. Test-time training / entropy-minimization adaptation of the cheap leg (TTT / TENT / MEMO)  ◆PASS4
- **Axis:** cheap-leg accuracy → rarer escalation + higher pooled acc. **Wall-sidestep:** changes the *model*, not the
  router — needs **no per-sample recoverability signal** and is not best-of-N selection.
- **Source:** Sun et al., *Test-Time Training with Self-Supervision* (arXiv 1909.13231, ICML 2020); Wang et al., *TENT:
  Fully Test-Time Adaptation by Entropy Minimization* (ICLR 2021, OpenReview uXl3bZLkr3c); Zhang et al., *MEMO* (arXiv
  2110.09506); test-time prompt tuning for VLMs (TPT).
- **Mechanism:** before predicting, update a few parameters (BN/LayerNorm affines, a LoRA, or a prompt) on the incoming
  test instance/batch via a label-free loss (prediction-entropy minimization or an SSL head) — adapting to the shifted
  test distribution with zero labels.
- **Map to us:** our eval spans 6 shifted medical domains and the cheap 7B is under-adapted per-domain. A cheap online
  adaptation (entropy-min over the 7B's answer-token logits per benchmark batch) lifts the 7B's *own* accuracy → **both**
  fewer escalations (rarer) AND a higher cheap-leg floor, all **without touching the walled router**. This is the one
  legal way to move the escalation-rate axis by making the cheap leg genuinely better rather than predicting recovery.
- **Concrete variation:** TENT-style entropy-min on the 7B (adapt only LayerNorm affines) per-benchmark batch; measure
  per-benchmark 7B acc lift and the escalation rate at parity vs the un-adapted 7B.
- **Expected ACC:** 7B ↑ per-domain ⇒ pooled ↑ / escalation ↓. **Latency/compute:** a few backward steps amortized over
  a domain batch; a per-domain-adapted 7B has ~zero marginal serving cost.
- **How to test:** GPU (adapt + re-eval). OFFLINE upper-bound proxy: does an oracle per-benchmark recalibration (or the
  measured 7B-think − 7B-nothink gap) already reveal cheap-leg headroom worth adapting for?
- **Novelty & risk:** high — TTT on a medical-VLM *cascade cheap leg* is new. **Risk:** entropy-min can collapse to
  confident-wrong answers (needs a few-step / stability guard); batch adaptation assumes domain-homogeneous batches.

### H2. kNN / retrieval-augmented gating over a labeled-outcome datastore (non-parametric slice router)  ◆PASS4  TOP-5-offline (#3)
- **Axis:** gate / escalation. **Wall-sidestep:** routes on the **empirical recovery rate of the query's nearest labeled
  neighbors** — a continuous, automatically-defined "learned slice," strictly richer than a scalar confidence and **not a
  per-sample recoverability prediction**.
- **Source:** Khandelwal et al., *Generalization through Memorization: Nearest Neighbor LMs (kNN-LM)* (arXiv 1911.00172,
  ICLR 2020); RETRO; retrieval/nonparametric conformal; "route by nearest labeled cases."
- **Mechanism:** store calibration examples as (embedding → outcome) in a datastore; for a new input retrieve k neighbors
  and predict from their labels — memorization generalizes where a parametric head cannot.
- **Map to us:** build a datastore of calibration (image⊕question embedding → {7B-correct?, 32B-correct?, recovered?});
  at test, escalate iff the *neighborhood* recovery rate × headroom > cost. Aggregating neighbor outcomes dodges the
  per-sample wall — §F1 uses hand-enumerated slices, H2 is a continuous data-driven neighborhood (a finer, automatic F1).
  (The both-neighbors-fail set is simply kept on the cheap leg — abstention is forbidden in this project.)
- **Concrete variation:** OFFLINE — embed dumps, leave-one-out kNN over calibration outcomes; compare neighborhood-recovery
  gate AUROC / deferral-curve vs the margin gate and vs F1's manual slices; sweep k and the datastore composition.
- **Expected ACC:** match-or-beat the gate at equal escalation where neighborhoods carry recovery signal.
  **Latency/compute:** an embed + ANN lookup ≪ a 32B call; can *reduce* escalation.
- **How to test:** **OFFLINE** (embeddings + outcome dumps).
- **Novelty & risk:** medium-high — non-parametric retrieval gating for a medical-VLM cascade. **Risk:** neighborhood
  recovery may itself approach ~0.6 AUROC on MCQ (the wall is partly intrinsic) — but it is strictly richer than a scalar
  and strongest on open-text; datastore coverage / embedding choice is load-bearing.

### H3. ⛔ REMOVED — abstention is permanently FORBIDDEN in this project (see the top banner). Do not re-add or re-propose in any form.

### H4. Learned data-centric error-slice discovery for the escalation router (Domino / Spotlight)  ◆PASS4  TOP-5-offline (#2)
- **Axis:** gate / slice (beat-32B + rarer-escalation). **Wall-sidestep:** discovers **coherent, observable** error
  slices in embedding space (a-priori-computable regions), not a per-sample recovery signal.
- **Source:** Eyuboglu et al., *Domino: Discovering Systematic Errors with Cross-Modal Embeddings* (arXiv 2203.14960,
  ICLR 2022); *Spotlight* (d'Eon et al.); *Slice Finder* (Chung et al.); *Active Slice Discovery in LLMs* (arXiv 2511.20713).
- **Mechanism:** fit an error-aware mixture over cross-modal embeddings to automatically *discover and name* the slices
  where a model systematically fails — no manual slice enumeration.
- **Map to us:** F1 routes on *hand-enumerated* slices and worries the beat rides on MMMU (n=150). Run slice-discovery on
  the 7B error set to *learn* which coherent image/text regions the 7B reliably loses (or reliably beats the 32B) → route
  those, guardrail-checked. Turns F1's open question ("is there a 7B-owned cell beyond MMMU?") into a data-driven search
  with natural-language slice descriptions (auditable for clinicians). Feeds H2's datastore and H8's shrinkage.
- **Concrete variation:** OFFLINE — Domino/mixture over (image⊕text) embeddings labeled by 7B-error and by 7B-beats-32B;
  extract high-precision slices with CI-lower-bound ≥ 32B; apply held-out; report macro/pooled vs F1's manual grid.
- **Expected ACC:** finds 7B-owned or always-escalate slices F1's coarse grid misses. **Latency/compute:** discovered
  7B-owned slices are cheap-leg-only (both-axes).
- **How to test:** **OFFLINE** (embeddings + outcome dumps).
- **Novelty & risk:** medium-high — slice-discovery as a cascade router is new here and directly de-risks F1's
  MMMU-dependence. **Risk:** discovered slices can overfit small support (pair with H8 credibility shrinkage); the NL
  descriptions need validation.

### H5. JIT tiered-compilation / profile-guided escalation with deoptimization (HotSpot)  ◆PASS4
- **Axis:** escalation-rate (stream-adaptive) + gate. **Wall-sidestep:** promotes/demotes *query clusters* by running
  invocation-and-recovery **counters over a stream** (online experience), not per-sample recovery prediction.
- **Source:** HotSpot JVM tiered compilation; profile-guided optimization; deoptimization / on-stack replacement (Hölzle,
  Chambers & Ungar, PLDI 1992); adaptive optimization systems.
- **Mechanism:** run cheap (interpreted) and only JIT-compile a method to a higher tier once its invocation counter shows
  it is "hot"; *deoptimize* back down when a speculative assumption stops holding.
- **Map to us:** in a deployment stream, keep per-query-cluster counters of *realized* 32B-recovery. Promote a cluster to
  "always-escalate" (tier-up) once its running recovery rate justifies the 665 ms, and *deoptimize* it to "7B-only" when
  the rate decays — a self-tuning online escalation policy needing only cluster-level statistics, no per-sample signal.
  Distinct from F1 (static slices) and G7 (answer caching): H5 adapts the *policy* over time and handles drift.
- **Concrete variation:** OFFLINE stream-replay — bucket the stream into clusters (H2/H4 regions), run counter-based
  promote/demote with hysteresis, compare accuracy/escalation vs a static gate on a shuffled replay.
- **Expected ACC:** ≈ static gate. **Latency/compute:** escalation concentrated on provably-hot (recoverable) clusters ⇒
  rarer, cheaper aggregate; drift-adaptive.
- **How to test:** **OFFLINE** (streamed replay of dumps).
- **Novelty & risk:** medium — tiered-JIT framing of adaptive escalation over a stream. **Risk:** per-cluster cold-start;
  counter thresholds/hysteresis need tuning; assumes recurring query types (holds for templated clinical VQA).

### H6. Cost-based query-optimizer plan selection + adaptive re-optimization (DB optimizers / eddies)  ◆PASS4
- **Axis:** gate / per-query plan selection. **Wall-sidestep:** chooses a *plan* from cheap cost/quality **estimates over
  observable features** + runtime re-planning; no per-sample recovery prediction.
- **Source:** Selinger et al., *Access Path Selection in a Relational DBMS* (SIGMOD 1979, cost-based optimization); Avnur
  & Hellerstein, *Eddies: Continuously Adaptive Query Processing* (SIGMOD 2000); learned cardinality estimation.
- **Mechanism:** enumerate plans, pick the min-estimated-cost one from statistics; adaptive processing *re-routes
  mid-execution* when the estimate proves wrong (eddies).
- **Map to us:** per query, choose among plans {7B-greedy, 7B-bo-N, 7B→32B, 32B-direct} by an estimated (cost, P(correct))
  from *cheap pre-execution statistics* (question length, modality, H2-neighbor difficulty); then the novel bit —
  **late re-optimization**: begin the cheap plan and switch mid-flight if a runtime statistic (7B decode entropy,
  self-verify) exceeds the estimate. "Predicate pushdown" = run the cheapest discriminative check first. Distinct from
  C2/VOC by centering *cost estimation from statistics* + runtime re-plan (DB practice), not a value-of-computation calc.
- **Concrete variation:** OFFLINE — fit a cheap per-plan cost/accuracy estimator on observable features; simulate min-cost
  plan selection + a re-plan trigger on runtime stats; compare the frontier vs the fixed gate and vs BEST-Route (E3).
- **Expected ACC:** match at lower cost by picking the cheapest *sufficient* plan. **Latency/compute:** per-query plan
  spends only where the estimate says it pays (both-axes).
- **How to test:** **OFFLINE** (dumps have the features + all plan outcomes).
- **Novelty & risk:** medium — DB-optimizer + eddy re-optimization framing for an LLM cascade. **Risk:** overlaps E3/C2
  in spirit; the fresh contribution is estimation-from-statistics + mid-flight re-plan; estimator quality bounds gains.

### H7. Network QoS / DiffServ admission control + traffic shaping + priority load-shedding of escalations  ◆PASS4
- **Axis:** escalation-rate under load / serving. **Wall-sidestep:** a **deployment/throughput** lever (differentiated,
  load-adaptive escalation) orthogonal to both accuracy walls.
- **Source:** DiffServ (RFC 2475); Weighted Fair Queueing; token-bucket / leaky-bucket traffic shaping; admission
  control; RED.
- **Mechanism:** mark packets into service classes for differentiated treatment; admission control refuses flows that
  would break the SLA; token-bucket shapes burst rate; under overload, low-priority classes are shed first (graceful
  degradation).
- **Map to us:** treat the 32B as a bandwidth-limited resource and each escalation as a packet with a QoS class from
  (clinical criticality × format × modality). Admission-control the escalation queue: under load, *shed* low-priority
  escalations (answer with the 7B) and reserve 32B "bandwidth" for high-criticality classes; a token-bucket caps the
  escalation burst so tail latency stays bounded. Distinct from C6 (fixed output-rate HEP gate): H7 is
  *priority-differentiated* + *load-adaptive* with explicit graceful degradation.
- **Concrete variation:** OFFLINE — simulate a load profile; assign QoS classes; apply admission control + WFQ +
  token-bucket; report per-class accuracy, escalation rate, and tail latency vs a flat gate under varying load.
- **Expected ACC:** ≈ gate at low load; degrades *gracefully by priority* under load (high-criticality accuracy
  preserved). **Latency/compute:** bounded tail latency + throughput guarantee under contention.
- **How to test:** **OFFLINE** (load simulation over dumps + measured costs).
- **Novelty & risk:** medium — QoS/admission-control framing of cascade serving. **Risk:** mostly a serving contribution
  (not an accuracy lift); needs a criticality taxonomy; value shows only under contention.

### H8. Actuarial credibility-weighted (Bühlmann) escalation pricing / bonus-malus  ◆PASS4  TOP-5-offline (#4)
- **Axis:** gate estimation robustness (enabler for §F/§G/H4). **Wall-sidestep:** fixes the **thin-slice-overfit failure
  mode** that F1/H4/G5 all flag — shrinks small-slice recovery estimates toward the global rate for *robust* per-slice
  routing; it is estimation machinery, not a new recovery signal.
- **Source:** Bühlmann credibility theory (1967); Bühlmann–Straub; bonus-malus systems (Lemaire); actuarial experience
  rating; insurance error-cost pricing.
- **Mechanism:** price a risk by blending its individual experience with the group mean, weighted by a credibility factor
  Z = n/(n+k) that grows with data — provably minimizes estimation MSE for thin data; bonus-malus updates the price by
  realized claims.
- **Map to us:** per-slice escalation decisions (F1/H4/G5) fail when a slice has little calibration support. Replace the
  raw slice recovery-rate with a **credibility-shrunk** estimate (Z·slice-rate + (1−Z)·global-rate) and "price" each
  escalation by expected error-cost from it. Bonus-malus = online update of a slice's escalation propensity from realized
  outcomes (composes with H5). This is the missing estimator that makes the whole per-slice program guardrail-honest.
- **Concrete variation:** OFFLINE — recompute F1/H4/G5 slice decisions with Bühlmann-shrunk recovery rates; compare
  held-out guardrail violations (slices that looked 7B-owned but regress) vs unshrunk; tune k by cross-validation.
- **Expected ACC:** fewer held-out guardrail failures ⇒ a *robust* (if smaller) beat/save. **Latency/compute:** pure
  estimation (free).
- **How to test:** **OFFLINE**.
- **Novelty & risk:** medium — credibility theory as the estimator for cascade slice-routing is novel and fixes a
  repeatedly-flagged risk. **Risk:** it *shrinks* apparent gains (honest, not a booster); the pooling hierarchy matters.

### H9. Neuro-symbolic medical-constraint filter + violation-triggered escalation  ◆PASS4  TOP-5-offline (#5)
- **Axis:** candidate-filter + gate. **Wall-sidestep:** **sidesteps BOTH walls** — a *hard logical* pre-filter over
  answers (not a learned recovery signal, not best-of-N scoring) that can catch confident-wrong answers *both* models make.
- **Source:** Xu et al., *A Semantic Loss Function for Deep Learning with Symbolic Knowledge* (arXiv 1711.11157, ICML
  2018); Manhaeve et al., *DeepProbLog* (NeurIPS 2018, arXiv 1805.10872); Logic Tensor Networks; constraint-satisfaction
  decoding.
- **Mechanism:** encode known constraints (mutual exclusivity, anatomical/laterality plausibility, unit/range validity)
  as logic and filter/penalize outputs that violate them.
- **Map to us:** apply a medical-constraint checker to *both legs'* answers: (a) prune symbolically-impossible answers
  before any fusion/selection, and (b) use a constraint *violation* as a high-precision escalation trigger (a 7B answer
  that contradicts a stated finding or laterality is likely wrong → escalate). Reaches the *shared* confident-wrong errors
  the recoverability wall otherwise leaves unfixable, via external knowledge rather than a better router.
- **Concrete variation:** OFFLINE — build a small constraint set (laterality consistency, yes/no-with-evidence, MCQ option
  mutual-exclusivity) from the benchmarks; scan dumped answers for violations; measure (i) what fraction of 7B errors are
  constraint-violations (a free escalation trigger) and (ii) the precision of "violation ⇒ wrong."
- **Expected ACC:** removes a class of shared confident-wrong answers + adds a high-precision escalation signal.
  **Latency/compute:** a symbolic check is ~free; violation-gating escalates only flagged items.
- **How to test:** **OFFLINE** if constraints are checkable from answer text; light GPU only if a constraint needs a parse.
- **Novelty & risk:** high — neuro-symbolic constraints as a cascade filter/gate is new here and uniquely sidesteps *both*
  walls. **Risk (honest):** clean, universal medical constraints are scarce in generic VQA (coverage is the bottleneck);
  a mis-specified constraint can reject correct answers — keep the constraint set high-precision and small.

### H10. Multicalibration / group-wise calibration to make the §F beat-32B fusion guardrail-safe  ◆PASS4
- **Axis:** enabler for beat-32B fusion (F3/F8/F11). **Wall-sidestep:** not a new signal — makes the *existing*
  cross-model confidence-advantage valid **simultaneously across all slices**, so §F beats hold per-slice not just marginally.
- **Source:** Hébert-Johnson et al., *Multicalibration* (ICML 2018, PMLR v80); Guo et al., *On Calibration of Modern
  Neural Networks* / temperature scaling (arXiv 1706.04599, ICML 2017); Platt / isotonic; group-wise calibration.
- **Mechanism:** post-process a predictor so it is calibrated *simultaneously* on every computationally-identifiable
  subgroup, not just marginally.
- **Map to us:** F3/F11 route/fuse on cross-model *calibrated* confidence — but marginal calibration drifts per slice/OOD,
  so the confidence-advantage sign can be wrong exactly where it is used. Multicalibrate each leg's confidence over the
  slice family (dataset × format × modality) so the advantage is trustworthy per slice → F3 (override) and F8 (veto)
  become guardrail-safe across all slices at once. Enabling infrastructure that determines whether the beat-32B claims survive.
- **Concrete variation:** OFFLINE — multicalibrate 7B/32B confidences over the slice family; re-run F3 and F8 with
  multicalibrated vs merely temperature-scaled confidences; report per-slice net-flip and guardrail violations.
- **Expected ACC:** converts marginal beats into per-slice-robust beats (or honestly reveals they don't hold).
  **Latency/compute:** calibration is free.
- **How to test:** **OFFLINE**.
- **Novelty & risk:** medium — multicalibration as the correctness guarantee for cross-model cascade fusion. **Risk:**
  multicalibration needs enough per-group data (couple with H8 shrinkage); it fixes calibration, not the advantage mass.

### H11. Real-time imprecise-computation + EDF scheduling of the shared 32B under a batch deadline  ◆PASS4
- **Axis:** latency / serving under SLA. **Wall-sidestep:** a deployment lever (both-axes under a deadline), orthogonal
  to the accuracy walls; complements D4 (single-query anytime) with a *batch scheduler over a shared strong resource*.
- **Source:** Liu et al., *imprecise computation* (mandatory + optional parts); Liu & Layland, EDF scheduling (JACM 1973);
  real-time admission control.
- **Mechanism:** every task has a *mandatory* part (must finish) and an *optional* part (refines the result if time
  allows); a scheduler (EDF + admission) maximizes total refinement value under a deadline.
- **Map to us:** every query's *mandatory* part = the 7B answer (always produced); the *optional* part = the 32B
  refinement, scheduled onto the shared 32B only if batch deadline slack permits, admission-ordered by expected
  recovery × criticality (from H2/H8). Under a tight SLA the system still returns every 7B answer and refines as many as
  fit. Distinct from D4 (per-query performance profile) and H7 (priority classes): H11 is the mandatory/optional
  decomposition + EDF over the shared 32B.
- **Concrete variation:** OFFLINE latency model — split GEN32 into schedulable units; simulate EDF + imprecise-computation
  admission over a batch at several SLAs; report refined-fraction, accuracy, and deadline-miss rate vs greedy escalate-all.
- **Expected ACC:** maximized-given-deadline; graceful under load. **Latency/compute:** SLA-guaranteed; 32B spent on the
  highest-value refinements.
- **How to test:** **OFFLINE** (scheduling sim over measured costs) + GPU for real interrupts.
- **Novelty & risk:** medium — imprecise-computation / EDF framing of cascade refinement under deadlines. **Risk:** a
  serving wrapper (gains SLA-dependent); needs a per-query refinement-value estimate (use robust H8 estimates).

### H12. Active-testing / label-efficient certification of the cascade guardrails  ◆PASS4
- **Axis:** deployment / label-efficiency (enabler). **Wall-sidestep:** orthogonal to both accuracy walls — makes the
  per-slice **guardrail certification** every §F/§G/H idea assumes affordable under a scarce clinician-labeling budget.
- **Source:** Kossen et al., *Active Testing: Sample-Efficient Model Evaluation* (arXiv 2103.05331, ICML 2021); *Active
  Surrogate Estimators* (arXiv 2202.06881); *ASPEST* (arXiv 2304.03870, active-learning × selective-prediction).
- **Mechanism:** estimate a model/policy's true performance with far fewer labels by *actively acquiring* the most
  informative test points (unbiased, variance-reduced).
- **Map to us:** the whole program rests on per-slice guardrails ("is this slice really 7B-owned? does the veto really not
  regress?") certified on labeled calibration data — but clinical labels are scarce. Use active testing to choose *which*
  items a clinician labels to certify a slice-routing / veto policy to a target CI with minimal budget, and to actively
  target the slices whose routing decision is most uncertain. Turns "assume a labeled calibration set" into a
  label-efficient protocol, and enables cheap re-certification under drift (composes with H5).
- **Concrete variation:** OFFLINE — simulate active-testing acquisition on the existing labels: how few labels certify
  F1/H4 slice guardrails and the F8 veto risk to a target CI, vs random labeling?
- **Expected ACC:** unchanged; the win is *certification cost* (labels) and tighter guardrail CIs per label.
  **Latency/compute:** n/a (a labeling-efficiency lever).
- **How to test:** **OFFLINE** (subsample the labeled dumps as an acquisition simulation).
- **Novelty & risk:** medium — active testing applied to cascade-guardrail certification is a fresh, practically-important
  axis. **Risk:** not an accuracy/latency lever directly; value is realized only in a label-constrained deployment.

### H.RANK — remaining-headroom axis prioritization (pass 4) + OFFLINE-testable flag

Ranked by (remaining-headroom impact GIVEN the walls × novelty × testability × cleanliness of the wall-sidestep):

| # | idea | axis / how it sidesteps the walls | both-axes? / effect | novelty | test | OFFLINE? |
|---:|---|---|---|---|---|:--:|
| ⛔ | ~~H3 three-way abstain-to-human~~ **REMOVED — abstention forbidden in this project; do not re-add** | — | — | — | — | — |
| **2** | **H4** Learned error-slice discovery (Domino) | routes on **learned observable slices**; de-risks F1's MMMU-dependence | **yes** — owned slices are cheap-leg-only | med-high | OFFLINE | **✓** |
| **3** | **H2** kNN retrieval-augmented gating | routes on **neighborhood** empirical recovery, not per-sample | acc≈/↑ at lower esc | med-high | OFFLINE | **✓** |
| **4** | **H8** Actuarial credibility shrinkage | fixes **thin-slice overfit** across §F/§G/H4 (estimator, not a signal) | robustifies the beat/save | med | OFFLINE | **✓** |
| **5** | **H9** Neuro-symbolic constraint filter/gate | **hard logical** check — sidesteps BOTH walls | acc↑ + high-precision esc trigger | high | OFFLINE | **✓** |
| 6 | **H1** Test-time training of the cheap leg | adapts the **model**, not the router | **yes** — 7B↑ ⇒ rarer esc | high | GPU (+OFFLINE proxy) | partial |
| 7 | **H10** Multicalibration of §F fusion | makes cross-model advantage valid **per-slice** | enables robust beat-32B | med | OFFLINE | ✓ |
| 8 | **H5** JIT tiered-compilation escalation | per-cluster running **counters** over a stream | **yes** — rarer, drift-adaptive | med | OFFLINE | ✓ |
| 9 | **H6** Cost-based query-planner routing | plan choice from **statistics** + runtime re-plan | acc≈, cost↓ | med | OFFLINE | ✓ |
| 10 | **H11** Imprecise-computation / EDF scheduling | mandatory 7B + optional 32B under deadline | latency guarantee | med | OFFLINE (+GPU) | ✓ |
| 11 | **H7** Network-QoS admission / load-shedding | priority-differentiated, load-adaptive escalation | throughput guarantee | med | OFFLINE | ✓ |
| 12 | **H12** Active-testing guardrail certification | label-efficient certification of the slice guardrails | −labeling cost | med | OFFLINE | ✓ |

**★ TOP OFFLINE-TESTABLE (pass 4), test first: H4, H2, H8, H9, H1** — all re-simulatable on the existing per-sample
dumps + measured cost constants, zero new inference, and each attacks the remaining headroom by a *different* legal
move: **H4** learns the observable slices F1's manual grid misses (de-risks the MMMU-only beat), **H2** routes on a
data-driven neighborhood recovery rate (richer than a scalar, not per-sample), **H8** shrinks thin-slice estimates so
the per-slice program is guardrail-honest, **H9** uses external symbolic knowledge to reach the *shared* confident-wrong
errors both walls otherwise leave untouched, and **H1** adapts the cheap leg (test-time training). They compose: H4
discovers slices → H8 shrinks their estimates → H2 makes them continuous → H9 catches the shared logical errors.
**Honest meta-caveat:** given the walls, the realizable MCQ beat stays small; H1 (adapt the cheap leg) is the pass-4
idea most likely to move a headline number, while H4/H8 mainly make the §F beat-32B claims *robust and honest* rather
than larger. **⛔ H3 (abstain) is REMOVED — abstention is forbidden in this project. §H lists H1–H12 but H3 is void;
backlog total = 35 (§A–E) + 11 (§F) + 10 (§G) + 12 (§H, H3 void) = 68 listed / 67 pursuable.**

**SOURCES (pass 4 — §H, remaining-headroom axis):**
- Test-time training / adaptation: Sun et al., *Test-Time Training with Self-Supervision* (arXiv 1909.13231, ICML 2020) ·
  Wang et al., *TENT: Fully Test-Time Adaptation by Entropy Minimization* (ICLR 2021, OpenReview uXl3bZLkr3c) · Zhang et
  al., *MEMO* (arXiv 2110.09506).
- Retrieval / non-parametric: Khandelwal et al., *Generalization through Memorization: kNN-LM* (arXiv 1911.00172, ICLR 2020).
- Selective prediction / abstention: El-Yaniv & Wiener, risk-coverage (2010) · Geifman & El-Yaniv, *Selective
  Classification for DNNs* (arXiv 1705.08500, NeurIPS 2017) · *SelectiveNet* (ICML 2019) · *ASPEST* (arXiv 2304.03870).
- Slice discovery: Eyuboglu et al., *Domino* (arXiv 2203.14960, ICLR 2022) · *Spotlight* (d'Eon et al.) · *Slice Finder*
  (Chung et al.) · *Active Slice Discovery in LLMs* (arXiv 2511.20713).
- JIT tiering: HotSpot tiered compilation · Hölzle, Chambers & Ungar, deoptimization / on-stack replacement (PLDI 1992).
- DB query optimization: Selinger et al., *Access Path Selection* (SIGMOD 1979) · Avnur & Hellerstein, *Eddies* (SIGMOD 2000).
- Network QoS: DiffServ (RFC 2475) · Weighted Fair Queueing · token-bucket / leaky-bucket shaping · admission control · RED.
- Actuarial credibility: Bühlmann credibility theory (1967) · Bühlmann–Straub · bonus-malus (Lemaire).
- Neuro-symbolic: Xu et al., *A Semantic Loss Function* (arXiv 1711.11157, ICML 2018) · Manhaeve et al., *DeepProbLog*
  (NeurIPS 2018, arXiv 1805.10872) · Logic Tensor Networks.
- Calibration: Hébert-Johnson et al., *Multicalibration* (ICML 2018, PMLR v80) · Guo et al., temperature scaling (arXiv
  1706.04599, ICML 2017).
- Real-time scheduling: Liu et al., imprecise computation · Liu & Layland, EDF (JACM 1973).
- Active testing: Kossen et al., *Active Testing* (arXiv 2103.05331, ICML 2021) · *Active Surrogate Estimators* (arXiv 2202.06881).
