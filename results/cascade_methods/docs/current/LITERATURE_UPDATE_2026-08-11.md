# LITERATURE UPDATE — 2026-08-11

**Scope.** A deep external literature sweep aimed at one question: *how do we beat a strong single forward
pass of a 32B medical VLM, on accuracy or on cost, using a 7B plus test-time compute?* Prioritises
2025–2026 work. Written against this project's state as of 2026-08-10. ~90 papers examined; ~70 verified.

---

## HOW TO READ THIS DOCUMENT — provenance rules

This file obeys CRITICAL RULE 7 with an extra layer, because every number here comes from *somebody
else's* paper:

| tag | meaning |
|---|---|
| `[V]` | **Verified by me.** I fetched the arXiv `/abs/` page or official page and read title + authors + date + the number there. |
| `[V-html]` | Verified by me from the paper's rendered HTML full text. |
| `[V*]` | **Verified by a delegated sub-sweep** which fetched the canonical page and read the metadata/number off it. One remove from me; treat as solid for citation metadata, and re-read the PDF before quoting a magnitude in the paper. |
| `[S]` | **Search-snippet only, unverified.** A lead, not a fact. Never quote. |
| `[MINE]` | **My arithmetic** on numbers the paper reports. Inputs are `[V]`; the derived quantity is mine. |
| `[REPO]` | This project's own measured number, artifact file named. Quoted from that artifact, not re-derived here. |

**Standing warnings.**

1. **A claim is not a measurement.** Every entry separates the paper's *claim* from what it *measured*.
2. **Venue attribution is unreliable for 2026 preprints.** Where unresolved, I say so; cite the arXiv id.
3. **CRITICAL RULE 6 applies.** The 2026 medical literature has a substantial abstention / triage-to-human
   strand. It is **recorded as landscape and explicitly excluded as a direction.** Nothing in the
   shortlist is an abstention method, and §11 lists the papers to recognise-and-avoid.
4. **One search-snippet claim was checked and found false** (§12). That is the failure rate to expect from
   anything marked `[S]`.

---

## 0. EXECUTIVE SUMMARY — the five things that most change what this project should do next

### 0.1 The tie is what the current literature *predicts*. Reframe the paper around that, and stop apologising.

Three independent 2025–2026 results converge on one boundary condition, and medical VQA sits on the wrong
side of all three:

- **Snell et al.** state the enabling condition verbatim: test-time compute beats a 14× larger model *"on
  problems where a smaller base model attains somewhat non-trivial success rates"* — on **hard** questions
  pretraining wins instead (arXiv:2408.03314) `[V]`.
- **Zhao, Hooi & Ng**: on **knowledge-intensive** tasks more test-time compute *"does not consistently
  improve accuracy and often leads to more hallucinations,"* with the information-theoretic reason that
  *"compute-only test-time scaling, as a post-processing procedure of a fixed model, cannot introduce new
  information about the ground-truth answer"* (arXiv:2509.06861) `[V]`.
- **Sammani, Chamiti & Deligiannis** (ECCV 2026): in the first broad LVLM test-time-scaling study, TTS
  *degrades* performance on perception-focused benchmarks (arXiv:2606.28864) `[V-html]`, `[V*]`.

Medical VQA is knowledge-intensive **and** perception-heavy: Thapa et al. measure that only **32.8%** of
questions across 11 biomedical benchmarks require complex reasoning (arXiv:2505.11462) `[V]`. §7 works the
enabling condition through cell by cell: **we satisfy one of five criteria.** The tie is the correct
scientific outcome. And note the bar the field actually publishes at — *Cluster, Route, Escalate* reports
**retaining 97–99%** of the strong model's accuracy `[V]`, and *SAFE-Cascade* claims **parity** and calls
its +1.4 pp statistically uncertain `[V*]`. **This project's tie is a stronger accuracy result than the
published applied comparators, on 100× the sample size, with CIs.**

### 0.2 The compute reversal is a *structural theorem about cascades*, now published twice — and the fix is named.

Bouchard (arXiv:2605.06350) `[V]`, `[V*]`, on 5 benchmarks × 8 models × 5 providers:

> *"A lightweight pre-generation router exceeds the best cascade policy on four of five datasets, **mainly
> because it avoids the cheap model's generation cost on queries sent directly to a larger model** rather
> than because of a stronger routing signal. These results suggest that **cascade performance is limited
> primarily by structural cost, since cascades pay the cheap model before any escalation decision**."*

Mahmood (arXiv:2602.09902) `[V]` reaches the same place from mechanism design: *"in nearly all cases, the
optimal routing policy involves a static policy with no cascading."* This is an independent, external,
theoretical account of why this project's macro compute came out at **1.196× / 1.410×** of always-32B-direct
`[REPO]` — at 44.24% MCQ escalation under equal weight you pay the 7B on every query and the 32B on nearly
half. **This converts an embarrassing 2026-07-30 correction into a predicted consequence of cascade
geometry.** Add a pre-generation-router operating point (shortlist **S2**, zero GPU).

### 0.3 Selection efficiency ≈ 0.78–0.81 is a field constant, and the medical verifier category is externally confirmed closed.

- **Agentic boosting (arXiv:2605.14163)** `[V]`: GPT-5.4 nano on SWE-bench Verified, **67.0% → 76.4%** with
  a critic-comparator committee at k=8, oracle best-of-8 **79.0%** ⇒ conversion **78.3%** `[MINE]`. This
  project: incumbent clean verifier **sel_eff 0.775204**, best of ~20 architectures **0.810627**
  (`headroom_percell_2026-08-10.json`, `COMPARATIVE_VERIFIER_2026-08-05.md`) `[REPO]`. Same number,
  different domain, frontier models.
- **Verification Mirage (arXiv:2605.10850)** `[V-html]`, `[V*]` — six VLMs **including Lingshu** × five
  medical VQA datasets (VQA-RAD, PathVQA, SLAKE, PMC-VQA, MedXpertQA): verifier error **>40%**, **FPR
  ≳60%**, ~**95–100%** FPR on differential diagnosis; a generator error raises the odds of verifier failure
  **57×** (p<0.001); over four turns **69.5–87.1%** of wrong answers are *locked in* and only **2.2–3.8%**
  corrected; and scaling to a larger same-family verifier gives **no significant FPR reduction for Lingshu
  (p = 0.782)**.
- **Best-of-Evidence (arXiv:2607.20950)** `[V-html]` — best-of-16 on four medical VQA datasets with a
  **Qwen3-VL-235B** judge over a **Qwen3-VL-30B** generator: **+0.26 to +0.58 pp** over raw majority.

⇒ **Stop verifier work.** §11 lists the kills, including two new ones the literature just closed.

### 0.4 Four papers from May–July 2026 now overlap this project's claims. This is the urgent item.

Nobody has published this project's paper — but the space closed in fast, and **none of these are in
CLAUDE.md or the retrospective**:

| paper | what it takes | what this project still owns |
|---|---|---|
| **Wasserstein Equilibrium Decoding for Reliable Medical VQA**, arXiv:2605.18313 `[V*]` | **generator–verifier best-of-N for medical VQA on small VLMs (2–8B)**, temperature-diverse sampling + semantic dedup + a stopping rule; +3.5 pp VQA-RAD (Qwen3-VL-2B, p<0.01) | 8 cells vs 2, cross-model escalation, trained LoRA verifier, format-aware routing, FLOP/latency/energy accounting |
| **Verification Mirage**, arXiv:2605.10850 `[V-html]` | the **selection wall**, on 5 medical datasets incl. Lingshu | ours is a *trained* verifier against their *untrained self*-verification — their result is the argument **for** our design |
| **Oracle Gap and Signal Fidelity**, arXiv:2607.17531 `[V]` | the **decomposition** (oracle gap / signal fidelity / recoverable mass / conditional selection quality / harm-to-correct) | ours is medical + multimodal; theirs is code/math. **Cite, do not claim priority** |
| **Best-of-Evidence**, arXiv:2607.20950 `[V-html]` | **medical best-of-N SOTA** with a 235B judge | ours is a deployable 7B-scale system; theirs needs a 235B judge for <1 pp |

**Action: read 2605.18313 and 2605.10850 in full this week and position explicitly against them.**

### 0.5 Training the cheap leg is the field's answer, this project has never done it, and there is now a 7B=32B result.

Every one of this project's ~16 recoverability mechanisms and ~20 selector architectures **froze the 7B**.

- **REOPOLD** (arXiv:2603.11137) `[V*]`: relaxed on-policy distillation *"enables a 7B student to match a
  32B teacher in visual reasoning with a ~3.32× inference speedup."* ⚠ Abstract-level claim; the benchmark
  carrying it is not named. **This is the "why didn't you just distil?" reviewer question, in one line.**
- **Cascade-Aware Training** (arXiv:2406.00060) `[V]`, `[V*]`: trains the small model aware of the
  downstream model, moving *both* its accuracy *and* its confidence calibration — the latter being what
  the gate thresholds. Full text: **−50% FLOPs at fixed 86% accuracy** on SuperGLUE/WMT22/FLAN2021 `[V*]`.
- **AutoRelAnnotator** (arXiv:2606.25871) `[V*]` gives the cleanest decomposition anywhere: *"fine-tuning
  contributes 20 accuracy points while cascading is approximately accuracy-neutral but halves compute
  cost."* **They are orthogonal levers; they do not substitute.**
- **Online in-context distillation** — Inter-Cascade (arXiv:2509.22984) `[V*]`: the strong model writes a
  reusable strategy on each deferral, augmenting the weak model's context. **Weak-model accuracy +33.06%,
  system accuracy +6.35%, strong-model calls −48.05%, cost −49.63% — with zero parameter updates.** Its
  VLM sibling ICD (arXiv:2510.18117) `[V*]` reports +33% on small VLMs using as little as **4%** teacher
  annotation, with student-uncertainty conditioning to minimise teacher queries.

And this project's own 2026-08-10 diagnosis points the same way: the open-arm coverage failure is
**"a CAPABILITY failure, not a DIVERSITY failure"** — 71.5% of no-coverage questions have **zero** gold
tokens anywhere in the 8-answer pool, and a 3×-budget independent redraw rescues only **21.2%**
(`coverage_diagnosis_2026-08-10.json`) `[REPO]`. **No amount of test-time sampling fixes a capability gap.**

**Bonus, and it is nearly free:** there is a **newer Lingshu you do not know about** — `Lingshu-I-8B`
(InternVL3-based, same pipeline, HF org updated **2026-02-24**) `[V*]`, average **64.5** vs Lingshu-7B's
61.8, with **PathVQA 74.9 (+13.0)** and **SLAKE 91.6 (+8.5)** but **MMMU-Med 49.1 (−4.9)**. PathVQA cells
are this project's load-bearing ones. See **S6**.

### 0.6 ⚠ MEASURED DURING THIS SWEEP — `test_2.csv` carries the **training** split's answer-position bias

Following up an unverified claim in the PMC-VQA literature (§5.4), I counted the gold-option distribution
in both splits directly on disk. **Zero GPU, read-only, one command** (`/data/dan/dataset/medevalkit/PMC-VQA/`):

| split | n | A | B | C | D | **B+C** |
|---|---:|---:|---:|---:|---:|---:|
| `test_clean.csv` (v1, human-verified) | 2,000 | 21.9% | 31.9% | 30.5% | 15.8% | **62.4%** |
| **`test_2.csv` (v2 — MedEvalKit/paper track)** | 33,430 | **13.2%** | **35.8%** | **37.8%** | **13.1%** | **73.6%** |

`[MINE]`, measured 2026-08-11 from the files at
`/data/dan/dataset/medevalkit/PMC-VQA/{test_clean.csv,test_2.csv}` (columns `Answer_label` and `Answer`
respectively).

**Three consequences.**

1. **`test_clean`'s 62.4% reproduces the original PMC-VQA paper's reported *test* distribution exactly**
   (21.9 / 31.9 / 30.5 / 15.8 — §5.4) `[S]`. **`test_2`'s 73.6% instead matches that paper's reported
   *train* distribution (73.5%)** `[S]`. The v2 test split has the *training* split's answer-position
   profile. *(The paper's per-split percentages are `[S]`; my two measured distributions are `[MINE]`.)*
2. **A constant "always C" guesser scores 37.8% on `test_2`** — against Lingshu-7B's measured **0.5427** and
   always-32B-direct's **0.5518** on that cell `[REPO]`. **A single-letter constant recovers 68.5% of the
   32B's accuracy** `[MINE]` on the cell carrying **79.2%** of the sample-weighted pool and **100% of this
   project's MCQ-side macro delta** `[REPO]`.
3. **This is a live confound for a confidence/margin gate**, and it is a second, independent reason —
   beyond the split-provenance landmine already in CLAUDE.md — to keep the macro convention. It is also
   the concrete finding the unwritten PMC-VQA split paper (§5.4) would be built on.

**Not established:** whether this changes any measured result. It does not alter accuracy numbers; it
changes how they should be *interpreted*, and it should become a landmine. The follow-up (option-order
circular consistency as a gate signal, retrospective §8.2 item 14) now has a stronger motivation on PMC
specifically.

---

## 1. AREA 1 — Test-time compute scaling, 2025–2026

### 1.1 The foundational compute-optimal results, and their fine print

**Snell, Lee, Xu & Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model
Parameters.* arXiv:2408.03314, 2024 (ICLR 2025).** `[V]`
- **Claims:** >4× TTS efficiency over a best-of-N baseline; a smaller model beating a **14×** larger one at
  matched FLOPs.
- **Measures:** math (MATH), PaLM-2-class models, revisions + PRM-guided search.
- **The load-bearing fine print, quoted:** the win holds *"on problems where a smaller base model attains
  somewhat non-trivial success rates."* On hard questions, and as the inference-to-pretraining token ratio
  grows, **pretraining is preferable.** This is the single most decision-relevant sentence in the sweep.

**Liu et al. *Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling.*
arXiv:2502.06703, 2025.** `[V]` — 1B>405B, 0.5B>GPT-4o, 3B>405B, 7B>o1/R1, **on MATH-500 and AIME24 only**,
with a PRM in the loop and a strategy tuned per (task, policy, PRM). A *math* result; it does not transfer
without a comparable verifier, and §2 says none exists in this domain.

### 1.2 The VLM-specific evidence — new since this project's design was fixed

**Sammani, Chamiti & Deligiannis. *On Test-Time Scaling for Vision-Language Models.* arXiv:2606.28864,
2026 — ECCV 2026** `[V-html]` (venue `[V*]`). *The most important single paper in this sweep.*
- **Measures:** 13 models — **Qwen3-VL** 2B/4B/8B/32B, **Qwen2.5-VL** 7B/32B/72B, **InternVL-3.5**
  2B/4B/8B/38B, **Molmo2** 4B/8B — × **9 TTS methods** (CoT, Structured-CoT, Plan-and-Solve,
  Self-Consistency, Self-Aggregation, Self-Refinement, Describe-Answer, Compositional-CoT, Prompt
  Repetition) × **6 benchmarks** (MMStar, RealWorldQA, HallusionBench, WeMath, LogicVista, A-OKVQA).
- **Abstract, verbatim:** *"1) different from previous findings, **small, well-performing models benefit
  the most from test-time scaling, enabling performance improvements of up to around 30%, reaching large
  models performance, and often outperforming them**, 2) LVLMs lose focus when given more compute than
  necessary, and 3) Visual information is encoded early in the reasoning chain, after which the chain is
  dominated by text-only reasoning and the contribution of image tokens drops significantly."*
- **Measured specifics:** *"Qwen3-VL-4B with simple CoT and S-CoT already surpasses the baseline performance
  of Qwen3-VL-32B on HallusionBench, WeMath and LogicVista. On LogicVista … +13% improvement over
  Qwen3-VL-32B."* Self-Consistency: Qwen3-VL-2B **35% → 64%** on WeMath.
- **The negative — this project's Finding 1, replicated externally in the general domain:** TTS methods
  *"often degrade performance (rather than maintaining it) on primarily perception-focused benchmarks that
  require limited or no reasoning (e.g., RealWorldQA)"*; *"perception and hallucination benchmarks favor
  concise outputs."*
- **Finding (3) is a mechanism for Finding 1** that this project does not have: image-token contribution
  decays along the reasoning chain, so longer chains are structurally worse for perception.
- **What it does NOT measure:** cost. "4B + CoT beats 32B" is an **accuracy** claim, not a compute claim.

**Ahmadpour, Meighani, Taebi, Ghahroodi, Izadi & Soleymani Baghshah. *Limits and Gains of Test-Time Scaling
in Vision-Language Reasoning.* arXiv:2512.11109, 2025.** `[V]`
- Qwen2.5-VL, InternVL2.5-8B, Mulberry-8B, Gemini 2.0 Flash, GPT-4o mini, Claude-3-Haiku on MathVista,
  MMMU, MMBench. **Measured:** best-of-N with external verification, Qwen MathVista **68.25% → 75.36%**;
  self-refinement helps closed models (Gemini **80.09% → 89.57%**) but **degrades open-source** ones;
  MMBench perception shows *"minimal or no improvement."*
- Conclusion: *"external verification provides the most reliable gains, whereas iterative refinement often
  degrades performance."*

**Baxevanakis & Yang. *Test-Time Scaling for Small VLMs on Multilingual Visual MCQ.* arXiv:2607.09438,
2026.** `[V*]` — EXAMS-V / ImageCLEF 2026, **not medical**. The scaling shape is striking and directly
relevant: raising the **token limit** 1k→2k gave **+3.7 pp**, while going from **8 to 16 sampled chains**
gave **+0.15 pp**. ⇒ For small VLMs on MCQ, *sequential* compute dominates *parallel* compute. See §8 S12
for the tension with this project's Finding 1.

**Li, Hao & Liu. *Inference-Time Agentic Decision Rules Beat Longer Evolving Search for Multi-Image Medical
Reasoning.* arXiv:2607.27564, 2026.** `[V*]` — MedFrameQA, frozen splits (1,331/665/855). Order-vote
**57.89 ± 0.65%** vs fixed baseline **52.73 ± 0.42%**; and **doubling the search budget (50→100
generations) made final-test accuracy DROP 57.89 → 56.02.** Their conclusion — the decision rule matters
more than the search budget — is this project's selection-wall-beats-more-sampling result, in a medical
multi-image setting.

**Diao et al. *Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning
Optimization.* arXiv:2601.04442, ACL 2026.** `[V]` — a **learned meta-reasoning controller** routing each
generation step among a fast path, a *slow perception* path (re-examine the image) and a *slow reasoning*
path, trained by multi-objective RL on ~790k samples with teacher-generated failure-attribution labels
(perceptual hallucination vs reasoning error). Their framing — reasoning failures come from poor
perception, not shallow reasoning — is this project's Finding 1 turned into a training objective.
⚠ Per-benchmark numbers not extracted; the "substantially improves accuracy and efficiency" claim is `[S]`.

**Rui et al. *AdaThink-Med.* arXiv:2509.24560 (rev. 2026-08-02).** `[V*]` — uncertainty-guided RL for
adaptive reasoning length; **4.7×–6.4× token reduction** across six medical benchmarks with minimal
accuracy loss, and *emergent* "thinking"/"non-thinking" modes **with no external router**. **Text-only
medical LLM.** This is the single-model alternative to a two-model cascade and should be cited as such —
a reviewer will ask why a router is needed if a model can learn to allocate its own compute.

### 1.3 Explicit negative and limit results

**Zhao, Hooi & Ng. *Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks
Yet.* arXiv:2509.06861.** `[V]` *(venue: listed at an ICLR 2026 workshop "Agentic AI in the Wild"; the
arXiv page also indicates COLM 2026. **Unresolved — cite the arXiv id.**)*
- 14 reasoning models; more test-time compute *"does not consistently improve accuracy and often leads to
  more hallucinations"*; pattern consistent with **confirmation bias**, extended reasoning reinforcing
  early incorrect beliefs with fabricated details.
- **Theoretical statement, verbatim:** *"compute-only test-time scaling, as a post-processing procedure of
  a fixed model, cannot introduce new information about the ground-truth answer."*
- ⚠ **The two benchmarks and the 14 model names are NOT verified** — do not attribute datasets to it.
- ⚠ **One of its mechanisms is abstention** (fewer hallucinations because models declined to answer).
  **CRITICAL RULE 6: closed to us.** It explains *their* numbers; it is not a lever we may adopt.

**Hariri et al. (14 authors). *Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and
Reproducibility.* arXiv:2608.04001, 2026-08-04.** `[V]` — three regimes: single-trajectory sequential;
**leaf-level scaling with terminal reduction** (sample completions, aggregate by voting/verification —
*our open arm*); prefix-level. **States without measuring:** *"if tasks are too easy, additional compute
has little effect; if they are uniformly difficult, scaling curves flatten because correct solutions are
rarely reachable."* Attributes non-monotone BoN to reward-model overoptimization **by citation** (Gao 2023,
Huang 2025, Khalaf 2025). Useful as a **protocol** reference. Worth adopting: *"even greedy decoding need
not be deterministic"* — separate **exact replay** from **distributional reproducibility**.

**Roberts et al. *Test-Time Scaling Makes Overtraining Compute-Optimal.* arXiv:2604.01411, 2026.** `[V]` —
**Train-to-Test (T²) scaling laws** over model size × training tokens × inference samples. Once inference
cost enters the budget, **optimal pretraining shifts far into the overtraining regime**; undertrained
models are a bad substrate for TTS. Second independent argument for a stronger, more heavily-trained cheap
leg.

**Prucs, Csutora, Antal & Marosi. *Compute-Accuracy Pareto Frontiers for Open-Source Reasoning LLMs.*
arXiv:2512.24776, 2025.** `[V-html]` — 19 models, 5 benchmarks, **FLOPs/query by KV-aware estimation
covering prefill and decode**, GQA/SwiGLU/MoE-aware. 1B–8B reach *"accuracy parity with 30B+ baselines
while incurring a lower compute cost"* via long CoT; **MoE dominates the frontier**; easy benchmarks
plateau below 10¹⁶ FLOPs; **97% of models spend more compute on incorrect traces than correct ones.** The
closest published template to this project's cost convention (§6).

**Byun, Park, Azizan & Chellappa. *Test-Time-Scaling for Zero-Shot Diagnosis with Visual-Language
Reasoning.* arXiv:2506.11166, 2025.** `[V-html]` — Llama-3.2-11B-Vision → Llama 1B/3B/8B or Med42-v2-8B,
**N=16 majority vote**, MedMNIST v2: Pneumonia AUC **0.821 vs 0.517**, Path **0.653 vs 0.544**, Retina
**0.705 vs 0.570**. ⚠ Three binary tasks with near-chance baselines. **Not comparable to MedEvalKit VQA**;
do not cite as evidence TTS works on medical VQA.

---

## 2. AREA 2 — Verification and selection limits

### 2.1 Is 0.78–0.81 a known regime? Yes.

This project, clean disjoint verifier, `sel_eff = (selected − greedy)/(oracle − greedy)`: incumbent
**0.775204**, best of ~20 architectures **0.810627**, per-set **0.850 / 0.762 / 0.723** (SLAKE-open /
VQA-RAD-open / PathVQA-open) (`headroom_percell_2026-08-10.json`) `[REPO]`.

| source | domain | single pass | with selection | oracle | conversion |
|---|---|---:|---:|---:|---:|
| Agentic boosting, arXiv:2605.14163 `[V]` | SWE-bench Verified, GPT-5.4 nano, k=8 | 67.0% | 76.4% | 79.0% | **78.3%** `[MINE]` |
| This project `[REPO]` | 3 open medical VQA cells, Lingshu-7B, N=8 | 0.4495 | 0.4853 | 0.6260 | **77.5%** |

Two domains, two model generations apart, the same number. And the boosting paper names our mechanism:
*"the remaining failures are mostly proposal-coverage failures, indicating shared blind spots that
stronger selection alone cannot close."* Compare this project's M3 (coverage wall ≈ 4.5× selection wall)
`[REPO]`. **Report 0.78–0.81 as a field constant, not a shortfall.**

### 2.2 Why verifiers cap — the 2026 explanations

**Khalaf, Verdun, Oesterling, Lakkaraju & Calmon. *Inference-Time Reward Hacking in Large Language Models.*
arXiv:2506.19248, NeurIPS 2025 (Spotlight).** `[V]` — **proves** the hacking pattern (reward rises then
declines with N) is *"an inevitable property of a broad class of inference-time mechanisms, including BoN
and BoP."* **This explains a measurement this project already has:** sel_eff falling monotonically
**1.000 → 0.914 → 0.841 → 0.770 (K=8) → 0.717 (K=16)** (retrospective §8.2 item 12) `[REPO]`. Not a LoRA
bug — the predicted shape.

**Di, Ji, Li, Zhao & Gu. *Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling.* ICLR
2026.** `[V]` — **proves** neither majority voting nor BoN scales desirably in k and N; proposes **BoM**:
restrict candidates to **high-frequency** responses, *then* take top-k by reward. Regret
O(ε_opt + √(ε_RM²·C*/k)) once N = Ω̃(C*); and **BoM's performance does not degrade as N increases.** The
most directly actionable theory result in the sweep — shortlist **S3**.

**Hu. *Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time Collaboration.*
arXiv:2607.17531, 2026.** `[V]` — independently invents almost exactly this project's decomposition:
oracle gap, signal fidelity, recoverable mass, verification-signal coverage, conditional selection quality,
harm to already-correct outputs. **Measured:** public-test verifier **+8.14 pp** at MCC 0.825 vs
generated-test verifier **+2.70 pp** at MCC 0.248 on the same pools (*selection gain tracks verifier
quality almost linearly*); MATH symbolic selector **+4.67 pp** over self-consistency; GPQA-Diamond only
**3.03% recoverable mass** with **87.54% answer-identical pools**. No medical/VLM experiments.
⇒ Our decomposition is defensible and citable — **and no longer novel. Cite him; do not claim priority.**

**Zhang, Hosseini, Bansal, Kazemi, Kumar & Agarwal. *Generative Verifiers.* arXiv:2408.15240, ICLR 2025.**
`[V]` — GenRM beats discriminative and DPO verifiers on best-of-N (**5%→45.3%** algorithmic, **73%→93.4%**
GSM8K). **All verifiable-answer text reasoning.** Retrospective §8.3 item 23 is this idea; §2.3 is why it
probably will not transfer.

**Saad-Falcon et al. *Shrinking the Generation-Verification Gap with Weak Verifiers.* arXiv:2506.18203,
NeurIPS 2025** `[V]` (Weaver, 87.7% avg with ≤70B verifiers) and **Lee, Ma, Zhao, Nair, Spector, Cohen &
Candès. *FUSE.* arXiv:2604.18547, 2026** `[V]` (unsupervised verifier ensembling with explicit
conditional-dependence control, on GPQA-Diamond / HLE / IMO Shortlist). ⚠ Both are verifier-*ensemble*
results and this project has already run all score fusions and six cross-family judges to a null. The one
untried ingredient is FUSE's **explicit conditional-dependence modelling** — and §2.4 explains why that is
worth at most 1–2 points.

**Kim et al. *Parallel Test-Time Scaling with Multi-Sequence Verifiers.* arXiv:2603.03417, 2026.** `[V*]` —
scores each candidate **conditioned on the full sampled set** rather than in isolation; +6% best-of-64 on
math, early-stopping at half the latency. ⚠ **This is this project's set-aware / DeepSets / attention-head
architecture, which came in n.s.** (`COMPARATIVE_VERIFIER_2026-08-05.md`: adding the set-aware head
*lowers* the fusion, 0.812670 → 0.807902) `[REPO]`. **KILL — rediscovery.** Recorded so it is recognised.

### 2.3 The medical-specific verification evidence — and it is bleak

**Jin, Zhao, Kang, Zhang & Li. *Verification Mirage: Mapping the Reliability Boundary of Self-Verification
in Medical VQA.* arXiv:2605.10850, 2026-05-11.** `[V-html]`, `[V*]` — **the most citable paper in this
sweep for this project's negative-results contribution.**
- **Measures:** six open-weight VLMs — **Qwen2.5-VL-7B-Instruct, Gemma-3, Phi-4-Multimodal-Instruct,
  MedGemma, HuatuoGPT-Vision, Lingshu** — × **five medical VQA datasets (VQA-RAD, PathVQA, SLAKE, PMC-VQA,
  MedXpertQA)** × seven task types. Decomposes verifier behaviour into **discrimination capability** and
  **agreement bias**.
- **Measured:** verifier error **>40%**, **FPR ≳60%**; differential diagnosis FPR **~95–100%**; a generator
  error raises the odds of verifier failure **57×** (p<0.001); cross-verification helps but does not fix it
  (FPR −12–20%, verifier error only −2–5%); four-turn loops **lock in 69.5–87.1%** of wrong answers,
  correcting only **2.2–3.8%**. They name a **"lazy verifier"** effect — verifiers under-attend to image
  evidence relative to generators.
- **On verifier scaling:** significant FPR reductions in some families (p<0.001), but **Lingshu is flat or
  mixed, no significant reduction, p = 0.782.**
- ⇒ **External confirmation of this project's Finding 3 and its cross-family-verifier negative, measured on
  this project's own model.** And their failure mode is *untrained self-verification* — which is the
  argument **for** this project's decision to train a verifier. Position on that.

**Zhang, Fang, Wang, Li, Dai & You. *Best-of-Evidence: Best-of-N Selection under Partial Verification.*
arXiv:2607.20950, 2026-07-23.** `[V-html]`
- **Measures:** VQA-Med (2,334), PathVQA (9,903), PMC-VQA (10,000), MedXpertQA-MM (2,000); **Qwen3-VL-30B**
  generator, **Qwen3-VL-235B** evidence judge, **K = 16**, budget C = 16.
- **Measured gains over raw majority: +0.26 to +0.58 pp.** Reported Oracle@K: VQA-Med 82.1%, PathVQA 61.5%,
  PMC-VQA 65.1%, MedXpertQA-MM 64.7%. *(Per-dataset pairing extracted from the HTML table — medium
  confidence on the pairing, high on the range.)*
- **Theory:** a **residual evidence-capacity limit**, E[improvement] ≤ √(m/2·Λ_C) with Λ_C the maximum
  mutual information obtainable within budget C.
- ⇒ **The state of the art in medical best-of-N selection buys <1 point with a 235B judge.** Also note their
  Oracle@16 on MedXpertQA-MM is 64.7% with a Qwen3-VL-30B generator, versus this project's Lingshu-7B
  MedXpert oracle@8 of **0.543** `[REPO]` — **a generator gap, not a selector gap.**

**Zhang et al. *Scaling Medical Reasoning Verification via Tool-Integrated Reinforcement Learning
(Med-TIV).* arXiv:2601.20221, 2026.** `[V]`, `[V*]` — iterative RL with trace-level supervision and
retrieval from medical corpora during verification; **+23.5% MedQA / +32.0% MedXpertQA** relative to the
base generator, **8× reduction in sampling budget** vs prior reward models. **Text-only medical.** The
multimodal version is unwritten. See §11 for how this bears on the RAG kill.

### 2.4 Ensembling: why cross-family verifiers under-deliver

**Bugaud. *Hidden Clones: Exposing and Fixing Family Bias in Vision-Language Model Ensembles.*
arXiv:2603.17111, 2026.** `[V]` — 17 VLMs / 8 families on VQAv2, TextVQA, GQA. Family-correlated errors
reduce **effective ensemble dimensionality to 2.5–3.6 independent voters**; on **1.5–6.5%** of questions
correlated majority errors drive accuracy to 0% despite the best model being correct. Hierarchical Family
Voting recovers **+18–26 pp** on that tier; a learned candidate scorer gains **+0.68 / +0.61 / +2.45%**
overall. ⇒ The *mechanism* behind this project's cross-family negative — and a caution: the overall fix is
worth ~1–2 points on general VQA, **below this project's macro CI half-width of 0.0029** `[REPO]`.

### 2.5 Process vs outcome reward models — structurally inapplicable here

The 2026 text/math consensus is that PRMs beat ORMs on best-of-N at 3–10× annotation cost `[S]`, with
DreamPRM (arXiv:2505.20241, NeurIPS 2025) and DreamPRM-1.5 (arXiv:2509.05542) extending to multimodal
reasoning `[S]`. **Unverified — do not quote.** More importantly, **PRMs score intermediate reasoning
steps**, and this project measured that reasoning hurts on 17/20 perception cells and that **0/9** explicit
reasoning-*trigger* effects were CI-significant once answer format was matched
(`medeval_matched_direct_2026-07-29.json`) `[REPO]`. **There are no steps to score. Do not pursue PRMs on
the closed cells.**

---

## 3. AREA 3 — Cascades and routing in 2026

### 3.1 Theory: cascades are structurally cost-disadvantaged

**Bouchard. *Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades.*
arXiv:2605.06350, 2026-05-07.** `[V]`, `[V*]` — **read this one.**
- **Measures:** MATH, MMLU, TriviaQA, SimpleQA, LiveCodeBench; 8 models, 5 providers.
- **Theory:** piecewise-concave cost-quality frontier with shadow prices; optimal k-model cascade
  **equalizes marginal quality-per-cost across stage boundaries**; the achievable k-model frontier equals
  **the envelope of all pairwise two-model cascades** (⇒ intermediate tiers buy less than assumed).
- **The quotable result:** *"A lightweight pre-generation router exceeds the best cascade policy on four of
  five datasets, mainly because it avoids the cheap model's generation cost … cascade performance is
  limited primarily by structural cost, since cascades pay the cheap model before any escalation
  decision."*
- ⇒ **This is the external explanation of this project's macro cost reversal, and it names the fix.**

**Mahmood. *Routing, Cascades, and User Choice for LLMs.* arXiv:2602.09902, 2026.** `[V]` — Stackelberg
model; **"in nearly all cases, the optimal routing policy involves a static policy with no cascading."**
⚠ Economics/mechanism-design, not an accuracy measurement. Use it to justify adding a static-routing arm,
not to retire the cascade.

### 3.2 The identifiability problem, measured — and three papers that formalise the ceiling

**Lu, Zhang, Zhang, Yu, Wang, Chen & Xing. *The Routing Plateau: Understanding and Breaking the Accuracy
Limits of LLM Routers.* arXiv:2606.07587, 2026.** `[V]` — **21 routing methods on 5 datasets.** Many
methods, *including k-NN*, "converge to a narrow performance range that remains far below the oracle
router." Named cause, the **predictability bottleneck**: routers "mainly learn global averaged
model-performance trends rather than fine-grained query-specific routing signals"; they "solve overlapping
easy queries but collectively fail on hard queries." Remedies proposed: larger training sets, stronger
encoders, end-to-end fine-tuning, **model-pool-aware objectives that compare models jointly**, lookahead
signals from partial generation.
⚠ **The numeric plateau value is NOT in the abstract and I did not read the PDF.** Do not quote a number.
⇒ This project's ~16 mechanisms at 0.5–0.6 AUROC **is** the routing plateau, reproduced at scale.

**Pona, Kazemi, Hosseini, Du, Watson, Simeone & Paoletti. *Calibrate-Then-Delegate.* arXiv:2604.14251,
2026.** `[V*]` — **the paper that builds exactly this project's probe.** Its motivating sentence is this
project's finding, stated as their premise: *"Existing cascades delegate based on probe uncertainty, but
uncertainty is a poor proxy for delegation benefit, as it ignores whether the expert would actually correct
the error."* They train a **delegation-value probe** on hidden representations to predict escalation
benefit directly. **Measured (full text):** MSE 0.17 / 0.21, **Spearman 0.27 (strong expert) / 0.49 (weak)**
— a logistic probe on Llama-3.2-1B layer 11, Gemma-3-27B expert, four safety datasets. **And it still beats
uncertainty-based delegation at every budget (+7% AUC, +9% accuracy.)**
⇒ **Two consequences.** (i) A Spearman of 0.27 is in this project's AUROC band — **the wall is confirmed,
not broken.** (ii) **A weak recoverability signal still beats a strong detection signal for routing** — an
external confirmation of retrospective §5.6, and a reframe this project can use.

**Chen (J.). *When Does Combining Language Models Help? A Co-Failure Ceiling.* arXiv:2606.27288, 2026.**
`[V*]` — 67 frontier models, 21 providers. *"Accuracy cannot exceed one minus beta, where beta is the rate
at which every model is wrong on the same query,"* and *"pairwise error correlation cannot capture this
ceiling."* Measured co-wrong rates: math **β = 0.052** observed vs 0.023 predicted (**2.5×
underestimation**), code 0.079, and **GPQA-Diamond β rises to 0.127 when switching from multiple-choice to
free-response.** ⇒ This is this project's **coverage wall** in another literature — and that last number is
an independent measurement in the neighbourhood of this project's **Finding 2** (answer format changes
everything). The author explicitly did **not** train a router and says whether the gain is capturable
*"is open."*

**Chen (T.-R.). *How Much of the Routing Gap Is Real?* arXiv:2607.03436, 2026.** `[V*]` — LLMRouterBench,
33 models, **391,645 instances**: *"12–36% of the reported router-to-oracle gap is single-draw label noise
that no single-commit router can capture,"* approaching **half** on the hardest queries; the majority is
genuine recoverable specialist advantage. ⇒ **Directly actionable and defensive:** a meaningful slice of
any oracle gap this project reports is stochastic and structurally unharvestable. Shortlist **S1**.

**Sun & Yang. *RouteGuard.* arXiv:2608.07583, 2026-08-05 (six days old).** `[V*]` — challenges *"the
assumption that advisor complementarity and gate AUC optimization suffice"*; certifies routing gain as
G = π·ΔE governed by a conditional-regret functional **rather than AUC**, with a Le Cam lower bound.
**The killer empirical result:** on RouterBench, *"gains appear under prompt-level sampling but disappear
under workload-cluster resampling due to concentration in 3 of 86 cells"*; on OpenRCA the oracle *"matched
or fell below independence"* and certification was withheld. ⇒ **This is this project's PathVQA-open
concentration problem, stated as a general certification failure mode.** It is the external citation for
"our gain is concentrated in a few cells and we say so," and it supplies a **test** — shortlist **S4**.

**Zeng, Wang, Chen & Lin. *ReLope: KL-Regularized LoRA Probes for Multimodal LLM Routing.*
arXiv:2603.24787, 2026.** `[V*]` — appears to beat the wall and does not, because its target is
**detection**, not recoverability. Qwen2.5-VL-7B avg AUC (MMMU / A-OKVQA / ScienceQA / ChartQA /
MathVision): baseline probe 83.06 → attention probe 84.44 → ReLope **88.21**; +9.48 on Gemma3-12B, +9.32 on
Phi-4-MM. **The finding this project should actually cite is orthogonal and quantifies Finding 2:**
*"the presence of visual inputs weakens the separability of correctness signals in hidden states"* —
ScienceQA **text-only probe AUC 95.36 vs 83.95 on the same benchmark's multimodal inputs, an 11.41-point
drop from adding images.** That is a clean external number for "multimodal routing signals are structurally
worse," which this project currently has no external anchor for.

**Varshney, Surla, Xu, Krishnan, Jeblick, Austin, Vaidya & Onofrio. *LLM Router: Rethinking Routing with
Prefill Activations.* arXiv:2603.20895, 2026.** `[V]` — route on **prefill activations**, layer chosen by
a **Fisher-separability** probe, with a **SharedTrunkNet predicting per-model correctness jointly**.
Reports **45.58%** of the strongest-standalone-to-oracle gap closed at **74.31%** cost saving.
⚠ Benchmarks and model pool not verified; re-read the PDF before citing the numbers.

**Warren & Dras. *Bi-directional Model Cascading with Proxy Confidence.* arXiv:2504.19391, 2025.** `[V*]` —
estimates the **large** model's pre-invocation confidence with a *tiny proxy model*, so the deferral rule
considers **both legs**. Evaluated on **multiple-choice** datasets. ⇒ **All 16 of this project's
recoverability mechanisms are small-model-side.** This is a structurally different, untried mechanism.

**Lugoloobi, Foster, Bankes & Russell. *LLMs Encode Their Failures.* arXiv:2602.09924, ICLR 2026 LIT
Workshop.** `[V*]` — linear probes on **pre-generation** activations route across a model pool to *"exceed
the best-performing model whilst reducing inference cost by up to 70% on MATH."* Per-model success
prediction; no AUROC in the abstract.

### 3.3 Applied cascade systems, 2026 — and the bar they publish at

- **Moslem et al. *Cluster, Route, Escalate.* arXiv:2606.27457, 2026.** `[V]` — cluster + route, then
  quality-estimation escalation. **Retains 97–99% of the strongest model's accuracy** at reduced TPOT.
  ⚠ *Retains*, not beats.
- **Dwivedi et al. *SAFE-Cascade.* arXiv:2606.19646, CIKM 2026 demo.** `[V*]` — chart QA; structurally this
  project's exact system (cheap text path → learned router → escalate to VLM). **69.1% vs 67.7%
  full-VLM at 73.1% VLM invocation on n=375**, and they explicitly say the +1.4 pp is statistically
  uncertain, claiming **parity at 26.9% fewer VLM calls but only 9.3% cost saving**. ⇒ **The gap between
  "26.9% fewer calls" and "9.3% cheaper" is this project's cost-accounting trap, published.** And it shows
  this project's evidentiary bar (n=42,224, CIs, decontamination) is far above the niche's norm.
- **Kotte. *UCCI.* arXiv:2605.18796, 2026.** `[V*]` — 75,000 production NER queries, 4B/12B: **−31% cost
  [95% CI 27–35%] at micro-F1 0.91**, ECE 0.12→0.03, **no accuracy claim at all**. ⇒ *"Matched quality,
  less cost, with a confidence interval"* is the field's current publishable shape.
- **Guo, Wu & Yiu. *RouteNLP.* arXiv:2604.23577, 2026.** `[V]` — router + conformal cascade thresholds +
  a **distillation-routing feedback loop** (distil on escalation failures, retrain the router). 8-week
  enterprise pilot: **58% cost reduction, 91% acceptance, P99 1,847 → 387 ms**; six-task benchmark
  **40–85% cost reduction**; **">2× the cost improvement" of untargeted distillation.** ⚠ Enterprise pilot,
  no public reproduction — directional.
- **Luo et al. *RouteLMT.* arXiv:2604.22520, ACL 2026 Industry Track.** `[V*]` — independently arrives at
  this project's framing: existing routers *"fail to capture whether the large model actually provides a
  worthwhile improvement over the small one,"* so they *"identify **marginal gain** … as the optimal signal
  for budgeted decisions."* ⚠ Machine translation (continuous quality metric — a far better-conditioned
  target than binary MCQ recoverability); Pareto-frontier win, **no AUROC**; they concede *"regression
  risks."*
- **Moslem & Kelleher. *Dynamic Model Routing and Cascading: A Survey.* arXiv:2603.04445, 2026.** `[V]`,
  `[V*]` — the canonical survey. Its own words on this project's modality: ***"multimodal routing remains
  underexplored compared to text-only settings."*** Of the methods it catalogs, only **Self-REF**
  (confidence-token fine-tuning) touches training the small model; **AutoMix** is explicitly *"solely
  few-shot prompting without fine-tuning"*; FrugalGPT and Cascade Routing use frozen models. **"Train the
  cheap leg" is NOT the field's default** — which is why S7 below is a real gap.
- **Liu, Zeng, Chang & Lin. *Forced Deferral: Manipulating Routing Decisions in Multimodal LLM Cascades.*
  arXiv:2606.15308, 2026.** `[V*]` — an adversarial **image** trigger that flattens the weak model's
  confidence and forces escalation, across datasets, families and deferral metrics. **Multimodal cascades
  specifically.** ⇒ A robustness caveat this project must acknowledge if it ever claims clinical
  deployability of a confidence gate.
- **Medical routing that exists but is not compute routing:** MedRoute (arXiv:2604.06180) `[V*]` — RL
  router selecting a *specialist*, not a cost tier; One VLM, Two Roles (arXiv:2508.16839) `[V]`, `[V*]` —
  modality → abnormality → model-card routing, +9/+11 pp; RVLM (arXiv:2603.24224) `[V*]` — an adaptive
  iteration-budget controller, but on BraTS/MIMIC-CXR with Gemini 2.5 Flash, no cascade, no VQA benchmarks.
- ⛔ **Abstention-based clinical triage exists and is out of scope by CRITICAL RULE 6.** See §11.

### 3.4 ⚠ False friend

**Nemotron-Cascade 2 (arXiv:2603.19220, NVIDIA)** `[V*]` is **not** an inference cascade. "Cascade RL"
there is a **training curriculum** over reasoning domains on a 30B-A3B MoE. Nothing about routing between
deployed models. Likewise **LLaVA-CKD (arXiv:2605.10641)** `[V*]` uses "cascade" to mean a *distillation
ladder*. Do not cite either as cascade prior art.

---

## 4. AREA 4 — Adapting/training the cheap model

### 4.1 The decomposition that answers the question

**Rokon, Desai, Yao & Lee. *AutoRelAnnotator: Calibrated Model Cascades for Cost-Efficient Relevance
Evaluation in Sponsored Search.* arXiv:2606.25871, SIGIR 2026 e-commerce workshop.** `[V*]` — the cleanest
published statement of the training-vs-cascading question, verbatim: ***"fine-tuning contributes 20
accuracy points while cascading is approximately accuracy-neutral but halves compute cost."*** 150M+
production annotations; isotonic calibration adds +0.6. Domain is sponsored-search relevance, so this
transfers as a **structural** claim: **the two levers are orthogonal and do not substitute.**

### 4.2 Cascade-aware training

**Wang, Augenstein, Rush, Jitkrittum, Narasimhan, Rawat, Menon & Go. *Cascade-Aware Training of Language
Models.* arXiv:2406.00060, 2024.** `[V]`, `[V*]`
- *"Training the small LM with awareness of its place in a cascade and downstream capabilities."*
  Identifies **two** channels: (i) the small model's **accuracy**, and (ii) the small model's **confidence
  calibration** — *which is what the deferral rule thresholds*. PaLM-2 Gecko/Otter, 60+ tasks from
  SuperGLUE, WMT22, FLAN2021.
- **Full text `[V*]`:** *"CAT-Xent reduces 50% FLOPs given fixed 86% accuracy requests"*; at a matched
  2B-FLOP budget, *"CAT-Xent gets 2% accuracy improvement."* Note the shape: a **cost** gain at matched
  quality — the same shape as this project's compute-lean point — via a **token-level loss modification**,
  not distillation.

**Rabanser, Rauschmayr, Kulshrestha, Poklukar, Jitkrittum, Augenstein, Wang & Tombari. *Gatekeeper:
Improving Model Cascades Through Confidence Tuning.* arXiv:2502.19335, ICML 2025 TTODLer-FM workshop.**
`[V*]` — a loss that *"fine-tunes the smaller model to confidently handle tasks it can perform correctly
while deferring complex tasks to the larger model,"* evaluated across encoder-only, decoder-only,
encoder-decoder, and **vision-language** tasks. ⚠ **Magnitudes unverified** (OpenReview PDF behind a bot
challenge).

⚠ **Framing discipline for both:** these are cascade-aware training for two-model systems that **always
answer**. They must never be written up with learning-to-defer / reject-option vocabulary (CRITICAL RULE 6).

### 4.3 Distillation — including a 7B = 32B result

- **Ko, Abdali, Kim, Chen & Cameron. *REOPOLD: Scaling Reasoning Efficiently via Relaxed On-Policy
  Distillation.* arXiv:2603.11137, 2026.** `[V*]` — *"enables a **7B student to match a 32B teacher in
  visual reasoning** with a ~3.32× inference speedup,"* 6.7–12× sample efficiency over recent RL.
  ⚠ **Abstract-level; the benchmark carrying the 7B=32B claim is not named. Pull the PDF before building
  an argument on it.** This is the "why didn't you just distil?" question in one sentence.
- **Lyu, Wang, Huang & Xu. *SCoRe: Student-Centered Distillation.* arXiv:2509.14257, ICML 2026.** `[V*]` —
  *"On 12 challenging benchmarks, a 7B-parameter student distilled with SCoRe closes the agentic
  performance gap with a 72B-parameter teacher."* Mechanism is relevant to the open arm: the student
  generates trajectories, the teacher **corrects only the earliest error**, then short-horizon RL restarts
  from the verified prefix.
- **Wu, Wu, Tao, Li & Sarwate. *Inter-Cascade: From Deferral to Learning — Online In-Context Knowledge
  Distillation for LLM Cascades.* arXiv:2509.22984, 2026.** `[V*]` — on each deferral the strong model
  writes a **reusable strategy** into a repository that later augments the weak model's context.
  **Measured: weak-model accuracy up to +33.06%, system accuracy +6.35%, strong-model calls −48.05%, cost
  −49.63% — with ZERO parameter updates.**
- **Kang, Aljundi, Dorovatas & Alahari. *Online In-Context Distillation for Low-Resource Vision Language
  Models (ICD).* arXiv:2510.18117, 2026.** `[V*]` — the **VLM** sibling. *"Boosts the performance of small
  models (up to 33%) using scarce teacher annotations (as low as 4%), and competes with the teacher's
  zero-shot performance,"* with **student-uncertainty conditioning to minimise teacher queries** — i.e. a
  gate, in this project's sense. **Arguably the most directly adjacent method to this project's cascade.**
- **Feng et al. *Direct-OPD: Weak-to-Strong Generalization via Direct On-Policy Distillation.*
  arXiv:2607.05394, 2026.** `[V*]` — Qwen3-1.7B **48.3% → 58.3% on AIME 2024** in 4 h on 8×A100.
- **Liu et al. *VA-OPD: Visual-Advantage On-Policy Distillation.* arXiv:2605.21924, 2026.** `[V*]` — the
  diagnosis that should worry this project: *"standard on-policy distillation can improve a student's
  output quality while **failing to strengthen its reliance on visual input**."* Qwen3-VL, teachers at 4B /
  8B / 32B, 8 benchmarks; gains grow monotonically with teacher size. **Med-OPD (arXiv:2607.16303)** `[V*]`
  is the medical version, weighting tokens by their dependence on visual evidence, evaluated on OmniMedVQA
  subsets.
- **Song & Zheng. *A Survey of On-Policy Distillation for LLMs.* arXiv:2604.00626, 2026.** `[V*]` — the
  orientation read; static imitation's exposure bias *"scales roughly with the square of sequence length"*.

### 4.4 Medical VLM adaptation — the gains are large, real, and cell-dependent

**Chen, Shi, Le, Yin, Lin, Ni, Gong & Li. *Why Does Grounding Hurt Medical VQA? Benchmarking, Diagnosis,
and Fine-Tuning of Vision-Language Models.* arXiv:2604.27720, 2026-04-30, **v2 2026-07-28 (retitled** — v1
was *"Auditing Frontier Vision-Language Models for Trustworthy Medical VQA…"*; cite the v2 title).** `[V]`,
`[V*]` — the most useful medical-adaptation artifact found, because it reports per-benchmark and includes a
regression. Qwen2.5-VL-7B, answer-only SFT vs zero-shot (full text `[V*]`):

| | zero-shot open / closed | answer-only SFT | best arm |
|---|---|---|---|
| **SLAKE** (n=1,061) | 37.2 / 64.6 | **80.0 / 77.1** | 83.0 / 79.7 |
| **VQA-RAD** (n=451) | 38.1 / 66.2 | **29.0 / 66.6** | 29.7 / 67.2 |

**Domain SFT of a 7B moves SLAKE open-ended +42.8 points — and moves VQA-RAD open-ended BACKWARDS by 9.1.**
Frontier zero-shot SLAKE closed is 58.1–80.1%, so the fine-tuned 7B lands *inside* the frontier band.
⇒ **Adaptation has exactly the per-cell heterogeneity problem that motivated this project's macro
averaging, seen from the training side.** This is the honest answer to "why not just fine-tune."

Also from this paper `[V]`: every off-the-shelf system scores **mean IoU 0.05–0.24** on SLAKE grounding
against a trivial baseline of 0.10; SFT of Qwen-2.5-VL-7B produced **0/418 parseable boxes** ("format
collapse"); and **cropping to perfect oracle bounding boxes LOWERED closed-ended accuracy by 0.9–18.0
points** across all models including Lingshu. ⇒ **A direct hazard for retrospective §8.3 item 22** — see S12.

Other verified medical adaptation results `[V*]`: **LiteMedCoT-VL** (arXiv:2605.09384) — LoRA CoT
distillation from a 235B teacher into a **2B** student, PMC-VQA **64.9%** vs Qwen3-VL-2B 48.7% and
Qwen3-VL-**4B** 53.9% ⚠ *on a 2,000-row test split whose file is never named — the `test_clean` shape, not
MedEvalKit's `test_2`; do not cross-compare*; **MedQwen / Sparse Spectral LoRA** (arXiv:2604.01310) —
*approaches* full FT with 339× fewer trainable params over 23 medical datasets, forgetting ~5% vs 20–50%;
**OpenMedQ** (arXiv:2606.12953, MIDL 2026) — a **7B** beating Med-PaLM M up to 562B on PathVQA **BLEU-1
75.9** (a generous metric, not accuracy); **OpenMedReason** (arXiv:2606.12169) — ~450K reasoning traces,
**+20% average VQA accuracy**, within 4.2% of comparable-scale medical LVLMs; **Zhu et al.**
(arXiv:2505.13973) — GRPO-based RL fine-tuning *"consistently outperforms standard SFT."*

**Verdict:** LoRA/SFT on a 7B medical VLM buys 10–40+ points in-domain, routinely enough to reach or pass
much larger general models — but (a) it can *regress* a benchmark, (b) standard distillation improves text
quality without improving **visual grounding** (VA-OPD and Med-OPD both diagnose this), and (c) **nobody
has published "LoRA-tuned Lingshu-7B = Lingshu-32B" on this suite.** The gap is benchmark-specific and
adaptation is not a uniform eraser.

### 4.5 Speculative decoding — the field's default cost answer, and it does not answer this question

Verified VLM speedups `[V*]`: **SpecVLM** (arXiv:2509.11815) **2.5–2.9×** end-to-end on LLaVA/MMMU (its
EAGLE-2-style baseline 1.5–2.3×); **Spec-LLaVA** (arXiv:2509.11961, ICML TTODLer-FM) up to **3.28×**;
**ViSpec** (arXiv:2509.15235, NeurIPS 2025) notes prior VLM methods were *"<1.5×"*; **Spec-VLA**
(arXiv:2507.22424, EMNLP 2025 main) **1.42×**, +44% acceptance length, success rate preserved; **HiViS**
(arXiv:2509.23928).

**The reviewer risk, and the rebuttal.** LongSpec (arXiv:2502.17421, **ACL 2025 main**) `[V*]` states the
framing the field has adopted, verbatim: *"Speculative decoding (SD) offers a promising lossless
acceleration technique compared to **lossy alternatives such as quantization and model cascades**."*
**The counter is structural and belongs in the paper in one sentence:** speculative decoding **runs the 32B
on every single query** — the target model does all the prefill and all the verification; the draft model
only shortens the *decode* serial chain. In a VLM the prefill is dominated by visual tokens (the *"visual
memory wall"* framing of the ACL 2026 Findings survey, arXiv:2604.05546 `[V*]`). So spec-decoding delivers
**latency, not FLOPs**, and **zero** accuracy change by construction; and against a *reasoning* baseline it
can accelerate the tokens but not remove them.

**No speculative-decoding paper for medical VLMs exists** — searched twice, a genuine gap. ⇒ **This
resolves retrospective §8.3 item 25's "weakened" verdict as under-argued** (see §10, A8).

---

## 5. AREA 5 — Medical VLM state of the art, 2026

### 5.1 The reference table, confirmed twice

The Lingshu benchmark table was confirmed independently from the arXiv HTML (**Table 6**) and the HF model
cards, which agree exactly `[V*]`:

| Model | MMMU-Med | VQA-RAD | SLAKE | PathVQA | PMC-VQA | OmniMedVQA | MedXpertQA-MM | Avg |
|---|---|---|---|---|---|---|---|---|
| Lingshu-7B | 54.0 | 67.9 | 83.1 | 61.9 | 56.3 | 82.9 | 26.7 | 61.8 |
| Lingshu-32B | 62.3 | 76.5 | 89.2 | 65.9 | 57.9 | 83.4 | 30.9 | 66.6 |
| **Lingshu-I-8B** | 49.1 | 73.0 | 91.6 | **74.9** | 55.8 | 79.7 | 27.5 | **64.5** |
| GPT-4.1 | 75.2 | 65.0 | 72.2 | 55.5 | 55.2 | 75.5 | 45.2 | — |
| HuatuoGPT-V-7B | 47.3 | 67.0 | 67.8 | 48.0 | 53.3 | 74.2 | 21.6 | — |

> ### ⚠ NEW MODEL NOT IN `CLAUDE.md`: **`Lingshu-I-8B`**
> InternVL3-based, run through the **same Lingshu training pipeline**, HF org page updated **2026-02-24**
> `[V*]`. Average **64.5** vs Lingshu-7B's 61.8 — **PathVQA +13.0, SLAKE +8.5, VQA-RAD +5.1**, but
> **MMMU-Med −4.9**. PathVQA-open is this project's load-bearing cell. **This is a protocol-matched,
> same-family, zero-training stronger cheap leg, available now.** See shortlist **S6**.
> *(The MMMU-Med drop is also an interesting cross-check on the MMMU contamination story: a model from the
> same pipeline on a different backbone loses 4.9 points there while gaining everywhere else.)*

**MedGemma-27B has no published medical-VQA numbers on this benchmark set** `[V*]`. The MedGemma technical
report (arXiv:2507.05201, rev. 2026-04-06) reports medical VQA only in **Table 9, as token F1** (MedGemma
4B: 72.3 SLAKE / 49.9 VQA-RAD), **does not evaluate the 27B on medical VQA at all**, and excludes PathVQA.
⇒ Any MedGemma-27B PathVQA number in this repo is from this project's own runs, with **no external number
to check it against.** State that.

### 5.2 Is Lingshu still a sensible anchor? Yes — with independent April-2026 evidence.

**Bao et al. *MedRCube.* arXiv:2604.13756, 2026-04-15.** `[V*]` — an independent fine-grained evaluation
framework whose abstract states verbatim: *"We benchmark 33 MLLMs, **Lingshu-32B** achieve top-tier
performance."* An outside group, ten months after release, 33 models, still placing Lingshu-32B at the top.
**This is the single best answer to the anchor question.** Supporting it: Verification Mirage picks Lingshu
as one of six *representative* open medical VLMs — i.e. Lingshu is now a standard model, not a novelty.

**The one credible challenger is Fleming-VL-38B** (arXiv:2511.00916) `[V*]` — open weights, Apache 2.0,
InternVL3-38B base. Its Table 3 reports Fleming-VL-38B at OmniMedVQA 87.9 / PMC-VQA 76.5 / PathVQA 68.0 /
SLAKE 89.8 / VQA-RAD 76.6, above Lingshu-32B on all five.
⚠ **Do not treat this as a protocol-matched comparison.** The Lingshu rows in Fleming's table are
byte-for-byte the numbers from Lingshu's own paper (so copied, not re-run); the paper **never states it uses
MedEvalKit**; it **never states its PMC-VQA split**; and it reports **neither MMMU-Med nor MedXpertQA**.
Fleming-VL-8B at PMC-VQA 64.3 against Lingshu-7B's 56.3 is an implausible protocol-matched jump and is far
more likely an in-domain-training or split effect.

**Recommendation: keep Lingshu-7B/32B as the anchor; cite MedRCube (arXiv:2604.13756) as independent
mid-2026 corroboration; and change the adjective** from "SOTA" to *"a strong, faithfully reproducible open
medical VLM family with a matched 7B/32B pair"* — which is the property the method actually needs. If a
reviewer asks "why not a 2026 model," the honest answer is that the 2026 models claiming to beat it do so
**without a matched evaluation harness**, which is exactly this project's methodological point. Note also
that **MedGemma has moved *away* from this benchmark axis** (1.5 is 4B-only, on 3D/WSI/localization), making
it a weaker anchor in 2026 than in 2025.

**The bigger threat is generational, not medical.** Sammani et al. `[V-html]` measure **Qwen3-VL**
2B/4B/8B/32B and **InternVL-3.5** 2B/4B/8B/38B, and find a **4B** of the current generation with plain CoT
beating a **32B** of the *same* generation on 3 of 6 benchmarks. If a current-generation small model closes
most of the 7B→32B gap unaided, the cascade's job shrinks.

**Other 2026 entrants** `[V*]`, none of which displaces the anchor: **MedGPT-oss** (arXiv:2603.00842, 20B,
**no benchmark numbers in the abstract at all**); **Aloe-Vision** (arXiv:2606.27500, 7B/72B, open recipes +
CareQA-Vision, no Lingshu/MedEvalKit comparison); **MediX-R1** (arXiv:2602.23363, abstract names no
benchmarks); **MMedExpert-R1** (arXiv:2601.10949 — the only one reporting on this suite in its abstract:
7B, **MedXpert-MM 27.50, OmniMedVQA 83.03**, i.e. ties Lingshu-7B and does **not** reach Lingshu-32B on
MedXpert); **MedMO** (arXiv:2602.06965, reports only *relative* gains, never mentions VQA-RAD/SLAKE/
PathVQA/OmniMedVQA/Lingshu/MedEvalKit); **Hulu-Med** (arXiv:2510.08668); **OctoMed** (arXiv:2511.23269);
**ClinFusion** (arXiv:2607.24743); **Citrus-V** (arXiv:2509.19090 — the "beats Lingshu-32B by 0.59%" claim
is `[S]`, **unconfirmed on the abs page**, and 0.59% is inside noise anyway).

⚠ **Ignore third-party leaderboards.** A MedXpertQA "leaderboard" (llm-stats.com, "June 2026") shows scores
of 0.784 / 0.673 that are wildly out of line with every peer-reviewed MedXpertQA-**MM** number (21–45).
Almost certainly the text subset and/or a different scoring rule. **Do not cite.**

### 5.3 Has anyone published a cascade / TTS result on medical VQA? Yes — three, none previously known here.

See §0.4. In addition: **no published small→large medical VLM cascade with compute accounting exists.**
That is the residual novelty, and it is now narrow: the novelty is the **cascade + format router + cost
accounting on eight cells**, not best-of-N-with-a-verifier on medical VQA, which arXiv:2605.18313 has taken.

### 5.4 PMC-VQA split quality — the gap is real and the paper is unwritten

**No paper, 2025 or 2026, examines the PMC-VQA `test_2` / `test_clean` distinction.** Searched from six
angles; genuinely absent. `PMCVQA_PROVENANCE_2026-07-30.md` appears to be the only document anywhere that
treats it. What *is* confirmed around it:

- **The split structure, from the official HF dataset card (`xmcmic/PMC-VQA`)** `[V*]`: v1 ships
  `train.csv`, `test.csv`, `test_clean.csv`; v2 ships `train_2.csv`, `test_2.csv`, `images_2.zip`
  (described as the non-compound-image version). **The card gives no row counts and makes no
  human-verification claim for the v2 files** — the verified-split language attaches only to v1's
  `test_clean`. There is currently a **dataset-viewer generation error caused by column mismatches between
  `train_2.csv` and the other files**, i.e. the v2 files are not even schema-consistent with v1.
  **That is a concrete, citable data point supporting this project's landmine.**
- **Option-position imbalance, from the original PMC-VQA paper (arXiv:2305.10415)** `[S]` — training set
  options B+C = **73.5%** (35.6 + 37.8), A and D each under 14%; test set B+C = **62.4%** (31.9 + 30.5),
  A 21.9%, D 15.8%; the authors themselves note this *"introduces an answer-position bias during
  fine-tuning."* They also flag academic-figure distribution bias (curated illustrative images with arrows
  and annotations) making the data easier than clinical practice. ⚠ **Read via search extraction from the
  HTML, not confirmed on the abs page, and the split these percentages describe is not stated.**
  ⇒ **✅ SETTLED LOCALLY DURING THIS SWEEP — see §0.6.** Measured on disk: `test_clean.csv` B+C = **62.4%**
  (reproducing the paper's reported *test* distribution exactly), `test_2.csv` B+C = **73.6%** (matching
  the paper's reported *train* distribution). The MedEvalKit-track split carries the training split's
  answer-position profile, and a constant-C guesser scores **37.8%** on it. `[MINE]`
- **MedGemma is on the record about two of this project's other cells** `[V*]`, verbatim: *"we and others
  have identified potential data quality issues in PathVQA and MedVQA. Thus, we removed them from the
  training dataset,"* and *"For VQA-RAD, we used splits from yang2024advancing to avoid the train/test
  image contamination present in the original splits."* **Google DeepMind, in an April-2026-revised
  technical report, saying PathVQA has data-quality problems and VQA-RAD's original splits are
  contaminated.** This is the strongest external support for benchmark skepticism in this suite. It does
  **not** mention PMC-VQA.
- **Xu, Wu & Ryu. *A Controlled Audit of Pretraining Contamination in Public Medical Vision–Language
  Benchmarks.* arXiv:2606.10066, 2026-06-08.** `[V-html]` — InternVL3-8B, Qwen2.5-VL-7B-Instruct,
  CheXagent-8b, LLaVA-OneVision-Qwen2-7B (+MedGemma-4B-IT on OmniMedVQA) over SLAKE-En (1,061), PathVQA
  (6,719), VQA-RAD (451), OmniMedVQA (4,999). **Measured: 19.8% of SLAKE images have an extreme same-view
  near-neighbour** in PMC-OA-beta under SigLIP-B-16 (**4.2%** under SO400M) against 0/2000 for
  out-of-domain controls; **VQA-RAD ≤0.9% and clean across all detectors**; text-side exchangeability fired
  for Qwen2.5-VL × SLAKE-En (p = 5.0×10⁻⁴) but the cohort-relative signals **collapsed when a BLIP-2
  external baseline was added**, indicating a calibration confound rather than memorisation; manual
  adjudication found the image matches were *same-modality, same-projection images of different patients*.
  ⚠ **Covers neither PMC-VQA nor MMMU-Medical.** ⇒ **A new risk to a cell this project treats as clean —
  add a SLAKE landmine (§10, A5).**
- **Auditing Data Leakage in Whole-Slide Image Multimodal Benchmarks** (arXiv:2607.12278) `[V*]` — **92.3–100%
  case-level train/test overlap** on TCGA-derived WSI VQA benchmarks. Pathology-specific; unclear whether
  PathVQA is covered.
- **ReMedQA: Are We Done With Medical Multiple-Choice Benchmarks?** (EACL 2026 main, ACL Anthology
  2026.eacl-long.124) `[V*]` — **text-only** medical MCQA. The finding that matters: *"MCQA underestimates
  smaller models while inflating large ones that exploit structural cues — with some exceeding 50% accuracy
  even when the original questions are hidden."* A **question-blind >50%** result. Also
  **Atabuzzaman, Asgarov & Thomas** (arXiv:2509.16805) `[V*]` — LVLM MCQA selection bias intensifies with
  difficulty; no medical benchmarks.
- **Liu, Pan et al. *How Far Have Medical Vision-Language Models Come?* arXiv:2507.11200, 2025.** `[V*]` —
  3B–72B across essentially this project's exact suite. Two verbatim conclusions worth having: *"Large
  general-purpose models already match or surpass medical-specific counterparts on several benchmarks"* and
  *"Performance varies widely across benchmarks, reflecting differences in task design, **annotation
  quality**, and knowledge demands. No model yet reaches the reliability threshold for clinical
  deployment."* **It does not state which PMC-VQA split it used — which is itself the finding.**

⚠ **Essentially no paper states which PMC-VQA split it used.** Not Fleming-VL, not MedRCube, not the
benchmarking study, not the MedEvalKit README. **This project's observation that `test_clean ∩ test_2 = 6
items` while both circulate under one name, with the choice hard-coded in vendor code at
`utils/PMC_VQA/PMC_VQA.py:39` and disclosed by nobody, appears to be unpublished and publishable** — and
there is a visible empty slot for it, since MedGemma has already normalised "this benchmark has data-quality
problems" as a publishable statement and arXiv:2606.10066 has normalised the controlled-audit format while
explicitly leaving PMC-VQA out.

### 5.5 Is MedEvalKit a standard? Partially — and thinner than it looks.

`[V*]` from the GitHub repo (`alibaba-damo-academy/MedEvalKit`): **249 stars**, v1.0 released **2025-06-12**,
**28 benchmarks** (11 multimodal, 17 text). HF org activity through **2026-02-24**. **The README does not
specify which PMC-VQA split it evaluates** — consistent with the hard-coding at `PMC_VQA.py:39`. That is a
real, writable reproducibility defect.

**Confirmed adopter:** InfiMed-Foundation (arXiv:2509.22261) `[V*]`.
**Claimed adopters, body-text only, unconfirmed** `[S]`: MedRCube; *Learning from Medical Entity Trees*
(arXiv:2604.25296). A "MedUniEval" surfaced in search with **no primary source at all** — treat as
nonexistent.
**Confirmed NON-adopters** `[V*]`: Fleming-VL, Hulu-Med, OctoMed, MedGemma / MedGemma 1.5, OpenMedQ,
Aloe-Vision, MMedExpert-R1, MedMO. Several ship *their own* harness instead (MedGPT-oss promises "a rigorous
evaluation harness"; Aloe-Vision ships CareQA-Vision; MedRCube ships a new framework).

⇒ **The dominant 2026 pattern is every lab releasing its own harness** — exactly the fragmentation that
makes this project's "the number depends on the harness and nobody says which" argument land. **Do not claim
MedEvalKit is a community standard.** Claim it is *the harness that faithfully reproduces the anchor model*
— defensible, and more interesting.

---

## 6. AREA 6 — Reporting conventions for accuracy-vs-compute

### 6.1 What is emerging as standard

- **Pareto frontiers over multiple cost axes.** The closest template is **Prucs et al., arXiv:2512.24776**
  `[V-html]`: FLOPs/query on a log x-axis by **KV-aware estimation covering prefill and decode**, accuracy
  on y, point size = parameters, colour = architecture. **This project's prefill-inclusive FLOP-eq
  convention already matches**; the gap is that we publish a table where the field now expects a frontier
  plot.
- **Conference-level compute disclosure now exists.** The **CVPR 2026 Compute Reporting Form** `[V]`
  requires: CPU/GPU model, GPU count and memory, RAM/storage; **FLOPs with the measuring tool named**
  (ptflops / fvcore / thop / DeepSpeed) *or* GPU+CPU compute-hours; dataset name and size; model
  parameters; batch size and epochs; **baseline method, its performance, and % improvement**; and a
  **compute-allocation breakdown across training / fine-tuning / distillation / hyperparameter search /
  ablation / inference**; plus framework, mixed precision and distributed settings.
  ⇒ **This project can satisfy this today except for the allocation breakdown — a documentation task.**
- **Energy is being formalised as "intelligence per watt."** *Intelligence per Watt: Measuring Intelligence
  Efficiency of Local AI*, arXiv:2511.07885 v3, 2026 `[V-html]` (author list anonymised in the fetched
  version) defines **accuracy per unit power / accuracy per joule**, measured over 20+ local LMs (≤20B
  active), 8 accelerators, 1M+ real queries (WildChat, NaturalReasoning, MMLU Pro, SuperGPQA). Headline
  measurements: local models handle **88.7%** of single-turn queries; coverage rose **23.2% (2023) → 71.3%
  (2025)**, a **3.1×** increase; intelligence-per-watt improved **5.3×** (3.1× models × 1.7× accelerators);
  **oracle routing would cut energy 80.4%, compute 77.3%, cost 73.8%** versus cloud-only.
  ⇒ **This project already measures joules per query.** Reporting **accuracy per joule** puts its strongest
  surviving result (−84.3% energy vs the reasoning baseline `[REPO]`) onto the axis the efficiency community
  is standardising on.
- **"Matched quality, less cost, with a CI" is the shape that gets published.** UCCI `[V*]`: −31% cost
  [95% CI 27–35%] at micro-F1 0.91, **no accuracy claim at all**. SAFE-Cascade `[V*]`: parity claimed, +1.4
  pp called statistically uncertain. Cluster-Route-Escalate `[V]`: **retains** 97–99%.

### 6.2 Where this project is ahead of the published conventions

**No standard exists for macro-vs-pooled averaging of *cost* in adaptive systems.** Nothing in this sweep
addresses the problem this project identified on 2026-07-30: cost is additive per query, so macro-averaging
cost answers a different question from sample-weighting it. This project's rule — *report accuracy on macro
and BOTH cost numbers, each labelled, and never pair a macro accuracy with a sample-weighted cost*
(CLAUDE.md §0) — appears **ahead of the field**, and should be stated as a methodological contribution
rather than buried as a caveat. **SAFE-Cascade's 26.9%-fewer-calls / 9.3%-cheaper gap** `[V*]` is a
published instance of exactly the trap this rule prevents — cite it as evidence the rule is needed.

---

## 7. DOES THE FIELD PREDICT OUR TIE?

**Yes. Plainly, and from three independent directions. Our regime is the one in which the literature says
7B + test-time compute will *not* beat a 32B single pass.**

### 7.1 The enabling condition, checked against our own measurements

| condition | requirement | our measurement (`headroom_percell_2026-08-10.json`, `coverage_diagnosis2_2026-08-10.json`) `[REPO]` | satisfied? |
|---|---|---|---|
| **(a)** small model attains *non-trivial* success (Snell) | 7B plausibly competitive per item | macro 7B 0.5971 vs 32B-direct 0.6567; but **MedXpert 7B = 0.2615**, near chance on 5 options | **partly** — fails on the hardest cell |
| **(b)** correct answers reachable by search (Snell; Hariri) | reachable share > what is needed to tie | LP capture–recapture reachable: SLAKE-open **0.917** / VQA-RAD-open **0.692** / PathVQA-open **0.626**; needed merely to *tie*: **0.958 / 0.868 / 0.518** | **NO on 2 of 3 open cells** |
| **(c)** a verifier good enough to harvest the pool (Hu; Best-of-Evidence) | selection converts the oracle gap | sel_eff **0.775**; external medical SOTA **+0.26–0.58 pp** with a 235B judge; medical verifier FPR ≳60% | **NO** |
| **(d)** task is not knowledge-intensive (Zhao et al.) | TTS can add information | ~67% of biomedical benchmark questions are factual recall (Thapa et al.) | **NO** |
| **(e)** task is not perception-dominated (Sammani et al.) | TTS does not degrade | 4 of 8 cells are closed perception VQA; this project measures reasoning hurting 17/20 perception cells; image-token contribution decays along the chain | **NO** |

**We satisfy (a) partially and fail (b), (c), (d), (e).** There is no reading of the current literature
under which a 7B plus test-time compute is expected to beat a 32B single forward pass on MedEvalKit medical
VQA. **The tie is the correct scientific outcome, obtained honestly, at an evidentiary standard well above
the applied comparators' — who publish "retains 97–99%" and "parity, +1.4 pp uncertain, n=375" as positive
results.**

### 7.2 Where the field says 7B + TTS *does* win, and how far that is from us

All in one place: **verifiable-answer reasoning with a good external verifier** — MATH-500, AIME24, GSM8K,
LiveCodeBench, SWE-bench, and on the vision side MathVista / WeMath / LogicVista. The mechanism is that
verification is cheap and near-exact (run the test, check symbolic equivalence, compile). Hu's diagnostic
makes the dependence explicit: **+8.14 pp with an MCC-0.825 verifier vs +2.70 pp with an MCC-0.248 verifier
on the same pools** `[V]`.

Medical VQA has no such verifier and, per Verification Mirage `[V-html]`, the ones we can build are
capacity-coupled to the generator (a generator error raises verifier-failure odds **57×**). **The gap
between our regime and theirs is the verifier, and it is not closable by architecture** — this project tried
~20; the field's best medical attempt buys <1 point with a 235B judge.

### 7.3 The one *unclaimed* win, and we already have it

Sammani et al.'s positive result is that a *current-generation small* VLM with **cheap prompt-level TTS**
matches a same-family 32B on **reasoning** benchmarks, while TTS **degrades** perception benchmarks — and
their finding (3) supplies the mechanism (image-token contribution decays along the reasoning chain).
Combine that with this project's measurement that Lingshu's reasoning mode costs 15–49× the latency for no
perception benefit `[REPO]`, and the defensible, literature-supported claim is:

> **Format- and task-aware *allocation* of test-time compute — including the decision not to spend it — is
> what wins in medical VQA, and it wins on latency and energy against a reasoning baseline, not on FLOPs
> against a direct baseline.**

That is exactly this project's surviving result (**+0.0615 [+0.0514, +0.0715]** vs always-32B-reasoning,
**−87.9% latency, −84.3% energy** `[REPO]`), and the 2026 VLM literature now supplies the general mechanism.
**Lead with it. It is the finding the field is converging on, not a consolation prize.**

---

## 8. RANKED SHORTLIST — concrete, testable, for this codebase

Ranked by (decision value) × P(survival) ÷ GPU cost. Each names the prior negative it most resembles.

---

### S1 — Report the oracle gap **net of single-draw resampling noise**. Zero GPU. Highest decision value.

- **Mechanism.** Chen (arXiv:2607.03436) `[V*]` measures that **12–36% of the reported router-to-oracle gap
  is single-draw label noise that no single-commit router can capture**, approaching **half** on the hardest
  queries. This project has the data to estimate its own version: the **sc8 and independent sc16 pools** are
  already on disk and judge-labelled (`coverage_diagnosis2_2026-08-10.json` uses exactly this pair for
  capture–recapture) `[REPO]`. Estimate the item-level Bernoulli noise from the two independent draws and
  subtract it from every ceiling in `headroom_percell`.
- **Why it matters more than any experiment.** It changes the **denominator** of every claim this project
  makes about unconverted headroom. The routing ceiling is currently reported as **+0.0661 macro with ~1.3%
  converted** `[REPO]` — a devastating-sounding conversion rate. If 12–36% of that is unharvestable noise,
  the honest conversion rate rises and the "we captured almost none of it" framing softens into a measured
  statement about a partly-fictional ceiling. It is also the **best defence of the tie** available.
- **Expected effect.** No accuracy change. A materially different, more defensible headroom table.
- **Cost.** **Zero GPU.** ~1 day of analysis.
- **Prior negative it resembles.** None — this is a *measurement-model* correction, not a method.

---

### S2 — Add a **static pre-generation router** operating point (no cheap pass on escalated items). Zero GPU.

- **Mechanism.** Bouchard `[V]` measures that a lightweight pre-generation router **beat the best cascade on
  4 of 5 datasets**, *"mainly because it avoids the cheap model's generation cost."* Build a **query-only**
  router (text features + image embedding, cross-fit) that sends each item directly to one model, and report
  it as an extra frontier point beside the cascade.
- **Expected effect.** *Cost:* the cascade pays 1.0 FLOP-unit of 7B on 100% of traffic; a pre-router saves
  that on the escalated fraction — at the measured MCQ escalation of **44.24%** under macro `[REPO]`, ≈
  0.44/4.57 ≈ **0.10× of a 32B-direct pass**, on top of the cost floor already found (cheapest
  tie-preserving policy = **0.87–0.93×** direct as-charged, `cost_floor_2026-08-10.json`) `[REPO]`.
  *Accuracy:* expect a **loss** — query-only routing is the weakest point on the routing plateau, and this
  project's Finding 2 says MCQ routing signals are degenerate. **The value is a frontier point and a
  reviewer answer, not a win.**
- **Cost.** **Zero GPU** — both models' per-item correctness are dumped for all 8 cells.
- **Prior negative it resembles.** None directly; a cost-accounting variant, not a new signal.
- **Why it matters even when it loses.** It converts *"our cascade costs 1.2–1.4× direct"* from an
  embarrassment into a **measured structural finding with the literature's own explanation attached.**

---

### S3 — **Best-of-Majority** gating before the verifier. Zero GPU.

- **Mechanism.** Di et al. (ICLR 2026) `[V]`: restrict the pool to the **high-frequency** answers, *then*
  apply the trained verifier's argmax. Two lines of change; re-scores existing dumps.
- **Not a rediscovery of "all score fusions."** A fusion adds self-consistency as a *feature* and re-ranks
  the full pool; BoM **hard-restricts the candidate set before the argmax** — a different estimator, with a
  minimax-optimality proof and the property that **performance does not degrade as N grows**.
- **Why it might survive here.** This project has measured exactly the pathology BoM removes: sel_eff
  **1.000 → 0.914 → 0.841 → 0.770 (K=8) → 0.717 (K=16)** `[REPO]`, which Khalaf et al. `[V]` prove is
  inevitable for BoN. And frequency is **known to be informative here**: no-coverage items have modal share
  **0.432** vs **0.701** on recoverable, normalized entropy **0.830** vs **0.522** `[REPO]` — a large,
  already-measured separation on precisely the statistic BoM thresholds.
- **Expected effect.** Modest on sel_eff at N=8 (mean 3.76 distinct candidates `[REPO]`); the prize is
  **restoring monotonicity at K=16**, making a larger pool usable at all. Macro accuracy **+0.000 to +0.004**.
- **Cost.** **Zero GPU.**
- **Prior negative it resembles.** "All score fusions" (null); self-consistency plurality alone at sel_eff
  **0.713896, −0.061308 vs incumbent** `[REPO]`. Plurality losing as a *selector* is not evidence against
  plurality as a *filter* — opposite failure modes.

---

### S4 — Re-certify the macro gain under **workload-cluster resampling**. Zero GPU.

- **Mechanism.** RouteGuard (arXiv:2608.07583) `[V*]` shows routing gains that *"appear under prompt-level
  sampling but disappear under workload-cluster resampling due to concentration in 3 of 86 cells."* This
  project bootstraps at the item level. Re-run the headline bootstrap **resampling whole cells/clusters**,
  not items.
- **Why it matters.** This project already knows its delta is concentrated (leave-one-cell-out range
  −0.0004 to +0.0024; 100% of the MCQ-side delta is PMC-VQA's certified veto) `[REPO]`. A cluster-resampled
  CI states that formally and pre-empts the exact objection RouteGuard formalises. **It may well widen the
  CI to the point where even the vs-reasoning win needs restating** — which is worth knowing before a
  reviewer finds it.
- **Cost.** **Zero GPU.**
- **Prior negative it resembles.** The guardrail discipline (retrospective §5.7). This extends it from
  per-benchmark guardrails to the CI itself.

---

### S5 — ✅ **DONE during this sweep.** PMC-VQA option-position distribution, both splits.

- **Result: `test_2.csv` has B+C = 73.6% (A 13.2 / B 35.8 / C 37.8 / D 13.1); `test_clean.csv` has B+C =
  62.4% (A 21.9 / B 31.9 / C 30.5 / D 15.8).** Full detail and consequences in **§0.6**. `[MINE]`
- **Follow-ons this creates** (each zero-GPU unless noted):
  (a) **Add the landmine** to CLAUDE.md §0 beside the existing two-splits entry — §10 A12.
  (b) **Re-check the certified veto's bins against option identity** on PMC. The veto is the *only* source
      of MCQ-side macro delta `[REPO]`; if its Wilson-certified high-precision bins are enriched in B/C
      items, the certificate is partly reading the position prior rather than the 7B's competence. This is
      a direct integrity check on the load-bearing lever, on existing dumps.
  (c) **Promote retrospective §8.2 item 14** (option-order circular consistency as a gate signal). Its
      stated caveat was that the pre-test data sat on MMMU; a 73.6% two-option concentration on PMC is a
      much stronger motivation to replicate it there — and item 14 already measured that 33% of 7B items
      and 27% of 32B items flip under cyclic option shifts `[REPO]`.
  (d) **Report the constant-C floor (37.8%) as a baseline row** on the PMC cell. Omitting it is the kind
      of thing ReMedQA (EACL 2026) `[V*]` criticises — their question-blind >50% result is the same genre.

---

### S6 — Swap the cheap leg to **`Lingshu-I-8B`**. ~1–2 GPU-days. Highest value per GPU-hour.

- **Mechanism.** Same family, same training pipeline, InternVL3 backbone, weights on HF, updated 2026-02-24
  `[V*]`. Published: **PathVQA 74.9 (+13.0 over Lingshu-7B), SLAKE 91.6 (+8.5), VQA-RAD 73.0 (+5.1),
  average 64.5 (+2.7)** — against Lingshu-32B's 66.6. **MMMU-Med 49.1 (−4.9)**, which is the excluded cell.
- **Why this is the best GPU spend on the list.** (i) It attacks the **capability** wall, which
  `coverage_diagnosis` says is the binding one `[REPO]`, with **zero training**. (ii) PathVQA-open is this
  project's load-bearing cell (dropping it takes accuracy-max vs reasoning from +0.0720 to +0.0318) `[REPO]`.
  (iii) It directly executes retrospective §8.3 item 21 ("a stronger cheap leg") with a *protocol-matched*
  model, removing that item's stated blocker (that published cross-model comparisons do not use this
  harness). (iv) Roberts et al. `[V]` and Sammani et al. `[V-html]` both argue the newest well-trained small
  model is the key variable.
- **Expected effect, and the honest counter-risk.** A better cheap leg **shrinks the cascade's job**. If the
  8B closes most of the gap, the honest outcome may be *"the cascade is unnecessary"* — which invalidates
  the headline rather than improving it. **Do the zero-GPU version first**: recompute the operating point
  with the cheap-leg accuracy vector replaced by the published Lingshu-I-8B numbers, cost model fixed. One
  afternoon; it says whether to spend the GPU-days.
- **Prior negative it resembles.** None. This is the one untried structural change.

---

### S7 — **Online in-context distillation** of the cheap leg (Inter-Cascade / ICD). ~1 GPU-day, no retraining.

- **Mechanism.** Inter-Cascade (arXiv:2509.22984) `[V*]`: on each escalation, have the 32B emit a **reusable
  strategy** for that question type; store it; retrieve it into the 7B's context on similar future
  questions. **Zero parameter updates**, so no cheap-leg checkpoint is invalidated. ICD (arXiv:2510.18117)
  `[V*]` is the VLM version and adds **student-uncertainty conditioning to minimise teacher queries** —
  i.e. this project's gate, reused as the distillation trigger.
- **Reported magnitudes** (text / general-VLM, **not** medical): weak-model accuracy **+33.06%**,
  strong-model calls **−48.05%**, cost **−49.63%** (Inter-Cascade); **+33%** on small VLMs with **4%**
  teacher annotation (ICD).
- **Why it might survive here.** It attacks capability (the binding wall) without touching weights, and it
  reuses two things this project already has: a gate and 32B outputs on every escalated item across all 8
  cells. **The failure mode to watch:** VA-OPD `[V*]` measures that on-policy distillation *"can improve a
  student's output quality while failing to strengthen its reliance on visual input"* — so **pre-register a
  blank-image control**, exactly as retrospective §8.2 item 10 does for the verifier.
- **Prior negative it resembles.** RAG/retrieval (killed: the 32B fixes knowledge and perception errors
  equally, 38% vs 36%). **This is a genuinely different mechanism** — it retrieves *the strong model's own
  strategy for this question type*, not external textbook knowledge, so the "capacity not retrievable
  knowledge" argument does not obviously apply. But it is close enough that **the RAG kill's reasoning must
  be re-examined before spending**: if the deficit is capacity, an in-context strategy may not help either.
  **Run a 200-item pilot before committing.**

---

### S8 — **Cascade-aware LoRA** on the 7B (CAT channel (ii): calibration, not just accuracy). ~1–2 GPU-days.

- **Mechanism.** Wang et al. (arXiv:2406.00060) `[V]`, `[V*]`. Fine-tune Lingshu-7B with a loss aware of
  32B correctness on the same item — down-weight items the 32B also gets right, up-weight the **confidence
  separation** between items where the 7B is right and items where the 32B rescues it. 32B labels exist for
  all 8 cells.
- **Not a rediscovery.** **Every** prior recoverability mechanism froze the 7B. This moves the
  *distribution the gate reads* rather than searching for a better read of a fixed distribution — and it is
  the only proposal here **not bounded above by `headroom_percell`'s cheap-signal AUROCs**, because those are
  properties of the current 7B. The 2026 survey confirms it is **unexplored in multimodal** `[V*]`.
- **Expected effect.** Unknown for medical VLMs — all published CAT evidence is text LLMs on
  SuperGLUE/WMT/FLAN, and Gatekeeper's magnitudes are unverified. **A bet.** Pre-register both a cascade
  endpoint and a plain-accuracy endpoint so a null is informative. Note arXiv:2604.27720's warning `[V*]`
  that medical SFT **regressed VQA-RAD open-ended by 9.1 points** while gaining 42.8 on SLAKE — **set a
  per-cell guardrail.**
- **Risk.** Any 7B retrain **invalidates every cached cheap-leg checkpoint for that family.** Budget the
  re-dump.
- ⚠ **Write-up discipline.** Cascade-aware training between two models that both answer. Never
  learning-to-defer vocabulary (CRITICAL RULE 6).

---

### S9 — A joint two-model correctness head on 7B **prefill activations**, as the *gate*. ~4–8 GPU-hours.

- **Mechanism.** Varshney et al. (arXiv:2603.20895) `[V]`: probe layer by **Fisher separability**, extract
  7B prefill activations, train a **SharedTrunkNet** with two heads — P(7B correct) and P(32B correct) —
  escalate on the **difference**. `feats_hidden/` (4.4 GB) already exists.
- **Not a rediscovery.** Hidden-state work here built **verifier** heads; gate work used margin /
  peer-difficulty / confidence *scalars*. A **model-pool-aware joint correctness head** is precisely the
  remedy the Routing Plateau paper names `[V]`. The killed single-model routing (~29σ below the
  random-allocation floor) was within-model axis routing on one model's own confidence — structurally
  different.
- **Expected effect — temper it hard.** Calibrate-Then-Delegate `[V*]` built this exact probe and got
  **Spearman 0.27** against the strong expert. `headroom_percell` `[REPO]` says the best cheap signal here
  reaches AUROC **0.5806–0.6391** on disagreements, cross-fit cell gains **+0.0036 to +0.0105**, macro value
  **≤ +0.0013**; perfect identification is **+0.0661 macro** of which **~1.3%** is converted. Realistic
  landing zone: **+0.002 to +0.006 macro**, i.e. 1–2× the CI half-width of 0.0029. **Exclude MedXpert** from
  the success criterion (7B-wins AUROC 0.4877 = chance; structurally immovable).
- **The reframe that makes it worth doing anyway.** CTD's result is that a **weak recoverability signal
  still beats a strong detection signal** (+7% AUC, +9% accuracy over uncertainty-based delegation) — an
  external confirmation of retrospective §5.6. **Pre-register the criterion** (macro ≥ +0.0044, the
  "comfortable win" threshold in `macro_sensitivity`) and stop if it misses.
- **Adjacent untried variant worth folding in:** Warren & Dras `[V*]` estimate the **large** model's
  pre-invocation confidence with a *tiny proxy*, on multiple-choice data. **All 16 mechanisms here are
  small-model-side.** A 2-head design already covers this in spirit; a tiny 32B-proxy is the cheap ablation.

---

### S10 — Heterogeneous-**config** pool union with family-aware aggregation, gated by S3. ~4–8 GPU-hours.

- **Mechanism.** This repo already measured that unioning the endpoint 8-pool with a **different-config**
  8-pool gives **+0.065 [+0.035, +0.100]** oracle (temp 1.0) vs **+0.035 [+0.015, +0.060]** for a second
  iid draw at the same config — config heterogeneity roughly **doubles** coverage gain per sample
  (`coverage_diagnosis_2026-08-10.json`, VQA-RAD-open n=200) `[REPO]`. Hidden Clones `[V]` supplies the
  aggregation fix; BoM (S3) supplies the selector that does not degrade as the pool grows.
- **Not a rediscovery of "diverse/DPP generation at fixed N."** That was *within-config* diversity at fixed
  N. This is **across-config** (temperature / think-mode / resolution) union, **family/config-aware**
  aggregation, and a **monotone** selector — three changes, each addressing a distinct measured failure.
- **Expected effect.** At the measured **0.45** marginal multiplier `[REPO]`, ≈ **+0.029 selected accuracy**
  on VQA-RAD-open for 2× generation cost, against a 0.120 gap to 32B-direct there. **On its own it loses on
  cost.** Only run it *after* S3 shows the selector is monotone.
- **Prior negative it resembles.** Diverse/DPP generation (significant LOSS post-decontamination); multi-view
  / permutation TTA on MCQ (negative upper bound). **Open cells only. Do not run on MCQ** — straight
  rediscovery.

---

### S11 — Reporting: accuracy per joule, a frontier plot, and the CVPR compute-allocation table. Zero GPU.

- §6. Zero effect on the science; substantial on reviewability. **This project's strongest surviving result
  is an energy result**, and energy is the axis the efficiency community is standardising `[V-html]`. Add
  SAFE-Cascade's calls-vs-cost gap as the published instance justifying the dual-cost rule.

---

### S12 — Two prior-plan revisions, both zero-cost desk work.

- **Prefill pruning (retrospective §8.3 item 22).** arXiv:2604.27720 `[V]` measures that cropping medical
  VQA images to **perfect oracle bounding boxes lowers closed-ended accuracy by 0.9–18.0 points**, including
  on Lingshu. Attention-guided token pruning is a softer version of the same operation. **Keep the
  chest-X-ray IoU decisive arm; lower the projected benefit; add a closed-ended accuracy guardrail on the
  escalated subset as a stopping criterion.**
- **Sequential vs parallel TTS on MedXpert — an open question, not a recommendation.** arXiv:2607.09438
  `[V*]` measures that for small VLMs on MCQ, a **token-limit increase 1k→2k gave +3.7 pp while 8→16
  sampled chains gave +0.15 pp**. That argues sequential compute over parallel. **It is in direct tension
  with this project's Finding 1** (reasoning hurts perception; 0/9 trigger effects significant once format
  was matched). The tension may resolve by cell: MedXpert is the one genuinely reasoning-heavy cell, and it
  is also the one where the 7B is near chance. **Record as an open question. Do not act without a
  format-matched, token-audited design** — the standing rule from 2026-07-30 applies.

---

## 9. KILLS — confirmed rediscoveries; do not propose again

| idea | why it is dead | source |
|---|---|---|
| **Another verifier architecture** (any) | ~20 architectures converge at 0.80–0.81 here `[REPO]`; the field's independent conversion is **78.3%** `[MINE]`; medical SOTA buys **<1 pp** with a 235B judge | arXiv:2605.14163 `[V]`, arXiv:2607.20950 `[V-html]` |
| **Set-aware / multi-sequence verifiers** (score candidates conditioned on the pool) | published as new in 2026 — but this project already built it: adding the set-aware head **lowers** the fusion 0.812670 → 0.807902 `[REPO]` | arXiv:2603.03417 `[V*]` |
| **Self-verification / self-critique loops** | verifier FPR ≳60% on medical VQA; generator error raises verifier-failure odds **57×**; **69.5–87.1%** of wrong answers *locked in* over 4 turns; only 2.2–3.8% corrected | arXiv:2605.10850 `[V-html]` |
| **A bigger same-family verifier** | **no significant FPR reduction for Lingshu specifically, p = 0.782** | arXiv:2605.10850 `[V-html]` |
| **Iterative self-refinement** | degrades open-weight models; helps only closed frontier models | arXiv:2512.11109 `[V]` |
| **Post-hoc calibration to rescue the gate** | Platt scaling is **strictly monotone ⇒ AUROC unchanged**. Only *hallucination-aware* recalibration moves AUROC, and its gains are **largest on open-ended** questions — an independent replication of Finding 2, and a reason not to try it on MCQ | arXiv:2604.02543 `[V*]` |
| **Cross-family verifier ensembles as a *new* idea** | mechanism now explained (effective dimensionality 2.5–3.6); the fix is worth ~1–2 pp on general VQA — **below our CI half-width of 0.0029** | arXiv:2603.17111 `[V]` |
| **Process reward models on our traces** | PRMs score reasoning steps; **0/9** reasoning-trigger effects were CI-significant once format was matched, and reasoning hurts 17/20 perception cells. There are no steps to score | `[REPO]` + arXiv:2606.28864 `[V-html]` |
| **More iid sampling on the open arm** | capture–recapture: SLAKE-open and VQA-RAD-open **unreachable by iid sampling at any N**; a 3×-budget redraw rescues 21.2%; and a medical multi-image study measured accuracy **dropping** 57.89 → 56.02 when the search budget doubled | `coverage_diagnosis_2026-08-10.json` `[REPO]` + arXiv:2607.27564 `[V*]` |
| **Within-config answer diversity / DPP at fixed N** | already a significant LOSS post-decontamination; the no-coverage subset is the *high*-diversity subset, so diversity is the wrong axis | `[REPO]` |
| **Multi-view / permutation TTA on MCQ** | eval-visible upper bound negative here; TTS degrades perception benchmarks generally | `[REPO]` + arXiv:2606.28864 `[V-html]` |
| **RAG / retrieval** | dead here (32B fixes knowledge and perception errors equally, 38% vs 36%). ⚠ **One piece of contrary external evidence:** tool-integrated retrieval *verification* reports **+23.5% MedQA / +32.0% MedXpertQA** over a baseline generator — but it is **text-only medical**, and its baseline is a generator without retrieval, not a 32B. **Restate the kill as *"retrieval does not fix our 7B's errors"*, not *"retrieval never helps in medicine"*** | `[REPO]` + arXiv:2601.20221 `[V]` |
| ⛔ **Abstention / reject-option / triage-to-human, in any form** | **CRITICAL RULE 6, permanent.** §11 lists the live 2026 medical strand **only so it is recognised and avoided** | — |

---

## 10. ASSUMPTIONS IN `CLAUDE.md` / THE RETROSPECTIVE THAT THE LITERATURE CONTRADICTS OR COMPLICATES

**A1 — Finding 1 is no longer novel; it is replicated, and someone has published the mechanism.**
CLAUDE.md §0 / retrospective §5.1 present "reasoning hurts perception" as one of three findings that
generalize. Sammani et al. `[V-html]` (ECCV 2026) and Ahmadpour et al. `[V]` now measure the same phenomenon
on general VLMs — and Sammani's finding (3), that **image-token contribution decays along the reasoning
chain**, is a *mechanism* this project does not have. **What remains genuinely ours:** the medical
replication at scale, the **format-matched, token-audited** protocol, and the finding that the apparent
reasoning gain is an **answer-format** effect. **Rewrite as confirmation-plus-mechanism, not discovery** —
a reviewer will find arXiv:2606.28864 in five minutes.

**A2 — The macro cost reversal is framed as a defect; it is a structural theorem, published.**
CLAUDE.md §0 treats 1.196× / 1.410× as a retraction. Bouchard `[V]` shows it is intrinsic (*"cascades pay
the cheap model before any escalation decision"*) and that a **pre-generation router beat the best cascade
on 4 of 5 datasets** for exactly that reason. **The reversal is not a failure of our gate; it is the
structural-cost term becoming visible once PMC-VQA stops carrying 79% of the weight.** Say so, and add S2.

**A3 — "Lingshu is SOTA" should become "faithfully reproducible with a matched 7B/32B pair."**
The anchor survives — MedRCube `[V*]` benchmarks **33 MLLMs** in April 2026 and still places Lingshu-32B
top-tier. But Fleming-VL-38B `[V*]` reports beating it on five VQA benchmarks (**with a copied baseline, no
MedEvalKit, and no split disclosed**), and Sammani et al. `[V-html]` show a **current-generation 4B beating
a same-generation 32B** on 3 of 6 benchmarks. **Change the adjective**; the property the method needs is a
matched pair on a faithful harness, which no successor has been shown to have.

**A4 — `Lingshu-I-8B` exists and is absent from CLAUDE.md.** InternVL3-based, same pipeline, HF updated
2026-02-24 `[V*]`; average 64.5 vs Lingshu-7B 61.8, **PathVQA +13.0**, **MMMU-Med −4.9**. This is a
protocol-matched, zero-training stronger cheap leg. **Add it to §8 of CLAUDE.md and run S6's zero-GPU
version.**

**A5 — SLAKE is treated as clean; there is now published contamination evidence against it, and MedGemma
has put PathVQA and VQA-RAD on the record too.** No CLAUDE.md landmine covers any of these.
arXiv:2606.10066 `[V-html]` measures **19.8% of SLAKE images with an extreme same-view near-neighbour** in
PMC-OA-beta (4.2% under a different encoder), while finding VQA-RAD clean (≤0.9%); the authors' adjudication
says these are same-modality/same-projection images of *different patients*, so **not** duplicate leakage —
but it is real, published and citable. Separately, MedGemma's technical report `[V*]` states verbatim that
*"we and others have identified potential data quality issues in PathVQA and MedVQA"* (and removed them from
training) and that VQA-RAD's original splits carry *"train/test image contamination."* **Add landmines for
SLAKE, PathVQA and VQA-RAD.**

**A6 — The "selection wall is 74–82%" reads as a local limitation. It is a field constant — and our
decomposition is no longer novel.** Two independent external measurements land in the band (78.3% `[MINE]`
from arXiv:2605.14163; Hu's fixed-pool diagnostic shows selection gain tracking verifier MCC almost
linearly `[V]`). **Promote it from caveat to contribution — and cite Hu (arXiv:2607.17531) rather than
claim priority on the decomposition.**

**A7 — Retrospective §5.6 ("rank gates by recoverability, not detection") is CONFIRMED externally, and is
more valuable than the repo treats it.** Calibrate-Then-Delegate `[V*]` states this project's finding as its
premise, builds the recoverability probe, gets **Spearman 0.27** — and **still beats uncertainty-based
delegation at every budget (+7% AUC, +9% accuracy).** RouteLMT `[V*]` independently identifies **marginal
gain** as the right signal. **This is one of the project's most transferable results and is currently
buried in §5.6.**

**A8 — Retrospective §8.3 item 25 weakens speculative decoding on the grounds that "reasoning buys
nothing." That inference does not hold, and the real reason is different and better.** Speculative decoding
runs the **32B on every query**; it shortens the *decode* chain only, while VLM cost is prefill-dominated
(the "visual memory wall", arXiv:2604.05546 `[V*]`). So it delivers **latency, not FLOPs**, and **zero**
accuracy change by construction. Measured VLM speedups are 1.42–3.28× `[V*]`. **Replace item 25's
justification.** And note the reviewer risk: LongSpec (ACL 2025 main) `[V*]` calls cascades *"lossy
alternatives"* in its opening framing — **pre-empt that sentence in the paper.**

**A9 — Nothing in this sweep supports the MMMU exclusion, and nothing contradicts it.** The only controlled
medical-VLM contamination audit `[V-html]` covers **neither MMMU nor PMC-VQA**. CLAUDE.md's 2026-07-30
instruction to re-argue the exclusion *on contamination grounds alone* therefore still has **no external
evidence to lean on.** Keep the internal adversarial audit as the sole support and say so. *(Weak
circumstantial datum: Lingshu-I-8B, same pipeline, different backbone, loses 4.9 points on MMMU-Med while
gaining everywhere else `[V*]` — consistent with MMMU-Med behaving differently from the rest of the suite,
but not evidence of contamination.)*

**A10 — "MedEvalKit is the faithful harness" is right; "MedEvalKit is a standard" would not be.** 249 stars;
one confirmed third-party adopter; eight confirmed non-adopters that ship their own harnesses `[V*]`. And
its README **never states which PMC-VQA split it evaluates** — the choice is hard-coded at
`utils/PMC_VQA/PMC_VQA.py:39`. **Claim faithfulness to the anchor, not community standardisation** — and
note the undisclosed split as a reproducibility defect worth writing up (§5.4).

**A12 — CLAUDE.md's PMC-VQA landmine covers *which split*, but not *what is in it*. Add the answer-position
bias.** Measured this sweep (§0.6) `[MINE]`: `test_2.csv` has **B+C = 73.6%** and a **37.8% constant-C
floor**, versus `test_clean.csv`'s 62.4% / 31.9%. The v2 test split carries the *training* split's
position profile `[S]` for the train comparison. Since PMC-VQA is **79.2% of the sample-weighted pool** and
**100% of the MCQ-side macro delta** `[REPO]`, this belongs beside the existing two-splits landmine, and it
makes S5(b) — auditing the certified veto's bins against option identity — an integrity check rather than a
nice-to-have.

**A11 — Confidence-gate robustness is unexamined and now has a published attack.** Forced Deferral
(arXiv:2606.15308) `[V*]` demonstrates an adversarial **image** trigger that flattens the weak model's
confidence and forces escalation, across datasets, families and deferral metrics, **specifically for
multimodal cascades**. **If the paper claims clinical deployability of a confidence gate, this must be
acknowledged.**

---

## 11. RECORDED AND EXCLUDED — the abstention strand (CRITICAL RULE 6)

Listed **only** so it is recognised and avoided, and so nobody re-discovers it as an opportunity. **None of
this may be researched, built, tested, or proposed.**

- **Calibrated Triage, Not Autonomy: Confidence Estimation for Medical Vision-Language Models** —
  Khanmohammadi, Thind & Ghassemi, arXiv:2606.15910, 2026 `[V*]`. Seven confidence estimators × five
  open-weight LVLMs × three medical VQA datasets. Explicitly *"bounded selective prediction"* / *"route the
  rest to a clinician"* — **the banned framing.**
  **The one usable, non-abstention fact:** *"the weakest baselines are confidently wrong on 41 to 45 percent
  of their errors against 1 to 4 percent for the best probe,"* and calibrated scores recover ~1/3 of
  **radiology** cases at 20% error tolerance but **"almost none of pathology."** ⇒ **That
  pathology-vs-radiology asymmetry is an independent corroboration of PathVQA being this project's hard
  cell. Use the asymmetry; discard the framing.**
- **AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays** (arXiv:2508.19322); **Conformal Triage for
  Medical Imaging AI Deployment** (medRxiv 2024.02.09.24302543). Abstention/defer-to-human. **Do not
  pursue.**
- The 2025–2026 **learning-to-defer** theory literature (H-consistency bounds, one-stage vs two-stage
  surrogates) is framed around deferring to a **human expert**. The only part that is *not* abstention, and
  is legitimately usable, is the **one-stage vs two-stage distinction** itself: one-stage = predictor and
  deferral rule trained jointly; two-stage = policy fit post-hoc on frozen predictions. **This project is
  two-stage; CAT/Gatekeeper (S8) are the one-stage move between two models that both answer.**
- **Flexible Routing via Uncertainty Decomposition** (arXiv:2605.07805) `[V*]` — its
  **reducible** (route to the big model) vs **irreducible** (inherently ambiguous) uncertainty
  decomposition is usable and interesting; its three-way policy including abstention is not. It also carries
  a caveat worth knowing: the benefit appears only *"whenever reducible and irreducible uncertainty are not
  too correlated."*

---

## 12. WHAT THIS SWEEP DID NOT ESTABLISH

Stated so gaps are not mistaken for absences of evidence.

- **Numbers inside four load-bearing PDFs are still `[S]` or abstract-only:** the Routing Plateau's actual
  plateau values; GPRO's per-benchmark results; Gatekeeper's magnitudes (OpenReview bot challenge);
  and REOPOLD's "7B matches 32B" benchmark. **Pull all four before building an argument on them.**
- **The PMC-VQA option-position percentages attributed to the *original paper*** (train 73.5% / test 62.4%)
  were read via search extraction from its HTML and remain `[S]`. **My own counts on the two CSVs are
  measured and reproducible** (§0.6) — but the *inference* that `test_2` "has the train distribution" rests
  on the paper's unverified train figure, so state it as "matches a reported 73.5% train figure" rather
  than as established fact until the PDF is checked.
- **A search-snippet claim was checked and found FALSE.** *Model Routing as a Trust Problem*
  (arXiv:2605.01710) was surfaced with the claim that routers *"do not reliably outperform a simple
  baseline."* The abs page was fetched: **that claim is not there** — it is a position paper on runtime
  transparency artifacts with **no measured routing results** `[V*]`. **This is the error rate to assume for
  everything marked `[S]`.**
- **Two `[S]` medical-VLM claims remain unconfirmed and should not be quoted:** Citrus-V-33B beating
  Lingshu-32B by 0.59%; MedMO-8B-Next at a 72.7% VQA average. Likewise MediX-R1's 68.8% / 73.6% figures.
- **A LoRA-vs-AdaLoRA medical VQA comparison** (Computers in Biology and Medicine, Dec 2025) reporting
  ~90% VQA-RAD / 93% SLAKE with Idefics3-8B is **paywalled, non-arXiv, and unverified.** If real, it is a
  strong "adaptation closes the gap" data point and worth an institutional fetch.
- **MedEvalKit adoption** is characterised from GitHub metadata and non-mention in eight papers — an
  absence-of-evidence argument, not a systematic survey.
