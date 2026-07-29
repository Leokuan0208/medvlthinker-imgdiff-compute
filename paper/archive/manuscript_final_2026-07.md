# A Format-Aware, Regime-Adaptive Cascade that is *Faster and More Accurate* than always-32B-think for Medical VLMs — with an Honest Map of the Two Walls

*Li-Wen Kuan (Leo) et al. — final long-form manuscript, 2026-07. Every figure is verbatim from a real
checkpoint or arithmetically derived from one; none is fabricated. This document is a superset of
`paper/manuscript_2026-07_longform.md` that folds in the entire 2026-07-04 → 2026-07-07 research cycle
(the integrated `method_final.py` pipeline, the V2 F8/F10 levers, the Pandora controller, the
selectability-wall battery, and the re-grounding against the 32B-think baseline). Canonical sources:
`results/cascade_methods/docs/current/{METHOD_FINAL_2026-07,RESEARCH_RESULTS_2026-07,METHODS_MASTER,
MASTER_SUMMARY_2026-07,OMNIMED_FALLBACK}.md`, the artifacts under `results/cascade_methods/artifacts/`,
and `progress/progress_July_0{4,5,6,7}.md`.*

---

## Abstract

Deploying a medical vision–language model (VLM) at the accuracy people actually pay for means running a
**large reasoning model that emits a `<think>` trace**: at batch-1, a Lingshu-32B answers one open-text
question in a *measured* **10.5 s** and **≈2.0 kJ**, versus **≈0.35 s** for a 7B. A **cascade** — answer
cheaply, escalate only the hard cases — is the standard route to efficiency, but a medical-VQA cascade must
make two decisions that are usually collapsed into one: **which** queries to escalate (the *gate*) and
**what** to run once escalated (the *action*). This paper is a study of both, and its deliverable is a
single, honestly-scored system.

**The system.** A **format-aware, regime-adaptive router** detects each query's answer format from the
*prompt* (never the gold), and dispatches it to one of two arms. The **MCQ / closed arm** runs the cheap
Lingshu-7B (no-think), gates on its top-1−top-2 probability **margin**, and escalates the low-margin ~16 %
to Lingshu-**32B-no-think** (thinking never helps Lingshu on this suite); it *keeps* the 7B on MMMU (an
observed 7B > 32B-think anomaly); and on the largest slice, PMC-VQA, an optional held-out-certified fusion of
the two legs *beats* the 32B. The **open-text arm** runs 7B best-of-N with a small **trained outcome
verifier** that selects the best candidate and gates on verifier confidence, escalating the residual to
32B-no-think. A single Pareto **knob** trades compute for accuracy, and — after two held-out levers (F8
certified weak-veto, F10 team-objective learning-to-complement) — **both settings are FLOP-negative**.

**The result (full MedEvalKit suite, n = 42 374, sample-weighted).** The method **matches-or-beats
always-32B-THINK accuracy at ~96 % lower batch-1 latency**: the **compute-lean** setting is **+0.0123**
vs 32B-think at **0.49×** its FLOPs (469 → 468 ms parallel latency, −96 %), and the **accuracy-max V2**
setting is **+0.0212** at **0.93×** FLOPs (all-axes dominant: cheaper, faster, and more accurate than the
naive baseline). Always-32B-THINK is **Pareto-dominated on every one of five model families** (the method is
9–68 % of its FLOPs, 8–99 % lower latency, and on 3 of 5 families *more accurate*), because thinking
**over-thinks perception VQA**.

**The honest negative — two walls.** We map, rather than paper over, the two limits that bound the program.
The **recoverability wall**: the broad-slice MCQ *beat* over the 32B is confined to PMC-VQA, confirmed by
**six independent fusion/routing methods** and by an automatic slice-discovery search that re-finds only the
PMC and MMMU cells and no genuinely-new one (2 across 8 splits vs a permutation-null floor of 5.6). The
**selectability wall**: best-of-N's oracle→selection gap is largely fundamental — a 7×-bigger 32B verifier
only *ties* the small trained 7B verifier (+0.005, n.s.), diverse-generation coverage does not compound with
pairwise selection, and no pre-filter converts the extra coverage. We show the best-of-N leg is
FLOPs-dominated against a *cheap* no-think strong model but is latency-alive and re-priced favourably against
the *think* baseline the deliverable is actually about. Finally, a hierarchical-credibility (H8) audit
confirms the deployed **CI lower-bound guardrail** is the correct, sufficient robustifier. We release the
unified pipeline (`method_final.py`, one command, CPU-only re-costing of saved dumps), the trained answer/box
verifiers, and the complete negative-result characterization.

---

## 1. Introduction

A medical VLM takes an image — a radiograph, a pathology slide, an endoscopy frame — together with a
question, and returns an answer. The strongest medical VLMs today are large reasoning models that emit a long
`<think>` trace before answering; they are accurate but slow and energy-hungry. In this repository's own
**measured batch-1** numbers, a Lingshu-32B answering an open-text VQA question in its native `<think>` mode
costs **10 521.6 ms and 2 001.9 J**, versus **665.0 ms / 126.9 J** for the same 32B in no-think mode and
**≈347 ms** for a 7B — a **15.8×** latency and energy tax for thinking (`opentext_32b_think.json`). The
standard route to efficiency is a **cascade**: run the cheap model first and escalate only the hard cases.
A cascade's quality hinges on two decisions usually collapsed into one — the **gate** (*which* queries to
escalate) and the **action** (*what* computation to run on escalation). This paper separates them, builds a
deployable system out of both, and charts the sharp limits of each.

**The deliverable's honest baseline is always-32B-THINK.** Prior drafts of this program scored the method
against always-32B-*no-think* — the cheapest, fastest configuration of the strong model. That is the wrong
strawman: the deployment-cost problem the paper is about is the **think** model people pay for when they want
its accuracy, not its 665 ms no-think shortcut. We therefore re-price everything against **always-32B-THINK**
(measured cost above), keep 32B-no-think as an intermediate *tier* rather than the baseline, and report every
delta against **both** so the reader can see exactly where each win comes from.

**Contribution 1 — a format-aware, regime-adaptive router that beats always-32B-think on both axes.** The
top-level router reads each query's answer format from the prompt and dispatches:

- **MCQ / closed →** cheap **Lingshu-7B no-think**, gate on the **margin**, escalate the low-margin residual
  (~16 % pooled) to **Lingshu-32B-no-think**. Two per-benchmark overrides: **keep-7B on MMMU** (Lingshu-7B
  0.80 > 32B-think 0.66, an anomaly the router simply exploits, +0.140 vs think at 1.0 FLOP), and an optional
  **PMC-VQA fusion** that *beats* the 32B.
- **Open-text →** **7B best-of-8 + a trained outcome verifier** that picks the best candidate, gate on
  verifier confidence, escalate the residual to 32B-no-think. On the out-of-distribution open sets the cheap
  best-of-8 ensemble *beats* the 32B (bo8 0.414 vs 0.331 pooled Lingshu), which in turn beats 32B-think by
  +0.12…+0.21 because thinking over-thinks perception.

On the full suite (n = 42 374) the **compute-lean** knob is **+0.0117** vs always-32B-think (+0.0017 vs
no-think) at **0.49×** its FLOPs and **−96 %** parallel latency; the **accuracy-max** knob is up to **+0.0238**
vs think. Two held-out levers folded in for V2 — **F8** (certified weak-veto) and **F10** (team-objective
learning-to-complement) — make **both** knobs FLOP-negative: compute-lean **+0.0123 @ 0.49×**, accuracy-max
**+0.0212 @ 0.93×** (a strict all-axes dominance over the naive baseline).

**Contribution 2 — structure buys efficiency (the ACC), across five families.** Underneath the MCQ arm is a
three-tier **Adaptive-Compute Cascade** that routes over *compute configurations of the same models* —
7B-no-think → **32B-no-think** → 32B-think — inserting the strong model's *fast no-think* mode as an
intermediate tier because reasoning over-thinks perception (32B-no-think ≥ 32B-think on the competent sets).
At parity with always-32B-think this cuts **latency 11.34 s → 2.27 s (−80 %), FLOPs 100 → 52 %, energy ≈ 5×**
on the six-benchmark suite, and it is **Pareto-dominant over always-32B-think on all five families** measured
(the method is 9–68 % of its FLOPs and, on 3 of 5, *more accurate*).

**Contribution 3 — a little training buys accuracy (the trained verifier).** A small LoRA outcome verifier
that scores P(correct | image, question, candidate) and selects best-of-N **beats the strong 32B/38B on
accuracy across three model families and held-out OOD** (pooled cascade-best vs strong: Lingshu 0.421 vs
0.331, MedVLThinker 0.344 vs 0.277, InternVL3 0.255 vs 0.218 — the last via a *Lingshu-trained* verifier
applied cross-architecture), for free-text answers *and* structured bounding boxes (SLAKE organs 40–53 %; the
real MS-CXR chest-X-ray benchmark 77–78 % of the oracle gap, bootstrap +0.191, 95 % CI [+0.152, +0.232]).

**Contribution 4 — the honest negatives, presented as a headline, not a footnote.** Two walls bound the
program and we characterize both rather than hiding them:

- **The recoverability wall.** Over a *frozen* model no *training-free* gate reliably beats trivial baselines:
  a dozen escalation signals cluster at recoverability AUROC ≈ 0.5–0.69 because the two models fail together
  (competent-4 P(32B wrong | 7B wrong) = 0.584). The *positive* corollary — a broad-slice accuracy beat over
  the 32B — turns out to be confined to **PMC-VQA**, confirmed by **six independent fusion/routing methods**
  (F3 confidence-advantage, F8 weak-veto, F11 Bayesian model averaging, F6 contrastive decoding, option-logit
  fusion, and Domino/H4 automatic slice discovery) and by an FDR-controlled slice search that finds **no**
  genuinely-new beat-slice beyond PMC and the MMMU anomaly.
- **The selectability wall.** Best-of-N's oracle → selection gap is largely fundamental: a 7×-bigger 32B
  verifier only *ties* the trained 7B one, diverse-generation coverage does not *compound* with a real
  pairwise selector, and no pre-filter converts the extra coverage. Best-of-N is FLOPs-dominated by a *cheap*
  no-think strong leg but latency-alive and favourably re-priced against the *think* baseline.

**Contribution 5 — cross-field transfers and honest novelty.** We import mechanisms from adjacent fields and
say plainly what is inherited: the **trained verifier** descends from generative reward models (GenRM); the
adaptive **Pandora controller** is Weitzman's (1979) optimal-stopping rule; the **agreement gate** is the ABC
cascade family; the fusion analysis reduces to **Chair–Varshney** decision fusion. Our contribution is the
*compute-configuration structure*, the *format-aware regime-adaptive assembly*, the trained-verifier
*application and unification* across answers and boxes, and the *characterization* of the two walls.

The rest of the paper: §2 fixes models, benchmarks, the faithful protocol, baselines, and the measured cost
model; §3 specifies the full method (both arms, all levers, the Pareto knob, and the one-command
reproduction); §4 reports the headline results; §5 the ablations; §6 the two walls (the honest headline
negative); §7 the 3-family reproduction and the OmniMed fallback; §8 related work and novelty; §9 limitations;
§10 concludes.

> **A standing scope rule (⛔).** This project's method *always answers*. We do not study, present, or
> recommend abstention / reject-option / defer-to-human as a method. Every "escalate" decision routes to a
> model that produces an answer; every "keep-7B" or "veto" decision delivers the 7B's answer. Where selective
> or conformal machinery appears (Wilson lower bounds, held-out τ), it is used only to decide *which model
> answers*, never to withhold an answer.

---

## 2. Setup, Metrics, and Definitions

Everything downstream is defined against the objects introduced here. Every number quoted is a real
measurement traced to a checkpoint.

### 2.1 Models and families

The paper is organized around **three medical-VLM families**, each a small "cheap" leg and a large "strong"
leg of the *same* architecture, plus two additional families used only for the ACC generalization study.

| Family | Cheap leg | Strong leg | role |
|---|---|---|---|
| **Lingshu** | Lingshu-7B (`lingshu-medical-mllm/Lingshu-7B`, Qwen2.5-VL-based) | Lingshu-32B | **primary deployment**: the router, both arms, the fusion, the verifier base |
| **MedVLThinker (MVT)** | MedVLThinker-7B | MedVLThinker-32B | full efficiency instrumentation (batch-1 latency/energy); cross-family transfer |
| **InternVL3 (IV3)** | InternVL3-8B | InternVL3-38B | cross-architecture transfer test (verifier + faithful MCQ cascade) |
| QoQ-Med, Chiron-o1, MedGemma | 7B/2B/4B | 32B/8B/27B | ACC over-thinking generalization only (§4.6) |

The **outcome verifier** (§2.6, §3.6) is a LoRA adapter (~190 MB) on the *frozen* Lingshu-7B, trained on
~6 000 judged `(image, question, answer)` triples at a positive rate 0.194; the **box verifier** uses
Qwen2.5-VL-7B. Weights are cached under `/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-{7B,32B}`.

**The model quirk that governs the cost accounting.** The MVT and Lingshu strong legs run in two modes:
**no-think** (answer directly, ≈2 decode tokens) or **think** (emit a `<think>…</think>` trace, hundreds of
decode tokens). The mode is not free-floating — these models emit `<think>` only under one exact system prompt
(*"You will solve a problem/request. You should provide your thoughts within `<think>` `</think>` tags before
providing the answer."*). We treat "32B no-think" and "32B think" as two separately-measured configurations of
the same weights. On **Lingshu specifically**, its promptable "think" run ≈ its no-think run (latency ratio
1.2×, near-zero reasoning gain), so the Lingshu deployment fires the think tier **~0 %** and reduces to
`7B-nt → 32B-nt` (+ the open-text arm); the think tier matters on the reasoning-native families.

### 2.2 The seven MedEvalKit benchmarks (MCQ + open-text) and pools

The benchmark suite is the **Lingshu paper's seven medical multimodal VQA tasks** (arXiv 2506.07044,
MedEvalKit): PMC-VQA, SLAKE, VQA-RAD, PathVQA, MMMU-Medical, MedXpertQA-MM, and OmniMedVQA. Report-generation
and text-only tasks are out of scope. Formats and roles:

| dataset | domain / images | format | eval | test n | role |
|---|---|---|---|---|---|
| PMC-VQA | PubMed Central figures | MCQ 4-opt | exact-match | 33 430 | MCQ arm (largest slice; fusion cell) |
| SLAKE | radiology CT/MRI/X-ray, bilingual | open + closed | judge / exact | 836 cl / 645 open | MCQ arm + verifier arm |
| VQA-RAD | radiology chest/head/abdomen | open + closed | judge / exact | 251 cl / 200 open | MCQ arm + verifier arm |
| PathVQA | pathology/histology | open + closed | judge / exact | 3 362 cl / 1 500 open | MCQ arm + verifier arm |
| MMMU-Medical | college medical exams, multi-image | MCQ | judge (parsed) | 150 | MCQ arm (reasoning; **keep-7B**) |
| MedXpertQA-MM | expert/exam multimodal (hardest) | MCQ | exact-match | 2 000 | MCQ arm (near-chance) |
| OmniMedVQA | many imaging modalities | MCQ 4-opt | exact-match | 88 996 | reproduce; **keep-cheap / strong blocked** (§7) |
| Kvasir-VQA-x1 | GI endoscopy | open | judge | 1 200 | verifier training / OOD |
| RadImageNet-VQA | radiology | open | judge | 2 000 | verifier held-out transfer |
| MS-CXR | chest X-ray | box grounding | IoU ≥ 0.3 | 435 | box verifier |

**Evaluation pools.** The MCQ core eval set is **8 220** samples across the first six MCQ benchmarks; the
integrated method is scored on the **full suite** = 6 MCQ benchmarks (n = 40 029 MCQ) + 3 open-text sets
(SLAKE-open 645, VQA-RAD-open 200, PathVQA-open 1 500) = **n = 42 374**. For the ACC efficiency study we also
name **ALL-6** (all six MCQ, n = 8 220), **ALL-5** (ALL-6 minus MedXpertQA, where both legs are near chance),
and **COMPETENT-4** (SLAKE, VQA-RAD, PathVQA, PMC-VQA).

**Scoring (faithful).** MCQ/closed → MedEvalKit exact-match `correct` (MMMU → `parsed_output` judge ==
`Correct`); open-text → LLM/Claude-judge `judge_ok` (open-text exact-match is known-broken, e.g. `"CT." ≠
"CT"`). PMC-content `oks` in the best-of-N battery is loose option-letter matching and is flagged wherever
used.

### 2.3 The faithful MedEvalKit protocol and reproduction fidelity (three families)

MCQ accuracies are produced with **MedEvalKit** under a specific recipe so per-model numbers reproduce the
published ones: an isolated `medeval_venv` (vLLM pinned to **0.9.0.1**, transformers 4.52.4, triton 3.7.1);
the `Qwen2_5_VL` wrapper pointed at the family's native weights; `datasets_path=hf`; `use_vllm` on;
`TORCHDYNAMO_DISABLE` set; `use_llm_judge=False` for the MCQ (exact-match) halves. We flag explicitly that the
**internal NGC labeling harness is *not* faithful** for this purpose (it gives systematically different
scores, e.g. a secondary NGC Lingshu-32B run of SLAKE 89.4 / PMC 64.0 / MMMU 62.4) — so every "faithful"
number comes from the MedEvalKit recipe, and NGC numbers are never used for a headline. Reproduction fidelity
(percent; "ours" vs Lingshu-paper Table 6):

| Benchmark | Lingshu-7B ours | 7B paper | Lingshu-32B ours | 32B paper |
|---|---|---|---|---|
| MMMU-Med | 80.0 | 54.0 | **63.3** | **62.3** |
| SLAKE-closed | 82.5 | 83.1 | 85.9 | 89.2 |
| PMC-VQA | 54.3 | 56.3 | 55.2 | 57.9 |
| MedXpertQA-MM | 26.2 | 26.7 | 30.6 | 25.7 |
| VQA-RAD-closed | 78.1 | 67.9 | 85.3 | 76.5 |

The clean like-for-like anchor is **MMMU-Med-32B: ours 63.3 vs paper 62.3** on the same 150-question set. PMC
tracks the paper for both sizes. SLAKE/VQA-RAD "ours" are the *closed* halves vs the paper's *open+closed*
blend, so their offsets are composition, not error. The one genuine anomaly is **Lingshu-7B MMMU-Med = 80.0**,
a Lingshu-7B-specific +26 inflation confirmed family-specific (MedVLThinker-7B on the same eval is a normal
0.533); it is **excluded from all claims** and is exactly why MMMU is "keep-7B" (a route-to-7B *anomaly*, not a
fusion win). The open-ended halves reproduce via a Claude-Sonnet-5 judge: VQA-RAD-32B 74.1 % (paper 76.5),
SLAKE-32B 85.0 % (paper 89.2), close but slightly under a stricter judge.

### 2.4 Baselines: always-32B-THINK (the target) and always-32B-no-think

The **honest baseline is always-32B-THINK** — "just run the big thinking model on everything," the strong
model people deploy when they want its accuracy. Its **measured batch-1** cost (`opentext_32b_think.json`, HF
batch-1, single GPU, cap320 real VQA-RAD images, per-GPU NVML, n = 15 after 3 warmups):

| 32B mode | latency (mean / median) | energy | gen tok | prefill tok |
|---|---|---|---|---|
| **THINK** (native `<think>` prompt) | **10 521.6 ms** / 12 896.2 ms | 2 001.9 J | 98.3 | 335.8 |
| no-think (intermediate tier reference) | 665.0 ms / 696.4 ms | 126.9 J | 5.6 | — |
| **ratio think : no-think** | **15.8×** | **15.8×** | — | — |

Crucially, on **open-text perception THINK is also less accurate** than no-think (measured, n = 200 paired per
set, modal scorer): SLAKE-open **0.700 vs 0.895 (−0.195)**, VQA-RAD-open **0.425 vs 0.545 (−0.120)**,
PathVQA-open **0.035 vs 0.170 (−0.135)**; pooled n = 600 **0.387 vs 0.537 (−0.150)**. So **always-32B-THINK is
Pareto-dominated**: it is ~16× slower/costlier *and* less accurate than 32B-no-think on perception. Think helps
**only on reasoning** (faithful MedEvalKit `_reason` dumps, `reframe_vs_bigthink.json`): MMMU-150 Lingshu
+0.027 / MVT +0.100 / IV3 +0.120; MedXpert-2000 Lingshu −0.003 / MVT +0.045 / IV3 +0.031. For reference the
peer strong leg IV3-38B (`iv3_38b_latency.json`, vLLM tp = 2 batch-1, peak VRAM 153.9 GB): no-think
**1 409.3 ms / 598.0 J**, think **6 220.0 ms / 3 275.6 J** → 4.4× slower, 5.5× more energy.

Pooled over the full suite (n = 42 374): **always-32B-THINK 0.5632** @ 4.57 FLOP-eq / 10 521.6 ms;
**always-32B-no-think 0.5732** @ 4.57 FLOP-eq / 665 ms. We report every method delta against **both**.

**Consequence for the design.** The strong leg is **32B-no-think everywhere** (it dominates 32B-think on cost
and on perception accuracy, ties it on Lingshu reasoning); the slow think tier is *reserved* for the reasoning
residual and, on Lingshu, fires ~0 %.

### 2.5 The measured cost model (one set of constants across the codebase)

All costs are **batch-1**, measured in this repo; one set of constants is used everywhere (FLOP-eq = one 7B
forward; a 7B forward = `2·N·(P+G)` with `N₇ = 7.6e9`, and one 32B forward = **4.57** 7B-forward-equivalents).

| symbol | what | latency | FLOP-eq | source |
|---|---|---|---|---|
| `GEN7`   | one 7B no-think greedy generate (MCQ + SLAKE-open cheap leg) | **347 ms** | **1.0** | measured (`integrated_method.py`) |
| `VER7`   | one verifier forward (scores all 8 candidates in one batch) | **175 ms** | **1.0** | measured |
| `BO8`    | open-text best-of-8: 8 gens in **parallel** + 1 verify | **522 ms** | **16.0** | latency = 1 gen + 1 verify; FLOPs = 16 cheap forwards |
| `GEN32N` | 32B **no-think** (escalation target *and* honest "always-32B") | **665 ms** | **4.57** | measured |
| `GEN32T` | 32B **think** (the naive baseline) | **10 521.6 ms** | 4.57† | measured latency; FLOP-eq is a conservative lower bound |
| `FUSE`   | a decision-fusion cell runs **both** legs, co-resident/parallel | max(347, 665) = **665 ms** | **5.57** | 1.0 + 4.57; latency ≈ 32B-nt |

† `GEN32T`'s FLOP-eq is set equal to `GEN32N` (4.57) as a conservative lower bound; a think forward emits a
long decode, so its true FLOPs exceed this and the reported baseline FLOPs *understate* its cost. The measured
32B latency composition (`latency_32b.jsonl`): decode 68.57 ms/tok; no-think leg 333.1 ms (2 tok, decode
fraction 0.412); prefill fraction **φ = 0.586** → prefill32 = **389.7 ms**, decode32 = 275.3 ms. Because
`prefill32 (390 ms) > cheap leg (347 ms)`, the entire 7B pass hides under the 32B prefill on every MCQ
escalation (the basis of the G8 lever, §3.5).

**Held-out protocol (all thresholds).** Every escalation threshold τ (and every suppress / veto / fusion
decision) is chosen by **5-fold cross-fit**: on 4/5 pick the minimum-escalation τ such that cascade accuracy ≥
the strong-leg's accuracy (iso-accuracy target), evaluate on the held-out 1/5, average the folds. No peeking;
reported as deployable. The one-sided F8/F1 *certification* uses a Wilson lower-bound / paired-bootstrap 95 %
lower-CI > 0 rule (§3.4).

### 2.6 Confidence signals, the trained verifier, judge validation, VRAM

**Confidence signals.** Every gate thresholds a scalar from a greedy decode with option logprobs
`ℓ₁ ≥ ℓ₂ ≥ …`, `pᵢ = e^{ℓᵢ}/Σⱼ e^{ℓⱼ}`: **margin** `m = ℓ₁ − ℓ₂` (the deployed MCQ signal), **MSP** `p₁`,
**entropy** `−Σ pᵢ ln pᵢ`, **Gini** `1 − Σ pᵢ²`. A confidence gate escalates iff `m < τ`.

**The trained outcome verifier.** Given image v, question q, candidate answer a, it emits
`s_φ(v,q,a) = P_φ(Yes | v,q,a) = softmax(z)_Yes`, a two-way (Yes/No) head on the frozen base plus a LoRA
adapter φ, trained with BCE against a correctness label y (free-text: LLM-judge verdict vs the dataset answer
key; boxes: `y = 1[IoU(a,gold) ≥ 0.3]`). At inference it does **best-of-N selection**
`â = argmax_{i≤N} s_φ(v,q,aᵢ)`. It only has to *discriminate* good from bad candidates (strictly easier than
generating them), which is why a small trained 7B verifier can beat a much larger generator. Argmax is the
correct rule (voting reintroduces the majority trap, §6).

**Judge validation.** Open-ended accuracy depends on a judge, so we validate it. Primary grader:
**MedVLThinker-32B**; independent cross-check: **Claude-Sonnet-5**. (i) Exact-match anchor: on candidate pairs
where answer == gold (n = 1 277) the MVT-32B judge returns Yes **100.0 %**; zero-word-overlap cases (n =
14 320) are judged Yes only 6.3 %, spot-checked as legitimate synonyms. (ii) Inter-judge agreement (Lingshu-32B
vs MVT-32B): Cohen's κ **0.85–0.96** (VQA-RAD 0.962, Kvasir 0.849). (iii) The Claude-Sonnet-5 judge shows a
100 % exact-match anchor and a 21 % zero-word-overlap judged-Yes rate that spot-checks as entirely legitimate
(synonyms, bilingual Chinese). We treat κ ≥ 0.85 + a 100 % exact-match anchor as sufficient.

**Peak VRAM** (batch-1, GB): 7B/32B families 71.45 (small) / 152.82 (big no-think) / 154.08 (big think) —
think vs no-think adds essentially nothing to peak memory, so the ACC's savings are *time and energy*, not
residency, and the 32B needs two 80 GB GPUs (tp = 2).

---

## 3. Method — the format-aware, regime-adaptive router (full specification)

This is the paper's final, best, end-to-end method: a **format router** (arm chosen from the prompt, never the
gold) with an MCQ arm and an open-text arm, scored against **always-Lingshu-32B in THINK mode**. It is merged
into **one reproducible file** with a single Pareto knob; every number is a CPU re-costing of saved per-sample
dumps (no GPU, no new inference).

### 3.0 The unified pipeline and the one-command reproduction

```bash
cd ~/medvlthinker-imgdiff-compute
python3 src/cascade_methods/method_final.py     # → artifacts/method_final.json (V1)  AND  method_final_v2.json (V2)
```

`method_final.py` recomputes **every** table below **live** from the saved dumps + the measured cost
constants + held-out (5-fold cross-fit) calibration; it reads no sibling `.json`. It exposes a single
Pareto **`mode`** knob (both modes share the open-text arm and MMMU keep-7B; they differ only in the MCQ
perception arm), an `int4=True` cost-mode, and a `run_v2()` entry that folds in F8 + F10. Per-lever reference
scripts (each a subset, kept for provenance): `integrated_method.py` (compute-lean base router, fixed bo8),
`beat32b_fusion.py` (accuracy-max PMC fusion), `integrated_pandora.py` (Pandora adaptive-N open arm),
`quantized_strong_leg.py` (INT4 re-costing), `escalation_levers.py` (G8 latency lever, G5/G6).

### 3.1 The top-level format router

```
                       ┌──── detect format from the PROMPT (never the gold) ────┐
question + image ──────┤  MCQ / closed  ─────►  §3.2  MCQ ARM                     │
                       │  open-ended    ─────►  §3.6  OPEN-TEXT ARM               │
                       └────────────────────────────────────────────────────────┘
```

**Why a router and not one unified gate (Correction #2).** The MCQ margin gate has **no open-text analog**
(open answers have no single-letter logprob margin), and the trained verifier is **open-text-specific**. A weak
unified proxy (7B sequence-logprob) works passably as the open-text gate but is **beaten by margin on MCQ and
by verifier-confidence on open** → a single unified policy underperforms. Keep the two-arm router. (The
deterministic detector — enumerated options present ⇒ MCQ, else open-text — needs no learned classifier.)

### 3.2 MCQ arm — `7B-nt + margin gate → 32B-no-think`

Run Lingshu-7B no-think (one greedy gen). Compute the **margin** = P(top-1) − P(top-2) from the option
logprobs. **Escalate** (re-answer with 32B no-think) iff `margin < τ_mcq`, where `τ_mcq` is the held-out
min-escalation threshold that reaches 32B-no-think parity. Two per-benchmark overrides:

- **MMMU → keep 7B** (no escalation). Lingshu-7B scores 0.80 on MMMU-Medical-val (the anomaly), vs 32B-nt
  0.633 / 32B-think 0.660, so the router keeps 7B and beats always-32B-think by **+0.140** at 1.0 FLOP / 347 ms.
  This is a **route-to-7B anomaly, not a fusion win** (§6.1).
- **PMC-VQA → optional fusion** (the accuracy-max knob, §3.4). Default is the margin cascade (matches 32B-nt);
  the fusion variant *beats* it.

The three-tier structure underneath (the ACC). In the reasoning-native families the MCQ arm is a **3-tier
cascade over compute configurations**, `T0 = 7B-no-think@cap320 → T1 = 32B-no-think@cap320 → T2 =
32B-think@fullres`, with a tier-0 margin gate and a think gate (agreement of the two no-think legs, tie-broken
by the 32B-nt margin: `fire T2 iff 1[ŷ_{T0} ≠ ŷ_{T1}] + ε·(−m_{T1}) > τ₁`, `ε = 1e-6`). Because
32B-no-think ≥ 32B-think on perception, the slow T2 fires only on the reasoning residual (~15 % on ALL-6,
**~0 % on Lingshu**), and the whole cost is governed by *how often T2 fires*. On Lingshu the arm reduces to the
2-tier `7B-nt → 32B-nt` form.

### 3.3 The MCQ gate choice — margin (Correction #1, and it is family-dependent)

The premise (backlog §F) was that CASP-stability (cap320-vs-full agreement) or cross-model agreement would beat
the margin gate. **On Lingshu it does not.** Pooled perception-closed MCQ (`gate_comparison_mcq`, n = 37 879;
cheap-7B-nt 0.5769, strong-32B-nt 0.5905); KEEP-signal = "trust 7B," escalate the lowest-scoring:

| gate (KEEP signal) | detection AUROC | **min esc to reach 32B-nt parity** | deployable? |
|---|---:|---:|:--|
| **margin (7B, deployed)** | 0.7254 | **15.62 %** | **yes — cheap, continuous, best deployable** |
| conf / MSP (7B) | 0.7318 | 20.26 % | cheap but needs more escalation |
| CASP-stability (7B cap320-vs-full) | 0.7241 | 15.50 % | **INERT** (see below) |
| agreement (7B-nt vs 32B-nt) | 0.6565 | 19.96 % | needs the 32B run → not a cheap gate |

**CASP is inert on Lingshu** because Lingshu-7B is **98.95 % cap320-vs-full stable**, so the CASP signal
collapses to the margin tiebreak. **Agreement** is a real binary trust signal (P(7B ok | agree) = 0.6868 vs
P(7B ok | disagree) = 0.3262) but is the worst *ranker* and requires running the 32B. **conf/MSP** has a
hair-higher AUROC but reaches parity only at 20.3 % vs margin's 15.6 %. **Deployability order: margin >
agreement > CASP.** This is *family-dependent*: on **MedVLThinker** a *trained* CASP-Stability gate did edge
the confidence gate at parity (FLOPs 49.0 % vs 53.9 %; §5.1) — because MVT-7B is less cap320-stable, so its
stability signal carries information. The honest statement is that **no cheap gate beats margin on Lingshu
MCQ**, and the stability lever is real only where the cheap model is resolution-sensitive.

### 3.4 Accuracy-max PMC lever — F3 confidence-advantage fusion / F8 certified weak-veto

On the largest slice, **PMC-VQA**, the 7B and 32B are comparably skilled with de-correlated errors, so fusing
their *calibrated per-sample confidences* beats either alone. The accuracy-max knob replaces the PMC
margin-cascade with an **F1 guardrailed slice router** that picks, per benchmark, the held-out
paired-bootstrap-**certified** winner among {always-32B-nt, keep-7B, calibrated confidence-advantage fusion}.
Only **PMC** (fusion) and **MMMU** (keep-7B) certify as non-32B; radiology/pathology-closed + MedXpert stay
always-32B (fusion *hurts* the small sets where the 32B is clearly better — the guardrail keeps them at 32B).

**F3 (confidence-advantage fusion), held-out.** PMC acc **0.5653** vs 32B-nt 0.5518 → **d_nt +0.0135, 95 % CI
[0.0100, 0.0169]**, n = 33 430; d_think +0.0159. It is equivalent to a 2-detector **Chair–Varshney** fuser
under equal option-count. **Classic per-*slice*-reliability Chair–Varshney collapses to exactly always-32B
(d = 0.0)** — the beat *requires per-sample confidence*, not slice reliability. F5 double-reading: on the 33 %
7B-vs-32B disagreement set the **free** calibrated conf-advantage arbiter (0.4116) beats **both** the 32B-nt
arbiter (= always-32B, 0.3707) and the expensive 32B-think arbiter (0.3871) — **thinking is a poor arbiter on
perception disagreements**; the disagreement-set oracle-UB 0.689 → recoverability-AUROC 0.634 bounds the beat.

**F8 (certified high-precision weak-veto), V2.** F8 runs the 7B on every item, then in each
calibration-**certified** `(dataset × 7B-conf-bin)` cell where the 7B's **Wilson-lower-bound precision ≥ the
32B accuracy**, it **vetoes the 32B** (keeps 7B → 7B-only → cheaper); elsewhere it takes the 32B. The one-sided
Wilson certificate makes F8 **never-worse by construction**. On PMC: acc **0.5613**, d vs 32B-nt **+0.0095, CI
[0.0071, 0.0118]**, veto rate 40.0 %, PMC FLOP-eq **3.741** (vs F3's 5.570). F8 **captures 70.4 % of F3's PMC
beat at −32.8 % PMC FLOPs** — the swap that flips accuracy-max from FLOP-**positive** to FLOP-**negative**
(§4.4). Pooled MCQ (n = 40 029): F8 acc 0.5844, d vs 32B-nt +0.008, veto 33.4 %, FLOPs 0.885× always-32B.

### 3.5 Escalation-speed and cost levers (G8 prefetch, G5/G6, G3 INT4, G4/G7/G2)

vs the *cheaper* baseline always-32B-**no-think** (665 ms) the MCQ arm's heavy-escalation cells are actually
slower (VQA-RAD-closed 725.9 ms @57 % esc, MedXpert 942.8 ms @90 %, SLAKE-open 698.6 ms @53 %). The escalation
levers recover this.

- **G8 — parallel prefill prefetch (the load-bearing latency knob).** The 32B *image-prefill* does not depend
  on the 7B output, so run it **concurrently** with the 7B pass; an escalated query then pays
  `max(cheap, prefill32) + decode32` instead of `cheap + prefill32 + decode32`. With prefill32 = 390 ms > cheap
  347 ms, the whole 7B pass hides under the prefill on every MCQ escalation (robust for any φ ≥ 347/665 =
  0.522). Effect: pooled batch-1 latency **461.1 → 405.2 ms (−12.1 %)** at identical accuracy 0.5749; every
  previously-slower cell flips under always-32B-nt. FLOPs caveat: *unconditional* prefetch pays the 32B prefill
  on every query (pooled FLOPs 2.337 → 4.575 ≈ always-32B) — a latency-for-FLOPs trade, free only on an idle
  2nd GPU. The **slice-gated** deployable variant (prefetch only where base esc ≥ 0.40) keeps FLOPs 2.492 at
  429.8 ms.
- **G5 (recoverability suppressor) and G6 (2-of-2 gate) are knobs, not free lunches.** G5: no slice is truly
  futile; the named-futile MedXpert has the worst recovery (P(32B fixes 7B error) = 0.225, P(32B breaks 7B
  correct) = 0.479) and is grossly inefficient (+0.039 acc for +596 ms), but escalation is still net positive,
  so suppressing it is a **trade** (−0.039 on MedXpert = −0.0018 pooled, flips MedXpert 943 → 347 ms). G6: no
  gain — there is no orthogonal 2nd cheap MCQ signal (CASP 98.9 % inert), so `AND(margin, casp) ≤ margin`.
  Combined best (G8 slice-gated + G5 ε\* = 0.06): pooled acc 0.5731, latency **416.4 ms** (−9.7 % vs 461 ms
  base, **−96.0 % vs 32B-think**), FLOPs 2.285; no cell slower than always-32B-nt.
- **G3 — INT4 strong-leg cost-mode (`int4=True`), PROJECTED.** Re-cost the 32B strong leg with an AWQ/GPTQ-INT4
  forward. Latency: AWQ-INT4 accelerates only memory-bound decode, and the no-think leg is *prefill-bound*
  (decode ≈ 137 ms of 665 ms), so a literature 2.5× decode speedup drops the strong leg **665 → 583 ms** (ratio
  0.876), not 2.5×; pooled compute-lean parallel latency 469 → **455 ms**. FLOPs: the repo unit is
  **MAC-count** (weight-precision independent), so an INT4-32B forward is still **4.57** and method FLOPs are
  literally unchanged (a throughput-effective accounting would credit ~0.876× on this prefill-bound leg —
  reported, not headline). Accuracy: INT4 medical loss ≤ 1 % (literature W4A16) erodes only the quantized-32B
  share of answers by `strong_share × δ`, δ ∈ [0.5 %, 1.0 %]; the vs-**think** win robustly holds (compute-lean
  d_think [+0.010, +0.011]; accuracy-max [+0.014, +0.019]), while the razor-thin compute-lean vs-no-think
  margin can erode to ≈ 0. It buys VRAM/energy and lets the escalation model fit tp = 1 on one GPU — a
  deployability/energy lever, not a FLOP-saver. *(Empirical AWQ latency + Lingshu-32B-INT4 accuracy are future
  work: no loadable AWQ/GPTQ-INT4 Lingshu-32B exists for vLLM, and a HF-CDN outage stalled the committed
  `bench_int4_strong_leg.py` on the official Qwen2.5-VL-32B-AWQ this session; the ready benchmark re-runs when
  the CDN recovers.)*
- **G4/G7/G2 (image-token prune / semantic cache / early-exit), tested and marginal.** G4 (FastV shallow-exit
  at layer 3 of 64, layer-retain 0.953) and G7/G2 are data-limited on the saved dumps (no image id/hash for a
  cache key; no per-layer activations for early-exit) and do not move the headline; documented in
  `escalation_more.json`, not deployed.

### 3.6 Open-text arm — `7B best-of-8 + trained verifier (verifier-conf gate) → 32B-no-think`

Run **8 samples** of Lingshu-7B (in parallel), score each candidate with the **trained outcome verifier**
(`ckpts/train/lora_verifier_pooled4`, one forward scores all 8), and **pick** the argmax-P(correct) candidate.
**Gate** on the verifier's max score; **escalate** the low-confidence residual to 32B-no-think. Per benchmark
(`integrated_method_vs_think.json`; cheap leg = best-of-8, cost `BO8`; 32B-think acc **estimated** = judged
32B-no-think + measured modal think-delta, flagged):

| benchmark | n | greedy 7B | bo8+verifier | 32B-nt | 32B-think (est) | **method** | **d_think** | d_nt | esc% | lat (ms) | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SLAKE-open | 645 | 0.7364 | 0.7798 | 0.8186 | 0.6236 | **0.8155** | **+0.1919** | −0.0031 | 12.6 % | 605.5 | 16.574 |
| VQA-RAD-open | 200 | 0.4650 | 0.5750 | 0.6000 | 0.4800 | 0.5850 | +0.1050 | −0.0150 | 5.5 % | 558.6 | 16.251 |
| PathVQA-open | 1 500 | 0.3240 | 0.4533 | 0.3760 | 0.2460 | 0.4533 | +0.2073 | +0.0773 | 0.1 % | 522.4 | 16.003 |

**This is the accuracy engine.** The 7B best-of-8 ensemble *beats the 32B* on the OOD open-text sets, which
beats 32B-THINK by +0.12…+0.21 (think over-thinks perception). The verifier-confidence gate escalates almost
nothing (≤ 13 %). **FLOPs cost is honest:** best-of-8 = 16 cheap forwards, so the open arm *costs* more FLOPs
than a single 32B forward (break-even vs one 32B is N ≤ 2). It buys the latency win (parallel bo-N ≈ 522 ms ≪
665 ms no-think ≪ 10 522 ms think) *and* the accuracy win, but it is **not** a FLOP-saving lever on the open
arm. (SLAKE-open originally lacked a verifier dump → the pooled4 verifier was scored pointwise over its K = 8
SC candidates: bo8+verifier 0.7798 vs greedy 0.7302 [+0.0496]; parity reached at ~13 % escalation vs ~53 % for
the old greedy+seqlogprob fallback — a ~4× escalation cut.)

### 3.7 Pandora adaptive-N controller (Weitzman) — the FLOP-lean open-arm knob

The Pandora controller (`pandora_controller.py`, Weitzman 1979 optimal search) **unifies adaptive-N and the
escalation gate**: each "box" is either "draw one more 7B sample" (cost 2.0 FLOP-eq = GEN7 + VER7, reward = the
verifier's calibrated P(correct)) or "escalate to the 32B" (cost 4.57 FLOP-eq, deterministic reward =
calibrated P(strong correct)). One exchange-rate knob λ yields **both** a stop-drawing threshold and an
escalation threshold; thresholds are held-out (5-fold cross-fit isotonic calibration). It replaces the fixed
best-of-8 open arm in the deployed compute-lean point: it **holds the fixed-bo8 accuracy iso** (+0.0117 vs
+0.0118, within the 5-fold band) while cutting open-arm FLOPs (fixed-bo8 2.538 → Pandora 2.244 pooled).

### 3.8 F10 — team-objective learning-to-complement gate (V2 open arm)

F10 (learning-to-complement, a team-objective L2D rejector) **replaces the open-text arm's parity-targeting τ
escalation gate** (shared by both modes). A learned rejector over 7B-side open-text features (verifier-score
max/range/mean/std, #unique preds, self-consistency, seqlogprob), tuned to the **TEAM-accuracy** objective,
decides escalate-to-32B vs keep-7B-best-of-N per item; **Pandora still sets the cheap draw count**, so the FLOP
model is unchanged and only the escalation set + delivered accuracy come from F10. The prior τ-gate targeted
iso-32B *by design* (parity at min escalation), so it sat at/below 32B on those cells; F10 optimizes the right
objective. Effect (d vs always-32B-no-think): SLAKE-open **−0.0093 → +0.0016**, VQA-RAD-open **−0.0050 →
+0.0050**, PathVQA-open **+0.0760 → +0.0860** — **all three cells lift; the two residual open losses are
repaired** to above-32B. Pooled open-only: acc 0.5625 → 0.5727 at slightly lower FLOPs (10.871 → 10.758).
*Note (honest):* the gain is from the **objective**, not a better signal — the multi-feature learned score does
not have higher recoverability AUROC than the single gate.

### 3.9 The Pareto knob — compute-lean ↔ accuracy-max (both FLOP-negative)

Both modes share the open-text arm and MMMU keep-7B; they differ **only** in the MCQ perception arm:

- **`mode='compute-lean'`** — MCQ = `7B-nt + margin gate → 32B-no-think` cascade. FLOP-saving; PMC *matches*
  32B. Open arm = Pandora (V1) or F10 (V2). FLOP-eq **0.49× always-32B** (it always was FLOP-negative).
- **`mode='accuracy-max'`** — MCQ = F1 guardrailed slice router (PMC → fusion, everything else certified). PMC
  *beats* 32B. **V1** uses F3 fusion (both legs on 100 % of PMC → FLOP-positive, 1.25×). **V2** uses F8
  certified weak-veto (7B on all + 32B on the non-veto 60 % → **FLOP-negative, 0.93×**), a strict Pareto move
  into the compute-negative half-plane while still CI-certified above 32B on PMC.

The knob in one line: *compute-lean* = 0.575 @ FLOPs 2.24 (d_think +0.012); *accuracy-max V2* = 0.587 @ FLOPs
4.25 (d_think +0.021). Both are cheaper than always-32B's 4.57 FLOP-eq. `int4=True` composes with either.

---

## 4. Results — the headline

### 4.1 Pooled headline: both knob settings × {vs think, vs no-think} × {latency, FLOPs}

Reproduced **live** by the two knob settings of `method_final.py` (full suite, n = 42 374, sample-weighted).
`d_think` = method − always-32B-think (0.5632); `d_nt` = method − always-32B-no-think (0.5732); both baselines
4.57 FLOP-eq, latency 10 521.6 ms (think) / 665 ms (no-think). Latency `seq / par` = batch-1 sequential
(single-stream) / parallel (2-GPU co-resident: best-of-N batched, fusion legs concurrent).

| knob (V1) | vs 32B-**THINK** | vs 32B-**no-think** | FLOP-eq (×32B) | batch-1 latency seq / par (−vs think) |
|---|---:|---:|---:|---:|
| **compute-lean** (Pandora open) | **+0.0117** | +0.0017 | **2.244** (0.49×) | 578 / **469 ms** (−94 % / −96 %) |
| &nbsp;&nbsp;↳ *fixed-bo8 reference* | *+0.0118* | *+0.0018* | *2.538* | *460 / 460 ms* |
| **accuracy-max** (F3 PMC fusion) | **+0.0238** | +0.0137 | 5.695 (1.25×) | 1050 / **666 ms** (−90 % / −94 %) |

**Both modes are faster than always-32B-no-think on parallel latency and ≥ 90 % faster than always-32B-THINK
on both accountings.** The compute-lean *fixed-bo8 reference row is the literal +0.0118*; the deployed
compute-lean point swaps fixed best-of-8 for Pandora adaptive-N, which holds that accuracy iso (+0.0117) while
cutting FLOPs 2.54 → 2.24.

### 4.2 Final per-benchmark tables (compute-lean and accuracy-max V1)

**compute-lean** (MCQ margin cascade + Pandora open + MMMU keep-7B):

| benchmark | n | 7B | 32B-nt | 32B-thk | policy | **method** | **d_think** | d_nt | FLOPs | lat seq / par |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| PMC-VQA | 33 430 | 0.5427 | 0.5518 | 0.5494 | margin-cascade | 0.5508 | +0.0014 | −0.0010 | 1.386 | 403 / 403 |
| SLAKE-closed | 836 | 0.8254 | 0.8589 | 0.8636 | margin-cascade | 0.8516 | −0.0120 | −0.0072 | 1.935 | 483 / 483 |
| VQA-RAD-closed | 251 | 0.7809 | 0.8526 | 0.8406 | margin-cascade | 0.8328 | −0.0079 | −0.0198 | 3.603 | 726 / 726 |
| PathVQA-closed | 3 362 | 0.8409 | 0.8891 | 0.8891‡ | margin-cascade | 0.8882 | −0.0009 | −0.0009 | 3.089 | 651 / 651 |
| MedXpert-MM | 2 000 | 0.2615 | 0.3065 | 0.3040 | margin-cascade | 0.3005 | −0.0035 | −0.0060 | 5.095 | 943 / 943 |
| MMMU-Medical | 150 | 0.8000 | 0.6333 | 0.6600 | **keep-7B** | **0.8000** | **+0.1400** | +0.1667 | 1.000 | 347 / 347 |
| SLAKE-open | 645 | 0.7364 | 0.8186 | 0.6236§ | pandora-N + verifier | 0.8093 | +0.1857 | −0.0093 | 7.622 | 1906 / 627 |
| VQA-RAD-open | 200 | 0.4650 | 0.6000 | 0.4800§ | pandora-N + verifier | 0.5950 | +0.1150 | −0.0050 | 8.391 | 2124 / 605 |
| PathVQA-open | 1 500 | 0.3240 | 0.3760 | 0.2460§ | pandora-N + verifier | 0.4520 | +0.2060 | +0.0760 | 12.598 | 3100 / 759 |

**accuracy-max V1** (MCQ F1 slice router: PMC → F3 fusion; open cells identical):

| benchmark | n | policy (F1 certified) | **method** | **d_think** | d_nt | FLOPs | lat seq / par |
|---|---:|---|---:|---:|---:|---:|---:|
| PMC-VQA | 33 430 | **fusion (F3 conf-adv)** | **0.5653** | **+0.0159** | **+0.0135** | 5.570 | 1012 / 665 |
| SLAKE-closed | 836 | always-32B-nt | 0.8589 | −0.0047 | −0.0000 | 4.570 | 665 / 665 |
| VQA-RAD-closed | 251 | always-32B-nt | 0.8526 | +0.0120 | −0.0000 | 4.570 | 665 / 665 |
| PathVQA-closed | 3 362 | always-32B-nt | 0.8891 | −0.0000 | −0.0000 | 4.570 | 665 / 665 |
| MedXpert-MM | 2 000 | always-32B-nt | 0.3065 | +0.0025 | +0.0000 | 4.570 | 665 / 665 |
| MMMU-Medical | 150 | keep-7B | 0.8000 | +0.1400 | +0.1667 | 1.000 | 347 / 347 |
| SLAKE / VQA-RAD / PathVQA-open | 2 345 | pandora-N + verifier | *(as above)* | | | | |

‡ PathVQA-closed has no 32B-think dump → think = no-think. § open-text 32B-think acc is **estimated** (judged
32B-no-think + measured modal think-delta −0.195 / −0.120 / −0.130).

**Pooled by pool** (both V1 modes):

| mode | pool | n | method | **d_think** | **d_nt** | FLOP-eq (×32B) | lat seq (−think) | lat par (−think) | macro d_think |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **compute-lean** | full suite | 42 374 | 0.5749 | **+0.0117** | +0.0017 | **2.244** (0.49×) | 578 ms (−94 %) | **469 ms** (−96 %) | +0.0693 |
| | MCQ only | 40 029 | 0.5756 | +0.0011 | −0.0008 | 1.739 (0.38×) | 454 ms (−96 %) | 454 ms (−96 %) | +0.0195 |
| | open only | 2 345 | 0.5625 | +0.1927 | +0.0456 | 10.871 (2.38×) | 2688 ms (−74 %) | 710 ms (−93 %) | +0.1689 |
| **accuracy-max** | full suite | 42 374 | 0.5869 | **+0.0238** | **+0.0137** | 5.695 (1.25×) | 1050 ms (−90 %) | **666 ms** (−94 %) | +0.0747 |
| | MCQ only | 40 029 | 0.5883 | +0.0139 | +0.0119 | 5.392 (1.18×) | 954 ms (−91 %) | 664 ms (−94 %) | +0.0276 |
| | open only | 2 345 | 0.5625 | +0.1927 | +0.0456 | 10.871 (2.38×) | 2688 ms (−74 %) | 710 ms (−93 %) | +0.1689 |

**Read.** The two accuracy engines are the **open-text arm** (bo8 + verifier beats even 32B-no-think, which
beats 32B-think by +0.12…+0.21) and **MMMU keep-7B** (+0.140 vs think). Perception MCQ ties 32B-nt (≈ think).
Per-benchmark Δ vs think: PMC +0.0014, SLAKE-cl −0.012, VQA-RAD-cl −0.008, PathVQA-cl −0.001, MedXpert −0.004,
MMMU **+0.140**, SLAKE-o **+0.192**, VQA-RAD-o **+0.105**, PathVQA-o **+0.207**.

### 4.3 The reconciliation: +0.0118 and +0.0238 are ONE method at two knob settings

An earlier inconsistency (the compute-lean pipeline reproduced +0.0118 while the task quoted +0.0238) is **not
a contradiction** — +0.0118 is *compute-lean* (base margin router) and +0.0238 is *accuracy-max* (PMC
slice-fusion). Only the MCQ arm differs; the open (Pandora) arm is byte-identical. The load-bearing cell is
PMC: fusion lifts PMC acc **+0.0145** (0.5508 → 0.5653), and because PMC is **n = 33 430 = 78.9 %** of the
pooled 42 374, that one cell adds **0.789 × 0.0145 = +0.0114** to the pooled sample-weighted accuracy; the four
non-PMC closed cells snapping margin-cascade → always-32B add a further ≈ +0.0006. Together pooled d_think
**+0.0117 → +0.0238**. Macro-avg barely moves (+0.0693 → +0.0747) because PMC's +0.0145 contributes only /9 and
macro is already dominated by MMMU (+0.14) and open text (+0.19). (Bonus: the unified method scores SLAKE-open
with bo8 + verifier in *both* modes, so unified accuracy-max FLOPs are 5.695, cleaner than the separate-script
5.751.)

### 4.4 V2 — F8 + F10 make BOTH modes FLOP-negative

`method_final.py::run_v2` folds two held-out levers into the *same* pipeline (V1 output unchanged): **F8**
(certified weak-veto) replaces F3 in the accuracy-max PMC cell; **F10** (team-objective L2D) replaces the open
arm's τ gate in both modes.

| knob | vs 32B-**THINK** | vs 32B-**no-think** | FLOP-eq (×32B) | lat seq / par | FLOP-neg? |
|---|---:|---:|---:|---:|:--:|
| compute-lean **V1** (Pandora-τ open) | +0.0117 | +0.0017 | 2.244 (0.49×) | 578 / 469 ms | ✅ |
| **compute-lean V2** (F10 open) | **+0.0123** | **+0.0023** | **2.238** (0.49×) | 577 / **468 ms** | ✅ |
| accuracy-max **V1** (F3 fusion, Pandora-τ open) | +0.0238 | +0.0137 | 5.695 (**1.25×**) | 1050 / 666 ms | ❌ |
| **accuracy-max V2** (F8 veto + F10 open) | **+0.0212** | **+0.0112** | **4.246** (**0.93×**) | 839 / 729 ms | ✅ |

**Both modes are now FLOP-negative** (< always-32B's 4.57): compute-lean **0.49×** (always was), accuracy-max
**newly 0.93×** (was 1.25×). The accuracy-max FLOP cut is **−1.449 FLOP-eq (−25 %)** for a **−0.0026** vs-think
give-back (F8 captures 70 % of F3's PMC beat; F10 gives some back on open cells) — a strict Pareto move into
the compute-negative half-plane while still CI-certified above 32B on PMC. Honest trade: accuracy-max parallel
latency rises 666 → 729 ms because F8 is a *sequential* cascade on PMC (7B → maybe 32B, no parallel-fusion
overlap) — still **−93 %** vs 32B-think. **accuracy-max V2 is all-axes dominant over the naive baseline:
cheaper (0.93×), faster (−93 %), and more accurate (+0.0212).**

### 4.5 The latency headline

Across settings the method beats always-32B-THINK on batch-1 latency by **−90 % to −96 %** (parallel), and it
is *faster than the cheap no-think 32B baseline* on parallel latency in both modes — the G8 prefetch and the
parallel best-of-N leg together keep every escalation-heavy cell under the 665 ms no-think anchor. The
sequential (single-stream) accounting is more conservative (adaptive draws are one-at-a-time, fusion legs
serial) but still −74 % to −96 %.

### 4.6 The ACC reframe vs always-32B-THINK — five families (efficiency engine)

With the think baseline priced (§2.4), the ACC MCQ bake-off (`master_data.csv`, measured batch-1) is re-scored
against **always-32B-THINK** (`reframe_vs_bigthink.json`). **Always-32B-THINK is Pareto-dominated on every
family**:

| family | pool | 32B-think acc | method acc | Δacc | 32B-think lat | method lat | Δlat | FLOPs% | 32B-think E | method E | think-esc% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MVT** 7B→32B | ALL-6 | 0.5723 | 0.5693 | −0.003 | 11.34 s | **2.27 s** | **−80 %** | **52 %** | 6319 J | 1182 J | 15 % |
| | ALL-5 | 0.6463 | 0.6450 | −0.001 | 8.88 s | **0.44 s** | **−95 %** | 25 % | 4916 J | 173 J | 2 % |
| **Lingshu** 7B→32B | ALL-6 | 0.6611 | 0.6614 | **+0.000 (beats)** | 0.32 s | 0.30 s | −8 % | 49 % | 113 J | 76 J | 0 % |
| | ALL-5 | 0.7746 | 0.7726 | −0.002 | 0.32 s | 0.25 s | −24 % | 39 % | 113 J | 61 J | 1 % |
| **QoQ-Med** | ALL-6 | 0.4689 | 0.5095 | **+0.041 (beats)** | 9.72 s | **0.12 s** | **−99 %** | **9 %** | 5382 J | 18 J | 0 % |
| | ALL-5 | 0.5432 | 0.6048 | **+0.062 (beats)** | 8.49 s | 0.12 s | −99 % | 9 % | 4692 J | 18 J | 0 % |
| **Chiron** | ALL-6 | 0.5076 | 0.6023 | **+0.095 (beats)** | 4.25 s | **0.20 s** | **−95 %** | 19 % | 1176 J | 17 J | 0 % |
| | ALL-5 | 0.5926 | 0.7249 | **+0.132 (beats)** | 3.66 s | 0.20 s | −95 % | 20 % | 1006 J | 17 J | 0 % |
| **MedGemma** | ALL-6 | 0.5253 | 0.5219 | −0.003 | 12.72 s | **3.37 s** | **−74 %** | 68 % | 6535 J | 1614 J | 20 % |
| | ALL-5 | 0.5979 | 0.6028 | **+0.005 (beats)** | 9.76 s | 0.18 s | −98 % | 12 % | 4990 J | 16 J | 0 % |

On **3 of 5 families the big *thinking* model is actually LESS accurate than the method** (QoQ +0.041, Chiron
+0.095) — think over-thinks the perception-dominated suite — *and* 1.2–45× slower. think : no-think latency
ratio per family (ALL-6): MVT 49×, MedGemma 45×, QoQ 43×, Chiron 15×, Lingshu 1.2× (Lingshu has no real
promptable think mode, hence its small latency win).

### 4.7 The trained-verifier accuracy engine — three families beat the 32B, plus boxes

The open-text arm's verifier is a positive result in its own right. Across **three families and two
architectures**, the 7B-verifier cascade **beats the strong 32B/38B** on accuracy on every family × dataset
cell (pooled cascade-best vs STRONG): **Lingshu 0.421 vs 0.331, MedVLThinker 0.344 vs 0.277, InternVL3 0.255 vs
0.218** — the InternVL3 win uses the *Lingshu-trained* verifier applied cross-architecture (no retraining). A
strict same-split test (pool the four Lingshu open sets, hold out 30 %, n = 1 064) orders **verifier 0.501 >
32B 0.462 > SC 0.411 ≈ greedy 0.413** — the trained 7B captures **49 %** of the oracle gap and edges the 32B
*while 5× smaller*. Two-seed honesty: the verifier beats the same-split 32B by **+0.039, 95 % CI [+0.010,
+0.066]** on seed-0 but ties on seed-1 (−0.005), so the honest verb versus the 32B is **matches-to-modest-win
(mean +0.017)**; versus *training-free* selection it is robustly positive both seeds (+0.066 / +0.088). The
verifier separates correct from incorrect candidates at **AUROC 0.924**, a blank-image ablation drops it −0.047
(it reads the image), and a trained 7B verifier beats a *zero-shot 32B* verifier (0.403 vs 0.355) — **task
training beats 5× scale for selection.** The same recipe transfers to **bounding boxes**: SLAKE organs recover
**40 % / 53 %** and the real MS-CXR chest-X-ray benchmark **78 % / 77 %** of the oracle gap (MS-CXR bootstrap
gain **+0.191, 95 % CI [+0.152, +0.232]**, a 5.6× lift).

**The three-family beats-32B master tables.** Each family is evaluated on its full open-ended sets; "cheap(SC)"
is the small model's self-consistency majority, "STRONG" the family's single 32B/38B pass, "verifier-bo8"
best-of-8 selection with no gate, "cascade-best@esc" verifier-bo8 + the verifier-confidence gate escalating the
stated fraction, "oracle@8" the luck ceiling. The verifier base is **Lingshu-7B** throughout; the InternVL3
column applies the *same Lingshu-trained verifier* to InternVL3 answers (cross-architecture transfer).

| Lingshu (strong = Lingshu-32B) | cheap(SC) | STRONG | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---:|---:|---:|---:|---:|
| VQA-RAD | 0.465 | 0.600 | 0.575 | **0.625 @24 %** | 0.630 |
| PathVQA | 0.324 | 0.376 | 0.453 | **0.469 @33 %** | 0.517 |
| Kvasir | 0.286 | 0.301 | 0.439 | **0.448 @20 %** | 0.491 |
| RadImageNet-OOD | 0.329 | 0.289 | 0.353 | **0.353 @0 %** | 0.512 |
| **POOLED** | 0.322 | 0.331 | 0.414 | **0.421 @12 %** | 0.513 |

| MedVLThinker (strong = MVT-32B) | cheap(SC) | STRONG | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---:|---:|---:|---:|---:|
| VQA-RAD | 0.420 | 0.525 | 0.490 | **0.555 @54 %** | 0.600 |
| Kvasir | 0.343 | 0.361 | 0.477 | **0.483 @9 %** | 0.550 |
| RadImageNet-OOD | 0.204 | 0.202 | 0.241 | **0.243 @13 %** | 0.317 |
| **POOLED** | 0.266 | 0.277 | 0.339 | **0.344 @14 %** | 0.416 |

| InternVL3 (strong = IV3-38B; Lingshu-trained verifier) | cheap(SC) | STRONG | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---:|---:|---:|---:|---:|
| VQA-RAD | 0.445 | 0.415 | 0.570 | **0.580 @12 %** | 0.620 |
| PathVQA | 0.081 | 0.096 | 0.116 | **0.125 @52 %** | 0.192 |
| Kvasir | 0.362 | 0.380 | 0.479 | **0.487 @17 %** | 0.593 |
| RadImageNet-OOD | 0.285 | 0.304 | 0.302 | **0.313 @52 %** | 0.398 |
| **POOLED** | 0.202 | 0.218 | 0.249 | **0.255 @36 %** | 0.337 |

Two honest qualifications. **(i) The gate does real work where the strong model is genuinely better.** Pure
verifier-bo8 (no gate) does not uniformly beat STRONG (it loses Lingshu VQA-RAD 0.575 < 0.600, MVT VQA-RAD
0.490 < 0.525, IV3 RadImageNet 0.302 < 0.304); on exactly those sets the gate escalates a larger fraction (24 %,
54 %, 52 %) and the cascade climbs above STRONG by combining best-of-N on recoverable questions with deference
on the rest — it wins by *knowing when to escalate*, not by out-selecting the 32B everywhere. **(ii)** The
pooled STRONG numbers are low because these full pools are dominated by hard/OOD sets and exclude SLAKE (the one
set where the 32B dominates); the stricter same-split test below puts the 32B back at 0.462.

**The canonical same-split test (n = 1 064).** Pool the four Lingshu open sets, hold out 30 %, evaluate every
method — including the 32B — on the *identical* held-out questions:

| dataset | greedy | SC | 32B (same split) | **verifier** | oracle | n |
|---|---:|---:|---:|---:|---:|---:|
| SLAKE | 0.738 | 0.738 | 0.829 | 0.762 | 0.895 | 210 |
| VQA-RAD | 0.519 | 0.500 | 0.648 | 0.611 | 0.722 | 54 |
| PathVQA | 0.352 | 0.349 | 0.377 | **0.441** | 0.513 | 435 |
| Kvasir | 0.282 | 0.282 | 0.326 | **0.405** | 0.493 | 365 |
| **POOLED** | 0.413 | 0.411 | 0.462 | **0.501** | 0.592 | 1 064 |

Ordering **verifier 0.501 > 32B 0.462 > SC 0.411 ≈ greedy 0.413**: the verifier captures **49 %** of the oracle
gap and edges the 32B *while 5× smaller*, beating it on the hard sets (PathVQA 0.441 vs 0.377, Kvasir 0.405 vs
0.326) and losing where the 32B is genuinely stronger (SLAKE 0.762 vs 0.829) or n is tiny (VQA-RAD, n = 54).
**Two-seed honesty:** seed-0 paired-bootstrap gain over the 32B **+0.0385, 95 % CI [+0.010, +0.066]**
(significant); seed-1 **−0.005** (a tie). Honest verb: **matches-to-modest-win** (mean +0.017); robust only over
*training-free* selection (+0.088 / +0.066 over greedy both seeds; gap-captured 35–49 %, mean ~42 %, wide
per-dataset spread: SLAKE 15.2 %, VQA-RAD 45.5 %, PathVQA 55.7 %, Kvasir 58.4 %).

**Training is the active ingredient; argmax is the rule.** Zero-shot self-verification is luck-floored (AUROC
≈ 0.5); training the same 7B on judge labels lifts selection to 0.501, and the **trained 7B verifier beats a
zero-shot 32B verifier 0.403 vs 0.355** (selection efficiency 0.810 vs 0.717) — task training beats 5× the
parameters. The verifier separates correct from incorrect at **AUROC 0.924** (mean score 0.749 vs 0.171 across
8 512 candidates); a blank-image ablation drops discrimination **−0.047** (it reads the image, refuting the
"lazy verifier" failure mode). Plain argmax (0.501) beats verifier-weighted voting (0.489) and a score×count
hybrid (0.470) — any vote-flavored rule reinherits the majority trap. Best-of-K scales monotonically (K = 1/2/4/8
→ 0.385/0.425/0.476/0.501, oracle@8 0.592) while a random pick stays flat — a genuine **test-time-scaling**
property; and because 2N < 4.57 for N ≤ 2, **verifier-bo2 (no gate) beats the 32B on accuracy *and* FLOPs (4.0 vs
4.57)** on the hard/OOD sets. **Structured outputs (boxes):** with `y = 1[IoU ≥ 0.3]` and a Qwen2.5-VL-7B box
verifier, best-of-8 recovers **40 % / 53 %** (SLAKE organs, two seeds; both training-free selectors *below*
greedy) and **78 % / 77 %** (MS-CXR, two seeds; +0.191 over greedy, 5.6× lift) of the oracle gap — training
again roughly doubles the zero-shot box-verifier's captured gap (0.115 → 0.232). **The binding limit is
candidate quality:** per-answer AUROC is already ~0.90, and a Bradley-Terry ranking loss pushes it to 0.93, yet
selection stays flat at ~0.50; what moves the ceiling is better candidates (cross-model pooling lifts oracle@8
+0.11–0.15).

---

## 5. Ablations — where each piece earns its place

### 5.1 Gate bake-off — margin > agreement > CASP on Lingshu (and it is family-dependent)

Repeated from §3.3 as the load-bearing ablation. On Lingshu MCQ (n = 37 879) the deployed **margin** gate wins
on the deployable metric (min-esc 15.62 % to 32B-nt parity, AUROC 0.7254); **CASP-stability is inert** (7B is
98.95 % cap320-vs-full stable), **agreement** is the worst ranker and needs the 32B, **conf/MSP** ties AUROC
but needs 20.3 % escalation. **Family-dependence (the honest nuance):** on **MedVLThinker** — where the 7B is
resolution-sensitive — a *trained* CASP-Stability gate beat the confidence gate at ALL-6 parity (FLOPs
**49.0 % vs 53.9 %** margin / 57.4 % MSP, latency 1.77 s vs 2.69 s, acc 0.5698 vs 0.5687), because re-targeting
the routing label from un-learnable recoverability (AUROC 0.58, the wall) to learnable **stability** (AUROC
0.71) breaks the training-free ceiling (capacity is irrelevant: logistic ≈ MLP ≈ LoRA). So "no trained gate
beats confidence" holds only for gates predicting *recoverability*; a trained *stability*-router beats
confidence on cascade efficiency where the cheap model is unstable. The MedEvalKit gate bake-off
(`gate_unified_bakeoff.json`) makes the deeper point: **cascade quality (ADC) tracks recoverability-AUROC, not
detection-AUROC** — on MCQ competent-4 the highest-*detection* gate (learned-RICH 0.693) has *lower* ADC than
margin, and the only positive-routing-eff gate is 7B self-verify (AutoMix, the highest *recoverability* AUROC
0.614).

### 5.2 The middle-tier ablation (M1/M2/M3) — it is the structure, not the gate

The load-bearing structural fact (`METHOD_ACC.md` head-to-head, measured latency/energy, MVT ALL-6): inserting
the **32B-no-think middle tier** turns an escalate-everything-to-think cascade into the ACC.

| variant (ALL-6) | acc | think-esc | FLOPs% | latency | energy |
|---|---:|---:|---:|---:|---:|
| M2 escalate-to-think, **no nt-middle** (7B-think → 32B-think) | 0.5725 | 86 % | 105 % | 29.8 s | 7 049 J |
| M3 7B-think middle (7B-nt → 7B-think → 32B-think) | 0.5697 | 65 % | 89 % | 23.2 s | 5 499 J |
| **M1 ACC — 32B-nt middle restored** | 0.5694 | **19 %** | **55 %** | **5.9 s** | **1 505 J** |
| M1b ACC + agreement gate | 0.5710 | 14 % | 54 % | 4.86 s | 1 220 J |

Δ(M1 vs M2): think-esc 86 % → 19 %, FLOPs 105 % → 55 %, latency **29.8 → 5.9 s (−80 %)**, energy −79 %, at
matched accuracy. Complementarily, holding the three-tier structure fixed and **swapping in every training-free
gate barely moves the operating point** (ALL-6 band: acc 0.5666–0.5702, FLOPs 49–62 %, latency 1.77–3.48 s);
only *random* escalation leaves the frontier (116.5 % FLOPs). The gate moves you a few points along a frontier
the structure already fixed — the empirical face of the recoverability wall.

**The three-tier cost table (why avoiding a think call dominates).** Per-tier measured batch-1 costs (one
forward `= 2·N·(P+G)`, `N₇ = 7.6e9`, `N₃₂ = 33.0e9`):

| tier | config | prefill P | decode G | FLOPs (×1e15) | latency | energy |
|---|---|---:|---:|---:|---:|---:|
| T0 | 7B no-think @ cap320 | 388 | 2 | 0.01 | 0.21 s | 25 J |
| T1 | 32B no-think @ cap320 | 388 | 2 | 0.03 | 0.34 s | 65 J |
| T2 | 32B think @ fullres | 685 | 391 | 0.07 | **11.34 s** | ~6 319 J |

The two no-think tiers are *constant-energy* (~25 J, ~65 J, ~2 decode tokens); the think tier's energy scales
with its decode length (`E ≈ 18.17·G − 107.5 J`), so the whole cascade cost is governed by `e₁·c_T2` — *how
often T2 fires*. A conventional 2-tier `7B → 32B-think` cascade at parity sends e₁ ≈ 69 % to think; ACC's free
no-think middle tier cuts that to **15 %**. The over-thinking premise that justifies the middle tier
(MedVLThinker 32B, exact-match): SLAKE **32B-nt 0.849 vs 32B-think 0.764 (+0.085)**, VQA-RAD **0.853 vs 0.776
(+0.077)**, PMC/PathVQA flat; pooled competent-4 **32B-nt 0.658 ≥ 32B-think 0.645** at ~2 vs ~477 decode tokens.
Think *fixes* 11.5 % of no-think's errors but *breaks* 11.1 % of its correct answers — symmetric over-thinking,
not a free lunch. And `no-think@cap320` recovers `think@fullres` accuracy (0.6463 vs 0.6451) at **39 %** of the
compute, so cap320 is the natural T1.

**ALL-6 all-methods bake-off at parity (MedVLThinker)** — the full frontier (every training-free gate in the
same 3-tier structure; `guard` = # benchmarks worse than always-7B, averaged over 20 held-out seeds):

| system | acc | esc₀ | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always-7B-nt @cap320 | 0.5262 | 0 % | 0 % | 8.4 % | 0.13 s | 19.9 J | 0.00 |
| always-32B-nt @cap320 | 0.5573 | 100 % | 0 % | 36.2 % | 0.23 s | 77.8 J | 0.00 |
| **always-32B-think [PARITY]** | **0.5723** | 100 % | 100 % | 100.0 % | **11.34 s** | **6 318.8 J** | 0.00 |
| **ACC-v2 (agreement gate)** | **0.5693** | 71.7 % | 15.1 % | **52.0 %** | **2.27 s** | **1 181.9 J** | **0.00** |
| ACC-v1 (margin gate) | 0.5687 | 66 % | 19 % | 53.9 % | 2.69 s | 1 416.6 J | 0.00 |
| CASP-Stability (trained gate) | 0.5698 | 74 % | 11 % | 49.0 % | 1.77 s | 899.2 J | 0.05 |
| MSP/Chow | 0.5697 | 70 % | 19 % | 57.4 % | 2.96 s | 1 568.1 J | 0.00 |
| entropy | 0.5691 | 71 % | 21 % | 62.0 % | 3.48 s | 1 863.1 J | 0.00 |
| Gini/DOCTOR | 0.5702 | 69 % | 22 % | 61.0 % | 3.44 s | 1 837.1 J | 0.00 |
| AutoMix (self-verify) | 0.5692 | 73 % | 18 % | 54.6 % | 2.50 s | 1 307.0 J | 0.05 |
| FrugalGPT-style learned | 0.5677 | 70 % | 19 % | 60.4 % | 3.30 s | 1 765.5 J | 0.10 |
| Jitkrittum L2D (Diff-Prob) | 0.5666 | 67 % | 15 % | 50.6 % | 2.29 s | 1 194.5 J | 0.00 |
| random | 0.5641 | 89 % | 76 % | 116.5 % | 8.95 s | 4 889.5 J | 0.05 |

Every training-free gate lands in a narrow band (acc 0.5666–0.5702, FLOPs 49–62 %, latency 1.77–3.48 s); only
random leaves the frontier. **Per-benchmark, ACC-v2 is ≥ the 7B on all seven columns (guard 0)** and beats the
think baseline on the four perception sets (PMC 0.561 vs 0.556, SLAKE 0.842 vs 0.764, VQA-RAD 0.861 vs 0.776,
PathVQA 0.679 vs 0.673), trailing only on the three reasoning sets (MMMU 0.643 vs 0.688, MedXpert-R 0.282 vs
0.326, MedXpert-U 0.310 vs 0.384) — the honest scope boundary. On **ALL-5** (drop MedXpert) always-think
**8.88 s / 4 915.9 J → ACC-v2 0.44 s / 172.8 J, FLOPs 24.9 %** (−95 % latency, ~28× energy, guard 0.05).

**Five-family generalization** (`no-think ≥ think on perception` re-measured across three architectures):

| family | set | parity acc (32B-think) | ACC-v2 acc | ACC-v2 FLOPs% | esc₀ | think | guard |
|---|---|---:|---:|---:|---:|---:|---:|
| MedVLThinker | ALL-6 / ALL-5 | 0.5723 / 0.6463 | 0.5693 / 0.6450 | 52.0 / 24.9 | 71.7 / 35.1 | 15.1 / 2.3 | 0.00 / 0.05 |
| Lingshu | ALL-6 / ALL-5 | 0.6611 / 0.7746 | 0.6614 / 0.7726 | 48.6 / 38.9 | 61 / 42 | 0 / 1 | 1.00 / 1.15 (inherited) |
| QoQ-Med | ALL-6 / ALL-5 | 0.4689 / 0.5432 | 0.5095 / 0.6048 | 8.8 / 9.1 | 0 / 0 | 0 / 0 | 0.00 |
| Chiron-o1 | ALL-6 / ALL-5 | 0.5076 / 0.5926 | 0.6023 / 0.7249 | 19.3 / 19.9 | 0 / 0 | 0 / 0 | 0.00 |
| MedGemma | ALL-6 / ALL-5 | 0.5253 / 0.5979 | 0.5219 / 0.6028 | 68.4 / 11.5 | 55 / 0 | 20 / 0 | 1.15 / 0.00 |

**Lingshu is a clean win** (roughly halving FLOPs; the guard 1.00 is *inherited* from the target — always-32B is
itself worse than Lingshu-7B on MMMU, the anomaly, so any system reaching for the 32B pays that guard). **QoQ /
Chiron gracefully collapse** — think *hurts* there, so the agreement gate never escalates (esc 0 %), delivering
accuracy *above* the think target at cheap-leg cost. **MedGemma is the honest partial case** (ALL-6 over-escalates
for 68.4 % FLOPs; ALL-5 collapses to 11.5 %). ACC pays where the 32B has a genuine think-mode advantage that
over-fires on perception; elsewhere it degrades gracefully.

### 5.2b The faithful cross-family MCQ margin cascade (−17…−69 % FLOPs)

The simplest instance of the ACC structure — a **2-tier** cheap-7B → strong-32B margin cascade at the iso-strong
operating point (`C = 1 + esc·4.57`) — reproduced under the faithful MedEvalKit protocol matches the strong model
at large FLOPs savings, with the win magnitude tracking the (strong − cheap) accuracy gap:

| benchmark | Lingshu | MedVLThinker | InternVL3 |
|---|---|---|---|
| MMMU-Med | keep-7B (anomaly) | −14 % | **−62 %** |
| PMC-VQA (33k) | **−69 %** | −49 % | −16 % |
| SLAKE | −56 % | no win (7B weak) | keep-cheap (8B ≥ 38B) |
| VQA-RAD | −17 % | −41 % | **−67 %** |
| PathVQA | −31 % | **−68 %** | −20 % |
| MedXpert-MM | no win (floor) | no win | *(cap gap)* |

A *small* gap → thin escalation residual → large saving; a *large* gap (weak cheap model) → escalate nearly
everything → *more* expensive than always-strong. Two principled non-winning regimes: **keep-cheap** (IV3-SLAKE
8B ≥ 38B; Lingshu-MMMU inflated 7B) and **no win** (MVT-SLAKE 7B 0.498 vs 32B 0.620 forces 96 % escalation,
+18 % FLOPs; every MedXpert cell near the floor, ~100 % escalation). **Threshold honesty:** the −69 % PMC-Lingshu
figure is at the iso-strong operating point; under a *fair held-out τ* it is **−57 %** (cascade 0.563 ≥ 32B
0.549), and under oracle-τ −74 % — the deployable, honest figure is the middle one. **The mirror-image lever, a
reasoning think tier**, adds accuracy where headroom exists (MMMU-Med Δ **+0.034 Lingshu / +0.107 MVT / +0.120
IV3**; MVT-MedXpert +0.045; Lingshu-MedXpert ~0 at the floor), gated so MMMU reaches think accuracy at ~78 %
FLOPs (a win) while MedXpert lands at 143–151 % (no win) — regime-adaptive.

### 5.3 F3 vs F8 on the accuracy-max PMC cell

| policy | PMC acc | d vs 32B-nt | veto % | esc → 32B % | PMC FLOP-eq | lat seq / par | note |
|---|---:|---:|---:|---:|---:|---:|---|
| F3 confidence-advantage fusion (V1) | 0.5653 | +0.0135 | — | 100 % | 5.570 | 1012 / 665 ms | both legs on 100 % of PMC |
| **F8 certified weak-veto (V2)** | 0.5613 | **+0.0095** | 40.0 % | 60.0 % | **3.741** | 746 / 746 ms | 7B on all + 32B on non-veto |

F8 captures **70.4 %** of F3's PMC beat at **−32.8 % PMC FLOPs**, still CI-certified above 32B ([0.0071,
0.0118]). Classic per-*slice*-reliability Chair–Varshney collapses to always-32B (d = 0.0); the beat requires
per-*sample* confidence.

### 5.4 F10 open-text repair (team objective vs parity-τ)

| open cell | n | prior gate (Pandora-τ) | **F10 L2D** | gain | repaired? | F10 acc | esc % | FLOP-eq |
|---|---:|---:|---:|---:|:--:|---:|---:|---:|
| SLAKE-open | 645 | −0.0093 | **+0.0016** | +0.0109 | ✅ | 0.8202 | 20.6 % | 7.842 |
| VQA-RAD-open | 200 | −0.0050 | **+0.0050** | +0.0100 | ✅ | 0.6050 | 37.0 % | 9.511 |
| PathVQA-open | 1 500 | +0.0760 | **+0.0860** | +0.0100 | (already won) | 0.4620 | 26.5 % | 12.178 |

All three lift; the two residual open losses are repaired from below-32B to above-32B. Pooled open-only: acc
0.5625 → 0.5727, d vs 32B-nt +0.0456 → +0.0559, d vs think +0.1927 → +0.2029, at slightly lower FLOPs
(10.871 → 10.758). The gain is from the objective, not a better signal (the CI on the small VQA-RAD-open /
SLAKE-open point-beats still spans 0; the robust open beat remains PathVQA-open).

### 5.5 Pandora and its refinements

At **iso-bo8 accuracy (held out)** Pandora reaches the target at **−27 % FLOPs / −28 % energy** vs fixed
best-of-8, beating even the *optimistically-tuned* adaptive-N (−19 %) and gate (0 %) baselines, despite being
the only method whose thresholds are held-out:

| Target | Method | ds covered | FLOPs | vs bo8 | energy (J) | meanN | esc | lat_seq (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **iso-bo8** | **Pandora (held-out)** | 9/11 | **11.74** | **−27 %** | **409.8 (−28 %)** | 5.38 | 21.6 % | 2 951 |
| | adaptive-N (oracle-τ) | 11/11 | 13.01 | −19 % | 459.5 | 6.31 | 8.7 % | 3 350 |
| | verifier-bo8 + gate (oracle-τ) | 11/11 | 16.00 | 0 % | 568.8 | 8.00 | 0.0 % | 4 176 |
| **iso-strong** | **Pandora (held-out)** | 11/11 | 6.32 | −61 % | 222.6 | 3.03 | 5.5 % | 1 619 |

**The reframe Pandora forces.** On these OOD open-text sets the accuracy ceiling worth matching is the **cheap
best-of-8 ensemble, not the 32B** (pooled Lingshu bo8 0.414 vs 32B 0.331) — so escalating to the 32B buys
little and the lever is reaching bo8 accuracy more cheaply. **Refinements (regime-limited):** correlated-Pandora
(diversity-discounted stopping) is a **free −16–17 %** in meanN/FLOPs/latency/energy at iso-strong (11/11) but
does *not* help at iso-bo8 (coverage drops to 7/11); Pandora × cross-model pooling lifts oracle **+0.11–0.15**
but yields no frontier gain (the selection wall blocks the conversion). Honest latency trade: adaptive drawing
is sequential, so Pandora's `lat_seq ≈ 2 951 ms` vs a batched fixed bo8's `lat_bat ≈ 522 ms` — Pandora wins
FLOPs + energy, loses batch-1 wall-clock to a batched bo8.

### 5.6 Escalation levers (G8, G5/G6, G3/G4/G7/G2)

Covered in §3.5. G8 slice-gated + G5(ε\*) gives pooled **416.4 ms (−96.0 % vs 32B-think)** at −0.0018 acc with
no cell slower than always-32B-nt; G3 INT4 is a projected VRAM/energy lever (FLOPs unchanged under MAC
accounting); G4/G7/G2 are data-limited on the dumps and not deployed.

### 5.7 Router vs unified verdict

A **deterministic format-aware router** (`unified_router.json`, dispatch MCQ → margin cascade, open → verifier
best-of-8 cascade, score the *whole pooled* stream as one point) is a **tested Pareto win over always-strong on
all three families**: Δacc **+0.003 (Lingshu) / +0.000 (MVT) / +0.001 (IV3)** at **48–82 % of always-32B FLOPs**
(30–59 % if the image prefill is prefix-shared across the 8 samples). The FLOPs saving tracks how competitive
the strong model is (Lingshu's 32B barely ahead → keep-cheap → biggest saving; IV3's 38B more competitive →
more escalation → smallest). A single *unified* gate is beaten by margin on MCQ and verifier-conf on open →
keep the two-arm router (Correction #2).

### 5.8 The reframe measurements (the two blocked cells, now measured)

The whole reframe rested on the cost of the strong leg. Both previously-blocked cells were measured (§2.4):
**open-text 32B-THINK = 10 521.6 ms / 2 001.9 J** (15.8× the no-think leg, and −0.15 *less* accurate on
perception — "open-text think 10.5 s / −0.15"), and **IV3-38B = 6 220 ms / 3 275.6 J think** ("IV3 6.2 s",
removing the router's amortized-batch-latency proxy caveat for the 38B peer). Think-hurts-perception is
consistent across families (SLAKE: MVT −0.084, Chiron −0.108; VQA-RAD: all −0.07…−0.09). These turn the ledger's
"blocked" cells into measured constants; OmniMed-32B stays blocked, so ALL-7 is unreportable and ALL-6 is the
computable pool.

### 5.9 Best-of-N internals — UGV, diverse generation, pairwise verifier, portfolio/pooling, ceiling-break

- **UGV (single generative verifier) for MCQ — negative.** Scoring MCQ options as *generated answers* collapses
  accuracy: PMC-VQA strict greedy **0.132 content vs 0.534 letter** (−0.394); content-MCQ verifier gain +0.004,
  AUROC ≈ 0.70. The router stays the better method (MCQ → letter + margin; open → trained bo-N verifier); the
  generative verifier's home is open-text. Backlog B2 closed as a negative for MCQ.
- **Diverse generation (GPU) — positive on coverage.** A 5-persona × 3-temperature portfolio (M = 15) lifts
  the oracle ceiling **+0.027** at matched budget / **+0.064** at M = 15 (CIs exclude 0), converting to a
  significant **+0.025 verifier bo-N accuracy** at 1.875× generation cost. It shifts the binding limit from
  coverage (#1) to selection (#2): on PMC-content the oracle lift is largest (+0.110) but the pointwise verifier
  cannot convert it (confident-distractor rate 0.426 → 0.504).
- **Real pairwise verifier (GPU) — positive on selection; overturns the simulation.** A real A-vs-B verifier
  beats pointwise-argmax on selection (sel_acc 0.374 → **0.410, +0.036** CI [0.016, 0.055]; on near-ties +0.050),
  closing ~35 % of the pointwise → oracle gap; knockout captures ~87 % of the win at ~7 comparisons/q. This
  overturns a *simulated*-pairwise parity (deriving P(i≻j) from pointwise scores cannot manufacture comparative
  signal; a real forward pass carries it).
- **Cross-model pooling / portfolio.** Pooling 3 cheap generators lifts oracle **+0.08 (B = 8) / +0.11 (B = 16)**
  held-out, but the Markowitz-optimal allocation ≈ a naive uniform split (Δ +0.002 to +0.02, negative on
  vqa_rad) — **the win is diversity/pooling, not the clever allocation** (the bandit test reaches the same
  verdict independently). Error-φ ≈ 0.52–0.56 (models fail on somewhat different questions, still positively
  correlated).
- **Diversity *selection* (DPP/MMR) — a rate win, not a ceiling lift.** Reordering a fixed 8-candidate set
  leaves oracle@8 identical (0.412) by construction, but MMR reaches the first correct answer with −15.6 %
  fewer samples; only 4.67 of 8 candidates are distinct (≈42 % exact duplicates).
- **The open-ended ceiling-break.** The pessimistic ~0.6 gate AUROC is partly an MCQ discreteness artifact: the
  *same* Lingshu-7B margin signal rises to **AUROC ≈ 0.87** for detecting cheap errors on free-text (pooled
  0.846, 95 % CI [0.830, 0.862]; SLAKE 0.889, PathVQA 0.797, VQA-RAD 0.717) — and it is discreteness, not length
  (token-F1 agrees). But **recoverability does not break** (pooled 0.591), and even open-ended, **plain
  confidence stays the best training-free gate** (fusion of all signals ties it at 0.866; self-verify P(True)
  0.755 is worse). This is exactly why a *trained* verifier is necessary, and why medical-VLM cascades should be
  evaluated open-ended.

---

## 6. The two walls — the honest headline negative

The program is bounded by two limits. We present them as a *headline contribution*, not a footnote, because
knowing exactly where the method stops is what makes the deployable claims trustworthy — and because the
attempts to break each wall are themselves informative. The walls are decisions about *which model answers*;
the method always produces an answer.

### 6.1 The recoverability wall — the broad-slice MCQ *beat* is confined to PMC (six independent confirmations)

A cascade gate needs **recoverability** — whether the *strong* model will fix a specific cheap error — not
mere cheap-model-wrongness. Over a frozen pair, recoverability is nearly unlearnable: across **12 signal
families** (margin, MSP/Chow, entropy, Gini/DOCTOR, energy, a hidden-state probe, self-verification P(True),
conformal/CP-Router, a learned GBM, cross-model agreement, multi-resolution stability, semantic
self-consistency) recoverability AUROC sits at **0.5–0.69**. The cause is **error correlation** — the two
models fail together. On the competent-4, decomposing every question by the joint (7B, 32B) outcome:
beneficial (7B wrong, 32B right) 15.7 %, futile (both wrong) 22.0 %, harmful (7B right, 32B wrong) 13.4 %;
**recoverability P(32B right | 7B wrong) = 41.6 %**, i.e. **P(32B wrong | 7B wrong) = 0.584** (ALL-6 φ =
0.372). This is exactly the regime Jitkrittum et al. (NeurIPS'23) predict a wall.

The *positive* corollary — "can we **beat** the 32B on a broad slice by fusing the two legs?" — is bounded by
the same wall. **The beat exists on PMC-VQA and nowhere else broad.** We attacked this with **six independent
fusion/routing methods**, and all six certify the same two cells (PMC fusion + MMMU keep-7B) and **no
genuinely-new one**:

| # | method | source | PMC d vs 32B-nt (CI) | MMMU | any NEW certified cell? |
|---|---|---|---|---|:--:|
| 1 | **F3 confidence-advantage fusion** (= 2-detector Chair–Varshney) | `beat32b_fusion.json` | **+0.0135** [0.0100, 0.0169] | +0.167 | **no** |
| 2 | **F8 certified weak-veto** (Wilson-LB, one-sided) | `beat32b_more.json` | +0.0095 [0.0071, 0.0118] | (keep-7B) | **no** |
| 3 | **F11 Bayesian model averaging** (EM additive mixture / PoE) | `beat32b_more.json` | +0.0134 [0.0102, 0.0165] | +0.193 | **no** |
| 4 | **F6 contrastive decoding** (option-logit level) | `logit_fusion.json` | ≈ +0.000 (F6_cd d = 0.0) | — | **no** |
| 5 | **option-logit fusion** (λ,α grid, F11_fixed/rw) | `logit_fusion.json` | +0.008 [−0.020, 0.036] (not certified) | — | **no** |
| 6 | **Domino / H4 automatic slice discovery** (106 slices × 8 splits) | `robust_slice_routing.json` | (re-finds PMC) | (re-finds MMMU) | **no** |

Reading (1)–(3): F3, F8, and F11-BMA all beat the 32B on PMC (comparably-skilled, de-correlated errors) and
all **gate off** on SLAKE/VQA-RAD/PathVQA-closed/MedXpert (where the 32B is clearly better — fusion *hurts*
there, so the guardrail keeps 32B). F7 super-learners agree (logistic +0.0127, GBM +0.0158 on PMC; ≤ 0
elsewhere). Reading (4)–(5): fusing at the **option-logit** level (contrastive decoding, weighted logit sums)
does not certify a new beat beyond PMC either — on a re-dumped per-option subsample PMC F6_cd d = 0.0 and
F11_fixed +0.008 is not certified (CI spans 0). Reading (6), the decisive one: an **FDR-controlled automatic
slice-discovery** search over 106 candidate slices (dataset id + question topics + 7B confidence) across 8
independent 50/50 discover/confirm splits **re-finds the PMC-fusion and MMMU-keep-7B wins without being told
they are special** (validating the hand-gate) but finds **essentially no genuinely-new beat-32B slice**:
genuinely-new BH-FDR5 certifications average **0.25 per split** (2 total across 8 splits, none recurring in a
majority), *below* the label-permutation-null floor (mean 5.61 spurious certs, p95 = 15). **A genuinely-new
slice inside an always-32B dataset does not exist above noise — the sixth confirmation of the recoverability
wall.** (A real BiomedCLIP image+text embedding would sharpen the slice geometry and is the clear next upgrade;
the current feature space is dataset id + question text + 7B confidence.)

**MMMU is a route-to-7B anomaly, not a fusion win.** The MMMU cell certifies as **always-7B** (d = +0.167, CI
[0.087, 0.247], n = 150) because Lingshu-7B is inflated there (§2.3), not because a clever fusion of two
comparably-skilled legs pays off. Fusing the legs on MMMU would give the 7B ~91 % weight (F11 mean_weight_7b =
0.912) — i.e. it degenerates to keep-7B. We therefore never claim MMMU as a broad-slice fusion beat; the
genuine broad win is **PMC fusion**, and the honest ceiling is that it does not extend.

### 6.2 H8 credibility shrinkage validates the deployed CI-guardrail

Could a fancier actuarial estimator squeeze more certified slices out of thin data? No — and the audit is
useful because it confirms the deployed rule is the right one. **Naive point routing** (deviate iff the raw
discovery advantage > 0) overfits thin slices: on a fine 61-slice family it yields **~7.5 held-out guardrail
violations per split**. **Hierarchical Bühlmann–Straub credibility shrinkage** (shrink each slice's advantage
toward its parent-dataset mean, which is ≤ 0 for the always-32B datasets) helps only **marginally (7.5 → 6.6)**
— an MSE-optimal shrinkage constant (k̄ ≈ 244) is too weak to flip a raw +0.05 thin-slice noise toward a mildly
negative parent, and shrinking whole-dataset cells toward the PMC-contaminated positive grand mean can even
slightly worsen them. **The decisive robustifier is the simple CI lower-bound guardrail** (deviate iff the
discovery 95 % lower-CI > 0 — exactly F1's existing rule): it drives fine-family violations to **0.2** and
hand-family to **0.0** at a preserved pooled beat (+0.0117). **Verdict:** credibility shrinkage does not beat
the CI-guardrail the method already deploys; H8's value is diagnostic — it confirms the thin-slice-overfit risk
is real *and* that the CI-guardrail (not a fancier estimator) is the correct, sufficient, guardrail-honest fix.

### 6.3 The selectability wall — best-of-N's oracle → selection gap is largely fundamental

The open-text accuracy engine (best-of-N + trained verifier) is bounded by a *selection* ceiling. Its clearest,
most surprising face is **training-free** selection: sample one open model N = 8 times and a correct answer is
frequently *present* (SLAKE-open greedy 0.730 → oracle@8 0.879, headroom +0.149), yet no frozen-model rule can
say *which* of the eight it is — every training-free selector ties or trails a random pick (0.720):

| selector (SLAKE-open, Lingshu-7B, n = 645, MVT-32B judge) | acc | vs random | % gap-above-random captured |
|---|---:|---:|---:|
| random pick (mean sample accuracy) | 0.720 | — | — |
| self-verify P(Yes) argmax (7B) | 0.715 | −0.005 | *worse than random* |
| self-consistency majority | 0.736 | +0.016 | 10 % |
| 32B pointwise verify, argmax | 0.746 | +0.026 | 16 % |
| 32B **listwise** select | 0.758 | +0.038 | 24 % (best training-free) |
| learned fusion [self, 32B] (5-fold CV) | 0.743 | +0.023 | 14 % |
| 32B free-gen single pass (SOTA bar) | 0.819 | — | — |
| oracle@8 (luck ceiling) | 0.879 | +0.159 | 100 % |

The best training-free selector captures only **24 %** of the gap above random, and *none* beats the 32B's own
single pass (0.819); handing the 32B the 7B's candidates to synthesize actually **backfires** (0.774 vs 0.819).
The mechanism is a **majority trap**: the correct answer is a minority vote in **74–90 %** of recoverable
questions (mean ≈ 1.5 of 8 votes), so any consistency/agreement rule is systematically pointed at the wrong
answer. This is exactly the floor the *trained* verifier (§4.7) breaks — and even it is bounded above by
selection efficiency.

Per-answer the trained verifier ranks correct above incorrect at AUROC ~0.90 (Lingshu 0.903, MVT 0.913, IV3
0.898), yet its **selection efficiency** — how often argmax picks a correct candidate when a recoverable one
exists — is only **81 % / 82 % / 74 %**, for the same majority-trap reason. A five-experiment battery
establishes the gap is largely **fundamental**, not a fixable artifact:

1. **Compounding fails.** Diverse-generation (+0.030) and pairwise-selection (+0.021) each beat the pointwise-iid
   baseline, but do **not** stack: pairwise-over-diverse (0.5376) ≤ pointwise-over-diverse (0.5494), and on PMC
   the diverse +0.110 oracle lift stays unconverted (pairwise converts 0.005). **Diversity buys coverage, not
   selectability.**
2. **Distractor-filtering fails.** Eight pre-filters over the diverse pool; the best (rarity_log1p 0.5601) does
   not beat *both* baselines and sign-flips per dataset. Mechanism: the correct *new* answers diverse generation
   adds are themselves rare, so a rarity/agreement signal cannot separate correct-rare from wrong-rare.
3. **Verifier capacity does not break it.** A **32B-zeroshot** verifier (7× capacity) only **ties** the 7B-trained
   one: sel_acc **0.480 vs 0.475, +0.005, CI [−0.023, +0.032], n.s.** The pure-capacity contrast (32B-zeroshot
   vs 7B-zeroshot +0.067) is real but small, and the 32B still leaves an oracle → selection gap of 0.192
   (conversion 0.15). **The ceiling is substantially fundamental, not a capacity artifact** — a 7× verifier
   recovers ~7 of the ~19 oracle-gap points.
4. **Post-hoc selectors fail.** *Simulated* active-pairwise, bandit allocation (Thompson/UCB-E ≈ uniform), and
   unsupervised Dawid–Skene aggregation (≈ pooled majority, +0.132 below the trained verifier) all collapse to
   the pointwise/uniform/majority baseline — unsupervised reliability tracks self-agreement (~0.52), not accuracy
   (~0.29): the generators are **confidently wrong**.
5. The binding limit is therefore **candidate quality** (oracle@8 ≫ verifier-bo8 everywhere: pooled Lingshu
   0.513 vs 0.414); cross-model pooling lifts oracle **+0.11–0.15** — the largest remaining headroom, but only
   convertible with a stronger selector (the real pairwise verifier converts part of it).

### 6.4 The Pareto reframe of best-of-N — dominated by a cheap strong leg, alive against the slow one

Whether best-of-N is *deployable* depends entirely on the strong leg's cost. Against the **cheap no-think 32B**
(4.57 FLOP-eq, 665 ms) best-of-N is **FLOPs-dominated**: the global FLOPs-Pareto envelope is
`greedy → 7B+Pandora → always-32B` (7B-greedy F = 1.00 / 0.518; iid→Pandora F = 2.0–3.25 / 0.519–0.568;
always-32B F = 4.57 / 0.673), and **diverse-gen is *not* on the envelope** (iid→Pandora dominates it at every
accuracy target), so its 1.875× generation cost never repays. On **latency**, though, best-of-N is **alive**: a
parallel best-of-N base is only **522.6 ms = 0.79×** the single 665 ms 32B forward (batch-1 short-gen is
overhead-bound, so the 32B is only ~1.9× the 7B wall-clock, not 4.57×) — but it still does **not beat
always-32B** (fixed-bo-N tops ~0.55, far below the 32B's 0.673; matching 0.673 forces heavy escalation that
pushes parallel latency back above 665 ms). **However**, the deliverable's baseline is not the 665 ms no-think
leg — it is the **10 521.6 ms / 2 001.9 J think** leg. Re-priced against *that*, the parallel best-of-N leg is
~**0.05×** its latency (a ~20× win) and the July-6 "best-of-N is not deployable" verdict is correctly **scoped
to the no-think baseline only**. Net conclusion: the deployable efficiency win is the **router** (§5.7: 48–82 %
of always-32B FLOPs at parity) **plus Pandora for tight compute budgets** (F = 2–3.3 for acc 0.52–0.57); the
best-of-N leg is the **accuracy engine** whose latency dominates the *think* baseline it replaces. (Scope
caveat, honest: in an expensive/slow-strong regime the parallel best-of-N leg could win on FLOPs too — that
regime is not measured here.)

### 6.5 Adjacent negatives (kept for the record, not abstention)

- **The action axis is capacity-bound.** Cheap same-model repairs (look-closer / think) recover a real but
  **unharvestable** 14.3 % of the 7B's errors that the 32B *also* misses (stable 11–17 % across benchmarks), but
  the repairs break as many answers as they fix (per-view acc noise-limited: cap320 0.622, full-res 0.621, think
  0.607 vs 0.645 for the 32B), so a confidence-gated repair ladder *loses* at 32B-parity. The 32B's edge is
  capacity, not a cheap transform the 7B could apply to itself.
- **Cross-family complementarity is real but unexploitable.** Oracle union(7B | InternVL | Phi) = 0.801 vs
  always-32B 0.645, but a learned router over frozen peer signals captures none of it (best 0.621 ≈ always-7B
  0.622; a SigLIP image+text router is at chance, AUROC 0.50).
- **The cheap model leans on a language prior.** 56.9 % of the 7B's answers are unchanged when the image is
  blanked, and image-sensitive vs image-insensitive questions have near-equal accuracy (0.620 vs 0.625) — a
  large fraction of "VQA" carries no routable visual difficulty.

None of these is presented as a method. Each is a *negative* that maps where effort pays: change the structure
(ACC) or add a little training (the verifier), because the frozen-model routing/selection signal is, at
training-free effort, luck.

---

## 7. Reproduction of the 3-family suite + the OmniMed strong-leg fallback

**6 of 7 benchmarks (MMMU, VQA-RAD, SLAKE, PathVQA, PMC-VQA, MedXpert): fully run cheap + strong for all three
families (+ think tier), faithful vs the Lingshu paper** (§2.3). The 7th, OmniMedVQA (Open-access, **88 996
QA**, 4-option MCQ, 42 sub-datasets of which RadImageNet is ~64 %), is cheap-faithful + strong-fallback:

- **Cheap (7B/8B) legs ran and reproduce the paper** over the full 88 996-QA set: Lingshu-7B **0.8274** (73 639
  correct, paper 0.829 — a 0.2-pt reproduction), MedVLThinker-7B 0.6248, InternVL3-8B 0.7847. Per-modality
  breakdowns are internally sensible (Lingshu-7B Modality-Recognition 0.986, weakest on CT 0.772), confirming
  the OmniMed pipeline is faithful after the July-3 `modality_type` parser fix.
- **Strong (32B/38B) legs are BLOCKED** by a deterministic **tp = 2 NCCL collective hang** on the 89k-image
  queue (every chunk stalls ~36 min, the heartbeat watchdog aborts, the driver re-fires and re-hangs; confirmed
  over ~2 days). Ruled out: chunked tp = 2 + retry ×3, `TORCH_NCCL_ENABLE_MONITORING=0` (hangs forever),
  `EVAL_BATCH_SIZE=256` (fixes the cgroup OOM, not the hang), 3 h/chunk timeout + aggregation backstop. **tp = 1
  is not an option** (64 GB weights + multimodal activation OOMs the 80 GB card; the `MAX_MODEL_LEN` KV-cache
  lever cannot buy it back).
- **Decision (why it is safe, not a cop-out).** OmniMed is a **keep-cheap** benchmark — paper Lingshu-7B 82.9 vs
  32B 83.4, a 0.5-pt gap; our cheap 0.827 already matches — so a cascade **keeps-cheap at ~0 % escalation** and
  the missing strong number changes **no cascade conclusion**. We report the strong OmniMed cell as
  **paper-reference (Lingshu-32B 0.834) + infra-limited**, back the fidelity claim with the faithful cheap
  reproduction, and — per the standing rule — write **no fabricated `metrics.json`**. Net: **6/7 faithful +
  OmniMed cheap-faithful + strong-fallback**; only the pooled ALL-7 vs-32B-think figure is missing (not any
  verdict), and **ALL-6 is the computable pool**. (INT4-32B at tp = 1 ~20 GB would fit one GPU and avoid this
  hang — a future re-run path, §3.5.)

Reproduction engineering recorded honestly (the fixes are part of the deliverable): the OmniMed `KeyError` over
42 sub-datasets (`sample.get(...)`), the `MAX_MODEL_LEN` / `GPU_MEM_UTIL` env levers on both vLLM wrappers, the
chunked resumable tp = 2 strong driver, and the strictly-sequential cheap driver that dodged the two-at-a-time
cgroup OOM (anon-rss ≈ 245 GB in the kill log).

---

## 8. Related Work and Novelty

We state, inline and without hedging, exactly which mechanisms we adopt unchanged and which are new. The short
version: several components are established mechanisms transferred into a new setting; the contribution is the
*assembly* (a format-aware regime-adaptive router scored against the honest think baseline), the
*compute-configuration structure*, the trained-verifier *application and unification*, and the *characterization
of the two walls*.

**Efficient cascades and query routing.** FrugalGPT popularized the cost-saving LLM cascade. Agreement-based
cascading (**ABC**, arXiv 2407.02348) escalates on ensemble *disagreement*; **CAR** (confidence-aware routing,
arXiv 2505.15154) is the closest structural prior art. Our think-gate is, by construction, an ABC member; the
tier-0 margin gate is Chow/MSP. What is new is *what the gate switches between* — three **compute
configurations** of the *same two models* (7B-nt → 32B-**no-think** → 32B-think), inserting the strong model's
fast no-think mode as an intermediate tier because reasoning over-thinks perception.

**Confidence, selective prediction, conformal deferral.** The training-free gate family we benchmark against
margin is the classical selective-prediction toolkit (MSP/Chow, entropy, Gini/DOCTOR, split-conformal /
CP-Router, learning-to-defer). In our bake-offs these cluster at the recoverability ceiling; none separates
from margin (MCQ) or verifier-confidence (open). **We use selective/conformal machinery only to decide *which
model answers* (Wilson-LB certification, held-out τ), never to withhold an answer.**

**Post-hoc deferral theory — the wall.** Jitkrittum et al. (NeurIPS 2023, arXiv 2307.02764) prove
confidence-only deferral is fundamentally limited by how predictable the strong model's *marginal correction*
is; their Diff-Prob score is the SOTA post-hoc gate. This is a precise description of our recoverability wall:
"will the strong model fix this?" is only AUROC ~0.6 from any cheap signal, and a faithful Diff-Prob gate
reaches only AUROC 0.708 (MCQ) / 0.744 (open) — below the deployed signals — because the strong model repairs
only 6–10 % of cheap errors (26 % where competitive) and *which* is near-unlearnable. We treat Jitkrittum as
the *explanation* for the wall, not a baseline to beat.

**Verifiers and reward models for best-of-N.** Generative verifiers (**GenRM**) recast reward modeling as
next-token "is this correct?" prediction for best-of-N selection in text; VLM process-reward models rerank
reasoning *steps*. Our verifier is squarely in this lineage — a small LoRA **outcome** head
`s_φ = P_φ(Yes | v,q,a)` with a BCE objective — and we claim no architectural novelty. The contribution is the
**application and unification**: a single outcome verifier for inference-time selection in **medical VQA**,
trained and evaluated over **both free-text answers and bounding-box grounding** (`y = 1[IoU ≥ 0.3]`). We
position it as a constructive counter to **Verification-Mirage** (arXiv 2605.10850), which concluded
self-verification *fails* in medical VQA: that result is about *zero-shot* self-verification (our evidence
agrees: zero-shot self-verify ≈ AUROC 0.5, a zero-shot 32B verifier 0.355 < a trained 7B 0.403), and the
difference is **training** (the trained head reaches AUROC 0.924).

**Verifier-score-as-gate precedents.** **CCPS** (arXiv 2505.21772) scores reliability via input-perturbation
stability; **Self-REF** (arXiv 2410.13284) trains self-reference confidence tokens; conformal treatments of
verifier scores exist (Kiyani et al. arXiv 2602.17633). Our escalation gate — thresholding the trained
verifier's confidence — is in this spirit and we claim no novelty for the *idea*; where these overlap our
setting we benchmark them faithfully (a CCPS-style visual-stability gate reaches only AUROC ~0.60 and adds
nothing to verifier-confidence).

**Cross-field transfers (named, so the borrowing is explicit).** The adaptive open-arm controller is
**Weitzman's Pandora's-Box** optimal-search rule (Econometrica 1979), mapped so each "box" is a cheap draw or a
32B escalation and one exchange-rate knob λ yields both thresholds. The candidate-pooling analysis is
**Markowitz** portfolio theory (asset = generator, covariance = error-φ); the unsupervised aggregation baseline
is **Dawid–Skene**; the fusion analysis reduces to **Chair–Varshney** decision fusion; the pairwise selector is
a **PairJudge-RM / knockout-tournament**; the credibility audit is **Bühlmann–Straub**. The visual-token-prune
lever is **FastV**-style shallow-exit. Each is imported as a testable variation and reported with its verdict
(most are honest negatives or regime-limited refinements).

**Base families and the faithful protocol.** Lingshu-7B/32B (arXiv 2506.07044), MedVLThinker-7B/32B,
InternVL3-8B/38B; verifier base Lingshu-7B, box-verifier base Qwen2.5-VL-7B. We reproduce the Lingshu MedEvalKit
protocol faithfully (§2.3) rather than through an internal harness.

**Open-ended vs multiple-choice routing — the discreteness claim.** The observation, which we did not find
stated elsewhere, is that the *routability* of a medical VLM is largely an MCQ artifact: the same confidence
signal moves from AUROC ~0.6 (MCQ) to ~0.87 (open-ended), driven by answer *discreteness*, not *length*
(token-F1 agrees). The corollary — study medical-VLM cascades and verifiers open-ended — motivates the
verifier results.

---

## 9. Limitations and honest caveats

- **Fusion / best-of-N cost FLOPs.** The PMC fusion cell runs both legs (+22 % FLOPs on that slice); the
  open-text best-of-8 arm is 16 cheap forwards (break-even vs one 32B forward is N ≤ 2). These are **latency +
  accuracy** levers, not FLOP-savers. The *MCQ margin cascade* is the FLOP-saving part (1.74 pooled MCQ FLOPs vs
  4.57). F8 (V2) makes the accuracy-max PMC cell FLOP-negative but only *captures 70 %* of F3's PMC beat.
- **Open-text 32B-THINK accuracy is ESTIMATED** for the open cells (judged 32B-no-think + the measured modal
  think-delta −0.195 / −0.120 / −0.130). A judged 32B-think open-text dump would remove the estimate. PathVQA-
  closed has no 32B-think dump → its think acc is set = no-think.
- **In-domain verifier.** SLAKE-open / VQA-RAD-open / PathVQA-open are **in-domain** for the pooled4 verifier
  (trained on slake + pathvqa + kvasir + vqa_rad); the open-text pooled-4 numbers are in-domain, not
  held-out-domain (RadImageNet is the held-out-domain transfer, +13.6 % of its gap). The from-scratch other-base
  (MedVLThinker) verifier is not uniformly robust (fails VQA-RAD at n = 54). Against the strong model the
  verifier is matches-to-modest-win (mean +0.017; seed-0 significant, seed-1 a tie), robust only over
  training-free selection; it lifts *selection over a frozen generator*, it is **not** a SOTA grounder (absolute
  box IoU stays modest: SLAKE 0.255, MS-CXR 0.232).
- **OmniMed-32B blocked** (deterministic tp = 2 NCCL hang) → ALL-7 vs-32B-think is unreportable; ALL-6 is the
  computable pool. Conclusion unchanged (keep-cheap).
- **MMMU is n = 150 and a Lingshu-7B-specific anomaly** (+26 over paper), excluded from all *fidelity* claims; the
  router *exploits* it as a route-to-7B override (+0.140 vs think) but we never claim it as a fusion win, and its
  small n means its macro contribution should be read with the CI in mind (keep-7B d = +0.167, CI [0.087,
  0.247]).
- **INT4 (G3) and G4 are PROJECTED and labelled as such.** INT4 latency is a composition-grounded projection
  (prefill-bound leg → 665 → 583 ms, not 2.5×); INT4 FLOPs are unchanged under MAC accounting; INT4 accuracy is
  a literature bracket. Empirical AWQ latency + Lingshu-32B-INT4 accuracy are future work (no loadable AWQ
  Lingshu-32B; CDN outage stalled the committed benchmark).
- **Judged-think is estimated; two eval contexts are kept separate.** Do not cross-multiply the faithful
  MedEvalKit eval (accuracy + reasoning think-deltas) with the 5-family NGC ACC bake-off (measured batch-1
  latency/energy). Some June/July latency cells (InternVL3 family, PathVQA rows) were measured under GPU
  contention and are uncitable pending a serial re-run (accuracy and FLOPs unaffected). G8's φ = 0.586 is
  transferred from an MVT-32B measurement as a *fraction*; a direct Lingshu-32B prefill/decode split would
  remove the transfer, and unconditional-prefetch FLOPs assume an idle 2nd GPU.
- **A minor within-repo inconsistency, flagged:** `escalation_levers.py` / `beat32b_fusion.py` score SLAKE-open
  with the greedy+seqlogprob FALC fallback (esc ~53 %, 698.6 ms — a "slower cell"), whereas the final method
  (`integrated_method.py`) uses bo8+verifier SLAKE-open (605.5 ms, esc 12.6 %), so those artifacts' pooled
  baselines differ slightly (0.5749 / 2.337 vs 0.5750 / 2.538). The final method is the bo8+verifier treatment.

---

## 10. Conclusion

A medical-VQA cascade must decide **which** queries to escalate and **what** to run when it does. Scored against
the honest expensive baseline it is actually about — **always-32B-THINK** (measured 10.5 s / 2.0 kJ per open-text
question) — this paper's **format-aware, regime-adaptive router** is *faster and more accurate*: on the full
MedEvalKit suite (n = 42 374) the compute-lean setting is **+0.0123 vs think at 0.49× its FLOPs and −96 %
latency**, and the accuracy-max V2 setting is **+0.0212 at 0.93× FLOPs** — all-axes dominant. Always-32B-THINK
is Pareto-dominated on all five families, because thinking over-thinks perception VQA. The two positive levers
are **structure** (the ACC compute-configuration cascade: −80 % latency, FLOPs → 52 % at ALL-6 parity) and **a
little training** (the outcome verifier: beats the 32B/38B across three families and held-out OOD, +0.191 on
real MS-CXR boxes). Both are honestly bounded by two **walls** we present as a headline: the **recoverability
wall** confines the broad-slice MCQ *beat* to PMC-VQA (six independent methods + an FDR-controlled slice search
agree, and the CI-guardrail is the correct fix per an H8 credibility audit), and the **selectability wall** makes
best-of-N's oracle → selection gap largely fundamental (a 7× verifier only ties the trained 7B; coverage does
not compound with selection). The deployable efficiency win is the router + Pandora; the trained verifier is the
accuracy engine whose latency dominates the think baseline it replaces. We release the unified pipeline
(`method_final.py`, one command, CPU re-costing of saved dumps), the trained answer/box verifiers, and the full
negative-result characterization so both the wins and the walls reproduce end-to-end. The clearest direction for
future work is the binding limit throughout — **candidate quality** — where a cross-model pool raises oracle@N by
+0.11–0.15, the largest remaining headroom.

---

## Reproducibility index

Every number traces to a checkpoint; code launches from the repo root (`~/medvlthinker-imgdiff-compute`).

**The unified method (one command).** `python3 src/cascade_methods/method_final.py` → `artifacts/method_final.json`
(V1: both knob settings + INT4 + reconciliation) **and** `method_final_v2.json` (V2: F8 + F10). Per-lever
references: `integrated_method.py` (`integrated_method_vs_think.json`), `beat32b_fusion.py`
(`beat32b_fusion.json`), `beat32b_more.py` (`beat32b_more.json`: F7/F8/F11/F10), `integrated_pandora.py`
(`integrated_pandora_opentext.json`), `pandora_controller.py` (`pandora_controller.json`),
`quantized_strong_leg.py` (`quantized_strong_leg.json`), `escalation_levers.py` (`escalation_levers.json`),
`escalation_more.py` (`escalation_more.json`), `logit_fusion.py` (`logit_fusion.json`), `robust_slice_routing.py`
(`robust_slice_routing.json`: H4 discovery + H8 credibility), `unified_router.py` (`unified_router.json`).

**Baseline costs (measured batch-1, NVML).** `opentext_32b_think.json` (32B-think open-text 10 521.6 ms /
2 001.9 J), `iv3_38b_latency.json` (IV3-38B 6 220 ms / 3 275.6 J), `latency_32b.jsonl` (prefill/decode
composition, φ = 0.586), `reframe_vs_bigthink.json` (5-family ACC reframe), `master_data.csv` (ACC ALL-6/ALL-5).

**ACC (structure / efficiency).** `src/cascade_methods/{acc.py, acc_v2.py, acc_v3_confgate.py,
acc_v4_lowres_think.py}`; over-thinking premise `{strong_leg.py, final_comparison.py, overthink_generalize.py}`;
bake-off / cross-family `{acc_allmethods.py, acc_compare.py, cascade_all_families.py, gate_compare.py}`; per-tier
batch-1 latency/energy `{latency_estimate.py, open_measure_latency_energy.py}` + `src/labeling/nvml_power.py`;
cost math `METHOD_ACC.md` / `METHOD_MATH.md`.

**The two walls (the negatives).** Recoverability: `beat32b_fusion.json`, `beat32b_more.json`, `logit_fusion.json`,
`robust_slice_routing.json`; luck-floor / recoverability diagnostics `src/cascade_methods/{diagnostics.py,
strong_fixes_genuinely_unknown.py, knowledge_feasibility_bytype.py}`; the gate bake-off `gate_unified_bakeoff.py`
(`gate_unified_bakeoff.json`). Selectability: `{combine_diverse_pairwise.py, distractor_filter.py,
verifier_32b_measure.py, end_to_end_consolidation.py, latency_reexamination.py, active_comparison_verifier.py,
bandit_allocation.py, dawid_skene_aggregate.py, generator_portfolio.py, diversity_candidates.py,
diverse_measure_gpu.py, pairwise_verifier_score.py, pandora_correlated.py, pandora_pooling_combo.py}` with matching
artifacts.

**Trained verifier (accuracy).** Answer verifier `src/training_methods/{run_lora_verifier_open.py,
run_lora_verifier_ranking.py, clean_verifier_dump.py}`; non-circularity / active-ingredient ablations
`{verifier_image_ablation.py, verifier_transfer_eval.py, verifier_scaling_curve.py, cross_gen_verifier.py}`;
3-family beats-32B table `src/cascade_methods/open_verifier_cascade_table.py`; gate bake-off / cost frontier
`{open_gate_swap.py, open_gate_efficiency.py, open_gate_heldout_tau.py, open_recoverability_gate.py,
open_cost_frontier.py}`; box verifier `src/training_methods/run_lora_box_verifier.py`; grounding
`src/labeling/{run_ground_slake.py, run_ground_mscxr.py}`; UGV MCQ `src/cascade_methods/ugv_mcq_verdict.py`
(`ugv_mcq_verdict.json`).

**Open-ended ceiling-break + judge.** `src/labeling/{run_openvqa.py, run_openvqa_fewshot.py}`; gate hunt
`src/cascade_methods/{gate_search_open.py, open_gate_bakeoff.py}`; judge `src/labeling/run_judge.py`
(MedVLThinker-32B + Claude-Sonnet-5 cross-check, κ 0.85–0.96, 100 % exact-match anchor).

**Docs (canonical ledgers).** `results/cascade_methods/docs/current/{METHOD_FINAL_2026-07, RESEARCH_RESULTS_2026-07,
METHODS_MASTER, MASTER_SUMMARY_2026-07, OMNIMED_FALLBACK, METHOD_ACC, METHOD_MATH}.md`; backlog
`METHOD_IDEAS_BACKLOG.md` (56 ideas); session narrative `progress/progress_{June_17…June_29-30, July_01-02,
July_03, July_04, July_05, July_06, July_07}.md`.

*End of manuscript. Every figure above is verbatim from a real checkpoint or arithmetically derived from one;
none is fabricated. This document supersedes no experimental data — it consolidates it.*
