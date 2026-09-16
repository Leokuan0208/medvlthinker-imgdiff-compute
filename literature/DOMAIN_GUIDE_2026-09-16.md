---
title: "The Domain Guide — probe verifiers, best-of-N and medical VQA"
subtitle: "Where the field is, where our method sits, and every paper you need to read"
author: "Prepared for Li-Wen (Leo) Kuan · medvlthinker-imgdiff-compute · 2026-09-16"
---

# 0. How to use this document

## 0.1 What this is

A single catch-up guide for the research domain around the **live** method in this project: a small
probe that reads a frozen medical vision-language model's own hidden states and picks the best of
eight sampled free-text answers.

> **194 paper cards covering 186 unique papers, in 9 categories, with 34 core PDFs in `papers/`.**
> Every arXiv identifier was checked against the arXiv API on 2026-09-16 — including all 156 cited
> across this project's own documents, every one of which resolved.

It has five jobs:

1. **Teach the vocabulary** the field actually uses (§1), so a report never again says "MLP head" when
   the reader expects "MLP probe used as a best-of-N verifier".
2. **State where we stand** in one place, with every number tied to the artifact it came from (§2).
3. **Map the literature** in nine categories, with a card per paper — models, method, datasets,
   experiments, results, conclusions, and *what it means for us* (§3).
4. **Position us against the neighbours** — who did what first, and what is genuinely ours (§4).
5. **Prepare the next professor report** — the comparison context, the conventions, the sentences
   to use and the claims not to make (§5). A reading order is in §6, the paper package in §7.

**Scope decision (2026-09-16).** This guide is centred on the **open-text probe-verifier arm** —
the September 2026 work. The July–August multiple-choice cascade (margin gate, certified veto,
32B escalation) is summarised only as history in §2.1; its literature (cascades, routing, deferral)
is deliberately out of scope here and is covered by `LITERATURE_UPDATE_2026-08-11.md` §3.

## 0.2 Provenance rules — read before quoting anything

- Every number about **our** work names its source file under
  `results/cascade_methods/artifacts/` or `results/cascade_methods/docs/current/`. If a number has no
  file after it, it is a paraphrase of a mechanism, not a measurement.
- Every number about **a paper** carries a tag: `[abstract]`, `[§x]`, `[Table n]`, or
  `[abs+search]`. Cards marked `provenance: abstract-only` were read from the arXiv abstract page only;
  read the PDF before citing a table value from them.
- Every arXiv identifier in this guide was checked against the arXiv API on 2026-09-16 (156 from
  the repo's own docs, plus everything the category sweeps added). A paper that failed verification is
  listed in the category's *dropped* list and nowhere else.
- Venues are given only where verified; otherwise "arXiv preprint".
- **No abstention.** Papers about reject-option / defer-to-human / selective prediction *as a method*
  are excluded by standing rule (CLAUDE.md rule 6). Uncertainty *estimation* and calibration are in
  scope because they produce a score, not a refusal.

## 0.3 Reading depths

- **One hour.** §1.1 (the ten terms), §2.2–2.4 (method and results), §4 (positioning), §5.1–5.3.
- **One day.** Add §1 fully, the *core* cards in each category of §3 (they are marked ★), and §6.
- **One week.** Everything, then the ★ PDFs in `papers/` in the order of §6.

## 0.4 Conventions in this document

- "**We**" / "**ours**" = the probe verifier as shipped on 2026-09-12 (`genframe_head_pooled_ens_v2`).
- "**Benchmark**" is used where older repo documents say "cell" (see §1.5).
- "**Judge currency**" = correctness decided by the Lingshu-32B LLM-judge; "**EM currency**" =
  normalised exact match. Where only one is available, it is named.
- ★ marks a *core* paper (PDF included in `papers/`). Priority 1/2/3 is the reading order within a
  category.


---

# 1. The vocabulary — what the field calls the things we do

Terms are grouped by the part of the pipeline they belong to. Each entry gives the field's name in
bold, a plain definition, and — where we have been using a different word — the mapping. The
terms the category sweeps introduced from individual papers are collected in §1.9.

## 1.1 The ten terms you cannot do without

| field term | plain meaning | what we used to say |
|---|---|---|
| **vision-language model (VLM / LVLM / MLLM)** | a model that takes an image and text and produces text: a vision encoder → a projector → a language model. "LVLM" (large VLM) and "MLLM" (multimodal LLM) are the same thing in different papers. | VLM ✓ |
| **medical VQA, open-ended vs closed-ended** | visual question answering on medical images. *Closed-ended* = the answer set is fixed (yes/no or A/B/C/D); *open-ended* = free text. The split is called the *open split* / *closed split* of a dataset. | "open cell / closed cell" |
| **greedy decoding** | generate one answer by always taking the most likely next token (temperature 0). The single-answer baseline every sampling method must beat. | "greedy" ✓ |
| **best-of-N (BoN) sampling** | draw N candidate answers from the model (usually at temperature > 0), then pick one with a scoring rule. The scoring rule is the *verifier* / *reward model* / *selector*. With N = 8 it is "best-of-8". | "best-of-N" ✓ |
| **candidate set** (candidate pool) | the N sampled answers for one question. After de-duplicating identical strings, ~3.9 distinct candidates remain at N = 8, T = 0.7 (`meetings/shipped_method_2026-09-12.html`). | "pool" |
| **verifier** (outcome reward model, ORM) | a model that scores a *finished* candidate answer for correctness so the best can be chosen. Ours outputs P(correct \| image, question, candidate). Contrast a *process* reward model (§1.3), which scores intermediate steps. | "the head", "the MLP" |
| **probe / probing classifier** | a small classifier trained on the *frozen* internal representations of a model to read out a property. A *linear probe* is a single linear layer; a *non-linear probe* / *MLP probe* has a hidden layer. **Ours is an MLP probe** (§1.2). | "the MLP head" |
| **hidden state** (activation, residual-stream vector) | the vector a transformer holds at one token position at one layer. Lingshu-7B's language model has hidden size **3584** and **28 layers**; our probe reads layers 18/20/22. | "features", "h_span" |
| **oracle@N** (pass@N, coverage) | the accuracy you would get if you always picked a correct candidate whenever one exists in the set of N. It is the ceiling of any selector. "Coverage" is the same quantity in the test-time-scaling literature. | "oracle@8" ✓ |
| **selection efficiency** | our name for the fraction of the oracle gap a selector converts: (selected − greedy)/(oracle@N − greedy). Not a standard name — define it at first use. The published analogue is Hu's *signal fidelity* / *conditional selection quality* (card `hu2026oracle`, §3.4). | "sel_eff" |

## 1.2 What our probe is, precisely — and the four ways "MLP" goes wrong

Our shipped scorer is, per probe:

```
Linear(3584 → 256)  →  GELU  →  Dropout(0.0)  →  Linear(256 → 1)
```

**918,017 parameters each** — verified by loading `ckpts/train/genframe_head_pooled_ens_v2/head_L18_seed0.pt`:
`f.0.weight` (256, 3584) = 917,504 · `f.0.bias` 256 · `f.3.weight` (1, 256) = 256 · `f.3.bias` 1.
(The dropout sits at index 2 with probability 0, which is why the output layer is `f.3` and not
`f.2`; it is a no-op at inference.) We ship 24 of them (3 layers × 8 seeds), 22.05 M parameters,
88.2 MB in float32. The recipe of record is `genframe_head_pooled_ens_v2/recipe.json`:
objective **BCE**, **AdamW lr 1e-3, weight decay 1e-2, 30 epochs, batch 256**, 8 seeds per layer,
112,770 training rows, 59 leaking rows dropped.

⚠️ The *older* artifact's docstring (`src/training_methods/genframe_selector.py`) prints **918,529**
for the same nominal shape and uses a **Bradley–Terry** objective; that number is 512 higher than the
shape implies and has not been re-verified here. Quote **918,017** and name the checkpoint.

The four sources of confusion when this is called "an MLP":

1. **"MLP" already means something else inside a transformer.** Every transformer layer contains a
   *feed-forward block*, which most papers call "the MLP" (or FFN): `Linear(d → k·d) → activation →
   Linear(k·d → d)`, an *expansion* (k = 4 in the original transformer; Qwen2.5 uses a gated SwiGLU
   variant). Ours is a *contraction* to a scalar and lives *outside* the model. Saying "we add an MLP"
   makes a reader picture a change to the model's own blocks. Say **"a standalone MLP probe on frozen
   hidden states"**.
2. **Layer counting is ambiguous.** "Two-layer MLP" can mean two *weight* layers (ours) or two *hidden*
   layers (not ours). The unambiguous phrasing is **"an MLP with one hidden layer of width 256"** or
   "two linear layers with a GELU between them".
3. **"Head" implies joint training with the backbone.** A *classification head* / *value head* /
   *reward head* is trained together with, and attached to, a model. Ours is trained *after* the
   generator is frozen and never touches it. The probing literature's word for this is **probe**; the
   test-time-compute literature's word for its *role* is **verifier**. Use both: *"an MLP probe used
   as a best-of-N verifier"* (`TERMINOLOGY_2026-08-24.md` §2).
4. **The activation and the objective are part of the name.** GELU (Gaussian Error Linear Unit,
   Hendrycks & Gimpel 2016) is the default in BERT/GPT/ViT-style networks and is what we use; the
   probe is trained with **binary cross-entropy (BCE)** on per-candidate correctness labels — i.e. it
   is a **pointwise, discriminative, outcome verifier**. (Our earlier `genframe_head_ens8` used a
   **Bradley–Terry** *pairwise* objective over correct/incorrect pairs within a question; the
   difference was a tie once data was matched, `OPENTEXT_CORRECTIONS_2026-08-19.md` §3.) State the
   objective; "an MLP" without it tells a verifier researcher nothing.

Two recipe details that the field will ask about because they are load-bearing:

- **Standardisation.** Inputs are z-scored with the *training* mean and standard deviation per
  dimension, frozen in `standardizer.npz`. Omitting it costs 0.0075 selection efficiency
  (`genframe_selector.py` docstring).
- **Rank averaging.** The 24 probes are on incompatible scales, so each probe's scores are converted to
  ranks *within the candidate set* and the ranks are averaged; the argmax is taken. This is
  *rank-average ensembling*; getting it wrong (averaging raw logits) is worth 0.008
  (`meetings/shipped_method_2026-09-12.html`).

**One sentence to use everywhere:** *"A lightweight MLP probe (one hidden layer of 256, GELU) reads
the frozen hidden states of Lingshu-7B — the mean over each candidate's own generated tokens at
layers 18/20/22 — and acts as a pointwise best-of-8 verifier trained with binary cross-entropy;
24 probes (3 layers × 8 seeds) are rank-averaged."*

## 1.3 Test-time compute and verification

- **Test-time compute / inference-time scaling / test-time scaling (TTS).** Spending more compute at
  inference (more samples, longer reasoning, search) instead of training a bigger model. Two axes:
  *parallel* (sample N in parallel — us) and *sequential* (longer chains, self-refinement).
- **Compute-optimal scaling.** Choosing, per question or per budget, how to split compute between
  model size and test-time methods. The result that a small model + TTS can beat a bigger model *on
  questions where the small model has non-trivial success* (Snell et al., §3.1).
- **Self-consistency (SC) / majority vote.** Sample N, return the most frequent answer. Needs answers
  that can be canonicalised (a number, a letter); on free text it needs normalisation or semantic
  clustering (*universal self-consistency*, MBR decoding, §3.5). On our benchmarks it sits at the
  random-pick floor and does not improve with N (`TRANSFER_WALL_2026-08-21.md` §4).
- **Answer-prior baseline.** Our control that scores a candidate by P(correct \| its normalised string),
  counted on the training rows — no image, no hidden state. It catches "the probe just learned which
  strings tend to be right" (`OPENTEXT_CORRECTIONS_2026-08-19.md` §1). Define it at first use; the
  nearest published notion is a *label prior* / *answer-frequency baseline*.
- **Outcome reward model (ORM)** vs **process reward model (PRM).** ORM scores the final answer; PRM
  scores each reasoning step. Ours is an ORM. PRMs are irrelevant on answers with no steps.
- **Discriminative vs generative verifier.** *Discriminative* = a classifier outputs a score (ours).
  *Generative* (GenRM) = an LLM is asked "is this correct?" and its "Yes" probability is the score.
  Our July LoRA verifier was generative-style (read P("Yes") from a fine-tuned 7B).
- **Self-verification.** The generator judges its own answer, zero-shot. Shown unreliable on medical
  VQA (*Verification Mirage*, §3.2) — the motivation for training a verifier.
- **Reward hacking / over-optimisation.** Picking the candidate with the highest *imperfect* score
  starts selecting for the scorer's errors as N grows; accuracy can *fall* with N (§3.4).
- **Weak verifier / verifier ensembling.** Combining several unreliable scorers (Weaver, FUSE, §3.2).
- **Best-of-Majority (BoM).** Take the frequent answers first, then the reward-maximal among them
  (§3.2/§3.4).

## 1.4 The walls (limits) and how they are measured

- **Coverage wall.** The fraction of questions for which *no* candidate in the set is correct. Ours:
  37.4 % of questions at N = 8 (CLAUDE.md §0; 40.8 % of 1,064 held-out questions in
  `PROJECT_RETROSPECTIVE_2026-07-29.md` §5.5). No selector can help there.
- **Selection wall.** The gap between what a selector picks and oracle@N; a *selection efficiency* of
  0.78–0.81 was our LoRA-era value and is close to what other groups report (§3.4).
- **Oracle gap.** oracle@N − single-answer accuracy: the most any selector can add.
- **Recoverable mass, conditional quality, conditional harm** (Hu 2026). *Recoverable* = greedy wrong
  and some candidate right; *quality* = P(pick right \| recoverable); *harm* = P(pick wrong \| greedy was
  right). Our LoRA-era values: quality 0.4541, harm 0.0987 (`PRIOR_ART_2026-08-11.md` §3.2).
- **Selection skill vs sampling penalty** (our decomposition, `decomposition_2026-08-24.json`):
  `verifier − greedy = (verifier − one random sample) − (greedy − one random sample)`. The first
  term is skill; the second is the price of sampling at temperature > 0. Not standard; define it.
- **Sampling penalty / temperature.** Higher temperature increases diversity (more distinct candidates,
  higher oracle) but lowers each sample's accuracy. The verifier wants T = 0.7 (§2.4).

## 1.5 Evaluation vocabulary

- **Benchmark / dataset / split.** The field says "eight benchmarks" or "the open-ended split of
  SLAKE". Our internal word "cell" (one dataset × one format) reads as a table cell; keep it only in
  file names (`TERMINOLOGY_2026-08-24.md` §1).
- **Exact match (EM)** — after normalisation (case, punctuation, articles), does the prediction equal
  the gold string? **Token F1** — overlap of tokens. **Recall / contains** — is the gold string inside
  the prediction? Standard open-ended medical-VQA papers report EM, F1, BLEU and/or recall.
- **LLM-as-a-judge.** A language model decides whether a free-text prediction matches the gold. Ours
  is **Lingshu-32B** judging Lingshu-7B's outputs — a *same-family judge*, which the judge literature
  flags for **self-preference bias** (§3.8). Our measured symptom is **paraphrase drift**: a newly
  trained verifier gains ~0.006–0.009 under the judge before doing anything useful (CLAUDE.md §0).
  Therefore every verifier comparison is to be reported in **both currencies** (judge and EM).
- **Macro average.** Equal weight per benchmark regardless of size (ours: 1/8 each). *Sample-weighted*
  (pooled) lets the largest benchmark dominate. Always say which.
- **By-image split.** Train/held-out halves are split on the *image* (`md5("nd"+image_md5) % 2`), never
  on the question, because several questions share one image. Field term: *group-wise split* /
  *image-level split*.
- **Bootstrap confidence interval (CI); clustered bootstrap.** Resample to get a 95 % interval. Ours
  resamples *images* or *benchmarks*, never questions i.i.d.: an i.i.d. interval on benchmark-clustered
  data was ~8× too narrow and turned a tie into a "win" (`TRANSFER_WALL_2026-08-21.md` §6). Field
  term: *cluster bootstrap*.
- **WIN / TIE / LOSS.** Our shorthand for "95 % CI excludes zero on the positive side / includes
  zero / excludes zero on the negative side". Say "significant at 95 %" in a report.
- **Leave-one-benchmark-out (LOBO).** Train on seven, test on the eighth: the test of whether the
  verifier is *domain-general*. Field terms: *leave-one-domain-out*, *cross-dataset generalisation*.
- **Guardrail.** "Never worse than the baseline on any single benchmark" — our per-benchmark
  non-inferiority check. Field term: *per-dataset non-inferiority* / *no-regression constraint*.
- **Held-out vs in-distribution vs out-of-distribution (OOD).** Held-out = unseen questions/images from a
  *seen* benchmark; OOD = a benchmark the verifier was never trained on. Our LOBO result says the
  verifier is per-benchmark, i.e. weak OOD.
- **Contamination / leakage.** Test images or questions present in training data (of the generator or
  of the verifier). Our audits: image-level overlap, perceptual-hash near-duplicates, split salts.

## 1.6 Cost vocabulary

- **FLOP-equivalent (FLOP-eq).** Our unit: one Lingshu-7B forward+generate on one question = 1.0.
  Best-of-8 with a shared image/prompt prefill measures **1.13 FLOP-eq** by forward-token count
  (`bestofn_vllm_2026-09-16.json`), versus **8.0** under the naive "charge every sample fully"
  convention. Say which convention.
- **Prefill vs decode.** *Prefill* = processing the prompt + image tokens (the expensive part for us:
  ~323 prompt tokens vs ~6 generated); *decode* = generating output tokens one at a time.
- **Prefix caching / shared prefill.** vLLM computes the prompt once and forks N samples; HF's
  `num_return_sequences` repeats the prompt N times. The two serving paths give different costs for the
  *same* method (2.74× vs 1.99× latency, `bestofn_vllm_2026-09-16.json`,
  `bestofn_latency_energy_2026-08-03.json`).
- **Latency / energy / VRAM / FLOPs** are four different axes; a method can win on one and lose on
  another. Report all measured ones and name the batch size (ours: batch 1).
- **"Verification is free".** Our probe adds ~1.8 MFLOP per candidate per probe on vectors already
  computed — but this *argument* is already in print (HSRM, CASE, Q-Probe; §3.3). State the cost as a
  property, not a discovery.

## 1.7 Model and architecture vocabulary

- **Vision encoder (ViT / SigLIP / NaViT)**, **projector / connector / merger**, **language model
  (decoder)**. The three generators, with the numbers a probe actually depends on:

  | generator | language model | hidden size | LM layers | vision tower | source of these numbers |
  |---|---|---:|---:|---|---|
  | Lingshu-7B | Qwen2.5-7B | **3584** | **28** | Qwen2.5-VL ViT (hidden 1280, 32 layers) | Qwen2.5-VL report, Table 1 (§3.7) |
  | Qwen2.5-VL-7B | Qwen2.5-7B | 3584 | 28 | same | same |
  | MedGemma-4b-it | Gemma 3 (`gemma3_text`) | **2560** | **34** | SigLIP (hidden 1152, 27 layers, 896 px) | **the model's own `config.json`** |

  ⚠️ The MedGemma row is read from
  `/data/dan/hf_cache/hub/models--google--medgemma-4b-it/snapshots/290cda…/config.json`, **not** from
  the Gemma 3 or MedGemma papers — a check of the Gemma 3 paper on 2026-09-16 could not locate a
  hidden-size/layer table in it. Cite the config, not the paper, for these two numbers.
  This matters because our ensemble layers are chosen by **relative depth**: Lingshu's [18,20,22] of
  28 is 0.643/0.714/0.786, and the depth-matched equivalent on 34 layers is [22,24,27] — which was
  pre-registered and did beat the absolute-matched choice (§2.4.4).
- **Layer depth.** "Layer 20 of 28" ≈ relative depth 0.71; when comparing generators with different
  depths, match *relative* depth (`TRANSFER_WALL_2026-08-21.md` §14).
- **Mean pooling over a span (h_span).** Averaging the hidden states over the positions of the
  candidate's own generated tokens. Alternatives in the literature: last-token state, per-token
  states with a gate (ELHSR), step-boundary states (HSRM).
- **LoRA (low-rank adaptation).** Small trainable matrices added to a frozen model. Cheap to *train*,
  but *inference* still runs the whole model — which is why the July LoRA verifier cost a full 7B
  forward per candidate and the probe replaced it.
- **Temperature, top-p (nucleus), min-p, repetition penalty.** Sampling controls. We use T = 0.7 for
  the shipped verifier.
- **Seed.** The random initialisation of a probe fit. A single seed spans [0.8025, 0.8140] selection
  efficiency across 16 seeds (`meetings/shipped_method_2026-09-12.html`); 8 seeds are averaged for
  stability. Report seed counts and spreads.
- **Thread count.** CPU fits are deterministic *given* the thread count, but the thread count moves
  the macro by 0.0033 (`repro_threading_2026-09-13.json`). Pin it.

## 1.8 Uncertainty and calibration vocabulary (for comparisons a reviewer will ask for)

- **Calibration, ECE, reliability diagram.** Whether a model's stated confidence matches its accuracy.
- **Maximum softmax probability (MSP), margin, entropy, semantic entropy, self-certainty.** Cheap
  confidence signals computed from output probabilities; *semantic entropy* clusters sampled answers by
  meaning first. These are the training-free confidence baselines; on open-ended text they are the
  natural competitors to a trained probe.
- **Hallucination (object hallucination, language prior, yes-bias).** A VLM answering from text
  statistics rather than the image. POPE-style yes/no probing measures it.
- **Probe-based confidence** (what we do, applied to *one* answer rather than to a set): a probe on
  hidden states predicting correctness — DualRead does this on our benchmarks (§3.3).

## 1.9 Terms introduced by individual papers

Collected from the per-paper cards in §3, de-duplicated. If a term above conflicts with one here, the
paper's own definition applies to that paper's card only.

| term | meaning | first used in |
|---|---|---|
| **"MLP" naming nuance (probe vs. transformer internals)** | Inside a transformer, "the MLP" almost always means the feed-forward block repeated in every layer: Linear(d→4d) → activation → Linear(4d→d), an internal expansion-then-contraction that is part of the frozen base model itself. When a probing paper (like this one, or ours) says it "tried an MLP", it means something completely different: a small, STANDALONE classifier trained on top of the model's already-computed hidden states — in our case Linear(3584→256) → GELU → Linear(256→1), i.e. a tiny external network, not a component of the transformer. Also, "2-layer MLP" is itself ambiguous (it can mean 2 weight layers, i.e. one hidden layer, or 2 hidden layers, i.e. 3 weight layers) — our probe should always be described unambiguously as having "one hidden layer of width 256", never just "a 2-layer MLP". | §3.3 |
| **activation-based classifier / probe** | A small supervised model trained to predict a target property (here, truthfulness) from a larger model's internal hidden-layer activations rather than from its text output. | §3.6 |
| **Active learning (for reward-model labeling)** | Choosing which examples (here, which reasoning steps) to get human labels for based on where the model is currently most uncertain or most useful to learn from, rather than labeling everything or labeling randomly. | §3.2 |
| **adaptive self-consistency / adaptive N** | Choosing how many samples to draw per question instead of a fixed N. | §3.1 |
| **adaptive/early stopping (in sampling)** | Drawing samples one at a time and stopping as soon as a cheap agreement/confidence check says the answer is settled, instead of always drawing a fixed number of samples. | §3.5 |
| **agglomerative clustering** | A clustering algorithm that starts with every point as its own cluster and repeatedly merges the closest pair of clusters until a stopping rule is met. | §3.5 |
| **ALFA (Alignment Ratio of Atomic Facts)** | A metric that decomposes a generated report/answer into individual atomic factual claims and measures what fraction are consistent with ground truth, used to derive hallucination labels without subjective human judgment. | §3.6 |
| **anchor regularizer** | A loss term that discourages a model's predicted confidence from collapsing to the extremes (always near 0 or always near 1), which would otherwise trivially minimize some calibration losses without being informative. | §3.6 |
| **answer derivability** | Whether a benchmark's annotated gold answer can actually be logically derived from its paired knowledge-base entry -- if not, no model, however capable, could be scored correct via genuine reasoning. | §3.8 |
| **answer-position / option bias** | A dataset flaw where the correct answer's letter/position (A/B/C/D) is not uniformly distributed, letting a model score above chance by exploiting the position alone rather than the content. | §3.8 |
| **attention head** | One of several parallel sub-units inside a transformer's attention layer, each computing its own weighted combination of other tokens' representations; different heads can specialize in different kinds of information (e.g., some heads here are found to carry more hallucination-relevant signal than others). | §3.3 |
| **AUARC** | Area Under the Accuracy-Rejection Curve -- summarizes how much accuracy improves as increasingly low-confidence answers are withheld. | §3.6 |
| **AUC / AP** | Area under the ROC curve and average precision; threshold-free measures of a binary classifier's ranking quality (0.5 AUC = chance). | §3.1 |
| **AURAC** | Accuracy-Rejection-curve Area -- area under a curve of accuracy on the remaining answered questions as increasingly uncertain ones are rejected. | §3.6 |
| **Automatic process labeling (for PRMs)** | Deriving step-level correctness labels without human annotators, e.g. by estimating each step's success rate via many rollouts from that step, rather than paying humans to label every step. | §3.2 |
| **automatic verifier** | A checker that can confirm an answer is right without a model (unit tests, proof checkers). Free-text medical VQA has none, which is why a learned verifier is needed. | §3.1 |
| **Bayesian Decoding Game (BDG)** | A training-free decoding method that models generation as a signalling game between a 'generator' and a 'verifier' language-model role and iterates until their preference orderings over candidates agree (equilibrium). | §3.5 |
| **beam search / DVTS** | Search methods that keep several partial solutions alive and extend the ones a PRM scores highest; DVTS splits the beam into independent subtrees for diversity. | §3.1 |
| **behaviour cloning (BC)** | Supervised fine-tuning on human demonstrations of the task. | §3.1 |
| **Best-of-N** | Sample N candidate answers from a generator, score each with a verifier, and return the highest-scoring one. The method both this paper and our project use. | §3.2 |
| **best-of-n policy** | The distribution over outputs obtained by sampling n and keeping the top-scored one; a new policy derived from the base model without changing weights. | §3.1 |
| **best-of-N weighted** | Best-of-N where the verifier scores of all samples that share the same final answer are summed and the answer with the largest total wins — a hybrid of best-of-N and majority vote. | §3.1 |
| **best-of-N with a learned verifier** | Sample N answers, score each with the verifier, return the highest. The whole of our selection method. | §3.1 |
| **Best-of-Poisson (BoP)** | A variant of Best-of-N designed to approximate the theoretically optimal reward-vs-KL-divergence tradeoff at inference time. | §3.4 |
| **Beta regression pass@k** | Modelling each question's single-sample accuracy as a Beta-distributed variable so that pass@k has a closed form. | §3.1 |
| **Bi-level optimization** | An optimization setup with two nested levels: an inner/lower optimization (e.g. fine-tune model weights) and an outer/upper optimization (e.g. adjust hyperparameters like domain weights) that uses the inner level's outcome as its objective, updating the outer parameters based on held-out performance. | §3.2 |
| **bidirectional entailment** | Two statements entail each other if each one logically implies the other is true; used here as a test for 'these two answers mean the same thing'. | §3.5 |
| **BLEURT** | A learned, neural evaluation metric for text generation quality (here, machine translation), trained to correlate with human judgments better than surface metrics like BLEU. | §3.5 |
| **Borda voting (weighted by confidence rank)** | An aggregation rule that gives each sample's answer a number of votes based on where that sample ranks by confidence, rather than one flat vote per sample. | §3.5 |
| **Bradley-Terry model** | A statistical model for the probability that item i beats item j in a pairwise comparison, P(i>j) = pi_i/(pi_i+pi_j), fit from win/loss data. The classical basis for pairwise ranking losses (as opposed to pointwise classification/regression or listwise ranking) — this project tested a Bradley-Terry-style pairwise objective for the verifier before moving to pointwise BCE. | §3.9 |
| **Bradley-Terry ranking loss** | A pairwise loss that trains a scoring function so that, for any pair of one correct and one incorrect candidate, the correct one gets a higher score, without forcing any particular order among multiple correct candidates (a "tie-safe" version explicitly avoids penalizing ties among equally-correct candidates). | §3.3 |
| **Brier score / Brier-style loss** | A proper scoring rule measuring the mean squared difference between a predicted probability and the actual binary outcome (correct/incorrect); minimizing it as a training loss directly optimizes calibration. | §3.6 |
| **budget forcing** | Forcing a model to keep thinking (or stop) by manipulating its reasoning tokens at inference. | §3.1 |
| **C_gen vs C_eval** | Compute spent generating candidates vs compute spent scoring/choosing among them. Must be reported separately. | §3.1 |
| **calibration** | How well a model's stated confidence matches its actual accuracy — e.g. among all answers the model says it is 80% confident in, roughly 80% should actually be correct. Calibration scores ONE already-chosen answer; it is a different problem from SELECTING the best among several candidates (best-of-N), even though both can use similar internal-state signals. | §3.3 |
| **Candidate-factor graph** | A data structure connecting each candidate answer to the individual verifiable claims ('factors') it depends on, allowing one verified claim to be reused as evidence for or against multiple candidates that mention it. | §3.2 |
| **canonical-order exchangeability** | A text-side contamination signal: if a model's output likelihood for a sequence changes when items in it are reordered relative to a fixed 'canonical' training order, that can indicate the model memorized the specific ordering seen during training. | §3.8 |
| **canonicalizable answer** | An answer that can be reduced to a small fixed set of equivalent strings (e.g. a number, a multiple-choice letter) so that two generations can be checked for exact agreement. | §3.5 |
| **CEUR-WS working notes** | CEUR-WS.org is an open-access repository that publishes 'working notes' (informal proceedings) for workshops and shared-task campaigns like CLEF; these are citable but not peer-reviewed in the same way as a conference paper. | §3.8 |
| **Chain-of-thought (CoT) verification** | Having the verifier write out its reasoning about why a candidate is correct or wrong before giving its final judgment, analogous to chain-of-thought prompting for the generator itself. | §3.2 |
| **Chinchilla / overtraining** | Chinchilla is the rule of ~20 training tokens per parameter for best loss per training FLOP; overtraining means training a smaller model on far more tokens than that, which is worse per training FLOP but cheaper at inference. | §3.1 |
| **classifier-free guidance (CFG)** | A technique (originally from diffusion image models) that steers a model's output distribution toward a desired condition by contrasting conditioned and unconditioned predictions; repurposed here to steer a MedVLM's token embeddings toward expert-highlighted content. | §3.6 |
| **closed- vs open-ended VQA** | Closed-ended: the question has a small fixed answer set (e.g. yes/no, or a short list of organs) and is scored as accuracy. Open-ended: the model must generate free text, scored by soft overlap (token F1) or an LLM judge. | §3.7 |
| **closed-book / knowledge-intensive task** | Questions whose answer is a fact the model must already store in its weights; no reasoning can derive it. | §3.1 |
| **closed-ended vs open-ended VQA** | Closed-ended answers are constrained (e.g. yes/no or a fixed option set); open-ended answers are free text. | §3.8 |
| **co-failure rate (beta)** | The fraction of queries on which every model in a pool gives a wrong answer simultaneously — a hard ceiling of 1−beta on any policy that must output one model's answer. | §3.4 |
| **complexity stratification** | Grouping generated questions into difficulty tiers so a benchmark can separately report performance on easy vs. hard questions. | §3.8 |
| **compound inference system** | A pipeline that makes several model calls and combines them (vote, filter, cascade). | §3.1 |
| **compute-bound vs memory-bound** | A stage limited by arithmetic throughput vs one limited by how fast weights and cache can be read from memory; decoding at small batch is memory-bound. | §3.1 |
| **compute-optimal scaling** | Choosing, for each question, the test-time strategy and budget that gives the highest accuracy for a fixed amount of compute, rather than using the same recipe everywhere. | §3.1 |
| **conditional harm** | The rate at which a selector overturns an already-correct answer into a wrong one — the cost side of the gain equation, easy to omit when only reporting net accuracy. | §3.4 |
| **conditional-regret functional** | A measure of achievable routing gain that accounts for how much regret remains even after optimal routing, not just how separable the router's scores are — contrasted with plain AUC. | §3.4 |
| **confabulation** | This paper's term for hallucinations caused by the model lacking the relevant knowledge, as opposed to other causes -- the class semantic entropy targets. | §3.6 |
| **confidence elicitation** | Getting a model to output a usable confidence score, whether by prompting for a verbalized number, sampling and measuring agreement, or reading internal states. | §3.6 |
| **confidence estimation module (CEM)** | In ASR, a small auxiliary network trained on top of a (typically frozen or jointly fine-tuned) sequence model to predict whether each output token/word is correct, used as a better confidence signal than the decoder's raw softmax probability. | §3.9 |
| **confidence model (docking)** | In DiffDock, a separately trained model that scores each diffusion-sampled ligand pose for how likely it is to be a good (accurate) docking prediction, used to rank and select among the sampled poses — the docking field's name for a candidate-scoring verifier. | §3.9 |
| **conformal prediction** | A distribution-free statistical framework for producing prediction sets/intervals with a guaranteed coverage probability, used here as a black-box VLM uncertainty measure. | §3.6 |
| **Contrast-Consistent Search (CCS)** | An unsupervised method for finding a 'truth direction' in hidden activations by requiring a statement and its negation get logically consistent probabilities (summing to ~1), with no labels or the model's own stated answers. | §3.6 |
| **control task** | An auxiliary task built from the same inputs as the real task but with randomly assigned outputs. Because the labels are random, the only way a probe can do well on it is by memorizing input-specific patterns (like word identity), not by reading real structure in the representation. | §3.3 |
| **Correctness Alignment / Ambiguity Calibration** | The two mechanisms the Bayesian Decoding Game uses to reach consensus: aligning on which output is 'correct' and calibrating how confident to be given ambiguity, both done through the game's iterative process rather than training. | §3.5 |
| **correlation ceiling** | The limiting 'effective' number of independent samples implied by how correlated repeated draws are for the same question, governed by the intraclass correlation ρ; reached quickly even if the raw sample count n keeps growing. | §3.4 |
| **counterfactual entity perturbation** | Deliberately altering a specific entity mentioned in a model's output (e.g. swapping 'left lung' for 'right lung') and re-checking whether the visual grounding model still finds equally strong evidence for it, to test whether the original grounding was genuinely evidence-based or spurious. | §3.6 |
| **coverage (pass@k)** | Fraction of questions for which at least one of the k samples is correct. It is an upper bound on what any selector can achieve; in our repo we call it oracle@8. | §3.1 |
| **coverage (policy sense)** | In inference-time alignment theory, the pre-trained model's probability of ever producing a high-quality response for a prompt — related to, but a distinct usage from, the pass@k "coverage" used elsewhere in this category. | §3.4 |
| **coverage / pass@k** | The fraction of problems for which at least one of k independently-sampled answers is correct — what an all-knowing oracle selector over the pool would achieve. | §3.4 |
| **coverage coefficient C*** | A distribution-mismatch constant bounding how much the sampling policy under- or over-represents the best responses relative to an ideal comparator policy. | §3.4 |
| **credibility evaluation** | Assessing whether a model's stated reasoning process is a trustworthy explanation of how it actually reached its answer, separate from whether the final answer itself is correct. | §3.7 |
| **curriculum learning** | Training in a deliberately ordered sequence of stages/tasks (typically easy-to-hard or coarse-to-fine), rather than presenting all training data in one undifferentiated pass. | §3.7 |
| **data synthesis (for leakage mitigation)** | Generating new question variants (rather than reusing scraped exam questions verbatim) specifically to reduce the chance a benchmark item was already seen during a model's pretraining. | §3.8 |
| **data-processing inequality** | Information-theory rule that post-processing a signal cannot increase the information it carries about a hidden variable. | §3.1 |
| **decision-state displacement** | How much a model's internal hidden state at its pre-answer decision point shifts/moves when a harder or more semantically confusing alternative is presented, compared to a baseline (meaning-preserving) alternative — used here as a signal of internal "stress" that behavioral accuracy alone would not reveal. | §3.3 |
| **decodability** | How well a linear gate can rank a question's correct candidate answers above its incorrect ones, measured in a way that removes "question-identity leakage" (see next term). High decodability predicts that hidden-state selection will beat majority voting; low decodability predicts the opposite. | §3.3 |
| **decoding algorithm** | How single tokens are chosen from the model's next-token distribution (greedy, sampling, beam). | §3.1 |
| **describe-then-diagnose** | Two-stage prompting: first get a neutral description of the image, then ask a (text) model for the diagnosis from that description. | §3.1 |
| **Diffusion Transformer (DiT)** | A diffusion-based image/video generator built from transformer blocks instead of the older U-Net architecture; like a language transformer, it has intermediate hidden representations that a probe/verifier can read directly. | §3.3 |
| **discrete semantic entropy (DSE)** | Semantic entropy computed from discrete counts of how many of N sampled responses fall into each meaning-cluster (rather than from continuous log-probabilities), which works for black-box models where only text output, not logits, is available. | §3.6 |
| **discriminative quality / AUROC** | How well a confidence score separates correct from incorrect predictions (area under the ROC curve); distinct from calibration, which measures whether the confidence value matches the true probability of correctness. | §3.6 |
| **discriminative reranking** | Training a separate model to re-score/re-order a fixed list of candidates from a generative model, using features or supervision the generative model's own training did not have access to — as opposed to changing the generative model itself. | §3.9 |
| **Discriminative verifier** | A verifier trained as a classifier: it maps a candidate answer to a single scalar score (e.g. via BCE loss against a correct/incorrect label), with no generated text. Our MLP probe is discriminative. | §3.2 |
| **discriminative vs. text-generative VLM** | A discriminative model selects from a closed label set; a text-generative model writes free-text output token by token. | §3.8 |
| **distillation (training)** | Training a model to match the output distribution of a larger/stronger 'teacher' model, rather than (or in addition to) matching raw ground-truth labels. | §3.7 |
| **distilled reasoning traces** | Step-by-step reasoning text generated by a stronger 'teacher' model, then used as training targets to teach a smaller/weaker 'student' model to reason similarly. | §3.7 |
| **distribution-free guarantee** | An error bound that holds without assuming a specific data distribution, typically via conformal-style calibration on held-out data. | §3.6 |
| **dORM / dPRM / gORM / gPRM** | This paper's shorthand for the four cells of the 2x2 taxonomy: discriminative-outcome, discriminative-process, generative-outcome, generative-process reward models. Our own probe is a dORM. | §3.2 |
| **DPO (Direct Preference Optimization)** | A way of aligning a language model to preference data WITHOUT training a separate explicit reward model — the policy itself implicitly defines a reward via its log-probability ratio to a reference model. RewardBench treats DPO models as reward models by using this implicit reward. | §3.2 |
| **dynamic resolution** | Letting the number of image tokens vary with the image's native resolution/aspect ratio, instead of always resizing to one fixed grid. | §3.7 |
| **Early stopping (in best-of-N)** | Stopping the sampling of further candidates once the verifier is confident enough in its current best pick, to save inference compute/latency, rather than always sampling the full fixed N. | §3.2 |
| **effective ensemble dimensionality** | The number of behaviourally-independent voters an ensemble is worth once shared, family-correlated errors are accounted for — can be far smaller than the nominal number of models. | §3.4 |
| **effective number of samples (n_eff)** | n_eff = n / [1+(n−1)ρ]: converts n correlated draws into an equivalent count of independent draws — the real information content of a sampling run. | §3.4 |
| **EHRQA** | Question-answering over electronic health record (EHR) text -- clinical notes, lab reports, etc. -- as opposed to imaging. | §3.7 |
| **EigenScore** | A hallucination-detection score computed from the eigenvalues of the covariance matrix of multiple sampled responses' internal-state embeddings, measuring how semantically consistent or diverse the responses are directly in embedding space. | §3.6 |
| **Embedding-Based Agreement (EBA)** | A training-free method that clusters sampled generations by their embedding vectors and returns the one nearest the centroid of the largest (most agreed-upon) cluster. | §3.5 |
| **evidence magnitude** | How much a model's per-token predictions shift when the image is included versus a text-only version of the same prompt -- large shifts indicate the model is actually using visual evidence rather than relying on language priors. | §3.6 |
| **exact match (EM) scoring** | Scoring an answer correct only if it matches the gold answer's text exactly (often after light normalization), as opposed to a softer LLM-judge score. | §3.8 |
| **exact replay vs distributional reproducibility** | Reproducing the identical samples bit-for-bit vs reproducing the same distribution of results across reruns. | §3.1 |
| **execution-based verification / clustering** | Using a program's actual runtime behaviour (does it run? does it match example outputs? do its outputs cluster with other candidates') as a correctness signal, instead of a trained model's prediction. Available in code generation because programs are executable; not available for free-text medical answers, which lack an executable ground truth. | §3.9 |
| **expected calibration error (ECE)** | A single number summarizing miscalibration: split predictions into confidence bins, average the gap between mean confidence and actual accuracy per bin, weighted by bin size. | §3.6 |
| **expert AGI (framing)** | The paper's stated goal: benchmarking progress toward multimodal models that can perform at the level of a human expert across many academic disciplines, as a step toward general intelligence. | §3.8 |
| **exponentiated power law** | The curve c = exp(a·k^b) that fits how coverage grows with the number of samples k. | §3.1 |
| **exponentiated power law (coverage scaling)** | A fitted curve coverage(k) ≈ exp(a·k^b) describing how coverage rises with more samples: fast at first, then with diminishing but non-saturating returns. | §3.4 |
| **external verification** | Using a separate model (not the generator) to score candidate answers. | §3.1 |
| **F1 inter-annotator agreement** | A score combining precision and recall used here to measure how consistently different human annotators agreed on paraphrased questions or ratings. | §3.8 |
| **failure prediction** | Using a confidence score to predict, in advance, whether a specific answer will turn out to be wrong. | §3.6 |
| **FAIR data principles** | Findable, Accessible, Interoperable, Reusable -- a standard for making research datasets properly documented and machine-usable. | §3.8 |
| **feature clipping** | Truncating unusually large activation values in a model's internal states at inference time, used here to reduce overconfident (and often hallucinated) generations. | §3.6 |
| **FLOPs-matched comparison** | Comparing two systems only after equalising their total floating-point operations, so a small model with many samples is charged the same as a big model with one. | §3.1 |
| **forced-choice A/B setup** | An evaluation design where the model must pick between exactly two options (here, a correct caption vs. a stress/distractor candidate), rather than freely generating or choosing among many; running it with the two options in both orders (swapped) checks whether the model's choice is driven by content or by position. | §3.3 |
| **four-quadrant diagnostic map** | A classification of each generated statement by two axes -- is the text itself factually plausible, and is it actually grounded in the image evidence -- yielding four categories used to separate different hallucination types. | §3.6 |
| **Fréchet mean** | A generalization of the arithmetic mean to non-Euclidean or weighted settings — the point that minimizes total (weighted) squared distance to a set of points, used here to find the 'semantic center' of a set of answer embeddings. | §3.5 |
| **gating mechanism** | A learned per-token weight (here, a sigmoid output) that decides how much each token contributes to a pooled score, instead of averaging all tokens equally. It lets the model down-weight uninformative tokens when combining a whole sequence into one number. | §3.3 |
| **generation-verification gap** | The difference between how well a model can produce a correct answer and how well it can recognize a correct one it is shown — the theoretical quantity that makes self-improvement or best-of-N selection possible at all. | §3.4 |
| **generative VQA (vs. classification-style VQA)** | Answering by generating free text token-by-token, as opposed to treating VQA as a classification problem over a fixed answer vocabulary. | §3.8 |
| **Generator-verifier coupling** | The degree to which a verifier's errors are correlated with the underlying generator's errors, because they share weights, training data, or representations. High coupling means the verifier's errors happen exactly when you most need it to catch a mistake. | §3.2 |
| **GenRM (generative verifier / generative reward model)** | A verifier trained with the same next-token-prediction objective as normal language model fine-tuning, so it 'answers' whether a candidate is correct by generating text (e.g. a Yes/No token, optionally after a chain-of-thought), instead of outputting one scalar via a classifier head. | §3.2 |
| **Goodhart's law** | "When a measure becomes a target, it ceases to be a good measure." Cited as the general principle behind reward-model overoptimization: once you optimize hard against a proxy for quality, the proxy stops tracking real quality. | §3.2 |
| **groundable / grounding explanation** | An explanation that points to the specific image region (a bounding box or mask) supporting an answer, not just a text justification. | §3.8 |
| **GRPO** | Group Relative Policy Optimization, a reinforcement-learning fine-tuning method (used e.g. by DeepSeek-R1 and many recent reasoning models) that updates a model by comparing a group of its own sampled outputs' rewards against their group average, without needing a separately-trained value/critic model. | §3.3 |
| **GSM8K** | 8.5K grade-school math word problems; the standard early benchmark for verifier work. | §3.1 |
| **hallucination-aware calibration (HAC)** | This paper's method of feeding a separate vision-grounding hallucination-detection score into the calibration function alongside raw model confidence, improving both calibration and ranking quality. | §3.6 |
| **HedgeTune** | An algorithm for choosing how strongly to trust a proxy reward at inference time, to avoid over-optimizing it. | §3.4 |
| **hidden size** | The width (dimensionality) of the vector representing each token at every layer of the network -- bigger hidden size means each token carries more numbers, generally more capacity. | §3.7 |
| **hidden state / internal state / activation** | The vector of numbers a transformer computes at a given layer and token position while processing input or generating output — the model's internal, non-text representation at that point. | §3.3 |
| **hierarchical hallucination categorization** | Classifying hallucinations by type and severity (e.g. minor phrasing error vs a clinically dangerous false finding) rather than treating every hallucination as equally bad. | §3.6 |
| **HyperKvasir / Kvasir-Instrument** | Earlier public GI-endoscopy image datasets (from the same Simula/Kvasir group) that Kvasir-VQA annotated with question-answer pairs. | §3.8 |
| **identifiability gap** | Coverage minus achieved selection accuracy — the fraction of questions where a correct answer exists somewhere in the sample pool but the selector could not identify it as correct. | §3.4 |
| **ImageCLEF** | A long-running annual shared-task/benchmark evaluation campaign (part of the CLEF initiative) covering image retrieval and analysis tasks, including a recurring medical VQA track. | §3.8 |
| **importance-weighted policy gradient** | A way to train the probe so it directly optimizes which completion gets chosen (the downstream sampling behavior), rather than merely fitting the probe's output to a reward value; it reweights each training example by how much more/less likely it becomes under the new (probe-adjusted) sampling policy compared to the original one. | §3.3 |
| **in-advance correctness direction** | A direction in activation space, found via a linear probe, along which a question's activations predict whether the model's eventual answer to it will be correct — estimated before the answer is generated. | §3.3 |
| **inference-time intervention** | Modifying a model's internal activations at generation time (e.g. shifting along a learned direction) to change its output behavior, as opposed to only reading activations to produce a score. | §3.6 |
| **inference-time scaling** | Spending more compute at generation/inference time (e.g. sampling more candidates and picking the best with a verifier) to improve output quality, as an alternative or complement to training a better generator. | §3.3 |
| **inference-time scaling (for generation)** | Improving output quality by generating multiple candidates and using a reward model/verifier to pick or reweight among them at inference time, rather than by training a better single-shot generator. | §3.3 |
| **Information Contribution to Residual Stream (ICR) Score** | A metric introduced by this paper measuring how much a given module's output changes (contributes to) the residual stream at a given layer, used as a dynamic, cross-layer alternative to reading a single static hidden-state snapshot. | §3.3 |
| **intelligence per watt (IPW)** | Task accuracy divided by average power draw; accuracy per joule is the per-query energy version. | §3.1 |
| **internal-external discrepancy** | A case where a model's internal representation appears to encode the correct answer (e.g., a probe or analysis can recover it), yet the model's actual generated output is a different, incorrect answer — evidence that generation and internal "knowledge" can come apart. | §3.3 |
| **Judge (as distinct from reward model)** | In this literature 'judge' and 'reward model' are often used near-interchangeably for a model that scores/compares candidate outputs; 'judge' sometimes specifically implies a prompted, generative LLM doing the scoring (vs. a purpose-trained discriminative reward model), though usage varies by paper. | §3.2 |
| **judge accuracy** | Accuracy scored by having another (usually larger) model judge whether a free-text answer is correct, rather than exact string match. | §3.5 |
| **judge alignment (score vs. rank)** | Two different senses in which a judge can 'agree' with humans: assigning similar absolute scores, versus placing models in the same relative order -- a judge can be good at one and bad at the other. | §3.8 |
| **Kaplan FLOPs formula** | The convention of counting roughly 2 x parameters floating-point operations per processed token (6 x parameters per training token). | §3.1 |
| **KL divergence** | A measure of how different one probability distribution is from another; here, how far best-of-n moves the model away from its own natural outputs. | §3.1 |
| **knowledge base / knowledge graph grounding** | Questions that require looking up a structured fact (e.g. 'which organ system does X belong to') from an external knowledge base, not just reading the image. | §3.8 |
| **knowledge erasure (linear orthogonalization)** | Removing a specific piece of information from a representation by projecting the representation so that it has zero component along a learned direction associated with that information (here, a "hallucinated object" direction) — mathematically, subtracting the component of the vector that points along the unwanted direction. | §3.3 |
| **knowledge-based VQA (KB-VQA)** | A VQA variant where answering correctly requires retrieving and reasoning over an external structured knowledge base, not just perceiving the image. | §3.8 |
| **KV cache** | The stored key/value tensors for every token already in context; reused at each decoding step so earlier tokens are not recomputed. | §3.1 |
| **KV-cache** | The stored key/value attention tensors from previously generated tokens, kept in memory so a transformer doesn't recompute them at every new generation step; its size grows with context length and is a major memory cost for long-context models. | §3.7 |
| **language prior** | A model's learned tendency to answer based on what's statistically likely from the text/question alone, rather than genuinely attending to the image -- a major cause of LVLM hallucination and yes-bias. | §3.6 |
| **latent verifier** | A verifier (candidate-scoring model) that reads a base model's internal/latent hidden states rather than its output text, avoiding the cost of re-processing generated text as new input. | §3.3 |
| **Lazy verifier** | The finding that when a VLM is used as its own verifier, it pays measurably less attention to the image than it did as the generator — i.e. it is 'reading the text of its own answer' more than re-checking the picture. | §3.2 |
| **Le Cam lower bound** | A minimax statistical lower bound establishing the best possible certification guarantee any protocol could achieve at a given sample size — used here to show RouteGuard's certification bracket is tight. | §3.4 |
| **leaf-level scaling / terminal reducer** | Sample N complete answers independently, then apply one function (vote, verifier argmax) to pick the output. Best-of-N is leaf-level scaling with an argmax-over-verifier reducer. | §3.1 |
| **leniency bias** | A judge's systematic tendency to rate outputs more favorably than a strict/human standard would. | §3.8 |
| **linear separability** | Whether classes can be separated by a straight line/hyperplane in a given representation space. Higher linear separability means a simple linear probe can already tell the classes apart, without needing a non-linear model. | §3.3 |
| **linguistic invariance** | The property that many different surface strings express the same underlying meaning; semantic entropy exploits this by clustering before measuring uncertainty. | §3.6 |
| **LLM-as-a-judge** | Using a strong LLM to score or compare the outputs of other models/systems, as a cheaper substitute for human evaluation. | §3.8 |
| **LMM-as-a-Judge / LLM-as-judge** | Using a (large multimodal or language) model, prompted to evaluate a candidate answer's quality/correctness, as the scoring mechanism — a generative alternative to a trained discriminative reward model. | §3.2 |
| **local identifiability** | Whether a critic/comparator, without access to the ground-truth verifier, can actually pick out the correct candidate from the pool. | §3.4 |
| **local inference** | Running the model on a user's own device (laptop, workstation) rather than a cloud datacentre. | §3.1 |
| **local vs. global attention layers** | Local attention layers let each token attend only to a nearby window of other tokens (cheap); global attention layers let every token attend to every other token in the full context (expensive but captures long-range dependencies). Mixing mostly-local with occasional global layers bounds the KV-cache cost. | §3.7 |
| **localisation confidence** | In object detection, an estimate of how well a predicted bounding box matches the true object location (e.g. predicted IoU with ground truth), distinct from the classifier's confidence that the box contains the right object class. | §3.9 |
| **logit lens / vocabulary projection** | A technique for reading an intermediate (not-yet-final) internal representation by projecting it through the model's own output (unembedding) layer, producing a distribution over vocabulary tokens as if that intermediate layer were the final one — used here to read out, layer-by-layer, how "confident" the model's image representation already is about specific objects, before any text is generated. | §3.3 |
| **LVLM (large vision-language model)** | A large neural network that takes both images and text as input and generates text output; synonym used interchangeably with VLM in much of this literature. | §3.8 |
| **M-RoPE (Multimodal Rotary Position Embedding)** | An extension of rotary position embeddings (a way transformers encode token order) so that position information is shared coherently across text, image, and video tokens in one sequence. | §3.7 |
| **majority vote / marginalization over reasoning paths** | Treating each sampled reasoning path as a noisy vote for its final answer and summing the votes to find the most-supported answer. | §3.5 |
| **Majority voting / self-consistency** | Selecting the most frequent answer among N sampled candidates, rather than using a learned reward model to pick one. One of our project's own baselines. | §3.2 |
| **MCTS (Monte Carlo Tree Search)** | A search algorithm that builds a tree of possible next reasoning steps and uses random rollouts to estimate which branches are promising; used here to help automatically construct step-level training labels for a PRM. | §3.2 |
| **MedMNIST** | A collection of small, standardised medical image classification datasets (originally 28x28) used for lightweight benchmarking. | §3.1 |
| **MedPix** | A public, freely searchable database of radiology teaching-file images (with case info) maintained by NLM/AFIP; the image source for VQA-RAD. | §3.8 |
| **MedSigLIP** | A medically fine-tuned version of the SigLIP vision encoder (see the SigLIP card), used as MedGemma's image-understanding backbone. | §3.7 |
| **MedVInT** | The generative medical-VQA model proposed alongside PMC-VQA: a vision encoder aligned to a pretrained LLM so the model writes free-text answers instead of picking from a fixed label set. | §3.8 |
| **meta-generation** | An algorithm that calls the generator several times and combines whole sequences (best-of-N, self-consistency, refinement, tree search). | §3.1 |
| **Min-K%++** | A membership-inference technique that flags training-set membership by looking at whether a model assigns unusually high probability to the lowest-probability (tail) tokens of a text, relative to what a model that never saw the text would do. | §3.8 |
| **Minimum Bayes Risk (MBR) decoding** | Choosing the output that minimizes expected error (or maximizes expected quality) against a distribution of possible correct outputs, typically approximated using a pool of sampled candidates, instead of choosing the single highest-probability output. | §3.5 |
| **Misleading tier** | The subset of questions where a correlated majority error among ensemble members drives accuracy to 0% even though the single best model would have answered correctly. | §3.4 |
| **mixed preference optimization (MPO)** | A post-training method that optimizes a model against a mix of preference signals (e.g. multiple types of preference data/objectives combined) rather than a single reward or preference source. | §3.7 |
| **mixture of experts (MoE)** | An architecture where only a subset of parameters ('experts') is active per token, so a large model can be cheap per token. | §3.1 |
| **MLLM** | Multimodal Large Language Model -- an LLM extended to take images (and sometimes other modalities) as input, not just text. | §3.7 |
| **modal ceiling** | The probability that the single most-frequently-sampled answer is actually correct — a hard cap on majority-vote/self-consistency accuracy that does not improve with more samples. | §3.4 |
| **model quality assessment / estimation of model accuracy (EMA)** | A standing category in the CASP (Critical Assessment of protein Structure Prediction) competitions: given a predicted structure, estimate how accurate it is (globally and per-residue) without knowing the true structure. pLDDT is AlphaFold's own built-in EMA output. | §3.9 |
| **multinomial logistic regression** | A linear classifier generalized from binary logistic regression to more than two classes (here, the multiple-choice answer options), producing a probability distribution over all classes via softmax. | §3.3 |
| **n-best list** | The top N candidate outputs a generative model produces for one input (e.g. N decoded transcripts, N parse trees, N sampled answers) — the pool a reranker or verifier chooses from. | §3.9 |
| **N-best list / reranking** | The set of N sampled candidates, and the act of ordering them with a scoring function to pick the top one. | §3.1 |
| **native multimodal pretraining** | Training vision and language capability together from the start of pretraining, as opposed to first pretraining a text-only LLM and only later 'bolting on' vision (post-hoc adaptation). | §3.7 |
| **near-neighbour overlap (embedding-space)** | Detecting likely-duplicate images by checking whether an evaluation image's embedding (e.g. from SigLIP) has an extremely close match in a pretraining-adjacent corpus. | §3.8 |
| **negative / pre-domain control** | Running the same detector on a model known not to have plausible exposure to the target domain (here, BLIP-2 for medical VQA) to check whether the detector's 'positive' signal is meaningful or just an artifact of the method. | §3.8 |
| **non-monotone scaling** | Performance that first improves then degrades as you add more calls/samples. | §3.1 |
| **NOTA (None-Of-The-Above) perturbation** | A stress test for multiple-choice VQA that removes/hides the correct answer option, forcing the model to pick a wrong one, to see whether its confidence/uncertainty score reacts appropriately to being forced into an error. | §3.6 |
| **number of layers** | How many transformer blocks a token's representation passes through, end to end, before the final output. A 'layer 20 of 28' probe reads the representation after it has been refined by 20 of the model's 28 blocks. | §3.7 |
| **NVML** | NVIDIA's management library; the tool used to sample GPU power draw at fixed intervals. | §3.1 |
| **o1-like reasoning model** | A model trained to produce extended step-by-step reasoning before its final answer, in the style of OpenAI's o1. | §3.8 |
| **object hallucination** | An LVLM describing an object as present in an image when it is not actually there. | §3.6 |
| **object hallucination (OH)** | When a vision-language model describes or claims an object is present in an image that is not actually there (or gets its attributes/relations wrong) — a specific, well-studied failure mode distinct from general answer incorrectness. | §3.3 |
| **open-ended VQA** | A visual question answering format where the model must produce free text as its answer, rather than selecting from given options. | §3.8 |
| **oracle gap** | The accuracy of an all-knowing ("any@k") oracle selector over a fixed candidate pool minus a reference/single-sample baseline accuracy — the total headroom any selector could ever capture from that pool. | §3.4 |
| **oracle router** | A hypothetical router that always picks the best-performing model for each individual query — the ceiling routing methods are compared against. | §3.4 |
| **oracle score** | The accuracy if you could magically always pick the best candidate out of the N samples — the upper bound any selection method is trying to approach. | §3.5 |
| **ORM (outcome reward model)** | A verifier/reward model that only looks at the final answer (the outcome) to score correctness, as opposed to scoring each intermediate reasoning step. Our own MLP probe is an ORM: it scores each full candidate answer once, using hidden states pooled over the candidate's own generated tokens. | §3.2 |
| **outcome / intrinsic reward** | A scalar score assigned to a whole candidate output reflecting how good it is judged to be, used to rank or select among several sampled candidates — as opposed to a reward given for each intermediate reasoning step (a process reward). | §3.3 |
| **outcome reward model (ORM)** | A verifier that scores only the finished answer as correct/incorrect. Our probe is ORM-style. | §3.1 |
| **Outcome-based vs process-based feedback** | Two ways to supervise a reasoning model: reward/penalize only the final answer's correctness (outcome-based) or reward/penalize each individual reasoning step (process-based). | §3.2 |
| **P(IK) / "I know"** | The model's predicted probability that it will be able to answer a question correctly at all, estimated before/without seeing any specific candidate answer. | §3.6 |
| **P(True)** | The model's own predicted probability that a specific proposed answer is correct, elicited by prompting it to judge its own (or another) answer. | §3.6 |
| **pairwise error correlation (rho)** | The standard, but shown-to-be-insufficient, diagnostic for how correlated two models' errors are; cannot by itself identify the all-wrong tail rate beta. | §3.4 |
| **pairwise reranking** | Ranking candidates by repeatedly comparing them two at a time and predicting which of the pair is better, then aggregating those pairwise judgments into an overall order. | §3.5 |
| **pairwise statistics (black-box reranking)** | Similarity or agreement measures computed between two generated outputs using only their text (e.g. overlap, embedding distance), without needing model internals or an extra trained model. | §3.5 |
| **Pareto frontier** | The set of configurations that are not beaten on both accuracy and cost by any other configuration. | §3.1 |
| **Pareto-optimal (cost vs accuracy)** | A configuration no other configuration beats on both accuracy and cost at once. | §3.1 |
| **parseability** | Whether the model's output contains an extractable final answer; unparseable chains count as wrong and can dominate a TTS result. | §3.1 |
| **Partial verification** | Verifying only a checkable piece of a candidate answer (a specific claim, value, or region) rather than judging the whole answer's correctness at once — used when no reliable whole-answer verifier exists. | §3.2 |
| **pass@1** | The probability a single random sample is correct; used here as the per-question difficulty measure. | §3.1 |
| **Pass@k** | A metric: out of N sampled candidates, you are allowed to submit up to k of them, and you get credit if ANY of the k is correct. Distinct from Best-of-N's usual setting of submitting exactly one final answer (k=1). | §3.2 |
| **Pass@k (regret sense)** | An evaluation that allows the system to submit up to k answers and scores only the best of them — looser than single-answer accuracy. | §3.4 |
| **pass@k vs maj@k** | Whether any of k samples is correct vs whether the majority answer is correct; the gap is the room a better reducer could recover. | §3.1 |
| **perception-focused benchmark** | A task where the difficulty is seeing what is in the image, not multi-step reasoning about it. Most medical VQA is perception-heavy. | §3.1 |
| **Platt scaling** | A post-hoc calibration method that fits a logistic regression from a model's raw confidence score to P(correct) using held-out labelled data; rescales confidence without changing which candidate ranks highest. | §3.6 |
| **pointwise vs listwise vs pairwise scoring** | Three ways to rank candidates: pointwise scores each one alone; pairwise compares two at a time; listwise looks at the whole set together to produce a ranking or selection. | §3.5 |
| **Pointwise vs pairwise scoring** | Pointwise: the reward model scores one candidate answer at a time, independently (our approach). Pairwise: the reward model compares two candidates and predicts which is better, without necessarily producing an absolute score for either alone. | §3.2 |
| **pointwise vs. pairwise vs. listwise ranking** | Three ways to train a ranker/reranker. Pointwise: train on each candidate independently against a correctness/quality label (what our BCE probe does). Pairwise: train on which of two candidates is better (Bradley-Terry-style). Listwise: train directly on the ranking/ordering of a whole list at once. Collins & Koo's boosting reranker and much of the discriminative-reranking literature use pairwise or listwise objectives; our earlier project iteration tested a Bradley-Terry pairwise objective before settling on pointwise BCE. | §3.9 |
| **policy model** | The generator that produces candidate answers (our Lingshu-7B). The word comes from reinforcement learning. | §3.1 |
| **polygenic risk score** | A single number summarizing an individual's genetic predisposition to a disease, computed by combining the effects of many genetic variants. | §3.8 |
| **POPE (Polling-based Object Probing Evaluation)** | An evaluation protocol that asks a model direct yes/no questions ('Is there a <object> in the image?') instead of parsing free-form generated captions, giving a more stable and less prompt-sensitive measurement of object hallucination. | §3.6 |
| **position bias** | A judge's tendency to favor a candidate because of where it appears in the list it is shown, rather than its actual quality. | §3.5 |
| **position bias (judging)** | A judge model's tendency to favor whichever answer is shown first (or second) in a pairwise comparison, independent of actual quality. | §3.8 |
| **post-hoc selector** | A scorer applied after all candidates are generated (our probe is one); contrast with search, which intervenes during generation. | §3.1 |
| **pre-generation vs. post-generation probing** | Pre-generation probing reads a model's internal state after it has processed the prompt/image but BEFORE it has generated any answer text — useful for predicting risk or difficulty in advance, at the cost of not knowing what the model would actually say. Post-generation (or per-candidate) probing, like our verifier, reads internal states produced WHILE generating a specific candidate answer, so it can score that particular candidate rather than the question in the abstract. | §3.3 |
| **predictability bottleneck** | Routers learning only coarse, per-model average performance rather than fine-grained, per-query signal, which caps achievable accuracy regardless of algorithm. | §3.4 |
| **prefill vs decode** | Prefill = processing the prompt (and image) once; decode = generating output tokens one at a time. Both cost FLOPs; honest accounting includes both. | §3.1 |
| **prefix-level scaling** | Spend compute on unfinished partial answers (beam search, tree search), pruning before completion. | §3.1 |
| **PRM (process reward model)** | A verifier that scores each intermediate reasoning step of a candidate solution, not just the final answer. Contrast with ORM, which scores the outcome only. | §3.2 |
| **PRM-guided beam search** | Keep the top-scoring partial reasoning chains according to a process reward model, extend them, repeat. | §3.1 |
| **probing classifier / probe** | A small classifier (often linear) trained to predict some property from a frozen (unmodified) model's internal activations. The base model's weights are never updated by this training — only the small probe is trained. | §3.3 |
| **probing evaluation (negation questions)** | Pairing a normal question with a variant that negates or falsifies an attribute (e.g. asking about a finding that is not actually present) to check whether the model is really grounding its answer in the image or just pattern-matching. | §3.8 |
| **procedural diagnosis** | Requiring a model to answer a structured sequence of sub-questions (modality, organ, finding, abnormality, location) for one image, rather than one free-standing question. | §3.8 |
| **process reward model (PRM)** | A verifier that scores each intermediate step of a solution, giving 'a prediction of the correctness of each intermediate step in a solution, rather than just the final answer'. | §3.1 |
| **projector / connector / merger** | A small network (here an MLP) that maps the vision encoder's output space into the language model's token-embedding space, so image features can be fed into the LLM alongside text tokens. | §3.7 |
| **prompt tuning** | Learning a small number of continuous 'prompt' vectors prepended to the input while keeping the rest of the model's weights frozen, as a lightweight alternative to full fine-tuning. | §3.6 |
| **proposal coverage** | Whether the pool of sampled candidates contains a correct answer at all, prior to any selection step — the same concept as coverage/pass@k elsewhere in this category, named differently here. | §3.4 |
| **PubMedVision** | A 1.3M-sample medical VQA dataset built by having GPT-4V clean up and reformat noisy image-caption pairs scraped from PubMed articles. | §3.7 |
| **query-token representation** | In vision-language decoder architectures that use learned "query tokens" to compress/summarize visual information for the text decoder, the internal representation at those query-token positions, integrating visual and textual context, read here just before generation begins. | §3.3 |
| **question-identity leakage** | A failure mode of probe evaluation where the probe appears to detect correctness but is actually partly just recognizing WHICH QUESTION it is looking at (because several sampled candidates for the same question share features), rather than reading a genuine correctness signal that would generalize. "Question-grouped evaluation" (keeping all candidates for one question in the same train/test split) removes this leakage. | §3.3 |
| **question-only probe** | A probe trained on the model's internal state right after it has read/processed the QUESTION but before it has generated any part of an answer — it can only ever reflect the model's anticipated difficulty with the question, not information about any specific candidate answer. | §3.3 |
| **ranked voting (Instant-runoff / Borda count / mean reciprocal rank)** | Voting schemes that use each voter's (here, each sample's) full preference order over options, not just their top pick, to decide a winner — used here to aggregate over LLM samples that each propose a ranked list of candidate answers. | §3.5 |
| **reasoning-step boundary** | A token position marking the end of one step of a multi-step reasoning chain (e.g. right before a paragraph break), used here as the specific positions whose hidden states get pooled, instead of every single generated token. | §3.3 |
| **recoverability asymmetry** | The proven fact that a single-commit selection floor unclosable by any single-commit router can nonetheless be closed by resampling the same already-chosen model — the floor is about commitment, not missing cross-model information. | §3.4 |
| **recoverable mass** | The fraction of questions where the reference/single-sample answer is wrong but at least one of the k sampled candidates is correct — the numerator of the oracle gap. | §3.4 |
| **rejection sampling (as used for scoring candidates)** | Generating multiple candidate outputs and using a scoring rule to keep (accept) some and discard (reject) others — here, used loosely to mean using the probe's score to pick among open-ended generations, rather than the stricter statistical technique of the same name. | §3.3 |
| **rejection sampling (in this paper's sense)** | Best-of-n: sample n, keep the top reward-model pick, discard the rest. | §3.1 |
| **reliability diagram** | A plot of predicted confidence (binned) against observed accuracy per bin; a perfectly calibrated model lies on the diagonal. | §3.6 |
| **repeated sampling** | Drawing many independent answers from the same model for the same question, at temperature > 0. | §3.1 |
| **resampling / best-of-N** | Drawing N candidate answers and returning the one a verifier accepts or scores highest. | §3.4 |
| **rescoring vs. reranking** | Rescoring assigns each n-best candidate a new score (often combined with the original model score); reranking is the resulting reordering/selection step. The terms are used near-interchangeably in speech and MT literature; our best-of-N verifier does exactly this over a pool of 8 sampled answers. | §3.9 |
| **residual stream** | The running sum of vector updates that passes through every layer of a transformer; each layer reads from and adds to this shared stream rather than replacing it, so a token's final hidden state is literally the accumulation of every layer's contribution along the way. | §3.3 |
| **reward hacking** | Optimizing hard against an imperfect proxy reward/verifier in a way that degrades true response quality. | §3.4 |
| **reward model** | A model trained on human preference comparisons to score answers; a verifier for quality rather than correctness. | §3.1 |
| **reward over-optimisation** | When picking the candidate with the highest verifier score finds answers the verifier likes but that are not actually better — the verifier's mistakes get selected for. | §3.1 |
| **reward-aware** | A test-time policy that changes its choice of method depending on which verifier (reward model) it is paired with. | §3.1 |
| **Reward-model benchmark** | A dataset built specifically to evaluate how good a reward model itself is (e.g. does it pick the better of two responses on cases with a clear, verifiable right answer), as opposed to a benchmark for evaluating the generator model. | §3.2 |
| **Reward-model overoptimization / reward hacking** | The phenomenon where optimizing a policy too hard against an imperfect (proxy) reward model improves the proxy score while true quality (measured by a gold/ground-truth standard) stops improving or gets worse, because the policy learns to exploit quirks of the proxy rather than genuinely improve. | §3.2 |
| **risk / utility function (in MBR)** | A function scoring how good or bad a candidate output is, usually defined via similarity to other plausible outputs; MBR decoding picks the candidate with the best expected score under this function. | §3.5 |
| **sampling budget** | The number of candidate generations drawn per question; the main cost driver in best-of-N / self-consistency methods. | §3.5 |
| **sampling-and-voting** | Running a model multiple times independently and combining the runs by majority vote, the same core idea as self-consistency but framed as multiple 'agents'. | §3.5 |
| **selective prediction** | A framework where a model can withhold a prediction on some inputs (defer them) for lower error on the rest; used here only as an evaluation lens. | §3.6 |
| **selectivity** | The gap between a probe's accuracy on the real task and its accuracy on the matched control task. A selective probe is accurate on the real task but near-chance on the control task, meaning its real-task accuracy is credible evidence about the representation, not about the probe's own capacity to memorize. | §3.3 |
| **self-aggregation** | Show the model all its sampled chains and ask it to write one merged answer. | §3.1 |
| **self-certainty** | A confidence score for a generated response computed only from its own token-level output probability distributions (how far they are from a uniform/uncertain distribution), requiring no external model, ground truth, or extra inference call. | §3.5 |
| **self-consistency** | Sample several reasoning chains and return the most frequent final answer (majority vote). A training-free selector; our baseline. | §3.1 |
| **self-enhancement / self-preference bias** | A judge model's tendency to rate outputs it (or a closely related model) generated more favorably than equally-good outputs from elsewhere. | §3.8 |
| **self-recognition (of own generations)** | A model's above-chance ability to identify which text it (vs. another model or a human) produced. | §3.8 |
| **self-refinement** | Ask the model to critique and rewrite its own answer for several rounds (sequential scaling). | §3.1 |
| **Self-verification** | Re-invoking the same (or a similar) model in a fresh prompt/context to check whether its own previously generated answer is correct, used as a cheap alternative to a trained verifier. | §3.2 |
| **semantic entropy** | Uncertainty measured over clusters of same-meaning answers (found via mutual entailment) instead of over exact answer strings, so that paraphrases of the same correct answer count as agreement. | §3.5 |
| **semantic entropy probe (SEP)** | A small probe trained on a single generation's hidden states to predict what semantic entropy (normally requiring many sampled generations plus NLI clustering) would have been, at near-zero extra inference cost. | §3.6 |
| **semantic hierarchy (of class labels)** | A tree/taxonomy relating class labels at different granularities (e.g. 'dog' under 'animal'), used here to generate harder follow-up questions that a merely-coarse-correct answer would fail. | §3.8 |
| **semi-automated dataset construction** | Building a dataset by combining an automatic pipeline (e.g. NLP QA-pair generation from text) with a manual verification pass, rather than either fully automatic templating or fully manual human authoring. | §3.8 |
| **sequential vs parallel scaling** | Sequential = one long or iteratively revised chain; parallel = many independent samples combined afterwards. Best-of-N is parallel. | §3.1 |
| **SFT (Supervised Fine-Tuning)** | Training a model to imitate given example outputs (here, reasoning traces distilled from a stronger model), by directly minimizing the difference between the model's output and the target text. | §3.7 |
| **shallow vs deep alignment** | Two-phase vision-language pretraining: 'shallow' alignment trains only the vision encoder + projector while the LLM stays frozen (cheap, coarse); 'deep' alignment then unfreezes the whole model for finer joint tuning. | §3.7 |
| **shared prefill** | Running the prompt+image through the model once and letting all N samples branch from the same KV cache, so only decoding is repeated. | §3.1 |
| **shortcut behavior** | A model reaching the right answer via a spurious correlation or superficial cue (e.g. answer-format patterns) rather than the intended reasoning process -- looks correct on the metric but is not robust. | §3.7 |
| **side-by-side expert evaluation** | Human experts (here, radiologists) compare two candidate outputs (e.g. an AI report vs. the original human report) directly against each other rather than scoring each in isolation. | §3.8 |
| **sigmoid loss (vs. softmax contrastive)** | A way to train an image-text matching model where each image-text pair is scored independently (sigmoid, yes/no match) instead of needing to normalize a whole batch of pairs against each other (softmax) -- cheaper to scale to large batches. | §3.7 |
| **signal fidelity (MCC)** | Matthews correlation coefficient between a verifier's pass/fail verdict and the true correctness label — how trustworthy the verification signal actually is, as opposed to how large the oracle gap is. | §3.4 |
| **signalling game / equilibrium decoding** | Framing text generation and scoring as two players in a game and searching for a stable joint solution (equilibrium) between them, instead of treating generation and scoring as independent steps. | §3.5 |
| **silhouette analysis** | A method for automatically choosing how many clusters to use, based on how well-separated the resulting clusters are. | §3.5 |
| **single-commit router** | A router/policy that must commit to one model's single output per query, with no ability to resample. | §3.4 |
| **Skip-connection Cross Attention (SkipCA)** | An added attention module that lets later-layer hidden representations directly attend back to early-layer visual features (via a skip/shortcut connection), intended to strengthen how well the model's later reasoning is grounded in the original visual input when scoring a text-image pair. | §3.3 |
| **Soft Best-of-n (SBoN)** | A softened version of best-of-n selection that samples the final answer with probability related to (but not strictly argmax of) the reward score, rather than always picking the single highest-scored candidate — trades off some accuracy for reduced reward-hacking risk. | §3.2 |
| **soft scoring (vs discrete voting)** | Ranking candidates by a continuous numeric score (e.g. derived from token probabilities) instead of counting how many other candidates exactly match them. | §3.5 |
| **solution-level vs token-level verifier** | Solution-level scores the whole answer once; token-level emits a correctness estimate after every token and is trained on all of them, which regularises it. | §3.1 |
| **solvability** | The fraction of rollouts that are correct for a question; a per-question difficulty score (same idea as pass@1). | §3.1 |
| **Spectral ensembling (unsupervised)** | A family of algorithms that combine multiple noisy predictors into one better estimate by analyzing the covariance/correlation structure between their outputs, without needing any ground-truth labels. | §3.2 |
| **supervised contrastive learning** | A training method that pulls representations of same-class examples (here, responses with the same answer) close together and pushes different-class examples apart, in embedding space. | §3.5 |
| **surrogate model** | Using a separate, accessible model to estimate the confidence of a closed/inaccessible target model when its internals or logits aren't available. | §3.6 |
| **system-mediated attention** | Attention allocated to the system prompt/instruction tokens (as opposed to the image or the user's text), proposed here as a hidden driver of yes-bias when it crowds out attention to the actual evidence-bearing inputs. | §3.6 |
| **temperature scaling** | Dividing a model's pre-softmax logits by a single learned scalar before softmax, so confidence values better match true accuracy, without changing which class ranks highest. | §3.6 |
| **test-time compute / inference-time scaling** | Spending extra computation when answering a question (more samples, longer reasoning, search) instead of making the model bigger or training it longer. | §3.1 |
| **Test-time scaling (TTS)** | Spending extra compute at inference time (e.g. sampling more candidates, searching more reasoning paths) to improve accuracy, as opposed to spending compute during training. Best-of-N is one form of test-time scaling. | §3.2 |
| **text-only / language-only ablation** | Re-running a model with the image removed (or blanked) to check whether it can still 'answer' from question text alone -- a high score here indicates a dataset shortcut, not real vision. | §3.8 |
| **Tissue Source Site (TSS)** | The institution/site that originally collected a tissue sample in a resource like TCGA; different cases from the same TSS can share scanner/staining artifacts that a model could exploit as a shortcut. | §3.8 |
| **token budget** | The maximum number of reasoning tokens a model is allowed to generate; raising it is the cheapest form of sequential scaling. | §3.1 |
| **token F1** | A soft text-overlap metric for open-ended answers: treats the predicted and gold answers as bags of tokens and computes precision/recall/F1 over the overlap, rather than requiring an exact string match. | §3.7 |
| **token reduction** | Compressing the number of vision tokens fed to the LLM (e.g. by merging or pruning redundant patches), which cuts compute cost, especially important for 3D volumes and video where the naive token count would be huge. | §3.7 |
| **Tool-integrated / agentic verifier** | A verifier that can take actions (e.g. call a search/retrieval tool against an external knowledge source) WHILE deciding its correctness judgment, rather than only reading the candidate answer in one forward pass. | §3.2 |
| **training-free logit baseline** | A confidence score computed directly from the model's own output token probabilities (e.g. max softmax probability), with no extra training. | §3.6 |
| **trustworthiness benchmark** | An evaluation suite measuring multiple independent axes of model reliability (factual correctness, fairness, robustness, privacy) rather than accuracy alone. | §3.6 |
| **truthful direction** | A single direction (vector) in a model's hidden-state space along which moving representations makes outputs more truthful/less hallucinated; found here to be largely shared across different LVLMs, suggesting a somewhat universal internal correlate of truthfulness. | §3.3 |
| **truthfulness encoding / truthfulness direction** | A hypothesized direction or pattern in a model's internal representation space that correlates with whether its output is true or false. This paper's key caveat: such directions are found to be "multifaceted" (different across datasets/tasks) rather than one single universal direction that works everywhere. | §3.3 |
| **TTFT / TPOT** | Time to first token (prefill latency) and time per output token (decode latency); the two standard latency metrics. | §3.1 |
| **uncertainty as diagnostic vs safety net** | This paper's distinction between using uncertainty to catch a specific failure live at inference time (safety net -- found not to work here) versus using clean-input uncertainty to identify which cases are generally fragile/failure-prone ahead of time (diagnostic -- found to work here). | §3.6 |
| **understanding vs. reasoning (as evaluated here)** | This paper's split of VQA performance into tasks needing mainly visual recognition/fact retrieval ("understanding") versus tasks needing multi-step inference over that information ("reasoning"), scored as separate sub-metrics. | §3.7 |
| **unified experimental setup (reproducibility)** | Evaluating multiple models under identical, fixed conditions (same prompts, decoding settings, data splits) so that reported score differences reflect real model differences, not incidental setup differences. | §3.8 |
| **universal self-consistency (USC)** | A version of self-consistency for free-form text: instead of a majority vote on exact-matching answers, the model itself is shown all sampled answers and asked to pick the most consistent one. | §3.5 |
| **Variational Information Bottleneck (VIB)** | A training principle that compresses a representation to keep only the information relevant to a target task (here, hallucination) while discarding everything else, by explicitly penalizing how much information the compressed representation retains about the (nuisance-heavy) input. It is a way to make a probe focus on task-relevant signal instead of memorizing irrelevant details. | §3.3 |
| **VASE metric** | The paper's robust hallucination/uncertainty metric within the HEDGE pipeline, found most reliable when paired with embedding-based clustering (full definition not extracted from the abstract). | §3.6 |
| **verbal vs. visual reflection** | This paper's terms for two components of a chain-of-thought trace: 'verbal reflection' (re-examining the reasoning in words) and 'visual reflection' (re-examining the image); they find visual reflection declines over the course of reasoning while verbal reflection does not. | §3.5 |
| **verifiable rewards (RLVR)** | Reinforcement learning where the reward is computed by an automatic checker (e.g. exact-match to a known correct answer) rather than a learned reward model or human preference. | §3.7 |
| **Verification mirage** | This paper's name for a failure regime where a self-verifier looks like it is doing useful checking (giving confident yes/no judgments) but actually has both high error and high agreement bias — it mostly just agrees with whatever the generator said, especially when the generator is wrong. | §3.2 |
| **verifier** | A model trained to predict whether a candidate answer is correct. Used to rank sampled answers; it does not generate. | §3.1 |
| **Verifier / reward model** | A second model (or a small scoring head) that reads a candidate answer and outputs a score for whether it is correct or good. Used to pick the best of several sampled candidates (best-of-N), or as a training signal. | §3.2 |
| **Verifier calibration** | Whether a verifier's output scores/probabilities match true likelihood of correctness (e.g. among candidates scored 0.8, are ~80% actually correct?), as distinct from just ranking candidates correctly relative to each other. | §3.2 |
| **verifier false-positive rate** | The rate at which a verifier accepts a wrong candidate answer as correct — irreducible by resampling alone, and the quantity that ultimately caps resampling-based accuracy. | §3.4 |
| **Vision Transformer (ViT)** | A transformer that treats an image as a sequence of fixed-size patches (like tokens) and processes them with the same self-attention architecture used for text. | §3.7 |
| **Vision-Amplified Semantic Entropy (VASE)** | A variant of semantic entropy that amplifies the contribution of vision-conditioned uncertainty, used here as a baseline that CEBaG beats by 8 AUC points on average. | §3.6 |
| **vision-conditioned entropy** | Computing semantic entropy twice -- once for the true image, once for a visually distorted version -- and using the contrast between them as the hallucination signal, so the score reflects reliance on actual visual evidence rather than just general overconfidence. | §3.6 |
| **Visual Dependency Probing (VDP)** | A method for identifying which decoder layers in an LVLM rely most heavily on visual (as opposed to purely textual) tokens when generating each output token. | §3.6 |
| **Visual premise verification** | Explicitly checking whether the visual facts (premises) a model's reasoning step relies on are actually true of the image, separately from checking whether the logical reasoning built on top of those premises is valid. | §3.2 |
| **visual prompt perturbation** | Creating semantically-equivalent but visually altered versions of the input image (e.g. crops, augmentations) to test whether the model's answer stays consistent -- inconsistency signals uncertainty/hallucination. | §3.6 |
| **visual token dominance** | In VLMs the image contributes hundreds to thousands of tokens, far more than the text, so image tokens dominate prefill cost and KV-cache size. | §3.1 |
| **visual token pruning** | Dropping redundant image tokens before prefill to cut cost; the direction our project started with and abandoned. | §3.1 |
| **VL-GenRM (vision-language generative reward model)** | A generative reward model (see GenRM) that additionally takes an image as input — i.e. a multimodal LLM-as-judge used to score/rank candidate multimodal responses. | §3.2 |
| **Vote / Filter-Vote** | Majority voting over N sampled LLM outputs, optionally with an LLM-based filter applied to the pool before the vote. | §3.4 |
| **Wasserstein-1 distance / optimal transport** | A way to measure the 'cost' of turning one probability distribution into another by moving probability mass; here it replaces exact-match agreement with a distance that counts moving mass between semantically similar answers as cheap. | §3.5 |
| **Weak supervision** | A statistical technique for estimating the accuracy/reliability of multiple noisy labeling sources (here, verifiers) and combining their outputs into a single better estimate, WITHOUT needing ground-truth labels to calibrate the weights. | §3.2 |
| **Weak supervision (for labels)** | Training labels that are noisy/heuristically derived (e.g. from an automated metric or rule) rather than from careful human annotation of every example. | §3.2 |
| **Weak-verifier ensembling** | Combining multiple imperfect verifiers (which individually make errors) into one stronger combined score, typically by weighting each verifier by an estimate of its reliability. | §3.2 |
| **weighted majority voting** | Majority vote where each sample's vote is weighted by a verifier score. | §3.1 |
| **White-box verifier** | A verifier that reads the generator model's internal states (e.g. hidden activations, attention, logits) rather than only its output text. Our probe is white-box; this paper's PRM is deliberately NOT (text-only), to stay model-agnostic. | §3.2 |
| **white-box vs black-box confidence estimation** | White-box methods need access to model internals (logits, hidden states, attention); black-box methods only need text outputs (e.g. via an API) -- prompting for a verbalized confidence, or checking agreement across repeated samples. | §3.6 |
| **whole-slide image (WSI)** | A gigapixel-scale digitized microscope slide of a tissue sample used in pathology, far larger than a typical photo, usually processed in tiles/patches. | §3.7 |
| **win rate** | Probability that a sample from one policy is preferred to a sample from another. | §3.1 |
| **yes-bias** | A VLM's tendency to answer 'yes' to yes/no questions regardless of whether the image actually supports a 'yes' answer, a common and well-documented VLM hallucination pattern. | §3.6 |
| **zero-shot accuracy** | Accuracy on a task/dataset the model was never explicitly trained or fine-tuned for -- here, classifying ImageNet images the model only ever saw via general image-text pretraining. | §3.7 |
| **zero-shot transfer** | Applying a model to a task/domain it was not specifically trained on (here: a general-purpose VLM applied directly to medical images) without any task-specific fine-tuning. | §3.7 |


---

# 2. Where we stand — the project, in the field's terms

Every number below names its source. Judge currency (Lingshu-32B judge) unless stated.

## 2.1 How we got here (one paragraph of history)

The project began as visual-token pruning, moved through image-difficulty routing and single-model
routing (all killed), and by June–July 2026 was a **7B→32B cascade** for medical VQA on the MedEvalKit
harness with a format-aware router: multiple-choice questions through a confidence-margin gate,
open-ended questions through best-of-N with a trained LoRA verifier. That work is the IEEE draft
`paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf` and CLAUDE.md §0; its headline against the
always-32B-direct baseline is a **tie** (`cascade_selector_rerun_2026-08-05.json`). On **16 August
2026** the open-ended arm was re-examined on cost: the LoRA verifier re-encoded the image once per
candidate to score a ~5-token string, and a probe on hidden states the generator had *already*
produced matched it at ~1/9000 of the compute (`CHEAP_VERIFIER_ON_7B_2026-08-16.md`). From 18 August
the arm grew from 3 to 8 open-ended benchmarks, the probe was retrained on all eight (24 August),
audited (12 September) and replicated on two other generators (13 September). **The probe verifier is
the live method.** The 32B is no longer a tier; it is the judge.

## 2.2 The method, in field terms

*A lightweight MLP probe (one hidden layer of 256, GELU) reads the frozen hidden states of
Lingshu-7B — the mean over each candidate's own generated tokens at layers 18/20/22 — and acts as a
pointwise best-of-8 verifier trained with binary cross-entropy; 24 probes (3 layers × 8 seeds) are
rank-averaged and the argmax candidate is returned.*

| component | what it is | source |
|---|---|---|
| generator | Lingshu-7B (8.29 B params; a Qwen2.5-VL-7B medical finetune), frozen, unmodified | `meetings/shipped_method_2026-09-12.html` |
| sampling | N = 8, T = 0.7, image encoded once (vLLM `n=8` shares the prefill); ~3.9 distinct candidates | same; `bestofn_vllm_2026-09-16.json` |
| representation | `h_span`: mean hidden state over the candidate's own tokens; 3584-d; layers 18/20/22; captured during generation (capturing vs re-computing changes 15 of 2,345 picks) | `opentext_progress_2026-09-14.html` |
| probe | Linear(3584→256) → GELU → Linear(256→1); 918,017 params; BCE on binary correctness; AdamW; per-layer frozen standardisers | same; `head_final_stack.py` |
| ensemble | 3 layers × 8 seeds = 24 probes; rank-average within the candidate set; argmax | same |
| training data | 112,770 rows = by-image *train halves* of all 8 benchmarks + the 4 July training domains; 59 leaking rows dropped | `head_final_stack_PVFIXED_2026-09-13.json` |
| artifact | `ckpts/train/genframe_head_pooled_ens_v2/` (24 probes + 3 standardisers, 88.3 MB); reload-verified +0.0737 vs +0.0736 at fit | `shipped_method_2026-09-12.html` |
| labels | correctness of each candidate and of greedy decided by a Lingshu-32B judge; exact match also stored | `OPENTEXT_FULL_RUNDOWN_2026-09-04.md` |

## 2.3 The data — eight open-ended benchmarks

`meetings/opentext_progress_2026-09-14.html` (counts from `ckpts/openvqa/cheap_lingshu7b/`):

| benchmark | imaging domain | questions | held-out | train half | note |
|---|---|---:|---:|---:|---|
| PathVQA (open) | pathology | 3,357 | 1,623 | 1,734 | original; was reported on 1,500 until 2026-09-12 (`AUDIT_2026-09-12.md` §3) |
| SLAKE (open) | radiology (CT/X-ray/MRI) | 645 | 330 | 315 | original |
| VQA-RAD (open) | radiology, clinician-written | 200 | 97 | 103 | original; smallest |
| RadImageNet-VQA | radiology (CT/MRI/US, 11 anatomies) | 2,000 | 1,004 | 996 | added 18 Aug |
| Kvasir-VQA-x1 | GI endoscopy | 10,121 | 5,152 | 4,969 | added 20 Aug; 1,052 contaminated images + 136 near-duplicates excluded |
| OmniMedVQA | 9 modalities, 42 sources | 8,883 | 4,461 | 4,422 | added 22 Aug; **converted from multiple choice** (options removed); RadImageNet source dropped |
| VQA-Med 2019 (C4 Abnormality) | radiology | 3,663 | 1,807 | 1,856 | added 20 Aug; the only genuinely open category of four |
| GEMeX | chest X-ray | 8,000 | 3,978 | 4,022 | added 22 Aug; most OOD; 4,315 distinct golds |
| **total** | | **36,869** | 18,452 | 18,417 | |

The three originals also have official training splits (PathVQA 9,903 · SLAKE 2,976 · VQA-RAD 853).
Only the three originals are MedEvalKit loaders; the other five are our own harness
(`OPENTEXT_CELL_SURVEY_2026-08-18.md` §0) — say so in any paper.

## 2.4 Results

### 2.4.1 The headline (shipped recipe, held-out halves)

`head_final_stack_PVFIXED_2026-09-13.json` (5 seeds, 112,770 rows; restated after the PathVQA
backfill), independently confirmed by the shipped artifact's own
`ckpts/train/genframe_head_pooled_ens_v2/recipe.json`, which records
`macro_verifier_minus_greedy = 0.07360735025242765` and `benchmarks_beaten = 6/8`:

| arm | macro Δ vs greedy | beats greedy |
|---|---:|---|
| previous recipe (4 domains, one layer, Bradley–Terry) | +0.0182 | 6/8 |
| pooled training, single layer | +0.0729 | 6/8 |
| **pooled + 3-layer ensemble — SHIPPED** | **+0.0736** | **6/8** |
| pooled + ensemble + self-consistency feature | +0.0720 | 7/8 |

Per benchmark (`meetings/shipped_method_2026-09-12.html`, from the same artifact):

| benchmark | held-out q | greedy | + probe | Δ |
|---|---:|---:|---:|---:|
| PathVQA | 1,623 | 0.3050 | 0.3580 | +0.0530 |
| SLAKE | 330 | 0.7121 | 0.7667 | +0.0545 |
| VQA-RAD | 97 | 0.5052 | 0.4639 | −0.0412 |
| RadImageNet | 1,004 | 0.3337 | 0.4592 | +0.1255 |
| Kvasir-x1 | 5,152 | 0.2811 | 0.4022 | +0.1211 |
| OmniMedVQA | 4,461 | 0.5216 | 0.6783 | +0.1567 |
| VQA-Med C4 | 1,807 | 0.0913 | 0.0897 | −0.0017 |
| GEMeX | 3,978 | 0.3997 | 0.5206 | +0.1209 |
| **macro** | 18,452 | | | **+0.0736** |

**Where the gain came from.** Training data did the work: pooled training +0.0547 of the +0.0554;
the layer ensemble +0.0007 (a hedge — it collapses the single-layer spread from 0.013 to 0.003); the
self-consistency input feature was worth +0.0098 from the four-domain base and −0.0016 once pooled,
so it is not shipped (`shipped_method_2026-09-12.html`; `head_final_stack.py` docstring).

### 2.4.2 Against the baselines a reviewer will ask for

Frozen four-domain probe on the *full* benchmarks, `free_signal_bakeoff_2026-08-21.json` (as printed
in `OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §1; PathVQA there is the 1,500-question prefix):

| benchmark | n | greedy | answer prior | self-consistency | verifier | oracle@8 |
|---|---:|---:|---:|---:|---:|---:|
| PathVQA | 1,500 | 0.3427 | 0.3333 | 0.3260 | **0.3900** | 0.5167 |
| SLAKE | 645 | 0.7302 | 0.7287 | 0.7395 | **0.7690** | 0.8791 |
| VQA-RAD | 200 | 0.4900 | 0.4350 | 0.4650 | 0.4650 | 0.6300 |
| RadImageNet | 2,000 | 0.3210 | 0.2825 | 0.3245 | 0.3295 | 0.5120 |
| Kvasir-x1 | 10,121 | 0.2849 | 0.2815 | 0.2699 | **0.3629** | 0.4696 |
| OmniMedVQA | 8,883 | **0.5164** | 0.4746 | 0.5162 | 0.4971 | 0.7007 |
| VQA-Med C4 | 3,663 | **0.0947** | 0.0459 | 0.0863 | 0.0688 | 0.2102 |
| GEMeX | 8,000 | 0.3974 | 0.3549 | 0.3794 | **0.4121** | 0.5864 |

The verifier beats the answer-prior baseline on all eight. On GEMeX — the benchmark with the least
answer-vocabulary to memorise — the answer prior *loses* to greedy by 0.0425 while the verifier wins
+0.0148 [+0.0057, +0.0238] over greedy and +0.0573 [+0.0483, +0.0663] over the prior
(`TRANSFER_WALL_2026-08-21.md` §10). That is the cleanest evidence the probe does verification rather
than memorisation.

### 2.4.3 July's verifier vs today's, same 2,345 questions

`free_head_2026-08-16.json` (accuracy) + `bestofn_latency_energy_2026-08-03.json` (cost, HF path):

| selector | accuracy | selection efficiency | FLOP-eq | latency | energy |
|---|---:|---:|---:|---:|---:|
| always-7B greedy | 0.4495 | — | 1.00× | 1.00× | 1.00× |
| self-consistency | 0.4469 | 0.7139 | 8.00× | 1.99× | 2.95× |
| LoRA verifier (July) | 0.4853 | 0.7752 | 15.18× | 3.68× | 5.77× |
| **probe verifier (today)** | **0.5015** | **0.8011** | 8.00× | 1.99× | 2.95× |
| both, ranks averaged | 0.5075 | 0.8106 | 15.18× | 3.68× | 5.77× |
| oracle@8 | 0.6260 | 1.0000 | — | — | — |

### 2.4.4 Replication on other generators

- **Qwen2.5-VL-7B-Instruct** (the base model Lingshu was finetuned from), identical pipeline:
  pooled + ensemble **+0.0820**, beats its own greedy on **8/8**; the pooled lever is +0.0594 (Lingshu
  +0.0554) (`head_final_stack_qwen_2026-09-13.json`; `TRANSFER_WALL_2026-08-21.md` §12).
- **MedGemma-4b-it** (Gemma 3 + SigLIP — a different LM family), four benchmarks, protocol matched
  across generators: MedGemma **+0.0248**, Lingshu +0.0375, Qwen +0.0178; the pre-registered
  depth-matched layers [22,24,27] beat the absolute-matched [18,20,22] (+0.0248 vs +0.0185)
  (`TRANSFER_WALL_2026-08-21.md` §14; `head_final_stack_medgemma_2026-09-13.json`,
  `head_final_stack_{lingshu,qwen}_matched_2026-09-13.json`). The claim: *not* a Lingshu, medical-
  finetuning or Qwen artifact. Caveat: a 4B generator, and the artifact `VERDICT`
  strings quote each run's *best* arm, which is not always the pre-registered one.
- 🆕 **MedGemma on all eight benchmarks finished 2026-09-16**
  (`head_final_stack_medgemma_ALL8_2026-09-16.json`, 115,692 pooled rows, 5 seeds): macro
  **+0.0481 on 8/8** for the single-layer probe, +0.0447 (7/8) for the 3-layer ensemble. So the third
  generator now replicates on the **full** benchmark set, not just four, which retires the "four
  benchmarks only" caveat above. Per benchmark (greedy → single-layer probe): OmniMedVQA
  0.5109 → 0.6382 (+0.1273), GEMeX 0.4253 → 0.5103 (+0.0850), RadImageNet 0.2978 → 0.3556 (+0.0578),
  Kvasir-x1 0.1623 → 0.2149 (+0.0526), PathVQA 0.0653 → 0.0893 (+0.0240), VQA-RAD 0.4948 → 0.5155
  (+0.0206), SLAKE 0.6273 → 0.6424 (+0.0152), VQA-Med 0.0094 → 0.0116 (+0.0022).
  Two things to say honestly about it: MedGemma-4b's **absolute** accuracy is far lower than
  Lingshu's on PathVQA (0.0653 vs 0.3050) and VQA-Med (0.0094 vs 0.0913), so part of the 8/8 is
  headroom rather than skill; and here the **single-layer** arm beats the shipped 3-layer ensemble by
  +0.0034, the reverse of Lingshu — a difference close to the ~0.003 reproducibility floor (§2.5.4),
  so it should be read as "the ensemble is not load-bearing on this generator", not as a ranking.

### 2.4.5 Mechanism and limits

- **Decomposition** (`decomposition_2026-08-24.json`, frozen four-domain probe, T = 0.7, all 8):
  selection skill positive on **7/8**; mean skill +0.0387 against mean sampling penalty +0.0257;
  mean verifier − greedy +0.0130. Where the method loses, the sampled candidate set is worse than
  greedy, not the ranker.
- **The verifier is per-benchmark.** LOBO: four-domain +0.0279 → leave-one-benchmark-out +0.0223 →
  pooled +0.0707; **breadth alone −0.0056, own training half +0.0485**
  (`head_lobo_pooled_2026-08-25.json`). The +0.0736 holds only where a labelled split exists.
- **Onboarding a benchmark.** From a verifier trained on the other seven: zero-shot macro +0.0251,
  6/8 already cross greedy at k = 0, median crossover k = 0; the first ~100 labelled questions carry
  most of the gain (`head_price_from_lobo_2026-08-30.json`).
- 🆕 **More samples — now settled with intervals on all eight (2026-09-16).**
  `coverage_sc16_ci_ALL_2026-09-16.json` (10,000 bootstrap resamples, **clustered by image within
  benchmark**, held-out halves, pooled verifier) supersedes the two-benchmark read that every earlier
  document hedged on. Doubling the budget 8 → 16 is worth **+0.0167 macro: 4 WIN, 0 LOSS, 4 TIE.**

  | benchmark | n | images | greedy | @8 | @16 | Δ(16−8) [95 % CI] | |
  |---|---:|---:|---:|---:|---:|---|---|
  | RadImageNet | 1,004 | 502 | 0.3337 | 0.4552 | 0.4880 | +0.0329 [+0.0149, +0.0518] | **WIN** |
  | Kvasir-x1 | 5,152 | 1,447 | 0.2811 | 0.4066 | 0.4375 | +0.0309 [+0.0206, +0.0414] | **WIN** |
  | OmniMedVQA | 4,461 | 4,151 | 0.5216 | 0.6792 | 0.7254 | +0.0462 [+0.0360, +0.0562] | **WIN** |
  | GEMeX | 3,978 | 1,742 | 0.3997 | 0.5194 | 0.5553 | +0.0359 [+0.0254, +0.0468] | **WIN** |
  | PathVQA | 1,623 | 420 | 0.3050 | 0.3555 | 0.3567 | +0.0012 [−0.0124, +0.0150] | tie |
  | SLAKE | 330 | 49 | 0.7121 | 0.7788 | 0.7606 | −0.0182 [−0.0428, +0.0064] | tie |
  | VQA-RAD | 97 | 58 | 0.5052 | 0.4639 | 0.4639 | +0.0000 [−0.0638, +0.0632] | tie |
  | VQA-Med | 1,807 | 1,807 | 0.0913 | 0.0880 | 0.0924 | +0.0044 [−0.0066, +0.0155] | tie |

  **This is a genuine Pareto point and it is now defensible**, where before it was one macro number
  with CIs on only the two least sampling-responsive benchmarks. It costs a doubling of generation,
  unlike everything else in §2.4.1. (The earlier read — macro +0.0147, 6/8 positive, 66.3 % of the
  oracle gain converted N = 2 → 16 — is `coverage_scaling_ALL_2026-09-01.json`, restated in
  `AUDIT_2026-09-12.md` §6.)
- **Temperature.** Pooled verifier: T = 0.2 +0.0401, 0.4 +0.0573, **0.7 +0.0816**, 1.0 +0.0759 —
  a stronger verifier prefers a more diverse candidate set (`head_temp_ensemble_2026-08-30.json`).
- **Two failures, diagnosed.** VQA-Med C4: coverage (oracle@8 0.2102 vs greedy 0.0913 — most sets have
  no correct answer). VQA-RAD: 365 training rows, and sampling at T = 0.7 costs it 0.0494 despite
  +0.0244 of genuine skill (`shipped_method_2026-09-12.html`).
- **What does not predict where it helps.** Neither OOD distance (GEMeX is the *most* OOD and the probe
  wins there), answer kind, nor any of six regime detectors orders the benchmarks
  (`regime_detector_2026-08-21.json`; `answer_kind_2026-08-22.json`). Twelve documented dead ends with
  measured bounds: `OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §4.
- **Visual features add nothing.** Appending the image representation: +0.0002; a question-token
  control does +0.0018 (`head_visual_features_2026-09-01.json`).

### 2.4.6 Cost

`bestofn_vllm_2026-09-16.json` (vLLM, one request in flight, cap320, A100 80GB, 20 timed calls):
greedy 174.0 ms / 34.3 J; best-of-8 476.6 ms / 123.2 J → **2.74× latency, 3.59× energy**; forward
tokens per question 328.5 → 371.2 = **1.13 FLOP-eq** with the shared prefill (8.0 "as charged"). On
the HF `num_return_sequences=8` path: 1.99× latency, 2.95× energy (`bestofn_latency_energy_2026-08-03.json`).
The probe itself: 24 × ~1.8 MFLOP on vectors already computed. Nothing has been run end to end as
one deployed system (CLAUDE.md §0 standing caveat).

## 2.5 The honest holes — what the next report must not hide

1. **Currency.** The September headline is **judge currency only**. In August the head-only arm was
   judge +0.0452 [+0.0294, +0.0606] but **exact match −0.0068 [−0.0226, +0.0090], negative on all
   three original benchmarks** (`CHEAP_VERIFIER_ON_7B_2026-08-16.md` §6), and a newly trained
   verifier gets a free +0.006–0.009 under the same-family judge (`coadapt_verifier_T04_2026-08-14.json`).
   The pooled, eight-benchmark probe has not been reported in EM. Until it is, the claim is
   "+0.0736 under a Lingshu-32B judge".
2. **Per-benchmark, not general.** Breadth buys nothing (LOBO); the method needs ~100 labelled
   questions per new benchmark. Say this before the headline, not after.
3. **The mechanism is not novel.** Hidden-state probes as best-of-N verifiers exist under several
   names (§3.3, §4); "verification is free" is in print. What is ours is the vision-language /
   medical transfer and the breadth of the characterisation.
4. **Reproducibility floor.** Thread count moves the macro by 0.0033; tie-breaking by cache row order
   is worth up to +0.0061; cross-run arm comparisons below ~0.003 are noise
   (`repro_threading_2026-09-13.json`, `tiebreak_2026-09-13.json`).
5. **Only three benchmarks are harness-faithful** (MedEvalKit loaders); five are our own builds, one
   (OmniMedVQA) converted from multiple choice with 51.7 % disease-diagnosis over 224 golds.
6. **One generator family dominates.** Lingshu and Qwen2.5-VL share a language model; MedGemma is a
   single 4B replication on four benchmarks.
7. **PathVQA truncation** invalidated every number printed before 2026-09-12 on that benchmark; the
   backfill moved the headline +0.0802 → +0.0736 (`AUDIT_2026-09-12.md` §3, `pathvqa_truncation_2026-09-13.json`).
8. **No control task has ever been run.** The probing literature's standard hygiene check
   (Hewitt & Liang 2019, arXiv:1909.03368, §3.3) is to train the same probe on a *randomised* version
   of the labels: a probe with enough capacity can fit anything, so the reported metric must be paired
   with **selectivity** — real-task performance minus control-task performance. We have a *permutation
   null* on the features (frozen head fed row-permuted features → 13.35 σ above null,
   `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §8), which is related but is not the same test: it randomises
   the inputs, not the labels. This is a cheap, CPU-only gap a probing-literate reviewer will notice.
9. **Question-identity leakage in pooled AUROC — flagged internally, never fixed.**
   `OPENTEXT_CORRECTIONS_2026-08-19.md` §7 records that `head_sweep.auroc` is "pooled across
   questions, the wrong AUROC for a selection task". CASE (arXiv:2608.17124) makes the same point from
   the other side and names it: a probe scored across questions can be rewarded for recognising *which
   question* it is looking at, so CASE proposes **question-grouped ("decodability") evaluation** to
   remove that leakage, and reports that decodability predicts whether hidden-state selection beats
   majority voting at r = 0.75. Our *endpoint* metrics are safe — selection is an argmax inside one
   question, so accuracy and selection efficiency are inherently within-question — but any AUROC we
   quote as diagnostic evidence is not. Either report within-question AUROC, or drop the metric.
10. **The generator arrives already RL-shaped, and we have never said so.** Lingshu's stage 4 is
    **GRPO reinforcement learning with verifiable rewards** (the Lingshu report, §3.7). So
    Lingshu-7B has already been optimised toward producing correct answers before our best-of-N ever
    runs. Two consequences worth pre-empting: (i) our candidate sets are drawn from a
    distribution that RL has already sharpened, which plausibly *reduces* the headroom best-of-N can
    recover and makes our gain a conservative estimate rather than an inflated one; (ii) a reviewer
    can reasonably ask whether the probe adds signal beyond what RL already supplied. The cleanest
    available answer is the Qwen2.5-VL replication — Qwen is the pre-medical-finetune base, and the
    probe works there too (+0.0820, 8/8) — but this should be stated deliberately rather than left
    for someone else to raise.

## 2.6 What is in flight (as of 2026-09-16, 18:00)

**Landed today**, and folded into §2.4 above: the MedGemma all-eight-benchmark run
(`head_final_stack_medgemma_ALL8_2026-09-16.json`) and the image-clustered N=16 intervals on all
eight (`coverage_sc16_ci_ALL_2026-09-16.json`). Both are improvements on what the 14 September deck
says, so **the deck is now behind the artifacts** on those two points.

**Still running:** 32-sample pool extraction for `vqamed_open` (`lingshu7b_sc32T07`, layers 18/20/22,
on GPU 1) — the coverage-limited benchmark, which is the right place to test whether more samples can
rescue a coverage failure; a re-run of `head_price_from_lobo.py`; and the automated campaign
(`runners/auto_{cpu,gpu0,gpu1}_wave*.json`) via `src/reporting/supervisor.py`. Full 4×4 train/deploy
temperature matching remains queued (`OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §8).


---

# 3. The literature, by category

Nine categories. Within each, papers are ordered by reading priority then by date. ★ marks a core paper whose PDF is in `papers/`. Every number in a card is copied from the paper with a provenance tag; `not extracted` means the source did not state it.

## 3.0 Index

| § | category | papers | core |
|---|---|---:|---|
| 3.1 | Test-time compute scaling and best-of-N: the foundations | 21 | 4 |
| 3.2 | Verifiers and Reward Models: Outcome vs Process, Discriminative vs Generative, and Verification in Medical VQA | 24 | 4 |
| 3.3 | Probing frozen hidden states: from probing classifiers to hidden-state verifiers | 21 | 4 |
| 3.4 | The Walls: Coverage vs. Selection, Imperfect Verifiers, and Bounded Best-of-N Gains | 16 | 4 |
| 3.5 | Training-free selection: self-consistency, majority vote, minimum Bayes risk, consensus, and logit-based scores | 20 | 4 |
| 3.6 | Uncertainty, Calibration, and Hallucination Detection in LLMs and VLMs -- with the Medical Evidence | 32 | 4 |
| 3.7 | Medical Vision-Language Models: The Generators We Use and the Ones We Compare Against | 21 | 4 |
| 3.8 | Medical VQA benchmarks, datasets, and the evaluation protocol | 23 | 4 |
| 3.9 | The older lineage: n-best rescoring, discriminative reranking, and confidence models in other fields | 16 | 4 |

## 3.1 Test-time compute scaling and best-of-N: the foundations

Test-time (inference-time) scaling means spending extra computation when a question is answered — more samples, longer reasoning, or search — instead of making the model bigger or training it longer. The canonical parallel form is best-of-N: draw N candidate answers from the same model and let a selector (majority vote, or a learned verifier) return one; its ceiling is 'coverage' (does any candidate get it right?) and its bottleneck is selection (can we find that candidate?). The compute-optimal literature (Snell; Wu; Liu; Roberts) shows that a small model plus sampling can beat a much larger model at matched FLOPs, but only on questions the small model can sometimes solve, and only if the selector is good. For vision-language models the picture is newer and more mixed (Sammani; Ahmadpour; Baxevanakis; the 2026 JMIR medical study): small capable models gain most, perception-heavy tasks gain little from longer reasoning, and verifier-based selection is the most reliable win on open models. Our method is best-of-8 with a 918k-parameter hidden-state probe as the verifier on open-ended medical VQA, so every claim we make — the +0.07 headline, the 78-81% selection efficiency, the 37% coverage wall, and especially 'verification is free' — has to be stated in this field's terms and its cost conventions: FLOPs per query with prefill and decode counted separately (2N FLOPs per token), generation cost separated from evaluation cost, and energy reported as a measurement only if it was measured.

#### ★ On Test-Time Scaling for Vision-Language Models

*Fawaz Sammani et al. · 2026 · ECCV 2026 · [arXiv:2606.28864](https://arxiv.org/abs/2606.28864) · read priority 1 · **PDF in `papers/`***

**In one line.** First broad study of LLM-style test-time scaling on vision-language models: small, good models gain the most (up to ~30 points via self-consistency), perception benchmarks often get worse, and the image stops mattering after roughly 200 generated tokens.

- **Models.** Qwen-2.5-VL {7B, 32B, 72B}, Qwen-3-VL {2B, 4B, 8B, 32B}, InternVL-3.5 {2B, 4B, 8B, 38B}, Molmo2 {4B, 8B} [§4].
- **Method.** Nine training-free test-time scaling methods [§3]: (1) Chain-of-Thought, (2) Structured CoT, (3) Plan-and-Solve, (4) Self-Consistency ('samples multiple independent CoTs ... producing b different reasoning chains ... and then selects the most frequent answer'), (5) Self-Aggregation (concatenate all b chains and prompt the LVLM to aggregate), (6) Self-Refinement (k rounds), (7) Describe-Answer (caption first), (8) Compositional CoT (scene graph), (9) Prompt Repetition. NOTE: no learned verifier / reward-model best-of-N is among the nine.
- **Datasets.** MMStar, RealWorldQA, HallusionBench, WeMath, LogicVista, A-OKVQA [§4] — none medical.
- **Experiments.** Accuracy of each method x each model on the six benchmarks; runtime in seconds on an H200; attention-to-image analysis over generation steps; an intervention that drops image tokens (and their KV cache) at a chosen generation step [Fig. 5, §S7]; reasoning-chain quality/information-sufficiency analysis.
- **Results.** 'self-consistency boosts the performance of Qwen3-VL-2B from 35% to 64% on WeMath (approximately a 30% increase)' [§4/Table 1]; 'Qwen3-VL-4B with simple CoT ... surpass[es] the baseline ... of Qwen3-VL-32B' [§4]; 'test-time scaling methods often degrade performance ... on primarily perception-focused benchmarks' [§4]; 'image attention peaks early and then drops rapidly, while attention to previously generated tokens increases'; 'After approximately 200 generation steps, removing the image tokens has almost no effect' [Fig. 5]; runtime: CoT on Qwen3-VL-4B '5.9' s vs baseline '0.1' s; self-consistency on larger models '88.9' s [Table 1]; KV-cache drop saves 'approximately 125 GFLOPs ... for WeMath, HallusionBench and MMStar, and around 325 GFLOPs ... for LogicVista' [§S7]; total experiment cost 'approximately $6,000'.
- **Conclusions.** TTS transfers to LVLMs but unevenly: it pays most for small capable models and for reasoning-style benchmarks; on perception-style tasks it can hurt; 'LVLMs lose focus when given more compute than necessary'; visual information is consumed early in the chain.

> **Why it matters to us.** The paper Leo must position against. (a) Its selection is majority vote only — it never tests a trained verifier — so our probe is exactly the missing method in their table; our headline that a learned verifier beats self-consistency on open-ended answers is a claim their setup could not make. (b) Their 'perception benchmarks degrade' result is the same as our repo's 'reasoning hurts perception' finding and should be cited as external agreement. (c) Their evidence that the image is folded into the chain within ~200 tokens supports reading a hidden-state summary of the candidate's own generated tokens: by then the representation already carries the visual evidence. (d) Their cost unit is wall-clock seconds on an H200 with FLOPs only for the KV-drop ablation — when we compare, we must say which currency we use. (e) None of their benchmarks are medical or open-ended free text: the domain/format transfer is our contribution.

<small>Read from: html.</small>

#### Model and Task-Aware Test-Time Scaling Strategies for Large Language and Vision-Language Models in Medicine: Evaluation Study

*Gyutaek Oh et al. · 2026 · Journal of Medical Internet Research 28:e90693 (2026), PMID 42490549 · [doi:10.2196/90693](https://pubmed.ncbi.nlm.nih.gov/42490549/) · read priority 1*

**In one line.** The one 2026 medical-domain evaluation of test-time scaling found: parallel scaling beats sequential on easier medical tasks, longer reasoning is not universally beneficial, and VLMs show 'a structural bottleneck in integrating visual clues' with 'limited benefit from token expansion'.

- **Models.** 'a diverse set of general and medical-specific LLMs and VLMs' [abstract]; names and sizes not extracted (full text not fetched — the journal page returned no content; PubMed abstract only).
- **Method.** 'three scaling conditions: increasing token budgets, iterative sequential scaling, and parallel scaling' [abstract]; robustness tested 'by embedding misleading hints with varying tones and levels of simulated clinical expertise into prompts' [abstract].
- **Datasets.** 'five textual medical benchmarks comprising over 5500 questions and two multimodal benchmarks comprising 7000 samples' [abstract]; benchmark names not extracted.
- **Experiments.** Accuracy vs token budget; sequential vs parallel scaling per task difficulty; medical-finetuned vs general models on clinical QA vs calculation tasks; adversarial-hint robustness.
- **Results.** [abstract] 'For nonreasoning LLMs, accuracy saturated quickly, with token usage often remaining under 500 tokens regardless of budget increases.' 'Reasoning models demonstrated significant performance gains on complex tasks as token budgets increased.' 'current VLMs showed a structural bottleneck in integrating visual clues and experienced limited benefit from token expansion.' 'medically fine-tuned LLMs excelled in clinical question answering but exhibited degraded scaling efficiency on calculation tasks compared to general-domain models.' 'parallel scaling outperformed sequential scaling on easier tasks. Conversely, extended sequential scaling or increased budgets proved essential for complex problem-solving.' No per-model accuracies extracted.
- **Conclusions.** 'Test-time scaling rules from general domains do not perfectly translate to medical AI. Longer reasoning is not universally beneficial. Concise reasoning with parallel scaling is optimal for simpler tasks.' [abstract]

> **Why it matters to us.** The paper to cite for 'TTS in medicine' context. It agrees with our two domain findings — medical VLM tasks are perception-bound so longer chains buy little, and parallel sampling (our regime) is the right axis for such tasks — and its parallel-scaling arm is, as far as the abstract shows, majority-vote self-consistency, so our learned-verifier selection is the next step it does not take. Because only the abstract was read, do not quote any number beyond those above; the article is open access and Leo should read the full text (models, the two multimodal benchmarks, per-strategy gains) before comparing.

<small>Read from: abstract-only.</small>

#### ★ Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

*Charlie Snell et al. · 2024 · arXiv preprint · [arXiv:2408.03314](https://arxiv.org/abs/2408.03314) · read priority 1 · **PDF in `papers/`***

**In one line.** Per-prompt, difficulty-aware allocation of test-time compute (search against a verifier, or sequential revision) beats a fixed best-of-N budget by up to 4x and, FLOPs-matched, lets a small model beat one ~14x larger on easy/medium prompts.

- **Models.** PaLM 2-S* (Codey) as the base LM; a process reward model (PRM) 'trained with soft values obtained from monte-carlo rollouts, using a binary cross entropy loss function'; an outcome reward model (ORM) as comparison baseline [html, method sections].
- **Method.** Two mechanisms for spending test-time compute: (1) search against a dense PRM — best-of-N weighted, beam search, lookahead search; (2) sequential revisions where the model rewrites its own answer conditioned on previous attempts. 'Compute-optimal' = choose, for each prompt, the allocation (method + hyperparameters) that maximises accuracy within a budget N. Difficulty is estimated by binning 'the model's pass@1 rate — estimated from 2048 samples — on each question into five quantiles' [html, compute-optimal section].
- **Datasets.** MATH (high-school competition problems), '12k train, 500 test questions' split of Lightman et al. [html, setup].
- **Experiments.** Accuracy vs generation budget for best-of-N weighted / beam / lookahead with the PRM; revisions vs parallel sampling; a FLOPs-matched comparison between test-time compute on the small model and a ~14x larger pretrained model; a pretraining-vs-inference exchange-rate analysis parameterised by R = D_inference / D_pretrain [html, §7, Fig. 9].
- **Results.** 'compute-optimal scaling can nearly outperform best-of-N using up to 4x less test-time compute (e.g. 16 verses 64 generations)' [Fig. 4, PRM search]; same 4x for revisions '(e.g. 64 samples verses 256)' [Fig. 8]; 'in a FLOPs-matched evaluation ... test-time compute can be used to outperform a 14x larger model' on 'problems where a smaller base model attains somewhat non-trivial success rates' [abstract]; when R << 1 'test-time compute is often preferable to additional pretraining', when R >> 1 'pretraining is preferable'; on hard questions 'pretraining is a more effective way to improve performance' [§7, Fig. 9].
- **Conclusions.** How much test-time compute helps depends strongly on prompt difficulty, so a per-prompt adaptive policy dominates uniform best-of-N; test-time compute substitutes for parameters only on questions the small model can already sometimes solve; it does not rescue the hardest questions.

> **Why it matters to us.** The framing paper for the whole project. Their cost convention — pretraining X = 6·N·D_pretrain, inference Y = 2·N·D_inference FLOPs, N = parameters [§7] — is the convention under which our 918k-parameter probe run on 8 candidates is negligible next to the 7B generator (this is what 'verification is free' means and it should be stated in exactly these units). Their verifier is a PRM on math steps; ours is an outcome-level scorer on hidden states, so their best-of-N-weighted results are the closest baseline family. Their finding that the hardest bin gains nothing from more samples is the same phenomenon as our 37% coverage wall. Their per-prompt adaptive allocation is the ancestor of any adaptive-N variant we build.

<small>Read from: html.</small>

#### ★ Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

*Bradley Brown et al. · 2024 · arXiv preprint · [arXiv:2407.21787](https://arxiv.org/abs/2407.21787) · read priority 1 · **PDF in `papers/`***

**In one line.** Repeated sampling raises 'coverage' (any-sample-correct) log-linearly over four orders of magnitude, but without an automatic verifier the common selectors (majority vote, reward model) plateau far below coverage — the origin of the 'selection wall'.

- **Models.** Llama-3 (8B, 8B-Instruct, 70B-Instruct), Gemma (2B, 7B), Pythia (70M-12B), DeepSeek-Coder-V2-Instruct, GPT-4o, Claude 3.5 Sonnet [html].
- **Method.** Draw k independent samples per problem at fixed temperature; measure coverage = 'the fraction of problems that are solved by any generated sample' [§2]; fit coverage vs k with an exponentiated power law 'c ≈ exp(a·k^b)' [§3.1]; on tasks without an automatic checker, compare majority voting, reward-model best-of-N and reward-model-weighted majority vote against coverage.
- **Datasets.** GSM8K, MATH, MiniF2F-MATH, CodeContests, SWE-bench Lite [html].
- **Experiments.** Coverage vs number of samples (1 to 10,000) across models and tasks; cost comparison of many samples from a cheap model vs one sample from a frontier model on SWE-bench Lite [Table 1]; selector saturation curves on GSM8K/MATH [§4.1].
- **Results.** SWE-bench Lite: DeepSeek-Coder-V2-Instruct 'increases from 15.9% with one sample to 56% with 250 samples, outperforming the single-sample state-of-the-art of 43%' [abstract]; 'sampling five times from the weaker and cheaper DeepSeek model solves more issues than single samples from Claude or GPT while also being over 3x cheaper' [§2.3] — Table 1: DeepSeek-Coder-V2-Instruct 5 attempts $10.80, 29.62% solved; GPT-4o 1 attempt $39, 24.00%; Claude 3.5 Sonnet 1 attempt $51, 26.70% [Table 1]; 'common methods for picking from a sample collection (majority voting and reward models) plateau beyond several hundred samples and fail to fully scale with the sample budget' [abstract]. The HTML extraction also reported a GSM8K/Llama-3-8B-Instruct coverage of 98.44% at k=10,000 and a majority-vote row moving only from 40.50% to 41.41% between 100 and 10,000 samples [§4.1 as extracted] — which model/dataset each of those rows belongs to could not be confirmed from the extraction; check Fig. 6 in the PDF before quoting them.
- **Conclusions.** Sampling is a genuine scaling axis for inference; when answers can be checked automatically the gains are real; when they cannot, the bottleneck moves from generating a correct answer to identifying it.

> **Why it matters to us.** Gives us the two vocabulary items we already use: 'coverage' (our 'coverage wall': 37% of questions have no correct candidate in 8) and the coverage-vs-selection gap (our 'selection efficiency', ~78-81% of oracle@8). Their exponentiated power law is the right functional form to fit our coverage-vs-N curve instead of eyeballing. Their reward-model-best-of-N plateau is the wall we are pushing on; note their reward models were external text RMs, not probes on the generator's own hidden states. Their cost argument (a cheap model sampled many times beats an expensive model once) is our 7B-vs-32B thesis, and their unit is dollars — we use FLOPs, so say so.

<small>Read from: html.</small>

#### ★ Training Verifiers to Solve Math Word Problems

*Karl Cobbe et al. · 2021 · arXiv preprint · [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) · read priority 1 · **PDF in `papers/`***

**In one line.** Introduces GSM8K and the learned-verifier best-of-N recipe: sample 100 solutions, train a model to predict correctness, return the top-scored one — a 6B model with a verifier slightly beats a fine-tuned 175B model.

- **Models.** GPT-3 family, primarily 6B and 175B parameters, used both as generator and (same size) as verifier [html].
- **Method.** 'Sample 100 completions from the generator for each training problem', label each 'correct or incorrect based solely on whether they reach the correct final answer' [§4.2]; train the verifier jointly with the language-modelling objective; token-level variant makes 'a scalar prediction after each token in the solution' [§4.3]; at test time 'Sample 100 completions to each test problem, rank them with the verifier, and then return the one with the highest verifier score' [§4.2].
- **Datasets.** GSM8K: '8.5K high quality grade school math problems', '7.5K training problems and 1K test problems' [§2].
- **Experiments.** Fine-tuning baseline vs verification at 6B and 175B as a function of training-set size; token-level vs solution-level verifier; effect of the number of sampled completions.
- **Results.** 6B verification 'slightly outperforms a finetuned 175B model, thereby offering a boost approximately equivalent to a 30x model size increase' [§6, Fig. 5]; 'Token-level verifiers are less prone to overfitting than solution-level verifiers' [§4.3]; verification is 'not beneficial at low dataset sizes' but gives a 'strong boost' once the dataset is 'sufficiently large', and verifiers scale 'far more favorably with additional data than baseline methods' [§4.2].
- **Conclusions.** Judging candidate solutions is easier than generating them, so a learned verifier converts sampling budget into accuracy more cheaply than scaling the generator — provided there is enough labelled data to train it.

> **Why it matters to us.** Prior art for the mechanism: our MLP probe is a solution-level, outcome-supervised verifier used in exactly this sample-then-rank loop. Differences to state explicitly: (i) they train a full-size LM as verifier on text; we train a 918k-parameter head on the generator's frozen hidden states, tapped during generation; (ii) their token-level > solution-level finding is relevant because our mean-over-generated-tokens pooling is a crude token-level aggregate — worth citing when justifying the pooling; (iii) they needed 7.5K problems x 100 samples of labels; our per-benchmark probe uses ~100 labelled questions x 8 samples, so their 'not beneficial at low dataset sizes' warning is the right frame for our data-efficiency claim; (iv) their '30x' should never be conflated with our 7B-vs-32B numbers — different task, format and era.

<small>Read from: html.</small>

#### Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility

*Mohsen Hariri et al. · 2026 · arXiv preprint · [arXiv:2608.04001](https://arxiv.org/abs/2608.04001) · read priority 2*

**In one line.** A 2026 formalisation of test-time scaling into three regimes with an explicit rule that evaluation cost must be accounted separately from generation cost — the accounting frame in which our 'verification is free' claim has to be stated.

- **Models.** Open-weight reasoning models including DeepSeek-R1, Qwen, Phi-4, Open-R1, NovaSky and others (Appendix D) [html].
- **Method.** Formalises TTS as budgeted inference over the prefix tree: (a) single-trajectory sequential scaling ('at every step, there is at most one unfinished active prefix' [§2.2]); (b) leaf-level scaling ('Additional compute produces a bank of completed candidates. Any interaction between candidate trajectories is deferred until after completion' [§2.3]) with a terminal reducer mapping N samples to one output; (c) prefix-level scaling (search over unfinished prefixes [§2.4]). Introduces a discovery-stability profile S_k with Pass@k = S_{k,1}, pass^k = S_{k,k}, Maj@k ≥ S_{k,floor(k/2)+1}, Geom@k = sqrt(S_{k,1}·S_{k,k}), plus a finite-bank hypergeometric version [html]. Compute accounting: 'C_total = C_gen + C_eval', 'C_eval = C_signal + C_control + C_decision'; 'Latency under parallel or overlapping stages, throughput, peak memory, and other noncommensurate resources are not additive terms in this equation and should be reported separately' [§3.3].
- **Datasets.** MMLU-Pro (14 domains) + BBH (23 tasks); AIME'24/'25, HMMT'25, BrUMO'25; AIME'26, HMMT, CMIMC, SMT; a SuperGPQA subset [Table 1].
- **Experiments.** Large candidate-bank release: abstract says '1,403,520 sampled model attempts'; Table 1 of the updated HTML lists '22,449 questions' and '1,948,821 responses' (the two figures differ between abstract and table — quote whichever version you cite). Blocks: 27 models x 1 response on MMLU-Pro/BBH = 500,661; 20 configs x 120 questions x 80 = 192,000; 7 models x 186 x 80 = 104,160; 4 models x 3,600 x 80 = 1,152,000 [Table 1].
- **Results.** Headline [§4.2]: candidate discovery (pass@k) 'substantially outpaces reducer accuracy' as budgets grow; 'Best-of-N and verifier-guided reranking can improve accuracy, but selection can overoptimize' [§3.3]; 'soft selection can overoptimize imperfect proxy scores as the candidate set grows' [§2.3.2]; non-monotone best-of-N on mathematics at larger candidate sets [html]. Reproducibility: exact replay needs 'inference library and version, numerical formats, quantization, kernels, and batch schedule'; distributional claims 'require independent protocol reruns or a resampling design' [§3.5].
- **Conclusions.** 'Test-time scaling' is several different algorithms; results are incomparable unless the regime, the reducer, the compute split (generation vs evaluation) and the reproducibility mode are all stated.

> **Why it matters to us.** Adopt their language: our method is leaf-level scaling with a learned terminal reducer; our probe's cost is C_signal (the scoring forward), with C_control = 0 (fixed N) and C_decision = argmax. 'Verification is free' then becomes a checkable statement: C_signal for a 918k-parameter MLP on already-computed hidden states is ~0 FLOPs relative to C_gen, and the hidden states are tapped during generation so there is no extra forward. Their pass@k-vs-reducer gap is our oracle@8 vs selected accuracy (selection efficiency). Their over-optimisation warning matches our observation that selection efficiency does not rise with N. Their reproducibility checklist (library version, numerics, batch schedule) is exactly the list of numerics landmines our repo keeps hitting (TF32, thread count, row order).

<small>Read from: html.</small>

#### Test-Time Scaling for Small VLMs on Multilingual Visual MCQ

*Spiros Baxevanakis and Peng-Jian Yang · 2026 · ImageCLEF 2026 · [arXiv:2607.09438](https://arxiv.org/abs/2607.09438) · read priority 2*

**In one line.** On multilingual visual multiple-choice with small VLMs, neither a training-free generative critic nor a trained multimodal PRM beats plain majority vote; the wins come from letting chains finish and from the policy model, not from the selector.

- **Models.** Qwen2.5-VL-7B-Instruct and Qwen3.5-4B (bfloat16); selectors: a training-free generative critic (three rubric axes, App. F) and the discriminative PRM Qwen-VL-PRM-7B [html].
- **Method.** Self-consistency with N in {8, 16} chains at T=0.7 and majority vote; describe-then-reason with PRM-guided beam search (PRM-BAS, N=2 descriptions, B0=4, B=2, τ=0.05, d=6) [§3]; two post-hoc selectors over the sampled chains; a guided answer-repair step for unparseable outputs. Cost reported as 'Calls/q' (VLM forwards per question) and seconds per question [Table 1].
- **Datasets.** EXAMS-V / ImageCLEF 2026 Visual MCQ: '4,651 questions, 11 languages' validation; '1,117' held-out test; questions are 'text-in-image panels' across 'thirteen languages and twenty school subjects' [§1].
- **Experiments.** Ablations over answer-format/parseability, per-chain token limit, number of chains, search vs sampling, selector type, policy model; leaderboard submission.
- **Results.** Token budget 1k → 2k: '+3.7 pp' [§5.3]; chains 8 → 16: '+0.15 pp' [§5.3]; PRM-BAS vs self-consistency: '−0.39 pp' at '8.7x' cost [§5.2]; policy switch Qwen2.5 → Qwen3.5: '+11.4 pp' [§5.5]; val-full '81.6%' [§6]; held-out test '84.1%' [§5.8, Table 12], first on the leaderboard. Abstract: 'neither a training-free generative critic nor a trained multimodal PRM beats majority vote across both policies'.
- **Conclusions.** 'What matters is the conditions under which TTS runs, not the search or verification machinery' — on this MCQ task the selector is not the lever.

> **Why it matters to us.** The cleanest 2026 negative for verifier-based selection on a VLM — and it is on multiple choice. That is consistent with our repo's format finding (selection signals are weak on discrete options, AUROC ~0.6, and strong on free text, ~0.87), so cite it as the MCQ side of the contrast, not as a contradiction of our open-ended result. Their trained PRM was a 7B model (Qwen-VL-PRM-7B) that still lost to majority vote; ours is a 918k probe that beats it on free text — the difference is the answer format, and we should say so explicitly. Their 'Calls/q' cost unit is worth adopting alongside FLOPs.

<small>Read from: html.</small>

#### AVIS: Adaptive Test-Time Scaling for Vision-Language Models

*Ahmadreza Jeddi et al. · 2026 · arXiv preprint · [arXiv:2606.11576](https://arxiv.org/abs/2606.11576) · read priority 2*

**In one line.** A 2026 adaptive test-time policy for VLMs that prunes visual tokens and picks the number of self-consistency rollouts per query with a learned difficulty predictor, all rollouts sharing one prefill/KV cache — accuracy up 3% at 52% less compute on Qwen2.5-VL-7B.

- **Models.** Qwen2.5-VL-7B [§5.1]; RL-post-trained variants VL-Rethinker, Vision-R1, OpenVLThinker [Table 4]; CoT mode with '<think> </think> and <answer> </answer> format' [§5.1].
- **Method.** Two axes: Visual Context Scaling via Key-Diversity-Visual (KDV) pruning, a training-free O(N) rule that removes redundant visual tokens before prefill; Visual Reasoning Scaling via adaptive self-consistency. The predictor 'takes as input the sequence of visual token embeddings', a short stack of '1D Conv-GroupNorm-SiLU layers ... global average pooling, and ... a shallow MLP' [§4.2]; supervision from K_max = 10 rollouts per calibration example, 'empirical solvability ... p(x_j) = 1/K_max Σ[ŷ^(k)(x_j) = y_j*]' binarised at 0.5; calibration set '5000 multi-choice questions (4000 images, 1000 videos)' [§4.2]; a 'calibrated piecewise-constant budget policy' maps predicted solvability to K in {1, 3, 5, 7}, following an 'inverted-U' ('rollouts help most for intermediate p̂(x) and are wasted at either extreme') [§4.2]. Final answer by majority vote over the K rollouts.
- **Datasets.** 12 image benchmarks (MathVista, MathVerse, MathVision, DocVQA, MMMU-Pro, MME, MMStar, MMBench, CVBench, POPE, BLINK, TreeBench) and 6 video benchmarks (VideoMME, TempCompass, Video-TT, MVBench, Q-Bench-Video, Video-MMMU) [§5.1]; none medical.
- **Experiments.** Accuracy vs normalised FLOPs (#F) for vanilla, pruning-only, fixed-K rollouts, and AVIS; matched-FLOPs comparison [Table 2]; latency [Table 3]; RL-post-trained backbones [Table 4].
- **Results.** Cost model: 'F(x_i; θ_i) ≈ F_v(I_i) + C_M(ρ_i·n_{v,i} + n_{q,i} + K_i·T_i)' [§3.1]; 'All K rollouts reuse the same visual features and prefilling KV cache, so increasing K mainly adds decoding-side cost rather than repeating the full multimodal forward pass' [§3.3]; image-benchmark average: '3% improvement in accuracy ... while reducing compute by 52%' vs vanilla [§5.2]; 'Under approximately matched FLOPs, AVIS outperforms the closest competing policy by 3.7%' [§5.2, Table 2]; MathVista: vanilla 67.5 (#F 1.0), KDV-75% 66.6 (#F 0.38), K=5 no pruning 70.5 (#F 1.45), AVIS 68.1 (#F 0.46) [Table 1]; latency: vanilla 795 s, KDV-only 558 s, both axes ρ=75%/K=5 789 s, AVIS 757 s [Table 3].
- **Conclusions.** Compute should be allocated jointly over visual context and reasoning rollouts, per query; shared-prefill makes parallel rollouts cheap enough that adaptive self-consistency can be net compute-negative.

> **Why it matters to us.** The most explicit statement in the VLM literature of the cost convention our 'verification is free' claim relies on: N rollouts = one prefill + N decodes (K_i·T_i term), so the marginal cost of a candidate is its generated tokens only. Their learned difficulty predictor reads visual-token embeddings before generation to choose K; our probe reads hidden states after generation to choose which candidate — complementary, and a natural combination (predict N, then select with the probe). Their selection is majority vote on MCQ, so for open-ended answers our probe is the missing reducer. Their inverted-U (rollouts pay off at intermediate solvability) is the same shape as where our gains concentrate.

<small>Read from: html.</small>

#### Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling

*Runze Liu et al. · 2025 · arXiv preprint · [arXiv:2502.06703](https://arxiv.org/abs/2502.06703) · read priority 2*

**In one line.** Systematises compute-optimal test-time scaling across policy models, PRMs and difficulty: the best strategy depends on all three, PRMs transfer poorly across policies, and with the right recipe very small policies beat much larger ones on MATH-500/AIME24.

- **Models.** Policy: Llama-3 (1B, 3B, 8B) and Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B). PRMs: Math-Shepherd-PRM-7B, RLHFlow-PRM-Mistral-8B, RLHFlow-PRM-Deepseek-8B, Skywork-PRM-1.5B/7B, Qwen2.5-Math-PRM-7B/72B [html].
- **Method.** Best-of-N, beam search and Diverse Verifier Tree Search (DVTS), each guided by a PRM; a 'reward-aware' compute-optimal policy that picks the method per (policy, PRM, difficulty); difficulty by absolute pass@1 thresholds 'easy: 50-100%, medium: 10-50%, hard: 0-10%', argued to be 'more effective than quantiles' [§3.2].
- **Datasets.** MATH-500, AIME24.
- **Experiments.** Grid over policy x PRM x method x budget; FLOPs-matched comparison of small policies with TTS against large models without TTS [Table 3, Table 4].
- **Results.** [Table 3, MATH-500, as extracted] Qwen2.5-0.5B '76.4' vs GPT-4o '74.6'; Llama-3.2-3B '78.2' vs Llama-3.1-405B '71.4'; DeepSeek-R1-Distill-7B '95.2' vs o1 '94.8' and DeepSeek-R1 '97.3'; the extraction also lists Llama-3.2-1B at '66.2' vs 405B '71.4' — the abstract's '1B LLM can exceed a 405B LLM on MATH-500' claim must therefore come from a larger-budget row not captured here; verify in the PDF before quoting. [Table 4] total inference FLOPs 3B model '3.07x10^17' vs 405B '4.25x10^17'; overall '100x~1000x' FLOPs reduction claimed. Findings: 'compute-optimal TTS strategy Should be Reward-Aware' [§3.1]; 'PRMs are hard to generalize across policy models and tasks' and 'optimal TTS method depends on the PRM used' [§4.2]; smaller models benefit from search on harder problems, larger models favour best-of-N [§4.3].
- **Conclusions.** There is no single best TTS method; the verifier and the policy must be chosen together, and reported together.

> **Why it matters to us.** Two usable points. First, their 'PRMs are hard to generalize across policy models' is a reason our design is sound: the probe is trained on the very generator it scores (policy-matched by construction), so there is no verifier/policy mismatch — but it also means our probe should not be expected to transfer to another generator without refitting (which we observed: per-benchmark, ~100 labelled questions). Second, their absolute pass@1 difficulty bins (0-10 / 10-50 / 50-100%) are a cleaner convention than quantiles for our per-question analysis of where the probe helps.

<small>Read from: html.</small>

#### Limits and Gains of Test-Time Scaling in Vision-Language Reasoning

*Mohammadjavad Ahmadpour et al. · 2025 · arXiv preprint · [arXiv:2512.11109](https://arxiv.org/abs/2512.11109) · read priority 2*

**In one line.** On open-source VLMs, best-of-N with an external verifier is the most reliable test-time gain (+7 to +10 points on MathVista/MMMU), self-refinement hurts, and perception-centric MMBench barely moves.

- **Models.** Open-source: Qwen2.5-VL-7B-Instruct, InternVL2.5-8B, Mulberry-8b. Closed-source: Gemini 2.0 Flash, GPT-4o mini, Claude-3-Haiku. External verifier: Gemini 2 Flash, scoring responses on 'a scale of 0.25, 0.5, 0.75, or 1' [html].
- **Method.** Chain-of-thought; Best-of-N ('generating N independent reasoning paths' scored by a reward mechanism, highest returned) with n=5; Self-Consistency (majority vote, n=5); Self-Refinement (max three iterations); beam search with confidence-based (token probabilities) or verifier-based scoring, beam width 2, max 5 steps [html].
- **Datasets.** MathVista ('6,141 examples drawn from 31 different datasets'), MMMU ('11,500 questions ... six major disciplines, 30 subjects'), MMBench ('3,000 questions across 20 skill domains') — none medical.
- **Experiments.** Each method x each model on the three benchmarks; reasoning-heavy (MathVista, MMMU) vs perception-heavy (MMBench) split.
- **Results.** MathVista: Best-of-N on Qwen '68.25%' → '75.36%' (+7.11) [Table 1]; Self-Refinement on Gemini '80.09%' → '89.57%' (+9.48) [Table 1]; self-refinement on open models degrades performance. MMMU: Best-of-N (verifier) on Qwen '36.7%' → '47.1%' (+10.4) [Table 2]. MMBench: 'minimal or no improvement over the strong Zero-Shot baseline' [html]. Abstract: 'external verification provides the most reliable gains, whereas iterative refinement often degrades performance'; gains are 'limited ... on perception-focused benchmarks'.
- **Conclusions.** 'TTS is not a universal solution and must be tailored to both model capabilities and task characteristics'; for open-source VLMs the win comes from selection by a verifier, not from the model rewriting itself.

> **Why it matters to us.** Strongest external support for our design choice (verifier-selected best-of-N on an open ~7B VLM, same Qwen2.5-VL-7B family as our generator). Differences to flag: their verifier is a second, closed-source model — an expensive C_eval — while ours is a probe on the generator's own hidden states with no extra forward pass; their N is 5 vs our 8; their tasks are multiple-choice/short-answer, not free text. Their perception-benchmark null result is the warning we already live with on perception-heavy medical VQA.

<small>Read from: html.</small>

#### Test-Time-Scaling for Zero-Shot Diagnosis with Visual-Language Reasoning

*Ji Young Byun et al. · 2025 · arXiv preprint · [arXiv:2506.11166](https://arxiv.org/abs/2506.11166) · read priority 2*

**In one line.** Medical-imaging test-time scaling by majority vote over N=16 describe-then-diagnose samples lifts a near-chance zero-shot VLM to AUC 0.82 on pneumonia X-rays — a parallel-sampling baseline on medical images without any learned selector.

- **Models.** Stage 1 VLM: Llama-3.2-11B-Vision-Instruct; Stage 2 LLM: Llama-3.2-11B-Vision-Instruct, with ablations on 'smaller 1B, 3B, and 8B sizes of Llama text-only models and a medical domain-specific text-only model, Med42-v2-8B' [§3].
- **Method.** Two-stage zero-shot pipeline: the VLM produces N sampled visual descriptions '{v^(i)}_{i=1}^N = VLM(x, q_1)' at temperature > 0; each is passed to the LLM for a diagnosis 'a^(i) = LLM(q_2(v^(i)))'; consolidation by majority voting, i.e. the fraction '1/N Σ 1(a^(i) = boxed{1})' [§2.2]; N = 16 in the main runs, N from 1 to 16 in ablation [Fig. 2].
- **Datasets.** PneumoniaMNIST ('390 pneumonia cases and 234 normal cases from frontal X-ray images'), PathMNIST ('1,233 cases of colorectal adenocarcinoma epithelium and 741 cases of normal colon mucosa'), RetinaMNIST ('226 cases of referable ... diabetic retinopathy and 174 normal cases') — all binary classification, 224x224 images [§3].
- **Experiments.** Single-sample vs TTS (N=16) for the proposed describe-then-diagnose pipeline, a zero-shot baseline and a one-stage CoT baseline; N sweep; LLM-size ablation.
- **Results.** [Table 1, AUC | AP, single → TTS] PneumoniaMNIST: proposed 0.517|0.634 → 0.821|0.862; zero-shot 0.495|0.621 → 0.737|0.790; one-stage CoT 0.530|0.643 → 0.779|0.831. PathMNIST: proposed 0.544|0.646 → 0.653|0.752; zero-shot 0.557|0.653 → 0.562|0.656; CoT 0.475|0.615 → 0.529|0.639. RetinaMNIST: proposed 0.570|0.606 → 0.705|0.736; zero-shot 0.607|0.625 → 0.707|0.736; CoT 0.580|0.609 → 0.672|0.743. Ablation: 'a 3B model performs comparably to an 11B model' in stage 2 [§3]; no runtime or FLOPs reported.
- **Conclusions.** Parallel test-time scaling improves zero-shot diagnostic reliability across radiology, histopathology and ophthalmology, for their method and for baselines.

> **Why it matters to us.** A medical-imaging TTS data point in our regime (parallel samples, frozen models, no training) — but it is binary classification on 28-pixel-derived MedMNIST images upscaled to 224, with a general VLM starting near chance (AUC ~0.5), and selection is a vote on a binary label; it is not free-text VQA and has no learned verifier. Use it as the 'vote-based consolidation' baseline family in the medical related-work paragraph, and as evidence that even majority vote has headroom when the single-sample model is weak. Its N=16 with no cost accounting is the practice our cost section improves on.

<small>Read from: html.</small>

#### Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models

*Yangzhen Wu et al. · 2024 · arXiv preprint · [arXiv:2408.00724](https://arxiv.org/abs/2408.00724) · read priority 2*

**In one line.** Charts accuracy vs inference FLOPs for greedy, majority vote, best-of-n, weighted vote and tree search across model sizes: a 7B model with a good inference strategy is Pareto-better than a 34B one, and voting-style selectors provably saturate.

- **Models.** Pythia 410M, 1.4B, 2.8B, 6.9B, 12B; Llemma-7B and Llemma-34B; Mistral-7B; Llama3-8B-Instruct. Reward model: 'Llemma-34B reward model, which we finetuned on the synthetic process reward modeling dataset, Math-Shepherd' [§4.1].
- **Method.** Inference strategies [§3.1]: greedy; majority voting; best-of-n ('chooses the one with the highest score given by the reward model'); weighted majority voting; MCTS; REBASE (reward-balanced tree search, 'node-quality reward to control node expansion, which eliminates the need for explicit rollouts'). Compute 'based on the commonly-used formula proposed by Kaplan et al. (2020)' in FLOPs per question [§4.2]; budgets at 2^i samples; temperature 1.0; max 1024 output tokens.
- **Datasets.** GSM8K; MATH500 (plus MATH-easy levels 1-2 and MATH-hard levels 3-5); MBPP code [App. D.2].
- **Experiments.** Accuracy-vs-FLOPs curves per model size and strategy; Llemma-7B vs Llemma-34B FLOP-matched; REBASE vs sampling at equal sample counts; theorems on the convergence of (weighted) majority voting.
- **Results.** 'Llemma-7B requires around 2x less total FLOPs than Llemma-34B to achieve comparable accuracy' [§4.2]; Llemma-7B outperforms Llemma-34B under weighted majority and best-of-n until ~128 samples [Fig. 4]; REBASE with 32 samples beats sampling with 256 samples [Table 1]; for Llemma-7B 'Rebase achieves higher accuracy with 7 times less compute' [§4.3]; Theorems 1 & 2: majority/weighted voting 'converges with infinite samples' with 'exponential convergence speed' [§3.1].
- **Conclusions.** Smaller models plus better inference algorithms are Pareto-optimal in cost-vs-accuracy; sampling-based selectors saturate, so beyond a point more samples buy nothing without a better selector or search.

> **Why it matters to us.** Same accounting convention as our repo (Kaplan-style FLOPs per question), same headline shape (7B + selection vs a ~5x larger model), so it is the natural template for our accuracy-vs-FLOPs figure. Two cautions: their reward model is a 34B model — larger than the policy — so their best-of-n cost includes a big verifier pass, whereas ours is a 918k-parameter head with no extra forward pass; and their saturation theorems are about voting on a fixed answer set, which is why a learned selector, not more votes, is the only way past our selection wall.

<small>Read from: html.</small>

#### From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models

*Sean Welleck et al. · 2024 · arXiv preprint · [arXiv:2406.16838](https://arxiv.org/abs/2406.16838) · read priority 2*

**In one line.** The survey that fixes the vocabulary: token-level 'decoding' vs sequence-level 'meta-generation' (chained, parallel, refinement, tree search), with best-of-N defined as reranking an N-best list by a scoring function such as a learned verifier.

- **Models.** Survey; no experiments of its own.
- **Method.** Unified formalism: token-level generation algorithms 'operate by sampling a single token at a time or constructing a token-level search space'; meta-generation algorithms 'work on partial or full sequences, incorporating domain knowledge, enabling backtracking, and integrating external information' and are 'algorithms that call sub-generators'. Parallel meta-generators 'generate multiple trajectories in parallel, then merge the resulting terminal states' via reranking ('orders an N-best list with a reranking function'), transformation, or sequence-level rejection sampling. Best-of-N 'refers to generating an N-best list and picking the best sequence according to a scoring function'; 'A common approach ... is to learn a verifier v_ψ(x,y) → [0,1] that predicts the probability that an output y is correct, and use it within Best-of-N' [html].
- **Datasets.** n/a (survey).
- **Experiments.** n/a (survey).
- **Results.** No numbers. Cost remarks: best-of-N 'usually incurs only a linear increase in computational complexity compared to top-1 decoding'; best-of-N 'may find sequences that overoptimize the reward' when the learned reward model is off-distribution [html]. Efficient generation is split into token-cost efficiency and speed (systems) efficiency.
- **Conclusions.** Inference-time algorithms form a coherent field with three communities (classic NLP decoding, LLM meta-generation, ML systems); best-of-N with a verifier is the canonical parallel meta-generator.

> **Why it matters to us.** This is where Leo should take his terminology from, so the professor's 'misuses field terminology' complaint goes away: our method is a parallel meta-generator with a learned-verifier reranker (best-of-N); majority vote is the transformation/vote alternative; the probe is the 'scoring function v(y)'. It also names the failure mode we must test for — reward over-optimisation of the probe as N grows.

<small>Read from: html.</small>


#### Also in this area (8), in brief

- **Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects** — Jun Zhang et al. (2026), [arXiv:2604.05546](https://arxiv.org/abs/2604.05546). Survey of LVLM inference efficiency organised by lifecycle stage — visual encoding (compute-bound), prefill (compute- and memory-bound, quadratic in visual tokens), decoding (memory-bandwidth-bound) — and the 'visual token dominance' problem.<br><small>*For us:* Gives the correct systems vocabulary for our cost story. Our best-of-8 runs one prefill (image + question) and eight decodes; per this survey decoding is memory-bound at batch 1, so batching the 8 decodes largely reuses the same weight and visual-KV reads — which is why 8 samples cost far less than 8x in wall-clock and energy even though FLOPs scale with generated tokens. It also names the real cost we under-report: peak memory (the KV cache is replicated per sample). Use TTFT/TPOT when reporting latency.</small>
- **Test-Time Scaling Makes Overtraining Compute-Optimal** — Nicholas Roberts et al. (2026), [arXiv:2604.01411](https://arxiv.org/abs/2604.01411). Once inference samples are in the budget (cost 2·N·k), compute-optimal pretraining shifts to much smaller, heavily overtrained models — small model + repeated sampling is a compute-optimal design, not a workaround.<br><small>*For us:* The cleanest theoretical justification for our 7B-with-8-samples design over a single 32B call, in the same 2·N·k inference-cost convention we use. One caveat we must add whenever we cite it: their objective is pass@k (oracle coverage), i.e. it assumes a perfect selector; our measured selection efficiency (~78-81% of oracle@8) is exactly the discount that has to be applied to their forecasts in domains without automatic checking.</small>
- **Compute-Accuracy Pareto Frontiers for Open-Source Reasoning Large Language Models** — Ákos Prucs et al. (2025), [arXiv:2512.24776](https://arxiv.org/abs/2512.24776). A compute-aware benchmark of 19 open reasoning LLMs with a prefill-inclusive, architecture-aware FLOPs-per-query formula: MoE models dominate the accuracy-vs-FLOPs Pareto frontier and inference compute saturates task by task.<br><small>*For us:* Provides a written-out prefill-inclusive FLOPs formula — the thing our repo calls 'prefill-inclusive FLOPs' but has never cited a source for. Use their Eq. 7 form (or state that we use the linear 2N-per-token approximation and why the attention term is negligible at our ~1k-token contexts; their 7-15% at 4K is the number to cite). Their 'incorrect traces are longer' is a reminder that candidate length is a confound for any selector, including a hidden-state probe averaged over generated tokens.</small>
- **Intelligence per Watt: Measuring Intelligence Efficiency of Local AI** — Jon Saad-Falcon et al. (2025), [arXiv:2511.07885](https://arxiv.org/abs/2511.07885). Defines intelligence-per-watt (accuracy per watt / per joule) and gives a measured, per-query energy protocol across 20+ local LMs and 8 accelerators — the reference for how to report energy honestly.<br><small>*For us:* Our repo's energy figures (e.g. '−84.3% energy') are CPU re-costings from saved dumps with per-leg constants, not measurements; this paper is the protocol to copy if we ever measure: NVML at 50 ms, integrate to joules, report per-query APJ, aggregate per GPU. Until then any energy number must be labelled as estimated, per CLAUDE.md rule 7. Note their metric mixes accuracy and power in one ratio — for a cascade/best-of-N comparison, keep accuracy and joules as two axes and report the Pareto point instead.</small>
- **Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet** — James Xu Zhao et al. (2025), [arXiv:2509.06861](https://arxiv.org/abs/2509.06861). On closed-book knowledge questions, more reasoning tokens give 'minimal or no accuracy gains' across 14 reasoning models, and a data-processing-inequality argument shows compute-only test-time scaling cannot add information about the answer that the fixed model lacks.<br><small>*For us:* Two uses. (1) A sharp reason why a trained probe can do what training-free selection cannot: their theorem covers compute-only post-processing of a fixed model; our probe is trained on ~100 labelled in-domain questions per benchmark, which is new information about the answer distribution — the only thing in our project that broke the random-pick floor was the trained verifier, exactly as this bound predicts. (2) Medical VQA is partly knowledge-intensive (diagnosis names, anatomy), so their null result on longer reasoning is consistent with our 'reasoning hurts perception' cells and argues for spending compute on parallel samples plus selection rather than on longer chains.</small>
- **Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems** — Lingjiao Chen et al. (2024), [arXiv:2403.02419](https://arxiv.org/abs/2403.02419). Majority voting over more LLM calls is not monotone: accuracy rises on 'easy' items and falls on 'hard' ones, so a mixed task can peak and then decline — and the optimum can be predicted from as few as five calls.<br><small>*For us:* Explains why our self-consistency baseline can be flat or negative on some benchmarks while the probe still gains: majority vote amplifies the generator's mode, which on 'hard' items is wrong; a verifier can pick a minority-but-correct candidate. Also a modelling tool: their easy/hard split is a two-population view of our coverage wall (37% 'hard' where no candidate is right in 8). Their K-selection-from-5-calls trick could set N per benchmark for us.</small>
- **Theoretical guarantees on the best-of-n alignment policy** — Ahmad Beirami et al. (2024), [arXiv:2401.01879](https://arxiv.org/abs/2401.01879). Proves that best-of-n drifts from the base model by at most KL ≤ log(n) − (n−1)/n (the commonly quoted formula is only an upper bound) and that its win rate over the base policy is at most n/(n+1).<br><small>*For us:* Gives two textbook facts for the paper's method section: (i) best-of-N is a policy in its own right (the 'best-of-N policy'), so our 7B+probe system can be described as a policy whose divergence from the base 7B is bounded; evaluating their bound at our n = 8 gives KL ≤ ln 8 − 7/8 ≈ 1.20 nats and win rate ≤ 8/9 ≈ 0.89 (these two numbers are our evaluation of their formulas, not figures from the paper); (ii) the win-rate cap n/(n+1) is a sanity check on any 'beats greedy' claim — improvement per question is bounded by how often a better candidate exists, which is our coverage wall again.</small>
- **WebGPT: Browser-assisted question-answering with human feedback** — Reiichiro Nakano et al. (2021), [arXiv:2112.09332](https://arxiv.org/abs/2112.09332). Earliest large-scale best-of-n on open-ended long-form answers: rank 4/16/64 samples with a human-preference reward model; the compute-optimal n grows with model size (760M→4, 13B→16, 175B→64).<br><small>*For us:* The closest historical precedent in answer format: free-text answers judged by a learned scorer, not exact match. Their finding that the compute-optimal n rises with model size suggests our N=8 on a 7B model is in the right range and that a sweep of N per model is a legitimate experiment. Their reward model is a full LM copy (same size as the policy) — the opposite end of the cost spectrum from our probe. Note the term 'rejection sampling' is used by this paper as a synonym for best-of-n.</small>

<details><summary>Considered and not carded (4)</summary>

- `2506.11989` — Verified (Pattern Recognition 179 (2026) 113639, DOI 10.1016/j.patcog.2026.113639) but not carded here: 'Thought Graph Traversal' is sequential test-time scaling (structured prompting + reasoning budget forcing) for chest X-ray report generation, with no sampling or candidate selection; it belongs in a medical-VLM or sequential-reasoning category, not the best-of-N foundations.
- `2511.12446` — Search hit 'CoTBox-TTT' is test-time *training* (weight updates at inference) for medical VQA, not test-time scaling of a frozen model; not verified or carded.
- `2506.13888` — Search hit 'VL-GenRM' (vision-language generative reward model evaluated by best-of-N accuracy) is a verifier/reward-model paper; not verified here — belongs to the verifier/PRM category.
- `2503.20271` — Search hit 'ViLBench' (vision-language process reward modelling benchmark with best-of-N evaluation) is a verifier/PRM paper; not verified here — belongs to the verifier/PRM category.

</details>

## 3.2 Verifiers and Reward Models: Outcome vs Process, Discriminative vs Generative, and Verification in Medical VQA

A 'verifier' or 'reward model' is a second model (or a small scoring head) that judges whether a candidate answer produced by a generator model is correct or good, so it can be used to pick the best of several sampled candidates (best-of-N) or as a training signal. Two axes of taxonomy recur throughout this literature, and together they classify our own scorer: (1) outcome vs process — an outcome reward model (ORM) scores only the final answer, a process reward model (PRM) scores each intermediate reasoning step; ORMs originated with Cobbe et al. (2021) and the ORM-vs-PRM comparison was sharpened on math reasoning (Uesato 2022; Lightman 2023; Wang/Math-Shepherd 2023); (2) discriminative vs generative — a discriminative verifier outputs a single scalar score (typically trained with BCE or MSE against a correctness label), while a generative verifier (GenRM) is trained to emit its judgment as next-token text, optionally with chain-of-thought (Zhang et al. 2024). Reward models are only imperfect proxies for true correctness, so optimizing too hard against one causes 'reward-model overoptimization' / 'reward hacking', observed both during RL training (Gao et al. 2022) and purely at inference time under best-of-N (Khalaf et al. 2025); dedicated benchmarks such as RewardBench, VL-RewardBench, and Med-RewardBench exist specifically to measure how good a reward model itself is. 'Self-verification' — re-querying the same or a similar model to judge its own answer — is a popular cheap alternative to a trained verifier, but medical-VQA-specific work (Verification Mirage, which tests the Lingshu family directly) shows it is unreliable because the verifier inherits the generator's own blind spots. Because no single verifier is perfect, several 2025-2026 papers instead ensemble many weak verifiers (Weaver, FUSE) or exploit partial, claim-level evidence rather than trusting one scalar whole-answer score (Best-of-Evidence, in medical VQA specifically). Our own scorer is a small MLP trained with binary cross-entropy on a frozen hidden state, read out once per candidate at the end of generation — i.e. it is a POINTWISE, DISCRIMINATIVE, OUTCOME reward model (ORM), in the same family as Cobbe's original verifier, just probing hidden states instead of appending a scalar head to token logits — and recent multi-domain evidence (Lee et al. 2025) suggests this corner of the taxonomy is not a compromise: discriminative ORM ties discriminative PRM outside math-adjacent domains, and even a paper built around training PRMs finds outcome-mode scoring beats process-mode step selection at test-time scaling (Ong et al. 2025).

#### ★ Best-of-Evidence: Best-of-N Selection under Partial Verification

*Cenwei Zhang et al. · 2026 · arXiv preprint · [arXiv:2607.20950](https://arxiv.org/abs/2607.20950) · read priority 1 · **PDF in `papers/`***

**In one line.** Best-of-N selection for medical VQA when no single reliable whole-answer verifier exists, only partial/claim-level checkable evidence; formalizes this as a candidate-factor graph with a budgeted evidence controller, and measures only modest, often not-significant gains over plain majority-vote/BoN.

- **Models.** Candidate generator: Qwen3-VL-30B-A3B. Evidence judge (scores individual claims/factors, not whole answers): Qwen3-VL-235B-A22B [html].
- **Method.** Best-of-Evidence (BoE) keeps the BoN candidate pool fixed and represents reusable claims shared across candidates as a signed candidate–factor graph (a claim can support one candidate and contradict another). A score-based controller spends a limited 'evidence action' budget to query partial verifications of claims/factors (not full-candidate verification) that are likely to change the final pick; zero budget recovers plain BoN. Theoretically: residual evidence capacity bounds any evidence-driven improvement, and shared factor queries give an O(log K) vs Θ(K) query-count separation over per-candidate verification in a factor-code model [abstract].
- **Datasets.** Four medical VQA benchmarks [html]: VQA-Med (2,334 questions, open-answer fixed battery), PathVQA (9,903 questions, open-answer fixed battery), PMC-VQA (10,000 questions, 'option-hidden open protocol'), MedXpertQA-MM (2,000 questions, single-answer 5-way MCQ, 1–6 images per question).
- **Experiments.** K=16 candidates sampled at temperature 1.1 per question; compares BoE against plain majority-vote/BoN and against a random-factor-acquisition ablation, at evidence budget C=16, reporting percentage-point deltas with confidence intervals, plus a 'rescue rate' on the subset of questions where plain BoN's majority vote was wrong [html].
- **Results.** At budget C=16 vs plain BoN, overall accuracy deltas with 95% CIs [html]: VQA-Med +0.26pp [−0.47,+0.99] (vs +1.11pp [+0.43,+1.80] for random-factor-acquisition); PathVQA +0.43pp [+0.05,+0.81]; PMC-VQA +0.58pp [+0.19,+0.97]; MedXpertQA-MM +0.40pp [−0.15,+0.95]. On the MajorityWrong subset (questions where plain BoN's vote failed): VQA-Med — BoE rescues 4.44% of raw failures vs 2.45% for random factor acquisition; PMC-VQA shows a smaller separation between BoE and random; PathVQA is 'effectively tied with random acquisition' [html]. The paper itself states results 'reveal... the channel-quality and candidate-generation limits that prevent universal gains' [abstract].
- **Conclusions.** Partial, claim-level evidence can rescue some best-of-N failures in medical VQA when the evidence is reliable, contrastive to specific claims, and decision-relevant, but the gains are small (well under 1 percentage point on most benchmarks) and on at least two of four benchmarks/CI ranges the improvement is not distinguishable from a random-acquisition baseline or from zero [abstract, html].

> **Why it matters to us.** The single most directly comparable BoN-selection paper for medical VQA: same problem shape (BoN over open-ended medical VQA answers, want a better selection mechanism than plain majority vote/verifier), but a different mechanism (claim-level evidence graph + budgeted queries against a giant judge model, vs. our frozen-hidden-state MLP probe with no extra inference calls). Their small, CI-crossing-zero deltas (≤0.6pp on 3 of 4 datasets) are a useful external data point for how hard the selection wall is on medical VQA even with a much more expensive, evidence-grounded selector than ours — consistent with our own ~78-81% selection-efficiency ceiling finding. Also: their generator/judge (Qwen3-VL family) and benchmark set (VQA-Med, PathVQA, PMC-VQA, MedXpertQA-MM) partially overlap ours (PathVQA-open, VQA-Med 2019 C4-Abnormality), giving a candidate cross-check point.

<small>Read from: html.</small>

#### ★ Verification Mirage: Mapping the Reliability Boundary of Self-Verification in Medical VQA

*Ruinan Jin et al. · 2026 · arXiv preprint · [arXiv:2605.10850](https://arxiv.org/abs/2605.10850) · read priority 1 · **PDF in `papers/`***

**In one line.** Shows that self-verification (re-invoking the same or a similar VLM in a fresh context to judge its own answer) is systematically unreliable in medical VQA — the verifier inherits the generator's blind spots ('verification mirage') and under-attends to the image ('lazy verifier'); Lingshu is one of the six tested models.

- **Models.** Six open-weight VLMs, including Lingshu: Qwen2.5-VL-7B-Instruct, Gemma-3, Phi-4-Multimodal-Instruct, MedGemma, HuatuoGPT-Vision, and Lingshu [html].
- **Method.** Self-verification = re-invoking the same VLM in a fresh context/prompt to check its own previously generated answer, used as a cheap safety layer. The paper decomposes verifier behavior into discrimination capability (can it tell right from wrong?) and agreement bias (does it just agree with the generator regardless?), and separately measures generator-verifier coupling (does verifier error correlate with generator error, i.e. do they share blind spots?) and an image-attention 'lazy verifier' saliency comparison between generator and verifier passes [abstract, html §3.2].
- **Datasets.** Five medical VQA datasets: VQA-RAD, PathVQA, SLAKE, PMC-VQA, MedXpertQA. Seven medical tasks (§3.2): modality recognition, anatomical identification, disease classification, spatial localization, quantitative measurement, differential diagnosis, causal explanation [html].
- **Experiments.** For each of 6 VLMs × 5 datasets × 7 task types, measure verifier discrimination/agreement-bias occupancy in the 'mirage' regime (§4.2); logistic mixed-effects models linking verifier error/agreement-bias to generator correctness (§4.3); saliency-based image-attention comparison between generator and verifier passes (§4.4, the 'lazy verifier'); and multi-turn actor-verifier feedback loops tracking whether initially-wrong answers get corrected or locked in (§4.6) [html].
- **Results.** Mirage regime (§4.2): 'verifier error ≳40%' with 'agreement bias well above the midpoint' and 'FPR ≳60%'; differential-diagnosis cells saturate FPR at '∼95–100%'. Coupling (§4.3): generator errors carry '57× higher odds of verifier failure' (p<0.001), and 'the most susceptible model shows coupling nearly 400× stronger than the most robust'. Lazy verifier (§4.4): verifier image-attention is significantly lower than generator image-attention (Δ=+0.027, Cohen's d=2.33, p<0.001). Multi-turn loops (§4.6): '69.5%–87.1% [of initially wrong answers] are locked in by false verification'; only '2.2–3.8% of initially wrong answers are corrected'. All quotes/figures from the fetched HTML, tagged to their section.
- **Conclusions.** Self-verification does not provide an independent safety signal in medical VQA: the verifier is capacity-coupled to the generator, systematically under-attends to the image relative to the generator, and reusing verification in multi-turn loops mostly locks in errors rather than fixing them; the failure is worst on knowledge-intensive tasks (e.g. differential diagnosis) and mildest on perceptual tasks. Cross-verification (a different model checking the answer) reduces but does not eliminate the mirage [abstract].

> **Why it matters to us.** Most important paper in this category for our project — same model family (Lingshu is one of the six tested VLMs), overlapping benchmarks (VQA-RAD, PathVQA, SLAKE, PMC-VQA all appear in our eval set too), and it is the sharpest available evidence for WHY we built a separate trained verifier instead of self-verification/LLM-as-judge: their 'lazy verifier' + 57x coupling results are a documented failure mode of exactly the cheap alternative (re-prompt the same VLM to check itself) that our MLP-probe design avoids. Caveat for us: our probe reads hidden states from the SAME frozen Lingshu-7B that generated the candidate, so it is a different mechanism (latent-space discriminative scoring, not a re-invoked text-based self-judgment) but is still coupled to the same underlying representations — this paper is a reason to explicitly check whether our probe's errors correlate with the generator's errors in the same way, rather than assume the coupling problem is solved just because we didn't use a text-prompted self-verifier.

<small>Read from: html.</small>

#### Rethinking Reward Models for Multi-Domain Test-Time Scaling

*Dong Bok Lee et al. · 2025 · arXiv preprint · [arXiv:2510.00492](https://arxiv.org/abs/2510.00492) · read priority 1*

**In one line.** First unified evaluation of all four reward-model types (discriminative ORM/PRM, generative ORM/PRM) across 14 diverse domains; finds discriminative ORM ties discriminative PRM, generative PRM is not competitive, and generative ORM is the most robust overall.

- **Models.** Four reward model variants compared: discriminative ORM (dORM), discriminative PRM (dPRM), generative ORM (gORM), generative PRM (gPRM) [abstract].
- **Method.** Trains/evaluates all four reward-model types under a common protocol across 14 domains (not all math-adjacent, unlike most prior PRM-vs-ORM work) [abstract].
- **Datasets.** 14 diverse domains (specific dataset names not given in abstract) [abstract].
- **Experiments.** Head-to-head comparison of dORM, dPRM, gORM, gPRM as test-time-scaling verifiers across the 14 domains; theoretical and empirical analysis of error compounding in stepwise (process) aggregation over long reasoning trajectories, including self-correcting reasoning [abstract].
- **Results.** '(i) dORM performs on par with dPRM, (ii) gPRM is not competitive, and (iii) overall, gORM is the most robust, yielding significant and consistent gains across every tested domain' [abstract]. gPRM's weakness is attributed to inheriting label noise from LLM-based automatic step labeling, which compounds over long trajectories [abstract].
- **Conclusions.** The common assumption that finer-grained (process) supervision is always better does not hold outside math-adjacent domains; discriminative ORM is just as good as discriminative PRM, and if choosing a generative verifier, outcome-level (gORM) is the more robust multi-domain choice [abstract].

> **Why it matters to us.** One of the most important non-medical cards for positioning our method: this is a DIRECT, multi-domain, controlled 2x2 study of exactly our taxonomy's two axes (outcome/process × discriminative/generative) and its headline result — 'dORM performs on par with dPRM' — is a strong, recent, broad-domain vindication of our own placement (discriminative ORM) as competitive with the fancier alternatives, not merely a cheap simplification; should be cited alongside GenRM and Uesato/Lightman as the paper that settles (for non-math domains) that our corner of the 2x2 taxonomy is not disadvantaged.

<small>Read from: abstract-only.</small>

#### ★ Generative Verifiers: Reward Modeling as Next-Token Prediction

*Lunjun Zhang et al. · 2024 · ICLR 2025 · [arXiv:2408.15240](https://arxiv.org/abs/2408.15240) · read priority 1 · **PDF in `papers/`***

**In one line.** Proposes GenRM: train the verifier to emit its correctness judgment as generated text (next-token prediction, optionally with chain-of-thought) instead of a single discriminative scalar score, and shows this beats discriminative verifiers and LLM-as-judge on best-of-N.

- **Models.** Gemma-2B, Gemma-7B, Gemma2-9B used as verifiers; Gemini 1.0 Pro used to generate solutions and synthetic verification chain-of-thought rationales [html].
- **Method.** Discriminative verifiers are trained as classifiers that output one scalar score per candidate. GenRM instead fine-tunes an LM with the standard next-token-prediction / SFT loss jointly on (a) verification examples — predicting a 'Yes/No' correctness token, optionally preceded by a generated chain-of-thought rationale — and (b) correct-solution generation examples: loss = L_SFT(verification data) + λ·L_SFT(correct-solution data) [html]. At test time the verifier's probability of the 'Yes' token (optionally averaged over multiple sampled CoT rationales, i.e. majority voting over verification itself) is the score used to rerank best-of-N candidates.
- **Datasets.** Algorithmic reasoning tasks, GSM8K, MATH (500-problem easy-to-hard split), MMLU abstract algebra / elementary / high-school / college mathematics [abstract, html Table 1].
- **Experiments.** Compare best-of-N accuracy using (i) a discriminative verifier (Disc-RM), (ii) GenRM with chain-of-thought (GenRM-CoT), and (iii) LLM-as-judge, across algorithmic tasks, GSM8K, MATH easy-to-hard generalization, and MMLU math subsets; also studies scaling with verifier model size and test-time compute (majority-voted verification) [abstract, html].
- **Results.** Best-of-N accuracy gains over the ungated base rate: algorithmic tasks 5% → 45.3% [abstract]; GSM8K 73% → 93.4% [abstract]; MATH easy-to-hard 28% → 44.6% [abstract]; MMLU abstract algebra 37.9% → 53.5% [abstract]. MMLU Mathematics breakdown, Base/Disc-RM/GenRM-CoT [html, Table 1]: Elementary Math 80.1%/90.6%/91.1%; High School Math 52.2%/74.8%/76.1%; College Math 47.6%/53%/56.1%; Abstract Algebra 37.9%/50%/53.5% — GenRM-CoT beats Disc-RM on every row shown.
- **Conclusions.** Reformulating verification as text generation (rather than a single discriminative scalar) lets the verifier use chain-of-thought reasoning and extra test-time compute (sampling+majority-voting its own judgment), and this outperforms discriminative verifiers and LLM-as-judge, with gains that grow with verifier model size [abstract].

> **Why it matters to us.** Direct taxonomic contrast for our card: our scorer is the DISCRIMINATIVE side of this paper's dichotomy (Disc-RM in their own Table 1) — a scalar BCE-trained head, not a generative next-token judge. GenRM's finding that Disc-RM already recovers most of the gain (e.g. 90.6% vs 91.1% on Elementary Math) supports treating a lightweight discriminative probe as a reasonable, much cheaper choice versus a full generative verifier, which is the design tradeoff we made (918k-parameter MLP vs. re-running a full LLM as judge).

<small>Read from: html.</small>

#### ★ Training Verifiers to Solve Math Word Problems

*Karl Cobbe et al. · 2021 · arXiv preprint · [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) · read priority 1 · **PDF in `papers/`***

**In one line.** Introduces the outcome reward model (ORM): sample many candidate solutions, score each with a trained verifier, and keep the top-scoring one — the origin of best-of-N verification.

- **Models.** GPT-3-style language models (6B and 175B) fine-tuned on GSM8K; the verifier is the SAME base language model architecture with a scalar head added: 'a single bias parameter and single gain parameter that operate on the logits outputted by the language model's final unembedding layer' [html, Appendix E].
- **Method.** Sample up to 100 (and up to 400 in an ablation) candidate solutions per problem from a fine-tuned generator; a separately trained verifier assigns a scalar correctness score to each candidate solution and the highest-scoring one is returned (best-of-N via a verifier). The verifier is trained with a joint objective; 'verifier loss [as] MSE' [html, Table 1/Appendix B], and it is trained 'to make a scalar prediction after each token in the solution. This can be viewed as a token-level value function' [html, §4.3] — i.e. a pointwise, discriminative outcome verifier, scored on the FINAL answer's correctness label even though it emits a token-level trace.
- **Datasets.** GSM8K — 8.5K grade-school math word problems introduced by this paper [abstract].
- **Experiments.** Compare (a) a fine-tuned generator alone (1 guess) vs (b) generation + verifier reranking, at various N (up to 100, with a 400-guess ablation), across two model scales (6B, 175B) [html §5.1, Fig. 7a].
- **Results.** 6B model: finetuning-only ~20.6% (1 guess) vs verification ~51.5% (100 guesses) [html, §5, approximate figures as summarized from the paper's reported results]. 175B model: finetuning-only ~58.8% vs verification ~67.4% [html, §5]. Authors conclude '6B verification slightly outperforms a finetuned 175B model, thereby offering a boost approximately equivalent to a 30x model size increase' [html, §6]. (These specific percentages were extracted via an automated page-summary of the arXiv HTML version, not read directly off a table image, so treat them as approximate pending a PDF read; the qualitative comparison — 6B+verifier ≈ 175B alone — is the paper's own stated conclusion.)
- **Conclusions.** Verification (train a separate scorer, sample many candidates, pick the highest-scored one) scales better with added data than just fine-tuning the generator harder, and is dramatically more compute-efficient than scaling model size for this task [abstract, html §6].

> **Why it matters to us.** This is the founding paper of our own method's family: our MLP probe is exactly this idea (pointwise, discriminative outcome verifier, best-of-N selection) transplanted to a frozen medical VLM and open-ended VQA, with the head reading a hidden state instead of appending a scalar head to logits. Every 'ORM' in the modern literature (including GenRM, Math-Shepherd's outcome variant, and our own scorer) is a lineal descendant of this design. Establishes the term VERIFIER as this field's name for what we call a probe/scorer.

<small>Read from: html.</small>

#### Scaling Medical Reasoning Verification via Tool-Integrated Reinforcement Learning

*Hang Zhang et al. · 2026 · arXiv preprint · [arXiv:2601.20221](https://arxiv.org/abs/2601.20221) · read priority 2*

**In one line.** An agentic medical reasoning verifier that iteratively QUERIES external medical corpora during evaluation (tool-integrated, trained via RL with only trace-level supervision), giving large accuracy gains and an 8x reduction in required sampling budget vs prior reward-model baselines.

- **Models.** An agentic verifier framework trained via reinforcement learning with tool integration (base model not specified in abstract) [abstract].
- **Method.** Instead of a scalar/text judgment from a single forward pass, the verifier can iteratively query external medical corpora (retrieval tool calls) WHILE verifying, adapting its evidence-gathering as verification proceeds; trained with an iterative RL paradigm needing only trace-level (not step-level) supervision, plus an adaptive curriculum that adjusts training-data distribution dynamically [abstract].
- **Datasets.** Four medical reasoning benchmarks including MedQA and MedXpertQA (two named explicitly in the results) [abstract].
- **Experiments.** Compare against existing verification methods and prior reward-model baselines on accuracy and on required sampling budget, across the four medical reasoning benchmarks [abstract].
- **Results.** 'Improving MedQA accuracy by 23.5% and MedXpertQA by 32.0% relative to the base generator' [abstract]; '8× reduction in sampling budget requirement compared to prior reward model baselines' [abstract].
- **Conclusions.** Grounding a medical verifier in dynamically retrieved external evidence (rather than only the model's own reasoning/representations) gives large accuracy gains and much greater sampling efficiency than prior scalar reward-model verifiers [abstract].

> **Why it matters to us.** A medical verifier that is the near-opposite design point from ours: agentic, tool-augmented, retrieval-grounded, and trained via RL, versus our frozen-forward-pass, no-extra-calls, BCE-trained MLP probe. Directly relevant to our project's own RAG-direction history (retrieval was investigated and killed in our project — see CLAUDE.md §2 item 3/6 — because the 32B model's failures were found to be capacity-limited, not knowledge-retrievable, in roughly equal measure on knowledge (38%) vs perception (36%) questions): Med-TIV's large gains from retrieval-augmented verification are a useful counterpoint to flag and reconcile (their task is medical REASONING verification/MedQA-style knowledge QA, ours is medical VQA correctness scoring — the domains where retrieval helps may differ, which is consistent with, not contradictory to, our own retrieval-killed finding being specific to image-grounded VQA).

<small>Read from: abstract-only.</small>

#### Med-RewardBench: Benchmarking Reward Models and Judges for Medical Multimodal Large Language Models

*Meidan Ding et al. · 2025 · arXiv preprint · [arXiv:2508.21430](https://arxiv.org/abs/2508.21430) · read priority 2*

**In one line.** The first benchmark specifically for medical multimodal reward models and judges (1,026 expert-annotated cases across 13 organ systems / 8 clinical departments, 6 clinically-critical evaluation dimensions), evaluating 32 MLLMs and finding substantial misalignment with expert judgment.

- **Models.** 32 state-of-the-art MLLMs evaluated as judges/reward models, spanning open-source, proprietary, and medical-specific models [abstract].
- **Method.** A rigorous three-step process builds a multimodal dataset spanning 13 organ systems and 8 clinical departments, with 1,026 expert-annotated cases scored along six clinically critical evaluation dimensions (e.g. presumably diagnostic accuracy, clinical relevance — exact 6 dimension names not given in abstract, 'not extracted') [abstract].
- **Datasets.** Med-RewardBench itself: 1,026 expert-annotated multimodal medical cases, 13 organ systems, 8 clinical departments [abstract].
- **Experiments.** Evaluate 32 MLLMs (open-source, proprietary, medical-specific) as reward models/judges against expert annotation; also train and evaluate baseline medical reward models via fine-tuning [abstract].
- **Results.** 'Substantial challenges in aligning outputs with expert judgment' across the 32 evaluated MLLMs [abstract]; fine-tuned baseline models 'demonstrate substantial performance improvements' over the off-the-shelf MLLM judges [abstract]. No specific accuracy/agreement numbers given in the abstract; 'not extracted'.
- **Conclusions.** Medical reward models/judges are underexplored and current general-purpose or even medical MLLMs judge poorly against clinical expert standards out of the box, but targeted fine-tuning substantially closes the gap — motivating dedicated medical reward-model development rather than reusing general VL judges [abstract].

> **Why it matters to us.** The direct medical-domain analogue of VL-RewardBench/RewardBench, and — per the task brief — new to this project and important to cite: establishes that medical multimodal reward modeling is now a recognized, actively benchmarked sub-area, with the same headline finding as our own project's third generalizing finding ('training, not size, is the active ingredient in verification') — their fine-tuned baseline reward models beat off-the-shelf judges/MLLMs, paralleling our result that a trained 7B verifier beats a zero-shot 32B verifier. Also a candidate future evaluation target for scoring our own probe's outputs against expert-annotated medical judgment rather than only an LLM-judge/exact-match currency.

<small>Read from: abstract-only.</small>

#### Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning: Key Insights and Lessons Learned

*Brandon Ong et al. · 2025 · arXiv preprint · [arXiv:2509.23250](https://arxiv.org/abs/2509.23250) · read priority 2*

**In one line.** A systematic study of vision-language process reward models (VL-PRMs) that finds, among other things, that using a VL-PRM as an OUTCOME reward model (ORM) at test-time scaling outperforms using it for process-level step selection — an ORM-beats-PRM-usage result even from a PRM-focused paper.

- **Models.** Vision-language PRMs (backbone not specified in abstract; 'smaller VL-PRMs' vs larger compared) [abstract].
- **Method.** Combines Monte Carlo Tree Search (MCTS) with judgments from a strong VLM to build more accurate step-level labels (hybrid data synthesis); adds perception-focused supervision so the PRM explicitly detects visual-grounding errors, not just logic errors; systematically evaluates multiple test-time-scaling (TTS) strategies [abstract].
- **Datasets.** MMMU, PuzzleVQA, AlgoPuzzleVQA, MathVista, MathVision — five multimodal benchmarks [abstract].
- **Experiments.** Compare VL-PRMs used as ORMs (scoring full outcomes) vs used for process-level step selection during test-time scaling; compare smaller vs larger VL-PRMs; test whether perception-focused supervision helps; test generalization to math datasets not seen during PRM training [abstract].
- **Results.** (i) 'VL-PRMs when used as Outcome Reward Models (ORMs) during test-time scaling (TTS) can outperform VL-PRM guided process step selection' [abstract]; (ii) 'smaller VL-PRMs can match or even surpass larger ones in detecting process errors' [abstract]; (iii) VL-PRMs 'uncover latent reasoning abilities in stronger VLM backbones' [abstract]; (iv) 'perception-level supervision leads to significant gains in test-time scaling' [abstract]; (v) TTS performance improves on advanced math datasets 'despite not training VL-PRMs on such datasets' [abstract]. No specific accuracy numbers given in the abstract.
- **Conclusions.** Even when you build a process reward model, using it in an OUTCOME (whole-candidate) scoring mode at test time can beat using its process-level step scores for guiding search; perception-focused supervision and model scale both matter, but not always in the expected direction (smaller can match larger) [abstract].

> **Why it matters to us.** Directly supports our own architecture choice from an unexpected angle: even a paper built around training PROCESS reward models finds that OUTCOME-mode usage of the reward model wins at test-time scaling — reinforcing that ORM-style scoring (our design) is not merely 'simpler', it can be the empirically stronger choice for selection, matching our own project's emphasis that best-of-N + outcome scoring is a strong, non-naive baseline design, not a fallback.

<small>Read from: abstract-only.</small>

#### Process Reward Models for Sentence-Level Verification of LVLM Radiology Reports

*Alois Thomas et al. · 2025 · arXiv preprint · [arXiv:2510.23217](https://arxiv.org/abs/2510.23217) · read priority 2*

**In one line.** A lightweight 0.5B-parameter sentence-level PRM for radiology report generation, trained with weak supervision on MIMIC-CXR, that outperforms white-box (internal-state) baselines, generalizes to an unseen LVLM, and improves weighted best-of-N selection.

- **Models.** A lightweight 0.5B-parameter PRM; tested for generalization on 'an unseen LVLM' (name not given in abstract) [abstract].
- **Method.** Sentence-level Process Reward Model: predicts the factual correctness of EACH generated sentence in a radiology report, conditioned on clinical context and preceding text; trained with weakly-supervised labels (not full human annotation) [abstract]. Explicitly contrasted with 'methods reliant on internal model states' (i.e. white-box / hidden-state probing baselines) — the PRM instead only needs the generated text, giving model-agnostic generalization [abstract].
- **Datasets.** MIMIC-CXR (fine-tuning); MIMIC-CXR test set (best-of-N evaluation) [abstract].
- **Experiments.** Compare the PRM to existing verification techniques and white-box baselines on outputs from one LVLM (in-domain); test generalization to an unseen LVLM; test report-filtering utility (discard worst-scored reports); test weighted best-of-N report selection [abstract].
- **Results.** 'Relative improvements of 7.5% in Matthews Correlation Coefficient and 1.8% in AUROC over strong white-box baselines' on one LVLM's outputs [abstract]. Filtering the worst 10% of reports by PRM score 'improv[es] F1-CheXbert scores by 4.5%' [abstract]. Weighted best-of-N selection guided by the PRM gives 'relative improvements in clinical metrics of 7.4% for F1-CheXbert and 0.6% for BERTScore' on the MIMIC-CXR test set [abstract].
- **Conclusions.** A small, text-only (not hidden-state-based), sentence-level process reward model can outperform white-box internal-state verification methods on radiology report factuality, and — unlike hidden-state methods — generalizes to LVLMs it was never trained on, at the cost of needing per-sentence weak labels rather than exploiting the generator's internals [abstract].

> **Why it matters to us.** Important contrast paper — per the task brief, new to this project and matters: this PRM is explicitly positioned AGAINST 'methods reliant on internal model states' (i.e. against our own family of hidden-state probes), and its headline selling point is cross-model generalization, which our probe explicitly lacks (needs ~100 in-domain labels per benchmark and is tied to one frozen backbone's hidden states). It is also SENTENCE-LEVEL (process, radiology reports are multi-sentence) rather than whole-candidate (outcome, our short VQA answers), so it is a PRM not an ORM — but its best-of-N gains (7.4% F1-CheXbert relative) are a useful same-modality (medical, LVLM-generated text), different-mechanism number to compare our own +0.0736 macro improvement against, with the big caveat that the metrics, tasks (report generation vs VQA), and improvement definitions are not directly comparable.

<small>Read from: abstract-only.</small>

#### Shrinking the Generation-Verification Gap with Weak Verifiers

*Jon Saad-Falcon et al. · 2025 · NeurIPS 2025 · [arXiv:2506.18203](https://arxiv.org/abs/2506.18203) · read priority 2*

**In one line.** Weaver: combine many weak, imperfect verifiers (LM judges + reward models) into a strong ensemble using weak supervision (to estimate each verifier's accuracy WITHOUT ground-truth labels), reaching o3-mini-level accuracy with a much weaker generator (Llama 3.3 70B).

- **Models.** Generator: Llama 3.3 70B Instruct. Verifier ensemble: 70B-or-smaller judge and reward models (unweighted count/names not given in abstract). Also distills the ensemble into a 400M cross-encoder [abstract].
- **Method.** Weighted ensembles of verifiers outperform unweighted ones, but weighting normally needs labeled data; Weaver instead uses WEAK SUPERVISION to estimate each verifier's accuracy from unlabeled data and combine their outputs into one unified score, handling inconsistent output formats and filtering low-quality verifiers via dataset statistics [abstract].
- **Datasets.** Reasoning and math tasks (specific benchmark names not given in abstract) [abstract].
- **Experiments.** Test-time repeated sampling: generate multiple candidates, select via the Weaver-combined verifier score vs Pass@1 baseline; compare against the jump from GPT-4o to o3-mini; distill the ensemble into a small 400M cross-encoder for efficiency [abstract].
- **Results.** Weaver + Llama 3.3 70B Instruct 'achiev[es] o3-mini-level accuracy... (87.7% average)' [abstract]. 'This gain mirrors the jump between GPT-4o and o3-mini (69.0% vs. 86.7%)' [abstract], which normally requires extensive finetuning/post-training.
- **Conclusions.** You do not need one great verifier or labeled data to weight an ensemble — weak supervision can estimate verifier reliability unsupervised, and a big-enough ensemble of weak/imperfect verifiers on a weaker generator can match the accuracy gain of a much more expensive frontier-model upgrade [abstract].

> **Why it matters to us.** Introduces WEAK-VERIFIER ENSEMBLING as a distinct strategy from our own (a single trained probe, 24 rank-averaged instances across 3 layers × 8 seeds — note: our '24 probes' are ensembled copies of the SAME architecture/training recipe, not diverse heterogeneous verifiers, so Weaver's regime (combining structurally different judges/RMs) is a different kind of ensembling than ours); relevant as a pointer for future work — our project could explore Weaver-style weak-supervision combination across our different candidate signals (probe score, self-consistency, answer-prior) instead of just rank-averaging homogeneous probe seeds.

<small>Read from: abstract-only.</small>

#### Inference-Time Reward Hacking in Large Language Models

*Hadi Khalaf et al. · 2025 · NeurIPS 2025 (Spotlight) · [arXiv:2506.19248](https://arxiv.org/abs/2506.19248) · read priority 2*

**In one line.** Shows reward hacking also occurs purely at INFERENCE time (not just during RL training) under Best-of-n and Soft-Best-of-n selection, characterizes it as inevitable for a broad class of inference-time mechanisms, and proposes HedgeTune / Best-of-Poisson to mitigate it by 'hedging' on the proxy reward.

- **Models.** General LLM inference-time alignment setups (specific backbone not named in abstract) [abstract].
- **Method.** Studies Best-of-n (BoN), Soft Best-of-n (SBoN), and introduces Best-of-Poisson (BoP) — 'an efficient, near-exact approximation of the optimal reward-KL divergence policy at inference time' [abstract]. Introduces HedgeTune, an algorithm to find the optimal inference-time hedging parameter that trades off proxy-reward optimization against distortion from the base policy [abstract].
- **Datasets.** Math, reasoning, and human-preference setups (specific benchmark names not given in abstract) [abstract].
- **Experiments.** Measure true (gold) reward as proxy-reward optimization pressure increases, under BoN/SBoN/BoP, to see whether the characteristic 'true reward rises then falls' overoptimization pattern (from Gao et al. 2022, observed mainly in RL) also appears purely at inference time; test HedgeTune's reward-distortion tradeoff [abstract].
- **Results.** 'The characteristic pattern of hacking as observed in practice (where the true reward first increases before declining) is an inevitable property of a broad class of inference-time mechanisms, including BoN and BoP' [abstract]. HedgeTune 'mitigates reward hacking and achieves superior reward-distortion tradeoffs' [abstract]. No specific numeric deltas given in the abstract.
- **Conclusions.** Reward-model overoptimization (Gao et al. 2022's RL-era finding) is not specific to RL training — it is an inherent property of best-of-n-style INFERENCE-time selection too, and can be mitigated (not eliminated) by tuning how hard you optimize against the proxy at inference time [abstract].

> **Why it matters to us.** Extends Gao et al. (2022)'s reward-overoptimization warning to exactly our regime — best-of-N selection at inference time with N=8 — meaning our own probe-guided selection is theoretically susceptible to the same 'true accuracy rises then plateaus/falls as N or probe reliance grows' pattern; a concrete, citable reason to check whether our probe's gains saturate or degrade as N is pushed past 8, rather than assuming larger N is strictly better.

<small>Read from: abstract-only.</small>

#### VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models

*Lei Li et al. · 2024 · CVPR 2025 · [arXiv:2411.17451](https://arxiv.org/abs/2411.17451) · read priority 2*

**In one line.** A 1,250-example benchmark spanning general multimodal queries, hallucination detection, and complex reasoning, designed to stress-test vision-language generative reward models (VL-GenRMs); even GPT-4o only reaches 65.4% and open-source models near random.

- **Models.** Evaluates 16 leading large vision-language models as reward models, including GPT-4o and Qwen2-VL-72B [abstract].
- **Method.** An AI-assisted annotation pipeline combining sample selection with human verification curates 1,250 examples specifically designed to probe VL-GenRM limitations, across three categories: general multimodal queries, visual hallucination detection, and complex reasoning [abstract].
- **Datasets.** VL-RewardBench itself (1,250 curated examples) [abstract]; correlated against MMMU-Pro accuracy under Best-of-N sampling.
- **Experiments.** Evaluate 16 VLMs as VL-GenRM judges on VL-RewardBench; correlate VL-RewardBench score with downstream MMMU-Pro Best-of-N accuracy; analyze failure modes by task type and by model capacity, and test whether training a model explicitly to judge boosts its judgment ability [abstract].
- **Results.** 'Even GPT-4o achieves only 65.4% accuracy' [abstract]; 'state-of-the-art open-source models such as Qwen2-VL-72B, struggle to surpass random-guessing' [abstract]. VL-RewardBench score correlates with MMMU-Pro Best-of-N accuracy at Pearson's r > 0.9 [abstract]. Models 'predominantly fail at basic visual perception tasks rather than reasoning tasks' [abstract]. Training a model explicitly to judge gives '+14.7% accuracy for a 7B VL-GenRM' [abstract].
- **Conclusions.** Current vision-language generative reward models are weak, especially at basic visual perception (not reasoning), inference-time scaling benefits them unevenly by model capacity, and explicit judge-training substantially helps [abstract].

> **Why it matters to us.** Load-bearing precedent for our field-generalizing finding #2 (answer-format/perception matters for routing signals): VL-RewardBench's headline finding — VL-GenRMs fail mainly at PERCEPTION, not reasoning — echoes our own finding that reasoning hurts perception and that routing signal quality is tied to answer/task structure, but from the reward-model-quality side rather than the routing side; also the direct multimodal analogue of RewardBench, giving us the 'VL-GenRM' term and the observation that even top closed models (GPT-4o, 65.4%) are weak multimodal judges — a reason to prefer training a small, task-specific discriminative probe (our approach) over relying on a big generic judge for scoring in-domain.

<small>Read from: abstract-only.</small>

#### Let's Verify Step by Step

*Hunter Lightman et al. · 2023 · arXiv preprint · [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) · read priority 2*

**In one line.** OpenAI's large-scale PRM study: process supervision significantly outperforms outcome supervision on the harder MATH dataset, and releases PRM800K, 800K step-level human labels.

- **Models.** Not extracted beyond 'large language models' [abstract]; exact model family/size not stated in the abstract.
- **Method.** Trains a process-supervised reward model (PRM) on human-labeled per-step correctness and compares it to an outcome-supervised reward model (ORM); adds active learning to select which steps to label [abstract].
- **Datasets.** MATH dataset (challenging math competition problems); PRM800K, the released dataset of 800,000 step-level human feedback labels [abstract].
- **Experiments.** Compare process-supervised vs outcome-supervised reward models on a representative subset of the MATH test set, and measure the effect of active learning on labeling efficiency [abstract].
- **Results.** Their process-supervised model 'solves 78% of problems from a representative subset of the MATH test set' [abstract]. 'Process supervision significantly outperforms outcome supervision' [abstract]. Active learning 'significantly improves the efficacy of process supervision' [abstract].
- **Conclusions.** For hard, multi-step math reasoning, step-level (process) supervision of the reward model beats supervising only the final outcome, and PRM800K is released to support further process-supervision research [abstract].

> **Why it matters to us.** The canonical PRM reference to cite opposite our ORM choice — it establishes that PRM > ORM specifically on hard, long multi-step math reasoning; since open-ended medical VQA answers in our setting are short (not long derivations), this is evidence that the domain where PRMs win (long chains) is not the domain we are in, supporting our ORM choice as domain-appropriate rather than a missed opportunity.

<small>Read from: abstract-only.</small>

#### Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations

*Peiyi Wang et al. · 2023 · arXiv preprint · [arXiv:2312.08935](https://arxiv.org/abs/2312.08935) · read priority 2*

**In one line.** Builds a process reward model with AUTOMATICALLY constructed step-level labels (no human annotation), used both to rerank candidates (verification) and to reinforce the generator via step-by-step PPO.

- **Models.** Mistral-7B and other open-source LLMs [abstract].
- **Method.** Automatically constructs process-wise (step-level) supervision data (removing the need for costly human step labels), then trains Math-Shepherd, a PRM assigning a reward score to each solution step; used in two modes: (1) Verification — reranking multiple LLM-generated outputs; (2) Reinforcement Learning — step-by-step PPO using Math-Shepherd as the reward [abstract].
- **Datasets.** GSM8K and MATH [abstract].
- **Experiments.** Measure accuracy of Mistral-7B before/after step-by-step PPO with Math-Shepherd, and before/after best-of-N verification with Math-Shepherd, on GSM8K and MATH [abstract].
- **Results.** Step-by-step PPO with Math-Shepherd improves Mistral-7B from 77.9% → 84.1% on GSM8K and 28.6% → 33.0% on MATH [abstract]. With verification (reranking) on top, accuracy further rises to 89.1% (GSM8K) and 43.5% (MATH) [abstract].
- **Conclusions.** Process supervision does not require expensive human step-level annotation — it can be constructed automatically — and the resulting PRM is useful both as a verifier for reranking (best-of-N) and as a reward signal for RL fine-tuning [abstract].

> **Why it matters to us.** Useful as the automatic-labeling counterpoint to Lightman's human-labeled PRM800K: shows that a PRM's step labels can come from automatic rollout-based estimation rather than annotators, which is conceptually adjacent to how OUR probe's binary correctness labels are also automatically derived (from an LLM-judge + exact match) rather than hand-annotated per-candidate; also a second data point (after Uesato) that reranking (verification) alone, without any generator retraining, is a large and cheap accuracy lever — matching our own best-of-N framing.

<small>Read from: abstract-only.</small>

#### Solving math word problems with process- and outcome-based feedback

*Jonathan Uesato et al. · 2022 · arXiv preprint · [arXiv:2211.14275](https://arxiv.org/abs/2211.14275) · read priority 2*

**In one line.** First controlled comparison of outcome-based vs process-based supervision for training math reasoning models, finding outcome supervision matches process supervision on final-answer accuracy but needs process supervision (or a learned reward model emulating it) to fix reasoning-step errors.

- **Models.** Not extracted beyond 'a natural language task' setup [abstract]; DeepMind-style LLMs, exact sizes not stated in abstract.
- **Method.** Outcome-based supervision trains only on whether the final answer is correct; process-based supervision (and reward models emulating process feedback) supervise the correctness of each reasoning step [abstract].
- **Datasets.** GSM8K [abstract].
- **Experiments.** Compare final-answer error rate and reasoning-step error rate (among final-answer-correct solutions) for models trained with outcome supervision vs process supervision vs learned reward models emulating process feedback [abstract].
- **Results.** Improve previous best results from 16.8% → 12.7% final-answer error, and 14.0% → 3.4% reasoning error among final-answer-correct solutions [abstract]. 'Pure outcome-based supervision produces similar final-answer error rates with less label supervision' but process-based (or reward-model-emulated process) supervision is needed for correct reasoning STEPS [abstract].
- **Conclusions.** Outcome supervision is nearly as good as process supervision for getting the right final answer, but process-level signal (real or reward-model-emulated) is needed if you also care that the intermediate reasoning is correct, not just the final number [abstract].

> **Why it matters to us.** Early, direct ORM-vs-PRM evidence: since our scorer is outcome-only (an ORM) and open-ended medical VQA answers are typically short, non-multi-step outputs (unlike GSM8K derivations), this paper's finding — outcome supervision suffices for final-answer accuracy — is a relevant precedent for why we did not need a process-level (step-by-step) verifier for our task; it also flags a limitation we should note: if Lingshu emits any multi-step reasoning inside a candidate, our ORM cannot catch a lucky-final-answer/wrong-reasoning case the way a PRM could.

<small>Read from: abstract-only.</small>

#### Scaling Laws for Reward Model Overoptimization

*Leo Gao et al. · 2022 · arXiv preprint · [arXiv:2210.10760](https://arxiv.org/abs/2210.10760) · read priority 2*

**In one line.** Shows that optimizing hard against a proxy reward model eventually hurts true (gold-standard) performance — reward-model overoptimization / Goodhart's law — and that the gold-vs-proxy relationship follows a smooth, reward-model-size-dependent functional form for both RL and best-of-n.

- **Models.** A synthetic setup with a fixed 'gold-standard' reward model standing in for humans, and proxy reward models of varying parameter counts trained on gold-model labels [abstract].
- **Method.** Optimize a policy against a PROXY reward model (via RL, or via best-of-n sampling) and track how the policy's score under the GOLD reward model changes as optimization pressure increases; study how the coefficients of this relationship scale with reward-model size, dataset size, and (for RL) the KL penalty [abstract].
- **Datasets.** Synthetic preference-labeling setup (gold model generates labels in place of humans); no external benchmark named in the abstract.
- **Experiments.** Vary reward-model parameter count, RM training-dataset size, policy parameter count, and KL penalty coefficient; measure gold-reward score as a function of proxy-reward optimization amount, separately for RL and best-of-n optimization [abstract].
- **Results.** 'The relationship follows a different functional form depending on the method of optimization, and... its coefficients scale smoothly with the number of reward model parameters' [abstract]. No single number is given in the abstract for a specific overoptimization magnitude.
- **Conclusions.** Because a reward model is only an imperfect proxy for true preferences, optimizing against it too hard predictably degrades true (gold) performance — this is Goodhart's law in action — and the paper gives a quantitative, size-dependent scaling law for when and how much this happens, for both RL and best-of-n [abstract].

> **Why it matters to us.** Directly relevant background for our own best-of-N selection: our probe is exactly a 'proxy reward model' selecting among N=8 candidates, so this paper's core warning — heavier reliance on the proxy (e.g. larger N, or a stronger probe) can eventually trade off against true correctness — is the theoretical frame for any future push to increase N or over-fit the probe; also gives us the standard TERM (reward-model overoptimization / reward hacking) for the failure mode our selection-efficiency and coverage walls sit adjacent to.

<small>Read from: abstract-only.</small>


#### Also in this area (8), in brief

- **Grounding the Score: Explicit Visual Premise Verification for Reliable Vision-Language Process Reward Models** — Junxin Wang et al. (2026), [arXiv:2603.16253](https://arxiv.org/abs/2603.16253). Diagnoses that VL-PRMs conflate 'the reasoning step is wrong' with 'the verifier misperceived the image', and fixes it by having the policy emit an explicit visual checklist, matched against independently-derived visual constraints, to gate step rewards by visual reliability.<br><small>*For us:* General-domain (not medical) but conceptually close to our project's second generalizing finding — 'answer format determines whether routing signals work' and reasoning-vs-perception confounds — here reframed as a VERIFIER-side confound: a PRM's score conflates reasoning quality with visual-perception quality. Relevant caution for interpreting OUR probe's errors too: since our probe reads hidden states that mix visual grounding and answer-text information, a similarly entangled failure mode (probe penalizing/rewarding for the wrong reason) is plausible and not something we have separately diagnosed.</small>
- **FUSE: Ensembling Verifiers with Zero Labeled Data** — Joonhyuk Lee et al. (2026), [arXiv:2604.18547](https://arxiv.org/abs/2604.18547). Fully Unsupervised Score Ensembling (FUSE): ensembles verifiers with ZERO ground-truth labels by controlling conditional dependencies between verifiers to improve spectral (unsupervised) ensembling algorithms, matching or beating semi-supervised alternatives.<br><small>*For us:* A step further than Weaver on the same axis (weak-verifier ensembling): relevant for our writeup's discussion of alternatives to labeled-data-hungry approaches, since our own probe DOES require ~100 labeled in-domain questions per benchmark — FUSE's zero-label regime is worth flagging as a possible future direction if per-benchmark labeling cost becomes a bottleneck, though FUSE ensembles pre-existing heterogeneous verifiers/judges rather than training a new scorer from hidden states, so it addresses a different part of the pipeline (combining scores) than our method (producing a score).</small>
- **Parallel Test-Time Scaling with Multi-Sequence Verifiers** — Yegon Kim et al. (2026), [arXiv:2603.03417](https://arxiv.org/abs/2603.03417). Argues standard verifiers score each best-of-N candidate in isolation and are therefore poorly calibrated; proposes the Multi-Sequence Verifier (MSV), which scores each candidate CONDITIONED ON THE WHOLE sampled set, improving calibration, best-of-N accuracy, and enabling early-stopping to cut latency.<br><small>*For us:* Direct architectural contrast for our card: our probe scores EACH candidate independently (pointwise, no cross-candidate conditioning) then rank-averages 24 independently-trained instances — MSV's finding that CONDITIONING on the full candidate set improves both accuracy and calibration is a concrete, citable idea we do not currently use and could test (e.g. feed the probe summary statistics of the other 7 candidates' hidden states or scores); also relevant to our project's own 'selection efficiency' wall (converting only ~78-81% of oracle-of-N) — better calibration via cross-candidate conditioning is one plausible lever we have not yet tried.</small>
- **Unified Reward Model for Multimodal Understanding and Generation** — Yibin Wang et al. (2025), [arXiv:2503.05236](https://arxiv.org/abs/2503.05236). First reward model trained jointly across BOTH multimodal understanding and generation (image and video) tasks, supporting pairwise ranking and pointwise scoring, used to build DPO preference data.<br><small>*For us:* Supports both pointwise and pairwise scoring in one model — useful vocabulary contrast for our own strictly POINTWISE scorer (each candidate scored independently, no pairwise comparison); the cross-task-transfer finding is a relevant caution for our per-benchmark probe design (our probe needs ~100 in-domain labels per benchmark and does not obviously transfer, which is the opposite regime from this paper's claimed joint-training synergy — worth noting as an open question rather than a contradiction, since their tasks (generation vs understanding) are more different from each other than our within-VQA benchmarks are).</small>
- **Best-of-Majority: Minimax-Optimal Strategy for Pass@$k$ Inference Scaling** — Qiwei Di et al. (2025), [arXiv:2510.03199](https://arxiv.org/abs/2510.03199). Proves neither majority voting (self-consistency) nor best-of-N is minimax-optimal for Pass@k selection, and proposes Best-of-Majority (BoM) — restrict candidates to high-frequency ones before picking top-k by reward — with a matching regret lower bound and no performance degradation as sampling budget N grows.<br><small>*For us:* A theoretically-grounded selection-strategy alternative worth flagging as future work: our current method is pure reward-based BoN (probe picks the argmax among N=8 rank-averaged candidates) with no frequency/self-consistency pre-filter; BoM's finding that combining self-consistency with reward-based reranking is provably better is a concrete, citable reason to test a BoM-style hybrid against our probe, especially since our project already tried plain self-consistency/majority-vote as ONE of our baselines and found it weaker than the trained verifier — BoM suggests the two are not mutually exclusive.</small>
- **DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning** — Qi Cao et al. (2025), [arXiv:2505.20241](https://arxiv.org/abs/2505.20241). A multimodal PRM trained with bi-level optimization that reweights training-dataset domains to counter quality imbalance across the many multimodal reasoning datasets used for training, improving test-time-scaling gains over other data-selection and TTS approaches.<br><small>*For us:* Relevant training-methodology contrast: our probe is trained PER-BENCHMARK (needs ~100 in-domain labelled questions per benchmark, no cross-domain reweighting), whereas DreamPRM is explicitly built to handle MULTIPLE domains at once by learning how much to trust each one — a candidate technique if we ever try to train one probe across our 8 open-text benchmarks jointly instead of 8 separate per-benchmark probes.</small>
- **RewardBench: Evaluating Reward Models for Language Modeling** — Nathan Lambert et al. (2024), [arXiv:2403.13787](https://arxiv.org/abs/2403.13787). The first general-purpose benchmark and leaderboard specifically for evaluating text reward models, using structured prompt-chosen-rejected triples across chat, reasoning, and safety, with subtle-but-verifiable preference cases.<br><small>*For us:* The text-only ancestor of VL-RewardBench and Med-RewardBench (both in this card set); establishes the 'reward-model benchmark' as its own genre of paper, distinct from papers proposing a new verifier — useful for the taxonomy section of our writeup (verifier method papers vs. verifier evaluation/benchmark papers) even though it is not multimodal or medical.</small>
- **LLaVA-Critic: Learning to Evaluate Multimodal Models** — Tianyi Xiong et al. (2024), [arXiv:2410.02712](https://arxiv.org/abs/2410.02712). First open-source large multimodal model built as a generalist 'judge': trained on a diverse critic-instruction dataset, it scores/evaluates other multimodal models' outputs and doubles as a preference-learning reward signal.<br><small>*For us:* A GENERATIVE, LLM-as-judge-style multimodal reward model (contrast with our discriminative MLP probe) — relevant as the general-domain analogue of the LLM-judge WE use to LABEL correctness for training our probe (a Lingshu-32B judge); also relevant background for why an LMM-as-judge is a heavyweight, separate-forward-pass alternative to our lightweight no-extra-forward-pass probe.</small>

## 3.3 Probing frozen hidden states: from probing classifiers to hidden-state verifiers

This category traces one continuous idea across nine years: attach a small, separately-trained classifier to a big frozen model's internal activations, and read off something the big model itself never says out loud. It starts as pure interpretability (Alain & Bengio: does layer 10 linearly encode 'cat'?), gets a methodological immune system (Hewitt & Liang: or is the probe just memorizing?), and turns operational once people realize the same trick can tell whether a generated STATEMENT is true (Azaria & Mitchell) and, from there, whether one sampled ANSWER among several is the correct one — which is exactly a best-of-N verifier. The eight papers ALSO cards in this file (Q-Probe through DualRead) are our direct lineage: each reads a frozen generator's hidden states to score or rank candidate outputs, and the differences between them — whether they pool over the candidate's own generated tokens, whether they actually select among candidates (vs. calibrate one, vs. steer decoding), and whether they truly reuse the generation pass or quietly pay for a second one — are exactly the four axes (A: frozen-hidden-state classifier, B: selects among candidates, C: pools over the candidate's own generated tokens, D: free, reused from generation, no extra forward pass) that separate a paper that merely reads hidden states from a paper that does what our method does. Two papers, SWIFT/ELHSR and HSRM, match all four; most of the others match two or three and are useful precisely for showing what changes when one axis is missing.

#### A Decodability Criterion Predicts When Hidden-State Selection Beats Majority Voting in Large Language Models (CASE)

*Zhixiang wang et al. · 2026 · arXiv preprint · [arXiv:2608.17124](https://arxiv.org/abs/2608.17124) · read priority 1*

**In one line.** CASE trains a linear gate on the answer-token hidden state to select the best of several sampled candidates, and proposes "decodability" — a leakage-free measure of how well the gate ranks correct-above-incorrect candidates — to predict IN ADVANCE whether hidden-state selection will beat majority voting, tested across general and MEDICAL LLMs.

- **Models.** general and medical LLMs (specific model names not extracted from abstract) [abstract]
- **Method.** "a dynamic selection combiner that trains a linear gate on the answer-token hidden state and selects the highest-scoring candidate" [abstract]; decodability computed under "question-grouped evaluation" to remove question-identity leakage that inflates a conventional probe's apparent accuracy [abstract]
- **Datasets.** medium-difficulty and hard questions across general and medical domains, plus one held-out "unseen scientific domain" [abstract]
- **Experiments.** Measure decodability vs. the accuracy gain of selection over majority voting; test transfer of the decodability criterion to an unseen scientific domain [abstract]
- **Results.** "decodability predicts the accuracy gain of selection over voting with a Pearson correlation r=0.75 and a decision threshold near AUC=0.60" [abstract]; "CASE improves over voting by up to 19 points on medium-difficulty questions and 16.8 points on hard questions" [abstract]; transfer to an unseen scientific domain "within 3.8 points" [abstract]
- **Conclusions.** Hidden-state selection does not always beat majority voting; decodability, measured under a leakage-free (question-grouped) evaluation, tells you in advance which regime you are in, and this generalizes across model scale and to a medical domain.

> **Why it matters to us.** Matches (A) frozen-hidden-state linear classifier and (B) explicit select-vs-vote best-of-N comparison, and is tested on MEDICAL LLMs like ours. (C) is a partial match: it reads "the answer-token hidden state" (a specific token position) rather than a pooled mean over ALL of the candidate's own generated tokens the way our probe and SWIFT do — not confirmed as a pooling operation. (D) not explicitly confirmed either way in the abstract. The methodologically important contribution for us is orthogonal to A-D: "decodability" under question-grouped (leakage-free) evaluation is the closest published answer to the control-task worry raised by Hewitt & Liang above, and is a diagnostic we currently lack — it would tell us, per-benchmark, whether our probe's AUROC is a real correctness signal or partly question-identity leakage.

<small>Read from: abstract-only.</small>

#### ★ HSRM: Hidden-State Reward Models for Test-Time Verification

*Xianzhi Li and Xiaodan Zhu · 2026 · EMNLP 2026 [comment] · [arXiv:2608.30841](https://arxiv.org/abs/2608.30841) · read priority 1 · **PDF in `papers/`***

**In one line.** HSRM extracts hidden states at reasoning-step boundaries from a frozen generator, mean-pools them through a tiny (~2M-parameter) Transformer encoder to rank candidates, and explicitly verifies it needs zero extra generator forward passes because it reuses representations already computed during generation.

- **Models.** Qwen3 (1.7B, 4B, 8B, 14B) as generators; Llama-3.2 and Llama-3.1 in ablations [html]
- **Method.** Hidden states are collected only at "reasoning-step boundaries" (token positions right before step delimiters like "\n\n"), giving a short sequence H ∈ R^{S×d_gen}, S≤100 [html]. Architecture: per-step linear projection → a small 2-layer Transformer encoder (d_model=256, 4 heads, 4× feed-forward width) → mean-pooled linear readout to a scalar score; total ~2.12M–3.4M params depending on generator size [html]. Trained on SELF-GENERATED trajectories with outcome labels via a tie-safe Bradley-Terry ranking loss (correct candidates must outscore incorrect ones, no arbitrary order imposed among multiple correct ones) [html].
- **Datasets.** GSM8K, MATH-500, AIME, OlympiadBench [html]
- **Experiments.** Best-of-8 accuracy and within-problem AUROC, HSRM vs. a 55M-parameter text-only "EORM" energy verifier, across Qwen3 scales 1.7B–14B [html]
- **Results.** "HSRM matches or outperforms a 55M-parameter text-only energy verifier in 15 of 16 generator-dataset settings" while using "only about 2M parameters" [abstract]; on GSM8K "HSRM outperforms EORM at every Qwen3 scale" [html]; on MATH-500/AIME/OlympiadBench it "generally matches or exceeds EORM while remaining below Qwen2.5-Math-PRM-7B" [html]; Figure 4 shows HSRM uses "five orders of magnitude fewer verification FLOPs per candidate" than a 7B PRM baseline while matching its GSM8K accuracy [html].
- **Conclusions.** A tiny probe over hidden states already produced during generation can match a 27x-larger text-based verifier at a small fraction of the FLOPs, with training requiring no human or larger-model process supervision — explicit confirmation that verification can be nearly free relative to generation.

> **Why it matters to us.** The clearest and most EXPLICITLY STATED match on all four dimensions among the eight closest prior-art papers: (A) frozen generator, small trained readout on its hidden states; (B) best-of-8 ranking/selection is the headline metric; (C) pooled (mean-pooled, via the Transformer encoder's readout) over hidden states drawn from the candidate's own generated tokens — though over a SUBSET (step boundaries only, S≤100 positions) rather than every generated token as SWIFT and our probe do; (D) explicitly and textually confirmed — "HSRM reuses representations already computed during generation, avoiding this re-encoding step" [html], and it scores all candidates in one batched forward pass with no additional generator calls. Differences from us: math reasoning only, uses a small Transformer encoder rather than a plain MLP, and pools over step-boundary tokens rather than a simple mean over all generated tokens.

<small>Read from: html.</small>

#### ★ MedProb: Probing Internal Representations of Vision-Language Models for Medical Question Answering

*Erfan Nourbakhsh et al. · 2026 · EMNLP Findings 2026 [comment] · [arXiv:2609.04336](https://arxiv.org/abs/2609.04336) · read priority 1 · **PDF in `papers/`***

**In one line.** MedProb's MAIN method is a multinomial logistic-regression probe on a frozen medical VLM's LAST-INPUT-TOKEN hidden state that predicts a multiple-choice answer WITHOUT any free-text generation at all; only a secondary Appendix-H extension applies the probe to open-ended generations, and it re-feeds the candidate text back into the model to score it, i.e. a second forward pass.

- **Models.** Qwen2.5-VL (3B/7B/32B), Qwen3-VL (2B/4B/8B/30B), Gemma (4B/27B), LLaVA-7B, Llama-3.2-11B-Vision, InternVL3-1B, OpenFlamingo-9B, plus 14 matched general/medical VLM pairs (e.g. Gemma-4B vs. MedGemma-4B, Qwen2.5-VL-7B vs. MedVLThinker-7B, LLaVA-7B vs. LLaVA-Med-7B) [html]
- **Method.** "multinomial logistic regression probe. The probe maps the representation to a distribution over answer choices as g_l(h_i^(l)) = softmax(W^(l) h_i^(l) + b^(l))" [html]; the feature h_i^(l) is "the hidden state of the LAST INPUT TOKEN at every transformer layer" [html] — i.e. read BEFORE any answer is generated, not pooled over generated output. "The best-performing layer is selected on a held-out validation split" with 5-fold cross-validated regularization [html]. A separate Appendix H extension uses the probe "to score open-ended candidate generations in a rejection-sampling framework" [html, abstract, conclusion, limitations — quoted verbatim in all four locations, but the appendix's own body text was not reachable through the HTML/PDF fetch in this pass].
- **Datasets.** PATH-VQA, SLAKE, VQA-RAD [html]
- **Experiments.** Table 1: probe vs. zero-shot prompting vs. medical VLMs/agentic systems across the three benchmarks and many model sizes; Table 2: 14 matched general-purpose vs. medically-adapted VLM pairs; Figure 6: answer-position bias of prompting vs. probing [html]
- **Results.** "Qwen2.5-VL-32B achieves best results: SLAKE F1 91.60%, VQA-RAD F1 92.33%, PATH-VQA F1 89.94%" [html, Table 1]; "MedProb consistently outperforms zero-shot prompting by 20+ percentage points on PATH-VQA" [html]; "InternVL3-1B (MedProb): average accuracy 79.32% vs. prompting 67.83%" [html, Table 1]; position bias (Figure 6): InternVL3-1B prompting 10.49pp gap vs. MedProb 2.10pp; LLaVA-V0-7B prompting 7.33pp vs. MedProb 3.58pp [html, Fig. 6]. "Across 14 matched general-purpose and medical VLM pairs, medical adaptation does not consistently improve this linear decodability" [abstract].
- **Conclusions.** Multiple-choice Med-VQA answer content is substantially recoverable from a frozen VLM's PRE-GENERATION hidden states alone, exceeding prompting-based evaluation and revealing more signal in small models than generation reveals; medical fine-tuning does not reliably add linearly-decodable signal beyond the base model. Open-ended generation is explicitly flagged as future work, with Appendix H only "an initial step" [html, Limitations].

> **Why it matters to us.** Important precise distinction for relation_to_us: MedProb's headline method matches (A) frozen-VLM hidden-state classifier, but is a PRE-GENERATION, question-only predictor over multiple-choice options — it never generates free text and there is no candidate to pool over, so it does NOT match (B) select-among-candidates or (C) pool-over-own-generated-tokens for its main results; it is closer in spirit to a question-difficulty/solvability probe (cf. 2602.09924 "LLMs Encode Their Failures", not carded here) than to a best-of-N verifier. Its Appendix H extension is the one piece of MedProb that attempts our exact setting — open-ended candidates, probe-as-selector — but per the project's own characterization it RE-FEEDS the generated candidate text back into the model as input to score it, which costs a second forward pass and therefore fails (D), the "no extra forward pass" property that is central to our method's efficiency claim (this specific re-feeding mechanism is asserted by the project brief; the appendix's body text itself was not retrievable through arXiv's HTML/PDF converters in this pass — flagged as not independently re-verified verbatim). If accurate, this makes MedProb the clearest illustration of the (A) vs (A+B+C+D) distinction this category is built around: reading a frozen hidden state is easy and common; reading it for free, from a pass that already happened, to select among the model's own sampled answers, is the rarer and more specific thing our method and SWIFT/HSRM above do.

<small>Read from: html.</small>

#### Tiny Inference-Time Scaling with Latent Verifiers (VHS)

*Davide Bucciarelli et al. · 2026 · Findings of CVPR 2026 [comment] · [arXiv:2603.22492](https://arxiv.org/abs/2603.22492) · read priority 1*

**In one line.** "Verifier on Hidden States" (VHS) scores candidates directly from a Diffusion Transformer's intermediate hidden representations — never decoding to pixel space or re-encoding — instead of using a full multimodal LLM as an image verifier, cutting joint generation+verification time 63.3% and FLOPs 51% at matched or better quality.

- **Models.** Diffusion Transformer (DiT) single-step image generators (specific model not named in abstract); compared against "a standard MLLM verifier" [abstract]
- **Method.** VHS is a verifier that "operates directly on intermediate hidden representations of Diffusion Transformer single-step generators", analyzing generator features "without decoding to pixel space" — avoiding the standard pipeline of decoding a candidate image to pixels and re-encoding it into a (separate) visual embedding space for an MLLM verifier to judge [abstract]
- **Datasets.** GenEval (image-generation benchmark) [abstract]
- **Experiments.** Compare VHS to a standard MLLM-based verifier under tiny inference budgets (few candidates per prompt), measuring generation+verification time, FLOPs, VRAM, and GenEval score [abstract]
- **Results.** "reducing joint generation-and-verification time by 63.3%, compute FLOPs by 51% and VRAM usage by 14.5% with respect to a standard MLLM verifier, achieving a +2.7% improvement on GenEval at the same inference-time budget" [abstract]
- **Conclusions.** For image generation, verifying directly from a frozen generator's own hidden states — skipping the decode/re-encode round trip a separate MLLM verifier requires — is both cheaper and more accurate under a fixed inference-time budget than using a full MLLM as judge.

> **Why it matters to us.** Flagged in the brief as a likely very close neighbour, and it is — but from a DIFFERENT generative modality: it verifies DIFFUSION image candidates, not language/VQA-answer candidates, so there are no "generated tokens" to pool over and dimension (C) does not apply in our sense. It still matches (A) frozen-generator hidden-state classifier, (B) selection among several generated candidates for best-of-N-style inference-time scaling, and (D) explicitly — avoiding the decode-then-re-encode round trip is precisely the same "do not pay for a second forward pass through a heavyweight judge" argument that motivates our probe reading hidden states already produced during the 7B's own generation, just instantiated for pixels instead of text. It is good independent, cross-modality evidence that reading a generator's own hidden states beats invoking a separate large multimodal judge on cost, not just on our text/VQA setting.

<small>Read from: abstract-only.</small>

#### Lightweight Latent Verifiers for Efficient Meta-Generation Strategies

*Bartosz Piotrowski et al. · 2025 · arXiv preprint · [arXiv:2504.16760](https://arxiv.org/abs/2504.16760) · read priority 1*

**In one line.** LiLaVe reliably extracts a correctness signal from a frozen base LLM's hidden states at a fraction of the compute of a full LLM verifier, and plugs it into best-of-n and self-consistency, plus two new LiLaVe-based strategies.

- **Models.** not extracted (abstract does not name specific base LLMs)
- **Method.** A lightweight verifier reading hidden states of the base LLM (frozen); the exact classifier head is not stated in the abstract — per the project's own prior notes the authors tried an MLP head and instead selected an XGBoost head [project brief, not independently re-verified here].
- **Datasets.** not extracted (abstract does not name specific benchmarks)
- **Experiments.** Couples LiLaVe with best-of-n and self-consistency; introduces conditional self-correction and conditional majority voting built on top of LiLaVe's correctness signal [abstract]
- **Results.** "significantly improve both accuracy and efficiency in generation tasks with smaller LLMs" [abstract] — not extracted as specific numbers; "only a small fraction of the computational budget required by traditional LLM-based verifiers" [abstract]
- **Conclusions.** Extracting a correctness signal from hidden states is practical and can be combined with, not just substituted for, classic meta-generation strategies like best-of-n and self-consistency.

> **Why it matters to us.** Directly matches (A) frozen-hidden-state classifier and (B) selection among candidates via best-of-n. (C) pooling over the candidate's own generated tokens and (D) whether it reuses generation-time hidden states without an extra pass are not stated in the abstract we have and were not fetched (budget reserved for core papers) — flag both as not extracted. The reported architecture choice (an MLP head tried and rejected in favor of XGBoost, per project prior notes) is the most direct precedent for our own head-architecture decision and is where the naming nuance below matters most: a paper that "tries an MLP" for a probe is not describing the transformer's internal feed-forward block, it is describing a small standalone classifier network — the same category of object our own probe is.

<small>Read from: abstract-only.</small>

#### ★ Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling (SWIFT)

*Jizhou Guo et al. · 2025 · KDD 2026 (Research Track) [comment] · [arXiv:2505.12225](https://arxiv.org/abs/2505.12225) · read priority 1 · **PDF in `papers/`***

**In one line.** SWIFT is a token-level linear gate+reward head on a frozen LLM's own per-token hidden states (concatenated across all layers), trained with BCE, that computes a gated weighted-average reward per candidate and picks the argmax candidate for best-of-N — the closest architectural sibling to our probe we found.

- **Models.** Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Ministral-8B-Instruct [html, §5.1]
- **Method.** For each token t in a candidate's reasoning path, concatenate and flatten hidden states (post-residual) across all L layers: h_t = [h_t^1;...;h_t^L] [html, §4.2, Eq.2]. A single linear matrix W_SWIFT ∈ R^{2×Ld} maps h_t to a token-level reward r_t and a gating value g_t; g_t is passed through a sigmoid (Eq.3); the path's final reward R is the gating-weighted average of the per-token rewards r_t over ALL tokens of that candidate (Eq.4) [html, §4.2]. Trained with binary cross-entropy with logits (Eq.5). At inference: select i* = argmax_i R_theta(h_i) and return that candidate (Eq.1) [html].
- **Datasets.** MATH, GSM8K, AQuA_RAT (Table 1); Imbue Code, HellaSwag, CoinFlip, PKU-SafeRLHF (Table 2) [html]
- **Experiments.** Best-of-N sampling (N=4/16/64) compared to EurusRM-7B (a much larger text-based reward model) across three model backbones and seven benchmarks [html]
- **Results.** MATH (Table 1): SWIFT averages 57.5% vs. EurusRM-7B 51.0% [html, Table 1]. GSM8K: SWIFT 89.0% vs. EurusRM-7B 84.8% [html, Table 1]. AQuA_RAT: SWIFT 72.8% vs. baseline 46.5% [html, Table 1]. "12.7% higher accuracy than EurusRM-7B on MATH" [abstract]. SWIFT parameter count: 1.8×10^5 (Llama-3.2-3B) to 3.0×10^5 (Ministral-8B), "less than 0.005% of their parameters" vs. baseline reward models [html, §4.2, Table 3].
- **Conclusions.** A linear head reading hidden states already produced during generation, pooled with a learned gate over the candidate's own tokens, matches or beats a 7B text-based reward model at best-of-N while using a negligible fraction of its parameters. Evaluated only on math/code/commonsense/alignment text tasks — no medical or vision-language domain [html].

> **Why it matters to us.** The single closest architectural match among all eight prior-art papers on dimensions (A) frozen-hidden-state classifier, (B) argmax best-of-N selection (explicit, Eq.1), and (C) pooled over the CANDIDATE'S OWN generated tokens (explicit weighted average over every token t of the reasoning path, Eq.4 — closer to our simple mean pooling than HSRM's step-boundary subset below). On (D) the fetched text does not use the phrase "no extra forward pass", but by construction the gate reads hidden states already produced when the path was generated, so it matches D architecturally even though the paper does not make the efficiency claim as explicit as HSRM does. Differences from us: SWIFT's classifier is LINEAR (a single matrix), ours is a small MLP with a hidden layer; SWIFT is text-only math/code/commonsense, never medical or vision-language.

<small>Read from: html.</small>

#### ★ Q-Probe: A Lightweight Approach to Reward Maximization for Language Models

*Kenneth Li et al. · 2024 · arXiv preprint · [arXiv:2402.14688](https://arxiv.org/abs/2402.14688) · read priority 1 · **PDF in `papers/`***

**In one line.** Learns a 1-layer LINEAR probe on a frozen model's embeddings to reweight (softmax-sample, not hard-argmax) sampled completions toward higher reward, trainable via reward-modeling loss or a novel importance-weighted policy-gradient objective.

- **Models.** Code-LLaMA-7B and -70B, LLaMA-7B; GPT-3.5-turbo-1106 via API embeddings [html]
- **Method.** "1-layer (linear) probe" on model embeddings [html]; for Code-LLaMA-7B, embeddings come from "the 2626th hidden layer of the same model" [html]. Trained with mean-squared-error reward modeling (L_Q), a classification loss for binary rewards (L_CE), or (the best-performing) direct policy learning via importance-weighted policy gradients (L_PG) [html]. At inference, samples k completions and reweights them by sampling from softmax(Q_theta(x,a_i)/beta) over the probe's Q-values — a soft reweighting, not a hard best-of-k pick [html].
- **Datasets.** MBPP (code generation, Table 1); a preference-learning dataset with LLaMA-7B (Table 4) [html]
- **Experiments.** Compare Q-probe (each training objective) to baseline pass@1, few-shot combination, and a pass@48 skyline on MBPP; compare to DPO and KTO+Q-probe on preference win rate [html]
- **Results.** MBPP Table 1: baseline Pass@1 0.29; Q-probe with L_PG 0.46; with 5-shot combination 0.52; Pass@48 skyline 0.76 [html, Table 1]. Preference learning Table 4 (LLaMA-7B): baseline win rate 37.86%; Q-probe 50.10%; DPO 44.97%; KTO+Q-probe 55.01% [html, Table 4].
- **Conclusions.** A cheap linear probe on frozen embeddings, trained with the right objective (importance-weighted policy gradient beats plain reward-modeling losses), can close much of the gap to finetuning, and can even be trained through an API that exposes only sampling and embeddings, not gradients.

> **Why it matters to us.** Direct prior art for dimension (A) (linear classifier/regressor on a frozen model's embeddings) and, loosely, (B) (its softmax reweighting over k samples is a soft cousin of best-of-N selection, not a hard argmax pick as ours is). On (C): unclear from the fetched text whether the embedding pools over all generated completion tokens or takes a single position — not confirmed, so treat as not extracted. On (D): the paper states it "only assumes access to sampling and embeddings" [html], which for the GPT-3.5 API experiments implies a SEPARATE embedding call after generation (i.e., closer to a second pass), so — unlike our method and unlike HSRM/SWIFT below — Q-Probe is not clearly "free" the way a probe reading hidden states already produced during generation is; for the local open-weight setting it may reuse generation-time hidden states, but this is not confirmed in what we read. Q-Probe is the earliest of the eight closest-prior-art papers and establishes the "linear probe on frozen embeddings reweights/selects candidates" template that later papers (ELHSR/SWIFT, HSRM) make argmax-exact and generation-pass-free.

<small>Read from: html.</small>

#### Separating Capability from Confidence: Grounded Dual-State Calibration for GRPO-Trained Medical Vision-Language Models (DualRead)

*Yangyang Xie et al. · 2026 · arXiv preprint · [arXiv:2609.06419](https://arxiv.org/abs/2609.06419) · read priority 2*

**In one line.** DualRead reads a frozen, GRPO-trained medical VLM's internal states at two points — pre-answer (solvability) and post-answer (support for the answer it already gave) — to CALIBRATE the confidence of the ONE answer the model produced, rather than to select among several candidates.

- **Models.** two VLM backbones, GRPO-trained (specific model names not extracted from abstract) [abstract]
- **Method.** Freezes the GRPO-trained actor; combines "pre-answer solvability" (a question-only, pre-generation read) with "a post-answer assessment of the generated answer and its visual support" (a read over/after the model's own generated answer tokens) [abstract]. Introduces Counterfactual Confidence Grounding AUC (CCG-AUC): swaps in a different real image and checks whether confidence drops when the answer flips from correct to incorrect [abstract].
- **Datasets.** in- and out-of-distribution medical VQA benchmarks (not individually named in abstract) [abstract]
- **Experiments.** Compare correctness discrimination and calibration of DualRead's dual-state confidence vs. verbalized (model states its own confidence in words) confidence, plus the new CCG-AUC counterfactual-image test [abstract]
- **Results.** "DualRead improves correctness discrimination and calibration over verbalized confidence while preserving answer accuracy" [abstract] — not extracted as specific numbers
- **Conclusions.** Reading internal states at both a pre-answer and a post-answer point separates "can the model solve this" from "is the model's stated confidence actually grounded in the image", and beats asking the model to just say how confident it is.

> **Why it matters to us.** Matches (A) frozen-model internal-state reading and, per the project's characterization, (C) the post-answer component mean-pools over the model's own generated answer tokens (on SLAKE/VQA-RAD/PathVQA/PMC-VQA/OmniMedVQA — benchmarks that overlap several of ours). It does NOT match (B): DualRead CALIBRATES confidence in the single answer the model already committed to; it never compares or selects among multiple sampled candidates for the same question, so there is no best-of-N mechanism here at all — this is the key structural difference from us despite reading similar internal states on an overlapping medical VQA benchmark suite. (D) is plausible (reading states at moments within the same generation trajectory) but not explicitly confirmed in the abstract.

<small>Read from: abstract-only.</small>

#### HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token

*Sai Akhil Kogilathota et al. · 2026 · EACL 2026 [journal_ref] · [arXiv:2603.05465](https://arxiv.org/abs/2603.05465) · read priority 2*

**In one line.** Trains probes on three families of PRE-GENERATION VLM internal representations (visual-only, vision-token-in-decoder, query-token) to predict hallucination risk in a single forward pass, before any answer token is generated, reaching up to 0.93 AUROC on some models.

- **Models.** eight modern VLMs including Llama-3.2-Vision, Gemma-3, Phi-4-VL, Qwen2.5-VL, and Molmo [abstract]
- **Method.** Probes trained on three representation families read BEFORE generation starts: (i) visual-only features without multimodal fusion, (ii) vision-token representations inside the text decoder, (iii) query-token representations integrating visual+textual information pre-generation [abstract]
- **Datasets.** "a diverse set of vision-language tasks" (not individually named in abstract)
- **Experiments.** Compare probes on each representation family across eight VLMs, measuring hallucination-detection AUROC pre-generation [abstract]
- **Results.** "up to 0.93 AUROC on Gemma-3-12B, Phi-4-VL 5.6B, and Molmo 7B" [abstract]; "late query-token states are the most predictive for most models, while visual or mid-layer features dominate in a few architectures (e.g., ~0.79 AUROC for Qwen2.5-VL-7B using visual-only features)" [abstract]
- **Conclusions.** Hallucination risk is detectable before a single token is generated, purely from a single forward pass over the prompt/image, though the most predictive layer and modality vary by architecture. Abstention component: out of scope — the abstract's closing sentence lists "early abstention" among possible downstream uses of the probe; this card covers only the pre-generation risk-scoring contribution, not any abstention/refusal mechanism, per this project's standing rule against abstention as a method.

> **Why it matters to us.** Matches (A) frozen-VLM classifier probe. Does NOT match (B) select-among-candidates — there is exactly one forward pass and no sampled candidates to choose between. Does NOT match (C) — by design this is PRE-generation, so there are no "candidate's own generated tokens" yet to pool over (the whole point is to predict risk before generation happens, the opposite of our post-hoc, post-generation verifier). On (D): trivially there is only one forward pass, but that is a different sense of "free" than ours — it never needed a second pass because it never reads the output of a first one. This is the same structural category as MedProb's main method and 2509.10625 below: a pre-generation solvability/risk predictor, complementary to (not a substitute for) a post-hoc best-of-N verifier like ours, since it cannot distinguish which of several sampled candidates is better — only whether the question is risky in general.

<small>Read from: abstract-only.</small>

#### ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models

*Jingwei Ni et al. · 2025 · ACL 2026 Main [comment] · [arXiv:2511.06209](https://arxiv.org/abs/2511.06209) · read priority 2*

**In one line.** A transformer-based probe (<10M params) on a frozen LLM's internal states estimates step-level credibility during multi-step reasoning, matching or beating Process Reward Models up to 810x larger.

- **Models.** not fully specified in abstract beyond "a frozen LLM"; annotated either by a larger LLM (DeepSeek-R1) or self-supervised by the model itself [abstract]
- **Method.** A transformer-based probe trained on internal states of a frozen LLM to score reasoning-step credibility during generation, guiding test-time scaling (sampling/verifying/selecting intermediate steps) [abstract]
- **Datasets.** Mathematics, planning, and general knowledge QA domains [abstract]
- **Experiments.** Compare probe-based step verification to Process Reward Models (PRMs) up to 810x larger, across multiple reasoning domains [abstract]
- **Results.** "our probes match or exceed the performance of PRMs that are up to 810x larger" [abstract]; probe size "fewer than 10M parameters" [abstract]
- **Conclusions.** LLM internal states encode confidence in the model's own reasoning process at the step level, and a small probe can substitute for a much larger text-based process reward model.

> **Why it matters to us.** Matches (A) frozen-LLM internal-state classifier, and (C) — per the project's characterization, it pools over the model's own generated tokens (its abstract's "during generation" framing is consistent with D as well: no separate re-encoding step is described). It differs on (B): ReProbe selects among candidate REASONING STEPS to guide search/tree expansion within one multi-step chain, not among N complete final-answer candidates the way best-of-N over whole answers works — architecturally the same "probe-then-select" idea, but at a finer granularity than our whole-candidate selection.

<small>Read from: abstract-only.</small>

#### Multimodal LLMs as Customized Reward Models for Text-to-Image Generation (LLaVA-Reward)

*Shijie Zhou et al. · 2025 · ICCV 2025 [comment] · [arXiv:2507.21391](https://arxiv.org/abs/2507.21391) · read priority 2*

**In one line.** LLaVA-Reward scores finished text-to-image generations by feeding the (prompt, generated image) pair into an MLLM and reading its hidden states directly — skipping the MLLM's text response entirely — with a Skip-connection Cross Attention module added to strengthen image-text interaction, then uses the score for inference-time best-of-N-style scaling.

- **Models.** pretrained multimodal LLMs (base model not individually named in the abstract), text-to-image generators evaluated as the target being scored
- **Method.** Directly utilizes MLLM hidden states given (text, image) pairs rather than analyzing a generated text response; adds a Skip-connection Cross Attention (SkipCA) module connecting early-layer visual features to later-layer hidden representations to strengthen text-image correlation reasoning; supports paired and unpaired preference data for fine-tuning [abstract]
- **Datasets.** not individually named in the abstract
- **Experiments.** Trains on four evaluation perspectives — text-image alignment, fidelity/artifact, safety, and overall ranking — and evaluates automatic-evaluation score alignment with humans plus inference-time scaling for T2I generation [abstract]
- **Results.** "LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations" [abstract] — not extracted as specific numbers
- **Conclusions.** An MLLM's hidden states, read directly from a (prompt, finished image) pair, are a more efficient and effective reward signal than asking the MLLM to produce and then analyze a text judgment, and this reward model can drive best-of-N-style inference-time scaling for image generation.

> **Why it matters to us.** Matches (A) reading hidden states of an MLLM instead of its text output, and (B) it is explicitly used for inference-time scaling / candidate scoring (a best-of-N-style use over generated images). It is an instructive NEGATIVE case for (C) and (D): the MLLM being probed is a JUDGE fed the FINISHED (prompt, image) pair as fresh input, not the same model whose own generation process produced hidden states already in memory — so scoring each candidate image costs a full extra forward pass through the judge MLLM (fails D), and there is no notion of pooling over "the candidate's own generated tokens" since the judged artifact is an image, not the judge model's own generated text (fails C in our sense). This is the same structural pattern as MedProb's Appendix H and Q-Probe's API-embedding mode: reading a frozen model's hidden states is not automatically free — it is only free when it is the SAME forward pass that produced the candidate, which is exactly what SWIFT and HSRM (and our method) achieve and this paper does not.

<small>Read from: abstract-only.</small>

#### No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes

*Iván Vicente Moreno Cencerrado et al. · 2025 · ICLR 2026 workshop (Principled Design for Trustworthy AI), poster [comment] · [arXiv:2509.10625](https://arxiv.org/abs/2509.10625) · read priority 2*

**In one line.** Extracts activations after a question is read but BEFORE any answer tokens are generated, and shows a linear probe on this "in-advance correctness direction" predicts whether the forthcoming answer will be correct, generalizing across models and out-of-distribution knowledge datasets, though it falters on math reasoning.

- **Models.** three open-source model families, 7B to 70B parameters [abstract]
- **Method.** Linear probes trained on post-question, pre-generation activations to predict eventual answer correctness ("in-advance correctness direction"), trained on generic trivia questions and tested in- and out-of-distribution [abstract]
- **Datasets.** generic trivia questions (training); "diverse out-of-distribution knowledge datasets" (not individually named) [abstract]
- **Experiments.** Compare the question-only probe's predictive power to black-box baselines and to the model's own verbalized confidence, across model families and datasets; test layer-wise saturation of predictive power; test generalization to math-reasoning questions; examine correlation with the model producing "I don't know" [abstract]
- **Results.** the probe "outperform[s] black-box baselines and verbalised predicted confidence" [abstract]; "predictive power saturates in intermediate layers" [abstract]; "generalisation falters on questions requiring mathematical reasoning" [abstract]; probe score "strongly correlates" with the model responding "I don't know" [abstract] — no specific numeric values extracted
- **Conclusions.** A question-only, pre-generation linear probe recovers a genuine, fairly general "will I get this right" signal (not just a dataset-specific artifact), sufficient to beat both simple black-box baselines and the model's own stated confidence — but this signal is weaker for reasoning-heavy (math) questions, and the same direction also tracks explicit refusal ("I don't know") behavior.

> **Why it matters to us.** Matches (A) linear probe on frozen-model activations. Does not match (B), (C), or (D): this is a PRE-generation, question-only predictor — there is no candidate to select among, no generated tokens to pool over, and no generation pass to reuse from, since the read happens before generation starts at all. Same structural family as HALP and MedProb's main method: a solvability/difficulty predictor, complementary to our post-hoc verifier rather than competing with it — in principle such a probe could be a cheap pre-filter (e.g., skip best-of-N sampling on questions predicted easy) layered in FRONT of our method, though that composition is untested here. Note: the paper's "I don't know" correlation is an observed property of the model's existing outputs, not an abstention mechanism the paper proposes or tests, so no abstention-scope flag is needed for this card.

<small>Read from: abstract-only.</small>

#### ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs

*Zhenliang Zhang et al. · 2025 · ACL 2025 (Main Conference) [comment] · [arXiv:2507.16488](https://arxiv.org/abs/2507.16488) · read priority 2*

**In one line.** Instead of probing a static hidden state, ICR Probe tracks how much each module CONTRIBUTES to the hidden state's update across layers (the "Information Contribution to Residual Stream", ICR Score), and shows this cross-layer dynamic signal detects hallucination more reliably, with far fewer parameters, than probing isolated static representations.

- **Models.** not individually named in the abstract
- **Method.** Introduces the ICR Score, a metric quantifying each module's contribution to the hidden state's update as it moves through the residual stream across layers; builds the ICR Probe on top of this cross-layer update signal rather than on a single layer's static representation [abstract]
- **Datasets.** not extracted (no specific dataset names in the abstract)
- **Experiments.** Empirical validation that the ICR Score distinguishes hallucinations; comparison of the ICR Probe's hallucination-detection performance and parameter count against prior static/isolated hidden-state probing methods; ablations and case analyses of the mechanism [abstract]
- **Results.** "the ICR Probe achieves superior performance with significantly fewer parameters" [abstract] — no specific numeric values extracted
- **Conclusions.** Focusing on how hidden states CHANGE across layers (the update/contribution process), rather than only on a hidden state's final value at one or a few layers, captures hallucination-relevant signal that static, isolated-layer probing methods miss, at lower parameter cost.

> **Why it matters to us.** Matches (A) only, and even that loosely: it is a probe on a frozen LLM's internal computation, but the object it probes is the layer-to-layer UPDATE/contribution dynamic (the residual-stream delta at each layer), not a pooled or single-layer static hidden state the way our probe (and SWIFT/HSRM/Q-Probe above) reads. Does not match (B) select-among-candidates — no best-of-N or candidate-ranking framing appears in the abstract; it detects hallucination in whichever single output was generated. Does not clearly match (C) or (D) — pooling over a candidate's own generated tokens and generation-pass reuse are not addressed. Chiefly useful to us as a reminder that "probe the hidden state" is a design space, not one fixed recipe: our project pools a MEAN over final hidden states at fixed layers (18/20/22) across a candidate's generated tokens, while ICR Probe instead reads the layer-to-layer CHANGE at each step — an architecture variant we have not tried and could consider if our per-layer pooled probe plateaus.

<small>Read from: abstract-only.</small>

#### LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations

*Hadas Orgad et al. · 2024 · arXiv preprint · [arXiv:2410.02707](https://arxiv.org/abs/2410.02707) · read priority 2*

**In one line.** Shows truthfulness information is concentrated in specific tokens of an LLM's internal representation (boosting error detection), that such error detectors do NOT generalize across datasets (truthfulness encoding is multifaceted, not universal), and — most strikingly — that an LLM can internally encode the correct answer while consistently generating an incorrect one out loud.

- **Models.** not individually named in the abstract (multiple open LLMs implied)
- **Method.** Trains error/truthfulness detectors on internal representations, localizing the informative signal to specific tokens rather than using the whole sequence uniformly; also trains classifiers to predict the TYPE of error the model is likely to make, and directly compares internally-encoded answers against externally-generated ones [abstract]
- **Datasets.** not extracted (no specific dataset names in the abstract)
- **Experiments.** Error-detection accuracy using token-localized internal representations vs. prior (non-localized) approaches; cross-dataset generalization tests; error-type prediction from internal representations; case analysis of internal-vs-external answer discrepancy [abstract]
- **Results.** "leveraging this property [token-concentrated truthfulness information] significantly enhances error detection performance" [abstract]; "such error detectors fail to generalize across datasets" [abstract] — no specific numbers extracted
- **Conclusions.** LLM internal representations encode substantially more truthfulness/correctness information than external behavior reveals, but that information is concentrated in specific tokens and is multifaceted (dataset-specific) rather than a single universal "truthfulness direction"; models can know the right answer internally and still say the wrong one.

> **Why it matters to us.** Matches (A) only, cleanly: a trained classifier on frozen internal representations for correctness/truthfulness. Does not match (B) — it detects/analyzes errors in whichever single answer was generated, it does not rank or select among several sampled candidate answers for the same question. Does not clearly match (C): "truthfulness information is concentrated in specific tokens" is about WHICH tokens carry signal within one answer, not about pooling a mean representation over an answer's tokens the way our probe does — a related but distinct idea (token-localization vs. token-pooling). (D) not addressed. Two findings are directly load-bearing for our project's own honest holes: (1) "error detectors fail to generalize across datasets" is independent textual confirmation of the exact per-benchmark-training requirement we already document ("probe is per-benchmark, needs ~100 labelled in-domain questions"); (2) "LLMs may encode the correct answer, yet consistently generate an incorrect one" is a general argument for WHY reading hidden states can outperform reading generated text at all — the premise our whole verifier approach depends on, stated here for the text-only case.

<small>Read from: abstract-only.</small>

#### The Internal State of an LLM Knows When It's Lying

*Amos Azaria and Tom Mitchell · 2023 · arXiv preprint · [arXiv:2304.13734](https://arxiv.org/abs/2304.13734) · read priority 2*

**In one line.** Trains a classifier on an LLM's hidden-layer activations, while it reads or generates a statement, to predict whether that statement is true or false.

- **Models.** not fully specified beyond "depending on the LLM base model" [abstract]
- **Method.** Supervised classifier on hidden-layer activations captured as the LLM reads/generates a sentence; outputs P(statement is truthful).
- **Datasets.** A set of test sentences, half true / half false [abstract]
- **Experiments.** Compare the trained classifier's truth-detection accuracy against approaches based on the LLM-assigned sentence probability, and examine the confound of sentence length / word frequency on the probability-based approach [abstract]
- **Results.** "our trained classifier achieves an average of 71% to 83% accuracy labeling which sentences are true versus false, depending on the LLM base model" [abstract]
- **Conclusions.** A classifier on internal activations is a more reliable truthfulness signal than the model's own assigned sentence probability, which is confounded by sentence length and word frequency.

> **Why it matters to us.** The closest early precedent for dimension (A) — train a classifier on frozen hidden states to predict correctness/truthfulness of a statement. It differs from us on (B) and (C): it scores generic true/false sentences one at a time rather than ranking N sampled candidate answers against each other for the same question (no best-of-N selection), and it is not framed as pooling specifically over a candidate's own newly generated tokens. It is the conceptual grandparent of the whole "hidden states know correctness" line this category traces to a verifier.

<small>Read from: abstract-only.</small>

#### Understanding Intermediate Layers Using Linear Classifier Probes

*Guillaume Alain and Yoshua Bengio · 2016 · arXiv preprint · [arXiv:1610.01644](https://arxiv.org/abs/1610.01644) · read priority 2*

**In one line.** Coins the term "probing classifier": attach a small classifier, trained independently of the base model, to a frozen intermediate layer to measure what that layer linearly encodes.

- **Models.** Inception v3, ResNet-50 (image classifiers) [abstract]
- **Method.** Linear classifiers ("probes") trained on the frozen activations of each layer; the base network's weights are never updated by probe training.
- **Datasets.** not extracted (abstract names only "popular models", no specific dataset)
- **Experiments.** Probe every layer of Inception v3 and ResNet-50 and measure classification accuracy per layer [abstract]
- **Results.** "the linear separability of features increase[s] monotonically along the depth of the model" [abstract]
- **Conclusions.** Probing classifiers are a general diagnostic tool for what a frozen network's intermediate representation linearly encodes, and can localize where a capability first appears in depth.

> **Why it matters to us.** This is the origin of the word "probe" that our verifier belongs to under the field's own naming. Two departures from the original definition matter: (1) the original probe is strictly LINEAR (a single weight matrix) used purely for diagnosis/interpretability; ours is a small non-linear network (one hidden layer) used operationally, as a component of an answer-producing pipeline (best-of-N selection), not for interpretability. (2) Alain & Bengio probe vision classifiers layer-by-layer at a single, static representation; we probe a generative LLM's hidden state pooled over a candidate's own sampled output tokens. Prior-art relationship: gives us the term "probe" and the general logic (frozen backbone + small trained readout on its activations) that every later paper in this category, including ours, inherits.

<small>Read from: abstract-only.</small>


#### Also in this area (5), in brief

- **VIB-Probe: Detecting and Mitigating Hallucinations in Vision-Language Models via Variational Information Bottleneck** — Feiran Zhang et al. (2026), [arXiv:2601.05547](https://arxiv.org/abs/2601.05547). Trains a probe on VLM attention-head outputs, using the Variational Information Bottleneck principle to filter out visual-linguistic entanglement/noise and isolate the heads that causally drive hallucination, then intervenes on those heads at inference time.<br><small>*For us:* Matches (A) a trained classifier (probe) on frozen VLM internal states. Does not match (B) — there is no candidate selection among several sampled outputs described; detection and mitigation both operate on a single generation trajectory. (C) and (D) are not established in the abstract: it is unclear whether the probe pools over the candidate's own generated tokens or reads a single decision point, and no forward-pass-reuse claim is made. Useful mainly as a contrast in probe INPUT granularity: rather than pooling a mean over hidden states across generated tokens (as our probe and SWIFT/HSRM do), it probes individual attention-head outputs and uses an information-theoretic regularizer to fight entanglement — a different design lever we have not tried.</small>
- **When Correct Decisions Hide Internal Stress: Decision-State Probing in Multimodal Language Models** — Haoran Zhao et al. (2026), [arXiv:2606.08394](https://arxiv.org/abs/2606.08394). Introduces S3E, a forced-choice A/B probing framework that extracts pre-answer decision-state hidden states and shows that even on trials where the model answers CORRECTLY and CONSISTENTLY (both option orders), semantically stressful wrong candidates still cause measurable internal decision-state displacement — correct behavior does not certify a stable internal decision.<br><small>*For us:* Matches (A) only, loosely: hidden states are extracted at a frozen model's pre-answer decision point. Does not match (B) — this is a two-option forced-choice contrast used for a stability DIAGNOSTIC, not a selection mechanism among several sampled free-text candidates. Does not match (C) — the read is at the PRE-answer decision state, not pooled over any candidate's own generated tokens (there is no free-text generation in this design at all). Does not match (D) — not applicable, no best-of-N framing exists here. The genuinely useful transfer to us is methodological rather than architectural: it is a cautionary result for evaluating any hidden-state probe/verifier purely by downstream accuracy — a probe or verifier that looks accurate on correct answers might still be operating on internally "stressed", less-robust representations that a harder distractor could destabilize; this is a different, complementary caution to Hewitt & Liang's control-task worry (this file, strand i) and to CASE's decodability/leakage worry above.</small>
- **TruthPrInt: Mitigating Large Vision-Language Models Object Hallucination Via Latent Truthful-Guided Pre-Intervention** — Jinhao Duan et al. (2025), [arXiv:2503.10602](https://arxiv.org/abs/2503.10602). Finds that LVLM hidden states are high-specificity PER-TOKEN hallucination indicators and that different LVLMs share a common latent "truthful direction", then uses that direction to steer (intervene on) decoding rather than to select among candidates.<br><small>*For us:* Matches (A) reading a frozen(ish) LVLM's internal states for a correctness-adjacent signal (object hallucination rather than open-ended VQA correctness), and is one of the few cards in this category on the VISION-language side rather than text-only. It differs sharply on (B): TruthPrInt's mechanism is an intervention that steers a SINGLE decode, not a selector that ranks/picks among multiple already-generated candidates — there is no best-of-N here. (C)/(D) are per-token, computed inline during the single decode being steered, so they are closer to "free" in spirit but are not pooling over a finished candidate's tokens for a post-hoc score the way our verifier does. Useful mainly as evidence that a shared, transferable "truthful direction" exists in LVLM hidden states across models/datasets — relevant if we ever want to explain why our probe transfers, or fails to transfer, across benchmarks.</small>
- **Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations** — Nick Jiang et al. (2024), [arXiv:2410.02762](https://arxiv.org/abs/2410.02762). Projects a VLM's internal image representations onto its language vocabulary (a "logit lens" style read), finds more confident output probabilities on real than hallucinated objects, and edits the representations (linear orthogonalization against hallucinated-object features) to remove hallucinations while preserving performance.<br><small>*For us:* Only a loose (A) match: it reads internal representations, but via a vocabulary-projection/interpretation technique rather than a trained supervised classifier probe, and it EDITS a single generation rather than reading a probe score. Does not match (B) select-among-candidates (no best-of-N here at all, it intervenes on one generation), (C) pooling over a candidate's own generated tokens (it operates on internal IMAGE feature representations, not generated text), or (D) in any way relevant to our framing. Useful mainly as a different family of "read more out of frozen VLM internals than the text output shows" evidence, and for the general finding that internal confidence on real-vs-hallucinated content exceeds what the decoded text reveals — a VLM-domain echo of the text-only "LLMs know more than they show" result below.</small>
- **Designing and Interpreting Probes with Control Tasks** — John Hewitt and Percy Liang (2019), [arXiv:1909.03368](https://arxiv.org/abs/1909.03368). Introduces "control tasks" (same inputs, random outputs) to test whether a probe's accuracy reflects real structure in the representation, or just the probe's own capacity to memorize the task.<br><small>*For us:* This is a standing methodological gap for our project: we have not run a control-task check on the MLP probe. A control task for us would be training the identical probe architecture to predict a randomly shuffled correct/incorrect label (or a per-question-identity-only signal) and confirming it fails (near-chance AUROC); without that, high probe accuracy is consistent with the probe partly memorizing per-question identity rather than reading a real correctness signal. CASE's "decodability" measure (2608.17124, this category) is a different, leakage-based way of getting at the same worry and is closer to what we would actually want to run.</small>

## 3.4 The Walls: Coverage vs. Selection, Imperfect Verifiers, and Bounded Best-of-N Gains

Best-of-N sampling — draw N candidate answers from a model and pick one — only helps as much as two separate quantities allow: coverage (does a correct answer exist anywhere in the pool of N, measured as pass@k / oracle@N) and selection (can any practical, non-omniscient method actually find it). This category collects the empirical scaling laws for coverage (it keeps climbing, log-linearly, out to tens of thousands of samples), the theory for why selection cannot keep up (imperfect verifiers have an irreducible false-positive rate; reward models over-optimize and true accuracy can fall as N grows; correlated errors within a model family, or within one model's own repeated draws, shrink the effective number of independent 'votes'), and several 2026 papers that formalize the resulting gap with almost exactly the vocabulary this project already uses internally: oracle gap, recoverable mass, and selection/conversion efficiency. For Leo's project specifically, this is the literature background for the two 'walls' the repo already measures — a ~78-81% selection efficiency and a 37-41% coverage wall at N=8 — and for whether either wall is a property of our probe or a field-wide constant that other groups hit too.

#### ★ Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time Collaboration

*Jie Hu · 2026 · arXiv preprint · [arXiv:2607.17531](https://arxiv.org/abs/2607.17531) · read priority 1 · **PDF in `papers/`***

**In one line.** Independently derives essentially our own decomposition of best-of-N/verifier gain into an oracle gap, a coverage term, a conditional-selection-quality term, and a conditional-harm term, and shows gains are bounded first by oracle gap, then by signal fidelity.

- **Models.** Not extracted (framework is applied to whatever policy model generated each benchmark's candidate pool; the base generator/policy LLM is not named in the abstract or fetched sections).
- **Method.** For a fixed candidate pool, decomposes a selector/verifier's net gain into four measurable factors: recoverable mass P(recoverable), where recoverable(x) = reference-wrong(x) AND any@k-correct(x) [html §3.2, Eq.3]; coverage = P(signal defined), i.e. usable verification evidence exists [html §3.3, Eq.4]; conditional selection quality = P(selected passes | recoverable, defined) [html §3.3, Eq.5]; harm rate = P(selected fails | reference passes, defined) [html §3.3, Eq.6]. Overall gain = P(recoverable ∧ defined)·quality − P(reference correct ∧ defined)·harm [html Introduction, Eq.1]. Signal fidelity is measured directly as candidate-level agreement (MCC) between verifier verdicts and official labels.
- **Datasets.** LiveCodeBench, MATH Level-5 hard subjects, GPQA-Diamond [abstract].
- **Experiments.** Compares verifier types with different signal fidelity (public-test verifier MCC 0.825; generated-test verifier MCC 0.248; an LLM selector; a symbolic answer-equivalence selector) across the three benchmarks, reporting net gain plus its four components [abstract; html Tables 3, 6, 7].
- **Results.** LiveCodeBench (Table 3): first-sample baseline 72.33%; public-test verifier 80.47% (+8.14pp) [abstract; html Table 3]; generated-test verifier 75.03% (+2.70pp) with near-zero harm vs the LLM selector's 4.69% harm rate [abstract; html Table 3]; oracle any@5 84.07% (+11.74pp recoverable-mass ceiling) [html Table 3]. MATH Level-5 (Table 6): symbolic answer-equivalence selector +4.67pp over self-consistency; LLM selector −3.20pp (net negative) [html Table 6]. GPQA-Diamond (Table 7): first-sample 47.64%; oracle any@5 50.67% (oracle gap only +3.03pp); 87.54% of pools are answer-identical [abstract]; LLM selector −1.68pp (net negative) [html Table 7].
- **Conclusions.** Training-free collaboration (self-consistency, best-of-N, verifier pipelines) is bounded first by the oracle gap (small and task/model/sampling-config-dependent — e.g. only 3.03% on GPQA-Diamond) and then multiplicatively by signal fidelity; many practical selectors are net-negative once the harm term is counted. Recommends measuring oracle gap, coverage, signal fidelity and harm BEFORE investing in a collaboration mechanism [abstract].

> **Why it matters to us.** This is the closest prior framework to ours: it independently invents essentially the same decomposition (oracle gap / coverage / a selection-quality term / a harm term) that we use internally for our own best-of-8 + hidden-state-probe pipeline. We must cite Hu and NOT claim priority on the decomposition itself, nor on the general claim 'coverage binds before selection' — Hu demonstrates this generally across LiveCodeBench/MATH/GPQA; our contribution is the specific medical-VQA, frozen-hidden-state-probe instantiation, with our own numbers (selection efficiency ~0.78-0.81, coverage wall 37-41% at N=8, selected = oracle@8 × selection_efficiency). Hu's 'harm rate' is the direct analogue of what we would call conditional harm (the probe demoting an already-correct candidate) and should be reported in the same units for comparability. Hu's GPQA result — recoverable mass only 3.03%, oracle gap a joint property of task/model/sampling — is useful outside confirmation that a small oracle gap, not selector skill, is often the real limiter.

<small>Read from: html.</small>

#### ★ When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling

*Yong Yi Bay and Kathleen A. Yearick · 2026 · arXiv preprint · [arXiv:2606.28661](https://arxiv.org/abs/2606.28661) · read priority 1 · **PDF in `papers/`***

**In one line.** New (2026) paper directly on-point for why more sampling stops helping: defines a modal ceiling and a correlation ceiling, both reached within a few dozen draws and both independent of sample budget, and proves self-consistency accuracy can fall toward 0 as coverage rises to 1.

- **Models.** Llama-3-8B-Instruct, Llama-3-70B-Instruct (independent-draw experiments, Table 1); Llama-3.2-1B-Instruct (dependent-draw, Fig.10, using the Beeching et al. release) [html Table 1, Fig.10].
- **Method.** Modal ceiling π_mode = P_q[a*_q = c_q] — the probability the single most-frequent sampled answer equals the correct one; provably independent of the sample budget n [html Eq.13]. Correlation ceiling = 1/ρ, the limit of the effective number of samples n_eff = n / [1+(n−1)ρ] as n→∞, where ρ is the intraclass (within-problem) correlation between repeated draws [html Eq.7, Corollary 1]. Proves (Corollary 3) that on problems whose modal answer is wrong, self-consistency accuracy can fall to 0 as n→∞ even as coverage rises to 1.
- **Datasets.** GSM8K, MATH, MATH-500 [html Table 1, Fig.10].
- **Experiments.** 10^4 samples/problem for Llama-3-8B/70B on GSM8K and MATH (independent draws, Table 1); 256 attempts/problem averaged over 5 sessions for Llama-3.2-1B on MATH-500 (dependent draws, Fig.10).
- **Results.** GSM8K (Llama-3-8B): estimated intraclass correlation ρ̂_b ≈ 0.47 → correlation ceiling 1/ρ̂_b ≈ 2.1 effective samples [html Table 1]. GSM8K: coverage reaches 1.00 while self-consistency plateaus at 0.87 — identifiability gap ≈0.13 ("an eighth of problems") [html Fig.9]. MATH-500 (Llama-3.2-1B, dependent draws): self-consistency plateaus at 0.45 by about n=64 (median ~13 distinct answers per problem); coverage climbs to 0.88 while plurality selection stalls at π_mode=0.45 — identifiability gap ≈0.43 [html Fig.10].
- **Conclusions.** The modal ceiling and correlation ceiling are reached within a few dozen draws (or sooner) and do NOT improve with more sampling; on problems whose plurality answer is wrong, self-consistency accuracy can degrade toward 0 as coverage rises to 1, i.e. more sampling actively hurts a vote-based selector even while it helps the oracle. Proposes 'effective number of samples' (n_eff, computable from any sampling run's own intraclass correlation) as a single stopping-rule diagnostic [abstract; html Eq.7].

> **Why it matters to us.** The most direct mechanistic account we have found for why our own best-of-8 selection efficiency sits at 78-81% rather than 100%, and it is new to this project. Their framing explains the shape as a property of within-problem answer correlation ρ (how similar repeated draws are) rather than of the selector's skill — worth checking whether our probe's mis-selections cluster the same way (e.g. high ρ on medical questions with a common plausible-but-wrong phrasing). Gives us vocabulary ("identifiability gap") for exactly the quantity we call the gap between coverage-wall-adjusted oracle@8 and what our probe converts, and a candidate diagnostic (n_eff) to justify fixing N=8 rather than sweeping N further, since correlation ceilings are typically reached in "a few dozen" draws — close to our own N=8.

<small>Read from: html.</small>

#### ★ Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

*Bradley Brown et al. · 2024 · arXiv preprint · [arXiv:2407.21787](https://arxiv.org/abs/2407.21787) · read priority 1 · **PDF in `papers/`***

**In one line.** Foundational empirical demonstration that coverage (pass@k / oracle accuracy) scales log-linearly over four+ orders of magnitude in sample count, while majority-vote/reward-model selection plateaus far below coverage once no automatic verifier exists.

- **Models.** Llama-3-8B-Instruct, Gemma-2B, Pythia-160M, DeepSeek-Coder-V2-Instruct [html §2.2, §4.1].
- **Method.** Repeated sampling: draw k candidates independently from a fixed model and define coverage = fraction of problems solved by ANY of the k samples (i.e. pass@k under an oracle selector). Models coverage(k) with an exponentiated power law c ≈ exp(a·k^b) [html Eq.3, §3.1]. In domains without automatic verifiers, compares coverage to what majority voting or a trained reward model can actually select.
- **Datasets.** SWE-bench Lite, MATH, CodeContests [html §2.1–§4.1]; also GSM8K-family tasks. Not extracted: full benchmark list beyond these.
- **Experiments.** Sweeps k up to 10,000+ samples per problem and measures coverage vs. the accuracy actually achieved by majority voting / reward-model selection, on both automatically-verifiable domains (code, with unit tests) and non-automatically-verifiable domains.
- **Results.** SWE-bench Lite (DeepSeek-Coder-V2-Instruct): 15.9% at k=1 → 56% at k=250, beating the prior single-sample SOTA of 43% [abstract; html §2.1]. MATH (Llama-3-8B-Instruct): coverage 82.9% at k=100 → 98.44% at k=10,000, while majority voting/reward-model selection plateau near k=100 and rise only from 40.50% to 41.41% over the same range — coverage gains +15.54pp while selection gains only +0.91pp [html §4.1, Fig.7]. CodeContests (Gemma-2B): 0.02% at pass@1 → 7.1% at pass@10k [html §2.2, Fig.3]. Pythia-160M on MATH: 0.27% pass@1 → 57% pass@10k [html §2.2].
- **Conclusions.** Coverage keeps climbing log-linearly (power law) across many orders of magnitude of samples and, with a perfect/automatic verifier, converts directly into accuracy (SWE-bench). Without an automatic verifier, practical selectors (majority vote, reward model) plateau far below coverage — the gap between what a large sample pool could answer and what any practical selector can pick widens as N grows [abstract; html §4.1].

> **Why it matters to us.** The foundational shape our own selection-efficiency ceiling reproduces: their MATH data shows coverage gaining +15.5pp from k=100→10,000 while selection gains only +0.9pp — the same qualitative widening gap as our own ~78-81% selection efficiency plateauing well under oracle@8. Their finding that a reward model does no better than plain majority voting on MATH is a caution for us: zero-shot/generic reward-style scoring does not automatically close the selection gap, which is consistent with why our MLP probe needs ~100 labelled in-domain questions per benchmark rather than working zero-shot. We should cite their power-law coverage form when describing how our own coverage wall (37-41% no correct candidate at N=8) would likely move if N were pushed higher.

<small>Read from: html.</small>

#### ★ The Limits of Inference Scaling Through Resampling

*Benedikt Stroebl et al. · 2024 · arXiv preprint · [arXiv:2411.17501](https://arxiv.org/abs/2411.17501) · read priority 1 · **PDF in `papers/`***

**In one line.** Proves that an imperfect verifier's non-zero false-positive rate imposes a hard upper bound on resampling-based inference scaling that no amount of additional compute can lift, and finds compute-optimal sample counts are typically single digits.

- **Models.** Four (unspecified individually beyond a cost-benefit sweep, html §4/Fig.4) code-generation models evaluated against unit-test verifiers.
- **Method.** Formal argument: if a strong model's single-sample accuracy already exceeds a weak model's accuracy conditioned on passing the (imperfect) verifier/unit tests, the weak model cannot catch up via more resampling, because resampling cannot lower the verifier's false-positive rate [abstract; html 'Upper Accuracy Limit']. Empirically correlates single-sample accuracy with false-positive rate, and sweeps a cost-benefit ratio to find the compute-optimal resampling budget K.
- **Datasets.** HumanEval(+), MBPP(+) [abstract].
- **Experiments.** Correlates each model's single-sample accuracy with its verifier false-positive rate (Fig.3, shown graphically, no tabulated percentages recovered); sweeps a cost-benefit ratio to find optimal K (Fig.4).
- **Results.** "Optimal number of samples is K≤5 for all four models" at a cost-benefit ratio of 4 [html §4, Fig.4 caption]; abstract states optimal sampling attempts are "often fewer than 10" [abstract]. Weaker models show a higher false-positive rate than stronger models on HumanEval+/MBPP+ (relationship shown graphically; specific percentages not extracted) [html Fig.3].
- **Conclusions.** An imperfect verifier caps resampling's achievable accuracy regardless of compute budget; the false-positive rate, not the sampling budget, is the binding constraint, so scaling curves bend downward once false positives outweigh marginal benefit — typically at single-digit N [abstract; html §4].

> **Why it matters to us.** Gives the formal reason our own probe-based selector cannot be scaled to arbitrarily large N for free: any imperfect verifier (including our BCE-trained hidden-state probe) has a floor mis-ranking/false-positive rate, so its achievable best-of-N accuracy is capped below the true oracle@N regardless of N — the theoretical counterpart to our measured 78-81% selection-efficiency ceiling. Their K≤5-optimal finding is a useful cross-check for our choice of N=8: it predicts we may already be near or past the point where more samples buy little without first improving the verifier itself (lowering its false-positive/mis-ranking rate), which matches our own finding that training (not model size) is the active ingredient in verification.

<small>Read from: html.</small>

#### Best-of-Evidence: Best-of-N Selection under Partial Verification

*Cenwei Zhang et al. · 2026 · arXiv preprint · [arXiv:2607.20950](https://arxiv.org/abs/2607.20950) · read priority 2*

**In one line.** Extends best-of-N to partial verification (only a finding/span/relation is checkable, not the whole response) via a signed candidate-factor graph and a budgeted evidence controller; on four medical VQA settings it rescues some BoN failures but is still capped by evidence-channel quality and candidate-pool coverage.

- **Models.** Not extracted (abstract does not name the underlying VLM/generator).
- **Method.** Best-of-Evidence (BoE): keeps the BoN candidate pool fixed, represents reusable claims as a signed candidate-factor graph, and allocates a limited query budget to evidence actions that could flip the final choice; the zero-budget case recovers plain BoN [abstract]. Proves a residual-evidence-capacity limit on any evidence-driven improvement and an O(log K) vs. Θ(K) query-count separation in a factor-code model.
- **Datasets.** "Four medical VQA settings" (common-ledger experiments) [abstract]; specific dataset names not extracted.
- **Experiments.** Compares BoE to fixed-pool BoN selection under partial (not whole-response) verifiability across the four medical VQA settings [abstract].
- **Results.** "BoE can improve fixed-pool selection and rescue some BoN failures when evidence is reliable, contrastive, and decision-relevant, while also revealing the channel-quality and candidate-generation limits that prevent universal gains" [abstract]; no specific accuracy numbers extracted.
- **Conclusions.** Partial, per-claim verification can rescue some best-of-N failures beyond what whole-response rerankers achieve, but remains capped by evidence-channel quality and by whether the candidate pool contains a rescuable answer at all — i.e. still bound by coverage [abstract].

> **Why it matters to us.** The closest prior-art paper that is specifically medical VQA + best-of-N — worth flagging even though non-core. Its "residual evidence capacity limits any evidence-driven improvement" result is the same coverage-wall logic we rely on. Unlike BoE, our probe uses only the generator's own hidden states (no external per-claim evidence retrieval), so we are a cheaper, whole-response verifier operating in the same coverage-bound regime BoE formalizes with partial verification.

<small>Read from: abstract-only.</small>

#### Agentic Systems as Boosting Weak Reasoning Models

*Varun Sunkaraneni et al. · 2026 · arXiv preprint · [arXiv:2605.14163](https://arxiv.org/abs/2605.14163) · read priority 2*

**In one line.** Separates proposal coverage from local identifiability, shows a critic-comparator orchestration over a weak model nearly reaches a much stronger model's accuracy, but the remaining failures are mostly proposal-coverage failures the selector cannot fix.

- **Models.** GPT-5.4 nano (proposer, single-sample and best-of-k); compared to standalone Gemini 3 Pro and Claude Opus 4.5 Thinking [abstract].
- **Method.** Verifier-backed committee search: separates proposal coverage (does the sample pool contain a correct answer at all), local identifiability (can a critic/comparator recover it without the hidden verifier), progress, and diversity. Proves coverage is amplifiable by repeated sampling alone, but useful critics/comparators additionally require a local-soundness signal (execution, proof-checking, tests, constraint solving) [abstract].
- **Datasets.** SWE-bench Verified [abstract].
- **Experiments.** Compares single-sample GPT-5.4 nano vs. k=8 critic-comparator orchestration vs. oracle best-of-8 vs. standalone stronger models [abstract].
- **Results.** Single GPT-5.4 nano proposal solves 67.0% of tasks [abstract]; critic-comparator orchestration with k=8 proposals reaches 76.4%, matching standalone Gemini 3 Pro and Claude Opus 4.5 Thinking [abstract]; oracle best-of-8 upper bound is 79.0% [abstract].
- **Conclusions.** Most of the gap between a weak model's single-sample accuracy and a strong model's accuracy is closeable by sampling + a soundness-gated critic/comparator (67.0%→76.4% vs. a 79.0% oracle), but the remaining gap is mostly "proposal-coverage failures" — shared blind spots no amount of better selection can fix [abstract].

> **Why it matters to us.** A close cousin of our own oracle-of-N / selection-efficiency setup, for agentic code tasks with execution-checkable local signals rather than free-text medical answers. Their own numbers imply a selection efficiency of (76.4−67.0)/(79.0−67.0) = 78.3% — remarkably close to our measured 78-81%, an independent cross-domain confirmation that roughly 4/5 of the oracle gap is a realistic ceiling for a trained selector even with a much richer per-claim verification signal than ours. Their "remaining failures are mostly proposal-coverage failures" is exactly our own coverage wall (37-41% no correct candidate at N=8), stated for a different domain.

<small>Read from: abstract-only.</small>

#### When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models

*Josef Chen · 2026 · arXiv preprint · [arXiv:2606.27288](https://arxiv.org/abs/2606.27288) · read priority 2*

**In one line.** Defines beta, the rate at which every model in a pool is wrong on the same query simultaneously, as a hard ceiling (1-beta) on any single-output routing/voting/cascade/fusion policy, and shows the standard pairwise-correlation diagnostic cannot identify beta.

- **Models.** 67 frontier models from 21 providers [abstract].
- **Method.** Defines beta = P(all models wrong on x); proves accuracy of any single-model-output policy is ≤ 1−beta; shows pairwise error correlation rho cannot identify beta (different error laws can share marginals/pairwise-correlations but differ in the all-wrong tail); gives a Clopper-Pearson finite-sample certificate on beta usable before training a router; fits a tetrachoric-calibrated single-factor Gaussian-copula model of joint errors [abstract].
- **Datasets.** Open-ended mathematics, execution-graded code, and GPQA-Diamond re-asked in free-response (not multiple-choice) form [abstract].
- **Experiments.** Measures observed beta vs. the beta predicted by the calibrated Gaussian-copula model across the 67-model pool; re-asks GPQA-Diamond in free-response form with a 5-judge panel [abstract].
- **Results.** Open-ended math: observed beta 0.052 vs. 0.023 predicted by the full 67-model Gaussian copula (~2.5x underpricing, 90% CI [1.7,3.4], k=17) [abstract]. Execution-graded code: beta 0.079 [abstract]. Free-response GPQA-Diamond: beta 0.127, 5-judge kappa 0.73-0.92 [abstract]. "At matched quality, low-rho heterogeneous ensembles beat high-rho Self-MoA, but... combining models rarely beats the single best model without a strong query-level routing signal" [abstract].
- **Conclusions.** The standard pairwise-correlation diagnostic systematically underestimates the true co-failure (all-wrong) ceiling that caps multi-model system accuracy; switching a benchmark from MCQ to free-response format reopens the co-failure tail — co-failure lives in answer format as much as in subject matter [abstract].

> **Why it matters to us.** A routing/ensemble-level analogue of our own coverage wall, expressed as "the rate every model is simultaneously wrong" (beta) rather than "the rate no sampled candidate is correct." Their MCQ-vs-free-response result (beta jumps when the format changes) directly parallels our own finding that answer format determines whether routing/verification signals work at all (AUROC ~0.6 MCQ vs. ~0.87 free text) — independent cross-domain evidence that open-ended answer format changes the achievable ceiling, not just detection difficulty. Gives us the term "co-failure ceiling"/beta as the multi-model analogue of our single-model coverage wall.

<small>Read from: abstract-only.</small>

#### Hidden Clones: Exposing and Fixing Family Bias in Vision-Language Model Ensembles

*Zacharie Bugaud · 2026 · arXiv preprint · [arXiv:2603.17111](https://arxiv.org/abs/2603.17111) · read priority 2*

**In one line.** Same-family VLMs share correlated errors that shrink an ensemble's effective dimensionality to just 2.5-3.6 independent voters and create a "Misleading tier" where correlated majority error destroys accuracy to 0% despite the best model being correct; proposes three family-aware fixes.

- **Models.** 17 VLMs from 8 families [abstract].
- **Method.** Measures family-correlated error structure across an ensemble; proposes Hierarchical Family Voting (HFV: aggregate within family before voting across families), QualRCCV (training-free weighting by calibration, family quality, and inverse family size), and Learned Candidate Scoring (LCS: a cross-validated classifier reranking candidates using support breadth, family diversity, and model quality) [abstract].
- **Datasets.** VQAv2, TextVQA, GQA [abstract].
- **Experiments.** Quantifies effective ensemble dimensionality and the "Misleading tier" (fraction of questions where correlated majority error zeroes out accuracy despite the best model being right); compares HFV/QualRCCV/LCS to calibrated voting [abstract].
- **Results.** Family-correlated errors reduce effective ensemble dimensionality to 2.5-3.6 independent voters [abstract]. Misleading tier = 1.5-6.5% of questions, where correlated majority error destroys accuracy to 0% [abstract]. HFV recovers +18-26pp on the Misleading tier [abstract]. QualRCCV is first to beat calibrated voting on all three benchmarks (p<0.05) [abstract]. LCS: +0.68% VQAv2, +0.61% TextVQA, +2.45% GQA (all significant), never degrades any benchmark [abstract]. LCS reaches 87.83% on VQAv2 test-standard (EvalAI) with 12 models [abstract].
- **Conclusions.** Naively pooling same-family VLMs overstates ensemble diversity because their errors are correlated (effectively only ~3 independent voters, not 17); family-aware weighting/voting/reranking recovers much of the lost accuracy on the worst-affected question tier without hurting other benchmarks [abstract].

> **Why it matters to us.** The concrete mechanism behind 'ceilings from correlated errors' in this category, and directly relevant to our own setup: our N=8 samples are drawn from the SAME Lingshu-7B checkpoint, not 8 independent models, so if those draws share correlated failure modes (the same wrong reasoning pattern repeated across 'independent' samples), the effective diversity of our pool could be much lower than 8 — a plausible contributor to why our coverage wall sits at 37-41% rather than shrinking toward 0. Also directly relevant to our peer-architecture replication (Qwen2.5-VL-7B, MedGemma-4b-it): if those share training-data lineage with Lingshu, our cross-model transfer claims should be read through this correlated-error lens.

<small>Read from: abstract-only.</small>

#### Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment

*Audrey Huang et al. · 2025 · arXiv preprint · [arXiv:2503.21878](https://arxiv.org/abs/2503.21878) · read priority 2*

**In one line.** Proves Best-of-N is optimal only under stringent coverage conditions on the base policy and provably reward-hacks for large N; introduces InferenceTimePessimism, a rejection-sampling algorithm proven optimal and "scaling-monotonic" (does not degrade with N).

- **Models.** Not extracted individually; theoretical framework validated on "a variety of tasks and models" [abstract].
- **Method.** Formalizes inference-time alignment (improving response quality given an imperfect reward model) and analyzes Best-of-N theoretically; introduces InferenceTimePessimism, which applies the principle of pessimism-under-uncertainty via rejection sampling [abstract].
- **Datasets.** Not extracted (abstract does not name specific benchmarks).
- **Experiments.** Theoretical analysis of Best-of-N under "stringent" vs. more realistic coverage assumptions; empirical evaluation of InferenceTimePessimism vs. Best-of-N "across a variety of tasks and models" [abstract].
- **Results.** Not extracted — abstract states qualitative claims (BoN "provably suffers from reward hacking when N is large"; InferenceTimePessimism's performance "is optimal and does not degrade with N") but gives no specific numeric results [abstract].
- **Conclusions.** Best-of-N with an ideal N is optimal only under stringent coverage of high-quality responses by the base policy; under realistic/looser coverage it fails to give tight guarantees and reward-hacks for large N. InferenceTimePessimism is proven optimal and scaling-monotonic [abstract].

> **Why it matters to us.** The formal theory behind why 'coverage binds before selection' in our domain: BoN's optimality is proven conditional on the base policy's coverage over high-quality responses — matching our empirical coverage wall (37-41% of questions have no correct candidate at N=8, a failure no selector, however good, can fix). Their proof that BoN reward-hacks for large N is the theoretical counterpart to Khalaf et al. (2506.19248) and to the modal-ceiling account in Bay & Yearick (2606.28661); together these justify not scaling N past 8 without first improving generator coverage.

<small>Read from: abstract-only.</small>

#### Inference-Time Reward Hacking in Large Language Models

*Hadi Khalaf et al. · 2025 · NeurIPS 2025 (Spotlight) · [arXiv:2506.19248](https://arxiv.org/abs/2506.19248) · read priority 2*

**In one line.** Proves the characteristic reward-hacking pattern (true reward rises then falls as N grows) is an INEVITABLE property of a broad class of inference-time selection mechanisms including Best-of-N, and introduces Best-of-Poisson (BoP) plus a HedgeTune algorithm to mitigate it.

- **Models.** Generic (math, reasoning, and human-preference setups); specific model names not extracted from the abstract.
- **Method.** Characterizes reward hacking under Best-of-n (BoN) and Soft Best-of-n (SBoN); introduces Best-of-Poisson (BoP), a near-exact approximation of the optimal reward-KL-divergence policy at inference time, and HedgeTune, an algorithm to find the optimal inference-time hedging parameter [abstract].
- **Datasets.** Math, reasoning, and human-preference benchmarks; specific names not extracted [abstract].
- **Experiments.** Measures true reward vs. proxy reward as N grows under BoN/SBoN/BoP; evaluates HedgeTune's reward-distortion tradeoff [abstract].
- **Results.** "The characteristic pattern of hacking as observed in practice (where the true reward first increases before declining) is an inevitable property of a broad class of inference-time mechanisms, including BoN and BoP" [abstract]. "Hedging mitigates reward hacking and achieves superior reward-distortion tradeoffs" [abstract]. No specific numeric values extracted (abstract-only).
- **Conclusions.** The rise-then-fall shape of accuracy/reward vs. N under best-of-N selection is not an artifact of a particular bad reward model — it is mathematically guaranteed for a broad class of selection mechanisms once the reward is imperfect; deliberate hedging (not fully maximizing the proxy reward) provably helps [abstract].

> **Why it matters to us.** The theory paper to cite for the specific SHAPE of the accuracy-vs-N curve this category teaches (rises then falls). Our own probe-based selector could in principle exhibit this same non-monotonicity if pushed past N=8 with a fixed probe, since the probe is exactly an imperfect proxy reward in their framework. HedgeTune's idea of deliberately not fully trusting the proxy score is conceptually related to why our project uses a Wilson-lower-bound-style certified veto elsewhere (keep the cheap answer unless the strong signal clears a conservative bar) rather than naively trusting the raw score.

<small>Read from: abstract-only.</small>

#### Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems

*Lingjiao Chen et al. · 2024 · arXiv preprint · [arXiv:2403.02419](https://arxiv.org/abs/2403.02419) · read priority 2*

**In one line.** Shows majority-vote (Vote) and Filter-Vote compound-system accuracy can rise then FALL as the number of LLM calls grows, because a task's easy and hard queries respond oppositely, and fits an analytic scaling model to predict the optimal number of calls.

- **Models.** Generic LLM calls in a task-agnostic theoretical + empirical framework [abstract].
- **Method.** Studies Vote (majority voting over independently-sampled LLM answers) and Filter-Vote (an LLM filter applied before voting); derives why performance is non-monotone in the number of calls and fits an analytical scaling law from a small number of samples to predict the optimum [abstract].
- **Datasets.** "Multiple language tasks" [abstract]; not extracted (specific benchmark names not given in the abstract).
- **Experiments.** Varies the number of LM calls and measures Vote/Filter-Vote accuracy; attributes non-monotonicity to a mixture of "easy" queries (more calls help) and "hard" queries (more calls hurt) [abstract].
- **Results.** "The performance of both Vote and Filter-Vote can first increase but then decrease as a function of the number of LM calls" [abstract]. Specific accuracy numbers: not extracted (abstract-only).
- **Conclusions.** More LLM calls is not always better; an analytical scaling model fit from few samples predicts the optimal number of calls per task [abstract].

> **Why it matters to us.** Directly names the phenomenon this category teaches — accuracy can fall as sampling/aggregation scales up — for plain majority voting rather than a trained verifier. A second, independent account (alongside Bay & Yearick 2606.28661) of why more sampling is not automatically better, and a term ("easy vs. hard query mixture") for why the effect is population-level rather than uniform, matching our own observation that OOD-ness does not predict where our probe helps.

<small>Read from: abstract-only.</small>

#### Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models

*Yuda Song et al. · 2024 · ICLR 2025 · [arXiv:2412.02674](https://arxiv.org/abs/2412.02674) · read priority 2*

**In one line.** Formalizes LLM self-improvement (generate, self-verify, filter/reweight, distill) around a "generation-verification gap" that scales monotonically with pretraining compute.

- **Models.** "Various model families" [abstract]; not extracted (specific model names not given in the abstract).
- **Method.** Mathematical formulation of self-improvement centered on the generation-verification gap (how much better a model is at judging its own outputs than at producing them); studies an iterative self-improvement procedure and when it is/is not possible [abstract].
- **Datasets.** Not extracted (abstract does not name specific benchmarks).
- **Experiments.** Measures how a variant of the generation-verification gap scales with model pretraining FLOPs; tests conditions under which iterative self-improvement succeeds [abstract].
- **Results.** "A variant of the generation-verification gap scales monotonically with the model pre-training flops" [abstract]. No specific numeric values extracted (abstract-only).
- **Conclusions.** Self-improvement is bounded by the generation-verification gap; the gap (verification relative to generation) shrinks as pretraining compute grows [abstract].

> **Why it matters to us.** The "generation-verification gap" is the same underlying quantity as our oracle gap / recoverable mass, but framed for a model verifying/distilling its OWN outputs across training rather than a frozen probe verifying candidates at inference time. Useful term for explaining to Leo why a bigger/better-pretrained generator (e.g. Lingshu-32B) might have an intrinsically smaller verification-worth gap than the 7B we probe — a testable prediction that our best-of-N + probe approach should matter progressively less as base-model scale grows.

<small>Read from: abstract-only.</small>


#### Also in this area (4), in brief

- **How Much of the Routing Gap Is Real? Decomposing the Router-to-Oracle Gap into Reproducible Specialist Advantage and Single-Draw Label Noise** — Teng-Ruei Chen (2026), [arXiv:2607.03436](https://arxiv.org/abs/2607.03436). Shows 12-36% of a commonly-reported router-to-oracle accuracy gap is single-draw label noise (an oracle built from one stochastic sample per model is not reproducible), not real recoverable specialist advantage, and proves this noise floor is closed by resampling the committed model but by no single-commit router.<br><small>*For us:* A methodological warning directly applicable to how we measure our own oracle@8: if an oracle pool were built from a single stochastic draw per configuration rather than resampled, part of the measured oracle gap / selection-efficiency shortfall could be single-draw noise rather than genuine selection failure. Our method already resamples N=8 times per question by construction, which largely addresses this for our oracle@8 estimate itself, but it is a reason for caution whenever oracle@8 is compared against a single-draw baseline (e.g. greedy) without accounting for the repo's own ±0.008 open-text reproducibility noise floor.</small>
- **RouteGuard: Certifying Routing Gain in LLM Multi-Agent Systems When Complementarity Is Not Enough** — Anchen Sun and Kaiqi Yang (2026), [arXiv:2608.07583](https://arxiv.org/abs/2608.07583). A pre-deployment certification protocol for whether a routing/multi-agent gain is real: routing gain decomposes as G = pi × Delta_E, and a benchmark "win" can evaporate once resampled at the correct unit (workload cluster instead of individual prompt).<br><small>*For us:* A cautionary methodological parallel for our own headline claims: our repo's own notes record a very similar correction (a router "WIN" that turned out to be an anticonservative CI of our own, per TRANSFER_WALL_2026-08-21.md), and our leave-one-cell-out check on the vs-direct delta (PMC_VQA load-bearing) is exactly the kind of "does the gain rest on a handful of cells" audit RouteGuard formalizes. Worth citing as external validation that reporting leave-one-cell-out deltas and CIs, rather than a single pooled accuracy number, is the field-recommended standard.</small>
- **The Routing Plateau: Understanding and Breaking the Accuracy Limits of LLM Routers** — Yifan Lu et al. (2026), [arXiv:2606.07587](https://arxiv.org/abs/2606.07587). 21 routing methods across 5 benchmarks converge to a narrow "routing plateau" far below the oracle router, caused by a predictability bottleneck: routers mostly learn coarse average model-performance trends, not query-specific signal.<br><small>*For us:* A routing-level version of our own selection-efficiency ceiling — 21 different routing designs converging to nearly the same accuracy is the routing-literature analogue of our observation that training-free selectors sit at the random-pick floor while only a trained probe breaks it. Independent evidence that the bottleneck across this whole family of problems (routing, best-of-N selection, verification) is information (can the signal even distinguish hard cases), not modeling cleverness. Useful citation for "why not just try another gate design": many different designs plateau at the same ceiling.</small>
- **Best-of-Majority: Minimax-Optimal Strategy for Pass@$k$ Inference Scaling** — Qiwei Di et al. (2025), [arXiv:2510.03199](https://arxiv.org/abs/2510.03199). Proves neither majority voting nor plain Best-of-N scales well with Pass@k, and proposes Best-of-Majority (BoM) — restrict to high-frequency candidates, then rerank by reward — which is minimax-optimal and, unlike BoN, does not degrade as sample budget N grows.<br><small>*For us:* A concrete alternative selector design worth naming when explaining how our probe differs from both plain self-consistency and plain BoN: BoM's "restrict to frequent answers, then rerank" is a hybrid we have not tried and could motivate a future ablation, since our measured selection efficiency (78-81%) is for pure trained-probe reranking only. Also introduces the term "coverage coefficient C*", used with a different but related meaning elsewhere in the routing/verifier literature (e.g. RouteGuard's conditional-regret functional).</small>

## 3.5 Training-free selection: self-consistency, majority vote, minimum Bayes risk, consensus, and logit-based scores

Training-free selection covers methods that pick (or fuse) a final answer from multiple sampled generations with no additional trained scoring model: majority-vote self-consistency and its extensions to free-form text (universal self-consistency's LLM-as-judge, semantic-entropy clustering, embedding/representation-space agreement, Minimum Bayes Risk decoding, game-theoretic equilibrium decoding), adaptive and early-stopping variants that cut the sampling budget, logit-only scores like self-certainty that need no extra model call at all, and pairwise/listwise LLM rerankers. This is the baseline family a reviewer will expect beside our trained hidden-state probe: it establishes what accuracy is achievable from consensus or the model's own confidence alone, and where that runs out. On our open-ended medical VQA benchmarks, exact-match self-consistency sits at the random-pick floor and does not improve from N=2 to N=16 even though oracle@N climbs from 0.156 to 0.418, and a training-free answer-prior baseline (string frequency) loses to plain greedy decoding on GEMeX — so this category also documents where the free lunch runs out for us specifically. Two 2026 papers (agreement in representation space, and latent self-consistency) are the training-free/near-training-free competitors closest to our probe's mechanism, since both replace exact-match voting with a geometric or embedding notion of agreement; their cards spell out exactly how they differ from a trained correctness probe.

#### ★ Wasserstein Equilibrium Decoding for Reliable Medical Visual Question Answering

*Luca Hagen et al. · 2026 · arXiv preprint · [arXiv:2605.18313](https://arxiv.org/abs/2605.18313) · read priority 1 · **PDF in `papers/`***

**In one line.** Extends game-theoretic Bayesian Decoding Game equilibrium search to open-ended medical VQA with a Wasserstein/optimal-transport stopping criterion over a biomedical embedding space, beating greedy decoding on VQA-RAD and PathVQA with small VLMs — the closest published neighbour to our best-of-N probe.

- **Models.** Qwen3-VL-2B/4B/8B, Gemma-3-4B, MedGemma-4B (domain-specialised Gemma-3-4B) [html]
- **Method.** BDG-W: a multistage Bayesian Decoding Game (generator vs. verifier signalling game) run to equilibrium, but convergence is tested with a Wasserstein-1 (optimal-transport) distance over a biomedical-concept embedding space (SapBERT) instead of classic BDG's lexical order-match/sigma-separation criterion, so near-synonymous candidates ('liver' vs 'hepatic region') are treated as agreeing and the game can stop earlier [html, §method].
- **Datasets.** VQA-RAD (315 radiology images, 3,515 QA pairs) and PathVQA (4,998 pathology images, 32,799 QA pairs), open-ended subsets excluding yes/no questions [html].
- **Experiments.** Compares greedy, self-contrastive decoding (SCD), a verifier-only baseline, classic BDG, and BDG-W (Wasserstein) across the 5 VLMs on both datasets; judge-based accuracy and convergence-iteration counts [html, §3, Table 2/3].
- **Results.** BDG-W improves Qwen3-VL-2B judge accuracy on VQA-RAD by +3.5 percentage points over greedy (p<0.01) [html, Table 3]. Gemma-3-4B with BDG (classic or Wasserstein) exceeds MedGemma-4B greedy on PathVQA despite no domain fine-tuning [html]. Wasserstein criterion cuts average convergence iterations ~20% vs classic BDG, e.g. Qwen-2B 27.46±1.52 vs 32.68±0.95, Gemma-4B 16.12±1.68 vs 22.18±0.53 iterations [html, Table 2].
- **Conclusions.** Semantic-aware equilibrium search is a training-free way to squeeze extra open-ended medical-VQA accuracy out of small VLMs, and it converges cheaper than lexical BDG without losing accuracy [html].

> **Why it matters to us.** The single closest published training-free competitor to our arm: same task (open-ended medical VQA, VQA-RAD/PathVQA overlap our benchmark list), same 2-8B model scale, no training, and it explicitly rejects lexical/exact-match agreement in favour of semantic embedding space — parallel to why plain self-consistency sits at our random-pick floor. It is a discriminative-game method, not a best-of-N pointwise scorer: it needs an iterative generator/verifier signalling loop per question rather than one forward pass per candidate, so it is far more expensive per question than our probe, and it never trains anything (our MLP probe is BCE-trained on ~100 in-domain labels). We should benchmark against it or at least cite it as 'nearest training-free result' and note our probe's per-question cost is O(N) hidden-state reads vs BDG-W's O(iterations) verifier calls.

<small>Read from: html.</small>

#### ★ Agreement in Representation Space for Open-Ended Self-Consistency

*Paula Ontalvilla et al. · 2026 · arXiv preprint · [arXiv:2606.12003](https://arxiv.org/abs/2606.12003) · read priority 1 · **PDF in `papers/`***

**In one line.** Reframes open-ended self-consistency as a geometric property: cluster sampled generations in embedding space and return the one closest to the dominant cluster's centroid (Embedding-Based Agreement, EBA), beating USC/random-selection baselines on code, math, and summarization.

- **Models.** Llama 3.1-8B, Llama 70B, Qwen 3-8B, Qwen 32B; embeddings from Qwen-Embedding-8B (also tested: Gemma embeddings, native hidden representations) [html].
- **Method.** Sample N generations, embed each, run agglomerative clustering (average linkage, cosine distance) with cluster count chosen via silhouette analysis, then select the sample closest to the centroid of the dominant cluster — no training, no learned classifier [html, §3.2].
- **Datasets.** HumanEval (code), MATH500 (math reasoning), CNN/DailyMail (summarization) [html].
- **Experiments.** Compares EBA vs random selection, vs USC, vs self-certainty-style scoring, across sample sizes up to N=256; ablation swapping in raw distance-to-global-centroid instead of the clustering step (§6.2) [html].
- **Results.** MATH500, Llama 8B, N=256 (Table 2): EBA 63.85±1.05 vs random 52.77 (+11.08 pts) [html]. HumanEval, Qwen 8B, N=256 (Table 1): EBA 88.11±1.31 vs random 83.45 (+4.66) [html]. CNN/DM, Llama 8B, N=256 (Table 3): EBA 30.02±0.10 vs random 29.20 (+0.82) [html]. Table 9 (Qwen 8B, N=256): nearest-centroid 85.31 vs farthest-from-centroid 74.73 vs full EBA 89.85±0.51, i.e. ~20-point spread between central and peripheral samples [html, §6.2].
- **Conclusions.** Consistency in open-ended generation can be read directly off the geometry of the embedding space (central = higher quality, peripheral = lower quality) without any exact-match canonicalisation or extra LLM judge call, and this holds across model families and embedding choices [html].

> **Why it matters to us.** This is the training-free method structurally closest to our approach: both score/select candidates using vector representations rather than string agreement. The key difference is where the signal comes from and what it optimizes — EBA measures mutual agreement among candidates in an off-the-shelf general-purpose embedding space (unsupervised clustering, no labels), whereas our probe is a BCE-trained classifier reading the generator's OWN hidden states against ground-truth correctness labels, so it can in principle detect a candidate that is confidently, consistently wrong (agreement without correctness) — a known failure mode of any consensus-based method. EBA is the paper we should cite as 'what a training-free version of our idea looks like' and as a natural additional baseline (clustering the mean hidden states we already tap, unsupervised) if we want an even-cheaper-than-probe comparison point.

<small>Read from: html.</small>

#### Scalable Best-of-N Selection for Large Language Models via Self-Certainty

*Zhewei Kang et al. · 2025 · NeurIPS 2025 · [arXiv:2502.18581](https://arxiv.org/abs/2502.18581) · read priority 1*

**In one line.** Self-certainty scores a generation by how far its token-level output distributions are from uniform (a KL-divergence-style logit-only quantity, no reward model or ground truth needed), and this score scales with sample size N and works on open-ended tasks (e.g. code generation) where exact-match self-consistency cannot apply.

- **Models.** Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama-8B, Qwen-2.5-Coder-32B-Instruct [html].
- **Method.** Self-certainty = −(1/nV) sum_i sum_j log(V*p(j|x,y<=i)) — the KL divergence of the model's predicted token distribution from uniform, averaged over generated tokens and vocabulary (an equivalent cross-entropy form also given); aggregated across N samples either by direct highest-score selection or by 'Borda voting' with vote weight v(r)=(N-r+1)^p by confidence rank (p=0 reduces to majority voting) [html, Eq. 9-11].
- **Datasets.** LiveBench-Math, GSM8K, MATH (math); CRUXEval-O (code reasoning); LiveCodeBench (open-ended code generation) [html].
- **Experiments.** Table 1 compares self-consistency, self-certainty, and Borda-voting variants at N=8 and N=64 across datasets; Figure 1 compares self-certainty's separation of correct/incorrect responses against perplexity on MATH Level 4; Figure 8 evaluates on open-ended LiveCodeBench where exact-match self-consistency cannot apply [html].
- **Results.** Table 1: MATH self-consistency 58.60 (N=8) → 63.40 (N=64); Borda(p=1.2) 58.86 (N=8) → 64.10 (N=64); GSM8K Borda(p=0.3) 89.57 (N=8) → 91.07 (N=64) [html]. On MATH Level 4, self-certainty 'clearly separates correct/incorrect responses' while 'perplexity fails to clearly distinguish' them (Figure 1) [html]. On LiveCodeBench (Figure 8), 'self-certainty consistently outperforms greedy decoding across both models and surpasses USC on Qwen-2.5-Coder-32B-Instruct, with performance scaling positively with sample size N' [html].
- **Conclusions.** A purely logit-derived, ground-truth-free, near-zero-overhead score can outperform token-probability/perplexity-based confidence, scale with sample size like a reward model, and — unlike exact-match self-consistency — extend to open-ended tasks with no fixed answer set [html].

> **Why it matters to us.** The direct logit-only baseline our category needs, and the one our project should most carefully distinguish itself from: self-certainty needs no labelled data (fully training-free, purely a function of the model's own output distribution) but only reflects the model's confidence in ITS OWN tokens, which is a known-unreliable signal for factual correctness in medical QA (the field constant our project documents: recoverability tops out ~0.5-0.6 AUROC from anything cheap). Our probe instead reads hidden states through a small BCE-trained head, trading self-certainty's zero-label training-free property for a trained, ground-truth-anchored correctness signal. This paper is the strongest existing evidence that 'open-ended-capable, logit-only, no extra LLM call' is achievable without training — worth citing to show we considered it and to justify why we still trained a probe (the trained-vs-frozen contrast we report generalizes: a trained head beats or ties a zero-shot judge).

<small>Read from: html.</small>

#### ★ Universal Self-Consistency for Large Language Model Generation

*Xinyun Chen et al. · 2023 · arXiv preprint · [arXiv:2311.17311](https://arxiv.org/abs/2311.17311) · read priority 1 · **PDF in `papers/`***

**In one line.** Extends self-consistency to free-form generation by asking the LLM itself to pick the most consistent candidate from the concatenated sample set, instead of exact-match voting.

- **Models.** PaLM 2-L, gpt-3.5-turbo [html]
- **Method.** Sample multiple candidate responses, concatenate them into one prompt, and ask the LLM (an extra query) to select the index of the most consistent response — no answer extraction or exact-match canonicalisation needed [html, §1].
- **Datasets.** GSM8K, MATH (math reasoning); BIRD-SQL, ARCADE (code generation); GovReport, SummScreen (summarization); TruthfulQA (open-ended QA) [html].
- **Experiments.** Compares USC vs standard self-consistency (where applicable) and vs greedy decoding across all six tasks; Table 7 compares against oracle scores; §4.3 measures position/ordering bias [html].
- **Results.** GSM8K (Table 1, PaLM 2-L): USC 90.2% vs SC 90.4% (matches, doesn't beat, exact-match SC) [html]. GovReport (Table 3): ROUGE-1 USC 40.2 vs greedy 38.8; ROUGE-Lsum 35.1 vs 33.8 [html]. TruthfulQA (Table 4): GPT-judge USC 67.7% vs greedy 62.1%; GPT-info USC 99.0% vs greedy 95.1% [html]. GSM8K oracle is 96.2% vs USC 90.2% — 'a notable gap to oracle' [html, Table 7]. Figure 3: accuracy can drop by N=16 on some tasks (diminishing/negative returns) [html].
- **Conclusions.** An LLM-as-judge over the whole candidate set is a workable training-free consistency signal on tasks where exact-match self-consistency cannot apply, but it costs an extra full LLM call, has an unaddressed position-bias risk, and still leaves a large oracle gap [html].

> **Why it matters to us.** The direct free-text analogue of vanilla self-consistency and the standard 'training-free open-ended baseline' a reviewer will ask about. Mechanistically it is a listwise LLM-judge reranker (one extra generative call over the concatenated pool), unlike our pointwise MLP probe (one score per candidate from hidden states already computed during generation, no extra LLM call). Its own reported oracle gap (96.2 vs 90.2 on GSM8K) is the same qualitative 'selection wall' shape we report (oracle@8 climbing to 0.418 while selection under-converts it), suggesting the wall is a general property of best-of-N selection, not specific to our probe or domain.

<small>Read from: html.</small>

#### ★ Self-Consistency Improves Chain of Thought Reasoning in Language Models

*Xuezhi Wang et al. · 2022 · ICLR 2023 · [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) · read priority 1 · **PDF in `papers/`***

**In one line.** Introduces self-consistency: sample multiple chain-of-thought reasoning paths and take a majority vote over the final answers instead of greedy decoding, for large gains on closed-form reasoning tasks.

- **Models.** UL2-20B, GPT-3-175B (code-davinci-001/002), LaMDA-137B, PaLM-540B [html]
- **Method.** Sample m independent reasoning paths at temperature > 0 (m=40 in main results), extract each path's final answer, and output the majority-vote answer: argmax_a sum_i 1(a_i = a) [html].
- **Datasets.** Arithmetic reasoning (GSM8K, AQuA, SVAMP), commonsense reasoning (StrategyQA, ARC-challenge) [abstract/html].
- **Experiments.** Ablates N in {1,5,10,20,40} (Figure 2); compares self-consistency vs single-sample chain-of-thought across models and tasks [html].
- **Results.** PaLM-540B, Table 2: GSM8K CoT 56.5 → self-consistency 74.4 (+17.9 abs pts); AQuA 35.8→48.3 (+12.5); SVAMP 79.0→86.6 (+7.6) [html, Table 2]. Table 3 commonsense: StrategyQA 75.3→81.6 (+6.3); ARC-challenge 85.2→88.7 (+3.5) [html, Table 3]. Abstract also reports AQuA +12.2%, StrategyQA +6.4%, ARC-challenge +3.9% [abstract] (minor rounding differences vs body tables).
- **Conclusions.** Marginalising over sampled reasoning paths via majority vote is a large, free (no extra training) accuracy gain, but it fundamentally requires answers 'from a fixed answer set' extractable by parsing rules [html].

> **Why it matters to us.** This is the training-free baseline our probe is measured against. Its core limitation — needing a canonicalisable/extractable final answer for the vote to be well-defined — is exactly why it degenerates on our free-text medical VQA: with no fixed answer set, exact-string majority vote sits at the random-pick floor in our benchmarks and doesn't improve from N=2 to N=16 while oracle@N climbs 0.156→0.418. Our hidden-state probe sidesteps this by scoring each candidate independently rather than requiring candidates to match each other.

<small>Read from: html.</small>

#### Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score

*Manh Nguyen et al. · 2026 · arXiv preprint · [arXiv:2604.12196](https://arxiv.org/abs/2604.12196) · read priority 2*

**In one line.** Radial Consensus Score (RCS) computes a weighted Fréchet mean ('semantic center') of answer embeddings and ranks candidates by distance to that center, generalizing majority voting to a geometric, training-free best-of-N method that improves with more weighting schemes and scales better with larger N than majority voting.

- **Models.** Five open-weight LLMs [abstract].
- **Method.** Embeds candidate answers, computes a weighted Fréchet mean (a geometric generalization of an average) of the embeddings as the 'semantic center', and ranks/selects candidates by radial distance to that center; supports uniform, frequency-based, and probability-based weighting schemes; fully black-box compatible [abstract].
- **Datasets.** Seven benchmarks spanning short-form QA and long-form reasoning [abstract].
- **Experiments.** Compares RCS variants to majority voting and other strong baselines across the seven benchmarks and five models, at increasing sampling budgets; also tests RCS as a drop-in majority-voting replacement inside multi-agent debate [abstract].
- **Results.** 'RCS variants consistently outperform strong baselines, with gains becoming more pronounced as the sampling budget increases' [abstract]. 'strong robustness in black-box scenarios' [abstract]. No specific percentage numbers given in the abstract; not extracted.
- **Conclusions.** Geometric consensus (weighted centroid distance in embedding space) is a more expressive, more scalable generalization of majority voting than discrete voting, and transfers to multi-agent debate settings [abstract].

> **Why it matters to us.** A third 2026 training-free geometric-consensus method (alongside EBA, 2606.12003, and LSC, 2508.18395) — all converging on 'measure agreement as distance in embedding/representation space' as the fix for self-consistency's exact-match limitation. Distinct from our probe in the same way as EBA: RCS still measures mutual agreement among candidates (semantic center = consensus), not P(correct) against ground truth, so like all consensus methods it cannot in principle catch a case where all N samples are confidently, consistently wrong — the coverage-wall failure mode we separately report (37% of questions have no correct candidate in 8).

<small>Read from: abstract-only.</small>

#### Latent Self-Consistency for Reliable Majority-Set Selection in Short- and Long-Answer Reasoning

*Jungsuk Oh and Jay-Yoon Lee · 2025 · arXiv preprint · [arXiv:2508.18395](https://arxiv.org/abs/2508.18395) · read priority 2*

**In one line.** Latent Self-Consistency (LSC) trains lightweight learnable 'summary token' embeddings via supervised contrastive learning to measure semantic agreement between sampled responses via the model's own KV cache, beating SC/USC/WUCS on both short- and long-form benchmarks with <1% runtime overhead — but note this is NOT training-free.

- **Models.** LLaMA3.1-8B-Instruct, LLaMA3.3-70B, Qwen3-8B [html].
- **Method.** Appends K learnable 'summary token' embeddings after each response's EOS, trained via supervised contrastive learning (pull same-answer responses together, push different-answer ones apart) on labelled QA data; at inference, reuses the existing KV cache to compute only K new tokens per response and ranks by exponentially-weighted cosine similarity between summary-token embeddings [html, updating '<3×10^-6 of parameters'].
- **Datasets.** Short-form: GSM8K, MATH, TriviaQA, MMLU (in-domain), CommonsenseQA, TruthfulQA-MC1 (OOD). Long-form: TruthfulQA, MSMARCO-NLG, HumanEval, MBPP, HumanEval+, MBPP+, CNN/DailyMail (all OOD) [html].
- **Experiments.** Compares LSC vs SC, WUCS, USC on short- and long-form benchmarks; measures runtime/memory overhead and calibration error (ECE); ablates off-the-shelf SBERT embeddings vs LSC's learned ones [html, Tables 1-6].
- **Results.** Short-form, LLaMA3.1-8B (Table 2): GSM8K SC 92.2% vs LSC 92.3%; MATH 52.5% vs 52.6%; TriviaQA 72.7% vs 74.1%; MMLU 73.6% vs 73.8% [html]. Runtime/memory overhead (Table 1): LSC-8B 0.2% time / 0.005% memory vs USC-8B 7.4% time / 16.2% memory [html]. Consistency-with-majority (Table 4): LSC 99.6% (short) / 93.0% (long) vs USC 86.2% / 81.0% [html].
- **Conclusions.** A small amount of supervised training on labelled correctness data (not training-free) buys near-negligible inference overhead and beats both exact-match SC and LLM-judge USC on both short- and long-form tasks, including strictly out-of-domain long-form benchmarks [html].

> **Why it matters to us.** The closest published method to our own mechanism among this category's papers — but it is explicitly NOT training-free: like our probe, it trains a small component (contrastive summary-token embeddings vs. our BCE-trained MLP) on labelled correctness data and reads it off the SAME forward pass used for generation (its KV cache; our tapped hidden states), giving near-zero extra inference cost in both cases. Key differences: LSC trains on ~1,500-2,700 general-domain examples and generalizes OOD without medical specificity, while our probe needs ~100 labelled in-domain (per-benchmark) questions; LSC's training target is inter-candidate agreement, ours is ground-truth P(correct) directly. Worth flagging as evidence that 'lightweight learned probe on internal representations, near-zero overhead' is an emerging pattern the field is converging on independently of us.

<small>Read from: html.</small>

#### Truth or Deceit? A Bayesian Decoding Game Enhances Consistency and Reliability

*Weitong Zhang et al. · 2024 · arXiv preprint · [arXiv:2410.01064](https://arxiv.org/abs/2410.01064) · read priority 2*

**In one line.** The Bayesian Decoding Game (predecessor to Wasserstein-BDG) models decoding as a multistage game achieving consensus through 'Correctness Alignment' and 'Ambiguity Calibration', letting a 13B model beat a 540B model on one reported comparison.

- **Models.** LLaMA-13B, PaLM-540B (comparison point), plus other LLM strategies integrated into the game framework [abstract].
- **Method.** Multistage Bayesian decoding game: dynamically converges to a consensus on the most reliable output and distinguishes {Valid, Specious} outputs 'without human feedback or additional training', via Correctness Alignment (consistency) and Ambiguity Calibration (reliability) [abstract].
- **Datasets.** Not individually named in the abstract [abstract].
- **Experiments.** Compares game-augmented decoding to baseline decoding across integrated LLM strategies and models [abstract].
- **Results.** 'e.g., 78.1 LLaMA13B vs 76.6 PaLM540B' as an example of the game mechanism letting a smaller model outperform a much larger one [abstract]. No further numbers extracted.
- **Conclusions.** A game-theoretic consensus mechanism run purely at decoding time (no training, no human feedback) can substitute for model scale on some tasks [abstract].

> **Why it matters to us.** This is the 'classic BDG' baseline that Wasserstein Equilibrium Decoding (2605.18313, our nearest medical-VQA neighbour) explicitly extends by swapping its lexical order-match/sigma-separation convergence test for a semantic Wasserstein-distance one. Reading this paper is what makes WED's contribution legible: WED's entire delta is in how convergence/agreement is measured, not in the game structure itself.

<small>Read from: abstract-only.</small>

#### Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation

*Lorenz Kuhn et al. · 2023 · ICLR 2023 (Spotlight) · [arXiv:2302.09664](https://arxiv.org/abs/2302.09664) · read priority 2*

**In one line.** Introduces semantic entropy: cluster sampled generations by whether they mean the same thing (bidirectional entailment), then compute entropy over meaning-clusters rather than exact strings, as an unsupervised uncertainty estimate that predicts QA accuracy better than naive baselines.

- **Models.** Not individually named in abstract; open-ended QA setting, single off-the-shelf LLM per experiment, unsupervised, no fine-tuning needed [abstract].
- **Method.** Cluster sampled answers into semantic equivalence classes via entailment (answers that mutually entail each other are treated as the same 'meaning'), then compute entropy over the resulting cluster distribution — 'semantic entropy' — as the uncertainty score [abstract].
- **Datasets.** Question answering datasets (not individually named in abstract) [abstract].
- **Experiments.** Ablation studies comparing semantic entropy to comparable uncertainty baselines for predicting model accuracy [abstract].
- **Results.** 'the semantic entropy is more predictive of model accuracy on question answering data sets than comparable baselines' [abstract]. No specific numeric scores given in the abstract; not extracted.
- **Conclusions.** Measuring uncertainty over meaning-clusters rather than exact-string clusters solves the 'semantic equivalence' problem (different strings, same meaning) that breaks naive token/answer-level uncertainty and self-consistency [abstract].

> **Why it matters to us.** The canonical citation for why exact-match self-consistency breaks in free text: different correct phrasings of the same medical finding will not exact-match each other, so a naive majority vote undercounts true consensus — plausibly part of why our answer-prior/self-consistency baselines sit at the random-pick floor on open-ended medical VQA. Semantic-entailment clustering is an alternative training-free fix to embedding-space clustering (EBA, 2606.12003) — worth noting that there are two competing training-free 'what counts as agreement' fixes: entailment-based (this paper) and embedding-distance-based (EBA, LSC).

<small>Read from: abstract-only.</small>

#### LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion

*Dongfu Jiang et al. · 2023 · ACL 2023 · [arXiv:2306.02561](https://arxiv.org/abs/2306.02561) · read priority 2*

**In one line.** LLM-Blender introduces PairRanker (pairwise cross-attention comparison of candidate pairs to rank them) and GenFuser (fuses the top-ranked candidates into an improved output), outperforming individual LLMs on a new MixInstruct benchmark.

- **Models.** Ensemble of multiple open-source LLMs [abstract].
- **Method.** PairRanker: jointly encodes the input and a pair of candidates with a cross-attention encoder to decide which of the two is better, aggregated into a full ranking; GenFuser: generates a fused answer from the top-K ranked candidates [abstract].
- **Datasets.** MixInstruct (introduced in this paper: a mixture of instruction datasets with oracle pairwise comparisons) [abstract].
- **Experiments.** Compares PairRanker's correlation with ChatGPT-based ranking against other ranking methods; evaluates LLM-Blender (PairRanker+GenFuser) vs individual LLMs and baselines on MixInstruct [abstract].
- **Results.** 'PairRanker exhibits the highest correlation with ChatGPT-based ranking' [abstract]. 'LLM-Blender significantly outperform individual LLMs and baseline methods across various metrics, establishing a substantial performance gap' [abstract]. No specific percentage numbers given in the abstract; not extracted.
- **Conclusions.** Pairwise comparison is a stronger ranking signal than pointwise scoring for picking among diverse-quality LLM outputs, and the best output can often be improved further by fusing top candidates rather than just selecting one [abstract].

> **Why it matters to us.** The canonical pairwise-reranking baseline for our category. It is a trained cross-attention ranker (not training-free), so it is closer in spirit to our probe than to self-consistency — but it compares candidate PAIRS (relative judgment) whereas our probe scores each candidate independently (absolute P(correct)), and it additionally fuses candidates (GenFuser) rather than just selecting one, which our best-of-N method does not do. Worth citing to justify why we chose pointwise scoring: pairwise ranking is O(N^2) comparisons for N candidates, and our per-candidate hidden states are already computed for free during generation, whereas a pairwise ranker needs an extra encoder pass per pair.

<small>Read from: abstract-only.</small>


#### Also in this area (10), in brief

- **Look Light, Think Heavy: What Multimodal Chain-of-Thought Reasoning Can and Cannot Do** — Zhuoran Jin et al. (2026), [arXiv:2606.22565](https://arxiv.org/abs/2606.22565). A systematic study across 12 multimodal tasks and 22 models finds chain-of-thought reasoning is not a free lunch for multimodal models — it hurts perception tasks (visual grounding, object counting) while helping math/science/multi-image reasoning, and multimodal CoT shows declining 'visual reflection' through the reasoning trace.<br><small>*For us:* External, independent 2026 confirmation of our own finding that 'reasoning hurts perception' — our pooled result is −0.0401 [−0.0456,−0.0347] over 30,250 paired perception samples, 17/20 negative cells. This paper reaches the same qualitative conclusion (CoT hurts visual grounding/counting, a perception task) on a different, broader multimodal benchmark suite and model set, strengthening our claim as a field-general phenomenon rather than a Lingshu-specific artifact. It is not a selection/consensus method itself — it is included here because it was explicitly flagged for this category; its relevance to CAT E is indirect (it motivates why our best-of-N pool is sampled under direct decoding rather than reasoning mode, and shows the reasoning-vs-direct axis is orthogonal to the training-free-selection methods surveyed here).</small>
- **Ranked Voting based Self-Consistency of Large Language Models** — Weiqin Wang et al. (2025), [arXiv:2505.10772](https://arxiv.org/abs/2505.10772). Ranked-Voting Self-Consistency has each sample produce a ranked list of candidate answers (not just one), then aggregates across samples using ranked-voting rules (instant-runoff, Borda count, mean reciprocal rank), beating plain majority-vote self-consistency on 3 multiple-choice and 3 open-ended QA datasets.<br><small>*For us:* Directly relevant as it claims to improve self-consistency specifically on open-ended QA (one of its 3 datasets) — a claim worth checking against our finding that self-consistency sits at the random-pick floor on our open-ended medical benchmarks; if their open-ended gain replicates, per-sample ranked lists (not just top-1 answers) could be a cheap addition to our pipeline; if it doesn't (e.g. medical free text is harder to canonicalize into a ranked candidate list than their open-ended tasks), that's a useful negative data point to cite.</small>
- **Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning** — Yiwei Li et al. (2024), [arXiv:2401.10480](https://arxiv.org/abs/2401.10480). Early-Stopping Self-Consistency (ESC) stops sampling reasoning paths once a running window shows the same answer repeatedly, cutting sample counts by 33-84% across six reasoning benchmarks at comparable accuracy.<br><small>*For us:* Another training-free early-stopping baseline (alongside Adaptive-Consistency, 2305.11860); both assume a canonicalisable, votable answer, so neither transfers directly to our open-ended setting — worth noting as a limitation shared across the whole early-stopping sub-literature when we cite it as field context rather than a directly-applicable method.</small>
- **Soft Self-Consistency Improves Language Model Agents** — Han Wang et al. (2024), [arXiv:2402.13212](https://arxiv.org/abs/2402.13212). Soft Self-Consistency replaces discrete majority-vote scoring with a continuous score computed from model likelihoods, which helps on long-horizon interactive tasks where valid answers are too sparse for exact-match voting to work with few samples.<br><small>*For us:* A logit-based continuous alternative to discrete majority voting, relevant to the 'logit-based scores' half of our category — but it still relies on the model's own token likelihoods (like self-certainty, 2502.18581) rather than a trained correctness signal, so it inherits self-certainty's basic risk: a confidently-worded wrong answer scores well. Useful contrast case for why our probe reads hidden states plus a trained head rather than raw likelihoods.</small>
- **More Agents Is All You Need** — Junyou Li et al. (2024), [arXiv:2402.05120](https://arxiv.org/abs/2402.05120). Shows LLM accuracy scales with the number of independently-sampled agent instances combined by simple sampling-and-voting ('Agent Forest'), with the degree of benefit correlated to task difficulty.<br><small>*For us:* Reinforces the point that plain voting/consensus scales with N mainly when the task has a votable structure; in our setting oracle@N climbs strongly with N (0.156→0.418) but our self-consistency baseline does NOT improve N=2→16, the opposite of the scaling this paper reports for tasks where voting is well-defined — a useful contrast to cite when arguing that our task's answer format (free text), not our N, is what breaks voting-based scaling.</small>
- **It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk** — Amanda Bertsch et al. (2023), [arXiv:2310.01387](https://arxiv.org/abs/2310.01387). A survey/unification paper showing several modern decoding methods (including some self-consistency-style methods) are special cases of Minimum Bayes Risk decoding, and gives concrete recommendations for using MBR in NLP.<br><small>*For us:* Gives us the correct field vocabulary to describe our own method's relationship to MBR: our probe is NOT MBR (it scores each candidate independently against a trained correctness model rather than against expected pairwise similarity/risk within the sample pool), and this paper's framework is useful for explicitly stating that distinction in a related-work section ('unlike MBR-style methods, which estimate utility from agreement within the pool, our verifier is trained against ground truth').</small>
- **Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs** — Pranjal Aggarwal et al. (2023), [arXiv:2305.11860](https://arxiv.org/abs/2305.11860). Adaptive-Consistency dynamically decides how many samples to draw per question (instead of a fixed N) using a lightweight stopping rule based on agreement-so-far, cutting sampling budget up to 7.9x with under 0.1% average accuracy loss.<br><small>*For us:* Directly relevant to our fixed-N=8 best-of-N design: this and the early-stopping paper (2401.10480) are the standard adaptive/early-stopping baselines a reviewer will expect us to discuss even though we didn't implement adaptive N — worth a sentence noting our probe could in principle be combined with an adaptive stopping rule to cut the 8x sampling cost, since our probe scores candidates as they're generated (hidden states tapped during generation, no extra forward pass).</small>
- **Lightweight reranking for language model generations** — Siddhartha Jain et al. (2023), [arXiv:2307.06857](https://arxiv.org/abs/2307.06857). Proposes a lightweight reranker using only pairwise statistics between black-box LLM generations (formalized as an extension of self-consistency), giving strong gains on code generation and robust gains on autoformalization, summarization, and translation.<br><small>*For us:* Another training-free pairwise/self-consistency-family reranker relevant to our category, notable for explicitly formalizing 'self-consistency is a special case of a broader pairwise-agreement framework' — the same conceptual move as the MBR-unification paper (2310.01387) and worth citing together as the theoretical backdrop for why exact-match self-consistency is a narrow special case that fails on free text.</small>
- **The Consensus Game: Language Model Generation via Equilibrium Search** — Athul Paul Jacob et al. (2023), [arXiv:2310.09139](https://arxiv.org/abs/2310.09139). The Consensus Game casts LM decoding as a signalling game between a generative and a discriminative view of the same model, and solving for its approximate equilibrium (Equilibrium-Ranking) improves accuracy enough that a 7B model can beat a 65B one on some benchmarks.<br><small>*For us:* The direct theoretical predecessor of the Bayesian Decoding Game family (2410.01064) that our nearest-neighbour paper, Wasserstein Equilibrium Decoding (2605.18313), builds on — good for the related-work lineage: Consensus Game (2023) → Bayesian Decoding Game (2024) → Wasserstein/semantic BDG for medical VQA (2026, WED). Like WED, it is a game/equilibrium method (iterative, generator-vs-discriminator), structurally different from our one-shot pointwise probe scoring.</small>
- **High Quality Rather than High Model Probability: Minimum Bayes Risk Decoding with Neural Metrics** — Markus Freitag et al. (2021), [arXiv:2111.09388](https://arxiv.org/abs/2111.09388). Shows a translation's estimated model probability barely correlates with human-judged quality, and that Minimum Bayes Risk decoding with a neural quality metric (BLEURT) over sampled candidates gives large human-eval gains over beam search.<br><small>*For us:* Foundational justification for scoring candidates by an external notion of expected quality/correctness rather than by generation probability or majority agreement — the same philosophical move our BCE-trained probe makes (score P(correct) directly, don't trust the model's own probabilities or make candidates agree with each other). MBR needs a pairwise/pointwise quality metric over the sample pool itself (e.g. average similarity to other samples); our probe instead uses a metric trained against ground-truth labels, so it doesn't depend on the pool containing consensus.</small>

## 3.6 Uncertainty, Calibration, and Hallucination Detection in LLMs and VLMs -- with the Medical Evidence

This category covers how to tell whether a model's answer should be trusted, and the closely related problem of hallucination -- confident but wrong or ungrounded output. Confidence/uncertainty signals split into four broad families: verbalized (ask the model to state a confidence in words), logit-based (read the model's own token probabilities), sampling-based / semantic entropy (generate several answers and measure how much they agree once paraphrases are grouped by meaning), and internal-state / probing-based (train a small classifier on the model's hidden activations). In vision-language models this gets harder: hallucination includes object hallucination (describing things not in the image), driven by language priors and training-set co-occurrence statistics, and 'yes-bias' (a default tendency to answer yes regardless of evidence). Medical VLMs make it harder still -- models are shown to be overconfident regardless of scale or prompting, semantic entropy needs to be vision-conditioned to work at all, and uncertainty quality tracks accuracy rather than being an intrinsic property of the method, degrading exactly where it is needed most. For our project, these are the alternative confidence signals a reviewer will expect us to compare our trained hidden-state MLP probe against, and the vocabulary (calibration, ECE, AUROC/discrimination, verbalized vs white-box, semantic entropy) a reviewer will expect Leo to use correctly.

#### ★ Calibrated Triage, Not Autonomy: Confidence Estimation for Medical Vision-Language Models

*Reza Khanmohammadi et al. · 2026 · arXiv preprint · [arXiv:2606.15910](https://arxiv.org/abs/2606.15910) · read priority 1 · **PDF in `papers/`***

**In one line.** A head-to-head benchmark of nine confidence estimators (training-free logit, verbalized prompting, trained internal probes) across five LVLMs and three medical VQA datasets finds no estimator reliably best, and even the strongest safely triages only ~25% of radiology cases at 20% error tolerance, and almost nothing in pathology.

- **Models.** Five open-weight LVLMs (names not given in the abstract/abs page -- "not extracted"); probes are trained only on natural (non-medical) images and applied to medicine with no adaptation.
- **Method.** Evaluates nine confidence estimators across three families -- training-free logit baselines, prompt-based self-reports, and trained internal probes -- recast as bounded selective prediction: given a score and an error-tolerance budget, how much of the workload can be deferred while keeping error under budget.
- **Datasets.** Three medical VQA datasets covering broad clinical imaging, radiology, and pathology (specific names not extracted).
- **Experiments.** Compares the nine estimators' discrimination and calibration across 5 LVLMs x 3 datasets; measures safe-deferral fraction under (a) a distribution-free guarantee and (b) a held-out threshold, at a fixed 20% error tolerance.
- **Results.** "Discrimination barely separates the estimators" and "a fixed high-confidence cutoff separates them far less than it appears, because their scores sit on incomparable scales" [abstract]. "At a 20% error tolerance the strongest estimator defers about a quarter of radiology cases under a distribution-free guarantee and a third under a held-out threshold, and little to none of pathology" [abstract].
- **Conclusions.** No estimator is reliably best across domains or models; base-model competence sets a ceiling on safe deferral and the confidence layer only determines how much of that ceiling is reachable. The authors frame the useful role as "calibrated triage under clinical oversight, not autonomous deferral."

> **Why it matters to us.** Direct comparison set for our probe -- it is exactly their "trained internal probe" family, but frozen/off-the-shelf and trained on natural images, vs. ours trained in-domain on our own 3-layer x candidate hidden states. Their transfer failure is a caution against assuming our probe generalizes beyond its trained benchmark, and their "incomparable scales" finding supports reporting our probe rank-averaged (which we already do). Its abstention/triage component is out of scope for us by standing rule.

<small>Read from: abstract-only.</small>

#### ★ Overconfidence and Calibration in Medical VQA: Empirical Findings and Hallucination-Aware Mitigation

*Ji Young Byun et al. · 2026 · arXiv preprint · [arXiv:2604.02543](https://arxiv.org/abs/2604.02543) · read priority 1 · **PDF in `papers/`***

**In one line.** Across three VLM families (2B-38B) and three medical VQA benchmarks, overconfidence persists regardless of scale or prompting (CoT, verbalized confidence); Platt scaling reliably beats prompt-based calibration but doesn't improve AUROC; adding hallucination-detection signals (their HAC method) improves both, especially on open-ended questions.

- **Models.** Three VLM families across scales 2B-38B: Qwen3-VL, InternVL3, LLaVA-NeXT.
- **Method.** Compares raw confidence, chain-of-thought and verbalized-confidence prompting, and post-hoc calibration (Platt scaling) for calibration error and AUROC; proposes Hallucination-Aware Calibration (HAC), adding vision-grounded hallucination-detection signals as extra inputs to the calibration function.
- **Datasets.** Three medical VQA benchmarks (names not extracted from the abstract/abs page).
- **Experiments.** Measures calibration error and AUROC per model/scale/prompting-strategy; ablates post-hoc calibration methods; evaluates HAC vs both prompting-based and plain post-hoc calibration, separately for closed- and open-ended questions.
- **Results.** "Overconfidence persists across model families and is not resolved by scaling or prompting, such as chain-of-thought and verbalized confidence variants" [abstract]. "Simple post-hoc calibration approaches, such as Platt scaling, reduce calibration error and consistently outperform the prompt-based strategy" but are "inherently limited in improving the discriminative quality of predictions, leaving AUROC at the same level" [abstract; confirmed via WebFetch of abs page, no new numbers surfaced]. HAC "improves both calibration and AUROC, with the largest gains on open-ended questions" [abstract]. No exact ECE/AUROC point values recoverable -- "not extracted".
- **Conclusions.** Recommend post-hoc calibration (e.g. Platt scaling) as standard practice for medical VLM deployment over raw/prompted confidence, and highlight hallucination-detection signals as a useful complementary calibration input, particularly for open-ended VQA.

> **Why it matters to us.** Platt scaling / post-hoc calibration is the cheap competitor to our probe, and their finding that it caps out on AUROC (recalibrates but can't improve ranking) is exactly why our probe's job -- ranking/selecting among candidates, not just recalibrating a scalar -- is a different, harder task. Their "largest gains on open-ended questions" for a hallucination-aware signal supports our choice to work on open-ended medical VQA, where routing signals behave differently than on MCQ.

<small>Read from: abstract-only.</small>

#### ★ Detecting hallucinations in large language models using semantic entropy

*Sebastian Farquhar et al. · 2024 · Nature 630, 625-630 (2024) · [doi:10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0) · read priority 1 · **PDF in `papers/`***

**In one line.** Introduces semantic entropy -- clustering sampled generations by bidirectional textual entailment and computing entropy over the resulting meaning-clusters -- as an unsupervised, training-free hallucination detector that beats naive token entropy and P(True) baselines.

- **Models.** LLaMA 2 Chat (7B/13B/70B), Falcon Instruct (7B/40B), Mistral Instruct (7B), GPT-4 (biography experiments); entailment judged with DeBERTa-Large-MNLI or GPT-3.5/GPT-4 as the NLI judge.
- **Method.** Sample multiple free-form answers per question at temperature>0; cluster answers into semantic-equivalence classes via bidirectional entailment (A entails B and B entails A); compute Shannon entropy over the cluster-probability distribution instead of over raw token sequences. A discrete variant works from cluster counts alone (no token log-probs needed), so it applies to black-box/API models.
- **Datasets.** TriviaQA, SQuAD 1.1, BioASQ, NQ-Open, SVAMP, plus a new FactualBio biography-generation dataset.
- **Experiments.** Sentence-length QA (30 model-task combinations): semantic entropy vs naive predictive entropy, P(True), and embedding-regression baselines, scored by AUROC for detecting wrong answers. Paragraph-length biography generation: discrete semantic entropy vs self-check and P(True) via AUROC/AURAC.
- **Results.** "AUROC value of 0.790" for semantic entropy averaged over 30 model-task combinations, vs 0.691 (naive entropy), 0.698 (P(True)), 0.687 (embedding regression) [Nature, via WebFetch summary of Fig. 2]; performance "ranged between 0.78 and 0.81 AUROC" across model families/scales [same]. On biographies, discrete semantic entropy had the highest AUROC/AURAC; "at 80%+ questions answered, semantic entropy maintained highest accuracy" [Nature, via WebFetch summary of Fig. 3].
- **Conclusions.** Semantic entropy needs no labels or fine-tuning and detects hallucinations caused by the model's lack of knowledge ("confabulations") better than logit-only or single-sample self-report methods. The authors note this could let a system "refuse to answer questions likely to cause confabulations."

> **Why it matters to us.** Founding paper for the sampling-based/semantic-entropy family -- the natural training-free competitor to our trained MLP probe. Both sample N candidates, but semantic entropy scores agreement across the whole set (a question-level uncertainty) where our probe scores each candidate individually (a per-candidate correctness score) for selection. Gives us the term "semantic entropy" and a self-consistency-at-N=8 baseline we should name explicitly. Its abstention/triage component is out of scope for us by standing rule.

<small>Read from: html.</small>

#### ★ A Survey of Confidence Estimation and Calibration in Large Language Models

*Jiahui Geng et al. · 2023 · arXiv preprint · [arXiv:2311.08298](https://arxiv.org/abs/2311.08298) · read priority 1 · **PDF in `papers/`***

**In one line.** Survey organizing LLM confidence-estimation methods into white-box (logit-based, internal-state-based, semantic) and black-box (verbalized, consistency-based, surrogate-model) families, cataloging calibration metrics and applications including hallucination detection and selective generation.

- **Models.** Survey -- covers methods applied across many LLMs, no single model evaluated.
- **Method.** Literature survey and taxonomy paper; no new empirical method proposed.
- **Datasets.** N/A (survey).
- **Experiments.** N/A (survey); summarizes techniques, metrics and applications from the literature it covers.
- **Results.** Descriptive, not numeric. Taxonomy: white-box = "logit-based methods" (token-level probabilities/entropy), "internal state-based methods" (hidden layer activations, attention), and semantic approaches; black-box = "linguistic confidence (verbalized method)", "consistency-based estimation" (agreement across samples), and "surrogate models" [arXiv HTML, via WebFetch]. Calibration metrics covered: ECE ("weighted average of the discrepancies between the mean predicted probability and the actual accuracy"), reliability diagrams, AUROC, AUARC, and a token-level ECE variant for sequence generation [arXiv HTML, via WebFetch].
- **Conclusions.** LLMs pose "unique challenges" for confidence estimation beyond classical classifier calibration; calls for comprehensive cross-domain benchmarks, extending methods to multi-modal LLMs, and calibration that accounts for legitimate human disagreement rather than a single ground truth.

> **Why it matters to us.** The map we use to name the four families of confidence/uncertainty signal in this category -- our probe is a "white-box, internal-state-based" method in this taxonomy, alongside semantic entropy probes and INSIDE; best-of-N + verifier is closest to their "consistency-based/surrogate" black-box family but ours reads internal states, not just output agreement. Its explicit call to "extend methods to multi-modal LLMs" is the gap our medical-VLM open-text work fills.

<small>Read from: html.</small>

#### Deterministic Hallucination Detection in Medical VQA via Confidence-Evidence Bayesian Gain

*Mohammad Asadi et al. · 2026 · arXiv preprint · [arXiv:2603.21693](https://arxiv.org/abs/2603.21693) · read priority 2*

**In one line.** Proposes Confidence-Evidence Bayesian Gain (CEBaG), a fully deterministic (no sampling, no external NLI model) hallucination detector for medical VQA combining token-level predictive-variance and image-evidence-magnitude signals, beating semantic-entropy-style methods (VASE) by 8 AUC points on average across 16 settings while being far cheaper.

- **Models.** Four medical MLLMs.
- **Method.** Combines two signals read directly from token log-probabilities during a single forward pass: token-level predictive variance (inconsistent confidence across the response's tokens) and evidence magnitude (how much the image shifts per-token predictions relative to a text-only version of the same query) -- no stochastic sampling, no external NLI model, no task-specific hyperparameters.
- **Datasets.** Three VQA benchmarks (names not extracted from the abstract).
- **Experiments.** Evaluates CEBaG against Semantic Entropy (SE) and Vision-Amplified Semantic Entropy (VASE) baselines across 4 MLLMs x 3 benchmarks = 16 experimental settings, measuring AUC for hallucination detection.
- **Results.** "CEBaG achieves the highest AUC in 13 of 16 settings and improves over VASE by 8 AUC points on average, while being fully deterministic and self-contained" [abstract].
- **Conclusions.** A deterministic, single-forward-pass, logit-only signal (no sampling, no external model) can beat 10-20x more expensive sampling-based semantic-entropy methods for medical VQA hallucination detection -- a strong efficiency and accuracy result against the "more sampling is better" assumption in this literature.

> **Why it matters to us.** An important efficiency counterpoint: CEBaG needs zero extra sampling and no probe training, yet beats semantic-entropy variants by a large margin (8 AUC points) -- worth directly comparing against our probe's own cost/accuracy tradeoff, since our probe is also read during generation with "no extra forward pass" but does need training, whereas CEBaG needs neither training nor extra sampling. Likely the single most relevant training-free competitor baseline in this category for our efficiency argument.

<small>Read from: abstract-only.</small>

#### Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?

*Arnisa Fazla et al. · 2026 · Findings of EMNLP 2026 · [arXiv:2606.16583](https://arxiv.org/abs/2606.16583) · read priority 2*

**In one line.** Benchmarking 8 uncertainty-estimation methods across 12 clinical VLMs finds UE quality is not intrinsic to the method but tracks the base model's accuracy (degrading exactly where reliability is most needed), and that hiding the correct MCQ option (NOTA perturbation) collapses accuracy while uncertainty barely moves -- yet uncertainty on the unperturbed input still predicts which cases will later collapse under NOTA.

- **Models.** 12 clinical VLMs.
- **Method.** Benchmarks 8 uncertainty-estimation (UE) methods on clinical VQA; introduces a stress test -- None-Of-The-Above (NOTA) perturbation, hiding the correct option among the multiple-choice answers -- to test whether UE degrades gracefully or stays miscalibrated when the model is forced to fail; also tests whether UE scores on the clean/unperturbed input predict which cases will collapse under NOTA.
- **Datasets.** Clinical VQA benchmarks, multiple-choice format (specific names not extracted from the abstract).
- **Experiments.** Evaluates all 8 UE methods x 12 VLMs on standard clinical VQA (accuracy vs UE quality correlation with base-model competence); applies NOTA perturbations and measures accuracy collapse vs UE-score shift; tests UE-on-clean-input as a predictor of NOTA collapse.
- **Results.** "UE quality is not an intrinsic property of the UE method: it tracks model accuracy, degrading precisely where the model performance is weakest" [abstract]. "When we stress-test models by hiding the correct option among the multiple-choice answers (NOTA perturbations), accuracy collapses while uncertainty barely changes, leaving models systematically miscalibrated" [abstract]. "Uncertainty on the unperturbed input reliably anticipates which predictions will collapse under NOTA, indicating that UE in current VLMs carries diagnostic information about model fragility" [abstract].
- **Conclusions.** This is a negative result for using UE as a real-time "safety net" (it fails exactly when needed most, and doesn't react to a scenario engineered to make the model wrong) but a positive result for UE as an offline diagnostic -- clean-input uncertainty predicts which questions are fragile to adversarial-style perturbation, useful for identifying weak spots before deployment rather than catching failures live.

> **Why it matters to us.** FLAGGED PAPER: the title suggests a blanket negative result about clinical VQA uncertainty, but the actual finding is nuanced, not "uncertainty is useless": UE fails as a live safety net (doesn't react to NOTA-induced failure) but succeeds as an offline fragility diagnostic (predicts which cases will later fail under stress). This nuance matters for us because our probe is used offline, at selection time over an already-sampled pool of 8 candidates, not as a live pre-answer safety net -- so the paper's negative finding (UE ignoring induced failure) is less directly damaging to our use case than its positive finding (UE-as-fragility-diagnostic) is validating. Its "safety net" / "escalate to a clinician" framing is an abstention/triage application. Its abstention/triage component is out of scope for us by standing rule. We card only its UE-benchmarking and NOTA-stress-test findings.

<small>Read from: abstract-only.</small>

#### Just how sure are you? Improving Verbalized Uncertainty Calibration in Medical VQA

*Eren Senoglu et al. · 2026 · arXiv preprint · [arXiv:2606.27023](https://arxiv.org/abs/2606.27023) · read priority 2*

**In one line.** Proposes a training-based framework that fine-tunes MLLMs with a composite loss (Brier-style calibration term, anchor regularizer, contrastive image-text alignment term, KL stabilization term) to improve verbalized-confidence calibration in medical VQA, cutting calibration error by 60%+ and improving discrimination by 26%+ while preserving accuracy.

- **Models.** MedGemma 4B IT, Qwen2-VL 7B Instruct.
- **Method.** Fine-tunes with a composite loss combining: a Brier-style calibration term, an anchor regularizer (prevents confidence collapsing to extreme 0/1 values), a contrastive image-text alignment term (derived from a 2x2 factorial design crossing image presence with text integrity, to probe reliance on vision vs language priors), and a top-K KL-divergence regularizer (protects answering ability/accuracy during fine-tuning).
- **Datasets.** Three medical VQA benchmarks (names not extracted from the abstract).
- **Experiments.** Compares against prompting-based, sampling-based, and other training-based calibration approaches across 3 benchmarks x 2 architectures; ablates each loss component.
- **Results.** "Our method reduces calibration error by 60% or more, and improves discrimination by 26% or more, while preserving predictive accuracy" [abstract]; "outperforms prompting based, sampling based, and training based approaches" on average across benchmarks [abstract]; "ablation experiments confirm that each component of the loss function is indeed necessary" [abstract].
- **Conclusions.** A composite training-based calibration loss -- that explicitly probes and corrects for image-vs-language-prior reliance via the 2x2 factorial alignment term -- substantially improves both calibration and discrimination for verbalized confidence in medical VQA, beating black-box (prompting/sampling) alternatives.

> **Why it matters to us.** A trained calibration baseline directly on our target task (medical VQA; MedGemma is one of our own backbone families) -- a reviewer will ask why we didn't instead fine-tune the generator itself for calibrated verbalized confidence rather than training a separate probe. This paper shows such fine-tuning works but changes the generator's weights, a heavier, more invasive intervention than our frozen-backbone probe. Its 2x2 image/text-integrity factorial design is also a useful diagnostic technique we could reuse to check whether our probe exploits genuine visual evidence or text-only shortcuts.

<small>Read from: abstract-only.</small>

#### System-Mediated Attention Imbalances Make Vision-Language Models Say Yes

*Tsan Tsai Chan et al. · 2026 · ACL Findings 2026 · [arXiv:2601.12430](https://arxiv.org/abs/2601.12430) · read priority 2*

**In one line.** Attributes the VLM 'yes-bias' (indiscriminately answering 'yes') to functionally redundant attention weight on the system-prompt modality crowding out attention to image and text, and shows causally redistributing attention away from the system modality substantially suppresses yes-bias, often beating existing mitigation methods.

- **Models.** Multiple VLMs (specific names not extracted from the abstract).
- **Method.** Analyzes attention allocation across three input modalities (system prompt, image, text) rather than just image-vs-text; proposes a "system-mediated" account of hallucination attributing yes-bias to redundant system-weight attention; causally intervenes by redistributing attention away from the system modality toward image and text.
- **Datasets.** Yes/no-style VQA evaluation setups (specific benchmark names not extracted from the abstract).
- **Experiments.** Measures yes-bias before/after causal attention redistribution; compares against existing (image-centric) mitigation strategies that only boost image attention.
- **Results.** "Causally redistributing attention from the system modality to image and textual inputs substantially suppresses this bias, often outperforming existing approaches" [abstract]. No specific numeric yes-bias-rate values given -- "not extracted".
- **Conclusions.** The yes-bias failure mode is better explained by a three-way (system/image/text) attention account than the usual two-way (image/text) one; system-prompt attention is itself a lever for mitigating hallucination, not just image attention as prior work assumed.

> **Why it matters to us.** Directly relevant to any yes/no or binary-judgment component in our pipeline (e.g. our 32B LLM-judge that labels correctness, or any yes/no-formatted sub-question in our 8 benchmarks) -- a caution that judge or generator yes/no outputs may be systematically biased toward "yes" for reasons unrelated to genuine evidence, worth checking for as a confound in our exact-match and judge-based correctness labels, especially on any yes/no sub-questions.

<small>Read from: abstract-only.</small>

#### Hallucination Filtering in Radiology Vision-Language Models Using Discrete Semantic Entropy

*Patrick Wienholt et al. · 2025 · European Radiology (2026) · [arXiv:2510.09256](https://arxiv.org/abs/2510.09256) · read priority 2*

**In one line.** Applies discrete semantic entropy to filter out radiology VQA questions likely to be hallucinated for black-box GPT-4o/GPT-4.1, raising accuracy on the retained subset from a 51.7%/54.8% baseline to 76.3%/63.8% at a DSE>0.3 threshold, at the cost of dropping roughly half the questions.

- **Models.** GPT-4o, GPT-4.1 (via API, black-box).
- **Method.** Discrete semantic entropy (DSE): sample each question's answer 15 times at temperature 1.0, group meaning-equivalent responses via bidirectional entailment checks, compute entropy over the resulting clusters, and reject/filter out questions above an entropy threshold before scoring accuracy on the retained subset (vs a low-temperature/greedy baseline).
- **Datasets.** VQA-Med 2019 (500 images with clinical questions and short-text answers) and a diagnostic radiology dataset of 206 cases (60 CT, 60 MRI, 60 radiograph, 26 angiogram) with ground-truth diagnoses.
- **Experiments.** Computes baseline accuracy at low temperature (0.1); recalculates accuracy after excluding questions with DSE>0.6 or DSE>0.3, across 706 total image-question pairs; statistical testing via bootstrap resampling with Bonferroni correction (p<.004).
- **Results.** "Baseline accuracy was 51.7% for GPT-4o and 54.8% for GPT-4.1" [abstract]. "After filtering out high-entropy questions (DSE > 0.3), accuracy on the remaining questions was 76.3% (retained questions: 334/706) for GPT-4o and 63.8% (retained questions: 499/706) for GPT-4.1 (both p < .001)" [abstract]. "Accuracy gains were observed across both datasets and largely remained statistically significant after Bonferroni correction" [abstract].
- **Conclusions.** DSE reliably flags questions likely to be answered wrong in black-box radiology VQA and, used as a filter, substantially raises accuracy on the retained set -- at a real coverage cost (roughly half the questions dropped at the DSE>0.3 threshold).

> **Why it matters to us.** Its tested method IS a reject-option/selective-prediction application (excluding/filtering out questions rather than answering them), so per standing rule we card only its underlying signal here: discrete semantic entropy as a training-free, sampling-based baseline applied to black-box radiology VQA, plus its accuracy-vs-coverage numbers as a directly comparable medical-VQA reference point (51.7-54.8% baseline accuracy). Its abstention/triage component is out of scope for us by standing rule.

<small>Read from: abstract-only.</small>

#### A Survey on Hallucination in Large Vision-Language Models

*Hanchao Liu et al. · 2024 · arXiv preprint · [arXiv:2402.00253](https://arxiv.org/abs/2402.00253) · read priority 2*

**In one line.** Survey establishing the standard taxonomy of LVLM hallucination (types, unique multimodal challenges vs text-only LLMs, causes in training data and model components) and cataloging evaluation benchmarks and mitigation methods.

- **Models.** Survey -- no single model.
- **Method.** Literature survey; taxonomizes hallucination symptoms, root causes (training data, vision encoder, cross-modal alignment, LLM backbone), benchmarks, and mitigation methods.
- **Datasets.** N/A (survey; catalogs many benchmarks).
- **Experiments.** N/A (survey).
- **Results.** Descriptive -- "clarification of the concept of hallucinations in LVLMs, presenting a variety of hallucination symptoms and highlighting the unique challenges inherent in LVLM hallucinations" [abstract]. No numeric results -- survey paper, "not extracted".
- **Conclusions.** LVLM hallucination has multimodal-specific causes distinct from text-only LLM hallucination (e.g. vision-language misalignment, over-reliance on language priors), requiring dedicated benchmarks and mitigation strategies; open questions and future directions are discussed.

> **Why it matters to us.** Standard reference for defining "hallucination" and its LVLM-specific taxonomy/causes for Leo's reports -- gives him the vocabulary (object hallucination, language priors, root-cause categories) a reviewer expects, and frames why medical VLM hallucination (CARES, Med-HallMark, etc., also in this category) needed dedicated benchmarks rather than reusing general-domain ones.

<small>Read from: abstract-only.</small>

#### INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection

*Chao Chen et al. · 2024 · ICLR 2024 · [arXiv:2402.03744](https://arxiv.org/abs/2402.03744) · read priority 2*

**In one line.** Proposes EigenScore, a hallucination-detection metric computed from the eigenvalues of the covariance matrix of an LLM's internal-state embeddings across multiple sampled responses, plus a feature-clipping trick to reduce overconfident generations.

- **Models.** Several popular LLMs (names not extracted from the abstract).
- **Method.** EigenScore: sample multiple responses, embed via internal states, compute the covariance matrix of those embeddings, and use its eigenvalues to measure semantic consistency/diversity directly in the dense embedding space (rather than discrete NLI clustering as in semantic entropy); also proposes test-time feature clipping to truncate extreme activations and reduce overconfident hallucinations.
- **Datasets.** Several popular QA benchmarks (names not extracted from the abstract).
- **Experiments.** Extensive experiments and ablations comparing EigenScore-based hallucination detection to logit-level and self-consistency baselines; ablates the feature-clipping intervention.
- **Results.** "Showing the effectiveness of our proposal" [abstract] -- no specific numeric AUROC/accuracy values given in the abstract -- "not extracted".
- **Conclusions.** Internal-state embeddings retain dense semantic information (beyond what survives token decoding) that a covariance/eigenvalue-based metric can exploit for hallucination detection more effectively than logit-only or naive self-consistency, without needing an external NLI model.

> **Why it matters to us.** A close relative of semantic entropy computed in continuous embedding space from internal states rather than discrete NLI-based clustering -- another internal-state, sampling-based hybrid baseline. Its "no external NLI model needed" framing is a practical advantage worth weighing against our probe (also needs no external model, just the frozen backbone's own hidden states) when discussing compute overhead of alternative confidence signals.

<small>Read from: abstract-only.</small>

#### CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models

*Peng Xia et al. · 2024 · NeurIPS 2024 Datasets and Benchmarks Track · [arXiv:2406.06007](https://arxiv.org/abs/2406.06007) · read priority 2*

**In one line.** A 41K-question, 5-dimension trustworthiness benchmark for medical LVLMs (trustfulness/fairness/safety/privacy/robustness) across 16 imaging modalities and 27 anatomical regions, finding consistent factual inaccuracies, fairness gaps across demographics, attack vulnerability, and poor privacy awareness.

- **Models.** Multiple Med-LVLMs evaluated (specific model list not given in the abstract -- "not extracted").
- **Method.** Benchmark construction and evaluation across five trustworthiness dimensions (trustfulness, fairness, safety, privacy, robustness), with both closed- and open-ended question formats.
- **Datasets.** CARES: ~41K QA pairs, 16 medical image modalities, 27 anatomical regions (their own constructed benchmark).
- **Experiments.** Evaluates Med-LVLMs on each of the five trustworthiness dimensions; analyzes fairness across demographic groups; tests robustness to attacks and privacy-awareness behavior.
- **Results.** "The models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups" and are "vulnerable to attacks" and show "a lack of privacy awareness" [abstract]. No specific numeric scores given -- "not extracted".
- **Conclusions.** Current Med-LVLMs are not trustworthy along multiple independent axes simultaneously; CARES is offered as a public benchmark/toolkit to measure this going forward.

> **Why it matters to us.** Establishes "trustfulness" (factual correctness under a trust lens) as one of five axes reviewers may expect us to at least acknowledge, even though our work only targets the correctness/selection axis via best-of-N + probe. Useful for explicitly scoping our probe ("we address trustfulness via correctness selection; CARES's other four axes are out of scope for this work") and as a possible future open+closed-format medical VQA eval set.

<small>Read from: abstract-only.</small>

#### Detecting and Evaluating Medical Hallucinations in Large Vision Language Models

*Jiawei Chen et al. · 2024 · arXiv preprint · [arXiv:2406.10185](https://arxiv.org/abs/2406.10185) · read priority 2*

**In one line.** Introduces Med-HallMark, the first dedicated medical-LVLM hallucination benchmark, plus the MediHall Score (a hierarchical severity/type-aware metric) and MediHallDetector, a multitask-trained model for hallucination detection.

- **Models.** Popular LVLMs evaluated as baselines, plus the paper's own MediHallDetector ("a novel Medical LVLM engineered for precise hallucination detection").
- **Method.** Constructs Med-HallMark (multi-task, multifaceted, hierarchically-categorized hallucination data); proposes MediHall Score, a hierarchical scoring metric weighting hallucination severity and type; trains MediHallDetector via multitask training.
- **Datasets.** Med-HallMark (their own new benchmark).
- **Experiments.** Establishes baselines for popular LVLMs on Med-HallMark using MediHall Score; evaluates MediHallDetector's detection performance against baselines.
- **Results.** "MediHall Score provides a more nuanced understanding of hallucination impacts compared to traditional metrics" and demonstrates "the enhanced performance of MediHallDetector" [abstract]. No specific numeric scores given -- "not extracted".
- **Conclusions.** Medical hallucination needs its own benchmark and a severity/type-aware metric (not a single flat hallucination rate), and a dedicated detector trained for this task outperforms general baselines.

> **Why it matters to us.** A benchmark/metric-design precedent for reporting hallucination severity rather than a flat rate -- relevant if we ever want to weight our probe's misses by clinical severity rather than treating all wrong answers equally. Also another example (alongside our own verifier) of a purpose-trained detector beating generic/zero-shot baselines in the medical domain, reinforcing our "training, not size, is the active ingredient" finding.

<small>Read from: abstract-only.</small>

#### Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs

*Jannik Kossen et al. · 2024 · arXiv preprint · [arXiv:2406.15927](https://arxiv.org/abs/2406.15927) · read priority 2*

**In one line.** Trains a cheap probe on a single generation's hidden states to approximate semantic entropy directly (no multiple generations, no external NLI model needed), retaining most of semantic entropy's hallucination-detection power at near-zero extra cost and generalizing better OOD than probes that directly predict accuracy.

- **Models.** Not named specifically in the abstract beyond "LLMs" (generic -- "not extracted").
- **Method.** Semantic Entropy Probes (SEPs): a probe trained on the hidden states of a single generation to predict the semantic entropy that would otherwise require 5-10x more compute (multiple samples + NLI clustering) to compute directly.
- **Datasets.** Multiple models and tasks (specific benchmark names "not extracted" from the abstract).
- **Experiments.** Compares SEP hallucination-detection performance to full semantic entropy and to probes trained to directly predict model accuracy; studies OOD generalization; ablates which token positions and model layers best capture semantic entropy.
- **Results.** "SEPs retain high performance for hallucination detection and generalize better to out-of-distribution data than previous probing methods that directly predict model accuracy" [abstract]. No specific numeric AUROC values given -- "not extracted".
- **Conclusions.** A single generation's hidden states already capture (an approximation of) semantic entropy, so a cheap probe recovers most of its hallucination-detection value without the 5-10x sampling+NLI overhead; probing for the uncertainty signal itself generalizes OOD better than probing directly for correctness.

> **Why it matters to us.** The closest methodological relative to our own probe in this whole category -- both are small probes reading hidden states of a single generation (specific layers/token positions tapped during generation, no extra forward pass) to avoid expensive sampling-based signals. Key difference: SEPs are trained to predict semantic entropy (an uncertainty proxy) whereas ours is trained directly on BCE correctness labels -- their finding that entropy-target probes generalize OOD better than accuracy-target probes is a concrete, citable caution about our probe's known OOD-transfer weakness (per-benchmark retraining requirement, "OOD-ness does not predict where it helps").

<small>Read from: abstract-only.</small>

#### Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation

*Lorenz Kuhn et al. · 2023 · ICLR 2023 (Spotlight) · [arXiv:2302.09664](https://arxiv.org/abs/2302.09664) · read priority 2*

**In one line.** The ICLR predecessor to the Nature semantic-entropy paper -- introduces semantic entropy for NLG, showing it predicts model accuracy on QA better than comparable baselines by clustering generations via shared meaning before computing entropy.

- **Models.** Not specified by name in the abstract ("large language models," generic -- "not extracted").
- **Method.** Same core idea as Farquhar et al. 2024 (semantic entropy over meaning-clusters built via linguistic-invariance/entailment), introduced here first: unsupervised, single-model, no fine-tuning.
- **Datasets.** Question-answering datasets used for ablation (not named in the abstract -- "not extracted").
- **Experiments.** Comprehensive ablation studies comparing semantic entropy's predictiveness of model accuracy to comparable uncertainty baselines.
- **Results.** "The semantic entropy is more predictive of model accuracy on question answering data sets than comparable baselines" [abstract]. No specific numbers given in the abstract -- "not extracted" (see the Farquhar Nature 2024 card for the follow-up work's numeric AUROC values).
- **Conclusions.** Semantic equivalence must be accounted for when measuring uncertainty in free-form text generation; semantic entropy is a practical unsupervised way to do this.

> **Why it matters to us.** The original semantic-entropy paper; pair with the Farquhar Nature 2024 card -- cite this one for priority/originality, the Nature paper for the larger-scale numbers. Same relation to us as that card: a training-free, whole-question uncertainty baseline distinct from our per-candidate correctness probe.

<small>Read from: abstract-only.</small>

#### The Internal State of an LLM Knows When It's Lying

*Amos Azaria and Tom Mitchell · 2023 · arXiv preprint · [arXiv:2304.13734](https://arxiv.org/abs/2304.13734) · read priority 2*

**In one line.** Trains a classifier on an LLM's hidden-layer activations to predict whether a statement (given or self-generated) is true or false, reaching 71-83% accuracy and beating a baseline that uses the LLM's own assigned sentence probability.

- **Models.** Multiple unspecified LLM base models (accuracy "depending on the LLM base model"; names "not extracted").
- **Method.** Supervised classifier trained on hidden-layer activations captured while the LLM reads/generates a statement, predicting P(statement is truthful); compared against a baseline using the LLM's own assigned sentence probability.
- **Datasets.** A curated set of true/false test sentences (half true, half false; dataset name "not extracted").
- **Experiments.** Trains/evaluates the activation-based classifier per base model; analyzes the relationship between classifier performance and LLM-assigned sentence probability, including the probability's dependence on sentence length and word frequency.
- **Results.** "Our trained classifier achieves an average of 71% to 83% accuracy labeling which sentences are true versus false, depending on the LLM base model" [abstract]; sentence probability "is also dependent on sentence length and the frequencies of words in the sentence," making the trained classifier "a more reliable approach" [abstract].
- **Conclusions.** LLM hidden states linearly encode a truthfulness signal that a small trained probe extracts more reliably than raw output probability, which is confounded by length/frequency -- direct precedent for hidden-state probing as a correctness signal.

> **Why it matters to us.** Closest early precedent to our own method's mechanism -- a small supervised classifier reading hidden activations to predict correctness/truthfulness, exactly analogous to our 918k-parameter MLP probe reading mean hidden states over generated tokens. Prior art that trained hidden-state probes beat raw-probability baselines, which is also one of our own findings (a trained 7B verifier beats zero-shot signals).

<small>Read from: abstract-only.</small>

#### Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback

*Katherine Tian et al. · 2023 · EMNLP 2023 (Camera Ready) · [arXiv:2305.14975](https://arxiv.org/abs/2305.14975) · read priority 2*

**In one line.** Finds verbalized confidence (asking RLHF-tuned LLMs like ChatGPT/GPT-4/Claude to state their confidence in words) is better calibrated than their raw conditional token probabilities, often halving expected calibration error.

- **Models.** RLHF-tuned LLMs: ChatGPT, GPT-4, Claude.
- **Method.** Evaluates and compares strategies for extracting confidence from RLHF-tuned LMs: raw conditional probability vs verbalized ("say a number/word for your confidence") elicitation, across several prompting variants.
- **Datasets.** TriviaQA, SciQ, TruthfulQA.
- **Experiments.** Measures calibration of verbalized vs conditional-probability confidence on the three QA benchmarks for each of the three RLHF-tuned model families.
- **Results.** "Verbalized confidences emitted as output tokens are typically better-calibrated than the model's conditional probabilities on the TriviaQA, SciQ, and TruthfulQA benchmarks, often reducing the expected calibration error by a relative 50%" [abstract].
- **Conclusions.** For RLHF-tuned LLMs specifically (where raw token probabilities are known to be poorly calibrated post-RLHF), asking the model to verbalize confidence in natural language is a simple, effective calibration strategy. The abstract frames calibration's purpose partly as "enabling deferral to an expert in cases of low-confidence predictions," but no deferral mechanism is built or evaluated -- the paper's actual method and results are calibration-only.

> **Why it matters to us.** Names and quantifies the "verbalized confidence" baseline family precisely -- a reviewer will ask why we didn't just prompt Lingshu-7B to state its own confidence instead of training a probe; this paper is evidence both for (verbalized confidence is a real, cheap alternative) and against (it's a black-box heuristic, not a trained per-candidate ranker) that choice, and gives us the relative-50%-ECE-reduction figure to cite as the bar a verbalized baseline would need to clear.

<small>Read from: abstract-only.</small>

#### Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

*Miao Xiong et al. · 2023 · ICLR 2024 · [arXiv:2306.13063](https://arxiv.org/abs/2306.13063) · read priority 2*

**In one line.** A systematic black-box benchmark of prompting/sampling/aggregation confidence-elicitation strategies finds LLMs are overconfident when verbalizing, calibration improves with scale, and white-box methods only narrowly beat the best black-box ones (0.522 to 0.605 AUROC).

- **Models.** Five widely-used LLMs including GPT-4 and LLaMA 2 Chat.
- **Method.** Defines a black-box confidence-elicitation framework with three components: prompting strategies for verbalized confidence, sampling methods for generating multiple responses, and aggregation techniques for computing consistency across them.
- **Datasets.** Five dataset types spanning commonsense and arithmetic reasoning (specific benchmark names "not extracted").
- **Experiments.** Benchmarks calibration and failure-prediction performance for prompting/sampling/aggregation combinations across 5 datasets x 5 LLMs; compares against white-box (logit-access) methods.
- **Results.** "LLMs, when verbalizing their confidence, tend to be overconfident" [abstract]; "as model capability scales up, both calibration and failure prediction performance improve" [abstract]; "comparisons with white-box methods indicate that while white-box methods perform better, the gap is narrow, e.g., 0.522 to 0.605 in AUROC" [abstract]; "none of these techniques consistently outperform others, and all investigated methods struggle in challenging tasks, such as those requiring professional knowledge" [abstract].
- **Conclusions.** Black-box confidence elicitation can approach white-box performance, but overconfidence and task-difficulty (especially professional-knowledge tasks) remain unsolved; no single strategy dominates.

> **Why it matters to us.** The "professional knowledge" failure mode they flag is exactly our setting (medical VQA); their finding that black-box consistency methods only narrowly trail white-box (logit/internal-state) ones calibrates how much our white-box probe should be expected to gain over a well-tuned black-box (e.g. self-consistency) baseline -- the 0.522->0.605 AUROC gap (~0.08) is a concrete "white-box premium" number to cite.

<small>Read from: abstract-only.</small>

#### Evaluating Object Hallucination in Large Vision-Language Models

*Yifan Li et al. · 2023 · EMNLP 2023 · [arXiv:2305.10355](https://arxiv.org/abs/2305.10355) · read priority 2*

**In one line.** The first systematic study of object hallucination in LVLMs, showing most LVLMs frequently describe objects not present in the image (especially ones that co-occur often with real image objects in the training distribution), and introducing POPE, a polling-based yes/no querying method for more stable, prompt-robust hallucination evaluation.

- **Models.** Several representative LVLMs (specific names not extracted from the abstract).
- **Method.** Systematic evaluation of object hallucination in LVLM image captioning/description; analysis of how visual-instruction object frequency and object co-occurrence with real image content predict hallucination; proposes POPE (Polling-based Object Probing Evaluation) -- asking the model direct yes/no questions about whether specific objects are present, rather than parsing free-form captions.
- **Datasets.** Built on existing LVLM captioning/instruction benchmarks (specific dataset names "not extracted" from the abstract).
- **Experiments.** Evaluates object hallucination rate across representative LVLMs; studies correlation between hallucination and (a) object frequency in visual instructions and (b) object co-occurrence with actual image content; compares POPE's stability/flexibility to prior caption-based hallucination metrics.
- **Results.** "They mostly suffer from severe object hallucination issue" [abstract]; "objects that frequently occur in the visual instructions or co-occur with the image objects, are obviously prone to be hallucinated by LVLMs" [abstract]; "existing evaluation methods might be affected by the input instructions and generation styles of LVLMs" [abstract]; POPE "can evaluate the object hallucination in a more stable and flexible way" [abstract]. No specific numeric hallucination rates given -- "not extracted".
- **Conclusions.** Object hallucination in LVLMs is driven substantially by training-distribution statistics (frequency/co-occurrence bias), not just genuine visual misperception, and evaluating it robustly requires a polling-based yes/no protocol (POPE) rather than parsing free-form generated captions.

> **Why it matters to us.** The canonical object-hallucination benchmark and the origin of "yes-bias"-style evaluation methodology that later medical-yes-bias papers (e.g. Chan 2026 in this category) build on; also a caution that the co-occurrence-bias mechanism it identifies (hallucinating statistically-likely-but-absent objects) is a language-prior effect distinct from, but possibly compounding, our own project's "reasoning hurts perception" finding. Gives Leo the terms "object hallucination" and "POPE" a reviewer will expect him to know.

<small>Read from: abstract-only.</small>

#### Language Models (Mostly) Know What They Know

*Saurav Kadavath et al. · 2022 · arXiv preprint · [arXiv:2207.05221](https://arxiv.org/abs/2207.05221) · read priority 2*

**In one line.** Shows large LMs are well-calibrated on MCQ/true-false in the right format, and can self-evaluate their own answers via P(True) and predict in advance whether they'll know an answer via P(IK).

- **Models.** Anthropic LM family across a range of sizes (specific sizes not in abstract -- "not extracted").
- **Method.** Sample an answer, then ask the model to predict P(True) that its own answer is correct; separately train models to predict P(IK) ("I know") the probability of getting a question right, without seeing any proposed answer.
- **Datasets.** Diverse multiple choice and true/false questions, plus open-ended sampling tasks including math word problems (specific benchmark names not in abstract -- "not extracted").
- **Experiments.** Calibration of raw MCQ/true-false probabilities vs model scale; P(True) self-evaluation with/without showing the model several of its own other samples first; P(IK) training and cross-task generalization, plus response to relevant context/hints.
- **Results.** "Larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format" [abstract]; "encouraging performance, calibration, and scaling for P(True)" that "further improves when we allow models to consider many of their own samples before predicting the validity of one specific possibility" [abstract]; P(IK) "partially generalize[s] across tasks, though they struggle with calibration of P(IK) on new tasks" [abstract]. No specific numeric values in the abstract -- "not extracted".
- **Conclusions.** Self-evaluation (verbalized/introspective confidence) is a viable, scalable confidence signal, especially when the model sees multiple of its own samples first -- a precursor to sampling-based/consistency confidence methods.

> **Why it matters to us.** Early evidence for letting a model see multiple of its own samples before judging correctness, directly analogous to our best-of-N setup -- except we read hidden states with a trained probe rather than asking the model to verbalize P(True). Names the zero-training alternative baseline: "ask the generator itself, self-evaluating over its own N samples."

<small>Read from: abstract-only.</small>

#### On Calibration of Modern Neural Networks

*Chuan Guo et al. · 2017 · ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) · read priority 2*

**In one line.** Shows modern deep classifiers are poorly calibrated despite high accuracy, and that a single-parameter temperature scaling fixes most of the miscalibration cheaply.

- **Models.** Modern CNN image/document classifiers of the era (specific architectures not named in the abstract -- "not extracted").
- **Method.** Empirically measures calibration (reliability diagrams, ECE) across architectures/training choices, then evaluates post-hoc calibration methods, showing temperature scaling -- one learned scalar dividing the logits before softmax -- is the simplest effective fix.
- **Datasets.** Image and document classification datasets (not named in the abstract -- "not extracted").
- **Experiments.** Studies effect of depth, width, weight decay, and batch normalization on calibration; compares post-hoc calibration methods on state-of-the-art architectures.
- **Results.** "We discover that modern neural networks, unlike those from a decade ago, are poorly calibrated" and temperature scaling is "surprisingly effective at calibrating predictions" [abstract]. No specific ECE numbers given in the abstract -- "not extracted".
- **Conclusions.** Calibration degrades with modern architectural trends (depth/width/no weight decay/batch norm), but one learned temperature parameter recovers most of the loss cheaply -- the standard baseline for post-hoc calibration cited across the field.

> **Why it matters to us.** The foundational reference for "calibration" and for temperature scaling as the default cheap baseline; whenever we report our probe's calibration (not just its ranking/AUROC), this is the citation and the comparison method (temperature-scale our probe's raw score) a reviewer will expect.

<small>Read from: abstract-only.</small>


#### Also in this area (11), in brief

- **VIHD: Visual Intervention-based Hallucination Detection for Medical Visual Question Answering** — Jiayi Chen et al. (2026), [arXiv:2605.20772](https://arxiv.org/abs/2605.20772). Improves on generic prompt-perturbation hallucination detection by locating the decoder layers most dependent on visual tokens, masking visual tokens specifically at those layers to calibrate the semantic-entropy distribution, and beating prior state-of-the-art on three medical VQA benchmarks.<br><small>*For us:* Same family as UniVRSE (vision-grounded semantic entropy for medical VLMs) but locates the mechanism inside the decoder layers rather than only contrasting input pairs -- relevant prior art if we ever want to argue our probe's choice of layers 18/20/22 is principled (they use a data-driven layer-selection step, VDP, whereas we picked 3 fixed layers); worth a closer read for how they justify layer selection.</small>
- **Hallucination Detection and Correction in Medical VLMs via Counter-Evidence Verification** — Nan Zhou et al. (2026), [arXiv:2606.18609](https://arxiv.org/abs/2606.18609). A training-free, plug-and-play framework (CoEV) that verifies whether each generated statement is actually supported by grounded visual evidence, both detecting and correcting medical VLM hallucinations, improving detection PR-AUC/ROC-AUC by ~3% and cutting hallucination rate by >11.9% on report generation.<br><small>*For us:* A training-free alternative worth naming when justifying why we trained a probe instead: CoEV needs no training but does need a visual-grounding module and evidence-region localizer (extra machinery per inference), a different cost tradeoff than our fixed, cheap forward-pass probe. Also gives a concrete external hallucination-rate reduction (>11.9%) and VQA-accuracy boost to compare our own +0.0736/+0.0820 macro gains against, in the same medical-VQA space.</small>
- **Detecting Clinical Hallucinations in LVLMs via Counterfactual Visual Grounding Uncertainty** — Xiao Song et al. (2026), [arXiv:2606.28520](https://arxiv.org/abs/2606.28520). A training/access-free hallucination detector for LVLM clinical outputs that grounds extracted entities on the image via a medical-adapted Qwen-VL grounder, then contrasts factual vs counterfactual (perturbed-entity) grounding confidence and overlap to produce an entity-level uncertainty score, without needing the target LVLM's hidden states.<br><small>*For us:* A "black-box, external verifier" contrast to our own "white-box, internal-state" probe -- it needs no access to the target model's hidden states at all (uses a separate grounding model instead), an important axis to contrast when describing our probe's requirement of hidden-state access as a deployment constraint (we need the generator's own activations; this method works even for closed/API-only target models).</small>
- **UniVRSE: Unified Vision-conditioned Response Semantic Entropy for Hallucination Detection in Medical Vision-Language Models** — Zehui Liao et al. (2025), [arXiv:2503.20504](https://arxiv.org/abs/2503.20504). Adapts semantic entropy for medical VLMs by contrasting the semantic distribution from the original image-text pair against a visually-distorted counterpart (since plain semantic entropy is unreliable in medical VLMs due to strong language-prior overconfidence), and introduces ALFA, a fine-grained factual-consistency metric for building ground-truth hallucination labels.<br><small>*For us:* Independently confirms that generic (text-domain) uncertainty methods degrade in medical VLMs due to language-prior overconfidence -- consonant with our own "reasoning hurts perception" and "training, not size, is the active ingredient" findings, and adds a concrete mechanism (visual-distortion contrasting) other than training a probe for making an uncertainty signal medical-domain-aware.</small>
- **HEDGE: Hallucination Estimation via Dense Geometric Entropy for VQA with Vision-Language Models** — Sushant Gautam et al. (2025), [arXiv:2511.12693](https://arxiv.org/abs/2511.12693). A unified hallucination-detection pipeline (HEDGE) combining visual perturbation, semantic clustering (NLI- and embedding-based), and uncertainty metrics, tested on VQA-RAD and KvasirVQA-x1 with LLaVA-Med/MedGemma/Qwen2.5-VL, finding the VASE metric with embedding clustering most robust and that architecture (dense vs restricted visual tokenization) strongly affects detectability.<br><small>*For us:* Directly relevant since it evaluates on Qwen2.5-VL and includes VQA-RAD, both overlapping our own setup (we replicate on Qwen2.5-VL-7B and use VQA-RAD-open as one of our 8 benchmarks) -- worth a closer read to see if their per-architecture detectability finding (Qwen2.5-VL easiest to detect hallucinations in) predicts anything about where our own probe should work best. Their n~10-15 "moderate sampling budget" finding is a useful cross-check against our own N=8 best-of-N choice.</small>
- **Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models** — Xiao Liang et al. (2025), [arXiv:2507.09209](https://arxiv.org/abs/2507.09209). Proposes Expert-CFG, a training-free expert-in-the-loop framework that uses uncertainty estimation to flag unreliable MedVLM outputs, retrieves references so a human expert can highlight key terms, then applies classifier-free guidance to steer token embeddings toward the expert's highlights -- a 4.2B-parameter model with this framework beats 13B state-of-the-art models on three medical VQA benchmarks.<br><small>*For us:* Uses uncertainty estimation only as a trigger/flag for when to bring in an expert and additional reference retrieval -- this is a human-in-the-loop escalation design (uncertainty estimation triggering expert intervention), so its expert-escalation component is out of scope for us by standing rule; we card only its uncertainty-flagging mechanism as a training-free "unreliable output" detector, distinct from our own always-answering cascade (which escalates to a bigger frozen model, never a human).</small>
- **Calibration-Aware Prompt Learning for Medical Vision-Language Models** — Abhishek Basu et al. (2025), [arXiv:2509.15226](https://arxiv.org/abs/2509.15226). CalibPrompt, the first prompt-tuning framework to calibrate medical VLMs, optimizes a small set of learnable prompts under scarce labelled data using an accuracy-confidence alignment regularizer plus an angular-separation loss on text features, consistently improving calibration without hurting clean accuracy across four Med-VLMs and five medical imaging datasets.<br><small>*For us:* A lightweight-adaptation calibration baseline (prompt tuning only, backbone frozen) analogous in spirit to our own frozen-backbone approach -- worth naming alongside Platt scaling and verbalized-confidence fine-tuning (Senoglu 2026) as the spectrum of "how much of the model do you have to touch to get good confidence," with our probe sitting at "touch nothing, train only a tiny external head" -- even lighter than prompt tuning since we add no prompt tokens and don't touch the frozen generator at inference beyond reading its already-computed hidden states.</small>
- **Uncertainty-Aware Evaluation for Vision-Language Models** — Vasily Kostumov et al. (2024), [arXiv:2402.14418](https://arxiv.org/abs/2402.14418). Benchmarks 20+ VLMs on multiple-choice VQA using conformal prediction for uncertainty quantification, finding model uncertainty is not aligned with accuracy -- the most accurate models can also be the most uncertain -- and that uncertainty correlates with the underlying LLM component.<br><small>*For us:* A caution against any claim that our probe's gains simply track the strong model's accuracy gains -- this paper's headline finding (accuracy and uncertainty can be uncorrelated or even inversely related) argues for always reporting our probe's calibration/AUROC separately from raw accuracy deltas. Also relevant to our multi-backbone replication (Lingshu-7B, Qwen2.5-VL-7B, MedGemma-4b), since it suggests probe behavior could shift with the LLM backbone even at fixed vision setup.</small>
- **VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model via Uncertainty Estimation** — Ruiyang Zhang et al. (2024), [arXiv:2411.11919](https://arxiv.org/abs/2411.11919). First uncertainty-based (no ground-truth/pseudo-label needed) LVLM hallucination detector, measuring response variance across semantically-perturbed visual and textual prompts and clustering by semantic content to compute entropy, outperforming strong baselines across 10 LVLMs and 4 benchmarks.<br><small>*For us:* A training-free, multimodal semantic-entropy baseline worth citing; it explicitly tests both MCQ and free-form formats, giving it direct bearing on our own "answer format determines whether routing signals work at all" finding (AUROC ~0.6 MCQ vs ~0.87 free text) -- worth checking in a full read whether VL-Uncertainty shows the same format split.</small>
- **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model** — Kenneth Li et al. (2023), [arXiv:2306.03341](https://arxiv.org/abs/2306.03341). Shows LLaMA models have internal directions encoding truthfulness that, when nudged at inference time across a limited set of attention heads, substantially raise TruthfulQA performance (Alpaca 32.5%->65.1%) with minimal added cost and little training data.<br><small>*For us:* An internal-state method, but an intervention (changes model behavior) rather than a detector (scores an existing candidate) -- worth distinguishing explicitly from our probe, which only scores/selects among already-generated candidates and never edits activations. Useful as a "what else you could do with hidden states" contrast in related work.</small>
- **Discovering Latent Knowledge in Language Models Without Supervision** — Collin Burns et al. (2022), [arXiv:2212.03827](https://arxiv.org/abs/2212.03827). Introduces Contrast-Consistent Search (CCS), an unsupervised probe that finds a hidden-activation direction tracking truth by enforcing logical consistency between a statement and its negation, recovering latent "knowledge" without labels.<br><small>*For us:* Prior art for internal-state probing and for the core premise our probe relies on -- hidden states carry a linearly recoverable correctness/truth signal separate from the generated text. CCS is unsupervised while our probe is supervised (BCE-trained on labelled correct/incorrect); worth citing as the unsupervised end of the same spectrum our probe sits on.</small>

## 3.7 Medical Vision-Language Models: The Generators We Use and the Ones We Compare Against

A vision-language model (VLM) answers a text question about an image by chaining three pieces: a vision encoder that turns pixels into a sequence of patch embeddings, a projector/connector that maps those embeddings into the language model's own embedding space, and a decoder-only language model that reads the projected image tokens plus the text prompt and generates an answer token by token. 'Hidden size' is the width of the vector representing every token at each layer of the language model (3584 for Lingshu-7B and Qwen2.5-VL-7B; 2560 commonly cited for MedGemma-4b-it's Gemma 3 backbone, though that pair of numbers was not independently located in the Gemma 3 arXiv text itself -- see that card); 'number of layers' is how many transformer blocks a token's representation passes through end to end (28 for the 7B Qwen line, confirmed in Qwen2.5-VL's Table 1; 34 commonly cited for Gemma 3 4B). Both matter directly to us: our probe reads the hidden state at layers 18/20/22 of Lingshu-7B's 28 language-model layers -- the back two-thirds of the LLM, never the vision tower. This category covers, in order: the shared encoder->projector->LLM architecture lineage (SigLIP, LLaVA/LLaVA-1.5, Qwen2-VL->Qwen2.5-VL, Gemma 3); the three models we actually run as generators (Lingshu-7B, Qwen2.5-VL, MedGemma-4b-it / MedGemma 1.5); and the wider 2026 medical-VLM field (MedVLThinker, Fleming-VL, Hulu-Med, InternVL3, and two independent benchmarking studies -- one of which rates Lingshu-32B top-tier) so that our own numbers have external reference points instead of floating alone.

#### ★ Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning

*LASA Team et al. · 2025 · arXiv preprint (Technical Report, 53 pages) [comment field] · [arXiv:2506.07044](https://arxiv.org/abs/2506.07044) · read priority 1 · **PDF in `papers/`***

**In one line.** Lingshu (7B/32B), a medical MLLM built on Qwen2.5-VL via a 4-stage pipeline (shallow align -> deep align -> instruction tuning -> GRPO RL), plus MedEvalKit, the unified medical eval harness the whole project depends on.

- **Models.** Lingshu-7B and Lingshu-32B, built on Qwen2.5-VL-7B-Instruct / Qwen2.5-VL-32B-Instruct: "Lingshu is built upon the Qwen2.5-VL model architecture" using "two parameter-scaled variants of Qwen2.5-VL, i.e., 7B-Instruct and 32B-Instruct." [html] Base architecture described only generically as "a large language model (LLM), a vision encoder, and an MLP-based projector" [html] -- exact hidden size/layer count are not restated in this paper (see the Qwen2.5-VL card for those, since Lingshu-7B inherits them unchanged).
- **Method.** Four sequential training stages [html]: (1) Medical Shallow Alignment (~927K samples; vision encoder + projector tuned, LLM frozen); (2) Medical Deep Alignment (~4.1M samples; full model unfrozen, medical + general captions); (3) Medical Instruction Tuning (~7.1M samples; full multimodal + text instruction data); (4) Medical-oriented Reinforcement Learning (~100K samples; GRPO with verifiable rewards). Also introduces MedEvalKit, "a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment" [abstract].
- **Datasets.** Curated medical multimodal dataset spanning imaging, medical text, and general-domain data; captions, VQA, and reasoning samples synthesized [abstract]. Evaluated on medical multimodal QA (Table 6: MMMU-Med, VQA-RAD, SLAKE, PathVQA, PMC-VQA, OmniMedVQA, MedXpertQA-MM), text-only medical QA (Table 7), and report generation (Table 8: MIMIC-CXR, CheXpert Plus, IU-Xray).
- **Experiments.** Evaluates Lingshu-7B/32B against other open-source MLLMs on three task families (multimodal QA, text QA, report generation) using the paper's own MedEvalKit harness [abstract].
- **Results.** Table 6 (medical multimodal benchmarks), Lingshu-7B: MMMU-Med 54.0, VQA-RAD 67.9, SLAKE 83.1, PathVQA 61.9, PMC-VQA 56.3, OmniMedVQA 82.9, MedXpertQA-MM 26.7, Average 61.8 [Table 6, html]. Lingshu-32B: VQA-RAD 76.5, SLAKE 89.2, PathVQA 65.9, PMC-VQA 57.9, OmniMedVQA 83.4, MedXpertQA-MM 30.9, Average 66.6 [Table 6, html] -- "Lingshu-32B attains a state-of-the-art average score of 66.6 across all benchmarks" [html]. Table 7 (text-only): Lingshu-7B average 52.8, Lingshu-32B average 61.8 [Table 7, html].
- **Conclusions.** "Lingshu consistently outperforms the existing open-source multimodal models on most tasks" [abstract]; scaling 7B->32B gives consistent gains across all three task families (multimodal QA, text QA, report generation) [html].

> **Why it matters to us.** This is our main generator (frozen, unmodified) and the source of MedEvalKit, the harness our repo's baseline-eval-protocol depends on. Table 6 is our numeric anchor table -- the only place our own Lingshu-7B replication numbers can be checked against a published reference on VQA-RAD/SLAKE/PathVQA/PMC-VQA/OmniMedVQA/MedXpertQA-MM. Also worth flagging: stage 4 is already GRPO reinforcement learning with verifiable rewards, so Lingshu-7B arrives at our pipeline already RL-shaped before we ever apply best-of-N + probe selection -- any claim that our method adds signal beyond "what RL already gave the model" should account for this.

<small>Read from: html.</small>

#### ★ MedGemma Technical Report

*Andrew Sellergren et al. · 2025 · arXiv preprint [comment field: "Fix references"] · [arXiv:2507.05201](https://arxiv.org/abs/2507.05201) · read priority 1 · **PDF in `papers/`***

**In one line.** MedGemma (Gemma 3 4B/27B + MedSigLIP) technical report: explicitly removed PathVQA and MedVQA from training over data-quality concerns and re-split VQA-RAD to fix train/test image contamination.

- **Models.** MedGemma 4B (Gemma 3 4B + MedSigLIP, medically fine-tuned) and MedGemma 27B; MedSigLIP, "a medically-tuned vision encoder derived from SigLIP" [abstract], also released standalone.
- **Method.** Fine-tunes Gemma 3 base models on medical image-text and text data; separately trains MedSigLIP as a medical vision encoder that "powers the visual understanding capabilities of MedGemma" [abstract].
- **Datasets.** VERBATIM [Section 2.1.1]: "Additionally, we and others have identified potential data quality issues in PathVQA and MedVQA. Thus, we removed them from the training dataset." VERBATIM [Section 3.4]: "For VQA-RAD, we used splits from Yang et al. (2024) to avoid the train/test image contamination present in the original splits." VERBATIM [Table 9 footnote, Section 4]: "Additionally, the original VQA-RAD test set includes some duplicated images in the train and test sets (with different questions). As such we have previously described our own splits to avoid this contamination." PathVQA, PMC-VQA, and OmniMedVQA do not appear with numbers in any table of this paper.
- **Experiments.** Evaluated on SLAKE and VQA-RAD (Table 9) and MedXpertQA text-only/multimodal-only splits (Table 4), against Gemma 3 (base, unfinetuned), BiomedGPT-B, LLaVA-Med (BioMedCLIP), Med-Gemini, Gemini 2.5 Flash/Pro, and o3.
- **Results.** SLAKE [Table 9]: MedGemma 4B 72.3 overall token F1 / 63.3 open-ended token recall / 87.6 closed-ended accuracy; Gemma 3 4B 40.2 / 33.3 / 53.0; Gemma 3 27B 42.5 / 30.8 / 64.5; BiomedGPT-B 85.2 / 87.1 / 89.9; LLaVA-Med (BioMedCLIP) -- / 87.1 / 86.8; Med-Gemini 75.8 / 72.2 / 84.6. VQA-RAD test split [Table 9]: MedGemma 4B 49.9 overall token F1 / 69.1 closed-Q&A accuracy; Gemma 3 4B 33.6 / 48.7; Gemma 3 27B 42.7 / 59.4; Med-Gemini 50.1 / 69.7. MedXpertQA [Table 4]: MedGemma 4B 14.2% text-only / 24.4% multimodal-only; Gemma 3 4B 11.6% / 22.3%; MedGemma 27B 25.7% text-only (multimodal-only not located); Gemma 3 27B 15.7% / 29.8%. Abstract-level deltas vs base Gemma 3 models: "2.6-10% improvement on medical multimodal question answering, 15.5-18.1% improvement on chest X-ray finding classification, and 10.8% improvement on agentic evaluations" [abstract]; fine-tuning "reduc[es] errors in electronic health record information retrieval by 50%" [abstract].
- **Conclusions.** MedGemma "significantly exceed[s] the performance of similar-sized generative models and approach[es] the performance of task-specific models, while maintaining the general capabilities of the Gemma 3 base models" [abstract].

> **Why it matters to us.** Third generator in our replication set (MedGemma-4b-it). The verbatim data-quality note is important standing evidence for us: MedGemma independently reached the same conclusion our project treats as a landmine for MMMU-Medical (CLAUDE.md SS0) -- PathVQA and VQA-RAD have documented contamination/quality problems. Concretely, because MedGemma never trained (and apparently never evaluated) on PathVQA, PMC-VQA, or OmniMedVQA, any 3-way Lingshu/Qwen2.5-VL/MedGemma comparison we run on those three benchmarks compares a model that saw the data in training against one that deliberately never did -- a confound that should be stated explicitly wherever we report that comparison.

<small>Read from: html.</small>

#### ★ Qwen2.5-VL Technical Report

*Shuai Bai et al. · 2025 · arXiv preprint · [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) · read priority 1 · **PDF in `papers/`***

**In one line.** Technical report for Qwen2.5-VL, the general-domain backbone Lingshu-7B/32B are medically fine-tuned from; defines the ViT -> MLP-merger -> LLM architecture and confirms the 7B config (hidden 3584, 28 LLM layers).

- **Models.** Qwen2.5-VL in three sizes: 2B, 7B, 72B [abstract].
- **Method.** A native dynamic-resolution Vision Transformer (ViT) "trained from scratch" with Window Attention, feeding an MLP-based merger, feeding the Qwen2.5 LLM. Merger mechanism, VERBATIM [html]: "Instead of directly using the raw patch features extracted by the Vision Transformer (ViT), we first group spatially adjacent sets of four patch features. These grouped features are then concatenated and passed through a two-layer multi-layer perceptron (MLP) to project them into a dimension that aligns with the text embeddings used in the LLM." Also introduces absolute time encoding for video and studies scaling laws for LVLMs across model size and data [abstract].
- **Datasets.** Not extracted (broad multimodal pretraining + instruction data; not itemized in the fetched sections).
- **Experiments.** Compares Qwen2.5-VL-72B against GPT-4o and Claude 3.5 Sonnet on document/diagram understanding and other multimodal benchmarks [abstract].
- **Results.** Table 1 (7B config) [html]: ViT hidden size 1280, LLM hidden size 3,584, LLM layers 28, ViT layers 32, KV heads 4. Abstract: "the flagship Qwen2.5-VL-72B model matches state-of-the-art models like GPT-4o and Claude 3.5 Sonnet, particularly excelling in document and diagram understanding" [abstract].
- **Conclusions.** Native dynamic resolution + window attention reduce compute overhead while preserving native-resolution perception, without the normalization tricks earlier fixed-resolution models needed [abstract].

> **Why it matters to us.** This IS the backbone architecture of our main generator: Lingshu-7B = Qwen2.5-VL-7B-Instruct + medical fine-tuning, so hidden size 3584 / 28 LLM layers (confirmed here, Table 1) is literally the architecture our probe taps at layers 18/20/22 of 28. Read this before the probe method itself: the ViT (32 layers, hidden 1280) produces patch features, the MLP merger (2-layer, 2x2 patch grouping) projects them into the LLM's 3584-dim token-embedding space, and our probe reads the LLM's own hidden states during generation -- it never touches the vision tower's activations directly.

<small>Read from: html.</small>

#### ★ How Far Have Medical Vision-Language Models Come? A Comprehensive Benchmarking Study

*Che Liu et al. · 2025 · arXiv preprint (Technical report [comment field]) · [arXiv:2507.11200](https://arxiv.org/abs/2507.11200) · read priority 1 · **PDF in `papers/`***

**In one line.** Independent benchmarking of general-purpose vs medically-specialized VLMs (3B-72B): general models often match or beat medical-specific ones, reasoning consistently underperforms understanding, and no model reaches a clinical-deployment reliability bar.

- **Models.** Open-source general-purpose VLMs and medically specialized VLMs, 3B to 72B parameters, including Qwen2.5-VL-32B/72B and Lingshu-7B/32B [Table 1, html].
- **Method.** Splits evaluation into an "understanding" component (visual recognition / factual retrieval) and a "reasoning" component, scored separately per model per benchmark [abstract].
- **Datasets.** Eight benchmarks named in the abstract as "MedXpert, OmniMedVQA, PMC-VQA, PathVQA, MMMU, SLAKE, and VQA-RAD" [abstract] (note: abstract says "eight" but names seven; not resolved further here).
- **Experiments.** Table 1: overall performance of each model across all benchmarks. Tables 2-3: understanding vs. reasoning breakdown [html].
- **Results.** Table 1 [html]: Qwen2.5-VL-32B -- VQA-RAD 0.7041, SLAKE 0.7326, PathVQA 0.6573, PMC-VQA 0.5331, OmniMedVQA 0.6383, MMMU 0.7075. Qwen2.5-VL-72B -- VQA-RAD 0.6835, SLAKE 0.7799, PathVQA 0.6597, PMC-VQA 0.5577, OmniMedVQA 0.6656, MMMU 0.7078. Lingshu-7B -- VQA-RAD 0.6574, SLAKE 0.8034, PathVQA 0.7192, PMC-VQA 0.5213, OmniMedVQA 0.6436, MMMU 0.6389. Lingshu-32B -- VQA-RAD 0.6295, SLAKE 0.7718, PathVQA 0.7609, PMC-VQA 0.5365, OmniMedVQA 0.7662, MMMU 0.6345 [Table 1, html]. MedGemma was not evaluated in this study [checked via html fetch, not found in any table].
- **Conclusions.** VERBATIM [abstract]: "large general-purpose models already match or surpass medical-specific counterparts on several benchmarks, demonstrating strong zero-shot transfer from natural to medical images"; "reasoning performance is consistently lower than understanding, highlighting a critical barrier to safe decision support"; "No model yet reaches the reliability threshold for clinical deployment, underscoring the need for stronger multimodal alignment and more rigorous, fine-grained evaluation protocols."

> **Why it matters to us.** Gives us external reference points for Lingshu-7B/32B and Qwen2.5-VL-32B/72B on exactly the benchmarks we overlap with (VQA-RAD, SLAKE, PathVQA, PMC-VQA, OmniMedVQA, MMMU), independently measured under a different harness than our MedEvalKit-faithful pipeline -- useful as a cross-check, but these appear to be closed-form/accuracy-style scores on datasets we mostly run open-ended, so they are NOT directly poolable with our open-text macro numbers. Their finding that reasoning underperforms understanding across the board is independent, external replication of our own retrospective finding #1 ("reasoning hurts perception, and the reasoning-heavy 'gain' is an answer-format effect") -- worth citing as corroboration in our reports.

<small>Read from: html.</small>

#### MedGemma 1.5 Technical Report

*Andrew Sellergren et al. · 2026 · arXiv preprint · [arXiv:2604.05081](https://arxiv.org/abs/2604.05081) · read priority 2*

**In one line.** MedGemma 1.5 4B extends MedGemma with 3D CT/MRI volumes, whole-slide pathology, multi-timepoint chest X-ray, and document/EHR understanding, plus text-only clinical-reasoning gains.

- **Models.** MedGemma 1.5 4B [abstract].
- **Method.** New training data plus "long-context 3D volume slicing, and whole-slide pathology sampling" added on top of the MedGemma 1 4B architecture [abstract].
- **Datasets.** 3D MRI/CT condition classification, whole-slide pathology images, multi-timepoint chest X-ray series, and lab-report/EHR extraction datasets ("EHR Datasets 2, 3, 4, and Mendeley Clinical Laboratory Test Reports") [abstract].
- **Experiments.** Compares MedGemma 1.5 4B against MedGemma 1 4B across the new modalities plus existing text benchmarks (MedQA, EHRQA) [abstract].
- **Results.** VERBATIM deltas vs MedGemma 1 4B [abstract]: "improving 3D MRI condition classification accuracy by 11% and 3D CT condition classification by 3% (absolute improvements)"; "a 47% macro F1 gain" in whole-slide pathology; "a 35% increase in Intersection over Union on chest X-rays" for anatomical localization; "4% macro accuracy for longitudinal (multi-timepoint) chest x-ray analysis"; "5% on MedQA accuracy and 22% on EHRQA accuracy"; "an average of 18% macro F1 on 4 different lab report information extraction datasets."
- **Conclusions.** VERBATIM [abstract]: "MedGemma 1.5 serves as a robust, open resource for the community, designed as an improved foundation on which developers can create the next generation of medical AI systems."

> **Why it matters to us.** Not a benchmark we currently use (3D volumes / pathology whole-slide / EHR text, not 2D open-text VQA), but it is the direct successor of our third generator (MedGemma-4b-it) and shows the MedGemma line was still actively developed as recently as April 2026 -- a reminder to re-check before describing MedGemma-4b-it as the current-best checkpoint in that family in any 2026 write-up.

<small>Read from: abstract-only.</small>

#### MedRCube: A Multidimensional Framework for Fine-Grained and In-Depth Evaluation of MLLMs in Medical Imaging

*Zhijie Bao et al. · 2026 · arXiv preprint · [arXiv:2604.13756](https://arxiv.org/abs/2604.13756) · read priority 2*

**In one line.** Fine-grained, multidimensional MLLM medical-imaging benchmark across 33 models that independently rates Lingshu-32B top-tier, and finds a significant positive association between 'shortcut' reasoning behavior and diagnostic-task performance.

- **Models.** 33 MLLMs benchmarked, explicitly including "Lingshu-32B" [abstract]; other models not individually named in the abstract.
- **Method.** "A two-stage systematic construction pipeline" for multidimensional, fine-grained, in-depth evaluation, instantiated as the MedRCube benchmark; includes "a credibility evaluation subset to quantify reasoning credibility" [abstract].
- **Datasets.** Not extracted -- the abstract does not name specific VQA benchmarks/datasets within MedRCube (abstract-only provenance; no full-text fetch performed for this non-core paper).
- **Experiments.** Benchmarks 33 MLLMs on the MedRCube framework, plus a separate credibility-evaluation subset relating shortcut behavior to diagnostic performance [abstract].
- **Results.** VERBATIM [abstract]: "we benchmark 33 MLLMs, Lingshu-32B achieve top-tier performance"; "a highly significant positive association between shortcut behavior and diagnostic task performance." No specific numeric scores given in the abstract; not extracted.
- **Conclusions.** The shortcut/performance association "rais[es] concerns for clinically trustworthy deployment" [abstract] -- good diagnostic scores can coexist with (or even come from) unreliable reasoning shortcuts.

> **Why it matters to us.** Independent, very recent (April 2026) confirmation that Lingshu-32B -- the model whose 'reasoning' mode is our project's baseline, per CLAUDE.md SS0 -- is genuinely strong by outside measurement, not just our own MedEvalKit replication. The shortcut/diagnostic-performance association is a useful caution flag structurally similar to our own retrospective finding #1 (reasoning hurts perception; the reasoning 'gain' is partly an answer-format effect) -- both papers independently warn that a model's good score can come from the wrong mechanism, and are worth citing together.

<small>Read from: abstract-only.</small>

#### MedVLThinker: Simple Baselines for Multimodal Medical Reasoning

*Xiaoke Huang et al. · 2025 · ML4H'25 [comment field: "Accepted by ML4H'25"] · [arXiv:2508.02669](https://arxiv.org/abs/2508.02669) · read priority 2*

**In one line.** Open recipe (SFT vs RLVR) for reasoning-centric medical LMMs on Qwen2.5-VL 3B/7B/32B; RLVR beats SFT, and (counter-intuitively) text-only reasoning data helps the multimodal model more than image-text reasoning data does.

- **Models.** Qwen2.5-VL model family, 3B and 7B (scaled to 32B in one experiment) [abstract].
- **Method.** Two training paradigms compared: (1) Supervised Fine-Tuning (SFT) on distilled reasoning traces; (2) Reinforcement Learning with Verifiable Rewards (RLVR) based on final-answer correctness [abstract].
- **Datasets.** Curated text-only and image-text medical data, "filtered according to varying levels of reasoning difficulty" [abstract]; evaluated on "six medical QA benchmarks" [abstract] (not individually named in the abstract).
- **Experiments.** Ablates SFT vs RLVR, and (within RLVR) text-only-data training vs multimodal image-text-data training, across model sizes [abstract].
- **Results.** VERBATIM [abstract]: "RLVR consistently and significantly outperforms SFT"; "a key, counter-intuitive finding is that training on our curated text-only reasoning data provides a more substantial performance boost than training on multimodal image-text data"; "Our best open 7B model, trained using the RLVR recipe on text-only data, establishes a new state-of-the-art on existing public VQA benchmarks, surpassing all previous open-source medical LMMs"; scaling to 32B "achieves performance on par with the proprietary GPT-4o." No per-benchmark numeric table extracted (abstract-only).
- **Conclusions.** RLVR is the more effective training paradigm for medical multimodal reasoning, and text-only reasoning data transfers surprisingly well into multimodal gains [abstract].

> **Why it matters to us.** This is plausibly the lineage of our project's earlier era: our repo's CLAUDE.md June-2026 'MedVLThinker era' generators (the 7B/32B RL_m23k weights at /data/dan/weights/MedVLThinker-*) are named after this project -- worth confirming the exact checkpoint provenance against this paper before citing it as our historical generator's origin. Its finding that RLVR text-only data transfers into multimodal gains is a useful prior-art parallel to our own retrospective finding #3 ("training, not size, is the active ingredient in verification"), here applied to the generator side rather than the verifier side.

<small>Read from: abstract-only.</small>

#### Fleming-VL: Towards Universal Medical Visual Reasoning with Multimodal LLMs

*Yan Shu et al. · 2025 · arXiv preprint · [arXiv:2511.00916](https://arxiv.org/abs/2511.00916) · read priority 2*

**In one line.** Unified medical MLLM spanning 2D images, 3D volumes, and video, built via data-centric pretraining + SFT + GRPO; claims state-of-the-art across medical VQA, video QA, and 3D understanding benchmarks.

- **Models.** Fleming-VL, released "in multiple model scales" [abstract].
- **Method.** Data-centric: (1) scale up pretraining with long-context natural + medical-specific data; (2) fine-tune with rare medical data including "holistic video analysis and underrepresented 2D modalities such as ultrasound and dermoscopy images"; (3) extend evaluation to 3D volumetric and video benchmarks. Trained via supervised fine-tuning (SFT) and group relative policy optimization (GRPO) [abstract].
- **Datasets.** 2D images, 3D volumetric scans, temporal video sequences; ultrasound and dermoscopy specifically called out as underrepresented modalities addressed [abstract].
- **Experiments.** Extensive experiments across "multiple benchmarks, including medical VQA, video QA, and 3D medical image understanding" [abstract].
- **Results.** VERBATIM [abstract]: "Fleming-VL achieves state-of-the-art performance across multiple benchmarks, including medical VQA, video QA, and 3D medical image understanding." No specific per-benchmark numbers given in the abstract; not extracted.
- **Conclusions.** Publicly released "to promote transparent, reproducible, and auditable progress in medical AI" [abstract].

> **Why it matters to us.** A 2026 peer generalist medical MLLM -- useful landscape context for 'what else exists besides Lingshu/Qwen2.5-VL/MedGemma', but its emphasis (3D volumes, video, rare 2D modalities like ultrasound/dermoscopy) sits mostly outside our 2D-image-plus-text open-VQA scope, so treat as background/SOTA-landscape only, not a direct numeric comparison point.

<small>Read from: abstract-only.</small>

#### Hulu-Med: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding

*Songtao Jiang et al. · 2025 · arXiv preprint · [arXiv:2510.08668](https://arxiv.org/abs/2510.08668) · read priority 2*

**In one line.** Fully transparent generalist medical VLM (16.7M public/synthetic training samples, 7B-32B scales) unifying text, 2D/3D imaging, and video; beats open-source peers on 27/30 benchmarks and GPT-4o on 16.

- **Models.** Hulu-Med, trained "at 7B-32B parameter scales" [abstract].
- **Method.** "A medical-aware token-reduction strategy that prunes redundant visual tokens, achieving up to a 55% reduction for 3D and video inputs" [abstract], to unify language-only, 2D/3D vision-language, and video understanding in one architecture.
- **Datasets.** "16.7 million samples, comprising exclusively public or synthetic data, spanning 12 major anatomical systems and 14 medical imaging modalities" [abstract]. Evaluated on "30 public in-domain and out-of-domain medical benchmarks -- covering text reasoning, visual question answering, report generation, multilingual dialogue, video understanding, and rare disease diagnosis" [abstract].
- **Experiments.** Compares against open-source models and proprietary systems (GPT-4o, GPT-o1) across the 30-benchmark suite [abstract].
- **Results.** VERBATIM [abstract]: "Hulu-Med surpasses existing open-source models on 27 of 30 benchmarks and outperforms proprietary systems such as GPT-4o on 16 benchmarks"; "Despite being a VLM, Hulu-Med outperforms GPT-4o and matches GPT-o1 on the text-only HealthBench." No per-benchmark numeric table extracted (abstract-only, not fetched given the per-category WebFetch budget).
- **Conclusions.** Releases "a fully transparent, reproducible and cost-effective pipeline for holistic medical vision-language understanding by releasing our end-to-end data curation, training procedures, and model parameters" [abstract].

> **Why it matters to us.** Another 2026 open, fully-transparent generalist peer. Relevant both as a 'why not use this as our generator' landscape point, and as evidence that releasing the full pipeline (data curation + training + weights), not just weights, is becoming a stated norm in this field -- a useful citation when we justify our own preservation/reproducibility discipline (CLAUDE.md's committed/pushed/backed-up preservation notes).

<small>Read from: abstract-only.</small>

#### Gemma 3 Technical Report

*Gemma Team et al. · 2025 · arXiv preprint · [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) · read priority 2*

**In one line.** Gemma 3 technical report: 1B-27B open multimodal model family (vision-capable from 4B up), 128K+ context, with more local-vs-global attention layers to bound KV-cache growth; the LLM backbone MedGemma fine-tunes.

- **Models.** Gemma 3, sizes "1 to 27 billion parameters": 1B, 4B, 12B, 27B [abstract].
- **Method.** Vision understanding added; "a wider coverage of languages and longer context - at least 128K tokens" [abstract]. Architecture change: VERBATIM [html] "a 5:1 interleaving of local/global layers" and "Grouped-Query Attention (GQA) with post-norm and pre-norm with RMSNorm," specifically to reduce KV-cache memory growth with long context [abstract, html]. Trained with distillation [abstract].
- **Datasets.** Not extracted.
- **Experiments.** Compares pretrained and instruction-finetuned Gemma 3 variants against Gemma 2 and Gemini 1.5 Pro [abstract].
- **Results.** Table 1 (parameter breakdown, html) for the 4B model: vision encoder 417M params, embedding params 675M, non-embedding params 3,209M [Table 1, html] -- a dedicated hidden-size/layer-count architecture table was not locatable in the fetched arXiv HTML text (checked twice; likely lives in the released model config rather than the paper body). VERBATIM [abstract]: "Gemma3-4B-IT competitive with Gemma2-27B-IT and Gemma3-27B-IT comparable to Gemini-1.5-Pro across benchmarks." Note: the project brief for this card set states MedGemma-4b-it's backbone as hidden size 2560 / 34 layers -- that specific pair of numbers was NOT independently verified against this paper's text and should be treated as "not extracted from this arXiv source" until confirmed against Gemma 3's released model config.
- **Conclusions.** The local/global attention interleaving change primarily targets long-context KV-cache memory cost, not accuracy; distillation-based post-training substantially improves math/chat/instruction-following/multilingual ability over Gemma 2 [abstract].

> **Why it matters to us.** This is the language-model half of our third generator: MedGemma-4b-it = Gemma 3 4B (this paper) + MedSigLIP (see SigLIP card), medically fine-tuned. Read alongside the Qwen2.5-VL card for the architecture-teaching goal -- two different LLM backbones (Gemma 3 vs Qwen2.5) our probe design would need adapting to if extended beyond Lingshu.

<small>Read from: html.</small>

#### InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models

*Jinguo Zhu et al. · 2025 · arXiv preprint (Technical Report [comment field]) · [arXiv:2504.10479](https://arxiv.org/abs/2504.10479) · read priority 2*

**In one line.** InternVL3 unifies multimodal and text pretraining in a single native stage (instead of adapting a text-only LLM after the fact), reaching 72.2 on MMMU with the 78B model -- a leading open general-purpose VLM peer.

- **Models.** InternVL3 family, up to "InternVL3-78B" [abstract].
- **Method.** "Native multimodal pre-training paradigm" -- jointly acquires multimodal and linguistic capability from mixed multimodal + pure-text data in one pretraining stage, rather than post-hoc adapting a text-only LLM. Also: variable visual position encoding (V2PE) for extended context, SFT + mixed preference optimization (MPO) post-training, and test-time scaling [abstract].
- **Datasets.** Not extracted.
- **Experiments.** Broad multi-modal benchmark evaluation including MMMU [abstract].
- **Results.** VERBATIM [abstract]: "InternVL3-78B achieves a score of 72.2 on the MMMU benchmark, setting a new state-of-the-art among open-source MLLMs." "Its capabilities remain highly competitive with leading proprietary models, including ChatGPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro, while also maintaining strong pure-language proficiency" [abstract].
- **Conclusions.** A single native multimodal pretraining stage avoids alignment problems that post-hoc adaptation pipelines run into, and produces competitive results with proprietary frontier models [abstract].

> **Why it matters to us.** A leading general-purpose open VLM peer (not medically specialized) -- relevant both as architecture-lineage context and because our own project has already hit this model in the field: CLAUDE.md's 'known infra wall' entry records that Lingshu-32B / InternVL3-38B under MedEvalKit with tp=2 hangs deterministically in an NCCL collective on OmniMedVQA. So this paper is both background reading and the origin of a model we've already run into an infra wall with.

<small>Read from: abstract-only.</small>

#### HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale

*Junying Chen et al. · 2024 · arXiv preprint · [arXiv:2406.19280](https://arxiv.org/abs/2406.19280) · read priority 2*

**In one line.** Builds PubMedVision (1.3M medical VQA samples denoised/reformatted from PubMed image-text pairs using GPT-4V) and trains a 34B medical MLLM, HuatuoGPT-Vision, that improves on the MMMU Health & Medicine track.

- **Models.** HuatuoGPT-Vision, a 34B medical MLLM [abstract].
- **Method.** Refines noisy PubMed image-text pairs by employing GPT-4V "in an 'unblinded' capacity to denoise and reformat the data" [abstract], producing the PubMedVision dataset.
- **Datasets.** PubMedVision: "1.3 million medical VQA samples" curated from PubMed [abstract].
- **Experiments.** Validates PubMedVision by fine-tuning current MLLMs with it, and separately trains the 34B HuatuoGPT-Vision model; evaluated on benchmarks "including the MMMU Health & Medicine track" [abstract]; also involves manual expert checks of data quality [abstract].
- **Results.** VERBATIM [abstract]: PubMedVision "can significantly enhance the medical multimodal capabilities of current MLLMs, showing significant improvement in benchmarks including the MMMU Health & Medicine track"; HuatuoGPT-Vision "shows superior performance in medical multimodal scenarios among open-source MLLMs." No specific numeric values given in the abstract; not extracted.
- **Conclusions.** Denoising/reformatting existing medical image-text pairs with a strong general VLM (GPT-4V-in-the-loop) yields higher-quality training data than raw scale alone [abstract].

> **Why it matters to us.** Another open medical-MLLM predecessor emphasizing data curation over scale -- the same broad message as MedGemma's PathVQA/MedVQA removal, and a generator-side parallel to our own retrospective finding #3 ("training, not size, is the active ingredient in verification"). Not evaluated on our 8 open-text benchmarks per its abstract; treat as lineage/landscape context.

<small>Read from: abstract-only.</small>

#### Sigmoid Loss for Language Image Pre-Training

*Xiaohua Zhai et al. · 2023 · ICCV 2023 Oral [comment field] · [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) · read priority 2*

**In one line.** Introduces SigLIP: a pairwise sigmoid loss for image-text pretraining that removes the need for a global batch-wide softmax normalization, enabling efficient training at very large batch sizes; the base architecture MedGemma's vision encoder (MedSigLIP) is derived from.

- **Models.** SigLiT and SigLIP, including "SigLIP Base, Large, Shape-Optimized 400M" variants [comment field].
- **Method.** A sigmoid pairwise loss on image-text pairs, in place of standard softmax-normalized contrastive (CLIP-style) loss; "does not require a global view of the pairwise similarities for normalization" [abstract].
- **Datasets.** Not extracted (image-text pretraining corpus not named in the abstract).
- **Experiments.** Combined with Locked-image Tuning; studies batch size scaling up to one million examples [abstract].
- **Results.** VERBATIM [abstract]: "with only four TPUv4 chips, we train a SigLiT model that achieves 84.5% ImageNet zero-shot accuracy in two days"; "the benefits of growing batch size quickly diminish, with a more reasonable batch size of 32k being sufficient."
- **Conclusions.** Decoupling batch size from the loss computation lets batch size scale far higher than softmax contrastive loss allows, but the useful gains plateau well before the extreme (1M) batch sizes tested [abstract].

> **Why it matters to us.** This is the vision-encoder family MedGemma's MedSigLIP (our third generator's vision tower) is fine-tuned from -- background reading for the 'vision encoder' half of the architecture-first teaching goal: SigLIP produces image-patch embeddings via a ViT trained with a sigmoid contrastive loss; MedGemma medically fine-tunes that encoder and feeds its output through a projector into Gemma 3.

<small>Read from: abstract-only.</small>

#### LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day

*Chunyuan Li et al. · 2023 · arXiv preprint · [arXiv:2306.00890](https://arxiv.org/abs/2306.00890) · read priority 2*

**In one line.** Cost-efficient recipe (<15h on 8 A100s) for a biomedical LLaVA variant, trained by curriculum learning on PubMed figure-caption pairs plus GPT-4-generated instruction data; appears as the 'LLaVA-Med (BioMedCLIP)' baseline in MedGemma's SLAKE/VQA-RAD tables.

- **Models.** LLaVA-Med, built on the general-domain LLaVA architecture [abstract].
- **Method.** Curriculum learning: first align biomedical vocabulary using figure-caption pairs "as is", then master open-ended conversational semantics using GPT-4-generated instruction-following data, "broadly mimicking how a layperson gradually acquires biomedical knowledge" [abstract].
- **Datasets.** "A large-scale, broad-coverage biomedical figure-caption dataset extracted from PubMed Central" [abstract]; evaluated on "three standard biomedical visual question answering datasets" [abstract] (not individually named in the abstract).
- **Experiments.** Trained in "less than 15 hours (with eight A100s)" [abstract]; compared against prior supervised SOTA on the three VQA datasets.
- **Results.** Own-paper abstract gives no specific numeric scores ("outperforms previous supervised state-of-the-art on certain metrics" [abstract]). External calibration point: appears as "LLaVA-Med (BioMedCLIP)" in MedGemma's Table 9: SLAKE 87.1 open-ended token recall / 86.8 closed-ended accuracy [Table 9 of arXiv:2507.05201, html].
- **Conclusions.** GPT-4-distilled instruction tuning, combined with a curriculum that first aligns biomedical vocabulary, transfers into strong biomedical VQA performance with modest compute [abstract].

> **Why it matters to us.** Direct medical-domain analog of general-purpose LLaVA, and a concrete external SLAKE number (via the MedGemma cross-reference table) alongside Lingshu's own Table 6 and the benchmarking-study's Table 1 -- but note the metric (token recall/F1) differs from our judge/exact-match macro accuracy, so it is NOT directly poolable with our own numbers, only usable as an independent calibration point.

<small>Read from: html.</small>


#### Also in this area (7), in brief

- **A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation** — Zhihong Chen et al. (2024), [arXiv:2401.12208](https://arxiv.org/abs/2401.12208). CheXagent: a chest-X-ray-specialized vision-language foundation model trained on the large CheXinstruct dataset and evaluated on the CheXbench benchmark across eight CXR interpretation task types; a radiologist study shows real drafting-time savings.<br><small>*For us:* Peripheral: a single-modality (chest X-ray only) specialist, not a general medical VQA generator -- relevant only as contrast: our three generators (Lingshu, Qwen2.5-VL, MedGemma) are broad multi-domain VQA models, not single-organ-system specialists like CheXagent.</small>
- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** — Peng Wang et al. (2024), [arXiv:2409.12191](https://arxiv.org/abs/2409.12191). Qwen2-VL: the direct predecessor of Qwen2.5-VL, introducing Naive Dynamic Resolution (variable visual token count per image) and M-RoPE (multimodal rotary position embeddings), at 2B/8B/72B scale.<br><small>*For us:* Peripheral/lineage: the immediate predecessor of Qwen2.5-VL, the backbone our main generator Lingshu-7B is fine-tuned from. Read only if the Qwen2.5-VL card's architecture description needs a 'what changed between versions' comparison (Qwen2.5-VL keeps M-RoPE-style ideas but reworks the merger/window-attention design); not otherwise load-bearing for our reports.</small>
- **BiomedGPT: A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks** — Kai Zhang et al. (2023), [arXiv:2305.17100](https://arxiv.org/abs/2305.17100). First open-source, lightweight generalist biomedical vision-language model (BiomedGPT), SOTA on 16/25 experiments at compute-friendly scale; appears as a strong closed-VQA baseline (BiomedGPT-B) in MedGemma's SLAKE table.<br><small>*For us:* Peripheral: an older (2023) generalist biomedical VLM, mainly useful here as the strongest closed-VQA baseline on SLAKE in MedGemma's own comparison table -- keep for benchmark calibration, not for architecture teaching.</small>
- **Med-Flamingo: a Multimodal Medical Few-shot Learner** — Michael Moor et al. (2023), [arXiv:2307.15189](https://arxiv.org/abs/2307.15189). Few-shot medical VLM (continues pretraining OpenFlamingo-9B on medical image-text data) targeting generative open-ended medical VQA under data scarcity.<br><small>*For us:* Peripheral/historical: an early (2023) few-shot generative medical VQA model, relevant mainly as a reminder that open-ended (not just MCQ) medical VQA, and human/clinician rating as an evaluation method, both predate our LLM-judge approach.</small>
- **Towards Generalist Biomedical AI** — Tao Tu et al. (2023), [arXiv:2307.14334](https://arxiv.org/abs/2307.14334). Med-PaLM Multimodal (Med-PaLM M): a single set of weights spanning 14 biomedical tasks (text, imaging, genomics) on the new MultiMedBench benchmark, competitive with or exceeding specialist SOTA on most of them.<br><small>*For us:* Peripheral/historical: proprietary (Google) generalist biomedical model, mainly useful as the earliest 'one model, many biomedical modalities' precedent for the generalist-medical-VLM lineage that Lingshu, MedGemma, Hulu-Med, and Fleming-VL continue.</small>
- **Visual Instruction Tuning** — Haotian Liu et al. (2023), [arXiv:2304.08485](https://arxiv.org/abs/2304.08485). Introduces LLaVA, the original general-domain vision-language assistant connecting a vision encoder + LLM via instruction tuning on GPT-4-generated multimodal data -- the architectural template LLaVA-Med (and much of the field) descends from.<br><small>*For us:* Peripheral/architecture-lineage only: the general-domain ancestor of LLaVA-Med. Its vision-encoder + connector + LLM pattern is the same three-part architecture Qwen2.5-VL and Gemma 3/MedGemma both use, making it a natural first stop for the architecture-first teaching goal even though we neither use nor directly compare against LLaVA itself.</small>
- **Improved Baselines with Visual Instruction Tuning** — Haotian Liu et al. (2023), [arXiv:2310.03744](https://arxiv.org/abs/2310.03744). LLaVA-1.5: shows a simple 2-layer MLP projector (instead of a linear one) plus academic-task VQA data gives state-of-the-art results across 11 benchmarks with only 1.2M training samples and about a day of training.<br><small>*For us:* Peripheral/architecture-lineage: the origin of the 'MLP projector between vision encoder and LLM' design pattern that Qwen2.5-VL's merger and (in spirit) MedGemma/Gemma 3's connector both use -- a supporting citation for the projector/connector part of the architecture-teaching goal, not a direct comparison point for our benchmarks.</small>

## 3.8 Medical VQA benchmarks, datasets, and the evaluation protocol

This category maps the empirical ground our project's numbers stand on: the eight open-ended medical-VQA benchmarks we evaluate on (their provenance, how each was actually built, and known construction flaws), plus the separate question of how you score free-text medical answers at all once you have them. Two threads run through it. First, dataset quality: several benchmarks here are not independently authored test sets but are templated from figure captions or classification labels (PathVQA, SLAKE, OmniMedVQA, VQA-Med), and recent audits show that 'authentic image' provenance does not rule out pretraining contamination (SLAKE image overlap, whole-slide-image benchmark leakage) or answer-position shortcuts (PMC-VQA's own ~31%-vs-25% skew toward option B). Second, evaluation protocol: because our method scores best-of-N candidates with a Lingshu-32B LLM-judge, the LLM-as-a-judge literature (MT-Bench, self-preference, judging-the-judges) is not background reading but a direct methodological risk assessment of our own pipeline, especially the same-model-family judge/generator pairing. Both threads matter because a headline accuracy number is only as trustworthy as the benchmark and the scorer behind it.

#### ★ A Controlled Audit of Pretraining Contamination in Public Medical Vision-Language Benchmarks

*Bruce Changlong Xu et al. · 2026 · arXiv preprint · [arXiv:2606.10066](https://arxiv.org/abs/2606.10066) · read priority 1 · **PDF in `papers/`***

**In one line.** Audits SLAKE-En, PathVQA, VQA-RAD and an OmniMedVQA mirror for pretraining contamination using 4 detector families; finds real image-side overlap on SLAKE-En but shows two of the four detector families are unreliable (a non-medical control model, BLIP-2, 'reproduces' their positive signals).

- **Models.** Audits open VLMs including Qwen2.5-VL and BLIP-2 (used as an out-of-domain negative control) [abstract].
- **Method.** Four detector families: (1) image-side near-neighbour overlap against PMC-OA-beta via SigLIP embeddings, (2) canonical-order exchangeability (text-side), (3) cohort-relative Min-K%++ tail enrichment, (4) cross-model top-K overlap [abstract].
- **Datasets.** SLAKE-En, PathVQA, VQA-RAD, and an 'auxiliary public OmniMedVQA mirror' [abstract].
- **Experiments.** Runs all 4 detectors per dataset/model; manual adjudication of flagged image pairs; an ordering ablation; and an external non-medical/pre-domain baseline (BLIP-2) to test whether detectors merely fire on generic structure [abstract].
- **Results.** '19.8% of images are flagged under SigLIP-B-16 and 4.2% under SigLIP-SO400M' on SLAKE-En, 'while out-of-domain controls produce 0/2000 flags' [abstract]. Manual check: 'same-modality, same-projection matches to different patients rather than verified pixel-level duplicates' [abstract] -- so the authors interpret this as source/distributional overlap, not confirmed per-image memorization. 'On the text side, Qwen2.5-VL on SLAKE-En shows a canonical-order exchangeability signal that survives ordering ablation and external non-medical baselines' [abstract]. 'Min-K%++ tail enrichment and cross-model top-K overlap collapse under an external pre-domain baseline: BLIP-2 reproduces the apparent positive signals despite lacking plausible medical-VQA exposure' [abstract].
- **Conclusions.** 'these cohort-relative detectors are unreliable as standalone membership-inference signals on small medical-VLM cohorts' [abstract] -- i.e. 2 of the 4 detector families are false-positive-prone and should not be trusted alone; the image-overlap and exchangeability signals on SLAKE-En are the more credible findings.

> **Why it matters to us.** One of our four core picks. Directly informs whether our SLAKE-open results (one of our 8 benchmarks) reflect genuine capability or memorization; the 19.8%/4.2% SLAKE-En image-overlap finding is already flagged as 'important' in our own task brief. Also a methodological caution: two of the four contamination-detection techniques used across the field (Min-K%++, cross-model top-K overlap) are shown here to be unreliable on small model cohorts like ours, which should make us skeptical of using those two methods ourselves without an equivalent negative-control ablation.

<small>Read from: abstract-only.</small>

#### ★ OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM

*Yutao Hu et al. · 2024 · arXiv preprint · [arXiv:2402.09181](https://arxiv.org/abs/2402.09181) · read priority 1 · **PDF in `papers/`***

**In one line.** A 73-source, 12-modality, >20-anatomical-region medical VQA benchmark built entirely from authentic (non-synthetic) clinical images, showing medical-specialized LVLMs can underperform general-domain ones.

- **Models.** Evaluates existing LVLMs, contrasting medical-specialized models against general-domain LVLMs [abstract]; specific model names not extracted -- two WebFetch attempts (abs page) returned only abstract-level content, not the results table, within this run's budget.
- **Method.** Collects and curates images from 73 distinct existing medical datasets across 12 imaging modalities and more than 20 anatomical regions, framed as a VQA benchmark using authentic clinical images (not synthetic/rendered) [abstract].
- **Datasets.** OmniMedVQA itself, sourced from 73 medical datasets, 12 modalities, >20 anatomical regions [abstract]. Exact total QA-pair count: not extracted (not stated in the abstract; full-text table was not reached within budget).
- **Experiments.** 'Extensive experiments' evaluating LVLMs' medical VQA ability [abstract]; design details beyond this not extracted.
- **Results.** 'existing LVLMs struggle to address these medical VQA problems effectively' [abstract]; 'medical-specialized LVLMs even exhibit inferior performance to those general-domain models' [abstract]. No numeric accuracy figures were extracted (not present in the abstract; the full-text results table was not reached).
- **Conclusions.** Calls for 'a more versatile and robust LVLM in the biomedical field'; current LVLM understanding of real medical images is limited even for medically fine-tuned models [abstract].

> **Why it matters to us.** OmniMedVQA (with MCQ options stripped) is one of our 8 open-ended eval benchmarks -- we deliberately convert its native multiple-choice format to open-ended, directly exploiting the format axis this category is about. Its own finding that medical-specialized LVLMs can lag general ones is a caution for us: Lingshu-7B/32B being 'medical' finetunes does not guarantee they beat general Qwen2.5-VL, worth checking against our own Qwen2.5-VL replication.

<small>Read from: abstract-only.</small>

#### LLM Evaluators Recognize and Favor Their Own Generations

*Arjun Panickssery et al. · 2024 · arXiv preprint · [arXiv:2404.13076](https://arxiv.org/abs/2404.13076) · read priority 1*

**In one line.** Shows LLM judges (GPT-4, Llama 2) can recognize their own outputs above chance, and that this self-recognition ability correlates linearly with the strength of their self-preference scoring bias.

- **Models.** GPT-4 and Llama 2, used as both generators and judges; fine-tuned variants used to strengthen/probe self-recognition [abstract].
- **Method.** Measures self-recognition accuracy (can a model tell its own output apart from others'/humans') and self-preference bias (does it score its own output higher despite equal human-judged quality); fine-tunes models to test the causal link between the two, with controlled experiments against confounders [abstract].
- **Datasets.** not extracted (no specific named dataset in the abstract beyond generated-text comparisons)
- **Experiments.** Out-of-the-box self-recognition measurement; fine-tuning to strengthen self-recognition and observe the effect on self-preference; confound-controlled causal experiments [abstract].
- **Results.** 'LLMs such as GPT-4 and Llama 2 have non-trivial accuracy at distinguishing themselves from other LLMs and humans' [abstract]; 'a linear correlation between self-recognition capability and the strength of self-preference bias' [abstract]. No specific numeric accuracy/correlation values are given in the abstract.
- **Conclusions.** Self-recognition capability is a partial, causally-implicated driver of self-preference bias in LLM evaluators, with implications for AI safety and any pipeline using an LLM to judge outputs, including its own [abstract].

> **Why it matters to us.** Our correctness labels come from Lingshu-32B judging Lingshu-7B outputs -- same model family (not literally 'self', but a closely related generation/lineage). This paper's finding that self-preference scales with self-recognition ability suggests a nonzero, family-correlated risk that our 32B judge is not perfectly neutral toward Lingshu-7B-style phrasing relative to, say, Qwen2.5-VL or MedGemma candidates -- a possible confound for any judge-based cross-model comparison (e.g. our 6/8 vs. 8/8 headline gain being partly a judge-family effect) that we have not controlled for and should flag as an open risk.

<small>Read from: abstract-only.</small>

#### ★ PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering

*Xiaoman Zhang et al. · 2023 · arXiv preprint · [arXiv:2305.10415](https://arxiv.org/abs/2305.10415) · read priority 1 · **PDF in `papers/`***

**In one line.** Introduces PMC-VQA (227k generative QA pairs from 149k images) and MedVInT, plus a manually-verified 2,000-pair test set; the paper's own data analysis shows the correct MCQ answer is skewed toward option B (~31% vs. 25% expected).

- **Models.** MedVInT: a vision encoder aligned to a pretrained LLM, using a generative (not classification) VQA formulation [abstract]. Pretrained on PMC-VQA then fine-tuned on VQA-RAD, SLAKE, and ImageCLEF-2019 [abstract].
- **Method.** Reframes MedVQA as text generation by aligning a pretrained vision encoder with an LLM; builds a scalable pipeline to construct PMC-VQA at scale from PMC figure-caption pairs [abstract].
- **Datasets.** PMC-VQA training set: '226,946 question-answer pairs corresponding to 149,075 images' [html, section 4.1]. Manually-verified test set ('PMC-VQA-test'): 2,000 QA pairs [html, section 4.1 / Table 3]. Also fine-tuned/evaluated on VQA-RAD, SLAKE, ImageCLEF-2019 [abstract].
- **Experiments.** Data-quality/bias analysis of the correct-answer letter distribution (section 2.1 of the paper); a language-only (no-image) baseline to test how guessable the answers are (section 2.3) [html].
- **Results.** 'The correct options were distributed as follows: A (24.07%), B (30.87%), C (29.09%), D (15.97%)' [html, section 2.1] -- option B is overrepresented (~31% vs. 25% chance) and D underrepresented (~16%). 'around 30% of the questions have "B" answers, making the 30.8% score nearly equivalent to the highest possible score attainable through guessing' [html, section 2.3] -- i.e. an always-answer-B, language-only baseline scores ~30.8%, near the ceiling achievable by guessing alone.
- **Conclusions.** PMC-VQA/MedVInT significantly outperforms prior MedVQA models on free-form answer generation across VQA-RAD/SLAKE/ImageCLEF-2019 [abstract]; the paper documents but does not correct the training-set answer-position imbalance [html].

> **Why it matters to us.** Central provenance paper: PMC-VQA is our project's historical training/eval benchmark from the June MedVLThinker era (CLAUDE.md SS0), and 'PMC_VQA test_2.csv' is a MedEvalKit-era eval cell in the live method. This paper's own admission of a ~31%-vs-25% option-B skew in its training data is exactly the kind of answer-position/format bias our project's finding #2 (AUROC ~0.6 on MCQ vs. ~0.87 on free text; option discreteness, not length, is the cause) would predict creates spuriously 'easy' MCQ signal -- a paper-verified instance of the same phenomenon, independent of our own results.

<small>Read from: html.</small>

#### ★ Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

*Lianmin Zheng et al. · 2023 · NeurIPS 2023 Datasets and Benchmarks Track · [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) · read priority 1 · **PDF in `papers/`***

**In one line.** Establishes and validates LLM-as-a-judge (e.g. GPT-4 scoring model outputs) against human preference, while explicitly naming position, verbosity, and self-enhancement bias as failure modes to correct for.

- **Models.** GPT-4 (and other strong LLMs) used as judges, scoring LLaMA/Vicuna variants as exam-takers [abstract].
- **Method.** Introduces MT-Bench (a multi-turn question set) and Chatbot Arena (crowdsourced pairwise battles); studies LLM-judge/human agreement and catalogs judge biases -- position bias, verbosity bias, self-enhancement (self-preference) bias, and limited reasoning ability -- proposing mitigations for some [abstract].
- **Datasets.** MT-bench questions, 3K expert votes, and 30K Chatbot Arena conversations with human preference labels, all released publicly [abstract].
- **Experiments.** Compares LLM-judge scores/rankings to human preference on both benchmarks; evaluates several LLaMA/Vicuna variants [abstract].
- **Results.** 'strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans' [abstract].
- **Conclusions.** LLM-as-a-judge is 'a scalable and explainable way to approximate human preferences' but requires bias-awareness (position/verbosity/self-enhancement); MT-Bench and human-preference benchmarks complement traditional NLP benchmarks [abstract].

> **Why it matters to us.** Direct methodological ancestor of our own evaluation design: we use a Lingshu-32B LLM-judge to label correctness of Lingshu-7B best-of-N candidates. This paper coins 'self-enhancement/self-preference bias,' which is exactly the risk 2404.13076 (also in this category) measures for a same-family judge/generator pairing like ours. Cite this paper for the terminology and the >80%-agreement sanity bar, and 2404.13076 for the specific same-family risk.

<small>Read from: abstract-only.</small>

#### A dataset of clinically generated visual questions and answers about radiology images

*Jason J. Lau et al. · 2018 · Scientific Data · [doi:10.1038/sdata.2018.251](https://www.nature.com/articles/sdata2018251) · read priority 1*

**In one line.** The original clinician-authored (not templated) radiology VQA dataset: physicians wrote free-form questions about MedPix CT/MRI/X-ray images, later paraphrased by lay participants.

- **Models.** not extracted as a list; the paper's own baseline, MCB-RAD, is reported at 25.4% accuracy on open-ended questions [fetched].
- **Method.** Two-phase construction: (1) clinicians write free-form questions about randomized images, phrased naturally as if 'asking a colleague or another physician'; (2) paired participants generate 'rephrased' and 'framed' versions of others' questions. All pairs are manually validated, with disagreements resolved by expert radiologists [fetched].
- **Datasets.** VQA-RAD itself: 315 radiological images from MedPix (104 head axial CT/MRI, 107 chest X-ray, 104 abdominal axial CT) [fetched]; 3,515 total questions (1,515 free-form, 733 rephrased, 1,267 framed), averaging ~10 questions/image [fetched]. Test set: 300 randomly chosen free-form questions plus 151 corresponding paraphrased questions; the remainder is training [fetched].
- **Experiments.** Scored with simple/mean accuracy, human inter-annotator F1 agreement (range 0.78-0.95, mean 0.85), BLEU (flagged by the authors as problematic for this task), and manual evaluation allowing partial credit [fetched].
- **Results.** Of the free-form subset: '42% (637) open-ended answer types and 58% (878) close-ended' [fetched]; 'Yes/no questions represent 92% of the close-ended QA pairs' [fetched]. Baseline MCB-RAD: 25.4% accuracy on open-ended questions [fetched].
- **Conclusions.** The authors state BLEU 'penalizes answers with varying lengths' and 'is not useful for medical VQA where there are many ways to phrase an answer'; contemporary (2018) models needed 'more data' and better learning of medical terminology, given the poor open-ended scores [fetched].

> **Why it matters to us.** VQA-RAD-open is one of our 8 eval benchmarks. Unlike SLAKE/PathVQA it is NOT templated -- genuinely clinician-authored free-form questions -- making it our closest-to-real-world open-text source. Its 92% yes/no skew within the closed-ended half is a reminder that 'closed-ended' in this older literature usually means yes/no, a different axis from our project's MCQ vs. open-text split. 2405.03162 (Med-Gemini, also in this category) is cited in our task brief as motivating a contamination-free VQA-RAD re-split -- worth checking directly before relying on any VQA-RAD re-split claim (see that card's flag).

<small>Read from: html.</small>

#### MedEvalKit

*Alibaba DAMO Academy · None · read priority 1*

**In one line.** The GitHub evaluation harness our project's live method uses; supports 26+ benchmarks including PMC_VQA/SLAKE/VQA-RAD/PathVQA/OmniMedVQA/MMMU-Medical and 16+ models, scoring by exact match with optional LLM-judge -- but its README does not state which PMC-VQA file/CSV it loads.

- **Models.** '16+ models' listed, including Qwen2.5-VL, Qwen2-VL, BiMediX2, LLaVA-Med, HuatuoGPT-vision, InternVL, Llama-3.2-vision, LLaVA, Janus, HealthGPT, BiomedGPT, MedGemma, Med_Flamingo, MedDr [github README, fetched].
- **Method.** A 'comprehensive evaluation framework for Large Medical Models (LMMs/LLMs)' with configurable generation settings (temperature, top_p, repetition penalty, max_new_tokens) [github README, fetched].
- **Datasets.** '26+ benchmarks' including PMC_VQA, SLAKE, VQA-RAD, PATH-VQA, OmniMedVQA, MMMU-Medical (test/val), IU XRAY, CheXpert Plus, MIMIC-CXR, MedFrameQA, plus text-only benchmarks (MedQA-USMLE, MedMCQA, PubMedQA, CMB, etc.) [github README, fetched]. PMC-VQA's source is given only as 'RadGenome/PMC-VQA' on HuggingFace, with NO specific file (test.csv / test_2.csv / test_clean.csv) named -- confirmed not stated in the README [github README, fetched].
- **Experiments.** N/A (this is a software repository, not a paper)
- **Results.** N/A (this is a software repository, not a paper)
- **Conclusions.** N/A (this is a software repository, not a paper)

> **Why it matters to us.** This IS our project's live evaluation harness (CLAUDE.md SS0/SS8: 'MedEvalKit... is the faithful harness every current paper number comes from'). The README's silence on which PMC-VQA file is loaded corroborates CLAUDE.md's own landmine note that PMC-VQA test_2.csv (v2, 33,430 items, 'zero published verification') is hard-coded at MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39 rather than documented in the README -- i.e. this silence is itself the finding the task brief anticipated: which file gets loaded is a code-level fact, not a documented or reviewable one.

<small>Read from: html.</small>

#### RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic Visual Question Answering

*Léo Butsanets et al. · 2025 · arXiv preprint · [arXiv:2512.17396](https://arxiv.org/abs/2512.17396) · read priority 2*

**In one line.** Large-scale (750K image / 7.5M QA) CT/MRI VQA benchmark from expert-curated annotations, explicitly designed and tested to resist text-only shortcut solving.

- **Models.** not extracted (abstract does not name the specific VLMs evaluated)
- **Method.** Built from expert-curated annotations across 8 anatomical regions and 97 pathology categories spanning 3 tasks (abnormality detection, anatomy recognition, pathology identification); supports open-ended, closed-ended and multiple-choice question formats; includes a text-only (no-image) ablation to test for linguistic shortcuts [abstract].
- **Datasets.** RadImageNet-VQA itself: 750K images, 7.5M QA pairs, 8 anatomical regions, 97 pathology categories [abstract].
- **Experiments.** 'Extensive experiments' with state-of-the-art VLMs, including fine-tuning, plus a text-only ablation [abstract].
- **Results.** 'state-of-the-art vision-language models still struggle with fine-grained pathology identification, particularly in open-ended settings and even after fine-tuning' [abstract]; 'model performance collapses to near-random without image inputs, confirming that RadImageNet-VQA is free from linguistic shortcuts' [abstract]. No specific numeric accuracy figures given in the abstract.
- **Conclusions.** Existing medical VQA datasets are limited in scale/modality diversity and prone to text shortcuts; RadImageNet-VQA is offered as a much larger, shortcut-resistant CT/MRI benchmark [abstract].

> **Why it matters to us.** RadImageNet-VQA (open-ended form) is one of our 8 eval benchmarks. Its own reported finding -- that current VLMs specifically struggle on open-ended (vs. MCQ/closed) fine-grained pathology ID -- is direct external support for our finding #2 (answer format determines whether routing signals work; open-text is harder and more informative than MCQ) and for why our best-of-N + verifier method targets the open-ended setting.

<small>Read from: abstract-only.</small>

#### Kvasir-VQA-x1: A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy

*Sushant Gautam et al. · 2025 · arXiv preprint · [arXiv:2506.09958](https://arxiv.org/abs/2506.09958) · read priority 2*

**In one line.** LLM-generated, complexity-stratified expansion of Kvasir-VQA (GI endoscopy) adding 159,549 new QA pairs plus a separate robustness track using synthetic visual-artifact augmentations.

- **Models.** not extracted (abstract does not name the VLMs evaluated)
- **Method.** QA pairs generated by a 'systematic method using large language models,' stratified by complexity to test deeper clinical reasoning; visual augmentations mimicking common imaging artifacts define a second, robustness-focused evaluation track [abstract].
- **Datasets.** Kvasir-VQA-x1 itself: 159,549 new QA pairs added on top of the original Kvasir-VQA GI-endoscopy image pool [abstract]. Two eval tracks: standard VQA performance, and robustness under visual perturbation [abstract].
- **Experiments.** not extracted beyond the two-track design stated in the abstract
- **Results.** not extracted (no accuracy numbers given in the abstract)
- **Conclusions.** Aims to provide a more clinically complex and visually diverse GI-endoscopy MedVQA benchmark than the original Kvasir-VQA, following FAIR data principles [abstract].

> **Why it matters to us.** Kvasir-VQA-x1 (open-ended) is one of our 8 eval benchmarks. Its LLM-generated-question construction is a relevant methodological parallel to our own use of an LLM (Lingshu-32B) as judge: both put an LLM inside the benchmark/evaluation loop, so the self-preference / judge-bias literature in this category (2404.13076, 2306.05685) is relevant to how its questions -- and our correctness labels -- were produced.

<small>Read from: abstract-only.</small>

#### GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis

*Bo Liu et al. · 2024 · arXiv preprint · [arXiv:2411.16778](https://arxiv.org/abs/2411.16778) · read priority 2*

**In one line.** The largest chest-X-ray VQA dataset to date (151,025 images / 1.6M questions) with four question types and built-in visual+textual explanations for every answer.

- **Models.** Evaluates 12 representative LVLMs; also fine-tunes an existing LVLM on GEMeX's training set as a strong baseline [abstract]. Specific model names not extracted from the abstract.
- **Method.** Provides, per QA pair, a 'multi-modal explainability mechanism' with detailed visual (grounding) and textual explanations; supports four question types -- open-ended, closed-ended, single-choice, and multiple-choice [abstract].
- **Datasets.** GEMeX itself: 151,025 images, 1,605,575 questions [abstract], chest X-ray only.
- **Experiments.** 12 representative LVLMs evaluated; one LVLM fine-tuned on the GEMeX training set for comparison [abstract].
- **Results.** 'Evaluation of 12 representative large vision language models (LVLMs) on GEMeX reveals suboptimal performance, underscoring the dataset's complexity' [abstract]; the fine-tuned model shows 'substantial performance improvement' (no specific numeric deltas given in the abstract) [abstract].
- **Conclusions.** Positions GEMeX as filling two gaps in Med-VQA benchmarks: lack of answer explanations, and narrow question-format coverage [abstract].

> **Why it matters to us.** GEMeX-open is one of our 8 eval benchmarks. Its four-question-type design (open/closed/single-choice/multi-choice) makes it one of the few source benchmarks natively spanning both format axes we study; worth checking whether its open-ended subset behaves like our other open-text benchmarks or is closer to its own closed-choice subset (relevant to our finding #2 on format-dependent routing-signal quality).

<small>Read from: abstract-only.</small>

#### Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges

*Aman Singh Thakur et al. · 2024 · Proceedings of the Fourth Workshop on Generation, Evaluation and Metrics (GEM^2) 2025 · [arXiv:2406.12624](https://arxiv.org/abs/2406.12624) · read priority 2*

**In one line.** A systematic study of 13 judge LLMs scoring 9 exam-taker models finds only the largest judges reasonably align with humans, and identifies leniency and prompt-sensitivity as systematic judge vulnerabilities.

- **Models.** 13 judge models of varying size/family, judging 9 exam-taker models (base and instruction-tuned) [abstract].
- **Method.** Compares LLM-judge scores to human scores in a high-inter-human-agreement setting; separately studies score calibration vs. ranking ability, and probes vulnerabilities via error analysis [abstract].
- **Datasets.** not extracted (no dataset name given beyond the judge/exam-taker model pairing setup)
- **Experiments.** Judge-vs-human absolute-score alignment; judge-vs-human ranking-of-models alignment (including smaller judges and a lexical-overlap metric); vulnerability probes for prompt complexity/length sensitivity and leniency [abstract].
- **Results.** 'only the best (and largest) models achieve reasonable alignment with humans' but 'are still quite far behind inter-human agreement' with scores that 'may still differ with up to 5 points from human-assigned scores' [abstract]. For ranking exam-takers (not absolute scoring), 'smaller models and even the lexical metric... may provide a reasonable signal' [abstract].
- **Conclusions.** Judge quality depends heavily on the task (absolute scoring vs. relative ranking) and on judge size/family; judges are sensitive to prompt complexity/length and biased toward leniency; high percent-agreement can mask large score-magnitude disagreement, so alignment metrics beyond percent-agreement are needed [abstract].

> **Why it matters to us.** Directly bears on our choice of a single 32B judge model and threshold-style correctness labeling: the finding that even large judges assign scores up to 5 points off from humans, and that 'leniency' is a systematic bias, is a reason to sanity-check our exact-match currency alongside the judge currency (already mandated in CLAUDE.md SS0 for verifier claims) rather than trust judge-only correctness labels.

<small>Read from: abstract-only.</small>

#### Open-ended VQA benchmarking of Vision-Language models by exploiting Classification datasets and their semantic hierarchy

*Simon Ging et al. · 2024 · ICLR 2024 (Spotlight) · [arXiv:2402.07270](https://arxiv.org/abs/2402.07270) · read priority 2*

**In one line.** Proposes turning existing visual-classification datasets into an open-ended VQA benchmark, using label semantic hierarchy to auto-generate follow-up questions, and validates NLP-metric vs. LLM-based scoring against human judgement.

- **Models.** Evaluates 'a suite of vision-language models' (text-generative), compared against discriminative VLMs [abstract]; specific model names not extracted.
- **Method.** Builds a VQA benchmark from classification-dataset labels; uses the label space's semantic hierarchy to auto-generate follow-up questions probing whether a coarse-but-technically-correct answer reflects real fine-grained understanding; compares traditional NLP metrics vs. LLM-based scoring against a human evaluation study to pick the final metric [abstract].
- **Datasets.** Built on 'well-known visual classification datasets' -- general-domain object/action/attribute classification, not medical [abstract].
- **Experiments.** Human evaluation study to validate the metric choice; benchmark applied across VLMs comparing object/action/attribute classification ability [abstract].
- **Results.** not extracted (no specific numeric results given in the abstract)
- **Conclusions.** Argues for benchmark designs that reuse classification datasets' scale/reliability and validate metric choice against human judgement rather than assuming it, as a foundation for more precise text-generative VQA evaluation [abstract].

> **Why it matters to us.** A methodological parallel, not medical: like our project, it must solve 'how do you score a free-text open-ended answer against a short gold label.' Its finding that metric choice (NLP vs. LLM-based) needs human validation rather than assumption supports our own practice of reporting both an LLM-judge and normalized exact-match currency (CLAUDE.md SS0 standing caveat) rather than trusting one scorer.

<small>Read from: abstract-only.</small>

#### Advancing Multimodal Medical Capabilities of Gemini

*Lin Yang et al. · 2024 · arXiv preprint · [arXiv:2405.03162](https://arxiv.org/abs/2405.03162) · read priority 2*

**In one line.** Introduces the Med-Gemini family (2D/3D radiology, histopathology, ophthalmology, dermatology, genomics), reporting CXR report-generation and VQA gains over prior SoTA.

- **Models.** Med-Gemini-2D, Med-Gemini-3D, and Med-Gemini-Polygenic, built on Gemini and fine-tuned on 2D/3D radiology, histopathology, ophthalmology, dermatology, and genomic data [abstract].
- **Method.** Fine-tunes Gemini's multimodal models on medical modalities; evaluated via expert (radiologist) side-by-side comparison for report generation, and via standard VQA/classification benchmarks [abstract].
- **Datasets.** Two (unnamed in the abstract) CXR report-generation datasets; CXR VQA benchmarks ('17 of 20 tasks', unnamed); histopathology/ophthalmology/dermatology classification benchmarks ('20 tasks', unnamed) [abstract].
- **Experiments.** Expert evaluation of AI vs. radiologist CXR/CT reports; VQA/classification benchmark comparison against SoTA/baselines [abstract].
- **Results.** Med-Gemini-2D CXR report generation 'exceeding previous best results... by an absolute margin of 1% and 12%' across two datasets, with '57% and 96%... on normal cases, and 43% and 65% on abnormal cases... evaluated as equivalent or better than the original radiologists' reports' [abstract]. Med-Gemini-3D: '53% of AI reports considered clinically acceptable' [abstract]. 'Med-Gemini-2D surpasses the previous best performance in CXR visual question answering (VQA)... exceeding SoTA or baselines on 17 of 20 tasks' [abstract]; 'surpasses baselines across 18 out of 20 tasks' in histopathology/ophthalmology/dermatology classification [abstract].
- **Conclusions.** Broad gains across medical imaging/genomic tasks, but the authors caveat that 'further development and evaluation are necessary in the safety-critical medical domain' [abstract].

> **Why it matters to us.** FLAG: our task brief describes this paper as 'the paper MedGemma cites for a contamination-free VQA-RAD re-split,' but the fetched abstract (abstract-only provenance -- the full body text was not reached within this run's WebFetch budget) does not itself mention VQA-RAD or a re-split. We report this as not extracted / not confirmed rather than assert it, per the no-fabrication rule. If a contamination-free VQA-RAD split is needed for our pipeline, verify it directly in the full PDF (or the MedGemma technical report) before relying on this claim.

<small>Read from: abstract-only.</small>

#### Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA

*Qianqi Yan et al. · 2024 · arXiv preprint · [arXiv:2405.20421](https://arxiv.org/abs/2405.20421) · read priority 2*

**In one line.** Introduces ProbMed, showing top LMMs (GPT-4o, GPT-4V, Gemini Pro) score below random-guess on negation-probed / hallucinated-attribute medical diagnosis questions.

- **Models.** GPT-4o, GPT-4V, Gemini Pro, LLaVA-Med, CheXagent [abstract].
- **Method.** 'Probing evaluation' pairs original questions with negation questions containing hallucinated attributes; 'procedural diagnosis' requires reasoning across modality recognition, organ identification, clinical findings, abnormalities, and positional grounding for each image [abstract].
- **Datasets.** ProbMed (Probing Evaluation for Medical Diagnosis), newly introduced [abstract]; size/splits not extracted (not stated in the abstract).
- **Experiments.** Compares top LMM performance on probing (original + negation) questions and on procedural multi-dimension diagnosis per image [abstract].
- **Results.** 'top-performing models like GPT-4o, GPT-4V, and Gemini Pro perform worse than random guessing on specialized diagnostic questions' [abstract]; 'models like LLaVA-Med struggle even with more general questions' [abstract]; CheXagent's results 'demonstrate the transferability of expertise across different modalities of the same organ' [abstract]. No specific numeric accuracy figures are given in the abstract.
- **Conclusions.** Current LMMs are 'still far from applicable' to medical diagnosis under rigorous (negation/hallucination-probed) evaluation, even when standard benchmark accuracy looks high [abstract].

> **Why it matters to us.** A strong external warning about benchmark inflation via standard accuracy on medical VQA -- directly relevant to why our project reports macro accuracy alongside a suite of walls/ceilings (coverage, selection, recoverability) rather than a single top-line accuracy number, and to why CLAUDE.md flags MMMU-Medical contamination as consequential. Suggests a possible future stress-test (negation-probing our own best-of-N pool) that we have not run.

<small>Read from: abstract-only.</small>

#### MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI

*Xiang Yue et al. · 2023 · CVPR 2024 (Oral) · [arXiv:2311.16502](https://arxiv.org/abs/2311.16502) · read priority 2*

**In one line.** General (not medical-only) massive multimodal benchmark spanning six disciplines including Health & Medicine, from which the field's 'MMMU-Medical' subset (our excluded, contaminated cell) is drawn.

- **Models.** 14 open-source LMMs, plus GPT-4V(ision) and Gemini evaluated [abstract].
- **Method.** 11.5K multimodal questions collected from college exams/quizzes/textbooks across 6 disciplines / 30 subjects / 183 subfields / 30 heterogeneous image types, testing 'advanced perception and reasoning with domain-specific knowledge' [abstract].
- **Datasets.** MMMU itself: 11.5K questions [abstract]; Health & Medicine is one of six top-level disciplines.
- **Experiments.** Benchmarks 14 open-source LMMs plus GPT-4V and Gemini Ultra [abstract].
- **Results.** 'Even the advanced GPT-4V and Gemini Ultra only achieve accuracies of 56% and 59% respectively' [abstract].
- **Conclusions.** Substantial headroom remains even for the strongest proprietary multimodal models on expert-level, college-exam-style multimodal reasoning [abstract].

> **Why it matters to us.** MMMU-Medical is the subset our project EXCLUDES from its 8-benchmark pool on contamination grounds (Lingshu-7B scores 0.80 vs. its own published 54.0 on MMMU, per CLAUDE.md SS0). This is the parent paper defining that subset's construction (exam/quiz/textbook sourced, expert-level, cross-disciplinary) -- useful for precisely describing what was excluded and why its provenance (public exam material, plausibly crawled pre-training) makes contamination plausible.

<small>Read from: abstract-only.</small>

#### SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering

*Bo Liu et al. · 2021 · ISBI 2021 · [arXiv:2102.09542](https://arxiv.org/abs/2102.09542) · read priority 2*

**In one line.** Bilingual (EN/ZH), physician-annotated medical VQA dataset with segmentation masks and a structured medical knowledge base for knowledge-grounded questions.

- **Models.** not extracted (baseline VQA models are evaluated; accuracy ranges given below, but specific model names were not in the fetched excerpt)
- **Method.** Physicians used an annotation system with pre-defined question templates per content type (modality, position, color, shape, size, plus knowledge-graph-triplet questions) and could choose, amend, or fully rewrite candidate questions [html, dataset-construction section].
- **Datasets.** SLAKE itself: 642 images (282 CT, 181 MRI, 179 X-ray) [html]; 14,028 QA pairs ('14K') [html]; official split 450 train / 96 val / 96 test images [html]. Bilingual: English and Chinese [html].
- **Experiments.** Baselines evaluated separately on vision-only vs. knowledge-based (KG-augmented) question subsets, and on open- vs. closed-ended questions [html].
- **Results.** Accuracy 'for vision-only tasks (ranging 72.73%-75.36%)' and 'for knowledge-based tasks (70.27%-75.01%)' [html, results table]. Paper notes baselines around 73% are 'still far away from practical use in the medical domain' [html].
- **Conclusions.** SLAKE is proposed as a richer alternative to VQA-RAD (bilingual, more modalities/body parts, segmentation masks, knowledge-base grounding) to facilitate Med-VQA development and evaluation [abstract+html].

> **Why it matters to us.** SLAKE-open is one of our 8 eval benchmarks. It is templated (predefined per-content-type question templates, physician-edited) rather than free clinical dialogue like VQA-RAD, so it is structurally distinct from VQA-RAD despite sitting in the same 'closed/open' bucket. Directly relevant: 2606.10066 (contamination audit, core in this category) flags 19.8% of SLAKE-En images as near-neighbours of PMC-OA under one detector -- a live question for whether our Lingshu SLAKE-open results reflect memorization rather than genuine capability.

<small>Read from: html.</small>

#### PathVQA: 30000+ Questions for Medical Visual Question Answering

*Xuehai He et al. · 2020 · arXiv preprint · [arXiv:2003.10286](https://arxiv.org/abs/2003.10286) · read priority 2*

**In one line.** Introduces PathVQA, the first pathology-image VQA dataset, built by semi-automatically mining textbook figure captions into QA pairs.

- **Models.** not extracted (dataset paper; early VQA baselines are evaluated but not named in the abstract)
- **Method.** Semi-automated pipeline: extract pathology images and captions from textbooks/online digital libraries, generate QA pairs from captions with NLP, then manually check every question for correctness [abstract].
- **Datasets.** PathVQA itself: '32,799 open-ended questions from 4,998 pathology images' [abstract]. Source material: pathology textbooks and online digital libraries, chosen because pathology images are otherwise private/inaccessible.
- **Experiments.** not extracted (abstract-only; no experiment table described in the abstract)
- **Results.** '32,799 open-ended questions from 4,998 pathology images where each question is manually checked to ensure correctness' [abstract]. No accuracy or split numbers given in the abstract.
- **Conclusions.** First dataset for pathology VQA, released publicly to promote research toward an 'AI Pathologist' able to pass board-certification-style exams [abstract].

> **Why it matters to us.** PathVQA-open is one of our 8 eval benchmarks. Knowing it is caption-derived (templated from textbook figure captions via NLP, not independently authored) rather than free clinician dialogue matters: it is structurally closer to SLAKE/OmniMedVQA than to VQA-RAD, and caption-derived answers can carry stylistic/text shortcuts unrelated to genuine image reasoning. Official train/val/test split sizes and any stated quality problems (e.g. yes/no skew) were not recoverable from the abstract alone within this run's fetch budget; flagged as 'not extracted' rather than guessed from memory.

<small>Read from: abstract-only.</small>


#### Also in this area (6), in brief

- **Auditing Data Leakage in Whole-Slide Image Multimodal Benchmarks** — Wenhao Zhang et al. (2026), [arXiv:2607.12278](https://arxiv.org/abs/2607.12278). Finds 92.3-100% case-level train/test overlap on TCGA-derived whole-slide-image (pathology) VQA benchmarks, meaning reported 'zero-shot' WSI-VLM performance largely reflects memorized patient/institution artifacts, not reasoning.<br><small>*For us:* Not one of our 8 benchmarks (we have no whole-slide-image benchmark; PathVQA is figure/caption-derived, not WSI) -- included as field context for how severe contamination/leakage can get in medical VLM benchmarking (92-100%, an order of magnitude worse than the 19.8% SLAKE finding in 2606.10066), useful for calibrating how seriously to take our own MMMU-Medical exclusion and SLAKE caution.</small>
- **Identifying and Resolving Pitfalls of Knowledge-Based VQA Benchmarks: Auditing, Repairing, and Augmenting** — Qian Ma et al. (2026), [arXiv:2607.00159](https://arxiv.org/abs/2607.00159). Audits knowledge-based VQA (KB-VQA, general-domain, not medical) benchmarks and finds systematic answer-derivability, question-underspecification, and visually-trivial-scene flaws that inflate/distort model rankings; proposes an audit-and-repair protocol.<br><small>*For us:* Not medical and not one of our 8 benchmarks, but directly analogous in spirit to our own benchmark-quality concerns (SLAKE contamination, PMC-VQA answer-position skew, MMMU-Medical exclusion): another field example of 'accuracy on a flawed benchmark overstates a real capability,' reinforcing why our project treats per-cell/per-benchmark auditing (not just pooled macro accuracy) as necessary.</small>
- **MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding** — Yuxin Zuo et al. (2025), [arXiv:2501.18362](https://arxiv.org/abs/2501.18362). A 4,460-question expert-level medical benchmark (Text and multimodal MM subsets) built with explicit data-synthesis steps to mitigate leakage.<br><small>*For us:* MedXpertQA-MM is one of the six MedVLThinker-Eval-era benchmarks per CLAUDE.md SS8, though it is not in this task's explicit list of our current 8 live open-ended benchmarks (PathVQA/SLAKE/VQA-RAD/RadImageNet-VQA/Kvasir-VQA-x1/OmniMedVQA/VQA-Med-C4/GEMeX) -- worth double-checking whether MedXpertQA-MM is still in the live pool or was superseded. Its explicit anti-leakage data-synthesis design is a useful methodological contrast to the contamination problems documented elsewhere in this category (2606.10066, 2607.12278) for benchmarks that did not take that precaution.</small>
- **Kvasir-VQA: A Text-Image Pair GI Tract Dataset** — Sushant Gautam et al. (2024), [arXiv:2409.01437](https://arxiv.org/abs/2409.01437). The original GI-tract VQA dataset (derived from HyperKvasir + Kvasir-Instrument) that Kvasir-VQA-x1 later expanded roughly 25x with harder, LLM-generated questions.<br><small>*For us:* Not itself one of our 8 benchmarks (we use its successor, Kvasir-VQA-x1) but is its direct ancestor dataset -- useful for provenance: the underlying images trace back to HyperKvasir/Kvasir-Instrument, and Kvasir-VQA-x1's 159,549 new QA pairs were added on top of this original 6,500-image pool.</small>
- **BESTMVQA: A Benchmark Evaluation System for Medical Visual Question Answering** — Xiaojie Hong et al. (2023), [arXiv:2312.07867](https://arxiv.org/abs/2312.07867). A tool/system (not a new dataset) for auto-generating Med-VQA datasets from raw clinical data and running a library of SOTA models under one unified experimental setup, aimed at the field's data-scarcity and reproducibility problems.<br><small>*For us:* A direct analogue, at the tooling level, to what MedEvalKit is for our own project (a 'unified experimental setup' harness) -- though BESTMVQA's differentiator is also auto-generating new datasets from raw clinical data, which we do not do (we only reformat/subset existing public benchmarks). The reproducibility problem it names -- 'many existing models have not been thoroughly evaluated in a unified experimental setup' [abstract] -- is the same problem CLAUDE.md SS0/SS8 describes our project having solved for Lingshu by adopting MedEvalKit.</small>
- **VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019** — Asma Ben Abacha et al. (2019), . Overview of the ImageCLEF 2019 shared task organizing radiology VQA into four categories (Modality, Plane, Organ system, Abnormality) over a 3,200-image training set.<br><small>*For us:* Our project uses 'VQA-Med 2019 C4-Abnormality' as one of our 8 open-ended eval benchmarks -- i.e. specifically the hardest, generation-style category (Abnormality), not the classification-style Modality/Plane/Organ categories this overview paper also defines. That distinction matters: our benchmark is a curated subset of the full 2019 shared task, not the whole thing.</small>

## 3.9 The older lineage: n-best rescoring, discriminative reranking, and confidence models in other fields

Long before anyone trained a probe on a vision-language model's hidden states, several other fields had already converged on the same basic recipe: let a big generative or search model produce several candidate outputs, then use a small, separately-trained model to score each candidate and pick the best one, trained on whether that candidate was actually correct. Speech recognition calls this rescoring an n-best list and has shipped dedicated 'confidence estimation modules' since at least 2020; parsing and machine translation called the same move discriminative reranking in the early 2000s; structural biology's CASP competitions have run a standing 'estimation of model accuracy' track for decades, and AlphaFold's own pLDDT head is exactly this pattern computed inside the network's own forward pass; molecular docking (DiffDock), object detection (IoU-Net) and code generation (LEVER, CodeRanker, AlphaCode, Codex's pass@k) each independently reinvented a close cousin. What differs across these lineages is only what the scorer reads (softmax probabilities vs. hidden states vs. execution traces), how it is trained (pointwise correctness vs. pairwise/listwise preference — Bradley-Terry is the classical pairwise-comparison model behind the latter, and the objective this project tested and moved away from), and what it is used for (reranking a fixed list vs. filtering vs. reporting confidence alongside an always-given answer). Our method — a frozen model's own hidden states, mean-pooled over a candidate's generated tokens, scored pointwise by a small trained head, used to pick among best-of-N samples — sits squarely inside this lineage: R-EBM (2021, ASR) is architecturally close to the same object five years earlier, and AlphaFold's pLDDT is the same idea computed in the same forward pass. The point of this category is not novelty-hunting for its own sake — it is that reading it lets us describe our contribution honestly, as the medical open-ended-VQA instantiation plus the identifiability/selection-wall analysis, not as an invention of the mechanism.

#### ★ Residual Energy-Based Models for End-to-End Speech Recognition

*Qiujia Li et al. · 2021 · Interspeech 2021 · [arXiv:2103.14152](https://arxiv.org/abs/2103.14152) · read priority 1 · **PDF in `papers/`***

**In one line.** A 2-layer BLSTM reranker reads an existing autoregressive ASR model's decoder hidden state, attention context, token embeddings and top-K softmax probabilities, mean-pools them over each hypothesis's own output tokens, and is BCE-trained to rerank an n-best list — architecturally the closest ancestor of our probe, five years earlier.

- **Models.** Base: an attention-based encoder-decoder (LAS-style) ASR model, including a self-supervised wav2vec 2.0 variant [abstract]. R-EBM: 2 layers of LSTMs, 512 units per direction, bidirectional [html, arxiv.org/html/2103.14152].
- **Method.** Residual energy-based model (R-EBM) trained as a discriminator between correct and incorrect output sequences. Per output-token step it reads four feature types — the current decoder hidden state, the acoustic context vector from attention, the output token embedding, and the top-K softmax probabilities [html] — then mean-pools the per-token hidden representations before an output layer [html], and is trained with binary cross-entropy to separate correct from incorrect sequences [html]. At inference, with a shared partition function the n-best hypotheses are re-ranked by joint (base-model + R-EBM) score to select the best candidate [html] — i.e. n-best rescoring / reranking, not resampling.
- **Datasets.** 100hr LibriSpeech subset (test-clean / test-other splits) [abstract].
- **Experiments.** Reranks n-best hypotheses from the base autoregressive decoder, measured as WER reduction and as confidence-estimation quality (precision-recall AUC) on the R-EBM's own scores; repeated on a wav2vec-2.0-based base model [abstract].
- **Results.** WER reduced by 8.2%/6.7% (test-clean/test-other) [abstract]; area under precision-recall curve of confidence scores improved by 12.6%/28.4% (test-clean/test-other) [abstract]; "R-EBMs still significantly improves both the WER and confidence estimation performance" on the wav2vec 2.0 base model [abstract].
- **Conclusions.** A small sequence model reading a frozen(-ish) generator's internal states, pooled over the candidate's own tokens and BCE-trained, is simultaneously a good reranker and a good confidence estimator [abstract].

> **Why it matters to us.** THE closest architectural ancestor we found: reads the base model's own decoder hidden states (not just its output probabilities), pools over the candidate's own generated tokens, trains pointwise with BCE, and reranks a fixed n-best list to pick one candidate — i.e. dimensions A (reads internal states, partially: also uses non-hidden-state features), B (mean-pooled over own tokens), and C (BCE / pointwise correctness training) of our four-dimension design space, essentially done in ASR in 2021. It predates our medical-VLM instantiation by five years and is evidence our mechanism is a considered re-instantiation of an established paradigm, not a novel one.

<small>Read from: html.</small>

#### ★ Highly accurate protein structure prediction with AlphaFold

*John Jumper et al. · 2021 · Nature 596, 583–589 (2021) · [doi:10.1038/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2) · read priority 1 · **PDF in `papers/`***

**In one line.** AlphaFold's pLDDT confidence head is a small per-residue network computed on the network's own final activations, in the same forward pass, and is used to estimate per-residue accuracy and to rank/select among predicted structures — model quality assessment folded directly into the generator.

- **Models.** AlphaFold: an end-to-end deep network (Evoformer + structure module) predicting 3D protein structure from sequence (and MSA/template features) [PMC fetch, pmc.ncbi.nlm.nih.gov/articles/PMC8371605].
- **Method.** "Predictions of side-chain χ angles as well as the final, per-residue accuracy of the structure (pLDDT) are computed with small per-residue networks on the final activations at the end of the network" [PMC fetch] — i.e. a small head reused for confidence, sharing the main forward pass rather than a separate scoring pass. pLDDT is used to rank/select among predicted models and reported per-residue as a reliability estimate [PMC fetch]. This is our dimensions A (reads the model's own internal activations) + C (correctness/accuracy-supervised) + D (used to rank/select candidates), missing only B (candidate-token pooling — AlphaFold has one structure per forward pass, not an n-best pool at this stage) [author's characterisation per task framing].
- **Datasets.** CASP14 (14th Critical Assessment of protein Structure Prediction) blind test set; PDB structures for training [PMC fetch].
- **Experiments.** Blind CASP14 competition evaluation of predicted-vs-experimental backbone and all-atom accuracy; calibration of pLDDT against the true lDDT-Cα accuracy metric [PMC fetch].
- **Results.** Median backbone accuracy of 0.96 Å r.m.s.d.95 (95% CI 0.85–1.16 Å) vs. 2.8 Å r.m.s.d.95 for the next-best method [PMC fetch]; all-atom accuracy 1.5 Å r.m.s.d.95 vs. 3.5 Å r.m.s.d.95 for competitors [PMC fetch]. pLDDT calibration: lDDT-Cα = 0.997 × pLDDT − 1.17, Pearson's r = 0.76 [PMC fetch].
- **Conclusions.** A confidence output computed in the same forward pass as the prediction, from the network's own final activations, gives a reliable, well-calibrated per-residue accuracy estimate at essentially zero extra cost [PMC fetch].

> **Why it matters to us.** The single clearest example outside NLP/speech of 'read the generator's own internal activations, in the same pass, to output a correctness/quality score used to rank/select' — our core design pattern, achieving field-defining accuracy. Differs from us in not pooling over an n-best/best-of-N candidate pool at the confidence-head stage (one structure per pass) and in training pLDDT against a continuous accuracy metric (lDDT) rather than our binary correctness label — a useful contrast point (pointwise regression vs. our pointwise BCE classification).

<small>Read from: html.</small>

#### ★ Confidence Estimation for Attention-based Sequence-to-sequence Models for Speech Recognition

*Qiujia Li et al. · 2020 · Submitted to ICASSP 2021 [comment field] · [arXiv:2010.11428](https://arxiv.org/abs/2010.11428) · read priority 1 · **PDF in `papers/`***

**In one line.** A single fully-connected layer (256 units) reads the decoder state, attention context and token embedding of an existing seq2seq ASR model and is BCE-trained on edit-distance-derived correct/incorrect token labels, beating raw softmax probability as a confidence signal on AUC and normalized cross entropy (NCE).

- **Models.** Attention-based sequence-to-sequence ASR model (encoder-decoder with attention) [abstract]; the confidence estimation module (CEM) itself is a single FC layer, 256 units, sigmoid output [html, arxiv.org/html/2010.11428].
- **Method.** CEM is "a lightweight module that can be easily configured on top of any attention-based sequence-to-sequence model" [html]. It gathers the attention context, decoder state and current token embedding, feeds them into the FC layer, and outputs a per-token confidence in [0,1] via sigmoid [html]. Training labels come from edit-distance alignment of n-best hypotheses against references: correct tokens are labeled 1, substituted/inserted tokens 0 [html]. Trained by minimizing binary cross-entropy between predicted confidence and the target label [html]. Evaluated against the decoder's own softmax probability as baseline.
- **Datasets.** LibriSpeech [abstract]; also tested on a moderately mismatched domain to check generalisation [abstract].
- **Experiments.** Token-level confidence quality compared to softmax-probability baseline, with and without shallow fusion of a language model (PWLM) [abstract, html].
- **Results.** On LibriSpeech test-clean/test-other with PWLM shallow fusion: AUC improves from 0.976/0.912 (softmax) to 0.990/0.958 (CEM); NCE improves from 0.166/0.172 (softmax) to 0.344/0.275 (CEM) [html, arxiv.org/html/2010.11428]. Softmax confidence shows "a sharp downward spike at the high-confidence region" that CEM removes [html].
- **Conclusions.** Reading internal seq2seq state through a tiny trained head gives calibrated, more reliable confidence than the generator's own softmax, and generalises across a domain shift [abstract].

> **Why it matters to us.** The field's name for our probe's ASR cousin: a token/candidate-level confidence head trained pointwise (BCE) on correctness labels, reading internal decoder state rather than only output probabilities. Confirms that 'read internal state, not just output logits/probabilities' is a known and load-bearing design choice elsewhere, and gives us the term 'confidence estimation module' as the field-standard name for this component.

<small>Read from: html.</small>

#### Fault-Aware Neural Code Rankers

*Jeevana Priya Inala et al. · 2022 · NeurIPS 2022 [comment field] · [arXiv:2206.03865](https://arxiv.org/abs/2206.03865) · read priority 2*

**In one line.** CodeRanker predicts a sampled program's correctness (and even its likely error type, e.g. IndexError) WITHOUT executing it, addressing the case where unit tests aren't available or execution is unsafe — a learned, execution-free verifier much closer to our own probe than AlphaCode's execution-based clustering.

- **Models.** A neural ranker (CodeRanker) trained on top of sampled outputs from code-generation LLMs (Codex, GPT-Neo, GPT-J) [abstract].
- **Method.** "Fault-aware" ranker: instead of running the code, it is trained to predict whether a sampled program is correct, and specifically what kind of compile/runtime error it would produce if wrong (e.g. IndexError, TypeError) [abstract] — a richer pointwise supervision signal than binary correct/incorrect, though still a pointwise classifier over one candidate at a time.
- **Datasets.** APPS, HumanEval, MBPP [abstract].
- **Experiments.** pass@1 accuracy with vs. without CodeRanker-based selection among sampled programs from Codex/GPT-Neo/GPT-J, without executing the candidates [abstract].
- **Results.** "significantly increase the pass@1 accuracy of various code generation models (including Codex, GPT-Neo, GPT-J) on APPS, HumanEval and MBPP datasets" [abstract] — exact deltas not stated in the abstract; not extracted.
- **Conclusions.** A learned ranker can substitute for execution when unit tests/safe execution are unavailable, and predicting fine-grained fault types (not just correct/incorrect) is a useful auxiliary training signal [abstract].

> **Why it matters to us.** The code-generation paper structurally nearest to ours: no execution oracle assumed, a trained pointwise classifier scores each sampled candidate, used to pick the best one — directly analogous to our BCE-trained probe picking among 8 sampled medical answers with no executable ground truth. The fault-type auxiliary objective (predict *why* wrong, not just *whether* wrong) is a concrete idea we have not tried and could adapt (e.g. predicting failure mode alongside correctness).

<small>Read from: abstract-only.</small>

#### ★ Discriminative Reranking for Natural Language Parsing

*Michael Collins and Terry Koo · 2005 · Computational Linguistics 31(1), pp. 25–70 · [doi:10.1162/0891201053630273](https://aclanthology.org/J05-1003/) · read priority 2 · **PDF in `papers/`***

**In one line.** A boosting-based discriminative reranker rescores an n-best list of candidate parse trees from a baseline generative parser, combining the baseline's log-likelihood with hundreds of thousands of additional tree features — one of the two founding papers of discriminative reranking in NLP.

- **Models.** Baseline: a generative statistical parser (Collins-style) producing an n-best list of candidate parses; reranker: a boosting-based discriminative model over tree features [WebFetch of aclanthology.org/J05-1003, search snippet].
- **Method.** The reranking task is framed as a ranking problem solved via a boosting approach; the reranker's score combines the baseline generative model's log-likelihood with evidence from roughly 500,000 additional features over parse trees that are not part of the original generative model [WebSearch snippet, not independently verified against full text]. This is discriminative reranking of a fixed n-best pool, i.e. the parsing-field analogue of best-of-N selection.
- **Datasets.** Penn Treebank Wall Street Journal parsing benchmark [general knowledge of the parsing literature at the time; not independently confirmed from a fetched abstract].
- **Experiments.** Not extracted (abstract text was not retrievable from the ACL Anthology landing page in this pass; numeric F-score results should be read from the PDF, which is the point of marking this card core).
- **Results.** not extracted — no numeric results could be confirmed from the fetched page or search snippets; do not cite a number for this paper until the PDF is read.
- **Conclusions.** not extracted (see above); the paper is widely cited as establishing that a discriminatively-trained reranker over a generative model's n-best output can improve on the generative model alone.

> **Why it matters to us.** The NLP-parsing instance of the same recipe one field earlier than ASR: fixed n-best pool from a generative model, small separately-trained discriminative scorer combining base-model score with additional learned features, used to pick the final output. Gives us the term 'discriminative reranking' as the general-NLP name for what our probe does to a best-of-N pool, and a second, independent-of-ASR confirmation that this recipe predates our project by two decades.

<small>Read from: html.</small>


#### Also in this area (11), in brief

- **The Evolution of Reranking Models in Information Retrieval: From Heuristic Methods to Large Language Models** — Tejul Pandit et al. (2025), [arXiv:2512.16236](https://arxiv.org/abs/2512.16236). A chronological survey of reranking models in IR/RAG — from heuristic methods through cross-encoders, T5-style sequence generation, and graph neural networks, to LLM-based reranking and prompting/fine-tuning strategies, plus distillation for efficiency.<br><small>*For us:* A 2025 confirmation that reranking-over-a-candidate-list remains an actively evolving, named subfield in IR/RAG, useful for citing the field's current shape and for cross-checking that our 'verifier reranks best-of-N' framing matches contemporary IR usage of 'reranking'.</small>
- **Large Language Models for Information Retrieval: A Survey** — Yutao Zhu et al. (2023), [arXiv:2308.07107](https://arxiv.org/abs/2308.07107). Survey mapping the confluence of LLMs and information-retrieval systems across query rewriters, retrievers, rerankers and readers, situating rerankers as one stage in a larger modern IR pipeline.<br><small>*For us:* Orientation/vocabulary source: places 'reranker' within the broader retriever→reranker→reader pipeline, useful for correctly using IR terminology when describing our verifier as a reranker-like component in the paper's related work.</small>
- **LEVER: Learning to Verify Language-to-Code Generation with Execution** — Ansong Ni et al. (2023), [arXiv:2302.08468](https://arxiv.org/abs/2302.08468). LEVER trains a verifier that reads the natural-language input, the generated program, AND its execution result, combines the verifier score with the LM's own generation probability, and marginalizes over programs sharing the same execution result — a hybrid of learned verification and execution evidence.<br><small>*For us:* Shows the code field converging on a fusion of learned-verifier-score + generator-signal + (there) execution evidence — directly analogous to our own 'certified veto' / fusion mechanisms that combine the trained probe's score with the generator's own signal rather than using the probe alone; useful precedent when we justify combining verifier confidence with cheap-model signal instead of using either in isolation.</small>
- **DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking** — Gabriele Corso et al. (2022), [arXiv:2210.01776](https://arxiv.org/abs/2210.01776). DiffDock generates multiple candidate ligand poses via a diffusion model and uses a separate learned confidence model to rank/select among them, reporting 'high selective accuracy' — a docking-field instance of generate-many-then-score-and-pick.<br><small>*For us:* A close structural cousin to our whole pipeline (sample multiple candidates from a generative model, train a separate small model to score/select among them), but from structural biology/chemistry rather than language. 'Confidence model' is DiffDock's own name for this component and is another field-specific synonym set (alongside verifier, reranker, R-EBM, CEM) for what our probe is.</small>
- **Competition-Level Code Generation with AlphaCode** — Yujia Li et al. (2022), [arXiv:2203.07814](https://arxiv.org/abs/2203.07814). AlphaCode generates very large numbers of candidate programs per competitive-programming problem, then filters/clusters them by execution behaviour on example inputs to shortlist a small final submission set — large-scale sampling plus execution-based clustering, not a learned scorer.<br><small>*For us:* A useful negative contrast: AlphaCode gets away with execution-based clustering instead of a trained scorer because competitive-programming problems come with cheaply checkable example test cases. Medical open-text VQA has no such cheap oracle, which is precisely the gap our trained hidden-state probe is built to fill — worth citing when justifying why a *learned* verifier, rather than a heuristic filter, is necessary in our setting.</small>
- **Learning Word-Level Confidence For Subword End-to-End ASR** — David Qiu et al. (2021), [arXiv:2103.06716](https://arxiv.org/abs/2103.06716). Extends the ASR confidence-model recipe to word-level (rather than subword-token-level) confidence using self-attention over multiple hypotheses, needed because word-piece tokenization makes ground-truth correctness labels ambiguous at the token level.<br><small>*For us:* A close relative of CEM/R-EBM showing the field iterating on exactly the granularity-of-pooling question our own design faces (we mean-pool a candidate's hidden states over its own generated tokens); this paper shows naive subword aggregation is not enough and a learned pooling/attention mechanism over hypotheses does better — a candidate ablation direction we have not tried.</small>
- **Evaluating Large Language Models Trained on Code** — Mark Chen et al. (2021), [arXiv:2107.03374](https://arxiv.org/abs/2107.03374). Introduces Codex and the pass@k metric, showing that repeated sampling (many candidate programs per problem) followed by selection is a surprisingly effective way to solve hard coding problems, even before any learned selector is added.<br><small>*For us:* Defines pass@k / oracle-of-k, the code-generation analogue of our 'oracle@8' upper bound — the ceiling our verifier is trying to approach by picking, rather than being given, the correct candidate. Codex itself doesn't need a learned selector because unit-test execution is a free, perfect oracle; our setting lacks that free oracle (no unit tests for medical open-text answers), which is exactly why a trained verifier is necessary at all — a useful contrast for motivating our method.</small>
- **Passage Re-ranking with BERT** — Rodrigo Nogueira and Kyunghyun Cho (2019), [arXiv:1901.04085](https://arxiv.org/abs/1901.04085). A BERT-based pointwise reranker (later widely known as monoBERT) scores each retrieved passage independently given the query, becoming state of the art on MS MARCO passage retrieval and TREC-CAR — the canonical modern pointwise-reranking baseline in IR.<br><small>*For us:* The canonical modern IR instance of pointwise reranking: scores each candidate independently (like our probe scores each of the 8 sampled answers independently) rather than comparing pairs. Establishes the IR field's term 'reranking' for exactly the selection step our probe performs over a best-of-N pool, and is the ancestor cited by almost every later cross-encoder / LLM reranker paper (see the two survey cards below).</small>
- **Acquisition of Localization Confidence for Accurate Object Detection** — Borui Jiang et al. (2018), [arXiv:1807.11590](https://arxiv.org/abs/1807.11590). IoU-Net trains a small head to predict the IoU between each detected bounding box and the (unknown at test time) ground-truth box, giving detectors a localisation confidence distinct from classification confidence, which improves non-max suppression and enables IoU-guided box refinement.<br><small>*For us:* A structurally distant but conceptually identical move: add a small trained head whose only job is to predict a correctness-adjacent quantity (IoU with ground truth) that the main network's own output score does not capture, and use it to select/refine among candidates (boxes here, sampled answers for us). Reinforces our framing that 'the generator's own confidence signal is not the same thing as a trained correctness predictor', one of our stated findings (AUROC gap between MCQ and open text is a related but distinct instance of this).</small>
- **Discriminative Reranking for Machine Translation** — Libin Shen et al. (2004), . Applies discriminative reranking (following Collins' parsing work) to machine translation n-best lists, one of the two founding discriminative-reranking papers and the one that brought the technique into MT.<br><small>*For us:* A second field (MT) adopting the same n-best-plus-discriminative-reranker recipe in the same two-year window as the parsing paper — evidence the recipe was independently useful across at least three classical NLP tasks (parsing, MT) before speech (CEM/R-EBM) picked it up again in 2020-21. Included for completeness of the 'discriminative reranking' lineage rather than for any number we can currently quote.</small>
- **Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons** — Ralph A. Bradley and Milton E. Terry (1952), [10.2307/2334029](https://doi.org/10.2307/2334029). The founding statistical model for ranking items from pairwise comparisons: P(i beats j) = pi_i / (pi_i + pi_j), estimated from win/loss counts — the model underlying every 'Bradley-Terry loss' used for pairwise preference training since (RLHF reward models, and the pairwise objective this project itself tested and moved away from).<br><small>*For us:* Defines the pairwise-comparison objective this project explicitly tested as its earlier objective before settling on pointwise BCE (per project context) — i.e. this is the mathematical ancestor of the road NOT taken in our current design, making it important background even though we do not use it. Also the mathematical basis of every 'pairwise preference'-style reward/verifier model (RLHF and otherwise) that a reviewer might expect us to compare against or justify not using.</small>


---

# 4. Where we sit — honest positioning

This section answers the question a professor, a reviewer and a competitor all ask first:
**has this been done before, and if so what is left?** It is a condensed and re-verified version of
`results/cascade_methods/docs/current/PRIOR_ART_PROBE_VERIFIER_2026-09-14.md`, which searched seven
literatures in parallel and independently re-read the closest hits.

## 4.1 The answer: yes, the mechanism is a known family

Our mechanism has four dimensions:

- **(A)** a lightweight classifier on **frozen** hidden states,
- **(B)** used to **select** among sampled candidates,
- **(C)** pooled over the **candidate's own generated tokens**,
- **(D)** **reused from the generation pass** at no extra forward pass.

All four together are published, repeatedly, in text-only LLMs. The names the field already uses:
**hidden-state reward model**, **latent verifier**, **intrinsic reward from hidden states**,
**confidence estimation module** (speech), **model quality assessment** (structural biology),
**confidence model** (docking). The nearest instances, each carded in §3.3:

| paper | what it does | dimensions |
|---|---|---|
| **Q-Probe** (arXiv:2402.14688) | linear probe on the frozen residual stream for reward maximisation; "essentially free in comparison to the base model" | A+B, earliest LLM instance found |
| **LiLaVe** (arXiv:2504.16760) | gradient-boosted trees on frozen hidden states at the decoded answer's last tokens; selects the highest-scoring candidate in best-of-n — *and it tried an MLP and rejected it* | A+B+C |
| **ELHSR / SWIFT** (arXiv:2505.12225) | gated linear head over per-token hidden states of the generated response, **BCE on binary correctness**, best-of-N by argmax, ~1.8–3.0×10⁵ params | A+B+C — our loss and our role at 1/100th our size |
| **ReProbe** (arXiv:2511.06209) | <10M-param probe on a frozen LLM, BCE with class weighting, best-of-N argmax; step vector is the **mean over the model's own generated tokens** | A+B+C — dimension (C) exactly |
| **CASE** (arXiv:2608.17124) | logistic regression on frozen activations at the answer token, N=16–20 at **T=0.7**, on **medical LLMs** and medical benchmarks | A+B+C+D, medical — minus the image |
| **HSRM** (arXiv:2608.30841) | ~2M-param encoder on a frozen generator's states, ranks best-of-N; sells "reusing representations already computed during generation … no additional generator forward passes" | **A+B+C+D — our exact pitch, in print** |

**Two papers match all four dimensions**, confirmed by independent re-reading (§3.3):
**ELHSR/SWIFT** (arXiv:2505.12225) and **HSRM** (arXiv:2608.30841). HSRM states the
no-extra-forward-pass claim most explicitly; **ELHSR pools over *all* generated tokens, which is
closer to our mean-pool than HSRM's step-boundary subset.** ELHSR is therefore the single nearest
published neighbour to our mechanism, and the paper to distinguish ourselves from first.

**Two consequences, and they are not negotiable.**

1. **Do not claim the mechanism.** It is a named sub-literature with at least five independent
   instantiations.
2. **Do not claim "verification is free" as the contribution.** HSRM, CASE and Q-Probe make that
   argument in their own abstracts. We may *state* the cost — it is ours and it is measured — as a
   property, not a discovery.

## 4.2 The lineage is older than LLMs

Worth one slide, because showing you know it is what separates a re-instantiation from a
reinvention (all carded in §3.9):

- **Speech recognition, 2021 — the true ancestor.** **R-EBM** (arXiv:2103.14152): a **2-layer,
  512-unit BLSTM** taking four input features *including the frozen ASR model's decoder hidden
  state*, **mean-pooled over the candidate's own output tokens**, trained with **binary
  cross-entropy**, used to **re-rank an n-best list**. That is A+B+C, five years before us, and these
  details are read from the paper rather than inferred. Its lineage: **CEM** (arXiv:2010.11428) — a
  single 256-unit fully-connected layer trained with BCE on edit-distance labels, AUC 0.976 → 0.990
  (test-clean) and 0.912 → 0.958 (test-other) — then R-EBM, then word-level confidence.
  **Dimension (D) is where speech stops:** R-EBM never argues the cost case and reports no FLOPs.
- **Structural biology, 2021.** AlphaFold2's **pLDDT** is small per-residue networks on the final
  activations, computed inside the same forward pass, used to pick the best of five predicted
  structures — A(partial)+B+C+D. The field calls this *model quality assessment* (MQA).
- **Molecular docking.** DiffDock (arXiv:2210.01776) samples N poses and ranks them with a 5M-param
  **confidence model** trained on binary RMSD<2Å labels.
- **Object detection.** IoU-Net (arXiv:1807.11590) predicts box quality from features the detector
  already computed and ranks candidates with it.
- **Code generation.** AlphaCode, LEVER and CodeRanker filter/rerank sampled programs; `pass@k`
  (arXiv:2107.03374) is where our "oracle@N" comes from.

## 4.3 The residual — stated as narrowly as it deserves

Three independent search angles returned the same negative:

> **No verified paper applies a probe on generation-pass internal states as a best-of-N selector in a
> VISION-LANGUAGE model.** Every A+B+C+D instance found is text-only, and almost all are mathematical
> reasoning.

The medical/VLM literature splits into two halves that do not touch, and **two September-2026 papers
are near-collisions that must be cited and distinguished**:

- **MedProb** (arXiv:2609.04336) — **less of a collision than it first appears, and the distinction is
  worth making precisely.** Its *headline* method is not a best-of-N verifier at all: it is a
  **pre-generation, question-only classifier on the last input token** for multiple choice, and it
  never generates free text. Only its **Appendix H** extends to open-ended medical VQA as a
  probe-as-selector, where the candidate is fed **back in as input**, costing a second forward pass per
  candidate — exactly the LoRA design we replaced. Single per-layer logistic regression, 100 test
  examples, and the authors call it "an initial demonstration."
  <br>⚠️ Appendix H's body text could not be reached through arXiv's HTML or PDF converters on
  2026-09-16 after two attempts; the re-feeding claim is carried from our own earlier prior-art read
  and is **not independently re-verified verbatim**. Confirm it from the PDF before putting it in a paper.
- **DualRead** (arXiv:2609.06419) — mean-pools hidden states over the model's **own generated answer
  tokens** on SLAKE / VQA-RAD / PathVQA / PMC-VQA / OmniMedVQA with Qwen3-VL and MedGemma — our
  pooling, our benchmarks, a generator family we also use. But it **calibrates confidence for one
  answer** and never selects among candidates, and it pays a teacher-forced replay rather than
  reading the generation pass.

Neither is fatal. Both would be fatal if a reviewer found them and we had not.

## 4.4 What is genuinely ours, ranked by defensibility

**Tier 1 — defensible.**

1. **The modality transfer, stated narrowly.** First application of *generation-pass hidden-state
   verification* to vision-language and to medical VQA, with the image in the loop. Narrow, but
   nothing found contradicts it.
2. **The empirical map.** Eight benchmarks, 36,869 questions, four temperatures, three generator
   families, ~24 measured negatives with bounds rather than nulls. No paper in §3.3 evaluates on more
   than a handful of datasets; ELHSR, ReProbe and HSRM are mathematical reasoning, CASE is text-only
   medical, MedProb's demonstration is 100 examples. **This is the strongest thing we have**, and it
   is a *characterisation* contribution, not a mechanism one.
3. **The generator-family replication.** The same recipe on Lingshu-7B, Qwen2.5-VL-7B and
   MedGemma-4b-it — with the MedGemma ensemble layers **pre-registered by relative depth before any
   probe was fitted**. Pre-registration inside an ablation is rare and is worth saying out loud.
4. **The LOBO limit as a positive result.** Breadth is worth −0.0056 and a benchmark's own training
   half +0.0485, with a measured onboarding price (~100 labelled questions). Papers in this family
   report a gain; almost none reports *where the gain does not transfer*. Publishing the boundary is a
   contribution, provided we lead with it rather than bury it.

**Tier 2 — defensible with a concession.**

5. **The decomposition** `verifier − greedy = selection skill − sampling penalty`, and the exact
   identity `selected = oracle@N × selection_efficiency`. **Concede the framework** to Hu
   (arXiv:2607.17531), who published oracle gap / signal fidelity / recoverable mass / conditional
   harm on 2026-07-20, before our 2026-08-24 artifact. What survives is the multimodal-medical
   instantiation and the fact that our identity is *multiplicative and exact* where his is additive.
6. **The answer-prior control.** Scoring candidates by P(correct | answer string) with no image and no
   hidden state, and beating it on all eight benchmarks including the one where it loses to greedy.
   Simple, and it is the control that a reviewer would otherwise demand. Not novel as an idea;
   valuable as evidence.

**A distinction to make explicitly, because the nearest neighbours blur it.** Several 2025–2026
methods also read hidden states of sampled candidates — *Latent Self-Consistency* (arXiv:2508.18395)
and *Embedding-space Agreement* (arXiv:2606.12003) among them — but they score **inter-candidate
agreement**: a candidate is good if it sits near the others in representation space. Ours scores
**ground-truth-anchored correctness**: a candidate is good if a probe trained on judged correctness
labels says so. That is a different supervision signal, and it is why our method can beat
self-consistency on benchmarks where the majority is wrong — which is 74–90 % of recoverable questions
here (retrospective §5.3). Note also that Latent Self-Consistency is **not training-free** despite
sitting in the self-consistency family (it trains contrastive summary-token embeddings), which makes
it the closest published structural analogue to our trained probe among the consensus methods.

**Tier 3 — evidence, not novelty.**

7. The cost measurement, the audits, the corrections log. These make the numbers trustworthy; they
   are not claims.

## 4.5 The one-paragraph statement of contribution

> We take a mechanism that is established in text-only LLMs — a lightweight probe on a frozen
> generator's own hidden states, used as a best-of-N verifier (Q-Probe; LiLaVe; ELHSR; ReProbe; HSRM;
> and, in speech, R-EBM a decade earlier) — and ask whether it survives the move to vision-language
> models and to medical visual question answering, where the answer depends on an image the probe
> never sees directly. It does: +0.0736 macro over greedy decoding across eight open-ended medical VQA
> benchmarks on Lingshu-7B, replicated on two further generators including one from a different
> language-model family. We then map where it stops working: it is a per-benchmark method (breadth
> buys −0.0056; a benchmark's own labelled split buys +0.0485, at a price of roughly 100 questions),
> its failures are candidate-set failures rather than ranker failures on 7 of 8 benchmarks, and no
> tested property of a benchmark — including distance from the training distribution — predicts where
> it helps.

## 4.6 Neighbours to watch

Three groups are working close enough that a new preprint could change our position: the
hidden-state-reward-model line (HSRM, ELHSR, ReProbe), the medical-probing line (MedProb, CASE,
DualRead), and the medical best-of-N line (Best-of-Evidence, Wasserstein Equilibrium Decoding).
A monthly arXiv check on the first two is cheap insurance; `arxiv_verify.py` in the package tmp
directory is the tool.


---

# 5. Preparing the next professor report

The complaint, restated precisely: the work has progress but **no comparison context** — it reads as
though it were done in a vacuum, with no prior art, no external baselines, and non-standard
vocabulary. Those are three separable defects with three separable fixes. This section is the fix.

## 5.1 The three defects, and what each one actually requires

| defect | what the professor sees | the fix | where it comes from |
|---|---|---|---|
| **no prior art** | a method presented as if invented from nothing | a named lineage: say which published family this belongs to and cite the five or six nearest papers *by name* in the first minute | §3.3 and §4.1 |
| **no comparisons** | numbers with nothing to compare to except our own earlier numbers | (a) internal baselines already measured — greedy, answer prior, self-consistency, oracle@8; (b) external reference points from published papers, quoted as *context* with their protocol stated | §2.4.2, §5.4 |
| **wrong terminology** | "MLP", "cell", "head" — words that mean something else to the reader | the mapping table in §1, applied everywhere including slide titles and axis labels | §1.1–1.5 |

## 5.2 The opening paragraph to use

A report that starts with what was done this week reads as vacuum work. Start with the placement:

> *"The method is a **best-of-N verifier**: the 7B samples eight answers and a small scorer picks
> one. Our scorer is an **MLP probe on frozen hidden states** — a probing classifier, in the sense of
> Alain & Bengio (2016) and Belinkov (2021), used in the role that the test-time-compute literature
> calls a **verifier** or **outcome reward model** (Cobbe et al. 2021). This is an established family:
> Q-Probe (2024), LiLaVe, ELHSR and ReProbe (2025) and HSRM (2026) all train a lightweight scorer on a
> frozen LLM's hidden states and use it to select among sampled candidates. **We are not claiming the
> mechanism.** Our contribution is that, as far as a seven-angle prior-art search could establish, no
> one has done it in a **vision-language** model or on **medical VQA**, and that we characterise it
> far more thoroughly than any of them: eight benchmarks, 36,869 questions, three generator
> families, with the failure modes measured rather than asserted."*

That paragraph does all three fixes at once. Everything after it is progress.

**If asked to be more precise about what kind of verifier it is**, the field's taxonomy has two axes
and our scorer sits at a named point on both: it scores the **finished answer**, not the reasoning
steps, so it is an **outcome reward model (ORM)** rather than a process reward model; and it emits a
**scalar** trained with binary cross-entropy rather than generating its verdict as text, so it is
**discriminative** rather than generative (GenRM). Scoring each candidate independently makes it
**pointwise** rather than pairwise or listwise. So: *a pointwise, discriminative outcome reward model,
in the same family as Cobbe et al.'s original verifier — differing in that it reads hidden states
instead of attaching a head to the output logits.*

That corner of the taxonomy is **not a compromise**, and saying so pre-emptively is worth doing,
because "why not a process reward model?" is a predictable question. Multi-domain evidence
(arXiv:2510.00492, §3.2) finds discriminative ORMs **tie** discriminative PRMs outside math-adjacent
domains. And our answers are a few words long with no intermediate steps, so a PRM would have nothing
to grade.

## 5.3 Terminology substitutions to make before the next report

Apply these globally — in prose, in table headers, in figure captions, in slide titles.

| stop saying | say instead | why |
|---|---|---|
| "the MLP" / "the MLP head" | "the **MLP probe**" (first use: "an MLP probe on frozen hidden states, used as a best-of-N verifier") | "MLP" inside a transformer means the feed-forward block; "head" implies joint training with the backbone (§1.2) |
| "a 2-layer MLP" | "an MLP with **one hidden layer of width 256**, GELU activation" | "2-layer" is ambiguous between weight layers and hidden layers |
| "cell" | "**benchmark**" (or "the open-ended split of X") | "cell" is an internal coinage; it reads as a table cell |
| "pool" | "**candidate set**" | |
| "string prior" | "**answer-prior baseline**", defined at first use | |
| "the head beats greedy" | "the **verifier** beats **greedy decoding**" | |
| "sel_eff" | "**selection efficiency**, defined as (selected − greedy)/(oracle@N − greedy)" | not a standard name — always define it |
| "WIN / TIE / LOSS" | "significant at 95 % / not significant / significantly worse" | |
| "free" (of the probe) | "adds no additional forward pass; ~1.8 MFLOP per candidate per probe" | the "free" *argument* is already published (§4.2) — state it as a property |

## 5.4 The comparison tables to put in the report

**Table 1 — internal baselines (we have these; they are the reviewer's first question).**
Use §2.4.2: greedy / answer prior / self-consistency / verifier / oracle@8, per benchmark. The two
sentences that make it land: *the verifier beats the answer-prior baseline on all eight benchmarks*,
and *on GEMeX the answer prior loses to greedy while the verifier beats both* — i.e. it is doing
verification, not vocabulary memorisation.

**Table 2 — cost, with the convention named.** From `bestofn_vllm_2026-09-16.json`: best-of-8 is
2.74× latency and 3.59× energy over greedy on the vLLM shared-prefill path, 1.13 FLOP-eq by
forward-token count (8.0 "as charged"). Say which serving path and which convention, because the
same method measures 1.99× latency on the HF path.

**Table 3 — external context, clearly marked NOT protocol-matched.** This is the table that has been
missing. Published numbers from other groups on the same *task* give the professor a frame, provided
each row carries its protocol. Candidate rows, all from papers carded in §3 (take the exact values
from the cards, and reproduce their protocol column):
- **Wasserstein Equilibrium Decoding** (arXiv:2605.18313) — the nearest neighbour: training-free
  candidate selection on open-ended VQA-RAD and PathVQA, small VLMs, judged by Grok-4.1-Fast.
- **Best-of-Evidence** (arXiv:2607.20950) — best-of-16 selection on medical VQA with a 235B judge.
  This is the state of the art in medical best-of-N selection and **its gains over majority voting are
  small and mostly not significant** (e.g. +0.26 pp [−0.47, +0.99] on VQA-Med). Worth stating plainly:
  it puts our +0.0736 in a context where the published alternative, using a 235B verifier, buys under
  a point.
- **Verification Mirage** (arXiv:2605.10850) — zero-shot self-verification on five medical VQA
  datasets (VQA-RAD, PathVQA, SLAKE, PMC-VQA, MedXpertQA) across six VLMs **including Lingshu**. This
  is the evidence that the *untrained* version of our component fails: verifier error ≳40 %, false
  positive rate ≳60 %, ~95–100 % on differential diagnosis; a generator error carries **57× higher
  odds** of verifier failure; and in multi-turn loops **69.5–87.1 %** of initially wrong answers are
  locked in while only 2.2–3.8 % are corrected. It also reports that scaling the verifier within the
  same family gives **no significant gain for Lingshu (p = 0.782)** — which closes the "just use a
  bigger verifier" line for our exact model family before anyone suggests it
  (verified verbatim in `PRIOR_ART_2026-08-11.md` §3.3).
  Use it as **motivation, not a competitor**: they show the untrained kind fails; we built the
  trained one.
- **Lingshu** (arXiv:2506.07044) Table 6 — the generator's own published benchmark numbers.

⚠️ **The protocol column is not optional.** Our project's own standing rule forbids comparing
open-text numbers even across serving configurations (±0.008 per benchmark). Across different
judges, splits and generators, a shared table is context, never a ranking. Label it
*"Published results on related tasks — protocols differ; not a controlled comparison."*

## 5.5 The claim to make, and the four claims not to make

**Make this claim:**
> An MLP probe on frozen Lingshu-7B hidden states, used as a best-of-8 verifier, improves open-ended
> medical VQA accuracy by **+0.0736 macro** over the model's own greedy decoding across eight
> benchmarks (18,452 held-out questions, judge currency, `head_final_stack_PVFIXED_2026-09-13.json`),
> beating greedy on 6 of 8 and an answer-frequency baseline on all 8, at no additional forward pass
> for verification. It replicates on **Qwen2.5-VL-7B** (+0.0820, 8/8,
> `head_final_stack_qwen_2026-09-13.json`) and on **MedGemma-4b-it** — a Gemma-3 model from a
> different language-model family — at **+0.0481, 8/8** across the same eight benchmarks
> (`head_final_stack_medgemma_ALL8_2026-09-16.json`).

**Do not make these four:**
1. ❌ *"We invented verification from hidden states"* — Q-Probe, LiLaVe, ELHSR, ReProbe, CASE, HSRM
   (§3.3). Concede the family explicitly; it makes the rest credible.
2. ❌ *"Verification is free"* as the contribution — HSRM, CASE and Q-Probe make that argument in
   their abstracts. State the cost; don't sell it as the discovery.
3. ❌ *"The verifier generalises"* — leave-one-benchmark-out is **−0.0056**; the benchmark's own
   training half is worth **+0.0485** (`head_lobo_pooled_2026-08-25.json`). It is a per-benchmark
   method that needs ~100 labelled questions to onboard a new benchmark. Say so *before* the headline.
4. ❌ *The judge-currency number alone* — the August head-only arm was judge +0.0452 and **exact match
   −0.0068, negative on all three benchmarks** (`CHEAP_VERIFIER_ON_7B_2026-08-16.md` §6), and a newly
   trained verifier gets a free +0.006–0.009 under a same-family judge. Until the eight-benchmark
   probe is re-scored in exact match, every headline says "under a Lingshu-32B judge".

## 5.6 Six questions to expect, with the answer and its source

| question | answer | source |
|---|---|---|
| "Has this been done before?" | Yes, in text-only LLMs, under five names; not in a VLM and not on medical VQA as far as a seven-angle search found. Name Q-Probe, ELHSR, ReProbe, HSRM, CASE; name MedProb and DualRead as the two September-2026 near-collisions and say how each differs. | §4.1, `PRIOR_ART_PROBE_VERIFIER_2026-09-14.md` |
| "Why not just use self-consistency? It's free." | Measured: it sits at the random-pick floor and does **not** improve from N=2 to N=16 (0.0933 → 0.0800) while oracle@N climbs 0.156 → 0.418. Majority voting is the wrong aggregator when the correct answer is a minority vote in 74–90 % of recoverable questions. | `TRANSFER_WALL_2026-08-21.md` §4; retrospective §5.3 |
| "Isn't the probe just learning which answer strings are usually right?" | That is exactly the answer-prior control, and it is beaten on all eight benchmarks. On GEMeX (4,315 distinct golds, top-10 gold coverage 12.7 %) the prior loses to greedy by 0.0425 and the probe beats the prior by +0.0573 [+0.0483, +0.0663]. | `TRANSFER_WALL_2026-08-21.md` §10 |
| "Is this specific to Lingshu / to medical fine-tuning?" | No — replicated on Qwen2.5-VL-7B (+0.0820, 8/8) and on MedGemma-4b-it, a Gemma-3 + SigLIP model from a different LM family (+0.0248 on a matched 4-benchmark protocol, with the ensemble layers pre-registered by relative depth). | `TRANSFER_WALL_2026-08-21.md` §12, §14 |
| "How do you know the probe isn't just fitting noise? It has 918k parameters." | Partly answered: a permutation null (frozen probe fed row-permuted features) sits 13.35 σ below the real score, and capacity is not the binding factor (a 32-unit probe scores 0.70090 against a 1024-unit probe's 0.69742). **Not yet answered in the probing literature's own terms** — we have never run a randomised-label control task or reported selectivity. Say so; it is a known, cheap gap (§5.8). | `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §8; `TRANSFER_WALL_2026-08-21.md` §2 |
| "Why does it fail on two benchmarks?" | Diagnosed, not hand-waved. VQA-Med C4 is a **coverage** failure (oracle@8 0.2102 against greedy 0.0913 — most candidate sets contain no correct answer). VQA-RAD is **data** (365 training rows, n=97 held out) plus a sampling penalty of 0.0494 against +0.0244 of genuine selection skill. | `shipped_method_2026-09-12.html`; `decomposition_2026-08-24.json` |

## 5.7 A suggested report structure

1. **Placement** — the paragraph in §5.2. One minute, before any result.
2. **What the method is** — the one sentence from §1.2 plus the pipeline figure from the deck.
3. **Baselines and results** — Table 1, then the headline, then the two failures with their diagnosis.
4. **Limits, stated by us first** — LOBO, currency, coverage. Volunteering these is what separates a
   report that survives questioning from one that does not.
5. **External context** — Table 3, with its protocol caveat.
6. **What is next** — the third generator on all eight benchmarks, exact-match currency, N=32 on the
   coverage-limited benchmarks (§2.6).

## 5.8 Two structural suggestions, both CPU-only

Neither is a new method; both close a hole a reviewer will otherwise open, and everything needed for
each is already on disk.

**1. Re-score the shipped eight-benchmark probe under normalised exact match.** This closes hole 1 in
§2.5 and converts the headline from "under our own family's judge" to a claim in two currencies —
the difference between a result that can be questioned and one that cannot. Highest value per hour of
anything currently available.

**2a. Quote only within-question AUROC.** Any pooled-across-questions AUROC rewards the probe for
recognising the question rather than the answer. Our own corrections file already flagged this
(`OPENTEXT_CORRECTIONS_2026-08-19.md` §7) and CASE (arXiv:2608.17124) names it and builds a
leakage-free measure around it. Zero cost: recompute within question, or stop quoting the number.

**2b. Run a control task, and report selectivity.** The probing literature's standard hygiene check
(Hewitt & Liang 2019, arXiv:1909.03368) is to refit the identical probe on **randomised labels** and
report **selectivity** = real-task metric − control-task metric, because a probe with enough capacity
can fit arbitrary labels. We have a permutation null on the *features* (13.35 σ), which is a different
test — it randomises inputs, not labels. A reader who works on probing classifiers will look for
selectivity and not find it. It is a few CPU-hours of refits on the cached features and it turns
"our probe scores well" into "our probe scores well *and* the representation is what carries it".


---

# 6. What to read, in what order

## 6.1 If you have one evening

Read these four, in this order. They give you the frame, the direct prior art, the limit, and the
medical reality — enough to hold a conversation about the domain.

1. **Snell et al., *Scaling LLM Test-Time Compute Optimally…*** (arXiv:2408.03314) — the paper that
   made "spend compute at inference instead of parameters" a research programme. Read the abstract,
   the compute-optimal section, and §7. The sentence that matters for us is the one restricting the
   win to problems where the small model already has non-trivial success.
2. **HSRM, *Hidden-State Reward Models for Test-Time Verification*** (arXiv:2608.30841) — our
   mechanism and our cost argument, in print, in a text-only LLM. Read it to know exactly what we
   cannot claim.
3. **Hu, *Oracle Gap and Signal Fidelity*** (arXiv:2607.17531) — the decomposition of a selection
   gain into coverage and selection quality, published before ours. Read it and adopt its vocabulary.
4. **Jin et al., *Verification Mirage*** (arXiv:2605.10850) — six VLMs (including the Lingshu family)
   × five medical VQA datasets showing that *untrained* self-verification fails. This is the paper
   that motivates training a verifier at all, and it is the best single citation we have.

## 6.2 If you have a week — the core papers by category

Read each category's ★ papers, in the order the category lists them. PDFs are in `papers/`,
named `<category letter>_<key>.pdf`.

**§3.1 — Test-time compute scaling and best-of-N: the foundations**

- **On Test-Time Scaling for Vision-Language Models** — Fawaz Sammani et al. (2026), arXiv:2606.28864. First broad study of LLM-style test-time scaling on vision-language models: small, good models gain the most (up to ~30 points via self-consistency), perception benchmarks often get worse, and the image stops mattering after roughly 200 generated tokens.
- **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** — Charlie Snell et al. (2024), arXiv:2408.03314. Per-prompt, difficulty-aware allocation of test-time compute (search against a verifier, or sequential revision) beats a fixed best-of-N budget by up to 4x and, FLOPs-matched, lets a small model beat one ~14x larger on easy/medium prompts.
- **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** — Bradley Brown et al. (2024), arXiv:2407.21787. Repeated sampling raises 'coverage' (any-sample-correct) log-linearly over four orders of magnitude, but without an automatic verifier the common selectors (majority vote, reward model) plateau far below coverage — the origin of the 'selection wall'.
- **Training Verifiers to Solve Math Word Problems** — Karl Cobbe et al. (2021), arXiv:2110.14168. Introduces GSM8K and the learned-verifier best-of-N recipe: sample 100 solutions, train a model to predict correctness, return the top-scored one — a 6B model with a verifier slightly beats a fine-tuned 175B model.

**§3.2 — Verifiers and Reward Models: Outcome vs Process, Discriminative vs Generative, and Verification in Medical VQA**

- **Best-of-Evidence: Best-of-N Selection under Partial Verification** — Cenwei Zhang et al. (2026), arXiv:2607.20950. Best-of-N selection for medical VQA when no single reliable whole-answer verifier exists, only partial/claim-level checkable evidence; formalizes this as a candidate-factor graph with a budgeted evidence controller, and measures only modest, often not-significant gains over plain majority-vote/BoN.
- **Verification Mirage: Mapping the Reliability Boundary of Self-Verification in Medical VQA** — Ruinan Jin et al. (2026), arXiv:2605.10850. Shows that self-verification (re-invoking the same or a similar VLM in a fresh context to judge its own answer) is systematically unreliable in medical VQA — the verifier inherits the generator's blind spots ('verification mirage') and under-attends to the image ('lazy verifier'); Lingshu is one of the six tested models.
- **Generative Verifiers: Reward Modeling as Next-Token Prediction** — Lunjun Zhang et al. (2024), arXiv:2408.15240. Proposes GenRM: train the verifier to emit its correctness judgment as generated text (next-token prediction, optionally with chain-of-thought) instead of a single discriminative scalar score, and shows this beats discriminative verifiers and LLM-as-judge on best-of-N.
- **Training Verifiers to Solve Math Word Problems** — Karl Cobbe et al. (2021), arXiv:2110.14168. Introduces the outcome reward model (ORM): sample many candidate solutions, score each with a trained verifier, and keep the top-scoring one — the origin of best-of-N verification.

**§3.3 — Probing frozen hidden states: from probing classifiers to hidden-state verifiers**

- **HSRM: Hidden-State Reward Models for Test-Time Verification** — Xianzhi Li and Xiaodan Zhu (2026), arXiv:2608.30841. HSRM extracts hidden states at reasoning-step boundaries from a frozen generator, mean-pools them through a tiny (~2M-parameter) Transformer encoder to rank candidates, and explicitly verifies it needs zero extra generator forward passes because it reuses representations already computed during generation.
- **MedProb: Probing Internal Representations of Vision-Language Models for Medical Question Answering** — Erfan Nourbakhsh et al. (2026), arXiv:2609.04336. MedProb's MAIN method is a multinomial logistic-regression probe on a frozen medical VLM's LAST-INPUT-TOKEN hidden state that predicts a multiple-choice answer WITHOUT any free-text generation at all; only a secondary Appendix-H extension applies the probe to open-ended generations, and it re-feeds the candidate text back into the model to score it, i.e. a second forward pass.
- **Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling (SWIFT)** — Jizhou Guo et al. (2025), arXiv:2505.12225. SWIFT is a token-level linear gate+reward head on a frozen LLM's own per-token hidden states (concatenated across all layers), trained with BCE, that computes a gated weighted-average reward per candidate and picks the argmax candidate for best-of-N — the closest architectural sibling to our probe we found.
- **Q-Probe: A Lightweight Approach to Reward Maximization for Language Models** — Kenneth Li et al. (2024), arXiv:2402.14688. Learns a 1-layer LINEAR probe on a frozen model's embeddings to reweight (softmax-sample, not hard-argmax) sampled completions toward higher reward, trainable via reward-modeling loss or a novel importance-weighted policy-gradient objective.

**§3.4 — The Walls: Coverage vs. Selection, Imperfect Verifiers, and Bounded Best-of-N Gains**

- **Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time Collaboration** — Jie Hu (2026), arXiv:2607.17531. Independently derives essentially our own decomposition of best-of-N/verifier gain into an oracle gap, a coverage term, a conditional-selection-quality term, and a conditional-harm term, and shows gains are bounded first by oracle gap, then by signal fidelity.
- **When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling** — Yong Yi Bay and Kathleen A. Yearick (2026), arXiv:2606.28661. New (2026) paper directly on-point for why more sampling stops helping: defines a modal ceiling and a correlation ceiling, both reached within a few dozen draws and both independent of sample budget, and proves self-consistency accuracy can fall toward 0 as coverage rises to 1.
- **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** — Bradley Brown et al. (2024), arXiv:2407.21787. Foundational empirical demonstration that coverage (pass@k / oracle accuracy) scales log-linearly over four+ orders of magnitude in sample count, while majority-vote/reward-model selection plateaus far below coverage once no automatic verifier exists.
- **The Limits of Inference Scaling Through Resampling** — Benedikt Stroebl et al. (2024), arXiv:2411.17501. Proves that an imperfect verifier's non-zero false-positive rate imposes a hard upper bound on resampling-based inference scaling that no amount of additional compute can lift, and finds compute-optimal sample counts are typically single digits.

**§3.5 — Training-free selection: self-consistency, majority vote, minimum Bayes risk, consensus, and logit-based scores**

- **Wasserstein Equilibrium Decoding for Reliable Medical Visual Question Answering** — Luca Hagen et al. (2026), arXiv:2605.18313. Extends game-theoretic Bayesian Decoding Game equilibrium search to open-ended medical VQA with a Wasserstein/optimal-transport stopping criterion over a biomedical embedding space, beating greedy decoding on VQA-RAD and PathVQA with small VLMs — the closest published neighbour to our best-of-N probe.
- **Agreement in Representation Space for Open-Ended Self-Consistency** — Paula Ontalvilla et al. (2026), arXiv:2606.12003. Reframes open-ended self-consistency as a geometric property: cluster sampled generations in embedding space and return the one closest to the dominant cluster's centroid (Embedding-Based Agreement, EBA), beating USC/random-selection baselines on code, math, and summarization.
- **Universal Self-Consistency for Large Language Model Generation** — Xinyun Chen et al. (2023), arXiv:2311.17311. Extends self-consistency to free-form generation by asking the LLM itself to pick the most consistent candidate from the concatenated sample set, instead of exact-match voting.
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models** — Xuezhi Wang et al. (2022), arXiv:2203.11171. Introduces self-consistency: sample multiple chain-of-thought reasoning paths and take a majority vote over the final answers instead of greedy decoding, for large gains on closed-form reasoning tasks.

**§3.6 — Uncertainty, Calibration, and Hallucination Detection in LLMs and VLMs -- with the Medical Evidence**

- **Calibrated Triage, Not Autonomy: Confidence Estimation for Medical Vision-Language Models** — Reza Khanmohammadi et al. (2026), arXiv:2606.15910. A head-to-head benchmark of nine confidence estimators (training-free logit, verbalized prompting, trained internal probes) across five LVLMs and three medical VQA datasets finds no estimator reliably best, and even the strongest safely triages only ~25% of radiology cases at 20% error tolerance, and almost nothing in pathology.
- **Overconfidence and Calibration in Medical VQA: Empirical Findings and Hallucination-Aware Mitigation** — Ji Young Byun et al. (2026), arXiv:2604.02543. Across three VLM families (2B-38B) and three medical VQA benchmarks, overconfidence persists regardless of scale or prompting (CoT, verbalized confidence); Platt scaling reliably beats prompt-based calibration but doesn't improve AUROC; adding hallucination-detection signals (their HAC method) improves both, especially on open-ended questions.
- **Detecting hallucinations in large language models using semantic entropy** — Sebastian Farquhar et al. (2024), 10.1038/s41586-024-07421-0. Introduces semantic entropy -- clustering sampled generations by bidirectional textual entailment and computing entropy over the resulting meaning-clusters -- as an unsupervised, training-free hallucination detector that beats naive token entropy and P(True) baselines.
- **A Survey of Confidence Estimation and Calibration in Large Language Models** — Jiahui Geng et al. (2023), arXiv:2311.08298. Survey organizing LLM confidence-estimation methods into white-box (logit-based, internal-state-based, semantic) and black-box (verbalized, consistency-based, surrogate-model) families, cataloging calibration metrics and applications including hallucination detection and selective generation.

**§3.7 — Medical Vision-Language Models: The Generators We Use and the Ones We Compare Against**

- **Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning** — LASA Team et al. (2025), arXiv:2506.07044. Lingshu (7B/32B), a medical MLLM built on Qwen2.5-VL via a 4-stage pipeline (shallow align -> deep align -> instruction tuning -> GRPO RL), plus MedEvalKit, the unified medical eval harness the whole project depends on.
- **MedGemma Technical Report** — Andrew Sellergren et al. (2025), arXiv:2507.05201. MedGemma (Gemma 3 4B/27B + MedSigLIP) technical report: explicitly removed PathVQA and MedVQA from training over data-quality concerns and re-split VQA-RAD to fix train/test image contamination.
- **Qwen2.5-VL Technical Report** — Shuai Bai et al. (2025), arXiv:2502.13923. Technical report for Qwen2.5-VL, the general-domain backbone Lingshu-7B/32B are medically fine-tuned from; defines the ViT -> MLP-merger -> LLM architecture and confirms the 7B config (hidden 3584, 28 LLM layers).
- **How Far Have Medical Vision-Language Models Come? A Comprehensive Benchmarking Study** — Che Liu et al. (2025), arXiv:2507.11200. Independent benchmarking of general-purpose vs medically-specialized VLMs (3B-72B): general models often match or beat medical-specific ones, reasoning consistently underperforms understanding, and no model reaches a clinical-deployment reliability bar.

**§3.8 — Medical VQA benchmarks, datasets, and the evaluation protocol**

- **A Controlled Audit of Pretraining Contamination in Public Medical Vision-Language Benchmarks** — Bruce Changlong Xu et al. (2026), arXiv:2606.10066. Audits SLAKE-En, PathVQA, VQA-RAD and an OmniMedVQA mirror for pretraining contamination using 4 detector families; finds real image-side overlap on SLAKE-En but shows two of the four detector families are unreliable (a non-medical control model, BLIP-2, 'reproduces' their positive signals).
- **OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM** — Yutao Hu et al. (2024), arXiv:2402.09181. A 73-source, 12-modality, >20-anatomical-region medical VQA benchmark built entirely from authentic (non-synthetic) clinical images, showing medical-specialized LVLMs can underperform general-domain ones.
- **PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering** — Xiaoman Zhang et al. (2023), arXiv:2305.10415. Introduces PMC-VQA (227k generative QA pairs from 149k images) and MedVInT, plus a manually-verified 2,000-pair test set; the paper's own data analysis shows the correct MCQ answer is skewed toward option B (~31% vs. 25% expected).
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Lianmin Zheng et al. (2023), arXiv:2306.05685. Establishes and validates LLM-as-a-judge (e.g. GPT-4 scoring model outputs) against human preference, while explicitly naming position, verbosity, and self-enhancement bias as failure modes to correct for.

**§3.9 — The older lineage: n-best rescoring, discriminative reranking, and confidence models in other fields**

- **Residual Energy-Based Models for End-to-End Speech Recognition** — Qiujia Li et al. (2021), arXiv:2103.14152. A 2-layer BLSTM reranker reads an existing autoregressive ASR model's decoder hidden state, attention context, token embeddings and top-K softmax probabilities, mean-pools them over each hypothesis's own output tokens, and is BCE-trained to rerank an n-best list — architecturally the closest ancestor of our probe, five years earlier.
- **Highly accurate protein structure prediction with AlphaFold** — John Jumper et al. (2021), 10.1038/s41586-021-03819-2. AlphaFold's pLDDT confidence head is a small per-residue network computed on the network's own final activations, in the same forward pass, and is used to estimate per-residue accuracy and to rank/select among predicted structures — model quality assessment folded directly into the generator.
- **Confidence Estimation for Attention-based Sequence-to-sequence Models for Speech Recognition** — Qiujia Li et al. (2020), arXiv:2010.11428. A single fully-connected layer (256 units) reads the decoder state, attention context and token embedding of an existing seq2seq ASR model and is BCE-trained on edit-distance-derived correct/incorrect token labels, beating raw softmax probability as a confidence signal on AUC and normalized cross entropy (NCE).
- **Discriminative Reranking for Natural Language Parsing** — Michael Collins and Terry Koo (2005), 10.1162/0891201053630273. A boosting-based discriminative reranker rescores an n-best list of candidate parse trees from a baseline generative parser, combining the baseline's log-likelihood with hundreds of thousands of additional tree features — one of the two founding papers of discriminative reranking in NLP.


## 6.3 Reading order by purpose

**"I need to defend the method's novelty."** §4, then §3.3 in full (every hidden-state-probe paper),
then MedProb and DualRead specifically.

**"I need baselines for the results table."** §3.5 (self-consistency, MBR, universal
self-consistency, consensus scoring) and §3.2 (verifiers and reward models). These are what a
reviewer will say we should have compared against.

**"I need to explain why the gains stop."** §3.4 in full. Coverage, selection efficiency, reward-model
over-optimisation, the modal and correlation ceilings.

**"I need to justify the datasets and the judge."** §3.8, especially the contamination audits and the
LLM-as-a-judge self-preference papers — the latter bears directly on our Lingshu-32B-judging-
Lingshu-7B protocol.

**"I need to place our generator among medical VLMs."** §3.7, starting with the Lingshu report's
benchmark table and the independent benchmarking studies.

**"I want to know where the idea really came from."** §3.9 — speech recognition n-best rescoring,
AlphaFold's pLDDT, docking confidence models. Short cards, big perspective.

## 6.4 How to keep this current

The domain moves monthly and four of the closest papers appeared in the last ninety days. To refresh:

1. Put new arXiv ids, one per line, in a text file.
2. `python3 arxiv_verify.py ids.txt verified_new.json` — verifies them and pulls title, authors,
   date, abstract, categories, DOI and the comment field (where venues appear).
3. Search terms that surfaced the closest work: *hidden-state reward model*, *latent verifier*,
   *intrinsic reward hidden states*, *probe best-of-N*, *generation-pass activations verifier*,
   *medical VQA best-of-N*, *confidence estimation module rescoring*.

The scripts that built this package are in the job's tmp directory and are listed in §7.

---

# 7. The package

```
literature/
├── DOMAIN_GUIDE_2026-09-16.md      this document (source of truth)
├── DOMAIN_GUIDE_2026-09-16.html    standalone, self-contained; open in a browser or mail it
├── DOMAIN_GUIDE_2026-09-16.docx    for inline comments
├── references.bib                  one BibTeX entry per carded paper
├── MANIFEST.md                     filename → citation → section
└── papers/                         ★ core PDFs (symlink to /data/dan/literature/…)
```

**Why the PDFs live on `/data`.** The main disk must stay below 75 % (it is at 73 %), so the PDFs are
stored on the 3.6 TB `/data` mount and `literature/papers` is a symlink to them. The text files are
small and are committed to git; the PDFs are not.

**Every non-core paper** has a working link in its card and an entry in `references.bib`, so nothing
is lost by not having its PDF.

**Provenance.** Every arXiv identifier here was checked against the arXiv API on 2026-09-16 — the
156 cited across the project's own documents (all 156 resolved) plus everything the category sweeps
added. Papers that could not be verified are listed in each category's *considered and not carded*
block, with the reason.
