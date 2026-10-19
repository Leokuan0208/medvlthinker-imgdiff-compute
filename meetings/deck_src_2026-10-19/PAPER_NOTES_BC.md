# PAPER_NOTES_BC — verifiers / reward models (B) and hidden-state probes (C)

Reader: deck builder. Every number below is verbatim from the PDF text extractions in
`/home/jamesyang/.claude/jobs/61c9eff9/tmp/txt/` (page = the `=====PAGE n=====` marker = PDF page), or from an
arXiv abstract page, tagged **[abstract]**. Where the fetch tool returned only a paraphrase of the abstract, the
tag is **[abstract, paraphrase]**, and nothing in quotation marks comes from it. **(inferred)** marks my own
reasoning. **(derived)** marks arithmetic I did on quoted numbers. "Not stated" means I searched the text and did
not find it.

Our method, for reference (from PAPER_READ_BRIEF / DECK_SPEC): Lingshu-7B (28 decoder blocks, d = 3584); N = 8 at
T = 0.7; a separate teacher-forced pass; mean pool over the answer tokens at layers 18/20/22; 24 MLP probes
(Linear 3584→256, GELU, Linear 256→1; BCE; 3 layers × 8 seeds); within-question rank average; vote-count
tie-break. Per-probe parameters: 3584·256 + 256 + 256 + 1 = **918,017 (derived)**.

---

# Part 1 — Core PDFs

## C1. HSRM — the closest published analogue to our pipeline

**Citation.** Xianzhi Li, Xiaodan Zhu. *HSRM: Hidden-State Reward Models for Test-Time Verification.*
arXiv:2608.30841v1, 31 Aug 2026. The card says EMNLP 2026; the PDF itself shows no venue. Code: github.com/JXL884/HSRM.

**1. Problem.** In best-of-N for math, the verifier is usually a text model that re-reads every candidate. That
makes verification expensive. HSRM asks whether the generator's own hidden states, already computed while
decoding, are enough to rank candidates. The answer it gives is yes, using a ~2M-parameter reader.

**2. Method.**
- **Input positions.** Hidden states are read only at *reasoning-step boundaries*: the token immediately before a
  delimiter in `["\n\n", ".\n", ":\n", "\n", ".\n\n", ":\n\n"]`, plus the final generated token (§3.1, p.3).
  $H^\ell(y)=[h^\ell_{t_1};\dots;h^\ell_{t_S}]\in\mathbb R^{S\times d_{gen}}$ (Eq. 1), with $S\le100$
  (Table 11, p.15).
- **Layer.** "Unless otherwise stated, we use the final generator layer ℓ = L" (p.3). $h^\ell_t$ is called "the
  residual-stream representation at token position t and layer ℓ" (p.3).
- **Architecture** (§3.2, p.3–4):
  - per-step projection $z^{(0)}_s=W_{in}h^\ell_{t_s}+b_{in}$ (Eq. 2);
  - a 2-layer Transformer encoder, $d_{model}=256$, 4 heads, feed-forward width $4d_{model}$, dropout 0.1 (Eq. 3; p.4);
  - mean pool followed by LayerNorm, $z=\mathrm{LN}(\tfrac1S\sum_s z_s)$ (Eq. 4);
  - a linear readout, $f_\phi(x,y)=w^\top z+b$ (Eq. 5).
- **Parameters** (Table 4, p.12): 2.12M (Qwen3-1.7B, $d_{gen}$ = 2048), 2.25M (4B), 2.65M (8B), ∼3.4M (14B).
- **Loss.** A tie-safe pairwise (Bradley–Terry) loss within each problem:
  $\mathcal L_{rank}=\frac{1}{|P||N|}\sum_{i\in P}\sum_{j\in N}\log(1+e^{-(s_i-s_j)})$ (Eq. 6, p.4). Problems that
  are all-correct or all-wrong "are omitted from the training loss".
- **Trained vs frozen.** The generator is frozen; only HSRM is trained. Training uses "cached hidden-state tensors"
  (§3.3, p.4).
- **Optimiser** (Table 6, p.13): AdamW, lr 1e-4, problem batch size 8, 1000 gradient steps, 20% problem-level
  validation split, final layer cached in float16, 5 seeds (3 for ablations).
- **Inference.** Score all N in one batched pass and return $\hat y=y_{\arg\max_i f_\phi(x,y_i)}$ (Eq. 7).
- **Sampling.** Training pools have N = 64 per problem; evaluation uses N = 8. Temperature 0.7, top-p 0.9, float16,
  non-thinking mode (App. A, p.12).

**3. Setup.**
- **Generators:** Qwen3 1.7B/4B/8B/14B, plus Llama-3.2-1B/3B and Llama-3.1-8B in §6.7.
- **Data** (Table 5, p.13): GSM8K (train 500 / eval 819), MATH-500 (150/350), AIME (20/70), OlympiadBench (150/523).
- **Labels:** a regex/symbolic labeler, then an LLM judge on unresolved cases. The judge changed 7,788 of ≈255K
  labels (App. B, p.12).
- **Metrics:** best-of-8 accuracy and within-problem AUROC, "computed by comparing verifier scores against binary
  correctness labels within each candidate pool and then averaging over problems for which AUROC is defined"
  (p.5). Oracle pass@N is the upper bound.
- **Baselines:**
  - EORM 55M, a text energy verifier with a BT loss. The table lists it at 53.4M parameters (p.13).
  - Qwen2.5-Math-PRM-7B.
  - Single pass.
  - Cheap heuristics: cumulative/mean log-prob, negative entropy, negative varentropy, length, and step count.

**4. Key results.**
- "matches or outperforms a 55M-parameter text-only energy verifier in 15 of 16 generator–dataset settings while
  using only about 2M parameters" [abstract; p.1].
- **Llama, best-of-8, HSRM / EORM / Oracle** (Table 3, p.9):

  | dataset | generator | HSRM | EORM | Oracle |
  |---|---|---:|---:|---:|
  | GSM8K | 8B | 88.3 | 85.2 | 96.0 |
  | MATH-500 | 8B | 56.8 | 52.3 | 87.1 |

  The full table has six rows. On GSM8K, "the improvement ranges from 3.1 to 8.1 percentage points" (p.9).
- **Input-modality ablation** (Table 1, p.7; GSM8K, Qwen3-1.7B):

  | verifier input | Bo8 acc. | within-AUROC |
  |---|---:|---:|
  | text-only | 82.23 | 0.519 |
  | last hidden state, last layer | 86.11 | 0.691 |
  | step states, top-4 layers concatenated (best) | 86.86 | 0.724 |
  | hybrid text + hidden states | 86.23 | 0.707 |

  The hybrid is no better than hidden states alone.
- **Loss ablation** (Table 9, p.14):

  | dataset | BT | ListMLE | BCE |
  |---|---:|---:|---:|
  | GSM8K | 85.93 ± 0.56 (AUROC 0.718) | 84.59 | 83.91 ± 0.45 (AUROC 0.687) |
  | MATH-500 | 66.91 ± 0.89 | 66.69 | 66.57 ± 2.15 |

  On MATH-500 the BCE gap is inside BCE's own SD.
- **Encoder ablation** (Table 10, p.15). DeepSet, which "mean-pools per-step MLP features" (d = 128, 0.40M
  parameters), is comparable to the d = 256 Transformer (2.12M): e.g. GSM8K 1.7B 85.76 vs 85.64. The paper says "no
  architecture dominates", and "the choice of encoder has a relatively small effect compared with the choice of
  input representation" (p.15).
- **Cheap heuristics** (Table 8, p.14). HSRM beats the best cheap scorer, chosen per cell, "in 12 of 16 settings,
  ties in 2, and underperforms in only 2". The margins are small: +2.4 at most, −1.6 at worst (AIME 8B).
- **Layers** (Fig. 3, p.6, GSM8K). "Correctness information is present across a broad range of layers, with the
  strongest performance generally appearing in the upper portion … The final layer is competitive, but it is not
  always the unique optimum" (p.6).
- **N-scaling** (Table 7, p.13). N = 8 → 64 adds only +0.0 to +1.2 points.
- **Thinking mode** (Table 2, p.9). Answer-only extraction beats full-trace extraction on within-AUROC:
  0.669 → 0.736 (1.7B) and 0.672 → 0.884 (4B).
- **Efficiency.** "about 3500× fewer parameters than the 7B PRM and roughly five orders of magnitude fewer
  verification FLOPs per candidate" (p.7). Fig. 4's caption: "matching the 7B PRM at the larger generator scales".

**5. Terminology.**
- *hidden-state reward model* / "lightweight hidden-state verifier for best-of-N reasoning" (p.3).
- *residual-stream representation* (p.3).
- *reasoning-step boundary*.
- *tie-safe ranking loss*.
- *verifier-best accuracy at N = 8* (p.5).
- *within-problem AUROC* (p.5).
- *oracle pass@N* "upper bound imposed by the sampled candidate pool" (p.5).
- *text re-encoding*. Table 11 (p.15) lists "Extra generator passes: 0" for HSRM.

**6. Figures.**
- **Fig. 1 (p.3)** is the pipeline swim-lane: frozen LLM; rows of generated tokens and last-layer hidden states;
  step-boundary states lifted into "[S]"; the text verifier lane re-encodes "[S × n]". This is the best template
  for our Fig. 1 (already studied in FIGURE_STYLE_NOTES §1.1).
- **Fig. 3 (p.6)**: accuracy and AUROC vs layer, one panel per model size. Use it as the model for our layer slide.
- **Fig. 4 (p.7)**: accuracy vs verification FLOPs per candidate on a log x-axis, "cheaper ←", "better ↑". A
  cost-frontier plot.
- **Fig. 5 (p.8)**: accuracy and AUROC vs training-pool size K. Analogous to our "onboarding labels" question.

**7. What it gives the deck.**
- **Slides:** method justification (hidden states beat text; the encoder matters little; upper layers), the
  comparison table, and terminology (within-problem AUROC, oracle pass@N).
- **Presenter sentence:** "HSRM is the same idea in text-only math: a ~2M-parameter reader on a frozen
  generator's hidden states that picks the best of 8. They find the input representation matters far more than
  the reader's architecture."
- **Findings that cut against us. Volunteer them.**
  - (a) HSRM's tie-safe BT loss beat BCE: GSM8K 85.93 vs 83.91 (Table 9). We use BCE. Our own BCE-vs-BT
    comparison is confounded (DESIGN_EVIDENCE §3).
  - (b) HSRM reads states *during generation* and lists 0 extra generator passes. Our headline features come from
    a separate teacher-forced pass.
  - (c) HSRM's default is the final layer, and its best variant concatenates the top 4. We use layers 18/20/22 of 28
    with no concatenation.
  - (d) Gains over cheap heuristics are small (Table 8). HSRM's evaluation is a useful precedent for reporting a
    heuristic envelope.
- **Findings that support us:**
  - DeepSet (mean-pooled per-step MLP) ≈ Transformer (Table 10). A mean-pool-plus-MLP reader is not a weak choice.
  - Answer-only extraction beats full-trace extraction (Table 2). This is consistent with pooling over the answer
    span only.

**8. Claims we must not make.**
- "HSRM proves verification is free." It *asserts* 0 extra passes, but trains and evaluates on cached tensors. It
  does not show the capture code or measure end-to-end latency.
- "HSRM matches the 7B PRM." That holds only on GSM8K "at the larger generator scales". The PRM "retains an
  advantage" on harder sets (p.5).
- Do not quote Qwen3 numbers from Fig. 2. They are plot-only. The tables also disagree with each other for the same
  cell: GSM8K Qwen3-1.7B is 84.8 in Table 7, 85.6 in Table 8 and 86.32 in Table 1 (also flagged in
  COMPARISON_BRIEF).

---

## C2. SWIFT (formerly "ELHSR")

**Citation.** Jizhou Guo, Zhaomin Wu, Hanchen Yang, Philip S. Yu. *Mining Intrinsic Rewards from LLM Hidden
States for Efficient Best-of-N Sampling.* arXiv:2505.12225v3 (8 Jan 2026), KDD '26. The method name in v3 is SWIFT
(Simple Weighted Intrinsic Feedback Technique). The string "ELHSR" does not occur in v3.

**1. Problem.** Text reward models for best-of-N are large (≈7–13B) and data-hungry. SWIFT learns a reward
directly from the task LLM's own hidden states with linear layers.

**2. Method** (§4.2, p.3–4).
- **Feature.** For each response token $t$, concatenate the post-residual hidden states of **all L layers**:
  $h_t=[h^1_t;\dots;h^L_t]\in\mathbb R^{Ld}$.
- **Head.** One linear map gives a gate and a token reward:
  $[\tilde g_t, r_t]^\top = W_{SWIFT}h_t+b_{SWIFT}$, with $W\in\mathbb R^{2\times Ld}$ (Eq. 2). The gate is
  $g_t=\sigma(\tilde g_t)$ (Eq. 3).
- **Pooling** is a gated weighted mean over all T tokens: $R=\sum_t g_t r_t / \max(\sum_t g_t,\epsilon)$ (Eq. 4).
- **Loss:** BCE with logits on the path-level correctness label (Eq. 5).
- **Selection:** $i^*=\arg\max_i R_\theta(h_i)$ (Eq. 1, p.3).
- **Training** (App. D.3, p.14): AdamW, lr 1e-4, weight decay 1e-5, batch size 16. Early stopping on validation
  BCE, patience 3, 80/20 split.
- **Data and generation** (§5.1, p.4): 6,000 training instances × 8 paths; 500 evaluation instances.
  Temperature 1.0, top-p 0.9, max 1024 new tokens.
- **Parameters** (Table 3, p.7): 1.8×10⁵ (Llama-3.2-3B), 2.7×10⁵ (Llama-3.1-8B), 3.0×10⁵ (Ministral-8B).
- **Motivation experiment** (§4.1, Fig. 2, p.3; App. D.2, p.14). They average the hidden states over all tokens
  of a path, per layer, then apply PCA to 50 dimensions and LDA, with 5-fold CV on 3,000 MATH instances
  (Llama-3.1-8B). "the accuracy is approximately 80% on each layer".

**3. Setup.**
- **Datasets:** MATH, GSM8K, AQuA_RAT (Table 1), Imbue Code, HellaSwag, CoinFlip (Table 2), and PKU-SafeRLHF with a
  GPT-4o judge (Table 4).
- **Generators:** Llama-3.2-3B, Llama-3.1-8B, Ministral-8B.
- **Baselines:** EurusRM-7B, Skywork-Llama3.1-8B, Starling-7B, UltraRM-13B, RLHFlow-8B-Deepseek,
  Math-Shepherd-7B, and a fine-tuned Eurus-7B.
- **Not included:** no majority-vote baseline and no greedy baseline. BoN@1 is reported as a row header.
- **Correctness labels:** checked with the Eurus repository's math-equivalence code (App. D.5).

**4. Key results.**
- **MATH, averaged across models and N** (Table 1, p.5): SWIFT 57.5 vs Eurus-7B 51.0. The "12.7% higher" in the
  abstract is **relative**: 57.5/51.0 = 1.127 (derived).
- **Llama-3.1-8B on MATH** (Table 1): BoN@1 47.2 → SWIFT @64 62.6.
- **GSM8K average** (Table 1): SWIFT 89.0 vs fine-tuned Eurus 88.4. Nearly a tie.
- **Loss ablation** (Table 7, p.17; MATH average): cross-entropy (BCE) 57.5 > DPO 57.0 > hinge 56.8 > InfoNCA 55.1 >
  NCA 54.4. "cross-entropy achieves the best average accuracy" (p.9).
- **Gating ablation** (Table 8, p.17; MATH average): with gating 57.5, without gating 54.8. At BoN@64: 49.4 → 53.6,
  57.6 → 62.6, 61.4 → 62.8 (p.9).
- **Layer ablations:**
  - Layers 16/24/28/32 of Llama-3.1-8B average 74.5 vs 74.6 for all layers (Table 5, p.8).
  - MATH BoN@64, first 4 layers: 38.2/44.6/45.4. Last 4 layers: 55.4/60.4/61.8. All layers: 53.6/62.6/62.8
    (Table 10, p.16).
- **Transfer** (§5.4, p.5–6). Pretrained on 10k DeepScaleR samples, BoN@64 Ministral: 62.8/93.6/75.8 on
  MATH/GSM8K/AQuA_RAT.
- **Logit-only SWIFT** (§5.8, p.8). Trained on logits rather than hidden states, it still beats many baselines:
  MATH average 52.8 (Table 6, p.9).

**5. Terminology.**
- *token-level reward*.
- *gating value* / *gating mechanism*.
- *intrinsic reward*.
- "hidden states (after residual connection)" (p.3).
- *rank selection* and *scaled averaging*: combining SWIFT with an external reward model. In rank selection, the
  rank is the max of the two ranks and the min is selected (App. C, Eqs. 6–11, p.13).

**6. Figures.**
- **Fig. 1 (p.2)**: a layer stack drawn as blocks, a tap off each block, and "Concatenate" → a linear head with
  "Token-level Linear Gating" and "Token-level Linear Reward" boxes. This is the canonical *multi-layer tap*
  drawing.
- **Fig. 2 (p.3)**: per-layer CV accuracy of PCA+LDA on mean-pooled states. A direct visual precedent for "mean
  pooled hidden states linearly encode correctness".
- **Fig. 3 (p.7)**: log-scale time and FLOPs bars ("47343×" etc.).

**7. What it gives the deck.**
- **Slides:** related work (nearest single neighbour), the loss justification, the pooling discussion, and the
  layer choice.
- **Presenter sentence:** "SWIFT is the closest single neighbour: a linear head on all-layer hidden states,
  pooled over every response token, trained with BCE, picking the argmax of N. It is text-only math, code and
  commonsense."
- **Supports us:**
  - BCE was the best of five losses (Table 7). This is the counterweight to HSRM's Table 9: the two papers disagree,
    so the loss choice is not settled in the literature.
  - The mean-pooled PCA+LDA motivation shows that a pooled representation is informative.
  - Later layers beat early layers.
- **Cuts against us:**
  - Learned **gated** pooling beat ungated pooling by 2.7 points on average (Table 8). We use a uniform mean, and
    attention or max pooling was "not tested" (DESIGN_EVIDENCE §2).
  - "Without gating" is not spelled out exactly. (inferred) Removing the gate makes the score a uniform mean of
    linear token rewards, which equals a linear probe on mean-pooled features. So Table 8 is close to a direct
    "mean pool vs learned weighting" test.

**8. Claims we must not make.**
- "SWIFT is 12.7 points better." It is relative, not absolute.
- "SWIFT reuses the generation pass." The problem statement says the policy "emits hidden states" while
  generating (p.3), but the paper never describes how the states were extracted or claims "no extra pass". So (D)
  is (inferred), not established.
- "SWIFT beats majority voting." It has no majority-vote baseline.

---

## C3. Q-Probe

**Citation.** Kenneth Li, Samy Jelassi, Hugh Zhang, Sham Kakade, Martin Wattenberg, David Brandfonbrener.
*Q-Probe: A Lightweight Approach to Reward Maximization for Language Models.* ICML 2024 (PMLR 235),
arXiv:2402.14688v2.

**1. Problem.** Adapt a frozen pre-trained LM to maximise a task reward, using something lighter than
fine-tuning but stronger than prompting: a linear probe on its embeddings that reweights k sampled completions.

**2. Method.**
- **Probe definition** (§2, p.2): "A probe is a classifier or regressor that takes internal activations of a
  network as its input and is trained to predict a feature of interest". The model is frozen and they "only
  train a small probe which is the same dimension as the residual stream (Alain and Bengio, 2016)" (p.1).
- **Policy** (§4.1, p.4): sample $a_i\sim p_0(\cdot|x)$ for $i=1..k$, then sample
  $a\sim\mathrm{softmax}(Q_\theta(x,a_1)/\beta,\dots,Q_\theta(x,a_k)/\beta)$.
- **Theorem 4.1** (p.4): as $k\to\infty$,
  $\pi_{\theta,k}(a|x)\to p_0(a|x)\exp(Q_\theta/\beta)/\mathbb E_b[\exp(Q_\theta(x,b)/\beta)]$, which is the
  KL-regularised optimum $\arg\max_\pi \mathbb E[Q_\theta]-\beta\,\mathrm{KL}(\pi\|p_0)$. β → 0 is a hard argmax
  (best-of-k).
- **Losses** (§5, p.4–5):
  - reward MSE: $L_Q=\mathbb E[(Q_\theta-r)^2]$ (Eq. 2);
  - cross-entropy for binary rewards, $L_{CE}$ (Eq. 3). This is the analogue of our BCE;
  - importance-weighted policy gradient: $L_{PG}=\mathbb E[-r(x,a)\,\rho_\theta(a,\{a_i\})]$ with
    $\rho_\theta=f(a)/(f(a)+\sum_{i\ge2} f(a_i))$, $f=\exp(Q/\beta)$ (Eqs. 4–5), and the mean reward as baseline;
  - Bradley–Terry for preferences, $L_{QP}$ (Eqs. 6–7).
- **Features:**
  - Code-LLaMA-7B: "we take the 26th hidden layer of the same model for embeddings" (p.5). Footnote: "Probe
    performance usually peaks at an intermediate layer".
  - Pooling over tokens: **not stated**. The embedding is "of the joint prompt-completion string" (§3.3).
  - The probe is "1-layer (linear)", Adam, lr 5e-5, batch 1000, 150 epochs (p.5). It is "a 4096-dimensional
    linear model" (p.2).
- **Inference:** k = 48, β = 0.1 for code (Table 1). β = 0, i.e. argmax, for preferences (§7.1, p.8).
- **Sampling:** temperature 0.8, top-p 0.95 (code). Temperature 1.0, p = 0.95 (preferences).

**3. Setup.**
- **Tasks:** MBPP (464 training prompts, 500 test), HumanEval (164), GSM8K (7,473 training).
- **Preference data:** HH + OASST + SHP, judged by GPT-4 win rate, 200,336 pairs (20% used).
- **Models:** Code-LLaMA-7B; gpt-3.5-turbo generations embedded by CodeLlama-70B-Python; LLaMA-7B after SFT.
- **Baselines:** pass@1, greedy, 5-shot, SFT on successes, Prompt RM (self-judge), Finetune RM (LoRA), DPO, KTO,
  PPO-offline.

**4. Key results.**
- **MBPP / HumanEval** (Table 1, p.6):

  | method | MBPP | HumanEval |
  |---|---:|---:|
  | pass@1 | 0.29 | — |
  | greedy | 0.38 | — |
  | Prompt RM | 0.31 | — |
  | Finetune RM | 0.34 | — |
  | Q-probe $L_Q$ | 0.38 | — |
  | Q-probe $L_{CE}$ | 0.40 | 0.32 |
  | **Q-probe $L_{PG}$** | **0.46** | 0.34 |
  | 5-shot + $L_{PG}$ | 0.52 | — |
  | Pass@48 skyline | 0.76 | — |

- **GSM8K** (Table 3, p.7): pass@1 0.25, greedy 0.29, $L_{CE}$ 0.43, $L_{PG}$ 0.45, Pass@48 0.80.
- **Preferences** (Table 4, p.8, win rate %): baseline 37.86, DPO 44.97, KTO 51.46, Q-probe 50.10,
  KTO + Q-probe 55.01.
- **MLP probes** (§7.2, p.9, preference data): "We also experiment with more powerful probe architectures, e.g.
  2 or 3-layer MLPs, discovering this actually harms performance by overfitting." App. C, Table 5 (p.14): 3-layer
  probes on OpenAI API embeddings are mixed (MBPP 0.67–0.68 vs 0.65–0.66 linear; HumanEval lower).
- **Mechanism** (Fig. 4, p.7). $L_Q$ and $L_{CE}$ "treat samples a from the same x independently … and end up
  allocating a good amount of capacity to classifying which prompts are hard", whereas $L_{PG}$ "learns a probe
  that is less prompt-dependent".
- **Scaling with k** (Fig. 3). $L_Q$ and $L_{CE}$ "see performance slightly degrading as we increase k" (p.6–7).

**5. Terminology.**
- *probe* (definition above).
- *Q-probe*, *rejection sampling* (used loosely; the analogy is spelled out on p.4).
- *KL-constrained policy*.
- *skyline* (= oracle pass@k).
- *generation–discrimination (GD) gap* (Saunders et al.; p.9).

**6. Figures.**
- **Fig. 1 (p.2)**: k colour-coded candidates above, matching colour-coded feature strips below, a column-vector
  probe, a softmax bar chart, then the sample. This is the best "every candidate gets its own vector" device.
- **Figs. 2–3**: reward vs training-set size and vs k, with 95% CI over 10 runs.
- **Fig. 4**: per-prompt scatter showing loss-dependent prompt-difficulty capture.

**7. What it gives the deck.**
- **Slides:** the definition of "probe", the earliest instance (2024), and the theory link between soft
  reweighting and best-of-N.
- **Presenter sentence:** "Q-Probe showed in 2024 that a linear probe on a frozen model's embeddings can rerank
  sampled completions. As β → 0 their softmax becomes our argmax."
- **Cut against us. Volunteer all three:**
  - (a) A pointwise CE loss, which is our BCE, was clearly weaker than a within-prompt contrastive loss on MBPP
    (0.40 vs 0.46).
  - (b) Their Fig. 4 explanation is exactly the risk for a pointwise BCE probe: it spends capacity on "which
    questions are hard". (inferred) Our within-question rank readout removes any per-question offset at inference
    time, but training is still pointwise.
  - (c) 2–3-layer MLPs overfit on their preference data. We use a 1-hidden-layer MLP. Our own sweep found linear
    worse by ≈ 0.03 SE (DESIGN_EVIDENCE §3), so the tasks differ.

**8. Claims we must not make.**
- "Q-Probe is free." The paper says "evaluating the Q-probe is essentially free in comparison to the base model"
  (p.2), but the preference experiments run a separate forward pass to extract embeddings (§7.1, p.8).
- The main "API" experiment uses **CodeLlama-70B embeddings** of gpt-3.5 outputs (§6.3, p.7), not API
  embeddings. The cards_C card implies API embeddings.
- Do not say Q-Probe mean-pools. Pooling is not stated.

---

## C4. MedProb (incl. App. H and App. L)

**Citation.** Erfan Nourbakhsh, Ke Yang, Anthony Rios. *MedProb: Probing Internal Representations of
Vision-Language Models for Medical Question Answering.* arXiv:2609.04336v1, 3 Sep 2026. The card says EMNLP
Findings 2026; the PDF shows no venue.

**1. Problem.** Med-VQA accuracy is usually measured by generation, which mixes what the VLM knows with how well
prompting elicits it. MedProb asks whether the answer to a closed (multiple-choice) question is already linearly
decodable from a frozen VLM's hidden state. It describes its contribution as "empirical and diagnostic rather than
algorithmic" (p.2).

**2. Method** (§3, p.3–4; Implementation, p.5).
- **Feature:** "the hidden state of the last input token" at every layer ℓ ∈ {0..L},
  $h^{(\ell)}_i=\psi_\ell(f,p_i)$. It is read **before any answer is generated**.
- **Probe:** per-layer multinomial logistic regression,
  $g_\ell(h)=\mathrm{softmax}(W^{(\ell)}h+b^{(\ell)})$, $W\in\mathbb R^{4\times d_\ell}$.
- **Answer space:** K = 2 for yes/no and K = 4 otherwise.
- **Preprocessing:** features standardised with training-set mean and variance.
- **Regularisation:** C chosen by 5-fold CV over {10⁻⁴…10⁴}, max 1,000 iterations.
- **Layer choice:** "The best-performing layer is selected on a held-out validation split strictly disjoint from
  the test set" (p.5).
- **Data budget:** up to 1,000 training examples per dataset. Test sets capped at 500 (App. A, p.14): PathVQA 500
  (stratified, yes/no only), SLAKE 415, VQA-RAD 263.

**3. Setup.**
- **Models:** 28 evaluated, including Qwen2.5-VL 3/7/32B and MedVLThinker 3/7/32B. 14 matched general/medical
  pairs (Table 2).
- **Data:** closed questions of PathVQA, SLAKE and VQA-RAD, converted to MCQ following Jeong et al. 2024.
  OmniMedVQA (8 modalities; 800 train / 400 eval) for the multiclass check.
- **Metrics:** macro-F1 and accuracy.
- **Baselines:** agentic systems (MAM, MMedAgent, UCAgents), medical VLMs, and benchmark-trained VLMs.

**4. Key results.**
- **Overall** (p.5): best is Qwen2.5-VL-32B with SLAKE F1 91.60, VQA-RAD F1 92.33, average F1 89.94.
- **InternVL3-1B** (Table 1, p.6): probe average accuracy 79.32 vs prompting 67.83.
- **Probe minus prompting, paired** (Table 10, p.26): PathVQA +35.0 / +26.4 / +27.2 for LLaVA-V0-7B /
  Llama-3.2-11B / Qwen3-VL-8B. 8 of 9 pairs significant. Qwen3-VL-8B on VQA-RAD is +5.1 [−0.4, 11.0], p = 0.098.
- **Option-likelihood control** (Table 9, p.26): it improves prompting by 4.6 points (59.8 → 64.4). "probing
  outperforms option-likelihood by a further 14.2 points".
- **Layers** (Fig. 5, p.7): "Qwen2.5-VL-7B peaks at 80.4% at layer 28 and MedVLThinker-7B at 79.5% at layer 22".
  Last-10-layer stability (Table 15, p.28): Qwen2.5-VL-7B PathVQA 82.06 ± 0.70. Best layers across resamples
  (Table 22, p.37): e.g. Qwen3-VL-8B 21/25/23.
- **Image controls** (Table 3, p.8): shuffling images costs "6.2 to 18.3 percentage points". Llama SLAKE
  82.4 → 66.5. Qwen3-VL-8B SLAKE 84.1 → 65.8.
- **Medical adaptation** (Table 2, p.8): it "does not consistently improve" linear decodability. 6 of 14 pairs
  favour the general model. Qwen2.5-VL-7B → MedVLThinker-7B is +1.15.
- **Cross-dataset** (Table 11, p.26): probes are "domain-locked". InternVL3-1B trained on PathVQA gives 57.35 on
  SLAKE and 57.41 on VQA-RAD.
- **App. L.1 — linear vs MLP** (p.36–37). "A small MLP probe on the same frozen embeddings". Architecture, width,
  depth and training are **not stated**. Text says "three representative models", but Table 21 shows **two**:

  | model | Log. acc | MLP acc | Log. F1 | MLP F1 |
  |---|---:|---:|---:|---:|
  | Llama-3.2-11B-Vision | 81.3 | 80.9 | 85.9 | 81.1 |
  | Qwen3-VL-8B | 81.4 | 81.6 | 82.3 | **71.3** |

  Their conclusion: "the clinical knowledge relevant to our benchmarks is largely linearly separable within a single
  transformer layer's activation space, validating logistic regression as the appropriate and parsimonious probe".
- **App. H — open-ended reranking** (p.27, Table 14). Restricted to "true multiclass items from OmniMedVQA, with
  500 training and 100 test examples". "The probe is trained to predict correct or incorrect given a single answer
  candidate (the question, image, and one open-ended candidate answer as input)." The VLM "generates multiple
  open-ended outputs"; the highest-scoring candidate is selected.

  | model | probe | no probe (single sample) | oracle |
  |---|---:|---:|---:|
  | Qwen2.5-VL-7B | 0.13 | 0.09 | 0.19 |
  | Llama-3.2-11B-Vision | 0.19 | 0.15 | 0.27 |

  **Not stated:** N, temperature, probe type (logistic or MLP), layer, token position or pooling, grader, CIs,
  seeds. Authors: "we view this as an initial demonstration rather than a complete open-ended Med-VQA solution".
  - **Two appendices are labelled "H".** The second, on p.34, is the modality table (Table 20). Cite App. H as
    "p.27, Rejection-Sampling with the Probe on Open-Ended Generations".

**5. Terminology.**
- *linear probing*, *linear decodability*, *layer-wise representation probing*.
- *elicitation vs representation*: "the bottleneck is not what VLMs know, but how that knowledge is elicited"
  (p.9).
- *option-likelihood scoring*.
- *rejection-sampling probe*.
- *image-ablated / image-shuffled control*.
- The probing caveat (Limitations, p.10): "linear probing measures what is linearly decodable from a chosen
  representation, not everything a model may know … It does not by itself show that the decoded directions are
  causally responsible for the model's predictions". This cites Belinkov (2022), *Computational Linguistics*
  48(1):207–219.

**6. Figures.**
- **Fig. 1 (p.1)**: two-panel "previous vs ours", with ❄ on the VLM and its hidden columns and 🔥 on the linear
  classifier. This is the frozen/trained icon convention (FIGURE_STYLE_NOTES §1.4).
- **Fig. 5 (p.7)**: accuracy vs layer, two models. Directly comparable to a Lingshu layer curve. Qwen2.5-VL-7B is
  Lingshu's base.
- **Fig. 6 (p.7)**: answer-position bias.

**7. What it gives the deck.**
- **Slides:** the "medical, but not this" row of the neighbour table; terminology (linear probe, decodability);
  the layer slide; the honest-gaps slide.
- **Presenter sentence:** "MedProb shows a frozen medical VLM's last input token linearly encodes closed-question
  answers. That is a pre-generation classifier, not a selector. Its one open-ended reranking test (App. H) is 100
  items with an unstated recipe."
- **Cut against us:**
  - (a) Linear ≈ MLP in their setting (App. L). We chose an MLP because our linear probe was ≈ 0.03 SE worse
    (DESIGN_EVIDENCE §3). Say the tasks differ: last-input-token MCQ vs post-hoc candidate correctness. (inferred)
  - (b) Probes are domain-locked across datasets (Table 11). This is consistent with our LOBO result (+0.0223 vs
    pooled +0.0707, per COMPARISON_BRIEF) and is a reason we pool all eight benchmarks.
  - (c) Qwen2.5-VL-7B's linear decodability peaks at the last layer (28) and MedVLThinker-7B's at 22. Our 18/20/22
    were chosen end-to-end. The task differs, so it is not evidence against us, but the deck should not claim
    "middle layers are known to be best".
- **Supports us:**
  - Medical fine-tuning does not reliably add decodable signal. This frames why probing a 7B is plausible.
  - Image-shuffle controls show the signal is image-dependent. (Our probe has no such control; see Part 3.)

**8. Claims we must not make.**
- "MedProb consistently beats prompting by 20+ points on PathVQA." The paper says "improvements exceeding 20
  percentage points in several cases" (p.6). The paired deltas in Table 10 are > 20 for the 3 models shown.
- "MedProb's App. H uses a logistic probe / the same layer." Not stated.
- "MedProb App. H needs a second forward pass." (inferred) Plausible from "question, image, and one … candidate
  answer as input", but it is not stated how the state was obtained.
- Do not compare 0.09 → 0.13 with our numbers. The grader and item set are unknown.

---

## B1. GenRM

**Citation.** Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal.
*Generative Verifiers: Reward Modeling as Next-Token Prediction.* ICLR 2025, arXiv:2408.15240v3.

**1. Problem.** Trained verifiers are usually discriminative classifiers and do not use the LLM's text-generation
ability. GenRM trains the verifier to *say* "Yes/No", optionally after a chain-of-thought, so it can reason and
spend test-time compute on verification.

**2. Method.**
- **Discriminative RM baseline** (Eq. 3, p.4):
  $L=-\mathbb E_{y^+}\log r_\theta-\mathbb E_{y^-}\log(1-r_\theta)$, with $r_\theta=\sigma(z_{cls})$ and
  $z_{cls}=\mathrm{logit}_\theta(cls|y,x)$.
  - It is a **fully fine-tuned Gemma LLM** using "a special token's logit for classification" (App. B, p.22).
  - Training data is balanced 50/50.
- **GenRM direct** (Eq. 4): the score is $p_\theta(\text{Yes}\mid x,y,I)$, with I = "Is the answer correct (Yes/No)?".
- **Unified loss** (Eq. 5): $L_{SFT}(D_{verify})+\lambda L_{SFT}(D_{correct})$. λ = 1/3 for algorithmic tasks and
  1/4 for GSM8K (p.22).
- **GenRM-CoT** (Eq. 6): generate a rationale, then score Yes.
- **Majority vote over rationales** (Eq. 7): average over K = 32 rationales.
- **Selection:** best-of-N argmax.
  - N = 32 for algorithmic tasks and MATH; N = 16 for GSM8K.
  - The GSM8K test set uses 16 solutions per problem (App. A, p.21).
  - Generator temperature: **not stated**.
- **Models:**
  - Verifiers: Gemma-2B, 7B and Gemma2-9B.
  - Generators: Gemma-2B (algorithmic tasks) and Gemini 1.0 Pro (GSM8K).
  - Synthetic rationales: Gemini 1.0 Pro, with reference-guided grading.

**3. Setup.**
- **Tasks:** Last-Letter Concatenation and Word Sorting (length generalisation); GSM8K (7.2K training,
  1.3K test); MATH500 (easy-to-hard); MMLU math subsets.
- **Baselines:** Discriminative RM ("ORM … our main baseline", p.7), LLM-as-a-Judge (Gemini 1.0 Pro, 32 CoT
  votes), DPO verifier, self-consistency.

**4. Key results.**
- **Headline** [abstract]: algorithmic "5% → 45.3%"; GSM8K "73% → 93.4%"; MATH "28% → 44.6%"; MMLU abstract
  algebra "37.9% → 53.5%". All are best-of-N with GenRM-CoT, against a single-sample base rate.
- **MMLU, Base / Disc-RM / GenRM-CoT** (Table 1, p.9):

  | subset | Base | Disc-RM | GenRM-CoT |
  |---|---:|---:|---:|
  | elementary | 80.1 | 90.6 | 91.1 |
  | high school | 52.2 | 74.8 | 76.1 |
  | college | 47.6 | 53 | 56.1 |
  | abstract algebra | 37.9 | 50 | 53.50 |

- **Direct GenRM vs discriminative** (Fig. D.1, p.23): "GenRM (without using CoT) performs slightly better or
  comparable to Discriminative RM". The gain comes mainly from CoT plus voting.
- **Training vs size** (p.13): "using more powerful LLMs (Gemini 1.0 Pro) as a judge is worse than our trained
  GenRM using weaker Gemma models".
- **Sample efficiency** (Fig. 5, p.8): on MATH, GenRM-CoT reaches the Disc-RM's best-of-32 "using 6.4× fewer
  solutions".

**5. Terminology.**
- *verifier* / *reward model*.
- *discriminative RM (ORM)* vs *generative verifier (GenRM)*.
- *GenRM-CoT*.
- *LLM-as-a-Judge*.
- *reference-guided grading*.
- *weighted self-consistency* (App. C).
- *RM accuracy* vs *Best-of-N* (p.7): "RM accuracy only evaluates the verifier's point-wise accuracy, while
  Best-of-N evaluates the verifier's ability to rank solutions".

**6. Figures.**
- **Fig. 3 (p.3)**: Problem/Solution chips → fine-tuned verifier → Yes/No token probabilities; CoT₁…CoT_N →
  "Average" → r.
- **Fig. 5 (p.8)**: best-of-N vs N curves for 5 verifier types. A clean model for our N-scaling plot.

**7. What it gives the deck.**
- **Slides:** terminology (discriminative vs generative; ORM) and the taxonomy slide placing us as a
  discriminative ORM.
- **Presenter sentence:** "In the field's taxonomy our probe is a *discriminative outcome verifier*: one scalar
  per whole answer, trained with BCE. GenRM is the generative alternative. It wins mainly by adding
  chain-of-thought and 32 verification samples per candidate, which costs far more compute."
- **Cuts against us:** a generative CoT verifier beats a discriminative one on every MMLU row. Our discriminative
  choice is a compute trade-off, not a quality claim.
- **Supports us:** a trained small verifier beats a prompted larger judge. This agrees with CLAUDE.md finding 3.

**8. Claims we must not make.**
- "Discriminative RMs are small heads like ours." GenRM's Disc-RM is a full fine-tuned LLM.
- "GenRM shows discriminative ≈ generative." That holds only for *direct* GenRM. GenRM-CoT is better.

---

## B2. Verification Mirage

**Citation.** Ruinan Jin, Beidi Zhao, Myeongkyun Kang, Qiong Zhang, Xiaoxiao Li. *Verification Mirage: Mapping
the Reliability Boundary of Self-Verification in Medical VQA.* arXiv:2605.10850v1, 11 May 2026.

**1. Problem.** Medical VQA pipelines often re-ask the same VLM, in a fresh context, whether its answer is
correct ("self-verification"). The paper measures when this works. It finds the verifier is coupled to the
generator and mostly agrees with it.

**2. Method.**
- **VeriMap framework** (§3, pp.4–6). The generator is $G$, with $\hat a=G(I,q)$. The verifier $V$ receives
  $(I,q,\hat a)$ and outputs Correct/Incorrect/Uncertain, mapped to $\hat y=1[v=\text{Correct}]$.
- **Oracle:** $y^\star=C(\hat a,a^\star)$, with $C$ = Gemini-2.5-Flash as a binary semantic judge (App. A, p.16).
- **Two axes of verifier behaviour:**
  - discrimination, $\mathrm{Acc}=\Pr(\hat y=y^\star)$;
  - agreement bias, $\mathrm{FPR}=\Pr(\hat y=1|y^\star=0)$, $\mathrm{Bias}=(FP-FN)/N$, and dSkew (p.5).
- **Statistics:** logistic GLMM for error coupling and LMMs for cell metrics (§3.4).
- **Saliency:** compares image attention between the generator and verifier passes (§4.4).
- **Feedback loop:** a 4-turn actor–verifier loop (§4.6).
- **Nothing is trained.** All verifiers are prompted (prompts in App. C, p.20).

**3. Setup.**
- **Models (6):** Qwen2.5-VL-7B-Instruct, Gemma-3, Phi-4-Multimodal-Instruct, MedGemma, HuatuoGPT-Vision,
  Lingshu. The Lingshu (and Gemma/MedGemma) **size is not stated**.
- **Datasets (5):** VQA-RAD, PathVQA, SLAKE, PMC-VQA, MedXpertQA. Split sizes: not stated in the main text.
- **Task labels (7)** by majority vote of GPT-5.4, Gemini 2.5-flash and Qwen3.
- **Not included:** no test-time selection and no accuracy-gain baseline.

**4. Key results.**
- **Mirage occupancy** (§4.2, p.6): "verifier error ≳40% and agreement bias well above the midpoint of every
  metric (FPR ≳60%, Bias ≳35%, D-Skew ≳30%)".
- **Differential diagnosis** (p.7): "several differential-diagnosis cells saturating FPR at ∼95-100%".
- **Coupling** (§4.3, p.8): "57× higher odds of verifier failure (p < 0.001)". Table 1 (p.24) gives β̂ = 4.050,
  OR **57.4**. "the most susceptible model shows coupling nearly 400× stronger than the most robust".
- **Lazy verifier** (§4.4, p.9): saliency Δ = +0.027, Cohen's d = 2.33; gradient Δ = +0.020, d = 0.60; both
  p < 0.001.
- **Cross-verification** (§4.5, p.10): "FPR drops by roughly ∼12–20% across tasks, while verifier error drops only
  ∼2–5%".
- **Larger same-family verifier** (App. B.4, p.19): "Lingshu shows flat or mixed trends and no significant
  reduction (p = 0.782)".
- **Loops** (§4.6, p.10–11): "Only 2.2–3.8% of initially wrong answers are corrected over four turns, while
  69.5%–87.1% are locked in by false verification".
- **Lingshu self-verifier, Overall acc / FPR / FNR %** (Tables 11–15, App. E, pp.28–30; LS column, my read of the
  table rows):

  | dataset | acc | FPR | FNR |
  |---|---:|---:|---:|
  | MedXpertQA | 45.0 | 62.8 | 30.6 |
  | PathVQA | 51.0 | 67.8 | 31.7 |
  | PMC-VQA | 48.1 | 66.1 | 19.3 |
  | SLAKE | 77.0 | 62.2 | 9.2 |
  | VQA-RAD | 63.4 | 72.5 | 11.7 |

  FPR range 62.2–72.5, matching COMPARISON_BRIEF.

**5. Terminology** (all defined in the paper).
- *self-verification*: "re-invoking the same vision language model (VLM) with identical parameters but in a
  fresh, independent context to check its own generated answer" (abstract).
- *verification mirage*: "a regime where the verifier is simultaneously high in verifier error and high in
  agreement bias".
- *agreement bias*.
- *lazy verifier*.
- *capacity-coupled*.
- *cross-verification*.

**6. Figures.**
- **Fig. 1 (p.2)**: study pipeline. Busy; do not copy.
- **Fig. 2 (p.7)**: scatter of verifier error vs FPR, with a shaded "mirage quadrant". A good one-panel slide
  visual.
- **Fig. 7 (p.11)**: stacked bars of wrong answers by fate (corrected / still wrong / locked).

**7. What it gives the deck.**
- **Slide:** motivation for a *trained* selector, i.e. why not just ask Lingshu "are you right?".
- **Presenter sentence:** "Asking Lingshu to check its own answer accepts 62–73% of its wrong answers. A larger
  Lingshu verifier did not fix that (p = 0.782). That is why our selector is trained on correctness labels rather
  than prompted."
- **Our own matching control:** zero-shot Lingshu-7B self-verification (P(Yes)) scored SE 0.7058 vs 0.7752 for the
  trained LoRA (DESIGN_EVIDENCE §1, JUL era).
- **Cuts against us (inferred):** our probe reads the *same* model's hidden states, so it may inherit the same
  blind spots: generator–verifier coupling. We have not measured the correlation between probe errors and greedy
  errors. Mirage's GLMM coupling analysis is a ready-made template.

**8. Claims we must not make.**
- "Mirage tested Lingshu-7B." The size is not stated.
- "Mirage shows probes fail / succeed." It tests only prompted self-verification.
- Do not set their Lingshu generator accuracies beside ours. Their judge is Gemini-2.5-Flash and their task
  labels are their own.

---

## B3. Best-of-Evidence (BoE)

**Citation.** Cenwei Zhang, Teng Fang, Yuxia Wang, Derek Li, Bryan Dai, Lei You. *Best-of-Evidence: Best-of-N
Selection under Partial Verification.* arXiv:2607.20950v1, 23 Jul 2026 (IQuest Research).

**1. Problem.** In medical VLM tasks no reliable whole-answer verifier may exist, but individual claims
(modality, anatomy, finding) can be checked. BoE keeps the BoN pool fixed and spends a budget of claim checks that
could change the pick.

**2. Method.**
- **Pool and cheap prior:** pool $Y_{1:K}\sim p^{gen}_\phi(\cdot|x)$ (Eq. 1). The cheap prior is answer-frequency
  scores $s^{(0)}_i$ "whose maximizer agrees with raw majority" (p.3).
- **Signed incidence matrix:** $B\in\{-1,0,+1\}^{K\times M}$ (Eq. 4), equivalent to a candidate–factor graph.
- **Budgeted objective** (Eq. 7):
  $\mathrm{OPT}^{sel}_C=\sup_\pi\mathbb E[\max_i\mathbb E[U_i|F,T^\pi]]$.
- **Implemented update** (Eq. 10): $\mathrm{logit}\,\tilde s_i\leftarrow\mathrm{logit}\,\tilde s_i+\alpha_{ie}(o)\,\mathrm{logit}\,\kappa(e)$.
- **Acquisition rule:** a plug-in EVSI per cost (Eq. 11), with costs 1 (structured), 4 (VLM factor), 8 (whole
  response) (p.8).
- **Calibration is in-sample.** κ is fitted "from the same ledger used for replay". "This protocol supports a
  mechanism analysis, not a held-out calibration estimate; deployment-oriented evaluation would require a separate
  calibration split or cross-fitting" (App. B.4, p.26).
- **Theory:** a residual-information ceiling $\mathbb E[V^\pi_m-V_m]\le\sqrt{\tfrac m2 I(U;T^\pi|F)}$ (Eq. 15), and
  $O(\log K)$ vs $\Theta(K)$ query separation in a factor-code model (§4.2).
- **Generation:** generator Qwen3-VL-30B-A3B; evidence judge Qwen3-VL-235B-A22B; K = 16 at T = 1.1; budget
  C = 16 (§5.1, p.7).

**3. Setup** (Table 1, p.7; App. B.2, pp.24–25).

| cell | questions | notes | Oracle@16 |
|---|---:|---|---:|
| VQA-Med | 2,334 | "the available local copy contains only the training partition" | 82.1% |
| PathVQA open | 9,903 | binary yes/no removed | 61.5% |
| PMC-VQA | 10,000 | options hidden; "not an official open-answer split" | 65.1% |
| MedXpertQA-MM | 2,000 | 5-way MCQ | 64.7% |

- **Grading:** open answers use deterministic matching plus semantic grading by "the same model tier" as the
  judge.
- **Baselines:** raw majority, random factor acquisition, whole-response judging, and a non-deployable myopic
  label-guided allocator.

**4. Key results** (C = 16).
- **BoE − raw majority, pp [95%]** (Table 2, p.9):

  | cell | BoE − raw |
  |---|---|
  | VQA-Med | +0.26 [−0.47, +0.99] |
  | PathVQA | +0.43 [+0.05, +0.81] |
  | PMC-VQA | +0.58 [+0.19, +0.97] |
  | MedXpertQA-MM | +0.40 [−0.15, +0.95] |

- **BoE − random factor:** VQA-Med +1.11 [+0.43, +1.80], McNemar p = 0.0022. The other three are n.s.
- **Full-set accuracy, PathVQA** (Fig. 3, p.8): raw majority 29.28, BoE 29.72.
- **Rescue on majority-wrong questions** (p.9): VQA-Med 4.44% vs 2.45% random.
- **MedXpertQA-MM coverage** (p.9): "only 627 of 1,333 raw failures contain a correct candidate".
- **Their caveat:** "All analyses are exploratory and uncorrected" (Table 2 caption).

**5. Terminology.**
- *partial verification*.
- *signed candidate–factor graph*.
- *evidence action*, *EVSI* (expected value of sample information).
- *Oracle@K*: "fraction of questions whose candidate pool contains at least one correct answer".
- *MajorityWrong*.
- *residual evidence capacity* $\Lambda_C$.

**6. Figures.**
- **Fig. 1 (p.1)**: busy pipeline; an example of what not to imitate.
- **Fig. 3 (p.8)**: paired bar charts (rescue rate; full-set accuracy).

**7. What it gives the deck.**
- **Slide:** medical test-time-compute comparison.
- **Presenter sentence:** "The most recent medical best-of-N paper, with a 235B evidence judge and 16 samples,
  moves accuracy by under one point over majority vote. Two of four intervals include zero."
- **Supports us:** coverage limits selection (627/1,333). This is the same coverage/selection split we use.
- **Methodological contrast (careful):** their channel weights are fitted in-sample and they say cross-fitting
  would be needed. We cross-fit by image.

**8. Claims we must not make.**
- "We beat BoE by N×." Different generator, judge, N, baseline and items.
- "BoE uses PathVQA's official test set." It is 9,903 non-yes/no items; the split is not stated.
- "BoE's VQA-Med cell is the 2019 test set." It is drawn from the training partition.

---

# Part 2 — Every card in categories B and C

Relevance key: **S** = slide-worthy, **M** = mention, **K** = skip.

### Category B (24 cards)

| id | title (short) | one line | rel. | why |
|---|---|---|---|---|
| cobbe2021verifiers | Training Verifiers (GSM8K) | origin of trained outcome verifier + best-of-N | S | founding paper; term "verifier" (in group A PDFs) |
| zhang2024genrm | GenRM | generative Yes/No verifier vs discriminative | S | taxonomy axis; read in full above |
| zhang2026bestofevidence | Best-of-Evidence | claim-level evidence BoN in medical VQA | S | medical TTC comparison; read above |
| jin2026verificationmirage | Verification Mirage | self-verification fails in med-VQA, Lingshu incl. | S | motivation for trained selector; read above |
| uesato2022outcome | process vs outcome feedback | outcome supervision matches final-answer error | M | ORM justification |
| lightman2023stepbystep | Let's Verify Step by Step | PRM > ORM on MATH | M | the counter-citation for ORM |
| gao2022overoptimization | RM overoptimization scaling | proxy RM overoptimization | M | term "reward hacking"; N-scaling caution |
| wang2023mathshepherd | Math-Shepherd | automatic step labels for PRM | K | process side only |
| lambert2024rewardbench | RewardBench | RM benchmark | K | genre only |
| xiong2024llavacritic | LLaVA-Critic | open LMM-as-judge | K | generative judge; not our axis |
| li2024vlrewardbench | VL-RewardBench | VL judges weak, fail at perception | M | supports "big judges are weak" |
| wang2025unifiedreward | UnifiedReward | pointwise+pairwise multimodal RM | K | vocabulary only |
| ding2025medrewardbench | Med-RewardBench | medical MLLM judges vs experts | M | medical RM benchmark exists |
| ong2025vlprm | VL-PRM lessons | VL-PRM used as ORM beats step selection | M | ORM support from VLM side |
| lee2025multidomain | Rethinking RMs multi-domain | dORM ≈ dPRM across 14 domains | S | places our dORM corner (taxonomy slide) |
| di2025bestofmajority | Best-of-Majority | frequency filter + RM rerank | M | future work; our vote tie-break is a weak form |
| thomas2025radiologyprm | Radiology sentence PRM | text PRM beats white-box baselines | M | the explicit anti-hidden-state argument (portability) |
| wang2026evpv | EVPV | VL-PRM errors are perception errors | K | process; general domain |
| saadfalcon2025weaver | Weaver | ensemble weak verifiers via weak supervision | M | contrast to our homogeneous 24-probe ensemble |
| lee2026fuse | FUSE | zero-label verifier ensembling | K | different pipeline stage |
| khalaf2025rewardhacking | Inference-time reward hacking | BoN itself can hack proxy | M | N-scaling caution |
| zhang2026medtiv | Med-TIV | agentic RL tool verifier for medical reasoning | M | opposite design point, medical |
| cao2025dreamprm | DreamPRM | domain-reweighted multimodal PRM | K | we tested per-dataset reweighting (worse; DESIGN_EVIDENCE §3) |
| kim2026msv | Multi-Sequence Verifier | score conditioned on whole candidate set | M | we tested set-aware features (worse; DESIGN_EVIDENCE §3) |

**Content for S/M rows** (cards are abstract-level unless marked; one paper re-fetched):

- **uesato2022 [card, abstract].**
  - "16.8% → 12.7% final-answer error, and 14.0% → 3.4% reasoning error among final-answer-correct solutions".
  - Pure outcome supervision gives "similar final-answer error rates with less label supervision".
  - Deck line: outcome supervision is enough when only the final answer is graded. Ours are short answers.
- **lightman2023 [card, abstract].**
  - "solves 78% of problems from a representative subset of the MATH test set".
  - "Process supervision significantly outperforms outcome supervision".
  - Present it as the strongest counter-evidence to an ORM, in the long-chain math setting.
- **gao2022 + khalaf2025 [card, abstract].**
  - Gao: a proxy-RM gold score degrades under optimisation pressure, with coefficients that "scale smoothly with
    the number of reward model parameters".
  - Khalaf: "true reward first increases before declining" is "an inevitable property" of BoN-type mechanisms.
  - Use for the N-scaling slide (why not N = 64).
- **li2024vlrewardbench [card, abstract].**
  - "Even GPT-4o achieves only 65.4% accuracy".
  - Judges "predominantly fail at basic visual perception tasks".
  - Training a judge gives "+14.7% accuracy for a 7B VL-GenRM".
- **ding2025medrewardbench [card, abstract].** 1,026 expert-annotated cases, 13 organ systems, 32 MLLMs; fine-tuned
  baselines "demonstrate substantial performance improvements". There are no numbers in the abstract.
- **ong2025vlprm [card, abstract].** "VL-PRMs when used as Outcome Reward Models (ORMs) during test-time scaling
  (TTS) can outperform VL-PRM guided process step selection".
- **lee2025multidomain [abstract, paraphrase, fetched 2026-10-19].**
  - Four RM variants across 14 domains.
  - Discriminative ORM ≈ discriminative PRM; generative PRM is not competitive; generative ORM is the most robust.
  - Card quote: "(i) dORM performs on par with dPRM, (ii) gPRM is not competitive, and (iii) overall, gORM is the
    most robust" [card].
  - Deck use: our corner (dORM) is not dominated by dPRM. Note that gORM is reported more robust, which cuts
    slightly against a discriminative choice.
- **di2025bestofmajority [card, abstract].** BoM restricts to high-frequency candidates before reward reranking;
  "its performance does not degrade when increasing N".
- **thomas2025radiologyprm [card, abstract].**
  - A 0.5B sentence-level PRM; "7.5% in Matthews Correlation Coefficient and 1.8% in AUROC over strong white-box
    baselines".
  - It generalises to an unseen LVLM. It is a direct statement of the portability cost of white-box probes.
- **saadfalcon2025weaver [card, abstract].**
  - "o3-mini-level accuracy … (87.7% average)" with Llama 3.3 70B.
  - Weighted ensemble of heterogeneous verifiers.
- **khalaf2025** — see Gao.
- **zhang2026medtiv [card, abstract].**
  - "Improving MedQA accuracy by 23.5% and MedXpertQA by 32.0% relative to the base generator".
  - "8× reduction in sampling budget".
  - It is text medical reasoning, not VQA.
- **kim2026msv [card, abstract].** "improves best-of-64 accuracy by up to 6% relative". Set-conditioned verifier.
  Our set-aware feature variants scored lower in the sweep (DESIGN_EVIDENCE §3, SWEEP CV).

### Category C (21 cards)

| id | title (short) | one line | rel. | why |
|---|---|---|---|---|
| alain2016understanding | Linear classifier probes | coins "probe"; linear separability rises with depth | S | terminology slide (probe, linear probe) |
| hewitt2019designing | Control tasks | control tasks + selectivity | S | terminology + honest gap (we have not run one) |
| azaria2023internal | Internal state knows lying | classifier on activations predicts truthfulness | M | lineage |
| li2024qprobe | Q-Probe | linear probe reweights k samples | S | read above |
| piotrowski2025lilave | LiLaVe | lightweight latent verifier + BoN/SC | M | neighbour; head type unverified |
| guo2025elhsr | SWIFT | gated linear head over all-layer token states | S | read above |
| ni2025reprobe | ReProbe | transformer probe for step credibility | M | process-level neighbour |
| wang2026case | CASE / decodability | linear gate on answer-token state; leakage-free decodability | S | med+general; leakage control is a gap for us |
| li2026hsrm | HSRM | step-boundary hidden-state reward model | S | read above |
| nourbakhsh2026medprob | MedProb | linear probe, frozen med VLM, MCQ | S | read above |
| xie2026dualread | DualRead | pre/post-answer state calibration in med VLM | M | medical, does not select |
| bucciarelli2026vhs | VHS | hidden-state verifier for diffusion BoN | M | cross-modality evidence |
| duan2025truthprint | TruthPrInt | truthful direction steering in LVLMs | K | steering, not selection |
| jiang2024interpreting | Interpreting VLM reps | logit-lens + editing | K | not a classifier probe |
| zhang2026vibprobe | VIB-Probe | VIB probe on attention heads | K | detection/intervention |
| kogilathota2026halp | HALP | pre-generation hallucination probes | M | pre- vs post-generation distinction |
| zhao2026decisionstate | Decision-state probing | forced-choice stress | K | diagnostic only |
| zhou2025llavareward | LLaVA-Reward | MLLM hidden states as T2I reward | K | judge re-reads → extra pass |
| orgad2024llmsknow | LLMs Know More | token-localized truthfulness; detectors don't transfer | M | "knows more than it says"; transfer caveat |
| morenocencerrado2025noanswer | No Answer Needed | question-only linear probe predicts correctness | M | pre-generation family |
| zhang2025icrprobe | ICR Probe | residual-stream update dynamics | K | design-space note only |

**Content for S/M rows:**

- **Alain & Bengio [abstract, paraphrase, fetched; card].**
  - Linear classifiers ("probes") are trained independently of the model on each layer's features. They used
    Inception v3 and ResNet-50.
  - "linear separability of features increase[s] monotonically along the depth of the model" [abstract quote via
    card and fetch].
  - The original probe is linear and diagnostic. Ours is non-linear and operational.
- **Hewitt & Liang [abstract, paraphrase, fetched; card].**
  - EMNLP 2019. A *control task* "associates word types with random outputs", learnable only by the probe.
  - A good probe is *selective*: high on the linguistic task, low on the control task.
  - Findings: "popular probes on ELMo representations are not selective"; dropout is "ineffective for improving
    selectivity of MLPs" while other regularisers work; layer-2 probes are "substantially more selective" though
    layer-1 is slightly more accurate on POS.
- **Belinkov (not carded; needed for the terminology slide) [abstract, paraphrase, fetched].**
  - *Probing Classifiers: Promises, Shortcomings, and Advances*, arXiv:2102.12452, Computational Linguistics
    (squib). Published as 48(1):207–219, 2022 per MedProb's reference list.
  - It "critically reviews the probing classifiers framework, highlighting their promises, shortcomings, and
    advances" [abstract, short quote].
  - The deck's caveat sentence can come from MedProb p.10, which cites it: probing shows decodability, not
    causation.
- **Azaria & Mitchell [card, abstract].** The classifier on hidden activations gets "an average of 71% to 83%
  accuracy labeling which sentences are true versus false, depending on the LLM base model". Lineage only.
- **LiLaVe [abstract, paraphrase, fetched].**
  - A lightweight verifier reading the base LLM's hidden states, combined with best-of-n and self-consistency,
    plus conditional self-correction and conditional majority voting.
  - The fetched abstract **does not mention the head type**. The card's "MLP tried, XGBoost selected" is
    unverified; do not quote it.
- **ReProbe [card, abstract].** "match or exceed the performance of PRMs that are up to 810x larger" with "fewer
  than 10M parameters". Step-level, not whole-answer.
- **CASE [abstract, paraphrase, fetched; card quotes].**
  - Correctness-Axis SElection: a linear gate on the answer-token hidden state picks the best candidate.
  - *Decodability* is "leakage-free" (question-grouped).
  - Card quotes [abstract]: decodability "predicts the accuracy gain of selection over voting with a Pearson
    correlation r=0.75 and a decision threshold near AUC=0.60"; "up to 19 points on medium-difficulty questions
    and 16.8 points on hard questions".
  - It covers general and **medical** LLMs (text). Deck use: the field's answer to question-identity leakage. Our
    image-level cross-fitting is the analogous control for train/test leakage, but not a within-question
    decodability metric.
- **DualRead [card, abstract].**
  - Combines pre-answer solvability with post-answer assessment in GRPO-trained medical VLMs, plus the CCG-AUC
    counterfactual-image metric.
  - "improves correctness discrimination and calibration over verbalized confidence".
  - It calibrates one answer; there is no selection.
- **VHS [card, abstract].** For diffusion BoN: "reducing joint generation-and-verification time by 63.3%, compute
  FLOPs by 51% and VRAM usage by 14.5%", "+2.7% improvement on GenEval".
- **HALP [card, abstract].**
  - Pre-generation probes, "up to 0.93 AUROC on Gemma-3-12B, Phi-4-VL 5.6B, and Molmo 7B".
  - "~0.79 AUROC for Qwen2.5-VL-7B using visual-only features".
  - Deck use: the pre- vs post-generation distinction.
- **Orgad et al. [card, abstract].**
  - Truthfulness signal is "concentrated in specific tokens".
  - "error detectors fail to generalize across datasets".
  - Models may "encode the correct answer, yet consistently generate an incorrect one".
- **No Answer Needed [card, abstract].** A question-only linear probe "outperform[s] black-box baselines and
  verbalised predicted confidence"; "predictive power saturates in intermediate layers".

---

# Part 3 — Recommendations

## Slides

- **Own slide:**
  - **HSRM**: same pipeline shape; Fig. 1 template; volunteer BT > BCE.
  - **SWIFT**: nearest single neighbour; BCE best of 5 losses; gated > uniform pooling.
  - **Verification Mirage**: why a trained selector, with Lingshu's own FPR.
  - Possibly a combined "probing methodology" slide: Alain/Hewitt/Belinkov plus MedProb's caveat.
- **Comparison tables:**
  - *Mechanism neighbours:* HSRM, SWIFT, Q-Probe, CASE, LiLaVe, ReProbe, MedProb (main + App. H), DualRead. Columns:
    task/modality, feature position, layer(s), pooling, head, loss, params, N, baseline, gain in own units, extra
    pass?
  - *Medical TTC:* BoE, Mirage, MedProb App. H.
  - *Taxonomy 2×2* (outcome/process × discriminative/generative): Cobbe, GenRM, Lightman, Lee 2025. We sit in dORM.
- **Mechanism-neighbour table rows, all verified above:**

  | paper | feature | layer | pool | head | loss | params | N |
  |---|---|---|---|---|---|---|---|
  | HSRM | step-boundary states | final (top-4 best) | mean after 2-layer Transformer | linear readout | tie-safe BT | 2.1–3.4M | train 64 / eval 8 |
  | SWIFT | all response tokens | all L concatenated | gated mean | linear 2×Ld | BCE | 1.8–3.0×10⁵ | 4/16/64 |
  | Q-Probe | prompt+completion embedding | layer 26 (Code-LLaMA-7B) | not stated | linear 4096 | $L_{PG}$ best ($L_{CE}$, $L_Q$, BT also) | 4096 + bias | k = 48 |
  | MedProb main | last input token, pre-generation | best-on-val per model | none | multinomial LR | CE | 4×d | — |
  | MedProb App. H | not stated | not stated | not stated | not stated | not stated | — | not stated |
  | **ours** | answer tokens, teacher-forced | 18/20/22 of 28 | mean | MLP 3584→256→1 ×24 | BCE | 918,017 each (derived) | 8 |

## Terms the deck must define (with the source to cite)

- **probe / probing classifier**: Q-Probe §2: "A probe is a classifier or regressor that takes internal
  activations of a network as its input and is trained to predict a feature of interest". Origin: Alain & Bengio
  2016.
- **linear probe**: a single affine map plus softmax/sigmoid on frozen activations (Alain & Bengio; MedProb
  $\mathrm{softmax}(W^{(\ell)}h+b)$).
- **MLP probe**: one or more hidden layers. MedProb App. L calls it a "nonlinear MLP probe". Say "one hidden layer
  of width 256", never just "2-layer MLP", because the term is ambiguous.
- **frozen backbone**.
- **hidden state / residual stream**: "the residual-stream representation at token position t and layer ℓ" (HSRM
  p.3).
- **control task**: same inputs, random outputs, learnable only by memorisation (Hewitt & Liang).
- **selectivity**: real-task accuracy minus control-task accuracy (Hewitt & Liang).
- **linear decodability**: MedProb.
- **verifier / reward model; ORM vs PRM; discriminative vs generative (GenRM)**.
- **best-of-N; oracle pass@N / Oracle@K / coverage**.
- **within-problem AUROC**: HSRM p.5.
- **BCE (pointwise) vs Bradley–Terry (pairwise, tie-safe)**.
- **self-verification, agreement bias, verification mirage, lazy verifier**: Mirage.
- **question-identity leakage / decodability**: CASE [abstract].
- **reward overoptimization**: Gao; Khalaf.
- **teacher forcing / re-encoding**: HSRM's Fig. 1 caption contrasts itself with "re-tokenize and re-embed".

## Honest gaps these papers expose (for the gaps slide)

1. **No control task / selectivity check on our probe.** Hewitt & Liang. DESIGN_EVIDENCE lists it as "never run";
   only a feature-permutation null was run.
2. **No within-question AUROC reported.** HSRM reports it as standard.
3. **No image-shuffle control.** MedProb Table 3 does one cheaply. (inferred) It is easy to run on our cached
   features.
4. **Uniform mean pool; learned pooling not tested.** SWIFT's gate gave +2.7 points average.
5. **Pointwise BCE.** HSRM (BT > BCE) and Q-Probe ($L_{PG}$ > $L_{CE}$) favour within-question objectives. SWIFT
   favours BCE. The literature is split; our own comparison is confounded.
6. **Generator–verifier coupling not measured.** Mirage's GLMM is a template.
7. **"No extra pass" not shown for the shipped probe.** HSRM and SWIFT read decoding-time states, but neither shows
   capture code or latency either.

## Errors found in cards and briefs

1. **cards_C SWIFT card / category intro: "SWIFT/ELHSR and HSRM match all four [A–D]".**
   - SWIFT never states how states were extracted, so (D) is inferred.
   - HSRM asserts 0 extra passes (Table 11) but trains and evaluates on cached tensors without showing capture.
   - Also, "ELHSR" does not appear in v3 (agrees with COMPARISON_BRIEF §5.6).
2. **cards_C HSRM card.**
   - "five orders of magnitude fewer verification FLOPs … while matching its GSM8K accuracy" overstates. Fig. 4's
     caption limits the match to "the larger generator scales".
   - The card omits that BCE was tested and lost (Table 9).
3. **cards_C Q-Probe card.**
   - It implies the API experiments use API embeddings. The main Table 2 uses CodeLlama-70B embeddings; API
     embeddings are App. C and gave no gain.
   - It omits the greedy baseline (0.38 on MBPP), which is the like-for-like reference for our "vs greedy" framing.
4. **cards_C MedProb card.**
   - "consistently outperforms zero-shot prompting by 20+ percentage points on PATH-VQA" is not the paper's wording
     ("exceeding 20 percentage points in several cases", p.6).
   - Its DATASETS field omits OmniMedVQA.
   - "feeds … back into the model … second forward pass" is inference, not stated.
   - The App. L text says three models while Table 21 shows two (an internal inconsistency in the PDF).
5. **cards_B BoE card.** It omits that VQA-Med is drawn from the *training partition*, and that κ is fitted
   in-sample (App. B.4).
6. **cards_B GenRM card.** It does not say the Disc-RM is a fully fine-tuned LLM (App. B), so "discriminative" ≠
   "small head". "Gains over the ungated base rate" should read "single-sample base rate".
7. **cards_C LiLaVe card.** The XGBoost/MLP remark is not in the abstract (re-fetched). Keep it marked unverified.
8. **COMPARISON_BRIEF §3 "Two findings that cut against our choices"** is incomplete. Add:
   - SWIFT gated > uniform pooling (Table 8);
   - Q-Probe $L_{CE}$ < $L_{PG}$ and the Fig. 4 prompt-difficulty explanation;
   - MedProb linear ≈ MLP (App. L);
   - HSRM's default final layer and best top-4 concatenation;
   - and, on the other side, SWIFT BCE best of 5 losses (Table 7).

   The Q-Probe MLP-overfitting sentence is about *preference* data (§7.2). App. C Table 5 shows 3-layer probes are
   mixed on code with API embeddings.
9. **Q-Probe's own bibliography** dates Belinkov's survey "2016" (p.9). It is 2021 arXiv / 2022 CL. Cite Belinkov
   2022.
10. **FIGURE_STYLE_NOTES §2.4.** "SWIFT h_t^i": in the PDF the per-layer superscript is the layer, $h^\ell_t$, and
    the concatenation is $h_t=[h^1_t;\dots;h^L_t]$. Not an error, but the deck should write the layer index in
    parentheses as recommended.
