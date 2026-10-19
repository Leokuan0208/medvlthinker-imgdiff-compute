# PAPER_NOTES_A: Test-time compute scaling and best-of-N, the foundations

Group A. The four core PDFs were read in full from extracted text
(`/home/jamesyang/.claude/jobs/61c9eff9/tmp/txt/A_*.txt`), with key pages rendered and viewed (Cobbe p.7–8,
Sammani p.8 and p.12). All 21 cards in `lit/cards_A.json` were checked against the PDFs (core) or arXiv abstract
pages (6 fetched: 2606.28864, 2512.11109, 2606.11576, 2607.09438, 2608.04001, 2406.16838).
Tags: `[p.N]` = PDF page, `[abstract]` = arXiv abstract, `(inferred)` = my reading and not stated in the source.
"Ours" numbers are quoted from `deck/NUMBERS_T07.md`, which is built from `final_eval_FINAL_cap320_T07.json`
(shipped single T = 0.7), or from `deck/DESIGN_EVIDENCE.md` with the source key it gives.

---

## Part 1: Core PDFs

### 1.1 Cobbe et al., 2021. *Training Verifiers to Solve Math Word Problems*
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Łukasz Kaiser, Matthias Plappert, Jerry Tworek,
Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman (OpenAI). arXiv:2110.14168 (v2, 18 Nov 2021).

**1. Problem.** Large language models make "catastrophic mistakes" in multi-step arithmetic reasoning. An
autoregressive model "ha[s] no mechanism to correct [its] own errors" [p.1]. Extrapolating, a fine-tuned model would
need about 10^16 parameters to reach an 80% solve rate on GSM8K [§4.1, p.6]. The paper proposes not making the generator
bigger, and instead training a second model to judge sampled solutions, then picking the best one. It also releases
the GSM8K benchmark.

**2. Method.**
- *Generator.* A GPT-3 model fine-tuned for 2 epochs on GSM8K. It is kept short on purpose because "test@100
  performance peaks within the first few epochs" and the diversity of solutions collapses after that [p.7, Fig. 3].
  It uses a calculator: annotations of the form `<<…>>` are evaluated with Python `eval` [App. C, p.17].
- *Labels.* "Sample 100 completions from the generator for each training problem", each labelled "correct or
  incorrect based solely on whether they reach the correct final answer" [§4.2, p.8]. These are outcome labels, so
  some false positives get through ("correct final answer using flawed reasoning") [p.7].
- *Verifier.* An LM of the same size, initialised from the generator ("the latter performed slightly better")
  [App. E, p.20]. The scalar head is "a single bias parameter and single gain parameter that operate on the logits
  … [of] a special token in the vocabulary" [App. E, p.20]. The verifier is trained for 1 epoch.
  - Loss: the verifier loss is **MSE** (Table 1, p.16), added unweighted to the language-modelling loss. Question
    tokens are masked out [App. E, Fig. 12]. The paper says "cross-entropy instead of mean squared error … had
    negligible effect" [App. B, p.16].
  - Default is **token-level**: "a scalar prediction after each token", described as "a token-level value function"
    [§4.3, p.9].
  - Joint objective (inferred LaTeX):
    $\mathcal{L}=\sum_{t\in S}\big(v_\phi(Q,S_{\le t})-y\big)^2+\mathcal{L}_{\text{LM}}$, with
    $y\in\{0,1\}$ the final-answer correctness. The exact formula is not printed in the paper.
- *Inference.* "Sample 100 completions to each test problem, rank them with the verifier, and then return the one with
  the highest verifier score" [p.8], i.e. $\hat y=\arg\max_{i\le N} v_\phi(x,y_i)$ with N = 100 and **T = 0.7**
  (Table 1). Test@1 uses T = 0. A variant takes a majority vote among the top-k verifier-ranked samples [§5.1, Fig. 7b].
- Trained: generator and verifier (full fine-tunes). Nothing is frozen.

**3. Setup.** GPT-3 6B and 175B (with 3B and 12B in Fig. 2). GSM8K has 8.5K problems: 7.5K train and 1K test [§2,
p.3]. Metric: test solve rate (exact final answer). Baseline: the fine-tuned generator with one T = 0 sample. There is
no judge (answers are numeric).

**4. Key results.**
- "6B verification slightly outperforms a finetuned 175B model, thereby offering a boost approximately equivalent to a
  30x model size increase" [§6, p.12; also p.2]. Baseline: 175B fine-tuning, full training set. This is read off
  Fig. 5 (p.8). Approximate plot values, which are my reading and not printed numbers: 6B verification ~39% vs 6B
  fine-tuning ~20%; 175B verification ~55% vs 175B fine-tuning ~34%. **Do not quote these as numbers.**
- "it is not beneficial to use verification at low dataset sizes" [p.8]. In Fig. 5, verification is below
  fine-tuning at 500–1000 training problems.
- Performance rises with N "up to 400. Beyond this point, performance start[s] to decrease", which the paper puts down
  to "adversarial solutions that fool the verifier" [§5.1, p.10, Fig. 7a].
- Token-level beats solution-level and "is still improving late in training, whereas the solution-level verifier
  quickly shows signs of overfitting" [§4.3, p.9, Fig. 6a]. With 20% dropout, solution-level "reaches a similar level
  of performance as token-level" [§5.2, p.11, Fig. 8b].
- "using a large generator with a small verifier performs significantly better than using a small generator with a
  large verifier … the verifier may often be relying on relatively coarse heuristics" [p.10, Fig. 6c].
- Direct-answer fine-tuning (no worked solution) drops performance "from 20.6% to 5.2%" [p.7].

**5. Terminology.** *verifier*: "Conditioned on the problem and a candidate solution, the verifier outputs the
probability that the solution is correct" [p.7]. *generator* [p.8]. *test@N*: "the percentage of problems solved
correctly at least once when allowing the model to make N separate guesses" [p.7]. This is the same quantity as
coverage / pass@N / our oracle@N. *token-level vs solution-level verifier* [§4.3]. *coverage*: used informally,
"poor coverage of the solution space" [p.7].

**6. Figures.** Fig. 4 (p.7) is the verification training pipeline: three numbered columns, plain outlined model boxes,
and blue Q / orange S / green Y token chips with S_i^1…S_i^100. It is the canonical "sample, label, train verifier"
diagram and a direct template for our training-data figure. Fig. 5 (p.8): solve rate vs training-set size (log x),
fine-tuning vs verification. It is a good shape for "verifier needs data". Fig. 7a (p.10): solve rate vs number of
completions (log x) with an inverted-U. Fig. 12 (p.20): token row with a grey "masked out" overlay, the right device
for showing "pool only over answer tokens".

**7. What it gives the deck.**
- *Slides:* terminology ("verifier", "best-of-N with a learned verifier", "test@N = coverage"); background (the
  origin of the trained-verifier recipe); method justification (outcome labels from final-answer correctness, the
  verifier initialised from or reading the generator, argmax over N).
- *Presenter sentence:* "Cobbe et al. established the recipe we follow: sample many answers, label each by whether its
  final answer is right, train a model to predict that label, and at test time return the highest-scored candidate.
  Our verifier is a small probe on the generator's frozen hidden states instead of a second full language model."
- *Findings that cut against us (volunteer these):*
  1. Their **token-level** verifier beat solution-level. Ours is solution-level (one score per candidate from
     mean-pooled states). Mean pooling is not token-level supervision. Do not claim it is.
  2. Joint LM training helped their verifier. Our probe has no such auxiliary objective; the backbone is frozen.
  3. Accuracy **fell beyond 400 samples** because of verifier exploitation. We measured only N = 8 and 16
     (`NUMBERS_T07.md`: T07_16 judge +0.0984, but the MedGemma labels for 16-sample pools are incomplete).
     Over-optimisation at large N is untested for us.
  4. Verification was not beneficial at small dataset sizes. That is relevant to onboarding a new benchmark with few
     labels.
  5. Their verifier loss was MSE; ours is BCE. They report the choice made no difference, so this is a neutral point,
     not a contradiction.
- *Supporting:* Fig. 6c shows that even a verifier much smaller than the generator works. That is the closest
  classical support for a tiny probe, though "much smaller" there means 6B vs 175B.

**8. Claims we must not make.** Do not say "Cobbe showed a 30× gain", or put their "30×" next to 7B-vs-32B: it is a
GSM8K, GPT-3, 100-sample result. Do not say they used a probe or frozen features: their verifier is a full fine-tuned
LM. Do not say they used BCE. Do not say N = 8.

---

### 1.2 Brown et al., 2024. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*
Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini
(Stanford / Oxford / Google DeepMind). arXiv:2407.21787 (v3, 30 Dec 2024).

**1. Problem.** Models are usually allowed one attempt per problem. The paper asks how performance scales when you
simply draw many independent samples. It splits the answer into two properties: **coverage** ("what fraction of
problems can we solve using any sample that was generated?") and **precision** ("How often can we identify correct
samples from our collection of generations?") [p.2].

**2. Method.** Repeated sampling at a fixed positive temperature, with k up to 10,000 [§2]. Per-task temperatures:
MATH 0.6, GSM8K 0.6, MiniF2F 0.5, CodeContests 0.6 with top-p 0.95, SWE-bench 1.6 [App. A–B].
- Coverage uses the unbiased pass@k estimator [Eq. 1, p.4]:
  $\text{pass@}k=\frac{1}{\#\text{problems}}\sum_{i}\Big(1-\binom{N-C_i}{k}\big/\binom{N}{k}\Big)$,
  where $C_i$ is the number of correct samples out of N.
- Scaling law [Eq. 2–3, p.8]: $\log c\approx a k^{b}$, so $c\approx\exp(a k^{b})$, fitted with SciPy `curve_fit`
  on 40 log-spaced points [App. C.1].
- FLOPs [p.6]:
  $\text{FLOPsPerToken}(L)\approx 2\,(N_{\text{params}}+2\,n_{\text{layers}}\,d\,L)$.
  Total = prompt tokens summed once + decode tokens × NumCompletions. The prompt is counted once, so the prefill is
  shared.
- Selectors on GSM8K and MATH [§4.1, App. D]: majority vote; reward model + best-of-N (ArmoRM-Llama3-8B-v0.1);
  reward model + majority vote (scores summed per final answer). Nothing is trained by the authors.

**3. Setup.** Models: Llama-3 8B, 8B-Instruct, 70B-Instruct; Gemma 2B and 7B; Pythia 70M–12B; DeepSeek-Coder-V2-Instruct;
GPT-4o and Claude 3.5 Sonnet as single-attempt baselines. Tasks [p.4]: GSM8K (128 random test problems); MATH (128);
MiniF2F-MATH (130, Lean4); CodeContests (140 without image tags, App. A.2); SWE-bench Lite (300; 266 without flaky
tests). Verification: exact match or `is_equiv` for math, unit tests or proof checker otherwise.

**4. Key results.**
- SWE-bench Lite, DeepSeek-Coder-V2-Instruct: "15.9% with one sample to 56% with 250 samples, outperforming the
  single-sample state-of-the-art of 43%" [abstract; Fig. 2, p.5].
- Cost [Table 1, p.7]:

  | system | attempts | solved | total cost | relative cost |
  |---|---:|---:|---:|---:|
  | DeepSeek | 5 | 29.62% | $10.8 | 1× |
  | GPT-4o | 1 | 24.00% | $39 | 3.6× |
  | Claude 3.5 Sonnet | 1 | 26.70% | $51 | 4.7× |

- Selection: MATH, Llama-3-8B-Instruct, "coverage increases from 82.9% with 100 samples to 98.44% with 10,000 samples.
  However, when using majority voting or reward models … the biggest performance increase is only from 40.50% to
  41.41%" [p.3, Fig. 7].
- CodeContests, Gemma-2B: "from 0.02% with one sample to 7.1% with 10,000 samples" [p.2].
- Scaling-law fit example: Llama-3-8B-Instruct on MATH, a = −1.33, b = −0.43, error 0.003 ± 0.0027 [Fig. 5, p.8].
- FLOPs-matched [p.7]: on MiniF2F, GSM8K and MATH, "Llama-3-8B-Instruct always obtains higher coverage than the larger
  … 70B model when the FLOP budget is fixed. However for CodeContests, the 70B model is almost always more cost
  effective."
- Human audit [Table 2, p.10]: "over 90% of the chains-of-thought that we graded are faithful", so there is signal for a
  verifier.

**5. Terminology.** *coverage*: "the fraction of problems that are solved by any generated sample" [abstract]. This is
"equivalent to the commonly-used pass@k metric" [p.4] and is "performance with a perfect verifier" [p.3].
*precision*: their name for the selection problem [p.2]. *repeated sampling*. *automatic verifier* (unit tests,
proof checkers). *oracle verifier* (for GSM8K/MATH coverage) [p.4]. *exponentiated power law*.

**6. Figures.** Fig. 1 (p.2): a two-step pipeline. Step 1 generates candidates as dashed pills (red = wrong,
green = right); Step 2 is the verifier. "Problem 1 (coverage)" and "Problem 2 (precision)" sit under each stage. This
is the best single figure for the coverage/selection split slide. Fig. 7 (p.10): coverage curve vs three selector
curves over k on a log axis. The gap between them is the "selection wall"; recreate it with our oracle@8, pick and
greedy per benchmark. Fig. 5 (p.8): data with the exp-power-law fit.

**7. What it gives the deck.**
- *Slides:* terminology (coverage = pass@k = oracle@N; "precision" is Brown's word for the selection problem);
  background; the oracle / selection-efficiency slide; the cost argument (cheap model × samples vs expensive model × 1).
- *Presenter sentence:* "Brown et al. split best-of-N into two problems: coverage (is a right answer anywhere in the
  pool?) and precision (can we find it?). Without an automatic checker, voting and off-the-shelf reward models capture
  little of the coverage. Our probe attacks the second problem. On our pools coverage is the ceiling, and the probe
  reaches about 81% of it (macro selection efficiency 0.8077, `NUMBERS_T07.md`)."
- *Findings that cut against us:*
  1. In their FLOPs-matched view the small model does **not** always win (CodeContests: 70B better). Their own text
     calls FLOPs "a crude cost metric" [p.7]. We should not present FLOPs parity as automatic.
  2. Their reward-model selector plateaued. Ours is not directly comparable, since their RM was an external text RM
     on math. But our selection efficiency is measured only at N = 8 and 16, so "it doesn't plateau" is untested.
  3. Coverage gains come mainly from problems where correct answers are rare (≤1% of samples, Fig. 8). A selector must
     find "the needle in the haystack". Where our oracle@8 is low (VQA-Med 0.2102, `NUMBERS_T07.md`), sel_eff is
     0.4325.

**8. Claims we must not make.** Do not write "Brown: selection plateaus at ~100 samples" without saying their text is
inconsistent: the abstract says "plateau beyond several hundred samples", §1 says "beyond approximately 100 samples"
[p.3], and the Fig. 7 caption says they "saturate before reaching 100 samples" [p.10]. Do not say their reward model was
trained by them or was a probe. Do not equate their dollar costs with our FLOPs.

---

### 1.3 Snell et al., 2024. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*
Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar (UC Berkeley / Google DeepMind). arXiv:2408.03314 (the PDF read is
v1, 6 Aug 2024). Sammani cites it as ICLR 2025 [Sammani ref. 30].

**1. Problem.** If an LLM gets a fixed but non-trivial inference budget, how much can it improve on a hard prompt, and
when is that better than a bigger model? Prior results were mixed. The paper unifies test-time methods as
**proposer + verifier** and asks how to allocate compute per prompt [p.1–3].

**2. Method.**
- *Two levers* [§2]: modify the **proposal distribution** (a fine-tuned revision model that rewrites its own answer),
  or search against a **verifier** (a PRM).
- *Compute-optimal strategy* [Eq. 1, p.5]:
  $\theta^{*}_{q,y^{*}(q)}(N)=\arg\max_{\theta}\,\mathbb{E}_{y\sim\text{Target}(\theta,N,q)}\big[\mathbb{1}_{y=y^{*}(q)}\big]$.
  It is approximated per difficulty bin.
- *Difficulty*: bin "the model's pass@1 rate – estimated from 2048 samples … into five quantiles" (oracle difficulty)
  or the mean PRM final score over the same 2048 samples (predicted difficulty) [§3.2, p.5–6]. They use **two-fold
  cross-validation** within each bin to pick the strategy [p.6]. The cost of estimating difficulty is not counted
  [p.6, App. C].
- *PRM*: PaLM 2-S* fine-tuned as a binary classifier on **soft Monte-Carlo-rollout values** (16 rollouts per step,
  16 samples per question) with BCE, $-(y\log\hat y+(1-y)\log(1-\hat y))$, using AdamW, lr 3e-5, batch 128,
  dropout 0.05 [App. D, p.22].
- *Step aggregation*: the last-step score [§5.1, App. E]. *Inter-answer aggregation*: **best-of-N weighted** [p.7],
  $\hat a=\arg\max_{a}\sum_{i=1}^{N}\mathbb{1}[\text{ans}(y_i)=a]\,v(y_i)$.
- *Search*: best-of-N weighted, beam search (N beams, width M, up to 40 rounds), lookahead (k-step rollouts at
  T = 0; cost N×(k+1)) [§5.2, p.7–9]. Budgets go up to 256.
- *Revisions*: SFT on trajectories of 0–4 incorrect answers followed by a correct one; parallel/sequential ratio swept
  [§6]. About 38% of correct answers get revised back to incorrect without a selector [p.12].
- *FLOPs exchange* [§7, p.14]: pretraining $X=6ND_{\text{pretrain}}$, inference $Y=2ND_{\text{inference}}$. To match a
  model M× larger, multiply the small model's inference compute by
  $M+3\big(D_{\text{pretrain}}/D_{\text{inference}}\big)(M-1)$, with $R=D_{\text{inference}}/D_{\text{pretrain}}$.

**3. Setup.** PaLM 2-S* (Codey) base [§4]. MATH split of Lightman et al.: 12k train, 500 test [§4, p.6]. Metric: MATH
test accuracy with the Lightman grader [App. G]. Baselines: majority vote, ORM best-of-N weighted, PRM best-of-N
weighted (parallel), and a "∼14× larger pretrained model" with greedy sampling [Fig. 1 caption, Fig. 9]. The larger
model is not named in v1 text.

**4. Key results.**
- PRM search, compute-optimal "can nearly outperform best-of-N using up to 4x less test-time compute (e.g. 16 verses 64
  generations)" [Fig. 4, p.10].
- Revisions, compute-optimal "can outperform best-of-N using up to 4x less test-time compute (e.g. 64 samples verses
  256)" [Fig. 8, p.13].
- FLOPs-matched: "on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute
  can be used to outperform a 14× larger model" [abstract]. Fig. 1 right, revisions, R << 1: easy +21.6%,
  medium +16.7%, hard +5.4%. Other printed bars (by row): +27.8, +3.5, −24.3, +11.8, −11.9, −37.2 [p.2]. **Which bar is
  which (R, difficulty) is a figure-layout reading. Render p.2 before quoting any bar other than the first three.**
- "on the most difficult questions (level 5), no method makes much meaningful progress" [p.9]. On hard questions or
  with R >> 1, "pretraining is a more effective way to improve performance" [Fig. 9 caption, p.15]. R values used:
  0.16, 0.79, 22 [p.15].
- Beam search beats best-of-N at low budgets but falls below it at high budgets. On easy bins it "degrades performance
  as the generation budget increases, suggesting signs of exploitation of the PRM signal" [p.9, Fig. 3].
- **PRM vs ORM**: "the PRM outperforms the ORM, and the … gap … grows with the number of samples" [App. F, p.24–25,
  Fig. 14]. With last-step aggregation the PRM is used "like an ORM", which suggests PRM training is "largely useful as
  a form of representation learning" [App. E, p.24].

**5. Terminology.** *test-time compute*; *proposal distribution*; *verifier*; *compute-optimal scaling strategy*:
"the strategy that chooses hyperparameters corresponding to a given test-time strategy for maximal performance benefits
on a given prompt" [§3.1]. *PRM*: "produces a prediction of the correctness of each intermediate step in an solution,
rather than just the final answer" [p.4]. *ORM*. *best-of-N weighted*: "marginalizes the verifier's correctness scores
across all solutions with the same final answer" [p.7]. *oracle vs predicted difficulty* [p.6]. *parallel vs
sequential* sampling [Fig. 5]. *generation budget* [p.9]. *FLOPs-matched* [§7].

**6. Figures.** Fig. 2 (p.8): three search panels (best-of-N, beam, lookahead), each with a question pill at the top,
fanning paths, and verifier boxes with nodes coloured green (selected) or red (rejected). A single key strip sits
underneath. This is the cleanest best-of-N schematic to adapt. Fig. 5 (p.11): parallel vs sequential, the
terminology slide. Fig. 3 right (p.9) and Fig. 9 (p.15): accuracy by difficulty bin. These are a template for our
per-benchmark "gain where coverage exists" chart. Fig. 4 (p.10): accuracy vs log2 generation budget, several selector
curves.

**7. What it gives the deck.**
- *Slides:* the field-framing slide (test-time compute as an alternative to parameters); terminology (verifier, ORM vs
  PRM, best-of-N weighted, compute-optimal, parallel vs sequential, FLOPs-matched); cost convention ($2N$ FLOPs per
  token); limitations (hard questions).
- *Presenter sentence:* "Snell et al. showed that extra inference compute can substitute for a ~14× larger model, but
  only on questions the small model can sometimes solve. On the hardest questions nothing helps. Our per-benchmark
  results show the same shape. VQA-Med has oracle@8 0.2102 and no gain (judge −0.0038 [−0.0131, +0.0052]); OmniMedVQA
  has oracle@8 0.7007 and the largest gain (+0.1604) (`NUMBERS_T07.md`)."
- *Findings that cut against us:*
  1. **ORM < PRM**, and the gap grows with N. Our probe is ORM-style (one outcome score per candidate). Our answers
     are short and have no steps (inferred), so a PRM may not apply. Still, the result means "outcome-level scorer"
     is not the strongest verifier type in their setting.
  2. **Best-of-N weighted** (sum verifier scores per distinct answer) is their default aggregator, and it is a strong
     baseline. We tested its analogue: "probe-weighted majority vote", J −0.0040 [−0.0134, +0.0057],
     EM **+0.0135 [+0.0036, +0.0237] WIN**, not adopted as "a currency trade"
     (`DESIGN_EVIDENCE.md`, `readout_variants_2026-10-15.json:macro.vote_prob_minus_rank_*`). The deck must say that
     under exact match their aggregator beats our shipped readout.
  3. Our method is fixed-N, not compute-optimal. Snell's central claim is that adaptive per-prompt allocation beats
     uniform best-of-N. Per-query N/T is future work for us.
  4. Their difficulty estimation used 2048 samples and its cost was left out. Do not imply their 4× is a deployable
     saving.
- *Supporting:* their two-fold cross-validation for strategy selection parallels our two-fold image cross-fitting.
  It is a reasonable precedent to mention (inferred analogy).

**8. Claims we must not make.** Do not say "Snell showed best-of-N with a verifier beats a 14× larger model". The win
is for **compute-optimal** revisions/search, FLOPs-matched, on easy and medium questions. Do not say "4× better
accuracy"; the 4× is compute efficiency. Do not say the 14× larger model was PaLM 2-L (not stated in v1). Do not
describe their verifier as an ORM; the main one is a PRM.

---

### 1.4 Sammani, Chamiti, Deligiannis, 2026. *On Test-Time Scaling for Vision-Language Models*
Fawaz Sammani, Tzoulio Chamiti, Nikos Deligiannis (Vrije Universiteit Brussel / imec). arXiv:2606.28864 (v1 27 Jun
2026, v2 1 Jul 2026). The arXiv Comments field says "ECCV 2026"; the page does not say whether it was accepted
[abstract page].

**1. Problem.** Do LLM test-time scaling methods transfer to large vision-language models (LVLMs)? Prior work reported
that CoT and TTS fail for small VLMs. The paper tests nine training-free methods on 13 instruction-tuned VLMs and asks
when TTS helps, why it hurts perception, and how the image is used during long chains [p.1–3].

**2. Method.** No training. A think-prompt is appended: `{question};{think-prompt};{post-prompt}` [§3, p.4]. If answer
extraction fails, a second "answer-only" pass is run [p.4]. Nine methods [§3, p.5–6]: CoT, S-CoT, Plan-and-Solve,
**Self-Consistency** ("samples multiple independent CoTs … then selects the most frequent answer"), Self-Aggregation
(b+1 passes), Self-Refinement (k+1 passes), Describe-Answer, Compositional CoT, Prompt Repetition.
- Hyperparameters [S11, p.31]: greedy (T = 0) for all methods except Self-Consistency (T = 0.8, top-p 0.95,
  **10 samples**: "We use 10 steps") and Self-Aggregation (T = 0.8, top-p 0.95, 4 candidates). Max tokens 1,024 [p.6].
- **No learned verifier or reward model** appears among the nine.
- Attention analysis [§5, p.11–12]: for generated token $g_y$ at layer $l$ and head $h$,
  $a^{img}=\sum a^h_l[1:N_m]$, $a^{pmt}=\sum a[N_m+1:N_m+T]$, $a^{gen}=\sum a[N_m+T+1:N_m+T+(y-1)]$, averaged over
  layers, heads and samples.
- Causal test: drop image KV cache at step ∈ {20, 50, 100, 200, 300} on 300 samples [p.13].

**3. Setup.** Models [§4, p.6]: Qwen2.5-VL {7B, 32B, 72B}; Qwen3-VL {2B, 4B, 8B, 32B}; InternVL-3.5 {2B, 4B, 8B, 38B};
Molmo2 {4B, 8B}; plus GPT-5.2 [S3.3]. Benchmarks: MMStar, RealWorldQA, HallusionBench, WeMath, LogicVista
[Table 1], and A-OKVQA [S3.1]. These are mostly MCQ; RealWorldQA has some open-ended items; HallusionBench is yes/no
[Table S9, p.31]. **None is medical.** Metric: accuracy. Runtime is in seconds on an H200 [p.6]. An LLM judge is used
only for rationale sufficiency (judge–GT 76.4% vs VLM–GT 75.9%, human–judge agreement 83.3%, Table S10, p.31).
Total cost ≈ $6,000 [p.6].

**4. Key results.**
- Qwen3-VL-2B WeMath: baseline 35.46 → Self-Consistency 64.25 [Table 1, p.7]. This is +28.79 points (derived). The
  text says "approximately a 30% increase" [p.8].
- Qwen3-VL-4B with CoT beats the Qwen3-VL-32B baseline on HallusionBench, WeMath and LogicVista [p.8]. LogicVista
  4B-CoT 57.59 vs 32B baseline 44.42 [Table 1].
- **Qwen2.5-VL-7B (Lingshu's base), Self-Consistency vs baseline** [Table 1, p.7]:

  | benchmark | baseline | Self-Consistency |
  |---|---:|---:|
  | MMStar | 62.57 | 65.29 |
  | RealWorldQA | 70.20 | 66.01 |
  | HallusionBench | 72.03 | 71.08 |
  | WeMath | 58.28 | 69.25 |
  | LogicVista | 39.06 | 41.52 |
  | A-OKVQA [Table S3, p.22] | 87.77 | 86.38 |

  Runtime 0.1 s → 17.3 s.
- "test-time scaling methods often degrade performance (rather than maintaining it) on primarily perception-focused
  benchmarks" [p.9]. Self-Refinement on RealWorldQA: Qwen2.5-VL-7B 70.20 → 47.97, Qwen2.5-VL-72B 69.67 → 49.67
  [Table 1].
- Token budget: Δ = score@300 − score@1024 is mostly negative on WeMath and LogicVista (e.g. Qwen3-VL-4B S-CoT
  −14.89 WeMath) and ≈0 or positive on RealWorldQA [Table 2, p.11].
- "After approximately 200 generation steps, removing the image tokens has almost no effect on accuracy" [p.13,
  Fig. 5]. This saves ≈125 GFLOPs (WeMath, HallusionBench, MMStar) and ≈325 GFLOPs (LogicVista) [S7, p.28].
- Runtime: self-consistency 88.9 s on Qwen3-VL-32B vs 0.2 s baseline [Table 1].

**5. Terminology.** *test-time scaling (TTS)*: "a paradigm where large models use additional compute at inference to
achieve better performance, without changing model weights" [abstract]. *zero-shot TTS* (training-free, prompting
or sampling) vs *thinking models* (trained) [p.2]. *self-consistency*, *self-aggregation*, *self-refinement*.
*perception-focused benchmark* ("tasks that do not require reasoning") [p.3]. *token budget*: "the maximum number of
test-time scaling tokens generated" [p.10]. *front-loaded visual grounding* [p.13]. *LVLM*.

**6. Figures.** Fig. 2 (p.8): accuracy vs average inference time per sample (log x), with the Qwen3-VL-32B baseline as
a dashed line and a green "4B beats 32B" region. This is the template for an accuracy-vs-cost Pareto slide. Fig. 4
(p.12): attention to image (solid) vs to generated tokens (dashed) over generated-token index. Fig. 5 (p.12): accuracy
vs image-KV drop step.

**7. What it gives the deck.**
- *Slides:* background ("TTS in VLMs is newer and mixed"); comparison ("the VLM TTS study had no learned verifier");
  limitation and perception framing.
- *Presenter sentence:* "The broadest study of test-time scaling for vision-language models (Sammani et al., 2026)
  tested nine training-free methods. None of them is a trained verifier, none of the benchmarks is medical, and
  extra reasoning often hurts perception tasks. Our method sits in the gap: no long reasoning, just eight short
  sampled answers and a trained selector."
- *Findings that cut against or complicate us:*
  1. **Self-consistency does not uniformly hurt perception.** On RealWorldQA it rises for Qwen3-VL-4B
     (71.76→72.55), 8B (69.54→72.68) and 32B (77.25→77.91). On A-OKVQA it rises for most Qwen3-VL sizes [Table 1,
     Table S3]. The degradation claim is driven mostly by CoT-type and refinement methods. It holds for
     self-consistency only on some models, **including Qwen2.5-VL-7B**, which is the one most relevant to Lingshu.
  2. Their self-consistency is **CoT + vote** with T = 0.8 and 10 samples. Our "self-consistency" baseline is a vote
     over short direct answers at T = 0.7, N = 8. These are different methods under the same name, so say this when
     citing them.
  3. "Small models benefit the most" is shown on reasoning benchmarks with CoT. It is not evidence for best-of-N on
     perception VQA.
  4. The 200-token image-attention finding is about long CoT chains. Our candidates are short answers (inferred from
     the open-ended post-prompt convention; check the mean answer length in our data before saying a number). So it
     does **not** support "the hidden state already carries the image" for our features. The card's
     relation_to_us (c) overreaches; see Part 3.

**8. Claims we must not make.** Do not say "Sammani showed verifiers beat voting": they had no verifier. Do not say
"TTS hurts medical VQA (Sammani)": nothing medical was tested. Do not say "+30% accuracy": it is +28.79 points on one
model and benchmark. Do not cite it as peer-reviewed ECCV without checking. Do not say "self-consistency hurts
perception" as a general finding (see 7.1).

---

## Part 2: All 21 cards in cards_A.json

| id | title (short) | one line | deck relevance | why |
|---|---|---|---|---|
| snell2024scaling | Scaling LLM Test-Time Compute Optimally | per-prompt compute-optimal TTS; FLOPs-matched beats ~14× larger on easy/medium | **slide-worthy** | field framing, ORM/PRM, best-of-N weighted, cost convention |
| sammani2026vlmtts | On Test-Time Scaling for VLMs | 9 training-free TTS methods on 13 VLMs; perception hurt | **slide-worthy** | positioning: VLM TTS without a learned verifier |
| brown2024monkeys | Large Language Monkeys | coverage scales with k; selectors plateau | **slide-worthy** | coverage vs precision; oracle@N |
| cobbe2021verifiers | Training Verifiers (GSM8K) | sample-label-train verifier, argmax over 100 | **slide-worthy** | the recipe we follow |
| liu2025compute | Can 1B Surpass 405B? | compute-optimal TTS depends on policy, PRM and difficulty | mention | PRMs transfer poorly across policies → policy-matched probe |
| wu2024inference | Inference Scaling Laws | accuracy vs FLOPs; 7B + strategy Pareto-beats 34B; voting saturates | mention | template for accuracy-vs-FLOPs; voting-saturation theorem |
| ahmadpour2025limits | Limits and Gains of TTS in VL Reasoning | external verification is the most reliable gain on open VLMs | **slide-worthy** (comparison row) | the only VLM study here where a verifier is best |
| hariri2026regimes | TTS Regimes, Evaluation, Reproducibility | three regimes; C_total = C_gen + C_eval | mention (terminology) | "leaf-level scaling with a terminal reducer" vocabulary |
| welleck2024meta | From Decoding to Meta-Generation | survey: decoding vs meta-generation; best-of-N = reranking | mention (terminology) | conventional vocabulary source |
| chen2024calls | Are More LLM Calls All You Need? | voting non-monotone in K; hard items hurt | mention | explains why our majority vote loses |
| baxevanakis2026small | TTS for Small VLMs on Multilingual Visual MCQ | trained PRM and critic do not beat majority vote on MCQ | mention | the MCQ-side negative; same Qwen2.5-VL-7B family |
| oh2026medical | Model and Task-Aware TTS in Medicine (JMIR) | parallel beats sequential on easier medical tasks; VLMs bottlenecked | mention | only medical TTS evaluation; abstract-only card |
| prucs2025pareto | Compute-Accuracy Pareto Frontiers | prefill-inclusive FLOPs formula; MoE dominates | skip (cite in cost backup) | source for a FLOPs formula only |
| saadfalcon2025ipw | Intelligence per Watt | per-query energy protocol via NVML | skip (backup) | only if energy is shown |
| zhang2026efficient | Efficient Inference for LVLMs (survey) | encoding / prefill / decode bottlenecks; TTFT/TPOT | skip (backup) | systems vocabulary only |
| roberts2026overtraining | TTS Makes Overtraining Compute-Optimal | small model + k samples is compute-optimal under pass@k | skip / one-line mention | assumes a perfect selector |
| zhao2025knowledge | TTS Not Effective for Knowledge-Intensive Tasks | longer reasoning gives no gain on closed-book QA | mention | supports "no long reasoning" for medical VQA |
| nakano2021webgpt | WebGPT | best-of-n with an RM on long-form free text; optimal n grows with size | mention | earliest free-text best-of-n with a learned scorer |
| beirami2024theoretical | Theoretical guarantees on best-of-n | KL ≤ log n − (n−1)/n; win rate ≤ n/(n+1) | skip (backup) | theory only |
| jeddi2026avis | AVIS: Adaptive TTS for VLMs | learned difficulty predictor picks K; shared prefill | mention | shared-prefill cost convention; adaptive-N future work |
| byun2025zeroshot | TTS for Zero-Shot Diagnosis | majority vote over N=16 describe-then-diagnose, MedMNIST | mention | medical-image TTS without a learned selector |

### Verified content for slide-worthy and mention cards

**snell, brown, cobbe, sammani.** See Part 1.

**ahmadpour2025limits** (arXiv:2512.11109)
- Abstract: "external verification provides the most reliable gains" [abstract]. For open-source VLMs, iterative
  refinement often hurts. Gains on perception-focused tasks are only modest [abstract, paraphrased by the fetch tool].
- The card's numbers (Qwen2.5-VL-7B MathVista 68.25 → 75.36 best-of-N; MMMU 36.7 → 47.1) come from its HTML Tables 1–2
  and were **not re-verified** here.
- The verifier is Gemini 2 Flash, a closed external model (card). It is a costly C_eval and not comparable to a probe.
- Use: the comparison table row "verifier-selected best-of-N on Qwen2.5-VL-7B helps; refinement hurts".

**hariri2026regimes** (arXiv:2608.04001)
- The abstract organizes TTS into three regimes: extending one trajectory, sampling and aggregating complete
  candidates, and searching over partial states. It releases "1,403,520 sampled model attempts" [abstract].
- There is no Comments field (no venue).
- Use its vocabulary: ours is "sample-and-aggregate (leaf-level) scaling with a learned reducer". The
  $C_{gen}+C_{eval}$ split is from the card (§3.3), not re-verified.

**welleck2024meta** (arXiv:2406.16838; v1 24 Jun 2024, v2 20 Nov 2024; no journal field on arXiv)
- The survey groups methods into token-level decoding, meta-generation over partial or full sequences, and efficient
  generation [abstract].
- Card quotes, not re-verified in a PDF: best-of-N "refers to generating an N-best list and picking the best sequence
  according to a scoring function"; a learned verifier $v_\psi(x,y)\to[0,1]$ "within Best-of-N".
- Use on the terminology slide: our method is a parallel meta-generator with a learned-verifier reranker.

**liu2025compute** (card only)
- The abstract claim that 1B beats 405B on MATH-500: the card itself says the extracted table row (66.2 vs 71.4)
  contradicts it. **Do not quote any Liu number.**
- Usable qualitative point (card): "PRMs are hard to generalize across policy models and tasks". This is a reason to
  train the verifier on the generator's own samples.

**wu2024inference** (card only)
- "Llemma-7B requires around 2x less total FLOPs than Llemma-34B to achieve comparable accuracy" (card, §4.2).
- Majority and weighted voting converge with infinite samples (Theorems 1–2), i.e. voting saturates.
- Mention-level only. Not re-verified.

**chen2024calls** (card only)
- Vote accuracy is non-monotone in K: an "inverse U-shape" on MMLU Physics (card, Fig. 1). More calls help easy items
  and hurt hard ones (Theorem 2).
- Use to explain why our self-consistency loses to greedy (judge −0.0094 [−0.0157, −0.0034], `NUMBERS_T07.md`), with
  the caveat that this is an analogy (inferred).

**baxevanakis2026small** (arXiv:2607.09438; Comments: "accepted at ImageCLEF 2026")
- The abstract (via fetch) says "neither a training-free generative critic nor a trained multimodal PRM beat majority
  vote".
- PRM-guided beam search trailed self-consistency by 0.39 pp "at over eight times the cost". Going from 8 to 16 chains
  added 0.15 pp. The policy-model switch gave +11.4 pp. Best held-out score 84.1% [abstract].
- Models: Qwen2.5-VL-7B-Instruct and Qwen3.5-4B on EXAMS-V (multilingual visual **MCQ**).
- Use: the MCQ-side negative, consistent with the repo's format finding (CLAUDE.md §0 finding 2). It is not a
  counter-example to open-text selection.

**oh2026medical** (JMIR 28:e90693; abstract-only card)
- Parallel scaling beat sequential on easier tasks. VLMs showed "a structural bottleneck in integrating visual clues"
  with "limited benefit from token expansion" (card, quoting the PubMed abstract).
- Models and benchmarks are not stated in the card. **Do not compare numbers.** The full text was not read by anyone.

**zhao2025knowledge** (card only; venue listed as COLM 2026, unverified)
- "increasing the reasoning length results in minimal or no accuracy gains" on SimpleQA, FACTS and FRAMES (card).
- Use for one line: knowledge-heavy questions do not benefit from longer reasoning. Avoid the hallucination and
  abstention parts (CLAUDE.md rule 6).

**nakano2021webgpt** (card only)
- Best-of-n with a reward model on ELI5 long-form answers. The compute-efficient n was 4, 16 and 64 for 760M, 13B and
  175B (card, §5.2).
- Use: the precedent for free-text best-of-n. "Rejection sampling" is their synonym for best-of-n, and the deck
  should note that.

**jeddi2026avis** (arXiv:2606.11576; Comments: project page only)
- The abstract says AVIS adapts visual-token pruning and the number of rollouts per query, using a learned difficulty
  predictor [abstract].
- Card: "All K rollouts reuse the same visual features and prefilling KV cache" (§3.3), and selection is by majority
  vote.
- Use: (a) the shared-prefill cost convention; (b) future work, "predict N, then select with the probe".
  Card numbers were not re-verified.

**byun2025zeroshot** (card only)
- Majority vote over N = 16 describe-then-diagnose samples. PneumoniaMNIST AUC goes from 0.517 to 0.821 (card,
  Table 1). These are binary MedMNIST tasks.
- Use: a medical-image parallel-TTS precedent without a learned selector. It is not VQA.

**roberts2026overtraining** (card only)
- Its optimisation target is pass@k, which assumes a perfect selector. If mentioned, pair it with our selection
  efficiency (0.8077 macro).

---

## Part 3: Recommendations

**Own slide (one paper each, or a two-paper slide):**
1. **Cobbe 2021 + Brown 2024** as the "anatomy of best-of-N" slide. Use Cobbe Fig. 4 grammar for sample → label →
   train verifier, and Brown Fig. 1 grammar for coverage vs precision. Then our numbers in the same frame:
   greedy, pick, oracle@8 per benchmark, and macro sel_eff 0.8077 (`NUMBERS_T07.md`).
2. **Snell 2024** as the field-framing slide: test-time compute vs parameters; the hard-question limit; ORM vs PRM;
   best-of-N weighted. Add our VQA-Med (no gain at low coverage) vs OmniMedVQA (largest gain) contrast.
3. **Sammani 2026** as the positioning slide: VLM TTS is training-free CoT, voting and refinement, with no learned
   verifier, no medical benchmarks, and perception often hurt. Put Ahmadpour (verifier best on open VLMs) and
   Baxevanakis (MCQ: trained PRM ≤ vote) beside it as a three-row mini-table.

**Comparison table rows:** Cobbe, Brown, Snell, Sammani, Ahmadpour, Baxevanakis, Byun, Oh (each in its own units,
baseline stated). Wu and Nakano are optional.

**Terminology slide:** test-time compute / inference-time scaling; parallel vs sequential scaling; repeated sampling;
best-of-N; verifier / reward model; ORM vs PRM; best-of-N weighted (Snell) vs majority vote / self-consistency;
coverage = pass@k = test@N = oracle@N; precision (Brown's name for selection) and our "selection efficiency" =
selected / oracle@N; compute-optimal; FLOPs-matched (2N FLOPs per token, 6ND for training); meta-generation /
reranking (Welleck); terminal reducer / leaf-level scaling (Hariri); reward over-optimisation (Cobbe §5.1, Snell §5.3).

**Errors found**

*In the cards*
- **sammani2026vlmtts.relation_to_us (c).** It claims the ~200-token image-attention result "supports reading a
  hidden-state summary of the candidate's own generated tokens: by then the representation already carries the
  visual evidence". The result is for long CoT chains (Fig. 4–5 go to ~600 tokens) and is not about short answers.
  Remove it or mark it as not applicable.
- **sammani2026vlmtts.relation_to_us (b).** "Their 'perception benchmarks degrade' result is the same as our repo's
  'reasoning hurts perception' finding." This is reasonable for CoT and refinement. But for self-consistency, Table 1
  and Table S3 show increases on RealWorldQA and A-OKVQA for several Qwen3-VL sizes.
- **sammani2026vlmtts.relation_to_us (a).** It speaks of "our headline that a learned verifier beats self-consistency".
  True in our data (probe vs vote J +0.0895 macro, `DESIGN_EVIDENCE.md` `readout_variants…sc_minus_rank_j` =
  −0.0895), but Sammani's self-consistency is CoT + 10 votes at T = 0.8, not our vote. Name the difference.
- **sammani2026vlmtts.method** omits that Self-Consistency uses 10 samples at T = 0.8, top-p 0.95 [S11]. Add it.
- **sammani venue "ECCV 2026".** This is only the arXiv Comments string; acceptance is not stated. Write
  "arXiv 2026 (Comments: ECCV 2026)".
- **brown2024monkeys.** The card's term list omits the paper's own word for selection, **"precision"** [p.2]. The card
  quotes "plateau beyond several hundred samples" (abstract) while the paper elsewhere says ~100 [p.3, p.10]. Note
  the inconsistency.
- **cobbe2021verifiers.** The card does not mention that the verifier loss is MSE (Table 1) or that accuracy declines
  beyond 400 completions (Fig. 7a). The latter is the over-optimisation evidence the deck needs.
- **snell2024scaling.** The card cites "[html]" sections, but the PDF on disk is v1 (6 Aug 2024). Page numbers above
  refer to v1. The card does not record the PRM > ORM result (App. F), which cuts against an ORM-style probe.
- **Category intro and brown/cobbe cards: "0.83 for the final method".** `NUMBERS_T07.md` gives macro selection
  efficiency **0.8077** for the shipped single-T = 0.7 method. 0.83 appears to belong to the per-benchmark temperature
  variant (inferred; its source was not located in this read).

*In the existing deck briefs*
- **COMPARISON_BRIEF.md uses non-shipped (per-benchmark-temperature or older) numbers in rows the deck will reuse.**
  The spec requires single T = 0.7.
  - Row "Large Language Monkeys" and MedProb row: "OmniMedVQA greedy 0.5164, probe 0.6894, oracle@8 0.7230" and
    "+0.1730 [+0.1647, +0.1813]". `NUMBERS_T07.md` has pick **0.6768**, oracle@8 **0.7007**, Δ **+0.1604
    [+0.1525, +0.1685]**.
  - Row "Snell": "VQA-Med C4 (oracle@8 0.1873)". T07 has **0.2102**.
  - §2.2 self-consistency row: "−0.0070 J macro, 1/8 positive". T07 has **−0.0094 [−0.0157, −0.0034], 2/8**.
  - "What the deck should say about §2": "+0.0897 J / +0.0783 XJ / +0.0165 EM". The shipped T07 values are
    **+0.0798 / +0.0710 / +0.0121**.
  - All of these must be replaced or relabelled as the domain-calibrated variant.
- **COMPARISON_BRIEF.md, Sammani row:** "voting helps reasoning-heavy visual math and hurts perception". This is
  overstated for voting specifically (see 1.4 §7.1). Say "CoT-type and refinement methods often hurt perception;
  self-consistency is mixed, and it does lower RealWorldQA, HallusionBench and A-OKVQA on Qwen2.5-VL-7B".
- **COMPARISON_BRIEF.md, Cobbe row:** "100 samples ranked by the verifier (§4.2, p.8)" is correct. Add T = 0.7 and the
  same-size verifier.
- **FIGURE_STYLE_NOTES.md §1.10–1.11 and the terminology table** match the PDFs. One addition: the table row
  "upper bound | oracle / pass@N / coverage" should also list Cobbe's **test@N** [p.7] and Brown's "precision" as the
  name of the complementary problem.

**Terms the deck must define** (conventional names first): best-of-N; verifier (outcome-supervised, ORM-style);
coverage / pass@k / oracle@N; selection efficiency (ours; define as selected / oracle@N and relate to Brown's
"precision"); majority vote / self-consistency (and how ours differs from Sammani's CoT version); best-of-N weighted;
parallel vs sequential test-time scaling; FLOPs-matched comparison and the 2N-per-token convention; reward
over-optimisation.
