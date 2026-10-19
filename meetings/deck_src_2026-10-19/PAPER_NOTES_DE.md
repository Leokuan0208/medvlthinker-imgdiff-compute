# PAPER_NOTES_DE — Group D (the walls: coverage vs selection, imperfect verifiers) and Group E (training-free selection)

Read in full: the 7 core PDFs in `literature/papers/` (text in `tmp/txt/`; WED Fig. 1 and Stroebl Fig. 1/4/5 were rendered and viewed).
Also read: WED authors' code (`/data/dan/repos/Wasserstein-BDG-medical-VQA-main`, `BDG.py`, `config.py`) and our adapter `src/exp_2026_10/wed/wed_select.py`, because the paper leaves out the final selection rule.
arXiv abstract pages fetched (6): 2310.01387, 2111.09388, 2410.01064, 2302.09664, 2502.18581, 2310.09139.

Tags: [p.N] = PDF page. [abstract] = arXiv abstract. **(inferred)** = my reading, not stated in the source. **(code)** = from the authors' repository, not the paper.

---

# Part 1 — Core PDFs

## D1. Bay & Yearick 2026 — the modal ceiling

**Citation.** Yong Yi Bay, Kathleen A. Yearick. "When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling." arXiv:2606.28661v1 [cs.LG], 27 Jun 2026. Preprint, not peer-reviewed. Both authors are listed as "PhD, University of Illinois at Urbana-Champaign" [p.1].

**1. Problem.**
- Drawing more samples keeps raising **coverage**, the fraction of problems where at least one sample is correct.
- A deployed system has to return one answer. Choosing it without a perfect checker is **selection**.
- Selection saturates, and it can get worse as more samples are drawn.
- The paper names the distance between the two curves the **identifiability gap**: "the answer a model can produce but not pick" [abstract].
- It also derives a separate ceiling on how precisely repeated samples estimate a benchmark mean.

**2. Method.** This is a derivation, not an algorithm. The paper says "This work is a citable derivation, not a new algorithm" [p.3].

Notation: n attempts per prompt q; $Y_i=\mathbb 1\{o_i \text{ correct}\}$; $K=\sum_i Y_i$; per-attempt success probability $s$ [p.3–4].

- **Headline quantities** (Eq. 3, p.4):
  $$\text{pass@}n=P[K\ge1]\ \text{(coverage)},\qquad \hat p=K/n,\qquad \mathbb 1\{K>n/2\}\ \text{(majority vote)}$$
- **Exchangeable / de Finetti model** (Eq. 4–5, p.4):
  $$Y_i\mid\theta\overset{iid}{\sim}\mathrm{Bern}(\theta),\quad \theta\sim G,\qquad \rho=\mathrm{Corr}(Y_i,Y_j)=\frac{\mathrm{Var}(\theta)}{s(1-s)}$$
- **Design effect and effective number of samples** (Eq. 1/7, pp.3, 5):
  $$n_{\text{eff}}=\frac{n}{1+(n-1)\rho}\;\to\;\frac1\rho\ \text{as } n\to\infty \quad\text{(correlation ceiling)}$$
- **Coverage** (Eq. 10, p.8):
  $$\text{pass@}n=1-\mathbb E_\theta[(1-\theta)^n]\;\le\;1-(1-s)^n \quad\text{(Prop. 3, "correlation tax")}$$
- **Selection ceiling** (Prop. 5, Eq. 13, p.10). Let $p_q(a)$ be the answer distribution for problem q, $a^\star_q=\arg\max_a p_q(a)$ its mode, and $c_q$ the correct answer. A plurality vote converges almost surely to $a^\star_q$, so self-consistency accuracy tends to
  $$\pi_{\text{mode}}=P_q[a^\star_q=c_q]\quad\text{(the modal ceiling, independent of } n).$$
- **Anti-scaling** (Cor. 3, p.11). If the mode is wrong but $p_q(c_q)>0$, self-consistency accuracy goes to 0 while coverage goes to 1.
- **How fast the vote settles** is set by the **effective number of answers** $1/\sum_a p_q(a)^2$ (p.11).
- **Majority-vote bound** (Cor. 4, pp.11–12):
  $$P[\theta>\tfrac12]\approx\Phi\!\big((s-\tfrac12)/\sqrt{\rho s(1-s)}\big)\ \le\ \pi_{\text{mode}}$$
- **Scorers.** "Best-of-n with a learned reward model is bounded instead by how well that scorer ranks answers: a perfect verifier lifts it to coverage, a frequency-like scorer collapses it to the modal ceiling, and real reward models fall between" (p.11).

Nothing is trained. There is no N or temperature of the authors' own. The data are re-analysed public logs.

**3. Setup.**
- Brown et al. logs: GSM8K and MATH, Llama-3-8B/70B-Instruct, $10^4$ independent samples per problem (Table 1, p.13).
- Beeching et al. log: MATH-500 (500 problems), Llama-3.2-1B-Instruct, 256 samples per problem at T = 0.8, top-p 1.0, five sessions, graded with math-verify (p.14).

**4. Key results.**
- **GSM8K, Llama-3-8B-Instruct** (Table 1, p.13; Fig. 9, p.14): coverage@$10^4$ 1.00, self-consistency 0.87, $\hat\rho_b$ = 0.47 [0.40, 0.54], 1/$\hat\rho_b$ = 2.1. The paper reads this as "For roughly an eighth of problems the correct answer is somewhere in the pool but the vote does not return it."
- **GSM8K, Llama-3-70B** (Table 1, p.13): coverage 1.00, self-consistency 0.97, $\hat\rho_b$ = 0.41.
- **MATH-500, Llama-3.2-1B** (Fig. 10, p.15; text p.14):
  - coverage → 0.88; self-consistency plateaus at $\pi_{\text{mode}}$ = 0.45 "by about n = 64";
  - the 256 answers carry "a median of only about thirteen distinct values";
  - majority bound $P[\theta>\frac12]$ = 0.20;
  - single-sample accuracy ≈ 0.27.
- **Within-problem correlation** (p.15): $\hat\rho_w\approx 0.0007$ [0.0005, 0.0009], so the pooled ρ is essentially between-problem difficulty spread: $\hat\rho$ = 0.401 vs $\hat\rho_b+(1-\hat\rho_b)\hat\rho_w$ = 0.402.

**5. Terminology** (the paper's words):
- **coverage** — "the fraction of problems with at least one correct try" [abstract]
- **selection** — choosing one answer "not knowing which try is right" [abstract]
- **identifiability gap** — "the answer a model can produce but not pick"
- **modal ceiling / modal-hit rate π_mode** — Eq. 13
- **anti-scaling of selection** — Cor. 3
- **effective number of answers** — $1/\sum_a p_q(a)^2$
- **design effect, effective number of samples n_eff, correlation ceiling 1/ρ**
- **intraclass correlation ρ**, split into **ρ_b** (between-problem) and **ρ_w** (within-problem)
- The checklist, item 5 (p.18): "Separate the answer ceiling from the verifier ceiling: one is whether the correct answer is the mode, the other whether the right try is identifiable by the scorer."

**6. Figures.**
- **Fig. 1a (p.2):** coverage vs self-consistency against log n, with the shaded identifiability gap and a dashed modal ceiling. Clean two-line plot. This is the best template for our "coverage vs selection" slide; redraw it with our oracle@N, greedy, probe and majority-vote curves.
- **Fig. 2 (p.2):** one problem's answer histogram. The tallest bar is wrong; the correct answer is a shorter bar. This explains in one picture why majority voting fails.
- **Fig. 7 (p.11):** two panels, "most likely answer correct" vs "wrong", showing anti-scaling.
- **Fig. 10 (p.15):** the real-log version, coverage → 0.88 vs self-consistency → 0.45.

**7. What it gives the deck.**
- **Slides:** terminology (coverage, selection, identifiability gap, modal ceiling); background for why majority vote fails; method justification for a trained scorer over voting.
- **Presenter sentence:** "Voting can never beat the rate at which the most common sampled answer is correct. A learned scorer is bounded instead by how well it ranks, which is the ceiling our probe attacks."
- **Cuts against us:**
  - (a) Our vote count is used as a tie-break. Within ties, that part of the readout inherits the modal-ceiling logic (inferred).
  - (b) Bay's recommendation for raising the selection plateau is decorrelation (temperature, nucleus sampling, prompt diversity, mixing models; p.16, p.18), not a larger n. Only the vote part of our method falls under this, and the plateau-raising lever is untested (Bay's own protocol is pre-registered, p.17).
  - (c) Bay offers no ceiling formula for a learned scorer; it says only that the scorer's "ranking" binds.

**8. Do not claim:**
- that Bay's modal ceiling bounds our probe (it bounds voting only);
- that Bay measured anything on VQA or medical data;
- that ρ explains selection. Bay says ρ governs *estimation*, and selection is a separate axis that "ρ does not see" (p.16).

The card's phrase "correlation ceiling reached within a few dozen draws" mixes the two ceilings. Per p.20, the modal ceiling is reached "within a few dozen draws" and estimation saturates "sooner still".

## D2. Hu 2026 — OracleGap

**Citation.** Jie Hu (Research Institute of China Telecom). "Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time Collaboration." arXiv:2607.17531v1 [cs.CL], 20 Jul 2026. Code: github.com/AmGarfield/OracleGap.

**1. Problem.** Self-consistency, best-of-N, critics and verifier pipelines have uneven and sometimes negative gains. When should a training-free selector be expected to help? The paper's answer: fix the candidate pool, then decompose the gain into measurable factors [abstract, pp.1–2].

**2. Method** (pp.4–5). Pool $C(x)=\{y_1,\dots,y_k\}$ with k = 5. The reference is the first sample $y_1$. Then:
$$\text{recoverable}(x)=\text{ref wrong}(x)\wedge\text{any@}k\text{ correct}(x)\quad\text{(Eq. 3)}$$
$$\text{coverage}=P(\text{signal defined}),\quad q=P(\text{selected passes}\mid\text{recoverable},\text{defined}),\quad h=P(\text{selected fails}\mid\text{ref passes},\text{defined})\quad\text{(Eq. 4–6)}$$
$$\text{gain}=P(R\wedge D)\,q-P(C\wedge D)\,h\quad\text{(Eq. 1, p.2)}$$

- **Oracle gap** = "the score difference between the any-of-k oracle and the chosen reference. It is an upper bound on every fixed-pool selector" (p.5).
- **Signal fidelity** = candidate-level agreement of verifier verdicts with official labels, reported as accuracy, FPR, FNR and MCC (p.5).
- **Effective capture / effective harm** count fallback behaviour over all recoverable and all reference-correct rows. The paper warns they "must not be confused with the conditional quality and harm terms" (p.5).
- Selector ladder (Table 1, p.5): sample0; L1 same-family LLM selector; L3 cross-family LLM selector; L4-gen generated tests; L4-public public tests; Oracle.
- Nothing is trained. Main model: Qwen3.6-35B-A3B-BF16 (p.6).

**3. Setup** (Table 2, p.6):
- LiveCodeBench: 1,055 tasks, three seeds, 2,888 task-seed rows.
- MATH Level-5 hard subjects: 250 per seed.
- GPQA-Diamond: 198 per seed.
- HumanEval+: 164.
- Grading: official evaluators; option-letter matching for GPQA.
- CIs: task-cluster hierarchical bootstrap.

**4. Key results.**
- **LiveCodeBench** (Table 3, p.6), first sample 72.33%:
  - L4-public 80.47% (+8.14 pp [6.99, 9.36], fixed/harmed 235/0)
  - L4-gen 75.03% (+2.70 [2.02, 3.43], 80/2)
  - L1 75.83% (+3.50 [2.26, 4.73], 199/98)
  - L3 74.31% (+1.97)
  - Oracle any@5 84.07% (+11.74)
- **Fidelity** (p.6): public tests 92.98% accuracy, MCC 0.825; generated tests 53.05%, MCC 0.248.
- **Effective capture / harm** (Table 4, p.7): L1 199/339 = 58.70% capture, 98/2089 = 4.69% harm; L4-public 69.32% capture, 0% harm.
- **MATH** (Table 6, p.8):
  - L4-symbolic (answer-equivalence clustering) beats SC by +4.67 [2.93, 6.40];
  - L1 vs SC −3.20 [−5.20, −1.20];
  - L3 vs SC −1.87.
- **GPQA-Diamond** (Table 7, p.9; p.8):
  - oracle gap 18/594 = 3.03 pp;
  - 87.54% of pools have five identical letters;
  - SC −1.68 pp, L1 −1.68 pp, L3 −2.36 pp;
  - Qwen3.5-9B pool: recoverable 0.67%, 94.44% identical pools.

**5. Terminology:** oracle gap; recoverable mass; any-of-k oracle; signal coverage; conditional selection quality; harm; effective capture; effective harm; signal fidelity (MCC); fixed/harmed counts; fixed-pool selection vs repair.

⚠ **"Coverage" here means "the signal is defined"**, not Brown's or Bay's "some sample is correct". The deck must not use the word for both.

**6. Figures.**
- **Fig. 1 (p.2):** (a) pool, then verification signal, then select, then fixed/harmed/unchanged accounting, with the gain formula. (b) A four-step pre-deployment workflow. A good model for our "gain decomposition" slide.
- **Fig. 2 (p.7):** ladder bars with a hatched, dashed oracle bar marked as non-deployable. Copy that convention for oracle@8 on our charts.

**7. What it gives the deck.**
- **Slides:** terminology (oracle gap, recoverable, quality, harm); the method's accounting; limitations.
- **Presenter sentence:** "No fixed-pool selector can beat the oracle gap. What it realises is how many recoverable questions it fixes minus how many already-correct ones it breaks, so we report both fixes and harms."
- **Cuts against us:**
  - (a) Hu reports fixed/harmed counts for every mechanism. The deck should show our probe's fixed vs harmed relative to greedy, not only net Δ, or say "not measured".
  - (b) Hu's reference ($y_1$) is a pool member. Our reference (greedy) is a separate decode, so some greedy-correct questions may have no correct sample in the pool. That harm is then unavoidable for any selector (inferred).
  - (c) Hu's "effective capture" is fixed / recoverable, a headroom fraction. Our "selection efficiency" is selected / oracle@N. These are different normalisations and must not be compared directly.

**8. Do not claim:**
- that Hu uses greedy as reference (it uses the first sample);
- that "gain is bounded multiplicatively by fidelity" (the card says this; the paper does not);
- that LLM selectors are negative everywhere (they are positive on LiveCodeBench, Table 3);
- that the 4.69% is the conditional h (it is effective harm, Table 4).

## D3. Stroebl, Kapoor, Narayanan 2024 — limits of resampling

**Citation.** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan (Princeton). "The Limits of Inference Scaling Through Resampling." arXiv:2411.17501v3 [cs.LG], 26 Mar 2026 (v3).

**1. Problem.** Resampling until a candidate passes a verifier (unit tests) is hoped to let weak models match strong ones. If the verifier has false positives, resampling cannot reduce their probability, so accuracy has a ceiling regardless of compute [abstract].

**2. Method.**
- **Setting.** Generate, then a verifier accepts or rejects. The first accepted sample is returned.
- **Ranking vs verification.** "ranking selects the best among multiple candidates, while verification accepts or rejects each candidate independently" (p.4). Their verifier "has imperfect coverage, but no false negatives" (p.9).
- **Core argument** (pp.5–6). If $P_{\text{strong}}(\text{Correct}) > P_{\text{weak}}(\text{Correct}\mid\text{Pass Verifier})$, the weak model cannot match the strong one at any budget. The infinite-budget accuracy is the conditional accuracy given a pass, i.e. the verifier's precision on that model's outputs (inferred restatement).
- **Reward model for Fig. 4** (p.6): a true positive is worth +1, a false positive costs the C/B ratio ∈ {0, 1, 2, 4, 8}, and no pass within K is worth 0. 200 samples per task; the mean over 1,000 random orderings.
- **Theoretical model** (App. C, pp.25–27). Easy and hard task types $T_1, T_2$ with priors $p_i$ and success rates $r_i$; completeness c; soundness s.
  $$\beta_i=(1-c)r_i+s(1-r_i),\qquad p^{(k)}_{T_i}=\frac{\beta_i^{k-1}p_i}{\beta_1^{k-1}p_1+\beta_2^{k-1}p_2}$$
  $$EV_k=\big[V_{TP}P^{(k)}_{TP}+V_{FP}P^{(k)}_{FP}\big]\big[\beta_1^{k-1}p^{(k)}_{T_1}+\beta_2^{k-1}p^{(k)}_{T_2}\big],\qquad K_{opt}=\arg\max_K\sum_{k\le K}EV_k$$
- Nothing is trained. Sampling temperature 0.8 (p.21).

**3. Setup.**
- Benchmarks: HumanEval+ and MBPP+ (350 tasks after exclusions, pp.22–23). The original tests serve as the verifier; the extended tests serve as ground truth.
- Models: Llama-3.1 and Code Llama families, the Command family, GPT-4o, Phi-3 and others (pp.5, 21).
- Samples: ≥200 per task on HumanEval+ (1,000 for Command Light); 50 on MBPP+.

**4. Key results.**
- **Fig. 1 left legend (p.2):** fit "0.22x + 0.72, R² = 0.891" between single-sample accuracy and conditional accuracy given passing. Weaker models produce false positives at higher rates.
- **§4 findings (p.7):** "at a cost-benefit ratio of 4, the optimal number of samples is K ≤ 5 for all four models" (Llama 3.1 8B/70B, Code Llama 7B/13B, Fig. 4).
- **Abstract:** "optimal sampling attempts are often fewer than 10".
- **Fig. 5 (p.7):** the false-positive rate rises with K because task difficulty is bimodal and the late attempts are on hard tasks.
- **App. C (p.24, Fig. 13):** K ≤ 3 with parameters fitted to Llama 3.1 8B (Table 3, p.26: r1 = 0.87, r2 = 0.13, c = 1, s = 0.75, p1 = 0.58).

**5. Terminology:** imperfect verifier; false positive (passes the verifier, fails the extended tests); oracle verifier; generalization gap; cost-benefit ratio; resampling vs ranking vs majority voting (Table 1, p.4); completeness / soundness (App. C).

**6. Figures.**
- **Fig. 1 left (p.2):** scatter of models with GPT-4o's Pass@1 as a dashed horizontal line. "Models below the line" can never catch up.
- **Fig. 4 (p.7):** reward vs log K for each C/B ratio, with crosses at the optimum.

**7. What it gives the deck.**
- **Slides:** terminology ("verifier false-positive ceiling"); limitations; why N = 8 is not obviously too small.
- **Presenter sentence:** "With an imperfect checker, extra samples eventually add more confident wrong answers than right ones; in code the best budget was under ten samples."
- **Cuts / caveats:**
  - (a) Their verifier is binary accept/reject with no false negatives. Our probe is a ranker over a fixed pool, so the ceiling transfers only by analogy (inferred).
  - (b) The domain is code with human unit tests.
  - (c) Their small-K optimum depends on a nonzero false-positive cost. At C/B = 0 the curves keep rising (Fig. 4).

**8. Do not claim:**
- that Stroebl proves our N = 8 is optimal;
- that their result is about best-of-N ranking;
- "four unspecified models" (the card's wording). Fig. 4 names Llama 3.1 8B/70B and Code Llama 7B/13B.

## E1. Wang et al. 2023 — self-consistency

**Citation.** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (Google). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023; arXiv:2203.11171v4.

**1. Problem.** Greedy chain-of-thought decoding commits to one reasoning path. Sampling several paths and taking the answer most of them reach is more reliable [abstract, pp.1–2].

**2. Method** (p.3). Sample m paths $(r_i, a_i)$, parse the final answer $a_i$, and take the majority vote:
$$\hat a=\arg\max_a\sum_{i=1}^m\mathbb 1(a_i=a)$$
- Weighted variants use $P(r_i,a_i\mid\cdot)=\exp\big(\frac1K\sum_k\log P(t_k\mid\cdot)\big)$ (Eq. 1, length-normalised).
- Sampling (p.4): T = 0.5, k = 40 (UL2, LaMDA); T = 0.7, k = 40 (PaLM); T = 0.7 without top-k (GPT-3).
- m = 40, averaged over 10 runs (p.5). Few-shot; nothing is trained.
- Requires "a fixed answer set" (p.4).

**3. Setup.**
- Models: UL2-20B, GPT-3 (code-davinci-001/002), LaMDA-137B, PaLM-540B.
- Tasks: arithmetic (GSM8K, SVAMP, AQuA, AddSub, MultiArith, ASDiv), commonsense (CSQA, StrategyQA, ARC), symbolic (pp.4–5).
- Baseline: greedy CoT.

**4. Key results.**
- **PaLM-540B** (Tables 2–3, p.5): GSM8K 56.5 → 74.4 (+17.9); AQuA 35.8 → 48.3 (+12.5); SVAMP 79.0 → 86.6 (+7.6); StrategyQA 75.3 → 81.6 (+6.3); ARC-c 85.2 → 88.7 (+3.5).
- **code-davinci-002**, the source of the abstract's figures: GSM8K 60.1 → 78.0 (+17.9); AQuA +12.2; SVAMP +11.0; StrategyQA +6.4; ARC-c +3.9.
- **Aggregation on PaLM-540B** (Table 1, p.3): normalised weighted sum and majority vote are the same on GSM8K (74.1 vs 74.4), while unnormalised weighted average gives 56.3. Footnote 2: the LM "is not well calibrated and thus cannot distinguish well between correct solutions and wrong solutions".
- **Sample-and-rank** (by sequence log-probability) gains "much smaller" than SC at equal samples (Fig. 3, p.7).
- **Consistency correlates with accuracy** (Fig. 5, p.8).
- The gain "becomes more significant" with model scale (p.5).

**5. Terminology:** self-consistency; "sample-and-marginalize"; majority vote ("unweighted sum"); sample-and-rank; CoT-prompting; consistency (percentage of decodes agreeing with the final answer).

**6. Figures.**
- **Fig. 1 (p.2):** greedy path (wrong, $14) vs three sampled paths, then a vote ($18). The canonical self-consistency diagram.
- **Fig. 2 (p.6):** accuracy vs the number of sampled paths.

**7. What it gives the deck.**
- **Slides:** terminology (self-consistency = majority vote); the baseline row.
- **Presenter sentence:** "Self-consistency needs answers that can be matched exactly. In open medical VQA they rarely repeat, and on our shipped T = 0.7 pools majority vote loses to greedy."
  - Our number: judge −0.0094 [−0.0157, −0.0034], 2/8 positive (`NUMBERS_T07.md`, `final_eval_FINAL_cap320_T07.json`, `macro_over_open8.all.sc_minus_greedy_judge`).
- **Supports us:** footnote 2 (raw LM probabilities are poorly calibrated) motivates training a scorer rather than ranking by likelihood.
- **Cuts against us:** none directly. SC's gains are on closed answers and large models.

**8. Do not claim:**
- that the abstract's AQuA, SVAMP, StrategyQA and ARC numbers are PaLM-540B rows (they are code-davinci-002);
- that SC was ever evaluated on VQA in this paper.

## E2. Chen et al. 2023 — Universal Self-Consistency (USC)

**Citation.** Xinyun Chen, Renat Aksitov, Uri Alon, Jie Ren, Kefan Xiao, Pengcheng Yin, Sushant Prakash, Charles Sutton, Xuezhi Wang, Denny Zhou (Google). "Universal Self-Consistency for Large Language Model Generation." arXiv:2311.17311v1, 29 Nov 2023.

**1. Problem.** Self-consistency needs an answer that can be extracted and exact-matched. Free-form outputs have none [abstract, p.2].

**2. Method** (p.2; prompt Fig. 6, p.15).
- Sample responses, concatenate them, and ask the same LLM: "Evaluate these responses. Select the most consistent response based on majority consensus. Start your answer with 'The most consistent response is Response X'."
- One extra LLM call per question, listwise. Nothing is trained.
- 8 samples by default. T = 0.6 (PaLM 2-L) and 1.0 (gpt-3.5-turbo) (p.4).

**3. Setup.**
- Benchmarks: GSM8K, MATH; BIRD-SQL, ARCADE; GovReport, SummScreen; TruthfulQA (817 questions, scored by the GPT-judge and GPT-info models).
- Baselines: greedy, random, SC, SC-Exec (p.4).

**4. Key results.**
- **GSM8K, PaLM 2-L** (Table 1, p.5): greedy 85.7, random 82.9, SC 90.4, USC 90.2.
- **MATH, PaLM 2-L** (Table 1, p.5): greedy 30.8, SC 37.9, USC 37.4.
- **TruthfulQA, PaLM 2-L** (Table 4, p.6): GPT-judge greedy 62.1 → USC 67.7. gpt-3.5-turbo: 79.8 → 82.5.
- **Oracle over the same 8 samples** (Tables 7 and 11, pp.13–14): GSM8K 96.2, MATH 57.2; TruthfulQA (PaLM 2-L) GPT-judge 93.8.
- **More samples** (p.6): "the accuracy on GSM8K decreases with 16 samples". Fig. 3 (p.7) shows 89.2 at k = 16, −1.4 vs SC.
- **Response order** (Table 5, p.6): little effect, GSM8K 89.7 ± 0.3 over 5 shuffles.

**5. Terminology:** USC; "most consistent response"; SC-Exec (execution-based voting); oracle selection.

**6. Figures.**
- **Fig. 1 (p.2):** workflow (sample, concatenate, "select most consistent", final). Simple.
- **Fig. 3 (p.7):** accuracy vs k.

**7. What it gives the deck.**
- **Slides:** the "training-free open-text selectors" landscape; honest gaps.
- **Presenter sentence:** "USC handles free text by asking the model which answer is most consistent. It matches voting on math but stays far below its own oracle (GSM8K 90.2 vs 96.2), and we have not run it on our pools."
- **Cuts against us:** it is a standard open-text baseline that we have not run (gap).

**8. Do not claim:**
- that USC beats SC (it matches);
- that position bias is unaddressed (the card says so, but Table 5 tests ordering);
- any USC number on medical VQA.

## E3. Ontalvilla, Azkune, Ormazabal 2026 — Embedding-Based Agreement (EBA)

**Citation.** Paula Ontalvilla, Gorka Azkune, Aitor Ormazabal (HiTZ, UPV/EHU). "Agreement in Representation Space for Open-Ended Self-Consistency." arXiv:2606.12003v1 [cs.CL], 10 Jun 2026.

**1. Problem.** Extend self-consistency to open-ended outputs by treating agreement as geometric concentration in embedding space instead of exact string overlap [abstract].

**2. Method** (§3.2, p.3; App. A, p.10).
- Sample N generations and embed them: $e_i=f(y_i)$ with Qwen3-Embedding-8B by default.
- Agglomerative clustering: cosine distance, average linkage, number of clusters chosen by the silhouette score.
- Return the sample closest to the centroid of the largest cluster.
- Nothing is trained.
- Variants:
  - "native hidden representations" = "the final sampled token's hidden representation" of the generator (§5.5, p.6; layer not stated);
  - nearest / farthest from the global centroid (§6.2).
- Sampling: T = 0.7, top_p 1.0; a fixed pool of 512 per input; n = 1…256; 50 bootstrap draws (p.4, p.10).

**3. Setup.**
- Models: Llama 3.1-8B, Qwen3-8B, plus Llama 70B and Qwen 32B.
- Tasks: HumanEval (accuracy), MATH500 (Qwen-8B judge, 95.5% agreement with human labels on 200 items), CNN/DM (first 1,000, ROUGE-1).
- Baselines: USC, Self-Certainty, random (pp.4, 11).

**4. Key results** (Tables, App. D, pp.14–16).
- **MATH500, Llama 8B** (Table 2): EBA 52.77 (n = 1) → 60.03 (n = 8) → 63.85 (n = 256); USC 59.66 at n = 8.
- **HumanEval, Qwen 8B** (Table 1): EBA 86.17 at n = 8; random 83.45.
- **CNN/DM, Llama 8B** (Table 3): EBA 29.58 at n = 8 vs random 29.20; USC 28.40 (below random).
- **Exact-match SC still wins on math** (Table 7, p.16; Llama 8B MATH500, n = 8): SC 63.85; EBA on the extracted answer 63.66; EBA on the full generation 60.03.
- **Centroid distance** (Table 9, p.16, n = 256):
  - Llama 8B: nearest-centroid 63.87, farthest 29.98, random 52.77.
  - Qwen 8B: EBA 89.85, nearest 85.31, farthest 74.73.
- **USC degrades with n** (p.5): "it loses around two points on HumanEval for the Llama model".

**5. Terminology:** Embedding-Based Agreement (EBA); "agreement in representation space"; dominant cluster; native hidden representations; Self-Certainty (SCe).

**6. Figures.**
- **Fig. 1 (p.1):** concept sketch, with consistent generations in a dense region and inconsistent ones scattered.
- **Fig. 6 (p.8):** nearest vs farthest from the centroid, against n. A strong visual.

**7. What it gives the deck.**
- **Slides:** comparison (closest *training-free* representation-space method); honest gaps.
- **Presenter sentence:** "EBA reads agreement from embeddings without labels. Our probe reads correctness from the generator's own hidden states with labels, so it can in principle reject a popular wrong answer."
- **Cuts against us:**
  - (a) A training-free hidden-state baseline exists (native embeddings, §5.5) and we have not run it on our pools. A reviewer may ask whether the labels are needed (gap).
  - (b) EBA shows representation geometry alone already beats random.

**8. Do not claim:**
- a "~20-point spread" between nearest and farthest for Qwen 8B (the card says this; the table gives 85.31 − 74.73 = 10.58). The "nearly 20 points" in the text refers to drops at large n (p.8), and the large gap is the Llama 8B row.
- that EBA beats SC on math (it does not, Table 7);
- that EBA is trained (it is not).

## E4. Hagen et al. 2026 — Wasserstein Equilibrium Decoding (WED / BDG-W): our one head-to-head baseline

**Citation.** Luca Hagen, Johanna P. Müller, Weitong Zhang, Mengyun Qiao, Bernhard Kainz (FAU Erlangen-Nürnberg; Imperial; UCL). "Wasserstein Equilibrium Decoding for Reliable Medical Visual Question Answering." arXiv:2605.18313v2 [cs.CV], 13 Jun 2026. Code: github.com/luca-hagen/Wasserstein-BDG-medical-VQA.
- The paper calls the method **BDG-W** or **W-BDG**.
- Lineage: Consensus Game (Jacob et al., 2310.09139), then Bayesian Decoding Game (BDG, Zhang, Zang & Kainz, 2410.01064 [abstract]; NeurIPS 2025 per WED ref. 31), then BDG-W.

**1. Problem.**
- Small VLMs (2–8B) suit on-premise clinical use but produce plausible wrong answers.
- Game-theoretic decoding (BDG) was text-only and closed-ended.
- In open-ended medical VQA, near-synonymous candidates ("liver" vs "hepatic region") stop BDG's exact rank-order stopping rule from converging (pp.1–2).

**2. Method, step by step** (§2, pp.3–6; hyperparameters p.7).

**Step 1 — candidate set.**
- Sample up to 8 candidates by nucleus sampling at **T = {0.5, 1.0}** (p.5; this *is* in the PDF).
- Canonicalise each sample: strip filler prefixes, punctuation and extra sentences.
- Sample until 8 unique candidates exist or the maximum number of calls is reached. "Nmax = 16 and n = 8" (p.7).
- In the code (code): top_p 0.98, top_k 100, at most 24 new tokens; candidates longer than 10 words are filtered; SapBERT semantic deduplication at 0.99; k-means picks an even number of candidates.

**Step 2 — signalling game** (p.4).
- Generator G receives a signal s ∈ {correct, incorrect} and plays a distribution over Y.
- Verifier V sees (x, y) and plays a distribution over {correct, incorrect}.
- Shared utility = whether their preference orderings agree:
  $$u_G=u_V=\tfrac12\sum_{s}\mathbb 1\{O_G=O_V\mid s\}\quad\text{(Eq. 1)}$$
- **σ-separated equilibrium** (anti-collusion):
  $$\min_y|a_G(y|x,\text{corr})-a_G(y|x,\text{inc})|>\sigma_G,\qquad \min_y|a_V(\text{corr}|x,y)-a_V(\text{inc}|x,y)|>\sigma_V$$

**Step 3 — initialisation** (pp.4–5). The same VLM is generator and verifier, teacher-forced with the image.
- Generator: log-probability of the candidate under a "correct" vs an "incorrect" prompt, softmax over candidates, giving $a^{(1)}_G(\cdot\mid x,s)$.
- Verifier: log-probability of the label tokens correct/incorrect given (x, y_i), giving $a^{(1)}_V(\cdot\mid x,y_i)$.
- The paper's canonical form scores the final label token of "Question … Answer: {y_i} Is this answer correct? Response: correct".
- (code) The generator actually scores the **candidate tokens** after "{question}\n\nAnswer: " under a "find the CORRECT answer" vs a "distractor selector" system prompt, with length normalisation.
- (code) Because `reverse_incorrect=True`, $G_i$ is the rank-reversal of $G_c$.
- (code) The verifier scores the token "C" vs "I" after "Verdict (correct/incorrect): ".
- Paper and code differ here. Describe it as "teacher-forced likelihood scoring under correct/incorrect framings".

**Step 4 — no-regret updates** (Eq. 2–3, p.4). For iterations t = 1, 2, …:
$$a_G^{(t+1)}(y|x,s)\propto\exp\!\Big(\frac{\tfrac12\tilde a_V^{(t)}(s|x,y)+\lambda_G\log a_G^{(t)}(y|x,s)}{1/(\eta_G t)+\lambda_G}\Big),$$
and symmetrically for V (λ = stiffness toward the prior; η = learning rate).
- These updates are arithmetic on n × 2 tables. **No further VLM calls are made after initialisation** (inferred from Eq. 2–3; confirmed by the code).

**Step 5 — stopping rule (the paper's contribution)** (Eq. 4–5, pp.5–6).
- Ground metric $D_{ij}$: cosine distance between SapBERT embeddings of the candidates.
- Let $p_G=a_G(\cdot|x,\text{corr})$ and $p_V=$ the normalised verifier "correct" scores. Then:
  $$W_1(p_G,p_V;D)=\min_{\gamma\in\Pi(p_G,p_V)}\sum_{ij}\gamma_{ij}D_{ij},\qquad \tilde W_1^{(t)}=\frac{W_1}{\sigma_V^{(t)}+\varepsilon}$$
- Stop when $\tilde W_1<\delta_W$ (and σ-separation holds).
- Classic BDG instead requires $O_G=O_V$ exactly.

**Step 6 — output.** Not stated in the paper. (code) `best_idx = argmax(aG_c)`: the candidate with the highest final generator probability under the "correct" signal.

**Hyperparameters** (p.7): λ_G = λ_V = 0.4; η_G = η_V = 0.4; σ = 0.005; δ_W = 0.2; at most T = 500 iterations; one A100 80GB.

**What is trained:** nothing.

**Per-question cost** (inferred from the code): about 3–4 teacher-forced VLM scoring passes per candidate (generator correct/incorrect, verifier C/I), plus SapBERT embeddings, plus cheap game iterations. Our probe needs 1 teacher-forced pass per candidate.

**3. Setup** (pp.6–7).
- Data: VQA-RAD (315 images, 3,515 QA) and PathVQA (4,998 images, 32,799 QA), the **open-ended test subsets with yes/no questions excluded**.
- Generators: Qwen3-VL-2B/4B/8B, Gemma-3-4B, MedGemma-4B.
- Metrics: EM, token-F1, SapBERT cosine ("Sem"), and judge accuracy by **Grok 4.1 Fast**. The code's default is `x-ai/grok-4-fast` (code).
- 5 seeds; paired bootstrap with 10k resamples, p < 0.01.
- Baselines: Greedy; Self-Contrastive Decoding (SCD); Verifier-only (argmax $a^{(1)}_V$); BDG-Classic; BDG-W.

**4. Key results.**
- **Gain over greedy, judge** (Table 1, p.8), VQA-RAD / PathVQA, in pp: Qwen-2B +3.5 / +2.2; Qwen-4B +3.8 / +3.2; **Qwen-8B −1.2 / +1.2**; Gemma-4B +2.3 / +5.0; MedGemma-4B +2.7 / +2.0. "Gains diminish with size."
- **Main results** (Table 3, p.8):
  - Qwen-2B VQA-RAD judge 33.50 → 37.04.
  - MedGemma-4B VQA-RAD judge 54.00 → 56.70.
  - PathVQA judge is low in absolute terms: greedy 4.98 (Qwen-2B) to 23.88 (MedGemma); EM 0.00–2.99.
- **Verifier-only sometimes ties or beats BDG-W** (Table 3): Qwen-4B PathVQA judge 13.33 vs 11.14; Qwen-4B VQA-RAD 40.90 vs 40.30; Qwen-8B VQA-RAD 39.80 vs 39.29.
- **BDG-W ≈ BDG-Classic in accuracy** (p.7). The Wasserstein rule cuts iterations: 20.17 ± 4.99 vs 25.22 ± 4.44 (Wilcoxon p < 0.001) (p.9), "approximately 20%" [abstract].
- **Abstract claim:** "Gemma-3-4B with BDG matches MedGemma-4B under greedy decoding". PathVQA judge: 23.89 BDG-W vs 23.88.

**Our controlled head-to-head** (from `COMPARISON_BRIEF.md` §1.1, `wed_vs_probe_2026-10-16.json`; not re-verified here):
- Setting: identical T = 0.7 cap320 pools, WED via the authors' code.
- Probe − WED, judge: PathVQA +0.0843 [+0.0711, +0.0978]; SLAKE +0.0574 [+0.0326, +0.0818]; VQA-RAD +0.0050 [−0.0515, +0.0580] (tie).
- WED − greedy on PathVQA: −0.0319 (loss).

**5. Terminology:** signalling game; generator / verifier (discriminator); Nash equilibrium; σ-separated equilibrium (σ-SE); collusion; Markovian no-regret update; stiffness λ; order-match stopping criterion; Wasserstein-1 / earth mover's distance; ground metric; separation-weighted W̃₁; temperature-diverse sampling; semantic deduplication; Self-Contrastive Decoding.

**6. Figures.**
- **Fig. 1 (p.3):**
  - CT thumbnail and "Q: Where is the abnormality located?";
  - four coloured candidates (Liver, Hepatic Region, Spleen, Abdomen) passing through solid Generator rank boxes and dashed Verifier rank boxes;
  - on the right, rank trajectories over iterations, with two vertical lines: "Semantic alignment: W-BDG stops" before "Exact alignment: BDG stops".
  - Recreate it on the WED slide in our palette.
- **Fig. 2 (p.6):** ranking phases (red, orange, green) and W̃₁ vs t with the δ_W line.

**7. What it gives the deck.**
- **Slides:** comparison (head-to-head); background (medical, training-free state of the art).
- **Presenter sentence:** "WED makes the model play a game against itself: a generator and a verifier re-rank the candidates until they agree up to synonyms. It needs no training. On identical Lingshu pools it does not beat greedy; our trained probe beats it on PathVQA and SLAKE and ties on VQA-RAD."
- **Cuts against us / caveats:**
  - (a) WED's gains shrink with model size and reverse on Qwen-8B VQA-RAD. Lingshu-7B is at the top of their size range, so WED ≈ greedy on our pools is consistent with their own trend and partly a size effect, not only a method effect (inferred).
  - (b) We fed WED our T = 0.7 samples, not its own T = {0.5, 1.0} sampling with up to 16 calls and its prompts.
  - (c) WED filters out answers longer than 10 words and falls back to the modal answer when nothing survives (our adapter; 55/645 on SLAKE).
  - (d) WED excludes yes/no questions; check that our subsets match (COMPARISON_BRIEF says VQA-RAD n = 200 is consistent).
  - (e) WED is training-free. Our probe used labelled in-domain candidates, so the comparison is "trained vs untrained" by design.

**8. Do not claim:**
- "we beat WED's published numbers" (different generators and judge);
- that WED's iterations are VLM calls (the card's "O(iterations) verifier calls" is wrong);
- that WED evaluated SLAKE;
- that WED's T values are absent from the paper (`COMPARISON_BRIEF.md` erratum 5 is wrong; see Part 3).

---

# Part 2 — All cards in categories D and E

Relevance: S = slide-worthy, M = mention, — = skip. Core papers are covered in Part 1.

| id | title (short) | one line | rel. | why |
|---|---|---|---|---|
| hu2026oraclegap | Oracle Gap & Signal Fidelity | gain = recoverable·quality − ref-correct·harm | S | the vocabulary for our accounting |
| brown2024monkeys | Large Language Monkeys | coverage scales log-linearly; voting and RMs plateau | S | the origin of the "coverage" term (Group A reads the PDF) |
| stroebl2024limits | Limits of resampling | verifier false positives cap accuracy; optimal K small | S | ceiling term |
| bay2026sampling | Modal & correlation ceiling | voting capped at π_mode; identifiability gap | S | ceiling term + figure template |
| chen2024morecalls | More LLM calls | Vote accuracy can rise then fall with calls | M | anti-scaling citation (Bay §4.3 cites it) |
| song2024mindgap | Mind the Gap | generation–verification gap | — | self-improvement, not selection |
| huang2025bestofn | Is BoN the best? | BoN optimal only under coverage; reward-hacks at large N | M | theory backdrop |
| di2025bestofmajority | Best-of-Majority | frequency filter then RM rerank | M | closest named relative of "score + vote" readout |
| zhang2026bestofevidence | Best-of-Evidence | partial verification over BoN, medical VQA | M | Group B covers it (B PDF) |
| sunkaraneni2026boosting | Agentic boosting | coverage vs identifiability on SWE-bench | — | domain far |
| chen2026cofailure | Co-failure ceiling | β = all-models-wrong rate caps ensembles | — | multi-model |
| chen2026routinggap | Routing gap noise | single-draw oracle noise | — | routing |
| sun2026routeguard | RouteGuard | certify routing gains | — | routing |
| lu2026routingplateau | Routing plateau | routers converge below oracle | — | routing |
| khalaf2025rewardhacking | Inference-time reward hacking | BoN reward rises then falls with N | M | "why not N = 64" |
| bugaud2026hiddenclones | Hidden Clones | same-family VLM errors correlated | — | ensembles of models |
| hagen2026wed | WED | game-theoretic medical VQA decoding | S | head-to-head |
| wang2023selfconsistency | Self-consistency | majority vote over CoT samples | S | the baseline term |
| chen2023usc | USC | LLM picks most consistent response | S (row) | open-text baseline, not run |
| ontalvilla2026eba | EBA | cluster embeddings, pick dominant centroid | S (row) | closest training-free relative |
| freitag2022mbr | MBR with neural metrics | MBR beats beam on human eval | M | MBR definition |
| bertsch2023mbr | MBR all the way down | many decoders are MBR | M | MBR term |
| kuhn2023semantic | Semantic entropy | entropy over meaning clusters | M | why exact match fails in free text |
| aggarwal2023adaptive | Adaptive-Consistency | adaptive stopping for SC | — | future-work line at most |
| li2024escape | ESC | early-stopping SC | — | same |
| wang2024soft | Soft-SC | likelihood-weighted SC | — | marginal |
| jiang2023llmblender | LLM-Blender | pairwise ranker + fuser | — | multi-LLM ensembling |
| jain2023lightweight | Lightweight reranking | pairwise-agreement reranker | — | covered by MBR mention |
| jacob2023consensus | Consensus Game | generator–discriminator equilibrium | M | WED lineage |
| zhang2024bdg | Bayesian Decoding Game | BDG, collusion-aware | M | WED lineage |
| li2024moreagents | More Agents | sample-and-vote scales | — | redundant with SC |
| wang2025ranked | Ranked-voting SC | ranked lists + Borda/IRV | — | marginal |
| oh2025lsc | Latent SC | trained summary tokens on KV cache | M | trained, internal-state relative |
| nguyen2026rcs | Radial Consensus Score | distance to embedding Fréchet mean | — | covered by EBA |
| kang2025selfcertainty | Self-certainty | logit-only confidence score for BoN | M | training-free logit baseline |
| jin2026looklight | Look Light, Think Heavy | CoT hurts multimodal perception | M | belongs with "why direct decoding", not E |

**Verified content for S and M cards** (core ones: see Part 1).

- **brown2024monkeys** — Group A reads it. Bay uses it: "majority voting and reward models plateau beyond several hundred samples and fail to fully scale with the sample budget" (Bay p.2 quoting Brown).
- **chen2024morecalls** — Bay p.14: Chen et al. "find voting performance non-monotone in the number of calls because a benchmark mixes easy and hard problems". Stroebl p.7 cites the same "inverse U-shaped accuracy curves". No numbers verified.
- **huang2025bestofn / khalaf2025rewardhacking / di2025bestofmajority** — card-only, not verified. Quote as "[card]" or leave out. Hu cites related theory: Dorner et al. ROC-n-reroll; Setlur et al. (Hu p.3).
- **freitag2022mbr** [abstract, verified]:
  - Freitag, Grangier, Tan, Liang; "Accepted at TACL, presented at NAACL22".
  - Finds "model estimates and translation quality only vaguely correlate".
  - Selects samples by highest estimated quality under BLEURT instead of highest probability, giving "significant improvement in human evaluations".
- **bertsch2023mbr** [abstract, verified]:
  - Bertsch, Xie, Neubig, Gormley; arXiv 2310.01387, "Under submission".
  - MBR chooses the output with the lowest expected risk among candidates; several recent methods are special cases of it.
  - **MBR definition for the deck** (standard textbook form, not from our PDFs — mark as such):
    $$\hat y=\arg\max_{y\in\mathcal H}\frac1{|\mathcal H|}\sum_{y'\in\mathcal H}u(y,y')$$
    With $u=\mathbb 1[y=y']$ this reduces to majority vote (inferred).
- **kuhn2023semantic** [abstract, verified]:
  - Kuhn, Gal, Farquhar, ICLR 2023 (Spotlight).
  - "Semantic entropy" is entropy over meaning-equivalence classes.
  - Unsupervised, single model; predicts QA accuracy better than baselines.
- **jacob2023consensus** [abstract, verified]:
  - Jacob, Shen, Farina, Andreas; arXiv 2310.09139.
  - Generator vs discriminator signalling game; EQUILIBRIUM-RANKING.
  - "LLaMA-7B ... outperforms LLaMA-65B and PaLM-540B" on several benchmarks.
  - "training-free".
- **zhang2024bdg** [abstract, verified]:
  - Zhang, Zang, Kainz (Kainz is also on WED); arXiv 2410.01064.
  - Multistage Bayesian game with "Correctness Alignment" and "Ambiguity Calibration"; no training.
  - The "78.1 vs 76.6" example is card-only and not verified.
  - WED p.2: BDG "enforces a σ-separated equilibrium condition to avoid" collusion.
- **kang2025selfcertainty** [abstract, verified]:
  - Kang, Zhao, Song, NeurIPS 2025.
  - A reward-free score from the model's own output distribution; scales with N; works on open-ended tasks.
  - EBA Table 2 (p.14): SCe on MATH500 Llama 8B peaks at 54.95 (n = 16) and falls to 51.65 (n = 256).
  - The formula in the card is not verified.
- **oh2025lsc** — card only. If shown, tag [card]. It is trained (contrastive summary tokens), so it is a "trained internal-state" relative.
- **jin2026looklight** — card only, abstract-level. Use it on the "why direct decoding" slide, not in E.

---

# Part 3 — Recommendations

**Own slide** (one each):
- **WED.** Method diagram (redrawn Fig. 1) plus the head-to-head table.
- **Bay.** The coverage-vs-selection plot with the identifiability gap and modal ceiling; redraw Fig. 1a/Fig. 10 using our oracle@N, probe and majority-vote numbers.
- **Hu.** The gain-decomposition slide: oracle gap → quality vs harm, showing fixed/harmed for our probe if we have them.

**Comparison table rows** (method | trained? | signal | needs extra model call? | open-text? | headline in own units):
- self-consistency (Wang)
- USC (Chen)
- EBA (Ontalvilla)
- WED (Hagen)
- self-certainty [abstract]
- MBR (Freitag/Bertsch) [abstract]
- our probe

**Terminology slide must define:**
- **oracle@N = pass@N = coverage.** The fraction of questions with at least one correct sample among N. Brown/Bay usage. Say explicitly that Hu's "coverage" means something else.
- **selection**
- **selection gap / identifiability gap** (Bay) = oracle@N − selected.
- **oracle gap** (Hu) = oracle − reference.
- **recoverable** (Hu).
- **conditional selection quality q and harm h** (Hu).
- **effective capture** (Hu) vs our **selection efficiency** (selected / oracle@N). Not the same.
- **modal ceiling π_mode** (Bay).
- **verifier false-positive ceiling** (Stroebl). Infinite-budget accuracy = P(correct | passes verifier).
- **self-consistency = majority vote** (Wang); **USC**; **MBR**; **semantic agreement** (EBA, semantic entropy).
- **greedy decoding**; **temperature sampling**.
- **WED/BDG terms:** signalling game, equilibrium, Wasserstein-1.

**Errors found:**

1. **`COMPARISON_BRIEF.md` erratum 5 is itself wrong.**
   - WED p.5 states: "we sample up to 8 candidate answers from the VLM using nucleus sampling at temperatures T = {0.5, 1.0}".
   - So T ∈ {0.5, 1.0} *is* in the PDF. It is also in the code (`config.py: temperatures = (0.5, 1.0)`).
   - "Nmax = 16 and n = 8" is on p.7.
2. **COMPARISON_BRIEF §1.1 footnote 3 is stale under DECK_SPEC item 8.** The shipped method is now single T = 0.7, so the VQA-RAD 0.4900 *is* the shipped-method number. The "final method uses T = 0.2 there" remark belongs only to the labelled per-benchmark variant.
3. **The self-consistency figure differs between files.**
   - COMPARISON_BRIEF, DESIGN_EVIDENCE and the E cards give −0.0070 [−0.0111, −0.0029], 1/8. That is from the per-benchmark-temperature final pools.
   - `NUMBERS_T07.md` (shipped T = 0.7) gives −0.0094 [−0.0157, −0.0034], 2/8 (judge), and EM −0.0015 (tie).
   - The deck should use the T07 numbers.
4. **WED card.**
   - "O(iterations) verifier calls" is wrong: the iterations are closed-form updates on precomputed tables (Eq. 2–3).
   - The card leaves out the final output rule (argmax generator-correct, code).
   - The +3.5 comes from Table 1 / text (37.04 − 33.50 in Table 3).
5. **EBA card.** "~20-point spread" uses the Qwen 8B row, where nearest − farthest = 10.58 (Table 9).
6. **USC card.** "unaddressed position-bias risk" contradicts Table 5 (the ordering ablation).
7. **Self-consistency card.** "minor rounding differences" is wrong; those are code-davinci-002 rows (already flagged in COMPARISON_BRIEF erratum 11).
8. **Stroebl card.**
   - "Four (unspecified) models": they are Llama 3.1 8B/70B and Code Llama 7B/13B (Fig. 4, p.7).
   - "K ≤ 5 … [Fig. 4 caption]" is actually in the §4 text (p.7), as COMPARISON_BRIEF erratum 14 already says.
   - The card's relation_to_us transfers an accept/reject false-positive ceiling to our ranking probe without saying that this is an analogy.
9. **Hu card.**
   - "multiplicatively by signal fidelity" is not in the paper.
   - "Models: not extracted": the PDF names Qwen3.6-35B-A3B-BF16 (p.6).
   - The 4.69% is effective harm (Table 4), not the conditional h of Eq. 6.
10. **Bay card.** The one-line summary drops the qualifier that self-consistency falls to 0 only on problems whose mode is wrong (Cor. 3).
11. **E category intro.** It states SC "sits at the random-pick floor and does not improve from N = 2 to N = 16" as if it held for "our open-ended medical VQA benchmarks". The cards themselves scope this to a fixed 225-question Kvasir-VQA set.
12. **`FIGURE_STYLE_NOTES.md`.** "none of the 34 papers uses 'Borda count'" holds for the core set only. The non-core self-certainty card describes "Borda voting" (Kang et al.), the nearest named relative of rank averaging. It is unverified from the PDF.

**Honest gaps to state:**
- USC, EBA (including its native-hidden-state variant), self-certainty and MBR were not run on our pools.
- WED was run only on our pools, not with its own sampling.
- Fixed/harmed counts in Hu's units should be reported if available; otherwise mark them "not measured".
