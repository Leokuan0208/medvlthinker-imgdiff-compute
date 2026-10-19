# PAPER_NOTES_F — Uncertainty, calibration, hallucination detection (Group F)

Read on 2026-10-19 from the four core PDFs (text in `tmp/txt/F_*.txt`, page numbers = PDF page; for the Nature paper
the journal page is PDF page + 624), the 32 cards in `cards_F.json`, and six arXiv abstract pages (fetched as
verification of non-core cards: 2406.15927, 2603.21693, 2511.12693, 2510.09256, 2207.05221, 2304.13734).
Tags: [p.N] = PDF page; [abstract] = arXiv abstract page; [card] = card only, not re-verified; (inferred) = my
reasoning, not stated in a source. Our-project numbers are quoted from `DESIGN_EVIDENCE.md` /
`PROJECT_RETROSPECTIVE_2026-07-29.md` with their source named.

**Standing rule reminder (CLAUDE.md rule 6).** Three of these papers frame their use case as abstention / deferral /
"refuse to answer" (Farquhar's AURAC, Khanmohammadi's triage, Wienholt's filtering). The deck may cite their
*confidence signals and measurements*; it must not present rejection/deferral as something we do or propose.

---

## 0. The one-slide synthesis (what the deck needs from this group)

**Uncertainty/confidence scores and best-of-N verification answer different questions.**

| | confidence / uncertainty estimation (this group) | best-of-N verification (our method) |
|---|---|---|
| question asked | "is *this* answer (or this question) likely wrong?" | "which of the N candidates for this question is right?" |
| unit scored | one answer per question (Byun, Khanmohammadi: greedy, one answer), or the *question* itself (semantic entropy: SE(x) is a property of the input x) | every candidate of the same question |
| what matters | ranking **across questions** (AUROC is pooled over questions) and/or **calibration** (score = probability) | ranking **within a question** only; the argmax |
| metric | AUROC, ECE/ACE, AURAC/AURC | accuracy of the selected answer; selection efficiency = selected / oracle@N |

Three consequences the presenter can state:
1. **A question-level uncertainty gives zero selection signal.** SE(x) is one number per question, identical for all 8
   candidates, so it cannot pick among them (inferred from the definition, Farquhar Eq. 3, [p.8]). The only
   per-candidate quantity inside semantic entropy is the cluster probability $P(C_i\mid x)$; in the discrete variant that
   is the cluster's share of the samples [p.8], so "pick the most probable cluster" **is a semantic majority vote**
   (inferred). Our string-level majority vote is measured: probe − vote = +0.0895 J / +0.0124 EM, both significant, measured on the
   September pooled probe, not re-run on the final method (DESIGN_EVIDENCE row "majority vote minus probe" −0.0895 / −0.0124, `readout_variants_2026-10-15.json`). The
   July-era "majority trap" (correct answer is a minority in 74–90% of recoverable questions; retrospective
   line ~399) is why voting is the wrong aggregator here.
2. **Calibration cannot change a selection.** Platt scaling / temperature scaling are strictly monotone, so they
   "preserve the rank ordering" and leave AUROC unchanged (Byun §6 [p.6], Table 2 [p.7]; Khanmohammadi S1-C [p.11]).
   The same monotonicity means they cannot change which candidate is the argmax within a question (inferred). Our
   rank-average readout is likewise invariant to any strictly monotone rescaling of each probe (inferred from the
   readout definition) — which also sidesteps Khanmohammadi's "incomparable scales" problem [p.4].
3. **Hidden-state probes are the shared mechanism, the target differs.** Farquhar's "embedding regression"
   (logistic regression on the final hidden state predicting correctness) [p.8–9], Kadavath's P(IK), Azaria's
   SAPLMA, Kossen's SEPs and Khanmohammadi's four trained probes all read internal states; ours is the
   *per-candidate correctness* variant, used as a within-question ranker rather than as a question-level alarm.

**Honest gap (selectors never run on the current method's pools):** mean token log-probability / perplexity
(uncomputable: "dumps store one scalar per question, not per candidate", `AUDIT_2026-09-18.md` line 230),
P(True)-style self-evaluation, semantic-cluster vote / semantic entropy, verbalized confidence, CEBaG/VASE-style
vision-contrast scores, and an SEP-style (entropy-target) probe. **Correction to DESIGN_EVIDENCE:** a P(True)-type
zero-shot self-verification (`p_yes_norm`) *was* run as a best-of-8 selector in the July era and scored **below random
(0.715 vs 0.720)** and below greedy (0.319 vs 0.343) (retrospective N32 and line ~1496). It was not re-run on the
final pools, so the honest wording is "tested once in an earlier era, failed; not re-tested on the final method".

**Medical-VLM overconfidence evidence (for the background slide):** Byun Table 1 [p.4] — all eight general VLMs
(2B–38B) overconfident on pooled VQA-RAD/SLAKE-EN/VQA-Med-2019, gap 0.192–0.338; gap larger on open-ended than
closed; scaling does not fix it. Khanmohammadi Table I [p.4] — inference-only estimators ECE 0.31–0.47 on GMAI and
PATH-VQA; best AUROC "near 0.63" [p.7].

---

# Part 1 — Core PDFs

## 1. Farquhar et al. 2024 — semantic entropy

**Citation.** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. "Detecting hallucinations in large language
models using semantic entropy." *Nature* 630, 625–630 (2024). doi:10.1038/s41586-024-07421-0. Open access. (No arXiv
id on the PDF; the ICLR 2023 predecessor is Kuhn et al., arXiv 2302.09664.)

**1. Problem.** LLMs "confabulate": they give fluent answers that are wrong *and arbitrary* — sensitive to the random
seed (e.g. KRASG12 'C' vs 'D' for Sotorasib) [p.1]. Naive entropy over token sequences is misleadingly high when the
same meaning is phrased many ways ("Paris", "It's Paris", "France's capital Paris") [p.2 Fig. 1a]. They want an
unsupervised detector that measures uncertainty over *meanings*. Explicitly *not* targeted: errors that are consistent
(trained-in misconceptions), lying for reward, systematic reasoning failure [p.1, p.5].

**2. Method.**
- Predictive entropy (baseline concept), Eq. (1) [p.7]:
  $$\mathrm{PE}(x) = H(Y\mid x) = -\sum_y P(y\mid x)\ln P(y\mid x)$$
- Sequence log-probability $\log P(\mathbf{s}\mid x) = \sum_i \log P(s_i\mid \mathbf{s}_{<i}, x)$; they length-normalise to
  $\frac{1}{N}\sum_i^N \log P(s_i\mid \mathbf{s}_{<i},x)$ [p.7] (this is the "naive entropy" baseline's ingredient and is
  the same quantity as mean token log-prob / log-perplexity).
- Pipeline [p.7]: (1) sample $M$ answers at temperature 1, nucleus $p=0.9$, top-$K=50$, plus one answer at $T=0.1$ as
  the "best generation" whose accuracy is assessed; (2) cluster by **bidirectional entailment** (A⇒B and B⇒A, in the
  context of the question), greedy: compare to the first member of each existing cluster (Extended Data Fig. 1 [p.12]);
  entailment by GPT-3.5 prompt (main) or DeBERTa-large-MNLI [p.9]; (3) entropy over clusters.
- Cluster probability, Eq. (2) [p.8]: $P(c\mid x) = \sum_{\mathbf{s}\in c} P(\mathbf{s}\mid x)$.
- Semantic entropy, Eq. (3)–(5) [p.8], estimated by Rao–Blackwellised Monte Carlo over the sampled clusters:
  $$\mathrm{SE}(x) \approx -\sum_{i=1}^{|C|} P(C_i\mid x)\log P(C_i\mid x),\qquad P(C_i\mid x)=\frac{P(c_i\mid x)}{\sum_c P(c\mid x)}$$
- **Discrete semantic entropy** [p.8]: $P(C_i\mid x)\approx \frac{1}{M}\sum_{m=1}^{M}\mathbb{1}[c_m=C_i]$ — the fraction of
  samples in the cluster; needs no token probabilities (black-box).
- Trained vs frozen: nothing trained for SE; the LLM is frozen. **N = 10 generations** ("We use ten generations",
  [p.4]); 3 new answers per question in the biography setting [p.5]. Output: one scalar per *question*. Selection rule:
  none — the score is used to decide whether to trust/answer, not which sample to return.

**3. Setup.** Models: LLaMA 2 Chat 7B/13B/70B, Falcon Instruct 7B/40B, Mistral Instruct 7B (sentence-length);
GPT-4 for FactualBio [p.2]. Data: TriviaQA, SQuAD 1.1, BioASQ (11B), NQ-Open, SVAMP; 400 train + 400 test per
dataset [p.8]; context passages withheld to induce confabulations [p.8]; answers 96 ± 70 characters [p.2].
FactualBio: 21 people, 150 claims, 45 incorrect [p.9–10]. Correctness: GPT-4 judge vs reference (sentence-length)
[p.9]; SQuAD-F1 > 0.5 (short-phrase) [p.9]. Metrics: AUROC for "answer is incorrect", AURAC [p.3, p.9]. Baselines:
naive entropy, P(True) (few-shot, up to 20 examples, "supervised in-context") [p.9], **embedding regression** —
"take the final hidden units and train a logistic regression classifier" to predict whether the model was correct
[p.8–9].

**4. Key results.**
- Averaged over 30 model×task combinations, AUROC: **semantic entropy 0.790**, naive entropy 0.691, P(True) 0.698,
  embedding regression 0.687 [p.3]; SE "between 0.78 and 0.81 AUROC" across families/scales [p.3–4].
- Embedding regression "performs well in-distribution but fails when the distribution of questions is different"
  [p.9]; OOD bars = trained on one of the four other datasets (Fig. 2, [p.3]).
- P(True) "seems to improve with model size" [p.4].
- Discrete SE "performs similarly" to the probability-weighted estimator [p.3].
- FactualBio (GPT-4): discrete SE has higher AUROC and AURAC than P(True)-variant and self-check; P(True) edges it only
  once >20% of questions are rejected (Fig. 3, [p.5]). Exact bar values are not printed in the text.

**5. Terminology (paper's words).** *Confabulation*: "LLMs fluently make claims that are both wrong and arbitrary—by
which we mean that the answer is sensitive to irrelevant details such as random seed" [p.1]. *Semantic entropy*: "an
entropy that is computed over meanings of sentences" [p.1]. *Bidirectional entailment*: "if sentence A entails that
sentence B is true and vice versa, then we consider them to be in the same semantic cluster" [p.2]. *Naive entropy*:
"the entropy of the length-normalized joint probability of the token sequences" [p.3]. *AUROC*: "the probability that a
randomly chosen correct answer has been assigned a higher confidence score than a randomly chosen incorrect answer"
[p.9]. *Rejection accuracy / AURAC* [p.9]. *P(True)* (ref. 24 = Kadavath) [p.9].

**6. Figures.** Fig. 1 [p.2] — two-panel schematic: (a) "Where is the Eiffel Tower?" → six LLM answers → two
bar-chart distributions (naive vs clustered), the canonical picture of why clustering matters; (b) FactualBio
decomposition flow. Clean, small multiples, real answer strings in boxes. **Best figure to adapt** for "what semantic
entropy is" (redraw with a medical example from our own pools, e.g. "pneumothorax / collapsed lung / pneumothorax on
the left" vs "pleural effusion"). Fig. 2 [p.3] — grouped bar chart AUROC/AURAC per model (6 panels). Table 1 [p.4] —
example table with SE vs naive predictions, including a failure case (date vs year clustered apart).

**7. What it gives the deck.** Terminology slide (semantic entropy, confabulation, naive entropy, AUROC) and the
"related but different" slide. Presenter line: *"Semantic entropy is the standard sampling-based uncertainty score: it
tells you how uncertain the model is about a question, but it is one number per question, so it cannot choose between
our eight candidates — its only per-candidate quantity is cluster size, which is a majority vote, and we measured that
the probe beats the vote by 0.09 under the judge."*
**Cuts against us:** (i) their supervised hidden-state probe (embedding regression) *loses* to the unsupervised score
and degrades OOD [p.3, p.9] — the same family as our probe; our defence is pooled in-domain training and the
LOBO/onboarding numbers (DESIGN_EVIDENCE: zero-shot new-benchmark macro +0.0196, ~100 labels to recover most of the
gain), not a claim of OOD robustness. (ii) They report SE *beats* trained probes; we never ran SE-derived selection.
(iii) Their "we significantly improve question-answering accuracy" is via refusing to answer [p.1, p.3] — not our setting.

**8. Must not claim.** That we compared against semantic entropy (never run). That SE "fails" in medicine (that is
UniVRSE/VASE's claim [card], not shown by us). That SE improves accuracy without refusing answers. That our probe beats
SE on AUROC.

---

## 2. Geng et al. 2024 — survey of confidence estimation and calibration

**Citation.** Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, Iryna Gurevych. "A Survey of
Confidence Estimation and Calibration in Large Language Models." arXiv:2311.08298v2 (25 Mar 2024). Published at
NAACL 2024, pp. 6577–6595 (venue from Byun's reference list [Byun p.11]; the PDF itself shows only arXiv).

**1. Problem.** Organise LLM confidence estimation and calibration: the output space is exponential in length, equal
meanings have different strings, models can verbalise confidence and are prompt-sensitive [p.1].

**2. Method (survey; no experiments).** Definitions [p.2]:
- *Relative* confidence (ranking), Eq. (1):
  $$\mathrm{conf}(x_i,\hat y_i)\le \mathrm{conf}(x_j,\hat y_j)\iff P(\hat y_i=y_i\mid x_i)\le P(\hat y_j=y_j\mid x_j)$$
- *Absolute* confidence / calibration, Eq. (2): $P(\hat y = y\mid \mathrm{conf}(x,\hat y)=q)=q$.
- ECE, Eq. (3) [p.3]: $\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{N}\,|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)|$; token-level
  ECE Eq. (4); reliability diagram — "points or lines above the diagonal indicating overconfidence" [p.3] (note: Byun
  plots accuracy on the y-axis, so for Byun overconfidence is *below* the diagonal).
- Taxonomy (Fig. 1 [p.2], Fig. 2 Venn [p.4]): **white-box** = logit-based, internal-state-based, semantics;
  **black-box** = linguistic (verbalized) confidence, consistency-based, surrogate models.
- Internal-state methods described as probes on activations: P(IK) classifier "on the activations of a network",
  trained on sampled answers labelled for correctness [p.4]; Li et al. (ITI) and Azaria & Mitchell "employed linear
  probes" and found "certain middle layers and a few attention heads exhibit strong performance … although the layer
  positions vary across models" [p.4]; CCS unsupervised [p.4–5].
- Post-hoc calibration: temperature scaling "re-weighting the logits before the softmax function by a learned scalar
  τ" [p.18]; "Post-hoc methods … can calibrate probabilities without impacting the model's performance" [p.18].

**3. Setup.** Not applicable (survey).

**4. Key "results" (statements, not numbers).** Logit methods: "low logit probabilities may reflect various properties
of language" [p.5]; internal-state methods "typically require supervised training on specially annotated data"; probes
have limits generalising to negations (Levinstein & Herrmann) [p.5]; consistency methods "computationally intensive
but have proven effective" [p.6]. Future direction: "it remains unclear whether these confidence estimation methods are
effective for multimodal large language models" [p.9].

**5. Terminology.** White-box ("the state at every position of the LLMs is accessible during inference") / black-box
("allowing access only to the generations") [p.3, p.5]; logit-based; internal state-based; linguistic confidence
(verbalized); consistency-based estimation; surrogate model; relative vs absolute confidence; calibration; ECE, ACE,
reliability diagram, AUROC, AUARC [p.2–3]; aleatoric vs epistemic uncertainty [p.2].

**6. Figures.** Fig. 1 [p.2] taxonomy tree (text boxes; not worth recreating as-is). **Fig. 2 [p.4] two Venn diagrams
(white-box: logit / internal state / semantics; black-box: consistency / linguistic / surrogate)** — a good template for
a terminology slide placing our probe in the "internal state" circle and placing majority vote / self-consistency in
"consistency".

**7. What it gives the deck.** The vocabulary and the map. Presenter line: *"In the standard taxonomy our probe is a
white-box, internal-state method; self-consistency and semantic entropy are consistency/semantic methods; verbalized
confidence is black-box. Selection only needs the survey's 'relative' confidence — ranking — not calibration."*
**Cuts against us:** none directly; it does note probes need labelled data and generalise imperfectly [p.5].

**8. Must not claim.** That the survey covers VLMs (it lists multimodal as an open direction [p.9]); that it gives
numbers; that it endorses best-of-N selection (it does not discuss it).

---

## 3. Byun et al. 2026 — overconfidence and calibration in medical VQA

**Citation.** Ji Young Byun, Young-Jin Park, Jean-Philippe Corbeil, Asma Ben Abacha. "Overconfidence and Calibration in
Medical VQA: Empirical Findings and Hallucination-Aware Mitigation." arXiv:2604.02543v1 (2 Apr 2026), preprint under
review.

**1. Problem.** Is the confidence of general VLMs on medical VQA trustworthy, does scale or prompting fix it, and what
should be the baseline practice? [p.1–2].

**2. Method.**
- Sampling-based confidence (primary) [p.17]: draw $N$ generations $\{a_1..a_N\}\sim P(\cdot\mid q,I)$,
  $$c_{\text{samp}}(q)=\frac1N\sum_{i=1}^{N}\mathbb{1}[\mathrm{parse}(a_i)=\hat a],\quad \hat a=\arg\max_a\sum_i \mathbb{1}[\mathrm{parse}(a_i)=a]$$
  i.e. the **plurality answer's vote share**; for open questions parse() clusters semantically equivalent answers with
  an LLM judge (Qwen3-4B-Instruct-2507) instead of DeBERTa [p.17]. **N = 20** [p.3].
- Verbalized: Vanilla, Vanilla+CoT, Two-Stage, Punish, Top-K, Linguistic (mapping table Table 6) [p.17–20].
- Logit-based (yes/no log-sum-exp, Eq. 3 [p.18]) defined but **omitted** because it is limited to closed form [p.18].
- Post-hoc: Platt scaling, isotonic regression, histogram binning, 5-fold CV [p.6].
- **HAC** [p.7]: hallucination score $h$ = VASE (10 generations on the original + 10 on a weakly perturbed image,
  entropy of the contrastive semantic distribution). HAC-Platt Eq. (1): $s(c,h)=\sigma(a\,c+b\,h+d)$; HAC-Gate Eq. (2):
  $s(c,h)=c\cdot\sigma(-\alpha h+\beta)$. Fitted by NLL on held-out folds.
- Nothing in the VLM is trained; only 2–3 scalar parameters.

**3. Setup.** Models: Qwen3-VL-Instruct 2B/8B/32B, InternVL3-Instruct 2B/8B/38B, LLaVA-NeXT 7B/34B [p.3]. Data: test
splits of VQA-RAD (451), SLAKE-EN (1,061), VQA-Med-2019 (500), pooled 2,012 = 1,342 open + 670 closed; "closed" =
yes/no only (Table 5, [p.17]). Judge: Qwen3-4B-Instruct (LLM-as-judge) [p.20]. Metrics: ECE, **ACE (primary)**,
AUROC [p.4]. Baselines: raw sampling/verbalized confidence, standard post-hoc methods.

**4. Key results.**
- Table 1 [p.4], pooled, Acc / Conf / Gap: Qwen3-VL-2B .562/.860/.298; -8B .605/.943/.338; -32B .651/.954/.304;
  InternVL3-2B .554/.774/.220; -8B .618/.811/.192; -38B .549/.767/.218; LLaVA-NeXT-7B .434/.640/.206; -34B .472/.751/.279.
  "every point falls below the diagonal" (Fig. 1) [p.5].
- Open-ended gap larger than closed (Table 7 [p.25], e.g. Qwen3-VL-8B open gap 0.400 vs closed 0.214).
- Scaling: 8 of 15 within-family comparisons improved ACE and 7 worsened, mean ΔACE = +0.012 [p.5].
- Post-hoc (7–8B average): verbalized ECE 0.243 → 0.038 (Platt) / 0.029 (isotonic) closed; 0.419 → 0.042 / 0.035
  open [p.6]; best prompting (Top-K) only reaches ECE 0.356 open [p.6].
- AUROC unchanged by Platt (Table 2 [p.7]); HAC +5.3 pp AUROC on average over 8 models, +7.3 pp open, +10.1 pp
  open-verbalized (Table 2 caption [p.7]). Example open-ended sampling AUROC uncalibrated → HAC-Platt: Qwen3-VL-8B
  0.643 → 0.750; InternVL3-8B 0.764 → 0.789; LLaVA-NeXT-7B 0.706 → 0.725 [p.7].
- VASE vs correctness Pearson r = −0.48 (sampling) / −0.46 (verbalized) on open, −0.20 / −0.16 closed [p.7–8].
- Sample-size ablation: "N ≥ 10–20 samples suffice for stable calibration estimates" [p.22].
- Cost (Table 4 [p.9]): sampling base open 7.2 ± 52.1 tokens/gen, total 72.0 at N=10; VASE 20 generations.

**5. Terminology.** Overconfidence gap = Conf − Acc [p.4]; ECE; ACE ("population-uniform bins") [p.4]; Platt scaling
("a logistic regression model in a strictly monotonic manner (i.e., preserving their relative ordering)") [p.6];
isotonic regression; histogram binning; sampling-based / verbalized / logit-based confidence [p.2–3, p.17];
hallucination-aware calibration; discriminative quality = AUROC [p.2]. They state: "No matter how well the scores are
recalibrated, a confidently incorrect prediction will always be ranked above a less confident but correct one" [p.2].

**6. Figures.** Fig. 1 [p.4] reliability-style scatter (mean confidence on x, accuracy on y, diagonal, grey
overconfident region, colours = model, markers = dataset, closed/open panels) — **recreate for the "medical VLMs are
overconfident" background slide**, values from Table 1/7. Fig. 3 [p.6] grouped bars before/after Platt. Table 2 [p.7]
AUROC table — the cleanest evidence that monotone calibration cannot reorder.

**7. What it gives the deck.** Background (overconfidence is pervasive, worse on open-ended), and the argument that
*reordering* needs a signal complementary to confidence — which is what any best-of-N verifier must supply.
Presenter line: *"On the very benchmarks we use — VQA-RAD, SLAKE, VQA-Med-2019 — general VLMs report 0.6–0.95
confidence while being 0.43–0.65 accurate; recalibration fixes the numbers but cannot change which answer ranks first."*
**Cuts against us / caveats:** (i) their sampling confidence already *is* our vote count, and HAC shows that adding a
vision-grounded signal improves ranking — an untested alternative to (or feature for) our probe; (ii) their AUROCs are
question-level error detection, not selection, and use a 4B judge — not comparable to anything we report; (iii) they
evaluate general VLMs, not medical fine-tunes like Lingshu ("domain-specific fine-tuned models … could be further
explored", Limitations [p.10]).

**8. Must not claim.** That Lingshu is overconfident (not measured by them or, to my knowledge, by us); that HAC
selects answers (it rescores one answer's confidence); that their AUROCs compare to our accuracy deltas; that VASE was
tested by us.

---

## 4. Khanmohammadi et al. 2026 — confidence estimators for medical LVLMs

**Citation.** Reza Khanmohammadi, Kundan Thind, Mohammad M. Ghassemi. "Calibrated Triage, Not Autonomy: Confidence
Estimation for Medical Vision-Language Models." arXiv:2606.15910v2 (20 Aug 2026).

**1. Problem.** Which confidence estimator, trained on ordinary images, still works on medical scans, and how much
workload can be automated at a fixed error tolerance (framed as selective prediction / deferral — out of scope for us;
we use only the estimator benchmark) [p.1–2].

**2. Method.** Nine estimators in three families [p.3]: inference-only (P(True), Self-Probing, Prompt Ensemble);
training-free logit (sequence log-likelihood, predictive entropy — "free at inference", truncated-softmax scale, so no
ECE); trained probes (SAPLMA = Azaria & Mitchell, reads the post-answer state; CCPS = perturbed-representation
stability; InternalInspector = pools attention, FFN and residual states through a learned encoder; BICR = contrastive
real-vs-blank-image ranking) [p.2–3]. **All probes trained only on GQA** (20,000 train, 5,000 val), 5 seeds, scores
seed-averaged [p.3]. Generation: greedy, 64 new tokens, one answer per question [p.3]. AUROC defined as
$\Pr(s_i>s_j\mid y_i=1,y_j=0)$, ties at ½ [p.9]; ECE 10 equal-width bins [p.10]; selective metrics AURC, $Y@\tau$
[p.10] (deferral-side; not for the deck).

**3. Setup.** LVLMs: Qwen3-VL-8B, LLaVA-NeXT-13B, InternVL3.5-14B, DeepSeek-VL2, Gemma-3-27B [p.3]. Data:
GMAI-MMBench (multiple-choice, n = 20,665–21,748 shared), SLAKE (n = 2,094, both languages), PATH-VQA (n = 6,111) [p.3–4;
GMAI multiple-choice-only per Supplementary S3-C]. Judge: gpt-5-mini sees image, question, gold, response; 96.5% agreement with a blind human on a free-form
sample (κ = 0.930) [p.3]. Base accuracy pooled over LVLMs: SLAKE 60.9%, GMAI 47.2%, PATH-VQA 39.5% [p.3].

**4. Key results.**
- Table I [p.4] AUROC (pooled over 5 LVLMs): SLAKE — SeqLogLik 65.3, PredEntropy 65.3, P(True) 64.3, InternalInspector
  67.3, **BICR 68.6**, CCPS 54.1; PATH-VQA — **PredEntropy 65.1**, SeqLogLik 64.9, Prompt Ensemble 64.0, InternalInspector
  61.4, BICR 58.9, SAPLMA 57.6. Training-free logit baselines are "competitive with the trained probes on the rank metrics".
- Inference-only estimators ECE 0.31–0.47 on GMAI and PATH-VQA [p.4].
- No estimator reliably best: across 15 (LVLM, dataset) cells no method wins more than 4 on AUROC; Friedman χ² = 17.78,
  p = 0.023; Wilcoxon vs BICR (Holm) significant for none, smallest adjusted p = 0.28 [p.4].
- "absolute discrimination is modest throughout (AUROC near 0.63 even for the best)" [p.7].
- **In-domain retraining (Table S18 [p.30], text [p.23])**: same BICR architecture trained on the SLAKE calibration half
  (1,047 labelled cases per LVLM) instead of GQA: AUROC **0.695 → 0.864**, paired Δ +0.170 [+0.153, +0.184]. "The signal
  needed to rank a medical model's correctness is present in the hidden state and legible to the probe" [p.23].
- No estimator reliably lowers confidence when the image was ignored (image-replacement test) [p.6].
- Cost (S2-G [p.21–22]): logit baselines add "well under a millisecond"; trained probes add "a single forward pass over
  stored states", 177–625 ms vs generation 248–2,121 ms; P(True)/Self-Probing ~ a second forward pass.

**5. Terminology.** Calibrated vs discriminative ("A score is calibrated when its value matches empirical accuracy …
it is discriminative when higher scores are reliably more often correct than lower ones, regardless of the absolute
scale") [p.2]; internal-state probes; training-free logit baselines; rank-invariance [p.11]; question-clustered
bootstrap [p.3]; top-decile error [p.10].

**6. Figures.** Table I [p.4] is the most usable artifact (estimator × metric × dataset). Table S18 [p.30] (OOD vs
in-domain probe) is a one-row-pair table worth quoting verbatim.

**7. What it gives the deck.** Comparison/background: *the closest benchmark of hidden-state probes on medical VQA.*
Presenter line: *"The one systematic comparison of confidence estimators on medical VLMs finds that probes trained on
natural images are no better than free log-likelihood — but the same probe retrained on in-domain SLAKE labels jumps
from 0.70 to 0.86 AUROC. That is the regime we are in: our probes are trained on labels from the same eight benchmarks."*
**Cuts against us:** (i) a free logit baseline (sequence log-likelihood / predictive entropy) matched the trained probes
out of domain — and we never ran mean log-prob as a selector; (ii) "no estimator reliably best across domains or
models" → our probe may not be best on other backbones (only Lingshu and Qwen2.5-VL replicated); (iii) no estimator
detects image bypass — we have no test of whether our probe does; (iv) the in-domain gain needs ~1,000 labels per
model, comparable to our labelling cost.

**8. Must not claim.** Anything about deferral/triage/automation yields as relevant to our method. That their AUROC
numbers are comparable to our selection gains (different task, greedy single answer, different judge). That their
in-domain result transfers beyond SLAKE (they warn it is "specific to SLAKE" [p.23]). That "defers a quarter of
radiology" means cases sent to a clinician — in their own definition $Y@\tau$ is the fraction *automated* (S1-B
[p.10]); the abstract's "defers" wording is ambiguous.

---

# Part 2 — All 32 cards in category F

| id | title (short) | one line | relevance | why |
|---|---|---|---|---|
| farquhar2024semanticentropy | Semantic entropy (Nature) | SE over entailment clusters; AUROC 0.790 vs 0.687–0.698 | **slide-worthy** | defines SE; explains why SE ≠ selection |
| khanmohammadi2026triage | Calibrated Triage | 9 estimators × 5 LVLMs × 3 medical sets; probes ~ logit baselines OOD; in-domain probe 0.695→0.864 | **slide-worthy** | closest probe benchmark in medical VQA |
| byun2026overconfidence | Overconfidence in Med VQA | general VLMs overconfident on our benchmarks; Platt can't reorder; HAC +5.3 pp AUROC | **slide-worthy** | background overconfidence + monotonicity argument |
| geng2023survey | Confidence survey | white-/black-box taxonomy, ECE, relative vs absolute | **slide-worthy** (terminology) | vocabulary and map |
| kossen2024sep | Semantic Entropy Probes | probe on one generation's hidden states predicts SE | **slide-worthy / comparison table** | nearest mechanism relative; different target |
| kadavath2022knowwhatknow | LMs (mostly) know | P(True), P(IK) probe | mention | names P(True); our July self-verify is P(True)-type |
| azaria2023lying | Internal state knows when lying (SAPLMA) | classifier on activations, 71–83% | mention | earliest correctness probe; = SAPLMA |
| asadi2026cebag | CEBaG | deterministic logprob + image-evidence; best in 13/16, +8 AUC over VASE | mention | strongest untested training-free competitor |
| guo2017calibration | On calibration | temperature scaling | mention (terminology) | name clash with sampling T |
| kuhn2023semanticuncertainty | Semantic uncertainty (ICLR) | SE predecessor | mention | priority citation |
| burns2022ccs | CCS | unsupervised truth direction | mention | unsupervised end of probe spectrum |
| chen2024inside | INSIDE / EigenScore | covariance of sampled embeddings | mention | internal-state consistency hybrid |
| xiong2023express | Can LLMs express uncertainty | verbalized overconfident; white-box narrowly better | mention | verbalized baseline family |
| gautam2025hedge | HEDGE | VASE robust; Qwen2.5-VL most detectable; VQA-RAD, Kvasir-VQA-x1 | mention | dataset/model overlap with ours |
| wienholt2025dse | DSE in radiology | discrete SE on GPT-4o/4.1, incl. VQA-Med 2019 | mention (signal only) | dataset overlap; filtering = out of scope |
| liao2025univrse | UniVRSE / VASE | vision-conditioned SE for medical VLMs | mention | the VASE used by Byun |
| fazla2026safetynet | UE not a safety net | UE quality tracks accuracy | mention | "signal degrades where model weakest" |
| senoglu2026verbalized | Verbalized calibration Med VQA | fine-tune for calibrated verbalized confidence | mention | alternative that changes the generator |
| tian2023justask | Just ask for calibration | verbalized better calibrated than logprobs in RLHF models | skip (or footnote) | text-only, calibration not ranking |
| li2023iti | ITI | intervene along truth directions | skip | intervention, not scoring |
| liu2024hallucinationsurvey | LVLM hallucination survey | taxonomy | skip | not needed |
| kostumov2024uncertaintyaware | Uncertainty-aware VLM eval | conformal on MCQ | skip | MCQ, conformal |
| xia2024cares | CARES | trustworthiness benchmark | skip | other axes |
| chen2024medhallmark | Med-HallMark | hallucination benchmark | skip | not our metric |
| zhang2024vluncertainty | VL-Uncertainty | perturbation SE for LVLMs | skip | covered by VASE/HEDGE |
| chen2026vihd | VIHD | layer-targeted visual masking SE | skip (optional footnote on layer choice) | MICCAI; abstract-only |
| zhou2026coev | CoEV | counter-evidence verification | skip | report generation focus |
| liang2025expertcfg | Expert-CFG | expert-in-the-loop | skip | escalation to human (rule 6) |
| song2026countervhd | Counterfactual grounding | external grounder | skip | not our setting |
| li2023pope | POPE | object hallucination yes/no | skip | not our setting |
| chan2026yesbias | Yes-bias attention | system-prompt attention | skip | judge-confound idea only |
| basu2025calibprompt | CalibPrompt | prompt-tuning calibration | skip | classification calibration |

### Verified content for slide-worthy / mention cards

**kossen2024sep** (arXiv 2406.15927, preprint; first three authors equal). [abstract] SEPs approximate semantic entropy
from the hidden states of a *single* generation, avoiding the multi-sample cost; they "retain high performance for
hallucination detection" and generalise OOD better than probes that predict accuracy directly; ablations over token
positions and layers. No numbers in the abstract. Deck use: comparison-table row — *same mechanism (MLP/linear probe on
hidden states), different target (SE vs our correctness label), different use (question-level detection vs within-
question ranking)*. Its OOD claim cuts against accuracy-target probes like ours (not tested by us).

**kadavath2022knowwhatknow** (arXiv 2207.05221 v4). [abstract] P(True): the model estimates the probability that a
proposed answer is correct; "improves when models consider many of their own samples"; P(IK) trained to predict
knowing the answer, partly generalises, poorly calibrated on new tasks. Farquhar's P(True) baseline: few-shot, model
sees all M samples then judges the top answer; AUROC 0.698 [Farquhar p.3, p.9]. Our July `p_yes_norm` self-verify is
a zero-shot single-pass P(True)-type score (`src/cascade_methods/baseline_compare.py` line 12–13).

**azaria2023lying** (arXiv 2304.13734 v2; Findings of EMNLP 2023 per Khanmohammadi ref [18]). [abstract] Classifier on
hidden activations predicts truthfulness, 71%–83% accuracy depending on base model; sentence probability confounded by
length and word frequency. Called **SAPLMA** in Khanmohammadi; there it reads the post-answer state and reaches AUROC
62.5 (SLAKE), 57.6 (PATH-VQA) out of domain [Khanmohammadi p.4].

**asadi2026cebag** (arXiv 2603.21693 v2, 17 Jul 2026). [abstract] Deterministic; combines token-level predictive
variance with how much the image shifts predictions vs text-only; SE and VASE need 10–20 generations plus an NLI model;
4 medical MLLMs × 3 benchmarks = 16 settings; highest AUC in 13 of 16, ~8 AUC points over VASE. Deck use: the
untested training-free competitor in the honest-gaps slide (a per-candidate score is computable from it in principle —
inferred).

**guo2017calibration** [card, abstract-only] Temperature scaling = one learned scalar dividing logits before softmax.
**Terminology warning for the deck:** "temperature" in calibration (post-hoc logit rescaling) is unrelated to our
sampling temperature T = 0.7; say "sampling temperature" every time.

**kuhn2023semanticuncertainty** [card] ICLR 2023 predecessor of Farquhar; cite for priority.

**burns2022ccs** [card] Unsupervised probe: statement and its negation should get probabilities summing to 1;
"outperforms zero-shot accuracy by 4% on average" over 6 models × 10 datasets [abstract via card, not re-verified].

**chen2024inside** [card] EigenScore = eigenvalues of the covariance of internal-state embeddings across sampled
responses; ICLR 2024. Question-level, like SE.

**xiong2023express** [card] Verbalized confidence overconfident; white-box only narrowly better than black-box,
"0.522 to 0.605 in AUROC" [abstract via card, not re-verified]; ICLR 2024 (Byun ref list [p.14] confirms ICLR).

**gautam2025hedge** (arXiv 2511.12693). [abstract] Pipeline: image perturbation + entailment- or embedding-based
clustering + uncertainty metrics; VQA-RAD and KvasirVQA-x1; LLaVA-Med, Med-Gemma, Qwen2.5-VL; detectability highest for
dense visual tokenization (Qwen2.5-VL), lowest for restricted (Med-Gemma); VASE most robust with embedding clustering and
~10–15 samples; library hedge-bench. Byun used hedge-bench for VASE because "The original VASE implementation … is not
publicly available" [Byun p.21].

**wienholt2025dse** (arXiv 2510.09256 v2; Eur Radiol 2026). [abstract] Discrete SE on GPT-4o / GPT-4.1, 706
image–question pairs from two datasets; filtering high-entropy questions raises accuracy ~52%→76% (GPT-4o) and
~55%→64% (GPT-4.1). The card's VQA-Med 2019 (500) + 206 radiology cases detail is [card]. Use only as "DSE has been
applied to radiology VQA"; the filtering is rejection (rule 6).

**liao2025univrse** [card] Vision-conditioned SE contrasting original vs distorted image. Byun cites the VASE method as
Liao et al., "Vision-amplified semantic entropy for hallucination detection in medical visual question answering",
MICCAI 2025, pp. 669–679 [Byun p.12]. Whether arXiv 2503.20504 (UniVRSE) is the same work renamed is **not verified**.

**fazla2026safetynet** [card] 8 UE methods × 12 clinical VLMs (MCQ); UE quality "tracks model accuracy"; NOTA
perturbation collapses accuracy while uncertainty barely moves. Agrees with Khanmohammadi's capability-tracks-signal
finding (r = 0.74 between base accuracy and confidence AUROC within GMAI [Khanmohammadi p.5]).

**senoglu2026verbalized** [card] Fine-tunes MedGemma-4B-IT / Qwen2-VL-7B with a composite calibration loss; ">60%"
lower calibration error, ">26%" better discrimination. Contrast: changes generator weights; our generator is frozen.

---

# Part 3 — Recommendations

**Own slide (1):** "Uncertainty estimation vs best-of-N selection" — the §0 table, built on Farquhar (SE is per
question), Byun (monotone calibration cannot reorder; vote share = sampling confidence), Geng (relative vs absolute
confidence). Include a small schematic: 8 candidates of one question → SE gives one number for the question; the probe
gives 8 numbers, one per candidate.

**Background slide (1):** "Medical VLMs are overconfident" — recreate Byun Fig. 1 from Table 1 (pooled) or Table 7
(open-ended, closer to our setting); add Khanmohammadi's ECE 0.31–0.47 and "AUROC near 0.63" as the second source.

**Comparison table rows (related-work / "where our probe sits"):** Farquhar embedding regression (logistic, final hidden
state, correctness, question-level, fails OOD); Kadavath P(IK); Azaria SAPLMA; Kossen SEP (target = SE);
Khanmohammadi BICR/InternalInspector (medical, OOD-trained; in-domain 0.864); ours (MLP, mean-pooled answer tokens at
layers 18/20/22, correctness label, per candidate, used for within-question ranking).
Columns: input states · target · trained on · unit scored · used for · medical? · reported metric.

**Terminology slide must define:** confidence vs uncertainty; calibration (ECE, ACE, reliability diagram) vs
discrimination (AUROC); relative vs absolute confidence; white-box vs black-box; logit-based (sequence log-likelihood,
predictive entropy, mean token log-prob ≈ log-perplexity); verbalized confidence; P(True)/P(IK); self-consistency /
majority vote; semantic entropy (and discrete SE); probe / internal-state method; Platt scaling, temperature scaling
(⚠ not the sampling temperature); hallucination vs confabulation; overconfidence gap.

**Honest-gaps slide:** list the untested selectors exactly as in §0 — mean log-prob/perplexity (uncomputable from
stored dumps; harvestable in an extraction pass, `AUDIT_2026-09-18.md` line 230), semantic-cluster vote / SE, verbalized
confidence, P(True) on current pools (July-era zero-shot version failed: 0.715 vs random 0.720, N32), CEBaG/VASE-style
image-contrast scores, SEP-style entropy-target probe; plus "probe calibration (ECE) and image-bypass sensitivity not
measured" (not found in the docs I read — **verify before stating**). Khanmohammadi is the reason this gap matters: free
log-likelihood matched trained probes there.

**Errors / gaps found in the cards and existing briefs.**
1. `DESIGN_EVIDENCE.md` row "mean log-prob / perplexity / P(True) as selector — not tested": **partly wrong.** A
   P(True)-type zero-shot self-verification selector (`p_yes_norm`) was tested in the July era: "below greedy (0.319 vs
   0.343) and below random (0.715 vs 0.720)" (`PROJECT_RETROSPECTIVE_2026-07-29.md`, negatives table row N32, and
   §turning point ~line 1496). Reword to "not tested on the final pools; an earlier zero-shot self-verification selector
   scored below random". Log-prob/perplexity remain untested.
2. `khanmohammadi2026triage` card: models and datasets "not extracted" → Qwen3-VL-8B, LLaVA-NeXT-13B, InternVL3.5-14B,
   DeepSeek-VL2, Gemma-3-27B; GMAI-MMBench, SLAKE, PATH-VQA; probes trained on GQA only [p.3]. The card omits the
   in-domain result (Table S18, AUROC 0.695→0.864), which is the most relevant finding for us. The card's "defers about
   a quarter" follows the abstract's wording, but $Y@\tau$ is the fraction *automated* [p.10].
3. `byun2026overconfidence` card: datasets "not extracted" → VQA-RAD, SLAKE-EN, VQA-Med-2019 (pooled 2,012) [p.3,
   p.17]; "No exact ECE/AUROC point values recoverable" → Table 1 [p.4], Table 2 [p.7], Table 3 [p.8], Table 7 [p.25]
   give them. Note judge = Qwen3-4B-Instruct [p.20].
4. `geng2023survey` card venue "arXiv preprint" → NAACL 2024 (per Byun ref list [p.11]).
5. `azaria2023lying` card venue "arXiv preprint" → Findings of EMNLP 2023 (Khanmohammadi ref [18]); it is the SAPLMA
   method.
6. `farquhar2024semanticentropy` card: numbers now verified from the PDF [p.3–4]; it omits that the strongest
   supervised baseline is a **hidden-state logistic probe that degrades OOD** [p.3, p.9] and that N = 10 at T = 1 [p.4,
   p.7]. Its "relation_to_us" says SE gives "a self-consistency-at-N=8 baseline" — more precisely, SE itself cannot
   select; the selection analogue is the semantic-cluster vote (inferred).
7. `kossen2024sep` card says SEPs use "no extra forward pass"; the abstract says single-generation hidden states and
   negligible cost — consistent, but the card's long parenthetical about our capture-inside-generation status should
   stay out of any slide (the shipped probe's features come from a separate teacher-forced pass, DECK_SPEC).
8. `liao2025univrse` vs Byun's "VASE (Liao et al., MICCAI 2025)": possibly the same group's work under two names;
   unverified — cite VASE via Byun's reference, not via the UniVRSE card.
9. `FIGURE_STYLE_NOTES.md` line 169 cites "Farquhar Nature Fig. 1 p.2" — correct (PDF p.2 = journal p.626).
10. Reliability-diagram direction differs between sources (Geng: above diagonal = overconfident with confidence on y;
    Byun Fig. 1: below diagonal = overconfident with accuracy on y). Pick one convention on the slide and label axes.
