# PAPER_NOTES_I — Group I: n-best rescoring, discriminative reranking, confidence models in other fields
# (+ gap-fill: LiLaVe, ReProbe, CASE, DualRead, Hewitt & Liang control tasks)

Written 2026-10-19 under `PAPER_READ_BRIEF.md`.

**Sources and tags.**
- The four core PDFs were read in full from the extracted text in `tmp/txt/I_*.txt`. The R-EBM and CEM figure
  pages were rendered to `tmp/deck/figI/*.png` and checked by eye.
- Gap-fill: the arXiv abstract page and the arXiv HTML full text were downloaded for all five papers
  (`tmp/arxivI/abs_*.html`, `html_*.html`, converted to `.txt`). Versions: LiLaVe v1, ReProbe v5, CASE v1,
  DualRead v1, Hewitt & Liang v1.
- Tags: **[PDF p.N, §/Table/Fig]** means the core PDF. **[abstract]** means the arXiv abstract. **[HTML §/Table]**
  means the arXiv HTML full text. **[card]** means `cards_I.json`, not re-verified. **(inferred)** means my
  inference.
- Every number below is copied verbatim from the tagged source.

Our method, used as the yardstick for relevance:
- Lingshu-7B samples N = 8 answers per question at T = 0.7.
- Each distinct answer is re-read with teacher forcing. Hidden states at layers 18, 20 and 22 are mean-pooled over
  the answer tokens.
- 24 MLP probes (3584→256→GELU→1, trained with BCE; 3 layers × 8 seeds) score each answer.
- The probe ranks are averaged, ties are broken by vote count, and the top answer is returned.

---

## Part 1 — Core PDFs, read in full

### 1.1 Li, Zhang, Li, Cao, Woodland — "Residual Energy-Based Models for End-to-End Speech Recognition"
arXiv:2103.14152 (v2, 23 Jun 2021), Interspeech 2021. Cambridge + Google.

**1. Problem.**
- An autoregressive ASR decoder scores a transcript as a product of per-token probabilities. These are "locally
  normalised" models.
- Such scores are sub-optimal for three reasons:
  - exposure bias: the model is trained on true histories but decodes on its own;
  - search errors: only a few hypotheses survive the beam;
  - training per token does not minimise word error rate.
- So the model's sequence probability does not match the real data distribution.
- The paper trains a second, separate model that re-scores the beam's n-best list. The same model doubles as an
  utterance-level confidence estimator.

**2. Method, precisely.**
- *Joint model* [PDF p.2, Eq. 2–3]:
  $P_\theta(y\mid X)=\dfrac{P(y\mid X)\exp(-E_\theta(X,y))}{Z_\theta(X)}$, with
  $Z_\theta(X)=\sum_y P(y\mid X)\exp(-E_\theta(X,y))$.
  $E_\theta$ is the "residual energy" learned on top of the fixed ASR model $P$.
- *Architecture* [PDF p.2 §2.1; p.3 §4.2]:
  - The R-EBM is a sequence model. At each output step it reads four features: the current decoder hidden state,
    the attention context vector, the output token embedding, and the top-K softmax probabilities.
  - "the hidden representations for each output token are gathered by **mean pooling** before being passed to the
    output layer that uses sigmoid activation function."
  - The final configuration is "two layers of LSTMs with 512 units in each direction", bi-directional.
- *Trained vs frozen*: "The baseline auto-regressive model is fixed during R-EBM training" [Fig. 1 caption, p.2;
  §3.2 p.3].
- *Loss* [p.2 Eq. 4–5]:
  - It is conditional noise-contrastive estimation, with the ASR model as the noise distribution. This reduces to
    binary cross-entropy:
    $\mathcal{L}(X;\theta)\approx\frac{1}{|Y^+|}\sum_{y\in Y^+}\log\frac{1}{1+\exp(E_\theta(X,y))}+\frac{1}{|Y^-|}\sum_{y\in Y^-}\log\frac{1}{1+\exp(-E_\theta(X,y))}$.
  - $Y^+\cup Y^-=\{y_{\text{ground truth}},\textsc{BeamSearch}(X,n)\}$. **The positives include the reference
    transcript itself**, not only correct hypotheses.
  - "there may be more than one element in $Y^+$" when the ground truth is in the n-best list.
- *Training data* [p.3 §3.2]:
  - The n-best lists for training are generated on the fly with heavier SpecAugment: 10 time masks of up to
    50 frames, versus 2 of up to 40 for the baseline.
  - The stated aim is "to simulate the errors made by the model during inference"; the WER on the augmented
    training set "should ideally match that of the dev set".
- *Inference* [p.2 §2.3; p.3 Eq. 7]:
  - With a shared partition function, re-rank the n-best by
    $\hat y=\arg\max_y \frac{\log P(y\mid X)}{|y|}-\alpha E_\theta(X,y)$.
  - This uses length-normalised log-likelihood plus the energy. α is tuned on dev.
  - Pre-sigmoid values are used for reranking; post-sigmoid values serve as confidence.
- *N*: beam size n ∈ {4, 8, 16, 32}; 32 is the final choice [Table 3].
- *Temperature*: not applicable. Candidates come from beam search, not sampling.

**3. Setup** [p.2–3 §3]:
- Training data: LibriSpeech train-clean-100, "28.5k utterances", 12.7 s average.
- Test data: dev/test clean/other, each over 5 h.
- Units: 1,024 word-pieces.
- Base model: LAS-style encoder-decoder, 145 M parameters, with shallow-fusion LSTM language model.
- Metrics: WER, and precision-recall AUC of confidence.
- Baselines: 1-best, and CEM token scores averaged to utterance level [Table 4].

**4. Key results.**

| Result | Value | Source |
|---|---|---|
| test-clean / test-other WER, baseline → +R-EBM | 5.61 → 5.15 / 18.68 → 17.42. The abstract's "8.2%/6.7%" relative reductions. | [Table 4, p.4] |
| P-R AUC of confidence, baseline → +R-EBM | 0.684 → 0.770 / 0.529 → 0.679. The abstract's "12.6%/28.4%". | [Table 4, p.4] |
| P-R AUC with CEM averaged per utterance | 0.697 / **0.501**. Utterance-level confidence from averaged token scores can underperform the baseline. | [Table 4, p.4; text p.4] |
| On wav2vec 2.0: WER and AUC | WER 2.63 → 2.49 / 4.74 → 4.53; AUC 0.786 → 0.928 / 0.684 → 0.890 | [Table 5, p.4] |
| Oracle WER vs n (dev-clean) | 4.19 (n=4) → 2.68 (n=32). 1-best stays 5.41 → 5.23; joint 5.30 → 4.80. WERR grows 2.0 → 8.2 %. | [Table 3, p.3] |
| **R-EBM score alone vs 1-best** | **Worse at small n** on dev-clean: 5.58 vs 5.41 (n=4) and 5.33 vs 5.28 (n=8). Joint with log-likelihood is better. | [Table 3] |
| Uni- vs bi-directional R-EBM, joint WER | 5.09/17.77 vs 5.07/17.69 | [Table 2] |

**5. Terminology.**
- **Residual energy-based model.** The model "learns 'residual' errors of the auto-regressive model to better match
  the data distribution" [p.2].
- **Locally vs globally normalised.**
- **Exposure bias.**
- **n-best hypotheses / re-ranking.**
- **Utterance-level confidence.**
- **Length normalisation (LN).**
- **WERR**: relative WER reduction.
- **Oracle WER**: "The performance ceiling of the joint model is the oracle WER of the n-best hypotheses" [p.3].
  This is the ASR name for our oracle@N.

**6. Figures.**
- **Fig. 1 (p.2): the schematic.**
  - Light-blue Encoder and Decoder boxes. The trained "Residual EBM" in salmon.
  - Circled operators: ⊗, exp(·), sigmoid(·).
  - Outputs in math: $P_\theta(y|X)$, Confidence(y), $-E_\theta(X,y)$.
  - A plain-text bullet list of the four input features sits beside the boxes.
  - Already described correctly in `FIGURE_STYLE_NOTES.md` §1.8.
- **Fig. 2 (p.4): two bar-like line plots.** Joint WERR and oracle WERR (%) against reference length (1–8 … >40).
  Shows the gain tracks the oracle headroom except for short utterances.
- **Fig. 3 (p.4): density plots of log P(y|X).** Baseline 1-best vs data (left); joint score vs data (right).

**7. What it gives the deck.**
- Slides:
  - background/lineage: "this recipe predates us";
  - terminology: n-best, rescoring, oracle WER;
  - method justification: mean-pool → sigmoid, frozen base, BCE.
- Presenter sentence: "In 2021 speech recognition already trained a small model on a frozen recogniser's decoder
  states. It mean-pools them over each hypothesis's tokens and is trained with binary cross-entropy to rerank the
  n-best list. Our probe is the same object for a medical VLM's sampled answers."
- **Cuts against us:**
  - R-EBM's score alone is **worse than 1-best** at n = 4 and 8 on dev-clean [Table 3]. Its gains need the joint
    score with the generator's own log-likelihood [Eq. 7].
  - We never combine the probe with the generator's likelihood. DESIGN_EVIDENCE.md line 53 says log-prob was
    "not tested — uncomputable: dumps store one scalar per question". This is a real untested design axis that
    both classical rerankers (R-EBM, Collins) consider essential.
  - The averaged-token CEM utterance score underperforms the baseline on test-other (0.501 vs 0.529). Averaging
    token-level *scores* differs from training on a sequence-level target. We train at the sequence level on a
    pooled state, which is the R-EBM choice, not the CEM-average choice. (inferred)
- **Gains grow with n** [Table 3]. This is consistent with a reranker that converts headroom. It is a useful
  contrast to our N = 8.

**8. Claims we must not make.**
- That R-EBM reads *only* hidden states. It also reads the attention context, token embeddings and top-K softmax
  probabilities.
- That it was trained only on the model's own samples. The positives include the ground-truth transcript, and the
  candidates come from beam search, not temperature sampling.
- That its "AUC" is comparable to our AUROC. It is the area under the **precision-recall** curve [p.3 §4.4].
- That it is a "probe" in the interpretability sense. It is a 2-layer BiLSTM sequence model.

---

### 1.2 Li, Qiu, Zhang, Li, He, Woodland, Cao, Strohman — "Confidence Estimation for Attention-based Sequence-to-sequence Models for Speech Recognition"
arXiv:2010.11428 (v2, 23 Oct 2020). Its follow-up R-EBM cites it as Proc. ICASSP 2021 [R-EBM ref. 24].

**1. Problem.**
- HMM-era ASR obtained word confidence from lattice posteriors.
- An attention seq2seq decoder has no lattice. Its softmax probability is the obvious fallback, but it is
  overconfident.
- Fig. 1 shows this: raising the confidence threshold does not monotonically lower WER for the end-to-end model.
  There is a "spike".
- The paper adds a tiny trained module that predicts per-token correctness.

**2. Method, precisely** [PDF p.2 §2]:
- *Backbone*: LAS.
  - $e_{1:L}=\text{Encoder}(x_{1:L})$
  - $a_t=\text{Attention}(a_{t-1},d_{t-1},e_{1:L})$
  - $d_t=\text{Decoder}(a_t,d_{t-1},\text{Emb}(y_{t-1}))$
  - $p(y_t|\cdot)=\text{Softmax}(d_t)$ [Eq. 1–4]
- *CEM*: $p_t=\text{Sigmoid}(\text{FC}(a_t,d_t,\text{Emb}(y_t)))$ [Eq. 5].
  - "The CEM only has one fully-connected layer with 256 units". This adds 0.4 % to the baseline's parameters
    [p.3 §4.2].
  - The activation is not stated.
  - (inferred) The structure is FC-256 then a sigmoid output, which is very close to our 256-unit hidden layer.
- *Labels*: decode the 8-best hypotheses, align each to the reference by edit distance, and label each token 1
  (correct) or 0 (substitution or insertion). The paper's example is "A B C D" vs "A C C D" → c = [1,0,1,1]
  [p.2].
- *Loss*: $\mathcal{L}(c,p)=-\frac1T\sum_{t=1}^T\big(c_t\log p_t+(1-c_t)\log(1-p_t)\big)$ [Eq. 6], summed over
  the n-best list.
- *Trained vs frozen*: "During training, all parameters for the attention-based sequence-to-sequence model are
  fixed" [p.2].
- *Training-set realism*: heavier SpecAugment (10 time masks, T = 50) "to increase the WER on the training set
  for more negative training samples" [p.3 §4.2].
- *Calibration*: a piece-wise linear mapping (PWLM) is fitted on dev. It is monotonic, so it changes NCE but not
  AUC [p.3].
- *Inference*: one score per token. Word scores are averages of token scores. **No reranking.**

**3. Setup** [p.3]:
- Data: LibriSpeech train-clean-100; 16k word-pieces.
- Model: LAS with 184 M parameters, trained in Lingvo with five regularisers.
- Metrics:
  - **NCE** = $(H(c)-H(c,p))/H(c)$ [Eq. 7].
  - **P-R AUC** [Eq. 8].
- Baseline: the softmax probability.
- Out-of-domain test: WSJ eval92 [Table 4].

**4. Key results.**

| Result | Value | Source |
|---|---|---|
| Token level, test-clean/other: AUC, softmax → CEM | 0.976/0.912 → 0.990/0.958 | [Table 2, p.3] |
| Token level: NCE with PWLM, softmax → CEM | 0.166/0.172 → 0.344/0.275 | [Table 2] |
| Word level: AUC | 0.981/0.927 → 0.990/0.962 | [Table 2] |
| Word level: NCE with PWLM | 0.269/0.195 → 0.350/0.270 | [Table 2] |
| LM shallow fusion, softmax NCE | WER falls 7.5/21.6 → 6.8/19.8, but softmax NCE falls 0.269/0.195 → 0.103/0.109. CEM stays at 0.337/0.263. | [Table 3, p.4] |
| WSJ eval92 (mismatched), AUC softmax → CEM | 0.935 → 0.970 without LM; 0.933 → 0.965 with LM | [Table 4, p.4] |
| Regularisers vs softmax confidence | Removing label smoothing *raises* AUC: 0.976/0.912 → 0.985/0.950 | [Table 1, p.3] |

**5. Terminology.**
- **Confidence estimation module (CEM)**: "a lightweight module that can be easily configured on top of any
  attention-based sequence-to-sequence model" [p.2].
- **Confidence score**: "the probability of the token being correct" [p.2].
- **Overconfidence.**
- **NCE (normalised cross-entropy).**
- **P-R curve / AUC.**
- **PWLM.**
- **Teacher forcing**: used in the sense that "calibration behaviour for sequences with an incorrect history is
  uncertain" because of teacher forcing [p.2].

**6. Figures.**
- **Fig. 2 (p.2), checked visually.**
  - The frozen LAS sits in a green/teal-outlined panel; the text says "the green block".
  - The CEM sits in a red/pink-outlined panel above it.
  - Small-caps module names: Encoder, Attention, Decoder, Softmax, FC, Sigmoid.
  - Bold vectors on the arrows. Manhattan-routed taps from the backbone into FC.
  - This is the cleanest template for our "probe internals" figure. FIGURE_STYLE_NOTES §1.9 is accurate.
- **Fig. 1 (p.1) and Fig. 4 (p.4).** WER of the retained subset against confidence threshold. The softmax curve has
  spikes; the CEM curve is monotone.
  - Do **not** recreate these: it is a filtering/selective plot, and abstention framing is forbidden.
- **Fig. 3 (p.3).** Precision-recall curves: softmax vs CEM.

**7. What it gives the deck.**
- Gives the field-standard **name** for our component's ASR cousin: "confidence estimation module".
- Gives the cleanest figure template.
- Presenter sentence: "Speech recognition found in 2020 that a decoder's softmax is overconfident. A
  256-unit trainable head on the frozen decoder's internal state gave a better correctness signal than the
  model's own probability. We make the same bet on a VLM."
- **Cuts against us / caveats:**
  - CEM is token-level and is not used to choose between candidates.
  - Its gains are in confidence quality, not accuracy.
  - The R-EBM paper (same group) shows that averaging CEM token scores to utterance level can underperform the
    baseline [R-EBM Table 4].

**8. Claims we must not make.**
- That CEM reranks or selects. It does not.
- That its "AUC" is ROC-AUC. It is P-R AUC.
- That it reads hidden states only. It reads the attention context, decoder state and token embedding (and, with
  LM fusion, the LM probability [p.4 §4.3]).
- Do not use its "filter by confidence threshold" plots as a framing for our method.

---

### 1.3 Jumper et al. — "Highly accurate protein structure prediction with AlphaFold"
Nature 596, 583–589 (2021). DOI 10.1038/s41586-021-03819-2. Open access.

**1. Problem.** Predict a protein's 3D structure from its amino-acid sequence and multiple-sequence alignment (MSA)
to near-experimental accuracy. The paper also gives a per-residue estimate of how reliable each part of the
prediction is.

**2. Method, precisely.**
- *Network*:
  - Evoformer trunk: 48 blocks, no shared weights, over the MSA representation (s, r, c) and the pair
    representation (r, r, c).
  - Structure module: 8 blocks with shared weights, using invariant point attention.
  - Recycling: three times [Fig. 1e p.2; Fig. 3 p.4].
- *Confidence head*:
  - "Predictions of side-chain χ angles as well as the final, per-residue accuracy of the structure (pLDDT) are
    computed with small per-residue networks on the final activations at the end of the network" [p.5].
  - "the output single representations of the structure module are used to predict binned per-residue lDDT-Cα
    values" [Methods, Training regimen, p.8].
  - **So pLDDT is a binned classification head trained jointly as an auxiliary loss. It is not a regression, and it
    does not sit on a frozen network.**
- *pTM*: "obtained from a pairwise error prediction that is computed as a linear projection from the final pair
  representation" [p.5]. It was added by fine-tuning after CASP14 [p.9].
- *Losses*:
  - FAPE: clamped L1 over N_frames × N_atoms distances [p.5].
  - Auxiliary losses: distogram cross-entropy, BERT-style masked MSA, binned lDDT-Cα, side-chain, and a violation
    loss during fine-tuning [p.8].
- *Selection*: "We train five different models using different random seeds … We inference the five trained
  models and use the predicted confidence score to select the best model per target" [Methods, Inference regimen,
  p.8–9]. This is **best-of-5 selection by a confidence score**. That the score is pLDDT is (inferred) from
  context; the sentence says "predicted confidence score".
- *Probing a frozen trunk*: "we trained a separate structure module for each of the 48 Evoformer blocks in the
  network while keeping all parameters of the main network frozen", giving "a trajectory of 192 intermediate
  structures" [p.5, "Interpreting the neural network"; Fig. 4b].
  - This is a **layer-wise probe on frozen intermediate activations**, the same experimental idiom as our layer
    sweep.

**3. Setup** [p.2–3, p.8–9]:
- CASP14 blind test: n = 87 domains, against the top-15 of 146 entries.
- Recent-PDB set: 10,795 chains released after the 30 Apr 2018 training cut-off, redundancy-reduced.
- Metrics:
  - r.m.s.d.95: Cα r.m.s.d. at 95 % coverage.
  - lDDT-Cα, GDT, TM-score.
  - CIs by 10,000 bootstrap samples.

**4. Key results.**

| Result | Value | Source |
|---|---|---|
| CASP14 median backbone accuracy | 0.96 Å r.m.s.d.95 (95 % CI 0.85–1.16 Å) vs next best 2.8 Å (2.7–4.0 Å) | [p.2] |
| All-atom accuracy | 1.5 Å (1.2–1.6) vs 3.5 Å (3.1–4.2) for the best alternative | [p.2] |
| pLDDT calibration | "lDDT-Cα = 0.997 × pLDDT − 1.17 (Pearson's r = 0.76). n = 10,795 protein chains" | [Fig. 2c legend, p.3] |
| pTM calibration | "TM-score = 0.98 × pTM + 0.07 (Pearson's r = 0.85)" | [Fig. 2d legend, p.3] |
| Recent-PDB median full-chain r.m.s.d.95 | 1.46 Å (1.40–1.56), n = 3,144 | [Fig. 2a legend] |
| Removing BFD / MGnify / both | −0.4 / −0.7 / −6.1 GDT | [Methods p.8] |
| Ensembling | "the accuracy of the network without ensembling is very close or equal to the accuracy with ensembling"; 8× faster | [p.9] |

**5. Terminology.**
- **pLDDT**: "our confidence measure, the predicted local-distance difference test (pLDDT), reliably predicts the
  Cα local-distance difference test (lDDT-Cα) accuracy of the corresponding prediction" [p.2].
- **pTM.**
- **Self-estimates of accuracy** [p.3].
- **Recycling.**
- **Self-distillation.**
- **Model quality assessment / EMA** is the card's CASP term [card]. The paper itself does not use "EMA".

**6. Figures.**
- **Fig. 2c (p.3).** Scatter of average pLDDT (x) against true lDDT-Cα (y), 0–100, with a fitted line and an
  inset. This is the canonical "predicted vs true accuracy" calibration plot, worth echoing for our probe
  reliability diagram.
- **Fig. 4b (p.5).** Domain GDT across 192 Evoformer blocks, from a frozen-trunk probe at each block. This is the
  analogue of a "signal vs layer" plot.
- **Fig. 1e.** The architecture flow. Not needed.

**7. What it gives the deck.**
- Slides: background (one line in a lineage slide) and method justification.
- The exact selection precedent: train 5 seeds and keep the prediction with the best predicted confidence.
- The probing precedent: freeze the trunk and probe every block.
- Presenter sentence: "Even AlphaFold picks among five predictions using its own predicted accuracy. Its authors
  also probed the frozen network block by block to see where the answer forms. Both moves are ones we make."
- **Cuts against us:** pLDDT is trained jointly with the generator (an auxiliary loss); our probe is post hoc on a
  frozen model. There is no evidence here that a post-hoc head matches a jointly trained one. (inferred)

**8. Claims we must not make.**
- "pLDDT is a probe on a frozen network." It is a jointly trained auxiliary head.
- "pLDDT is pointwise regression." It predicts *binned* lDDT-Cα.
- "pLDDT costs nothing extra." This is plausible, since it reads the final activations of the same pass, but the
  paper does not state a cost. (inferred)
- "pLDDT selects among N sampled outputs of one model." The five candidates come from five separately trained
  models (seeds).

---

### 1.4 Collins & Koo — "Discriminative Reranking for Natural Language Parsing"
Computational Linguistics 31(1):25–70, 2005. DOI 10.1162/0891201053630273.

**1. Problem.**
- History-based generative parsers make it awkward to add arbitrary global features of a tree.
- The paper keeps the base parser, takes its n-best parses, and trains a second, discriminative model that
  re-ranks them using ~500,000 extra tree features [abstract, p.1].

**2. Method, precisely** [§4, p.13–25]:
- *Ranking function*: $F(x,\bar a)=a_0L(x)+\sum_{k=1}^m a_kh_k(x)$.
  - $L(x)=\log Q(x)$ is the base parser's log-probability.
  - $h_k$ are binary indicator features.
  - Output: $\arg\max_{x\in\{x_1..x_N\}}F(x,\bar a)$ [p.14].
- *Training target*: $x_{i,1}$ is "the highest-scoring parse" in the candidate list, measured by F-measure against
  gold. It may differ from the gold parse ("the parser may fail to propose the correct parse anywhere in its list")
  [p.14]. **This is the n-best oracle as the target.**
- *Margins*: $M_{ij}=F(x_{i,1})-F(x_{i,j})$.
- *Losses*:
  - Pairwise **ExpLoss** $=\sum_i\sum_{j\ge2}S_{i,j}e^{-M_{i,j}}$ [Eq. 10 with weights], where
    $S_{i,j}=\text{Score}(x_{i,1})-\text{Score}(x_{i,j})$ [Eq. 11].
  - Listwise/conditional **LogLoss** $=\sum_i\log(1+\sum_{j\ge2}e^{-M_{i,j}})$ [Eq. 9].
- *Optimisation*:
  - Boosting-style greedy feature selection, one feature per round.
  - Closed-form update $\delta=\frac12\log\frac{W_k^++\epsilon Z}{W_k^-+\epsilon Z}$ [Eq. 17].
  - $a_0$ is fitted first by brute force over 0.001–10 [Eq. 12]. This step is "crucial".
- *Trained vs frozen*: the base parser is fixed, and its log-likelihood is one feature.
- *N*: "all parses that fall within the beam" (beam width 1,000, dynamic programming disabled), on average
  "27 parses per sentence" [p.20, p.25].
- **Cross-fitted candidate generation** [§5.1, p.25]:
  - "the 36,000 training sentences were parsed in 2,000 sentence chunks, each chunk being parsed with a model
    trained on the remaining 34,000 sentences (this prevented the initial model from being unrealistically 'good'
    on the training sentences)."
  - This is a 2005 precedent for out-of-fold generation of the reranker's training data. It is related to, but not
    the same as, our image-level cross-fitting: they jackknife the *generator*; we cross-fit the *probe*.

**3. Setup.**
- Penn WSJ treebank: sections 2–21 (~40,000 sentences) for training, of which 36,000 are train and 4,000 are dev;
  section 23 is the test set [p.25].
- About 1,000,000 parse trees and over 500,000 features [p.20].
- Metrics: labelled recall and precision, crossing brackets [Table 1].
- Baselines: Collins 1999, Charniak 1997/2000, Ratnaparkhi 1998.

**4. Key results.**

| Result | Value | Source |
|---|---|---|
| F-measure, baseline → reranker | 88.2 % → 89.75 %, "a 13% relative decrease in F-measure error" | [abstract p.1; §1 p.2; §7 p.38] |
| Section 23, ≤100 words (2,416 sentences), LR/LP | Collins 1999 88.1/88.3; ExpLoss 89.6/89.9; Charniak 2000 89.6/89.5 | [Table 1, p.29] |
| ≤40 words (2,245 sentences), LR/LP | Collins 1999 88.5/88.7; ExpLoss 90.2/90.4 | [Table 1] |
| Weighted vs unweighted pairs, best dev score | 101.864 vs 101.699 at ε = 0.0025 | [Table 3, p.32] |
| Efficiency | 100,000 rounds ≈ 37.1 passes, "a saving of a factor of 2,692" | [p.33] |
| ExpLoss vs LogLoss | ExpLoss was "similar, perhaps slightly better"; LogLoss was about 85× slower | [p.28] |

**5. Terminology.**
- **Reranking**: "A second model then attempts to improve upon this initial ranking" [abstract].
- **N-best output.**
- **GEN(x)**: the candidate set.
- **History-based model.**
- **Margin.**
- **ExpLoss / LogLoss.**
- **Ranking error.**
- **Boosting.**
- **Feature selection.**
- **"Head room" for the reranking module** [fn. 9, p.14]: the paper's term for oracle-minus-baseline.

**6. Figures.** All are plain line plots and tables of 2005 vintage.
- Fig. 2: the three potential functions $e^{-z}$, $\log(1+e^{-z})$ and the step function.
- Figs. 5–7: dev learning curves.
- Figs. 8–9: work and savings curves.
- Fig. 2 could illustrate "pointwise BCE vs pairwise exponential" losses if the deck needs it. None is
  otherwise worth recreating.

**7. What it gives the deck.**
- Slides: terminology ("discriminative reranking", "n-best", "head room") and background lineage.
- Method justification: the oracle-best target, and out-of-fold generation of training candidates.
- Presenter sentence: "Our selector is what NLP has called discriminative reranking since Collins 2005. A
  generative model proposes candidates, and a separately trained scorer picks among them. Our scorer reads
  hidden states instead of hand-built tree features."
- **Cuts against us:**
  - Collins keeps the base model's log-likelihood as a calibrated feature ($a_0$), and says this is crucial [p.18].
    We do not use the generator's likelihood at all, which is untested (DESIGN_EVIDENCE line 53).
  - His losses are pairwise or listwise. Ours is pointwise BCE. Our own pairwise (Bradley-Terry) result is in
    DESIGN_EVIDENCE, not here.

**8. Claims we must not make.**
- "Collins & Koo used a neural network." They used boosting over sparse binary features.
- "They train on correctness labels." They train on the margin between the oracle-best parse and every competitor.
- Do not quote 89.75 % as a Table 1 number. It is in the abstract and §1 only; Table 1 reports LR/LP.

---

## Part 1b — Gap-fill: closest-mechanism papers with no PDF on disk (arXiv abstract + HTML full text)

### G1. LiLaVe — Piotrowski, Drzewakowski, Staniszewski, Miłoś, "Lightweight Latent Verifiers for Efficient Meta-Generation Strategies"
arXiv:2504.16760v1, 23 Apr 2025, cs.AI. No venue listed on the abstract page.

- **Problem** [abstract]: verifiers are usually LLMs "as large (or larger) than the base model". LiLaVe "reliably
  extracts correctness signals from the hidden states of the base LLM".
- **Method** [HTML §3.1]:
  - *Features*: hidden states $h_t^l$ captured **during decoding**.
  - *Classifier*: an XGBoost classifier $M(h_t^l,l,t)\in[0,1]$, trained on binary final-answer correctness.
  - *Aggregation*: per-location scores are **averaged** ("taking minimum, maximum, or average score – … we chose
    to use averaging").
  - *Locations*: layers $(-1,-2,-4,-8,-16)$ × the last 16 tokens [HTML §4.2]. This is **score-averaging over
    (layer, token) locations**, not a pooled state.
  - *Hyperparameters, as reported* [HTML A.2]: "maximum tree depth to 5" (A.2 text) vs "max_depth=10, eta=0.1,
    nrounds=30" (A.2.1). This is **inconsistent within the paper**.
- **MLP comparison** [HTML App. A.2, A.2.1]:
  - XGBoost was compared with logistic regression, "a Multi-Layer Perceptron (MLP)" and SwiGLU, 10 seeds each
    → "XGBoost's consistent superiority".
  - The MLP had "a hidden layer of size 16", 20 epochs, Adam 1e-4. The authors note "only a limited sweep".
  - This **verifies** the previously unverified remark in COMPARISON_BRIEF.md line 351, with the caveat that it was
    an MLP of width 16.
- **Setup** [HTML §4.1–4.2, App. A.6]:
  - Base model: Llama 3.1 8B (32 layers). Also Gemma 2 2B and Phi-3.5-mini.
  - Datasets: GSM8K, GSM-Symbolic (+p2), algebra_linear_1d, MATH500.
  - Data size: "training dataset-specific verifiers on 1000 examples"; test sets of 500–1319.
    App. C says "just 5k samples per benchmark", which is not reconciled in the paper.
  - Training samples come from a mixture of temperatures {0, 0.25, 0.5, 0.75, 1.0}. Best-of-n uses T = 1.0,
    with 1–16 samples.
- **Key results.**
  - Table 1, AUC of correctness prediction [HTML §4.3]:

    | Benchmark | LiLaVe | self-reflect | logprobs | ORM-Mistral | ORM-Deepseek |
    |---|---|---|---|---|---|
    | GSM8K | 0.86 | 0.68 | 0.78 | 0.81 | 0.88 |
    | MATH500 | 0.88 | 0.79 | 0.67 | 0.79 | 0.90 |
    | algebra_linear_1d | 0.93 | — | — | — | — |
    | GSM-Symbolic-p2 | 0.78 | — | — | — | — |

  - **Selection** [HTML §4.4, Fig. 3]: "Weighted majority voting performs best for all the datasets". "Best-of-n
    is weaker than weighted majority voting, and **for higher numbers of samples, also weaker than standard
    majority voting**". They attribute this to false positives that grow with n.
  - **Temperature** [HTML §4.2, Fig. 2]: "the predictive performance of the verifier increases both with the
    temperature of the evaluation samples as well as training samples" (GSM8K).
  - **Cross-model re-reading** [HTML App. A.7, Table 3]: Phi responses re-read by Llama and scored by
    Llama-LiLaVe vs scored by Phi-LiLaVe. GSM8K 0.83 vs 0.83; MATH500 0.91 vs 0.93. This supports scoring a
    response from a *separate teacher-forced pass*.
  - **Cross-dataset transfer** [Table 4]: trained on GSM8K → 0.87 algebra, 0.84 MATH; trained on MATH →
    0.53 algebra.
- **Cuts against us:**
  - Their MLP lost to trees. But their MLP had 16 units; our own sweep shows linear < MLP-256 by 0.03 SE
    (DESIGN_EVIDENCE line 130).
  - Their pure best-of-n fell **below plain majority voting** at large n. In our data majority voting loses to
    greedy (−0.0070 judge, DESIGN_EVIDENCE line 46), and probe-weighted voting was a currency trade: judge
    −0.0040 n.s., EM +0.0135 WIN (line 175).
- **Must not claim:**
  - That LiLaVe uses an MLP or mean-pooled states. It averages per-location tree scores.
  - That its numbers are comparable to ours. It is text-only math, reports AUC, and uses a different N and T.

### G2. ReProbe — Ni, Fadeeva, Wu, Akhtar, Zhang, Ash, Leippold, Baldwin, Ng, Shelmanov, Sachan, "Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models"
arXiv:2511.06209v5 (23 Apr 2026). Comments: "ACL 2026 Main".

- **Method** [HTML §3, App. B.2]:
  - *Features*: per-token features $F$ from a **frozen** LLM, either "Hidden states from all layers" or
    attention-to-5-previous-tokens + top-K logits.
  - *Pipeline*:
    - $\mathbf h_0=\text{Linear}_{proj}(F)$;
    - L transformer layers (final config: **1 layer**, hidden size 512, 16 heads);
    - **mean over the step's tokens**, $h_r=\frac1{|r_t|}\sum_{\tau\in r_t}h_L^\tau$;
    - $z=\text{Linear}_2(\text{GeLU}(\text{Dropout}(\text{Linear}_1(h_r))))$.
  - *Loss*: BCE with class weighting (positive weight 3).
  - *Training*: lr 5e-4, batch 128, 5 epochs; fewer than 10 M parameters.
  - **The classifier head is the same shape as ours (Linear → GeLU → Linear) after a mean over the span's tokens.
    It adds a transformer encoder before pooling.**
- **Labels** [HTML §3]:
  - Step-level correctness from an LLM judge (DeepSeek-R1, given the ground truth) or from the model itself
    ("self-supervised").
  - 10.8K PRM800K problems × 3 trajectories ≈ 32K samples, T = 1.0.
  - Judge validation: R1 "95% acc. on PRM800k and ~90% on other datasets" [HTML §4.2].
- **Use** [HTML §4.2]: step scoring, **and** whole-trajectory best-of-N (N = 10 for math and QA, N = 5 for
  planning, T = 1.0), **and** beam search (B = 5, N = 5).
  - For BoN the paper describes aggregating step scores for PRMs, "e.g., minimum step score". The aggregation it
    uses for ReProbe is not stated in the main text.
- **Key results.**
  - Abstract: "match or exceed the performance of PRMs that are up to 810x larger".
  - Table 3, Qwen3-8B BoN accuracy [HTML]:

    | Arm | Overall accuracy | GSM8K | StrQA | SciQA |
    |---|---|---|---|---|
    | pass@1 | 59.8 | — | — | — |
    | pass@N | 72.1 | — | — | — |
    | Best PRM | 61.5 | — | — | — |
    | ReProbe variants | **61.9–62.8** | 96.1–97.8 | 88.1–88.6 | 95.8–97.1 |
    | Majority voting | — | **97.6** | 86.6 | 92.5 |

  - Runtime: 2.6×–25× faster than PRMs [§6].
  - A step-level ReProbe beats token-level linear and token-level transformer probes [§6, Table 11].
- **Internal inconsistency**: the annotation cost is "only $200" (§3; App. B.3 "approximately $200 per model"),
  but the Limitations section says annotating 32K trajectories with R1 "cost >2000 USD".
- **Relevance**: the closest *architecture* to our head (mean-pool then Linear-GeLU-Linear, BCE, frozen LLM) and
  the closest *labelling* scheme (an LLM judge that sees the gold, like our MedVLThinker-32B judge).
- **Cuts against us**: BoN gains over pass@1 are modest (≈ +2–3 points overall), well short of pass@N.
- **Must not claim**: "ReProbe only selects steps". It is also evaluated as a whole-trajectory BoN selector
  (see Part 3 errors).

### G3. CASE — Wang, Hong, Bagci, "A decodability criterion predicts when hidden-state selection beats majority voting in large language models"
arXiv:2608.17124v1 (17 Aug 2026). The HTML header says "Journal: Information Fusion"; the abstract page lists no
venue.

- **Method** [HTML §3.1–3.3]:
  - Operators: $\hat a_{\text{vote}}=\arg\max_a\sum_i\mathbb 1[\text{ans}(c_i)=a]$ and
    $\hat a_{\text{CASE}}=\text{ans}(\arg\max_i g(h_i))$.
  - $h_i$ is the residual-stream state at the **answer-token position** (option letter or final answer token).
    It is recorded by "one teacher-forced forward pass" at calibration; at deployment it is "already produced
    during the generation".
  - Gate $g(h)=\sigma(w^\top h+b)$: per-feature standardisation + ℓ2 logistic regression (C = 1.0).
  - **5-fold GroupKFold with question as the group.**
  - Operating layer: the late layer maximising pooled within-question AUC.
- **Decodability** [Eq. 3–4]:
  - $\text{AUC}_q=\Pr[g(h_i)>g(h_j)\mid y_i=1,y_j=0]$ within question q.
  - $\text{AUC}_{wq}$ averages $\text{AUC}_q$ over questions with mixed correctness.
- **Setup** [HTML §5, §3.5]:
  - Models: Qwen2.5-1.5/3/7/14B, Qwen2.5-Math-7B, Llama-3-8B, OpenBioLLM-8B/70B, BioMistral-7B, Med42-8B,
    MMed-Llama-3-8B, meditron-7B. **Text only.**
  - Benchmarks: LogiQA, MedQA, MedMCQA, PubMedQA (open/closed-book), MATH-500, GSM8K, GPQA. **Multiple-choice or
    short-answer.**
  - Sampling: 16–20 candidates at τ = 0.7, top-p 0.95.
  - Main metric: medium-difficulty CASE−vote gain, averaged over 200 random N = 4 draws, with 95 % bootstrap CI
    over questions.
- **Key results** [HTML]:
  - **Leakage.** On LogiQA, random-split CV 0.610 → grouped 0.542 → within-question AUC **0.502**
    [§6.1, Fig. 2a].
  - **Medium bin.** OpenBioLLM-8B gains, all medium bin [Table 1]:
    - MedMCQA: **+19.1 [+11.1, +27.0]**, AUC 0.794;
    - MedQA: +11.9 [+0.6, +23.5];
    - PubMedQA-cb: +17.0.
  - **General models lose** [Table 1]: Qwen2.5-7B MedQA −4.4; Llama-3-8B MedQA −4.2.
  - **The law** [§6.4, Eq. 11]: $\widehat\Delta=100.8\,\text{AUC}_{wq}-58.5$, r = 0.96 in-sample. Held-out r = 0.75,
    95 % CI [0.51, 0.93]. Threshold ≈ 0.60, with an indeterminate band of 0.55–0.65.
  - **Unconditional gain** [Table 3 caption; §7]: "roughly a third of the medium-bin gain";
    $\widehat\Delta_{\text{uncond}}=27.6\,\text{AUC}_{wq}-15.6$.
  - **N-scaling on the minority-correct subset** [Table 4]: CASE 20.4 → 47.3 % and voting 20.4 → 6.0 % for
    N = 1 → 16. On the hard bin the gain is +16.8. **On the medium bin voting overtakes CASE by N = 16 (−2.0).**
  - **Pooling / position** [§3.2]: the answer token beats last token and "mean over the final K tokens"; "mean-K in
    particular averages the signal away, leaving every layer at chance on LogiQA". A SWIFT-style token-pooled
    readout gave +4.3 vs CASE +9.0 vs a LiLaVe-style tree +10.0 [§6.8].
  - **Classifier** [§3.3, §6.6]: "logistic 0.596 ≥ MLP 0.537 ≥ SVM 0.535" (MLP-256, best Qwen-7B cell);
    "larger discriminators merely overfit the small per-setting data".
  - **Text classifier on the same labels** (TF-IDF): within-question AUC 0.44, vs 0.70 for the hidden-state gate
    [§6.8].
- **Cuts against us — volunteer these:**
  1. A **MLP-256** lost to logistic regression in their small-data setting. Ours is an MLP-256, and our own sweep
     finds linear −0.03 SE worse. Explain that our labelled pool is much larger; CASE's per-setting data are a few
     hundred questions.
  2. **Mean-pooling** hurt in their MCQ/CoT setting. Ours pools only over a short open answer, and our own data
     show span-mean ≥ last token at layers ≥ 14 (DESIGN_EVIDENCE line 87). The settings differ, so say so.
  3. Their **within-question AUC** is the leakage-free diagnostic we have not run (DESIGN_EVIDENCE line 311 and
     gap 18). Our image-level cross-fitting already removes question/image identity leakage from the *training*
     split; the per-question ranking AUC is the missing diagnostic.
- **Must not claim**:
  - That CASE covers VLMs or open-ended answers. Its future work lists free-form generation as untested.
  - That "+19 points" is an unconditional accuracy gain. It is medium-bin only.

### G4. DualRead — Xie, Hao, Liu, Gu, Zhang, "Separating Capability from Confidence: Grounded Dual-State Calibration for GRPO-Trained Medical Vision-Language Models"
arXiv:2609.06419v1 (6 Sep 2026), cs.CV/cs.LG. Shanghai Jiao Tong University.

- **Method** [HTML §"DualRead Confidence Calibration"]:
  - *Actor*: a GRPO-trained actor (group size 8, trained on SLAKE-train), then **frozen**.
  - *Replay*: its greedy answer is replayed "under **teacher forcing**". This is the same "separate re-read pass" as
    ours.
  - *Features*: final-layer states only.
    - **PSR**: the last prompt token before the answer, fed to an MLP (256 → 128) with class-weighted BCE.
    - **P-State**: the state at the closing answer token, fed to a linear head.
    - **P-Visual**: the answer-token mean $h^{ans}=\frac1{|A|}\sum_{t\in A}H^{(L)}_t$ is the **query** for
      cosine-softmax pooling over image-token states (κ = 0.1). The features are
      $[h^{global};h^{evid};h^{ans}\odot h^{evid};|h^{ans}-h^{evid}|]$, fed to a residual head.
  - *Calibration*: each branch is temperature-scaled; the two are fused by an equal-weight log-odds average
    [Eq. 1–8].
  - *Splits*: readers are fit on image-disjoint partitions of SLAKE-train.
- **New metric**: CCG-AUC [Eq. 9–10]. It asks whether confidence drops when the image is swapped for a real hard
  negative that flips the actor from correct to incorrect.
- **Setup**:
  - Backbones: Qwen3-VL-2B-Instruct, MedGemma1.5-4B.
  - In-distribution: SLAKE-test (1,061).
  - OOD: VQA-RAD 600, PathVQA 1,000, PMC-VQA 1,000, OmniMedVQA 920, ReXVQA 1,000, VQA-Med 2019 closed subset 64.
    4,584 in total, macro-averaged.
  - Three seeds.
- **Key results** [Table 1]:
  - Qwen3-VL-2B SLAKE AUROC: **0.952** (DualRead) vs SAPLMA 0.942 vs BICR 0.934.
  - **OOD-averaged AUROC: 0.666.** That is a large in-distribution → OOD drop.
  - MedGemma: 0.947 in-distribution, 0.647 OOD.
  - Table 3: on OOD VQA-RAD (Qwen3-VL-2B), SC@8 0.765 and semantic uncertainty 0.766 **beat** DualRead 0.728.
- **Relevance**: same domain and similar benchmarks, but **it does not select**. It calibrates the single greedy
  answer, so accuracy is unchanged by design.
- **Caveats**:
  - Its motivation is explicitly deferral ("uncertain cases are deferred to clinicians"), and it reports AURC
    (selective risk). **Do not adopt that framing (CRITICAL RULE 6).** Cite it only for teacher-forced replay of a
    frozen medical VLM and for its ID → OOD gap.
- **Must not claim**: that DualRead mean-pools answer tokens as its main feature. The answer mean is only the
  query for image pooling; the main reads are single final-layer tokens.

### G5. Hewitt & Liang — "Designing and Interpreting Probes with Control Tasks"
arXiv:1909.03368v1 (8 Sep 2019). Comments: EMNLP 2019.

- **Definitions** [abstract; HTML §1–2]:
  - A **probe** is a supervised model "trained to predict a property … from a constrained view of the
    representation".
  - A **control task** associates "word types with random outputs".
  - **Selectivity** = linguistic-task accuracy − control-task accuracy.
- **Setup**:
  - Representation: ELMo, layers 1–2.
  - Tasks: POS tagging, dependency edge prediction.
  - Probes: linear, MLP-1 (one ReLU hidden layer, default width 1000), MLP-2.
  - Complexity controls: rank/width, dropout, training size, weight decay, early stopping.
- **Key results** [HTML Table 1, §1]:

  | Probe (POS) | Accuracy | Control | Selectivity |
  |---|---|---|---|
  | MLP-1, default | 97.3 | 92.8 | **4.5** |
  | Linear | 97.2 | 71.2 | **26.0** |
  | MLP-1, 10-dim hidden ("designed with control tasks") | — | — | 16.6 |

  - "dropout … is ineffective for improving selectivity of MLPs" [abstract].
  - "MLP probes with hundreds or 1000 hidden units, as is common, are overparameterized" [§3.5].
  - ELMo1 vs ELMo2: POS 97.2 vs 96.6, but selectivity 26.0 vs 31.4 [§1].
- **Internal inconsistencies**:
  - §3.5 says the default bilinear probe "achieves 16.7 selectivity". Table 1 shows 6.6 for default and 16.7 with
    0.4 dropout.
  - §3.5 says rank 45 was chosen for MLP POS probes. The Table 1 caption says all POS probes designed with control
    tasks "use rank-10".
- **Relevance**:
  - Gives the conventional term and the control we have not run (DESIGN_EVIDENCE gap 18).
  - Our analogue would be a probe trained on **permuted correctness labels** (within question or within image) at
    the same width.
  - Our task differs: correctness of a candidate, which no single token type determines. So the word-type control
    does not port directly; a label-shuffle or "string-prior" control is the analogue. (inferred)
- **Cuts against us**: our 256-unit MLP is in the width range Hewitt & Liang call "overparameterized" for
  linguistic probes. Our own width sweep shows capacity is irrelevant above 32 (DESIGN_EVIDENCE line 131), which
  is evidence in their direction.
- **Must not claim**: that our probe is "selective" in their sense. It has not been measured.

---

## Part 2 — All 16 cards in cards_I.json

| id | title | one line | deck relevance | why |
|---|---|---|---|---|
| li2021rebm | Residual Energy-Based Models for E2E ASR | BiLSTM reranker over frozen decoder states, mean-pooled, BCE | **slide-worthy** | closest classical ancestor; figure template |
| li2020cem | Confidence Estimation for Attention-based S2S ASR | 256-unit FC head on frozen decoder state, BCE vs softmax | **slide-worthy** (shared with R-EBM) | field name "CEM"; cleanest figure template |
| jumper2021alphafold | AlphaFold | pLDDT confidence head; best-of-5 by predicted confidence | mention | selection-by-own-confidence and frozen-trunk probing precedent |
| collins2005reranking | Discriminative Reranking for NL Parsing | boosting reranker over n-best parses | **slide-worthy** (lineage/terminology) | names the paradigm; oracle-best target; out-of-fold candidates |
| qiu2021wordconf | Learning Word-Level Confidence for Subword E2E ASR | self-attention word-level confidence | mention | pooling-granularity precedent; abstract only, no numbers |
| nogueira2019monobert | Passage Re-ranking with BERT | pointwise BERT reranker | mention | "pointwise reranker" term in IR |
| zhu2023llmir | LLMs for IR: A Survey | survey | skip | vocabulary only |
| pandit2025rerankevolution | Evolution of Reranking Models | survey | skip | vocabulary only |
| jiang2018iounet | IoU-Net | head predicts IoU for each box; used in NMS | mention | "own score ≠ correctness" in vision |
| corso2022diffdock | DiffDock | diffusion poses + separate confidence model selects | mention | sample-many-then-score in science |
| chen2021codex | Codex / pass@k | repeated sampling; pass@k | **mention (terminology)** | pass@k = oracle@N term |
| li2022alphacode | AlphaCode | large-scale sampling + execution filtering/clustering | mention | contrast: needs an executable oracle |
| inala2022coderanker | Fault-Aware Neural Code Rankers | execution-free learned code ranker | mention | nearest code analogue; numbers not extracted |
| ni2023lever | LEVER | verifier + LM probability + execution marginalisation | mention | verifier + generator-signal fusion |
| shen2004mtreranking | Discriminative Reranking for MT | n-best reranking in MT | mention (lineage only) | no numbers verified |
| bradley1952paired | Bradley–Terry paired comparisons | P(i>j)=π_i/(π_i+π_j) | **mention (terminology)** | defines the pairwise objective we tested and moved away from |

**Verified usable content for "slide-worthy" and "mention" rows.** Non-core rows are tagged [card] unless I
verified them.

- **R-EBM, CEM, AlphaFold, Collins**: see Part 1.
- **qiu2021wordconf**:
  - "Two confidence models of increasing complexity; the final one uses self-attention to directly learn word-level
    confidence" [card, abstract].
  - Reports NCE/AUC/RMSE "improving substantially" without numbers in the abstract.
  - Use it only as "ASR iterated on how to pool token-level evidence".
- **nogueira2019monobert**:
  - BERT scores each (query, passage) pair independently.
  - "outperforming the previous state of the art by 27% (relative) in MRR@10" on MS MARCO [card, abstract].
  - Term: pointwise reranking.
- **jiang2018iounet**:
  - Predicts IoU per box. Used for NMS and refinement.
  - No mAP number in the abstract [card].
  - Use: "classification confidence ≠ localisation quality".
- **corso2022diffdock**:
  - "38% top-1 success rate (RMSD<2Å) on PDBBind vs. 23% … and 20%" [card, abstract].
  - A separate confidence model ranks the diffusion samples.
- **chen2021codex**:
  - pass@k is the fraction of problems with at least one correct sample in k.
  - "28.8%" single sample; "70.2% … using 100 samples per problem" on HumanEval [card, abstract].
  - Use it to define pass@N = oracle@N and to note that code has a free oracle (unit tests) and we do not.
- **li2022alphacode**:
  - "Average ranking of top 54.3%" on Codeforces [card, abstract].
  - Selection is by execution filtering and clustering, not a learned scorer.
- **inala2022coderanker**:
  - Predicts correctness and the error class without execution.
  - "significantly increase the pass@1 accuracy" (no number) [card, abstract].
- **ni2023lever**:
  - Verifier probability × LM probability, marginalised over programs with the same execution result.
  - "4.6% to 10.9% with code-davinci-002" [card, abstract].
  - Relevant to "should we fuse the probe with the generator's likelihood" (untested by us).
- **shen2004mtreranking**: authors and venue only. No number may be cited [card says so].
- **bradley1952paired**:
  - $P(i>j)=\pi_i/(\pi_i+\pi_j)$.
  - The card says it was not fetched. Cite it for the formula only.

---

## Part 3 — Recommendations

**Own slide.**
1. **"The recipe predates us: n-best reranking".** One lineage slide:
   - Collins & Koo 2005 (parsing, boosting reranker);
   - CEM 2020 → R-EBM 2021 (ASR, frozen decoder states, BCE, mean-pool);
   - AlphaFold 2021 (best-of-5 by predicted confidence; frozen-trunk probe);
   - LiLaVe 2025 / ReProbe 2025–26 / CASE 2026 (LLM hidden-state verifiers);
   - DualRead 2026 (frozen medical VLM, teacher-forced replay, confidence only).
   Put R-EBM's Fig. 1 beside our probe figure. The point is that our contribution is the medical open-ended VLM
   instantiation plus the evaluation, not the mechanism.
2. **CASE deserves its own "closest competitor / what it warns us about" slide.**
   - It is the only paper that directly measures hidden-state selection against majority voting, includes medical
     models, and isolates question-identity leakage.
   - Pair it with our two facts: majority voting loses to greedy for us; and image cross-fitting is our leakage
     control.
   - Name the missing within-question AUROC.

**Comparison table rows** (protocol differences, not numbers):
- LiLaVe, ReProbe, CASE, DualRead, R-EBM.
- Columns: modality; open vs MCQ; generator frozen?; features (layer, position, pooling); head; loss;
  N / T; selection rule; majority-vote baseline reported?; leakage control.
- Fill-ins:
  - ReProbe reports majority voting for GSM8K, StrQA and SciQA [Table 3].
  - CASE reports it everywhere.
  - LiLaVe reports it [Fig. 3].
  - DualRead reports SC@K as a *confidence* baseline only.

**Terminology slide must define:**
- n-best list / candidate pool;
- reranking / rescoring;
- discriminative reranking;
- oracle WER / pass@k / oracle@N ("head room");
- confidence estimation module;
- verifier / outcome reward model;
- **probe** (Hewitt & Liang's definition);
- **control task and selectivity**;
- teacher forcing / teacher-forced replay;
- pointwise vs pairwise vs listwise ranking (Bradley–Terry);
- **within-question (decodability) AUC**;
- question-identity leakage;
- majority voting / self-consistency;
- Byzantine (minority-correct) regime;
- precision-recall AUC vs ROC-AUC (ASR papers use P-R AUC);
- calibration vs discrimination.

**Untested design axes these papers highlight. Add them to the "honest gaps" slide.**
1. Combining the probe with the generator's own log-likelihood (R-EBM Eq. 7; Collins $a_0L(x)$; LEVER). Not
   testable from the current dumps.
2. Within-question AUROC and a control-task (label-permutation) baseline (CASE §3.4, Hewitt & Liang).
3. A non-MLP head (gradient-boosted trees; LiLaVe) and a linear gate at our data scale (CASE).
4. A mixed-temperature training pool (LiLaVe §4.2: higher training temperature helped). This is relevant to the
   mixed-temperature future-work item.

**Errors found in the cards and the three deck briefs.**

| # | Where | Error | Evidence |
|---|---|---|---|
| 1 | cards_I.json, jumper2021alphafold | Says pLDDT is trained "against a continuous accuracy metric … pointwise regression". The paper predicts *binned* per-residue lDDT-Cα as a jointly trained auxiliary loss. "Essentially zero extra cost" is not stated. | [PDF p.8 Methods, Training regimen; p.5] |
| 2 | cards_I.json, li2021rebm | "frozen(-ish) generator". The paper says the baseline "is fixed". The card's method omits that the positives include the ground-truth transcript, and that the inference score adds length-normalised log-likelihood with α. | [Fig. 1 caption; Eq. 5, 7] |
| 3 | cards_I.json, collins2005reranking | "models" and "method" are tagged as an unverified search snippet, and "datasets" as "general knowledge". All are now verified from the PDF (WSJ §2–21/23; 36,000 + 4,000; ~27 parses per sentence; ExpLoss pairwise boosting). | [PDF p.20, p.25] |
| 4 | COMPARISON_BRIEF.md line 351 (LiLaVe) | "no numbers extracted" and "MLP … not re-verified". Now verified: XGBoost beat an MLP of width 16, LR and SwiGLU [App. A.2], and Table 1 AUCs exist. Also add that best-of-n < weighted majority voting, and < plain majority voting at larger n. | [HTML §4.4, A.2] |
| 5 | COMPARISON_BRIEF.md line 352 (ReProbe) | "selects steps inside a chain (… not whole answers)". Incomplete: ReProbe is also evaluated as a whole-trajectory best-of-N selector (Table 3, N = 10/5) and in beam search. | [HTML §4.2, Table 3] |
| 6 | COMPARISON_BRIEF.md line 353 (DualRead) | "same benchmarks and similar pooling". Benchmarks: in-distribution SLAKE-test only; OOD VQA-RAD, PathVQA, PMC-VQA, OmniMedVQA, **ReXVQA**, VQA-Med 2019 closed (64). Pooling is *not* similar: the main reads are single final-layer tokens, and the answer-mean only queries image pooling. | [HTML Datasets; Eq. 1–6] |
| 7 | COMPARISON_BRIEF.md line 350 (CASE) | Correct as far as it goes. Add that it is MCQ/short-answer text only, that its N = 4 evaluation is medium-bin, and that unconditional gains are "roughly a third". Also note that the HTML calls the venue Information Fusion. | [HTML §3.5, Table 3 caption] |
| 8 | FIGURE_STYLE_NOTES §2.4 / terminology | "teacher forcing — only CEM uses the word". Now false: CASE ("one teacher-forced forward pass") and DualRead ("replayed under teacher forcing") both use it for exactly our separate re-read pass. LiLaVe App. A.7 shows re-reading by a pass that did not generate the answer works. | [HTML CASE §3.2; DualRead §Natural-response Replay] |
| 9 | Both ASR papers vs any deck table | Their "AUC" is precision-recall AUC, not ROC-AUC. Never put it in a column with our AUROC. | [CEM Eq. 8; R-EBM §4.4] |
| 10 | Papers' internal inconsistencies to avoid quoting | LiLaVe max_depth 5 vs 10; LiLaVe 1000 questions vs "5k samples"; ReProbe $200 vs >2000 USD; Hewitt & Liang bilinear selectivity 16.7 (text) vs 6.6 (table) and rank 45 vs 10. | as cited |

**Standing cautions.**
- CEM (filtering plots), DualRead (deferral, AURC) and CASE ("retain voting" decision zones) all contain
  selective-prediction framing. Borrow their mechanisms and terms only, never that framing (CRITICAL RULE 6).
- No number from these papers belongs in the same column as our numbers. Every one is a different task, metric
  and grader.
