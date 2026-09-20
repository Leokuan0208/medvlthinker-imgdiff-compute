# lit-vlm-med REPORT — VLM & medical landscape around the residual novelty claim
Agent `lit-vlm-med` · 2026-09-20 · APPEND-ONLY, written incrementally.

Claim under adversarial test (PRIOR_ART_PROBE_VERIFIER_2026-09-14.md §3):
> "No verified paper applies a probe on generation-pass internal states as a best-of-N selector in a
> VISION-LANGUAGE model."

## V2 — what the four local full texts ACTUALLY say (read from local files, not fetched)

### MedProb — arXiv:2609.04336 "MedProb: Probing Internal Representations of Vision-Language Models for
Medical Question Answering" (local `medprob.txt`)
Main method: for each layer l, extract **the hidden state of the last INPUT token** as a summary of the
multimodal context (l.167: "we extract the hidden state of the last input token as a summary of the full
multimodal context") and fit a logistic regression to decode the answer option directly. MCQ-only main
paper (PATH-VQA, SLAKE, VQA-RAD binary/multiclass + OmniMedVQA true-multiclass). 14 matched
general/medical architecture pairs; conclusion: domain-adaptive pretraining does NOT reliably enrich
internal clinical representations (general beats medical in 6/14); probing is more robust to option
ordering than prompting.

**Appendix H, verbatim-checked (l.2671-2696):**
- Setting: "true multiclass items from OmniMedVQA, with **500 training and 100 test examples**."
- Probe input: "The probe is trained to predict correct or incorrect given a single answer candidate
  (**the question, image, and one open-ended candidate answer as input**)." → the candidate is **RE-FED as
  input**; combined with the main-paper feature ("last input token") this is a **second forward pass per
  candidate**. The project doc's characterisation is CORRECT.
- Procedure: "the VLM generates multiple open-ended outputs per question; the probe scores each sampled
  candidate, and we select the highest-scoring one."
- Table 14 (the only numbers): Qwen2.5-VL-7B-Instruct **0.13** probe / **0.09** no-probe / **0.19** oracle;
  Llama-3.2-11B-Vision-Instruct **0.19** / **0.15** / **0.27**. No CI, no n per cell beyond 100 test items.
- Authors' own framing: "an **initial demonstration** rather than a complete open-ended Med-VQA solution",
  "While preliminary", and in the Conclusion (l.907): "**As an initial step toward the open-ended
  setting**, we show in Appendix H that the probe can rerank open-ended VLM generations in a
  rejection-sampling framework, though future work should explore this in more depth."
- **NOT SPECIFIED anywhere in the text I could find: N (number of sampled candidates), the sampling
  temperature for this experiment, the correctness metric used to score open-ended candidates, layer
  choice for the appendix probe, or any significance test.** (grep for temperature/N/num_samples returns
  only main-paper per-model prompting settings.) Their Conclusion also lists "nonlinear probes" and
  "multi-layer aggregation" as FUTURE WORK — both of which the project already does (MLP + 3-layer
  rank-averaged ensemble).

**What this does to the residual claim:** MedProb Appendix H **is** a probe used as a best-of-N selector
in a vision-language model. The claim survives ONLY in the narrow "**generation-pass** internal states"
form (MedProb re-feeds; it does not read the generation pass). The wide form — "nobody used a probe as a
best-of-N selector in a VLM" — is **DEAD as of Sept 2026**. Anyone writing "first application of
hidden-state verification to vision-language / medical VQA" (PRIOR_ART §4.2) would be contradicted by a
reviewer holding MedProb.

### DualRead — arXiv:2609.06419 "Separating Capability from Confidence: Grounded Dual-State Calibration
for GRPO-Trained Medical Vision-Language Models" (local `2609.06419.txt`)
Freezes a **GRPO-post-trained actor**, generates its natural response, then "The input and generated
response are then fed back into the frozen actor for a **teacher-forced replay**" (l.105, l.119) — a
second pass, confirmed. Two reads from the FINAL layer: **PSR** = state at the last non-padding prompt
position (pre-answer solvability; MLP 256-128, class-weighted BCE); **PER** = (a) P-State, the final-layer
state at the **closing position of the answer tokens** (a linear head), and (b) P-Visual, which averages
final-layer states over the answer token positions to form a query and then does answer-conditioned
cosine-similarity pooling (κ=0.1) over the **image** states. Calibrated separately, fused by equal-weight
mean in log-odds. Metrics: AUROC / Brier / ECE-15 / AURC (selective prediction) + a new
**CCG-AUC (Counterfactual Confidence Grounding)** which substitutes a same-question hard-negative image
and asks whether confidence drops. **No best-of-N selection anywhere**; SC@K and semantic uncertainty
appear only as baselines.

**Three things the project's PRIOR_ART doc gets WRONG or imprecise about DualRead:**
1. Backbones: the doc says "Qwen3-VL and MedGemma". The paper (l.505): "**Qwen3-VL-2B-Instruct** and
   **MedGemma1.5-4B**, fully post-trained on SLAKE-train with **GRPO**" — 2B/4B RLVR actors, not
   off-the-shelf 7B models. The GRPO framing is load-bearing for their whole story and is omitted.
2. Benchmarks: the doc lists "SLAKE / VQA-RAD / PathVQA / PMC-VQA / OmniMedVQA". The paper (l.502): SLAKE-test
   (n=1,061, ID) + **six** OOD sets — VQA-RAD 600, PathVQA 1,000, PMC-VQA 1,000, OmniMedVQA 920,
   **ReXVQA 1,000**, **VQA-Med-2019 closed subset 64** = 4,584 OOD. Two sets are missing from the doc.
3. Pooling: the doc says DualRead "mean-pools hidden states over the model's own generated answer tokens
   … our pooling". Only *partly* true — the answer-token mean is a **query vector for visual pooling**
   inside P-Visual; the actual post-answer correctness feature (P-State) is the **single closing token**.
   So DualRead does **not** use our mean-pooled-over-generated-tokens correctness feature.
The doc is CORRECT that DualRead pays a teacher-forced replay and never selects among candidates.

### CASE — arXiv:2608.17124 "A decodability criterion predicts when hidden-state selection beats majority
voting in large language models" (local `2608.17124.txt`) — TEXT-ONLY LLMs, no images. Confirmed.
CASE = a **linear gate on the answer-token hidden state**, selects the highest-scoring candidate. Its
headline contribution is **decodability**: a leakage-free within-question AUC that predicts whether
hidden-state selection beats voting (r=0.75 out-of-sample, threshold ≈ **AUC 0.60**); +16.8 pp on hard
questions at N=16, +41.3 pp where the correct answer is a minority; "matches a generative verifier at
negligible cost". **Critical for this project:** CASE's §1 shows "the naive late-layer probe fails under
leakage-free evaluation; the cause is **question-identity leakage** in random-split cross-validation …
within-question AUC is 0.502 on LogiQA, with only a +0.2-point gain (p=0.66)". This is *exactly* the
project's own open limitation ("pooled AUROC leakage"), already published with a name, a diagnostic and a
threshold — so the project must report within-question/grouped AUC or a reviewer will assume leakage.

### HSRM — arXiv:2608.30841 "HSRM: Hidden-State Reward Models for Test-Time Verification"
(local `2608.30841.txt`) — **mathematical reasoning, text-only**; grep for vision/image/multimodal/VQA
returns no method hits (only an OlympiadBench citation). Confirmed as the doc describes.

---
## V1 — ADVERSARIAL RE-TEST OF THE RESIDUAL CLAIM: **it does not survive in its stated form**

Three papers found today each break a different part of "no verified paper applies a probe on
generation-pass internal states as a best-of-N selector in a VISION-LANGUAGE model".

### [V1-a] arXiv:2605.28527 — "What Frozen VLAs Already Know About Success: A Probing Study of
Value-Like Structure in Foundation Robot Policies" · Zhang, Nie, Lao et al. · submitted 2026-05-27 ·
https://arxiv.org/abs/2605.28527 · **DONE (kills the claim's broad form)**
Linear probes on the FROZEN features of vision-language-action policies (OpenVLA, Pi0.5; plus DINOv2 and
CLIP) recover Monte-Carlo outcome targets from mixed success/failure manipulation trajectories on
LIBERO-Goal. Matched same-task/same-timestep controls give ~92% pairwise ordering accuracy with
label-shuffled controls at chance. Abstract, verbatim: "**Used as a test-time selector over sampled Pi0.5
action prefixes, the same probe turns this offline finding into behavior: on push-plate, success rises
from 26.7% under greedy decoding to 44.3%**, with a second positive case on wine-rack."
**Relation:** a frozen VISION-LANGUAGE(-action) model, a lightweight probe on its own internal states,
used as a best-of-N selector over its own sampled candidates, measured against greedy — structurally our
paper with actions instead of text. A reviewer who knows robotics will produce this. The only surviving
distinctions: candidates are action prefixes not free-text answers, and the domain is manipulation.

### [V1-b] arXiv:2603.22492 — "Tiny Inference-Time Scaling with Latent Verifiers" (VHS = Verifier on
Hidden States) · Bucciarelli, Turri, Baraldi, Cornia, Cucchiara · submitted 2026-03-23 ·
https://arxiv.org/abs/2603.22492 · **DONE for the visual-modality + "no decode/no re-encode" argument**
A verifier that operates directly on the intermediate hidden representations of Diffusion-Transformer
single-step image generators, scoring candidates without decoding to pixel space, explicitly to avoid the
cost of an MLLM verifier that must re-encode each candidate. "reducing joint generation-and-verification
time by 63.3%, compute FLOPs by 51% and VRAM usage by 14.5% … +2.7% on GenEval at the same
inference-time budget."
**Relation:** this is our exact *economic* argument (read the generator's own latents instead of paying a
second encoder per candidate) already published in a visual generative setting, with measured FLOP/VRAM
savings — which is more than the project has for VRAM. It is distinct from PRIOR_ART §2's
arXiv:2603.02829; **both** should be cited. Do not re-pitch "verification is free" as novel.

### [V1-c] arXiv:2608.10835 — "UniProbe: A Learnable Token-Level Hallucination Detector for Large VLMs
using Multi-Structural Internal Representations" · Samuel, Bar-Shalom, Frasca, Fetaya, Ziser, Chechik,
Maron · submitted 2026-08-11 · https://arxiv.org/abs/2608.10835 · **ADJACENT (detector, but acts)**
A lightweight learnable detector over "a frozen LVLM's heterogeneous computational trace from a **single
forward pass**": a directed graph over image patches, query tokens and generated tokens with attention
weights as relations, processed by interleaved GNN / ViT / GRU. Includes "a streaming variant for
hallucination-aware decoding, which **detects and resamples hallucinated tokens during generation**", and
"a self-adaptation strategy aligning the detector with the LVLM's own generations". "reduces object
hallucinations by up to 55% at **1.06× the latency** of standard generation."
**Relation:** generation-pass internal states of a VLM, reused for free, used to CHANGE the output — token
resampling rather than whole-candidate best-of-N. It occupies the "the image tokens matter" design space
(it explicitly models image patches) which the project measured as worth only +0.0002 — a direct,
citable contrast. It also pre-empts "our probe reads the generation pass at no extra pass, in a VLM".

### Verdict V1: **the residual claim must be re-stated or dropped.**
Defensible narrow residual after today: *no paper applies a probe read from the generation pass, pooled
over the candidate's own generated tokens, as a best-of-N selector over free-text answers of a
vision-language model, at the scale of 8 open-ended medical VQA benchmarks.* Every clause in that
sentence is now load-bearing, and a claim that needs five qualifiers is not a contribution. **Stop
selling mechanism novelty; sell the measurements** (see Directions).

---
## V3 — Test-time scaling for medical VLMs/LLMs. Verdict: **OPEN where it matters.**

- **arXiv:2605.10850 — "Verification Mirage: Mapping the Reliability Boundary of Self-Verification in
  Medical VQA"** · Ruinan Jin, Beidi Zhao, Myeongkyun Kang, Qiong Zhang, Xiaoxiao Li · 2026-05-11 ·
  https://arxiv.org/abs/2605.10850 · **ADJACENT, the most useful V3+V5 citation found.**
  Six open-weight VLMs x five medical VQA datasets x seven medical tasks. Self-verification (re-invoking
  the same VLM in a fresh context to check its own answer) fails in a "verification mirage: a regime with
  both high verifier error and high agreement bias" from capacity coupling; knowledge-intensive clinical
  tasks suffer most, perceptual tasks least; verifiers "**under-attend to image evidence**" (the "lazy
  verifier"); multi-turn loops lock in wrong answers; cross-verification only partly mitigates.
  **Use:** the published justification for NOT using a generative self-verifier in medical VQA — the
  strongest external support for the project's finding #3 (training, not size, is the active ingredient).
- **arXiv:2607.20950 — "Best-of-Evidence: Best-of-N Selection under Partial Verification"** · Cenwei
  Zhang, Teng Fang, Yuxia Wang, Derek Li, Bryan Dai, Lei You · 2026-07-23 ·
  https://arxiv.org/abs/2607.20950 · **ADJACENT.** Keeps the BoN pool fixed, represents reusable claims
  with a signed candidate-factor graph, spends a limited budget on evidence actions; theory on residual
  evidence capacity and an O(log K) vs Theta(K) query separation. Verbatim: "Common-ledger experiments on
  **four medical VQA settings** show that BoE can improve fixed-pool selection and rescue some BoN
  failures when evidence is reliable, contrastive, and decision-relevant, while also revealing the
  channel-quality and candidate-generation limits that prevent universal gains." No models, benchmarks or
  numbers in the abstract. Their "limits that prevent universal gains" is our coverage/selection wall.
- **arXiv:2506.13102 — "Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for
  LLMs and VLMs"** · Gyutaek Oh, Seoyeon Kim, Sangjoon Park, Byung-Hoon Kim · 2025-06-16 ·
  https://arxiv.org/abs/2506.13102 · journal version *JMIR* 2026, doi 10.2196/90693, PMID 42490549.
  Five textual medical benchmarks (>5,500 questions) + two multimodal benchmarks (7,000 samples).
  Published finding: concise reasoning with **parallel** scaling for simple tasks, extended CoT /
  **sequential** scaling for complex ones. **NOT VERIFIED:** whether its multimodal half has any
  open-ended free-text task (only the abs page was fetchable).
- Medical PRM line — existence from search only, **cards NOT independently verified**: Med-PRM, MedS3
  (arXiv:2501.12051), MedPRMBench (arXiv:2604.17282), MAPLE (arXiv:2603.08987). All are text-only
  medical reasoning (MedQA/MedMCQA-style), step-level, MCQ-scored.

### V3 verdict — best published evidence that a test-time method beats greedy on OPEN-ENDED medical VQA
**There essentially isn't one, and that is the opening.** The only *verified* head-to-head numbers are
**MedProb Table 14**: 0.13 vs 0.09 (Qwen2.5-VL-7B-Instruct) and 0.19 vs 0.15 (Llama-3.2-11B-Vision) on
**100 test items** of OmniMedVQA, **against a single-sample baseline, not greedy**, no CI, no stated N,
no stated metric, self-described as preliminary. Everything else in medical test-time scaling is MCQ or
text-only. The broadest VLM study, **arXiv:2606.28864 "On Test-Time Scaling for Vision-Language Models"**
(Fawaz Sammani, Tzoulio Chamiti, Nikos Deligiannis, 2026-06-27; "nine test-time scaling methods and six
diverse benchmarks", "improvements of up to around 30%" for small models, "LVLMs lose focus when given
more compute than necessary"), could not be confirmed to contain any medical or open-ended task
(**NOT VERIFIED**).
**Consequence:** the project's 18,452-held-out-question, 8-benchmark, three-currency measurement is, as
far as this search can tell, **the largest measured open-ended medical VQA test-time-selection result in
existence.** That is a contribution even with a known mechanism, and the overlap objection cannot reach
it.

---
## V4 — Evaluation validity for open-ended medical VQA. Verdict: **ADJACENT and moving fast; the exact
## question the project can own is still OPEN.**

### The standard a reviewer will expect (verified on disk + in the Lingshu paper)
- **Lingshu (arXiv:2506.07044v3) evaluation section, verbatim:** for open-ended questions "we directly
  leverage **GPT-4.1 (2025-04-14 version)** to assess consistency between model outputs and reference
  answers, with the prompting strategy detailed in Appendix B."
- **MedEvalKit on disk agrees**: `MedEvalKit/utils/utils.py:294` defaults
  `judge_model = "gpt-4.1-2025-04-14"`; `init_judger()` (l.395-404) supports openai/claude/deepseek/gemini.
- **The judge prompt** (`MedEvalKit/utils/utils.py:49-67`, `get_compare_messages`), verbatim opening:
  *"Your task is to determine whether the user's answer is correct based on the provided questions and
  standard answers (for example, if the user expresses a similar meaning to the standard answer, or
  another interpretation of the standard answer, it is considered correct.)"*, output
  `<think>...</think><judge>{0/1}</judge>` where **0 = correct**.
  **The judge receives question + gold + candidate and NOT the image.** By construction it is a
  **paraphrase-equivalence-to-the-gold detector**, not a clinical-correctness detector. A verifier trained
  on its labels is trained toward gold phrasings — which is precisely the project's own suspicion, and it
  is provable from the harness source rather than from intuition.
- **The harness's non-judge open metrics** (`judge_open_end_vqa`, l.181-200) are EM, BLEU-1..4,
  ROUGE-1/2/L and precision/recall/F1; and `VQA_RAD.py:109/119`, `SLAKE.py:121/127/141`,
  `PATH_VQA.py:123/133` compute **both** the string metrics and the LLM-judge verdict on the same items.
  So dual-currency reporting is the harness's own design — an easy ask of reviewers, not an exotic one.

### Verified cards
- **arXiv:2606.19544 — "Reliability without Validity: A Systematic, Large-Scale Evaluation of
  LLM-as-a-Judge Models Across Agreement, Consistency, and Bias"** · Justin D. Norman et al. · 2026-06-17
  · https://arxiv.org/abs/2606.19544 · **ADJACENT (general domain).** 21 judges, 9 providers, 3 benchmarks
  (MT-Bench, JudgeBench, RewardBench), 118 runs, ~541,000 judgments. Validating a judge by exact-match
  agreement "does not correct for chance and systematically overstates discriminative ability";
  "kappa deflation between exact match and Cohen's kappa is universal (**33-41 pp on MT-Bench**)".
  No medical content. **Use:** report a chance-corrected kappa for any judge the project relies on.
- **arXiv:2607.01103 — "Clinician-Level Agreement Without Clinical Caution: LLM Evaluator Limits in
  Medical AI Benchmarking"** · William Philipp, Finn Fassbender, ... Sebastian Fudickar · 2026-07-01
  (rev. 2026-07-31) · https://arxiv.org/abs/2607.01103 · **ADJACENT, medical, on point.** MedQADE, a
  German open-response clinical benchmark, 3,800 items, **ten practising physicians** and **nine LLM
  evaluators**. Best judge (Gemini 3 Flash) reaches "alignment consistent with the physician ceiling
  (**kappa = 0.694 vs kappa = 0.709**)" but "automated evaluators exhibited **near-absent clinical
  metacognition**". Open-response medical judging is statistically solved and behaviourally not.
- **ACL/LREC 2026 — "Overview of the MEDIQA-EVAL 2026 Shared Task on Evaluation Metrics in Medical
  Multimodal Question Answering"** · Asma Ben Abacha, Wen-wai Yim · 8th Clinical NLP Workshop @ LREC 2026
  · https://aclanthology.org/2026.clinicalnlp-1.1/ · **THE key V4 card — it proves the venue exists.**
  Verbatim: "*Evaluating clinical text generation remains challenging, as automatic metrics often
  correlate weakly with clinician judgments. This issue is particularly pronounced in medical multimodal
  question answering (MMQA) ... To our knowledge, this is the first shared task focused on evaluating
  automatic metrics in this setting. We release a dataset of medical visual question-answer pairs
  annotated with multidimensional clinician judgments. Systems are evaluated by the correlation of their
  metric scores with expert ratings on a held-out test set ... model-based evaluators achieve stronger
  alignment with human judgments than traditional NLG metrics, particularly on English data, while
  performance remains lower on Chinese.*" Participants used VLMs, retrieval-augmented judging,
  metric-specific classifiers, RL and LLM-as-a-judge frameworks.
  **Delta left for us:** MEDIQA-EVAL asks *which metric correlates with clinicians*. It does **not** ask
  *whether the choice of metric changes which test-time method wins* — which is exactly the project's
  measured +0.0737 (judge) / +0.0047 (lenient EM, TIE) / +0.0156 (token-F1) split on identical picks.
- **arXiv:2505.19176 — "Assistant-Guided Mitigation of Teacher Preference Bias in LLM-as-a-Judge"** ·
  Zhuo Liu, Moxin Li, Xun Deng, Qifan Wang, Fuli Feng · 2025-05-25 (rev. 2025-09-18) ·
  https://arxiv.org/abs/2505.19176 · **ADJACENT — closest prior art to "a verifier trained on judge
  labels distils the judge".** Verbatim: "training proxy judge models using evaluation data generated by
  powerful teacher models introduces a critical yet previously overlooked issue: **teacher preference
  bias, where the proxy judge model learns a biased preference for responses from the teacher model**."
  Fix = AGDe-Judge, a 3-stage debias over labels and feedback using an unbiased assistant; six benchmarks.
  **Delta:** their proxy is a *generative judge* scoring other models' text, biased *toward the teacher's
  own outputs*. Ours is a *latent probe* inheriting an image-blind judge's **paraphrase-acceptance**
  criterion and then being *measured by that same judge* — a closed evaluation loop, in medicine.
- **arXiv:2606.12639 — "The Metric Picks the Winner: Evaluation Choice Flips Model Rankings for
  Drug-Response Prediction in Unseen Chemistry"** · https://arxiv.org/html/2606.12639 · **ADJACENT, other
  domain** — precedent that "metric choice reverses the ranking" is publishable on its own in a
  biomedical setting. (Title/URL verified via search; **authors/date NOT VERIFIED**.)

### V4 verdict
Judge reliability in medicine: **ADJACENT / partly DONE** (2607.01103, 2606.19544, MEDIQA-EVAL 2026).
"Which currency you score in decides whether a test-time selection method wins": **OPEN** — I found no
paper reporting one selection method's gain in judge / EM / token-F1 currencies side by side on identical
picks in medical VQA. The project already holds that measurement
(`em-rescore/em_rescore_pooled_probe_2026-09-18.json`).

---
## V5 — Do latent verifiers use the image? Verdict: **partly ANSWERED for confidence estimators and for
## generative self-verifiers; OPEN for a latent best-of-N selector.**

- **DualRead (2609.06419)** introduces **CCG-AUC (Counterfactual Confidence Grounding AUC)** for exactly
  this question: substitute a same-question hard-negative REAL image, hold question and gold fixed, and
  ask whether confidence falls when the actor flips correct->incorrect. It also runs **BICR**, a baseline
  that "contrasts real-image and **black-image** states from the same actor" (l.1303). Blind-image
  controls for hidden-state confidence in medical VLMs therefore **exist in print as of Sept 2026**.
  The project's "+0.0002 from image-token features" and "57% of MCQ answers unchanged when the image is
  blanked" are the same experiment family and must be positioned against DualRead, not sold as new.
- **Verification Mirage (2605.10850)** independently finds medical VQA verifiers "**under-attend to image
  evidence**" — the generative-verifier version of the same result.
- **arXiv:2605.08200 — "Where Reliability Lives in Vision-Language Models: A Mechanistic Study of
  Attention, Hidden States, and Causal Circuits"** · Logan Mann, Ajit Saravanan, Ishan Dave, Shikhar
  Shiromani, Saadullah Ismail, Yi Xia, Emily Huang · 2026-05-05 · https://arxiv.org/abs/2605.08200 ·
  **ADJACENT.** LLaVA-1.5, PaliGemma, Qwen2-VL (3-7B), POPE, n=3,090: "a single **hidden-state linear
  probe reaches AUROC>0.95** on POPE for two of three families", while "**attention structure is a
  near-zero predictor of correctness**" although causally necessary for feature extraction; reliability
  is legible late in the computation. Probes PREDICT, never SELECT.
- **arXiv:2606.10400 — "Do Vision-Language Models See or Guess? Measuring and Reducing Textual-Prior
  Reliance with a Phrasing-Controlled Benchmark"** · Pratham Singla, Shivank Garg, Vihan Singh, Paras
  Chopra · 2026-06-09 · https://arxiv.org/abs/2606.10400 · **ADJACENT, non-medical.** 540 images, six
  reasoning categories, four phrasings per image, 11 VLMs; the **no-image ablation** "collapses the
  open-weight models to their text-only floor (**1 to 9 percent**)". Directly contrastable with the
  project's medical measurement that 57% of a 7B medical VLM's MCQ answers are unchanged when the image
  is blanked — a publishable one-line contrast (general VQA: image is load-bearing; medical MCQ: mostly
  not).
- Named but **NOT independently verified here**: VL-RewardBench (arXiv:2411.17451), Multimodal
  RewardBench 2 (CVPR 2026, Hu et al.), ViLBench (arXiv:2503.20271), InternLM-XComposer2.5-Reward
  (arXiv:2501.12368).
**Still OPEN:** nobody has measured how much of a *latent best-of-N selector's gain* survives removing
the image (from the probe features, or from the generator), nor tied a probe's modality attribution to
its selection gain.

---
## V6 — What medical venues ask of a selection method (no abstention). Low-confidence, brief.
- **MedReason Challenge 2026 (MICCAI)** · https://medreason26.github.io/ · one system must handle **both**
  MCQ and open-ended (the latter returning `reasoning_trace` + `answer`); official ranking is three
  equally weighted metrics — **MCQ Accuracy, Open-ended GT, Open-ended VA** — over public and private
  splits, under OOD shifts including **low-dose X-ray and 7T brain MRI**. The page does **not define** the
  open-ended metrics (no judge model, rubric or calibration procedure) — itself evidence for the V4 gap.
  Format-aware dual-policy evaluation is now an expectation, which retro-justifies the project's
  format-aware framing.
- **MEDIQA-EVAL 2026** (above): venues now want **multidimensional clinician judgments** (factual
  accuracy, visual grounding, completeness, coherence), not one scalar.
- Calibration of a returned answer is an accepted medical-venue contribution and is **not abstention**
  (a probability is reported next to an answer that is always given): MICCAI 2025 "Confidence Calibration
  for Multimodal LLMs: An Empirical Study Through Medical VQA"
  (papers.miccai.org/miccai-2025/paper/1840_paper.pdf) and **arXiv:2606.27023 "Just how sure are you?
  Improving Verbalized Uncertainty Calibration in Medical VQA"**. (Search-level only; **cards NOT
  independently verified**.) This is the one reliability-flavoured axis open to the project under
  CRITICAL RULE 6 — note that DualRead's and 2607.01103's framings both slide into deferral/abstention
  and must not be copied.

---
## Corrections owed to `PRIOR_ART_PROBE_VERIFIER_2026-09-14.md`

- **C1** §3 residual claim: **WRONG as stated.** arXiv:2605.28527 uses a frozen-VLA probe as a test-time
  selector over sampled action prefixes (26.7% -> 44.3%); arXiv:2603.22492 (VHS) does generation-pass
  latent best-of-N for image generation. Narrow to *free-text answers of a VLM*, or drop.
- **C2** §4.2 "First application of generation-pass hidden-state verification to vision-language /
  medical VQA": **overreach.** MedProb App. H already reranks open-ended medical VQA generations with a
  probe. Keep "generation-pass" and "at scale" in the same sentence as the MedProb citation.
- **C3** DualRead backbones: doc says "Qwen3-VL and MedGemma"; paper l.505 says **Qwen3-VL-2B-Instruct**
  and **MedGemma1.5-4B**, "fully post-trained on SLAKE-train with **GRPO**". Add sizes + RLVR framing.
- **C4** DualRead benchmarks: doc lists five; paper l.502 has SLAKE-test 1,061 ID + **six** OOD
  (VQA-RAD 600, PathVQA 1,000, PMC-VQA 1,000, OmniMedVQA 920, **ReXVQA 1,000**, **VQA-Med-2019 closed 64**).
- **C5** DualRead pooling: doc says it mean-pools over generated answer tokens, "our pooling". The
  answer-token mean is only a **query vector** for answer-conditioned visual pooling inside P-Visual; the
  post-answer correctness feature (P-State) is the **single closing token**. Remove "our pooling".
- **C6** Missing: DualRead already ships **CCG-AUC** (hard-negative image substitution) and a
  **black-image** baseline (BICR, l.1303). The project's image-ablation results must be positioned
  against these rather than presented as new diagnostics.
- **C7** Missing: CASE's headline mechanism finding is **question-identity leakage** — naive random-split
  probes look strong, within-question AUC collapses to 0.502 on LogiQA (+0.2 pp, p=0.66), decision
  threshold ~AUC 0.60. The project's "pooled AUROC leakage" is this, already named and published. Report
  within-question / question-grouped AUC or expect the objection.
- **VERIFIED CORRECT** in the doc: CASE "N=16-20 at temperature 0.7" (l.192: "16-20 candidates by
  nucleus sampling (tau=0.7, top-p=0.95, up to 512 new tokens)"); MedProb "candidate fed back as input,
  100 test examples, 'an initial demonstration'" (l.2673, l.2696); HSRM text-only maths.

---
## Directions that survive this search

### D1 (RECOMMENDED; evaluated critically as asked) — "The currency decides the winner: evaluation-currency
### sensitivity of best-of-N gains in open-ended medical VQA, and judge distillation in a latent verifier."
**Pitch.** On identical picks over 18,452 held-out questions the same selector is worth **+0.0737 under a
32B LLM judge, +0.0047 (TIE) under lenient exact match, +0.0156 under token-F1**. That is one
intervention in three currencies that disagree about whether it works at all. Add the mechanism: the
harness judge (`MedEvalKit/utils/utils.py:49-67`) **never sees the image** and is explicitly a
paraphrase-equivalence test against the gold, so a probe trained on its labels is optimised toward gold
phrasings and then graded by the thing it imitates. The paper is measurement + protocol: a
**train-judge x eval-judge transfer matrix** quantifying how much of a latent verifier's reported gain is
judge distillation, plus a reporting standard (two currencies, chance-corrected kappa, cross-family judge).
**Nearest prior art and delta.** MEDIQA-EVAL 2026 (metric-vs-clinician correlation; never asks whether
metric choice changes which *method* wins); 2607.01103 (judges vs clinicians, no method under test);
2606.19544 (kappa deflation, general domain); 2505.19176 (teacher-preference bias in *generative* proxy
judges, not latent verifiers); 2606.12639 (metric flips rankings, drug response). Nobody has shown a
selection method's verdict flip across currencies in medical VQA, nor tested judge distillation in a
latent verifier.
**First experiment (data on disk).** Frozen 24-probe picks + greedy for 8 benchmarks (`ckpts/openvqa/`,
`em-rescore/em_rescore_pooled_probe_2026-09-18.json`). (i) relabel ~37k picks+greedy answers with the
running MedGemma-27B judge plus one more family; (ii) train a probe on judge-B labels (CPU, <=4 threads,
cached features, hours); (iii) report gain(train-judge, eval-judge) as a 2x2 plus EM/F1/BLEU/ROUGE from
`judge_open_end_vqa`; (iv) Cohen's kappa between judges; (v) adjudicate a stratified sample of the ~32% of
upward judge flips with ZERO token overlap with the gold (Kvasir 191/755, GEMeX 237/579, OmniMed 272/736,
`me/judge_length_bias.json`).
**What kills it.** gain(A->B) ~= gain(B->B) and a shrinking currency spread under the cross-family judge:
then there is no distillation and the story collapses into "judges are noisy", which is published.
**Main risk.** It retires the project's own +0.0737 headline, and without a small human-adjudicated
subset a reviewer will note the arbiter is still a model. Budget that adjudication into the plan.

### D2 — "Does a latent verifier look at the image? Modality attribution of a best-of-N selector."
**Pitch.** Decompose the selection *gain* into (a) question-only difficulty (probe on pre-answer prompt
states = DualRead's PSR), (b) the candidate's own generated tokens (ours), (c) image tokens (+0.0002
measured); then re-select with the image blanked and with a same-question hard-negative image (DualRead's
CCG protocol) and report how much of the **gain**, not the AUROC, survives. In a domain where 57% of a 7B
medical VLM's MCQ answers are unchanged without the image, against a 1-9% text-only floor on a
phrasing-controlled general benchmark (2606.10400), the answer is likely to be surprising.
**Nearest prior art and delta.** DualRead (CCG-AUC + black-image BICR, but for confidence on ONE answer);
Verification Mirage ("verifiers under-attend to image evidence", generative verifiers); 2605.08200
(probe AUROC>0.95 on POPE, predict-only); UniProbe (models image patches, detection + token resampling).
Nobody reports modality attribution of a *selection gain*.
**First experiment.** CPU-only on `feats_hidden/`: prompt-state-only vs generated-token-only vs
image-inclusive probes, same splits, 8 seeds, compare GAINS not AUROCs. GPU only for step 2: regenerate
N=8 candidates + hidden states with a blanked image on PathVQA and SLAKE (hours on one A100).
**What kills it.** A prompt-only probe recovering >=90% of the gain means this is a difficulty estimator,
not a verifier — still publishable, but it must then be reframed honestly.
**Risk.** Close enough to DualRead that the framing must stay "gain attribution", never "grounding metric".

### D3 — "A correctness readout that survives the model it was trained on: zero-label onboarding of a
### latent verifier across a fine-tuning lineage."
**Pitch.** Frozen Lingshu-7B probes applied zero-shot to Qwen2.5-VL-7B hidden states give **+0.0451**
macro vs **+0.0820** for a Qwen-native probe (`me/pilot_cross_model_transfer.json`): ~55% of the gain
transfers with ZERO labels on the target model. Extend to MedGemma-4b (different family) and to 7B->32B;
test whether a shared base (Lingshu is a Qwen2.5-VL-7B fine-tune) is required.
**Nearest prior art and delta.** CASE shows the correctness axis is shared across BENCHMARKS and DOMAINS
**within one model** (cosine 0.934 MedQA vs MedMCQA; a gate trained on one benchmark selects on another
at within-question AUC 0.77; cross-domain AUC 0.55-0.69) — it never tests **cross-model** transfer of a
frozen gate. ELHSR/HSRM/MedProb/DualRead all train one probe per model.
**First experiment.** Pure CPU on cached features for all three generators: the full 3x3 matrix of
(probe trained on X) x (features of Y), reporting macro GAIN and within-question AUC, plus the cosine
between class-mean correctness directions across models.
**What kills it.** If a question-difficulty control probe transfers just as well, there is no shared
correctness axis, only shared difficulty.
**Risk.** Different families have different hidden spaces; any learned alignment map re-introduces target
labels and voids the zero-label claim. The unaligned number must be the headline.

### D4 (cheap, high reviewer-value, a control rather than a contribution) — port CASE's leakage
diagnostic to the VLM setting. Compute within-question decodability AUC per benchmark per generator and
test the ~0.60 threshold and the r~0.75 law against the project's measured per-benchmark gains. CPU,
hours, existing features. If the law holds in vision-language it is a one-figure external replication
that pre-empts "your AUROC is leakage"; if it fails, the failure is itself a modality finding. Frame it
as CASE's diagnostic and cite it as such.

---
## Query log (WebSearch unless noted)
1. hidden-state reward model vision-language model best-of-N selection probe internal representations 2026
2. multimodal LLM hallucination detection internal states probe classifier arXiv 2026
3. latent verifier MLLM rerank sampled answers hidden states correctness probe test-time
4. medical LLM test-time scaling best-of-N process reward model Med-PRM MedS3 verifier 2026
5. LLM-as-judge open-ended medical VQA reliability clinician agreement exact match bias 2026
6. multimodal reward model blind image text-only ablation VL-RewardBench language prior reliance 2026
7. open-ended medical VQA free-text best-of-N self-consistency gain over greedy GPT-4 judge 2026
8. Lingshu MedGemma HuatuoGPT-Vision Hulu-Med open-ended medical VQA evaluation protocol judge prompt
9. "Model and Task-Aware Test-Time Scaling Strategies" medicine LLM VLM evaluation study JMIR
10. reward model trained on LLM judge labels inherits judge bias distillation evaluation currency best-of-N
11. choice of evaluation metric changes which method wins reranking conclusions flip metric sensitivity 2026
12. truthfulness probe multimodal hidden states visual tokens modality attribution image ablation VLM
13. MICCAI 2026 clinical evaluation requirements answer selection calibration error severity robustness

WebFetch (verified abs / HTML / ACL pages): 2608.10835 · 2603.22492 · 2605.28527 · 2606.19544 ·
2607.01103 · 2607.20950 · 2605.10850 · 2606.28864 · 2606.10400 · 2605.08200 · 2505.19176 · 2506.13102 ·
arxiv.org/html/2506.07044v3 (Lingshu) · aclanthology.org/2026.clinicalnlp-1.1/ · medreason26.github.io
Local full texts read: medprob.txt · 2609.06419.txt · 2608.17124.txt · 2608.30841.txt
On-disk source read: MedEvalKit/utils/utils.py; utils/{VQA_RAD,SLAKE,PATH_VQA}/*.py

## Could not verify
- 2506.13102 / JMIR 10.2196/90693: whether the multimodal half includes open-ended free-text tasks.
- 2606.28864: whether any of its six benchmarks is medical or open-ended.
- 2607.20950: models, datasets, numbers (abstract only).
- Search-level only, no abs-page verification: Med-PRM, MedS3 (2501.12051), MedPRMBench (2604.17282),
  MAPLE (2603.08987), VL-RewardBench (2411.17451), Multimodal RewardBench 2 (CVPR 2026), ViLBench
  (2503.20271), 2606.27023, MICCAI-2025 calibration paper, 2606.12639 authors/date, TruthLens (2608.05616).
- MedProb Appendix H: N, temperature, correctness metric, layer, significance test — **absent from the
  paper itself**, not merely unfetched.

STATUS: complete.
