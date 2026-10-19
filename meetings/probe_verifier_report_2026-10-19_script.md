# Speaker script — probe_verifier_report_2026-10-19.html

72 slides · main talk ≈ 101 minutes · backup slides are only for questions.

**How to use this script.** Each slide has three parts:
- **On screen** — what the audience is looking at, and where to point.
- **Say** — the words to speak, verbatim. Numbers in the text are the same numbers as on the slide.
- **If asked** — likely questions with a sourced answer. Don't volunteer these unless asked.

Every number comes from a named artifact or paper; the source line at the bottom of each slide names it.


---

## Opening

### 1 · Title — choosing the right answer from a medical VLM's own samples  ·  ~1 min
**Say:**

Good afternoon. Today I want to do three things. First, give you the vocabulary of the research area our
work sits in, so every term I use later is defined. Second, walk through our method piece by piece — what it reads,
how it decides, and why each piece is the way it is. Third, compare it honestly with what other groups have published.
The short version of the method: a medical vision-language model writes eight possible answers to a question, and a
small trained classifier that reads the model's internal states picks one of them.

### 2 · The whole talk on one slide  ·  ~2.5 min
*On screen:* Left: the three boxes. Right: the three numbers — the top one is the grader the probe never trained on.

**Say:**

Here is the whole talk on one slide. The model we use, Lingshu-7B, is a medical vision-language model. For
each question we let it write eight answers, using a bit of randomness. A small trained classifier — in the field this is
called a probe — reads the model's internal numbers for each answer and gives it a score. We train 24 of these probes,
average their rankings, and return the answer they rank first.
The question is whether that beats the model's normal answer, which is the single most likely answer, called the greedy
answer. It does. Averaged over eight benchmarks, a grader from a completely different model family, MedGemma-27B, which the
probe never saw during training, measures a gain of 0.071 — about seven points. The grader whose labels we trained on, a
32-billion-parameter medical model, measures 0.080. Because the probe learned from that second grader, I treat the
MedGemma number as the cleaner one. A lenient string-matching score, which counts an answer as right if it equals or contains
the gold answer, also improves, by 0.012: small, but significant on these benchmarks.
I want to be upfront about what is new. Using a probe on hidden states to pick among sampled answers has been done for
text-only language models, mostly on math, and one medical paper tried it on a hundred questions. What we add is the
measurement: eight benchmarks, held-out images, two graders from different families plus string matching, and a direct
comparison with a published method. I will show you exactly where it works and where it does not.

**If asked:**

- *Is the +0.08 partly the probe learning the training judge's taste?* — Partly, yes. In this project a newly trained verifier gains about 0.006–0.009 under its training judge from paraphrase drift alone (coadapt_verifier_T04_2026-08-14.json). That is why the cross-family judge, never used in training, is quoted first: +0.0710 [+0.0622, +0.0797]. The gap to +0.0798 is 0.0089, the size of that drift. In a September 2×2 each probe scored best under its own training judge and the cross-judge gain was +0.0565 to +0.0658 (judge_2x2_2026-09-20.json).
- *Does the exact-match gain generalise beyond these eight benchmarks?* — Not shown. It is significant with images resampled within these eight benchmarks. Treating the benchmarks themselves as the sampling unit (t-interval over the 8 deltas, df 7) the judge gains stay positive — training judge [+0.0277, +0.1319], cross-family [+0.0248, +0.1171] — but lenient exact match becomes a tie, [−0.0234, +0.0475] (numbers_review_part1.json (derived from final_eval_FINAL_cap320_T07.json by review/scr/w1_micro.py)). Dropping OmniMedVQA or RadImageNet alone also makes it a tie (review/rr_lobo.py).
- *Macro vs micro — what is the question-weighted gain?* — Weighting each question equally instead of each benchmark: training judge +0.1097, cross-family +0.0985, lenient EM +0.0098 (no CI computed; numbers_review_part1.json (derived from final_eval_FINAL_cap320_T07.json by review/scr/w1_micro.py)). Macro is reported so the 10,121-question Kvasir set cannot dominate.
- *Would it work on a new benchmark you have no labels for?* — Not shown, and probably not as well. The claim is about new questions and images from these 8 benchmarks, each of which contributed labelled training questions. Leaving a benchmark out of training gave no benefit over a probe trained on the four-domain set (the PathVQA, SLAKE and VQA-RAD training splits plus Kvasir-VQA) (pooling slide, Part IV); a new benchmark needs some labelled questions.
- *Is 'exact match' strict?* — No — it is lenient: after removing case, punctuation and a few stop-words, an answer counts as correct if it equals the gold or either string contains the other (em_score in src/audit_2026_09_18/em_rescore_pooled_probe.py, verbatim from run_openvqa.py). Strict equality has not been computed for the final method — not measured.
- *Why compare against greedy and not against a bigger model?* — Greedy is what you get from the same model with no extra machinery, so the gain is attributable to the selector alone. Against Lingshu-32B greedy we only have an older, four-benchmark comparison: under the judge one loss (VQA-RAD), two ties, one win; under exact match two losses — so we do not claim to beat the 32B (audit_stats_baselines_2026-09-20.json).
- *What is 'macro'?* — Each benchmark counts one eighth regardless of size, so the 10,121-question Kvasir set cannot dominate the average.

<sub>Source: final_eval_FINAL_cap320_T07.json — macro_over_open8.all.pick_minus_greedy_{judge,xjudge,em}; every question scored by probes that never saw its image</sub>

### 3 · Roadmap  ·  ~0.7 min
**Say:**

The talk has seven parts. Parts one and two are background: the words the field uses and the papers that
matter, including how their methods work. Part three is our method step by step with the math. Part four goes through every
design choice and shows what we tried instead. Then results, then a careful comparison with published work, and finally the
limits of what we can claim.


---

## I · Vocabulary

### 4 · (section divider)  ·  ~0.2 min
**Say:**

Part one: the vocabulary.

### 5 · Medical visual question answering: two answer formats  ·  ~1.5 min
**Say:**

Visual question answering, VQA, means the model gets an image and a question and has to answer. In medicine there
are two answer formats. Closed-ended means the answer comes from a fixed set — yes or no, or a letter from multiple choices.
That is easy to grade. Open-ended means the model writes the answer itself, like the example on the right: the gold answer is
"Large Bowel", but "colon" or "large intestine" would also be right. So grading open answers is itself a problem, and I will
come back to how we do it.
We work on open-ended questions for two reasons: it is the realistic setting, and in an earlier phase of this project we found
that signals about whether an answer is right work much better on free text than on four-option multiple choice.

**If asked:**

- *Where does the 0.87 vs 0.6 come from?* — From the earlier multiple-choice phase of this project (finding 2 in the project retrospective): routing/error-detection signals reached AUROC ≈ 0.6 on 4-option MCQ and ≈ 0.87 on free text. It motivated the switch, it is not a result of this method.

<sub>Source: AUROC figures: project finding 2 (PROJECT_RETROSPECTIVE_2026-07-29.md §5.2), from the earlier multiple-choice phase</sub>

### 6 · How a VLM answers: Lingshu-7B (Qwen2.5-VL-7B architecture)  ·  ~2 min
*On screen:* Left to right along the figure: image, patches, ViT, merger, the token strip, then down to the decoder. End on the violet 'hidden state' card.

**Say:**

This is how the model we use works. Lingshu-7B is a medical fine-tune of Qwen2.5-VL-7B and has exactly its
architecture. The image is cut into 14-by-14-pixel patches. A vision transformer with 32 layers turns each patch into a
vector. A small two-layer network called the merger, or projector, groups each 2-by-2 block of patches into one vector of size
3,584, which is the width of the language model. So one image token covers a 28-by-28-pixel region, and an image capped at our
budget of about 250 thousand pixels becomes at most 320 image tokens.
The language model then reads the system prompt, the image tokens and the question, and writes the answer one token at a time.
It has 28 blocks, and after every block each position has a vector of 3,584 numbers. That vector is called the hidden state, or
the residual stream. Those vectors are what our probe reads — so keep this picture in mind.

**If asked:**

- *Is the 320-token bound stated in the Qwen paper?* — No — it is derived: 14-px patches merged 2×2 give one token per 28×28 px, so tokens ≤ max_pixels/784 = 250,880/784 = 320. Measured: GEMeX averages 303 image tokens at cap320 (FINAL_METHOD_2026-10-16.md).
- *Was Lingshu trained with reinforcement learning?* — No. The released Lingshu-7B is the three supervised stages; RL produced a separate Lingshu-RL that we do not use (Lingshu report, Fig. 6 caption and §3.1.4).

<sub>Source: Qwen2.5-VL report Table 1 p.4, §2.1 p.3–5; Lingshu §3 p.9 (built on Qwen2.5-VL-7B-Instruct); token bound derived (14-px patches, 2×2 merge)</sub>

### 7 · Decoding: greedy vs sampling, and what temperature does  ·  ~2 min
*On screen:* Top chart first (distinct answers rise with T), then the bottom chart (coverage rises with T). Point at 0.7 on both.

**Say:**

To get several different answers we cannot use greedy decoding, because greedy always takes the most likely next
token and gives the same answer every time. Instead we sample: the model's scores for the next token are turned into
probabilities with this softmax, and the temperature T controls how spread out they are. At low temperature the samples are
almost identical; at high temperature they vary more.
The two charts are our own measurements over the eight benchmarks. At the top, the number of distinct answers among eight
samples rises from about 2 at temperature 0.2 to about 5 at 1.0. At the bottom, the chance that at least one of the eight is
correct rises from 0.46 to 0.57. Note that neither axis starts at zero. The price, typically, is that each individual sample is less accurate — that is not shown
here. That trade-off is why the temperature matters. We ship one temperature, 0.7, for every question. An earlier version of the
method picked a different temperature per benchmark; we dropped it, because that needs to know which dataset a question comes
from, which a deployed system does not. I come back to this in part four.
One warning about words: "temperature scaling" in the calibration literature is a different thing — it rescales a trained
classifier's outputs. When I say temperature I always mean the sampling temperature.

**If asked:**

- *Any top-p, top-k or repetition penalty?* — None set: SamplingParams gets temperature, max_tokens = 64, n and logprobs only (src/labeling/run_openvqa.py:186). Whether vLLM 0.9.0.1 applies the model's own generation_config defaults when SamplingParams is built by hand was not separately verified.
- *Is the temperature chosen per benchmark?* — Not in the shipped method: one T = 0.7 for all questions. A per-benchmark map was adopted in FINAL_METHOD_2026-10-16.md and reversed on 2026-10-18, because it needs each question's dataset identity at test time; it is kept as a labelled backup variant. The probe itself needs no dataset label.

<sub>Source: head_temp_ensemble_2026-08-30.json — cells.*.{T}.mean_distinct and .oracle, macro over 8 benchmarks, held-out image halves, N = 8</sub>

### 8 · Best-of-N: coverage and selection are two different problems  ·  ~2 min
*On screen:* The three bars left to right (greedy, probe pick, oracle), then the two brackets on the right: red = coverage wall, orange = selection gap.

**Say:**

This is the central idea of the field we are in. Best-of-N means: sample N answers and return one of them. It
splits into two separate problems. The first is coverage: is a correct answer anywhere among the N? That is measured by
oracle-at-N — the accuracy you would get if someone told you which answer was right. The second is selection: can you find the
right one without knowing the answer?
The bars are our real numbers, averaged over the eight benchmarks. Greedy decoding is right 0.394 of the time.
Among our eight samples a correct one exists 0.559 of the time — that is the ceiling. Our probe picks a correct
answer 0.473 of the time. So 44% of questions are lost because no sample is right — the coverage
wall — and 0.085 is lost to imperfect selection. The ratio of picked to oracle we call selection
efficiency. If you divide the two bars you get 0.85; we compute it benchmark by
benchmark and then average, which gives 0.81, because one weak benchmark, VQA-Med, pulls the average down.

**If asked:**

- *Is selection efficiency a standard metric?* — No. Brown et al. call the problem 'precision', Hu (2026) uses 'effective capture' = fixed/recoverable, which is a different normalisation. Ours is selected/oracle@N per benchmark, then averaged; we always define it.
- *Per benchmark?* — pick ÷ oracle@8 under the judge: PathVQA 0.76, SLAKE 0.90, VQA-RAD 0.78, RadImageNet 0.89, Kvasir-x1 0.85, OmniMed 0.97, VQA-Med 0.43, GEMeX 0.89 (numbers_review_part1.json (derived from final_eval_FINAL_cap320_T07.json by review/scr/w1_micro.py)).
- *What is the coverage wall per benchmark?* — 1 − oracle@8 ranges from 0.12 (SLAKE) to 0.79 (VQA-Med) — see the per-benchmark results slide.

<sub>Source: final_eval_FINAL_cap320_T07.json — cells.*.all.{greedy,pick,oracle}_judge, macro; selection efficiency per benchmark: numbers_review_part1.json (derived from final_eval_FINAL_cap320_T07.json by review/scr/w1_micro.py); Brown et al. 2024 p.2; Bay & Yearick 2026 abstract</sub>

### 9 · Who chooses? The selectors the field uses  ·  ~2 min
**Say:**

Who does the choosing? The field has roughly six families. The simplest is majority vote, also called
self-consistency: return the answer string that appears most often. There are training-free extensions for free text —
asking the model which answer is most consistent, or clustering answers by meaning. You can rank by the model's own
likelihood, or ask the model to check itself. One recent medical paper, WED, makes two roles of the same model negotiate.
And finally there are trained verifiers: a model trained on labelled correct and incorrect answers. That is our family.
Two distinctions the literature uses. Outcome versus process: an outcome reward model scores the final answer, a process reward
model scores every reasoning step. Lightman and colleagues showed process supervision wins on long math solutions, while Uesato
and colleagues found outcome supervision reaches similar final-answer error with less labelling. Our answers are about five
tokens long with no steps, so ours is outcome. And discriminative
versus generative: a discriminative verifier is a classifier that outputs a number; a generative one is a language model that
writes "yes" or "no". Ours is a discriminative outcome verifier, built as a probe on the model's hidden states.
The right column already tells you something about results: majority vote actually loses to greedy on our data, and asking the
model to check itself failed when we tried it earlier.

**If asked:**

- *Which training-free selectors were not run, and why does it matter?* — USC, embedding agreement (EBA), MBR, and likelihood-based ranking were not run on our pools. Per-candidate log-probabilities were not stored for the shipped cap320 pools; they exist for the October 1-megapixel-cap pools on 7 of 8 benchmarks (cumlogprob_all, src/labeling/run_openvqa.py:235–241; RadImageNet has none), so a likelihood-ranking baseline can be run there on CPU — it is first on the owed list. Self-certainty (Kang et al., NeurIPS 2025), the strongest logit-only best-of-N score for open-ended output, was not run either. These are honest gaps — a reviewer may ask whether labels are needed at all; the answer-prior and vote comparisons partly address it.
- *Why an outcome scorer, when Lightman 2023 and Snell found process reward models better?* — Process supervision wins on multi-step math ('process supervision significantly outperforms outcome supervision', Lightman, arXiv:2305.20050). Our answers are ~5 tokens with no steps, so a process reward has nothing to score; Uesato 2022 found 'pure outcome-based supervision produces similar final-answer error rates with less label supervision' (arXiv:2211.14275). We claim nothing about long-chain settings.
- *When was self-verification tested?* — In July with the earlier LoRA-era pools: a zero-shot P(True)-type selector scored below random pick (0.715 vs 0.720 selection efficiency) and below greedy (0.319 vs 0.343). Not re-run on the final pools (PROJECT_RETROSPECTIVE N32).

<sub>Source: ORM/PRM: Uesato et al. arXiv:2211.14275; Lightman et al. arXiv:2305.20050; Snell et al. 2024; Zhang et al. 2025 GenRM; Kang et al. arXiv:2502.18581; Jacob et al. arXiv:2310.09139; BDG arXiv:2410.01064; status column from final_eval_FINAL_cap320_T07.json and PROJECT_RETROSPECTIVE N32</sub>

### 10 · What a probe is — and what it does not prove  ·  ~2.4 min
**Say:**

The core object of our method is a probe. In the interpretability literature a probe is a small classifier that
takes a network's internal vectors as input and is trained to predict some property — while the network itself is frozen, never
changed. The classic version is a linear probe: one weighted sum and a sigmoid. Ours has one hidden layer of width 256, which
makes it an MLP probe.
A note on words: say "one hidden layer of width 256", not "a two-layer MLP", because people count layers differently.
That hidden states carry information about whether a statement is true is an old finding. Azaria and Mitchell read true or false
from a model's activations with 71 to 83 percent accuracy; the contrast-consistent search of Burns and colleagues found it without
labels, about four percent over zero-shot prompting; Kadavath and colleagues at Anthropic trained models to predict whether they
know an answer. But each of them scores one statement on its own.
We need something narrower: among eight answers to the same question, which is right. And Orgad and colleagues warn that such
detectors often fail to carry over to a new dataset — we see the same thing, as I will show.
Two cautions from that literature. First, being able to decode something from a vector does not mean the model uses it — it is
correlation, not mechanism. Second, a probe with enough capacity can memorise labels, so the careful check is a control task:
train the same probe on random labels and measure the difference, called selectivity. We have not run that control yet, and I
list it among the gaps.
Our use is different from the diagnostic use: we do not claim anything about how the model thinks. We use the probe as a tool to
choose an answer, and we judge it only by how often the chosen answer is right on images it never saw.

**If asked:**

- *Hidden states encoding correctness is old (Azaria & Mitchell, CCS, Orgad). What is different?* — Decodability is old and we cite it. Those works classify one statement in isolation; we rank the N answers to the same question, which is a harder within-question task — CASE shows AUC falling from 0.610 to 0.502 on LogiQA once question identity is removed. We have not yet reported within-question AUROC for our probe; it is on the gap list.
- *Your probe fails to transfer to a left-out benchmark. Is that expected?* — It matches Orgad et al. ('such error detectors fail to generalize across datasets'). Kossen et al.'s semantic-entropy probes 'generalize better to out-of-distribution data than previous probing methods that directly predict model accuracy' — an auxiliary target worth trying; untested here.

<sub>Source: Q-Probe arXiv:2402.14688 §2 p.2; Alain & Bengio arXiv:1610.01644; Hewitt & Liang EMNLP 2019; Belinkov, Computational Linguistics 48(1) 2022; Azaria & Mitchell arXiv:2304.13734; Burns et al. arXiv:2212.03827 (ICLR 2023); Kadavath et al. arXiv:2207.05221; Orgad et al. arXiv:2410.02707 (abstracts, via guide cards)</sub>

### 11 · Grading open answers: LLM judges and exact match  ·  ~2 min
**Say:**

Because open answers can be right in many wordings, we need a grader. The field's standard is an LLM judge: a
language model that sees the question, the gold answer and one candidate, and says correct or incorrect. In the MT-Bench
paper's vocabulary that is reference-guided single-answer grading.
We use two judges, and they play different roles. MedVLThinker-32B produced the labels the probe was trained on, so I call it the
training judge. It comes from the same Qwen family as our generator, and a probe trained on its labels can learn its taste. So every
headline number is also graded by MedGemma-27B, from Google's Gemma family, which was never used in training — that is the
cleaner number. Both judges read text only; they do not see the image.
We also report a lenient string-matching score, which counts an answer as right if it equals the gold or either contains the other,
and LLaVA-Med's token recall, so we can compare with their paper. Lenient means a partial answer contained in the gold also passes,
and a correct paraphrase fails. In our files we call these different metrics "currencies"; from now on I just say metric, and I never
show a judge number without the others.
On biases: position bias needs two answers side by side, so it cannot happen here. Verbosity bias — preferring longer answers —
we checked: always picking the longest answer loses 0.029. Self-preference and learning the training judge's taste are why we have
the second judge. This is also the field's norm: each benchmark paper uses its own LLM judge — GPT-4.1 for Lingshu, Mistral-Large for
RadImageNet-VQA, Qwen3-30B for Kvasir-VQA-x1, GPT-4o for GEMeX.

**If asked:**

- *Has the judge been validated against humans?* — No human has rated this method's picks. Earlier spot checks, on July pools: an LLM rater (results/cascade_methods/claude_judge/) on SLAKE and VQA-RAD, and an audit of 50 + 90 PathVQA items (pathvqa_judge_audit.json). On the 50 direct (short) answers the judge was wrong 2 times: 0.04, Wilson 95% [0.011, 0.135] (computed for 2/50); on 90 long reasoning-mode answers it erred more (up to 0.28 on rejections). Who labelled them is not recorded, so we do not call it human. Nothing at all on the other 5 benchmarks. The cross-family judge agreeing in sign is our main defence; a human rating is listed as a gap.
- *How often do the two judges agree?* — On September training candidates (112,770 rows) the two judges agree on 92.9% of labels, Cohen's κ = 0.79 (judge_2x2_2026-09-20.json train_label_agreement_AB and train_label_pos_rate; κ from the marginals, numbers_review_part2.json judge_agreement). MedGemma says 'correct' more often (23.9% vs 19.4%). Not yet measured on the October final pools. On the final method, the two judges' per-benchmark deltas agree in sign on 7 of 8 benchmarks; on VQA-RAD the training judge gives +0.0000 and MedGemma +0.0100 (final_eval_FINAL_cap320_T07.json cells.*.all).
- *How does the judge decide?* — Deterministically: temperature 0, text-only prompt in run_judge.py; correct iff the first-token probability of 'Yes' is at least that of 'No'. MedGemma-27B-it runs through the same script and prompt.
- *Why was MedVLThinker-32B chosen as judge?* — It was the labelling script's default, not a deliberate choice (AUDIT_2026-09-18.md §1). It is text-only and sees the gold answer.

<sub>Source: MT-Bench arXiv:2306.05685 §3; src/labeling/run_judge.py (MedGemma via the same script); EM/F1: src/audit_2026_09_18/em_rescore_pooled_probe.py; length bias: AUDIT_2026-09-18.md §2.2 — full provenance in the script</sub>

### 12 · Related but different: uncertainty estimation is not selection  ·  ~1.5 min
**Say:**

One neighbouring field is easy to confuse with ours: uncertainty estimation. Those methods ask "is this answer
probably wrong?" — one score per answer or per question, judged by how well it separates right from wrong across many
questions. We ask a different question: among the eight answers to one question, which is right? Only the order within one
question matters.
That difference has consequences. Semantic entropy, the best-known uncertainty method, gives one number per question, the same
for all eight candidates, so it cannot choose between them. Calibration methods only stretch the scores, so they never change
which answer is first. The shared ingredient is that hidden-state probes appear in both areas — ours is the variant that scores
each candidate's correctness.

**If asked:**

- *If calibration cannot reorder, why test Platt averaging later?* — Calibrating one probe cannot reorder its own scores. Calibrating each of the 24 probes before averaging them can change the order of the average. That is what the readout comparison tested, and it tied under both judges.
- *Isn't P(True) an uncertainty method?* — It scores one answer at a time, so it can rank candidates — used that way it is the self-verification selector, which failed on our earlier pools (0.715 vs 0.720 random-pick selection efficiency; PROJECT_RETROSPECTIVE N32).

<sub>Source: Farquhar et al. Nature 630 (2024) Eq. 3 p.8; Kossen et al. arXiv:2406.15927; Byun et al. arXiv:2604.02543 Table 2 p.7; Geng et al. NAACL 2024</sub>

### 13 · Glossary — the terms as used in this talk  ·  ~0.5 min
**Say:**

This slide collects the terms in one place; I will not read it, it is here for reference while we go through the
method.

<sub>Source: π = share of correct training rows 0.2071 / 0.2085 → (1−π)/π = 3.83 / 3.80 (numbers_review_part1.json, review/scr/w1_pi.py); head_sweep.py:172–174; FLOP-eq unit: free_head_2026-08-16.json DELIVERABLE_3_cost.unit</sub>


---

## II · Key papers

### 14 · (section divider)  ·  ~0.2 min
**Say:**

Part two: the papers this work stands on, and how their methods work.

<sub>Source: domain guide cards_*.json: 194 cards, 186 distinct papers (review/scr/w1_count.py)</sub>

### 15 · The idea is old: rerank a generator's candidates with a trained scorer  ·  ~1.5 min
*On screen:* Walk the timeline left to right, alternating top and bottom labels; the colour legend at the bottom groups them.

**Say:**

The core idea is much older than large language models. In 2005 Collins and Koo took a parser's list of
candidate parses and trained a second model to rerank them. Speech recognition did related things: a trained scorer reads the
frozen recogniser's internal features to estimate confidence, in 2020, and to rescore the list of candidate transcripts, in 2021.
Cobbe and colleagues brought trained verifiers to language models in 2021, and the same year WebGPT used a learned reward model to
pick the best of up to 64 long free-text answers. AlphaFold picks the best of five protein structures by its own predicted
confidence. From 2024 on, several groups put probes on language models' hidden states — Q-Probe, SWIFT, LiLaVe, HSRM, CASE. In
2026 the idea also reached vision: a linear probe on a frozen vision-language-action robot policy was used to choose among sampled
action sequences. And two papers probed medical vision-language models — mostly for multiple choice and confidence, plus a
hundred-question demonstration of reranking open answers in MedProb's appendix.
So I want to be clear from the start: our contribution is not the mechanism.

**If asked:**

- *Hasn't a probe on a frozen vision-language model already been used as a best-of-N selector?* — Yes, outside VQA: 'What Frozen VLAs Already Know About Success' (arXiv:2605.28527) uses a linear probe on a frozen Pi0.5 policy as a test-time selector over sampled action prefixes — 'on push-plate, success rises from 26.7% under greedy decoding to 44.3%' (abstract, via AUDIT_2026-09-18_appendix/lit-vlm-med.md [V1-a]). In medical VQA, MedProb App. H reranked open answers on 100 items. We claim the measurement, not the idea.
- *Why not N = 64 or more?* — More samples raise coverage but also the chance of fooling the scorer — reward over-optimisation (Gao, Schulman & Hilton 2022); Cobbe saw accuracy fall beyond 400 samples. Our largest measured N is 32 on one benchmark; see the N slide.

<sub>Source: Collins & Koo, Comp. Ling. 31(1) 2005; Li et al. arXiv:2010.11428 (CEM), 2103.14152 (R-EBM); Cobbe arXiv:2110.14168; Nakano arXiv:2112.09332; Jumper, Nature 2021; arXiv:2402.14688, 2505.12225 (SWIFT, KDD 2026), 2504.16760, 2608.30841, 2608.17124, 2605.28527 (VLA probe), 2609.04336, 2609.06419</sub>

### 16 · Trained verifiers (Cobbe 2021) and coverage vs precision (Brown 2024)  ·  ~2 min
**Say:**

Two founding papers. Cobbe and colleagues at OpenAI in 2021 established the recipe we follow. Sample a hundred
solutions to each training problem, label each one correct or incorrect by its final answer, train a verifier to predict that
label, and at test time return the highest-scored of a hundred samples. On grade-school math a 6-billion model with a verifier
slightly beat a fine-tuned 175-billion model — they describe it as worth a 30-times increase in model size. Two of their findings
cut against us: their token-level verifier was better than a whole-answer verifier, and past 400 samples accuracy fell, because
the generator started producing answers that fooled the verifier — the field calls this reward over-optimisation.
Brown and colleagues in 2024 showed why selection is the hard part. With ten thousand samples from Llama-3-8B-Instruct, a correct
answer exists for 98 percent of the MATH problems — but voting or off-the-shelf reward models only got from 40.5 to 41.4 percent. Coverage grows; selection
does not keep up. That split frames all our results.
What we take from them: Cobbe's recipe — outcome labels, a trained scorer, pick the highest score — but with the verifier replaced
by a small probe on our generator's own frozen internal states.

<sub>Source: Cobbe §4.2 p.8, §5.1 p.10, §6 p.12; Brown p.2–3, Fig. 7 p.10 (PAPER_NOTES_A); Gao et al. arXiv:2210.10760</sub>

### 17 · Test-time compute vs model size (Snell et al. 2024)  ·  ~1.5 min
**Say:**

Snell and colleagues framed the modern debate: is it better to spend compute at test time or on a bigger model?
They describe every method as a proposer plus a verifier, and they choose the strategy per question according to its difficulty;
their verifiers are process reward models. Their result is that smart test-time search can match plain best-of-N with four times
less compute, and can beat a 14-times-larger model — but only, in their words, "on problems where a smaller base model attains
somewhat non-trivial success rates". On the hardest questions, they say, pretraining is the more effective way.
Our data show a similar shape. On OmniMedVQA, where a correct answer is among the eight samples 70 percent of the time, we gain 16
points. On VQA-Med, where that is only 21 percent, we gain nothing — though I will show that there the probe also fails to find the
correct answers that do exist. Two points against us: they found that step-level verifiers beat outcome-level ones, more so as N
grows; and their method chooses the budget per question, while ours uses one fixed setting. Their default way of combining scores —
adding up the verifier's score over identical answers — is something we tested against our own readout; I show it in part four.

<sub>Source: Snell v1 abstract, Fig. 4 p.10, Fig. 9 p.15, App. F p.24–25; ours: final_eval_FINAL_cap320_T07.json cells.{omnimed_open,vqamed_open}</sub>

### 18 · Training-free selection and its ceiling: majority vote, USC, embeddings  ·  ~1.5 min
*On screen:* Right chart: most dots sit left of the zero line; end on the bold macro row at the bottom (hollow dot = not significant).

**Say:**

The simplest selector is majority vote, which the field calls self-consistency. On math it is very strong — on
grade-school math with a big model it raised accuracy from 56 to 74 percent — because a correct number tends to come up again and
again. Bay and Yearick showed its hard ceiling: as you add samples, voting converges to the most frequent answer, so it can never
beat the rate at which the most frequent answer is correct. A learned scorer is not bound by that.
The chart is majority vote minus greedy on our pools. On most benchmarks it is negative, and averaged it loses
0.0094. Free-text medical answers rarely repeat exactly, and the right answer is often a minority — like "Large Bowel",
which appeared once in our example. There are training-free methods designed for free text: USC asks the language model which
answer is the most consistent with the others; EBA picks the centre of the largest cluster of answer embeddings; and self-certainty
scores each sample from the model's own output probabilities. We have not run them, and that is a gap.

**If asked:**

- *LiLaVe and CASE found pure best-of-n can fall below voting. Why not here?* — Their answers repeat (math, multiple choice), so voting is strong. On our open answers voting itself loses to greedy (−0.0094 [−0.0157, −0.0034]), and the probe beats voting by +0.0893 [+0.0811, +0.0972] under the training judge and +0.0136 [+0.0053, +0.0216] under lenient EM (extra_numbers.json).

<sub>Source: final_eval_FINAL_cap320_T07.json cells.*.all.sc_minus_greedy_judge (+CI), macro_over_open8.all.sc_minus_greedy_judge; Wang 2023 Table 2 p.5; Bay Prop. 5 Eq. 13 p.10</sub>

### 19 · Hidden-state verifiers in text LLMs: Q-Probe, SWIFT, HSRM  ·  ~2.5 min
**Say:**

These are our three closest relatives, all on text-only language models. Q-Probe in 2024 put a linear probe on a
frozen code model's embeddings to rerank 48 sampled programs. SWIFT reads every token of every layer, pools them with a learned gate,
and trains with binary cross-entropy — the same kind of loss as ours; ours additionally up-weights the correct answers. HSRM reads the states at the end of each reasoning step with a small
Transformer and picks the best of eight — the same N as ours.
The last row is us, so you can compare column by column: we read only the answer's own tokens, at three middle layers, with a simple
mean, and a one-hidden-layer network.
I want to be honest about what these papers say against our choices. HSRM found a pairwise ranking loss, Bradley–Terry, better than
our loss — 85.9 versus 83.9 on grade-school math. Q-Probe found a within-question loss better than a pointwise one, and saw MLPs
overfit on preference data. SWIFT found that learning which tokens to weight beats a plain average, by 2.7 points. On the other
side, SWIFT found our loss the best of five, and HSRM found the reader's architecture matters much less than what you feed it: a
mean-pooled MLP reader did about as well as their Transformer reader. I will come back to each of these when I justify our choices.

**If asked:**

- *Is the BCE-vs-ranking-loss question settled?* — No. SWIFT finds BCE best; HSRM and Q-Probe find within-question losses better. Our own BCE-vs-Bradley–Terry comparison is confounded (Bradley–Terry sees 38.9% of rows and 3.26× fewer steps), so we do not claim BCE is better; a ranking loss on the final recipe is untested.

<sub>Source: Q-Probe Table 1 p.6, §7.2; SWIFT Tables 1 (MATH Avg. column, arxiv.org/html/2505.12225), 7, 8; HSRM Tables 3, 9, 10 (PAPER_NOTES_BC); ours final_eval_FINAL_cap320_T07.json</sub>

### 20 · CASE, LiLaVe, ReProbe: when hidden-state selection beats voting  ·  ~2 min
**Say:**

Three more relatives, each teaching something. CASE, from August 2026, asks exactly when choosing by hidden states
beats majority voting. Their key contribution is a measure called within-question AUC: how well the probe orders the candidates of
the same question. They show that ordinary cross-validation badly overstates probes, because a probe can learn which questions are
easy instead of which answers are right — 0.61 drops to 0.50 when measured properly. That metric predicts how much you gain over
voting; on a medical multiple-choice set, the medium-difficulty questions, it gained 19 points over voting. Two of their findings
cut against us: a small logistic regression, 0.596, beat a 256-unit MLP, 0.537, in their small-data setting, and averaging over
tokens hurt in their multiple-choice setting. Also, on those medium questions, voting caught up and overtook them by N of 16. Our
evaluation avoids the leakage they warn about by splitting by image, but we have not reported their within-question AUC, and we should.
LiLaVe uses gradient-boosted trees on the hidden states of the last 16 tokens of five layers; interestingly, in their data plain
best-of-n eventually fell below voting, and weighted voting was best. ReProbe has the same classifier shape as ours — one Transformer
layer, a mean over the span, then linear, GELU, linear, trained with binary cross-entropy on labels from an LLM judge — and matches
process reward models up to 810 times larger.

<sub>Source: CASE HTML §3.2, §6.1, §6.4, §6.6, Tables 1, 4; LiLaVe HTML Table 1, §4.4; ReProbe HTML §3 (PAPER_NOTES_I)</sub>

### 21 · Medical side: self-verification fails; probes do MCQ or a demo  ·  ~2.2 min
**Say:**

On the medical side, three papers set the context. Verification Mirage asked medical VLMs, including Lingshu, to
check their own answers. Lingshu accepted 62 to 73 percent of its own wrong answers, a bigger model of the same family did not
significantly help, and over four turns of conversation the wrong answers got locked in. That is the reason our selector is trained
on labels rather than prompted.
One point to keep apart: in part four I show that a separate 32-billion model, judging our 7B's candidates without training, does
better than a 7B doing the same. That is not in conflict with Mirage. Mirage measures whether a model accepts its own wrong answers
when it re-checks them; we measure best-of-8 selection accuracy, with a different model as the judge. Both point the same way:
training, not prompting, is what makes a selector work.
MedProb showed that a simple linear probe on a medical VLM's internal state can answer closed multiple-choice questions better than
the model's own generated answer — by more than 20 points in several cases — but it reads the state before any answer is written,
so it is a classifier, not a selector. In an appendix they tried reranking open answers on a hundred OmniMedVQA questions, from 0.09
to 0.13, which they call an initial demonstration; the number of samples, the grader and the probe type are not stated. DualRead
re-reads a frozen medical VLM's answer with teacher forcing — the same re-reading we do — but only to calibrate confidence, and its
quality drops sharply on new datasets: AUROC 0.95 in distribution, 0.67 outside.

**If asked:**

- *Did Verification Mirage use Lingshu-7B?* — The paper does not state Lingshu's size; do not call it 7B.
- *Is MedProb App. H comparable to our OmniMedVQA result?* — No: 100 test items, unstated grader and recipe, baseline is a single sample rather than greedy. Ours on OmniMedVQA: 8,883 questions, greedy 0.516 → 0.677 judge. We make no priority claim — its authors call it 'an initial demonstration'.
- *Why not just ask the model, or use its likelihood — no labels needed?* — Prompted self-verification: Mirage shows Lingshu accepts 62–73% of its own wrong answers. Plain log-probability ranking and self-certainty have not been run on our pools — an owed baseline on the gap list.

<sub>Source: Mirage Tables 1, 11–15, App. B.4; MedProb p.6, App. H p.27 Table 14, App. L Table 21; DualRead Table 1 (PAPER_NOTES_BC, PAPER_NOTES_I)</sub>

### 22 · Published medical test-time selectors: WED and Best-of-Evidence  ·  ~2 min
**Say:**

Two recent papers do test-time selection specifically for medical VQA. WED is training-free. It samples up to
eight distinct candidates, using up to sixteen draws at temperatures 0.5 and 1.0, and merges answers with the same meaning. Then the
same model plays two roles: a generator that scores how likely each candidate is, and a verifier that says whether it is correct. The
two roles adjust toward each other in a game until their rankings agree, measured with a Wasserstein distance so that synonyms count
as agreement. The game itself comes from earlier work, the Consensus Game and the Bayesian Decoding Game; WED's addition is that
stopping rule. Graded by the Grok model, it reports one to five points over greedy on most small models, and a small loss for the
largest one on VQA-RAD, so gains shrink as models get bigger. Best-of-Evidence samples sixteen answers and uses a
235-billion-parameter judge to check individual claims — modality, anatomy and so on; it gains under one point over majority vote,
significant on two of four datasets, with no greedy baseline, and its VQA-Med items come from the training split.
WED matters most for us, because it is the only published selector evaluated on two of our open benchmarks, VQA-RAD and PathVQA. So
rather than compare numbers across papers, we ran WED's own code on our candidates — I show that result in part six.

<sub>Source: WED §2 p.3–6, p.5 (≤ 8 candidates, ≤ 16 calls), p.7, Table 1 p.8, authors' code BDG.py; BoE §5.1, Table 2 p.9, App. B.2 (PAPER_NOTES_DE, PAPER_NOTES_BC); Jacob et al. arXiv:2310.09139; BDG arXiv:2410.01064</sub>

### 23 · Where this work sits  ·  ~1.5 min
**Say:**

Putting it together. The literature's hidden-state verifiers share four properties: a probe on frozen states, used to
select among samples, pooled over the answer's own tokens, and in principle needing no extra pass through the model, because the
states were already computed while generating. The text-only papers have most of these; the medical papers mostly probe the model
without selecting free-text answers. In vision the idea also exists, for robot actions and for image generation.
Our row has all of them except one honest asterisk: in our experiments we re-read each answer in a separate pass, so "no extra pass"
is true by design, not yet as run. For an earlier probe, capturing the states inside generation changed 24 of 2,345 picks and left
accuracy unchanged; for the shipped probe, re-extracting the features at the generation resolution is running now.
So the contribution is the evaluation, not the mechanism. MedProb's appendix tried this on a hundred questions; we measure it on eight
benchmarks, with held-out images, a second grader from another model family plus string matching, and against a published selector on
identical candidates.

**If asked:**

- *Is this just HSRM / SWIFT / Q-Probe with a different backbone?* — In mechanism, yes, and we say so. What differs is the setting and the measurement: an image-conditioned model, short free-text answers with no exact checker, labels from an LLM judge, 8 benchmarks cross-fitted by image, a cross-family judge and lenient EM, and a controlled comparison with a published medical selector. Some of their findings cut against our choices: HSRM's Bradley–Terry > BCE, SWIFT's gate > mean, Q-Probe's within-question loss > CE.
- *Why not fine-tune the generator on the same labels instead?* — A frozen generator keeps the model unchanged and the verifier swappable. A generator LoRA-SFT baseline exists only from an earlier era (+0.0142 at 1× cost, ckpts/cheapleg/scores_*) and has not been re-run against the final method; it is owed. Do not set it beside +0.0798 as a ranking.

<sub>Source: literature/DOMAIN_GUIDE_2026-09-30.md §4.1 (branch worktree-lit-domain-package); PAPER_NOTES_BC/I; VLA probe arXiv:2605.28527; VHS arXiv:2603.22492; novelty withdrawn: AUDIT_2026-09-18_appendix/lit-vlm-med.md C2 — full provenance in the script</sub>


---

## III · Method

### 24 · (section divider)  ·  ~0.2 min
**Say:**

Part three: our method, one step at a time.

### 25 · The method in one figure  ·  ~2.5 min
*On screen:* Walk left to right: the CT and its question; step 1, the eight sampled pills; the three merged answers with their vote badges; step 2, the violet taps at blocks 22 / 20 / 18; step 3, the 24 probe tiles; step 4, the rank bars and the green returned answer.

**Say:**

Here is the whole method on one real example from SLAKE: an abdominal CT, and the question "what organ is the
gray part on the left of the image?" The correct answer is large bowel.
Step one: the frozen Lingshu model writes eight answers at temperature 0.7. Five say small bowel, two say spleen, and only one says
large bowel. Identical answers are merged, and we remember how many votes each got.
Step two: each distinct answer is fed back through the same frozen model together with the image and question — this is called
teacher forcing — and we read the hidden states at decoder layers 18, 20 and 22, averaged over that answer's own tokens. In our runs
this re-read used a higher image resolution than the generation; I come back to that.
Step three: 24 small trained probes — three layers times eight training seeds — each give every answer a score.
Step four: each probe's scores are turned into ranks within this question, the ranks are averaged, and the highest wins. Here
large bowel gets a mean rank of 0.917, and 20 of the 24 probes put it first. Majority vote would have returned small bowel, which is
wrong. Everything on this slide is a real number from the shipped probes, and these probes never saw this image during training.

**If asked:**

- *What if two answers tie in mean rank?* — The one produced by more of the 8 samples wins (vote-count tie-break); if the votes also tie, the alphabetically first (lower-cased) answer — arbitrary but deterministic (final_method_eval.py, top[argmax(votes)]). Rank ties occurred on 0.58% of questions for the September probe (AUDIT_2026-09-18.md, statistical-validity table); double ties: not measured.
- *Why does the model need two passes?* — It does not have to — the states could be captured while generating (shown for an earlier probe). In our experiments we re-read each answer in a separate pass.
- *Is the greedy answer one of the candidates?* — No. The 8 candidates are samples only; greedy is a separate decode and the baseline. Adding greedy as a ninth candidate has not been tested.
- *How often is there nothing to choose?* — 16.3% of the 36,869 questions have a single distinct answer (from 0.7% on VQA-Med to 56.6% on SLAKE); the method returns that answer, which may differ from greedy (numbers_review_part2.json open8_pools, from the feature-cache metas).
- *Which probes would you actually deploy?* — For evaluation there are two sets of 24, one per fold, each trained without the images it scores (ckpts/train/final_probe_2026-10-16/fold0, fold1) — 48 in total. Either set is deployable; a set refit on all images has not been trained or measured.

<sub>Source: worked_example_12231.json (computed with ckpts/train/final_probe_2026-10-16/fold0, the probes trained without this image); shipped recipe: FINAL_METHOD_2026-10-16.md, single T = 0.7</sub>

### 26 · The data: eight open-ended medical VQA benchmarks  ·  ~2 min
*On screen:* The table top to bottom: the three classic benchmarks, then the five that broaden coverage; then the three boxes on the right.

**Say:**

Our data: eight open-ended medical VQA benchmarks, 36,869 questions over 20,389 images, covering pathology,
radiology, endoscopy and chest X-ray. Three are the classic ones — PathVQA, SLAKE and VQA-RAD — with the official open-ended test
questions. The others broaden the coverage. Two have whole-sentence answers written by a language model, Kvasir and GEMeX, which
matters later for exact match.
Two practical points. Many questions share one image, so every split is made by image: all questions about an image go to the same
half. And our generator was trained on the training splits of four of these benchmarks — our VQA-Med cell is mostly that training
split — so its absolute accuracy there is not a fair measure of the model. Our comparisons are always the same model against itself,
so exposure affects both arms. The gain is largest on the four benchmarks that are not in Lingshu's supervised training data — which
is reassuring, but those four also differ in other ways, and one of them, OmniMedVQA, has reported text exposure for the base model,
so I do not claim this proves anything; the effect of exposure on our gain is not measured.
The probe's training data is the other image half of every set we score, including three closed-question sets and the VQA-Med 2019
test, plus about six thousand questions from the training splits of PathVQA, SLAKE and VQA-RAD and from Kvasir-VQA.

**If asked:**

- *How was OmniMedVQA made open-ended?* — Its multiple-choice options were removed; only modality and anatomy questions kept; the RadImageNet source dropped because RadImageNet-VQA is a separate benchmark of ours. Scores are therefore not comparable to published OmniMedVQA (MCQ) numbers.
- *Is the GEMeX cell the GEMeX test set?* — No — open-ended rows from the GEMeX-ThinkVG release (gold ≤ 12 words), not the 300-image GEMeX test. VQA-Med C4 is mostly the 2019 training split; the official 500-question test is scored separately.
- *What exactly is the training set?* — Per fold: the opposite image half of the 8 open benchmarks (81,331 / 81,554 rows) and of the 4 comparison sets (3,366 / 3,325 rows: closed VQA-RAD, PathVQA, SLAKE and the VQA-Med 2019 test), plus a fixed 6,029-question subsample of Kvasir-VQA and the PathVQA/SLAKE/VQA-RAD training splits (31,237 / 31,121 rows after removal). Any row whose image is in the scored half is removed (exact pixel match). Totals 115,934 / 116,000 (numbers_review_part2.json; reproduces final_eval_FINAL_cap320_T07.json folds.*.train_rows).
- *Why Lingshu-7B?* — An open 7B medical VLM whose published numbers MedEvalKit reproduces, so every claim is anchored to a public baseline. The MedGemma-4B replication was withdrawn (its sampled answers ran to the token cap); Qwen2.5-VL-7B, Lingshu's base, is the second generator.

<sub>Source: final_eval_FINAL_cap320_T07.json cells.*.all.{n,n_images}, folds.{0,1}.train_rows (row breakdown: numbers_review_part2.json, recomputed from the feature-cache metas, reproduces both totals); Lingshu Table 5 p.12; benchmark sources per PAPER_NOTES_GH</sub>

### 27 · Step 1 — sampling: 8 answers, one image encoding  ·  ~1.5 min
*On screen:* The three boxes on the left, top to bottom; then the cost table — point at the ratio column.

**Say:**

Step one in detail. For each question the model writes eight answers at temperature 0.7, with the prompt "Answer the
question with a short, specific phrase. Do not explain." The image is capped at about 250 thousand pixels, so it becomes at most 320
image tokens, and each answer is at most 64 tokens.
Sampling eight answers is cheaper than it sounds, because the serving engine reads the image and question once — the shared prefill —
and then continues eight times. In modelled arithmetic it is only 1.13 times greedy. But arithmetic understates the real cost: measured
on one A100, a question takes 174 milliseconds greedy and 477 milliseconds for eight answers, and energy goes from 34 to 123 joules —
about 2.7 times the time and 3.6 times the energy. Those measurements are on short VQA-RAD answers; on the benchmarks with sentence
answers it would be more, and we have not measured it there.
Identical answers are merged — only case and surrounding spaces are ignored — keeping the count of how many samples produced each; on
average there are 4.4 distinct answers per question.

<sub>Source: bestofn_vllm_2026-09-16.json (ms and J measured: NVML, 20 timed calls after 3 warm-ups, vqa_rad open items, max_new 32; 1.13× FLOP-eq); GFLOPs modelled = analytic FLOP model (free_head_2026-08-16.json null_tests.flop_model_rebuild) × forward-token ratio; 4.42 distinct = 162,885 rows / 36,869 questions (numbers_review_part2.json); dedup: extract_generator_hidden.py norm(); sampling: run_openvqa.py:186</sub>

### 28 · Step 2 — where the probe reads: the answer's own tokens  ·  ~2.5 min
*On screen:* The dashed teal box (the answer's tokens), then the three violet lines (blocks 18, 20, 22), then the formula at the bottom.

**Say:**

Step two: where exactly the probe reads. This is the sequence the language model sees: the system prompt, the image
tokens, the question, and then one candidate answer. Each horizontal line is the residual stream after one of the 28 blocks. The model
computes a hidden state at every position after every block, but the probe reads only two things: the positions of the answer's own
tokens — the dashed box — and only three blocks, 18, 20 and 22. At each of those three blocks it averages the answer's vectors, which
gives one 3,584-number vector per layer: that is the formula at the bottom. The answer comes after the image and the question, and
the model attends to everything before it, so each answer vector already mixes in the image and the question — that is how a probe that
reads only answer tokens is image-aware. For scale: one image token covers 28 by 28 pixels, so the cap gives at most 320 image tokens
— about 274 on average — and in this example the question is 13 tokens and the answer 4.
How we get these states: we take each distinct answer and feed it back through the frozen model after the image and question — teacher
forcing — one pass per answer. I have to be precise about one detail: in our runs this re-reading used a higher image budget than the
generation did — a one-megapixel cap, which our files call "full resolution", instead of 250 thousand pixels. That is a difference between
how we ran it and how it would be deployed. The direct test — re-extracting these states at the generation budget — is running now.
Indirect evidence: when we regenerated the samples at the one-megapixel cap, so that generation and features matched, the method gave
plus 0.0745 instead of plus 0.0790; matching did not help, but that test also changed the samples.

**If asked:**

- *Why the answer tokens and not the last token or the question?* — Measured: mean over the answer span ≥ last token at every layer from 14 up; adding question or image token means adds ≤ +0.0018 (below the refit floor). See the feature-choice slide.
- *How does a probe that reads only answer tokens see the image?* — Through causal attention: every answer position attends to the image and question tokens before it. Explicitly adding image- or question-token means added at most +0.0018 (judge, 3 seeds; head_visual_features_2026-09-01.json).
- *Exactly which positions are pooled?* — The answer's own tokens as the tokenizer splits the answer string, including a trailing period; no end-of-turn token (nothing is appended after the answer). extract_generator_hidden.py: the last n_ans_tok positions.
- *Why not capture the states while generating?* — That is the design. It was tested for an earlier (August, layer-21) probe: states read from model.generate()'s own steps vs a teacher-forced pass at the same resolution changed 24 of 2,345 picks, accuracy 0.0000 [−0.0021, +0.0021] (free_head_2026-08-16.json). Not yet repeated for the shipped probe inside vLLM.

<sub>Source: extract_generator_hidden.py (h_span = mean over the candidate's tokens; layers index output_hidden_states, 0 = embeddings); token counts: numbers_review_part2.json (Lingshu tokenizer), mean ≈ 274 image tokens at cap320 (free_head_2026-08-16.json); bound = max_pixels/784 (2×2 merge of 14-px patches); AUDIT_2026-09-18.md §4 (two resolutions); select_policy_cap320_vs_fullres_T07.json</sub>

### 29 · Which layers: 18, 20, 22 of 28 — upper-middle depth  ·  ~1.5 min
*On screen:* The violet rows on the left (blocks 18, 20, 22), then the five bar values on the right, then the ensemble line.

**Say:**

Why layers 18, 20 and 22? First, a precise definition: layer 18 means the vector after decoder block 18 of 28, so
these sit at about two-thirds to four-fifths of the depth. The bars show the gain over greedy for a probe trained on each single layer,
on the five benchmarks where this was measured: all between 0.101 and 0.104, so the differences are at noise level. The three-layer
ensemble is 0.105. Honestly, the ensemble is a hedge against choosing the wrong single layer rather than a real gain — on the full pool
it adds 0.0007.
The literature agrees that correctness information is spread over the upper half of the network. HSRM finds the upper layers best; MedProb
finds a later peak for a different, multiple-choice task.

**If asked:**

- *Were the layers chosen on test data?* — Yes, partly: the set was the argmax of 8 layer sets scored end-to-end on image half A, on 5 of 8 benchmarks (AUDIT_2026-09-18.md §5). All candidates lie within noise of each other, so the choice cannot have bought much. Image half B is out of sample for it: judge +0.0846 [+0.0729, +0.0965], lenient EM +0.0165 [+0.0026, +0.0299] (final_fold1_headline_2026-10-16.json).

<sub>Source: head_ens_width_2026-08-25.json macro.* (5 of 8 benchmarks, 5 seeds, August-era pooled probe, judge); head_final_stack_PVFIXED_2026-09-13.json pooled_ens − pooled_singlelayer; HSRM Fig. 3 p.6; MedProb Fig. 5 p.7</sub>

### 30 · Step 3 — the probe: one hidden layer, 918,017 parameters  ·  ~1.5 min
*On screen:* The figure bottom to top: input vector, standardise, the two trained Linear layers with GELU between, the score at the top; the dashed sigmoid box is training only. Then the two equations.

**Say:**

Step three: the probe itself. It takes the 3,584-number vector, standardises each dimension with the mean and
standard deviation from training, multiplies by a 256-by-3,584 weight matrix, applies the GELU non-linearity, and a final weighted sum
gives one number: a score. Higher means the probe believes more strongly that the answer is correct, but it is not a calibrated
probability — the loss up-weights correct answers, and on our example the correct answer's scores are all negative — and we only ever
use its order within a question. The sigmoid in the dashed box exists only inside the training loss.
That is 918 thousand parameters — tiny next to the 7-billion-parameter model it reads.
It is pointwise: it looks at one answer at a time and never compares answers directly; the comparison happens in the next step.

<sub>Source: ckpts/train/final_probe_2026-10-16/fold{0,1} (recipe.json: Linear(3584→256)-GELU-Linear(256→1)); 3584·256+256+256+1 = 918,017; standardiser: final_method_eval.py (std + 1e-6)</sub>

### 31 · Training the probe: labels, loss, data  ·  ~2 min
*On screen:* The loss formula in the teal box — point at the weight w₊; then the training-rows box.

**Say:**

How the probe is trained. For every training question we have the eight sampled answers, and each distinct answer is
labelled correct or incorrect by the judge model. The probe is trained with binary cross-entropy — the standard loss for predicting a
yes-or-no label — with the correct answers up-weighted, because only about one training answer in five is correct; each correct answer
counts about 3.8 times. It uses every answer, including questions where all answers are right or all are wrong, which ranking losses
have to throw away. The optimiser is AdamW with learning rate ten to the minus three and weight decay ten to the minus two, for 30
epochs at batch size 256, with no early stopping. Eight random seeds for each of the three layers gives the 24 probes.
The training data is the other half of the images of all eight benchmarks and of the closed-question comparison sets, plus about six
thousand questions from the PathVQA, SLAKE and VQA-RAD training splits and Kvasir-VQA — about 116 thousand answer rows per fold. Two consequences: the probe learns the training
judge's idea of correct, which is why we check with a second judge; and it needs labelled data from each benchmark, which I quantify
later.

<sub>Source: final_eval_FINAL_cap320_T07.json folds.*.train_rows; head_sweep.py:172-174 (pos_weight = (1−π)/π); π and row breakdown: numbers_review_part2.json; final probes fitted on T = 0.7 pools</sub>

### 32 · Step 4 — the readout: rank averaging with a vote tie-break  ·  ~1.5 min
*On screen:* Read one row of the grid — Large Bowel, 24 cells — then the mean column; then the formula.

**Say:**

Step four: combining the 24 probes. Each probe ranks the distinct answers of this question from worst to best — n is the number of
distinct answers — we divide by n minus one so ranks run from 0 to 1, and average over the 24 probes. In voting-theory terms this is a
Borda count. The answer with the highest average rank wins; if two
are exactly tied, the one more samples produced wins.
These are the real ranks for our example. Large bowel is ranked first by 20 of the 24 probes. Small bowel had five votes but is ranked
first by only four probes. Spleen is last everywhere. We use ranks rather than averaging the raw scores because each probe is trained
separately and their scores live on different scales: on this very question the spleen scores minus 37.4 under one layer-18 probe and
minus 7.2 under a layer-22 probe. The raw numbers are not comparable across probes, but the order is. When we tested averaging raw,
standardised or calibrated scores instead, all of them tied with rank averaging under both judges, so ranks are the scale-free
default, not a measured win.

<sub>Source: worked_example_12231.json (raw scores: candidates[*].probe_scores); readout: genframe_data.rank_avg (ties averaged, ÷ max(n−1, 1)), final_method_eval.py tie-break; adopted 2026-10-15 under a pre-specified rule (readout_variants_2026-10-15.json)</sub>

### 33 · 24 probes ≠ 8 samples — the probes work for any N  ·  ~1.5 min
*On screen:* Three panels, left to right: the sample column grows, the 24-probe grid stays the same.

**Say:**

A question that always comes up: we ship 24 probes and sample 8 answers — are they paired? No. The 24 are three layers
times eight training seeds; the eight seeds are just eight independent training runs. Each probe scores each distinct answer on its
own, so with four samples every probe scores at most four answers, and with sixteen at most sixteen. Nothing is dropped and nothing is
reused. N equal to four itself we have not measured with the probe.
And it works: the same probes gave plus 0.0167 when going from 8 to 16 samples, with gains on four benchmarks and no losses. On
RadImageNet the gain keeps rising up to 32 samples, although that 32-sample pool was drawn at a higher temperature. With the final
probes, sixteen samples gives plus 0.098 instead of plus 0.080 under the judge, and a little more under exact match too — at roughly
twice the generation cost.

**If asked:**

- *Were the probes trained on 8-sample pools?* — Yes — but the probe is pointwise and ranks are divided by (n−1), so nothing in it depends on N. The N = 16 and N = 32 results above are measured, not assumed.
- *What if you used one probe instead of 24?* — One layer instead of three: −0.0007 (head_final_stack_PVFIXED_2026-09-13.json, pooled_ens − pooled_singlelayer, September). Seeds: an August readout with 2 seeds gave selection efficiency 0.8120 vs 0.8283 with 8 (OPENTEXT_CORRECTIONS_2026-08-19.md §2) — but the same doc shows CPU thread count alone moves that number by up to 0.035, so it is indicative only. A single probe on the final recipe: not measured.
- *Is N = 16 better in every metric?* — Under the training judge (+0.0984 vs +0.0798) and lenient EM (+0.0140 vs +0.0121), yes in point estimate. The cross-family number for 16-sample pools (+0.0674) is computed on incomplete MedGemma labels, so it is not a valid comparison yet (final_eval_FINAL_cap320_MIX_T07_16.json).

<sub>Source: coverage_sc16_ci_ALL_2026-09-16.json (September probe, judge); budget_conversion_sc32_2026-09-17.json (RadImageNet N = 32 sampled at T = 1.0); final_eval_FINAL_cap320_MIX_T07_16.json in the main checkout's artifacts/ (MedGemma labels incomplete for 16-sample pools)</sub>

### 34 · Evaluation: scored only by probes that never saw the image  ·  ~1.5 min
*On screen:* The two fold rows: train on one half, score the other; then the green box on the right.

**Say:**

How we evaluate. Every image is hashed into one of two halves, A and B. In fold zero we train the probes on half B and score
half A; in fold one the reverse. So every one of the 36 thousand questions is scored by probes that never saw its image or any question
about it — an exact pixel match; near-duplicate images were not checked. This is called cross-fitting; in machine-learning terms it is
two-fold cross-validation grouped by image, and it means we train two sets of 24 probes, one per fold.
We compare against the same model's greedy answer, under five metrics. Note that the main judge also produced the training labels,
which is why the second judge, from another model family, matters most. Averages are macro — each benchmark counts one eighth — and
confidence intervals come from a bootstrap that resamples whole images, because several questions about the same image are not
independent; resampling individual questions would make the intervals too narrow.

**If asked:**

- *Why not just one held-out half?* — That was the September protocol (+0.0737 on one half, exact match a tie). Cross-fitting doubles the scored sample; the method also changed since. Lenient EM is significant pooled (+0.0121 [+0.0030, +0.0210]) and on half B alone (+0.0165 [+0.0026, +0.0299]; final_fold1_headline_2026-10-16.json).
- *Do the intervals include the randomness of probe training?* — No — the bootstrap holds the probes and the fold split fixed. Refit noise is about 0.003; fold-split variance is not measured. The macro CI is conditional on these eight benchmarks. Listed as a gap.
- *Was the image matching only exact?* — Yes: images are hashed on decoded pixels, so exact duplicates across sets fall into the same half and are removed from training. Re-encoded or resized near-duplicates are not detected (AUDIT_2026-09-18 appendix DI-5 lists perceptual-hash candidates, not resolved).
- *How does the judge decide?* — See the answer prepared for slide 11 (grading open answers).
- *How often do the two judges agree?* — On the 112,770 September training rows, the two judges give the same correct/incorrect label to 92.9% of candidates; Cohen's κ = 0.79, derived from that agreement and the two positive rates (0.194 vs 0.239) (judge_2x2_2026-09-20.json train_label_agreement_AB, train_label_pos_rate; κ in numbers_review_part2.json). Not recomputed on the final pools.

<sub>Source: final_eval_FINAL_cap320_T07.json (bootstrap: 10,000 resamples, clustered by image within benchmark); split: md5('nd'+pixel_md5) mod 2, fold f scores half f; run_judge.py (both judges)</sub>

### 35 · What it costs: cheap sampling, negligible probe, costly re-read  ·  ~1 min
*On screen:* Left chart (milliseconds), middle chart (joules), then the three boxes — the orange one is the honest caveat.

**Say:**

What it costs. Sampling eight answers with a shared prefill costs 2.7 times the time and 3.6 times the energy of greedy,
measured on short VQA-RAD answers. The probe itself is about 22 million multiply-adds per answer — negligible. The honest caveat is the
re-reading pass: as we ran it, every distinct answer was re-read at the one-megapixel cap, and that pass is not small — about 6.9 greedy
passes' worth of arithmetic per question. That figure comes from a different measurement setup, so I do
not add the two into one total. By design those states can be captured while generating, at no extra cost, and we showed for an earlier
probe that this gives the same picks; the re-read could also stop at block 22. We have not measured the whole system end to end.

**If asked:**

- *Total cost as run vs greedy?* — Not measured end to end, and the two components come from different harnesses: sampling with vLLM's shared prefill 1.13× FLOP-eq (modelled), 2.74× latency, 3.59× energy on 20 VQA-RAD questions (bestofn_vllm_2026-09-16.json); the HF re-read 6.90 greedy-pass equivalents per question (free_head_2026-08-16.json, 3 benchmarks). Because the accountings differ, I do not add them into one number. The re-read pass was never timed.

<sub>Source: bestofn_vllm_2026-09-16.json measured.gen1/gen8 (20 vqa_rad open items, max_new 32; vLLM shared prefill); probe = 918,017 multiply-adds per answer per probe (by design); free_head_2026-08-16.json DELIVERABLE_3_cost (HF accounting: passes_per_question 3.81 on 3 benchmarks, per_forward_flopeq.head_extraction_fullres_DEPLOYED 1.81; capture test)</sub>


---

## IV · Why

### 36 · (section divider)  ·  ~0.5 min
**Say:**

Part four: for every design choice, what we tried instead and why we chose what we did. Some of the evidence comes from
earlier versions of the probe, so here is the timeline. In July the project used a fine-tuned LoRA verifier. In August we built the
first probe on a single layer, then pooled the training data. In September that became the pooled probe, version two, scored on one
image half. The October probes, cross-fitted on both halves, give every headline number. When a slide says July, August or September,
it is evidence from an earlier version.

### 37 · Design decisions at a glance  ·  ~1.5 min
*On screen:* Read the table row by row only for the rows I name: selector, verifier type, label source, temperature. The two right-hand columns are the two metrics.

**Say:**

This table is the summary of part four; the next slides give the evidence for each row. Two things to notice first.
One: almost every design choice was made on one half of the images, half A. Half B played no part in those choices, and on half B
alone the method gains plus 0.085 under the judge and plus 0.017 in exact match, both significant — so the choices did not inflate the
headline. Two: most verdicts are measured with the training judge, which also produced the labels, so I show exact match next to it
wherever it exists. Training a selector holds under both metrics: it beats majority vote, the answer prior, random and longest pick, and self-verification;
it ties a much larger untrained verifier; and against the published selector WED it wins on two benchmarks and ties on the third. Likelihood ranking and USC were not run. Some choices are ties
where we picked the simpler option; the temperature is the one where the two metrics disagree. One small surprise in the label-source
row: under exact match, a probe trained on the other judge's labels does a little better — a reminder that every judge has its own taste.

**If asked:**

- *Which design verdicts survive exact match?* — Measured in both: probe vs majority vote (EM +0.0136 [+0.0053, +0.0216]); readout tie-break (EM +0.0006, significant); cap320 vs the 1-MP cap (EM tie, +0.0122 vs +0.0111); probe vs LoRA verifier (EM loss −0.0145, earlier probe); temperature (EM prefers 0.2–0.4). Judge-only: token reading, layers, width, pooling. The final method is EM-significant pooled (+0.0121) and on half B alone (+0.0165).
- *Which judge's labels should train the probe?* — Tested at the September stage (layer 20, 3 seeds): a probe trained on MedVLThinker labels and graded by MedGemma gains +0.0658; trained on MedGemma labels and graded by MedVLThinker +0.0565; consensus labels +0.0703 / +0.0733. Under lenient EM: MedVLThinker labels +0.0026, MedGemma labels +0.0101, consensus +0.0033, soft +0.0057 (no intervals). The choice is at the noise floor; 7–30% of a judge-measured gain is judge-specific (judge_2x2_2026-09-20.json, judge_consensus_2026-09-20.json).
- *Why 8 seeds?* — Never swept. 8 seeds ≥ the best single seed (CHEAP_VERIFIER_ON_7B_2026-08-16.md §6). The intervals hold the probes fixed, so refit variance (≈ 0.003) is not in them.

<sub>Source: final_fold1_headline_2026-10-16.json policies.T07.macro; extra_numbers.json probe_minus_vote_em; free_head_2026-08-16.json; verifier_32b_gpu.json; wed_vs_probe_2026-10-16.json; readout_variants_2026-10-15.json; judge_2x2_2026-09-20.json, judge_consensus_2026-09-20.json (layer 20, 3 seeds); temp_crossfit_2026-10-15.json; select_policy_cap320_vs_fullres_T07.json; final_eval_FINAL_cap320_MIX_T07_16.json</sub>

### 38 · Why train a selector at all  ·  ~2 min
*On screen:* The top two dots first (probe, majority vote); then the rows below the dashed September line; then the orange answer-prior box.

**Say:**

Why train anything? Every untrained way of choosing that we ran on these pools loses to greedy or to the probe. Majority vote loses to greedy. Random
choice or the longest answer both lose. Asking Lingshu itself whether its answer is correct did worse than picking at random when we
tried it.
One baseline needs care: choosing the answer string that is most often right in training. Over all benchmarks it recovers only about
a quarter of the gain, and its interval includes zero. But per benchmark it is uneven: on OmniMedVQA and RadImageNet, where the
questions ask for a modality or an organ from a small vocabulary, the prior alone gets about two-thirds of the way. So on those two, a
large part of the gain is knowing which answers are usually right, and the probe adds the rest; under exact match on RadImageNet the
prior ties the probe. I say this plainly because those are the two benchmarks the exact-match win rests on.
A model four and a half times bigger is a better zero-shot verifier than the small one — size does help — but untrained it only tied a
small trained verifier. That is not in conflict with Verification Mirage: they measure whether a model accepts its own wrong answers when
it re-checks them; we measure best-of-eight selection accuracy, with the 32B judging the 7B's samples. Both agree that training, not
prompting, is what makes a selector work.
The probe gains plus 0.080, and on exactly the same pools it beats majority vote by 0.089 under the judge and by
0.014 in exact match, both significant.

**If asked:**

- *Doesn't the answer-string prior show the probe just memorises common answers?* — Partly, on two benchmarks. Overall the prior recovers ~25% of the September probe's judge gain (+0.0185 vs +0.0737, interval includes 0). Per benchmark (judge, image half A): OmniMedVQA 60%, RadImageNet 67%, GEMeX 45%, Kvasir-x1 23% — and the probe is still significantly ahead of the prior on all four (e.g. RadImageNet +0.0398 [+0.0219, +0.0578]). Under lenient EM on RadImageNet the prior ties the probe (+0.0050 [−0.0080, +0.0179]). (audit_stats_baselines_2026-09-20.json per_benchmark; shares in numbers_review_part2.json answer_prior_share; the prior is leave-one-out and optimistic.) On gold answers never seen in training the probe still gains +0.092 judge (but loses in exact match) — see the 'does it memorise' slide.
- *Mirage says a bigger verifier does not help; you say the 32B helps. Which is it?* — Different questions. Mirage measures self-verification: whether a model accepts its own wrong answers when re-asked (same family, p = 0.782 for a larger verifier). We measure best-of-8 selection accuracy, with a separate 32B judging the 7B's samples zero-shot: +0.067 [+0.038, +0.095] over a 7B zero-shot verifier (July pools, EM/substring, n = 600; verifier_32b_gpu.json). Both agree that a trained small selector matches or beats an untrained large one.

<sub>Source: final_eval_FINAL_cap320_T07.json; extra_numbers.json probe_minus_vote_*; audit_stats_baselines_2026-09-20.json summary_by_currency.{judge,lenientEM}.* (benchmark_t_interval_df7) and per_benchmark.* (shares: numbers_review_part2.json answer_prior_share); PROJECT_RETROSPECTIVE N32; verifier_32b_gpu.json pooled.deltas_vs_32B</sub>

### 39 · Why a probe on hidden states instead of a fine-tuned verifier model  ·  ~2 min
*On screen:* The first table (accuracy), then the two cost tables — the bottom one is in the HF accounting.

**Say:**

Before the probe, this project used a fine-tuned verifier: Lingshu-7B with a LoRA adapter, asked "is this answer
correct". On the same questions, an earlier single-layer probe was better under the judge and worse under exact match, so on accuracy
it is a wash.
The decision was about cost by design. The LoRA verifier re-reads the image and runs the whole 7-billion model once per candidate:
generating eight answers and verifying them costs 3.7 times the latency and 5.8 times the energy of greedy, and the scoring alone is
1.7 and 2.8 times. And the fast serving engine cannot even run its visual adapter correctly.
As we actually ran it, though, the probe needed its own re-reading pass, and then — in the HF accounting, where all eight samples are
separate passes — it costs almost the same arithmetic as the LoRA verifier, about two percent less. The saving is real only once the
states are captured during generation, which we showed equivalent only for an earlier probe.
The probe is also cheap enough to train on all benchmarks, which turned out to be the biggest single improvement. A LoRA verifier
trained on the same pooled data was never built, so I cannot claim the probe is more accurate than that.

<sub>Source: free_head_2026-08-16.json DELIVERABLE_4_head_only.head_vs_incumbent, DELIVERABLE_3_cost.arms_fixed_bestof8.{bo8_head_TODAY,bo8_incumbent_LoRA}.flopeq_measured_geometry; bestofn_latency_energy_2026-08-03.json measured.{bo8_total,verify8,gen1}; CLAUDE.md §0 (vLLM LoRA landmine); HSRM Table 1 p.7</sub>

### 40 · Why read the answer tokens, with the answering prompt  ·  ~1.5 min
*On screen:* The highlighted top row of each table; then the 'against us' box.

**Say:**

Why read the answer's tokens? We compared reading the mean over the answer tokens, the last token, and the same span
when the model is instead asked to grade the answer. The mean over the answer, read while the model is answering, was best at
layer 21 — but these are optimistic in-sample numbers without intervals. The span beats the last token at every layer from 14 up, but at
the last layer the grading prompt is slightly ahead of the answering prompt, so I read this only as: the answer's own tokens carry the
signal. A cleaner cross-validation also puts the span ahead of the last token, 0.680 against 0.655. Adding the
question tokens or the image tokens on top changed the result by less than two thousandths — under the noise level from retraining —
so we keep one vector per layer. Two papers point the other way: SWIFT found learned token weights better than a plain mean, and CASE
found averaging harmful in a multiple-choice setting. Learned pooling is something we have not tried.

<sub>Source: verifarch_hidden_promptform_indomain_2026-08-04.json (in-eval CV, upper bound); _head_sweep_journal*.jsonl A_objbt_h256, B_L21_last (clean CV, Bradley–Terry; sd 0.016 / 0.009); head_visual_features_2026-09-01.json macro.*; SWIFT Table 8; CASE §3.2</sub>

### 41 · Why an MLP of width 256 — the probe-size plateau  ·  ~1.5 min
*On screen:* The linear dot against the four MLP dots; then the axis table on the right.

**Say:**

Why this probe size? We ran a sweep of about 150 runs, 128 distinct configurations. The one clear finding is that a linear probe is worse by
about three points of selection efficiency. Once there is one hidden layer of at least 128 units, nothing else we varied — width,
depth, dropout, training length, weight decay — moves the result by more than the noise. So we took a simple point on the plateau.
When we re-tested the sweep's best setting end to end, it gained two thousandths, inside the noise, so we kept width 256.
Several papers prefer linear probes: CASE, MedProb, and the probing-methodology literature warn that wide MLP probes can memorise. Our
situation differs — over a hundred thousand labelled rows instead of a few hundred — and our own linear probe was clearly worse, but a
control-task check would settle this properly. Two caveats: once, in July, the model this cross-validation picked was not the best on
evaluation; and LayerNorm in the probe and the learning rate were never tested end to end.

**If asked:**

- *Where does sd ≈ 0.013 come from?* — The fold-to-fold standard deviation of the 5-fold CV selection efficiency in the sweep journal (e.g. A_objbce_h256 sd 0.0133).
- *The sweep was on an older setup — does it transfer?* — It ran on the four-domain, layer-21 setup, so it informed rather than selected the pooled recipe — evidence of a plateau, not of an optimum. CV picked BCE/h256 in July and it scored 0.7507 on eval, below listwise at 0.7895 (VERIFIER_ARCHITECTURES_2026-08-04.md §6.3, unresolved). The one end-to-end re-test of the best cell gained +0.0021 (judge only, inside the ~0.003 refit floor).

<sub>Source: _head_sweep_journal*.jsonl tags A_objbce_h{0,128,256,512,1024}, REG_drop*, REG_ep*, REG_wd*, R2_d2_dr0.0, R2_normz_ln1 (150 runs, 128 distinct _key configurations; fold sd A_objbce_h256 0.0133); head_final_stack_08{ctrl,wide}_2026-10-15.json (judge, 0.0761 vs 0.0740); VERIFIER_ARCHITECTURES_2026-08-04.md §6 item 3</sub>

### 42 · Why BCE — and why we do not claim it beats ranking losses  ·  ~1.5 min
*On screen:* The table (BCE highlighted), then the red 'confounded' box — that is why there is no claim.

**Say:**

On the loss function, I want to be careful. In our sweep, binary cross-entropy scored higher than pairwise and
listwise ranking losses — but that comparison is not fair: a ranking loss can only learn from questions that have both a right and a
wrong answer, which is 39 percent of our rows, and it took three times fewer training steps. So we do not claim BCE is better. We keep
it for three reasons: it uses every row, it matches the yes-or-no label the judge gives, and our rank-based readout cancels the one
known weakness of pointwise losses — learning which questions are hard. The literature is split: SWIFT found BCE best of five losses;
HSRM found a pairwise loss better on math; Q-Probe found a within-question loss better on code. A ranking loss on the final recipe is
one of the experiments still owed.

**If asked:**

- *Is there any controlled evidence on the loss?* — Two data points, one each way, both from the July LoRA-feature era. For BCE: training BCE on exactly the restricted rows a ranking loss can use cost −0.0020 [−0.0116, +0.0075], while swapping the loss on those same rows cost −0.0817 [−0.1042, −0.0586]. Against: listwise scored higher on eval than the CV-chosen BCE head (0.7895 vs 0.7507). VERIFIER_ARCHITECTURES_2026-08-04.md §4.3, §6.3. No claim for the final recipe.

<sub>Source: _head_sweep_journal*.jsonl A_obj{bce,bt,listwise}_h256, F_hybrid1.0, BUD_bt_ep98, BUD_bce_ep30; OPENTEXT_CORRECTIONS_2026-08-19.md §3b; SWIFT Table 7; HSRM Table 9 (GSM8K); Q-Probe Table 1, Fig. 4 (MBPP)</sub>

### 43 · Why rank averaging with a vote tie-break  ·  ~1.5 min
*On screen:* The row marked with a tick — rank + vote tie — then the bottom row, probe-weighted vote: its violet EM dot is right, its two judge dots are left.

**Say:**

Why combine the probes by rank? We tested the alternatives on identical held-out questions, under three metrics, with a
rule written down before seeing the results: only adopt a change if it helps under both judges, significantly under at least one, and
does not significantly hurt exact match. Averaging raw scores, standardised scores, probabilities or calibrated probabilities all tie
under both judges; z-score averaging does gain a third of a point in exact match, but it fails the rule.
Breaking exact ties by vote count passed the rule — six hundredths of a point, but significant and free, and it passed a rule we wrote
down first — so it is part of the method. This test used the September probes on image half A.
One result I want to show rather than hide: weighting the votes by the probe's scores, which is the default aggregator in Snell et
al., improves exact match by 1.35 points while both judges come out slightly lower — ties, minus 0.004 and minus 0.006. It trades one
metric for another, so under our rule it was not adopted — but if exact match were the target, it would be the better readout.

<sub>Source: readout_variants_2026-10-15.json macro.*_minus_rank_{j,x,em} (rank_vote CI: judge [+0.00005, +0.00152], EM [+0.00003, +0.00154]), adoption_rule (September pooled probe, image half A, n = 18,452; not re-checked on the final probes)</sub>

### 44 · Why train on all eight benchmarks: the largest single lever  ·  ~1.5 min
*On screen:* The three bars left to right: 4 domains, left out, pooled — the left-out bar is no higher than the first.

**Say:**

The single biggest improvement in the project was what the probe is trained on. In this run, a probe trained only on
the four-domain set (the PathVQA, SLAKE and VQA-RAD training splits plus Kvasir-VQA) gains about 0.03; trained on the training halves of all eight benchmarks, about 0.07 — two and a half times as
much, and four times in a later September re-run. The same jump happens on Qwen2.5-VL.
But this comes with an honest limitation: if you leave a benchmark out of training, the probe does no better on it than the four-domain
probe. On the four benchmarks that have no training split in that set the gap is stark: left out plus 0.003, pooled plus 0.099. So the claim is about new questions and images from benchmarks we have labels for: a new benchmark needs some labelled
questions, although at inference the probe needs no dataset label. These numbers are measured with the training judge only, without
intervals. The cost of a new benchmark is modest: with zero labels it gains plus 0.0196 overall in a three-seed re-run, five of eight
benchmarks already gain and a sixth after 50 labels; and on the two hardest cases, RadImageNet and OmniMedVQA, about a hundred labelled
questions recover about half and two-thirds of the full gain.

<sub>Source: head_lobo_pooled_2026-08-25.json macro.* (chart, −0.0056, +0.0485); head_final_stack_PVFIXED_2026-09-13.json (+0.0182 → +0.0736, September, script only); head_price_from_lobo_2026-08-30.json summary.zero_shot_macro; TRANSFER_WALL_2026-08-21.md §12 (Qwen, head_final_stack_qwen pooled ensemble)</sub>

### 45 · Why N = 8 — and what more samples buy  ·  ~1 min
*On screen:* The teal RadImageNet line rising; the indigo VQA-Med line flat.

**Say:**

Why eight samples? More samples help — doubling to sixteen gains another 0.017 overall, with significant wins on
RadImageNet, Kvasir-x1, OmniMedVQA and GEMeX — but only on the benchmarks where the probe already selects well. VQA-Med shows the other case: with more samples a correct answer appears more often,
but the probe cannot find it, so accuracy does not move. More samples also give a selector more chances to be fooled. So eight is our
default, inherited from earlier work, where the cost was measured, and sixteen is an option if accuracy matters more than time. We never
tried fewer than eight with the probe; with the earlier LoRA verifier, four samples at a lower temperature did as well as eight, so
that is worth testing.

<sub>Source: coverage_sc16_ci_ALL_2026-09-16.json (judge); budget_conversion_sc32_2026-09-17.json cells.{radimagenet_open,vqamed_open} (September pooled probe; RadImageNet N = 32 sampled at T = 1.0); CHEAP_VERIFIER_ON_7B_2026-08-16.md §4 (N = 4: judge +0.00159 [−0.0051, +0.0083], EM +0.00825 [+0.0015, +0.0151])</sub>

### 46 · Why one temperature, T = 0.7  ·  ~2 min
*On screen:* The left chart: the indigo judge line peaks at 0.7, the violet EM line falls from 0.2; then the table's 'best' column; then the red box.

**Say:**

Why one temperature of 0.7? The left chart is the gain over greedy when every benchmark uses the same temperature.
Under the judge, 0.7 has the highest point estimate, though only by half a point over 1.0. Under exact match, 0.7 is third of four —
colder sampling is better there. So the best temperature depends on the metric, and 0.7 was already the project's default before we
measured this, so it was not chosen by looking at the test results.
The table shows that each benchmark has its own preference: the benchmarks where the probe selects well want hotter sampling, the two
where it struggles want colder sampling. Choosing a temperature per benchmark does help, by about 0.009 on images it was not chosen on, and our final-method
document of October 16 adopted it. We reversed that on October 18: it requires knowing which dataset each incoming question comes
from, which a deployed system does not know. Training the probe needs labelled data per domain, but the shipped probe then scores any
question without a dataset label, and we keep it that way. We also tried a version that needs no dataset label — two samples at each
of four temperatures — and its point estimate was lower than eight samples at 0.7 in every metric.

**If asked:**

- *Could the model choose the temperature per question, e.g. by imaging modality?* — Untested. Modality alone seems unlikely to drive it: GEMeX (chest X-ray) prefers T = 1.0 while VQA-RAD (which includes chest X-rays) prefers 0.2; the preference tracks how well the probe selects on that benchmark. A per-modality test (chosen on one image half, tested on the other) is planned, as is a per-question adaptive scheme.
- *What about mixing temperatures with 16 samples?* — MIX16 (4 samples each at 0.2/0.4/0.7/1.0) vs 16 samples at 0.7: judge +0.0927 vs +0.0984, lenient EM +0.0108 vs +0.0140 — lower again (point estimates; final_eval_FINAL_cap320_MIX_{MIX16,T07_16}.json in the main checkout's artifacts/; MedGemma labels incomplete for 16-sample pools).
- *Why is the 0.7 here +0.0736 and the headline +0.0798?* — Different probes and protocol: this curve is the September pooled probe (v2) on image half A; the table is the August v1 probe (+0.0773 at 0.7); the headline is the final cross-fitted method on both halves.
- *Your final-method document adopted the per-benchmark map (+0.0897), and it passed your own out-of-sample test. Why ship +0.0798?* — The map is real: +0.0091 judge [+0.0007, +0.0182], +0.0090 MedGemma [+0.0010, +0.0175] on image half B, which played no part in choosing it; EM a tie (tmap_vs_t07_2026-10-16.json). The difference is the inference-time input: the map needs a dataset label on every incoming question; the pooled probe does not. We ship the version with fewer inference-time assumptions and report the map as a domain-calibrated variant (backup).

<sub>Source: chart: temp_crossfit_2026-10-15.json macro.fixed_*_minus_greedy_{judge,em} (September pooled probe v2, image half A); table: head_temp_ensemble_2026-08-30.json cells.*.{T}.minus_greedy (August pooled probe v1); tmap_vs_t07_2026-10-16.json macro.fold1_out_of_sample; final_eval_FINAL_cap320_MIX_{T07,MIX8}.json (main checkout artifacts/)</sub>

### 47 · Why cap320 instead of the 1-megapixel cap  ·  ~1 min
*On screen:* The top table's first two rows; then the GEMeX row in red.

**Say:**

Why do we cap the image at about 250 thousand pixels? We measured the whole method at a one-megapixel cap as well — what our
files call "full resolution": its point estimate was slightly lower under the judge, a tie in exact match, and it uses 2.8 times the
vision tokens — four times on GEMeX. The main effect is GEMeX: its chest X-rays are around eight megapixels, and at the one-megapixel
cap the model's own greedy answers get four points worse. So the cheaper setting is not worse — we do not claim it is better, because
we did not compute a paired interval. One detail is still open — the probe's features were read at the one-megapixel cap — and an
experiment re-reading everything at the generation budget is running.

**If asked:**

- *Why does a 1-MP cap hurt a 7.8-MP image that is downsized either way?* — Not explained by a measurement. At cap320 a GEMeX X-ray becomes about 303 image tokens on average; at the 1-MP cap about 1,237; at native resolution it would be about 9,092 (image_geometry_all8_2026-10-15.json benchmarks.gemex_open.mean_tokens). So neither setting is native resolution. The model's greedy accuracy is 4 points lower with the longer image sequence; why is not measured.

<sub>Source: select_policy_cap320_vs_fullres_T07.json macro.{cap320,fullres}_minus_ref_greedy_{judge,em}; resolution_check_2026-10-15.json (7 cells, no radimagenet_open); image_geometry_all8_2026-10-15.json pooled_question_weighted (2.76×), GEMeX 303 → 1,237 tokens</sub>


---

## V · Results

### 48 · (section divider)  ·  ~0.2 min
**Say:**

Part five: results.

### 49 · Headline: significant macro gains under every metric  ·  ~2 min
*On screen:* the first row (cross-family judge), then the last column (image half B).

**Say:**

The headline. Averaged over the eight benchmarks, the method improves on the model's own greedy answer by 0.0710
under a judge from another model family, MedGemma-27B, which the probe never saw during training. That is the number I lead with. Under
our main judge, MedVLThinker-32B, the gain is 0.0798 — but that judge also produced the probe's training labels, so it is not
independent. Lenient exact match, which counts an answer right if it equals or contains the gold string, gains 0.0121; token overlap
0.0232; LLaVA-Med's recall metric 0.0324. All five intervals exclude zero.
Two things to read carefully. The column "benchmarks above zero" counts positive point estimates, not significant wins; per benchmark,
under the judge, six of eight are significant wins and two are ties. And the last column uses only image half B. Half B played no part in
the September choices — layers, probe size, loss — nor in the readout tie-break or the temperature decision; the October check that kept
the cap320 image budget used both halves. The gains on half B are, if anything, slightly larger. Exact match gains much less than the
judges, and per benchmark it loses on Kvasir, GEMeX and VQA-Med; I will explain why on the next slide.

**If asked:**

- *Isn't the +0.08 partly the probe learning the judge's taste?* — See the answer on the opening slide: the cross-family judge, which was never used in training, gives +0.0710 [+0.0622, +0.0797]; the 0.009 gap to the training judge matches the measured paraphrase drift (0.006–0.009).
- *Were hyperparameters tuned on the test data?* — The September choices (layers, width, loss, T) were made on image half A. Half B alone: judge +0.0846 [+0.0729, +0.0965], larger than half A (+0.0747). The October keep-cap320 check used both halves but kept the incumbent.
- *Is the exact-match gain real?* — Significant on these eight benchmarks. It depends on OmniMedVQA and RadImageNet: drop either and it becomes a tie (+0.0032 [−0.0074, +0.0136]; +0.0034 [−0.0071, +0.0136]). With benchmarks as the sampling unit it is [−0.0234, +0.0475], a tie. Source: numbers_review_part3.json (part3_numbers.py on final_eval_FINAL_cap320_T07_records.json; image-clustered bootstrap, 10,000 resamples).

<sub>Source: final_eval_FINAL_cap320_T07.json macro_over_open8.all.pick_minus_greedy_*; final_fold1_headline_2026-10-16.json policies.T07.macro</sub>

### 50 · Per benchmark: where it helps, where it ties, where exact match disagrees  ·  ~2 min
*On screen:* filled vs hollow markers; the indigo (judge) and teal (MedGemma) dots move together, the violet (lenient exact match) ones do not.

**Say:**

Per benchmark. The largest gains, 11 to 16 points under the judge and 10 to 15 under MedGemma, are on OmniMedVQA,
RadImageNet, GEMeX and Kvasir, and the two judges agree closely. Two benchmarks tie: VQA-RAD, which has only 200 questions so its interval
is wide, and VQA-Med. Neither tie is for lack of correct answers among the samples — I will show in two slides that the probe simply fails
to pick them there. Exact match disagrees on three benchmarks. On Kvasir and GEMeX the gold answers are whole sentences generated by a
language model, so a correct answer in different words fails string matching while both judges accept it. On VQA-Med the exact-match
loss is small but real. If you correct for testing eight benchmarks, all six judge wins remain, and the GEMeX exact-match loss becomes a
tie.

**If asked:**

- *You test 8 benchmarks × 5 metrics. Multiple comparisons?* — Bonferroni: the 6 judge wins survive even a 40-test correction (smallest lower bound SLAKE +0.0141). MedGemma-SLAKE does not at ×40 [−0.0046, +0.0912]. EM: GEMeX loss becomes a tie at ×8; Kvasir-x1 and VQA-Med losses hold. Source: numbers_review_part3.json (part3_numbers.py on final_eval_FINAL_cap320_T07_records.json; image-clustered bootstrap, 10,000 resamples).

<sub>Source: final_eval_FINAL_cap320_T07.json cells.*.all.{pick_minus_greedy,ci_pick_minus_greedy}_{judge,xjudge,em}; Bonferroni: numbers_review_part3.json (part3_numbers.py on final_eval_FINAL_cap320_T07_records.json; image-clustered bootstrap, 10,000 resamples)</sub>

### 51 · Fixes vs harms: what the selector changes, question by question  ·  ~1.5 min
*On screen:* the fixed and harmed columns, then the VQA-Med row.

**Say:**

A net gain hides two movements, so here they are separately, using the accounting from Hu's 2026 paper. "Fixed" means
greedy was wrong and our pick is right; "harmed" means greedy was right and our pick is wrong; "recoverable" means greedy was wrong but at
least one of the eight samples was right. Over all
36 thousand questions, the selector fixes 4,890 questions that greedy got wrong, and breaks 847 that greedy got right
— 149 of those could not have been avoided by any selector, because none of the eight samples
was correct. On most benchmarks harms are a few percent of greedy's correct answers. VQA-Med is the exception: there are 453
questions where greedy is wrong but a correct sample exists, yet the probe fixes only 139 and breaks 153 — 44%
of greedy's correct answers. That is not a shortage of correct samples; the probe cannot tell them apart there.

**If asked:**

- *Does the method ever make things worse?* — Yes, per question always some, and on VQA-Med net: 139 fixed vs 153 broken, with a significant lenient-EM loss −0.0137 [−0.0227, −0.0049]. There is no per-benchmark guarantee of never being worse than greedy.

<sub>Source: extra_numbers.json (from final_eval_FINAL_cap320_T07_records.json, judge); answer-changed rate: numbers_review_part3.json; accounting: Hu 2026 arXiv:2607.17531 Eq. 1–6</sub>

### 52 · Gain follows headroom; selection fails on VQA-RAD, VQA-Med  ·  ~1.5 min
*On screen:* the gap between the grey and teal bars, then between teal and blue, benchmark by benchmark.

**Say:**

This chart puts greedy, our pick, and the oracle side by side for each benchmark. The gap between the grey and teal
bars is our gain; the gap between the teal and blue bars is what a perfect selector would still add. The fair way to score a selector is the share of that available
headroom it captures: the gain divided by oracle minus greedy. Averaged over benchmarks we capture 0.44 of the headroom —
0.87 on OmniMedVQA, about two-thirds on RadImageNet, Kvasir and GEMeX, about a third on PathVQA and SLAKE, and nothing
on VQA-RAD and VQA-Med. Those two are not out of reach: VQA-Med still has 0.116 of headroom. The probe just cannot pick
the right sample there. Across benchmarks the gain follows the headroom closely, much more than it follows how high the oracle is.

**If asked:**

- *You said selection efficiency is 0.81 — which is it, 0.81 or 0.44?* — Different quantities. pick ÷ oracle@8, averaged over benchmarks, is 0.81; it is flattered because greedy alone already reaches most of the oracle (VQA-RAD scores 0.78 with zero gain). The share of the headroom captured, (pick − greedy)/(oracle − greedy), is 0.44. The second is the honest measure of the selector.
- *Why does it fail on VQA-RAD and VQA-Med?* — Not for lack of correct samples. On VQA-RAD 17 fixed vs 17 broken; on VQA-Med 139 vs 153. We have no diagnosis yet; the within-question AUROC (slide on gaps) is the measurement that would show whether the probe ranks at random there. VQA-Med is also mostly the 2019 training split, which Lingshu was fine-tuned on (FINAL_METHOD_2026-10-16.md §6.4).

<sub>Source: final_eval_FINAL_cap320_T07.json cells.*.all.{greedy,pick,oracle}_judge; headroom / correlations: numbers_review_part3.json (part3_numbers.py on final_eval_FINAL_cap320_T07_records.json; image-clustered bootstrap, 10,000 resamples)</sub>

### 53 · Does the probe just learn which answers are common?  ·  ~2 min
*On screen:* the prior's share per benchmark (top-right box), then the 'never seen' rows of the table.

**Say:**

A fair question is whether the probe simply learned which answers are common. Two checks. First, a baseline that scores
each answer only by how often that exact string was correct in training. Over all benchmarks it recovers about a quarter of the probe's
gain. But per benchmark it is uneven. On OmniMedVQA and RadImageNet, where many questions ask for a modality or an organ from a small
vocabulary, the prior alone gets about two-thirds of the way — 60% and
67% of the probe's gain. So on those two, a large part of the gain is knowing which
answers are usually right, and the probe adds the rest. Those are exactly the two benchmarks the exact-match macro win rests on. On GEMeX
the prior gets less than half, on Kvasir about a quarter, and on PathVQA, SLAKE and VQA-Med it is below greedy. Second, pooled over all
benchmarks, the judge gain is just as large on question
templates the probe never saw, and still clearly positive on gold answers it never saw. But I have to be careful: those groups come
mostly from different benchmarks, and inside PathVQA alone the gain on unseen templates is not significant. So what the evidence supports
is "not only an answer prior", not "no memorisation". The test that would settle it — swapping in a wrong image — has not been run yet.
And note the exact-match column: on unseen answers it is a loss, partly because those strata are the sentence-gold benchmarks where
exact match is unreliable.

**If asked:**

- *Is OmniMedVQA / RadImageNet just an answer prior, or a modality classifier?* — Largely, on those two: the leave-one-out answer prior reaches 60% / 67% of the probe's judge gain (prior +0.0944 vs probe +0.1583 on OmniMedVQA; +0.0827 vs +0.1225 on RadImageNet). September probe, image half A, audit_stats_baselines_2026-09-20.json per_benchmark; the prior is leave-one-out and optimistic. The probe adds the rest; the image-shuffle control that would separate image use from answer frequency is owed.
- *Why does the exact-match gain turn negative on unseen golds?* — Those strata are dominated by GEMeX and Kvasir-x1, whose golds are LLM-written sentences where string matching rejects paraphrases; so it is not clean evidence either way. The decisive controls (image-shuffle; Hewitt–Liang control task) are on the gap list.

<sub>Source: template_stratified_gain_2026-09-20.json pooled_micro, cells.pathvqa_open.{repeat,novel}; audit_stats_baselines_2026-09-20.json summary_by_currency.judge and per_benchmark.*.judge (September probe, half A; shares: numbers_review_part3.json, r2_part3_extra.py)</sub>

### 54 · A second training recipe of the same language model: Qwen2.5-VL-7B  ·  ~1 min
**Say:**

Does it work on another model? On Qwen2.5-VL-7B the same pipeline, in its September form, gains eight points under the
judge and is significant under every metric, including strict exact match. But Qwen is Lingshu's own base model, so this is the same
language model with different training, not an independent replication, and it uses the same judge. An attempt on MedGemma-4b was
withdrawn because its sampled answers were broken. A replication on a genuinely different model family is still owed.

**If asked:**

- *Is the Qwen result an independent replication?* — No: Lingshu's base model, the September recipe (single layer, 5 seeds, half A), the same judge. The +0.0814 here is the single-layer refit; the pooled ensemble in head_final_stack_qwen_2026-09-13.json gives +0.0820.

<sub>Source: replication_currency_2026-09-20.json qwen.macro.probe_minus_greedy_{judge,em,strict_em,token_f1}; AUDIT_2026-09-18.md §3</sub>


---

## VI · Comparison

### 55 · (section divider)  ·  ~0.2 min
**Say:**

Part six: how this compares with published work.

### 56 · How to compare honestly  ·  ~1 min
**Say:**

Before the numbers, how to compare honestly. The strongest comparison holds everything fixed except the selector:
same questions, same eight samples, same graders. We have one of those, against WED. The second level uses the same official test
split and the same scoring code, which tells you where our numbers sit, but the generators differ, so it is not a ranking. The third
level is everything else, which is context only. To show why this discipline matters: the very same Lingshu-7B model is reported at
61.9 on PathVQA in its own paper and 71.9 in an independent benchmark. The prompt, the grader, the answer format and possibly the
question set all differ — evaluation choices alone move the same model by about ten points.

<sub>Source: Lingshu Table 6 p.17, §4.2 p.16; Liu et al. 2025 Table 1 p.4 (Lingshu-7B PathVQA 0.7192) — PAPER_NOTES_GH G4</sub>

### 57 · Tier 1 — head-to-head with WED on identical candidates  ·  ~2 min
*On screen:* the probe − WED column, PathVQA first; then the red box.

**Say:**

The controlled comparison. We ran WED's own code on our candidates, so the only thing that differs is how the answer
is chosen. We score it under seven metrics; the table shows two, the judge and lenient exact match. On PathVQA the probe beats WED under
all seven, by about eight points under the judge. On SLAKE it wins under six of seven; under our lenient exact match it ties. On VQA-RAD
everything ties — with 200 questions that means we cannot tell, not that they are equal.
One result needs explaining: on PathVQA, WED is significantly worse than greedy decoding, even though its own paper reports gains there.
Two asymmetries matter. WED normally runs its own sampling loop until it has eight distinct answers; we gave it our eight samples instead,
and after its clean-up step it often had only one candidate left — on 392 of 645 SLAKE questions and
490 of 3,357 PathVQA questions — so its game never ran there. And WED read the image at our generation
budget, cap320, while our probe's features used the higher one-megapixel cap. To check that these do not explain the result, restrict to
the questions where WED's game actually runs: there the probe's margin is larger, not smaller —
9.5 points on PathVQA and 13.0
on SLAKE under the judge. Finally, in fairness: WED needs no training, ours needs labelled data.

**If asked:**

- *What are the seven metrics, and the other numbers?* — judge, MedGemma, lenient EM, WED's own EM, token-F1, WED's F1, LLaVA recall (wed_vs_probe_2026-10-16.json; verdicts in numbers_review_part3.json wed.verdict_per_grader). MedGemma and WED's EM, probe − WED: PathVQA MedGemma +0.0783 [+0.0647, +0.0918], WED's EM +0.0608 [+0.0497, +0.0722]; SLAKE MedGemma +0.0496 [+0.0216, +0.0775], WED's EM +0.0295 [+0.0048, +0.0546]; VQA-RAD MedGemma +0.0100 [−0.0479, +0.0663], WED's EM +0.0400 [−0.0053, +0.0826].
- *Where WED's game actually runs?* — Restricting to questions where WED kept ≥ 2 candidates, probe − WED under the judge: PathVQA +0.0949 [+0.0795, +0.1102] (n = 2,867) · SLAKE +0.1304 [+0.0694, +0.1917] (n = 253) · VQA-RAD +0.0000 [−0.0738, +0.0710] (n = 151) (image-clustered bootstrap, numbers_review_part3.json).
- *Did you beat WED, or a crippled WED?* — Same 8 candidates, the authors' code for scoring, game and stopping. Where WED's game runs (≥ 2 candidates) the probe's margin is larger (PathVQA +0.0949 [+0.0795, +0.1102] (n = 2,867) · SLAKE +0.1304 [+0.0694, +0.1917] (n = 253) · VQA-RAD +0.0000 [−0.0738, +0.0710] (n = 151)). None of WED's four internal variants (BDG-W, BDG-classic, verifier-only, generator-only) beats greedy on PathVQA (point estimates 0.270–0.286 vs greedy 0.314; review/scr/wed_variants.py — no CIs, not on a slide). A 'WED as published' arm with its own sampling loop is owed.
- *Why is WED below greedy on PathVQA when its paper reports gains?* — Different operating regime (our fixed 8 samples, often 1–2 distinct candidates after its canonicalisation), a different generator (Lingshu-7B vs its 2–8B general VLMs) and lower image resolution in its scoring pass. We report it as measured and do not claim WED is broken in general.

<sub>Source: wed_vs_probe_2026-10-16.json cells.*.{probe_minus_wed,wed_minus_greedy}_* (T = 0.7 pools; authors' code, src/exp_2026_10/wed/wed_select.py); game-active subset: numbers_review_part3.json; WED Table 1 p.8</sub>

### 58 · Tier 2 — LLaVA-Med's metric on the full official test splits  ·  ~1.5 min
*On screen:* the two bottom rows of the table (greedy vs + probe), then the claim box.

**Say:**

Second level: LLaVA-Med's own metric, computed with their published code, on the full official test splits. Under
their metric our method improves open-question recall significantly on PathVQA and SLAKE; on VQA-RAD it ties. Closed yes/no accuracy
rises significantly on PathVQA — but I want to be precise: the probe's training data included the other image half of the PathVQA closed
questions, 2,246 and 2,123 rows in the two folds, so this is an in-domain gain, not transfer from open questions. SLAKE closed dips
slightly, a tie. Where do we sit? LLaVA-Med fine-tuned separately on each dataset is higher on all six columns, so this is not a ranking —
different generator, different training. The test counts also differ: their VQA-RAD has 179 open and 272 closed questions against our 200
and 251, and their PathVQA 6,761 against our 6,719. SLAKE English is the one split that matches theirs exactly.

**If asked:**

- *Did the probe train on closed questions?* — Yes. Each fold's training set includes the other image half of VQA-RAD-, PathVQA- and SLAKE-closed and of the VQA-Med 2019 test: 3,366 rows (fold 0) and 3,325 (fold 1) of 115,934 / 116,000. Image-disjoint from what is scored, so no leakage, but not transfer.

<sub>Source: final_eval_FINAL_cap320_T07.json full_official_splits.*, cells.*.all.ci_pick_minus_greedy_llava_recall, cells.pathvqa_closed; training rows: numbers_review_part3.json training_rows_by_source; LLaVA-Med arXiv:2306.00890 Tables 3–4; LLaVA-Med v1.0.0 eval code imported verbatim</sub>

### 59 · Tier 2/3 — ImageCLEF VQA-Med 2019 and Lingshu's published numbers  ·  ~1.5 min
**Say:**

Two more anchors. On the official 2019 VQA-Med test, the probe neither helps nor hurts — both changes are ties — and
strict string accuracy is far below the best 2019 team, because those systems were trained to output the challenge's fixed vocabulary
while our model answers in free text. On the right, Lingshu's own published scores. It is tempting to put our numbers next to theirs,
but they grade open questions with GPT-4.1 and closed ones by rule, their SLAKE includes the Chinese questions, and they use a different
image resolution, so only our two rows are comparable to each other. I will not claim we match the 32-billion model.

<sub>Source: final_eval_FINAL_cap320_T07.json cells.vqamed2019_test, full_official_splits; VQA-Med 2019 overview (CEUR Vol-2380) p.1, p.6 (best team 0.624); Lingshu arXiv:2506.07044v4 Table 6 p.17 (= v5 Table I), §4.2 p.16</sub>

### 60 · Tier 3 — medical test-time methods, each in its own units  ·  ~1.5 min
**Say:**

For context, other medical test-time methods in their own units, grouped by baseline. Best-of-Evidence, with a
235-billion-parameter judge and sixteen samples, gains under one point over majority vote; the matching number for us, also over majority
vote, is 8.9 points under our judge — but on our short free-text answers majority vote is already
0.9 points below greedy, so this delta mostly shows how weak voting is here, not that we beat Best-of-Evidence. WED's own paper reports between minus one and plus five points over greedy on small
models. MedProb's appendix test is a hundred questions. Over greedy we gain 7.1 points under the cross-family judge. But these
use different models, graders and question sets, so I present this as context, not as a ranking — the only controlled comparison is the
WED head-to-head.

<sub>Source: BoE Table 2 p.9; WED p.5, Table 1 p.8; MedProb Table 14 p.27; Mirage Tables 11–15; ours final_eval_FINAL_cap320_T07.json and extra_numbers.json (probe − vote, paired image-clustered bootstrap)</sub>


---

## VII · Limits

### 61 · (section divider)  ·  ~0.2 min
**Say:**

Finally, what we cannot claim, and what comes next.

### 62 · What the evidence does not support  ·  ~2.5 min
*On screen:* left column (what we cannot claim), then right column (costs and risks).

**Say:**

What the evidence does not support. First, the mechanism is not ours: probes on hidden states used as best-of-N
verifiers already exist in text models, and MedProb showed one on a hundred medical questions. Second, it is not domain-general: every
one of our eight benchmarks contributed labelled training questions, and a benchmark left out of training gains no more than a probe
trained on the four-domain set. Third, it is judge-sensitive: the gains are much larger under the judges than under exact match, and the
main judge is also where the training labels came from — which is why I lead with the cross-family judge. Exact match loses on Kvasir
and VQA-Med. Fourth, no human has rated this method's answers. The earlier checks were an LLM adjudicator on SLAKE and VQA-RAD and a small
hand-checked sample on PathVQA, all on older July outputs, and nothing on the other five benchmarks. Fifth, one model family: Lingshu and
its own base model; a MedGemma-4b attempt was withdrawn.
On the right. Cost, as run, comes from two different measurements that I do not add together. Sampling eight answers with vLLM's shared
prefill costs 1.13 times greedy in compute; the measured latency and energy, 2.74 and 3.59 times, come from twenty short VQA-RAD
questions, not from the sentence-answer benchmarks. Separately, the re-read of each distinct answer at the one-megapixel cap, in the
HuggingFace pipeline, costs about seven greedy passes. So "no extra forward pass" is true by design, not as we ran it; end-to-end cost and
memory were not measured. Where selection fails, on VQA-Med, the method breaks more answers than it fixes. Our VQA-Med set is mostly data
the generator was fine-tuned on, and whether that changes the delta is not measured. And against Lingshu-32B greedy we do not claim to
match it: one loss, two ties and one win under the judge, on four benchmarks, with an earlier probe.

**If asked:**

- *Has any human checked these answers?* — Not on this method's picks. Earlier checks (July pools): an LLM adjudicator (claude_judge/verdicts_lingshu32b/) on SLAKE and VQA-RAD; on PathVQA, 50 judge decisions on direct (short) answers, 2 wrong: 0.04, Wilson 95% [0.011, 0.135]; the same audit found more judge errors on long reasoning-mode answers (up to 0.28 on rejections) and a 90-item disagreement sample (clear judge error 7.8% [0.0382, 0.1519]) — pathvqa_judge_audit.json; who labelled those is not recorded. A stratified human rating is owed.
- *How often do your two judges agree?* — On September training candidates (112,770 rows) the two judges agree on 92.9% of labels, Cohen's κ = 0.79 (judge_2x2_2026-09-20.json train_label_agreement_AB and train_label_pos_rate; κ from the marginals, numbers_review_part2.json judge_agreement). MedGemma says 'correct' more often (23.9% vs 19.4%). Not yet measured on the October final pools. On the final method the per-benchmark deltas agree in sign on 7 of 8 benchmarks (VQA-RAD: +0.0000 vs +0.0100).
- *What does it cost compared with greedy?* — Not measured end to end, and the two parts come from different harnesses, so I do not add them. Sampling 8 with vLLM's shared prefill: 1.13× FLOP-eq; 2.74× latency and 3.59× energy measured on 20 VQA-RAD open questions, ≤ 32 new tokens, one request at a time (bestofn_vllm_2026-09-16.json) — not measured on the sentence-answer benchmarks. The HF re-read: 3.81 passes × 1.81 = 6.90 greedy-pass equivalents (free_head_2026-08-16.json, 3 benchmarks). The probe itself is negligible. Capturing states during generation would remove the re-read.
- *What does 'full resolution' mean here?* — Not native resolution: a 1-megapixel cap (max_pixels 1,003,520, ≤ 1,280 image tokens) — 'full resolution' is only our file name. cap320 is max_pixels 250,880 (≤ 320 tokens). GEMeX images have a median of 7.8 MP, so both caps downsample them.

<sub>Source: literature/DOMAIN_GUIDE_2026-09-30.md §2.5 (branch worktree-lit-domain-package); AUDIT_2026-09-18.md; pathvqa_judge_audit.json; free_head_2026-08-16.json; bestofn_vllm_2026-09-16.json; extra_numbers.json; audit_stats_baselines_2026-09-20.json bigger_model_reference_32B</sub>

### 63 · How far the statistics reach  ·  ~2 min
*On screen:* the 'benchmarks as the unit' column: the two judge rows exclude zero, the exact-match row does not.

**Say:**

How far do the statistics reach? Our main intervals resample images inside each of the eight benchmarks, so they say
the gain is real on these eight benchmarks. A stricter question: if we drew a new benchmark of the same kind and labelled part of it for
training, would the gain hold? Treating the eight benchmarks themselves as the sample, both judge gains stay positive, but the
exact-match interval crosses zero. And that assumes each new benchmark comes with labelled training questions, as all eight of ours did;
without labels from the new benchmark, the leave-one-out result applies — no better than a probe trained on the four-domain set. The
exact-match gain also depends on two benchmarks: drop either OmniMedVQA or RadImageNet and it becomes a tie.
What our intervals hold fixed: the trained probes and the image split. Refitting the probes with other seeds or thread counts moves the
macro by up to about 0.003; resampling the training set or the split was not measured. And they cluster by image, not by question template.
On tuning: the September choices were made on image half A; half B alone gives a slightly larger gain. Weighting by question instead of by
benchmark makes the judge gains larger and the exact-match gain slightly smaller.

**If asked:**

- *Your CIs hold the trained probes fixed. What is the real uncertainty?* — Refitting with other seeds/thread counts moves macros by up to ~0.003 (AUDIT_2026-09-18.md §5). Resampling the training set or the image split was not measured. Fold to fold (different probes, different questions): +0.0747 vs +0.0846, the same order as the CI width.
- *Could the 4 extra training domains leak test images?* — Exact pixel-md5 overlaps exist only between VQA-RAD-train and VQA-Med / VQA-RAD-closed / VQA-Med-2019-test, and are removed (261 / 377 rows per fold). Near-duplicates by perceptual hash were flagged (e.g. 100 SLAKE-train candidates, AUDIT appendix DI-5) and not resolved — owed.

<sub>Source: numbers_review_part3.json (part3_numbers.py on final_eval_FINAL_cap320_T07_records.json; image-clustered bootstrap, 10,000 resamples); refit floor AUDIT_2026-09-18.md §5</sub>

### 64 · Never tested — the honest gap list  ·  ~1.5 min
**Say:**

And the list of what was never tested. Several selectors we have not run on our data, most importantly ranking by the
model's own likelihood and combining that with the probe, which classical rerankers rely on. The likelihoods were not stored for the
shipped cap320 pools, but they were stored for the October one-megapixel pools on seven of the eight benchmarks, so this baseline can be
run on the CPU there. Also WED with its own sampling procedure.
Several design axes, including a ranking loss on the final recipe and learned pooling. And several diagnostics: a control task with random
labels, the within-question AUROC that CASE introduced, a test where the image is swapped, a human rating of our picks, and the variance
from retraining. Most of these are cheap to run on cached features, and they are first on the list.

<sub>Source: DESIGN_EVIDENCE.md §9 (21 gaps); PAPER_NOTES_* 'honest gaps' sections; review/R_results.md, R_papers.md</sub>

### 65 · Next steps  ·  ~1.5 min
**Say:**

Next steps. One experiment is running now. All features have been re-extracted at the same image budget the answers were
generated at, cap320, and the probes are being refit and re-evaluated with the same protocol. If it holds, the re-read no longer needs the
higher-resolution pass, and the WED comparison becomes symmetric in resolution.
After that, the cheap experiments on the CPU. First, ranking by the model's own likelihood, and likelihood plus probe — the log
probabilities were stored for the October one-megapixel pools on seven of eight benchmarks, so this needs no GPU. Then the within-question
AUROC, which should tell us why VQA-RAD and VQA-Med fail, the control task, the image-swap test, and the agreement between our two judges.
Then the temperature question you raised: whether the model can pick its temperature without knowing the dataset — by modality, although
GEMeX and VQA-RAD preferring opposite temperatures suggests modality is not the driver, or per question from the probe's own scores. And the
remaining baselines, a human rating, and the second model family that a reviewer would ask for.

<sub>Source: DESIGN_EVIDENCE.md §9; this deck's limits and gap slides; log-probs: src/labeling/run_openvqa.py cumlogprob_all (October 1-MP pools, RadImageNet missing)</sub>

### 66 · Summary — the claim, stated precisely  ·  ~1.5 min
*On screen:* the three numbers left to right, then the four lines.

**Say:**

To summarise in one sentence: a 918-thousand-parameter probe reading Lingshu-7B's frozen internal states, choosing among
eight sampled answers at temperature 0.7, improves open-ended medical VQA over the model's own greedy answer on eight benchmarks — by
0.071 under a judge from another model family that it never trained on, 0.080 under the judge that supplied its labels, and
0.012 in lenient exact match, all significant on these benchmarks. With benchmarks as the unit, the judge gains stay significant and
the exact-match gain does not — and that assumes each new benchmark comes with labelled training data. Under the judge it wins on six of
eight benchmarks and ties on two, where the right answer is among the samples but the probe fails to pick it. On identical candidates it
beats the training-free selector WED on PathVQA and SLAKE, and ties on VQA-RAD. The mechanism is not new; the setting, the breadth and the
care of the evaluation are. Thank you — I am happy to go into any part in detail; there are backup slides on every component.

**If asked:**

- *What generalises beyond this project?* — Training, not prompting, is what makes a selector work here: a trained 7B selector ties a zero-shot 32B verifier (+0.005 [−0.023, +0.032], n = 600, July pools, EM/substring; verifier_32b_gpu.json) and beats asking the model to verify itself; and selection gains follow the headroom only where the probe can find the right sample.

<sub>Source: final_eval_FINAL_cap320_T07.json macro_over_open8.all; wed_vs_probe_2026-10-16.json; numbers_review_part3.json</sub>


---

## Backup

### 67 · (section divider)
**Say:**

(Backup — only if asked.)

### 68 · Backup — every benchmark, every metric (shipped method)
<sub>Source: final_eval_FINAL_cap320_T07.json cells.*.all; headroom captured and micro: numbers_review_part3.json (part3_numbers.py on final_eval_FINAL_cap320_T07_records.json; image-clustered bootstrap, 10,000 resamples)</sub>

### 69 · Backup — per-benchmark temperature (adopted, reversed)
**If asked:**

- *Why ship the lower number?* — Because the +0.0091 comes from knowing which benchmark a question belongs to. The shipped method must answer an unlabelled question; choosing temperature from the question itself (modality, or the probe's own scores) is a next step.

<sub>Source: tmap_vs_t07_2026-10-16.json macro.fold1_out_of_sample; frozen_temperature_map_2026-10-15.json; final_eval_FINAL_cap320_Tmap.json; FINAL_METHOD_2026-10-16.md</sub>

### 70 · Backup — the full recipe
<sub>Source: src/training_methods/head_sweep.py:170–175; src/exp_2026_10/final_method_eval.py; src/labeling/run_openvqa.py:51; training rows numbers_review_part3.json</sub>

### 71 · Backup — data exposure and contamination
<sub>Source: Lingshu arXiv:2506.07044v4; Xu, Wu, Ryu arXiv:2606.10066; FINAL_METHOD_2026-10-16.md §6.4; AUDIT_2026-09-18 appendix DI-5</sub>

### 72 · Backup — references