# New directions, 2026-09-20 — what the literature leaves open, and where to attack
**Companion to `AUDIT_2026-09-18.md`.** Four literature sweeps (over 100 papers fetched and verified from their
arXiv / ACL pages; cards, query logs and "could not verify" lists in `AUDIT_2026-09-18_appendix/lit-*.md`)
plus six pilots run on data already on disk. Every project number names its artifact under
`results/cascade_methods/artifacts/`. Literature claims carry an arXiv id; anything a sweep could not
verify is marked there and is not used here.

---

## 0. The short version

The professor is right about the overlap, and it is worse than the 14 September prior-art note says: the
mechanism is published for text LLMs (arXiv:2402.14688, 2504.16760, 2505.12225, 2511.06209, 2608.17124,
2608.30841) **and** the "but not in a vision-language model" residual no longer holds as written
(arXiv:2605.28527 probes on frozen vision-language-action models used as a test-time selector;
2603.22492 a verifier on generator hidden states for best-of-N image generation; 2608.10835
generation-pass LVLM states). **Stop selling the mechanism.**

What the project has that nobody else has is *not* a mechanism. It is:

1. the **largest measured open-ended medical-VQA test-time-selection result in existence** — 18,452
   held-out questions, 8 benchmarks, now in **four currencies and two independent LLM judges** (the only
   published head-to-head found is MedProb's 100-item appendix, arXiv:2609.04336);
2. **198,378 candidate answers labelled by two judges from different model families**, with cached hidden
   states for every one of them — an asset for questions about *verifiers and judges* that no paper in the
   sweep could ask;
3. a frozen probe whose gain is **known to be mostly real** (+0.0679 under a judge it never saw;
   +0.057–0.066 cross-judge) and known *not* to be a template or answer prior.

Three directions follow, in the order I would run them. **D1 is Leo's fine-tuning idea in the one form the
literature says is both open and safe. D2 is a paper the project has already half-written by auditing
itself. D3 is the cheap generalisation that takes the work out of medicine.**

| | direction | new? | first experiment | pilot evidence already in hand |
|---|---|---|---|---|
| **D1** | **Probe-selected rejection-sampling fine-tuning (PRFT)** — move the best-of-N gain into the weights so *greedy* gets it | selector is OPEN in text *and* vision | 5–6 LoRA runs on Qwen2.5-VL-7B, ~1 GPU-day each | probe-filtered pseudo-labels 0.829 precise vs 0.679 for self-consistency, under a judge the probe never saw |
| **D2** | **"The currency decides the winner"** — evaluation-currency sensitivity of test-time scaling, and judge distillation in a latent verifier | OPEN in this form | one more judge + a 300-item human-rated subset | four currencies, a 2×2 and a consensus probe are done |
| **D3** | **Zero-label verifier transfer along a fine-tuning lineage** | OPEN for VLMs | reverse direction + a second lineage pair (CPU) | +0.0451 zero-shot → **+0.0603 with a label-free linear map** = 74 % of native. *2026-09-28: see the D3 status block below* |
| D4 | Tiny probe vs multi-billion-parameter multimodal reward models, general-domain VQA | OPEN | new generation on general benchmarks (GPU-days) | — |
| D5 | A predictive law for best-of-N gain, tested on our grid | OPEN (three untested theories) | CPU only | per-candidate scores dumped |
| D6 | Does a latent verifier look at the image? | partly done (DualRead) | ablated re-extraction (GPU-hours) | image-token features +0.0002 |

---

## 1. Ground truth after the audit — what any new direction can rely on

| | value | artifact |
|---|---|---|
| shipped probe, judge of record (MedVLThinker-32B) | +0.0737 [+0.0608, +0.0864] macro, 6/8 | `em_rescore_pooled_probe_2026-09-18.json` |
| same picks, cross-family judge (MedGemma-27B-it) | **+0.0679**, 7/8 positive, 5/8 significant | `xjudge_rescore_medgemma27b_2026-09-20.json` |
| same picks, token-F1 / lenient EM / strict EM | +0.0156 / +0.0047 (tie) / +0.0103 (tie) | `em_rescore_…` |
| probe graded by a judge it was not trained on | +0.0565 to +0.0658 | `judge_2x2_2026-09-20.json` |
| consensus-label probe, worst case over two judges | +0.0703 | `judge_consensus_2026-09-20.json` |
| Qwen2.5-VL-7B native probe | judge +0.0814, EM +0.0276, F1 +0.0317 — all significant | `replication_currency_2026-09-20.json` |
| Lingshu probe applied zero-shot to Qwen states | **+0.0451** (native +0.0820; native 4-domain +0.0227) | `pilot_cross_model_transfer_2026-09-18.json` |
| … after a label-free ridge map Qwen→Lingshu space (9,688 unlabelled pairs) | **+0.0603** (+0.0152 over zero-shot) | `pilot_lineage_alignment_2026-09-20.json` |
| gain on template-novel / gold-unseen questions | +0.1162 / +0.0918 (micro, judge of record) | `template_stratified_gain_2026-09-20.json` |
| generator LoRA-SFT, greedy (3 benchmarks) | +0.0142 at 1× cost; +0.0374 with best-of-8 on top | `ckpts/cheapleg/scores_*/score_summary.json` |
| 7B + probe vs Lingshu-32B greedy (4 benchmarks) | judge +0.0101, EM −0.0305 | `audit_stats_baselines_2026-09-20.json` |
| N 8 → 16 | +0.0167, 4 WIN / 4 TIE | `coverage_sc16_ci_ALL_2026-09-16.json` |
| cost of best-of-8 (vLLM, batch 1, cap320) | 1.13× FLOPs, 2.74× latency, 3.59× energy | `bestofn_vllm_2026-09-16.json` |

**Not usable until repaired:** every MedGemma-4b sampled pool (broken decode); the "no extra forward pass"
cost claim (never run); anything that needs per-candidate log-probabilities (not stored).

---

## 2. What the literature says, question by question

**2.1 A probe on the policy's own activations as the RL reward — DONE in text, and it collapses.**
PAIR (arXiv:2605.17877) states the hypothesis verbatim and feeds GRPO; SR-GRPO (2512.02807) validates an
internal signal by best-of-N and then promotes it to the reward — our pipeline's shape; Latent-GRPO
(2601.08427) does it unsupervised. The evidence on what happens next is uniformly bad: SRT (2505.21444)
reports "sudden and complete performance collapse" under prolonged self-reward; "No Free Lunch" (2506.17219)
finds internal-feedback RL degrades below the starting model and gives "little improvement for
instruction-tuned models" — all of ours are; Co-rewarding (2508.00410) calls the single-view signal a
"self-consistent illusion". Only the multimodal + open-ended + judge-labelled combination is left, and that
is too thin to build on.

**2.2 Optimising against a probe that reads your own activations — DONE, and the safe region is mapped.**
Cundy & Gleave (2505.13787): GRPO against an activation detector reaches > 85 % evasion, **off-policy DPO
stays < 25 %**. RL-Obfuscation (2506.14261): token-level monitors are evaded, **pooled monitors are
robust** — ours is mean-pooled. Wehner & Fritz (2510.21531): refitting the probe after training recovers
its accuracy; ensembles are unnecessary. So: no gradient through the probe, no on-policy reward, a held-out
probe as the tripwire.

**2.3 The probe as the *selector* for rejection-sampling fine-tuning — OPEN.** RAFT, ReST, ReST-EM, STaR,
RFT, West-of-N, BOND, BoNBoN, ScPO all run this loop; the selector is always a rule, a text reward model
or self-consistency. No paper found uses a trained hidden-state probe as the selector, in text or vision
(eight query formulations; Q-Probe, 2402.14688, names it as future work and supplies the matched-label
probe-vs-LoRA table for *inference*). VLM self-improvement (NOPD 2607.23125, STIC 2405.19716, SIMA
2405.15973, SelTDA 2306.03932, Vision-SR1 2508.19652) uses no latent signal. MedProb's open-ended
extension is reranking at inference only.

**2.4 Reward for open-ended medical RL — a publicly acknowledged bottleneck.** MedVLThinker's own
conclusion (2508.02669): *"Extending RLVR to open-ended generation … is an interesting challenge."*
Lingshu (2506.07044) keeps answers string-checkable "while maintaining verifiability". Nothing found uses
model-internal signals for it.

**2.5 Evaluation currency and judge inheritance — OPEN in our form.** MEDIQA-EVAL 2026 (LREC clinical-NLP
workshop) asks which metric correlates with clinicians; it does not ask whether the metric changes *which
test-time method wins*. 2607.01103: the best LLM judge reaches physician-level agreement (κ 0.694 vs
0.709) on open-response medical QA. 2606.19544: exact-match agreement overstates judge validity; report κ.
2505.19176 names "teacher preference bias" in *generative* proxy judges; 2603.12520 shows a judge's global
agreement captures only 21 % of oracle selection gain. **Nobody has shown a latent probe inheriting its
judge, and no paper reports best-of-N gains in several independent currencies as protocol.** The harness
standard is GPT-4.1 (`MedEvalKit/utils/utils.py:294`; Lingshu paper), text-only with the gold.

**2.6 Generalisation and transfer of correctness probes.** "Geometries of Truth Are Orthogonal Across
Tasks" (2506.08572) is our LOBO result for bare true/false statements, including that probe *mixtures* do
not fix it — so "per-benchmark" is a known property, and nobody has built the few-shot adaptation method.
Lineage transfer: "Task Matrices" (2512.14880) learns linear maps between base and fine-tune hidden states
but never for a verifier; 2507.00239 shows base→instruct probe transfer for refusal; cross-*architecture*
transfer fails (2606.20225). CASE (2608.17124) transfers only within a model and already names the
pooled-AUROC leakage we still have. **Our 55 % zero-shot survival across a fine-tune is an unpublished
kind of number.**

**2.7 Adaptive N.** Weitzman / Pandora's-box stopping for best-of-N is published with an *external* reward
model, text-only (2510.01394, 15–35 % fewer generations). At N = 8 our FLOPs are already 1.13× and the
real cost is latency, which sequential stopping makes worse. Worth doing only at N ≥ 16.

**2.8 General-domain multimodal reward models.** VisualPRM, LLaVA-Critic, Skywork-VL-Reward (2505.07263 —
built on **Qwen2.5-VL-7B-Instruct**, the exact backbone we have a native probe for), IXC-2.5-Reward. No
paper compares a < 1 M-parameter probe with these for best-of-N on open-ended general VQA.

**2.9 Predictive laws for best-of-N gain.** Three untested-against-each-other theories: verifier-ROC
geometry (2507.12399), cheap-statistics ridge (2606.02981, ρ = 0.90), oracle-gap decomposition
(2607.17531). None tested on a VLM.

**2.10 Does a verifier use the image?** DualRead (2609.06419) ships a counterfactual-image metric and a
black-image baseline for *confidence*; Verification Mirage (2605.10850) finds medical self-verifiers
"under-attend to image evidence"; ReLope (2603.24787) finds vision tokens degrade correctness-probe
separability. Open: how much of a latent *selector's gain* survives removing the image.

**2.11 Steering along a correctness direction.** ITI reports a truthfulness–helpfulness trade-off; a 2026
specificity critique (2602.06256) generalises it. No verified positive on QA accuracy. **Not recommended.**

---

## 3. The directions

### D1 — PRFT: probe-selected rejection-sampling fine-tuning  ⭐ Leo's idea, in its open-and-safe form

**Pitch.** Sample 8 answers per training question (done), let the frozen probe pick one (CPU, minutes), LoRA-
fine-tune the generator on (image, question, picked answer), then evaluate **greedy** decoding on the
held-out half. The question is exactly Leo's: *does the best-of-N gain move into the weights?* No gold
answer, no judge call and no external verifier is used at fine-tuning time.

**Why this form.** The on-policy version (probe as RL reward) is done in text and collapses (§2.1); the
off-policy version is the part nobody has done (§2.3) and is the regime measured as safe (§2.2): the probe is
applied once to a fixed pool from the pre-update policy, no gradient flows through it, and it is mean-pooled.

**Evidence already in hand.**
- *The selector is cleaner than the literature's default.* Keeping the most confident 25 % of held-out
  questions, probe-picked pseudo-labels are **0.829** precise under MedGemma-27B — a judge the probe never
  saw — against **0.679** for the self-consistency filter; at 50 %, 0.726 vs 0.576
  (`pseudolabel_precision_2026-09-20.json`). Under lenient EM the two tie (0.594 vs 0.606), so this must be
  judged in two currencies.
- *Fine-tuning and verification compose.* The 2026-08-11 LoRA-SFT run: greedy +0.0142, best-of-8 +0.0229,
  both +0.0374 (three benchmarks, LoRA verifier).
- *A probe survives a fine-tune.* 55 % of the gain transfers Lingshu→Qwen zero-shot, so the selector does
  not go fully stale after one round.

**First experiment (2 × A100-80 GB).** Generator Qwen2.5-VL-7B-Instruct (largest clean gain, the
non-medical control, solid in every currency). Six arms, identical LoRA recipe (r = 16, vision tower
frozen, 1 epoch, ≥ 3 seeds each — the project's own rule), same training questions:
1. untouched greedy; 2. **random-pick RFT** (the decisive control — NOPD-style plain self-distillation is
strong); 3. self-consistency-selected RFT; 4. **probe-selected RFT**; 5. LoRA-SFT on gold answers (the
supervised reference); 6. judge-selected RFT (the expensive upper bound). Plus best-of-8 on the untouched
model (the incumbent) and **best-of-8 on top of arm 4** — if the probe's headroom collapses after PRFT, the
gain moved into the weights. One extra arm gives D3 for free: **Lingshu's probes selecting Qwen's data.**
Roughly one GPU-day per arm-seed.

**Protocol that survives the currency problem.** Every arm in four numbers on identical outputs —
MedVLThinker-32B, MedGemma-27B, lenient EM, token-F1; the headline only where the independent judge and EM
(on the short-gold benchmarks: VQA-RAD, SLAKE, PathVQA, RadImageNet, OmniMedVQA) agree in sign.
pass@8 before and after (is it just sharpening?), mean answer length (is it the judge's register?).
Generate and extract at **one** resolution, with a stop-token check, and score LoRA adapters under HF or
verify them module-by-module first (the vLLM visual-LoRA landmine).

**Kills it.** PRFT ≤ random-pick RFT, or ≤ self-consistency RFT, under the independent judge *and* under EM
on the short-gold benchmarks. **Main risk.** The selection gain is only +0.0047 in EM, so PRFT may copy the
judge's phrasing into the weights and look like a win under the training judge alone — which converts the
result into D2, still publishable.

**What makes it a paper.** Not "the probe improves the model" (self-improvement is crowded). The matched-
label, matched-compute comparison: *for a fixed labelling budget, is a ~1 M-parameter activation probe a
better place to spend it than LoRA-SFT on gold, self-consistency filtering, or judge calls?* Q-Probe
answered that for inference in text; nobody has for fine-tuning, in any modality. It also answers a
bottleneck the medical-VLM RL papers name themselves (§2.4): a cheap reward for open-ended answers.

### D2 — "The currency decides the winner"  ⭐ the paper the audit already half-wrote

**Pitch.** One selection method, identical picks, 18,452 questions: **+0.0737 / +0.0679 / +0.0156 /
+0.0047** under judge A / judge B / token-F1 / exact match — a 15× range, with the *sign* flipping on two
benchmarks (Kvasir-x1, GEMeX). Then the mechanism: a verifier trained on one judge's labels scores 7–30 % higher under that
judge than under an independent one; training on the judges' consensus removes most of the gap. Deliverables:
(i) the first multi-currency best-of-N result in medical VQA, (ii) the first measurement of judge
inheritance in a *latent* verifier, (iii) a reporting protocol (cross-judge off-diagonal gain + κ + EM only
where golds are short), (iv) the released 198 k dual-judge labels.

**Already done (this audit).** Four currencies; MedGemma-27B re-label of all 198,378 pairs; κ per benchmark
(0.71–0.90); the 2×2; the consensus probe; the length analysis (it is *not* verbosity: pick-the-longest loses
−0.0294); the manual reading (EM fails on sentence golds; the judge is phrase-sensitive); the template
stratification.

**What is missing.** (1) A third judge — the harness-standard GPT-4.1, or an open non-Qwen non-Gemma model —
so "two judges agree because they are right" can be separated from "because they share a register";
(2) **a human-rated subset**: ~300 items stratified on judge disagreement, two raters — the one thing that
turns this from an LLM-vs-LLM comparison into a validity claim; (3) a phrasing-invariance test (paraphrase
the pick, re-judge); (4) the same table for Qwen (labels for B on Qwen pools: ~1 GPU-hour).

**Kills it.** Already partly tested: had the cross-family judge *failed* to reproduce the gain this would be a
judge-artifact paper; it reproduced 92 %, so the finding is "EM is the wrong currency for sentence golds, and
judge inheritance is real but bounded at 7–30 %". That is a solid workshop/short paper (MEDIQA-EVAL's venue
exists) and the evaluation section of D1 — not a flagship on its own. **Risk:** low; cost is mostly writing.

### D3 — A verifier that survives the model it was trained on: zero-label transfer along a fine-tuning lineage

> **Status 2026-09-28 (judge currency only — not yet a claim).** Run: (c) a "second pair", QoQ-Med-VL-7B —
> but it is **not independent**: 0.11 % weight distance from Qwen2.5-VL in the probed layers vs Lingshu's
> 3.2 %, so it re-measures pair 1 (`CHECK_2026-09-28.md`, d3-method P1). The pair-efficiency sweep first
> read as "the map needs >3,000 pairs to beat plain re-standardisation"; that was an artifact of shrinking
> the ridge toward zero. Shrunk toward the identity: **+0.0541 at 100 pairs, +0.0649 at 9,688**
> (`d3_ridge_identity_prior_2026-09-28.json`) — so onboarding a related model does look cheap. Writeup,
> bannered: `LINEAGE_TRANSFER_2026-09-28.md`. **Still open:** the reverse direction (a); a genuinely different
> fine-tune for (c); exact-match currency and paired CIs, both required before this is quoted.

**Pitch.** Lingshu's probes, unchanged, applied to Qwen2.5-VL-7B's states give **+0.0451** with zero Qwen
labels (4/8 benchmarks individually significant; the three large ones — Kvasir-x1, OmniMedVQA, GEMeX — keep
63–76 % of the native gain).
Make it a method: fit a linear map between the two models' hidden states on *unlabelled* prompts
("Task Matrices", 2512.14880, shows such maps exist) and ask how much of the missing 45 % it recovers. If a
verifier trained once on a base model serves every fine-tune of it, the onboarding cost changes from
"~100 labelled questions per benchmark per model" to "per benchmark, once per model family" — and this is
**not medical**: Qwen2.5-VL → Lingshu is just the pair we have.

**Already run (`pilot_lineage_alignment_2026-09-20.json`).** A ridge map from Qwen's standardised hidden space to
Lingshu's, fitted with **no labels** on 9,688 train-half rows where the two generators happened to give the same
answer to the same question (so both were teacher-forced on the same text), ridge strength chosen by alignment
error on a 10 % split: alignment MSE 0.81–0.91 → 0.14–0.15 per layer, and the frozen Lingshu probes on the *mapped*
held-out Qwen states give **+0.0603** macro (zero-shot +0.0451, Qwen-native +0.0820) — **41 % of the gap closed,
74 % of the native gain, zero Qwen labels.** OmniMedVQA +0.1291 → +0.1710 (native +0.1870); GEMeX +0.0872 →
+0.1028; Kvasir-x1 flat (+0.0804 → +0.0774).

**Adversarial novelty check (2026-09-20, arXiv listing search; gists read off the results page, abs pages NOT
individually fetched — verify before citing).** Transfer of activation-space objects between a model and its
variants is an active line in *safety / monitoring*: persona vectors moved from fine-tuned variants to base
models (2607.13162), a deception probe working zero-shot across model families (2606.17229), refusal steering
vectors across same-architecture variants (2603.13359), bidirectional activity transfer between networks
(2501.06164); LYNX (2512.05325) reuses an early-exit probe across tasks, not models. **None transfers a
*correctness verifier used for answer selection*, none is in a VLM, and none fits the map without labels from
coincidentally identical generations.** So D3 is adjacent-but-open; cite this line and position against it.

**Next.** CPU only, caches on disk: (a) reverse direction (Qwen probe → Lingshu states; needs one Qwen refit,
~25 min); (b) how few unlabelled pairs suffice (the pairs are free: they need both models' samples, not labels);
(c) a second lineage pair (any other Qwen2.5-VL-7B fine-tune on HF) — one pair is an anecdote.
**Kills it.** The reverse direction is ≈ 0, or a second lineage pair does not reproduce.
**Risk.** One lineage pair is an anecdote; needs a second pair to be a claim. Pairs naturally with D1's
cross-lineage arm (a probe trained on model A selects the data that fine-tunes model B).

### D4 — A < 1 M-parameter probe vs multi-billion-parameter multimodal reward models, general-domain VQA
The direct "not only medical" play. Skywork-VL-Reward sits on the same Qwen2.5-VL-7B backbone. Generate
8-sample pools on open-ended general benchmarks (OK-VQA / A-OKVQA direct-answer, TextVQA, DocVQA, ChartQA,
InfoVQA, MM-Vet), label with two judges + the benchmarks' own string metrics (which, unlike ours, are
standard and short-gold), train the probe, race it against the reward models at matched labels and measured
latency. **Cost:** GPU-days of new generation and extraction — do it *after* the regeneration pipeline is
fixed. **Risk:** the residual "VLM" novelty is already thin (§0); the contribution would be the comparison.

### D5 — A predictive law for best-of-N gain (CPU only)
Three published laws, never cross-tested, none on a VLM. We have per-candidate probe scores on 18,452
questions, two clean generators, Lingshu at four temperatures, N ∈ {8, 16} (32 on two benchmarks) and two
label sets. Fit each law on half the
(generator, benchmark, T) cells, predict the other half. It would also finally answer the project's own open
hole — *which benchmarks will the verifier help?* — using within-question AUROC and coverage rather than
benchmark identity. Small, cheap, a good section or a short paper. Requires fixing the pooled-AUROC leakage
first (report CASE's grouped AUC).

### D6 — Does the latent verifier look at the image?
Re-extract with the image blanked / swapped for a same-question hard negative (DualRead's protocol) and
measure how much *selection gain* survives — not AUROC. `extract_generator_hidden_ablated.py` exists and was
never run. GPU-hours. Position against DualRead; do not sell as new diagnostics.

### Not recommended
On-policy RL with the probe as reward (done; collapses; evasion). Activation steering (negative
literature). Adaptive N at N ≤ 8 (FLOPs already 1.13×; published with an external RM). More probe
architecture (seven variants lost; data is the lever). More training breadth (LOBO −0.0056; orthogonal
truth geometries are a known result). Anything abstention-shaped (CRITICAL RULE 6) — note that DualRead's
and 2607.01103's framings slide into deferral and must not be copied.

---

## 4. Order of work

**Before any new result (one pipeline pass, shared by everything):**
1. Regenerate the pools at full resolution (already decided 2026-09-17) **with a 10-question stop-token check
   per generator** and **capture the hidden states in the generation pass itself** — that closes the
   resolution mismatch and makes the cost claim true for the first time. Store per-candidate log-probs while
   there (they are free and unlock every logit baseline).
2. Label with **both judges as standard**; train on consensus labels; report the cross-judge number.
3. Regenerate MedGemma properly → the only cross-family replication.
4. 32B greedy on the four missing benchmarks (the first question any reviewer asks).

**Then:** D2's missing pieces in parallel with step 1 (CPU + a small human-rating task) → **D1 round 1**
(six arms + the cross-lineage arm) → D3's CPU alignment while D1 trains → D5 on the new pools → D4 only if
D1/D3 land and a general-domain paper is wanted.

## 5. Which paper

- **If D1 works:** *"A one-million-parameter latent verifier as the reward for open-ended medical VQA:
  matched-label comparison against gold SFT, self-consistency and judge-selected fine-tuning"* — method +
  D2 as its evaluation protocol + D3 as the zero-label variant. Medical venue or a general ML venue with
  Qwen as the lead model.
- **If D1 fails:** *"The currency decides the winner"* (D2) + the map the project already has (8 benchmarks,
  two dozen measured negatives, the LOBO limit) + D5. The negative D1 result belongs in it: *the best-of-N gain of
  a latent verifier does not move into the weights, and here is the currency analysis showing why.*

Either way the first paragraph concedes the mechanism and names its lineage; the contribution is the
measurement and the comparison, not the probe.

## 6. Verified reading list for these directions (ids only; cards in the appendix)
Probe-as-reward and its collapse: 2605.17877, 2512.02807, 2601.08427, 2505.21444, 2506.17219, 2508.00410.
Training against probes: 2505.13787, 2506.14261, 2510.21531, 2210.10760.
Rejection-sampling / BoN distillation: 2304.06767, 2312.06585, 2308.01825, 2407.14622, 2411.04109,
2402.14688. VLM self-improvement: 2607.23125, 2405.19716, 2405.15973, 2306.03932, 2508.19652.
Medical RL rewards: 2508.02669, 2506.07044, 2506.18254, 2601.18533.
Judges and currencies: 2607.01103, 2606.19544, 2505.19176, 2603.12520, MEDIQA-EVAL 2026
(aclanthology.org/2026.clinicalnlp-1.1), 2605.10850.
Probe generalisation and transfer: 2506.08572, 2512.14880, 2507.00239, 2606.20225, 2608.17124.
New collisions with the old residual claim: 2605.28527, 2603.22492, 2608.10835, 2609.04336, 2609.06419.
BoN laws: 2507.12399, 2606.02981, 2607.17531. Adaptive N: 2510.01394. Multimodal RMs: 2505.07263.
