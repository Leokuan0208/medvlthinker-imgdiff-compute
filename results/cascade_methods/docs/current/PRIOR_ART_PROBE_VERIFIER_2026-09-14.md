# PRIOR ART: has "a probe on frozen hidden states, used as a best-of-N verifier" been done?

**2026-09-14** · seven literatures searched in parallel, closest hits independently re-fetched and
adversarially re-read · supersedes nothing, but **materially changes how the open-text arm should be
positioned**

---

## CORRECTIONS 2026-09-20 — read before §3 and §4

*Source: `AUDIT_2026-09-18.md` §7 and its appendix `AUDIT_2026-09-18_appendix/lit-vlm-med.md`
(corrections C1–C7 there, reproduced here in full). This block corrects this file; §3's DualRead
facts are also fixed in place below.*

### A. The residual novelty sentence of §3 **no longer holds as written**

Three papers found on 2026-09-20 each break a different clause of *"no verified paper applies a
probe on generation-pass internal states as a best-of-N selector in a VISION-LANGUAGE model"*:

| id | what it does | which clause it breaks |
|---|---|---|
| **arXiv:2605.28527** — "What Frozen VLAs Already Know About Success" | linear probes on the frozen features of vision-language-**action** policies (OpenVLA, π0.5), used as a **test-time selector over sampled action prefixes**: push-plate success **26.7 % → 44.3 %** vs greedy | "vision-language model" + "best-of-N selector" — structurally this paper, with actions instead of text |
| **arXiv:2603.22492** — "Tiny Inference-Time Scaling with Latent Verifiers" (VHS) | a verifier on the intermediate hidden states of a Diffusion-Transformer image generator, scoring candidates without decoding to pixels, explicitly to avoid re-encoding each candidate: −63.3 % generation+verification time, −51 % FLOPs, −14.5 % VRAM, +2.7 % GenEval | the **economic** argument (dimension D), already published in a visual generative setting — with measured VRAM savings, which this project does not have |
| **arXiv:2608.10835** — "UniProbe" | a learnable detector over a frozen LVLM's **single-forward-pass** computational trace (image patches, query tokens, generated tokens), with a streaming variant that resamples hallucinated tokens **during generation**, at **1.06× latency** | "generation-pass internal states of a VLM, reused for free, used to change the output" |

**C1.** §3's residual claim is **wrong as stated**. The defensible narrow residual is: *no paper
applies a probe read from the generation pass, pooled over the candidate's own generated tokens, as
a best-of-N selector over **free-text answers** of a vision-language model, at the scale of 8
open-ended medical VQA benchmarks.* Every clause is load-bearing, and **a claim that needs five
qualifiers is not a contribution — stop selling mechanism novelty; sell the measurements.**

**C2.** §4 item 2's *"First application of generation-pass hidden-state verification to
vision-language / medical VQA"* is an **overreach**: MedProb Appendix H already reranks open-ended
medical VQA generations with a probe. Keep "generation-pass" and "at scale" in the same sentence as
the MedProb citation.

### B. DualRead and MedProb — five factual corrections (C3–C7)

- **C3 — DualRead backbones.** §3 says "Qwen3-VL and MedGemma". The paper (l.505) says
  **Qwen3-VL-2B-Instruct** and **MedGemma1.5-4B**, "fully post-trained on SLAKE-train with
  **GRPO**". The sizes and the RLVR framing both matter.
- **C4 — DualRead benchmarks.** §3 lists five. The paper (l.502) has SLAKE-test 1,061 in-domain
  plus **six** out-of-domain: VQA-RAD 600, PathVQA 1,000, PMC-VQA 1,000, OmniMedVQA 920,
  **ReXVQA 1,000**, **VQA-Med-2019 closed 64**.
- **C5 — DualRead pooling is not our pooling.** The mean over generated answer tokens is only a
  **query vector** for answer-conditioned visual pooling inside P-Visual; the post-answer
  correctness feature (P-State) is the **single closing token**. Remove "our pooling".
- **C6 — DualRead already ships the diagnostics we were going to present as new.** It has
  **CCG-AUC** (hard-negative image substitution) and a **black-image** baseline (BICR, l.1303).
  This project's image-ablation work must be positioned against these, not offered as new.
- **C7 — CASE already names our "pooled AUROC leakage".** CASE's headline mechanism finding is
  **question-identity leakage**: naively random-split probes look strong, within-question AUC
  collapses to 0.502 on LogiQA (+0.2 pp, p = 0.66), decision threshold ≈ AUC 0.60. Report
  **within-question / question-grouped AUC** or expect the objection.

**Verified correct in this file, re-checked:** CASE's "N = 16–20 at temperature 0.7" (l.192:
"16–20 candidates by nucleus sampling, τ = 0.7, top-p = 0.95, up to 512 new tokens"); MedProb's
"candidate fed back as input, 100 test examples, *an initial demonstration*" (l.2673, l.2696);
HSRM being text-only mathematics.

### C. Dimension (D) describes the design, not what was run

§0's dimension **(D) "reused from the generation pass at no extra forward pass"** — and the same
words at §3 ("it pays a teacher-forced replay rather than reading the generation pass", offered as
a *distinction* from DualRead) — **do not describe this project's code.** Features come from a
**separate teacher-forced HuggingFace pass** (`src/training_methods/extract_generator_hidden.py:437-439`)
at `max_pixels = 1,003,520`, while the candidates were generated under vLLM at `cap320 = 250,880`.
So (i) every reported number comes from states the generator never produced, and (ii) on this
dimension the project is **in the same position as DualRead**, not distinguished from it. The
honest sentence already existed in the repo and was never propagated
(`VERIFIER_ARCHITECTURES_2026-08-04.md:166-169`: *"free-at-generation is an inference from the
architecture, not a measurement"*). `AUDIT_2026-09-18.md` §4.

### D. The project numbers this file quotes

- **The judge is `MedVLThinker-32B`** (`src/labeling/run_judge.py:21` default; 0 of ~90 runner
  invocations override it), text-only, and it sees the gold. It is **not** Lingshu-32B, so
  "same-family judge" is wrong wherever it is written.
- **The headline is +0.0736 macro in judge currency** (`head_final_stack_PVFIXED_2026-09-13.json`;
  +0.0737 re-scored from the frozen probes, `em_rescore_pooled_probe_2026-09-18.json`) — *not*
  the +0.0802 of the pre-2026-09-13 artifacts. §4 item 2's "8 benchmarks, 36,869 questions" is
  correct. On identical picks the same probe is **+0.0679** under a cross-family judge
  (MedGemma-27B-it), **+0.0156** token-F1 and **+0.0047 [−0.0077, +0.0171], a tie,** under lenient
  exact match.

---

## 0. THE ANSWER: YES. Extensively, recently, and under established names.

Our four-dimensional mechanism —
**(A)** a lightweight classifier on **frozen** hidden states,
**(B)** used to **select** among sampled candidates,
**(C)** pooled over the **candidate's own generated tokens**,
**(D)** **reused from the generation pass** at no extra forward pass — ⚠️ *this describes the
design, not the code as run; see corrections block §C* —
is **not novel**. It is a named sub-literature with at least five independent instantiations, and
the same structural idea is the *default design* in three other fields.

**Do not claim the mechanism. Do not claim the "free" argument. Both are taken.**

The names the field already uses: **hidden-state reward model**, **latent verifier**, **intrinsic
reward from hidden states**, **confidence estimation module** (speech), **model quality assessment /
self-assessment** (structural biology), **confidence model** (docking).

---

## 1. The papers that match all four dimensions

Each was fetched and read, not taken from a summary. IDs verified against arXiv abs pages.

| id | short name | what it does | note |
|---|---|---|---|
| **arXiv:2402.14688** | **Q-Probe** (ICML 2024) | linear probe on the frozen model's residual stream, used for reward maximisation; "essentially free in comparison to the base model" | earliest LLM instance found |
| **arXiv:2504.16760** | **LiLaVe** (Apr 2025) | XGBoost on frozen hidden states at the **decoded answer's** last-16 token positions, layers {-1,-2,-4,-8,-16}; "as the final response in best-of-n, we select the one with the highest LiLaVe score" | **they tried an MLP and rejected it for XGBoost** |
| **arXiv:2505.12225** | **SWIFT / ELHSR** (May 2025) | gated linear head over per-token hidden states of the **generated response**, all L layers concatenated, **BCE on binary correctness**, best-of-N by argmax; ~1.8–3.0e5 params, <0.005% of a 7B RM | our loss and our role, at 1/100th our size |
| **arXiv:2511.06209** | **ReProbe** | <10M-param probe on a frozen LLM, BCE + class weighting, best-of-N argmax; step vector is the **mean over the model's own generated tokens** (eq. 3) | **dimension (C) exactly** |
| **arXiv:2608.17124** | **CASE** (Aug 2026) | logistic-regression gate on frozen activations at the answer-token position, N=16–20 at **temperature 0.7**, on **medical LLMs** (OpenBioLLM, BioMistral, Med42, meditron) and **medical benchmarks** (MedQA, MedMCQA, PubMedQA) | same temperature, same role, medical — minus the image |
| **arXiv:2608.30841** | **HSRM** (Aug 2026) | ~2M-param encoder on a frozen generator's states at reasoning-step boundaries, ranks best-of-N; abstract sells "reusing representations already computed during generation", "does not require any additional generator forward passes", ~5 orders of magnitude fewer verification FLOPs than a 7B PRM | **our exact pitch, in print** |

### The "free" argument is stated verbatim, by at least four papers
> CASE: *"at deployment the operating-layer activation is already produced during the generation of
> each candidate, so scoring adds no forward pass."*
> HSRM: *"reusing representations already computed during generation."*
> Q-Probe: *"essentially free in comparison to the base model."*
> Semantic Entropy Probes: *"reducing the overhead ... to almost zero."*

---

## 2. The same idea is older, and lives outside NLP

- **Speech recognition, 2021 — this is the real ancestor.** **R-EBM** (Li et al., Interspeech 2021,
  arXiv:2103.14152): a 2-layer BLSTM on a **frozen** 145M ASR model, taking decoder hidden states,
  attention context and token embeddings, **mean-pooled over the candidate's own output tokens**,
  trained with **binary cross-entropy**, used to **re-rank an n-best list** of 4–32. That is
  A+B+C, in 2021, five years before us. Lineage: CEM (ICASSP 2021) → R-EBM → Qiu et al.
  (Interspeech 2021). **Dimension (D) is where speech stops** — R-EBM never argues the cost case and
  reports no FLOPs.
- **Structural biology, 2021.** AlphaFold2's **pLDDT** is "small per-residue networks on the final
  activations", computed inside the same forward pass, and the paper uses it to "select the best
  model per target" from five. A(partial)+B+C+D. Field terms: *model quality assessment (MQA)*,
  *estimation of model accuracy (EMA)* — a standing CASP category.
- **Molecular docking.** DiffDock samples N poses and ranks them with a 5M-param **confidence
  model** trained with **binary** RMSD<2Å labels — our recipe, but paying its own forward pass.
- **Object detection.** IoU-Net predicts box quality from features the detector already computed and
  ranks candidates with it.
- **Image generation.** arXiv:2603.02829 — MLP probe (512/256, sigmoid) on intermediate activations
  of a **frozen** text-to-image denoiser, all four dimensions.

---

## 3. What the search did NOT find — the residual, stated narrowly

Three independent angles returned the same negative, which is the only defensible novelty left:

> ⛔ **THIS SENTENCE IS WITHDRAWN (2026-09-20). See "CORRECTIONS 2026-09-20 → A" at the top.**
> **No verified paper applies a probe on generation-pass internal states as a best-of-N selector in
> a VISION-LANGUAGE model.** Every A+B+C+D instance is text-only, and almost all are mathematical
> reasoning.

*(arXiv:2605.28527, arXiv:2603.22492 and arXiv:2608.10835 each break a clause of the sentence
above; the narrowed residual, and why it is too narrow to be a contribution, are in the corrections
block at the top of this file.)*

The medical/VLM literature splits into two halves that do not touch:

- **MedProb (arXiv:2609.04336, Sept 2026)** — the closest, and it must be cited. Its **Appendix H**
  does probe-as-best-of-N-selector for open-ended medical VQA. But the candidate is fed **back in as
  input** and it reads the **last input token**, so it costs a second forward pass per candidate —
  exactly the LoRA design we replaced. Single per-layer logistic regression, 100 test examples, and
  the authors call it *"an initial demonstration."*
- **DualRead (arXiv:2609.06419, Sept 2026)** — reads hidden states over the model's **own
  generated answer tokens** on **SLAKE-test (1,061 in-domain) plus six out-of-domain benchmarks —
  VQA-RAD 600, PathVQA 1,000, PMC-VQA 1,000, OmniMedVQA 920, ReXVQA 1,000 and VQA-Med-2019 closed
  64** (corrected 2026-09-20: this file previously said five) — with **Qwen3-VL-2B-Instruct** and
  **MedGemma1.5-4B** (corrected: sizes were missing), both **fully post-trained on SLAKE-train with
  GRPO**. Our benchmarks, and generators we also use. Three corrections to how this file first read
  it: (i) the answer-token mean is only a **query vector** for answer-conditioned visual pooling
  inside P-Visual — the post-answer correctness feature (P-State) is the **single closing token**,
  so it is **not** our mean-pool; (ii) it already ships **CCG-AUC** (hard-negative image
  substitution) and a **black-image** baseline (BICR), so our image-ablation diagnostics are not
  new; (iii) it pays a teacher-forced replay rather than reading the generation pass — **and so do
  we** (see corrections block §C), so that is not a distinction we hold. What does still
  distinguish it: it **calibrates confidence for one answer** and never selects among candidates.
- Best-of-N in medical VQA exists, but every selector reads **text** (semantic consensus, sentence
  embeddings), not internal states.

---

## 4. What this means for how we write the work up

1. **Retire "verification is free" as the contribution.** It is HSRM's, CASE's, Q-Probe's abstract,
   in those words. We can still *state* the cost result — it is ours and it is measured — but as a
   property, not a discovery.
2. **The honest claim is the transfer, plus the map.** ~~First application of generation-pass
   hidden-state verification to vision-language / medical VQA~~ — **overreach, withdrawn
   2026-09-20 (correction C2): MedProb Appendix H already reranks open-ended medical VQA
   generations with a probe, and this project does not read the generation pass either.** What
   survives is the second half, and it is the stronger half: **the most thorough empirical
   characterisation anywhere — 8 benchmarks, 36,869 questions, four evaluation currencies on
   identical picks, two independent judges, the per-benchmark (LOBO) limit, the coverage wall, 13
   measured negatives.**
3. **Two papers from THIS MONTH are near-collisions** and must be cited and distinguished explicitly
   (MedProb Appendix H; DualRead). Neither is fatal; both would be fatal if a reviewer found them
   and we had not.
4. **Position against the right lineage.** Cite R-EBM/CEM as the ASR ancestor and AlphaFold's pLDDT
   as the cross-field one. A paper that shows it knows this is a re-instantiation of a known family
   in a new modality reads far better than one that claims the family.
