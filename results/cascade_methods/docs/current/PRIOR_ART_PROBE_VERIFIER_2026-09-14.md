# PRIOR ART: has "a probe on frozen hidden states, used as a best-of-N verifier" been done?

**2026-09-14** · seven literatures searched in parallel, closest hits independently re-fetched and
adversarially re-read · supersedes nothing, but **materially changes how the open-text arm should be
positioned**

---

## 0. THE ANSWER: YES. Extensively, recently, and under established names.

Our four-dimensional mechanism —
**(A)** a lightweight classifier on **frozen** hidden states,
**(B)** used to **select** among sampled candidates,
**(C)** pooled over the **candidate's own generated tokens**,
**(D)** **reused from the generation pass** at no extra forward pass —
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

> **No verified paper applies a probe on generation-pass internal states as a best-of-N selector in
> a VISION-LANGUAGE model.** Every A+B+C+D instance is text-only, and almost all are mathematical
> reasoning.

The medical/VLM literature splits into two halves that do not touch:

- **MedProb (arXiv:2609.04336, Sept 2026)** — the closest, and it must be cited. Its **Appendix H**
  does probe-as-best-of-N-selector for open-ended medical VQA. But the candidate is fed **back in as
  input** and it reads the **last input token**, so it costs a second forward pass per candidate —
  exactly the LoRA design we replaced. Single per-layer logistic regression, 100 test examples, and
  the authors call it *"an initial demonstration."*
- **DualRead (arXiv:2609.06419, Sept 2026)** — mean-pools hidden states over the model's **own
  generated answer tokens** on SLAKE / VQA-RAD / PathVQA / PMC-VQA / OmniMedVQA with Qwen3-VL and
  **MedGemma** — our pooling, our benchmarks, a generator we also use. But it **calibrates
  confidence for one answer** and never selects among candidates, and it pays a teacher-forced
  replay rather than reading the generation pass.
- Best-of-N in medical VQA exists, but every selector reads **text** (semantic consensus, sentence
  embeddings), not internal states.

---

## 4. What this means for how we write the work up

1. **Retire "verification is free" as the contribution.** It is HSRM's, CASE's, Q-Probe's abstract,
   in those words. We can still *state* the cost result — it is ours and it is measured — but as a
   property, not a discovery.
2. **The honest claim is the transfer, plus the map.** First application of generation-pass
   hidden-state verification to vision-language / medical VQA, and — more defensibly — the most
   thorough empirical characterisation anywhere: 8 benchmarks, 36,869 questions, the per-benchmark
   (LOBO) limit, the coverage wall, 13 measured negatives.
3. **Two papers from THIS MONTH are near-collisions** and must be cited and distinguished explicitly
   (MedProb Appendix H; DualRead). Neither is fatal; both would be fatal if a reviewer found them
   and we had not.
4. **Position against the right lineage.** Cite R-EBM/CEM as the ASR ancestor and AlphaFold's pLDDT
   as the cross-field one. A paper that shows it knows this is a re-instantiation of a known family
   in a new modality reads far better than one that claims the family.
