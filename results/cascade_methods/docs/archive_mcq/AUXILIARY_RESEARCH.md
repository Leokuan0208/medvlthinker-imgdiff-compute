# Auxiliary research log — adjacent/exploratory findings (NOT core cascade results)

> Per the 2026-06-24 directive: anything **not directly part of the core Medical-VLM model-cascade paper**
> (broad literature survey of adjacent areas, datasets/models explored but not adopted, tangential
> negative results, tooling notes) lives **here**, separate from the core cascade docs
> (`OPENENDED_CASCADE.md`, `ACCV3_V4_AND_NOVELTY.md`, `RESCUE_INTO_ACCV2.md`, paper §5). Core results do
> **not** go here. No fabricated numbers.

---

## Literature surveys (background agents, 2026-06-24)

### Learned (training-required) cascade/routing/deferral baselines (verified arXiv IDs)
- **Jitkrittum "When does confidence-based cascade deferral suffice?" (2307.02764, NeurIPS'23)** — THE method
  documented to beat confidence-deferral, but ONLY when the strong model is a *specialist* (label noise /
  distribution shift / subset-better). For a *uniform* improver, confidence is near-Bayes-optimal. → We use
  this as the theory backing for §5.2 (`uniform_improver_diag.py`: breakage 0.22 MCQ / 0.14 open = uniform
  improver) and ran its Diff-01 learned-deferral (ties/marginal). **Folded into the paper §5.2.**
- **Hybrid-LLM (2404.14618, ICLR'24)** — learned PRE-generation query-difficulty router (BERT/DeBERTa on the
  query/image, decides before the small model answers). A design cell our gates don't cover. Code unverified.
  **Candidate to run** (Hybrid-LLM-style router on pre-answer hidden state). Already in our MCQ bake-off is a
  *post*-generation learned scorer (FrugalGPT/Jitkrittum logistic); Hybrid-LLM is the pre-generation variant.
- **RouteLLM (2406.18665, ICLR'25, code lm-sys/RouteLLM)** — learned router (MF/BERT), wins vs *random* not vs
  a confidence gate → weaker threat; covered by Hybrid-LLM. **Cite.**
- **Co-LLM (2403.03870, ACL'24, clinicalml/co-llm)** — learned token-level fusion; adjacent to our CALM
  negative; single-letter MCQ leaves little room. **Cite, don't run.**
- **Cascade-Routing (2410.10347, ICML'25, eth-sri/cascade-routing)** — unifying router+cascade framework;
  reduces to Jitkrittum for 2 models. **Cite.**
- Already covered: FrugalGPT (2305.05176), AutoMix (2310.12963), MoT-cascade (2310.03094). Lossless/orthogonal
  (one related-work sentence): speculative decoding (EAGLE 2401.15077, BiLD 2302.07863, Speculative Cascades
  2405.19261). Out of scope (N>2 model selectors, baseline=random): RouterDC, ZOOTER, FORC, GraphRouter, etc.
- **Medical-specific:** a learned confidence-gated cascade between two different-size *medical VLMs* with
  FLOP/latency accounting appears **near-novel** (nearest: CAR 2505.15154 within-model; Med-MoE 2404.10237
  intra-model MoE; CoMed-TR fusion). Strengthens our positioning.

### Open-ended medical-VQA scoring metrics (verified arXiv IDs)
- **Semantic (general):** BERTScore (1904.09675) — the standard semantic metric for short open-ended medical
  VQA (SLAKE/VQA-RAD/PathVQA), synonym-tolerant. **→ adding as a 4th scorer.**
- **Radiology-factual (report-level, mostly CXR-only):** GREEN (2405.03595, strongest by human correlation,
  LLM-based), RaTEScore (2406.16845, entity-based, most transferable to short answers), RadGraph-F1
  (2106.14463, low radiologist-corr despite popularity), CheXbert/CheXpert-F1 (2004.09167/1901.07031, locked
  to 14 CXR findings), RadCliQ (Patterns'23, no arXiv). Framework: RadEval (2509.18030). These are CXR-report
  metrics, **not** general medical-VQA metrics → not adopted (our datasets are general VQA, short answers).
- **LLM-judge protocols:** our binary-correctness-vs-reference judge matches the modern standard — **Lingshu
  (2506.07044)** GPT-4.1 binary consistency, **LLaVA-Med (2306.00890)** GPT-4 pairwise-vs-reference. Both are
  **text-only** (image-blindness, a known limitation; critiques: MT-Bench 2306.05685, "Justice or Prejudice?"
  2410.02736). HuatuoGPT-Vision (2406.19280) / MedGemma (2507.05201) use accuracy/F1. **→ §5.7 cites this.**

## Open-ended medical VQA datasets (survey + adoption decisions)

**Adopted / running:** `SimulaMet/Kvasir-VQA-x1` (test 15 955, **GI endoscopy** — a NEW modality; genuinely
free-text, median ~9 answer tokens, 1591 unique/2000; CC BY-NC; images are HF URLs → local 1200-subset
prep in `prep_kvasir.py` → judge-scored like PathVQA). Added as a 4th open-ended set for §5.7 generality.

**Candidate, gated:** `wisdomik/Quilt_VQA` (open-ended histopathology, 1 283, independent of PathVQA) — HF
**gated** (needs auth) → not run. `rippleripple/ProbMed`, `OpenGVLab/GMAI-MMBench`, `TsinghuaC3I/MedXpertQA`
(MM, already eval'd as MCQ), `SuhaoYu1020/MedFrameQA` (multi-image reasoning, MCQ) — reasoning-heavy
candidates for future think-tier work.

**Not adopted (with reasons):**
- `Medical_Multimodal_Evaluation_Data` / `foreverbeliever/OmniMedVQA`: MCQ (has options) → not open-ended.
- `SimulaMet-HOST/Kvasir-VQA` (older, 58 849): semi-structured (categories/yes-no/counts), NOT free-text → use Kvasir-VQA-x1.
- `FreedomIntelligence/PubMedVision`, `UCSC-VLAA/MedTrinity-25M`: training corpora / captioning, not clean held-out VQA.
- ROCOv2 / pmc_oa: captioning, not VQA. Medical-Diff-VQA / MIMIC-CXR-VQA / EHRXQA: PhysioNet-gated.

## Strong medical VLMs surveyed (cross-family cascade legs, not yet run)

- `ZJU-AI4H/Hulu-Med-7B/14B/32B` (SigLIP-NaViT vision stack, Apache-2.0, vLLM — best cross-family add).
- `FreedomIntelligence/HuatuoGPT-Vision-7B / 34B-hf` (LLaVA-v1.5 family, Apache-2.0, vLLM `-hf` variant).
- `Sunanhe/MedDr_0401` (40B, InternVL family; flagged high luck-rate — validate). Generalist strong legs:
  `Qwen2.5-VL-72B-Instruct`, `InternVL3-38B/78B`. `microsoft/llava-med-v1.5-mistral-7b` (diverse, painful vLLM).
  (Candidates if we extend the cross-family cascade beyond Lingshu; not run this round.)

## Scoring (defends our LLM-judge choice)
Field converged on **reference-anchored LLM-judge + lexical/semantic backup**. Token-recall = LLaVA-Med
(2306.00890); token-F1 still used by MedGemma (2507.05201, which deliberately avoids an LLM judge — honesty
counterpoint). LLM-judge precedent: LLaVA-Med (1–10 vs reference), Lingshu/MedEvalKit (2506.07044, GPT-4.1
binary consistency). Exact-match is dead on long answers (PathVQA EM 0.4–2.9%). BERTScore (1904.09675) is the
standard semantic backup — *available but not run* (we already triangulate with exact-match + token-F1 +
LLM-judge). Bias caveats: MT-Bench (2306.05685), "Justice or Prejudice?" (2410.02736) → we pin the judge
model, anchor on references, and report alongside lexical metrics.

## Datasets & models explored but not adopted (with reasons)

*(e.g. datasets that turned out MCQ-only, unscoreable, license-blocked, or redundant)*

## Tangential explorations / negative results outside the core cascade

*(things tried that inform the domain but are not part of the cascade paper's claims)*
