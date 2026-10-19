# PAPER_NOTES_GH — Generators (G) and Benchmarks / Evaluation protocol (H)

Prepared 2026-10-19 for the professor deck. Read-only research.

**Sources and tags**
- `[PDF p.N]` means the text was read in full from `/home/jamesyang/.claude/jobs/61c9eff9/tmp/txt/<file>.txt`, extracted from the core PDFs in `.../lit-domain-package/literature/papers/`. Page = the PDF page marker.
- `[abstract]` means the arXiv abstract.
- `[arXiv HTML, fetched]` means I fetched the arXiv HTML page on 2026-10-19 and a summariser read it. These are four fetches: RadImageNet-VQA, Kvasir-VQA-x1, GEMeX, PathVQA. **Treat these as needing a PDF re-check before they go on a slide.**
- `[config]` means the local HF snapshot `models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd4…/config.json` or `preprocessor_config.json`.
- `[repo]` means a file in this repo, named.
- **(derived)** marks arithmetic on quoted numbers, with the formula shown.
- **(inferred)** marks my interpretation.

Two further text extractions outside the GH core set were used for benchmark facts:
- `vqamed2019.txt`: the ImageCLEF 2019 VQA-Med overview.
- `2306.00890.txt`: LLaVA-Med. This is HTML-extracted, so it has no page numbers.

---

## Headline findings for the deck lead (read this first)

1. **Image-token count is a derived bound, not a sentence in the Qwen paper.** The paper says images are resized to multiples of 28, the ViT uses 14-px patches, and the merger groups 4 adjacent patches [PDF p.3–4].
   - So one LLM image token covers one 28×28-pixel region **(derived)**: 14×2 = 28.
   - The token count is therefore ≤ max_pixels / 784: 250,880 / 784 = **320** (cap320) and 1,003,520 / 784 = **1,280** (derived).
   - It is an upper bound. The real count depends on each image's size after rounding.
   - The sequence also carries the `<|vision_start|>`/`<|vision_end|>` tokens [config: `vision_start_token_id` 151652, `vision_end_token_id` 151653].
2. **Our cross-family judge "MedGemma-27B-it" is the *multimodal* 27B variant.**
   - The local `google/medgemma-27b-it` config contains a `siglip_vision_model` sub-config [config of that snapshot].
   - The MedGemma report's main text evaluates the *text-only* 27B. The multimodal 27B appears only in **Appendix F, Table A14 (p.60)**, as "preliminary results".
   - The deck must not cite main-text "MedGemma 27B" numbers as describing our judge.
3. **The released Lingshu-7B is the 3-stage SFT model, not Lingshu-RL.** Fig. 6 caption, p.11: "our Lingshu is trained after the first three SFT stages. We additionally apply RL on top of it, leading to Lingshu-RL version."
   - Kvasir-VQA appears **only** in the RL-stage data (Table 5, p.12).
   - So COMPARISON_BRIEF §1.4's contamination note, which lists the RL-stage datasets as data "the generator has seen", over-reaches for Kvasir-VQA.
   - The paper text gives no evidence that the released checkpoint saw Kvasir-VQA. The HF config does not say which stage it is (inferred).
4. **Lingshu claims it deduplicated against its evaluation benchmarks.** §2.3, p.9: "To prevent data contamination, we perform strict image and text deduplication by excluding any images and samples overlapping with the evaluation benchmarks in MedEvalKit."
   - That covers the MedEvalKit *test* splits of VQA-RAD/SLAKE/PathVQA/PMC-VQA/OmniMedVQA/MMMU/MedXpertQA, not their *training* splits. Those training splits are in Stage 3 (Table 5, p.12).
   - VQA-Med-2019 is not a MedEvalKit benchmark, so this dedup claim does not cover it. Our VQA-Med cell is mostly the 2019 *training* split, and Lingshu Stage 3 trains on "VQA-Med-2019". The project already flags this (FINAL_METHOD §6.4). The paper confirms the exposure route.
5. **Liu et al. 2025's headline claim contradicts its own tables.** The abstract says "reasoning performance is consistently lower than understanding" [p.1]. But Table 2 (reasoning) is *higher* than Table 3 (understanding) for most cells [p.4].
   - Example: Lingshu-7B PathVQA reasoning 0.7445 vs understanding 0.6681; Qwen2.5-VL-7B VQA-RAD 0.6456 vs 0.5890.
   - The G-card's "independent replication of our finding #1" must be dropped.
6. **MT-Bench did not establish self-enhancement bias.** It adopts the term and then says "our study cannot determine whether the models exhibit a self-enhancement bias" [PDF p.5]. The H-card's "coins / measures" wording overstates this.
7. **Our OmniMedVQA cell dropped RadImageNet; this is load-bearing and should be said.**
   - RadImageNet is the largest single source in OmniMedVQA: 55,443 images / 56,697 QA items, open access [OmniMedVQA Table 8, p.10].
   - Our cell excluded it because RadImageNet-VQA is a separate benchmark of ours [repo `OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §2].

---

# Part 1 — Core PDFs

## G1. Qwen2.5-VL Technical Report

**Citation.** Qwen Team, Alibaba Group (S. Bai et al.), "Qwen2.5-VL Technical Report," arXiv:2502.13923v1, 19 Feb 2025 [PDF p.1].

**1. Problem.** Open vision-language models handle fine-grained perception, documents, grounding and long video badly or expensively. Native-resolution input makes a ViT's attention cost grow quadratically with image size. The report builds a model that reads images at their native resolution without that cost blow-up, and keeps the language ability of the Qwen2.5 LLM.

**2. Method: the architecture our generator inherits** (§2.1, p.3–5; Table 1, p.4). Three components [p.3]:

- **Vision encoder.** A ViT trained from scratch, "redesigned" with 2D-RoPE, window attention, RMSNorm and SwiGLU [Fig. 1 caption, p.3; §2.1.1, p.4].
  - "During both training and inference, the height and width of the input images are resized to multiples of 28 before being fed into the ViT. The vision encoder processes images by splitting them into patches with a stride of 14" [p.3].
  - Window attention: "only four layers employ full self-attention, while the remaining layers utilize windowed attention with a maximum window size of 112×112 (corresponding to 8×8 patches)" [p.4].
  - Table 1, 7B column [p.4]: ViT hidden 1280, 32 layers, 16 heads, intermediate 3456, patch size 14, window size 112, full-attention block indexes {7, 15, 23, 31}.
  - Cost claim: window attention makes "computational cost scale[] linearly with the number of patches rather than quadratically" [p.4].
- **MLP-based vision-language merger** (the paper's name; the field calls it a *projector* or *connector*).
  - "we first group spatially adjacent sets of four patch features. These grouped features are then concatenated and passed through a two-layer multi-layer perceptron (MLP) to project them into a dimension that aligns with the text embeddings used in the LLM" [p.3–4].
  - Table 1: merger in-channel 1280, out-channel **3584** (7B) [p.4].
  - Figure equation (derived from the text): $z_j = \mathrm{MLP}_2\big([\,p_{2r,2c}\,\|\,p_{2r,2c+1}\,\|\,p_{2r+1,2c}\,\|\,p_{2r+1,2c+1}\,]\big)\in\mathbb{R}^{3584}$, with $p\in\mathbb{R}^{1280}$. The input is $4\times1280 = 5120$-d (derived). The 2×2 spatial grouping is in `spatial_merge_size: 2` [config].
- **LLM.** Initialised from Qwen2.5. Table 1, 7B [p.4]:
  - hidden size **3,584**, **28** layers, 4 KV heads, head size 128, intermediate 18944;
  - no embedding tying; vocabulary 151646; 4.1T trained tokens.
- **M-RoPE** (§2.1.3, p.5). It "decomposes the position embedding into three distinct components: temporal, height, and width." For text "all three components use identical position IDs, making MRoPE functionally equivalent to traditional 1D RoPE." For images "the temporal ID remains constant across visual tokens, while unique IDs are assigned to the height and width components based on each token's spatial position." Qwen2.5-VL's change is aligning the temporal IDs to absolute time, which matters for video only.
  - Config detail (not in paper): `mrope_section [16,24,24]`, `rope_theta 1e6` [config].
- **Native dynamic resolution** (§2.1.2, p.5): "dynamically converts images of varying sizes into sequences of tokens with corresponding lengths."

**Image-token arithmetic (derived; put on the architecture slide with a "(derived)" tag).**
- Patches $= (H/14)(W/14)$.
- After the 2×2 merge, tokens $= (H/28)(W/28) = HW/784$.
- `max_pixels` caps $HW$, so tokens ≤ max_pixels/784.
  - cap320: 250,880/784 = **320**.
  - full-res run: 1,003,520/784 = **1,280**.
  - Model default `max_pixels` 12,845,056 [config] → 16,384 (derived).
  - `min_pixels` 3,136 [config] → 4 (derived).
- The project measured GEMeX at 303 tokens (cap320) vs 1,237 (full-res) [repo `FINAL_METHOD_2026-10-16.md` l.75]. Both are consistent with the bounds.

**Training** (§2.2, Table 2, p.8; §2.3, p.8–10).
- Pre-training has three phases:
  - visual pre-training (ViT only), 1.5T tokens;
  - multimodal pre-training (ViT & LLM), 2T;
  - long-context (ViT & LLM), 0.6T, sequence length 32768.
- Post-training is SFT (~2 million entries, 50% text / 50% multimodal) then DPO, "both with the Vision Transformer (ViT) parameters frozen" [p.10].
- Rejection sampling for CoT data [§2.3.3, p.10].

**3. Setup.** General benchmarks only, none medical. Qwen2.5-VL-7B MMMU-val 58.6 [Table 3, p.11].

**4. Key results.** Only the architecture matters for us. One general number: 7B MMMU-val 58.6 [Table 3, p.11].

**5. Terminology.**
- LVLM / MLLM.
- Vision Transformer (ViT).
- "window attention" / "windowed attention".
- "MLP-based Vision-Language Merger".
- MRoPE / "Multimodal Rotary Position Embedding Aligned to Absolute Time".
- "native dynamic resolution".
- SFT, DPO, rejection sampling.

**6. Figures.** Fig. 1, p.3: vision encoder (native-resolution images → variable-length token sequences) → "Qwen2.5 LM Decoder" bar with M-RoPE IDs. Style is described in FIGURE_STYLE_NOTES §1.6. Recreate it as our architecture panel, adding the 2×2 merge glyph and the "≤ max_pixels/784 tokens" bracket.

**7. What it gives the deck.**
- Slides: architecture/background (where the probe reads) and terminology (projector, M-RoPE, hidden size, layer).
- Presenter sentence: "Lingshu-7B is a Qwen2.5-VL-7B. A 32-layer ViT cuts the image into 14-pixel patches, a 2×2 MLP merger turns every 28×28-pixel region into one 3,584-d token, and a 28-layer language model reads those tokens. Our probes read that language model's residual stream at layers 18, 20 and 22."
- **Cuts against us.** The ViT is trained at native resolution [p.4–5], and cap320 throws pixels away. The project measured that full-res hurts GEMeX overall, so this is a known trade-off, not a contradiction.

**8. Must not claim.**
- That the paper states "max_pixels/28²" (it is derived).
- That the paper gives a Q-head count for the 7B LLM. It does not; 28 heads come from config.
- **Paper vs config discrepancies (state both if quoted):**
  - ViT intermediate size: Table 1 says 3456, `vision_config.intermediate_size` says 3420.
  - Vocabulary: Table 1 says 151646, config `vocab_size` says 152064 (padded embedding matrix, inferred).
- That Qwen2.5-VL has any medical training (no medical data is named).

---

## G2. Lingshu

**Citation.** LASA Team, W. Xu, H. P. Chan, L. Li, et al. (DAMO Academy, Alibaba), "Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning," arXiv:2506.07044v4, 13 Jun 2025 (dated 16 Jun 2025) [PDF p.1]. COMPARISON_BRIEF says v5 (21 Sep 2026) renames the model "Lingshu-Med" and carries the same Table I values. I did not re-read v5.

**1. Problem.** Medical MLLMs (a) cover too little knowledge beyond imaging, (b) hallucinate because of poor data curation, and (c) lack medical reasoning [abstract, p.1]. Medical evaluation is also non-standardised. Lingshu answers with a data-curation pipeline, a multi-stage training recipe on Qwen2.5-VL, and MedEvalKit.

**2. Method.**
- **Backbone.** "Lingshu is built upon the Qwen2.5-VL model architecture … a large language model (LLM), a vision encoder, and an MLP-based projector. We employ … 7B-Instruct and 32B-Instruct, as the foundational models" [§3, p.9]. No architectural change is described. The local config is `Qwen2_5_VLForConditionalGeneration` with hidden 3584 and 28 layers [config].
- **Training stages** (Fig. 6, p.11; Table 5, p.12; §3.1–3.2, p.11–14). Data amounts, trained/frozen, learning rates and epochs:

| stage | data (Table 5) | amount | trained / frozen | LR, epochs (§3.2, p.14) |
|---|---|---:|---|---|
| 1 Medical Shallow Alignment | PMC-OA and ROCO (coarse captions) | ~927K | vision encoder + projector trained; **LLM frozen** | ViT 2e-6, projector 1e-5; 1 epoch |
| 2 Medical Deep Alignment | ROCOv2, LLaVA-Med, Quilt-LLaVA, PubMedVision, MIMIC-CXR, FairVLMed, MedICaT, MedPix-2.0, synthetic long captions + LLaVA-1.5 captions, PixMo | ~4.1M | all unfrozen | LLM 1e-5; 1 epoch |
| 3 Medical Instruction Tuning | **PathVQA, PMC-VQA, SLAKE**, Quilt-LLaVA, **VQA-Med-2019**, MIMIC-Ext-MIMIC-CXR-VQA, PubMedVision, LLaVA-Med (en & zh), **VQA-RAD**, CheXpert Plus, MIMIC-CXR, ROCOv2, IU-Xray + synthetic QA/OCR/CoT; general multimodal; medical text; general text | ~7.1M | all unfrozen; data packing | LLM 1e-5; 2 epochs |
| 4 Medical-oriented RL (→ **Lingshu-RL only**) | MIMIC-Ext-MIMIC-CXR-VQA, PMC-VQA, **Kvasir-VQA**, Path-VQA, SLAKE, VQA-Med-2019, VQA-RAD + synthetic | ~100K | GRPO, 1 epoch | LR 1e-6, T = 1.0, KL 1e-3, 16 responses/prompt (p.14) |

  - "Note that our Lingshu is trained after the first three stages" [§3.1, p.10].
  - Common settings: AdamW, cosine schedule, warm-up 100, max sequence length 8,192, per-device batch 1, gradient accumulation 8 [§3.2, p.13].
  - Loss: standard SFT next-token cross-entropy (inferred; the paper names no loss formula).
  - RL reward: "a strict format reward and an accuracy reward, with respective weights of 0.5 and 1" [§3.1.4, p.13]. As an equation: $r = 0.5\,r_{\text{format}} + 1.0\,r_{\text{acc}}$ (the paper's weights, written as a sum).
- **Data totals.** "3.75M high-quality open-source medical samples and 1.30M high-quality synthetic medical samples" [§2.3, p.9].
  - Synthetic: ~100K long captions (GPT-4o, 5-stage pipeline) [p.7]; 50K OCR samples [p.8]; template-based + self-instruct VQA [p.8]; GPT-4o CoT distillation with GPT-4o consistency check [p.9].
  - Medical imaging sources used for synthesis include "**RadIMageNet (subset of ultrasound images)**" [Table 3, p.6]. Our RadImageNet-VQA cell is CT/MRI (see H-benchmarks), so no image overlap is implied (inferred).
- **Dedup claim**, verbatim [§2.3, p.9]: "To prevent data contamination, we perform strict image and text deduplication by excluding any images and samples overlapping with the evaluation benchmarks in MedEvalKit."
- **MedEvalKit evaluation** (§4, p.14–16; Fig. 8, p.15; Table 24, p.52; Table 25, p.53).
  - Prompts [Fig. 8, p.15]:
    - open-ended: "Answer the question concisely."
    - close-ended: "Answer the question using a single word or phrase."
    - MCQ: "Answer with the option's letter from the given choices directly."
  - Scoring [§4.2, p.16]:
    - MCQ: rule-based matching, then the MMMU codebase similarity.
    - Open-ended: "we directly leverage GPT-4.1 (2025-04-14 version) to assess consistency between model outputs and reference answers."
    - Judge prompt [Table 24, p.52]: the judge sees question + standard answer + user's answer and outputs `<think>…</think><judge>0/1</judge>`, "(0 represents correct, 1 represents incorrect)". This is reference-guided, single-answer, binary grading (MT-Bench §3.1 vocabulary).

**3. Setup.**
- Baselines (§5.1, p.16): GPT-4.1, Claude Sonnet 4, Gemini-2.5-Flash; medical MLLMs (BiomedGPT, Med-R1, MedVLM-R1, MedGemma, LLaVA-Med, HuatuoGPT-V, BioMediX2, HealthGPT, MedDr); general MLLMs (Qwen2.5-VL, InternVL2.5/3). All run in MedEvalKit.
- Multimodal benchmark sizes [Table 25, p.53]:

| benchmark | modality | MCQ | close | open | total Q | images |
|---|---|---:|---:|---:|---:|---:|
| VQA-RAD | X-ray, CT, MRI | — | 251 | **200** | 451 | 203 |
| SLAKE | X-ray, CT, MRI | — | 836 | 1,258 | 2,094 | 180 |
| PathVQA | Mixed | — | 3,362 | **3,357** | 6,719 | 858 |
| PMC-VQA | Mixed | 33,430 | — | — | 33,430 | 29,021 |
| OmniMedVQA | Mixed | 88,996 | — | — | 88,996 | 82,059 |
| MMMU | Mixed | 1,927 | — | — | 1,927 | 1,927 |
| MedXpertQA | Mixed | 4,500 | — | — | 4,500 | 2,852 |

Note on SLAKE: 2,094 = 1,061 English + 1,033 Chinese, per COMPARISON_BRIEF's count from the local `test.json`. The paper does not state the language split. OmniMedVQA here is "the open-access part" [§4.1, p.15] and is scored as MCQ.

**4. Key results.**
- Table 6, p.17, Lingshu-7B: MMMU-Med 54.0, VQA-RAD 67.9, SLAKE 83.1, PathVQA 61.9, PMC-VQA 56.3, OMVQA 82.9, MedXQA 26.7, **Avg 61.8**.
- Same table, Qwen2.5VL-7B: 50.6 / 64.5 / 67.2 / 44.1 / 51.9 / 63.6 / 22.3, Avg 52.0.
- Same table, Lingshu-32B: 62.3 / 76.5 / 89.2 / 65.9 / 57.9 / 83.4 / 30.9, Avg 66.6.
- Same table, MedGemma-4B-IT: 43.7 / 72.5 / 76.4 / 48.8 / 49.9 / 69.8 / 22.3, Avg 54.8.
- **RL does not help** [Table 9, p.19]: Lingshu-RL-7B averages 61.5 vs Lingshu-7B 61.8. The authors: "the current RLVR training has not achieved the same improvements as those observed in the math/code domains". One cited cause: rule-based rewards are unreliable for open-ended answers with "substantial linguistic variability" [p.19].
- **In-domain data drives the benchmark scores** [Table 10, p.20]. Removing the medical-multimodal data (2.7M) takes PathVQA 61.9 → **45.2**, SLAKE 83.1 → **69.0**, OMVQA 82.9 → 78.0.
- Training data for the multimodal benchmarks [Fig. 7a, p.14]: 135,617 samples total, OmniMedVQA 65.6%, PMC-VQA 24.7%, PathVQA 5.0%.

**5. Terminology.**
- "Medical Shallow Alignment / Deep Alignment / Instruction Tuning / Medical-oriented RL".
- RLVR, GRPO.
- "LLM-as-a-Judge" (their wording: "a dual verification mechanism that combines rule-based evaluation with an LLM-as-a-Judge strategy" [p.15]).
- MedEvalKit.

**6. Figures.**
- Fig. 6, p.11: four-stage pipeline with 🔥 trainable icons. This is the trained/frozen convention noted in FIGURE_STYLE_NOTES §1.7.
- Fig. 8, p.15: prompt templates. Worth showing beside our prompt.
- Table 10, p.20: data ablation. Good one-row contamination evidence: "remove medical multimodal → PathVQA −16.7" (derived: 61.9 − 45.2).

**7. What it gives the deck.**
- Slides: generator background, contamination/limitations, comparison (protocol differences), terminology.
- Presenter sentence: "Our generator is Lingshu-7B: Qwen2.5-VL-7B with three supervised stages of medical training, about 12 million samples. Those samples include the *training* splits of PathVQA, SLAKE, VQA-RAD and VQA-Med-2019, so absolute accuracies on those benchmarks are in-distribution. Our gains are measured against the same model's own greedy answer, so that exposure is held constant."
  - The ~12M figure is derived: 0.927 + 4.1 + 7.1 = 12.127M.
- **Cuts against us:**
  - (a) Lingshu's own RL with exact-match-style rewards did not help open-ended medical QA. This supports our view that open-ended correctness is hard to verify by rules, which is why we use a trained probe with judge labels.
  - (b) Lingshu's published numbers use GPT-4.1 as judge, combine open + closed, include SLAKE-zh, and run at default resolution. They are not comparable to ours.

**8. Must not claim.**
- That Lingshu-7B was trained on Kvasir-VQA (only the RL variant's data lists it).
- That Lingshu's dedup covers training splits.
- "Lingshu-32B judge" (our judge is MedVLThinker-32B).
- That Table 6 numbers are open-ended accuracy (they are combined accuracy).

---

## G3. MedGemma Technical Report

**Citation.** Google Research and Google DeepMind (A. Sellergren et al.), "MedGemma Technical Report," arXiv:2507.05201v4, 6 Apr 2026 [PDF p.1].

**1. Problem.** Build open medical VLM foundation models that perform well across medical tasks with little task-specific tuning, while keeping Gemma 3's general ability [abstract, p.1].

**2. Method.**
- **Models.** MedGemma 4B multimodal; MedGemma 27B text-only. Also a **MedGemma 27B multimodal** variant "being released along with the other models. More thorough evaluation … is ongoing and preliminary results can be found in Appendix Section F" [p.3]. Unless noted, "MedGemma 27B" in the report means text-only [p.3]. MedSigLIP is a 400M encoder.
- **Architecture** [§2.2.1, p.4]. Gemma 3; vision encoder "the 400M variant of the SigLIP encoder"; "input image resolution is 896×896"; long context (128k). Images are resized to 896×896 [§2.1.2, p.4].
  - So the token count is fixed, unlike Qwen's dynamic resolution. The number of image tokens is not stated in this paper.
  - CT images: three windows mapped to RGB [p.4].
- **Training** [§2.2.2, p.5]:
  1. vision-encoder enhancement on >33M medical image-text pairs, medical data at 2% weight;
  2. multimodal decoder pretraining, medical image data at 10% weight, ~5 epochs on the medical mixture;
  3. post-training by distillation (text) and RL. "For multimodal training, we found that RL enables better generalization compared to supervised fine-tuning, so all multimodal post-training was performed via RL."
  - The 27B text-only model used "the post-training stage alone."
- **Training data relevant to our benchmarks** [Table 1, p.4]:
  - SLAKE: 450, "Captions generated from QA pairs", stages Vision/PT/RL.
  - **VQA-Rad: 1,391, RL**.
  - "we and others have identified potential data quality issues in PathVQA and MedVQA. Thus, we removed them from the training dataset" [§2.1.1, p.3].
- **Appendix F** [p.60]: the 27B multimodal variant was trained with "the same [methodology] as for the MedGemma 4B multimodal model with the addition of two training datasets: EHRQA … and Chest ImaGenome".
- **Inference** [§3.1, p.5]: "a temperature of 0.0 was used on medical benchmarks."

**3. Setup.**
- VQA [§3.4, p.7–8]: SLAKE English (1,061) and VQA-RAD (2,248) [Table 2, p.6].
- Metric: "average tokenized F1 … across open and closed QAs, as well as accuracy on the subset of yes/no questions."
- VQA-RAD uses a Google re-split "to avoid the train/test image contamination present in the original splits" [p.8]. Table 9 header: "Test split from (Xu et al., 2023)" [p.15]. The source inconsistency is noted in the H cards.

**4. Key results.**
- Table 9, p.15, SLAKE-En overall token F1 / open token recall / closed acc:
  - MedGemma 4B 72.3 / 63.3 / 87.6;
  - Gemma 3 4B 40.2 / 33.3 / 53.0;
  - Med-Gemini 75.8 / 72.2 / 84.6.
- VQA-RAD token F1 / closed acc: MedGemma 4B 49.9 / 69.1.
- **27B multimodal (our judge's checkpoint)** [Table A14, p.60]:
  - MedQA 85.3 (text-only 27B: 87.7);
  - SlakeVQA token F1 70.3; VQA-Rad token F1 46.7;
  - MedXpertQA (text+MM) 26.8;
  - EyePACS 75.3.
- MedSigLIP linear probe [§6, p.18]: "extracting image embeddings from MedSigLIP … and training a logistic regression using the SAGA solver." This is a field precedent for the word "linear probe".

**5. Terminology.**
- "linear probe classification (also referred to as data-efficient classification)" [p.18].
- token F1; "out-of-distribution (OOD): Data not seen during any model development stages" [Table 2 footnote, p.6].
- distillation; RL.

**6. Figures.** No architecture figure worth recreating. Fig. 1 (p.2) is a collection overview. Table A14 is the citation for our judge.

**7. What it gives the deck.**
- Slides: evaluation protocol (cross-family judge), terminology (linear probe precedent), comparison caveats.
- Presenter sentence: "Our second judge is MedGemma-27B-it, Google's medical Gemma 3. It is a different model family from the Qwen line that both our generator and our main judge come from. Google itself removed PathVQA from its training over data-quality concerns."
- **Cuts against us:**
  - (a) Google judged PathVQA/VQA-Med (ImageCLEF) data-quality problematic enough to exclude from training. PathVQA is our largest classic cell, 3,357 questions.
  - (b) MedGemma's multimodal 27B was trained (via the 4B recipe) on SLAKE and VQA-RAD data. So our cross-family judge is not naïve to those benchmarks (inferred from Appendix F, "same methodology").
  - (c) The MedGemma family's own default is temperature 0. Our T = 0.7 sampling is a deliberate departure.

**8. Must not claim.**
- That main-text "MedGemma 27B" results describe our judge.
- That MedGemma-4B replication results exist (our MedGemma-4b pools are withdrawn).
- Which paper the VQA-RAD re-split comes from (the paper is internally inconsistent).

---

## G4. Liu et al. 2025: How Far Have Medical VLMs Come?

**Citation.** C. Liu, J. Pan, W. Shen, W. Bai, D. Rueckert, R. Arcucci, "How Far Have Medical Vision-Language Models Come? A Comprehensive Benchmarking Study," arXiv:2507.11200v2, 18 Jul 2025 [p.1].

**1. Problem.** Medical VLMs are evaluated piecemeal. The paper compares general vs medical VLMs (3B–72B) on 7 benchmarks; the abstract says "eight" but lists seven [p.1]. It splits questions into "understanding" vs "reasoning" [p.1].

**2. Method** [§2.2–2.3, p.3].
- A binary classifier on gme-Qwen2-VL-2B embeddings, trained on MedXpertQA's understanding/reasoning labels, assigns each question of the other datasets to one category.
- All models are prompted "Please reason step by step, and put the final answer in \boxed{}". The boxed answer is compared "against the ground truth option to compute accuracy" [p.3].

**3. Setup.** Qwen2.5-VL 3/7/32/72B, MiMo-VL-7B SFT/RL, Lingshu-7B/32B, Huatuo-GPT-Vision-7B/32B (as named in the paper). Accuracy throughout.

**4. Key results.**
- Table 1, p.4, Lingshu-7B: MedXpert 0.2505, OmniMedVQA 0.6436, PMC-VQA 0.5213, PathVQA 0.7192, MMMU 0.6389, SLAKE 0.8034, VQA-RAD 0.6574.
- Same table, Qwen2.5-VL-7B: 0.2305 / 0.6072 / 0.4420 / 0.6205 / 0.5379 / 0.6068 / 0.6255.
- **Table 2 (reasoning) vs Table 3 (understanding), p.4.** Lingshu-7B PathVQA 0.7445 vs 0.6681; SLAKE 0.8039 vs 0.7102; VQA-RAD 0.6748 vs 0.6112.
  - Reasoning is *higher* in these cells. This contradicts the abstract's "reasoning performance is consistently lower than understanding" [p.1] and §3.2 [p.5].

**5. Terminology.** "understanding vs reasoning" (their classifier-based split); "zero-shot transfer".

**6. Figures.** Fig. 1 (p.2) is a generic VLM sketch, not worth recreating.

**7. What it gives the deck.**
- Slides: background/field state (one bullet).
- Presenter sentence: "An independent benchmark rates Lingshu-7B the strongest 7B medical model on PathVQA and SLAKE. It uses multiple-choice style accuracy with a \boxed{} reasoning prompt, so it says nothing about open-ended answers."
- **Cuts against us.** Their Lingshu-7B numbers (e.g. PathVQA 0.7192) are far from Lingshu's own (61.9) on the "same" benchmark. Harness and prompt choices move scores by about 10 points (derived: 71.92 − 61.9 = 10.0). This is a good argument for the deck's rule of comparing only within one protocol.

**8. Must not claim.**
- That it corroborates "reasoning hurts perception". Its tables say the opposite of its text, and it uses no format-matched comparison.
- That its scores are open-ended accuracy.

---

## H1. OmniMedVQA

**Citation.** Y. Hu, T. Li, Q. Lu, W. Shao, J. He, Y. Qiao, P. Luo, "OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM," arXiv:2402.09181v2, 21 Apr 2024 (CVPR 2024 per the guide; not stated in the PDF text) [p.1].

**1. Problem.** Existing medical VQA sets are small and cover few modalities and body regions [Table 1, p.1]. The authors build a large benchmark from real clinical images.

**2. Construction** [§3, p.3–5].
- Steps: collect 73 medical *classification* datasets (12 modalities, >20 anatomical regions) → design QA templates per dataset → sample by "Inverse Proportional Sampling" → GPT-3.5 rephrases questions and generates 2–4 distractor options → "Human double check" [p.5].
- Five question types [Table 3, p.5]:
  - Modality Recognition 19,427;
  - Anatomy Identification 20,330;
  - Disease Diagnosis 73,455;
  - Lesion Grading 2,621;
  - Other Biological Attributes 12,162;
  - total **127,995 items / 118,010 images**.
- **Native format is multiple-choice.** Two scores [§4, p.5–6]:
  - "Question-answering Score": generate, then match to options by similarity;
  - "Prefix-based Score": likelihood of each option.
- Release: 42 open-access datasets, 31 restricted [p.9].
- **RadImageNet: CT, MR, US; 55,443 images, 56,697 items, open access** [Table 8, p.10].
- Kvasir: 1,225 images, 1,537 items, restricted [Table 8, p.10].

**3. Setup.** Zero-shot evaluation of 12 LVLMs (8 general, 4 medical) [§5, p.6].

**4. Key results.** Table 5, p.7, overall QA-score / prefix-score:
- BLIP-2 50.69 / 33.43 (best);
- LLaVA-Med 28.78 / 24.06;
- random guess 28.28.

**5. Terminology.** Question-answering score, prefix-based score, LVLM, the five question types.

**6. Figures.** Fig. 1 (p.4): body map + five question-type examples. Useful as a benchmark-example tile.

**7. What it gives the deck.**
- Slides: benchmark description; protocol (MCQ → open conversion).
- Presenter sentence: "OmniMedVQA is classification datasets turned into multiple-choice questions. We strip the options and ask the question open-ended. We keep only modality and anatomy questions and drop the RadImageNet source, because RadImageNet-VQA is a separate benchmark of ours."
  - Our cell: 8,883 questions [repo `OPENTEXT_FULL_RUNDOWN` §2].
- **Cuts against us:**
  - (a) Our cell removes the options the benchmark was designed around, so its numbers are not comparable to any published OmniMedVQA score (Lingshu's 82.9 is MCQ).
  - (b) Contamination audit (H6): the public OmniMedVQA mirror fires the exchangeability detector for all five VLMs tested.
  - (c) Lingshu's training mix includes OmniMedVQA-like synthetic template VQA. Lingshu claims dedup against the MedEvalKit OmniMedVQA test (p.9). Whether our 8,883 subset lies inside that deduped set is not established (inferred open question).

**8. Must not claim.**
- That OmniMedVQA is an open-ended benchmark.
- That medical models beat general models on it (the paper finds the reverse).
- **Internal contradiction:** §1 p.2 says "medical-specialized LVLMs exhibit superior performance compared to general-domain LVLMs". The abstract, Table 5 and the conclusion say inferior. The tables support "inferior".

---

## H2. PMC-VQA

**Citation.** X. Zhang, C. Wu, Z. Zhao, W. Lin, Y. Zhang, Y. Wang, W. Xie, "PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering," arXiv:2305.10415v6, 8 Sep 2024 [p.1].

**1. Problem.** MedVQA had been treated as classification over a fixed answer set. The authors make it generative and build a large training set [p.1].

**2. Method and construction** [§4.1, p.10–11].
- ChatGPT generates 5 MCQ QA pairs per PMC-OA caption, giving 1,497,808 pairs.
- A LLaMA-7B text-only filter removes questions answerable without the image, leaving 848,433. A trained image-answerability classifier (81.77% accuracy) cleans further.
- Result: **226,946 QA / 149,075 images**.
- A 50,000-pair "PMC-VQA-test-initial" is randomly selected, with no image in both train and test. A manually verified **2,000-pair PMC-VQA-test** follows [p.11].
- Model: MedVInT-TE / -TD (vision encoder aligned to an LLM).
- Correct-option distribution: "A (24.07%), B (30.87%), C (29.09%), D (15.97%)" [§2.1, p.4].

**3. Setup.** Fine-tuning on VQA-RAD, SLAKE, ImageCLEF-2019. PMC-VQA-test in Blank (open) and Choice formats [§2.3, p.4].

**4. Key results.**
- Language-only LLaMA: 27.2% Blank, 30.8% Choice. "around 30% of the questions have 'B' answers, making the 30.8% score nearly equivalent to the highest possible score attainable through guessing" [p.4].
- MedVInT-TD VQA-RAD open 73.7, SLAKE open 84.5 [Table 2, p.5]; MedVInT-TE ImageCLEF-2019 70.5 [p.4].

**5. Terminology.** generative MedVQA; Blank vs Choice; answer-position bias (the term is the card's; the paper just reports the distribution).

**6. Figures.** Fig. 1 (p.2) gives open vs MCQ answer examples for the same question. A nice illustration of the format axis.

**7. What it gives the deck.**
- Slides: limitation/background only. PMC-VQA is **not** one of our 8 benchmarks.
- Its training split is in Lingshu Stage 3 and RL (Table 5, p.12), and it is the 24.7% share of Lingshu's multimodal-benchmark data [Fig. 7a, p.14].
- One sentence: "Benchmarks generated by LLMs from captions carry artefacts. In PMC-VQA a text-only guesser reaches the guessing ceiling, which is why we report several graders."

**8. Must not claim.**
- That this paper describes `test_2.csv` / "v2". The v6 PDF never mentions a v2 test file. Lingshu calls its MedEvalKit split "PMC-VQA (v2)" [p.15].
- That the B-skew applies to v2.

---

## H3. MT-Bench / Chatbot Arena

**Citation.** L. Zheng, W.-L. Chiang, Y. Sheng, et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023 Datasets & Benchmarks, arXiv:2306.05685.

**1. Problem.** Human evaluation of open-ended answers is slow and costly. Can a strong LLM replace it, and how is it biased? [p.2–4]

**2. Method.**
- Three judge types [§3.1, p.4]:
  - **Pairwise comparison**;
  - **Single answer grading** ("directly assign a score to a single answer");
  - **Reference-guided grading** ("provide a reference solution if applicable").
- Biases defined [§3.3, p.4–5]:
  - **Position bias**: "an LLM exhibits a propensity to favor certain positions over others."
  - **Verbosity bias**: "an LLM judge favors longer, verbose responses, even if they are not as clear, high-quality, or accurate as shorter alternatives."
  - **Self-enhancement bias**: "LLM judges may favor the answers generated by themselves" (term adopted from social cognition).
  - **Limited capability in grading math and reasoning.**
- Mitigations [§3.4, p.6]: swap positions; few-shot judging; CoT judge; reference-guided judge; fine-tuned judge.
- MT-bench: 80 multi-turn questions in 8 categories [§2.2, p.3–4].

**3. Setup.** 6 models answer MT-bench. 58 expert labellers give ~3K votes, plus 3K sampled Arena votes [§4.1, p.7]. Agreement = probability that two random judges of each type agree.

**4. Key results.**
- **Position bias** [Table 2, p.5]: GPT-4 consistency 65.0% (default prompt); Claude-v1 23.8%; GPT-3.5 46.2%.
- **Verbosity** "repetitive list" attack failure rate [Table 3, p.5]: Claude-v1 91.3%, GPT-3.5 91.3%, GPT-4 8.7%.
- **Self-enhancement** [p.5]: "GPT-4 favors itself with a 10% higher win rate; Claude-v1 favors itself with a 25% higher win rate … our study cannot determine whether the models exhibit a self-enhancement bias."
- **Reference-guided judging** fixes math grading: failure 14/20 (default) → 6/20 (CoT) → 3/20 (reference) [Table 4, p.5]. Text: "from 70% to 15%" [p.6].
- **Agreement** [Table 5, p.8]: GPT-4 (pair) vs human **85%** (setup S2, no ties); human–human **81%**.

**5. Terminology.** LLM-as-a-judge; pairwise / single-answer / reference-guided grading; position, verbosity and self-enhancement bias; agreement (S1/S2).

**6. Figures.** Table 2/3 as a three-bias panel. Fig. 2 (p.8): agreement rises with the win-rate gap.

**7. What it gives the deck.**
- Slides: terminology (judge vocabulary) and evaluation-protocol risk.
- Presenter sentence: "Our judge does *reference-guided single-answer grading*. It sees the question, the gold answer and one candidate, and says correct or incorrect. Position bias cannot occur because there is no pair. Verbosity and self-preference can, so we checked both: picking the longest answer loses, and a second judge from another model family reproduces about 87% of the gain."
  - The 87% is derived: XJ +0.0783 / J +0.0897 = 0.873, from COMPARISON_BRIEF §0. DESIGN_EVIDENCE quotes "92%" for the September numbers (+0.0679/+0.0737 = 0.921, derived). **Use the one matching the headline being shown.**
- **Cuts against us.** MT-Bench's own judge (GPT-4) is far larger than ours, and its agreement figures do not transfer to MedVLThinker-32B. Our judge has independent human validation on only 3 of 8 benchmarks (DESIGN_EVIDENCE l.299).

**8. Must not claim.**
- That MT-Bench measured or proved self-enhancement bias.
- That its 80%+ agreement applies to our judge or to medical answers.
- That our setting has position bias (single-answer grading has no positions).

---

## H4 (core). Xu, Wu, Ryu 2026: Controlled Audit of Pretraining Contamination

**Citation.** B. C. Xu, L. Wu, A. Ryu, "A Controlled Audit of Pretraining Contamination in Public Medical Vision–Language Benchmarks," arXiv:2606.10066v1, 8 Jun 2026 [p.1].

**1. Problem.** Public medical VQA benchmarks have been downloadable for 3–7 years. Reported accuracy assumes they were absent from pretraining. The authors test that assumption, and also test whether the detectors work [p.1].

**2. Method** [§3, p.4–6]:
- **Models:** InternVL3-8B, Qwen2.5-VL-7B-Instruct, CheXagent-8b, LLaVA-OneVision-Qwen2-7B; plus MedGemma-4B-IT for the text-side extension; BLIP-2 and InstructBLIP as non-medical controls.
- **Benchmarks:** SLAKE-En test (1,061), PathVQA test (6,719), VQA-RAD test (451) [p.4]; plus a 4,999-example public OmniMedVQA mirror ("exposes only a train split") [p.4–5].
- **Detectors:**
  1. **Image-side near-neighbour overlap** against PMC-OA-beta (1,848,719 figures) via SigLIP B-16 and SO400M embeddings. τ is the α = 0.01 quantile of within-corpus nearest-neighbour distance (τ_B16 = 0.0180, τ_SO400M = 0.0154) [p.5].
  2. **Canonical-order exchangeability** (Oren et al. 2024): $p = \frac{1+|\{\pi:\ell(\pi)\ge\ell(\pi_0)\}|}{10{,}001}$ over 10,000 permutations [p.5].
  3. **Min-K%++ tail enrichment**, cohort-relative: $\Delta_i = m_i^{\text{target}} - \mathrm{median}_{j\neq\text{target}} m_i^{j}$, K = 20%, flagged if Pr[Δ > 100] > 5% [p.5–6].
  4. **Cross-model top-K Jaccard**, K = 25 [p.6].

**3. Setup.** Negative controls: 2,000 WikiArt paintings (out-of-domain); VQA-RAD as the in-domain clean reference [§3.6, p.6].

**4. Key results.**
- **SLAKE-En image overlap.** "19.8 % … under SigLIP-B-16, 4.2 % under SO400M, versus ≤0.9 % for in-domain VQA-RAD and 0/2000 for two out-of-domain controls" [abstract p.1].
  - SO400M "is the default for headline numbers" [p.8].
  - Manual adjudication: "same-modality, same-projection matches to different patients rather than pixel-level duplicates" [p.1].
  - The effect on gold-answer likelihood is "below the cross-model noise floor": flagged-minus-unflagged gaps of −0.22 to +0.27 nats/token, with BLIP-2 at +0.03 [p.9].
- **Qwen2.5-VL × SLAKE-En exchangeability** p = 5.0×10⁻⁴. It survives the ordering ablation (hash p = 0.68) and the external baselines. This is "the only held-out text-side signal that … survives every robustness check" [p.7].
- **VQA-RAD is clean** on both signals [p.8]. The paper recommends it as "the comparatively safer held-out benchmark" [p.1–2].
- **PathVQA** exchangeability is reattributed to a release-order artefact, because BLIP-2 reproduces it [p.7–8, p.13–14].
- **OmniMedVQA mirror:** exchangeability fires for all 5 VLMs while BLIP-2 stays clean [Table 2, p.13]. The paper recommends treating the public mirror "as contaminated for text-side evaluation" [p.2].
- **Detectors 3 and 4 collapse** when BLIP-2 is added. They are "unreliable as standalone membership-inference signals on small medical-VLM cohorts" [p.1].

**5. Terminology.** contamination, benchmark leakage, membership inference, near-neighbour overlap, canonical-order exchangeability, Min-K%++, external pre-domain baseline, source/distributional overlap vs per-image memorization.

**6. Figures.** Fig. 2 (p.8): per-benchmark flag rates, two backbones, dashed α line. A clean bar chart to recreate. Fig. 4 (p.11): side-by-side SLAKE vs PMC near-neighbour pair.

**7. What it gives the deck.**
- Slides: limitations/contamination, benchmark descriptions.
- Presenter sentence: "The only controlled contamination audit of these benchmarks finds VQA-RAD clean. SLAKE shares image sources with PubMed figures, and that sharing does not measurably raise answer likelihood. PathVQA's apparent signal is an ordering artefact. The OmniMedVQA public mirror shows text-side exposure in every model tested, including Qwen2.5-VL, our generator's base."
- **Cuts against us:**
  - (a) Qwen2.5-VL, Lingshu's base, carries a SLAKE-En text-side exposure signal.
  - (b) OmniMedVQA text is likely seen in pretraining by Qwen-family models.
  - (c) Lingshu itself was not audited.
  - All of this affects absolute accuracy more than our greedy-vs-probe Δ (inferred). The Δ compares two outputs of the same model on the same question.

**8. Must not claim.**
- That 19.8% of SLAKE images are duplicates or memorised.
- That Lingshu was audited.
- That Min-K%++-style signals show contamination.
- That PathVQA is contaminated (the audit says artefact).
- **Headline-number choice:** the paper's default is SO400M (4.2%). If a single number is quoted, quote 4.2% with 19.8% as the B-16 sensitivity, or both.

---

# Benchmark sheet — all 8 of our open-ended benchmarks (for one comparison table)

"Our n" values come from COMPARISON_BRIEF §0 (`final_eval_FINAL_cap320_Tmap.json → cells.*`). Source-paper facts are tagged.

| # | benchmark (our cell) | source paper | images / modality | how QA were made | answer style | published split sizes | our cell |
|---|---|---|---|---|---|---|---|
| 1 | **PathVQA-open** | He et al. 2020, arXiv:2003.10286 | pathology images from textbooks / digital libraries | "semi-automated pipeline … uses NLP to generate question-answer pairs" from captions; every question manually checked [abstract, fetched] | open (what/where/how…) + yes/no closed [LLaVA-Med §5.2 text] | 4,998 images / 32,799 QA [abstract]. Test: 858 images, 6,761 QA, 3,370 open (LLaVA-Med Table 3); MedEvalKit test 6,719 total / **3,357 open** [Lingshu Table 25, p.53] | 3,357 (= MedEvalKit open test) |
| 2 | **SLAKE-open (English)** | Liu et al. 2021 (ISBI), arXiv:2102.09542 | 642 images (282 CT, 181 MRI, 179 X-ray) [card, html] | physicians, template-assisted, bilingual EN/ZH [card] | short open + closed | 14,028 QA total [card]. English test: 96 images, 1,061 QA, **645 open** (LLaVA-Med Table 3) | 645 (= English open test) |
| 3 | **VQA-RAD-open** | Lau et al. 2018, *Scientific Data* | 315 MedPix images: head CT/MRI, chest X-ray, abdominal CT [card] | clinician-authored free-form + paraphrases [card] | short | 3,515 QA [card; OmniMedVQA Table 1 p.1]. Test 451 (300 free-form + 151 paraphrased) [card]; MedEvalKit test 451 = 251 close + **200 open** [Lingshu Table 25] | 200 (= MedEvalKit open test) |
| 4 | **RadImageNet-VQA** | Butsanets et al. 2025, arXiv:2512.17396 | RadImageNet **CT and MRI** subsets ("we use the CT and MRI subsets") [§3.1, fetched] | from expert-curated annotations; 3 tasks (abnormality detection, anatomy recognition, pathology identification) [abstract] | open, closed (±), MCQ [abstract] | ~750K images / 7.5M samples train; benchmark 1,000 images / 9,000 QA from the RadImageNet test split [§3.1–3.2, fetched]. Open graded by **Mistral-Large 2.1 judge**, 90.6% agreement with 2 humans on 380 samples, κ = 0.72 [App. B.1, fetched] | 2,000 q / 1,000 images [repo `OPENTEXT_CELL_SURVEY` l.32]: 2 open questions per benchmark image (inferred: anatomy-open + pathology-open) |
| 5 | **Kvasir-VQA-x1** | Gautam, Riegler, Halvorsen 2025, arXiv:2506.09958 | GI endoscopy (HyperKvasir / Kvasir-Instrument via Kvasir-VQA, 6,500 images [Kvasir-VQA abstract]) | **Qwen3-30B-A3B** merges 1–3 atomic Kvasir-VQA QA pairs and "naturalizes" answers; complexity levels 1/2/3 [§2.1–2.3, fetched] | **sentences** ("fluent, medically appropriate natural language") [fetched] | 159,549 new QA [abstract]; level totals 54,856 / 52,349 / 52,344 [Table 4, fetched]; train/test sizes **not reported** [fetched]; judge = Qwen3-30B-A3B [fetched] | 10,121 q / 2,870 images from the official test, minus 1,052 contaminated frames + 136 near-duplicates [repo `OPENTEXT_FULL_RUNDOWN` §2] |
| 6 | **OmniMedVQA (open)** | Hu et al. 2024, arXiv:2402.09181 | 73 classification datasets, 12 modalities [p.1] | templates + GPT-3.5 rephrasing/distractors; human double check [p.3–5] | native MCQ (2–4 options) | 127,995 items / 118,010 images [Table 3, p.5]; MedEvalKit open-access part 88,996 [Lingshu Table 25] | 8,883 q: options stripped, modality & anatomy types only, **RadImageNet source dropped**, 7 modalities [repo `OPENTEXT_FULL_RUNDOWN` §2] |
| 7 | **VQA-Med 2019 C4 (abnormality)** | Ben Abacha et al. 2019, CLEF Working Notes | MedPix radiology [vqamed2019 p.3] | auto-generated from patterns drawn from VQA-RAD questions; test manually double-validated (15 answers = 3% corrected) [p.3–5] | category C4 abnormality = "answer generation problem" [p.2] | train 3,200 images / 12,792 QA; val 500 / 2,000; test 500 images / 500 Q [p.5–6]; metrics strict accuracy + BLEU; best team 0.624 acc / 0.644 BLEU [p.1, p.6] | 3,663 q, **mostly the 2019 training split**; 10 MedPix collisions excluded [repo]; official test (500) scored separately |
| 8 | **GEMeX (open)** | Liu et al. 2024, arXiv:2411.16778 | chest X-ray, MIMIC-CXR via Chest ImaGenome [§3.1, fetched] | **GPT-4o (2024-08-06)** from re-grounded reports [§3.2, fetched] | concise answer + reason + box (e.g. "Linear atelectasis") [fetched] | 151,025 images / 1,605,575 Q [abstract]; train 149,535 img / 1,588,975 Q; val 1,190 / 12,640; test 300 / 3,960; open-ended 441,471 / 3,524 / 1,144 [Table 2, fetched] | 8,000 q / 3,514 images from the **GEMeX-ThinkVG** release, open-ended rows, gold ≤ 12 words [repo `src/data_prep/build_gemex_cell.py` docstring]; not the 300-image GEMeX test |

**Notes for the benchmark slide**

- **Three published judges besides ours.** RadImageNet-VQA (Mistral-Large 2.1), Kvasir-VQA-x1 (Qwen3-30B-A3B) and GEMeX (GPT-4o GPTScore) each grade open answers with an LLM [fetched]. Lingshu/MedEvalKit uses GPT-4.1 [p.16]. VQA-Med 2019 and LLaVA-Med use strict accuracy / token recall. **LLM-as-a-judge is the field norm for open-ended medical VQA.** That justifies our protocol, but every paper's judge differs.
- **Kvasir-VQA-x1 and GEMeX golds are LLM-written sentences.** That is why EM loses there while both judges gain (COMPARISON_BRIEF §0).
- **Kvasir-VQA-x1's judge is Qwen3,** and the paper itself notes self-enhancement risk with Qwen2.5-VL [§4.2.5, fetched]. Parallel to our MedVLThinker-32B × Lingshu risk.
- **Training-split exposure of Lingshu-7B** [Lingshu Table 5, p.12]:
  - PathVQA, SLAKE, VQA-RAD and VQA-Med-2019 training splits are in Stage 3.
  - RadImageNet *ultrasound* images were used for synthetic data [Table 3, p.6]; our RadImageNet-VQA cell is CT/MRI.
  - Kvasir-VQA appears only in the RL stage, i.e. Lingshu-RL.
  - OmniMedVQA is not listed as training data. The public mirror carries a text-exposure signal for Qwen2.5-VL [audit Table 2, p.13].
  - GEMeX is not listed. MIMIC-CXR images *and reports* are in Stages 2–3 (Table 5). GEMeX's QA are generated from MIMIC-CXR-derived reports, so report-level exposure is plausible (inferred).

---

# Part 2 — Every card in G and H

## Category G (21 cards)

| id | title | one line | deck relevance | why |
|---|---|---|---|---|
| 2506.07044 | Lingshu | our generator; 3 SFT stages; MedEvalKit | **slide-worthy** | generator, training exposure, protocol |
| 2507.05201 | MedGemma | Gemma 3 + MedSigLIP; 27B multimodal = our judge | **slide-worthy** | cross-family judge; linear-probe precedent |
| 2502.13923 | Qwen2.5-VL | architecture of our generator | **slide-worthy** | architecture figure |
| 2507.11200 | How far have medical VLMs come | independent benchmark of Lingshu/Qwen | mention | protocol sensitivity; card claim must be fixed |
| 2604.05081 | MedGemma 1.5 | 3D/WSI/EHR successor | skip | out of scope |
| 2508.02669 | MedVLThinker | RLVR recipe; **our judge's family** | mention | judge provenance (MedVLThinker-32B) |
| 2604.13756 | MedRCube | rates Lingshu-32B top-tier; shortcut warning | mention | external standing of Lingshu |
| 2511.00916 | Fleming-VL | 2D/3D/video medical MLLM | skip | landscape only |
| 2510.08668 | Hulu-Med | transparent generalist | skip | landscape only |
| 2303.15343 | SigLIP | sigmoid contrastive loss | mention | MedGemma encoder lineage |
| 2503.19786 | Gemma 3 | MedGemma backbone | mention | judge backbone |
| 2504.10479 | InternVL3 | open general VLM | skip | landscape |
| 2306.00890 | LLaVA-Med | biomedical LLaVA; recall metric | **slide-worthy** (comparison) | our LLaVA-recall currency and Table 3 split sizes |
| 2406.19280 | HuatuoGPT-Vision | PubMedVision | skip | landscape (in Lingshu's data) |
| 2305.17100 | BiomedGPT | generalist biomedical | skip | |
| 2307.15189 | Med-Flamingo | few-shot medical VLM | skip | |
| 2307.14334 | Med-PaLM M | generalist biomedical | skip | |
| 2401.12208 | CheXagent | CXR specialist | skip | (appears in the contamination audit cohort) |
| 2304.08485 | LLaVA | visual instruction tuning | mention | terminology: encoder → projector → LLM |
| 2310.03744 | LLaVA-1.5 | MLP projector | mention | origin of the MLP projector |
| 2409.12191 | Qwen2-VL | dynamic resolution, M-RoPE | mention | origin of M-RoPE / dynamic resolution |

**Verified content for slide-worthy / mention cards**

- **LLaVA-Med (2306.00890).**
  - Open-set metric: "we use recall to evaluate the ratio that ground-truth tokens appear in the generated sequences" [§5.2 text, html].
  - Split statistics (Table 3, html extraction):
    - VQA-RAD test: 203 images / 451 QA / 179 open.
    - SLAKE-En test: 96 / 1,061 / 645 open.
    - PathVQA test: 858 / 6,761 / 3,370 open.
  - The VQA-RAD and PathVQA open counts differ from MedEvalKit's 200 / 3,357 [Lingshu Table 25]. **Say which split definition each number uses.**
- **MedVLThinker (2508.02669)** [abstract via card]. RLVR on Qwen2.5-VL 3B/7B/32B; "RLVR consistently and significantly outperforms SFT"; text-only reasoning data helps more than image-text data.
  - Deck use: one line on the judge slide, "Judge = MedVLThinker-32B, an RLVR-trained Qwen2.5-VL-32B medical model; text-only use, sees the gold."
  - The judge's base model being Qwen2.5-VL-32B is inferred from the card's "on Qwen2.5-VL 3B/7B/32B". Verify against the paper before printing.
- **MedRCube (2604.13756)** [abstract via card]: "we benchmark 33 MLLMs, Lingshu-32B achieve top-tier performance"; and a "highly significant positive association between shortcut behavior and diagnostic task performance". One background bullet.
- **SigLIP / Gemma 3** [card]. MedGemma's vision tower is SigLIP-400M [MedGemma p.4]. Gemma 3 uses "a 5:1 interleaving of local/global layers" [card, html]. The Gemma 3 4B hidden size / layer count is **not verified** (card's own warning).
- **LLaVA / LLaVA-1.5** [abstracts via cards]. LLaVA-1.5's change: "swap the connector for an MLP". This is the terminology origin of "MLP projector", which Qwen2.5-VL calls a "merger".
- **Qwen2-VL** [abstract via card]: "Naive Dynamic Resolution … into different numbers of visual tokens"; "Multimodal Rotary Position Embedding (M-RoPE)". Cite it as the origin of both terms; Qwen2.5-VL extends M-RoPE to absolute time [Qwen2.5-VL p.5].
- **Liu 2025.** See G4. Use only as evidence that protocol moves Lingshu-7B's scores by about 10 points.

## Category H (23 cards)

| id | title | one line | deck relevance | why |
|---|---|---|---|---|
| 2003.10286 | PathVQA | caption-derived pathology VQA | **slide-worthy** (benchmark table) | one of our 8 |
| 2102.09542 | SLAKE | bilingual physician-templated radiology VQA | **slide-worthy** (table) | one of our 8 |
| VQA-RAD (Sci. Data 2018) | VQA-RAD | clinician-authored radiology VQA | **slide-worthy** (table) | one of our 8; audit says cleanest |
| 2512.17396 | RadImageNet-VQA | CT/MRI, shortcut-resistant | **slide-worthy** (table) | one of our 8 |
| 2506.09958 | Kvasir-VQA-x1 | LLM-composed GI QA, sentence answers | **slide-worthy** (table) | one of our 8; explains EM loss |
| 2409.01437 | Kvasir-VQA | parent dataset | mention | provenance; Lingshu-RL data |
| 2402.09181 | OmniMedVQA | 73 classification sets → MCQ | **slide-worthy** | one of our 8 (converted) |
| VQA-Med 2019 | ImageCLEF overview | 4 categories; C4 = generation | **slide-worthy** (table) | one of our 8 |
| 2411.16778 | GEMeX | CXR VQA with explanations | **slide-worthy** (table) | one of our 8 |
| 2305.10415 | PMC-VQA | caption-generated VQA, option skew | mention | Lingshu training data; artefact example |
| 2306.05685 | MT-Bench | LLM-as-a-judge + biases | **slide-worthy** | judge terminology |
| 2404.13076 | Self-recognition / self-preference | judges favour own generations | mention | same-family judge risk |
| 2406.12624 | Judging the Judges | leniency; score vs rank alignment | mention | why EM is reported beside judges |
| 2402.07270 | Open-ended VQA from classification | metric validation vs humans | mention | same move as OmniMedVQA-open |
| 2405.03162 | Med-Gemini | Google medical Gemini | skip | only the VQA-RAD re-split question |
| 2405.20421 | ProbMed ("Worse than random") | negation probing | mention | future stress-test |
| 2311.16502 | MMMU | exam-style benchmark | skip | not in our live 8 |
| 2501.18362 | MedXpertQA | expert MCQ | skip | not in our live 8 |
| 2606.10066 | Contamination audit | controlled detector audit | **slide-worthy** | contamination slide |
| 2607.12278 | WSI leakage audit | 92.3–100% case overlap | mention | field context (not our data) |
| 2312.07867 | BESTMVQA | eval harness tool | skip | |
| 2607.00159 | KB-VQA pitfalls | benchmark repair | skip | general domain |
| MedEvalKit | harness | Lingshu's eval kit | mention | source of 3 of our test splits |

**Verified content for slide-worthy / mention cards (not already covered above)**

- **Self-preference (2404.13076)** [abstract via card]: "LLMs such as GPT-4 and Llama 2 have non-trivial accuracy at distinguishing themselves from other LLMs and humans"; "a linear correlation between self-recognition capability and the strength of self-preference bias".
  - Our mitigation is a cross-family judge.
  - The project's own check: P(judge = 1 | EM = 0) is Lingshu 0.1389, Qwen 0.1674, MedGemma-4b 0.1659 (DESIGN_EVIDENCE l.297), so no same-family preference is visible.
- **Judging the Judges (2406.12624)** [abstract via card]: only the largest judges align reasonably with humans, scores "may still differ with up to 5 points"; leniency bias; small models may still rank well. Supports reporting EM / F1 beside the judge.
- **Open-ended VQA from classification (2402.07270)** [abstract via card]. ICLR 2024. Turns classification datasets into open-ended VQA and validates the metric against humans. This is exactly what our OmniMedVQA-open conversion does, so cite it as precedent for the conversion.
- **ProbMed (2405.20421)** [abstract via card]: "top-performing models like GPT-4o, GPT-4V, and Gemini Pro perform worse than random guessing on specialized diagnostic questions". Future-work bullet: negation-probe our pools (not run).
- **WSI leakage (2607.12278)** [abstract via card]: "case level train test overlaps of 92.3~100% on TCGA-derived benchmarks". Context only. Not PathVQA, which is textbook figures.
- **Kvasir-VQA (2409.01437)** [abstract via card]: 6,500 images; yes/no, choice, location, count questions. These atomic QA are what Kvasir-VQA-x1 merges [fetched].
- **MedEvalKit** [Lingshu §4, p.14–16; Table 25, p.53]. 16 benchmarks, 152,066 samples. Open-ended graded by GPT-4.1 with the Table 24 prompt. The card's "exact match with optional LLM-judge" is inaccurate for open-ended questions.

---

# Part 3 — Recommendations

**Papers that deserve their own slide**

1. **Qwen2.5-VL → Lingshu-7B architecture.** One figure:
   - image → ViT (32 layers, 14-px patches, window attention, full attention at blocks 7/15/23/31);
   - 2×2 MLP merger → ≤ max_pixels/784 tokens of width 3,584;
   - 28-layer LLM with M-RoPE;
   - probe taps at 18/20/22.
   Sources: Qwen Table 1 p.4, §2.1 p.3–5, config. Tag the token count "(derived)".
2. **Lingshu training stages and data exposure.** A 4-stage strip with 🔥 icons (Fig. 6 style), Table 5 amounts, and highlighted benchmark training splits. "Lingshu = stages 1–3; RL → Lingshu-RL (not used)". Add the Table 10 ablation as one bar: PathVQA 61.9 → 45.2 without medical multimodal data.
3. **The 8 benchmarks.** The table above, plus one example image/QA per benchmark.
4. **Evaluation protocol and judge terminology.** MT-Bench vocabulary (reference-guided single-answer grading; position / verbosity / self-enhancement bias), our two judges, and the field's judges (GPT-4.1, Mistral-Large 2.1, Qwen3-30B, GPT-4o).
5. **Contamination.** The audit's three regimes:
   - SLAKE: source overlap, no likelihood effect;
   - VQA-RAD: clean;
   - PathVQA: artefact;
   - OmniMedVQA: text exposure.
   Plus Lingshu's training-split exposure. Closing line: our Δ is within-model, so exposure affects both arms equally (stated as reasoning, inferred).

**Comparison table (not their own slide)**
- Lingshu Table 6 (with the protocol column).
- Liu 2025 Table 1, as a protocol-sensitivity example.
- MedGemma Table 9 / A14.
- LLaVA-Med Table 3 split sizes.

**Terms the deck must define**
- VLM/MLLM/LVLM.
- vision encoder (ViT).
- patch.
- projector / merger / connector.
- image token.
- `max_pixels` / dynamic resolution.
- hidden size; layer / residual stream.
- M-RoPE.
- SFT; RL / RLVR / GRPO.
- greedy decoding.
- open-ended vs closed-ended vs MCQ.
- LLM-as-a-judge; reference-guided grading.
- position / verbosity / self-enhancement bias.
- cross-family judge.
- exact match, token F1, token recall (LLaVA-Med).
- contamination / leakage; source overlap vs memorisation.
- linear probe (MedGemma §6 precedent) vs MLP probe.

**Errors found in cards and briefs (with evidence)**

1. **COMPARISON_BRIEF §1.4 contamination note** lists the RL-stage data (including Kvasir-VQA) as something "the generator has seen".
   - Lingshu Fig. 6 caption (p.11) and §3.1 (p.10) say released Lingshu = first three SFT stages; RL produces Lingshu-RL.
   - Fix: Stage-3 exposure = PathVQA, PMC-VQA, SLAKE, VQA-Med-2019, VQA-RAD (training splits). Kvasir-VQA appears only in Lingshu-RL's data.
   - The brief also omits Lingshu's dedup-against-MedEvalKit-test claim (§2.3, p.9). It should be mentioned, with the note that it does not cover VQA-Med-2019.
2. **G card 2507.11200 (Liu).**
   - "independent, external replication of our own retrospective finding #1": wrong. Its reasoning scores exceed understanding scores in most cells (Tables 2–3, p.4), contradicting its own abstract.
   - It uses a "\boxed{}" step-by-step prompt for all models [p.3]. Per CLAUDE.md §0, that is itself a reasoning trigger.
3. **G card 2502.13923 (Qwen2.5-VL)** says the abstract "studies scaling laws for LVLMs". The Qwen2.5-VL abstract (p.1) does not mention scaling laws. That sentence belongs to Qwen2-VL.
4. **G card 2507.05201 (MedGemma)** omits that MedGemma-27B-it (our cross-family judge, multimodal per local config) is described only in Appendix F / Table A14 (p.60). It also says MedGemma-27B multimodal-only MedXpertQA was "not located". Table A14 gives MedXpertQA (text + MM) 26.8 for the 27B multimodal.
5. **H card 2306.05685 (MT-Bench)** says the paper "coins 'self-enhancement/self-preference bias'" and measures it.
   - The paper adopts the term from social cognition and says "our study cannot determine whether the models exhibit a self-enhancement bias" [p.5].
   - The ">80% agreement" bar is GPT-4 vs humans on general chat [p.2–3, Table 5 p.8], not a sanity bar transferable to a 32B medical judge.
6. **H card 2402.09181 (OmniMedVQA)** says "No numeric accuracy figures were extracted". Table 5 (p.7) gives them; BLIP-2 is best at 50.69 QA-score. The paper's §1 sentence "medical-specialized LVLMs exhibit superior performance" (p.2) contradicts its abstract and tables.
   - Also: the card's relation_to_us omits that RadImageNet (55,443 images) is part of OmniMedVQA, and that our cell drops it.
7. **H card VQA-Med 2019.** Test-set size "not extracted": it is 500 images / 500 questions [vqamed2019 p.6]; validation 500 / 2,000 [p.5].
8. **H card 2606.10066 (audit) and H card SLAKE** quote "19.8%" alone. The paper's default headline backbone is SO400M, 4.2% [p.8]. Quote both, plus the adjudication (different patients) and the null likelihood effect [p.9].
9. **H card MedEvalKit** describes scoring as "exact match with optional LLM-judge". Per Lingshu §4.2 (p.16), open-ended questions are graded directly by GPT-4.1; MCQ uses rule-based matching then the MMMU codebase.
10. **H card 2506.09958 (Kvasir-VQA-x1)** leaves generation and grading unnamed. The fetched paper says Qwen3-30B-A3B for both, with self-enhancement explicitly flagged for Qwen2.5-VL [§4.2.5, fetched]. Re-check against the PDF.
11. **H card 2411.16778 (GEMeX).** Our cell is not from the GEMeX benchmark test (300 images). It comes from the GEMeX-ThinkVG release's open-ended rows [repo `build_gemex_cell.py`]. Any comparison to GEMeX's published Table 4/5 numbers is invalid.
12. **FIGURE_STYLE_NOTES §(c)** writes "320 at cap320 = 250,880 / 28²". Correct as an upper bound, but it should say "≤ 320" (actual counts vary; GEMeX averages 303).
13. **Paper vs config inconsistencies to footnote if architecture numbers are printed.** ViT intermediate 3456 (paper) vs 3420 (config); vocabulary 151646 (paper) vs 152064 (config).

**Gaps I could not close (not stated / not verified)**
- Train/test sizes of Kvasir-VQA-x1 are not reported by the fetched paper.
- RadImageNet-VQA open-ended numbers. The summariser reported Lingshu-7B anatomy-open 49.6% / pathology-open 15.7% and Qwen2.5-VL-7B 37.5% / 9.8% [Table 2, fetched], with a warning about table-layout parsing. **Do not print until checked against the PDF.**
- PathVQA's official split sizes come from LLaVA-Med Table 3 (secondary), not the PathVQA paper (abstract only).
- Whether our 8,883 OmniMedVQA questions fall inside Lingshu's deduped MedEvalKit OmniMedVQA set is not established.
- The base model of MedVLThinker-32B (our judge) comes from its card's abstract only, not read in full.
