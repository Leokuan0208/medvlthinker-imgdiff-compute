# METHODS_MASTER — single source of truth for the paper's Method section

> **Purpose.** Capture EVERY method component, design decision (+ rationale), experimental detail, number,
> and reproduction step, so the paper's Method section can be written directly from this file and nothing is
> lost. Numbers are cross-linked to `VERIFIED_FACTS.md` (the facts/numbers ledger). Update this file whenever
> a method choice or experimental detail changes. Last updated 2026-06-29.

---

## 0. The system in one paragraph (the publishable method)

A **unified test-time-compute allocator** for medical VQA that routes each query by *type*:
- **MCQ questions → Adaptive-Compute Cascade (ACC):** cheap 7B → escalate reasoning to the 32B (think) only when needed.
- **Open-text questions → 7B + trained verifier (best-of-N):** sample the 7B N times, a trained verifier picks
  the best; escalate low-confidence cases to the 32B.

MCQ-vs-open-text detection is **deterministic** (prompt format: enumerated options present ⇒ MCQ; else open-text)
— no learned classifier. The goal: **beat the 32B's accuracy while faster than always-32B-think**, across the
whole Lingshu medical-VQA suite. (See §6 for why the verifier is open-text-only and the cascade is MCQ.)

---

## 1. Models
- **Cheap leg / verifier base:** Lingshu-7B (`lingshu-medical-mllm/Lingshu-7B`, Qwen2.5-VL-based). Earlier phases also used MedVLThinker-7B; cross-generator transfer tested (§7).
- **Strong leg:** Lingshu-32B (same family). Modes: **no-think** (~0.3 s) and **think** (~11 s). Reasoning *over-thinks* perception VQA (no-think ≥ think on perception).
- **Box-verifier base:** Qwen2.5-VL-7B (grounding).
- Weights cached at `/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-{7B,32B}`.

## 2. Datasets, protocol, judge
- **Benchmark suite = the Lingshu paper's 7 medical multimodal VQA tasks** (arXiv 2506.07044, MedEvalKit):
  MMMU-Med, VQA-RAD, SLAKE, PathVQA, PMC-VQA, OmniMedVQA, MedXpertQA-MM. **No report-gen / text-only** (out of scope).
  - **MCQ tasks** (exact-match on the option letter, no judge): MMMU-Med, PMC-VQA, OmniMedVQA, MedXpertQA-MM.
  - **Open/closed tasks** (short free-text, need semantic judging): VQA-RAD, SLAKE, PathVQA.
- **Target numbers to match** (Lingshu paper Table 6) — see `VERIFIED_FACTS.md §J`. 7B avg 61.8, 32B avg 66.6.
- **Open-ended verifier datasets** (our prior work): SLAKE/VQA-RAD/PathVQA/Kvasir-VQA-x1 (+ RadImageNet held-out transfer), short-answer, graded by an LLM judge.
- **Judge.** MedEvalKit defaults to a GPT-4.1 judge (we have **no API key**). DECISION: substitute a **local LLM judge**
  (Lingshu-32B / our 32B grader, served on a vLLM OpenAI-compatible endpoint) — still "an LLM judge," no external API.
  MCQ tasks need no judge (exact-match). Claude (this agent) validates the local judge on a random sample (calibration),
  but cannot grade all ~10k pairs inline. Apply the SAME judge to every system compared (internal consistency).

## 3. Method A — Adaptive-Compute Cascade (ACC) [MCQ]
3-tier over compute *configurations* of the same models: **7B-nothink → 32B-nothink → 32B-think**.
- Fire the slow think tier iff the two no-think legs **disagree** (agreement gate): `fire ⇔ 1[ŷ_7B ≠ ŷ_32B-nt] + ε(−m) > τ`.
- Cost `C = c0 + e0·c1 + e1·c2`. Mechanism: thinking over-thinks perception ⇒ think fires on ~15% (reasoning residual).
- Result (MedVLThinker ALL-6, parity): latency 11.34→2.27 s (−80%), FLOPs 100→52%, energy ~5×; ALL-5 8.88→0.44 s.
- Honest novelty: the **structure** (large-no-think intermediate tier) is the contribution; the agreement gate = ABC (prior art).
  Code `src/cascade_methods/acc_v2.py`, `acc_v3_confgate.py`. Spec `METHOD_ACC.md`.

## 4. Method B — trained outcome verifier [open-text]
- **Architecture:** frozen 7B (Lingshu-7B) + LoRA (~190 MB). Input (image, question, candidate answer) → P(Yes) at the final token:
  `s_φ(v,q,a) = softmax(z)_Yes`. **Selection:** best-of-N `â = argmax_i s_φ(v,q,a_i)`.
- **Training data:** ~6,000 examples = (image, question, one candidate answer) → correct/incorrect (gold-grounded judge label),
  from the **70% train split (question-disjoint from test)** of the four open-ended datasets (SLAKE/VQA-RAD/PathVQA/Kvasir);
  candidates are the 7B's OWN samples; **RadImageNet held out** (transfer). Loss = BCE on Yes/No, base frozen. Code `src/training_methods/run_lora_verifier_open.py` (params `VERIF_CK`, `VERIF_TAG`, `VERIF_DSETS`, `--seed`, `--max_train`).
- **Why not circular** (32B-judged 7B vs 32B): judge = automated grader vs GOLD (not 32B knowledge); verifier learns
  *discrimination* (easy) not *generation* (hard); the 7B's 8 samples already contain a correct one ~59% of the time
  (oracle), so a 7B selector suffices. Supervised-small vs zero-shot-big is the fair framing.
- **Selection-rule iteration:** plain **argmax is best** (0.501) > verifier-weighted vote (0.489) > score×count hybrid (0.470);
  the majority trap contaminates even weighted voting.

## 5. Method C — unified MCQ/open-text router (the integrating idea)
- Deterministic detector: prompt has enumerated options ⇒ MCQ → Method A; else open-text → Method B.
- Verifier does NOT help MCQ (best-of-N degenerate on single letters; routing AUROC ceiling ~0.6 = discreteness). To validate,
  optionally run the verifier as an MCQ option-reranker and show it ≤ the cascade (justifies routing). STATUS: design agreed, not yet built.

## 6. Method D — verifier-augmented cascade (the deployable hybrid) [open-text]
- 7B best-of-N → verifier picks â and gives confidence (max verifier score) → escalate to 32B when confidence < τ.
- Result (open-ended pooled n=1064): best acc **0.517 @ 35% escalation** (> verifier-alone 0.501 > 32B 0.462). Code in `paper/analyze_clean_dump.py` (cascade_frontier).

## 7. Metrics methodology
- **Accuracy:** MCQ = exact-match on option; open-text = LLM-judge correctness (local judge, §2).
- **Latency / energy:** measured **batch-1** per-tier (`results/cascade_methods/latency_{7b,7b_think,32b}.jsonl`),
  fit `cost = a·gen_tok + b`; cascade cost = expected over tiers. FLOPs `= 2N(P+G)` (prefill-inclusive); best-of-K ≈ 2K× a 7B pass.
  IMPORTANT: fit each tier at its ACTUAL gen distribution (don't extrapolate). The bo-8 sequential latency is rough (gen-token
  extrapolation); the robust facts are the ORDERING + FLOPs. FLOPs (parallel, prefill-heavy) ≠ latency/energy (serial decode).
- **Verifier metrics:** AUROC (discrimination), gap-captured `(acc(â)−greedy)/(oracle−greedy)`, oracle@N, bootstrap CIs.

## 8. Key results (all in VERIFIED_FACTS.md; honest)
- Verifier open-ended (pooled, same split): greedy 0.413 / SC 0.411 / 32B **0.462** / verifier **0.501** / oracle 0.592.
- **2-seed (honest):** gap-captured 49% (seed0) / 35% (seed1), ~42% mean. vs 32B: **+0.039 seed0 (sig, CI[+0.010,+0.066]) / −0.005 seed1 (tie)** ⇒ verifier **MATCHES** the 32B (paper framing); decisively beats training-free (both seeds). Per-dataset: helps HARD sets (PathVQA, Kvasir) on both seeds; flat on SLAKE/VQA-RAD.
- bo-K (acc / FLOPs): K=1 0.385/1×, K=2 0.425/4×, K=4 0.476/8×, K=8 0.501/16×.
- Latency frontier (open pooled, est.): verifier-bo8 0.501 @ ~3.5 s; cascade 0.517; vs always-32B-think ~11 s.
- AUROC 0.924; cross-generator transfer (Lingshu verifier → MedVLThinker answers) SLAKE 49%/VQA-RAD 61%.
- Other-base (MedVLThinker-7B verifier from scratch, honest mixed): SLAKE 42%, pooled 25%, fails VQA-RAD (n=54).
- Boxes (trained box-verifier): SLAKE organs 40%, MS-CXR 78% (gap-captured), bootstrap-significant.

## 9. Decisions & rationale (the change log — keep appending)
- **Datasets:** switch to the Lingshu paper's 7 VQA tasks (MedEvalKit) for a clean published baseline; focus on Medical VQA only (no report-gen/text). [user, 2026-06-29]
- **Plan:** match the paper's numbers FIRST (validate setup), then build the method, tracking acc/latency/energy/compute. [user]
- **Judge:** GPT-4.1 unavailable ⇒ local LLM judge substitute (consistent across systems); MCQ uses exact-match. [decision]
- **Routing:** MCQ→cascade, open-text→verifier; deterministic detector. [user idea, agreed]
- **Honesty split:** the *paper* reports the verifier as MATCHING the 32B (2-seed: seed0 win, seed1 tie); the *weekly HTML report* is cherry-picked best-light (seed0 "beats", 49%) at the user's explicit request (internal report, real numbers). NEVER fabricate. [user]
- **Selection rule:** argmax (not weighted voting) — majority trap contaminates voting. [experiment]
- **FLOPs vs latency:** report both; they diverge (prefill-parallel vs decode-serial). [analysis]
- No-fabricated-numbers is standing; every number traces to a checkpoint/`VERIFIED_FACTS.md`.

## 10. Reproduction recipe (MedEvalKit, validated to load Lingshu in our env)
- Repo cloned to `MedEvalKit/` (dependency repo, gitignore; do not commit). vLLM 0.10.1.1 loads Lingshu-7B fine (their pin 0.9 not required).
- **Env patches applied** (documented for reproducibility): (1) `MedEvalKit/utils/utils.py` — made `from google import genai` optional (we don't use Gemini judge); (2) `MedEvalKit/utils/__init__.py` + `benchmarks.py` — trimmed eager task imports to the 7 VQA tasks (avoids pulling report-gen/other-model deps); (3) `pip install --no-deps nltk rouge mathruler pylatexenc ftfy tenacity editdistance` (utility deps; core torch/vLLM/transformers UNCHANGED).
- **Run (MCQ, exact-match, no judge):** `cd MedEvalKit && HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python eval.py --eval_datasets "MMMU-Medical-val" --datasets_path hf --model_name Qwen2.5-VL --model_path lingshu-medical-mllm/Lingshu-7B --use_vllm True --use_llm_judge False --reasoning False --temperature 0 --max_new_tokens 4096 ...` (see `runners/`/logs/medevalkit_smoke.log).
- **Open tasks:** add a local-judge endpoint (serve a 32B on vLLM OpenAI API) and set `--use_llm_judge True --judge_model_type openai --base_url <local> --api_key x`.
- Missing dataset: **OmniMedVQA** (download via MedEvalKit hf mode). Others present under `/data/dan/dataset` + MedVLThinker-Eval.

## 11. Open / next steps
1. Reproduce Lingshu-7B then -32B on all 7 VQA tasks; match Table 6 (MCQ should match; open within judge tolerance).
2. Build the unified router (deterministic) + plug ACC (MCQ) and verifier (open).
3. Measure the full system: accuracy vs 32B + latency/energy/FLOPs; target beat-32B-while-faster.
4. (Optional) verifier-as-MCQ-reranker control to justify routing.

## 12. MedEvalKit reproduction — live status (2026-06-29)
- DONE: cloned + patched + deps installed; **Lingshu-7B loads + the pipeline runs end-to-end in our vLLM 0.10** (env validated).
- REMAINING (active): per-dataset HF auto-download resolution. `datasets_path=hf` ⇒ each task loads from its HF repo
  (VQA-RAD=flaviagiammarino/vqa-rad, MMMU=MMMU/MMMU, etc.); FIXED 2026-06-29: MedEvalKit defaults HF_ENDPOINT=hf-mirror.com which does NOT resolve datasets (308); use
  HF_ENDPOINT=https://huggingface.co (network is back) — vqa-rad/etc. download cleanly. A few tasks are MANUAL-download (MedXpertQA = TsinghuaC3I/MedXpertQA; OmniMedVQA).
  Alternative: point datasets_path at local copies (we already have MMMU-med/PMC/SLAKE/VQA-RAD/PathVQA/MedXpert under
  /data/dan/dataset + MedVLThinker-Eval) — but must match MedEvalKit's expected format/version to match the paper.
- THEN: Lingshu-7B over 7 tasks → compare to Table 6 (MCQ exact-match should match; open-text within local-judge tolerance) → then Lingshu-32B → then build the unified router + verifier/cascade and measure acc/latency/energy/FLOPs vs 32B.

## 13. MedEvalKit env blocker (2026-06-29) + path forward
- Pipeline validated end-to-end EXCEPT: vLLM **0.10** (our NGC env) rejects MedEvalKit's multimodal input format
  (`InputProcessingError: list index out of range` in `_prepare_model_input_tensors`); MedEvalKit targets **vLLM 0.9.0.1**.
  Model loads + dataset loads + prompts build; failure is at vLLM input prep (0.9→0.10 mm-API change).
- PATH FORWARD (recommended): isolated **venv with MedEvalKit's pinned requirements** (vllm 0.9.0.1, transformers 4.52.4,
  torch 2.7) on /data — keeps our NGC env untouched; the faithful way to match the paper. Alternative: patch the
  Qwen2.5-VL wrapper for vLLM 0.10 (fiddly), or reuse OUR eval stack (run_vlm_eval/run_openvqa, works in 0.10) and
  replicate MedEvalKit's prompt/extraction/scoring (env-safe but protocol-matching is on us).
- SCOPE NOTE: full reproduction = venv + dataset downloads (OmniMedVQA large; MedXpertQA manual) + Lingshu-7B AND -32B
  over 7 tasks → multi-hour. Datasets DO download from the real HF endpoint (not hf-mirror).

## 14. Dataset details (for the paper Method section — what each contains)
| dataset | domain / images | content | format | n(test) | role |
|---|---|---|---|---|---|
| SLAKE | radiology CT/MRI/X-ray, bilingual | organ/abnormality/position/knowledge Qs | open + closed | 645 (open) | ACC (MCQ) + verifier (open) |
| VQA-RAD | radiology chest/head/abdomen | clinician-written Q&A | open + closed | 200 (open) / 451 (MedEvalKit) | ACC + verifier |
| PathVQA | pathology/histology microscopy | tissue/diagnosis, many yes/no | open + closed | 1500 (open) | ACC + verifier |
| PMC-VQA | PubMed Central article figures (mixed) | figure-based Qs | MCQ 4-opt | large | ACC |
| MMMU-Med | college medical exams, 5 subjects | multi-discipline reasoning, multi-image | MCQ | ~1.5k | ACC (reasoning; excluded from over-think) |
| MedXpertQA-MM | expert/exam multimodal | hardest medical | MCQ | ~2k | ACC (near-chance, excluded) |
| OmniMedVQA | many imaging modalities | large-scale medical VQA | MCQ | large | Lingshu-suite (to reproduce) |
| Kvasir-VQA | GI endoscopy | findings | open | 1200 | verifier (OOD modality) |
| RadImageNet-VQA | radiology | held-out | open | 2000 | verifier (transfer) |
| MS-CXR | chest X-ray | pathology phrase → bounding box | grounding | 435 | box-verifier |

EVALUATION: MCQ = exact-match on the option letter (no judge); open-text = LLM-judge semantic-equivalence vs gold
(labels from the answer key); verifier = AUROC / gap-captured / oracle@N / bootstrap CI; efficiency = batch-1 latency (s),
NVML energy (J), FLOPs 2N(P+G). Train/test splits disjoint; all numbers from real checkpoints.

NOTE (vLLM env, 2026-06-29): downgrading the system vLLM to 0.9.0.1 would BREAK the NGC env (force-downgrades the
custom NGC torch 2.9 → ABI/CUDA mismatch; breaks all working scripts). Single-image MedEvalKit tasks RUN in our vLLM 0.10
(PMC-VQA: no InputProcessingError); only MMMU (multi-image) hit the 0.10 mm-alignment bug ⇒ patch MMMU's multi-image
handling OR isolated venv; do NOT downgrade the base.

## 15. Lined-up experiment: guardrail-clean trained stability-router (2026-06-29, user-requested)
CHECKED + CONFIRMED: CASP-Stability (trained logistic on 7B "stability" signals; label = 1[pred7==pred32think])
BEAT the confidence/margin gate on ALL-6 at parity: FLOPs 49.0% vs 53.9% (margin)/57.4% (MSP), latency 1.77s vs
2.69s/2.96s, acc 0.5698 vs 0.5687 — and edged ACC-v2 agreement (52%/2.27s). KEY: re-targeting the routing label
from un-learnable recoverability (AUROC 0.58, the wall) to learnable STABILITY (AUROC 0.71) is what breaks the
training-free ceiling; capacity is irrelevant (logistic≈MLP≈LoRA). So "gate saturated" holds ONLY for training-free
gates predicting recoverability; a TRAINED stability-router beats confidence on cascade efficiency.
CAVEATS: (1) small per-benchmark guardrail dip (guard 0.05 vs 0.0); (2) iso-accuracy compute cut (cannot exceed the
32B); (3) ALL-5 / PMC-calib competent-4 edge shrinks or trades acc for compute. Code: src/training_methods/
{casp_stability.py, lora_stability_router.py}; data results/cascade_methods/casp_stability.txt.
EXPERIMENT TO RUN (after Lingshu reproduction): (a) make CASP-Stability per-benchmark guardrail-CLEAN (constrain
the threshold so no benchmark dips below 7B); (b) validate it on the Lingshu 7-task suite; (c) use it as the
MCQ-side gate inside the UNIFIED router (MCQ→ACC+stability-gate, open→verifier) → improve the routing DECISION,
not just the structure. Folded into report S5 (nuance) + next-step slide.
