# Repository Structure & File Guide

> **New here? Read `READING_GUIDE.md` first** — it's the step-by-step *reading order* (which section of which
> file, and why) to understand the whole research process. This file (STRUCTURE.md) is the *file index* it
> points you to.
>
> **What this is.** A map of the whole repo: every directory, and a one-line purpose for every script.
> Generated 2026-06-26; `results/` layout refreshed 2026-07-02; **file index and the `paper/` section
> refreshed 2026-07-29** (the July Lingshu scripts, the July-8 paper reorg, and the retrospective).
> Companion to `CLAUDE.md` (project context + safety rules) and
> `results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md` (**the definitive account of
> the project — the source of every number; this file is only the *where*, never the *what***).
>
> **How to read it.** Top-level map first, then one section per `src/` subfolder. The `src/<stage>/`
> layout groups code by pipeline stage; **always launch scripts from the repo root** (paths inside them are
> resolved relative to the launch directory — see CLAUDE.md §7).
>
> **Naming note.** Most folders already use descriptive `lowercase_with_underscores` names. The exception is
> `src/cascade_methods/` — the research-loop working directory, where names are terse (`compare.py`,
> `frontier.py`, `ceiling.py`). These are **deliberately not renamed**: each is referenced by the paper's
> reproducibility index and by the five `progress/progress_*.md` historical logs, and several are shared modules
> imported by ~36 siblings — renaming would silently desync the paper trail and break imports. This guide is
> the descriptive index for them instead. Each script's own docstring (first line) carries the same summary.

---

## Top-level map

```
medvlthinker-imgdiff-compute/
├── CLAUDE.md            project context + safety rules (read first)
├── STRUCTURE.md         this file
├── README.md  RESULTS.md   project readme + running results log
├── PROJECT_OVERVIEW.md  READING_GUIDE.md   plain-language overview + guided reading order
├── INCONSISTENCIES.md   dated (2026-06-27) numeric-consistency audit + canonical resolutions
│
├── progress/            13 dated daily progress logs (the paper trail; June 17 → July 8)
├── meetings/            meeting/presentation exports (dated .html decks) + report_template.html
├── src/                 ALL active Python, grouped by pipeline stage (see sections below)
├── runners/             38 shell launchers (.sh) that drive the src/ scripts (each cd's to repo root)
├── paper/               the CURRENT IEEE paper + build/figure scripts + figs_final/;  paper/archive/ = superseded drafts
├── docx/                generated Word exports (paper, overview, structure, reading guide)
│
├── ckpts/               per-sample JSONL checkpoints + trained adapters + gates  (gitignored, resumable)
├── results/cascade_methods/   docs/ (writeups) + artifacts/ (raw outputs) + claude_judge/  (gitignored)
├── logs/                nohup run logs  (gitignored)
├── data/                small inputs (subset.csv)  (gitignored)
├── feats/ feats_full/ feats_peer/   saved hidden-state features from killed probes  (gitignored)
├── archive/             killed research directions, kept as the negative-result record
│
├── MedRAG/  MedVLThinker/  MedEvalKit/   external dependency git repos — DO NOT MOVE/RENAME
```
`MedEvalKit/` is the faithful Lingshu-MCQ eval harness (run in the isolated `/data/dan/medeval_venv`).

Weights and datasets live OUTSIDE the repo under `/data/dan/...` (never committed). See CLAUDE.md §8.

---

## `src/labeling/` — run a model over data → per-sample JSONL checkpoints
The "labelers": each runs a VLM over a dataset and writes resumable per-sample JSONL.

| file | purpose |
|---|---|
| `run_7b_vllm.py` | 7B cheap-arm (no-think) vLLM runner, full eval set |
| `run_7b_think_vllm.py` | 7B in THINK mode (the paper's think baseline) |
| `run_7b_hf_labeler.py` | unified compute-router gate for the 7B (HF, OOM-guarded) |
| `run_7b_selfverify_vllm.py` | AutoMix/P(True)-style self-verification signal for the cheap 7B leg |
| `run_32b_vllm.py` | vLLM tensor-parallel (2-GPU) runner for the 32B |
| `run_32b_hf.py` | 32B (think) on the seed-42 500-question subset (HF, for VRAM) |
| `run_32b_modes_vllm.py` | strong-leg ablation: vary the 32B escalation target's mode |
| `run_openvqa.py` | generative (open-ended) medical-VQA runner — the open-ended cascade engine |
| `run_openvqa_verify.py` | open-ended self-verification (AutoMix/P(True)) for the gate hunt |
| `run_openvqa_verify_persample.py` | per-sample self/cross verification for verifier-guided selection |
| `run_openvqa_select_listwise.py` | listwise verifier-guided selection over an sc8 checkpoint |
| `run_openvqa_synth.py` | candidate-conditioned generation (strong model primed with cheap answers) |
| `run_openvqa_fewshot.py` | few-shot in-context exemplars to align answer style |
| `run_judge.py` | LLM-judge scorer for open-ended VQA (strong neutral grader). **Required** — MedEvalKit's open-half exact match is broken (gold `"CT"` vs `"CT."` scores wrong at ROUGE-1 ≈ 1.0) |
| `run_mcq_generate_verify.py` | MCQ generate-then-verify dump for the Unified Generative Verifier test (GPU) |
| `run_artifact_audit.py` | audit an open set for ANSWERABLE vs ARTIFACT questions |
| `run_ground_slake.py` | medical grounding on SLAKE (zero-download, uses detection.json boxes) |
| `run_ground_mscxr.py` | phrase grounding on the real MS-CXR benchmark (PhysioNet, 1448 boxes) |
| `run_vlm_eval.py` | general Qwen2.5-VL-family eval runner (any local/HF model) |
| `run_peer_eval.py` | model-agnostic vLLM eval for cross-family peer VLMs |
| `run_pmctrain_vllm.py` | run a model on the held-out PMC-VQA TRAIN sample |
| `run_judge.py` | (see above) |
| `embed_siglip.py` | frozen SigLIP image+text embeddings for the competent-4 samples |
| `nvml_power.py` | lightweight GPU power sampler for batch-1 energy measurement |

## `src/sweep/` — resolution / compute sweeps + calibration
| file | purpose |
|---|---|
| `run_7b_prune_sweep.py` | 7B no-think vision-token-budget (resolution) sweep |
| `tokens_per_cap.py` | exact 7B prompt-token count (text+vision) at each resolution cap |
| `grid_resolution_tau.py` | calibrated resolution × τ grid → the cap320 / τ=0.426 operating point |
| `cascade_resolution_sweep.py` | 2-D efficiency sweep over (resolution cap × escalation threshold) |
| `cascade_heldout_frontier.py` | honest held-out test-time eval of the resolution × escalation cascade |

## `src/gate/` — train + freeze the deployed margin gate
| file | purpose |
|---|---|
| `train_margin_gate.py` | train the frozen margin gate on the clean PMC-VQA train split → `router_margin.pkl` |
| `refit_gate_tau_per_cap.py` | refit the frozen gate's τ at each resolution cap |

## `src/cascade/` — the live co-resident cascade + real-time measurement
| file | purpose |
|---|---|
| `live_cascade.py` | live co-resident 7B→32B cascade, full eval, real-time single-query routing |
| `analyze_live_cascade.py` | analyze a live-cascade JSONL (safe mid-run) |
| `measure_single_leg.py` | batch-1 latency / power / VRAM for one model leg |
| `measure_config.py` | batch-1 latency/energy across multiple (arm × resolution-cap) configs |
| `report_cascade_from_legs.py` | reconstruct deployed cascade cost from the two measured legs |

## `src/analysis/cascade/` — analyses of the live cascade
| file | purpose |
|---|---|
| `cascade_per_benchmark_breakdown.py` | per-benchmark cap320 cascade behavior from validated labels |
| `cascade_cost_decode_flops.py` | decode-only compute number for the cascade |
| `cascade_cost_prefill_flops.py` | prefill-inclusive compute (review-proof) |
| `cascade_cost_accuracy_pareto.py` | cost-vs-accuracy operating curve |
| `cascade_gain_bootstrap_ci.py` | is the cascade accuracy gain real or small-sample noise? |
| `frozen_gate_transfer_bootstrap_ci.py` | paired-bootstrap CIs for the frozen gate's transfer |
| `cascade_escalation_signal_early.py` | the escalation-gate experiment (superseded) |
| `margin_gate_mechanism_diag.py` | mechanism proof: why no margin threshold beats parity |
| `gate_head_to_head.py` | CPU-only head-to-head of escalation gates (competent-4) |
| `recompute_energy.py` | CORRECTED per-dataset time/energy saved |
| `recompute_energy_superseded.py` | earlier energy proxy (kept for record) |

## `src/analysis/ablations/` — gate alternatives that LOST to the margin gate
| file | purpose |
|---|---|
| `gate_ablation_bakeoff.py` | offline router bake-off (margin vs FBE/conformal/HistGBM), no inference |
| `gate_alt_conformal.py` / `gate_alt_conformal_6datasets.py` | CP-Router-style split-conformal gate (4-ds / all-6) — over-escalates 69–80% |
| `gate_alt_learned_gbm.py` / `gate_alt_learned_gbm_6datasets.py` | learned gradient-boosted multi-feature router (4-ds / all-6) — top detection AUROC, **lowest** cascade quality |

## `src/reporting/` — build the paper's tables + harness validation
| file | purpose |
|---|---|
| `build_table1_accuracy.py` | Table 1 (accuracy, four competent benchmarks) |
| `build_table2_efficiency.py` | Table 2 (efficiency, four competent benchmarks) |
| `report_efficiency_per_dataset.py` | per-dataset time/energy saved vs always-32B |
| `report_cascade_per_dataset.py` | per-dataset breakdown of the finished cascade run |
| `report_medxpert_dilution.py` | effect of including near-chance MedXpert on the headline |
| `merge_7b_think_and_validate.py` | merge the 7B-think shards + validate the harness |
| `inspect_timing_fields.py` | inspect timing/energy fields in the checkpoints |

## `src/data_prep/`
| file | purpose |
|---|---|
| `build_eval_subset.py` | build `subset.csv`, a small eval slice |
| `sample_pmcvqa_train_heldout.py` | fixed-seed PMC-VQA train sample for held-out threshold fitting |
| `prep_pmcvqa_train_sample.py` | sample a clean training subset from PMC-VQA train |
| `prep_kvasir.py` | prep the open-ended Kvasir-VQA-x1 subset (GI endoscopy) |
| `prep_radimagenet.py` | prep the RadImageNet-VQA open-ended slice (5th dataset) |

## `src/training_methods/` — the TRAINED methods (need `peft`)
This is where the project's **only** break of the luck floor lives: the trained outcome verifier. Its
deployed adapter is `ckpts/train/lora_verifier_pooled4` (per-answer AUROC 0.924).
| file | purpose |
|---|---|
| `run_lora_verifier_open.py` | **trained verifier for open-ended best-of-N** (the free-text positive: 49% of the oracle gap) |
| `run_lora_box_verifier.py` | **structured-output box-verifier** (SLAKE organs 40% / real MS-CXR chest X-ray 78% of the gap) |
| `run_lora_verifier_ranking.py` | ranking/contrastive (Bradley–Terry) verifier — lifts per-answer AUROC 0.90→0.93 but selection is **FLAT** |
| `verifier_image_ablation.py` | does the verifier use the image? (refutes "lazy verifier"). ⚠️ Retrospective §8.2 item 10: **its result is nowhere on disk** — a ~30-min rerun that could invalidate a load-bearing conclusion |
| `verifier_transfer_eval.py` | cross-dataset transfer of the trained verifier (zero-shot, +0.024) |
| `cross_gen_verifier.py` | cross-**generator** transfer: score a different model's answers (generator-agnosticism) |
| `verifier_scaling_curve.py` | best-of-K test-time-scaling curve + bootstrap CI (0.385 → 0.501 over K=1→8) |
| `clean_verifier_dump.py`, `reconstruct_clean_dump.py` | rebuild/clean the per-question verifier score dumps feeding the scaling curve |
| `calm_fuse.py` | CALM-Fuse: trained answer-fusion of the two no-think tiers |
| `casp_stability.py` | trained gate whose target is cascade-cost optimality |
| `fld_distill.py` | FastLeg-Distill: LoRA-distill the big no-think model into the small one |
| `lora_stability_router.py` | LoRA-trained gate predicting the 7B's own resolution-stability |

## `src/legacy_retrieval/`
| file | purpose |
|---|---|
| `retrieve.py` | MedRAG retrieval — leftover from the killed RAG direction, kept for record |

---

## `src/cascade_methods/` — the research-loop working dir (terse names; full index here)

The fast-moving scratch space where the cascade-method search ran. Grouped by theme for navigation.

**Shared modules (imported by many siblings — DO NOT rename/move):**
| file | purpose |
|---|---|
| `harness.py` | unified offline evaluation harness for two-model cascade gating — imported by **39** files (36 here + 3 in `training_methods/`). Superseded as a source of *headline* numbers (the July Lingshu chain imports it zero times) but still the reproducibility anchor for `docs/archive_mcq/` |
| `methods.py` | registry of training-free cascade gating methods (each a scoring function) |
| `methods_deferral.py` | recoverability-aware deferral gates (the novel deferral direction) |
| `frontier.py` | accuracy-vs-compute frontiers for escalation scores (imported by comparisons) |
| `acc_compare.py` | head-to-head of the three cascade methods on 5 metrics (imported by ACC runners) |

**ACC — the Adaptive-Compute Cascade (the paper's structural method):**
| file | purpose |
|---|---|
| `acc.py` | the core ACC: confidence-gated 3-tier compute-configuration cascade |
| `acc_v2.py` | ACC-A: the recommended variant, cross-model-agreement think gate |
| `acc_v3_confgate.py` | confidence-tightened think-gate, validated across families |
| `acc_v4_lowres_think.py` | ACC-v4 = v3 + resolution-decoupled think tier |
| `acc_2size.py` | ACC generalization to other 2-size families (Lingshu, MedGemma) |
| `acc_allmethods.py` | the full 3-tier bake-off run on all families |
| `acc_rescue_allfam.py` | integrate the visual-stability rescue into ACC-v2 |
| `multitier.py` | mode-adaptive multi-tier cascade (no-think ≥ think on perception) |
| `overthink_generalize.py` | does ACC's "no-think ≥ think on perception VQA" premise generalize? |
| `final_3tier.py` | honest all-6 evaluation of the 3-tier cascade |
| `final_3tier_comparison.py` | complete method comparison under the fixed 3-tier ACC config |

**Gate / method leaderboards & comparisons:**
| file | purpose |
|---|---|
| `compare.py` | unified fair leaderboard for cascade gating methods |
| `evaluate.py` | the definitive cascade leaderboard (every method) |
| `final_comparison.py` | definitive leaderboard: SOTA training-free gates × configs |
| `escalation_leaderboard.py` | primary metric = min 32B escalation rate at iso-accuracy |
| `gate_compare.py` | hold the ACC 3-tier config fixed, vary only the gate |
| `sota_comparison.py` | head-to-head vs current-SOTA training-free gates |
| `baseline_compare.py` | corrected (post-audit) training-free gate bake-off |
| `deferral_curve.py` | canonical routing figure: accuracy vs escalation |
| `frontier_compare.py` | escalation-accuracy frontier (the VADR direction) |
| `diagnostics.py` | why confidence-only gating wastes compute |
| `uniform_improver_diag.py` | why confidence is near-optimal (Jitkrittum et al. NeurIPS'23) |
| `gate_data_size.py` | does more PMC-VQA-train data change the gate result? |
| `compare_native_think.py` | native-prompt vs foreign-prompt think vs no-think, per family |

**Visual-stability "rescue" family (a training-free robustness signal):**
| file | purpose |
|---|---|
| `resolution_consistency.py` | the novel signal: VISUAL STABILITY across resolutions |
| `bidirectional_stability.py` | test the bidirectional visual-stability gate premise |
| `control_think_signals.py` | is resolution-stability the right think-tier signal? |
| `combined_rescue.py` | do multiple orthogonal robustness signals compound? |
| `resolution_tta_and_tier.py` | two more training-free ideas from multi-resolution data |
| `stability_rescue_validate.py` | honest validation of the visual-stability rescue gate |
| `stability_rescue_bootstrap.py` | paired bootstrap CIs for the rescue headline |
| `stability_rescue_cost.py` | prefill-inclusive FLOPs + accuracy + guardrail for the rescue |
| `final_robustness_rescue.py` | canonical validation of the robustness-rescue family |
| `think_rescue_mechanism.py` | why rescue@think works (mechanism) |

**Cross-family / peer complementarity:**
| file | purpose |
|---|---|
| `crossfamily_agree.py` | cross-family agreement as a correctness signal (decorrelated errors) |
| `peer_premise.py` | premise test for cross-family complementarity |
| `peer_router.py` | does a cheap router capture cross-family complementarity? |
| `peer_router_img.py` | decisive test: router on frozen peer signals |

**Open-ended selection / verifier feasibility (Phase A of this session):**
| file | purpose |
|---|---|
| `open_cascade_analyze.py` | open-ended cascade: is semantic self-consistency a better gate? |
| `open_ablations.py` | ablations for the open-ended ceiling-break (§5.7) |
| `gate_search_open.py` | hunt for an open-ended routing signal that beats confidence |
| `select_eval.py` | evaluate verifier-guided SELECTION over sc8, judged semantically |
| `explode_sc_for_judge.py` | explode an sc8 checkpoint into one judge-input file (idx `origidx#k`) |
| `strong_fixes_genuinely_unknown.py` | does the 32B fix the 7B's genuinely-unknown (oracle-wrong) errors? |
| `knowledge_feasibility_bytype.py` | RAG feasibility: decompose open errors by content type |
| `ground_analyze.py` | grounding: does spatial self-consistency predict box IoU? |

**Abstention / deferral (deprioritized direction — §5.8):**
| file | purpose |
|---|---|
| `selective_abstain.py` | training-free selective prediction / safe abstention |
| `abstain_calibration.py` | is the abstention threshold deployable at a target risk r*? |
| `metarouter.py` | unsolvable-aware deferral-aware cascade router |
| `metarouter_honest.py` | deployable-honest evaluation of the meta-router |
| `triage_3way.py` | unified 3-way triage for open-ended medical VLMs |

**Other diagnostics / one-offs:**
| file | purpose |
|---|---|
| `ceiling.py` | in-distribution ceiling of recoverability-aware gating (K-fold CV) |
| `cheap_strong.py` | re-score the cascade with a cheaper strong leg (32B no-think) |
| `strong_leg.py` | compare cheaper escalation targets vs the 32B-think leg |
| `latency_estimate.py` | estimated batch-1 latency reduction of VADR vs SOTA confidence |
| `sd_test.py` | feasibility + speedup of lossless speculative decoding for 32B-think |
| `vision_sensitivity.py` | is the blank-image counterfactual a usable gate signal? |
| `lingshu_prompt_probe.py` | does Lingshu truly answer directly, or is it our prompt? |

**Chart / table builders:**
| file | purpose |
|---|---|
| `make_detailed_table.py` | detailed readable number tables across families |
| `make_full_record.py` | consolidate every data point into `results/cascade_methods/docs/archive_mcq/FULL_RECORD.md` |
| `make_master_charts.py` | every data point into comparable charts + tables |
| `make_open_chart.py` | charts for the open-ended self-consistency cascade |
| `make_rescue_charts.py` | charts for the visual-stability rescue into ACC-v2 |

**2026-07 final-method suite (the reframe → unified method → beat-32B / escalation / slice-structure passes):**
The terse-named scripts from the 2026-07-06/07 loop that assemble and probe the deployable method. Full write-ups:
`docs/current/METHOD_FINAL_2026-07.md` + `docs/current/RESEARCH_RESULTS_2026-07.md` §7; progress `progress/progress_July_07.md`.
| file | purpose |
|---|---|
| `method_final.py` | **THE** single unified reproducible paper method; writes `method_final.json` (v1, F3-fusion) + `method_final_v2.json` (v2, F8+F10 folded → **both** Pareto modes FLOP-negative) |
| `integrated_method.py` | the best integrated format-aware cascade scored vs always-32B-**THINK** (→ `integrated_method_vs_think.json`) |
| `integrated_pandora.py` | the integrated cascade with the open-text arm's fixed Pandora adaptive-N draw count (→ `integrated_pandora_opentext.json`) |
| `best_method_lingshu.py` | FALC — the format-aware 2-tier cascade, faster-and-matches always-32B-nt on faithful Lingshu (→ `best_method_lingshu_medeval.json`) |
| `pandora_controller.py` | Pandora's-Box (Weitzman) adaptive candidate-draw + escalation controller (→ `pandora_controller.json`) |
| `beat32b_fusion.py` | backlog §F "[BEAT-32B]" pass-3: slice-gated decision fusion/routing; the F3 conf-advantage PMC beat (→ `beat32b_fusion.json`) |
| `beat32b_more.py` | §F pass-4: F8 certified veto / F7 super-learner / F11 BMA / F10 open-text L2D (→ `beat32b_more.json`) |
| `robust_slice_routing.py` | §H pass-4: H4 error-slice discovery / H8 Bühlmann credibility shrinkage / H2 kNN gate — all negative, 6th wall confirmation (→ `robust_slice_routing.json`) |
| `escalation_levers.py` | §G speed levers G5/G6/**G8** (parallel prefill prefetch) on the integrated cascade (→ `escalation_levers.json`) |
| `escalation_more.py` | §G pass-4 speed levers G7 semantic cache / G2 early-exit / G4 32B image-token prune (→ `escalation_more.json`) |
| `quantized_strong_leg.py` | G3: re-cost the cascade with an AWQ/GPTQ-INT4 strong leg (a VRAM/energy win, **not** a FLOPs win) (→ `quantized_strong_leg.json`) |
| `bench_int4_strong_leg.py` | committed vLLM benchmark for real batch-1 INT4-vs-FP16 32B latency (blocked this session by an HF-CDN outage) |
| `logit_fusion.py` | full-posterior (per-option logprob) MCQ fusion of Lingshu-7B+32B — negative, does not extend the beat past F3 (→ `logit_fusion.json`) |
| `diversity_generate_gpu.py` | GPU diverse-generation candidate dump for idea A1 (prompt-persona × temperature portfolio) |
| `pairwise_verifier_score.py` | GPU pass: real A-vs-B pairwise verifier verdicts (both orders, position-debiased) |
| `pairwise_verifier_diverse.py` | GPU pass: pairwise verdicts over the diverse-generation candidate sets (the compounding test) |

**★ The 2026-07-08/09 headline chain (the scripts that produce the paper's current numbers):**
These four were missing from this index until 2026-07-29, including the one that produces the headline CI.

| file | purpose |
|---|---|
| `paper_baselines.py` | **the paper's CORE results table** — our method vs every single-model strategy, per-benchmark + pooled, on accuracy / batch-1 latency / FLOP-eq / energy, with paired-bootstrap 95% CIs on every delta (→ `paper_baselines.json`). Its `build_cells()` is the shared cell builder (~42 s, CPU only, 9 cells = 42,374 rows). |
| `method_final_mmmu_corrected.py` | the core table RECOMPUTED with the MMMU cell corrected for train-set contamination — **Variant A (escalated) vs Variant B (excluded, n=42,224)** × 3 modes, plus the oracle-mode-32B baseline and the Pareto frontier (→ `method_final_mmmu_corrected.json`) |
| `opentext_32b_think_full.py` | replaces the **estimated** open-text always-32B-THINK accuracy with a fully **measured** one (SLAKE-open 0.6791 / VQA-RAD-open 0.5450 / PathVQA-open 0.1087) and recomputes the comparison end-to-end (→ `opentext_32b_think_full.json`) |
| **`f8_mode_vsthink_ci.py`** | **produces the canonical headline CI** — attaches a paired-bootstrap 95% CI to the FLOP-negative accuracy-max mode's vs-32B-THINK delta using the measured per-sample vectors: **+0.0245 [+0.0216, +0.0274], n=42,224** (→ `f8_mode_vsthink_ci.json`) |

**Open-text gate / verifier / cascade suite (the June-29→July-2 open-text arm):**
| file | purpose |
|---|---|
| `open_gate_bakeoff.py` | does a TRAINED gate beat the verifier-confidence gate? (no) |
| `open_gate_swap.py` | controlled gate swap: verifier + best-of-8 selection held fixed, only the gate varies |
| `open_gate_efficiency.py` | the efficiency leg: at iso-accuracy, which gate escalates least (measured latency/energy)? |
| `open_gate_heldout_tau.py` | the honest deployable value of the gate — 5-fold cross-fit τ vs oracle τ vs no gate |
| `open_recoverability_gate.py` | THE decisive test (Jitkrittum NeurIPS'23): can any trained gate predict *recoverability*? |
| `open_vstab_gate.py` | 3-family faithful CASP/CCPS test: does visual stability add signal beyond the verifier? |
| `open_bestofN_adaptive.py` | compute/accuracy trade-off of longer answers + adaptive best-of-N |
| `open_cost_frontier.py` | cost-optimized verifier-cascade frontier, per-dataset + pooled |
| `open_verifier_cascade_table.py` | the verifier-augmented cascade table for one family, from saved artifacts |
| `open_measure_latency_energy.py` | batch-1 latency + NVML energy for the open-text tiers, on real images |
| `gen_slake_open_bestofN.py` | fills the one missing open-text best-of-N verifier dump (SLAKE-open) |
| `gate_unified_bakeoff.py` | clean gate bake-off in BOTH regimes + the new EG-RC gate (wins one regime, loses another) |

**Faithful-MedEvalKit + unified-router suite (2026-07-01→05):**
| file | purpose |
|---|---|
| `lingshu_medeval_cascade.py` | clean 2-tier cascade + efficiency on faithful MedEvalKit Lingshu outputs |
| `cascade_all_families.py` | the faithful 2-tier cascade across all 6 benchmarks, any family |
| `unified_router.py` | Method C: the deterministic **format-aware router** over a pooled MCQ+open workload |
| `measure_batch1_latency_unified.py` | replaces the amortized-batch latency proxy with real batch-1 wall-clock (GPU) |
| `ugv_mcq_verdict.py` | the verdict for the Unified Generative Verifier on the MCQ half (**negative**) |
| `latency_reexamination.py` | re-examines the verifier best-of-N cascade on batch-1 latency rather than FLOPs |
| `end_to_end_consolidation.py` | consolidates the validated open-text levers into one pipeline on the cost frontier |

**Backlog pass-3/4 experiments (2026-07-06/08) — mostly documented negatives:**
| file | purpose |
|---|---|
| `diversity_candidates.py` | idea A1 offline: diversity-maximized candidate sets (DPP/MMR) |
| `diverse_measure_gpu.py` | the matched-design measure for diverse generation vs iid@8 |
| `generator_portfolio.py` | idea A2: generator portfolio via error correlation (Markowitz) — ≈ uniform |
| `pandora_pooling_combo.py` | stacks the two validated levers: Pandora controller × cross-model pooling |
| `pandora_correlated.py` | a correlation-aware variant of the Pandora controller |
| `bandit_allocation.py` | idea C7: pure-exploration bandit budget allocation (Δ +0.002, negative) |
| `active_comparison_verifier.py` | idea C9: information-directed active-comparison verifier (TrueSkill/IDS) |
| `combine_diverse_pairwise.py` | the **compounding** test: diverse × pairwise do not stack (−0.0117) |
| `distractor_filter.py` | 8 distractor pre-filter rules on the diverse pool — none significant |
| `dawid_skene_aggregate.py` | idea B5: unsupervised Dawid–Skene truth inference (≈ majority, −0.013) |
| `verifier_32b_gpu.py` / `verifier_32b_measure.py` | does a 32B zero-shot verifier break the selection wall? (**ties** the trained 7B) |
| `ttt_cheap_leg.py` | idea H1: test-time training / TENT / SHOT on the cheap leg — **collapses it (−0.159)** |
| `neurosymbolic_gate.py` | idea H9: neuro-symbolic constraint filter — fires on ~1 sample, **negative** |

> ⛔ **Abstention is permanently forbidden** across this suite (CLAUDE.md critical rule 6, effective
> 2026-07-07). Every mechanism above answers — the F8 "certified veto" keeps the 7B's answer, it does not
> defer to a human. Backlog idea H3 (abstain-to-human) was excised. **Six files from the June abstention
> work still physically sit in this directory** — `selective_abstain.py`, `abstain_calibration.py`,
> `triage_3way.py`, `deferral_curve.py`, `methods_deferral.py`, `lingshu_deferral_apgr.py` — plus four
> artifacts and `paper/figs/open/fig_triage.png`. **Historical record only; never a live direction.**

> **Code-health note (retrospective §9.6):** there are **two disjoint code universes** here — the June
> MCQ era rooted at `harness.py` (imported by 39 files) and the July Lingshu chain (~10 files) which
> imports `harness.py` **zero** times. A mechanical scan found ~66 scripts in this directory that neither
> write a surviving artifact nor are imported by anything. Known duplication: the cost constants exist in
> **3 copies** (`integrated_method.py`, `pandora_controller.py`, `paper_baselines.py` — values agree),
> `auroc` is defined **12 times**, and bootstrap resample counts differ (2,000 in one file, 10,000
> elsewhere). Consolidate before extending, not while extending.

---

## `runners/` — shell launchers (moved out of the repo root 2026-06-26)
**38** convenience `*.sh` scripts that drive the `src/` python (each `cd`s to the repo root first, so they
can live anywhere). Names indicate the job: `run_openvqa_*.sh` (open-ended generation), `run_judge_*.sh`
(LLM-judge scoring), `run_*_acc.sh` (per-family ACC runs: chiron/lingshu/medgemma/qoq), `run_2size_all.sh`,
`run_latency_all.sh` (batch-1 latency), `run_native_*.sh` (native-prompt think), `run_*_medeval*.sh` (the
faithful MedEvalKit runs), `dl_verify_newmed.sh` / `run_finalize_newmed.sh` (the new-method data pulls).
Launch with e.g. `bash runners/run_judge_all.sh`.

⚠️ **Two runner landmines.** (1) Weight paths are hard-coded **HuggingFace snapshot hashes** (e.g.
`models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd4…`) — a cache refresh changes the hash and
breaks them. (2) `run_full_matrix_medeval.sh` treats *any* JSON under the output path as "done", with no
row-count validation, so a **crashed partial run is skipped forever**. 12 runner invocations use a third,
previously-undocumented interpreter: **`/data/dan/medeval_venv/bin/python`** (vLLM 0.9.0.1).

## `paper/`

> **Reorganized 2026-07-08 10:06** — the naming convention is now `<topic-slug>_<venue>_<YYYY-MM-DD>.{tex,pdf}`,
> newest date = current, everything superseded moves to `archive/`. Full convention: **`paper/README.md`**.

| file | purpose |
|---|---|
| `adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}` | **THE CURRENT PAPER** — 9 pages, IEEEtran. Rebuilt 2026-07-09 with the measured / Variant-B / CI'd numbers. With the July-27 deck, the only prose artifact carrying the corrected headline. |
| `build_ieee.sh` | IEEE build wrapper (tectonic — there is no system LaTeX on this VM) |
| `make_ieee_figs.py` | generates the current paper's figures → `figs_final/` |
| `IEEEtran.cls` | IEEE document class |
| `figs_final/` | the current paper's figures (`fig_schematic.pdf`, `fig_overthink.pdf`, `fig_pareto.pdf`) |
| `figs/` | figures of the **archived** Markdown drafts (`fig1_latency_accuracy_frontier.png`, `fig2_overthinking_perbench.png`, + `limits/ master/ open/ rescue/`) |
| `build_report_v2.py`, `build_professor_html.py` | HTML progress-report builders (not the paper) |
| `build_professor_html_2026-07-27.py` | builds `meetings/progress_report_professor_2026-07-27.html` — the newest file in the repo (79 KB). ⚠️ It makes **zero** JSON reads: all 116 four-decimal figures are hand-typed literals, so nothing enforces its "read from real artifacts" footer. Verify against the artifacts before reusing its numbers. |
| `archive/` | superseded drafts, kept as the record: `manuscript_final_2026-07.{md,pdf}`, `manuscript_2026-07_longform.{md,pdf}`, `conference_2026-07.{md,pdf}`, `cvgip2026_draft.md`, `cvgip2026_ieee.{tex,pdf}`, `hello_ieee.*`, and `archive/scripts/` (the old figure/render/util scripts: `make_figs.py`, `make_verifier_fig.py`, `make_scaling_fig.py`, `make_pareto_fig.py`, `make_limits_fig.py`, `make_verifier_discrim_fig.py`, `md2docx.py`, `analyze_clean_dump.py`, `_report_slides.py`) |

⚠️ **Docs written before 2026-07-08 point at `paper/cvgip2026_draft.md` or `paper/manuscript_final_2026-07.md`
as "the paper" and cite its §5.x numbering.** Both are in `archive/`; the IEEE paper uses different sectioning
(Abstract, §I Intro with contributions C1–C4, §III Findings, §IV Method, §V Main result, §VI Limits) and no codenames.

## `meetings/`
| file | purpose |
|---|---|
| `progress_report_professor_2026-07-27.html` | **the most current and most rigorous single summary of the project** — 13 source-cited sections + glossary (458 KB). Built by `paper/build_professor_html_2026-07-27.py`. ⚠️ Its slide-6 baseline table and its hero claim label **two different variants** "accuracy-max" (5.70 compute-units = 1.25× fusion, vs 0.93× veto). |
| `progress_report_professor_2026-06-29.html` | the earlier deck (821 KB) — superseded |
| `report_template.html` | the deck template |

## Data & record directories (gitignored unless noted)
- `ckpts/` — per-sample JSONL checkpoints (resumable). Do not relocate while a run could resume into it
  (CLAUDE.md §7). Contents by era:
  - *June / MedVLThinker:* `gate_7b_*`, `gate_32b*`, `pmctrain/`, `_legacy/`, `router_margin.pkl`
    (the frozen τ=0.426 gate), `router_{learned,conformal}*.pkl` (the losing ablation gates),
    `token_cache.json`, `rt_cascade_cap320.jsonl` (**the one genuine live cascade run in the repo**).
  - *July / Lingshu:* `train/` (LoRA adapters incl. **`lora_verifier_pooled4`**, the deployed verifier),
    `openvqa/` (open-text generations + judge labels, incl. `strong_lingshu` and `strong_lingshu_think`),
    `ground/` (SLAKE-organ + MS-CXR box outputs), `pairwise/` + `pairwise_diverse/` (real A-vs-B verdicts),
    `acc_gen/` (5-family + peer-architecture mode dumps), `peer/`, `mcq_gen_verify/`,
    `gate_lingshu7b_mcq/` + `gate_lingshu32b_mcq/`.
- `MedEvalKit/eval_results_*/` — **the faithful evaluation dumps every current paper number is read from.**
  Gitignored *vendor* territory. Do not clean. ⚠️ Two dumps have unidentified provenance
  (`eval_results_reason` — documented as a **different model**, not Lingshu-32B — and the undated
  `eval_results` / `eval_results_32b`); they sit next to the canonical directories with no README.
- `results/cascade_methods/` — the research writeups + outputs (reorganized 2026-07-02; see its `README.md` index):
  - `docs/current/` — canonical writeups + specs. **`PROJECT_RETROSPECTIVE_2026-07-29.md` is the entry
    document**; **`PMCVQA_PROVENANCE_2026-07-30.md`** — how PMC-VQA is built, how thinly it is validated,
    which splits exist and which of the project's two evaluation tracks used which (**read before quoting
    any PMC-VQA number**); then `METHOD_FINAL_2026-07`, `RESEARCH_RESULTS_2026-07`, `TECHNICAL_REPORT_2026-07`,
    `OMNIMED_FALLBACK`, `VERIFIED_FACTS`, `OPENTEXT_*`, and the historical `METHOD_ACC` / `METHOD_MATH` /
    `METHOD` / `METHOD_deferral_router` / `METHODS_MASTER` / `MASTER_SUMMARY_2026-07`.
  - `docs/archive_mcq/` — superseded MCQ-era research-loop writeups (kept as the negative-result record).
  - `artifacts/` — ~107 raw `.json`/`.txt`/`.jsonl`/`.csv` outputs (gitignored, regeneratable). Headline
    chain: `f8_mode_vsthink_ci.json`, `method_final{,_v2,_mmmu_corrected}.json`, `paper_baselines.json`,
    `opentext_32b_think_full.json`.
  - `claude_judge/` — Claude-as-judge verdicts for open-text grading (committed).
  - `METHOD_IDEAS_BACKLOG.md` — 68 cross-field ideas, §A–H. ⚠️ **No per-idea status field**: several
    entries (H1 test-time training, H9 neuro-symbolic, A1/A2/B5/C1/C7/C9) were executed in July and are
    documented negatives, but still read as open proposals. Cross-check retrospective §6 first.
- `docx/` — generated Word exports of the paper + docs (not committed).
- `archive/` — killed directions (image-difficulty, old gate scripts, single-model routing), kept as the
  negative-result record. Never deleted.
- `MedRAG/`, `MedVLThinker/`, `MedEvalKit/` — external dependency git repos. **Never move or rename** (imports depend on them).

---

## ⚠️ Preservation status (2026-07-29)

The last git commit is **`8cdefef`, 2026-07-02**. Everything from the Lingshu era exists **only in the
working tree**: 44 untracked `.py` files under `src/` (including the entire live headline chain), all nine
July diaries, `paper/adaptive-cascade-medvqa_ieee_2026-07-08.*`, `paper/README.md`, `paper/figs_final/`,
and `meetings/progress_report_professor_2026-07-27.html`. `results/` and `MedEvalKit/` are gitignored, so
all 107 numeric artifacts and the primary evaluation dumps are untracked too. **The paper's method, its
inputs and its outputs currently exist on one disk.** Committing is the standing top-priority chore
(retrospective §8.1 item 2) — it blocks nothing and risks everything.
