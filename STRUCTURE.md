# Repository Structure & File Guide

> **What this is.** A map of the whole repo: every directory, and a one-line purpose for every script.
> Generated 2026-06-26. Companion to `CLAUDE.md` (which holds the project context + the safety rules).
>
> **How to read it.** Top-level map first, then one section per `src/` subfolder. The `src/<stage>/`
> layout groups code by pipeline stage; **always launch scripts from the repo root** (paths inside them are
> resolved relative to the launch directory — see CLAUDE.md §7).
>
> **Naming note.** Most folders already use descriptive `lowercase_with_underscores` names. The exception is
> `src/cascade_methods/` — the research-loop working directory, where names are terse (`compare.py`,
> `frontier.py`, `ceiling.py`). These are **deliberately not renamed**: each is referenced by the paper's
> reproducibility index and by the five `progress_*.md` historical logs, and several are shared modules
> imported by ~36 siblings — renaming would silently desync the paper trail and break imports. This guide is
> the descriptive index for them instead. Each script's own docstring (first line) carries the same summary.

---

## Top-level map

```
medvlthinker-imgdiff-compute/
├── CLAUDE.md            project context + safety rules (read first)
├── STRUCTURE.md         this file
├── README.md  RESULTS.md   project readme + running results log
├── progress_June_*.md   dated session logs (the paper trail; June 17 → 25-26)
├── SESSION_REPORT_trained_verifier.md   narrative report of the trained-verifier program
│
├── src/                 ALL active Python, grouped by pipeline stage (see sections below)
├── runners/             shell launchers (.sh) that drive the src/ scripts — NEW: decluttered from root
├── paper/               the CVGIP 2026 draft + figure scripts + figs/
│
├── ckpts/               per-sample JSONL checkpoints + trained adapters + gates  (gitignored, resumable)
├── results/             writeups + tables for the cascade-methods research  (gitignored)
├── logs/                nohup run logs  (gitignored)
├── data/                small inputs (subset.csv)  (gitignored)
├── feats/ feats_full/ feats_peer/   saved hidden-state features from killed probes  (gitignored)
├── archive/             killed research directions, kept as the negative-result record
│
├── MedRAG/  MedVLThinker/   external dependency git repos — DO NOT MOVE/RENAME
```

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
| `run_judge.py` | LLM-judge scorer for open-ended VQA (strong neutral grader) |
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
| `gate_alt_conformal.py` / `_6datasets.py` | CP-Router-style split-conformal gate (4-ds / all-6) |
| `gate_alt_learned_gbm.py` / `_6datasets.py` | learned gradient-boosted multi-feature router (4-ds / all-6) |

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
This is where the session's headline positive lives (the trained verifier, §5.10).
| file | purpose |
|---|---|
| `run_lora_verifier_open.py` | **trained verifier for open-ended best-of-N** (the §5.10 free-text positive) |
| `run_lora_box_verifier.py` | **structured-output box-verifier** (SLAKE + MS-CXR grounding positive) |
| `verifier_image_ablation.py` | does the verifier use the image? (refutes "lazy verifier") |
| `verifier_transfer_eval.py` | cross-dataset transfer of the trained verifier |
| `verifier_scaling_curve.py` | best-of-K test-time-scaling curve + bootstrap CI |
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
| `harness.py` | unified offline evaluation harness for two-model cascade gating (imported ~36×) |
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
| `make_full_record.py` | consolidate every data point into `results/cascade_methods/FULL_RECORD.md` |
| `make_master_charts.py` | every data point into comparable charts + tables |
| `make_open_chart.py` | charts for the open-ended self-consistency cascade |
| `make_rescue_charts.py` | charts for the visual-stability rescue into ACC-v2 |

---

## `runners/` — shell launchers (NEW: moved out of the root this session)
23 convenience `*.sh` scripts that drive the `src/` python (each `cd`s to the repo root first, so they can
live anywhere). Names indicate the job: `run_openvqa_*.sh` (open-ended generation), `run_judge_*.sh`
(LLM-judge scoring), `run_*_acc.sh` (per-family ACC runs: chiron/lingshu/medgemma/qoq), `run_2size_all.sh`,
`run_latency_all.sh` (batch-1 latency), `run_native_*.sh` (native-prompt think), `dl_verify_newmed.sh` /
`run_finalize_newmed.sh` (the new-method data pulls). Launch with e.g. `bash runners/run_judge_all.sh`.

## `paper/`
`cvgip2026_draft.md` (the CVGIP 2026 draft), `make_figs.py` + `make_verifier_fig.py` / `make_scaling_fig.py`
/ `make_pareto_fig.py` (figure generators), and `figs/` (generated PNGs, incl. `figs/limits/` for the
§5.9–§5.10 luck-floor / trained-verifier figures).

## Data & record directories (gitignored unless noted)
- `ckpts/` — per-sample JSONL checkpoints (resumable), trained LoRA adapters (`ckpts/train/`), open-ended
  generations + judge labels (`ckpts/openvqa/`), grounding outputs (`ckpts/ground/`), the deployed gate
  (`router_margin.pkl`). Do not relocate while a run could resume into it (CLAUDE.md §7).
- `results/cascade_methods/` — markdown writeups + master tables for the research loop.
- `archive/` — killed directions (image-difficulty, old gate scripts, single-model routing), kept as the
  negative-result record. Never deleted.
- `MedRAG/`, `MedVLThinker/` — external dependency git repos. **Never move or rename** (imports depend on them).
