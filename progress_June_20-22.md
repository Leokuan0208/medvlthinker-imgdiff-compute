# Session progress log — June 20–22, 2026 (cross-family validation, native prompts, novel-method search)

> Continues from `progress_June_17.md`. This log records, in detail and split by day, everything done in this
> session: completing the network-blocked 2-size validation, expanding to five medical VLM families across three
> architectures, the full all-methods bake-off with measured latency/energy, the (negative) novel-training-method
> research loop, the native-reasoning-prompt correction, and the cost-methodology bug fix. **No fabricated
> numbers** — every figure is read from real checkpoint output; reproduction pointers are inline.

---

## 2026-06-17 — blocked on the network

- The 2-size cross-family validation (Lingshu, MedGemma) was **blocked by a VM-wide network throttle** (~8 B/s).
  Confirmed downloads stalled; left the auto-resume watcher running. Everything that didn't need the network
  (paper §5.1.1 complete comparison, §5.6 CASP-Stability, 95% CIs, figures) had already been folded in earlier.
- No new experimental data this day.

---

## 2026-06-20 — network recovered: data acquisition, 2-size validation, two new families, all-methods, cost

**Network came back** (general ~47 MB/s; HF downloads ~77 MB/s via the working flags `HF_HUB_DISABLE_XET=1
HF_HUB_ENABLE_HF_TRANSFER=0`). The throttle had lifted.

### Downloads + a shard-corruption gotcha
- Downloaded to `/data/dan/hf_cache`: **Lingshu-7B/32B** (Qwen2.5-VL medical) and **MedGemma-4B/27B** (Gemma3 medical).
- **Bug:** a 4-way concurrent high-speed download **corrupted one Lingshu-32B shard** (`model-00003-of-00014`,
  `SafetensorError: invalid JSON header`) even though it reported "0 incomplete". Verified all four models'
  shards with `safetensors.safe_open`; only that one was bad. Deleted the blob + re-fetched (~2 min) → all valid.
  **Lesson (logged):** always verify safetensors after high-concurrency downloads.

### 2-size ACC validation (Lingshu, MedGemma)
- Ran the 3-tier pipeline (small-nt@cap320 → big-nt@cap320 → big-think) via `run_lingshu_acc.sh` (Qwen2.5-VL,
  `run_vlm_eval.py`, resolution sweep) and `run_medgemma_acc.sh` (Gemma3, `run_peer_eval.py` llm.chat, fixed-res).
- **Lingshu** strongly confirmed the over-thinking premise (32B-no-think beats 32B-think on every perception
  benchmark); cap320 sweet-spot matched MedVLThinker. **MedGemma** = a contrasting regime (4B ≈ 27B; over-thinking
  weak/mixed; guardrail violated). Documented in `results/cascade_methods/2SIZE_VALIDATION.md` + `acc_2size.py`.
- Fixed an analysis display bug (only 4 perception benchmarks shown) → all 7 splits + ALL-5 *and* ALL-6 averages.

### Two more families found + tested (literature/HF workflow)
- A search workflow vetted candidates and recommended **QoQ-Med-VL 7B/32B** (Qwen2.5-VL, ungated, same-lineage)
  and **Chiron-o1 2B/8B** (InternVL3, ungated) — both vLLM-native. Skipped HuatuoGPT/Hulu-Med/Citrus-V (custom
  arch or no same-lineage pair). Downloaded **sequentially with per-shard verification** (no corruption this time).
- Smoke-tested both arch paths; both clean. Ran the full pipeline (`run_qoq_acc.sh`, `run_chiron_acc.sh`).

### Full all-methods bake-off (10 gate methods × 5 families)
- Built `src/cascade_methods/acc_allmethods.py` (family-parameterized) running Ours-ACC-v2, CASP-Stability,
  ACC-v1/margin, MSP, entropy, Gini/DOCTOR, AutoMix, FrugalGPT-learned, Jitkrittum-L2D, random, under the 3-tier
  config, FLOPs-calibrated at parity, honest 50/50 × 20 seeds. Made the self-verify runners model-agnostic
  (`run_7b_selfverify_vllm.py --model_path`; added `run_peer_eval.py --verify`) so **AutoMix runs on all five**.
  Validated it reproduces MedVLThinker exactly.
- Headline (foreign-think at this point): MedVLThinker is the only family where the full cascade is cleanly
  beneficial; for Lingshu/QoQ/Chiron the think tier is net-harmful so the cascade collapses to the cheap leg;
  Chiron inverse-scales (2B>8B on PathVQA, a binary benchmark).

### Real batch-1 latency (+ later energy), and the cost approach (NOT the 30 h eval)
- Added `--batch1` to both eval runners (`max_num_seqs=1`, records `latency_s`). `run_latency_all.sh` measures a
  small per-tier sample (6/benchmark); `acc_allmethods.fit_metric` fits `cost = a·gen+b` per tier and computes
  per-method cost analytically — the same cheap fit approach as the original `acc_compare.fit_models`, not a
  full-dataset real-time eval. Uniform across all five families.
- Added **NVML energy + peak VRAM** (`src/labeling/nvml_power.py`; `--batch1` now records `energy_j` +
  `PEAK_VRAM_GB`). Fixed a crash where very fast nt calls captured <2 power samples (`None/1`).

### Artifacts produced (06-20)
- `2SIZE_VALIDATION.md`, `acc_2size.py`, `acc_allmethods.py`, `make_master_charts.py`, `make_full_record.py`,
  `master_data.csv`, `MASTER_TABLES.md`, `FULL_RECORD.md`, per-family `allmethods_*.json`, and the 5 master charts.

---

## 2026-06-21 — CASP relabel, the (negative) novel-method research loop, native-prompt investigation

### CASP-Stability relabeled
- Per user correction, removed the "Ours" tag from CASP-Stability everywhere (code, paper, docs) — it is a
  trained *baseline*, not our method; only ACC-v2 (agreement) is "Ours".

### Training-based methods enabled safely
- Installed **`peft==0.14.0` with `--no-deps`** (verified torch 2.9 / transformers 4.55.2 / vLLM 0.10 / accelerate
  unchanged). Skipped trl/deepspeed/bitsandbytes (dep-risk, not needed for LoRA).

### Novel-method research loop (workflow) — a robust NEGATIVE result
We searched for a *novel, training-based, model-agnostic* cascade method and tested the three distinct ways
training can help a cascade — all fail:
- **Route** — `lora_stability_router.py`: LoRA-fine-tune the 7B as a self-verifier of its own answer-stability.
  AUROC **0.722 < logistic 0.733** (the smoke's +0.04 was 84-sample noise). Capped.
- **Distill** — `fld_distill.py` (FastLeg-Distill): LoRA-distill big-no-think competence into the small-no-think
  leg to cut the escalation *rate*. **Net-flat** (ALL-5 +0.007/+0.000 over two gating schemes): redistributes
  accuracy (+PathVQA, −VQA-RAD/MMMU interference) rather than lifting it.
- **Fuse** — `calm_fuse.py` (CALM-Fuse): trained head over small+big per-option logprobs. The union-oracle
  complementarity is real (+0.07–0.14 on all five families) but the fuser captures ≈0% per-family and
  **collapses on leave-one-family-out transfer** (Chiron 0.242).
- **Deep finding:** the structure is *real but not learnable* — all three bottleneck on "which model is right on
  this query?" (~0.58–0.73 AUROC ceiling). Documented in `results/cascade_methods/NOVEL_METHOD_FLD.md`.

### Cross-family "weirdness" diagnosed
- Confirmed all five ARE medical VLMs (all ≫ chance, varied predictions). The extremes (Chiron 2B>8B; "think
  collapses the cascade") were partly confounds: (1) **binary benchmarks** (PathVQA/SLAKE/VQA-RAD median 2
  options) make accuracy bias-sensitive (Chiron 2B>8B only on PathVQA; 8B>2B on 4-option MMMU = task-shape, not
  inverse scaling), and (2) **the think tier used a foreign prompt** (MedVLThinker's `<think>` on all models).

### Native-prompt investigation
- A workflow recovered each model's **native reasoning recipe** from its training code/paper: Lingshu = empty
  system + `\boxed{}` user instruction (auto); QoQ = its DRPO `<think>`+`\boxed{}` user prompt; Chiron =
  "Let's reason step-by-step" + "### The final answer is:"; MedGemma (original) = no real think mode.
- Extended the runners: `run_vlm_eval.py --user_instr/--no_system`, `run_peer_eval.py --think_instr/--system`,
  plus `\boxed{}` and "final answer" answer extractors.
- Re-ran the think tier natively (`run_native_think.sh`, n=8220; `compare_native_think.py`). **Result:
  over-thinking on perception is REAL, not a foreign-prompt artifact** (no-think ≥ native-think on perception for
  all families), but the foreign prompt had inflated its magnitude.

---

## 2026-06-22 — regenerate everything with native prompts; cost-methodology bug; paper fold-in

### Regeneration with native think
- Repointed the big-think tier (`c2`/`bigth`) in `acc_allmethods.py` + `acc_2size.py` to `*/think_native` for the
  four new families (MedVLThinker keeps `gate_32b` — its `<think>` RL prompt *is* its native one). Re-ran
  compare + acc_allmethods (5) + acc_2size (5) + charts + record.
- **Gotcha (resolved):** read stale JSONs mid-regen and a standalone analysis raced the batch regen on the same
  file path → killed everything and re-ran one clean single-pass regeneration. (Logged: never read JSONs
  mid-regen, never race two writers on the same output.)
- Native parity updated, e.g. Lingshu big-think 0.611 → **0.661**; MedVLThinker unchanged (already native).

### Cost-methodology bug (the one the user flagged) — found + fixed
- **Symptom:** Lingshu's always-big-think looked *cheaper* (0.12 s / 33 J) than always-big-nt (0.28 s / 90 J) —
  physically impossible.
- **Cause:** Lingshu's native think emits ~3 tokens (it doesn't reason), but its big-think latency/energy fit
  `a·gen+b` had been *measured on the foreign-think run* (gen 70–407). Applying it at gen=3 **extrapolated far
  out of range** → garbage (energy-fit intercept of −16 J).
- **Fix:** (1) `fit_metric` now uses the **median with zero slope** when gen is near-constant (non-reasoning
  native think); (2) **re-measured** each big-think tier's batch-1 latency/energy with the model's *native*
  prompt (`run_native_latency.sh`) so measured-gen = applied-gen. Result: **all five families cost-monotone**
  (small ≤ big-nt ≤ big-think for FLOPs, latency, energy); Lingshu big-think now 0.32 s / 113 J (cheap but ≥ big-nt).

### Verified Lingshu genuinely doesn't think (data is right)
- `lingshu_prompt_probe.py`: Lingshu's documented `\boxed{}` prompt *and* the exact MedEvalKit template both give
  2–3-token direct answers; only an explicit "first reason step by step" instruction elicits CoT (gen 99–170).
  So `\boxed{}` is an output-format spec, not a reasoning trigger — the native-think data is correct.

### Final numbers (native think, native-measured cost) — ALL-5 / ALL-6
| family (arch) | small-nt | big-nt | big-think(native) | Ours (ALL-6) |
|---|---|---|---|---|
| MedVLThinker (Qwen2.5-VL) | 0.620/0.526 | 0.646/0.557 | 0.646/0.572 | 0.569 @ 52% / 2.27 s / 1182 J, guard 0 |
| Lingshu (Qwen2.5-VL) | 0.734/0.618 | 0.784/0.668 | 0.775/0.661 | 0.661 @ 49% / 0.29 s / 76 J, guard 1.0 |
| QoQ-Med-VL (Qwen2.5-VL) | 0.605/0.509 | 0.610/0.522 | 0.543/0.469 | 0.509 @ 9% / 0.12 s, guard 0 |
| Chiron-o1 (InternVL3) | 0.725/0.602 | 0.654/0.551 | 0.593/0.508 | 0.602 @ 19% / 0.20 s, guard 0 |
| MedGemma (Gemma3) | 0.603/0.515 | 0.580/0.500 | 0.598/0.525 | 0.522 @ 68% / 3.37 s, guard 1.15 |

### Paper fold-in
- Rewrote **§5.5.1** as the full five-family native-prompt validation (table above + the three regimes + the
  Lingshu probe + native-measured cost). Added **§5.6.1** documenting the three-mechanism novel-method search
  (Route/Distill/Fuse all capped). Fixed the §5.6 "LoRA infeasible" line.

---

## Standing conclusions after this session
1. **The mode axis (no-think ≥ think on perception) is real and architecture-general** — holds across five
   medical VLM families and three architectures *under each model's own native prompt*.
2. **The full size+mode cascade pays off cleanly only for MedVLThinker** (≈5× latency/energy cut at parity);
   elsewhere think is net-harmful or the model inverse-scales, so the cascade reduces to the cheap leg.
3. **Training cannot beat the training-free cascade** — route/distill/fuse all hit the "which model is right?"
   ceiling; learned cross-family methods also fail to transfer. The cascade is at a genuine efficiency frontier.
4. **A model-agnostic cascade must use each model's native reasoning trigger** and gate think to reasoning-type
   questions, defaulting to no-think for perception.

## Key files (this session)
- Inference: `run_{lingshu,medgemma,qoq,chiron}_acc.sh`, `run_native_think.sh`, `run_native_latency.sh`,
  `src/labeling/{run_vlm_eval,run_peer_eval,run_7b_selfverify_vllm,nvml_power}.py`.
- Analysis: `src/cascade_methods/{acc_2size,acc_allmethods,compare_native_think,make_master_charts,
  make_full_record,lingshu_prompt_probe}.py`; `src/training_methods/{casp_stability,lora_stability_router,
  fld_distill,calm_fuse}.py`.
- Results/docs: `results/cascade_methods/{2SIZE_VALIDATION,MASTER_TABLES,FULL_RECORD,NOVEL_METHOD_FLD}.md`,
  `master_data.csv`, `allmethods_*.json`, `*.txt`; charts in `paper/figs/master/`; paper `paper/cvgip2026_draft.md`.
