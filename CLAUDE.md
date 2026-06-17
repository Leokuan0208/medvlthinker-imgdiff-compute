# CLAUDE.md — Project Context for `medvlthinker-imgdiff-compute`

> **What this file is.** This is a context/briefing file for Claude Code. When Claude Code
> opens this folder it reads `CLAUDE.md` automatically. The purpose of this file is to (1)
> explain what this research project is, (2) describe what is currently in the folder and what
> each piece does, (3) propose a cleaner folder structure, and (4) list the rules that must be
> followed so the reorganization does not break anything.
>
> **The immediate task** is a *cleanup*: move and rename files into a structure that is easier
> to understand. Read the **"Rules & landmines"** and **"Safe cleanup procedure"** sections
> before touching anything. Several things in this folder will silently break if moved
> carelessly — those are spelled out below.
>
> **About the author.** The researcher (Leo / Li-Wen Kuan) is an engineering student who is
> relatively new to computer science. When you explain a command or a plan, be explicit and
> step-by-step; don't assume familiarity with shell, git internals, or Python packaging.

---

## 1. The project in one paragraph

This is a **medical Vision-Language Model (VLM) compute-efficiency** research project. A VLM is
a model that takes an **image + a text question** and produces a text answer — here, the images
are medical (radiology, pathology, etc.) and the task is multiple-choice medical visual question
answering (VQA). The headline result is a **two-model cascade**: a small, cheap **7B** model
answers most questions, and a large, expensive **32B** model is only called ("escalated to")
when the small model is not confident. A simple **confidence gate** decides when to escalate.
The point of the paper is **efficiency, not accuracy**: the cascade matches the big model's
accuracy while using much less compute. The deliverable is a conference paper for **CVGIP 2026**.

**Current headline numbers (treat as ground truth for context, do not edit experimental data):**

- Both models are `MedVLThinker-*-RL_m23k` (a 7B and a 32B variant).
- The gate is a **frozen confidence-margin threshold**, `τ (tau) = 0.426`, trained on a clean
  PMC-VQA training split (all eval samples held out).
- Serving the 7B at a reduced image resolution ("**cap320**"), the cascade reaches **exact
  parity** with always-using-32B accuracy (`0.572 = 0.572`) at about **74%** of the
  always-32B compute (measured with prefill-inclusive FLOPs).
- The cascade is **never worse than always-7B** on any of the six benchmarks.
- Six benchmarks total. **Four "competent" benchmarks** where the method works:
  **PMC-VQA, SLAKE, VQA-RAD, PathVQA**. **Two "excluded" benchmarks**: **MMMU** and
  **MedXpert** (both models are near chance on MedXpert; these are intentionally excluded
  from the main claims).

---

## 2. How the project got to its current state (why "dead" files exist)

This project **pivoted several times**. That history matters because the folder still contains
artifacts from abandoned directions. **Do not delete these** during cleanup — archive them.
They are the record of *why* the final method is what it is (negative results are a core part of
the paper). The arc, oldest to newest:

1. **Question-aware visual token pruning** (the original project name). Idea: drop unimportant
   image tokens to save compute. → did not yield a usable accuracy cliff.
2. **Image-difficulty-driven adaptive compute** (the `complexity` / `difficulty` / `lesion`
   files). Idea: spend more compute on "hard" images. → killed (correlations were the wrong
   sign / near zero).
3. **Single-model routing** across "axes" like think-vs-no-think and retrieval (RAG) — the
   `gate_probe`, `gate_rag`, `retrieve.py`, and `MedRAG/` artifacts. Idea: route *within one
   model*. → **definitively killed**: a "luck-floor" audit showed the oracle was ~29σ below
   the random-allocation floor, i.e. one model's confidence signals are mutually redundant and
   carry no routable signal.
4. **Cross-model 7B→32B cascade** (the current, successful direction). The structural fix was
   to route *between two different models*, not within one. This is the live work.

So in the repo you will find three layers: **live cascade code** (keep, organize),
**dependency repos** (`MedRAG/`, `MedVLThinker/` — never move), and **archived dead
directions** (keep but tuck away in `archive/`).

---

## 3. Key terms (glossary)

- **VLM** — vision-language model; image + text in, text out.
- **VQA** — visual question answering. Here it's multiple-choice (the model picks A/B/C/D...).
- **7B / 32B** — model sizes (billions of parameters). 32B is the big, accurate, expensive one.
- **Cascade** — run the cheap model first; only call the expensive model on hard cases.
- **Gate / router** — the small decision rule that decides "escalate to 32B or not."
- **τ (tau)** — the confidence threshold the gate uses. Below threshold confidence → escalate.
- **Margin** — gap between the model's top-1 and top-2 answer probabilities; the confidence
  signal the gate thresholds.
- **Escalation rate** — fraction of questions handed up to the 32B.
- **cap320 / cap640 / fullres** — image-resolution budgets (a cap on pixels via `max_pixels`).
  Lower cap = fewer image tokens = cheaper. "cap320" is the chosen operating point.
- **Prefill-inclusive FLOPs** — honest compute accounting that includes the cost of reading the
  prompt+image, not just generating the answer. (An earlier decode-only estimate was too rosy.)
- **think / no-think** — these models can emit a `<think>...</think>` reasoning trace or answer
  directly. The 7B cheap leg runs **no-think**; the 32B runs **think**.
- **RAG / retrieval** — pulling in external text (e.g. textbooks) to help answer. Part of a
  *killed* direction; `MedRAG/` and `retrieve.py` are its leftovers.
- **HistGBM / Conformal / FBE / CP-Router** — fancier gate alternatives that were tested and
  **lost** to the simple margin threshold. Their scripts are real results, kept as ablations.
- **HF vs vLLM** — two ways to run the models. **vLLM** is fast (~35× speedup) but hides true
  GPU memory use. **HF** (HuggingFace transformers) is slower but measures real VRAM, so it's
  used for the live cascade memory/energy measurement.

---

## 4. Current repository inventory (verify against the live tree first)

> **Important:** this inventory reflects the project as of mid-June 2026 and may have drifted.
> **Before acting, run a real listing** and reconcile it with this list (see the procedure in
> §6). Treat anything below as "expected, confirm on disk," not "guaranteed present."

### 4.1 Active Python code — now under `src/` (see §5 for the full map)

> **Note (post-2026-06-16 cleanup):** these scripts were moved out of the old `scripts/` into
> `src/<stage>/` and renamed for clarity. The table below uses the **original** names for the
> historical record; see §5 for each file's current path/name. The descriptions still hold.

These are the live scripts. Names in (parentheses) note history.

| File | What it does | Stage |
|---|---|---|
| `gate_router.py` | Runs a model over a dataset and writes per-sample JSONL checkpoints (the "labeler"). Produces the confidence signals the gate uses. **Has a relative `CKPT_DIR`** — see landmines. | gate / labeling |
| `run_32b_vllm.py` | Runs the 32B (vLLM, think mode) to produce the strong-model labels in `ckpts/gate_32b/`. | labeling (strong) |
| `run_7b_prune_sweep.py` | Sweeps image-resolution caps (`max_pixels`) on the 7B to produce the cheap-leg checkpoints across cap80/160/320/640/fullres. | sweep |
| `tokens_per_cap.py` | Processor-only token-count cache builder (how many image tokens each cap costs). | sweep |
| `grid_resolution_tau.py` | The calibrated 5×5 resolution×τ grid with prefill-inclusive FLOPs accounting; this is where the cap320 / τ=0.426 operating point comes from. | sweep / calibration |
| `rt_cascade.py` | The **live** co-resident cascade: 7B on GPU0, 32B on GPU1, real escalation, dual-GPU power logging via NVML. Writes `rt_cascade_cap320.jsonl`. | cascade (live) |
| `rt_analyze.py` | Analyzes the live run: per-benchmark escalation 2×2, gate discrimination, faithfulness vs vLLM labels. | cascade (analysis) |
| `cascade_breakdown.py` | Offline per-benchmark cascade breakdown from validated vLLM labels. | cascade (analysis) |
| `analyze_router.py` | The current analyzer for the gate/router checkpoints (superseded the old `analyze.py`). | analysis |
| `analyze_7b_think.py` | Merges 7B-think shards and runs the harness sanity check. | analysis |
| `fbe_and_signals.py` | Offline "bake-off" comparing the margin gate vs FBE / conformal / HistGBM, entirely on saved checkpoints (no new inference). Source of the "fancier gates lose" ablation. | analysis (ablation) |
| `oracle_luck_floor.py` | The luck-floor audit that killed single-model routing (~−29σ). | analysis (negative result) |
| `recompute_energy2.py` | Recomputes the energy-saved numbers from checkpoint data (replaces an earlier flat-power proxy). | analysis |
| `build_subset.py` | Builds `subset.csv`, a small evaluation slice. | data prep |
| `retrieve.py` | MedRAG retrieval (`--corpus` arg). **Leftover from the killed RAG direction**; keep for record. | legacy (retrieval) |

> Some scripts **import from each other** and several **hard-code file paths** (to weights, to
> datasets under `/data/dan/...`, and to checkpoint folders). This is why renaming is risky —
> see landmines.

### 4.2 Checkpoints — `ckpts/` (gitignored, often resumable, do not casually move)

- `gate_7b_pmctrain/ckpt_nothink.jsonl` — 3,000 rows, PMC-VQA **train** split. The gate's
  calibration data.
- `gate_7b_prune/<cap>/ckpt_*_nothink_norag.jsonl` — the cheap 7B leg across resolution
  caps; 8,220 rows per benchmark set.
- `gate_32b/ckpt_*_think_norag.jsonl` — the strong 32B leg; 8,220 rows. (`opt_logprobs`
  is empty here because the 32B is a reasoning model.)

> **Shard-tag convention (updated 2026-06-16):** checkpoints/feats from a **single-shard** run
> carry **no `_sKofN` suffix** (the redundant `_s0of1` was stripped). Only genuinely **sharded**
> runs are tagged — e.g. `gate_7b_think/` and `archive/single-model-routing/gate_7b_rag_axes/`
> (the archived RAG-axes grid, formerly `ckpts/gate_7b_v2/`) keep `_s0of2`/`_s1of2`. The labelers
> write the tag only when `N>1` (`SHARD_TAG`), and every reader treats `(?:_s\d+of\d+)?` as
> optional, so both forms load and shards still merge by `idx`.
- `router_margin.pkl` — the **gate artifact** itself: keys `gate`, `tau`, `signal`,
  `trained_on`. This is the deployable result.
- `rt_cascade_cap320.jsonl` — output of the live cascade run.
- `_legacy/` — old field-poor checkpoints from the 3B and early-7B runs. Keep, do not delete.

**Checkpoint JSONL schema** (so you can read them without guessing): per-sample keys are
`idx, gold, pred, ok, parse_ok, opt_logprobs (letter→logprob dict), gen_tokens, latency_s,
raw_output`. The live cascade JSONL uses `idx, dataset, escalate, ok, final` (and in some
versions `pred7, pred32, gold, margin, latency_s, energy_j, gen7, gen32`).

### 4.3 Other top-level items

- `logs/` — `nohup` output from long runs. Gitignored.
- `data/` — small inputs like `subset.csv`. Gitignored.
- `results/` — run artifacts (may be empty). Gitignored.
- `archive/` — killed directions. Subfolders: `image-difficulty/` (complexity / difficulty /
  lesion files), `old-gate-scripts/` (`gate_probe.py`, `gate_rag.py`, old `analyze.py`), and
  `single-model-routing/` (the killed direction #3: `oracle_luck_floor.py`, `router_scalar.py`,
  `router_hidden.py`, `analyze_router.py`, `pmcvqa_recoverability.py`, `extract_features.py`,
  `gap_router_probe.py`, plus `cascade_complementarity_check[_corrected].py` and
  `validate_think_harness_vs_paper.py` — all early scripts that read the dead n≈500 RAG-axes
  grid — and the grid's data itself, `gate_7b_rag_axes/` (formerly `ckpts/gate_7b_v2/`,
  gitignored). Kept as the negative-result record).
- `MedRAG/` — **dependency git repo. DO NOT MOVE OR RENAME.** `retrieve.py` imports from it.
- `MedVLThinker/` — **dependency git repo. DO NOT MOVE OR RENAME.** The eval stack uses it.
- `README.md`, `.gitignore`, `env_backup_*.txt` (a local env snapshot, gitignored).

---

## 5. Repository structure (AS BUILT — the cleanup is done)

The 2026-06-16 cleanup moved every active script out of the old flat `scripts/` into `src/`,
grouped by pipeline stage, and gave each file a self-explanatory name. **`scripts/` no longer
exists.** Killed directions were moved to `archive/`. Dependencies, checkpoints, and data were
left untouched. **Always launch scripts from the repo root** (see §7).

```
medvlthinker-imgdiff-compute/
├── CLAUDE.md   README.md   RESULTS.md   .gitignore
│
├── src/                          # all ACTIVE python (was: scripts/)
│   ├── labeling/                 # run a model over data → per-sample JSONL checkpoints
│   │   ├── run_7b_hf_labeler.py  # (was gate_router.py) — HF labeler, OOM-guarded
│   │   ├── run_7b_vllm.py        # cheap 7B no-think, full eval
│   │   ├── run_7b_think_vllm.py  # 7B think baseline
│   │   ├── run_32b_hf.py         # (was run_32b.py)
│   │   ├── run_32b_vllm.py       # strong 32B labels (TP=2)
│   │   └── run_pmctrain_vllm.py  # label the held-out PMC-VQA train sample
│   ├── sweep/                    # resolution / compute sweeps + calibration
│   │   ├── run_7b_prune_sweep.py   tokens_per_cap.py   grid_resolution_tau.py
│   │   ├── cascade_resolution_sweep.py   cascade_heldout_frontier.py
│   ├── gate/                     # train + freeze the DEPLOYED margin gate
│   │   ├── train_margin_gate.py        # (was router_train.py) → ckpts/router_margin.pkl
│   │   └── refit_gate_tau_per_cap.py   # (was router_tau_per_cap.py)
│   ├── cascade/                  # the LIVE cascade + real-time measurement
│   │   ├── live_cascade.py             # (was rt_cascade.py) → rt_cascade_cap320.jsonl
│   │   ├── measure_single_leg.py       # (was rt_measure.py)
│   │   ├── report_cascade_from_legs.py # (was rt_report.py)
│   │   └── analyze_live_cascade.py     # (was rt_analyze.py)
│   ├── analysis/
│   │   ├── cascade/              # analyses of the live cascade
│   │   │   ├── cascade_per_benchmark_breakdown.py        # (was cascade_breakdown.py)
│   │   │   ├── cascade_cost_decode_flops.py              # (was router_cost.py)
│   │   │   ├── cascade_cost_prefill_flops.py             # (was router_cost_prefill.py)
│   │   │   ├── cascade_cost_accuracy_pareto.py           # (was router_pareto.py)
│   │   │   ├── cascade_gain_bootstrap_ci.py              # (was router_bootstrap.py)
│   │   │   ├── frozen_gate_transfer_bootstrap_ci.py      # (was router_train_bootstrap.py)
│   │   │   ├── margin_gate_mechanism_diag.py             # (was router_signal_diag.py)
│   │   │   ├── gate_head_to_head.py                      # (was head_to_head.py)
│   │   │   ├── cascade_escalation_signal_early.py        # (was router_escalate.py; superseded)
│   │   │   ├── recompute_energy.py                       # (was recompute_energy2.py; CORRECTED)
│   │   │   └── recompute_energy_superseded.py            # (was recompute_energy.py)
│   │   └── ablations/           # gate alternatives that LOST to the margin gate
│   │       ├── gate_ablation_bakeoff.py                  # (was fbe_and_signals.py)
│   │       ├── gate_alt_conformal.py / _6datasets.py     # (was router_conformal[_6ds].py)
│   │       └── gate_alt_learned_gbm.py / _6datasets.py   # (was router_learned[_6ds].py)
│   ├── reporting/               # build the paper's tables + harness validation
│   │   ├── build_table1_accuracy.py        # (was extract_paper_numbers.py)
│   │   ├── build_table2_efficiency.py      # (was extract_efficiency.py)
│   │   ├── report_efficiency_per_dataset.py# (was extract_efficiency_perdataset.py)
│   │   ├── report_cascade_per_dataset.py   # (was extract_perdataset_report.py)
│   │   ├── report_medxpert_dilution.py     # (was medxpert_impact.py)
│   │   ├── merge_7b_think_and_validate.py  # (was analyze_7b_think.py)
│   │   └── inspect_timing_fields.py        # (was check_timing_fields.py, root)
│   ├── data_prep/
│   │   ├── build_eval_subset.py            # (was build_subset.py)
│   │   ├── sample_pmcvqa_train_heldout.py  # (was pmcvqa_train_sample.py)
│   │   └── prep_pmcvqa_train_sample.py     # (was prep_pmctrain_sample.py)
│   └── legacy_retrieval/
│       └── retrieve.py          # killed RAG direction, kept for record
│
├── ckpts/  logs/  data/  results/  feats/  feats_full/   # gitignored data — UNTOUCHED
├── archive/                     # killed directions (keep, don't delete)
│   ├── image-difficulty/        old-gate-scripts/
│   └── single-model-routing/    # oracle_luck_floor, router_scalar/hidden, analyze_router,
│                                #   pmcvqa_recoverability, extract_features, gap_router_probe,
│                                #   cascade_complementarity_check[_corrected], validate_think_harness_vs_paper
│                                #   (all read the dead RAG-axes grid), + gate_7b_rag_axes/ data (gitignored)
├── rt_cascade_cap320.jsonl      # live cascade output — left at root (hub: default arg of many scripts)
│
├── MedRAG/                      # dependency repo — DO NOT MOVE/RENAME
└── MedVLThinker/                # dependency repo — DO NOT MOVE/RENAME
```

**Naming conventions used:** lowercase, `_`-separated, names that describe the *role* and read
clearly **without** the folder for context (Leo's request). Archived files kept their original
names so they still match the `logs/*.log` files keyed to them.

---

## 6. Safe cleanup procedure (follow this order)

The previous reorganizations on this project followed a strict "inspect first, move second"
discipline because a wrong move can orphan a resumable checkpoint or break an import. Replicate
that discipline:

1. **Inspect the real tree.** Run a listing and compare it to §4. Do not trust this file over
   what's on disk.
   ```bash
   cd ~/medvlthinker-imgdiff-compute
   { command -v tree >/dev/null 2>&1 && tree -L 2 -a -I '.git|__pycache__|*.pyc'; } \
     || find . -maxdepth 2 -not -path './.git/*' -not -path '*/__pycache__/*' -print | sort
   ```

2. **Find cross-references before renaming anything.** Renaming a file is only safe if every
   place that mentions it is updated too. Grep for imports and hard-coded paths:
   ```bash
   # which scripts import which (so a rename doesn't break an import)
   grep -rn "import\|from " --include=*.py scripts/ | grep -vi "^.*#"
   # hard-coded paths to weights / datasets / checkpoint dirs
   grep -rn "/data/dan\|ckpts\|CKPT_DIR\|gate_7b\|gate_32b\|router_margin" --include=*.py scripts/
   ```
   Anything that shows up here must be updated in the same change as the move/rename.

3. **Propose the full move/rename plan to Leo as a list, and wait for confirmation.** Do **not**
   write a blind `mv` script. Present: "these N files move here, these get renamed to X, here's
   every reference I'll update." Renaming files that other files import or that checkpoint
   folders are keyed to is exactly how a half-finished run gets orphaned.

4. **Prefer `git mv` over `mv`** if the file is tracked, so history is preserved. (Untracked,
   gitignored things like `ckpts/` use plain `mv` — but see the landmine about not moving live
   checkpoints at all.)

5. **Dry-run, then verify.** After moving, re-run the tree listing, then confirm nothing broke:
   the scripts still import cleanly (`python3 -c "import ast,sys; ast.parse(open(f).read())"`
   per file, or a real `--help`), and `git status` shows only the intended changes.

6. **Never `rm -rf` a whole directory** to "tidy." `ckpts/` holds resume state; `archive/` and
   `_legacy/` are the project's record of negative results. Move, don't delete.

---

## 7. Rules & landmines (read before moving anything)

- **`MedRAG/` and `MedVLThinker/` are separate git repositories and live dependencies.**
  `retrieve.py` imports from `MedRAG`; the eval stack uses `MedVLThinker`. Moving or renaming
  either breaks imports for zero benefit. **Leave them at the root, untouched.**
- **`ckpts/` checkpoints are resumable and gitignored.** Scripts write per-sample JSONL and
  resume from the last completed line if restarted. **Moving a checkpoint folder while a run
  could resume into it orphans that resume state.** Do not relocate `ckpts/` contents as part of
  cleanup unless Leo explicitly says a run is finished. The `router_margin.pkl` gate artifact
  lives here and is the deployable result — handle with care.
- **`CKPT_DIR` and similar paths are relative to the launch directory, not the script.** Several
  scripts resolve `ckpts/...` against the current working directory at launch time. The
  invariant that keeps everything working is: **always run from the repo root**
  (`cd ~/medvlthinker-imgdiff-compute` first). If you move scripts into subfolders, do **not**
  also start running them from inside those subfolders — keep launching from the repo root
  (e.g. `python3 src/cascade/live_cascade.py`), or the relative checkpoint paths will point at the
  wrong place and silently re-run from scratch.
- **`.gitignore` keeps `ckpts/`, `logs/`, `data/`, `results/`, env backups, and the big
  dependency repos out of git.** The committed tree is meant to be **code only**. After any
  restructure, re-check `.gitignore` still matches the new paths, and before any commit, inspect
  the staged file list and scan for unexpectedly large files.
- **No fabricated numbers — ever.** This is a standing, non-negotiable rule for this project.
  Every figure in the paper, slides, or site must come verbatim from real experimental output.
  If a number is needed and not available, recompute it from checkpoints; never invent it.
- **Code-delivery convention Leo uses:** brand-new files / standalone scripts are delivered as
  a heredoc (`cat > path << 'EOF' ... EOF`) so he can paste them whole. **Edits to existing
  files** are delivered as a plain code block (the snippet to change), which Leo applies himself
  — do **not** wrap an edit-to-an-existing-file in a heredoc that overwrites it. Do not append
  execution commands onto a file-creation heredoc.
- **Long-running jobs use `nohup`, never `tmux`.** Use checkpointed, resumable runs with
  per-sample error guards.
- **Two GPUs, shared storage.** The VM has dual A100 80GB GPUs (user `jamesyang`), a shared
  `/data` mount (weights + datasets live there, not in the repo), and a shared home. Code lives
  in the repo; **weights and datasets never go in the repo** — they stay under `/data/dan/...`.

---

## 8. Environment & where things live (for running, not editing)

- **Repo:** `~/medvlthinker-imgdiff-compute`
- **Weights:** 7B at `/data/dan/weights/MedVLThinker-7B-RL_m23k`; 32B at
  `/data/dan/weights/MedVLThinker-32B-RL_m23k`. (A legacy `MedVLThinker-3B-RL_m23k` exists from
  the early phase.)
- **Eval data:** `/data/dan/dataset/MedVLThinker-Eval` — 8,220 samples across the six
  benchmarks (`pmc_vqa`, `pathvqa_closed`, `slake_closed`, `vqa_rad_closed`, `MMMU-medical`,
  `MedXpertQA-MM`). Train splits: PMC-VQA at `/data/dan/dataset/pmc_vqa_train`; others under
  `/data/dan/dataset/{vqa_rad, path_vqa, slake}`.
- **Inference:** vLLM (NGC container, ~35× faster) for bulk labeling; HuggingFace transformers
  for the live cascade VRAM/energy measurement.
- **Critical model quirk:** these models only emit a `<think>` trace when the system prompt is
  *exactly*: "You will solve a problem/request. You should provide your thoughts within
  `<think>` `</think>` tags before providing the answer." Without that exact prompt they answer
  directly. Don't paraphrase it.

---

## 9. Quick "what is this file?" decision guide for cleanup

When you encounter a file and aren't sure where it belongs:

- Is it `MedRAG/` or `MedVLThinker/`? → **dependency, leave at root, don't touch.**
- Is it a `.jsonl`, `.pkl`, or under `ckpts/`/`logs/`/`results/`? → **data/checkpoint, gitignored,
  do not move during a code cleanup unless told the run is done.**
- Does its name contain `complexity`, `difficulty`, `lesion`, `gate_probe`, `gate_rag`? →
  **killed direction → `archive/`.**
- Is it `retrieve.py` / RAG-related? → **legacy retrieval, keep for record (own subfolder).**
- Is it one of the active scripts in §4.1? → **goes under the new `src/<stage>/` grouping.**
- Unsure? → **leave it, list it, and ask Leo.** Never guess-delete.
```
