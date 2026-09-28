# `src/audit_2026_09_18/` — the scripts of the 2026-09-18 → 09-20 full project audit

**Findings:** `results/cascade_methods/docs/current/AUDIT_2026-09-18.md`.
**Sweep reports:** `results/cascade_methods/docs/current/AUDIT_2026-09-18_appendix/`.
**Artifacts these scripts wrote:** `results/cascade_methods/artifacts/*_2026-09-{18,20}.json`.

Everything here is **read-only on the repository**. Nothing refits or overwrites a shipped
checkpoint; `ckpts/train/genframe_head_pooled_ens_v2` (24 frozen heads) is loaded and never written.
Two scripts need a GPU (the cross-family judge); the rest are CPU.

> ⚠️ **Thread rule.** Run every CPU fit at **≤ 4 threads** (`OMP_NUM_THREADS=4`,
> `torch.set_num_threads(4)`), and treat even that as **load-dependent**: during this audit a
> 4-thread fit segfaulted at load average ≈ 21 and ran clean at 2. Shard for parallelism instead of
> raising the thread count. The crash gives no traceback, so run long fits under
> `src/reporting/supervisor.py` with retries. (CLAUDE.md §0 landmine, amended by this audit.)

> ⚠️ **Always launch from the repo root** — these scripts resolve `results/…` and `ckpts/…` against
> the current working directory, like everything else in this project.

---

## Run order

Each step's output is the next step's input. Steps 5 and 8 are the only GPU jobs.

| # | script | needs | writes |
|---|---|---|---|
| 1 | `em_rescore_pooled_probe.py` | the frozen probes, the feature caches, the sc8 dumps + judge labels | `em_rescore_pooled_probe_2026-09-18.json` **and the per-question cache** every later step reads |
| 2 | `dump_probe_scores.py` | the frozen probes + feature caches | `probe_scores_heldout_lingshu.json` (per-candidate scores, so nothing else has to re-read 20 GB of features) |
| 3 | `judge_length_bias.py` | the step-1 cache | `judge_length_bias_2026-09-20.json` |
| 4 | `build_xjudge_preds.py` | the step-1 cache + the sc8 dumps | `heldout_lingshu7b_medgemma27b.jsonl` (one row per distinct held-out *(benchmark, idx, answer string)*, plus each question's greedy answer, in `run_judge.py`'s schema) |
| 5 | **`launch_medgemma27b.sh`** *(GPU)* | step 4's file | `heldout_lingshu7b_medgemma27b.judge.jsonl` — `src/labeling/run_judge.py` with `--judge_model google/medgemma-27b-it --tp 1`, under the project's per-GPU `flock`, same prompt and decoding as the labels of record |
| 6 | `rescore_xjudge.py` | steps 1 + 5 | `xjudge_rescore_medgemma27b_2026-09-20.json` |
| 7 | `build_xjudge_train_preds.py` | the probe's training rows' metas + the sc8 dumps | `train_lingshu7b_medgemma27b.jsonl` |
| 8 | **`launch_medgemma27b_train.sh`** *(GPU)* | step 7's file | `train_lingshu7b_medgemma27b.judge.jsonl` |
| 9 | `judge_2x2_refit.py` | steps 1 + 5 + 8 | `judge_2x2_2026-09-20.json` |
| 9b | `judge_consensus_refit.py` | the same inputs | `judge_consensus_2026-09-20.json` |
| 10 | `pseudolabel_precision.py` | steps 1 + 2 | `pseudolabel_precision_2026-09-20.json` |
| 11 | `template_stratified_gain.py` | steps 1 + 5 | `template_stratified_gain_2026-09-20.json` |
| 12 | `pilot_cross_model_transfer.py` | the frozen probes + Qwen's feature caches | `pilot_cross_model_transfer_2026-09-18.json` |

Steps 3, 10, 11 and 12 are independent of each other once step 1 (and, for 11, step 5) has run.

**Not in the chain:**
- `verify_agent_claims.py` — independent spot-verification of the sweeps' most serious claims,
  run last, read-only.
- `sample_flips.py` — samples upward judge flips that share **zero** gold tokens, for reading by
  hand (36 were read; see `AUDIT_2026-09-18.md` §2.2).

---

## What each script answers

- **`em_rescore_pooled_probe.py`** — the centrepiece. Re-scores the **shipped** eight-benchmark probe
  verifier (`ckpts/train/genframe_head_pooled_ens_v2`, 24 frozen heads) on the held-out image halves
  in **four currencies on identical picks**: the judge of record, the project's lenient normalised
  exact match, strict exact match and token-F1. Nothing is fitted. It reuses the project's own split
  (`md5("nd"+img_md5) % 2 == 0` is held out), feature loading, frozen standardizers and `rank_avg`
  readout, so its judge macro must reproduce the recipe's — it does, to +0.000113.
- **`dump_probe_scores.py`** — caches the ensemble's per-candidate scores (`p_mean`, `rank_mean`) so
  later analyses never touch the feature caches again.
- **`judge_length_bias.py`** — is the judge-currency gain a length or leniency artefact? Length-only
  selectors (pick-the-longest / shortest), judge-positive rate by word-count bucket holding lenient
  EM at 0, and the gain restricted to questions where the pick is not longer than greedy.
- **`build_xjudge_preds.py` / `rescore_xjudge.py`** — re-grade the **same picks** with a judge from a
  different model family. Nothing is refit, so this isolates the question: do picks made by a probe
  trained on judge A's labels still beat greedy when judge B grades them?
- **`build_xjudge_train_preds.py` / `judge_2x2_refit.py`** — the 2×2. Two probes, the same rows, the
  same recipe and seeds, differing **only** in whose labels they were trained on; each graded by both
  judges and by lenient EM. The off-diagonal is the honest cross-judge gain.
- **`judge_consensus_refit.py`** — the same design with consensus labels (both judges say correct)
  and soft labels ((A+B)/2).
- **`pseudolabel_precision.py`** — how clean would probe-selected self-training data be, against the
  literature's default self-consistency filter? **This is data filtering for training; the deployed
  model still answers every question** (CLAUDE.md critical rule 6 — abstention is forbidden and this
  is not it).
- **`template_stratified_gain.py`** — the by-image split keeps images disjoint but 55–100 % of
  held-out questions on five benchmarks reuse a question string from the training half. Stratifies
  the gain by question-string repeat and by gold-answer-seen-in-train, in three currencies.
- **`pilot_cross_model_transfer.py`** — applies the frozen **Lingshu** probes, unmodified, to
  **Qwen's** hidden states, with and without a label-free standardizer re-estimate, against a control
  that must reproduce the recipe.
- **`pilot_lineage_alignment.py`** — the follow-up: can a **label-free** ridge map
  Z<sub>qwen</sub> → Z<sub>lingshu</sub>, fitted only on rows where both generators produced the same
  normalised answer to the same question, close the gap that pilot leaves? Writes
  `pilot_lineage_alignment_2026-09-20.json`.

---

## Where the intermediates live

**`/data/dan/audit_2026-09-18/tmp/`** — not in the repo, and not in `ckpts/`.

```
tmp/em-rescore/   the per-question cache (step 1) + its log
tmp/me/           probe scores, the CPU pilots' JSON + logs
tmp/me/xjudge/    the two judge input files, the two .judge.jsonl label files
                  (198,378 MedGemma-27B judge labels), the 2x2 outputs + logs
tmp/{code-audit,data-integrity,docs-html,docs-md,docs-top,replication,stats-baselines}/
                  each sweep's own scratch (see sweeps/ below)
```

These are expensive to regenerate — the two GPU judge passes are the bulk of the audit's cost — and
they were backed up 2026-09-28, sha256-verified, to `/data/dan/backups/medvlthinker-imgdiff-compute/2026-09-28/`audit_2026-09-18/.

## D3 and the 2026-09-28 check (run from the worktree or repo root; CPU, 4 threads)

```
python3 src/audit_2026_09_18/d3_lineage_pair.py --target qoq --out results/cascade_methods/artifacts/d3_lineage_qoq_2026-09-28.json
python3 src/audit_2026_09_18/d3_pair_efficiency.py       results/cascade_methods/artifacts/d3_pair_efficiency_2026-09-21.json   # ~43 min (2,565 s)
python3 src/audit_2026_09_18/d3_ridge_identity_prior.py  results/cascade_methods/artifacts/d3_ridge_identity_prior_2026-09-28.json  # ~29 min (1,760.6 s)
python3 src/audit_2026_09_18/d3_writeup.py             > results/cascade_methods/docs/current/LINEAGE_TRANSFER_2026-09-28.md
python3 src/audit_2026_09_18/check_2026_09_28_report.py > results/cascade_methods/docs/current/CHECK_2026-09-28.md
```

Inputs: `feats_hidden/generator_eval_{finelayer*,qwen_*,qoq_*}`, the frozen probes in
`ckpts/train/genframe_head_pooled_ens_v2`, and the judge labels `ckpts/openvqa/cheap_lingshu7b/ckpt_*_{qwen25vl7b,qoq7b}.judge.jsonl`.
The pair-2 native reference comes from `src/training_methods/head_final_stack.py --generator qoq`.

## `sweeps/`

The seven parallel sub-audits' own scripts, one directory each: `code-audit/`, `data-integrity/`,
`docs-html/`, `docs-md/`, `docs-top/`, `replication/`, `stats-baselines/`. Each sweep's written
report is the matching file in
`results/cascade_methods/docs/current/AUDIT_2026-09-18_appendix/`.
