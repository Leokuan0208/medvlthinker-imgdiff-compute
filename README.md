# medvlthinker-imgdiff-compute

**Test-time compute for medical vision-language models: what actually helps.**

The accuracy–cost tension in medical VQA is not a law — it is a consequence of spending test-time
compute **uniformly**: the same reasoning, the same number of samples, the same model, on every
question regardless of what the question needs. This repo builds a cascade that spends selectively.

**The current method** is a **format-aware adaptive cascade** between **Lingshu-7B** and
**Lingshu-32B**, evaluated on **MedEvalKit** (the harness that reproduces Lingshu's published
numbers exactly). It detects from the prompt text alone whether a question is multiple-choice or
open-ended and runs a different policy for each — an MCQ arm (margin gate → 32B in *direct* mode)
and an open-text arm (7B best-of-N with adaptive N → a **trained LoRA verifier** selects). One knob,
two settings, **both using less compute than a single 32B forward pass**.

**Headline** — Variant B (MMMU excluded), 5 benchmarks / 8 cells, **n = 42,224**, vs
always-32B-with-reasoning (0.5591 measured):

| setting | accuracy | compute (× one 32B pass) | latency (par.) | Δ vs 32B-reasoning, 95% CI |
|---|---|---|---|---|
| compute-lean | 0.5741 | **0.49×** | 469 ms | **+0.0150 [+0.0107, +0.0192]** |
| **accuracy-max** | 0.5836 | **0.93×** | 731 ms | **+0.0245 [+0.0216, +0.0274]** |

Source: `results/cascade_methods/artifacts/f8_mode_vsthink_ci.json`. **The honest one-line claim** is
*"matches the strong model at roughly half the compute, with a significant accuracy gain on two
specific cells — open-ended free text and PMC-VQA — plus a measured characterization of why the
remaining cells are unwinnable."* Reproduce: `python3 src/cascade_methods/method_final.py`.

**Paper:** `paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf` (9 pages, IEEEtran).

> ## ➜ Start here
> **[`results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`](results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md)**
> — the definitive account of the whole project (2026-06-17 → 2026-07-29): the arc, the method, the
> results with CIs, ~90 negative results, 16 honest holes, and a corrections log. **Read it before
> quoting any number from anywhere in this repo.**
> Then: `PROJECT_OVERVIEW.md` (plain-language, 30 min) · `READING_GUIDE.md` (the full reading order) ·
> `meetings/progress_report_professor_2026-07-27.html` (13 source-cited sections) ·
> `CLAUDE.md` (project rules — including the **permanent abstention prohibition** and the
> no-fabricated-numbers rule) · `STRUCTURE.md` (the code map) · `RESULTS.md` (where each number lives).

> **The negative results are half the contribution.** Two walls were mapped: *recoverability*
> ("will the strong model fix **this** error?" — ~0.5–0.6 AUROC from anything cheap, **16 independent
> mechanisms** hit it) and *selection* (a verifier converts only 74–82% of oracle-of-N, **13 attempts**).
> Every genuine positive came from **changing what is being routed rather than improving the router**.

> **Historical (2026-06, MedVLThinker era).** The project began as a two-model MCQ cascade: a frozen
> **confidence-margin gate** (τ = 0.426) escalating low-confidence questions from a 7B to a 32B, with
> the 7B served at reduced resolution (cap320) — parity with always-32B (0.5718 vs 0.572) at ~**74%**
> of always-32B compute over 6 benchmarks / 8,220 samples, never worse than always-7B. Its successor,
> the 3-tier compute-configuration cascade (**ACC**), reached parity with always-32B-think at
> **11.34 s → 2.27 s latency (−80%), 100 → 52% compute, ~5.3× less energy**. Both are real results on
> the **internal** harness (evaluation context B) and must never be mixed with the MedEvalKit figures
> above — see `RESULTS.md` §A.2. ACC's agreement gate later turned out to be prior art
> (Agreement-Based Cascading, arXiv 2407.02348); what carried forward is the *structure*.
> Original target venue was CVGIP 2026; drafts are in `paper/archive/`.

> **⚠️ Preservation.** Last commit `8cdefef` (2026-07-02). The entire Lingshu/July chain — the IEEE
> paper, the nine July diaries, the July-27 deck, `paper/figs_final/`, and every headline script — is
> **untracked**; `results/` and `MedEvalKit/` are gitignored. Committing the working tree is the
> standing top-priority chore.

## Layout

All active code lives under `src/`, grouped by pipeline stage. **Always run scripts from the
repo root** (e.g. `python3 src/cascade/live_cascade.py`) — several resolve `ckpts/...` paths
relative to the launch directory.

```
src/
├── labeling/      run a model over a dataset -> per-sample JSONL checkpoints
│                  run_7b_vllm.py, run_32b_vllm.py, run_openvqa.py, run_judge.py,
│                  run_ground_{slake,mscxr}.py, run_peer_eval.py, nvml_power.py, ...
├── sweep/         resolution / compute sweeps + the calibrated res×τ grid
├── gate/          train + freeze the deployed margin gate (-> ckpts/router_margin.pkl)
├── cascade/       the LIVE co-resident cascade + real-time measurement (MedVLThinker era)
├── cascade_methods/  122 files. The July LINGSHU headline chain -- method_final.py,
│                  paper_baselines.py, integrated_method.py, beat32b_{fusion,more}.py,
│                  integrated_pandora.py, opentext_32b_think_full.py,
│                  method_final_mmmu_corrected.py, f8_mode_vsthink_ci.py -- plus the
│                  June research loop rooted at harness.py (acc.py, compare.py, ...)
├── training_methods/  the TRAINED methods -- run_lora_verifier_open.py (the verifier),
│                  run_lora_box_verifier.py, verifier_{scaling_curve,transfer_eval}.py
├── analysis/
│   ├── cascade/   analyses of the live cascade (cost, complementarity, mechanism, energy)
│   └── ablations/ gate alternatives that LOST to the margin gate (conformal, learned, FBE)
├── reporting/     build the paper's accuracy/efficiency tables + harness validation
├── data_prep/     build eval subsets, sample held-out splits, prep Kvasir / RadImageNet
└── legacy_retrieval/  retrieve.py — leftover from the killed RAG direction

runners/            38 shell launchers (each cd's to the repo root first)
progress/           13 dated daily diaries (June 17 -> July 8) — the primary narrative record
paper/              the IEEE deliverable + build/figure scripts + figs_final/;  archive/ = old drafts
meetings/           dated .html decks (2026-07-27 is the best summary in the repo)
docx/               generated Word exports
archive/            killed directions, kept as the record of negative results:
                    image-difficulty/, old-gate-scripts/, single-model-routing/
MedRAG/, MedVLThinker/, MedEvalKit/   dependency repos — DO NOT move or rename
                    (MedEvalKit/eval_results_*/ holds the faithful eval dumps — do not clean)
ckpts/ logs/ data/ results/ feats*/   gitignored data/checkpoints
```

Full per-file index: **`STRUCTURE.md`**.

## Pipelines

**Current (Lingshu / MedEvalKit).** Faithful evaluation dumps are produced by `MedEvalKit/eval.py`
(via `runners/run_*_medeval*.sh`, using `/data/dan/medeval_venv`) into `MedEvalKit/eval_results_*/`;
open-text answers are generated by `src/labeling/run_openvqa.py` and graded by `src/labeling/run_judge.py`
(MedEvalKit's open-half exact match is known-broken). Everything after that is **offline CPU re-costing**
over the saved per-sample dumps: `src/cascade_methods/paper_baselines.py` builds the 9 cells, and
`method_final.py` / `method_final_mmmu_corrected.py` / `opentext_32b_think_full.py` /
`f8_mode_vsthink_ci.py` produce the tables and CIs. Number → script → artifact map: `RESULTS.md` §A.1.

> *Honest caveat:* the final method has **never been executed end-to-end as a live pipeline** —
> escalation is `np.where(margin < τ, ok_32B, ok_7B)` over recorded correctness, with latency and energy
> from measured per-leg batch-1 constants. The one genuine live cascade run in the repo,
> `ckpts/rt_cascade_cap320.jsonl`, belongs to the older MedVLThinker work below.

**Historical (MedVLThinker, 2026-06).** 1. **Label** the eval sets and the held-out PMC-VQA train split
with the 7B (no-think) and 32B (think) — `src/labeling/`. 2. **Sweep** image-resolution caps and calibrate
the (resolution, τ) operating point — `src/sweep/` → cap320, τ = 0.426. 3. **Train + freeze** the margin
gate — `src/gate/` → `ckpts/router_margin.pkl`. 4. **Run the live cascade** co-resident (7B on GPU0, 32B
on GPU1) with real escalation and NVML power logging — `src/cascade/live_cascade.py`. 5. **Analyze /
report** — `src/analysis/`, `src/reporting/`.

## Always run from the repo root

Several scripts resolve `ckpts/...` relative to the launch directory, and the `src/cascade_methods/`
modules use bare sibling imports that only resolve when run as `python3 src/cascade_methods/<x>.py` from
`~/medvlthinker-imgdiff-compute`. Long jobs use `nohup`, never `tmux`.
