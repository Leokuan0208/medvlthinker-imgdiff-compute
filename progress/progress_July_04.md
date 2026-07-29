# Progress — July 3 (evening) – July 4, 2026

> Continues `progress_July_03.md` (which ended with OmniMedVQA running under the fixed parser). This
> entry covers the paper rewrite, the website push, the OmniMed reproduction saga + its fixes, and — the
> real content — a three-workstream research investigation (eval-validity / gate / architecture) that
> re-pointed the whole project at ONE unified method with efficiency as a first-class axis. No GPU job
> was launched to write this; every number below is read off an artifact/doc/log named inline.

## 1. Paper rewrite — the "strong-but-honest, efficiency+accuracy" reframing

Wrote two coupled documents under a single unified framing:

- **Long-form manuscript** — `paper/manuscript_2026-07_longform.md` (+ `.pdf`, ~140 KB / ~264 KB).
- **Conference distillation** — `paper/conference_2026-07.md` (+ `.pdf`), title *"Structure and a Little
  Training Beat the Gate."*

The thesis (from the conference abstract/intro, all numbers verbatim from checkpoints): deployment cost
is the problem (a 32B reasoning model ≈ **11 s / ≈ 6 kJ** per batch-1 question vs ≈ **0.2 s** for a 7B),
and the paper reports *one hard negative + two positive levers + a unifying explanation* rather than a
"better gate." (i) **The luck floor** — over a *frozen* model no training-free signal beats trivial
baselines; a dozen escalation signals cap at recoverability AUROC ≈ 0.5–0.69 because the two models fail
together (φ = 0.372; competent-set P(32B wrong | 7B wrong) = 0.584). (ii) **Structure (ACC)** — a
compute-configuration cascade cutting latency **11.34 → 2.27 s (−80 %)**, FLOPs to 52 %, energy ≈ 5× at
parity; plus the faithful cross-family MCQ margin cascade at **−17 … −69 % FLOPs** with a think tier
adding **+0.03 … +0.12** MMMU across three families. (iii) **A little training** — a small trained
outcome verifier that breaks the selection floor and **beats the strong 32B/38B** on accuracy (Lingshu
0.421 vs 0.331, MedVLThinker 0.344 vs 0.277, InternVL3 0.255 vs 0.218) for free-text and boxes (MS-CXR
77–78 %, a 5.6× lift). Honesty is written into the contributions: the agreement *gate* is explicitly
shared with prior cascading (FrugalGPT/ABC/CAR); the novelty is the compute-configuration *structure*,
the verifier's *unification* across formats, and the luck-floor *characterization*. Reproducibility
anchor kept front-and-centre: Lingshu-32B MMMU = **0.633 = paper 62.3** under the faithful MedEvalKit
protocol.

**PDF pipeline note.** No LaTeX is installed on this VM (`which pdflatex xelatex` → nothing), so the
Markdown → PDF path is a pure-Python render (xhtml2pdf/pisa + bundled DejaVu fonts for the Unicode math
glyphs φ, τ, ≈, −). Both `.pdf`s in `paper/` were produced this way; regenerate by re-running that
render, not a `latexmk`.

## 2. Website — MkDocs site pushed (separate repo)

Updated and pushed `~/question-aware-vtp-medvlm` (its own GitHub repo, summarized here only): split the
weekly progress logs into individual per-day pages; added **Weeks 7–8**, the **Lingshu / InternVL3
baseline** pages, and **bugs #12–14**. Nothing in this repo depends on it.

## 3. OmniMedVQA reproduction saga (the 7th benchmark) — four failure modes, four fixes

OmniMed is the last of Lingshu's paper suite (Open-access ≈ 89k QA). Getting it to run end-to-end was a
day of whack-a-mole; recording each failure honestly because the fixes are the deliverable:

- **(a) Parser `KeyError` on `modality_type`.** `cal_metrics` in
  `MedEvalKit/utils/OmniMedVQA/OmniMedVQA.py` assumed every sample carries `modality_type` /
  `question_type`; several of the 42 sub-datasets omit them, so the metric pass crashed *after hours of
  generation*. Fixed with `sample.get("modality_type", "unknown")` (and the same for `question_type`) so
  aggregation tolerates the missing keys.
- **(b) `MAX_MODEL_LEN` env lever on both vLLM wrappers.**
  `MedEvalKit/models/{InternVL/InternVL_vllm.py, Qwen2_5_VL/Qwen2_5_VL_vllm.py}` now read an optional
  `MAX_MODEL_LEN` (and the Qwen wrapper a `GPU_MEM_UTIL`) to cap the KV cache; unset = model default, so
  existing behaviour is unchanged. InternVL3-38B advertises `max_seq_len 65536` which will not fit the
  tp=2 KV cache, so it is pinned to 16384 (24000 for the long MedXpert prompts).
- **(c) Strong 32B/38B legs: NCCL heartbeat hang at tp=2.** Intermittent tail-hang on a stuck collective.
  Rather than disable the heartbeat monitor, we *keep* it (it aborts the stuck collective fast) and moved
  the strong legs onto a **chunked tp=2 driver** (`runners/run_omnimed_strong_chunked.sh`): each leg is 6
  resumable `--num_chunks/--chunk_idx` shards, so a hang costs ~1/6 of the run and is retried on a
  freshly-cleaned GPU (×3), with an aggregation backstop that re-fires the last chunk if `metrics.json`
  didn't materialise. **tp=1 for the 32B was ruled out**: 64 GB of weights + multimodal activation
  exceeds the 80 GB card (the `MAX_MODEL_LEN` lever alone can't buy it back), so the 32B *needs* tp=2.
- **(d) Cheap 7B/8B legs: container cgroup OOM when run two-at-a-time.** Two concurrent cheap legs each
  build a ~200 GB image batch in preprocessing → memory-cgroup OOM-kill (anon-rss ≈ 245 GB in the kill
  log). One leg fits (an earlier single run reached 43/45 sub-sets), so switched to a **strictly
  sequential cheap driver** (`runners/run_omnimed_cheap_seq.sh`, tp=1, one GPU, retry ×3, skip-if-done)
  that emits `OMNIMED_CHEAP_DONE` to gate the strong driver. (The earlier `run_omnimed_cheap.sh` /
  `_parallel.sh` / `_fix.sh` are the dead-end predecessors, left for the record.)
- Also queued the one missing baseline cell as the strong driver's first job: **IV3-38B MedXpert
  DIRECT** (no-think) at cap 24000.

## 4. Three-workstream research investigation (the important part)

The user re-focused the research: stop enumerating gates, converge on **one unified method with
efficiency (latency + compute) as a first-class axis alongside accuracy**, over the whole Lingshu suite.
Three parallel investigations:

### 4a. Eval validity — verdict SOUND

Our faithful MedEvalKit eval *does* reproduce Lingshu paper Table 6 once metric definitions are matched.
The "massive" VQA-RAD / PathVQA gaps were never a bug — they are a **closed-only vs open+closed-blend**
metric mismatch; the clean-MCQ/closed cells match at both sizes. The open-ended exact-match "failure" was
a scorer bug, fixed by the validated LLM judge. The single genuine anomaly is **Lingshu-7B MMMU +26**,
which stays excluded from claims. Metric rule locked for the paper:

- **Pure MCQ exact-match**: MMMU, PMC-VQA, MedXpert.
- **Judge-scored open+closed blend**: SLAKE, VQA-RAD, PathVQA.

### 4b. Gate — the proxy was wrong, and the answer is a trained verifier

Ran the offline, CPU-only clean bake-off (`src/cascade_methods/gate_unified_bakeoff.py` →
`results/cascade_methods/artifacts/gate_unified_bakeoff.json`), 5-fold out-of-fold, in both regimes.
Findings:

- **Cascade quality tracks recoverability-AUROC, not detection-AUROC.** The earlier bake-off ranked gates
  by *detection*-AUROC (does the cheap model err) — the wrong proxy. In the artifact the highest
  *detect*-AUROC gate loses: `learned-RICH` hits detect-AUROC **0.693** (top) in MCQ competent-4 yet its
  ADC is *below* the deployed margin gate (Δ mean −0.0015). What predicts a good cascade is
  *recoverability*-AUROC (will the strong model fix it), exactly Jitkrittum's wall. The session's
  cross-gate summary put this at r ≈ **+0.65** (recover) vs r ≈ **−0.21** (detect).
- **"CASP-stability > agreement" is a regime-dependent tie**, not a win: CASP-stability's ADC vs deployed
  is −0.0008 (win-frac 0.034) on MCQ competent-4 — a wash.
- **A new gate, EG-RC** (Expected-Gain via Recovery-Calibration) — decomposes the Bayes-optimal deferral
  score into a *learnable* calibrated detector × a *low-variance* per-decile recovery prior, so it reduces
  to plain confidence exactly when recovery is bin-independent (which is *why* confidence is near-optimal).
  Honest result: EG-RC **wins in the weak-strong open-text regime** (OPEN pooled routing-eff **0.232**,
  beating the deployed verifier-conf 0.198 on 99.2 % of seeds) but is **beaten there by the literal
  Jitkrittum Diff-Prob** (routing-eff **0.271**, +ADC on 100 % of seeds), and it **loses on VQA-RAD-only**
  (routing-eff 0.029). No robust cross-regime win → not a headline.
- **Unifying insight**: across both regimes the gate that actually helps is **trained self-verification /
  verifier-P(correct)** — the format-agnostic, Jitkrittum-optimal signal. In MCQ competent-4 the 7B
  self-verify (AutoMix) gate is the only one with positive routing-eff vs deployed (**0.0995** vs margin
  0.0876); in open-text the trained verifier-confidence *is* the deployed gate. Literature placed:
  Jitkrittum 2307.02764, FrugalGPT, AutoMix, CCPS 2505.21772, CP-Router — the **medical-VQA cascade-gate
  niche is open**.

### 4c. Architecture — the deterministic router works; the unified verifier is the recommendation

- **Deterministic format-aware ROUTER (Method C)** — `src/cascade_methods/unified_router.py` →
  `unified_router.json`. Detects each item's format *from the prompt text, never the gold*, and dispatches
  MCQ/closed → margin cascade, open → verifier best-of-8 cascade, then scores the *whole pooled* stream as
  one accuracy-vs-cost number. It is a **tested Pareto win over always-strong on both accuracy and
  cost, across all three families**:

  | Family | router acc | always-strong | Δ | FLOPs (prefix-shared) | latency | MCQ esc |
  |---|---|---|---|---|---|---|
  | Lingshu | 0.5461 | 0.5462 | −0.0001 (parity) | 0.378 (0.328) | 0.685 | 0.10 |
  | MedVLThinker | 0.5215 | 0.5218 | −0.0003 (parity) | 0.544 (0.508) | 0.881 | 0.29 |
  | InternVL3 | 0.5136 | 0.5135 | +0.00004 (parity) | 0.771 (0.707) | 0.694 | 0.51 |

  i.e. it holds always-strong accuracy at **0.38–0.77× FLOPs** and **0.68–0.88× latency** (recovering
  ~all of the strong-vs-cheap gap — router is +0.009 / +0.015 / +0.025 over always-cheap).

- **The recommended no-router unified method = the Unified Generative Verifier (UGV)**: one trained
  verifier over *generated* answers serves BOTH best-of-N selection AND the gate, for both formats. On the
  open half it already beats the 32B (`open_bestofN_adaptive.json`), and an adaptive-N policy nearly
  matches best-of-8 at a lower mean sample count:

  | Family | strong (32B) | verifier bo8 | adaptive (mean N) | cost vs strong |
  |---|---|---|---|---|
  | Lingshu | 0.331 | 0.414 (+0.082) | 0.411 (6.4) | 1.47× FLOPs |
  | MedVLThinker | 0.277 | 0.333 (+0.056) | 0.330 (5.8) | 1.32× FLOPs |
  | InternVL3 | 0.218 | 0.250 (+0.032) | 0.248 (6.9) | 1.59× FLOPs |

  Honest reading: this is an **accuracy** win (test-time scaling), *not* a compute saving — the adaptive
  policy never escalates (esc = 0) and spends ~1.3–1.6× the always-strong FLOPs. The compute win comes
  from the router/cascade side; the accuracy win comes from the verifier side; UGV is the bid to make
  *one* mechanism carry both.

- **Selective think-tier was a negative result for Lingshu** (no accuracy headroom to justify the extra
  tier there) — kept as a documented negative, not part of the unified recommendation.

- **The decisive open question**: does the UGV hold on **MCQ-as-generation** — i.e. when we run MCQ items
  as free-text generation and score them with the *same* trained verifier? That is exactly what
  `src/labeling/run_mcq_generate_verify.py` produces (N=8 samples/item, `--mode content` options-hidden
  vs `--mode letter` options-shown), driven by `runners/run_ugv_experiments.sh`. If yes, one method covers
  the whole suite; if no, MCQ and open stay two pipelines behind the deterministic router.

## 5. Standing state / what runs next

The GPU pipeline is chained and self-supervising (each stage waits on the previous stage's DONE sentinel
in its log):

1. **Sequential cheap OmniMed** (`run_omnimed_cheap_seq.sh`) — in flight (Lingshu-7B leg generating; no
   `metrics.json` yet) → emits `OMNIMED_CHEAP_DONE`.
2. **Chunked strong OmniMed + IV3-38B MedXpert-direct** (`run_omnimed_strong_chunked.sh`) — waits on the
   cheap sentinel → emits `OMNIMED_STRONG_DONE`.
3. **UGV experiments** (`run_ugv_experiments.sh` → `run_mcq_generate_verify.py` + batch-1 latency +
   re-run of `unified_router.py`) — waits on the strong sentinel.

Decision recorded: the project is now aimed at **one unified method with efficiency as a first-class
axis** over the full Lingshu suite; the queued UGV run is the experiment that decides whether the unified
generative verifier survives MCQ-as-generation. Everything above is offline analysis and engineering —
no GPU/eval job was started to produce this log.
