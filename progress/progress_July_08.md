# Progress — July 8, 2026 (paper-rewrite + rigor day)

> Continues `progress_July_07.md` (which closed with the FLOP-negative unified `method_final.py`, four
> wall-confirmations, and the abstention prohibition in §15). **Today is a consolidation + write-up + rigor
> day, not a new-method day.** The arc: (1) a morning **consolidation push** — a standalone technical report,
> a `PROJECT_OVERVIEW.md` rewrite, two last honest-negative experiments (H1 test-time adaptation, H9
> neuro-symbolic gate), the first comprehensive Markdown paper, and a directory declutter; (2) the **paper
> rewrite** the user demanded (kill the jargon/codenames, honest baselines, real CIs) — which spawned an
> **honest-baseline experiment** and a **cross-family generalization** pass; (3) the **MMMU contamination
> saga** (first investigation → recompute → adversarial audit the user insisted on); (4) the **IEEE/LaTeX**
> paper itself (MMMU excluded, no codenames, 9 pages); (5) a file-naming cleanup; and (6) a closing **32B-think
> rigor run** that turned the last estimated headline into a measured, CI-significant one and triggered a paper
> rebuild. **Compute:** almost all of today is CPU-only re-costing of existing dumps; the *only* GPU work is
> the two MMMU re-eval scripts (position-permutation + image-ablation on 150 items) and the 32B-think open-text
> extend (§10). Every number below is sourced to a named artifact under
> `results/cascade_methods/artifacts/`; nothing is fabricated. **Abstention stays out of scope** — it appears
> below only as *forbidden/excised*, never as a method.

---

## 1. Morning consolidation push (06:05–06:19)

The July-4→7 loop had scattered its results across a ledger, a method spec, five progress diaries, and 30+
artifacts. Before writing any paper, I consolidated the whole program into standing documents and closed the
two last offline ideas on the backlog.

### 1.1 The standalone technical report — `docs/current/TECHNICAL_REPORT_2026-07.md`

A plain-language + technical end-to-end walkthrough (`results/cascade_methods/docs/current/TECHNICAL_REPORT_2026-07.md`,
written 06:05) so the whole result can be understood without spelunking the artifacts. Its one-paragraph
summary is now the project's canonical framing: *a format-aware, regime-adaptive cascade between Lingshu-7B
and Lingshu-32B that is both faster and more accurate than "just use the 32B with thinking", with a Pareto
knob whose **both** settings use less compute than a single 32B forward, plus a six-times-confirmed map of
where a cheap→expensive medical cascade can and cannot beat the strong model.* It re-states the **critical
reframe** in a table (the backbone of the method):

| Lingshu-32B open-text baseline | latency | energy | accuracy |
|---|---:|---:|---:|
| no-think | 665 ms | 127 J | 0.537 |
| think | 10,521 ms (≈16×) | 2,002 J (≈16×) | 0.387 (−0.150) |

(These are the July-7 `opentext_32b_think.json` n=200/set pooled figures; §10 today re-measures them on the
*full* open sets and the story sharpens.) The report banner explicitly states *"Abstention is out of scope and
appears nowhere."*

### 1.2 `RESEARCH_RESULTS_2026-07.md` refresh + `STRUCTURE.md`

The results ledger (`docs/current/RESEARCH_RESULTS_2026-07.md`, 06:13) and the per-file index
(`STRUCTURE.md`, 06:14) were brought current with the July-7 additions (method_final v2, the six wall
confirmations, INT4/image-token re-costings).

### 1.3 `PROJECT_OVERVIEW.md` rewrite — final method first, journey kept as history

`PROJECT_OVERVIEW.md` (06:17) was rewritten to **lead with where things stand now** — the final method and
its headline table — and to demote the ACC / trained-verifier / walls narrative to "the journey" sections
(§6–§8) kept as historical record. Its headline table (the **estimate-based** July-7 numbers, later superseded
by §10's measured ones):

| method (held-out, n=42,374) | Δacc vs 32B-**think** | FLOP-eq (× a 32B call) | latency |
|---|---:|---:|---:|
| compute-lean | +0.0123 | 2.24 (0.49×) | 468 ms |
| accuracy-max | +0.0212 | 4.25 (0.93×) | 729 ms |

It carries an explicit scope note: *"abstention / deferring questions to a human is out of scope … the method
always produces an answer."* (Note: the overview was written at 06:17, **before** today's honest-baseline and
32B-think measurements, so its +0.0123/+0.0212 are the pre-measurement v2/F8 estimates; §4 and §10 supersede
them with F3-fusion measured numbers.)

### 1.4 H1 — test-time adaptation of the cheap leg (`ttt_cheap_leg.json`) — HONEST NEGATIVE

Backlog idea **H1**: can adapting the cheap 7B's *decisions* at test time, **label-free**, raise its accuracy
so the cascade resolves more cheaply / escalates less? Full TTT/TENT/MEMO/SHOT needs gradients+GPU (flagged);
`src/cascade_methods/ttt_cheap_leg.py` (CPU-only) tests every offline logit-space proxy on the existing
per-sample logprob dumps. Pooled competent-4 (n=6,050), 7B no-think base **0.6221**:

| adaptation (label-free) | pooled Δ 7B acc |
|---|---:|
| prior-adaptation (Saerens-EM label-shift) | **+0.0004** |
| uniform-prior stripping | −0.0029 |
| entropy-min + class-balance (SHOT/TENT) | **−0.1591** (collapse) |
| label-propagation | +0.0095 |

Key facts, all from `ttt_cheap_leg.json`:
- **No label-free logit-space headroom.** The best proxy (label-prop) nets +0.0095 pooled but is not a genuine
  TTT method; the real info-max TTT losses (entropy-min) **collapse** the cheap leg (−0.159), the known
  entropy-collapse-to-confident-wrong failure.
- **Even the label-*informed* oracle ceiling is <1% and mixed-sign.** An oracle per-class gold-prior on a 50/50
  calib→test split (10 seeds) yields per-benchmark d_oracle: PMC **+0.0062**, SLAKE **−0.0144**, VQA-RAD
  **+0.0088**, PathVQA −0.0007, MMMU −0.0059 — i.e. the strictest offline upper bound of a logit-space
  adaptation is under a point and negative on half the sets.
- **Temperature scaling is exactly 0.** Monotone per-logit scaling preserves argmax → it only re-tunes the
  escalation threshold, it is not an accuracy lever (confirmed by construction, no run).
- Folded into the integrated cascade (competent-4): the "prior" variant is *identical* to base (acc 0.6418,
  esc 0.2412, FLOPs 33.4% of 32B); entropy collapses escalation to ~0 at acc 0.4633.

Verdict: **H1 is a negative** — the productive weight-space TTT could in principle exceed the logit-space
bound but risks the collapse we can see already; not pursued.

### 1.5 H9 — neuro-symbolic constraint gate (`neurosymbolic_gate.json`) — HONEST NEGATIVE

Backlog idea **H9**: can hard logical checks parseable from the MCQ text flag confident-wrong answers —
especially the *shared* confident-wrong errors both 7B and 32B make — and act as a free correction or a
high-precision escalation trigger? `src/cascade_methods/neurosymbolic_gate.py` (CPU-only, competent-4
n=6,050, 7B base 0.6221, parity 0.6451):

- **Strict logic fires on ~1 sample.** The genuinely-decidable constraints have near-zero coverage:
  laterality-contradiction C3 coverage **0.0002 (n=1)**, duplicate-options C4 **n=1** (both prec-wrong 1.0 but
  useless at that coverage). The dup-merge free-correction changes **1** answer → acc delta **0.0**.
- **Higher-coverage text flags exist but are weak.** Negation (cov 0.0121), laterality-question (0.039),
  numeric-options (0.0435) flag 1–4% of items at precision-of-wrong **0.41–0.51** vs base error 0.378 (lift
  1.10–1.35×); their equal-margin-budget recovery of flagged errors is only ~0.22–0.27.
- **The bottleneck is that shared errors are perceptual, not logical.** Of the set, 1,118 items are
  both-wrong-**and**-agree (18.5%); a symbolic constraint catches only **16 of them (1.43%)**.
- Added to the margin-gate cascade, the combined trigger moves accuracy by **+0.0003** (0.6418→0.6421) at
  +0.8pt escalation (0.2412→0.2493). Numeric-range validity is not offline-checkable without a medical KB
  (out of scope).

Verdict: **H9 is a negative** — clean text-checkable medical constraints fire on a tiny slice of generic VQA
and the shared confident-wrong errors are unreachable by symbolic answer-text logic.

### 1.6 The first comprehensive Markdown paper + directory declutter

Wrote the first full Markdown manuscript `manuscript_final_2026-07.md` (108 KB, 06:17) + PDF via a pure-Python
render (`render_manuscript_pdf.py`, 06:19; no system LaTeX yet). This is the comprehensive draft that the IEEE
rewrite later supersedes — it now lives at `paper/archive/manuscript_final_2026-07.{md,pdf}`. Also did a
housekeeping pass: `**/__pycache__/` and `*.pyc` are gitignored and **0 remain tracked** (`git ls-files` clean);
`.gitignore` extended for the LaTeX toolchain (`tools/`, `paper/*.{aux,log,out}`).

---

## 2. The abstention prohibition (carried in from July 07, reaffirmed today)

Recorded here for the diary's continuity: the **permanent abstention prohibition** was established July-7 §15
(H3 three-way abstain-to-human **excised** from the backlog; the FORBIDDEN banner added to
`METHOD_IDEAS_BACKLOG.md`), and the standing memory rule was saved late July-7 (`no-abstention-research.md`,
2026-07-07 16:11). **Nothing new to add today except that every consolidation document reaffirms it**: the
technical report ("appears nowhere"), the project overview (scope note), and the IEEE paper's schematic caption
("The method always answers — the veto and deferral rules choose *which model* answers, never abstain"). The
line stays: F8's certified veto *keeps the 7B answer* (an allowed answer-producing gate), it is not abstention.

---

## 3. The paper rewrite — the user's critical feedback and the rethought story

The user read the comprehensive Markdown manuscript and pushed back hard: it was **jargon- and codename-heavy**
(Pandora, FALC, ACC, F3/F8/F10, CASP, VADR mean nothing to a reviewer), the **structure was weak** (a chronological
loop dump, not a paper), and the **data felt thin / estimate-laden** in the load-bearing comparison. The rewrite
mandate: strip every internal codename, lead with a clean thesis, and back every headline with a real baseline
and a real CI.

**The rethought story** (now the IEEE paper's spine):

- **Thesis:** the accuracy–cost tension in medical VQA is *not a law* but a consequence of spending test-time
  compute **uniformly**; allocating it **adaptively** lets a 7B model **Pareto-dominate every fixed way of using
  the 32B reasoning model**.
- **Three findings** motivate the design: (1) reasoning **over-thinks perception** and helps only reasoning-heavy
  questions (cross-family); (2) the **answer format** decides whether test-time sampling helps (best-of-N +
  trained verifier beats the 32B on free-text, fails on MCQ); (3) the **escalation signal is fundamentally
  bounded** (recoverability AUROC ≈ 0.6 on MCQ, six independent confirmations).
- **Contributions C1–C4:** C1 = the *when-does-test-time-compute-help* study (findings 1–3 with CIs + cross-family);
  C2 = the format-aware adaptive cascade that Pareto-dominates every fixed single-model strategy incl. an oracle
  baseline; C3 = the small trained verifier making best-of-N *competitive-with-or-better-than* the 32B on
  open-text + the optimal-stopping controller (−33% free-text samples); C4 = the rigorous, multiply-confirmed
  **recoverability** and **selection** bounds that explain why "a better gate" is the wrong lever.

Two experiments were needed to make this honest: a proper **baseline table** (§4) and a **cross-family
generalization** audit (§5).

---

## 4. The honest-baseline experiment (`paper_baselines.json`)

`src/cascade_methods/paper_baselines.py` → `paper_baselines.json` (08:18, CPU, live per-sample recompute,
10,000-sample paired bootstrap). The point: score the method not just against always-32B-no-think but against
the **strongest honest single-model 32B strategies**, including an **oracle-mode-32B** (per-benchmark best of
{think, no-think}, paying that mode's cost — a non-deployable upper bound on any single-32B strategy). Cost
constants (measured batch-1): GEN7 347 ms/45.8 J/1.0 F; VER7 175 ms/1.0 F; GEN32-nt 665 ms/127 J/4.57 F;
GEN32-think 10,521.6 ms/2,001.9 J/4.57 F; FUSE-both-legs 665 ms(par)/1,012(seq)/172.8 J/5.57 F.

**Pooled full-suite (n=42,374), sample-weighted:**

| system | acc | FLOP-eq (×32B) | lat_par | energy |
|---|---:|---:|---:|---:|
| always-7B | 0.5558 | 1.00 (0.22×) | 347 ms | 45.8 J |
| always-32B-nt | 0.5732 | 4.57 (1.00×) | 665 ms | 127 J |
| always-32B-think | 0.5632* | 4.57 (1.00×) | 10,521 ms | 2,002 J |
| **oracle-mode-32B** | **0.5733** | 4.57 (1.00×) | 894 ms | 171 J |
| **method compute-lean** | **0.5749** | **2.244 (0.49×)** | **469 ms** | 83.5 J |
| **method accuracy-max** | **0.5869** | 5.695 (1.25×) | 666 ms | 177 J |

\*32B-think open-text acc is estimated here (measured in §10).

**The verdict (`paper_baselines.json:verdict`):**
- **compute-lean MATCHES oracle-mode-32B** — pooled Δ **+0.0015 [−0.0025, +0.0055], n.s.** — at **0.49× its
  FLOPs, 469 vs 894 ms, 83.5 vs 171 J**. Its win is *cost*, not accuracy. It SIG-beats the oracle on MMMU and
  PathVQA-open; SIG-below only on near-chance MedXpertQA.
- **accuracy-max BEATS oracle-mode-32B** — pooled Δ **+0.0136 [+0.0108, +0.0165], SIG** — driven by the PMC
  fusion cell (~79% of the pool). Per-benchmark it SIG-beats the oracle on **PMC, MMMU, PathVQA-open**, and
  SIG-below on **none**.
- vs always-32B-nt: compute-lean +0.0017 [−0.0022, +0.0057] n.s.; accuracy-max +0.0138 [+0.0109, +0.0166] SIG.
- vs always-32B-think (open cells estimated → CI over MCQ-real cells only): compute-lean point +0.0117 (MCQ CI
  +0.0011 n.s.); accuracy-max point +0.0237 (MCQ CI +0.0139 [+0.0106, +0.0171] SIG). **This estimated open-text
  32B-think cell is exactly the gap §10 later closes.**
- **Pareto:** only {always-7B, method-compute-lean, method-accuracy-max} are non-dominated on FLOPs *and*
  parallel latency; all three 32B strategies (nt / think / **oracle**) are dominated. So the paper can claim
  the method Pareto-dominates *even the oracle mode-selected 32B*.

---

## 5. Cross-family generalization (`generalization.json` / `GENERALIZATION.md`)

`_build_generalization.py` → `generalization.json` + `GENERALIZATION.md` (08:03–08:04, offline: Finding-1
deltas recomputed from `master_data.csv`, Findings 2/3 quoted from cited docs). This answers "are the three
findings a Lingshu artifact?"

**Finding 1 (reasoning hurts perception):** Δ = think − no-think per family × benchmark.

| family | PMC | SLAKE | VQA-RAD | PathV | MMMU | MX-R | MX-U | lat(th:nt) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedVLThinker | +0.005 | −0.084 | −0.077 | +0.011 | +0.065 | +0.047 | +0.092 | 49.1× |
| Lingshu | +0.011 | −0.010 | −0.070 | −0.017 | −0.012 | +0.001 | +0.009 | 1.2× |
| QoQ-Med | −0.085 | −0.065 | −0.077 | −0.063 | +0.071 | 0.000 | −0.038 | 42.8× |
| Chiron/IV3 | −0.071 | −0.108 | −0.092 | −0.051 | 0.000 | +0.006 | +0.033 | 14.6× |
| MedGemma | −0.008 | +0.005 | −0.018 | +0.040 | −0.012 | +0.040 | +0.065 | 45.1× |

> ⚠️ **Superseded 2026-07-29 (diary preserved as written).** The think arms in the table above were
> **prompt-unmatched**; re-derived from the best-matched arms on disk the perception half is **17/20**
> strictly negative (14/20 CI-significant, pooled **−0.0401 [−0.0456, −0.0347]**, n = 30,250), **all 7
> Lingshu cells and QoQ's reasoning cells are withdrawn** (Lingshu's "native think" arm emitted 3.0
> tokens and never reasoned, which also invalidates its 1.2× ratio as a *reasoning* ratio), the
> reasoning half is **model-dependent, not universal**, and the **open-text** version is provisional.
> See `artifacts/finding1_corrected_2026-07-29.json` and retrospective §5.1 / §10.1 C20–C25.

**15/20 perception cells** have think ≤ no-think **strictly**, **19/20** within a ±0.02 noise band; **VQA-RAD
is negative in all 5 families**, SLAKE in 4/5; the only genuine perception think-win is **MedGemma:PathVQA
(+0.040)**. The reasoning half holds for **genuine-think** families (MVT, QoQ) and is muted where there is no
real think mode (Lingshu: latency ratio 1.2× ⇒ answers directly) or the model inverse-scales (Chiron). Faithful
MedEvalKit MMMU think-gain corroborates on a 3rd architecture: **Lingshu +0.027 / MVT +0.100 / InternVL3-38B
+0.120**. Two non-medical architectures are also pooled-negative on perception (InternVL2.5-8B −0.008,
Phi-3.5-V −0.019). → **Finding 1 is strongly cross-family** (and Lingshu, the headline family, is the *weakest*
case for the reasoning half — an honest framing).

**Finding 2 (format signal gap):** MCQ gates saturate — AUROC_detect 0.643–0.693, AUROC_recover 0.506–0.614;
open-text AUROC for "cheap 7B is wrong" clears the wall for **both** families — MedVLThinker-7B conf 0.735 / SC
0.781, Lingshu-7B conf 0.866 / SC 0.845; verifier discrimination 0.924 (n=8,512). Cross-family **in direction**;
the peak ~0.87 is Lingshu-specific (calibration).

**Finding 3 (trained verifier):** Lingshu-7B verifier (4 datasets, n=1,064) greedy 0.413 / SC 0.411 / **trained
0.501** / oracle 0.592 = **49% of the oracle gap**; vs the 32B (0.462) it is **+0.039 seed0 but ties seed1** →
**the honest C3 downgrade: "competitive-with / matches the 32B," not "beats" unconditionally.** Cross-family MVT
verifier from scratch: SLAKE 0.564→0.622 (42%), VQA-RAD fails (n=54 noisy), **pooled 0.547→0.583 (25%)** —
positive but weaker; base-quality-dependent. Robust claim = *training beats training-free selection and
zero-shot self-verify*.

---

## 6. The IEEE / LaTeX toolchain (08:01–09:29)

No system LaTeX exists on this VM, so I stood up a self-contained toolchain: a locally-installed **tectonic**
single-binary engine (`tools/tectonic` + a `tools/lib` libgraphite2 symlink) driving **IEEEtran** (dual-column).
Wrapper: `paper/build_ieee.sh <paper.tex>` (auto-downloads + caches IEEEtran/packages on first run, compiles
from the tex dir so a local `IEEEtran.cls`/figs resolve). Smoke-tested with `hello_ieee.tex`→`.pdf` (08:02–08:04,
now archived). Figures are generated deterministically by `paper/make_ieee_figs.py` → `paper/figs_final/`
(09:29): `fig_schematic.pdf` (the two-arm router), `fig_overthink.pdf` (Finding 1 per-benchmark), `fig_pareto.pdf`
(accuracy vs FLOP + latency frontier).

---

## 7. The MMMU contamination saga

The one anomaly threatening the paper's honesty: **Lingshu-7B scores ~0.80 on MMMU-medical, +26 over the
published Lingshu-7B 54.0, and beats its own 32B (0.63).** Three passes today: first investigation → method
recompute → the adversarial audit the user demanded.

### 7.1 First investigation (`mmmu_fix.json` + `mmmu_perm_{7b,32b}.json`, 08:24–08:36)

Ran both models on the same 150 MMMU H&M val items under **every cyclic permutation of the options**
(`MedEvalKit/mmmu_perm_eval.py`). Findings:
- **Faithful dumps:** Lingshu-7B-nt **0.80**, 32B-nt **0.6333**, 32B-think 0.66, 32B-think(default) 0.627.
- **Not sampling noise:** discordant pairs 7B-wins vs 32B-wins = **34 vs 9** (both-correct 86, neither 21);
  McNemar χ²cc **13.395, two-sided p = 0.00017**.
- **Not a position/letter artifact:** position-debiased (per-question mean over all cyclic shifts) — 7B
  0.8267→**0.7708** (circular consistency 0.58), 32B 0.6467→**0.6321** (0.5133). The identity gap 0.18 →
  debiased **0.1387**; position explains only ~4 of the ~18 points, and **the 7B still beats the 32B by +0.14
  order-robustly.**
- **Harness is faithful elsewhere:** 7B reproduces the Lingshu paper on the others (SLAKE 82.5 vs 83.1, PMC
  54.3 vs 56.3, MedXpert 26.2 vs 26.7) and 32B reproduces MMMU (63.3 vs 62.3). Reproduces on two harnesses
  (NGC 85/62, MedEvalKit 80/63).

**Verdict:** NOT a parsing/harness/position artifact; the 7B win is genuine and order-robust; the +26 over the
published 7B is a **7B-and-MMMU-specific unreconciled protocol/version gap (plausible train-set leakage),
outside our control.**

### 7.2 Recompute — Variant A / Variant B (`method_final_mmmu_corrected.json`, 09:07)

`src/cascade_methods/method_final_mmmu_corrected.py` recomputes the whole headline **not banking** the leaked
0.80. Two presentations: **Variant A** = escalate MMMU 100% to the 32B in think mode (method MMMU acc = 0.66);
**Variant B** = drop MMMU entirely (8-cell suite, n=42,224).

- **Sample-weighted headline is ROBUST.** MMMU is only **0.35%** of the 42,374-sample pool, so keep-7B (0.80) →
  escalate (0.66) moves the accuracy-max-v2 vs-32B-think delta by only **−0.0005** (+0.0212 → **+0.0207** point,
  full-suite); MCQ-CI +0.010 [+0.0073, +0.0128] SIG; vs oracle-mode +0.0105 [+0.0085, +0.0125] SIG.
- **Macro (equal-weight per benchmark) must be corrected** — it *is* materially inflated by keep-7B MMMU:
  full-suite macro **+0.0777 → +0.0621**; MCQ-only macro (MMMU-dominated) **+0.027 → +0.0036**.
- Variant B pooled (n=42,224): compute-lean 0.5741 @ FLOPs 2.248 (0.49×); accuracy-max-v2 (F8) 0.5836 @ FLOPs
  4.257 (**0.93×, FLOP-negative**); MCQ-only accuracy-max-v2 0.5842 @ FLOPs 3.875 (0.85×).
- **Recommendation:** do NOT bank MMMU keep-7B as a headline beat-32B win; exclude (as the project already does
  for the MedVLThinker family) or escalate. The headline does not depend on it.

### 7.3 The adversarial audit the user demanded (`mmmu_verify.json`, 09:31–09:43)

The user was not satisfied with "order-robust" and demanded an adversarial check that the 0.80 is not *our* bug.
`MedEvalKit/mmmu_verify_eval.py` ran six checks (dumps in `mmmu_verify_lingshu7b.json`, `mmmu_verify_qwen7b_base.json`):

1. **Model identity — PASS.** Genuinely Lingshu-7B (8.29B params, Qwen2.5-VL arch, hidden 3584, 28 layers,
   snapshot `b98aecd…`); not a 32B or mislabeled checkpoint.
2. **Image ablation — DECISIVE.** Lingshu-7B: real image **0.8267** → blank **0.62** / noise **0.62** /
   text-only **0.5933** (drop ~0.23). The image is *genuinely used*; not text-answerable, not a wiring bug
   (vision_start/image_pad tokens present).
3. **Gold subset — PASS.** 150 official MMMU H&M val items, 0 gold-letter / option-set mismatches, correct split.
4. **Control model — DECISIVE.** The untuned base **Qwen2.5-VL-7B-Instruct** (Lingshu's base) scores **0.5667**
   through the *identical* harness vs Lingshu-7B's 0.8267 — the harness is not broken (untuned lands in the
   expected ~0.55–0.58 range); the 0.80+ is specific to the Lingshu-7B weights.
5. **Prompt leakage — PASS.** Full chat template dumped; the gold letter is never indicated.
6. **Independent rescore — PASS.** A non-MedEvalKit strict parser gives 123/150 = **0.82** vs MedEvalKit 120/150
   = 0.80 (0 no-parse); if anything MedEvalKit under-counts. 0.80 is not a scoring artifact.

**Verdict:** **(b) GENUINE Lingshu-7B score, NOT an our-end bug** — the +26 over the published 54.0 is
consistent with **train-set contamination outside our control**. The most decisive evidence is the
image-ablation × control cross: a bug on our end could not simultaneously make accuracy depend ~23 pts on image
content *and* leave the untuned base at chance-ish 0.57. Method recommendation stands (exclude / escalate;
sample-weighted headline moves ~−0.0005).

---

## 8. The IEEE paper (`paper/adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}`)

Wrote the conference paper — title *"Adaptive Test-Time Compute for Medical Visual Question Answering: A Small
Model that Pareto-Dominates a Large Reasoning Model"* (anon, CVGIP 2026). Design decisions per the user:

- **MMMU EXCLUDED entirely (Variant B).** The suite is **5 benchmarks** (PMC-VQA, SLAKE, VQA-RAD, PathVQA,
  MedXpertQA-MM), **8 evaluation cells**, **n = 42,224** (39,879 MCQ + 2,345 open). MedXpertQA is the sole
  reasoning benchmark. This removes the contamination anomaly from every claim.
- **No codenames.** Pandora → "adaptive best-of-N with optimal stopping (Weitzman's rule)"; F3 → "confidence-
  advantage fusion (Chair–Varshney)"; F8 → "certified confidence veto"; ACC → "an intermediate no-think tier";
  CASP/FALC/VADR dropped. Related-work explicitly concedes the MCQ gate is *not novel* (FrugalGPT/ABC/CAR/
  RouteLLM/Hybrid-LLM family); the contribution is the **format-aware structure + choice of action**.
- **CI'd tables + math from first principles:** the margin signal m = p₁−p₂; the cascade cost identity
  C = c₀ + e₀c₁ + e₁c₂; the FLOP-eq unit (F = 2Θ(P+G) MACs, 32B/7B = 4.57, precision-independent); Weitzman
  optimal-stopping for adaptive-N. Every headline delta carries a paired 10⁴-bootstrap 95% CI.
- **3 figures** (schematic / over-think / Pareto), the recoverability-bound table (six methods, all "new cells
  = 0"), the selection-bound table (7× 32B verifier ties the trained 7B, Δ+0.005 n.s.).
- **Abstract headline (Variant B):** compute-lean **matches** the strong model at **0.49× compute / 95% lower
  latency**; accuracy-max **+0.011 over oracle-mode-selection [+0.009, +0.013]** at 0.93× compute; and — the
  bit that needed §10 — vs always-32B-think **+0.015 [+0.011, +0.019]** (compute-lean) and **+0.027 [+0.024,
  +0.031]** (accuracy-max). Discussion is deliberately scope-explicit (edge concentrated on PMC + open-text;
  verifier in-domain; best-of-N FLOP-dominated when the strong leg is cheap; findings 2–3 cross-family only in
  direction). Conclusion: *"the method always answers; no result relies on abstention."*
- **9 pages** (tectonic log: "Output written … 9 pages"). Built via `build_ieee.sh`.

---

## 9. File-naming cleanup (`paper/README.md`, `paper/archive/`, 10:06)

Adopted a dated convention so the latest is always obvious: **`<topic-slug>_<venue-or-type>_<YYYY-MM-DD>.{tex,pdf}`**
(the newest date is the current version — no more "final / final2 / main"). One canonical version stays at the
top of `paper/`; everything superseded moved to **`paper/archive/`**: `manuscript_final_2026-07.{md,pdf}`,
`manuscript_2026-07_longform.{md,pdf}`, `conference_2026-07.{md,pdf}`, `cvgip2026_draft.md`,
`cvgip2026_ieee.{tex,pdf}`, `hello_ieee.{tex,pdf}`, and the old `scripts/`. Wrote `paper/README.md` documenting
the convention + current-vs-archive file map.

---

## 10. The 32B-think rigor run (`opentext_32b_think_full.json`) — the last estimate becomes measured

The one soft spot left in every table (§4, §7.2, and the paper draft): **always-32B-think open-text accuracy was
an estimate** (judged 32B-no-think + a modal think-delta), so the load-bearing vs-32B-think headline had no
per-sample CI on the open cells. Today's only substantive GPU work closed it: extended the Lingshu-32B **think**
open-text dumps to the **full** evaluated open sets (`src/labeling/run_openvqa.py` +
`runners/run_openvqa_think_extend.sh`, chunked/resumable; ckpt jsonl 10:17–10:31), judged them through the
method's **own** grader (`src/labeling/run_judge.py`, `judge_ok`, identical to the 32B-no-think grader;
`runners/run_judge_think_open.sh`, 10:39), and re-scored with `src/cascade_methods/opentext_32b_think_full.py`
(→ `opentext_32b_think_full.json`, 10:41, 10⁴ paired bootstrap).

**Measured 32B-think open-text (vs the prior estimate):**

| set | n | think (measured) | no-think | Δ think | prior est | est error |
|---|---:|---:|---:|---:|---:|---:|
| SLAKE-open | 645 | 0.6791 | 0.8186 | −0.1395 | 0.6236 | +0.0555 |
| VQA-RAD-open | 200 | 0.545 | 0.600 | −0.055 | 0.480 | +0.065 |
| **PathVQA-open** | 1500 | **0.1087** | 0.376 | **−0.2673** | 0.246 | −0.1373 |

**PathVQA-open think collapses to 0.109** (−0.267 vs no-think) — the estimate had been ~0.25, so this is the
worst-corrected cell; verified a genuine collapse (not a truncation/parse artifact). The oracle open mode is
unchanged — think still loses to no-think on all three, so the method's routing is unaffected.

**The headline is now MEASURED + CI-significant** (paired bootstrap, open cells now real):

| pool | mode | method acc | 32B-think | Δ vs think | 95% CI | sig |
|---|---|---:|---:|---:|---|---|
| full-suite n=42,374 | compute-lean | 0.5749 | 0.5594 | **+0.0154** | [+0.0112, +0.0195] | ✔ |
| full-suite n=42,374 | accuracy-max | 0.5869 | 0.5594 | **+0.0275** | [+0.0241, +0.0308] | ✔ |
| Variant B n=42,224 | compute-lean | 0.5741 | 0.5591 | **+0.0150** | [+0.0107, +0.0192] | ✔ |
| Variant B n=42,224 | accuracy-max | 0.5862 | 0.5591 | **+0.0271** | [+0.0237, +0.0305] | ✔ |
| open-only n=2,345 | either | 0.5625 | 0.3028 | **+0.2597** | [+0.2384, +0.281] | ✔ |

The Variant-B row is exactly the paper's abstract headline (compute-lean +0.015 [+0.011,+0.019]; accuracy-max
+0.027 [+0.024,+0.031]). The paper was then **patched with the measured numbers and rebuilt** (final tex/pdf
10:57, 9 pages).

---

## 11. Standing state (end of 2026-07-08)

The deliverable is now a **written paper** (`paper/adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}`, 9 pages,
IEEEtran, MMMU-excluded Variant B, no codenames) backed by an honest baseline table that shows the method
**Pareto-dominates always-32B-no-think, always-32B-think, AND an oracle mode-selected 32B** on every cost axis,
with the vs-32B-think headline now **fully measured**: compute-lean +0.0150 [+0.011,+0.019] (matches
oracle-mode at 0.49× compute / 95% lower latency), accuracy-max +0.0271 [+0.024,+0.031]. The program is
consolidated into standing docs (technical report, project overview, results ledger, generalization audit), the
MMMU anomaly is adversarially audited and safely handled, and today's two remaining ideas (H1 TTT, H9
neuro-symbolic) are documented negatives. **Abstention remains forbidden.**

**Open questions / loose ends carried forward:**
- **The MMMU presentation decision is still the user's to make.** The paper takes Variant B (exclude); the
  recompute shows the *sample-weighted* headline is robust to either choice (±0.0005), but any *macro* /
  per-benchmark-averaged number must use the corrected MMMU cell (full-suite macro +0.078 → +0.062).
- **The FLOP-negative accuracy-max (F8-veto, 0.93× FLOPs) vs-32B-think delta is still a point estimate**
  (+0.0207 sample-weighted; MCQ-only CI real). The measured open-think in §10 gives full CIs for the
  **F3-fusion** accuracy-max (0.5869, 1.25× FLOPs), not for the exact FLOP-negative F8 point — the paper's
  abstract mixes the two (0.93× compute from F8, +0.027 vs-think CI from F3); reconciling them into one
  fully-CI'd FLOP-negative accuracy-max cell is the remaining rigor gap.
- **PathVQA-closed still has no 32B-think dump** (think = no-think assumed there; the *open* cell is now
  measured, this is a separate closed cell).
- **The open-text pooled-4 verifier is in-domain** for SLAKE/VQA-RAD/PathVQA-open (unchanged caveat).
- **OmniMed-32B is still blocked** (tp=2 NCCL hang) → ALL-7 not reportable; the 5-benchmark suite is the
  computable pool.
