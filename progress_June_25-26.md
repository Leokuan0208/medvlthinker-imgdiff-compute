# Session progress log — June 25–26, 2026 (the trained-verifier program)

> Continues from `progress_June_24.md`. Records the 2026-06-25→26 work: after the user dropped abstention,
> a fresh new-method loop that (a) mapped the *luck floor* of training-free selection across every output
> type, then (b) found and exhaustively validated the genuine positive — a **trained outcome verifier** that
> breaks the luck floor for both free-text answers and structured grounding boxes, including a real
> PhysioNet benchmark. All numbers from real checkpoint output (standing no-fabrication rule); every headline
> figure audited against its `result.json`. Paper §5.9–§5.10 + abstract + intro contributions updated.
> Full detail in `results/cascade_methods/{OPENENDED_SELECTION_LUCKFLOOR, RECOVERABILITY_IS_CAPACITY_BOUND,
> KNOWLEDGE_AUGMENTATION_FEASIBILITY, NEW_DIRECTIONS_2026-06-25, TRAINED_VERIFIER_RESULT, BOX_VERIFIER_RESULT}.md`.

## Phase A — the luck floor (training-free is bounded everywhere; §5.9)
Probed every axis left after the gate: **action** (cheap same-model repairs recover 14% of errors the 32B
misses but are unharvestable — net-flat, ladder loses at parity → capacity-bound); **selection** (open-ended
oracle gap large, e.g. SLAKE 0.730→0.879, but a *majority trap* makes self-consistency fail and every
training-free selector sits at the random-pick floor: self-verify 0.715 < random 0.720, 32B listwise 0.758
< 32B single-pass 0.819); **synthesis** (priming the 32B with cheap candidates backfires, 0.774);
**knowledge augmentation / RAG** (genuinely-unknown errors are capacity-bound, not knowledge-retrievable;
PathVQA difficulty is genuine, a systematic audit *refuted* my own caption-artifact hypothesis);
**cross-family agreement** (real signal but collapses to "trust the stronger model"); **few-shot ICL**
(hurts). **Structured outputs too:** SLAKE/MS-CXR grounding SC-medoid is luck-floored, and the apparent
"escape" with a weak grounder is an artifact of incompetence (vanishes for a competent grounder). →
**Unified negative: training-free selection over a single model's samples is luck-bound regardless of output
type.** It's *luck, not latent knowledge* — the model doesn't know which sample is right.

## Phase B — THE POSITIVE: a trained outcome verifier breaks the luck floor (§5.10)
LoRA-fine-tune a verifier to score P(correct | image, question, output) on per-sample LLM-judge labels,
then select best-of-8. **Three confirmations, all 2-seed robust, all audited:**

| setting | greedy | training-free selector | **trained verifier** | oracle | gap captured |
|---|---|---|---|---|---|
| free-text answers (pooled-4 datasets, n=1064) | 0.413 | SC 0.411 / zero-shot-32B 0.357 | **0.501** | 0.592 | **49%** |
| SLAKE organ boxes (n=487, IoU≥0.3) | 0.197 | SC-medoid 0.164 / zero-shot 0.177 | **0.255/0.257** | 0.343 | 40–53% |
| **MS-CXR chest-X-ray pathology boxes (real benchmark, n=435)** | 0.041 | SC-medoid 0.053 / zero-shot 0.115 | **0.232/0.230** | 0.285 | **78/77%** |

Key sub-results:
- **Training is the active ingredient.** Zero-shot verification is luck-floored (free-text P(True) 0.319 <
  greedy; SLAKE box 0.177 ≤ greedy); training beats it on all (free-text 7B verifier beats the zero-shot
  **32B** verifier despite 5× smaller; MS-CXR training doubles the captured gap 30%→78%).
- **Generalizes:** the pooled-4 free-text verifier transfers zero-shot to a held-out **5th dataset**
  (RadImageNet-VQA, +0.024) and across modality (Kvasir-OOD).
- **Image-grounded** (blank-image ablation drops 0.047 — refutes Verification-Mirage's "lazy verifier").
- **A real test-time-scaling method:** best-of-K rises monotonically (0.385→0.501 at K=8; K=16 keeps rising
  to 0.424 on a fresh larger sample, with honest diminishing returns) while random stays flat.
- **Test-time compute beats parameters:** verifier-bo8 on the **7B (0.501)** beats the **32B single pass
  (0.444 pooled)** — when scaling params buys little (§5.7), test-time compute on the small model buys more.
- Published baselines: GenRM (ICLR'25), Weaver (NeurIPS'25), P(True), Med-RewardBench; constructive rebuttal
  of *Verification Mirage* (2605.10850); the novelty cell (trained multimodal outcome verifier for open-ended
  medical VQA + grounding best-of-N on judge labels) is unoccupied.

## Data / infra acquired (user granted access)
- **RadImageNet-VQA** (HF, auto-gated): 2000 open-ended Qs prepped (`prep_radimagenet.py`) — 5th dataset.
- **MS-CXR 1.1.0** (PhysioNet creds): 1448 phrase-grounding boxes + 1047 selectively-downloaded MIMIC-CXR-JPG
  chest X-rays (robust validated-retry beat the PhysioNet throttle). MedGround-R1 SOTA checkpoint is unreleased.
- **GOTCHA fixed:** Qwen2.5-VL emits boxes in SMART-RESIZED coords; for large images (chest X-rays) scale by
  W/w_bar,H/h_bar (no-op for small images). An earlier MS-CXR "negative" (IoU 0.022) was this coord bug.
- Reclaimed GPU 0 (68 GB leaked from a kill-9'd vLLM worker — killed the orphaned `VLLM::EngineCore`).

## Figures / paper
`fig_trained_verifier_unified.png` (3-positive bars), `fig_verifier_scaling.png` (best-of-K TTS curve),
`fig_selection_luckfloor.png` (§5.9). Abstract finding (1), intro contributions (v)/(vi)/(vii), §5.9, §5.10,
conclusion, and reproducibility index all updated and number-consistent.

## Bottom line
**Training-free selection is luck-floored across every output type (§5.9); a trained verifier is the
universal active ingredient that breaks it (§5.10) — 40–78% of the oracle gap for free-text answers AND
bounding boxes, on five VQA datasets and a real chest-X-ray benchmark, 2-seed robust, a genuine
test-time-scaling method, and accuracy-optimal vs simply using a 5× larger model.** The session's genuine,
exhaustively-validated novel positive. Open items (user's): rotate HF token + PhysioNet password; framing &
§5.8-abstention decisions; optional `.bib` conversion.

---

## Detailed narrative (folded from the former SESSION_REPORT, 2026-06-27)


**Project:** `medvlthinker-imgdiff-compute` (medical-VLM compute-efficiency, CVGIP 2026).
**Window:** 2026-06-25 → 26. **Scope:** everything done in this chat. **Discipline:** no fabricated numbers —
every figure here is from a real `result.json` / checkpoint, and the headline figures were audited against
their source files.

This report is organized as a *decision narrative*: for each move it gives the **question**, the
**method**, the **data**, the **result**, and — the part you asked for — the **reason behind the move**
(why that experiment, why then). A one-paragraph TL;DR first, then the phases.

---

## TL;DR

We started by deleting a finished direction (abstention) at your request and asking: *is there ANY
training-free way to improve open-ended medical-VQA accuracy?* The answer, after seven distinct mechanisms,
was a hard **no** — they are all "luck-floored" (the oracle gaps are real but unharvestable). We turned that
clean negative into the motivation for the positive: a **trained outcome verifier** that selects among a
model's sampled outputs. It **breaks the luck floor** for both **free-text answers** (49% of the oracle gap,
pooled over four datasets, transferring to a fifth) and **structured bounding boxes** (SLAKE organs 40%, the
real **MS-CXR** chest-X-ray pathology benchmark 78%). It is 2-seed robust, beats zero-shot verification and
even the 5×-larger 32B single pass, behaves as a genuine test-time-scaling method, and is the
accuracy-optimal operating point. *Training is the universal active ingredient that breaks the luck floor
across output types and domains.* This is the session's genuine, exhaustively-validated novel positive.

---

## Phase 0 — The pivot (why we threw away working code)

**Where we started.** The prior session had built §5.8, a training-free *safe-abstention / clinician-referral*
system for open-ended medical VQA, with a calibrated risk-coverage deployment.

**Your instruction.** "Not interested in anything that has to do with abstention or referral — find a
different direction."

**Reasoning for the move.** Abstention is one of three things you can do with a good error signal
(answer / escalate / abstain). With abstention off the table, and the *gate* (when to escalate) already
proven saturated in earlier sessions, the two unexplored axes were the **action** (what you DO on
escalation) and **selection** (which of N sampled answers to return). So the plan became: exhaust those two
axes, honestly, before claiming anything.

---

## Phase A — Mapping the luck floor (the negative that earns the positive)

The thesis we were testing: *can any training-free method convert the cheap model's large but unharvested
"oracle gap" into accuracy?* Each sub-experiment below was chosen to kill one specific hope.

### A.1 Action axis — cheap test-time repair (why: the gate was mined, the action wasn't)
- **Method.** Decompose the 7B's errors by repair type: LOOK-CLOSER (re-ask at full resolution), THINK
  (re-ask in reasoning mode), SCALE (escalate to 32B). All offline from existing checkpoints.
- **Data/result.** Cheap same-model repairs recover **14% of errors the 5×-larger 32B misses** (stable
  11–17% across four benchmarks) — a genuinely novel observation (scaling up is *not* a superset of
  intervening cheaply). **But unharvestable:** repairs break as many answers as they fix (per-view acc all
  ≈0.62 vs 32B 0.645); a confidence-gated 4-rung ladder *loses* at parity (43% vs 39% compute);
  max-confidence / majority ensembles saturate at the best single view.
- **Conclusion + reason it mattered.** The 32B's advantage is **capacity-bound** — no cheap transform
  substitutes for it. Doc: `RECOVERABILITY_IS_CAPACITY_BOUND.md`.

### A.2 Selection axis — open-ended best-of-N (why: free-text breaks MCQ degeneracy)
- **Reason for the move.** Earlier sessions showed MCQ routing is a *discreteness* artifact (AUROC ~0.6);
  open-ended free-text is where signals live (AUROC ~0.85, §5.7). So if selection can ever work, here.
- **Method.** Generate 8 samples per question (Lingshu-7B), LLM-judge each (neutral MedVLThinker-32B grader,
  one consistent run), compare selectors: self-consistency, self-verify P(Yes), 32B pointwise/listwise
  verify, learned fusion, candidate-conditioned synthesis. The "explode → judge → select_eval" pipeline.
- **Data/result.** The oracle gap is large and survives the judge (SLAKE 0.730→**0.879**). But
  self-consistency *fails* via a **majority trap** (the right answer is a minority vote in 74–90% of
  recoverable cases). Every training-free selector sits at the **random-pick floor** (0.720): self-verify
  0.715 (*below* random), 32B listwise 0.758 — and *none beats the 32B single pass (0.819)*. Candidate
  **synthesis backfires** (0.774, the trap drags the strong model down).
- **Conclusion + reason it mattered.** The gap is **sampling luck, not latent knowledge** — the model does
  not *know* which sample is right (if it did, that would be its greedy answer). Same luck-floor structure
  that sank single-model routing earlier in the project. Doc: `OPENENDED_SELECTION_LUCKFLOOR.md`.

### A.3 The "is it the benchmark?" check (why: rule out an artifact escape-hatch)
- **Reason.** Before concluding the *methods* are bounded, rule out that the *benchmark* is. PathVQA's
  difficulty looked (from 14 eyeballed cases) like caption-extraction artifact.
- **Method.** A systematic LLM audit classifying each question ANSWERABLE vs ARTIFACT.
- **Result — it refuted my own hypothesis.** PathVQA difficulty is **genuine** (well-formed questions are
  the *hardest*; "artifacts" had *higher* accuracy). Honest correction committed. The headroom is real but
  unharvestable, not a cleanable artifact. (Methodological point: a small eyeball can mislead; the
  systematic test corrected it — and we documented the correction rather than burying it.)

### A.4 Knowledge augmentation / RAG feasibility (why: "add information" isn't re-selection)
- **Reason.** Selection re-ranks the model's own outputs (luck-floored). Retrieval *adds* information, so it
  could escape the floor — *if* errors are knowledge-limited.
- **Method (zero-GPU).** Of the genuinely-unknown errors (oracle@8 wrong), what fraction does the higher-
  capacity 32B fix, split by knowledge vs perception question type?
- **Result.** The 32B fixes them **equally** across knowledge and perception (38% vs 36%) → the deficit is
  *general capacity*, not a retrievable knowledge gap. RAG not indicated. Doc:
  `KNOWLEDGE_AUGMENTATION_FEASIBILITY.md`.

### A.5 Cross-family agreement & few-shot ICL (why: decorrelated errors / answer-style)
- **Cross-family agreement** (two independently-trained VLMs): a real reliability signal (P(correct|agree)
  0.819 vs 0.649) but as an accuracy selector it collapses to "trust the stronger model."
- **Few-shot ICL** (align answer style): *hurts* (PathVQA 0.343→0.203).

### A.6 Structured outputs — grounding (why: verifiable outputs might escape)
- **Reason.** "Verification Mirage" (2606.10850, found via a lit-scan) says self-verification fails on
  medical VQA *except on measurable tasks*. So a *verifiable* output (bounding-box IoU) might escape.
- **Method.** Zero-download: SLAKE ships per-image `detection.json` gold boxes; generate 8 boxes, measure
  whether spatial self-consistency predicts box correctness.
- **Result — the escape is an artifact of incompetence.** A *weak* grounder (Lingshu) shows AUROC 0.82, but
  a *competent* grounder (Qwen2.5-VL) collapses to chance (0.557) and medoid-selection ties greedy. Diverse
  boxes from a weak model make rare agreements spuriously correlate with correctness; a competent model's
  consistent boxes carry no discriminative signal.
- **Conclusion.** **The luck floor GENERALIZES to verifiable/structured outputs** once the model is
  competent — a stronger, more general negative than the original free-text finding.

### A.7 The unified negative (paper §5.9)
Across **gate, action, selection, synthesis, retrieval, cross-family agreement, in-context prompting, AND
structured grounding**, training-free selection over a single model's samples is **luck-bound**. The binding
constraint is the genuine "which answer is right?" knowledge the frozen models lack. *This is what earns and
motivates the positive.*

---

## Phase B — The positive: a trained outcome verifier (paper §5.10)

**The hypothesis, and why now.** The luck floor is a property of *frozen* models — the signal can't be
*surfaced* zero-shot. The lit-scan said *trained* verifiers (GenRM, ICLR'25; Weaver, NeurIPS'25) succeed
where training-free fails. The unoccupied cell: a trained multimodal *outcome* verifier for open-ended
medical VQA / grounding best-of-N, on LLM-judge labels. So: train one.

### B.1 Free-text answer-verifier
- **Method.** LoRA-fine-tune Lingshu-7B to score P(correct | image, question, answer) on the 8,234
  per-sample judge labels; select best-of-8; honest grouped split by question (no leakage).
- **Result (definitive, pooled over all 4 datasets, n=1064).** Lifts **every** dataset: PathVQA 0.352→0.441,
  Kvasir 0.282→0.405, VQA-RAD 0.519→0.611, SLAKE 0.738→0.762; **pooled 0.413→0.501 (+0.088, captures 49% of
  the oracle gap)** vs ≤24% for any training-free selector.
- **Why each follow-up.** (i) *Is training the cause?* Zero-shot P(True) is luck-floored (0.319 < greedy);
  the trained **7B beats the zero-shot 32B verifier (0.357)** despite 5× fewer params. (ii) *Robust?* 2
  seeds (0.414/0.426 on PathVQA). (iii) *Does it use the image?* Blank-image ablation drops it 0.047 (refutes
  Verification-Mirage's "lazy verifier"). (iv) *Generalize?* Zero-shot transfer to a **5th held-out dataset**
  (RadImageNet-VQA, +0.024) and across modality (Kvasir-OOD). (v) *Test-time scaling?* Best-of-K rises
  monotonically (0.385→0.501, K=1→8; K=16 keeps rising with diminishing returns) while random stays flat.
  (vi) *Worth the compute?* Verifier-bo8 on the **7B (0.501) beats the 32B single pass (0.444 pooled)** —
  test-time compute beats parameters where scaling buys little (per-dataset: wins on VQA-RAD/PathVQA/Kvasir,
  loses on SLAKE where the 32B is genuinely stronger).

### B.2 Structured box-verifier (the generalization that makes it a *principle*)
- **Reason for the move.** If "training breaks the luck floor" is a real principle, it must hold for
  *structured* outputs too — where §5.9 showed training-free selection is also luck-floored.
- **Method.** LoRA-train Qwen2.5-VL to judge "does this red box localize the {target}?" (IoU≥0.3 = label),
  select best-of-8 boxes.
- **Result.** SLAKE organs: SC-medoid 0.164 (luck-floored) → **trained 0.255** (40% of gap), 2-seed
  (0.255/0.257). **Real MS-CXR chest-X-ray pathology** (PhysioNet, 1448 boxes, n=435): SC-medoid 0.053 →
  **trained 0.230** (77% of gap, 5.6× lift over greedy), 2-seed (0.232/0.230).
- **A bug worth recording.** An earlier MS-CXR pass looked like a *negative* (IoU 0.022) — it was a
  **coordinate-space bug**: Qwen emits boxes in its *smart-resized* space, and chest X-rays are large, so
  resized ≠ original. Scaling by `W/w_bar, H/h_bar` (no-op for small images) turned it into the headline
  positive. *Reason it matters:* the negative would have been wrong; always check the coordinate convention
  for grounding on large images.

### B.3 Why this is the contribution
Training-free selection is luck-floored across every output type (§5.9); a trained verifier recovers
**40–77%** of the oracle gap for free-text answers AND boxes, on five VQA datasets and a real chest-X-ray
benchmark — 2-seed robust, a genuine TTS method, accuracy-optimal vs a 5×-larger model. *Training is the
universal active ingredient.* Unified figure: `fig_trained_verifier_unified.png`.

---

## Data acquisition (you granted access mid-session)

- **RadImageNet-VQA** (HF, auto-gated): authenticated with your token; pulled the eval slice; prepped 2000
  open-ended questions (`prep_radimagenet.py`) → the 5th dataset.
- **MS-CXR 1.1.0** (PhysioNet, your credentials): downloaded the 1448-box phrase-grounding CSV, then
  *selectively* fetched only the 1047 referenced chest X-rays from MIMIC-CXR-JPG (not the 570 GB full set).
  PhysioNet throttling truncated many large files; a **validate-and-retry** downloader (full PIL decode +
  retry) eventually got all 1047 clean. MedGround-R1 (the MS-CXR SOTA grounder) has an **unreleased**
  checkpoint, so we trained our own grounder/verifier and framed the result as selection-over-a-weak-grounder
  (not beating the SOTA grounder's absolute IoU).
- **Security note (standing):** the HF token and PhysioNet password passed through chat → please rotate both.

## Engineering issues solved (so the science was honest)
- **Coordinate-space bug** (above) — turned a false negative into the headline positive.
- **GPU 0 leak**: a force-killed vLLM worker left an orphaned `VLLM::EngineCore` holding 68 GB; found and
  killed it, reclaiming the GPU (halved subsequent wall-clock by enabling parallel runs).
- **Degenerate-box crash**: a rare off-image box crashed PIL's `rectangle`; guarded `draw()` (sort+clamp+skip).
- **Racy/partial files, truncated downloads, stale wait-loops** — each caught and fixed; results re-run clean.

## Deliverables
- **Paper** (`paper/cvgip2026_draft.md`): abstract finding (1), intro contributions (v)/(vi)/(vii), §5.9
  (luck floor), §5.10 (trained verifier), conclusion, and reproducibility index — all updated and
  number-consistent (audited vs `result.json`).
- **Figures** (`paper/figs/limits/`): `fig_selection_luckfloor`, `fig_trained_verifier_unified`,
  `fig_verifier_scaling`, `fig_verifier_pareto`.
- **Code** (`src/`): `run_lora_verifier_open.py` (+ image-ablation, transfer, scaling-curve), the
  `run_ground_{slake,mscxr}.py` + `ground_analyze.py` + `run_lora_box_verifier.py` grounding/box-verifier
  stack, `prep_radimagenet.py`, and the Phase-A diagnostic scripts.
- **Writeups** (gitignored `results/cascade_methods/`): `OPENENDED_SELECTION_LUCKFLOOR`,
  `RECOVERABILITY_IS_CAPACITY_BOUND`, `KNOWLEDGE_AUGMENTATION_FEASIBILITY`, `TRAINED_VERIFIER_RESULT`,
  `BOX_VERIFIER_RESULT`, `NEW_DIRECTIONS_2026-06-25`; progress doc `progress_June_25-26.md`; memory updated.

## Honest scope / what we did NOT claim
- The verifier lifts *selection over a frozen model* — it does not beat a *trained* SOTA grounder
  (MedGround-R1) in absolute IoU, nor a strong base model where one exists (SLAKE: the 32B wins).
- Zero-shot *box* verification has modest signal (it judges a concrete drawn box) — so for boxes the claim is
  "training doubles/strongly improves it," not "zero-shot is useless."
- Best-of-N has diminishing returns (the verifier captures a shrinking fraction of the *growing* oracle gap).
- Statistical rigor: gains are large (≈9 s.e.); 2000-resample bootstrap CIs on the two headline gains are
  appended below.

## Bootstrap CIs (appended when the runs land)
- Free-text verifier (pooled-4, best-of-8 vs single-sample): gain **+0.116, 95% CI [+0.092, +0.139]** (n=1064) — excludes zero comfortably.
- MS-CXR box-verifier (trained vs greedy): gain **+0.191, 95% CI [+0.152, +0.232]** (n=435) — excludes zero by a wide margin.

*Both headline positives are statistically significant by 2000-resample bootstrap; the report is complete.*
