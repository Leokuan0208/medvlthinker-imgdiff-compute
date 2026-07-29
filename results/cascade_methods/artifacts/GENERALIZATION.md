# Cross-family external validity of the paper's key findings

> # ⚠️ FINDING 1 IS SUPERSEDED — annotated 2026-07-29
>
> **The Finding-1 section of this file (the 5×7 table, the 15/20 count and the reasoning-half verdict) is
> superseded.** Its think arms were **prompt-unmatched**, and for MedVLThinker also **resolution**-unmatched.
> The re-derivation from the best-matched arms already on disk is:
>
> **`artifacts/finding1_corrected_2026-07-29.json`** (code `src/cascade_methods/finding1_corrected.py`),
> preceded by the audit that found the defect, **`artifacts/finding1_prompt_matching_audit.json`**.
>
> **What changed:**
> - **Perception half is STRONGER: 17/20 strictly negative** (not 15/20), **19/20** no better than +0.02,
>   **14/20** with 95% CIs excluding zero, pooled **−0.0401 [−0.0456, −0.0347]** over **30,250** paired
>   samples. Robust: three independent correction policies all give 17/20 — the as-published 15/20 is the
>   outlier. Two cells **flip positive→negative**: MedVLThinker PMC-VQA +0.0055 → −0.0075, Lingshu PMC-VQA
>   +0.0115 → −0.0425.
> - **Reasoning half is DOWNGRADED to model-dependent, not universal**: 12/15 point-positive but only
>   **4/15** with a CI excluding zero and **1/15 significantly negative**.
> - **WITHDRAWN — all 7 Lingshu-32B cells, both directions.** The "native think" instruction
>   (`runners/run_native_think.sh:7`) is purely an answer-**format** string with no reasoning trigger
>   (measured 3.0 generated tokens vs 3.0–3.3). Its repaired genuinely-reasoning arm
>   (`ckpts/acc_gen/lingshu32b/think_fullres`, 150–259 tokens) says: perception — reasoning **hurts**, 4/4
>   strictly negative, all CIs excluding zero, pooled **−0.0866 [−0.0972, −0.0757]** (resolution-matched
>   pairing); reasoning — **nothing** (MMMU +0.0000, MX-R +0.0048, MX-U +0.0271, none significant).
>   ⇒ *Lingshu-32B must not be cited as evidence that reasoning helps reasoning-heavy medical VQA.*
> - **WITHDRAWN — QoQ-Med-VL-32B as reasoning evidence**: MMMU +0.0706 → **+0.0118** (CI spans zero), and
>   MedXpert-Understanding is significantly **negative** (−0.0433, p = 0.022).
> - **MedGemma-27B's PathVQA win is REAL and survives full matching**: +0.0399 → **+0.0413
>   [+0.0220, +0.0607]** on a fully-matched pair. It is the **one** genuine exception to the perception half.
> - **STILL BROKEN, not repairable offline:** the **open-text** think-vs-direct comparison
>   (`src/labeling/run_openvqa.py:26/27`) has a live style/length grading channel. A matched-prompt re-run is
>   in flight; treat the open-text half of Finding 1 as **provisional**.
> - **Dependency issue (recorded, not fixed):** local uncommitted edits (mtime 2026-07-02) to
>   `MedEvalKit/utils/question_formats.py:11` and `MedEvalKit/utils/MMMU/data_utils.py:158` added a reasoning
>   trigger but **deleted** the answer-format clause the direct arm still carries. Pre-edit
>   `eval_results_*_think` dumps are therefore invalid as reasoning evidence (2.6–3.2 generated tokens);
>   post-edit `*_reason` dumps do reason but are format-unmatched. **MedEvalKit is a protected dependency —
>   it was not modified.** Reverting/repairing is a decision for the researcher.
>
> **Findings 2 and 3 below are unaffected.** Everything below this banner is kept verbatim as the record of
> what was published on 2026-07-08.

**OFFLINE (existing checkpoints/docs), no GPU. No fabricated numbers.** Finding-1 deltas are recomputed
directly from `master_data.csv` by `_build_generalization.py`; Findings 2/3 are quoted verbatim from the
cited docs. Machine-readable copy: `generalization.json`. Generated 2026-07-08.

Sources: `artifacts/master_data.csv`, `artifacts/reframe_vs_bigthink.json`, `artifacts/overthink_generalize.txt`,
`docs/archive_mcq/{2SIZE_VALIDATION,OPENENDED_CASCADE,TRAINED_VERIFIER_RESULT}.md`,
`docs/current/{RESEARCH_RESULTS_2026-07,VERIFIED_FACTS}.md`.

---

## FINDING 1 — reasoning hurts perception, helps reasoning (ACROSS 5 medical families)

> ⚠️ **SUPERSEDED (2026-07-29) — prompt-unmatched arms.** Read
> `artifacts/finding1_corrected_2026-07-29.json` instead. Corrected headline: **17/20** strictly negative,
> pooled **−0.0401 [−0.0456, −0.0347]** (n = 30,250); all 7 Lingshu cells and QoQ's reasoning cells are
> **withdrawn**; the reasoning half is **model-dependent**. The corrected cross-family table (best-matched
> arms, bold = 95% CI excludes zero) is:
>
> | family | PMC | SLAKE | VQA-RAD | PathV | MMMU | MX-R | MX-U |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | MedVLThinker-32B | −0.0075 | **−0.1274** | **−0.0846** | +0.0012 | **+0.0882** | **+0.0491** | **+0.0884** |
> | Lingshu-32B | **−0.0425** | **−0.0649** | **−0.0919** | **−0.1017** | +0.0059 | +0.0000 | +0.0235 |
> | QoQ-Med-VL-32B | **−0.0585** | −0.0144 | **−0.0662** | **−0.0523** | +0.0118 | −0.0131 | **−0.0433** |
> | Chiron-o1-8B | **−0.0680** | **−0.1010** | **−0.1103** | **−0.0654** | +0.0294 | +0.0021 | +0.0273 |
> | MedGemma-27B | −0.0135 | +0.0144 | **−0.0735** | **+0.0413** | +0.0353 | +0.0263 | **+0.0830** |
>
> The `lat(th:nt)` column below is **also affected for Lingshu**: its 1.2× ratio was measured on the
> withdrawn native-think arm, which generated **3.0 tokens** — it is a format-prompt ratio, not a
> reasoning:direct cost ratio.

Δ = (always-32B-**think**) − (always-32B-**no-think**) accuracy, per family × benchmark, ALL-6 (NGC harness).
Perception = PMC/SLAKE/VQA-RAD/PathVQA; Reasoning = MMMU/MedXpert-R/MedXpert-U. Last column = measured
batch-1 think:no-think latency ratio.

| family | PMC | SLAKE | VQA-RAD | PathV | MMMU | MX-R | MX-U | lat(th:nt) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedVLThinker | +0.005 | **−0.084** | **−0.077** | +0.011 | +0.065 | +0.047 | +0.092 | 49.1× |
| Lingshu | +0.011 | −0.010 | **−0.070** | −0.017 | −0.012 | +0.001 | +0.009 | 1.2× |
| QoQ-Med | **−0.085** | **−0.065** | **−0.077** | **−0.063** | +0.071 | 0.000 | −0.038 | 42.8× |
| Chiron/InternVL3 | **−0.071** | **−0.108** | **−0.092** | **−0.051** | 0.000 | +0.006 | +0.033 | 14.6× |
| MedGemma | −0.008 | +0.005 | −0.018 | **+0.040** | −0.012 | +0.040 | +0.065 | 45.1× |

*(Everything from here to the end of this Finding-1 section is the superseded 2026-07-08 text — kept as the
record. See the banner and the corrected table above.)*

**Consistency (perception half):** **15/20** perception cells have think ≤ no-think **strictly**; **19/20**
within a ±0.02 noise band. VQA-RAD is negative for **all 5** families; SLAKE for **4/5** (MedGemma +0.005 ≈ 0).
The single genuine perception win is **MedGemma:PathVQA (+0.040)**.

**Reasoning half:** think clearly helps the *genuine-think* families — MedVLThinker (MMMU +0.065, MX-R +0.047,
MX-U +0.092) and QoQ (MMMU +0.071). It is muted where the model has **no promptable think mode** (Lingshu:
MMMU −0.012, latency ratio 1.2× ⇒ it answers directly even when asked to think) or **inverse-scales** (Chiron:
MMMU 0.0). Confirmed on a *faithful* separate harness (MedEvalKit, `reframe_vs_bigthink.json`): MMMU think-gain
Lingshu +0.027 / MVT +0.100 / **InternVL3-38B +0.120** — a 3rd architecture. Supplementary: two non-medical
architectures (`overthink_generalize.txt`) are also pooled-negative on perception — InternVL2.5-8B −0.008,
Phi-3.5-V −0.019.

**Verdict (Finding 1):** the "think hurts perception" half is **strongly cross-family** (5 medical families
+ 2 extra architectures; the two big drops SLAKE/VQA-RAD are negative in nearly every family). The "think helps
reasoning" half is **cross-family for models that have a real think mode** (MedVLThinker, InternVL3, QoQ-on-MMMU)
and **muted where none exists** (Lingshu) or the model inverse-scales (Chiron). Net: **externally valid, not a
Lingshu artifact** — indeed Lingshu is the *weakest* case for the reasoning half.

---

## FINDING 2 — MCQ-vs-open-text signal gap (~0.6 AUROC MCQ → ~0.87 open-text)

- **MCQ ceiling (~0.6).** Gate bake-off on MedVLThinker competent-4 MCQ (`RESEARCH_RESULTS §1.5`): every gate
  saturates — AUROC_detect 0.643–0.693, AUROC_recover 0.506–0.614. Cross-family corroboration is indirect: in
  the 5-family ACC bake-off all 10 gate methods cluster within ~0.003 accuracy per family, i.e. no gate
  separates from any other on MCQ.
- **Open-text (~0.87).** `OPENENDED_CASCADE.md §2c`, AUROC for "cheap 7B is wrong": **MedVLThinker-7B** cheap →
  confidence **0.735** / self-consistency **0.781**; **Lingshu-7B** cheap → confidence **0.866** / SC 0.845.
  Verifier discrimination AUROC 0.924 (n=8512). Answers are median 1–2 tokens, so the driver is 4-option
  discreteness, **not** answer length.

**Verdict (Finding 2):** the **direction is cross-family** — open-text routing signal clears the MCQ ~0.6 wall
for **both** MedVLThinker-7B (0.735–0.781) and Lingshu-7B (0.845–0.866). The **peak ~0.87 magnitude is
Lingshu-specific** (a natively-calibrated open-ended cheap model); MedVLThinker's MCQ-RL miscalibration caps it
lower but still above the wall. The MCQ ~0.6 saturation itself is primarily established on MedVLThinker (the
deployed family), with only indirect cross-family support.

---

## FINDING 3 — trained best-of-N outcome verifier

- **Headline (Lingshu-7B verifier):** 4 datasets, n=1064 held-out — greedy 0.413 / SC 0.411 / **trained 0.501**
  / oracle 0.592 = **49% of the oracle gap**; matches/beats 32B (0.462) — but "beats 32B" is seed-dependent
  (win seed0 +0.039, tie seed1), so the honest claim is *competitive with the 5× model*.
- **Cross-family (MedVLThinker-7B, the original family), `VERIFIED_FACTS §H`:** from-scratch verifier on
  SLAKE+VQA-RAD — SLAKE 0.564→**0.622** (42% of gap, works); VQA-RAD 0.500→0.470 (fails, n=54 noisy);
  **POOLED 0.547→0.583 (25% of gap, positive but weaker than Lingshu)**. The Lingshu-trained verifier even
  transfers *onto* MedVLThinker outputs (49–61%) better than a from-scratch MVT verifier (25%).
- **Best-of-N signal on MedVLThinker** (`OPENENDED_CASCADE §2c`): self-consistency/diversity error-detection was
  demonstrated on MedVLThinker-7B (SC AUROC 0.781 > confidence 0.735). Generator portfolio spans 3 families
  {Lingshu-7B, MedVLThinker-7B, InternVL3-8B} (`RESEARCH_RESULTS §1.3`).

**Verdict (Finding 3):** **cross-family but base-quality-dependent.** Demonstrated on **two** base families —
Lingshu-7B (strong, 49%) and MedVLThinker-7B (partial, 25% pooled). The **robust** cross-family claim is
*training beats training-free selection and zero-shot self-verify*; the *beats-the-32B* claim is seed-dependent
even on Lingshu. So it is **not Lingshu-only**, but the magnitude and the "beats 32B" framing do not transfer
uniformly.

---

## One-line external-validity summary
- **Finding 1 (CORRECTED 2026-07-29): perception half EXTERNALLY VALID (stronger than published) — 17/20
  strictly negative, 14/20 CI-significant, pooled −0.0401 [−0.0456, −0.0347] on 30,250 paired samples across
  5 medical families, plus 2 extra architectures (7/8 cells). Reasoning half MODEL-DEPENDENT, not universal:
  only 4/15 cells CI-significant; MedVLThinker-32B / MedGemma-27B / InternVL3-38B yes, Lingshu-32B and
  QoQ-Med-VL-32B no. Open-text half PROVISIONAL (re-run pending).**
  *Superseded text:* ~~Perception over-thinking holds across 5 medical families + 2 extra
  architectures; reasoning-gain holds wherever a real think mode exists.~~
- **Finding 2: EXTERNALLY VALID IN DIRECTION.** Open >> MCQ signal on 2 families; the peak ~0.87 is
  Lingshu-specific (calibration).
- **Finding 3: PARTIALLY CROSS-FAMILY.** Trained verifier positive on 2 base families; magnitude/"beats-32B"
  is base- and seed-dependent (Lingshu-strongest).
