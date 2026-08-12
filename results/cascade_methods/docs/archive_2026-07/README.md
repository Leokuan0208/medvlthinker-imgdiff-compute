# Archived docs — moved out of `docs/current/` on 2026-08-11

These describe **retired states of the project**. They are kept as the record (this project's rule is
*move, never delete* — the negative results and superseded reasoning are part of the contribution), but
they are **not** a source for any current number and were cluttering the working set.

**Before reading anything here, read `docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md` §4 and §10.**
Every number below is superseded by at least one of: the macro re-basing (2026-07-30), the verifier
decontamination (2026-08-05), or the prompt-matched reasoning baseline (2026-08-03).

| file | what it was | why archived |
|---|---|---|
| `METHOD_ACC.md` | the June 3-tier compute-configuration cascade spec | historical era; internal NGC harness, PMC-VQA `test_clean.csv` — not comparable to the live MedEvalKit track |
| `METHOD.md`, `METHODS_MASTER.md`, `MASTER_SUMMARY_2026-07.md` | early method write-ups | superseded by the retrospective and `METHOD_FINAL_2026-07.md` |
| `METHOD_MATH.md` | June-era derivations | superseded; the live cost model is in `artifacts/flop_ratio_derivation_2026-08-03.json` |
| `UNIFIED_METHOD_EXPERIMENTS.md` | July-2 experiment log | superseded by the retrospective §6 negatives catalogue |
| `OPENTEXT_BASELINE.md`, `OPENTEXT_MASTER_TABLE.md` | June open-text tables | pre-decontamination; every open-text number in them used the contaminated verifier |
| `VERIFIED_FACTS.md` | June fact list | several entries since retracted (see retrospective §10) |
| `COMPREHENSIVE_WRITEUP_2026-07-30.md` | first comprehensive write-up | superseded by `COMPREHENSIVE_WRITEUP_2026-08-03.md`, itself now pre-decontamination |
| `METHOD_deferral_router.md` | a deferral/routing spec | ⛔ **abstention-adjacent — CRITICAL RULE 6.** Historical record only; must never be revived as a direction |

**`docs/archive_mcq/`** (separate folder) holds the June MCQ-era archive and is unchanged.
