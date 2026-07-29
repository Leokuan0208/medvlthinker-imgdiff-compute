# Progress — June 29–30, 2026

> **Reconstructed 2026-07-02** from the commit log + `docs/current/{OPENTEXT_BASELINE,OPENTEXT_MASTER_TABLE,UNIFIED_METHOD_EXPERIMENTS}.md`
> and `docs/archive_mcq/` writeups. All numbers quoted from those commits/docs.

## Theme: the open-text unified-method search — the verifier-augmented cascade **beats the 32B**

This is the window where the headline open-text result landed and the gate question was settled exhaustively.

### June 29 — baselines → gate → the accuracy win
- **Phase 0 (baselines):** built the complete open-text baseline table (judge-based). Lingshu = the strong-32B
  target; MedVLThinker-32B is genuinely weak on free-text (confirmed real, not a bug). Added the running
  experiment log (`UNIFIED_METHOD_EXPERIMENTS.md`): baselines → gate → verifier → integration, with a cost model.
- **Phase 1 (gate):** **verifier-confidence is the best cascade gate** (0.518 @ 34% escalation); self-consistency
  and n_distinct don't help.
- **Phase 3 (the headline):** the **verifier-augmented cascade beats Lingshu-32B on accuracy** — 0.517 vs 0.462
  @ 35% escalation, per-dataset. Open problem flagged: cost tension (~3.8× FLOPs from best-of-N).
- **Generalization:** held-out OOD — 7B+verifier 0.353 **beats** Lingshu-32B 0.289 (generalizes + a cross-size
  win); works cross-family (MedVLThinker) too. **Full-set result: the cascade beats the 32B across all datasets,
  both families, in-dist + held-out OOD.** Variant check: agreement-on-picks *loses* to verifier-confidence
  (over-escalates).

### June 30 — cost frontier, the exhaustive gate bake-off, 3rd architecture, verifier headroom
- **Cost frontier + gate honesty:** verifier-bo2 beats weak→strong on accuracy AND FLOPs; on a **held-out τ**,
  the gate is redundant at N=8 — **selection is the real lever**. Instrumented `run_openvqa` (logprobs/timing).
- **Gate bake-off (the verdict):** **verifier-confidence IS the best gate.** Trained gates (CASP/CCPS,
  recoverability-targeted) don't beat it across **3 families, both regimes, the full feature set**; the
  **recoverability wall** (Jitkrittum, ~0.6 AUROC) is the binding limit. The faithful CASP/CCPS test on Lingshu:
  input-perturbation visual-stability is a weak gate (0.60) and adds **nothing** to verifier-conf (0.852→0.853).
- **3rd architecture:** InternVL3 3-family cascade — the cross-architecture verifier beats the 38B + held-out OOD;
  feature-complete gate bake-off = 3rd confirmation that verifier-conf is best; measured batch-1 latency/energy.
- **3-family open-text master table** (`OPENTEXT_MASTER_TABLE.md`): the method beats the strong model in **all 12
  family×dataset cells + 3 held-out OOD**.
- **Verifier headroom diagnosed:** per-answer AUROC is good (0.90) but **selection efficiency is only 74–82%** =
  the headroom. Added a ranking/contrastive verifier trainer (pointwise BCE + within-question Bradley–Terry on the
  Yes−No margin, gradient checkpointing). Ranking lifts per-answer AUROC 0.90→0.93 but **not** selection — the
  ceiling is real (compound free-text grounding difficulty), so the higher-EV levers are a larger/structured
  verifier or **better candidates**, not more cheap-verifier training.

**Standing conclusion end of June 30:** open-text cascade beats the strong model (3 families, in-dist + OOD);
the gate is settled (verifier-confidence, unbeatable); the binding limit is candidate quality / verifier ceiling.
Next: reproduce the **faithful Lingshu MCQ baseline** (MedEvalKit) to anchor the MCQ half.
