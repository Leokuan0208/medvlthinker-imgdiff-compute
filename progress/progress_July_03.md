# Progress — July 2 (evening) – July 3, 2026

> Continues `progress_July_01-02.md` (which ended at the full-matrix launch). This entry covers the
> matrix RESULTS + the day's engineering. All numbers from real eval output.

## The full 3-family × 7-benchmark matrix — results landing

**Phase 1 — 2-tier cascade (cheap→strong, margin gate, iso-strong), all 3 architectures.** The
efficiency result generalizes cross-architecture (InternVL3 is non-Qwen): the cascade matches the
strong model at large FLOPs savings wherever the small model is competitive.

| Benchmark | Lingshu | MedVLThinker | InternVL3 |
|---|---|---|---|
| MMMU-Med | keep-cheap (7B anomaly) | −14% | **−62%** |
| PMC-VQA (33k) | **−69%** | −49% | −16% |
| SLAKE | −56% | no win (7B weak) | keep-cheap (8B≥38B) |
| VQA-RAD | −17% | −41% | **−67%** |
| PathVQA | −31% | **−68%** | −20% |
| MedXpert-MM | no win (floor) | no win | *(cap gap, fixup)* |

**Phase 2 — 3-tier think tier (strong-leg reasoning) on the reasoning benchmarks.** Reasoning helps
on MMMU across ALL THREE architectures; MVT (RL-reasoning-trained) gains most and also gains on MedXpert:

| Strong leg | MMMU-Med Δ | MedXpert Δ |
|---|---|---|
| Lingshu-32B | +0.034 (0.633→0.667) | ~0 (floor) |
| MedVLThinker-32B | **+0.107** (0.613→0.720) | +0.045 (0.299→0.344) |
| InternVL3-38B | **+0.120** (0.633→0.753) | *(cap gap, fixup)* |

Regime-adaptive: think where there is headroom (MMMU), not at the floor (MedXpert). This is the
faithful, cross-architecture answer to "why not 3 tiers" — the think tier is warranted on the
reasoning benchmarks.

**Reasoning correction confirmed:** all three families reason on a *generic* "reason step by step"
prompt (gen_toks 3→275/561/368); MVT does NOT need its native `<think>` tag to reason (emitted 0
`<think>` tags yet still +0.107). So the earlier "Lingshu has no think mode" claim is fully retired.

## Engineering (the day's fixes)

- **InternVL3-38B unblocked** (KV-cache `MAX_MODEL_LEN` lever). New failure found: **MedXpert prompts
  are ~20k tokens > the 16384 cap** → both direct + reasoning MedXpert fail; scheduled for re-run at
  cap 24000 in the post-matrix fixup.
- **OmniMedVQA parser bug** — `cal_metrics` assumed every sample has `modality_type`; some of the 42
  sub-datasets omit it, so all cheap legs crashed *after ~3.5h of generation* (`KeyError`). Fixed with
  `.get(..., "unknown")`. Verified: OmniMed Open-access = **88,996** QA (RadImageNet = 57k = 64%),
  evaluation-only; **Lingshu's paper uses the full set** (7B 82.9% / 32B 83.4%), so full (not a
  sample) is the faithful choice.
- **Latency-comparability audit** (multi-agent) — the cascade `latency_s` IS a fair cheap-vs-strong
  relative number (identical batch size + tp within family, serialized), NOT to be dismissed as
  "not batch-1". BUT the InternVL3 family + all PathVQA rows were measured under GPU contention →
  their latency is unciteable until re-run serial/matched (accuracy/FLOPs unaffected). Clean re-runs
  queued. Fixed the honest label + the `run_regen_native.sh` doc-path bug.
- **Code review** (xhigh, 5 finder angles) — 15 findings; 3 safe fixes applied to non-running code
  (`cascade_all_families.py` hardening, `run_regen_native.sh`, `open_measure_latency_energy.py`
  UUID guard). Live-matrix / MedEvalKit items deferred.
- **Tooling:** adopted **graphify** (76k-star code→knowledge-graph skill) for token-efficient repo
  navigation — ran code-only/local on `src/` (0 egress, 0 cost); `graphify query` traverses the graph
  to pinpoint code instead of reading whole files.

## Status
6/7 of Lingshu's suite complete across 3 families; **OmniMedVQA (full 89k) running** with the fixed
parser (MVT-32B confirming it at scale). Post-matrix fixup queued: IV3-38B MedXpert (direct+reasoning,
cap 24000) + clean-latency re-runs (iv3_8b, 4 PathVQA legs) + the 4 failed OmniMed legs.
