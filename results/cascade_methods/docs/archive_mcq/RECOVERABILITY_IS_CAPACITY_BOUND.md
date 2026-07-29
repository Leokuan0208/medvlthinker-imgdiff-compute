# The recoverability wall is CAPACITY-bound, not action-bound

> New-method-loop, 2026-06-25 (Direction 2, after abstention was dropped at user request).
> A negative result that strengthens the meta-finding. Fully offline from existing checkpoints
> (no GPU). All numbers from real checkpoint output (standing no-fabrication rule).
> Code: `tmp/repair_decomp.py`, `tmp/repair_complement.py`, `tmp/ladder_sim.py`, `tmp/maxconf_ensemble.py`
> (diagnostics; promote to `src/cascade_methods/` only if this gets written up).

## Question
Every method in this loop optimized the GATE (when to escalate) — and confidence is unbeatable
there. The untouched axis is the ACTION (what you do on escalation). The cascade-gain ceiling is the
recoverability wall: the 32B fixes only ~42% of the cheap 7B's errors. Is that wall a property of the
ACTION SET (fixable by adding cheap repair actions) or of model CAPACITY (only a bigger model helps)?

## Setup
Cheap 7B@cap320 error set, competent-4 (PMC-VQA/SLAKE/VQA-RAD/PathVQA), n=6050. Three repair actions:
- **LOOK-CLOSER** = same 7B, no-think, FULL resolution (more pixels on the same image)
- **THINK** = same 7B, think mode (reasoning trace)
- **SCALE** = 32B, think (the deployed escalation)

## Result 1 — repairs ARE complementary (the encouraging part)
Of the 2286 cap320 errors:
| recovered by | fraction |
|---|---|
| SCALE (32B) | 41.6% |
| cheap 7B repair (closer OR think) | 33.7% |
| **multi-action union** | **56.0%** |
| **cheap 7B catches what 32B MISSES** | **14.3%** (328 samples; per-bench 11–17%, stable) |
| unrecoverable by anything | 44.0% |

So the wall is NOT a fixed 42% — a richer action set raises the oracle ceiling to 56%, and cheap
same-model transforms recover a *disjoint* 14% the 5×-larger model cannot. Scaling up is **not a
superset** of intervening cheaply. (Novel, robust observation about medical-VLM error structure.)

## Result 2 — but the complementarity is UNHARVESTABLE (three independent kills)
The union is an oracle. Can any cheap rule reach it? No — the repairs are NOISE-LIMITED: they fix
some errors and break a similar number, so net accuracy is flat (per-view acc: cap320 0.622,
fullres 0.621, think 0.607; only the 32B lifts it to 0.645).

1. **Ladder cascade** (cap320→closer→think→32B, each rung confidence-gated). Even with optimistic
   in-sample thresholds, at 32B-parity it costs **43% vs the 2-rung baseline's 39%** — compounding
   rung-cost beats the savings. It only wins in a narrow sub-parity regime (17.7% vs 27.5% @ acc 0.635).
2. **Max-confidence selection** across {5 resolutions + think}: acc **0.608–0.622** — does not exceed
   the best single view, nowhere near 0.645.
3. **Majority vote** across the same views: acc **0.624–0.626** — same saturation.

Adding the THINK view (a different error profile from resolution) does NOT help — confidence is
*overconfident-and-wrong* on enough think outputs to cancel the gains.

## Conclusion (a real contribution, stated honestly)
**The 32B's advantage is capacity-bound.** Its error-fixing power cannot be replicated OR approximated
by cheap same-model input/mode transforms, and the confidence signal — unbeatable for *detection* —
cannot separate a good cheap repair from a bad one. This (a) explains *why the margin gate is
saturated* (there is no cheaper action that substitutes for capacity), and (b) closes the action-side
of the cascade design space for MCQ: neither the gate nor the action admits a training-free win beyond
what ACC + Visual-Stability Rescue already capture.

Triangulates the loop's meta-finding: **confidence is unbeatable as a signal, and on MCQ the levers
are exhausted — the remaining headroom is in the OPEN-ENDED regime (§5.7), not in MCQ cascade design.**
