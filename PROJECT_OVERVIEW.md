# Medical-VLM Efficiency & Accuracy — Project Overview (a 30-minute read)

*A plain-language tour of the whole project: what we set out to do, what we found, and where it stands.
Every number here is real (from `results/cascade_methods/GROUND_TRUTH_NUMBERS.md`). Section pointers tell you
where to read more.*

---

## 1. What this project is (in one paragraph)

A **medical vision–language model (VLM)** takes a medical **image** (a chest X-ray, a pathology slide) plus a
**question** and returns an **answer**. The best models are large "reasoning" models that think step by step —
**accurate but slow and power-hungry** (a 32-billion-parameter model takes ~11 seconds and ~6 kilojoules per
question). A small 7-billion model answers in ~0.2 seconds but is less accurate. **Our goal is efficiency:
match the big model's accuracy for far less compute** — and, separately, to *improve* accuracy cheaply. The
deliverable is a paper for **CVGIP 2026**.

## 2. Key terms you'll see (the only vocabulary you need)

| term | plain meaning |
|---|---|
| **Cascade** | Run the cheap model first; only call the expensive model on hard questions. |
| **Gate** | The rule that decides "is this question hard enough to escalate?" |
| **Confidence / margin** | How sure the model is (gap between its top-2 answer probabilities). |
| **Think vs no-think** | The big model can either emit a long reasoning trace ("think", slow) or answer directly ("no-think", fast). |
| **Oracle gap** | If you sample the model several times, how much better you'd do *if* you could always pick its best try. |
| **Luck floor** | Our key negative: with no training, you *can't* pick the best try better than random. The oracle gap is real but unreachable. |
| **Verifier** | A small trained helper that scores "is this answer correct?" and picks the best of several tries. |
| **Best-of-N** | Sample N answers, keep the one the verifier likes most. |
| **FLOPs / latency / energy** | Three ways to measure cost (math operations / wall-clock seconds / joules). |
| **AUROC** | How well a score separates right from wrong answers (0.5 = useless, 1.0 = perfect). |

## 3. The one-sentence story

> **You cannot out-engineer the cascade's *decision* with clever untrained tricks (they all hit the same
> wall), but two things give real gains: better *structure* (our ACC cascade — big efficiency win) and a
> little *training* (our verifier — big accuracy win).**

Everything below is that sentence, expanded.

---

## 4. Result 1 — ACC: same accuracy, a fraction of the cost (the efficiency win)

**The idea.** The big 32B model has a *fast* mode (no-think) that, surprisingly, is **as good as or better
than** its slow think mode on perception questions — *thinking over-thinks them*. So we build a **three-tier
cascade** out of *compute configurations of the same two models*:

```
cheap 7B (fast)  →  big 32B (fast / no-think)  →  big 32B (slow / think)
```

The slow think pass — the only expensive part — fires **only when the two fast legs disagree** (~15% of
questions). Everything else is answered fast.

**The payoff (at equal accuracy, 6-benchmark average):**

| | always use big-think | **ACC (ours)** | change |
|---|---|---|---|
| accuracy | 0.572 | 0.569 | parity |
| latency | 11.34 s | **2.27 s** | **−80%** |
| energy | 6,319 J | **1,182 J** | **~5× less** |
| compute (FLOPs) | 100% | **52%** | **halved** |

On the 5-benchmark set the latency drops from 8.9 s to **0.44 s** (−95%). And it's **never worse than the
cheap 7B on any benchmark** (a safety guarantee). See Figure below; detail in **paper §5.1**,
math in `results/cascade_methods/METHOD_MATH.md`.

`figs/fig1_latency_accuracy_frontier.png`
`figs/fig2_overthinking_perbench.png`

**Honest note.** The *agreement* rule we use to decide when to think is not new (it's shared with prior
"agreement-based cascading"). What's new is the *structure* — using the big model's fast mode as a middle
tier. The paper's thesis: the *structure* (the fast middle tier), not the gate, is the win.

---

## 5. The wall — why a "smarter gate" doesn't exist (the negative that motivates everything)

We tried, exhaustively, to build a *better decision rule* with no extra training. **Twelve different signals**
(confidence, entropy, agreement, self-checking, conformal prediction, learned routers, …) **all hit the same
ceiling** (AUROC ≈ 0.5–0.69). The reason: the cheap and big models **fail on the same questions** (correlation
φ = 0.37) — 58% of the cheap model's mistakes the big model also gets wrong. You can tell *that* the cheap
model is unsure, but not *whether the big model would fix it*.

The same wall appears for **selection**: sample the model 8 times and a correct answer is often in there
(e.g., one task: 73% → 88% if you could always pick the right try), **but no untrained rule beats picking at
random** — the **luck floor**. It even holds for bounding boxes. Detail in **paper §5.2** + writeups
`OPENENDED_SELECTION_LUCKFLOOR.md`, `RECOVERABILITY_IS_CAPACITY_BOUND.md`.

**A useful twist (paper §5.3):** this wall looks worse than it is on *multiple-choice* tests, because a
single A/B/C/D letter is too coarse a signal. On *open-ended* (free-text) answers the same confidence signal
jumps from AUROC ~0.6 to **~0.87**. So medical-VLM cascades should be tested open-ended — which is exactly
where the verifier (next) works.

---

## 6. Result 2 — the trained verifier: breaking the luck floor (the accuracy win)

**The idea.** The luck floor is a property of *frozen* models. If we **train** a small helper to score
"is this candidate answer correct?", it can pick the best of N tries — and it works.

**For free-text answers**, the trained verifier recovers **35–49%** of the oracle gap (two seeds; seed-0 table below averaged over 4 datasets,
and it transfers to a 5th it never saw):

| dataset | cheap model | **+ trained verifier** | best possible (oracle) |
|---|---|---|---|
| PathVQA | 0.352 | **0.441** | 0.513 |
| Kvasir | 0.282 | **0.405** | 0.493 |
| VQA-RAD | 0.519 | **0.611** | 0.722 |
| SLAKE | 0.738 | **0.762** | 0.895 |
| **average** | 0.413 | **0.501** | 0.592 |

**For bounding boxes** (pointing to the right place in the image), the *same idea* works — including on a
**real chest-X-ray benchmark (MS-CXR)**:

| task | cheap | **+ trained verifier** | gap captured |
|---|---|---|---|
| SLAKE organs | 0.197 | **0.255** | 40% |
| **MS-CXR chest X-ray** | 0.041 | **0.232** | **78%** (a 5.6× lift) |

**Why we trust it:**
- It **genuinely tells right from wrong** (AUROC **0.924** at scoring individual answers — see figure), and it
  **uses the image** (blanking the image hurts it).
- It **beats** an *untrained* version, and **matches** the **5× bigger 32B model's** answer (a significant win on one split, a tie on another).
- It's **statistically significant** (bootstrap confidence intervals exclude zero on both headline results).
- It's a **test-time-scaling method**: give it more samples, it gets better (random selection does not).
- **Test-time compute beats parameters:** the small 7B *with the verifier* (0.501) **matches** the big 32B's single
  answer (0.462, same questions; a tie on a 2nd seed) — because, for these questions, thinking bigger barely helps but *checking* helps a lot.

`figs/limits/fig_trained_verifier_unified.png`
`figs/limits/fig_verifier_discrimination.png`
`figs/limits/fig_verifier_scaling.png`
`figs/limits/fig_verifier_pareto.png`

Detail in **paper §6** + writeups `TRAINED_VERIFIER_RESULT.md`, `BOX_VERIFIER_RESULT.md`. Is it novel? The
*mechanism* (a trained verifier for best-of-N) exists in text NLP; what's new is the **medical application +
unifying answers and boxes**, as a direct rebuttal to a recent paper ("Verification Mirage") that said
self-checking fails in medical VQA.

---

## 7. Where the project stands

- **Two solid, complementary contributions:** ACC (efficiency: −80% latency, ~5× energy, at parity) and the
  trained verifier (accuracy: 35–78% of the oracle gap across two seeds, answers *and* boxes, a real benchmark).
- **A thorough negative-result map** (the luck floor) that makes both positives meaningful.
- **Paper** restructured around these two results (`paper/cvgip2026_draft.md`), with an **IEEE PDF**
  (`paper/cvgip2026_ieee.pdf`) and this overview.
- **All numbers audited** against the raw result files; inconsistencies found and fixed
  (`INCONSISTENCIES.md`, `GROUND_TRUTH_NUMBERS.md`).

**Where to read more (in order):** this file → `READING_GUIDE.md` (a step-by-step tour) → `paper/cvgip2026_draft.md`
(the full paper) → the `results/cascade_methods/*.md` deep-dives. **Open question / next step:** push the
verifier further (bigger verifier, more datasets, cross-model reuse) to strengthen the novelty.
