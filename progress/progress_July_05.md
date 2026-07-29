# Progress — July 5, 2026

> Continues `progress_July_04.md` (which set up the OmniMed reproduction pipeline as a chained,
> self-supervising GPU job and re-pointed the project at ONE unified method with efficiency as a
> first-class axis). This entry covers the day the OmniMed pipeline's **results landed** — the cheap
> (7B/8B) legs reproduced the paper, the strong (32B/38B) leg hit a deterministic infrastructure wall —
> and the offline pivot that used the GPU-blocked time to build a **cross-field method-idea backlog**.
> A quiet, infra-heavy day: the only compute that ran was the July-4 sequential cheap driver (which
> finished ~00:43) and the doomed strong driver (which hung all day). Every number below is read off a
> log/artifact named inline.

## 1. OmniMedVQA cheap legs — faithful reproduction (the 7th benchmark closes on the cheap side)

The July-4 sequential cheap driver (`runners/run_omnimed_cheap_seq.sh`, tp=1, one leg at a time to dodge
the two-at-a-time cgroup OOM) finished overnight — the last leg's `metrics.json` materialised at
**2026-07-05 00:43** (`logs/omnimed_cheap.log`). All three cheap legs ran the **full 88,996-QA**
OmniMedVQA Open-access set (not a sample — Lingshu's paper uses the full set, so full is the faithful
choice) and reproduce the paper:

| cheap leg | OmniMed acc (n=88,996) | right | paper (Lingshu-7B) | verdict |
|---|---:|---:|---:|---|
| **Lingshu-7B** | **0.8274** | 73,639 | 0.829 | **matches to 0.2 pt** |
| MedVLThinker-7B | 0.6248 | 55,602 | — | family-consistent |
| InternVL3-8B | 0.7847 | 69,832 | — | family-consistent |

**Why this matters.** Lingshu-7B at **0.827 vs paper 0.829** is a 0.2-point reproduction — it validates
that our OmniMedVQA eval pipeline (post the July-3 `modality_type` parser fix) is **faithful**, closing
the fidelity question for the 7th and last benchmark of Lingshu's suite. The per-question-type and
per-modality breakdowns from the log are internally sensible (Lingshu-7B: Modality-Recognition **0.986**,
Fundus-Photography **0.889**, OCT **0.898**, X-Ray **0.862**, Disease-Diagnosis 0.801, weakest on
CT **0.772**) — i.e. the model is near-ceiling on the easy "what modality is this?" bucket and drops on
fine-grained diagnosis, exactly the expected shape. MedVLThinker-7B's low 0.625 is concentrated in
Anatomy-Identification (0.436) and ultrasound (0.337), a real family weakness, not a harness bug (its
Modality-Recognition is still 0.973).

**Learned.** OmniMed is now cheap-faithful across all three families. The remaining question is only the
*strong* leg, which is where the day went sideways.

## 2. OmniMedVQA strong (32B/38B) leg — the deterministic tp=2 NCCL hang, and the keep-cheap fallback

The chained strong driver (`runners/run_omnimed_strong_chunked.sh`, the 6-shard resumable tp=2 driver
built July 4) fired once the cheap sentinel (`OMNIMED_CHEAP_DONE`) landed, and immediately ran into a
**deterministic NCCL collective hang** at tensor-parallel=2. `logs/omnimed_strong.log` shows the same
failure mode recurring all day: every chunk attempt stalls, the NCCL heartbeat watchdog aborts the stuck
collective after ~36 min, the driver re-fires on a freshly-cleaned GPU, and it hangs again.

Everything that could be tried within the infra was tried and ruled out (these span July 4 evening →
July 6 midday, ~2 days of attempts total):

- **Chunked tp=2 + retry ×3** — the hang is not a transient; it recurs on every chunk.
- **`TORCH_NCCL_ENABLE_MONITORING=0`** (disable the heartbeat monitor) — removes the auto-abort, so the
  run just hangs *forever* instead of hanging-then-retrying. Worse, not better.
- **`EVAL_BATCH_SIZE=256`** — fixes the *container cgroup OOM* (the separate July-4 failure mode) but does
  **not** touch the collective hang.
- **3 h/chunk timeout + auto-recovery + aggregation backstop** — the backstop re-fires the last chunk if
  `metrics.json` didn't materialise, but the chunk re-hangs, so the run never completes.

**tp=1 is not an option** and this was re-confirmed: a 32B is **64 GB of weights + multimodal activation**,
which OOMs the 80 GB card; the `MAX_MODEL_LEN` KV-cache lever alone cannot buy that back. So the 32B
*needs* tp=2, and tp=2 is exactly what hangs on the 89k-image OmniMed workload.

**The keep-cheap fallback decision (converged today, formally written the next morning as
`results/cascade_methods/docs/current/OMNIMED_FALLBACK.md`).** Why it is safe, not a cop-out:

- **OmniMed is a keep-cheap benchmark.** In the paper the cheap and strong models are essentially tied —
  Lingshu-7B **82.9** vs 32B **83.4**, a **0.5-point** gap. Our cheap 0.827 *already* matches the paper's
  cheap number, so a cascade **keeps-cheap on OmniMed** (near-zero escalation) and the missing strong
  number changes **no cascade conclusion**.
- Therefore: report the strong OmniMed cell as **paper-reference (Lingshu-32B 0.834) + infra-limited**,
  back the fidelity claim with our faithful cheap reproduction, and — per the standing rule —
  **write NO fabricated `metrics.json`** to the results dir.
- Net reproduction status: **6/7 benchmarks** (MMMU, VQA-RAD, SLAKE, PathVQA, PMC-VQA, MedXpert) fully run
  cheap + strong (+ think tier) across all 3 families, faithful vs paper; **OmniMed (7th) = cheap-faithful
  + strong-fallback.**

**Open question.** The tp=2 hang is almost certainly OmniMed-scale-specific (the 6 smaller benchmarks ran
tp=2 fine) — likely a driver/NCCL interaction on the very long 89k-image queue, not a code bug. Not worth
more GPU-days for a keep-cheap cell; parked as infra-limited. The immediate payoff of parking it: the GPUs
are freed for the higher-value UGV / diverse-generation / Pandora research that runs July 6.

## 3. Cross-field method-idea research — building the idea backlog (the offline pivot)

With the GPU tied up all day on the doomed strong OmniMed leg, the productive move was to go fully offline
and **widen the method search**. The July-4 investigation had converged the project onto one unified
test-time-compute method (cheap 7B/8B medical VLM + trained best-of-N **outcome verifier** + confidence
gate, cascading to a strong 32B/38B, scored on BOTH accuracy AND latency/compute/energy) but the honest
reading of the gate bake-off was that we had exhausted the obvious *gate* levers. So the day's offline work
was a **systematic cross-field sweep**: read mechanisms from related and (mostly) unrelated fields —
economics of information, portfolio theory, crowdsourcing truth-inference, coding theory, social choice,
sequential analysis, pure-exploration bandits, computer architecture — and map each onto a concrete,
testable variation for our cascade/verifier.

Every idea is judged against the project's **binding limits** (the framing that organizes the whole
backlog):

1. **Candidate quality / oracle@N is the wall** — oracle@8 ≫ verifier-bo8 everywhere (Lingshu pooled
   0.513 vs 0.414); a cross-model pool raises oracle **+0.11–0.15**. This is where the accuracy headroom
   lives.
2. **Verifier selection efficiency ≈ 74–82 %** — per-answer AUROC 0.90–0.93 but it loses within-question
   near-ties.
3. **The recoverability wall** (Jitkrittum 2307.02764) — the strong model fixes only 6–26 % of cheap
   errors and *which* is near-unlearnable (AUROC ≈ 0.6); the gate is already near-optimal.
4. **Cost tension** — best-of-N base cost = 2N cheap forwards ⇒ FLOP break-even vs one 32B forward is
   **N ≤ 2**; at batch-1 the 32B is only ~1.9× the 7B latency (bandwidth-bound), so latency ≠ FLOPs.

The backlog (`results/cascade_methods/METHOD_IDEAS_BACKLOG.md`) uses a fixed entry schema — name · source
field + key paper · mechanism · map-to-our-cascade · concrete testable variation · expected ACC / expected
LATENCY-COMPUTE effect · how-to-test (OFFLINE on dumps vs a GPU job) · novelty & risk — with the rule
**append + re-rank, do not overwrite**, and **no fabricated experimental numbers** (project numbers cited
are real; every "expected effect" is flagged as a HYPOTHESIS). The initial population organizes ideas into
four families: **A. candidate-quality / generation levers**, **B. verifier / selection levers**,
**C. gate / stopping / compute-allocation controllers**, **D. structural / cascade-architecture levers**.

The **first-pass TOP-5-to-test** (both-axes potential × novelty × testability) that this seeds tomorrow's
work:

1. **Pandora's-Box adaptive controller** (C1, Weitzman reservation values) — one optimal rule unifying
   adaptive-N *and* the escalation gate.
2. **Diversity-maximized candidate set** (A1, DPP/MMR) — attacks limit #1 (raise oracle@N by covering the
   answer space instead of redundant iid samples).
3. **Generator portfolio from the error-correlation matrix** (A2, Markowitz) — attacks limit #1 on dumps
   we already have.
4. **Speculative cascade — 32B as a verifier** (D1, Narasimhan 2405.19261) — strongest both-axes
   structural lever; resolves the cost tension.
5. **Pairwise / knockout-tournament verifier** (B1, PairJudge RM) — attacks the selection ceiling
   (limit #2).

**Status at end of July 5.** The reproduction is settled (6/7 faithful + OmniMed cheap-faithful +
strong-fallback). The idea backlog is seeded and ranked. The GPUs are free. Tomorrow (July 6) is the
execution marathon — most of the top-5 plus a second pass of cross-field ideas get *tested*, offline and
on GPU. No experimental result was produced today beyond the OmniMed cheap reproduction; the rest is
engineering, an infra post-mortem, and literature.
