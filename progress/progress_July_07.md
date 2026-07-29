# Progress — July 7, 2026 (in progress)

> Continues `progress_July_06.md` (the execution marathon that closed the best-of-N program and wrote the
> ledger). This entry is **in progress** — it records a single conceptual pivot that reframes yesterday's
> "best-of-N is not deployable" verdict, and the investigation it launches. **No experiment has run yet
> today and no new number is produced below**; every figure cited is a *previously-measured* value read off
> yesterday's artifacts or the prior ACC record, used here only to state the hypothesis. Anything about the
> re-grounded comparison is an explicit HYPOTHESIS to be tested, not a result.

## 1. The re-grounding — we have been comparing against the wrong "strong" model

Reading back the July-6 selectability battery (`progress_July_06.md` §5.4/§5.5, artifacts
`end_to_end_consolidation.json` / `latency_reexamination.json`), the entire "best-of-N is Pareto-dominated,
the router is the only deployable lever" conclusion rests on **one load-bearing quantity**: the cost of the
strong open-text leg, measured as **GEN32 = 665 ms / F = 4.57 FLOP-eq**. That number is the 32B's
**NO-THINK** generate. Both §5.4 and §5.5 conclude "always-32B is cheap AND fast, so escalating to it
dominates spending budget on cheap draws" — but that is only true against the *no-think* 32B.

The problem: **the paper's motivating deployment-cost baseline is the 32B-THINK model, not the no-think
one.** From the project's own prior measurements (CLAUDE.md §0; `progress_July_04.md` §1; the ACC record):

- 32B **no-think** batch-1 ≈ **0.34 s** (the fast mode; ≈665 ms for the open-text *generate* leg measured
  yesterday, which includes a longer decode);
- 32B **think** batch-1 ≈ **11 s / ≈6 kJ** — this is the ~30× slower reasoning pass that the ACC was
  *built to avoid* (ACC latency: ALL-6 **11.34 s → 2.27 s**, ALL-5 **8.88 s → 0.44 s**).

So yesterday's battery quietly benchmarked the cheap best-of-N leg against the **cheapest, fastest** version
of the strong model. Against *that* strawman, best-of-N loses on FLOPs and ties on latency — unsurprising.
But the deliverable's whole premise ("a 32B reasoning model costs ≈11 s / ≈6 kJ per question") is about the
**think** model. **The honest comparison for the efficiency deliverable is against the strong model people
actually pay for when they want its accuracy — the think model — not its no-think shortcut.**

This is not a new caveat invented today; it is exactly the scope caveat the July-6 latency artifact wrote
itself: *"conditioned on a cheap, fast strong model; in an expensive or slow-strong regime the parallel
best-of-N leg could still win — that regime is not measured here."* Today's move is to stop treating that as
a footnote and **measure it.**

## 2. Why the re-grounding could flip the July-6 verdict (hypotheses, not results)

Restating the two July-6 negatives against the *think* baseline turns each into an open question:

- **FLOPs (§5.4).** The envelope `greedy → 7B+Pandora → always-32B` was drawn with always-32B at F=4.57
  (no-think). A think forward emits a long `<think>` trace, so its FLOP-eq is **much higher** than 4.57.
  *Hypothesis:* re-pricing the strong leg at its think cost moves it far to the right on the FLOPs axis, so
  the cheap best-of-N points (iid-bo8 F=16, diverse-bo15 F=30) may no longer be Pareto-dominated at the
  accuracy targets where the *think* model is the only way to reach them. **To measure:** the real gen-token
  count / FLOP-eq of the 32B-think open-text leg (we have think dumps on the reasoning benchmarks; need the
  open-text think cost).
- **Latency (§5.5).** The parallel best-of-N base is **522.6 ms** (GEN7 347.1 + VER7 175.5). Against the
  665 ms no-think forward it was 0.79× — "latency-alive but doesn't beat." Against an **~11 s think**
  forward it is **~0.05×**. *Hypothesis:* on the latency axis the best-of-N leg does not merely survive — it
  **dominates** the think model by ~20×, and even a heavily-escalating gated cascade (which pushed parallel
  latency to ~1.14–1.19 s yesterday) stays an order of magnitude under 11 s. **This is the regime the ACC
  was designed for**, and it is where a cheap parallel leg should look best.

If both hold, the July-6 "best-of-N is not deployable" conclusion is **scoped to the no-think baseline
only**, and the deployable story against the *think* baseline is materially different — potentially
best-of-N (or a best-of-N tier inside the cascade) is back on the table on latency, and possibly on FLOPs.

## 3. Revisiting the 3-tier structure (ACC) under the corrected baseline

The re-grounding points straight back at the **Adaptive-Compute Cascade** — the project's genuine
structural win — whose whole point was to keep the slow think pass off the critical path:

```
7B-nothink@cap320  →  32B-NO-THINK@cap320  →  32B-think@fullres
   (bulk)              (fast intermediate tier)     (reasoning residual only)
```

The insight the cascade encodes (thinking *overthinks* perception VQA; 32B-no-think ≥ 32B-think on the
competent-4 sets) means the **no-think 32B is the right *tier*, but the *baseline* to beat is the
think 32B.** Yesterday's battery collapsed those two roles into one number (665 ms), which is why it read
best-of-N as pointless. Under the corrected framing there are really **three cost regimes** to hold
separate, and the deployable method should be scored against the most expensive one it replaces:

1. **Perception VQA (competent-4):** 32B-no-think is both the tier and ≈ the accuracy ceiling → the router /
   ACC keeps-cheap or escalates to no-think; best-of-N is genuinely dominated (yesterday's verdict stands
   *here*).
2. **Open-text OOD (the §5 sets):** the cheap best-of-8 ensemble *beats* the no-think 32B on accuracy
   (bo8 0.414 vs 0.331) → the target is the ensemble ceiling, and the strong leg's *think* cost is the
   thing worth saving.
3. **Reasoning (MMMU / MedXpert):** only the **think** model reaches the accuracy → here the ~11 s / ~6 kJ
   think cost is unavoidable for the residual, and the whole value proposition is firing it on as *few*
   questions as possible — which is what the 3-tier gate does.

**Open question (the crux of the investigation):** which of the July-6 conclusions survive when each method
is scored against the *think* baseline in the regime where think is actually required? Specifically —
(a) does re-priced-strong FLOPs put diverse-bo15 or iid-bo8 back on the Pareto envelope anywhere;
(b) on latency, is the right deliverable "cascade whose expensive tier is think, fired on the small residual"
rather than "always-think", and what is the honest saved-latency/energy number against always-think (not
always-no-think); (c) does the ACC's 3-tier structure, re-measured with the open-text + reasoning regimes
included (not just the competent-4 MCQ sets it was validated on), still hold its guardrails?

## 4. Investigation launched — plan and standing state

Launched today; **no GPU/eval job has completed and no artifact has been written yet.** The plan:

1. **Establish the correct strong-baseline costs.** Get the measured batch-1 latency, energy, and FLOP-eq of
   the **32B-think open-text generate** leg (we have think latency on the reasoning benchmarks; the open-text
   think cost is the missing cell). This replaces the 665 ms / F=4.57 no-think number as the baseline in a
   re-run of `end_to_end_consolidation.py` and `latency_reexamination.py`.
2. **Re-score the July-6 envelopes against the think baseline**, per-regime (perception / open-text OOD /
   reasoning), keeping the no-think 32B as an *intermediate tier*, not the baseline. Report which §5.4/§5.5
   conclusions are baseline-scoped vs robust.
3. **Re-validate the 3-tier ACC** on the full Lingshu suite (open-text + reasoning included), against
   always-think, with per-benchmark guardrails — the honest "saved vs the model you'd otherwise deploy"
   number.

**Standing state.** The July-6 ledger conclusions stand *as written* (they are correctly scoped to the
no-think baseline, and the artifacts flagged that scope). What changes today is the **question**: the
deployable-efficiency claim must be stated against the **think** model — the expensive baseline the paper is
actually about — and the re-grounded comparison is the next experiment, not yet run. Nothing was committed or
measured today; this entry records the pivot so the reasoning is not lost.

---

> **UPDATE (later on 2026-07-07) — the §4 plan has now been EXECUTED.** Everything below supersedes the
> "no experiment has run yet today" framing of §1. Two small GPU cost-measurements were taken (the two cells
> §5.5 of the ledger flagged as blocked — the 32B-think open-text latency and the IV3-38B latency); everything
> else is CPU re-costing of existing dumps. Every number below is sourced to an artifact under
> `results/cascade_methods/artifacts/`. The day's arc: **reframe → baseline-cost measurements → the integrated
> method win → two research passes (§F, §G) → three pushes.**

## 5. Establishing the correct strong-baseline costs (the two blocked cells, now measured)

The reframe needs the cost of the *think* model, not its no-think shortcut. Both missing cells were measured
today (batch-1, NVML energy):

- **M1 — 32B-THINK open-text latency/energy** (`opentext_32b_think.json`, HF batch-1, single GPU, cap320 real
  VQA-RAD images, n=15 after 3 warmups, native `<think>` prompt): **10 521.6 ms mean** / 12 896.2 ms median,
  **2 001.9 J**, gen_tok 98.3, prefill_tok 335.8. No-think reference on the same harness: **665.0 ms / 126.9 J /
  gen_tok 5.6**. ⇒ **think : no-think = 15.8× latency, 15.8× energy.** And think is *less accurate* on open-text
  (vLLM greedy, n=200 paired/set, modal scorer): SLAKE-open **0.700 vs 0.895 (−0.195)**, VQA-RAD-open
  **0.425 vs 0.545 (−0.120)**, PathVQA-open **0.035 vs 0.170 (−0.135)**; pooled n=600 **0.387 vs 0.537 (−0.150)**.
  → the headline figure **"open-text think 10.5 s / −0.15"**. This is the number that priced the naive baseline and
  supplied the `THINK_DELTA_OPEN` used to estimate judged 32B-think open-text accuracy downstream.
- **M3 — IV3-38B batch-1 latency/energy** (`iv3_38b_latency.json`, vLLM tp=2, batch-1, real VQA-RAD images,
  peak VRAM 153.9 GB, n=10/mode): **no-think 1 409.3 ms / 598.0 J / gen_tok 2.0**; **think 6 220.0 ms / 3 275.6 J /
  gen_tok 173.2** → **4.4× latency, 5.5× energy**. → the figure **"IV3 6.2 s"**. This removes the router's
  amortized-batch-latency proxy caveat for the 38B peer strong leg.

These two runs turn ledger §5.5's "blocked / not computable" cells into measured constants. (OmniMed-32B stays
blocked — deterministic tp=2 NCCL hang — so ALL-7 remains unreportable; ALL-6 is the computable pool.)

## 6. The reframe result — method vs always-32B-THINK per family (`reframe_vs_bigthink.json`)

With the think baseline priced, the ACC 5-family MCQ bake-off (`master_data.csv`, measured batch-1) was re-scored
against **always-32B-THINK**. **Always-32B-THINK is Pareto-dominated on every family** — the method matches-or-beats
its accuracy at **9–68 % of its FLOPs**, **8–99 % lower latency**, **33–99.7 % lower energy**:

| family | pool | 32B-think acc | method acc | Δacc | 32B-think lat | method lat | Δlat | FLOPs% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MVT 7B→32B | ALL-6 | 0.5723 | 0.5693 | −0.003 | 11.34 s | 2.27 s | −80 % | 52 % |
| Lingshu 7B→32B | ALL-6 | 0.6611 | 0.6614 | **+0.000 (beats)** | 0.32 s | 0.30 s | −8 % | 49 % |
| QoQ-Med | ALL-6 | 0.4689 | 0.5095 | **+0.041 (beats)** | 9.72 s | 0.12 s | −99 % | 9 % |
| Chiron | ALL-6 | 0.5076 | 0.6023 | **+0.095 (beats)** | 4.25 s | 0.20 s | −95 % | 19 % |
| MedGemma | ALL-6 | 0.5253 | 0.5219 | −0.003 | 12.72 s | 3.37 s | −74 % | 68 % |

On **3 of 5 families the big *thinking* model is actually LESS accurate than the method** (think over-thinks the
perception-dominated suite). think : no-think latency ratio (ALL-6): **MVT 49×, MedGemma 45×, QoQ 43×, Chiron 15×,
Lingshu 1.2×** (Lingshu has no real promptable think mode — its "think" run ≈ no-think, hence the small win).

**Regime split** (why the win has two halves): reasoning (MMMU/MedXpert) — faithful MedEvalKit think gains
**MMMU +0.027/+0.100/+0.120** (Lingshu/MVT/IV3), **MedXpert −0.003/+0.045/+0.031** — think *is* the target and it is
slow, so a **gated** think tier captures it (MMMU 3-tier matches 0.688 at 78 % FLOPs / ~31 % latency); perception —
think Δ negative-or-flat (SLAKE MVT −0.084, Chiron −0.108; VQA-RAD all −0.07…−0.09), so the method routes to fast
no-think and **think fires ~0 %**. **Load-bearing structural fact** (`METHOD_ACC.md` head-to-head, ALL-6): restoring
the **32B-no-think middle tier** turns an escalate-everything-to-think cascade into the ACC — **M2** (no nt-middle,
7B-think→32B-think) 0.5725 / think-esc 86 % / FLOPs 105 % / **29.8 s** / 7 049 J → **M1 ACC** 0.5694 / **19 %** /
**55 %** / **5.9 s** / 1 505 J (latency **−80 %**, energy −79 %, matched acc); M1b +agreement 0.5710 / 14 % / 54 % /
4.86 s. This is the MCQ arm's engine.

## 7. FALC → the integrated method win (`integrated_method_vs_think.json`)

FALC (`best_method_lingshu.py`) was the format-aware 2-tier cascade scored against always-32B-**no-think** (7B-nt +
margin gate → 32B-nt for MCQ; 7B greedy + verifier/seqlogprob gate for open-text). Two upgrades assembled the
**integrated method** and re-scored it against the harder always-32B-**THINK** baseline: **(a)** the open-text cheap
leg became **7B best-of-8 + trained verifier selection** (the accuracy-winning piece FALC left on the table), and
**(b)** MMMU became **keep-7B** (Lingshu-7B 0.80 > 32B-think 0.66). Held-out 5-fold, batch-1 measured costs:

- **Pooled (full suite, n=42 374): method 0.5750 vs 32B-think 0.5631 (Δ +0.0118) vs 32B-nt 0.5732 (Δ +0.0018);
  esc 15.5 %; latency 459.6 ms vs 10 521.6 ms (−95.6 %); FLOPs 2.538.** Macro-avg 0.6753 vs 0.6063 (Δ +0.069).
- **Per-benchmark Δ vs think:** PMC +0.0014, SLAKE-cl −0.012, VQA-RAD-cl −0.008, PathVQA-cl −0.001, MedXpert −0.004,
  **MMMU +0.140**, **SLAKE-open +0.192**, **VQA-RAD-open +0.105**, **PathVQA-open +0.207**. The accuracy engines are
  the open-text arm (bo8+verifier beats even 32B-no-think) and MMMU keep-7B; perception MCQ ties 32B-nt(≈think).
- **Correction #1 (gate):** on real Lingshu MCQ, **margin is the best gate** — margin AUROC 0.7254 / min-esc 15.62 %;
  CASP-stability is **INERT** (7B is 98.95 % cap320-vs-full stable → collapses to margin); agreement is the worst
  ranker (AUROC 0.657) and needs the 32B (not a cheap gate). Deployability order **margin > agreement > CASP**; the
  premise "CASP/agreement beat the margin gate" is false for Lingshu.
- **Correction #2 (router):** a **router is required** — the MCQ margin gate has no open-text analog and the verifier
  is open-text-specific; a single unified gate (7B seqlogprob) is beaten by margin on MCQ and verifier-conf on open.

Spec written up in full: **`results/cascade_methods/docs/current/METHOD_FINAL_2026-07.md`**.

## 8. Two research passes on the idea backlog (§F beat-32B, §G escalation; 35 → 56 ideas)

The reframe opened two axes the first **35** backlog ideas (all oriented at *matching* the 32B/ensemble ceiling more
cheaply) did not cover:

- **Pass-3 (§F "[BEAT-32B]", +11 → 46 ideas):** can the method *beat* always-32B on accuracy, and honestly *where*
  (broad slices vs the MMMU n=150 anomaly)? Exploits the project's complementarity facts to route/fuse on observable
  slices with calibrated confidence — sidestepping the recoverability wall (Jitkrittum AUROC ≈ 0.6 for "will the 32B
  fix it", but the two legs are comparably skilled with de-correlated errors on PMC). Coded in `beat32b_fusion.py`.
- **Pass-3b (§G "★ESC", +10 → 56 ideas):** the integrated cascade is faster than always-32B-THINK everywhere, but vs
  the *cheaper* always-32B-no-think its speed win is *lost* on three heavy-escalation cells — recover the latency at
  ~0 accuracy cost. Coded in `escalation_levers.py`. **Running backlog total: 35 (§A–E) + 11 (§F) + 10 (§G) = 56.**

## 9. The three pushes (results)

1. **Push 1 — beat-32B FUSION on PMC** (`beat32b_fusion.json`). A held-out-guardrailed slice router picks, per
   benchmark, the certified winner among {always-32B-nt, keep-7B, calibrated **confidence-advantage** fusion}. Only
   **PMC** (fusion) and **MMMU** (keep-7B) certify as non-32B; the radiology/pathology closed sets keep 32B (fusion
   hurts there — guardrail). **PMC fusion (F3 conf-advantage ≡ 2-detector Chair-Varshney): 0.5653 vs 32B-nt 0.5518 =
   +0.0135, 95 % CI [0.0100, 0.0169], n=33 430, held-out** — a genuine broad-slice win, NOT the MMMU anomaly. (Classic
   per-*slice*-reliability C-V collapses to always-32B, d=0.0 — the beat needs *per-sample* confidence. F5
   double-reading: on the 33 % disagreement set the *free* conf-advantage arbiter 0.412 beats both 32B-nt 0.371 and
   the expensive 32B-think 0.387 → think is a poor arbiter.) **Pooled full-suite rises to 0.5869 vs 32B-nt 0.5732
   (+0.0138) / vs 32B-think 0.5631 (+0.0238)** — up from the integrated method's +0.0017 / +0.0118, because PMC is
   ~79 % of samples. **Honest cost:** the fusion cell runs both legs (+22 % FLOPs on PMC → pooled 5.751 = 1.26×
   always-32B); it is an **accuracy lever, not a compute-saver** — the Pareto knob is compute-lean cascade (0.575 @
   FLOPs 2.54) ↔ accuracy-max fusion (0.587 @ FLOPs 5.75).
2. **Push 2 — G8 parallel prefill prefetch** (`escalation_levers.json`). Run the 32B image-prefill concurrently with
   the 7B pass; escalated leg = max(cheap, prefill32)+decode32. Measured φ=0.586 → prefill32=390 ms > cheap 347 ms, so
   the whole 7B pass hides under the prefill on every MCQ escalation. **Zero accuracy change: pooled 461.1 → 405.2 ms
   (−12.1 %)**, and the three slower cells (VQA-RAD-cl 726 ms, MedXpert 943 ms, SLAKE-open 699 ms) all flip under
   always-32B-nt. FLOPs caveat: unconditional prefetch pays the 32B prefill on every query (2.337 → 4.575 ≈
   always-32B) — a latency-for-FLOPs trade, free only on an idle 2nd GPU; slice-gated (esc ≥ 0.40) keeps FLOPs 2.492 at
   429.8 ms. G5 (suppressor) and G6 (2-of-2 gate) are knobs, not free lunches (G5 ε\*=0.06 suppresses MedXpert for
   −0.0018 pooled / 943→347 ms; G6 no gain — CASP inert, no orthogonal 2nd MCQ signal). Combined G8+G5(ε\*): 416.4 ms
   (**−96.0 % vs 32B-think**) at −0.0018 acc.
3. **Push 3 — SLAKE-open best-of-8 verifier fill** (`slake_open_bestofN.json`). The last open-text cell without a
   verifier dump: pooled4 verifier scored pointwise over the K=8 SC candidates (n=645). **bo8+verifier 0.7798** vs
   greedy_t0 0.7302 (**+0.0496**) / SC-modal 0.7364 (+0.0434); oracle@8 0.8791; 32B-nt 0.8186. With the verifier-conf
   gate SLAKE-open reaches 32B-nt parity at **~13 % escalation vs ~53 %** for the old greedy+seqlogprob fallback — a
   **~4× escalation cut** — which is what upgrades the final integrated SLAKE-open cell to bo8+verifier (0.8155 @
   605 ms). Caveat: SLAKE-open is **in-domain** for pooled4 (trained on slake+pathvqa+kvasir+vqa_rad).

## 10. Standing state (end of 2026-07-07)

The deliverable now has an honest end-to-end story against the **think** baseline: a single **format router**
(MCQ margin-cascade → 32B-nt + MMMU keep-7B + optional PMC fusion + G8 prefetch; open-text bo8+verifier cascade) that
**matches-or-beats always-32B-THINK at −95.6 % latency** (0.575 vs 0.563, +0.012; up to +0.024 with PMC fusion, FLOPs
1.26× always-32B). Two corrections stand (margin > agreement > CASP; router required). Caveats carried forward:
open-text 32B-think accuracy is **estimated**; open-text pooled-4 verifier is **in-domain**; **OmniMed-32B blocked**
(ALL-6 is the computable pool); fusion/best-of-N are latency+accuracy levers that **cost FLOPs**. Full method spec:
**`METHOD_FINAL_2026-07.md`**; ledger §6 updated; backlog 35 → 56. Nothing in `paper/` was touched.

---

> **UPDATE (evening 2026-07-07) — the loop did NOT stop at §10's "standing state."** The rest of the day
> (a) ran a **pass-4** of the idea backlog (§11), (b) folded the two best new levers (F8, F10) into a **single
> unified `method_final.py`** whose two Pareto modes are **BOTH now FLOP-negative** (§12), (c) spent the day's
> remaining GPU budget on three cost/accuracy probes (§13), (d) ran the pass-4 §H "remaining-headroom" ideas as
> `robust_slice_routing.py` (§14), and (e) recorded the **abstention prohibition** that governs the whole program
> (§15). Every number below is sourced to a named artifact under `results/cascade_methods/artifacts/`; §13's
> logit-fusion and the two quant/prune re-costings are the only GPU-adjacent items and each is flagged
> measured-vs-projected. Updated standing state in §16.

## 11. Pass-4 of the idea backlog (§H "remaining-headroom", 56 → 68) + the §F/§G extensions run

The two axes opened in §8 were pushed one tier deeper, and a **fourth pass** added a new section:

- **§F pass-4 ("[BEAT-32B]", extensions):** can the accuracy beat over always-32B extend **past PMC** via a
  smarter combiner? Coded in `beat32b_more.py` → `beat32b_more.json`. Four ideas: **F8** certified weak-veto,
  **F7** super-learner (logistic-frugal + GBM-rich stacks), **F11** decision-level Bayesian model averaging
  (additive + product-of-experts), **F10** learning-to-defer on open-text. Champion to beat = the §9 F3
  confidence-advantage fusion (PMC +0.0135, gated OFF elsewhere).
- **§G pass-4 ("★ESC", extensions):** three untested escalation-speed levers — **G7** semantic escalation cache,
  **G2** early-exit (generated-token patience / layer-depth), **G4** 32B image-token pruning. Coded in
  `escalation_more.py` → `escalation_more.json` (companion to §9's `escalation_levers.json` for G5/G6/G8 and to
  §13's `quantized_strong_leg.json` for G3).
- **§H pass-4 (NEW SECTION, "◆PASS4 remaining-headroom", +12 → 68):** ideas that **sidestep** the established
  walls rather than attack them again — H2 kNN/retrieval gating, H4 learned error-slice discovery
  (Domino/Spotlight), H8 actuarial (Bühlmann) credibility shrinkage, plus H1/H9/H10 and scheduling ideas. Run as
  `robust_slice_routing.py` (§14). **Running total: 35 (§A–E) + 11 (§F) + 10 (§G) + 12 (§H) = 68 ideas**
  (`METHOD_IDEAS_BACKLOG.md`). **H3 (three-way abstain-to-human) was EXCISED — abstention is out of scope (§15).**

### 11.1 §F pass-4 results (`beat32b_more.json`) — the MCQ beat does NOT extend past PMC; open-text does

All four are OFFLINE (no new inference). Every combiner independently **re-confirms the recoverability wall**: none
certifies a NEW beat-32B *closed* slice beyond PMC (F3's baseline: PMC +0.0135 CI[0.0103,0.0169] beats; MMMU +0.1733
CI[0.1067,0.24] the n=150 route-to-7B anomaly; all other closed sets d=0 → keep 32B).

- **F8 — certified weak-veto (the deployable bonus).** A cross-fit (5-fold) conformal Wilson-LB certificate per
  (dataset × 7B-confidence quantile-bin): **veto the 32B and keep the 7B** only where LB(7B precision) ≥ 32B accuracy
  on TRAIN (one-sided ⇒ never-worse); vetoed cells are **7B-ONLY** (skip the 32B ⇒ cheaper). PMC **0.5613 vs 32B-nt
  0.5518 = +0.0095, CI[0.0071,0.0118]**, veto_rate 0.4002. Pooled MCQ (n=40 029): **0.5844 vs 0.5765 = +0.0080,
  CI[0.0061,0.0099]**, veto_rate 0.3342, **FLOPs 0.885× always-32B** (vs F3's 1.22× because F3 runs both legs
  everywhere). **NEW accuracy cells vs F3: NONE** — no high-precision 7B bin exists inside the strong-32B closed sets.
  **Value = COST + SAFETY** (a certified, never-worse, *cheaper* route to the PMC beat), not new accuracy. This is the
  lever that makes the accuracy-max mode FLOP-negative in §12.
- **F7 — super-learner.** Frugal(logistic) pooled **+0.0110 CI[0.0089,0.0132]**; rich(GBM) pooled **+0.0136
  CI[0.0108,0.0164]**. NEW cells vs F3: **NONE**. Honest note: on these calibrated low-dim features the GBM does **not**
  catastrophically overfit (nets slightly higher pooled via PMC, only <0.002 regressions on PathVQA/MedXpert) — the
  CALM-Fuse overfit story is **not** reproduced here; both stacks just reproduce PMC + MMMU.
- **F11 — decision-level BMA.** Additive pooled **+0.0116 CI[0.0089,0.0142]**; PoE ≈ additive. NEW cells: **NONE**.
  The per-slice EM weight→32B auto-gates the strong slices (reproduces F3's PMC win, defaults toward 32B elsewhere) but
  is **not** perfectly guardrail-safe (small <0.003 leaks on SLAKE/PathVQA/MedXpert), unlike F1/F8 hard gating.
- **F10 — learning-to-defer on OPEN-TEXT (the one lever that moves a new axis).** A learned team-objective logistic
  rejector over **7B-side open-text features only** (verifier score max/range/mean/std, #unique preds, self-consistency,
  seqlogprob; no cross-model feature — open dumps carry only judge_ok, not the 32B answer text), threshold tuned on
  TRAIN to maximise TEAM accuracy, cross-fit. Per cell vs 32B-nt: **PathVQA_open 0.462 = +0.086 CI[0.064,0.106] BEATS**;
  **SLAKE_open 0.8202 = +0.0016** and **VQA_RAD_open 0.605 = +0.005** (point-positive, CI spans 0 at n=200–645) — both
  **flipped from below-32B to iso/above**, and F10 beats the deployed parity-targeting τ-gate on all three. Only
  CI-certified NEW open beat = PathVQA_open (already won by the verifier cascade; F10 improves +0.0773→+0.0860).
- **Verdict (`beat32b_more.json:verdict`):** *extends past PMC?* **PARTIAL** — on **MCQ NO** (F8/F7/F11 all confirm the
  recoverability wall; the closed beat is intrinsically bounded to comparable-skill/de-correlated slices = PMC, with
  radiology/pathology/MedXpert correctly kept at 32B); on **OPEN-TEXT YES in direction** (F10 removes the two residual
  open losses and improves the PathVQA_open win). This **sharpens** the paper's story rather than widening the claim.

### 11.2 §G pass-4 results (`escalation_more.json`) — G4 is the one live speed lever; G7/G2 are offline dead-ends here

Measured batch-1 cost constants: GEN7 347 ms, VER7 175 ms, GEN32-nt 665 ms; φ=0.586 ⇒ prefill32 = 389.7 ms,
decode32 = 275.3 ms.

- **G7 semantic escalation cache — DEAD offline.** No dump carries an image id/hash, so only normalized question
  **text** is a usable key, which conflates different images sharing a templated question. Duplication on the
  escalation-heavy cells is ~0 (MedXpert dup 0.0, PMC 0.0014, VQA-RAD 0.0637); where it is high it is **templated**
  (SLAKE 0.815, PathVQA 0.372, SLAKE-open 0.738) with *different* images → unsafe cross-image reuse. (The question-keyed
  sim on SLAKE-closed "cuts" esc 0.2045→0.0586 but that is the template artifact, not a real cache hit.) No image key
  in the dumps ⇒ the safe semantic-image cache is unmeasurable → future work.
- **G2 early-exit — DATA-ABSENT for its productive mechanism.** Both the 7B cheap leg and the 32B strong leg emit
  **~3 generated tokens on EVERY benchmark** (median 3, p90 3–7): the answer is a letter/word, so both legs are
  **PREFILL-BOUND** and generated-token patience saves ~0. The lever with real headroom is **layer-depth early-exit**
  (CALM/LayerSkip, ~2–3× on the whole forward incl. prefill), which needs intermediate-layer logits / an exit head not
  in the dumps → GPU probe (flagged, not run).
- **G4 32B image-token pruning — the single new lever with quantified offline headroom.** Analytical model
  (prefill-dominant, LAYER_RETAIN = 1 − 3/64 = 0.953, real token_cache image fractions): **projected per-escalation 32B
  latency/FLOPs −26% @ p=0.50 / −39% @ p=0.75** on the most image-token-rich cell (VQA-RAD-closed); **pooled cascade
  latency 459 → 432 ms @ p=0.50 (−5.9%)**. **Accuracy NEEDS a GPU CONFIRM** (labeled projection). Stacks with the
  §9-validated G8 (prefill prefetch) and G5 (futility suppressor). The GPU-feasibility investigation of G4 is §13.2.

## 12. The unified method (`method_final.py`) — folding F8 + F10 → BOTH Pareto modes go FLOP-negative

`method_final.py` merges the project's proven pieces into **one reproducible script** that writes `method_final.json`
(v1, the §6 F3-fusion configuration) **and** `method_final_v2.json` (v2, with **F8** replacing the accuracy-max PMC
cell and **F10** replacing the open-text arm's parity-τ gate, shared by both modes). Cost constants (measured
batch-1): GEN7 347 ms/1 FLOP, VER7 175 ms/1 FLOP, BO8 522 ms/16 FLOP, GEN32-nt 665 ms/4.57 FLOP, GEN32-think
10 521.6 ms/4.57 FLOP, FUSE-both-legs 665 ms/5.57 FLOP. All thresholds held-out (5-fold cross-fit). Two operating
modes on the Pareto knob, full suite n=42 374, sample-weighted:

| mode (v2) | acc | Δ vs 32B-think | Δ vs 32B-nt | esc | FLOP-eq | ×always-32B | lat_seq | lat_par | lat saved vs think (par) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **compute-lean** | 0.5754 | **+0.0123** | +0.0023 | 16.7% | **2.238** | **0.49×** | 577 ms | **468 ms** | **95.6%** |
| **accuracy-max** | 0.5844 | **+0.0212** | +0.0112 | 64.0% | **4.246** | **0.93×** | 839 ms | 729 ms | 93.1% |

- **Both modes are now FLOP-negative** (`confirmations.both_modes_flop_negative = true`). The change is the
  accuracy-max arm: the prior F3 fusion ran **both legs on 100% of PMC** = 5.57 FLOP/PMC-item → full-suite **5.695 FLOP
  = 1.246× always-32B (FLOP-POSITIVE)**. Swapping in **F8's certified veto** (7B on all + 32B only on the ~60% non-veto
  PMC cells = 3.74 FLOP/PMC) cuts full-suite FLOPs **5.695 → 4.246 (1.25× → 0.93×)** and **flips the arm FLOP-negative**,
  while **retaining 70.4% of F3's PMC beat** over 32B (PMC d_vs_nt +0.0135 → +0.0095, still CI-certified above 32B) and
  cutting the PMC cell's FLOPs 32.8%. Trade recorded honestly in `data_gaps`.
- **F10 lifts all three open cells** (shared by both modes): SLAKE_open +0.0109, VQA_RAD_open +0.0100, PathVQA_open
  +0.0100 vs the prior parity-τ gate; **the SLAKE_open and VQA_RAD_open losses are repaired** (both now ≥ 32B-nt). The
  parity-τ gate targeted iso-32B *by design*, so it sat at/below 32B on those cells; F10's team objective lifts them.
- **v2 vs v1 compute-lean** is a wash (+0.0005 acc, −0.006 FLOPs) — the compute-lean mode was already FLOP-negative in
  v1 (0.491×); F8/F10's value is concentrated in flipping the **accuracy-max** mode. Headline:
  **compute-lean +0.0123 @ 0.49× always-32B FLOPs; accuracy-max +0.0212 @ 0.93× (FLOP-negative)** — the Pareto knob is
  now FLOP-negative at **both** ends. Carried-over caveats unchanged (open-text 32B-think acc estimated; pooled-4
  verifier in-domain; F10 routes on 7B-side features only; PathVQA-closed has no 32B-think dump).

## 13. The GPU round — three cost/accuracy probes (one negative, two structural re-costings)

The day's remaining GPU/compute budget went to three questions the reframe surfaced. All are flagged
measured-vs-projected; none fabricates an accuracy number.

### 13.1 Logit-level fusion (`logit_fusion.py` → `logit_fusion.json`) — NEGATIVE; another wall confirmation

Does **full-posterior** (per-option logprob) fusion of Lingshu-7B + Lingshu-32B beat always-32B on **more cells** than
the decision-level F3 (which needs only top-1 conf/margin)? Tested OFFLINE on the only dumps that carry the full
option vector — `ckpts/gate_lingshu{7b,32b}_mcq` (300–500/slice, idx+gold aligned). Four combiners: F11_fixed /
F11_reweighted (log-opinion pool), F6 contrastive-decoding, F3_confadv.

- **Certified cells vs 32B (n_certified):** F3 = **{MMMU}**, F11_fixed = **{MMMU}**, F11_rw = **{MMMU}**, F6_cd = **{}**.
  Every method certifies **exactly one cell, and it is MMMU** — a route-to-7B anomaly (acc7 0.853 ≫ acc32 0.624; F11_rw
  learns λ≈0.04 = all-7B there), **not** a genuine fusion win.
- On the **4 perception sets** (broad4, n=1688), held-out fusion **collapses to λ≈1 / α≈0** ("just use 32B"): F11_rw
  −0.0047 (not certified), F6_cd exactly 0.0 (certifies **0** cells — the 7B "amateur" is not uniformly worse, so
  subtracting it is unsafe), F3 −0.0095. On the **PMC subsample (n≈500)** neither F3 nor F11 can certify a
  +0.0135-size effect (CI ~±0.03) and F11 vs F3 is within noise.
- **Why it can't be settled at full power:** the 33k MedEvalKit PMC dumps carry only top-1 conf/margin, not the full
  option vector — a power-matched full-posterior PMC test would need a **33k × 2-model GPU re-dump** (long tp=2 run,
  excluded by the SHORT-run guardrail). **Headline:** logit fusion does **NOT** extend the beat past PMC / past F3's
  cell coverage. This is a further independent confirmation of the fusion/recoverability wall.

### 13.2 G4 image-token pruning feasibility (`imagetoken_prune_gpu.json`) — DEFERRED with a concrete plan; projection stands

The G4 −26% FLOP saving (§11.2) is a **projection**; making it a measured accuracy claim was investigated and
**deliberately deferred** (a wrong impl would fabricate a lesion-safety accuracy number — no-fabrication rule).

- **Measured grounding (real, offline):** full-res image-token fraction of the prefill — VQA-RAD **0.918**, PathVQA
  0.895, PMC 0.852, SLAKE 0.844, MedXpert 0.68–0.80, MMMU 0.644 → the 32B short-answer forward **is** image-token-bound
  (the premise the projection rests on). The **measured resolution-cap analog** (encoder-side image-token reduction,
  `UNIFIED_METHOD_EXPERIMENTS.md` L616–617): PMC 0.543→0.542 (**−0.001, FREE**), SLAKE 0.825→0.809 (**−0.017**), VQA-RAD
  0.781→0.741 (**−0.040**) ⇒ cutting image information is FREE on PMC/pathology but **COSTS on radiology** → a 50% prune
  on the escalation-heavy radiology cells carries real accuracy risk → **per-benchmark radiology guardrail mandatory**.
- **Why not quick (feasibility):** production 32B inference here is vLLM (paged-KV + CUDA-graph don't support
  mid-network per-layer token drop without a custom plugin); Qwen2.5-VL is 64-layer with **3D M-RoPE** (pruning needs
  correct re-indexing of the retained tokens' 3D position ids); the vision-token span is dynamic; attention-ranking
  needs `attn_implementation='eager'` (slow/heavy at 32B). A **Path-A HF smoke** (locate image tokens, hook after layer
  K∈{2,3,5}, keep top-r, rebuild M-RoPE, parity-gate r=1.0 to full accuracy first; ~200/benchmark on
  VQA-RAD/SLAKE/MedXpert) is the recommended several-hours next step. Until then the −26% FLOP is a labeled projection,
  gated behind the radiology guardrail.

### 13.3 Quantized (INT4) strong leg (`quantized_strong_leg.py` → `quantized_strong_leg.json`, backlog G3) — a VRAM/energy win, NOT a FLOPs win

Re-cost the integrated cascade with an AWQ/GPTQ-INT4 32B strong leg. No pre-quantized Lingshu-32B is loadable in vLLM
and quantizing a 32B VLM ourselves is the forbidden rabbit hole; a ready benchmark (`bench_int4_strong_leg.py`) is
committed but a **HF-CDN outage stalled 2/6 AWQ shards** this session — so latency is a **composition-grounded
projection** and INT4 accuracy is from literature (Δ ≈ −0.005…−0.010).

- **FLOPs — NO win under the repo's unit.** The repo's FLOP-eq is a **MAC count** (7B-forward-equivalents), which is
  precision-independent: a 32B-INT4 forward is **still 4.57 FLOP-eq**, so method FLOPs are literally unchanged
  (full-suite 2.538, MCQ 1.738, open 16.181). Only under a *throughput-effective* accounting (~1.5 for decode) does the
  strong leg get cheaper, and even then only on the **high-escalation MCQ** cells (MedXpert 5.09→2.34, VQA-RAD-cl
  3.60→1.85, PathVQA-cl 3.09→1.69; pooled 2.54→2.06); the **open-text arm is unchanged** (16.0→~16.0 — its cost is
  97–98% best-of-8 cheap-7B forwards, the strong leg only 2–3%, so strong-leg quantization is the *wrong* lever there).
- **Latency — only ~12%, not ~2.5×.** The strong leg is **32B-NO-THINK = PREFILL-bound** (measured decode 69 ms/tok,
  ~2 gen tokens, decode ≈21% of the 665 ms call). AWQ-INT4 speeds **decode only** (~2–3× on A100), not the compute-bound
  prefill (dequant to fp16) nor the FP16-kept vision tower → **665 → 582.7 ms (ratio 0.876, ≈12% cut)**, not the ~2.5–3×
  the hypothesis assumed. (The INT4 decode win *does* apply to always-32B-THINK's ~300 decode tokens, 10 521→~4 525 ms,
  but the method already beats that baseline by ~23× on latency → moot.)
- **Bottom line:** INT4 strong leg is worth adopting for **VRAM / energy / deployability** — notably it fits the 32B at
  **tp=1 (~20 GB) on one GPU**, which sidesteps the OmniMed tp=2 NCCL hang — and it cleans up the MCQ FLOP story under a
  throughput-effective view, but it does **NOT** make the method uniformly FLOP-dominant vs always-32B-nt and does
  **NOT** materially change latency. **Quantization is a memory/energy lever, not a FLOPs lever, for this method.**

## 14. Robust slice-routing (`robust_slice_routing.py` → `robust_slice_routing.json`, §H pass-4) — H4/H8/H2 all negative; the beat is bounded to PMC/MMMU and the guardrail is validated

The three top offline §H ideas were run to try to **harden or extend** the beat-always-32B claim on the six Lingshu-7B/
32B MCQ cells (n_total 40 029). Baseline F1-certified non-32B cells: **PMC** (F3 fusion +0.0135) and **MMMU** (keep-7B
+0.167 CI[0.087,0.247], n=150 anomaly); all other closed sets → always-32B.

- **H4 — learned error-slice discovery (Domino/Spotlight): NO genuinely-new slice.** Over 8 stratified
  DISCOVER/CONFIRM splits (106 candidate slices, BH-FDR5 + Bonferroni, "genuinely-new" = a slice living **inside an
  always-32B dataset**, calibrated against a label-permutation null): mean BH-FDR5 survivor count **0.25/split**; the raw
  genuinely-new count (**1.62/split**) sits **below the permutation null** (mean 5.61, p95 15); **no** new slice recurs in
  more than 5/8 splits (closest near-miss: MedXpertQA-MM|bsys=Respiratory, 5/8). Discovery instead **reliably re-finds
  the known PMC/MMMU wins** as echoes (PMC|wh=what 8/8, PMC|margmid 8/8, PMC|marghi 8/8, MMMU|wh=other 7/8) **without
  being told they are special → validates the F1 hand-gate**. The beat does **not** extend past PMC/MMMU via slice
  structure → the artifact records this as the **6th independent confirmation of the recoverability wall** (the honest
  deliverable).
- **H8 — Bühlmann-Straub credibility shrinkage: the overfit risk is REAL but F1's existing guardrail already fixes it.**
  Naive point routing (deviate iff raw discovery advantage > 0) **overfits thin slices**: on the fine 61-slice family it
  yields **~7.5 held-out guardrail violations/split** (of ~24.5 deviating slices). Bühlmann shrinkage helps only
  **marginally (7.5 → 6.62)** — an MSE-optimal k (mean 244) is too weak to flip a large thin-slice noise (~+0.05) toward
  a mildly-negative parent (~−0.01). The **decisive robustifier is the simple CI lower-bound guardrail** (deviate iff the
  discovery 95% lower-CI > 0 — **exactly F1's existing rule**): it drives fine-family violations to **0.25** and
  hand-family to **0.0** at a **preserved pooled beat +0.0117**. So credibility shrinkage does **not** beat the guardrail
  the method already deploys; **H8's value is diagnostic** (it confirms the overfit risk is real and that F1's CI
  guardrail — not a fancier actuarial estimator — is the correct, sufficient fix that keeps the per-slice program
  guardrail-honest).
- **H2 — kNN neighborhood-recovery gate: 0/5 datasets beat the scalar margin gate.** kNN's low-budget accuracy area
  loses to margin on every MCQ set (e.g. MedXpert 0.2684 vs 0.2746, PMC 0.5576 vs 0.5597, SLAKE 0.8577 vs 0.8598,
  VQA-RAD 0.8127 vs 0.8295, PathVQA 0.8768 vs 0.8792). **`knn_beats_margin = false` on all 5** → the escalation signal is
  intrinsically weak on MCQ (the recoverability wall, ~0.6 AUROC), consistent with H2's own risk note.
- **Net:** the beat-always-32B claim is **robustly bounded to PMC (broad comparable-skill slice) + MMMU (anomaly)**; no
  automatic slice-structure method extends it, thin-slice overfit is a real failure mode, and the fix is **F1's
  CI-lower-bound guardrail** (already deployed), which H8 validates. The per-slice routing program is guardrail-honest.

## 15. Abstention prohibition — H3 removed (permanent scope rule)

Recorded so the reasoning is not lost: **abstention is permanently out of scope in this project — the method must
always answer.** Accordingly, backlog idea **H3 (three-way abstain-to-human) was EXCISED**, and no
reject-option / defer-to-human / selective-prediction-as-a-method is proposed, built, or tested anywhere in this arc
(`METHOD_IDEAS_BACKLOG.md` top banner). This is a hard rule, not a preference. Note it does **not** touch the answering
mechanisms above: **F8's "certified veto"** vetoes *escalation to the 32B* and **keeps the 7B answer** — it always emits
a prediction, so it is an allowed answer-producing gate, not abstention.

## 16. Standing state (end of 2026-07-07, evening — supersedes §10)

The deliverable is a single **format-aware cascade** (`method_final.py`) with a **FLOP-negative Pareto knob at both
ends**: **compute-lean +0.0123 vs always-32B-THINK @ 0.49× always-32B FLOPs / 468 ms** and **accuracy-max +0.0212 @
0.93× (FLOP-negative)** — the accuracy-max mode flipped FLOP-negative today by swapping F3 fusion for **F8's certified,
cheaper, never-worse veto** (retains 70% of the PMC beat), and **F10** repaired the two residual open-text losses. The
day closed **four independent negatives that sharpen (not shrink) the story**: logit-level fusion (§13.1), and the three
§H slice-structure ideas H4/H8/H2 (§14) all **re-confirm the recoverability wall** — the MCQ beat is intrinsically
bounded to **PMC + MMMU**, and F1's existing **CI-lower-bound guardrail** is validated as the correct fix (H8). Two
re-costings landed: **G4 image-token pruning** is deferred-with-a-plan (projection −26% FLOP stands, needs a GPU
lesion-safety confirm), and the **INT4 strong leg** is a **VRAM/energy/deployability** win (fits tp=1, dodges the
OmniMed hang) but **not** a FLOPs win under the repo's MAC unit and only ~12% on latency. Backlog **56 → 68** (pass-4,
new §H). Standing caveats carried forward: open-text 32B-think accuracy **estimated**; pooled-4 open-text verifier
**in-domain**; **OmniMed-32B blocked** (ALL-6 is the computable pool); best-of-N / F3-fusion are latency+accuracy levers
that **cost FLOPs**; G4 accuracy and INT4 latency/accuracy are **projections** pending GPU. **Abstention remains
forbidden (§15).** Nothing in `paper/` was touched.
