# REPORT SCAFFOLD — "A cheap verifier for a 7B medical VLM"
Structure agreed with Leo 2026-08-16. Data-path order, one section per component,
experiments inline with the piece they shaped, results at the end.
Audience: coming in cold. No contamination content. Unflattering numbers included.
STATUS KEY:  [SETTLED] = number is final   [PENDING] = waiting on a running job

---

## 0 · The map                                                          [SETTLED]

Define, for a cold reader:
  - VLM, medical VQA, the two answer formats (multiple-choice vs open-ended)
  - CELL = one benchmark x one answer format. We report 8 of them.
      PMC-VQA (MCQ, n=33,430) · SLAKE-closed (836) · VQA-RAD-closed (251)
      PathVQA-closed (3,362) · MedXpert-MM (2,000)
      SLAKE-open (645) · VQA-RAD-open (200) · PathVQA-open (1,500)
  - MACRO = every cell counts 1/8, regardless of size. Why: sample-weighting would
      give PMC-VQA 79.2% of the score.
  - FLOP-eq = one Lingshu-7B forward pass. The baseline always-7B = 1.0 by definition.
  - THE BASELINE IS always-7B (macro 0.5971). Everything is measured against it.
Diagram: the data path, each box carrying its measured contribution.

## 1 · The router                                                       [SETTLED]

Mechanism: reads the prompt text only (never the gold answer) and decides MCQ vs open.
Reliable because the harness templates differ structurally: option-bearing prompts
supply an answer space, open ones do not.
Experiments that shaped it — both attempts to REMOVE this piece failed:
  - (choice)(why): treat MCQ as constrained open text so the verifier sees a
    justification.  sel_eff 0.7751 vs 0.7977 letter-only = -0.0226 [-0.0433,-0.0024]
    SIGNIFICANT LOSS.                        [choicewhy_measure_2026-08-03.json]
  - unified candidate-set pipeline: one scorer, candidates read off the prompt.
    Beat the 7B's own argmax on 0 of 4 MCQ cells; two significant losses.
                                             [unified_pipeline_2026-08-12.json]
Contribution: the two-arm split is empirically forced, not a design preference.

## 2 · The cheap leg — the 7B generator                                 [SETTLED]

Mechanism: Lingshu-7B, 8.29 B params, 15.45 GiB bf16. Samples N candidate answers.
Experiments:
  - TEMPERATURE LADDER (8 rungs, in-session matched control, 3 seeds, both currencies)
      T      dist/8  oracle  sel_eff  SELECTED   judge          EM
      0.00    1.00   0.4627  1.0000   0.46269   -0.0178 LOSS   -0.0085 tie
      0.10    1.38   0.4967  0.9511   0.47235   -0.0081 tie    +0.0014 tie
      0.20    1.79   0.5305  0.9108   0.48316   +0.0027 tie    +0.0091 win
      0.30    2.18   0.5579  0.8717   0.48628   +0.0058 tie    +0.0125 win
      0.40    2.58   0.5864  0.8356   0.48984   +0.0094 WIN    +0.0135 WIN   <- PEAK
      0.50    2.96   0.6013  0.8098   0.48671   +0.0063 win    +0.0095 win
      0.70    3.66   0.6277  0.7655   0.48045   control        control
    Peak T=0.4, argmax in BOTH currencies, guardrail-clean (every cell non-negative).
    Mechanism: three monotone curves. distinct-candidates and oracle rise with T;
    selection efficiency falls. SELECTED is their exact product, so it peaks inside.
    Greedy is not the limit of a good trend — it is a significant LOSS, because at
    T=0 the pool holds ONE answer and there is nothing to select from.
                                             [decoding_ladder_cold_2026-08-14.json]
  - N: T=0.4 at N=4 (0.48205) BEATS deployed T=0.7 at N=8 (0.48045).
  - OTHER DECODING PARAMS: min-p weakly positive; repetition_penalty is a currency
    conflict (wins under the judge, loses under exact match) — diagnosed as verbosity
    harvesting the judge's instructed leniency, not a real gain.
                                             [decoding_sweep_2026-08-13.json]
  - RESOLUTION: the MCQ cells already run at native (the cap binds on 0.000 of images
    in all five), so raising it is impossible. Cutting to cap320 costs 0.0076 macro.
    Open generator sits at the knee.         [resolution_sweep_2026-08-13.json]
  - VISION-AXIS DIVERSITY: raises oracle@8 (+0.0267, 3/3 draws) and even moves the
    coverage ceiling, but 0 of 12 arm-draws reach SELECTED, at 13x the prefill cost.
                                             [vision_diversity_2026-08-13.json]
Contribution: T=0.4 is free (+0.0094 judge / +0.0135 EM), needs no retraining.

## 3 · The verifier — what actually picks the answer                    [SCOPE FIXED 2026-08-16]

>> DECISION: THE METHOD IS HEAD-ONLY. The LoRA adapter is on hold until after this report.
>> Present the head as the verifier. Mention the adapter only as the more expensive alternative
>> it replaces (0.7752 at a full 7B forward pass per candidate, vs the head's 0.7956 at ~1.8 MFLOP).
>> Do NOT present the fused 0.8065 / 0.810627 numbers as the method -- they include the adapter.

Explain BOTH components plainly for a cold reader:
  (a) THE LoRA ADAPTER. A set of small correction matrices (47.6 M params, 182 MB)
      applied on top of Lingshu-7B. It cannot run alone — the thing that executes is
      the whole 8.29 B model with modified weights. So scoring one candidate costs a
      FULL 7B forward pass. LoRA makes training and storage cheap; it does NOT make
      inference cheap. sel_eff 0.7752.
  (b) THE HEAD. A standalone two-layer network, 3584 -> 256 -> 1, 918,017 params
      (x8 seeds = 7.34 M, 28 MB). Reads the layer-21 hidden state the base model
      already computed while processing the candidate, and outputs one score.
      ~1.8 MFLOP per candidate — about a millionth of a 7B forward pass.
      sel_eff 0.7956 — HIGHER than the adapter, at ~1/9000 the compute.
      Trains in ~15 seconds on CPU per seed (vs 107.7 GPU-minutes for the adapter).
      8 seeds are averaged for STABILITY, not accuracy: single seeds vary by ~0.021.
  Fused: 0.8065. Frozen 8-seed ensemble fused: 0.810627.
  Random-pick floor: 0.6763.
[PENDING] which structure is recommended — head-only vs adapter-only vs fused, with
          measured cost.                     [verifier_restructure_2026-08-16.json]

## 4 · How far does 7B + head get, on its own?                          [PARTLY PENDING]

>> COST STORY IS NOW SIMPLE, and this is the spine of the section:
>>   head-only + states captured during generation  =>  verification cost ~= 0
>>   therefore  THE ARM'S COST IS THE GENERATION COST, AND NOTHING ELSE.
>> Measured generation at N=8 (controlled cache-on/off A/B, 2026-08-16):
>>   the project's old cost model .......... 8.00 FLOP-eq   (overcharged 3.5x)
>>   MEASURED, prefix caching on ........... 2.28 FLOP-eq
>>   LM prefill shares (1.16 of N=1) but the VISION TOWER does not (4.74 of 8)
>>   if the ViT is made to share ........... ~1.3-1.4 FLOP-eq  [PENDING]
>> Against the always-7B baseline of 1.0. Wall clock: 8 samples = 1.86x the time of 1.

Your requested section: the verifier's value BEFORE any cascade.
  cell            7B greedy  +verifier(N=8)  gain     cost vs one 32B pass
  SLAKE-open        0.7364      0.7473      +0.0109        2.43x
  VQA-RAD-open      0.4650      0.4800      +0.0150        2.90x
  PathVQA-open      0.3240      0.3733      +0.0493        1.91x
[PENDING] the same table restated against the ALWAYS-7B baseline (=1.0), with the
          restructured cost, plus latency.
[PENDING] whether the closed cells join this table when asked open-ended.
                                             [closed_as_open_2026-08-16.json]

## 5 · Adaptive-N — how many samples to draw                            [SETTLED mechanism, PENDING refit]

Source: Weitzman, "Optimal Search for the Best Alternative", Econometrica 1979 —
the Pandora's Box problem. Each extra sample is a box you pay to open; the rule
computes a reservation value per box and stops when no box is worth opening.
One lambda yields BOTH thresholds: draw-another, and escalate.
Contribution: iso-accuracy at 11.74 vs 16.0 FLOP-eq = -27% of the open arm's cost,
mean 4.37-6.63 draws instead of a fixed 8.
[PENDING] the refit at T=0.4 with near-zero verification cost, which should reduce N
          further.                           [weitzman_T04_2026-08-15.json + restructure]

## 6 · The margin gate                                                  [SETTLED]

Mechanism: gap between the top-1 and top-2 option log-probabilities; escalate below a
cross-fit threshold.
Experiments: beat 13 alternatives including learned GBM, conformal and CP-Router;
best escalation-at-parity (32.5%). The best DETECTOR is not the best gate.
                                             [gate_unified_bakeoff.json]
Limit: recoverability — predicting "will the big model fix this?" caps at ~0.6 AUROC
across 16 mechanisms.

## 7 · The certified veto                                               [SETTLED]

EXPLAIN THE WILSON BOUND PROPERLY, with a worked example. The idea: don't ask "is the
7B probably right?", ask "am I CONFIDENT ENOUGH that the 7B is right?". Group items
into confidence bins; in each bin compute a conservative lower bound on the 7B's true
accuracy from a finite sample; if that lower bound still beats the 32B's accuracy in
the same bin, keep the 7B's answer and never call the 32B. It ANSWERS — it is not a
reject option.
NEW RESULT: the binning was backwards. COARSER is better — n_bins 2-3 (shipped: 5)
raises the PMC veto rate from 0.400 to 0.667 at +0.0090 to +0.0105, AND is cheaper.
Small bins made the Wilson intervals too wide to certify anything, which is why 4 of 5
MCQ cells sat at veto rate 0.0000.           [veto_binning_2026-08-15.json]

## 8 · Escalation and the strong leg                                    [SETTLED]

Where compute goes: 32B escalation 46.4%, cheap generation 27.6%, verifier 26.0%.
By phase: LM prefill 82.1%, vision towers 16.7%, ALL decode 1.2% — prefill-bound, so
decode-side tricks cannot matter.
44.1% of compute is cheap-side work on questions that get escalated anyway.
                                             [cost_decomposition_2026-08-12.json]

## 9 · Results                                                          [PENDING]

Per-benchmark improvement over always-7B, with compute and latency. Then the 32B
comparison as CONTEXT, clearly secondary.
[PENDING] everything.

---
### Numbers that must NOT appear
- anything about verifier contamination or the decontamination history
- any in-flight result
- the retracted prompt-frame mechanism for why the head works
