# Open-ended generation–selection gap is a LUCK floor, not harvestable signal

> New-method-loop, 2026-06-25 (Direction 3, after abstention was dropped at user request).
> Deep-dive on SLAKE-open (Lingshu-7B generator, n=645), all LLM-judge scored under ONE consistent
> judge run (MedVLThinker-32B grader). PathVQA-open de-risk confirms the same shape. All numbers from
> real checkpoint output (standing no-fabrication rule). Code (committed): `run_openvqa_verify_persample.py`,
> `run_openvqa_select_listwise.py`, `run_openvqa_synth.py`, `explode_sc_for_judge.py`, `select_eval.py`.

## Question
Is there a NON-abstention, training-free way to lift open-ended medical-VQA accuracy? The cheap model's
8 sampled answers have a large oracle gap (best-of-8 ≫ greedy). Can a verifier SELECT the right sample,
or can the strong model SYNTHESIZE from the candidates, to beat the SOTA single pass?

## The headroom is real and large (survives the semantic judge)
| dataset | greedy | SC-majority | oracle@8 | headroom |
|---|---|---|---|---|
| SLAKE-open | 0.730 | 0.736 | **0.879** | +0.149 |
| PathVQA-open | 0.343 | 0.324 | **0.517** | +0.174 |

- **Self-consistency FAILS** — flat on SLAKE, *hurts* on PathVQA (majority < greedy). Mechanism: a
  **majority trap** — in 74–90% of recoverable questions the correct answer is a MINORITY vote
  (mean ~1.5 / 8 votes). Majority voting structurally cannot reach it.
- Inspection: most recoverable cases are genuine *content* recovery (GOLD 'Lung Cancer' vs greedy
  'Pneumonia'; '4 organs' vs '3'), with some judge phrasing-noise — so the headroom is mostly real.

## But every selector sits at the LUCK floor (SLAKE, one consistent judge run, n=645)
| selector | judge acc | vs random | vs oracle gap |
|---|---|---|---|
| RANDOM pick (mean sample acc) | 0.720 | — | — |
| self-verify P(Yes) argmax (Lingshu-7B) | **0.715** | **−0.005 (worse than random)** | — |
| SC-majority | 0.736 | +0.016 | 10% |
| 32B pointwise verify argmax | 0.746 | +0.026 | 16% |
| 32B LISTWISE select (compare all candidates) | **0.758** | +0.038 | **24%** |
| learned fusion [self,32B] (5-fold CV) | 0.743 | +0.023 | 14% |
| **oracle@8 (luck ceiling)** | **0.879** | +0.159 | 100% |
| **32B free-gen single pass (SOTA bar)** | **0.819** | — | — |

- The best training-free selector (listwise-32B, 0.758) captures only **24% of the gap above random**
  and is **far below the 32B single-pass SOTA (0.819)**. Self-verification is *worse than random*.
- **Candidate-conditioned SYNTHESIS backfires:** priming the 32B with the 7B's candidate answers gives
  **0.774 vs 0.819 free-gen (−0.045)** — the majority trap drags the strong model down.

## Conclusion — a luck floor, unifying with the project's backbone
The open-ended generation–selection gap is **sampling LUCK, not latent knowledge.** The model does not
*know* which sample is right (if it did, that would be its greedy answer), so no verifier built from
these models can reliably find it — self-verify lands *below* random, the strongest training-free
selector reaches only ~24% of the gap, and no method beats just running the 32B once. This is the SAME
**luck-floor** structure that killed single-model routing earlier in the project (oracle looks
exploitable; it is a luck artifact). 

## Additional mechanisms tested 2026-06-25 (all bounded)
- **Cross-family agreement** (MedVLThinker-7B + Lingshu-7B, decorrelated errors): a real reliability signal
  (P(correct|agree)=0.819 vs disagree 0.649, beats the 0.586 MCQ gate) BUT as an *accuracy* selector it
  collapses to "trust the stronger model" (consensus 0.730 = Lingshu alone) — the weak model is rarely right
  when it dissents (0.289). Its only clean use is abstention (rejected by user). `tmp/crossfamily_agree.py`.
- **Few-shot ICL** (k=5 random train exemplars to align answer style, `run_openvqa_fewshot.py`): HURTS —
  PathVQA 0.343→0.203 (−0.140), SLAKE 0.730→0.705 (−0.025). Random exemplars inject answer-bias and disrupt
  native answering; the style-alignment hypothesis fails.

This is the **fourth independent confirmation** of the loop's meta-finding, now across GATE (margin
saturated), ACTION (recoverability capacity-bound, [[recovery-is-capacity-bound]]), SELECTION (this
doc), SYNTHESIS, cross-family agreement, and few-shot ICL: **oracle gaps are real and large everywhere, but training-free signals
(confidence, verification, agreement, selection) are too weak to harvest them, and naive aggregation
actively hurts via the majority trap.** The genuine positives remain the structural/regime
contributions already in the paper (ACC, Visual-Stability Rescue, §5.7 open-ended detection ceiling-break).
A training-free SOTA-beating *accuracy* method does not exist in this family; the only avenue the
evidence leaves open is a *trained* multimodal verifier — and the near-random verifier signal predicts
that, too, would capture little (low EV).
