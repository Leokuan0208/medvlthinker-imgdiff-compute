# VERIFIED FACTS (2026-06-28) — every fact traced to source; build only on these.

## A. Per-benchmark accuracy — MedVLThinker, from results/cascade_methods/master_data.csv (ALL-6 rows)
| benchmark | 7B no-think | 32B no-think | 32B think | character |
|---|---|---|---|---|
| PMC-VQA | 0.543 | 0.551 | 0.556 | competent MCQ, tiny 7B→32B gap |
| SLAKE | 0.762 | 0.849 | 0.764 | competent; no-think > think (over-thinking) |
| VQA-RAD | 0.761 | 0.853 | 0.776 | competent; no-think > think |
| PathVQA | 0.641 | 0.661 | 0.672 | competent; think ~ no-think |
| MMMU-medical | 0.547 | 0.624 | 0.688 | COMPETENT (NOT near-chance); reasoning — think > no-think |
| MedXpert-R | 0.225 | 0.279 | 0.326 | near-chance (7B below 4-opt chance 0.25); excluded from headline |
| MedXpert-U | 0.256 | 0.292 | 0.385 | near-chance for 7B; excluded from headline |
SOURCE: master_data.csv, family=medvlthinker, pool=ALL-6.

## B. Answer-format availability (which datasets have OPEN-ENDED data; verified by file presence)
- HAVE open-ended sc8: SLAKE, VQA-RAD, PathVQA, Kvasir, RadImageNet.
- MCQ-ONLY (no open-ended): PMC-VQA, MMMU-medical, MedXpert. ⇒ verifier (open-ended) set is FORMAT-determined.

## C. Open-ended accuracy (LLM-judge), full set — cheap=Lingshu-7B, strong=Lingshu-32B (verified, computed)
| dataset | greedy | self-consistency | random | 32B single | oracle@8 | n |
|---|---|---|---|---|---|---|
| SLAKE | 0.722 | 0.736 | 0.720 | 0.819 | 0.879 | 645 |
| VQA-RAD | 0.420 | 0.465 | 0.441 | 0.600 | 0.630 | 200 |
| PathVQA | 0.295 | 0.324 | 0.294 | 0.376 | 0.517 | 1500 |
| Kvasir | 0.287 | 0.286 | 0.282 | 0.301 | 0.491 | 1200 |
| POOLED | 0.377 | 0.394 | 0.375 | 0.444 | 0.580 | 3545 |
SOURCE: ckpts/openvqa/{cheap_lingshu7b,strong_lingshu} sc8+judge (computed 2026-06-28). 32B pooled 0.444 confirmed.

## D. Trained verifier (held-out test split, n=1064) — from lora_verifier_pooled4/result.json + GROUND_TRUTH_NUMBERS.md
pooled trained 0.501 (greedy 0.413, SC 0.411, oracle 0.592, 49% gap). Per-ds: PathVQA 0.441, Kvasir 0.405,
VQA-RAD 0.611, SLAKE 0.762. Discrimination AUROC 0.924 (n=8512). Bootstrap gain +0.116 [+0.092,+0.139].
Boxes: SLAKE 0.255 (40%), MS-CXR 0.232 (78%, bootstrap +0.191 [+0.152,+0.232]).

## E. Peer provenance (VERIFY each before citing; filled from web search)
- Self-consistency: Wang et al., "Self-Consistency Improves CoT Reasoning", ICLR 2023. [confident]
- P(True) self-verification: Kadavath et al., "Language Models (Mostly) Know What They Know", 2022 (Anthropic). [confident]
- GenRM generative verifier: Zhang et al., arXiv:2408.15240, ICLR 2025. [confident]
- Self-certainty BoN: arXiv:2502.18581, NeurIPS 2025. [confident]
- CP-Router: Su et al., arXiv:2505.19970, 2025. [confident]
- FrugalGPT: Chen et al., 2023. [confident]
- ABC = "Agreement-Based Cascading for Efficient Inference", arXiv:2407.02348, ICML 2024 ES-FOMO workshop. [VERIFIED] (prior art for our agreement gate)
- CAR = "Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning", arXiv:2505.15154, 2025. [VERIFIED] (closest prior art to ACC think-gating; multimodal)
- Jitkrittum et al., "When does confidence-based cascade deferral suffice?", NeurIPS 2023. [VERIFIED] (theory: optimal deferral needs BOTH models' confidence — backs our gate-saturated finding)

## F. CANONICAL same-split open-ended table (held-out test, n=1064) — the verifier comparison
Source: result.json (greedy/SC/verifier/oracle, true greedy) + clean_dump analysis (32B same-split). All n=1064.
| dataset | greedy | self-consistency | 32B same-split | verifier | oracle | n |
|---|---|---|---|---|---|---|
| SLAKE   | 0.738 | 0.738 | 0.829 | 0.762 | 0.895 | 210 |
| VQA-RAD | 0.519 | 0.500 | 0.648 | 0.611 | 0.722 | 54 |
| PathVQA | 0.352 | 0.349 | 0.377 | 0.441 | 0.513 | 435 |
| Kvasir  | 0.282 | 0.282 | 0.326 | 0.405 | 0.493 | 365 |
| POOLED  | 0.413 | 0.411 | 0.462 | 0.501 | 0.592 | 1064 |
KEY: verifier 0.501 > 32B 0.462 > SC 0.411 ≈ greedy 0.413 (SC below greedy = majority trap). gap captured 49%.
Per-ds: verifier BEATS 32B on PathVQA + Kvasir; 32B wins SLAKE + VQA-RAD. (32B full-set pooled = 0.444; same-split = 0.462.)
Verifier-augmented cascade: 0.517 @ 35% escalation (cost 17.5 7B-equiv) — accuracy-optimal, compute premium.

## G. Headline-claim rigor + selection-rule iteration (offline, n=1064, 2026-06-28)
- verifier(argmax) 0.5009 vs 32B 0.4624: paired bootstrap **+0.0385, 95% CI [+0.010, +0.066] — EXCLUDES 0**
  (the "beats the bigger model" claim is significant, though a modest margin).
- verifier vs greedy(first-sample 0.385): +0.116 [+0.092, +0.140].
- Selection-rule iteration (which rule is best?): verifier-ARGMAX **0.501** > verifier-weighted-vote 0.489
  (−0.012 [−0.024,0.000]) > score×count hybrid 0.470 (−0.031) > majority 0.411. => argmax is best; mixing in
  vote counts HURTS because the majority trap contaminates even score-weighted voting. Pure verifier-argmax is the rule.

## H. Other-base verifier (MedVLThinker-7B, full method from scratch, slake+vqa_rad) — HONEST mixed result
SLAKE: greedy 0.564 -> trained 0.622 (42% of gap, oracle 0.702) — works.
VQA-RAD: greedy 0.500 -> trained 0.470 (-20%, WORSE than greedy; n=54, noisy) — fails.
POOLED: greedy 0.547 -> trained 0.583 (25% of gap, oracle 0.689) — positive but weaker than Lingshu (49%).
TAKEAWAY: the method transfers to a 2nd base but NOT uniformly; base-model quality matters (the Lingshu verifier
even TRANSFERS to MedVLThinker outputs better, 49-61%, than a from-scratch MedVLThinker verifier achieves, 25%).
Headline = Lingshu verifier (49%, 4 datasets, 2-seed); other-base = partial validation + honest caveat.

## I. 2-SEED ROBUSTNESS — CRITICAL HONESTY CORRECTION (2026-06-29)
Pooled-4 verifier, two seeds (different 70/30 splits; split reconstruction validated: seed0 32B=0.462 matches clean_dump exactly):
- seed0: greedy 0.413 -> verifier 0.501 (49% of gap); 32B on seed0 split = 0.462 -> verifier BEATS 32B by +0.039 [+0.010,+0.066].
- seed1: greedy 0.379 -> verifier 0.445 (35% of gap); 32B on seed1 split = 0.450 -> verifier 0.445 ~= 32B 0.450 (TIE, -0.005).
=> "beats the 32B" is SEED-DEPENDENT (win on seed0, tie on seed1). HONEST claim: verifier MATCHES / is competitive
   with the 32B (a 7B reaches a 5x model's accuracy via test-time compute), avg +0.017. DO NOT claim "beats the 32B" unconditionally.
- ROBUST claim (both seeds): verifier DECISIVELY beats training-free selection (+0.066 seed1 / +0.088 seed0 over greedy; SC~=greedy).
- Per-dataset robustness: verifier helps HARD sets on BOTH seeds (PathVQA 36-?%, Kvasir 51%/?%); FLAT on easy/high-baseline
  or tiny sets (seed1 SLAKE -4% @ greedy 0.774; VQA-RAD 0% @ n=69). Gap-captured 35-49% (mean ~42%), hard-set-driven.

## J. LINGSHU PAPER TARGET NUMBERS (arXiv 2506.07044 Table 6, medical multimodal VQA — VERIFIED from PDF)
Protocol = MedEvalKit (their unified eval framework). 7 multimodal VQA tasks. OMVQA=OmniMedVQA, MedXQA=MedXpertQA-MM.
| model | MMMU-Med | VQA-RAD | SLAKE | PathVQA | PMC-VQA | OmniMedVQA | MedXpertQA | Avg |
| Lingshu-7B  | 54.0 | 67.9 | 83.1 | 61.9 | 56.3 | 82.9 | 26.7 | 61.8 |
| Lingshu-32B | 62.3 | 76.5 | 89.2 | 65.9 | 57.9 | 83.4 | 30.9 | 66.6 |
(ref: Qwen2.5VL-7B 50.6/64.5/67.2/44.1/51.9/63.6/22.3 avg 52.0). GOAL: reproduce these (match), then build verifier+cascade
to beat 32B (66.6) at lower latency than 32B. Focus = Medical VQA only (no report-gen / textual). [[verify-foundational-facts]]
