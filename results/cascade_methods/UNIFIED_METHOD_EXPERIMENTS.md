# Unified-method experiment log (open-text medical VQA) — running record
Goal: a unified test-time-compute method (cascade gate + verifier) that beats the strong 32B on the open-text
suite (accuracy + FLOPs; latency/energy where the family allows), across MedVLThinker, Lingshu, (InternVL3).
Datasets: SLAKE, VQA-RAD, PathVQA, Kvasir (in-dist) + RadImageNet (HELD-OUT OOD). Judge-based accuracy.
All inference captures: per-sample preds, judge label, margin/conf/seqlogprob (gate signals), gen_tokens(+all), lat_s.
Cost model: FLOPs=2N(P+G) from gen_tokens; latency/energy from measured batch-1 per-tier fits (per family).
Drive: keep main <80% (big data -> /data; no OmniMed).

## Phase 0 — Baselines (greedy / SC / oracle@8 / 32B), judge-based
See OPENTEXT_BASELINE.md. Lingshu complete (7B 5ds, 32B 4ds). MedVLThinker gap-fill RUNNING (7B kvasir/radimagenet
+ 32B all-5, instrumented). [TABLE TO FILL WHEN RUNS FINISH]

## Phase 1 — Best cascade gate (escalate cheap->strong)
Candidates: confidence/seqlogprob, margin(top1-top2), self-consistency(n_distinct), verifier-confidence(max sc),
CASP-stability(trained logistic on cheap signals), learned/MLP. Metric: acc-vs-escalation frontier vs always-32B.
[RESULTS TBD]

## Phase 2 — Best verifier setup (best-of-N selection)
Vary N in {2,4,8,(16)}; selection rule (argmax confirmed best > weighted-vote); which model generates the samples;
verifier base. Metric: gap-captured, acc, cost. [RESULTS TBD]

## Phase 3 — Integration (the two methods + variants)
M1 (router): detect MCQ vs open-text -> cascade vs verifier. (Open-text suite => verifier path mostly.)
M2 (unified pipeline): every prompt -> verifier-pick -> gate -> escalate; per-tier (7B-nt, 32B-nt, 32B-think for MedVLThinker).
Variants: unified trained module (verifier+gate jointly) vs separate; agreement on the verifier-picks of the two small models.
Metric: acc + FLOPs + latency + energy vs always-32B, in-dist + held-out OOD. [RESULTS TBD]

## Findings (append as they land)
- (pending baselines)

## Phase 0 RESULTS — open-text baseline (judge-based), 2026-06-29
| greedy / 32B(acc) | SLAKE | VQA-RAD | PathVQA | Kvasir | RadImageNet(OOD) |
| Lingshu-7B greedy | 0.722 | 0.420 | 0.295 | 0.287 | 0.321 |
| Lingshu-7B oracle@8 | 0.879 | 0.630 | 0.517 | 0.491 | 0.512 |
| Lingshu-32B | 0.819 | 0.600 | 0.376 | 0.301 | (pending) |
| MedVLThinker-7B greedy | 0.543 | 0.395 | (sc8 gap) | 0.324 | 0.201 |
| MedVLThinker-32B | 0.623 | 0.525 | 0.074* | 0.361 | 0.202 |
FINDINGS:
- Lingshu = the STRONG-32B target (competent open-text: SLAKE 0.819, VQA-RAD 0.600) -> the real "beat the 32B" challenge. HEADLINE family.
- MedVLThinker-32B is WEAK on free-text (MCQ-RL-tuned): PathVQA *0.074 (full n=3357, judge verified real, not a bug), RadImageNet 0.202.
  => 7B+verifier trivially beats it on free-text; MedVLThinker shows the verifier's value when the big model is poorly adapted. Secondary family.
- 7B oracle@8 high everywhere -> large verifier headroom.
DATA FIXES NEEDED before experiments: (1) PathVQA question-set mismatch — MedVLThinker uses full 3357, Lingshu uses a 1500 subset
  -> unify (use the same pathvqa set per family/cross-family). (2) MedVLThinker-7B PathVQA has only greedy (no sc8) -> generate sc8.
  (3) Lingshu-32B @ RadImageNet (held-out) not yet generated. (4) MedVLThinker-7B-pathvqa greedy judge running.

## Phase 3 — first integration result: verifier-augmented cascade (Lingshu, seed-0 split, clean_dump n=1064)
| dataset | greedy | verifier-bo8 | 32B | cascade(esc->32B) best-acc @ esc% |
| SLAKE   | 0.729 | 0.762 | 0.829 | 0.829 @ 100% (can only match 32B; verifier<32B here) |
| Kvasir  | 0.274 | 0.405 | 0.326 | 0.422 @ 36%  BEATS 32B |
| PathVQA | 0.306 | 0.441 | 0.377 | 0.453 @ 30%  BEATS 32B |
| VQA-RAD | 0.444 | 0.611 | 0.648 | 0.667 @ 37%  BEATS 32B (n=54) |
| POOLED  | 0.385 | 0.501 | 0.462 | 0.517 @ 35%  BEATS 32B on accuracy |
FINDING: the verifier-augmented cascade BEATS Lingshu-32B on accuracy (pooled 0.517 vs 0.462) escalating ~35%.
COST TENSION (the key open problem): best-of-8 ~= 16 7B-fwd + 0.35*(32B~=4.6 7B-fwd) ~= 17.6 vs always-32B ~= 4.6
  => ~3.8x the FLOPs. So it WINS accuracy, LOSES FLOPs. SLAKE only matches (verifier 0.762<32B 0.829 -> escalate-all).
CAVEAT: seed-0 split (favorable); 2-seed range applies (verifier 0.501/0.445). Per-dataset WIN pattern (kvasir/pathvqa/vqa_rad) is robust.
NEXT: cost-frontier search — vary N {1,2,4,8}, cheaper verifier, smarter gate (verifier-conf vs 7B-conf vs CASP) — find any config
  that beats 32B on accuracy AND FLOPs; + the integration variants (router/unified/agreement-on-picks); + held-out (radimagenet).

## Phase 1 RESULT — best cascade gate (Lingshu, clean_dump n=1064)
escalation signal -> best cascade acc @ esc%: VERIFIER-CONFIDENCE 0.518 @ 34% (BEST) > self-consistency 0.501 @ 0% (no gain) > n_distinct 0.501 @ 0%.
=> the trained verifier's own confidence is the best escalation gate; SC/answer-diversity don't beat the verifier-alone. (7B-seqlogprob + CASP-stability gates pending the captured cheap-leg + training.)
RUNNING (batch3): GPU0 = Lingshu verifier full-set scoring (vqa_rad/pathvqa/kvasir + radimagenet held-out); GPU1 = MedVLThinker verifier training (4 in-dist, radimagenet held-out).

## Phase 2/3 — verifier on FULL sets + held-out + cross-family (batch3, 2026-06-29)
Lingshu verifier (full per-dataset): VQA-RAD 0.575 (32B 0.600), PathVQA 0.453 (32B 0.376 BEAT), Kvasir 0.439 (32B 0.301 BEAT),
  RadImageNet-HELD-OUT 0.353 (greedy 0.329 -> generalizes OOD; 32B pending). MedVLThinker verifier: slake 0.583, vqa_rad 0.456,
  kvasir 0.424 (32B 0.361 BEAT), pooled 0.476. => verifier works + generalizes in BOTH families; beats the (weak/competent) 32B where the 32B is weak.
NEXT: InternVL3-8B/38B (download+baseline+verifier); full-set cascade + integration variants (agreement-on-picks).

## Held-out OOD (RadImageNet) — verifier generalizes AND beats the 32B
Lingshu-32B 0.289 | Lingshu-7B greedy 0.329 | 7B+verifier 0.353 | oracle@8 0.512.
=> on a dataset the verifier was NEVER trained on, 7B+verifier (0.353) BEATS Lingshu-32B (0.289); the 32B doesn't even beat
   the 7B greedy here. Strong generalization + cross-size win on OOD.
RUNNING: batch4 = per-sample verifier dumps (both families) for the full-set cascade + cross-family + agreement-on-picks;
  InternVL3-8B/38B downloading (~28%). Monitor set.
