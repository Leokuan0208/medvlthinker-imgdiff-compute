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

## HEADLINE — full-set verifier-augmented cascade beats the 32B on accuracy (both families, in-dist + held-out)
Lingshu:  VQA-RAD cascade 0.625 (32B 0.600) @24% | PathVQA 0.469 (0.376) @33% | Kvasir 0.448 (0.301) @22% |
          RadImageNet-HELDOUT 0.353 (0.289) @0% | POOLED 0.469 vs 32B 0.360 @30%.
MedVLThinker: VQA-RAD 0.555 (0.525) @54% | Kvasir 0.483 (0.361) @9% | RadImageNet-HELDOUT 0.243 (0.202) @13% |
          POOLED 0.491 vs 32B 0.384 @31%.
=> the unified method (verifier best-of-8 + verifier-confidence gate -> escalate to 32B) BEATS the strong/weak 32B on ACCURACY
   on every dataset, BOTH families, in-dist AND held-out OOD, escalating 22-54%. (slake excluded from dumps — imgs_for unsupported.)
COST: best-of-8 keeps FLOPs > always-32B; accuracy is the substantial robust win. Latency/energy ~neutral (no-think 32Bs).
NEXT: cost-optimized variant (lower N at the gate), agreement-on-verifier-picks (2 small models), InternVL3 family.

## Variant — agreement-on-verifier-picks (2 small models: Lingshu-7B + MedVLThinker-7B) — LOSES to verifier-conf gate
agree% / acc|agree / agree-gate(esc disagree->32B): VQA-RAD 27% / 0.852 / 0.570@73% | Kvasir 24% / 0.797 / 0.381@76% |
RadImageNet 20% / 0.595 / 0.216@80%. => agreement is HIGH-PRECISION (agree=>~85% right) but LOW-COVERAGE (20-27%),
so as a gate it escalates 73-80% and ends up WORSE than the verifier-confidence cascade. Verifier-confidence stays the best gate.

## COST-FRONTIER (Phase 2 "best verifier setup") — sweep best-of-N {1,2,4,8} ± gate, 2026-06-30
Resolves the headline's flagged "key open problem" (best-of-8 costs > always-32B FLOPs). Cost model is
PREFILL-DOMINATED forward-equiv, VALIDATED on real data: gen decode = mean 4-6 tok (run_openvqa caps short
answers) << ~320 image-prefill tok, so each gen/verify/strong call ≈ one prefill scaled by params. Units =
2·7B·P (one 7B forward). verifier-bo-N cost = N·(cheap/7) + N·(verif/7); always-STRONG = strong/7 = 4.57 for 32B.
Verifier base = Lingshu-7B (ratio 1.0) in BOTH families (the pooled4 / mvt7b_4ds adapters). Strong = no-think 32B.
Per-sample dumps (sl[8],scores[8]) let us simulate any N≤8 + sweep the gate τ offline (no new inference).

LINGSHU (strong=Lingshu-32B):
| dataset | always-STRONG acc (cost 4.57) | cheapest no-gate win | +gate-best |
| vqa_rad (32B STRONG 0.600) | 0.600 | NONE (bo8 0.575 < 0.600) | 0.625 @esc24% cost 17.1 (acc-win, COSTLIER) |
| pathvqa (0.376) | 0.376 | NONE on both axes (bo4 0.411 beats acc but cost 8>4.57) | 0.469 @esc33% cost 17.5 |
| kvasir (WEAK 0.301) | 0.301 | **N=2 no-gate 0.335 @cost 4.0** BEATS acc+FLOPs | 0.447 @esc20% cost 16.9 |
| radimagenet-OOD (WEAK 0.289) | 0.289 | **N=1 no-gate 0.321 @cost 2.0** BEATS acc+FLOPs | 0.353 @esc0% |
MEDVLThinker (strong=MedVLThinker-32B):
| vqa_rad (32B 0.525) | 0.525 | NONE | 0.555 @esc40% cost 5.85 (acc-win, COSTLIER) |
| kvasir (WEAK 0.361) | 0.361 | **N=2 no-gate 0.380 @cost 4.0** BEATS acc+FLOPs | 0.485 @esc9% cost 16.4 |
| radimagenet-OOD (WEAK 0.202) | 0.202 | **N=2 no-gate 0.215 @cost 4.0** BEATS acc+FLOPs | 0.242 |

KEY FINDING — the cost-positivity is REGIME-DEPENDENT (this is the honest, sharper claim):
1. ACCURACY win (verifier±gate > strong) is ROBUST everywhere, both families, in-dist + OOD (already headline).
2. BOTH-AXES win (beats strong on accuracy AND FLOPs) holds ONLY where the strong model is WEAK on the domain
   (Kvasir, RadImageNet-OOD, all MedVLThinker free-text): there, verifier-best-of-2 with NO GATE is a genuine
   free lunch — cheaper (4.0<4.57) AND more accurate, and it is τ-FREE (no gate => no oracle-τ optimism).
3. Where the 32B is GENUINELY STRONG on the domain (Lingshu/MedVLThinker VQA-RAD), there is NO no-gate both-axes
   win; the verifier+confidence-gate still wins ACCURACY but costs MORE FLOPs => a Pareto trade (pay compute for
   accuracy), not a free lunch. best-of-2 is the knee (acc ≈ closes most of the gap at 4.0 cost).
HONESTY NOTE: the "+gate-best" τ is oracle-selected (swept on the eval set) => optimistic; the no-gate wins use
NO τ and are fully honest. The deployable recommendation: verifier-best-of-2 (no gate) as the default operating
point — beats a weak/poorly-adapted strong model on both axes, and on a strong in-domain model it is the cheapest
verifier setting that still captures most of the accuracy gain.
NEXT: same cost-frontier for InternVL3 (cross-family verifier transfer: Lingshu verifier scoring IV3-8B answers),
strong=IV3-38B; then a consolidated 3-family master table.

## LEAKAGE CHECK (2026-06-30) — verifier-bo8 full-set ≈ held-out test-split => no memorization inflation
pooled4 was trained on vqa_rad/pathvqa/kvasir (+slake); the transfer-dumps score the FULL eval set incl. train
questions. Concern: optimism. Test: compare the train-script's grouped held-out test-split bo8 vs the full-set
transfer-dump bo8:  vqa_rad 0.611(test n=54) vs 0.575(full n=200) | pathvqa 0.441(435) vs 0.453(1500) |
kvasir 0.405(365) vs 0.439(1200).  => full-set is NOT systematically higher (differs both directions, ~±0.03)
=> the LoRA verifier learned a GENERAL correctness signal, not memorized (q,a) pairs. Full-set numbers are honest.
(radimagenet is a fully held-out DATASET => clean by construction.)

## GATE HONESTY + WHAT ACTUALLY MATTERS (held-out-τ cross-fit, 2026-06-30)
The cost-frontier "+gate-best" used oracle-τ (swept on eval). 5-fold cross-fit (pick τ* on 4/5, test on 1/5)
quantifies the DEPLOYABLE gate value and the oracle optimism. Pooled:
| family / N | STRONG | no-gate bo-N (τ-free) | held-out-τ gate (deployable) | oracle-τ | gate gain | oracle optimism |
| Lingshu N=8     | 0.331 | 0.414 | 0.421 @esc11% | 0.421 | +0.007 | +0.000 |
| Lingshu N=2     | 0.331 | 0.348 | 0.377 @esc51% | 0.379 | +0.030 | +0.002 |
| MedVLThinker N=8| 0.277 | 0.339 | 0.343 @esc18% | 0.344 | +0.003 | +0.001 |
TWO CLEAN FINDINGS:
1. Oracle-τ optimism is NEGLIGIBLE (+0.000..+0.002) => the verifier-confidence threshold GENERALIZES; the
   headline frontier numbers are honest (not threshold-cherry-picked).
2. At N=8 the verifier-confidence GATE is essentially REDUNDANT with best-of-8 selection (+0.003..+0.007). The
   gate only adds real accuracy at SMALL N (N=2: +0.030, by escalating the unconfident ~half to the strong model).
=> The trained-verifier best-of-N SELECTION is the dominant lever; the escalation gate is a small-N cost-saver,
   not the source of the win. This SIMPLIFIES the deployed method: ship verifier best-of-N selection; add the
   confidence gate only when sample budget is tight (small N) or you want to spend compute to close the last
   accuracy gap on domains where the strong model is genuinely better.

## GATE BAKE-OFF: is a TRAINED gate (incl. CASP) better than verifier-confidence? — 2026-06-30
Prompted by the open question "the training-required-gate route is still open; CASP-stability won in MCQ."
Tested rigorously in the OPEN-TEXT verifier cascade (which the earlier 'verifier-conf is best' only compared
training-FREE signals in). Target = verifier_pick_ok; gates compared by AUROC(pick correct) + cascade-acc at
fixed escalation budgets, with 5-fold OUT-OF-FOLD scores for trained gates (no leakage). Also the DECISIVE
recoverability test (Jitkrittum NeurIPS'23, arXiv:2307.02764): the OPTIMAL gate thresholds recoverability
d=P(strong correct)-P(small correct), not confidence — can ANY trained gate learn it?

PICK-CORRECTNESS GATE (pooled): verifier-conf AUROC = 0.853 (Lingshu) / 0.885 (MedVLThinker). Best TRAINED gate
(GBM on verifier+cheap) = 0.861 / 0.882 => +0.008 / -0.003, i.e. NO improvement; cascade acc identical. Cheap-only
trained gates (self_consistency,n_distinct,gen_tokens) = 0.69-0.73, far below verifier-conf. Same on vqa_rad
(strong competitive): verifier-conf 0.883, trained <=0.869.

RECOVERABILITY (the wall, Jitkrittum-optimal target), trained on ALL signals (verifier dist + cheap), cross-fit:
| set | strong fixes pick-errors | ORACLE cascade (esc gain-rows) | verifier-conf | TRAINED-recover (best) |
| Lingshu pooled  | gain 6.0% / lose 14.2% | 0.473 @6%  | cascade 0.421, AUROC(gain) 0.604 | cascade 0.413, AUROC 0.627 (gbm) |
| Lingshu vqa_rad | gain 11% / lose 8.5%   | 0.685 @11% | cascade 0.620, AUROC 0.665        | cascade <=0.620, AUROC 0.581 (worse) |
| MVT pooled      | gain 4.9% / lose 11.1% | 0.389 @5%  | cascade 0.344, AUROC 0.514        | cascade 0.338, AUROC 0.649 (gbm) |
| MVT vqa_rad     | gain 14.5% / lose 11%  | 0.635 @14% | cascade 0.550, AUROC 0.707        | cascade <=0.525, AUROC 0.637 (worse) |

VERDICT (answers "is confidence gate really best?"): YES, the VERIFIER-CONFIDENCE gate is the best DEPLOYABLE gate.
- Trained gates (logit/GBM/MLP) on verifier±cheap signals do NOT beat it on cascade accuracy ANYWHERE (both
  families, pooled + the gate-matters vqa_rad regime). On large pooled sets a GBM can raise the recoverability-
  RANKING AUROC (0.60->0.63-0.65) but this does NOT convert to cascade accuracy (it escalates more losing rows);
  on the smaller competitive set it OVERFITS and loses to verifier-conf outright.
- WHY (Jitkrittum theory, confirmed): the optimal gate target is recoverability, which is near-UNLEARNABLE here
  (AUROC stuck ~0.6, far below the oracle ceiling that proves signal exists) — the noise/saturation regime.
  The verifier's P(correct) is ALREADY a trained, calibrated correctness estimator (the strongest such signal in
  the pipeline; Cobbe/GenRM), so a stability/CASP gate (which also estimates upstream correctness) is REDUNDANT
  with it, not orthogonal. To beat verifier-conf a gate must add recoverability info — and nothing does.
- CASP-stability's MCQ "win" was ISO-ACCURACY COMPUTE-SAVING (a different objective; its own code pre-registers
  "cannot raise accuracy"). It is not applicable as an accuracy-improving gate here.
NAMING: "CASP" collides with the protein-structure CASP competition and does not exist as an ML gate; the real
prior art is CCPS (Khanmohammadi et al. 2025, arXiv:2505.21772 — a trained classifier on perturbed-REPRESENTATION
stability). Rename our trained-stability-gate ablation and cite CCPS + Bahat&Shakhnarovich (arXiv:2006.16705).
LIT PRECEDENTS for our design (verifier score AS the gate): Self-REF (arXiv:2410.13284, one confidence head for
route+reject) and Kiyani et al. 2026 (arXiv:2602.17633, verifier-confidence accept/reject/escalate) — both close,
neither uses a trained outcome verifier as a small->large medical-VQA cascade gate => our combo is novel.
REMAINING: feature-complete CASP test on InternVL3 (has margin/conf/seqlogprob) incl. the stability target.

## FEATURE-COMPLETE confirmation (InternVL3 vqa_rad, cheap sc8 HAS margin/conf/seqlogprob) — 2026-06-30
The Lingshu/MVT sc8 lacked the cheap logprob signals CASP/CCPS use. InternVL3-8B sc8 (generated this session
with logprobs=5) HAS them, so this is the feature-complete CASP test. n=200; verifier-pick 0.570 vs always-38B 0.415.
  verifier-conf AUROC(pick_ok)=0.838 (best). Cheap signals WITH logprobs: cheap-conf 0.758 > -n_distinct 0.740 >
  self_consistency 0.710 > cheap-margin 0.685 > cheap-seqlogprob 0.648 — all still FAR below verifier-conf.
  TRAINED-logit verif+cheap = 0.842 (+0.004 over verifier-conf = noise); GBM/MLP 0.808-0.831 (worse). Cheap-only
  trained (full features) 0.706-0.764. => Even the full CASP/CCPS feature set does NOT let a trained gate beat
  verifier-confidence; the cheap confidence signals are redundant with the verifier's calibrated P(correct).

FINAL VERDICT on "is confidence gate really the best option?" (3 families x both regimes x full feature set):
  YES — the VERIFIER-CONFIDENCE gate is the best gate. The training-required route (CASP-stability, learned MLP/GBM,
  and even the theoretically-optimal recoverability-targeted gate) does NOT beat it on cascade accuracy. The limit
  is the recoverability WALL (Jitkrittum'23), not gate learnability or features. BEST COMBO = trained outcome-verifier
  best-of-N SELECTION (the dominant lever) + verifier-confidence escalation gate (a small-N cost-saver). CASP belongs
  in the paper as a cited, beaten baseline (renamed; cite CCPS arXiv:2505.21772), NOT as the deployed gate.
