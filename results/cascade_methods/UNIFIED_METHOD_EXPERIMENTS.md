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

## InternVL3 (3rd family, CROSS-FAMILY verifier) — completes the 3-family generality claim, 2026-06-30
The Lingshu-trained verifier (pooled4) scores InternVL3-8B's answers (cross-architecture transfer, no IV3 verifier
trained). Cascade IV3-8B+verifier -> IV3-38B vs always-38B:
| dataset | SC/greedy | verifier-bo8 | IV3-38B | oracle@8 | cascade-best @esc |
| vqa_rad         | 0.445 | 0.570 | 0.415 | 0.620 | 0.580 @12% BEATS |
| pathvqa         | 0.081 | 0.116 | 0.096 | 0.192 | 0.125 @52% BEATS (both near-floor; IV3 weak free-text) |
| kvasir          | 0.362 | 0.479 | 0.380 | 0.593 | 0.487 @17% BEATS |
| radimagenet-OOD | 0.285 | 0.302 | 0.304 | 0.398 | 0.313 @52% BEATS (cross-family verifier TIES then edges the 38B on held-out OOD) |
| POOLED          | 0.202 | 0.249 | 0.218 | 0.337 | 0.255 @36% BEATS |
=> 3rd family confirms: verifier best-of-8 + verifier-conf gate BEATS the strong 38B on accuracy, every dataset +
   held-out OOD. AND the verifier transfers ACROSS ARCHITECTURES (trained on Lingshu-7B, scores InternVL3-8B) — a
   stronger generality statement than a same-family verifier.

## InternVL3 feature-complete gate bake-off (pooled n=6757, full signals) — 3rd confirmation of the gate verdict
verifier-conf AUROC(pick_ok)=0.875 (best). Best TRAINED gate (gbm verif+cheap)=0.879 (+0.004=noise). Cheap-only
WITH full logprob signals (cheap-conf 0.753, cheap-margin 0.712, cheap-seqlogprob 0.685) = 0.78 trained — still far
below verifier-conf. Recoverability wall: strong fixes 6.1% of pick-errors, verifier-conf AUROC 0.367 (<chance).
=> SAME verdict on the largest sample with full features: verifier-confidence is the best gate; no trained gate beats it.

## MEASURED batch-1 latency + energy per tier (HF, clean per-GPU NVML), 2026-06-30
| tier | latency (ms, mean) | energy (J) | prefill tok | gen tok |
| Lingshu-7B generate  | 347 | 45.8  | 327 | 5.6 |
| Lingshu-7B VERIFY    | 175 | 25.3  | 362 | 1.0 |  (one forward ~= 0.5x a generation)
| MedVLThinker-7B gen  | 246 | 31.1  | 327 | 3.7 |
| Lingshu-32B generate | 665 | 126.9 | 327 | 5.6 |
| MedVLThinker-32B gen | 633 | 118.4 | 327 | 5.1 |
KEY: at batch-1, the 32B is only ~1.9x the 7B's LATENCY and ~2.8x the energy (NOT 4.6x like FLOPs) -- batch-1 is
bandwidth/overhead-bound, not compute-bound, at these tiny gen-token counts. CONSEQUENCE for cost:
- verifier-bo-N latency (SEQUENTIAL) = N*(gen7 347 + verify7 175) + esc*gen32 665 ms. N=2,esc=0 => 1044ms vs
  always-32B 665ms => best-of-N is SLOWER at strict batch-1 sequential. BUT the N samples are embarrassingly
  parallel: BATCHED, best-of-N ~= 1 gen-call + 1 verify-batch ~= 0.5-0.6s ~ NEUTRAL vs the 32B. So the FLOPs win
  (verifier-bo2 < always-32B FLOPs) is real; the LATENCY/energy is ~neutral-to-slightly-worse and depends on batching.
  HONEST framing: the method's win is ACCURACY (robust) + FLOPs (regime-dependent); latency/energy ~neutral.

## FAITHFUL CASP/CCPS test: input-perturbation VISUAL-STABILITY as a gate (Lingshu, 2026-06-30)
The bake-off tested trained gates on existing signals; CASP/CCPS is specifically STABILITY-UNDER-PERTURBATION.
Generated Lingshu-7B answers at perturbed resolutions (cap160, cap80) via vLLM; visual-stability = answer agreement
with the cap320 modal answer. Input-perturbation analog of CCPS (arXiv:2505.21772, perturbs representations) /
Bahat&Shakhnarovich (arXiv:2006.16705, TTA-consistency). n=4900; raw agree@cap160=0.583, @cap80=0.572.
  AUROC(pick_ok): verifier-conf 0.853 | visual-stability-frac (pure CASP gate) 0.604 | vstab trained(160+80) 0.597 |
  verifier-conf + stability (trained) 0.852.
=> visual-stability is a WEAK gate (0.60, like self-consistency) and adds NOTHING to verifier-conf (0.852 vs 0.853).
   The faithful CASP/CCPS signal is REDUNDANT with the verifier (both estimate upstream correctness) — final
   confirmation that the trained-stability-gate route does not beat verifier-confidence. (3-family extension running.)

## 3-FAMILY visual-stability confirmation (2026-06-30) — CASP/CCPS redundant in ALL families
verifier-conf vs pure-CASP visual-stability(frac) vs verifier+vstab (AUROC for pick_ok):
| family | verifier-conf | pure-CASP vstab | verifier+vstab | adds? |
| Lingshu      | 0.853 | 0.604 | 0.852 | no |
| MedVLThinker | 0.885 | 0.568 | 0.883 | no |
| InternVL3    | 0.875 | 0.605 | 0.875 | no |
=> The faithful input-perturbation stability gate (CCPS/CASP-style) is a WEAK gate (0.57-0.61) and adds NOTHING to
   verifier-confidence in every family. FINAL, EXHAUSTIVE confirmation: verifier-confidence is the best gate; the
   trained/stability-gate route does not beat it. Scripts: src/cascade_methods/open_vstab_gate.py (+perturb gen via
   run_openvqa --cap cap160/cap80 into ckpts/openvqa/*_perturb).

## OPEN FORWARD POINTER (where the real headroom is — the VERIFIER, not the gate)
oracle@8 >> verifier-bo8 everywhere (Lingshu pooled 0.513 vs 0.414; MVT 0.416 vs 0.339; IV3 0.337 vs 0.249) => the
binding limit on accuracy is the VERIFIER's selection quality, not the gate. Highest-EV next direction = a BETTER
verifier (more/cleaner training data, per-family or larger base, process-style signals), which raises the whole
cascade. The gate is settled (verifier-confidence); compute spent on gate variants is confirmatory.

## VERIFIER IMPROVEMENT (2026-06-30) — diagnostic + experiments
User directive: improve the verifier (the real accuracy headroom; oracle@8 >> verifier-bo8).
SELECTION-RULE bake-off (free, no training): argmax-verifier ~= weighted_mean (best); adding answer-frequency
(SC) HURTS (weighted_sum/verif_x_freq < argmax). => no free win from the selection rule; the verifier's SCORES
must improve.
DIAGNOSTIC (per-answer verifier quality vs selection efficiency):
| family | per-answer AUROC | mean P(Yes) correct/wrong (sep) | selection efficiency (argmax picks the correct one | recoverable Qs) |
| Lingshu      | 0.903 | 0.715 / 0.176 (0.538) | 81% (2027/2514) |
| MedVLThinker | 0.913 | 0.795 / 0.236 (0.560) | 82% (1154/1414) |
| InternVL3    | 0.898 | 0.676 / 0.141 (0.535) | 74% (1688/2277) |
=> The verifier is GOOD per-answer (AUROC ~0.90) but selection efficiency is only 74-82% (= verifier-bo8/oracle@8
   ratio). The headroom is entirely the within-question NEAR-TIES where a wrong answer outscores the correct one.
   IMPLICATION: the verifier is trained POINTWISE (each answer labeled independently) but the task is RANKING within
   a question => a within-question contrastive/ranking objective (Bradley-Terry) is the most aligned lever.
TRAINING DATA: only 6000 of 14635 judged pairs were used (1 epoch). Retraining on all 10364 TRAIN pairs (pos rate
0.194, imbalanced) x 2 epochs, r=16 (lora_verifier_pooled4_v2) and r=32 (lora_verifier_pooled4_r32); same seed-0
held-out 1064-Q split for fair comparison vs pooled4 baseline. [results pending]
NEXT LEVERS (ranked): (1) within-question ranking/contrastive loss (targets selection efficiency directly);
(2) class balancing (pos rate 0.194); (3) CoT/generative verifier (GenRM) if rationale data can be synthesized.

## VERIFIER IMPROVEMENT — RESULTS + ceiling analysis (2026-07-01)
Held-out 1064-Q split (identical), POOLED selection accuracy (trained-verify) + per-answer AUROC:
| verifier | selection acc | per-answer AUROC |
| baseline pooled4 (6k pairs, 1ep, pointwise) | 0.5009 | 0.903 |
| v2 (10.4k pairs, 2ep, pointwise)            | 0.5056 (+0.005, ~noise) | - |
| ranking clean λ=0.5 (BCE + within-Q Bradley-Terry) | 0.5009 (+0.000) | 0.931 |
| ranking clean λ=1.0                          | 0.4991 (-0.002) | 0.933 |
KEY RESULT: the ranking loss makes the verifier SHARPER per-answer (AUROC 0.903 -> 0.931/0.933, a real global-
discrimination gain) but SELECTION accuracy stays flat (~0.50). Better global discrimination does NOT convert to
within-question selection => the selection failures are on genuinely-hard near-ties, not undertraining.

CEILING ANALYSIS (inspection of selection-failures, oracle-right but verifier-picks-wrong): the judge-'correct'
answers ARE legit (semantic matches to gold that a crude string-match misses, e.g. "Right cerebellum"~="right
posteroinferior cerebellum"; "Both"~="bilateral lungs") => NOT judge noise; the oracle ceiling is real. The
failures are COMPOUND multi-part free-text answers (esp. Kvasir: "what color, what instrument, where") where the
verifier's pick is fluent but wrong on one component. Distinguishing them needs fine-grained visual grounding of
each sub-claim -- beyond a cheap 7B pointwise/ranking verifier.

CONCLUSION on "improve the verifier": the cheap 7B verifier is near its PRACTICAL SELECTION CEILING on these
candidates. Levers tried: more data (+0.005 noise), more epochs, within-question ranking (AUROC up, selection
flat). The one genuine gain is a better-CALIBRATED verifier (ranking λ=0.5, AUROC 0.931) -- useful for the
verifier-confidence GATE, though the recoverability wall caps how much the gate can add. Selection headroom to
oracle (0.50->0.59) is intrinsic (compound-answer grounding difficulty), not capturable by training a cheap
verifier harder. HIGHER-EV levers (not "train the cheap verifier more"): (a) larger verifier base (more grounding,
costs efficiency); (b) STRUCTURED/decomposed verifier that grades each sub-claim of compound answers; (c) better
CANDIDATES (generator side: cleaner/more-distinct answers, higher N) so a clearly-best answer exists to select.
Deployable pick: keep the pointwise verifier (or ranking-λ0.5 for a sharper gate); best-of-N selection remains
the dominant lever and it is at ceiling for the cheap-verifier regime.

## CORRECTION (stratified) — the selection ceiling is BROAD, not compound-specific (2026-07-01)
Selection efficiency by gold-answer length (Lingshu, among oracle-recoverable Qs): short(<=3w) 79% (n=1928),
medium(4-8w) 90% (n=343), long(>8w compound) 80% (n=243). Accuracy gap to oracle: short 0.117 (n=3463, the BULK),
medium 0.048, long 0.068. => The earlier "compound answers" framing was anecdotal; the stratified data shows the
verifier is broadly ~80% selection-efficient and the LARGEST volume+headroom is SHORT terse medical answers (e.g.
"Right." vs "Both." vs "Left." — anatomical/finding one-liners the cheap verifier can't reliably ground to the
image). Confirms the ceiling is intrinsic grounding difficulty across answer types, not a fixable training/format
issue for a cheap 7B verifier.

## DEFINITIVE: verifier ceiling is INTRINSIC, not capability-bound (32B zero-shot diagnostic, 2026-07-01)
Scored the SAME 8 candidates with a 32B ZERO-SHOT verifier vs the trained 7B verifier, n=600 sampled Qs:
  selection accuracy: 7B-trained=0.403  32B-zeroshot=0.355  oracle@8=0.490
  selection efficiency: 7B=0.810  32B=0.717  => 32B delta = -0.048 (WORSE)
=> A bigger verifier does NOT help; the trained 7B BEATS the zero-shot 32B. TWO conclusions: (1) task-TRAINING >>
   SIZE for the verifier (a task-trained 7B > a zero-shot 32B); (2) the selection ceiling is INTRINSIC -- if the
   candidate answers were easy to ground to the image, even a bigger verifier would pick them, but it can't.
CAVEAT (honest): the 32B was zero-shot (not LoRA'd on the verify task); a trained 32B might edge the 7B, BUT a 32B
   verifier costs ~as much as the strong model => not cost-sensible for a cheap-model cascade (you'd just use the
   32B's own answer). Within the DEPLOYABLE cheap-verifier regime, the trained 7B is near-optimal.

## FINAL CONCLUSION — "improve the verifier"
The cheap 7B verifier is at its practical ceiling: training levers (more data +0.005; ranking: per-answer AUROC
0.90->0.93 but selection FLAT) and a bigger (zero-shot 32B) verifier do NOT raise selection accuracy. The ceiling
is real (not judge noise) and intrinsic (image-grounding of terse medical answers). => The binding limit on cascade
accuracy is now CANDIDATE QUALITY (the cheap generator's 8 samples), not the verifier. HIGHEST-EV next direction
(generator-side, a scope change from "the verifier"): raise oracle@N via (a) more candidates (best-of-16/32 -> more
chances the correct answer appears), (b) better sampling (temperature/diversity, or diverse prompts), (c) a stronger
cheap generator. Deployable verifier stays: trained 7B pointwise (or ranking-lambda0.5 for a sharper confidence gate).

## CANDIDATE-QUALITY sweep (2026-07-01) — the binding limit is candidate quality; testing every generator-side lever
Prior finding: cheap 7B verifier at ceiling; oracle@N is the wall. Testing what raises oracle@N.

### EXP1 (FREE, CPU): cross-model candidate POOL — pool candidates from 3 cheap models (Lingshu-7B+MVT-7B+IV3-8B)
| dataset | oracle@8 Lingshu / MVT / IV3 | POOLED@24 | gain over best single |
| vqa_rad         | 0.630 / 0.600 / 0.620 | 0.780 | +0.150 |
| kvasir          | 0.491 / 0.550 / 0.593 | 0.731 | +0.138 |
| radimagenet-OOD | 0.512 / 0.317 / 0.398 | 0.625 | +0.113 |
=> Diverse cheap models get DIFFERENT questions right; the union raises oracle by +0.11..+0.15 (huge). Candidate
   DIVERSITY (cross-model) is a major lever. Capturing it needs the verifier to select from the pool (EXP2).
RUNNING: EXP2 = Lingshu verifier scores the pooled candidates -> pooled SELECTION accuracy (GPU0). EXP3 = candidate
N-scaling (sc32), higher-temp (t=1.0), and think-mode reasoning candidates for Lingshu-7B, then judge -> oracle@N.

## USER Q — LLM-judge trustworthiness (2026-07-01)
Anchor validation of the MedVLThinker-32B judge on 21477 Lingshu candidate pairs (overall judge=Yes 17.7%):
  exact-match (ans==gold, n=1277): judge=Yes 100.0% (SHOULD ~100 ✓) | containment (n=2423): 82.5% (sensible) |
  ZERO word-overlap (n=14320): judge=Yes 6.3% (SHOULD ~0; the 6.3% are mostly legit synonyms "Both"~="bilateral",
  "Right cerebellum"~="right posteroinferior cerebellum"). => judge does SEMANTIC grading (correct), well-calibrated
  at the anchors. TODO (GPU): independent 2nd-judge (different family, e.g. Lingshu-32B) agreement/kappa on a sample.

## USER Q — does a better GATE lift the MCQ part? (MCQ gate bake-off on existing ckpts, competent-4, n=6050)
7B-cap320 acc=0.622, 32B-think acc=0.645 (gap only +0.023 => MCQ nearly SATURATED). Recoverability: among 7B-wrong
(2286), 32B-right rate 0.416, margin AUROC for recoverability 0.578 (the WALL, ~chance). Gates (AUROC for 7B-correct
| cascade-acc @esc20/30/40): margin(deployed) 0.667 | 0.639/0.645/0.648 ; maxprob 0.677 | 0.634/0.641/0.647 ;
TRAINED-logit 0.688 | 0.627/0.635/0.644 ; TRAINED-gbm 0.675 | 0.627/0.638/0.641.
=> A better gate does NOT lift MCQ: trained gates get marginally higher AUROC but do NOT beat the margin gate on the
   cascade (margin best at practical esc). Same story as open-text (gate near-optimal, recoverability wall). MCQ is
   nearly saturated (7B~32B) so the cascade MATCHES the 32B (0.648@40%esc slightly exceeds 0.645) but can't
   meaningfully BEAT it. => the method's BEAT-the-strong-model win is intrinsically an OPEN-TEXT phenomenon
   (un-saturated); MCQ contributes efficiency (match 32B at reduced compute), not an accuracy lift. Consistent with
   the earlier "MCQ gate is saturated = benchmark artifact" finding.

## EFFICIENCY-LEG gate bake-off (measured latency/energy, 2026-07-01) — SOTA trained gates vs verifier-conf
User focus: efficiency (latency/FLOPs/energy) at ISO-ACCURACY, not accuracy. Measured batch-1 (Lingshu):
gen7=347ms/45.8J/1.0FLOP, verify7=175ms/25.3J/1.0, gen32=665ms/127J/4.57. Cascade = verifier-boN + gate->32B.
Metric: min escalation to MATCH always-strong accuracy -> resulting FLOPs/energy/latency.

VQA-RAD (strong 0.600 COMPETITIVE > verifier-bo8 0.575, so gate must escalate), N=8, target=0.600:
| gate | esc@iso-acc | FLOPs | energy(J) | lat batched(ms) |
| VERIFIER-CONF | 8%  | 16.4 | 580 | 579 |  <- most efficient gate
| trained-logit(pickok) | 12% | 16.6 | 585 | 605 |
| trained-gbm(pickok)   | 16% | 16.8 | 590 | 632 |
| SOTA Diff-Prob/Jitkrittum(gbm) | 34% | 17.5 | 611 | 745 |
| margin | 64% | 18.9 | 650 | 948 |
| self-consistency | 98% | 20.5 | 694 | 1177 |
N=2, VQA-RAD: trained-gbm 44% (FLOPs 6.0) slightly beats verifier-conf 52% (6.4); others worse.
=> VERIFIER-CONFIDENCE is the most EFFICIENT gate (fewest escalations at iso-accuracy); the SOTA post-hoc trained
   gate (Diff-Prob) is WORSE (34% vs 8%). Trained gates give at most a tiny edge at low N. Consistent with the
   accuracy finding: verifier-confidence is the best gate on BOTH accuracy and efficiency.

KEY EFFICIENCY TENSION (must report honestly): best-of-N base cost = 2N cheap-forwards (N gen + N verify).
Break-even vs one 32B forward (4.57 7B-eq): 2N < 4.57 => N<=2. So:
- POOLED (weak/OOD strong, verifier-bo2 0.348 > strong 0.331): at N=2, 0% escalation, cascade cost FLOPs=4.0<4.57,
  latency(batched)=522<665ms, energy=142~127J => BEATS always-strong on FLOPs+latency (~parity energy) AND accuracy.
- VQA-RAD (competitive strong): best-of-N base overhead makes FLOPs/energy > always-32B; only batched-LATENCY wins
  (N=8 verifier-conf 579<665ms). => the FLOPs/energy efficiency win is a WEAK/OOD-strong regime phenomenon; on a
  competitive strong model the method trades compute for the accuracy lift.
DEPLOYABLE EFFICIENCY PICK: verifier best-of-2 + verifier-confidence gate. N=2 is the FLOP break-even; the gate adds
escalation only where it beats the base. Latency wins via BATCHING the N samples (one batched gen + one batched verify).

## CONTROLLED GATE SWAP (2026-07-01) — fix verifier+selection, swap ONLY the gate; does anything beat confidence?
Deferral curve (cascade acc vs escalation%) + AUROC(pick_ok) + ADC (area under deferral curve = gate quality).
Lingshu POOLED (verifier-bo8 fixed=0.414, strong=0.331):
| gate | AUROC | ADC |
| verifier-confidence [CURRENT] | 0.853 | 0.3923  <- BEST |
| TRAINED-gbm (verifier+cheap)  | 0.854 | 0.3914 (ties, -0.001) |
| TRAINED-logit                 | 0.853 | 0.3911 (ties) |
| verifier-mean                 | 0.834 | 0.3855 |
| SOTA Diff-Prob (Jitkrittum)   | 0.708 | 0.3832 |
| self-consistency / -n_distinct / -answer-entropy | ~0.69 | ~0.368 |
| verifier-margin / verifier-negstd | 0.40/0.44 | ~0.36 |
VQA-RAD (gate matters): verifier-conf ADC 0.6042 (best) > verifier-mean 0.6010 > TRAINED-logit 0.6016 > rest.
ANSWER: swapping the confidence gate for another mechanism does NOT improve the cascade. verifier-confidence has
the highest ADC+AUROC in both regimes; its deferral curve dominates (>= every alternative at every budget). A
TRAINED gate on ALL signals only RECOVERS verifier-conf (ties, doesn't beat) because verifier-conf is already the
dominant feature. All simpler signals (margin, self-consistency, answer-entropy) and the SOTA post-hoc recoverability
gate (Diff-Prob) are strictly worse. => the verifier's OWN confidence is the optimal gate; it cannot be improved by
substituting another gate mechanism. Scripts: src/cascade_methods/open_gate_swap.py, open_gate_efficiency.py.

## SOTA trained-gate lit search + reconciliation + setup note (2026-07-01)
SOTA post-hoc trained gate = Jitkrittum NeurIPS'23 (arXiv:2307.02764) Diff-01/Diff-Prob (target 1[32B right]-1[7B
right] = our recoverability). ALREADY TESTED in the controlled swap ("SOTA Diff-Prob"): ADC 0.3832 < verifier-conf
0.3923 => verifier-conf beats the SOTA. FrugalGPT(2305.05176)=our verifier scorer itself; AutoMix(2310.12963)/Gupta
quantile(2404.10136)=learned layers our TRAINED-gbm subsumes (ties verifier-conf); RouteLLM/HybridLLM=query-only
(off-paradigm); CAT/Gatekeeper/Self-REF=require retraining base (not post-hoc). External validation: measured
latency+FLOPs+energy is RARE in this lit (only Semantic Agreement 2509.21837 reports latency) => our measured
efficiency is a real differentiator. Report metrics: deferral curve + escalation%@iso-acc (RouteLLM CPT) + APGR.

RECONCILIATION (trained gate "beat agreement before" but "ties us now" — no contradiction): the trained gate beats
WEAK gates (agreement/self-consistency ADC ~0.368; agreement escalates 73-80%) — STILL TRUE (TRAINED-gbm 0.3914 >>
0.368). It never beat the BEST simple gate: in MCQ it TIED the margin gate (within 0.003); in open-text it TIES
verifier-conf (0.3914 vs 0.3923). Reason: verifier-conf is the dominant input feature, so a learned gate just
recovers it. The comparison BASELINE changed (agreement -> verifier-conf), not the trained gate's quality.

CURRENT MODEL SETUP (open-text, verified gen_tokens~4-6=no-think): cheap small(7B/8B) NO-THINK best-of-N(temp0.7) ->
trained verifier SELECTS -> verifier-confidence gate -> strong big(32B/38B) NO-THINK(t0). TWO tiers, BOTH no-think;
best-of-N selection REPLACES the think tier. This is NOT the MCQ-era ACC 3-tier (small/nothink + big/nothink +
big/think). (The lingshu7b_think candidate variant currently generating is an EXPERIMENT, not the deployed setup.)

## USER Q — why 2 tiers? add the THINK tier for reasoning datasets (MMMU, MedXpert). 3-tier cascade test (2026-07-01)
Think genuinely helps on reasoning MCQ: MMMU 32B nt->think 0.624->0.688 (+0.065); MedXpert-Reasoning 0.279->0.326
(+0.047); MedXpert-Understanding 0.292->0.384 (+0.092). think decodes ~474-694 tok (vs 2 for no-think) => dominant cost.
3-tier margin-gated cascade (7B-nt -> 32B-nt -> 32B-think), min-cost config @ iso-accuracy (match always-32B-think):
| dataset | always-think acc | cascade acc | ->32Bnt | ->think(fire) | FLOPs% vs always-think |
| MMMU                   | 0.688 | 0.688 | 68% | 28% | 78% (latency/energy ~31%, think-decode-dominated) |  <- WIN
| MedXpert-Reasoning     | 0.326 | 0.326 | 100% | 92% | 143% |  <- no win (near-floor cheap tiers -> escalate ~all)
| MedXpert-Understanding | 0.384 | 0.384 | 96% | 96% | 151% |  <- no win
CONCLUSION: the tier structure should be REGIME-ADAPTIVE, not fixed 2-tier:
- Perception VQA (competent-4 + open-text sets): 2-tier NO-THINK (thinking overthinks; best-of-N + verifier is the win).
- Reasoning VQA the cheap model can partly handle (MMMU): gated 3rd THINK tier -> match 32B-think at ~78% FLOPs /
  ~31% latency+energy (think fires on only 28%). This is the ACC result, now shown INCLUDING MMMU.
- Reasoning VQA at the cheap-model FLOOR (MedXpert): cascading CANNOT save compute (7B near-chance => escalate ~all,
  gating overhead makes it costlier than always-think); matches accuracy but no efficiency gain. (Why ACC excluded it.)
=> The 2-tier decision was correct FOR PERCEPTION; MMMU warrants the 3rd tier (efficiency win); MedXpert is the hard
   regime where no cascade helps. Full "beat/match original across the suite" = perception(open-text, beat via
   verifier) + MMMU(3-tier, match at ~1/3 compute) + MedXpert(match, no compute win).

## Candidate-quality (oracle@N) + JUDGE TRUST results (2026-07-01)
ORACLE@N (Lingshu-7B, judged): more samples raise the ceiling -> candidate quality IS the lever.
| dataset | @8 | @16 | @32 | think@8 | temp1.0@8 |
| vqa_rad | 0.625 | 0.675 | 0.700 | 0.660 | 0.645 |
| kvasir  | 0.486 | 0.538 | 0.607 | 0.489 | 0.492 |
| radimagenet | 0.503 | 0.566 | 0.635 | 0.570 | 0.489 |
=> sc32 adds +0.08..+0.13 oracle over sc8; think candidates help on vqa_rad/radimagenet. Verifier headroom grows with
   more/better candidates (consistent with cross-model pool +0.11..+0.15). The binding limit is candidate quality.
JUDGE TRUST (independent 2nd judge Lingshu-32B vs MedVLThinker-32B): vqa_rad agreement 0.984 kappa 0.962;
kvasir agreement 0.958 kappa 0.849. + anchor check (100% exact-match). => LLM judge is TRUSTWORTHY.

## LINGSHU BASELINE — faithful protocol decided (user: BOTH). NGC harness != Lingshu eval.
NGC-harness Lingshu (MedVLThinker-Eval, secondary): 32B VQA-RAD 81.6/SLAKE 89.4/PathVQA 87.0/PMC 64.0/MMMU 62.4/
MedXpert-R 28.4/MedXpert-U 33.2. MATCHES paper on SLAKE(89.2)/MMMU(62.3) but PathVQA(87 vs 65.9) way off + 7B-MMMU
0.853 anomaly => NGC harness NOT faithful (MedVLThinker's reformatted benchmarks). FAITHFUL = MedEvalKit (MMMU-32B
0.633=paper 62.3). Plan: fix MedEvalKit fragility (vllm 0.9 InputProcessingError), run Lingshu-7B/32B baseline +
our cascade methods within MedEvalKit; metrics = accuracy + latency + FLOPs + energy + %-strong-calls + deferral/APGR.

## FAITHFUL Lingshu baseline via MedEvalKit — Stage 1+2 (2026-07-01)
Recipe (WORKS): medeval_venv (vllm 0.9.0.1) + Qwen2_5_VL wrapper + Lingshu weights + datasets_path=hf (cached) +
use_vllm True + TORCHDYNAMO_DISABLE=1 + use_llm_judge False (MCQ exact-match). Per-benchmark processes (one crash
!= abort). Lingshu-32B, no-think:
| benchmark | ours | closed | paper |
| MMMU-Medical-val | 63.3% | - | 62.3 (EXACT MATCH ✓) |
| MedXpertQA-MM | 30.6% | - | 25.7 |
| PMC_VQA | 55.2% | - | 57.9 |
| SLAKE | 34.3% | 85.9% | 89.2 (closed comparable) |
| VQA_RAD | 47.5% | 85.3% | 76.5 (closed comparable) |
| PATH_VQA | FAIL (data not cached) | - | 65.9 |
=> MMMU reproduces the paper exactly; others ballpark. SLAKE/VQA_RAD open-ended halves need the LLM judge (score ~0
   via exact-match) -> use the validated local MedVLThinker-32B judge (kappa 0.85-0.96) to match paper open-ended.
   NGC harness was NOT faithful (MMMU 85 vs 62) -> MedEvalKit is the correct baseline tool. TODO: Lingshu-7B (cheap
   leg), reasoning/think variant (3-tier), PATH_VQA data, open-ended judge, then methods-within-MedEvalKit.

## Overnight (2026-07-01) — Lingshu-7B MedEvalKit baseline + a flagged anomaly
Lingshu-7B (MedEvalKit, faithful): MMMU-Medical-val = 80.0% (vs Lingshu-32B 63.3% on the SAME 150-Q set).
ANOMALY: 7B > 32B on MMMU-Med, reproducible on BOTH harnesses (NGC 85 vs 62; MedEvalKit 80 vs 63). 32B matches
paper (62.3); 7B unverified vs paper. IMPLICATION if real: on MMMU the cheap 7B beats the strong 32B => the
cascade should NOT escalate on MMMU (efficiency win: keep 7B). IF a 7B-run artifact: must fix before trusting MMMU
cascade. TO RESOLVE (post-pipeline): (1) check Lingshu paper 7B MMMU-Med number; (2) per-sample spot-check 7B-right/
32B-wrong on MMMU — genuine or parse-inflated. OVERNIGHT pipeline running: MMMU meta re-runs -> 32B-think + 7B-think
(all 5) -> cascade_medeval.py (2-tier/3-tier + gate sweep, accuracy + measured latency + gen-token FLOPs).

## Faithful baseline VALIDATION + MMMU-7B anomaly diagnosis (2026-07-01, vs Lingshu paper Table 6)
Paper: 7B/32B = MMMU-Med 54.0/62.3, VQA-RAD 67.9/76.5, SLAKE 83.1/89.2, PMC-VQA 56.3/57.9, MedXpert-MM 26.7/25.7.
Ours (MedEvalKit): 
| bench | 7B ours | 7B paper | 32B ours | 32B paper | verdict |
| MMMU-Med | 80.0 | 54.0 | 63.3 | 62.3 | 7B INFLATED +26 ; 32B ✓ |
| SLAKE(close) | 82.5 | 83.1 | 85.9 | 89.2 | ✓ both reproduce |
| PMC_VQA | 54.3 | 56.3 | 55.2 | 57.9 | ✓ both reproduce |
| MedXpert-MM | 26.2 | 26.7 | 30.6 | 25.7 | ✓ (32B +5) |
| VQA_RAD(close) | 78.1 | 67.9 | 85.3 | 76.5 | both ~+10 (protocol) |
=> SLAKE/PMC/MedXpert faithfully reproduce for BOTH sizes -> harness VALIDATED. MMMU-7B is the outlier: spot-check
   confirms it is GENUINE (clean parsing; 7B outputs the correct letter more often than 32B; 7B 0.800 vs 32B 0.633
   on same 150 Qs; replicates across NGC(85) + MedEvalKit(80)). 32B-MMMU matches paper. So it's a 7B-SPECIFIC
   protocol difference (our no-think prompt vs the paper's 7B protocol), NOT a bug in our harness or parser.
   HYPOTHESIS: paper 7B used reasoning/different prompt; the overnight 7B-think MMMU run will test if it -> ~54.
   IMPLICATION: EXCLUDE MMMU from faithful cascade CLAIMS until the 7B protocol is reconciled; rely on
   SLAKE/PMC/MedXpert (reproduce cleanly for both sizes) + VQA_RAD (offset but consistent). (Mirrors the original
   project's MMMU/MedXpert exclusion.) Overnight pipeline continues: 32B-MMMU re-run -> think tiers -> cascade.

## HEADLINE — faithful 2-tier cascade on Lingshu MedEvalKit eval (2026-07-01, overnight)
Clean 2-tier (Lingshu-7B -> Lingshu-32B, margin gate) at ISO-accuracy (match always-32B), closed subsets, per-sample
'correct' field. FLOPs = prefill-dominated (7B=1, 32B=4.57; MCQ decode~0). Latency = measured latency_s.
| benchmark | 7B | 32B | 2-tier | esc% | FLOPs vs 32B | latency vs 32B |
| PMC_VQA (n=33430) | 0.543 | 0.552 | 0.552 | 9%  | -69% | -33% |  <- HEADLINE
| SLAKE-closed (836)| 0.825 | 0.859 | 0.861 | 22% | -56% | -22% |
| VQA_RAD-closed(251)| 0.781| 0.853 | 0.853 | 61% | -17% | +21% (esc-heavy) |
| MedXpert-MM (2000)| 0.262 | 0.306 | 0.307 | 95% | +17% | +60% (no win; 7B near-floor) |
| MMMU (150) | 0.80* | 0.64 | - | - | - | (*7B protocol-inflated, unreliable) |
=> On the FAITHFUL Lingshu eval, the 2-tier cascade MATCHES Lingshu-32B accuracy at large compute savings where the
   7B is competitive: PMC-VQA -69% FLOPs / -33% latency @9% escalation (33k samples); SLAKE -56%/-22% @22%. Mixed on
   VQA-RAD (FLOPs win, latency loss from 61% esc). No win on MedXpert (near-floor) or MMMU (7B inflated). This is the
   classic cascade efficiency claim, now on Lingshu's published protocol. Script: src/cascade_methods/lingshu_medeval_cascade.py.
KNOWN ISSUES to fix: (1) think tier / 3-tier: reasoning=True did NOT engage CoT (gen_toks ~3; MedEvalKit is_reasoning
prompt only asks for \boxed letter, no step-by-step) -> need a real CoT prompt for Lingshu. (2) MMMU-7B protocol
reconciliation. (3) open-ended halves need the local judge; PATH_VQA data.

## Gate-signal variation (MCQ 2-tier, faithful Lingshu) — margin is best
Min FLOPs @ iso-32B accuracy by gate signal (closed subsets): PMC_VQA margin 1.41(-69%) < conf 1.55(-66%) <
cum_logprob 1.64(-64%); SLAKE margin 2.01(-56%) < conf 2.06 < cum_logprob 2.19; VQA_RAD margin 3.79(-17%) ~ conf ~
cum_logprob. => MARGIN (top1-top2 first-token prob) is the best cascade gate signal for MCQ, marginally over conf/
cum_logprob (consistent with the original project's deployed margin gate). CoT think-tier test (32B MMMU, CoT prompt
patched into MedEvalKit is_reasoning) running to check if reasoning engages + helps.

## KEY FINDING — Lingshu has NO promptable think mode -> the Lingshu cascade is 2-TIER (2026-07-01)
Tested reasoning for Lingshu two ways on MedEvalKit MMMU-32B: (1) reasoning=True with the default \boxed prompt,
(2) an explicit "Reason step by step, then \boxed{answer}" prompt. BOTH: Lingshu-32B outputs just the letter
(gen_toks=2-3, e.g. "C"), acc 0.627-0.633 == no-think. Lingshu does NOT produce chain-of-thought on demand for MCQ
(it is RL-tuned to answer directly). => The salvaged eval_results_reason (MMMU 0.773) was a DIFFERENT model, not
Lingshu-32B. CONSEQUENCE: the "think tier" cannot be built for Lingshu by prompting; the Lingshu cascade is
INHERENTLY 2-TIER (7B-nt -> 32B-nt). This resolves "why 2 tiers?" for Lingshu: the model has no <think> mode.
(The 3-tier think tier applies to models that DO reason on demand -- MedVLThinker with its <think> prompt, already
analyzed: MMMU 3-tier matches 32B-think at ~78% FLOPs. So 3-tier is a MedVLThinker story, 2-tier is the Lingshu story.)
NOTE: the overnight *_think tier runs (reasoning=True, old prompt) therefore == no-think duplicates (gen_toks~3);
they are not a valid think tier and are excluded. question_formats.py reverted to original.

## OVERNIGHT SUMMARY (Lingshu Medical VQA, faithful MedEvalKit)
1. Baseline reproduced (SLAKE/PMC/MedXpert both sizes; 32B-MMMU exact). 2. 2-tier cascade is the deployable method:
matches Lingshu-32B accuracy at PMC-VQA -69% FLOPs/-33% latency (@9% esc, 33k), SLAKE -56%/-22% (@22%); mixed VQA-RAD;
no win MedXpert(floor)/MMMU(7B-inflated). 3. margin = best gate. 4. Lingshu has no think mode -> 2-tier only.

## CROSS-FAMILY faithful 2-tier cascade (MedVLThinker on MedEvalKit) + MMMU-anomaly resolution (2026-07-01)
MedVLThinker 2-tier (7B->32B, margin gate) on faithful MedEvalKit, match-32B min-FLOPs, closed subsets:
| bench | 7B | 32B | 2-tier | esc% | FLOPs vs 32B |
| PMC_VQA(33k) | 0.521 | 0.537 | 0.537 | 29% | -49% |
| VQA_RAD-cl(251) | 0.765 | 0.865 | 0.865 | 37% | -41% |
| MMMU(150) | 0.533 | 0.613 | 0.613 | 64% | -14% |
| SLAKE-cl(836) | 0.498 | 0.620 | - | 96% | +18% (no win, 7B weak) |
| MedXpert(2000) | 0.239 | 0.299 | - | 100% | +22% (no win, floor) |
CONCLUSIONS:
(1) The 2-tier cascade efficiency GENERALIZES across families where the cheap 7B is competitive: PMC-VQA (-49% MVT /
    -69% Lingshu FLOPs), VQA-RAD (-41% / -17%). Win magnitude ~ (32B-7B gap): small gap -> big win; large gap
    (SLAKE-MVT, MedXpert both) -> no win. So the method's efficiency is real+cross-family but benchmark/model-gap-dependent.
(2) MMMU-7B ANOMALY RESOLVED: MedVLThinker-7B MMMU=0.533 (NORMAL, < 32B 0.613) on the SAME faithful eval where
    Lingshu-7B=0.80. => Lingshu-7B's inflation is LINGSHU-7B-SPECIFIC (likely MMMU-medical-like train contamination),
    NOT a harness/subset/parser artifact. Definitively closes the anomaly.

## CONSOLIDATED faithful-eval cascade master (both families, MedEvalKit, 2-tier margin gate)
Where 7B competitive (PMC, VQA-RAD): cascade matches 32B at big FLOPs savings BOTH families. Where 7B weak vs 32B
(SLAKE, MedXpert) or 7B inflated (Lingshu-MMMU): no clean win. margin=best gate. Lingshu=2-tier only (no think mode);
MedVLThinker 3-tier(think) on NGC harness = MMMU ~78% FLOPs (separate, native <think>). Judge validated (kappa 0.85-0.96).
