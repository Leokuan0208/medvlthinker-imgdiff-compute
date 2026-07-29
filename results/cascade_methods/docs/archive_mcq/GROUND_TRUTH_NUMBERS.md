# GROUND TRUTH NUMBERS (authoritative — from raw ckpts/train/*/*.json)
# Built 2026-06-27 by audit. Docs/paper MUST match this. frac = (trained-greedy)/(oracle-greedy).

## Free-text verifier — pooled-4 (HEADLINE), n=1064  [lora_verifier_pooled4/result.json]
greedy 0.4126 | sc 0.4107 | trained 0.5009 | oracle 0.5921 | gap-captured 49.2%
bootstrap gain +0.1156, 95% CI [+0.0921, +0.1391], n=1064 (vs first-sample K=1)
per-ds: pathvqa 0.352->0.441 (55.7%) | kvasir 0.282->0.405 (58.4%) | vqa_rad 0.519->0.611 (45.5%) | slake 0.738->0.762 (15.2%)
verifier discrimination AUROC=0.924 (n=8512 samples; mean score correct 0.749 vs incorrect 0.171)

## Free-text transfer (OOD)
radimagenet (pooled4 verifier, zero-shot): greedy 0.3285 -> trained 0.3535, oracle 0.512, n=2000 => 13.6%
kvasir (2-ds verifier): 0.2858 -> 0.3267, oracle 0.4908, n=1200 => 19.9%

## Scaling curves
K<=8 (n=1064): K1 0.385 / K2 0.425 / K4 0.476 / K8 0.501 ; oracle K8 0.592 ; random ~0.39
K<=16 (n=1621, DIFFERENT pool): K1 0.356 / K2 0.394 / K4 0.411 / K8 0.417 / K16 0.424 (diminishing)

## Box verifier — SLAKE (IoU>=0.3, thr 0.3)
seed0 [lora_box_verifier]:      greedy 0.197 | sc_medoid 0.164 | trained 0.255 | oracle 0.343 => 39.4%
seed1 [lora_box_verifier_slake_s1]: greedy 0.177 | trained 0.257 | oracle 0.329 => 52.7%
zeroshot [boxverify_slake_zeroshot]: 0.177 (BELOW greedy 0.199) => luck-floored

## Box verifier — MS-CXR (real PhysioNet benchmark, n=435)  *** CORRECTED ***
seed0 [lora_box_verifier_mscxr_full / _boot]: greedy 0.0414 | sc_medoid 0.0529 | trained 0.232 | oracle 0.285 => 78.3%
seed1 [lora_box_verifier_mscxr_s1]:           greedy 0.0414 | trained 0.230 | oracle 0.285 => 77.4%
zeroshot [boxverify_mscxr_zeroshot]:          trained 0.115 => 30.2% (above greedy, modest)
bootstrap (boot, seed0): gain +0.1908, 95% CI [+0.1517, +0.2322], n=435 (vs greedy)
NOTE: earlier-reported "0.248 / oracle 0.313" for seed1 is a SUPERSEDED pre-coord-fix artifact; NOT in any current file. Use 0.232/0.230.

## Trained-gate / distill (context, not verifier)
FLD (FastLeg-Distill): all5 orig=distilled 0.6377; all6 0.5056->0.5093 [fld/result.json]
lora_stability (visual-stability trained gate): auc_margin 0.681 / auc_casp 0.733 / auc_lora 0.723

## ACC (the structural cascade) — CANONICAL from master_data.csv (June-24 regen, post cost-fix)
MedVLThinker ALL-6 (parity acc 0.5723):
  always-32B-think:  acc 0.5723 | FLOPs 100% | latency 11.34s | energy 6318.8J
  Ours (ACC-v2 agreement): acc 0.5693 | esc0 71.7% | think 15.1% | FLOPs 52.0% | latency 2.27s | energy 1181.9J | guard 0.0
  => latency -80%, energy ~5.3x (-81%), FLOPs halved; acc delta -0.003 (parity)
  ACC-v1 (margin): acc 0.5687 | FLOPs 53.9% | 2.69s | 1416.6J
  CASP-Stability (trained): acc 0.5698 | FLOPs 49.0% | 1.77s | 899.2J | guard 0.05
MedVLThinker ALL-5 (parity acc 0.6463):
  always-32B-think: 0.6463 | 8.88s | 4915.9J | FLOPs 100%
  Ours (ACC-v2): acc 0.645 | esc0 35.1% | think 2.3% | FLOPs 24.9% | latency 0.44s | energy 172.8J | guard 0.05
  => latency -95%, energy ~28x; FLOPs to ~25%
Over-thinking premise (cap320 no-think vs fullres think, master_data.csv):
  SLAKE  big-nt 0.849 vs big-think 0.764 (+0.085) ; VQA-RAD big-nt 0.853 vs 0.776 (+0.077)
Deployed margin gate anchor (separate 2-tier system): tau=0.426, acc 0.5718, esc 63.3%, backbone 73.6%
