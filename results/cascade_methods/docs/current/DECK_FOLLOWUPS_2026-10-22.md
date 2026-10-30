# Deck follow-ups, 2026-10-21/22 — seeds, layers, label-free temperature, baselines, evaluation

> Scope: the experiments held over from the 2026-10-19 professor deck (`meetings/probe_verifier_report_2026-10-19.html`),
> plus two questions Leo added on 2026-10-21 ("do we actually need 8 seeds", "test deeper layers and different
> fusions") and deck options 2, 3, 4 (2026-10-22). Everything runs under the FINAL protocol of
> `FINAL_METHOD_2026-10-16.md` (cap320 pools, frozen probes `ckpts/train/final_probe_2026-10-16/fold{0,1}`,
> rank_vote, two-fold image cross-fitting, paired image-clustered bootstrap 10,000, macro over the 8 open benchmarks).
> Two policies are reported where it matters: **T07** (the shipped method and the deck's headline: every benchmark
> sampled at T=0.7) and **Tmap** (the frozen per-benchmark temperature map — FINAL_METHOD's *domain-calibrated
> variant, not shipped*, reversed 2026-10-18 because it needs the dataset label at test time).
> **Currencies:** judge = MedVLThinker-32B (training judge, judge of record) · MedGemma = MedGemma-27B-it
> (cross-family) · EM = lenient exact match. Every number below is copied from the named artifact in
> `results/cascade_methods/artifacts/`; the scripts that recompute picks first check that they reproduce the
shipped ones (each artifact's `checks` block).

**Status (2026-10-23 12:10 UTC): every section is final.** All 23 GPU jobs of `runners/pool_1022.json` exited 0. The
headline change is §4a: **a generator fine-tuned on the same labelled data beats the probe by ≈ 0.10 in every
currency**, mostly by learning each benchmark's answer vocabulary.

## 1. Do we need 8 seeds / 24 probes? — no; 12 probes (3 layers × 4 seeds) ties 24

`probe_count_sweep_2026-10-21.json` (Tmap) and `probe_count_sweep_T07_2026-10-22.json` (T07): all 1,020 seed
subsets of the frozen 24 probes (3 layers × K seeds, or one layer × K seeds), each scored on all 36,869 questions.
Checks: 0 / 36,869 pick mismatches against the shipped run; macro and CI reproduced exactly. Rows below are the
pre-specified subsets (seeds 0..K−1), as the difference from the shipped 24 probes:

| probes | Tmap judge | Tmap MedGemma | Tmap EM | T07 judge | T07 MedGemma | T07 EM |
|---|---|---|---|---|---|---|
| 3 layers × 1 seed (3) | −0.0041 LOSS | −0.0032 LOSS | +0.0004 TIE | −0.0062 LOSS | −0.0054 LOSS | −0.0017 TIE |
| 3 layers × 2 seeds (6) | −0.0025 LOSS | −0.0023 LOSS | −0.0009 TIE | −0.0038 LOSS | −0.0026 TIE | −0.0018 TIE |
| **3 layers × 4 seeds (12)** | **−0.0009 [−0.0022, +0.0004] TIE** | −0.0004 TIE | −0.0006 TIE | **−0.0016 [−0.0041, +0.0008] TIE** | −0.0010 TIE | −0.0002 TIE |
| layer 18 × 8 seeds (8) | −0.0033 LOSS | −0.0029 LOSS | +0.0005 TIE | −0.0025 TIE | −0.0039 TIE | +0.0015 TIE |
| layer 20 × 8 seeds (8) | −0.0042 LOSS | −0.0046 LOSS | −0.0043 LOSS | −0.0027 TIE | −0.0020 TIE | −0.0041 LOSS |
| layer 22 × 8 seeds (8) | −0.0024 LOSS | −0.0030 LOSS | −0.0021 TIE | −0.0054 LOSS | −0.0067 LOSS | −0.0033 TIE |

**Reading.** Seeds help, but they saturate by 4. Eight seeds on a single layer lose to the shipped ensemble on
Tmap in both judges, so **layer diversity is worth more than seed count**. 24 is not a special number; 12 probes is
the cheapest tie in every currency, for both policies. This is a tie, not a proof that 12 equals 24.

## 2. Evaluation owed (deck option 4)

### 2a. Template-clustered CIs — the judge headlines survive; T07's exact-match gain does not

`template_clustered_ci_2026-10-22.json`. Template = the normalised question string. A connected component =
images and templates linked by co-occurrence. SLAKE and RadImageNet are each a single component. Gain over greedy:

| | judge, image CI | judge, image × template | MedGemma, image × template | EM, image CI | EM, image × template |
|---|---|---|---|---|---|
| T07 | +0.0798 [+0.0710, +0.0883] | [+0.0632, +0.0959] WIN | [+0.0544, +0.0871] WIN | +0.0121 [+0.0030, +0.0209] | **[−0.0048, +0.0281] TIE** |
| Tmap | +0.0897 [+0.0829, +0.0969] | [+0.0770, +0.1028] WIN | [+0.0655, +0.0917] WIN | +0.0165 [+0.0105, +0.0225] | [+0.0034, +0.0295] WIN |

Two-way clustering roughly doubles the half-width (judge T07: 0.0086 → 0.0163). **The deck's "EM +0.0121 WIN" for
T07 should be quoted as a tie under template clustering.** Tmap's EM gain survives every clustering tried.

### 2b. Refit variance — the shipped probes sit inside a narrow spread

`refit_variance_2026-10-22.json`: the whole FINAL fit was repeated with probe seeds offset by 100, 200, 300 and 400,
data and recipe unchanged (`refit_variance.py`), giving 5 replicates including the shipped one.

| | judge mean (sd) [range] | MedGemma mean (sd) | EM mean (sd) [range] | picks changed vs shipped |
|---|---|---|---|---|
| T07 | 0.0791 (0.0010) [0.0776, 0.0802] | 0.0700 (0.0009) | 0.0119 (0.0014) [0.0104, 0.0141] | 7.8–8.0 % |
| Tmap | 0.0892 (0.0008) [0.0881, 0.0901] | 0.0774 (0.0010) | 0.0158 (0.0009) [0.0150, 0.0169] | 7.9–8.1 % |

The shipped run (replicate 0, fitted at 4 threads) is 0.0798 / 0.0897 in judge: inside the range, near its top for
T07 (2nd of 5). **A sixth refit came out of the layer study** (§5: same seeds 0–7, bit-identical features, but
2 threads): **0.0823 (T07) / 0.0902 (Tmap) judge, EM 0.0154 / 0.0174** (`layer_fusion_2026-10-22.json`, control
row). That sits above the other five. Thread count alone therefore moves T07 judge by +0.0025 (CLAUDE.md's
numerics landmines already list CPU thread count, at +0.0048). **Across all six refits T07 judge spans [0.0776, 0.0823]** — about a quarter of the
bootstrap CI width. That is still not what limits any claim. But a single refit's macro carries roughly ±0.002–0.003,
so deltas smaller than that between two separately fitted probe sets are not interpretable. Comparisons in this
document are always against a same-session control. About 8 % of picks change under a refit.

## 3. Temperature without the dataset label (deck option 2)

### 3a. Label-free temperature policies — none beats fixed T=0.7; the frozen map still needs the label

`temp_labelfree_2026-10-22.json`. 34,812 questions (all four pools present; Kvasir has 8,064 of 10,121). Checks:
fixed T07, Tmap and MIX8 reproduce their shipped picks (0 mismatches). Difference from fixed T=0.7:

| policy (uses the benchmark label?) | judge | MedGemma | EM |
|---|---|---|---|
| fixed T=0.2 / 0.4 / 1.0 (no) | −0.0408 LOSS / −0.0236 LOSS / −0.0029 TIE | −0.0365 LOSS / −0.0218 LOSS / −0.0059 TIE | +0.0011 TIE / +0.0032 TIE / −0.0131 LOSS |
| **Tmap (yes)** | **+0.0099 [+0.0040, +0.0162] WIN** | **+0.0073 [+0.0012, +0.0135] WIN** | +0.0044 TIE |
| mix8: 2 samples at each T (no) | −0.0066 [−0.0127, −0.0003] LOSS | −0.0045 TIE | −0.0023 TIE |
| explore-if-unsure, cross-fitted τ (no) | +0.0018 [−0.0021, +0.0056] TIE | +0.0027 [−0.0015, +0.0068] TIE | −0.0038 TIE |
| best pair of T by confidence, cross-fitted (no) | −0.0013 TIE | −0.0013 TIE | −0.0044 TIE |
| oracle T per question (ceiling, uses gold) | +0.0857 WIN | +0.0595 WIN | +0.0425 WIN |

Cross-fitting picked very different thresholds on the two folds (τ = 0.78 vs 0.08), so the confidence-based policy
is unstable. **Today the label-free options tie or lose to fixed T=0.7.** The map's +0.0099 needs to know the
benchmark, and the per-question oracle shows +0.0857 of headroom (judge; +0.0595 MedGemma) that no cheap signal
reaches. MedGemma labels for every pick of every policy were generated for this (24,038 new labels, 0 missing).
Per-modality temperature, where the model names its own modality, is §3c.

### 3b. Adaptive sample count — about 6 samples instead of 8 for free (label-free)

`adaptive_n_2026-10-22.json` (36,869 q, T07). Draw 4 samples, then draw samples 5–8 only for the least-confident
fraction (confidence = mean sigmoid over the 24 probes; the cut is a quantile, so no labels are used):

| re-sample | mean N | vs fixed N=8: judge / MedGemma / EM | vs fixed N at equal cost: judge / MedGemma / EM |
|---|---|---|---|
| 25 % | 5.0 | −0.0145 LOSS / −0.0131 LOSS / −0.0021 TIE | vs N=5: +0.0056 WIN / +0.0043 WIN / +0.0016 TIE |
| **50 %** | **6.0** | **−0.0016 TIE / −0.0021 TIE / +0.0023 TIE** | vs N=6: **+0.0087 WIN / +0.0058 WIN / +0.0037 WIN** |
| 75 % | 7.0 | −0.0001 TIE / −0.0003 LOSS / +0.0007 WIN | vs N=7: +0.0045 WIN / +0.0040 WIN / +0.0003 TIE |

**Re-sampling only the least-confident half ties N=8 in all three currencies at 6 samples on average**, and beats a
fixed N=6 in all three. The 50 % budget was pre-specified (budgets 5/6/7), with no tuning.

### 3c. Per-modality temperature from the model's own modality guess — ties T=0.7, loses to the map

Lingshu-7B answered "What is the imaging modality of this image?" once per unique image (cap320, greedy). The reply
was mapped to 12 classes (`predict_modality.py`, `modality_pred_2026-10-22.json`). It agrees with the datasets' own
modality labels on SLAKE 1.0, GEMeX 1.0, Kvasir 0.9996, RadImageNet 0.987 and OmniMed 0.932. PathVQA scores 0.728,
because 745 of its 3,357 images come back as "other". A temperature was then chosen per predicted modality on one
image fold and applied to the other (`temp_modality.py`, `temp_modality_2026-10-22.json`, 34,812 q, modalities
with ≥ 50 questions):

| | judge gain over greedy | vs fixed T=0.7 | vs Tmap | EM vs fixed T=0.7 |
|---|---|---|---|---|
| per-modality T, cross-fitted (no label) | +0.0809 [+0.0722, +0.0897] | +0.0011 [−0.0042, +0.0065] TIE | **−0.0089 [−0.0154, −0.0026] LOSS** | **−0.0086 [−0.0140, −0.0033] LOSS** |

Both folds choose T=1.0 for X-ray and CT. That matches GEMeX (map T=1.0, 8,000 X-ray questions), but it conflicts
with VQA-RAD (map 0.2) and SLAKE (map 0.7), which are also mostly X-ray and CT. Per-cell deltas were not computed.
**Reading: the temperature preference follows the benchmark (its answer style), not the imaging modality.** So "know the benchmark" cannot be swapped for "know the modality". (Caveat as in the artifact: the
per-modality choice is not re-made inside each bootstrap resample.)

## 4. Baselines owed (deck option 3)

`baselines_compare_2026-10-22.json` (T07 records; paired, image-clustered; every arm is scored on the same
questions as the probe).

| baseline | n | baseline − 7B greedy | probe − baseline | status |
|---|---|---|---|---|
| **Lingshu-32B greedy** (same prompt, cap320, 8 cells) | 36,869 | judge +0.0506 [+0.0397, +0.0615] · MedGemma +0.0438 · EM +0.0299 [+0.0184, +0.0415] | **judge +0.0292 [+0.0191, +0.0393] WIN · MedGemma +0.0272 [+0.0171, +0.0372] WIN · EM −0.0178 [−0.0284, −0.0073] LOSS** | final |
| WED, ported to our pools (Oct) | 4,202 (3 cells) | judge −0.0123 TIE · MedGemma −0.0147 TIE · EM −0.0153 TIE | judge +0.0489 [+0.0279, +0.0691] WIN · MedGemma +0.0460 WIN · EM +0.0189 [−0.0044, +0.0417] TIE (matches `wed_vs_probe_2026-10-16.json`) | final |
| **USC** (Chen et al. 2023 prompt + image, Lingshu-7B greedy) on the T07 pools | 36,869 | judge −0.0028 TIE · MedGemma +0.0013 TIE · EM −0.0183 [−0.0254, −0.0116] LOSS | **judge +0.0826 [+0.0751, +0.0901] · MedGemma +0.0697 · EM +0.0304, all WIN** | final |
| **USC** on the Tmap pools (vs the Tmap probe) | 36,869 | judge +0.0030 TIE · MedGemma +0.0031 TIE · EM −0.0156 LOSS | **judge +0.0868 [+0.0793, +0.0942] · MedGemma +0.0752 · EM +0.0321, all WIN** | final |
| **LoRA verifier retrained on the probe's pooled data** (July recipe, 30k rows/fold, cross-fitted, T=0.7 pool) | 36,869 | judge +0.0567 [+0.0474, +0.0658] · MedGemma +0.0568 · EM +0.0027 TIE | **judge +0.0232 [+0.0156, +0.0309] WIN · MedGemma +0.0142 [+0.0065, +0.0221] WIN · EM +0.0093 [+0.0025, +0.0163] WIN** | final |
| **generator LoRA-SFT on gold answers of the same data universe** (cross-fitted, merged, 1 greedy pass) | 36,869 | **judge +0.1817 [+0.1719, +0.1919] · MedGemma +0.1717 · EM +0.1186 [+0.1084, +0.1292]** | **judge −0.1019 [−0.1123, −0.0916] · MedGemma −0.1008 [−0.1115, −0.0903] · EM −0.1066 [−0.1172, −0.0964], all LOSS** | final — §4a |
| WED as published (authors' own sampling and candidate sets; PathVQA, SLAKE, VQA-RAD) | 4,202 (3 cells) | judge −0.0532 [−0.0722, −0.0343] · MedGemma −0.0514 · EM −0.0691, all LOSS | judge +0.0898 [+0.0659, +0.1130] · MedGemma +0.0827 · EM +0.0726, all WIN | final; 37 of 4,202 (0.88 %) had no valid WED candidate → greedy answer |

**USC buys nothing.** Asking the generator to pick the most consistent of its own 8 answers ties greedy in both
judges and loses to it in exact match. Every one of the 36,869 replies parsed, so no fallback was needed. **The probe beats a LoRA verifier trained on its own data, in every currency.** This removes the July confound
(that verifier was trained on four other domains). The LoRA verifier's batched scoring was checked against
one-at-a-time scoring and agreed on 19/20 picks per fold (max ΔP(Yes) 0.060 / 0.031), exactly at the threshold the
script asserts. **The 32B result splits by currency.** Under both judges (MedVLThinker-32B, and the cross-family MedGemma-27B at full coverage),
the 7B + probe beats Lingshu-32B greedy on all 8 benchmarks pooled. Under exact match, Lingshu-32B greedy wins.
Quote both, always. **WED as published is worse than greedy** in all three currencies on its three benchmarks. Its
own sampling (≈ 7.5 s per question) loses to the greedy answer, so the October port (WED's selector run over our
pools) remains the fairer comparison.

### 4a. The generator fine-tune beats the probe — mostly by learning each benchmark's answer vocabulary

This is the most consequential result in the document. Lingshu-7B was LoRA-fine-tuned (`train_cheap_leg.py`
`open_generic`, r 16, 1 epoch, answer tokens only) on the **gold answers** of exactly the questions the probe's
fitting rows come from (fold f trains on the other image half plus the four train domains;
`build_sft_manifest.py` asserts that no evaluation image is in the training set). It was merged and answered each
held-out question with one greedy pass. No answer reached the 64-token cap (mean 4.2–14.4 tokens per cell;
`stop_token_audit`).

Per benchmark (judge of record, accuracy; oracle@8 = a perfect selector over the 7B's 8 samples):

| | PathVQA | SLAKE | VQA-RAD | RadImageNet | Kvasir | OmniMed | VQA-Med | GEMeX | macro |
|---|---|---|---|---|---|---|---|---|---|
| 7B greedy | 0.3140 | 0.7302 | 0.4900 | 0.3210 | 0.2849 | 0.5164 | 0.0947 | 0.3974 | 0.3936 |
| probe (T07) | **0.3664** | **0.7876** | 0.4900 | 0.4545 | 0.3992 | 0.6768 | **0.0909** | 0.5218 | 0.4734 |
| SFT greedy | 0.3482 | 0.7535 | **0.5400** | **0.6255** | **0.7066** | **0.9358** | 0.0830 | **0.6096** | **0.5753** |
| oracle@8 | 0.4808 | 0.8791 | 0.6300 | 0.5120 | 0.4696 | 0.7007 | 0.2102 | 0.5864 | 0.5586 |

(Computed from `final_eval_FINAL_cap320_T07_records.json` and `xjudge_1022/new_all.judge.jsonl`. The macro is
the plain mean of the 8 cells, as in every artifact.) **On RadImageNet, Kvasir, OmniMed and GEMeX the fine-tune
beats even a perfect selector over the 8 samples.** No selector can do that, because the right answer is not in
the pool. On PathVQA, SLAKE and VQA-Med the probe beats the fine-tune.

**Stress test** (`sft_template_split_2026-10-23.json`, `sft_template_split.py`). The folds are split by image, so an
evaluation question can share its template, or its gold answer string, with a training question (macro over the
cells with ≥ 20 questions in the row — 6 for unseen templates, 7 for unseen answers, 8 for seen answers):

| held-out questions | n | SFT − greedy, judge / EM | probe − SFT, judge | probe − SFT, EM |
|---|---|---|---|---|
| template never seen in training | 13,615 | +0.1918 / +0.0954 | −0.1137 [−0.1359, −0.0902] LOSS | −0.1119 LOSS |
| gold answer string seen in training (any question) | 23,502 | +0.1946 / +0.1554 | −0.1083 [−0.1257, −0.0909] LOSS | −0.1301 LOSS |
| **gold answer string never seen in training** | 13,367 | +0.0779 / −0.0127 TIE | **−0.0391 [−0.0666, −0.0071] LOSS** | **−0.0174 [−0.0475, +0.0125] TIE** |

Unseen templates are not the right test, because Kvasir, OmniMed and GEMeX reuse a closed answer set across
templates. Unseen gold answers are the right test. 64 % of the held-out questions (OmniMed 100 %, RadImageNet
98.7 %, SLAKE 96.9 %) have a gold answer the fine-tune was trained to emit. On those it is far ahead. On the 36 %
whose answer it never saw, it still leads the probe by 0.039 under the judge, but **ties in exact match**.

**Reading.**
- **Given the same labelled data, fine-tuning the generator beats training a verifier over its samples.** It wins
  by about 0.10 in every currency, at one greedy pass instead of eight samples plus a feature pass.
- **The gain is mostly benchmark answer vocabulary.** That is legitimate in-distribution learning (the probe saw the
  same questions), but it also means the large cells reward memorising their label set.
- **The probe still wins where the generator is already strong** (PathVQA, SLAKE).
- **It is not yet known whether the two compose:** the probe has not been run over samples from the fine-tuned
  generator. That is the obvious next experiment.

## 5. Layer study: deeper layers and fusions — deeper is worse, and no fusion beats {18, 20, 22}

`layer_fusion_2026-10-22.json`. Twelve layers (16, 18–28; Lingshu-7B has 28) were extracted to
`/data/dan/feats_deep_2026-10-21`. Each layer was fitted with the shipped recipe, 8 seeds × both folds, at 2 threads
(`layer_fit.py`, `ckpts/train/layer_study_2026-10-22/`). Then 829 layer configurations (every subset of size 1–4,
contiguous windows, and all 12) were read out with rank_vote and scored on all 36,869 questions. Each one is compared
with a **same-session refit of {18, 20, 22}** (the control), never with the frozen probes. The deep caches are
bit-identical to the shipped caches (checked on RadImageNet, layers 18/20/22), so the only difference from the
shipped probes is the refit (see §2b).

Single layers vs the control (judge / MedGemma / EM), T07:

| layer | 16 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| judge gain | +0.0778 | +0.0788 | +0.0781 | +0.0778 | +0.0733 | +0.0756 | +0.0743 | +0.0736 | +0.0716 | +0.0727 | +0.0725 | +0.0698 |
| − control, judge | −0.0045 T | −0.0035 T | −0.0043 T | −0.0045 L | −0.0091 L | −0.0067 L | −0.0080 L | −0.0087 L | −0.0107 L | −0.0096 L | −0.0099 L | −0.0125 L |
| − control, MedGemma | −0.0068 L | −0.0047 L | −0.0060 L | −0.0042 L | −0.0087 L | −0.0068 L | −0.0100 L | −0.0104 L | −0.0148 L | −0.0135 L | −0.0137 L | −0.0159 L |
| − control, EM | −0.0026 T | +0.0017 T | −0.0013 T | −0.0048 L | −0.0085 L | −0.0052 L | −0.0057 L | −0.0064 L | −0.0092 L | −0.0099 L | −0.0090 L | −0.0097 L |

(T = tie, L = loss; CIs are in the artifact. Under Tmap every layer from 23 to 28 also loses in all three currencies.)

Fusions vs the control:

| configuration | T07 judge | T07 MedGemma | T07 EM | Tmap judge | Tmap MedGemma | Tmap EM |
|---|---|---|---|---|---|---|
| deep only: 22,24,26 / 24,26,28 / 23–28 | LOSS / LOSS / LOSS | LOSS ×3 | LOSS ×3 | TIE / LOSS / TIE | TIE / LOSS / LOSS | TIE / LOSS / LOSS |
| spread: 16,22,28 | −0.0031 TIE | −0.0038 TIE | −0.0033 TIE | −0.0001 TIE | −0.0009 TIE | −0.0007 TIE |
| all 12 layers | −0.0017 TIE | −0.0041 LOSS | −0.0027 TIE | +0.0007 TIE | −0.0005 TIE | −0.0002 TIE |
| **best of 829, chosen on the other fold** | **−0.0002 [−0.0050, +0.0044] TIE** | −0.0015 TIE | −0.0014 TIE | **−0.0013 [−0.0036, +0.0007] TIE** | −0.0014 TIE | −0.0003 TIE |

The two folds chose different winners (T07: 16,18,21,26 vs 18,19,21,28; Tmap: 18,22,24,28 vs 16,19,21,26). The
largest in-sample point gain over the control among the 829 is +0.0028 (T07) / +0.0016 (Tmap). Those are selected on
the evaluation data and are not a result.

**Reading.**
- **Deeper layers are worse:** the single-layer gain falls, with small bumps, from layer 18/19 down to layer 28.
- **Layers 21–28 are each beaten by {18, 20, 22}** in every currency on T07.
- **Fusing deep layers in does not help:** all 12 layers ties the control.
- **No layer combination beats {18, 20, 22} once the choice is cross-fitted**, under either policy.
- **The shipped mid-depth choice stands.** Layer 16 is close to the single layers 18–20 (T07 +0.0778 vs
  +0.0778–0.0788), so the useful information sits in the middle of the network, not at its end.

## 6. Incidents and corrections

**6a. `timeout … on G` exited 127 (2026-10-22).** `on` was a bash function, and `timeout` can only run programs.
Every GPU step written that way in `opt234_chain1.sh` / `opt234_chain2.sh` (modality, both USC runs, both MedGemma
batches, all of chain 2) exited 127 at once. The steps written as `timeout … flock …` (32B greedy, 32B judge) ran
fine. Fix: `runners/gpu_run.sh` (a real executable), and then `runners/pool_worker.py` (one worker per GPU over
`runners/pool_1022.json`; a job runs once its dependencies have *finished*, whether they produced output or
failed). Every chain-2 script was smoke-tested end to end first (`runners/smoke_1022.sh`).

**6b. An orphaned vLLM engine held both GPU locks.** Killing chain 2's judge left its `VLLM::EngineCore`
(TP0/TP1) alive with PPID 1. It held 73 GB on each card, and it held both `logs/.gpuG.lock` fds, which it had
inherited from `flock`. So the next jobs waited silently. `pool_worker.py` deliberately passes the lock fd on to
its jobs, so an orphan keeps the card marked busy rather than letting a new job share it. **Check `/proc/<pid>/fd`
for the lock files before you assume a card is free.**

**6c. GEMeX T=1.0 deep extraction: an empty answer, not fp16 overflow.** The extraction failed 4 times. My first
diagnosis, fp16 overflow, was wrong (`--f32_on_overflow` was added for it and is harmless). The real cause is that
one sampled answer for GEMeX q7502 is the empty string. Its answer span is empty, so the mean-pool is NaN at every
layer (12 × 3,584 = 43,008 NaNs). Fix: `--drop_nonfinite_rows` (`extract_generator_hidden.py`) zeroes such a row,
marks it `n_tok = −1` (the readers already skip those) and lists it in the meta.
**Latent bug in the Tmap-variant evaluation:** the legacy GEMeX T=1.0 cache kept that NaN row, and `rank_avg` sorts
NaN as best. So the Tmap run (`final_eval_FINAL_cap320_Tmap`) picks the empty answer for GEMeX q7502
(`final_eval_FINAL_cap320_Tmap_records.json`: `pick_ans ''`; T07 picks "Proper placement."). That question's
`oracle_judge` is 0, so the **judge-of-record macro is unchanged**. The MedGemma record for that pick is 1 (an
empty answer labelled correct, itself suspect): 1 of 8,000 GEMeX questions, ≤ 0.00002 macro. The shipped T07 run is not affected. Its eight caches (`generator_eval_finelayer*`, the rows it uses:
16,119 / 1,313 / 709 / 8,155 / 47,618 / 34,020 / 23,491 / 31,460) contain **0 non-finite rows on layers 18/20/22
and 0 empty answers**. That was checked 2026-10-23 by reading every cache. The Tmap numbers are left as they are. The NaN-safe ranking belongs in any future refit.

## 7. How it ran (2026-10-22 10:45 → 2026-10-23 12:04 UTC)

- GPU0: GEMeX T=1.0 deep re-extraction (`runners/deep_1021c_gpu0.json`, 3.6 h; 1 row dropped, q7502's empty
  answer). Then `runners/layer_study_chain.sh 209621`: the cache check, 12 layer fits on 4 CPU queues (64–74 min each),
  and `layer_fusion.py` (52 min).
- Both GPUs: `runners/pool_1022.json` via `runners/pool_worker.py`, 23 jobs, all rc 0:
  - modality 0.6 h; MedGemma on the temperature picks 2 × 0.15 h; LoRA verifier 2 × 11.9 h; SFT 4.7 / 5.1 h;
    USC 1.1 / 1.2 h; MedGemma on 32B greedy 0.2 / 0.1 h.
  - WED-published: 4 PathVQA shards × ~1.75 h, plus SLAKE and VQA-RAD.
  - The USC/LoRA-pick MedGemma list was empty, because every pick was already labelled.
  - Judge of record on 41,071 new answers 0.2 h; MedGemma on them 2 × 0.1 h.
  - State is in `runners/pool_1022.state/`, the log in `logs/pool_1022.log`.
- CPU: `runners/pool_1022_finish.sh` re-scored each result as its inputs finished. The final pass rebuilt
  `temp_labelfree`, `temp_modality` and `baselines_compare` on all 101,978 MedGemma labels.
- Script fixes made during the run:
  - The MedGemma label lookup now keeps the first LABELLED entry per (question, answer). A 7B pick identical to a 32B
    answer had been read as unlabelled.
  - `baselines_compare.py` survives an arm without judge labels, and compares Tmap-pool baselines with the Tmap probe.
  - `predict_modality.py` metadata agreement: a defaultdict `.get()` returned None.
