# Hostile-reviewer audit: statistical validity + missing baselines
Agent `stats-baselines` · 2026-09-20 · READ-ONLY on MAIN

## Sources reused (not recomputed)
- `../em-rescore/em_rescore_pooled_probe_2026-09-18.json` + `em_rescore_stdout.log` (EM re-score, identical picks)
- `../em-rescore/em_rescore_per_question_cache.json` (per-question cache)
- `../me/judge_length_bias.json`
- `../replication/s02_rowcounts.out`, `../code-audit/out/s02.txt`

> IN PROGRESS — appended incrementally.

---
# PART A — statistical validity

## A0. The single biggest structural defect: the headline artifacts carry NO intervals
Scan script `s01_artifact_scan.py`, output `s01.out`. Top-level keys of every headline artifact:

| artifact | has CI/bootstrap key? | seeds | resampling unit |
|---|---|---:|---|
| `head_final_stack_PVFIXED_2026-09-13.json` (THE +0.0736) | **NO** | 5 | none |
| `head_final_stack_qwen_2026-09-13.json` (+0.0820 8/8) | **NO** | 5 | none |
| `head_final_stack_medgemma_ALL8_2026-09-16.json` (+0.0481 8/8) | **NO** | 5 | none |
| `head_lobo_pooled_2026-08-25.json` (breadth/own-half) | **NO** | 5 | none |
| `head_price_from_lobo_2026-08-30.json` (+0.0196 at k=0) | **NO** | **3** | none |
| `head_temp_ensemble_2026-08-30.json` (T=0.7 best) | **NO** | n/a | none |
| `head_pooled_alldomains_2026-08-24.json` (+0.0565 lever) | **NO** | 5 | none |
| `head_arch_transfer_2026-08-19.json` (arch ranking) | **NO** | n/a | none |
| `decomposition_2026-08-24.json` (skill 7/8) | **NO** | n/a | none |
| `coverage_sc16_ci_ALL_2026-09-16.json` (N 8->16) | **YES** (10,000, img-clustered) | n/a | image-within-benchmark |
| `tiebreak_2026-09-13.json` | YES | n/a | image |

`AUDIT_2026-09-12.md` §6 concedes this ("Most `head_*` artifacts carry **no interval at all**. ...
should not be quoted as such") — yet `DOMAIN_GUIDE` §2.4.1, the rundown §9 and the decks all quote
these point estimates as results. **The only interval that exists for the +0.0736 headline is the one
computed on 2026-09-18 by this audit** (`../em-rescore/...json`: img-clustered [+0.0608,+0.0864];
benchmark-level n=8 [+0.0260,+0.1165]).

## A1. The headline claim table (computed: `s03_stats.py` -> `s03_stats.json`)

All arms below share IDENTICAL held-out questions, identical picks, image-clustered bootstrap
(10,000, clustered by `img_md5` within benchmark), n = 18,452 questions / 15,376 images / 8 benchmarks.

| arm minus greedy | judge macro | lenient EM | strict EM | token F1 |
|---|---:|---:|---:|---:|
| **probe (SHIPPED)** | **+0.0737** 6/8 | +0.0047 3/8 | +0.0103 4/8 | +0.0156 6/8 |
| majority vote (self-consistency@8) | **-0.0149** 3/8 | -0.0062 3/8 | -0.0061 3/8 | -0.0148 2/8 |
| random pick from the 8 samples | **-0.0271** 0/8 | -0.0307 0/8 | -0.0233 0/8 | -0.0308 0/8 |
| answer-prior counter, LOO, **fitted on held-out labels (optimistic)** | **+0.0185** 5/8 | -0.0208 3/8 | -0.0228 1/8 | -0.0135 3/8 |
| oracle@8 | +0.1822 | +0.1301 | — | — |

**Sign test on 6/8: two-sided p = 0.289. Not significant.** (EM 3/8, p = 1.000; strict EM 4/8,
p = 1.000; token-F1 6/8, p = 0.289.) Benchmark-level t-interval, df 7: judge [+0.0156, +0.1319];
EM [-0.0320, +0.0414]; strict EM [-0.0122, +0.0329]; F1 [-0.0180, +0.0493].

**Macro vs micro (question-weighted).** judge macro +0.0737 vs micro **+0.1109** (gap -0.0372, i.e.
**13x the thread-count floor and 50% of the effect**). EM macro +0.0047 vs micro +0.0057. The macro
convention is *conservative* here — but the two differ by half the headline, so the weighting must
be named every time, and "8 benchmarks, equal weight" is a choice that is never defended in any doc.

**Leave-one-benchmark-out of the judge macro: [+0.0616 (drop OmniMedVQA), +0.0901 (drop VQA-RAD)]** —
stays positive on all 8 drops. In EM: **[-0.0040 (drop RadImageNet), +0.0127 (drop Kvasir-x1)]** —
**crosses zero**, so the EM result is not even sign-stable to dropping one benchmark.

## A2. The claim-by-claim verdict

| # | claim | n / unit / CI? / seeds | tuned on the same held-out data? | clears the noise floors? | what is overstated | exact safer rewording |
|---|---|---|---|---|---|---|
| 1 | **+0.0736 macro, 6/8** (`head_final_stack_PVFIXED_2026-09-13.json`) | 18,452 q, 15,376 imgs, **no CI in the artifact**, 5 seeds; the only CI is this audit's [+0.0608,+0.0864] img-clustered / [+0.0260,+0.1165] benchmark-level | **Yes, jointly**: pooling, layer set {18,20,22}, rank_avg, T=0.7, BCE, h256 were each read off these same held-out halves | 22x thread floor (0.0033), 12x tie-break band (0.0061) — yes, in judge currency | (a) no interval in the artifact; (b) **judge currency only** — identical picks give lenient EM **+0.0047 [-0.0077,+0.0171] TIE**; (c) 6/8 has sign-test p=0.289; (d) the number itself is from an invocation whose thread count was never recorded — current code at 4 threads gives **+0.0707** (`repro_threading_2026-09-13.json`) | "Under a Lingshu-32B LLM judge the shipped probe adds **+0.0737 macro [+0.0608, +0.0864]** (image-clustered) over greedy on 18,452 held-out questions, positive on 6 of 8 benchmarks (sign-test p = 0.29). **On the identical picks the project's normalised exact match gives +0.0047 [-0.0077, +0.0171], a tie.** The pipeline was selected on these same held-out halves." |
| 2 | **pooled-training lever +0.0547 / +0.0554** | same 18,452; no CI; 5 seeds; within one run so thread count is common | the arm ladder is fair *within* a run | ~17x the floor — yes | only that it has no interval, and that it is **not** a "method" effect: it is the value of a benchmark's own labelled training half (identical content to claim 3) | "Retraining the probe on the training half of all eight benchmarks instead of four is worth **+0.0554 macro** over the four-domain probe (point estimate, no interval computed; both arms fitted in the same run at the same thread count)." |
| 3 | **LOBO breadth -0.0056 / own-half +0.0485** | `head_lobo_pooled_2026-08-25.json`: macro four_domain 0.027858, lobo 0.022264, pooled 0.070747; **no CI**; 5 seeds | LOBO is honest by construction | breadth -0.0056 is **1.7x the thread floor and BELOW the tie-break band (0.0061)** — it is indistinguishable from zero *and* from noise | (a) no interval on a number used to conclude "breadth buys nothing"; (b) **Kvasir-x1's LOBO fold is leaky**: it drops 23,221 rows (112,770 -> 89,549 = exactly Kvasir-x1's own half) and **leaves `kvasir_open` — the same GI-endoscopy source — in the training pool** (the rundown's own correction 5 says the source is counted twice). Kvasir-x1 is the benchmark with the largest k=0 LOBO gain (+0.0922); (c) **`OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §5.2/§9 still print "+0.0312 -> +0.0304 -> +0.0803, breadth -0.0008, own data +0.0500", which matches nothing on disk** | "Leave-one-benchmark-out: four-domain +0.0279, LOBO +0.0223, pooled +0.0707 (point estimates, no intervals). Breadth alone is **-0.0056, which is inside this project's own +-0.0061 tie-break ambiguity band and cannot be distinguished from zero.** The Kvasir-x1 fold is not a clean hold-out: `kvasir_open` shares its source and stays in training." |
| 4 | **onboarding "~100 labelled questions"; +0.0196 at k=0** | `head_price_from_lobo_2026-08-30.json`: ks = [0,50,100,250,500,1000,2000], **seeds = 3**, no CI, `zero_shot_macro` = 0.019639503 (verified) | k is not tuned, but the base is the LOBO probe whose Kvasir fold leaks (claim 3) | +0.0196 is 6x the thread floor; the per-benchmark k=100 numbers are not intervalled at all | **"the first ~100 questions carry most of the gain" is false on the two benchmarks that carry the macro.** Fraction of the max-k gain reached at k=100 (computed from the artifact's own curves): PathVQA 98.8%, SLAKE 85.7%, Kvasir-x1 81.0%, GEMeX 72.0%, **OmniMedVQA 65.9%, RadImageNet 46.4%**; VQA-RAD and VQA-Med never cross. **3 seeds violates this project's own "≥10 training seeds" standing caveat** (CLAUDE.md §0, `coadapt_verifier_T04_2026-08-14.json`, where the gain went +0.00384 at 1 seed to -0.00173 at 10) | "From a probe trained on the other seven benchmarks, the zero-shot macro is **+0.0196** and 6 of 8 already beat greedy at k = 0 (3 seeds, no intervals). **100 labelled questions buy 72-99% of the attainable gain on four benchmarks but only 46% on RadImageNet and 66% on OmniMedVQA — the two largest contributors to the headline — so quote a range, 100-500, not '~100'.** Two benchmarks never cross at any k." |
| 5 | **N 8 -> 16, +0.0167** | `coverage_sc16_ci_ALL_2026-09-16.json`: 8 benchmarks, **10,000 bootstrap, image-clustered within benchmark — the only headline-grade artifact with a proper interval**, 4 WIN 0 LOSS 4 TIE | no | yes, 5x the floor; and it is a paired same-question comparison | almost nothing. The one caution: it is judge currency only and has never been re-scored in EM; and it costs a real doubling of generation, so it is not free | "Doubling the sampling budget 8 -> 16 is worth **+0.0167 macro (4 WIN / 4 TIE / 0 LOSS, image-clustered 10,000-resample intervals)** — the best-evidenced positive result in the open-text arm. Judge currency; not yet re-scored in exact match." |
| 6 | **T = 0.7 is best** | `head_temp_ensemble_2026-08-30.json`: macro by T = 0.2 +0.0393, 0.4 +0.0539, **0.7 +0.0773**, 1.0 +0.0710; **no CI anywhere in the artifact** | **Yes — T was chosen by reading these same held-out halves**, and 4 temperatures x 8 benchmarks = 32 comparisons | T=0.7 minus T=1.0 = **+0.0063**, which is *just* above the tie-break band (0.0061) and 1.9x the thread floor. The per-benchmark "best T" column is a 4-way argmax on n as small as 97 | the artifact itself concedes per-benchmark tuning is in-sample (+0.0103); the *global* T = 0.7 choice is equally in-sample and is never labelled as such. And the headline's candidate pools ARE T = 0.7, so the headline inherits this selection | "Across a fixed grid of four temperatures **evaluated on the same held-out halves used for the headline**, T = 0.7 gives the largest macro (+0.0773 vs +0.0710 at T = 1.0). **The 0.0063 margin is at the edge of this project's own +-0.0061 tie-break ambiguity band; treat T in {0.7, 1.0} as tied.** Three of the four benchmarks with the largest gains prefer T = 1.0." |
| 7 | **layer ensemble +0.0007** | difference of two arms inside `head_final_stack_PVFIXED`; no CI | yes, chosen on the eval half | **NO. +0.0007 is 4.7x BELOW the thread-count floor (0.0033) and 8.7x below the tie-break band (0.0061).** `repro_threading_2026-09-13.json` shows the same arm moving 0.0029 between two invocations of identical code, and on **5 of 6 runs the un-shipped `pooled_ens_sc` arm beats the shipped `pooled_ens`** | it should not be reported as a positive number at all. TRANSFER_WALL §13 already says "that choice is not robust and must not be reported as a win" — and DOMAIN_GUIDE §2.4.1 reports it as "+0.0007 (a hedge)" anyway | "The 3-layer ensemble is **not measurably better than the best single layer (+0.0007, an order of magnitude below the +-0.0033 reproducibility floor).** It is shipped as a *variance hedge* — it collapses the single-layer spread from 0.013 to 0.003 — not as an accuracy gain. On MedGemma the single-layer arm is +0.0034 better, which is the same null read from the other side." |
| 8 | **Qwen +0.0820, 8/8** | `head_final_stack_qwen_2026-09-13.json`, 5 seeds, **no CI**, same 18,452 questions | same pipeline, no re-tuning — genuinely a replication | 8/8 sign test p = 0.0078 — the only benchmark-count claim in the project that *is* significant | (a) **Lingshu-7B is a Qwen2.5-VL-7B finetune**, so this is one language model, two training recipes — TRANSFER_WALL §14 says so, the DOMAIN_GUIDE §2.4.4 bullet does not; (b) no CI; (c) Qwen's greedy is much weaker (PathVQA 0.0715 vs Lingshu 0.3050) so part of 8/8 is headroom; (d) **no EM re-score has been run for Qwen or MedGemma at all** — given Lingshu's judge-to-EM collapse (+0.0737 -> +0.0047), these two replications replicate an effect that may be a judge effect | "On Qwen2.5-VL-7B-Instruct the identical pipeline gives **+0.0820 macro, positive on 8 of 8** (point estimate, no interval; judge currency, never re-scored in exact match). Qwen is the base model Lingshu was finetuned from, so this is a second training recipe over the same language model, not an independent family." |
| 9 | **MedGemma +0.0481, 8/8** | `head_final_stack_medgemma_ALL8_2026-09-16.json`, 5 seeds, **no CI**, full provenance block (argv/threads=4/git sha — the only artifact of the three that has one) | no | 8/8 p = 0.0078 | (a) no CI, no EM; (b) the **shipped** 3-layer arm gives +0.0447 (7/8) and the un-shipped single-layer arm gives +0.0481 (8/8) — the headline quotes the arm that is *not* the shipped recipe; (c) MedGemma's absolute greedy is 0.0653 on PathVQA and 0.0094 on VQA-Med, so several of the eight are headroom, not skill | "On MedGemma-4b-it (Gemma 3 + SigLIP) the **single-layer** probe gives +0.0481 (8/8) and the **shipped 3-layer** recipe gives +0.0447 (7/8); the difference is inside the reproducibility floor. Judge currency, no intervals, and MedGemma's absolute accuracy is far below Lingshu's on three of the eight." |
| 10 | **"beats the answer-prior baseline on all 8"** | `free_signal_bakeoff_2026-08-21.json` / rundown §1: **a DIFFERENT probe (frozen four-domain), on the FULL benchmarks (n = 35,012 incl. training images), PathVQA at the truncated 1,500** | the prior is train-fitted; fair | — | **The sentence is quoted beside the +0.0736 table as though it were about the shipped probe. It is not: different probe, different n, different split (full benchmarks, not held-out halves).** The shipped probe has never been compared to an answer prior on the held-out halves. My own hostile version — an answer-string counter allowed to peek at held-out labels (leave-one-question-out, therefore an *upper bound* on a counter) — recovers **+0.0185 of the +0.0737 judge macro = 25.1%**, and per benchmark **RadImageNet 67.5%, OmniMedVQA 59.6%, GEMeX 45.0%** of the gain, and it **beats the probe outright on VQA-RAD** (0.5155 vs 0.4639). In EM currency the same counter is -0.0208, i.e. the counter's advantage is judge-specific too | "The **frozen four-domain** probe beats a train-fitted answer-frequency counter on all eight *full* benchmarks (`free_signal_bakeoff_2026-08-21.json`). **The shipped pooled probe has not been compared against an answer prior on the held-out halves.** An optimistic held-out-fitted counter recovers 25% of the shipped macro overall and 45-68% on the three benchmarks that carry it, so the memorisation control must be re-run against the shipped arm before the claim is repeated." |
| 11 | **GEMeX +0.0148 [+0.0057, +0.0238]** as "the cleanest evidence of real verification" | TRANSFER_WALL §10 / rundown §5.5: **frozen four-domain probe, full 8,000 questions.** The shipped probe on GEMeX's 3,978 held-out questions is **+0.1197** | — | yes | **Currency-fragile, and worse than any other benchmark.** On identical picks GEMeX is judge **+0.1197 [+0.1075,+0.1319]** but lenient EM **-0.0096 [-0.0201,+0.0010] TIE**, strict EM **-0.0020**, token-F1 **+0.0056** — a judge-minus-EM gap of **0.129**, the largest of the eight. Greedy itself reads 0.3997 under the judge and 0.1870 under EM. **The benchmark held up as proof the probe verifies rather than memorises is the one where the two currencies disagree most**, and 237 of 579 of its upward judge flips have ZERO token overlap with the gold (`../me/judge_length_bias.json`) | "On GEMeX the shipped probe is **+0.1197 [+0.1075, +0.1319] under the judge and -0.0096 [-0.0201, +0.0010] under exact match** — the largest currency divergence of the eight benchmarks. Until a non-Lingshu judge or human adjudication resolves it, GEMeX cannot be cited as evidence about the *mechanism*; it is evidence that the judge and the string metric disagree." |
| 12 | **decomposition: selection skill positive on 7/8** | `decomposition_2026-08-24.json`: mean skill 0.038713, mean penalty 0.025680, mean head-greedy 0.013032; **frozen four-domain probe, no CI, judge currency** | no | the per-benchmark skill values have no intervals | (a) it is about the **superseded** four-domain probe, not the shipped one, yet is quoted in DOMAIN_GUIDE §2.4.5 next to the shipped numbers; (b) the identity `verifier - greedy = skill - penalty` is an algebraic rearrangement, not a finding — the content is entirely in which term is larger, and that has no interval; (c) `AUDIT_2026-09-12.md` §2.1 records that the sibling implementation of this quantity carried a 12.4x currency bug, and §5b records that two artifacts still disagree on head_minus_greedy for PathVQA (0.048667 vs 0.047333) and SLAKE (0.041860 vs 0.038760) | "For the **frozen four-domain** probe, decomposing verifier - greedy into selection skill minus sampling penalty gives positive skill on 7 of 8 benchmarks (mean +0.0387 against mean penalty +0.0257; point estimates, no intervals). **This has not been recomputed for the shipped pooled probe.**" |
| 13 | **selection efficiency ~0.80 is a "field constant"** | CLAUDE.md §0 / retrospective; `OPENTEXT_CORRECTIONS_2026-08-19.md` §1 already **retracted** the companion claim that it is a domain-coverage artifact | — | — | **Not reproduced by the current data under either standard normalisation.** Computed on the shipped probe's own picks (`s03_stats.json`): with a random-pick floor, sel_eff = **0.513 macro, range 0.158 (VQA-RAD) to 0.879 (OmniMedVQA)**; with a greedy floor, **0.392 macro, range -0.333 to +0.870**. In EM currency: 0.211 and **0.012**. A quantity that ranges over 0.16-0.88 across eight benchmarks and moves by 0.30-0.38 when you change the accuracy metric is not a constant | "Selection efficiency on the shipped probe is **0.51 macro (random-pick floor) / 0.39 (greedy floor), ranging 0.16-0.88 across benchmarks and collapsing to 0.21 / 0.01 in exact-match currency.** The '0.78-0.81 field constant' is a June/July MCQ-era figure under a different normalisation and **must not be carried into the open-text arm**; quote a per-benchmark range instead." |

## A3. Cross-cutting statistical defects

1. **Multiple-comparison exposure is never accounted for anywhere.** On the *same* held-out halves the
   project has evaluated: 7 architectures, 5 layers x 3 ensemble sets, 4 temperatures, 20 pruning
   rules, 6 regime detectors, 2 objectives, 4 tie-break rules, 4 generators, 2 verifier bases,
   k in 7 onboarding sizes. That is well over 100 evaluations. **A significant win at 8 benchmarks
   needs a benchmark-level t-interval excluding zero; at alpha = 0.05 uncorrected the judge headline
   just clears it ([+0.0156, +0.1319]) and nothing else in the project does.**
2. **The reproducibility floors are larger than half the reported effects.** thread count 0.0033 ·
   tie-break band 0.0061 · serving config +-0.008 per benchmark · new-verifier judge drift
   +0.006-0.009. Claims 3 (breadth -0.0056), 7 (+0.0007), and the T=0.7-vs-T=1.0 margin (+0.0063)
   are all inside one or more of them. The **+-0.008 serving-config floor is never applied to any
   September number at all**, although every 32B / cross-generator comparison crosses serving configs.
3. **The judge-drift caveat is stated and then not applied.** CLAUDE.md §0 says a newly trained
   verifier gets a free +0.006-0.009 under the same-family judge. **Every probe in the September work
   is newly trained and every September number is judge-only.** The EM re-score is the test of this,
   and it is far worse than +0.009: the macro falls from +0.0737 to +0.0047.
4. **The arms do share questions and the pairing is verified clean** — `em_rescore_stdout.log`
   "pairing null tests": 0 held-out questions without a greedy judge label, 0 sc8 slots without a
   judged row, 0 questions with more than one `img_md5`, 0 null `img_md5`, on all 8 benchmarks.
   This is the one thing that is unambiguously right.
5. **Kvasir-x1 (n = 5,152, 27.9% of held-out questions) is the weakest cell in the whole table.**
   Judge +0.1240, lenient EM **-0.0425 [-0.0511, -0.0342] LOSS**, and **strict EM delta is exactly
   0.0000 with oracle@8 strict EM = 0.0004** — i.e. on Kvasir-x1 essentially no candidate ever
   exactly matches the gold, so the "gain" is entirely a judge-leniency judgement about long free
   text. It also shares a source with the `kvasir_open` training domain.

---
# PART B — the baselines a reviewer will demand

| # | baseline | status | evidence / cost |
|---|---|---|---|
| 1 | **LoRA/SFT of the GENERATOR on the same labelled half, greedy-decoded** ("why not just fine-tune?") | **EXISTS, RUN 2026-08-11, NEVER REPORTED IN ANY DOC** | see B1 below |
| 2 | **Text-based (July LoRA) verifier on the same candidates, all 8 benchmarks** | **MISSING** — 3 of 8 only | see B2 |
| 3 | **Logit-based selectors (mean token log-prob, length-normalised likelihood, P(True))** | **MISSING AND CURRENTLY UNCOMPUTABLE** | see B3 |
| 4 | **Published nearest neighbours re-implemented (ELHSR/SWIFT, LiLaVe, CASE)** | **MISSING** | see B4 |
| 5 | **Probe-score-weighted majority vote** | **MISSING** (and not computable from the cache — no probe scores in it) | see B5 |
| 6 | **Greedy-included candidate set** | **MISSING** (needs probe scores) | see B6 |
| 7 | **7B + probe vs Lingshu-32B greedy** | **DUMPS EXIST FOR 4 OF 8 — COMPUTED HERE FOR THE FIRST TIME** | see B7 |
| 8 | **Image ablation of the probe** | **MISSING — scripts exist, no artifact, no features** | see B8 |
| 9 | **Independent validation of the 32B judge on the five new benchmarks** | **MISSING — SLAKE/VQA-RAD/PathVQA only, all July** | see B9 |

## B1. Generator LoRA SFT — it exists, it works, and no document mentions it
`ckpts/train/lora_cheapleg_s0` (trained 2026-08-11), merged to `ckpts/train/merged_cheapleg_s0`,
generated by the SAME scripts in the SAME serving config as its control
(`results/cascade_methods/artifacts/train_cheap_leg_2026-08-11_preregistration.json`, which is a
proper pre-registration). Results, **judge currency** (`cheapleg_score_open.py:141` reads `judge_ok`):

| benchmark | n | frozen-7B greedy | **LoRA-SFT-7B greedy** | delta | frozen-7B + July LoRA verifier bo8 | LoRA-SFT-7B + verifier bo8 |
|---|---:|---:|---:|---:|---:|---:|
| SLAKE-open | 645 | 0.72558 | **0.75349** | **+0.0279** | 0.74729 | 0.75659 |
| VQA-RAD-open | 200 | 0.46500 | 0.46500 | 0.0000 | 0.48000 | 0.50500 |
| PathVQA-open | 1,500 | 0.32933 | **0.34400** | **+0.0147** | 0.36133 | 0.37067 |
| **macro (3)** | | **0.50664** | **0.52083** | **+0.0142** | 0.52954 (+0.0229) | **0.54409 (+0.0375)** |

Source: `ckpts/cheapleg/scores_base7b/score_summary.json` and `ckpts/cheapleg/scores_adapt7b_s0/score_summary.json`.

**A reviewer's reading:** fine-tuning the generator on the same labelled data and decoding greedily
buys **+0.0142 macro at 1x inference cost**, versus **+0.0229 for best-of-8 + a verifier at ~2.7x
latency / 3.6x energy**. That is 62% of the verification gain for none of the inference cost — and
the two **compose** to +0.0375. This is the single most damaging omission in the write-up: the
project ran the obvious competing baseline, it came out comparable, and the result lives only in an
untracked checkpoint directory. Caveats to state with it: 3 benchmarks, 1 seed, the July LoRA
verifier (not the shipped probe), one NaN training step, and a train/eval resolution mismatch — all
recorded in the pre-registration.
**Cost to finish properly:** the missing piece is an SFT arm on the 5 new benchmarks + scoring with
the *shipped* probe. Training is ~6 GPU-h/seed; generation on 8 benchmarks ~1 GPU-day. Non-trivial,
but the 3-benchmark result can and must be reported today at zero cost.

## B2. Text verifier on all 8
`free_head_2026-08-16.json` + DOMAIN_GUIDE §2.4.3: the July LoRA verifier comparison is on
**2,345 questions across 3 benchmarks** (SLAKE, VQA-RAD, PathVQA) = **12.7% of the 18,452**. The
claim "the probe verifier beats the July LoRA verifier by +0.0162 accuracy" therefore covers three
of eight benchmarks, none of them the five where the currency problem is worst.
**Cost:** the LoRA verifier needs one forward pass per candidate over ~147,000 held-out candidate
slots. At the measured ~82% LM-prefill workload this is ~1-2 GPU-days. **Not cheap — but it is the
only way to answer "is the frozen-state probe better than a trained text verifier, or just cheaper?"**

## B3. Logit-based confidence selectors — cannot be computed from anything on disk
Every open-text dump stores **one scalar** `seqlogprob` per *question* (plus `conf`, `margin`), not
one per candidate: `ckpt_*_sc8.jsonl` has `preds` = list of 8 and `seqlogprob` = a single float
(and `null` in the older `kvasir_open` build). Grep for `seqlogprobs`/`logprobs`/`cand_logprob`
across `ckpts/openvqa/` returns **nothing**.
**Consequence: mean token log-prob, length-normalised likelihood, perplexity and P(True) — the four
selectors every reviewer of a best-of-N verifier paper asks for, and the baselines ELHSR/SWIFT/HSRM
all report against — are not measurable from the saved data.** This is the most serious *missing*
baseline, because it is the cheapest possible competitor and the probe's whole claim is "a tiny head
beats what you already have".
**Cost:** near zero if done right. `extract_generator_hidden.py` already runs a teacher-forced
forward over each candidate's own tokens to produce `h_span`; the per-token logits are in that same
pass. Storing `sum logprob` and `n_tokens` per candidate costs **no extra GPU time** — only a
re-extraction (the extraction of all 8 benchmarks x 4 temperatures has been done before, so call it
~1 GPU-day). Alternatively vLLM `SamplingParams(logprobs=1)` at generation time, also free.

## B4. What `head_arch_transfer_2026-08-19.json` actually tested — and what it did not
`endpoint` = "leave-one-domain-out **sel_eff** on the held-out domain, MINUS the string-prior null".
Seven variants, all of them **feature transforms of the same mean-pooled MLP**:
`raw` 0.07338 · `domadv` 0.05578 · `poolnorm+raw` 0.05269 · `poolnorm` 0.04093 · `pca32` 0.03860 ·
`pca128` 0.03388 · `rankonly` 0.03015. Folds: `kvasir_open`, `pathvqa_open_train`,
`slake_open_train`, `vqa_rad_open_train` — **the four July domains only, not the eight benchmarks**,
on sel_eff not accuracy, five weeks before the pooled probe existed.

**None of the three published nearest neighbours is among them:**
- **ELHSR / SWIFT** — a *gated linear head over per-token hidden states* (token-level gating, then a
  weighted sum). Our `raw` is a mean-pool then MLP; the gate is exactly the difference. NOT TESTED.
- **LiLaVe** — **XGBoost on last-k token hidden states**. No gradient-boosted model appears anywhere
  in `src/training_methods/`. NOT TESTED.
- **CASE** — a **single-layer logistic** probe plus question-grouped ("decodability") evaluation.
  The logistic head is trivially a special case (h256 -> h0), and `head_sweep.py` does contain
  `A_objbce_h0` (0.66426) — but it was run on sel_eff in the four-domain era, never on the eight
  benchmarks, and CASE's *evaluation* protocol (within-question AUROC) is exactly the hole
  DOMAIN_GUIDE §2.5 item 9 admits is open.
**Cost: ~1-2 h CPU each** on the existing cached features. This is the cheapest gap in the list and
the one a reviewer will be least forgiving about, because all three are cited in the project's own
prior-art doc as the mechanism's ancestors.

## B5 / B6. Probe-score-weighted vote, and a greedy-included candidate set
Both **MISSING**, and both **uncomputable from the per-question cache** — it stores the probe's
`pick` index but not the probe's scores. Grep for `greedy_included`/`weighted_major`/`score_weighted`
across `src/` finds only unrelated files (`dawid_skene_aggregate.py`, a backlog entry).
**Cost: ~10 min CPU each** once the 24 frozen probes are reloaded and their scores dumped — exactly
what `em_rescore_pooled_probe.py` already does internally but does not save. Worth doing: (a)
score-weighted vote is the standard "verifier + self-consistency" hybrid (Li et al. weighted
majority) and beats plain best-of-N in most of the literature; (b) a greedy-included pool is the
guardrail that would fix VQA-RAD and VQA-Med, the two benchmarks the method loses, since the probe
currently cannot choose the greedy answer unless sampling happened to produce it.

## B7. Bigger-model reference: 7B + probe vs Lingshu-32B greedy — COMPUTED HERE
Judged Lingshu-32B **greedy** dumps exist on disk for **4 of the 8** benchmarks
(`ckpts/openvqa/strong_lingshu/ckpt_{slake,vqa_rad,pathvqa}_open_lingshu32b.jsonl` and
`ckpt_radimagenet_open_lingshu32b_t0.jsonl`, each with a `.judge.jsonl`). Joined by `idx` to the
**same held-out questions**, image-clustered bootstrap:

| benchmark | matched | 32B greedy (J) | 7B greedy (J) | **7B + probe bo8 (J)** | probe - 32B (J) | probe - 32B (lenient EM) |
|---|---:|---:|---:|---:|---|---|
| SLAKE | 330 / 330 | 0.7939 | 0.7121 | 0.7636 | **-0.0303 [-0.0716, +0.0119] TIE** | **-0.0788 [-0.1247, -0.0340] LOSS** |
| VQA-RAD | 97 / 97 | 0.5773 | 0.5052 | 0.4639 | **-0.1134 [-0.1868, -0.0495] LOSS** | **-0.1546 [-0.2386, -0.0769] LOSS** |
| PathVQA | **700 / 1,623** | 0.3729 | 0.3214 | 0.4086 | +0.0357 [-0.0061, +0.0779] TIE | +0.0129 [-0.0245, +0.0512] TIE |
| RadImageNet | 1,004 / 1,004 | 0.3078 | 0.3337 | 0.4562 | **+0.1484 [+0.1215, +0.1763] WIN** | **+0.0986 [+0.0767, +0.1215] WIN** |
| **4-benchmark macro** | | 0.5130 | 0.4681 | 0.5231 | **+0.0101** | **-0.0305** |

**Reviewer's reading: on the only four benchmarks where the comparison is measurable, a 7B with the
probe LOSES to a plain 32B greedy on two, ties on one, and wins on one — and the one win is on
RadImageNet, where the 32B (0.3078) is actually WORSE than the 7B (0.3337), so it is a win over a
degraded reference, not over a stronger model.** In exact-match currency the 4-benchmark macro is
**-0.0305**, i.e. the 32B is better. Caveats that must travel with this table: (i) PathVQA matches
only 700 of 1,623 — the 32B dump is on the truncated prefix, so that row is on the *easier* half
(`AUDIT_2026-09-12.md` §3); (ii) ~~the judge is Lingshu-32B scoring its own greedy output on the
32B rows — self-judging, which biases towards the 32B~~ *corrected 2026-09-28 (CHECK_2026-09-28.md C4c):*
the judge is **MedVLThinker-32B** (`src/labeling/run_judge.py:21`), not Lingshu-32B, so this is **not**
self-judging and no bias toward the 32B follows; the artifact's `note` field still carries the old label; (iii) the 32B dumps predate the current
serving config, so the +-0.008 caveat applies; (iv) **the comparison is not cost-matched** — 7B
best-of-8 is ~1.13x 7B FLOPs / 2.74x latency (`bestofn_vllm_2026-09-16.json`) against a 32B's ~4.6x
weights, so the honest framing is a Pareto one, not "we beat the big model".
**Cost to complete:** 32B greedy on the 4 missing benchmarks (Kvasir-x1, OmniMedVQA, VQA-Med, GEMeX)
= 15,398 held-out questions, ~1-1.5 GPU-days at tp=2, plus judging. **This should be the highest-
priority GPU job in the project** — it is the first question any reviewer asks and the answer on the
measurable half is currently unflattering.

## B8. Image ablation of the probe — MISSING
`src/training_methods/extract_generator_hidden_ablated.py` (writes `feats_hidden_noise`) and
`src/training_methods/langside_image_dependence_cv.py` (writes
`artifacts/_free_head_parts/langside_image_dependence.json`) both exist. **Neither output exists:**
`feats_hidden/` contains no `*noise*`/`*ablat*` directory, and `_free_head_parts/` holds only
`equivalence.json` and `selectors.json`. `open_ablations.json` is a pool-size (K) ablation, not an
image ablation; `head_visual_features_2026-09-01.json` *adds* image features (+0.0002) rather than
removing them.
**This is the control that answers "is the probe reading the image at all, or just the answer
string?"** — and my optimistic answer-prior result (A2 claim 10: a string counter recovers 25% of the
macro and 45-68% on the three benchmarks that carry it) makes it urgent rather than optional.
**Cost:** re-extract hidden states with the image replaced by noise/blank on the 8 eval halves; the
script exists, so this is ~4-8 GPU-h plus a CPU refit. **Pair it with the Hewitt-Liang control task
(label-randomised probe, pure CPU, ~1 h)** that DOMAIN_GUIDE §2.5 item 8 already admits is missing.

## B9. Independent validation of the Lingshu-32B judge — MISSING on 88.9% of the questions
Everything that exists is July 2026 and covers three benchmarks:
- `results/cascade_methods/claude_judge/verdicts_lingshu32b/` — **SLAKE and VQA_RAD only**, 20 files
  of up to 75 records each (`{"i": int, "correct": 0/1}`).
- `results/cascade_methods/claude_judge/pathvqa_granularity/` — the PathVQA hand-label batches.
- `artifacts/pathvqa_judge_audit.json` (2026-07-29): hand validation of **25 accepts + 25 rejects**
  on PathVQA — error rate 0.04 fair / 0.04 strict, **Wilson 95% [0.0071, 0.1954]**, i.e. the judge's
  error rate could be anywhere up to ~20%. Plus a 90-item stratified hand-label of one disagreement
  pool: **a 28.9% / b 36.7% / c 7.8% / d 26.7%**, "not genuinely wrong (b+c)" **44.4%
  [0.3462, 0.5473]**, clear judge error (c) **7.8% [0.0382, 0.1519]**.
- `choicewhy_judge_concordance.json` (2026-08-03) is MCQ, 600 items, agreement 0.9967-1.0 — it
  validates the judge on *multiple choice*, which is the easy case and not the currency at issue.

**Zero independent adjudication exists for RadImageNet, Kvasir-x1, OmniMedVQA, VQA-Med or GEMeX** —
**16,402 of the 18,452 held-out questions (88.9%)**, and exactly the five where judge-minus-EM is
largest (Kvasir-x1 +0.1665, GEMeX +0.1292, OmniMedVQA +0.0946 per `../em-rescore/...log`). The
existing PathVQA audit already showed the judge's own error rate is only bounded at [0.7%, 19.5%] on
n = 50 — which is wider than the entire headline.
**Cost: LOW and it is the highest-value cheap job in the project.** 200 stratified items per new
benchmark (1,000 total), adjudicated by a *different* model family (or by hand), is a few hours of
API/CPU. Given the +0.0737-judge / +0.0047-EM split, **nothing about the mechanism can be concluded
until this is done.**

---
# Findings table (id · severity · claim · evidence · correction)

| id | sev | finding | evidence | correction |
|---|---|---|---|---|
| S1 | **CRITICAL** | The headline +0.0736 and 9 of the 11 other headline artifacts contain **no confidence interval of any kind** | `s01.out`: top-level keys of `head_final_stack_PVFIXED_2026-09-13.json` = title/date/no_fabricated_numbers/seeds/pooled_rows/original_rows/cells/macro/beats_greedy/VERDICT. Same for qwen, medgemma_ALL8, lobo_pooled, price_from_lobo, temp_ensemble, pooled_alldomains, arch_transfer, decomposition | Add the intervals from `../em-rescore/em_rescore_pooled_probe_2026-09-18.json`; stop mixing intervalled and un-intervalled numbers in one table |
| S2 | **CRITICAL** | The headline is judge-only and **collapses to a tie in every string currency** | `em_rescore_stdout.log`: judge +0.0737 [+0.0608,+0.0864] vs lenient EM +0.0047 [-0.0077,+0.0171] TIE (3/8), strict EM +0.0103 TIE, F1 +0.0156. LOBO of the EM macro **crosses zero**: [-0.0040,+0.0127] | Report both currencies in every table; drop "beats greedy on 6/8" as a headline |
| S3 | **CRITICAL** | The "why not just fine-tune the generator?" baseline **was run on 2026-08-11 and appears in no document** | `ckpts/cheapleg/scores_{base7b,adapt7b_s0}/score_summary.json`: LoRA-SFT greedy +0.0142 macro on 3 benchmarks at 1x cost vs +0.0229 for bo8+verifier at 2.74x latency | Report B1's table with its caveats before a reviewer finds it |
| S4 | **CRITICAL** | **7B+probe loses to plain Lingshu-32B greedy on 2 of the 4 benchmarks where the comparison is measurable**, and the one win is over a 32B that is worse than the 7B | B7 table, computed into `s03_stats.json` from `ckpts/openvqa/strong_lingshu/*` | State it, frame as Pareto (cost), and run the 4 missing 32B greedy dumps |
| S5 | **HIGH** | **`OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §5.2 / §9 print LOBO numbers that match nothing on disk** — "+0.0312 -> +0.0304 -> +0.0803, breadth -0.0008, own data +0.0500" vs the artifact's 0.027858 / 0.022264 / 0.070747, breadth **-0.005594**, own **+0.048482** | `head_lobo_pooled_2026-08-25.json` keys `macro`, `macro_breadth_gain`, `macro_own_data_gain` | Replace both passages; the rundown's audit banner claims corrections were "applied in place" and these were not |
| S6 | **HIGH** | **Kvasir-x1's LOBO fold leaks**: it drops 23,221 rows (its own half) and leaves `kvasir_open`, the same GI-endoscopy source, in training | `head_lobo_pooled_2026-08-25.json` `cells.kvasir_x1_open.lobo_rows` = 89,549 vs `pooled_rows` = 112,770; rundown correction 5 states the source is counted twice | Re-run that fold dropping `kvasir_open` too, or exclude Kvasir-x1 from breadth/onboarding claims |
| S7 | **HIGH** | "**~100 labelled questions**" is false on the two benchmarks that carry the macro | computed from `head_price_from_lobo_2026-08-30.json` curves: share of max-k gain at k=100 — RadImageNet **46.4%**, OmniMedVQA **65.9%** (vs PathVQA 98.8%, SLAKE 85.7%, Kvasir 81.0%, GEMeX 72.0%) | "100-500 labelled questions, benchmark-dependent" |
| S8 | **HIGH** | The onboarding price curve is **3 seeds**, against the project's own standing "≥10 seeds" rule | `head_price_from_lobo_2026-08-30.json` `seeds` = 3; CLAUDE.md §0 seed-depth caveat | Re-fit at 10 seeds (CPU-only) or label it "3 seeds, provisional" |
| S9 | **HIGH** | "Layer ensemble +0.0007" is reported as a gain although it is **4.7x below the thread-count floor**, and the shipped arm loses to the un-shipped one on 5 of 6 runs | `repro_threading_2026-09-13.json` `thread_sweep_spread` = 0.003279, `best_arm.post_refactor` = "pooled_ens_sc (5 of 6 runs)" | Reframe as a variance hedge with no accuracy claim |
| S10 | **HIGH** | **GEMeX, cited as "the cleanest evidence of real verification", has the largest judge-vs-EM divergence of the eight** (judge +0.1197, lenient EM -0.0096, strict EM -0.0020) | `em_rescore_stdout.log`, gemex row | Retire the "cleanest evidence" framing until an independent judge exists |
| S11 | **HIGH** | **No logit-based selector baseline is computable** — the sc8 dumps store one `seqlogprob` per *question*, not per candidate | `ckpt_gemex_open_lingshu7b_sc8.jsonl` line 1: `preds` = 8 strings, `seqlogprob` = one float; grep for per-candidate logprob fields across `ckpts/openvqa/` returns nothing | Harvest `sum logprob` + `n_tokens` per candidate in the existing hidden-state extraction pass (zero extra GPU) |
| S12 | **HIGH** | **The 32B judge has never been independently validated on 88.9% of the held-out questions** | `claude_judge/` holds SLAKE + VQA_RAD verdicts only; `pathvqa_judge_audit.json` is n=50, Wilson [0.0071, 0.1954] | 200 stratified items x 5 new benchmarks, cross-family adjudication |
| S13 | **MED** | An answer-string counter fitted on held-out labels (an upper bound, but no image and no hidden state) recovers **25.1% of the judge macro**, and **67.5% / 59.6% / 45.0%** on RadImageNet / OmniMedVQA / GEMeX; it **beats the probe on VQA-RAD** | `s03_stats.json` `per_benchmark.*.judge.answer_prior_LOO_optimistic_minus_greedy` | Re-run the memorisation control against the **shipped** probe on the held-out halves; do not reuse the four-domain-probe sentence |
| S14 | **MED** | "6/8 benchmarks" has **sign-test p = 0.289**; it is used throughout as if it were evidence | `s03_stats.json` `summary_by_currency.judge.probe.sign_test_two_sided_p` | Quote the interval, not the count |
| S15 | **MED** | Macro and micro differ by **-0.0372** (judge), half the effect; no doc names the weighting | `s03_stats.json` `macro_minus_micro` | Name the convention in every table (micro is *more* favourable here: +0.1109) |
| S16 | **MED** | "Selection efficiency ~0.80, a field constant" is **not reproduced**: 0.513 (random floor) / 0.392 (greedy floor) macro, range 0.16-0.88; in EM, 0.211 / 0.012 | computed from `s03_stats.json` | Quote a per-benchmark range; do not carry the MCQ-era constant into the open-text arm |
| S17 | **MED** | "T = 0.7 is best" is a **4-way argmax read on the same held-out halves**, margin over T=1.0 = +0.0063, at the edge of the +-0.0061 tie-break band | `head_temp_ensemble_2026-08-30.json` `macro_by_temperature` | "T in {0.7, 1.0} are tied; three of the four largest-gain benchmarks prefer 1.0" |
| S18 | **MED** | Qwen (+0.0820) and MedGemma (+0.0481) replications are **judge-only, never EM re-scored**, so they may replicate a judge-specific effect | `head_final_stack_{qwen,medgemma_ALL8}*.json` have no EM arm | Run the EM re-score on both (CPU-only, ~5 min each with the existing script) |
| S19 | **MED** | MedGemma's headline quotes the **single-layer** arm (+0.0481, 8/8) while the shipped recipe is the 3-layer (+0.0447, 7/8) | `head_final_stack_medgemma_ALL8_2026-09-16.json` | Quote the shipped arm; note the single-layer is +0.0034 better and that this is inside the floor |
| S20 | **LOW** | `head_final_stack_PVFIXED_2026-09-13.json` and `..._qwen_2026-09-13.json` carry `"date": "2026-08-24"` — a stale date inside the artifact of record; only the MedGemma artifact has a `provenance` block (argv, threads=4, git sha) | `s01.out` | Stamp the real run date and add provenance to the other two |

## Verified clean (checked, found correct)
- **Pairing.** 0 held-out questions without a greedy judge label, 0 sc8 slots without a judged row,
  0 questions with more than one `img_md5`, 0 null `img_md5`, on all 8 benchmarks (n = 18,452;
  `em_rescore_stdout.log` "pairing null tests"). Every arm in this report is paired on identical questions.
- **Reproduction from the frozen checkpoint.** The 24 frozen probes on disk give +0.073720 against
  the recipe's +0.073607 (deviation +0.000113), with n and greedy exact-matching on all 8.
- **`coverage_sc16_ci_ALL_2026-09-16.json`** is methodologically correct: 10,000 resamples,
  image-clustered within benchmark, held-out only, correct verifier, a `supersedes` field naming what
  it replaces, and it keeps its own wrong run (`..._incumbent_...json`). This is the model the other
  artifacts should follow.
- **`tiebreak_2026-09-13.json`** and **`repro_threading_2026-09-13.json`** are honest, intervalled and
  self-retracting where required; TRANSFER_WALL §13's retraction banner is exemplary.
- **`train_cheap_leg_2026-08-11_preregistration.json`** is a genuine pre-registration written before
  any adapted-arm number existed, with null tests recorded.
- **Selection is within-question** (argmax inside one question), so the accuracy endpoints are not
  exposed to the pooled-AUROC leakage that DOMAIN_GUIDE §2.5 item 9 flags.
- `AUDIT_2026-09-12.md` §6's own admission that most `head_*` artifacts carry no interval is accurate
  — I confirmed it on 9 artifacts.

## Could not verify
- Whether T = 0.7 was *chosen* before or after the held-out halves were first read. T = 0.7 is the
  June-era default, so the sweep may be confirmation rather than selection, but no artifact records
  the order. **NOT VERIFIED.**
- Whether the Lingshu-32B greedy dumps in `ckpts/openvqa/strong_lingshu/` used the same image
  resolution / serving config as the 7B pools. The B7 table therefore carries the +-0.008 caveat.
  **NOT VERIFIED.**
- The exact definition of `genframe_data.sel_eff` behind the "0.78-0.81" figure. I computed two
  standard normalisations; a third would change the level but not the 0.16-0.88 spread.
- Whether any Qwen/MedGemma judge labels came from a judge in the same family as the generator,
  which would change the drift direction. **NOT VERIFIED.**

## Scripts and outputs (all under `/home/jamesyang/.claude/jobs/37d73e6f/tmp/stats-baselines/`)
`s01_artifact_scan.py` -> `s01.out` (artifact metadata scan) ·
`s02_partB.sh` (Part B disk search) ·
`s03_stats.py` -> `s03_stats.json` (all Part A computations plus the B7 32B comparison).
Nothing in MAIN was written, moved or modified.
