# REPLICATION AUDIT — "the probe verifier replicates on other generators"

Agent `replication`, 2026-09-20. Claim audited: **Qwen2.5-VL-7B-Instruct +0.0820 macro, beats its own
greedy on 8/8; MedGemma-4b-it +0.0481, 8/8.**
All numbers below are read from a named file or produced by a script in this directory. Every script
and every number is in `replication_currency_2026-09-20.json`.

## BOTTOM LINE

| claim | verdict | one-line reason |
|---|---|---|
| Qwen +0.0820 macro, 8/8, judge currency | **SOLID** | reproduced bitwise by an independent refit; pool is clean; survives all four currencies |
| "Qwen improvements across the board" as a research direction | **HOLDS WITH CAVEAT** | true in judge currency, but the effect is **3x smaller in exact match** (+0.0276) and only 3/8 cells are individually significant there |
| MedGemma +0.0481, 8/8 | **WRONG as a replication** | 100.0% of MedGemma's 8-sample candidates were truncated at the 64-token cap; greedy was not. The pool is a broken generation, not a MedGemma sample |
| "the probe verifier is not a property of Lingshu / the Qwen LM" (TRANSFER_WALL §14) | **OVERSTATED** | its only cross-family evidence is the defective MedGemma pool |

## FINDINGS

| id | sev | claim | evidence | correction |
|---|---|---|---|---|
| R1 | **CRITICAL** | **The MedGemma best-of-8 pool is a mis-generated arm.** `gen_tokens` is **exactly 64.00 mean, 100.0% at the cap, on all 8 benchmarks** (n=294,856 candidates) while MedGemma **greedy** stops naturally (mean 4.24-9.93, frac-at-max ~0). Lingshu and Qwen sc8 hit the cap in <=0.64% of samples. So MedGemma's sampled arm ran with EOS/`<end_of_turn>` not honoured and the greedy arm did. | `replication_currency_2026-09-20.json:generation_config_audit`; `s13_merge.py` output. Raw: `ckpts/openvqa/cheap_lingshu7b/ckpt_*_medgemma4b_sc8.jsonl` field `gen_tokens_all` | Do not quote +0.0481 or the §14 cross-family claim until MedGemma's sc8 is regenerated with correct stop handling. |
| R2 | **CRITICAL** | Consequence of R1: **58.5% of MedGemma candidates are degenerate text** (Gemma turn markers `\nmodel` / `\nimage` / `\nuser`, code fences, sentences repeated >=3x); **92.6% of questions have at least one**; **0.0% of greedy answers are degenerate**. Mean words: greedy 2.56, candidate pool 14.75, probe's pick **16.87** (6.6x greedy). Qwen: 0.02% degenerate, pick 3.13 words vs greedy 2.62. | `s12_degenerate.json`, `s11_length.json`. e.g. omnimed medgemma sc8: `\nmodel` in 20,008/71,064 candidates, code fence 3,295; greedy 0/8,883 contain any newline | — |
| R3 | **CRITICAL** | On the questions whose whole MedGemma pool is clean, the probe's judge macro falls **+0.0436 -> +0.0096** (6 cells with n>=40; 3 of 6 negative). Only 3-25% of questions per cell survive. | `s12_degenerate.py` output, `s12_degenerate.json:medgemma.clean_subset` | the clean subset is itself selected on pool cleanliness, so it is a diagnostic, not an unbiased estimate — but it shows the headline is not robust to R1 |
| R4 | **HIGH** | **MedGemma's gain is judge-only and reverses in every exact-match currency.** On identical picks (my 2-seed L24 refit): judge **+0.0436** [+0.0367,+0.0505] WIN · lenient EM **-0.0011** [-0.0070,+0.0048] **TIE** · strict EM **-0.0751** [-0.0872,-0.0635] **LOSS (0/8 positive, 6/8 significantly negative)** · token-F1 **-0.0311** LOSS. | `s07_currency.out`, `replication_currency_2026-09-20.json:medgemma.macro` | — |
| R5 | **HIGH** | **The "8/8" pair mixes arms.** Qwen's +0.0820/8-8 is `pooled_ens` (the shipped arm). MedGemma's +0.0481/8-8 is `pooled_singlelayer`. Under the **shipped** `pooled_ens` arm MedGemma is **+0.0447 at 7/8**, not 8/8 (it loses VQA-RAD, -0.0103). | `head_final_stack_medgemma_ALL8_2026-09-16.json`: `macro.pooled_ens=0.044717`, `beats_greedy.pooled_ens="7/8"`, `macro.pooled_singlelayer=0.048085`, `"8/8"` | Old: "MedGemma-4b-it +0.0481, 8/8". New: "MedGemma-4b-it +0.0447 (7/8) under the shipped ensemble arm; +0.0481 (8/8) under its best-of-four arm, selected on the evaluation half." |
| R6 | **HIGH** | `head_final_stack.py:337` builds VERDICT as `best = max(art["macro"], key=...)` — **every artifact's VERDICT quotes its own best arm, chosen on the eval half**, and it is a *different* arm per generator: medgemma->`pooled_ens_sc` (+0.0292), lingshu_matched->`pooled_ens_sc` (+0.0404), qwen_matched->`pooled_singlelayer` (+0.0224), qwen->`pooled_ens`, medgemma_ALL8->`pooled_singlelayer`. | `src/training_methods/head_final_stack.py:335-341`; `artifact_facts` in the JSON | VERDICT strings must never be quoted side by side. TRANSFER_WALL §14's matched table is **clean** — it uses `pooled_ens` for all three. |
| R7 | **MED** | **Neither headline artifact records its thread count.** `head_final_stack_PVFIXED_2026-09-13.json` and `head_final_stack_qwen_2026-09-13.json` have **no `provenance` block**; the four later artifacts do. §13 of TRANSFER_WALL establishes thread count moves this macro by 0.0033. So the two numbers the replication claim rests on are the two with no recorded invocation. | `s13_merge.py` / `artifact_facts.*.has_provenance` | State "+0.0736 / +0.0820 were produced before the provenance stamp; thread count NOT VERIFIED". |
| R8 | **MED** | **An answer-string prior with no image and no hidden state recovers ~30% of the judge gain.** P(correct given normalised answer) counted on the by-image train half, argmax on held-out (ties->more votes): Qwen **+0.0237** judge macro (6/8 positive) = 29.1% of +0.0814; MedGemma **+0.0131** = 30.1% of +0.0436. On OmniMedVQA/Qwen the prior alone is +0.1430 of the probe's +0.1894. | `s07_currency.py`, `replication_currency_2026-09-20.json:{qwen,medgemma}.macro.answerprior_minus_greedy_judge` | The probe **does** beat the prior (Qwen +0.0577 WIN, MedGemma +0.0305 WIN), but the prior is the control that belongs in any writeup and has never been run. |
| R9 | **LOW** | MedGemma headline reproducibility: my 2-seed / 2-thread refit of the **same arm** gives +0.0436 vs the artifact's 5-seed / 4-thread **+0.0481** — a -0.0045 gap, ~9% of the headline, consistent with the documented ±0.0033 thread floor plus seed count. The Qwen refit, at 5 seeds, is **bitwise identical** to the artifact on all 8 cells. | `s10_tables.out` §6 | — |
| R10 | **LOW** | My MedGemma refit crashed with SIGSEGV at 4 threads under load average ~21 and ran clean at 2 threads — the CLAUDE.md §0 CPU-head-fit landmine reproduces, and "4 threads runs clean" is **load-dependent**, not absolute. | `refit/medgemma_L24.log` (signal 11, `ucs_handle_error`), `s05_retry_medgemma.sh` | Amend the landmine: "<=4 threads AND check machine load". |

## 1. Per-benchmark table, held-out halves, JUDGE currency

probe = my local pooled single-layer refit (Qwen L20 5 seeds; MedGemma L24 2 seeds), which reproduces
head_final_stack.py's pooled_singlelayer arm exactly for Qwen. selEff = probe/oracle@8,
gapNrm = (probe-random)/(oracle-random).

```
gen        bench             n  greedy  random   major   probe oracle8  selEff  gapNrm
qwen       pathvqa        1623  0.0715  0.0607  0.0690  0.1017  0.1405   0.724   0.514
qwen       slake           330  0.5394  0.5258  0.5333  0.6182  0.6909   0.895   0.560
qwen       vqa_rad          97  0.3918  0.3544  0.3608  0.4433  0.5670   0.782   0.418
qwen       radimagenet    1004  0.2251  0.2156  0.2251  0.2859  0.3277   0.872   0.627
qwen       kvasir         5152  0.3445  0.3089  0.3205  0.4466  0.5462   0.818   0.580
qwen       omnimed        4461  0.4481  0.4379  0.4528  0.6375  0.6485   0.983   0.948
qwen       vqamed         1807  0.0133  0.0098  0.0089  0.0177  0.0382   0.464   0.280
qwen       gemex          3978  0.3937  0.3554  0.3658  0.5274  0.6280   0.840   0.631
qwen       MACRO         18452  0.3034  0.2835  0.2920  0.3848  0.4484   0.858
medgemma   pathvqa        1623  0.0653  0.0717  0.0690  0.0863  0.1590   0.543   0.167
medgemma   slake           330  0.6273  0.6242  0.6303  0.6394  0.6818   0.938   0.263
medgemma   vqa_rad          97  0.4948  0.4704  0.4639  0.4948  0.5361   0.923   0.373
medgemma   radimagenet    1004  0.2978  0.2986  0.2968  0.3556  0.4323   0.823   0.426
medgemma   kvasir         5152  0.1623  0.1586  0.1634  0.2139  0.2756   0.776   0.473
medgemma   omnimed        4461  0.5109  0.5433  0.5393  0.6351  0.6777   0.937   0.683
medgemma   vqamed         1807  0.0094  0.0103  0.0105  0.0122  0.0293   0.415   0.098
medgemma   gemex          3978  0.4253  0.4243  0.4238  0.5045  0.6013   0.839   0.453
medgemma   MACRO         18452  0.3241  0.3252  0.3246  0.3677  0.4241   0.867
lingshu-   MACRO         18452  0.3937       -  0.3780  0.4674  0.5566   0.840
```
lingshu- = the em-rescore agent's frozen 24-head ENSEMBLE (em_rescore_pooled_probe_2026-09-18.json);
its "major" column is the self-consistency pick, not a strict majority vote. Full per-cell Lingshu rows
are in s10_tables.out.

Two things the table says that the claim does not:
- MedGemma's random-pick (0.3252) is already ABOVE its greedy (0.3241). Its "gain over greedy" starts
  from a baseline that a coin flip over the pool already matches, because the pool is longer text that
  the judge likes (R1/R2). Qwen's random-pick is 0.0199 BELOW greedy and Lingshu's 0.0257 below, which
  is the normal shape.
- Majority vote LOSES for Qwen (-0.0114 macro, image-clustered LOSS, 1/8 positive) and ties for
  MedGemma (+0.0005). The probe's advantage over self-consistency is genuine on Qwen.

## 2. THE CURRENCY QUESTION - does Lingshu's collapse repeat?

probe - greedy, macro over 8, identical picks, 10,000-resample image-clustered bootstrap (and the n=8
benchmark-level bootstrap). sig+ / sig- = benchmarks whose own image-clustered CI excludes 0.

```
gen        currency        macro   image-clustered CI    verd   benchmark n=8 CI       verd  pos sig+ sig-
qwen       judge         +0.0814  [+0.0708,+0.0924]  WIN   [+0.0457,+0.1217] WIN    8/8   6    0
qwen       em            +0.0276  [+0.0168,+0.0384]  WIN   [+0.0100,+0.0472] WIN    7/8   3    0
qwen       strict_em     +0.0187  [+0.0076,+0.0298]  WIN   [+0.0067,+0.0318] WIN    6/8   4    0
qwen       token_f1      +0.0317  [+0.0217,+0.0425]  WIN   [+0.0184,+0.0450] WIN    8/8   6    0
medgemma   judge         +0.0436  [+0.0367,+0.0505]  WIN   [+0.0174,+0.0736] WIN    7/8   5    0
medgemma   em            -0.0011  [-0.0070,+0.0048]  TIE   [-0.0318,+0.0265] TIE    3/8   3    2
medgemma   strict_em     -0.0751  [-0.0872,-0.0635]  LOSS  [-0.1323,-0.0270] LOSS   0/8   0    6
medgemma   token_f1      -0.0311  [-0.0373,-0.0250]  LOSS  [-0.0560,-0.0093] LOSS   1/8   1    6
lingshu-   judge         +0.0737  [+0.0608,+0.0864]  WIN   [+0.0260,+0.1165] WIN    6/8   6    0
lingshu-   em            +0.0047  [-0.0077,+0.0171]  TIE   [-0.0232,+0.0329] TIE    3/8   3    2
lingshu-   strict_em     +0.0103  [-0.0014,+0.0212]  TIE   [-0.0055,+0.0290] TIE    4/8   2    1
lingshu-   token_f1      +0.0156  [+0.0044,+0.0265]  WIN   [-0.0106,+0.0422] TIE    6/8   4    1
```

ANSWER: the collapse does NOT repeat on Qwen; it repeats and WORSENS on MedGemma.

- Qwen is the strongest of the three generators and the only one whose gain survives in every currency.
  But the magnitude drops 3.0x (0.0814 -> 0.0276) and the "8/8" becomes 3/8 individually significant
  under EM (7/8 merely positive). Per cell (judge/EM): omnimed +0.1894/+0.0800, gemex +0.1337/+0.0055
  (TIE), kvasir +0.1021/+0.0103, pathvqa +0.0302/+0.0018 (TIE), vqamed +0.0044/-0.0044. The two biggest
  judge cells lose 88-96 percent of their gain under EM.
- MedGemma reverses: EM TIE, strict EM and token-F1 significant LOSSES on 6/8. Its probe picks answers
  6.6x longer than greedy (R2) - the lenient EM's contains-clause rescues some, strict EM cannot,
  token-F1 is diluted by the extra tokens.
- Length is not a sufficient explanation on its own: a pick-the-longest selector LOSES on all three
  generators (Qwen -0.0169, MedGemma -0.0063, Lingshu -0.0303 judge macro; s11_length.json). The
  MedGemma mechanism is pool contamination (R1/R2), not naive verbosity harvesting.

## 3. HEADROOM - is the macro carried by degenerate low-accuracy cells?

No, and the opposite of what was feared: the low-accuracy cells DRAG THE MACRO DOWN.

- Qwen: greedy < 0.10 on PathVQA (0.0715) and VQA-Med (0.0133). Those two contribute +0.00433 of the
  +0.0814 macro = 5.3 percent. Macro restricted to the 6 cells with greedy >= 0.20: +0.1027.
- MedGemma: < 0.10 on PathVQA (0.0653) and VQA-Med (0.0094), contributing +0.00296 = 6.8 percent of
  +0.0436. Macro on the 5 cells with greedy >= 0.20: +0.0547.
- Lingshu: only VQA-Med < 0.10; it contributes -0.0002. Macro on the 7 cells >= 0.20: +0.0845 (vs +0.0737).

The real headroom problem is different and worth stating: Qwen's absolute accuracy AFTER best-of-8 is
still below plain Lingshu greedy on 4 of 8 cells - PathVQA 0.1017 vs 0.3050, RadImageNet 0.2859 vs
0.3337, VQA-RAD 0.4433 vs 0.5052, VQA-Med 0.0177 vs 0.0913. "Improvements across the board" is a gain
over a much weaker generator, not a better system.

## 4. ANSWER-PRIOR CONTROL (never run before; judge currency)

P(correct | normalised answer string), Laplace-smoothed, counted on the by-image TRAIN half pooled over
the 8 benchmarks (Qwen 41,191 distinct strings / 90,112 rows; MedGemma 77,196 / 115,692), argmax on the
held-out half, ties broken by vote count. No image, no hidden state, no question.

```
gen       greedy   prior   probe   prior-greedy  probe-prior   verdict(probe > prior)
qwen      0.3034  0.3271  0.3848      +0.0237      +0.0577     WIN [+0.0450,+0.0712]
medgemma  0.3241  0.3372  0.3677      +0.0131      +0.0305     WIN [+0.0252,+0.0362]
```

- The prior alone recovers 29.1 percent (Qwen) / 30.1 percent (MedGemma) of the probe's judge gain.
- Per benchmark the prior beats greedy on 6/8 for both generators; it LOSES on Qwen VQA-RAD (-0.0412),
  Qwen PathVQA (-0.0055) and MedGemma VQA-RAD (-0.0206).
- The probe beats the answer prior on EVERY benchmark for both generators - Qwen WIN on 6/8, TIE on
  VQA-RAD (n=97) and VQA-Med; MedGemma WIN on 5/8, TIE on SLAKE / VQA-RAD / VQA-Med; no losses anywhere.
  So the probe is NOT merely an answer prior, but about 30 percent of what it is credited with is
  available from a bag-of-strings lookup table, and that has never been subtracted.
- Biggest single case: Qwen OmniMedVQA, greedy 0.4481 -> prior 0.5911 -> probe 0.6375. The prior alone
  delivers 75 percent of that cell's gain.

## 5. JUDGE FAMILY - is there a same-family self-preference?

The judge is Lingshu-32B (a Qwen2.5-VL-32B finetune): same family as Lingshu-7B AND Qwen2.5-VL-7B,
different family from MedGemma (Gemma 3 + SigLIP). Conditioned leniency, full benchmarks
(s09_judgefamily.json):

```
generator   greedy judge  greedy EM   judge-EM   P(judge=1|EM=0)  P(judge=1|F1=0)
lingshu       0.3936      0.3427      +0.0509        0.1389           0.1126
qwen          0.3082      0.2060      +0.1022        0.1674           0.1586
medgemma      0.3252      0.2381      +0.0871        0.1659           0.1381
```

NO same-family signature. The properly conditioned quantity - how often the judge overrides an
exact-match failure - is 0.1674 for Qwen and 0.1659 for MedGemma, a 0.0015 difference, while the
supposedly most-favoured generator (Lingshu: same family AND the judge's own base model) is the LOWEST
at 0.1389. The raw judge-minus-EM gap (Qwen +0.1022 > MedGemma +0.0871 > Lingshu +0.0509) is a base-rate
artefact of how paraphrastic each generator's output is, not self-preference. The candidate-pool version
gives the same ordering (0.1534 / 0.1793 / 0.1359).
CAVEAT: this rules out a FAMILY effect on this axis only. It does not rule out judge error in general,
and R2 means MedGemma's judge labels are partly measuring text degeneration.

## 6. THE "MATCHED PROTOCOL" COMPARISON (MedGemma +0.0248 / Lingshu +0.0375 / Qwen +0.0178)

WHAT WAS MATCHED (head_final_stack.py lines 100-130): no dedicated train-domain feature cache; the same
four benchmarks (PathVQA, SLAKE, VQA-RAD, RadImageNet); training only on their by-image train halves;
5 seeds; 4 threads (all three artifacts carry a provenance block, git d8b77bd16f58).

WHAT WAS NOT MATCHED: training rows (MedGemma 20,102 / Lingshu 13,304 / Qwen 14,436); distinct
candidates per question (6.58 / 4.36 / 4.73 - the doc does state this); ensemble layers (22,24,27 for
MedGemma vs 18,20,22); and, undeclared, THE GENERATION CONFIGURATION - MedGemma's pool is 100 percent
truncated (R1) and the other two are not. The last one dominates the other three.

DOC vs ARTIFACTS. TRANSFER_WALL section 14's matched table is arithmetically clean: every cell and every
macro is pooled_ens for all three generators, and I verified all 15 numbers against the artifacts
(MedGemma +0.0246 / +0.0182 / -0.0103 / +0.0667, macro +0.0248; Lingshu +0.0320 / +0.0152 / -0.0206 /
+0.1235, macro +0.0375; Qwen +0.0302 / +0.0333 / -0.0619 / +0.0697, macro +0.0178). Section 14's first
table also checks out (Qwen full-protocol macro over those 4 = +0.054689; MedGemma +0.024813; Lingshu
+0.043263, which comes from head_final_stack_REGRESSION_2026-09-13.json, i.e. the post-refactor 4-thread
run and NOT the shipped PVFIXED run whose value is +0.047949 - a cross-run comparison that section 13
itself warns against, though here it is the conservative choice).

BUT THE VERDICT STRINGS DO NOT. Each artifact's VERDICT quotes max(macro) - its own best arm, selected
on the evaluation half - and the winning arm differs by generator: MedGemma pooled_ens_sc +0.0292,
Lingshu-matched pooled_ens_sc +0.0404, Qwen-matched pooled_singlelayer +0.0224. Quoting the three
VERDICT strings side by side would give +0.0292 / +0.0404 / +0.0224 - three different methods. The doc
avoided this. The 8-benchmark headline pair commits it (R5).

## 7. VERDICT TABLE

| # | claim | verdict | evidence |
|---|---|---|---|
| 1 | Qwen probe beats Qwen greedy, +0.0820 macro, 8/8, judge currency | **SOLID** | independently refitted, bitwise identical to head_final_stack_qwen_2026-09-13.json on all 8 cells; pool 99.98 percent clean; image-clustered CI [+0.0708,+0.0924], benchmark-level [+0.0457,+0.1217] |
| 2 | ...and it is not a judge artefact | **HOLDS WITH CAVEAT** | survives EM +0.0276 / strict EM +0.0187 / F1 +0.0317, all WIN - but 3.0x smaller, and only 3/8 cells individually significant under EM |
| 3 | ...and it is not an answer prior | **HOLDS WITH CAVEAT** | probe minus prior +0.0577 WIN, no benchmark lost - but the prior alone is +0.0237 (29 percent of the gain) and was never reported |
| 4 | ...and it is not self-consistency | **SOLID** | majority vote is -0.0114 macro (LOSS, 1/8 positive) on Qwen |
| 5 | MedGemma +0.0481 macro, 8/8 | **WRONG** | the 8-sample pool is 100 percent truncated at 64 tokens on all 8 benchmarks; greedy is not; 58.5 percent of candidates are chat-template garbage; on clean questions the gain is +0.0096 |
| 6 | MedGemma "8/8" | **OVERSTATED even on its own terms** | 8/8 is pooled_singlelayer; the shipped pooled_ens arm is +0.0447 at 7/8 |
| 7 | "the probe is not a property of Lingshu, medical finetuning, or the Qwen LM" (section 14) | **OVERSTATED** | rests entirely on the defective MedGemma run; Lingshu and Qwen are the same LM family, so there is currently NO valid cross-family replication |
| 8 | "the pooled lever is the same size on both generators" (section 12, +0.0554 vs +0.0594) | **HOLDS WITH CAVEAT** | verified in the artifacts, but neither artifact records its thread count and section 13 prices that at 0.0033, most of the 0.0040 difference |
| 9 | TRANSFER_WALL section 14's matched-protocol table | **SOLID** | all 15 numbers verified against the artifacts; consistent arm (pooled_ens) throughout |
| 10 | The headline judge gain generalises to how a user would score the system | **WRONG for MedGemma, WEAK for Lingshu, HOLDS for Qwen** | strict EM: MedGemma -0.0751 (0/8), Lingshu +0.0103 TIE, Qwen +0.0187 WIN |

## WHAT I WOULD TELL THE OWNER

"Our probes on Qwen produce improvements across the board" is TRUE AND DEFENSIBLE IN JUDGE CURRENCY, and
the Qwen half of the claim is the most solid result in this line of work: it is the only generator whose
gain survives exact match, token-F1, the majority-vote control AND the answer-prior control. If you
build a direction, build it on Qwen.

Three things must go into any writeup or it will not survive review:

1. Report BOTH currencies. Qwen is +0.0814 judge / +0.0276 EM. Publishing only the first repeats the
   error the project already caught on Lingshu.
2. Report the ANSWER-PRIOR control (+0.0237 Qwen / +0.0131 MedGemma). It is cheap, it is 30 percent of
   the gain, and a reviewer will ask for it.
3. WITHDRAW MEDGEMMA until it is regenerated. As it stands the third generator is a broken decode, and
   with it goes the only cross-family evidence in the project. Regenerating MedGemma's 8-by-8 sc8 dumps
   with correct end-of-turn stop handling is the single highest-value experiment available, and it would
   re-open the cross-family claim properly.

## VERIFIED CLEAN

- Qwen pooled_singlelayer reproduces BITWISE from an independent reimplementation on 8/8 cells
  (s10_tables.out section 6, diff +0.0000 everywhere); macro +0.08137 both.
- Held-out question counts match the artifacts exactly for both generators on all 8 cells
  (1623 / 330 / 97 / 1004 / 5152 / 4461 / 1807 / 3978; 18,452 total each).
- Every candidate slot in every sc8 pool has a judge label for all three generators (0 dropped
  questions; s07_currency.out reports an empty dropped-dict for all 16 cells).
- TRANSFER_WALL section 12 table (10 numbers) and section 14 tables (19 numbers) verified against the
  artifacts.
- Qwen and Lingshu candidate pools are clean: at most 0.64 percent truncated, at most 0.02 percent
  degenerate.
- head_final_stack.py's dedupe of the four overlapping Qwen train caches is real: 132,274 usable rows
  become 55,065 unique keys (s02_rowcounts.out), matching pooled_rows 145,085 once the eval train halves
  are added.
- Per-seed spread of the probe's judge macro is small: Qwen 5 seeds +0.0738 to +0.0776 (spread 0.0038),
  MedGemma 2 seeds +0.0434 / +0.0434.

## COULD NOT VERIFY

- The thread count of the two headline runs (PVFIXED, qwen) - no provenance block exists (R7).
- Whether MedGemma's sc8 truncation came from a 64-token max_tokens alone or from missing end-of-turn
  stop ids: src/labeling/run_openvqa.py line 180 passes no stop / stop_token_ids to SamplingParams at
  all, and the MedGemma sc8 runner JSONs I found (runners/auto_gpu0_wave94.json, auto_gpu1_wave89.json,
  auto_gpu1_wave77.json) only re-explode and re-judge, they do not regenerate. The FACT of the
  truncation is verified; its exact cause is not.
- Whether the MedGemma clean-subset estimate (+0.0096) is unbiased - it conditions on a property of the
  pool, so treat it as a diagnostic only.
- A Lingshu single-layer refit under my own harness (I used the em-rescore agent's frozen 24-head
  ensemble), so the Lingshu row is ensemble-vs-single-layer against the other two.

## FILES

- replication_currency_2026-09-20.json - every number computed here (per-cell, per-currency, CIs,
  answer prior, degeneration, generation-config audit, artifact facts).
- s06_dump_table.py/.json/.out, s07_currency.py/.out, s09_judgefamily.py/.json, s10_tables.py/.out,
  s11_length.py/.json/.out, s12_degenerate.py/.json/.out, s13_merge.py, s08_check_doc_numbers.py.
- refit/qwen_L20.npz + .meta.json (predecessor, 5 seeds, 4 threads, 13.3 min),
  refit/medgemma_L24.npz + .meta.json (mine, 2 seeds, 2 threads, 5.0 min), s05_retry_medgemma.sh.
