# Shipped-method numbers (single T = 0.7), from `final_eval_FINAL_cap320_T07.json` unless noted

## Macro over the 8 open benchmarks (both image folds, 36,869 questions)

| currency | Δ vs greedy | CI | verdict | n positive |
|---|---|---|---|---|
| judge | +0.0798 | [+0.0711, +0.0887] | WIN | 6/8 |
| xjudge | +0.0710 | [+0.0622, +0.0797] | WIN | 7/8 |
| em | +0.0121 | [+0.0030, +0.0210] | WIN | 4/8 |
| f1 | +0.0232 | [+0.0160, +0.0303] | WIN | 6/8 |
| llava_recall | +0.0324 | [+0.0250, +0.0396] | WIN | 7/8 |
| llava_f1 | +0.0237 | [+0.0165, +0.0307] | WIN | 6/8 |
| self-consistency judge | -0.0094 | [-0.0157, -0.0034] | LOSS | 2/8 |
| self-consistency xjudge | -0.0074 | [-0.0137, -0.0011] | LOSS | 2/8 |
| self-consistency em | -0.0015 | [-0.0072, +0.0040] | TIE | 3/8 |
| self-consistency f1 | -0.0101 | [-0.0149, -0.0056] | LOSS | 2/8 |
| self-consistency llava_recall | -0.0128 | [-0.0179, -0.0080] | LOSS | 1/8 |

keys: `macro_over_open8.all.pick_minus_greedy_<c>`, `...sc_minus_greedy_<c>`

## Image half 1 only (`final_fold1_headline_2026-10-16.json` policies.T07.macro)

```
{
 "judge": {
  "macro": 0.08456642832802867,
  "ci": [
   0.0728722712602788,
   0.09648403614605246
  ],
  "verdict": "WIN",
  "n_positive": 7
 },
 "xjudge": {
  "macro": 0.07207620362469025,
  "ci": [
   0.060086501384865376,
   0.08413843498889545
  ],
  "verdict": "WIN",
  "n_positive": 7
 },
 "em": {
  "macro": 0.016450228027830797,
  "ci": [
   0.0025826710552022897,
   0.02992243880337293
  ],
  "verdict": "WIN",
  "n_positive": 4
 },
 "f1": {
  "macro": 0.02877052799266925,
  "ci": [
   0.018202331041991312,
   0.038794083127574555
  ],
  "verdict": "WIN",
  "n_positive": 7
 },
 "llava_recall": {
  "macro": 0.03893918833198059,
  "ci": [
   0.02877055458980284,
   0.048944164913115255
  ],
  "verdict": "WIN",
  "n_positive": 7
 }
}
```

## Per benchmark (both folds)

| benchmark | n | images | greedy J | pick J | Δ J [CI] | Δ MG [CI] | Δ EM [CI] | Δ LLaVA | oracle@8 J | SC−greedy J | sel_eff (pick/oracle) |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| pathvqa_open | 3357 | 849 | 0.3140 | 0.3664 | +0.0524 [+0.0389, +0.0661] | +0.0420 [+0.0295, +0.0547] | +0.0185 [+0.0069, +0.0301] | +0.0278 | 0.4808 | -0.0200 | 0.7621 |
| slake_open | 645 | 96 | 0.7302 | 0.7876 | +0.0574 [+0.0291, +0.0868] | +0.0419 [+0.0131, +0.0715] | +0.0171 [-0.0096, +0.0452] | +0.0300 | 0.8791 | +0.0124 | 0.8959 |
| vqa_rad_open | 200 | 120 | 0.4900 | 0.4900 | +0.0000 [-0.0564, +0.0580] | +0.0100 [-0.0474, +0.0680] | -0.0250 [-0.0895, +0.0357] | +0.0048 | 0.6300 | -0.0300 | 0.7778 |
| radimagenet_open | 2000 | 1000 | 0.3210 | 0.4545 | +0.1335 [+0.1165, +0.1500] | +0.1160 [+0.1005, +0.1320] | +0.0730 [+0.0605, +0.0860] | +0.0770 | 0.5120 | +0.0035 | 0.8877 |
| kvasir_x1_open | 10121 | 2860 | 0.2849 | 0.3992 | +0.1143 [+0.1068, +0.1218] | +0.1048 [+0.0970, +0.1128] | -0.0381 [-0.0441, -0.0322] | +0.0339 | 0.4696 | -0.0149 | 0.8500 |
| omnimed_open | 8883 | 8287 | 0.5164 | 0.6768 | +0.1604 [+0.1525, +0.1685] | +0.1485 [+0.1406, +0.1567] | +0.0738 [+0.0662, +0.0816] | +0.0630 | 0.7007 | +0.0000 | 0.9659 |
| vqamed_open | 3663 | 3663 | 0.0947 | 0.0909 | -0.0038 [-0.0131, +0.0052] | -0.0027 [-0.0126, +0.0068] | -0.0137 [-0.0227, -0.0049] | -0.0095 | 0.2102 | -0.0085 | 0.4325 |
| gemex_open | 8000 | 3514 | 0.3974 | 0.5218 | +0.1244 [+0.1159, +0.1330] | +0.1072 [+0.0987, +0.1160] | -0.0091 [-0.0162, -0.0021] | +0.0325 | 0.5864 | -0.0180 | 0.8898 |

selection efficiency macro (mean of pick/oracle over 8): 0.8077; coverage wall macro (1 − mean oracle): 0.4414

## Comparability cells (closed halves, VQA-Med 2019 official test)

- **vqa_rad_closed** n=251: judge +0.0319 [-0.0079, +0.0736]; em +0.0319 [-0.0079, +0.0736]; f1 +0.0319 [-0.0079, +0.0736]; llava_closed +0.0319 [-0.0079, +0.0736]; xjudge +0.0319 [-0.0079, +0.0736]
  greedy/pick judge 0.7610 / 0.7928
- **pathvqa_closed** n=3362: judge +0.0297 [+0.0197, +0.0397]; em +0.0297 [+0.0197, +0.0397]; f1 +0.0297 [+0.0197, +0.0397]; llava_closed +0.0297 [+0.0197, +0.0397]; xjudge +0.0300 [+0.0200, +0.0401]
  greedy/pick judge 0.8679 / 0.8977
- **slake_closed** n=416: judge -0.0144 [-0.0443, +0.0164]; em -0.0192 [-0.0501, +0.0121]; f1 -0.0176 [-0.0479, +0.0135]; llava_closed -0.0048 [-0.0310, +0.0215]; xjudge -0.0144 [-0.0443, +0.0164]
  greedy/pick judge 0.8341 / 0.8197
- **vqamed2019_test** n=500: judge +0.0080 [-0.0180, +0.0320]; em -0.0060 [-0.0320, +0.0200]; f1 -0.0001 [-0.0242, +0.0241]; llava_recall +0.0041 [-0.0201, +0.0286]; llava_f1 +0.0016 [-0.0224, +0.0259]; imageclef -0.0060 [-0.0300, +0.0180]; xjudge +0.0020 [-0.0240, +0.0280]
  greedy/pick judge 0.5300 / 0.5380; imageclef 0.3460 / 0.3400

## Full official splits (`full_official_splits`), open+closed combined

```
{
 "vqa_rad": {
  "n": 451,
  "n_open": 200,
  "n_closed": 251,
  "pick_judge": 0.6585365853658537,
  "pick_llava_open_recall": 0.3900416666666666,
  "pick_llava_closed_acc": 0.7888446215139442,
  "greedy_judge": 0.6407982261640798,
  "greedy_llava_open_recall": 0.38520833333333326,
  "greedy_llava_closed_acc": 0.7569721115537849,
  "sc_judge": 0.6297117516629712,
  "sc_llava_open_recall": 0.3744583333333333,
  "sc_llava_closed_acc": 0.7609561752988048
 },
 "pathvqa": {
  "n": 6719,
  "n_open": 3357,
  "n_closed": 3362,
  "pick_judge": 0.6322369400208364,
  "pick_llava_open_recall": 0.33425583191179026,
  "pick_llava_closed_acc": 0.89738251041047,
  "greedy_judge": 0.5911593987200476,
  "greedy_llava_open_recall": 0.3064532713981695,
  "greedy_llava_closed_acc": 0.8676383105294467,
  "sc_judge": 0.5773180532817384,
  "sc_llava_open_recall": 0.28893158366910543,
  "sc_llava_closed_acc": 0.8599048185603807
 },
 "slake_en": {
  "n": 1061,
  "n_open": 645,
  "n_closed": 416,
  "pick_judge": 0.8001885014137606,
  "pick_llava_open_recall": 0.8114909896305245,
  "pick_llava_closed_acc": 0.7163461538461539,
  "greedy_judge": 0.7709707822808671,
  "greedy_llava_open_recall": 0.7814658210007046,
  "greedy_llava_closed_acc": 0.7211538461538461,
  "sc_judge": 0.7728557964184731,
  "sc_llava_open_recall": 0.7822926943857177,
  "sc_llava_closed_acc": 0.7115384615384616
 }
}
```

## Mixed-temperature test (2026-10-18, frozen final probes, `final_eval_FINAL_cap320_MIX_<policy>.json`)

| policy | samples | J | MG | EM | LLaVA |
|---|---|---|---|---|---|
| T07 | 8 @ T=0.7 | +0.0798 [+0.0711, +0.0887] | +0.0710 [+0.0622, +0.0797] | +0.0121 [+0.0030, +0.0210] | +0.0324 [+0.0250, +0.0396] |
| MIX8 | 2 each @ 0.2/0.4/0.7/1.0 | +0.0726 [+0.0641, +0.0812] | +0.0628 [+0.0541, +0.0716] | +0.0101 [+0.0016, +0.0186] | +0.0293 [+0.0222, +0.0365] |
| T07_16 | 16 @ T=0.7 | +0.0984 [+0.0888, +0.1079] | +0.0674 [+0.0566, +0.0782] | +0.0140 [+0.0043, +0.0237] | +0.0377 [+0.0291, +0.0461] |
| MIX16 | 4 each @ 0.2/0.4/0.7/1.0 | +0.0927 [+0.0833, +0.1021] | +0.0778 [+0.0689, +0.0867] | +0.0108 [+0.0015, +0.0202] | +0.0356 [+0.0281, +0.0434] |

⚠ MedGemma labels are incomplete for the extra candidates of 16-sample pools (unlabelled → excluded); treat MG for T07_16 / MIX16 as unreliable.


folds: {"0": {"train_rows": 115934, "train_rows_original_domains": 31237, "eval_images": 10403, "probes": "fitted, hidden 256, epochs 30, sched none, seeds 8"}, "1": {"train_rows": 116000, "train_rows_original_domains": 31121, "eval_images": 10451, "probes": "fitted, hidden 256, epochs 30, sched none, seeds 8"}}
bootstrap: {"n": 10000, "seed": 20261017, "cluster": "img_md5 within cell, paired"}
