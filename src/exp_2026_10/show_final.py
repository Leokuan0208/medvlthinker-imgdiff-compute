#!/usr/bin/env python3
"""Print the headline tables of the FINAL evaluation artifacts (read-only)."""
import json

A = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
for pol in ("T07", "Tmap"):
    a = json.load(open(f"{A}/final_eval_FINAL_cap320_{pol}.json"))
    print(f"\n=== policy {pol} — macro over the eight open cells, both folds ===")
    for k, v in sorted(a["macro_over_open8"]["all"].items()):
        if k.startswith("pick"):
            print(f"  {k.replace('pick_minus_greedy_', ''):14} {v['macro']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
                  f"{v['verdict']:4} {v['n_positive']}/8")
    print("  full official splits:")
    for fam, o in a["full_official_splits"].items():
        print(f"    {fam:9} n {o['n']} (open {o['n_open']}, closed {o['n_closed']}) | judge greedy {o['greedy_judge']:.4f} "
              f"pick {o['pick_judge']:.4f} | LLaVA open recall greedy {o['greedy_llava_open_recall']:.4f} pick "
              f"{o['pick_llava_open_recall']:.4f} | LLaVA closed acc greedy {o['greedy_llava_closed_acc']:.4f} pick "
              f"{o['pick_llava_closed_acc']:.4f}")
    for c in ("vqa_rad_closed", "pathvqa_closed", "slake_closed", "vqamed2019_test"):
        o = a["cells"][c]["all"]
        extra = ""
        if "pick_imageclef" in o:
            extra = f" | ImageCLEF strict acc greedy {o['greedy_imageclef']:.4f} pick {o['pick_imageclef']:.4f}"
        if "pick_llava_closed" in o:
            extra = f" | LLaVA closed acc greedy {o['greedy_llava_closed']:.4f} pick {o['pick_llava_closed']:.4f}"
        print(f"  {c:16} n {o['n']:5} | judge greedy {o['greedy_judge']:.4f} pick {o['pick_judge']:.4f} "
              f"[{o['ci_pick_minus_greedy_judge'][0]:+.4f},{o['ci_pick_minus_greedy_judge'][1]:+.4f}] | xjudge d "
              f"{o.get('pick_minus_greedy_xjudge', float('nan')):+.4f}{extra}")
    print("  per open cell (judge / xjudge / EM / LLaVA recall deltas):")
    for c in ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
              "vqamed_open", "gemex_open"]:
        o = a["cells"][c]["all"]
        print(f"    {c:17} n {o['n']:5} greedy {o['greedy_judge']:.4f} | dJ {o['pick_minus_greedy_judge']:+.4f} "
              f"dX {o.get('pick_minus_greedy_xjudge', float('nan')):+.4f} dEM {o['pick_minus_greedy_em']:+.4f} "
              f"dR {o['pick_minus_greedy_llava_recall']:+.4f}")
