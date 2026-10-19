#!/usr/bin/env python3
"""Dump every number the deck uses for the SHIPPED method (single T = 0.7) into NUMBERS_T07.md / .json,
each with its artifact and key, so the deck and script cite one checked source."""
import json

ART = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19"
F = "final_eval_FINAL_cap320_T07.json"
d = json.load(open(f"{ART}/{F}"))
f1 = json.load(open(f"{ART}/final_fold1_headline_2026-10-16.json"))["policies"]["T07"]
mix = {p: json.load(open(f"{ART}/final_eval_FINAL_cap320_MIX_{p}.json")) for p in ["T07", "MIX8", "T07_16", "MIX16"]}

OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open"]
CUR = ["judge", "xjudge", "em", "f1", "llava_recall"]
out, lines = {}, [f"# Shipped-method numbers (single T = 0.7), from `{F}` unless noted\n"]


def r4(x):
    return None if x is None else round(float(x), 4)


lines.append("## Macro over the 8 open benchmarks (both image folds, 36,869 questions)\n")
lines.append("| currency | Δ vs greedy | CI | verdict | n positive |\n|---|---|---|---|---|")
for c in CUR + ["llava_f1"]:
    m = d["macro_over_open8"]["all"][f"pick_minus_greedy_{c}"]
    out[f"macro.{c}"] = m
    lines.append(f"| {c} | {m['macro']:+.4f} | [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}] | {m['verdict']} | {m['n_positive']}/8 |")
for c in CUR:
    m = d["macro_over_open8"]["all"].get(f"sc_minus_greedy_{c}")
    if m:
        out[f"macro.sc.{c}"] = m
        lines.append(f"| self-consistency {c} | {m['macro']:+.4f} | [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}] | {m['verdict']} | {m['n_positive']}/8 |")
lines.append(f"\nkeys: `macro_over_open8.all.pick_minus_greedy_<c>`, `...sc_minus_greedy_<c>`\n")

lines.append("## Image half 1 only (`final_fold1_headline_2026-10-16.json` policies.T07.macro)\n")
lines.append("```\n" + json.dumps(f1.get("macro", f1), indent=1)[:2500] + "\n```\n")
out["fold1"] = f1.get("macro", f1)

lines.append("## Per benchmark (both folds)\n")
lines.append("| benchmark | n | images | greedy J | pick J | Δ J [CI] | Δ MG [CI] | Δ EM [CI] | Δ LLaVA | oracle@8 J | SC−greedy J | sel_eff (pick/oracle) |")
lines.append("|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|")
se = []
for c in OPEN8:
    a = d["cells"][c]["all"]
    out[f"cell.{c}"] = {k: a[k] for k in a}
    s = a["pick_judge"] / a["oracle_judge"]; se.append(s)

    def ci(k):
        lo, hi = a[f"ci_pick_minus_greedy_{k}"]
        return f"{a[f'pick_minus_greedy_{k}']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    lines.append(f"| {c} | {a['n']} | {a['n_images']} | {a['greedy_judge']:.4f} | {a['pick_judge']:.4f} | {ci('judge')} | "
                 f"{ci('xjudge')} | {ci('em')} | {a['pick_minus_greedy_llava_recall']:+.4f} | {a['oracle_judge']:.4f} | "
                 f"{a['sc_minus_greedy_judge']:+.4f} | {s:.4f} |")
lines.append(f"\nselection efficiency macro (mean of pick/oracle over 8): {sum(se)/8:.4f}; "
             f"coverage wall macro (1 − mean oracle): {1 - sum(d['cells'][c]['all']['oracle_judge'] for c in OPEN8)/8:.4f}\n")
out["sel_eff_macro"] = sum(se) / 8

lines.append("## Comparability cells (closed halves, VQA-Med 2019 official test)\n")
for c in ["vqa_rad_closed", "pathvqa_closed", "slake_closed", "vqamed2019_test"]:
    if c in d["cells"]:
        a = d["cells"][c]["all"]
        out[f"cell.{c}"] = {k: a[k] for k in a}
        ks = [k for k in a if k.startswith("pick_minus_greedy_")]
        lines.append(f"- **{c}** n={a['n']}: " + "; ".join(
            f"{k[18:]} {a[k]:+.4f} [{a['ci_'+k][0]:+.4f}, {a['ci_'+k][1]:+.4f}]" for k in ks if 'ci_' + k in a))
        lines.append(f"  greedy/pick judge {a['greedy_judge']:.4f} / {a['pick_judge']:.4f}" +
                     (f"; imageclef {a.get('greedy_imageclef'):.4f} / {a.get('pick_imageclef'):.4f}" if 'pick_imageclef' in a else ""))
lines.append("\n## Full official splits (`full_official_splits`), open+closed combined\n")
lines.append("```\n" + json.dumps(d["full_official_splits"], indent=1) + "\n```\n")
out["full_official_splits"] = d["full_official_splits"]

lines.append("## Mixed-temperature test (2026-10-18, frozen final probes, `final_eval_FINAL_cap320_MIX_<policy>.json`)\n")
lines.append("| policy | samples | J | MG | EM | LLaVA |\n|---|---|---|---|---|---|")
for p, lab in [("T07", "8 @ T=0.7"), ("MIX8", "2 each @ 0.2/0.4/0.7/1.0"), ("T07_16", "16 @ T=0.7"),
               ("MIX16", "4 each @ 0.2/0.4/0.7/1.0")]:
    m = mix[p]["macro_over_open8"]["all"]
    out[f"mix.{p}"] = {c: m[f"pick_minus_greedy_{c}"] for c in CUR}
    lines.append(f"| {p} | {lab} | " + " | ".join(
        f"{m[f'pick_minus_greedy_{c}']['macro']:+.4f} [{m[f'pick_minus_greedy_{c}']['ci'][0]:+.4f}, {m[f'pick_minus_greedy_{c}']['ci'][1]:+.4f}]"
        for c in ["judge", "xjudge", "em", "llava_recall"]) + " |")
lines.append("\n⚠ MedGemma labels are incomplete for the extra candidates of 16-sample pools (unlabelled → excluded); "
             "treat MG for T07_16 / MIX16 as unreliable.\n")
lines.append(f"\nfolds: {json.dumps(d['folds'])}\nbootstrap: {json.dumps(d['bootstrap'])}\n")
open(f"{OUT}/NUMBERS_T07.md", "w").write("\n".join(lines))
json.dump(out, open(f"{OUT}/NUMBERS_T07.json", "w"), indent=1, default=str)
print("\n".join(lines))
