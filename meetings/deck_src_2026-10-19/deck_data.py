"""deck_data.py — every number the deck shows, loaded from the artifacts (never typed by hand where an artifact
holds it). Each value carries the file it came from, so slides can print a source line."""
import json

ART = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
DK = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19"


def J(name):
    return json.load(open(f"{ART}/{name}"))


F07 = "final_eval_FINAL_cap320_T07.json"
fin = J(F07)
fold1 = J("final_fold1_headline_2026-10-16.json")["policies"]["T07"]
tmap = J("final_eval_FINAL_cap320_Tmap.json")
tmap_vs = J("tmap_vs_t07_2026-10-16.json")
mix = {p: J(f"final_eval_FINAL_cap320_MIX_{p}.json") for p in ["T07", "MIX8", "T07_16", "MIX16"]}
temp_ens = J("head_temp_ensemble_2026-08-30.json")
temp_cf = J("temp_crossfit_2026-10-15.json")
readout = J("readout_variants_2026-10-15.json")
wed = J("wed_vs_probe_2026-10-16.json")
cost = J("bestofn_vllm_2026-09-16.json")
vqm = J("vqamed2019_by_category_2026-10-16.json")
EX = json.load(open(f"{DK}/worked_example_12231.json"))

OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open"]
NAME = {"pathvqa_open": "PathVQA", "slake_open": "SLAKE", "vqa_rad_open": "VQA-RAD", "radimagenet_open": "RadImageNet-VQA",
        "kvasir_x1_open": "Kvasir-VQA-x1", "omnimed_open": "OmniMedVQA", "vqamed_open": "VQA-Med 2019 (C4)",
        "gemex_open": "GEMeX"}
SHORT = {"pathvqa_open": "PathVQA", "slake_open": "SLAKE", "vqa_rad_open": "VQA-RAD", "radimagenet_open": "RadImageNet",
         "kvasir_x1_open": "Kvasir-x1", "omnimed_open": "OmniMed", "vqamed_open": "VQA-Med", "gemex_open": "GEMeX"}


def cell(c):
    return fin["cells"][c]["all"]


def macro(cur, src=None):
    m = (src or fin)["macro_over_open8"]["all"][f"pick_minus_greedy_{cur}"]
    return m["macro"], m["ci"][0], m["ci"][1], m["verdict"], m["n_positive"]


def sc_macro(cur):
    m = fin["macro_over_open8"]["all"][f"sc_minus_greedy_{cur}"]
    return m["macro"], m["ci"][0], m["ci"][1], m["verdict"], m["n_positive"]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


GREEDY_MACRO = mean(cell(c)["greedy_judge"] for c in OPEN8)
PICK_MACRO = mean(cell(c)["pick_judge"] for c in OPEN8)
ORACLE_MACRO = mean(cell(c)["oracle_judge"] for c in OPEN8)
SELEFF = mean(cell(c)["pick_judge"] / cell(c)["oracle_judge"] for c in OPEN8)
N_Q = sum(cell(c)["n"] for c in OPEN8)
N_IMG = sum(cell(c)["n_images"] for c in OPEN8)


def verdict(lo, hi):
    return "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")


def pm(v, nd=4):
    return (f"{v:+.{nd}f}").replace("-", "−")


def ci(lo, hi, nd=4):
    return f"[{pm(lo, nd)}, {pm(hi, nd)}]"


# ---- temperature (SEP-pooled probe, held-out halves) — macro over 8 benchmarks
TEMPS = ["0.2", "0.4", "0.7", "1.0"]
T_J = [temp_cf["macro"][f"fixed_{t}_minus_greedy_judge"] for t in TEMPS]
T_EM = [temp_cf["macro"][f"fixed_{t}_minus_greedy_em"] for t in TEMPS]
T_ORACLE = [mean(temp_ens["cells"][c][t]["oracle"] for c in OPEN8) for t in TEMPS]
T_DISTINCT = [mean(temp_ens["cells"][c][t]["mean_distinct"] for c in OPEN8) for t in TEMPS]
T_PER = {c: [temp_ens["cells"][c][t]["minus_greedy"] for t in TEMPS] for c in OPEN8}
T_BEST = {c: temp_ens["cells"][c]["best_T"] for c in OPEN8}

if __name__ == "__main__":
    print("greedy/pick/oracle macro", GREEDY_MACRO, PICK_MACRO, ORACLE_MACRO, "sel_eff", SELEFF, "n", N_Q, N_IMG)
    print("T_J", T_J, "T_EM", T_EM, "oracle", T_ORACLE, "distinct", T_DISTINCT)
    print("best T", T_BEST)
    print("readout keys", list(readout["macro"].keys())[:40])
    print("cost keys", list(cost.keys()))
    print("wed keys", list(wed.keys()), list(wed.get("cells", {}).get("pathvqa_open", {}).keys())[:30])
    print("vqm keys", list(vqm.keys()))
    print("tmap_vs", json.dumps(tmap_vs.get("macro", {}), indent=0)[:600])
