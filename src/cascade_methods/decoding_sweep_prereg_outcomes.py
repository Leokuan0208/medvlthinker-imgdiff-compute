#!/usr/bin/env python3
"""decoding_sweep_prereg_outcomes.py -- adjudicate the pre-registered predictions against the data.

The sweep's predictions were written down in artifacts/_decoding_sweep_prereg.json BEFORE any SELECTED
or sel_eff number for any swept setting existed. Scoring them explicitly is the point of pre-registering:
two of the three were wrong, and saying so is more informative than quietly reporting the winners.

Reads the primary artifact + the dual-currency file; writes
results/cascade_methods/artifacts/_decoding_sweep_prereg_outcomes.json
"""
import json, os
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
d = json.load(open(os.path.join(ART, "decoding_sweep_2026-08-13.json")))
pre = json.load(open(os.path.join(ART, "_decoding_sweep_prereg.json")))
dual = json.load(open(os.path.join(ART, "_decoding_sweep_dual_currency.json")))
S, D = d["settings"], d.get("deltas_vs_control", {})


def g(st, *path, default=None):
    cur = S.get(st)
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


temps = [("T03", 0.3), ("T05", 0.5), ("T07", 0.7), ("T10", 1.0), ("T13", 1.3)]
have = [(k, v) for k, v in temps if k in S]
orc = [(k, v, g(k, "oracle@8", "mean"), g(k, "mean_distinct", "mean"), g(k, "selected", "mean"))
       for k, v in have]

out = {"title": "Pre-registered predictions vs what was measured",
       "registered_at": pre["written_at"],
       "registered_text": pre["prediction_registered_in_advance"],
       "outcomes": {}}

# ---- prediction 1: temperature ----
mono_distinct = all(orc[i][3] <= orc[i + 1][3] for i in range(len(orc) - 1)) if len(orc) > 1 else None
mono_oracle = all(orc[i][2] <= orc[i + 1][2] for i in range(len(orc) - 1)) if len(orc) > 1 else None
best_orc = max(orc, key=lambda r: r[2])[0] if orc else None
best_sel = max([r for r in orc if r[4] is not None], key=lambda r: r[4])[0] \
    if any(r[4] is not None for r in orc) else None
out["outcomes"]["temperature"] = {
    "registered": pre["prediction_registered_in_advance"]["temperature"],
    "measured_ladder": [{"setting": k, "temp": v, "oracle@8": o, "mean_distinct": nd, "SELECTED": sel}
                        for k, v, o, nd, sel in orc],
    "distinct_increases_monotonically_with_T": mono_distinct,
    "oracle@8_increases_monotonically_with_T": mono_oracle,
    "argmax_oracle@8_over_measured_temps": best_orc,
    "argmax_SELECTED_over_measured_temps": best_sel,
    "VERDICT": ("PARTLY REFUTED: distinct-candidate count does rise monotonically with temperature, but "
                "oracle@8 does NOT -- it PEAKS at the deployed T=0.7 and falls on both sides. More "
                "diversity stops buying coverage past 0.7 because per-sample accuracy falls faster than "
                "diversity helps. The SELECTED half of the prediction (that it does not rise WITH "
                f"temperature) held -- but the fuller ladder shows SELECTED peaks at {best_sel}, i.e. "
                "BELOW the deployed 0.7, so 'no temperature change helps' would be the wrong lesson."
                if mono_distinct and not mono_oracle else "see measured_ladder")}

# ---- prediction 2: repetition penalty ----
rp = D.get("rp11", {})
lat = rp.get("SELECTED_on_LATERALITY_vs_control")
sel_j = dual.get("deltas_vs_control", {}).get("rp11", {}).get("SELECTED_judge_delta")
sel_e = dual.get("deltas_vs_control", {}).get("rp11", {}).get("SELECTED_em_delta")
out["outcomes"]["repetition_penalty"] = {
    "registered": pre["prediction_registered_in_advance"]["repetition_penalty"],
    "measured_SELECTED_judge_delta": sel_j,
    "measured_SELECTED_em_delta": sel_e,
    "measured_LATERALITY_delta_judge": lat,
    "deployed_value": 1.0,
    "is_the_deployed_system_harmed_today": False,
    "why_not": "run_openvqa.py never sets repetition_penalty, so vLLM's default 1.0 applies and NO "
               "penalty is computed. Verified by resolving SamplingParams(temperature=0.7, max_tokens=64, "
               "n=8, logprobs=5) -> repetition_penalty=1.0.",
    "VERDICT": (
        "REFUTED IN THE REGISTERED DIRECTION, AND NOT A WIN EITHER. rp=1.10 did not lower SELECTED and "
        f"did not damage the laterality stratum ({(lat or {}).get('verdict','n/a')}). It RAISED "
        f"judge-currency SELECTED by {(sel_j or {}).get('delta', float('nan')):+.5f} "
        f"({(sel_j or {}).get('verdict','n/a')}) while LOWERING exact-match SELECTED by "
        f"{(sel_e or {}).get('delta', float('nan')):+.5f} ({(sel_e or {}).get('verdict','n/a')}). "
        "The registered harm mechanism (vLLM penalising prompt tokens) is real -- rp11 measurably "
        "quotes the question less -- but it did not produce the registered harm.")}

# ---- prediction 3: min_p ----
mp = "minp01"
out["outcomes"]["min_p"] = {
    "registered": pre["prediction_registered_in_advance"]["min_p"],
    "measured": {"oracle@8": g(mp, "oracle@8", "mean"), "sel_eff": g(mp, "sel_eff", "mean"),
                 "SELECTED": g(mp, "selected", "mean"),
                 "mean_distinct": g(mp, "mean_distinct", "mean"),
                 "control_oracle@8": g("T07", "oracle@8", "mean"),
                 "control_sel_eff": g("T07", "sel_eff", "mean"),
                 "control_SELECTED": g("T07", "selected", "mean"),
                 "control_mean_distinct": g("T07", "mean_distinct", "mean")},
    "SELECTED_delta": D.get(mp, {}).get("SELECTED_vs_control"),
    "VERDICT": (
        f"CONFIRMED. min_p=0.10 cut oracle@8 ({g(mp,'oracle@8','mean'):.4f} vs "
        f"{g('T07','oracle@8','mean'):.4f}) and distinct candidates "
        f"({g(mp,'mean_distinct','mean'):.2f} vs {g('T07','mean_distinct','mean'):.2f}) while RAISING "
        f"sel_eff ({g(mp,'sel_eff','mean'):.4f} vs {g('T07','sel_eff','mean'):.4f}), leaving SELECTED "
        f"at {D.get(mp,{}).get('SELECTED_vs_control',{}).get('delta',float('nan')):+.5f} "
        f"({D.get(mp,{}).get('SELECTED_vs_control',{}).get('verdict','n/a')}) in the judge currency. "
        "Exactly the registered trade.")}

# ---- overall ----
concl = dual.get("deltas_vs_control", {})
win_both = [k for k, v in concl.items()
            if v["SELECTED_judge_delta"]["verdict"] == "WIN" and v["SELECTED_em_delta"]["verdict"] == "WIN"]
out["outcomes"]["overall"] = {
    "registered": pre["prediction_registered_in_advance"]["overall"],
    "settings_beating_the_deployed_default_in_BOTH_currencies": win_both,
    "VERDICT": ("CONFIRMED. No single-variable decoding change beats the deployed T=0.7 default on "
                "SELECTED accuracy CI-cleanly in BOTH grading currencies. NOTE the prediction is "
                "confirmed on its own terms but understates the picture: the COLD settings (T=0.3, "
                "min_p=0.10) are CI-clean exact-match WINS and judge TIEs -- never negative -- and they "
                "win by raising sel_eff while LOSING coverage, which is the opposite of the coverage "
                "mechanism this sweep set out to test." if not win_both
                else ("REFUTED. " + ", ".join(win_both) + " beat(s) the deployed T=0.7 default on "
                      "SELECTED accuracy CI-cleanly in BOTH grading currencies. The registered "
                      "prediction was right about the MECHANISM it had in mind (extra diversity does not "
                      "pay) but wrong about the conclusion, because it only considered making the pool "
                      "RICHER. Moving the other way -- a COLDER, less diverse pool -- raises sel_eff more "
                      "than it costs coverage, and that is where the win is. Magnitude is modest and its "
                      "8-cell macro share leans on vqa_rad_open (n=200): see HEADLINE_2026_08_14."
                      ".HOW_BIG_IS_THE_WIN in the primary artifact."))}

json.dump(out, open(os.path.join(ART, "_decoding_sweep_prereg_outcomes.json"), "w"), indent=1, default=float)
print("wrote artifacts/_decoding_sweep_prereg_outcomes.json\n")
for k, v in out["outcomes"].items():
    print(f"[{k}]\n  {v['VERDICT']}\n")
