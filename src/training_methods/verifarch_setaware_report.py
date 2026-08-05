#!/usr/bin/env python3
"""Print the verifarch_setaware artifact as the tables the round is reported in.
Read-only: it never recomputes anything, it only formats what the artifact stores."""
import json
import os
import sys

P = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.expanduser("~/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/"
                       "verifarch_setaware_2026-08-04.json")
d = json.load(open(P))

print("=" * 100)
print("NULL TEST      pass =", d["null_test"]["pass"],
      "  max abs deviation =", d["null_test"]["max_abs_deviation"])
print("DISJOINTNESS   train_images=%d eval_images=%d pixel-md5 intersection=%d"
      % (d["disjointness"]["train_images"], d["disjointness"]["eval_images"],
         d["disjointness"]["image_pixel_md5_intersection"]))
hv = d["harness_validation"]
print("HARNESS        published bar %.6f | refit cpu seed0 %.6f (max abs dev %s) | same cfg gpu %.6f"
      % (hv["published_bar"]["sel_eff"], hv["refit_here"]["cpu"]["sel_eff"],
         hv["max_abs_deviation_cpu_seed0"],
         hv["refit_here"].get("cuda", hv["refit_here"]["cpu"])["sel_eff"]))
print("               device swing at a SINGLE seed =", hv["device_swing_seed0"])

pr = d["pre_registration"]
print("\nPRE-REGISTRATION (train CV only, %d folds x %d seeds)" % (pr["folds"], pr["cv_seeds"]))
for k in ["headline_setaware", "point_control", "best_overall_any_arch"]:
    c = pr[k]
    print("  %-22s %s/%s h%s wd%s -> cv_sel_eff %.4f" %
          (k, c["arch"], c["objective"], c["hidden"], c["wd"], c["cv_sel_eff"]))
print("\n  full CV grid (train only), sorted:")
for r in sorted(pr["grid"], key=lambda r: -r["cv_sel_eff"]):
    print("    %-16s %-8s h%-4s wd%-5s  cv_sel_eff %.4f (sd %.4f)  cv_auroc %.4f" %
          (r["arch"], r["objective"], r["hidden"], r["wd"], r["cv_sel_eff"],
           r["cv_sel_eff_sd"], r["cv_auroc"]))

print("\n" + "=" * 100)
print("ARMS  (seed-averaged rank ensemble is the deployable number)")
hdr = ("%-22s %-14s %-8s | seed mean   sd     [min,max]   | ENSEMBLE  contested  "
       "slake/vqarad/pathvqa            d vs incumbent            guard")
print(hdr)
for tag, a in d["arms"].items():
    if "alias_of" in a:
        print("%-22s  == %s" % (tag, a["alias_of"]))
        continue
    c, s, e = a["config"], a["seed_stats"], a["ensemble"]
    pd_ = e["per_ds"]
    print("%-22s %-14s %-8s | %.4f %.4f [%.4f,%.4f] | %.6f %.6f  %.4f/%.4f/%.4f  %+.4f %s  %s"
          % (tag, c["arch"], c["objective"], s["mean"], s["sd"], s["min"], s["max"],
             e["sel_eff"], e["contested_sel_eff"], pd_["slake_open"], pd_["vqa_rad_open"],
             pd_["pathvqa_open"], e["vs_incumbent"]["d_sel_eff"],
             str(e["vs_incumbent"]["ci"]), e["guardrail_clean"]))

hh = d["setaware_vs_pointwise_SAME_FEATURES"]
print("\n" + "=" * 100)
print("SET-AWARE vs POINTWISE, identical features, same harness")
print("  set-aware  %-22s ensemble %.6f" % (hh["setaware_arm"], hh["setaware_ensemble_sel_eff"]))
print("  pointwise  %-22s ensemble %.6f" % (hh["pointwise_arm"], hh["pointwise_ensemble_sel_eff"]))
print("  d = %+.6f  95%% CI %s   (contested: %+.6f  %s)"
      % (hh["d_sel_eff"], hh["ci"], hh["d_sel_eff_contested"], hh["ci_contested"]))
print("  per-seed paired deltas:", hh["per_seed_paired_delta"],
      " mean", hh["per_seed_paired_delta_mean"])

print("\nCONTEXT ABLATION -- does the head use its siblings?")
for t, r in d["context_ablation"]["arms"].items():
    print("  %-20s true %.6f | singleton %.6f (%+.4f) | foreign %.6f (%+.4f)"
          % (t, r["true_pool"], r["singleton"], -r["delta_true_minus_singleton"],
             r["foreign"], -r["delta_true_minus_foreign"]))

print("\nCENTROID ABLATION")
for a, r in d["centroid_ablation"]["arms"].items():
    print("  %-16s ensemble %.6f  contested %.6f  seed mean %.6f (sd %.4f)"
          % (a, r["ensemble_sel_eff"], r["contested"], r["seed_mean"], r["seed_sd"]))

print("\nBY POOL SIZE (n distinct candidates), recoverable items only")
for t, r in d["by_pool_size"]["arms"].items():
    print("  %-18s " % t + "  ".join("%s: %.4f (n=%d)" % (k, v["sel_eff"], v["n"])
                                     for k, v in r.items()))

print("\nFUSION (parameter-free rank_avg; nothing fitted on eval)")
for k, v in d["fusion"]["arms"].items():
    print("  %-46s %.6f  contested %.6f  d_vs_inc %+.4f %s  guard %s"
          % (k, v["sel_eff"], v["contested_sel_eff"], v["vs_incumbent"]["d_sel_eff"],
             v["vs_incumbent"]["ci"], v["guardrail_clean"]))

print("\nCOST:", d["cost"]["extra_LLM_forward_passes_per_question_over_the_8_generations"],
      "extra LLM forward passes/question;", d["cost"]["head_evaluations_per_question"])
