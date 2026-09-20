#!/usr/bin/env python3
"""s10: print the deliverable tables from replication_currency_2026-09-20.json + the artifacts +
the em-rescore agent's Lingshu JSON. READ-ONLY, pure formatting."""
import json, os
import numpy as np
T = "/data/dan/audit_2026-09-18/tmp"
R = json.load(open(f"{T}/replication/replication_currency_2026-09-20.json"))
D6 = json.load(open(f"{T}/replication/s06_dump_table.json"))
EM = json.load(open(f"{T}/em-rescore/em_rescore_pooled_probe_2026-09-18.json"))
ART = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
B = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
     "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SH = {b: b.replace("_open", "").replace("_x1", "") for b in B}

print("\n########## 1. PER-BENCHMARK, HELD-OUT HALVES, JUDGE CURRENCY ##########")
print("(greedy / random-pick over 8 / majority vote / probe / oracle@8 ; sel_eff = probe/oracle,")
print(" gapnorm = (probe-random)/(oracle-random).  probe = LOCAL single-layer refit, 8-benchmark pooled.)")
hdr = "%-10s %-12s %6s %7s %7s %7s %7s %7s %7s %7s" % (
    "gen", "bench", "n", "greedy", "random", "major", "probe", "oracle8", "selEff", "gapNrm")
for gen in ("qwen", "medgemma"):
    print(hdr)
    C = R[gen]["cells"]
    for b in B:
        o = C[b]
        print("%-10s %-12s %6d %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f %7.3f" % (
            gen, SH[b], o["n_questions"], o["greedy_judge"], o["random_pick_judge"],
            o["majority_judge"], o["probe_judge"], o["oracle8_judge"],
            o["sel_eff_over_oracle_judge"], o["sel_eff_gap_normalised_judge"]))
    m = R[gen]["macro"]
    print("%-10s %-12s %6d %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f %7s" % (
        gen, "MACRO", m["n_questions"], m["greedy_judge"], m["random_pick_judge"],
        m["majority_judge"], m["probe_judge"], m["oracle8_judge"],
        m["probe_judge"] / m["oracle8_judge"], ""))
# lingshu from the em-rescore agent (frozen 24-head ensemble)
print(hdr)
for b in B:
    o = EM["per_benchmark"][b]
    d6 = D6["lingshu"]["cells"][b]
    print("%-10s %-12s %6d %7.4f %7s %7.4f %7.4f %7.4f %7.3f %7s" % (
        "lingshu*", SH[b], o["n"], o["greedy_judge"], "-", o["sc_judge"], o["verifier_judge"],
        o["oracle_judge"], o["sel_eff_judge"], "-"))
mm = EM["macro"]
print("%-10s %-12s %6d %7.4f %7s %7.4f %7.4f %7.4f %7.3f" % (
    "lingshu*", "MACRO", mm["n_questions"], mm["greedy_judge"], "-", mm["sc_judge"],
    mm["verifier_judge"], mm["oracle_judge"], mm["verifier_judge"] / mm["oracle_judge"]))
print("* lingshu row = em-rescore agent's frozen 24-head ENSEMBLE (majority column is its")
print("  self-consistency pick, not a strict majority vote); random-pick not computed there.")

print("\n########## 2. CURRENCY: probe - greedy, macro over 8, four currencies ##########")
print("%-10s %-14s %9s %-26s %-6s %-26s %-6s %5s %5s %5s" % (
    "gen", "currency", "macro", "img-clustered CI", "verd", "benchmark-level n=8 CI", "verd",
    "pos", "sig+", "sig-"))
for gen in ("qwen", "medgemma"):
    for cur in ("judge", "em", "strict_em", "token_f1"):
        k = f"probe_minus_greedy_{cur}"
        m = R[gen]["macro"][k]
        print("%-10s %-14s %+9.4f [%+.4f,%+.4f] %-6s [%+.4f,%+.4f] %-6s %5d %5d %5d" % (
            gen, cur, m["macro"], m["ci_image_clustered"][0], m["ci_image_clustered"][1],
            m["verdict_image_clustered"], m["ci_benchmark_level_n8"][0],
            m["ci_benchmark_level_n8"][1], m["verdict_benchmark_level"],
            m["n_positive"], m["n_significant_positive_image_clustered"],
            m["n_significant_negative_image_clustered"]))
for cur, k in (("judge", "delta_judge"), ("em", "delta_em"),
               ("strict_em", "delta_strict_em"), ("token_f1", "delta_token_f1")):
    m = EM["macro"][k]
    nsig = sum(1 for b in B if EM["per_benchmark"][b]["ci_" + k.replace("delta_", "")][0] > 0)
    nneg = sum(1 for b in B if EM["per_benchmark"][b]["ci_" + k.replace("delta_", "")][1] < 0)
    print("%-10s %-14s %+9.4f [%+.4f,%+.4f] %-6s [%+.4f,%+.4f] %-6s %5d %5d %5d" % (
        "lingshu*", cur, m["macro"], m["ci_image_clustered_within_benchmark"][0],
        m["ci_image_clustered_within_benchmark"][1], m["verdict_image_clustered"],
        m["ci_benchmark_level_n8"][0], m["ci_benchmark_level_n8"][1], m["verdict_benchmark_level"],
        m["n_positive_benchmarks"], nsig, nneg))

print("\n########## 2b. per-benchmark probe-greedy in each currency (mine) ##########")
for gen in ("qwen", "medgemma"):
    print("%-10s %-12s %9s %9s %9s %9s   %s" % (gen, "bench", "judge", "EM", "strictEM", "tokF1",
                                                "verdicts J/EM/st/F1"))
    C = R[gen]["cells"]
    for b in B:
        o = C[b]
        print("%-10s %-12s %+9.4f %+9.4f %+9.4f %+9.4f   %s/%s/%s/%s" % (
            gen, SH[b], o["probe_minus_greedy_judge"], o["probe_minus_greedy_em"],
            o["probe_minus_greedy_strict_em"], o["probe_minus_greedy_token_f1"],
            o["verdict_probe_minus_greedy_judge"], o["verdict_probe_minus_greedy_em"],
            o["verdict_probe_minus_greedy_strict_em"], o["verdict_probe_minus_greedy_token_f1"]))

print("\n########## 3. HEADROOM ##########")
for gen in ("qwen", "medgemma"):
    h = R[gen]["macro"]["headroom"]
    C = R[gen]["cells"]
    print(gen, json.dumps(h, indent=1))
    print("   greedy per cell:", {SH[b]: round(C[b]["greedy_judge"], 4) for b in B})
print("lingshu* greedy per cell:", {SH[b]: round(EM["per_benchmark"][b]["greedy_judge"], 4) for b in B})
dl = {b: EM["per_benchmark"][b]["delta_judge"] for b in B}
lo = [b for b in B if EM["per_benchmark"][b]["greedy_judge"] < 0.10]
hi = [b for b in B if EM["per_benchmark"][b]["greedy_judge"] >= 0.20]
print("lingshu* cells<0.10:", lo, "contribution", round(sum(dl[b] for b in lo) / 8, 4),
      "| cells>=0.20:", hi, "macro_restricted", round(float(np.mean([dl[b] for b in hi])), 4),
      "| macro_all8", round(float(np.mean(list(dl.values()))), 4))

print("\n########## 4. ANSWER-PRIOR CONTROL (judge currency) ##########")
print("%-10s %-12s %8s %8s %8s %9s %9s %s" % ("gen", "bench", "greedy", "prior", "probe",
                                              "prior-grd", "probe-prior", "verd(probe>prior)"))
for gen in ("qwen", "medgemma"):
    C = R[gen]["cells"]
    for b in B:
        o = C[b]
        print("%-10s %-12s %8.4f %8.4f %8.4f %+9.4f %+9.4f %s" % (
            gen, SH[b], o["greedy_judge"], o["answer_prior_judge"], o["probe_judge"],
            o["answerprior_minus_greedy_judge"], o["probe_minus_answerprior_judge"],
            o["verdict_probe_minus_answerprior_judge"]))
    m = R[gen]["macro"]
    print("%-10s %-12s %8.4f %8.4f %8.4f %+9.4f %+9.4f %s" % (
        gen, "MACRO", m["greedy_judge"], m["answer_prior_judge"], m["probe_judge"],
        m["answerprior_minus_greedy_judge"]["macro"], m["probe_minus_answerprior_judge"]["macro"],
        m["probe_minus_answerprior_judge"]["verdict_image_clustered"]))
    print("   share of the probe's judge gain that the answer prior alone recovers: %.1f%%" % (
        100 * m["answerprior_minus_greedy_judge"]["macro"] / m["probe_minus_greedy_judge"]["macro"]))

print("\n########## 5. PER-SEED SPREAD of probe-greedy judge, per benchmark ##########")
for gen in ("qwen", "medgemma"):
    C = R[gen]["cells"]
    for b in B:
        v = C[b]["per_seed_probe_minus_greedy_judge"]
        print("%-10s %-12s seeds %s  spread %.4f" % (
            gen, SH[b], " ".join("%+.4f" % x for x in v), max(v) - min(v)))
    mv = np.array([C[b]["per_seed_probe_minus_greedy_judge"] for b in B]).mean(0)
    print("%-10s %-12s macro per seed %s  spread %.4f" % (
        gen, "MACRO", " ".join("%+.4f" % x for x in mv), mv.max() - mv.min()))

print("\n########## 6. REFIT vs ARTIFACT (does my refit reproduce pooled_singlelayer?) ##########")
for gen, af in (("qwen", "head_final_stack_qwen_2026-09-13.json"),
                ("medgemma", "head_final_stack_medgemma_ALL8_2026-09-16.json")):
    a = json.load(open(os.path.join(ART, af)))
    C = R[gen]["cells"]
    print("%-10s %-12s %9s %9s %9s" % (gen, "bench", "artifact", "my refit", "diff"))
    ds = []
    for b in B:
        av = a["cells"][b]["pooled_singlelayer"]
        mv = C[b]["probe_judge"]
        ds.append(mv - av)
        print("%-10s %-12s %9.4f %9.4f %+9.4f" % (gen, SH[b], av, mv, mv - av))
    print("%-10s %-12s %9.4f %9.4f %+9.4f  (macro delta-vs-greedy: artifact %+0.4f, mine %+0.4f)" % (
        gen, "MACRO", np.mean([a["cells"][b]["pooled_singlelayer"] for b in B]),
        R[gen]["macro"]["probe_judge"], float(np.mean(ds)),
        a["macro"]["pooled_singlelayer"], R[gen]["macro"]["probe_minus_greedy_judge"]["macro"]))
