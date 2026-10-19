"""Round-2 worker 2: inter-judge agreement on the September training labels, from judge_2x2_2026-09-20.json.
Cohen's kappa is derived from the artifact's own observed agreement and the two positive rates (no new data).
Adds key 'judge_agreement' to numbers_review_part2.json; also stores the label-source EM cells with their labels."""
import json
ART = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part2.json"
a = json.load(open(ART + "judge_2x2_2026-09-20.json"))
c = json.load(open(ART + "judge_consensus_2026-09-20.json"))
po = a["train_label_agreement_AB"]
pa, pb = a["train_label_pos_rate"]["A"], a["train_label_pos_rate"]["B"]
pe = pa * pb + (1 - pa) * (1 - pb)
kappa = (po - pe) / (1 - pe)
o = json.load(open(OUT))
o["judge_agreement"] = {"value": {"rows": a["train_rows"], "observed_agreement": round(po, 4),
                                  "pos_rate_MedVLThinker": round(pa, 4), "pos_rate_MedGemma": round(pb, 4),
                                  "cohen_kappa": round(kappa, 3)},
                        "computed_by": __file__,
                        "inputs": [ART + "judge_2x2_2026-09-20.json train_label_agreement_AB, train_label_pos_rate, train_rows"],
                        "note": "September training rows (layer-20 2x2 study); candidate-level labels; kappa derived from the marginals"}
o["label_source_em"] = {"value": {"MedVLThinker_labels": round(a["macro"]["probe_A_graded_EM"], 4),
                                  "MedGemma_labels": round(a["macro"]["probe_B_graded_EM"], 4),
                                  "consensus_AND": round(c["macro"]["probe_AND_graded_EM"], 4),
                                  "consensus_SOFT": round(c["macro"]["probe_SOFT_graded_EM"], 4)},
                        "computed_by": __file__,
                        "inputs": [ART + "judge_2x2_2026-09-20.json macro.probe_{A,B}_graded_EM",
                                   ART + "judge_consensus_2026-09-20.json macro.probe_{AND,SOFT}_graded_EM"]}
json.dump(o, open(OUT, "w"), indent=1)
print(o["judge_agreement"]["value"], o["label_source_em"]["value"])
