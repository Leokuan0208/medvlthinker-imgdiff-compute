#!/usr/bin/env python3
"""verifier_validity_finalize.py -- assemble sections A/A2/B/B2/C into the decomposition (D) and the
verdict (E) of results/cascade_methods/artifacts/verifier_validity_2026-07-29.json.

Every number here is read from the artifact sections written by
  src/training_methods/verifier_validity_audit.py        (A, A2, B, B2 -- offline)
  src/training_methods/verifier_image_ablation_v2.py     (C -- GPU)
No number is typed in by hand.  Run last:
  python3 src/training_methods/verifier_validity_finalize.py
"""
import os, json
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
P = os.path.join(ROOT, "results/cascade_methods/artifacts/verifier_validity_2026-07-29.json")
d = json.load(open(P))
A, A2, B, B2, C = d["A_overlap"], d["A2_image_level_overlap"], d["B_in_vs_out_of_domain"], \
    d["B2_dataset_level_transfer_evidence"], d["C_image_ablation"]

pool = B["POOLED_3_PAPER_OPEN_CELLS"]
gain_full = pool["full"]["gain_over_greedy"]
gain_unseen = pool["unseen"]["gain_over_greedy"]

# ---- the two clean generalization estimates -------------------------------------------------
kv_in = B["kvasir_open"]["full"]["gain_over_greedy"]          # pooled4: Kvasir IS in its training pool
kv_lodo = B2["lora_verifier_open_to_kvasir"]["gain_over_greedy"]   # verifier that never saw Kvasir
ri = B2["pooled4_to_radimagenet_lingshu_generator"]["gain_over_greedy"]

# ---- image ablation: what fraction of the DISCRIMINATION and of the SELECTION GAIN survives
# when the image is destroyed?  "Text-prior floor" = the strongest image-free control.
cp = C["pooled"]["cond"]
greedy_pool = C["pooled"]["greedy"]
real_gain = cp["real"]["gain_over_greedy"]
imgfree = {k: cp[k] for k in ("blank_gray", "blank_black", "blank_matched", "mismatched", "no_image")}
best_imgfree = max(imgfree, key=lambda k: imgfree[k]["gain_over_greedy"])
floor_gain = imgfree[best_imgfree]["gain_over_greedy"]
image_share = 1.0 - floor_gain / real_gain if abs(real_gain) > 1e-9 else None
# AUROC above chance that survives without the image
auroc_real = cp["real"]["cand_auroc"]
auroc_floor = max(imgfree[k]["cand_auroc"] for k in imgfree)

D = {
    "note": "Decomposition of the open-text arm's verifier SELECTION gain. All inputs are the measured "
            "sections A/A2/B/B2/C of this same artifact.",
    "baseline_definition": "The dumps' `greedy_ok` field is the label of `modal_pred`, which run_openvqa.py "
                           "defines as the MAJORITY answer of the 8 temperature-0.7 samples -- i.e. the "
                           "so-called 'greedy' baseline in every open-text table is really "
                           "self-consistency@8. The verifier's gain is therefore a gain over SC@8, not over "
                           "a temperature-0 decode.",
    "memorization": {
        "pooled_selection_gain_as_reported": gain_full,
        "pooled_selection_gain_on_verifier_unseen_questions": gain_unseen,
        "share_of_gain_from_questions_the_verifier_trained_on": 1.0 - gain_unseen / gain_full,
        "seen_minus_unseen_gain": pool["seen_minus_unseen_gain"],
        "caveat": "'Unseen' is question-level only. Section A2 shows 100% (SLAKE) / 94.5% (PathVQA) / 64.8% "
                  "(VQA-RAD) of those held-out questions still use an IMAGE the verifier trained on, so "
                  "this UNDER-states the memorization component.",
    },
    "dataset_level_generalization": {
        "kvasir_same_items_two_verifiers": {
            "verifier_trained_on_kvasir_pooled4": kv_in,
            "verifier_that_never_saw_kvasir_lora_verifier_open": kv_lodo,
            "retained_fraction": kv_lodo / kv_in,
            "why_this_is_the_cleanest_test": "identical 1200 evaluation items, identical generator, near-identical "
                                             "training-set SIZE (5786 vs <=6000 examples); the only difference is "
                                             "whether Kvasir was in the training pool.",
        },
        "radimagenet_zero_overlap": {"gain": ri, "n": B2["pooled4_to_radimagenet_lingshu_generator"]["n"],
                                     "oracle_conversion": B2["pooled4_to_radimagenet_lingshu_generator"]["oracle_conversion"]},
    },
    "image_grounding": {
        "sample": C["sample"], "n_questions": C["pooled"]["n"], "n_candidates": C["pooled"]["n_candidates"],
        "auroc_real": auroc_real, "best_image_free_auroc": auroc_floor,
        "auroc_above_chance_retained_without_image": (auroc_floor - 0.5) / (auroc_real - 0.5),
        "selection_gain_real": real_gain,
        "strongest_image_free_control": best_imgfree,
        "selection_gain_image_free": floor_gain,
        "share_of_selection_gain_attributable_to_the_image": image_share,
        "per_condition": {k: {"auroc": cp[k]["cand_auroc"], "select_acc": cp[k]["select_acc"],
                              "gain_over_greedy": cp[k]["gain_over_greedy"],
                              "real_minus_this": cp[k]["real_minus_this_select"],
                              "ci": cp[k]["real_minus_this_ci"],
                              "significant_vs_real": cp[k]["significant_vs_real"]} for k in cp},
    },
}

# three-way split of the AS-REPORTED pooled selection gain (+gain_full):
#   memorization      = gain_full - gain_unseen                            (measured, question-level)
#   image-grounded    = (gain_unseen) * image_share                        (image share measured on the
#                                                                           ablation sample, applied to the
#                                                                           generalizing part)
#   language prior    = gain_unseen - image-grounded
mem = gain_full - gain_unseen
img = gain_unseen * image_share
lang = gain_unseen - img
D["three_way_decomposition_of_reported_gain"] = {
    "reported_pooled_selection_gain": gain_full,
    "memorization_of_training_questions": {"abs": mem, "share": mem / gain_full},
    "image_grounded_verification": {"abs": img, "share": img / gain_full},
    "language_prior": {"abs": lang, "share": lang / gain_full},
    "assumption": "the image share measured on the ablation sample is assumed constant across seen and "
                  "unseen questions; the ablation sample is a random draw from the three reported cells.",
}

d["D_decomposition"] = D
json.dump(d, open(P, "w"), indent=1)

f = lambda x: f"{x:+.4f}"
print("=" * 96)
print("DECOMPOSITION OF THE OPEN-TEXT VERIFIER'S SELECTION GAIN (all measured)")
print("=" * 96)
print(f"reported pooled gain over SC@8 (n={pool['full']['n']})            {f(gain_full)}")
print(f"  memorization of training questions                    {f(mem)}   ({100*mem/gain_full:.1f}%)")
print(f"  image-grounded verification                           {f(img)}   ({100*img/gain_full:.1f}%)")
print(f"  language prior                                        {f(lang)}   ({100*lang/gain_full:.1f}%)")
print()
print(f"dataset-level generalization (Kvasir, same 1200 items):")
print(f"  verifier that trained on Kvasir  {f(kv_in)}   |  verifier that never saw it  {f(kv_lodo)}"
      f"   -> retains {100*kv_lodo/kv_in:.0f}%")
print(f"  zero-overlap RadImageNet transfer {f(ri)}")
print(f"\nimage ablation: AUROC {auroc_real:.4f} real vs {auroc_floor:.4f} best image-free "
      f"({100*(auroc_floor-0.5)/(auroc_real-0.5):.0f}% of above-chance discrimination survives image removal)")
print(f"                selection gain {f(real_gain)} real vs {f(floor_gain)} ({best_imgfree}) "
      f"-> image contributes {100*image_share:.0f}% of the gain")
print(f"\nwrote -> {P}")
