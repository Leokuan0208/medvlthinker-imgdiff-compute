import json
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
o = json.load(open(MAIN + "/results/cascade_methods/artifacts/free_signal_bakeoff_2026-08-21.json"))
doc = {
    "PathVQA": (1500, 0.3427, 0.3333, 0.3260, 0.3900, 0.5167),
    "SLAKE": (645, 0.7302, 0.7287, 0.7395, 0.7690, 0.8791),
    "VQA-RAD": (200, 0.4900, 0.4350, 0.4650, 0.4650, 0.6300),
    "RadImageNet": (2000, 0.3210, 0.2825, 0.3245, 0.3295, 0.5120),
    "Kvasir-x1": (10121, 0.2849, 0.2815, 0.2699, 0.3629, 0.4696),
    "OmniMedVQA": (8883, 0.5164, 0.4746, 0.5162, 0.4971, 0.7007),
    "VQA-Med C4": (3663, 0.0947, 0.0459, 0.0863, 0.0688, 0.2102),
    "GEMeX": (8000, 0.3974, 0.3549, 0.3794, 0.4121, 0.5864),
}
keymap = {
    "PathVQA": "pathvqa_open", "SLAKE": "slake_open", "VQA-RAD": "vqa_rad_open",
    "RadImageNet": "radimagenet_open", "Kvasir-x1": "kvasir_x1_open",
    "OmniMedVQA": "omnimed_open", "VQA-Med C4": "vqamed_open", "GEMeX": "gemex_open",
}
for label, (n, g, p, sc, h, orc) in doc.items():
    c = o["cells"][keymap[label]]
    a = c["arms_judge"]
    cur = (
        c["n_questions"], round(a["always_7b_greedy"], 4), round(a["string_prior"], 4),
        round(a["self_consistency"], 4), round(a["frozen_head"], 4), round(a["oracle_at_8"], 4),
    )
    printed = (n, g, p, sc, h, orc)
    mism = [
        f"{col}:{pv}->{cv}"
        for col, pv, cv in zip(["n", "greedy", "prior", "sc", "head", "oracle"], printed, cur)
        if pv != cv
    ]
    print(label, "OK" if not mism else "MISMATCH " + " ".join(mism))
