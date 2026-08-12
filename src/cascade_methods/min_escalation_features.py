#!/usr/bin/env python3
"""
min_escalation_features.py -- STAGE A of ATTACK 4.

Extracts the PER-ITEM GATE FEATURES that cost_floor.py explicitly could not reach:

    "The policy family is limited to the SHIPPED operating points, because vec_disjoint.npz stores
     only per-item CORRECTNESS, not the per-item gate features.  Re-tuning an escalation threshold
     inside an arm is therefore out of reach of this attack."
     -- cost_floor_2026-08-10.json:limitations[0]

For every one of the 8 Variant-B reporting cells it writes, aligned item-by-item:
   MCQ  (5 cells): ok7, ok32, okT, margin7 (top1-top2 prob), conf7 (top1 prob), agree(7B,32B)
   OPEN (3 cells): greedy_ok (7B greedy floor), ok32 (32B-direct judged), scores[8] (verifier
                   P(correct) per candidate, CLEAN DISJOINT verifier), sl[8] (per-candidate judge_ok)

Sources are exactly the ones the published macro path reads:
   MCQ   MedEvalKit/eval_results_lingshu{7b,32b}_{full,think,reason}/*/<ds>/results.json  via integrated_method
   OPEN  ckpts/train/lora_verifier_disjoint/transfer_dump_<ds>_lingshu7b.json  + the 32B judge jsonl

CPU only, no GPU, no new inference.  Launch from the repo root:
    python3 src/cascade_methods/min_escalation_features.py
Writes results/cascade_methods/artifacts/_min_escalation_parts/features.npz
"""
import os, sys, json
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import integrated_method as IM        # noqa: E402
import integrated_pandora as IP       # noqa: E402

ROOT = IM.ROOT
PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_min_escalation_parts")
os.makedirs(PARTS, exist_ok=True)
OUT = os.path.join(PARTS, "features.npz")

# the CLEAN, decontaminated verifier -- the same source cascade_selector_rerun.py calls "disjoint"
VERIFIER_DIR = "ckpts/train/lora_verifier_disjoint"
IM.OPEN_VERIFIER_DIR = VERIFIER_DIR
IP.ADAPTER = VERIFIER_DIR

MCQ_SPECS = [("PMC_VQA", "PMC_VQA", None), ("SLAKE_closed", "SLAKE", "SLAKE"),
             ("VQA_RAD_closed", "VQA_RAD", "YESNO"), ("PATH_VQA_closed", "PATH_VQA", "YESNO")]
OPEN_KEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}


def main():
    out = {}
    meta = {}

    for name, ds, closed in MCQ_SPECS:
        d = IM.mcq_closed(ds, closed)
        assert d is not None, name
        okT = d["okT"] if d["okT"] is not None else d["ok32"]
        out[f"{name}|ok7"] = d["ok7"].astype(np.int8)
        out[f"{name}|ok32"] = d["ok32"].astype(np.int8)
        out[f"{name}|okT"] = okT.astype(np.int8)
        out[f"{name}|margin"] = d["margin"].astype(np.float64)
        out[f"{name}|conf"] = d["conf"].astype(np.float64)
        out[f"{name}|agree"] = d["agree"].astype(np.int8)
        meta[name] = dict(format="MCQ", n=int(len(d["ok7"])),
                          think_dump=bool(d["okT"] is not None),
                          n_distinct_margin=int(len(set(d["margin"].tolist()))))

    d = IM.mcq_medxpert()
    out["MedXpertQA-MM|ok7"] = d["ok7"].astype(np.int8)
    out["MedXpertQA-MM|ok32"] = d["ok32"].astype(np.int8)
    out["MedXpertQA-MM|okT"] = (d["okT"] if d["okT"] is not None else d["ok32"]).astype(np.int8)
    out["MedXpertQA-MM|margin"] = d["margin"].astype(np.float64)
    out["MedXpertQA-MM|conf"] = d["conf"].astype(np.float64)
    out["MedXpertQA-MM|agree"] = np.zeros(len(d["ok7"]), np.int8)   # not produced by this loader
    meta["MedXpertQA-MM"] = dict(format="MCQ", n=int(len(d["ok7"])), think_dump=bool(d["okT"] is not None),
                                 n_distinct_margin=int(len(set(d["margin"].tolist()))),
                                 agree="NOT AVAILABLE from mcq_medxpert -- zeros, never used")

    for name, dskey in OPEN_KEY.items():
        rows = IP.load_open_rows(dskey)
        assert rows is not None, name
        n = len(rows)
        K = min(len(r["scores"]) for r in rows)
        assert K >= 8, (name, K)
        sc = np.array([np.asarray(r["scores"][:8], float) for r in rows])
        sl = np.array([[0 if x in (None, -1) else int(x) for x in r["sl"][:8]] for r in rows], np.int8)
        out[f"{name}|scores"] = sc
        out[f"{name}|sl"] = sl
        out[f"{name}|greedy"] = np.array([r["greedy"] for r in rows], np.int8)
        out[f"{name}|ok32"] = np.array([r["strong"] for r in rows], np.int8)
        meta[name] = dict(format="open", n=int(n), K=8, verifier=VERIFIER_DIR)

    np.savez_compressed(OUT, **out)
    json.dump(meta, open(os.path.join(PARTS, "features_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
