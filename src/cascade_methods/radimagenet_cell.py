#!/usr/bin/env python3
"""radimagenet_cell.py -- turn radimagenet_open into a reportable 4th open-text cell.

WHY IT IS A CLEAN TEST.  radimagenet_open contributes ZERO rows to the head's training pool
(feats_hidden/generator_train_*: pathvqa 22,390 / kvasir 5,562 / slake 2,207 / vqa_rad 1,339 -- no
radimagenet) and ZERO examples to lora_verifier_disjoint.  It was reserved as a training source in
verifier_disjoint_split.json and never spent.  So applying the FROZEN 8-seed head to it is a
genuine out-of-domain transfer test, not a re-read of something it was fitted on.

ARMS (all already on disk except the greedy, which run_radimagenet_greedy.sh produced):
  always-7B greedy        ckpts/openvqa/cheap_lingshu7b/ckpt_radimagenet_open_lingshu7b.judge.jsonl
  7B best-of-8 pool       .../ckpt_radimagenet_open_lingshu7b_sc8_scexploded.judge.jsonl
  Lingshu-32B direct      ckpts/openvqa/strong_lingshu/ckpt_radimagenet_open_lingshu32b_t0.judge.jsonl
  head features           feats_hidden/generator_eval_radimagenet.npz

  python3 src/cascade_methods/radimagenet_cell.py \
      --out results/cascade_methods/artifacts/radimagenet_cell_2026-08-18.json
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
STRONG = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu")
LAYERS = [7, 14, 21, 28]


def loadj(p):
    return {json.loads(l)["idx"]: json.loads(l) for l in open(p) if l.strip()}


def boot_delta(a, b, nboot=10000, seed=20260818, clusters=None):
    """Paired bootstrap of mean(a) - mean(b), resampling IMAGES when clusters are given.

    BUGFIX 2026-08-19: resampling questions i.i.d. ignores that this cell is 2,000 questions over
    1,000 images (2 per image), so every interval was too narrow.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(nboot)
    groups = ([np.where(np.asarray(clusters) == c)[0] for c in np.unique(clusters)]
              if clusters is not None else None)
    for i in range(nboot):
        if groups is None:
            s = rng.integers(0, n, n)
        else:
            s = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        d[i] = a[s].mean() - b[s].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(a.mean() - b.mean()), "ci": [float(lo), float(hi)], "n": n,
            "verdict": "WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.join(FEATS, "generator_eval_radimagenet"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/cascade_methods/artifacts/radimagenet_cell_2026-08-18.json"))
    ap.add_argument("--nboot", type=int, default=10000)
    A = ap.parse_args()

    z = np.load(A.feats + ".npz")
    meta = json.load(open(A.feats + ".meta.json"))
    rows = meta["rows"]
    assert list(meta["layers"]) == LAYERS, meta["layers"]
    li = LAYERS.index(21)
    H = z["h_span"][:, li].astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=int)

    byq = defaultdict(list)
    for i, r in enumerate(rows):
        byq[r["idx"]].append(i)
    qids = sorted(byq, key=lambda k: (len(str(k)), str(k)))
    q_img = {r["idx"]: r["img_md5"] for r in rows}
    clusters = np.array([q_img[q] for q in qids])

    # ---- the frozen head, applied out of domain --------------------------------------------
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()
    L = sel.head_logits(H)                                   # (8 seeds, n_rows)

    # BUGFIX 2026-08-19: this used L.mean(0), a SEED-MEAN LOGIT.  The deployed convention
    # (genframe_selector.FrozenSelector.head_rank) is per-seed WITHIN-POOL rank_avg, then the mean
    # over seeds.  Mean-of-logits is not scale-invariant across seeds -- one seed with larger logit
    # magnitude dominates the ensemble -- so the previous number was not the deployed readout, and
    # the artifact's caveat list did not say so.
    picked_ok, oracle_ok, picked_ok_OLD = [], [], []
    for q in qids:
        ii = np.array(byq[q])
        score = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
        picked_ok.append(int(y[ii][int(np.argmax(score))]))
        picked_ok_OLD.append(int(y[ii][int(np.argmax(L.mean(0)[ii]))]))
        oracle_ok.append(int(y[ii].max()))
    picked_ok_OLD = np.array(picked_ok_OLD)
    picked_ok = np.array(picked_ok); oracle_ok = np.array(oracle_ok)

    # ---- the fixed baselines ----------------------------------------------------------------
    gj = loadj(os.path.join(CK, "ckpt_radimagenet_open_lingshu7b.judge.jsonl"))
    s32 = loadj(os.path.join(STRONG, "ckpt_radimagenet_open_lingshu32b_t0.judge.jsonl"))

    def align(d):
        out = []
        for q in qids:
            k = q if q in d else (int(q) if str(q).lstrip("-").isdigit() and int(q) in d else None)
            out.append(int(d[k]["judge_ok"]) if k is not None else -1)
        return np.array(out)

    greedy = align(gj)
    strong = align(s32)
    keep = (greedy >= 0) & (strong >= 0)
    n_drop = int((~keep).sum())

    art = {
        "title": "radimagenet_open as a 4th open-text cell -- OUT-OF-DOMAIN transfer of the frozen head",
        "date": "2026-08-18",
        "no_fabricated_numbers": True,
        "cleanliness": {
            "rows_in_head_train_pool": 0,
            "examples_in_lora_verifier_disjoint": 0,
            "note": "reserved as a train source in verifier_disjoint_split.json and never spent; "
                    "the frozen head has never seen a radimagenet image or question.",
        },
        "n_questions": len(qids), "n_candidate_rows": len(rows),
        "n_dropped_missing_baseline": n_drop,
        "mean_distinct_candidates": round(len(rows) / max(1, len(qids)), 4),
        "arms_judge": {
            "always_7b_greedy": float(greedy[keep].mean()),
            "lingshu32b_direct": float(strong[keep].mean()),
            "head_bestof8_selected": float(picked_ok[keep].mean()),
            "oracle_at_8": float(oracle_ok[keep].mean()),
            "sel_eff": float(picked_ok[keep][oracle_ok[keep] == 1].mean()),
        },
        "deltas": {
            "head_vs_always_7b": boot_delta(picked_ok[keep], greedy[keep], A.nboot, clusters=clusters[keep]),
            "head_vs_lingshu32b_direct": boot_delta(picked_ok[keep], strong[keep], A.nboot, clusters=clusters[keep]),
            "always_7b_vs_lingshu32b_direct": boot_delta(greedy[keep], strong[keep], A.nboot, clusters=clusters[keep]),
        },
        "provenance": {
            "features": A.feats + ".npz",
            "head": "ckpts/train/genframe_head_ens8 (frozen, 8 seeds, layer 21 h_span)",
            "greedy": "ckpts/openvqa/cheap_lingshu7b/ckpt_radimagenet_open_lingshu7b.judge.jsonl",
            "pool": "ckpts/openvqa/cheap_lingshu7b/ckpt_radimagenet_open_lingshu7b_sc8_scexploded.judge.jsonl",
            "strong": "ckpts/openvqa/strong_lingshu/ckpt_radimagenet_open_lingshu32b_t0.judge.jsonl",
        },
        "readout_correction_2026_08_19": {
            "what": "the deployed readout is per-seed within-pool rank_avg then mean over seeds; "
                    "the first version of this script used a seed-mean LOGIT, which is not "
                    "scale-invariant across seeds. Both are reported so the size of the error is "
                    "visible rather than silently replaced.",
            "sel_eff_deployed_readout": None,
            "sel_eff_seed_mean_logit_WRONG": None,
        },
        "caveats": [
            "JUDGE ONLY. Exact match is not reported: radimagenet golds are terse anatomy/pathology "
            "labels and EM would be dominated by surface form. The judge was audited -- of the 1,422 "
            "answers it marks wrong for the 32B, only 3 contain the gold string -- so the low 32B "
            "score is not a formatting artifact.",
            "Picks are over DISTINCT normalised answers, not the 8 raw slots. Duplicates share a "
            "feature vector and a label, so the picked ANSWER is the same either way; only the "
            "tie-break differs from the 8-slot convention used on the frozen three cells.",
            "The head is applied with its FROZEN train standardiser (mu/sd fitted on the four "
            "in-domain sets). No refit, no recalibration on radimagenet.",
        ],
    }
    okw = picked_ok_OLD[keep]
    art["readout_correction_2026_08_19"]["sel_eff_deployed_readout"] = art["arms_judge"]["sel_eff"]
    art["readout_correction_2026_08_19"]["sel_eff_seed_mean_logit_WRONG"] = float(
        okw[oracle_ok[keep] == 1].mean())
    art["readout_correction_2026_08_19"]["acc_seed_mean_logit_WRONG"] = float(okw.mean())
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(art, open(A.out, "w"), indent=1)
    a = art["arms_judge"]
    print(json.dumps(art["deltas"], indent=1))
    print(f"\nn={len(qids)} questions, {len(rows)} candidates ({art['mean_distinct_candidates']}/q)")
    print(f"  always-7B greedy    {a['always_7b_greedy']:.4f}")
    print(f"  Lingshu-32B direct  {a['lingshu32b_direct']:.4f}")
    print(f"  7B + frozen head    {a['head_bestof8_selected']:.4f}   (sel_eff {a['sel_eff']:.4f})")
    print(f"  oracle@8            {a['oracle_at_8']:.4f}")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
