#!/usr/bin/env python3
"""pathvqa_truncation_impact.py -- did reporting PathVQA on 1,500 of 3,357 questions bias it?

THE AUDIT FINDING (2026-09-12).  run_openvqa.py:158 does `items = items[:A.n]`, and the pathvqa_open
T=0.7 sc8 and greedy runs were launched with a cap: they held 1,500 questions while every other tag
on that benchmark held all 3,357.  Nothing downstream records the dataset's true size, so it was
invisible.  The cut is a clean idx prefix (0..2854 kept) and the omitted half is HARDER -- exact
match 0.2645 vs 0.3132, gold 2.65 vs 2.36 words, 3.37 vs 3.15 distinct candidates.

So every PathVQA figure in the project sits on an easier-than-average prefix, and PathVQA is
load-bearing.  Generation, judging and extraction have now been completed to 3,357.  This measures
the bias directly.

WHICH PROBE.  The frozen incumbent (genframe_head_ens8, layer 21).  It is trained on
pathvqa_open_TRAIN -- the dataset's official train split -- and never on pathvqa_open, so all 3,357
evaluation questions are clean for it.  The pooled probe cannot be used here: its training half was
drawn from the OLD 1,500, so the 1,857 restored questions were never assigned a side.

Reports verifier, greedy and oracle on the reported subset and on the full benchmark, so the
question "how wrong was the published number" has a number.

  python3 src/cascade_methods/pathvqa_truncation_impact.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/pathvqa_truncation_2026-09-13.json")


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()

    stem = f"{FEATS}/generator_eval_pathvqa_full"
    z = np.load(stem + ".npz"); m = json.load(open(stem + ".meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0]
    rows = [m["rows"][i] for i in keep]
    H = z["h_span"][keep, lay.index(21)].astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=int)
    L = sel.head_logits(H)
    gok = {}
    for l in open(f"{CK}/ckpt_pathvqa_open_lingshu7b.judge.jsonl"):
        if l.strip():
            d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
    byq = defaultdict(list)
    for i, r in enumerate(rows):
        byq[r["idx"]].append(i)
    img = {r["idx"]: r["img_md5"] for r in rows}

    # The reported subset CANNOT be read from the sc8 dump any more -- that dump has been
    # backfilled to 3,357, so max(idx) is now 6717 and every question would count as "reported"
    # (caught immediately: restored_only came out n=0 with nan accuracies).
    # The audit established the truncation was `items[:1500]`, i.e. a clean idx prefix: the kept
    # set is idx 0..2854 and the dropped set 2855..6717, with the kept set contiguous in sorted
    # order. Reconstruct it from that boundary and ASSERT the size, so a wrong boundary fails loudly
    # instead of silently reporting a bias of zero.
    CUT = 2854
    cut = CUT

    ver, gre, orc, inrep, cl = [], [], [], [], []
    for q in byq:
        if q not in gok:
            continue
        ii = np.array(byq[q])
        hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
        ver.append(int(y[ii][int(np.argmax(hr))])); gre.append(gok[q])
        orc.append(int(y[ii].max())); inrep.append(q <= cut); cl.append(img[q])
    ver, gre, orc, inrep, cl = map(np.array, (ver, gre, orc, inrep, cl))
    new = ~inrep
    assert 1400 <= inrep.sum() <= 1500, (
        f"reconstructed reported subset is {inrep.sum()} questions, expected ~1,500 -- the idx "
        f"prefix boundary {CUT} is wrong and the bias estimate would be meaningless")
    assert new.sum() > 0, "no restored questions found"

    def blk(mask):
        return {"n_questions": int(mask.sum()), "verifier": float(ver[mask].mean()),
                "greedy": float(gre[mask].mean()), "oracle_at_8": float(orc[mask].mean()),
                "verifier_minus_greedy": float(ver[mask].mean() - gre[mask].mean())}

    rng = np.random.default_rng(20260913)
    def boot(mask, nboot=10000):
        idx = np.where(mask)[0]
        groups = [np.where(cl[idx] == c)[0] for c in np.unique(cl[idx])]
        a, b = ver[idx], gre[idx]
        d = np.empty(nboot)
        for i in range(nboot):
            s = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            d[i] = a[s].mean() - b[s].mean()
        lo, hi = np.percentile(d, [2.5, 97.5])
        return [float(lo), float(hi)]

    art = {"title": "Did truncating PathVQA to 1,500 of 3,357 questions bias the reported number?",
           "date": "2026-09-13", "no_fabricated_numbers": True,
           "probe": "genframe_head_ens8 (frozen incumbent, layer 21) -- trained on "
                    "pathvqa_open_TRAIN, never on pathvqa_open, so all 3,357 are clean for it",
           "reported_subset": blk(inrep), "restored_only": blk(new), "full_benchmark": blk(~np.zeros(len(ver), bool))}
    art["reported_subset"]["ci_image_clustered"] = boot(inrep)
    art["full_benchmark"]["ci_image_clustered"] = boot(np.ones(len(ver), bool))
    art["bias"] = (art["full_benchmark"]["verifier_minus_greedy"]
                   - art["reported_subset"]["verifier_minus_greedy"])
    art["VERDICT"] = (
        f"the reported 1,500-question subset gives verifier-minus-greedy "
        f"{art['reported_subset']['verifier_minus_greedy']:+.4f}; the full 3,357-question benchmark "
        f"gives {art['full_benchmark']['verifier_minus_greedy']:+.4f} "
        f"({art['bias']:+.4f}). The 1,857 restored questions are harder -- greedy "
        f"{art['restored_only']['greedy']:.4f} against {art['reported_subset']['greedy']:.4f} -- "
        + ("but the DELTA is essentially unchanged, so the truncation biased the absolute "
           "accuracies without biasing the verifier's measured gain."
           if abs(art["bias"]) < 0.01 else
           "and the verifier's measured gain moves materially, so every PathVQA delta in the "
           "project must be restated on the full benchmark."))
    json.dump(art, open(OUT, "w"), indent=1)
    for k in ("reported_subset", "restored_only", "full_benchmark"):
        v = art[k]
        print(f"  {k:18} n{v['n_questions']:6}  verifier {v['verifier']:.4f}  greedy "
              f"{v['greedy']:.4f}  delta {v['verifier_minus_greedy']:+.4f}  oracle "
              f"{v['oracle_at_8']:.4f}")
    print(f"\n  bias from truncation: {art['bias']:+.5f}")
    print(f"\n=> {art['VERDICT']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
