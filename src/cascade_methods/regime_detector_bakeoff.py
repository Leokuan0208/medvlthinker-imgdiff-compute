#!/usr/bin/env python3
"""regime_detector_bakeoff.py -- CAN we tell, at test time, whether selection is going to help?

THE REGIME.  Measured 2026-08-21 (free_signal_bakeoff, ood_gate_diag):

    in-domain    the trained head beats greedy by +0.039 to +0.078 and beats self-consistency
    out-of-domain  NO selector beats greedy.  The head loses by -0.026 to -0.049; self-consistency,
                 which is training-free, loses far less (-0.004 on omnimed) but still loses.

So the value of the whole best-of-8 apparatus is decided by which regime a cell is in, and the
regimes are separated by a large, consistent effect.  If the regime were DETECTABLE from what is on
hand at inference time, the wall would become a routing decision: select in domain, take the greedy
answer out of domain -- cheaper AND more accurate.  Note this is not abstention; every branch
returns an answer, and the out-of-domain branch returns the model's own.

ALREADY REFUTED.  Mahalanobis distance of layer-21 h_span to the head's training distribution in a
64-d PCA basis does NOT order the cells: cell-level spearman +0.214 (the WRONG sign -- the two cells
where the head fails score 1.0247 and 0.9383, below in-domain slake's 1.1903).

DETECTORS TESTED HERE, each cheap and each available at inference:
  maha_pca64      the refuted baseline, carried so the comparison is on one table
  knn_train       mean distance to the 10 nearest training rows in the same PCA basis -- a
                  non-parametric density estimate, which does not assume one Gaussian blob
  domclf_maxprob  max softmax of a 4-way classifier over the head's training domains; low means
                  "resembles none of the domains I was fitted on"
  vocab_coverage  fraction of the pool's candidate strings that occur in the head's TRAINING answer
                  vocabulary.  Motivated by the 2026-08-19 audit: a counter over answer strings
                  reproduced 101% of the head's donor gain on RadImageNet, so the head's advantage
                  is partly vocabulary -- and vocabulary overlap is directly measurable.
  head_spread     sd of the head's within-pool scores; a head with nothing to say may say it flatly
  sc_entropy      entropy of the generator's own sampling distribution -- training-free

ENDPOINTS.  (1) cell-level Spearman against the measured head-minus-greedy, over 7 cells -- few
points, so it is an ordering check and not an estimate; (2) per-question AUROC for "the head's pick
was right where greedy was wrong", pooled, which has thousands of points and is what a deployed
router would actually need.

  python3 src/cascade_methods/regime_detector_bakeoff.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/regime_detector_2026-08-21.json")
LAYERS = [7, 14, 21, 28]
from free_signal_bakeoff import CELLS, RAW, norm, load_cell

TRAINED = {"pathvqa_open", "slake_open", "vqa_rad_open", "kvasir_x1_open"}


def auroc(score, lab):
    lab = np.asarray(lab).astype(int)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return float("nan")
    o = np.argsort(score)
    r = np.empty(len(score), float)
    r[o] = np.arange(1, len(score) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import torch, torch.nn as nn
    torch.set_num_threads(4)
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()

    # ---- training reference: features, domain labels, answer vocabulary --------------------
    TR, trds, trvocab = [], [], set()
    for sh in (0, 1):
        z = np.load(os.path.join(FEATS, f"generator_train_s{sh}of2.npz"))
        m = json.load(open(os.path.join(FEATS, f"generator_train_s{sh}of2.meta.json")))
        keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0]
        TR.append(z["h_span"][keep, LAYERS.index(21)].astype(np.float32))
        for i in keep:
            trds.append(m["rows"][i]["ds"]); trvocab.add(norm(m["rows"][i]["na"]))
    TR = np.concatenate(TR); trds = np.array(trds)
    mu = TR.mean(0); Xc = TR - mu
    rng = np.random.default_rng(0)
    g = rng.standard_normal((Xc.shape[1], 80)).astype(np.float32)
    Q, _ = np.linalg.qr(Xc @ g)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    P = Vt[:64].T.astype(np.float32)
    ZT = Xc @ P
    sd = ZT.std(0) + 1e-6
    ZTn = ZT / sd
    print(f"reference: {len(TR)} rows, {len(trvocab)} distinct training answers, "
          f"{len(set(trds))} domains", flush=True)

    # 4-way domain classifier on the PCA basis
    dnames = sorted(set(trds)); dcode = {d: i for i, d in enumerate(dnames)}
    dy = torch.tensor([dcode[d] for d in trds], dtype=torch.long)
    clf = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, len(dnames)))
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-2, foreach=False)
    Zt = torch.tensor(ZTn)
    for ep in range(15):
        perm = torch.randperm(len(Zt))
        for i in range(0, len(Zt), 512):
            j = perm[i:i + 512]
            opt.zero_grad()
            nn.functional.cross_entropy(clf(Zt[j]), dy[j]).backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        acc = (clf(Zt).argmax(1) == dy).float().mean().item()
    print(f"domain classifier train acc {acc:.3f}", flush=True)

    DET = ["maha_pca64", "knn_train", "domclf_maxprob", "vocab_coverage", "head_spread", "sc_entropy"]
    art = {"title": "Can the selection regime be detected at inference time?",
           "date": "2026-08-21", "no_fabricated_numbers": True,
           "detectors": DET, "cells": {},
           "note": "cell-level spearman over 7 cells is an ORDERING CHECK, not an estimate"}
    pooled = {d: [] for d in DET}; pooled_lab = []
    DUMP = {}

    for cell in CELLS:
        got = load_cell(cell)
        if got is None:
            continue
        H, rows = got
        stem = RAW.get(cell, cell)
        rawp = os.path.join(CK, f"ckpt_{stem}_lingshu7b_sc8.jsonl")
        jp = os.path.join(CK, f"ckpt_{stem}_lingshu7b.judge.jsonl")
        if not (os.path.exists(rawp) and os.path.exists(jp)):
            continue
        y = np.array([r["y"] for r in rows], dtype=int)
        na = np.array([norm(r["na"]) for r in rows])
        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        qids = sorted(byq, key=lambda k: (len(str(k)), str(k)))
        raw = {}
        for l in open(rawp):
            if l.strip():
                d = json.loads(l); raw[d["idx"]] = d
        gj = {}
        for l in open(jp):
            if l.strip():
                d = json.loads(l); gj[d["idx"]] = d

        Zc = ((H - mu) @ P) / sd
        maha = np.sqrt((Zc ** 2).mean(1))
        # kNN distance in chunks
        knn = np.empty(len(Zc), np.float32)
        tn = (ZTn ** 2).sum(1)
        for b0 in range(0, len(Zc), 2048):
            b1 = min(b0 + 2048, len(Zc))
            B = Zc[b0:b1]
            d2 = (B ** 2).sum(1)[:, None] + tn[None, :] - 2 * B @ ZTn.T
            knn[b0:b1] = np.sqrt(np.maximum(np.partition(d2, 10, axis=1)[:, :10], 0)).mean(1)
        with torch.no_grad():
            mp = torch.softmax(clf(torch.tensor(Zc.astype(np.float32))), 1).max(1).values.numpy()
        invocab = np.array([1.0 if a in trvocab else 0.0 for a in na])
        L = sel.head_logits(H)

        rec = {d: [] for d in DET}; hq, gq = [], []
        for q in qids:
            if q not in gj:
                continue
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
            hq.append(int(y[ii][int(np.argmax(hr))])); gq.append(int(gj[q]["judge_ok"]))
            d = raw.get(q)
            if d:
                c = np.array(list(Counter(norm(p) for p in d["preds"]).values()), float)
                p_ = c / c.sum()
                ent = float(-(p_ * np.log(p_ + 1e-12)).sum())
            else:
                ent = 0.0
            rec["maha_pca64"].append(float(maha[ii].mean()))
            rec["knn_train"].append(float(knn[ii].mean()))
            rec["domclf_maxprob"].append(float(mp[ii].mean()))
            rec["vocab_coverage"].append(float(invocab[ii].mean()))
            rec["head_spread"].append(float(np.std(hr)))
            rec["sc_entropy"].append(ent)
        hq, gq = np.array(hq), np.array(gq)
        DUMP[cell] = {"head_ok": hq, "greedy_ok": gq,
                      **{d: np.array(rec[d]) for d in DET}}
        art["cells"][cell] = {"domain": "IN" if cell in TRAINED else "OUT",
                              "n_questions": int(len(hq)),
                              "head_minus_greedy": float(hq.mean() - gq.mean()),
                              "detector_means": {d: float(np.mean(v)) for d, v in rec.items()}}
        # pooled per-question target: head right where greedy wrong (1) vs greedy right where head wrong (0)
        disc = (hq != gq)
        for d in DET:
            pooled[d] += list(np.array(rec[d])[disc])
        pooled_lab += list(hq[disc])
        print(f"  {cell:17} {art['cells'][cell]['domain']:4} "
              f"head-greedy {art['cells'][cell]['head_minus_greedy']:+.4f}  " +
              "  ".join(f"{d.split('_')[0]} {np.mean(rec[d]):.3f}" for d in DET), flush=True)
        json.dump(art, open(OUT, "w"), indent=1)

    hm = np.array([v["head_minus_greedy"] for v in art["cells"].values()])
    art["cell_level_spearman"] = {}
    art["pooled_per_question_auroc"] = {}
    for d in DET:
        x = np.array([v["detector_means"][d] for v in art["cells"].values()])
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(hm))
        art["cell_level_spearman"][d] = float(np.corrcoef(rx, ry)[0, 1])
        art["pooled_per_question_auroc"][d] = auroc(np.array(pooled[d]), pooled_lab)
    best = max(DET, key=lambda d: abs(art["cell_level_spearman"][d]))
    art["VERDICT"] = (f"{best} orders the cells (spearman "
                      f"{art['cell_level_spearman'][best]:+.3f}) -- worth building a router on"
                      if abs(art["cell_level_spearman"][best]) >= 0.75 else
                      "NO detector orders the cells; the regime is not identifiable at inference "
                      "from anything tested, so selection cannot be routed on and must be presented "
                      "as an in-domain-only method")
    print("\n  detector          cell-spearman   pooled per-question AUROC")
    for d in DET:
        print(f"  {d:17} {art['cell_level_spearman'][d]:+.3f}          "
              f"{art['pooled_per_question_auroc'][d]:.4f}")
    print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    npz = OUT.replace(".json", "_perquestion.npz")
    np.savez_compressed(npz, **{f"{c}|{k}": v for c, d in DUMP.items() for k, v in d.items()})
    print(f"wrote {OUT}\n      {npz}")


if __name__ == "__main__":
    main()
