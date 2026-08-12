#!/usr/bin/env python3
"""vision_verifier_fit.py -- fit ONE arm x ONE seed of the vision-aware verifier round and write a
resumable part file.  Nothing is aggregated here; vision_verifier_report.py assembles the artifact.

Every arm uses the DEPLOYED language-side recipe unchanged (layer 21, span pooling, train-mu/sd
standardisation, Linear(d,256)->GELU->Linear(256,1), Bradley-Terry over within-question pairs,
AdamW lr 1e-3 wd 1e-2, 30 epochs, group batch 64, row order 'concat').  The ONLY variable is what
goes into the feature vector -- that is what makes it a clean ablation of "inject the vision
signal" rather than a new pipeline.

  OMP_NUM_THREADS=8 python3 -u src/training_methods/vision_verifier_fit.py \
      --arm L --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G          # noqa: E402
import visverif_lib as V           # noqa: E402
import fit_hidden_head as FH       # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts")
CONCAT_ARMS = ["L", "Vmean", "L_Vmean", "L_prod", "L_simgrid", "L_maxsim", "L_prod_sim"]


# ---------------------------------------------------------------- cross-attention head (attack 1b)
class XAttn(nn.Module):
    """score(candidate, image) with EXPLICIT spatial attention over the patch grid.

    q comes from the candidate's language-side vector; K,V from the pooled patch grid, so the
    score depends on WHERE in the image the evidence is, and the attention map is inspectable.
    """

    def __init__(self, d=3584, dk=256, dh=256):
        super().__init__()
        self.q = nn.Linear(d, dk)
        self.k = nn.Linear(d, dk)
        self.v = nn.Linear(d, dk)
        self.p = nn.Linear(d, dh)
        self.out = nn.Sequential(nn.Linear(dh + dk, dh), nn.GELU(), nn.Linear(dh, 1))
        self.dk = dk

    def kv(self, Vg):
        """Project a patch bank once. Vg (N,P2,d) -> k,v each (N,P2,dk)."""
        return self.k(Vg), self.v(Vg)

    def attn(self, h, Vg=None, k=None):
        if k is None:
            k = self.k(Vg)
        q = self.q(h).unsqueeze(1)                                  # (B,1,dk)
        return torch.softmax((q * k).sum(-1) / math.sqrt(self.dk), -1)   # (B,P2)

    def forward(self, h, Vg=None, k=None, v=None):
        """k/v may be supplied pre-projected. All candidates of a question share ONE image, so
        projecting the bank per ROW re-does identical work up to 8x; callers pass the deduplicated
        projection instead. The arithmetic is unchanged -- k and v are a function of the image
        alone, so projecting once and indexing is the same tensor the per-row path would build."""
        if k is None or v is None:
            k, v = self.kv(Vg)
        a = self.attn(h, k=k)
        c = (a.unsqueeze(-1) * v).sum(1)                            # (B,dk)
        return self.out(torch.cat([self.p(h), c], -1)).squeeze(-1)


def _groups(gtr, ytr):
    from collections import defaultdict
    byq = defaultdict(list)
    for i, g in enumerate(gtr):
        byq[g].append(i)
    return [np.array(v) for v in byq.values() if ytr[v].sum() > 0 and (1 - ytr[v]).sum() > 0]


def fit_xattn(Htr, Gtr_idx, VgB, ytr, gtr, seed, epochs=30, lr=1e-3, wd=1e-2, gb=64):
    """Htr (n,3584) standardised candidate vectors; Gtr_idx (n,) image row; VgB (n_img,P2,3584)
    standardised patch bank. BT loss over (correct, incorrect) pairs inside a question."""
    torch.manual_seed(seed)
    m = XAttn(d=Htr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    H = torch.tensor(Htr); B = torch.tensor(VgB); GI = torch.tensor(Gtr_idx)
    y = torch.tensor(ytr)
    groups = _groups(gtr, ytr)
    if not groups:
        m.eval(); return m
    L = max(len(g) for g in groups)
    idx = np.zeros((len(groups), L), np.int64); msk = np.zeros((len(groups), L), np.float32)
    for k, g in enumerate(groups):
        idx[k, :len(g)] = g; msk[k, :len(g)] = 1.0
    idx = torch.tensor(idx); msk = torch.tensor(msk)
    NG = idx.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(NG)
        for i in range(0, NG, gb):
            j = perm[i:i + gb]
            gi, gm = idx[j], msk[j]
            flat = gi.reshape(-1)
            img = GI[flat]
            uniq, inv = torch.unique(img, return_inverse=True)      # dedupe: 1 image per question
            ku, vu = m.kv(B[uniq])
            s = m(H[flat], k=ku[inv], v=vu[inv]).reshape(gi.shape)
            yy = y[flat].reshape(gi.shape) * gm
            s = s.masked_fill(gm == 0, -1e9)
            pm = yy.unsqueeze(2); nm = ((1 - yy) * gm).unsqueeze(1)
            d = s.unsqueeze(2) - s.unsqueeze(1)
            w = pm * nm
            l = ((nn.functional.softplus(-d) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)).mean()
            opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def predict_xattn(m, Hev, Gev_idx, VgB, bs=512, want_attn=False):
    H = torch.tensor(Hev); B = torch.tensor(VgB); GI = torch.tensor(Gev_idx)
    outs, atts = [], []
    with torch.no_grad():
        for i in range(0, len(Hev), bs):
            h = H[i:i + bs]
            uniq, inv = torch.unique(GI[i:i + bs], return_inverse=True)
            ku, vu = m.kv(B[uniq])
            k, v = ku[inv], vu[inv]
            outs.append(m(h, k=k, v=v).numpy())
            if want_attn:
                atts.append(m.attn(h, k=k).numpy().astype(np.float32))
    return np.concatenate(outs), (np.concatenate(atts) if want_attn else None)


# ---------------------------------------------------------------- data
def load_all(layer, grid_layer, ablate_eval="none", lang_eval_featdir=None):
    tr = G.load_candidates("train", layers=[layer], pooling=("span",))
    ev = G.load_candidates("eval", layers=[layer], pooling=("span",),
                           **({"featdir": lang_eval_featdir} if lang_eval_featdir else {}))
    vtr = V.load_vision("train")
    vev = V.load_vision("eval", ablate=ablate_eval)
    Vm_tr, Vg_tr, itr = V.align(tr, vtr, layer, grid_layer)
    Vm_ev, Vg_ev, iev = V.align(ev, vev, layer, grid_layer)
    return tr, ev, (Vm_tr, Vg_tr, itr, vtr), (Vm_ev, Vg_ev, iev, vev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--grid_layer", type=int, default=21)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ablate_eval", choices=["none", "blank", "noise"], default="none")
    ap.add_argument("--no_vision", type=int, default=0, help="skip the vision cache (arm L only)")
    ap.add_argument("--lang_eval_featdir", default=None,
                    help="LANGUAGE-SIDE image ablation: load the EVAL hidden-state cache from this "
                         "directory instead of feats_hidden (e.g. feats_hidden_noise, built by "
                         "extract_generator_hidden_ablated.py). Training rows are ALWAYS the real "
                         "ones, so this measures how much of the head's eval-time behaviour depends "
                         "on image content reaching the language-side vector.")
    ap.add_argument("--save_attn", type=int, default=0)
    ap.add_argument("--perm_vision", type=int, default=0,
                    help="PERMUTATION NULL: give every question a DIFFERENT question's image "
                         "features (candidates of a question still share one image, but the wrong "
                         "one). Any gain that survives this is not coming from the image content.")
    A = ap.parse_args()
    torch.set_num_threads(A.threads)
    os.makedirs(PARTS, exist_ok=True)

    tag = A.tag or (A.ablate_eval if A.ablate_eval != "none" else "")
    todo = [s for s in A.seeds
            if not os.path.exists(os.path.join(PARTS, f"{A.arm}{('_' + tag) if tag else ''}_seed{s}.npz"))]
    if not todo:
        print(f"[skip] {A.arm}{tag} all seeds present", flush=True)
        return

    t0 = time.time()
    if A.no_vision:
        tr = G.load_candidates("train", layers=[A.layer], pooling=("span",))
        ev = G.load_candidates("eval", layers=[A.layer], pooling=("span",),
                               **({"featdir": A.lang_eval_featdir} if A.lang_eval_featdir else {}))
        Vm_tr = Vg_tr = itr = Vm_ev = Vg_ev = iev = None
    else:
        tr, ev, (Vm_tr, Vg_tr, itr, vtr), (Vm_ev, Vg_ev, iev, vev) = \
            load_all(A.layer, A.grid_layer, A.ablate_eval, A.lang_eval_featdir)
        if A.perm_vision:
            # derange at the IMAGE level, then re-broadcast to rows, so the within-question
            # structure (all candidates share ONE image) is preserved and only the CONTENT is wrong
            for nm, (sp, vv, ii) in (("train", (tr, vtr, itr)), ("eval", (ev, vev, iev))):
                rng = np.random.default_rng(20260812 + (0 if nm == "train" else 1))
                nimg = vv["v_mean"].shape[0]
                p = rng.permutation(nimg)
                p = np.where(p == np.arange(nimg), (p + 1) % nimg, p)   # no fixed points
                ii[:] = p[ii]
            li = vtr["layers"].index(A.layer); gi = vtr["grid_layers"].index(A.grid_layer)
            Vm_tr = vtr["v_mean"][itr, li].astype(np.float32)
            Vg_tr = vtr["v_grid"][itr, gi].astype(np.float32)
            li = vev["layers"].index(A.layer); gi = vev["grid_layers"].index(A.grid_layer)
            Vm_ev = vev["v_mean"][iev, li].astype(np.float32)
            Vg_ev = vev["v_grid"][iev, gi].astype(np.float32)
            print("[perm_vision] image features deranged (no fixed points)", flush=True)
    Htr = tr.matrix("span", A.layer); Hev = ev.matrix("span", A.layer)
    ytr = np.array([r["y"] for r in tr.rows], dtype=np.float32)
    qtr = V.qid(tr.rows)
    print(f"[load] {time.time()-t0:.1f}s  train {Htr.shape} eval {Hev.shape}", flush=True)

    if A.arm == "xattn":
        Hs, mu, sd = V.zstd(Htr); Hes = (Hev - mu) / sd
        # patch bank standardised with the TRAIN patch statistics (per feature dim, over all patches)
        bank_tr = vtr["v_grid"][:, vtr["grid_layers"].index(A.grid_layer)].astype(np.float32)
        bank_ev = vev["v_grid"][:, vev["grid_layers"].index(A.grid_layer)].astype(np.float32)
        vmu = bank_tr.reshape(-1, bank_tr.shape[-1]).mean(0)
        vsd = bank_tr.reshape(-1, bank_tr.shape[-1]).std(0) + 1e-6
        bank_tr = (bank_tr - vmu) / vsd; bank_ev = (bank_ev - vmu) / vsd
    else:
        Xtr = V.build_features(A.arm, Htr, Vm_tr, Vg_tr)
        Xev = V.build_features(A.arm, Hev, Vm_ev, Vg_ev)
        Xtr, mu, sd = V.zstd(Xtr); Xev = (Xev - mu) / sd
        print(f"[feat] {A.arm}: d={Xtr.shape[1]}", flush=True)

    for s in todo:
        t1 = time.time()
        if A.arm == "xattn":
            m = fit_xattn(Hs, itr, bank_tr, ytr, qtr, seed=s)
            sc, att = predict_xattn(m, Hes, iev, bank_ev, want_attn=bool(A.save_attn))
        else:
            sc, m = V.fit_and_score(Xtr, ytr, qtr, Xev, seed=s)
            att = None
        r = G.sel_eff(V.scores_by_cand(ev, sc))
        out = {"scores": sc.astype(np.float32), "got": r["got"].astype(np.int8),
               "picks": r["picks"].astype(np.int8)}
        if att is not None:
            out["attn"] = att
        np.savez(os.path.join(PARTS, f"{A.arm}{('_' + tag) if tag else ''}_seed{s}.npz"), **out)
        summ = {"arm": A.arm, "tag": tag, "seed": s, "layer": A.layer, "grid_layer": A.grid_layer,
                "threads": A.threads, "sel_eff": r["sel_eff"], "acc": r["acc"],
                "per_ds": {k: v["sel_eff"] for k, v in r["per_ds"].items()},
                "contested": r["contested"]["sel_eff"],
                "cand_auroc": G.cand_auroc(V.scores_by_cand(ev, sc)),
                "minutes": round((time.time() - t1) / 60, 2)}
        json.dump(summ, open(os.path.join(
            PARTS, f"{A.arm}{('_' + tag) if tag else ''}_seed{s}.json"), "w"), indent=1)
        print(f"[{A.arm}{tag} seed {s}] sel_eff={r['sel_eff']:.6f} acc={r['acc']:.6f} "
              f"contested={r['contested']['sel_eff']:.6f} ({summ['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
