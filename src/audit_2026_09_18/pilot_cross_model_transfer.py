#!/usr/bin/env python3
"""PILOT (2026-09-18, audit session) -- does a correctness probe survive along a fine-tuning lineage?

Lingshu-7B is a medical fine-tune of Qwen2.5-VL-7B-Instruct (same 28-layer, 3584-d trunk). The shipped
probes (ckpts/train/genframe_head_pooled_ens_v2, 24 heads) were fitted ONLY on Lingshu hidden states.
Here they are applied, frozen and unmodified, to QWEN's hidden states of QWEN's own candidates on the
held-out image halves, and used as Qwen's best-of-N verifier.

  arm A  lingshu probes + lingshu standardizer                      (pure zero-shot transfer)
  arm B  lingshu probes + standardizer re-estimated on Qwen's TRAIN-half states (label-free adaptation)
  ctrl   lingshu probes on lingshu states                           (must reproduce recipe +0.0737)
  ref    Qwen-native pooled_ens, read from head_final_stack_qwen_2026-09-13.json (NOT refit here)

Read-only. CPU, <=4 threads. Judge currency (row['y'], greedy judge_ok) -- same currency as the headline.
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

torch.set_num_threads(4)
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import head_sweep as HS                      # noqa: E402
from genframe_data import rank_avg           # noqa: E402

FE = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
PD = os.path.join(ROOT, "ckpts/train/genframe_head_pooled_ens_v2")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]
RNG = np.random.default_rng(20260918)


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load(stem, dsf=None):
    z = np.load(f"{FE}/{stem}.npz"); m = json.load(open(f"{FE}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"])
            if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    rr = [m["rows"][i] for i in keep]
    H = z["h_span"]
    return {L: H[keep, lay.index(L)].astype(np.float32) for L in ENS}, rr


def heads():
    out = {}
    for L in ENS:
        st = np.load(f"{PD}/standardizer_L{L}.npz")
        ms = []
        for s in range(8):
            m = HS.MLP(3584, 256, 1, 0.0, False)
            m.load_state_dict(torch.load(f"{PD}/head_L{L}_seed{s}.pt", map_location="cpu"))
            m.eval(); ms.append(m)
        out[L] = (st["mu"], st["sd"], ms)
    return out


def select(H, X, rows, gok, stdz=None):
    """returns per-question arrays: img, greedy_ok, pick_ok, rand_ok, oracle_ok"""
    S = []
    for L in ENS:
        mu, sd, ms = H[L]
        if stdz is not None:
            mu, sd = stdz[L]
        Xs = (X[L] - mu) / sd
        S.append(np.stack([HS.predict(m, Xs) for m in ms]))
    y = np.array([r["y"] for r in rows], dtype=int)
    byq = {}
    for i, r in enumerate(rows):
        byq.setdefault(r["idx"], []).append(i)
    res = []
    for q, ii in byq.items():
        if q not in gok:
            continue
        ii = np.array(ii)
        hr = np.mean([rank_avg(s[k][ii]) for s in S for k in range(s.shape[0])], axis=0)
        res.append((rows[ii[0]]["img_md5"], gok[q], int(y[ii][int(np.argmax(hr))]),
                    float(y[ii].mean()), int(y[ii].max())))
    return res


def boot(res, cols, B=10000):
    """image-clustered bootstrap of mean(col_a - col_b)."""
    imgs = {}
    for r in res:
        imgs.setdefault(r[0], []).append(r)
    keys = list(imgs)
    d = np.array([sum(x[cols[0]] - x[cols[1]] for x in imgs[k]) for k in keys], float)
    n = np.array([len(imgs[k]) for k in keys], float)
    idx = RNG.integers(0, len(keys), size=(B, len(keys)))
    b = d[idx].sum(1) / n[idx].sum(1)
    return float(d.sum() / n.sum()), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def gjudge(cell, tag):
    g = {}
    for l in open(f"{CK}/ckpt_{cell}_{tag}.judge.jsonl"):
        if l.strip():
            d = json.loads(l); g[d["idx"]] = int(d["judge_ok"])
    return g


def main():
    t0 = time.time()
    H = heads()
    ref = json.load(open(f"{ART}/head_final_stack_qwen_2026-09-13.json"))
    out = {"title": "PILOT: frozen Lingshu probes applied zero-shot to Qwen2.5-VL-7B hidden states",
           "date": "2026-09-18", "no_fabricated_numbers": True, "currency": "Lingshu-32B judge",
           "probe_dir": os.path.relpath(PD, ROOT), "threads": 4, "bootstrap": "10000, clustered by img_md5",
           "cells": {}}
    for cell in BENCH:
        # ---- control: lingshu probes on lingshu states
        stem = "generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}"
        X, rr = load(stem, {cell} if cell in SHARED else None)
        ho = np.array([half(r["img_md5"]) == 0 for r in rr])
        rc = select(H, {L: X[L][ho] for L in ENS}, [rr[i] for i in np.where(ho)[0]],
                    gjudge(cell, "lingshu7b"))
        del X
        # ---- transfer: lingshu probes on qwen states
        Xq, rq = load(f"generator_eval_qwen_{cell}")
        hq = np.array([half(r["img_md5"]) == 0 for r in rq])
        gq = gjudge(cell, "qwen25vl7b")
        Xho = {L: Xq[L][hq] for L in ENS}; rho = [rq[i] for i in np.where(hq)[0]]
        ra = select(H, Xho, rho, gq)
        stdz = {L: (Xq[L][~hq].mean(0), Xq[L][~hq].std(0) + 1e-6) for L in ENS}
        rb = select(H, Xho, rho, gq, stdz)
        del Xq
        c = {"n_q_lingshu": len(rc), "n_q_qwen": len(ra)}
        c["ctrl_lingshu_on_lingshu"], c["ctrl_ci"] = boot(rc, (2, 1))
        c["qwen_greedy"] = float(np.mean([r[1] for r in ra]))
        c["qwen_random_pick"] = float(np.mean([r[3] for r in ra]))
        c["qwen_oracle"] = float(np.mean([r[4] for r in ra]))
        c["A_zeroshot_minus_greedy"], c["A_ci"] = boot(ra, (2, 1))
        c["A_zeroshot_minus_random"], c["A_vs_random_ci"] = boot(ra, (2, 3))
        c["B_restd_minus_greedy"], c["B_ci"] = boot(rb, (2, 1))
        c["B_restd_minus_random"], c["B_vs_random_ci"] = boot(rb, (2, 3))
        rv = ref["cells"].get(cell, {})
        c["ref_qwen_native_pooled_ens_minus_greedy"] = rv.get("pooled_ens_minus_greedy")
        c["ref_qwen_native_4dom_minus_greedy"] = rv.get("deployed_4dom_L21ish_minus_greedy")
        c["ref_n_questions"] = rv.get("n_questions"); c["ref_greedy"] = rv.get("greedy")
        out["cells"][cell] = c
        print(f"{cell:17} ctrl {c['ctrl_lingshu_on_lingshu']:+.4f} | qwen n{c['n_q_qwen']:5} greedy "
              f"{c['qwen_greedy']:.4f} rand {c['qwen_random_pick']:.4f} | A {c['A_zeroshot_minus_greedy']:+.4f} "
              f"{c['A_ci']} | B {c['B_restd_minus_greedy']:+.4f} {c['B_ci']} | native "
              f"{c['ref_qwen_native_pooled_ens_minus_greedy']}", flush=True)
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    cs = out["cells"].values()
    out["macro"] = {k: float(np.mean([v[k] for v in cs])) for k in
                    ("ctrl_lingshu_on_lingshu", "A_zeroshot_minus_greedy", "A_zeroshot_minus_random",
                     "B_restd_minus_greedy", "B_restd_minus_random",
                     "ref_qwen_native_pooled_ens_minus_greedy", "ref_qwen_native_4dom_minus_greedy")}
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out["macro"], indent=1))
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
