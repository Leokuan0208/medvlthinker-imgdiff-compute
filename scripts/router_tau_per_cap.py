#!/usr/bin/env python3
"""
router_tau_per_cap.py - refit the frozen margin gate at EACH resolution.
For full-res + each cap, fits the SAME pipeline as router_train.py (StandardScaler +
LogisticRegression on the 1-D margin; tau = err-rate-budget quantile of P) on the
PMC-VQA-train labels AT THAT RESOLUTION, then applies the frozen gate+tau to the eval
sets AT THAT RESOLUTION. full-res tau should reproduce the deployed 0.426.
CPU only; reuses existing data; no GPU.
"""
import json, glob, os, re, argparse
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

os.environ.setdefault("OMP_NUM_THREADS", "1")
np.random.seed(42)
COMPETENT = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
DIR_32B = "ckpts/gate_32b"
RES = [   # label -> (pmctrain_dir, eval_dir)
    ("fullres", "ckpts/gate_7b_pmctrain",             "ckpts/gate_7b_vllm"),
    ("cap640",  "ckpts/gate_7b_pmctrain_prune/cap640","ckpts/gate_7b_prune/cap640"),
    ("cap320",  "ckpts/gate_7b_pmctrain_prune/cap320","ckpts/gate_7b_prune/cap320"),
    ("cap160",  "ckpts/gate_7b_pmctrain_prune/cap160","ckpts/gate_7b_prune/cap160"),
    ("cap80",   "ckpts/gate_7b_pmctrain_prune/cap80", "ckpts/gate_7b_prune/cap80"),
]

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}_s\d+of\d+\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            l = l.strip()
            if not l: continue
            try: r = json.loads(l)
            except Exception: continue
            if "idx" in r: d[m.group(1)][r["idx"]] = r
    return d

def load_pmctrain(ckdir):
    rows = []
    for f in glob.glob(os.path.join(ckdir, "ckpt_nothink*.jsonl")):
        for l in open(f):
            l = l.strip()
            if l:
                try: rows.append(json.loads(l))
                except Exception: pass
    return rows

def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

def pick_tau(P_tr, y_tr):                      # identical to router_train.py
    err = 1.0 - float(np.mean(y_tr))
    return float(np.quantile(P_tr, min(max(err, 0.0), 1.0)))

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    A = ap.parse_args(); repo = A.repo
    r32 = load_arm(os.path.join(repo, DIR_32B), "think_norag")

    hdr = (f"{'res':<9}{'train_n':>8}{'train_acc':>10}{'tau':>8}{'train_esc%':>11}"
           f"{'eval_routed':>12}{'eval_32B':>10}{'eval_esc%':>10}")
    print(hdr); print("-"*len(hdr))
    for label, ptr_dir, ev_dir in RES:
        train_rows = load_pmctrain(os.path.join(repo, ptr_dir))
        if not train_rows:
            print(f"{label:<9}  (no pmctrain labels at {ptr_dir})"); continue
        y_tr = np.array([r["ok"] for r in train_rows], float)
        m_tr = np.array([[margin(r)] for r in train_rows], float)
        gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        gate.fit(m_tr, y_tr)
        P_tr = gate.predict_proba(m_tr)[:, 1]
        tau = pick_tau(P_tr, y_tr)
        train_esc = float(np.mean(P_tr < tau))

        ev = load_arm(os.path.join(repo, ev_dir), "nothink_norag")
        ok7, ok32, escs = [], [], []
        for ds in COMPETENT:
            idx = sorted(set(ev.get(ds, {})) & set(r32.get(ds, {})))
            if not idx: continue
            mg = np.array([[margin(ev[ds][i])] for i in idx], float)
            e = gate.predict_proba(mg)[:, 1] < tau
            ok7.append(np.array([ev[ds][i]["ok"] for i in idx], float))
            ok32.append(np.array([r32[ds][i]["ok"] for i in idx], float))
            escs.append(e)
        if not ok7:
            print(f"{label:<9}{len(train_rows):>8}{y_tr.mean():>10.3f}{tau:>8.3f}{100*train_esc:>10.1f}%   (no eval at res)")
            continue
        a7 = np.concatenate(ok7); a32 = np.concatenate(ok32); e = np.concatenate(escs)
        routed = np.where(e, a32, a7)
        print(f"{label:<9}{len(train_rows):>8}{y_tr.mean():>10.3f}{tau:>8.3f}{100*train_esc:>10.1f}%"
              f"{routed.mean():>12.3f}{a32.mean():>10.3f}{100*e.mean():>9.1f}%")
    print("\nRead: full-res tau should reproduce the deployed 0.426. Stable tau across caps =>")
    print("the gate transfers across resolution. eval_routed vs eval_32B => parity held or not.")

if __name__ == "__main__":
    run()
