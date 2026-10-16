#!/usr/bin/env python3
"""final_method_eval.py -- fit, freeze and score the probe verifier on EVERY test question of every
cell, in every currency the comparable papers use, with a probe that never saw that question's image.

TWO-FOLD CROSS-FITTING OVER IMAGES. Fold f in {0,1} scores the questions whose image has
md5("nd"+img_md5) % 2 == f, with probes fitted on the four original train domains plus every
evaluation cell's OTHER image half. Any training row whose image appears in fold f of ANY scored
cell is dropped (this also covers images shared between an open cell and its closed cell, and the
VQA-Med 2019 test images that sit in our vqamed cell). Fold 0 alone is the project's historical
protocol (the +0.0737 headline is fold 0 of the cap320 pools); folds 0+1 cover the full official
test split, which is what LLaVA-Med / Lingshu / MedRegA / MedGemma report.

PROBE: Linear(3584,H) -> GELU -> Linear(H,1) on h_span, layers 18/20/22, BCE, AdamW wd 1e-2, standardised
per layer -- the shipped recipe; H / epochs / schedule are flags (defaults = shipped).
READOUT: --readout rank (shipped) or rank_vote (rank, ties broken by vote count; adopted 2026-10-15 by
the pre-specified rule in readout_variants_2026-10-15.json).
POLICY: --policy T07 (every cell's N=8 T=0.7 pool) or Tmap (frozen_temperature_map_2026-10-15.json:
the frozen per-benchmark temperature where a pool exists, else T=0.7). The probe is always FITTED on
T=0.7 rows, as the temperature experiment that motivated the map was.

CURRENCIES (identical picks):
  judge     MedVLThinker-32B label (candidate: feature-meta row y; greedy: ckpt_<cell>_<greedy_tag>.judge.jsonl)
  em        run_openvqa.py's lenient scorer (equality or containment), via em_rescore_pooled_probe
  f1        project token-F1 (open_cascade_analyze.py), via em_rescore_pooled_probe
  llava_recall / llava_f1   LLaVA-Med v1.0.0 calculate_f1score + normalize_word, imported VERBATIM from
            /data/dan/repos/llava_med_v1.0.0_eval (open-question 'recall' is what LLaVA-Med reports)
  llava_closed  LLaVA-Med v1.0.0 closed rule, verbatim logic of run_eval.py: hit iff the normalised
            prediction contains 'yes' or 'no' AND contains the normalised gold (closed cells only)
  imageclef VQA-Med 2019 strict accuracy, RE-IMPLEMENTED (no official code read): lowercase, strip
            punctuation, collapse spaces; correct iff equal to any '#' alternative, with each '()' part
            either kept or dropped (vqamed2019_test only)
  xjudge    MedGemma-27B-it labels, when --xjudge <preds.jsonl> (+ .judge.jsonl) is given

  OMP_NUM_THREADS=4 python3 final_method_eval.py --name <run> [--legacy_cap320 --frozen_fold0 DIR] ...
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse, json, re, string, time
from collections import defaultdict
import numpy as np
import torch

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
WT = MAIN + "/.claude/worktrees/exp-2026-10-15"
sys.path.insert(0, WT + "/src/audit_2026_09_18")
sys.path.insert(0, WT + "/src/training_methods")
sys.path.insert(0, "/data/dan/repos/llava_med_v1.0.0_eval")
import em_rescore_pooled_probe as ER                       # noqa: E402  scorers, half(), loaders
import head_sweep as HS                                    # noqa: E402
from genframe_data import rank_avg                         # noqa: E402
from eval_metrics.evaluate_metrics import calculate_f1score  # noqa: E402  LLaVA-Med v1.0.0, verbatim
from eval_metrics.glossary import normalize_word           # noqa: E402  LLaVA-Med v1.0.0, verbatim

CK = MAIN + "/ckpts/openvqa/cheap_lingshu7b"
FEATS = MAIN + "/feats_hidden"
ART = MAIN + "/results/cascade_methods/artifacts"
ENS = [18, 20, 22]
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
COMP = ["vqa_rad_closed", "pathvqa_closed", "slake_closed", "vqamed2019_test"]
CLOSED = {"vqa_rad_closed", "pathvqa_closed", "slake_closed"}
TMAP = json.load(open(WT + "/results/cascade_methods/artifacts/frozen_temperature_map_2026-10-15.json"))["frozen_T"]
TMAP_COMP = {"vqa_rad_closed": 0.2, "vqamed2019_test": 0.4}     # make_comparability_queue.py, pre-specified
NBOOT, SEED = 10000, 20261017


# ------------------------------------------------------------------ currencies
def llava(pred, gold):
    f1, _, rec = calculate_f1score(normalize_word(str(pred).lower()), normalize_word(str(gold).lower()))
    return float(rec), float(f1)


def llava_closed(pred, gold):
    p, g = normalize_word(str(pred).lower()), normalize_word(str(gold).lower())
    return float(("yes" in p or "no" in p) and g in p)


def _clef_norm(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def imageclef(pred, ref):
    p = _clef_norm(pred)
    for alt in ref.split("#"):
        kept = alt.replace("(", "").replace(")", "")
        dropped = re.sub(r"\([^)]*\)", "", alt)
        if p in (_clef_norm(kept), _clef_norm(dropped)):
            return 1.0
    return 0.0


# ------------------------------------------------------------------ cell layout
def pool_tag(cell, policy, legacy):
    if legacy:
        # cap320 pools at other temperatures exist from the August temperature sweep (lingshu7bT02/T04/T10)
        if policy == "Tmap":
            T = TMAP.get(cell, 0.7)
            tag = f"lingshu7bT{int(round(T * 10)):02d}"
            if T != 0.7 and os.path.exists(f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"):
                return tag
        return "lingshu7b"
    if policy == "Tmap":
        T = TMAP.get(cell, TMAP_COMP.get(cell, 0.7))
        if T != 0.7:
            tag = f"lingshu7bFRT{int(round(T * 10)):02d}"
            if os.path.exists(f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"):
                return tag
    return "lingshu7bFR"


def eval_stem(cell, tag, legacy):
    if legacy:
        if tag != "lingshu7b":                     # cap320 temperature-sweep caches (layers 18/20/22)
            return f"generator_eval_ens_{tag.replace('lingshu7b', '')}_{cell}", None
        return ("generator_eval_finelayer", {cell}) if cell in ER.SHARED else (f"generator_eval_finelayer_{cell}", None)
    return f"generator_eval_{tag.replace('lingshu7b', '')}_{cell}", None


def load_cache(stem, dsf):
    z = np.load(f"{FEATS}/{stem}.npz"); m = json.load(open(f"{FEATS}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    H = z["h_span"]
    X = {L: H[:, lay.index(L)][keep].astype(np.float32) for L in ENS}
    return X, [m["rows"][i] for i in keep]


def loadj(p):
    return ER.loadj(p) if os.path.exists(p) else {}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cells", nargs="+", default=OPEN8 + COMP)
    ap.add_argument("--policy", choices=["T07", "Tmap"], default="T07")
    ap.add_argument("--readout", choices=["rank", "rank_vote"], default="rank_vote")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sched", choices=["none", "cos"], default="none")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--legacy_cap320", action="store_true", help="score the stored cap320 pools/caches (validation)")
    ap.add_argument("--frozen_fold0", default=None, help="use these frozen probes for fold 0 instead of fitting")
    ap.add_argument("--save_probes", default=None, help="directory to freeze fitted probes into (per fold)")
    ap.add_argument("--xjudge", default=None, help="MedGemma-27B preds jsonl with idx 'cell|q|k' (+ .judge.jsonl)")
    A = ap.parse_args()
    torch.set_num_threads(A.threads)
    t0 = time.time()
    legacy = A.legacy_cap320
    gtag = "lingshu7b" if legacy else "lingshu7bFR"
    cells = [c for c in A.cells if not (legacy and c in COMP)]

    # ---- training-side rows: four original domains (T=0.7) + every cell's T=0.7 cache
    tr_stem = "generator_train_finelayer" if legacy else "generator_train_FR"
    Xo, ro = load_cache(tr_stem, TRAIN_DOMAINS)
    fit_X, fit_rows, fit_src, fit_isorig = {L: [Xo[L]] for L in ENS}, list(ro), [r["ds"] for r in ro], [True] * len(ro)
    ev = {}
    for c in cells:
        stem07, dsf = eval_stem(c, "lingshu7b" if legacy else "lingshu7bFR", legacy)
        if not os.path.exists(f"{FEATS}/{stem07}.npz"):
            print(f"  [skip] {c}: no T=0.7 cache {stem07}", flush=True); continue
        X07, r07 = load_cache(stem07, dsf)
        for L in ENS:
            fit_X[L].append(X07[L])
        fit_rows += r07; fit_src += [c] * len(r07); fit_isorig += [False] * len(r07)
        tag = pool_tag(c, A.policy, legacy)
        if tag == ("lingshu7b" if legacy else "lingshu7bFR"):
            Xe, re_ = X07, r07
        else:
            st, ds2 = eval_stem(c, tag, legacy)
            Xe, re_ = load_cache(st, ds2)
        ev[c] = {"tag": tag, "X": Xe, "rows": re_}
        print(f"  loaded {c:17} eval tag {tag} rows {len(re_)} ({time.time() - t0:.0f}s)", flush=True)
    FX = {L: np.concatenate(fit_X[L]) for L in ENS}
    Fy = np.array([r["y"] for r in fit_rows], np.float32)
    Fq = np.array([f"{s}|{r['idx']}" for s, r in zip(fit_src, fit_rows)])
    Fimg = np.array([r["img_md5"] for r in fit_rows])
    Fhalf = np.array([ER.half(i) for i in Fimg])
    Forig = np.array(fit_isorig)

    # ---- greedy, pools, cross-judge labels
    xlab, xj = {}, {}
    if A.xjudge:
        for l in open(A.xjudge):
            if l.strip():
                d = json.loads(l); xlab[tuple(d["idx"].split("|")[:2]) + (d["modal_pred"],)] = d["idx"]
        for l in open(A.xjudge.replace(".jsonl", ".judge.jsonl")):
            if l.strip():
                d = json.loads(l); xj.setdefault(d["idx"], int(d["judge_ok"]))
    refs = {}
    vqm = "/data/dan/dataset/comparability_cells/vqamed2019_test.json"
    if os.path.exists(vqm):
        refs = {r["idx"]: r["answer_ref"] for r in json.load(open(vqm))}

    records = defaultdict(list)
    fold_info = {}
    for f in A.folds:
        evimg = set()
        for c, d in ev.items():
            evimg |= {r["img_md5"] for r in d["rows"] if ER.half(r["img_md5"]) == f}
        tr = np.array([(Forig[i] or Fhalf[i] == 1 - f) and Fimg[i] not in evimg for i in range(len(Fy))])
        fold_info[f] = {"train_rows": int(tr.sum()), "train_rows_original_domains": int((tr & Forig).sum()),
                        "eval_images": len(evimg)}
        # probes
        if f == 0 and A.frozen_fold0:
            heads, stats, _ = ER.load_probes(A.frozen_fold0)
            fold_info[f]["probes"] = f"frozen {A.frozen_fold0}"
        else:
            heads, stats = {}, {}
            for L in ENS:
                mu, sd = FX[L][tr].mean(0), FX[L][tr].std(0) + 1e-6
                Xs = (FX[L][tr] - mu) / sd
                heads[L] = [HS.fit(Xs, Fy[tr], Fq[tr], None, objective="bce", hidden=A.hidden, wd=1e-2,
                                   epochs=A.epochs, sched=A.sched, seed=s) for s in range(A.seeds)]
                stats[L] = (mu.astype(np.float32), sd.astype(np.float32))
                print(f"  fold {f} layer {L}: {A.seeds} probes fitted ({time.time() - t0:.0f}s)", flush=True)
                if A.save_probes:
                    od = f"{A.save_probes}/fold{f}"; os.makedirs(od, exist_ok=True)
                    for s, m in enumerate(heads[L]):
                        torch.save(m.state_dict(), f"{od}/head_L{L}_seed{s}.pt")
                    np.savez(f"{od}/standardizer_L{L}.npz", mu=mu, sd=sd)
            fold_info[f]["probes"] = f"fitted, hidden {A.hidden}, epochs {A.epochs}, sched {A.sched}, seeds {A.seeds}"
        # score fold f of every cell
        for c, d in ev.items():
            idx = [i for i, r in enumerate(d["rows"]) if ER.half(r["img_md5"]) == f]
            if not idx:
                continue
            rr = [d["rows"][i] for i in idx]
            S = []
            with torch.no_grad():
                for L in ENS:
                    mu, sd = stats[L]
                    Z = torch.tensor(((d["X"][L][idx] - mu) / sd).astype(np.float32))
                    for m in heads[L]:
                        m.eval(); S.append(m(Z).numpy())
            S = np.stack(S)
            sc8 = loadj(f"{CK}/ckpt_{c}_{d['tag']}_sc8.jsonl")
            greedy = loadj(f"{CK}/ckpt_{c}_{gtag}.jsonl")
            gj = {k: int(v["judge_ok"]) for k, v in loadj(f"{CK}/ckpt_{c}_{gtag}.judge.jsonl").items()}
            byq = defaultdict(list)
            for i, r in enumerate(rr):
                byq[r["idx"]].append(i)
            for q, ii in byq.items():
                if q not in gj or q not in greedy or q not in sc8:
                    continue
                ii = np.array(ii)
                rk = np.mean([rank_avg(S[k][ii]) for k in range(S.shape[0])], 0)
                pn = [ER.row_norm(a) for a in sc8[q]["preds"]]
                votes = np.array([pn.count(rr[i]["na"]) for i in ii], float)
                top = np.flatnonzero(rk == rk.max())
                p = int(top[np.argmax(votes[top])]) if A.readout == "rank_vote" else int(np.argmax(rk))
                sc_p = int(np.argmax(votes))
                gold = sc8[q]["gold"]
                gpred = greedy[q]["preds"][0]

                def cur(ans, jy):
                    o = {"judge": jy, "em": ER.em_score(ans, gold), "f1": ER.token_f1(ans, gold)}
                    if c in CLOSED:
                        o["llava_closed"] = llava_closed(ans, gold)
                    else:
                        o["llava_recall"], o["llava_f1"] = llava(ans, gold)
                    if c == "vqamed2019_test":
                        o["imageclef"] = imageclef(ans, refs[q])
                    if A.xjudge:
                        k = xlab.get((c, str(q), ans))
                        o["xjudge"] = xj.get(k) if k is not None else None
                    return o
                records[c].append({"idx": q, "img": rr[ii[0]]["img_md5"], "fold": f,
                                   "pick": cur(rr[ii[p]]["ans"], int(rr[ii[p]]["y"])),
                                   "sc": cur(rr[ii[sc_p]]["ans"], int(rr[ii[sc_p]]["y"])),
                                   "greedy": cur(gpred, gj[q]),
                                   "oracle_judge": int(max(rr[i]["y"] for i in ii)),
                                   "pick_ans": rr[ii[p]]["ans"], "greedy_ans": gpred})
            print(f"  fold {f} scored {c:17} questions {sum(1 for r in records[c] if r['fold'] == f)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- summaries + paired image-clustered bootstrap (pick - greedy, sc - greedy), per cell and fold
    ss = np.random.SeedSequence(SEED)
    out_cells, macro_parts = {}, defaultdict(lambda: defaultdict(list))
    for b, (c, recs) in enumerate(records.items()):
        rng = np.random.default_rng(ss.spawn(1)[0])
        out_cells[c] = {}
        for scope in [("fold0", [0]), ("all", A.folds)]:
            R = [r for r in recs if r["fold"] in scope[1]]
            if not R:
                continue
            keys, inv = np.unique([r["img"] for r in R], return_inverse=True)
            W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=NBOOT).astype(float)
            o = {"n": len(R), "n_images": len(keys), "oracle_judge": float(np.mean([r["oracle_judge"] for r in R]))}
            for cu in R[0]["pick"]:
                for arm in ("pick", "sc", "greedy"):
                    v = [r[arm][cu] for r in R]
                    ok = [x is not None for x in v]
                    o[f"{arm}_{cu}"] = float(np.mean([x for x in v if x is not None])) if any(ok) else None
                for arm in ("pick", "sc"):
                    m = np.array([r[arm][cu] is not None and r["greedy"][cu] is not None for r in R])
                    if not m.any():
                        continue
                    dlt = np.array([(r[arm][cu] - r["greedy"][cu]) if mm else 0.0 for r, mm in zip(R, m)])
                    Rs = (W @ np.bincount(inv, weights=dlt, minlength=len(keys))) / np.maximum(
                        W @ np.bincount(inv, weights=m.astype(float), minlength=len(keys)), 1e-12)
                    pt = float(dlt[m].mean())
                    lo, hi = float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))
                    o[f"{arm}_minus_greedy_{cu}"] = pt
                    o[f"ci_{arm}_minus_greedy_{cu}"] = [lo, hi]
                    if c in OPEN8:
                        macro_parts[scope[0]][f"{arm}_minus_greedy_{cu}"].append((pt, Rs))
            out_cells[c][scope[0]] = o
    macro = {}
    for scope, parts in macro_parts.items():
        macro[scope] = {}
        for key, lst in parts.items():
            if len(lst) != len([c for c in OPEN8 if c in records]):
                continue
            pts = np.array([x[0] for x in lst]); Rm = np.stack([x[1] for x in lst], 1).mean(1)
            lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
            macro[scope][key] = {"macro": float(pts.mean()), "ci": [lo, hi],
                                 "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                                 "n_positive": int((pts > 0).sum()), "n_cells": len(pts)}
    # full official test splits: open + closed cells combined, question-weighted (as the papers report)
    full = {}
    for fam, (o_, c_) in {"vqa_rad": ("vqa_rad_open", "vqa_rad_closed"), "pathvqa": ("pathvqa_open", "pathvqa_closed"),
                          "slake_en": ("slake_open", "slake_closed")}.items():
        if o_ in records and c_ in records:
            R = records[o_] + records[c_]
            full[fam] = {"n": len(R), "n_open": len(records[o_]), "n_closed": len(records[c_])}
            for arm in ("pick", "greedy", "sc"):
                full[fam][f"{arm}_judge"] = float(np.mean([r[arm]["judge"] for r in R]))
                full[fam][f"{arm}_llava_open_recall"] = float(np.mean([r[arm]["llava_recall"] for r in records[o_]]))
                full[fam][f"{arm}_llava_closed_acc"] = float(np.mean([r[arm]["llava_closed"] for r in records[c_]]))
    art = {"title": f"Final-method evaluation: {A.name}", "date": time.strftime("%Y-%m-%d"),
           "no_fabricated_numbers": True, "script": os.path.abspath(__file__), "args": vars(A),
           "legacy_cap320": legacy, "greedy_tag": gtag, "eval_tags": {c: d["tag"] for c, d in ev.items()},
           "folds": fold_info, "macro_over_open8": macro, "full_official_splits": full, "cells": out_cells,
           "bootstrap": {"n": NBOOT, "seed": SEED, "cluster": "img_md5 within cell, paired"},
           "seconds": round(time.time() - t0, 1)}
    out = f"{ART}/final_eval_{A.name}.json"
    json.dump(art, open(out, "w"), indent=1)
    json.dump({c: v for c, v in records.items()}, open(f"/data/dan/exp_2026_10/final_eval_{A.name}_records.json", "w"))
    for scope in macro:
        print(f"\nMACRO over open8, {scope}:")
        for k in sorted(macro[scope]):
            v = macro[scope][k]
            print(f"  {k:32} {v['macro']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] {v['verdict']:4} "
                  f"{v['n_positive']}/{v['n_cells']}")
    print("\nFULL OFFICIAL SPLITS:", json.dumps(full, indent=1))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    os.makedirs("/data/dan/exp_2026_10", exist_ok=True)
    main()
