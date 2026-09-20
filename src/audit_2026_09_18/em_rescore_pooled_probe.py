#!/usr/bin/env python3
"""em_rescore_pooled_probe.py -- re-score the SHIPPED eight-benchmark probe verifier
(ckpts/train/genframe_head_pooled_ens_v2, 24 frozen heads) in BOTH currencies on IDENTICAL picks.

READ-ONLY.  Nothing is fitted.  Nothing is written inside the repository (bytecode writing is
disabled before the project modules are imported).  All outputs go to --outdir.

Reused from the project, not re-invented:
  split            md5("nd"+img_md5) % 2 == 0 is HELD OUT       (freeze_pooled_selector.py / recipe.json)
  feature loading  h_span, layers 18/20/22, rows with n_tok > 0   (pooled_selector.verify)
  standardizer     standardizer_L{18,20,22}.npz  (mu, sd), frozen
  probe            head_sweep.MLP(3584, hidden=256), state_dicts head_L*_seed*.pt, frozen
  readout          genframe_data.rank_avg per head within the candidate set; mean over all 24 rank
                   vectors; argmax, FIRST-INDEX tie-break                       (recipe.json "readout")
  judge currency   candidate label = meta row "y"; greedy label = ckpt_{cell}_lingshu7b.judge.jsonl
  EM currency      run_openvqa.py's scorer, VERBATIM (src/labeling/run_openvqa.py:83-93) -- the same
                   definition the August dual-currency round used
                   (src/cascade_methods/verifier_hparams_nulls.py:33-47), which is what the stored
                   `oks` field holds.  NOTE: this "normalised exact match" is RELAXED -- it also
                   returns 1 on substring containment in either direction.
  token F1         src/cascade_methods/open_cascade_analyze.py:25 token_f1, VERBATIM (third currency)
Added here and labelled as such (NOT a stored project currency):
  strict EM        em_norm(pred) == em_norm(gold), i.e. the project's normaliser WITHOUT the
                   containment clause.

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 em_rescore_pooled_probe.py
"""
import os
import sys

sys.dont_write_bytecode = True          # importing project modules must not write __pycache__ into the repo
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse, hashlib, json, re, string, time
from collections import Counter, defaultdict

import numpy as np
import torch

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(MAIN, "src/training_methods"))
import head_sweep as HS                     # noqa: E402  (MLP only; import is side-effect free)
from genframe_data import rank_avg          # noqa: E402  (THE canonical tie-aware ranker)

FEATS = os.path.join(MAIN, "feats_hidden")
CK = os.path.join(MAIN, "ckpts/openvqa/cheap_lingshu7b")
PROBE_DIR = os.path.join(MAIN, "ckpts/train/genframe_head_pooled_ens_v2")
HEADLINE = os.path.join(MAIN, "results/cascade_methods/artifacts/head_final_stack_PVFIXED_2026-09-13.json")
ENS = [18, 20, 22]
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"slake_open", "vqa_rad_open"}
BOOT_SEED = 20260918
NBOOT = 10000


# ---- run_openvqa.py's scorer, VERBATIM (src/labeling/run_openvqa.py:83-93) -------------------
def em_norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def em_score(pred, gold):
    p, g = em_norm(pred), em_norm(gold)
    if not p:
        return 0
    if p == g:
        return 1
    if g and (g in p.split() or p in g.split() or g in p or p in g):
        return 1
    return 0


# ---- strict variant: the same normaliser, NO containment clause (added here, labelled) -------
def strict_em(pred, gold):
    p, g = em_norm(pred), em_norm(gold)
    return int(bool(p) and p == g)


# ---- open_cascade_analyze.py:25 token_f1, VERBATIM -------------------------------------------
def token_f1(pred, gold):
    p, g = set(em_norm(pred).split()), set(em_norm(gold).split())
    if not p or not g:
        return 0.0
    ov = len(p & g)
    if ov == 0:
        return 0.0
    prec, rec = ov / len(p), ov / len(g)
    return 2 * prec * rec / (prec + rec)


def row_norm(s):
    """extract_generator_hidden.py:67 -- the identity under which candidate rows were deduped."""
    return str(s).strip().lower()


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def nwords(s):
    return len(str(s).split())


def loadj(p):
    out = {}
    for l in open(p):
        if l.strip():
            d = json.loads(l)
            out[d["idx"]] = d
    return out


def finfo(p):
    st = os.stat(p)
    return {"path": p, "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "is_symlink": os.path.islink(p), "realpath": os.path.realpath(p)}


# ================================================================================ probes
def load_probes(d):
    recipe = json.load(open(os.path.join(d, "recipe.json")))
    heads, stats = {}, {}
    for L in ENS:
        z = np.load(os.path.join(d, f"standardizer_L{L}.npz"))
        stats[L] = (z["mu"].astype(np.float32), z["sd"].astype(np.float32))
        hs = []
        for s in range(recipe["seeds_per_layer"]):
            m = HS.MLP(len(z["mu"]), hidden=256)
            m.load_state_dict(torch.load(os.path.join(d, f"head_L{L}_seed{s}.pt"), map_location="cpu"))
            m.eval()
            hs.append(m)
        heads[L] = hs
    return heads, stats, recipe


def batched_scores(heads, stats, X, bs=8192):
    """(24, n_rows) raw head outputs, batch 8192 (HS.predict's batch size)."""
    out = []
    with torch.no_grad():
        for L in ENS:
            mu, sd = stats[L]
            Z = ((X[L] - mu) / sd).astype(np.float32)
            for m in heads[L]:
                out.append(np.concatenate([m(torch.tensor(Z[i:i + bs])).numpy()
                                           for i in range(0, len(Z), bs)]))
    return np.stack(out)


# ================================================================================ stage A
def stage_a(args):
    heads, stats, recipe = load_probes(PROBE_DIR)
    print(f"loaded {sum(len(v) for v in heads.values())} frozen heads from {PROBE_DIR}", flush=True)
    records, files, nulls = {}, {}, {}
    for cell in BENCH:
        t0 = time.time()
        stem = "generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}"
        npz, mj = f"{FEATS}/{stem}.npz", f"{FEATS}/{stem}.meta.json"
        gp, gjp, scp = (f"{CK}/ckpt_{cell}_lingshu7b.jsonl", f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl",
                        f"{CK}/ckpt_{cell}_lingshu7b_sc8.jsonl")
        files[cell] = {k: finfo(p) for k, p in (("features_npz", npz), ("features_meta", mj),
                                                ("greedy_dump", gp), ("greedy_judge", gjp),
                                                ("sc8_dump", scp))}
        z = np.load(npz)
        meta = json.load(open(mj))
        lay = [int(x) for x in z["layers"]]
        dsf = {cell} if cell in SHARED else None
        keep = [i for i, r in enumerate(meta["rows"])
                if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)
                and half(r["img_md5"]) == 0]
        rr = [meta["rows"][i] for i in keep]
        H = z["h_span"]                                   # read ONCE
        X = {L: H[:, lay.index(L)][keep].astype(np.float32) for L in ENS}
        del H
        S = batched_scores(heads, stats, X)               # (24, n_rows)
        greedy, gjudge, sc8 = loadj(gp), loadj(gjp), loadj(scp)
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        qs = sorted([q for q in byq if q in gjudge], key=lambda k: (len(str(k)), str(k)))
        nl = Counter()
        nl["heldout_questions_without_greedy_judge"] = sum(1 for q in byq if q not in gjudge)
        recs = []
        hl = {L: heads[L] for L in ENS}
        for q in qs:
            ii = np.array(byq[q])
            ranks = np.stack([rank_avg(S[k][ii]) for k in range(S.shape[0])])
            hr = ranks.mean(0)
            pick = int(np.argmax(hr))                                   # first-index tie-break
            n_top = int((hr == hr.max()).sum())
            pick_last = int(len(hr) - 1 - np.argmax(hr[::-1]))          # sensitivity: last-index
            pick_pq = None
            if args.per_question_forward:
                rk = []
                with torch.no_grad():
                    for L in ENS:
                        mu, sd = stats[L]
                        Zq = torch.tensor(((X[L][ii] - mu) / sd), dtype=torch.float32)
                        for m in hl[L]:
                            rk.append(rank_avg(np.atleast_1d(m(Zq).numpy())))
                pick_pq = int(np.argmax(np.mean(rk, axis=0)))
            g, s8 = greedy.get(q), sc8.get(q)
            if g is None or s8 is None:
                nl["question_missing_greedy_or_sc8_dump"] += 1
                continue
            gold = s8["gold"]
            if g["gold"] != gold:
                nl["gold_differs_greedy_vs_sc8_dump"] += 1
            preds = list(s8["preds"])
            pn = [row_norm(a) for a in preds]
            cands = []
            for i in ii:
                r = rr[i]
                em = em_score(r["ans"], gold)
                if r["na"] in pn:
                    j = pn.index(r["na"])
                    if int(s8["oks"][j]) != em:
                        nl["candidate_EM_recomputed_differs_from_stored_oks"] += 1
                else:
                    nl["candidate_row_not_found_in_sc8_preds"] += 1
                cands.append({"ans": r["ans"], "y": int(r["y"]), "em": em,
                              "st": strict_em(r["ans"], gold), "f1": token_f1(r["ans"], gold),
                              "votes": pn.count(r["na"]), "w": nwords(r["ans"])})
            if len(preds) != 8:
                nl["sc8_pool_not_8"] += 1
            nas = {rr[i]["na"] for i in ii}
            nl["sc8_slots_without_a_judged_row"] += sum(1 for a in pn if a not in nas)
            nl["sc8_slots_total"] += len(pn)
            # self-consistency: run_openvqa.py:223 grouping (em_norm), restricted to the judged rows
            grp_votes = Counter(em_norm(a) for a in preds)
            first_row = {}
            for ci, i in enumerate(ii):
                first_row.setdefault(em_norm(rr[i]["ans"]), ci)
            order = []
            for a in preds:
                k = em_norm(a)
                if k in first_row and k not in order:
                    order.append(k)
            for k in first_row:                       # rows whose group never appears (should not happen)
                if k not in order:
                    order.append(k)
            sc_key = max(order, key=lambda k: grp_votes.get(k, 0))      # max() keeps the FIRST maximum
            sc_pick = first_row[sc_key]
            if em_norm(s8["modal_pred"]) != sc_key:
                nl["sc_pick_group_differs_from_stored_modal_pred"] += 1
            gpred = g["preds"][0]
            gem = em_score(gpred, gold)
            if int(g["oks"][0]) != gem:
                nl["greedy_EM_recomputed_differs_from_stored_oks"] += 1
            gna = row_norm(gpred)
            g_pool_y = next((c["y"] for c, i in zip(cands, ii) if rr[i]["na"] == gna), None)
            recs.append({"idx": q, "img": rr[ii[0]]["img_md5"], "gold": gold, "gold_w": nwords(gold),
                         "cands": cands, "pick": pick, "pick_last": pick_last, "pick_pq": pick_pq,
                         "n_top": n_top, "sc_pick": sc_pick,
                         "greedy": {"ans": gpred, "y": int(gjudge[q]["judge_ok"]), "em": gem,
                                    "st": strict_em(gpred, gold), "f1": token_f1(gpred, gold),
                                    "w": nwords(gpred), "pool_y": g_pool_y}})
        imgs_per_q = {q: {rr[i]["img_md5"] for i in byq[q]} for q in qs}
        nl["questions_with_more_than_one_img_md5"] = sum(1 for v in imgs_per_q.values() if len(v) > 1)
        nl["rows_with_null_img_md5"] = sum(1 for r in rr if not r.get("img_md5"))
        records[cell] = recs
        nulls[cell] = dict(nl)
        vj = np.mean([r["cands"][r["pick"]]["y"] for r in recs])
        gj = np.mean([r["greedy"]["y"] for r in recs])
        print(f"  [A] {cell:17} rows {len(rr):6}  n {len(recs):5}  greedy_J {gj:.4f}  verifier_J {vj:.4f}"
              f"  delta_J {vj - gj:+.4f}   ({time.time() - t0:.0f}s)", flush=True)
    return records, files, nulls, recipe


# ================================================================================ stage B
COLS = ["g_j", "v_j", "sc_j", "or_j", "g_em", "v_em", "sc_em", "or_em",
        "g_st", "v_st", "sc_st", "or_st", "g_f1", "v_f1", "sc_f1", "or_f1", "gpool_j"]
CI = lambda i: COLS.index(i)   # noqa: E731


def matrix(recs):
    M = np.zeros((len(recs), len(COLS)))
    for n, r in enumerate(recs):
        c, g = r["cands"], r["greedy"]
        v, s = c[r["pick"]], c[r["sc_pick"]]
        for cur, key in (("j", "y"), ("em", "em"), ("st", "st"), ("f1", "f1")):
            M[n, CI(f"g_{cur}")] = g[key]
            M[n, CI(f"v_{cur}")] = v[key]
            M[n, CI(f"sc_{cur}")] = s[key]
            M[n, CI(f"or_{cur}")] = max(x[key] for x in c)
        # sensitivity: greedy's judge label replaced by the POOL's judge label for the same string
        M[n, CI("gpool_j")] = g["pool_y"] if g["pool_y"] is not None else g["y"]
    return M


def cluster_boot(M, clusters, nboot, rng, chunk=500):
    """Paired cluster bootstrap over IMAGES.  Returns (nboot, n_cols) resampled means.
    Vectorised: resampling clusters with replacement == multinomial cluster weights."""
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    ncl = len(keys)
    S = np.zeros((ncl, M.shape[1]))
    np.add.at(S, inv, M)
    cnt = np.bincount(inv, minlength=ncl).astype(float)
    out = np.empty((nboot, M.shape[1]))
    p = np.full(ncl, 1.0 / ncl)
    for a in range(0, nboot, chunk):
        W = rng.multinomial(ncl, p, size=min(chunk, nboot - a)).astype(float)
        out[a:a + W.shape[0]] = (W @ S) / (W @ cnt)[:, None]
    return out, ncl, int(cnt.max())


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def verdict(lo, hi):
    return "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")


CONTRASTS = {  # name -> (col_a, col_b): a - b
    "delta_judge": ("v_j", "g_j"), "delta_em": ("v_em", "g_em"),
    "delta_strict_em": ("v_st", "g_st"), "delta_token_f1": ("v_f1", "g_f1"),
    "delta_judge_greedy_relabelled_by_pool": ("v_j", "gpool_j"),
    "sc_minus_greedy_judge": ("sc_j", "g_j"), "sc_minus_greedy_em": ("sc_em", "g_em"),
    "verifier_minus_sc_judge": ("v_j", "sc_j"), "verifier_minus_sc_em": ("v_em", "sc_em"),
    "verifier_minus_sc_strict_em": ("v_st", "sc_st"), "verifier_minus_sc_token_f1": ("v_f1", "sc_f1"),
}


def stage_b(records, nboot):
    ss = np.random.SeedSequence(BOOT_SEED)
    child = ss.spawn(len(BENCH) + 1)
    per, reps, point = {}, {}, {}
    for b, cell in enumerate(BENCH):
        recs = records[cell]
        M = matrix(recs)
        imgs = [r["img"] for r in recs]
        R, ncl, maxcl = cluster_boot(M, imgs, nboot, np.random.default_rng(child[b]))
        mean = M.mean(0)
        o = {"n": len(recs), "n_images": ncl, "max_questions_per_image": maxcl}
        for cur, nm in (("j", "judge"), ("em", "em"), ("st", "strict_em"), ("f1", "token_f1")):
            o[f"greedy_{nm}"] = float(mean[CI(f"g_{cur}")])
            o[f"verifier_{nm}"] = float(mean[CI(f"v_{cur}")])
            o[f"sc_{nm}"] = float(mean[CI(f"sc_{cur}")])
            o[f"oracle_{nm}"] = float(mean[CI(f"or_{cur}")])
            o[f"sel_eff_{nm}"] = float(mean[CI(f"v_{cur}")] / mean[CI(f"or_{cur}")]) if mean[CI(f"or_{cur}")] > 0 else None
        reps[cell], point[cell] = {}, {}
        for nm, (a, c) in CONTRASTS.items():
            d = float(mean[CI(a)] - mean[CI(c)])
            rv = R[:, CI(a)] - R[:, CI(c)]
            lo, hi = ci(rv)
            o[nm] = d
            o["ci_" + nm.replace("delta_", "")] = [lo, hi]
            o["verdict_" + nm.replace("delta_", "")] = verdict(lo, hi)
            reps[cell][nm], point[cell][nm] = rv, d
        # currency gap: how much of the judge delta is NOT there under EM
        gap = (R[:, CI("v_j")] - R[:, CI("g_j")]) - (R[:, CI("v_em")] - R[:, CI("g_em")])
        o["currency_gap_judge_minus_em"] = o["delta_judge"] - o["delta_em"]
        o["ci_currency_gap"] = ci(gap)
        reps[cell]["currency_gap"], point[cell]["currency_gap"] = gap, o["currency_gap_judge_minus_em"]
        # paraphrase-drift diagnostic
        g_y = np.array([r["greedy"]["y"] for r in recs]); g_e = np.array([r["greedy"]["em"] for r in recs])
        v_y = np.array([r["cands"][r["pick"]]["y"] for r in recs])
        v_e = np.array([r["cands"][r["pick"]]["em"] for r in recs])
        o["drift"] = {
            "greedy_judgeYes_emNo": float(((g_y == 1) & (g_e == 0)).mean()),
            "verifier_judgeYes_emNo": float(((v_y == 1) & (v_e == 0)).mean()),
            "greedy_judgeNo_emYes": float(((g_y == 0) & (g_e == 1)).mean()),
            "verifier_judgeNo_emYes": float(((v_y == 0) & (v_e == 1)).mean()),
            "greedy_currency_agreement": float((g_y == g_e).mean()),
            "verifier_currency_agreement": float((v_y == v_e).mean())}
        # answer lengths (whitespace words)
        gw = np.array([r["greedy"]["w"] for r in recs]); vw = np.array([r["cands"][r["pick"]]["w"] for r in recs])
        sw = np.array([r["cands"][r["sc_pick"]]["w"] for r in recs]); goldw = np.array([r["gold_w"] for r in recs])
        o["answer_length_words"] = {
            "greedy_mean": float(gw.mean()), "verifier_pick_mean": float(vw.mean()),
            "sc_pick_mean": float(sw.mean()), "gold_mean": float(goldw.mean()),
            "gold_median": float(np.median(goldw)), "frac_gold_over_3_words": float((goldw > 3).mean()),
            "frac_pick_longer_than_greedy": float((vw > gw).mean()),
            "frac_pick_shorter_than_greedy": float((vw < gw).mean())}
        # is the pick the same string as greedy?
        same = np.array([row_norm(r["cands"][r["pick"]]["ans"]) == row_norm(r["greedy"]["ans"]) for r in recs])
        o["frac_pick_same_string_as_greedy"] = float(same.mean())
        o["mean_candidates_per_question"] = float(np.mean([len(r["cands"]) for r in recs]))
        # judge self-consistency: greedy's judge label vs the pool's label for the SAME string
        both = [(r["greedy"]["y"], r["greedy"]["pool_y"]) for r in recs if r["greedy"]["pool_y"] is not None]
        if both:
            a = np.array(both)
            o["judge_relabel_check"] = {
                "n_greedy_string_in_pool": int(len(a)), "frac_of_questions": float(len(a) / len(recs)),
                "label_agreement": float((a[:, 0] == a[:, 1]).mean()),
                "greedyNo_poolYes": int(((a[:, 0] == 0) & (a[:, 1] == 1)).sum()),
                "greedyYes_poolNo": int(((a[:, 0] == 1) & (a[:, 1] == 0)).sum())}
        # tie-break / forward-pass sensitivity
        pl = np.array([r["cands"][r["pick_last"]]["y"] for r in recs])
        ple = np.array([r["cands"][r["pick_last"]]["em"] for r in recs])
        o["tiebreak_sensitivity"] = {
            "n_questions_with_tied_top_score": int(sum(1 for r in recs if r["n_top"] > 1 and len(r["cands"]) > 1)),
            "verifier_judge_last_index": float(pl.mean()), "verifier_em_last_index": float(ple.mean())}
        if recs and recs[0]["pick_pq"] is not None:
            o["per_question_forward_picks_differ"] = int(sum(1 for r in recs if r["pick_pq"] != r["pick"]))
        per[cell] = o
    # ---------------- macro
    macro = {"n_benchmarks": len(BENCH), "n_questions": int(sum(per[c]["n"] for c in BENCH))}
    for cur in ("judge", "em", "strict_em", "token_f1"):
        for arm in ("greedy", "verifier", "sc", "oracle"):
            macro[f"{arm}_{cur}"] = float(np.mean([per[c][f"{arm}_{cur}"] for c in BENCH]))
    rng8 = np.random.default_rng(child[-1])
    bidx = rng8.integers(0, len(BENCH), size=(nboot, len(BENCH)))       # benchmark-level resample, n=8
    for nm in list(CONTRASTS) + ["currency_gap"]:
        pts = np.array([point[c][nm] for c in BENCH])
        RR = np.stack([reps[c][nm] for c in BENCH], axis=1)            # (nboot, 8), independent per benchmark
        m = float(pts.mean())
        lo1, hi1 = ci(RR.mean(1))
        bm = pts[bidx].mean(1)
        lo2, hi2 = ci(bm)
        two = np.take_along_axis(RR, bidx, axis=1).mean(1)             # resample benchmarks AND images
        lo3, hi3 = ci(two)
        sd = pts.std(ddof=1); t975_df7 = 2.3646                          # Student t, df = 7
        macro[nm] = {
            "macro": m, "n_positive_benchmarks": int((pts > 0).sum()),
            "ci_image_clustered_within_benchmark": [lo1, hi1], "verdict_image_clustered": verdict(lo1, hi1),
            "ci_benchmark_level_n8": [lo2, hi2], "verdict_benchmark_level": verdict(lo2, hi2),
            "frac_benchmark_resamples_le_0": float((bm <= 0).mean()),
            "ci_two_stage_benchmarks_then_images": [lo3, hi3],
            "t_interval_df7_over_benchmarks": [float(m - t975_df7 * sd / np.sqrt(len(pts))),
                                               float(m + t975_df7 * sd / np.sqrt(len(pts)))],
            "sd_across_benchmarks": float(sd)}
    # question-weighted (micro) for reference only
    macro["micro_reference"] = {}
    N = macro["n_questions"]
    for cur in ("judge", "em", "strict_em", "token_f1"):
        macro["micro_reference"][f"delta_{cur}"] = float(sum(per[c][f"delta_{cur}"] * per[c]["n"] for c in BENCH) / N)
    return per, macro


def cross_benchmark_images(records):
    seen = defaultdict(set)
    for cell, recs in records.items():
        for r in recs:
            seen[r["img"]].add(cell)
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    pairs = Counter(tuple(sorted(v)) for v in multi.values())
    return {"n_distinct_images_all_benchmarks": len(seen), "n_images_in_more_than_one_benchmark": len(multi),
            "by_benchmark_set": {" + ".join(k): v for k, v in pairs.most_common()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--per_question_forward", action="store_true",
                    help="ALSO score each candidate set with its own forward pass (pooled_selector.verify's "
                         "path) and count picks that differ from the batch-8192 path")
    ap.add_argument("--reuse_cache", action="store_true")
    A = ap.parse_args()
    torch.set_num_threads(A.threads)
    torch.manual_seed(0)
    t0 = time.time()
    cache = os.path.join(A.outdir, "em_rescore_per_question_cache.json")
    if A.reuse_cache and os.path.exists(cache):
        c = json.load(open(cache))
        records, files, nulls, recipe = c["records"], c["files"], c["nulls"], c["recipe"]
        print(f"reusing {cache}", flush=True)
    else:
        records, files, nulls, recipe = stage_a(A)
        json.dump({"records": records, "files": files, "nulls": nulls, "recipe": recipe}, open(cache, "w"))
    per, macro = stage_b(records, A.nboot)

    head = json.load(open(HEADLINE))
    claim = recipe["measured_on_held_out_halves"]
    repro = {
        "recipe_macro_verifier_minus_greedy": claim["macro_verifier_minus_greedy"],
        "recipe_reload_verified_macro": claim.get("reload_verified_macro"),
        "this_run_macro_delta_judge": macro["delta_judge"]["macro"],
        "deviation_vs_recipe_macro": macro["delta_judge"]["macro"] - claim["macro_verifier_minus_greedy"],
        "headline_artifact": HEADLINE,
        "headline_artifact_seeds": head.get("seeds"),
        "per_benchmark": {c: {
            "n_headline": head["cells"][c]["n_questions"], "n_this_run": per[c]["n"],
            "greedy_headline": head["cells"][c]["greedy"], "greedy_this_run": per[c]["greedy_judge"],
            "delta_headline_refit": head["cells"][c]["pooled_ens_minus_greedy"],
            "delta_this_run_frozen24": per[c]["delta_judge"]} for c in BENCH}}
    repro["n_and_greedy_match_exactly"] = bool(all(
        v["n_headline"] == v["n_this_run"] and abs(v["greedy_headline"] - v["greedy_this_run"]) < 1e-12
        for v in repro["per_benchmark"].values()))
    repro["macro_matches_to_1e-4"] = bool(abs(repro["deviation_vs_recipe_macro"]) <= 1.5e-4)

    conv = {
        "threads": A.threads, "torch_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
        "gpu_used": False, "refit": False, "probe_dir": PROBE_DIR,
        "probe_files": [finfo(os.path.join(PROBE_DIR, f)) for f in sorted(os.listdir(PROBE_DIR))],
        "n_heads": 24, "layers": ENS, "forward_pass": "batch 8192 over the whole held-out matrix (HS.predict's batch size)",
        "readout": "genframe_data.rank_avg per head within the candidate set; mean of all 24 rank vectors; "
                   "np.argmax = FIRST-INDEX tie-break; row order = feature-cache meta order",
        "split": 'held out = md5("nd"+img_md5) % 2 == 0; rows with n_tok > 0; questions with a greedy judge label',
        "candidate_set": "DISTINCT sampled answers (identity = str.strip().lower()) of the T=0.7 N=8 pool that carry a judge label",
        "judge_currency": "candidate: meta row y (Lingshu-32B judge); greedy: ckpt_{cell}_lingshu7b.judge.jsonl judge_ok",
        "em_currency": "run_openvqa.py score(): normalise (lowercase, drop articles/stop-words the|a|an|is|are|of|in|on|at|"
                       "this|image|picture, strip punctuation) then equality OR substring containment either way. "
                       "This is the stored `oks` currency and the August 2026-08-16 'normalised exact match'.",
        "strict_em_currency": "same normaliser, equality only. ADDED HERE; not a stored project currency.",
        "token_f1_currency": "open_cascade_analyze.py token_f1 (set-of-tokens F1 on the same normaliser); mean over questions",
        "self_consistency": "majority vote over the 8 samples grouped by run_openvqa's normaliser (run_openvqa.py:223), "
                            "first-seen group wins ties, representative = first judged row of that group",
        "oracle": "max over the judged candidate rows of the question, per currency",
        "bootstrap": {"n_resamples": A.nboot, "seed": BOOT_SEED,
                      "seeding": "np.random.SeedSequence(seed).spawn(9): one child per benchmark in BENCH order, the 9th for the n=8 benchmark resample",
                      "cluster": "img_md5 within benchmark; paired (one draw of image weights feeds every column)",
                      "interval": "percentile 2.5/97.5",
                      "macro_i": "independent image-clustered resample inside each benchmark, macro = mean of 8",
                      "macro_ii": "resample the 8 benchmark point deltas with replacement (n=8)",
                      "macro_iii_extra": "two-stage: resample benchmarks, then use that benchmark's image-resampled delta"},
        "feature_and_dump_files": files,
        "seconds": round(time.time() - t0, 1)}

    art = {"title": "Shipped eight-benchmark probe verifier re-scored in judge AND exact-match currency on identical picks",
           "date": "2026-09-18", "no_fabricated_numbers": True, "cpu_only": True, "no_refit": True,
           "script": os.path.abspath(__file__),
           "reproduction_of_judge_headline": repro, "per_benchmark": per, "macro": macro,
           "pairing_null_tests": nulls, "cross_benchmark_image_overlap": cross_benchmark_images(records),
           "conventions": conv}
    out = os.path.join(A.outdir, "em_rescore_pooled_probe_2026-09-18.json")
    json.dump(art, open(out, "w"), indent=1)

    # ---------------- stdout tables
    print("\n=== REPRODUCTION (judge currency) ===")
    print(f"  recipe macro {claim['macro_verifier_minus_greedy']:+.6f} | reload-verified {claim.get('reload_verified_macro')} | "
          f"this run {macro['delta_judge']['macro']:+.6f} | deviation {repro['deviation_vs_recipe_macro']:+.6f} | "
          f"n & greedy exact match: {repro['n_and_greedy_match_exactly']}")
    for c in BENCH:
        v = repro["per_benchmark"][c]
        print(f"  {c:17} n {v['n_this_run']:5} (headline {v['n_headline']:5})  delta frozen24 {v['delta_this_run_frozen24']:+.4f}"
              f"  headline 5-seed refit {v['delta_headline_refit']:+.4f}")
    print("\n=== TWO-CURRENCY TABLE (identical picks) ===")
    print(f"  {'benchmark':17} {'n':>5} {'imgs':>5} | {'grdy_J':>6} {'ver_J':>6} {'dJ':>7} {'CI_J':>19} | "
          f"{'grdy_EM':>7} {'ver_EM':>6} {'dEM':>7} {'CI_EM':>19}")
    for c in BENCH:
        o = per[c]
        print(f"  {c:17} {o['n']:5} {o['n_images']:5} | {o['greedy_judge']:.4f} {o['verifier_judge']:.4f} {o['delta_judge']:+.4f} "
              f"[{o['ci_judge'][0]:+.4f},{o['ci_judge'][1]:+.4f}] | {o['greedy_em']:.4f}  {o['verifier_em']:.4f} {o['delta_em']:+.4f} "
              f"[{o['ci_em'][0]:+.4f},{o['ci_em'][1]:+.4f}] {o['verdict_em']}")
    print("\n=== strict EM / token-F1 / SC / oracle ===")
    for c in BENCH:
        o = per[c]
        print(f"  {c:17} dSTRICT {o['delta_strict_em']:+.4f} [{o['ci_strict_em'][0]:+.4f},{o['ci_strict_em'][1]:+.4f}]  "
              f"dF1 {o['delta_token_f1']:+.4f} [{o['ci_token_f1'][0]:+.4f},{o['ci_token_f1'][1]:+.4f}] | "
              f"SC_J {o['sc_judge']:.4f} SC_EM {o['sc_em']:.4f} | or_J {o['oracle_judge']:.4f} or_EM {o['oracle_em']:.4f} "
              f"or_ST {o['oracle_strict_em']:.4f} | V-SC J {o['verifier_minus_sc_judge']:+.4f} EM {o['verifier_minus_sc_em']:+.4f}")
    print("\n=== drift + length ===")
    for c in BENCH:
        o = per[c]; d = o["drift"]; L = o["answer_length_words"]
        print(f"  {c:17} JyEn g {d['greedy_judgeYes_emNo']:.4f} v {d['verifier_judgeYes_emNo']:.4f} | JnEy g {d['greedy_judgeNo_emYes']:.4f} "
              f"v {d['verifier_judgeNo_emYes']:.4f} | words g {L['greedy_mean']:.2f} v {L['verifier_pick_mean']:.2f} gold {L['gold_mean']:.2f} "
              f"| gap {o['currency_gap_judge_minus_em']:+.4f} [{o['ci_currency_gap'][0]:+.4f},{o['ci_currency_gap'][1]:+.4f}]")
    print("\n=== MACRO ===")
    for nm in list(CONTRASTS) + ["currency_gap"]:
        m = macro[nm]
        print(f"  {nm:40} {m['macro']:+.4f}  img-clustered [{m['ci_image_clustered_within_benchmark'][0]:+.4f},"
              f"{m['ci_image_clustered_within_benchmark'][1]:+.4f}] {m['verdict_image_clustered']:4} | bench n=8 "
              f"[{m['ci_benchmark_level_n8'][0]:+.4f},{m['ci_benchmark_level_n8'][1]:+.4f}] {m['verdict_benchmark_level']:4} "
              f"| t df7 [{m['t_interval_df7_over_benchmarks'][0]:+.4f},{m['t_interval_df7_over_benchmarks'][1]:+.4f}] "
              f"| +benchmarks {m['n_positive_benchmarks']}/8")
    print("\n=== pairing null tests ===")
    for c in BENCH:
        print(f"  {c:17} {nulls[c]}")
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
