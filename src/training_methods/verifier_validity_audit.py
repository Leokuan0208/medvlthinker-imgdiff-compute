#!/usr/bin/env python3
"""verifier_validity_audit.py -- OFFLINE (no GPU) audit of the trained open-text verifier's validity.

Answers two of the three questions in the 2026-07-29 corrective experiment:

  (A) TRAINING-DATA OVERLAP. Reconstruct the EXACT seed-0 grouped train/test split used by
      run_lora_verifier_open.py to train ckpts/train/lora_verifier_pooled4, and intersect the train
      question set with the question sets of the three open cells actually reported in the paper
      (transfer_dump_{slake,pathvqa,vqa_rad}_open_lingshu7b.json; n = 645 / 1500 / 200).
      The reconstruction is VALIDATED against ckpts/train/lora_verifier_pooled4/clean_dump.json,
      which reconstruct_clean_dump.py already verified byte-for-byte against the saved per-question
      verifier scores (perq_sc8.json).  If the reconstructed held-out 30% does not reproduce
      clean_dump exactly, the script REFUSES to report overlap numbers.

  (B) IN-DOMAIN vs OUT-OF-DOMAIN. Split each reported open cell into SEEN (question was in the
      verifier's training split) and UNSEEN (question was held out), and report, per stratum:
      greedy accuracy, self-consistency, verifier best-of-8 selection accuracy, oracle@8, the
      selection GAIN over greedy, per-candidate AUROC, and the oracle conversion rate.
      Paired question-level bootstrap CIs on the gain and on the seen-minus-unseen gain difference.

  Also reports the same numbers on the two TRUE held-out transfer sets (kvasir_open, radimagenet_open)
  which contain zero training questions for the radimagenet case and 70%-training for kvasir.

Run (from repo root, CPU only, ~1-2 min; the parquet reads dominate):
  python3 src/training_methods/verifier_validity_audit.py
Writes: results/cascade_methods/artifacts/verifier_validity_2026-07-29.json  (section "A"/"B")
"""
import os, sys, json, glob, io, random, math
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
CK = J("ckpts/openvqa/cheap_lingshu7b")
ADAPTER = J("ckpts/train/lora_verifier_pooled4")
TAG = "lingshu7b"
SEED = 0
DSETS = ["slake_open", "pathvqa_open", "vqa_rad_open", "kvasir_open"]   # run_lora_verifier_open.py default
OUT = J("results/cascade_methods/artifacts/verifier_validity_2026-07-29.json")

norm = lambda s: str(s).strip().lower()


def loadj(p):
    return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}


# ---------------------------------------------------------------- image-key maps (membership only:
# we deliberately do NOT decode the images -- the training script's membership test is exactly
# `isinstance(img, dict) and "bytes" in img`, which needs no decode).
def slake_keys():
    ks = []
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            if os.path.exists(os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])):
                ks.append(x["qid"])
    return set(ks)


def parquet_keys(base):
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/test-*.parquet"))], ignore_index=True)
    ks = set()
    for i, r in df.iterrows():
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]; a = conv[1]["value"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            ks.add(int(i))
    return ks


def kvasir_keys():
    return {r["idx"] for r in json.load(open("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"))
            if os.path.exists(r["img_path"])}


def build_split():
    """Reconstruct run_lora_verifier_open.py's QREC key order + seed-0 grouped 70/30 split."""
    IMG = {"slake_open": slake_keys(),
           "pathvqa_open": parquet_keys("/data/dan/dataset/path_vqa/data"),
           "vqa_rad_open": parquet_keys("/data/dan/dataset/vqa_rad/data"),
           "kvasir_open": kvasir_keys()}
    qkeys, slabels = [], {}
    for ds in DSETS:
        sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
        exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
        jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
        aj = defaultdict(dict)
        for cid, r in exp.items():
            if cid in jud:
                oi = cid.split("#")[0]; oi = int(oi) if oi.lstrip("-").isdigit() else oi
                aj[oi][norm(r["modal_pred"])] = jud[cid]
        for i in sc:                                   # dict order == file order == training script
            if i in IMG[ds] and i in aj:
                qkeys.append((ds, i)); slabels[(ds, i)] = (sc[i]["preds"], dict(aj[i]))
    keys = list(qkeys)
    rng = random.Random(SEED); rng.shuffle(keys)
    ntr = int(0.7 * len(keys))
    return qkeys, keys[:ntr], keys[ntr:], slabels


def validate_against_clean_dump(test_keys_ordered, slabels):
    """clean_dump.json was produced by reconstruct_clean_dump.py and validated element-wise against the
    saved verifier scores.  Reproducing it exactly (same order, same idx) proves our split is THE split."""
    cd = json.load(open(os.path.join(ADAPTER, "clean_dump.json")))
    recon = []
    for k in test_keys_ordered:
        preds, sl_map = slabels[k]
        sl = [sl_map.get(norm(a)) for a in preds]
        if all(x is None for x in sl):
            continue
        recon.append((k[0], k[1]))
    ref = [(r["ds"], r["idx"]) for r in cd]
    ok = (len(recon) == len(ref)) and all(a == b for a, b in zip(recon, ref))
    return ok, len(recon), len(ref)


# ---------------------------------------------------------------- metrics
def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    m = y >= 0
    s, y = s[m], y[m]
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    us, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(us)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cell_metrics(rows):
    """rows: list of transfer-dump records. Returns per-question metric vectors."""
    g, sc_, tr, orc = [], [], [], []
    cs, cl = [], []
    for r in rows:
        sl = r["sl"]; s = r["scores"]; preds = r["preds"]
        g.append(int(r["greedy_ok"]))
        c = Counter(norm(a) for a in preds); top = c.most_common(1)[0][0]
        # self-consistency: label of the modal answer, matched through the candidate list
        scv = 0
        for a, l in zip(preds, sl):
            if norm(a) == top:
                scv = max(0, l); break
        sc_.append(scv)
        k = int(np.argmax(s)); tr.append(max(0, sl[k]))
        orc.append(max([x for x in sl if x >= 0] or [0]))
        for a, l, ss in zip(preds, sl, s):
            cs.append(ss); cl.append(l)
    return dict(greedy=g, sc=sc_, trained=tr, oracle=orc, cand_scores=cs, cand_labels=cl)


def boot_diff(a, b, n=10000, seed=0):
    """paired question-level bootstrap of mean(a)-mean(b) (a,b same length, same questions)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed); n_i = len(a)
    idx = rng.integers(0, n_i, size=(n, n_i))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_2sample_diff(a1, b1, a2, b2, n=10000, seed=0):
    """(mean(a1)-mean(b1)) - (mean(a2)-mean(b2)); independent resamples of the two strata."""
    a1, b1, a2, b2 = map(lambda x: np.asarray(x, float), (a1, b1, a2, b2))
    rng = np.random.default_rng(seed)
    i1 = rng.integers(0, len(a1), size=(n, len(a1)))
    i2 = rng.integers(0, len(a2), size=(n, len(a2)))
    d = (a1[i1].mean(1) - b1[i1].mean(1)) - (a2[i2].mean(1) - b2[i2].mean(1))
    pt = (a1.mean() - b1.mean()) - (a2.mean() - b2.mean())
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summarize(rows, label):
    if not rows:
        return None
    m = cell_metrics(rows)
    gain, lo, hi = boot_diff(m["trained"], m["greedy"])
    out = dict(stratum=label, n=len(rows),
               greedy=float(np.mean(m["greedy"])), sc=float(np.mean(m["sc"])),
               verifier=float(np.mean(m["trained"])), oracle=float(np.mean(m["oracle"])),
               gain_over_greedy=gain, gain_ci=[lo, hi],
               cand_auroc=auroc(m["cand_scores"], m["cand_labels"]),
               n_candidates=int(sum(1 for x in m["cand_labels"] if x >= 0)))
    head = float(np.mean(m["oracle"])) - float(np.mean(m["greedy"]))
    out["oracle_conversion"] = (gain / head) if head > 1e-9 else None
    return out


def image_keys_for(ds, want):
    """idx -> a stable image identity, so we can measure IMAGE-level (not just question-level) leakage.
    SLAKE reuses one scan across many questions, so a 'held-out question' can still show the verifier an
    image it trained on."""
    import hashlib
    if ds == "slake_open":
        return {x["qid"]: x["img_name"] for x in json.load(open("/data/dan/dataset/slake/test.json"))
                if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en" and x["qid"] in want}
    if ds == "kvasir_open":
        return {r["idx"]: os.path.basename(r["img_path"])
                for r in json.load(open("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"))
                if r["idx"] in want}
    base = {"pathvqa_open": "/data/dan/dataset/path_vqa/data",
            "vqa_rad_open": "/data/dan/dataset/vqa_rad/data"}.get(ds)
    if base is None:
        return {}
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/test-*.parquet"))], ignore_index=True)
    out = {}
    for i in want:
        b = df.iloc[i]["image"]["bytes"]
        out[i] = hashlib.md5(b).hexdigest()
    return out


def image_overlap(TRAIN, TEST, per_ds_rows):
    """For each reported open cell: of the questions NOT in training, how many still reuse an image that a
    TRAINING question used?"""
    out = {}
    for ds, rows in per_ds_rows.items():
        idxs = [r["idx"] for r in rows]
        # all pool questions of this ds (train+test), so we know which images training saw
        pool = [k[1] for k in list(TRAIN) + list(TEST) if k[0] == ds]
        want = set(idxs) | set(pool)
        imap = image_keys_for(ds, want)
        if not imap:
            continue
        train_imgs = {imap[i] for i in pool if (ds, i) in TRAIN and i in imap}
        unseen = [i for i in idxs if (ds, i) not in TRAIN and i in imap]
        reuse = [i for i in unseen if imap[i] in train_imgs]
        n_img_total = len({imap[i] for i in idxs if i in imap})
        out[ds] = dict(n_eval=len(idxs), n_distinct_images=n_img_total,
                       n_questions_unseen=len(unseen),
                       n_unseen_whose_image_was_in_training=len(reuse),
                       pct_unseen_with_trained_image=(100.0 * len(reuse) / len(unseen)) if unseen else None)
        print(f"      {ds:<18} distinct images={n_img_total:>4}  unseen-q={len(unseen):>4}  "
              f"of which image-seen-in-training={len(reuse):>4} "
              f"({out[ds]['pct_unseen_with_trained_image']:.1f}%)")
    return out


def lodo_evidence():
    """Dataset-level (leave-one-dataset-out-style) transfer that ALREADY exists on disk, read verbatim."""
    ev = {}
    p = J("ckpts/train/lora_verifier_open/transfer_result.json")
    if os.path.exists(p):
        r = json.load(open(p))["kvasir_open"]
        ev["lora_verifier_open_to_kvasir"] = dict(
            note="verifier trained on SLAKE-open + PathVQA-open ONLY (run_lora_verifier_open.py, "
                 "VERIF_DSETS=slake,pathvqa); Kvasir is a dataset it never saw at all",
            source="ckpts/train/lora_verifier_open/transfer_result.json", **r,
            gain_over_greedy=r["trained"] - r["greedy"],
            oracle_conversion=(r["trained"] - r["greedy"]) / (r["oracle"] - r["greedy"]))
    # NB: ckpts/train/lora_verifier_pooled4/transfer_result.json is NOT usable here -- it was last
    # overwritten by the MedVLThinker-7B-generator run (TAG=7b, same timestamp as
    # transfer_dump_radimagenet_open_7b.json), so its greedy values are a different generator's.
    # Compute the Lingshu-generator RadImageNet transfer directly from the lingshu7b dump instead.
    p = J("ckpts/train/lora_verifier_pooled4/transfer_dump_radimagenet_open_lingshu7b.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        s = summarize(rows, "full")
        ev["pooled4_to_radimagenet_lingshu_generator"] = dict(
            note="RadImageNet-open is NOT in the pooled4 training pool at all (0/2000 overlap, verified above); "
                 "computed from the lingshu7b dump, not from transfer_result.json (which holds a 7B-generator run)",
            source="ckpts/train/lora_verifier_pooled4/transfer_dump_radimagenet_open_lingshu7b.json",
            n=s["n"], greedy=s["greedy"], sc=s["sc"], trained=s["verifier"], oracle=s["oracle"],
            gain_over_greedy=s["gain_over_greedy"], gain_ci=s["gain_ci"],
            cand_auroc=s["cand_auroc"], oracle_conversion=s["oracle_conversion"])
    return ev


def main():
    print("[1/3] reconstructing the seed-0 grouped split ...", flush=True)
    qkeys, train_keys_o, test_keys_o, slabels = build_split()
    print(f"      QREC questions = {len(qkeys)}  train = {len(train_keys_o)}  test = {len(test_keys_o)}")
    ok, nrec, nref = validate_against_clean_dump(test_keys_o, slabels)
    print(f"      validation vs clean_dump.json: match={ok}  ({nrec} reconstructed vs {nref} saved)")
    if not ok:
        print("REFUSING to report overlap: split reconstruction did not reproduce clean_dump.json.")
        sys.exit(2)
    TRAIN = set(train_keys_o); TEST = set(test_keys_o)

    res = {"provenance": {
        "adapter": "ckpts/train/lora_verifier_pooled4",
        "trainer": "src/training_methods/run_lora_verifier_open.py (seed 0, grouped 70/30 by question idx)",
        "training_pool_datasets": DSETS,
        "n_questions_in_pool": len(qkeys),
        "n_train_questions": len(train_keys_o),
        "n_heldout_questions": len(test_keys_o),
        "split_validated_against": "ckpts/train/lora_verifier_pooled4/clean_dump.json (exact order+idx match)",
        "eval_dumps": "ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_lingshu7b.json",
    }}

    # -------- (A) overlap table
    print("[2/3] overlap table ...", flush=True)
    A = {}
    per_ds_rows = {}
    for ds in ["slake_open", "pathvqa_open", "vqa_rad_open", "kvasir_open", "radimagenet_open"]:
        p = os.path.join(ADAPTER, f"transfer_dump_{ds}_{TAG}.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        per_ds_rows[ds] = rows
        seen = [r for r in rows if (ds, r["idx"]) in TRAIN]
        unseen_held = [r for r in rows if (ds, r["idx"]) in TEST]
        unseen_never = [r for r in rows if (ds, r["idx"]) not in TRAIN and (ds, r["idx"]) not in TEST]
        A[ds] = dict(n_eval=len(rows), n_seen_in_training=len(seen),
                     pct_seen=100.0 * len(seen) / len(rows),
                     n_heldout_30pct=len(unseen_held),
                     n_outside_training_pool=len(unseen_never))
        print(f"      {ds:<18} n={len(rows):>4}  seen={len(seen):>4} ({100.0*len(seen)/len(rows):5.1f}%)  "
              f"held-out={len(unseen_held):>4}  outside-pool={len(unseen_never):>4}")
    res["A_overlap"] = A

    # -------- (B) in-domain vs out-of-domain
    print("[3/3] in-domain vs out-of-domain selection gain ...", flush=True)
    B = {}
    pooled = {"seen": [], "unseen": []}
    for ds, rows in per_ds_rows.items():
        seen = [r for r in rows if (ds, r["idx"]) in TRAIN]
        unseen = [r for r in rows if (ds, r["idx"]) not in TRAIN]
        cell = {"full": summarize(rows, "full"),
                "seen": summarize(seen, "seen_in_training"),
                "unseen": summarize(unseen, "unseen")}
        if seen and unseen:
            ms, mu = cell_metrics(seen), cell_metrics(unseen)
            pt, lo, hi = boot_2sample_diff(ms["trained"], ms["greedy"], mu["trained"], mu["greedy"])
            cell["seen_minus_unseen_gain"] = {"delta": pt, "ci": [lo, hi],
                                              "significant": bool(lo > 0 or hi < 0)}
            infl = (cell["full"]["gain_over_greedy"] / cell["unseen"]["gain_over_greedy"]
                    if abs(cell["unseen"]["gain_over_greedy"]) > 1e-9 else None)
            cell["inflation_full_over_unseen"] = infl
        B[ds] = cell
        if ds in ("slake_open", "pathvqa_open", "vqa_rad_open"):
            pooled["seen"] += seen; pooled["unseen"] += unseen
        print(f"      {ds:<18} full gain={cell['full']['gain_over_greedy']:+.4f}  "
              f"seen={cell['seen']['gain_over_greedy'] if cell['seen'] else float('nan'):+.4f}  "
              f"unseen={cell['unseen']['gain_over_greedy'] if cell['unseen'] else float('nan'):+.4f}")

    # pooled over the three PAPER open cells only
    all3 = pooled["seen"] + pooled["unseen"]
    pool = {"full": summarize(all3, "full"),
            "seen": summarize(pooled["seen"], "seen_in_training"),
            "unseen": summarize(pooled["unseen"], "unseen")}
    ms, mu = cell_metrics(pooled["seen"]), cell_metrics(pooled["unseen"])
    pt, lo, hi = boot_2sample_diff(ms["trained"], ms["greedy"], mu["trained"], mu["greedy"])
    pool["seen_minus_unseen_gain"] = {"delta": pt, "ci": [lo, hi], "significant": bool(lo > 0 or hi < 0)}
    pool["memorization_share_of_gain"] = (1.0 - pool["unseen"]["gain_over_greedy"] /
                                          pool["full"]["gain_over_greedy"])
    # cell-n-weighted variant of the unseen gain (weights each cell's UNSEEN gain by its FULL n),
    # reported alongside the direct pooled mean because the two answer slightly different questions
    w = {ds: B[ds]["full"]["n"] for ds in ("slake_open", "pathvqa_open", "vqa_rad_open")}
    tw = sum(w.values())
    pool["nweighted_full_gain"] = sum(w[ds] * B[ds]["full"]["gain_over_greedy"] for ds in w) / tw
    pool["nweighted_unseen_gain"] = sum(w[ds] * B[ds]["unseen"]["gain_over_greedy"] for ds in w) / tw
    pool["nweighted_memorization_share"] = 1.0 - pool["nweighted_unseen_gain"] / pool["nweighted_full_gain"]
    B["POOLED_3_PAPER_OPEN_CELLS"] = pool
    res["B_in_vs_out_of_domain"] = B

    print("[3b] image-level leakage ...", flush=True)
    res["A2_image_level_overlap"] = image_overlap(TRAIN, TEST,
                                                  {k: v for k, v in per_ds_rows.items() if k != "radimagenet_open"})
    res["B2_dataset_level_transfer_evidence"] = lodo_evidence()
    for k, v in res["B2_dataset_level_transfer_evidence"].items():
        print(f"      {k}: n={v['n']} greedy={v['greedy']:.4f} trained={v['trained']:.4f} "
              f"gain={v['gain_over_greedy']:+.4f} conv={v['oracle_conversion']:.3f}")
    print(f"      POOLED(3 paper cells) full={pool['full']['gain_over_greedy']:+.4f}  "
          f"unseen={pool['unseen']['gain_over_greedy']:+.4f}  "
          f"memorization share={100*pool['memorization_share_of_gain']:.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev.update(res)
    json.dump(prev, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
