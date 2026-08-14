#!/usr/bin/env python3
"""coadapt_T04_build_pools.py -- turn the raw T=0.4 TRAIN pools into judge-labelled training data.

Pre-registration: results/cascade_methods/artifacts/_coadapt_verifier_prereg_2026-08-14.json

Three stages, each idempotent:

  explode   one judge row per unique (question, answer) pair, byte-identical logic to
            src/cascade_methods/explode_sc_for_judge.py (same dedup key, same `idx = orig#k` scheme).

  preload   the judge (src/labeling/run_judge.py, MedVLThinker-32B, temperature 0, text-only) is a
            deterministic function of (question, gold, candidate text).  A T=0.4 candidate string that
            the SAME train question already produced at T=0.7 therefore already has a label, and
            re-judging it would only burn GPU.  This stage copies those labels into the T=0.4
            .judge.jsonl so run_judge.py's own resume logic skips them.

            SOURCES ARE SAME-SOURCE ONLY.  A *_train label is never allowed to reach a test idx and
            vice versa: the train and test parquet index spaces collide numerically, so crossing them
            would silently attach the wrong label (the same reason
            decoding_sweep_judgepreload.py excludes *_train).

            HOLDOUT NULL TEST.  A seeded random sample of preloadable rows is deliberately NOT
            preloaded, so the judge relabels them from scratch in this round.  `validate` then compares
            those fresh labels against what the preload would have said.  Any disagreement means the
            preload is not sound HERE, not merely that it was sound in some earlier round.

  validate  runs that comparison and writes the report.

  python3 src/training_methods/coadapt_T04_build_pools.py --stage explode
  python3 src/training_methods/coadapt_T04_build_pools.py --stage preload
  python3 src/training_methods/coadapt_T04_build_pools.py --stage validate
"""
import argparse, json, os, random, sys
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)

HOT = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")          # the incumbent's T=0.7 pools
COLD = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b_T04")     # this round's T=0.4 pools
SOURCES = ["slake_open_train", "vqa_rad_open_train", "pathvqa_open_train",
           "kvasir_open", "radimagenet_open"]
HOLDOUT_N = 300        # per source, capped at the number of preloadable rows
HOLDOUT_SEED = 20260814
REPORT = os.path.join(COLD, "judgepreload_report_T04.json")


def norm(s):
    """The project's answer-identity normalizer (genframe_data.norm)."""
    return str(s).strip().lower()


def cold_sc8(ds):
    return os.path.join(COLD, f"ckpt_{ds}_lingshu7bT04_sc8.jsonl")


def cold_exp(ds):
    return os.path.join(COLD, f"ckpt_{ds}_lingshu7bT04_sc8_scexploded.jsonl")


def hot_stems(ds):
    """Every already-judged exploded dump of the SAME source. Same index space, same questions."""
    out = []
    for stem in (f"ckpt_{ds}_lingshu7b_sc8_scexploded", f"ckpt_{ds}_lingshu7b_sc16_scexploded"):
        p = os.path.join(HOT, stem + ".jsonl")
        j = os.path.join(HOT, stem + ".judge.jsonl")
        if os.path.exists(p) and os.path.exists(j):
            out.append(stem)
    return out


def hot_labels(ds):
    """{(idx, na) -> judge_ok} from the T=0.7 dumps of this source; conflicting keys are DROPPED."""
    cache, conflicted, nconf, used = {}, set(), 0, []
    for stem in hot_stems(ds):
        jud = {}
        for l in open(os.path.join(HOT, stem + ".judge.jsonl")):
            if l.strip():
                r = json.loads(l); jud[r["idx"]] = int(r["judge_ok"])
        n = 0
        for l in open(os.path.join(HOT, stem + ".jsonl")):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["idx"] not in jud:
                continue
            base = str(r["idx"]).rsplit("#", 1)[0]
            idx = int(base) if base.lstrip("-").isdigit() else base
            k = (idx, norm(r["modal_pred"]))
            v = jud[r["idx"]]
            if k in cache and cache[k] != v:
                conflicted.add(k); nconf += 1
                continue
            cache[k] = v; n += 1
        used.append({"stem": stem, "rows_used": n})
    for k in conflicted:
        cache.pop(k, None)
    return cache, used, nconf, len(conflicted)


# ------------------------------------------------------------------ explode
def do_explode():
    rep = {}
    for ds in SOURCES:
        src = cold_sc8(ds)
        if not os.path.exists(src):
            print(f"  !! {ds}: no T=0.4 pool yet ({src})", flush=True); continue
        out = cold_exp(ds)
        n_in = n_out = 0
        with open(out, "w") as fh:
            for l in open(src):
                if not l.strip():
                    continue
                r = json.loads(l); n_in += 1
                seen = set()
                for k, ans in enumerate(r["preds"]):
                    key = norm(ans)
                    if key in seen:
                        continue
                    seen.add(key)
                    fh.write(json.dumps({"idx": f'{r["idx"]}#{k}', "question": r["question"],
                                         "gold": r["gold"], "modal_pred": ans}) + "\n")
                    n_out += 1
        rep[ds] = {"questions": n_in, "unique_judge_rows": n_out,
                   "mean_distinct_of_8": round(n_out / max(n_in, 1), 4)}
        print(f"  {ds:20s} {n_in:6d} questions -> {n_out:6d} unique (idx,answer) rows "
              f"({n_out/max(n_in,1):.3f} distinct/8)", flush=True)
    return rep


# ------------------------------------------------------------------ preload
def do_preload():
    rep = {}
    rng = random.Random(HOLDOUT_SEED)
    for ds in SOURCES:
        exp = cold_exp(ds)
        if not os.path.exists(exp):
            print(f"  !! {ds}: not exploded yet"); continue
        cache, used, nconf, ndrop = hot_labels(ds)
        rows = [json.loads(l) for l in open(exp) if l.strip()]
        hits = [r for r in rows if (int(str(r["idx"]).rsplit("#", 1)[0])
                                    if str(r["idx"]).rsplit("#", 1)[0].lstrip("-").isdigit()
                                    else str(r["idx"]).rsplit("#", 1)[0],
                                    norm(r["modal_pred"])) in cache]
        hold = set(x["idx"] for x in rng.sample(hits, min(HOLDOUT_N, len(hits))))
        jf = exp.replace(".jsonl", ".judge.jsonl")
        already = set()
        if os.path.exists(jf):
            for l in open(jf):
                if l.strip():
                    already.add(json.loads(l)["idx"])
        n_written = 0
        with open(jf, "a") as fh:
            for r in rows:
                if r["idx"] in hold or r["idx"] in already:
                    continue
                base = str(r["idx"]).rsplit("#", 1)[0]
                idx = int(base) if base.lstrip("-").isdigit() else base
                v = cache.get((idx, norm(r["modal_pred"])))
                if v is None:
                    continue
                fh.write(json.dumps({"idx": r["idx"], "judge_ok": int(v)}) + "\n")
                n_written += 1
        rep[ds] = {"hot_sources_used": used, "hot_keys": len(cache),
                   "hot_internal_conflict_rows": nconf, "hot_conflicted_keys_dropped": ndrop,
                   "cold_judge_rows": len(rows), "preloadable": len(hits),
                   "preload_hit_rate": round(len(hits) / max(len(rows), 1), 4),
                   "holdout_left_for_the_judge": len(hold),
                   "preloaded_written_this_call": n_written,
                   "still_to_judge": len(rows) - len(already) - n_written}
        print(f"  {ds:20s} rows={len(rows):6d} preloadable={len(hits):6d} "
              f"({100*len(hits)/max(len(rows),1):.1f}%) holdout={len(hold)} "
              f"-> still to judge {rep[ds]['still_to_judge']}", flush=True)
        json.dump(sorted(hold), open(exp.replace(".jsonl", ".holdout.json"), "w"))
    return rep


# ------------------------------------------------------------------ validate
def do_validate():
    rep, agree, disagree, absent = {}, 0, 0, 0
    for ds in SOURCES:
        exp = cold_exp(ds)
        hp = exp.replace(".jsonl", ".holdout.json")
        jf = exp.replace(".jsonl", ".judge.jsonl")
        if not (os.path.exists(hp) and os.path.exists(jf)):
            continue
        hold = set(json.load(open(hp)))
        cache, _, _, _ = hot_labels(ds)
        fresh = {}
        for l in open(jf):
            if l.strip():
                r = json.loads(l)
                if r["idx"] in hold:
                    fresh[r["idx"]] = int(r["judge_ok"])
        a = d = m = 0
        for l in open(exp):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["idx"] not in hold:
                continue
            if r["idx"] not in fresh:
                m += 1; continue
            base = str(r["idx"]).rsplit("#", 1)[0]
            idx = int(base) if base.lstrip("-").isdigit() else base
            v = cache.get((idx, norm(r["modal_pred"])))
            if v is None:
                m += 1
            elif int(v) == fresh[r["idx"]]:
                a += 1
            else:
                d += 1
        rep[ds] = {"holdout_n": len(hold), "agree": a, "disagree": d, "unresolved": m,
                   "agreement_rate": round(a / max(a + d, 1), 6)}
        agree += a; disagree += d; absent += m
        print(f"  {ds:20s} holdout={len(hold):4d} agree={a:4d} disagree={d:3d} unresolved={m:3d}",
              flush=True)
    rep["TOTAL"] = {"agree": agree, "disagree": disagree, "unresolved": absent,
                    "agreement_rate": round(agree / max(agree + disagree, 1), 6),
                    "pass": bool(disagree == 0)}
    print(f"\nPRELOAD NULL TEST: {agree}/{agree+disagree} agree "
          f"({rep['TOTAL']['agreement_rate']:.6f}), pass={rep['TOTAL']['pass']}", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["explode", "preload", "validate"])
    A = ap.parse_args()
    old = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    old[A.stage] = {"explode": do_explode, "preload": do_preload, "validate": do_validate}[A.stage]()
    json.dump(old, open(REPORT, "w"), indent=1, default=str)
    print(f"wrote {REPORT} [{A.stage}]")
