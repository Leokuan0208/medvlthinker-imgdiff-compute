"""Q1/Q2: count rows / unique keys in the Qwen train caches (metas only) and eval-half n per benchmark."""
import json, hashlib, sys, os
from collections import Counter, defaultdict
sys.path.insert(0, "/home/jamesyang/medvlthinker-imgdiff-compute/src/training_methods")
F = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden/"
CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b/"
import re, string


def norm_local(s):
    return s


try:
    from head_domain_scaling import norm
except Exception as e:
    print("could not import norm:", e); raise

TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


print("=== Qwen train caches")
seen_norm, seen_raw, seen_slot = set(), set(), set()
tot = 0
for c in ("kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"):
    m = json.load(open(F + f"generator_train_qwen_{c}.meta.json"))
    rows = m["rows"]
    ok = [r for r in rows if r.get("n_tok", -1) > 0 and r.get("ds") in TRAIN_DOMAINS]
    tot += len(ok)
    dsc = Counter(r["ds"] for r in ok)
    kn = {(r["ds"], r["idx"], norm(r["na"])) for r in ok}
    kr = {(r["ds"], r["idx"], r["ans"]) for r in ok}
    print(f"  {c:22} meta.n={m['n']:6} rows={len(rows):6} usable={len(ok):6} "
          f"uniq(ds,idx,norm)={len(kn):6} uniq(ds,idx,raw ans)={len(kr):6}  by ds={dict(dsc)}")
    print(f"     model={m['model']}  layers={m['layers']}  max_pixels={m['max_pixels']}")
    seen_norm |= kn; seen_raw |= kr
print(f"  TOTAL usable rows over 4 caches = {tot:,}; unique (ds,idx,norm(na)) = {len(seen_norm):,}; "
      f"unique (ds,idx,raw ans) = {len(seen_raw):,}; dup dropped = {tot-len(seen_norm):,}")
print("  unique by ds:", dict(Counter(k[0] for k in seen_norm)))
print("  unique questions by ds:", {d: len({k[1] for k in seen_norm if k[0] == d}) for d in TRAIN_DOMAINS})

for gen, stemf in (("qwen", lambda c: f"generator_eval_qwen_{c}"),
                   ("medgemma", lambda c: f"generator_eval_medgemma_{c}"),
                   ("lingshu", lambda c: ("generator_eval_finelayer" if c in ("slake_open", "vqa_rad_open")
                                          else f"generator_eval_finelayer_{c}"))):
    tag = {"qwen": "qwen25vl7b", "medgemma": "medgemma4b", "lingshu": "lingshu7b"}[gen]
    print(f"=== {gen} eval caches")
    ntr_total = 0
    for c in BENCH:
        m = json.load(open(F + stemf(c) + ".meta.json"))
        rows = [r for r in m["rows"] if r.get("n_tok", -1) > 0 and r.get("ds") == c]
        n_all_rows = len([r for r in m["rows"] if r.get("ds") == c])
        gok = {}
        for l in open(CK + f"ckpt_{c}_{tag}.judge.jsonl"):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = d["judge_ok"]
        q_all = {r["idx"] for r in rows}
        q_ho = {r["idx"] for r in rows if half(r["img_md5"]) == 0}
        q_tr = {r["idx"] for r in rows if half(r["img_md5"]) == 1}
        r_tr = sum(1 for r in rows if half(r["img_md5"]) == 1)
        r_ho = sum(1 for r in rows if half(r["img_md5"]) == 0)
        ntr_total += r_tr
        dupkeys = len(rows) - len({(r["idx"], norm(r["na"])) for r in rows})
        imgs_ho = len({r["img_md5"] for r in rows if half(r["img_md5"]) == 0})
        print(f"  {c:18} layers={m['layers']} rows={n_all_rows:6} usable={len(rows):6} dup(idx,norm)={dupkeys:5} "
              f"questions={len(q_all):6} (dump has {len(gok):6}) heldout_q={len(q_ho & set(gok)):5} "
              f"heldout_imgs={imgs_ho:5} train_q={len(q_tr):5} train_rows={r_tr:6} heldout_rows={r_ho:6} "
              f"max_idx={max(q_all)}")
    print(f"  total train-half rows = {ntr_total:,}")
